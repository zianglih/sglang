"""Native EP4 integration smoke; exercises real AUTO, pooling and graph replay.

Launch with torchrun, pinned SGLang source on PYTHONPATH, and FlashInfer with
the released FlashInfer 0.7.4 getter/workspace APIs. Run first with --reuse-cache 0,
then --reuse-cache 1 using the same --cache directory and precision/activation.
No candidate or kernel is replaced. The AUTO wrapper only counts invocations.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import types
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reuse-cache", type=int, choices=(0, 1), required=True)
    parser.add_argument("--precision", choices=("w4a4", "w4a16", "mxfp8"), required=True)
    parser.add_argument("--activation", choices=("silu", "situ"), default="silu")
    args = parser.parse_args()
    if args.precision == "mxfp8" and args.activation != "silu":
        parser.error("the SGLang MXFP8 MegaMoE adapter supports only silu")
    os.environ.update(
        SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=(
            "1" if args.precision == "w4a16" else "0"
        ),
        SGLANG_FLASHINFER_MEGAMOE_IN_KERNEL_FC2_REDUCE="0",
        SGLANG_FLASHINFER_MEGAMOE_COMBINE_DTYPE="bf16",
        SGLANG_FLASHINFER_MEGAMOE_MAX_TOKENS_PER_RANK="4",
        SGLANG_FLASHINFER_MEGAMOE_DECODE_MAX_TOKENS_PER_RANK="2",
        FLASHINFER_MOE_EP_KNOB_CACHE="0",
    )
    import flashinfer

    assert flashinfer.__version__ == "0.7.4", flashinfer.__version__
    assert os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK", "0") == "0"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    torch.manual_seed(100 + rank)

    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.moe_ep import MoEEpMegaLayer, MoEEpTensors
    from flashinfer.moe_ep.core.kernel.workspace_pool import pooled_workspace_refcount
    from sglang.srt.layers import dp_attention
    from sglang.srt.layers.moe import flashinfer_megamoe as adapter
    from sglang.srt.layers.moe import flashinfer_megamoe_autotune as autotune_adapter
    from sglang.srt.layers.moe.flashinfer_megamoe_autotune import (
        _CACHE_NAMESPACE,
        megamoe_autotune_context,
    )
    from sglang.srt.runtime_context import get_parallel

    source_root = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]).resolve()
    module_sources = {}
    for name, module in (("adapter", adapter), ("autotune", autotune_adapter)):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(source_root), (name, path, source_root)
        module_sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert _CACHE_NAMESPACE == "sglang_flashinfer_megamoe_knobs", _CACHE_NAMESPACE
    assert hasattr(MoEEpMegaLayer, "get_tuned_knobs"), "FlashInfer getter is required"
    assert "knobs" in inspect.signature(MoEEpMegaLayer.create_workspace).parameters

    # Synthetic EP metadata uses the current runtime-context ownership API.
    # The real distributed accessor reads this same group during layer construction.
    get_parallel().override_permanently(
        moe_ep_group=types.SimpleNamespace(
            device_group=dist.group.WORLD, world_size=world, rank_in_group=rank
        )
    )
    counts = [2] * world
    dp_attention.get_dp_global_num_tokens = lambda: counts
    if args.precision == "w4a16":
        from flashinfer.moe_ep.cute_dsl.megamoe import bf16_nvfp4 as native

        auto_name = "autotune_bf16_nvfp4_mega_moe"
    elif args.precision == "w4a4":
        from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as native

        auto_name = "autotune_nvfp4_mega_moe"
    else:
        from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as native

        auto_name = "autotune_mxfp8_mega_moe"
    auto_calls = 0
    original_auto = getattr(native, auto_name)

    def count_auto(*a, **kw):
        nonlocal auto_calls
        auto_calls += 1
        return original_auto(*a, **kw)

    setattr(native, auto_name, count_auto)
    hidden, intermediate, experts, topk = 256, 256, 8, 4
    local_experts = experts // world
    layers = []
    for index in range(2):
        layer = torch.nn.Module()
        layer.layer_id = index
        layer.hidden_size = hidden
        layer.intermediate_size_per_partition = intermediate
        layer.num_experts = experts
        layer.num_local_experts = local_experts
        layer.moe_ep_size, layer.moe_ep_rank = world, rank
        layer.top_k = topk
        layer.quant_config = types.SimpleNamespace(use_per_token_activation=False)
        layer.moe_runner_config = types.SimpleNamespace(
            is_gated=True,
            activation=args.activation,
            gemm1_alpha=4.0,
            gemm1_clamp_limit=25.0,
            swiglu_limit=1.5,
            apply_router_weight_on_input=False,
            routed_scaling_factor=1.0,
        )
        if args.precision != "mxfp8":
            layer.g1_alphas = torch.full((local_experts,), (1 + index) / 16, device="cuda")
            layer.g1_alphas_up = layer.g1_alphas
            layer.g2_alphas = torch.full((local_experts,), (1 + index) / 32, device="cuda")
        if args.precision == "w4a4":
            layer.w13_input_scale_quant = torch.ones((), device="cuda")
            layer.w2_input_scale_quant = torch.full(
                (local_experts,), 1.0 + index, device="cuda"
            )
        for prefix, rows, columns in (
            ("w13", 2 * intermediate, hidden),
            ("w2", hidden, intermediate),
        ):
            if args.precision == "mxfp8":
                # Canonical SGLang MXFP8: unpacked E4M3 weights and one
                # E8M0 scale byte per 32 K values. 127 encodes scale 1.0.
                weight = (
                    torch.randn(local_experts, rows, columns, device="cuda")
                    .mul_(0.125 * (1 + index))
                    .to(torch.float8_e4m3fn)
                )
                scale = torch.full(
                    (local_experts, rows, columns // 32),
                    127,
                    dtype=torch.uint8,
                    device="cuda",
                )
                layer.register_parameter(
                    f"{prefix}_weight",
                    torch.nn.Parameter(weight, requires_grad=False),
                )
                layer.register_parameter(
                    f"{prefix}_weight_scale_inv",
                    torch.nn.Parameter(scale, requires_grad=False),
                )
                continue
            layer.register_parameter(
                f"{prefix}_weight",
                torch.nn.Parameter(
                    torch.randint(
                        256,
                        (local_experts, rows, columns // 2),
                        dtype=torch.uint8,
                        device="cuda",
                    ),
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                f"{prefix}_weight_scale",
                torch.nn.Parameter(
                    torch.ones(local_experts, rows, columns // 16, device="cuda").to(
                        torch.float8_e4m3fn
                    ),
                    requires_grad=False,
                ),
            )
        if args.precision == "mxfp8":
            adapter.prepare_mxfp8_moe_weights_for_flashinfer_megamoe(layer)
            adapter.ensure_mxfp8_moe_layer_for_flashinfer_megamoe(layer)
            # The layer's scale parameter is the transformed, swizzled plane
            # that is bound into both the live session and reference sessions.
            transformed_scale = (
                layer._flashinfer_megamoe_forward.backend.transformed_weights[0][1]
            )
            assert transformed_scale.data_ptr() == layer.w13_weight_scale_inv.data_ptr()
        else:
            adapter.prepare_nvfp4_moe_weights_for_flashinfer_megamoe(layer)
            adapter.ensure_nvfp4_moe_layer_for_flashinfer_megamoe(layer)
        state = layer._flashinfer_megamoe_forward
        assert state._decode_max_tokens_per_rank == (0 if args.precision == "mxfp8" else 2)
        assert state._decode_workspace is None
        layers.append(layer)

    x = torch.randn(16, hidden, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:topk] for _ in range(16)]
    ).int()
    scores = torch.rand(16, topk, dtype=torch.float32, device="cuda")
    scores /= scores.sum(dim=1, keepdim=True)

    def tensors(layer):
        n = counts[rank]
        return MoEEpTensors(
            x[:n],
            ids[:n],
            scores[:n],
            fc1_alpha=layer.g1_alphas if args.precision != "mxfp8" else None,
            fc2_alpha=layer.g2_alphas if args.precision != "mxfp8" else None,
            fc1_norm_const=(
                layer.w2_input_scale_quant if args.precision == "w4a4" else None
            ),
        )

    def forward(layer):
        t = tensors(layer)
        dispatch = types.SimpleNamespace(
            hidden_states=t.hidden_states,
            topk_output=types.SimpleNamespace(
                topk_ids=t.topk_ids, topk_weights=t.topk_weights
            ),
        )
        info = adapter.FlashInferMegaMoeQuantInfo(
            mega=layer._flashinfer_megamoe_layer,
            mega_forward=layer._flashinfer_megamoe_forward,
            fc1_alpha=t.fc1_alpha,
            fc2_alpha=t.fc2_alpha,
            fc1_norm_const=t.fc1_norm_const,
        )
        return adapter.run_flashinfer_megamoe(
            dispatch, info, layer.moe_runner_config
        ).hidden_states

    def sync():
        torch.cuda.synchronize()
        dist.barrier()

    cache_path = args.cache / args.precision / args.activation / f"rank{rank}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tuner = AutoTuner.get()
    if args.reuse_cache:
        assert cache_path.is_file(), cache_path
        tuner.load_configs(str(cache_path))
    else:
        cache_path.unlink(missing_ok=True)
    # Both layers belong to one startup pass. The context's direct knob
    # records deduplicate tuning; the next process tests persisted reuse.
    with (
        autotune(cache=None),
        megamoe_autotune_context(decode_num_tokens=2, prefill_num_tokens=8),
    ):
        for layer in layers:
            forward(layer)
    tuner.save_configs(str(cache_path))
    records = tuner.get_namespaced_records(_CACHE_NAMESPACE)
    assert len(records) == 3
    assert all(isinstance(knobs, dict) and knobs for knobs in records.values())
    assert all("entries" not in knobs for knobs in records.values())
    # Persist only rank-consistent knobs, without native cache envelopes.
    gathered = [None] * world
    dist.all_gather_object(gathered, records)
    assert all(item == records for item in gathered)
    assert cache_path.is_file()
    sync()
    capacities = [1, 2, 8]
    expected_calls = 0 if args.reuse_cache else len(capacities)
    assert auto_calls == expected_calls, (auto_calls, expected_calls)
    for layer in layers:
        state = layer._flashinfer_megamoe_forward
        assert sorted(state.workspaces) == capacities
        # Startup tuning supersedes the optional untuned decode-only profile.
        assert state._decode_workspace is None
    for capacity in capacities:
        left = (
            layers[0]
            ._flashinfer_megamoe_forward.workspaces[capacity]
            ._backend_workspace
        )
        right = (
            layers[1]
            ._flashinfer_megamoe_forward.workspaces[capacity]
            ._backend_workspace
        )
        assert left is right
        assert pooled_workspace_refcount(left) == 2

    # Independent fixed-default sessions do not reuse the tuned pool entries.
    controls = []
    for layer in layers:
        state = layer._flashinfer_megamoe_forward
        control = MoEEpMegaLayer(
            bootstrap=state.bootstrap,
            fleet_params=dataclasses.replace(
                state.fleet_params, max_tokens_per_rank=16
            ),
            weights=None,
            backend=dataclasses.replace(
                state.backend,
                megakernel=dataclasses.replace(state.backend.megakernel, knobs={}),
            ),
        )
        control.warmup(tensors(layer))
        controls.append(control)

    checked = []
    shapes = [[0, 1, 1, 2], [0, 1, 2, 7], [8, 8, 8, 8]]
    for shape in shapes:
        counts[:] = shape
        eager = [forward(layer).clone() for layer in layers]
        expected = [
            control.forward(tensors(layer)).clone()
            for layer, control in zip(layers, controls, strict=True)
        ]
        sync()
        for actual, ref in zip(eager, expected, strict=True):
            assert torch.isfinite(actual).all()
            assert torch.equal(actual.view(torch.int16), ref.view(torch.int16))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = [forward(layer).clone() for layer in layers]
        sync()
        for _ in range(2):
            graph.replay()
            sync()
        for actual, ref in zip(captured, eager, strict=True):
            assert torch.equal(actual.view(torch.int16), ref.view(torch.int16))
        # The API returns borrowed buffers. Keep snapshots independent of
        # subsequent fixed-reference forwards and require the scale edit to
        # affect every nonempty layer output.
        before_scale_mutation = [value.clone() for value in expected]
        for layer in layers:
            if args.precision == "mxfp8":
                # MXFP8 has no per-expert alpha. Halve the prepared FC1
                # E8M0 block scales in place, keeping padding bytes at zero
                # and preserving the weight storage captured by the graph.
                live_scale = (
                    layer._flashinfer_megamoe_forward.backend.transformed_weights[0][1]
                )
                assert live_scale.data_ptr() == layer.w13_weight_scale_inv.data_ptr()
                scale_bytes = live_scale.view(torch.uint8)
                scale_bytes.sub_((scale_bytes > 0).to(torch.uint8))
            else:
                layer.g1_alphas.mul_(0.75)
        graph.replay()
        sync()
        expected = [
            control.forward(tensors(layer)).clone()
            for layer, control in zip(layers, controls, strict=True)
        ]
        sync()
        for before, actual, ref in zip(
            before_scale_mutation, captured, expected, strict=True
        ):
            assert torch.isfinite(actual).all()
            assert torch.isfinite(ref).all()
            assert torch.equal(actual.view(torch.int16), ref.view(torch.int16))
            if ref.numel():
                assert not torch.equal(
                    before.view(torch.int16), ref.view(torch.int16)
                ), "FC1 scale mutation did not change the reference output"
        graph.reset()
        checked.append(shape)

    assert auto_calls == expected_calls
    sync()
    survivors = list(layers[1]._flashinfer_megamoe_forward.workspaces.values())
    pooled = [handle._backend_workspace for handle in survivors]
    layers[0]._flashinfer_megamoe_layer.destroy()
    for handle in survivors:
        assert pooled_workspace_refcount(handle._backend_workspace) == 1
    layers[1]._flashinfer_megamoe_layer.destroy()
    assert all(pooled_workspace_refcount(workspace) == 0 for workspace in pooled)
    for control in controls:
        control.destroy()
    print(
        "MEGAMOE_AUTOTUNE_SMOKE="
        + json.dumps(
            dict(
                rank=rank,
                passed=True,
                activation=args.activation,
                precision=args.precision,
                flashinfer_version=flashinfer.__version__,
                source_commit=os.environ["SGLANG_VALIDATION_COMMIT"],
                module_sources=module_sources,
                cache_namespace=_CACHE_NAMESPACE,
                configured_decode_workspace_capacity=(
                    0 if args.precision == "mxfp8" else 2
                ),
                decode_workspace_created=False,
                auto_calls=auto_calls,
                reuse_cache=bool(args.reuse_cache),
                capacities=capacities,
                bootstrap_capacity=4,
                decode_num_tokens=2,
                prefill_num_tokens=8,
                startup_contexts=1,
                layers_per_context=2,
                checked_counts=checked,
                graph_replays=3 * len(shapes),
                bit_exact=True,
                borrowed_outputs_cloned=True,
                fc1_scale_mutation_observed=True,
                pool_refcounts=[2, 1, 0],
            )
        ),
        flush=True,
    )
    get_parallel().clear_stamp()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
