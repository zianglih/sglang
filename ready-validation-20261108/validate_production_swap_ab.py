#!/usr/bin/env python3
"""Focused real-SGLang loader/forward/graph/reload validation on SM100/SM103.

Run with the effort's SGLang python/ on PYTHONPATH and installed FlashInfer v0.7.4rc1.
This is a local validation artifact, not an upstream test or benchmark claim.
Both arms use the same FlashInfer source; the control forces swap_ab=False.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.quant_ref_utils import quantize_nvfp4_shard


MODES = ("w4a4_scalar", "w4a4_per_token", "w4a16")
DEFAULT_CASES = [
    (mode, activation) for mode in MODES for activation in ("silu", "relu2")
] + [("w4a16", "situ")]
PATCH_TARGET = (
    "sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl.use_cutedsl_moe_swap_ab"
)
KERNEL_ENV_NAMES = (
    "FLASHINFER_NVFP4_4OVER6",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH",
    "TRTLLM_ENABLE_PDL",
    "FLASHINFER_DIST_AWARE_AUTOTUNE",
    "FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY",
    "FLASHINFER_CUDA_ARCH_LIST",
    "CUTE_DSL_CACHE_DIR",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION",
    "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16",
    "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE",
    "SGLANG_MOE_NVFP4_DISPATCH",
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def get_output(layer, x, routes):
    value = layer.forward(x, routes)
    if isinstance(value, torch.Tensor):
        return value
    return value[0] if isinstance(value, tuple) else value.hidden_states


def statistics_between(actual, expected):
    a, b = actual.float(), expected.float()
    diff = (a - b).abs()
    return {
        "cosine": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item(),
        "max_abs_error": diff.max().item(),
        "reference_abs_max": b.abs().max().item(),
        "relative_l2_error": (
            torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)
        ).item(),
        "bitwise_equal": bool(torch.equal(actual, expected)),
    }


def assert_numerics(actual, expected, *, mode, label):
    metrics = statistics_between(actual, expected)
    # The legacy kernel and the swap-AB kernel have different quantization /
    # accumulation instruction paths. Record full errors, not an exactness claim.
    cosine = 0.9995 if mode == "w4a16" else 0.995
    fraction = 0.03 if mode == "w4a16" else 0.10
    assert metrics["cosine"] > cosine, (label, metrics)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=fraction,
        atol=fraction * metrics["reference_abs_max"],
        msg=label,
    )
    return metrics


def make_checkpoint(args, activation, seed):
    torch.manual_seed(seed)
    is_gated = activation != "relu2"
    values, refs = [], {}
    weight_std = 0.25 if activation == "situ" else 0.05
    for expert in range(args.experts):
        weights = {
            "w1": torch.randn(
                args.intermediate, args.hidden, dtype=torch.bfloat16, device="cuda"
            )
            * weight_std,
            "w2": torch.randn(
                args.hidden, args.intermediate, dtype=torch.bfloat16, device="cuda"
            )
            * weight_std,
        }
        if is_gated:
            weights["w3"] = (
                torch.randn(
                    args.intermediate,
                    args.hidden,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                * weight_std
            )
            gate_up_amax = torch.maximum(
                weights["w1"].abs().max(), weights["w3"].abs().max()
            )
            gate_up_scale = 2688.0 / gate_up_amax.float()
        for shard, weight in weights.items():
            scale = gate_up_scale if is_gated and shard != "w2" else None
            packed, block_scale, scale, dequantized = quantize_nvfp4_shard(
                weight, scale
            )
            refs.setdefault(shard, []).append(dequantized)
            prefix = "w2" if shard == "w2" else "w13"
            # Synthetic calibrated activation range 8; W4A16 neutralizes it,
            # per-token mode supplies its own dynamic activation scales.
            act_scale = torch.tensor(8.0 / 2688.0, device="cuda")
            for suffix, tensor in (
                ("weight", packed),
                ("weight_scale", block_scale),
                ("weight_scale_2", (1.0 / scale).clone()),
                ("input_scale", act_scale),
            ):
                values.append((expert, shard, f"{prefix}_{suffix}", tensor))
    return values, {name: torch.stack(rows) for name, rows in refs.items()}


def load_checkpoint(layer, values):
    for expert, shard, name, tensor in values:
        layer.weight_loader(
            getattr(layer, name), tensor, name, shard_id=shard, expert_id=expert
        )
    layer.quant_method.process_weights_after_loading(layer)


def make_layer(args, mode, activation):
    return FusedMoE(
        num_experts=args.experts,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        layer_id=0,
        top_k=args.topk,
        params_dtype=torch.bfloat16,
        quant_config=ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=mode == "w4a4_per_token",
        ),
        activation=activation,
        is_gated=activation != "relu2",
        gemm1_alpha=4.0 if activation == "situ" else None,
        gemm1_clamp_limit=25.0 if activation == "situ" else None,
        gate_up_interleaved=False,
    ).cuda()


def reference_w4a16(x, routes, refs, activation):
    """Independent FP32 matmuls on dequantized weights, BF16 FC1 activation."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(refs["w1"].shape[0]):
        token, slot = torch.where(routes.topk_ids == expert)
        xf = x[token].float()
        gate = xf @ refs["w1"][expert].T
        if activation == "relu2":
            act = torch.relu(gate).square()
        else:
            up = xf @ refs["w3"][expert].T
            if activation == "situ":
                act = (
                    4.0
                    * torch.tanh(gate / 4.0)
                    * torch.sigmoid(gate)
                    * (25.0 * torch.tanh(up / 25.0))
                )
            else:
                act = torch.nn.functional.silu(gate) * up
        result = act.bfloat16().float() @ refs["w2"][expert].T
        out.index_add_(0, token, result * routes.topk_weights[token, slot, None])
    return out


def pointers(layer):
    names = ("w13_weight", "w2_weight", "w13_blockscale_mma", "w2_blockscale_mma")
    data = {name: getattr(layer, name).data_ptr() for name in names}
    data.update(
        {
            f"resolved_scale_{i}": value.data_ptr()
            for i, value in enumerate(layer._cutedsl_scales)
        }
    )
    data["resolved_input_scale"] = layer._cutedsl_input_scale.data_ptr()
    return data


@contextlib.contextmanager
def layer_context(mode, swap_ab):
    with (
        envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(mode == "w4a16"),
        envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.override(False),
        envs.SGLANG_MOE_NVFP4_DISPATCH.override(False),
        get_context().override_server_args(model_path="dummy"),
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend("flashinfer_cutedsl"),
            a2a_backend=MoeA2ABackend("none"),
        ),
        get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ),
        patch(PATCH_TARGET, return_value=swap_ab),
    ):
        yield


def measure_graph(graph, warmup, iterations):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0)
    return samples


def run_arm(args, mode, activation, tokens, swap_ab, checkpoints, x, routes):
    record = dict(mode=mode, activation=activation, tokens=tokens, swap_ab=swap_ab)
    start = time.perf_counter()
    with (
        layer_context(mode, swap_ab),
        torch.no_grad(),
        contextlib.ExitStack() as tracing,
    ):
        layer = make_layer(args, mode, activation)
        load_checkpoint(layer, checkpoints[0][0])
        selected_trace = []
        if args.autotune:
            from flashinfer import autotune
            from flashinfer.autotuner import AutoTuner

            tuner = AutoTuner.get()
            tuner.clear_cache()
            tuner.reset_statistics()
            cache_dir = args.tactic_cache_dir or args.output / "tactics"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = (
                cache_dir / f"{mode}-{activation}-m{tokens}-swap{int(swap_ab)}.json"
            )
            reuse_tactics = cache_path.is_file()
            with autotune(
                not reuse_tactics, cache=str(cache_path), tuning_buckets=(tokens,)
            ):
                get_output(layer, x, routes)
            cache_bytes = cache_path.read_bytes()
            record["autotune"] = {
                "cache": str(cache_path),
                "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
                "loaded_existing_cache": reuse_tactics,
                "tuning_buckets": [tokens],
                "selection": json.loads(cache_bytes),
                "statistics_after_selection": str(tuner.stats),
            }
            before_misses = tuner.stats.cache_misses
            original_choose_one = tuner.choose_one

            def trace_choose_one(*call_args, **call_kwargs):
                selected = original_choose_one(*call_args, **call_kwargs)
                if call_args[0].startswith("CuteDslMoEWrapper"):
                    selected_trace.append(
                        {
                            "op": call_args[0],
                            "runner": type(selected[0]).__name__,
                            "tactic": json.loads(json.dumps(selected[1], default=str)),
                            "tuning_mode": tuner.is_tuning_mode,
                        }
                    )
                return selected

            tracing.enter_context(patch.object(tuner, "choose_one", trace_choose_one))
            print("TACTICS " + json.dumps(record["autotune"]), flush=True)
        eager = get_output(layer, x, routes).clone()
        assert layer._cutedsl_wrapper.swap_ab is swap_ab
        # Follow PyTorch's documented side-stream warmup before graph capture.
        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                get_output(layer, x, routes)
        torch.cuda.current_stream().wait_stream(warm_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = get_output(layer, x, routes)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, eager, rtol=0, atol=0)
        record["graph_matches_eager_exactly"] = True
        if mode == "w4a16":
            ref = reference_w4a16(x, routes, checkpoints[0][1], activation)
            record["reference_initial"] = assert_numerics(
                eager, ref, mode=mode, label="initial W4A16 dequantized reference"
            )
        initial_output = eager.detach().cpu()
        before = pointers(layer)
        load_checkpoint(layer, checkpoints[1][0])
        after = pointers(layer)
        assert before == after, {"before": before, "after": after}
        record["reload_pointers_unchanged"] = before
        reloaded = get_output(layer, x, routes).clone()
        assert not torch.equal(reloaded, eager), "Reload did not change the output"
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, reloaded, rtol=0, atol=0)
        record["graph_reload_matches_eager_exactly"] = True
        if mode == "w4a16":
            ref = reference_w4a16(x, routes, checkpoints[1][1], activation)
            record["reference_reload"] = assert_numerics(
                reloaded, ref, mode=mode, label="reloaded W4A16 dequantized reference"
            )
        samples = measure_graph(graph, args.warmup, args.iters) if args.timing else []
        record["latency_us_samples"] = samples
        if samples:
            record["median_us"] = statistics.median(samples)
        if args.autotune:
            assert selected_trace and all(
                not item["tuning_mode"] for item in selected_trace
            )
            assert all(item == selected_trace[0] for item in selected_trace)
            assert tuner.stats.cache_misses == before_misses
            record["selected_tactics_outside_tuning"] = selected_trace
            record["cache_misses_after_tuning"] = (
                tuner.stats.cache_misses - before_misses
            )
            assert (
                hashlib.sha256(cache_path.read_bytes()).hexdigest()
                == record["autotune"]["cache_sha256"]
            )
        record["elapsed_seconds"] = time.perf_counter() - start
        return record, initial_output, reloaded.detach().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", default="32")
    parser.add_argument("--cases", default="", help="Comma-separated mode:activation")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--tactic-cache-dir", type=Path)
    parser.add_argument(
        "--arm-order", choices=("legacy-first", "swap-first"), default="legacy-first"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--port", type=int, default=29673)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    init_single_process_dist(master_port=args.port)
    torch.set_default_device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    import flashinfer
    import sglang
    from flashinfer.quantization.nvfp4_quantization_utils import (
        current_nvfp4_4over6_config,
    )

    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "flashinfer": flashinfer.__version__,
            "flashinfer_file": flashinfer.__file__,
            "sglang_file": sglang.__file__,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "inherited_kernel_environment": {
                name: os.environ.get(name) for name in KERNEL_ENV_NAMES
            },
            "nvfp4_4over6_config": repr(current_nvfp4_4over6_config()),
        },
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "method": (
            "Real SGLang FusedMoE TP1/EP1; serialized NVFP4 through production loaders. "
            "Identical checkpoints, BF16 inputs and precomputed routes in both arms. "
            "Both arms use installed FlashInfer v0.7.4rc1; legacy forces swap_ab=False. "
            "Deterministic unfused finalize; no autotuning. W4A16 has an independent "
            "dequantized-weight FP32 matmul/BF16-activation reference; W4A4 checks "
            "cross-layout parity, graph equality and reload. Optional timings use "
            "CUDA events around one whole-forward graph replay per sample, warm L2, "
            "initial quantization included, reload/weight prep/capture excluded; "
            "fixed defaults are mode-specific, not a fixed-tactic or CUPTI study."
        ),
        "arms": [],
        "comparisons": [],
    }
    if args.autotune:
        results["method"] = (
            results["method"]
            .replace(
                "no autotuning.",
                "each mode/layout tunes only the requested token bucket before measurement, "
                "then retains the selected cache for graph and reload checks.",
            )
            .replace(
                "fixed defaults are mode-specific, not a fixed-tactic or CUPTI study.",
                "tactics are independently selected per mode/layout and frozen across repeats; "
                "this is not an isolated common-tactic kernel or CUPTI study.",
            )
        )
    write_json(args.output / "results.json", results)
    cases = (
        [tuple(s.split(":")) for s in args.cases.split(",")]
        if args.cases
        else DEFAULT_CASES
    )
    for mode, activation in cases:
        checkpoints = [make_checkpoint(args, activation, seed) for seed in (1103, 1104)]
        for tokens in map(int, args.tokens.split(",")):
            torch.manual_seed(1105)
            x = torch.randn(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
            x *= 2.0 if activation == "situ" else 0.2
            with layer_context(mode, False):
                routes = select_experts(
                    hidden_states=x,
                    router_logits=torch.randn(
                        tokens, args.experts, dtype=torch.float32, device="cuda"
                    ),
                    topk_config=TopKConfig(
                        top_k=args.topk, renormalize=True, torch_native=True
                    ),
                )
            arms = []
            arm_order = (
                (False, True) if args.arm_order == "legacy-first" else (True, False)
            )
            for swap_ab in arm_order:
                print(
                    f"START mode={mode} activation={activation} tokens={tokens} swap_ab={swap_ab}",
                    flush=True,
                )
                arm = run_arm(
                    args, mode, activation, tokens, swap_ab, checkpoints, x, routes
                )
                results["arms"].append(arm[0])
                write_json(args.output / "results.json", results)
                arms.append(arm)
                print(
                    "PASS "
                    + json.dumps(
                        {
                            k: v
                            for k, v in arm[0].items()
                            if k
                            not in ("latency_us_samples", "reload_pointers_unchanged")
                        }
                    ),
                    flush=True,
                )
            arms.sort(key=lambda arm: arm[0]["swap_ab"])
            comparison = dict(mode=mode, activation=activation, tokens=tokens)
            for index, label in ((1, "initial"), (2, "reloaded")):
                comparison[label] = assert_numerics(
                    arms[1][index],
                    arms[0][index],
                    mode=mode,
                    label=f"layout parity {label}",
                )
            if args.timing:
                comparison["legacy_median_us"] = arms[0][0]["median_us"]
                comparison["swap_ab_median_us"] = arms[1][0]["median_us"]
                comparison["legacy_over_swap_ab"] = (
                    comparison["legacy_median_us"] / comparison["swap_ab_median_us"]
                )
            results["comparisons"].append(comparison)
            write_json(args.output / "results.json", results)
            print("PAIR_PASS " + json.dumps(comparison), flush=True)
    with (args.output / "raw-timing.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["mode", "activation", "tokens", "swap_ab", "sample", "microseconds"]
        )
        for arm in results["arms"]:
            for i, sample in enumerate(arm["latency_us_samples"]):
                writer.writerow(
                    [
                        arm["mode"],
                        arm["activation"],
                        arm["tokens"],
                        arm["swap_ab"],
                        i,
                        sample,
                    ]
                )
    results["complete"] = True
    write_json(args.output / "results.json", results)
    print(
        f"COMPLETE {len(results['arms'])} arms / {len(results['comparisons'])} matched cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
