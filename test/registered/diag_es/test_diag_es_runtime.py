from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.diag_es.manager import DiagESManager, compose_diag_es_extra_key
from sglang.srt.diag_es.manifest import (
    DenseSite,
    DiagESManifest,
    Qwen3DiagESManifest,
    compute_effective_model_digest,
    register_qwen3_30b_a3b_dense_sites,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class _Linear:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.es_site_id = None
        self.es_site_width = None
        self.es_pre_site_id = None
        self.es_pre_site_width = None
        self.es_post_site_id = None
        self.es_post_site_width = None


def _make_model():
    layers = []
    for _ in range(48):
        runner_config = SimpleNamespace(es_layer_id=None)
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    qkv_proj=_Linear(2048, 5120),
                    o_proj=_Linear(4096, 2048),
                ),
                mlp=SimpleNamespace(
                    gate=_Linear(2048, 128),
                    experts=SimpleNamespace(moe_runner_config=runner_config),
                ),
            )
        )
    return SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=2048,
            num_experts=128,
            moe_intermediate_size=768,
        ),
        model=SimpleNamespace(layers=layers),
        lm_head=_Linear(2048, 151936),
    )


@pytest.mark.parametrize(
    ("placement", "expected_dense", "expected_grouped", "expected_digest"),
    [
        (
            "pre",
            [("qkv_proj.input", 2048), ("o_proj.input", 4096)],
            {
                "moe_fc1_pre": (48, 128, 2048),
                "moe_fc2_pre": (48, 128, 768),
            },
            "3650c468725df70692ae0470aa3769eb97ec91f9a04fe4de1f419c36a017b956",
        ),
        (
            "post",
            [("qkv_proj.output", 5120), ("o_proj.output", 2048)],
            {
                "moe_fc1_post": (48, 128, 1536),
                "moe_fc2_post": (48, 128, 2048),
            },
            "7502d6adee8003103476e4faf4bf9f3bef2ed19bf4a1a8bfd8e5e011da5b3342",
        ),
        (
            "both",
            [
                ("qkv_proj.input", 2048),
                ("qkv_proj.output", 5120),
                ("o_proj.input", 4096),
                ("o_proj.output", 2048),
            ],
            {
                "moe_fc1_pre": (48, 128, 2048),
                "moe_fc2_pre": (48, 128, 768),
                "moe_fc1_post": (48, 128, 1536),
                "moe_fc2_post": (48, 128, 2048),
            },
            "d63a0480f277675fce4af8646bed81cad5c004002b44e8150a22b8121e6d7e60",
        ),
    ],
)
def test_qwen_manifest_is_semantic_and_excludes_router(
    placement, expected_dense, expected_grouped, expected_digest
):
    model = _make_model()
    manifest = register_qwen3_30b_a3b_dense_sites(model, placement=placement)

    assert manifest.placement == placement
    assert len(manifest.dense_sites) == 48 * len(expected_dense)
    assert [
        (site.site_id.removeprefix("model.layers.0.self_attn."), site.width)
        for site in manifest.dense_sites[: len(expected_dense)]
    ] == expected_dense
    assert manifest.grouped_gate_shapes == expected_grouped
    assert manifest.schema_digest == expected_digest
    assert model.model.layers[47].mlp.experts.moe_runner_config.es_layer_id == 47
    assert model.model.layers[0].mlp.gate.es_site_id is None
    assert model.model.layers[0].mlp.gate.es_post_site_id is None
    assert model.lm_head.es_pre_site_id is None
    assert model.lm_head.es_post_site_id is None


@pytest.mark.parametrize("placement", ["post", "both"])
def test_qwen_post_manifest_rejects_tp_greater_than_one(placement):
    with pytest.raises(ValueError, match="tp_size=1"):
        register_qwen3_30b_a3b_dense_sites(
            _make_model(), placement=placement, tp_size=2
        )


def test_effective_digest_hashes_exact_fp32_delta_payload():
    dense = {"site-a": torch.tensor([1e-4, -1e-4], dtype=torch.float32)}
    fc1 = torch.zeros((1, 1, 2), dtype=torch.float32)
    fc2 = torch.zeros((1, 1, 1), dtype=torch.float32)

    digest = compute_effective_model_digest(
        model_artifact_id="Qwen/Qwen3-30B-A3B",
        schema_id="qwen3-30b-a3b-diag-es-v2",
        schema_digest="schema",
        dense_deltas=dense,
        grouped_deltas={"moe_fc1_pre": fc1, "moe_fc2_pre": fc2},
    )
    assert digest != compute_effective_model_digest(
        model_artifact_id="Qwen/Qwen3-30B-A3B",
        schema_id="qwen3-30b-a3b-diag-es-v2",
        schema_digest="schema",
        dense_deltas={"site-a": torch.zeros(2, dtype=torch.float32)},
        grouped_deltas={"moe_fc1_pre": fc1, "moe_fc2_pre": fc2},
    )

    dense["site-a"][0] = 2e-4
    assert digest != compute_effective_model_digest(
        model_artifact_id="Qwen/Qwen3-30B-A3B",
        schema_id="qwen3-30b-a3b-diag-es-v2",
        schema_digest="schema",
        dense_deltas=dense,
        grouped_deltas={"moe_fc1_pre": fc1, "moe_fc2_pre": fc2},
    )


def test_cache_namespace_uses_effective_digest_and_existing_key():
    digest_a = "00" * 32
    digest_b = "01" * 32
    assert compose_diag_es_extra_key("tenant-a", digest_a) == (
        compose_diag_es_extra_key("tenant-a", digest_a)
    )
    assert compose_diag_es_extra_key("tenant-a", digest_a) != (
        compose_diag_es_extra_key("tenant-b", digest_a)
    )
    assert compose_diag_es_extra_key("tenant-a", digest_a) != (
        compose_diag_es_extra_key("tenant-a", digest_b)
    )


def test_effective_digest_is_the_radix_cache_identity():
    """Operational candidate IDs do not enter semantic KV-cache identity."""
    prompt_tokens = array("q", [11, 12, 13, 14])
    candidate_a = {
        "candidate_id": "candidate-a",
        "effective_model_digest": "00" * 32,
    }
    candidate_b_same_model = {
        "candidate_id": "candidate-b",
        "effective_model_digest": "00" * 32,
    }
    candidate_c_different_model = {
        "candidate_id": "candidate-c",
        "effective_model_digest": "01" * 32,
    }

    def radix_key(candidate):
        return RadixKey(
            token_ids=array("q", prompt_tokens),
            extra_key=compose_diag_es_extra_key(
                "tenant-a", candidate["effective_model_digest"]
            ),
        )

    cache = RadixCache.create_simulated()
    cache.insert(InsertParams(key=radix_key(candidate_a)))

    same_model_match = cache.match_prefix(
        MatchPrefixParams(key=radix_key(candidate_b_same_model))
    )
    different_model_match = cache.match_prefix(
        MatchPrefixParams(key=radix_key(candidate_c_different_model))
    )

    assert same_model_match.device_indices.numel() == len(prompt_tokens)
    assert different_model_match.device_indices.numel() == 0


@pytest.mark.parametrize(
    ("rows", "width", "expected"),
    [
        (8, 2048, (256, None)),
        (256, 4096, (256, None)),
        (512, 2048, (256, None)),
        (512, 4096, (512, 4)),
        (1024, 2048, (512, 4)),
        (1024, 4096, (2048, 4)),
        (2048, 2048, (2048, 4)),
        (2048, 4096, (2048, 8)),
        (4096, 2048, (2048, 8)),
        (4096, 4096, (4096, 8)),
        (4096, 8960, (256, None)),
    ],
)
def test_dense_delta_launch_config(rows, width, expected):
    from sglang.srt.diag_es.ops import _dense_delta_launch_config

    assert _dense_delta_launch_config(rows, width) == expected


@pytest.mark.parametrize("width", [1536, 2048, 4096, 8960])
def test_triton_dense_delta_mixed_slots_and_identity(width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang.srt.diag_es.ops import apply_dense_delta

    torch.manual_seed(20260811 + width)
    rows = 11
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    delta_bank = torch.zeros((3, width), dtype=torch.float32, device="cuda")
    delta_bank[1:].uniform_(-0.5, 0.5)
    slots = torch.tensor(
        [0, 1, 2, 0, 2, 1, 1, 0, 2, 0, 1],
        dtype=torch.int32,
        device="cuda",
    )

    out = apply_dense_delta(x, delta_bank, slots)
    x_fp32 = x.float()
    delta = delta_bank[slots.long()]
    expected = torch.addcmul(x_fp32, x_fp32, delta).to(torch.bfloat16)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    identity_rows = slots == 0
    assert torch.equal(out[identity_rows], x[identity_rows])


@pytest.mark.parametrize(
    ("rows", "width"),
    [(512, 2048), (1024, 4096), (2048, 2048), (4096, 4096)],
)
def test_triton_dense_delta_tuned_launch_and_caller_owned_output(rows, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang.srt.diag_es.ops import apply_dense_delta_out

    torch.manual_seed(20260819 + rows + width)
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    delta_bank = torch.empty((9, width), dtype=torch.float32, device="cuda")
    delta_bank.uniform_(-0.02, 0.02)
    delta_bank[0].zero_()
    slots = (torch.arange(rows, dtype=torch.int32, device="cuda") % 8) + 1
    output = torch.empty_like(x)

    returned = apply_dense_delta_out(x, delta_bank, slots, output)
    expected = torch.addcmul(x.float(), x.float(), delta_bank[slots.long()]).bfloat16()

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_triton_fp32_delta_preserves_signal_lost_by_bf16_multiplier():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang.srt.diag_es.ops import apply_dense_delta

    # BF16(1 + 0.003) is exactly one, so the historical multiplier payload
    # erases this perturbation before it reaches the activation.  Applying the
    # zero-centered FP32 delta to the FP32 activation changes these values
    # before the single final BF16 rounding step.
    x = torch.tensor(
        [[1.9921875, 1.984375, 1.5, 127.5]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    delta_bank = torch.full((1, x.shape[1]), 0.003, dtype=torch.float32, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")

    actual = apply_dense_delta(x, delta_bank, slots)
    expected = torch.addcmul(x.float(), x.float(), delta_bank).to(torch.bfloat16)
    legacy = x * (1.0 + delta_bank).to(torch.bfloat16)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal((1.0 + delta_bank).to(torch.bfloat16), torch.ones_like(x))
    assert not torch.equal(actual, legacy)


@pytest.mark.parametrize(("rows", "width"), [(17, 2048), (1024, 4096), (4096, 2048)])
def test_triton_fp32_delta_cuda_graph_replay_observes_live_bank_and_slots(rows, width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang.srt.diag_es.ops import apply_dense_delta

    torch.manual_seed(20260819 + rows + width)
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    delta_bank = torch.zeros((3, width), dtype=torch.float32, device="cuda")
    delta_bank[1:].uniform_(-0.01, 0.01)
    slots = torch.arange(rows, dtype=torch.int32, device="cuda") % 3

    # Compile before capture. The captured graph must retain only stable
    # pointers; candidate slots and resident delta rows remain mutable data.
    apply_dense_delta(x, delta_bank, slots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = apply_dense_delta(x, delta_bank, slots)

    graph.replay()
    torch.cuda.synchronize()
    expected = torch.addcmul(x.float(), x.float(), delta_bank[slots.long()]).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    x.copy_(torch.randn_like(x))
    slots.copy_((torch.arange(rows, device="cuda", dtype=torch.int32) + 1) % 3)
    delta_bank[1:].uniform_(-0.02, 0.02)
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.addcmul(x.float(), x.float(), delta_bank[slots.long()]).bfloat16()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("width", "rows"), [(1536, 9), (8960, 9), (2048, 1024), (4096, 1024)]
)
def test_unquantized_linear_apply_and_apply_into_use_dense_delta_hook(
    monkeypatch, width, rows
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang.srt.diag_es import manager as manager_module
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    torch.manual_seed(20260812 + width)
    site_id = "model.layers.0.test_proj.input"
    manifest = DiagESManifest(
        schema_id="test-dense-diag-es-v1",
        dense_sites=(DenseSite(site_id, width),),
        grouped_gate_shapes={},
        schema_digest="ab" * 32,
    )
    manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=2,
        model_artifact_id="test-artifact",
        device=torch.device("cuda"),
    )
    deltas = {
        "candidate-a": torch.rand(width, dtype=torch.float32) - 0.5,
        "candidate-b": torch.rand(width, dtype=torch.float32) - 0.5,
    }
    slots = [0]
    for candidate_id, delta in deltas.items():
        registered = manager.register_candidate(
            candidate_id=candidate_id,
            dense_deltas={site_id: delta},
            grouped_deltas={},
        )
        slots.append(registered["resident_slot"])
    monkeypatch.setattr(manager_module, "_manager", manager)

    output_width = 64
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((output_width, width), dtype=torch.bfloat16, device="cuda")
    layer = SimpleNamespace(
        weight=weight,
        es_site_id=site_id,
        es_site_width=width,
        es_pre_site_id=site_id,
        es_pre_site_width=width,
        es_post_site_id=None,
        es_post_site_width=None,
    )
    candidate_slot_pattern = torch.tensor(
        [
            slots[0],
            slots[1],
            slots[2],
            slots[0],
            slots[2],
            slots[1],
            slots[1],
            slots[0],
            slots[2],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    candidate_slots = candidate_slot_pattern.repeat(
        (rows + candidate_slot_pattern.numel() - 1) // candidate_slot_pattern.numel()
    )[:rows].contiguous()
    delta_bank = manager.get_dense_delta_bank(site_id)
    x_fp32 = x.float()
    delta = delta_bank[candidate_slots.long()]
    steered_oracle = torch.addcmul(x_fp32, x_fp32, delta).to(torch.bfloat16)
    expected = torch.nn.functional.linear(steered_oracle, weight)

    method = UnquantizedLinearMethod()
    output = torch.empty((rows, output_width), dtype=torch.bfloat16, device="cuda")
    graph_output = torch.empty_like(output)
    graph = None
    with forward_context(
        ForwardContext(attn_backend=None, es_candidate_slots=candidate_slots)
    ):
        applied = method.apply(layer, x)
        applied_into = method.apply_into(layer, x, output)
        if width in (2048, 4096):
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                method.apply_into(layer, x, graph_output)

    assert applied_into.data_ptr() == output.data_ptr()
    torch.testing.assert_close(applied, expected, rtol=0, atol=0)
    torch.testing.assert_close(applied_into, expected, rtol=0, atol=0)
    if graph is not None:
        x.copy_(torch.randn_like(x))
        candidate_slots.copy_(candidate_slots.roll(1))
        delta_bank[slots[1]].uniform_(-0.02, 0.02)
        graph.replay()
        torch.cuda.synchronize()
        steered = torch.addcmul(
            x.float(),
            x.float(),
            delta_bank[candidate_slots.long()],
        ).bfloat16()
        expected_replay = torch.nn.functional.linear(steered, weight)
        torch.testing.assert_close(graph_output, expected_replay, rtol=0, atol=0)


def test_unquantized_linear_both_placement_threads_post_accumulator_inputs(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang.srt.diag_es import manager as manager_module
    from sglang.srt.layers.quantization import unquant as unquant_module
    from sglang.srt.layers.quantization.unquant import (
        Bf16GemmBackend,
        UnquantizedLinearMethod,
    )
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    torch.manual_seed(20260820)
    rows, input_width, output_width = 17, 256, 128
    pre_site = "model.layers.0.test_proj.input"
    post_site = "model.layers.0.test_proj.output"
    manifest = DiagESManifest(
        schema_id="test-dense-diag-es-v2",
        dense_sites=(
            DenseSite(pre_site, input_width, "pre"),
            DenseSite(post_site, output_width, "post"),
        ),
        grouped_gate_shapes={},
        schema_digest="cd" * 32,
        placement="both",
    )
    manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=2,
        model_artifact_id="test-artifact",
        device=torch.device("cuda"),
    )
    resident_slots = []
    for candidate in range(2):
        registered = manager.register_candidate(
            candidate_id=f"candidate-{candidate}",
            dense_deltas={
                pre_site: torch.empty(input_width, dtype=torch.float32).uniform_(
                    -0.02, 0.02
                ),
                post_site: torch.empty(output_width, dtype=torch.float32).uniform_(
                    -0.03, 0.03
                ),
            },
            grouped_deltas={},
        )
        resident_slots.append(registered["resident_slot"])
    monkeypatch.setattr(manager_module, "_manager", manager)
    monkeypatch.setattr(
        unquant_module,
        "_BF16_GEMM_BACKEND",
        Bf16GemmBackend.TRITON,
    )

    x = torch.randn((rows, input_width), dtype=torch.bfloat16, device="cuda")
    weight = (
        torch.randn((output_width, input_width), dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    bias = torch.randn(output_width, dtype=torch.bfloat16, device="cuda") * 0.1
    layer = SimpleNamespace(
        weight=weight,
        es_pre_site_id=pre_site,
        es_pre_site_width=input_width,
        es_post_site_id=post_site,
        es_post_site_width=output_width,
    )
    candidate_slots = torch.tensor(
        [0, *resident_slots] * 6,
        dtype=torch.int32,
        device="cuda",
    )[:rows].contiguous()
    pre_bank = manager.get_dense_delta_bank(pre_site)
    post_bank = manager.get_dense_delta_bank(post_site)
    pre = torch.addcmul(
        x.float(), x.float(), pre_bank[candidate_slots.long()]
    ).bfloat16()
    affine = pre.float() @ weight.float().T + bias.float()
    expected = torch.addcmul(
        affine,
        affine,
        post_bank[candidate_slots.long()],
    ).bfloat16()

    method = UnquantizedLinearMethod()
    caller_output = torch.empty_like(expected)
    with forward_context(
        ForwardContext(attn_backend=None, es_candidate_slots=candidate_slots)
    ):
        allocated = method.apply(layer, x, bias)
        returned = method.apply_into(layer, x, caller_output, bias)

    assert returned.data_ptr() == caller_output.data_ptr()
    torch.testing.assert_close(allocated, expected, rtol=0.02, atol=0.5)
    torch.testing.assert_close(caller_output, expected, rtol=0.02, atol=0.5)
    assert torch.equal(allocated, caller_output)

    # The server captures the caller-owned path.  Capture after eager warmup,
    # then mutate every live input while preserving the captured pointers.
    graph_output = torch.empty_like(expected)
    graph = torch.cuda.CUDAGraph()
    with forward_context(
        ForwardContext(attn_backend=None, es_candidate_slots=candidate_slots)
    ):
        with torch.cuda.graph(graph):
            method.apply_into(layer, x, graph_output, bias)

    x.copy_(torch.randn_like(x))
    candidate_slots.copy_(
        torch.tensor(
            [resident_slots[1], 0, resident_slots[0]] * 6,
            dtype=torch.int32,
            device="cuda",
        )[:rows]
    )
    pre_bank[resident_slots[0]].uniform_(-0.04, 0.04)
    post_bank[resident_slots[1]].uniform_(-0.05, 0.05)
    graph.replay()
    torch.cuda.synchronize()

    replay_pre = torch.addcmul(
        x.float(), x.float(), pre_bank[candidate_slots.long()]
    ).bfloat16()
    replay_affine = replay_pre.float() @ weight.float().T + bias.float()
    replay_expected = torch.addcmul(
        replay_affine,
        replay_affine,
        post_bank[candidate_slots.long()],
    ).bfloat16()
    torch.testing.assert_close(graph_output, replay_expected, rtol=0.02, atol=0.5)


def test_resident_manager_retire_is_nonblocking_and_backpressures():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    manifest = Qwen3DiagESManifest(
        dense_sites=(DenseSite("dense", 4),),
        num_layers=1,
        num_experts=2,
        hidden_size=4,
        moe_intermediate_size=3,
        schema_digest="ab" * 32,
    )
    manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=1,
        base_model_revision="test-model",
        device=torch.device("cuda"),
    )
    payload = {
        "dense_deltas": {"dense": torch.zeros(4, dtype=torch.float32)},
        "grouped_deltas": {
            "moe_fc1_pre": torch.zeros((1, 2, 4), dtype=torch.float32),
            "moe_fc2_pre": torch.zeros((1, 2, 3), dtype=torch.float32),
        },
    }

    registered = manager.register_candidate(
        candidate_id="candidate-a",
        effective_model_digest="00" * 32,
        **payload,
    )
    assert registered["effective_model_digest"] != "00" * 32
    slot = registered["resident_slot"]
    manager.acquire("candidate-a")
    assert manager.retire_candidate("candidate-a")["state"] == "RETIRING"
    with pytest.raises(AssertionError):
        manager.acquire("candidate-a")
    with pytest.raises(RuntimeError, match="capacity is exhausted"):
        manager.register_candidate(candidate_id="candidate-b", **payload)

    # Existing requests may submit their last read after retirement. The slot
    # is reclaimed only after their ref drops and that read fence completes.
    torch.cuda._sleep(1_000_000)
    manager.note_slots_read([slot])
    manager.release("candidate-a")
    torch.cuda.synchronize()
    status = manager.status()
    assert "candidate-a" not in status["candidates"]
    assert status["free_slots"] == [slot]
    assert (
        manager.register_candidate(candidate_id="candidate-b", **payload)[
            "resident_slot"
        ]
        == slot
    )
