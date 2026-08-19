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
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.es_site_id = None
        self.es_site_width = None


def _make_model():
    layers = []
    for _ in range(48):
        runner_config = SimpleNamespace(es_layer_id=None)
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    qkv_proj=_Linear(2048),
                    o_proj=_Linear(4096),
                ),
                mlp=SimpleNamespace(
                    gate=_Linear(2048),
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
    )


def test_qwen_manifest_is_semantic_and_excludes_router():
    model = _make_model()
    manifest = register_qwen3_30b_a3b_dense_sites(model)

    assert len(manifest.dense_sites) == 96
    assert manifest.dense_sites[0].site_id == (
        "model.layers.0.self_attn.qkv_proj.input"
    )
    assert manifest.dense_sites[0].input_width == 2048
    assert manifest.dense_sites[1].site_id == "model.layers.0.self_attn.o_proj.input"
    assert manifest.dense_sites[1].input_width == 4096
    assert model.model.layers[47].mlp.experts.moe_runner_config.es_layer_id == 47
    assert model.model.layers[0].mlp.gate.es_site_id is None


def test_effective_digest_hashes_exact_fp32_delta_payload():
    dense = {"site-a": torch.tensor([1e-4, -1e-4], dtype=torch.float32)}
    fc1 = torch.zeros((1, 1, 2), dtype=torch.float32)
    fc2 = torch.zeros((1, 1, 1), dtype=torch.float32)

    digest = compute_effective_model_digest(
        base_model_revision="Qwen/Qwen3-30B-A3B",
        schema_digest="schema",
        dense_deltas=dense,
        expert_fc1_deltas=fc1,
        expert_fc2_deltas=fc2,
    )
    assert digest != compute_effective_model_digest(
        base_model_revision="Qwen/Qwen3-30B-A3B",
        schema_digest="schema",
        dense_deltas={"site-a": torch.zeros(2, dtype=torch.float32)},
        expert_fc1_deltas=fc1,
        expert_fc2_deltas=fc2,
    )

    dense["site-a"][0] = 2e-4
    assert digest != compute_effective_model_digest(
        base_model_revision="Qwen/Qwen3-30B-A3B",
        schema_digest="schema",
        dense_deltas=dense,
        expert_fc1_deltas=fc1,
        expert_fc2_deltas=fc2,
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
        "expert_fc1_deltas": torch.zeros((1, 2, 4), dtype=torch.float32),
        "expert_fc2_deltas": torch.zeros((1, 2, 3), dtype=torch.float32),
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
