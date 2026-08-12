from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.diag_es.manager import DiagESManager, compose_diag_es_extra_key
from sglang.srt.diag_es.manifest import (
    DenseSite,
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


def test_effective_digest_hashes_bf16_payload_not_operational_identity():
    dense = {"site-a": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    fc1 = torch.ones((1, 1, 2), dtype=torch.bfloat16)
    fc2 = torch.ones((1, 1, 1), dtype=torch.bfloat16)

    digest = compute_effective_model_digest(
        base_model_revision="Qwen/Qwen3-30B-A3B",
        schema_digest="schema",
        dense_gates=dense,
        expert_fc1_gates=fc1,
        expert_fc2_gates=fc2,
    )
    assert digest == compute_effective_model_digest(
        base_model_revision="Qwen/Qwen3-30B-A3B",
        schema_digest="schema",
        dense_gates={"site-a": dense["site-a"].to(torch.bfloat16)},
        expert_fc1_gates=fc1,
        expert_fc2_gates=fc2,
    )

    dense["site-a"][0] = 3
    assert digest != compute_effective_model_digest(
        base_model_revision="Qwen/Qwen3-30B-A3B",
        schema_digest="schema",
        dense_gates=dense,
        expert_fc1_gates=fc1,
        expert_fc2_gates=fc2,
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


@pytest.mark.parametrize("width", [2048, 4096])
def test_triton_dense_gate_mixed_slots_and_identity(width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from sglang.srt.diag_es.ops import apply_dense_gate

    torch.manual_seed(20260811 + width)
    rows = 11
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    gate_bank = torch.ones((3, width), dtype=torch.bfloat16, device="cuda")
    gate_bank[1:].copy_(
        (0.5 + torch.rand((2, width), dtype=torch.float32, device="cuda")).to(
            torch.bfloat16
        )
    )
    slots = torch.tensor(
        [0, 1, 2, 0, 2, 1, 1, 0, 2, 0, 1],
        dtype=torch.int32,
        device="cuda",
    )

    out = apply_dense_gate(x, gate_bank, slots)
    expected = (x.float() * gate_bank[slots.long()].float()).to(torch.bfloat16)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    identity_rows = slots == 0
    assert torch.equal(out[identity_rows], x[identity_rows])


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
        "dense_gates": {"dense": torch.ones(4, dtype=torch.bfloat16)},
        "expert_fc1_gates": torch.ones((1, 2, 4), dtype=torch.bfloat16),
        "expert_fc2_gates": torch.ones((1, 2, 3), dtype=torch.bfloat16),
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
