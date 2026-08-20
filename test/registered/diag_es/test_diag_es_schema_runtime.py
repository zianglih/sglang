from types import SimpleNamespace

import pytest
import torch

from sglang.srt.diag_es.manager import (
    DiagESCandidateNotFoundError,
    DiagESCandidateRetiringError,
    DiagESInvalidCandidateError,
    DiagESManager,
)
from sglang.srt.diag_es.manifest import (
    QWEN3_30B_A3B_SCHEMA_ID,
    DenseSite,
    DiagESManifest,
    compute_effective_model_digest,
    register_qwen3_30b_a3b_manifest,
)
from sglang.srt.diag_es.protocol import (
    parse_register_payload,
    prepare_register_payload,
    validate_registry_request,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Linear:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        dtype=torch.bfloat16,
        block_fp8=False,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.empty((output_size, input_size), dtype=dtype, device="meta")
        self.es_pre_site_id = None
        self.es_post_site_id = None
        if block_fp8:
            self.weight_scale_inv = torch.empty(
                (output_size // 128, input_size // 128),
                dtype=torch.float32,
                device="meta",
            )
            self.input_scale = None


class _Experts:
    def __init__(self, layer_id: int, *, dtype=torch.bfloat16, block_fp8=False):
        self.num_experts = 128
        self.num_local_experts = 128
        self.hidden_size = 2048
        self.intermediate_size_per_partition = 768
        self.w13_weight = torch.empty((128, 1536, 2048), dtype=dtype, device="meta")
        self.w2_weight = torch.empty((128, 2048, 768), dtype=dtype, device="meta")
        if block_fp8:
            self.w13_weight_scale_inv = torch.empty(
                (128, 12, 16), dtype=torch.float32, device="meta"
            )
            self.w2_weight_scale_inv = torch.empty(
                (128, 16, 6), dtype=torch.float32, device="meta"
            )
            self.w13_input_scale = None
            self.w2_input_scale = None
        self.moe_runner_config = SimpleNamespace(
            num_experts=128,
            num_local_experts=128,
            hidden_size=2048,
            intermediate_size_per_partition=768,
            layer_id=layer_id,
            top_k=8,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16 if block_fp8 else dtype,
        )


class Fp8Config:
    is_checkpoint_fp8_serialized = True
    activation_scheme = "dynamic"
    weight_block_size = [128, 128]
    use_mxfp8 = False
    is_fp4_experts = False


class Qwen3MoeForCausalLM:
    def __init__(
        self,
        *,
        num_layers=48,
        hidden_size=2048,
        dtype=torch.bfloat16,
        block_fp8=False,
    ):
        self.quant_config = Fp8Config() if block_fp8 else None
        if block_fp8:
            dtype = torch.float8_e4m3fn
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            num_experts=128,
            moe_intermediate_size=768,
            num_experts_per_tok=8,
        )
        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        qkv_proj=_Linear(
                            2048, 5120, dtype=dtype, block_fp8=block_fp8
                        ),
                        o_proj=_Linear(
                            4096, 2048, dtype=dtype, block_fp8=block_fp8
                        ),
                    ),
                    mlp=SimpleNamespace(
                        gate=_Linear(2048, 128, dtype=dtype),
                        experts=_Experts(
                            layer_id, dtype=dtype, block_fp8=block_fp8
                        ),
                    ),
                )
                for layer_id in range(num_layers)
            ]
        )


@pytest.mark.parametrize(
    ("placement", "first_sites", "grouped_shapes", "schema_digest"),
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
def test_qwen3_v2_manifest_exact_contract(
    placement, first_sites, grouped_shapes, schema_digest
):
    model = Qwen3MoeForCausalLM()
    manifest = register_qwen3_30b_a3b_manifest(model, placement=placement)

    assert manifest.schema_id == QWEN3_30B_A3B_SCHEMA_ID
    assert manifest.placement == placement
    assert manifest.schema_digest == schema_digest
    assert manifest.grouped_delta_shapes == grouped_shapes
    assert [
        (site.site_id.removeprefix("model.layers.0.self_attn."), site.width)
        for site in manifest.dense_sites[: len(first_sites)]
    ] == first_sites
    assert model.model.layers[0].mlp.gate.es_pre_site_id is None
    assert model.model.layers[0].mlp.gate.es_post_site_id is None
    assert not hasattr(model.model.layers[0].self_attn.qkv_proj, "es_pre_site_width")


def test_qwen3_v2_manifest_accepts_only_post_deepseek_block_fp8():
    model = Qwen3MoeForCausalLM(block_fp8=True)
    manifest = register_qwen3_30b_a3b_manifest(model, placement="post")
    assert manifest.schema_digest == (
        "7502d6adee8003103476e4faf4bf9f3bef2ed19bf4a1a8bfd8e5e011da5b3342"
    )
    with pytest.raises(ValueError, match="supports post placement only"):
        register_qwen3_30b_a3b_manifest(model, placement="pre")


def test_qwen3_v2_manifest_rejects_mxfp8_and_non_fp32_block_scales():
    model = Qwen3MoeForCausalLM(block_fp8=True)
    model.quant_config.use_mxfp8 = True
    with pytest.raises(ValueError, match="use_mxfp8=True"):
        register_qwen3_30b_a3b_manifest(model, placement="post")

    model = Qwen3MoeForCausalLM(block_fp8=True)
    model.model.layers[0].self_attn.qkv_proj.weight_scale_inv = torch.empty(
        (40, 16), dtype=torch.uint8, device="meta"
    )
    with pytest.raises(ValueError, match="must have dtype torch.float32"):
        register_qwen3_30b_a3b_manifest(model, placement="post")


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (Qwen3MoeForCausalLM(num_layers=47), "num_hidden_layers"),
        (Qwen3MoeForCausalLM(hidden_size=1024), "hidden_size"),
        (Qwen3MoeForCausalLM(dtype=torch.float16), "torch.bfloat16"),
        (SimpleNamespace(), "Qwen3MoeForCausalLM"),
    ],
)
def test_qwen3_v2_manifest_rejects_unsupported_models(model, match):
    with pytest.raises((TypeError, ValueError), match=match):
        register_qwen3_30b_a3b_manifest(model, placement="pre")


def test_qwen3_v2_manifest_rejects_noncontiguous_expert_weights():
    model = Qwen3MoeForCausalLM()
    model.model.layers[0].mlp.experts.w13_weight = torch.empty(
        (128, 2048, 1536), dtype=torch.bfloat16, device="meta"
    ).transpose(-1, -2)
    with pytest.raises(ValueError, match="w13_weight must be contiguous"):
        register_qwen3_30b_a3b_manifest(model, placement="pre")


def test_qwen3_v2_manifest_rejects_noncontiguous_dense_weights():
    model = Qwen3MoeForCausalLM()
    model.model.layers[0].self_attn.qkv_proj.weight = torch.empty(
        (2048, 5120), dtype=torch.bfloat16, device="meta"
    ).transpose(-1, -2)
    with pytest.raises(ValueError, match="qkv_proj.weight must be contiguous"):
        register_qwen3_30b_a3b_manifest(model, placement="pre")


def _manifest() -> DiagESManifest:
    return DiagESManifest(
        schema_id=QWEN3_30B_A3B_SCHEMA_ID,
        placement="pre",
        dense_sites=(DenseSite("dense", 3),),
        grouped_delta_shapes={"moe_fc1_pre": (2, 3)},
        schema_digest="ab" * 32,
    )


def _payload():
    return (
        {"dense": torch.arange(3, dtype=torch.float32)},
        {"moe_fc1_pre": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
    )


def _cpu_manager() -> DiagESManager:
    manager = DiagESManager.__new__(DiagESManager)
    manager.manifest = _manifest()
    manager.model_artifact_id = "qwen3-artifact"
    manager.device = torch.device("cpu")
    manager.physical_slots = 3
    manager._records = {}
    manager._free_slots = [1, 2]
    manager._slot_last_read_events = [None, None, None]
    manager._dense_delta_banks = {"dense": torch.zeros((3, 3))}
    manager._grouped_delta_banks = {"moe_fc1_pre": torch.zeros((2, 3, 3))}
    return manager


def _digest(dense, grouped):
    return compute_effective_model_digest(
        model_artifact_id="qwen3-artifact",
        schema_id=QWEN3_30B_A3B_SCHEMA_ID,
        schema_digest="ab" * 32,
        dense_deltas=dense,
        grouped_deltas=grouped,
    )


def test_manager_status_preserves_external_grouped_gate_shapes_key():
    status = _cpu_manager().status()
    assert status["grouped_gate_shapes"] == {"moe_fc1_pre": [2, 3]}
    assert "grouped_delta_shapes" not in status


def test_manager_registration_is_transactional_and_rejects_digest_mismatch(
    monkeypatch,
):
    manager = _cpu_manager()
    dense, grouped = _payload()
    attempts = 0

    def fail_once(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("upload failed")

    monkeypatch.setattr(manager, "_upload_candidate", fail_once)
    with pytest.raises(RuntimeError, match="upload failed"):
        manager.register_candidate(
            candidate_id="candidate",
            dense_deltas=dense,
            grouped_deltas=grouped,
            effective_model_digest=_digest(dense, grouped),
        )
    assert manager.status()["free_slots"] == [1, 2]
    assert manager.status()["candidates"] == {}

    with pytest.raises(ValueError, match="does not match"):
        manager.register_candidate(
            candidate_id="candidate",
            dense_deltas=dense,
            grouped_deltas=grouped,
            effective_model_digest="00" * 32,
        )
    registered = manager.register_candidate(
        candidate_id="candidate",
        dense_deltas=dense,
        grouped_deltas=grouped,
        effective_model_digest=_digest(dense, grouped),
    )
    assert registered["state"] == "READY"
    assert registered["resident_slot"] == 1


def test_manager_candidate_errors_are_request_local_and_typed(monkeypatch):
    manager = _cpu_manager()
    monkeypatch.setattr(manager, "_upload_candidate", lambda **_kwargs: None)
    with pytest.raises(DiagESCandidateNotFoundError):
        manager.acquire("missing")

    dense, grouped = _payload()
    manager.register_candidate(
        candidate_id="candidate",
        dense_deltas=dense,
        grouped_deltas=grouped,
        effective_model_digest=_digest(dense, grouped),
    )
    manager.acquire("candidate")
    assert manager.retire_candidate("candidate")["state"] == "RETIRING"
    with pytest.raises(DiagESCandidateRetiringError):
        manager.acquire("candidate")
    manager.release("candidate")
    assert manager.status()["candidates"] == {}


def test_effective_digest_requires_exact_cpu_fp32_payload():
    dense, grouped = _payload()
    digest = compute_effective_model_digest(
        model_artifact_id="qwen3-artifact",
        schema_id=QWEN3_30B_A3B_SCHEMA_ID,
        schema_digest="ab" * 32,
        dense_deltas=dense,
        grouped_deltas=grouped,
    )
    assert len(digest) == 64
    with pytest.raises(ValueError, match="float32"):
        compute_effective_model_digest(
            model_artifact_id="qwen3-artifact",
            schema_id=QWEN3_30B_A3B_SCHEMA_ID,
            schema_digest="ab" * 32,
            dense_deltas={"dense": dense["dense"].bfloat16()},
            grouped_deltas=grouped,
        )


def test_register_protocol_round_trip_and_duplicate_rejection():
    dense, grouped = _payload()
    encoded = prepare_register_payload(dense, grouped)
    decoded_dense, decoded_grouped = parse_register_payload(encoded)
    assert decoded_dense.keys() == dense.keys()
    assert decoded_grouped.keys() == grouped.keys()
    assert torch.equal(decoded_dense["dense"], dense["dense"])
    assert torch.equal(decoded_grouped["moe_fc1_pre"], grouped["moe_fc1_pre"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_register_payload([encoded[0], encoded[0]])
    with pytest.raises(ValueError, match="unknown"):
        parse_register_payload([("legacy:dense", dense["dense"])])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action": "register", "candidate_id": None, "serialized_deltas": [b"x"]},
        {"action": "register", "candidate_id": "c", "serialized_deltas": None},
        {"action": "retire", "candidate_id": "c", "serialized_deltas": [b"x"]},
        {"action": "status", "candidate_id": "c", "serialized_deltas": None},
        {"action": "legacy", "candidate_id": None, "serialized_deltas": None},
    ],
)
def test_registry_protocol_rejects_invalid_action_field_combinations(kwargs):
    expected_error = (
        DiagESInvalidCandidateError
        if kwargs["action"] in ("register", "retire") and kwargs["candidate_id"] is None
        else ValueError
    )
    with pytest.raises(expected_error):
        validate_registry_request(effective_model_digest=None, **kwargs)
