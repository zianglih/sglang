from types import SimpleNamespace

import pytest
import torch

from sglang.srt.diag_es.manifest import (
    DiagESManifest,
    DenseSite,
    Qwen3DiagESManifest,
    QWEN2_5_1_5B_SCHEMA_ID,
    compute_effective_model_digest,
    register_qwen2_5_1_5b_dense_sites,
)
from sglang.srt.diag_es.manager import DiagESManager
from sglang.srt.diag_es.protocol import prepare_register_payload


class _Linear:
    def __init__(self, input_size: int, weight_dtype=torch.bfloat16):
        self.input_size = input_size
        self.weight = SimpleNamespace(dtype=weight_dtype)
        self.es_site_id = None
        self.es_site_width = None


class Qwen2ForCausalLM:
    def __init__(
        self,
        *,
        num_layers=28,
        hidden_size=1536,
        intermediate_size=8960,
        head_dim=None,
        quant_config=None,
        weight_dtype=torch.bfloat16,
    ):
        self.quant_config = quant_config
        self.config = SimpleNamespace(
            architectures=["Qwen2ForCausalLM"],
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=12,
            num_key_value_heads=2,
        )
        if head_dim is not None:
            self.config.head_dim = head_dim
        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        qkv_proj=_Linear(hidden_size, weight_dtype),
                        o_proj=_Linear(hidden_size, weight_dtype),
                    ),
                    mlp=SimpleNamespace(
                        gate_up_proj=_Linear(hidden_size, weight_dtype),
                        down_proj=_Linear(intermediate_size, weight_dtype),
                    ),
                )
                for _ in range(num_layers)
            ]
        )


def test_qwen2_manifest_registers_exact_dense_physical_sites():
    model = Qwen2ForCausalLM()
    manifest = register_qwen2_5_1_5b_dense_sites(model, tp_size=1)

    assert manifest.schema_id == QWEN2_5_1_5B_SCHEMA_ID
    assert manifest.schema_digest == (
        "28c1333e3b33f3d18308a36d331a19e0fbc0257ad72e8fd716807114f40555da"
    )
    assert len(manifest.dense_sites) == 112
    assert manifest.grouped_gate_shapes == {}
    assert [(site.site_id, site.input_width) for site in manifest.dense_sites[:4]] == [
        ("model.layers.0.self_attn.qkv_proj.input", 1536),
        ("model.layers.0.self_attn.o_proj.input", 1536),
        ("model.layers.0.mlp.gate_up_proj.input", 1536),
        ("model.layers.0.mlp.down_proj.input", 8960),
    ]
    layer = model.model.layers[0]
    assert layer.self_attn.qkv_proj.es_site_width == 1536
    assert layer.mlp.down_proj.es_site_width == 8960


@pytest.mark.parametrize(
    ("model", "tp_size", "match"),
    [
        (Qwen2ForCausalLM(num_layers=27), 1, "num_hidden_layers"),
        (Qwen2ForCausalLM(hidden_size=1024), 1, "hidden_size"),
        (Qwen2ForCausalLM(intermediate_size=8192), 1, "intermediate_size"),
        (Qwen2ForCausalLM(head_dim=64), 1, "head_dim"),
        (Qwen2ForCausalLM(quant_config=object()), 1, "unquantized"),
        (Qwen2ForCausalLM(weight_dtype=torch.float16), 1, "bfloat16"),
        (Qwen2ForCausalLM(), 2, "tp_size"),
        (SimpleNamespace(config=Qwen2ForCausalLM().config), 1, "Qwen2ForCausalLM"),
    ],
)
def test_qwen2_manifest_rejects_wrong_architecture_shape_or_tp(model, tp_size, match):
    with pytest.raises((TypeError, ValueError), match=match):
        register_qwen2_5_1_5b_dense_sites(model, tp_size=tp_size)


def test_generic_digest_hashes_empty_grouped_qwen2_fp32_delta_payload():
    dense = {"site": torch.zeros(3, dtype=torch.float32)}
    digest = compute_effective_model_digest(
        model_artifact_id="qwen2-local",
        schema_id=QWEN2_5_1_5B_SCHEMA_ID,
        schema_digest="ab" * 32,
        dense_deltas=dense,
        grouped_deltas={},
    )
    assert len(digest) == 64
    assert digest == compute_effective_model_digest(
        model_artifact_id="qwen2-local",
        schema_id=QWEN2_5_1_5B_SCHEMA_ID,
        schema_digest="ab" * 32,
        dense_deltas=dense,
        grouped_deltas={},
    )


def test_generic_digest_rejects_grouped_and_legacy_expert_conflict():
    expert = torch.zeros((1, 1, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="conflict"):
        compute_effective_model_digest(
            base_model_revision="legacy",
            schema_digest="schema",
            dense_deltas={},
            grouped_deltas={"moe_fc1": expert, "moe_fc2": expert},
            expert_fc1_deltas=expert,
            expert_fc2_deltas=expert,
        )


def test_digest_rejects_conflicting_legacy_and_generic_artifact_identity():
    expert = torch.zeros((1, 1, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="conflict"):
        compute_effective_model_digest(
            base_model_revision="legacy-artifact",
            model_artifact_id="different-artifact",
            schema_digest="schema",
            dense_deltas={},
            expert_fc1_deltas=expert,
            expert_fc2_deltas=expert,
        )


def test_dense_manager_status_uses_generic_identity_and_no_expert_fields():
    manifest = DiagESManifest(
        schema_id=QWEN2_5_1_5B_SCHEMA_ID,
        dense_sites=(DenseSite("dense", 3),),
        grouped_gate_shapes={},
        schema_digest="ab" * 32,
    )
    manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=2,
        model_artifact_id="qwen2-local",
        device=torch.device("cpu"),
    )

    status = manager.status()
    assert status["schema_id"] == QWEN2_5_1_5B_SCHEMA_ID
    assert status["model_artifact_id"] == "qwen2-local"
    assert status["dense_sites"] == {"dense": 3}
    assert status["grouped_gate_shapes"] == {}
    assert "base_model_revision" not in status
    assert "expert_fc1_shape" not in status
    assert "expert_fc2_shape" not in status


def test_manager_rejects_blank_artifact_identity():
    manifest = DiagESManifest(
        schema_id=QWEN2_5_1_5B_SCHEMA_ID,
        dense_sites=(),
        grouped_gate_shapes={},
        schema_digest="ab" * 32,
    )
    with pytest.raises(ValueError, match="model_artifact_id"):
        DiagESManager(
            manifest=manifest,
            resident_candidate_slots=1,
            model_artifact_id="  ",
            device=torch.device("cpu"),
        )


def test_dense_manager_accepts_empty_grouped_payload(monkeypatch):
    manifest = DiagESManifest(
        schema_id=QWEN2_5_1_5B_SCHEMA_ID,
        dense_sites=(DenseSite("dense", 3),),
        grouped_gate_shapes={},
        schema_digest="ab" * 32,
    )
    manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=1,
        model_artifact_id="qwen2-local",
        device=torch.device("cpu"),
    )

    class _Stream:
        def synchronize(self):
            pass

    class _StreamContext:
        def __enter__(self):
            pass

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: _StreamContext())
    registered = manager.register_candidate(
        candidate_id="candidate",
        dense_deltas={"dense": torch.zeros(3, dtype=torch.float32)},
        grouped_deltas={},
    )
    assert registered["state"] == "READY"


def test_legacy_manager_status_retains_qwen3_fields():
    manifest = Qwen3DiagESManifest(
        dense_sites=(DenseSite("dense", 3),),
        num_layers=1,
        num_experts=2,
        hidden_size=3,
        moe_intermediate_size=4,
        schema_digest="cd" * 32,
    )
    manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=1,
        base_model_revision="legacy-artifact",
        device=torch.device("cpu"),
    )

    status = manager.status()
    assert status["base_model_revision"] == "legacy-artifact"
    assert status["expert_fc1_shape"] == [1, 2, 3]
    assert status["expert_fc2_shape"] == [1, 2, 4]
    assert status["grouped_gate_shapes"] == {
        "moe_fc1": [1, 2, 3],
        "moe_fc2": [1, 2, 4],
    }


def test_register_protocol_serializes_generic_grouped_names():
    delta = torch.zeros(2, dtype=torch.float32)
    named, digest = prepare_register_payload(
        {"site": delta},
        grouped_deltas={"custom": delta},
        effective_model_digest="digest",
    )
    assert dict(named) == {
        "dense_delta:site": delta,
        "grouped_delta:custom": delta,
    }
    assert digest == "digest"


def test_register_protocol_preserves_legacy_positional_adapter():
    fc1 = torch.zeros(2, dtype=torch.float32)
    fc2 = torch.zeros(3, dtype=torch.float32)
    named, digest = prepare_register_payload({}, fc1, fc2, "legacy-digest")
    assert dict(named) == {
        "expert_delta:moe_fc1": fc1,
        "expert_delta:moe_fc2": fc2,
    }
    assert digest == "legacy-digest"


def test_register_protocol_rejects_grouped_and_expert_keywords():
    delta = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="conflict"):
        prepare_register_payload({}, delta, delta, grouped_deltas={})
