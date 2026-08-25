from types import SimpleNamespace

import pytest
import torch

import sglang.srt.diag_es.manager as manager_module
import sglang.srt.diag_es.manifest as manifest_module
from sglang.srt.diag_es.manager import (
    DiagESCandidateNotFoundError,
    DiagESCandidateRetiringError,
    DiagESInvalidCandidateError,
    DiagESManager,
)
from sglang.srt.diag_es.manifest import (
    JOYAI_LLM_FLASH_MTP_SCHEMA_ID,
    QWEN3_30B_A3B_SCHEMA_ID,
    DenseSite,
    DiagESManifest,
    compute_effective_model_digest,
    register_joyai_llm_flash_mtp_manifest,
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
        self.calls = []
        if block_fp8:
            self.weight_scale_inv = torch.empty(
                (output_size // 128, input_size // 128),
                dtype=torch.float32,
                device="meta",
            )
            self.input_scale = None

    def __call__(self, value):
        self.calls.append(value.shape[0])
        return (
            torch.zeros(
                (value.shape[0], self.output_size),
                dtype=value.dtype,
                device=value.device,
            ),
            None,
        )


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
                        qkv_proj=_Linear(2048, 5120, dtype=dtype, block_fp8=block_fp8),
                        o_proj=_Linear(4096, 2048, dtype=dtype, block_fp8=block_fp8),
                    ),
                    mlp=SimpleNamespace(
                        gate=_Linear(2048, 128, dtype=dtype),
                        experts=_Experts(layer_id, dtype=dtype, block_fp8=block_fp8),
                    ),
                )
                for layer_id in range(num_layers)
            ]
        )


class Qwen2ForCausalLM:
    def __init__(
        self,
        *,
        num_layers=28,
        hidden_size=1536,
        dtype=torch.bfloat16,
        quant_config=None,
    ):
        self.quant_config = quant_config
        self.config = SimpleNamespace(
            architectures=["Qwen2ForCausalLM"],
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=8960,
            num_attention_heads=12,
            num_key_value_heads=2,
            head_dim=128,
        )
        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        qkv_proj=_Linear(1536, 2048, dtype=dtype),
                        o_proj=_Linear(1536, 1536, dtype=dtype),
                    ),
                    mlp=SimpleNamespace(
                        gate_up_proj=_Linear(1536, 17920, dtype=dtype),
                        down_proj=_Linear(8960, 1536, dtype=dtype),
                    ),
                )
                for _ in range(num_layers)
            ]
        )


class DeepseekV2MLP:
    def __init__(self):
        self.gate_up_proj = _Linear(2048, 14336)
        self.down_proj = _Linear(7168, 2048)


class JoyAIDenseNextNDecoderLayer:
    def __init__(self):
        self.self_attn = SimpleNamespace(
            fused_qkv_a_proj_with_mqa=_Linear(2048, 2112),
            q_b_proj=_Linear(1536, 6144),
            kv_b_proj=_Linear(512, 8192),
            o_proj=_Linear(4096, 2048),
        )
        self.mlp = DeepseekV2MLP()


class JoyAILLMFlashForCausalLMNextN:
    def __init__(self):
        self.quant_config = None
        self.config = SimpleNamespace(
            num_hidden_layers=40,
            hidden_size=2048,
            num_attention_heads=32,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            intermediate_size=7168,
            num_nextn_predict_layers=1,
        )
        self.model = SimpleNamespace(decoder=JoyAIDenseNextNDecoderLayer())


@pytest.mark.parametrize(
    ("placement", "sites_per_layer", "parameter_count", "schema_digest"),
    [
        (
            "pre",
            [
                ("self_attn.qkv_proj.input", 1536),
                ("self_attn.o_proj.input", 1536),
                ("mlp.gate_up_proj.input", 1536),
                ("mlp.down_proj.input", 8960),
            ],
            379_904,
            "991d7c66de5120c2b0442efe41e0f1a2e25b6cb04785940804850f32d5c3ba09",
        ),
        (
            "post",
            [
                ("self_attn.qkv_proj.output", 2048),
                ("self_attn.o_proj.output", 1536),
                ("mlp.gate_up_proj.output", 17920),
                ("mlp.down_proj.output", 1536),
            ],
            645_120,
            "77572f5350d4564f67e189d4ac95c8b056c6adb8d719d438ae6e1cf3442fb517",
        ),
        (
            "both",
            [
                ("self_attn.qkv_proj.input", 1536),
                ("self_attn.qkv_proj.output", 2048),
                ("self_attn.o_proj.input", 1536),
                ("self_attn.o_proj.output", 1536),
                ("mlp.gate_up_proj.input", 1536),
                ("mlp.gate_up_proj.output", 17920),
                ("mlp.down_proj.input", 8960),
                ("mlp.down_proj.output", 1536),
            ],
            1_025_024,
            "95a117e2b166f05f3e030645f83e28d8cea24aa8c20b3fcfdd4d027124f9e033",
        ),
    ],
)
def test_qwen2_v2_manifest_exact_contract(
    placement, sites_per_layer, parameter_count, schema_digest
):
    register = getattr(manifest_module, "register_qwen2_5_1_5b_manifest", None)
    assert callable(register), "Qwen2.5 v2 manifest registration is missing"

    manifest = register(Qwen2ForCausalLM(), placement=placement)

    assert manifest.schema_id == "qwen2.5-1.5b-instruct-dense-diag-es-v2"
    assert manifest.placement == placement
    assert manifest.schema_digest == schema_digest
    assert manifest.grouped_delta_shapes == {}
    assert len(manifest.dense_sites) == len(sites_per_layer) * 28
    assert sum(site.width for site in manifest.dense_sites) == parameter_count
    assert [
        (site.site_id.removeprefix("model.layers.0."), site.width)
        for site in manifest.dense_sites[: len(sites_per_layer)]
    ] == sites_per_layer


def test_qwen2_v2_manifest_rejects_wrong_dtype_quantization_and_geometry():
    register = manifest_module.register_qwen2_5_1_5b_manifest
    for model, match in (
        (SimpleNamespace(), "Qwen2ForCausalLM"),
        (Qwen2ForCausalLM(dtype=torch.float16), "bfloat16"),
        (Qwen2ForCausalLM(quant_config=SimpleNamespace()), "unquantized"),
        (Qwen2ForCausalLM(num_layers=27), "num_hidden_layers"),
        (Qwen2ForCausalLM(hidden_size=2048), "hidden_size"),
    ):
        with pytest.raises((TypeError, ValueError), match=match):
            register(model, placement="pre")

    model = Qwen2ForCausalLM()
    model.model.layers[0].self_attn.qkv_proj.weight = torch.empty(
        (2048, 1535), dtype=torch.bfloat16, device="meta"
    )
    with pytest.raises(ValueError, match="qkv_proj.weight"):
        register(model, placement="pre")


def test_qwen3_v1_manifest_preserves_pre_only_resume_identity():
    register = getattr(manifest_module, "register_qwen3_30b_a3b_v1_manifest", None)
    assert callable(register), "Qwen3 v1 compatibility manifest is missing"

    manifest = register(Qwen3MoeForCausalLM(), placement="pre")
    assert manifest.schema_id == "qwen3-30b-a3b-diag-es-v1"
    assert manifest.schema_digest == (
        "65fc2ae92a979c997d1ee37f5f909663091272bc7465c629e9884cbeb031025a"
    )
    assert manifest.grouped_delta_shapes == {
        "moe_fc1": (48, 128, 2048),
        "moe_fc2": (48, 128, 768),
    }
    with pytest.raises(ValueError, match="pre"):
        register(Qwen3MoeForCausalLM(), placement="post")


def test_effective_digest_supports_qwen2_v4_and_preserves_qwen3_v1_v3():
    dense = {"dense": torch.tensor([1e-4, -1e-4], dtype=torch.float32)}
    qwen2 = manifest_module.compute_effective_model_digest(
        model_artifact_id="qwen2-artifact",
        schema_id="qwen2.5-1.5b-instruct-dense-diag-es-v2",
        schema_digest="ab" * 32,
        dense_deltas=dense,
        grouped_deltas={},
    )
    assert len(qwen2) == 64

    qwen3_v1 = manifest_module.compute_effective_model_digest(
        model_artifact_id="qwen3-artifact",
        schema_id="qwen3-30b-a3b-diag-es-v1",
        schema_digest="ab" * 32,
        dense_deltas=dense,
        grouped_deltas={
            "moe_fc1": torch.tensor([[1.0]], dtype=torch.float32),
            "moe_fc2": torch.tensor([[2.0]], dtype=torch.float32),
        },
    )
    assert qwen3_v1 == (
        "7b74b178734a6d79b1c8ebc07c03297f9ebc758f49b6c805486d564e170317d5"
    )


class _StubManager:
    def __init__(
        self, *, manifest, resident_candidate_slots, model_artifact_id, device
    ):
        self.manifest = manifest
        self.resident_candidate_slots = resident_candidate_slots
        self.model_artifact_id = model_artifact_id
        self.device = device
        self.requested_grouped_banks = []

    def get_dense_delta_bank(self, site_id):
        return ("dense", site_id)

    def get_grouped_delta_bank(self, name):
        self.requested_grouped_banks.append(name)
        return [("grouped", name, layer_id) for layer_id in range(48)]


@pytest.mark.parametrize("placement", ["pre", "post", "both"])
def test_manager_dispatches_qwen2_and_binds_all_dense_projections(
    monkeypatch, placement
):
    monkeypatch.setattr(manager_module, "DiagESManager", _StubManager)
    monkeypatch.setattr(manager_module, "_target_manager", None)
    model = Qwen2ForCausalLM()
    model.parameters = lambda: iter((torch.empty(0),))

    manager = manager_module.register_diag_es_model(
        model,
        schema_id="qwen2.5-1.5b-instruct-dense-diag-es-v2",
        resident_candidate_slots=2,
        model_artifact_id="qwen2-artifact",
        placement=placement,
    )

    assert manager.manifest.placement == placement
    assert manager.manifest.grouped_delta_shapes == {}
    for linear in (
        model.model.layers[0].self_attn.qkv_proj,
        model.model.layers[0].self_attn.o_proj,
        model.model.layers[0].mlp.gate_up_proj,
        model.model.layers[0].mlp.down_proj,
    ):
        assert (linear.es_pre_delta_bank is not None) is (placement in ("pre", "both"))
        assert (linear.es_post_delta_bank is not None) is (
            placement in ("post", "both")
        )


def test_manager_maps_qwen3_v1_grouped_names_to_pre_hot_path(monkeypatch):
    monkeypatch.setattr(manager_module, "DiagESManager", _StubManager)
    monkeypatch.setattr(manager_module, "_target_manager", None)
    model = Qwen3MoeForCausalLM()
    model.parameters = lambda: iter((torch.empty(0),))

    manager = manager_module.register_diag_es_model(
        model,
        schema_id="qwen3-30b-a3b-diag-es-v1",
        resident_candidate_slots=2,
        model_artifact_id="qwen3-artifact",
        placement="pre",
    )

    banks = model.model.layers[0].mlp.experts.moe_runner_config.diag_es_delta_banks
    assert banks.fc1_pre is not None and banks.fc2_pre is not None
    assert banks.fc1_post is None and banks.fc2_post is None
    assert set(manager.requested_grouped_banks) == {"moe_fc1", "moe_fc2"}


@pytest.mark.parametrize(
    ("placement", "expected_sites"),
    [
        (
            "pre",
            [
                (
                    "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.input",
                    2048,
                    None,
                ),
                ("model.decoder.self_attn.q_b_proj.input", 1536, None),
                ("model.decoder.self_attn.o_proj.input", 4096, None),
                ("model.decoder.mlp.gate_up_proj.input", 2048, None),
                ("model.decoder.mlp.down_proj.input", 7168, None),
            ],
        ),
        (
            "post",
            [
                (
                    "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.output",
                    2112,
                    None,
                ),
                ("model.decoder.self_attn.q_b_proj.output", 6144, None),
                ("model.decoder.self_attn.o_proj.output", 2048, None),
                ("model.decoder.mlp.gate_up_proj.output", 14336, None),
                ("model.decoder.mlp.down_proj.output", 2048, None),
            ],
        ),
        (
            "both",
            [
                (
                    "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.input",
                    2048,
                    None,
                ),
                (
                    "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.output",
                    2112,
                    None,
                ),
                ("model.decoder.self_attn.q_b_proj.input", 1536, None),
                ("model.decoder.self_attn.q_b_proj.output", 6144, None),
                ("model.decoder.self_attn.o_proj.input", 4096, None),
                ("model.decoder.self_attn.o_proj.output", 2048, None),
                ("model.decoder.mlp.gate_up_proj.input", 2048, None),
                ("model.decoder.mlp.gate_up_proj.output", 14336, None),
                ("model.decoder.mlp.down_proj.input", 7168, None),
                ("model.decoder.mlp.down_proj.output", 2048, None),
            ],
        ),
    ],
)
def test_joyai_mtp_manifest_is_exact_dense_search_space(placement, expected_sites):
    model = JoyAILLMFlashForCausalLMNextN()
    attention = model.model.decoder.self_attn
    attention.kv_b_proj.es_post_delta_bank = object()
    stale_fused_a_override = False if placement == "pre" else None
    attention._use_min_latency_fused_a_gemm = stale_fused_a_override
    attention._use_min_latency_q_b_gemm = stale_fused_a_override
    manifest = register_joyai_llm_flash_mtp_manifest(model, placement=placement)

    assert manifest.schema_id == JOYAI_LLM_FLASH_MTP_SCHEMA_ID
    assert manifest.placement == placement
    assert manifest.grouped_delta_shapes == {}
    assert [
        (site.site_id, site.width, site.active_width) for site in manifest.dense_sites
    ] == expected_sites
    assert attention._use_min_latency_fused_a_gemm is False
    expected_q_b_override = None if placement == "pre" else False
    assert attention._use_min_latency_q_b_gemm is expected_q_b_override
    # kv_b is absorbed into w_kc/w_vc by the active MLA backend, bypassing
    # LinearBase; it remains an explicit non-site to preserve the GEMM contract.
    assert attention.kv_b_proj.es_pre_site_id is None
    assert attention.kv_b_proj.es_post_site_id is None
    assert attention.kv_b_proj.es_pre_delta_bank is None
    assert attention.kv_b_proj.es_post_delta_bank is None
    assert (attention.fused_qkv_a_proj_with_mqa.es_pre_site_id is not None) == (
        placement in ("pre", "both")
    )
    assert (attention.fused_qkv_a_proj_with_mqa.es_post_site_id is not None) == (
        placement in ("post", "both")
    )


def test_joyai_mtp_v2_post_schema_digest_is_stable():
    manifest = register_joyai_llm_flash_mtp_manifest(
        JoyAILLMFlashForCausalLMNextN(), placement="post"
    )
    assert manifest.schema_digest == (
        "7cb156b471d337fa56184defb0296782b66a17fa14b11104c242d114345eb37e"
    )


def test_joyai_mtp_manifest_rejects_unknown_placement():
    with pytest.raises(ValueError, match="pre, post, or both"):
        register_joyai_llm_flash_mtp_manifest(
            JoyAILLMFlashForCausalLMNextN(), placement="sideways"
        )


@pytest.mark.parametrize("placement", ["post", "both"])
@pytest.mark.parametrize("num_tokens", [1, 64])
def test_joyai_mtp_post_query_sites_never_bypass_bound_linear_epilogue(
    monkeypatch, num_tokens, placement
):
    import sglang.srt.models.deepseek_v2 as deepseek_v2

    model = JoyAILLMFlashForCausalLMNextN()
    register_joyai_llm_flash_mtp_manifest(model, placement=placement)
    attention = model.model.decoder.self_attn
    attention.q_lora_rank = 1536
    attention.num_local_heads = 32
    attention.qk_head_dim = 192

    def bypass_is_forbidden(*_args, **_kwargs):
        raise AssertionError("min-latency fused-A bypassed the ES epilogue")

    monkeypatch.setattr(deepseek_v2, "linear_with_fused_a_gemm", bypass_is_forbidden)
    q_a = deepseek_v2.DeepseekV2AttentionMLA.prepare_qkv_latent(
        attention,
        torch.zeros(num_tokens, 2048, dtype=torch.bfloat16),
        forward_batch=None,
    )
    q_b = deepseek_v2.DeepseekV2AttentionMLA.q_b_proj_forward(
        attention,
        torch.zeros(num_tokens, 1536, dtype=torch.bfloat16),
    )

    assert q_a.shape == (num_tokens, 2112)
    assert q_b.shape == (num_tokens, 32, 192)
    assert attention.fused_qkv_a_proj_with_mqa.calls == [num_tokens]
    assert attention.q_b_proj.calls == [num_tokens]


def test_joyai_mtp_pre_keeps_cache_writing_q_a_on_replay_linear(monkeypatch):
    import sglang.srt.models.deepseek_v2 as deepseek_v2

    model = JoyAILLMFlashForCausalLMNextN()
    register_joyai_llm_flash_mtp_manifest(model, placement="pre")
    attention = model.model.decoder.self_attn
    attention.q_lora_rank = 1536

    def fused_a_is_forbidden(*_args, **_kwargs):
        raise AssertionError("cache-writing q_a must match the replay GEMM path")

    monkeypatch.setattr(deepseek_v2, "linear_with_fused_a_gemm", fused_a_is_forbidden)
    q_a = deepseek_v2.DeepseekV2AttentionMLA.prepare_qkv_latent(
        attention,
        torch.zeros(2, 2048, dtype=torch.bfloat16),
        forward_batch=None,
    )

    assert q_a.shape == (2, 2112)
    assert attention.fused_qkv_a_proj_with_mqa.calls == [2]


@pytest.mark.parametrize(
    ("num_tokens", "pre_bound", "expect_fused", "expect_pre"),
    [
        (2, False, True, False),
        (2, True, True, True),
        (17, True, False, False),
    ],
)
def test_min_latency_fused_a_applies_diag_es_pre_exactly_on_direct_gemm(
    monkeypatch, num_tokens, pre_bound, expect_fused, expect_pre
):
    import sglang.kernels.ops.gemm.fused_a_gemm as fused_a_gemm
    import sglang.srt.diag_es.ops as diag_es_ops

    linear = _Linear(2048, 2112)
    linear.es_pre_delta_bank = object() if pre_bound else None
    hidden_states = torch.zeros(num_tokens, 2048, dtype=torch.bfloat16)
    steered = torch.ones_like(hidden_states)
    pre_calls = []
    fused_calls = []

    def apply_pre(layer, value):
        pre_calls.append((layer, value))
        return steered

    def fused(value, weight, output=None, backend="auto"):
        fused_calls.append((value, weight, output, backend))
        return torch.zeros(num_tokens, 2112, dtype=torch.bfloat16)

    monkeypatch.setattr(diag_es_ops, "maybe_apply_diag_es_pre", apply_pre)
    monkeypatch.setattr(fused_a_gemm, "dsv3_fused_a_gemm", fused)

    output = fused_a_gemm.linear_with_fused_a_gemm(linear, hidden_states, backend="jit")

    assert output.shape == (num_tokens, 2112)
    assert bool(fused_calls) is expect_fused
    assert bool(pre_calls) is expect_pre
    if expect_fused:
        expected_input = steered if expect_pre else hidden_states
        assert fused_calls[0][0] is expected_input
        assert fused_calls[0][1]._base is linear.weight
        assert fused_calls[0][2:] == (None, "jit")
        assert linear.calls == []
    else:
        assert linear.calls == [num_tokens]


def test_joyai_pre_fused_a_captures_candidate_neutral_replay_input(monkeypatch):
    import sglang.srt.models.deepseek_v2 as deepseek_v2

    attention = JoyAILLMFlashForCausalLMNextN().model.decoder.self_attn
    attention.fused_qkv_a_proj_with_mqa.es_pre_delta_bank = object()
    attention.q_lora_rank = 1536
    attention.has_fused_proj = True
    attention.is_packed_weight = False
    attention._use_min_latency_fused_a_gemm = None
    attention.fused_a_gemm_backend = "jit"
    events = []

    attention.diag_es_mtp_kv_replay = SimpleNamespace(
        capture=lambda hidden_states, *_args: events.append(
            ("capture", hidden_states.clone())
        )
    )
    monkeypatch.setattr(
        deepseek_v2,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        ),
    )
    monkeypatch.setattr(
        deepseek_v2, "fused_a_gemm_weight_eligible", lambda _linear: True
    )

    def fused(linear, hidden_states, *, backend):
        events.append(("fused", hidden_states))
        return torch.zeros(hidden_states.shape[0], 2112, dtype=hidden_states.dtype)

    monkeypatch.setattr(deepseek_v2, "linear_with_fused_a_gemm", fused)
    hidden_states = torch.arange(2 * 2048, dtype=torch.int32).to(torch.bfloat16)
    hidden_states = hidden_states.view(2, 2048)
    forward_batch = SimpleNamespace(
        positions=torch.tensor([7, 8], dtype=torch.int64),
        forward_mode="draft_extend",
        out_cache_loc=torch.tensor([3, 4], dtype=torch.int64),
    )

    output = deepseek_v2.DeepseekV2AttentionMLA.prepare_qkv_latent(
        attention, hidden_states, forward_batch
    )

    assert output.shape == (2, 2112)
    assert [event[0] for event in events] == ["capture", "fused"]
    assert torch.equal(events[0][1], hidden_states)
    assert events[1][1] is hidden_states


@pytest.mark.parametrize(
    ("deterministic_inference", "expect_fused_a"),
    [(False, True), (True, False)],
)
def test_joyai_q_b_min_latency_fused_a_respects_deterministic_inference(
    monkeypatch, deterministic_inference, expect_fused_a
):
    import sglang.srt.models.deepseek_v2 as deepseek_v2

    attention = JoyAILLMFlashForCausalLMNextN().model.decoder.self_attn
    attention._use_min_latency_q_b_gemm = None
    attention._q_b_proj_verified_shape = True
    attention.fused_a_gemm_backend = "auto"
    attention.num_local_heads = 32
    attention.qk_head_dim = 192
    fused_calls = []

    monkeypatch.setattr(
        deepseek_v2,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(
                enable_deterministic_inference=deterministic_inference
            )
        ),
    )
    monkeypatch.setattr(
        deepseek_v2, "fused_a_gemm_weight_eligible", lambda _linear: True
    )

    def fused_a(linear, value, *, backend):
        fused_calls.append((backend, value.shape[0]))
        return torch.zeros(
            (value.shape[0], linear.output_size),
            dtype=value.dtype,
            device=value.device,
        )

    monkeypatch.setattr(deepseek_v2, "linear_with_fused_a_gemm", fused_a)
    q_b = deepseek_v2.DeepseekV2AttentionMLA.q_b_proj_forward(
        attention,
        torch.zeros(2, 1536, dtype=torch.bfloat16),
    )

    assert q_b.shape == (2, 32, 192)
    assert attention._use_min_latency_q_b_gemm is expect_fused_a
    assert fused_calls == ([("auto", 2)] if expect_fused_a else [])
    assert attention.q_b_proj.calls == ([] if expect_fused_a else [2])


def test_joyai_mtp_manifest_fails_closed_on_generic_sparse_nextn_layout():
    model = JoyAILLMFlashForCausalLMNextN()
    model.model.decoder.mlp = SimpleNamespace(experts=object())
    with pytest.raises(TypeError, match="dense NextN decoder layout"):
        register_joyai_llm_flash_mtp_manifest(model, placement="post")


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
