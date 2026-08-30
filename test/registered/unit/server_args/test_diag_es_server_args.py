from types import SimpleNamespace

import pytest
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _runtime_view(**overrides):
    values = {
        "device": "cuda",
        "nnodes": 1,
        "node_rank": 0,
        "tp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
        "dcp_size": 1,
        "attn_cp_size": 1,
        "moe_dp_size": 1,
        "dwdp_size": 1,
        "enable_prefill_cp": False,
        "enable_prefill_context_parallel": False,
        "enable_dsa_prefill_context_parallel": False,
        "enable_dp_attention": False,
        "moe_a2a_backend": "none",
        "moe_runner_backend": "triton",
        "bf16_gemm_backend": "triton",
        "fp8_gemm_runner_backend": "auto",
        "quantization": None,
        "attention_backend": "trtllm_mha",
        "decode_attention_backend": None,
        "prefill_attention_backend": None,
        "page_size": 1,
        "enable_torch_compile": False,
        "ep_num_redundant_experts": 0,
        "enable_eplb": False,
        "elastic_ep_backend": None,
        "disable_radix_cache": False,
        "radix_cache_backend": None,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "enable_flexkv": False,
        "enable_lora": None,
        "lora_paths": None,
        "speculative_algorithm": None,
        "speculative_num_steps": 3,
        "speculative_num_draft_tokens": 4,
        "speculative_eagle_topk": 1,
        "speculative_adaptive": False,
        "enable_multi_layer_eagle": False,
        "disable_overlap_schedule": True,
        "max_running_requests": 64,
        "disaggregation_mode": "null",
        "enable_two_batch_overlap": False,
        "enable_fused_moe_sum_all_reduce": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate_runtime(view, *, placement="pre"):
    subject = SimpleNamespace(
        diag_es_target_placement=placement,
        diag_es_mtp_placement="off",
        _resolved=lambda: view,
    )
    ServerArgs._handle_diag_es_runtime_contract(subject)


def test_diag_es_accepts_only_the_exact_runtime_contract():
    _validate_runtime(_runtime_view())


def test_target_diag_es_may_coexist_with_clean_mtp():
    # The speculative hook canonicalizes an admitted NEXTN request to EAGLE
    # before the final runtime-contract validation.
    _validate_runtime(_runtime_view(speculative_algorithm="EAGLE"))
    with pytest.raises(ValueError, match="NEXTN resolved as 'EAGLE'"):
        _validate_runtime(_runtime_view(speculative_algorithm="EAGLE3"))


def test_diag_es_accepts_post_and_rank1_with_deepseek_block_fp8_triton():
    _validate_runtime(
        _runtime_view(quantization="fp8", fp8_gemm_runner_backend="triton"),
        placement="post",
    )
    _validate_runtime(
        _runtime_view(quantization="fp8", fp8_gemm_runner_backend="triton"),
        placement="rank1",
    )
    with pytest.raises(ValueError, match="block-FP8 requires 'post' or 'rank1'"):
        _validate_runtime(
            _runtime_view(quantization="fp8", fp8_gemm_runner_backend="triton"),
            placement="pre",
        )
    with pytest.raises(ValueError, match="block-FP8 requires 'triton'"):
        _validate_runtime(
            _runtime_view(quantization="fp8", fp8_gemm_runner_backend="auto"),
            placement="post",
        )
    with pytest.raises(ValueError, match="requires None or 'fp8'"):
        _validate_runtime(
            _runtime_view(quantization="mxfp8", fp8_gemm_runner_backend="triton"),
            placement="post",
        )


def test_rank1_rejects_fused_moe_sum_all_reduce():
    with pytest.raises(ValueError, match="rank1 requires per-route MoE outputs"):
        _validate_runtime(
            _runtime_view(enable_fused_moe_sum_all_reduce=True),
            placement="rank1",
        )

    # The restriction is specific to the standalone rank1 residual path.
    _validate_runtime(
        _runtime_view(enable_fused_moe_sum_all_reduce=True),
        placement="post",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "cpu"),
        ("enable_prefill_cp", True),
        ("attention_backend", "fa3"),
        ("decode_attention_backend", "fa3"),
        ("enable_torch_compile", True),
        ("ep_num_redundant_experts", 1),
        ("enable_hierarchical_cache", True),
        ("radix_cache_backend", "custom"),
        ("enable_lora", True),
        ("lora_paths", ["adapter"]),
    ],
)
def test_diag_es_rejects_runtime_contract_drift(field, value):
    with pytest.raises(ValueError, match="diagonal ES supports only"):
        _validate_runtime(_runtime_view(**{field: value}))


def _validate_identity(**overrides):
    values = {
        "diag_es_target_placement": "off",
        "diag_es_mtp_placement": "off",
        "diag_es_schema_id": None,
        "diag_es_model_artifact_id": None,
        "diag_es_resident_candidate_slots": 0,
        "diag_es_mtp_schema_id": None,
        "diag_es_mtp_model_artifact_id": None,
        "diag_es_mtp_max_sessions": 64,
        "speculative_algorithm": None,
    }
    values.update(overrides)
    ServerArgs._handle_diag_es_identity(SimpleNamespace(**values))


def test_diag_es_identity_is_role_scoped():
    args = ServerArgs(model_path="dummy")
    assert args.diag_es_target_placement == "off"
    assert args.diag_es_mtp_placement == "off"
    _validate_identity()
    _validate_identity(
        diag_es_target_placement="pre",
        diag_es_schema_id="qwen3-30b-a3b-diag-es-v2",
        diag_es_model_artifact_id="artifact",
        diag_es_resident_candidate_slots=1,
        speculative_algorithm="NEXTN",
    )
    _validate_identity(
        diag_es_target_placement="pre",
        diag_es_schema_id="qwen3-30b-a3b-diag-es-v2",
        diag_es_model_artifact_id="artifact",
        diag_es_resident_candidate_slots=1,
        speculative_algorithm="nextn",
    )
    _validate_identity(
        diag_es_target_placement="rank1",
        diag_es_schema_id="qwen3-30b-a3b-rank1-es-v1",
        diag_es_model_artifact_id="artifact",
        diag_es_resident_candidate_slots=1,
        speculative_algorithm="NEXTN",
    )
    with pytest.raises(ValueError, match="rank1-es-v1"):
        _validate_identity(
            diag_es_target_placement="rank1",
            diag_es_schema_id="qwen3-30b-a3b-diag-es-v2",
            diag_es_model_artifact_id="artifact",
            diag_es_resident_candidate_slots=1,
        )

    with pytest.raises(ValueError, match="only --speculative-algorithm NEXTN"):
        _validate_identity(
            diag_es_target_placement="pre", speculative_algorithm="EAGLE"
        )


def test_mtp_diag_es_identity_is_exact_and_allows_independent_target_role():
    values = dict(
        diag_es_mtp_placement="post",
        diag_es_mtp_schema_id="joyai-llm-flash-mtp-diag-es-v2",
        diag_es_mtp_model_artifact_id="joyai@sha256:abc",
        speculative_algorithm="EAGLE",
    )
    _validate_identity(**values)
    _validate_identity(
        **{
            **values,
            "diag_es_target_placement": "post",
            "diag_es_schema_id": "qwen3-30b-a3b-diag-es-v2",
            "diag_es_model_artifact_id": "target",
            "diag_es_resident_candidate_slots": 2,
            "speculative_algorithm": "NEXTN",
        }
    )
    _validate_identity(
        **{
            **values,
            "diag_es_target_placement": "rank1",
            "diag_es_schema_id": "qwen3-30b-a3b-rank1-es-v1",
            "diag_es_model_artifact_id": "target",
            "diag_es_resident_candidate_slots": 2,
            "speculative_algorithm": "NEXTN",
        }
    )
    _validate_identity(**{**values, "diag_es_mtp_placement": "pre"})
    _validate_identity(**{**values, "diag_es_mtp_placement": "both"})
    with pytest.raises(ValueError, match="schema_id"):
        _validate_identity(**{**values, "diag_es_mtp_schema_id": "wrong"})


def test_mtp_diag_es_runtime_allows_target_moe_backend_auto():
    subject = SimpleNamespace(
        diag_es_target_placement="off",
        diag_es_mtp_placement="post",
        diag_es_mtp_max_sessions=64,
        _resolved=lambda: _runtime_view(
            speculative_algorithm="EAGLE",
            moe_runner_backend="auto",
            attention_backend="triton",
        ),
    )
    subject._handle_diag_es_mtp_runtime_contract = lambda: (
        ServerArgs._handle_diag_es_mtp_runtime_contract(subject)
    )
    ServerArgs._handle_diag_es_runtime_contract(subject)


def test_mtp_diag_es_runtime_allows_official_topk2_shape():
    subject = SimpleNamespace(
        diag_es_target_placement="off",
        diag_es_mtp_placement="both",
        diag_es_mtp_max_sessions=64,
        _resolved=lambda: _runtime_view(
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            speculative_eagle_topk=2,
            attention_backend="triton",
            page_size=64,
        ),
    )
    subject._handle_diag_es_mtp_runtime_contract = lambda: (
        ServerArgs._handle_diag_es_mtp_runtime_contract(subject)
    )
    ServerArgs._handle_diag_es_runtime_contract(subject)


def test_mtp_diag_es_rejects_flashinfer_tree_path():
    subject = SimpleNamespace(
        diag_es_target_placement="off",
        diag_es_mtp_placement="both",
        diag_es_mtp_max_sessions=64,
        _resolved=lambda: _runtime_view(
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            speculative_eagle_topk=2,
            attention_backend="flashinfer",
        ),
    )
    with pytest.raises(ValueError, match="attention_backend='flashinfer'"):
        ServerArgs._handle_diag_es_mtp_runtime_contract(subject)


def test_mtp_diag_es_rejects_topk2_trtllm_mla_page_tree():
    subject = SimpleNamespace(
        diag_es_target_placement="off",
        diag_es_mtp_placement="both",
        diag_es_mtp_max_sessions=64,
        _resolved=lambda: _runtime_view(
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            speculative_eagle_topk=2,
            attention_backend="trtllm_mla",
            page_size=64,
        ),
    )
    with pytest.raises(ValueError, match="topk>1 with page_size>1"):
        ServerArgs._handle_diag_es_mtp_runtime_contract(subject)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("radix_cache_backend", "custom"),
        ("enable_hierarchical_cache", True),
        ("enable_lmcache", True),
        ("enable_flexkv", True),
    ],
)
def test_mtp_diag_es_rejects_external_cache_backends(field, value):
    subject = SimpleNamespace(
        diag_es_target_placement="off",
        diag_es_mtp_placement="post",
        diag_es_mtp_max_sessions=64,
        _resolved=lambda: _runtime_view(
            speculative_algorithm="EAGLE",
            attention_backend="triton",
            **{field: value},
        ),
    )
    with pytest.raises(ValueError, match="Triton-attention/GEMM"):
        ServerArgs._handle_diag_es_mtp_runtime_contract(subject)


def test_mtp_diag_es_rejects_multi_layer_eagle():
    subject = SimpleNamespace(
        diag_es_target_placement="off",
        diag_es_mtp_placement="post",
        diag_es_mtp_max_sessions=64,
        _resolved=lambda: _runtime_view(
            speculative_algorithm="EAGLE",
            attention_backend="triton",
            enable_multi_layer_eagle=True,
        ),
    )
    subject._handle_diag_es_mtp_runtime_contract = lambda: (
        ServerArgs._handle_diag_es_mtp_runtime_contract(subject)
    )
    with pytest.raises(ValueError, match="enable_multi_layer_eagle"):
        ServerArgs._handle_diag_es_runtime_contract(subject)


def test_invalid_role_placement_is_rejected():
    with pytest.raises(ValueError, match="diag_es_target_placement"):
        _validate_identity(diag_es_target_placement="invalid")
    with pytest.raises(ValueError, match="diag_es_mtp_placement"):
        _validate_identity(diag_es_mtp_placement="invalid")
    with pytest.raises(ValueError, match="diag_es_mtp_placement"):
        _validate_identity(diag_es_mtp_placement="rank1")


def test_diag_es_runner_role_helpers():
    from sglang.srt.diag_es import get_diag_es_placement, is_diag_es_enabled

    args = SimpleNamespace(
        diag_es_target_placement="both",
        diag_es_mtp_placement="off",
    )
    assert get_diag_es_placement(args, is_draft_worker=False) == "both"
    assert get_diag_es_placement(args, is_draft_worker=True) is None
    assert is_diag_es_enabled(args, is_draft_worker=False)
    assert not is_diag_es_enabled(args, is_draft_worker=True)

    rank1_args = SimpleNamespace(
        diag_es_target_placement="rank1",
        diag_es_mtp_placement="off",
    )
    assert get_diag_es_placement(rank1_args, is_draft_worker=False) == "rank1"
    assert get_diag_es_placement(rank1_args, is_draft_worker=True) is None

    mixed_args = SimpleNamespace(
        diag_es_target_placement="rank1",
        diag_es_mtp_placement="post",
    )
    assert get_diag_es_placement(mixed_args, is_draft_worker=False) == "rank1"
    assert get_diag_es_placement(mixed_args, is_draft_worker=True) == "post"
