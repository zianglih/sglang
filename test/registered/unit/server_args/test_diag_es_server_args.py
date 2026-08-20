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
        "disaggregation_mode": "null",
        "enable_two_batch_overlap": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _validate_runtime(view, *, placement="pre"):
    subject = SimpleNamespace(
        diag_es_target_placement=placement,
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


def test_diag_es_accepts_only_post_with_deepseek_block_fp8_triton():
    _validate_runtime(
        _runtime_view(quantization="fp8", fp8_gemm_runner_backend="triton"),
        placement="post",
    )
    with pytest.raises(ValueError, match="block-FP8 requires 'post'"):
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
            _runtime_view(
                quantization="mxfp8", fp8_gemm_runner_backend="triton"
            ),
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

    with pytest.raises(ValueError, match="only --speculative-algorithm NEXTN"):
        _validate_identity(
            diag_es_target_placement="pre", speculative_algorithm="EAGLE"
        )


@pytest.mark.parametrize("placement", ["pre", "post", "both"])
def test_mtp_diag_es_interface_fails_closed(placement):
    with pytest.raises(NotImplementedError, match="interface-only"):
        _validate_identity(diag_es_mtp_placement=placement)


def test_invalid_role_placement_is_rejected():
    with pytest.raises(ValueError, match="diag_es_target_placement"):
        _validate_identity(diag_es_target_placement="invalid")
    with pytest.raises(ValueError, match="diag_es_mtp_placement"):
        _validate_identity(diag_es_mtp_placement="invalid")


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
