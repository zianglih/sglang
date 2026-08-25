"""GPU parity for JoyAI MTP draft-KV prefix replay.

This test deliberately consumes only the already-prepared local MTP artifact;
it never resolves a Hugging Face model ID or downloads weights.  On C2:

.. code-block:: bash

   SGLANG_JOYAI_MTP_TEST_MODEL_PATH=/data/ziangli/models/JoyAI-LLM-Flash-MTP \
     PYTHONPATH=python python -m pytest -q \
     test/registered/diag_es/test_diag_es_mtp_kv_replay_gpu.py
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b", runner_config="1-gpu-small")

_ARTIFACT_ENV = "SGLANG_JOYAI_MTP_TEST_MODEL_PATH"
_DEFAULT_ARTIFACT = "/data/ziangli/models/JoyAI-LLM-Flash-MTP"
_SHARD_NAME = "mtp-1-of-1.safetensors"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="JoyAI MTP replay parity requires CUDA"
)


def _unique_suffix(keys: list[str], suffix: str) -> str:
    matches = [key for key in keys if key.endswith(suffix)]
    assert len(matches) == 1, (
        f"expected exactly one MTP tensor ending in {suffix!r}, found {matches}"
    )
    return matches[0]


def _load_local_attention_artifact() -> tuple[dict, torch.Tensor, torch.Tensor]:
    artifact = Path(os.environ.get(_ARTIFACT_ENV, _DEFAULT_ARTIFACT)).resolve()
    config_path = artifact / "config.json"
    shard_path = artifact / _SHARD_NAME
    if not config_path.is_file() or not shard_path.is_file():
        pytest.skip(
            f"set {_ARTIFACT_ENV} to the prepared JoyAI MTP-only artifact "
            f"containing config.json and {_SHARD_NAME}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "model_type": "joyai_llm_flash",
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_nextn_predict_layers": 1,
    }
    for name, value in expected.items():
        assert config.get(name) == value, (
            f"prepared JoyAI MTP config has {name}={config.get(name)!r}, "
            f"expected {value!r}"
        )

    from safetensors import safe_open

    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
        keys = list(shard.keys())
        fused_matches = [
            key
            for key in keys
            if key.endswith(".self_attn.fused_qkv_a_proj_with_mqa.weight")
        ]
        if fused_matches:
            assert len(fused_matches) == 1, (
                f"expected one fused q_a/kv_a tensor, found {fused_matches}"
            )
            fused_weight = shard.get_tensor(fused_matches[0])
        else:
            q_a = shard.get_tensor(_unique_suffix(keys, ".self_attn.q_a_proj.weight"))
            kv_a = shard.get_tensor(
                _unique_suffix(keys, ".self_attn.kv_a_proj_with_mqa.weight")
            )
            fused_weight = torch.cat((q_a, kv_a), dim=0)
        kv_norm_weight = shard.get_tensor(
            _unique_suffix(keys, ".self_attn.kv_a_layernorm.weight")
        )

    assert fused_weight.shape == (2112, 2048)
    assert kv_norm_weight.shape == (512,)
    assert fused_weight.dtype == torch.bfloat16
    assert kv_norm_weight.dtype == torch.bfloat16
    return config, fused_weight.contiguous(), kv_norm_weight.contiguous()


def _build_real_joyai_attention_shell(
    config: dict,
    fused_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    device: torch.device,
):
    """Build the real replay-facing JoyAI ops without allocating the full model."""

    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.linear import ReplicatedLinear
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.layers.rotary_embedding import get_rope_wrapper
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.models.joyai_llm_flash_nextn import (
        JoyAIDenseNextNDecoderLayer,
        JoyAILLMFlashForCausalLMNextN,
    )

    rope_scaling = copy.deepcopy(config.get("rope_scaling"))
    if rope_scaling:
        # This is the normalization performed by DeepseekV2AttentionMLA.__init__.
        rope_scaling["rope_type"] = "deepseek_yarn"

    # DefaultModelLoader constructs the model inside this same dtype/device
    # scope.  In particular, it makes the RoPE cache on CUDA rather than
    # computing a subtly different CPU cache and copying it afterward.
    with set_default_torch_dtype(torch.bfloat16), device:
        projection = ReplicatedLinear(2048, 2112, bias=False)
        kv_norm = RMSNorm(512, eps=float(config["rms_norm_eps"]))
        rotary_emb = get_rope_wrapper(
            64,
            rotary_dim=64,
            max_position=int(config["max_position_embeddings"]),
            base=float(config["rope_theta"]),
            rope_scaling=rope_scaling,
            is_neox_style=not bool(config.get("rope_interleave", True)),
            dtype=torch.bfloat16,
            device=str(device),
        )

    projection.weight.data.copy_(fused_weight.to(device))
    kv_norm.weight.data.copy_(kv_norm_weight.to(device))

    attn_mqa = RadixAttention(
        num_heads=32,
        head_dim=576,
        scaling=(128 + 64) ** -0.5,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=512,
    )

    # Keep the exact production classes so DiagESMTPDraftKVReplay's
    # architecture gate and every replay-facing op are exercised.  Bypassing
    # their large constructors avoids allocating unrelated embedding/MLP/q_b
    # weights; the three tensors used here come from the real MTP shard.
    attention = DeepseekV2AttentionMLA.__new__(DeepseekV2AttentionMLA)
    torch.nn.Module.__init__(attention)
    attention.fused_qkv_a_proj_with_mqa = projection
    attention.kv_a_layernorm = kv_norm
    attention.rotary_emb = rotary_emb
    attention.attn_mqa = attn_mqa
    attention.q_lora_rank = 1536
    attention.kv_lora_rank = 512

    decoder = JoyAIDenseNextNDecoderLayer.__new__(JoyAIDenseNextNDecoderLayer)
    torch.nn.Module.__init__(decoder)
    decoder.self_attn = attention

    model = JoyAILLMFlashForCausalLMNextN.__new__(JoyAILLMFlashForCausalLMNextN)
    torch.nn.Module.__init__(model)
    model.quant_config = None
    model.config = SimpleNamespace(**config)
    model.model = torch.nn.Module()
    model.model.decoder = decoder
    return model, attention


def _new_mla_pool(device: torch.device):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return MLATokenToKVPool(
        size=128,
        page_size=64,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=1,
        device=str(device),
        enable_memory_saver=False,
    )


def _delta_bank(slots: int, width: int, device: torch.device, phase: int):
    values = torch.arange(slots * width, dtype=torch.int64, device=device)
    values = ((values + phase) % 29 - 14).to(torch.float32).mul_(2**-12)
    bank = values.view(slots, width).contiguous()
    bank[0].zero_()
    return bank


@torch.no_grad()
def _ordinary_fresh_mla_write(
    attention,
    pool,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    candidate_slots: torch.Tensor,
) -> None:
    """Independent copy of the ordinary fused projection-to-cache dataflow."""

    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    with forward_context(
        ForwardContext(attn_backend=None, es_candidate_slots=candidate_slots)
    ):
        qkv_latent = attention.fused_qkv_a_proj_with_mqa(hidden_states)[0]
    latent_cache = qkv_latent[:, attention.q_lora_rank :]
    k_nope = attention.kv_a_layernorm(
        latent_cache[:, : attention.kv_lora_rank]
    ).unsqueeze(1)
    k_pe = latent_cache[:, attention.kv_lora_rank :].unsqueeze(1)
    _, k_pe = attention.rotary_emb(positions, torch.zeros_like(k_pe), k_pe)
    pool.set_mla_kv_buffer(attention.attn_mqa, loc, k_nope, k_pe)


def _assert_bf16_bytes_equal(
    expected: torch.Tensor, actual: torch.Tensor, component: str
) -> None:
    assert expected.dtype == actual.dtype == torch.bfloat16
    expected_bytes = expected.contiguous().view(torch.uint8)
    actual_bytes = actual.contiguous().view(torch.uint8)
    assert torch.equal(expected_bytes, actual_bytes), (
        f"JoyAI replay {component} differs at BF16 byte level"
    )


@pytest.mark.parametrize("placement", ["pre", "post", "both"])
def test_real_joyai_mtp_replay_matches_fresh_mla_write(placement, monkeypatch):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("JoyAI MTP replay parity requires CUDA BF16 support")

    config, fused_weight, kv_norm_weight = _load_local_attention_artifact()
    assert int(config["max_position_embeddings"]) > 12289

    from sglang.srt.diag_es.mtp_kv_replay import DiagESMTPDraftKVReplay
    from sglang.srt.layers.quantization import unquant
    from sglang.srt.layers.quantization.unquant import Bf16GemmBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.runtime_context import get_context, get_parallel

    monkeypatch.setattr(unquant, "_BF16_GEMM_BACKEND", Bf16GemmBackend.TRITON)
    device = torch.device("cuda", torch.cuda.current_device())

    with (
        get_context().override_server_args(bf16_gemm_backend="triton"),
        get_parallel().override(
            dcp_enabled=False,
            attn_dcp_size=1,
            attn_dcp_rank=0,
        ),
    ):
        model, attention = _build_real_joyai_attention_shell(
            config, fused_weight, kv_norm_weight, device
        )
        projection = attention.fused_qkv_a_proj_with_mqa
        projection.es_pre_delta_bank = (
            _delta_bank(4, 2048, device, phase=3)
            if placement in ("pre", "both")
            else None
        )
        projection.es_post_delta_bank = (
            _delta_bank(4, 2112, device, phase=11)
            if placement in ("post", "both")
            else None
        )
        if projection.es_post_delta_bank is not None:
            # v2 steers the complete fused q_a+kv_a epilogue, including the
            # 576 values that persist in MLA KV.
            assert torch.count_nonzero(projection.es_post_delta_bank[:, 1536:]).item()

        reference_pool = _new_mla_pool(device)
        replay_pool = _new_mla_pool(device)
        target_pool = _new_mla_pool(device)
        replay = DiagESMTPDraftKVReplay(
            model=model,
            token_to_kv_pool=replay_pool,
            chunk_tokens=16,
        )

        loc = torch.tensor([5, 73, 18, 121, 42, 90], device=device, dtype=torch.int64)
        positions = torch.tensor(
            [4096, 17, 8191, 233, 12289, 1024],
            device=device,
            dtype=torch.int64,
        )
        candidate_slots = torch.tensor(
            [1, 3, 2, 1, 2, 3], device=device, dtype=torch.int32
        )
        input_values = torch.arange(
            loc.numel() * 2048, device=device, dtype=torch.int64
        )
        hidden_states = (
            ((input_values % 257) - 128)
            .to(torch.float32)
            .mul_(2**-7)
            .to(torch.bfloat16)
            .view(loc.numel(), 2048)
            .contiguous()
        )

        _ordinary_fresh_mla_write(
            attention,
            reference_pool,
            hidden_states,
            positions,
            loc,
            candidate_slots,
        )

        target_buffer = target_pool.get_key_buffer(0)
        sentinel = torch.arange(target_buffer.numel(), device=device, dtype=torch.int64)
        target_buffer.copy_(
            ((sentinel % 127) - 63)
            .to(torch.float32)
            .mul_(2**-3)
            .to(torch.bfloat16)
            .view_as(target_buffer)
        )
        target_before_replay = target_buffer.clone()
        assert target_buffer.data_ptr() != replay_pool.get_key_buffer(0).data_ptr()

        replay.capture(hidden_states, positions, ForwardMode.EXTEND, loc)
        _assert_bf16_bytes_equal(
            hidden_states,
            replay.activation_buffer.index_select(0, loc),
            "captured projection input",
        )
        assert torch.equal(replay.position_buffer.index_select(0, loc), positions)
        replay._replay_chunk(
            loc=loc,
            positions=replay.position_buffer.index_select(0, loc),
            candidate_slots=candidate_slots,
        )
        torch.cuda.synchronize(device)

        reference_rows = reference_pool.get_key_buffer(0).index_select(0, loc)
        replay_rows = replay_pool.get_key_buffer(0).index_select(0, loc)
        _assert_bf16_bytes_equal(
            reference_rows[..., :512], replay_rows[..., :512], "latent512"
        )
        _assert_bf16_bytes_equal(
            reference_rows[..., 512:], replay_rows[..., 512:], "rope64"
        )
        _assert_bf16_bytes_equal(
            target_before_replay, target_pool.get_key_buffer(0), "target KV pool"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
