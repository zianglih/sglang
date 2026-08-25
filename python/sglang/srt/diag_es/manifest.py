from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping

import torch

QWEN3_30B_A3B_SCHEMA_ID = "qwen3-30b-a3b-diag-es-v2"
JOYAI_LLM_FLASH_MTP_SCHEMA_ID = "joyai-llm-flash-mtp-diag-es-v1"
DiagESPlacement = Literal["pre", "post", "both"]

_SUPPORTED_PLACEMENTS = frozenset(("pre", "post", "both"))


@dataclass(frozen=True, slots=True)
class DenseSite:
    site_id: str
    width: int
    # Some fused GEMMs contain a KV-bearing suffix. The physical delta bank
    # still spans the full epilogue width, while entries outside active_width
    # are fixed to exact zero and therefore leave that suffix bit-identical.
    active_width: int | None = None


@dataclass(frozen=True, slots=True)
class DiagESManifest:
    schema_id: str
    placement: DiagESPlacement
    dense_sites: tuple[DenseSite, ...]
    grouped_delta_shapes: Mapping[str, tuple[int, ...]]
    schema_digest: str


def _grouped_delta_shapes(
    placement: DiagESPlacement,
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    if placement in ("pre", "both"):
        shapes.update(
            {
                "moe_fc1_pre": (num_layers, num_experts, hidden_size),
                "moe_fc2_pre": (num_layers, num_experts, moe_intermediate_size),
            }
        )
    if placement in ("post", "both"):
        shapes.update(
            {
                "moe_fc1_post": (
                    num_layers,
                    num_experts,
                    2 * moe_intermediate_size,
                ),
                "moe_fc2_post": (num_layers, num_experts, hidden_size),
            }
        )
    return shapes


def _schema_digest(
    dense_sites: tuple[DenseSite, ...],
    *,
    placement: DiagESPlacement,
    grouped_delta_shapes: Mapping[str, tuple[int, ...]],
) -> str:
    # Keep the external grouped_gate_shapes spelling as part of the v2 schema
    # codec. Renaming this JSON field would silently change every persisted
    # schema digest and wrapper checkpoint identity.
    payload = {
        "version": QWEN3_30B_A3B_SCHEMA_ID,
        "placement": placement,
        "dense_sites": [(site.site_id, site.width) for site in dense_sites],
        "grouped_gate_shapes": dict(grouped_delta_shapes),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_attr_value(obj: object, name: str, expected: int) -> None:
    actual = getattr(obj, name, None)
    if actual != expected:
        raise ValueError(f"Qwen3 diagonal ES requires {name}={expected}, got {actual}")


def _require_contiguous_tensor(
    tensor: object,
    *,
    path: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{path} must be a tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{path} must have dtype {dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{path} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{path} must be contiguous")


def _is_deepseek_block_fp8_model(
    model: torch.nn.Module, *, placement: DiagESPlacement
) -> bool:
    quant_config = getattr(model, "quant_config", None)
    if quant_config is None:
        return False
    if placement != "post":
        raise ValueError("block-FP8 diagonal ES supports post placement only")
    expected = {
        "is_checkpoint_fp8_serialized": True,
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "use_mxfp8": False,
        "is_fp4_experts": False,
    }
    mismatches = [
        f"{name}={getattr(quant_config, name, None)!r} (requires {value!r})"
        for name, value in expected.items()
        if getattr(quant_config, name, None) != value
    ]
    if quant_config.__class__.__name__ != "Fp8Config":
        mismatches.append(
            f"quant_config={quant_config.__class__.__name__!r} (requires 'Fp8Config')"
        )
    if mismatches:
        raise ValueError(
            "block-FP8 diagonal ES requires serialized DeepSeek-style E4M3 "
            "weights with FP32 128x128 scales and dynamic FP32 1x128 "
            "activation scales: "
            + ", ".join(mismatches)
        )
    return True


def register_qwen3_30b_a3b_manifest(
    model: torch.nn.Module,
    *,
    placement: DiagESPlacement,
) -> DiagESManifest:
    """Validate and register the exact Qwen3-30B-A3B diagonal-ES sites."""

    if placement not in _SUPPORTED_PLACEMENTS:
        raise ValueError("diagonal-ES placement must be pre, post, or both")
    if model.__class__.__name__ != "Qwen3MoeForCausalLM":
        raise TypeError("Qwen3 diagonal ES requires Qwen3MoeForCausalLM")
    block_fp8 = _is_deepseek_block_fp8_model(model, placement=placement)
    if not hasattr(model, "config") or not hasattr(model, "model"):
        raise TypeError("Qwen3 diagonal ES requires a loaded Qwen3 MoE model")

    config = model.config
    for name, expected in (
        ("num_hidden_layers", 48),
        ("hidden_size", 2048),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 4),
        ("head_dim", 128),
        ("num_experts", 128),
        ("moe_intermediate_size", 768),
        ("num_experts_per_tok", 8),
    ):
        _require_attr_value(config, name, expected)

    layers = getattr(model.model, "layers", None)
    if layers is None or len(layers) != 48:
        actual = None if layers is None else len(layers)
        raise ValueError(f"Qwen3 diagonal ES requires 48 layers, got {actual}")

    dense_sites: list[DenseSite] = []
    for layer_id, decoder_layer in enumerate(layers):
        layer_path = f"model.layers.{layer_id}"
        try:
            qkv = decoder_layer.self_attn.qkv_proj
            out = decoder_layer.self_attn.o_proj
            experts = decoder_layer.mlp.experts
            runner_config = experts.moe_runner_config
        except AttributeError as exc:
            raise TypeError(
                f"{layer_path} does not match the Qwen3-30B-A3B layer layout"
            ) from exc

        weight_dtype = torch.float8_e4m3fn if block_fp8 else torch.bfloat16
        for tensor, path, shape in (
            (qkv.weight, "self_attn.qkv_proj.weight", (5120, 2048)),
            (out.weight, "self_attn.o_proj.weight", (2048, 4096)),
            (experts.w13_weight, "mlp.experts.w13_weight", (128, 1536, 2048)),
            (experts.w2_weight, "mlp.experts.w2_weight", (128, 2048, 768)),
        ):
            _require_contiguous_tensor(
                tensor,
                path=f"{layer_path}.{path}",
                shape=shape,
                dtype=weight_dtype,
            )
        if block_fp8:
            for tensor, path, shape in (
                (
                    qkv.weight_scale_inv,
                    "self_attn.qkv_proj.weight_scale_inv",
                    (40, 16),
                ),
                (
                    out.weight_scale_inv,
                    "self_attn.o_proj.weight_scale_inv",
                    (16, 32),
                ),
                (
                    experts.w13_weight_scale_inv,
                    "mlp.experts.w13_weight_scale_inv",
                    (128, 12, 16),
                ),
                (
                    experts.w2_weight_scale_inv,
                    "mlp.experts.w2_weight_scale_inv",
                    (128, 16, 6),
                ),
            ):
                _require_contiguous_tensor(
                    tensor,
                    path=f"{layer_path}.{path}",
                    shape=shape,
                    dtype=torch.float32,
                )
            if any(
                value is not None
                for value in (
                    getattr(qkv, "input_scale", None),
                    getattr(out, "input_scale", None),
                    getattr(experts, "w13_input_scale", None),
                    getattr(experts, "w2_input_scale", None),
                )
            ):
                raise ValueError(
                    f"{layer_path} block-FP8 activations must use dynamic "
                    "per-token-group scales"
                )
        if getattr(runner_config, "params_dtype", None) != torch.bfloat16:
            raise ValueError(
                f"{layer_path}.mlp.experts runner params_dtype must be bfloat16"
            )

        for linear, path, input_width, output_width in (
            (qkv, "self_attn.qkv_proj", 2048, 5120),
            (out, "self_attn.o_proj", 4096, 2048),
        ):
            linear.es_pre_site_id = None
            linear.es_post_site_id = None
            if placement in ("pre", "both"):
                site_id = f"{layer_path}.{path}.input"
                linear.es_pre_site_id = site_id
                dense_sites.append(DenseSite(site_id, input_width))
            if placement in ("post", "both"):
                site_id = f"{layer_path}.{path}.output"
                linear.es_post_site_id = site_id
                dense_sites.append(DenseSite(site_id, output_width))

    dense_sites_tuple = tuple(dense_sites)
    grouped_shapes = _grouped_delta_shapes(
        placement,
        num_layers=48,
        num_experts=128,
        hidden_size=2048,
        moe_intermediate_size=768,
    )
    return DiagESManifest(
        schema_id=QWEN3_30B_A3B_SCHEMA_ID,
        placement=placement,
        dense_sites=dense_sites_tuple,
        grouped_delta_shapes=grouped_shapes,
        schema_digest=_schema_digest(
            dense_sites_tuple,
            placement=placement,
            grouped_delta_shapes=grouped_shapes,
        ),
    )


def _joyai_mtp_schema_digest(
    dense_sites: tuple[DenseSite, ...],
) -> str:
    payload = {
        "version": JOYAI_LLM_FLASH_MTP_SCHEMA_ID,
        "placement": "post",
        "kv_contract": "target-and-direct-draft-kv-neutral-v1",
        "dense_sites": [
            (site.site_id, site.width, site.active_width)
            for site in dense_sites
        ],
        "excluded_sites": [
            "model.eh_proj.output",
            "model.decoder.self_attn.kv_a_proj_with_mqa.output",
            "model.decoder.self_attn.kv_b_proj.output",
            "model.shared_head.head.output",
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def register_joyai_llm_flash_mtp_manifest(
    model: torch.nn.Module,
    *,
    placement: DiagESPlacement,
) -> DiagESManifest:
    """Register the KV-neutral post-only search space for JoyAI's MTP layer.

    The q_a projection is fused with kv_a in SGLang. Its bank therefore spans
    the physical 2112-wide epilogue, but only the leading 1536 q_a values are
    trainable; the 576-wide kv_a suffix is kept at an exact zero delta.
    """

    if placement != "post":
        raise ValueError("JoyAI MTP diagonal ES supports post placement only")
    if model.__class__.__name__ != "JoyAILLMFlashForCausalLMNextN":
        raise TypeError(
            "JoyAI MTP diagonal ES requires JoyAILLMFlashForCausalLMNextN"
        )
    if getattr(model, "quant_config", None) is not None:
        raise ValueError("JoyAI MTP diagonal ES requires an unquantized BF16 draft")
    if not hasattr(model, "config") or not hasattr(model, "model"):
        raise TypeError("JoyAI MTP diagonal ES requires a loaded NextN model")

    config = model.config
    for name, expected in (
        ("num_hidden_layers", 40),
        ("hidden_size", 2048),
        ("num_attention_heads", 32),
        ("q_lora_rank", 1536),
        ("kv_lora_rank", 512),
        ("qk_nope_head_dim", 128),
        ("qk_rope_head_dim", 64),
        ("v_head_dim", 128),
        ("intermediate_size", 7168),
        ("num_nextn_predict_layers", 1),
    ):
        _require_attr_value(config, name, expected)

    try:
        decoder = model.model.decoder
        attention = decoder.self_attn
        mlp = decoder.mlp
        fused_q_a_kv_a = attention.fused_qkv_a_proj_with_mqa
        q_b = attention.q_b_proj
        kv_b = attention.kv_b_proj
        out = attention.o_proj
        gate_up = mlp.gate_up_proj
        down = mlp.down_proj
    except AttributeError as exc:
        raise TypeError(
            "JoyAI MTP diagonal ES requires the JoyAI dense NextN decoder layout"
        ) from exc

    if decoder.__class__.__name__ != "JoyAIDenseNextNDecoderLayer":
        raise TypeError(
            "JoyAI MTP diagonal ES requires JoyAIDenseNextNDecoderLayer"
        )
    if mlp.__class__.__name__ != "DeepseekV2MLP":
        raise TypeError("JoyAI MTP diagonal ES requires a dense DeepseekV2MLP")

    for tensor, path, shape in (
        (
            fused_q_a_kv_a.weight,
            "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.weight",
            (2112, 2048),
        ),
        (
            q_b.weight,
            "model.decoder.self_attn.q_b_proj.weight",
            (6144, 1536),
        ),
        (
            kv_b.weight,
            "model.decoder.self_attn.kv_b_proj.weight",
            (8192, 512),
        ),
        (
            out.weight,
            "model.decoder.self_attn.o_proj.weight",
            (2048, 4096),
        ),
        (
            gate_up.weight,
            "model.decoder.mlp.gate_up_proj.weight",
            (14336, 2048),
        ),
        (
            down.weight,
            "model.decoder.mlp.down_proj.weight",
            (2048, 7168),
        ),
    ):
        _require_contiguous_tensor(
            tensor,
            path=path,
            shape=shape,
            dtype=torch.bfloat16,
        )
    # Reset all relevant sites explicitly so a reused model object cannot carry
    # stale bindings from a failed or earlier registration.
    for linear in (fused_q_a_kv_a, q_b, kv_b, out, gate_up, down):
        linear.es_pre_site_id = None
        linear.es_post_site_id = None
        linear.es_pre_delta_bank = None
        linear.es_post_delta_bank = None

    # DeepSeek's min-latency fused-A helpers invoke GEMM directly and bypass
    # LinearBase.quant_method.apply, which owns the diagonal-ES epilogue. JoyAI
    # decode shapes are eligible at small M, so fail closed onto the regular
    # Triton linear path for both steered query projections at every batch size.
    attention._use_min_latency_fused_a_gemm = False
    attention._use_min_latency_q_b_gemm = False

    q_a_site = "model.decoder.self_attn.q_a_proj.output"
    q_b_site = "model.decoder.self_attn.q_b_proj.output"
    out_site = "model.decoder.self_attn.o_proj.output"
    gate_up_site = "model.decoder.mlp.gate_up_proj.output"
    down_site = "model.decoder.mlp.down_proj.output"
    fused_q_a_kv_a.es_post_site_id = q_a_site
    q_b.es_post_site_id = q_b_site
    out.es_post_site_id = out_site
    gate_up.es_post_site_id = gate_up_site
    down.es_post_site_id = down_site
    # kv_b remains unbound. The kv_a suffix of the fused bank is fixed-zero by
    # active_width=1536 in the session manager.
    dense_sites = (
        DenseSite(q_a_site, 2112, active_width=1536),
        DenseSite(q_b_site, 6144),
        DenseSite(out_site, 2048),
        DenseSite(gate_up_site, 14336),
        DenseSite(down_site, 2048),
    )
    grouped_shapes: dict[str, tuple[int, ...]] = {}
    return DiagESManifest(
        schema_id=JOYAI_LLM_FLASH_MTP_SCHEMA_ID,
        placement="post",
        dense_sites=dense_sites,
        grouped_delta_shapes=grouped_shapes,
        schema_digest=_joyai_mtp_schema_digest(dense_sites),
    )


def _validate_digest_identity(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL bytes")


def validate_sha256_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must contain 64 lowercase hex characters")


def _validate_digest_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(name, str) or not name or "\0" in name:
        raise ValueError("diagonal-ES delta names must be non-empty strings")
    if not torch.is_tensor(tensor):
        raise TypeError(f"diagonal-ES delta {name!r} must be a tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"diagonal-ES delta {name!r} must be on CPU")
    if tensor.dtype != torch.float32:
        raise ValueError(f"diagonal-ES delta {name!r} must be float32")
    if not tensor.is_contiguous():
        raise ValueError(f"diagonal-ES delta {name!r} must be contiguous")
    return tensor.detach()


def compute_effective_model_digest(
    *,
    model_artifact_id: str,
    schema_id: str,
    schema_digest: str,
    dense_deltas: Mapping[str, torch.Tensor],
    grouped_deltas: Mapping[str, torch.Tensor],
) -> str:
    """Hash the exact CPU FP32 residual deltas used as the KV namespace."""

    _validate_digest_identity("model_artifact_id", model_artifact_id)
    if schema_id != QWEN3_30B_A3B_SCHEMA_ID:
        raise ValueError(f"unsupported diagonal-ES schema ID: {schema_id!r}")
    validate_sha256_digest("schema_digest", schema_digest)
    if not isinstance(dense_deltas, Mapping):
        raise TypeError("dense_deltas must be a mapping")
    if not isinstance(grouped_deltas, Mapping):
        raise TypeError("grouped_deltas must be a mapping")

    digest = hashlib.sha256()
    digest.update(b"diag-es-effective-model-fp32-delta-v4\0")
    digest.update(model_artifact_id.encode())
    digest.update(b"\0")
    digest.update(schema_id.encode())
    digest.update(b"\0")
    digest.update(schema_digest.encode())

    def update_tensor(name: str, tensor: torch.Tensor) -> None:
        value = _validate_digest_tensor(name, tensor)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0fp32-delta\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())

    for site_id in sorted(dense_deltas):
        update_tensor(f"dense:{site_id}", dense_deltas[site_id])
    for delta_name in sorted(grouped_deltas):
        update_tensor(f"grouped:{delta_name}", grouped_deltas[delta_name])
    return digest.hexdigest()
