from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping

import torch

QWEN3_30B_A3B_SCHEMA_ID = "qwen3-30b-a3b-diag-es-v2"
DiagESPlacement = Literal["pre", "post", "both"]

_SUPPORTED_PLACEMENTS = frozenset(("pre", "post", "both"))


@dataclass(frozen=True, slots=True)
class DenseSite:
    site_id: str
    width: int


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


def _require_bf16_contiguous_tensor(
    tensor: object, *, path: str, shape: tuple[int, ...]
) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{path} must be a tensor")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{path} must be bfloat16")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{path} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{path} must be contiguous")


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
    if getattr(model, "quant_config", None) is not None:
        raise ValueError("Qwen3 diagonal ES requires unquantized BF16 weights")
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

        for tensor, path, shape in (
            (qkv.weight, "self_attn.qkv_proj.weight", (5120, 2048)),
            (out.weight, "self_attn.o_proj.weight", (2048, 4096)),
            (experts.w13_weight, "mlp.experts.w13_weight", (128, 1536, 2048)),
            (experts.w2_weight, "mlp.experts.w2_weight", (128, 2048, 768)),
        ):
            _require_bf16_contiguous_tensor(
                tensor, path=f"{layer_path}.{path}", shape=shape
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
