from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping, Optional

import torch

QWEN3_30B_A3B_SCHEMA_ID = "qwen3-30b-a3b-diag-es-v2"
QWEN2_5_1_5B_SCHEMA_ID = "qwen2.5-1.5b-instruct-dense-diag-es-v1"

DiagESPlacement = Literal["pre", "post", "both"]


@dataclass(frozen=True, slots=True)
class DenseSite:
    site_id: str
    width: int
    position: Literal["pre", "post"] = "pre"

    @property
    def input_width(self) -> int:
        """Compatibility name for the input-only Qwen2 manifest."""

        return self.width


@dataclass(frozen=True, slots=True)
class Qwen3DiagESManifest:
    dense_sites: tuple[DenseSite, ...]
    num_layers: int
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    schema_digest: str
    placement: DiagESPlacement = "pre"

    @property
    def schema_id(self) -> str:
        return QWEN3_30B_A3B_SCHEMA_ID

    @property
    def grouped_gate_shapes(self) -> Mapping[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        if self.placement in ("pre", "both"):
            shapes.update(
                {
                    "moe_fc1_pre": (
                        self.num_layers,
                        self.num_experts,
                        self.hidden_size,
                    ),
                    "moe_fc2_pre": (
                        self.num_layers,
                        self.num_experts,
                        self.moe_intermediate_size,
                    ),
                }
            )
        if self.placement in ("post", "both"):
            shapes.update(
                {
                    "moe_fc1_post": (
                        self.num_layers,
                        self.num_experts,
                        2 * self.moe_intermediate_size,
                    ),
                    "moe_fc2_post": (
                        self.num_layers,
                        self.num_experts,
                        self.hidden_size,
                    ),
                }
            )
        return shapes


@dataclass(frozen=True, slots=True)
class DiagESManifest:
    schema_id: str
    dense_sites: tuple[DenseSite, ...]
    grouped_gate_shapes: Mapping[str, tuple[int, ...]]
    schema_digest: str
    placement: DiagESPlacement = "pre"


def _schema_digest(
    dense_sites: tuple[DenseSite, ...],
    *,
    placement: DiagESPlacement,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
) -> str:
    grouped_gate_shapes: dict[str, tuple[int, ...]] = {}
    if placement in ("pre", "both"):
        grouped_gate_shapes.update(
            {
                "moe_fc1_pre": (num_layers, num_experts, hidden_size),
                "moe_fc2_pre": (
                    num_layers,
                    num_experts,
                    moe_intermediate_size,
                ),
            }
        )
    if placement in ("post", "both"):
        grouped_gate_shapes.update(
            {
                "moe_fc1_post": (
                    num_layers,
                    num_experts,
                    2 * moe_intermediate_size,
                ),
                "moe_fc2_post": (num_layers, num_experts, hidden_size),
            }
        )
    payload = {
        "version": QWEN3_30B_A3B_SCHEMA_ID,
        "placement": placement,
        "dense_sites": [(site.site_id, site.width) for site in dense_sites],
        "grouped_gate_shapes": grouped_gate_shapes,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def register_qwen3_30b_a3b_dense_sites(
    model: torch.nn.Module,
    *,
    placement: DiagESPlacement = "pre",
    tp_size: int = 1,
) -> Qwen3DiagESManifest:
    """Attach stable diagonal-ES metadata to the target Qwen model.

    This is deliberately an exact model manifest: attention QKV and output
    projections are dense sites, expert FC1/FC2 are registered through their
    fused runner config, and the router and LM head remain outside the search
    space.
    """

    if placement not in ("pre", "post", "both"):
        raise ValueError("diagonal-ES placement must be pre, post, or both")
    if placement in ("post", "both") and tp_size != 1:
        raise ValueError("Qwen3 diagonal-ES post placement requires tp_size=1")

    config = model.config
    layers = model.model.layers
    assert len(layers) == 48
    assert config.hidden_size == 2048
    assert config.num_experts == 128
    assert config.moe_intermediate_size == 768

    dense_sites: list[DenseSite] = []
    for layer_id, decoder_layer in enumerate(layers):
        qkv = decoder_layer.self_attn.qkv_proj
        out = decoder_layer.self_attn.o_proj
        assert qkv.input_size == 2048 and qkv.output_size == 5120
        assert out.input_size == 4096 and out.output_size == 2048

        for linear, path in (
            (qkv, "self_attn.qkv_proj"),
            (out, "self_attn.o_proj"),
        ):
            linear.es_pre_site_id = None
            linear.es_pre_site_width = None
            linear.es_post_site_id = None
            linear.es_post_site_width = None
            if placement in ("pre", "both"):
                site_id = f"model.layers.{layer_id}.{path}.input"
                linear.es_pre_site_id = site_id
                linear.es_pre_site_width = linear.input_size
                # Keep the historical attributes as aliases for pre-only
                # consumers while the v2 runtime uses position-specific names.
                linear.es_site_id = site_id
                linear.es_site_width = linear.input_size
                dense_sites.append(DenseSite(site_id, linear.input_size, "pre"))
            else:
                linear.es_site_id = None
                linear.es_site_width = None
            if placement in ("post", "both"):
                site_id = f"model.layers.{layer_id}.{path}.output"
                linear.es_post_site_id = site_id
                linear.es_post_site_width = linear.output_size
                dense_sites.append(DenseSite(site_id, linear.output_size, "post"))

        # The router is intentionally clean. Experts are stacked parameters,
        # so their semantic layer marker lives on the runner configuration.
        router = decoder_layer.mlp.gate
        router.es_site_id = None
        router.es_site_width = None
        router.es_pre_site_id = None
        router.es_pre_site_width = None
        router.es_post_site_id = None
        router.es_post_site_width = None
        decoder_layer.mlp.experts.moe_runner_config.es_layer_id = layer_id

    dense_sites_tuple = tuple(dense_sites)
    return Qwen3DiagESManifest(
        dense_sites=dense_sites_tuple,
        num_layers=len(layers),
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        placement=placement,
        schema_digest=_schema_digest(
            dense_sites_tuple,
            placement=placement,
            num_layers=len(layers),
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
        ),
    )


def _qwen2_5_1_5b_schema_digest(dense_sites: tuple[DenseSite, ...]) -> str:
    payload = {
        "version": QWEN2_5_1_5B_SCHEMA_ID,
        "dense_sites": [(site.site_id, site.input_width) for site in dense_sites],
        "grouped_sites": [],
        "num_layers": 28,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def register_qwen2_5_1_5b_dense_sites(
    model: torch.nn.Module, *, tp_size: int
) -> DiagESManifest:
    """Attach the exact dense diagonal-ES manifest for Qwen2.5-1.5B."""

    if model.__class__.__name__ != "Qwen2ForCausalLM" or not hasattr(model, "model"):
        raise TypeError("Qwen2 diagonal ES requires Qwen2ForCausalLM")
    if tp_size != 1:
        raise ValueError("Qwen2 diagonal ES requires tp_size=1")
    if getattr(model, "quant_config", None) is not None:
        raise ValueError("Qwen2 diagonal ES requires unquantized linear weights")

    config = model.config
    expected = {
        "num_hidden_layers": 28,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
    }
    for name, value in expected.items():
        actual = getattr(config, name, None)
        if actual != value:
            raise ValueError(f"{name} must be {value}, got {actual}")
    configured_head_dim = getattr(config, "head_dim", None)
    if configured_head_dim is None:
        configured_head_dim = config.hidden_size // config.num_attention_heads
    if configured_head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {configured_head_dim}")
    architectures = getattr(config, "architectures", None)
    if architectures is not None and "Qwen2ForCausalLM" not in architectures:
        raise TypeError("Qwen2 diagonal ES requires Qwen2ForCausalLM config")

    layers = model.model.layers
    if len(layers) != 28:
        raise ValueError(f"num_hidden_layers must be 28, got {len(layers)}")
    dense_sites: list[DenseSite] = []
    for layer_id, decoder_layer in enumerate(layers):
        physical_sites = (
            (decoder_layer.self_attn.qkv_proj, "self_attn.qkv_proj", 1536),
            (decoder_layer.self_attn.o_proj, "self_attn.o_proj", 1536),
            (decoder_layer.mlp.gate_up_proj, "mlp.gate_up_proj", 1536),
            (decoder_layer.mlp.down_proj, "mlp.down_proj", 8960),
        )
        for layer, path, width in physical_sites:
            weight = getattr(layer, "weight", None)
            if weight is None or weight.dtype != torch.bfloat16:
                raise ValueError(
                    f"model.layers.{layer_id}.{path} weight must be bfloat16"
                )
            if getattr(layer, "input_size", None) != width:
                raise ValueError(
                    f"model.layers.{layer_id}.{path}.input width must be {width}"
                )
            site_id = f"model.layers.{layer_id}.{path}.input"
            layer.es_site_id = site_id
            layer.es_site_width = width
            layer.es_pre_site_id = site_id
            layer.es_pre_site_width = width
            layer.es_post_site_id = None
            layer.es_post_site_width = None
            dense_sites.append(DenseSite(site_id, width))

    dense_sites_tuple = tuple(dense_sites)
    return DiagESManifest(
        schema_id=QWEN2_5_1_5B_SCHEMA_ID,
        dense_sites=dense_sites_tuple,
        grouped_gate_shapes={},
        schema_digest=_qwen2_5_1_5b_schema_digest(dense_sites_tuple),
    )


def compute_effective_model_digest(
    *,
    base_model_revision: Optional[str] = None,
    model_artifact_id: Optional[str] = None,
    schema_id: Optional[str] = None,
    schema_digest: str,
    dense_deltas: Mapping[str, torch.Tensor],
    grouped_deltas: Optional[Mapping[str, torch.Tensor]] = None,
) -> str:
    """Hash the exact FP32 residual deltas used as the KV namespace."""

    grouped_deltas = dict(grouped_deltas or {})
    if (
        base_model_revision is not None
        and model_artifact_id is not None
        and base_model_revision != model_artifact_id
    ):
        raise ValueError("base_model_revision conflicts with model_artifact_id")
    artifact_id = (
        model_artifact_id if model_artifact_id is not None else base_model_revision
    )
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("model_artifact_id must be a non-empty string")
    digest = hashlib.sha256()
    digest.update(b"diag-es-effective-model-fp32-delta-v4\0")
    digest.update(artifact_id.encode())
    digest.update(b"\0")
    digest.update((schema_id or "").encode())
    digest.update(b"\0")
    digest.update(schema_digest.encode())

    def update_tensor(name: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
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
