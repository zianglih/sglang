from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class DenseSite:
    site_id: str
    input_width: int


@dataclass(frozen=True, slots=True)
class Qwen3DiagESManifest:
    dense_sites: tuple[DenseSite, ...]
    num_layers: int
    num_experts: int
    hidden_size: int
    moe_intermediate_size: int
    schema_digest: str


def _schema_digest(
    dense_sites: tuple[DenseSite, ...],
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
) -> str:
    payload = {
        "version": "qwen3-30b-a3b-diag-es-v1",
        "dense_sites": [(site.site_id, site.input_width) for site in dense_sites],
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden_size": hidden_size,
        "moe_intermediate_size": moe_intermediate_size,
        "expert_sites": ["moe_fc1", "moe_fc2"],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def register_qwen3_30b_a3b_dense_sites(model: torch.nn.Module) -> Qwen3DiagESManifest:
    """Attach stable diagonal-ES metadata to the target Qwen model.

    This is deliberately an exact model manifest: attention QKV and output
    projections are dense sites, expert FC1/FC2 are registered through their
    fused runner config, and the router and LM head remain outside the search
    space.
    """

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

        qkv_site_id = f"model.layers.{layer_id}.self_attn.qkv_proj.input"
        out_site_id = f"model.layers.{layer_id}.self_attn.o_proj.input"
        qkv.es_site_id = qkv_site_id
        qkv.es_site_width = qkv.input_size
        out.es_site_id = out_site_id
        out.es_site_width = out.input_size
        dense_sites.extend(
            (
                DenseSite(qkv_site_id, qkv.input_size),
                DenseSite(out_site_id, out.input_size),
            )
        )

        # The router is intentionally clean. Experts are stacked parameters,
        # so their semantic layer marker lives on the runner configuration.
        decoder_layer.mlp.gate.es_site_id = None
        decoder_layer.mlp.experts.moe_runner_config.es_layer_id = layer_id

    dense_sites_tuple = tuple(dense_sites)
    return Qwen3DiagESManifest(
        dense_sites=dense_sites_tuple,
        num_layers=len(layers),
        num_experts=config.num_experts,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        schema_digest=_schema_digest(
            dense_sites_tuple,
            num_layers=len(layers),
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
        ),
    )


def compute_effective_model_digest(
    *,
    base_model_revision: str,
    schema_digest: str,
    dense_gates: Mapping[str, torch.Tensor],
    expert_fc1_gates: torch.Tensor,
    expert_fc2_gates: torch.Tensor,
) -> str:
    """Hash the actual logical BF16 gate payload used as the KV namespace."""

    digest = hashlib.sha256()
    digest.update(b"diag-es-effective-model-v1\0")
    digest.update(base_model_revision.encode())
    digest.update(b"\0")
    digest.update(schema_digest.encode())

    def update_tensor(name: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode())
        digest.update(b"\0bf16\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())

    for site_id in sorted(dense_gates):
        update_tensor(f"dense:{site_id}", dense_gates[site_id])
    update_tensor("expert:moe_fc1", expert_fc1_gates)
    update_tensor("expert:moe_fc2", expert_fc2_gates)
    return digest.hexdigest()
