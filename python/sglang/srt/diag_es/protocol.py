from __future__ import annotations

from typing import Mapping, Optional

import torch


def prepare_register_payload(
    dense_deltas: Mapping[str, torch.Tensor],
    expert_fc1_deltas: Optional[torch.Tensor] = None,
    expert_fc2_deltas: Optional[torch.Tensor] = None,
    effective_model_digest: Optional[str] = None,
    *,
    grouped_deltas: Optional[Mapping[str, torch.Tensor]] = None,
) -> tuple[list[tuple[str, torch.Tensor]], Optional[str]]:
    """Normalize generic and legacy Engine register arguments for serialization."""

    named_deltas = [
        (f"dense_delta:{site_id}", delta) for site_id, delta in dense_deltas.items()
    ]
    if grouped_deltas is not None:
        if expert_fc1_deltas is not None or expert_fc2_deltas is not None:
            raise ValueError(
                "grouped_deltas conflict with legacy expert delta arguments"
            )
        named_deltas.extend(
            (f"grouped_delta:{name}", delta) for name, delta in grouped_deltas.items()
        )
    elif expert_fc1_deltas is not None or expert_fc2_deltas is not None:
        if not torch.is_tensor(expert_fc1_deltas) or not torch.is_tensor(
            expert_fc2_deltas
        ):
            raise ValueError("legacy expert delta arguments must be provided together")
        named_deltas.extend(
            (
                ("expert_delta:moe_fc1", expert_fc1_deltas),
                ("expert_delta:moe_fc2", expert_fc2_deltas),
            )
        )
    if effective_model_digest is not None and not isinstance(
        effective_model_digest, str
    ):
        raise ValueError("effective_model_digest must be a string")
    return named_deltas, effective_model_digest
