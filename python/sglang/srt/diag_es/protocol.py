from __future__ import annotations

from typing import Mapping, Optional

import torch


def prepare_register_payload(
    dense_gates: Mapping[str, torch.Tensor],
    expert_fc1_gates: Optional[torch.Tensor] = None,
    expert_fc2_gates: Optional[torch.Tensor] = None,
    effective_model_digest: Optional[str] = None,
    *,
    grouped_gates: Optional[Mapping[str, torch.Tensor]] = None,
) -> tuple[list[tuple[str, torch.Tensor]], Optional[str]]:
    """Normalize generic and legacy Engine register arguments for serialization."""

    named_gates = [
        (f"dense:{site_id}", gate) for site_id, gate in dense_gates.items()
    ]
    if grouped_gates is not None:
        if expert_fc1_gates is not None or expert_fc2_gates is not None:
            raise ValueError("grouped_gates conflict with legacy expert gate arguments")
        named_gates.extend(
            (f"grouped:{name}", gate) for name, gate in grouped_gates.items()
        )
    elif expert_fc1_gates is not None or expert_fc2_gates is not None:
        if not torch.is_tensor(expert_fc1_gates) or not torch.is_tensor(
            expert_fc2_gates
        ):
            raise ValueError("legacy expert gate arguments must be provided together")
        named_gates.extend(
            (
                ("expert:moe_fc1", expert_fc1_gates),
                ("expert:moe_fc2", expert_fc2_gates),
            )
        )
    if effective_model_digest is not None and not isinstance(
        effective_model_digest, str
    ):
        raise ValueError("effective_model_digest must be a string")
    return named_gates, effective_model_digest
