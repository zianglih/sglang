from __future__ import annotations

from typing import Mapping, Optional

import torch


def prepare_register_payload(
    dense_deltas: Mapping[str, torch.Tensor],
    grouped_deltas: Optional[Mapping[str, torch.Tensor]] = None,
    effective_model_digest: Optional[str] = None,
) -> tuple[list[tuple[str, torch.Tensor]], Optional[str]]:
    """Normalize generic Engine register arguments for serialization."""

    named_deltas = [
        (f"dense_delta:{site_id}", delta) for site_id, delta in dense_deltas.items()
    ]
    if grouped_deltas is not None:
        named_deltas.extend(
            (f"grouped_delta:{name}", delta) for name, delta in grouped_deltas.items()
        )
    if effective_model_digest is not None and not isinstance(
        effective_model_digest, str
    ):
        raise ValueError("effective_model_digest must be a string")
    return named_deltas, effective_model_digest
