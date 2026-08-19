from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

_DENSE_PREFIX = "dense_delta:"
_GROUPED_PREFIX = "grouped_delta:"


def validate_effective_model_digest(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(
            "effective_model_digest must contain 64 lowercase hex characters"
        )


def _validate_delta_name(name: str, *, kind: str) -> None:
    if not isinstance(name, str) or not name or "\0" in name:
        raise ValueError(
            f"diagonal-ES {kind} delta names must be non-empty strings without NUL bytes"
        )


def _validate_candidate_id(candidate_id: str | None) -> None:
    if (
        not isinstance(candidate_id, str)
        or not candidate_id.strip()
        or "\0" in candidate_id
    ):
        from sglang.srt.diag_es.manager import DiagESInvalidCandidateError

        raise DiagESInvalidCandidateError(
            "diagonal-ES candidate_id must be a non-empty string without NUL bytes"
        )


def validate_registry_request(
    *,
    action: str,
    candidate_id: str | None,
    effective_model_digest: str | None,
    serialized_deltas: list[bytes] | None,
) -> None:
    """Validate the complete registry control-message contract."""

    if action == "register":
        _validate_candidate_id(candidate_id)
        if (
            not isinstance(serialized_deltas, list)
            or not serialized_deltas
            or any(not isinstance(payload, bytes) for payload in serialized_deltas)
        ):
            raise ValueError("register requires serialized diagonal-ES deltas")
        if effective_model_digest is not None:
            validate_effective_model_digest(effective_model_digest)
        return
    if action == "retire":
        _validate_candidate_id(candidate_id)
        if effective_model_digest is not None or serialized_deltas is not None:
            raise ValueError("retire does not accept a digest or serialized deltas")
        return
    if action == "status":
        if (
            candidate_id is not None
            or effective_model_digest is not None
            or serialized_deltas is not None
        ):
            raise ValueError("status does not accept candidate or delta fields")
        return
    raise ValueError(f"unsupported diagonal-ES registry action: {action!r}")


def prepare_register_payload(
    dense_deltas: Mapping[str, torch.Tensor],
    grouped_deltas: Mapping[str, torch.Tensor] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Encode the exact named FP32-delta maps for Engine IPC serialization."""

    if not isinstance(dense_deltas, Mapping):
        raise TypeError("dense_deltas must be a mapping")
    if grouped_deltas is not None and not isinstance(grouped_deltas, Mapping):
        raise TypeError("grouped_deltas must be a mapping")
    named_deltas: list[tuple[str, torch.Tensor]] = []
    for site_id, delta in dense_deltas.items():
        _validate_delta_name(site_id, kind="dense")
        named_deltas.append((f"{_DENSE_PREFIX}{site_id}", delta))
    for name, delta in (grouped_deltas or {}).items():
        _validate_delta_name(name, kind="grouped")
        named_deltas.append((f"{_GROUPED_PREFIX}{name}", delta))
    return named_deltas


def parse_register_payload(
    named_deltas: Iterable[tuple[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Decode Engine IPC payloads without silently collapsing duplicate names."""

    dense_deltas: dict[str, torch.Tensor] = {}
    grouped_deltas: dict[str, torch.Tensor] = {}
    serialized_names: set[str] = set()
    for item in named_deltas:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "serialized diagonal-ES deltas must be (name, tensor) pairs"
            )
        serialized_name, delta = item
        if not isinstance(serialized_name, str):
            raise ValueError("serialized diagonal-ES delta names must be strings")
        if serialized_name in serialized_names:
            raise ValueError(
                f"duplicate serialized diagonal-ES delta name: {serialized_name!r}"
            )
        serialized_names.add(serialized_name)

        if serialized_name.startswith(_DENSE_PREFIX):
            name = serialized_name.removeprefix(_DENSE_PREFIX)
            _validate_delta_name(name, kind="dense")
            dense_deltas[name] = delta
        elif serialized_name.startswith(_GROUPED_PREFIX):
            name = serialized_name.removeprefix(_GROUPED_PREFIX)
            _validate_delta_name(name, kind="grouped")
            grouped_deltas[name] = delta
        else:
            raise ValueError(
                f"unknown serialized diagonal-ES delta prefix: {serialized_name!r}"
            )
    return dense_deltas, grouped_deltas
