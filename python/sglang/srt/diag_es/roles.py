from __future__ import annotations

from typing import Literal, Protocol, cast

from sglang.srt.diag_es.manifest import DiagESPlacement

DiagESRolePlacement = Literal["off", "pre", "post", "both"]


class _DiagESRoleArgs(Protocol):
    diag_es_target_placement: DiagESRolePlacement
    diag_es_mtp_placement: DiagESRolePlacement


def get_diag_es_placement(
    server_args: _DiagESRoleArgs, *, is_draft_worker: bool
) -> DiagESPlacement | None:
    """Return the placement owned by one model runner, or ``None`` if clean."""

    placement = (
        server_args.diag_es_mtp_placement
        if is_draft_worker
        else server_args.diag_es_target_placement
    )
    return None if placement == "off" else cast(DiagESPlacement, placement)


def is_diag_es_enabled(server_args: _DiagESRoleArgs, *, is_draft_worker: bool) -> bool:
    """Whether diagonal ES is enabled for one model-runner role."""

    return (
        get_diag_es_placement(server_args, is_draft_worker=is_draft_worker) is not None
    )
