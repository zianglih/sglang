from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from sglang.srt.diag_es.manifest import (
    QWEN3_30B_A3B_SCHEMA_ID,
    DiagESManifest,
    DiagESPlacement,
    compute_effective_model_digest,
    register_qwen3_30b_a3b_manifest,
)


class DiagESCandidateError(RuntimeError):
    """Base class for request-admission candidate lifecycle errors."""


class DiagESCandidateNotFoundError(DiagESCandidateError):
    pass


class DiagESCandidateRetiringError(DiagESCandidateError):
    pass


class DiagESInvalidCandidateError(DiagESCandidateError):
    pass


class DiagESNotEnabledError(DiagESCandidateError):
    pass


@dataclass(slots=True)
class CandidateRecord:
    candidate_id: str
    resident_slot: int
    effective_model_digest: str
    refs: int = 0
    retiring: bool = False


@dataclass(frozen=True, slots=True)
class CandidateLease:
    candidate_id: str
    resident_slot: int
    effective_model_digest: str


def compose_diag_es_extra_key(
    existing_extra_key: str | None, effective_model_digest: str
) -> str:
    """Compose a cache namespace from the semantic effective-model identity."""

    existing = (existing_extra_key or "").encode()
    digest = hashlib.sha256()
    digest.update(len(existing).to_bytes(8, "little"))
    digest.update(existing)
    digest.update(b"diag-es-v1")
    digest.update(bytes.fromhex(effective_model_digest))
    return f"diag-es-v1:{digest.hexdigest()}"


class DiagESManager:
    """Fixed-address resident FP32 delta banks for one Qwen3 ES engine."""

    def __init__(
        self,
        *,
        manifest: DiagESManifest,
        resident_candidate_slots: int,
        model_artifact_id: str,
        device: torch.device,
    ) -> None:
        self.manifest = manifest
        self.model_artifact_id = model_artifact_id
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("diagonal ES requires a CUDA device")
        self.physical_slots = resident_candidate_slots + 1
        self._records: dict[str, CandidateRecord] = {}
        self._free_slots = list(range(1, self.physical_slots))
        self._slot_last_read_events: list[torch.cuda.Event | None] = [
            None for _ in range(self.physical_slots)
        ]
        self._upload_stream = torch.cuda.Stream(device=self.device)

        self._dense_delta_banks = {
            site.site_id: torch.zeros(
                (self.physical_slots, site.width),
                dtype=torch.float32,
                device=self.device,
            )
            for site in manifest.dense_sites
        }
        self._grouped_delta_banks = {
            name: torch.zeros(
                (*shape[:-1], self.physical_slots, shape[-1]),
                dtype=torch.float32,
                device=self.device,
            )
            for name, shape in manifest.grouped_delta_shapes.items()
        }

    def get_dense_delta_bank(self, site_id: str) -> torch.Tensor:
        return self._dense_delta_banks[site_id]

    def get_grouped_delta_bank(self, name: str) -> torch.Tensor:
        return self._grouped_delta_banks[name]

    @staticmethod
    def _validate_candidate_id(candidate_id: str) -> None:
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or "\0" in candidate_id
        ):
            raise DiagESInvalidCandidateError(
                "diagonal-ES candidate_id must be a non-empty string without NUL bytes"
            )

    @staticmethod
    def _validate_delta(
        *, name: str, delta: torch.Tensor, expected_shape: tuple[int, ...]
    ) -> None:
        if not torch.is_tensor(delta):
            raise TypeError(f"diagonal-ES delta {name!r} must be a tensor")
        if delta.device.type != "cpu":
            raise ValueError(f"diagonal-ES delta {name!r} must be on CPU")
        if delta.dtype != torch.float32:
            raise ValueError(f"diagonal-ES delta {name!r} must be float32")
        if not delta.is_contiguous():
            raise ValueError(f"diagonal-ES delta {name!r} must be contiguous")
        if tuple(delta.shape) != expected_shape:
            raise ValueError(
                f"diagonal-ES delta {name!r} has shape {tuple(delta.shape)}, "
                f"expected {expected_shape}"
            )
        if not bool(torch.isfinite(delta).all()):
            raise ValueError(
                f"diagonal-ES delta {name!r} must contain only finite values"
            )

    @staticmethod
    def _validate_exact_names(
        *, kind: str, actual: set[str], expected: set[str]
    ) -> None:
        if actual == expected:
            return
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"diagonal-ES {kind} names do not match the manifest: "
            f"missing={missing}, unexpected={unexpected}"
        )

    def _validate_payload(
        self,
        dense_deltas: Mapping[str, torch.Tensor],
        grouped_deltas: Mapping[str, torch.Tensor],
    ) -> None:
        expected_dense = {
            site.site_id: site.width for site in self.manifest.dense_sites
        }
        self._validate_exact_names(
            kind="dense delta",
            actual=set(dense_deltas),
            expected=set(expected_dense),
        )
        for site_id, width in expected_dense.items():
            self._validate_delta(
                name=site_id,
                delta=dense_deltas[site_id],
                expected_shape=(width,),
            )

        expected_grouped = dict(self.manifest.grouped_delta_shapes)
        self._validate_exact_names(
            kind="grouped delta",
            actual=set(grouped_deltas),
            expected=set(expected_grouped),
        )
        for name, shape in expected_grouped.items():
            self._validate_delta(
                name=name,
                delta=grouped_deltas[name],
                expected_shape=shape,
            )

    def _upload_candidate(
        self,
        *,
        slot: int,
        dense_deltas: Mapping[str, torch.Tensor],
        grouped_deltas: Mapping[str, torch.Tensor],
    ) -> None:
        with torch.cuda.stream(self._upload_stream):
            for site in self.manifest.dense_sites:
                self._dense_delta_banks[site.site_id][slot].copy_(
                    dense_deltas[site.site_id], non_blocking=True
                )
            for name, delta in grouped_deltas.items():
                self._grouped_delta_banks[name][..., slot, :].copy_(
                    delta, non_blocking=True
                )
        self._upload_stream.synchronize()

    def register_candidate(
        self,
        *,
        candidate_id: str,
        dense_deltas: Mapping[str, torch.Tensor],
        grouped_deltas: Mapping[str, torch.Tensor],
        effective_model_digest: str,
    ) -> dict[str, Any]:
        self._validate_payload(dense_deltas, grouped_deltas)
        actual_digest = compute_effective_model_digest(
            model_artifact_id=self.model_artifact_id,
            schema_id=self.manifest.schema_id,
            schema_digest=self.manifest.schema_digest,
            dense_deltas=dense_deltas,
            grouped_deltas=grouped_deltas,
        )
        if effective_model_digest != actual_digest:
            raise ValueError(
                "supplied effective_model_digest does not match the canonical "
                f"server digest: supplied={effective_model_digest}, "
                f"canonical={actual_digest}"
            )

        self._reclaim_retired()
        existing = self._records.get(candidate_id)
        if existing is not None:
            if existing.retiring:
                raise DiagESCandidateRetiringError(
                    f"diagonal-ES candidate {candidate_id!r} is retiring"
                )
            if existing.effective_model_digest != actual_digest:
                raise ValueError(
                    f"diagonal-ES candidate {candidate_id!r} is already registered "
                    "with a different effective model digest"
                )
            return self._record_status(existing)
        if not self._free_slots:
            raise RuntimeError(
                "diagonal-ES resident candidate capacity is exhausted; "
                "wait for a RETIRING candidate to reach FREE before registering"
            )
        slot = self._free_slots.pop(0)

        try:
            self._upload_candidate(
                slot=slot,
                dense_deltas=dense_deltas,
                grouped_deltas=grouped_deltas,
            )
        except BaseException:
            self._free_slots.append(slot)
            self._free_slots.sort()
            raise

        record = CandidateRecord(
            candidate_id=candidate_id,
            resident_slot=slot,
            effective_model_digest=actual_digest,
        )
        self._records[candidate_id] = record
        return self._record_status(record)

    def acquire(self, candidate_id: str) -> CandidateLease:
        self._validate_candidate_id(candidate_id)
        record = self._records.get(candidate_id)
        if record is None:
            raise DiagESCandidateNotFoundError(
                f"diagonal-ES candidate {candidate_id!r} is not registered"
            )
        if record.retiring:
            raise DiagESCandidateRetiringError(
                f"diagonal-ES candidate {candidate_id!r} is retiring"
            )
        record.refs += 1
        return CandidateLease(
            candidate_id=record.candidate_id,
            resident_slot=record.resident_slot,
            effective_model_digest=record.effective_model_digest,
        )

    def release(self, candidate_id: str) -> None:
        record = self._records.get(candidate_id)
        if record is None:
            raise DiagESCandidateNotFoundError(
                f"diagonal-ES candidate {candidate_id!r} is not registered"
            )
        if record.refs <= 0:
            raise DiagESCandidateError(
                f"diagonal-ES candidate {candidate_id!r} has no acquired reference"
            )
        record.refs -= 1
        self._reclaim_retired()

    def note_slots_read(self, resident_slots: list[int] | tuple[int, ...]) -> None:
        """Fence the last submitted forward that reads each resident slot."""

        stream = torch.cuda.current_stream(self.device)
        for slot in set(resident_slots):
            if slot == 0:
                continue
            event = self._slot_last_read_events[slot]
            if event is None:
                event = torch.cuda.Event(enable_timing=False)
                self._slot_last_read_events[slot] = event
            event.record(stream)

    def retire_candidate(self, candidate_id: str) -> dict[str, Any]:
        record = self._records.get(candidate_id)
        if record is None:
            raise DiagESCandidateNotFoundError(
                f"diagonal-ES candidate {candidate_id!r} is not registered"
            )
        record.retiring = True
        resident_slot = record.resident_slot
        self._reclaim_retired()
        record = self._records.get(candidate_id)
        if record is not None:
            return self._record_status(record)
        return self._retired_status(candidate_id, resident_slot)

    def status(self) -> dict[str, Any]:
        self._reclaim_retired()
        return {
            "schema_id": self.manifest.schema_id,
            "schema_digest": self.manifest.schema_digest,
            "placement": self.manifest.placement,
            "model_artifact_id": self.model_artifact_id,
            "physical_slots": self.physical_slots,
            "free_slots": list(self._free_slots),
            "dense_sites": {
                site.site_id: site.width for site in self.manifest.dense_sites
            },
            # External v2 registry ABI; keep this key stable for the wrapper.
            "grouped_gate_shapes": {
                name: list(shape)
                for name, shape in self.manifest.grouped_delta_shapes.items()
            },
            "candidates": {
                candidate_id: self._record_status(record)
                for candidate_id, record in self._records.items()
            },
        }

    def _reclaim_retired(self) -> None:
        reclaimed: list[tuple[str, int]] = []
        for candidate_id, record in self._records.items():
            if not record.retiring or record.refs:
                continue
            event = self._slot_last_read_events[record.resident_slot]
            if event is not None and not event.query():
                continue
            reclaimed.append((candidate_id, record.resident_slot))
        for candidate_id, slot in reclaimed:
            del self._records[candidate_id]
            self._slot_last_read_events[slot] = None
            self._free_slots.append(slot)
        if reclaimed:
            self._free_slots.sort()

    @staticmethod
    def _retired_status(candidate_id: str, resident_slot: int) -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "resident_slot": resident_slot,
            "state": "FREE",
        }

    @staticmethod
    def _record_status(record: CandidateRecord) -> dict[str, Any]:
        return {
            "candidate_id": record.candidate_id,
            "resident_slot": record.resident_slot,
            "effective_model_digest": record.effective_model_digest,
            "refs": record.refs,
            "state": (
                "RETIRING" if record.retiring else "ACTIVE" if record.refs else "READY"
            ),
        }


_manager: DiagESManager | None = None


def register_diag_es_model(
    model: torch.nn.Module,
    *,
    schema_id: str,
    resident_candidate_slots: int,
    model_artifact_id: str,
    placement: DiagESPlacement,
) -> DiagESManager:
    global _manager
    if schema_id != QWEN3_30B_A3B_SCHEMA_ID:
        raise ValueError(f"unsupported diagonal-ES schema ID: {schema_id!r}")
    manifest = register_qwen3_30b_a3b_manifest(model, placement=placement)
    manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=resident_candidate_slots,
        model_artifact_id=model_artifact_id,
        device=next(model.parameters()).device,
    )

    from sglang.srt.diag_es.moe_ops import MoeDeltaBanks

    grouped = manifest.grouped_delta_shapes
    for layer_id, decoder_layer in enumerate(model.model.layers):
        for linear in (
            decoder_layer.self_attn.qkv_proj,
            decoder_layer.self_attn.o_proj,
        ):
            pre_site_id = linear.es_pre_site_id
            post_site_id = linear.es_post_site_id
            linear.es_pre_delta_bank = (
                manager.get_dense_delta_bank(pre_site_id)
                if pre_site_id is not None
                else None
            )
            linear.es_post_delta_bank = (
                manager.get_dense_delta_bank(post_site_id)
                if post_site_id is not None
                else None
            )

        def layer_bank(name: str) -> torch.Tensor | None:
            if name not in grouped:
                return None
            return manager.get_grouped_delta_bank(name)[layer_id]

        decoder_layer.mlp.experts.moe_runner_config.diag_es_delta_banks = MoeDeltaBanks(
            token_slots=None,
            fc1_pre=layer_bank("moe_fc1_pre"),
            fc1_post=layer_bank("moe_fc1_post"),
            fc2_pre=layer_bank("moe_fc2_pre"),
            fc2_post=layer_bank("moe_fc2_post"),
        )

    # Publish only after validation, allocation, and all hot-path bindings
    # succeed. A failed startup cannot expose a partially initialized manager.
    _manager = manager
    return manager


def get_diag_es_manager() -> DiagESManager:
    if _manager is None:
        raise DiagESNotEnabledError(
            "diagonal ES is not enabled or has not been initialized"
        )
    return _manager


def release_req_candidate(req: Any) -> None:
    if req.es_candidate_id is not None and not req.es_candidate_released:
        get_diag_es_manager().release(req.es_candidate_id)
        req.es_candidate_released = True
