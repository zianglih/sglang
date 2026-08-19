from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import torch

from sglang.srt.diag_es.manifest import (
    QWEN3_30B_A3B_SCHEMA_ID,
    QWEN2_5_1_5B_SCHEMA_ID,
    DiagESManifest,
    Qwen3DiagESManifest,
    compute_effective_model_digest,
    register_qwen2_5_1_5b_dense_sites,
    register_qwen3_30b_a3b_dense_sites,
)

ExpertGateKind = Literal["moe_fc1", "moe_fc2"]


@dataclass(slots=True)
class CandidateRecord:
    candidate_id: str
    resident_slot: int
    effective_model_digest: str
    refs: int = 0
    retiring: bool = False


def compose_diag_es_extra_key(
    existing_extra_key: Optional[str], effective_model_digest: str
) -> str:
    """Compose an unambiguous cache namespace without exposing slot identity."""

    existing = (existing_extra_key or "").encode()
    digest = hashlib.sha256()
    digest.update(len(existing).to_bytes(8, "little"))
    digest.update(existing)
    digest.update(b"diag-es-v1")
    digest.update(bytes.fromhex(effective_model_digest))
    return f"diag-es-v1:{digest.hexdigest()}"


class DiagESManager:
    """Fixed-address resident bank for the target diagonal-ES experiment."""

    def __init__(
        self,
        *,
        manifest: Qwen3DiagESManifest | DiagESManifest,
        resident_candidate_slots: int,
        device: torch.device,
        base_model_revision: Optional[str] = None,
        model_artifact_id: Optional[str] = None,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> None:
        self.manifest = manifest
        if (
            base_model_revision is not None
            and model_artifact_id is not None
            and base_model_revision != model_artifact_id
        ):
            raise ValueError(
                "base_model_revision conflicts with model_artifact_id"
            )
        self.model_artifact_id = (
            model_artifact_id
            if model_artifact_id is not None
            else base_model_revision
        )
        if (
            not isinstance(self.model_artifact_id, str)
            or not self.model_artifact_id.strip()
        ):
            raise ValueError("model_artifact_id must be a non-empty string")
        self.base_model_revision = base_model_revision
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.physical_slots = resident_candidate_slots + 1
        self._lock = threading.Lock()
        self._records: dict[str, CandidateRecord] = {}
        self._free_slots = list(range(1, self.physical_slots))
        # Re-recording a CUDA event moves the fence to the most recently
        # submitted read on that slot.  A retired slot is not returned to the
        # free list until this fence has completed, so registration never
        # overwrites a gate bank still consumed by an in-flight forward.
        self._slot_last_read_events: list[Optional[torch.cuda.Event]] = [
            None for _ in range(self.physical_slots)
        ]

        self._dense_delta_banks = {
            site.site_id: torch.zeros(
                (
                    self.physical_slots,
                    site.input_width // self.tp_size
                    if self.tp_size > 1 and ".o_proj.input" in site.site_id
                    else site.input_width,
                ),
                dtype=torch.float32,
                device=device,
            )
            for site in manifest.dense_sites
        }
        self._grouped_delta_banks = {
            name: torch.zeros(
                (
                    *shape[:-1],
                    self.physical_slots,
                    shape[-1] // self.tp_size
                    if name == "moe_fc2" and self.tp_size > 1
                    else shape[-1],
                ),
                dtype=torch.float32,
                device=device,
            )
            for name, shape in manifest.grouped_gate_shapes.items()
        }

    def get_dense_delta_bank(self, site_id: str) -> torch.Tensor:
        return self._dense_delta_banks[site_id]

    def _local_dense_delta(self, site_id: str, delta: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1 or ".o_proj.input" not in site_id:
            return delta
        assert delta.shape[0] % self.tp_size == 0
        width = delta.shape[0] // self.tp_size
        start = self.tp_rank * width
        return delta[start : start + width].contiguous()

    def get_expert_delta_bank(self, layer_id: int, kind: ExpertGateKind) -> torch.Tensor:
        return self._grouped_delta_banks[kind][layer_id]

    def get_grouped_delta_bank(self, name: str) -> torch.Tensor:
        return self._grouped_delta_banks[name]

    def _local_grouped_delta(self, name: str, delta: torch.Tensor) -> torch.Tensor:
        if name != "moe_fc2" or self.tp_size == 1:
            return delta
        assert delta.shape[-1] % self.tp_size == 0
        width = delta.shape[-1] // self.tp_size
        start = self.tp_rank * width
        return delta[..., start : start + width].contiguous()

    def register_candidate(
        self,
        *,
        candidate_id: str,
        dense_deltas: Mapping[str, torch.Tensor],
        grouped_deltas: Optional[Mapping[str, torch.Tensor]] = None,
        expert_fc1_deltas: Optional[torch.Tensor] = None,
        expert_fc2_deltas: Optional[torch.Tensor] = None,
        effective_model_digest: Optional[str] = None,
    ) -> dict[str, Any]:
        if grouped_deltas is not None and (
            expert_fc1_deltas is not None or expert_fc2_deltas is not None
        ):
            raise ValueError("grouped_deltas conflict with legacy expert delta arguments")
        if (expert_fc1_deltas is None) != (expert_fc2_deltas is None):
            raise ValueError("legacy expert delta arguments must be provided together")
        if expert_fc1_deltas is not None:
            grouped_deltas = {
                "moe_fc1": expert_fc1_deltas,
                "moe_fc2": expert_fc2_deltas,
            }
        grouped_deltas = dict(grouped_deltas or {})
        expected_dense_sites = {
            site.site_id: site.input_width for site in self.manifest.dense_sites
        }
        assert set(dense_deltas) == set(expected_dense_sites)
        for site_id, width in expected_dense_sites.items():
            delta = dense_deltas[site_id]
            assert delta.device.type == "cpu"
            assert delta.dtype == torch.float32
            assert delta.is_contiguous()
            assert tuple(delta.shape) == (width,)
        assert set(grouped_deltas) == set(self.manifest.grouped_gate_shapes)
        for name, shape in self.manifest.grouped_gate_shapes.items():
            delta = grouped_deltas[name]
            assert delta.device.type == "cpu"
            assert delta.dtype == torch.float32
            assert delta.is_contiguous()
            assert tuple(delta.shape) == tuple(shape)

        # The optional caller value is audit metadata only.  The server owns
        # the cache identity and always derives it from the FP32 delta payload
        # that passed the exact target-shape contract above.
        _ = effective_model_digest
        actual_digest = compute_effective_model_digest(
            model_artifact_id=self.model_artifact_id,
            schema_id=self.manifest.schema_id,
            schema_digest=self.manifest.schema_digest,
            dense_deltas=dense_deltas,
            grouped_deltas=grouped_deltas,
        )
        with self._lock:
            self._reclaim_retired_locked()
            existing = self._records.get(candidate_id)
            if existing is not None:
                assert existing.effective_model_digest == actual_digest
                return self._record_status(existing)
            if not self._free_slots:
                raise RuntimeError(
                    "diagonal-ES resident candidate capacity is exhausted; "
                    "wait for a RETIRING candidate to reach FREE before registering"
                )
            slot = self._free_slots.pop(0)

        stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(stream):
            for site in self.manifest.dense_sites:
                self._dense_delta_banks[site.site_id][slot].copy_(
                    self._local_dense_delta(
                        site.site_id, dense_deltas[site.site_id]
                    ),
                    non_blocking=True,
                )
            for name, delta in grouped_deltas.items():
                self._grouped_delta_banks[name][..., slot, :].copy_(
                    self._local_grouped_delta(name, delta), non_blocking=True
                )
        stream.synchronize()

        record = CandidateRecord(
            candidate_id=candidate_id,
            resident_slot=slot,
            effective_model_digest=actual_digest,
        )
        with self._lock:
            self._records[candidate_id] = record
        return self._record_status(record)

    def acquire(self, candidate_id: str) -> CandidateRecord:
        with self._lock:
            record = self._records[candidate_id]
            assert not record.retiring
            record.refs += 1
            return CandidateRecord(
                candidate_id=record.candidate_id,
                resident_slot=record.resident_slot,
                effective_model_digest=record.effective_model_digest,
                refs=record.refs,
                retiring=record.retiring,
            )

    def release(self, candidate_id: str) -> None:
        with self._lock:
            record = self._records[candidate_id]
            assert record.refs > 0
            record.refs -= 1
            self._reclaim_retired_locked()

    def note_slots_read(self, resident_slots: list[int] | tuple[int, ...]) -> None:
        """Fence the last submitted forward that reads each resident slot."""

        stream = torch.cuda.current_stream(self.device)
        with self._lock:
            for slot in set(resident_slots):
                if slot == 0:
                    continue
                event = self._slot_last_read_events[slot]
                if event is None:
                    event = torch.cuda.Event(enable_timing=False)
                    self._slot_last_read_events[slot] = event
                event.record(stream)

    def retire_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records[candidate_id]
            record.retiring = True
            resident_slot = record.resident_slot
            self._reclaim_retired_locked()
            record = self._records.get(candidate_id)
            if record is not None:
                return self._record_status(record)
        return self._retired_status(candidate_id, resident_slot)

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reclaim_retired_locked()
            status = {
                "schema_id": self.manifest.schema_id,
                "schema_digest": self.manifest.schema_digest,
                "model_artifact_id": self.model_artifact_id,
                "physical_slots": self.physical_slots,
                "free_slots": list(self._free_slots),
                "dense_sites": {
                    site.site_id: site.input_width for site in self.manifest.dense_sites
                },
                "grouped_gate_shapes": {
                    name: list(shape)
                    for name, shape in self.manifest.grouped_gate_shapes.items()
                },
                "candidates": {
                    candidate_id: self._record_status(record)
                    for candidate_id, record in self._records.items()
                },
            }
            if self.manifest.schema_id == QWEN3_30B_A3B_SCHEMA_ID:
                status.update(
                    {
                        "base_model_revision": self.model_artifact_id,
                        "expert_fc1_shape": list(
                            self.manifest.grouped_gate_shapes["moe_fc1"]
                        ),
                        "expert_fc2_shape": list(
                            self.manifest.grouped_gate_shapes["moe_fc2"]
                        ),
                    }
                )
            return status

    def _reclaim_retired_locked(self) -> None:
        reclaimed = []
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


_manager: Optional[DiagESManager] = None


def register_qwen3_30b_a3b(
    model: torch.nn.Module,
    *,
    resident_candidate_slots: int,
    base_model_revision: str,
) -> DiagESManager:
    global _manager
    manifest = register_qwen3_30b_a3b_dense_sites(model)
    _manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=resident_candidate_slots,
        base_model_revision=base_model_revision,
        device=next(model.parameters()).device,
    )
    return _manager


def register_diag_es_model(
    model: torch.nn.Module,
    *,
    schema_id: str,
    resident_candidate_slots: int,
    model_artifact_id: str,
    tp_size: int,
) -> DiagESManager:
    global _manager
    from sglang.srt.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
    )

    if schema_id == QWEN3_30B_A3B_SCHEMA_ID:
        manifest = register_qwen3_30b_a3b_dense_sites(model)
    elif schema_id == QWEN2_5_1_5B_SCHEMA_ID:
        manifest = register_qwen2_5_1_5b_dense_sites(model, tp_size=tp_size)
    else:
        raise ValueError(f"unknown diagonal-ES schema ID: {schema_id!r}")
    _manager = DiagESManager(
        manifest=manifest,
        resident_candidate_slots=resident_candidate_slots,
        model_artifact_id=model_artifact_id,
        device=next(model.parameters()).device,
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=tp_size,
    )
    return _manager


def has_diag_es_manager() -> bool:
    return _manager is not None


def get_diag_es_manager() -> DiagESManager:
    assert _manager is not None
    return _manager


def get_expert_delta_bank(layer_id: int, kind: ExpertGateKind) -> torch.Tensor:
    return get_diag_es_manager().get_expert_delta_bank(layer_id, kind)


def get_grouped_delta_bank(name: str) -> torch.Tensor:
    return get_diag_es_manager().get_grouped_delta_bank(name)


def release_req_candidate(req: Any) -> None:
    candidate_id = getattr(req, "es_candidate_id", None)
    if candidate_id is not None and not req.es_candidate_released:
        get_diag_es_manager().release(candidate_id)
        req.es_candidate_released = True
