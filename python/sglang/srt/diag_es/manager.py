from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

import torch

from sglang.srt.diag_es.manifest import (
    Qwen3DiagESManifest,
    compute_effective_model_digest,
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
        manifest: Qwen3DiagESManifest,
        resident_candidate_slots: int,
        base_model_revision: str,
        device: torch.device,
    ) -> None:
        self.manifest = manifest
        self.base_model_revision = base_model_revision
        self.device = device
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

        self._dense_gate_banks = {
            site.site_id: torch.ones(
                (self.physical_slots, site.input_width),
                dtype=torch.bfloat16,
                device=device,
            )
            for site in manifest.dense_sites
        }
        self._expert_fc1_gate_bank = torch.ones(
            (
                manifest.num_layers,
                manifest.num_experts,
                self.physical_slots,
                manifest.hidden_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        self._expert_fc2_gate_bank = torch.ones(
            (
                manifest.num_layers,
                manifest.num_experts,
                self.physical_slots,
                manifest.moe_intermediate_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

    def get_dense_gate_bank(self, site_id: str) -> torch.Tensor:
        return self._dense_gate_banks[site_id]

    def get_expert_gate_bank(self, layer_id: int, kind: ExpertGateKind) -> torch.Tensor:
        bank = (
            self._expert_fc1_gate_bank
            if kind == "moe_fc1"
            else self._expert_fc2_gate_bank
        )
        return bank[layer_id]

    def register_candidate(
        self,
        *,
        candidate_id: str,
        dense_gates: Mapping[str, torch.Tensor],
        expert_fc1_gates: torch.Tensor,
        expert_fc2_gates: torch.Tensor,
        effective_model_digest: Optional[str] = None,
    ) -> dict[str, Any]:
        expected_dense_sites = {
            site.site_id: site.input_width for site in self.manifest.dense_sites
        }
        assert set(dense_gates) == set(expected_dense_sites)
        for site_id, width in expected_dense_sites.items():
            gate = dense_gates[site_id]
            assert gate.device.type == "cpu"
            assert gate.dtype == torch.bfloat16
            assert gate.is_contiguous()
            assert tuple(gate.shape) == (width,)
        assert expert_fc1_gates.device.type == "cpu"
        assert expert_fc1_gates.dtype == torch.bfloat16
        assert expert_fc1_gates.is_contiguous()
        assert tuple(expert_fc1_gates.shape) == (
            self.manifest.num_layers,
            self.manifest.num_experts,
            self.manifest.hidden_size,
        )
        assert expert_fc2_gates.device.type == "cpu"
        assert expert_fc2_gates.dtype == torch.bfloat16
        assert expert_fc2_gates.is_contiguous()
        assert tuple(expert_fc2_gates.shape) == (
            self.manifest.num_layers,
            self.manifest.num_experts,
            self.manifest.moe_intermediate_size,
        )

        # The optional caller value is audit metadata only.  The server owns
        # the cache identity and always derives it from the BF16 payload that
        # passed the exact target-shape contract above.
        _ = effective_model_digest
        actual_digest = compute_effective_model_digest(
            base_model_revision=self.base_model_revision,
            schema_digest=self.manifest.schema_digest,
            dense_gates=dense_gates,
            expert_fc1_gates=expert_fc1_gates,
            expert_fc2_gates=expert_fc2_gates,
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
                self._dense_gate_banks[site.site_id][slot].copy_(
                    dense_gates[site.site_id], non_blocking=True
                )
            self._expert_fc1_gate_bank[:, :, slot, :].copy_(
                expert_fc1_gates, non_blocking=True
            )
            self._expert_fc2_gate_bank[:, :, slot, :].copy_(
                expert_fc2_gates, non_blocking=True
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
            return {
                "schema_digest": self.manifest.schema_digest,
                "base_model_revision": self.base_model_revision,
                "physical_slots": self.physical_slots,
                "free_slots": list(self._free_slots),
                "dense_sites": {
                    site.site_id: site.input_width for site in self.manifest.dense_sites
                },
                "expert_fc1_shape": [
                    self.manifest.num_layers,
                    self.manifest.num_experts,
                    self.manifest.hidden_size,
                ],
                "expert_fc2_shape": [
                    self.manifest.num_layers,
                    self.manifest.num_experts,
                    self.manifest.moe_intermediate_size,
                ],
                "candidates": {
                    candidate_id: self._record_status(record)
                    for candidate_id, record in self._records.items()
                },
            }

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


def has_diag_es_manager() -> bool:
    return _manager is not None


def get_diag_es_manager() -> DiagESManager:
    assert _manager is not None
    return _manager


def get_expert_gate_bank(layer_id: int, kind: ExpertGateKind) -> torch.Tensor:
    return get_diag_es_manager().get_expert_gate_bank(layer_id, kind)


def release_req_candidate(req: Any) -> None:
    candidate_id = getattr(req, "es_candidate_id", None)
    if candidate_id is not None and not req.es_candidate_released:
        get_diag_es_manager().release(candidate_id)
        req.es_candidate_released = True
