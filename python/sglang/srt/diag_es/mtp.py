from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from sglang.srt.diag_es.manifest import DiagESManifest

MTP_RNG_VERSION = "numpy-philox-site-v1"
MTP_MAX_PENDING_EVENTS = 1_000_000


class DiagESMTPSessionError(RuntimeError):
    pass


class DiagESMTPUpdateRejected(DiagESMTPSessionError):
    def __init__(self, reasons: list[str], stats: dict[str, Any]) -> None:
        self.reasons = tuple(reasons)
        self.stats = stats
        super().__init__("MTP diagonal-ES update rejected: " + "; ".join(reasons))


def _stable_u64(*parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"diag-es-v1")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little", signed=False)


def mtp_candidate_seed(root_seed: int, theta_version: int, population_index: int) -> int:
    """Match the wrapper's one-sided diagonal-ES candidate-seed contract."""

    return _stable_u64("candidate", root_seed, theta_version, population_index)


def mtp_normal_for_site(
    seed: int, site_id: str, shape: tuple[int, ...]
) -> torch.Tensor:
    site_seed = _stable_u64(MTP_RNG_VERSION, seed, site_id)
    generator = np.random.Generator(np.random.Philox(site_seed))
    return torch.from_numpy(generator.standard_normal(shape, dtype=np.float32))


@dataclass(frozen=True, slots=True)
class DiagESMTPSessionConfig:
    seed: int
    population_size: int = 16
    sigma: float = 0.01
    learning_rate: float = 0.0
    attempts_per_candidate: int = 4
    estimator: str = "population_zscore"
    reward_zscore_epsilon: float = 1e-8
    max_update_rms_ratio: float = 10.0
    max_update_abs_max_ratio: float = 100.0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("MTP diagonal-ES seed must be an integer")
        if self.population_size < 2:
            raise ValueError("MTP diagonal-ES population_size must be at least 2")
        if self.attempts_per_candidate < 1:
            raise ValueError(
                "MTP diagonal-ES attempts_per_candidate must be positive"
            )
        if self.estimator != "population_zscore":
            raise ValueError(
                "MTP diagonal ES supports only estimator='population_zscore'"
            )
        for name, value, allow_zero in (
            ("sigma", self.sigma, False),
            ("learning_rate", self.learning_rate, True),
            ("reward_zscore_epsilon", self.reward_zscore_epsilon, False),
            ("max_update_rms_ratio", self.max_update_rms_ratio, False),
            ("max_update_abs_max_ratio", self.max_update_abs_max_ratio, False),
        ):
            if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
                relation = "non-negative" if allow_zero else "positive"
                raise ValueError(f"MTP diagonal-ES {name} must be finite and {relation}")


@dataclass(slots=True)
class _MTPSessionState:
    session_id: str
    resident_slot: int
    config: DiagESMTPSessionConfig
    theta_dense: dict[str, torch.Tensor]
    noise_sum_dense: dict[str, torch.Tensor]
    rewarded_noise_sum_dense: dict[str, torch.Tensor]
    current_noise_dense: dict[str, torch.Tensor] = field(default_factory=dict)
    candidate_rewards: list[float] = field(default_factory=list)
    candidate_accept_sum: int = 0
    candidate_attempts: int = 0
    theta_version: int = 0
    population_index: int = 0
    total_attempts: int = 0
    committed_updates: int = 0
    rejected_updates: int = 0
    latest_accept_length: int | None = None
    latest_accepted_drafts: int | None = None
    latest_attempt_theta_version: int | None = None
    latest_attempt_population_index: int | None = None
    latest_attempt_perturbation_seed: int | None = None
    active_rid: str | None = None
    last_released_rid: str | None = None

    @property
    def perturbation_seed(self) -> int:
        return mtp_candidate_seed(
            self.config.seed, self.theta_version, self.population_index
        )


def _zeros_like_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in values.items()}


class DiagESMTPSessionManager:
    """Session-local online one-sided ES over fixed-address MTP delta banks."""

    def __init__(
        self,
        *,
        manifest: DiagESManifest,
        max_sessions: int,
        model_artifact_id: str,
        device: torch.device,
        max_correct_drafts: int,
        max_pending_events: int = MTP_MAX_PENDING_EVENTS,
    ) -> None:
        self.manifest = manifest
        self.max_sessions = max_sessions
        self.model_artifact_id = model_artifact_id
        self.device = torch.device(device)
        self.max_correct_drafts = max_correct_drafts
        self.max_pending_events = max_pending_events
        if self.device.type != "cuda":
            raise ValueError("MTP diagonal ES requires a CUDA device")
        if max_sessions < 1:
            raise ValueError("MTP diagonal-ES max_sessions must be positive")
        if max_correct_drafts < 1:
            raise ValueError("MTP diagonal ES requires at least one drafted token")
        if max_pending_events < 1:
            raise ValueError("MTP diagonal ES max_pending_events must be positive")
        if manifest.grouped_delta_shapes:
            raise ValueError("JoyAI MTP diagonal ES supports dense sites only")

        self.physical_slots = max_sessions + 1
        self._free_slots = list(range(1, self.physical_slots))
        self._sessions: dict[str, _MTPSessionState] = {}
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
        self._events: list[dict[str, Any]] = []
        self._next_event_id = 1

    def get_dense_delta_bank(self, site_id: str) -> torch.Tensor:
        return self._dense_delta_banks[site_id]

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if (
            not isinstance(session_id, str)
            or not session_id.strip()
            or "\0" in session_id
        ):
            raise ValueError(
                "diag_es_mtp_session_id must be a non-empty string without NUL bytes"
            )

    def _new_tensor_map(self) -> dict[str, torch.Tensor]:
        return {
            site.site_id: torch.zeros(site.width, dtype=torch.float32)
            for site in self.manifest.dense_sites
        }

    def _candidate_noise(self, seed: int) -> dict[str, torch.Tensor]:
        dense: dict[str, torch.Tensor] = {}
        for site in self.manifest.dense_sites:
            active_width = (
                site.width if site.active_width is None else site.active_width
            )
            if active_width < 1 or active_width > site.width:
                raise ValueError(
                    f"invalid active width for MTP dense site {site.site_id!r}: "
                    f"{active_width} of {site.width}"
                )
            value = torch.zeros(site.width, dtype=torch.float32)
            value[:active_width].copy_(
                mtp_normal_for_site(seed, site.site_id, (active_width,))
            )
            dense[site.site_id] = value
        return dense

    def _wait_for_slot(self, slot: int) -> None:
        event = self._slot_last_read_events[slot]
        if event is not None:
            event.synchronize()

    def _upload_candidate(self, state: _MTPSessionState) -> None:
        self._wait_for_slot(state.resident_slot)
        seed = state.perturbation_seed
        noise_dense = self._candidate_noise(seed)
        state.current_noise_dense = noise_dense
        with torch.cuda.stream(self._upload_stream):
            for site in self.manifest.dense_sites:
                candidate = state.theta_dense[site.site_id].add(
                    noise_dense[site.site_id], alpha=state.config.sigma
                )
                self._dense_delta_banks[site.site_id][state.resident_slot].copy_(
                    candidate, non_blocking=True
                )
        self._upload_stream.synchronize()

    def _emit(self, event: str, state: _MTPSessionState, **fields: Any) -> None:
        self._reserve_event_capacity(1)
        self._events.append(
            {
                "event_id": self._next_event_id,
                "timestamp": time.time(),
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "event": event,
                "session_id": state.session_id,
                **fields,
            }
        )
        self._next_event_id += 1

    def _reserve_event_capacity(self, count: int) -> None:
        if len(self._events) + count > self.max_pending_events:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES pending event queue reached its fail-closed "
                f"limit of {self.max_pending_events}; drain events before "
                "continuing inference"
            )

    def register_session(
        self, *, session_id: str, config: DiagESMTPSessionConfig
    ) -> dict[str, Any]:
        self._validate_session_id(session_id)
        existing = self._sessions.get(session_id)
        if existing is not None:
            if existing.config != config:
                raise ValueError(
                    f"MTP diagonal-ES session {session_id!r} is already registered "
                    "with different hyperparameters"
                )
            return self._session_status(existing)
        if not self._free_slots:
            raise RuntimeError("MTP diagonal-ES session capacity is exhausted")
        self._reserve_event_capacity(1)

        theta_dense = self._new_tensor_map()
        slot = self._free_slots.pop(0)
        state = _MTPSessionState(
            session_id=session_id,
            resident_slot=slot,
            config=config,
            theta_dense=theta_dense,
            noise_sum_dense=_zeros_like_map(theta_dense),
            rewarded_noise_sum_dense=_zeros_like_map(theta_dense),
        )
        try:
            self._upload_candidate(state)
        except BaseException:
            self._free_slots.append(slot)
            self._free_slots.sort()
            raise
        self._sessions[session_id] = state
        self._emit(
            "session_registered",
            state,
            resident_slot=slot,
            theta_version=0,
            population_index=0,
            perturbation_seed=state.perturbation_seed,
        )
        return self._session_status(state)

    def bind_request(self, *, session_id: str, rid: str) -> dict[str, Any]:
        self._validate_session_id(session_id)
        state = self._sessions.get(session_id)
        if state is None:
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} is not registered"
            )
        if state.active_rid not in (None, rid):
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} already has live request "
                f"{state.active_rid!r}"
            )
        state.active_rid = rid
        return self._session_status(state)

    def release_request(self, *, session_id: str, rid: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} is not registered"
            )
        if state.active_rid == rid:
            state.active_rid = None
            state.last_released_rid = rid
            return
        if state.active_rid is None and state.last_released_rid == rid:
            return
        raise DiagESMTPSessionError(
            f"MTP diagonal-ES session {session_id!r} is not bound to request "
            f"{rid!r}; active request is {state.active_rid!r}"
        )

    def retire_session(self, session_id: str) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        if state is None:
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} is not registered"
            )
        if state.active_rid is not None:
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} still owns live request "
                f"{state.active_rid!r}"
            )
        self._wait_for_slot(state.resident_slot)
        with torch.cuda.stream(self._upload_stream):
            for bank in self._dense_delta_banks.values():
                bank[state.resident_slot].zero_()
        self._upload_stream.synchronize()
        del self._sessions[session_id]
        self._slot_last_read_events[state.resident_slot] = None
        self._free_slots.append(state.resident_slot)
        self._free_slots.sort()
        return {"session_id": session_id, "state": "FREE"}

    def note_slots_read(self, resident_slots: list[int] | tuple[int, ...]) -> None:
        stream = torch.cuda.current_stream(self.device)
        for slot in set(resident_slots):
            if slot == 0:
                continue
            event = self._slot_last_read_events[slot]
            if event is None:
                event = torch.cuda.Event(enable_timing=False)
                self._slot_last_read_events[slot] = event
            event.record(stream)

    @staticmethod
    def _accumulate_noise(
        state: _MTPSessionState, candidate_reward: float
    ) -> None:
        for name, noise in state.current_noise_dense.items():
            state.noise_sum_dense[name].add_(noise)
            state.rewarded_noise_sum_dense[name].add_(noise, alpha=candidate_reward)

    @staticmethod
    def _clear_accumulators(state: _MTPSessionState) -> None:
        for values in (
            state.noise_sum_dense,
            state.rewarded_noise_sum_dense,
        ):
            for value in values.values():
                value.zero_()

    @staticmethod
    def _stage_update(
        state: _MTPSessionState,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        rewards = np.asarray(state.candidate_rewards, dtype=np.float64)
        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std(ddof=0))
        scale = (
            0.0
            if reward_std == 0.0 or state.config.learning_rate == 0.0
            else state.config.learning_rate
            / (
                state.config.population_size
                * (reward_std + state.config.reward_zscore_epsilon)
            )
        )

        def stage(
            rewarded: Mapping[str, torch.Tensor], noise_sum: Mapping[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return {
                name: rewarded[name]
                .add(noise_sum[name], alpha=-reward_mean)
                .mul(scale)
                for name in rewarded
            }

        dense = stage(state.rewarded_noise_sum_dense, state.noise_sum_dense)
        all_updates = list(dense.values())
        nonfinite = sum(
            value.numel() - int(torch.isfinite(value).sum()) for value in all_updates
        )
        count = sum(value.numel() for value in all_updates)
        sum_sq = sum(float(value.double().square().sum()) for value in all_updates)
        update_rms = math.sqrt(sum_sq / count)
        update_abs_max = max(float(value.abs().max()) for value in all_updates)
        stats = {
            "candidate_rewards": rewards.tolist(),
            "candidate_reward_mean": reward_mean,
            "candidate_reward_std": reward_std,
            "update_rms": update_rms,
            "update_abs_max": update_abs_max,
            "update_rms_ratio": update_rms / state.config.sigma,
            "update_abs_max_ratio": update_abs_max / state.config.sigma,
            "update_nonfinite_count": nonfinite,
        }
        reasons = []
        if nonfinite:
            reasons.append(f"staged update has {nonfinite} non-finite values")
        if stats["update_rms_ratio"] > state.config.max_update_rms_ratio:
            reasons.append(
                f"update_rms_ratio {stats['update_rms_ratio']:.6g} exceeds "
                f"{state.config.max_update_rms_ratio:.6g}"
            )
        if stats["update_abs_max_ratio"] > state.config.max_update_abs_max_ratio:
            reasons.append(
                f"update_abs_max_ratio {stats['update_abs_max_ratio']:.6g} exceeds "
                f"{state.config.max_update_abs_max_ratio:.6g}"
            )
        stats["update_rejection_reasons"] = reasons
        if reasons:
            raise DiagESMTPUpdateRejected(reasons, stats)
        return dense, stats

    def record_acceptance(
        self,
        *,
        session_id: str,
        rid: str,
        accepted_drafts: int,
    ) -> dict[str, Any]:
        state = self._sessions.get(session_id)
        if state is None:
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} is not registered"
            )
        if state.active_rid != rid:
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} is not bound to request "
                f"{rid!r}"
            )
        if (
            not isinstance(accepted_drafts, int)
            or isinstance(accepted_drafts, bool)
            or accepted_drafts < 0
            or accepted_drafts > self.max_correct_drafts
        ):
            raise ValueError(
                "accepted_drafts must be an integer in "
                f"[0, {self.max_correct_drafts}], got {accepted_drafts!r}"
            )

        completing_candidate = (
            state.candidate_attempts + 1 == state.config.attempts_per_candidate
        )
        completing_population = (
            completing_candidate
            and state.population_index + 1 == state.config.population_size
        )
        self._reserve_event_capacity(
            1 + int(completing_candidate) + int(completing_population)
        )

        state.candidate_attempts += 1
        state.total_attempts += 1
        state.candidate_accept_sum += accepted_drafts
        state.latest_accept_length = accepted_drafts + 1
        state.latest_accepted_drafts = accepted_drafts
        state.latest_attempt_theta_version = state.theta_version
        state.latest_attempt_population_index = state.population_index
        state.latest_attempt_perturbation_seed = state.perturbation_seed
        self._emit(
            "verify_attempt",
            state,
            rid=rid,
            theta_version=state.theta_version,
            population_index=state.population_index,
            perturbation_seed=state.perturbation_seed,
            attempt_index=state.candidate_attempts,
            accepted_drafts=accepted_drafts,
            accept_length=accepted_drafts + 1,
            total_attempts=state.total_attempts,
        )
        if state.candidate_attempts < state.config.attempts_per_candidate:
            return self._session_status(state)

        candidate_reward = (
            state.candidate_accept_sum / state.config.attempts_per_candidate
        )
        state.candidate_rewards.append(candidate_reward)
        self._accumulate_noise(state, candidate_reward)
        self._emit(
            "candidate_completed",
            state,
            rid=rid,
            theta_version=state.theta_version,
            population_index=state.population_index,
            perturbation_seed=state.perturbation_seed,
            attempts=state.candidate_attempts,
            candidate_reward_mean=candidate_reward,
            candidate_rewards=list(state.candidate_rewards),
        )
        state.candidate_attempts = 0
        state.candidate_accept_sum = 0

        if state.population_index + 1 < state.config.population_size:
            state.population_index += 1
            self._upload_candidate(state)
            return self._session_status(state)

        source_version = state.theta_version
        try:
            dense_update, stats = self._stage_update(state)
        except DiagESMTPUpdateRejected as exc:
            state.rejected_updates += 1
            self._emit(
                "update_rejected",
                state,
                rid=rid,
                theta_version=source_version,
                next_theta_version=source_version + 1,
                **exc.stats,
            )
        else:
            for name, update in dense_update.items():
                state.theta_dense[name].add_(update)
            state.committed_updates += 1
            self._emit(
                "update_committed",
                state,
                rid=rid,
                theta_version=source_version,
                next_theta_version=source_version + 1,
                learning_rate=state.config.learning_rate,
                sigma=state.config.sigma,
                **stats,
            )

        state.theta_version += 1
        state.population_index = 0
        state.candidate_rewards.clear()
        self._clear_accumulators(state)
        self._upload_candidate(state)
        return self._session_status(state)

    def drain_events(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id is None:
            events = self._events
            self._events = []
        else:
            self._validate_session_id(session_id)
            events = [e for e in self._events if e["session_id"] == session_id]
            self._events = [e for e in self._events if e["session_id"] != session_id]
        return {"events": events, "next_event_id": self._next_event_id}

    def status(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id is not None:
            state = self._sessions.get(session_id)
            if state is None:
                raise DiagESMTPSessionError(
                    f"MTP diagonal-ES session {session_id!r} is not registered"
                )
            return self._session_status(state)
        return {
            "schema_id": self.manifest.schema_id,
            "schema_digest": self.manifest.schema_digest,
            "placement": self.manifest.placement,
            "model_artifact_id": self.model_artifact_id,
            "rng_version": MTP_RNG_VERSION,
            "max_sessions": self.max_sessions,
            "physical_slots": self.physical_slots,
            "free_slots": list(self._free_slots),
            "free_slot_count": len(self._free_slots),
            "max_correct_drafts": self.max_correct_drafts,
            "max_pending_events": self.max_pending_events,
            "draft_architecture_id": "joyai-llm-flash-dense-nextn-v1",
            "draft_model_class": "JoyAILLMFlashForCausalLMNextN",
            "draft_decoder_class": "JoyAIDenseNextNDecoderLayer",
            "draft_mlp_class": "DeepseekV2MLP",
            "dense_sites": {
                site.site_id: site.width for site in self.manifest.dense_sites
            },
            "dense_active_widths": {
                site.site_id: (
                    site.width if site.active_width is None else site.active_width
                )
                for site in self.manifest.dense_sites
            },
            "pending_event_count": len(self._events),
            "sessions": {
                name: self._session_status(state)
                for name, state in self._sessions.items()
            },
        }

    @staticmethod
    def _session_status(state: _MTPSessionState) -> dict[str, Any]:
        config = state.config
        return {
            "session_id": state.session_id,
            "resident_slot": state.resident_slot,
            "state": "ACTIVE" if state.active_rid is not None else "READY",
            "active_rid": state.active_rid,
            "seed": config.seed,
            "rng_version": MTP_RNG_VERSION,
            "population_size": config.population_size,
            "sigma": config.sigma,
            "learning_rate": config.learning_rate,
            "attempts_per_candidate": config.attempts_per_candidate,
            "estimator": config.estimator,
            "reward_zscore_epsilon": config.reward_zscore_epsilon,
            "max_update_rms_ratio": config.max_update_rms_ratio,
            "max_update_abs_max_ratio": config.max_update_abs_max_ratio,
            "theta_version": state.theta_version,
            "update_index": state.theta_version,
            "population_index": state.population_index,
            "candidate_index": state.population_index,
            "perturbation_seed": state.perturbation_seed,
            "candidate_attempts": state.candidate_attempts,
            "candidate_reward_sum": state.candidate_accept_sum,
            "candidate_reward_mean": (
                state.candidate_accept_sum / state.candidate_attempts
                if state.candidate_attempts
                else None
            ),
            "candidate_rewards": list(state.candidate_rewards),
            "total_attempts": state.total_attempts,
            "latest_accept_length": state.latest_accept_length,
            "latest_accepted_drafts": state.latest_accepted_drafts,
            "latest_attempt_theta_version": state.latest_attempt_theta_version,
            "latest_attempt_population_index": (
                state.latest_attempt_population_index
            ),
            "latest_attempt_perturbation_seed": (
                state.latest_attempt_perturbation_seed
            ),
            "committed_updates": state.committed_updates,
            "rejected_updates": state.rejected_updates,
        }
