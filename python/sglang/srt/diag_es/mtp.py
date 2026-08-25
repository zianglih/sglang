from __future__ import annotations

import copy
import hashlib
import math
import struct
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from sglang.srt.diag_es.manifest import DiagESManifest

MTP_RNG_VERSION = "numpy-philox-site-v1"
MTP_SCHEDULE_RNG_VERSION = "numpy-philox-block-interleaved-v1"
MTP_SESSION_STATE_ABI = "joyai-mtp-session-state-v1"
MTP_MAX_PENDING_EVENTS = 1_000_000
MTP_MAX_EVENT_READ_LIMIT = 4096
MTPCandidateSchedule = Literal["contiguous", "round_robin", "block_interleaved"]

_MTP_SESSION_SNAPSHOT_KEYS = frozenset(
    ("state_abi", "identity", "config", "state", "tensors")
)
_MTP_SESSION_STATE_KEYS = frozenset(
    (
        "candidate_rewards",
        "candidate_accept_sum",
        "candidate_attempts",
        "theta_version",
        "population_index",
        "total_attempts",
        "committed_updates",
        "rejected_updates",
        "latest_accept_length",
        "latest_accepted_drafts",
        "latest_attempt_theta_version",
        "latest_attempt_population_index",
        "latest_attempt_perturbation_seed",
        "round_robin_accept_sums",
        "round_robin_attempt_counts",
        "block_interleaved_accept_sums",
        "block_interleaved_attempt_counts",
        "block_schedule_position",
        "block_attempt_index",
        "latest_attempt_visit_index",
        "latest_attempt_block_attempt_index",
        "latest_attempt_schedule_position",
        "latest_attempt_schedule_order_seed",
    )
)
_MTP_SESSION_TENSOR_KEYS = frozenset(
    ("theta_dense", "noise_sum_dense", "rewarded_noise_sum_dense")
)


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


def mtp_candidate_seed(
    root_seed: int, theta_version: int, population_index: int
) -> int:
    """Match the wrapper's one-sided diagonal-ES candidate-seed contract."""

    return _stable_u64("candidate", root_seed, theta_version, population_index)


def mtp_schedule_order_seed(schedule_seed: int, theta_version: int) -> int:
    """Derive the Philox seed for one theta's candidate-order permutation."""

    return _stable_u64(MTP_SCHEDULE_RNG_VERSION, schedule_seed, theta_version)


def mtp_block_interleaved_orders(
    *,
    schedule_seed: int,
    theta_version: int,
    population_size: int,
    schedule_lane: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Build a deterministic two-visit, boundary-safe candidate schedule.

    Lanes are left rotations of one paired base permutation. The second visit
    reverses the first and swaps its first two entries, so the two visits never
    put the same candidate on both sides of their shared boundary.
    """

    order_seed = mtp_schedule_order_seed(schedule_seed, theta_version)
    generator = np.random.Generator(np.random.Philox(order_seed))
    base_order = tuple(
        int(index) for index in generator.permutation(population_size).tolist()
    )
    offset = schedule_lane % population_size
    visit_0 = base_order[offset:] + base_order[:offset]
    visit_1_values = list(reversed(visit_0))
    visit_1_values[0], visit_1_values[1] = visit_1_values[1], visit_1_values[0]
    visit_1 = tuple(visit_1_values)
    if visit_0[-1] == visit_1[0]:
        raise DiagESMTPSessionError(
            "MTP block-interleaved schedule repeated a candidate at the visit boundary"
        )
    return order_seed, visit_0, visit_1


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
    candidate_schedule: MTPCandidateSchedule = "contiguous"
    candidate_dwell_attempts: int | None = None
    schedule_seed: int | None = None
    schedule_lane: int | None = None
    max_theta_rms_ratio: float | None = None
    max_theta_abs_max_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("MTP diagonal-ES seed must be an integer")
        if self.population_size < 2:
            raise ValueError("MTP diagonal-ES population_size must be at least 2")
        if self.attempts_per_candidate < 1:
            raise ValueError("MTP diagonal-ES attempts_per_candidate must be positive")
        if self.estimator != "population_zscore":
            raise ValueError(
                "MTP diagonal ES supports only estimator='population_zscore'"
            )
        valid_schedules = ("contiguous", "round_robin", "block_interleaved")
        if self.candidate_schedule not in valid_schedules:
            raise ValueError(
                "MTP diagonal-ES candidate_schedule must be 'contiguous', "
                "'round_robin', or 'block_interleaved'"
            )
        schedule_pair = (self.schedule_seed, self.schedule_lane)
        if (schedule_pair[0] is None) != (schedule_pair[1] is None):
            raise ValueError(
                "MTP diagonal-ES schedule_seed and schedule_lane must be provided "
                "together"
            )
        for name, value in zip(("schedule_seed", "schedule_lane"), schedule_pair):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(f"MTP diagonal-ES {name} must be an integer")
        if self.candidate_schedule == "block_interleaved":
            if self.schedule_seed is None:
                raise ValueError(
                    "MTP diagonal-ES block_interleaved requires schedule_seed and "
                    "schedule_lane"
                )
            if (
                not isinstance(self.candidate_dwell_attempts, int)
                or isinstance(self.candidate_dwell_attempts, bool)
                or self.candidate_dwell_attempts < 1
            ):
                raise ValueError(
                    "MTP diagonal-ES block_interleaved candidate_dwell_attempts "
                    "must be a positive integer"
                )
            if self.attempts_per_candidate != 2 * self.candidate_dwell_attempts:
                raise ValueError(
                    "MTP diagonal-ES block_interleaved currently requires "
                    "attempts_per_candidate == 2 * candidate_dwell_attempts"
                )
        elif (
            self.candidate_dwell_attempts is not None or self.schedule_seed is not None
        ):
            raise ValueError(
                "MTP diagonal-ES block scheduling fields require "
                "candidate_schedule='block_interleaved'"
            )
        for name, value, allow_zero in (
            ("sigma", self.sigma, True),
            ("learning_rate", self.learning_rate, True),
            ("reward_zscore_epsilon", self.reward_zscore_epsilon, False),
            ("max_update_rms_ratio", self.max_update_rms_ratio, False),
            ("max_update_abs_max_ratio", self.max_update_abs_max_ratio, False),
        ):
            if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
                relation = "non-negative" if allow_zero else "positive"
                raise ValueError(
                    f"MTP diagonal-ES {name} must be finite and {relation}"
                )
        for name, value in (
            ("max_theta_rms_ratio", self.max_theta_rms_ratio),
            ("max_theta_abs_max_ratio", self.max_theta_abs_max_ratio),
        ):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"MTP diagonal-ES {name} must be finite and positive")
        if self.sigma == 0.0 and self.learning_rate != 0.0:
            raise ValueError(
                "MTP diagonal-ES sigma may be zero only when learning_rate is zero"
            )


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
    round_robin_accept_sums: list[int] = field(default_factory=list)
    round_robin_attempt_counts: list[int] = field(default_factory=list)
    block_interleaved_accept_sums: list[int] = field(default_factory=list)
    block_interleaved_attempt_counts: list[int] = field(default_factory=list)
    block_schedule_position: int = 0
    block_attempt_index: int = 0
    block_schedule_order_seed: int | None = None
    block_visit_0: tuple[int, ...] = ()
    block_visit_1: tuple[int, ...] = ()
    latest_attempt_visit_index: int | None = None
    latest_attempt_block_attempt_index: int | None = None
    latest_attempt_schedule_position: int | None = None
    latest_attempt_schedule_order_seed: int | None = None
    theta_stats: dict[str, Any] = field(default_factory=dict)
    effective_delta_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def perturbation_seed(self) -> int:
        return mtp_candidate_seed(
            self.config.seed, self.theta_version, self.population_index
        )


@dataclass(frozen=True, slots=True)
class _MTPAcceptanceBatchReservation:
    nonce: int
    start_event_id: int
    acceptance_event_count: int
    expects_kv_replay: bool
    attempts: tuple[tuple[str, str, int], ...]
    kv_replay_requests: tuple[tuple[str, str], ...]


def _zeros_like_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: torch.zeros_like(value) for name, value in values.items()}


def _same_typed_tree(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_typed_tree(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_tree(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return actual == expected


def _require_exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DiagESMTPSessionError(
            f"MTP session snapshot {name} must be an integer >= {minimum}"
        )
    return value


def _require_optional_exact_int(
    value: object, *, name: str, minimum: int = 0
) -> int | None:
    if value is None:
        return None
    return _require_exact_int(value, name=name, minimum=minimum)


def _delta_stats(values: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Summarize CPU delta tensors without reading their resident GPU banks."""

    per_site: dict[str, dict[str, int | float]] = {}
    total_count = 0
    total_nonfinite = 0
    total_scale_nonfinite = 0
    total_sum_sq = 0.0
    aggregate_min = math.inf
    aggregate_max = -math.inf
    aggregate_min_scale = math.inf
    aggregate_max_scale = -math.inf

    for name, value in values.items():
        if value.device.type != "cpu":
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES stats require CPU tensor {name!r}, got {value.device}"
            )
        flat = value.detach().numpy().reshape(-1)
        count = flat.size
        finite = np.isfinite(flat)
        nonfinite = count - int(np.count_nonzero(finite))
        scales = np.add(flat, np.float32(1.0), dtype=np.float32)
        finite_scales_mask = np.isfinite(scales)
        scale_nonfinite = count - int(np.count_nonzero(finite_scales_mask))
        finite_values = flat if nonfinite == 0 else flat[finite]
        if finite_values.size == 0:
            value_min = value_max = value_rms = value_abs_max = math.nan
        else:
            value_min = float(np.min(finite_values))
            value_max = float(np.max(finite_values))
            value_sum_sq = float(
                np.square(finite_values, dtype=np.float64).sum(dtype=np.float64)
            )
            value_rms = math.sqrt(value_sum_sq / finite_values.size)
            value_abs_max = max(abs(value_min), abs(value_max))
        finite_scales = scales if scale_nonfinite == 0 else scales[finite_scales_mask]
        if finite_scales.size == 0:
            min_scale = max_scale = math.nan
        else:
            min_scale = float(np.min(finite_scales))
            max_scale = float(np.max(finite_scales))
        per_site[name] = {
            "count": count,
            "nonfinite_count": nonfinite,
            "scale_nonfinite_count": scale_nonfinite,
            "rms": value_rms,
            "absmax": value_abs_max,
            "min": value_min,
            "max": value_max,
            "min_scale": min_scale,
            "max_scale": max_scale,
        }
        total_count += count
        total_nonfinite += nonfinite
        total_scale_nonfinite += scale_nonfinite
        total_sum_sq += 0.0 if finite_values.size == 0 else value_sum_sq
        if finite_values.size:
            aggregate_min = min(aggregate_min, value_min)
            aggregate_max = max(aggregate_max, value_max)
        if finite_scales.size:
            aggregate_min_scale = min(aggregate_min_scale, min_scale)
            aggregate_max_scale = max(aggregate_max_scale, max_scale)

    finite_count = total_count - total_nonfinite
    aggregate = {
        "count": total_count,
        "nonfinite_count": total_nonfinite,
        "scale_nonfinite_count": total_scale_nonfinite,
        "rms": (math.sqrt(total_sum_sq / finite_count) if finite_count else math.nan),
        "absmax": (
            max(abs(aggregate_min), abs(aggregate_max)) if finite_count else math.nan
        ),
        "min": aggregate_min if finite_count else math.nan,
        "max": aggregate_max if finite_count else math.nan,
        "min_scale": aggregate_min_scale if finite_count else math.nan,
        "max_scale": aggregate_max_scale if finite_count else math.nan,
    }
    return {"aggregate": aggregate, "per_site": per_site}


def _require_finite_delta_stats(stats: Mapping[str, Any], *, name: str) -> None:
    aggregate = stats["aggregate"]
    if aggregate["nonfinite_count"] or aggregate["scale_nonfinite_count"]:
        raise DiagESMTPSessionError(
            f"MTP diagonal-ES {name} has non-finite values: "
            f"delta={aggregate['nonfinite_count']}, "
            f"scale={aggregate['scale_nonfinite_count']}"
        )


def _theta_ratio_stats(
    stats: Mapping[str, Any], *, sigma: float, prefix: str = "theta"
) -> dict[str, float]:
    aggregate = stats["aggregate"]

    def ratio(value: float) -> float:
        if sigma == 0.0:
            return 0.0 if value == 0.0 else math.inf
        return value / sigma

    return {
        f"{prefix}_rms_ratio": ratio(float(aggregate["rms"])),
        f"{prefix}_abs_max_ratio": ratio(float(aggregate["absmax"])),
    }


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
        self._initialize_event_protocol()
        self._kv_replay_batch_count = 0
        self._kv_replay_transitioned_requests = 0
        self._kv_replayed_rows = 0
        self._kv_replay_enqueue_time_ms = 0.0
        self._next_acceptance_batch_nonce = 1
        self._active_acceptance_batch: _MTPAcceptanceBatchReservation | None = None
        self._active_acceptance_batch_cursor = 0

    def _initialize_event_protocol(self) -> None:
        self.engine_epoch = str(uuid.uuid4())
        self._events: list[dict[str, Any]] = []
        self._next_event_id = 1
        self._acked_through_event_id = 0
        self._highest_read_through_event_id = 0

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

    def _snapshot_identity(self, session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "model_artifact_id": self.model_artifact_id,
            "schema_id": self.manifest.schema_id,
            "schema_digest": self.manifest.schema_digest,
            "placement": self.manifest.placement,
            "rng_version": MTP_RNG_VERSION,
            "schedule_rng_version": MTP_SCHEDULE_RNG_VERSION,
            "max_correct_drafts": self.max_correct_drafts,
            "dense_sites": [
                {
                    "site_id": site.site_id,
                    "width": site.width,
                    "active_width": (
                        site.width if site.active_width is None else site.active_width
                    ),
                }
                for site in self.manifest.dense_sites
            ],
        }

    @staticmethod
    def _validated_config_payload(
        config: DiagESMTPSessionConfig,
    ) -> dict[str, Any]:
        if type(config) is not DiagESMTPSessionConfig:
            raise DiagESMTPSessionError(
                "MTP session snapshot config has an unsupported type"
            )
        payload = asdict(config)
        try:
            rebuilt = DiagESMTPSessionConfig(**payload)
        except (TypeError, ValueError) as exc:
            raise DiagESMTPSessionError(
                "MTP session snapshot config is invalid"
            ) from exc
        if not _same_typed_tree(asdict(rebuilt), payload):
            raise DiagESMTPSessionError("MTP session snapshot config is not exact")
        return payload

    @staticmethod
    def _require_snapshot_mapping(
        value: object, *, name: str, keys: frozenset[str]
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise DiagESMTPSessionError(
                f"MTP session snapshot {name} must be a mapping"
            )
        actual_keys = set(value)
        if actual_keys != keys:
            missing = sorted(repr(key) for key in keys - actual_keys)
            extra = sorted(repr(key) for key in actual_keys - keys)
            raise DiagESMTPSessionError(
                f"MTP session snapshot {name} keys mismatch: "
                f"missing={missing}, extra={extra}"
            )
        return value

    def _validated_snapshot_tensor_map(
        self, value: object, *, name: str
    ) -> dict[str, torch.Tensor]:
        expected_sites = {site.site_id: site for site in self.manifest.dense_sites}
        if not isinstance(value, Mapping):
            raise DiagESMTPSessionError(
                f"MTP session snapshot tensor map {name!r} must be a mapping"
            )
        actual_sites = set(value)
        if actual_sites != set(expected_sites):
            missing = sorted(repr(site) for site in set(expected_sites) - actual_sites)
            extra = sorted(repr(site) for site in actual_sites - set(expected_sites))
            raise DiagESMTPSessionError(
                f"MTP session snapshot tensor map {name!r} site mismatch: "
                f"missing={missing}, extra={extra}"
            )

        result: dict[str, torch.Tensor] = {}
        for site_id, site in expected_sites.items():
            tensor = value[site_id]
            if not torch.is_tensor(tensor):
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] must be a tensor"
                )
            if tensor.device.type != "cpu":
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] must be on CPU"
                )
            if tensor.dtype != torch.float32:
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] must be FP32"
                )
            if tuple(tensor.shape) != (site.width,):
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] must have shape "
                    f"({site.width},), got {tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous():
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] must be contiguous"
                )
            if tensor.requires_grad:
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] must not require grad"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] has non-finite values"
                )
            active_width = (
                site.width if site.active_width is None else site.active_width
            )
            if int(torch.count_nonzero(tensor[active_width:])) != 0:
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name}[{site_id!r}] has a nonzero "
                    "inactive suffix"
                )
            result[site_id] = tensor.detach().clone()
        return result

    def _clear_resident_slot(self, slot: int) -> None:
        self._wait_for_slot(slot)
        with torch.cuda.stream(self._upload_stream):
            for bank in self._dense_delta_banks.values():
                bank[slot].zero_()
        self._upload_stream.synchronize()
        self._slot_last_read_events[slot] = None

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

    def _materialize_candidate(
        self, state: _MTPSessionState
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if state.config.sigma == 0.0:
            # A sigma=0/LR=0 arm is an exact identity control. Do not materialize
            # Philox noise merely because its population bookkeeping advances.
            noise_dense = _zeros_like_map(state.theta_dense)
            candidate_dense = {
                site.site_id: state.theta_dense[site.site_id].clone()
                for site in self.manifest.dense_sites
            }
        else:
            noise_dense = self._candidate_noise(state.perturbation_seed)
            candidate_dense = {
                site.site_id: state.theta_dense[site.site_id].add(
                    noise_dense[site.site_id], alpha=state.config.sigma
                )
                for site in self.manifest.dense_sites
            }
        return noise_dense, candidate_dense

    def _upload_candidate(self, state: _MTPSessionState) -> None:
        if state.config.sigma == 0.0:
            raise DiagESMTPSessionError(
                "MTP identity controls must not enter the candidate upload path"
            )
        self._wait_for_slot(state.resident_slot)
        noise_dense, candidate_dense = self._materialize_candidate(state)
        effective_delta_stats = _delta_stats(candidate_dense)
        _require_finite_delta_stats(
            effective_delta_stats, name="effective candidate delta"
        )
        with torch.cuda.stream(self._upload_stream):
            for site in self.manifest.dense_sites:
                self._dense_delta_banks[site.site_id][state.resident_slot].copy_(
                    candidate_dense[site.site_id], non_blocking=True
                )
        self._upload_stream.synchronize()
        state.current_noise_dense = noise_dense
        state.effective_delta_stats = effective_delta_stats

    @staticmethod
    def _initialize_identity_candidate(state: _MTPSessionState) -> None:
        state.current_noise_dense = _zeros_like_map(state.theta_dense)
        state.effective_delta_stats = state.theta_stats

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
                f"limit of {self.max_pending_events}; read and acknowledge "
                "events before continuing inference"
            )

    def _validate_acceptance(
        self, *, session_id: str, rid: str, accepted_drafts: int
    ) -> _MTPSessionState:
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
        return state

    @staticmethod
    def _block_interleaved_schedule(
        state: _MTPSessionState,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        config = state.config
        if config.schedule_seed is None or config.schedule_lane is None:
            raise DiagESMTPSessionError(
                "MTP block-interleaved session is missing its paired schedule "
                "configuration"
            )
        order_seed = mtp_schedule_order_seed(config.schedule_seed, state.theta_version)
        if (
            state.block_schedule_order_seed == order_seed
            and len(state.block_visit_0) == config.population_size
            and len(state.block_visit_1) == config.population_size
        ):
            return order_seed, state.block_visit_0, state.block_visit_1
        result = mtp_block_interleaved_orders(
            schedule_seed=config.schedule_seed,
            theta_version=state.theta_version,
            population_size=config.population_size,
            schedule_lane=config.schedule_lane,
        )
        state.block_schedule_order_seed, state.block_visit_0, state.block_visit_1 = (
            result
        )
        return result

    def _validate_persistable_state(self, state: _MTPSessionState) -> dict[str, Any]:
        config = state.config
        population_size = config.population_size
        attempts_per_candidate = config.attempts_per_candidate

        for name in (
            "candidate_accept_sum",
            "candidate_attempts",
            "theta_version",
            "population_index",
            "total_attempts",
            "committed_updates",
            "rejected_updates",
            "block_schedule_position",
            "block_attempt_index",
        ):
            _require_exact_int(getattr(state, name), name=f"state.{name}")
        if state.population_index >= population_size:
            raise DiagESMTPSessionError(
                "MTP session snapshot population_index is out of range"
            )
        if state.committed_updates + state.rejected_updates != state.theta_version:
            raise DiagESMTPSessionError(
                "MTP session snapshot update counters do not equal theta_version"
            )

        if type(state.candidate_rewards) is not list or any(
            type(reward) is not float
            or not math.isfinite(reward)
            or reward < 0.0
            or reward > self.max_correct_drafts
            for reward in state.candidate_rewards
        ):
            raise DiagESMTPSessionError(
                "MTP session snapshot candidate_rewards must be finite float values "
                f"in [0, {self.max_correct_drafts}]"
            )
        if any(
            (accepted_sum := round(reward * attempts_per_candidate)) < 0
            or accepted_sum > attempts_per_candidate * self.max_correct_drafts
            or accepted_sum / attempts_per_candidate != reward
            for reward in state.candidate_rewards
        ):
            raise DiagESMTPSessionError(
                "MTP session snapshot candidate reward is not reachable from an "
                "integer accepted-draft sum"
            )

        accounting_lists = {
            "round_robin_accept_sums": state.round_robin_accept_sums,
            "round_robin_attempt_counts": state.round_robin_attempt_counts,
            "block_interleaved_accept_sums": state.block_interleaved_accept_sums,
            "block_interleaved_attempt_counts": (
                state.block_interleaved_attempt_counts
            ),
        }
        for name, values in accounting_lists.items():
            if type(values) is not list or len(values) != population_size:
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name} must contain "
                    f"{population_size} entries"
                )
            for value in values:
                _require_exact_int(value, name=f"state.{name}[]")

        common_latest = (
            "latest_accept_length",
            "latest_accepted_drafts",
            "latest_attempt_theta_version",
            "latest_attempt_population_index",
            "latest_attempt_perturbation_seed",
        )
        block_latest = (
            "latest_attempt_visit_index",
            "latest_attempt_block_attempt_index",
            "latest_attempt_schedule_position",
            "latest_attempt_schedule_order_seed",
        )
        for name in common_latest + block_latest:
            _require_optional_exact_int(getattr(state, name), name=f"state.{name}")

        zero_round_robin = all(
            value == 0
            for values in (
                state.round_robin_accept_sums,
                state.round_robin_attempt_counts,
            )
            for value in values
        )
        zero_block = all(
            value == 0
            for values in (
                state.block_interleaved_accept_sums,
                state.block_interleaved_attempt_counts,
            )
            for value in values
        )

        if config.candidate_schedule == "contiguous":
            if not zero_round_robin or not zero_block:
                raise DiagESMTPSessionError(
                    "MTP contiguous snapshot has schedule-specific accounting"
                )
            if state.block_schedule_position or state.block_attempt_index:
                raise DiagESMTPSessionError(
                    "MTP contiguous snapshot has block schedule coordinates"
                )
            if state.candidate_attempts >= attempts_per_candidate:
                raise DiagESMTPSessionError(
                    "MTP contiguous snapshot candidate_attempts is out of range"
                )
            if len(state.candidate_rewards) != state.population_index:
                raise DiagESMTPSessionError(
                    "MTP contiguous snapshot rewards do not match population_index"
                )
            progress = (
                state.population_index * attempts_per_candidate
                + state.candidate_attempts
            )
            completed_population_indices = list(range(state.population_index))
        elif config.candidate_schedule == "round_robin":
            if not zero_block:
                raise DiagESMTPSessionError(
                    "MTP round-robin snapshot has block-interleaved accounting"
                )
            if state.block_schedule_position or state.block_attempt_index:
                raise DiagESMTPSessionError(
                    "MTP round-robin snapshot has block schedule coordinates"
                )
            progress = sum(state.round_robin_attempt_counts)
            completed_rounds, expected_population_index = divmod(
                progress, population_size
            )
            expected_counts = [
                completed_rounds + int(index < expected_population_index)
                for index in range(population_size)
            ]
            if (
                completed_rounds >= attempts_per_candidate
                or state.population_index != expected_population_index
                or state.round_robin_attempt_counts != expected_counts
            ):
                raise DiagESMTPSessionError(
                    "MTP round-robin snapshot attempt schedule is inconsistent"
                )
            if any(
                accepted_sum > attempt_count * self.max_correct_drafts
                for accepted_sum, attempt_count in zip(
                    state.round_robin_accept_sums,
                    state.round_robin_attempt_counts,
                )
            ):
                raise DiagESMTPSessionError(
                    "MTP round-robin snapshot accepted-draft sum is out of range"
                )
            completed_candidates = (
                expected_population_index
                if completed_rounds == attempts_per_candidate - 1
                else 0
            )
            expected_rewards = [
                state.round_robin_accept_sums[index] / attempts_per_candidate
                for index in range(completed_candidates)
            ]
            if state.candidate_rewards != expected_rewards:
                raise DiagESMTPSessionError(
                    "MTP round-robin snapshot completed rewards are inconsistent"
                )
            completed_population_indices = list(range(completed_candidates))
            if (
                state.candidate_attempts
                != state.round_robin_attempt_counts[state.population_index]
                or state.candidate_accept_sum
                != state.round_robin_accept_sums[state.population_index]
            ):
                raise DiagESMTPSessionError(
                    "MTP round-robin snapshot active candidate accounting is "
                    "inconsistent"
                )
        else:
            if not zero_round_robin:
                raise DiagESMTPSessionError(
                    "MTP block-interleaved snapshot has round-robin accounting"
                )
            dwell_attempts = config.candidate_dwell_attempts
            if dwell_attempts is None:
                raise DiagESMTPSessionError(
                    "MTP block-interleaved snapshot is missing dwell attempts"
                )
            if (
                state.block_schedule_position >= 2 * population_size
                or state.block_attempt_index >= dwell_attempts
            ):
                raise DiagESMTPSessionError(
                    "MTP block-interleaved snapshot schedule coordinate is out of range"
                )
            _, visit_0, visit_1 = mtp_block_interleaved_orders(
                schedule_seed=config.schedule_seed,
                theta_version=state.theta_version,
                population_size=population_size,
                schedule_lane=config.schedule_lane,
            )
            schedule = visit_0 + visit_1
            expected_counts = [0] * population_size
            for candidate in schedule[: state.block_schedule_position]:
                expected_counts[candidate] += dwell_attempts
            active_candidate = schedule[state.block_schedule_position]
            expected_counts[active_candidate] += state.block_attempt_index
            if (
                state.population_index != active_candidate
                or state.block_interleaved_attempt_counts != expected_counts
            ):
                raise DiagESMTPSessionError(
                    "MTP block-interleaved snapshot attempt schedule is inconsistent"
                )
            if any(
                accepted_sum > attempt_count * self.max_correct_drafts
                for accepted_sum, attempt_count in zip(
                    state.block_interleaved_accept_sums,
                    state.block_interleaved_attempt_counts,
                )
            ):
                raise DiagESMTPSessionError(
                    "MTP block-interleaved snapshot accepted-draft sum is out of range"
                )
            completed_candidates = max(
                0, state.block_schedule_position - population_size
            )
            expected_rewards = [
                state.block_interleaved_accept_sums[candidate] / attempts_per_candidate
                for candidate in visit_1[:completed_candidates]
            ]
            if state.candidate_rewards != expected_rewards:
                raise DiagESMTPSessionError(
                    "MTP block-interleaved snapshot completed rewards are inconsistent"
                )
            completed_population_indices = list(visit_1[:completed_candidates])
            if (
                state.candidate_attempts
                != state.block_interleaved_attempt_counts[active_candidate]
                or state.candidate_accept_sum
                != state.block_interleaved_accept_sums[active_candidate]
            ):
                raise DiagESMTPSessionError(
                    "MTP block-interleaved snapshot active candidate accounting is "
                    "inconsistent"
                )
            progress = (
                state.block_schedule_position * dwell_attempts
                + state.block_attempt_index
            )

        if state.candidate_attempts >= attempts_per_candidate:
            raise DiagESMTPSessionError(
                "MTP session snapshot active candidate is already complete"
            )
        if (
            state.candidate_accept_sum
            > state.candidate_attempts * self.max_correct_drafts
        ):
            raise DiagESMTPSessionError(
                "MTP session snapshot active candidate reward sum is out of range"
            )
        attempts_per_population = population_size * attempts_per_candidate
        if progress >= attempts_per_population:
            raise DiagESMTPSessionError(
                "MTP session snapshot current population is already complete"
            )
        expected_total_attempts = (
            state.theta_version * attempts_per_population + progress
        )
        if state.total_attempts != expected_total_attempts:
            raise DiagESMTPSessionError(
                "MTP session snapshot total_attempts is inconsistent"
            )

        if state.total_attempts == 0:
            if any(getattr(state, name) is not None for name in common_latest):
                raise DiagESMTPSessionError(
                    "MTP session snapshot has latest-attempt state before any attempt"
                )
        else:
            if any(getattr(state, name) is None for name in common_latest):
                raise DiagESMTPSessionError(
                    "MTP session snapshot latest-attempt state is incomplete"
                )
            latest_accepted_drafts = state.latest_accepted_drafts
            if (
                latest_accepted_drafts > self.max_correct_drafts
                or state.latest_accept_length != latest_accepted_drafts + 1
            ):
                raise DiagESMTPSessionError(
                    "MTP session snapshot latest acceptance length is inconsistent"
                )
            expected_latest_theta = (
                state.theta_version if progress else state.theta_version - 1
            )
            if state.latest_attempt_theta_version != expected_latest_theta:
                raise DiagESMTPSessionError(
                    "MTP session snapshot latest theta coordinate is inconsistent"
                )
            if config.candidate_schedule == "contiguous":
                expected_latest_population = (
                    state.population_index
                    if state.candidate_attempts
                    else (
                        state.population_index - 1 if progress else population_size - 1
                    )
                )
            elif config.candidate_schedule == "round_robin":
                expected_latest_population = (
                    (state.population_index - 1) % population_size
                    if progress
                    else population_size - 1
                )
            else:
                dwell_attempts = config.candidate_dwell_attempts
                latest_schedule_position = (
                    state.block_schedule_position
                    if state.block_attempt_index
                    else (
                        state.block_schedule_position - 1
                        if progress
                        else 2 * population_size - 1
                    )
                )
                latest_block_attempt = (
                    state.block_attempt_index
                    if state.block_attempt_index
                    else dwell_attempts
                )
                _, latest_visit_0, latest_visit_1 = mtp_block_interleaved_orders(
                    schedule_seed=config.schedule_seed,
                    theta_version=expected_latest_theta,
                    population_size=population_size,
                    schedule_lane=config.schedule_lane,
                )
                latest_schedule = latest_visit_0 + latest_visit_1
                expected_latest_population = latest_schedule[latest_schedule_position]
                expected_latest_order_seed = mtp_schedule_order_seed(
                    config.schedule_seed, expected_latest_theta
                )
                expected_block_latest = (
                    latest_schedule_position // population_size,
                    latest_block_attempt,
                    latest_schedule_position,
                    expected_latest_order_seed,
                )
                actual_block_latest = tuple(
                    getattr(state, name) for name in block_latest
                )
                if actual_block_latest != expected_block_latest:
                    raise DiagESMTPSessionError(
                        "MTP block-interleaved snapshot latest-attempt schedule is "
                        "inconsistent"
                    )
            if state.latest_attempt_population_index != expected_latest_population:
                raise DiagESMTPSessionError(
                    "MTP session snapshot latest population coordinate is inconsistent"
                )
            expected_latest_seed = mtp_candidate_seed(
                config.seed, expected_latest_theta, expected_latest_population
            )
            if state.latest_attempt_perturbation_seed != expected_latest_seed:
                raise DiagESMTPSessionError(
                    "MTP session snapshot latest perturbation seed is inconsistent"
                )

        if config.candidate_schedule != "block_interleaved" and any(
            getattr(state, name) is not None for name in block_latest
        ):
            raise DiagESMTPSessionError(
                "MTP non-block snapshot has block latest-attempt coordinates"
            )
        if config.candidate_schedule == "block_interleaved" and (
            (state.total_attempts == 0)
            != all(getattr(state, name) is None for name in block_latest)
        ):
            raise DiagESMTPSessionError(
                "MTP block-interleaved snapshot latest-attempt state is incomplete"
            )

        theta_stats = _delta_stats(state.theta_dense)
        _require_finite_delta_stats(theta_stats, name="restored theta")
        theta_ratios = _theta_ratio_stats(theta_stats, sigma=config.sigma)
        if (
            config.max_theta_rms_ratio is not None
            and theta_ratios["theta_rms_ratio"] > config.max_theta_rms_ratio
        ):
            raise DiagESMTPSessionError(
                "MTP session snapshot theta exceeds max_theta_rms_ratio"
            )
        if (
            config.max_theta_abs_max_ratio is not None
            and theta_ratios["theta_abs_max_ratio"] > config.max_theta_abs_max_ratio
        ):
            raise DiagESMTPSessionError(
                "MTP session snapshot theta exceeds max_theta_abs_max_ratio"
            )
        for name, values in (
            ("noise_sum", state.noise_sum_dense),
            ("rewarded_noise_sum", state.rewarded_noise_sum_dense),
        ):
            _require_finite_delta_stats(_delta_stats(values), name=f"restored {name}")
        if config.sigma == 0.0 and any(
            int(torch.count_nonzero(value)) for value in state.theta_dense.values()
        ):
            raise DiagESMTPSessionError(
                "MTP session snapshot sigma-zero state has nonzero theta"
            )

        expected_noise_sum = self._new_tensor_map()
        expected_rewarded_noise_sum = self._new_tensor_map()
        if config.sigma != 0.0:
            for population_index, reward in zip(
                completed_population_indices, state.candidate_rewards
            ):
                noise = self._candidate_noise(
                    mtp_candidate_seed(
                        config.seed, state.theta_version, population_index
                    )
                )
                for site_id, value in noise.items():
                    expected_noise_sum[site_id].add_(value)
                    expected_rewarded_noise_sum[site_id].add_(value, alpha=reward)
        for name, actual, expected in (
            ("noise_sum", state.noise_sum_dense, expected_noise_sum),
            (
                "rewarded_noise_sum",
                state.rewarded_noise_sum_dense,
                expected_rewarded_noise_sum,
            ),
        ):
            if any(
                not torch.equal(actual[site_id], expected[site_id])
                for site_id in expected
            ):
                raise DiagESMTPSessionError(
                    f"MTP session snapshot {name} is inconsistent with completed "
                    "candidate rewards"
                )
        return theta_stats

    @staticmethod
    def _prepare_block_interleaved_tracking(state: _MTPSessionState) -> None:
        population_size = state.config.population_size
        if (
            not state.block_interleaved_accept_sums
            and not state.block_interleaved_attempt_counts
        ):
            state.block_interleaved_accept_sums = [0] * population_size
            state.block_interleaved_attempt_counts = [0] * population_size
        if (
            len(state.block_interleaved_accept_sums) != population_size
            or len(state.block_interleaved_attempt_counts) != population_size
        ):
            raise DiagESMTPSessionError(
                "MTP diagonal-ES block-interleaved accounting does not match the "
                f"population size {population_size}"
            )

    @classmethod
    def _activate_block_schedule_position(
        cls, state: _MTPSessionState, schedule_position: int
    ) -> None:
        population_size = state.config.population_size
        if schedule_position < 0 or schedule_position >= 2 * population_size:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES block schedule position is out of range"
            )
        cls._prepare_block_interleaved_tracking(state)
        _, visit_0, visit_1 = cls._block_interleaved_schedule(state)
        visit_index, visit_position = divmod(schedule_position, population_size)
        order = visit_0 if visit_index == 0 else visit_1
        population_index = order[visit_position]
        state.block_schedule_position = schedule_position
        state.block_attempt_index = 0
        state.population_index = population_index
        state.candidate_attempts = state.block_interleaved_attempt_counts[
            population_index
        ]
        state.candidate_accept_sum = state.block_interleaved_accept_sums[
            population_index
        ]

    @staticmethod
    def _acceptance_event_plan(state: _MTPSessionState) -> tuple[int, bool]:
        if state.config.candidate_schedule == "round_robin":
            counts = state.round_robin_attempt_counts
            if len(counts) != state.config.population_size:
                raise DiagESMTPSessionError(
                    "MTP diagonal-ES round-robin accounting does not match the "
                    f"population size {state.config.population_size}"
                )
            attempt_count = counts[state.population_index] + 1
            completing_candidate = attempt_count == state.config.attempts_per_candidate
            if attempt_count > state.config.attempts_per_candidate:
                raise DiagESMTPSessionError(
                    "MTP diagonal-ES round-robin attempt count exceeded its limit"
                )
            completing_population = (
                completing_candidate
                and state.population_index + 1 == state.config.population_size
            )
            # Round-robin advances to a different population member after every
            # attempt. Sigma-zero controls are the sole effective-identity case.
            changes_candidate = state.config.sigma != 0.0
        elif state.config.candidate_schedule == "block_interleaved":
            dwell_attempts = state.config.candidate_dwell_attempts
            if dwell_attempts is None:
                raise DiagESMTPSessionError(
                    "MTP block-interleaved session is missing dwell attempts"
                )
            completing_block = state.block_attempt_index + 1 == dwell_attempts
            if state.block_attempt_index + 1 > dwell_attempts:
                raise DiagESMTPSessionError(
                    "MTP diagonal-ES block attempt count exceeded its limit"
                )
            visit_index = state.block_schedule_position // state.config.population_size
            completing_candidate = completing_block and visit_index == 1
            completing_population = (
                completing_candidate
                and state.block_schedule_position
                == 2 * state.config.population_size - 1
            )
            changes_candidate = state.config.sigma != 0.0 and completing_block
        else:
            completing_candidate = (
                state.candidate_attempts + 1 == state.config.attempts_per_candidate
            )
            completing_population = (
                completing_candidate
                and state.population_index + 1 == state.config.population_size
            )
            changes_candidate = state.config.sigma != 0.0 and completing_candidate
        return (
            1 + int(completing_candidate) + int(completing_population),
            changes_candidate,
        )

    def preflight_acceptance_batch(
        self, attempts: Sequence[tuple[str, str, int]]
    ) -> _MTPAcceptanceBatchReservation:
        """Reserve a synchronous verify batch's events before any state change."""

        if self._active_acceptance_batch is not None:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES acceptance batch reservation is already active"
            )
        normalized = tuple(attempts)
        if not normalized:
            raise ValueError("MTP diagonal-ES acceptance preflight requires attempts")
        if len({session_id for session_id, _, _ in normalized}) != len(normalized):
            raise DiagESMTPSessionError(
                "MTP diagonal-ES acceptance batch contains a duplicate session"
            )

        acceptance_event_count = 0
        kv_replay_requests = []
        for session_id, rid, accepted_drafts in normalized:
            state = self._validate_acceptance(
                session_id=session_id,
                rid=rid,
                accepted_drafts=accepted_drafts,
            )
            event_count, changes_candidate = self._acceptance_event_plan(state)
            acceptance_event_count += event_count
            if changes_candidate:
                kv_replay_requests.append((session_id, rid))

        replay_requests = tuple(kv_replay_requests)
        expects_kv_replay = bool(replay_requests)

        self._reserve_event_capacity(acceptance_event_count + int(expects_kv_replay))
        reservation = _MTPAcceptanceBatchReservation(
            nonce=self._next_acceptance_batch_nonce,
            start_event_id=self._next_event_id,
            acceptance_event_count=acceptance_event_count,
            expects_kv_replay=expects_kv_replay,
            attempts=normalized,
            kv_replay_requests=replay_requests,
        )
        self._next_acceptance_batch_nonce += 1
        self._active_acceptance_batch = reservation
        self._active_acceptance_batch_cursor = 0
        return reservation

    def _validate_acceptance_batch_progress(
        self,
        reservation: _MTPAcceptanceBatchReservation,
        *,
        expects_kv_replay: bool,
    ) -> None:
        if self._active_acceptance_batch is not reservation:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES acceptance batch reservation is not active"
            )
        if reservation.expects_kv_replay != expects_kv_replay:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES acceptance batch replay prediction mismatched"
            )
        if self._active_acceptance_batch_cursor != len(reservation.attempts):
            raise DiagESMTPSessionError(
                "MTP diagonal-ES acceptance batch did not consume every attempt"
            )
        expected_next_event_id = (
            reservation.start_event_id + reservation.acceptance_event_count
        )
        if self._next_event_id != expected_next_event_id:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES acceptance batch event ordering was interrupted"
            )

    def finish_acceptance_batch(
        self, reservation: _MTPAcceptanceBatchReservation
    ) -> None:
        self._validate_acceptance_batch_progress(reservation, expects_kv_replay=False)
        self._active_acceptance_batch = None
        self._active_acceptance_batch_cursor = 0

    @staticmethod
    def _snapshot_state_payload(state: _MTPSessionState) -> dict[str, Any]:
        return {
            "candidate_rewards": list(state.candidate_rewards),
            "candidate_accept_sum": state.candidate_accept_sum,
            "candidate_attempts": state.candidate_attempts,
            "theta_version": state.theta_version,
            "population_index": state.population_index,
            "total_attempts": state.total_attempts,
            "committed_updates": state.committed_updates,
            "rejected_updates": state.rejected_updates,
            "latest_accept_length": state.latest_accept_length,
            "latest_accepted_drafts": state.latest_accepted_drafts,
            "latest_attempt_theta_version": state.latest_attempt_theta_version,
            "latest_attempt_population_index": (state.latest_attempt_population_index),
            "latest_attempt_perturbation_seed": (
                state.latest_attempt_perturbation_seed
            ),
            "round_robin_accept_sums": list(state.round_robin_accept_sums),
            "round_robin_attempt_counts": list(state.round_robin_attempt_counts),
            "block_interleaved_accept_sums": list(state.block_interleaved_accept_sums),
            "block_interleaved_attempt_counts": list(
                state.block_interleaved_attempt_counts
            ),
            "block_schedule_position": state.block_schedule_position,
            "block_attempt_index": state.block_attempt_index,
            "latest_attempt_visit_index": state.latest_attempt_visit_index,
            "latest_attempt_block_attempt_index": (
                state.latest_attempt_block_attempt_index
            ),
            "latest_attempt_schedule_position": (
                state.latest_attempt_schedule_position
            ),
            "latest_attempt_schedule_order_seed": (
                state.latest_attempt_schedule_order_seed
            ),
        }

    def export_session_state(self, session_id: str) -> dict[str, Any]:
        """Return a detached, versioned snapshot at a between-request boundary."""

        self._validate_session_id(session_id)
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
        if self._active_acceptance_batch is not None:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES session state cannot be exported during an "
                "active acceptance batch reservation"
            )

        tensor_payload = {
            "theta_dense": self._validated_snapshot_tensor_map(
                state.theta_dense, name="theta_dense"
            ),
            "noise_sum_dense": self._validated_snapshot_tensor_map(
                state.noise_sum_dense, name="noise_sum_dense"
            ),
            "rewarded_noise_sum_dense": self._validated_snapshot_tensor_map(
                state.rewarded_noise_sum_dense, name="rewarded_noise_sum_dense"
            ),
        }
        self._validate_persistable_state(state)
        return {
            "state_abi": MTP_SESSION_STATE_ABI,
            "identity": self._snapshot_identity(session_id),
            "config": self._validated_config_payload(state.config),
            "state": self._snapshot_state_payload(state),
            "tensors": tensor_payload,
        }

    def export_session_state_with_frontier(self, session_id: str) -> dict[str, Any]:
        """Atomically pair a portable session snapshot with its event frontier."""

        session_state = self.export_session_state(session_id)
        return {
            "session_state": session_state,
            "telemetry_frontier": {
                "engine_epoch": self.engine_epoch,
                "event_high_watermark": self._next_event_id - 1,
            },
        }

    def _decode_session_snapshot(
        self,
        *,
        session_id: str,
        config: DiagESMTPSessionConfig,
        snapshot: Mapping[str, Any],
    ) -> _MTPSessionState:
        root = self._require_snapshot_mapping(
            snapshot, name="root", keys=_MTP_SESSION_SNAPSHOT_KEYS
        )
        if root["state_abi"] != MTP_SESSION_STATE_ABI:
            raise DiagESMTPSessionError("MTP session snapshot state_abi is unsupported")
        expected_identity = self._snapshot_identity(session_id)
        if not _same_typed_tree(root["identity"], expected_identity):
            raise DiagESMTPSessionError(
                "MTP session snapshot runtime identity does not match"
            )
        expected_config = self._validated_config_payload(config)
        if not _same_typed_tree(root["config"], expected_config):
            raise DiagESMTPSessionError(
                "MTP session snapshot config does not match registration"
            )
        state_payload = self._require_snapshot_mapping(
            root["state"], name="state", keys=_MTP_SESSION_STATE_KEYS
        )
        tensor_payload = self._require_snapshot_mapping(
            root["tensors"], name="tensors", keys=_MTP_SESSION_TENSOR_KEYS
        )
        theta_dense = self._validated_snapshot_tensor_map(
            tensor_payload["theta_dense"], name="theta_dense"
        )
        noise_sum_dense = self._validated_snapshot_tensor_map(
            tensor_payload["noise_sum_dense"], name="noise_sum_dense"
        )
        rewarded_noise_sum_dense = self._validated_snapshot_tensor_map(
            tensor_payload["rewarded_noise_sum_dense"],
            name="rewarded_noise_sum_dense",
        )

        state = _MTPSessionState(
            session_id=session_id,
            resident_slot=-1,
            config=config,
            theta_dense=theta_dense,
            noise_sum_dense=noise_sum_dense,
            rewarded_noise_sum_dense=rewarded_noise_sum_dense,
            candidate_rewards=state_payload["candidate_rewards"],
            candidate_accept_sum=state_payload["candidate_accept_sum"],
            candidate_attempts=state_payload["candidate_attempts"],
            theta_version=state_payload["theta_version"],
            population_index=state_payload["population_index"],
            total_attempts=state_payload["total_attempts"],
            committed_updates=state_payload["committed_updates"],
            rejected_updates=state_payload["rejected_updates"],
            latest_accept_length=state_payload["latest_accept_length"],
            latest_accepted_drafts=state_payload["latest_accepted_drafts"],
            latest_attempt_theta_version=state_payload["latest_attempt_theta_version"],
            latest_attempt_population_index=state_payload[
                "latest_attempt_population_index"
            ],
            latest_attempt_perturbation_seed=state_payload[
                "latest_attempt_perturbation_seed"
            ],
            round_robin_accept_sums=state_payload["round_robin_accept_sums"],
            round_robin_attempt_counts=state_payload["round_robin_attempt_counts"],
            block_interleaved_accept_sums=state_payload[
                "block_interleaved_accept_sums"
            ],
            block_interleaved_attempt_counts=state_payload[
                "block_interleaved_attempt_counts"
            ],
            block_schedule_position=state_payload["block_schedule_position"],
            block_attempt_index=state_payload["block_attempt_index"],
            latest_attempt_visit_index=state_payload["latest_attempt_visit_index"],
            latest_attempt_block_attempt_index=state_payload[
                "latest_attempt_block_attempt_index"
            ],
            latest_attempt_schedule_position=state_payload[
                "latest_attempt_schedule_position"
            ],
            latest_attempt_schedule_order_seed=state_payload[
                "latest_attempt_schedule_order_seed"
            ],
        )
        state.theta_stats = self._validate_persistable_state(state)
        state.candidate_rewards = list(state.candidate_rewards)
        state.round_robin_accept_sums = list(state.round_robin_accept_sums)
        state.round_robin_attempt_counts = list(state.round_robin_attempt_counts)
        state.block_interleaved_accept_sums = list(state.block_interleaved_accept_sums)
        state.block_interleaved_attempt_counts = list(
            state.block_interleaved_attempt_counts
        )
        if config.candidate_schedule == "block_interleaved":
            (
                state.block_schedule_order_seed,
                state.block_visit_0,
                state.block_visit_1,
            ) = mtp_block_interleaved_orders(
                schedule_seed=config.schedule_seed,
                theta_version=state.theta_version,
                population_size=config.population_size,
                schedule_lane=config.schedule_lane,
            )
        noise_dense, candidate_dense = self._materialize_candidate(state)
        effective_delta_stats = _delta_stats(candidate_dense)
        _require_finite_delta_stats(
            effective_delta_stats, name="restored effective candidate delta"
        )
        state.current_noise_dense = noise_dense
        state.effective_delta_stats = effective_delta_stats
        return state

    def import_session_state(
        self,
        *,
        session_id: str,
        config: DiagESMTPSessionConfig,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Restore one absent session without advancing optimizer state."""

        self._validate_session_id(session_id)
        self._validated_config_payload(config)
        if self._active_acceptance_batch is not None:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES session state cannot be imported during an "
                "active acceptance batch reservation"
            )
        if session_id in self._sessions:
            raise DiagESMTPSessionError(
                f"MTP diagonal-ES session {session_id!r} is already registered"
            )
        if not self._free_slots:
            raise RuntimeError("MTP diagonal-ES session capacity is exhausted")

        state = self._decode_session_snapshot(
            session_id=session_id, config=config, snapshot=snapshot
        )
        slot = self._free_slots.pop(0)
        state.resident_slot = slot
        try:
            if config.sigma == 0.0:
                self._clear_resident_slot(slot)
                self._initialize_identity_candidate(state)
            else:
                self._upload_candidate(state)
            status = self._session_status(state)
            self._sessions[session_id] = state
        except BaseException:
            self._sessions.pop(session_id, None)
            try:
                self._clear_resident_slot(slot)
            except BaseException as cleanup_error:
                raise DiagESMTPSessionError(
                    "MTP session restore failed and its resident slot could not be "
                    "cleaned; the slot was quarantined"
                ) from cleanup_error
            self._free_slots.append(slot)
            self._free_slots.sort()
            raise

        return status

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
            round_robin_accept_sums=[0] * config.population_size,
            round_robin_attempt_counts=[0] * config.population_size,
            block_interleaved_accept_sums=[0] * config.population_size,
            block_interleaved_attempt_counts=[0] * config.population_size,
            theta_stats=_delta_stats(theta_dense),
        )
        if config.candidate_schedule == "block_interleaved":
            self._activate_block_schedule_position(state, 0)
        if config.sigma == 0.0:
            # Banks are allocated as exact zeros and retired slots are zeroed
            # before reuse. Identity controls therefore need no upload/sync.
            self._initialize_identity_candidate(state)
        else:
            try:
                self._upload_candidate(state)
            except BaseException:
                try:
                    self._clear_resident_slot(slot)
                except BaseException as cleanup_error:
                    raise DiagESMTPSessionError(
                        "MTP session registration failed and its resident slot "
                        "could not be cleaned; the slot was quarantined"
                    ) from cleanup_error
                self._free_slots.append(slot)
                self._free_slots.sort()
                raise
        self._sessions[session_id] = state
        self._emit(
            "session_registered",
            state,
            resident_slot=slot,
            theta_version=0,
            population_index=state.population_index,
            perturbation_seed=state.perturbation_seed,
            **(
                {
                    "candidate_schedule": config.candidate_schedule,
                    "candidate_dwell_attempts": config.candidate_dwell_attempts,
                    "schedule_seed": config.schedule_seed,
                    "schedule_lane": config.schedule_lane,
                    "schedule_order_seed": state.block_schedule_order_seed,
                }
                if config.candidate_schedule == "block_interleaved"
                else {}
            ),
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
        self._clear_resident_slot(state.resident_slot)
        del self._sessions[session_id]
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

    def record_kv_replay(
        self,
        *,
        acceptance_batch_reservation: _MTPAcceptanceBatchReservation,
        session_ids: list[str],
        request_rows: list[int],
        replayed_rows: int,
        enqueue_time_ms: float,
    ) -> None:
        """Append one ordered aggregate event for a candidate-switch replay."""

        if len(session_ids) != len(request_rows) or not session_ids:
            raise ValueError("MTP draft-KV replay telemetry has invalid request rows")
        if replayed_rows != sum(request_rows) or replayed_rows < 0:
            raise ValueError("MTP draft-KV replay telemetry row total is inconsistent")
        if not math.isfinite(enqueue_time_ms) or enqueue_time_ms < 0:
            raise ValueError("MTP draft-KV replay telemetry enqueue time is invalid")
        self._validate_acceptance_batch_progress(
            acceptance_batch_reservation, expects_kv_replay=True
        )
        expected_session_ids = tuple(
            session_id
            for session_id, _ in acceptance_batch_reservation.kv_replay_requests
        )
        if tuple(session_ids) != expected_session_ids:
            raise DiagESMTPSessionError(
                "MTP draft-KV replay telemetry sessions do not match the active "
                "acceptance batch order"
            )
        request_ids = dict(acceptance_batch_reservation.kv_replay_requests)
        self._kv_replay_batch_count += 1
        self._kv_replay_transitioned_requests += len(session_ids)
        self._kv_replayed_rows += replayed_rows
        self._kv_replay_enqueue_time_ms += enqueue_time_ms
        self._events.append(
            {
                "event_id": self._next_event_id,
                "timestamp": time.time(),
                "monotonic_timestamp_ns": time.monotonic_ns(),
                "event": "draft_kv_prefix_replay",
                "session_id": None,
                "event_scope": "global_batch",
                "session_ids": list(session_ids),
                "request_ids": request_ids,
                "request_replayed_rows": dict(zip(session_ids, request_rows)),
                "transitioned_request_count": len(session_ids),
                "replayed_rows": replayed_rows,
                "enqueue_time_ms": enqueue_time_ms,
            }
        )
        self._next_event_id += 1
        self._active_acceptance_batch = None
        self._active_acceptance_batch_cursor = 0

    @staticmethod
    def _accumulate_noise(state: _MTPSessionState, candidate_reward: float) -> None:
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
                name: rewarded[name].add(noise_sum[name], alpha=-reward_mean).mul(scale)
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
        if state.config.sigma == 0.0:
            # The only valid sigma-zero arm also has LR=0, hence an exact zero
            # staged update. Keep safety telemetry finite without dividing by 0.
            update_rms_ratio = 0.0 if update_rms == 0.0 else math.inf
            update_abs_max_ratio = 0.0 if update_abs_max == 0.0 else math.inf
        else:
            update_rms_ratio = update_rms / state.config.sigma
            update_abs_max_ratio = update_abs_max / state.config.sigma
        stats = {
            "candidate_rewards": rewards.tolist(),
            "candidate_reward_mean": reward_mean,
            "candidate_reward_std": reward_std,
            "update_rms": update_rms,
            "update_abs_max": update_abs_max,
            "update_rms_ratio": update_rms_ratio,
            "update_abs_max_ratio": update_abs_max_ratio,
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
        return dense, stats

    def _finish_population_update(
        self, *, state: _MTPSessionState, rid: str
    ) -> dict[str, Any]:
        source_version = state.theta_version
        try:
            dense_update, stats = self._stage_update(state)
            if state.config.sigma == 0.0:
                if any(
                    int(torch.count_nonzero(value)) != 0
                    for value in dense_update.values()
                ):
                    raise DiagESMTPSessionError(
                        "MTP identity control produced a nonzero update"
                    )
                next_theta = None
                next_theta_stats = state.theta_stats
            else:
                next_theta = {
                    name: state.theta_dense[name].add(update)
                    for name, update in dense_update.items()
                }
                next_theta_stats = _delta_stats(next_theta)
            proposed_theta_ratios = _theta_ratio_stats(
                next_theta_stats,
                sigma=state.config.sigma,
                prefix="proposed_theta",
            )
            stats = {
                **stats,
                "proposed_theta_stats": next_theta_stats,
                **proposed_theta_ratios,
            }
            reasons = list(stats["update_rejection_reasons"])
            try:
                _require_finite_delta_stats(next_theta_stats, name="proposed theta")
            except DiagESMTPSessionError as exc:
                reasons.append(str(exc))
            if (
                state.config.max_theta_rms_ratio is not None
                and proposed_theta_ratios["proposed_theta_rms_ratio"]
                > state.config.max_theta_rms_ratio
            ):
                reasons.append(
                    "proposed_theta_rms_ratio "
                    f"{proposed_theta_ratios['proposed_theta_rms_ratio']:.6g} "
                    "exceeds max_theta_rms_ratio "
                    f"{state.config.max_theta_rms_ratio:.6g}"
                )
            if (
                state.config.max_theta_abs_max_ratio is not None
                and proposed_theta_ratios["proposed_theta_abs_max_ratio"]
                > state.config.max_theta_abs_max_ratio
            ):
                reasons.append(
                    "proposed_theta_abs_max_ratio "
                    f"{proposed_theta_ratios['proposed_theta_abs_max_ratio']:.6g} "
                    "exceeds max_theta_abs_max_ratio "
                    f"{state.config.max_theta_abs_max_ratio:.6g}"
                )
            if reasons:
                stats["update_rejection_reasons"] = reasons
                raise DiagESMTPUpdateRejected(reasons, stats)
        except DiagESMTPUpdateRejected as exc:
            state.rejected_updates += 1
            self._emit(
                "update_rejected",
                state,
                rid=rid,
                theta_version=source_version,
                next_theta_version=source_version + 1,
                theta_stats=state.theta_stats,
                **exc.stats,
            )
        else:
            if next_theta is not None:
                for name, value in next_theta.items():
                    state.theta_dense[name].copy_(value)
            state.theta_stats = next_theta_stats
            state.committed_updates += 1
            self._emit(
                "update_committed",
                state,
                rid=rid,
                theta_version=source_version,
                next_theta_version=source_version + 1,
                learning_rate=state.config.learning_rate,
                sigma=state.config.sigma,
                theta_stats=state.theta_stats,
                **stats,
            )

        state.theta_version += 1
        state.population_index = 0
        state.candidate_rewards.clear()
        state.candidate_attempts = 0
        state.candidate_accept_sum = 0
        state.round_robin_accept_sums = [0] * state.config.population_size
        state.round_robin_attempt_counts = [0] * state.config.population_size
        state.block_interleaved_accept_sums = [0] * state.config.population_size
        state.block_interleaved_attempt_counts = [0] * state.config.population_size
        state.block_schedule_position = 0
        state.block_attempt_index = 0
        if state.config.candidate_schedule == "block_interleaved":
            self._activate_block_schedule_position(state, 0)
        self._clear_accumulators(state)
        if state.config.sigma != 0.0:
            self._upload_candidate(state)
        return self._session_status(state)

    @staticmethod
    def _prepare_round_robin_tracking(state: _MTPSessionState) -> None:
        population_size = state.config.population_size
        if not state.round_robin_accept_sums and not state.round_robin_attempt_counts:
            state.round_robin_accept_sums = [0] * population_size
            state.round_robin_attempt_counts = [0] * population_size
        if (
            len(state.round_robin_accept_sums) != population_size
            or len(state.round_robin_attempt_counts) != population_size
        ):
            raise DiagESMTPSessionError(
                "MTP diagonal-ES round-robin accounting does not match the "
                f"population size {population_size}"
            )

    def _record_acceptance_round_robin(
        self,
        *,
        state: _MTPSessionState,
        rid: str,
        accepted_drafts: int,
    ) -> dict[str, Any]:
        self._prepare_round_robin_tracking(state)
        population_index = state.population_index
        attempt_count = state.round_robin_attempt_counts[population_index] + 1
        completing_candidate = attempt_count == state.config.attempts_per_candidate
        completing_population = (
            completing_candidate
            and population_index + 1 == state.config.population_size
        )
        self._reserve_event_capacity(
            1 + int(completing_candidate) + int(completing_population)
        )

        state.round_robin_attempt_counts[population_index] = attempt_count
        state.round_robin_accept_sums[population_index] += accepted_drafts
        state.candidate_attempts = attempt_count
        state.candidate_accept_sum = state.round_robin_accept_sums[population_index]
        state.total_attempts += 1
        state.latest_accept_length = accepted_drafts + 1
        state.latest_accepted_drafts = accepted_drafts
        state.latest_attempt_theta_version = state.theta_version
        state.latest_attempt_population_index = population_index
        state.latest_attempt_perturbation_seed = state.perturbation_seed
        self._emit(
            "verify_attempt",
            state,
            rid=rid,
            theta_version=state.theta_version,
            population_index=population_index,
            perturbation_seed=state.perturbation_seed,
            attempt_index=attempt_count,
            accepted_drafts=accepted_drafts,
            accept_length=accepted_drafts + 1,
            total_attempts=state.total_attempts,
        )

        if completing_candidate:
            candidate_reward = (
                state.round_robin_accept_sums[population_index]
                / state.config.attempts_per_candidate
            )
            state.candidate_rewards.append(candidate_reward)
            self._accumulate_noise(state, candidate_reward)
            self._emit(
                "candidate_completed",
                state,
                rid=rid,
                theta_version=state.theta_version,
                population_index=population_index,
                perturbation_seed=state.perturbation_seed,
                attempts=attempt_count,
                candidate_reward_mean=candidate_reward,
                candidate_rewards=list(state.candidate_rewards),
                effective_delta_stats_aggregate=state.effective_delta_stats.get(
                    "aggregate", {}
                ),
            )

        if completing_population:
            return self._finish_population_update(state=state, rid=rid)

        state.population_index = (population_index + 1) % state.config.population_size
        state.candidate_attempts = state.round_robin_attempt_counts[
            state.population_index
        ]
        state.candidate_accept_sum = state.round_robin_accept_sums[
            state.population_index
        ]
        if state.config.sigma != 0.0:
            self._upload_candidate(state)
        return self._session_status(state)

    def _record_acceptance_block_interleaved(
        self,
        *,
        state: _MTPSessionState,
        rid: str,
        accepted_drafts: int,
    ) -> dict[str, Any]:
        """Record one attempt in a two-visit candidate schedule.

        ``schedule_position`` is the zero-based global block ordinal in
        ``[0, 2 * population_size)``; it does not reset for visit 1.
        """

        self._prepare_block_interleaved_tracking(state)
        config = state.config
        dwell_attempts = config.candidate_dwell_attempts
        if dwell_attempts is None:
            raise DiagESMTPSessionError(
                "MTP block-interleaved session is missing dwell attempts"
            )
        order_seed, visit_0, visit_1 = self._block_interleaved_schedule(state)
        visit_index, visit_position = divmod(
            state.block_schedule_position, config.population_size
        )
        order = visit_0 if visit_index == 0 else visit_1
        population_index = order[visit_position]
        if state.population_index != population_index:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES active candidate disagrees with its block schedule"
            )

        block_attempt_index = state.block_attempt_index + 1
        attempt_count = state.block_interleaved_attempt_counts[population_index] + 1
        completing_block = block_attempt_index == dwell_attempts
        completing_candidate = completing_block and visit_index == 1
        completing_population = (
            completing_candidate
            and state.block_schedule_position == 2 * config.population_size - 1
        )
        if block_attempt_index > dwell_attempts:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES block attempt count exceeded its limit"
            )
        if attempt_count > config.attempts_per_candidate:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES candidate attempt count exceeded its limit"
            )
        if completing_candidate != (attempt_count == config.attempts_per_candidate):
            raise DiagESMTPSessionError(
                "MTP diagonal-ES block schedule candidate completion is inconsistent"
            )
        self._reserve_event_capacity(
            1 + int(completing_candidate) + int(completing_population)
        )

        state.block_interleaved_attempt_counts[population_index] = attempt_count
        state.block_interleaved_accept_sums[population_index] += accepted_drafts
        state.block_attempt_index = block_attempt_index
        state.candidate_attempts = attempt_count
        state.candidate_accept_sum = state.block_interleaved_accept_sums[
            population_index
        ]
        state.total_attempts += 1
        state.latest_accept_length = accepted_drafts + 1
        state.latest_accepted_drafts = accepted_drafts
        state.latest_attempt_theta_version = state.theta_version
        state.latest_attempt_population_index = population_index
        state.latest_attempt_perturbation_seed = state.perturbation_seed
        state.latest_attempt_visit_index = visit_index
        state.latest_attempt_block_attempt_index = block_attempt_index
        state.latest_attempt_schedule_position = state.block_schedule_position
        state.latest_attempt_schedule_order_seed = order_seed
        self._emit(
            "verify_attempt",
            state,
            rid=rid,
            theta_version=state.theta_version,
            population_index=population_index,
            perturbation_seed=state.perturbation_seed,
            attempt_index=attempt_count,
            visit_index=visit_index,
            block_attempt_index=block_attempt_index,
            schedule_position=state.block_schedule_position,
            schedule_order_seed=order_seed,
            accepted_drafts=accepted_drafts,
            accept_length=accepted_drafts + 1,
            total_attempts=state.total_attempts,
        )

        if completing_candidate:
            candidate_reward = (
                state.block_interleaved_accept_sums[population_index]
                / config.attempts_per_candidate
            )
            state.candidate_rewards.append(candidate_reward)
            self._accumulate_noise(state, candidate_reward)
            self._emit(
                "candidate_completed",
                state,
                rid=rid,
                theta_version=state.theta_version,
                population_index=population_index,
                perturbation_seed=state.perturbation_seed,
                attempts=attempt_count,
                visit_index=visit_index,
                schedule_position=state.block_schedule_position,
                schedule_order_seed=order_seed,
                candidate_reward_mean=candidate_reward,
                candidate_rewards=list(state.candidate_rewards),
                effective_delta_stats_aggregate=state.effective_delta_stats.get(
                    "aggregate", {}
                ),
            )

        if completing_population:
            return self._finish_population_update(state=state, rid=rid)

        if completing_block:
            self._activate_block_schedule_position(
                state, state.block_schedule_position + 1
            )
            if config.sigma != 0.0:
                self._upload_candidate(state)
        return self._session_status(state)

    def record_acceptance(
        self,
        *,
        session_id: str,
        rid: str,
        accepted_drafts: int,
    ) -> dict[str, Any]:
        reservation = getattr(self, "_active_acceptance_batch", None)
        if reservation is not None:
            cursor = self._active_acceptance_batch_cursor
            if cursor >= len(reservation.attempts) or reservation.attempts[cursor] != (
                session_id,
                rid,
                accepted_drafts,
            ):
                raise DiagESMTPSessionError(
                    "MTP diagonal-ES acceptance call does not match its batch preflight"
                )
        result = self._record_acceptance(
            session_id=session_id,
            rid=rid,
            accepted_drafts=accepted_drafts,
        )
        if reservation is not None:
            self._active_acceptance_batch_cursor += 1
        return result

    def _record_acceptance(
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

        if state.config.candidate_schedule == "round_robin":
            return self._record_acceptance_round_robin(
                state=state,
                rid=rid,
                accepted_drafts=accepted_drafts,
            )
        if state.config.candidate_schedule == "block_interleaved":
            return self._record_acceptance_block_interleaved(
                state=state,
                rid=rid,
                accepted_drafts=accepted_drafts,
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
            effective_delta_stats_aggregate=state.effective_delta_stats.get(
                "aggregate", {}
            ),
        )
        state.candidate_attempts = 0
        state.candidate_accept_sum = 0

        if state.population_index + 1 < state.config.population_size:
            state.population_index += 1
            if state.config.sigma != 0.0:
                self._upload_candidate(state)
            return self._session_status(state)

        return self._finish_population_update(state=state, rid=rid)

    def _validate_event_request_epoch(self, engine_epoch: str) -> None:
        if engine_epoch != self.engine_epoch:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES event engine epoch does not match the active engine"
            )
        if self._active_acceptance_batch is not None:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES events cannot be read or acknowledged during an "
                "active acceptance batch reservation"
            )

    def _event_high_watermark(self) -> int:
        return self._next_event_id - 1

    def _validate_event_queue_bounds(self) -> tuple[int, int]:
        acked = self._acked_through_event_id
        high_watermark = self._event_high_watermark()
        if len(self._events) != high_watermark - acked:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES retained event queue is not contiguous"
            )
        return acked, high_watermark

    def read_events(
        self, *, engine_epoch: str, after_event_id: int, limit: int
    ) -> dict[str, Any]:
        """Read one retained, engine-global event page without releasing it."""

        if type(after_event_id) is not int or after_event_id < 0:
            raise ValueError(
                "MTP diagonal-ES after_event_id must be a non-negative int"
            )
        if type(limit) is not int or not 1 <= limit <= MTP_MAX_EVENT_READ_LIMIT:
            raise ValueError(
                "MTP diagonal-ES event read limit must be an int in "
                f"[1, {MTP_MAX_EVENT_READ_LIMIT}]"
            )
        self._validate_event_request_epoch(engine_epoch)
        acked, high_watermark = self._validate_event_queue_bounds()
        if after_event_id < acked:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES after_event_id precedes acknowledged event history"
            )
        if after_event_id > high_watermark:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES after_event_id exceeds the event high watermark"
            )
        if after_event_id > self._highest_read_through_event_id:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES events must be read contiguously from the "
                "acknowledged frontier"
            )

        offset = after_event_id - acked
        retained_page = self._events[offset : offset + limit]
        for index, event in enumerate(retained_page, start=after_event_id + 1):
            if event.get("event_id") != index:
                raise DiagESMTPSessionError(
                    "MTP diagonal-ES retained event queue is not contiguous"
                )
        events = copy.deepcopy(retained_page)
        read_through_event_id = after_event_id + len(events)
        self._highest_read_through_event_id = max(
            self._highest_read_through_event_id, read_through_event_id
        )
        return {
            "engine_epoch": self.engine_epoch,
            "acked_through_event_id": acked,
            "event_high_watermark": high_watermark,
            "read_through_event_id": read_through_event_id,
            "events": events,
        }

    def ack_events(self, *, engine_epoch: str, through_event_id: int) -> dict[str, Any]:
        """Release the exact retained prefix already exposed by a successful read."""

        if type(through_event_id) is not int or through_event_id < 0:
            raise ValueError(
                "MTP diagonal-ES through_event_id must be a non-negative int"
            )
        self._validate_event_request_epoch(engine_epoch)
        acked, high_watermark = self._validate_event_queue_bounds()
        if through_event_id < acked:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES event acknowledgement is regressive"
            )
        if through_event_id > high_watermark:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES event acknowledgement exceeds the high watermark"
            )
        if through_event_id > self._highest_read_through_event_id:
            raise DiagESMTPSessionError(
                "MTP diagonal-ES events must be read before they are acknowledged"
            )

        release_count = through_event_id - acked
        for index, event in enumerate(self._events[:release_count], start=acked + 1):
            if event.get("event_id") != index:
                raise DiagESMTPSessionError(
                    "MTP diagonal-ES retained event queue is not contiguous"
                )
        del self._events[:release_count]
        self._acked_through_event_id = through_event_id
        return {
            "engine_epoch": self.engine_epoch,
            "acked_through_event_id": self._acked_through_event_id,
            "event_high_watermark": high_watermark,
            "pending_event_count": len(self._events),
        }

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
            "max_event_read_limit": MTP_MAX_EVENT_READ_LIMIT,
            "engine_epoch": self.engine_epoch,
            "acked_through_event_id": self._acked_through_event_id,
            "event_high_watermark": self._event_high_watermark(),
            "highest_read_through_event_id": self._highest_read_through_event_id,
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
            "draft_kv_replay": {
                "batch_count": self._kv_replay_batch_count,
                "transitioned_request_count": (self._kv_replay_transitioned_requests),
                "replayed_rows": self._kv_replayed_rows,
                "enqueue_time_ms": self._kv_replay_enqueue_time_ms,
            },
            "sessions": {
                name: self._session_status(state)
                for name, state in self._sessions.items()
            },
        }

    @classmethod
    def _session_status(cls, state: _MTPSessionState) -> dict[str, Any]:
        config = state.config
        theta_ratios = _theta_ratio_stats(state.theta_stats, sigma=config.sigma)
        block_status: dict[str, Any] = {}
        if config.candidate_schedule == "block_interleaved":
            order_seed, visit_0, visit_1 = cls._block_interleaved_schedule(state)
            block_status = {
                "candidate_dwell_attempts": config.candidate_dwell_attempts,
                "schedule_seed": config.schedule_seed,
                "schedule_lane": config.schedule_lane,
                "schedule_rng_version": MTP_SCHEDULE_RNG_VERSION,
                "schedule_order_seed": order_seed,
                "schedule_position": state.block_schedule_position,
                "visit_index": (
                    state.block_schedule_position // config.population_size
                ),
                "visit_schedule_position": (
                    state.block_schedule_position % config.population_size
                ),
                "block_attempt_index": state.block_attempt_index,
                "candidate_visit_orders": [list(visit_0), list(visit_1)],
                "block_interleaved_attempt_counts": list(
                    state.block_interleaved_attempt_counts
                ),
                "block_interleaved_accept_sums": list(
                    state.block_interleaved_accept_sums
                ),
                "latest_attempt_visit_index": state.latest_attempt_visit_index,
                "latest_attempt_block_attempt_index": (
                    state.latest_attempt_block_attempt_index
                ),
                "latest_attempt_schedule_position": (
                    state.latest_attempt_schedule_position
                ),
                "latest_attempt_schedule_order_seed": (
                    state.latest_attempt_schedule_order_seed
                ),
            }
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
            "candidate_schedule": config.candidate_schedule,
            "estimator": config.estimator,
            "reward_zscore_epsilon": config.reward_zscore_epsilon,
            "max_update_rms_ratio": config.max_update_rms_ratio,
            "max_update_abs_max_ratio": config.max_update_abs_max_ratio,
            "max_theta_rms_ratio": config.max_theta_rms_ratio,
            "max_theta_abs_max_ratio": config.max_theta_abs_max_ratio,
            "theta_version": state.theta_version,
            "update_index": state.theta_version,
            "population_index": state.population_index,
            "candidate_index": state.population_index,
            "perturbation_seed": state.perturbation_seed,
            "effective_candidate_key": (
                "identity"
                if config.sigma == 0.0
                else f"theta={state.theta_version}:seed={state.perturbation_seed}"
            ),
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
            "latest_attempt_population_index": (state.latest_attempt_population_index),
            "latest_attempt_perturbation_seed": (
                state.latest_attempt_perturbation_seed
            ),
            "committed_updates": state.committed_updates,
            "rejected_updates": state.rejected_updates,
            "theta_stats": state.theta_stats,
            **theta_ratios,
            "effective_delta_stats": state.effective_delta_stats,
            **(
                {
                    "round_robin_attempt_counts": list(
                        state.round_robin_attempt_counts
                    ),
                    "round_robin_accept_sums": list(state.round_robin_accept_sums),
                }
                if config.candidate_schedule == "round_robin"
                else {}
            ),
            **block_status,
        }
