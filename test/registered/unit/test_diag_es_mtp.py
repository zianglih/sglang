import asyncio
import copy
import inspect
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import sglang.srt.diag_es.manager as manager_module
import sglang.srt.diag_es.mtp as mtp_module
import torch
from sglang.srt.diag_es.manager import DiagESManager
from sglang.srt.diag_es.mtp import (
    DiagESMTPSessionConfig,
    DiagESMTPSessionError,
    DiagESMTPSessionManager,
    MTP_MAX_EVENT_READ_LIMIT,
    MTP_SESSION_STATE_ABI,
    _delta_stats,
    _MTPSessionState,
    _require_finite_delta_stats,
    mtp_block_interleaved_orders,
    mtp_candidate_seed,
    mtp_normal_for_site,
)
from sglang.srt.diag_es.mtp_kv_replay import mtp_candidate_requires_kv_replay
from sglang.srt.diag_es.manifest import DenseSite
from sglang.srt.diag_es.ops import get_diag_es_post_inputs
from sglang.srt.managers.io_struct import DiagESMTPSessionReqInput
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _state(*, rewards, learning_rate=0.2):
    config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=max(2, len(rewards)),
        sigma=0.1,
        learning_rate=learning_rate,
        attempts_per_candidate=2,
    )
    zero = torch.zeros(2, dtype=torch.float32)
    # Four deterministic illustrative noise vectors have zero sum and a
    # reward-weighted sum [-2, -2] for rewards [0, 1, 2, 3].
    return _MTPSessionState(
        session_id="s",
        resident_slot=1,
        config=config,
        theta_dense={"site": zero.clone()},
        noise_sum_dense={"site": zero.clone()},
        rewarded_noise_sum_dense={"site": torch.tensor([-2.0, -2.0])},
        current_noise_dense={"site": zero.clone()},
        candidate_rewards=list(rewards),
        theta_stats=_delta_stats({"site": zero}),
        effective_delta_stats=_delta_stats({"site": zero}),
    )


def _recording_manager(state):
    manager = DiagESMTPSessionManager.__new__(DiagESMTPSessionManager)
    manager.max_correct_drafts = 3
    manager.max_pending_events = 1_000_000
    manager._initialize_event_protocol()
    manager._sessions = {state.session_id: state}
    manager._next_acceptance_batch_nonce = 1
    manager._active_acceptance_batch = None
    manager._active_acceptance_batch_cursor = 0
    manager._kv_replay_batch_count = 0
    manager._kv_replay_transitioned_requests = 0
    manager._kv_replayed_rows = 0
    manager._kv_replay_enqueue_time_ms = 0.0
    activations = []

    def upload(active_state):
        population_index = active_state.population_index
        noise = torch.full((2,), float(population_index + 1), dtype=torch.float32)
        active_state.current_noise_dense = {"site": noise}
        active_state.effective_delta_stats = _delta_stats(
            {
                "site": active_state.theta_dense["site"].add(
                    noise, alpha=active_state.config.sigma
                )
            }
        )
        activations.append((active_state.theta_version, population_index))

    manager._upload_candidate = upload
    upload(state)
    return manager, activations


def _snapshot_test_manager(*, active_width=None):
    manager = DiagESMTPSessionManager.__new__(DiagESMTPSessionManager)
    manager.manifest = SimpleNamespace(
        schema_id="snapshot-test-v1",
        schema_digest="0123456789abcdef",
        placement="post",
        dense_sites=(DenseSite("site", 2, active_width),),
        grouped_delta_shapes={},
    )
    manager.max_sessions = 2
    manager.model_artifact_id = "snapshot-test-model"
    manager.device = torch.device("cpu")
    manager.max_correct_drafts = 3
    manager.max_pending_events = 1_000_000
    manager.physical_slots = 3
    manager._free_slots = [1, 2]
    manager._sessions = {}
    manager._slot_last_read_events = [None, None, None]
    manager._dense_delta_banks = {"site": torch.zeros(3, 2, dtype=torch.float32)}
    manager._initialize_event_protocol()
    manager._kv_replay_batch_count = 0
    manager._kv_replay_transitioned_requests = 0
    manager._kv_replayed_rows = 0
    manager._kv_replay_enqueue_time_ms = 0.0
    manager._next_acceptance_batch_nonce = 1
    manager._active_acceptance_batch = None
    manager._active_acceptance_batch_cursor = 0

    def upload(state):
        noise, candidate = manager._materialize_candidate(state)
        effective_stats = _delta_stats(candidate)
        _require_finite_delta_stats(effective_stats, name="test candidate")
        manager._dense_delta_banks["site"][state.resident_slot].copy_(candidate["site"])
        state.current_noise_dense = noise
        state.effective_delta_stats = effective_stats

    def clear(slot):
        manager._dense_delta_banks["site"][slot].zero_()
        manager._slot_last_read_events[slot] = None

    manager._upload_candidate = upload
    manager._clear_resident_slot = clear
    return manager


def _advance_snapshot_session(manager, session_id, accepted_drafts):
    manager.bind_request(session_id=session_id, rid="rid")
    for accepted in accepted_drafts:
        manager.record_acceptance(
            session_id=session_id, rid="rid", accepted_drafts=accepted
        )
    manager.release_request(session_id=session_id, rid="rid")


def _read_all_events(manager):
    result = manager.read_events(
        engine_epoch=manager.engine_epoch,
        after_event_id=manager._acked_through_event_id,
        limit=MTP_MAX_EVENT_READ_LIMIT,
    )
    assert result["read_through_event_id"] == result["event_high_watermark"]
    return result["events"]


def _assert_session_snapshots_equal(left, right):
    assert {key: value for key, value in left.items() if key != "tensors"} == {
        key: value for key, value in right.items() if key != "tensors"
    }
    for tensor_map_name in left["tensors"]:
        for site_id, left_tensor in left["tensors"][tensor_map_name].items():
            torch.testing.assert_close(
                left_tensor,
                right["tensors"][tensor_map_name][site_id],
                rtol=0,
                atol=0,
            )


def test_mtp_event_stream_epochs_are_fresh_and_visible_in_status():
    first = _snapshot_test_manager()
    second = _snapshot_test_manager()

    assert first.engine_epoch != second.engine_epoch
    status = first.status()
    assert status["engine_epoch"] == first.engine_epoch
    assert status["acked_through_event_id"] == 0
    assert status["event_high_watermark"] == 0
    assert status["highest_read_through_event_id"] == 0
    assert status["pending_event_count"] == 0
    assert status["max_event_read_limit"] == MTP_MAX_EVENT_READ_LIMIT


def test_mtp_event_read_is_paginated_non_destructive_and_copy_isolated():
    state = _state(rewards=[])
    manager, _ = _recording_manager(state)
    for value in range(3):
        manager._emit("probe", state, payload={"values": [value]})

    first = manager.read_events(
        engine_epoch=manager.engine_epoch, after_event_id=0, limit=2
    )
    assert set(first) == {
        "engine_epoch",
        "acked_through_event_id",
        "event_high_watermark",
        "read_through_event_id",
        "events",
    }
    assert first["acked_through_event_id"] == 0
    assert first["event_high_watermark"] == 3
    assert first["read_through_event_id"] == 2
    assert [event["event_id"] for event in first["events"]] == [1, 2]
    assert len(manager._events) == 3

    first["events"][0]["payload"]["values"].append(99)
    repeated = manager.read_events(
        engine_epoch=manager.engine_epoch, after_event_id=0, limit=2
    )
    assert repeated["events"][0]["payload"] == {"values": [0]}
    assert manager._events[0]["payload"] == {"values": [0]}

    ack = manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=2)
    assert ack == {
        "engine_epoch": manager.engine_epoch,
        "acked_through_event_id": 2,
        "event_high_watermark": 3,
        "pending_event_count": 1,
    }
    assert [event["event_id"] for event in manager._events] == [3]
    assert (
        manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=2) == ack
    )

    final_page = manager.read_events(
        engine_epoch=manager.engine_epoch, after_event_id=2, limit=1
    )
    assert [event["event_id"] for event in final_page["events"]] == [3]
    manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=3)
    assert manager._events == []


def test_mtp_event_read_and_ack_reject_stale_or_invalid_cursors():
    state = _state(rewards=[])
    manager, _ = _recording_manager(state)
    manager._emit("probe", state)
    manager._emit("probe", state)

    with pytest.raises(DiagESMTPSessionError, match="epoch"):
        manager.read_events(engine_epoch="stale", after_event_id=0, limit=1)
    with pytest.raises(DiagESMTPSessionError, match="epoch"):
        manager.ack_events(engine_epoch="stale", through_event_id=0)
    with pytest.raises(ValueError, match="after_event_id"):
        manager.read_events(
            engine_epoch=manager.engine_epoch, after_event_id=True, limit=1
        )
    with pytest.raises(ValueError, match="read limit"):
        manager.read_events(
            engine_epoch=manager.engine_epoch,
            after_event_id=0,
            limit=MTP_MAX_EVENT_READ_LIMIT + 1,
        )
    with pytest.raises(DiagESMTPSessionError, match="high watermark"):
        manager.read_events(
            engine_epoch=manager.engine_epoch, after_event_id=3, limit=1
        )
    with pytest.raises(DiagESMTPSessionError, match="read contiguously"):
        manager.read_events(
            engine_epoch=manager.engine_epoch, after_event_id=1, limit=1
        )
    with pytest.raises(DiagESMTPSessionError, match="must be read"):
        manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=1)

    manager.read_events(engine_epoch=manager.engine_epoch, after_event_id=0, limit=1)
    with pytest.raises(DiagESMTPSessionError, match="read contiguously"):
        manager.read_events(
            engine_epoch=manager.engine_epoch, after_event_id=2, limit=1
        )
    with pytest.raises(DiagESMTPSessionError, match="must be read"):
        manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=2)
    manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=1)
    with pytest.raises(DiagESMTPSessionError, match="regressive"):
        manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=0)
    with pytest.raises(DiagESMTPSessionError, match="acknowledged event history"):
        manager.read_events(
            engine_epoch=manager.engine_epoch, after_event_id=0, limit=1
        )


def test_mtp_event_read_and_ack_reject_active_acceptance_batch():
    state = _state(rewards=[])
    state.active_rid = "rid"
    manager, _ = _recording_manager(state)
    reservation = manager.preflight_acceptance_batch([("s", "rid", 1)])
    manager.record_acceptance(session_id="s", rid="rid", accepted_drafts=1)

    with pytest.raises(DiagESMTPSessionError, match="active acceptance batch"):
        manager.read_events(
            engine_epoch=manager.engine_epoch, after_event_id=0, limit=1
        )
    with pytest.raises(DiagESMTPSessionError, match="active acceptance batch"):
        manager.ack_events(engine_epoch=manager.engine_epoch, through_event_id=0)

    manager.finish_acceptance_batch(reservation)
    assert (
        manager.read_events(
            engine_epoch=manager.engine_epoch, after_event_id=0, limit=1
        )["read_through_event_id"]
        == 1
    )


def test_mtp_event_capacity_is_released_only_by_ack():
    state = _state(rewards=[])
    manager, _ = _recording_manager(state)
    manager.max_pending_events = 1
    manager._emit("first", state)

    page = manager.read_events(
        engine_epoch=manager.engine_epoch, after_event_id=0, limit=1
    )
    with pytest.raises(DiagESMTPSessionError, match="fail-closed limit"):
        manager._emit("second", state)
    manager.ack_events(
        engine_epoch=manager.engine_epoch,
        through_event_id=page["read_through_event_id"],
    )
    manager._emit("second", state)
    assert [event["event_id"] for event in manager._events] == [2]


def test_mtp_session_export_pairs_unchanged_state_abi_with_event_frontier():
    manager = _snapshot_test_manager()
    config = DiagESMTPSessionConfig(seed=99, population_size=2)
    manager.register_session(session_id="session", config=config)
    page = manager.read_events(
        engine_epoch=manager.engine_epoch, after_event_id=0, limit=1
    )
    manager.ack_events(
        engine_epoch=manager.engine_epoch,
        through_event_id=page["read_through_event_id"],
    )

    envelope = manager.export_session_state_with_frontier("session")
    assert set(envelope) == {"session_state", "telemetry_frontier"}
    assert set(envelope["session_state"]) == {
        "state_abi",
        "identity",
        "config",
        "state",
        "tensors",
    }
    assert envelope["session_state"]["state_abi"] == MTP_SESSION_STATE_ABI
    assert envelope["telemetry_frontier"] == {
        "engine_epoch": manager.engine_epoch,
        "event_high_watermark": 1,
    }

    restored = _snapshot_test_manager()
    restored_epoch = restored.engine_epoch
    restored.import_session_state(
        session_id="session", config=config, snapshot=envelope["session_state"]
    )
    assert restored.engine_epoch == restored_epoch
    assert restored.engine_epoch != manager.engine_epoch
    assert restored._next_event_id == 1
    assert restored._events == []


@pytest.mark.parametrize(
    ("config", "accepted_drafts"),
    [
        (
            DiagESMTPSessionConfig(
                seed=101,
                population_size=3,
                sigma=0.1,
                learning_rate=0.01,
                attempts_per_candidate=2,
                max_theta_rms_ratio=100.0,
                max_theta_abs_max_ratio=100.0,
            ),
            [0, 2, 1, 3, 2, 0, 1],
        ),
        (
            DiagESMTPSessionConfig(
                seed=102,
                population_size=3,
                sigma=0.1,
                learning_rate=0.01,
                attempts_per_candidate=2,
                candidate_schedule="round_robin",
            ),
            [0, 1, 2, 3],
        ),
        (
            DiagESMTPSessionConfig(
                seed=103,
                population_size=4,
                sigma=0.1,
                learning_rate=0.01,
                attempts_per_candidate=4,
                candidate_schedule="block_interleaved",
                candidate_dwell_attempts=2,
                schedule_seed=17,
                schedule_lane=1,
            ),
            [0, 1, 2],
        ),
        (
            DiagESMTPSessionConfig(
                seed=104,
                population_size=4,
                sigma=0.1,
                learning_rate=0.01,
                attempts_per_candidate=4,
                candidate_schedule="block_interleaved",
                candidate_dwell_attempts=2,
                schedule_seed=18,
                schedule_lane=2,
            ),
            [0, 1, 2, 3, 1, 0, 2, 1, 3, 2, 0],
        ),
    ],
)
def test_mtp_session_state_round_trip_and_next_attempt_equivalence(
    config, accepted_drafts
):
    uninterrupted = _snapshot_test_manager()
    uninterrupted.register_session(session_id="session", config=config)
    _advance_snapshot_session(uninterrupted, "session", accepted_drafts)
    event_count = len(uninterrupted._events)
    next_event_id = uninterrupted._next_event_id

    snapshot = uninterrupted.export_session_state("session")
    assert snapshot["state_abi"] == MTP_SESSION_STATE_ABI
    assert snapshot["identity"]["dense_sites"] == [
        {"site_id": "site", "width": 2, "active_width": 2}
    ]
    assert snapshot["config"]["max_theta_rms_ratio"] == (config.max_theta_rms_ratio)
    assert len(uninterrupted._events) == event_count
    assert uninterrupted._next_event_id == next_event_id
    snapshot["tensors"]["theta_dense"]["site"].add_(1.0)
    snapshot["state"]["round_robin_accept_sums"][0] += 1
    assert not torch.equal(
        snapshot["tensors"]["theta_dense"]["site"],
        uninterrupted._sessions["session"].theta_dense["site"],
    )
    assert (
        snapshot["state"]["round_robin_accept_sums"]
        != uninterrupted._sessions["session"].round_robin_accept_sums
    )
    snapshot = uninterrupted.export_session_state("session")

    restored = _snapshot_test_manager()
    restored_event_id = restored._next_event_id
    status = restored.import_session_state(
        session_id="session", config=config, snapshot=snapshot
    )
    assert status["state"] == "READY"
    assert restored._events == []
    assert restored._next_event_id == restored_event_id
    assert restored._next_acceptance_batch_nonce == 1
    _assert_session_snapshots_equal(snapshot, restored.export_session_state("session"))
    source_state = uninterrupted._sessions["session"]
    restored_state = restored._sessions["session"]
    snapshot["tensors"]["theta_dense"]["site"].add_(1.0)
    snapshot["state"]["block_interleaved_attempt_counts"][0] += 1
    assert not torch.equal(
        snapshot["tensors"]["theta_dense"]["site"],
        restored_state.theta_dense["site"],
    )
    assert (
        snapshot["state"]["block_interleaved_attempt_counts"]
        != restored_state.block_interleaved_attempt_counts
    )
    torch.testing.assert_close(
        source_state.current_noise_dense["site"],
        restored_state.current_noise_dense["site"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        uninterrupted._dense_delta_banks["site"][source_state.resident_slot],
        restored._dense_delta_banks["site"][restored_state.resident_slot],
        rtol=0,
        atol=0,
    )

    previous_theta_version = source_state.theta_version
    attempts_per_population = config.population_size * config.attempts_per_candidate
    current_progress = source_state.total_attempts % attempts_per_population
    remaining_attempts = attempts_per_population - current_progress
    _advance_snapshot_session(uninterrupted, "session", [2] * remaining_attempts)
    _advance_snapshot_session(restored, "session", [2] * remaining_attempts)
    assert uninterrupted._sessions["session"].theta_version == (
        previous_theta_version + 1
    )
    assert restored._sessions["session"].theta_version == previous_theta_version + 1
    _assert_session_snapshots_equal(
        uninterrupted.export_session_state("session"),
        restored.export_session_state("session"),
    )


def test_mtp_session_state_sigma_zero_restores_exact_identity_without_rng():
    config = DiagESMTPSessionConfig(
        seed=201,
        population_size=2,
        sigma=0.0,
        learning_rate=0.0,
        attempts_per_candidate=2,
    )
    source = _snapshot_test_manager()
    source.register_session(session_id="identity", config=config)
    _advance_snapshot_session(source, "identity", [3, 0, 2])
    snapshot = source.export_session_state("identity")

    restored = _snapshot_test_manager()
    restored._candidate_noise = lambda _seed: pytest.fail(
        "sigma-zero restore materialized noise"
    )
    restored._upload_candidate = lambda _state: pytest.fail(
        "sigma-zero restore uploaded a candidate"
    )
    restored.import_session_state(
        session_id="identity", config=config, snapshot=snapshot
    )
    state = restored._sessions["identity"]
    assert torch.count_nonzero(state.current_noise_dense["site"]) == 0
    assert (
        torch.count_nonzero(restored._dense_delta_banks["site"][state.resident_slot])
        == 0
    )
    _assert_session_snapshots_equal(snapshot, restored.export_session_state("identity"))
    previous_theta_version = source._sessions["identity"].theta_version
    _advance_snapshot_session(source, "identity", [1])
    _advance_snapshot_session(restored, "identity", [1])
    assert source._sessions["identity"].theta_version == previous_theta_version + 1
    _assert_session_snapshots_equal(
        source.export_session_state("identity"),
        restored.export_session_state("identity"),
    )


def test_mtp_session_state_restores_after_rejected_update():
    config = DiagESMTPSessionConfig(
        seed=251,
        population_size=2,
        sigma=0.1,
        learning_rate=0.1,
        attempts_per_candidate=1,
        max_theta_rms_ratio=1e-6,
    )
    source = _snapshot_test_manager()
    source.register_session(session_id="session", config=config)
    _advance_snapshot_session(source, "session", [0, 3])
    state = source._sessions["session"]
    assert state.theta_version == 1
    assert state.committed_updates == 0
    assert state.rejected_updates == 1
    assert torch.count_nonzero(state.theta_dense["site"]) == 0
    assert any(
        event["event"] == "update_rejected"
        and any(
            "max_theta_rms_ratio" in reason
            for reason in event["update_rejection_reasons"]
        )
        for event in source._events
    )

    restored = _snapshot_test_manager()
    restored.import_session_state(
        session_id="session",
        config=config,
        snapshot=source.export_session_state("session"),
    )
    _advance_snapshot_session(source, "session", [3, 0])
    _advance_snapshot_session(restored, "session", [3, 0])
    assert restored._sessions["session"].theta_version == 2
    assert restored._sessions["session"].rejected_updates == 2
    _assert_session_snapshots_equal(
        source.export_session_state("session"),
        restored.export_session_state("session"),
    )


def test_mtp_session_state_requires_quiescent_absent_session():
    config = DiagESMTPSessionConfig(seed=301, population_size=2)
    source = _snapshot_test_manager()
    source.register_session(session_id="session", config=config)
    source.bind_request(session_id="session", rid="rid")
    with pytest.raises(DiagESMTPSessionError, match="live request"):
        source.export_session_state("session")
    source.release_request(session_id="session", rid="rid")

    source.register_session(session_id="other", config=config)
    source.bind_request(session_id="other", rid="other-rid")
    reservation = source.preflight_acceptance_batch([("other", "other-rid", 1)])
    # State export is session-local: an unrelated in-progress verify batch must
    # not starve a completed session's checkpoint while other requests run.
    snapshot = source.export_session_state("session")
    source.record_acceptance(session_id="other", rid="other-rid", accepted_drafts=1)
    source.finish_acceptance_batch(reservation)
    source.release_request(session_id="other", rid="other-rid")

    restored = _snapshot_test_manager()
    restored._active_acceptance_batch = object()
    with pytest.raises(DiagESMTPSessionError, match="acceptance batch"):
        restored.import_session_state(
            session_id="session", config=config, snapshot=snapshot
        )
    restored._active_acceptance_batch = None
    restored.register_session(session_id="session", config=config)
    with pytest.raises(DiagESMTPSessionError, match="already registered"):
        restored.import_session_state(
            session_id="session", config=config, snapshot=snapshot
        )


def test_mtp_session_state_import_rejects_corruption_without_mutation():
    config = DiagESMTPSessionConfig(
        seed=401,
        population_size=3,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=2,
        candidate_schedule="round_robin",
    )
    source = _snapshot_test_manager()
    source.register_session(session_id="session", config=config)
    _advance_snapshot_session(source, "session", [0, 1, 2, 3])
    snapshot = source.export_session_state("session")

    corruptions = []
    bad = copy.deepcopy(snapshot)
    bad["state_abi"] = "unknown"
    corruptions.append((bad, "state_abi"))
    bad = copy.deepcopy(snapshot)
    bad["identity"]["dense_sites"][0]["active_width"] = 1
    corruptions.append((bad, "runtime identity"))
    bad = copy.deepcopy(snapshot)
    bad["config"]["sigma"] = 0.2
    corruptions.append((bad, "config"))
    bad = copy.deepcopy(snapshot)
    bad["state"]["total_attempts"] += 1
    corruptions.append((bad, "total_attempts"))
    bad = copy.deepcopy(snapshot)
    bad["state"]["candidate_rewards"] = tuple(bad["state"]["candidate_rewards"])
    corruptions.append((bad, "candidate_rewards"))
    bad = copy.deepcopy(snapshot)
    bad["tensors"]["theta_dense"]["site"] = bad["tensors"]["theta_dense"]["site"].to(
        torch.float64
    )
    corruptions.append((bad, "FP32"))
    bad = copy.deepcopy(snapshot)
    bad["tensors"]["theta_dense"]["site"] = torch.zeros(3, dtype=torch.float32)
    corruptions.append((bad, "shape"))
    bad = copy.deepcopy(snapshot)
    bad["tensors"]["theta_dense"]["extra"] = torch.zeros(2)
    corruptions.append((bad, "site mismatch"))
    bad = copy.deepcopy(snapshot)
    bad["tensors"]["noise_sum_dense"]["site"][0] = math.inf
    corruptions.append((bad, "non-finite"))
    bad = copy.deepcopy(snapshot)
    bad["tensors"]["rewarded_noise_sum_dense"]["site"][0] += 1.0
    corruptions.append((bad, "inconsistent with completed candidate rewards"))

    for corrupt_snapshot, match in corruptions:
        restored = _snapshot_test_manager()
        free_slots = list(restored._free_slots)
        with pytest.raises(DiagESMTPSessionError, match=match):
            restored.import_session_state(
                session_id="session",
                config=config,
                snapshot=corrupt_snapshot,
            )
        assert restored._sessions == {}
        assert restored._free_slots == free_slots
        assert restored._events == []
        assert restored._next_event_id == 1


def test_mtp_session_state_rejects_unreachable_contiguous_reward():
    config = DiagESMTPSessionConfig(
        seed=451,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=2,
    )
    source = _snapshot_test_manager()
    source.register_session(session_id="session", config=config)
    _advance_snapshot_session(source, "session", [0, 1])
    snapshot = source.export_session_state("session")
    snapshot["state"]["candidate_rewards"][0] = 0.123
    snapshot["tensors"]["rewarded_noise_sum_dense"]["site"].copy_(
        snapshot["tensors"]["noise_sum_dense"]["site"].mul(0.123)
    )

    restored = _snapshot_test_manager()
    with pytest.raises(DiagESMTPSessionError, match="not reachable"):
        restored.import_session_state(
            session_id="session", config=config, snapshot=snapshot
        )
    assert restored._sessions == {}
    assert restored._free_slots == [1, 2]


def test_mtp_session_state_rejects_nonzero_inactive_suffix():
    config = DiagESMTPSessionConfig(seed=475, sigma=0.1)
    source = _snapshot_test_manager(active_width=1)
    source.register_session(session_id="session", config=config)
    snapshot = source.export_session_state("session")
    assert snapshot["identity"]["dense_sites"][0]["active_width"] == 1
    snapshot["tensors"]["theta_dense"]["site"][1] = 0.25

    restored = _snapshot_test_manager(active_width=1)
    with pytest.raises(DiagESMTPSessionError, match="nonzero inactive suffix"):
        restored.import_session_state(
            session_id="session", config=config, snapshot=snapshot
        )
    assert restored._sessions == {}
    assert restored._free_slots == [1, 2]


@pytest.mark.parametrize("failure_phase", ["upload", "status"])
def test_mtp_session_state_failed_restore_cleans_and_returns_slot(failure_phase):
    config = DiagESMTPSessionConfig(
        seed=501,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
    )
    source = _snapshot_test_manager()
    source.register_session(session_id="session", config=config)
    snapshot = source.export_session_state("session")

    restored = _snapshot_test_manager()
    restored._dense_delta_banks["site"][1].fill_(7.0)

    def fail_upload(state):
        restored._dense_delta_banks["site"][state.resident_slot].fill_(9.0)
        raise RuntimeError("upload failed")

    if failure_phase == "upload":
        restored._upload_candidate = fail_upload
    else:
        restored._session_status = lambda _state: (_ for _ in ()).throw(
            RuntimeError("status failed")
        )
    with pytest.raises(RuntimeError, match=f"{failure_phase} failed"):
        restored.import_session_state(
            session_id="session", config=config, snapshot=snapshot
        )
    assert restored._sessions == {}
    assert restored._free_slots == [1, 2]
    assert torch.count_nonzero(restored._dense_delta_banks["site"][1]) == 0
    assert restored._events == []
    assert restored._next_event_id == 1


def test_mtp_session_state_cleanup_failure_quarantines_slot():
    config = DiagESMTPSessionConfig(seed=525, sigma=0.1)
    source = _snapshot_test_manager()
    source.register_session(session_id="session", config=config)
    snapshot = source.export_session_state("session")

    restored = _snapshot_test_manager()

    def fail_upload(state):
        restored._dense_delta_banks["site"][state.resident_slot].fill_(9.0)
        raise RuntimeError("upload failed")

    restored._upload_candidate = fail_upload
    restored._clear_resident_slot = lambda _slot: (_ for _ in ()).throw(
        RuntimeError("clear failed")
    )
    with pytest.raises(DiagESMTPSessionError, match="quarantined"):
        restored.import_session_state(
            session_id="session", config=config, snapshot=snapshot
        )
    assert restored._sessions == {}
    assert restored._free_slots == [2]
    assert torch.count_nonzero(restored._dense_delta_banks["site"][1]) == 2


def test_mtp_session_failed_registration_cleans_slot_before_identity_reuse():
    manager = _snapshot_test_manager()
    original_upload = manager._upload_candidate
    active_config = DiagESMTPSessionConfig(seed=551, sigma=0.1)

    def fail_upload(state):
        manager._dense_delta_banks["site"][state.resident_slot].fill_(9.0)
        raise RuntimeError("registration upload failed")

    manager._upload_candidate = fail_upload
    with pytest.raises(RuntimeError, match="registration upload failed"):
        manager.register_session(session_id="failed", config=active_config)
    assert manager._sessions == {}
    assert manager._free_slots == [1, 2]
    assert torch.count_nonzero(manager._dense_delta_banks["site"][1]) == 0

    manager._upload_candidate = original_upload
    identity_config = DiagESMTPSessionConfig(seed=552, sigma=0.0, learning_rate=0.0)
    status = manager.register_session(session_id="identity", config=identity_config)
    assert status["resident_slot"] == 1
    assert torch.count_nonzero(manager._dense_delta_banks["site"][1]) == 0


def test_mtp_rng_matches_numpy_philox_site_contract():
    seed = mtp_candidate_seed(1234, 5, 7)
    assert seed == 5342915800119454817
    torch.testing.assert_close(
        mtp_normal_for_site(
            seed,
            "model.decoder.self_attn.fused_qkv_a_proj_with_mqa.output",
            (6,),
        ),
        torch.tensor(
            [
                0.18441987037658691,
                -1.937996745109558,
                0.9834115505218506,
                0.414375364780426,
                0.09737052768468857,
                0.9326829314231873,
            ],
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )


def test_population_zscore_update_has_no_inverse_sigma_and_lr_zero_is_control():
    state = _state(rewards=[0.0, 1.0, 2.0, 3.0])
    update, stats = DiagESMTPSessionManager._stage_update(state)
    expected_scale = 0.2 / (4 * (5**0.5 / 2 + 1e-8))
    torch.testing.assert_close(update["site"], torch.full((2,), -2 * expected_scale))
    assert stats["candidate_reward_mean"] == pytest.approx(1.5)

    control, control_stats = DiagESMTPSessionManager._stage_update(
        _state(rewards=[0.0, 1.0, 2.0, 3.0], learning_rate=0.0)
    )
    assert torch.count_nonzero(control["site"]) == 0
    assert control_stats["update_rms"] == 0


def test_equal_candidate_rewards_produce_exact_zero_update():
    update, _ = DiagESMTPSessionManager._stage_update(
        _state(rewards=[2.0, 2.0, 2.0, 2.0])
    )
    assert torch.count_nonzero(update["site"]) == 0


def test_null_theta_trust_region_preserves_unbounded_commit():
    state = _state(rewards=[0.0, 1.0, 2.0, 3.0])
    assert state.config.max_theta_rms_ratio is None
    assert state.config.max_theta_abs_max_ratio is None
    manager, activations = _recording_manager(state)

    status = manager._finish_population_update(state=state, rid="rid")
    event = manager._events[-1]

    assert event["event"] == "update_committed"
    assert event["proposed_theta_rms_ratio"] == pytest.approx(status["theta_rms_ratio"])
    assert event["proposed_theta_abs_max_ratio"] == pytest.approx(
        status["theta_abs_max_ratio"]
    )
    assert event["proposed_theta_stats"] == status["theta_stats"]
    assert status["max_theta_rms_ratio"] is None
    assert status["max_theta_abs_max_ratio"] is None
    assert status["committed_updates"] == 1
    assert status["rejected_updates"] == 0
    assert state.theta_version == 1
    assert state.perturbation_seed == mtp_candidate_seed(1234, 1, 0)
    assert activations == [(0, 0), (1, 0)]


@pytest.mark.parametrize(
    ("limit_field", "ratio_field"),
    [
        ("max_theta_rms_ratio", "proposed_theta_rms_ratio"),
        ("max_theta_abs_max_ratio", "proposed_theta_abs_max_ratio"),
    ],
)
def test_theta_trust_region_rejects_atomically_and_advances_seed(
    limit_field, ratio_field
):
    state = _state(rewards=[0.0, 1.0, 2.0, 3.0])
    state.config = replace(state.config, **{limit_field: 5.0})
    state.theta_dense["site"].fill_(0.45)
    state.theta_stats = _delta_stats(state.theta_dense)
    state.rewarded_noise_sum_dense["site"].mul_(-1)
    initial_theta = state.theta_dense["site"].clone()
    initial_theta_stats = state.theta_stats
    manager, activations = _recording_manager(state)

    status = manager._finish_population_update(state=state, rid="rid")
    event = manager._events[-1]

    assert event["event"] == "update_rejected"
    assert event["update_rms_ratio"] < state.config.max_update_rms_ratio
    assert event["update_abs_max_ratio"] < state.config.max_update_abs_max_ratio
    assert event[ratio_field] > 5.0
    assert event["proposed_theta_stats"]["aggregate"]["rms"] > 0
    assert event["theta_stats"] is initial_theta_stats
    assert len(event["update_rejection_reasons"]) == 1
    assert limit_field in event["update_rejection_reasons"][0]
    torch.testing.assert_close(state.theta_dense["site"], initial_theta)
    assert state.theta_stats is initial_theta_stats
    assert status[limit_field] == 5.0
    assert status["theta_rms_ratio"] == pytest.approx(4.5)
    assert status["theta_abs_max_ratio"] == pytest.approx(4.5)
    assert status["committed_updates"] == 0
    assert status["rejected_updates"] == 1
    assert status["theta_version"] == 1
    assert status["population_index"] == 0
    assert status["perturbation_seed"] == mtp_candidate_seed(1234, 1, 0)
    assert state.candidate_rewards == []
    assert torch.count_nonzero(state.noise_sum_dense["site"]) == 0
    assert torch.count_nonzero(state.rewarded_noise_sum_dense["site"]) == 0
    assert activations == [(0, 0), (1, 0)]


def test_step_limit_rejection_still_reports_finite_proposed_theta():
    state = _state(rewards=[0.0, 1.0, 2.0, 3.0])
    state.config = replace(state.config, max_update_rms_ratio=0.5)
    initial_theta = state.theta_dense["site"].clone()
    manager, _ = _recording_manager(state)

    manager._finish_population_update(state=state, rid="rid")
    event = manager._events[-1]

    assert event["event"] == "update_rejected"
    assert event["update_rms_ratio"] > 0.5
    assert event["proposed_theta_stats"]["aggregate"]["nonfinite_count"] == 0
    assert math.isfinite(event["proposed_theta_rms_ratio"])
    assert math.isfinite(event["proposed_theta_abs_max_ratio"])
    assert "update_rms_ratio" in event["update_rejection_reasons"][0]
    torch.testing.assert_close(state.theta_dense["site"], initial_theta)


def test_nonfinite_update_rejection_reports_proposed_stats_without_mutation():
    state = _state(rewards=[0.0, 1.0, 2.0, 3.0])
    state.rewarded_noise_sum_dense["site"].fill_(math.inf)
    initial_theta = state.theta_dense["site"].clone()
    manager, _ = _recording_manager(state)

    manager._finish_population_update(state=state, rid="rid")
    event = manager._events[-1]

    assert event["event"] == "update_rejected"
    assert event["update_nonfinite_count"] == 2
    assert event["proposed_theta_stats"]["aggregate"]["nonfinite_count"] == 2
    assert math.isnan(event["proposed_theta_rms_ratio"])
    assert math.isnan(event["proposed_theta_abs_max_ratio"])
    assert any(
        "staged update has 2 non-finite values" in reason
        for reason in event["update_rejection_reasons"]
    )
    assert any(
        "proposed theta has non-finite values" in reason
        for reason in event["update_rejection_reasons"]
    )
    torch.testing.assert_close(state.theta_dense["site"], initial_theta)


def test_sigma_zero_lr_zero_is_exact_identity_without_rng_materialization():
    state = _state(rewards=[0.0, 1.0])
    state.config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=2,
        sigma=0.0,
        learning_rate=0.0,
        attempts_per_candidate=1,
    )
    manager = DiagESMTPSessionManager.__new__(DiagESMTPSessionManager)
    manager.manifest = SimpleNamespace(dense_sites=(DenseSite("site", 2),))
    manager._candidate_noise = lambda _seed: pytest.fail(
        "sigma-zero control materialized RNG noise"
    )

    noise, candidate = manager._materialize_candidate(state)
    assert torch.count_nonzero(noise["site"]) == 0
    assert torch.count_nonzero(candidate["site"]) == 0

    update, stats = manager._stage_update(state)
    assert torch.count_nonzero(update["site"]) == 0
    assert stats["update_rms_ratio"] == 0.0
    assert stats["update_abs_max_ratio"] == 0.0


def test_effective_identity_candidate_cycles_do_not_request_kv_replay():
    identity_a = {
        "sigma": 0.0,
        "learning_rate": 0.0,
        "theta_version": 0,
        "perturbation_seed": 1,
    }
    identity_b = {
        **identity_a,
        "theta_version": 9,
        "perturbation_seed": 999,
    }
    assert not mtp_candidate_requires_kv_replay(identity_a, identity_b)

    perturbed_a = {**identity_a, "sigma": 0.01}
    perturbed_b = {**identity_b, "sigma": 0.01}
    assert mtp_candidate_requires_kv_replay(perturbed_a, perturbed_b)


def test_identity_control_cycles_without_candidate_upload():
    state = _state(rewards=[])
    state.config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=2,
        sigma=0.0,
        learning_rate=0.0,
        attempts_per_candidate=1,
        max_theta_rms_ratio=0.1,
        max_theta_abs_max_ratio=0.1,
    )
    state.active_rid = "rid"
    manager, activations = _recording_manager(state)
    activations.clear()
    manager._upload_candidate = lambda _state: pytest.fail(
        "identity candidate entered upload path"
    )

    for accepted_drafts in (1, 2):
        reservation = manager.preflight_acceptance_batch(
            [("s", "rid", accepted_drafts)]
        )
        assert not reservation.expects_kv_replay
        manager.record_acceptance(
            session_id="s", rid="rid", accepted_drafts=accepted_drafts
        )
        manager.finish_acceptance_batch(reservation)

    assert activations == []
    assert state.theta_version == 1
    assert torch.count_nonzero(state.theta_dense["site"]) == 0
    status = manager._session_status(state)
    assert status["theta_rms_ratio"] == 0.0
    assert status["theta_abs_max_ratio"] == 0.0
    update = next(
        event for event in manager._events if event["event"].startswith("update_")
    )
    assert update["event"] == "update_committed"
    assert update["proposed_theta_rms_ratio"] == 0.0
    assert update["proposed_theta_abs_max_ratio"] == 0.0


def test_acceptance_batch_preflights_mixed_replay_event_at_queue_boundary():
    transitioning = _state(rewards=[])
    transitioning.config = DiagESMTPSessionConfig(
        seed=1,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=1,
    )
    transitioning.active_rid = "rid-a"
    stationary = _state(rewards=[])
    stationary.session_id = "stationary"
    stationary.resident_slot = 2
    stationary.config = DiagESMTPSessionConfig(
        seed=2,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=2,
    )
    stationary.active_rid = "rid-b"

    manager, _ = _recording_manager(transitioning)
    manager._sessions[stationary.session_id] = stationary
    attempts = [("s", "rid-a", 1), ("stationary", "rid-b", 2)]
    manager.max_pending_events = 3
    with pytest.raises(DiagESMTPSessionError, match="pending event queue"):
        manager.preflight_acceptance_batch(attempts)
    assert transitioning.total_attempts == 0
    assert stationary.total_attempts == 0
    assert manager._events == []

    manager.max_pending_events = 4
    reservation = manager.preflight_acceptance_batch(attempts)
    assert reservation.acceptance_event_count == 3
    assert reservation.expects_kv_replay
    assert reservation.kv_replay_requests == (("s", "rid-a"),)
    manager.record_acceptance(session_id="s", rid="rid-a", accepted_drafts=1)
    manager.record_acceptance(session_id="stationary", rid="rid-b", accepted_drafts=2)
    manager.record_kv_replay(
        acceptance_batch_reservation=reservation,
        session_ids=["s"],
        request_rows=[5],
        replayed_rows=5,
        enqueue_time_ms=0.25,
    )
    assert [event["event"] for event in manager._events] == [
        "verify_attempt",
        "candidate_completed",
        "verify_attempt",
        "draft_kv_prefix_replay",
    ]
    page = manager.read_events(
        engine_epoch=manager.engine_epoch, after_event_id=0, limit=4
    )
    assert len(page["events"]) == 4
    assert page["events"][-1]["event_scope"] == "global_batch"
    assert page["events"][-1]["request_ids"] == {"s": "rid-a"}
    assert len(manager._events) == 4
    manager.ack_events(
        engine_epoch=manager.engine_epoch,
        through_event_id=page["read_through_event_id"],
    )
    assert manager._events == []


def _prepared_two_session_replay_batch():
    first = _state(rewards=[])
    first.config = DiagESMTPSessionConfig(
        seed=1,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=1,
    )
    first.active_rid = "rid-a"
    second = _state(rewards=[])
    second.session_id = "second"
    second.resident_slot = 2
    second.config = DiagESMTPSessionConfig(
        seed=2,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=1,
    )
    second.active_rid = "rid-b"

    manager, _ = _recording_manager(first)
    manager._sessions[second.session_id] = second
    reservation = manager.preflight_acceptance_batch(
        [("s", "rid-a", 1), ("second", "rid-b", 2)]
    )
    manager.record_acceptance(session_id="s", rid="rid-a", accepted_drafts=1)
    manager.record_acceptance(session_id="second", rid="rid-b", accepted_drafts=2)
    return manager, reservation


def test_kv_replay_event_attributes_reserved_request_ids(monkeypatch):
    monkeypatch.setattr(
        mtp_module,
        "time",
        SimpleNamespace(time=lambda: 123.5, monotonic_ns=lambda: 456),
    )
    manager, reservation = _prepared_two_session_replay_batch()

    manager.record_kv_replay(
        acceptance_batch_reservation=reservation,
        session_ids=["s", "second"],
        request_rows=[5, 7],
        replayed_rows=12,
        enqueue_time_ms=0.25,
    )

    assert manager._events[-1] == {
        "event_id": 5,
        "timestamp": 123.5,
        "monotonic_timestamp_ns": 456,
        "event": "draft_kv_prefix_replay",
        "session_id": None,
        "event_scope": "global_batch",
        "session_ids": ["s", "second"],
        "request_ids": {"s": "rid-a", "second": "rid-b"},
        "request_replayed_rows": {"s": 5, "second": 7},
        "transitioned_request_count": 2,
        "replayed_rows": 12,
        "enqueue_time_ms": 0.25,
    }


def test_kv_replay_rejects_session_order_mismatch_without_mutation():
    manager, reservation = _prepared_two_session_replay_batch()
    events_before = copy.deepcopy(manager._events)
    counters_before = (
        manager._kv_replay_batch_count,
        manager._kv_replay_transitioned_requests,
        manager._kv_replayed_rows,
        manager._kv_replay_enqueue_time_ms,
    )

    with pytest.raises(DiagESMTPSessionError, match="acceptance batch order"):
        manager.record_kv_replay(
            acceptance_batch_reservation=reservation,
            session_ids=["second", "s"],
            request_rows=[7, 5],
            replayed_rows=12,
            enqueue_time_ms=0.25,
        )

    assert manager._events == events_before
    assert (
        manager._kv_replay_batch_count,
        manager._kv_replay_transitioned_requests,
        manager._kv_replayed_rows,
        manager._kv_replay_enqueue_time_ms,
    ) == counters_before
    assert manager._active_acceptance_batch is reservation


def test_delta_stats_are_cpu_cached_and_include_scale_ranges(monkeypatch):
    stats = _delta_stats(
        {
            "a": torch.tensor([0.0, 1.0]),
            "b": torch.tensor([-1.0, 2.0]),
        }
    )
    assert stats["aggregate"] == pytest.approx(
        {
            "count": 4,
            "nonfinite_count": 0,
            "scale_nonfinite_count": 0,
            "rms": math.sqrt(1.5),
            "absmax": 2.0,
            "min": -1.0,
            "max": 2.0,
            "min_scale": 0.0,
            "max_scale": 3.0,
        }
    )
    assert stats["per_site"]["a"]["rms"] == pytest.approx(math.sqrt(0.5))
    assert stats["per_site"]["b"]["min_scale"] == 0.0

    finite_extremes = _delta_stats({"site": torch.tensor([-100.0, 100.0])})
    _require_finite_delta_stats(finite_extremes, name="candidate")
    assert finite_extremes["aggregate"]["min_scale"] == -99.0
    assert finite_extremes["aggregate"]["max_scale"] == 101.0

    nonfinite_stats = _delta_stats({"bad": torch.tensor([float("inf")])})
    with pytest.raises(DiagESMTPSessionError, match="non-finite values"):
        _require_finite_delta_stats(nonfinite_stats, name="candidate")

    state = _state(rewards=[])
    state.theta_stats = stats
    state.effective_delta_stats = stats
    monkeypatch.setattr(
        mtp_module,
        "_delta_stats",
        lambda *_args, **_kwargs: pytest.fail("status recomputed cached stats"),
    )
    status = DiagESMTPSessionManager._session_status(state)
    assert status["theta_stats"] is stats
    assert status["effective_delta_stats"] is stats


def test_round_robin_visits_each_candidate_once_per_round_and_averages_reward():
    state = _state(rewards=[])
    state.config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=3,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=2,
        candidate_schedule="round_robin",
    )
    state.active_rid = "rid"
    state.round_robin_accept_sums = [0, 0, 0]
    state.round_robin_attempt_counts = [0, 0, 0]
    manager, activations = _recording_manager(state)

    intermediate = []
    for accepted_drafts in (0, 1, 2, 2, 3, 0):
        intermediate.append(
            manager.record_acceptance(
                session_id="s", rid="rid", accepted_drafts=accepted_drafts
            )
        )

    events = _read_all_events(manager)
    verify_events = [event for event in events if event["event"] == "verify_attempt"]
    completed_events = [
        event for event in events if event["event"] == "candidate_completed"
    ]
    update_events = [event for event in events if event["event"] == "update_committed"]
    assert [event["population_index"] for event in verify_events] == [
        0,
        1,
        2,
        0,
        1,
        2,
    ]
    assert [event["attempt_index"] for event in verify_events] == [1, 1, 1, 2, 2, 2]
    assert [event["population_index"] for event in completed_events] == [0, 1, 2]
    assert [event["candidate_reward_mean"] for event in completed_events] == [
        1.0,
        2.0,
        1.0,
    ]
    assert update_events[0]["candidate_rewards"] == [1.0, 2.0, 1.0]
    assert [event["event"] for event in events] == [
        "verify_attempt",
        "verify_attempt",
        "verify_attempt",
        "verify_attempt",
        "candidate_completed",
        "verify_attempt",
        "candidate_completed",
        "verify_attempt",
        "candidate_completed",
        "update_committed",
    ]
    assert activations == [(0, 0), (0, 1), (0, 2), (0, 0), (0, 1), (0, 2), (1, 0)]
    assert intermediate[2]["population_index"] == 0
    assert intermediate[2]["candidate_attempts"] == 1
    assert intermediate[2]["round_robin_attempt_counts"] == [1, 1, 1]
    assert intermediate[-1]["theta_version"] == 1
    assert intermediate[-1]["round_robin_attempt_counts"] == [0, 0, 0]


def test_contiguous_default_keeps_candidate_and_event_sequence():
    state = _state(rewards=[])
    state.config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=3,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=2,
    )
    state.active_rid = "rid"
    manager, activations = _recording_manager(state)

    for accepted_drafts in (0, 2, 1, 3, 2, 0):
        manager.record_acceptance(
            session_id="s", rid="rid", accepted_drafts=accepted_drafts
        )

    events = _read_all_events(manager)
    verify_events = [event for event in events if event["event"] == "verify_attempt"]
    assert [event["population_index"] for event in verify_events] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert [event["attempt_index"] for event in verify_events] == [1, 2, 1, 2, 1, 2]
    assert [event["event"] for event in events] == [
        "verify_attempt",
        "verify_attempt",
        "candidate_completed",
        "verify_attempt",
        "verify_attempt",
        "candidate_completed",
        "verify_attempt",
        "verify_attempt",
        "candidate_completed",
        "update_committed",
    ]
    assert activations == [(0, 0), (0, 1), (0, 2), (1, 0)]


def test_block_interleaved_p4_b4_k8_schedule_and_boundary_transitions():
    order_seed, visit_0, visit_1 = mtp_block_interleaved_orders(
        schedule_seed=8,
        theta_version=0,
        population_size=4,
        schedule_lane=0,
    )
    assert order_seed == 17395538561835657799
    assert visit_0 == (0, 1, 2, 3)
    assert visit_1 == (2, 3, 1, 0)
    _, lane_1_visit_0, lane_1_visit_1 = mtp_block_interleaved_orders(
        schedule_seed=8,
        theta_version=0,
        population_size=4,
        schedule_lane=1,
    )
    assert lane_1_visit_0 == (1, 2, 3, 0)
    assert lane_1_visit_1 == (3, 0, 2, 1)

    state = _state(rewards=[])
    state.config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=4,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=8,
        candidate_schedule="block_interleaved",
        candidate_dwell_attempts=4,
        schedule_seed=8,
        schedule_lane=0,
    )
    state.active_rid = "rid"
    state.block_interleaved_accept_sums = [0, 0, 0, 0]
    state.block_interleaved_attempt_counts = [0, 0, 0, 0]
    DiagESMTPSessionManager._activate_block_schedule_position(state, 0)
    manager, activations = _recording_manager(state)

    expected_candidate_blocks = [0, 1, 2, 3, 2, 3, 1, 0]
    for attempt in range(32):
        previous_status = manager._session_status(state)
        reservation = manager.preflight_acceptance_batch([("s", "rid", 1)])
        block_boundary = (attempt + 1) % 4 == 0
        assert reservation.expects_kv_replay is block_boundary
        candidate_completion = block_boundary and attempt >= 16
        population_completion = attempt == 31
        assert reservation.acceptance_event_count == (
            1 + int(candidate_completion) + int(population_completion)
        )
        next_status = manager.record_acceptance(
            session_id="s", rid="rid", accepted_drafts=1
        )
        assert (
            mtp_candidate_requires_kv_replay(previous_status, next_status)
            is block_boundary
        )
        if attempt == 0:
            assert next_status["candidate_dwell_attempts"] == 4
            assert next_status["schedule_seed"] == 8
            assert next_status["schedule_lane"] == 0
            assert next_status["schedule_order_seed"] == order_seed
            assert next_status["candidate_visit_orders"] == [
                [0, 1, 2, 3],
                [2, 3, 1, 0],
            ]
            assert next_status["schedule_position"] == 0
            assert next_status["visit_index"] == 0
            assert next_status["block_attempt_index"] == 1
            assert next_status["latest_attempt_block_attempt_index"] == 1
        if block_boundary:
            manager.record_kv_replay(
                acceptance_batch_reservation=reservation,
                session_ids=["s"],
                request_rows=[5],
                replayed_rows=5,
                enqueue_time_ms=0.1,
            )
        else:
            manager.finish_acceptance_batch(reservation)

    events = _read_all_events(manager)
    verify_events = [event for event in events if event["event"] == "verify_attempt"]
    completed_events = [
        event for event in events if event["event"] == "candidate_completed"
    ]
    update_events = [event for event in events if event["event"] == "update_committed"]
    replay_events = [
        event for event in events if event["event"] == "draft_kv_prefix_replay"
    ]
    assert [event["population_index"] for event in verify_events] == [
        candidate for candidate in expected_candidate_blocks for _ in range(4)
    ]
    assert [event["visit_index"] for event in verify_events] == [0] * 16 + [1] * 16
    assert [event["block_attempt_index"] for event in verify_events] == [
        1,
        2,
        3,
        4,
    ] * 8
    assert [event["schedule_position"] for event in verify_events] == [
        position for position in range(8) for _ in range(4)
    ]
    assert {event["schedule_order_seed"] for event in verify_events} == {order_seed}
    assert [event["attempt_index"] for event in verify_events] == (
        [1, 2, 3, 4] * 4 + [5, 6, 7, 8] * 4
    )
    assert [event["population_index"] for event in completed_events] == [2, 3, 1, 0]
    assert all(event["attempts"] == 8 for event in completed_events)
    assert len(update_events) == 1
    assert update_events[0]["theta_version"] == 0
    assert len(replay_events) == 8
    assert activations[:8] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 2),
        (0, 3),
        (0, 1),
        (0, 0),
    ]
    assert len(activations) == 9
    assert activations[-1][0] == 1
    assert all(
        event["perturbation_seed"]
        == mtp_candidate_seed(1234, 0, event["population_index"])
        for event in verify_events
    )
    assert state.theta_version == 1
    assert state.total_attempts == 32
    assert state.block_schedule_position == 0
    assert state.block_attempt_index == 0
    theta_1_order_seed, theta_1_visit_0, _ = mtp_block_interleaved_orders(
        schedule_seed=8,
        theta_version=1,
        population_size=4,
        schedule_lane=0,
    )
    assert state.block_schedule_order_seed == theta_1_order_seed
    assert state.population_index == theta_1_visit_0[0]


def test_theta_stats_are_recomputed_at_population_update_only(monkeypatch):
    state = _state(rewards=[])
    state.config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=1,
    )
    state.active_rid = "rid"
    manager, _ = _recording_manager(state)
    original_delta_stats = mtp_module._delta_stats
    calls = []

    def record_stats(values):
        calls.append(tuple(values))
        return original_delta_stats(values)

    monkeypatch.setattr(mtp_module, "_delta_stats", record_stats)
    manager.record_acceptance(session_id="s", rid="rid", accepted_drafts=1)
    assert calls == []

    manager.record_acceptance(session_id="s", rid="rid", accepted_drafts=2)
    assert calls == [("site",)]


def test_round_robin_registration_io_and_config_validation():
    request = DiagESMTPSessionReqInput(
        action="register",
        session_id="s",
        seed=1,
        candidate_schedule="round_robin",
    )
    assert request.candidate_schedule == "round_robin"
    assert request.max_theta_rms_ratio is None
    assert request.max_theta_abs_max_ratio is None
    assert (
        DiagESMTPSessionReqInput(
            action="register", session_id="s", seed=1
        ).candidate_schedule
        == "contiguous"
    )
    with pytest.raises(ValueError, match="candidate_schedule"):
        DiagESMTPSessionReqInput(
            action="register",
            session_id="s",
            seed=1,
            candidate_schedule="invalid",
        )
    with pytest.raises(ValueError, match="candidate_schedule"):
        DiagESMTPSessionConfig(seed=1, candidate_schedule="invalid")

    block_request = DiagESMTPSessionReqInput(
        action="register",
        session_id="block",
        seed=1,
        attempts_per_candidate=8,
        candidate_schedule="block_interleaved",
        candidate_dwell_attempts=4,
        schedule_seed=-9,
        schedule_lane=3,
    )
    assert block_request.candidate_dwell_attempts == 4
    assert block_request.schedule_seed == -9
    assert block_request.schedule_lane == 3
    with pytest.raises(ValueError, match="provided together"):
        DiagESMTPSessionReqInput(
            action="register",
            session_id="block",
            seed=1,
            candidate_schedule="block_interleaved",
            schedule_seed=9,
        )
    with pytest.raises(ValueError, match=r"2 \* candidate_dwell_attempts"):
        DiagESMTPSessionConfig(
            seed=1,
            attempts_per_candidate=7,
            candidate_schedule="block_interleaved",
            candidate_dwell_attempts=4,
            schedule_seed=9,
            schedule_lane=0,
        )
    with pytest.raises(ValueError, match="block scheduling fields"):
        DiagESMTPSessionConfig(
            seed=1,
            schedule_seed=9,
            schedule_lane=0,
        )

    positional = DiagESMTPSessionConfig(
        1,
        16,
        0.01,
        0.0,
        4,
        "population_zscore",
        1e-8,
        10.0,
        100.0,
        "round_robin",
        None,
        None,
        None,
    )
    assert positional.candidate_schedule == "round_robin"
    assert positional.max_theta_rms_ratio is None
    assert positional.max_theta_abs_max_ratio is None


@pytest.mark.parametrize("field", ["max_theta_rms_ratio", "max_theta_abs_max_ratio"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan, True, "1"])
def test_theta_trust_region_config_requires_optional_positive_finite_float(
    field, value
):
    with pytest.raises(ValueError, match=field):
        DiagESMTPSessionConfig(seed=1, **{field: value})


def test_engine_api_propagates_round_robin_registration():
    from sglang.srt.entrypoints.engine import Engine

    captured = {}

    class TokenizerManager:
        async def diag_es_mtp_session(self, request):
            captured["request"] = request
            return SimpleNamespace(success=True, message="", status={"ok": True})

    engine = Engine.__new__(Engine)
    engine.tokenizer_manager = TokenizerManager()
    status = asyncio.run(
        engine.async_register_diag_es_mtp_session(
            session_id="s",
            seed=1,
            candidate_schedule="round_robin",
            max_theta_rms_ratio=2.5,
            max_theta_abs_max_ratio=7.5,
        )
    )

    assert status == {"ok": True}
    assert captured["request"].candidate_schedule == "round_robin"
    assert captured["request"].max_theta_rms_ratio == 2.5
    assert captured["request"].max_theta_abs_max_ratio == 7.5

    status = asyncio.run(
        engine.async_register_diag_es_mtp_session(
            session_id="block",
            seed=1,
            attempts_per_candidate=8,
            candidate_schedule="block_interleaved",
            candidate_dwell_attempts=4,
            schedule_seed=88,
            schedule_lane=2,
        )
    )
    assert status == {"ok": True}
    assert captured["request"].candidate_schedule == "block_interleaved"
    assert captured["request"].candidate_dwell_attempts == 4
    assert captured["request"].schedule_seed == 88
    assert captured["request"].schedule_lane == 2
    for method in (
        Engine.register_diag_es_mtp_session,
        Engine.async_register_diag_es_mtp_session,
    ):
        parameters = list(inspect.signature(method).parameters)
        assert parameters.index("schedule_lane") < parameters.index(
            "max_theta_rms_ratio"
        )


def test_tp_worker_propagates_theta_trust_region(monkeypatch):
    import sglang.srt.diag_es as diag_es_package
    from sglang.srt.managers.tp_worker import BaseTpWorker

    captured = {}

    class Manager:
        @staticmethod
        def register_session(*, session_id, config):
            captured["session_id"] = session_id
            captured["config"] = config
            return {"ok": True}

    monkeypatch.setattr(diag_es_package, "get_diag_es_mtp_manager", lambda: Manager())
    request = DiagESMTPSessionReqInput(
        action="register",
        session_id="s",
        seed=1,
        max_theta_rms_ratio=2.5,
        max_theta_abs_max_ratio=7.5,
    )

    status = BaseTpWorker.diag_es_mtp_session(object(), request)

    assert status == {"ok": True}
    assert captured["session_id"] == "s"
    assert captured["config"].max_theta_rms_ratio == 2.5
    assert captured["config"].max_theta_abs_max_ratio == 7.5


def test_target_and_mtp_managers_are_role_keyed(monkeypatch):
    target = DiagESManager.__new__(DiagESManager)
    mtp = DiagESMTPSessionManager.__new__(DiagESMTPSessionManager)
    monkeypatch.setattr(manager_module, "_target_manager", target)
    monkeypatch.setattr(manager_module, "_mtp_manager", mtp)

    assert manager_module.get_diag_es_manager() is target
    assert manager_module.get_diag_es_mtp_manager() is mtp


def test_latest_acceptance_keeps_attempt_candidate_identity_after_rollover():
    manager = DiagESMTPSessionManager.__new__(DiagESMTPSessionManager)
    state = _state(rewards=[])
    state.config = DiagESMTPSessionConfig(
        seed=1234,
        population_size=2,
        sigma=0.1,
        learning_rate=0.0,
        attempts_per_candidate=1,
    )
    state.active_rid = "rid"
    state.current_noise_dense = {"site": torch.zeros(2)}
    manager.max_correct_drafts = 3
    manager._sessions = {"s": state}
    manager._events = []
    manager.max_pending_events = 1_000_000
    manager._next_event_id = 1
    manager._upload_candidate = lambda _state: None

    status = manager.record_acceptance(session_id="s", rid="rid", accepted_drafts=2)

    assert status["population_index"] == 1
    assert status["latest_accepted_drafts"] == 2
    assert status["latest_attempt_theta_version"] == 0
    assert status["latest_attempt_population_index"] == 0
    assert status["latest_attempt_perturbation_seed"] == mtp_candidate_seed(1234, 0, 0)


def test_event_cap_fails_before_attempt_state_changes_or_drops_existing_events():
    manager = DiagESMTPSessionManager.__new__(DiagESMTPSessionManager)
    state = _state(rewards=[])
    state.active_rid = "rid"
    manager.max_correct_drafts = 3
    manager.max_pending_events = 1
    existing = {"event_id": 1, "event": "verify_attempt", "session_id": "old"}
    manager._events = [existing]
    manager._next_event_id = 2
    manager._sessions = {"s": state}

    with pytest.raises(DiagESMTPSessionError, match="fail-closed limit"):
        manager.record_acceptance(session_id="s", rid="rid", accepted_drafts=1)

    assert manager._events == [existing]
    assert state.total_attempts == 0
    assert state.candidate_attempts == 0
    assert state.candidate_accept_sum == 0


def test_release_request_is_exact_and_idempotent_for_same_rid_only():
    manager = DiagESMTPSessionManager.__new__(DiagESMTPSessionManager)
    state = _state(rewards=[])
    state.active_rid = "rid"
    manager._sessions = {"s": state}

    manager.release_request(session_id="s", rid="rid")
    manager.release_request(session_id="s", rid="rid")
    assert state.active_rid is None

    with pytest.raises(DiagESMTPSessionError, match="not bound"):
        manager.release_request(session_id="s", rid="other")
    with pytest.raises(DiagESMTPSessionError, match="not registered"):
        manager.release_request(session_id="missing", rid="rid")


def test_active_post_bank_without_candidate_slots_fails_closed():
    layer = SimpleNamespace(es_post_delta_bank=torch.zeros(2, 3))
    with forward_context(ForwardContext(attn_backend=None)):
        with pytest.raises(RuntimeError, match="candidate slots are missing"):
            get_diag_es_post_inputs(layer)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 1},
        {"attempts_per_candidate": 0},
        {"sigma": 0.0, "learning_rate": 0.1},
        {"learning_rate": -1.0},
        {"estimator": "loo"},
    ],
)
def test_mtp_session_config_fails_closed(kwargs):
    with pytest.raises(ValueError):
        DiagESMTPSessionConfig(seed=1, **kwargs)
