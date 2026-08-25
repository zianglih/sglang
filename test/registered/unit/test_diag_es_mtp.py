import asyncio
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
    manager._events = []
    manager._next_event_id = 1
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
    torch.testing.assert_close(
        update["site"], torch.full((2,), -2 * expected_scale)
    )
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
    assert event["proposed_theta_rms_ratio"] == pytest.approx(
        status["theta_rms_ratio"]
    )
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
    update = next(event for event in manager._events if event["event"].startswith("update_"))
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
    manager.record_acceptance(session_id="s", rid="rid-a", accepted_drafts=1)
    manager.record_acceptance(
        session_id="stationary", rid="rid-b", accepted_drafts=2
    )
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
    with pytest.raises(DiagESMTPSessionError, match="global batch event"):
        manager.drain_events("s")
    assert len(manager.drain_events()["events"]) == 4


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

    events = manager.drain_events()["events"]
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

    events = manager.drain_events()["events"]
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

    events = manager.drain_events()["events"]
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

    monkeypatch.setattr(
        diag_es_package, "get_diag_es_mtp_manager", lambda: Manager()
    )
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

    status = manager.record_acceptance(
        session_id="s", rid="rid", accepted_drafts=2
    )

    assert status["population_index"] == 1
    assert status["latest_accepted_drafts"] == 2
    assert status["latest_attempt_theta_version"] == 0
    assert status["latest_attempt_population_index"] == 0
    assert status["latest_attempt_perturbation_seed"] == mtp_candidate_seed(
        1234, 0, 0
    )


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
