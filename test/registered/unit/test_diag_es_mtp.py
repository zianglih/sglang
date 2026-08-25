import asyncio
import math
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
        )
    )

    assert status == {"ok": True}
    assert captured["request"].candidate_schedule == "round_robin"


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
