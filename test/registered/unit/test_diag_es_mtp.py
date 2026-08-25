from types import SimpleNamespace

import pytest
import sglang.srt.diag_es.manager as manager_module
import torch
from sglang.srt.diag_es.manager import DiagESManager
from sglang.srt.diag_es.mtp import (
    DiagESMTPSessionConfig,
    DiagESMTPSessionError,
    DiagESMTPSessionManager,
    _MTPSessionState,
    mtp_candidate_seed,
    mtp_normal_for_site,
)
from sglang.srt.diag_es.ops import get_diag_es_post_inputs
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
    )


def test_mtp_rng_matches_numpy_philox_site_contract():
    seed = mtp_candidate_seed(1234, 5, 7)
    assert seed == 5342915800119454817
    torch.testing.assert_close(
        mtp_normal_for_site(
            seed, "model.decoder.self_attn.q_a_proj.output", (6,)
        ),
        torch.tensor(
            [
                -0.8728805184364319,
                -1.3948218822479248,
                -0.6058406829833984,
                -1.2000948190689087,
                0.33840417861938477,
                1.4031062126159668,
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
        {"sigma": 0.0},
        {"learning_rate": -1.0},
        {"estimator": "loo"},
    ],
)
def test_mtp_session_config_fails_closed(kwargs):
    with pytest.raises(ValueError):
        DiagESMTPSessionConfig(seed=1, **kwargs)
