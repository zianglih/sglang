from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sglang.srt.diag_es import (
    DiagESCandidateNotFoundError,
    DiagESCandidateRetiringError,
    DiagESInvalidCandidateError,
    DiagESNotEnabledError,
)
from sglang.srt.diag_es.manager import (
    DiagESManager,
    compose_diag_es_mtp_request_extra_key,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid, _input_text, _input_ids, _sampling_params, **kwargs):
        self.rid = rid
        self.es_candidate_id = kwargs.get("es_candidate_id")
        self.es_candidate_released = True
        self.es_candidate_slot = 0
        self.diag_es_mtp_session_id = kwargs.get("diag_es_mtp_session_id")
        self.diag_es_mtp_session_released = True
        self.diag_es_mtp_slot = 0
        self.diag_es_mtp_status = None
        self.extra_key = kwargs.get("extra_key")
        self.finished_reason = None
        self.tokenizer = None

    def set_finish_with_abort(self, message):
        self.finished_reason = message


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._diag_es_mtp_request_bind_generation = 0
    scheduler.model_config = SimpleNamespace(vocab_size=100, hf_eos_token_id=2)
    scheduler.tokenizer = object()
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.dllm_config = None
    scheduler.init_req_max_new_tokens = Mock()
    scheduler._add_request_to_queue = Mock()
    return scheduler


def _request(**overrides):
    values = dict(
        rid="rid",
        input_text="hello",
        input_ids=[1],
        sampling_params=object(),
        es_candidate_id="missing",
        diag_es_mtp_session_id=None,
        session_params=None,
        session_id=None,
        input_embeds=None,
        bootstrap_port=123,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        return_flat_raw_top_logprobs=False,
        stream=False,
        lora_id=None,
        positional_embed_overrides=None,
        token_type_ids=None,
        custom_logit_processor=None,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        return_indexer_topk=False,
        bootstrap_host=None,
        bootstrap_room=None,
        routed_dp_rank=None,
        disagg_prefill_dp_rank=None,
        priority=0,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
        multi_item_delimiter_indices=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "session_fields",
    [
        {"session_params": SimpleNamespace(id="legacy")},
        {"session_id": "radix-native"},
    ],
)
def test_diag_es_sessions_fail_before_session_state_is_touched(session_fields):
    scheduler = _scheduler()
    recv_req = _request(**session_fields)

    with patch.object(scheduler_module, "Req", _FakeReq):
        scheduler.handle_generate_request(recv_req)

    queued_req = scheduler._add_request_to_queue.call_args.args[0]
    assert "does not use SGLang KV sessions" in queued_req.finished_reason
    assert queued_req.es_candidate_released
    scheduler.init_req_max_new_tokens.assert_called_once_with(queued_req)


@pytest.mark.parametrize(
    "candidate_error",
    [
        DiagESCandidateNotFoundError("unknown candidate"),
        DiagESCandidateRetiringError("retiring candidate"),
        DiagESNotEnabledError("diagonal ES is not enabled"),
    ],
)
def test_candidate_admission_errors_are_request_local(monkeypatch, candidate_error):
    scheduler = _scheduler()
    recv_req = _request()
    manager = SimpleNamespace(acquire=Mock(side_effect=candidate_error))

    import sglang.srt.diag_es as diag_es

    monkeypatch.setattr(diag_es, "get_diag_es_manager", lambda: manager)
    with patch.object(scheduler_module, "Req", _FakeReq):
        scheduler.handle_generate_request(recv_req)

    queued_req = scheduler._add_request_to_queue.call_args.args[0]
    assert queued_req.finished_reason == str(candidate_error)
    assert queued_req.es_candidate_released
    scheduler.init_req_max_new_tokens.assert_called_once_with(queued_req)


@pytest.mark.parametrize("candidate_id", ["", "   "])
def test_blank_candidate_id_is_request_local(monkeypatch, candidate_id):
    scheduler = _scheduler()
    recv_req = _request(es_candidate_id=candidate_id)
    manager = DiagESManager.__new__(DiagESManager)
    with pytest.raises(DiagESInvalidCandidateError):
        manager.acquire(candidate_id)

    import sglang.srt.diag_es as diag_es

    monkeypatch.setattr(diag_es, "get_diag_es_manager", lambda: manager)
    with patch.object(scheduler_module, "Req", _FakeReq):
        scheduler.handle_generate_request(recv_req)

    queued_req = scheduler._add_request_to_queue.call_args.args[0]
    assert queued_req.es_candidate_released
    assert "non-empty" in queued_req.finished_reason


def test_mtp_admission_binds_slot_and_request_private_cache_key(monkeypatch):
    scheduler = _scheduler()
    req = _FakeReq(
        "rid", "hello", [1], object(), extra_key="tenant", es_candidate_id="main"
    )
    status = {"resident_slot": 7, "population_index": 3}
    manager = SimpleNamespace(bind_request=Mock(return_value=status))

    import sglang.srt.diag_es as diag_es

    monkeypatch.setattr(diag_es, "get_diag_es_mtp_manager", lambda: manager)
    scheduler._bind_diag_es_mtp_request(req, "mtp-session")

    manager.bind_request.assert_called_once_with(
        session_id="mtp-session", rid="rid"
    )
    assert req.diag_es_mtp_slot == 7
    assert req.es_candidate_slot == 0
    assert req.extra_key == compose_diag_es_mtp_request_extra_key(
        "tenant", session_id="mtp-session", rid="rid", bind_generation=1
    )
    assert not req.diag_es_mtp_session_released


def test_mtp_request_private_cache_key_changes_on_reused_rid_binding():
    key = compose_diag_es_mtp_request_extra_key(
        "tenant", session_id="session-a", rid="rid-a", bind_generation=1
    )
    assert key == compose_diag_es_mtp_request_extra_key(
        "tenant", session_id="session-a", rid="rid-a", bind_generation=1
    )
    assert key != compose_diag_es_mtp_request_extra_key(
        "tenant", session_id="session-a", rid="rid-a", bind_generation=2
    )
    assert key != compose_diag_es_mtp_request_extra_key(
        "tenant", session_id="session-a", rid="rid-b", bind_generation=1
    )
    assert key != compose_diag_es_mtp_request_extra_key(
        "tenant", session_id="session-b", rid="rid-a", bind_generation=1
    )


def test_mtp_rebinding_same_rid_gets_a_fresh_private_cache_namespace(monkeypatch):
    scheduler = _scheduler()
    manager = SimpleNamespace(
        bind_request=Mock(return_value={"resident_slot": 7, "population_index": 0})
    )

    import sglang.srt.diag_es as diag_es

    monkeypatch.setattr(diag_es, "get_diag_es_mtp_manager", lambda: manager)
    first = _FakeReq("rid", "hello", [1], object(), extra_key="tenant")
    second = _FakeReq("rid", "hello", [1], object(), extra_key="tenant")

    scheduler._bind_diag_es_mtp_request(first, "mtp-session")
    scheduler._bind_diag_es_mtp_request(second, "mtp-session")

    assert first.extra_key != second.extra_key
    assert scheduler._diag_es_mtp_request_bind_generation == 2


def test_mtp_id_on_clean_server_is_request_local(monkeypatch):
    scheduler = _scheduler()
    recv_req = _request(
        es_candidate_id=None, diag_es_mtp_session_id="mtp-session"
    )

    import sglang.srt.diag_es as diag_es

    def disabled():
        raise DiagESNotEnabledError("MTP diagonal ES is not enabled")

    monkeypatch.setattr(diag_es, "get_diag_es_mtp_manager", disabled)
    with patch.object(scheduler_module, "Req", _FakeReq):
        scheduler.handle_generate_request(recv_req)

    queued_req = scheduler._add_request_to_queue.call_args.args[0]
    assert queued_req.finished_reason == "MTP diagonal ES is not enabled"
    assert queued_req.diag_es_mtp_session_released


@pytest.mark.parametrize("session_id", ["", "   ", "bad\0id"])
def test_invalid_mtp_session_id_is_request_local(monkeypatch, session_id):
    scheduler = _scheduler()
    recv_req = _request(
        es_candidate_id=None, diag_es_mtp_session_id=session_id
    )

    class _Manager:
        @staticmethod
        def bind_request(*, session_id, rid):
            from sglang.srt.diag_es.mtp import DiagESMTPSessionManager

            DiagESMTPSessionManager._validate_session_id(session_id)

    import sglang.srt.diag_es as diag_es

    monkeypatch.setattr(diag_es, "get_diag_es_mtp_manager", lambda: _Manager())
    with patch.object(scheduler_module, "Req", _FakeReq):
        scheduler.handle_generate_request(recv_req)

    queued_req = scheduler._add_request_to_queue.call_args.args[0]
    assert "non-empty string without NUL bytes" in queued_req.finished_reason
    assert queued_req.diag_es_mtp_session_released
