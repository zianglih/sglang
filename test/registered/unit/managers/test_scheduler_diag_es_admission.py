from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.diag_es import (
    DiagESCandidateNotFoundError,
    DiagESCandidateRetiringError,
    DiagESInvalidCandidateError,
    DiagESNotEnabledError,
)
from sglang.srt.diag_es.manager import DiagESManager
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
        self.finished_reason = None
        self.tokenizer = None

    def set_finish_with_abort(self, message):
        self.finished_reason = message


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
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
    assert "does not support sessions" in queued_req.finished_reason
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
