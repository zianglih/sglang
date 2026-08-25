import asyncio
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.io_struct import (
    DiagESMTPSessionReqInput,
    DiagESMTPSessionReqOutput,
    PickleWrapper,
    msgpack_decode,
    msgpack_encode,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _snapshot():
    return {
        "state_abi": "joyai-mtp-session-state-v1",
        "identity": {"session_id": "session"},
        "config": {"seed": 7},
        "state": {"theta_version": 3},
        "tensors": {"theta_dense": {"site": torch.tensor([1.0, -2.0])}},
    }


def _export_envelope():
    return {
        "session_state": _snapshot(),
        "telemetry_frontier": {
            "engine_epoch": "engine-epoch",
            "event_high_watermark": 11,
        },
    }


def _event_page():
    return {
        "engine_epoch": "engine-epoch",
        "acked_through_event_id": 0,
        "event_high_watermark": 2,
        "read_through_event_id": 2,
        "events": [
            {
                "event_id": 1,
                "timestamp": 1.25,
                "monotonic_timestamp_ns": 123456,
                "event": "candidate_completed",
                "session_id": "session",
                "perturbation_seed": 2**64 - 1,
                "candidate_rewards": [1.0, 2.0],
                "effective_delta_stats_aggregate": {
                    "count": 4,
                    "nonfinite_count": 0,
                    "min": -0.25,
                    "max": 0.5,
                },
                "optional": None,
            },
            {
                "event_id": 2,
                "timestamp": 2.5,
                "monotonic_timestamp_ns": 123457,
                "event": "verify_attempt",
                "session_id": "session",
                "accepted_drafts": 3,
                "accept_length": 4,
            },
        ],
    }


def _materialized_state(request):
    return unwrap_from_pickle(request.session_state)


def test_state_request_contract_and_msgpack_round_trip():
    snapshot = _snapshot()
    request = DiagESMTPSessionReqInput(
        action="import_state",
        session_id="session",
        session_state=snapshot,
        seed=7,
    )

    if isinstance(request.session_state, PickleWrapper):
        assert request.session_state.data
    restored_request = msgpack_decode(msgpack_encode(request))
    restored = _materialized_state(restored_request)
    assert restored["state"]["theta_version"] == 3
    restored_tensor = restored["tensors"]["theta_dense"]["site"]
    assert restored_tensor.device.type == "cpu"
    assert restored_tensor.dtype == torch.float32
    assert restored_tensor.is_contiguous()
    assert (
        restored_tensor.data_ptr()
        != snapshot["tensors"]["theta_dense"]["site"].data_ptr()
    )
    torch.testing.assert_close(
        restored_tensor,
        snapshot["tensors"]["theta_dense"]["site"],
    )

    with pytest.raises(ValueError, match="requires session_id"):
        DiagESMTPSessionReqInput(action="export_state")
    with pytest.raises(ValueError, match="requires a session_state mapping"):
        DiagESMTPSessionReqInput(
            action="import_state",
            session_id="session",
            session_state=["not", "a", "mapping"],
            seed=7,
        )
    with pytest.raises(ValueError, match="does not accept session_state"):
        DiagESMTPSessionReqInput(
            action="register",
            session_id="session",
            session_state=snapshot,
            seed=7,
        )


def test_event_request_contract_rejects_invalid_or_ambiguous_forms():
    read_request = DiagESMTPSessionReqInput(
        action="read_events",
        engine_epoch="engine-epoch",
        after_event_id=0,
        event_limit=4096,
    )
    assert read_request.after_event_id == 0
    assert read_request.event_limit == 4096
    restored_read_request = msgpack_decode(msgpack_encode(read_request))
    assert restored_read_request.engine_epoch == "engine-epoch"
    assert restored_read_request.after_event_id == 0
    assert restored_read_request.event_limit == 4096

    ack_request = DiagESMTPSessionReqInput(
        action="ack_events",
        engine_epoch="engine-epoch",
        through_event_id=0,
    )
    assert ack_request.through_event_id == 0
    restored_ack_request = msgpack_decode(msgpack_encode(ack_request))
    assert restored_ack_request.engine_epoch == "engine-epoch"
    assert restored_ack_request.through_event_id == 0

    with pytest.raises(ValueError, match="engine-global"):
        DiagESMTPSessionReqInput(
            action="read_events",
            session_id="session",
            engine_epoch="engine-epoch",
            after_event_id=0,
            event_limit=1,
        )
    with pytest.raises(ValueError, match="non-empty engine_epoch"):
        DiagESMTPSessionReqInput(
            action="read_events",
            engine_epoch="",
            after_event_id=0,
            event_limit=1,
        )
    with pytest.raises(ValueError, match="non-negative after_event_id"):
        DiagESMTPSessionReqInput(
            action="read_events",
            engine_epoch="engine-epoch",
            after_event_id=True,
            event_limit=1,
        )
    with pytest.raises(ValueError, match="positive event_limit"):
        DiagESMTPSessionReqInput(
            action="read_events",
            engine_epoch="engine-epoch",
            after_event_id=0,
            event_limit=0,
        )
    with pytest.raises(ValueError, match="does not accept through_event_id"):
        DiagESMTPSessionReqInput(
            action="read_events",
            engine_epoch="engine-epoch",
            after_event_id=0,
            event_limit=1,
            through_event_id=0,
        )
    with pytest.raises(ValueError, match="non-negative through_event_id"):
        DiagESMTPSessionReqInput(
            action="ack_events",
            engine_epoch="engine-epoch",
            through_event_id=False,
        )
    with pytest.raises(ValueError, match="does not accept read cursor fields"):
        DiagESMTPSessionReqInput(
            action="ack_events",
            engine_epoch="engine-epoch",
            through_event_id=0,
            after_event_id=0,
        )


def test_event_page_survives_typed_status_msgpack_round_trip():
    page = _event_page()
    output = DiagESMTPSessionReqOutput(
        success=True,
        message="",
        status=page,
    )
    restored = msgpack_decode(msgpack_encode(output))

    assert restored.status == page
    assert isinstance(restored.status["events"], list)
    assert isinstance(restored.status["events"][0], dict)
    assert restored.status["events"][0]["perturbation_seed"] == 2**64 - 1
    assert restored.session_state_export is None


def test_engine_event_apis_preserve_cursors_and_status():
    from sglang.srt.entrypoints.engine import Engine

    page = _event_page()
    ack_status = {
        "engine_epoch": "engine-epoch",
        "acked_through_event_id": 2,
        "event_high_watermark": 2,
        "pending_event_count": 0,
    }
    requests = []

    class TokenizerManager:
        async def diag_es_mtp_session(self, request):
            requests.append(request)
            status = page if request.action == "read_events" else ack_status
            return SimpleNamespace(
                success=True,
                message="",
                status=status,
                session_state_export=None,
            )

    engine = Engine.__new__(Engine)
    engine.tokenizer_manager = TokenizerManager()

    assert (
        asyncio.run(
            engine.async_read_diag_es_mtp_events(
                engine_epoch="engine-epoch",
                after_event_id=0,
                limit=2,
            )
        )
        == page
    )
    assert (
        asyncio.run(
            engine.async_ack_diag_es_mtp_events(
                engine_epoch="engine-epoch",
                through_event_id=2,
            )
        )
        == ack_status
    )

    class Loop:
        @staticmethod
        def run_until_complete(awaitable):
            return asyncio.run(awaitable)

    engine.loop = Loop()
    assert engine.read_diag_es_mtp_events("engine-epoch", 0, 2) == page
    assert engine.ack_diag_es_mtp_events("engine-epoch", 2) == ack_status

    assert [request.action for request in requests] == [
        "read_events",
        "ack_events",
        "read_events",
        "ack_events",
    ]
    read_request, ack_request = requests[:2]
    assert read_request.action == "read_events"
    assert read_request.engine_epoch == "engine-epoch"
    assert read_request.after_event_id == 0
    assert read_request.event_limit == 2
    assert read_request.through_event_id is None
    assert ack_request.action == "ack_events"
    assert ack_request.engine_epoch == "engine-epoch"
    assert ack_request.through_event_id == 2
    assert ack_request.after_event_id is None
    assert ack_request.event_limit is None


@pytest.mark.parametrize("action", ["read_events", "ack_events"])
def test_engine_event_apis_propagate_control_failure(action):
    from sglang.srt.entrypoints.engine import Engine

    class TokenizerManager:
        async def diag_es_mtp_session(self, _request):
            return SimpleNamespace(
                success=False,
                message="event cursor rejected",
                status={},
                session_state_export=None,
            )

    engine = Engine.__new__(Engine)
    engine.tokenizer_manager = TokenizerManager()
    with pytest.raises(RuntimeError, match="event cursor rejected"):
        if action == "read_events":
            asyncio.run(
                engine.async_read_diag_es_mtp_events(
                    engine_epoch="engine-epoch",
                    after_event_id=0,
                    limit=1,
                )
            )
        else:
            asyncio.run(
                engine.async_ack_diag_es_mtp_events(
                    engine_epoch="engine-epoch",
                    through_event_id=0,
                )
            )


def test_engine_export_and_import_state_apis_preserve_mapping_and_config():
    from sglang.srt.entrypoints.engine import Engine

    envelope = _export_envelope()
    snapshot = envelope["session_state"]
    requests = []

    class TokenizerManager:
        async def diag_es_mtp_session(self, request):
            requests.append(request)
            if request.action == "export_state":
                return SimpleNamespace(
                    success=True,
                    message="",
                    status={},
                    session_state_export=wrap_as_pickle(envelope),
                )
            return SimpleNamespace(
                success=True,
                message="",
                status={"session_id": request.session_id, "theta_version": 3},
                session_state_export=None,
            )

    engine = Engine.__new__(Engine)
    engine.tokenizer_manager = TokenizerManager()

    exported_envelope = asyncio.run(
        engine.async_export_diag_es_mtp_session_state("session")
    )
    assert exported_envelope is not envelope
    assert exported_envelope["telemetry_frontier"] == {
        "engine_epoch": "engine-epoch",
        "event_high_watermark": 11,
    }
    exported = exported_envelope["session_state"]
    assert exported is not snapshot
    torch.testing.assert_close(
        exported["tensors"]["theta_dense"]["site"],
        snapshot["tensors"]["theta_dense"]["site"],
    )

    status = asyncio.run(
        engine.async_import_diag_es_mtp_session_state(
            session_id="session",
            session_state=exported,
            seed=7,
            population_size=8,
            sigma=0.02,
            learning_rate=0.001,
            attempts_per_candidate=6,
            candidate_schedule="block_interleaved",
            candidate_dwell_attempts=3,
            schedule_seed=101,
            schedule_lane=2,
            max_theta_rms_ratio=0.5,
            max_theta_abs_max_ratio=3.0,
        )
    )
    assert status == {"session_id": "session", "theta_version": 3}
    import_request = requests[-1]
    assert import_request.action == "import_state"
    assert import_request.population_size == 8
    assert import_request.candidate_schedule == "block_interleaved"
    assert import_request.max_theta_rms_ratio == 0.5
    assert import_request.max_theta_abs_max_ratio == 3.0
    torch.testing.assert_close(
        _materialized_state(import_request)["tensors"]["theta_dense"]["site"],
        snapshot["tensors"]["theta_dense"]["site"],
    )


def test_engine_state_api_propagates_control_failure():
    from sglang.srt.entrypoints.engine import Engine

    class TokenizerManager:
        async def diag_es_mtp_session(self, _request):
            return SimpleNamespace(
                success=False,
                message="state boundary is busy",
                status={},
                session_state_export=None,
            )

    engine = Engine.__new__(Engine)
    engine.tokenizer_manager = TokenizerManager()
    with pytest.raises(RuntimeError, match="state boundary is busy"):
        asyncio.run(engine.async_export_diag_es_mtp_session_state("session"))


def test_tp_worker_dispatches_state_calls_at_tp1(monkeypatch):
    import sglang.srt.diag_es as diag_es_package
    from sglang.srt.managers.tp_worker import BaseTpWorker

    envelope = _export_envelope()
    snapshot = envelope["session_state"]
    calls = []

    class Manager:
        def export_session_state_with_frontier(self, session_id):
            calls.append(("export", session_id))
            return envelope

        def import_session_state(self, *, session_id, config, snapshot):
            calls.append(("import", session_id, config, snapshot))
            return {"session_id": session_id, "theta_version": 3}

    monkeypatch.setattr(diag_es_package, "get_diag_es_mtp_manager", Manager)
    worker = SimpleNamespace(ps=SimpleNamespace(tp_size=1))

    exported = BaseTpWorker.diag_es_mtp_session(
        worker,
        DiagESMTPSessionReqInput(
            action="export_state",
            session_id="session",
        ),
    )
    assert exported is envelope

    status = BaseTpWorker.diag_es_mtp_session(
        worker,
        DiagESMTPSessionReqInput(
            action="import_state",
            session_id="session",
            session_state=snapshot,
            seed=7,
            sigma=0.02,
        ),
    )
    assert status == {"session_id": "session", "theta_version": 3}
    assert calls[0] == ("export", "session")
    assert calls[1][0:2] == ("import", "session")
    assert calls[1][2].seed == 7
    assert calls[1][2].sigma == 0.02
    assert calls[1][3]["state"]["theta_version"] == 3


def test_tp_worker_dispatches_event_calls_at_tp1(monkeypatch):
    import sglang.srt.diag_es as diag_es_package
    from sglang.srt.managers.tp_worker import BaseTpWorker

    page = _event_page()
    ack_status = {
        "engine_epoch": "engine-epoch",
        "acked_through_event_id": 2,
        "event_high_watermark": 2,
        "pending_event_count": 0,
    }
    calls = []

    class Manager:
        def read_events(self, *, engine_epoch, after_event_id, limit):
            calls.append(("read", engine_epoch, after_event_id, limit))
            return page

        def ack_events(self, *, engine_epoch, through_event_id):
            calls.append(("ack", engine_epoch, through_event_id))
            return ack_status

    monkeypatch.setattr(diag_es_package, "get_diag_es_mtp_manager", Manager)
    worker = SimpleNamespace(ps=SimpleNamespace(tp_size=1))

    read_status = BaseTpWorker.diag_es_mtp_session(
        worker,
        DiagESMTPSessionReqInput(
            action="read_events",
            engine_epoch="engine-epoch",
            after_event_id=0,
            event_limit=2,
        ),
    )
    acknowledged = BaseTpWorker.diag_es_mtp_session(
        worker,
        DiagESMTPSessionReqInput(
            action="ack_events",
            engine_epoch="engine-epoch",
            through_event_id=2,
        ),
    )

    assert read_status is page
    assert acknowledged is ack_status
    assert calls == [
        ("read", "engine-epoch", 0, 2),
        ("ack", "engine-epoch", 2),
    ]


@pytest.mark.parametrize(
    ("action", "kwargs"),
    [
        ("export_state", {"session_id": "session"}),
        (
            "import_state",
            {"session_id": "session", "session_state": _snapshot(), "seed": 7},
        ),
        (
            "read_events",
            {
                "engine_epoch": "engine-epoch",
                "after_event_id": 0,
                "event_limit": 1,
            },
        ),
        (
            "ack_events",
            {"engine_epoch": "engine-epoch", "through_event_id": 0},
        ),
    ],
)
def test_tp_worker_control_calls_fail_closed_above_tp1(
    monkeypatch, action, kwargs
):
    import sglang.srt.diag_es as diag_es_package
    from sglang.srt.managers.tp_worker import BaseTpWorker

    class Manager:
        def __getattr__(self, _name):
            raise AssertionError("manager state method must not run at TP>1")

    monkeypatch.setattr(diag_es_package, "get_diag_es_mtp_manager", Manager)
    worker = SimpleNamespace(ps=SimpleNamespace(tp_size=2))

    with pytest.raises(RuntimeError, match="supports only TP1"):
        BaseTpWorker.diag_es_mtp_session(
            worker,
            DiagESMTPSessionReqInput(action=action, **kwargs),
        )


def test_scheduler_uses_dedicated_opaque_export_field():
    from sglang.srt.managers.scheduler import Scheduler

    envelope = _export_envelope()
    snapshot = envelope["session_state"]
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            diag_es_mtp_session=lambda _request: envelope,
        )
    )
    output = Scheduler.handle_diag_es_mtp_session(
        scheduler,
        DiagESMTPSessionReqInput(
            action="export_state",
            session_id="session",
        ),
    )

    assert output.success
    assert output.status == {}
    transported_output = msgpack_decode(msgpack_encode(output))
    restored_envelope = unwrap_from_pickle(transported_output.session_state_export)
    assert restored_envelope["telemetry_frontier"] == {
        "engine_epoch": "engine-epoch",
        "event_high_watermark": 11,
    }
    restored = restored_envelope["session_state"]
    restored_tensor = restored["tensors"]["theta_dense"]["site"]
    assert restored_tensor.device.type == "cpu"
    assert restored_tensor.dtype == torch.float32
    torch.testing.assert_close(
        restored_tensor,
        snapshot["tensors"]["theta_dense"]["site"],
    )


def test_scheduler_propagates_event_page_in_typed_status():
    from sglang.srt.managers.scheduler import Scheduler

    page = _event_page()
    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(
            diag_es_mtp_session=lambda _request: page,
        )
    )
    output = Scheduler.handle_diag_es_mtp_session(
        scheduler,
        DiagESMTPSessionReqInput(
            action="read_events",
            engine_epoch="engine-epoch",
            after_event_id=0,
            event_limit=2,
        ),
    )

    assert output.success
    assert output.status is page
    assert output.session_state_export is None
    transported_output = msgpack_decode(msgpack_encode(output))
    assert transported_output.status == page


@pytest.mark.parametrize(
    "control_request",
    [
        DiagESMTPSessionReqInput(
            action="export_state",
            session_id="session",
        ),
        DiagESMTPSessionReqInput(
            action="read_events",
            engine_epoch="engine-epoch",
            after_event_id=0,
            event_limit=1,
        ),
    ],
)
def test_scheduler_converts_control_rpc_error_to_failed_response(control_request):
    from sglang.srt.managers.scheduler import Scheduler

    def fail(_request):
        raise RuntimeError("TP1 only")

    scheduler = SimpleNamespace(tp_worker=SimpleNamespace(diag_es_mtp_session=fail))
    output = Scheduler.handle_diag_es_mtp_session(
        scheduler,
        control_request,
    )

    assert not output.success
    assert output.message == "TP1 only"
    assert output.status == {}
    assert output.session_state_export is None
