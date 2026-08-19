import multiprocessing.resource_sharer
import os
import socket
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Watchdog:
    def __init__(self, events):
        self.events = events

    def stop(self):
        self.events.append("watchdog")


class _Transport:
    def __init__(self, events):
        self.events = events

    def shutdown(self):
        self.events.append("cuda-vmm")


class _TokenizerManager:
    def __init__(self, events):
        self._subprocess_watchdog = _Watchdog(events)
        self.cuda_vmm_feature_transport = _Transport(events)


class _Socket:
    def __init__(self, events):
        self.events = events

    def close(self, *, linger):
        self.events.append(f"zmq:{linger}")


class TestEngineShutdown(unittest.TestCase):
    def _make_engine(self, events):
        engine = object.__new__(Engine)
        engine.tokenizer_manager = _TokenizerManager(events)
        engine.send_to_rpc = _Socket(events)
        engine._weight_cache_daemon_procs = [object()]
        engine._terminate_weight_cache_daemons = lambda _procs: events.append(
            "weight-daemons"
        )
        return engine

    def test_resource_sharer_stops_after_children_and_before_cuda_transport(self):
        events = []
        engine = self._make_engine(events)

        with (
            patch.object(engine_module, "TokenizerManager", _TokenizerManager),
            patch.object(engine_module, "kill_process_tree") as kill_children,
            patch.object(
                multiprocessing.resource_sharer,
                "stop",
                side_effect=lambda: events.append("resource-sharer"),
            ),
        ):
            kill_children.side_effect = lambda *_args, **_kwargs: events.append(
                "children"
            )
            engine.shutdown()

        kill_children.assert_called_once_with(
            os.getpid(), include_parent=False, wait_timeout=60
        )
        self.assertEqual(
            events,
            [
                "watchdog",
                "zmq:0",
                "weight-daemons",
                "children",
                "resource-sharer",
                "cuda-vmm",
            ],
        )

    def test_resource_sharer_cleanup_is_idempotent(self):
        engine = object.__new__(Engine)
        engine.tokenizer_manager = None
        left, right = socket.socketpair()
        try:
            shared_fd = multiprocessing.resource_sharer.DupFd(left.fileno())
        except PermissionError as exc:
            left.close()
            right.close()
            self.skipTest(f"sandbox blocks AF_UNIX resource-sharer listener: {exc}")
        sharer_address = multiprocessing.resource_sharer._resource_sharer._address
        self.assertTrue(os.path.exists(sharer_address))

        try:
            with patch.object(engine_module, "kill_process_tree"):
                engine.shutdown()
                self.assertFalse(os.path.exists(sharer_address))
                engine.shutdown()
        finally:
            del shared_fd
            left.close()
            right.close()

        self.assertIsNone(multiprocessing.resource_sharer._resource_sharer._address)


if __name__ == "__main__":
    unittest.main()
