import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerTqdmLock(unittest.TestCase):
    def _run_isolated_probe(self, probe: str):
        repo_root = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as flashinfer_workspace:
            env = os.environ.copy()
            env["FLASHINFER_WORKSPACE_BASE"] = flashinfer_workspace
            result = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(probe)],
                cwd=repo_root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tqdm_lock_test_import_does_not_swallow_sgl_kernel_errors(self):
        repo_root = Path(__file__).resolve().parents[4]
        tqdm_lock_test = (
            repo_root / "test/registered/unit/model_loader/test_tqdm_lock.py"
        )
        self._run_isolated_probe(
            f"""
            import builtins
            import runpy
            import sys

            import sglang.srt.managers.scheduler
            import sglang.srt.model_loader.weight_utils

            meta_path_before = list(sys.meta_path)
            sgl_kernel_modules_before = {{
                name: module
                for name, module in sys.modules.items()
                if name == "sgl_kernel" or name.startswith("sgl_kernel.")
            }}
            real_import = builtins.__import__

            def fail_sgl_kernel_import(name, *args, **kwargs):
                if name == "sgl_kernel":
                    raise OSError("forced sgl_kernel import failure")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = fail_sgl_kernel_import
            try:
                runpy.run_path(
                    {str(tqdm_lock_test)!r}, run_name="tqdm_lock_isolation_probe"
                )
            finally:
                builtins.__import__ = real_import

            assert sys.meta_path == meta_path_before
            assert {{
                name: module
                for name, module in sys.modules.items()
                if name == "sgl_kernel" or name.startswith("sgl_kernel.")
            }} == sgl_kernel_modules_before
            """
        )

    def test_entry_installs_process_local_std_tqdm_lock_before_init(self):
        self._run_isolated_probe(
            """
            import importlib.abc
            import importlib.machinery
            import sys
            import threading
            from unittest.mock import MagicMock, patch

            from tqdm.std import TqdmDefaultWriteLock, tqdm as std_tqdm

            class SglKernelLoader(importlib.abc.Loader):
                def create_module(self, spec):
                    return None

                def exec_module(self, module):
                    module.__getattr__ = lambda name: MagicMock()

            class SglKernelFinder(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
                        return importlib.machinery.ModuleSpec(
                            fullname, SglKernelLoader(), is_package=True
                        )
                    return None

            missing = object()
            previous_tqdm_lock = std_tqdm.__dict__.get("_lock", missing)
            previous_mp_lock = TqdmDefaultWriteLock.__dict__.get("mp_lock", missing)
            meta_path_before = list(sys.meta_path)
            modules_before = dict(sys.modules)
            finder = SglKernelFinder()
            multiprocessing_rlocks = []
            events = []
            locks_seen = []

            if previous_tqdm_lock is not missing:
                delattr(std_tqdm, "_lock")
            if previous_mp_lock is not missing:
                delattr(TqdmDefaultWriteLock, "mp_lock")

            try:
                sys.meta_path.insert(0, finder)
                from sglang.srt.managers import scheduler as scheduler_module

                class FakeScheduler:
                    def __init__(self, *args):
                        events.append("scheduler")
                        locks_seen.append(std_tqdm.get_lock())
                        self.metrics_reporter = type(
                            "MetricsReporter",
                            (), {"_shutdown_fpm": lambda self: None},
                        )()
                        self.gracefully_exit = False

                    def get_init_info(self):
                        return {}

                    def run_event_loop(self):
                        pass

                def record_multiprocessing_rlock():
                    multiprocessing_rlocks.append(True)
                    return threading.RLock()

                def fake_load_plugins():
                    events.append("plugins")
                    locks_seen.append(std_tqdm.get_lock())

                def fake_configure_scheduler_process(*args, **kwargs):
                    events.append("config")
                    locks_seen.append(std_tqdm.get_lock())
                    return 0

                with patch(
                    "multiprocessing.RLock", side_effect=record_multiprocessing_rlock
                ), patch.object(
                    scheduler_module, "load_plugins", side_effect=fake_load_plugins
                ), patch.object(
                    scheduler_module,
                    "configure_scheduler_process",
                    side_effect=fake_configure_scheduler_process,
                ), patch.object(scheduler_module, "publish"), patch.object(
                    scheduler_module, "psutil"
                ), patch.object(
                    scheduler_module, "Scheduler", side_effect=FakeScheduler
                ):
                    scheduler_module.run_scheduler_process(
                        server_args=type("ServerArgs", (), {"enable_trace": False})(),
                        port_args=object(),
                        gpu_id=0,
                        tp_rank=0,
                        attn_cp_rank=0,
                        moe_dp_rank=0,
                        moe_ep_rank=0,
                        pp_rank=0,
                        dp_rank=0,
                        pipe_writer=type(
                            "PipeWriter", (), {"send": lambda self, _: None}
                        )(),
                    )

                assert multiprocessing_rlocks == []
                assert events == ["plugins", "config", "scheduler"]
                assert all(
                    isinstance(lock, type(threading.RLock())) for lock in locks_seen
                )
            finally:
                if "_lock" in std_tqdm.__dict__:
                    delattr(std_tqdm, "_lock")
                if previous_tqdm_lock is not missing:
                    std_tqdm._lock = previous_tqdm_lock

                if "mp_lock" in TqdmDefaultWriteLock.__dict__:
                    delattr(TqdmDefaultWriteLock, "mp_lock")
                if previous_mp_lock is not missing:
                    TqdmDefaultWriteLock.mp_lock = previous_mp_lock

                sys.meta_path[:] = meta_path_before
                for name in tuple(sys.modules):
                    if name not in modules_before:
                        del sys.modules[name]
                sys.modules.update(modules_before)

            assert sys.meta_path == meta_path_before
            assert sys.modules == modules_before
            """
        )


if __name__ == "__main__":
    unittest.main()
