import contextlib
import io
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import safetensors.torch
import torch
from tqdm.std import TqdmDefaultWriteLock

from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    tqdm,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTqdmLock(unittest.TestCase):
    def test_buffered_loader_uses_process_local_tqdm_lock(self):
        missing = object()
        previous_tqdm_lock = tqdm.__dict__.get("_lock", missing)
        previous_mp_lock = TqdmDefaultWriteLock.__dict__.get("mp_lock", missing)
        multiprocessing_rlocks = []

        if previous_tqdm_lock is not missing:
            delattr(tqdm, "_lock")
        if previous_mp_lock is not missing:
            delattr(TqdmDefaultWriteLock, "mp_lock")

        try:
            def record_multiprocessing_rlock():
                multiprocessing_rlocks.append(True)
                return threading.RLock()

            with patch(
                "multiprocessing.RLock", side_effect=record_multiprocessing_rlock
            ):
                with tempfile.TemporaryDirectory() as tmpdir:
                    shard = os.path.join(tmpdir, "model.safetensors")
                    expected = torch.tensor([1.0, 2.0])
                    safetensors.torch.save_file({"weight": expected}, shard)
                    progress = io.StringIO()
                    with contextlib.redirect_stderr(progress):
                        loaded = {
                            name: value.clone()
                            for name, value in (
                                buffered_multi_thread_safetensors_weights_iterator(
                                    [shard], max_workers=1
                                )
                            )
                        }
                installed_lock = tqdm.get_lock()

            torch.testing.assert_close(loaded["weight"], expected)
            self.assertIn("Multi-thread loading shards", progress.getvalue())
            self.assertIn("1/1", progress.getvalue())
            self.assertEqual(multiprocessing_rlocks, [])
            self.assertIsInstance(installed_lock, type(threading.RLock()))
        finally:
            if "_lock" in tqdm.__dict__:
                delattr(tqdm, "_lock")
            if previous_tqdm_lock is not missing:
                tqdm._lock = previous_tqdm_lock

            if "mp_lock" in TqdmDefaultWriteLock.__dict__:
                delattr(TqdmDefaultWriteLock, "mp_lock")
            if previous_mp_lock is not missing:
                TqdmDefaultWriteLock.mp_lock = previous_mp_lock


if __name__ == "__main__":
    unittest.main()
