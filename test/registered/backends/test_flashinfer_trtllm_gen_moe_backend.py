import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, suite="nightly-4-gpu-b200", nightly=True)


class FlashinferTrtllmGenMoeBackendFP8Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--attention-backend",
                "triton",
                "--moe-runner-backend",
                cls.backend,
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--mamba-ssm-dtype",
                "bfloat16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.89)


class FlashinferTrtllmGenMoeBackendBF16Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--moe-runner-backend",
                cls.backend,
                "--cuda-graph-max-bs",
                "512",
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--mamba-ssm-dtype",
                "bfloat16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.93)


class FlashinferTrtllmGenMoeBackendMXFP8Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "zianglih/Qwen3-30B-A3B-Instruct-2507-MXFP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--fp8-gemm-backend",
                "flashinfer_cutlass",
                "--moe-runner-backend",
                cls.backend,
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.93)


class FlashinferTrtllmGenMoeBackendMXFP8MixedBF16Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "zianglih/JoyAI-LLM-Flash-MXFP8-last-6-BF16"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--kv-cache-dtype",
                "bf16",
                "--fp8-gemm-backend",
                "flashinfer_cutlass",
                "--moe-runner-backend",
                cls.backend,
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_platinum(self):
        repo_root = Path(__file__).resolve().parents[3]
        benchmark = repo_root / "benchmark" / "gsm8k" / "bench_sglang.py"
        parsed_url = urlparse(self.base_url)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                sys.executable,
                str(benchmark),
                "--num-shots",
                "8",
                "--num-questions",
                "1209",
                "--parallel",
                "1209",
                "--platinum",
                "--host",
                parsed_url.hostname or "127.0.0.1",
                "--port",
                str(parsed_url.port or 30000),
                "--result-file",
                str(Path(tmp_dir) / "result.jsonl"),
                "--raw-result-file",
                str(Path(tmp_dir) / "raw_result.jsonl"),
            ]
            result = subprocess.run(
                cmd,
                cwd=tmp_dir,
                text=True,
                capture_output=True,
                check=False,
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)

        match = re.search(r"Accuracy:\s*([0-9.]+)", output)
        self.assertIsNotNone(match, output)
        accuracy = float(match.group(1))
        summary = "\n".join(
            line
            for line in output.splitlines()
            if line.startswith(
                ("Accuracy:", "Invalid:", "Latency:", "Output throughput:")
            )
        )
        print(summary)
        self.assertGreater(accuracy, 0.92, output)


class FlashinferTrtllmGenMoeBackendNVFP4Base:
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Qwen3-30B-A3B-NVFP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env={**os.environ, "SGLANG_ENABLE_JIT_DEEPGEMM": "False"},
            other_args=[
                "--moe-runner-backend",
                cls.backend,
                "--tp-size",
                "4",
                "--ep-size",
                "4",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.89)


class TestFlashinferTrtllmGenMoeBackendFP8(
    FlashinferTrtllmGenMoeBackendFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendMXFP8(
    FlashinferTrtllmGenMoeBackendMXFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendBF16(
    FlashinferTrtllmGenMoeBackendBF16Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendNVFP4(
    FlashinferTrtllmGenMoeBackendNVFP4Base, CustomTestCase
):
    backend = "flashinfer_trtllm"


class TestFlashinferTrtllmGenMoeBackendFP8Routed(
    FlashinferTrtllmGenMoeBackendFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendMXFP8Routed(
    FlashinferTrtllmGenMoeBackendMXFP8Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmRoutedMxfp8MixedBF16(
    FlashinferTrtllmGenMoeBackendMXFP8MixedBF16Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendBF16Routed(
    FlashinferTrtllmGenMoeBackendBF16Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


class TestFlashinferTrtllmGenMoeBackendNVFP4Routed(
    FlashinferTrtllmGenMoeBackendNVFP4Base, CustomTestCase
):
    backend = "flashinfer_trtllm_routed"


if __name__ == "__main__":
    unittest.main()
