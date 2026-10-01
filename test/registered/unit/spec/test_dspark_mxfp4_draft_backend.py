import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WORKER = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
MOE = "sglang.srt.layers.moe.utils"


def _args(**updates):
    values = dict(
        model_path="target",
        device="cuda",
        tp_size=8,
        ep_size=8,
        dp_size=8,
        enable_dp_attention=True,
        enable_dp_lm_head=True,
        moe_a2a_backend="flashinfer_megamoe",
        moe_runner_backend="flashinfer_megamoe",
        speculative_algorithm="DSPARK",
        speculative_draft_model_path="bundled-draft",
        speculative_dspark_block_size=5,
        speculative_moe_a2a_backend="none",
        speculative_moe_runner_backend="flashinfer_mxfp4",
    )
    values.update(updates)
    return ServerArgs(**values)


def _resolve(args, mode="static"):
    with (
        envs.SGLANG_RAGGED_VERIFY_MODE.override(mode),
        patch(
            "sglang.srt.speculative.dspark_components.dspark_config.read_draft_checkpoint_config",
            return_value=None,
        ),
    ):
        _handle_dspark(args)


@contextmanager
def _runtime(target_a2a="flashinfer_megamoe", target_runner="flashinfer_megamoe"):
    flags = SimpleNamespace(
        moe=SimpleNamespace(
            a2a_backend=MoeA2ABackend(target_a2a),
            runner_backend=MoeRunnerBackend(target_runner),
            speculative_a2a_backend=MoeA2ABackend.NONE,
            speculative_runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4,
            disable_fp4_allgather=False,
            speculative_context=False,
        )
    )
    with patch(f"{MOE}.get_flags", return_value=flags):
        yield flags.moe


def _worker():
    worker = object.__new__(DSparkWorkerV2)
    worker._hosts_draft = True
    worker._draft_is_moe = True
    worker._draft_moe_context_enabled = True
    worker._draft_dp_context_enabled = False
    worker.device = "cpu"
    return worker


class TestDSparkDraftBackendResolution(unittest.TestCase):
    def test_both_mega_precisions_and_trt_preserve_draft_and_gamma(self):
        for a2a, runner, w4a16 in (
            ("flashinfer_megamoe", "flashinfer_megamoe", False),
            ("flashinfer_megamoe", "flashinfer_megamoe", True),
            ("none", "flashinfer_trtllm_routed", False),
        ):
            with self.subTest(a2a=a2a, w4a16=w4a16):
                args = _args(moe_a2a_backend=a2a, moe_runner_backend=runner)
                with envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(w4a16):
                    _resolve(args)
                self.assertEqual(
                    resolution_result(args, "speculative_num_draft_tokens"), 6
                )
                self.assertEqual(
                    args.speculative_moe_runner_backend, "flashinfer_mxfp4"
                )
                self.assertEqual(args.speculative_moe_a2a_backend, "none")

    def test_mega_rejects_inherited_or_inconsistent_draft_layout(self):
        for overrides in (
            dict(speculative_moe_a2a_backend=None),
            dict(speculative_moe_runner_backend="flashinfer_megamoe"),
            dict(speculative_moe_a2a_backend="flashinfer_megamoe"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "preserve the bundled MXFP4"):
                    _resolve(_args(**overrides))

    def test_existing_dp_safety_gates_remain(self):
        for overrides, message in (
            (dict(enable_dp_lm_head=False), "enable-dp-lm-head"),
            (dict(attn_cp_size=2), "context parallel"),
            (dict(moe_a2a_backend="deepep"), "supports moe_a2a_backend"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    _resolve(_args(**overrides))
        with self.assertRaisesRegex(ValueError, "RAGGED_VERIFY_MODE=static"):
            _resolve(_args(), mode="compact")

    def test_other_mismatched_draft_backends_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "match the target"):
            _resolve(
                _args(
                    moe_a2a_backend="megamoe", speculative_moe_runner_backend="triton"
                )
            )


class TestDSparkDraftBackendScope(unittest.TestCase):
    def _assert_draft(self):
        self.assertEqual(get_moe_a2a_backend(), MoeA2ABackend.NONE)
        self.assertEqual(get_moe_runner_backend(), MoeRunnerBackend.FLASHINFER_MXFP4)

    def test_scope_restores_target_on_failure_without_changing_groups(self):
        worker = _worker()
        for a2a, runner in (
            ("flashinfer_megamoe", "flashinfer_megamoe"),
            ("none", "flashinfer_trtllm_routed"),
        ):
            with (
                _runtime(a2a, runner) as flags,
                patch(f"{WORKER}.draft_tp_context") as tp,
            ):
                original = vars(flags).copy()
                with self.assertRaisesRegex(RuntimeError, "draft failure"):
                    with worker._draft_context():
                        self._assert_draft()
                        self.assertTrue(flags.speculative_context)
                        raise RuntimeError("draft failure")
                self.assertEqual(vars(flags), original)
                tp.assert_not_called()

    def test_hybrid_config_keeps_nvfp4_target_and_native_mxfp4_draft(self):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.modelopt_quant import (
            HybridFp8NvFp4Config,
            ModelOptFp4Config,
            ModelOptNvFp4FusedMoEMethod,
        )
        from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
            Mxfp4FlashinferTrtllmMoEMethod,
        )

        # These are the hybrid loader's native precision/exclusion fields.
        # Exercise its real dispatch without creating any GPU weights.
        config = object.__new__(HybridFp8NvFp4Config)
        config.is_fp4_experts = True
        config.dequant_fp4_to_fp8 = False
        config.use_mxfp8 = True
        config.weight_block_size = None
        config.nvfp4_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.decoder.*", "stages.*"],
        )
        layer = object.__new__(FusedMoE)
        worker = _worker()
        for a2a, runner, w4a16 in (
            ("flashinfer_megamoe", "flashinfer_megamoe", False),
            ("flashinfer_megamoe", "flashinfer_megamoe", True),
            ("none", "flashinfer_trtllm_routed", False),
        ):
            with (
                self.subTest(a2a=a2a, w4a16=w4a16),
                _runtime(a2a, runner),
                envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(w4a16),
                patch(
                    "sglang.srt.layers.quantization.modelopt_quant.get_platform",
                    return_value=SimpleNamespace(is_blackwell=True),
                ),
                patch(
                    "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe.get_exec",
                    return_value=SimpleNamespace(
                        moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
                    ),
                ),
            ):
                target = config.get_quant_method(layer, "model.layers.0.mlp.experts")
                self.assertIsInstance(target, ModelOptNvFp4FusedMoEMethod)
                with worker._draft_context():
                    draft = config.get_quant_method(layer, "stages.0.mlp.experts")
                    self.assertIsInstance(draft, Mxfp4FlashinferTrtllmMoEMethod)
                    self.assertTrue(draft._fp8.is_fp4_expert)
                    self.assertFalse(draft._fp8.dequant_fp4_to_fp8)
                    self.assertEqual(draft.flashinfer_mxfp4_moe_precision, "default")

    def test_construction_uses_fixed_backend_before_loading_weights(self):
        def build(**_kwargs):
            self._assert_draft()
            raise RuntimeError("stop before GPU construction")

        target = SimpleNamespace(
            model_runner=SimpleNamespace(model_config=object()),
            device="cpu",
            random_seed=1,
        )
        parallel = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            enable_dp_attention=True,
            attn_tp_size=1,
        )
        graph = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=Backend.DISABLED)
            )
        )
        with (
            _runtime() as flags,
            patch(f"{WORKER}._is_npu", False),
            patch(f"{WORKER}.draft_is_deepseek_v4", return_value=True),
            patch(f"{WORKER}.get_parallel", return_value=parallel),
            patch(f"{WORKER}.get_schedule", return_value=SimpleNamespace(page_size=1)),
            patch(
                f"{WORKER}.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch(f"{WORKER}.get_exec", return_value=SimpleNamespace(graph=graph)),
            patch(f"{WORKER}.get_spec", return_value=_args()),
            patch(f"{WORKER}.draft_pp_context", side_effect=nullcontext),
            patch(f"{WORKER}.build_draft_tp_worker", side_effect=build),
        ):
            original = vars(flags).copy()
            with self.assertRaisesRegex(RuntimeError, "stop before GPU"):
                DSparkWorkerV2(_args(), 0, 0, target)
            self.assertEqual(vars(flags), original)

    def test_graph_capture_uses_draft_scope(self):
        worker = _worker()
        worker._draft_worker = SimpleNamespace(
            init_cuda_graphs=lambda **kw: self._assert_draft(),
        )
        worker._decode_graph_allowed = True
        worker._draft_sampler = None
        worker._proposer = Mock()
        worker._tp_sync = SimpleNamespace(available_memory_gb=lambda *a, **kw: 10.0)
        worker.gpu_id = 0
        worker._draft_graph_group = object()
        with (
            _runtime() as flags,
            patch(f"{WORKER}.draft_pp_context", side_effect=nullcontext),
            patch(f"{WORKER}.is_cuda_alike", return_value=False),
            envs.SGLANG_DSPARK_FOLDED_PROPOSAL.override(False),
        ):
            original = vars(flags).copy()
            worker.init_cuda_graphs()
            self.assertEqual(vars(flags), original)

    def test_idle_draft_participates_then_target_verifies_in_target_scope(self):
        worker = _worker()
        worker._observers = Mock()
        worker._proposer = SimpleNamespace(
            run_idle_participation=lambda batch: self._assert_draft()
        )
        seen = []
        worker._verify_executor = SimpleNamespace(
            run_idle_participation=lambda **kw: seen.append(get_moe_a2a_backend())
        )
        worker._idle_verify_ragged_layout = lambda batch: None
        worker._decode_idle_result = lambda **kw: "idle done"
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            spec_info=DFlashDraftInputV2.create_idle_input(device="cpu"),
        )
        with (
            _runtime(),
            patch(
                f"{WORKER}.get_parallel",
                return_value=SimpleNamespace(enable_dp_attention=True),
            ),
        ):
            self.assertEqual(worker._forward_decode(batch, None), "idle done")
        self.assertEqual(seen, [MoeA2ABackend.FLASHINFER_MEGAMOE])

    def test_active_proposal_uses_draft_scope_and_restores_on_failure(self):
        worker = _worker()
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model=object())
        )
        worker.model_runner = worker._target_worker.model_runner
        worker.verify_num_draft_tokens = 6
        worker._block_pos_offsets = object()
        worker._observers = SimpleNamespace(
            begin_step=lambda: None, segment=lambda _: nullcontext()
        )

        def propose(**kw):
            self._assert_draft()
            raise RuntimeError("stop after draft dispatch")

        worker._proposer = SimpleNamespace(propose=propose)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=DFlashDraftInputV2.create_idle_input(device="cpu"),
            seq_lens=MagicMock(),
            sampling_info=object(),
            req_pool_indices=object(),
        )
        batch.seq_lens.__len__.return_value = 1
        with (
            _runtime() as flags,
            patch(f"{WORKER}.torch.get_device_module", return_value=Mock()),
            patch(f"{WORKER}.alloc_verify_window", return_value=object()),
        ):
            original = vars(flags).copy()
            with self.assertRaisesRegex(RuntimeError, "stop after draft dispatch"):
                worker._forward_decode(batch, None)
            self.assertEqual(vars(flags), original)


if __name__ == "__main__":
    unittest.main()
