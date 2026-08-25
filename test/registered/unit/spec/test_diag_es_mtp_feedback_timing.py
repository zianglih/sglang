import contextlib
from types import SimpleNamespace

import sglang.srt.diag_es as diag_es
import sglang.srt.speculative.eagle_worker_v2 as eagle_module
import torch
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _NoReqScanBatch:
    @property
    def reqs(self):
        raise AssertionError("clean MTP path scanned requests")


def test_feedback_callback_is_zero_work_when_mtp_is_off():
    worker = object.__new__(BaseSpecWorker)
    worker._diag_es_mtp_enabled = False
    worker._record_diag_es_mtp_feedback(_NoReqScanBatch(), object())


def test_feedback_switch_and_graph_read_fence_bracket_draft_extend(monkeypatch):
    order = []
    candidate = {"index": 0}

    class _Manager:
        reservation = SimpleNamespace(expects_kv_replay=True)

        @classmethod
        def preflight_acceptance_batch(cls, attempts):
            assert attempts == [("session", "rid", 1)]
            order.append("preflight")
            return cls.reservation

        @staticmethod
        def record_acceptance(**kwargs):
            order.append("feedback")
            assert kwargs["accepted_drafts"] == 1
            candidate["index"] = 1
            return {
                "population_index": 1,
                "theta_version": 0,
                "perturbation_seed": 2,
                "sigma": 0.01,
                "learning_rate": 0.1,
            }

        @staticmethod
        def record_kv_replay(**kwargs):
            assert kwargs["acceptance_batch_reservation"] is _Manager.reservation
            assert kwargs["session_ids"] == ["session"]
            assert kwargs["request_rows"] == [10]
            order.append("replay_telemetry")

        @staticmethod
        def note_slots_read(slots):
            assert tuple(slots) == (7,)
            assert order[-1] == "cuda_graph_replay"
            order.append("slot_read_fence")

    monkeypatch.setattr(diag_es, "get_diag_es_mtp_manager", lambda: _Manager())
    monkeypatch.setattr(
        eagle_module, "speculative_moe_backend_context", contextlib.nullcontext
    )
    monkeypatch.setattr(
        eagle_module, "speculative_moe_a2a_backend_context", contextlib.nullcontext
    )
    monkeypatch.setattr(
        eagle_module, "spec_stage_span", lambda _name: contextlib.nullcontext()
    )

    verify_input = SimpleNamespace(is_verify_input=lambda: True)

    def draft(_batch):
        order.append("draft")
        return verify_input

    def draft_extend(_batch, _output):
        # This spy represents the specialized CUDA-graph replay path. The
        # candidate switch must be visible before replay, and the slot-read
        # event must be recorded only after the complete replay returns.
        assert candidate["index"] == 1
        order.append("cuda_graph_replay")

    class _Replay:
        @staticmethod
        def replay_transitioned_prefixes(_batch, transitions):
            assert transitions == (True,)
            assert candidate["index"] == 1
            order.append("kv_replay")
            return SimpleNamespace(
                replayed_rows=10,
                request_rows=(10,),
                enqueue_time_ms=0.25,
            )

    draft_worker = SimpleNamespace(
        draft=lambda batch: draft(batch),
        _draft_extend_for_decode=draft_extend,
        draft_runner=SimpleNamespace(tp_group=None),
        draft_tp_context=lambda _group: contextlib.nullcontext(),
        diag_es_mtp_kv_replay=_Replay(),
    )
    output = SimpleNamespace(
        accept_lens=torch.tensor([2]),
        new_seq_lens=torch.tensor([11]),
    )
    req = SimpleNamespace(
        rid="rid",
        diag_es_mtp_session_id="session",
        diag_es_mtp_slot=7,
        diag_es_mtp_status={
            "population_index": 0,
            "theta_version": 0,
            "perturbation_seed": 1,
            "sigma": 0.01,
            "learning_rate": 0.1,
        },
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False),
        is_extend_in_batch=False,
        spec_info=object(),
        seq_lens=torch.tensor([10]),
        reqs=[req],
    )
    worker = object.__new__(EAGLEWorkerV2)
    worker._draft_worker = draft_worker
    worker._diag_es_mtp_enabled = True
    worker._diag_es_mtp_requires_kv_replay = True
    worker.speculative_num_steps = 3
    worker.activate_step_by_batch = lambda _bs: None

    def verify(_batch, grammar_barrier=None):
        order.append("verify")
        return output

    worker.verify = verify

    def publish(_seq_lens):
        assert order[-1] == "replay_telemetry"
        order.append("publish")

    result = EAGLEWorkerV2.forward_batch_generation(worker, batch, on_publish=publish)

    assert result is output
    assert req.diag_es_mtp_status["population_index"] == 1
    assert order == [
        "draft",
        "verify",
        "preflight",
        "feedback",
        "kv_replay",
        "replay_telemetry",
        "publish",
        "cuda_graph_replay",
        "slot_read_fence",
    ]


def test_eager_draft_inner_forward_preserves_mtp_candidate_slots(monkeypatch):
    monkeypatch.setattr(eagle_module, "_is_cuda", False)
    monkeypatch.setattr(
        eagle_module,
        "get_spec",
        lambda: SimpleNamespace(speculative_use_rejection_sampling=False),
    )
    slots = torch.tensor([7], dtype=torch.int32)

    class _DraftRunner:
        canary_manager = None
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["JoyAILLMFlashForCausalLMNextN"])
        )

        @staticmethod
        def forward(_forward_batch):
            assert get_forward_context().es_candidate_slots is slots
            return SimpleNamespace(
                logits_output=SimpleNamespace(
                    next_token_logits=torch.tensor([[0.0, 1.0, 0.0]]),
                    hidden_states=torch.zeros(1, 2),
                )
            )

    worker = object.__new__(EagleDraftWorker)
    worker.speculative_num_steps = 2
    worker.speculative_num_draft_tokens = 3
    worker.topk = 1
    worker.hot_token_id = None
    worker.index_share_for_mtp_iteration = False
    worker.seed_dsa_topk_from_draft_extend = False
    worker._topk1_parents_prealloc = torch.tensor([[-1, 0]], dtype=torch.long)
    worker._topk1_score_indices_prealloc = torch.tensor([[0, 1]], dtype=torch.long)
    worker.draft_runner = _DraftRunner()
    worker.draft_attn_backend = SimpleNamespace(attn_backends=[object()])

    forward_batch = SimpleNamespace(
        batch_size=1,
        input_ids=None,
        positions=torch.tensor([0], dtype=torch.long),
        out_cache_loc=torch.tensor([0, 1], dtype=torch.long),
        es_candidate_slots=slots,
        sampling_info=None,
        spec_info=EagleDraftInput(
            topk_p=torch.ones(1, 1),
            topk_index=torch.ones(1, 1, dtype=torch.long),
            hidden_states=torch.zeros(1, 2),
        ),
    )

    _, _, draft_tokens, _ = EagleDraftWorker.draft_forward(worker, forward_batch)
    assert draft_tokens.shape == (1, 2)
