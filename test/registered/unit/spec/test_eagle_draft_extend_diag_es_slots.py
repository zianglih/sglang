from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Event:
    def record(self):
        pass


class _DraftExtendBackend:
    def init_forward_metadata_out_graph(self, _forward_batch):
        pass


def _runner():
    width = 4
    max_bs = 4
    runner = EAGLEDraftExtendCudaGraphRunner.__new__(
        EAGLEDraftExtendCudaGraphRunner
    )
    runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
    runner.captured_req_width = width
    runner.capture_bs = [1, max_bs]
    runner.require_mlp_tp_gather = False
    runner.require_gathered_buffer = False
    runner.seq_len_fill_value = 1
    runner.extend_seq_lens_cpu = [width] * max_bs
    runner.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    runner.draft_extend_attn_backend = _DraftExtendBackend()
    runner.device_module = SimpleNamespace(Event=_Event)
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        device_timer=None,
        war_fastpath_read_done_event=None,
    )
    runner.buffers = SimpleNamespace(
        input_ids=torch.zeros(max_bs * width, dtype=torch.int64),
        seq_lens=torch.zeros(max_bs, dtype=torch.int64),
        seq_lens_cpu=torch.zeros(max_bs, dtype=torch.int64),
        out_cache_loc=torch.zeros(max_bs * width, dtype=torch.int64),
        positions=torch.zeros(max_bs * width, dtype=torch.int64),
        req_pool_indices=torch.zeros(max_bs, dtype=torch.int64),
        extend_seq_lens=torch.zeros(max_bs, dtype=torch.int32),
        num_correct_drafts=torch.zeros(max_bs, dtype=torch.int32),
        num_accept_tokens=torch.zeros(max_bs, dtype=torch.int32),
        hidden_states=None,
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
        es_candidate_slots=torch.full(
            (max_bs * width,), -1, dtype=torch.int32
        ),
    )
    runner._replay_graph = lambda _shape, _batch: SimpleNamespace(
        next_token_logits=torch.zeros(max_bs * width, 2),
        hidden_states=torch.zeros(max_bs * width, 3),
    )
    return runner


def _forward_batch(*, with_slots=True):
    width = 4
    raw_bs = 2
    return SimpleNamespace(
        batch_size=raw_bs,
        input_ids=torch.arange(raw_bs * width, dtype=torch.int64),
        seq_lens=torch.tensor([10, 11], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([10, 11], dtype=torch.int64),
        seq_lens_sum=21,
        extend_seq_lens=torch.full((raw_bs,), width, dtype=torch.int32),
        extend_seq_lens_cpu=[width] * raw_bs,
        out_cache_loc=torch.arange(raw_bs * width, dtype=torch.int64),
        positions=torch.arange(raw_bs * width, dtype=torch.int64),
        req_pool_indices=torch.arange(raw_bs, dtype=torch.int64),
        global_num_tokens_cpu=None,
        es_candidate_slots=(
            torch.tensor([5] * width + [7] * width, dtype=torch.int32)
            if with_slots
            else None
        ),
        spec_info=SimpleNamespace(
            hidden_states=None,
            num_correct_drafts=torch.tensor([1, 2], dtype=torch.int32),
            num_accept_tokens=torch.tensor([2, 3], dtype=torch.int32),
            positions=None,
        ),
    )


def test_draft_extend_graph_slots_copy_and_pad_with_identity():
    runner = _runner()

    runner.execute(_forward_batch())

    torch.testing.assert_close(
        runner.buffers.es_candidate_slots,
        torch.tensor([5] * 4 + [7] * 4 + [0] * 8, dtype=torch.int32),
    )


def test_draft_extend_graph_fails_closed_without_slots():
    with pytest.raises(RuntimeError, match="missing candidate slots"):
        _runner().execute(_forward_batch(with_slots=False))
