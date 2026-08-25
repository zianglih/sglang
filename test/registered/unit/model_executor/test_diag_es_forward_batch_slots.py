"""CPU coverage for role-local diagonal-ES ForwardBatch slot construction."""

import unittest
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    _build_diag_es_candidate_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _runner(enabled: bool, *, is_draft_worker=False):
    return SimpleNamespace(
        diag_es_enabled=enabled,
        is_draft_worker=is_draft_worker,
        device=torch.device("cpu"),
    )


def _batch(mode: ForwardMode, slots, *, token_count, extend_lens=None, width=None):
    return SimpleNamespace(
        forward_mode=mode,
        reqs=[
            SimpleNamespace(es_candidate_slot=slot, diag_es_mtp_slot=slot + 10)
            for slot in slots
        ],
        input_ids=torch.zeros(token_count, dtype=torch.int64),
        extend_lens=extend_lens,
        extend_num_tokens=token_count,
        spec_info=(
            SimpleNamespace(num_tokens_per_req=width) if width is not None else None
        ),
    )


class TestDiagESForwardBatchSlots(unittest.TestCase):
    def test_clean_draft_role_does_not_read_request_slots(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            reqs=[SimpleNamespace()],
        )

        token_slots, request_slots = _build_diag_es_candidate_slots(
            batch, _runner(False)
        )

        self.assertIsNone(token_slots)
        self.assertIsNone(request_slots)

    def test_decode_keeps_one_slot_per_request(self):
        token_slots, request_slots = _build_diag_es_candidate_slots(
            _batch(ForwardMode.DECODE, [3, 1], token_count=2), _runner(True)
        )

        torch.testing.assert_close(token_slots, torch.tensor([3, 1], dtype=torch.int32))
        self.assertEqual(request_slots, (3, 1))

    def test_draft_role_uses_independent_mtp_slots(self):
        token_slots, request_slots = _build_diag_es_candidate_slots(
            _batch(ForwardMode.DECODE, [3, 1], token_count=2),
            _runner(True, is_draft_worker=True),
        )

        torch.testing.assert_close(
            token_slots, torch.tensor([13, 11], dtype=torch.int32)
        )
        self.assertEqual(request_slots, (13, 11))

    def test_topk2_draft_decode_expands_mtp_slots_per_activation_row(self):
        batch = _batch(ForwardMode.DECODE, [3, 1], token_count=4, width=2)
        # Spec-v2 prepares draft token IDs directly on device and deliberately
        # leaves the ScheduleBatch CPU input_ids field unset.
        batch.input_ids = None
        token_slots, request_slots = _build_diag_es_candidate_slots(
            batch, _runner(True, is_draft_worker=True)
        )

        torch.testing.assert_close(
            token_slots, torch.tensor([13, 13, 11, 11], dtype=torch.int32)
        )
        self.assertEqual(request_slots, (13, 11))

    def test_prefill_expands_slots_by_extend_lengths(self):
        token_slots, request_slots = _build_diag_es_candidate_slots(
            _batch(
                ForwardMode.EXTEND,
                [3, 1],
                token_count=5,
                extend_lens=[2, 3],
            ),
            _runner(True),
        )

        torch.testing.assert_close(
            token_slots, torch.tensor([3, 3, 1, 1, 1], dtype=torch.int32)
        )
        self.assertEqual(request_slots, (3, 1))

    def test_target_verify_expands_each_target_slot_across_verify_window(self):
        batch = _batch(
            ForwardMode.TARGET_VERIFY,
            [3, 1],
            token_count=8,
            # Deliberately stale: target verify must ignore this phase's
            # prior extend metadata and use the verification width.
            extend_lens=[17, 23],
            width=4,
        )
        batch.input_ids = None
        token_slots, request_slots = _build_diag_es_candidate_slots(
            batch, _runner(True)
        )

        torch.testing.assert_close(
            token_slots,
            torch.tensor([3, 3, 3, 3, 1, 1, 1, 1], dtype=torch.int32),
        )
        self.assertEqual(request_slots, (3, 1))

    def test_target_verify_slots_fill_graph_buffer_and_zero_padded_rows(self):
        token_slots, _ = _build_diag_es_candidate_slots(
            _batch(
                ForwardMode.TARGET_VERIFY,
                [3, 1],
                token_count=8,
                extend_lens=[17, 23],
                width=4,
            ),
            _runner(True),
        )
        registry = build_decode_registry(
            device=torch.device("cpu"),
            max_bs=3,
            max_num_token=12,
            seq_len_fill_value=1,
            cache_loc_dtype=torch.int64,
            enable_diag_es=True,
            share_pool=False,
        )

        registry.fill_from(
            SimpleNamespace(es_candidate_slots=token_slots),
            raw_bs=2,
            padded_bs=3,
            raw_num_tokens=8,
            padded_num_tokens=12,
        )

        graph_slots = registry.get_slot("es_candidate_slots").buffer
        torch.testing.assert_close(graph_slots[:8], token_slots)
        torch.testing.assert_close(
            graph_slots[8:12], torch.zeros(4, dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
