from types import SimpleNamespace

import torch
from sglang.srt.diag_es.mtp_kv_replay import (
    DiagESMTPDraftKVReplay,
    is_target_driven_mtp_forward,
    mtp_replay_bytes_per_token,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_capture_modes_exclude_candidate_dependent_draft_decode():
    assert is_target_driven_mtp_forward(ForwardMode.EXTEND)
    assert is_target_driven_mtp_forward(ForwardMode.DRAFT_EXTEND_V2)
    assert not is_target_driven_mtp_forward(ForwardMode.DECODE)
    assert not is_target_driven_mtp_forward(ForwardMode.TARGET_VERIFY)
    assert mtp_replay_bytes_per_token(2048) == 4104


def test_capture_persists_exact_positions_only_for_target_driven_rows():
    replay = DiagESMTPDraftKVReplay.__new__(DiagESMTPDraftKVReplay)
    replay.hidden_size = 2048
    replay.activation_buffer = torch.zeros(8, 2048, dtype=torch.bfloat16)
    replay.position_buffer = torch.zeros(8, dtype=torch.int64)
    loc = torch.tensor([2, 6], dtype=torch.int64)
    hidden = torch.stack(
        [
            torch.full((2048,), 3, dtype=torch.bfloat16),
            torch.full((2048,), 5, dtype=torch.bfloat16),
        ]
    )

    replay.capture(
        hidden,
        torch.tensor([101, 207], dtype=torch.int64),
        ForwardMode.DRAFT_EXTEND_V2,
        loc,
    )
    replay.capture(
        torch.zeros_like(hidden),
        torch.tensor([999, 999], dtype=torch.int64),
        ForwardMode.DECODE,
        loc,
    )

    torch.testing.assert_close(replay.activation_buffer[loc], hidden)
    torch.testing.assert_close(
        replay.position_buffer[loc], torch.tensor([101, 207], dtype=torch.int64)
    )


def test_replay_batches_mixed_requests_and_stops_at_old_prefix_for_topk2():
    replay = DiagESMTPDraftKVReplay.__new__(DiagESMTPDraftKVReplay)
    replay.chunk_tokens = 4
    replay.activation_buffer = torch.empty(1)
    replay.position_buffer = torch.zeros(40, dtype=torch.int64)
    replay.position_buffer[[10, 11, 12, 30, 31]] = torch.tensor(
        [100, 102, 105, 7, 11], dtype=torch.int64
    )
    calls = []

    def replay_chunk(*, loc, positions, candidate_slots):
        calls.append(
            (
                loc.tolist(),
                positions.tolist(),
                candidate_slots.tolist(),
            )
        )

    replay._replay_chunk = replay_chunk
    req_to_token = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [10, 11, 12, 13, 14, 15, 16],
            [20, 21, 22, 23, 24, 25, 26],
            [30, 31, 32, 33, 34, 35, 36],
        ],
        dtype=torch.int32,
    )
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(diag_es_mtp_slot=7),
            SimpleNamespace(diag_es_mtp_slot=8),
            SimpleNamespace(diag_es_mtp_slot=9),
        ],
        # In topk=2/steps=2/draft_tokens=3, verify has already compacted the
        # accepted path, but batch.seq_lens remains the committed OLD prefix.
        # Locations at/after these lengths belong to the next draft-extend
        # window and must not be replayed here.
        seq_lens_cpu=torch.tensor([3, 5, 2], dtype=torch.int64),
        seq_lens=torch.tensor([3, 5, 2], dtype=torch.int64),
        req_pool_indices=torch.tensor([1, 2, 3], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([1, 2, 3], dtype=torch.int64),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )

    stats = replay.replay_transitioned_prefixes(batch, (True, False, True))

    assert calls == [
        ([10, 11, 12, 30], [100, 102, 105, 7], [7, 7, 7, 9]),
        ([31], [11], [9]),
    ]
    assert stats.replayed_rows == 5
    assert stats.transitioned_requests == 2
    assert stats.request_rows == (3, 0, 2)
