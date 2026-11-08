"""Native nvCOMP CUDA conformance, two-buffer reuse and stream ownership.

Requires the paired Miles image with nvCOMP 6. Missing CUDA or dependencies fail.
Tests never feed malformed compressed streams to the GPU decoder.
"""

import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.weight_sync.gpu_delta.apply import prepare_status_check
from sglang.srt.weight_sync.gpu_delta.codec import NvcompDecoder
from sglang.srt.weight_sync.gpu_delta.host_worker import TENSOR_DTYPE
from sglang.srt.weight_sync.gpu_delta.receiver import (
    PreparedDelta,
    _plan_decode,
    _PreparedBatch,
)


def _frame_table(batches):
    rows = [row for batch in batches for row in batch]
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 4).T.copy()
    return table, list(map(len, batches))


def _encode(values, codec, offsets=None):
    from miles.utils.gpu_delta.nvcomp import NvcompCompressor

    device = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(device)
    tensors = [
        torch.frombuffer(bytearray(value), dtype=torch.uint8).to(device)
        for value in values
    ]
    result = NvcompCompressor(codec, device).compress(tensors, stream)
    stream.synchronize()
    assert result.statuses.tolist() == [0] * len(values)
    encoded_frames = [
        value[:size].cpu().numpy().tobytes()
        for value, size in zip(result.outputs, result.sizes.tolist())
    ]
    payload = bytearray()
    frames = []
    offset = 0
    for index, value in enumerate(values):
        if offsets is not None:
            offset = offsets[index]
        payload.extend(bytes((-len(payload)) % 16))
        encoded = encoded_frames[index]
        frames.append((len(payload), len(encoded), len(value), offset))
        payload.extend(encoded)
        offset += len(value)
    return payload, frames


# Outer Zstd is covered by the host suite. Exercise each native inner codec
# through the same production H2D/decode/apply loop, including empty batches.
@pytest.mark.parametrize("codec", ["snappy", "lz4", "bitcomp"])
def test_batched_plans_share_metadata_and_reuse_tensor_scratch(codec):
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder(device, codec)
    # Full 4 MiB nonzero frame, 1 MiB zero frame, and a partial final frame.
    values = [
        [bytes(1 << 20), bytes(range(256)) * 16384, bytes(range(253)) * 3],
        [bytes([7]) * (64 << 10), bytes(range(251)) * 3],
    ]
    payload_a, frames_a = _encode(values[0], codec)
    assert frames_a[1][2] == 4 << 20
    assert decoder._options.backend == 2
    # A different frame count, partial final frame and sparse output offsets
    # exercise views whose row stride is the complete metadata slab's width.
    payload_b, frames_b = _encode(values[1], codec, [128, 128 + (64 << 10) + 256])
    input_b = (len(payload_a) + 15) // 16 * 16
    host = torch.empty(input_b + len(payload_b), dtype=torch.uint8, pin_memory=True)
    host[: len(payload_a)].numpy()[:] = memoryview(payload_a)
    host[input_b : input_b + len(payload_b)].numpy()[:] = memoryview(payload_b)
    batches = [
        frames_a,
        [
            (input_b + offset, encoded, decoded, output)
            for offset, encoded, decoded, output in frames_b
        ],
    ]
    batches = [batches[index % 2] for index in range(5)]
    batches[-1] = []  # A fully omitted final batch still zeroes/reuses its slot.
    stream = torch.cuda.Stream(device=device)
    h2d_stream = torch.cuda.Stream(device=device)
    table, counts = _frame_table(batches)
    plan = decoder.prepare_batches(table, counts, host, stream)
    assert plan.stream is stream
    workspace = plan.workspace
    # Large outputs are allocated only at the paused binding boundary.
    decoded = [
        torch.empty(sum(map(len, values[0])), dtype=torch.uint8, device=device)
        for _ in range(2)
    ]
    plans = plan.bind_outputs(decoded)
    assert len(plan.input_slots) == 2
    assert plan.input_capacity_bytes <= 2 * host.numel()
    # Native status/size views exercise both typed allocation paths, including
    # nonzero-offset ring slots; successful decode below verifies their contents.
    assert workspace.statuses.dtype == torch.int32
    assert workspace.actual_sizes.dtype == plan.metadata.dtype == torch.int64
    assert workspace.statuses.device == workspace.actual_sizes.device == device
    del plan
    assert plans[0].metadata.untyped_storage().data_ptr() == (
        plans[1].metadata.untyped_storage().data_ptr()
    )
    assert plans[0].host_metadata.untyped_storage().data_ptr() == (
        plans[1].host_metadata.untyped_storage().data_ptr()
    )
    assert all(plan.metadata.stride(0) == sum(map(len, batches)) for plan in plans)
    assert plans[0].statuses.data_ptr() != plans[1].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() != plans[1].actual_sizes.data_ptr()
    assert plans[0].statuses.data_ptr() == plans[2].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() == plans[2].actual_sizes.data_ptr()
    assert all(plan.host_input is host for plan in plans)
    for batch in plans:
        assert batch.stream is stream
        assert batch._arguments[-1] == batch.stream.cuda_stream
        temporary = workspace.temporary
        assert batch._arguments[5:7] == (temporary.data_ptr(), temporary.numel())
    prepared = PreparedDelta.__new__(PreparedDelta)
    prepared.device = device
    prepared.stream, prepared.io_stream = stream, h2d_stream
    prepared.input_ready = [torch.cuda.Event() for _ in range(2)]
    prepared.input_free = [torch.cuda.Event() for _ in range(2)]
    # B leaves a prefix, an interior hole and a short tail; the unused part of
    # larger scratch retains its prior bytes and is not read by B's apply plan.
    output_sizes = [
        decoded[0].numel() if index % 2 == 0 else frames_b[-1][3] + frames_b[-1][2] + 64
        for index in range(len(batches))
    ]
    output_sizes[-1] = 64
    prepared_batches = []
    # Output allocation happened on the caller stream; the compute stream
    # waits before any decode or apply.
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.device(device), torch.cuda.stream(stream):
        prepared.error = torch.zeros(1, dtype=torch.int32, device=device)
        for index, (plan, frames, size) in enumerate(zip(plans, batches, output_sizes)):
            tensors = np.zeros(1, dtype=TENSOR_DTYPE)
            tensors["frame_count"] = len(frames)
            holes, cursor = [], 0
            for _, _, count, offset in frames:
                if offset > cursor:
                    holes.append((cursor, offset - cursor))
                cursor = offset + count
            if cursor < size:
                holes.append((cursor, size - cursor))
            tensors["gap_count"] = len(holes)
            publication = SimpleNamespace(
                tensors=tensors,
                frames=_frame_table([frames])[0],
                gaps=np.asarray(holes, dtype="<i8").reshape(-1, 2),
            )
            mapped, mapped_counts, gaps = _plan_decode(
                [(np.array([[0], [0]], dtype="<i8"), size, None)],
                publication,
                np.zeros(1, dtype=np.int64),
            )
            np.testing.assert_array_equal(mapped, _frame_table([frames])[0])
            assert mapped_counts == [len(frames)]
            prepared_batches.append(
                _PreparedBatch(
                    decoder=plan,
                    apply=None,
                    zero_ranges=[
                        decoded[index % 2][offset : offset + length]
                        for offset, length in gaps[0]
                    ],
                    check_status=prepare_status_check(plan, prepared.error),
                )
            )
        for output in decoded:
            output.fill_(0xA5)
        ready = torch.cuda.Event()
        ready.record(stream)
    h2d_stream.wait_event(ready)
    assert not prepared_batches[0].zero_ranges
    assert [view.numel() for view in prepared_batches[1].zero_ranges] == [128, 256, 64]
    observations = []

    # Run the production H2D/decode/apply loop, retaining snapshots as an ordered
    # apply consumer. Prepared metadata/output binding above is already real.
    # Distinct A/B bytes cross two ring boundaries, including the final empty
    # batch reusing the first output/status slot after both earlier decodes.
    def observe(batch):
        index = len(observations)
        PreparedDelta._apply_batch(prepared, batch)
        observations.append(
            (
                prepared.decoded[index % 2].clone(),
                batch.decoder.statuses.clone(),
                batch.decoder.actual_sizes.clone(),
            )
        )

    prepared.backend = SimpleNamespace(
        layout=SimpleNamespace(check_identity=lambda: None)
    )
    prepared._allocate_paused = lambda: None
    prepared._apply_batch = observe
    prepared.timings = {}
    prepared.raw_copies, prepared.derived = {}, []
    prepared.batches = list(prepared_batches)
    prepared.status_checks, prepared.apply_bindings, prepared.apply_source_patches = (
        [],
        [],
        [],
    )
    prepared.apply_fallback_groups = set()
    prepared.decoded, prepared.workspace, prepared.decode_plan = (
        decoded,
        workspace,
        None,
    )
    prepared.target_version, prepared.matrix_tensor_count, prepared.raw_tensor_count = (
        1,
        1,
        0,
    )
    prepared.h2d_bytes = 0
    error = prepared.error
    assert prepared.apply()["applied"]
    prepared.error = error
    del prepared._apply_batch, prepared._allocate_paused
    expected_slots = [bytearray([0xA5]) * output.numel() for output in decoded]
    for index, (output, statuses, sizes) in enumerate(observations):
        expected = expected_slots[index % 2]
        expected[: output_sizes[index]] = bytes(output_sizes[index])
        source_values = values[index % 2] if batches[index] else []
        for frame, value in zip(batches[index], source_values):
            expected[frame[3] : frame[3] + len(value)] = value
        assert statuses.tolist() == [0] * len(batches[index])
        assert sizes.tolist() == list(map(len, source_values))
        assert plans[index].expected_sizes.tolist() == sizes.tolist()
        assert bytes(output.cpu().numpy()) == expected
    assert prepared.error.item() == 0
    assert bytes(host[: len(payload_a)].numpy()) == payload_a
    assert bytes(host[input_b : input_b + len(payload_b)].numpy()) == payload_b
    # Each slot's callback reads its own status/size row. Mutate metadata only
    # after the real CUDA work completes.
    for index, batch in enumerate(prepared_batches[:2]):
        field = batch.decoder.statuses if index % 2 == 0 else batch.decoder.actual_sizes
        prepared.error.zero_()
        before = field[0].clone()
        field[0].add_(1)
        batch.check_status()
        assert prepared.error.item() == 1
        field[0].copy_(before)
        batch.check_status()
        assert prepared.error.item() == 1
    # Inject status metadata only, never a malformed GPU-compressed stream.
    # Failure in either CTA and a size mismatch all set the same sticky flag;
    # a later successful batch must not clear it.
    status = SimpleNamespace(
        statuses=torch.zeros(1025, dtype=torch.int32, device=device),
        actual_sizes=torch.ones(1025, dtype=torch.int64, device=device),
        expected_sizes=torch.ones(1025, dtype=torch.int64, device=device),
    )
    check = prepare_status_check(status, prepared.error)
    for field, index in (
        (status.statuses, 0),
        (status.statuses, 1024),
        (status.actual_sizes, 1024),
    ):
        prepared.error.zero_()
        before = field[index].clone()
        field[index].add_(1)
        check()
        assert prepared.error.item() == 1
        field[index].copy_(before)
        check()
        assert prepared.error.item() == 1
    # Exercise the same final-release path after all H2D/decode/apply work drains.
    # Per-batch leases keep every decoded slot alive after the plan is discarded.
    output_refs = [weakref.ref(output) for output in decoded]
    prepared.batches = prepared_batches
    prepared.status_checks = []
    prepared.apply_bindings, prepared.apply_source_patches = [], []
    prepared.apply_fallback_groups = set()
    prepared.raw_copies = {}
    prepared.decoded = decoded
    prepared.workspace = workspace
    prepared.decode_plan = None
    del decoded
    assert all(ref() is not None for ref in output_refs)
    prepared._release_gpu()
    plans.clear()
    del plan, batch
    assert all(ref() is None for ref in output_refs)
    # Original host input is released only after all streams' readers finish.
    del host


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
