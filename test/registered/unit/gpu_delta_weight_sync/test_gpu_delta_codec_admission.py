"""CPU codec admission, outer payload integrity and prepared metadata ownership."""

import ctypes
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zstandard as zstd

from sglang.srt.weight_sync.gpu_delta.host_worker import HostPayloadPool, validate_codec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/weight_sync/gpu_delta/codec.py"
)
_spec = importlib.util.spec_from_file_location("gpu_delta_codec_under_test", _path)
codec = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = codec
_spec.loader.exec_module(codec)


def _frame_table(batches):
    rows = [row for batch in batches for row in batch]
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 4).T.copy()
    return table, list(map(len, batches))


@pytest.mark.parametrize(
    "name,algorithm,options_type,reserved_offset",
    [
        ("snappy", "Snappy", codec._SnappyOptions, 8),
        ("lz4", "LZ4", codec._Lz4Options, 16),
        ("bitcomp", "Bitcomp", codec._BitcompOptions, 4),
    ],
)
def test_codec_abi_uses_cuda_options(
    monkeypatch, name, algorithm, options_type, reserved_offset
):
    # nvCOMP 6 retains the hardware-sort ABI field even for CUDA decoding.
    assert ctypes.sizeof(options_type) == 64
    assert ctypes.alignment(options_type) == ctypes.alignment(ctypes.c_int)
    assert options_type.backend.offset == 0
    if name != "bitcomp":
        assert options_type.sort_before_hw_decompress.offset == 4
    assert options_type.reserved.offset == reserved_offset
    if name == "lz4":
        assert options_type.data_type.offset == 8
        assert options_type.bitshuffle_mode.offset == 12
    calls = []

    class Function:
        def __init__(self, suffix):
            self.suffix = suffix

        def __call__(self, *arguments):
            options = arguments[0 if self.suffix == "GetRequiredAlignments" else 2]
            assert type(options) is options_type
            calls.append((self.suffix, bytes(options)))
            if self.suffix == "GetRequiredAlignments":
                ctypes.cast(arguments[1], ctypes.POINTER(codec._Alignments))[0] = (
                    codec._Alignments(16, 16, 16)
                )
            else:
                ctypes.cast(arguments[3], ctypes.POINTER(ctypes.c_size_t))[0] = 64
            return 0

    functions = {
        "nvcompBatched" + algorithm + "Decompress" + suffix: Function(suffix)
        for suffix in ("GetRequiredAlignments", "GetTempSize", "Async")
    }
    distribution = SimpleNamespace(version="6.0.0.12", locate_file=lambda path: path)
    monkeypatch.setattr(
        codec.importlib.metadata, "distribution", lambda _: distribution
    )

    def load_library(path):
        assert path == "nvidia/libnvcomp/lib64/libnvcomp.so.6"
        return SimpleNamespace(**functions)

    monkeypatch.setattr(codec.ctypes, "CDLL", load_library)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    decoder = codec.NvcompDecoder(torch.device("cuda", 0), name)
    assert decoder.inner_codec == name
    assert decoder._options.backend == 2
    assert bytes(decoder._options)[4:] == bytes(60)
    align_function = functions[
        f"nvcompBatched{algorithm}DecompressGetRequiredAlignments"
    ]
    assert align_function.argtypes == [options_type, ctypes.POINTER(codec._Alignments)]
    assert decoder._temporary.argtypes == [
        ctypes.c_size_t,
        ctypes.c_size_t,
        options_type,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    assert decoder._decode.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        options_type,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    assert decoder._decode.restype is ctypes.c_int
    # Alignment and workspace queries must receive the same admitted options.
    geometry = (1, 64, 64)
    assert decoder.temporary_bytes(geometry, 7) == 64
    assert [suffix for suffix, _ in calls] == [
        "GetRequiredAlignments",
        "GetTempSize",
    ]
    assert calls[0][1] == calls[1][1]
    assert decoder.temporary_bytes(geometry, 7) == 64
    distribution.version = "5.3.0.16"
    with pytest.raises(RuntimeError, match="nvCOMP 6 ABI"):
        codec.NvcompDecoder(decoder.device, name)


def test_workspace_query_cache_uses_geometry_and_stream_and_preserves_failure(
    monkeypatch,
):
    # Exercise the real query/cache code without a CUDA allocation or library.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cuda", 0)
    decoder.inner_codec = "snappy"
    decoder._options = codec._SnappyOptions()
    decoder._temporary_sizes = {}
    calls, fail = [], True

    def query(count, maximum, options, size, total, stream):
        calls.append((count, maximum, total, stream))
        if total == 31 and fail:
            return 1
        # Zero is also a valid cache entry; it must not trigger another query.
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        return 0

    decoder._temporary = query
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    assert decoder.temporary_bytes((0, 0, 0), 7) == 0
    assert decoder.temporary_bytes((2, 20, 30), 7) == 0
    assert decoder.temporary_bytes((2, 20, 30), 7) == 0
    assert calls == [(2, 20, 30, 7)]
    # Same count and maximum, different total: the native query must run again.
    with pytest.raises(RuntimeError, match="status=1"):
        decoder.temporary_bytes((2, 20, 31), 7)
    fail = False
    assert decoder.temporary_bytes((2, 20, 31), 7) == 0
    assert decoder.temporary_bytes((2, 20, 31), 7) == 0
    # Query the actual compute stream once, then retain its own zero-sized cache entry.
    assert decoder.temporary_bytes((2, 20, 30), 9) == 0
    assert decoder.temporary_bytes((2, 20, 30), 9) == 0
    assert decoder.temporary_bytes((2, 20, 30), 7) == 0
    assert calls == [(2, 20, 30, 7), (2, 20, 31, 7), (2, 20, 31, 7), (2, 20, 30, 9)]


def test_prepared_stream_scratch_and_slot_status_ownership(monkeypatch):
    # Real tensor views/pointers exercise nonzero, non-aligned native workspace
    # requirements without a device. Native cases cover decoding and buffer reuse.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cpu")
    decoder.alignments = codec._Alignments(16, 16, 16)
    decoder._options = codec._SnappyOptions()
    decoder.temporary_bytes = lambda geometry, stream: geometry[0] * 13
    recorded = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: recorded.append(
            (tensor.untyped_storage().data_ptr(), stream.cuda_stream)
        ),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    stream = SimpleNamespace(device=decoder.device, cuda_stream=1)
    counts = [1, 2, 0, 3] * 3
    workspace = decoder._allocate_workspace(
        [(count, 32, count * 32) for count in counts], stream.cuda_stream
    )
    metadata = torch.zeros((4, sum(counts)), dtype=torch.int64)
    host_metadata = metadata.clone()
    plan = codec.PreparedDecodePlan(
        decoder,
        counts,
        torch.empty(128, dtype=torch.uint8),
        workspace,
        host_metadata,
        metadata,
        stream,
        np.zeros(sum(counts), dtype=np.int64),
        [128, 128],
        [0, 0],
        np.zeros((2, sum(counts)), dtype=np.int64),
    )
    assert workspace.temporary.shape == (39,)
    assert workspace.temporary.data_ptr() % decoder.alignments.temp == 0
    assert workspace.statuses.shape == workspace.actual_sizes.shape == (2, 3)
    for index, batch in enumerate(plan.batches):
        slot = index % 2
        assert batch.stream is stream
        assert batch._arguments[-1] == stream.cuda_stream
        assert batch._arguments[5:7] == (
            workspace.temporary.data_ptr(),
            workspace.temporary.numel(),
        )
        assert batch.statuses.storage_offset() == slot * 3
        assert batch.actual_sizes.storage_offset() == slot * 3
        assert batch._arguments[3] == batch.actual_sizes.data_ptr()
        assert batch._arguments[9] == batch.statuses.data_ptr()
        if not counts[index]:
            batch.enqueue()  # Empty metadata must not call the native decoder.
    outputs = [torch.empty(128, dtype=torch.uint8) for _ in range(2)]
    assert plan.bind_outputs(outputs) is plan.batches
    assert recorded == [
        (tensor.untyped_storage().data_ptr(), stream.cuda_stream) for tensor in outputs
    ]
    offset = 0
    for index, count in enumerate(counts):
        assert (
            metadata[3, offset : offset + count].tolist()
            == [outputs[index % 2].data_ptr()] * count
        )
        offset += count


def test_host_input_plans_reject_invalid_geometry_before_device_work(monkeypatch):
    # Native tests exercise successful slab construction/slot reuse. Keep the
    # vector arithmetic and output-lease failures here without a fake CUDA API.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cpu")
    decoder.inner_codec = "snappy"
    decoder.alignments = codec._Alignments(16, 16, 16)
    table, counts = _frame_table([[(0, 20, 64, 0), (32, 17, 32, 128)]])
    geometry, bounds, remainders = decoder._frame_geometry(table, counts, 0, 512)
    assert (geometry, bounds, remainders) == (
        [(2, 64, 96)],
        [160, 0],
        [0, None],
    )
    for rows, error in (
        ([(496, 32, 64, 0)], "outside input"),
        ([(np.iinfo(np.int64).max, 32, 64, 0)], "outside input"),
        ([(0, 32, 64, np.iinfo(np.int64).max)], "Output arithmetic"),
        ([(1, 32, 64, 0)], "Misaligned encoded"),
        ([(0, 32, 64, 0), (32, 32, 64, 32)], "Overlapping"),
        ([(0, 32, 64, 0), (32, 32, 64, 65)], "Misaligned decoded"),
    ):
        with pytest.raises(ValueError, match=error):
            decoder._frame_geometry(*_frame_table([rows]), 0, 512)
    for invalid in (table.astype(np.int32), table[:, ::-1], table[:, :-1]):
        with pytest.raises(ValueError, match="contiguous int64"):
            decoder._frame_geometry(invalid, counts, 0, 512)
    maximum = codec._MAX_FRAME_BYTES
    assert decoder._frame_geometry(*_frame_table([[(0, 20, maximum, 0)]]), 0, 512)[
        1
    ] == [maximum, 0]
    with pytest.raises(ValueError, match="<=4 MiB output"):
        decoder._frame_geometry(*_frame_table([[(0, 20, maximum + 1, 0)]]), 0, 512)
    assert decoder._frame_geometry(*_frame_table([[], []]), 0, 512) == (
        [(0, 0, 0), (0, 0, 0)],
        [0, 0],
        [None, None],
    )

    plan = object.__new__(codec.PreparedDecodePlan)
    plan.decoder = decoder
    plan.output_bounds, plan.output_remainders = [128, 128], [0, 0]
    plan.decoded_slots = []
    # Every malformed output must fail before CUDA context, allocation or upload.
    monkeypatch.setattr(torch.cuda, "device", lambda _: pytest.fail("entered CUDA"))
    with pytest.raises(ValueError, match="Decoder stream/device mismatch"):
        decoder.prepare_batches(
            table,
            counts,
            torch.empty(512, dtype=torch.uint8),
            SimpleNamespace(device=torch.device("cuda", 0)),
        )
    a, b = [torch.empty(256, dtype=torch.uint8) for _ in range(2)]
    for slots, error in (
        ((a, b[:64]), "Out-of-bounds"),
        ((a, b[1:]), "Misaligned decoded"),
        ((a, a[16:]), "must not overlap"),
        ((a,), "count differs"),
        ((a, b, a), "count differs"),
    ):
        with pytest.raises(ValueError, match=error):
            plan.bind_outputs(slots)
        assert not plan.decoded_slots


def test_cuda_input_packing_preserves_frame_bytes_and_alignment():
    # Reordered, duplicate and disjoint frames must share merged H2D spans
    # without copying large holes in the persistent host arena.
    inputs = np.array([160, 0, 16, 160, 1 << 20], dtype=np.int64)
    sizes = np.array([8, 7, 20, 8, 9], dtype=np.int64)
    offsets, ranges, capacity = codec._pack_inputs(inputs, sizes, 16)
    assert ranges == ((0, 0, 36), (160, 48, 8), (1 << 20, 64, 9))
    assert capacity == 73
    assert offsets.tolist() == [48, 0, 16, 48, 64]
    source = np.arange((1 << 20) + 9, dtype=np.uint8)
    destination = np.zeros(capacity, dtype=np.uint8)
    for start, offset, length in ranges:
        destination[offset : offset + length] = source[start : start + length]
    for start, offset, size in zip(inputs, offsets, sizes):
        np.testing.assert_array_equal(
            destination[offset : offset + size], source[start : start + size]
        )
    assert codec._pack_inputs(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64), 16
    )[1:] == ((), 0)


class TestOuterZstd(unittest.TestCase):
    def test_manifest_codec_and_frame_geometry_admission(self):
        admitted = dict(codec="snappy-zstd", frame_bytes=1 << 20)
        for codec in ("snappy-zstd", "lz4-zstd", "lz4", "bitcomp", "bitcomp-zstd"):
            for size in (1 << 16, 192 << 10, 1 << 19, 1 << 20, 2 << 20, 4 << 20):
                with self.subTest(codec=codec, size=size):
                    validate_codec(admitted | {"codec": codec, "frame_bytes": size})
        for patch_value in (
            {"codec": "zstd"},
            {"codec": None},
            {"frame_bytes": 0},
            {"frame_bytes": -1},
            {"frame_bytes": (4 << 20) + 1},
            {"frame_bytes": 1.5},
            {"frame_bytes": True},
        ):
            with (
                self.subTest(patch=patch_value),
                self.assertRaisesRegex(ValueError, "GPU delta requires"),
            ):
                validate_codec(admitted | patch_value)

    def test_gpu_outer_chunks_decode_directly_to_one_destination(self):
        values = [bytes(range(256)) * 4096, bytes(range(19))]
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        for known_size in (False, True):
            payload, chunks = bytearray(), []
            for index, value in enumerate(values):
                payload.extend(bytes((-len(payload)) % 16))
                compressed = zstd.ZstdCompressor(
                    level=1, write_content_size=known_size
                ).compress(value)
                chunks.append(
                    dict(
                        encoded_offset=len(payload),
                        encoded_bytes=len(compressed),
                        decoded_offset=index * (1 << 20),
                        decoded_bytes=len(value),
                    )
                )
                payload.extend(compressed)
            target = bytearray(sum(map(len, values)))
            pool.executor.submit(
                pool.decode_zstd,
                zstd.BufferWithSegments(
                    payload,
                    np.asarray(
                        [
                            [chunk["encoded_offset"], chunk["encoded_bytes"]]
                            for chunk in chunks
                        ],
                        dtype=np.uint64,
                    ),
                ),
                [chunk["decoded_bytes"] for chunk in chunks],
                [chunk["decoded_offset"] for chunk in chunks],
                memoryview(target),
            ).result()
            self.assertEqual(target, b"".join(values))

    def test_invalid_frames_fail_without_publishing_a_decoded_result(self):
        raw = bytes(range(64)) * 128
        encoded = zstd.ZstdCompressor(write_checksum=True).compress(raw)
        corrupt = bytearray(encoded)
        corrupt[-1] ^= 1
        pool = HostPayloadPool(2)
        self.addCleanup(pool.close)
        for payload, size in (
            (encoded, len(raw) - 1),
            (encoded, len(raw) + 1),
            (encoded[:-1], len(raw)),
            (corrupt, len(raw)),
            (b"not-zstd", len(raw)),
        ):
            target = bytearray(2 * len(raw) + 1)
            with self.subTest(size=size, payload=bytes(payload[:8])):
                with self.assertRaises(zstd.ZstdError):
                    pool.executor.submit(
                        pool.decode_zstd,
                        zstd.BufferWithSegments(
                            encoded + bytes(payload),
                            np.asarray(
                                [[0, len(encoded)], [len(encoded), len(payload)]],
                                dtype=np.uint64,
                            ),
                        ),
                        [len(raw), size],
                        [0, len(raw)],
                        memoryview(target),
                    ).result()
                self.assertEqual(target, bytes(len(target)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
