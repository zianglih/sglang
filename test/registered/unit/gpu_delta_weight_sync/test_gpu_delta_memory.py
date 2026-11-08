"""Shared pinned-mapping registration and CUDA context ownership."""

import ctypes
import mmap
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.weight_sync.gpu_delta import memory as memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("registration_failure", [False, True])
def test_shared_host_registration_owns_mapping_until_close(
    monkeypatch, registration_failure
):
    mapping = mmap.mmap(-1, 4096)
    calls = []
    contexts = [13]

    def register(address, size, flags):
        assert size == len(mapping) and flags == 0
        calls.append(("register", address))
        return (int(registration_failure),)

    def unregister(address):
        assert not mapping.closed and contexts[-1] == 13
        calls.append(("unregister", address))
        return (0,)

    def push(context):
        contexts.append(context)
        calls.append(("push", context))
        return (0,)

    def pop():
        context = contexts.pop()
        calls.append(("pop", context))
        return 0, context

    monkeypatch.setattr(
        memory,
        "_driver",
        lambda: SimpleNamespace(
            cuMemHostRegister=register,
            cuMemHostUnregister=unregister,
            cuCtxGetCurrent=lambda: (0, contexts[-1]),
            cuCtxPushCurrent=push,
            cuCtxPopCurrent=pop,
        ),
    )
    if registration_failure:
        with pytest.raises(RuntimeError, match="cuMemHostRegister"):
            memory.HostMapping(mapping)
        assert len(calls) == 1
    else:
        allocation = memory.HostMapping(mapping)
        allocation.view[:4] = b"test"
        escaped = allocation.view[:4]
        assert allocation.mapping is mapping and bytes(escaped) == b"test"
        assert ctypes.string_at(allocation.address, 4) == b"test"
        contexts[-1] = 29  # Explicit clear can run on a different caller thread.
        allocation.close()
        allocation.close()
        assert contexts == [29]
        assert allocation.mapping is None and allocation.view is None
        assert bytes(escaped) == b"test"
        escaped.release()
        assert calls == [
            ("register", allocation.address),
            ("push", 13),
            ("unregister", allocation.address),
            ("pop", 13),
        ]
    assert not mapping.closed  # This local owner outlives the allocation wrapper.
    mapping.close()
    assert mapping.closed


def test_cuda_driver_initialization_is_cached_and_checks_status(monkeypatch):
    calls = []
    driver = SimpleNamespace(cuInit=lambda flags: (calls.append(flags) or 0,))
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(driver=driver))
    memory._driver.cache_clear()
    try:
        assert memory._driver() is driver
        assert memory._driver() is driver
        assert calls == [0]
        memory._driver.cache_clear()
        driver.cuInit = lambda flags: (1,)
        with pytest.raises(RuntimeError, match="cuInit failed"):
            memory._driver()
        assert memory._driver.cache_info().currsize == 0
    finally:
        memory._driver.cache_clear()
