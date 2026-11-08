# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Persistent engine-shared host mappings registered for CUDA H2D reads."""

import ctypes
from functools import cache


@cache
def _driver():
    from cuda.bindings import driver

    _check(driver.cuInit(0), "cuInit")
    return driver


def _check(result, operation):
    code = result[0]
    if code:
        raise RuntimeError(f"GPU-delta allocation {operation} failed: CUDA {code}")
    return result[1] if len(result) > 1 else None


class HostMapping:
    """Own one registered shared mapping across updates that fit its capacity.

    The caller establishes the CUDA context before construction and drains H2D
    readers before closing or replacing the mapping.
    """

    def __init__(self, mapping):
        self.mapping = mapping
        self.view = memoryview(mapping)
        self.registered = False
        try:
            self.driver = driver = _driver()
            self.context = _check(driver.cuCtxGetCurrent(), "cuCtxGetCurrent")
            self.address = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
            _check(
                driver.cuMemHostRegister(self.address, len(mapping), 0),
                "cuMemHostRegister",
            )
            self.registered = True
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.registered:
            driver = self.driver
            # Clear may run on another thread; the model owns this context.
            _check(driver.cuCtxPushCurrent(self.context), "cuCtxPushCurrent")
            try:
                _check(driver.cuMemHostUnregister(self.address), "cuMemHostUnregister")
                self.registered = False
            finally:
                _check(driver.cuCtxPopCurrent(), "cuCtxPopCurrent")
        if self.view is not None:
            self.view.release()
            self.view = None
        # A drained worker's exception traceback can still retain a byte view.
        # Its final reference closes the mmap without invalidating that lease.
        self.mapping = None
