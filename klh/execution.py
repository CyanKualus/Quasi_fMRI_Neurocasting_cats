"""Bounded, ordered CPU work outside the superlet process pool."""
from __future__ import annotations

import hashlib
import os
import threading
import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
from threadpoolctl import threadpool_limits

DEFAULT_WORKERS = 2
_workers = DEFAULT_WORKERS
_local = threading.local()
_blas_lock = threading.RLock()


def configure_workers(workers: int) -> int:
    """0/1 runs serially; higher counts are capped to available CPUs."""
    global _workers
    if int(workers) != workers or workers < 0:
        raise ValueError("analysis workers must be a non-negative integer")
    _workers = min(int(workers), os.cpu_count() or 1)
    return _workers


def available_memory() -> int | None:
    """Available physical bytes, when the platform exposes them."""
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_uint32),
                            ("load", ctypes.c_uint32)] + [
                                (name, ctypes.c_uint64) for name in
                                ("total", "available", "total_page", "available_page",
                                 "total_virtual", "available_virtual", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
        elif hasattr(os, "sysconf"):
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        pass
    return None


def ordered_map(function, items, *, blas=False, max_workers=None,
                working_bytes=0):
    """Return results in input order, propagating errors and joining workers.

    Tasks share read-only inputs and own their outputs. Nested calls stay
    serial. BLAS limits apply around the entire executor, never independently
    in its threads: the native-library setting is process-wide.
    """
    items = list(items)
    count = min(_workers, len(items))
    if max_workers is not None:
        count = min(count, max_workers)
    if count > 1 and working_bytes:
        available = available_memory()
        if available is not None:
            # Leave headroom for the UI, outputs and the operating system.
            # Serial execution remains available even below this estimate.
            count = min(count, max(1, (available - (256 << 20)) // working_bytes))
    if count < 2 or getattr(_local, "active", False):
        return [function(item) for item in items]

    def invoke(item):
        _local.active = True
        try:
            return function(item)
        finally:
            _local.active = False

    with _blas_lock if blas else nullcontext():
        with threadpool_limits(limits=1, user_api="blas") if blas else nullcontext():
            with ThreadPoolExecutor(max_workers=count,
                                    thread_name_prefix="klh-analysis") as pool:
                return list(pool.map(invoke, items))


def array_digest(*arrays) -> bytes:
    """Content key, including in-place edits, without copying a whole EEG array."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        digest.update(repr((array.shape, array.dtype.str)).encode("ascii"))
        if not array.size:
            continue
        if array.flags.c_contiguous:
            digest.update(b"C")
            digest.update(memoryview(array).cast("B"))
        elif array.flags.f_contiguous:
            digest.update(b"F")
            digest.update(memoryview(array.T).cast("B"))
        else:
            digest.update(b"C")
            for block in np.nditer(array, flags=["external_loop", "buffered",
                                                "zerosize_ok"],
                                   op_flags=["readonly"], order="C",
                                   buffersize=131072):
                digest.update(np.ascontiguousarray(block).tobytes())
    return digest.digest()
