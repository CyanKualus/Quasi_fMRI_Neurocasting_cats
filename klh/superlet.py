"""Adaptive superresolution wavelet (superlet) transform — port of ``aslt.m``.

Original author: Harald Barzan (April 2019). Moca et al., "Superlets:
time-frequency super-resolution using wavelet sets".

Produces a (n_frequencies, n_samples) power spectrogram. For each frequency,
the closest integer order in ``ord`` selects how many Morlet wavelets (with an
increasing number of cycles) are combined via geometric mean.

The pipeline's 3-30 Hz, order 1-30 configuration means 826 wavelets, and a run
issues six to ten transforms. Rather than rebuild the wavelet set and
re-transform the signal for every one of those convolutions, the set and its
spectra are cached and each buffer is transformed once per overlap-save block.
``tests/test_transform_equivalence.py`` pins the result against the direct
per-wavelet ``fftconvolve`` this replaced.

Frequencies are independent of one another, so the spectrum can also be split
across worker processes; see :func:`configure_workers`. Threads do not help --
the per-block bookkeeping is Python-level and holds the GIL -- and the cost is
paid in memory, one wavelet bank per worker.
"""
from __future__ import annotations

import atexit
import os
import multiprocessing
import numpy as np
import scipy.fft as sfft

# Overlap-save block length.  Small blocks keep every transform inside L2/L3 --
# measurably faster than one transform over the whole concatenated signal -- and
# keep the cached wavelet spectra to tens of megabytes rather than hundreds.
_BLOCK = 4096
# Wavelet spectra transformed per batched inverse FFT.
_CHUNK = 8
# Cached wavelet spectra are large, so the cache is bounded by weight rather
# than by entry count: one full 3-30 Hz order 1-30 bank is ~54 MB.
_BANK_BUDGET_BYTES = 256 << 20
_BANK: dict = {}

# Worker processes for the frequency split.  ``None`` means "not configured
# yet" and is resolved once, from KLH_SUPERLET_WORKERS or the default below.
_WORKERS: int | None = None
_POOL = None
# Measured on a 6-core i7-8700: 4 processes gave 2.4x, 6 gave 1.2x because
# hyperthread siblings contend for the same FPU and cache.  Each worker is a
# fresh interpreter that re-imports the application, so the setting buys speed
# with roughly 170 MB of resident memory apiece.
DEFAULT_WORKERS = 4
# Below this much work the pool costs more than it saves.
_MIN_POINTS_FOR_POOL = 20_000


def configure_workers(workers: int | None) -> int:
    """Set how many processes split the frequency axis; return what was set.

    ``0`` or ``1`` keeps everything in this process. ``None`` restores the
    default, which honours ``KLH_SUPERLET_WORKERS`` if it is set. Changing the
    count shuts any existing pool down, so callers may set it freely.
    """
    global _WORKERS
    if workers is None:
        _WORKERS = None
        shutdown_workers()
        return _resolve_workers()
    workers = int(workers)
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if workers != _WORKERS:
        shutdown_workers()
    _WORKERS = workers
    return _WORKERS


def _resolve_workers() -> int:
    global _WORKERS
    if _WORKERS is not None:
        return _WORKERS
    raw = os.environ.get("KLH_SUPERLET_WORKERS")
    if raw is not None:
        try:
            _WORKERS = max(0, int(raw))
        except ValueError:
            _WORKERS = DEFAULT_WORKERS
    else:
        _WORKERS = DEFAULT_WORKERS
    _WORKERS = min(_WORKERS, os.cpu_count() or 1)
    return _WORKERS


def _in_worker_process() -> bool:
    """A worker must never build a pool of its own."""
    try:
        return multiprocessing.current_process().name != "MainProcess"
    except Exception:
        return True


def _frequency_groups(n_frequencies, n_points):
    """Frequency indices per worker, or ``None`` to stay serial.

    Superresolution order rises with frequency, so a wavelet count -- and
    therefore the work -- rises across the spectrum. Interleaving rather than
    slicing contiguously gives every worker a comparable share.
    """
    workers = _resolve_workers()
    if (workers < 2 or _in_worker_process()
            or n_points < _MIN_POINTS_FOR_POOL
            or n_frequencies < 2 * workers):
        return None
    return [np.arange(w, n_frequencies, workers) for w in range(workers)]


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        # Workers persist for the session so each pays the interpreter start
        # and its share of the wavelet bank once, not once per transform.
        _POOL = ProcessPoolExecutor(max_workers=_resolve_workers())
    return _POOL


def shutdown_workers() -> None:
    """Release the worker pool, if one is running."""
    global _POOL
    pool, _POOL = _POOL, None
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:            # Python < 3.9
            pool.shutdown(wait=False)


atexit.register(shutdown_workers)


def _fix(x):
    """MATLAB ``fix`` — truncate toward zero."""
    return np.trunc(x).astype(int)


def cxmorlet(Fc, Nc, Fs):
    """Complex Morlet wavelet, center freq Fc, Nc cycles, sampling rate Fs."""
    sd = (Nc / 2.0) * (1.0 / Fc) / 2.5
    wl = 2 * int(np.floor(int(_fix(6 * sd * Fs)) / 2)) + 1
    off = int(_fix(wl / 2))
    i = np.arange(wl)
    t = (i - off) / Fs
    cnorm = 1.0 / (sd * np.sqrt(2 * np.pi))
    envelope = cnorm * np.exp(-(t ** 2) / (2 * sd ** 2))
    w = np.exp(2j * np.pi * Fc * t) * envelope
    gi = np.sum(envelope)
    return w / gi


def _derive_orders(n_frequencies, ord_interval):
    """Per-frequency superresolution order, as ``aslt.m`` picks it."""
    if ord_interval is not None:
        order_ls = _fix(np.linspace(
            ord_interval[0], ord_interval[1], n_frequencies))
    else:
        order_ls = np.ones(n_frequencies, dtype=int)
    return np.maximum(order_ls, 1)


def _wavelet_length(Fc, Nc, Fs):
    """Length :func:`cxmorlet` will produce, without building the wavelet."""
    sd = (Nc / 2.0) * (1.0 / Fc) / 2.5
    return 2 * int(np.floor(int(_fix(6 * sd * Fs)) / 2)) + 1


def _geometry(Fs, F, Ncyc, order_ls, mult):
    """Padding and block length, derived from the *whole* frequency list.

    Both depend only on the longest wavelet. Computing them over the full
    spectrum and handing them to every worker is what makes a split run
    bit-identical to a serial one: each frequency then meets the same block
    boundaries, and so the same floating-point additions, either way.
    """
    max_wl = 0
    for fi in range(F.size):
        for oi in range(int(order_ls[fi])):
            cycles = Ncyc * (oi + 1) if mult else Ncyc + oi
            max_wl = max(max_wl, _wavelet_length(F[fi], cycles, Fs))
    padding = max_wl // 2
    # Overlap-save needs the block to exceed the longest wavelet; doubling it
    # keeps the per-block yield (n_fft - max_wl + 1) from collapsing.
    n_fft = sfft.next_fast_len(max(_BLOCK, 2 * max_wl))
    return padding, n_fft, max_wl


def _wavelet_bank(Fs, F, Ncyc, order_ls, mult, geometry):
    """Build (and cache) the wavelet set and its block-length spectra.

    Rebuilding the set costs one ``cxmorlet`` call and one FFT per wavelet --
    826 of each for the pipeline's 3-30 Hz, order 1-30 configuration -- which
    is pure repetition across the six to ten transforms a single run issues.

    ``order_ls`` is passed in rather than derived here so that a worker handling
    a subset of the spectrum uses the orders those frequencies have in the whole
    spectrum, not the orders they would get if the subset were the whole.
    """
    padding, n_fft, max_wl = geometry
    key = (float(Fs), F.tobytes(), Ncyc, order_ls.tobytes(), mult, n_fft)
    cached = _BANK.get(key)
    if cached is not None:
        return cached

    wavelets = []
    for fi in range(F.size):
        for oi in range(order_ls[fi]):
            if mult != 0:
                w = cxmorlet(F[fi], Ncyc * (oi + 1), Fs)
            else:
                w = cxmorlet(F[fi], Ncyc + oi, Fs)
            wavelets.append((fi, w))

    spectra = np.zeros((len(wavelets), n_fft), dtype=np.complex128)
    for k, (_, w) in enumerate(wavelets):
        spectra[k, :w.size] = w
    spectra = sfft.fft(spectra, axis=-1, workers=-1)
    # ``offset`` is where this wavelet's "same"-mode output starts inside its
    # full convolution, matching ``fftconvolve(..., mode="same")``.
    meta = [(fi, (w.size - 1) // 2, w.size) for fi, w in wavelets]

    bank = (order_ls, padding, n_fft, max_wl, spectra, meta)
    while (_BANK and sum(b[4].nbytes for b in _BANK.values())
           + spectra.nbytes > _BANK_BUDGET_BYTES):
        _BANK.pop(next(iter(_BANK)))
    _BANK[key] = bank
    return bank


def aslt(sig, Fs, F, Ncyc, ord_interval=None, mult=0):
    """Superlet transform.

    Parameters
    ----------
    sig : array
        1-D signal (single buffer) or 2-D (n_buffers, n_samples).
    Fs : float
        Sampling frequency (Hz).
    F : array
        Frequencies of interest (Hz).
    Ncyc : int
        Base number of wavelet cycles.
    ord_interval : (lo, hi) or None
        Superresolution order interval. None => order 1 everywhere (plain CWT).
    mult : int
        0 => additive superresolution, else multiplicative.

    Returns
    -------
    wt : (n_frequencies, n_samples) power spectrogram
    """
    F = np.asarray(F, dtype=np.float64).ravel()
    if F.size == 0:
        raise ValueError("frequencies not defined")
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    elif sig.shape[1] == 1 and sig.shape[0] > 1:
        sig = sig.T

    order_ls = _derive_orders(F.size, ord_interval)
    # Derived over the whole spectrum, then shared, so that splitting the work
    # cannot move a single bit of the result.
    geometry = _geometry(Fs, F, Ncyc, order_ls, mult)
    groups = _frequency_groups(F.size, sig.shape[1])
    if groups is None:
        return _aslt_core(sig, Fs, F, Ncyc, order_ls, mult, geometry)

    jobs = [(sig, Fs, F[idx], Ncyc, order_ls[idx], mult, geometry)
            for idx in groups]
    try:
        blocks = list(_pool().map(_aslt_group, jobs))
    except Exception:
        # A pool that cannot be created or dies mid-run must not fail the
        # analysis; the serial path is always available and gives the same
        # numbers.
        shutdown_workers()
        return _aslt_core(sig, Fs, F, Ncyc, order_ls, mult, geometry)
    wt = np.empty((F.size, sig.shape[1]))
    for idx, block in zip(groups, blocks):
        wt[idx] = block
    return wt


def _aslt_group(args):
    """Worker entry point: one frequency subset of one transform."""
    return _aslt_core(*args)


def _aslt_core(sig, Fs, F, Ncyc, order_ls, mult, geometry):
    """Serial transform of ``F`` with the orders those frequencies carry."""
    n_buffers, n_points = sig.shape
    _, padding, n_fft, max_wl, spectra, meta = _wavelet_bank(
        Fs, F, Ncyc, order_ls, mult, geometry)
    # Outputs per block that are free of circular wrap-around.  The shortest
    # wavelet yields more than this, but a common hop keeps consecutive blocks
    # from writing the same sample twice.
    hop = n_fft - max_wl + 1

    wt = np.zeros((F.size, n_points))
    for ib in range(n_buffers):
        buffer = np.zeros(n_points + 2 * padding)
        buffer[padding:padding + n_points] = sig[ib]
        temp = np.ones((F.size, n_points))
        for start in range(0, buffer.size, hop):
            segment = buffer[start:start + n_fft]
            if segment.size < n_fft:
                segment = np.concatenate(
                    (segment, np.zeros(n_fft - segment.size)))
            spectrum = sfft.fft(segment)
            for lo in range(0, len(meta), _CHUNK):
                conv = sfft.ifft(spectra[lo:lo + _CHUNK] * spectrum, axis=-1,
                                 overwrite_x=True, workers=1)
                for j, (fi, offset, wl) in enumerate(meta[lo:lo + _CHUNK]):
                    # Samples wl-1.. are the wrap-free part of the circular
                    # convolution; the first of them is "same" output index
                    # ``src`` of the whole buffer.
                    valid = conv[j, wl - 1:wl - 1 + hop]
                    src = start + wl - 1 - offset
                    a = max(padding, src)
                    b = min(padding + n_points, src + valid.size)
                    if b <= a:
                        continue
                    block = valid[a - src:b - src]
                    temp[fi, a - padding:b - padding] *= np.abs(block) * 2.0
        for fi in range(F.size):
            root = 1.0 / order_ls[fi]
            wt[fi] += (temp[fi] ** root) ** 2
    return wt / n_buffers
