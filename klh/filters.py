"""Band-pass FIR filters.

The MATLAB app pre-computes three FIR filters and stores them in
``helper_files.mat``:

* ``b2`` : high-pass, f_low = 3 Hz
* ``b1`` : low-pass,  f_high = 43 Hz   (together b1(b2(x)) => 3-43 Hz broadband)
* ``bpa``: band-pass, 9.0-14.5 Hz alpha

We reuse those exact coefficients by default (loaded from assets/helper.npz),
so filtering matches MATLAB bit-for-bit up to filtfilt edge handling. A pure
Python re-implementation of ``get_bandpassFIR`` (Parks-McClellan) is also
provided for when the bands need to change.

Those coefficient sets run to ~2500 taps, which makes scipy's time-domain
``filtfilt`` cost O(samples * taps) -- around 90 s for a ten-minute 64-channel
recording. Inputs long enough to pad therefore take an equivalent FFT
convolution path; see :func:`_filtfilt_fir_fft`.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.signal import filtfilt, oaconvolve, remez
from scipy.signal._arraytools import odd_ext

from .assets import load_helper

FS_DEFAULT = 1000.0

# Below this tap count the time-domain filtfilt is already cheap and the FFT
# machinery would only add overhead, so scipy keeps handling those calls.
_FFT_MIN_TAPS = 32


def load_prebuilt():
    """Return (b1, b2, bpa) exactly as stored in helper_files.mat."""
    h = load_helper()
    return h["b1"], h["b2"], h["bpa"]


def _fir_lfilter_fft(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``lfilter(b, [1], x, zi=lfilter_zi(b, [1]) * x[0])`` along axis 0.

    For an FIR filter the steady-state initial condition is exactly the
    assumption that the signal held its first value for all time before sample
    zero, so the result is a plain convolution of the value-extended signal --
    which FFT convolution computes in O(n log n) instead of O(n * taps).
    """
    m = b.size - 1
    if m == 0:
        return b[0] * x
    head = np.repeat(x[:1], m, axis=0)
    ext = np.concatenate((head, x), axis=0)
    kernel = b.reshape((-1,) + (1,) * (x.ndim - 1))
    full = oaconvolve(ext, kernel, mode="full", axes=0)
    return full[m:m + x.shape[0]]


def _filtfilt_fir_fft(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``scipy.signal.filtfilt(b, [1.0], x, axis=0)`` via FFT convolution.

    Reproduces scipy's default ``padtype="odd"`` extension of ``3 * ntaps``
    samples and its steady-state initial conditions, replacing only the two
    ``lfilter`` passes.  Agreement with scipy is to floating-point rounding.
    """
    padlen = 3 * b.size
    ext = odd_ext(x, padlen, axis=0)
    y = _fir_lfilter_fft(b, ext)
    y = _fir_lfilter_fft(b, np.ascontiguousarray(y[::-1]))[::-1]
    return y[padlen:-padlen]


def apply_filtfilt(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase FIR filtering along axis 0 (time), per channel.

    Mirrors MATLAB ``filtfilt(b, 1, x)`` which operates column-wise.  The
    stored coefficient sets run to ~2500 taps, for which scipy's time-domain
    ``filtfilt`` costs O(n * taps); long inputs therefore take the equivalent
    FFT-convolution path in :func:`_filtfilt_fir_fft`.
    """
    b = np.asarray(b, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)
    if b.size >= _FFT_MIN_TAPS and x.shape[0] > 3 * b.size:
        return _filtfilt_fir_fft(b, x)
    return filtfilt(b, [1.0], x, axis=0)


def broadband(x: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Reproduce ``filtfilt(b1, 1, filtfilt(b2, 1, x))`` (3-43 Hz)."""
    return apply_filtfilt(b1, apply_filtfilt(b2, x))


def alpha(x: np.ndarray, bpa: np.ndarray) -> np.ndarray:
    """Reproduce ``filtfilt(bpa, 1, x)`` (9-14.5 Hz)."""
    return apply_filtfilt(bpa, x)


# --------------------------------------------------------------------------
# Pure-Python port of get_bandpassFIR (only needed if bands change).
# --------------------------------------------------------------------------
def _firpmord(freqs, amps, devs, fs):
    """Kaiser/Herrmann order estimate approximating MATLAB ``firpmord``.

    Returns (N, bands, amps, weights) usable with scipy ``remez``. This is an
    approximation of MATLAB's estimator; the resulting order can differ by a
    tap or two, which is why the pre-built coefficients are preferred.
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    devs = np.asarray(devs, dtype=float)

    # Per-band transition widths.
    df = np.diff(freqs)[::2] / fs
    # Herrmann et al. order estimate for the tightest transition.
    dev_pass = devs[amps == 1].min() if np.any(amps == 1) else devs.min()
    dev_stop = devs[amps == 0].min() if np.any(amps == 0) else devs.min()
    d = ((0.005309 * np.log10(dev_pass) ** 2 + 0.07114 * np.log10(dev_pass)
          - 0.4761) * np.log10(dev_stop)
         - (0.00266 * np.log10(dev_pass) ** 2 + 0.5941 * np.log10(dev_pass)
            + 0.4278))
    f_fit = 11.01217 + 0.51244 * (np.log10(dev_pass) - np.log10(dev_stop))
    dfmin = df.min()
    N = int(np.ceil((d - f_fit * dfmin ** 2) / dfmin)) + 1
    weights = devs.min() / devs
    return N, freqs / (fs / 2), amps, weights


def get_bandpass_fir(f_low, f_high, fs=FS_DEFAULT, transition_band=1.0):
    """Port of ``get_bandpassFIR`` returning FIR coefficients ``b``.

    Only the band-pass branch (both f_low and f_high given) is exercised by the
    app for the alpha filter; high-pass-only and low-pass-only are supported
    for the broadband pair.
    """
    Dstop1, Dpass, Dstop2 = 0.01, 0.005, 0.005

    if f_low is not None and f_low > 2:
        Fstop1, Fpass1 = f_low - transition_band, f_low
    elif f_low is not None:
        Fstop1, Fpass1 = f_low * 0.8, f_low * 1.2
    if f_high is not None:
        Fpass2, Fstop2 = f_high, f_high + transition_band

    if f_high is None:  # high-pass
        N, bands, amps, w = _firpmord([Fstop1, Fpass1], [0, 1], [Dstop1, Dpass / 2], fs)
        b = remez(N + 1, [0, Fstop1, Fpass1, fs / 2], [0, 1], weight=w, fs=fs)
    elif f_low is None:  # low-pass
        N, bands, amps, w = _firpmord([Fpass2, Fstop2], [1, 0], [Dpass / 2, Dstop2], fs)
        b = remez(N + 1, [0, Fpass2, Fstop2, fs / 2], [1, 0], weight=w, fs=fs)
    else:  # band-pass
        N, bands, amps, w = _firpmord(
            [Fstop1, Fpass1, Fpass2, Fstop2], [0, 1, 0], [Dstop1, Dpass, Dstop2], fs)
        N = N + (N % 2)  # keep even order like the MATLAB `rem` adjustment
        b = remez(N + 1, [0, Fstop1, Fpass1, Fpass2, Fstop2, fs / 2],
                  [0, 1, 0], weight=w, fs=fs)
    return b
