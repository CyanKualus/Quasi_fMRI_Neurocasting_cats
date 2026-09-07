"""Small numeric helpers matching MATLAB semantics exactly."""
from __future__ import annotations

import warnings

import numpy as np


def matlab_prctile(x, p):
    """MATLAB ``prctile`` (default method).

    Sorted samples are assigned plotting positions 100*(i-0.5)/n and the
    requested percentile is obtained by linear interpolation, clamping to the
    extremes. Operates over the flattened array.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan
    xs = np.sort(x)
    pos = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    return np.interp(p, pos, xs, left=xs[0], right=xs[-1])


def mad1(x):
    """MATLAB ``mad(x, 1)`` — median absolute deviation about the median."""
    x = np.asarray(x, dtype=np.float64).ravel()
    return np.median(np.abs(x - np.median(x)))


def nearest_index(vec, value):
    """0-based index of the element of ``vec`` closest to ``value`` (dsearchn)."""
    vec = np.asarray(vec, dtype=np.float64).ravel()
    return int(np.abs(vec - value).argmin())


def window_support(vec, a, b, *, label=None):
    """Describe whether an inclusive window is supported by ``vec``.

    The returned dictionary is intentionally JSON-serialisable so callers can
    attach it to analysis provenance as well as show it interactively.
    """
    vec = np.asarray(vec, dtype=np.float64).ravel()
    if vec.size == 0:
        raise ValueError("cannot select a window from an empty vector")
    if b < a:
        raise ValueError(f"window end ({b}) precedes its start ({a})")
    lo, hi = float(np.nanmin(vec)), float(np.nanmax(vec))
    return {
        "label": label,
        "requested": [float(a), float(b)],
        "available": [lo, hi],
        "clamped_start": bool(a < lo),
        "clamped_end": bool(b > hi),
        "clamped": bool(a < lo or b > hi),
    }


def rng_inclusive(vec, a, b, *, strict=False, warn=True, label=None,
                  diagnostics=None):
    """Inclusive nearest-index slice with visible out-of-range handling.

    This retains MATLAB's ``dsearchn(vec,a):dsearchn(vec,b)`` selection when the
    request is supported.  Previously an unsupported endpoint silently snapped
    to the edge of the available epoch.  It now warns by default, can raise with
    ``strict=True``, and can append a serialisable record to ``diagnostics``.
    """
    support = window_support(vec, a, b, label=label)
    if diagnostics is not None:
        diagnostics.append(support.copy())
    if support["clamped"]:
        name = f" for {label}" if label else ""
        message = (
            f"requested window{name} [{a:g}, {b:g}] lies outside available "
            f"[{support['available'][0]:g}, {support['available'][1]:g}] and "
            "would be clamped")
        if strict:
            raise ValueError(message)
        if warn:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    ia = nearest_index(vec, a)
    ib = nearest_index(vec, b)
    return slice(ia, ib + 1)
