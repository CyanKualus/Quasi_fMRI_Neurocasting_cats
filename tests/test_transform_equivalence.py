"""The fast filter and superlet paths must agree with what they replaced.

Both optimisations are pure implementation changes: FFT convolution replaces a
time-domain ``filtfilt`` and a per-call wavelet rebuild, and neither is allowed
to move a result. The superseded implementations are vendored here as
references so the equivalence is checked rather than assumed, and so a future
edit to the fast paths cannot quietly change a score.

Agreement is to floating-point rounding, not bit-for-bit: reassociating a
convolution into the frequency domain reorders the additions. The tolerances
below are tight enough that no score, gauge or provenance digest can move.
"""
import numpy as np
import pytest
from scipy.signal import fftconvolve, filtfilt

from klh import filters, superlet
from klh.superlet import cxmorlet, _fix


# ---------------------------------------------------------------------------
# References: the implementations in use before the FFT paths were introduced.
# ---------------------------------------------------------------------------
def reference_filtfilt(b, x):
    """The previous body of ``filters.apply_filtfilt``."""
    b = np.asarray(b, dtype=np.float64).ravel()
    return filtfilt(b, [1.0], np.asarray(x, dtype=np.float64), axis=0)


def reference_aslt(sig, Fs, F, Ncyc, ord_interval=None, mult=0):
    """The previous body of ``superlet.aslt``: direct per-wavelet fftconvolve."""
    F = np.asarray(F, dtype=np.float64).ravel()
    if F.size == 0:
        raise ValueError("frequencies not defined")
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    elif sig.shape[1] == 1 and sig.shape[0] > 1:
        sig = sig.T
    n_buffers, n_points = sig.shape

    if ord_interval is not None:
        order_ls = _fix(np.linspace(ord_interval[0], ord_interval[1], F.size))
    else:
        order_ls = np.ones(F.size, dtype=int)
    order_ls = np.maximum(order_ls, 1)
    max_ord = int(order_ls.max())

    wavelets = [[None] * max_ord for _ in range(F.size)]
    padding = 0
    for fi in range(F.size):
        for oi in range(order_ls[fi]):
            if mult != 0:
                w = cxmorlet(F[fi], Ncyc * (oi + 1), Fs)
            else:
                w = cxmorlet(F[fi], Ncyc + oi, Fs)
            wavelets[fi][oi] = w
            padding = max(padding, int(_fix(w.size / 2)))

    wt = np.zeros((F.size, n_points))
    for ib in range(n_buffers):
        buffer = np.zeros(n_points + 2 * padding)
        buffer[padding:padding + n_points] = sig[ib]
        for fi in range(F.size):
            temp = np.ones(n_points)
            for oi in range(order_ls[fi]):
                conv = fftconvolve(buffer, wavelets[fi][oi], mode="same")
                valid = np.abs(conv[padding:padding + n_points]) * 2.0
                temp = temp * valid
            root = 1.0 / order_ls[fi]
            wt[fi] += (temp ** root) ** 2
    return wt / n_buffers


def _max_rel(actual, expected):
    scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
    return float(np.max(np.abs(actual - expected) / scale))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def coefficients():
    return filters.load_prebuilt()


@pytest.mark.parametrize("n_channels", [1, 8])
def test_fft_filtfilt_matches_scipy_on_stored_coefficients(
        coefficients, n_channels):
    b1, b2, bpa = coefficients
    rng = np.random.default_rng(0)
    x = rng.standard_normal((30_000, n_channels))
    for b in (b1, b2, bpa):
        actual = filters.apply_filtfilt(b, x)
        expected = reference_filtfilt(b, x)
        assert actual.shape == expected.shape
        # Relative to the amplitude of the filtered signal, not to zero
        # crossings of it.
        assert np.max(np.abs(actual - expected)) < 1e-9 * np.std(expected)


def test_fft_filtfilt_matches_scipy_on_one_dimensional_input(coefficients):
    b1, _, _ = coefficients
    x = np.random.default_rng(1).standard_normal(30_000)
    actual = filters.apply_filtfilt(b1, x)
    expected = reference_filtfilt(b1, x)
    assert actual.shape == expected.shape == (30_000,)
    assert np.max(np.abs(actual - expected)) < 1e-9 * np.std(expected)


def test_broadband_chain_matches_reference(coefficients):
    b1, b2, _ = coefficients
    x = np.random.default_rng(2).standard_normal((40_000, 4))
    actual = filters.broadband(x, b1, b2)
    expected = reference_filtfilt(b1, reference_filtfilt(b2, x))
    assert np.max(np.abs(actual - expected)) < 1e-9 * np.std(expected)


def test_input_too_short_to_pad_still_raises(coefficients):
    """The FFT threshold is scipy's own minimum length, so an input scipy
    cannot filter must still reach scipy and produce scipy's error rather
    than being silently handled a second way."""
    b1, _, _ = coefficients
    x = np.random.default_rng(3).standard_normal((3 * b1.size, 2))
    with pytest.raises(ValueError):
        filters.apply_filtfilt(b1, x)
    with pytest.raises(ValueError):
        reference_filtfilt(b1, x)


def test_few_tap_filter_stays_on_the_scipy_path():
    b = np.array([0.25, 0.5, 0.25])
    x = np.random.default_rng(4).standard_normal((5_000, 3))
    assert np.array_equal(
        filters.apply_filtfilt(b, x), reference_filtfilt(b, x))


# ---------------------------------------------------------------------------
# Superlet
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_points", [
    2_000,      # shorter than one overlap-save block
    9_000,      # one padded marker epoch
    45_000,     # several blocks: exercises the hop bookkeeping
])
def test_aslt_matches_direct_convolution(n_points):
    fs = 1000.0
    F = np.arange(3, 30.0001, 0.5)
    sig = np.random.default_rng(5).standard_normal(n_points)
    actual = superlet.aslt(sig, fs, F, 5, (1, 30), 0)
    expected = reference_aslt(sig, fs, F, 5, (1, 30), 0)
    assert actual.shape == expected.shape == (F.size, n_points)
    assert _max_rel(actual, expected) < 1e-9


def test_aslt_matches_reference_for_multiple_buffers():
    fs = 1000.0
    F = np.arange(5, 20.0001, 1.0)
    sig = np.random.default_rng(6).standard_normal((3, 12_000))
    actual = superlet.aslt(sig, fs, F, 5, (1, 8), 0)
    expected = reference_aslt(sig, fs, F, 5, (1, 8), 0)
    assert _max_rel(actual, expected) < 1e-9


def test_aslt_matches_reference_without_superresolution():
    """``ord_interval=None`` is the plain-CWT path."""
    fs = 500.0
    F = np.arange(4, 25.0001, 1.0)
    sig = np.random.default_rng(7).standard_normal(20_000)
    actual = superlet.aslt(sig, fs, F, 5, None, 0)
    expected = reference_aslt(sig, fs, F, 5, None, 0)
    assert _max_rel(actual, expected) < 1e-9


def test_aslt_matches_reference_for_multiplicative_superresolution():
    fs = 1000.0
    F = np.arange(6, 18.0001, 1.0)
    sig = np.random.default_rng(8).standard_normal(15_000)
    actual = superlet.aslt(sig, fs, F, 3, (1, 5), 1)
    expected = reference_aslt(sig, fs, F, 3, (1, 5), 1)
    assert _max_rel(actual, expected) < 1e-9


def test_aslt_accepts_a_column_vector_like_the_reference():
    fs = 1000.0
    F = np.arange(8, 14.0001, 1.0)
    sig = np.random.default_rng(9).standard_normal((10_000, 1))
    actual = superlet.aslt(sig, fs, F, 5, (1, 4), 0)
    expected = reference_aslt(sig, fs, F, 5, (1, 4), 0)
    assert actual.shape == (F.size, 10_000)
    assert _max_rel(actual, expected) < 1e-9


def test_aslt_still_rejects_an_empty_frequency_list():
    with pytest.raises(ValueError, match="frequencies not defined"):
        superlet.aslt(np.zeros(100), 1000.0, [], 5, (1, 3), 0)


def test_wavelet_bank_is_reused_across_calls():
    """The bank is the expensive part; a second call must not rebuild it."""
    superlet._BANK.clear()
    fs = 1000.0
    F = np.arange(10, 14.0001, 1.0)
    sig = np.random.default_rng(10).standard_normal(5_000)
    superlet.aslt(sig, fs, F, 5, (1, 3), 0)
    assert len(superlet._BANK) == 1
    first = next(iter(superlet._BANK.values()))
    superlet.aslt(sig, fs, F, 5, (1, 3), 0)
    assert len(superlet._BANK) == 1
    assert next(iter(superlet._BANK.values())) is first


def test_wavelet_bank_distinguishes_configurations():
    superlet._BANK.clear()
    fs = 1000.0
    sig = np.random.default_rng(11).standard_normal(4_000)
    superlet.aslt(sig, fs, np.arange(10, 13.0001, 1.0), 5, (1, 3), 0)
    superlet.aslt(sig, fs, np.arange(10, 13.0001, 1.0), 7, (1, 3), 0)
    assert len(superlet._BANK) == 2


def test_wavelet_bank_is_bounded_by_weight(monkeypatch):
    """Cached spectra are large, so the cache evicts on bytes, not entries."""
    superlet._BANK.clear()
    fs = 1000.0
    sig = np.random.default_rng(12).standard_normal(4_000)
    # A budget smaller than a single bank forces eviction on every insert.
    monkeypatch.setattr(superlet, "_BANK_BUDGET_BYTES", 1)
    for ncyc in (5, 7, 9):
        superlet.aslt(sig, fs, np.arange(10, 13.0001, 1.0), ncyc, (1, 3), 0)
        assert len(superlet._BANK) == 1
