"""Splitting the spectrum across processes must not change the spectrum.

The frequency axis is separable, so the split is a scheduling decision -- but
only if each worker gives its frequencies the superresolution order they carry
in the *whole* spectrum. Deriving orders from a subset would silently change
every result, so that is what these tests are mostly guarding.
"""
import numpy as np
import pytest

from klh import superlet


@pytest.fixture(autouse=True)
def restore_worker_setting():
    yield
    superlet.configure_workers(0)
    superlet.shutdown_workers()


FS = 1000.0
F = np.arange(3, 30.0001, 0.5)


def test_orders_come_from_the_whole_spectrum_not_the_subset():
    """The property the parallel split depends on."""
    orders = superlet._derive_orders(F.size, (1, 30))
    interleaved = orders[np.arange(0, F.size, 4)]
    # A worker re-deriving orders from its own 14 frequencies would get
    # 1..30 spread over 14 points; it must instead keep the parent values.
    naive = superlet._derive_orders(interleaved.size, (1, 30))
    assert not np.array_equal(interleaved, naive)


def test_geometry_comes_from_the_whole_spectrum_not_the_subset():
    """Block geometry must be shared, or the split would change rounding."""
    orders = superlet._derive_orders(F.size, (1, 30))
    whole = superlet._geometry(FS, F, 5, orders, 0)
    idx = np.arange(1, F.size, 4)          # a subset with no 3 Hz wavelet
    subset = superlet._geometry(FS, F[idx], 5, orders[idx], 0)
    assert whole != subset
    # The longest wavelet is the lowest frequency at order 1, so a subset
    # missing it would choose a smaller padding and block length.
    assert whole[0] > subset[0]


def test_parallel_matches_serial():
    sig = np.random.default_rng(0).standard_normal(60_000)
    superlet.configure_workers(0)
    serial = superlet.aslt(sig, FS, F, 5, (1, 30), 0)
    superlet.configure_workers(3)
    parallel = superlet.aslt(sig, FS, F, 5, (1, 30), 0)
    assert parallel.shape == serial.shape
    # Same operations in the same order, merely in another process.
    assert np.array_equal(parallel, serial)


def test_parallel_matches_serial_for_multiple_buffers():
    sig = np.random.default_rng(1).standard_normal((2, 30_000))
    superlet.configure_workers(0)
    serial = superlet.aslt(sig, FS, F, 5, (1, 12), 0)
    superlet.configure_workers(2)
    assert np.array_equal(
        superlet.aslt(sig, FS, F, 5, (1, 12), 0), serial)


def test_short_signals_stay_serial():
    """Below the work threshold the pool must not even be built."""
    superlet.configure_workers(4)
    superlet.shutdown_workers()
    sig = np.random.default_rng(2).standard_normal(2_000)
    superlet.aslt(sig, FS, F, 5, (1, 6), 0)
    assert superlet._POOL is None


def test_too_few_frequencies_stay_serial():
    superlet.configure_workers(4)
    superlet.shutdown_workers()
    sig = np.random.default_rng(3).standard_normal(40_000)
    superlet.aslt(sig, FS, np.arange(9, 13.0001, 1.0), 5, (1, 4), 0)
    assert superlet._POOL is None


def test_zero_and_one_worker_mean_serial():
    for workers in (0, 1):
        assert superlet.configure_workers(workers) == workers
        assert superlet._frequency_groups(F.size, 100_000) is None


def test_groups_cover_every_frequency_exactly_once():
    superlet.configure_workers(4)
    groups = superlet._frequency_groups(F.size, 100_000)
    covered = np.concatenate(groups)
    assert np.array_equal(np.sort(covered), np.arange(F.size))


def test_groups_are_balanced_by_wavelet_count():
    """Interleaving, not slicing: order rises with frequency."""
    superlet.configure_workers(4)
    orders = superlet._derive_orders(F.size, (1, 30))
    loads = [int(orders[idx].sum())
             for idx in superlet._frequency_groups(F.size, 100_000)]
    # Contiguous slicing would put ~4x the work in the last group; interleaving
    # keeps every worker within a small fraction of the heaviest.
    assert max(loads) - min(loads) <= 0.15 * max(loads)
    contiguous = [int(block.sum()) for block in np.array_split(orders, 4)]
    assert max(contiguous) > 3 * min(contiguous)


def test_negative_worker_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        superlet.configure_workers(-1)


def test_a_broken_pool_falls_back_to_serial(monkeypatch):
    """A pool that cannot start must not fail the analysis."""
    superlet.configure_workers(3)
    superlet.shutdown_workers()

    def refuse():
        raise OSError("no processes available")

    monkeypatch.setattr(superlet, "_pool", refuse)
    sig = np.random.default_rng(4).standard_normal(40_000)
    result = superlet.aslt(sig, FS, F, 5, (1, 8), 0)

    superlet.configure_workers(0)
    assert np.array_equal(result, superlet.aslt(sig, FS, F, 5, (1, 8), 0))


def test_shutdown_is_safe_when_no_pool_exists():
    superlet.shutdown_workers()
    superlet.shutdown_workers()
    assert superlet._POOL is None
