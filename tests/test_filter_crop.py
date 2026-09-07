"""Filtering only the epoched span must not change a single epoch sample.

The saving is real work removed, not an approximation: the crop keeps enough
real signal on each side that every epoched sample has the same filter support
it would have had in the whole recording. These tests pin both halves of that
claim -- the sample values, and the set of trials the boundary check keeps.
"""
import numpy as np
import pytest

from klh import filters, io_h5
from klh.pipeline import KLHPipeline, _crop_to_epoch_span


@pytest.fixture(scope="module")
def coefficients():
    return filters.load_prebuilt()


@pytest.fixture(scope="module")
def margin():
    return KLHPipeline()._filter_edge_margin()


def test_margin_covers_the_longest_filter_chain(coefficients, margin):
    b1, b2, bpa = coefficients
    # Broadband is b2 then b1, so their supports add; alpha is a single pass.
    assert margin >= (b1.size - 1) + (b2.size - 1)
    assert margin >= bpa.size - 1


def _epoch_span_window(n_onsets, spacing, epoind, start=50_000):
    onsets = start + spacing * np.arange(n_onsets)
    return onsets, epoind


def test_cropped_broadband_matches_whole_recording(coefficients, margin):
    b1, b2, _ = coefficients
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((400_000, 4))
    epoind = np.arange(-2500, 5500)
    onsets, _ = _epoch_span_window(6, 20_000, epoind, start=120_000)

    offset, cropped = _crop_to_epoch_span(
        signal, ((onsets, epoind),), margin=margin)
    assert offset > 0 and cropped.shape[0] < signal.shape[0]

    whole = filters.broadband(signal, b1, b2)
    part = filters.broadband(cropped, b1, b2)

    expected = io_h5.epoch(whole, onsets, epoind)
    actual = io_h5.epoch(part, onsets - offset, epoind)
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < 1e-9 * np.std(expected)


def test_cropped_alpha_matches_whole_recording(coefficients, margin):
    _, _, bpa = coefficients
    rng = np.random.default_rng(1)
    signal = rng.standard_normal((300_000, 3))
    epoind = np.arange(0, 4000)
    onsets = np.array([80_000, 150_000, 210_000])

    offset, cropped = _crop_to_epoch_span(
        signal, ((onsets, epoind),), margin=margin)
    expected = io_h5.epoch(filters.alpha(signal, bpa), onsets, epoind)
    actual = io_h5.epoch(filters.alpha(cropped, bpa), onsets - offset, epoind)
    assert np.max(np.abs(actual - expected)) < 1e-9 * np.std(expected)


def test_crop_covers_two_window_kinds(margin):
    """Active and rest windows are epoched from one filtered array."""
    signal = np.zeros((200_000, 2))
    active = (np.array([100_000]), np.arange(0, 3000))
    rest = (np.array([20_000]), np.arange(0, 5000))
    offset, cropped = _crop_to_epoch_span(signal, (active, rest), margin=margin)
    # Must span the earliest rest sample to the latest active sample.
    assert offset <= 20_000 - margin
    assert offset + cropped.shape[0] >= 103_000 + margin


def test_trial_rejection_is_unchanged_by_cropping(margin):
    """Onsets that overran the recording must still be dropped."""
    rng = np.random.default_rng(2)
    signal = rng.standard_normal((120_000, 2))
    epoind = np.arange(-1000, 3000)
    # First runs off the start, last runs off the end; two fit.
    onsets = np.array([500, 40_000, 80_000, 119_000])

    _, valid_whole = io_h5.epoch(signal, onsets, epoind, return_valid=True)
    offset, cropped = _crop_to_epoch_span(
        signal, ((onsets, epoind),), margin=margin)
    _, valid_crop = io_h5.epoch(
        cropped, onsets - offset, epoind, return_valid=True)

    assert list(valid_whole) == [False, True, True, False]
    assert np.array_equal(valid_whole, valid_crop)


def test_crop_is_skipped_when_nothing_can_be_removed(margin):
    signal = np.zeros((5_000, 2))
    onsets = np.array([100])
    epoind = np.arange(0, 4000)
    offset, cropped = _crop_to_epoch_span(
        signal, ((onsets, epoind),), margin=margin)
    assert offset == 0
    assert cropped is signal


def test_crop_is_skipped_when_no_window_fits(margin):
    """With every onset out of bounds there is no span to crop against."""
    signal = np.zeros((10_000, 2))
    onsets = np.array([9_500])
    epoind = np.arange(0, 4000)
    offset, cropped = _crop_to_epoch_span(
        signal, ((onsets, epoind),), margin=margin)
    assert offset == 0
    assert cropped is signal


def test_crop_ignores_out_of_bounds_onsets_when_sizing_the_span(margin):
    """An unusable onset must not drag the crop across the whole recording."""
    signal = np.zeros((400_000, 2))
    epoind = np.arange(0, 2000)
    # 399_000 cannot be epoched (399_000 + 2000 > 400_000); it must not
    # stretch the kept span to the end of the recording.
    onsets = np.array([50_000, 399_000])
    offset, cropped = _crop_to_epoch_span(
        signal, ((onsets, epoind),), margin=margin)
    assert offset + cropped.shape[0] < 399_000
