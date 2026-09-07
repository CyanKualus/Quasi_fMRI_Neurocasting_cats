"""Regression tests for the continuous rolling-median peak ERD score."""
from dataclasses import replace

import numpy as np
import pytest

from klh.scoring import (
    DEFAULT_V4_THRESHOLDS,
    SCORING_V3,
    SCORING_V4,
    ScoringV4Thresholds,
    compute_scores,
    compute_scores_v3,
    compute_scores_v4,
)


FREQS = np.arange(5.0, 16.01, 0.5)
TVEC = np.arange(0.0, 2.01, 0.1)
PATTERN = np.array([1.0, 2.0, 4.0, 8.0])


def _maps(depth=1.0):
    shape = (FREQS.size, TVEC.size)
    return [np.full(shape, -float(depth)) for _ in range(3)]


def _score(maps, *, version=SCORING_V4):
    return compute_scores(
        maps,
        FREQS,
        TVEC,
        PATTERN,
        PATTERN,
        1,
        version=version,
        normalization_mode="condition-specific-marker-rest",
    )


def test_v4_rejects_a_deep_but_brief_patch_that_v3_rewards():
    maps = _maps(1.0)
    brief = (TVEC >= 0.8) & (TVEC < 1.0)  # 200 ms
    maps[0][:, brief] = -20.0

    v3 = _score(maps, version=SCORING_V3)
    v4 = _score(maps)

    assert v3.peakERD[0] == pytest.approx(20.0)
    assert v3.scores[2] == 3
    assert v4.peakERD[0] == pytest.approx(1.0)
    assert v4.scores[2] == 0


def test_v4_rewards_a_continuous_half_second_and_reports_its_bounds():
    maps = _maps(1.0)
    sustained = (TVEC >= 0.8) & (TVEC < 1.4)
    maps[0][:, sustained] = -8.5

    result = _score(maps)

    assert result.peakERD[0] == pytest.approx(8.5)
    assert result.scores[2] == 3
    assert result.peak_erd_strongest_window_s["overt"] == pytest.approx(
        (0.8, 1.3))


def test_v4_frequency_median_rejects_one_extreme_frequency_bin():
    maps = _maps(1.0)
    maps[0][0, :] = -30.0

    result = _score(maps)

    assert result.peakERD[0] == pytest.approx(1.0)


def test_v4_result_records_the_complete_peak_definition():
    result = _score(_maps(4.0))

    assert result.scoring_version == SCORING_V4
    assert result.peak_erd_method == "strongest-continuous-rolling-median"
    assert result.peak_erd_frequency_window_hz == (5.0, 16.0)
    assert result.peak_erd_rolling_window_s == pytest.approx(0.5)
    assert set(result.peak_erd_strongest_window_s) == {
        "overt", "quasi", "imagery"}


@pytest.mark.parametrize(
    ("magnitude", "points"),
    [
        (1.0, 0),
        (1.0001, 1),
        (3.0, 1),
        (3.0001, 2),
        (5.0, 2),
        (5.0001, 3),
        (5.19, 3),
    ],
)
def test_v4_uses_rolling_peak_cutoffs_on_the_new_scale(magnitude, points):
    result = _score(_maps(magnitude))

    assert result.peakERD[0] == pytest.approx(magnitude)
    assert result.scores[2] == points


@pytest.mark.parametrize(
    ("magnitude", "points"),
    [
        (2.0, 0),
        (2.0001, 1),
        (3.0, 1),
        (3.0001, 2),
        (4.0, 2),
        (4.0001, 3),
    ],
)
def test_v4_uses_revised_sustained_erd_cutoffs(magnitude, points):
    result = _score(_maps(magnitude))

    assert result.sustained_erd_db == pytest.approx({
        "quasi": magnitude,
        "imagery": magnitude,
    })
    assert result.scores[4] == points
    assert result.scores[5] == points


def test_v3_remains_the_historical_third_percentile_definition():
    result = compute_scores_v3(
        _maps(4.0), FREQS, TVEC, PATTERN, PATTERN, 1,
        normalization_mode="condition-specific-marker-rest")

    assert result.scoring_version == SCORING_V3
    assert result.peak_erd_method == "negative-third-percentile"
    assert result.peak_erd_rolling_window_s is None


def test_v4_alias_dispatches_to_the_new_formula():
    explicit = compute_scores_v4(
        _maps(4.0), FREQS, TVEC, PATTERN, PATTERN, 1,
        normalization_mode="condition-specific-marker-rest")
    alias = _score(_maps(4.0), version="v4")

    np.testing.assert_allclose(alias.peakERD, explicit.peakERD)
    assert alias.scoring_version == explicit.scoring_version == SCORING_V4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("peak_erd_frequency_window_hz", (16.0, 5.0)),
        ("peak_erd_rolling_window_s", 0.0),
        ("peak_erd_rolling_window_s", np.nan),
    ],
)
def test_invalid_v4_peak_profiles_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(DEFAULT_V4_THRESHOLDS, **{field: value})


def test_v4_profile_is_publicly_provisional():
    assert isinstance(DEFAULT_V4_THRESHOLDS, ScoringV4Thresholds)
    assert DEFAULT_V4_THRESHOLDS.calibrated is False
    assert "provisional" in DEFAULT_V4_THRESHOLDS.profile
    assert DEFAULT_V4_THRESHOLDS.profile.endswith("-v3")
    assert DEFAULT_V4_THRESHOLDS.overt_erd_db == (5.0, 3.0, 1.0)
    assert DEFAULT_V4_THRESHOLDS.sustained_erd_db == (4.0, 3.0, 2.0)
