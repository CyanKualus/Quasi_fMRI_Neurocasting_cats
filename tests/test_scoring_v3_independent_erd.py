"""Regression tests for independent sustained quasi/imagery ERD scoring."""
from dataclasses import replace

import numpy as np
import pytest

from klh.pipeline import IMAG, QUASI, KLHPipeline
from klh.scoring import (
    DEFAULT_V3_THRESHOLDS,
    SCORING_V3,
    ScoringV3Thresholds,
    compute_scores,
    compute_scores_v3,
)


FREQS = np.arange(5.0, 17.01, 0.5)
TVEC = np.arange(-0.5, 2.51, 0.25)
PATTERN = np.array([1.0, 2.0, 4.0, 8.0])


def _constant_tf(overt_erd, quasi_erd, imagery_erd):
    shape = (FREQS.size, TVEC.size)
    return [
        np.full(shape, -float(overt_erd)),
        np.full(shape, -float(quasi_erd)),
        np.full(shape, -float(imagery_erd)),
    ]


@pytest.mark.parametrize(
    ("quasi", "imagery", "expected"),
    [
        (3.5, 2.4, [3, 2]),
        (1.7, 3.2, [1, 3]),
        (3.5, 0.4, [3, 0]),
        (-0.5, 2.5, [0, 2]),
        (2.5, 2.5, [2, 2]),
    ],
)
def test_v3_matches_the_proposal_examples(quasi, imagery, expected):
    result = compute_scores_v3(
        _constant_tf(10, quasi, imagery),
        FREQS, TVEC, PATTERN, PATTERN, 3,
    )

    assert result.scores[4:6].astype(int).tolist() == expected
    assert result.sustained_erd_db == pytest.approx({
        "quasi": quasi,
        "imagery": imagery,
    })


@pytest.mark.parametrize(
    ("magnitude", "points"),
    [(1.0, 0), (1.0001, 1), (2.0, 1), (2.0001, 2),
     (3.0, 2), (3.0001, 3)],
)
def test_v3_uses_the_declared_strict_db_boundaries(magnitude, points):
    result = compute_scores_v3(
        _constant_tf(10, magnitude, magnitude),
        FREQS, TVEC, PATTERN, PATTERN, 1,
    )
    assert result.scores[4:6].tolist() == [points, points]


def test_v3_condition_points_are_independent_of_the_other_condition():
    imagery_changes = [
        compute_scores_v3(
            _constant_tf(10, 2.5, imagery),
            FREQS, TVEC, PATTERN, PATTERN, 1,
        )
        for imagery in (-3.0, 0.0, 5.0)
    ]
    quasi_changes = [
        compute_scores_v3(
            _constant_tf(10, quasi, 2.5),
            FREQS, TVEC, PATTERN, PATTERN, 1,
        )
        for quasi in (-3.0, 0.0, 5.0)
    ]

    assert [result.scores[4] for result in imagery_changes] == [2, 2, 2]
    assert [result.scores[5] for result in quasi_changes] == [2, 2, 2]


def test_v3_regional_median_does_not_reward_one_extreme_patch():
    maps = _constant_tf(10, 0.5, 0.5)
    fsl = (FREQS >= 9) & (FREQS <= 13)
    tsl = (TVEC >= 0.5) & (TVEC <= 2.0)
    region = maps[QUASI][np.ix_(fsl, tsl)]
    region[0, 0] = -30.0
    maps[QUASI][np.ix_(fsl, tsl)] = region

    result = compute_scores_v3(
        maps, FREQS, TVEC, PATTERN, PATTERN, 1)

    assert result.sustained_erd_db["quasi"] == pytest.approx(0.5)
    assert result.scores[4] == 0


def test_v3_reports_temporal_coverage_without_gating_points():
    maps = _constant_tf(10, 0.0, 2.5)
    fsl = (FREQS >= 9) & (FREQS <= 13)
    active_times = np.flatnonzero((TVEC >= 0.5) & (TVEC <= 2.0))
    split = active_times.size // 2
    maps[QUASI][np.ix_(fsl, active_times[:split])] = -2.0
    maps[QUASI][np.ix_(fsl, active_times[split:])] = 2.0

    result = compute_scores_v3(
        maps, FREQS, TVEC, PATTERN, PATTERN, 1)

    assert result.sustained_erd_temporal_coverage["quasi"] == pytest.approx(
        split / active_times.size)
    # Coverage is a report-only safeguard in the provisional profile.
    assert result.scores[4] == 0


def test_v3_dispatch_profile_provenance_and_support_are_explicit():
    support = {
        "ci_db": {"quasi": [1.0, 3.0], "imagery": [2.0, 4.0]},
        "uncertainty": {"method": "test-bootstrap"},
        "run_consistency": {
            "quasi": {"retained_runs": 2, "positive_erd_runs": 2},
        },
    }
    result = compute_scores(
        _constant_tf(10, 2.5, 3.5),
        FREQS, TVEC, PATTERN, PATTERN, 1,
        version="v3",
        data_path_version="klh-data-v4:test",
        normalization_mode="condition-specific-marker-rest",
        sustained_erd_support=support,
    )

    assert result.scoring_version == SCORING_V3
    assert result.threshold_profile == DEFAULT_V3_THRESHOLDS.profile
    assert result.thresholds_calibrated is False
    assert result.data_path_version == "klh-data-v4:test"
    assert result.normalization_mode == "condition-specific-marker-rest"
    assert result.sustained_erd_frequency_window_hz == (9.0, 13.0)
    assert result.sustained_erd_time_window_s == (0.5, 2.0)
    assert result.sustained_erd_ci_db == support["ci_db"]
    assert result.sustained_erd_uncertainty == support["uncertainty"]
    assert result.sustained_erd_run_consistency == support["run_consistency"]


def test_v3_threshold_profile_can_be_calibrated_without_changing_v2():
    thresholds = replace(
        DEFAULT_V3_THRESHOLDS,
        profile="cohort-2027-independent-erd",
        calibrated=True,
        sustained_erd_db=(4.0, 3.0, 2.0),
    )
    result = compute_scores_v3(
        _constant_tf(10, 2.5, 3.5),
        FREQS, TVEC, PATTERN, PATTERN, 1,
        thresholds=thresholds,
    )

    assert result.scores[4:6].tolist() == [1, 2]
    assert result.threshold_profile == "cohort-2027-independent-erd"
    assert result.thresholds_calibrated is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sustained_erd_db", (1.0, 2.0, 3.0)),
        ("frequency_window_hz", (13.0, 9.0)),
        ("time_window_s", (0.5, 0.5)),
    ],
)
def test_invalid_v3_profiles_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(DEFAULT_V3_THRESHOLDS, **{field: value})


def test_v3_profile_is_publicly_marked_provisional():
    assert isinstance(DEFAULT_V3_THRESHOLDS, ScoringV3Thresholds)
    assert DEFAULT_V3_THRESHOLDS.calibrated is False
    assert "provisional" in DEFAULT_V3_THRESHOLDS.profile


def test_pipeline_support_reports_trial_ci_and_retained_run_consistency():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.freqNeeded = np.arange(8.0, 15.0)
    pipe.tvec_active = np.arange(0.0, 2.51, 0.25)
    base = np.ones(pipe.freqNeeded.size)

    def power_for_erd(values):
        ratios = 10.0 ** (-np.asarray(values, dtype=float) / 10.0)
        return np.broadcast_to(
            ratios[None, :, None],
            (pipe.tvec_active.size, ratios.size, pipe.freqNeeded.size),
        ).copy()

    trial_power = [
        power_for_erd([1, 1, 1, 1]),
        power_for_erd([1, 2, 3, 4]),
        power_for_erd([-1, -1, 2, 2]),
    ]
    pipe._file_epoch_slices = [
        {"file": "q1", "condition": "quasi", "active": [0, 2]},
        {"file": "q2", "condition": "quasi", "active": [2, 4]},
        {"file": "i1", "condition": "imag", "active": [0, 2]},
        {"file": "i2", "condition": "imag", "active": [2, 4]},
    ]

    support = pipe._sustained_erd_support(
        trial_power, [base, base, base],
        selected_comp=2, data_path_version="test-path",
        bootstrap_resamples=100,
    )

    assert np.all(np.isfinite(support["ci_db"]["quasi"]))
    assert support["uncertainty"]["finite_trials"] == {
        "quasi": 4, "imagery": 4}
    assert support["run_consistency"]["quasi"]["positive_erd_runs"] == 2
    assert support["run_consistency"]["quasi"]["retained_runs"] == 2
    assert support["run_consistency"]["imagery"]["positive_erd_runs"] == 1
    assert support["run_consistency"]["imagery"]["retained_runs"] == 2
