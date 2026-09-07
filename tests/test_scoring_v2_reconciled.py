"""Regression and edge-case tests for the reconciled scoring implementation."""
from dataclasses import replace

import numpy as np
import pytest

from klh.scoring import (
    DEFAULT_V2_THRESHOLDS,
    SCORING_LEGACY,
    SCORING_V2,
    ScoringV2Thresholds,
    compute_scores,
    compute_scores_legacy,
    compute_scores_v2,
    pattern_similarity,
)


FREQS = np.arange(3.0, 18.0)
TVEC = np.linspace(-0.5, 3.0, 36)
PATTERN = np.array([1.0, 2.0, 4.0, 8.0])
IDEAL = PATTERN.copy()


def _constant_tf(overt_erd, quasi_erd, imagery_erd):
    """Build maps whose peak-ERD diagnostic equals the three given values."""
    shape = (FREQS.size, TVEC.size)
    return [
        np.full(shape, -float(overt_erd)),
        np.full(shape, -float(quasi_erd)),
        np.full(shape, -float(imagery_erd)),
    ]


def _legacy_and_v2(tf_norm, *, n_erd=3):
    legacy = compute_scores_legacy(
        tf_norm, FREQS, TVEC, PATTERN, IDEAL, n_erd)
    v2 = compute_scores_v2(
        tf_norm, FREQS, TVEC, PATTERN, IDEAL, n_erd)
    return legacy, v2


def test_legacy_snapshot_and_default_dispatch_are_unchanged():
    tf_norm = _constant_tf(10.0, 4.0, 6.0)
    explicit = compute_scores_legacy(
        tf_norm, FREQS, TVEC, PATTERN, IDEAL, 3,
        data_path_version="trigger-native-v1",
        normalization_mode="trigger-prestimulus",
    )
    default = compute_scores(
        tf_norm, FREQS, TVEC, PATTERN, IDEAL, 3,
        data_path_version="trigger-native-v1",
        normalization_mode="trigger-prestimulus",
    )

    # Strict threshold boundaries are part of the historical behavior: 6/10
    # is exactly 60%, so criterion 4 receives one rather than two points.
    assert explicit.scores.tolist() == [2, 1, 2, 1, 3, 3]
    np.testing.assert_array_equal(default.scores, explicit.scores)
    np.testing.assert_allclose(default.peakERD, explicit.peakERD)
    assert default.scoring_version == explicit.scoring_version == SCORING_LEGACY
    assert default.threshold_profile == "matlab-original"
    assert default.data_path_version == "trigger-native-v1"
    assert default.normalization_mode == "trigger-prestimulus"


def test_version_aliases_are_canonical_and_unknown_version_is_rejected():
    tf_norm = _constant_tf(10, 4, 6)
    result = compute_scores(
        tf_norm, FREQS, TVEC, PATTERN, IDEAL, 1, version="v2")
    assert result.scoring_version == SCORING_V2
    assert result.version == SCORING_V2

    with pytest.raises(ValueError, match="unknown scoring version"):
        compute_scores(
            tf_norm, FREQS, TVEC, PATTERN, IDEAL, 1, version="future-v9")
    with pytest.raises(ValueError, match="cannot be applied"):
        compute_scores(
            tf_norm, FREQS, TVEC, PATTERN, IDEAL, 1,
            version=SCORING_LEGACY,
            thresholds=DEFAULT_V2_THRESHOLDS,
        )


def test_v2_gates_only_criterion_4_when_overt_erd_is_nonpositive():
    # Overt and quasi show ERS (negative ERD magnitudes), while imagery shows a
    # strong ERD.  Criteria 5/6 should reward that scientifically useful split.
    tf_norm = _constant_tf(-1.0, -2.0, 6.0)
    legacy, v2 = _legacy_and_v2(tf_norm, n_erd=0)

    assert legacy.scores.tolist() == [0, 1, 0, 3, 3, 0]
    assert v2.scores.tolist() == [0, 1, 0, 0, 3, 3]
    assert v2.peak_iq_diff_db == pytest.approx(8.0)
    assert v2.iq_diff_db == pytest.approx(8.0)
    assert np.isnan(v2.rel_nonreal_pct)
    assert np.isnan(v2.rel_iq_pct)


@pytest.mark.parametrize(
    ("quasi_erd", "imagery_erd", "expected"),
    [
        (5.0, 5.5, 3),   # difference +0.5 > 0
        (5.0, 4.5, 2),   # difference -0.5 > -1
        (5.0, 3.5, 1),   # difference -1.5 > -2
        (6.0, 3.5, 0),   # difference -2.5
    ],
)
def test_v2_criterion_5_uses_graded_peak_db_differences(
    quasi_erd, imagery_erd, expected,
):
    result = compute_scores_v2(
        _constant_tf(10, quasi_erd, imagery_erd),
        FREQS, TVEC, PATTERN, IDEAL, 1,
    )
    assert result.scores[4] == expected
    assert result.peak_iq_diff_db == pytest.approx(imagery_erd - quasi_erd)


def test_v2_imagery_vs_quasi_remains_favorable_when_quasi_is_ers_or_zero():
    quasi_ers = compute_scores_v2(
        _constant_tf(8, -1, 4), FREQS, TVEC, PATTERN, IDEAL, 1)
    quasi_zero = compute_scores_v2(
        _constant_tf(8, 0, 4), FREQS, TVEC, PATTERN, IDEAL, 1)

    assert quasi_ers.scores[4:6].tolist() == [3, 3]
    assert quasi_zero.scores[4:6].tolist() == [3, 3]
    assert quasi_ers.iq_diff_db == pytest.approx(5.0)
    assert quasi_zero.iq_diff_db == pytest.approx(4.0)
    assert np.isnan(quasi_ers.rel_iq_pct)
    assert np.isnan(quasi_zero.rel_iq_pct)
    # V2 never computes the old ratio between two dB values.
    assert np.isnan(quasi_ers.iq_diff_mean_pct)
    assert np.isnan(quasi_zero.iq_diff_mean_pct)


def test_custom_threshold_profile_is_applied_and_recorded():
    thresholds = replace(
        DEFAULT_V2_THRESHOLDS,
        profile="cohort-2026-calibrated",
        calibrated=True,
        imagery_minus_quasi_peak_db=(10.0, 5.0, 1.0),
        imagery_minus_quasi_mean_db=(10.0, 5.0, 1.0),
    )
    result = compute_scores_v2(
        _constant_tf(8, 0, 4), FREQS, TVEC, PATTERN, IDEAL, 0,
        thresholds=thresholds,
        data_path_version="marker-padded-v2",
        normalization_mode="condition-specific-marker-rest",
    )

    assert result.scores[4:6].tolist() == [1, 1]
    assert result.threshold_profile == "cohort-2026-calibrated"
    assert result.thresholds_calibrated is True
    assert result.data_path_version == "marker-padded-v2"
    assert result.normalization_mode == "condition-specific-marker-rest"


@pytest.mark.parametrize(
    "field,value",
    [
        ("imagery_minus_quasi_peak_db", (0.0, 0.0, -1.0)),
        ("imagery_minus_quasi_mean_db", (-1.0, 0.0, 1.0)),
        ("ratio_denominator_epsilon", 0.0),
    ],
)
def test_invalid_v2_threshold_profiles_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(DEFAULT_V2_THRESHOLDS, **{field: value})


def test_pattern_similarity_is_sign_invariant_and_label_aligned():
    pattern = np.array([10.0, 20.0, 30.0])
    pattern_labels = ["C3", "C4", "Cz"]
    # Reference is reordered, polarity flipped, and has an extra channel.
    ideal = np.array([-30.0, -99.0, -10.0, -20.0])
    ideal_labels = ["cz", "Pz", "c3", "C4"]
    result = compute_scores_v2(
        _constant_tf(8, 3, 4), FREQS, TVEC,
        pattern, ideal, 1,
        pattern_labels=pattern_labels,
        ideal_pattern_labels=ideal_labels,
    )

    assert result.pat_sim == pytest.approx(1.0)
    assert result.scores[1] == 1
    assert result.pattern_match_mode == "labels"
    assert result.pattern_match_n == 3


def test_label_matching_tolerates_omitted_reference_channel():
    value = pattern_similarity(
        [10.0, 30.0],
        [-10.0, -20.0, -30.0],
        pattern_labels=["C3", "Cz"],
        ideal_pattern_labels=["C3", "C4", "Cz"],
    )
    assert value == pytest.approx(1.0)


def test_pattern_similarity_handles_nan_and_constant_patterns_safely():
    # One NaN pair is omitted; the remaining three points are sufficient.
    partial = compute_scores_v2(
        _constant_tf(8, 3, 4), FREQS, TVEC,
        [1.0, np.nan, 3.0, 4.0], [2.0, 99.0, 6.0, 8.0], 1,
    )
    constant = compute_scores_v2(
        _constant_tf(8, 3, 4), FREQS, TVEC,
        [1.0, 1.0, 1.0], [2.0, 3.0, 4.0], 1,
    )

    assert partial.pat_sim == pytest.approx(1.0)
    assert partial.pattern_match_n == 3
    assert np.isnan(constant.pat_sim)
    assert constant.scores[1] == 0


def test_bad_or_ambiguous_label_metadata_fails_loudly():
    kwargs = dict(
        tf_norm=_constant_tf(8, 3, 4),
        freqNeeded=FREQS,
        tvec=TVEC,
        pattern=[1.0, 2.0, 3.0],
        ideal_pattern=[1.0, 2.0, 3.0],
        n_erd=1,
    )
    with pytest.raises(ValueError, match="supplied together"):
        compute_scores_v2(**kwargs, pattern_labels=["C3", "C4", "Cz"])
    with pytest.raises(ValueError, match="duplicate"):
        compute_scores_v2(
            **kwargs,
            pattern_labels=["C3", "c3", "Cz"],
            ideal_pattern_labels=["C3", "C4", "Cz"],
        )


def test_safe_display_ratios_use_nan_for_near_zero_denominators():
    thresholds = replace(DEFAULT_V2_THRESHOLDS, ratio_denominator_epsilon=1e-5)
    result = compute_scores_v2(
        _constant_tf(1e-8, 1e-8, 4),
        FREQS, TVEC, PATTERN, IDEAL, 0,
        thresholds=thresholds,
    )
    assert np.isnan(result.rel_nonreal_pct)
    assert np.isnan(result.rel_iq_pct)
    assert np.isfinite(result.iq_diff_db)


def test_zero_candidates_and_subject_component_split_are_explicit():
    result = compute_scores_v2(
        _constant_tf(10, 4, 6), FREQS, TVEC, PATTERN, IDEAL, 0)

    assert result.n_erd == 0
    assert result.scores[0] == 0
    assert result.subject_score == 0
    assert result.subject_score_max == 2
    assert result.component_score == int(result.scores[1:].sum())
    assert result.component_score_max == 13
    assert result.total == result.subject_score + result.component_score
    assert result.total_max == 15
    assert result.score_kind == "overt-template-transfer"


def test_all_nan_condition_produces_zero_criteria_not_infinities():
    tf_norm = _constant_tf(8, 3, 4)
    tf_norm[2][:] = np.nan
    result = compute_scores_v2(
        tf_norm, FREQS, TVEC, PATTERN, IDEAL, np.nan)

    assert result.n_erd == 0
    assert np.isnan(result.peakERD[2])
    assert result.scores[4] == 0
    assert result.scores[5] == 0
    assert np.isnan(result.iq_diff_db)
    assert np.isnan(result.rel_iq_pct)


def test_v2_validates_tf_shapes_and_threshold_type():
    tf_norm = _constant_tf(8, 3, 4)
    with pytest.raises(ValueError, match="exactly three"):
        compute_scores_v2(
            tf_norm[:2], FREQS, TVEC, PATTERN, IDEAL, 1)
    with pytest.raises(ValueError, match="expected"):
        compute_scores_v2(
            [tf_norm[0][:-1], tf_norm[1], tf_norm[2]],
            FREQS, TVEC, PATTERN, IDEAL, 1,
        )
    with pytest.raises(TypeError, match="ScoringV2Thresholds"):
        compute_scores_v2(
            tf_norm, FREQS, TVEC, PATTERN, IDEAL, 1,
            thresholds={"profile": "not-a-profile"},
        )


def test_threshold_defaults_are_publicly_marked_as_provisional():
    assert isinstance(DEFAULT_V2_THRESHOLDS, ScoringV2Thresholds)
    assert DEFAULT_V2_THRESHOLDS.calibrated is False
    assert "provisional" in DEFAULT_V2_THRESHOLDS.profile


@pytest.mark.parametrize("version", [SCORING_LEGACY, SCORING_V2])
@pytest.mark.parametrize(
    "marker_mode",
    ["pooled-marker-rest", "condition-specific-marker-rest"],
)
def test_marker_scores_explicitly_exclude_negative_time(version, marker_mode):
    # A large negative-time ERD distinguishes the two definitions.  Marker
    # scoring must ignore it even if a future padded time vector exposes it;
    # trigger/H5 scoring must retain the historical -0.25 s start.
    freqs = np.arange(5.0, 17.0)
    tvec = np.array([-0.25, 0.0, 0.5, 1.0, 2.0, 2.5])
    one_map = np.full((freqs.size, tvec.size), -4.0)
    one_map[:, 0] = -20.0
    tf_norm = [one_map.copy() for _ in range(3)]

    marker = compute_scores(
        tf_norm, freqs, tvec, PATTERN, IDEAL, 0,
        version=version,
        normalization_mode=marker_mode,
    )
    trigger = compute_scores(
        tf_norm, freqs, tvec, PATTERN, IDEAL, 0,
        version=version,
        normalization_mode="condition-specific-prestimulus",
    )

    np.testing.assert_allclose(marker.peakERD, [4.0, 4.0, 4.0])
    np.testing.assert_allclose(trigger.peakERD, [20.0, 20.0, 20.0])
    assert marker.scores[2] == 0
    assert trigger.scores[2] == 3


@pytest.mark.parametrize("version", [SCORING_LEGACY, SCORING_V2])
def test_marker_peak_erd_includes_post_movement_recovery(version):
    freqs = np.arange(5.0, 17.0)
    tvec = np.arange(-0.5, 4.01, 0.5)
    one_map = np.full((freqs.size, tvec.size), -4.0)
    one_map[:, tvec >= 3.0] = -20.0
    tf_norm = [one_map.copy() for _ in range(3)]

    marker = compute_scores(
        tf_norm, freqs, tvec, PATTERN, IDEAL, 0,
        version=version,
        normalization_mode="condition-specific-marker-rest",
    )
    trigger = compute_scores(
        tf_norm, freqs, tvec, PATTERN, IDEAL, 0,
        version=version,
        normalization_mode="condition-specific-prestimulus",
    )

    np.testing.assert_allclose(marker.peakERD, [20.0, 20.0, 20.0])
    np.testing.assert_allclose(trigger.peakERD, [4.0, 4.0, 4.0])
