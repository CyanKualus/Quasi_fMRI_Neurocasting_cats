"""Score-tab gauges must agree with the formula that awards the points.

The gauges previously carried hand-written cutoffs.  They had drifted: the
non-real-versus-overt panel drew lines at 50/80 % while every scoring version
uses 40/60/75 %, three-level criteria showed only two of their three cutoffs,
and out-of-range values were silently clipped out of the axes.  These tests
lock the gauges to the threshold profiles and to the earned points.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from app import _cutoffs, _gauge_scale, _score_gauge_specs
from klh.scoring import (DEFAULT_V2_THRESHOLDS, DEFAULT_V3_THRESHOLDS,
                         DEFAULT_V4_THRESHOLDS, SCORING_LEGACY, SCORING_V2,
                         SCORING_V3, SCORING_V4,
                         compute_scores)

N_CRITERIA = 6


def synth_tf(rng, quasi_db=3.0, imagery_db=2.0, overt_db=6.0, n_f=40, n_t=60):
    """Three normalized TF maps with a controllable mu-band ERD per condition."""
    freq = np.linspace(4.0, 30.0, n_f)
    tvec = np.linspace(-1.0, 3.0, n_t)
    fmask = (freq >= 8) & (freq <= 14)
    tmask = (tvec >= 0.0) & (tvec <= 2.5)
    maps = []
    for depth in (overt_db, quasi_db, imagery_db):
        m = rng.normal(0.0, 0.05, (n_f, n_t))
        m[np.ix_(fmask, tmask)] -= depth
        maps.append(m)
    return maps, freq, tvec


def score_for(version, rng=None, **kwargs):
    rng = np.random.default_rng(0) if rng is None else rng
    tf, freq, tvec = synth_tf(rng, **kwargs)
    pattern = rng.normal(size=8)
    ideal = pattern + rng.normal(0, 0.05, 8)
    return compute_scores(tf, freq, tvec, pattern, ideal, 3, version=version)


def earned(spec):
    """Points the drawn cutoffs claim for the plotted value (strict ``>``)."""
    value = float(spec["value"])
    for points, cut in spec["cutoffs"]:
        if value > cut:
            return points
    return 0


@pytest.mark.parametrize(
    "version", [SCORING_V4, SCORING_V3, SCORING_V2, SCORING_LEGACY])
def test_every_version_yields_six_complete_gauges(version):
    specs = _score_gauge_specs(score_for(version))
    assert len(specs) == N_CRITERIA
    for spec in specs:
        assert spec["cutoffs"], f"{spec['title']} has no cutoff line"
        # A three-point criterion must show all three cutoffs, not just two.
        assert len(spec["cutoffs"]) == spec["points_max"]
        points = [p for p, _ in spec["cutoffs"]]
        assert points == sorted(points, reverse=True)
        assert points[0] == spec["points_max"]


def test_v3_cutoffs_come_from_the_v3_threshold_profile():
    thr = DEFAULT_V3_THRESHOLDS
    specs = _score_gauge_specs(score_for(SCORING_V3))

    assert specs[1]["cutoffs"] == [(1, thr.pattern_similarity_min)]
    assert specs[2]["cutoffs"] == _cutoffs(thr.overt_erd_db)
    assert specs[3]["cutoffs"] == _cutoffs(
        tuple(100.0 * f for f in thr.nonreal_overt_fraction))
    # The historical 50/80 % lines matched no threshold in any version.
    assert [cut for _, cut in specs[3]["cutoffs"]] == [75.0, 60.0, 40.0]
    for spec in specs[4:]:
        assert spec["cutoffs"] == _cutoffs(thr.sustained_erd_db)
        # v3 scores quasi and imagery independently; neither is gated.
        assert spec.get("gate") is None


def test_v4_cutoffs_come_from_the_v4_threshold_profile():
    thr = DEFAULT_V4_THRESHOLDS
    specs = _score_gauge_specs(score_for(SCORING_V4))

    assert specs[1]["cutoffs"] == [(1, thr.pattern_similarity_min)]
    assert specs[2]["cutoffs"] == _cutoffs(thr.overt_erd_db)
    assert specs[3]["cutoffs"] == _cutoffs(
        tuple(100.0 * f for f in thr.nonreal_overt_fraction))
    assert "500 ms" in specs[2]["title"]
    for spec in specs[4:]:
        assert spec["cutoffs"] == _cutoffs(thr.sustained_erd_db)
        assert spec.get("gate") is None


def test_v2_cutoffs_come_from_the_v2_threshold_profile():
    thr = DEFAULT_V2_THRESHOLDS
    specs = _score_gauge_specs(score_for(SCORING_V2))
    assert specs[2]["cutoffs"] == _cutoffs(thr.overt_erd_db)
    assert specs[4]["cutoffs"] == _cutoffs(thr.imagery_minus_quasi_peak_db)
    assert specs[5]["cutoffs"] == _cutoffs(thr.imagery_minus_quasi_mean_db)


@pytest.mark.parametrize(
    "version", [SCORING_V4, SCORING_V3, SCORING_V2, SCORING_LEGACY])
@pytest.mark.parametrize("seed", range(6))
def test_marker_side_of_each_line_matches_the_points_awarded(version, seed):
    rng = np.random.default_rng(seed)
    score = score_for(
        version, rng,
        overt_db=float(rng.uniform(0.5, 16.0)),
        quasi_db=float(rng.uniform(0.0, 6.0)),
        imagery_db=float(rng.uniform(0.0, 6.0)))
    specs = _score_gauge_specs(score)
    for i, spec in enumerate(specs):
        if not np.isfinite(float(spec["value"])):
            continue
        if spec.get("gate"):
            # A gated criterion scores zero wherever the marker sits; the
            # panel says so instead of pretending the lines explain it.
            assert score.scores[i] == 0
            continue
        assert earned(spec) == score.scores[i], (
            f"{version} criterion {i + 1} ({spec['title']}): marker at "
            f"{spec['value']} vs cutoffs {spec['cutoffs']} but scored "
            f"{score.scores[i]}")


def test_gate_is_flagged_when_imagery_floor_is_not_cleared():
    # Imagery peak ERD below the floor zeroes v2 criteria 5 and 6 regardless
    # of the plotted difference, which used to leave the marker unexplained.
    score = score_for(SCORING_V2, imagery_db=0.2, quasi_db=0.1)
    specs = _score_gauge_specs(score)
    assert score.peakERD[2] <= DEFAULT_V2_THRESHOLDS.imagery_min_erd_db
    for spec in specs[4:]:
        assert spec["gate"] and "imagery peak ERD" in spec["gate"]
    assert list(score.scores[4:]) == [0.0, 0.0]


def test_legacy_similarity_axis_admits_negative_correlations():
    # legacy-v1 pat_sim is np.min(corrcoef) and is routinely negative; a 0..1
    # axis clipped the marker away and read as missing data.
    specs = _score_gauge_specs(score_for(SCORING_LEGACY))
    lo, _ = specs[1]["lim"]
    assert lo <= -1.0


@pytest.mark.parametrize("value", [-900.0, -12.0, 0.0, 7.5, 42.0, 5000.0])
def test_no_finite_value_is_ever_clipped_out_of_the_gauge(value):
    cutoffs = _cutoffs((3.0, 2.0, 1.0))
    (lo, hi), marker_x, marker = _gauge_scale((-2, 6), cutoffs, value)
    assert lo < marker_x < hi, "marker fell outside the axes and would vanish"
    for _, cut in cutoffs:
        assert lo < cut < hi, "a cutoff line fell outside the axes"
    if marker == "D":
        assert marker_x == pytest.approx(value)
    else:
        # Pinned to the edge: the arrow must point the way the value went.
        assert marker == ("<" if value < 0 else ">")


@pytest.mark.parametrize(
    "version", [SCORING_V4, SCORING_V3, SCORING_V2, SCORING_LEGACY])
def test_bounded_and_discrete_criteria_declare_their_domain(version):
    specs = _score_gauge_specs(score_for(version))
    # A component count is discrete and non-negative; fractional ticks such as
    # 2.5 or 7.5 name quantities the criterion cannot take.
    assert specs[0]["integer_ticks"] is True
    assert specs[0]["bounds"] == (-0.5, None)
    # Similarity is a correlation: |corr| <= 1 in v2/v3, corr >= -1 in legacy.
    lo, hi = specs[1]["bounds"]
    assert hi == 1.0
    assert lo == (-1.0 if version == SCORING_LEGACY else 0.0)
    # The unbounded dB and percentage criteria stay free to autoscale.
    for spec in specs[2:]:
        assert spec.get("bounds") is None


@pytest.mark.parametrize("value", [0.0, 0.5, 0.885, 1.0])
def test_similarity_axis_never_extends_past_the_correlation_domain(value):
    cutoffs = _cutoffs((0.9,), points_max=1)
    (lo, hi), marker_x, marker = _gauge_scale(
        (0.0, 1.0), cutoffs, value, bounds=(0.0, 1.0))
    assert (lo, hi) == (0.0, 1.0)
    assert marker == "D" and marker_x == pytest.approx(value)


def test_count_axis_never_extends_below_zero_components():
    cutoffs = _cutoffs((2.5, 0.5), points_max=2)
    (lo, hi), marker_x, marker = _gauge_scale(
        (-0.5, 10.5), cutoffs, 0, bounds=(-0.5, None))
    assert lo == -0.5
    assert hi > 10.5, "the upper end stays free to autoscale"
    assert marker == "D" and marker_x == 0


def test_domain_bounds_never_hide_a_cutoff_line():
    # A bound tighter than a cutoff must yield to the cutoff, not swallow it.
    cutoffs = _cutoffs((0.9,), points_max=1)
    (lo, hi), _, _ = _gauge_scale(
        (0.0, 1.0), cutoffs, 0.5, bounds=(0.0, 0.8))
    assert lo < 0.9 < hi


def test_non_finite_value_keeps_the_nominal_range():
    cutoffs = _cutoffs((3.0, 2.0, 1.0))
    (lo, hi), marker_x, marker = _gauge_scale((-2, 6), cutoffs, np.nan)
    assert not np.isfinite(marker_x)
    assert marker == "D"
    assert lo < -2 and hi > 6
