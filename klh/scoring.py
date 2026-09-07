"""Versioned scoring for overt-template transfer components.

The historical KLH score is retained as :data:`SCORING_LEGACY`.  Its arithmetic
is intentionally unchanged, including the ratio of dB values in criterion 6,
so archived MATLAB/Python results remain reproducible.

The corrected :data:`SCORING_V2` score fixes the known edge cases:

* criterion 4 is unavailable when overt ERD is non-positive;
* criteria 5 and 6 compare imagery with quasi by dB differences;
* pattern correlation is polarity invariant and can be aligned by channel
  label; and
* ratios used only for display become NaN when their denominator is invalid.

The :data:`SCORING_V3` score keeps v2 criteria 1--4 and replaces the two
imagery-versus-quasi comparisons.  Criteria 5 and 6 independently score the
sustained 9--13 Hz ERD of quasi-movement and motor imagery in the fixed
0.5--2.0 s active window.

The :data:`SCORING_V4` score preserves v3's independent non-real criteria but
replaces the extreme third-percentile peak diagnostic used by criteria 3 and
4.  Its peak is the strongest continuous 500 ms rolling median of the
frequency-median 5--16 Hz time course.

Condition order everywhere: 0 = real/overt, 1 = quasi, 2 = imagery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .mathutil import matlab_prctile, rng_inclusive


SCORING_LEGACY = "legacy-v1"
SCORING_V2 = "scoring-v2"
SCORING_V3 = "scoring-v3-independent-nonreal-erd"
SCORING_V4 = "scoring-v4-rolling-peak-independent-nonreal-erd"
SUPPORTED_SCORING_VERSIONS = (
    SCORING_LEGACY, SCORING_V2, SCORING_V3, SCORING_V4)

SCORES_MAX = np.array([2, 1, 3, 3, 3, 3])
SUBJECT_SCORE_SLICE = slice(0, 1)
COMPONENT_SCORE_SLICE = slice(1, None)
MARKER_NORMALIZATION_MODES = frozenset({
    "pooled-marker-rest",
    "condition-specific-marker-rest",
})


@dataclass(frozen=True)
class ScoringV2Thresholds:
    """Threshold profile used by :func:`compute_scores_v2`.

    The default difference thresholds are *provisional*, not cohort
    calibrated.  Criterion 5 is anchored to the historical ratios at a
    representative 4 dB quasi ERD: 100%, 75%, and 50% become imagery-minus-
    quasi differences of 0, -1, and -2 dB.  Criterion 6 maps the historical
    +/-15% cutoffs at the same -4 dB reference to +/-0.6 dB.  Replace this
    profile after re-scoring the reference cohort and set ``calibrated=True``.

    Triplets are ordered from the three-point cutoff through the one-point
    cutoff.  Comparisons retain the historical strict ``>`` semantics.
    """

    profile: str = "provisional-minus4db-v1"
    calibrated: bool = False
    pattern_similarity_min: float = 0.90
    overt_erd_db: tuple[float, float, float] = (12.0, 8.0, 5.0)
    nonreal_overt_fraction: tuple[float, float, float] = (0.75, 0.60, 0.40)
    imagery_min_erd_db: float = 3.0
    imagery_minus_quasi_peak_db: tuple[float, float, float] = (0.0, -1.0, -2.0)
    imagery_minus_quasi_mean_db: tuple[float, float, float] = (0.60, 0.0, -0.60)
    ratio_denominator_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not self.profile.strip():
            raise ValueError("threshold profile must have a non-empty name")
        for name in (
            "overt_erd_db",
            "nonreal_overt_fraction",
            "imagery_minus_quasi_peak_db",
            "imagery_minus_quasi_mean_db",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite thresholds")
            if not (values[0] > values[1] > values[2]):
                raise ValueError(
                    f"{name} thresholds must be strictly descending (3, 2, 1 points)")
        if not np.isfinite(self.pattern_similarity_min):
            raise ValueError("pattern_similarity_min must be finite")
        if not np.isfinite(self.imagery_min_erd_db):
            raise ValueError("imagery_min_erd_db must be finite")
        if not np.isfinite(self.ratio_denominator_epsilon) or self.ratio_denominator_epsilon <= 0:
            raise ValueError("ratio_denominator_epsilon must be finite and positive")


DEFAULT_V2_THRESHOLDS = ScoringV2Thresholds()


@dataclass(frozen=True)
class ScoringV3Thresholds:
    """Threshold profile for independent sustained quasi/imagery mu-ERD.

    The shipped 1/2/3 dB cutoffs are interpretable starting anchors, not a
    cohort norm.  Triplets are ordered from the three-point cutoff through the
    one-point cutoff and use strict ``>`` comparisons, so exactly 1 dB earns
    zero points, exactly 2 dB earns one, and exactly 3 dB earns two.
    """

    profile: str = "provisional-independent-mu-erd-v1"
    calibrated: bool = False
    pattern_similarity_min: float = 0.90
    overt_erd_db: tuple[float, float, float] = (12.0, 8.0, 5.0)
    nonreal_overt_fraction: tuple[float, float, float] = (0.75, 0.60, 0.40)
    sustained_erd_db: tuple[float, float, float] = (3.0, 2.0, 1.0)
    frequency_window_hz: tuple[float, float] = (9.0, 13.0)
    time_window_s: tuple[float, float] = (0.5, 2.0)
    ratio_denominator_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if not self.profile.strip():
            raise ValueError("threshold profile must have a non-empty name")
        for name in (
            "overt_erd_db",
            "nonreal_overt_fraction",
            "sustained_erd_db",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite thresholds")
            if not (values[0] > values[1] > values[2]):
                raise ValueError(
                    f"{name} thresholds must be strictly descending "
                    "(3, 2, 1 points)")
        for name in ("frequency_window_hz", "time_window_s"):
            values = np.asarray(getattr(self, name), dtype=float)
            if (values.shape != (2,) or not np.all(np.isfinite(values))
                    or values[0] >= values[1]):
                raise ValueError(
                    f"{name} must contain two finite increasing endpoints")
        if not np.isfinite(self.pattern_similarity_min):
            raise ValueError("pattern_similarity_min must be finite")
        if (not np.isfinite(self.ratio_denominator_epsilon)
                or self.ratio_denominator_epsilon <= 0):
            raise ValueError(
                "ratio_denominator_epsilon must be finite and positive")


DEFAULT_V3_THRESHOLDS = ScoringV3Thresholds()


@dataclass(frozen=True)
class ScoringV4Thresholds(ScoringV3Thresholds):
    """Threshold profile for the rolling-peak independent-ERD score.

    The rolling-peak cutoffs are provisional starting values chosen on the
    statistic's interpretable power-reduction scale: 1, 3 and 5 dB correspond
    to about 21%, 50% and 68% reductions.  Sustained quasi and imagery ERD use
    the more conservative 2, 3 and 4 dB boundaries.  Both sets still require
    cohort validation.
    """

    profile: str = "provisional-rolling500ms-independent-mu-erd-v3"
    overt_erd_db: tuple[float, float, float] = (5.0, 3.0, 1.0)
    sustained_erd_db: tuple[float, float, float] = (4.0, 3.0, 2.0)
    peak_erd_frequency_window_hz: tuple[float, float] = (5.0, 16.0)
    peak_erd_rolling_window_s: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        frequency = np.asarray(self.peak_erd_frequency_window_hz, dtype=float)
        if (frequency.shape != (2,) or not np.all(np.isfinite(frequency))
                or frequency[0] >= frequency[1]):
            raise ValueError(
                "peak_erd_frequency_window_hz must contain two finite "
                "increasing endpoints")
        if (not np.isfinite(self.peak_erd_rolling_window_s)
                or self.peak_erd_rolling_window_s <= 0):
            raise ValueError(
                "peak_erd_rolling_window_s must be finite and positive")


DEFAULT_V4_THRESHOLDS = ScoringV4Thresholds()


@dataclass
class ScoreResult:
    """A six-criterion score plus the diagnostics needed to interpret it."""

    scores: np.ndarray                 # length 6
    scores_max: np.ndarray = field(default_factory=lambda: SCORES_MAX.copy())
    pat_sim: float = np.nan
    peakERD: np.ndarray = field(default_factory=lambda: np.zeros(3))
    peak_erd_method: str = "negative-third-percentile"
    peak_erd_frequency_window_hz: tuple[float, float] | None = (5.0, 16.0)
    peak_erd_rolling_window_s: float | None = None
    peak_erd_strongest_window_s: dict[str, tuple[float, float]] = field(
        default_factory=dict)
    n_erd: int = 0
    rel_nonreal_pct: float = np.nan     # display-only ratio
    rel_iq_pct: float = np.nan          # display-only ratio
    iq_diff_mean_pct: float = np.nan    # legacy dB ratio; unavailable in v2
    iq_diff_db: float = np.nan          # median quasi dB - median imagery dB
    peak_iq_diff_db: float = np.nan     # peak imagery ERD - peak quasi ERD
    # V3 diagnostics.  Dictionaries use explicit condition names so JSON
    # exports do not require consumers to know an array-order convention.
    sustained_erd_db: dict[str, float] = field(default_factory=dict)
    sustained_erd_temporal_coverage: dict[str, float] = field(default_factory=dict)
    sustained_erd_frequency_window_hz: tuple[float, float] | None = None
    sustained_erd_time_window_s: tuple[float, float] | None = None
    sustained_erd_ci_db: dict = field(default_factory=dict)
    sustained_erd_uncertainty: dict = field(default_factory=dict)
    sustained_erd_run_consistency: dict = field(default_factory=dict)

    # Explicit result provenance.  Data-path values are supplied by the
    # pipeline because scoring.py cannot infer resampling/baseline decisions.
    scoring_version: str = SCORING_LEGACY
    score_kind: str = "overt-template-transfer"
    threshold_profile: str = "matlab-original"
    thresholds_calibrated: bool | None = None
    data_path_version: str | None = None
    normalization_mode: str | None = None
    pattern_match_mode: str = "legacy-positional"
    pattern_match_n: int = 0

    @property
    def total(self) -> int:
        return int(self.scores.sum())

    @property
    def total_max(self) -> int:
        return int(self.scores_max.sum())

    @property
    def subject_score(self) -> int:
        """Criterion 1: subject/eigenvalue-spectrum score (maximum 2)."""
        return int(self.scores[SUBJECT_SCORE_SLICE].sum())

    @property
    def subject_score_max(self) -> int:
        return int(self.scores_max[SUBJECT_SCORE_SLICE].sum())

    @property
    def component_score(self) -> int:
        """Criteria 2-6: selected-component transfer score (maximum 13)."""
        return int(self.scores[COMPONENT_SCORE_SLICE].sum())

    @property
    def component_score_max(self) -> int:
        return int(self.scores_max[COMPONENT_SCORE_SLICE].sum())

    @property
    def version(self) -> str:
        """Short compatibility alias for exporters that use ``version``."""
        return self.scoring_version


def _peak_erd_time_slice(tvec, normalization_mode=None) -> slice:
    """Return the score interval shared by historical and rolling peaks."""
    if normalization_mode in MARKER_NORMALIZATION_MODES:
        return rng_inclusive(tvec, 0.0, float(np.max(tvec)))
    return rng_inclusive(tvec, -0.25, 2.5)


def _historical_peak_erd(
    tf_norm, freq_needed, tvec, normalization_mode=None,
) -> np.ndarray:
    """Historical negative-third-percentile ERD per condition.

    Trigger/pre-stimulus data retain the historical -0.25..2.5 s interval.
    Marker epochs begin at motor-cue onset after preparation has been excluded;
    their complete non-negative core contains both motor execution and the
    configured post-movement recovery interval. Relying on nearest-index
    clamping would conceal that semantic difference.
    """
    fsl = rng_inclusive(freq_needed, 5, 16)
    tsl = _peak_erd_time_slice(tvec, normalization_mode)
    return np.array([
        -matlab_prctile(tf_norm[c][fsl, tsl], 3) for c in range(3)
    ])


def _rolling_median_peak_erd(
    tf_norm,
    freq_needed,
    tvec,
    normalization_mode=None,
    *,
    frequency_window_hz=(5.0, 16.0),
    rolling_window_s=0.5,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Return the strongest continuous rolling-median ERD per condition.

    Each time point is first reduced to the median normalized dB change across
    the declared frequency band.  A rolling median is then taken over complete
    ``rolling_window_s`` windows, and the most negative window is reported as
    a positive ERD magnitude.  Unlike a percentile over flattened TF cells,
    every contributing sample therefore belongs to one continuous interval.
    """
    fsl = rng_inclusive(freq_needed, *frequency_window_hz)
    tsl = _peak_erd_time_slice(tvec, normalization_mode)
    score_times = np.asarray(tvec[tsl], dtype=float)
    condition_names = ("overt", "quasi", "imagery")
    values = np.full(3, np.nan)
    bounds: dict[str, tuple[float, float]] = {}

    if score_times.size < 2:
        return values, bounds
    steps = np.diff(score_times)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("tvec must be finite and strictly increasing")
    sample_period_s = float(np.median(steps))
    window_samples = max(
        1, int(np.ceil(float(rolling_window_s) / sample_period_s - 1e-12)))
    if window_samples > score_times.size:
        return values, bounds

    for index, condition in enumerate(condition_names):
        region = np.asarray(tf_norm[index][fsl, tsl], dtype=float)
        frequency_median = np.full(region.shape[1], np.nan)
        usable = np.any(np.isfinite(region), axis=0)
        if np.any(usable):
            frequency_median[usable] = np.nanmedian(
                region[:, usable], axis=0)
        windows = np.lib.stride_tricks.sliding_window_view(
            frequency_median, window_samples)
        rolling = np.median(windows, axis=-1)
        finite = np.flatnonzero(np.isfinite(rolling))
        if not finite.size:
            continue
        minimum = float(np.min(rolling[finite]))
        tied = finite[np.isclose(rolling[finite], minimum, rtol=0.0,
                                 atol=np.finfo(float).eps * 16)]
        # Broad plateaus often give several windows the same median.  The mean
        # is used only to report the most representative location; it never
        # changes the median-based peak magnitude or awarded points.
        if tied.size > 1:
            tied_means = np.mean(windows[tied], axis=-1)
            start_index = int(tied[np.argmin(tied_means)])
        else:
            start_index = int(tied[0])
        values[index] = float(-rolling[start_index])
        start_s = float(score_times[start_index])
        bounds[condition] = (
            start_s, float(start_s + float(rolling_window_s)))
    return values, bounds


def _three_level_score(value: float, thresholds: Sequence[float]) -> float:
    if not np.isfinite(value):
        return 0.0
    for points, threshold in zip((3.0, 2.0, 1.0), thresholds):
        if value > threshold:
            return points
    return 0.0


def _canonical_labels(labels: Sequence[str], expected: int, name: str) -> list[str]:
    if len(labels) != expected:
        raise ValueError(f"{name} has {len(labels)} labels for {expected} pattern values")
    canonical = []
    for label in labels:
        if isinstance(label, bytes):
            label = label.decode("utf-8", errors="replace")
        key = str(label).strip().casefold()
        if not key:
            raise ValueError(f"{name} contains an empty channel label")
        canonical.append(key)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{name} contains duplicate channel labels")
    return canonical


def _safe_pattern_similarity(
    pattern,
    ideal_pattern,
    *,
    pattern_labels: Sequence[str] | None = None,
    ideal_pattern_labels: Sequence[str] | None = None,
) -> tuple[float, str, int]:
    """Return absolute off-diagonal correlation and alignment diagnostics."""
    p = np.asarray(pattern, dtype=float).ravel()
    ip = np.asarray(ideal_pattern, dtype=float).ravel()

    if (pattern_labels is None) != (ideal_pattern_labels is None):
        raise ValueError(
            "pattern_labels and ideal_pattern_labels must be supplied together")

    if pattern_labels is not None:
        p_labels = _canonical_labels(pattern_labels, p.size, "pattern_labels")
        ip_labels = _canonical_labels(
            ideal_pattern_labels, ip.size, "ideal_pattern_labels")
        ideal_by_label = {label: idx for idx, label in enumerate(ip_labels)}
        pairs = [
            (idx, ideal_by_label[label])
            for idx, label in enumerate(p_labels)
            if label in ideal_by_label
        ]
        if pairs:
            p = p[[pair[0] for pair in pairs]]
            ip = ip[[pair[1] for pair in pairs]]
        else:
            p = np.empty(0, dtype=float)
            ip = np.empty(0, dtype=float)
        mode = "labels"
    else:
        n = min(p.size, ip.size)
        p = p[:n]
        ip = ip[:n]
        mode = "positional"

    finite = np.isfinite(p) & np.isfinite(ip)
    p = p[finite]
    ip = ip[finite]
    n_finite = int(p.size)
    if n_finite < 2:
        return np.nan, mode, n_finite

    p = p - p.mean()
    ip = ip - ip.mean()
    denom = float(np.linalg.norm(p) * np.linalg.norm(ip))
    if not np.isfinite(denom) or denom <= np.finfo(float).eps:
        return np.nan, mode, n_finite
    corr = float(np.dot(p, ip) / denom)
    if not np.isfinite(corr):
        return np.nan, mode, n_finite
    return abs(float(np.clip(corr, -1.0, 1.0))), mode, n_finite


def pattern_similarity(
    pattern,
    ideal_pattern,
    *,
    pattern_labels: Sequence[str] | None = None,
    ideal_pattern_labels: Sequence[str] | None = None,
) -> float:
    """Safe polarity-invariant correlation, optionally matched by EEG label."""
    value, _, _ = _safe_pattern_similarity(
        pattern,
        ideal_pattern,
        pattern_labels=pattern_labels,
        ideal_pattern_labels=ideal_pattern_labels,
    )
    return value


def _safe_display_ratio(numerator: float, denominator: float, epsilon: float) -> float:
    """Return a percentage or NaN for non-finite/non-positive denominators."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if denominator <= epsilon:
        return np.nan
    value = 100.0 * numerator / denominator
    return float(value) if np.isfinite(value) else np.nan


def _validated_tf(tf_norm, freq_needed, tvec) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    freq_needed = np.asarray(freq_needed, dtype=float).ravel()
    tvec = np.asarray(tvec, dtype=float).ravel()
    if freq_needed.size == 0 or tvec.size == 0:
        raise ValueError("frequency and time axes must be non-empty")
    if len(tf_norm) != 3:
        raise ValueError("tf_norm must contain exactly three conditions")
    maps = [np.asarray(item, dtype=float) for item in tf_norm]
    expected = (freq_needed.size, tvec.size)
    for index, item in enumerate(maps):
        if item.shape != expected:
            raise ValueError(
                f"tf_norm[{index}] has shape {item.shape}; expected {expected}")
    return maps, freq_needed, tvec


def _sustained_erd_diagnostics(
    tf_norm: Sequence[np.ndarray],
    freq_needed: np.ndarray,
    tvec: np.ndarray,
    frequency_window_hz: tuple[float, float],
    time_window_s: tuple[float, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return regional ERD magnitude and temporal coverage for quasi/imagery.

    Magnitude is the negative median of the complete time-frequency region.
    Coverage is evaluated time sample by time sample after taking the median
    across frequency, and is descriptive only: it never gates v3 points.
    """
    fsl = rng_inclusive(
        freq_needed, *frequency_window_hz, strict=True, warn=False,
        label="sustained mu-ERD frequency window")
    tsl = rng_inclusive(
        tvec, *time_window_s, strict=True, warn=False,
        label="sustained mu-ERD active window")
    magnitudes: dict[str, float] = {}
    coverage: dict[str, float] = {}
    for index, condition in ((1, "quasi"), (2, "imagery")):
        region = np.asarray(tf_norm[index][fsl, tsl], dtype=float)
        magnitudes[condition] = float(-np.median(region))
        frequency_median = np.median(region, axis=0)
        finite = np.isfinite(frequency_median)
        coverage[condition] = (
            float(np.mean(frequency_median[finite] < 0.0))
            if np.any(finite) else np.nan)
    return magnitudes, coverage


def compute_scores_legacy(
    tf_norm,
    freqNeeded,
    tvec,
    pattern,
    ideal_pattern,
    n_erd,
    *,
    data_path_version: str | None = None,
    normalization_mode: str | None = None,
) -> ScoreResult:
    """Compute the historical score exactly as the original Python port did.

    This function deliberately retains sign-sensitive positional correlation,
    ungated criterion 4, dB ratios, and historical zero-denominator display
    values.  Use it only for reproducibility/comparison with archived scores.
    """
    freqNeeded = np.asarray(freqNeeded)
    tvec = np.asarray(tvec)
    scores = np.zeros(6)

    # Keep this expression verbatim: np.min selects a negative off-diagonal
    # correlation but selects the off-diagonal value for positive correlation.
    p = np.asarray(pattern).ravel()
    ip = np.asarray(ideal_pattern).ravel()
    n = min(p.size, ip.size)
    pat_sim = float(np.min(np.corrcoef(p[:n], ip[:n])))

    peakERD = _historical_peak_erd(
        tf_norm, freqNeeded, tvec, normalization_mode=normalization_mode)

    if n_erd > 2:
        scores[0] = 2
    elif n_erd >= 1:
        scores[0] = 1

    if pat_sim > 0.9:
        scores[1] = 1

    if peakERD[0] > 12:
        scores[2] = 3
    elif peakERD[0] > 8:
        scores[2] = 2
    elif peakERD[0] > 5:
        scores[2] = 1

    rel_nonreal = max(peakERD[1], peakERD[2])
    if rel_nonreal > peakERD[0] * 0.75:
        scores[3] = 3
    elif rel_nonreal > peakERD[0] * 0.60:
        scores[3] = 2
    elif rel_nonreal > peakERD[0] * 0.40:
        scores[3] = 1

    if peakERD[2] > 3 and peakERD[2] > peakERD[1] * 1.0:
        scores[4] = 3
    elif peakERD[2] > 3 and peakERD[2] > peakERD[1] * 0.75:
        scores[4] = 2
    elif peakERD[2] > 3 and peakERD[2] > peakERD[1] * 0.5:
        scores[4] = 1

    f2 = rng_inclusive(freqNeeded, 9, 13)
    t2 = rng_inclusive(tvec, 0.5, 2.0)
    med_imag = np.median(tf_norm[2][f2, t2])
    med_quasi = np.median(tf_norm[1][f2, t2])
    iq_diff_mean = 100.0 * (med_imag / med_quasi) - 100.0

    if peakERD[2] > 3 and iq_diff_mean > 15:
        scores[5] = 3
    elif peakERD[2] > 3 and iq_diff_mean > 0:
        scores[5] = 2
    elif peakERD[2] > 3 and iq_diff_mean > -15:
        scores[5] = 1

    rel_nonreal_pct = 100.0 * (rel_nonreal / peakERD[0]) if peakERD[0] else 0.0
    rel_iq_pct = 100.0 * (peakERD[2] / peakERD[1]) if peakERD[1] else 0.0

    return ScoreResult(
        scores=scores,
        pat_sim=pat_sim,
        peakERD=peakERD,
        n_erd=int(n_erd),
        rel_nonreal_pct=rel_nonreal_pct,
        rel_iq_pct=rel_iq_pct,
        iq_diff_mean_pct=iq_diff_mean,
        iq_diff_db=float(med_quasi - med_imag),
        peak_iq_diff_db=float(peakERD[2] - peakERD[1]),
        scoring_version=SCORING_LEGACY,
        threshold_profile="matlab-original",
        thresholds_calibrated=None,
        data_path_version=data_path_version,
        normalization_mode=normalization_mode,
        pattern_match_mode="legacy-positional",
        pattern_match_n=n,
    )


def compute_scores_v2(
    tf_norm,
    freqNeeded,
    tvec,
    pattern,
    ideal_pattern,
    n_erd,
    *,
    thresholds: ScoringV2Thresholds | None = None,
    pattern_labels: Sequence[str] | None = None,
    ideal_pattern_labels: Sequence[str] | None = None,
    data_path_version: str | None = None,
    normalization_mode: str | None = None,
) -> ScoreResult:
    """Compute the corrected, explicitly versioned v2 transfer score."""
    thresholds = DEFAULT_V2_THRESHOLDS if thresholds is None else thresholds
    if not isinstance(thresholds, ScoringV2Thresholds):
        raise TypeError("thresholds must be a ScoringV2Thresholds instance")

    tf_norm, freqNeeded, tvec = _validated_tf(tf_norm, freqNeeded, tvec)
    scores = np.zeros(6)
    pat_sim, match_mode, match_n = _safe_pattern_similarity(
        pattern,
        ideal_pattern,
        pattern_labels=pattern_labels,
        ideal_pattern_labels=ideal_pattern_labels,
    )
    peakERD = _historical_peak_erd(
        tf_norm, freqNeeded, tvec, normalization_mode=normalization_mode)

    try:
        n_erd_value = float(n_erd)
    except (TypeError, ValueError):
        n_erd_value = np.nan
    n_erd_clean = max(0, int(n_erd_value)) if np.isfinite(n_erd_value) else 0

    # (1) Subject-level number of candidate ERD components.
    if n_erd_clean > 2:
        scores[0] = 2
    elif n_erd_clean >= 1:
        scores[0] = 1

    # (2) Iconic pattern similarity.  CSP polarity is arbitrary.
    if np.isfinite(pat_sim) and pat_sim > thresholds.pattern_similarity_min:
        scores[1] = 1

    # (3) Absolute overt ERD magnitude.
    scores[2] = _three_level_score(peakERD[0], thresholds.overt_erd_db)

    finite_nonreal = peakERD[1:][np.isfinite(peakERD[1:])]
    rel_nonreal = float(np.max(finite_nonreal)) if finite_nonreal.size else np.nan

    # (4) Transfer relative to overt is undefined without a positive overt ERD.
    if np.isfinite(peakERD[0]) and peakERD[0] > 0 and np.isfinite(rel_nonreal):
        scores[3] = _three_level_score(
            rel_nonreal / peakERD[0], thresholds.nonreal_overt_fraction)

    # (5) Peak imagery-minus-quasi ERD magnitude, expressed directly in dB.
    peak_iq_diff_db = float(peakERD[2] - peakERD[1])
    if np.isfinite(peakERD[2]) and peakERD[2] > thresholds.imagery_min_erd_db:
        scores[4] = _three_level_score(
            peak_iq_diff_db, thresholds.imagery_minus_quasi_peak_db)

    # (6) Mean-region difference.  This is a log power ratio in dB, not a
    # dimensionally invalid ratio between two logarithms.
    f2 = rng_inclusive(freqNeeded, 9, 13)
    t2 = rng_inclusive(tvec, 0.5, 2.0)
    med_imag_db = float(np.median(tf_norm[2][f2, t2]))
    med_quasi_db = float(np.median(tf_norm[1][f2, t2]))
    iq_diff_db = float(med_quasi_db - med_imag_db)
    if np.isfinite(peakERD[2]) and peakERD[2] > thresholds.imagery_min_erd_db:
        scores[5] = _three_level_score(
            iq_diff_db, thresholds.imagery_minus_quasi_mean_db)

    epsilon = thresholds.ratio_denominator_epsilon
    rel_nonreal_pct = _safe_display_ratio(rel_nonreal, peakERD[0], epsilon)
    rel_iq_pct = _safe_display_ratio(peakERD[2], peakERD[1], epsilon)

    return ScoreResult(
        scores=scores,
        pat_sim=pat_sim,
        peakERD=peakERD,
        n_erd=n_erd_clean,
        rel_nonreal_pct=rel_nonreal_pct,
        rel_iq_pct=rel_iq_pct,
        iq_diff_mean_pct=np.nan,
        iq_diff_db=iq_diff_db,
        peak_iq_diff_db=peak_iq_diff_db,
        scoring_version=SCORING_V2,
        threshold_profile=thresholds.profile,
        thresholds_calibrated=thresholds.calibrated,
        data_path_version=data_path_version,
        normalization_mode=normalization_mode,
        pattern_match_mode=match_mode,
        pattern_match_n=match_n,
    )


def compute_scores_v3(
    tf_norm,
    freqNeeded,
    tvec,
    pattern,
    ideal_pattern,
    n_erd,
    *,
    thresholds: ScoringV3Thresholds | None = None,
    pattern_labels: Sequence[str] | None = None,
    ideal_pattern_labels: Sequence[str] | None = None,
    data_path_version: str | None = None,
    normalization_mode: str | None = None,
    sustained_erd_support: dict | None = None,
) -> ScoreResult:
    """Score sustained quasi and imagery ERD independently.

    Criteria 1--4 retain the corrected v2 definitions.  Criterion 5 uses only
    quasi-movement's regional ERD and criterion 6 uses only motor imagery's;
    neither condition's points depend on their ordering or difference.
    """
    thresholds = DEFAULT_V3_THRESHOLDS if thresholds is None else thresholds
    if type(thresholds) is not ScoringV3Thresholds:
        raise TypeError("thresholds must be a ScoringV3Thresholds instance")

    tf_norm, freqNeeded, tvec = _validated_tf(tf_norm, freqNeeded, tvec)
    scores = np.zeros(6)
    pat_sim, match_mode, match_n = _safe_pattern_similarity(
        pattern,
        ideal_pattern,
        pattern_labels=pattern_labels,
        ideal_pattern_labels=ideal_pattern_labels,
    )
    peakERD = _historical_peak_erd(
        tf_norm, freqNeeded, tvec, normalization_mode=normalization_mode)

    try:
        n_erd_value = float(n_erd)
    except (TypeError, ValueError):
        n_erd_value = np.nan
    n_erd_clean = max(0, int(n_erd_value)) if np.isfinite(n_erd_value) else 0

    # Criteria 1--4 intentionally match scoring-v2.
    if n_erd_clean > 2:
        scores[0] = 2
    elif n_erd_clean >= 1:
        scores[0] = 1
    if np.isfinite(pat_sim) and pat_sim > thresholds.pattern_similarity_min:
        scores[1] = 1
    scores[2] = _three_level_score(peakERD[0], thresholds.overt_erd_db)

    finite_nonreal = peakERD[1:][np.isfinite(peakERD[1:])]
    rel_nonreal = float(np.max(finite_nonreal)) if finite_nonreal.size else np.nan
    if np.isfinite(peakERD[0]) and peakERD[0] > 0 and np.isfinite(rel_nonreal):
        scores[3] = _three_level_score(
            rel_nonreal / peakERD[0], thresholds.nonreal_overt_fraction)

    sustained_erd_db, temporal_coverage = _sustained_erd_diagnostics(
        tf_norm, freqNeeded, tvec,
        thresholds.frequency_window_hz, thresholds.time_window_s)
    scores[4] = _three_level_score(
        sustained_erd_db["quasi"], thresholds.sustained_erd_db)
    scores[5] = _three_level_score(
        sustained_erd_db["imagery"], thresholds.sustained_erd_db)

    # Retain the old imagery-minus-quasi quantities as diagnostics only.
    peak_iq_diff_db = float(peakERD[2] - peakERD[1])
    iq_diff_db = float(
        sustained_erd_db["imagery"] - sustained_erd_db["quasi"])
    epsilon = thresholds.ratio_denominator_epsilon
    rel_nonreal_pct = _safe_display_ratio(rel_nonreal, peakERD[0], epsilon)
    rel_iq_pct = _safe_display_ratio(peakERD[2], peakERD[1], epsilon)

    support = {} if sustained_erd_support is None else sustained_erd_support
    if not isinstance(support, dict):
        raise TypeError("sustained_erd_support must be a dictionary or None")
    return ScoreResult(
        scores=scores,
        pat_sim=pat_sim,
        peakERD=peakERD,
        n_erd=n_erd_clean,
        rel_nonreal_pct=rel_nonreal_pct,
        rel_iq_pct=rel_iq_pct,
        iq_diff_mean_pct=np.nan,
        iq_diff_db=iq_diff_db,
        peak_iq_diff_db=peak_iq_diff_db,
        sustained_erd_db=sustained_erd_db,
        sustained_erd_temporal_coverage=temporal_coverage,
        sustained_erd_frequency_window_hz=thresholds.frequency_window_hz,
        sustained_erd_time_window_s=thresholds.time_window_s,
        sustained_erd_ci_db=dict(support.get("ci_db", {})),
        sustained_erd_uncertainty=dict(support.get("uncertainty", {})),
        sustained_erd_run_consistency=dict(
            support.get("run_consistency", {})),
        scoring_version=SCORING_V3,
        threshold_profile=thresholds.profile,
        thresholds_calibrated=thresholds.calibrated,
        data_path_version=data_path_version,
        normalization_mode=normalization_mode,
        pattern_match_mode=match_mode,
        pattern_match_n=match_n,
    )


def compute_scores_v4(
    tf_norm,
    freqNeeded,
    tvec,
    pattern,
    ideal_pattern,
    n_erd,
    *,
    thresholds: ScoringV4Thresholds | None = None,
    pattern_labels: Sequence[str] | None = None,
    ideal_pattern_labels: Sequence[str] | None = None,
    data_path_version: str | None = None,
    normalization_mode: str | None = None,
    sustained_erd_support: dict | None = None,
) -> ScoreResult:
    """Score independent non-real ERD with a continuous 500 ms peak.

    Criteria 1, 2, 5 and 6 retain v3's definitions.  Criteria 3 and 4 use the
    strongest continuous rolling-median ERD rather than the historical extreme
    percentile over flattened time-frequency cells.
    """
    thresholds = DEFAULT_V4_THRESHOLDS if thresholds is None else thresholds
    if not isinstance(thresholds, ScoringV4Thresholds):
        raise TypeError("thresholds must be a ScoringV4Thresholds instance")

    tf_norm, freqNeeded, tvec = _validated_tf(tf_norm, freqNeeded, tvec)
    scores = np.zeros(6)
    pat_sim, match_mode, match_n = _safe_pattern_similarity(
        pattern,
        ideal_pattern,
        pattern_labels=pattern_labels,
        ideal_pattern_labels=ideal_pattern_labels,
    )
    peakERD, peak_bounds = _rolling_median_peak_erd(
        tf_norm,
        freqNeeded,
        tvec,
        normalization_mode=normalization_mode,
        frequency_window_hz=thresholds.peak_erd_frequency_window_hz,
        rolling_window_s=thresholds.peak_erd_rolling_window_s,
    )

    try:
        n_erd_value = float(n_erd)
    except (TypeError, ValueError):
        n_erd_value = np.nan
    n_erd_clean = max(0, int(n_erd_value)) if np.isfinite(n_erd_value) else 0

    if n_erd_clean > 2:
        scores[0] = 2
    elif n_erd_clean >= 1:
        scores[0] = 1
    if np.isfinite(pat_sim) and pat_sim > thresholds.pattern_similarity_min:
        scores[1] = 1
    scores[2] = _three_level_score(peakERD[0], thresholds.overt_erd_db)

    finite_nonreal = peakERD[1:][np.isfinite(peakERD[1:])]
    rel_nonreal = float(np.max(finite_nonreal)) if finite_nonreal.size else np.nan
    if np.isfinite(peakERD[0]) and peakERD[0] > 0 and np.isfinite(rel_nonreal):
        scores[3] = _three_level_score(
            rel_nonreal / peakERD[0], thresholds.nonreal_overt_fraction)

    sustained_erd_db, temporal_coverage = _sustained_erd_diagnostics(
        tf_norm, freqNeeded, tvec,
        thresholds.frequency_window_hz, thresholds.time_window_s)
    scores[4] = _three_level_score(
        sustained_erd_db["quasi"], thresholds.sustained_erd_db)
    scores[5] = _three_level_score(
        sustained_erd_db["imagery"], thresholds.sustained_erd_db)

    peak_iq_diff_db = float(peakERD[2] - peakERD[1])
    iq_diff_db = float(
        sustained_erd_db["imagery"] - sustained_erd_db["quasi"])
    epsilon = thresholds.ratio_denominator_epsilon
    rel_nonreal_pct = _safe_display_ratio(rel_nonreal, peakERD[0], epsilon)
    rel_iq_pct = _safe_display_ratio(peakERD[2], peakERD[1], epsilon)

    support = {} if sustained_erd_support is None else sustained_erd_support
    if not isinstance(support, dict):
        raise TypeError("sustained_erd_support must be a dictionary or None")
    return ScoreResult(
        scores=scores,
        pat_sim=pat_sim,
        peakERD=peakERD,
        peak_erd_method="strongest-continuous-rolling-median",
        peak_erd_frequency_window_hz=thresholds.peak_erd_frequency_window_hz,
        peak_erd_rolling_window_s=thresholds.peak_erd_rolling_window_s,
        peak_erd_strongest_window_s=peak_bounds,
        n_erd=n_erd_clean,
        rel_nonreal_pct=rel_nonreal_pct,
        rel_iq_pct=rel_iq_pct,
        iq_diff_mean_pct=np.nan,
        iq_diff_db=iq_diff_db,
        peak_iq_diff_db=peak_iq_diff_db,
        sustained_erd_db=sustained_erd_db,
        sustained_erd_temporal_coverage=temporal_coverage,
        sustained_erd_frequency_window_hz=thresholds.frequency_window_hz,
        sustained_erd_time_window_s=thresholds.time_window_s,
        sustained_erd_ci_db=dict(support.get("ci_db", {})),
        sustained_erd_uncertainty=dict(support.get("uncertainty", {})),
        sustained_erd_run_consistency=dict(
            support.get("run_consistency", {})),
        scoring_version=SCORING_V4,
        threshold_profile=thresholds.profile,
        thresholds_calibrated=thresholds.calibrated,
        data_path_version=data_path_version,
        normalization_mode=normalization_mode,
        pattern_match_mode=match_mode,
        pattern_match_n=match_n,
    )


_VERSION_ALIASES = {
    SCORING_LEGACY: SCORING_LEGACY,
    "legacy": SCORING_LEGACY,
    "v1": SCORING_LEGACY,
    SCORING_V2: SCORING_V2,
    "v2": SCORING_V2,
    SCORING_V3: SCORING_V3,
    "v3": SCORING_V3,
    SCORING_V4: SCORING_V4,
    "v4": SCORING_V4,
}


def compute_scores(
    tf_norm,
    freqNeeded,
    tvec,
    pattern,
    ideal_pattern,
    n_erd,
    *,
    version: str = SCORING_LEGACY,
    thresholds: (
        ScoringV2Thresholds | ScoringV3Thresholds | ScoringV4Thresholds | None
    ) = None,
    pattern_labels: Sequence[str] | None = None,
    ideal_pattern_labels: Sequence[str] | None = None,
    data_path_version: str | None = None,
    normalization_mode: str | None = None,
    sustained_erd_support: dict | None = None,
) -> ScoreResult:
    """Compatibility dispatcher for explicit versioned scoring.

    The default remains ``legacy-v1`` so third-party callers that have not yet
    chosen a version keep their archived behavior.  New pipeline code should
    pass a version explicitly; aliases ``legacy`` and ``v1`` through ``v4``
    are accepted at interactive boundaries.
    """
    try:
        canonical_version = _VERSION_ALIASES[str(version).strip().casefold()]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_SCORING_VERSIONS)
        raise ValueError(f"unknown scoring version {version!r}; choose {supported}") from exc

    if canonical_version == SCORING_LEGACY:
        if thresholds is not None:
            raise ValueError("custom thresholds cannot be applied to legacy-v1")
        # Label arguments are intentionally ignored: legacy-v1 is defined by
        # positional, sign-sensitive matching and must remain reproducible.
        return compute_scores_legacy(
            tf_norm,
            freqNeeded,
            tvec,
            pattern,
            ideal_pattern,
            n_erd,
            data_path_version=data_path_version,
            normalization_mode=normalization_mode,
        )

    if canonical_version == SCORING_V2:
        return compute_scores_v2(
            tf_norm,
            freqNeeded,
            tvec,
            pattern,
            ideal_pattern,
            n_erd,
            thresholds=thresholds,
            pattern_labels=pattern_labels,
            ideal_pattern_labels=ideal_pattern_labels,
            data_path_version=data_path_version,
            normalization_mode=normalization_mode,
        )

    if canonical_version == SCORING_V3:
        return compute_scores_v3(
            tf_norm,
            freqNeeded,
            tvec,
            pattern,
            ideal_pattern,
            n_erd,
            thresholds=thresholds,
            pattern_labels=pattern_labels,
            ideal_pattern_labels=ideal_pattern_labels,
            data_path_version=data_path_version,
            normalization_mode=normalization_mode,
            sustained_erd_support=sustained_erd_support,
        )

    return compute_scores_v4(
        tf_norm,
        freqNeeded,
        tvec,
        pattern,
        ideal_pattern,
        n_erd,
        thresholds=thresholds,
        pattern_labels=pattern_labels,
        ideal_pattern_labels=ideal_pattern_labels,
        data_path_version=data_path_version,
        normalization_mode=normalization_mode,
        sustained_erd_support=sustained_erd_support,
    )


__all__ = [
    "SCORING_LEGACY",
    "SCORING_V2",
    "SCORING_V3",
    "SCORING_V4",
    "SUPPORTED_SCORING_VERSIONS",
    "SCORES_MAX",
    "MARKER_NORMALIZATION_MODES",
    "ScoringV2Thresholds",
    "DEFAULT_V2_THRESHOLDS",
    "ScoringV3Thresholds",
    "DEFAULT_V3_THRESHOLDS",
    "ScoringV4Thresholds",
    "DEFAULT_V4_THRESHOLDS",
    "ScoreResult",
    "pattern_similarity",
    "compute_scores_legacy",
    "compute_scores_v2",
    "compute_scores_v3",
    "compute_scores_v4",
    "compute_scores",
]
