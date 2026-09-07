"""Physiological sensorimotor quality-control helpers.

The historical KLH score asks whether an overt-derived CSP component transfers
to quasi-movement and imagery.  This module answers a complementary question:
does each condition contain a plausible sensor-space mu/beta response with the
expected contralateral distribution?

No value produced here contributes to the historical score.  The reference,
time windows, frequency bands and retained trial counts are carried with every
result so the QC cannot be mistaken for a cohort-compatible KLH score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


DEFAULT_BANDS = {"mu": (8.0, 13.0), "beta": (15.0, 25.0)}
QC_VERSION = "sensor-space-fixed-band-v2-per-recording"
LEFT_MOTOR_ROI = ("C3", "C1", "C5", "FC3", "CP3")
RIGHT_MOTOR_ROI = ("C4", "C2", "C6", "FC4", "CP4")
CENTRAL_MOTOR_ROI = LEFT_MOTOR_ROI + RIGHT_MOTOR_ROI
PERIPHERAL_ROI = ("Fp1", "Fp2", "Fpz", "AF7", "AF8", "FT9", "FT10",
                  "TP9", "TP10", "T7", "T8")


@dataclass
class BandConditionQC:
    """One condition/band sensor-space active-versus-rest contrast."""

    condition: str
    band: str
    band_hz: tuple[float, float]
    channel_db: np.ndarray
    channel_erd_pct: np.ndarray
    c3_db: float = np.nan
    c4_db: float = np.nan
    contralateral_db: float = np.nan
    ipsilateral_db: float = np.nan
    laterality_db: float = np.nan
    active_trials: int = 0
    rest_trials: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["channel_db"] = self.channel_db.tolist()
        result["channel_erd_pct"] = self.channel_erd_pct.tolist()
        return result


@dataclass
class ComponentPlausibility:
    """Transparent advisory ranking for a candidate CSP pattern."""

    component: int
    central_energy_fraction: float
    peripheral_energy_fraction: float
    ideal_similarity_abs: float
    plausibility: float
    energy_reference: str = "spatial-mean"


@dataclass
class PhysiologicalQCResult:
    """Sensor-space QC kept separate from the overt-template score."""

    active_condition: str
    reference: str
    active_window_s: tuple[float, float]
    rest_window_s: tuple[float, float]
    channel_labels: list[str]
    results: dict[str, dict[str, BandConditionQC]] = field(default_factory=dict)
    per_recording: dict[str, dict[str, dict[str, BandConditionQC]]] = field(
        default_factory=dict)
    recording_status: dict[str, dict[str, dict]] = field(default_factory=dict)
    consistency: dict[str, dict[str, dict]] = field(default_factory=dict)
    consistency_definition: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    qc_version: str = QC_VERSION

    def to_dict(self) -> dict:
        return {
            "active_condition": self.active_condition,
            "qc_version": self.qc_version,
            "reference": self.reference,
            "active_window_s": list(self.active_window_s),
            "rest_window_s": list(self.rest_window_s),
            "channel_labels": self.channel_labels,
            "warnings": self.warnings,
            "results": {
                condition: {band: value.to_dict() for band, value in bands.items()}
                for condition, bands in self.results.items()
            },
            "per_recording": {
                condition: {
                    recording: {
                        band: value.to_dict() for band, value in band_values.items()
                    }
                    for recording, band_values in recordings.items()
                }
                for condition, recordings in self.per_recording.items()
            },
            "recording_status": self.recording_status,
            "consistency": self.consistency,
            "consistency_definition": self.consistency_definition,
        }


def _normalise_label(label: str) -> str:
    return "".join(ch for ch in str(label).casefold() if ch.isalnum())


def _label_indices(labels: Sequence[str], wanted: Sequence[str]) -> list[int]:
    lut = {_normalise_label(label): i for i, label in enumerate(labels)}
    return [lut[key] for key in map(_normalise_label, wanted) if key in lut]


def _motor_roi_indices(
    labels: Sequence[str], active_condition: str
) -> tuple[list[int], list[int]]:
    """Return ``(contralateral, ipsilateral)`` motor-ROI indices for the hand.

    Both lists are empty when the moving hand cannot be read off
    ``active_condition``; either can be empty when the montage does not carry
    that ROI.
    """

    hand = str(active_condition).casefold()
    if "right" in hand:
        contra_names, ipsi_names = LEFT_MOTOR_ROI, RIGHT_MOTOR_ROI
    elif "left" in hand:
        contra_names, ipsi_names = RIGHT_MOTOR_ROI, LEFT_MOTOR_ROI
    else:
        return [], []
    return _label_indices(labels, contra_names), _label_indices(labels, ipsi_names)


def _window_indices(tvec: np.ndarray, window: tuple[float, float], name: str) -> np.ndarray:
    tvec = np.asarray(tvec, dtype=float).ravel()
    if tvec.size == 0:
        raise ValueError(f"{name} time vector is empty")
    lo, hi = map(float, window)
    if lo > hi:
        raise ValueError(f"{name} window start exceeds its end: {window}")
    if lo < tvec[0] - 1e-9 or hi > tvec[-1] + 1e-9:
        raise ValueError(
            f"{name} window {lo:g}..{hi:g}s is outside available "
            f"{tvec[0]:g}..{tvec[-1]:g}s")
    idx = np.flatnonzero((tvec >= lo - 1e-12) & (tvec <= hi + 1e-12))
    if idx.size < 2:
        raise ValueError(f"{name} window contains fewer than two samples")
    return idx


def _band_trial_power(
    epochs: np.ndarray,
    fs: float,
    bands: Mapping[str, tuple[float, float]],
    *,
    common_average: bool,
) -> dict[str, np.ndarray]:
    """Return band power as ``{band: (trials, channels)}``."""

    x = np.asarray(epochs, dtype=float)
    if x.ndim != 3:
        raise ValueError("epochs must have shape (time, trials, channels)")
    if x.shape[0] < 2 or x.shape[1] == 0 or x.shape[2] == 0:
        raise ValueError(f"epochs have unusable shape {x.shape}")
    if common_average and x.shape[2] > 1:
        x = x - np.mean(x, axis=2, keepdims=True)

    # One-second Welch windows retain useful frequency resolution while keeping
    # results stable for the 2-3 second active windows used by KLH.
    nperseg = min(x.shape[0], max(8, int(round(float(fs)))))
    noverlap = nperseg // 2 if x.shape[0] > nperseg else 0
    freq, psd = welch(
        x, fs=float(fs), axis=0, nperseg=nperseg, noverlap=noverlap,
        detrend="constant", scaling="density")
    result: dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        mask = (freq >= lo) & (freq <= hi)
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"band {name!r} ({lo:g}..{hi:g} Hz) has fewer than two "
                f"frequency bins at fs={fs:g}")
        result[name] = trapezoid(psd[mask], freq[mask], axis=0)
    return result


def _trial_slice(parent: np.ndarray, child: np.ndarray) -> slice | None:
    """Recognise an exact trial-axis view; unrelated arrays never reuse PSDs."""
    if (parent.ndim != 3 or child.ndim != 3 or not child.size
            or parent.dtype != child.dtype
            or parent.shape[::2] != child.shape[::2]
            or parent.strides != child.strides or parent.strides[1] <= 0
            or not np.shares_memory(parent, child)):
        return None
    offset = child.ctypes.data - parent.ctypes.data
    start, remainder = divmod(offset, parent.strides[1])
    stop = start + child.shape[1]
    if remainder or start < 0 or stop > parent.shape[1]:
        return None
    return slice(start, stop)


def _condition_contrast(
    condition: str,
    active: np.ndarray,
    rest: np.ndarray,
    *,
    ia: np.ndarray,
    ir: np.ndarray,
    fs: float,
    bands: Mapping[str, tuple[float, float]],
    common_average: bool,
    labels: Sequence[str],
    contra_idx: Sequence[int],
    ipsi_idx: Sequence[int],
    c3_idx: Sequence[int],
    c4_idx: Sequence[int],
    power_cache: list | None = None,
) -> dict[str, BandConditionQC]:
    active = np.asarray(active)
    rest = np.asarray(rest)
    if active.ndim != 3 or rest.ndim != 3:
        raise ValueError(f"{condition}: epochs must be three-dimensional")
    if active.shape[2] != len(labels) or rest.shape[2] != len(labels):
        raise ValueError(
            f"{condition}: epoch channel count does not match channel_labels")
    pa = pr = None
    # Within one QC call, fs, windows, bands and referencing are fixed. The
    # recording arrays from the pipeline are views of their condition arrays,
    # so their per-trial Welch powers are already available without another FFT.
    for (parent_active, parent_rest), (cached_a, cached_r) in (power_cache or []):
        a_slice = _trial_slice(parent_active, active)
        r_slice = _trial_slice(parent_rest, rest)
        if a_slice is not None and r_slice is not None:
            pa = {band: values[a_slice] for band, values in cached_a.items()}
            pr = {band: values[r_slice] for band, values in cached_r.items()}
            break
    if pa is None:
        pa = _band_trial_power(active[ia], fs, bands, common_average=common_average)
        pr = _band_trial_power(rest[ir], fs, bands, common_average=common_average)
        if power_cache is not None:
            power_cache.append(((active, rest), (pa, pr)))
    tiny = np.finfo(float).tiny
    output: dict[str, BandConditionQC] = {}
    for band, limits in bands.items():
        active_med = np.median(pa[band], axis=0)
        rest_med = np.median(pr[band], axis=0)
        ratio = np.maximum(active_med, tiny) / np.maximum(rest_med, tiny)
        db = 10.0 * np.log10(ratio)
        erd_pct = 100.0 * (1.0 - ratio)
        contra = float(np.median(db[list(contra_idx)])) if contra_idx else np.nan
        ipsi = float(np.median(db[list(ipsi_idx)])) if ipsi_idx else np.nan
        output[band] = BandConditionQC(
            condition=str(condition), band=band,
            band_hz=(float(limits[0]), float(limits[1])),
            channel_db=db, channel_erd_pct=erd_pct,
            c3_db=float(db[c3_idx[0]]) if c3_idx else np.nan,
            c4_db=float(db[c4_idx[0]]) if c4_idx else np.nan,
            contralateral_db=contra, ipsilateral_db=ipsi,
            laterality_db=contra - ipsi,
            active_trials=int(active.shape[1]), rest_trials=int(rest.shape[1]),
        )
    return output


def compute_sensorimotor_qc(
    condition_epochs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    fs: float,
    tvec_active: Sequence[float],
    tvec_rest: Sequence[float],
    channel_labels: Sequence[str],
    active_condition: str,
    active_window: tuple[float, float] = (0.5, 2.5),
    rest_window: tuple[float, float] = (1.0, 4.0),
    bands: Mapping[str, tuple[float, float]] | None = None,
    common_average: bool = True,
    recording_epochs: Mapping[
        str, Mapping[str, tuple[np.ndarray, np.ndarray]]] | None = None,
) -> PhysiologicalQCResult:
    """Compute condition-specific sensor-space mu/beta QC.

    ``condition_epochs`` maps display names (for example ``"overt"``) to
    ``(active_epochs, rest_epochs)`` arrays shaped ``time x trials x channels``.
    Each condition is normalised to its own rest epochs.  Negative dB denotes
    ERD; ``channel_erd_pct`` expresses the same effect as positive percent power
    reduction.
    """

    labels = list(map(str, channel_labels))
    if not labels:
        raise ValueError("channel labels are required for physiological QC")
    t_active = np.asarray(tvec_active, dtype=float)
    t_rest = np.asarray(tvec_rest, dtype=float)
    ia = _window_indices(t_active, active_window, "active")
    ir = _window_indices(t_rest, rest_window, "rest")
    bands = dict(DEFAULT_BANDS if bands is None else bands)

    contra_idx, ipsi_idx = _motor_roi_indices(labels, active_condition)
    c3_idx = _label_indices(labels, ("C3",))
    c4_idx = _label_indices(labels, ("C4",))
    warnings: list[str] = []
    if not contra_idx or not ipsi_idx:
        warnings.append("contralateral/ipsilateral motor ROIs are incomplete")

    output: dict[str, dict[str, BandConditionQC]] = {}
    power_cache = []
    for condition, (active, rest) in condition_epochs.items():
        output[str(condition)] = _condition_contrast(
            str(condition), active, rest, ia=ia, ir=ir, fs=fs, bands=bands,
            common_average=common_average, labels=labels,
            contra_idx=contra_idx, ipsi_idx=ipsi_idx,
            c3_idx=c3_idx, c4_idx=c4_idx, power_cache=power_cache)

    per_recording: dict[str, dict[str, dict[str, BandConditionQC]]] = {}
    recording_status: dict[str, dict[str, dict]] = {}
    consistency: dict[str, dict[str, dict]] = {}
    for condition, recordings in (recording_epochs or {}).items():
        condition = str(condition)
        per_recording[condition] = {}
        recording_status[condition] = {}
        for recording, (active, rest) in recordings.items():
            recording = str(recording)
            active = np.asarray(active)
            rest = np.asarray(rest)
            active_trials = int(active.shape[1]) if active.ndim == 3 else 0
            rest_trials = int(rest.shape[1]) if rest.ndim == 3 else 0
            status = {
                "available": bool(active_trials and rest_trials),
                "active_trials": active_trials,
                "rest_trials": rest_trials,
            }
            if not status["available"]:
                status["reason"] = "no retained active or rest trials"
                per_recording[condition][recording] = {}
                recording_status[condition][recording] = status
                warnings.append(
                    f"{condition} {recording}: per-recording QC unavailable "
                    f"({active_trials} active/{rest_trials} rest trials)")
                continue
            per_recording[condition][recording] = _condition_contrast(
                condition, active, rest, ia=ia, ir=ir, fs=fs,
                bands=bands, common_average=common_average, labels=labels,
                contra_idx=contra_idx, ipsi_idx=ipsi_idx,
                c3_idx=c3_idx, c4_idx=c4_idx, power_cache=power_cache)
            recording_status[condition][recording] = status
        consistency[condition] = {}
        for band in bands:
            values = np.asarray([
                result[band].laterality_db
                for result in per_recording[condition].values()
                if band in result and np.isfinite(result[band].laterality_db)
            ], dtype=float)
            total = len(recordings)
            summary = {
                "recordings": int(values.size),
                "total_recordings": int(total),
                "unavailable_recordings": int(total - values.size),
                "expected_laterality_count": int(np.count_nonzero(values < 0)),
                "expected_laterality_fraction": (
                    float(np.mean(values < 0)) if values.size else np.nan),
                "median_laterality_db": np.nan,
                "mad_laterality_db": np.nan,
                "range_laterality_db": [],
            }
            if values.size:
                centre = float(np.median(values))
                summary["median_laterality_db"] = centre
                summary["mad_laterality_db"] = float(
                    np.median(np.abs(values - centre)))
                summary["range_laterality_db"] = [
                    float(values.min()), float(values.max())]
                if values.size >= 2 and summary["expected_laterality_fraction"] < 0.75:
                    warnings.append(
                        f"{condition} {band} laterality is inconsistent across "
                        f"recordings ({summary['expected_laterality_count']}/"
                        f"{summary['recordings']} in the expected direction)")
            consistency[condition][band] = summary

    consistency_definition = {
        "unit": "recording",
        "included": "finite per-recording laterality estimates only",
        "laterality": "contralateral_db - ipsilateral_db",
        "expected_direction": "laterality_db < 0",
        "warning_rule": "at least 2 finite recordings and expected fraction < 0.75",
        "warning_fraction_threshold": 0.75,
    }
    return PhysiologicalQCResult(
        active_condition=str(active_condition),
        reference="common-average" if common_average else "as-recorded",
        active_window_s=tuple(map(float, active_window)),
        rest_window_s=tuple(map(float, rest_window)),
        channel_labels=labels, results=output, per_recording=per_recording,
        recording_status=recording_status, consistency=consistency,
        consistency_definition=consistency_definition, warnings=warnings)


def _center_component_patterns(patterns: np.ndarray) -> np.ndarray:
    """Remove the spatial offset without altering the projection patterns.

    Subtracting one channel first makes constant patterns exactly zero even
    when their original value is not exactly representable in floating point.
    Non-finite columns remain unavailable to the selection rules.
    """
    with np.errstate(invalid="ignore"):
        centered = patterns - patterns[:1]
        return centered - np.mean(centered, axis=0, keepdims=True)


def rank_component_patterns(
    patterns: np.ndarray,
    channel_labels: Sequence[str],
    *,
    ideal_pattern: np.ndarray | None = None,
    ideal_labels: Sequence[str] | None = None,
) -> list[ComponentPlausibility]:
    """Rank CSP patterns by transparent motor-topography heuristics.

    It combines spatially centered central energy, a peripheral-energy penalty
    and (when available) sign-invariant similarity with the historical ideal
    pattern. The spatial mean is removed before measuring pattern energy, so
    an additive reference offset cannot determine the localization ranking.
    The ranking never scores a recording; :func:`select_motor_component` uses it
    as one ingredient of the first-pass selection offered on the slider, which
    the operator is always free to override.
    """

    p = np.asarray(patterns, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError("patterns must contain at least one channel")
    labels = list(map(str, channel_labels))
    if p.shape[0] != len(labels):
        raise ValueError("pattern rows must match channel_labels")
    central = _label_indices(labels, CENTRAL_MOTOR_ROI)
    peripheral = _label_indices(labels, PERIPHERAL_ROI)
    centered = _center_component_patterns(p)

    ideal_by_label: dict[str, float] = {}
    if ideal_pattern is not None:
        ip = np.asarray(ideal_pattern, dtype=float).ravel()
        il = labels if ideal_labels is None else list(map(str, ideal_labels))
        if len(il) != ip.size:
            raise ValueError("ideal_pattern and ideal_labels differ in length")
        ideal_by_label = {_normalise_label(label): float(value)
                          for label, value in zip(il, ip)}

    rankings: list[ComponentPlausibility] = []
    for ci in range(p.shape[1]):
        values = centered[:, ci]
        energy = values * values
        total = float(np.sum(energy))
        central_fraction = float(np.sum(energy[central]) / total) if total > 0 else 0.0
        peripheral_fraction = float(np.sum(energy[peripheral]) / total) if total > 0 else 0.0
        similarity = np.nan
        if ideal_by_label:
            pairs = [(values[i], ideal_by_label[_normalise_label(label)])
                     for i, label in enumerate(labels)
                     if _normalise_label(label) in ideal_by_label]
            if len(pairs) >= 2:
                a, b = np.asarray(pairs, dtype=float).T
                if np.std(a) > 0 and np.std(b) > 0:
                    similarity = float(abs(np.corrcoef(a, b)[0, 1]))
        sim_for_rank = 0.0 if not np.isfinite(similarity) else similarity
        plausibility = central_fraction - peripheral_fraction + 0.5 * sim_for_rank
        rankings.append(ComponentPlausibility(
            component=ci + 1,
            central_energy_fraction=central_fraction,
            peripheral_energy_fraction=peripheral_fraction,
            ideal_similarity_abs=similarity,
            plausibility=float(plausibility)))
    return sorted(rankings, key=lambda value: value.plausibility, reverse=True)


# ---------------------------------------------------------------------------
# First-pass component selection.
#
# When the primary analysis finishes, two pieces of evidence exist for each of
# the inspectable patterns.  The bounded CSP eigenvalue says whether the
# component's share of mu power fell (ERD) or rose (ERS) between rest and the
# overt active window, and the forward pattern says where on the scalp the
# component sits.  Together they are enough to preselect the component an
# operator would normally look for first -- contralateral mu-ERD -- without
# claiming any of the accuracy of the time-frequency analysis that follows.
# ---------------------------------------------------------------------------

# The pencil eig(R1, R1 + R2) is solved on trace-normalised covariances, so 0.5
# is the neutral point: a component whose mu power holds the same share of the
# total in the active window as at rest.  The margin keeps a component sitting
# on that point from being claimed as either direction.
NEUTRAL_EIGENVALUE = 0.5
DIRECTION_MARGIN = 0.02

# Pattern-energy laterality index, (contra - ipsi) / (contra + ipsi).  A tenth
# of the two ROIs' combined energy is the least imbalance still worth calling
# lateralised; below it the pattern is treated as bilateral and matches no
# lateralised category.
LATERALITY_MARGIN = 0.10

# Weight of the laterality index when ordering components *within* one
# category.  It matches the weight the plausibility formula already gives to
# ideal-pattern similarity, so no single ingredient dominates the order.
LATERALITY_WEIGHT = 0.5

# Preference order requested for the first-pass selection.  Names are stable;
# they appear in the report and in saved results.
SELECTION_PRIORITY = ("contralateral-mu-erd", "mu-ers", "ipsilateral-mu-erd")
SELECTION_LABELS = {
    "contralateral-mu-erd": "contralateral mu-ERD",
    "mu-ers": "mu-ERS",
    "ipsilateral-mu-erd": "ipsilateral mu-ERD",
}
DIRECTION_LABELS = {
    "mu-erd": "mu-ERD",
    "mu-ers": "mu-ERS",
    "indeterminate": "no clear mu change",
}
SELECTION_FALLBACK_MESSAGE = (
    "No qualifying contralateral mu-ERD candidate was found. "
    "None of the inspectable components could be recognised as a "
    "contralateral mu-ERD, a mu-ERS or an ipsilateral mu-ERD, so component 1 "
    "was preselected as a starting point only.\n\n"
    "Automatic selection can fail: it sees only the CSP eigenvalue and the "
    "scalp pattern, not the time-frequency response. Stepping through the "
    "topographies and choosing the component by hand is still worth doing."
)


@dataclass
class ComponentResponse:
    """What the eigenvalue and the pattern say about one CSP component."""

    component: int
    eigenvalue: float
    direction: str                       # "mu-erd", "mu-ers" or "indeterminate"
    laterality_index: float              # +1 fully contralateral, -1 ipsilateral
    contralateral_energy_fraction: float
    ipsilateral_energy_fraction: float
    central_energy_fraction: float
    peripheral_energy_fraction: float
    plausibility: float
    motor_plausible: bool
    category: str | None = None
    selection_score: float = np.nan


@dataclass
class ComponentSelection:
    """The component preselected on the slider, and why."""

    component: int
    category: str | None
    automatic: bool
    summary: str
    message: str = ""
    responses: list[ComponentResponse] = field(default_factory=list)
    criteria_version: str = "centered-pattern-soft-peripheral-v2"


def _mu_direction(eigenvalue: float) -> str:
    if not np.isfinite(eigenvalue):
        return "indeterminate"
    if eigenvalue <= NEUTRAL_EIGENVALUE - DIRECTION_MARGIN:
        return "mu-erd"
    if eigenvalue >= NEUTRAL_EIGENVALUE + DIRECTION_MARGIN:
        return "mu-ers"
    return "indeterminate"


def _roi_energy_fraction(energy: np.ndarray, total: float, idx: Sequence[int]) -> float:
    """Share of a pattern's energy inside one ROI, NaN when unmeasurable."""
    if not idx or not np.isfinite(total) or total <= 0:
        return np.nan
    return float(np.sum(energy[list(idx)]) / total)


def select_motor_component(
    patterns: np.ndarray,
    channel_labels: Sequence[str],
    evals: Sequence[float],
    *,
    active_condition: str,
    ideal_pattern: np.ndarray | None = None,
    ideal_labels: Sequence[str] | None = None,
) -> ComponentSelection:
    """Preselect the component an operator would normally inspect first.

    ``patterns`` are the inspectable forward patterns (channels x components)
    and ``evals`` the CSP eigenvalues in the same component order.  Components
    are sorted into three categories, in the order they are preferred:

    ``contralateral-mu-erd``
        mu power drops in the active window and the pattern sits over the
        motor ROI opposite the moving hand.
    ``mu-ers``
        mu power rises in the active window over a motor pattern.  Laterality
        is not required here, but a contralateral pattern is preferred.
    ``ipsilateral-mu-erd``
        mu power drops over the motor ROI on the same side as the moving hand.

    Energy and laterality are measured after subtracting each pattern's
    spatial mean. Every category requires the central ROI to carry at least
    its own share of the montage's centered energy. Peripheral energy lowers
    the ranking score but cannot veto a centrally concentrated pattern.
    A selected contralateral component is a candidate for time-frequency
    inspection; ERS and ipsilateral fallbacks do not establish the target
    response. When no component matches, component 1 is returned with
    ``automatic=False`` and a message asking for manual inspection.
    """

    p = np.asarray(patterns, dtype=float)
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("at least one component pattern is required")
    labels = list(map(str, channel_labels))
    if p.shape[0] != len(labels):
        raise ValueError("pattern rows must match channel_labels")
    spectrum = np.asarray(evals, dtype=float).ravel()
    if spectrum.size < p.shape[1]:
        raise ValueError("evals must cover every supplied component pattern")

    ranking = {item.component: item for item in rank_component_patterns(
        p, labels, ideal_pattern=ideal_pattern, ideal_labels=ideal_labels)}
    centered = _center_component_patterns(p)
    contra_idx, ipsi_idx = _motor_roi_indices(labels, active_condition)
    # A pattern that spreads its energy evenly over the montage gives the
    # central ROI exactly its share of the channels, so that share is the floor
    # below which "central" means nothing.
    central_share = len(_label_indices(labels, CENTRAL_MOTOR_ROI)) / len(labels)

    responses: list[ComponentResponse] = []
    for ci in range(p.shape[1]):
        item = ranking[ci + 1]
        energy = centered[:, ci] ** 2
        total = float(np.sum(energy))
        contra = _roi_energy_fraction(energy, total, contra_idx)
        ipsi = _roi_energy_fraction(energy, total, ipsi_idx)
        laterality = np.nan
        if np.isfinite(contra) and np.isfinite(ipsi) and (contra + ipsi) > 0:
            laterality = float((contra - ipsi) / (contra + ipsi))
        plausible = bool(
            central_share > 0 and np.isfinite(total) and total > 0
            and item.central_energy_fraction >= central_share)

        direction = _mu_direction(spectrum[ci])
        category = None
        if plausible:
            # NaN laterality compares false against both margins, so a montage
            # without the motor ROIs can only ever match the unlateralised
            # mu-ERS category.
            if direction == "mu-erd" and laterality >= LATERALITY_MARGIN:
                category = "contralateral-mu-erd"
            elif direction == "mu-ers":
                category = "mu-ers"
            elif direction == "mu-erd" and laterality <= -LATERALITY_MARGIN:
                category = "ipsilateral-mu-erd"
        toward = -1.0 if category == "ipsilateral-mu-erd" else 1.0
        score = item.plausibility + LATERALITY_WEIGHT * toward * (
            0.0 if not np.isfinite(laterality) else laterality)

        responses.append(ComponentResponse(
            component=ci + 1,
            eigenvalue=float(spectrum[ci]),
            direction=direction,
            laterality_index=laterality,
            contralateral_energy_fraction=contra,
            ipsilateral_energy_fraction=ipsi,
            central_energy_fraction=item.central_energy_fraction,
            peripheral_energy_fraction=item.peripheral_energy_fraction,
            plausibility=item.plausibility,
            motor_plausible=plausible,
            category=category,
            selection_score=float(score)))

    for category in SELECTION_PRIORITY:
        tier = [r for r in responses if r.category == category]
        if not tier:
            continue
        # Best score wins; an exact tie goes to the lower component number,
        # which is the stronger end of the eigenvalue spectrum.
        chosen = max(tier, key=lambda r: (r.selection_score, -r.component))
        if category == "contralateral-mu-erd":
            summary = (
                f"selected component {chosen.component} automatically: "
                "contralateral mu-ERD candidate; confirm mu suppression "
                "in the time-frequency response.")
        else:
            summary = (
                "no qualifying contralateral mu-ERD candidate; "
                f"component {chosen.component} preselected for inspection: "
                f"{SELECTION_LABELS[category]}.")
        return ComponentSelection(
            component=chosen.component, category=category, automatic=True,
            summary=summary,
            responses=responses)

    return ComponentSelection(
        component=1, category=None, automatic=False,
        summary=("no qualifying contralateral mu-ERD candidate; "
                 "component 1 preselected for manual inspection"),
        message=SELECTION_FALLBACK_MESSAGE, responses=responses)


def describe_component_response(response: ComponentResponse) -> str:
    """One compact, unit-carrying description of a component's evidence."""
    laterality = ("laterality n/a" if not np.isfinite(response.laterality_index)
                  else f"laterality {response.laterality_index:+.2f}")
    return (f"eigenvalue {response.eigenvalue:.3f}, {laterality}, "
            f"centered central energy {response.central_energy_fraction:.2f}, "
            f"peripheral energy {response.peripheral_energy_fraction:.2f}")


def _laterality_word(index: float) -> str | None:
    """Name the side a pattern favours, or None when there are no ROIs."""
    if not np.isfinite(index):
        return None
    if index >= LATERALITY_MARGIN:
        return "contralateral"
    if index <= -LATERALITY_MARGIN:
        return "ipsilateral"
    return "bilateral"


def summarize_component_responses(
    responses: Sequence[ComponentResponse]) -> str:
    """What each component was found to be, for the analysis report."""
    parts = []
    for response in responses:
        words = [w for w in (_laterality_word(response.laterality_index),
                             DIRECTION_LABELS.get(response.direction,
                                                  response.direction)) if w]
        if not response.motor_plausible:
            words.append("(non-motor pattern)")
        parts.append(f"#{response.component} {' '.join(words)} "
                     f"[{describe_component_response(response)}]")
    return "; ".join(parts)


def detect_persistent_channel_outliers(
    epoch_groups: Sequence[np.ndarray],
    channel_labels: Sequence[str],
    *,
    z_threshold: float = 5.0,
) -> dict:
    """Report channels with persistent extreme variance across epoch groups."""

    labels = list(map(str, channel_labels))
    per_group = []
    tiny = np.finfo(float).tiny
    for epochs in epoch_groups:
        x = np.asarray(epochs, dtype=float)
        if x.ndim != 3 or x.shape[2] != len(labels) or x.shape[1] == 0:
            continue
        variance = np.median(np.var(x, axis=0), axis=0)
        log_variance = np.log(np.maximum(variance, tiny))
        centre = np.median(log_variance)
        scale = 1.4826 * np.median(np.abs(log_variance - centre))
        z = np.zeros_like(log_variance) if scale <= tiny else (log_variance - centre) / scale
        per_group.append(z)
    if not per_group:
        return {"labels": labels, "robust_z": [], "flagged": [], "groups": 0}
    robust_z = np.median(np.abs(np.asarray(per_group)), axis=0)
    flagged = [labels[i] for i in np.flatnonzero(robust_z > float(z_threshold))]
    return {
        "labels": labels,
        "robust_z": robust_z.tolist(),
        "flagged": flagged,
        "groups": len(per_group),
        "threshold": float(z_threshold),
    }
