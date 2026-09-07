"""Kitties' Little Helper — Quasi-fMRI EEG and hand-EMG desktop application.

Run:  python app.py

Tabs:
  Start          -> participant, recordings, EMG electrodes + START
  EEG Components -> eigenvalue spectrum, top-5 CSP topographies (slider select),
                    three condition TF maps, Calculate FT
  EEG Score      -> six sub-score gauges + final cat image
  EMG Analysis   -> mean left/right hand envelopes per recording, the share of
                    trials carrying high EMG in each hand, and a trial browser

The EEG half is the former KITTIES' LIL HELPER (:mod:`klh`); the EMG half is
the former EMGcasting (:mod:`emgcasting`). Both read the same recordings, with
the same participant details and the same marker-time correction, so the start
tab describes a run once and START performs both analyses on it. The EMG half
runs first because it is much the faster of the two, so its verdict is on
screen while the CSP fit is still going; each half raises its own tab as it
finishes. They share no results: either half can fail without taking the
other's away.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import traceback
import matplotlib
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from klh.pipeline import KLHPipeline, ExtractConfig, DEFAULT_CSP_CV_FOLDS
from klh import qc, superlet, topo, execution
from klh.assets import asset_path
from klh.scoring import (DEFAULT_V2_THRESHOLDS, DEFAULT_V3_THRESHOLDS,
                         DEFAULT_V4_THRESHOLDS, SCORING_V2, SCORING_V3,
                         SCORING_V4)
from emgcasting.core import (DEFAULT_OUTPUT_ROOT as EMG_OUTPUT_ROOT,
                             ProcessingConfig as EMGConfig, parse_pair)
from emg_view import EMGTab
from shared import theme
from shared.runtime import application_dir
from shared.recordings import discover_recordings, participant_code

PALETTE = theme.PALETTE

HAXBY = None  # lazy colormap

# The single place the study's event names are written down. The EEG half
# analyses one hand at a time; the EMG half always analyses both, and takes its
# rest reference from the same marker the EEG extraction uses.
LEFT_CONDITION = "left_microrepeat"
RIGHT_CONDITION = "right_microrepeat"
REST_CONDITION = getattr(ExtractConfig, "rest_cond", "rest")
MOVEMENT_CONDITIONS = (
    ("Right hand", RIGHT_CONDITION),
    ("Left hand", LEFT_CONDITION),
)
EMG_LEFT_CHANNELS_DEFAULT = "Aux 1.1, Aux 1.2"
EMG_RIGHT_CHANNELS_DEFAULT = "Aux 2.1, Aux 2.2"

FILE_TYPES = (("H5", "h5"), ("XDF", "xdf"))
EXTRACT_MODES = (("Trigger channel", "trigger"),
                 ("Embedded XDF markers", "markers"))
DATA_FOLDER_DEFAULT = r"D:\ExpData\MEG\Quasi fMRI\data\SUBJECT_NAME"
_EXTENSIONS = {"h5": ".h5", "xdf": ".xdf"}
MIN_H5_CHANNELS = 2
MAX_REFERENCE_CHANNELS = 64
MAX_DISPLAY_COMPONENTS = 5

# Widths for the start form's fields. A QLineEdit in a form layout grows to
# whatever width is going, which on a 1200-pixel window turns a five-character
# subject code into a banner; these cap each field near the value it holds.
FIELD_WIDTH_PATH = 620
FIELD_WIDTH_LIST = 440
FIELD_WIDTH_SHORT = 240
# The column the form and START share: the label gutter plus the widest field.
# The longest name on the form is "right hand EMG (+, −)", which at the form's
# own weight needs most of the difference between this and FIELD_WIDTH_PATH.
START_BODY_WIDTH = 800
# Air between that column and the edge of the card it sits on: left, top,
# right, bottom.
START_CARD_PADDING = (30, 16, 30, 16)
# The cat sits beside the title rather than above it, so its height is the
# height of the whole header rather than added to it.
INTRO_IMAGE_PX = 120
# Lines the start tab's status line is given, whatever it currently says. Two
# is what a finished run needs -- one line for each half -- and it stays below
# the height of the START button sharing the row, so the row never moves.
STATUS_LINE_COUNT = 2

# Top of the first row of score gauges, as a fraction of its figure. The score
# picture beside them is started at the same height, so the two line up.
SCORE_AXES_TOP = 0.95

# Pixel margins for the EEG Components plots. The eigenvalue panel and the
# three time-frequency panels are separate figures in the same grid columns,
# so equal pixel margins put their four boxes exactly under one another. The
# eigenvalue legend sits outside its axes; the width it takes is reserved on
# both figures, or the boxes would line up everywhere except where it is.
EEG_AXES_LEFT_PX = 62
EEG_LEGEND_PAD_PX = 12
EEG_LEGEND_FALLBACK_PX = 92

MARKER_BASELINES = (
    ("Condition-specific rest", "condition-specific"),
    ("Pooled rest", "pooled"),
)
SCORING_VERSIONS = (
    ("Scoring v4 (500 ms rolling peak + independent ERD; provisional)",
     SCORING_V4),
    ("Scoring v3 (independent sustained ERD; provisional)",
     SCORING_V3),
    ("Scoring v2 (corrected; provisional thresholds)", "scoring-v2"),
    ("Legacy v1 (historical formula)", "legacy-v1"),
)
RECORDING_RATES = (("1000 Hz", 1000), ("500 Hz", 500))

# The three time-frequency panels, in the order the pipeline returns them.
TF_CONDITION_NAMES = ("OVERT", "QUASI", "IMAGERY")

# ---------------------------------------------------------------------------
# Empty panels.
#
# A matplotlib axes with nothing plotted on it still advertises a range, and
# the default range is 0..1 on both scales.  The eigenvalue panel therefore
# opened claiming components 0.0 to 1.0, the time-frequency maps a one-second
# window over one hertz, and every score gauge a criterion running from 0 to 1
# -- numbers belonging to no recording and no score, drawn in the same type as
# the real ones and impossible to tell from them.  Until a panel has a result,
# it carries no scale at all: only a line naming what will fill it in.
# ---------------------------------------------------------------------------
PLACEHOLDER_TOPOS = "CSP scalp patterns\nappear when START\nhas finished"
PLACEHOLDER_EVALS = "eigenvalue spectrum\nappears when START has finished"
PLACEHOLDER_TF = "{name}\ntime–frequency map appears after “Calculate FT”"
PLACEHOLDER_GAUGE = "criterion {number}\nnot scored yet"
PLACEHOLDER_SCORE_IMAGE = "the cat arrives\nwith the score"
NO_SCORE_TEXT = "No score yet"

# The score formula is fixed for this build.  It is still written to the
# settings file so a result can be traced back to the formula that produced it,
# but a different value there is corrected on load rather than obeyed.
LOCKED_SCORING_VERSION = SCORING_V4
SETTINGS_FILE_NAME = "neurocasting_settings.json"

# The size the window opens at when the profile has no size of its own. The
# height is a request rather than a promise: Qt raises it to the tallest tab's
# own minimum, which is the start form, and that minimum moves a little with
# the metrics of the typeface a machine actually has.
WINDOW_SIZE = (1180, 760)
# Bumped whenever WINDOW_SIZE changes, or whenever the layout stops producing
# the heights already saved. The profile remembers the size the window was
# last closed at, which is what an operator wants -- but it also means a new
# default would never be seen by anyone who has already run the application
# once, so a geometry saved under an older number is dropped.
GEOMETRY_VERSION = 3


def _split_files(text, folder, ext=".h5"):
    names = [s.strip() for s in text.split(",") if s.strip()]
    return [os.path.join(folder, n + ext) for n in names]


def _supports_parameter(callable_obj, name):
    """Return whether *callable_obj* explicitly accepts ``name``.

    The GUI is intentionally tolerant of an older ``klh`` package: controls
    for a new option are disabled instead of passing an unknown keyword and
    crashing the worker thread.
    """
    try:
        params = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind == inspect.Parameter.VAR_KEYWORD
               for p in params)


def _make_extract_config(**values):
    """Construct ``ExtractConfig`` with only fields supported by this build."""
    supported = {
        key: value for key, value in values.items()
        if _supports_parameter(ExtractConfig, key)
    }
    return ExtractConfig(**supported)


def _component_count(result):
    """Number of components that can safely be displayed and selected."""
    candidates = []
    declared = getattr(result, "n_components", None)
    if declared is not None:
        try:
            candidates.append(int(declared))
        except (TypeError, ValueError):
            pass
    evals = np.asarray(getattr(result, "evals", []))
    if evals.ndim:
        candidates.append(int(evals.size))
    patterns = np.asarray(getattr(result, "top_patterns", []))
    if patterns.ndim == 2:
        candidates.append(int(patterns.shape[1]))
    else:
        candidates.append(0)
    available = [n for n in candidates if n >= 0]
    return min([MAX_DISPLAY_COMPONENTS, *available]) if available else 0


def _blank_axes(ax, text="", fontsize=9):
    """Empty an axes and take its scales away with the data.

    See the placeholder constants above: an axes drawn with no ticks, no
    spines and one grey line of text says "nothing has been computed here
    yet", where the same axes left alone says "computed, and the answer runs
    from zero to one".
    """
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Tinted rather than blank: an empty panel still has to hold its place in
    # the layout, or the tab reads as broken instead of as not yet run.
    ax.set_facecolor(PALETTE["page"])
    if text:
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center",
                va="center", fontsize=fontsize, color=PALETTE["faint"],
                linespacing=1.6)


def _restore_axes(ax):
    """Clear an axes for real data, undoing :func:`_blank_axes`.

    ``Axes.clear`` restores the ticks and the limits, but a hidden spine stays
    hidden and the background colour it restores is the last one *set* rather
    than the default -- so a blanked panel would otherwise come back frameless
    and still wearing its placeholder tint.
    """
    ax.clear()
    ax.set_facecolor(matplotlib.rcParams["axes.facecolor"])
    for spine in ax.spines.values():
        spine.set_visible(True)


# ---------------------------------------------------------------------------
# Score-gauge geometry.
#
# Every cutoff drawn on a gauge is derived from the threshold profile that
# actually awards the points, so the two cannot drift apart.  legacy-v1 keeps
# its cutoffs inline in compute_scores_legacy(), so they are mirrored here; v2
# and v3 read theirs straight off the shipped threshold dataclasses.
# ---------------------------------------------------------------------------
LEGACY_PATTERN_SIMILARITY_MIN = 0.90
LEGACY_OVERT_ERD_DB = (12.0, 8.0, 5.0)
LEGACY_NONREAL_OVERT_PCT = (75.0, 60.0, 40.0)
LEGACY_IMAGERY_MIN_ERD_DB = 3.0
LEGACY_REL_IQ_PCT = (100.0, 75.0, 50.0)
LEGACY_IQ_DIFF_MEAN_PCT = (15.0, 0.0, -15.0)

# How far past the nominal display range a value is still worth autoscaling to.
# Beyond this the gauge would be squashed into illegibility, so the marker is
# pinned to the edge and drawn as an out-of-range arrow instead.
GAUGE_AUTOSCALE_SPAN = 3.0

# Approximate width of a "3 pts" cutoff label as a fraction of the panel.
GAUGE_LABEL_WIDTH = 0.17

# Keeps status text legible where it crosses a cutoff line.
_GAUGE_TEXT_BBOX = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)


def _cutoffs(thresholds, points_max=3):
    """Pair a descending threshold triple with the points it earns."""
    return list(zip(range(points_max, points_max - len(thresholds), -1),
                    [float(t) for t in thresholds]))


def _cutoff_style(points, points_max):
    """Green marks full credit; partial credit fades with the points earned."""
    if points >= points_max:
        return PALETTE["good"], "-", 1.4
    if points_max - points == 1:
        return "0.35", "-.", 1.1
    return "0.6", ":", 1.0


def _gauge_scale(base, cutoffs, value, bounds=None):
    """Return (xlim, marker_x, marker).

    Values outside the nominal range used to be clipped away by matplotlib,
    leaving an empty axis that reads as "no data" rather than "off the scale".
    The limits therefore always grow to contain the plotted value, up to
    ``GAUGE_AUTOSCALE_SPAN``; past that the marker is pinned just inside the
    edge and drawn as an arrow, so it stays visible without collapsing the
    spacing between the cutoffs.

    ``bounds`` gives the hard domain of the quantity, either end optional.  A
    correlation magnitude cannot exceed 1 and a component count cannot go
    negative, so padding the axis into those regions invents readable values
    that the criterion can never take.
    """
    lo, hi = float(base[0]), float(base[1])
    for _, cut in cutoffs:
        lo, hi = min(lo, cut), max(hi, cut)
    span = hi - lo
    if np.isfinite(value):
        value = float(value)
        if (lo - GAUGE_AUTOSCALE_SPAN * span <= value
                <= hi + GAUGE_AUTOSCALE_SPAN * span):
            lo, hi = min(lo, value), max(hi, value)
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    if bounds is not None:
        # Never let the domain clip a cutoff line out of view.
        cuts = [cut for _, cut in cutoffs]
        if bounds[0] is not None:
            lo = min(max(lo, float(bounds[0])), min(cuts) - 0.02 * span)
        if bounds[1] is not None:
            hi = max(min(hi, float(bounds[1])), max(cuts) + 0.02 * span)
    if not np.isfinite(value):
        return (lo, hi), np.nan, "D"
    if lo <= value <= hi:
        return (lo, hi), value, "D"
    inset = 0.03 * (hi - lo)
    if value < lo:
        return (lo, hi), lo + inset, "<"
    return (lo, hi), hi - inset, ">"


def _combo_field(widget):
    """Read/write a combo box by its item data, never by index."""
    def write(value):
        index = widget.findData(value)
        if index < 0:
            allowed = [widget.itemData(i) for i in range(widget.count())]
            raise ValueError(f"{value!r} is not one of {allowed}")
        widget.setCurrentIndex(index)
    return widget.currentData, write


def _number_field(widget, cast):
    """Read/write a spin box, refusing values its range cannot represent."""
    def write(value):
        if isinstance(value, bool):
            raise ValueError(f"{value!r} is not a number")
        number = cast(value)
        if not widget.minimum() <= number <= widget.maximum():
            raise ValueError(
                f"{number} is outside {widget.minimum()}..{widget.maximum()}")
        widget.setValue(number)
    return (lambda: cast(widget.value())), write


def _flag_field(widget):
    """Read/write a check box, refusing anything but a JSON boolean.

    ``0`` and ``"no"`` are not accepted: a settings file that spells a switch
    in a way this application has to guess at is a file whose behaviour cannot
    be read off it, so it is reported and the default is kept.
    """
    def write(value):
        if not isinstance(value, bool):
            raise ValueError(f"{value!r} is not true or false")
        widget.setChecked(value)
    return widget.isChecked, write


def _text_field(widget):
    """Read/write a free-text line, refusing non-string values."""
    def write(value):
        if not isinstance(value, str):
            raise ValueError(f"{value!r} is not text")
        widget.setText(value)
    return (lambda: widget.text()), write


def _score_gauge_specs(s):
    """Build the six gauge specifications for the active scoring version.

    Cutoffs come from the threshold profile in force, so a gauge cannot
    show a line the formula does not use.  Criteria whose points are gated
    on another quantity carry that gate as an explicit note: without it a
    marker can sit past the full-credit line while the criterion scores 0.
    """
    scoring_version = getattr(s, "scoring_version", "legacy-v1")
    v2 = scoring_version == SCORING_V2
    v3 = scoring_version == SCORING_V3
    v4 = scoring_version == SCORING_V4
    independent = v3 or v4
    thr = (DEFAULT_V4_THRESHOLDS if v4 else
           DEFAULT_V3_THRESHOLDS if v3 else DEFAULT_V2_THRESHOLDS)
    peak = np.asarray(getattr(s, "peakERD", np.full(3, np.nan)),
                      dtype=float).ravel()
    peak = np.pad(peak, (0, max(0, 3 - peak.size)),
                  constant_values=np.nan)[:3]

    if v2 or independent:
        similarity_min = thr.pattern_similarity_min
        overt_erd_db = thr.overt_erd_db
        nonreal_pct = tuple(100.0 * f for f in thr.nonreal_overt_fraction)
        # v2/v3 similarity is |corr|, so the axis has a hard 0..1 domain.
        similarity_lim = (0.0, 1.0)
    else:
        similarity_min = LEGACY_PATTERN_SIMILARITY_MIN
        overt_erd_db = LEGACY_OVERT_ERD_DB
        nonreal_pct = LEGACY_NONREAL_OVERT_PCT
        # legacy-v1 similarity is np.min(corrcoef), which is routinely
        # negative; a 0..1 axis hid exactly those components.
        similarity_lim = (-1.0, 1.0)

    # Criterion 4 needs a positive overt ERD to be a meaningful fraction.
    overt_positive = bool(np.isfinite(peak[0]) and peak[0] > 0)
    nonreal_gate = None if overt_positive else (
        "overt ERD ≤ 0 → 0 pts" if v2 or independent else
        "overt ERD ≤ 0: legacy ratio not monotone with the score")

    specs = [
        # A component count is discrete and non-negative: half-integer ticks
        # such as 2.5 or 7.5 name quantities that cannot exist.  The cutoffs
        # themselves sit between integers because the rule is "3 or more".
        {"title": "ERD candidate\ncomponents", "value": getattr(s, "n_erd", np.nan),
         "cutoffs": _cutoffs((2.5, 0.5), points_max=2), "points_max": 2,
         "lim": (-0.5, 10.5), "bounds": (-0.5, None), "integer_ticks": True},
        {"title": "pattern similarity", "value": getattr(s, "pat_sim", np.nan),
         "cutoffs": _cutoffs((similarity_min,), points_max=1), "points_max": 1,
         "lim": similarity_lim, "bounds": similarity_lim},
        {"title": ("overt 500 ms peak ERD\n[dB]" if v4 else
                   "peak ERD @overt [dB]"), "value": peak[0],
         "cutoffs": _cutoffs(overt_erd_db), "points_max": 3,
         "lim": ((0, 10) if v4 else (0, 20))},
        {"title": ("non-real/overt 500 ms peak\n[%]" if v4 else
                   "non-real ERD vs overt [%]"),
         "value": getattr(s, "rel_nonreal_pct", np.nan),
         "cutoffs": _cutoffs(nonreal_pct), "points_max": 3,
         "lim": (0, 150), "gate": nonreal_gate},
    ]

    if independent:
        # v3/v4 score each condition on its own; neither criterion is gated.
        sustained = getattr(s, "sustained_erd_db", {}) or {}
        cutoffs = _cutoffs(thr.sustained_erd_db)
        specs += [
            {"title": "sustained quasi mu-ERD\n[dB]",
             "value": sustained.get("quasi", np.nan),
             "cutoffs": cutoffs, "points_max": 3, "lim": (-2, 6)},
            {"title": "sustained imagery mu-ERD\n[dB]",
             "value": sustained.get("imagery", np.nan),
             "cutoffs": cutoffs, "points_max": 3, "lim": (-2, 6)},
        ]
        return specs

    # legacy-v1 and v2 both zero criteria 5 and 6 unless imagery peak ERD
    # clears its floor, whatever the plotted difference or ratio says.
    floor = (thr.imagery_min_erd_db if v2 else LEGACY_IMAGERY_MIN_ERD_DB)
    imagery_ok = bool(np.isfinite(peak[2]) and peak[2] > floor)
    gate = None if imagery_ok else f"imagery peak ERD ≤ {floor:g} dB → 0 pts"

    if v2:
        # The v2 difference cutoffs span 2 dB and 1.2 dB; a +/-10 dB panel
        # crushed all three into a single visual line near the middle.
        specs += [
            {"title": "peak imagery − quasi ERD [dB]",
             "value": getattr(s, "peak_iq_diff_db", np.nan),
             "cutoffs": _cutoffs(thr.imagery_minus_quasi_peak_db),
             "points_max": 3, "lim": (-6, 6), "gate": gate},
            {"title": "mean imagery − quasi ERD [dB]",
             "value": getattr(s, "iq_diff_db", np.nan),
             "cutoffs": _cutoffs(thr.imagery_minus_quasi_mean_db),
             "points_max": 3, "lim": (-3, 3), "gate": gate},
        ]
        return specs

    # legacy-v1 criterion 5 compares peakERD[2] > peakERD[1] * k but the
    # panel can only plot their ratio; the two stop agreeing once the quasi
    # denominator is non-positive, so flag that rather than draw a lie.
    quasi_note = None if np.isfinite(peak[1]) and peak[1] > 0 else (
        "quasi peak ERD ≤ 0: ratio not monotone with the score")
    specs += [
        {"title": "peak imagery vs quasi [%]",
         "value": getattr(s, "rel_iq_pct", np.nan),
         "cutoffs": _cutoffs(LEGACY_REL_IQ_PCT), "points_max": 3,
         "lim": (0, 150),
         "gate": "\n".join(n for n in (gate, quasi_note) if n) or None},
        {"title": "mean imagery vs quasi [%]",
         "value": getattr(s, "iq_diff_mean_pct", np.nan),
         "cutoffs": _cutoffs(LEGACY_IQ_DIFF_MEAN_PCT), "points_max": 3,
         "lim": (-50, 50), "gate": gate},
    ]
    return specs


def _draw_gauge(ax, spec):
    """Draw one criterion gauge: cutoffs, gate state, and the measured value."""
    _restore_axes(ax)
    # A gauge is one axis with a marker on it, so it is drawn as a rule under
    # that axis rather than as a box: three sides of a frame around a single
    # scale are furniture the reader has to look past.
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["line"])
    ax.set_title(spec["title"], fontsize=8, pad=4, linespacing=0.9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)

    cutoffs = spec["cutoffs"]
    points_max = spec["points_max"]
    value = spec["value"]
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    xlim, marker_x, marker = _gauge_scale(
        spec["lim"], cutoffs, value, bounds=spec.get("bounds"))
    out_of_range = marker != "D"
    ax.set_xlim(*xlim)
    if spec.get("integer_ticks"):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # A failed gate zeroes the criterion no matter where the value sits, so say
    # so on the panel instead of leaving a marker stranded past a green line.
    gate = spec.get("gate")
    if gate:
        ax.set_facecolor((1.0, 0.94, 0.94))

    # Naming the points each cutoff earns removes the need to decode the line
    # colours.  Cutoffs run right to left, so labels are stacked onto further
    # rows whenever tightly spaced thresholds would otherwise overprint.
    span = xlim[1] - xlim[0]
    rows: list[float] = []
    for points, cut in cutoffs:
        color, style, width = _cutoff_style(points, points_max)
        ax.axvline(cut, color=color, ls=style, lw=width)
        position = (cut - xlim[0]) / span if span else 0.0
        for row, placed in enumerate(rows):
            if placed - position >= GAUGE_LABEL_WIDTH:
                rows[row] = position
                break
        else:
            rows.append(position)
            row = len(rows) - 1
        # A cutoff close to the right edge takes its label on the inside.
        inside = position + GAUGE_LABEL_WIDTH <= 1.0
        ax.annotate(f"{points} pt" + ("" if points == 1 else "s"),
                    xy=(cut, 0.97 - 0.12 * row),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(2 if inside else -2, 0), textcoords="offset points",
                    ha="left" if inside else "right", va="top", fontsize=6,
                    color=color if points >= points_max else "0.35")

    if np.isfinite(value):
        # clip_on=False keeps a marker sitting on a domain bound (|corr| = 1,
        # zero components) whole instead of sliced in half by the spine.
        ax.scatter([marker_x], [0.5], s=90, zorder=3, marker=marker,
                   color=PALETTE["mark"], clip_on=False)
        if out_of_range:
            ax.text(0.5, 0.78, f"off scale: {value:.3g}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color=PALETTE["mark"],
                    bbox=_GAUGE_TEXT_BBOX)
    else:
        ax.text(0.5, 0.5, "unavailable", transform=ax.transAxes,
                ha="center", va="center", color=PALETTE["faint"],
                bbox=_GAUGE_TEXT_BBOX)

    if gate:
        ax.text(0.5, 0.16, gate, transform=ax.transAxes, ha="center",
                va="center", fontsize=6, color="0.25")
    ax.set_xlabel(spec["xlabel"], fontsize=8)


def _first_metadata(objects, *names):
    """Find the first non-empty metadata attribute across result objects."""
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is not None and not (isinstance(value, str) and not value):
                return value
        provenance = getattr(obj, "provenance", None)
        if isinstance(provenance, dict):
            for name in names:
                if name in provenance:
                    value = provenance[name]
                    if value is not None and not (
                            isinstance(value, str) and not value):
                        return value
    return None


class WrapLabel(QtWidgets.QLabel):
    """A word-wrapped label that is actually given the height its text needs.

    A QLabel reports a size hint computed for a width the layout has not chosen
    yet, so a wrapped label inside a width-constrained column is routinely
    allotted one line and has its second sliced in half. Pinning the minimum
    height to what the width in force requires fixes it at every width the
    column is ever given, including after the text changes.
    """

    def __init__(self, text=""):
        super().__init__(text)
        self.setWordWrap(True)

    def setText(self, text):     # noqa: N802 - Qt's own spelling
        super().setText(text)
        self._fit()

    def resizeEvent(self, event):    # noqa: N802 - Qt's own spelling
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        width = self.width()
        if width <= 0:
            return
        needed = self.heightForWidth(width)
        # Only when it changes: assigning the same minimum on every resize
        # would relayout the parent for nothing.
        if needed > 0 and needed != self.minimumHeight():
            self.setMinimumHeight(needed)


class Canvas(FigureCanvas):
    def __init__(self, nrows=1, ncols=1, facecolor=None):
        self.fig = Figure(facecolor=facecolor or PALETTE["surface"],
                          tight_layout=True)
        super().__init__(self.fig)


# ---------------------------------------------------------------------------
# Background worker so the UI does not freeze during the heavy analysis.
# ---------------------------------------------------------------------------
class StartWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, pipe, fr, fq, fi, extract=None, file_type="h5",
                 ch_num=64, fs=1000.0, processing_fs=1000.0,
                 candidate_scale="raw", csp_shrinkage=0.0):
        super().__init__()
        self.pipe, self.fr, self.fq, self.fi = pipe, fr, fq, fi
        self.extract = extract
        self.file_type, self.ch_num, self.fs = file_type, ch_num, fs
        self.processing_fs = processing_fs
        self.candidate_scale = candidate_scale
        self.csp_shrinkage = csp_shrinkage

    def run(self):
        try:
            kwargs = {
                "extract": self.extract,
                "file_type": self.file_type,
                "ch_num": self.ch_num,
                "fs": self.fs,
            }
            if _supports_parameter(self.pipe.run_start, "processing_fs"):
                kwargs["processing_fs"] = self.processing_fs
            if _supports_parameter(self.pipe.run_start, "candidate_scale"):
                kwargs["candidate_scale"] = self.candidate_scale
            if _supports_parameter(self.pipe.run_start, "csp_shrinkage"):
                kwargs["csp_shrinkage"] = self.csp_shrinkage
            self.done.emit(self.pipe.run_start(
                self.fr, self.fq, self.fi, **kwargs))
        except Exception:
            self.failed.emit(traceback.format_exc())


class FTWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, pipe, comp,
                 scoring_version=SCORING_V4,
                 csp_cv_folds=DEFAULT_CSP_CV_FOLDS):
        super().__init__()
        self.pipe, self.comp = pipe, comp
        self.scoring_version = scoring_version
        self.csp_cv_folds = csp_cv_folds

    def run(self):
        try:
            # Backward compatibility for a pipeline predating versioned scoring or
            # cross-validated overt projection.
            kwargs = {}
            if _supports_parameter(self.pipe.compute_ft, "scoring_version"):
                kwargs["scoring_version"] = self.scoring_version
            if _supports_parameter(self.pipe.compute_ft, "csp_cv_folds"):
                kwargs["csp_cv_folds"] = self.csp_cv_folds
            result = self.pipe.compute_ft(self.comp, **kwargs)
            self.done.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class NeuroCastingApp(QtWidgets.QMainWindow):
    def __init__(self, settings=None, config_path=None):
        super().__init__()
        self.settings = settings or QtCore.QSettings(
            "NeuroCasting", "NeuroCasting Quasi-fMRI")
        self.config_path = config_path or str(application_dir() / SETTINGS_FILE_NAME)
        self._config_notes = []
        self.pipe = KLHPipeline()
        self.start_result = None
        self.ft_result = None
        self._topo_labels = []
        self._n_display_components = MAX_DISPLAY_COMPONENTS
        self._auto_selection = None
        self._evals_legend = None
        self._input_files = None
        self._eeg_status = ""
        self._emg_status = ""
        # The EEG run START has lined up, held until the EMG half is finished.
        self._pending_eeg = None
        self.setWindowTitle("Kitties' Little Helper")
        self.resize(*WINDOW_SIZE)
        # Typeface, palette and matplotlib defaults, before a single widget or
        # figure is built: the panels take their colours from the same place
        # the window does, so neither has to be corrected afterwards.
        theme.apply(self)
        # Keep the window/taskbar branding consistent with the cat shown on
        # the start screen.
        icon = asset_path("klh_intro.png")
        if os.path.exists(icon):
            self.setWindowIcon(QtGui.QIcon(icon))

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self._build_welcome()
        self._build_evaluate()
        self._build_rating()
        self._build_emg()
        self._restore_settings()

    # ---- Start tab ----
    @staticmethod
    def _field_label(text):
        """The name of one form field, in the form's own quieter type."""
        label = QtWidgets.QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    @staticmethod
    def _section_header(text, top=14):
        """A heading over a group of fields, spanning both form columns.

        The start form is one flat list of a dozen fields describing three
        different things -- which recordings to read, how this session was
        run, and where everything else is configured. Naming the groups costs
        no row of its own and turns one long list into three short ones; the
        air above each heading is what separates them.
        """
        label = QtWidgets.QLabel(text)
        label.setObjectName("sectionHeader")
        label.setContentsMargins(0, top, 0, 2)
        theme.label_font(label, point_size=8.5,
                         weight=QtGui.QFont.Weight.Bold,
                         letter_spacing=112.0, uppercase=True)
        return label

    def _build_welcome(self):
        w = QtWidgets.QWidget()
        # Held as an attribute so the set of parameters still on the form is
        # inspectable rather than implied by the order of addRow calls.
        self.form_welcome = form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                               | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(6)
        self.ed_folder = QtWidgets.QLineEdit(DATA_FOLDER_DEFAULT)
        self.ed_subj = QtWidgets.QLineEdit("01TST")
        self.ed_folder.setToolTip(
            "Enter a folder path, optionally using SUBJECT_NAME, or browse "
            "to the folder containing the participant's recordings.")
        self.btn_browse_folder = QtWidgets.QPushButton("Browse...")
        self.btn_browse_folder.setToolTip(
            "Choose a recording folder and fill in its path and subject name.")
        self.btn_browse_folder.clicked.connect(self._browse_folder)
        self.ed_real = QtWidgets.QLineEdit("om1, om2")
        self.ed_quasi = QtWidgets.QLineEdit("qm1, qm2")
        self.ed_imag = QtWidgets.QLineEdit("im1, im2")
        for edit in (self.ed_real, self.ed_quasi, self.ed_imag):
            edit.setToolTip(
                "Filled in from the resolved folder. Edit freely: nothing is "
                "detected again until the folder or subject changes, or "
                "'detect from folder' is pressed.")
        self.ed_folder.editingFinished.connect(self._folder_changed)
        self.ed_subj.editingFinished.connect(self._folder_changed)
        self.btn_detect_files = QtWidgets.QPushButton("detect from folder")
        self.btn_detect_files.setToolTip(
            "Re-read the folder and refill the three filename lists.")
        self.btn_detect_files.clicked.connect(self._detect_recordings)
        # A button stretched the width of the form reads as a banner rather
        # than something to press, so it keeps its own width.
        self.btn_detect_files.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed)
        self.lbl_detected_files = WrapLabel()
        self.lbl_detected_files.setObjectName("hint")
        self.cmb_ftype = QtWidgets.QComboBox()
        for label, value in FILE_TYPES:
            self.cmb_ftype.addItem(label, value)
        self.cmb_ftype.setCurrentIndex(1)
        self.cmb_fs = QtWidgets.QComboBox()
        for label, value in RECORDING_RATES:
            self.cmb_fs.addItem(label, value)
        self.cmb_fs.setCurrentIndex(1)
        self.spin_nchan = QtWidgets.QSpinBox()
        self.spin_nchan.setRange(MIN_H5_CHANNELS, MAX_REFERENCE_CHANNELS)
        self.spin_nchan.setValue(64)
        self.spin_nchan.setToolTip(
            "H5 data must use 2–64 channels from the 64-channel reference montage.")
        # Each field is given the width its value needs, rather than left to
        # grow into whatever a wide window has going -- and rather than left at
        # the default hint, which is narrower than any of these values. The
        # folder template is the one that routinely carries a long value.
        folder_row = QtWidgets.QWidget()
        folder_row.setFixedWidth(FIELD_WIDTH_PATH)
        folder_layout = QtWidgets.QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(6)
        folder_layout.addWidget(self.ed_folder, 1)
        folder_layout.addWidget(self.btn_browse_folder)
        self.ed_subj.setFixedWidth(FIELD_WIDTH_SHORT)
        for edit in (self.ed_real, self.ed_quasi, self.ed_imag):
            edit.setFixedWidth(FIELD_WIDTH_LIST)
        form.addRow(self._section_header("Recordings", top=0))
        form.addRow(self._field_label("folder / template"), folder_row)
        form.addRow(self._field_label("subj name"), self.ed_subj)
        form.addRow(self._field_label("filenames real"), self.ed_real)
        form.addRow(self._field_label("filenames quasi"), self.ed_quasi)
        form.addRow(self._field_label("filenames imag"), self.ed_imag)
        form.addRow("", self.btn_detect_files)
        form.addRow("", self.lbl_detected_files)

        # ---- data extraction: trigger channel vs embedded XDF markers ----
        self.cmb_extract = QtWidgets.QComboBox()
        for label, value in EXTRACT_MODES:
            self.cmb_extract.addItem(label, value)
        self.cmb_movement = QtWidgets.QComboBox()
        for label, condition in MOVEMENT_CONDITIONS:
            self.cmb_movement.addItem(label, condition)
        self.cmb_movement.setToolTip(
            "Selects which hand's microrepeat events are used as active trials.")
        self.spin_preparation = QtWidgets.QDoubleSpinBox()
        self.spin_preparation.setRange(0.0, 60.0)
        self.spin_preparation.setSingleStep(0.5)
        self.spin_preparation.setValue(2.0)
        self.spin_preparation.setSuffix(" s")
        self.spin_preparation.setToolTip(
            "Preparation excluded after the corrected marker; motor-task time "
            "zero begins after this interval.")
        self.spin_recovery = QtWidgets.QDoubleSpinBox()
        self.spin_recovery.setRange(0.0, 60.0)
        self.spin_recovery.setSingleStep(0.5)
        self.spin_recovery.setValue(2.0)
        self.spin_recovery.setSuffix(" s")
        self.spin_recovery.setToolTip(
            "Real continuous data retained after the marker-defined trial for "
            "post-movement ERD/ERS recovery.")
        self.spin_cut_first = QtWidgets.QDoubleSpinBox()
        self.spin_cut_first.setRange(0.0, 60.0)
        self.spin_cut_first.setSingleStep(0.5)
        self.spin_cut_first.setValue(0.0)
        self.spin_cut_first.setSuffix(" s")
        self.spin_cut_first.setToolTip(
            "Optional extra trim after the preparation interval.")
        self.spin_cut_last = QtWidgets.QDoubleSpinBox()
        self.spin_cut_last.setRange(0.0, 60.0)
        self.spin_cut_last.setSingleStep(0.5)
        self.spin_cut_last.setValue(0.0)
        self.spin_cut_last.setSuffix(" s")
        self.cmb_marker_baseline = QtWidgets.QComboBox()
        for label, value in MARKER_BASELINES:
            self.cmb_marker_baseline.addItem(label, value)
        self.cmb_marker_baseline.setToolTip(
            "Rest normalization used by embedded-marker recordings.")
        self.spin_rest_skip = QtWidgets.QDoubleSpinBox()
        self.spin_rest_skip.setRange(0.0, 30.0)
        self.spin_rest_skip.setDecimals(2)
        self.spin_rest_skip.setSingleStep(0.25)
        self.spin_rest_skip.setValue(1.0)
        self.spin_rest_skip.setSuffix(" s")
        self.spin_rest_skip.setToolTip(
            "Exclude the nonstationary transition at the start of each rest event.")
        self.spin_marker_padding = QtWidgets.QDoubleSpinBox()
        self.spin_marker_padding.setRange(1.0, 10.0)
        self.spin_marker_padding.setDecimals(2)
        self.spin_marker_padding.setSingleStep(0.25)
        self.spin_marker_padding.setValue(1.5)
        self.spin_marker_padding.setSuffix(" s")
        self.spin_marker_padding.setToolTip(
            "Real neighboring data retained around marker epochs for TF transforms.")
        self.chk_auto_marker_shift = QtWidgets.QCheckBox("detect per recording")
        self.chk_auto_marker_shift.setChecked(True)
        self.chk_auto_marker_shift.setToolTip(
            "Measure each recording's own amplifier timestamp lag from its XDF "
            "and use it as that recording's shift. Untick to type one shift "
            "and apply it to every recording.")
        self.spin_marker_shift = QtWidgets.QDoubleSpinBox()
        self.spin_marker_shift.setRange(-30.0, 30.0)
        self.spin_marker_shift.setDecimals(3)
        self.spin_marker_shift.setSingleStep(0.1)
        self.spin_marker_shift.setValue(-4.0)
        self.spin_marker_shift.setSuffix(" s")
        self.spin_marker_shift.setToolTip(
            "Added to every embedded active and rest marker onset. "
            "-4 moves all trial windows four seconds earlier.")
        self.chk_auto_marker_shift.toggled.connect(self._marker_shift_mode_changed)
        self.cmb_score_version = QtWidgets.QComboBox()
        for label, value in SCORING_VERSIONS:
            self.cmb_score_version.addItem(label, value)
        self.cmb_score_version.setToolTip(
            "v4 adds a continuous 500 ms rolling-median peak and independently "
            "scores sustained quasi and imagery mu-ERD; v3 retains its archived "
            "third-percentile peak, v2 retains imagery-minus-quasi comparisons, "
            "and legacy-v1 reproduces the historical formula.")
        self.cmb_candidate_scale = QtWidgets.QComboBox()
        self.cmb_candidate_scale.addItem("Raw eigenvalues (historical)", "raw")
        self.cmb_candidate_scale.addItem("Log-odds eigenvalues", "logit")
        self.cmb_candidate_scale.setToolTip(
            "Candidate detection scale. Raw preserves historical behavior; "
            "log-odds treats both CSP spectrum ends symmetrically.")
        self.spin_csp_shrinkage = QtWidgets.QDoubleSpinBox()
        self.spin_csp_shrinkage.setRange(0.0, 0.95)
        self.spin_csp_shrinkage.setDecimals(3)
        self.spin_csp_shrinkage.setSingleStep(0.01)
        self.spin_csp_shrinkage.setValue(0.0)
        self.spin_csp_shrinkage.setToolTip(
            "Optional CSP covariance shrinkage. Zero preserves historical behavior.")
        self.spin_csp_cv_folds = QtWidgets.QSpinBox()
        self.spin_csp_cv_folds.setRange(0, 20)
        self.spin_csp_cv_folds.setValue(DEFAULT_CSP_CV_FOLDS)
        self.spin_csp_cv_folds.setToolTip(
            "Folds used to score overt ERD on held-out trials. The CSP filter "
            "is fitted on overt movement, so an in-sample overt estimate "
            "inflates criterion 3 and biases criterion 4 low. 0 or 1 disables "
            "cross-validation and restores the historical in-sample estimate.")
        self.spin_superlet_workers = QtWidgets.QSpinBox()
        self.spin_analysis_workers = QtWidgets.QSpinBox()
        self.spin_analysis_workers.setRange(0, 32)
        self.spin_analysis_workers.setValue(execution.DEFAULT_WORKERS)
        self.spin_analysis_workers.setToolTip(
            "Threads for independent covariance fits, bootstrap batches and "
            "filter bands. Default 2 limits memory use; 0 or 1 runs serially. "
            "BLAS uses one thread per concurrent covariance fit.")
        self.spin_superlet_workers.setRange(0, 32)
        self.spin_superlet_workers.setValue(superlet.DEFAULT_WORKERS)
        self.spin_superlet_workers.setToolTip(
            "Worker processes that split the time-frequency transform across "
            "cores. Results do not depend on this: the frequency axis is "
            "separable and every worker uses the whole spectrum's block "
            "geometry, so any setting gives bit-identical output. Costs about "
            "170 MB per worker. 0 or 1 keeps the transform in one process.")
        shift_row = QtWidgets.QHBoxLayout()
        shift_row.setContentsMargins(0, 0, 0, 0)
        shift_row.addWidget(self.chk_auto_marker_shift)
        shift_row.addWidget(self.spin_marker_shift)
        shift_row.addStretch(1)
        shift_holder = QtWidgets.QWidget(w)
        shift_holder.setLayout(shift_row)

        # ---- EMG electrodes ----
        # The EEG half analyses the hand chosen above; the EMG half always
        # analyses both, so both bipolar pairs are named here regardless.
        self.ed_emg_left = QtWidgets.QLineEdit(EMG_LEFT_CHANNELS_DEFAULT)
        self.ed_emg_right = QtWidgets.QLineEdit(EMG_RIGHT_CHANNELS_DEFAULT)
        for edit, side in ((self.ed_emg_left, "left"),
                           (self.ed_emg_right, "right")):
            edit.setToolTip(
                f"Positive and negative channel of the {side}-hand bipolar EMG "
                "derivation, as labelled in the recording. Channel numbers "
                "(1-based) are accepted where a file carries no labels.")

        self.cmb_movement.setFixedWidth(FIELD_WIDTH_SHORT)
        self.ed_emg_left.setFixedWidth(FIELD_WIDTH_LIST)
        self.ed_emg_right.setFixedWidth(FIELD_WIDTH_LIST)
        form.addRow(self._section_header("This session"))
        form.addRow(self._field_label("movement trials"), self.cmb_movement)
        form.addRow(self._field_label("left hand EMG (+, −)"), self.ed_emg_left)
        form.addRow(self._field_label("right hand EMG (+, −)"),
                    self.ed_emg_right)
        form.addRow(self._field_label("marker time shift"), shift_holder)

        # Everything else is configured in the settings file rather than on
        # this form.  The widgets stay alive as the in-memory model: they carry
        # the valid ranges, the defaults and the format-dependent enable rules
        # that the pipeline calls still rely on.  Parenting them to a hidden
        # container keeps them off screen without leaving stray top-level
        # windows behind.
        self._advanced_holder = QtWidgets.QWidget(w)
        self._advanced_holder.setVisible(False)
        hidden = QtWidgets.QFormLayout(self._advanced_holder)
        for widget in (
            self.cmb_ftype, self.cmb_fs, self.spin_nchan, self.cmb_extract,
            self.spin_preparation, self.spin_recovery, self.spin_cut_first,
            self.spin_cut_last, self.cmb_marker_baseline, self.spin_rest_skip,
            self.spin_marker_padding, self.cmb_candidate_scale,
            self.spin_csp_shrinkage, self.spin_csp_cv_folds,
            self.spin_superlet_workers, self.spin_analysis_workers,
            self.cmb_score_version,
            *self._build_emg_parameters(),
        ):
            hidden.addRow(widget)

        self.lbl_settings_file = WrapLabel()
        self.lbl_settings_file.setObjectName("hint")
        self.lbl_settings_file.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self._section_header("Advanced settings"))
        # A whole-width row rather than a field: the value is one long absolute
        # path, and the label gutter is width it would otherwise wrap around.
        form.addRow(self.lbl_settings_file)

        self.cmb_extract.currentIndexChanged.connect(self._extract_mode_changed)
        self.cmb_ftype.currentIndexChanged.connect(self._ftype_changed)
        self._ftype_changed()

        title = QtWidgets.QLabel("Kitties' Little Helper")
        title.setObjectName("appTitle")
        subtitle = QtWidgets.QLabel(
            "Quasi-fMRI · EEG transfer score + hand-EMG screening")
        subtitle.setObjectName("appSubtitle")
        intro = QtWidgets.QLabel()
        p = asset_path("klh_intro.png")
        if os.path.exists(p):
            intro.setPixmap(QtGui.QPixmap(p).scaled(INTRO_IMAGE_PX,
                INTRO_IMAGE_PX,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))
            intro.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.btn_start = QtWidgets.QPushButton("START  ▸")
        self.btn_start.setObjectName("startButton")
        self.btn_start.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.on_start)
        # Deliberately not a WrapLabel: this one reports a run in progress, and
        # a label that takes the height its text needs would make the window
        # grow the moment START was pressed and shrink again when the run
        # finished. It is given a fixed two lines -- less than the height of
        # the button beside it, so it cannot set the row's height either way --
        # and the full text, which is never only here, is on the tooltip.
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setObjectName("statusLine")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft
                                     | QtCore.Qt.AlignmentFlag.AlignTop)
        self.lbl_status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed)

        # The form and START share one centred column no wider than the fields
        # need. Without it START drifts to the far edge of a wide window, half
        # a screen from the form it acts on. The status takes the rest of the
        # row, so its two lines are as wide as they can be before they wrap.
        h = QtWidgets.QHBoxLayout()
        h.setSpacing(16)
        h.addWidget(self.lbl_status, 1)
        h.addWidget(self.btn_start, 0)
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addLayout(form)
        body_layout.addSpacing(14)
        body_layout.addLayout(h)
        body.setMaximumWidth(START_BODY_WIDTH)
        # The settings-file path wraps inside this column, and a wrapped label's
        # height depends on the width it is finally given. Without this the
        # column is sized for one line and the second is sliced in half.
        policy = body.sizePolicy()
        policy.setHeightForWidth(True)
        body.setSizePolicy(policy)

        # The form is a sheet of paper on the page rather than text floating on
        # it: one white card, so the eye is told where the session is described
        # and where it ends.
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(*START_CARD_PADDING)
        card_layout.addWidget(body)
        card.setMaximumWidth(START_BODY_WIDTH + START_CARD_PADDING[0]
                             + START_CARD_PADDING[2])
        policy = card.sizePolicy()
        policy.setHeightForWidth(True)
        card.setSizePolicy(policy)

        # The cat beside the title rather than over it: stacked, the three of
        # them cost the height of the picture plus two lines of type, which is
        # most of what made this tab taller than the window it opens in.
        wordmark = QtWidgets.QVBoxLayout()
        wordmark.setContentsMargins(0, 0, 0, 0)
        wordmark.setSpacing(2)
        # The two lines are one block centred against the picture; without the
        # stretches they are stretched apart to the picture's own height.
        wordmark.addStretch(1)
        wordmark.addWidget(title)
        wordmark.addWidget(subtitle)
        wordmark.addStretch(1)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(18)
        header.addStretch(1)
        header.addWidget(intro, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header.addLayout(wordmark, 0)
        header.addStretch(1)

        # Every block takes its own height and one stretch absorbs the rest,
        # so the header and the form stay a single group at the top instead of
        # being pushed apart by the slack a tall window leaves.
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(16, 8, 16, 10)
        lay.addLayout(header, 0)
        lay.addSpacing(10)
        lay.addWidget(card, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(1)
        self.tabs.addTab(w, "Start")

        # Measured only now: a widget with no parent yet still carries the
        # application's font rather than this window's, so asking it for its
        # line height before it is in the tree sizes it for the wrong type.
        metrics = QtGui.QFontMetrics(self.lbl_status.font())
        self.lbl_status.setFixedHeight(
            STATUS_LINE_COUNT * metrics.lineSpacing())

    # ---- EMG analysis parameters (settings file) ----
    @staticmethod
    def _spin(minimum, maximum, value, suffix, decimals=2, tooltip=""):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setToolTip(tooltip)
        return widget

    def _build_emg_parameters(self):
        """Create the EMG processing widgets and return them.

        These never appear on a form: like the EEG analysis parameters, they
        are read from the settings file, and the widgets exist to hold the
        ranges and defaults that decide what that file is allowed to say. Every
        default is taken from :class:`emgcasting.core.ProcessingConfig`, so the
        two cannot disagree about what the analysis does by default.
        """
        defaults = EMGConfig
        self.cmb_emg_envelope = QtWidgets.QComboBox()
        self.cmb_emg_envelope.addItem("TKEO energy (reference pipeline)", "tkeo")
        self.cmb_emg_envelope.addItem("Plain RMS", "rms")
        self.cmb_emg_unit = QtWidgets.QComboBox()
        for unit in ("V", "mV"):
            self.cmb_emg_unit.addItem(unit, unit)
        self.spin_emg_rate = self._spin(
            0, 10000, 0.0, " Hz", decimals=1,
            tooltip="Rate the EMG is resampled to before filtering. 0 keeps "
                    "each recording's own rate.")
        self.spin_emg_notch = self._spin(0, 500, defaults.notch_hz, " Hz")
        self.spin_emg_band_low = self._spin(0.1, 1000, defaults.band_low_hz, " Hz")
        self.spin_emg_band_high = self._spin(0.1, 2000, defaults.band_high_hz, " Hz")
        self.spin_emg_window = self._spin(1, 5000, defaults.window_ms, " ms")
        self.spin_emg_step = self._spin(1, 5000, defaults.step_ms, " ms")
        self.spin_emg_pre = self._spin(0, 60, defaults.pre_s, " s")
        self.spin_emg_post = self._spin(0, 60, defaults.post_s, " s")
        self.spin_emg_tail = self._spin(
            0, 5000, 1000.0 * defaults.trial_tail_s, " ms",
            tooltip="Extra classification-only time after the scheduled trial, "
                    "so a burst crossing the end boundary is not truncated.")
        self.spin_emg_rest_trim_start = self._spin(
            0, 60, defaults.rest_trim_start_s, " s",
            tooltip="Dropped from the start of every rest block, to avoid "
                    "carry-over from the preceding task block.")
        self.spin_emg_rest_trim_end = self._spin(
            0, 60, defaults.rest_trim_end_s, " s",
            tooltip="Dropped from the end of every rest block; movement starts "
                    "before the task-block cue.")
        self.spin_emg_peak = self._spin(
            1.0, 1000.0, defaults.peak_multiplier, "x",
            tooltip="How tall a peak must be to count, as a multiple of this "
                    "recording's own clean-rest background.")
        self.spin_emg_width = self._spin(
            1.0, 100.0, defaults.background_multiplier, "x",
            tooltip="The lower shoulder a peak's width is measured at, as a "
                    "multiple of the same background.")
        self.spin_emg_preref_start = self._spin(
            0, 5000, 1000.0 * defaults.pre_reference_start_s, " ms")
        self.spin_emg_preref_end = self._spin(
            0, 5000, 1000.0 * defaults.pre_reference_end_s, " ms")
        self.spin_emg_adaptive_peak = self._spin(
            0.1, 1000.0, defaults.secondary_pre_multiplier, "x",
            tooltip="Adaptive branch: the peak must also exceed this multiple "
                    "of its own trial's pre-movement median.")
        self.spin_emg_adaptive_width = self._spin(
            0.1, 1000.0, defaults.secondary_width_multiplier, "x")
        self.spin_emg_min_burst = self._spin(
            0, 5000, defaults.min_burst_ms, " ms",
            tooltip="How wide a peak must be at the shoulder to count.")
        self.spin_emg_rest_warn = self._spin(
            0.1, 100.0, 100.0 * defaults.rest_fpr_warn, "%",
            tooltip="If this share of a run's own rest windows also fires, its "
                    "count is reported as an underestimate.")
        self.chk_emg_trial_figures = QtWidgets.QCheckBox(
            "write one figure per trial when saving")
        self.chk_emg_trial_figures.setChecked(defaults.trial_figures)
        self.ed_emg_output = QtWidgets.QLineEdit(str(EMG_OUTPUT_ROOT))
        self.ed_emg_output.setToolTip(
            "Root the EMG tab's Save button writes a participant folder into. "
            "A relative path is taken from the application folder, so results "
            "stay inside this installation wherever it is copied to; give an "
            "absolute path to write somewhere else.")
        return (
            self.cmb_emg_envelope, self.cmb_emg_unit, self.spin_emg_rate,
            self.spin_emg_notch, self.spin_emg_band_low, self.spin_emg_band_high,
            self.spin_emg_window, self.spin_emg_step, self.spin_emg_pre,
            self.spin_emg_post, self.spin_emg_tail,
            self.spin_emg_rest_trim_start, self.spin_emg_rest_trim_end,
            self.spin_emg_peak, self.spin_emg_width, self.spin_emg_preref_start,
            self.spin_emg_preref_end, self.spin_emg_adaptive_peak,
            self.spin_emg_adaptive_width, self.spin_emg_min_burst,
            self.spin_emg_rest_warn, self.chk_emg_trial_figures,
            self.ed_emg_output,
        )

    # ---- EEG Components tab ----
    def _build_evaluate(self):
        w = QtWidgets.QWidget()
        # A page that is nothing but figures takes the figures' own white, so
        # the panels are part of it rather than four bright rectangles on it.
        w.setObjectName("plotPage")
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(10, 10, 12, 10)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.slider.setMinimum(1)
        self.slider.setMaximum(MAX_DISPLAY_COMPONENTS)
        self.slider.setValue(MAX_DISPLAY_COMPONENTS)  # top -> component 1
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksRight)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setFixedWidth(28)   # keep the slider column narrow
        self.slider.valueChanged.connect(self._highlight_selected_topo)
        # The slider lives in a padded holder rather than directly in the grid:
        # its travel has to be inset to the topography band centres, and the
        # inset depends on the handle size and the canvas height at run time.
        self.slider_holder = QtWidgets.QWidget()
        holder = QtWidgets.QVBoxLayout(self.slider_holder)
        holder.setContentsMargins(0, 0, 0, 0)
        holder.setSpacing(0)
        holder.addWidget(self.slider)
        grid.addWidget(self.slider_holder, 0, 0, 3, 1)

        self.canvas_topos = Canvas()
        # Fixed margins (no auto-layout) so the five maps sit in equal vertical
        # bands; the 1-5 selection numbers are drawn in the reserved left strip,
        # lining up exactly with the maps in place of the slider's tick marks.
        self.canvas_topos.fig.set_layout_engine("none")
        self.canvas_topos.fig.subplots_adjust(left=0.22, right=0.99,
                                               top=0.99, bottom=0.01, hspace=0.12)
        self.ax_topos = [
            self.canvas_topos.fig.add_subplot(MAX_DISPLAY_COMPONENTS, 1, i + 1)
            for i in range(MAX_DISPLAY_COMPONENTS)
        ]
        for a in self.ax_topos:
            a.axis("off")
        self._blank_topos()
        # Re-inset the slider whenever the maps are re-laid out by a resize.
        self.canvas_topos.installEventFilter(self)
        grid.addWidget(self.canvas_topos, 0, 1, 3, 1)

        # Both panels take fixed margins rather than tight-layout, so their
        # boxes can be given one common x-span; see _align_eeg_axes.
        self.canvas_evals = Canvas()
        self.canvas_evals.fig.set_layout_engine("none")
        self.ax_evals = self.canvas_evals.fig.add_subplot(111)
        _blank_axes(self.ax_evals, PLACEHOLDER_EVALS)
        self.canvas_evals.installEventFilter(self)
        grid.addWidget(self.canvas_evals, 0, 2, 1, 2)

        self.canvas_tf = Canvas()
        self.canvas_tf.fig.set_layout_engine("none")
        self.ax_tf = [self.canvas_tf.fig.add_subplot(3, 1, i + 1) for i in range(3)]
        for name, ax in zip(TF_CONDITION_NAMES, self.ax_tf):
            _blank_axes(ax, PLACEHOLDER_TF.format(name=name))
        self.canvas_tf.setMinimumHeight(300)   # avoid squashed TF strips
        self.canvas_tf.installEventFilter(self)
        grid.addWidget(self.canvas_tf, 1, 2, 2, 2)

        self.btn_ft = QtWidgets.QPushButton("Calculate FT")
        self.btn_ft.setObjectName("primaryButton")
        self.btn_ft.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_ft.clicked.connect(self.on_ft)
        self.btn_ft.setEnabled(False)
        self.btn_rating = QtWidgets.QPushButton("Score  ▸")
        self.btn_rating.setToolTip("Show the six criterion gauges and the total.")
        self.btn_rating.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_rating.setEnabled(False)
        self.btn_rating.clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.tab_score))
        bl = QtWidgets.QHBoxLayout()
        bl.setContentsMargins(4, 2, 4, 2)
        bl.setSpacing(10)
        bl.addWidget(self.btn_ft)
        bl.addWidget(self.btn_rating)
        self.lbl_component_selection = WrapLabel()
        self.lbl_component_selection.setObjectName("hint")
        self._update_component_selection_summary()
        bl.addWidget(self.lbl_component_selection, 1)
        holder = QtWidgets.QWidget()
        holder.setLayout(bl)
        grid.addWidget(holder, 3, 0, 1, 4)

        # Bias the layout toward the plots so the time-frequency maps are not
        # squashed into thin strips: give the right-hand column most of the
        # width and the TF rows (1-2) most of the height; keep the slider
        # column narrow and the button row at fixed height.
        grid.setColumnStretch(0, 0)   # slider
        # The maps are square and their height is fixed by five equal bands, so
        # a wider column only pads white space either side of them -- and pushes
        # the slider away from the maps it selects.
        grid.setColumnStretch(1, 1)   # topographies
        grid.setColumnStretch(2, 3)   # eigenvalues / TF maps span cols 2-3
        grid.setColumnStretch(3, 3)
        grid.setRowStretch(0, 2)      # eigenvalue spectrum
        grid.setRowStretch(1, 3)      # TF maps (rows 1-2)
        grid.setRowStretch(2, 3)
        grid.setRowStretch(3, 0)      # button bar
        self.tab_components = w
        self.tabs.addTab(w, "EEG Components")

    # ---- Score tab ----
    def _build_rating(self):
        w = QtWidgets.QWidget()
        w.setObjectName("plotPage")
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(10, 10, 12, 10)
        self.canvas_scores = Canvas()
        # Tight-layout reacts to long score titles by collapsing the axes at
        # narrower window sizes. Fixed margins plus wrapped titles keep all six
        # gauges legible and separated.
        self.canvas_scores.fig.set_layout_engine("none")
        self.canvas_scores.fig.subplots_adjust(
            left=0.10, right=0.98, top=SCORE_AXES_TOP, bottom=0.06,
            wspace=0.42, hspace=0.62)
        self.canvas_scores.setMinimumWidth(520)
        self.canvas_scores.installEventFilter(self)
        self.ax_scores = [
            self.canvas_scores.fig.add_subplot(3, 2, i + 1)
            for i in range(6)
        ]
        for number, ax in enumerate(self.ax_scores, start=1):
            _blank_axes(ax, PLACEHOLDER_GAUGE.format(number=number), fontsize=8)
        grid.addWidget(self.canvas_scores, 0, 0, 1, 2)

        # The picture fills its canvas edge to edge, and the canvas is kept to
        # the picture's own proportions, so the total sits directly beneath it
        # rather than across whatever slack a square image leaves in a tall
        # column. A spacer as deep as the gauge figure's top margin then starts
        # the picture on the top edge of the first row of gauges.
        self.canvas_finalimg = Canvas()
        self.canvas_finalimg.fig.set_layout_engine("none")
        self.canvas_finalimg.fig.subplots_adjust(
            left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.ax_finalimg = self.canvas_finalimg.fig.add_subplot(111)
        self._blank_score_image()
        self.canvas_finalimg.installEventFilter(self)

        # "Total score: 0 / 15" before anything has been calculated is a
        # result, and the wrong one: nothing has been scored zero yet.
        self.lbl_final = QtWidgets.QLabel(NO_SCORE_TEXT)
        self.lbl_final.setObjectName("totalScore")
        self.lbl_final.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_final.setWordWrap(True)

        self.score_top_spacer = QtWidgets.QWidget()
        self.score_top_spacer.setFixedHeight(0)
        column = QtWidgets.QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        # No inter-widget spacing: the spacer is measured against the gauge
        # figure's own top margin, and anything added to it offsets the
        # picture from the row of gauges it is supposed to line up with.
        column.setSpacing(0)
        column.addWidget(self.score_top_spacer)
        column.addWidget(self.canvas_finalimg)
        column.addSpacing(8)
        column.addWidget(self.lbl_final)
        column.addStretch(1)
        score_holder = QtWidgets.QWidget()
        score_holder.setLayout(column)
        grid.addWidget(score_holder, 0, 2)

        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 3)
        grid.setRowStretch(0, 1)
        self.tab_score = w
        self.tabs.addTab(w, "EEG Score")

    # ---- EMG Analysis tab ----
    def _build_emg(self):
        self.emg_tab = EMGTab(self._emg_config)
        self.emg_tab.status_changed.connect(
            lambda text: self._set_status(emg=text))
        # The EEG half only starts once the EMG half is off the recordings.
        self.emg_tab.analysis_finished.connect(self._emg_finished)
        self.tabs.addTab(self.emg_tab, "EMG Analysis")

    def _set_status(self, eeg=None, emg=None):
        """Report both halves of a run on the start tab's one status line.

        START launches both, and they finish at different times, so neither
        half may overwrite the other's last word.
        """
        if eeg is not None:
            self._eeg_status = eeg
        if emg is not None:
            self._emg_status = emg
        text = self._eeg_status
        if self._emg_status:
            text = f"{text}\nEMG: {self._emg_status}" if text \
                else f"EMG: {self._emg_status}"
        self.lbl_status.setText(text)
        # The line is height-capped, so anything past two lines is readable
        # here rather than lost. Nothing lives only in this label: the EMG
        # half's own words are on its tab, and a failure raises a dialog.
        self.lbl_status.setToolTip(text)

    # ---- persistent parameters ----
    def _setting(self, key, default, value_type):
        """Read one setting, falling back cleanly if stored data is invalid."""
        try:
            return self.settings.value(key, default, type=value_type)
        except (TypeError, ValueError):
            return default

    def _restore_settings(self):
        """Restore the parameters saved on the previous clean window close."""
        self.ed_folder.setText(self._setting(
            "welcome/folder", self.ed_folder.text(), str))
        self.ed_subj.setText(self._setting(
            "welcome/subject", self.ed_subj.text(), str))
        self.ed_real.setText(self._setting(
            "welcome/files_real", self.ed_real.text(), str))
        self.ed_quasi.setText(self._setting(
            "welcome/files_quasi", self.ed_quasi.text(), str))
        self.ed_imag.setText(self._setting(
            "welcome/files_imag", self.ed_imag.text(), str))
        self.ed_emg_left.setText(self._setting(
            "welcome/emg_left_channels", self.ed_emg_left.text(), str))
        self.ed_emg_right.setText(self._setting(
            "welcome/emg_right_channels", self.ed_emg_right.text(), str))

        movement = self._setting(
            "extraction/movement", self.cmb_movement.currentData(), str)
        movement_index = self.cmb_movement.findData(movement)
        if movement_index >= 0:
            self.cmb_movement.setCurrentIndex(movement_index)
        self.spin_marker_shift.setValue(self._setting(
            "extraction/marker_shift",
            self.spin_marker_shift.value(), float))
        self.chk_auto_marker_shift.setChecked(self._setting(
            "extraction/auto_marker_shift",
            self.chk_auto_marker_shift.isChecked(), bool))

        # Everything not on the welcome form comes from the settings file.
        self._load_advanced_settings()

        self.slider.setValue(self._setting(
            "evaluate/component_slider", self.slider.value(), int))

        geometry = self.settings.value("window/geometry")
        remembered = self._setting("window/geometry_version", 0, int)
        if geometry is not None and remembered == GEOMETRY_VERSION:
            self.restoreGeometry(geometry)

        # File type constrains extraction mode and field availability, so apply
        # those relationships once all stored values have been restored.
        self._ftype_changed()
        # The restored lists are the operator's own, even where they came from
        # an earlier detection. Treating the restored folder as already scanned
        # keeps a later click through the folder field from overwriting them.
        self._detected_folder = self._resolved_folder()

    def _save_settings(self):
        """Save the parameters the window itself owns.

        Only the fields the welcome form still exposes are stored here.  The
        rest live in :data:`SETTINGS_FILE_NAME`, which is the single source of
        truth for them: mirroring them into the profile as well would let a
        stale profile silently outrank the file the operator edited.
        """
        values = {
            "welcome/folder": self.ed_folder.text(),
            "welcome/subject": self.ed_subj.text(),
            "welcome/files_real": self.ed_real.text(),
            "welcome/files_quasi": self.ed_quasi.text(),
            "welcome/files_imag": self.ed_imag.text(),
            "welcome/emg_left_channels": self.ed_emg_left.text(),
            "welcome/emg_right_channels": self.ed_emg_right.text(),
            "extraction/movement": self.cmb_movement.currentData(),
            "extraction/marker_shift": self.spin_marker_shift.value(),
            "extraction/auto_marker_shift": self.chk_auto_marker_shift.isChecked(),
            "evaluate/component_slider": self.slider.value(),
            "window/geometry": self.saveGeometry(),
            "window/geometry_version": GEOMETRY_VERSION,
        }
        for key, value in values.items():
            self.settings.setValue(key, value)
        self.settings.sync()

    # ---- advanced parameters (settings file) ----
    def _advanced_fields(self):
        """Settings-file key -> (reader, writer) for each hidden parameter.

        One file for both halves of the run. EEG keys are unprefixed for
        continuity with the recordings already analysed by earlier versions;
        every EMG key carries an ``emg_`` prefix, so which analysis a line
        belongs to is readable without consulting this table.
        """
        return {
            "file_type": _combo_field(self.cmb_ftype),
            "recording_sampling_rate_hz": _combo_field(self.cmb_fs),
            "eeg_channels_first_n": _number_field(self.spin_nchan, int),
            "preparation_excluded_s": _number_field(self.spin_preparation, float),
            "motor_recovery_s": _number_field(self.spin_recovery, float),
            "extra_cut_after_preparation_s": _number_field(
                self.spin_cut_first, float),
            "cut_last_s": _number_field(self.spin_cut_last, float),
            "marker_baseline": _combo_field(self.cmb_marker_baseline),
            "skip_rest_transition_s": _number_field(self.spin_rest_skip, float),
            "tf_marker_padding_s": _number_field(self.spin_marker_padding, float),
            "candidate_detection_scale": _combo_field(self.cmb_candidate_scale),
            "csp_shrinkage": _number_field(self.spin_csp_shrinkage, float),
            "overt_cv_folds": _number_field(self.spin_csp_cv_folds, int),
            "superlet_workers": _number_field(self.spin_superlet_workers, int),
            "analysis_workers": _number_field(self.spin_analysis_workers, int),
            "scoring_version": _combo_field(self.cmb_score_version),
            "emg_envelope": _combo_field(self.cmb_emg_envelope),
            "emg_input_unit": _combo_field(self.cmb_emg_unit),
            "emg_processing_rate_hz": _number_field(self.spin_emg_rate, float),
            "emg_notch_hz": _number_field(self.spin_emg_notch, float),
            "emg_band_low_hz": _number_field(self.spin_emg_band_low, float),
            "emg_band_high_hz": _number_field(self.spin_emg_band_high, float),
            "emg_envelope_window_ms": _number_field(self.spin_emg_window, float),
            "emg_envelope_step_ms": _number_field(self.spin_emg_step, float),
            "emg_plot_before_onset_s": _number_field(self.spin_emg_pre, float),
            "emg_plot_after_movement_s": _number_field(self.spin_emg_post, float),
            "emg_classification_tail_ms": _number_field(self.spin_emg_tail, float),
            "emg_rest_trim_start_s": _number_field(
                self.spin_emg_rest_trim_start, float),
            "emg_rest_trim_end_s": _number_field(
                self.spin_emg_rest_trim_end, float),
            "emg_peak_multiplier": _number_field(self.spin_emg_peak, float),
            "emg_width_multiplier": _number_field(self.spin_emg_width, float),
            "emg_pre_reference_start_ms": _number_field(
                self.spin_emg_preref_start, float),
            "emg_pre_reference_end_ms": _number_field(
                self.spin_emg_preref_end, float),
            "emg_adaptive_peak_multiplier": _number_field(
                self.spin_emg_adaptive_peak, float),
            "emg_adaptive_width_multiplier": _number_field(
                self.spin_emg_adaptive_width, float),
            "emg_min_burst_ms": _number_field(self.spin_emg_min_burst, float),
            "emg_rest_warning_percent": _number_field(
                self.spin_emg_rest_warn, float),
            "emg_save_trial_figures": _flag_field(self.chk_emg_trial_figures),
            "emg_output_root": _text_field(self.ed_emg_output),
        }

    def _load_advanced_settings(self):
        """Apply the settings file, repairing whatever it cannot supply.

        A missing file, an unreadable one, or a single bad value must not stop
        an analysis: the built-in default stands in, the reason is reported on
        the welcome tab, and the repaired file is written back so the operator
        can see the values actually in force.
        """
        fields = self._advanced_fields()
        name = os.path.basename(self.config_path)
        notes = []
        stored = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except (OSError, ValueError) as exc:
                notes.append(f"{name} could not be read ({exc}); defaults used")
            else:
                if isinstance(loaded, dict):
                    stored = loaded
                else:
                    notes.append(
                        f"{name} must contain a JSON object; defaults used")

        for key, (read, write) in fields.items():
            if key not in stored:
                continue
            try:
                write(stored[key])
            except (TypeError, ValueError) as exc:
                notes.append(f"{key}: {exc}; kept {read()!r}")
        for key in sorted(set(stored) - set(fields)):
            notes.append(f"{key}: not a known setting; ignored")

        if self.cmb_score_version.currentData() != LOCKED_SCORING_VERSION:
            fields["scoring_version"][1](LOCKED_SCORING_VERSION)
            notes.append(
                f"scoring_version is locked to {LOCKED_SCORING_VERSION}")

        resolved = {key: read() for key, (read, _) in fields.items()}
        # Purely a scheduling choice -- the transform is bit-identical at any
        # worker count -- so it is applied as soon as it is known rather than
        # threaded through the pipeline call.
        superlet.configure_workers(resolved["superlet_workers"])
        execution.configure_workers(resolved["analysis_workers"])
        if resolved != stored:
            notes.extend(self._write_advanced_settings(resolved))
        self._config_notes = notes
        self._describe_settings_file()
        return resolved

    def _write_advanced_settings(self, resolved):
        """Write the settings file; return any note about failing to."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as handle:
                json.dump(resolved, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as exc:
            return [f"{os.path.basename(self.config_path)} could not be "
                    f"written ({exc})"]
        return []

    def _describe_settings_file(self):
        """Point the operator at the file and at anything wrong with it."""
        text = f"{self.config_path} (edit and restart)"
        if self._config_notes:
            text += "\n" + "\n".join(self._config_notes)
        self.lbl_settings_file.setText(text)
        # The colour is a style-sheet rule rather than an inline one, so the
        # palette stays in a single file; swapping the name the rule matches
        # needs the widget re-polished to take effect.
        self.lbl_settings_file.setObjectName(
            "warning" if self._config_notes else "hint")
        self.lbl_settings_file.style().unpolish(self.lbl_settings_file)
        self.lbl_settings_file.style().polish(self.lbl_settings_file)
        self.lbl_settings_file.setToolTip(
            "Analysis parameters not shown on this form are read from this "
            f"file. The score formula is locked to {LOCKED_SCORING_VERSION}.")

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    # ================= callbacks =================
    def _ftype_changed(self):
        """Each format has exactly one timing source, so pin the selector to it.

        XDF carries event markers in the sample clock; H5 carries the trigger
        channel. Neither can use the other's mechanism, so the mode follows the
        format instead of being a free choice that can be left on a stale value.
        """
        xdf = self.cmb_ftype.currentData() == "xdf"
        self.cmb_extract.setCurrentIndex(
            self.cmb_extract.findData("markers" if xdf else "trigger"))
        self.cmb_extract.setEnabled(False)
        self.cmb_fs.setEnabled(not xdf)
        if xdf:
            self.cmb_fs.setToolTip(
                "Ignored for XDF: each native recording rate is read from the "
                "stream metadata.")
        else:
            self.cmb_fs.setToolTip(
                "Native H5 recording rate used for trigger timing; EEG is "
                "processed at the filter design rate.")
        self.spin_nchan.setEnabled(not xdf)   # XDF matches by label
        candidate_supported = _supports_parameter(
            self.pipe.run_start, "candidate_scale")
        self.cmb_candidate_scale.setEnabled(candidate_supported)
        if not candidate_supported:
            self.cmb_candidate_scale.setToolTip(
                "This pipeline build does not expose candidate-scale selection.")
        shrinkage_supported = _supports_parameter(
            self.pipe.run_start, "csp_shrinkage")
        self.spin_csp_shrinkage.setEnabled(shrinkage_supported)
        if not shrinkage_supported:
            self.spin_csp_shrinkage.setToolTip(
                "This pipeline build does not expose CSP shrinkage.")
        scoring_supported = _supports_parameter(
            self.pipe.compute_ft, "scoring_version")
        self.cmb_score_version.setEnabled(scoring_supported)
        if not scoring_supported:
            self.cmb_score_version.setToolTip(
                "This pipeline build does not expose score-formula selection.")
        cv_supported = _supports_parameter(self.pipe.compute_ft, "csp_cv_folds")
        self.spin_csp_cv_folds.setEnabled(cv_supported)
        if not cv_supported:
            self.spin_csp_cv_folds.setToolTip(
                "This pipeline build cannot cross-validate the overt projection.")
        self._extract_mode_changed()

    def _extract_mode_changed(self):
        """Enable the per-event window fields only in marker mode."""
        markers = self.cmb_extract.currentData() == "markers"
        for w in (self.cmb_movement, self.spin_preparation, self.spin_recovery,
                  self.spin_cut_first, self.spin_cut_last):
            w.setEnabled(markers)
        optional = (
            (self.cmb_marker_baseline, "marker_baseline"),
            (self.spin_rest_skip, "rest_transition_skip"),
            (self.spin_marker_padding, "marker_padding"),
            (self.spin_marker_shift, "marker_shift_s"),
            (self.chk_auto_marker_shift, "auto_marker_shift"),
        )
        for widget, field_name in optional:
            supported = _supports_parameter(ExtractConfig, field_name)
            widget.setEnabled(markers and supported)
            if not supported:
                widget.setToolTip(
                    f"This pipeline build does not support {field_name!r}.")
        self._marker_shift_mode_changed()

    def _marker_shift_mode_changed(self, *_):
        """Grey out the typed shift while it is being detected per recording.

        A build without ``auto_marker_shift`` leaves the box editable: there the
        typed value is the only shift there is.
        """
        if not _supports_parameter(ExtractConfig, "auto_marker_shift"):
            return
        typed_shift_used = not self.chk_auto_marker_shift.isChecked()
        self.spin_marker_shift.setEnabled(
            typed_shift_used
            and self.cmb_extract.currentData() == "markers"
            and _supports_parameter(ExtractConfig, "marker_shift_s"))

    def _extract_config(self):
        return _make_extract_config(
            mode=self.cmb_extract.currentData(),
            active_cond=self.cmb_movement.currentData(),
            preparation_s=self.spin_preparation.value(),
            recovery_s=self.spin_recovery.value(),
            cut_first=self.spin_cut_first.value(),
            cut_last=self.spin_cut_last.value(),
            marker_baseline=self.cmb_marker_baseline.currentData(),
            rest_transition_skip=self.spin_rest_skip.value(),
            marker_padding=self.spin_marker_padding.value(),
            auto_marker_shift=self.chk_auto_marker_shift.isChecked(),
            marker_shift_s=self.spin_marker_shift.value(),
        )

    def _emg_recordings(self):
        """Every recording named on the welcome tab, in the order listed.

        The EMG analysis is not conditional on movement type: overt, quasi and
        imagery runs are all screened for hand EMG, which is precisely how a
        quasi or imagery run is shown to be contaminated. Duplicates are
        dropped so a stem listed twice is not analysed twice.
        """
        names = []
        for edit in (self.ed_real, self.ed_quasi, self.ed_imag):
            for token in edit.text().split(","):
                stem = token.strip()
                if stem and stem not in names:
                    names.append(stem)
        return names

    def _emg_config(self):
        """The EMG run described by the welcome tab and the settings file."""
        return EMGConfig(
            data_dir=self._resolved_folder(),
            recordings=self._emg_recordings(),
            participant=self.ed_subj.text().strip(),
            target_fs=self.spin_emg_rate.value() or None,
            left_channels=parse_pair(self.ed_emg_left.text()),
            right_channels=parse_pair(self.ed_emg_right.text()),
            left_condition=LEFT_CONDITION,
            right_condition=RIGHT_CONDITION,
            rest_condition=REST_CONDITION,
            envelope=self.cmb_emg_envelope.currentData(),
            input_unit=self.cmb_emg_unit.currentData(),
            notch_hz=self.spin_emg_notch.value(),
            band_low_hz=self.spin_emg_band_low.value(),
            band_high_hz=self.spin_emg_band_high.value(),
            window_ms=self.spin_emg_window.value(),
            step_ms=self.spin_emg_step.value(),
            pre_s=self.spin_emg_pre.value(),
            post_s=self.spin_emg_post.value(),
            trial_tail_s=self.spin_emg_tail.value() / 1000.0,
            rest_trim_start_s=self.spin_emg_rest_trim_start.value(),
            rest_trim_end_s=self.spin_emg_rest_trim_end.value(),
            peak_multiplier=self.spin_emg_peak.value(),
            background_multiplier=self.spin_emg_width.value(),
            pre_reference_start_s=self.spin_emg_preref_start.value() / 1000.0,
            pre_reference_end_s=self.spin_emg_preref_end.value() / 1000.0,
            secondary_pre_multiplier=self.spin_emg_adaptive_peak.value(),
            secondary_width_multiplier=self.spin_emg_adaptive_width.value(),
            min_burst_ms=self.spin_emg_min_burst.value(),
            rest_fpr_warn=self.spin_emg_rest_warn.value() / 100.0,
            # The same correction the EEG extraction applies, from the same
            # two controls: one run, one time base.
            auto_marker_shift=self.chk_auto_marker_shift.isChecked(),
            marker_shift_s=self.spin_marker_shift.value(),
            trial_figures=self.chk_emg_trial_figures.isChecked(),
            output_root=self.ed_emg_output.text().strip().strip('"'),
        )

    def _selected_component(self):
        """Return the 1-based component represented by the inverted slider."""
        return self.slider.maximum() - round(self.slider.value()) + 1

    def _set_selected_component(self, component):
        """Point the slider at a 1-based component, undoing its inversion."""
        component = min(max(1, int(component)), self.slider.maximum())
        self.slider.setValue(self.slider.maximum() - component + 1)

    def _configure_component_selector(self, result):
        """Clamp the selector to the components actually returned by CSP."""
        old_component = self._selected_component()
        count = _component_count(result)
        self._n_display_components = count
        self.slider.blockSignals(True)
        self.slider.setMinimum(1)
        self.slider.setMaximum(max(1, count))
        self._set_selected_component(old_component)
        self.slider.setEnabled(count > 1)
        self.slider.blockSignals(False)
        # Fewer components means fewer bands to span, so the inset changes too.
        self._align_slider_to_topos()
        return count

    def _auto_select_component(self, result, count):
        """Preselect the component to inspect first, once START has finished.

        Contralateral mu-ERD is what the analysis is looking for; mu-ERS and
        then ipsilateral mu-ERD are the fallbacks.  When none of the three can
        be recognised the slider stays on component 1 and the operator is told
        that the choice is theirs -- the eigenvalue and the scalp pattern are
        the only evidence available before the time-frequency step, so this
        preselection is a starting point, never a result.
        """
        self._auto_selection = None
        self._update_component_selection_summary()
        if count <= 0:
            return None
        patterns = np.asarray(getattr(result, "top_patterns", []))
        labels = list(getattr(result, "channel_labels", None) or [])
        evals = np.asarray(getattr(result, "evals", []))
        if (patterns.ndim != 2 or patterns.shape[1] < count
                or patterns.shape[0] != len(labels) or evals.size < count):
            return None
        # The historical template sharpens the plausibility ranking, but a
        # montage it does not cover is a reason to drop it, not to fail.
        ideal = getattr(self.pipe, "ideal_pattern", None)
        ideal = None if ideal is None else np.asarray(ideal, float).ravel()
        if ideal is not None and ideal.size != len(labels):
            ideal = None
        selection = qc.select_motor_component(
            patterns[:, :count], labels, evals[:count],
            active_condition=getattr(self.pipe.extract, "active_cond", ""),
            ideal_pattern=ideal, ideal_labels=labels)
        self._auto_selection = selection
        self._set_selected_component(selection.component)
        self._update_component_selection_summary()
        return selection

    def _update_component_selection_summary(self):
        """Keep preselection evidence visible, including after a manual choice."""
        selection = self._auto_selection
        if selection is None:
            text = ("Run START to find a contralateral mu-ERD candidate."
                    if self.start_result is None else
                    "Automatic component selection unavailable; inspect "
                    "the components by hand.")
            tooltip = ""
        else:
            text = selection.summary
            component = self._selected_component()
            if component != selection.component:
                text = f"Manual selection: component {component}. " + text
            tooltip = qc.summarize_component_responses(selection.responses)
        self.lbl_component_selection.setText(text)
        self.lbl_component_selection.setToolTip(tooltip)

    def _warn_manual_component_selection(self, message):
        QtWidgets.QMessageBox.information(
            self, "Automatic component selection", message)

    def _resolved_folder(self):
        """The recording folder the template and subject name point at."""
        return self.ed_folder.text().replace(
            "SUBJECT_NAME", self.ed_subj.text()).strip().strip('"')

    def _browse_folder(self):
        """Select a recording folder directly, including a session subfolder."""
        initial = self._resolved_folder()
        # Start at the nearest existing parent when a template names a
        # participant whose folder does not exist yet.
        while initial and not os.path.isdir(initial):
            parent = os.path.dirname(initial)
            if parent == initial:
                initial = ""
                break
            initial = parent
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select participant recording folder", initial)
        if not folder:
            return
        folder = os.path.normpath(folder)
        self.ed_folder.setText(folder)
        self.ed_subj.setText(participant_code(folder))
        self._folder_changed()

    def _folder_changed(self):
        """React to a new folder or subject by listing that folder's files.

        ``editingFinished`` fires on leaving a field whether or not anything
        changed, so detection is tied to the resolved path actually differing.
        Lists edited by hand therefore survive clicking between fields, and
        'detect from folder' is the way to ask for them again.
        """
        folder = self._resolved_folder()
        if folder and folder != getattr(self, "_detected_folder", None):
            self._detect_recordings()

    def _detect_recordings(self):
        folder = self._resolved_folder()
        extension = _EXTENSIONS[self.cmb_ftype.currentData()]
        found = discover_recordings(folder, extension)
        self._detected_folder = folder
        if not found.total:
            # Leave the lists alone: an empty result is much more often a
            # half-typed path or an unmounted drive than an empty session.
            self.lbl_detected_files.setText(
                f"No {extension} recordings recognised in {folder or 'that path'}; "
                f"the three lists are unchanged.")
            return
        for edit, condition in ((self.ed_real, "overt"),
                                (self.ed_quasi, "quasi"),
                                (self.ed_imag, "imagery")):
            edit.setText(", ".join(found[condition]))
        self.lbl_detected_files.setText(found.summary())

    def on_start(self):
        folder = self._resolved_folder()
        file_type = self.cmb_ftype.currentData()
        fs = float(self.cmb_fs.currentData())
        ch_num = self.spin_nchan.value()
        if (file_type == "h5" and
                not MIN_H5_CHANNELS <= ch_num <= MAX_REFERENCE_CHANNELS):
            QtWidgets.QMessageBox.warning(
                self, "Invalid H5 montage",
                f"H5 channel count must be between {MIN_H5_CHANNELS} and "
                f"{MAX_REFERENCE_CHANNELS}.")
            return
        ext = _EXTENSIONS[file_type]
        fr = _split_files(self.ed_real.text(), folder, ext)
        fq = _split_files(self.ed_quasi.text(), folder, ext)
        fi = _split_files(self.ed_imag.text(), folder, ext)
        extract = self._extract_config()
        self._input_files = {"overt": fr, "quasi": fq, "imagery": fi}
        missing = [f for f in fr + fq + fi if not os.path.exists(f)]
        if missing:
            QtWidgets.QMessageBox.warning(self, "Missing files",
                "These files were not found:\n" + "\n".join(missing[:8]))
            return
        self.btn_start.setEnabled(False)
        self._clear_eeg_results()
        # The two analyses read the same files, so they are run one after the
        # other rather than at once: two full XDF loads in flight would double
        # the peak memory of a session for no gain. EMG goes first because it
        # finishes in a fraction of the time the CSP fit needs, so the
        # contamination verdict is readable while the EEG half is still going.
        self._pending_eeg = dict(
            fr=fr, fq=fq, fi=fi, extract=extract, file_type=file_type,
            ch_num=ch_num, fs=fs)
        self._set_status(eeg="queued behind the EMG screening…")
        if not self._start_emg():
            self._start_eeg()

    def _start_emg(self):
        """Hand the start tab's settings to the EMG half of the run.

        Returns whether an EMG analysis is now running, so a run that carries
        none does not leave the EEG half waiting for a signal never sent.
        """
        if self.cmb_ftype.currentData() != "xdf":
            self.emg_tab.skip(
                "EMG analysis needs XDF recordings; this run is H5, which "
                "carries no auxiliary EMG channels.")
            return False
        return self.emg_tab.start()

    def _emg_finished(self, analysed):
        """The EMG half is off the recordings: show it, then start the EEG half.

        A failed EMG analysis is no reason to withhold the transfer score, so
        the EEG half starts either way; only a finished one raises its tab.
        """
        if analysed:
            self.tabs.setCurrentWidget(self.emg_tab)
        self._start_eeg()

    def _start_eeg(self):
        """Run the EEG half on the parameters START captured."""
        params = self._pending_eeg
        self._pending_eeg = None
        if params is None:
            return
        self._set_status(eeg="crunching… (robust cov + CSP)")
        self.worker = StartWorker(
            self.pipe, params["fr"], params["fq"], params["fi"],
            params["extract"], params["file_type"], params["ch_num"],
            params["fs"],
            processing_fs=1000.0,
            candidate_scale=self.cmb_candidate_scale.currentData(),
            csp_shrinkage=self.spin_csp_shrinkage.value())
        self.worker.done.connect(self._start_done)
        self.worker.failed.connect(self._worker_failed)
        self.worker.start()

    def _start_done(self, sr):
        self.start_result = sr
        self.btn_start.setEnabled(True)
        offsets_map = getattr(sr, "video_onset_offsets", {})
        warning = _first_metadata(
            (sr,), "condition_warning", "conditioning_warning")
        if offsets_map:
            # Informational only; shown on the same corrected marker time axis.
            offsets = ", ".join(
                f"{name} {value:+.3f}s"
                for name, value in offsets_map.items())
            self._set_status(eeg=f"done. video_onset at: {offsets}")
        elif warning:
            self._set_status(eeg="done with a covariance-conditioning warning.")
        else:
            self._set_status(eeg="done.")
        n_components = self._configure_component_selector(sr)
        selection = self._auto_select_component(sr, n_components)
        self._plot_evals(sr)
        self._plot_topos(sr)
        self.btn_ft.setEnabled(n_components > 0)
        self.tabs.setCurrentWidget(self.tab_components)
        # Raised last, over the maps the operator now has to judge by hand.
        if selection is not None and not selection.automatic:
            self._warn_manual_component_selection(selection.message)

    def on_ft(self):
        if self.start_result is None:
            return
        comp = self._selected_component()
        if not 1 <= comp <= self._n_display_components:
            QtWidgets.QMessageBox.warning(
                self, "Invalid component",
                "The selected CSP component is not available in this result.")
            return
        self.btn_ft.setEnabled(False)
        self._set_status(eeg="computing time-frequency…")
        self.ftworker = FTWorker(
            self.pipe, comp, self.cmb_score_version.currentData(),
            self.spin_csp_cv_folds.value())
        self.ftworker.done.connect(self._ft_done)
        self.ftworker.failed.connect(self._worker_failed)
        self.ftworker.start()

    def _ft_done(self, ft):
        self.ft_result = ft
        self.btn_ft.setEnabled(True)
        self.btn_rating.setEnabled(True)
        self._plot_tf(ft)
        self._plot_scores(ft)
        self._set_status(eeg="done.")

    def _worker_failed(self, tb):
        self.btn_start.setEnabled(True)
        self.btn_ft.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Error", tb)

    # ================= plotting =================
    def _blank_score_image(self):
        """The cat's panel, before there is a score for a cat to stand for."""
        _blank_axes(self.ax_finalimg, PLACEHOLDER_SCORE_IMAGE, fontsize=11)

    def _blank_topos(self):
        """Empty the five map bands, keeping the 1-5 numbers beside them.

        The hint goes on the middle band rather than on the figure, because
        ``_plot_topos`` clears every band and would leave a figure-level text
        printed over the first real map.
        """
        for ax in self.ax_topos:
            ax.clear()
            ax.set_axis_off()
        middle = self.ax_topos[len(self.ax_topos) // 2]
        middle.text(0.5, 0.5, PLACEHOLDER_TOPOS, transform=middle.transAxes,
                    ha="center", va="center", fontsize=9,
                    color=PALETTE["faint"], linespacing=1.6)
        self._draw_topo_scale()

    def _clear_eeg_results(self):
        """Take the previous run's panels down before the next one starts.

        The maps, the gauges and the total belong to the recordings they were
        computed from. Left on screen while the next participant is being
        analysed they are read as that participant's, which is the same fault
        as an empty axis inventing a scale: something on screen that no
        finished calculation stands behind.
        """
        self.start_result = None
        self.ft_result = None
        self._auto_selection = None
        self._update_component_selection_summary()
        self.btn_ft.setEnabled(False)
        self.btn_rating.setEnabled(False)

        _blank_axes(self.ax_evals, PLACEHOLDER_EVALS)
        self.canvas_evals.draw_idle()
        for name, ax in zip(TF_CONDITION_NAMES, self.ax_tf):
            _blank_axes(ax, PLACEHOLDER_TF.format(name=name))
        self.canvas_tf.draw_idle()
        self._blank_topos()
        self.canvas_topos.draw_idle()
        for number, ax in enumerate(self.ax_scores, start=1):
            _blank_axes(ax, PLACEHOLDER_GAUGE.format(number=number), fontsize=8)
        self.canvas_scores.draw_idle()
        self._blank_score_image()
        self.canvas_finalimg.draw_idle()
        self.lbl_final.setText(NO_SCORE_TEXT)

    def _plot_evals(self, sr):
        ax = self.ax_evals
        _restore_axes(ax)
        # The spectrum is a curve, not a picture, so it is read off a left and
        # a bottom rule; a full box only encloses white space above it.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        evals = np.asarray(getattr(sr, "evals", []))
        low = np.asarray(getattr(sr, "ok_low_inds", []), dtype=int)
        high = np.asarray(getattr(sr, "ok_high_inds", []), dtype=int)
        ax.plot(np.arange(1, evals.size + 1), evals, "-*",
                color=PALETTE["text"], label="evals")
        low = low[(low >= 0) & (low < evals.size)]
        high = high[(high >= 0) & (high < evals.size)]
        if low.size:
            ax.plot(low + 1, evals[low], "o", ms=9, color=PALETTE["accent"],
                    label="ERD OK")
        if high.size:
            ax.plot(high + 1, evals[high], "o", ms=9, color=PALETTE["mark"],
                    label="ERS OK")
        ax.set_title("GED eigenvalues")
        ax.set_xlabel("comp. #")
        if evals.size:
            # The first and last components are exactly the ones marked as ERD
            # and ERS candidates, and a marker centred on the axis limit is
            # drawn half outside the spine, so pad both ends. The candidates
            # can fall anywhere on the curve and this panel is short, so the
            # legend goes beside the axes rather than into a corner it would
            # sooner or later cover a marker in.
            ax.set_xlim(0.4, evals.size + 0.6)
            ax.margins(y=0.12)
            self._evals_legend = ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                borderaxespad=0.0)
        ax.grid(True, axis="y")
        # Drawn once before aligning: an unrendered legend cannot report the
        # width the two panels have to reserve for it.
        self.canvas_evals.draw()
        self._align_eeg_axes()

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Resize:
            if obj is self.canvas_topos:
                self._align_slider_to_topos()
            elif obj in (self.canvas_evals, self.canvas_tf):
                self._align_eeg_axes()
            elif obj in (self.canvas_scores, self.canvas_finalimg):
                # Deferred: this handler runs before the canvas's own, and
                # resizing it here would deliver the now-stale event to
                # matplotlib afterwards, leaving the figure sized for a
                # geometry the widget no longer has.
                QtCore.QTimer.singleShot(0, self._align_score_image)
        return super().eventFilter(obj, event)

    def _legend_width_px(self):
        """Width the eigenvalue legend needs beside its axes."""
        legend = getattr(self, "_evals_legend", None)
        if legend is None:
            return EEG_LEGEND_FALLBACK_PX
        try:
            renderer = self.canvas_evals.get_renderer()
            return float(legend.get_window_extent(renderer).width)
        except Exception:
            # A legend that has not been rendered yet cannot measure itself;
            # its contents are fixed, so the nominal width is a fair stand-in.
            return EEG_LEGEND_FALLBACK_PX

    def _align_eeg_axes(self):
        """Give the eigenvalue box and the three TF boxes one common x-span.

        The two figures live in the same grid columns and so are always the
        same number of pixels wide; converting one pair of pixel margins into
        each figure's own fractions puts all four boxes exactly under one
        another, at any window size. Vertical margins are left to differ,
        because the two panels carry different amounts of furniture.
        """
        width = self.canvas_evals.width()
        if width <= 1 or self.canvas_tf.width() != width:
            return
        left = EEG_AXES_LEFT_PX / width
        right = 1.0 - (self._legend_width_px() + EEG_LEGEND_PAD_PX) / width
        if not 0.0 < left < right < 1.0:
            return

        height = max(1, self.canvas_evals.height())
        self.canvas_evals.fig.subplots_adjust(
            left=left, right=right,
            top=1.0 - 26.0 / height, bottom=min(0.6, 42.0 / height))
        self.canvas_evals.draw_idle()

        height = max(1, self.canvas_tf.height())
        # Each map carries its own title, so the gap between them is a fixed
        # number of pixels rather than a share of the panel.
        top, bottom = 1.0 - 20.0 / height, min(0.4, 44.0 / height)
        gap = 24.0
        axes_height = max(1.0, ((top - bottom) * height - 2 * gap) / 3.0)
        self.canvas_tf.fig.subplots_adjust(
            left=left, right=right, top=top, bottom=bottom,
            hspace=gap / axes_height)
        self.canvas_tf.draw_idle()

    def _score_image_aspect(self):
        """Height/width of the score picture, so its canvas can match it."""
        shape = getattr(getattr(self.pipe, "score_img", None), "shape", None)
        if shape and len(shape) >= 2 and shape[1]:
            return float(shape[0]) / float(shape[1])
        return 1.0

    def _align_score_image(self):
        """Start the picture on the top edge of the first row of gauges.

        Both sit in the same grid row, so a spacer as deep as the gauge
        figure's own top margin is exactly the offset needed. Pinning the
        canvas to the picture's proportions then leaves no slack between the
        picture and the total underneath it.
        """
        top = round((1.0 - SCORE_AXES_TOP) * self.canvas_scores.height())
        if self.score_top_spacer.height() != top:
            self.score_top_spacer.setFixedHeight(max(0, top))
        width = self.canvas_finalimg.width()
        height = round(width * self._score_image_aspect())
        if height > 0 and self.canvas_finalimg.height() != height:
            self.canvas_finalimg.setFixedHeight(height)
            # The picture was rendered for the height the canvas had before
            # this call; without a redraw the widget keeps showing it.
            self.canvas_finalimg.draw_idle()

    def _topo_band_centers(self):
        """Vertical centre of each topography as a fraction of canvas height."""
        centers = []
        for ax in self.ax_topos:
            box = ax.get_position()
            centers.append(1.0 - (box.y0 + box.height / 2.0))
        return centers

    def _slider_handle_length(self):
        option = QtWidgets.QStyleOptionSlider()
        self.slider.initStyleOption(option)
        return self.slider.style().pixelMetric(
            QtWidgets.QStyle.PixelMetric.PM_SliderLength, option, self.slider)

    def _align_slider_to_topos(self):
        """Inset the slider so every position lands on its own topography.

        A QSlider spreads its positions across the groove minus one handle
        length.  That span is taller than the five topography bands, so the
        extreme positions sat most of a half-band away from the map they
        select -- the handle pointed at the gap above map 1 while map 1 itself
        was drawn further down.  Padding the slider's holder by the distance
        between the canvas edge and the first/last band centre makes the
        handle travel coincide with the bands, and the tick marks with it.
        """
        parent = self.slider_holder.parentWidget()
        height = self.canvas_topos.height()
        if parent is None or height <= 1 or not self.ax_topos:
            return
        count = max(1, min(self._n_display_components, len(self.ax_topos)))
        centers = self._topo_band_centers()
        origin = QtCore.QPoint(0, 0)
        offset = (self.canvas_topos.mapTo(parent, origin).y()
                  - self.slider_holder.mapTo(parent, origin).y())
        handle = self._slider_handle_length()
        top = offset + centers[0] * height - handle / 2.0
        bottom = (self.slider_holder.height()
                  - (offset + centers[count - 1] * height) - handle / 2.0)
        layout = self.slider_holder.layout()
        current = layout.contentsMargins()
        top, bottom = max(0, round(top)), max(0, round(bottom))
        if (current.top(), current.bottom()) != (top, bottom):
            layout.setContentsMargins(0, top, 0, bottom)

    def _draw_topo_scale(self):
        """(Re)create the 1-5 component numbers aligned with each topography.

        They live in the reserved left margin of the topo figure so they line up
        exactly with the maps. ``topo.draw`` clears the axes, so this is called
        again after (re)plotting the maps.
        """
        self._topo_labels = []
        for i, ax in enumerate(self.ax_topos[:self._n_display_components]):
            t = ax.text(-0.16, 0.5, str(i + 1), transform=ax.transAxes,
                        ha="center", va="center", fontsize=12,
                        fontweight="bold", color=PALETTE["faint"],
                        clip_on=False)
            self._topo_labels.append(t)

    def _plot_topos(self, sr):
        count = self._n_display_components
        patterns = np.asarray(getattr(sr, "top_patterns", []))
        labels = getattr(sr, "channel_labels", None)
        for i, ax in enumerate(self.ax_topos):
            ax.set_visible(i < count)
            if i < count:
                topo.draw(ax, patterns[:, i], labels=labels)
            else:
                ax.clear()
        self._draw_topo_scale()          # topo.draw() cleared the axes
        self._highlight_selected_topo()
        self.canvas_topos.draw()

    def _highlight_selected_topo(self):
        if self.start_result is None:
            return
        self._update_component_selection_summary()
        sel = self._selected_component()
        for i, ax in enumerate(self.ax_topos):
            on = i < self._n_display_components and (i + 1 == sel)
            # ``topo.draw`` switches the axes off, and a switched-off axes
            # draws no spines however visible they are made -- which is why
            # the frame this has always asked for never appeared. The one map
            # that carries it therefore has its axes switched back on, with
            # empty ticks so nothing but the frame comes back with it.
            if on:
                ax.set_axis_on()
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.set_axis_off()
            for sp in ax.spines.values():
                sp.set_visible(on)
                sp.set_color(PALETTE["accent"])
                sp.set_linewidth(2)
            if i < len(self._topo_labels):
                lbl = self._topo_labels[i]
                lbl.set_color(PALETTE["accent"] if on else PALETTE["faint"])
                lbl.set_fontsize(16 if on else 12)
        self.canvas_topos.draw()

    def _plot_tf(self, ft):
        names = TF_CONDITION_NAMES
        tvec = self.pipe.tvec
        f = self.pipe.freqNeeded
        # Trigger mode keeps the fixed -2..5.5 s view; marker windows start at
        # the (trimmed) task onset, so show their full 0..L span instead.
        xlim = (float(tvec[0]), float(tvec[-1])) \
            if self.pipe.extract.mode == "markers" else (-2, 5.5)
        for i, ax in enumerate(self.ax_tf):
            _restore_axes(ax)
            m = ft.tf_norm[i]
            ax.contourf(tvec, f, m, 60, cmap="jet", vmin=-7, vmax=7)
            ax.set_xlim(*xlim)
            ax.axvline(0, color="k")
            if self.pipe.extract.mode == "markers":
                motor_end = getattr(self.pipe, "_motor_duration_s", None)
                if motor_end is not None:
                    ax.axvline(
                        motor_end, color="k", linestyle="--", linewidth=0.8)
            ax.set_title(names[i], fontsize=9)
            ax.set_ylabel("Hz")
            # One shared time axis under three stacked maps: repeating its
            # labels between them only crowds the next map's title.
            ax.tick_params(labelbottom=i == len(self.ax_tf) - 1)
        self.ax_tf[-1].set_xlabel("time [s]")
        self._align_eeg_axes()
        self.canvas_tf.draw()

    def _plot_scores(self, ft):
        s = ft.score
        specs = _score_gauge_specs(s)
        for i, ax in enumerate(self.ax_scores):
            spec = dict(specs[i])
            spec["xlabel"] = (
                f"score {int(s.scores[i])}/{int(s.scores_max[i])}")
            _draw_gauge(ax, spec)
        self.canvas_scores.draw()

        self.ax_finalimg.clear()
        self.ax_finalimg.axis("off")
        self.ax_finalimg.set_anchor("S")   # clear() resets it to centred
        if s.total > 0:
            idx = min(round(s.total) - 1, self.pipe.score_img.shape[3] - 1)
            self.ax_finalimg.imshow(self.pipe.score_img[:, :, :, idx])
        else:
            # The bundled cat stack starts at 1/15.  Showing its first frame for
            # zero points falsely implies a nonzero result, so keep this panel
            # deliberately neutral and explicit.
            self.ax_finalimg.text(
                0.5, 0.5, "0 / 15\nNo score image",
                transform=self.ax_finalimg.transAxes,
                ha="center", va="center", color=PALETTE["muted"], fontsize=16,
                linespacing=1.5)
        self.canvas_finalimg.draw()
        self.lbl_final.setText(f"Total score: {s.total} / {s.total_max}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    icon = asset_path("klh_intro.png")
    if os.path.exists(icon):
        app.setWindowIcon(QtGui.QIcon(icon))
    win = NeuroCastingApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
