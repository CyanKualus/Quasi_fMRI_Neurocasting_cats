"""End-to-end orchestration — port of STARTButtonPushed + CalculateFTButtonPushed.

Encapsulates the analysis state that the MATLAB app kept as properties and
exposes two stages:

* :meth:`run_start`  -> load/filter/epoch, robust covariances, CSP, eigenvalue
  candidate detection, up to five forward patterns.
* :meth:`compute_ft` -> project the chosen component, superlet TF maps for the
  three conditions, and the six sub-scores.
"""
from __future__ import annotations

import os
import copy
import hashlib
from dataclasses import asdict, dataclass, field
import numpy as np

from . import io_h5, filters, csp, superlet, qc, execution
from shared.marker_shift import resolve_marker_shift
from .robustcov import olivehawkins
from .scoring import compute_scores, ScoreResult, SCORING_V3, SCORING_V4
from .mathutil import mad1, rng_inclusive
from .assets import load_helper, load_ced

# Condition indices
REAL, QUASI, IMAG = 0, 1, 2
ACTIVE, PASSIVE = 0, 1
CONDITION_NAMES = ("real", "quasi", "imag")
EPOCH_NAMES = ("active", "rest")
DATA_PATH_VERSION = "klh-data-v4"
CONDITION_WARNING_THRESHOLD = 1e12
# Folds used to re-estimate the CSP filter without the overt trials it is then
# measured on.  Five keeps 80% of the overt data in every training fit while
# adding five robust-covariance pairs to the run.
DEFAULT_CSP_CV_FOLDS = 5
# A fold whose best-matching component barely beats the runner-up has no stable
# counterpart to the operator's selection; the run reports it rather than
# silently averaging unrelated components.
CV_MATCH_MARGIN_WARNING = 0.10


@dataclass
class ExtractConfig:
    """How epochs are cut out of each recording.

    ``trigger`` reproduces the original behaviour (trigger channel + fixed 8 s
    epoch) and needs the H5 trigger column. ``markers`` uses the event markers
    embedded in an XDF recording. An LSL timing correction is added to every
    marker onset; with ``auto_marker_shift`` it is measured separately from
    each recording (see :mod:`shared.marker_shift`), otherwise the single
    ``marker_shift_s`` is applied to all of them. Active events contain a
    preparation interval before movement and retain real post-event data for
    movement recovery. Their core is therefore
    ``[onset+preparation_s+cut_first,
       onset+duration-cut_last+recovery_s]``.
    Rest events keep their full marker-defined window. Event durations are
    taken from the gap to the next marker and are not shifted.
    """
    mode: str = "trigger"                 # "trigger" | "markers"
    active_cond: str = "right_microrepeat"
    rest_cond: str = "rest"
    preparation_s: float = 2.0             # excluded before motor-task time zero
    recovery_s: float = 2.0                # retained after the marker-defined trial
    cut_first: float = 0.0                 # optional extra trim after preparation
    cut_last: float = 0.0                  # seconds trimmed from each window end
    marker_baseline: str = "condition-specific"  # "condition-specific" | "pooled"
    rest_transition_skip: float = 1.0      # seconds excluded at each rest start
    marker_padding: float = 1.5            # real context on both sides for TF
    auto_marker_shift: bool = True         # measure the shift from each file
    marker_shift_s: float = -4.0           # added to active and rest onsets


@dataclass
class StartResult:
    evals: np.ndarray
    projInverse: np.ndarray
    projForward: np.ndarray
    ok_low_inds: np.ndarray          # 0-based eigenvalue indices (ERD candidates)
    ok_high_inds: np.ndarray         # 0-based eigenvalue indices (ERS candidates)
    n_erd: int
    top_patterns: np.ndarray         # (n_ch, min(5, n_components))
    video_onset_offsets: dict[str, float] = field(default_factory=dict)
    channel_labels: list[str] = field(default_factory=list)
    n_components: int = 0
    condition_number: float = np.nan
    condition_warning: str | None = None
    trial_counts: dict = field(default_factory=dict)
    component_ranking: list[dict] = field(default_factory=list)
    channel_qc: dict = field(default_factory=dict)
    file_epoch_slices: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


@dataclass
class FTResult:
    tf_norm: list                    # 3 x (n_freq, n_time) dB maps
    score: ScoreResult
    selected_comp: int               # 1-based
    pattern: np.ndarray
    normalization: str = ""
    trial_counts: dict = field(default_factory=dict)
    physiological_qc: qc.PhysiologicalQCResult | None = None
    provenance: dict = field(default_factory=dict)
    # How the overt condition was projected: "cross-validated-kfold" means its
    # peak ERD is a held-out estimate comparable with quasi and imagery.
    overt_projection: str = "in-sample"
    csp_cv: dict = field(default_factory=dict)


class KLHPipeline:
    def __init__(self):
        h = load_helper()
        self.b1, self.b2, self.bpa = h["b1"], h["b2"], h["bpa"]
        self._ideal_pattern_full = h["ideal_pattern"]
        self.ideal_pattern = self._ideal_pattern_full.copy()
        self.score_img = h["score_img"]              # (400,400,3,15)
        self.subj_dB = h["subj_dB"]
        self.anatoly_scores_all = h["anatoly_scores_all"]

        self.freqNeeded = np.arange(3, 30.0001, 0.5)  # 3:0.5:30
        self.ch_num = io_h5.N_EEG
        self.file_type = "h5"
        self.extract = ExtractConfig()
        # sampling rate + the fixed -2.5..5.5 s epoch window derived from it
        # (self.fs, self.epoind, self.tvec, self.tvec_active/rest, self._La/_Lr).
        self._configure_fs(1000.0)
        self.processing_fs = filters.FS_DEFAULT
        self.recording_fs = filters.FS_DEFAULT

        # state populated by run_start
        self._xdf_pick = None            # electrode names to select for XDF caps
        self.channel_labels = list(load_ced()["labels"])
        self.broad = None
        self.alpha = None
        self.projInverse = None
        self.projForward = None
        self.evals = None
        self.n_erd = 0
        self.video_onset_offsets = {}
        self._marker_shifts = {}         # abspath -> shift actually applied
        self._marker_shift_notes = []    # one human-readable line per file
        self._pad = 0
        self._required_tf_pad = 0
        self.condition_number = np.nan
        self.condition_warning = None
        self.candidate_scale = "raw"
        self.candidate_diagnostic = {}
        self.csp_shrinkage = 0.0
        self.trial_counts = _empty_trial_counts()
        self.component_ranking = []
        self.channel_qc = {}
        self.physiological_qc = None
        self._recording_details = []
        self._file_epoch_slices = []
        self._window_diagnostics = []
        self._montage_details = {}
        self.provenance = {}

    def _filter_edge_margin(self) -> int:
        """Samples of real signal a cropped recording must keep on each side.

        ``filtfilt`` runs the FIR forward and backward, so an output sample
        depends on inputs within ``taps - 1`` either way; the broadband chain
        applies ``b2`` then ``b1``, and those supports add.  Keeping this much
        beyond the epoched span makes every epoched sample identical to
        filtering the recording whole, and the ``padlen`` transient never
        reaches it because scipy pads a further ``3 * taps`` of its own.
        """
        chains = (int(self.b1.size) - 1 + int(self.b2.size) - 1,
                  int(self.bpa.size) - 1)
        return max(chains)

    def _core_epochs(self, epochs: np.ndarray, kind: int) -> np.ndarray:
        """Return the unpadded task/rest core used by covariance estimates."""
        if epochs is None or epochs.shape[1] == 0:
            raise ValueError(f"no {EPOCH_NAMES[kind]} epochs available")
        if self.extract.mode != "markers":
            return epochs
        core_len = self._La if kind == ACTIVE else self._Lr
        expected = core_len + 2 * self._pad
        if epochs.shape[0] != expected:
            raise ValueError(
                f"unexpected padded {EPOCH_NAMES[kind]} length "
                f"{epochs.shape[0]} (expected {expected})")
        return epochs[self._pad:self._pad + core_len]

    def _normalization_mode(self) -> str:
        if self.extract.mode != "markers":
            return "condition-specific-prestimulus"
        if self.extract.marker_baseline == "pooled":
            return "pooled-marker-rest"
        return "condition-specific-marker-rest"

    def _build_provenance(self) -> dict:
        """Return a serialisable description of every behavior-changing path."""
        if self.extract.mode == "markers":
            path_name = "xdf-marker-padded"
        else:
            path_name = "h5-native-trigger-eeg-resample"
        coeff_bytes = b"".join(
            np.ascontiguousarray(np.asarray(b, dtype=np.float64)).tobytes()
            for b in (self.b1, self.b2, self.bpa))
        filter_hash = hashlib.sha256(coeff_bytes).hexdigest()
        rate_signature = sorted({
            (float(row["recording_fs"]), bool(row["resampled"]))
            for row in self._recording_details
        })
        # The readable path name is paired with a short deterministic digest.
        # Any behavior-changing rate/filter/epoch/baseline/montage choice thus
        # produces a different identifier even when the score formula is fixed.
        path_signature = repr((
            path_name,
            float(self.fs),
            rate_signature,
            filter_hash,
            self.extract.active_cond,
            self.extract.rest_cond,
            float(self.extract.preparation_s)
            if self.extract.mode == "markers" else 0.0,
            float(self.extract.recovery_s)
            if self.extract.mode == "markers" else 0.0,
            float(self.extract.cut_first),
            float(self.extract.cut_last),
            self.extract.marker_baseline if self.extract.mode == "markers" else None,
            float(self.extract.rest_transition_skip)
            if self.extract.mode == "markers" else 0.0,
            float(self.extract.marker_padding)
            if self.extract.mode == "markers" else 0.0,
            self._marker_shift_signature(),
            tuple(self.channel_labels),
            self.candidate_scale,
            float(self.csp_shrinkage),
        )).encode("utf-8")
        path_digest = hashlib.sha256(path_signature).hexdigest()[:12]
        path_id = f"{DATA_PATH_VERSION}:{path_name}:{path_digest}"
        provenance = {
            "data_path_version": path_id,
            "pipeline_version": DATA_PATH_VERSION,
            "file_type": self.file_type,
            "extraction_mode": self.extract.mode,
            "input_files": copy.deepcopy(self._input_files),
            "processing_fs": float(self.fs),
            "recording_fs": (
                float(self.recording_fs)
                if self.recording_fs is not None else "per-file-xdf-metadata"),
            "recordings": copy.deepcopy(self._recording_details),
            "file_epoch_slices": copy.deepcopy(self._file_epoch_slices),
            "filters": {
                "source": "prebuilt-helper-1000Hz",
                "design_fs": float(filters.FS_DEFAULT),
                "sha256": filter_hash,
            },
            "normalization_mode": self._normalization_mode(),
            "marker_baseline": (
                self.extract.marker_baseline
                if self.extract.mode == "markers" else None),
            "marker_padding_s": (
                float(self.extract.marker_padding)
                if self.extract.mode == "markers" else 0.0),
            "marker_padding_samples": int(self._pad),
            "marker_shift_mode": self._marker_shift_mode(),
            # A single number only when one was applied to every recording.
            # Detected shifts differ per file and are in ``recordings`` below.
            "marker_time_shift_s": (
                float(self.extract.marker_shift_s)
                if self._marker_shift_mode() == "entered-by-hand" else None),
            "marker_preparation_s": (
                float(self.extract.preparation_s)
                if self.extract.mode == "markers" else 0.0),
            "marker_recovery_s": (
                float(self.extract.recovery_s)
                if self.extract.mode == "markers" else 0.0),
            "marker_extra_cut_first_s": (
                float(self.extract.cut_first)
                if self.extract.mode == "markers" else 0.0),
            "marker_cut_last_s": (
                float(self.extract.cut_last)
                if self.extract.mode == "markers" else 0.0),
            "active_time_zero": (
                "corrected-marker-plus-preparation-plus-extra-cut"
                if self.extract.mode == "markers" else "trigger-onset"),
            "motor_task_duration_s": (
                float(self._motor_duration_s)
                if (self.extract.mode == "markers"
                    and self._motor_duration_s is not None) else None),
            "active_core_duration_s": float(self._La / self.fs),
            "required_tf_padding_samples": int(self._required_tf_pad),
            "rest_transition_skip_s": (
                float(self.extract.rest_transition_skip)
                if self.extract.mode == "markers" else 0.0),
            "candidate_scale": self.candidate_scale,
            "csp_shrinkage": float(self.csp_shrinkage),
            "candidate_cutoffs": {
                "low_fraction": 10 / io_h5.N_EEG,
                "high_fraction": 30 / io_h5.N_EEG,
            },
            "candidate_diagnostic": copy.deepcopy(self.candidate_diagnostic),
            "montage": {
                "channel_count": int(self.ch_num),
                "channel_labels": self.channel_labels.copy(),
                "selection": (
                    "all-file-normalized-label-intersection-reference-order"
                    if self.file_type == "xdf" else "KLH-reference-leading-channels"),
                **copy.deepcopy(self._montage_details),
            },
            "condition_number": float(self.condition_number),
            "condition_warning": self.condition_warning,
            "trial_counts": copy.deepcopy(self.trial_counts),
            "window_diagnostics": copy.deepcopy(self._window_diagnostics),
        }
        return provenance

    # ------------------------------------------------------------------
    def _marker_shift_mode(self) -> str | None:
        if self.extract.mode != "markers":
            return None
        return ("detected-per-recording" if self.extract.auto_marker_shift
                else "entered-by-hand")

    def _marker_shift_signature(self):
        """What the data-path digest has to cover for marker time shifts.

        Detected shifts are part of the analysis even though no one typed them,
        so the per-file values enter the digest. They are rounded to a
        millisecond: two runs over the same recordings must produce the same
        identifier, and the measurement is nowhere near that precise anyway.
        """
        mode = self._marker_shift_mode()
        if mode is None:
            return None
        if mode == "entered-by-hand":
            return (mode, round(float(self.extract.marker_shift_s), 3))
        resolved = getattr(self, "_marker_shifts", {}) or {}
        return (mode, tuple(sorted(
            (os.path.basename(path), round(float(value), 3))
            for path, value in resolved.items())))

    def _configure_fs(self, fs) -> None:
        """Set the processing rate and derive the fixed 8 s epoch window.

        The epoch spans -2.5..5.5 s regardless of rate. Public analyses keep
        this at 1000 Hz because the pre-built filters are fixed-rate; accepting
        a rate here remains useful for small helper tests.
        """
        self.fs = float(fs)
        self.processing_fs = self.fs
        self.epoind = np.arange(int(round(-2.5 * self.fs)), int(round(5.5 * self.fs)))
        self.tvec = np.round(self.epoind / self.fs, 3)
        self.tvec_active = self.tvec
        self.tvec_rest = self.tvec
        self._La = self.epoind.size
        self._Lr = self.epoind.size
        self._motor_duration_s = None

    def _prepare_markers(self, all_files) -> None:
        """Scan every recording's markers to fix common window lengths (samples).

        Windows are cropped to the shortest ``active``/``rest`` event across all
        files so the resulting epochs stack along the trial axis. The active
        core begins after preparation and includes the configured recovery
        interval beyond the marker-defined event. Also sets the active/rest
        time axes. Only the marker stream is parsed here, so this pre-scan costs
        milliseconds per file. Raises informative errors for absent conditions
        or exclusions that consume the motor-task portion of a microrepeat.

        Each recording's marker time shift is resolved in the same pass. The
        shift moves active and rest onsets equally and event durations are gaps
        between markers, so no window *length* fixed here depends on it.
        """
        cfg = self.extract
        self._validate_extract_config()
        wanted = {cfg.active_cond, cfg.rest_cond}
        motor_durations, dur_rest = [], []
        self._marker_shifts = {}
        self._marker_shift_notes = []
        for path in all_files:
            name = os.path.basename(path)
            shift, note, _ = resolve_marker_shift(
                path, cfg.auto_marker_shift, cfg.marker_shift_s)
            self._marker_shifts[os.path.abspath(path)] = float(shift)
            self._marker_shift_notes.append(f"{name}: {note}")
            durations = io_h5.load_xdf_event_durations(path, wanted)
            # The final marker of a file has no successor, so its duration is
            # NaN; such an event cannot define a window and is dropped.
            dur_a = durations[cfg.active_cond]
            dur_r = durations[cfg.rest_cond]
            dur_a = dur_a[np.isfinite(dur_a)]
            dur_r = dur_r[np.isfinite(dur_r)]
            if dur_a.size == 0:
                raise KeyError(f"no '{cfg.active_cond}' markers in {name}")
            if dur_r.size == 0:
                raise KeyError(f"no '{cfg.rest_cond}' markers in {name}")
            motor = dur_a - cfg.preparation_s - cfg.cut_first - cfg.cut_last
            if float(np.min(motor)) <= 0:
                raise ValueError(
                    "preparation_s + cut_first + cut_last "
                    f"({cfg.preparation_s}+{cfg.cut_first}+{cfg.cut_last}s) "
                    f"leaves no motor-task data for '{cfg.active_cond}' in {name}")
            motor_durations.append(float(np.min(motor)))
            dur_rest.append(float(np.min(dur_r)))

        self._motor_duration_s = float(min(motor_durations))
        self._La = int(np.floor(
            (self._motor_duration_s + cfg.recovery_s) * self.fs))
        self._Lr = int(np.floor(min(dur_rest) * self.fs))
        self.tvec_active = np.round(np.arange(self._La) / self.fs, 3)
        self.tvec_rest = np.round(np.arange(self._Lr) / self.fs, 3)
        self.tvec = self.tvec_active
        self._pad = int(round(cfg.marker_padding * self.fs))
        self._required_tf_pad = _superlet_padding_samples(
            getattr(self, "freqNeeded", np.arange(3, 30.0001, 0.5)),
            self.fs, ncyc=5, order_interval=(1, 30))
        if self._pad < self._required_tf_pad:
            required_s = self._required_tf_pad / self.fs
            raise ValueError(
                f"marker_padding={cfg.marker_padding:g}s is too short for the "
                f"configured superlet transform; use at least {required_s:g}s")

    def _validate_extract_config(self) -> None:
        """Canonicalise and validate extraction options in one place."""
        cfg = self.extract
        cfg.mode = str(cfg.mode).strip().lower()
        baseline = str(cfg.marker_baseline).strip().lower().replace("_", "-")
        aliases = {
            "condition-specific": "condition-specific",
            "per-condition": "condition-specific",
            "condition": "condition-specific",
            "pooled": "pooled",
        }
        if cfg.mode not in {"trigger", "markers"}:
            raise ValueError("extract.mode must be 'trigger' or 'markers'")
        if baseline not in aliases:
            raise ValueError(
                "marker_baseline must be 'condition-specific' or 'pooled'")
        cfg.marker_baseline = aliases[baseline]
        for name in ("preparation_s", "recovery_s", "cut_first", "cut_last",
                     "rest_transition_skip", "marker_padding"):
            value = float(getattr(cfg, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative value")
            setattr(cfg, name, value)
        cfg.marker_shift_s = float(cfg.marker_shift_s)
        if not np.isfinite(cfg.marker_shift_s):
            raise ValueError("marker_shift_s must be finite")
        cfg.auto_marker_shift = bool(cfg.auto_marker_shift)

    def _marker_windows(self, events, shift_s, *, return_counts: bool = False):
        """Return (active_starts, active_epoind, rest_starts, rest_epoind).

        ``events`` holds raw marker onsets measured from the recording's first
        sample. ``shift_s`` -- this recording's marker time shift, detected or
        entered -- is added equally to active and rest onsets. Active time zero
        additionally skips ``preparation_s`` and any optional ``cut_first``;
        recovery changes the prepared active core length rather than its onset.
        Rest timing is unaffected by all three active settings.
        """
        cfg = self.extract
        shift_s = float(shift_s)
        on_a, dur_a = events[cfg.active_cond]
        on_r, dur_r = events[cfg.rest_cond]
        on_a = np.asarray(on_a, dtype=float)
        on_r = np.asarray(on_r, dtype=float)
        dur_a = np.asarray(dur_a, dtype=float)
        dur_r = np.asarray(dur_r, dtype=float)

        # A trailing marker has no following marker and therefore no duration.
        # It cannot define a task/rest window and is counted as a dropped marker
        # rather than silently entering epoching with the common length.
        # Common lengths have already been fixed from the shortest finite
        # duration in ``_prepare_markers``.  Only duration-less trailing events
        # need removal here.
        keep_a = np.isfinite(dur_a)
        keep_r = np.isfinite(dur_r)
        eligible_a = on_a[keep_a] + shift_s
        eligible_r = on_r[keep_r] + shift_s
        active_starts = np.rint(
            (eligible_a + cfg.preparation_s + cfg.cut_first) * self.fs
        ).astype(int)
        rest_starts = np.rint(eligible_r * self.fs).astype(int)
        pad = int(getattr(self, "_pad", 0))
        result = (
            active_starts,
            np.arange(-pad, self._La + pad),
            rest_starts,
            np.arange(-pad, self._Lr + pad),
        )
        if return_counts:
            return result + ({
                ACTIVE: {"requested": int(on_a.size),
                         "eligible": int(eligible_a.size)},
                PASSIVE: {"requested": int(on_r.size),
                          "eligible": int(eligible_r.size)},
            },)
        return result

    def _gather(self, files, cond, scale, broad, alpha):
        """Load a list of files for one condition into the epoch stores."""
        # Append blocks, then allocate each final store once. Repeatedly
        # concatenating the growing store copies early recordings many times.
        broad_blocks = [[broad[cond][kind]] if broad[cond][kind] is not None
                        else [] for kind in (ACTIVE, PASSIVE)]
        alpha_blocks = [[alpha[cond][kind]] if alpha[cond][kind] is not None
                        else [] for kind in (ACTIVE, PASSIVE)]
        counts = [sum(block.shape[1] for block in blocks)
                  for blocks in broad_blocks]
        for path in files:
            file_epoch_slices = {
                "file": os.path.abspath(path),
                "condition": CONDITION_NAMES[cond],
            }
            if self.file_type == "xdf":
                eeg_data, events, video_onset_s, load_info = io_h5.load_xdf_eeg(
                    path, {self.extract.active_cond, self.extract.rest_cond},
                    target_fs=self.fs, pick_labels=self._xdf_pick,
                    return_info=True)
                marker_shift = self._marker_shifts[os.path.abspath(path)]
                if video_onset_s is not None:
                    self.video_onset_offsets[os.path.basename(path)] = (
                        video_onset_s + marker_shift)
                EEG = eeg_data[:, :self.ch_num] * scale
                del eeg_data
                windows = self._marker_windows(
                    events, marker_shift, return_counts=True)
                a_onsets, a_epoind, r_onsets, r_epoind, count_info = windows
                recording_fs = float(load_info["recording_fs"])
                resampled = bool(load_info["resampled"])
                event_path = "embedded-markers-seconds-with-configured-shift"
            else:
                eeg_full = io_h5.load_h5_eeg(path)
                if eeg_full.ndim != 2 or eeg_full.shape[1] <= self.ch_num:
                    raise ValueError(
                        f"{os.path.basename(path)} has {eeg_full.shape[1]} columns; "
                        f"need {self.ch_num} EEG columns plus a trigger column")
                # Detect the categorical trigger at its native sampling rate.
                # Only the continuous EEG is then passed through resample_poly.
                trig = io_h5.decode_trigger(eeg_full)
                ind_rest_native, ind_single_native, _, _ = io_h5.trig_detector(
                    trig, self.recording_fs)
                a_onsets = io_h5.remap_sample_indices(
                    ind_single_native, self.recording_fs, self.fs)
                r_onsets = io_h5.remap_sample_indices(
                    ind_rest_native, self.recording_fs, self.fs)
                a_epoind = r_epoind = self.epoind
                count_info = {
                    ACTIVE: {"requested": int(a_onsets.size),
                             "eligible": int(a_onsets.size)},
                    PASSIVE: {"requested": int(r_onsets.size),
                              "eligible": int(r_onsets.size)},
                }
                eeg_native = eeg_full[:, :self.ch_num]
                EEG = io_h5.resample_eeg(
                    eeg_native, self.recording_fs, self.fs) * scale
                del eeg_native, eeg_full
                recording_fs = float(self.recording_fs)
                resampled = abs(recording_fs - self.fs) > 1e-6
                event_path = "native-trigger-detection"
                marker_shift = 0.0

            self._recording_details.append({
                "file": os.path.abspath(path),
                "condition": CONDITION_NAMES[cond],
                "recording_fs": recording_fs,
                "processing_fs": float(self.fs),
                "resampled": resampled,
                "resampling_method": (
                    "scipy.signal.resample_poly" if resampled else "none"),
                "event_path": event_path,
                "marker_time_shift_s": float(marker_shift),
            })

            # Only the sample spans that become epochs are filtered.  Zero-phase
            # FIR filtering costs O(samples * taps) and the stored coefficients
            # run to ~2500 taps, so the stretch of a recording no trial reaches
            # is the largest avoidable cost in the run.
            full_length = int(EEG.shape[0])
            crop_offset, EEG = _crop_to_epoch_span(
                EEG, ((a_onsets, a_epoind), (r_onsets, r_epoind)),
                margin=self._filter_edge_margin())
            a_onsets = a_onsets - crop_offset
            r_onsets = r_onsets - crop_offset
            self._recording_details[-1]["filter_crop"] = {
                "applied": bool(crop_offset or EEG.shape[0] != full_length),
                "offset_samples": int(crop_offset),
                "kept_samples": int(EEG.shape[0]),
                "recording_samples": int(full_length),
                "edge_margin_samples": int(self._filter_edge_margin()),
            }

            def filter_band(band):
                if band == 0:
                    return filters.broadband(EEG, self.b1, self.b2)
                return filters.alpha(EEG, self.bpa)

            EEGf1, EEGf2 = execution.ordered_map(
                filter_band, (0, 1), working_bytes=8 * EEG.nbytes)

            for kind, onsets, epoind_k in ((ACTIVE, a_onsets, a_epoind),
                                           (PASSIVE, r_onsets, r_epoind)):
                start = counts[kind]
                ep_b, valid = io_h5.epoch(
                    EEGf1, onsets, epoind_k, return_valid=True)
                ep_a = io_h5.epoch(EEGf2, onsets[valid], epoind_k)
                broad_blocks[kind].append(ep_b)
                alpha_blocks[kind].append(ep_a)
                counts[kind] += int(valid.sum())
                self._record_trial_counts(
                    cond, kind, path,
                    requested=count_info[kind]["requested"],
                    eligible=count_info[kind]["eligible"],
                    retained=int(valid.sum()))
                file_epoch_slices[EPOCH_NAMES[kind]] = [
                    start, start + int(valid.sum())]
            # Be defensive for lightweight test/custom instances constructed
            # without __init__; normal application instances already own it.
            if not hasattr(self, "_file_epoch_slices"):
                self._file_epoch_slices = []
            self._file_epoch_slices.append(file_epoch_slices)
            # Epoch extraction owns its samples; release continuous recordings
            # before loading the next file or assembling the final stores.
            del EEG, EEGf1, EEGf2

        for kind in (ACTIVE, PASSIVE):
            for store, blocks in ((broad, broad_blocks), (alpha, alpha_blocks)):
                parts = blocks[kind]
                store[cond][kind] = (None if not parts else parts[0]
                                    if len(parts) == 1 else
                                    np.concatenate(parts, axis=1))
                parts.clear()

    def _record_trial_counts(self, cond, kind, path, *, requested, eligible,
                             retained) -> None:
        """Accumulate marker/epoch losses at condition and file resolution."""
        requested, eligible, retained = map(int, (requested, eligible, retained))
        summary = self.trial_counts[CONDITION_NAMES[cond]][EPOCH_NAMES[kind]]
        summary["requested"] += requested
        summary["eligible"] += eligible
        summary["retained"] += retained
        summary["invalid_duration"] += requested - eligible
        summary["out_of_bounds"] += eligible - retained
        summary["dropped"] += requested - retained
        self.trial_counts["by_file"].append({
            "file": os.path.abspath(path),
            "condition": CONDITION_NAMES[cond],
            "epoch": EPOCH_NAMES[kind],
            "requested": requested,
            "eligible": eligible,
            "retained": retained,
            "invalid_duration": requested - eligible,
            "out_of_bounds": eligible - retained,
            "dropped": requested - retained,
        })

    def run_start(self, files_real, files_quasi, files_imag,
                  extract: ExtractConfig | None = None,
                  file_type: str = "h5", ch_num: int = io_h5.N_EEG,
                  fs: float = 1000.0, *,
                  processing_fs: float = filters.FS_DEFAULT,
                  candidate_scale: str = "raw",
                  csp_shrinkage: float = 0.0) -> StartResult:
        """Run preprocessing and CSP candidate detection.

        ``fs`` remains positional for compatibility, but now means the native
        *recording* rate for H5 input.  All EEG is processed at the fixed
        ``processing_fs`` required by the stored 1000-Hz FIR coefficients.
        XDF native rates are read from each stream and the ``fs`` hint is ignored.
        """
        self.clear_analysis_cache()
        files_real = list(files_real)
        files_quasi = list(files_quasi)
        files_imag = list(files_imag)
        _validate_unique_input_files(
            ("real", files_real),
            ("quasi", files_quasi),
            ("imag", files_imag),
        )
        self.extract = extract or ExtractConfig()
        self._validate_extract_config()
        self.file_type = str(file_type).strip().lower()
        if self.file_type not in {"h5", "xdf"}:
            raise ValueError("file_type must be 'h5' or 'xdf'")
        self.ch_num = int(ch_num)
        if self.ch_num < 2:
            raise ValueError("at least two matched EEG channels are required")
        if self.file_type == "h5" and self.ch_num > io_h5.N_EEG:
            raise ValueError(
                f"H5 ch_num cannot exceed the {io_h5.N_EEG}-channel KLH "
                "reference montage")
        processing_fs = float(processing_fs)
        if abs(processing_fs - filters.FS_DEFAULT) > 1e-6:
            raise ValueError(
                f"the stored FIR coefficients require processing_fs="
                f"{filters.FS_DEFAULT:g} Hz (got {processing_fs:g} Hz)")
        if float(fs) <= 0:
            raise ValueError("recording sampling rate must be positive")
        self.recording_fs = float(fs) if self.file_type == "h5" else None
        candidate_scale = str(candidate_scale).strip().lower()
        if candidate_scale not in {"raw", "logit"}:
            raise ValueError("candidate_scale must be 'raw' or 'logit'")
        self.candidate_scale = candidate_scale
        csp_shrinkage = float(csp_shrinkage)
        if not np.isfinite(csp_shrinkage) or not 0.0 <= csp_shrinkage < 1.0:
            raise ValueError("csp_shrinkage must satisfy 0 <= value < 1")
        self.csp_shrinkage = csp_shrinkage
        self._configure_fs(processing_fs)
        self.video_onset_offsets = {}
        self._marker_shifts = {}
        self._marker_shift_notes = []
        self.trial_counts = _empty_trial_counts()
        self._recording_details = []
        self._file_epoch_slices = []
        self._window_diagnostics = []
        self._montage_details = {}
        self.condition_number = np.nan
        self.condition_warning = None
        self.component_ranking = []
        self.channel_qc = {}
        self.physiological_qc = None
        self.candidate_diagnostic = {}
        self._pad = 0
        self._required_tf_pad = 0
        self._input_files = {
            "real": [os.path.abspath(path) for path in files_real],
            "quasi": [os.path.abspath(path) for path in files_quasi],
            "imag": [os.path.abspath(path) for path in files_imag],
        }
        # XDF: restrict every selected recording to the same intersection of
        # reference-montage electrodes.  The final order always follows the .ced
        # reference, never the input-file order or the cap's acquisition order.
        if self.file_type == "xdf":
            all_files = files_real + files_quasi + files_imag
            if not all_files:
                raise ValueError("at least one XDF recording is required")
            reference_labels = list(load_ced()["labels"])
            self._xdf_pick, self._montage_details = _common_xdf_montage(
                all_files, reference_labels)
            keep = [i for i, label in enumerate(reference_labels)
                    if label in self._xdf_pick]
            self.channel_labels = self._xdf_pick.copy()
            self.ideal_pattern = self._ideal_pattern_full[keep]
            self.ch_num = len(self._xdf_pick)
        else:
            self._xdf_pick = None
            self.channel_labels = list(load_ced()["labels"][:self.ch_num])
            self.ideal_pattern = self._ideal_pattern_full[:self.ch_num]
        if self.file_type == "xdf" and self.extract.mode == "trigger":
            raise ValueError(
                "XDF recordings are epoched from their embedded event markers; "
                "trigger-channel mode requires H5.")
        if self.file_type != "xdf" and self.extract.mode == "markers":
            raise ValueError(
                "embedded event markers are available only in XDF recordings; "
                f"{file_type.upper()} needs trigger-channel extraction.")
        if self.extract.mode == "markers":
            self._prepare_markers(files_real + files_quasi + files_imag)
            # Marker active time zero is motor-cue onset after the excluded
            # preparation interval. The full core contains motor execution and
            # post-movement recovery and is eligible for peak-ERD scoring.
            rng_inclusive(
                self.tvec_active, 0.0, float(self.tvec_active[-1]),
                strict=True, warn=False,
                label="marker scoring", diagnostics=self._window_diagnostics)
            if self.extract.rest_transition_skip >= self._Lr / self.fs:
                raise ValueError(
                    "rest_transition_skip removes the entire marker rest epoch")

        broad = [[None, None] for _ in range(3)]
        alpha = [[None, None] for _ in range(3)]

        # Per-condition input scale. In the original H5 data the imagery files
        # were stored in volts while real/quasi were in microvolts, so imagery
        # was multiplied by 1e6 to match. XDF recordings come from one amplifier
        # in volts for all three conditions, so they share a single scale
        # (1e6: V -> uV, keeping magnitudes in the range the pipeline expects).
        if self.file_type == "xdf":
            s_real = s_quasi = s_imag = 1e6
        else:
            s_real = s_quasi = 1.0
            s_imag = 1e6
        self._gather(files_real, REAL, s_real, broad, alpha)
        self._gather(files_quasi, QUASI, s_quasi, broad, alpha)
        self._gather(files_imag, IMAG, s_imag, broad, alpha)

        self.broad, self.alpha = broad, alpha

        # --- covariances for CSP ---
        ind0 = rng_inclusive(
            self.tvec_rest, 0.5, 4.0, strict=True, warn=False,
            label="CSP rest", diagnostics=self._window_diagnostics)
        ind1 = rng_inclusive(
            self.tvec_active, 0.5, 2.0, strict=True, warn=False,
            label="CSP active", diagnostics=self._window_diagnostics)
        # Kept so cross-validated refits use identical covariance windows.
        self._csp_ind0, self._csp_ind1 = ind0, ind1

        # rest = all conditions' passive alpha epochs concatenated along trials
        rest_epochs = _cat_trials(
            _cat_trials(self._core_epochs(alpha[REAL][PASSIVE], PASSIVE),
                        self._core_epochs(alpha[QUASI][PASSIVE], PASSIVE)),
            self._core_epochs(alpha[IMAG][PASSIVE], PASSIVE))
        eeg0 = rest_epochs[ind0, :, :]
        eeg1 = self._core_epochs(alpha[REAL][ACTIVE], ACTIVE)[ind1, :, :]

        X0 = eeg0.reshape(-1, self.ch_num, order="F")
        X1 = eeg1.reshape(-1, self.ch_num, order="F")

        (cov0, _), (cov1, _) = execution.ordered_map(
            olivehawkins, (X0, X1), blas=True,
            working_bytes=8 * max(X0.nbytes, X1.nbytes))

        self.condition_number = covariance_pencil_condition(cov1, cov0)
        if (not np.isfinite(self.condition_number)
                or self.condition_number >= CONDITION_WARNING_THRESHOLD):
            self.condition_warning = (
                "CSP covariance pencil is ill-conditioned "
                f"(condition number {self.condition_number:.3g}); component "
                "estimates may be unstable")
        try:
            (self.projInverse, self.projForward, _, _,
             self.evals) = csp.calc_csp_cov(
                 cov1, cov0, shrinkage=self.csp_shrinkage,
                 include_ged=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "CSP eigendecomposition failed; covariance-pencil condition "
                f"number is {self.condition_number:.3g}") from exc

        # --- eigenvalue candidate detection ---
        (ok_low_inds, ok_high_inds, self.n_erd,
         self.candidate_diagnostic) = detect_eigenvalue_candidates(
            self.evals, self.ch_num, scale=self.candidate_scale)
        n_components = min(5, self.projForward.shape[1])
        self.component_ranking = [
            asdict(item) for item in qc.rank_component_patterns(
                self.projForward[:, :n_components], self.channel_labels,
                ideal_pattern=self.ideal_pattern,
                ideal_labels=self.channel_labels)
        ]
        epoch_groups = [
            self._core_epochs(self.broad[cond][kind], kind)
            for cond in (REAL, QUASI, IMAG)
            for kind in (ACTIVE, PASSIVE)
        ]
        self.channel_qc = qc.detect_persistent_channel_outliers(
            epoch_groups, self.channel_labels)
        self.provenance = self._build_provenance()

        return StartResult(
            evals=self.evals, projInverse=self.projInverse, projForward=self.projForward,
            ok_low_inds=ok_low_inds, ok_high_inds=ok_high_inds, n_erd=self.n_erd,
            top_patterns=self.projForward[:, :n_components].copy(),
            video_onset_offsets=self.video_onset_offsets.copy(),
            channel_labels=self.channel_labels.copy(), n_components=n_components,
            condition_number=self.condition_number,
            condition_warning=self.condition_warning,
            trial_counts=copy.deepcopy(self.trial_counts),
            component_ranking=copy.deepcopy(self.component_ranking),
            channel_qc=copy.deepcopy(self.channel_qc),
            file_epoch_slices=copy.deepcopy(self._file_epoch_slices),
            provenance=copy.deepcopy(self.provenance))

    # ------------------------------------------------------------------
    def _fold_assignment(self, n_trials: int, n_folds: int) -> np.ndarray:
        """Interleaved fold labels, so every fold spans all input recordings.

        Trials are concatenated recording by recording, so contiguous blocks
        would put whole recordings in a single fold and confound the held-out
        estimate with between-session differences.
        """
        return np.arange(int(n_trials)) % int(n_folds)

    def _match_component(self, fold_forward: np.ndarray,
                         reference_pattern: np.ndarray) -> tuple[int, float, float]:
        """Match a fold's components to the selected full-data pattern.

        CSP eigenvectors carry no stable identity across refits: their order
        and sign both depend on the training set.  The fold component is the
        one whose forward pattern correlates most strongly in absolute value
        with the reference.  The margin over the runner-up is returned so an
        ambiguous match can be reported rather than silently trusted.
        """
        ref = np.asarray(reference_pattern, dtype=float).ravel()
        ref = ref - ref.mean()
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm <= np.finfo(float).eps:
            raise ValueError("reference pattern is constant; cannot match folds")
        columns = np.asarray(fold_forward, dtype=float)
        centred = columns - columns.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centred, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.abs((ref @ centred) / (ref_norm * norms))
        corr[~np.isfinite(corr)] = 0.0
        best = int(np.argmax(corr))
        ordered = np.sort(corr)[::-1]
        margin = float(ordered[0] - ordered[1]) if corr.size > 1 else float(ordered[0])
        return best, float(corr[best]), margin

    def clear_analysis_cache(self) -> None:
        """Release cached fold fits and sensor-space QC for this session."""
        self._cv_fit_cache = None
        self._qc_cache = None

    def _cv_overt_filters(self, idx: int, n_folds: int) -> list[dict]:
        """Refit CSP per fold with the held-out overt trials fully excluded.

        The selected component's *identity* is still chosen on the full data —
        that is the component the operator picked and is what criterion 2
        describes.  Only the filter **weights** used to measure held-out overt
        ERD are re-estimated without those trials, which is where the optimism
        in criterion 3 comes from.

        Quasi and imagery rest epochs stay in the training pool for every fold.
        They contribute to the CSP denominator, so a second-order leak remains
        under a pooled baseline; the overt active trials that drive the
        numerator, and the overt rest trials that form the condition-specific
        baseline, are held out cleanly.
        """
        active_all = self._core_epochs(
            self.alpha[REAL][ACTIVE], ACTIVE)[self._csp_ind1]
        rest_by_cond = [
            self._core_epochs(self.alpha[cond][PASSIVE], PASSIVE)[self._csp_ind0]
            for cond in (REAL, QUASI, IMAG)
        ]
        n_active = active_all.shape[1]
        n_rest_real = rest_by_cond[REAL].shape[1]
        if n_folds > min(n_active, n_rest_real):
            raise ValueError(
                f"csp_cv_folds={n_folds} exceeds the {min(n_active, n_rest_real)} "
                "overt active/rest trials available to split")

        reference = self.projForward[:, idx]
        active_folds = self._fold_assignment(n_active, n_folds)
        rest_folds = self._fold_assignment(n_rest_real, n_folds)
        other_rest = np.concatenate(
            [rest_by_cond[QUASI], rest_by_cond[IMAG]], axis=1)

        # Only matching a fitted component to the selected full-data pattern
        # depends on idx. Keep one configuration's small matrices, never its
        # large training arrays. Hash the actual covariance windows so edits
        # to externally supplied epochs cannot silently reuse an old fit.
        cache_key = (n_folds, self.ch_num, float(self.csp_shrinkage),
                     execution.array_digest(active_all, *rest_by_cond))
        cached = getattr(self, "_cv_fit_cache", None)

        def training_rest(k):
            return np.concatenate(
                [rest_by_cond[REAL][:, rest_folds != k, :], other_rest], axis=1)

        def fit_fold(k):
            train_active = active_all[:, active_folds != k, :]
            train_rest = training_rest(k)
            X1 = train_active.reshape(-1, self.ch_num, order="F")
            X0 = train_rest.reshape(-1, self.ch_num, order="F")
            cov1, _ = olivehawkins(X1)
            cov0, _ = olivehawkins(X0)
            fold_inverse, fold_forward, _, _, _ = csp.calc_csp_cov(
                cov1, cov0, shrinkage=self.csp_shrinkage, include_ged=False)
            return cov0, cov1, fold_inverse, fold_forward

        if cached is not None and cached[0] == cache_key:
            fits = cached[1]
        else:
            fits = execution.ordered_map(
                fit_fold, range(n_folds), blas=True,
                working_bytes=8 * (active_all.nbytes
                                   + sum(rest.nbytes for rest in rest_by_cond)))
            self._cv_fit_cache = (cache_key, fits)

        folds = []
        for k, (_, _, fold_inverse, fold_forward) in enumerate(fits):
            comp, similarity, margin = self._match_component(
                fold_forward, reference)
            weights = fold_inverse[:, comp].astype(float)
            # Fold filters are defined only up to scale, so held-out trials
            # from different folds are not directly poolable.  Normalising by
            # the training-rest output standard deviation puts every fold on
            # one unit without touching held-out data.
            X0 = training_rest(k).reshape(-1, self.ch_num, order="F")
            scale = float(np.std(X0 @ weights))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"fold {k} filter has degenerate training-rest scale")
            folds.append({
                "fold": k,
                "weights": weights / scale,
                "active_test": np.flatnonzero(active_folds == k),
                "rest_test": np.flatnonzero(rest_folds == k),
                "n_train_active": int(np.count_nonzero(active_folds != k)),
                "n_train_rest": int(np.count_nonzero(rest_folds != k)
                                    + other_rest.shape[1]),
                "matched_component": comp + 1,
                "pattern_similarity_to_selected": similarity,
                "match_margin": margin,
            })
        return folds

    def _cv_overt_tf(self, idx: int, folds: list[dict],
                     with_rest: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """Held-out overt (active, rest) TF power, in original trial order.

        Every trial is transformed exactly once, through the one fold filter
        that did not train on it, so the returned arrays keep the full trial
        counts. Preserving the count matters for both the archived extreme
        percentile and the rolling-median peak: thinning the overt trial
        median would raise its noise floor relative to quasi and imagery and
        re-bias the very ratio this change exists to correct.

        ``with_rest=False`` skips the rest epochs entirely.  Trigger mode
        baselines the prestimulus part of the active epochs, so transforming
        rest there would be pure waste.
        """
        active_epochs = self.broad[REAL][ACTIVE]
        rest_epochs = self.broad[REAL][PASSIVE] if with_rest else None
        active_out = None
        rest_out = None
        for fold in folds:
            weights = fold["weights"]
            a_idx = fold["active_test"]
            if a_idx.size:
                block = self._tf_trials(
                    active_epochs[:, a_idx, :], idx, ACTIVE, weights=weights)
                if active_out is None:
                    active_out = np.empty(
                        (block.shape[0], active_epochs.shape[1], block.shape[2]),
                        dtype=block.dtype)
                active_out[:, a_idx, :] = block
            r_idx = fold["rest_test"]
            if rest_epochs is not None and r_idx.size:
                block = self._tf_trials(
                    rest_epochs[:, r_idx, :], idx, PASSIVE, weights=weights)
                if rest_out is None:
                    rest_out = np.empty(
                        (block.shape[0], rest_epochs.shape[1], block.shape[2]),
                        dtype=block.dtype)
                rest_out[:, r_idx, :] = block
        if active_out is None or (with_rest and rest_out is None):
            raise ValueError("cross-validation produced no held-out overt trials")
        return active_out, rest_out

    def _cv_summary(self, folds: list[dict], requested_folds: int,
                    cross_validated: bool) -> dict:
        """Auditable record of how each fold's filter was chosen.

        A held-out overt ERD is only meaningful if every fold actually
        re-found the operator's component, so the per-fold match similarity
        and its margin over the runner-up are reported alongside the result
        rather than being collapsed into a single number.
        """
        if not cross_validated:
            return {
                "enabled": False,
                "requested_folds": int(requested_folds),
                "reason": ("cross-validation disabled; overt peak ERD is an "
                           "in-sample estimate and criterion 4 is biased low"),
            }
        similarities = [f["pattern_similarity_to_selected"] for f in folds]
        margins = [f["match_margin"] for f in folds]
        ambiguous = [f["fold"] for f in folds
                     if f["match_margin"] < CV_MATCH_MARGIN_WARNING]
        warnings = []
        if ambiguous:
            warnings.append(
                f"folds {ambiguous} matched the selected component by a margin "
                f"below {CV_MATCH_MARGIN_WARNING}; the held-out overt ERD may "
                "mix different components")
        if min(similarities) < 0.5:
            warnings.append(
                f"weakest fold pattern similarity is {min(similarities):.2f}; "
                "the component is not stable across overt trial subsets")
        return {
            "enabled": True,
            "folds": int(len(folds)),
            "requested_folds": int(requested_folds),
            "scope": "overt-active-and-overt-rest-held-out",
            "residual_leak": ("quasi/imagery rest epochs remain in the CSP "
                              "training pool for every fold"),
            "min_pattern_similarity": float(min(similarities)),
            "median_pattern_similarity": float(np.median(similarities)),
            "min_match_margin": float(min(margins)),
            "matched_components": [f["matched_component"] for f in folds],
            "per_fold": [
                {k: v for k, v in f.items()
                 if k not in ("weights", "active_test", "rest_test")}
                | {"n_active_test": int(f["active_test"].size),
                   "n_rest_test": int(f["rest_test"].size)}
                for f in folds
            ],
            "warnings": warnings,
        }

    def _cv_marker_baseline(self, idx: int, folds: list[dict],
                            rest_power_real: np.ndarray) -> np.ndarray:
        """Marker-rest baseline for the cross-validated overt map.

        Fold filters are scaled in training-rest units, so the full-data
        baseline computed by :meth:`_rest_baseline` is in the wrong units and
        cannot be reused here.  Under a pooled baseline the other conditions'
        rest epochs are projected fold-wise too — round-robin, so each trial is
        transformed exactly once and the pooled trial count is preserved.
        """
        blocks = [rest_power_real]
        if self.extract.marker_baseline == "pooled":
            other = _cat_trials(self.broad[QUASI][PASSIVE], self.broad[IMAG][PASSIVE])
            assignment = self._fold_assignment(other.shape[1], len(folds))
            for fold in folds:
                take = np.flatnonzero(assignment == fold["fold"])
                if take.size:
                    blocks.append(self._tf_trials(
                        other[:, take, :], idx, PASSIVE, weights=fold["weights"]))
        pooled = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
        skip = int(round(self.extract.rest_transition_skip * self.fs))
        if skip >= pooled.shape[0]:
            raise ValueError("rest_transition_skip removes the entire baseline")
        return np.median(pooled[skip:], axis=(0, 1))

    def _tf_trials(self, epochs: np.ndarray, idx: int, kind: int,
                   weights: np.ndarray | None = None) -> np.ndarray:
        """Transform padded epochs and return only their uncontaminated cores.

        Epochs remain concatenated for computational efficiency, but every core
        is separated from an artificial boundary by real recorded context at
        least as long as the largest wavelet half-support.

        ``weights`` overrides the full-data spatial filter with an explicit
        one.  Cross-validated scoring uses it to project held-out trials
        through a filter that never saw them; ``idx`` is then ignored.
        """
        if epochs is None or epochs.shape[1] == 0:
            raise ValueError(f"no {EPOCH_NAMES[kind]} epochs available")
        core_len = self._La if kind == ACTIVE else self._Lr
        pad = self._pad if self.extract.mode == "markers" else 0
        padded_len = core_len + 2 * pad
        if epochs.shape[0] != padded_len:
            raise ValueError(
                f"unexpected {EPOCH_NAMES[kind]} epoch length {epochs.shape[0]} "
                f"(expected {padded_len})")
        if weights is None:
            filt = self.projInverse[:, idx]
        else:
            filt = np.asarray(weights, dtype=float).ravel()
            if filt.size != self.ch_num:
                raise ValueError(
                    f"weights must have {self.ch_num} entries, got {filt.size}")
        sig = epochs.reshape(-1, self.ch_num, order="F") @ filt
        wt = superlet.aslt(
            sig, self.fs, self.freqNeeded, 5, (1, 30), 0)
        n_trials = epochs.shape[1]
        X = wt.T.reshape(
            padded_len, n_trials, self.freqNeeded.size, order="F")
        return X[pad:pad + core_len]

    def _rest_baseline(self, idx: int, condition: int | None = None) -> np.ndarray:
        """Marker-rest baseline after padding removal and transition exclusion.

        ``condition=None`` pools rest trials from all recordings.  Passing one
        of ``REAL``, ``QUASI`` or ``IMAG`` yields a condition-specific baseline.
        """
        if condition is None:
            rest_epochs = _cat_trials(
                _cat_trials(self.broad[REAL][PASSIVE],
                            self.broad[QUASI][PASSIVE]),
                self.broad[IMAG][PASSIVE])
        else:
            rest_epochs = self.broad[condition][PASSIVE]
        if rest_epochs is None or rest_epochs.shape[1] == 0:
            raise ValueError("no rest epochs available for the marker baseline")
        X = self._tf_trials(rest_epochs, idx, PASSIVE)
        skip = int(round(self.extract.rest_transition_skip * self.fs))
        if skip >= X.shape[0]:
            raise ValueError("rest_transition_skip removes the entire baseline")
        return np.median(X[skip:], axis=(0, 1))

    def compute_physiological_qc(self) -> qc.PhysiologicalQCResult:
        """Return condition-specific sensor-space QC, separate from scoring."""
        if self.broad is None:
            raise RuntimeError("run_start must complete before physiological QC")
        condition_epochs = {}
        for cond, name in zip((REAL, QUASI, IMAG), ("overt", "quasi", "imagery")):
            active = self._core_epochs(self.broad[cond][ACTIVE], ACTIVE)
            if self.extract.mode == "markers":
                rest = self._core_epochs(self.broad[cond][PASSIVE], PASSIVE)
            else:
                # Trigger-mode scores use the condition's own prestimulus
                # segment from the active epochs, not a pooled rest recording.
                rest = active
            condition_epochs[name] = (active, rest)

        condition_index = {"real": REAL, "quasi": QUASI, "imag": IMAG}
        display_name = {"real": "overt", "quasi": "quasi", "imag": "imagery"}
        recording_epochs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
            "overt": {}, "quasi": {}, "imagery": {}}
        for row in getattr(self, "_file_epoch_slices", []):
            cond_key = row["condition"]
            cond = condition_index[cond_key]
            active_start, active_stop = row["active"]
            active = self._core_epochs(
                self.broad[cond][ACTIVE], ACTIVE)[:, active_start:active_stop]
            if self.extract.mode == "markers":
                rest_start, rest_stop = row["rest"]
                rest = self._core_epochs(
                    self.broad[cond][PASSIVE], PASSIVE)[:, rest_start:rest_stop]
            else:
                rest = active
            recording_epochs[display_name[cond_key]][row["file"]] = (active, rest)

        if self.extract.mode == "markers":
            rest_start = float(self.extract.rest_transition_skip)
            rest_end = min(rest_start + 3.0, float(self.tvec_rest[-1]))
            rest_window = (rest_start, rest_end)
        else:
            rest_window = (-1.5, -1.0)
        # Hash only the windows QC consumes, using views for the normal
        # monotonic time axes. Component selection is intentionally absent.
        ia = qc._window_indices(self.tvec_active, (0.5, 2.5), "active")
        ir = qc._window_indices(self.tvec_rest, rest_window, "rest")
        def window_view(epochs, indices):
            if np.all(np.diff(indices) == 1):
                return epochs[indices[0]:indices[-1] + 1]
            return epochs[indices]
        cache_key = (float(self.fs), tuple(self.channel_labels),
                     self.extract.active_cond, rest_window,
                     tuple(qc.DEFAULT_BANDS.items()),
                     repr(getattr(self, "_file_epoch_slices", [])),
                     execution.array_digest(
                         self.tvec_active, self.tvec_rest,
                         *(window_view(epochs, indices)
                           for pair in condition_epochs.values()
                           for epochs, indices in zip(pair, (ia, ir)))))
        cached = getattr(self, "_qc_cache", None)
        if cached is not None and cached[0] == cache_key:
            self.physiological_qc = copy.deepcopy(cached[1])
            return self.physiological_qc
        result = qc.compute_sensorimotor_qc(
            condition_epochs, fs=self.fs,
            tvec_active=self.tvec_active, tvec_rest=self.tvec_rest,
            channel_labels=self.channel_labels,
            active_condition=self.extract.active_cond,
            active_window=(0.5, 2.5), rest_window=rest_window,
            common_average=True, recording_epochs=recording_epochs)
        self._qc_cache = (cache_key, copy.deepcopy(result))
        self.physiological_qc = result
        return result

    def _sustained_erd_support(
        self,
        trial_power: list[np.ndarray],
        baselines: list[np.ndarray],
        *,
        selected_comp: int,
        data_path_version: str,
        bootstrap_resamples: int = 500,
    ) -> dict:
        """Build descriptive v3 uncertainty and per-recording diagnostics.

        The score itself is still calculated from each condition's pooled,
        normalized component map.  Bootstrap samples recompute that pooled-map
        statistic after resampling active trials, with the scoring baseline
        held fixed.  Per-run values use the same baseline and component as the
        pooled score, so no post-hoc component selection or normalization
        change is introduced.
        """
        fsl = rng_inclusive(
            self.freqNeeded, 9.0, 13.0, strict=True, warn=False,
            label="v3 sustained mu-ERD frequency window")
        tsl = rng_inclusive(
            self.tvec_active, 0.5, 2.0, strict=True, warn=False,
            label="v3 sustained mu-ERD active window")
        bootstrap_resamples = int(bootstrap_resamples)
        if bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")

        ci_db = {}
        run_consistency = {}
        uncertainty_counts = {}
        for cond, condition in ((QUASI, "quasi"), (IMAG, "imagery")):
            X = np.asarray(trial_power[cond], dtype=float)
            base = np.asarray(baselines[cond], dtype=float).ravel()
            region = X[tsl, :, :][:, :, fsl]
            with np.errstate(divide="ignore", invalid="ignore"):
                trial_region_db = 10.0 * np.log10(
                    region / base[fsl][None, None, :])
            finite_trials = np.all(np.isfinite(trial_region_db), axis=(0, 2))
            finite_region = region[:, finite_trials, :]
            n_finite = int(finite_region.shape[1])
            uncertainty_counts[condition] = n_finite
            if n_finite:
                seed_material = (
                    f"{data_path_version}|component={selected_comp}|{condition}"
                ).encode("utf-8")
                seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                estimates = _bootstrap_erd_estimates(
                    finite_region, base[fsl], rng, bootstrap_resamples)
                lo, hi = np.percentile(estimates, [2.5, 97.5])
                ci_db[condition] = [float(lo), float(hi)]
            else:
                ci_db[condition] = [np.nan, np.nan]

            per_run = []
            for row in getattr(self, "_file_epoch_slices", []):
                if row.get("condition") != CONDITION_NAMES[cond]:
                    continue
                start, stop = map(int, row.get("active", (0, 0)))
                retained = max(0, stop - start)
                run_d = np.nan
                if retained:
                    run_power = np.median(X[:, start:stop, :], axis=1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        run_tf_db = (
                            10.0 * np.log10(run_power / base[None, :])).T
                    run_d = float(-np.median(run_tf_db[fsl, tsl]))
                per_run.append({
                    "file": str(row.get("file", "unreported")),
                    "active_trials": int(retained),
                    "sustained_erd_db": run_d,
                    "erd_positive": bool(np.isfinite(run_d) and run_d > 0.0),
                })
            available = [
                row for row in per_run
                if row["active_trials"] > 0
                and np.isfinite(row["sustained_erd_db"])
            ]
            positive = sum(row["erd_positive"] for row in available)
            run_consistency[condition] = {
                "total_runs": int(len(per_run)),
                "retained_runs": int(len(available)),
                "unavailable_runs": int(len(per_run) - len(available)),
                "positive_erd_runs": int(positive),
                "positive_erd_fraction": (
                    float(positive / len(available)) if available else np.nan),
                "per_run": per_run,
            }

        return {
            "ci_db": ci_db,
            "uncertainty": {
                "method": "trial-bootstrap-pooled-map-sustained-erd",
                "confidence_level": 0.95,
                "resamples": bootstrap_resamples,
                "finite_trials": uncertainty_counts,
                "estimator": (
                    "resample active trials, median trial power, normalize to "
                    "the fixed scoring baseline, then take negative regional median"),
                "baseline_treatment": (
                    "fixed scoring baseline; baseline uncertainty not resampled"),
                "randomization": (
                    "deterministic seed from data-path, component, and condition"),
            },
            "run_consistency": run_consistency,
        }

    def compute_ft(self, selected_comp: int,
                   scoring_version: str = SCORING_V4, *,
                   include_qc: bool = True,
                   csp_cv_folds: int = DEFAULT_CSP_CV_FOLDS) -> FTResult:
        """Transform/score a 1-based component (up to five are shown by UI).

        ``csp_cv_folds`` controls how the overt condition is projected.  The
        CSP filter is fitted on overt movement, so projecting the same overt
        trials through it makes criterion 3 an in-sample statistic while quasi
        and imagery remain out-of-sample — the transfer ratio in criterion 4 is
        biased low as a result.  With ``csp_cv_folds >= 2`` the overt map is
        built from held-out trials only, putting all three conditions on the
        same footing.  Pass 0 or 1 for the historical in-sample behaviour.
        """
        if self.projForward is None or self.projInverse is None:
            raise RuntimeError("run_start must complete before compute_ft")
        if selected_comp < 1 or selected_comp > self.projForward.shape[1]:
            raise ValueError(
                f"selected_comp must be between 1 and "
                f"{self.projForward.shape[1]}")
        idx = selected_comp - 1
        pattern = self.projForward[:, idx]

        csp_cv_folds = int(csp_cv_folds)
        if csp_cv_folds < 0:
            raise ValueError("csp_cv_folds must be non-negative")
        cross_validated = csp_cv_folds >= 2
        folds = self._cv_overt_filters(idx, csp_cv_folds) if cross_validated else []

        markers_mode = self.extract.mode == "markers"
        normalization = self._normalization_mode()
        tf_norm = []
        trial_power = []
        baselines = []
        pooled_base = None
        if markers_mode and self.extract.marker_baseline == "pooled":
            pooled_base = self._rest_baseline(idx, condition=None)
        elif not markers_mode:
            base_slice = rng_inclusive(
                self.tvec, -1.5, -1.0, strict=True, warn=False,
                label="trigger prestimulus baseline",
                diagnostics=self._window_diagnostics)
        for cond in (REAL, QUASI, IMAG):
            cv_cond = cross_validated and cond == REAL
            if cv_cond:
                X, rest_power = self._cv_overt_tf(
                    idx, folds, with_rest=markers_mode)
            else:
                X = self._tf_trials(self.broad[cond][ACTIVE], idx, ACTIVE)
            Xmed = np.median(X, axis=1)
            if markers_mode:
                if cv_cond:
                    # Held-out overt rest, in the fold filters' own units.
                    base = self._cv_marker_baseline(idx, folds, rest_power)
                else:
                    base = (pooled_base if pooled_base is not None
                            else self._rest_baseline(idx, condition=cond))
            else:
                # Trigger mode baselines the prestimulus part of the same
                # epochs, so the held-out projection already covers it.
                base = np.median(X[base_slice, :, :], axis=(0, 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                tf_db = (10.0 * np.log10(Xmed / base[None, :])).T
            tf_norm.append(tf_db)
            trial_power.append(X)
            baselines.append(base)

        current_provenance = self._build_provenance()
        # The historical implementation awarded one criterion-1 point when no
        # low-end jump existed.  Preserve that quirk only when legacy scoring is
        # explicitly requested; v2/v3 use the truthful candidate count of zero.
        legacy_score = str(scoring_version).strip().casefold() in {
            "legacy", "v1", "legacy-v1"}
        scored_n_erd = max(self.n_erd, 1) if legacy_score else self.n_erd
        independent_erd_score = str(scoring_version).strip().casefold() in {
            "v3", SCORING_V3, "v4", SCORING_V4}
        sustained_support = (
            self._sustained_erd_support(
                trial_power, baselines, selected_comp=selected_comp,
                data_path_version=current_provenance["data_path_version"])
            if independent_erd_score else None)
        score = compute_scores(
            tf_norm, self.freqNeeded, self.tvec_active, pattern,
            self.ideal_pattern, scored_n_erd, version=scoring_version,
            pattern_labels=self.channel_labels,
            ideal_pattern_labels=self.channel_labels,
            data_path_version=current_provenance["data_path_version"],
            normalization_mode=normalization,
            sustained_erd_support=sustained_support)
        current_provenance["scoring_version"] = getattr(
            score, "scoring_version", scoring_version)
        current_provenance["threshold_profile"] = getattr(
            score, "threshold_profile", None)
        current_provenance["thresholds_calibrated"] = getattr(
            score, "thresholds_calibrated", None)
        current_provenance["peak_erd_definition"] = {
            "method": getattr(score, "peak_erd_method", None),
            "frequency_window_hz": getattr(
                score, "peak_erd_frequency_window_hz", None),
            "rolling_window_s": getattr(
                score, "peak_erd_rolling_window_s", None),
            "strongest_window_s": getattr(
                score, "peak_erd_strongest_window_s", {}),
        }
        if sustained_support is not None:
            current_provenance["sustained_erd_definition"] = {
                "frequency_window_hz": [9.0, 13.0],
                "time_window_s": [0.5, 2.0],
                "statistic": "negative median of normalized component TF dB",
                "temporal_coverage": (
                    "fraction of time samples with frequency-median change < 0 dB"),
            }
        current_provenance["detected_n_erd"] = int(self.n_erd)
        current_provenance["scored_n_erd"] = int(scored_n_erd)
        current_provenance["selected_component"] = int(selected_comp)
        current_provenance["physiological_qc"] = (
            qc.QC_VERSION if include_qc else "not-computed")
        overt_projection = (
            "cross-validated-kfold" if cross_validated else "in-sample")
        csp_cv = self._cv_summary(folds, csp_cv_folds, cross_validated)
        current_provenance["overt_projection"] = overt_projection
        current_provenance["csp_cv"] = copy.deepcopy(csp_cv)
        physiological_qc = self.compute_physiological_qc() if include_qc else None
        self.provenance = current_provenance
        return FTResult(tf_norm=tf_norm, score=score, selected_comp=selected_comp,
                        pattern=pattern, normalization=normalization,
                        trial_counts=copy.deepcopy(self.trial_counts),
                        physiological_qc=physiological_qc,
                        provenance=copy.deepcopy(current_provenance),
                        overt_projection=overt_projection,
                        csp_cv=copy.deepcopy(csp_cv))


def _bootstrap_erd_estimates(region, baseline, rng, resamples):
    """Schedule small batches without changing RNG order or any reduction.

    Only trial indices are queued; each worker allocates one sampled region
    at a time instead of a resamples x time x trials x frequencies tensor.
    """
    n_trials = region.shape[1]
    draws = [rng.integers(0, n_trials, size=n_trials) for _ in range(resamples)]

    def estimate_batch(batch):
        values = []
        for take in batch:
            power = np.median(region[:, take, :], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                db = 10.0 * np.log10(power / baseline[None, :])
            values.append(-np.median(db))
        return values

    batches = [draws[start:start + 16] for start in range(0, resamples, 16)]
    return np.concatenate(execution.ordered_map(
        estimate_batch, batches, working_bytes=4 * region.nbytes))


def _crop_to_epoch_span(signal, windows, *, margin):
    """Restrict a recording to the span its epochs actually read.

    Returns ``(offset, cropped)``; add ``-offset`` to every onset to keep it
    pointing at the same sample.  Only windows that lie wholly inside the
    recording are covered, which is exactly the set :func:`io_h5.epoch` keeps,
    so cropping cannot change which trials survive the boundary check: a window
    that overran the recording still overruns the crop, and one that fitted is
    covered by construction.

    ``margin`` is the number of samples of real signal kept beyond that span on
    each side.  Zero-phase FIR filtering makes output sample ``n`` depend on
    input samples ``n ± (taps - 1)``, so a margin at least that wide leaves
    every epoched sample bit-identical to filtering the whole recording; see
    :meth:`KLHPipeline._filter_edge_margin`.
    """
    signal = np.asarray(signal)
    n_samples = int(signal.shape[0])
    margin = int(margin)
    lo, hi = None, None
    for onsets, epoind in windows:
        onsets = np.asarray(onsets, dtype=int).ravel()
        if onsets.size == 0 or epoind.size == 0:
            continue
        first = int(epoind[0])
        last = int(epoind[-1])
        starts = onsets + first
        stops = onsets + last + 1
        # Mirror io_h5.epoch: a window is read only if it fits entirely.
        keep = (starts >= 0) & (stops <= n_samples)
        if not np.any(keep):
            continue
        lo = int(starts[keep].min()) if lo is None else min(
            lo, int(starts[keep].min()))
        hi = int(stops[keep].max()) if hi is None else max(
            hi, int(stops[keep].max()))
    if lo is None:
        # No epoch survives the boundary check; nothing to crop against.
        return 0, signal
    lo = max(0, lo - margin)
    hi = min(n_samples, hi + margin)
    if lo == 0 and hi == n_samples:
        return 0, signal
    return lo, signal[lo:hi]


def _cat_trials(a, b):
    """Concatenate two (n_time, n_trials, n_ch) arrays along the trial axis."""
    if a is None:
        return b
    if b is None or b.shape[1] == 0:
        return a
    return np.concatenate((a, b), axis=1)


def _common_xdf_montage(files, reference_labels):
    """Return the reference-ordered channel intersection for all XDF files.

    Matching uses the same case/punctuation-insensitive normalisation as the
    full XDF loader.  The accompanying details make every channel excluded by
    the common-montage decision auditable, both globally and per recording.
    """
    files = list(files)
    reference_labels = list(reference_labels)
    if not files:
        raise ValueError("at least one XDF recording is required")

    reference_keys = [io_h5._norm_label(label) for label in reference_labels]
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError("KLH reference montage contains ambiguous channel labels")
    reference_key_set = set(reference_keys)
    key_to_reference = dict(zip(reference_keys, reference_labels))

    common_keys = reference_key_set.copy()
    file_rows = []
    for path in files:
        labels = io_h5.load_xdf_channel_labels(path)
        if labels is None:
            raise ValueError(
                f"channel labels missing in {os.path.basename(path)}; cannot "
                "match a common KLH reference montage")
        labels = [str(label) for label in labels]
        normalised = [io_h5._norm_label(label) for label in labels]
        available_keys = set(normalised)
        matched_keys = reference_key_set.intersection(available_keys)
        common_keys.intersection_update(matched_keys)

        duplicate_keys = sorted({key for key in normalised
                                 if key and normalised.count(key) > 1})
        duplicate_reference_keys = [
            key for key in duplicate_keys if key in reference_key_set]
        if duplicate_reference_keys:
            ambiguous = [key_to_reference[key]
                         for key in duplicate_reference_keys]
            raise ValueError(
                f"duplicate channel labels matching the KLH reference montage "
                f"in {os.path.basename(path)}: {ambiguous}")
        file_rows.append({
            "file": os.path.abspath(path),
            "recorded_channel_count": len(labels),
            "matched_reference_count": len(matched_keys),
            "_matched_keys": matched_keys,
            "missing_reference_labels": [
                label for label, key in zip(reference_labels, reference_keys)
                if key not in matched_keys],
            "non_reference_labels": [
                label for label, key in zip(labels, normalised)
                if key not in reference_key_set],
            "duplicate_normalized_labels": [
                key_to_reference.get(key, key) for key in duplicate_keys],
        })

    selected = [label for label, key in zip(reference_labels, reference_keys)
                if key in common_keys]
    if len(selected) < 2:
        raise ValueError(
            "selected XDF recordings share fewer than two channels from the "
            "KLH reference montage")

    for row in file_rows:
        matched_keys = row.pop("_matched_keys")
        row["selected_channel_count"] = len(selected)
        row["present_but_excluded_from_common"] = [
            label for label, key in zip(reference_labels, reference_keys)
            if key in matched_keys and key not in common_keys]
    # A deterministic order keeps montage provenance stable when callers merely
    # reorder the same selected recordings.
    file_rows.sort(key=lambda row: os.path.normcase(row["file"]))
    details = {
        "strategy": "intersection-of-normalized-labels-across-all-files",
        "files_scanned": len(files),
        "reference_channel_count": len(reference_labels),
        "common_channel_count": len(selected),
        "excluded_reference_labels": [
            label for label, key in zip(reference_labels, reference_keys)
            if key not in common_keys],
        "per_file": file_rows,
    }
    return selected, details


def _validate_unique_input_files(*condition_files) -> None:
    """Reject a recording listed more than once within or across conditions.

    Epoch arrays are pooled along their trial axis, whereas per-recording QC is
    keyed by file path.  Allowing a duplicate would therefore weight its trials
    more than once while representing the run only once in the consistency
    summary.  Compare normalized absolute paths so harmless spelling variants
    such as ``run.h5`` and ``.\\run.h5`` cannot bypass the guard.
    """
    seen: dict[str, tuple[str, str]] = {}
    for condition, paths in condition_files:
        condition = str(condition)
        for path in paths:
            absolute = os.path.abspath(os.fspath(path))
            canonical = os.path.normcase(os.path.normpath(absolute))
            previous = seen.get(canonical)
            if previous is not None:
                previous_condition, previous_path = previous
                if previous_condition == condition:
                    detail = f"more than once in condition '{condition}'"
                else:
                    detail = (
                        f"in conditions '{previous_condition}' and "
                        f"'{condition}'")
                raise ValueError(
                    f"duplicate input recording {absolute!r}: listed {detail}; "
                    f"first occurrence was {previous_path!r}")
            seen[canonical] = (condition, absolute)


def detect_eigenvalue_candidates(evals, ch_num: int, *, scale: str = "raw"):
    """Detect low/high CSP candidates with montage-proportional cutoffs.

    Returns ``(low, high, n_erd, diagnostic)``.  ``n_erd`` and ``low`` are zero
    and empty when no significant low-end gap exists.  This is deliberately
    separate from the always-inspectable top-five component patterns. ``scale``
    can be the MATLAB-compatible raw eigenvalues or symmetric log-odds.
    """
    evals = np.asarray(evals, dtype=float).ravel()
    ch_num = int(ch_num)
    if evals.size != ch_num or ch_num < 2:
        raise ValueError("evals must contain one value per channel/component")
    scale = str(scale).strip().lower()
    if scale == "logit":
        eps = np.finfo(float).eps
        clipped = np.clip(evals, eps, 1.0 - eps)
        spectrum = np.log(clipped / (1.0 - clipped))
    elif scale == "raw":
        spectrum = evals
    else:
        raise ValueError("candidate scale must be 'raw' or 'logit'")
    ediff = np.diff(spectrum)
    threshold = float(np.median(ediff) + 1.4826 * 3 * mad1(ediff))
    steps = np.where(ediff > threshold)[0] + 1
    low_lim = max(2, int(round(ch_num * 10 / io_h5.N_EEG)))
    high_lim = int(round(ch_num * 30 / io_h5.N_EEG))
    low = steps[steps < low_lim]
    high = steps[steps > high_lim]
    n_erd = int(low.max()) if low.size else 0
    high_start = int(high.min()) if high.size else ch_num
    low_inds = np.arange(0, n_erd, dtype=int)
    high_inds = np.arange(high_start, ch_num, dtype=int)
    diagnostic = {
        "scale": scale,
        "threshold": threshold,
        "step_indices": steps.astype(int).tolist(),
        "low_limit": int(low_lim),
        "high_limit": int(high_lim),
        "detected_low_steps": low.astype(int).tolist(),
        "detected_high_steps": high.astype(int).tolist(),
    }
    return low_inds, high_inds, n_erd, diagnostic


def covariance_pencil_condition(cov_active, cov_rest) -> float:
    """Condition number of the trace-normalized bounded-CSP denominator."""
    cov_active = np.asarray(cov_active, dtype=float)
    cov_rest = np.asarray(cov_rest, dtype=float)
    if (cov_active.ndim != 2 or cov_active.shape[0] != cov_active.shape[1]
            or cov_rest.shape != cov_active.shape):
        raise ValueError("CSP covariances must be same-sized square matrices")
    trace_active = float(np.trace(cov_active))
    trace_rest = float(np.trace(cov_rest))
    if (not np.isfinite(trace_active + trace_rest)
            or trace_active <= 0 or trace_rest <= 0):
        raise ValueError("CSP covariance has a non-positive or non-finite trace")
    pencil = cov_active / trace_active + cov_rest / trace_rest
    return float(np.linalg.cond(pencil))


def _empty_trial_counts() -> dict:
    """Fresh nested retained/dropped counter structure."""
    def counter():
        return {
            "requested": 0,
            "eligible": 0,
            "retained": 0,
            "invalid_duration": 0,
            "out_of_bounds": 0,
            "dropped": 0,
        }

    return {
        name: {"active": counter(), "rest": counter()}
        for name in CONDITION_NAMES
    } | {"by_file": []}


def _superlet_padding_samples(freqs, fs, *, ncyc, order_interval) -> int:
    """Largest half-wavelet support used by :func:`superlet.aslt`."""
    freqs = np.asarray(freqs, dtype=float).ravel()
    orders = np.trunc(np.linspace(
        order_interval[0], order_interval[1], freqs.size)).astype(int)
    orders = np.maximum(orders, 1)
    largest = 0
    for freq, order in zip(freqs, orders):
        # Additive superresolution (mult=0): cycles ncyc..ncyc+order-1.
        for cycles in range(ncyc, ncyc + int(order)):
            sd = (cycles / 2.0) * (1.0 / freq) / 2.5
            raw_length = int(np.trunc(6 * sd * fs))
            largest = max(largest, raw_length // 2)
    return int(largest)
