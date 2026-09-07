r"""Headless runner for the KLH overt-template transfer analysis.

Examples
--------
H5 with the trigger channel::

    python run_cli.py --folder "..\Casting\RawData" \
        --real om_s1,om_s2 --quasi qm_s1,qm_s2 --imag im_s1,im_s2 \
        --component 1 --out result.png

XDF with embedded markers::

    python run_cli.py --folder "D:\ExpData\...\02TST" \
        --file-type xdf --extract markers \
        --real om1,om2 --quasi qm1,qm2 --imag im1,im2 \
        --active-cond right_microrepeat --rest-cond rest \
        --marker-shift -4 --preparation 2 --motor-recovery 2 \
        --baseline condition-specific --padding 1.5 --score-version v4 \
        --component 1 --out result.png

Filenames are comma-separated and omit the extension. ``--fs`` is the native
H5 recording rate; XDF rates come from stream metadata and ``--fs`` is ignored
for XDF. EEG (not a categorical trigger) is resampled to ``--processing-fs``
so fixed-rate filters remain valid.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from klh import superlet, execution
from klh.pipeline import KLHPipeline, ExtractConfig, DEFAULT_CSP_CV_FOLDS
from klh.scoring import SCORING_LEGACY, SCORING_V2, SCORING_V3, SCORING_V4

_EXT = {"h5": ".h5", "xdf": ".xdf"}
MIN_H5_CHANNELS = 2
MAX_REFERENCE_CHANNELS = 64
MAX_DISPLAY_COMPONENTS = 5
SCORE_ALIASES = {
    "v4": SCORING_V4,
    SCORING_V4: SCORING_V4,
    "v3": SCORING_V3,
    SCORING_V3: SCORING_V3,
    "v2": SCORING_V2,
    SCORING_V2: SCORING_V2,
    "legacy": SCORING_LEGACY,
    SCORING_LEGACY: SCORING_LEGACY,
}


def _files(folder, names, ext):
    return [os.path.join(folder, name.strip() + ext)
            for name in names.split(",") if name.strip()]


def _score_version(value):
    try:
        return SCORE_ALIASES[value.casefold()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"choose v4/{SCORING_V4}, v3/{SCORING_V3}, "
            "v2/scoring-v2, or legacy/legacy-v1") from exc


def _supports_parameter(callable_obj, name):
    try:
        params = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind == inspect.Parameter.VAR_KEYWORD
               for p in params)


def _make_extract_config(args):
    values = {
        "mode": args.extract,
        "active_cond": args.active_cond,
        "rest_cond": args.rest_cond,
        "preparation_s": args.preparation,
        "recovery_s": args.motor_recovery,
        "cut_first": args.cut_first,
        "cut_last": args.cut_last,
        "marker_baseline": args.marker_baseline,
        "rest_transition_skip": args.rest_transition_skip,
        "marker_padding": args.marker_padding,
        "auto_marker_shift": not args.no_auto_marker_shift,
        "marker_shift_s": args.marker_shift,
    }
    supported = {
        key: value for key, value in values.items()
        if _supports_parameter(ExtractConfig, key)
    }
    ignored = sorted(set(values) - set(supported))
    return ExtractConfig(**supported), ignored


def _component_count(result):
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
    available = [number for number in candidates if number >= 0]
    return min([MAX_DISPLAY_COMPONENTS, *available]) if available else 0


def _score_split(score):
    fields = (
        "subject_score", "subject_score_max",
        "component_score", "component_score_max",
    )
    values = [getattr(score, field, None) for field in fields]
    if all(value is not None for value in values):
        return tuple(int(value) for value in values)
    scores = np.asarray(getattr(score, "scores", []), dtype=float).ravel()
    maxima = np.asarray(getattr(score, "scores_max", []), dtype=float).ravel()
    if scores.size and maxima.size == scores.size:
        return (int(np.nansum(scores[:1])), int(np.nansum(maxima[:1])),
                int(np.nansum(scores[1:])), int(np.nansum(maxima[1:])))
    return None


def _print_csp_cv(result):
    """Report how the overt condition was projected, and how well folds matched."""
    projection = getattr(result, "overt_projection", None)
    info = getattr(result, "csp_cv", None)
    if projection is None and not info:
        return
    print(f"  overt projection : {projection or 'unreported'}")
    if not isinstance(info, dict) or not info:
        return
    if not info.get("enabled", False):
        reason = info.get("reason")
        if reason:
            print(f"    NOTE: {reason}")
        return
    print(f"    folds : {info.get('folds', 'unreported')}"
          f"  ({info.get('scope', 'unreported')})")
    print(f"    fold pattern similarity to the selected component : "
          f"min {info.get('min_pattern_similarity', float('nan')):.3f}, "
          f"median {info.get('median_pattern_similarity', float('nan')):.3f}")
    print(f"    matched components per fold : {info.get('matched_components')}"
          f"  (min margin {info.get('min_match_margin', float('nan')):.3f})")
    residual = info.get("residual_leak")
    if residual:
        print(f"    residual leak : {residual}")
    for warning in info.get("warnings", []):
        print(f"    WARNING: {warning}")


def _first_metadata(objects, *names):
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


def _compact(value):
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(value).tolist(), ensure_ascii=False,
                          default=str)
    return str(value)


def _recording_rate_message(args):
    """Explain whether the user-supplied recording-rate hint is operative."""
    if args.file_type == "xdf":
        return ("XDF recording rates: read from stream metadata "
                "(--fs is ignored for XDF).")
    return f"H5 recording rate: {args.fs:g} Hz (used for trigger timing)."


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run KLH overt-template transfer analysis headlessly.")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--quasi", required=True)
    parser.add_argument("--imag", required=True)
    parser.add_argument("--file-type", choices=["h5", "xdf"], default="h5")
    parser.add_argument(
        "--fs", type=float, default=1000.0,
        help=("native H5 recording rate used for trigger timing; accepted but "
              "ignored for XDF, whose rates come from stream metadata"))
    parser.add_argument(
        "--processing-fs", type=float, default=1000.0,
        help="EEG processing rate (fixed-rate filters require 1000)")
    parser.add_argument(
        "--nchan", type=int, default=64,
        help="first N EEG channels for H5 (2..64); XDF matches by label")
    parser.add_argument(
        "--extract", choices=["trigger", "markers"], default="trigger",
        help="trigger channel for H5 or embedded markers for XDF")
    parser.add_argument("--active-cond", default="right_microrepeat")
    parser.add_argument("--rest-cond", default="rest")
    parser.add_argument(
        "--preparation", type=float, default=2.0,
        help="seconds excluded after the corrected active marker before motor time zero")
    parser.add_argument(
        "--motor-recovery", type=float, default=2.0,
        help="seconds retained after each marker-defined active trial")
    parser.add_argument(
        "--cut-first", type=float, default=0.0,
        help="optional extra trim after the preparation interval")
    parser.add_argument("--cut-last", type=float, default=0.0)
    parser.add_argument(
        "--marker-baseline", "--baseline", dest="marker_baseline",
        choices=["condition-specific", "pooled"],
        default="condition-specific",
        help="rest normalization for marker recordings")
    parser.add_argument(
        "--rest-transition-skip", type=float, default=1.0,
        help="seconds excluded from the beginning of marker rest")
    parser.add_argument(
        "--marker-padding", "--padding", dest="marker_padding",
        type=float, default=1.5,
        help="seconds of real neighboring data around marker TF epochs")
    parser.add_argument(
        "--no-auto-marker-shift", action="store_true",
        help="do not measure each recording's amplifier timestamp lag; apply "
             "--marker-shift to every recording instead")
    parser.add_argument(
        "--marker-shift", type=float, default=-4.0,
        help="seconds added to all embedded active/rest onsets when detection "
             "is switched off with --no-auto-marker-shift (default: -4.0)")
    parser.add_argument(
        "--candidate-scale", choices=["raw", "logit"], default="raw",
        help="eigenvalue candidate scale (raw is historical; logit is optional)")
    parser.add_argument(
        "--csp-shrinkage", type=float, default=0.0,
        help="optional covariance shrinkage in [0,1); zero is historical")
    parser.add_argument(
        "--score-version", type=_score_version,
        default=SCORING_V4,
        metavar="{v4,v3,v2,legacy}",
        help=("500 ms rolling-peak + independent quasi/imagery ERD scoring-v4 "
              "with provisional thresholds (default), archived scoring-v3, "
              "scoring-v2, or legacy-v1"))
    parser.add_argument(
        "--csp-cv-folds", type=int, default=DEFAULT_CSP_CV_FOLDS,
        help="folds used to score overt ERD on held-out trials (default "
             f"{DEFAULT_CSP_CV_FOLDS}); 0 or 1 restores the in-sample "
             "overt estimate, which biases criterion 4 low")
    parser.add_argument(
        "--analysis-workers", type=int, default=execution.DEFAULT_WORKERS,
        help="threads for covariance fits, bootstrap batches and filter bands "
             "(default 2); 0 or 1 runs serially; higher counts use more memory")
    parser.add_argument(
        "--superlet-workers", type=int, default=None,
        help=(f"processes splitting the time-frequency transform (default "
              f"{superlet.DEFAULT_WORKERS}, or KLH_SUPERLET_WORKERS); 0 or 1 "
              "keeps it in one process. Output is bit-identical either way; "
              "each worker costs roughly 170 MB"))
    parser.add_argument(
        "--component", type=int, default=1,
        help="1..min(5, available CSP components); 1 = strongest ERD")
    parser.add_argument(
        "--out", default=None,
        help="optional PNG of the TF maps and transfer score")
    parser.add_argument(
        "--report-json", default=None,
        help="optional machine-readable score, QC and provenance report")
    return parser


def _validate_args(parser, args):
    if args.analysis_workers < 0:
        parser.error("--analysis-workers cannot be negative")
    if args.fs <= 0 or args.processing_fs <= 0:
        parser.error("--fs and --processing-fs must be positive")
    if not 1 <= args.component <= MAX_DISPLAY_COMPONENTS:
        parser.error(
            f"--component must be between 1 and {MAX_DISPLAY_COMPONENTS}")
    if min(args.preparation, args.motor_recovery, args.cut_first, args.cut_last,
           args.rest_transition_skip, args.marker_padding) < 0:
        parser.error(
            "preparation, recovery, cuts, transition skip and padding cannot "
            "be negative")
    if not np.isfinite(args.marker_shift):
        parser.error("--marker-shift must be finite")
    if not 0.0 <= args.csp_shrinkage < 1.0:
        parser.error("--csp-shrinkage must satisfy 0 <= value < 1")
    if args.extract == "markers" and args.marker_padding < 1.0:
        parser.error(
            "--marker-padding must be at least 1.0 s for the current TF grid")
    if args.file_type == "h5":
        if not MIN_H5_CHANNELS <= args.nchan <= MAX_REFERENCE_CHANNELS:
            parser.error(
                f"H5 --nchan must be {MIN_H5_CHANNELS}.."
                f"{MAX_REFERENCE_CHANNELS}; the reference montage has 64 channels")
        if args.extract != "trigger":
            parser.error("H5 recordings require --extract trigger")
    elif args.extract != "markers":
        parser.error("XDF recordings require --extract markers")
    return args


def main(argv=None):
    parser = build_parser()
    args = _validate_args(parser, parser.parse_args(argv))
    execution.configure_workers(args.analysis_workers)
    if args.superlet_workers is not None:
        superlet.configure_workers(args.superlet_workers)

    extension = _EXT[args.file_type]
    pipe = KLHPipeline()
    files_real = _files(args.folder, args.real, extension)
    files_quasi = _files(args.folder, args.quasi, extension)
    files_imag = _files(args.folder, args.imag, extension)
    all_files = files_real + files_quasi + files_imag
    missing = [path for path in all_files if not os.path.exists(path)]
    if missing:
        parser.error("input file(s) not found: " + ", ".join(missing[:8]))

    extract, ignored = _make_extract_config(args)
    if ignored:
        print("WARNING: this pipeline build ignores unsupported extraction "
              f"option(s): {', '.join(ignored)}", file=sys.stderr)
    if args.file_type == "xdf" and args.score_version == "legacy-v1":
        print(
            "WARNING: legacy-v1 selects the historical score formula only; "
            "the XDF marker data path is not MATLAB/cohort-compatible.",
            file=sys.stderr,
        )

    print("input recordings:")
    for condition, condition_paths in (
            ("overt", files_real), ("quasi", files_quasi),
            ("imagery", files_imag)):
        names = ", ".join(os.path.basename(path) for path in condition_paths)
        print(f"  {condition}: {names or '(none)'}")
    print(_recording_rate_message(args))

    print("running analysis (this can take a minute)…")
    start_kwargs = {
        "extract": extract,
        "file_type": args.file_type,
        "ch_num": args.nchan,
        "fs": args.fs,
    }
    if _supports_parameter(pipe.run_start, "processing_fs"):
        start_kwargs["processing_fs"] = args.processing_fs
    elif args.processing_fs != 1000.0:
        print("WARNING: this pipeline build cannot set --processing-fs; "
              "using its internal rate", file=sys.stderr)
    if _supports_parameter(pipe.run_start, "candidate_scale"):
        start_kwargs["candidate_scale"] = args.candidate_scale
    elif args.candidate_scale != "raw":
        print("WARNING: this pipeline build cannot set --candidate-scale; "
              "using raw candidates", file=sys.stderr)
    if _supports_parameter(pipe.run_start, "csp_shrinkage"):
        start_kwargs["csp_shrinkage"] = args.csp_shrinkage
    elif args.csp_shrinkage:
        print("WARNING: this pipeline build cannot set --csp-shrinkage; "
              "using zero", file=sys.stderr)
    start = pipe.run_start(
        files_real, files_quasi, files_imag, **start_kwargs)

    offsets = getattr(start, "video_onset_offsets", {})
    if offsets:
        print("video_onset position in each recording:")
        for name, offset in offsets.items():
            print(f"  {name}: {offset:+.3f} s")
    evals = np.asarray(getattr(start, "evals", []))
    if evals.size:
        print(f"eigenvalues: {evals.min():.3f}..{evals.max():.3f}  "
              f"| ERD candidates: {getattr(start, 'n_erd', 'unreported')}")

    condition_number = _first_metadata(
        (start,), "condition_number", "covariance_condition_number")
    if condition_number is not None:
        try:
            rendered_condition = f"{float(condition_number):.3g}"
        except (TypeError, ValueError):
            rendered_condition = _compact(condition_number)
        print(f"CSP covariance condition number: {rendered_condition}")
    warning = _first_metadata(
        (start,), "condition_warning", "conditioning_warning")
    if warning:
        print(f"WARNING: {warning}")
    start_trials = _first_metadata(
        (start,), "trial_counts", "retained_dropped_trials")
    if start_trials is not None:
        print(f"retained/dropped trials: {_compact(start_trials)}")
    start_provenance = getattr(start, "provenance", None)
    if start_provenance:
        print(f"start provenance: {_compact(start_provenance)}")
    channel_qc = getattr(start, "channel_qc", {})
    if isinstance(channel_qc, dict):
        flagged = channel_qc.get("flagged", [])
        print("persistent channel outliers: " +
              (", ".join(flagged) if flagged else "none detected"))
    ranking = getattr(start, "component_ranking", [])
    if ranking:
        print("advisory motor-topography ranking:")
        for item in ranking:
            print(
                f"  component #{item['component']}: "
                f"plausibility={item['plausibility']:.3f}, "
                f"central={item['central_energy_fraction']:.3f}, "
                f"peripheral={item['peripheral_energy_fraction']:.3f}, "
                f"ideal|r|={item['ideal_similarity_abs']:.3f}")
    labels = getattr(start, "channel_labels", None)
    if labels is not None and len(labels) < MAX_REFERENCE_CHANNELS:
        print(
            f"WARNING: {len(labels)}-channel montage; transfer scores are not "
            "directly comparable with the 64-channel reference cohort.")

    available_components = _component_count(start)
    if not 1 <= args.component <= available_components:
        parser.error(
            f"component {args.component} unavailable; this result exposes "
            f"1..{available_components}")

    ft_kwargs = {}
    if _supports_parameter(pipe.compute_ft, "scoring_version"):
        ft_kwargs["scoring_version"] = args.score_version
    elif args.score_version != "legacy-v1":
        print("WARNING: this pipeline build does not support score-version selection; "
              "using its default score", file=sys.stderr)
    if _supports_parameter(pipe.compute_ft, "csp_cv_folds"):
        ft_kwargs["csp_cv_folds"] = args.csp_cv_folds
    elif args.csp_cv_folds >= 2:
        print("WARNING: this pipeline build cannot cross-validate the CSP; "
              "overt peak ERD stays in-sample", file=sys.stderr)
    result = pipe.compute_ft(args.component, **ft_kwargs)
    score = result.score

    print(f"\nselected component #{args.component}")
    print(f"  pattern similarity : {score.pat_sim:.3f}")
    print(f"  peak ERD [overt,quasi,imagery] : {score.peakERD.round(2)}")
    peak_method = getattr(score, "peak_erd_method", None)
    if peak_method:
        description = (
            f"{peak_method}; "
            f"{getattr(score, 'peak_erd_frequency_window_hz', None)} Hz")
        rolling_window = getattr(score, "peak_erd_rolling_window_s", None)
        if rolling_window is not None:
            description += f"; rolling window {rolling_window} s"
        print(f"  peak definition : {description}")
        strongest = getattr(score, "peak_erd_strongest_window_s", {}) or {}
        if strongest:
            print(f"  strongest peak windows : {_compact(strongest)}")
    sustained = getattr(score, "sustained_erd_db", {})
    if sustained:
        print(
            "  sustained mu ERD [quasi,imagery] : "
            f"[{sustained.get('quasi', np.nan):.2f}, "
            f"{sustained.get('imagery', np.nan):.2f}] dB")
        coverage = getattr(score, "sustained_erd_temporal_coverage", {})
        print(
            "  temporal ERD coverage [quasi,imagery] : "
            f"[{coverage.get('quasi', np.nan):.1%}, "
            f"{coverage.get('imagery', np.nan):.1%}]")
        ci = getattr(score, "sustained_erd_ci_db", {})
        for condition in ("quasi", "imagery"):
            bounds = ci.get(condition, [np.nan, np.nan])
            print(
                f"  {condition} sustained ERD 95% CI : "
                f"{bounds[0]:.2f}..{bounds[1]:.2f} dB")
        for condition, summary in getattr(
                score, "sustained_erd_run_consistency", {}).items():
            print(
                f"  {condition} component run consistency : "
                f"{summary.get('positive_erd_runs', 0)}/"
                f"{summary.get('retained_runs', 0)} retained runs with ERD")
    print(f"  sub-scores : {score.scores.astype(int)}  "
          f"(max {score.scores_max})")
    split = _score_split(score)
    if split is not None:
        subject, subject_max, component, component_max = split
        print(f"  subject-level : {subject} / {subject_max}")
        print(f"  selected component : {component} / {component_max}")
    print(f"  OVERT-TEMPLATE TRANSFER SCORE : "
          f"{score.total} / {score.total_max}")
    print("  (This is not a general sensorimotor data-quality rating.)")
    print(f"  threshold profile : {getattr(score, 'threshold_profile', 'unreported')}")
    if getattr(score, "thresholds_calibrated", None) is False:
        print(
            f"  WARNING: {getattr(score, 'scoring_version', 'score')} thresholds "
            "are provisional and not cohort-calibrated")
    _print_csp_cv(result)

    objects = (result, score, pipe)
    normalization = _first_metadata(
        objects, "normalization", "normalization_mode", "baseline_mode")
    score_version = _first_metadata(
        objects, "scoring_version", "score_version")
    data_path = _first_metadata(
        objects, "data_path_version", "data_path", "pipeline_version")
    result_trials = _first_metadata(
        objects, "trial_counts", "retained_dropped_trials")
    print("  normalization : " +
          (_compact(normalization) if normalization is not None else "unreported"))
    print("  score formula : " +
          (_compact(score_version) if score_version is not None else "unreported"))
    print("  data-path version : " +
          (_compact(data_path) if data_path is not None else "unreported"))
    if result_trials is not None:
        print(f"  retained/dropped trials : {_compact(result_trials)}")
    result_provenance = getattr(result, "provenance", None)
    if result_provenance:
        print(f"  result provenance : {_compact(result_provenance)}")

    physiological = getattr(result, "physiological_qc", None)
    if physiological is not None:
        print("\nPHYSIOLOGICAL QC (separate from transfer score)")
        print(f"  hand/event: {physiological.active_condition}")
        print(f"  reference: {physiological.reference}")
        for condition, bands in physiological.results.items():
            for band_name in ("mu", "beta"):
                value = bands.get(band_name)
                if value is None:
                    continue
                print(
                    f"  {condition} {band_name}: C3 {value.c3_db:+.2f} dB, "
                    f"C4 {value.c4_db:+.2f} dB, contra-ipsi "
                    f"{value.laterality_db:+.2f} dB "
                    f"(active/rest trials {value.active_trials}/{value.rest_trials})")
        for condition, bands in getattr(physiological, "consistency", {}).items():
            for band_name in ("mu", "beta"):
                summary = bands.get(band_name)
                if not summary or not summary.get("recordings"):
                    continue
                lo, hi = summary.get("range_laterality_db", [np.nan, np.nan])
                print(
                    f"  {condition} {band_name} run consistency: "
                    f"{summary['expected_laterality_count']}/"
                    f"{summary['recordings']} expected; median "
                    f"{summary['median_laterality_db']:+.2f} dB, range "
                    f"{lo:+.2f}..{hi:+.2f} dB")
        for warning in physiological.warnings:
            print(f"  WARNING: {warning}")

    if args.report_json:
        report = {
            "report_schema": "klh-fmri-quasi-report-v1",
            "inputs": {
                "overt": files_real,
                "quasi": files_quasi,
                "imagery": files_imag,
            },
            "selected_component": int(args.component),
            "score": _jsonable(score),
            "overt_projection": getattr(result, "overt_projection", None),
            "csp_cv": _jsonable(getattr(result, "csp_cv", {})),
            "physiological_qc": _jsonable(physiological),
            "component_ranking": _jsonable(
                getattr(start, "component_ranking", [])),
            "channel_qc": _jsonable(getattr(start, "channel_qc", {})),
            "trial_counts": _jsonable(result_trials),
            "provenance": _jsonable(result_provenance),
        }
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2,
                      allow_nan=False)
        print(f"\nsaved {args.report_json}")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(7, 8))
        names = ["OVERT", "QUASI", "IMAGERY"]
        time = getattr(pipe, "tvec_active", pipe.tvec)
        xlim = ((float(time[0]), float(time[-1]))
                if extract.mode == "markers" else (-2, 5.5))
        for index, axis in enumerate(axes):
            axis.contourf(time, pipe.freqNeeded, result.tf_norm[index], 60,
                          cmap="jet", vmin=-7, vmax=7)
            axis.axvline(0, color="k")
            if extract.mode == "markers":
                motor_end = getattr(pipe, "_motor_duration_s", None)
                if motor_end is not None:
                    axis.axvline(
                        motor_end, color="k", linestyle="--", linewidth=0.8)
            axis.set_xlim(*xlim)
            axis.set_title(names[index])
            axis.set_ylabel("Hz")
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(
            f"component #{args.component} — overt-template transfer "
            f"{score.total}/{score.total_max} "
            f"({getattr(score, 'scoring_version', 'unversioned')})")
        fig.tight_layout()
        fig.savefig(args.out, dpi=100)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
