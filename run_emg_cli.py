r"""Headless runner for the hand-EMG half of NeuroCasting.

Analyses every named recording and writes the whole output tree: a mean figure
per hand, one figure per trial, ``trial_metrics.csv`` beside them, and one
``emg_summary.csv`` for the participant. The desktop application runs the same
analysis but draws it on screen first and writes only when asked; ``run_cli.py``
is the equivalent runner for the EEG transfer score.

Example::

    python run_emg_cli.py --folder "D:\ExpData\MEG\Quasi fMRI\data\007TST" \
        --files om1,qm1,im1 --participant 007TST
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emgcasting.core import (DEFAULT_OUTPUT_ROOT, ProcessingConfig, parse_pair,
                             process_batch)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create per-recording mean left/right hand EMG figures.")
    parser.add_argument("--folder", required=True, help="participant data folder")
    parser.add_argument("--files", required=True, help="comma-separated recordings")
    parser.add_argument("--participant", default="",
                        help="output subfolder (default: data folder name)")
    parser.add_argument("--fs", type=float, default=0.0,
                        help="processing rate; 0 uses the recorded rate")
    parser.add_argument("--data-stream", default="NVX136_Data")
    parser.add_argument("--marker-stream", default="PsychoPyMarkers")
    parser.add_argument(
        "--no-auto-marker-shift", action="store_true",
        help="do not measure each recording's amplifier timestamp lag; apply "
             "--marker-shift to every recording instead")
    parser.add_argument(
        "--marker-shift", type=float, default=-4.0,
        help="seconds added to all embedded marker onsets when detection is "
             "switched off with --no-auto-marker-shift (default: -4.0)")
    parser.add_argument("--left-emg", default="Aux 1.1,Aux 1.2",
                        help="left bipolar pair: positive,negative")
    parser.add_argument("--right-emg", default="Aux 2.1,Aux 2.2",
                        help="right bipolar pair: positive,negative")
    parser.add_argument("--left-cond", default="left_microrepeat")
    parser.add_argument("--right-cond", default="right_microrepeat")
    parser.add_argument("--rest-cond", default="rest")
    parser.add_argument("--envelope", choices=["tkeo", "rms"], default="tkeo")
    parser.add_argument("--input-unit", choices=["V", "mV"], default="V")
    parser.add_argument("--notch", type=float, default=50.0)
    parser.add_argument("--band", default="20,150", help="low,high Hz")
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--step-ms", type=float, default=20.0)
    parser.add_argument("--pre", type=float, default=2.0)
    parser.add_argument("--post", type=float, default=2.0)
    parser.add_argument(
        "--trial-tail-ms", type=float,
        default=1000.0 * ProcessingConfig.trial_tail_s,
        help="classification tolerance appended after each scheduled trial "
             f"(default: {1000 * ProcessingConfig.trial_tail_s:g} ms)")
    parser.add_argument("--rest-trim", type=float, default=2.0,
                        help="seconds dropped from the end of each rest block "
                             "before the baseline is taken")
    parser.add_argument("--rest-trim-start", type=float, default=1.0,
                        help="seconds dropped from the start of each rest block "
                             "for trial-metric calibration")
    # Defaults come from ProcessingConfig so the CLI cannot drift from it.
    parser.add_argument("--peak-multiplier", type=float,
                        default=ProcessingConfig.peak_multiplier,
                        help="height a peak must reach, as a multiple of the "
                             "recording's own clean-rest background (default: "
                             f"{ProcessingConfig.peak_multiplier:g}); this is "
                             "the bar that carries the high/low decision")
    parser.add_argument("--background-multiplier", type=float,
                        default=ProcessingConfig.background_multiplier,
                        help="the lower shoulder the peak's width is measured "
                             "at, as a multiple of that same background "
                             f"(default: {ProcessingConfig.background_multiplier:g})")
    parser.add_argument(
        "--pre-reference-start-ms", type=float,
        default=1000.0 * ProcessingConfig.pre_reference_start_s,
        help="start of the per-trial quiet pre-movement reference interval")
    parser.add_argument(
        "--pre-reference-end-ms", type=float,
        default=1000.0 * ProcessingConfig.pre_reference_end_s,
        help="end of the quiet reference and start of the adaptive peak search")
    parser.add_argument(
        "--secondary-pre-multiplier", type=float,
        default=ProcessingConfig.secondary_pre_multiplier,
        help="adaptive peak height as a multiple of the trial's quiet interval")
    parser.add_argument(
        "--secondary-width-multiplier", type=float,
        default=ProcessingConfig.secondary_width_multiplier,
        help="adaptive width shoulder as a multiple of the quiet interval")
    parser.add_argument("--min-burst-ms", type=float,
                        default=ProcessingConfig.min_burst_ms,
                        help="width a peak must reach at the shoulder to count "
                             f"(default: {ProcessingConfig.min_burst_ms:g}); a "
                             "single spike already spans one envelope window, "
                             "so values at or below --window-ms only require "
                             "that the peak be resolvable at all")
    parser.add_argument("--rest-fpr-warn", type=float,
                        default=100.0 * ProcessingConfig.rest_fpr_warn,
                        help="percent of a run's own rest windows that may "
                             "fire before its count is flagged unreliable")
    parser.add_argument("--no-trial-figures", action="store_true",
                        help="skip the per-trial figures under trials/"
                             "<recording>/<hand>/")
    parser.add_argument("--out-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="participant folders are created below this path; "
                             "a relative path is taken from the application "
                             "folder, not from the working directory "
                             f"(default: {DEFAULT_OUTPUT_ROOT}/ beside app.py)")
    args = parser.parse_args()

    try:
        band = tuple(float(x.strip()) for x in args.band.split(","))
        if len(band) != 2:
            raise ValueError("--band needs low,high")
        config = ProcessingConfig(
            data_dir=args.folder,
            recordings=[x.strip() for x in args.files.split(",") if x.strip()],
            participant=args.participant,
            target_fs=args.fs or None,
            data_stream=args.data_stream,
            marker_stream=args.marker_stream,
            auto_marker_shift=not args.no_auto_marker_shift,
            marker_shift_s=args.marker_shift,
            left_channels=parse_pair(args.left_emg),
            right_channels=parse_pair(args.right_emg),
            left_condition=args.left_cond,
            right_condition=args.right_cond,
            rest_condition=args.rest_cond,
            envelope=args.envelope,
            input_unit=args.input_unit,
            notch_hz=args.notch,
            band_low_hz=band[0],
            band_high_hz=band[1],
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            pre_s=args.pre,
            post_s=args.post,
            trial_tail_s=args.trial_tail_ms / 1000.0,
            rest_trim_start_s=args.rest_trim_start,
            rest_trim_end_s=args.rest_trim,
            peak_multiplier=args.peak_multiplier,
            background_multiplier=args.background_multiplier,
            pre_reference_start_s=args.pre_reference_start_ms / 1000.0,
            pre_reference_end_s=args.pre_reference_end_ms / 1000.0,
            secondary_pre_multiplier=args.secondary_pre_multiplier,
            secondary_width_multiplier=args.secondary_width_multiplier,
            min_burst_ms=args.min_burst_ms,
            rest_fpr_warn=args.rest_fpr_warn / 100.0,
            trial_figures=not args.no_trial_figures,
            output_root=args.out_root,
        )
        result = process_batch(config, print)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nCompleted {len(result.recordings)} recordings.")
    print(f"Figures: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
