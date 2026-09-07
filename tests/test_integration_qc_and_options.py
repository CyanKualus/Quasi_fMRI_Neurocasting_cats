"""Integration coverage for QC, optional shrinkage and JSON reporting."""
import numpy as np
import pytest

from klh.pipeline import ACTIVE, PASSIVE, ExtractConfig, KLHPipeline
from run_cli import _jsonable, build_parser, _validate_args


def _epochs(amplitudes, *, fs=100.0, seconds=4.0, trials=5):
    time = np.arange(int(fs * seconds)) / fs
    signal = np.sin(2 * np.pi * 10 * time)[:, None, None]
    values = np.asarray(amplitudes)[None, None, :]
    return np.repeat(signal * values, trials, axis=1)


def test_pipeline_exposes_separate_physiological_qc():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs = 100.0
    pipe.extract = ExtractConfig(
        mode="markers", active_cond="right_microrepeat",
        rest_transition_skip=1.0)
    pipe.channel_labels = ["C3", "C4", "FC3", "FC4", "CP3", "CP4"]
    pipe._La = pipe._Lr = 400
    pipe._pad = 0
    pipe.tvec_active = pipe.tvec_rest = np.arange(400) / pipe.fs
    pipe.broad = [[None, None] for _ in range(3)]
    for condition in range(3):
        pipe.broad[condition][ACTIVE] = _epochs([0.5, 1.2, 0.5, 1.2, 0.5, 1.2])
        pipe.broad[condition][PASSIVE] = _epochs([2, 2, 2, 2, 2, 2])

    result = pipe.compute_physiological_qc()
    assert result.reference == "common-average"
    assert set(result.results) == {"overt", "quasi", "imagery"}
    assert result.results["imagery"]["mu"].channel_db.shape == (6,)
    assert result.results["imagery"]["mu"].active_trials == 5
    assert pipe.physiological_qc is result


def test_pipeline_rejects_invalid_optional_shrinkage_before_loading():
    pipe = KLHPipeline()
    with pytest.raises(ValueError, match="csp_shrinkage"):
        pipe.run_start([], [], [], csp_shrinkage=1.0)


def test_cli_validates_shrinkage_and_accepts_json_report():
    parser = build_parser()
    args = parser.parse_args([
        "--folder", ".", "--real", "a", "--quasi", "b", "--imag", "c",
        "--csp-shrinkage", "0.05", "--report-json", "report.json",
    ])
    args = _validate_args(parser, args)
    assert args.csp_shrinkage == 0.05
    assert args.report_json == "report.json"


def test_json_report_converts_nonfinite_diagnostics_to_null():
    converted = _jsonable({"nan": np.nan, "inf": np.float64(np.inf)})
    assert converted == {"nan": None, "inf": None}
