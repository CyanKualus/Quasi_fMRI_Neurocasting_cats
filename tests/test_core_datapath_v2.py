"""Focused tests for the revised sampling/epoch/provenance data path."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klh import filters, io_h5, superlet
from klh.mathutil import rng_inclusive
from klh.pipeline import (
    ACTIVE,
    PASSIVE,
    REAL,
    ExtractConfig,
    KLHPipeline,
    _empty_trial_counts,
    covariance_pencil_condition,
    detect_eigenvalue_candidates,
)


def test_window_selection_warns_records_and_can_be_strict():
    vec = np.arange(0.0, 2.0, 0.1)
    diagnostics = []
    with pytest.warns(RuntimeWarning, match="would be clamped"):
        selected = rng_inclusive(
            vec, -0.25, 2.5, diagnostics=diagnostics, label="score")
    assert selected.start == 0 and selected.stop == vec.size
    assert diagnostics[0]["clamped_start"]
    assert diagnostics[0]["clamped_end"]
    with pytest.raises(ValueError, match="would be clamped"):
        rng_inclusive(vec, -0.25, 2.5, strict=True)


def test_resampling_and_onset_mapping_are_separate_operations():
    eeg = np.arange(20.0).reshape(10, 2)
    upsampled = io_h5.resample_eeg(eeg, 500.0, 1000.0)
    assert upsampled.shape == (20, 2)
    mapped = io_h5.remap_sample_indices(
        np.array([0, 125, 499]), 500.0, 1000.0)
    assert mapped.tolist() == [0, 250, 998]


def test_h5_gather_detects_native_trigger_and_resamples_eeg_only(monkeypatch):
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.file_type = "h5"
    pipe.extract = ExtractConfig(mode="trigger")
    pipe.ch_num = 2
    pipe.recording_fs = 500.0
    pipe.fs = 1000.0
    pipe.epoind = np.arange(-1, 2)
    pipe.b1 = pipe.b2 = pipe.bpa = np.array([1.0])
    pipe.trial_counts = _empty_trial_counts()
    pipe._recording_details = []
    pipe.video_onset_offsets = {}

    raw = np.zeros((30, 3))
    raw[:, -1] = 2.0  # categorical trigger column, never sent to resample_eeg

    monkeypatch.setattr(io_h5, "load_h5_eeg", lambda path: raw)

    def decode(value):
        assert value.shape == (30, 3)
        return np.zeros(30)

    def detect(value, fs):
        assert value.shape == (30,)
        assert fs == 500.0
        return np.array([5]), np.array([10]), np.array([]), np.array([])

    def resample(value, recording_fs, processing_fs):
        assert value.shape == (30, 2)
        assert recording_fs == 500.0 and processing_fs == 1000.0
        return np.repeat(value, 2, axis=0)

    monkeypatch.setattr(io_h5, "decode_trigger", decode)
    monkeypatch.setattr(io_h5, "trig_detector", detect)
    monkeypatch.setattr(io_h5, "resample_eeg", resample)
    monkeypatch.setattr(filters, "broadband", lambda value, *args: value)
    monkeypatch.setattr(filters, "alpha", lambda value, *args: value)

    broad = [[None, None] for _ in range(3)]
    alpha = [[None, None] for _ in range(3)]
    pipe._gather(["native.h5"], REAL, 1.0, broad, alpha)

    assert broad[REAL][ACTIVE].shape == (3, 1, 2)
    assert broad[REAL][PASSIVE].shape == (3, 1, 2)
    assert pipe.trial_counts["real"]["active"]["retained"] == 1
    assert pipe._recording_details[0]["event_path"] == "native-trigger-detection"
    assert pipe._recording_details[0]["resampled"] is True


def test_marker_windows_include_real_padding_and_count_durationless_events():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs, pipe._La, pipe._Lr, pipe._pad = 10.0, 40, 100, 15
    pipe.extract = ExtractConfig(mode="markers", marker_shift_s=0.0)
    events = {
        "right_microrepeat": (
            np.array([2.0, 8.0]), np.array([4.0, np.nan])),
        "rest": (np.array([12.0]), np.array([10.0])),
    }
    active, ai, rest, ri, counts = pipe._marker_windows(
        events, 0.0, return_counts=True)
    assert active.tolist() == [40]
    assert rest.tolist() == [120]
    assert (ai[0], ai[-1], ai.size) == (-15, 54, 70)
    assert (ri[0], ri[-1], ri.size) == (-15, 114, 130)
    assert counts[ACTIVE] == {"requested": 2, "eligible": 1}


@pytest.mark.parametrize("kind,core_len", [(ACTIVE, 4), (PASSIVE, 5)])
def test_padded_tf_matches_independent_trial_cores(monkeypatch, kind, core_len):
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs = 10.0
    pipe.freqNeeded = np.array([1.0])
    pipe.ch_num = 1
    pipe.projInverse = np.ones((1, 1))
    pipe.extract = ExtractConfig(mode="markers", marker_shift_s=0.0)
    pipe._La, pipe._Lr, pipe._pad = 4, 5, 2
    padded_len = core_len + 4
    epochs = np.empty((padded_len, 2, 1))
    epochs[:, 0, 0] = np.arange(padded_len) + 10.0
    epochs[:, 1, 0] = -(np.arange(padded_len) + 100.0)

    def moving_transform(sig, *args, **kwargs):
        return np.convolve(np.asarray(sig), np.ones(5), mode="same")[None, :]

    monkeypatch.setattr(superlet, "aslt", moving_transform)
    actual = pipe._tf_trials(epochs, 0, kind)
    expected = np.stack([
        moving_transform(epochs[:, trial, 0])[0, 2:2 + core_len]
        for trial in range(2)
    ], axis=1)[:, :, None]
    assert np.array_equal(actual, expected)


def test_marker_rest_baseline_supports_condition_specific_pooling_and_skip(monkeypatch):
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs = 1.0
    pipe.freqNeeded = np.array([1.0])
    pipe.ch_num = 1
    pipe.projInverse = np.ones((1, 1))
    pipe.extract = ExtractConfig(
        mode="markers", marker_baseline="condition-specific",
        rest_transition_skip=1.0)
    pipe._La, pipe._Lr, pipe._pad = 2, 2, 1
    pipe.broad = [[None, None] for _ in range(3)]
    cores = ([100.0, 1.0], [4.0, 4.0], [9.0, 9.0])
    for cond, core in enumerate(cores):
        pipe.broad[cond][PASSIVE] = np.array(
            [0.0, *core, 0.0], dtype=float).reshape(4, 1, 1)

    monkeypatch.setattr(
        superlet, "aslt",
        lambda sig, *args, **kwargs: np.asarray(sig, dtype=float)[None, :])
    assert pipe._rest_baseline(0, condition=REAL).item() == 1.0
    assert pipe._rest_baseline(0, condition=None).item() == 4.0


def test_candidate_detection_reports_no_low_candidates_when_no_gap():
    low, high, n_erd, diagnostic = detect_eigenvalue_candidates(
        np.linspace(0.1, 0.9, 24), 24, scale="raw")
    assert n_erd == 0
    assert low.size == 0
    assert high.size == 0
    assert diagnostic["low_limit"] == 4
    assert diagnostic["high_limit"] == 11


def test_candidate_detection_supports_symmetric_log_odds_scale():
    evals = np.array([0.0, 0.01, 0.02, 0.5, 0.98, 0.99, 1.0])
    low, high, n_erd, diagnostic = detect_eigenvalue_candidates(
        evals, evals.size, scale="logit")
    assert np.isfinite(diagnostic["threshold"])
    assert low.size >= 1
    assert isinstance(n_erd, int)


def test_covariance_condition_diagnostic_detects_a_singular_pencil():
    singular = np.diag([1.0, 0.0])
    assert np.isinf(covariance_pencil_condition(singular, singular))


def test_fixed_filter_rate_and_h5_reference_channel_cap_are_enforced():
    pipe = KLHPipeline()
    with pytest.raises(ValueError, match="require processing_fs=1000"):
        pipe.run_start([], [], [], processing_fs=500.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        pipe.run_start([], [], [], ch_num=65)


def test_run_start_rejects_duplicate_input_within_one_condition(tmp_path):
    path = tmp_path / "run.h5"
    spelling_variant = os.path.join(str(tmp_path), ".", "run.h5")
    pipe = KLHPipeline()

    with pytest.raises(
            ValueError, match=r"duplicate input recording.*more than once.*real"):
        pipe.run_start([path, spelling_variant], [], [])


def test_run_start_rejects_duplicate_input_across_conditions(tmp_path):
    path = tmp_path / "run.h5"
    spelling_variant = os.path.join(
        str(tmp_path), "unused-directory", "..", "run.h5")
    pipe = KLHPipeline()

    with pytest.raises(
            ValueError, match=r"duplicate input recording.*real.*quasi"):
        pipe.run_start([path], [spelling_variant], [])


def test_data_path_identifier_changes_with_marker_baseline():
    pipe = KLHPipeline()
    pipe.file_type = "xdf"
    pipe._input_files = {"real": [], "quasi": [], "imag": []}
    pipe._recording_details = []
    pipe._window_diagnostics = []
    pipe._pad = 1500
    pipe._required_tf_pad = 1000
    pipe.candidate_diagnostic = {}
    pipe.extract = ExtractConfig(mode="markers", marker_baseline="pooled")
    pooled = pipe._build_provenance()["data_path_version"]
    pipe.extract.marker_baseline = "condition-specific"
    per_condition = pipe._build_provenance()["data_path_version"]
    assert pooled != per_condition


def test_data_path_provenance_records_and_hashes_marker_shift():
    pipe = KLHPipeline()
    pipe.file_type = "xdf"
    pipe._input_files = {"real": [], "quasi": [], "imag": []}
    pipe._recording_details = []
    pipe._window_diagnostics = []
    pipe._pad = 1500
    pipe._required_tf_pad = 1000
    pipe.candidate_diagnostic = {}
    pipe.extract = ExtractConfig(
        mode="markers", auto_marker_shift=False, marker_shift_s=-4.0)
    shifted = pipe._build_provenance()
    pipe.extract.marker_shift_s = 0.0
    raw = pipe._build_provenance()

    assert shifted["marker_shift_mode"] == "entered-by-hand"
    assert shifted["marker_time_shift_s"] == -4.0
    assert shifted["data_path_version"] != raw["data_path_version"]


def test_data_path_hashes_detected_marker_shifts_per_recording():
    """Detected shifts are analysis choices too, even though nobody typed them.

    Two runs over recordings that measured different lags must not share a data
    path identifier, and the identifier must not depend on the untyped value
    left in the manual box.
    """
    pipe = KLHPipeline()
    pipe.file_type = "xdf"
    pipe._input_files = {"real": [], "quasi": [], "imag": []}
    pipe._recording_details = []
    pipe._window_diagnostics = []
    pipe._pad = 1500
    pipe._required_tf_pad = 1000
    pipe.candidate_diagnostic = {}
    pipe.extract = ExtractConfig(mode="markers", auto_marker_shift=True)

    pipe._marker_shifts = {"/data/om1.xdf": -5.024}
    first = pipe._build_provenance()
    pipe._marker_shifts = {"/data/om1.xdf": -5.437}
    second = pipe._build_provenance()
    pipe._marker_shifts = {"/data/om1.xdf": -5.024}
    pipe.extract.marker_shift_s = -1.0
    repeat = pipe._build_provenance()

    assert first["marker_shift_mode"] == "detected-per-recording"
    assert first["marker_time_shift_s"] is None
    assert first["data_path_version"] != second["data_path_version"]
    assert first["data_path_version"] == repeat["data_path_version"]


def test_data_path_provenance_records_and_hashes_motor_timing():
    pipe = KLHPipeline()
    pipe.file_type = "xdf"
    pipe._input_files = {"real": [], "quasi": [], "imag": []}
    pipe._recording_details = []
    pipe._window_diagnostics = []
    pipe._pad = 1500
    pipe._required_tf_pad = 1000
    pipe.candidate_diagnostic = {}
    pipe.extract = ExtractConfig(
        mode="markers", preparation_s=2.0, recovery_s=2.0)
    requested = pipe._build_provenance()
    pipe.extract.recovery_s = 1.0
    shorter = pipe._build_provenance()

    assert requested["marker_preparation_s"] == 2.0
    assert requested["marker_recovery_s"] == 2.0
    assert requested["data_path_version"] != shorter["data_path_version"]
