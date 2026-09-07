"""Lightweight smoke tests for the pure-numeric modules.

Run:  python -m pytest tests -q      (or just: python tests/test_smoke.py)

These check shapes, invariants and MATLAB-compat helpers — they do not require
the real .h5 data.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klh import csp, superlet, io_h5, topo
from klh.assets import load_ced
from klh.pipeline import KLHPipeline, ExtractConfig
from klh.robustcov import olivehawkins
from klh.mathutil import matlab_prctile, mad1, nearest_index


def test_matlab_prctile_endpoints():
    x = np.arange(1, 101, dtype=float)
    assert abs(matlab_prctile(x, 50) - 50.5) < 1e-9
    assert matlab_prctile(x, 0) == 1.0
    assert matlab_prctile(x, 100) == 100.0


def test_mad1_and_nearest():
    assert abs(mad1(np.array([1, 2, 3, 4, 5.0])) - 1.0) < 1e-12
    assert nearest_index(np.linspace(0, 1, 11), 0.53) == 5


def test_csp_shapes_and_eig_range():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((500, 10)); S1 = np.cov(A, rowvar=False)
    B = rng.standard_normal((500, 10)); S2 = np.cov(B, rowvar=False)
    pinv, pfwd, gi, gf, ev = csp.calc_csp_cov(S1, S2)
    assert pinv.shape == (10, 10)
    # eig(R1, R1+R2) eigenvalues live in [0, 1]
    assert ev.min() > -1e-9 and ev.max() < 1 + 1e-9
    assert np.all(np.diff(ev) >= -1e-9)  # ascending


def test_robustcov_matches_classical_on_clean_gaussian():
    rng = np.random.default_rng(2)
    true = np.array([[3.0, 1.0], [1.0, 2.0]])
    L = np.linalg.cholesky(true)
    X = rng.standard_normal((4000, 2)) @ L.T
    C, T = olivehawkins(X)
    # On clean data the robust estimate should be close to the truth.
    assert np.allclose(C, true, atol=0.5)
    assert np.allclose(T, 0, atol=0.2)


def test_robustcov_resists_outliers():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((2000, 3))
    Xc = X.copy()
    Xc[:200] += 50.0  # 10% gross outliers
    C_clean, _ = olivehawkins(X)
    C_contam, _ = olivehawkins(Xc)
    # robust estimate barely moves despite the outliers
    assert np.linalg.norm(C_clean - C_contam) < 1.0


def test_superlet_positive_power():
    fs = 1000.0
    t = np.arange(0, 2, 1 / fs)
    sig = np.sin(2 * np.pi * 10 * t)
    wt = superlet.aslt(sig, fs, np.arange(5, 20, 1.0), 5, (1, 5), 0)
    assert wt.shape == (15, t.size)
    assert np.all(wt >= 0)
    # peak power should sit near 10 Hz
    freqs = np.arange(5, 20, 1.0)
    peak_f = freqs[np.argmax(wt.mean(axis=1))]
    assert abs(peak_f - 10) <= 1.0


def _fake_xdf(marker_labels, marker_times, n_samples=4, t0=100.0, fs=500.0):
    """A minimal two-stream XDF in one shared Unix-epoch clock."""
    data_stream = {
        "info": {"name": ["NVX136_Data"], "nominal_srate": [str(fs)],
                 "channel_count": ["2"]},
        "time_series": np.zeros((n_samples, 2)),
        "time_stamps": t0 + np.arange(n_samples) / fs,
    }
    marker_stream = {
        "info": {"name": ["PsychoPyMarkers"], "nominal_srate": ["0"],
                 "channel_count": ["1"]},
        "time_series": [[label] for label in marker_labels],
        "time_stamps": t0 + np.asarray(marker_times, dtype=float),
    }
    return data_stream, marker_stream


def test_marker_events_use_the_data_clock_and_next_marker_duration():
    _, marker = _fake_xdf(
        ["video_onset;file=x.mp4", "right_microrepeat", "right_microrepeat",
         "rest", "right_task_block"],
        [5.0, 7.0, 11.0, 15.0, 29.0])
    events = io_h5.marker_event_sets(marker, 100.0, 40.0,
                                     {"right_microrepeat", "rest"})
    onsets, durations = events["right_microrepeat"]
    assert onsets.tolist() == [7.0, 11.0]
    assert durations.tolist() == [4.0, 4.0]
    # Rest runs to the block cue, not to the next micro-repeat.
    assert events["rest"][0].tolist() == [15.0]
    assert events["rest"][1].tolist() == [14.0]
    assert io_h5.marker_video_onset(marker, 100.0) == 5.0


def test_marker_events_reject_an_incompatible_clock():
    _, marker = _fake_xdf(["rest"], [50000.0])
    try:
        io_h5.marker_event_sets(marker, 100.0, 40.0, {"rest"})
    except ValueError as exc:
        assert "incompatible clocks" in str(exc)
    else:
        raise AssertionError("a marker far outside the recording must raise")


def test_load_xdf_eeg_returns_events_in_sample_time():
    data_stream, marker_stream = _fake_xdf(
        ["video_onset;file=x.mp4", "right_microrepeat", "rest"],
        [0.002, 0.004, 0.006], n_samples=8)

    def fake_load_xdf(path, **kwargs):
        return [data_stream, marker_stream], {}

    fake_pyxdf = SimpleNamespace(load_xdf=fake_load_xdf)
    with patch.dict(sys.modules, {"pyxdf": fake_pyxdf}):
        eeg, events, video = io_h5.load_xdf_eeg(
            "fake.xdf", {"right_microrepeat", "rest"})
    assert eeg.shape == (8, 2)
    assert abs(video - 0.002) < 1e-12
    assert abs(events["right_microrepeat"][0][0] - 0.004) < 1e-12
    assert abs(events["right_microrepeat"][1][0] - 0.002) < 1e-12


def test_topography_label_subset_preserves_nonleading_montage_order():
    ced = load_ced()
    subset = topo._select_labels(ced, ["T7", "Cz", "TP8"])
    assert subset["labels"] == ["T7", "Cz", "TP8"]
    source = {label: i for i, label in enumerate(ced["labels"])}
    assert np.array_equal(
        subset["theta"], ced["theta"][[source["T7"], source["Cz"], source["TP8"]]])


def test_marker_windows_apply_preparation_then_extra_cut_to_active_only():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs, pipe._La, pipe._Lr = 10.0, 40, 140
    pipe.extract = ExtractConfig(
        mode="markers", cut_first=0.2, marker_shift_s=0.0)
    events = {
        "right_microrepeat": (np.array([1.6]), np.array([4.0])),
        "rest": (np.array([3.6]), np.array([14.0])),
    }
    active, _, rest, _ = pipe._marker_windows(events, 0.0)
    # Active time zero follows 2 s preparation plus the optional 0.2 s trim.
    assert active.tolist() == [38]
    assert rest.tolist() == [36]    # rest is untouched by active exclusions


def test_marker_windows_apply_same_four_second_shift_to_active_and_rest():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs, pipe._La, pipe._Lr = 10.0, 40, 140
    pipe.extract = ExtractConfig(mode="markers")
    events = {
        "right_microrepeat": (np.array([7.0]), np.array([4.0])),
        "rest": (np.array([20.0]), np.array([14.0])),
    }

    active, _, rest, _ = pipe._marker_windows(events, -4.0)

    # Raw 7 s marker - 4 s LSL correction + 2 s preparation = 5 s.
    assert active.tolist() == [50]
    assert rest.tolist() == [160]


def test_marker_windows_left_hand_selection_uses_different_trials():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs, pipe._La, pipe._Lr = 10.0, 40, 140
    pipe.extract = ExtractConfig(
        mode="markers", active_cond="left_microrepeat", marker_shift_s=0.0)
    events = {
        "left_microrepeat": (np.array([5.0, 9.0]), np.array([4.0, 4.0])),
        "right_microrepeat": (np.array([1.0, 13.0]), np.array([4.0, 4.0])),
        "rest": (np.array([20.0]), np.array([14.0])),
    }
    active, _, rest, _ = pipe._marker_windows(events, 0.0)
    assert active.tolist() == [70, 110]
    assert rest.tolist() == [200]


def test_prepare_markers_drops_the_trailing_nan_duration():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs = 10.0
    pipe.extract = ExtractConfig(
        mode="markers", cut_last=0.5, auto_marker_shift=False,
        marker_shift_s=0.0)
    durations = {
        "right_microrepeat": np.array([4.0, 3.0, np.nan]),
        "rest": np.array([14.0, np.nan]),
    }
    with patch.object(io_h5, "load_xdf_event_durations",
                      lambda path, wanted: durations):
        pipe._prepare_markers(["a.xdf"])
    # Shortest event: 3.0 - 2.0 preparation - 0.5 cut + 2.0 recovery.
    assert pipe._motor_duration_s == 0.5
    assert pipe._La == 25
    assert pipe._Lr == 140
    assert pipe.tvec is pipe.tvec_active


def test_prepare_markers_builds_two_second_motor_plus_recovery_core():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs = 10.0
    pipe.freqNeeded = np.array([10.0])
    pipe.extract = ExtractConfig(mode="markers", auto_marker_shift=False)
    durations = {
        "right_microrepeat": np.array([4.0, 4.0]),
        "rest": np.array([14.0]),
    }
    with patch.object(io_h5, "load_xdf_event_durations",
                      lambda path, wanted: durations):
        pipe._prepare_markers(["a.xdf"])

    assert pipe.extract.marker_shift_s == -4.0
    assert pipe._motor_duration_s == 2.0
    assert pipe._La == 40
    assert pipe.tvec_active[[0, -1]].tolist() == [0.0, 3.9]


def test_recovery_cannot_rescue_a_fully_excluded_motor_interval():
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.fs = 10.0
    pipe.freqNeeded = np.array([10.0])
    pipe.extract = ExtractConfig(
        mode="markers", preparation_s=4.0, recovery_s=2.0,
        auto_marker_shift=False)
    durations = {
        "right_microrepeat": np.array([4.0]),
        "rest": np.array([14.0]),
    }
    with patch.object(io_h5, "load_xdf_event_durations",
                      lambda path, wanted: durations):
        with pytest.raises(ValueError, match="leaves no motor-task data"):
            pipe._prepare_markers(["a.xdf"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("all smoke tests passed")
