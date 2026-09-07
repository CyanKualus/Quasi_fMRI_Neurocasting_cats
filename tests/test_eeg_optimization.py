"""Regression checks for scheduling, cache validity and unchanged EEG results."""
import copy
import threading

import numpy as np
import pytest
from threadpoolctl import threadpool_info

from klh import execution, pipeline, qc
from klh.pipeline import ACTIVE, PASSIVE, REAL, QUASI, IMAG, _bootstrap_erd_estimates
from test_csp_crossvalidation import _pipe, _ft_ready


@pytest.fixture(autouse=True)
def restore_workers(monkeypatch):
    previous = execution._workers
    # These fixtures use small arrays. Make parallel coverage independent of
    # unrelated desktop applications' memory use during the test run.
    monkeypatch.setattr(execution, "available_memory", lambda: 1 << 30)
    yield
    execution.configure_workers(previous)


def _reference_folds(pipe, idx, n_folds):
    """The original uncached, sequential fit and selected-filter scaling."""
    active = pipe._core_epochs(pipe.alpha[REAL][ACTIVE], ACTIVE)[pipe._csp_ind1]
    rests = [pipe._core_epochs(pipe.alpha[c][PASSIVE], PASSIVE)[pipe._csp_ind0]
             for c in (REAL, QUASI, IMAG)]
    af = pipe._fold_assignment(active.shape[1], n_folds)
    rf = pipe._fold_assignment(rests[0].shape[1], n_folds)
    other = np.concatenate(rests[1:], axis=1)
    result = []
    for k in range(n_folds):
        xa = active[:, af != k].reshape(-1, pipe.ch_num, order="F")
        xr = np.concatenate([rests[0][:, rf != k], other], axis=1)
        xr = xr.reshape(-1, pipe.ch_num, order="F")
        ca, _ = pipeline.olivehawkins(xa)
        cr, _ = pipeline.olivehawkins(xr)
        inverse, forward, _, _, _ = pipeline.csp.calc_csp_cov(
            ca, cr, shrinkage=pipe.csp_shrinkage, include_ged=False)
        component, similarity, margin = pipe._match_component(
            forward, pipe.projForward[:, idx])
        weights = inverse[:, component].astype(float)
        result.append((weights / np.std(xr @ weights), component + 1,
                       similarity, margin))
    return result


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("shrinkage", [0.0, 0.15])
def test_cached_cv_matches_original_for_multiple_selected_components(workers, shrinkage):
    pipe = _pipe()
    pipe.csp_shrinkage = shrinkage
    execution.configure_workers(workers)
    for idx in (0, 1):
        expected = _reference_folds(pipe, idx, 3)
        actual = pipe._cv_overt_filters(idx, 3)
        for fold, (weights, component, similarity, margin) in zip(actual, expected):
            np.testing.assert_allclose(fold["weights"], weights, rtol=1e-10, atol=1e-12)
            assert fold["matched_component"] == component
            assert fold["pattern_similarity_to_selected"] == pytest.approx(similarity)
            assert fold["match_margin"] == pytest.approx(margin)


def test_cv_cache_reuses_fits_but_invalidates_data_settings_and_new_runs(monkeypatch):
    pipe = _pipe()
    original = pipeline.olivehawkins
    calls = []

    def count(x):
        calls.append(x.shape)
        return original(x)

    monkeypatch.setattr(pipeline, "olivehawkins", count)
    pipe._cv_overt_filters(0, 3)
    assert len(calls) == 6
    pipe._cv_overt_filters(1, 3)
    assert len(calls) == 6
    pipe.alpha[REAL][ACTIVE][0, 0, 0] += 0.5  # in-place changes count
    pipe._cv_overt_filters(0, 3)
    assert len(calls) == 12
    pipe.csp_shrinkage = 0.1
    pipe._cv_overt_filters(0, 3)
    assert len(calls) == 18
    pipe._cv_overt_filters(0, 2)
    assert len(calls) == 22
    pipe._csp_ind0 = slice(1, 40)
    pipe._cv_overt_filters(0, 2)
    assert len(calls) == 26
    pipe._qc_cache = ("old", object())
    with pytest.raises(ValueError):
        pipe.run_start([], [], [], csp_shrinkage=1.0)
    assert pipe._cv_fit_cache is None and pipe._qc_cache is None


def test_failed_parallel_fit_does_not_cache_partial_results(monkeypatch):
    pipe = _pipe()
    execution.configure_workers(2)
    def fail(x):
        raise RuntimeError("covariance failed")
    monkeypatch.setattr(pipeline, "olivehawkins", fail)
    with pytest.raises(RuntimeError, match="covariance failed"):
        pipe._cv_overt_filters(0, 3)
    assert getattr(pipe, "_cv_fit_cache", None) is None


@pytest.mark.parametrize("workers", [0, 2, 4])
def test_bootstrap_matches_original_draws_reductions_and_rng_state(workers):
    region = np.exp(np.random.default_rng(4).normal(size=(151, 9, 5)))
    baseline = np.linspace(0.7, 1.3, 5)
    rng = np.random.default_rng(42)
    expected = []
    for _ in range(37):  # includes an incomplete scheduling batch
        take = rng.integers(0, 9, size=9)
        power = np.median(region[:, take, :], axis=1)
        expected.append(-np.median(10 * np.log10(power / baseline[None, :])))
    execution.configure_workers(workers)
    actual_rng = np.random.default_rng(42)
    actual = _bootstrap_erd_estimates(region, baseline, actual_rng, 37)
    assert np.array_equal(actual, expected)
    assert actual_rng.bit_generator.state == rng.bit_generator.state


def test_qc_reuses_trial_psds_only_for_actual_recording_views(monkeypatch):
    rng = np.random.default_rng(11)
    active, rest = [rng.normal(size=(400, 8, 4)) for _ in range(2)]
    options = dict(fs=100.0, tvec_active=np.arange(400) / 100,
                   tvec_rest=np.arange(400) / 100,
                   channel_labels=["C3", "C4", "CP3", "CP4"],
                   active_condition="right_microrepeat", rest_window=(1, 3))
    recordings = {"quasi": {"a": (active[:, :3], rest[:, :3]),
                             "b": (active[:, 3:], rest[:, 3:])}}
    # Independent copies force the original per-recording spectral calculation.
    expected = qc.compute_sensorimotor_qc(
        {"quasi": (active, rest)}, recording_epochs=copy.deepcopy(recordings), **options)
    original = qc._band_trial_power
    calls = []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(qc, "_band_trial_power", count)
    actual = qc.compute_sensorimotor_qc(
        {"quasi": (active, rest)}, recording_epochs=recordings, **options)
    assert len(calls) == 2  # active/rest once, not again for each recording
    from run_cli import _jsonable
    assert _jsonable(actual) == _jsonable(expected)
    # Stepped/reversed trial views must not be mistaken for contiguous slices.
    assert qc._trial_slice(active, active[:, ::2]) is None
    assert qc._trial_slice(active, active[:, ::-1]) is None


def test_pipeline_qc_cache_is_independent_of_component_and_detects_changes(monkeypatch):
    pipe = _ft_ready(_pipe())
    pipe._La = pipe._Lr = 600
    pipe.tvec_active = pipe.tvec_rest = np.arange(600) / pipe.fs - 2
    rng = np.random.default_rng(22)
    pipe.broad = [[rng.normal(size=(600, 8, 4)) for _ in range(2)]
                  for _ in range(3)]
    calls = []
    original = qc.compute_sensorimotor_qc
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(qc, "compute_sensorimotor_qc", count)
    first = pipe.compute_physiological_qc()
    pipe.projInverse *= 2
    second = pipe.compute_physiological_qc()
    assert len(calls) == 1
    assert first is not second
    first.results["overt"]["mu"].channel_db[:] = 999
    third = pipe.compute_physiological_qc()
    assert not np.any(third.results["overt"]["mu"].channel_db == 999)
    pipe.broad[REAL][ACTIVE][:] *= 1.5
    pipe.compute_physiological_qc()
    assert len(calls) == 2
    pipe.extract.active_cond = "left_microrepeat"
    pipe.compute_physiological_qc()
    assert len(calls) == 3


def test_ordered_workers_overlap_restore_blas_and_do_not_nest():
    execution.configure_workers(2)
    barrier = threading.Barrier(2)
    before = [(row["filepath"], row["num_threads"]) for row in threadpool_info()]
    def work(value):
        barrier.wait(timeout=10)
        ident = threading.get_ident()
        nested = execution.ordered_map(lambda _: threading.get_ident(), (0, 1))
        assert nested == [ident, ident]
        assert all(row["num_threads"] == 1 for row in threadpool_info()
                   if row["user_api"] == "blas")
        return value
    assert execution.ordered_map(work, [2, 1], blas=True) == [2, 1]
    assert [(row["filepath"], row["num_threads"]) for row in threadpool_info()] == before
    with pytest.raises(RuntimeError):
        execution.ordered_map(lambda _: (_ for _ in ()).throw(RuntimeError()),
                              [0, 1], blas=True)
    assert [(row["filepath"], row["num_threads"]) for row in threadpool_info()] == before


def test_low_memory_falls_back_to_serial_without_changing_results(monkeypatch):
    execution.configure_workers(4)
    monkeypatch.setattr(execution, "available_memory", lambda: 100 << 20)
    main_thread = threading.get_ident()
    result = execution.ordered_map(lambda x: (x * 2, threading.get_ident()),
                                   [1, 2, 3], working_bytes=10 << 20)
    assert result == [(2, main_thread), (4, main_thread), (6, main_thread)]


def test_cli_accepts_serial_or_parallel_analysis_and_rejects_negative_workers():
    from run_cli import build_parser, _validate_args
    parser = build_parser()
    base = ["--folder", ".", "--real", "a", "--quasi", "b", "--imag", "c"]
    for workers in (0, 1, 2):
        args = _validate_args(parser, parser.parse_args(
            base + ["--analysis-workers", str(workers)]))
        assert args.analysis_workers == workers
    with pytest.raises(SystemExit):
        _validate_args(parser, parser.parse_args(base + ["--analysis-workers", "-1"]))


def test_filter_scheduling_and_single_concatenation_keep_trials_and_provenance(monkeypatch):
    from klh import io_h5, filters
    from klh.pipeline import KLHPipeline, ExtractConfig, _empty_trial_counts
    raw = np.column_stack([np.arange(30), np.arange(30) * 2, np.zeros(30)])
    monkeypatch.setattr(io_h5, "load_h5_eeg",
                        lambda path: raw + (100 if path == "b.h5" else 0))
    monkeypatch.setattr(io_h5, "decode_trigger", lambda x: x[:, -1])
    monkeypatch.setattr(io_h5, "trig_detector", lambda x, fs:
                        (np.array([5]), np.array([0, 10]), None, None))
    monkeypatch.setattr(filters, "broadband", lambda x, *args: x * 2)
    monkeypatch.setattr(filters, "alpha", lambda x, *args: x * 3)

    def gather(workers):
        execution.configure_workers(workers)
        pipe = KLHPipeline.__new__(KLHPipeline)
        pipe.file_type = "h5"
        pipe.extract = ExtractConfig(mode="trigger")
        pipe.ch_num = 2
        pipe.recording_fs = pipe.fs = 1000.0
        pipe.epoind = np.arange(-1, 2)
        pipe.b1 = pipe.b2 = pipe.bpa = np.ones(1)
        pipe.trial_counts = _empty_trial_counts()
        pipe._recording_details = []
        pipe.video_onset_offsets = {}
        broad, alpha = [[[None, None] for _ in range(3)] for _ in range(2)]
        pipe._gather(["a.h5", "b.h5"], REAL, 1.0, broad, alpha)
        return pipe, broad, alpha

    serial, sb, sa = gather(0)
    parallel, pb, pa = gather(2)
    for kind in (ACTIVE, PASSIVE):
        assert np.array_equal(sb[REAL][kind], pb[REAL][kind])
        assert np.array_equal(sa[REAL][kind], pa[REAL][kind])
        assert pb[REAL][kind].shape == (3, 2, 2)
    assert pb[REAL][ACTIVE][1, :, 0].tolist() == [20, 220]
    assert parallel._recording_details == serial._recording_details
    assert parallel.trial_counts == serial.trial_counts
    assert parallel.trial_counts["real"]["active"]["retained"] == 2
    assert [row["active"] for row in parallel._file_epoch_slices] == [[0, 1], [1, 2]]
    # Also preserve the helper's ability to extend an existing store.
    parallel._gather(["a.h5"], REAL, 1.0, pb, pa)
    assert pb[REAL][ACTIVE][1, :, 0].tolist() == [20, 220, 20]
    assert parallel._file_epoch_slices[-1]["active"] == [2, 3]


@pytest.mark.parametrize("workers", [0, 2])
def test_cv_scoring_results_remain_equivalent_after_caching(workers):
    from run_cli import _jsonable
    pipe = _ft_ready(_pipe())
    execution.configure_workers(workers)
    first = pipe.compute_ft(1, include_qc=False, csp_cv_folds=3)
    # Switch away, then return to the original component on the cached fits.
    pipe.compute_ft(2, include_qc=False, csp_cv_folds=3)
    repeated = pipe.compute_ft(1, include_qc=False, csp_cv_folds=3)
    a, b = _jsonable(first), _jsonable(repeated)
    # The existing provenance log records every window request across calls.
    # Compare the scientific result and definitions, excluding that history.
    a["provenance"].pop("window_diagnostics")
    b["provenance"].pop("window_diagnostics")
    assert a == b
