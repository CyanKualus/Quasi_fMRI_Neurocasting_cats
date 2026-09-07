"""Tests for the cross-validated overt projection.

The CSP filter is fitted on overt movement, so projecting the same overt trials
through it makes criterion 3 an in-sample statistic while quasi and imagery stay
out-of-sample.  These tests pin the properties that make the held-out estimate
trustworthy: every trial is scored exactly once by a filter that never trained
on it, folds are matched back to the operator's component, and fold filters are
placed on a common scale before their outputs are pooled.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klh.pipeline import (
    ACTIVE,
    CV_MATCH_MARGIN_WARNING,
    DEFAULT_CSP_CV_FOLDS,
    IMAG,
    PASSIVE,
    QUASI,
    REAL,
    ExtractConfig,
    KLHPipeline,
)


CH = 4
N_ACTIVE = 12
N_REST = 9
N_TIME = 40


def _pipe(n_folds_data_seed=0, ch_num=CH):
    """A minimal pipeline carrying only what the CV helpers read."""
    rng = np.random.default_rng(n_folds_data_seed)
    pipe = KLHPipeline.__new__(KLHPipeline)
    pipe.ch_num = ch_num
    pipe.fs = 100.0
    pipe.extract = ExtractConfig(mode="trigger")
    pipe.csp_shrinkage = 0.0
    pipe._pad = 0
    pipe._La = N_TIME
    pipe._Lr = N_TIME
    pipe._csp_ind0 = slice(0, N_TIME)
    pipe._csp_ind1 = slice(0, N_TIME)

    def block(n_trials, gain):
        base = rng.standard_normal((N_TIME, n_trials, ch_num))
        # Give channel 0 condition-dependent power so CSP has a real axis to
        # find and the matched component is stable across folds.
        base[:, :, 0] *= gain
        return base

    pipe.alpha = [[None, None] for _ in range(3)]
    pipe.broad = [[None, None] for _ in range(3)]
    pipe.alpha[REAL][ACTIVE] = block(N_ACTIVE, 3.0)
    pipe.alpha[REAL][PASSIVE] = block(N_REST, 1.0)
    pipe.alpha[QUASI][PASSIVE] = block(N_REST, 1.0)
    pipe.alpha[IMAG][PASSIVE] = block(N_REST, 1.0)
    for cond in (REAL, QUASI, IMAG):
        pipe.broad[cond][ACTIVE] = block(N_ACTIVE, 2.0)
        pipe.broad[cond][PASSIVE] = block(N_REST, 1.0)

    # A forward pattern dominated by the same channel the CSP will latch onto.
    pipe.projForward = np.eye(ch_num)
    pipe.projInverse = np.eye(ch_num)
    return pipe


# --------------------------------------------------------------- fold layout

def test_fold_assignment_interleaves_so_folds_span_all_recordings():
    pipe = _pipe()
    labels = pipe._fold_assignment(10, 3)
    assert labels.tolist() == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]


def test_fold_assignment_covers_every_trial_exactly_once():
    pipe = _pipe()
    labels = pipe._fold_assignment(N_ACTIVE, DEFAULT_CSP_CV_FOLDS)
    seen = np.concatenate([
        np.flatnonzero(labels == k) for k in range(DEFAULT_CSP_CV_FOLDS)])
    assert sorted(seen.tolist()) == list(range(N_ACTIVE))


# ---------------------------------------------------- component re-matching

def test_match_component_is_invariant_to_fold_sign_and_scale():
    pipe = _pipe()
    reference = np.array([1.0, -2.0, 0.5, 3.0])
    fold_forward = np.column_stack([
        np.array([1.0, 1.0, -1.0, -1.0]),     # unrelated topography
        -4.0 * reference,                      # the same pattern, flipped/scaled
        np.array([2.0, 1.0, 1.0, 1.0]),
    ])
    index, similarity, margin = pipe._match_component(fold_forward, reference)
    assert index == 1
    assert similarity == pytest.approx(1.0)
    assert margin > CV_MATCH_MARGIN_WARNING


def test_match_component_reports_a_small_margin_when_two_columns_tie():
    pipe = _pipe()
    reference = np.array([1.0, -1.0, 2.0, 0.0])
    fold_forward = np.column_stack([reference, reference * 0.999, np.zeros(4)])
    _, _, margin = pipe._match_component(fold_forward, reference)
    assert margin < CV_MATCH_MARGIN_WARNING


def test_match_component_rejects_a_constant_reference_pattern():
    pipe = _pipe()
    with pytest.raises(ValueError, match="constant"):
        pipe._match_component(np.eye(4), np.ones(4))


# ------------------------------------------------------------- fold fitting

def test_cv_folds_hold_out_every_overt_trial_exactly_once():
    pipe = _pipe()
    folds = pipe._cv_overt_filters(0, 3)
    assert len(folds) == 3
    active = np.concatenate([f["active_test"] for f in folds])
    rest = np.concatenate([f["rest_test"] for f in folds])
    assert sorted(active.tolist()) == list(range(N_ACTIVE))
    assert sorted(rest.tolist()) == list(range(N_REST))
    for fold in folds:
        assert fold["n_train_active"] == N_ACTIVE - fold["active_test"].size
        # Training rest keeps the other two conditions in full.
        assert fold["n_train_rest"] == (
            N_REST - fold["rest_test"].size + 2 * N_REST)


def test_cv_fold_filters_share_a_common_training_rest_scale():
    """Held-out trials from different folds are pooled, so units must match."""
    pipe = _pipe()
    folds = pipe._cv_overt_filters(0, 3)
    rest_by_cond = [
        pipe._core_epochs(pipe.alpha[cond][PASSIVE], PASSIVE)[pipe._csp_ind0]
        for cond in (REAL, QUASI, IMAG)
    ]
    rest_folds = pipe._fold_assignment(N_REST, 3)
    for fold in folds:
        train_rest = np.concatenate(
            [rest_by_cond[REAL][:, rest_folds != fold["fold"], :],
             rest_by_cond[QUASI], rest_by_cond[IMAG]], axis=1)
        projected = train_rest.reshape(-1, CH, order="F") @ fold["weights"]
        assert np.std(projected) == pytest.approx(1.0)


def test_cv_folds_reject_more_folds_than_available_trials():
    pipe = _pipe()
    with pytest.raises(ValueError, match="exceeds"):
        pipe._cv_overt_filters(0, N_REST + 1)


# ------------------------------------------------- held-out TF reassembly

def test_cv_overt_tf_returns_each_trial_in_its_original_position():
    """Every trial is transformed once, by the fold filter that skipped it."""
    pipe = _pipe()
    n_freq = 5
    calls = []

    def fake_tf(epochs, idx, kind, weights=None):
        calls.append((kind, epochs.shape[1], float(weights[0])))
        # Encode the fold's identity so misplacement is detectable.
        block = np.empty((N_TIME, epochs.shape[1], n_freq))
        block[:] = float(weights[0])
        return block

    pipe._tf_trials = fake_tf
    folds = [
        {"fold": 0, "weights": np.array([10.0, 0, 0, 0]),
         "active_test": np.array([0, 2]), "rest_test": np.array([0])},
        {"fold": 1, "weights": np.array([20.0, 0, 0, 0]),
         "active_test": np.array([1]), "rest_test": np.array([1, 2])},
    ]
    pipe.broad[REAL][ACTIVE] = np.zeros((N_TIME, 3, CH))
    pipe.broad[REAL][PASSIVE] = np.zeros((N_TIME, 3, CH))

    active, rest = pipe._cv_overt_tf(0, folds, with_rest=True)
    assert active.shape == (N_TIME, 3, n_freq)
    assert rest.shape == (N_TIME, 3, n_freq)
    # Trials 0 and 2 came from fold 0; trial 1 from fold 1.
    assert active[0, :, 0].tolist() == [10.0, 20.0, 10.0]
    assert rest[0, :, 0].tolist() == [10.0, 20.0, 20.0]
    # No trial is transformed twice.
    assert sum(count for kind, count, _ in calls if kind == ACTIVE) == 3
    assert sum(count for kind, count, _ in calls if kind == PASSIVE) == 3


def test_cv_overt_tf_skips_rest_when_the_baseline_does_not_need_it():
    """Trigger mode baselines the active epochs, so rest must not be transformed."""
    pipe = _pipe()
    calls = []

    def fake_tf(epochs, idx, kind, weights=None):
        calls.append(kind)
        return np.zeros((N_TIME, epochs.shape[1], 4))

    pipe._tf_trials = fake_tf
    folds = [{"fold": 0, "weights": np.array([1.0, 0, 0, 0]),
              "active_test": np.array([0, 1]), "rest_test": np.array([0, 1])}]
    pipe.broad[REAL][ACTIVE] = np.zeros((N_TIME, 2, CH))
    pipe.broad[REAL][PASSIVE] = np.zeros((N_TIME, 2, CH))

    active, rest = pipe._cv_overt_tf(0, folds, with_rest=False)
    assert active is not None
    assert rest is None
    assert PASSIVE not in calls


# ------------------------------------------------------------- reporting

def test_cv_summary_states_the_bias_when_cross_validation_is_off():
    pipe = _pipe()
    summary = pipe._cv_summary([], 1, False)
    assert summary["enabled"] is False
    assert "in-sample" in summary["reason"]
    assert "criterion 4" in summary["reason"]


def test_cv_summary_warns_when_a_fold_match_is_ambiguous():
    pipe = _pipe()
    folds = [
        {"fold": 0, "matched_component": 1, "match_margin": 0.5,
         "pattern_similarity_to_selected": 0.9,
         "active_test": np.array([0]), "rest_test": np.array([0]),
         "weights": np.zeros(CH)},
        {"fold": 1, "matched_component": 3, "match_margin": 0.01,
         "pattern_similarity_to_selected": 0.8,
         "active_test": np.array([1]), "rest_test": np.array([1]),
         "weights": np.zeros(CH)},
    ]
    summary = pipe._cv_summary(folds, 2, True)
    assert summary["enabled"] is True
    assert summary["matched_components"] == [1, 3]
    assert summary["min_match_margin"] == pytest.approx(0.01)
    assert any("margin" in w for w in summary["warnings"])
    # Weights and index arrays never reach the serialisable report.
    assert all("weights" not in row for row in summary["per_fold"])
    assert summary["per_fold"][0]["n_active_test"] == 1


def test_cv_summary_warns_when_the_component_is_unstable_across_folds():
    pipe = _pipe()
    folds = [
        {"fold": 0, "matched_component": 1, "match_margin": 0.4,
         "pattern_similarity_to_selected": 0.31,
         "active_test": np.array([0]), "rest_test": np.array([0]),
         "weights": np.zeros(CH)},
    ]
    summary = pipe._cv_summary(folds, 1, True)
    assert any("not stable" in w for w in summary["warnings"])


# -------------------------------------------------------- compute_ft wiring

def _ft_ready(pipe):
    """Fill in the state compute_ft needs beyond the CV helpers."""
    pipe.file_type = "h5"
    pipe.recording_fs = pipe.fs
    pipe.processing_fs = pipe.fs
    # Axes wide enough for every window scoring asks for (5-16 Hz, the
    # -1.5..-1.0 s trigger baseline, and the -0.25..2.5 s score window), so the
    # test exercises the real path instead of clamped stand-ins.
    pipe.freqNeeded = np.linspace(5.0, 16.0, 6)
    pipe.tvec = np.linspace(-1.6, 2.6, N_TIME)
    pipe.tvec_active = pipe.tvec
    pipe.tvec_rest = pipe.tvec
    pipe.channel_labels = [f"E{i}" for i in range(CH)]
    pipe.ideal_pattern = np.arange(float(CH))
    pipe.n_erd = 2
    pipe.candidate_scale = "raw"
    pipe.trial_counts = {}
    pipe.b1 = pipe.b2 = pipe.bpa = np.array([1.0])
    pipe._recording_details = []
    pipe._window_diagnostics = []
    pipe._file_epoch_slices = []
    pipe._montage_details = {}
    pipe._required_tf_pad = 0
    pipe.condition_number = float("nan")
    pipe.condition_warning = None
    pipe.candidate_diagnostic = {}
    pipe._input_files = {"real": [], "quasi": [], "imag": []}
    return pipe


def test_compute_ft_changes_only_the_overt_map():
    """Quasi and imagery are already out-of-sample and must be untouched."""
    pipe = _ft_ready(_pipe())
    in_sample = pipe.compute_ft(1, include_qc=False, csp_cv_folds=0)
    cross = pipe.compute_ft(1, include_qc=False, csp_cv_folds=3)

    assert np.array_equal(in_sample.tf_norm[QUASI], cross.tf_norm[QUASI])
    assert np.array_equal(in_sample.tf_norm[IMAG], cross.tf_norm[IMAG])
    assert not np.array_equal(in_sample.tf_norm[REAL], cross.tf_norm[REAL])


def test_compute_ft_records_how_the_overt_map_was_built():
    pipe = _ft_ready(_pipe())
    cross = pipe.compute_ft(1, include_qc=False, csp_cv_folds=3)
    assert cross.overt_projection == "cross-validated-kfold"
    assert cross.provenance["overt_projection"] == "cross-validated-kfold"
    assert cross.csp_cv["enabled"] is True
    assert cross.csp_cv["folds"] == 3

    in_sample = pipe.compute_ft(1, include_qc=False, csp_cv_folds=0)
    assert in_sample.overt_projection == "in-sample"
    assert in_sample.csp_cv["enabled"] is False


def test_compute_ft_rejects_a_negative_fold_count():
    pipe = _ft_ready(_pipe())
    with pytest.raises(ValueError, match="non-negative"):
        pipe.compute_ft(1, include_qc=False, csp_cv_folds=-1)


# ----------------------------------------------------- explicit filter path

def test_tf_trials_weights_override_replaces_the_full_data_filter():
    pipe = _pipe()
    pipe.freqNeeded = np.array([10.0])
    recorded = {}

    def fake_aslt(sig, fs, freqs, ncyc, order, mult):
        recorded["sig"] = sig
        return np.zeros((freqs.size, sig.size))

    import klh.pipeline as pipeline_module
    original = pipeline_module.superlet.aslt
    pipeline_module.superlet.aslt = fake_aslt
    try:
        epochs = np.ones((N_TIME, 2, CH))
        pipe._tf_trials(epochs, 0, ACTIVE, weights=np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.allclose(recorded["sig"], 10.0)   # 1+2+3+4 per sample
    finally:
        pipeline_module.superlet.aslt = original


def test_tf_trials_rejects_weights_of_the_wrong_length():
    pipe = _pipe()
    with pytest.raises(ValueError, match="weights must have 4 entries"):
        pipe._tf_trials(np.ones((N_TIME, 2, CH)), 0, ACTIVE,
                        weights=np.ones(3))
