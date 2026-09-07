"""Tests for physiological QC kept separate from KLH transfer scoring."""
import numpy as np
import pytest

from klh.qc import compute_sensorimotor_qc, rank_component_patterns


def _osc_epochs(freq, amplitudes, fs=100.0, seconds=4.0, trials=8):
    t = np.arange(0, seconds, 1 / fs)
    base = np.sin(2 * np.pi * freq * t)[:, None, None]
    amp = np.asarray(amplitudes, dtype=float)[None, None, :]
    return np.repeat(base * amp, trials, axis=1)


def test_sensor_qc_detects_right_hand_contralateral_mu_erd():
    labels = ["C3", "C4", "FC3", "FC4", "CP3", "CP4"]
    fs = 100.0
    tvec = np.arange(400) / fs
    rest = _osc_epochs(10, [2, 2, 2, 2, 2, 2], fs=fs)
    # Stronger attenuation over the left hemisphere (C3/FC3/CP3).
    active = _osc_epochs(10, [0.5, 1.2, 0.5, 1.2, 0.5, 1.2], fs=fs)
    result = compute_sensorimotor_qc(
        {"imagery": (active, rest)}, fs=fs,
        tvec_active=tvec, tvec_rest=tvec, channel_labels=labels,
        active_condition="right_microrepeat", active_window=(0.5, 2.5),
        rest_window=(1.0, 3.0), common_average=False)
    mu = result.results["imagery"]["mu"]
    assert mu.c3_db < mu.c4_db
    assert mu.contralateral_db < mu.ipsilateral_db
    assert mu.laterality_db < 0
    assert mu.channel_erd_pct[0] > mu.channel_erd_pct[1]


def test_sensor_qc_rejects_clamped_windows():
    labels = ["C3", "C4"]
    fs = 100.0
    tvec = np.arange(200) / fs
    epochs = _osc_epochs(10, [1, 1], fs=fs, seconds=2.0)
    try:
        compute_sensorimotor_qc(
            {"overt": (epochs, epochs)}, fs=fs,
            tvec_active=tvec, tvec_rest=tvec, channel_labels=labels,
            active_condition="right", active_window=(0.0, 2.5),
            rest_window=(0.0, 1.0))
    except ValueError as exc:
        assert "outside available" in str(exc)
    else:
        raise AssertionError("an unavailable QC window must not clamp silently")


def test_component_ranking_penalises_peripheral_dominance():
    labels = ["C3", "C4", "CP3", "CP4", "Fp1", "Fp2", "FT9", "FT10"]
    patterns = np.zeros((len(labels), 2))
    patterns[:4, 0] = [2.0, -2.0, 1.0, -1.0]
    patterns[4:, 1] = [2.0, 2.0, 2.0, 2.0]
    ranked = rank_component_patterns(patterns, labels)
    assert ranked[0].component == 1
    assert ranked[0].central_energy_fraction > ranked[1].central_energy_fraction
    assert ranked[0].peripheral_energy_fraction < ranked[1].peripheral_energy_fraction


def test_peripheral_loading_lowers_rank_with_identical_central_concentration():
    labels = ["C3", "C4", "CP3", "CP4", "Fp1", "Fp2", "FT9", "FT10",
              "P3", "P4", "O1", "O2"]
    patterns = np.zeros((len(labels), 2))
    patterns[:4, :] = np.array([2.0, -2.0, 1.0, -1.0])[:, None]
    patterns[4:6, 0] = [2.0, -2.0]
    patterns[8:10, 1] = [2.0, -2.0]
    ranked = rank_component_patterns(patterns, labels)
    assert [item.component for item in ranked] == [2, 1]
    assert ranked[0].central_energy_fraction == pytest.approx(
        ranked[1].central_energy_fraction)
    assert ranked[0].peripheral_energy_fraction < ranked[1].peripheral_energy_fraction
    assert ranked[0].plausibility > ranked[1].plausibility


def test_pattern_ranking_is_invariant_to_independent_reference_scale_and_polarity():
    labels = ["C3", "C4", "CP3", "CP4", "Fp1", "Fp2", "FT9", "FT10"]
    patterns = np.array([
        [2.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, -2.0],
        [-0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, 0.5],
    ])
    ideal = patterns[:, 0]
    expected = rank_component_patterns(patterns, labels, ideal_pattern=ideal)
    changed = rank_component_patterns(
        patterns * [0.5, -7.0] + [40.0, -10.0], labels,
        ideal_pattern=-3.0 * ideal + 12.0)
    assert [item.component for item in changed] == [item.component for item in expected]
    for original, transformed in zip(expected, changed):
        for name in ("central_energy_fraction", "peripheral_energy_fraction",
                     "ideal_similarity_abs", "plausibility"):
            assert getattr(transformed, name) == pytest.approx(getattr(original, name))


def test_component_similarity_is_sign_invariant_and_label_matched():
    labels = ["C4", "C3", "Fp1"]
    ideal_labels = ["Fp1", "C3", "C4"]
    ideal = np.array([0.0, 2.0, -1.0])
    pattern = np.array([[1.0], [-2.0], [0.0]])
    a = rank_component_patterns(
        pattern, labels, ideal_pattern=ideal, ideal_labels=ideal_labels)[0]
    b = rank_component_patterns(
        -pattern, labels, ideal_pattern=ideal, ideal_labels=ideal_labels)[0]
    assert abs(a.ideal_similarity_abs - 1.0) < 1e-12
    assert a.ideal_similarity_abs == b.ideal_similarity_abs


def test_per_recording_qc_reports_consistency_and_skips_empty_run():
    labels = ["C3", "C4", "FC3", "FC4", "CP3", "CP4"]
    fs = 100.0
    tvec = np.arange(400) / fs
    rest = _osc_epochs(10, [2, 2, 2, 2, 2, 2], fs=fs)
    expected = _osc_epochs(10, [0.5, 1.2, 0.5, 1.2, 0.5, 1.2], fs=fs)
    reversed_side = _osc_epochs(
        10, [1.2, 0.5, 1.2, 0.5, 1.2, 0.5], fs=fs)
    empty = expected[:, :0]
    result = compute_sensorimotor_qc(
        {"imagery": (expected, rest)}, fs=fs,
        tvec_active=tvec, tvec_rest=tvec, channel_labels=labels,
        active_condition="right_microrepeat", active_window=(0.5, 2.5),
        rest_window=(1.0, 3.0), common_average=False,
        recording_epochs={"imagery": {
            "mi1_finished.xdf": (expected, rest),
            "mi2.xdf": (reversed_side, rest),
            "empty.xdf": (empty, rest),
        }})
    summary = result.consistency["imagery"]["mu"]
    assert summary["recordings"] == 2
    assert summary["total_recordings"] == 3
    assert summary["unavailable_recordings"] == 1
    assert summary["expected_laterality_count"] == 1
    assert result.per_recording["imagery"]["empty.xdf"] == {}
    assert not result.recording_status["imagery"]["empty.xdf"]["available"]
    assert any("per-recording QC unavailable" in text for text in result.warnings)
    assert any("inconsistent across recordings" in text for text in result.warnings)
    payload = result.to_dict()
    assert payload["qc_version"] == "sensor-space-fixed-band-v2-per-recording"
    assert payload["recording_status"]["imagery"]["empty.xdf"]["active_trials"] == 0
    definition = payload["consistency_definition"]
    assert definition["expected_direction"] == "laterality_db < 0"
    assert definition["warning_fraction_threshold"] == 0.75
