"""Advanced analysis parameters live in the settings file, not on the form.

The welcome tab keeps only the fields that change from subject to subject --
including the EMG electrode labels, which are a property of how a participant
was wired up. Everything else, for the EEG and the EMG half alike, is read from
``neurocasting_settings.json`` so a site's analysis parameters are one
reviewable file rather than forty widgets that can be nudged between runs.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtCore, QtWidgets

from app import (EMG_OUTPUT_ROOT, LOCKED_SCORING_VERSION, EMGConfig,
                 NeuroCastingApp)
from klh import superlet, execution

VISIBLE_WIDGETS = (
    "ed_folder", "ed_subj", "ed_real", "ed_quasi", "ed_imag",
    "ed_emg_left", "ed_emg_right",
    "cmb_movement", "spin_marker_shift", "chk_auto_marker_shift",
)
HIDDEN_WIDGETS = (
    "cmb_ftype", "cmb_fs", "spin_nchan", "cmb_extract", "spin_preparation",
    "spin_recovery", "spin_cut_first", "spin_cut_last", "cmb_marker_baseline",
    "spin_rest_skip", "spin_marker_padding", "cmb_candidate_scale",
    "spin_csp_shrinkage", "spin_csp_cv_folds", "spin_superlet_workers",
    "spin_analysis_workers",
    "cmb_score_version",
    "cmb_emg_envelope", "cmb_emg_unit", "spin_emg_rate", "spin_emg_notch",
    "spin_emg_band_low", "spin_emg_band_high", "spin_emg_window",
    "spin_emg_step", "spin_emg_pre", "spin_emg_post", "spin_emg_tail",
    "spin_emg_rest_trim_start", "spin_emg_rest_trim_end", "spin_emg_peak",
    "spin_emg_width", "spin_emg_preref_start", "spin_emg_preref_end",
    "spin_emg_adaptive_peak", "spin_emg_adaptive_width", "spin_emg_min_burst",
    "spin_emg_rest_warn", "chk_emg_trial_figures", "ed_emg_output",
)


_QAPP = None


@pytest.fixture
def window(tmp_path):
    # The application object must outlive every window; dropping the reference
    # destroys it mid-test and takes the interpreter down with it.
    global _QAPP
    _QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def build(config=None):
        config_path = tmp_path / "neurocasting_settings.json"
        if config is not None:
            config_path.write_text(json.dumps(config), encoding="utf-8")
        settings = QtCore.QSettings(
            str(tmp_path / "profile.ini"), QtCore.QSettings.Format.IniFormat)
        build.opened = NeuroCastingApp(settings=settings, config_path=str(config_path))
        build.path = config_path
        return build.opened

    build.opened = None
    yield build
    if build.opened is not None:
        build.opened.close()


def stored(build):
    return json.loads(build.path.read_text(encoding="utf-8"))


def test_only_the_per_subject_fields_remain_on_the_welcome_form(window):
    app = window()
    form = app.form_welcome
    # A row may hold a container rather than a bare field -- the marker shift
    # pairs its detect checkbox with its spin box -- so a widget on the form can
    # be a descendant of a row instead of the row's own widget.
    shown = set()
    for i in range(form.count()):
        widget = form.itemAt(i).widget()
        if widget is None:
            continue
        shown.add(widget)
        shown.update(widget.findChildren(QtWidgets.QWidget))
    for name in VISIBLE_WIDGETS:
        assert getattr(app, name) in shown, f"{name} should stay on the form"
    for name in HIDDEN_WIDGETS:
        widget = getattr(app, name)
        assert widget not in shown, f"{name} should have moved to the file"
        # Kept alive as the in-memory model, but parented off screen.
        assert widget.parentWidget() is app._advanced_holder
    assert not app._advanced_holder.isVisible()


def test_missing_file_is_created_with_the_built_in_defaults(window):
    app = window()
    assert stored(window) == {
        "analysis_workers": execution.DEFAULT_WORKERS,
        "candidate_detection_scale": "raw",
        "csp_shrinkage": 0.0,
        "cut_last_s": 0.0,
        "eeg_channels_first_n": 64,
        "extra_cut_after_preparation_s": 0.0,
        "file_type": "xdf",
        "marker_baseline": "condition-specific",
        "motor_recovery_s": 2.0,
        "overt_cv_folds": 5,
        "preparation_excluded_s": 2.0,
        "recording_sampling_rate_hz": 500,
        "scoring_version": LOCKED_SCORING_VERSION,
        "skip_rest_transition_s": 1.0,
        "superlet_workers": superlet.DEFAULT_WORKERS,
        "tf_marker_padding_s": 1.5,
        "emg_adaptive_peak_multiplier": 3.0,
        "emg_adaptive_width_multiplier": 3.0,
        "emg_band_high_hz": 150.0,
        "emg_band_low_hz": 20.0,
        "emg_classification_tail_ms": 200.0,
        "emg_envelope": "tkeo",
        "emg_envelope_step_ms": 20.0,
        "emg_envelope_window_ms": 100.0,
        "emg_input_unit": "V",
        "emg_min_burst_ms": 50.0,
        "emg_notch_hz": 50.0,
        "emg_output_root": str(EMG_OUTPUT_ROOT),
        "emg_peak_multiplier": 7.0,
        "emg_plot_after_movement_s": 2.0,
        "emg_plot_before_onset_s": 2.0,
        "emg_pre_reference_end_ms": 1800.0,
        "emg_pre_reference_start_ms": 500.0,
        "emg_processing_rate_hz": 0.0,
        "emg_rest_trim_end_s": 2.0,
        "emg_rest_trim_start_s": 1.0,
        "emg_rest_warning_percent": 5.0,
        "emg_save_trial_figures": True,
        "emg_width_multiplier": 6.0,
    }
    assert app._config_notes == []


def test_the_emg_defaults_are_the_processing_defaults(window):
    """The file states what the EMG analysis does, not a second opinion."""
    app = window()
    config = app._emg_config()
    for name in ("peak_multiplier", "background_multiplier", "min_burst_ms",
                 "band_low_hz", "band_high_hz", "window_ms", "step_ms",
                 "rest_trim_start_s", "rest_trim_end_s", "trial_tail_s",
                 "pre_reference_start_s", "pre_reference_end_s",
                 "secondary_pre_multiplier", "secondary_width_multiplier",
                 "rest_fpr_warn", "envelope", "input_unit"):
        assert getattr(config, name) == getattr(EMGConfig, name), name
    # 0 Hz on the form is "keep each recording's own rate", not "resample to 0".
    assert config.target_fs is None


def test_analysis_worker_setting_reaches_the_scheduler(window, monkeypatch):
    monkeypatch.setattr(execution, "_workers", execution.DEFAULT_WORKERS)
    app = window({"analysis_workers": 0})
    assert execution._workers == 0
    assert stored(window)["analysis_workers"] == 0
    assert app.spin_analysis_workers.value() == 0


def test_stored_emg_values_drive_the_emg_configuration(window):
    app = window({
        "emg_envelope": "rms",
        "emg_input_unit": "mV",
        "emg_processing_rate_hz": 250.0,
        "emg_band_low_hz": 30.0,
        "emg_band_high_hz": 200.0,
        "emg_peak_multiplier": 9.0,
        "emg_width_multiplier": 5.0,
        "emg_min_burst_ms": 80.0,
        "emg_classification_tail_ms": 350.0,
        "emg_pre_reference_start_ms": 400.0,
        "emg_pre_reference_end_ms": 1500.0,
        "emg_rest_warning_percent": 12.5,
        "emg_save_trial_figures": False,
        "emg_output_root": r"D:\emg-output",
    })
    config = app._emg_config()
    assert app._config_notes == []
    assert config.envelope == "rms"
    assert config.input_unit == "mV"
    assert config.target_fs == 250.0
    assert (config.band_low_hz, config.band_high_hz) == (30.0, 200.0)
    assert config.peak_multiplier == 9.0
    assert config.background_multiplier == 5.0
    assert config.min_burst_ms == 80.0
    assert config.trial_tail_s == 0.35
    assert config.pre_reference_start_s == 0.4
    assert config.pre_reference_end_s == 1.5
    assert config.rest_fpr_warn == 0.125
    assert config.trial_figures is False
    assert config.output_root == r"D:\emg-output"
    config.validate()


@pytest.mark.parametrize("bad, key", [
    ({"emg_save_trial_figures": "yes"}, "emg_save_trial_figures"),
    ({"emg_save_trial_figures": 1}, "emg_save_trial_figures"),
    ({"emg_output_root": 5}, "emg_output_root"),
    ({"emg_envelope": "hilbert"}, "emg_envelope"),
    ({"emg_peak_multiplier": 5000.0}, "emg_peak_multiplier"),
])
def test_an_unusable_emg_value_falls_back_and_is_reported(window, bad, key):
    app = window(dict(bad))
    assert any(note.startswith(f"{key}:") for note in app._config_notes)
    assert app.chk_emg_trial_figures.isChecked() is True
    assert app.cmb_emg_envelope.currentData() == "tkeo"
    assert app.spin_emg_peak.value() == EMGConfig.peak_multiplier
    assert app.ed_emg_output.text() == str(EMG_OUTPUT_ROOT)
    # The repaired file shows the value actually in force, not the rejected one.
    read = app._advanced_fields()[key][0]
    assert stored(window)[key] == read()


def test_stored_values_drive_the_analysis_parameters(window):
    app = window({
        "file_type": "h5",
        "recording_sampling_rate_hz": 1000,
        "eeg_channels_first_n": 32,
        "preparation_excluded_s": 1.25,
        "motor_recovery_s": 3.5,
        "extra_cut_after_preparation_s": 0.75,
        "cut_last_s": 0.5,
        "marker_baseline": "pooled",
        "skip_rest_transition_s": 0.25,
        "tf_marker_padding_s": 2.0,
        "candidate_detection_scale": "logit",
        "csp_shrinkage": 0.05,
        "overt_cv_folds": 8,
        "scoring_version": LOCKED_SCORING_VERSION,
    })
    assert app.cmb_ftype.currentData() == "h5"
    assert app.cmb_fs.currentData() == 1000
    assert app.spin_nchan.value() == 32
    assert app.spin_preparation.value() == 1.25
    assert app.spin_recovery.value() == 3.5
    assert app.spin_cut_first.value() == 0.75
    assert app.spin_cut_last.value() == 0.5
    assert app.cmb_marker_baseline.currentData() == "pooled"
    assert app.spin_rest_skip.value() == 0.25
    assert app.spin_marker_padding.value() == 2.0
    assert app.cmb_candidate_scale.currentData() == "logit"
    assert app.spin_csp_shrinkage.value() == 0.05
    assert app.spin_csp_cv_folds.value() == 8
    assert app._config_notes == []
    # H5 selects the trigger timing source, exactly as the combo used to.
    assert app.cmb_extract.currentData() == "trigger"


def test_stored_values_reach_the_extraction_config(window):
    app = window({
        "preparation_excluded_s": 1.25,
        "motor_recovery_s": 3.5,
        "cut_last_s": 0.5,
        "marker_baseline": "pooled",
        "tf_marker_padding_s": 2.0,
    })
    extract = app._extract_config()
    assert extract.preparation_s == 1.25
    assert extract.recovery_s == 3.5
    assert extract.cut_last == 0.5
    assert extract.marker_baseline == "pooled"
    assert extract.marker_padding == 2.0


@pytest.mark.parametrize("bad", [
    {"preparation_excluded_s": 999.0},      # outside the widget's range
    {"preparation_excluded_s": "soon"},     # not a number
    {"marker_baseline": "made-up"},         # not an offered choice
    {"eeg_channels_first_n": None},
])
def test_an_unusable_value_falls_back_and_is_reported(window, bad):
    app = window(dict(bad))
    key = next(iter(bad))
    assert app.spin_preparation.value() == 2.0
    assert app.cmb_marker_baseline.currentData() == "condition-specific"
    assert app.spin_nchan.value() == 64
    assert any(note.startswith(f"{key}:") for note in app._config_notes)
    # The repaired value is written back so the file shows what is in force.
    assert stored(window)[key] not in (bad[key],)


def test_unreadable_file_falls_back_without_blocking_the_analysis(window,
                                                                 tmp_path):
    path = tmp_path / "neurocasting_settings.json"
    path.write_text("{not json", encoding="utf-8")
    settings = QtCore.QSettings(
        str(tmp_path / "profile.ini"), QtCore.QSettings.Format.IniFormat)
    app = NeuroCastingApp(settings=settings, config_path=str(path))
    try:
        assert app.spin_preparation.value() == 2.0
        assert any("could not be read" in note for note in app._config_notes)
        assert json.loads(path.read_text(encoding="utf-8"))["file_type"] == "xdf"
    finally:
        app.close()


def test_unknown_keys_are_reported_and_dropped(window):
    app = window({"prepairation_excluded_s": 1.0})
    assert any(note.startswith("prepairation_excluded_s:")
               for note in app._config_notes)
    assert "prepairation_excluded_s" not in stored(window)


def test_scoring_version_is_locked_and_recorded(window):
    app = window({"scoring_version": "scoring-v2"})
    assert app.cmb_score_version.currentData() == LOCKED_SCORING_VERSION
    assert any("locked" in note for note in app._config_notes)
    assert stored(window)["scoring_version"] == LOCKED_SCORING_VERSION


def test_locked_scoring_version_reaches_the_pipeline(window):
    app = window({"scoring_version": "legacy-v1"})
    assert app.cmb_score_version.currentData() == LOCKED_SCORING_VERSION


def test_a_valid_file_is_left_alone(window):
    window()
    before = window.path.read_text(encoding="utf-8")
    window.opened.close()
    window.opened = None
    mtime = window.path.stat().st_mtime_ns
    app = window()
    assert window.path.read_text(encoding="utf-8") == before
    assert window.path.stat().st_mtime_ns == mtime
    assert app._config_notes == []


def test_advanced_parameters_are_not_mirrored_into_the_profile(window):
    app = window()
    app.close()
    keys = set(app.settings.allKeys())
    window.opened = None
    for stale in ("welcome/file_type", "welcome/sampling_rate",
                  "welcome/channel_count", "extraction/preparation_s",
                  "scoring/version", "csp/cv_folds"):
        assert stale not in keys
    assert "extraction/marker_shift" in keys
    assert "extraction/movement" in keys
