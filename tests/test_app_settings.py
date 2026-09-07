"""GUI defaults and persistence tests."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

from app import DATA_FOLDER_DEFAULT, LOCKED_SCORING_VERSION, NeuroCastingApp


def test_requested_defaults_and_close_reopen_persistence(tmp_path):
    settings_file = str(tmp_path / "klh-test.ini")
    config_file = str(tmp_path / "neurocasting_settings.json")
    settings = QtCore.QSettings(
        settings_file, QtCore.QSettings.Format.IniFormat)
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    first = NeuroCastingApp(settings=settings, config_path=config_file)
    try:
        assert first.ed_folder.text() == DATA_FOLDER_DEFAULT
        assert first.cmb_fs.currentText() == "500 Hz"
        assert first.cmb_ftype.currentData() == "xdf"
        # XDF has exactly one timing source, so the mode follows the format.
        assert first.cmb_extract.currentData() == "markers"
        assert not first.cmb_extract.isEnabled()
        assert first.spin_marker_shift.value() == -4.0
        assert first.spin_preparation.value() == 2.0
        assert first.spin_recovery.value() == 2.0
        assert first.cmb_score_version.currentData() == LOCKED_SCORING_VERSION

        # Every field the welcome form still exposes.
        first.ed_folder.setText(r"D:\custom\SUBJECT_NAME")
        first.ed_subj.setText("CUSTOM")
        first.ed_real.setText("real_a, real_b")
        first.ed_quasi.setText("quasi_a, quasi_b")
        first.ed_imag.setText("imag_a, imag_b")
        first.ed_emg_left.setText("EMG 1, EMG 2")
        first.ed_emg_right.setText("EMG 3, EMG 4")
        first.cmb_movement.setCurrentIndex(1)
        first.spin_marker_shift.setValue(-3.75)
        first.slider.setValue(3)
        first.close()
        qapp.processEvents()

        reopened_settings = QtCore.QSettings(
            settings_file, QtCore.QSettings.Format.IniFormat)
        second = NeuroCastingApp(settings=reopened_settings, config_path=config_file)
        assert second.ed_folder.text() == r"D:\custom\SUBJECT_NAME"
        assert second.ed_subj.text() == "CUSTOM"
        assert second.ed_real.text() == "real_a, real_b"
        assert second.ed_quasi.text() == "quasi_a, quasi_b"
        assert second.ed_imag.text() == "imag_a, imag_b"
        assert second.ed_emg_left.text() == "EMG 1, EMG 2"
        assert second.ed_emg_right.text() == "EMG 3, EMG 4"
        assert second.cmb_movement.currentIndex() == 1
        assert second.spin_marker_shift.value() == -3.75
        assert second.slider.value() == 3
        second.close()
        qapp.processEvents()
    finally:
        first.close()


def _window(tmp_path, name):
    settings = QtCore.QSettings(
        str(tmp_path / name), QtCore.QSettings.Format.IniFormat)
    return NeuroCastingApp(
        settings=settings,
        config_path=str(tmp_path / "neurocasting_settings.json"))


def test_defaults_are_the_documented_study_defaults(tmp_path):
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _window(tmp_path, "defaults.ini")
    try:
        assert window.ed_folder.text() == DATA_FOLDER_DEFAULT
        assert window.ed_subj.text() == "01TST"
        assert window.ed_emg_left.text() == "Aux 1.1, Aux 1.2"
        assert window.ed_emg_right.text() == "Aux 2.1, Aux 2.2"
    finally:
        window.close()
        qapp.processEvents()


def test_electrode_labels_reach_the_emg_configuration(tmp_path):
    """The welcome tab is the only place the EMG derivations are named."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _window(tmp_path, "electrodes.ini")
    try:
        window.ed_emg_left.setText(" Aux 3.1 , Aux 3.2 ")
        window.ed_emg_right.setText("Aux 4.1,Aux 4.2")
        window.ed_real.setText("om1, om2")
        window.ed_quasi.setText("qm1")
        window.ed_imag.setText("im1, om1")   # a repeat is not analysed twice
        config = window._emg_config()
        assert config.left_channels == ("Aux 3.1", "Aux 3.2")
        assert config.right_channels == ("Aux 4.1", "Aux 4.2")
        assert config.recordings == ["om1", "om2", "qm1", "im1"]
        # Both hands are always screened, whichever one the EEG half analyses.
        assert config.left_condition == "left_microrepeat"
        assert config.right_condition == "right_microrepeat"
        assert config.rest_condition == "rest"
        config.validate()
    finally:
        window.close()
        qapp.processEvents()


def test_the_marker_correction_is_shared_by_both_analyses(tmp_path):
    """One run, one time base: the EMG half reads the same two controls."""
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _window(tmp_path, "timing.ini")
    try:
        window.chk_auto_marker_shift.setChecked(False)
        window.spin_marker_shift.setValue(-3.5)
        emg = window._emg_config()
        extract = window._extract_config()
        assert emg.auto_marker_shift is False
        assert emg.marker_shift_s == -3.5
        assert extract.auto_marker_shift is False
        assert extract.marker_shift_s == -3.5

        window.chk_auto_marker_shift.setChecked(True)
        assert window._emg_config().auto_marker_shift is True
        assert window._extract_config().auto_marker_shift is True
    finally:
        window.close()
        qapp.processEvents()
