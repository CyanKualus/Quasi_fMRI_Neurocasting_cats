"""The EMG analysis tab: what it shows, and where its numbers come from.

The tab draws nothing of its own. Every panel comes from the arrays
``analyze_batch`` produced and from the same drawing functions that write the
saved figures, so these tests check the wiring and the reporting rather than
the classifier, which ``test_emg_core.py`` covers.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as gui                                              # noqa: E402
import emg_view                                                # noqa: E402
from emgcasting import core                                    # noqa: E402
from emgcasting.core import (EventSet, LoadedRecording,        # noqa: E402
                             ProcessingConfig)

_QAPP = None


@pytest.fixture(scope="module")
def qapp():
    global _QAPP
    _QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _QAPP


def _recording(rng, active_left, active_right):
    """One synthetic recording: 12 trials a hand, plus a long rest block.

    ``active_*`` is how many of the twelve trials carry a burst, so the share
    the tab reports has a known right answer.
    """
    fs = 500.0
    time = np.arange(0, 320, 1 / fs)
    signals = {name: rng.normal(scale=1e-6, size=time.size)
               for name in ("L+", "L-", "R+", "R-")}
    left = np.arange(20.0, 116.0, 8.0)
    right = np.arange(160.0, 256.0, 8.0)
    for channel, onsets, active in (("L+", left, active_left),
                                    ("R+", right, active_right)):
        for onset in onsets[:active]:
            window = (time >= onset + 2.2) & (time < onset + 3.2)
            signals[channel][window] += 1.2e-4 * np.sin(
                2 * np.pi * 70 * time[window])
    events = {
        "left": EventSet(left, np.full(left.size, 8.0)),
        "right": EventSet(right, np.full(right.size, 8.0)),
        "rest": EventSet(np.array([120.0, 260.0]), np.full(2, 35.0)),
    }
    return signals, fs, float(time[-1]), events


@pytest.fixture
def batch(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    signals, fs, duration, events = _recording(rng, active_left=9,
                                               active_right=3)
    monkeypatch.setattr(core, "load_xdf", lambda path, requested, config:
                        LoadedRecording(signals, fs, duration, events, 1.0))
    for name in ("om1", "qm1"):
        (tmp_path / f"{name}.xdf").write_bytes(b"")
    config = ProcessingConfig(
        data_dir=str(tmp_path), recordings=["om1", "qm1"], participant="P42",
        left_channels=("L+", "L-"), right_channels=("R+", "R-"),
        left_condition="left", right_condition="right", rest_condition="rest",
        output_root=str(tmp_path / "out"), auto_marker_shift=False,
        marker_shift_s=0.0)
    return core.analyze_batch(config, require_metrics=False), config


def test_every_recording_gets_a_card_with_both_hands(qapp, batch):
    analysed, config = batch
    tab = emg_view.EMGTab(lambda: config)
    tab.config = config
    tab._analysis_done(analysed)
    try:
        cards = [tab.cards_layout.itemAt(i).widget()
                 for i in range(tab.cards_layout.count() - 1)]
        assert len(cards) == len(analysed.recordings)
        for card, result in zip(cards, analysed.recordings):
            assert result.recording in card.title()
            # Two mean panels and two trial buttons, left hand then right.
            buttons = card.findChildren(QtWidgets.QPushButton)
            assert [b.text().split("-")[0] for b in buttons] == ["left", "right"]
            assert all(b.isEnabled() for b in buttons)
        assert "2 recording(s)" in tab.lbl_status.text()
        assert "4 hands scored" in tab.lbl_status.text()
    finally:
        tab.deleteLater()


def test_a_figure_does_not_eat_the_wheel_the_page_needs_to_scroll(qapp, batch):
    """Matplotlib's canvas accepts every wheel turn; a card's must not.

    The tab is mostly figures, so a canvas that keeps the event leaves the
    scroll area unable to scroll wherever the pointer actually is.
    """
    analysed, config = batch
    tab = emg_view.EMGTab(lambda: config)
    tab.config = config
    tab._analysis_done(analysed)
    try:
        canvases = tab.cards.findChildren(emg_view.StaticCanvas)
        assert canvases
        point = QtCore.QPointF(10.0, 10.0)
        event = QtGui.QWheelEvent(
            point, point, QtCore.QPoint(0, 0), QtCore.QPoint(0, -120),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase, False)
        canvases[0].wheelEvent(event)
        assert not event.isAccepted()
    finally:
        tab.deleteLater()


def test_the_reported_share_is_the_classifier_count(qapp, batch):
    analysed, _config = batch
    left, right = analysed.recordings[0].hands
    assert (left.n_high_trials, left.n_trials) == (9, 12)
    assert (right.n_high_trials, right.n_trials) == (3, 12)
    headline, note = emg_view._percent_text(left)
    assert headline == "9/12 high EMG = 75%"
    # A share the detector stands behind carries no second number: the rest
    # hit rate is a property of the analysis, is reported in emg_summary.csv,
    # and beside the share was read as a second result.
    assert note == ""
    assert "rest" not in headline
    assert emg_view._percent_text(right)[0] == "3/12 high EMG = 25%"


def test_a_share_known_to_be_an_underestimate_still_says_so(qapp):
    """Dropping the rest floor must not drop the caveat on the number shown."""
    hand = core.HandResult(
        "left", "left", ("L+", "L-"), 12, 4.0, np.nan, np.nan, np.nan, "",
        n_high_trials=9, threshold_note="this count is an underestimate")
    headline, note = emg_view._percent_text(hand)
    assert headline.startswith("9/12 high EMG")
    assert note == "this count is an underestimate"


def test_an_unscored_hand_says_so_instead_of_showing_zero(qapp):
    hand = core.HandResult(
        "left", "left", ("L+", "L-"), 12, 4.0, np.nan, np.nan, np.nan, "",
        n_high_trials=None, threshold_note="not scored: no clean rest")
    headline, note = emg_view._percent_text(hand)
    assert headline == "not scored"
    assert "no clean rest" in note


def test_the_trial_window_shows_every_trial_and_enlarges_one(qapp, batch):
    analysed, config = batch
    hand = analysed.recordings[0].hands[0]
    window = emg_view.TrialsWindow("om1", hand, config)
    try:
        assert len(window._axes) == hand.n_trials
        assert window.selected == 0

        # Clicking a thumbnail is the same act as stepping onto it.
        axes = {index: ax for ax, index in window._axes.items()}
        window._clicked(type("Event", (), {"inaxes": axes[4]})())
        assert window.selected == 4
        window._step(1)
        assert window.selected == 5
        window._step(-6)                       # wraps rather than sticking
        assert window.selected == hand.n_trials - 1

        report = window._metrics_report(0)
        assert "trial 1 of 12" in report
        assert "HIGH EMG" in report            # the first trial carries a burst
        assert "primary branch" in report and "adaptive branch" in report
        # The recording and hand are named once, in the title bar, so nothing
        # above the trial grid repeats them.
        assert not hasattr(window, "_header_text")
        assert "om1" in window.windowTitle()
        assert "left hand" in window.windowTitle()
    finally:
        window.deleteLater()


def test_high_trials_are_marked_and_low_ones_are_not(qapp, batch):
    analysed, config = batch
    hand = analysed.recordings[0].hands[0]
    metrics = hand.analysis.metrics
    window = emg_view.TrialsWindow("om1", hand, config)
    try:
        pink = {}
        for ax, index in window._axes.items():
            pink[index] = ax.get_facecolor()[:3] != (1.0, 1.0, 1.0)
        assert pink == {i: bool(metrics.high[i]) for i in range(hand.n_trials)}
    finally:
        window.deleteLater()


def test_a_skipped_run_clears_the_previous_participant(qapp, batch):
    analysed, config = batch
    tab = emg_view.EMGTab(lambda: config)
    tab.config = config
    tab._analysis_done(analysed)
    assert tab.cards_layout.count() - 1 == 2
    try:
        tab.skip("EMG analysis needs XDF recordings")
        assert tab.cards_layout.count() - 1 == 0
        assert tab.batch is None
        assert not tab.btn_save.isEnabled()
        assert "XDF" in tab.lbl_status.text()
    finally:
        tab.deleteLater()


def test_nothing_is_written_until_saving_is_asked_for(batch):
    analysed, config = batch
    assert not os.path.exists(analysed.output_dir)
    assert analysed.summary_csv == ""
    assert all(hand.figure_path == ""
               for rec in analysed.recordings for hand in rec.hands)

    core.save_batch_outputs(analysed, config)

    assert os.path.isfile(analysed.summary_csv)
    for rec in analysed.recordings:
        for hand in rec.hands:
            assert os.path.isfile(hand.figure_path)
            assert os.path.isfile(hand.trial_metrics_csv)
            figures = os.listdir(hand.trial_dir)
            assert len(figures) == hand.n_trials + 1   # + trial_metrics.csv


def test_start_runs_the_emg_half_first_and_then_the_eeg_one(
        qapp, tmp_path, monkeypatch):
    """START is one action: the faster EMG half runs, then releases the EEG."""
    folder = tmp_path / "007TST"
    folder.mkdir()
    for name in ("om1", "qm1", "im1"):
        (folder / f"{name}.xdf").write_bytes(b"")

    class StubWorker(QtCore.QThread):
        done = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()

        def run(self):
            self.failed.emit("stub EEG failure")

    monkeypatch.setattr(gui, "StartWorker", StubWorker)
    settings = QtCore.QSettings(
        str(tmp_path / "start.ini"), QtCore.QSettings.Format.IniFormat)
    window = gui.NeuroCastingApp(
        settings=settings,
        config_path=str(tmp_path / "neurocasting_settings.json"))
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical",
        staticmethod(lambda *args, **kwargs: None))
    started = []

    def fake_start():
        started.append(True)
        return True

    monkeypatch.setattr(window.emg_tab, "start", fake_start)
    try:
        window.ed_folder.setText(str(tmp_path / "SUBJECT_NAME"))
        window.ed_subj.setText("007TST")
        window.ed_real.setText("om1")
        window.ed_quasi.setText("qm1")
        window.ed_imag.setText("im1")
        window.on_start()
        # The EMG half is running and the EEG half is queued behind it.
        assert started == [True]
        assert window._pending_eeg is not None
        assert not hasattr(window, "worker")

        window.emg_tab.analysis_finished.emit(True)
        qapp.processEvents()
        window.worker.wait(5000)
        qapp.processEvents()
        # A finished EMG half raises its own tab, then releases the EEG one --
        # which failed here, without taking the EMG screening down with it.
        assert window.tabs.currentWidget() is window.emg_tab
        assert window._pending_eeg is None

        # Both halves report on the start tab's one status line, and neither
        # overwrites the other's last word.
        window._set_status(eeg="done.")
        window.emg_tab._set_status("4 hands scored")
        assert window.lbl_status.text() == "done.\nEMG: 4 hands scored"

        # H5 has no auxiliary EMG channels, so that run is reported, not run.
        window.cmb_ftype.setCurrentIndex(window.cmb_ftype.findData("h5"))
        assert window._start_emg() is False
        assert started == [True]
        assert "XDF" in window.emg_tab.lbl_status.text()
    finally:
        window.close()
        qapp.processEvents()
