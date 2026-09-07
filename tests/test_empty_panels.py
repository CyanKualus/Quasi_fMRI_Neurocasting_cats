"""A panel with no result behind it must not advertise a scale.

An untouched matplotlib axes reports a range of 0..1 on both of its scales, so
the EEG tabs used to open showing components 0.0 to 1.0, a one-second window
over one hertz, and six criteria running from zero to one. None of those
numbers came from a recording, and nothing on screen distinguished them from
the ones that would replace them. These tests hold the panels blank until the
step that fills them has run, and blank again when the next run starts.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

import app as gui
from klh.scoring import ScoreResult

_QAPP = None


@pytest.fixture
def window(tmp_path):
    global _QAPP
    _QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(
        str(tmp_path / "empty-panels.ini"), QtCore.QSettings.Format.IniFormat)
    opened = gui.NeuroCastingApp(
        settings=settings,
        config_path=str(tmp_path / "neurocasting_settings.json"))
    yield opened
    opened.close()


def is_blank(ax):
    """Whether an axes shows neither a tick nor a frame."""
    return (not ax.get_xticks().size and not ax.get_yticks().size
            and not any(spine.get_visible() for spine in ax.spines.values()))


def eeg_panels(window):
    return [window.ax_evals, *window.ax_tf, *window.ax_scores,
            window.ax_finalimg]


def start_result(n_components=3):
    """A CSP result of the shape the panels are drawn from."""
    rng = np.random.default_rng(0)
    labels = ["Fp1", "Fp2", "F7", "F3"]
    return SimpleNamespace(
        n_components=n_components,
        evals=np.linspace(0.2, 0.8, n_components),
        ok_low_inds=np.array([0]),
        ok_high_inds=np.array([n_components - 1]),
        top_patterns=rng.normal(size=(len(labels), n_components)),
        channel_labels=labels,
    )


def test_a_fresh_window_shows_no_scale_on_any_eeg_panel(window):
    for ax in eeg_panels(window):
        assert is_blank(ax), ax.get_title() or ax.get_xlabel()
    # And says so in words, rather than leaving the reader to notice.
    for ax in eeg_panels(window):
        assert any(text.get_text().strip() for text in ax.texts)


def test_the_total_is_absent_rather_than_zero_before_anything_is_scored(window):
    assert window.lbl_final.text() == gui.NO_SCORE_TEXT
    assert "0" not in window.lbl_final.text()


def test_a_plotted_panel_gets_its_frame_and_ticks_back(window):
    """The blanking must not survive the data it was standing in for."""
    window._plot_evals(start_result())
    assert not is_blank(window.ax_evals)
    assert window.ax_evals.get_xlim() != (0.0, 1.0)

    window._plot_scores(SimpleNamespace(score=ScoreResult(
        scores=np.array([2, 1, 3, 2, 3, 1]),
        scoring_version=gui.SCORING_V4,
        n_erd=4, pat_sim=0.83, peakERD=np.array([7.5, 2.1, 3.4]),
        rel_nonreal_pct=64.0,
        sustained_erd_db={"quasi": 3.1, "imagery": 1.4})))
    for ax in window.ax_scores:
        assert ax.get_xticks().size, "a drawn gauge needs its scale"
        assert ax.get_facecolor()[:3] != (1.0, 0.94, 0.94), "no gate here"
    assert window.lbl_final.text() == "Total score: 12 / 15"


def test_the_selected_topography_actually_carries_its_frame(window, monkeypatch):
    """The selection frame has to be drawn, not merely made visible.

    ``topo.draw`` switches its axes off, and a switched-off axes draws no
    spines however visible they have been made -- so the frame the selection
    has always asked for never reached the screen.
    """
    def fake_draw(axis, values, labels=None):
        axis.clear()
        axis.axis("off")

    monkeypatch.setattr(gui.topo, "draw", fake_draw)
    result = start_result()
    window.start_result = result
    window._configure_component_selector(result)
    window._set_selected_component(2)
    window._plot_topos(result)

    chosen = window.ax_topos[1]
    assert chosen.axison, "the frame cannot be drawn on a switched-off axes"
    assert all(spine.get_visible() for spine in chosen.spines.values())
    assert not chosen.get_xticks().size, "only the frame comes back, not ticks"
    for other in window.ax_topos[:1] + window.ax_topos[2:]:
        assert not other.axison


def test_the_next_run_takes_the_previous_one_s_panels_down(window):
    """Results belong to the recordings they came from, not to the window."""
    window._plot_evals(start_result())
    window._plot_scores(SimpleNamespace(score=ScoreResult(scores=np.zeros(6))))
    window.btn_ft.setEnabled(True)
    window.btn_rating.setEnabled(True)

    window._clear_eeg_results()

    for ax in eeg_panels(window):
        assert is_blank(ax)
    assert window.lbl_final.text() == gui.NO_SCORE_TEXT
    assert window.start_result is None and window.ft_result is None
    assert not window.btn_ft.isEnabled()
    assert not window.btn_rating.isEnabled()
