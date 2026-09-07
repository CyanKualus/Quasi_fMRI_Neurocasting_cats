"""Regression tests for result-size guards and public CLI controls."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

import app as gui
import run_cli
from klh.scoring import ScoreResult


_QAPP = None


def _window(tmp_path):
    global _QAPP
    settings = QtCore.QSettings(
        str(tmp_path / "ui-reporting.ini"),
        QtCore.QSettings.Format.IniFormat,
    )
    _QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return gui.NeuroCastingApp(
        settings=settings, config_path=str(tmp_path / "neurocasting_settings.json"))


def test_h5_channel_widget_is_bounded_to_reference_montage(tmp_path):
    window = _window(tmp_path)
    try:
        assert window.spin_nchan.minimum() == gui.MIN_H5_CHANNELS
        assert window.spin_nchan.maximum() == gui.MAX_REFERENCE_CHANNELS
    finally:
        window.close()


def test_xdf_rate_control_is_disabled_and_cli_explains_fs_is_ignored(tmp_path):
    window = _window(tmp_path)
    try:
        assert window.cmb_ftype.currentData() == "xdf"
        assert not window.cmb_fs.isEnabled()
        assert "stream metadata" in window.cmb_fs.toolTip()

        h5_index = window.cmb_ftype.findData("h5")
        window.cmb_ftype.setCurrentIndex(h5_index)
        assert window.cmb_fs.isEnabled()
    finally:
        window.close()

    parser = run_cli.build_parser()
    assert "ignored for XDF" in parser.format_help()
    args = parser.parse_args([
        "--folder", ".", "--real", "r", "--quasi", "q", "--imag", "i",
        "--file-type", "xdf", "--extract", "markers", "--fs", "123",
    ])
    assert "--fs is ignored for XDF" in run_cli._recording_rate_message(args)


def test_topographies_and_selector_follow_returned_component_count(
        tmp_path, monkeypatch):
    window = _window(tmp_path)
    result = SimpleNamespace(
        n_components=3,
        evals=np.array([0.1, 0.2, 0.3]),
        top_patterns=np.arange(12, dtype=float).reshape(4, 3),
        channel_labels=["Fp1", "Fp2", "F7", "F3"],
    )
    calls = []

    def fake_draw(axis, values, labels=None):
        calls.append((np.asarray(values).copy(), labels))
        axis.clear()

    monkeypatch.setattr(gui.topo, "draw", fake_draw)
    try:
        window.start_result = result
        assert window._configure_component_selector(result) == 3
        assert window.slider.maximum() == 3
        assert window._selected_component() == 1

        window._plot_topos(result)
        assert len(calls) == 3
        assert all(axis.get_visible() for axis in window.ax_topos[:3])
        assert not any(axis.get_visible() for axis in window.ax_topos[3:])

        window.slider.setValue(1)
        assert window._selected_component() == 3
    finally:
        window.close()


def test_score_split_falls_back_to_legacy_arrays():
    """The split survives where it is still reported: the CLI.

    The GUI's score tab shows the total alone, so it no longer splits one.
    """
    score = SimpleNamespace(
        scores=np.array([2, 1, 2, 3, 1, 2]),
        scores_max=np.array([2, 1, 3, 3, 3, 3]),
    )
    assert run_cli._score_split(score) == (2, 2, 9, 13)


def test_zero_total_uses_placeholder_instead_of_one_point_cat(tmp_path):
    window = _window(tmp_path)
    zero_score = ScoreResult(scores=np.zeros(6))
    try:
        window._plot_scores(SimpleNamespace(score=zero_score))
        assert len(window.ax_finalimg.images) == 0
        assert any("No score image" in text.get_text()
                   for text in window.ax_finalimg.texts)
        assert "0 / 15" in window.lbl_final.text()
    finally:
        window.close()


def test_the_score_tab_states_the_total_and_nothing_else(tmp_path):
    """The gauges carry the per-criterion detail; the label carries the sum."""
    window = _window(tmp_path)
    score = ScoreResult(scores=np.array([2, 0, 1, 2, 3, 2]))
    try:
        window._plot_scores(SimpleNamespace(score=score))
        assert window.lbl_final.text() == "Total score: 10 / 15"
    finally:
        window.close()


def test_score_gauges_keep_fixed_spacing_and_wrapped_long_titles(tmp_path):
    window = _window(tmp_path)
    try:
        assert window.canvas_scores.minimumWidth() == 520
        left, right = window.ax_scores[:2]
        assert right.get_position().x0 - left.get_position().x1 > 0.1
        score = ScoreResult(
            scores=np.zeros(6),
            scoring_version=gui.SCORING_V4,
            sustained_erd_db={"quasi": 2.0, "imagery": 2.0},
        )
        specs = gui._score_gauge_specs(score)
        assert "\n" in specs[0]["title"]
        assert "\n" in specs[2]["title"]
        assert "\n" in specs[3]["title"]
        assert "\n" in specs[4]["title"]
        assert "\n" in specs[5]["title"]
    finally:
        window.close()


def test_cli_aliases_and_h5_channel_validation():
    parser = run_cli.build_parser()
    common = [
        "--folder", ".", "--real", "r", "--quasi", "q", "--imag", "i",
    ]
    args = parser.parse_args(common + [
        "--score-version", "legacy", "--baseline", "pooled",
        "--padding", "2.0",
        "--marker-shift", "-3.75",
        "--preparation", "1.75",
        "--motor-recovery", "2.25",
    ])
    assert args.score_version == "legacy-v1"
    assert args.marker_baseline == "pooled"
    assert args.marker_padding == 2.0
    assert args.marker_shift == -3.75
    assert args.preparation == 1.75
    assert args.motor_recovery == 2.25

    default_args = parser.parse_args(common)
    assert (default_args.score_version
            == "scoring-v4-rolling-peak-independent-nonreal-erd")
    v4_args = parser.parse_args(common + ["--score-version", "v4"])
    assert (v4_args.score_version
            == "scoring-v4-rolling-peak-independent-nonreal-erd")
    v3_args = parser.parse_args(common + ["--score-version", "v3"])
    assert v3_args.score_version == "scoring-v3-independent-nonreal-erd"

    invalid = parser.parse_args(common + ["--nchan", "65"])
    with pytest.raises(SystemExit):
        run_cli._validate_args(parser, invalid)


def test_cli_requires_format_specific_extraction_mode():
    parser = run_cli.build_parser()
    args = parser.parse_args([
        "--folder", ".", "--real", "r", "--quasi", "q", "--imag", "i",
        "--file-type", "xdf",
    ])
    with pytest.raises(SystemExit):
        run_cli._validate_args(parser, args)
