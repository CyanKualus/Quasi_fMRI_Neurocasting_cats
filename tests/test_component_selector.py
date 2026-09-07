"""The component slider must point at the topography it selects.

A QSlider spreads its positions over the groove minus one handle length, which
is a taller span than the five topography bands drawn beside it.  Left alone,
the handle at the top of its travel sat above map 1 rather than on it, so the
selector and the maps disagreed by up to half a band.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

import app as gui

TOLERANCE_PX = 2
_QAPP = None


@pytest.fixture
def window(tmp_path):
    global _QAPP
    _QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(
        str(tmp_path / "selector.ini"), QtCore.QSettings.Format.IniFormat)
    win = gui.NeuroCastingApp(
        settings=settings, config_path=str(tmp_path / "neurocasting_settings.json"))
    win.resize(1180, 820)
    win.show()
    win.tabs.setCurrentIndex(1)          # Evaluate
    _QAPP.processEvents()
    win._align_slider_to_topos()
    _QAPP.processEvents()
    yield win
    win.close()
    _QAPP.processEvents()


def handle_center(win):
    """Vertical centre of the slider handle in the tab's own coordinates."""
    option = QtWidgets.QStyleOptionSlider()
    win.slider.initStyleOption(option)
    rect = win.slider.style().subControlRect(
        QtWidgets.QStyle.ComplexControl.CC_Slider, option,
        QtWidgets.QStyle.SubControl.SC_SliderHandle, win.slider)
    parent = win.slider_holder.parentWidget()
    return win.slider.mapTo(parent, rect.center()).y()


def band_center(win, index):
    """Vertical centre of topography ``index`` in the same coordinates."""
    parent = win.slider_holder.parentWidget()
    top = win.canvas_topos.mapTo(parent, QtCore.QPoint(0, 0)).y()
    return top + win._topo_band_centers()[index] * win.canvas_topos.height()


def test_every_slider_position_lands_on_its_topography(window):
    assert window.canvas_topos.height() > 100, "no usable geometry to test"
    for index in range(gui.MAX_DISPLAY_COMPONENTS):
        window.slider.setValue(window.slider.maximum() - index)
        _QAPP.processEvents()
        assert window._selected_component() == index + 1
        offset = abs(handle_center(window) - band_center(window, index))
        assert offset <= TOLERANCE_PX, (
            f"component {index + 1}: handle is {offset:.1f} px from its map")


def test_alignment_survives_a_resize(window):
    window.resize(900, 600)
    _QAPP.processEvents()
    window.slider.setValue(window.slider.maximum())      # component 1
    _QAPP.processEvents()
    assert abs(handle_center(window) - band_center(window, 0)) <= TOLERANCE_PX


def test_alignment_follows_a_shorter_component_list(window):
    # Three components occupy the top three bands; the slider must span those
    # three bands rather than stretching its travel over the whole column.
    result = SimpleNamespace(
        n_components=3, evals=np.zeros(3), top_patterns=np.zeros((8, 3)))
    window._configure_component_selector(result)
    _QAPP.processEvents()
    assert window.slider.maximum() == 3
    for index in range(3):
        window.slider.setValue(window.slider.maximum() - index)
        _QAPP.processEvents()
        offset = abs(handle_center(window) - band_center(window, index))
        assert offset <= TOLERANCE_PX, (
            f"component {index + 1}: handle is {offset:.1f} px from its map")


def test_ticks_are_drawn_one_per_component(window):
    assert (window.slider.tickPosition()
            != QtWidgets.QSlider.TickPosition.NoTicks)
    assert window.slider.tickInterval() == 1
    assert window.slider.singleStep() == 1
