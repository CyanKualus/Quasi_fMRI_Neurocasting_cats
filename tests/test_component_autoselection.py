"""First-pass component selection offered when the primary analysis finishes.

Contralateral mu-ERD is what the operator normally looks for, so the slider is
moved there automatically.  mu-ERS and then ipsilateral mu-ERD are the
fallbacks.  When none of the three is recognisable the selection is not
guessed: component 1 is preselected and the operator is told to judge the
topographies by hand.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

import app as gui
from klh import qc

# A 64-slot montage: the two motor ROIs, the peripheral/ocular channels the
# plausibility heuristic penalises, and enough filler to keep the central ROI's
# share of the channels realistic (10/64).
LABELS = list(qc.CENTRAL_MOTOR_ROI + qc.PERIPHERAL_ROI)
LABELS += [f"X{i}" for i in range(64 - len(LABELS))]

CONTRA_RIGHT_HAND = qc.LEFT_MOTOR_ROI      # right hand -> left motor ROI
IPSI_RIGHT_HAND = qc.RIGHT_MOTOR_ROI

ERD_EIGENVALUE = 0.30
ERS_EIGENVALUE = 0.70
_QAPP = None


def pattern(*focus_groups):
    """A forward pattern whose energy sits on the named channels."""
    values = np.full(len(LABELS), 0.02)
    for names in focus_groups:
        for name in names:
            values[LABELS.index(name)] = 1.0
    return values


def result_of(components):
    """Assemble a start-result-like namespace from (eigenvalue, pattern)."""
    evals, patterns = zip(*components)
    return SimpleNamespace(
        n_components=len(components),
        evals=np.asarray(evals, dtype=float),
        top_patterns=np.column_stack(patterns),
        channel_labels=list(LABELS),
        component_ranking=[],
        channel_qc={},
    )


def select(components, active_condition="right_microrepeat"):
    start = result_of(components)
    return qc.select_motor_component(
        start.top_patterns, start.channel_labels, start.evals,
        active_condition=active_condition)


OCULAR_ERD = (ERD_EIGENVALUE, pattern(("Fp1", "Fp2")))
CONTRA_ERD = (0.35, pattern(CONTRA_RIGHT_HAND))
IPSI_ERD = (ERD_EIGENVALUE, pattern(IPSI_RIGHT_HAND))
MOTOR_ERS = (ERS_EIGENVALUE, pattern(CONTRA_RIGHT_HAND))


def test_contralateral_mu_erd_is_preferred_over_every_other_candidate():
    selection = select([OCULAR_ERD, IPSI_ERD, CONTRA_ERD, MOTOR_ERS])
    assert selection.automatic
    assert selection.component == 3
    assert selection.category == "contralateral-mu-erd"
    assert "contralateral mu-ERD" in selection.summary
    assert "candidate" in selection.summary
    assert "time-frequency" in selection.summary
    assert selection.message == ""


def test_mu_ers_is_second_choice_when_no_contralateral_erd_exists():
    selection = select([OCULAR_ERD, IPSI_ERD, MOTOR_ERS])
    assert selection.automatic
    assert selection.component == 3
    assert selection.category == "mu-ers"
    assert "no qualifying contralateral mu-erd candidate" in selection.summary.casefold()


def test_ipsilateral_mu_erd_is_the_last_accepted_choice():
    selection = select([OCULAR_ERD, IPSI_ERD])
    assert selection.automatic
    assert selection.component == 2
    assert selection.category == "ipsilateral-mu-erd"
    assert "no qualifying contralateral mu-erd candidate" in selection.summary.casefold()


def test_contralateral_ers_wins_its_tier_over_an_ipsilateral_one():
    selection = select([
        (ERS_EIGENVALUE, pattern(IPSI_RIGHT_HAND)),
        (ERS_EIGENVALUE, pattern(CONTRA_RIGHT_HAND)),
    ])
    assert selection.component == 2
    assert selection.category == "mu-ers"


def test_the_moving_hand_decides_which_side_is_contralateral():
    components = [IPSI_ERD, CONTRA_ERD]
    right = select(components, active_condition="right_microrepeat")
    left = select(components, active_condition="left_microrepeat")
    assert (right.component, right.category) == (2, "contralateral-mu-erd")
    assert (left.component, left.category) == (1, "contralateral-mu-erd")


def test_non_motor_topographies_are_never_selected_automatically():
    selection = select([OCULAR_ERD, (0.25, pattern(("T7", "T8")))])
    assert not selection.automatic
    assert selection.component == 1
    assert all(not response.motor_plausible
               for response in selection.responses)


def test_selection_is_invariant_to_each_patterns_reference_scale_and_polarity():
    components = [OCULAR_ERD, IPSI_ERD, CONTRA_ERD, MOTOR_ERS]
    expected = select(components)
    transformed = select([
        (eigenvalue, values * scale + offset)
        for (eigenvalue, values), scale, offset in zip(
            components, [0.25, -3.0, -0.5, 12.0], [8.0, -20.0, 4.0, 100.0])
    ])
    assert transformed.component == expected.component == 3
    assert transformed.category == expected.category
    for original, changed in zip(expected.responses, transformed.responses):
        assert changed.motor_plausible == original.motor_plausible
        assert changed.category == original.category
        for name in (
                "central_energy_fraction", "peripheral_energy_fraction",
                "contralateral_energy_fraction", "ipsilateral_energy_fraction",
                "laterality_index", "plausibility", "selection_score"):
            assert getattr(changed, name) == pytest.approx(getattr(original, name))


def test_peripheral_loading_penalises_but_does_not_veto_a_motor_candidate():
    # A left motor lobe accompanied by a stronger, opposite-polarity frontal
    # lobe remains centrally concentrated. Move that extra lobe to nonperipheral
    # channels in the second component, keeping motor energy and laterality equal.
    frontal = pattern(CONTRA_RIGHT_HAND)
    frontal[[LABELS.index("Fp1"), LABELS.index("Fp2")]] = -2.0
    other = pattern(CONTRA_RIGHT_HAND)
    other[[LABELS.index("X0"), LABELS.index("X1")]] = -2.0
    selection = select([(ERD_EIGENVALUE, frontal), (ERD_EIGENVALUE, other)])
    first, second = selection.responses
    assert first.peripheral_energy_fraction > first.central_energy_fraction
    assert first.central_energy_fraction >= len(qc.CENTRAL_MOTOR_ROI) / len(LABELS)
    assert first.motor_plausible
    assert first.category == second.category == "contralateral-mu-erd"
    assert first.central_energy_fraction == pytest.approx(second.central_energy_fraction)
    assert first.laterality_index == pytest.approx(second.laterality_index)
    assert first.selection_score < second.selection_score
    assert selection.component == 2
    assert select([(ERD_EIGENVALUE, frontal)]).automatic


def test_motor_template_similarity_orders_equally_localised_contralateral_candidates():
    first = pattern(CONTRA_RIGHT_HAND)
    second = first.copy()
    first[LABELS.index("C3")] = 2.0
    second[LABELS.index("CP3")] = 2.0
    start = result_of([(ERD_EIGENVALUE, first), (ERD_EIGENVALUE, second)])
    selection = qc.select_motor_component(
        start.top_patterns, start.channel_labels, start.evals,
        active_condition="right_microrepeat", ideal_pattern=second)
    assert selection.component == 2
    assert all(response.category == "contralateral-mu-erd"
               for response in selection.responses)
    a, b = selection.responses
    assert a.central_energy_fraction == pytest.approx(b.central_energy_fraction)
    assert a.laterality_index == pytest.approx(b.laterality_index)
    assert a.selection_score < b.selection_score


@pytest.mark.parametrize("constant", [0.0, 3.25, 7.3])
@pytest.mark.parametrize("eigenvalue", [ERD_EIGENVALUE, ERS_EIGENVALUE])
def test_spatially_constant_patterns_are_not_motor_candidates(constant, eigenvalue):
    selection = select([(eigenvalue, np.full(len(LABELS), constant))])
    assert not selection.automatic
    assert not selection.responses[0].motor_plausible
    assert selection.responses[0].category is None


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_an_invalid_channel_prevents_a_motor_candidate(invalid):
    values = pattern(CONTRA_RIGHT_HAND)
    values[LABELS.index("X0")] = invalid
    selection = select([(ERS_EIGENVALUE, values)])
    assert not selection.automatic
    assert not selection.responses[0].motor_plausible


def test_a_montage_without_central_channels_has_no_motor_candidate():
    labels = [name for name in LABELS if name not in qc.CENTRAL_MOTOR_ROI]
    values = np.zeros(len(labels))
    values[labels.index("X0")] = 1.0
    selection = qc.select_motor_component(
        values, labels, [ERS_EIGENVALUE], active_condition="right_microrepeat")
    assert not selection.automatic
    assert not selection.responses[0].motor_plausible


def test_a_bilateral_motor_erd_is_not_claimed_as_lateralised():
    selection = select([(ERD_EIGENVALUE,
                         pattern(CONTRA_RIGHT_HAND, IPSI_RIGHT_HAND))])
    assert not selection.automatic
    assert selection.responses[0].motor_plausible
    assert abs(selection.responses[0].laterality_index) < qc.LATERALITY_MARGIN


def test_a_component_at_the_neutral_eigenvalue_has_no_direction():
    selection = select([(qc.NEUTRAL_EIGENVALUE, pattern(CONTRA_RIGHT_HAND))])
    assert selection.responses[0].direction == "indeterminate"
    assert not selection.automatic


def test_the_fallback_message_asks_for_manual_selection():
    selection = select([OCULAR_ERD])
    assert not selection.automatic
    assert selection.component == 1
    assert selection.category is None
    message = selection.message.casefold()
    assert "can fail" in message
    assert "by hand" in message


def test_an_unknown_hand_still_allows_the_unlateralised_ers_choice():
    selection = select([OCULAR_ERD, MOTOR_ERS], active_condition="")
    assert selection.category == "mu-ers"
    assert np.isnan(selection.responses[0].laterality_index)


def test_evals_must_cover_every_supplied_pattern():
    start = result_of([CONTRA_ERD, IPSI_ERD])
    with pytest.raises(ValueError):
        qc.select_motor_component(
            start.top_patterns, start.channel_labels, start.evals[:1],
            active_condition="right_microrepeat")


# --------------------------------------------------------------------------
# GUI wiring
# --------------------------------------------------------------------------
@pytest.fixture
def window(tmp_path, monkeypatch):
    global _QAPP
    _QAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(
        str(tmp_path / "autoselect.ini"), QtCore.QSettings.Format.IniFormat)
    monkeypatch.setattr(gui.topo, "draw", lambda axis, values, labels=None: None)
    win = gui.NeuroCastingApp(
        settings=settings, config_path=str(tmp_path / "neurocasting_settings.json"))
    yield win
    win.close()
    _QAPP.processEvents()


def notices(window, monkeypatch):
    seen = []
    monkeypatch.setattr(
        window, "_warn_manual_component_selection", seen.append)
    return seen


def test_finishing_the_analysis_moves_the_slider_to_the_component(
        window, monkeypatch):
    seen = notices(window, monkeypatch)
    window._start_done(result_of([OCULAR_ERD, IPSI_ERD, CONTRA_ERD]))
    assert window._selected_component() == 3
    assert seen == []
    assert "contralateral mu-ERD" in window._auto_selection.summary
    assert window.lbl_component_selection.text() == window._auto_selection.summary
    assert "#3 contralateral mu-ERD" in window.lbl_component_selection.toolTip()


def test_a_failed_selection_falls_back_to_component_one_and_says_so(
        window, monkeypatch):
    seen = notices(window, monkeypatch)
    window.slider.setValue(window.slider.maximum() - 2)     # component 3
    window._start_done(result_of([OCULAR_ERD, OCULAR_ERD, OCULAR_ERD]))
    assert window._selected_component() == 1
    assert len(seen) == 1
    assert "by hand" in seen[0]
    assert ("preselected for manual inspection"
            in window._auto_selection.summary)
    assert "no qualifying contralateral mu-erd candidate" in (
        window.lbl_component_selection.text().casefold())


@pytest.mark.parametrize("fallback", [IPSI_ERD, MOTOR_ERS])
def test_gui_fallback_makes_the_missing_contralateral_candidate_visible(
        window, monkeypatch, fallback):
    notices(window, monkeypatch)
    window._start_done(result_of([OCULAR_ERD, fallback]))
    assert window._selected_component() == 2
    assert "no qualifying contralateral mu-erd candidate" in (
        window.lbl_component_selection.text().casefold())


def test_selection_summary_tracks_manual_choice_and_clears_before_a_new_run(
        window, monkeypatch):
    notices(window, monkeypatch)
    window._start_done(result_of([OCULAR_ERD, CONTRA_ERD]))
    summary = window._auto_selection.summary
    window._set_selected_component(1)
    assert window.lbl_component_selection.text().startswith("Manual selection: component 1.")
    assert summary in window.lbl_component_selection.text()
    window._set_selected_component(2)
    assert window.lbl_component_selection.text() == summary
    window._clear_eeg_results()
    assert window._auto_selection is None
    assert "Run START" in window.lbl_component_selection.text()
    assert "selected component" not in window.lbl_component_selection.text()
    assert window.lbl_component_selection.toolTip() == ""


def test_the_selection_survives_into_the_time_frequency_step(
        window, monkeypatch):
    notices(window, monkeypatch)
    window._start_done(result_of([OCULAR_ERD, CONTRA_ERD]))
    selection = window._auto_selection
    assert selection.summary.startswith(
        "selected component 2 automatically")
    evidence = qc.summarize_component_responses(selection.responses)
    assert "#1 bilateral mu-ERD (non-motor pattern)" in evidence
    assert "#2 contralateral mu-ERD" in evidence
    # The slider is where the preselection put it, so nothing was overridden.
    assert window._selected_component() == selection.component
