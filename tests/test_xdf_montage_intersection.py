"""Focused coverage for all-recording XDF montage matching."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from klh import io_h5
from klh.pipeline import KLHPipeline, _common_xdf_montage


def test_common_xdf_montage_is_normalized_reference_ordered_and_order_independent(
        monkeypatch):
    reference = ["Fp1", "C-3", "Cz", "C4"]
    labels = {
        "first.xdf": ["CZ", "c3", "FP1", "C4", "AUX-1"],
        "second.xdf": ["c.4", "fp-1", "c_3", "TRIGGER"],
    }
    monkeypatch.setattr(
        io_h5, "load_xdf_channel_labels", lambda path: labels[path])

    selected, details = _common_xdf_montage(
        ["first.xdf", "second.xdf"], reference)
    reversed_selected, reversed_details = _common_xdf_montage(
        ["second.xdf", "first.xdf"], reference)

    assert selected == ["Fp1", "C-3", "C4"]
    assert reversed_selected == selected
    assert reversed_details == details
    assert details["strategy"] == (
        "intersection-of-normalized-labels-across-all-files")
    assert details["excluded_reference_labels"] == ["Cz"]
    per_file = {os.path.basename(row["file"]): row
                for row in details["per_file"]}
    assert per_file["first.xdf"]["present_but_excluded_from_common"] == ["Cz"]
    assert per_file["second.xdf"]["missing_reference_labels"] == ["Cz"]
    assert per_file["first.xdf"]["non_reference_labels"] == ["AUX-1"]
    assert per_file["second.xdf"]["non_reference_labels"] == ["TRIGGER"]


def test_common_xdf_montage_checks_labels_in_every_selected_file(monkeypatch):
    monkeypatch.setattr(
        io_h5, "load_xdf_channel_labels",
        lambda path: ["C3", "C4"] if path == "first.xdf" else None)

    with pytest.raises(ValueError, match="second.xdf"):
        _common_xdf_montage(["first.xdf", "second.xdf"], ["C3", "C4"])


def test_common_xdf_montage_rejects_ambiguous_duplicate_reference_label(
        monkeypatch):
    monkeypatch.setattr(
        io_h5, "load_xdf_channel_labels",
        lambda path: ["C3", "c-3", "C4", "AUX", "aux"])

    with pytest.raises(ValueError, match=r"duplicate.*ambiguous\.xdf.*C3"):
        _common_xdf_montage(["ambiguous.xdf"], ["C3", "C4"])


def test_xdf_montage_exclusions_are_embedded_in_pipeline_provenance(monkeypatch):
    reference = ["C3", "Cz", "C4"]
    labels = {
        "a.xdf": ["C3", "Cz", "C4"],
        "b.xdf": ["C3", "C4"],
    }
    monkeypatch.setattr(
        io_h5, "load_xdf_channel_labels", lambda path: labels[path])
    selected, details = _common_xdf_montage(labels, reference)

    pipe = KLHPipeline()
    pipe.file_type = "xdf"
    pipe.channel_labels = selected
    pipe.ch_num = len(selected)
    pipe._montage_details = details
    pipe._input_files = {"real": ["a.xdf"], "quasi": ["b.xdf"], "imag": []}
    provenance = pipe._build_provenance()

    assert provenance["montage"]["selection"] == (
        "all-file-normalized-label-intersection-reference-order")
    assert provenance["montage"]["excluded_reference_labels"] == ["Cz"]
    assert provenance["montage"]["common_channel_count"] == 2
    assert len(provenance["montage"]["per_file"]) == 2
