"""Recognising a participant's recordings from their filenames."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.recordings import (  # noqa: E402
    classify_recording,
    discover_recordings,
    participant_code,
)


def touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


@pytest.mark.parametrize("stem,condition", [
    # Every spelling that appears in the study's own recordings.
    ("om1", "overt"), ("om2", "overt"),
    ("om1_order_01", "overt"), ("origmov1", "overt"), ("full1", "overt"),
    ("qm1", "quasi"), ("quasi2", "quasi"), ("qm4_order_03", "quasi"),
    ("im1", "imagery"), ("im1_order_01", "imagery"),
    ("mi2", "imagery"), ("mi1_finished", "imagery"),
    ("OM1", "overt"),                     # case is not part of the convention
])
def test_known_spellings_are_recognised(stem, condition):
    assert classify_recording(stem) == condition


@pytest.mark.parametrize("stem", [
    # Each of these really does begin with a condition prefix, and is still
    # not a recording, because no run number follows it.
    "important_notes",   # im
    "omitted_run",       # om
    "full_montage",      # full
    "quasi_pilot",       # quasi
    "mixed_reference",   # mi
    "realtime_check",    # real
    "notes",             # nothing at all
])
def test_a_prefix_without_a_run_number_is_not_a_condition(stem):
    """The digit is what separates a condition prefix from an ordinary word."""
    assert classify_recording(stem) is None


def test_recordings_are_split_by_condition_and_naturally_sorted(tmp_path):
    folder = touch(
        tmp_path / "008TST" / "D1",
        "om10_order_01.xdf", "om2_order_02.xdf",
        "qm1_order_03.xdf", "qm2_order_01.xdf",
        "im1_order_01.xdf", "im2_order_02.xdf")

    found = discover_recordings(folder)

    # om2 before om10: numbers compare as numbers, not as text.
    assert found.overt == ["om2_order_02", "om10_order_01"]
    assert found.quasi == ["qm1_order_03", "qm2_order_01"]
    assert found.imagery == ["im1_order_01", "im2_order_02"]
    assert found.total == 6
    # A single combined list runs overt, then quasi, then imagery.
    assert found.all() == found.overt + found.quasi + found.imagery


def test_unrecognised_names_are_reported_rather_than_filed_under_a_guess(tmp_path):
    folder = touch(tmp_path / "005TST", "om1.xdf", "practice_run.xdf")

    found = discover_recordings(folder)

    assert found.overt == ["om1"]
    assert found.unmatched == ["practice_run"]
    assert "not recognised" in found.summary()


def test_only_the_requested_extension_is_read(tmp_path):
    folder = touch(tmp_path / "005TST",
                   "om1.xdf", "om2.h5", "om1_psychopy.csv", "om1_run.json")

    assert discover_recordings(folder, ".xdf").overt == ["om1"]
    assert discover_recordings(folder, ".h5").overt == ["om2"]


def test_companion_files_never_become_recordings(tmp_path):
    """Every XDF has a psychopy/launcher/validation sidecar beside it."""
    folder = touch(tmp_path / "008TST" / "D1",
                   "om1_order_01.xdf", "om1_order_01_validation.json",
                   "om1_order_01_run.json", "om1_order_01_launcher.log")

    assert discover_recordings(folder).all() == ["om1_order_01"]


def test_subfolders_are_not_searched(tmp_path):
    """A per-day folder is a recording folder in its own right, not a source."""
    root = touch(tmp_path / "008TST")
    touch(root / "D1", "om1.xdf")

    assert discover_recordings(root).total == 0
    assert discover_recordings(root / "D1").overt == ["om1"]


def test_a_folder_that_does_not_exist_is_empty_not_an_error(tmp_path):
    """This runs while a path is still being typed."""
    found = discover_recordings(tmp_path / "nowhere")

    assert found.total == 0
    assert found.unmatched == []


@pytest.mark.parametrize("folder,expected", [
    (os.path.join("data", "new", "008TST", "D1"), "008TST"),
    (os.path.join("data", "new", "008TST", "d2"), "008TST"),
    (os.path.join("data", "new", "008TST", "Day1"), "008TST"),
    (os.path.join("data", "new", "008TST"), "008TST"),
    (os.path.join("data", "new", "008TST", "D1") + os.sep, "008TST"),
    (os.path.join("data", "new", "007TST", "D1", "D2"), "007TST"),
])
def test_a_session_subfolder_is_never_taken_as_the_participant(folder, expected):
    assert participant_code(folder) == expected


def test_a_quoted_path_is_accepted(tmp_path):
    """Paths pasted from Explorer arrive wrapped in quotes."""
    folder = touch(tmp_path / "005TST", "om1.xdf")

    assert discover_recordings(f'"{folder}"').overt == ["om1"]
    assert participant_code('"C:\\data\\005TST\\D1"') == "005TST"
