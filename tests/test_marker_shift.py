"""Measuring each recording's marker-to-sample time shift from its own XDF.

The fixtures below write real (tiny) XDF files rather than mocking the reader:
the whole point of the module under test is that it reads the container's chunk
*order*, which no stub can stand in for.
"""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.marker_shift import (  # noqa: E402
    MarkerShiftError,
    estimate_marker_shift,
    resolve_marker_shift,
)


def _varlen(value: int) -> bytes:
    return bytes([4]) + int(value).to_bytes(4, "little")


def _chunk(tag: int, content: bytes) -> bytes:
    body = struct.pack("<H", tag) + content
    return _varlen(len(body)) + body


def _stream_header(stream_id, name, fmt, channels, srate):
    xml = (f"<?xml version=\"1.0\"?><info><name>{name}</name>"
           f"<type>Data</type><channel_count>{channels}</channel_count>"
           f"<nominal_srate>{srate}</nominal_srate>"
           f"<channel_format>{fmt}</channel_format></info>")
    return _chunk(2, struct.pack("<I", stream_id) + xml.encode("utf-8"))


def _data_chunk(stream_id, first_timestamp, n_samples, channels, srate):
    """One float32 sample block: only the first sample carries a timestamp."""
    payload = struct.pack("<I", stream_id) + _varlen(n_samples)
    for index in range(n_samples):
        if index == 0:
            payload += bytes([8]) + struct.pack("<d", first_timestamp)
        else:
            payload += bytes([0])
        payload += struct.pack(f"<{channels}f", *([0.0] * channels))
    return _chunk(3, payload)


def _marker_chunk(stream_id, timestamp, label):
    text = label.encode("utf-8")
    payload = (struct.pack("<I", stream_id) + _varlen(1)
               + bytes([8]) + struct.pack("<d", timestamp)
               + _varlen(len(text)) + text)
    return _chunk(3, payload)


def write_xdf(path, lag_s=5.0, n_markers=12, chunk_samples=250, srate=500.0,
              data_name="NVX136_Data", marker_name="PsychoPyMarkers",
              markers_before_data=0):
    """An XDF whose amplifier timestamps run ``lag_s`` behind its markers.

    Data arrives in blocks; after each block a marker is written whose own
    timestamp is ``lag_s`` past the newest sample in that block. That is exactly
    the relation the estimator has to recover.
    """
    parts = [b"XDF:", _chunk(1, b"<?xml version=\"1.0\"?><info>"
                                b"<version>1.0</version></info>")]
    parts.append(_stream_header(1, data_name, "float32", 2, srate))
    parts.append(_stream_header(2, marker_name, "string", 1, 0))
    for index in range(markers_before_data):
        parts.append(_marker_chunk(2, 100.0 + index, "early"))
    for index in range(n_markers):
        block_start = 100.0 + index * chunk_samples / srate
        newest = block_start + (chunk_samples - 1) / srate
        parts.append(_data_chunk(1, block_start, chunk_samples, 2, srate))
        parts.append(_marker_chunk(2, newest + lag_s, f"event_{index}"))
    path.write_bytes(b"".join(parts))
    return path


def test_lag_between_markers_and_samples_is_recovered(tmp_path):
    estimate = estimate_marker_shift(write_xdf(tmp_path / "run.xdf", lag_s=5.0))

    assert estimate.lag_s == pytest.approx(5.0, abs=1e-6)
    # The shift is the negated lag: data is stamped early, so markers move back.
    assert estimate.shift_s == pytest.approx(-5.0, abs=1e-6)
    assert estimate.n_markers == 12
    assert estimate.spread_s == pytest.approx(0.0, abs=1e-6)
    assert not estimate.warnings


def test_each_recording_gets_its_own_shift(tmp_path):
    """Two files from one session may still need different corrections."""
    first = estimate_marker_shift(write_xdf(tmp_path / "a.xdf", lag_s=4.82))
    second = estimate_marker_shift(write_xdf(tmp_path / "b.xdf", lag_s=5.44))

    assert first.shift_s == pytest.approx(-4.82, abs=1e-6)
    assert second.shift_s == pytest.approx(-5.44, abs=1e-6)


def test_markers_written_before_any_data_are_skipped(tmp_path):
    """They have no sample to be measured against, and must not become a lag."""
    path = write_xdf(tmp_path / "run.xdf", lag_s=5.0, markers_before_data=3)
    estimate = estimate_marker_shift(path)

    assert estimate.n_markers == 12          # the three early ones are dropped
    assert estimate.lag_s == pytest.approx(5.0, abs=1e-6)


def test_a_drifting_lag_is_reported_rather_than_hidden(tmp_path):
    """A recording whose lag walks is fitted by one shift only loosely."""
    parts = [b"XDF:", _chunk(1, b"<?xml version=\"1.0\"?><info>"
                                b"<version>1.0</version></info>"),
             _stream_header(1, "NVX136_Data", "float32", 2, 500.0),
             _stream_header(2, "PsychoPyMarkers", "string", 1, 0)]
    for index in range(12):
        block_start = 100.0 + index * 0.5
        newest = block_start + 249 / 500.0
        parts.append(_data_chunk(1, block_start, 250, 2, 500.0))
        # A tenth of a second of extra lag per block.
        parts.append(_marker_chunk(2, newest + 5.0 + 0.1 * index, "e"))
    (tmp_path / "drift.xdf").write_bytes(b"".join(parts))

    estimate = estimate_marker_shift(tmp_path / "drift.xdf")

    assert any("drift" in warning for warning in estimate.warnings)
    # Still usable, just not equally well at both ends of the recording.
    assert estimate.lag_s == pytest.approx(5.55, abs=0.05)


def test_a_scattered_lag_is_reported_rather_than_hidden(tmp_path):
    """No trend, but the per-marker measurements disagree a lot."""
    parts = [b"XDF:", _chunk(1, b"<?xml version=\"1.0\"?><info>"
                                b"<version>1.0</version></info>"),
             _stream_header(1, "NVX136_Data", "float32", 2, 500.0),
             _stream_header(2, "PsychoPyMarkers", "string", 1, 0)]
    for index in range(12):
        block_start = 100.0 + index * 0.5
        newest = block_start + 249 / 500.0
        jitter = 1.2 if index % 2 else -1.2
        parts.append(_data_chunk(1, block_start, 250, 2, 500.0))
        parts.append(_marker_chunk(2, newest + 5.0 + jitter, "e"))
    (tmp_path / "scattered.xdf").write_bytes(b"".join(parts))

    estimate = estimate_marker_shift(tmp_path / "scattered.xdf")

    assert any("varies" in warning for warning in estimate.warnings)


def test_a_missing_marker_stream_is_an_error_not_a_guess(tmp_path):
    """Nothing about an irregular stream's shape says "these are the markers"."""
    path = write_xdf(tmp_path / "run.xdf")

    with pytest.raises(MarkerShiftError, match="no 'OtherMarkers' stream"):
        estimate_marker_shift(path, marker_stream="OtherMarkers")


def test_an_unnamed_data_stream_falls_back_to_the_widest_continuous_one(tmp_path):
    """The app's loader falls back the same way, so this must not refuse first."""
    path = write_xdf(tmp_path / "run.xdf", lag_s=5.0, data_name="AmpXYZ")

    estimate = estimate_marker_shift(path)          # asks for NVX136_Data

    assert estimate.data_stream == "AmpXYZ"
    assert estimate.lag_s == pytest.approx(5.0, abs=1e-6)


def test_no_continuous_stream_at_all_is_an_error(tmp_path):
    parts = [b"XDF:", _chunk(1, b"<?xml version=\"1.0\"?><info>"
                                b"<version>1.0</version></info>"),
             _stream_header(2, "PsychoPyMarkers", "string", 1, 0)]
    for index in range(8):
        parts.append(_marker_chunk(2, 100.0 + index, "e"))
    (tmp_path / "markers_only.xdf").write_bytes(b"".join(parts))

    with pytest.raises(MarkerShiftError, match="no other continuous-data"):
        estimate_marker_shift(tmp_path / "markers_only.xdf")


def test_too_few_markers_is_an_error(tmp_path):
    path = write_xdf(tmp_path / "short.xdf", n_markers=3)

    with pytest.raises(MarkerShiftError, match="at least 5"):
        estimate_marker_shift(path)


def test_an_implausible_lag_is_refused(tmp_path):
    path = write_xdf(tmp_path / "wild.xdf", lag_s=900.0)

    with pytest.raises(MarkerShiftError, match="implausible"):
        estimate_marker_shift(path)


def test_a_non_xdf_file_is_refused(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"this is not a recording")

    with pytest.raises(MarkerShiftError, match="not an XDF file"):
        estimate_marker_shift(path)


def test_resolve_prefers_the_measurement_when_detection_is_on(tmp_path):
    path = write_xdf(tmp_path / "run.xdf", lag_s=5.1)

    shift, note, estimate = resolve_marker_shift(path, True, -4.0)

    assert shift == pytest.approx(-5.1, abs=1e-6)
    assert estimate is not None
    assert "detected" in note


def test_resolve_keeps_a_typed_shift_but_still_reports_the_measurement(tmp_path):
    """An operator who overrides the detector should see what they overrode."""
    path = write_xdf(tmp_path / "run.xdf", lag_s=5.1)

    shift, note, estimate = resolve_marker_shift(path, False, -4.0)

    assert shift == -4.0
    assert estimate is not None
    assert "entered by hand" in note
    assert "-5.100" in note


def test_a_typed_shift_survives_an_unmeasurable_recording(tmp_path):
    """Detection failing must not stop a run that was not relying on it."""
    path = tmp_path / "broken.xdf"
    path.write_bytes(b"not an xdf at all")

    shift, note, estimate = resolve_marker_shift(path, False, -4.0)

    assert shift == -4.0
    assert estimate is None
    assert "auto-detection unavailable" in note


def test_a_missing_file_is_raised_when_detection_is_on(tmp_path):
    with pytest.raises(OSError):
        resolve_marker_shift(tmp_path / "absent.xdf", True, -4.0)


def test_an_unmeasurable_recording_says_how_to_proceed(tmp_path):
    """The operator is told the way out instead of being left with a stack."""
    path = write_xdf(tmp_path / "run.xdf")

    with pytest.raises(MarkerShiftError, match="Untick automatic detection"):
        resolve_marker_shift(path, True, -4.0, marker_stream="Absent")
