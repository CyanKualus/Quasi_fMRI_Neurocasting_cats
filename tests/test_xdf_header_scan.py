"""Montage inspection reads XML headers and seeks over sample payloads."""
import gzip
import struct

import pytest

from klh.io_h5 import load_xdf_channel_labels


def _chunk(tag, payload):
    body = struct.pack("<H", tag) + payload
    return b"\x04" + struct.pack("<I", len(body)) + body


def _header(sid, name, labels, rate):
    desc = "" if labels is None else (
        "<desc><channels>" + "".join(
            f"<channel><label>{label}</label></channel>" for label in labels)
        + "</channels></desc>")
    xml = (f"<info><name>{name}</name><channel_count>{len(labels or [0])}</channel_count>"
           f"<nominal_srate>{rate}</nominal_srate>{desc}</info>").encode()
    return _chunk(2, struct.pack("<I", sid) + xml)


@pytest.mark.parametrize("suffix", [".xdf", ".xdfz", ".xdf.gz"])
def test_reads_headers_without_decoding_samples(tmp_path, monkeypatch, suffix):
    import pyxdf
    monkeypatch.setattr(pyxdf, "load_xdf", lambda *a, **kw:
                        pytest.fail("montage scan must not load EEG samples"))
    # This payload deliberately cannot be decoded as EEG. Later stream headers
    # must still be found, without inspecting it.
    content = (b"XDF:" + _header(1, "Markers", ["event"], 0)
               + _chunk(3, struct.pack("<I", 1) + b"not samples" * 10000)
               + _header(2, "NVX136_Data", ["C4", "Cz", "C3"], 500))
    path = tmp_path / ("session" + suffix)
    path.write_bytes(content if suffix == ".xdf" else gzip.compress(content))
    assert load_xdf_channel_labels(path) == ["C4", "Cz", "C3"]


def test_renamed_amplifier_uses_widest_continuous_stream(tmp_path):
    path = tmp_path / "renamed.xdf"
    path.write_bytes(b"XDF:" + _header(1, "Markers", list("abcdef"), 0)
                     + _header(2, "aux", ["EOG"], 500)
                     + _header(3, "EEG", ["C3", "C4"], 500))
    assert load_xdf_channel_labels(path) == ["C3", "C4"]
    assert load_xdf_channel_labels(path, "aux") == ["EOG"]


def test_missing_labels_remain_unavailable(tmp_path):
    path = tmp_path / "no_labels.xdf"
    path.write_bytes(b"XDF:" + _header(1, "NVX136_Data", None, 500))
    assert load_xdf_channel_labels(path) is None


@pytest.mark.parametrize("content", [
    b"other", b"XDF:\x02", b"XDF:\x04\x01", b"XDF:\x01\x00",
    b"XDF:\x01\x06\x02\x00", b"XDF:" + _chunk(2, b"short"[:2]),
])
def test_invalid_or_truncated_headers_fail_clearly(tmp_path, content):
    path = tmp_path / "broken.xdf"
    path.write_bytes(content)
    with pytest.raises(OSError, match="XDF"):
        load_xdf_channel_labels(path)
