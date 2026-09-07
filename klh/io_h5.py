"""Data loading, trigger decoding and epoching.

Port of the HDF5 reading + ``trigDetector`` logic from the MATLAB app
(KLH_100.mlapp / musaccade_checkERD.m).

The .h5 files store a single group ``/eeg`` with a dataset ``channels`` of
shape (n_samples, 67): columns 0..63 are the 64 EEG channels and the last
column holds a decimally-encoded trigger word.
"""
from __future__ import annotations

import os
import gzip
import struct
import xml.etree.ElementTree as ET
from fractions import Fraction

import numpy as np
import h5py
from scipy import ndimage
from scipy.signal import resample_poly


N_EEG = 64  # number of EEG channels used


def resample_eeg(eeg: np.ndarray, recording_fs: float,
                 processing_fs: float) -> np.ndarray:
    """Polyphase-resample continuous EEG without touching event channels.

    Trigger/event streams are categorical data and must never be passed through
    a reconstruction filter.  Callers detect events at ``recording_fs`` and use
    :func:`remap_sample_indices` to place their onsets in the returned signal.
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    recording_fs = float(recording_fs)
    processing_fs = float(processing_fs)
    if recording_fs <= 0 or processing_fs <= 0:
        raise ValueError("recording_fs and processing_fs must be positive")
    if abs(recording_fs - processing_fs) <= 1e-6:
        return eeg
    frac = Fraction(processing_fs / recording_fs).limit_denominator(1000)
    return resample_poly(eeg, frac.numerator, frac.denominator, axis=0)


def remap_sample_indices(indices: np.ndarray, recording_fs: float,
                         processing_fs: float) -> np.ndarray:
    """Map native sample indices to another rate through elapsed seconds."""
    indices = np.asarray(indices, dtype=np.float64)
    recording_fs = float(recording_fs)
    processing_fs = float(processing_fs)
    if recording_fs <= 0 or processing_fs <= 0:
        raise ValueError("recording_fs and processing_fs must be positive")
    return np.rint(indices / recording_fs * processing_fs).astype(int)


def load_h5_eeg(path: str) -> np.ndarray:
    """Return the raw ``/eeg/channels`` matrix as (n_samples, n_cols) float64."""
    with h5py.File(path, "r") as f:
        data = np.asarray(f["eeg"]["channels"][()], dtype=np.float64)
    # Guard against transposed storage: we expect many more samples than columns.
    if data.shape[0] < data.shape[1]:
        data = data.T
    return data


def _pick_xdf_stream(streams, data_stream: str):
    """Choose the continuous-data stream from a loaded XDF file.

    Prefers a stream named ``data_stream`` (e.g. ``NVX136_Data``); otherwise
    falls back to the numeric stream with a positive sample rate and the most
    channels (the amplifier stream, never the 1-channel marker/event streams).
    """
    for s in streams:
        if s["info"]["name"] and s["info"]["name"][0] == data_stream:
            return s
    cand = [s for s in streams if float(s["info"]["nominal_srate"][0]) > 0]
    if not cand:
        raise ValueError("no continuous-data stream found in XDF file")
    return max(cand, key=lambda s: int(s["info"]["channel_count"][0]))


def _xdf_channel_labels(stream) -> list[str] | None:
    """Channel labels from an XDF stream's ``desc``, or None if absent."""
    try:
        chs = stream["info"]["desc"][0]["channels"][0]["channel"]
        return [c["label"][0] for c in chs]
    except (KeyError, IndexError, TypeError):
        return None


def load_xdf_channel_labels(path: str, data_stream: str = "NVX136_Data") \
        -> list[str] | None:
    """Return channel labels by seeking over XDF sample payloads.

    This is primarily used to match the recording montage before the full XDF
    signal is loaded.  Some NVX recordings omit the reference electrode (Cz),
    so assuming that every XDF contains all 64 reference-montage channels makes
    otherwise valid recordings unusable.
    """
    # pyxdf.load_xdf(select_streams=...) still decodes the selected stream's
    # samples. Montage discovery needs only its XML header. Keep all headers
    # so a renamed amplifier uses the same widest-stream fallback as loading.
    compressed = str(path).lower().endswith((".xdfz", ".gz"))
    opener = gzip.open if compressed else open
    streams = []
    with opener(path, "rb") as handle:
        def read_exact(size):
            value = handle.read(size)
            if len(value) != size:
                raise OSError(f"truncated XDF header in {path}")
            return value

        if read_exact(4) != b"XDF:":
            raise OSError(f"invalid XDF file {path}")
        file_size = None if compressed else os.fstat(handle.fileno()).st_size
        while True:
            width = handle.read(1)
            if not width:
                break
            if width[0] not in (1, 4, 8):
                raise OSError(f"invalid XDF chunk length in {path}")
            size = int.from_bytes(read_exact(width[0]), "little")
            end = handle.tell() + size
            if size < 2 or (file_size is not None and end > file_size):
                raise OSError(f"invalid or truncated XDF chunk in {path}")
            tag = struct.unpack("<H", read_exact(2))[0]
            if tag == 2:
                if size < 6:
                    raise OSError(f"invalid XDF stream header in {path}")
                read_exact(4)  # stream id
                root = ET.fromstring(read_exact(size - 6).decode("utf-8", "replace"))
                channels = root.findall("./desc/channels/channel")
                labels = [node.findtext("label") for node in channels]
                streams.append({
                    "info": {key: [root.findtext(key) or default]
                             for key, default in (("name", ""),
                                                  ("nominal_srate", "0"),
                                                  ("channel_count", "0"))},
                    "labels": labels if labels and None not in labels else None,
                })
            handle.seek(end)
    return _pick_xdf_stream(streams, data_stream)["labels"]


def _marker_values(marker) -> list[str]:
    """Flatten an XDF marker stream's ``time_series`` rows to label strings."""
    return [str(row[0] if isinstance(row, (list, tuple, np.ndarray)) else row)
            for row in marker.get("time_series", [])]


def marker_event_sets(marker, data_start: float, duration_s: float,
                      wanted) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{condition: (onsets_s, durations_s)}`` from a marker stream.

    Onsets are measured from ``data_start``, the timestamp of the continuous
    stream's first sample, so they are directly comparable with sample indices.
    A marker's duration is the gap to the next marker of *any* label, which is
    how the recorder encodes block structure — the paradigm emits a `rest` or
    `*_task_block` marker at the end of every micro-repeat.

    The recorder writes both streams in one clock. If a file's markers fall
    outside its own recording this raises rather than guessing an offset:
    without a shared clock the events cannot be placed at all.
    """
    stamps = np.asarray(marker.get("time_stamps", []), dtype=np.float64)
    values = _marker_values(marker)
    relative = stamps - data_start
    if stamps.size and (np.nanmax(relative) < -1.0
                        or np.nanmin(relative) > duration_s + 1.0):
        raise ValueError(
            "marker and data timestamps are in incompatible clocks; this "
            "recording cannot be epoched from its own markers")

    unique_times = np.unique(relative[np.isfinite(relative)])
    events: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for condition in wanted:
        mask = np.array([v == condition for v in values], dtype=bool) \
            if values else np.zeros(relative.size, dtype=bool)
        onsets = relative[mask]
        durations = []
        for onset in onsets:
            later = unique_times[unique_times > onset + 1e-6]
            durations.append(float(later[0] - onset) if later.size else np.nan)
        events[condition] = (np.asarray(onsets, dtype=np.float64),
                             np.asarray(durations, dtype=np.float64))
    return events


def marker_video_onset(marker, data_start: float,
                       prefix: str = "video_onset") -> float | None:
    """Seconds from the first sample to the ``video_onset`` marker, or None.

    Purely informational now that epochs come from the markers themselves; it
    is reported so that an unexpected acquisition/stimulus gap stays visible.
    """
    stamps = np.asarray(marker.get("time_stamps", []), dtype=np.float64)
    for value, stamp in zip(_marker_values(marker), stamps):
        if value.startswith(prefix):
            return float(stamp - data_start)
    return None


def load_xdf_event_durations(path: str, wanted,
                             marker_stream: str = "PsychoPyMarkers") \
        -> dict[str, np.ndarray]:
    """Event durations per condition, without reading the samples.

    Durations are differences between marker timestamps, so they need no
    reference to the data clock and only the marker stream has to be parsed.
    That keeps the pre-scan which fixes common window lengths cheap — reading
    one marker stream costs milliseconds against a fraction of a second for the
    whole recording.
    """
    import pyxdf
    streams, _ = pyxdf.load_xdf(
        path, select_streams=[{"name": marker_stream}],
        synchronize_clocks=False, dejitter_timestamps=False)
    if not streams:
        raise ValueError(
            f"marker stream '{marker_stream}' not found in "
            f"{os.path.basename(path)}")
    events = marker_event_sets(streams[0], 0.0, np.inf, wanted)
    return {name: durations for name, (_, durations) in events.items()}


def _norm_label(s: str) -> str:
    """Normalise an electrode name for matching (case/punctuation-insensitive)."""
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_xdf_eeg(path: str, event_conditions, target_fs: float | None = None,
                  pick_labels: list[str] | None = None,
                  data_stream: str = "NVX136_Data",
                  marker_stream: str = "PsychoPyMarkers",
                  return_info: bool = False):
    """Load an XDF recording together with its embedded event markers.

    Returns ``(data, events, video_onset_s)``: the amplifier stream's samples as
    (n_samples, n_channels) float64 in acquisition order, the marker events for
    the labels in ``event_conditions`` (see :func:`marker_event_sets`), and the
    informational position of the ``video_onset`` marker.

    With ``return_info=True`` a fourth dictionary reports the native/output
    rates and whether polyphase resampling occurred.  The default three-value
    return is retained for existing callers.

    Marker and sample timestamps are used exactly as recorded. pyxdf clock
    synchronization stays disabled: this recorder writes both streams in one
    Unix-epoch clock, so applying LSL clock corrections to those deliberately
    non-LSL timestamps would shift the two streams by different amounts.

    ``pick_labels`` selects and reorders channels by electrode name (case- and
    punctuation-insensitive). This is used to reduce a high-density cap to the
    reference montage: e.g. a 128-ch recording is restricted to the 64
    electrodes of ``mumeg_mks64.ced`` in the requested reference order, so the
    CSP pattern, topographies and scoring line up with the reference
    ``ideal_pattern``.
    Without it, channels are kept in file order (pipeline slices first ``ch_num``).

    When ``target_fs`` differs from the recording's nominal rate the signal is
    polyphase-resampled to it. This is how 500 Hz recordings are brought to the
    1000 Hz the pre-built FIR filters (``b1``/``b2``/``bpa``) were designed for,
    so the 3-43 Hz broadband and 9-14.5 Hz alpha bands stay physically correct.
    Event onsets are in seconds, so resampling leaves them untouched.
    """
    import pyxdf  # imported lazily so the H5 path has no pyxdf dependency

    streams, _ = pyxdf.load_xdf(
        path, synchronize_clocks=False, dejitter_timestamps=False)
    stream = _pick_xdf_stream(streams, data_stream)
    data = np.asarray(stream["time_series"], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"unexpected XDF data shape {data.shape} in {path}")

    if pick_labels is not None:
        labels = _xdf_channel_labels(stream)
        if labels is None:
            raise ValueError(
                f"channel labels missing in {os.path.basename(path)}; cannot "
                "select by name")
        lut = {_norm_label(l): i for i, l in enumerate(labels)}
        idx, missing = [], []
        for name in pick_labels:
            key = _norm_label(name)
            (idx.append(lut[key]) if key in lut else missing.append(name))
        if missing:
            raise ValueError(
                f"electrodes not found in {os.path.basename(path)}: {missing}")
        data = data[:, idx]

    fs = float(stream["info"]["nominal_srate"][0])
    marker = next(
        (s for s in streams
         if s["info"]["name"] and s["info"]["name"][0] == marker_stream), None)
    if marker is None:
        raise ValueError(
            f"marker stream '{marker_stream}' not found in "
            f"{os.path.basename(path)}")
    stamps = np.asarray(stream.get("time_stamps", []), dtype=np.float64)
    if stamps.size < 2:
        raise ValueError(
            f"no sample timestamps in {os.path.basename(path)}; events cannot "
            "be placed in the recording")
    duration = float(stamps[-1] - stamps[0])
    events = marker_event_sets(marker, float(stamps[0]), duration,
                               set(event_conditions))
    video_onset_s = marker_video_onset(marker, float(stamps[0]))

    output_fs = fs if target_fs is None else float(target_fs)
    was_resampled = abs(fs - output_fs) > 1e-6
    data = resample_eeg(data, fs, output_fs)
    result = (data, events, video_onset_s)
    if return_info:
        return result + ({
            "recording_fs": fs,
            "processing_fs": output_fs,
            "resampled": was_resampled,
            "resampling_method": "scipy.signal.resample_poly" if was_resampled else "none",
        },)
    return result


def decode_trigger(eeg_full: np.ndarray) -> np.ndarray:
    """Reproduce ``TRIG = de2bi(eeg(:,end)); TRIG = 1 - TRIG(:,1)``.

    ``de2bi`` returns the binary digits least-significant-bit first, so
    ``TRIG(:,1)`` is bit 0 of the last column. TRIG is then inverted.
    """
    last = np.rint(eeg_full[:, -1]).astype(np.int64)
    bit0 = last & 1
    return (1 - bit0).astype(np.float64)


def _pixel_index_lists(mask: np.ndarray) -> list[np.ndarray]:
    """Emulate MATLAB ``bwconncomp(mask).PixelIdxList`` for a 1-D mask.

    Returns a list of 0-based index arrays, one per connected run of True.
    """
    labels, n = ndimage.label(mask.astype(int))
    return [np.where(labels == i)[0] for i in range(1, n + 1)]


def _strfind(seq: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """0-based equivalent of MATLAB ``strfind(seq, pattern)`` on numeric rows.

    Returns start indices where ``pattern`` occurs contiguously in ``seq``.
    """
    n, m = seq.size, pattern.size
    if m == 0 or m > n:
        return np.array([], dtype=int)
    # Sliding-window comparison.
    windows = np.lib.stride_tricks.sliding_window_view(seq, m)
    hits = np.all(windows == pattern[None, :], axis=1)
    return np.where(hits)[0]


def _dsearchn(points: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Nearest-neighbour indices of ``query`` into 1-D sorted-ish ``points``.

    MATLAB ``dsearchn(points, query)`` returns, for each query, the index of
    the closest point. Values returned here are 0-based positions into
    ``points``.
    """
    points = np.asarray(points, dtype=np.float64).ravel()
    query = np.asarray(query, dtype=np.float64).ravel()
    idx = np.abs(points[None, :] - query[:, None]).argmin(axis=1)
    return idx


def trig_detector(trig: np.ndarray, fs: float = 1000.0):
    """Port of the ``trigDetector`` helper.

    Returns (tr, ts, td1, td2) — 0-based sample indices of the onset markers
    for rest, single, and (two) double events. Empty arrays when a class has
    too few detections. ``td2`` is only meaningful for the double condition
    (and mirrors a latent MATLAB bug — see note below).

    The MATLAB detector's window lengths are sample counts tuned for 1000 Hz;
    they are scaled by ``fs / 1000`` so timing is preserved at other rates.
    At ``fs == 1000`` the constants are unchanged (identical to the original).
    """
    trig = np.asarray(trig, dtype=np.float64).ravel()
    k = fs / 1000.0
    n120, n10, n250, n350 = (int(round(n * k)) for n in (120, 10, 250, 350))
    n150, n100 = int(round(150 * k)), int(round(100 * k))
    d = np.diff(trig)
    upstair = np.concatenate(([0.0], (d > 0).astype(float)))
    downstair = np.concatenate(([0.0], (d < 0).astype(float)))

    waitup = _strfind(trig, np.concatenate((np.zeros(n120), [1.0]))) + n120
    waitupplato = _strfind(trig, np.concatenate((np.zeros(n10), np.ones(n250)))) + n10
    dipdown = _strfind(trig, np.concatenate((np.ones(n250), [0.0]))) + n250

    # movsum with window 350 (centred, like MATLAB movsum default).
    sup = _movsum(upstair, n350)
    sdown = _movsum(downstair, n350)

    def _onsets(mask, min_len):
        runs = _pixel_index_lists(mask)
        runs = [r for r in runs if r.size >= min_len]
        if not runs:
            return np.array([], dtype=int)
        firsts = np.array([r[0] for r in runs], dtype=float)
        return waitup[_dsearchn(waitup, firsts)]

    trials_rest = _onsets((sup == 2) & (sdown == 1), n150)
    trials_single = _onsets((sup == 2) & (sdown == 2), n100)

    # doubles: no length filter in the MATLAB source
    runs_d = _pixel_index_lists((sup == 3) & (sdown == 2))
    if runs_d:
        firsts_d = np.array([r[0] for r in runs_d], dtype=float)
        trials_double = waitup[_dsearchn(waitup, firsts_d)]
    else:
        trials_double = np.array([], dtype=int)

    tr = ts = td1 = td2 = np.array([], dtype=int)
    if trials_rest.size > 3:
        tr = waitupplato[_dsearchn(waitupplato, trials_rest.astype(float))]
    if trials_single.size > 3:
        ts = waitupplato[_dsearchn(waitupplato, trials_single.astype(float))]
    if trials_double.size > 5:
        td1 = waitupplato[_dsearchn(waitupplato, trials_double.astype(float))]
        # NOTE: MATLAB references undefined `trials_double_first` here, so this
        # branch would error in MATLAB too. Kept faithful but guarded.
        td2 = dipdown[_dsearchn(dipdown, trials_double.astype(float))]
    return tr, ts, td1, td2


def _movsum(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving sum matching MATLAB ``movsum(x, w)`` (integer window)."""
    x = np.asarray(x, dtype=np.float64)
    # MATLAB centred window of length w: for even w, it uses w samples with
    # the extra element on the trailing side.
    left = w // 2
    right = w - 1 - left
    padded = np.concatenate((np.zeros(left), x, np.zeros(right)))
    csum = np.cumsum(padded)
    csum = np.concatenate(([0.0], csum))
    return csum[w:] - csum[:-w]


def epoch(signal: np.ndarray, onsets: np.ndarray, epoind: np.ndarray,
          *, return_valid: bool = False):
    """Cut epochs from a (n_samples, n_ch) signal.

    Returns an array shaped (n_time, n_trials, n_ch) matching the MATLAB
    ``reshape(EEGf(epoind_matrix, :), numel(epoind), [], n_ch)`` pattern. With
    ``return_valid=True``, also returns the boolean mask of requested onsets
    retained after the recording-boundary check.
    """
    onsets = np.asarray(onsets, dtype=int).ravel()
    if onsets.size == 0:
        out = np.empty((epoind.size, 0, signal.shape[1]))
        valid = np.zeros(0, dtype=bool)
        return (out, valid) if return_valid else out
    idx = onsets[None, :] + epoind[:, None]  # (n_time, n_trials)
    # Guard against out-of-range indices.
    valid = np.all((idx >= 0) & (idx < signal.shape[0]), axis=0)
    idx = idx[:, valid]
    out = signal[idx.ravel(order="F"), :]  # column-major to match MATLAB
    out = out.reshape(epoind.size, idx.shape[1], signal.shape[1], order="F")
    return (out, valid) if return_valid else out
