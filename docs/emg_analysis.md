# Hand EMG analysis — the `emgcasting` half of NeuroCasting

This document covers NeuroCasting's **EMG analysis** tab, the `emgcasting`
package and `run_emg_cli.py`. The EEG transfer score that shares the same
Start tab is documented in
[`eeg_transfer_score.md`](eeg_transfer_score.md);
[`../README.md`](../README.md) describes the merged application as a whole.

Every recording produces two mean envelopes, one per bipolar hand derivation,
and a share of trials carrying high EMG in each hand. The purpose is screening:
overt runs are expected to be full of hand EMG, and quasi-movement or imagery
runs are not, so a high share there says that recording's EEG is contaminated
by real movement.

Outputs never go into the participant data directory. Saving writes below
`output/<participant>/`, together with `emg_summary.csv`. A new participant name
therefore creates a new project-local output folder.

## Run it

```powershell
pip install -r requirements.txt
python app.py
```

On Windows, `launch_NeuroCasting.bat` provides the same launch action.

**START on the Start tab runs both analyses**: the hand EMG screening first,
then the EEG transfer score on the same recordings, using the same participant
folder, the same three filename lists and the same marker time shift. EMG goes
first because it finishes in a fraction of the time the CSP fit needs, and its
tab is raised as soon as it is done. The two are independent — a failure in one
is reported without withholding the other's result — and there is no separate
EMG run button.

The EMG tab therefore has no input fields of its own. What it needs beyond the
EEG fields is on the Start tab:

- **left hand EMG (+, −)**: `Aux 1.1, Aux 1.2`
- **right hand EMG (+, −)**: `Aux 2.1, Aux 2.2`

The hand mapping is based on the actual overt recordings: Aux pair 1 activates
strongly during left-labelled events and Aux pair 2 during right-labelled ones.
Channel numbers (1-based) are accepted where a file carries no labels. Event
names are not asked for: `left_microrepeat`, `right_microrepeat` and `rest` are
the study's own, and are written down once in `app.py` for both halves.

Everything else — filters, envelope, the detector's multipliers, the rest
trims, the output root — is read from `neurocasting_settings.json` next to
`app.py`, whose full path is shown on the Start tab. Every EMG key there
carries an `emg_` prefix.

## What the tab shows

One card per recording, in the order the three filename lists give:

- both mean envelopes side by side, left hand and right hand;
- under each, the share of trials the detector called high — `9/12 high EMG =
  75%` — and, only when there is one, the caveat on that share: that it is a
  known underestimate, or that the hand could not be scored at all;
- a button per hand opening every trial of that hand.

The false-positive floor the same detector produced on that recording's own
rest is measured but deliberately not shown here: beside the share it reads as
a second result rather than as a property of the analysis. It is in
`emg_summary.csv` (`rest_false_positive_percent`) and in the contamination table
`run_emg_cli.py` prints, for a post-hoc look.

Progress and the per-recording timing appear on the status line while the run
is going; the contamination table for the whole participant is printed by
`run_emg_cli.py`.

**Nothing is written until you press Save figures + CSV.** The analysis is
complete before that: the button turns the result on screen into the file tree
described under [Output layout](#output-layout), several hundred per-trial
figures included. Analysing and saving are separated because the screening
question is usually answered on screen in seconds, and writing the figures is
the slow part.

### The trial browser

**Show N left-hand trials…** opens every trial of that recording and hand at
once: a grid of thumbnails on the left, the selected trial enlarged on the
right. A pink thumbnail is a trial classified as high EMG, so the overall
picture — how many, and whether they cluster — is readable before any trial is
opened.

Clicking a thumbnail, the arrow keys, or the previous/next buttons enlarge one
trial into the right-hand panel, which carries matplotlib's own toolbar: zoom
to a rectangle, pan, and go back to the full view. Under it is the evidence
behind that trial's verdict — which branch passed, the peak as a multiple of
rest, the width at the shoulder, the trial's own pre-movement reference and the
adaptive branch's numbers — beside the bar each had to clear. The window is not
modal, so several hands can be compared side by side.

Thumbnails and enlargement are drawn by the same function, from the same
arrays, as the figures Save writes: the screen and a saved file cannot show a
trial differently.

## Choosing a folder fills in the rest

Setting the **folder template** or **subj name** on the Start tab reads the
folder they resolve to and fills in the three filename lists. They stay fully
editable afterwards, and the EMG analysis takes all three, overt first, then
quasi, then imagery.

Each list is in natural run order (`om2` before `om10`). The condition comes
from the filename prefix; every spelling the study has used is recognised —
`om`/`origmov`/`full` for overt, `qm`/`quasi` for quasi, `im`/`mi` for imagery —
and a prefix only counts when a run number follows it, so `omitted_run.xdf` is
not mistaken for an overt recording. A name that matches nothing is reported
below the lists and left out rather than filed under a guess.

Detection runs when the resolved path *changes*, so a hand-edited list survives
clicking between fields. **detect from folder** re-reads on demand. If a folder
holds no recognisable recordings the existing lists are kept, since that is much
more often a half-typed path than an empty session.

The **subj name** is also the EMG output folder name. Note that a per-day
session folder is then not part of the output path, so processing `D1` and `D2`
of one participant writes into the same `output/008TST/` folder — change the
field if you want them kept apart.

`mi1.xdf` in the 002TEST session is incomplete; the intended first imagery
recording is `mi1_finished.xdf`.

## Timing

Timing is read from the event markers embedded in the XDF recording. Each
`left_microrepeat`, `right_microrepeat`, and `rest` duration is inferred from
the next embedded marker. A **marker time shift** is then applied equally to
all task and rest onsets (durations are unchanged).

This is the only supported source of timing, deliberately. An external schedule
has to be anchored to the start of the video, and LabRecorder begins a few
seconds before it — by 5.5 to 7.2 s across the 002TEST runs, differing per file.
Treating the first sample as schedule zero silently shifts every epoch by that
amount.

### The shift is measured, not typed

**marker time shift → detect per recording** on the Start tab is ticked by
default, and the same control governs the EEG half: one run has one time base.
Each recording is then corrected by its own measured shift; the value used is
named on that recording's card, and saved in `emg_summary.csv` as
`marker_shift_s`.

The NeoRec/NVX outlet back-dates its sample timestamps by roughly five seconds:
a sample stamped `S` was really acquired at about `S + lag`. PsychoPy's markers
are flip-locked and correct, so raw marker times cut every epoch about five
seconds too late. The amount is not the same in every file — across the 29
recordings in `data/new` it ranges from 4.82 s to 5.65 s — so no single typed
constant fits a study, and the historical −4.0 s default is about a second short
of every one of them.

`shared/marker_shift.py` measures it from the recording itself. LabRecorder
writes stream chunks to the XDF in arrival order, so the newest sample already
committed to the file when a marker chunk is written is the newest sample that
had arrived by that marker's wall-clock instant; the gap between the two
timestamps is the lag, sampled once per marker. Sixty-odd markers give a median
with a spread of 0.03–0.15 s. It costs about ten milliseconds per file, because
only chunk headers are read and every payload is skipped.

What the median cannot separate out is the real transport latency and half a
chunk of write granularity, together a few tenths of a second. Treat the result
as good to about ±0.3 s.

Untick the box to type one shift and apply it to every recording (CLI:
`--no-auto-marker-shift --marker-shift -5.1`). The measurement still runs and is
reported next to the typed value, so an override can be compared against what
the recording says. If a recording cannot be measured while detection is on, the
run stops and says so rather than falling back to a default silently.

### Why the corrected shift can be trusted

The stimulus video shows a hand pictogram — the movement cue — two seconds after
every `*_microrepeat` marker, alternating with a neutral dot. Two independent
checks agree:

- In the overt runs whose EMG is strong enough to cross-correlate against the
  microrepeat schedule, the detected shift puts the EMG burst 1.9–3.0 s after
  the corrected marker, i.e. on the pictogram, in every case.
- The occipital evoked response to the same pictogram lands in the same place.

Uncorrected, both fall about 2.5 s *before* the marker — before the block
instruction is even on screen. This remains an estimate of the outlet's
timestamp back-dating cross-checked against physiology, not a photodiode
measurement of the physical acquisition offset.

The mean-EMG figures show the effect directly: with the detected shift the
envelope is flat through the shaded 0–2 s preparation interval and fires inside
the 2–4 s cue interval, where the paradigm puts the movement.

### One consequence worth revisiting

`Trim end of rest blocks` defaults to 2 s, and the rest baseline takes no trim
at its start. Both were calibrated when the shift was a second late, which
happened to keep the previous block's movement tail out of the rest window. With
the shift corrected, the rest window starts at the true rest onset and can catch
that tail again. The defaults are unchanged here — it is a scientific choice —
but `Trim start of rest blocks` and the rest-block trims are worth re-checking
against the corrected alignment.

## Signal processing

The defaults reproduce the existing `../emg_om_rms.py` reference pipeline:

1. input volts are converted to millivolts;
2. causal 50 Hz IIR notch, 1 Hz width;
3. causal 20–150 Hz Butterworth band-pass, order 4 prototype;
4. continuous filtering before epoching;
5. discrete Teager–Kaiser energy operator (TKEO);
6. mean TKEO energy in 100 ms windows stepped every 20 ms;
7. event-locked median and 25–75th percentile, plus the resting reference.

Plain RMS can be selected instead of TKEO. Filter limits, envelope grid and
pre/post plotting intervals are `emg_`-prefixed keys in
`neurocasting_settings.json`, and flags on `run_emg_cli.py`.

## Reading the figures

The figures carry no legend. The encoding is fixed:

| mark | meaning |
|---|---|
| solid red | across-trial **median** |
| red shading | **25–75th percentile** across trials |
| dotted black | across-trial **mean** — the value `movement_rest_ratio` uses |
| dashed green | `rest_baseline` |
| darker first 2 s of the grey/pink span | pre-cue part of the motor trial |
| lighter remainder of the grey/pink span | motor cue is visible |
| narrow hatched span after the scheduled trial | 200 ms classification tail |

The band is deliberately not a mean ± SEM. This envelope is an energy and its
across-trial distribution is strongly right-skewed: at the peak of `quasi2`'s
right hand, 96% of the summed energy comes from a single trial, and in both
overt right-hand runs the 25th percentile sits at the noise floor while the
mean is well above it. A mean-centred interval draws a tight ribbon straight
through that, whereas the quartiles show how many repeats actually carry the
response. Where median and mean diverge sharply, the response is intermittent
across trials rather than uniformly present.

**The y-axis is scaled to the quartile band, not to the mean.** The mean is
allowed to run off the top of the frame, because on a recording where one
artefactual trial dominates it would otherwise compress the band into a flat
line and hide the distribution — which is the whole point of the figure.

## Output layout

Everything is written inside the application folder. `emg_output_root` is a
path *relative to that folder* (`output` by default), not to whatever
directory the program was started from, so a copied or forwarded installation
keeps its results inside itself rather than sending them back to the machine
the settings file was written on. An absolute root is still obeyed as given,
for a site that wants results on a shared drive.

```
<application folder>/output/<participant>/        (written by Save, or the CLI)
    emg_summary.csv
    <recording>_<hand>_hand_<envelope>.png        one per recording and hand
    trials/
        <recording>/
            <hand>/
                trial_01.png … trial_20.png       one per event marker
```

Each trial figure is scaled to the shaded scheduled trial plus its narrow
hatched classification tail — the same span the high/low decision was taken
on, so the y-axis shows the evidence the classifier actually used. The
pre-onset lead-in and signal after that tolerance are still drawn, but are
deliberately allowed to run off the top: letting them set the scale flattens
the trial itself into a line. On 002TEST that happens on 30 of 240 trials,
four of them scaled to something 18–34x the in-trial peak.

Amplitudes are therefore **not comparable between trial figures by eye**. The
primary peak bar is a purple dash-dot line; the adaptive width shoulder is blue
and dotted, and its quiet pre-movement reference interval is faint blue. A light
pink background marks a trial classified as high EMG. The across-trial median
remains behind each trace in grey. The accompanying `trial_metrics.csv` records
the detailed classifier evidence in a machine-readable form.

Activity outside the shaded and hatched span plays no part in the verdict.
Both scaling and classification use the half-open span
`[onset, onset + duration + trial_tail_s)`. The scheduled duration remains
unchanged in the summary; `classification_tail_ms` records the extra tolerance.

## Detecting high-EMG trials

The purpose is screening: measure how contaminated each condition is, and use
that to decide whether a participant continues to the second session.

A trial is high when either of two branches passes:

    primary:  peak >= 7 x recording rest
              AND >= 50 ms wide at 6 x recording rest

    adaptive: take the trial median from +0.5 through +1.8 s
              then search only from +1.8 s through trial end + tail
              peak >= 3 x that trial median
              AND >= 50 ms wide at 3 x that trial median

The branches are joined by **OR**, but every requirement inside a branch is
joined by **AND**. Thus the adaptive rule can recover a distinct peak in an
unusually quiet trial independently of whether it reaches the rest bar. The
+0.5 to +1.8 s interval avoids the initial cue response and the final 200 ms
before the movement cue, where a premature reaction can occur.

The classifier appends `trial_tail_s` (200 ms by default) after the scheduled
trial. This prevents a burst crossing the nominal endpoint from being split
into two fragments shorter than the width requirement. Matched rest windows
use the same extended duration.

The background is the **median** of that recording's clean rest envelope;
threshold calibration excludes the first 1 s and final 2 s of each rest block
by default.

Within either branch, height carries the decision; the width bar only rejects
excursions too narrow for the envelope to resolve as events at all. The
adaptive width shoulder is tied to the trial's own quiet reference, so
elevated preparation activity can only make that branch stricter.

**Nothing is shared between recordings.** The multipliers are study constants.
The primary branch uses only that recording's rest; the adaptive branch uses
only that trial's preparation interval.

The median matters. It tracks the background level; a high percentile of rest
tracks rest's *upper tail* and is therefore dragged upward by any peaks rest
happens to contain — the very thing being detected. On 002TEST the rest
background moves by 1.7x across the six runs while the 99th percentile of the
same rest moves by 10x.

### The rest floor, and when a run cannot be scored

The same two-branch detector is run over matched rest windows, including the
local +0.5 to +1.8 s reference inside each pseudo-trial. `emg_summary.csv`
reports the union as `rest_false_positive_percent` and also reports each
branch separately. **Read the trial figure against that floor**, not against
zero.

A run whose own rest fires above `rest_fpr_warn` (5% by default) is flagged in
`rest_warning` and in the console report. Its rest contains peaks, so its
threshold has been lifted by them and its trial count is an underestimate — for
a go/no-go decision that has to read as "cannot be scored", not as a low
number.

### Choosing the numbers

These were set on all 40 recordings of the seven participants, scored against
two references: **overt movement** as the positive control (it must come out
near 100%) and the **matched rest windows** as the floor (it should come out
near 0%).

The primary `peak_multiplier` (7) retains a conservative recording-wide test,
and the headroom is large.
The tallest sample of an ordinary rest window sits at **2x** its own background
at the median and 3.2x at the upper quartile; an overt-movement trial reaches
**1700x** at the median and 86x at its 10th percentile. Two orders of magnitude
separate the two, so the bar is not delicate — but it does have to be placed
inside the gap rather than under it. The previous 3.5x sat at roughly the
**77th percentile of resting noise**, which is why the rule was firing on rest.

The primary bar was first revised from 15x to 9x after visual review supplied 82
positive trials, 35 of which the 15x rule missed. A second review identified
additional 7–8x peaks and a burst cut by the scheduled endpoint, motivating
the 7x rule and an initial 100 ms tail. The classification tail is now 200 ms.
A further review motivated the adaptive preparation-relative branch for
distinct peaks in very quiet trials; the current experiment sets that peak bar
to 3x preparation. At the previously benchmarked settings, the combined rule
kept the overt positive control at 99.4% and put the matched-rest floor at
14.8%.

`background_multiplier` (6) is no longer a detection threshold. It only says
where a peak starts and stops, so that the width can be measured, and where the
`active time` percentage is counted from.

`min_burst_ms` (50) **cannot be read as a physiological width**, and is not
meant to be one. The envelope window is 100 ms wide, while widths are sampled
every 20 ms. Consequently the 50 ms cutoff requires three above-threshold
samples, corresponding to a reported width of at least 60 ms. The deliberately
short cutoff admits the brief repeated micro-movements produced by this task.

The width is measured on the same span `burst_energy` is measured on; both are
recorded in `trial_metrics.csv`.

### What the change bought

Against the previous rule (a 3.5x peak, `>=150 ms` **or** `>=800` background x
ms), over the same 40 recordings:

Across the 80 hand-runs, with the former 15x rule and the 20x variant shown for
reference:

| | overt (control) | quasi | imagery | rest floor |
|---|---|---|---|---|
| before (3.5x, width **or** energy) | 98.9% | 46.0% | 14.4% | 13.6% |
| former (15x and 100 ms) | 98.8% | 36.1% | 7.5% | 8.0% |
| former (9x and 100 ms, no tail) | 99.0% | 42.7% | 11.7% | 10.3% |
| primary only (7x, 100 ms width, +100 ms tail) | 99.0% | 45.2% | 12.7% | 11.3% |
| 5x pre-movement adaptive | 99.4% | 52.6% | 16.5% | 13.0% |
| combined rule at the former 100 ms width | 99.4% | 55.1% | 19.2% | 14.8% |
| (20x and 100 ms) | 98.0% | 32.6% | 6.7% | 6.8% |

Against the original rule, the positive control rises by 0.5 points but the
false-positive floor rises by 1.2 points. Relative to the primary-only 7x rule,
the 3x adaptive branch gains 9.9 points in quasi and 6.5 points in imagery for
3.5 points of matched-rest floor; 61 of 80 hand-runs exceed `rest_fpr_warn`.
Relative to the 5x preparation experiment, 3x adds 15 quasi and 13 imagery
trials without gaining another overt trial, while raising the rest floor by
1.8 points.

These cohort figures predate the current 50 ms width cutoff and 200 ms tail and
should not be treated as performance estimates for the current classifier.

Activity between onset and +1.8 s is intentionally unavailable to the adaptive
branch, because this is the interval used to establish trial quietness and can
contain cue reactions or premature movement. It can still qualify through the
primary 7x-rest branch. A lower adaptive search boundary would make the signal
define its own baseline and then classify itself, defeating the safeguard.

Peaks separated by a gap shorter than one `window_ms` are measured as **one**
burst. The envelope is a moving window that wide, so it cannot resolve them as
separate events in the first place, and splitting a doublet halves the energy
of something that is physically one contraction. There is no separate setting
for this; it follows the envelope window.

Six recordings produce 240 trial figures in about 35 s; untick **Per-trial
figures** (or pass `--no-trial-figures`) to skip them.

## Resting reference

`rest_baseline` is the median of the per-block mean envelope, after the last
`Trim end of rest blocks` seconds (2 s by default) are dropped from each rest
window.

Both departures from a plain pooled mean are needed. Movement starts about a
second before the task-block cue, so the tail of every rest window already
carries the next block; since TKEO is an energy, including it lifted the
reference by 21x to 72x on the overt runs. And a mean over rest is set by
whichever single block holds the largest artefact — one rest block in
`origmov1` sits 213x above the block median, and on that basis `mi1_finished`
reported imagery *below* rest.

Note that this is a recording-level reference, not a per-trial baseline, and
the epochs themselves are not baseline-corrected. A pre-trial baseline is not
available in this design: micro-repeats run back-to-back inside a 16 s block,
so for 15 of 20 trials the pre-onset window is the preceding micro-repeat, and
for the other 5 it is the task-block cue, during which movement has already
begun.

## CLI

```powershell
python run_emg_cli.py `
  --folder "D:\ExpData\MEG\Quasi fMRI\data\002TEST" `
  --files "origmov1,origmov2,quasi1,quasi2,mi1_finished,mi2" `
  --participant 002TEST
```

Unlike the application, the CLI analyses and writes in one call. Use
`python run_emg_cli.py --help` for channel, filter, baseline and output
options; `run_cli.py` is the EEG transfer-score runner.

XDF is the only supported format: it carries both the embedded markers and the
named channels the pipeline needs.
