# EEG overt-template transfer score — the `klh` half of NeuroCasting

This document covers the EEG analysis: the **EEG Components** and
**EEG Score** tabs, the `klh` package and `run_cli.py`. The hand-EMG
analysis, which shares the same Start tab and the same recordings, is
documented in [`emg_analysis.md`](emg_analysis.md);
[`../README.md`](../README.md) describes the merged application as a whole.

This half is a versioned extension of the Python port of the MATLAB app
**KLH_100.mlapp** (“KITTIES' LIL HELPER”). It analyses motor EEG recorded during
overt movement, quasi-movement and motor imagery. The original filter-bank CSP
and 15-point KLH calculation remain available for reproducibility, while the
revised data path fixes sampling, marker-boundary and reporting problems and
adds a separate sensorimotor quality-control view.

Original MATLAB algorithm by **A.N. Vasilyev (2020)**; superlet transform by
Harald Barzan; app “KITTIES' LIL HELPER” (KLH_100).

## Read this first: there are two different outputs

The program deliberately keeps two questions separate:

| Output | Question answered | Interpretation |
|---|---|---|
| **Overt-template transfer score** | Does a CSP component learned from overt movement retain the historical pattern and transfer to quasi-movement and imagery? | A versioned 15-point KLH score. It is **not** a general sensorimotor-data-quality rating. |
| **Physiological QC** | Does each condition show plausible sensor-space mu/beta ERD and the expected contralateral distribution? | C3/C4 and contralateral-versus-ipsilateral diagnostics. It contributes no points to the KLH score. |

The distinction matters for data such as `002TEST`: credible contralateral
imagery ERD can coexist with a low score for transfer of an overt-derived CSP
component.

The 15 points are also reported as two parts: criterion 1 is a **subject-level
eigenvalue-spectrum score (0–2)**, while criteria 2–6 form the **selected
component transfer score (0–13)**. These are a decomposition of one total, not
two competing ratings: `total transfer score = ERD-candidate spectrum +
selected-component score`.

## Implemented defaults

These are the defaults on the CLI and in a freshly created `neurocasting_settings.json`.
An edited settings file can differ; the GUI also remembers the Start fields
across sessions.

| Setting | Default | Status |
|---|---:|---|
| Score formula | `scoring-v4-rolling-peak-independent-nonreal-erd` | Uses a strongest-continuous 500 ms rolling-median peak and independently scores sustained quasi and imagery mu-ERD; all cutoffs remain provisional and not cohort-calibrated. |
| EEG processing rate | 1000 Hz | Required by the stored 1000-Hz FIR coefficients. |
| XDF marker time shift | detected per recording | Corrects the amplifier's back-dated sample timestamps before task timing is interpreted. Measured from each recording (-4.8 to -5.7 s across the recordings in `data/new`); untick to type one value instead. |
| Active preparation exclusion | 2.0 s | The first two seconds after the corrected microrepeat onset are preparation and do not enter the active core or score. |
| Motor recovery | 2.0 s | Continuous EEG after the marker-defined trial is retained so the TF map and peak-ERD score include post-movement recovery. |
| XDF marker baseline | Condition-specific rest | Each overt/quasi/imagery condition uses its own rest recording. |
| Rest-transition exclusion | 1.0 s | Removes the nonstationary start of every marker rest event; no direction of bias is assumed. |
| Marker TF context | 1.5 s on each side | Real neighbouring samples protect both active and rest TF cores from artificial epoch boundaries. |
| Candidate spectrum | Raw CSP eigenvalues | Preserves the historical scale; montage cutoffs now scale with channel count. |
| CSP covariance shrinkage | 0 | Exact historical covariance behaviour; non-zero shrinkage is optional. |
| Overt CV folds | 5 | Overt peak ERD is measured on held-out trials, so criteria 3 and 4 are not in-sample. Set 0 or 1 for the historical in-sample estimate. |

For H5/trigger data, marker baseline, transition and padding controls do not
apply. H5 uses the condition's own prestimulus interval.

## Analysis pipeline

1. Load H5 data with a categorical trigger channel, or XDF data with embedded
   PsychoPy markers.
2. Detect events in their native time base, then process continuous EEG at
   1000 Hz. Categorical triggers are never resampled.
3. Apply the shipped broadband (3–43 Hz) and alpha (9–14.5 Hz) FIR filters from
   `assets/helper.npz`.
4. Epoch rest and active data, recording requested, eligible, retained and
   dropped trial counts.
5. Estimate Olive–Hawkins robust covariances. CSP is still an **overt
   template**: its active covariance comes from overt trials (0.5–2.0 s), while
   rest alpha epochs (0.5–4.0 s) are pooled across the three conditions.
6. Solve CSP, report covariance-pencil conditioning, and find ERD/ERS
   candidates from eigenvalue gaps. Zero candidates are reported as zero while
   component 1 remains inspectable.
7. Display up to `min(5, n_components)` forward patterns and preselect one of
   them on the slider (see [First-pass component
   selection](#first-pass-component-selection)). A transparent
   central-versus-peripheral/topographic ranking is reported alongside it, and
   the preselection is always overridable by hand.
8. Project the selected component, compute superlet TF maps for overt, quasi
   and imagery, normalize them to the selected baseline, and calculate the
   requested version of the transfer score.
9. Separately calculate sensor-space physiological QC and report persistent
   channel outliers, retained trials and full analysis provenance.

The fourth tab is the hand-EMG screening described in
[`emg_analysis.md`](emg_analysis.md). It runs on the same recordings from the
same START button and contributes no points to the transfer score; a
contaminated quasi or imagery run is a reason to distrust that recording's
score, not a term inside it. (It replaces the single-file parieto-occipital
alpha screening of the former KitEX tab, whose alpha peak was in any case never
reused as an individualized sensorimotor-mu band.)

## Score versions

### `legacy-v1`

`legacy-v1` preserves the historical score arithmetic and thresholds,
including sign-sensitive positional pattern correlation, the ungated fourth
criterion, and the old ratio between dB values in criterion 6. It exists for
archived-result and MATLAB/cohort comparisons.

Selecting `legacy-v1` changes the **score formula only**. It does not undo EEG
resampling, marker padding, baseline selection, label matching or another
revised data-path decision. The GUI is locked to v4, so this selection is made
with `run_cli.py --score-version`. Therefore:

- A 64-channel, 1000-Hz H5/trigger analysis with raw candidates and zero
  shrinkage is the closest supported route to the historical MATLAB path.
- An XDF/marker result is not made MATLAB- or cohort-compatible merely by
  choosing `legacy-v1`; its data-path identifier must accompany the result.

### `scoring-v2` (retained, provisional thresholds)

`scoring-v2` implements the reconciled corrections:

- pattern similarity is channel-label aligned, safe for missing reference
  channels, and invariant to arbitrary CSP polarity;
- constant/invalid patterns and unsafe display ratios return unavailable
  diagnostics rather than confident infinities;
- criterion 4 is scored only when overt ERD is positive; criteria 5 and 6 are
  intentionally not gated on positive quasi ERD;
- imagery-versus-quasi criteria use direct dB differences rather than ratios
  between logarithmic quantities; and
- no significant ERD candidate produces `n_erd = 0`, not a fabricated
  one-component count.

The shipped threshold profile is named `provisional-minus4db-v1`. Its dB
difference cutoffs are a documented mapping around a representative −4 dB
quasi response, **not a re-fit of the reference cohort**. Use v2 for corrected
arithmetic and explicit provenance, but do not yet interpret its total as a
validated replacement cohort scale.

Trigger-mode scoring retains the historical −0.25–2.5 s peak-ERD window.
Marker-mode time zero is motor-cue onset after preparation has been excluded.
Its peak-ERD score uses the complete non-negative active core: normally 0–4 s,
comprising 2 s of motor task and 2 s of recovery. The mean 9–13 Hz comparison
and CSP covariance remain fixed at 0.5–2.0 s so they measure motor execution.

### `scoring-v3-independent-nonreal-erd` (retained, provisional thresholds)

`scoring-v3-independent-nonreal-erd` keeps v2 criteria 1–4 unchanged and
replaces only the six imagery-versus-quasi points:

- criterion 5 independently scores sustained quasi-movement mu-ERD;
- criterion 6 independently scores sustained motor-imagery mu-ERD; and
- neither criterion depends on which condition is stronger or on the sign of
  an imagery-minus-quasi difference.

For each condition, sustained ERD is the negative median of the selected
overt-derived component's normalized TF map over 9–13 Hz and 0.5–2.0 s.
Magnitude `<= 1` dB earns 0 points, `> 1` earns 1, `> 2` earns 2, and `> 3`
earns 3. The selected component is never reselected after inspecting quasi or
imagery. The profile `provisional-independent-mu-erd-v1` is explicitly
uncalibrated; its total must not be used as a cohort norm until the cutoffs are
calibrated and validated on recordings processed by this pipeline.

The score and JSON report also retain the old peak and mean-region
imagery-minus-quasi differences as non-scoring diagnostics. V3 additionally
reports the frequency/time windows, temporal ERD coverage, a deterministic
trial-bootstrap 95% interval with the scoring baseline held fixed, and the
number/fraction of retained recordings whose component-level sustained ERD is
positive. These diagnostics are reported only and do not gate the provisional
points. Separate common-average sensor-space QC remains visible and contributes
no points.

V3 remains available for reproduction of results whose criteria 3 and 4 used
the historical negative third percentile.

### `scoring-v4-rolling-peak-independent-nonreal-erd` (public default)

V4 preserves v3's independent sustained quasi/imagery estimator but uses its
own current magnitude thresholds and replaces the peak statistic used by
criteria 3 and 4. For each condition it:

1. takes the median normalized dB change across 5--16 Hz at every time point;
2. takes a rolling median over every complete continuous 500 ms window; and
3. reports the most negative rolling value as a positive ERD magnitude.

For marker recordings, candidate windows cover the complete non-negative
active core (normally 0--4 s, including configured recovery). Trigger-mode
windows retain the historical -0.25--2.5 s score interval. The report records
the method, frequency band, rolling duration, and winning interval separately
for overt, quasi and imagery.

The profile `provisional-rolling500ms-independent-mu-erd-v3` uses provisional
rolling-peak boundaries of >1, >3 and >5 dB for 1, 2 and 3 points. These are
approximately 21%, 50% and 68% power reductions. Independently for sustained
quasi and imagery ERD, >2, >3 and >4 dB earn 1, 2 and 3 points. The 40/60/75%
relative-transfer cutoffs are retained. These boundaries have not been
cohort-calibrated, so v4 totals are not directly comparable with v3 totals or
a reference cohort.

The GUI is locked to `scoring-v4-rolling-peak-independent-nonreal-erd` and the
CLI defaults to it. The lower-level
`klh.scoring.compute_scores()` dispatcher defaults to `legacy-v1` only to avoid
silently changing third-party callers; pipeline code passes a version
explicitly.

### Cross-validated overt projection

The CSP filter is fitted on overt movement versus rest. Projecting the same
overt trials back through that filter makes criterion 3 an **in-sample**
statistic while quasi and imagery remain out-of-sample, which inflates the
overt reference and biases the criterion-4 transfer ratio low.

With `--csp-cv-folds N` (default 5, `overt_cv_folds` in `neurocasting_settings.json`) the overt map
is built only from held-out trials. Each fold refits the CSP with its share of
the overt active *and* overt rest epochs removed, re-identifies the operator's
component by absolute pattern correlation, and rescales the filter to unit
variance on its own training rest so held-out projections from different folds
can be pooled. Every trial is transformed exactly once, so the overt median
keeps its full trial count — thinning it would raise the overt noise floor
relative to quasi and imagery and re-bias the ratio in the other direction.

Two limits are reported rather than assumed away. The component's *identity*
is still chosen on the full data — that is the component the operator picked —
so only the filter weights are held out. And quasi/imagery rest epochs stay in
the CSP training pool for every fold, leaving a second-order leak under a
pooled baseline. Both appear in `csp_cv` in the JSON report, alongside the
per-fold match similarity and the margin over the runner-up component; a fold
that matched ambiguously raises a warning instead of being silently averaged in.

On band-limited noise the change removes most of the measurable optimism, but
this has not yet been quantified on the reference cohort.

## Sampling and epoch data paths

### H5 with native trigger detection

H5 input is expected to contain the first 2–64 channels of the KLH reference
montage plus a trigger column. Values above 64 are rejected because the shipped
reference pattern and scalp layout contain 64 EEG electrodes.

`--fs` is the **native recording rate**. The trigger is decoded and detected at
that native rate, its onsets are mapped through seconds to processing indices,
and only continuous EEG is polyphase-resampled to 1000 Hz. This prevents the
ringing and false transitions that would result from resampling a categorical
trigger channel. `--processing-fs` is exposed for provenance and validation,
but currently must remain 1000 Hz because the stored FIR filters are fixed-rate.

Trigger epochs retain the historical −2.5–5.5 s window. TF normalization is
condition-specific and uses the selected active trials' −1.5–−1.0 s
prestimulus segment.

### XDF with embedded markers

XDF timing comes from the embedded marker stream. An event's raw onset is its
timestamp relative to the first EEG sample, and duration is the gap to the next
marker. A **marker time shift** is applied equally to active and rest onsets;
durations are unchanged. pyxdf clock correction remains disabled; `video_onset`
is reported on the corrected time axis.

**Marker time shift → detect per recording** is ticked by default, and each
recording is corrected by its own measured shift.

The NeoRec/NVX outlet back-dates its sample timestamps by roughly five seconds:
a sample stamped `S` was really acquired at about `S + lag`. PsychoPy's markers
are flip-locked and correct, so raw marker times cut every epoch about five
seconds too late. The amount differs per file — 4.82 s to 5.65 s across the 29
recordings in `data/new` — so no single typed constant fits a study, and the
historical −4.0 s default is about a second short of all of them.

`shared/marker_shift.py` measures it from the recording itself. LabRecorder writes
stream chunks to the XDF in arrival order, so the newest sample already
committed to the file when a marker chunk is written is the newest sample that
had arrived by that marker's wall-clock instant; the gap between the two
timestamps is the lag, sampled once per marker. Sixty-odd markers give a median
with a spread of 0.03–0.15 s, in about ten milliseconds per file, because only
chunk headers are read and every payload is skipped. What the median cannot
separate out is the real transport latency and half a chunk of write
granularity, together a few tenths of a second, so treat the result as good to
about ±0.3 s.

Untick the box to type one shift for every recording (CLI:
`--no-auto-marker-shift --marker-shift -5.1`); the measurement still runs and is
reported beside the typed value. A recording that cannot be measured while
detection is on stops the run rather than falling back to a default silently.

Detected shifts are recorded per file in provenance under
`recordings[].marker_time_shift_s`, and `marker_shift_mode` says which route was
used. The top-level `marker_time_shift_s` carries a number only when one typed
value was applied to everything, and is `null` when shifts were detected. The
per-file values enter the `data_path_version` digest, so runs over recordings
with different lags cannot share a data-path identifier.

On `008TST/D1` the correction is worth about 2 dB of sensorimotor ERD. Against a
typed −4.0 s, detection moves overt mu from −1.6/−1.9 dB (C3/C4) to −3.7/−3.5 dB
and overt beta from −0.4/−0.9 dB to −4.3/−4.1 dB; quasi beta flips from a
spurious +0.3/+0.8 dB ERS to −2.9/−1.7 dB ERD. The uncorrected epochs were
landing on the preparation interval rather than on the movement.

Independent evidence that the corrected alignment is the right one: the stimulus
video shows the hand pictogram two seconds after each `*_microrepeat` marker,
and with the detected shift both the overt EMG burst and the occipital evoked
response to that pictogram land there. This remains an estimate of the outlet's
back-dating cross-checked against physiology, not a photodiode measurement of
the physical acquisition offset.

The native EEG rate is read per file and EEG is resampled to 1000 Hz. Channels
are matched by label to the available portion of the 64-channel reference
montage; auxiliary inputs are excluded and an omitted reference electrode such
as `Cz` is handled consistently.

For active microrepeats, the first 2 s after the corrected onset are preparation
and are excluded. Active time zero is therefore the motor-cue onset. The next
2 s contain the motor task and another 2 s of continuous EEG are retained after
the marker-defined trial for recovery, even though those samples formally
belong to the following trial. Preparation and recovery are explicit settings
(`preparation_excluded_s` / `motor_recovery_s` in `neurocasting_settings.json`,
`--preparation` / `--motor-recovery` on the CLI); the marker time shift remains
a separate operation. The 2 s preparation matches the stimulus: the movement cue
appears two seconds after the microrepeat marker, and a neutral dot is shown
until then.

Marker active/rest cores are cropped to common eligible lengths across the
supplied recordings so epochs can be stacked. Before epoching, the program also
retains real recorded context on both sides (1.5 s by default). Every core is
therefore separated from a concatenation boundary by at least the required
wavelet support; after the transform, only the core is retained. Events without
a finite duration and windows too close to recording boundaries are counted as
dropped.

The first 1.0 s of each marker rest core is excluded from TF baseline estimates
by default because the rest onset is nonstationary. Condition-specific rest is
the default normalization; pooled rest remains available for explicit
sensitivity analyses. Both the active TF maps and rest baselines receive the
same boundary protection.

## Physiological QC and advisory diagnostics

Physiological QC is computed in common-average sensor space and normalized
within each condition. The current fixed definitions are:

- active QC window: 0.5–2.5 s (motor execution plus the first 0.5 s of recovery);
- marker rest window: from the configured transition exclusion for up to 3 s;
- trigger rest window: −1.5–−1.0 s;
- mu: 8–13 Hz; beta: 15–25 Hz; and
- hand-specific left/right motor ROIs, with C3, C4,
  contralateral-minus-ipsilateral dB and percent-ERD summaries.

Negative dB means active power is below rest (ERD). Results are reported for
overt, quasi and imagery with their active/rest trial counts. In addition to
the pooled condition summary, every run with retained active and rest trials
is reported separately, together with the fraction showing the expected
laterality direction, the median, MAD and range. Runs missing either side are
marked unavailable rather than causing the pooled analysis to fail.

Persistent channel-variance outliers are read-only warnings. They do not
reject channels or alter score points automatically.

## First-pass component selection

When the primary analysis finishes, the slider is moved to the component that
is worth looking at first. Two pieces of evidence exist at that point, and
only two: the bounded CSP eigenvalue, and the forward pattern. This identifies
a candidate for inspection; confirmation of contralateral mu-ERD requires
task-related power reduction relative to rest in the time-frequency response,
with support across trials or recordings.

- The pencil `eig(R1, R1 + R2)` is solved on trace-normalized covariances, so
  0.5 is neutral. Below it the component's share of mu (9–14.5 Hz) power fell
  between rest and the overt active window (**mu-ERD candidate**), above it the
  share rose (**mu-ERS candidate**). The unchanged margin of 0.02 admits ERD
  candidates at eigenvalues `<= 0.48` and ERS candidates at `>= 0.52`; values
  in between are indeterminate. A relative power-share change alone does not
  establish task-related suppression against the TF baseline.
- Each forward pattern is centered by subtracting its mean across the available
  channels before squaring its values. All central, peripheral and motor-ROI
  energy fractions, laterality and ranking use this centered pattern, making
  localization independent of a constant reference offset.
- Centered pattern energy over the hand-specific motor ROIs gives a laterality
  index, `(contra − ipsi) / (contra + ipsi)`. The unchanged threshold is at least
  `+0.10` for contralateral or at most `−0.10` for ipsilateral lateralization.
- Every category also requires a motor-plausible pattern: the central ROI must
  carry at least its share of the montage's channels in centered pattern energy
  (`central energy fraction >= central channel count / total channel count`).
  Peripheral/ocular energy reduces the ranking score; it does not veto a
  candidate. No threshold depends on an individual participant.

Categories are then taken in this order, which is a preference order, not a
ranking of evidence strength:

1. **contralateral mu-ERD candidate** — the target for subsequent TF inspection;
2. **mu-ERS candidate** — an inspection fallback, with contralateral laterality
   contributing to its ranking. Laterality is not required. The five
   inspectable components are the low end of the eigenvalue spectrum, so this
   category is only reachable when one of them nevertheless reaches the ERS
   threshold;
3. **ipsilateral mu-ERD candidate** — an inspection fallback.

Within a category, the plausibility ranking is `central energy fraction −
peripheral energy fraction + 0.5 × absolute ideal-pattern correlation`.
The ideal-pattern correlation keeps its existing weight and channel-label
alignment; an unavailable correlation contributes zero. The selection score
adds half the laterality index (subtracts it for the ipsilateral category),
with ties going to the lower component number.

ERS and ipsilateral fallbacks do not count as detection of the contralateral
mu-ERD target. A bilateral ERD pattern that misses the lateralization threshold
is also excluded from this lateralized preselection; that does not establish
absence of contralateral ERD. Sensor-space QC reports contralateral suppression
and the contralateral-minus-ipsilateral difference separately, so the presence
of suppression and stronger suppression on the contralateral side remain
separate questions.

If no component matches any category, **component 1 is preselected and a
message says so**: automatic selection sees neither the time-frequency
response nor the transfer score, so it can fail, and stepping through the
topographies by hand is still worth doing. The GUI shows the preselection
summary and candidate status, with a reminder to confirm the response in TF.
The slider can always be moved before `Calculate FT`; automatic preselection
does not replace that inspection.

`run_cli.py` is unaffected: it keeps `--component` (default 1) so batch runs
stay explicit and reproducible.

## Install and run

```powershell
python -m pip install -r requirements.txt
python app.py
```

The Start page carries only what changes from subject to subject: folder
template, subject name, the three filename lists, movement trials and the
marker time shift (detected per recording unless the box is unticked, in which
case the spin box beside it becomes editable). Every other analysis parameter is
read from
`neurocasting_settings.json` next to `app.py`; the page shows its full path. The file is
created with the defaults below on first run, and a missing key, an unusable
value or an unknown key is reported on that same line and repaired in place.

Setting the folder template or the subject name fills the three filename lists
from the folder they resolve to (`SUBJECT_NAME` is substituted first). The
condition comes from the filename prefix; every spelling the study has used is
recognised — `om`/`origmov`/`full` for real, `qm`/`quasi` for quasi, `im`/`mi`
for imagery — and a prefix only counts when a run number follows it, so
`omitted_run.xdf` is not mistaken for a real-movement recording. Each list is in
natural run order (`om2` before `om10`), and names matching nothing are reported
under the lists and left out rather than filed under a guess. The extension
follows the configured file type.

The lists stay editable. Detection runs only when the resolved path *changes*,
so a hand-edited list survives clicking between fields and is restored intact on
the next start; **detect from folder** re-reads on demand. A folder holding no
recognisable recordings leaves the lists alone, since that is much more often a
half-typed path than an empty session.

The score formula is locked to `scoring-v4-rolling-peak-independent-nonreal-erd` in this
build. It is written to `neurocasting_settings.json` so a result can be traced back to
the formula that produced it, but editing it there does not change the formula;
the value is corrected on the next start. Use `run_cli.py --score-version` for
an explicitly labelled formula comparison against v3, v2 or legacy-v1.

Normalization, score/data-path versions, retained trials, provenance and
physiological QC are reported by `run_cli.py`; the GUI tabs show the result
itself rather than its provenance.

### GUI example: `002TEST`

Use these fields:

| Start field | Value |
|---|---|
| folder | `D:\ExpData\MEG\Quasi fMRI\data\002TEST` |
| filenames real | `origmov1,origmov2` |
| filenames quasi | `quasi1,quasi2` |
| filenames imag | `mi1_finished,mi2` |
| movement trials | Right hand or Left hand, analysed separately |
| marker time shift | detect per recording (ticked) |

The rest comes from `neurocasting_settings.json`, whose defaults already suit this
recording:

| `neurocasting_settings.json` key | Value |
|---|---|
| `file_type` | `xdf` (LabRecorder) |
| `preparation_excluded_s` / `motor_recovery_s` | 2.0 / 2.0 |
| `marker_baseline` / `skip_rest_transition_s` / `tf_marker_padding_s` | `condition-specific` / 1.0 / 1.5 |
| `candidate_detection_scale` / `csp_shrinkage` | `raw` / 0 |
| `scoring_version` | `scoring-v4-rolling-peak-independent-nonreal-erd` (locked) |

`mi1.xdf` is an incomplete attempt (about 189 seconds and 38 markers); use
`mi1_finished.xdf` and `mi2.xdf` for imagery. Filenames in the GUI and CLI are
comma-separated and omit the extension.

Start the analysis, check the component preselected on the slider against the
scalp maps and the ranking, correct it if the maps disagree, and calculate FT.
Repeat the run with the other hand selected; right and left microrepeat
markers are not combined into one CSP analysis.

### CLI example: `002TEST`

The following records the corrected default choices explicitly and writes both
a figure and a machine-readable report:

```powershell
python run_cli.py `
  --folder "D:\ExpData\MEG\Quasi fMRI\data\002TEST" `
  --file-type xdf --extract markers `
  --real origmov1,origmov2 --quasi quasi1,quasi2 --imag mi1_finished,mi2 `
  --active-cond right_microrepeat --rest-cond rest `
  --marker-shift -4.0 --preparation 2.0 --motor-recovery 2.0 `
  --marker-baseline condition-specific --rest-transition-skip 1.0 `
  --marker-padding 1.5 --candidate-scale raw --csp-shrinkage 0 `
  --score-version v4 --csp-cv-folds 5 --component 2 `
  --out 002TEST_right_component2.png `
  --report-json 002TEST_right_component2.json
```

Component 2 is shown because it was the more plausible right-hand motor
topography in the earlier `002TEST` inspection. The CLI never preselects a
component — the first-pass selection described above is a GUI convenience, and
`--component` stays explicit so a batch run reproduces exactly what it
records. Inspect each new analysis before choosing `--component`. For the
left-hand analysis, rerun with `--active-cond left_microrepeat` and choose the
component supported by that run's maps/ranking.

### Included `002TEST` acceptance result

The copied project includes
The repository retains the pre-v3 reference artifacts
[`002TEST_v2_component2_final.json`](002TEST_v2_component2_final.json) and
[`002TEST_v2_component2_final.png`](002TEST_v2_component2_final.png), produced
with the analogous command using `--score-version v2` for the right-hand marker
and component 2. These totals are labelled v2 and must not be interpreted as
v3 totals. The run used
only `mi1_finished.xdf` and `mi2.xdf` for imagery. It found the same 63 reference
channels in all six files (`Cz` was absent), resampled EEG from 500 to 1000 Hz,
and retained 40/40 active plus 18/18 rest epochs in each condition.

That archived provisional scoring-v2 result was **11/15**: subject spectrum 2/2 and
selected-component transfer 9/13, with sub-scores `[2, 0, 0, 3, 3, 3]`. This is
an overt-template transfer score, not a data-quality grade. The separate pooled
imagery QC showed C3/C4 mu changes of -5.36/-4.09 dB and
contralateral-minus-ipsilateral mu of -0.58 dB. Run-level imagery laterality was
mixed in mu (1/2 runs in the expected direction) but consistent in beta (2/2),
which is more informative than calling the complete recording simply “bad.”
The scoring-v2 thresholds remain provisional until the reference cohort is
re-scored.

### H5 example

For a 500-Hz H5 recording, specify 500 Hz as the recording rate. Trigger
detection remains at 500 Hz and EEG alone is resampled to 1000 Hz:

```powershell
python run_cli.py `
  --folder "..\Casting\RawData" --file-type h5 --extract trigger `
  --fs 500 --processing-fs 1000 --nchan 64 `
  --real om_s1,om_s2 --quasi qm_s1,qm_s2 --imag im_s1,im_s2 `
  --candidate-scale raw --csp-shrinkage 0 --score-version legacy `
  --component 1 --out result.png --report-json result.json
```

Use the actual native rate; do not label 1000-Hz input as 500 Hz or vice versa.

## Reports and provenance

The CLI always prints input files, eigenvalue candidates, covariance
conditioning, retained/dropped trials, channel warnings, advisory component
ranking, the subject/component score split, normalization, score version,
data-path identifier and physiological QC. `--report-json PATH` additionally
writes schema `klh-fmri-quasi-report-v1` with:

- input file lists and selected component;
- the complete score diagnostics and threshold profile;
- for v3, independent quasi/imagery sustained ERD, the fixed windows, temporal
  coverage, trial-bootstrap intervals and per-run component consistency;
- `overt_projection` and `csp_cv`: whether overt peak ERD is a held-out or an
  in-sample estimate, with per-fold component-match diagnostics;
- physiological QC, component ranking and channel QC;
- retained/dropped trial counts; and
- provenance for recording/processing rates, per-file resampling and event
  paths, filter hash, montage, extraction windows, baseline, padding,
  transition exclusion, candidate scale, shrinkage and score formula.

Non-finite diagnostics are serialized as JSON `null`. The deterministic
`klh-data-v4:<path>:<digest>` identifier changes when a behavior-changing
data-path choice changes, so equal score-formula labels cannot conceal
different preprocessing.

## Implemented alternatives and research status

The following options exist, but availability is not the same as scientific
validation:

| Item | Implementation status | Validation claim |
|---|---|---|
| Condition-specific marker rest | Implemented default | Preferred reconciled normalization; provenance records the choice. |
| Pooled marker rest | Implemented option | Sensitivity/compatibility comparison, not the default. |
| Raw eigenvalue candidates | Implemented default | Historical scale with channel-count-proportional cutoffs. |
| Log-odds candidates | Implemented option | Symmetric treatment of ERD/ERS spectrum ends; not permutation-calibrated or cohort-validated. |
| CSP shrinkage | Implemented option, default 0 | Useful for conditioning experiments; no non-zero value is recommended as a validated default. |
| `scoring-v2` thresholds | Implemented provisional profile | Correct arithmetic, but the reference cohort has not yet been re-scored to calibrate cutoffs. |
| `scoring-v3-independent-nonreal-erd` thresholds | Retained provisional archive formula | Requires both quasi and imagery to earn their own sustained-ERD points; criteria 3/4 retain the historical third-percentile peak. |
| `scoring-v4-rolling-peak-independent-nonreal-erd` thresholds | Implemented provisional default | Replaces the extreme peak percentile with a continuous 500 ms rolling median; peak and sustained cutoffs are not cohort-calibrated. |
| Cross-validated overt projection | Implemented default, 5 folds | Removes the in-sample advantage the overt condition had over quasi and imagery. Verified on band-limited synthetic noise; the size of the correction on the reference cohort is not yet measured, and archived in-sample totals are not comparable with cross-validated ones. |
| Physiological QC | Implemented advisory view | A transparent fixed-band QC, not a clinical/behavioral score or cohort norm. |
| First-pass component selection | Implemented in the GUI | Prefers a contralateral mu-ERD candidate using the eigenvalue and centered pattern, with a montage-scaled central-energy floor and peripheral-energy ranking penalty. ERS and ipsilateral candidates are inspection fallbacks only; TF confirmation is still required, and no match does not establish absence of contralateral ERD. |

Not implemented or not claimed complete: blockwise cross-validation with
component matching, block/hierarchical bootstrap uncertainty (v3 currently
reports a fixed-baseline trial bootstrap), permutation-based
candidate significance, cross-validated individualized sensorimotor bands,
using the reported run-consistency diagnostic as a score, and automatic artifact/channel/component
rejection. These remain research items rather than hidden defaults.

## Module map (MATLAB → Python)

| MATLAB / responsibility | Python | Notes |
|---|---|---|
| `loadh5`, trigger, XDF markers and resampling | `klh/io_h5.py` | Native trigger detection; EEG-only polyphase resampling; marker/channel parsing. |
| `get_bandpassFIR` + `filtfilt` | `klh/filters.py` | Reuses stored 1000-Hz coefficients; optional `firpm`-style design. Long inputs take an FFT-convolution path equivalent to `scipy.signal.filtfilt`. |
| `robustcov('olivehawkins')` | `klh/robustcov.py` → `klh/_ohcov.py` | Lab-verified MATLAB-compatible estimator. |
| `calcCSP_cov` | `klh/csp.py` | `eig(A,B)` via `scipy.linalg.eigh`; optional scaled-identity shrinkage. |
| `aslt` | `klh/superlet.py` | Overlap-save FFT convolution against a cached wavelet bank; marker cores protected by real context in the pipeline. |
| SCORES block | `klh/scoring.py` | Explicit archived v1--v3 formulas and provisional rolling-peak independent-ERD `scoring-v4`. |
| Sensorimotor/artifact diagnostics | `klh/qc.py` | C3/C4/ROI QC, component plausibility ranking, first-pass component selection and channel warnings. |
| `topoplot` | `klh/topo.py` | EEGLAB `v4` biharmonic interpolation. |
| `STARTButtonPushed` + `CalculateFT` | `klh/pipeline.py` | Versioned orchestration, trial accounting and provenance. |
| App Designer UI / batch runner | `app.py`, `run_cli.py` | Dynamic component count and GUI reports; CLI JSON export. |
| Recording discovery, amplifier timestamp lag | `shared/recordings.py`, `shared/marker_shift.py` | Shared with the EMG half, so both read one participant's files on one time base. |

Assets exported from `helper_files.mat` live in `assets/helper.npz`
(`b1`, `b2`, `bpa`, `ideal_pattern`, `score_img`, cohort `subj_dB` and
`anatoly_scores_all`), alongside the `mumeg_mks64.ced` layout and app images.

## MATLAB fidelity and caveats

This remains **functionally equivalent where the legacy path is selected**, not
bit-for-bit identical to MATLAB. Known divergences, worst-first:

1. **`robustcov('olivehawkins')`** uses vendored `klh/_ohcov.py`. The lab has
   tested this estimator to give identical output to MATLAB; the wrapper only
   adapts its signature.
2. **`filtfilt`**, floating-point edge handling and optional filter design can
   differ subtly from MATLAB. Shipping the original coefficients limits the
   filter-design difference. Polyphase resampling is an intentional revised
   path when the native rate is not 1000 Hz.

   The stored coefficient sets run to ~2500 taps, for which a time-domain
   `filtfilt` costs O(samples x taps); inputs long enough to pad are therefore
   filtered by FFT convolution instead (`klh/filters.py`), reproducing scipy's
   odd extension and steady-state initial conditions rather than approximating
   them. Likewise `aslt` convolves against a cached wavelet bank in
   overlap-save blocks instead of rebuilding 826 wavelets and re-transforming
   the signal once per wavelet. Both are implementation changes only:
   `tests/test_transform_equivalence.py` vendors the superseded
   implementations and asserts agreement, which is to floating-point rounding
   (~1e-13 relative) because moving a convolution into the frequency domain
   reorders its additions. A full scored run is unaffected, including the
   `data_path_version` digest.

## Cost of a run

Two stages dominate, and both are shaped by decisions above rather than by the
score formula.

**Filtering** is applied per recording, so it scales with recording length
rather than with trial count. `_gather` filters only the span the epochs
actually read, plus a margin of `(b1 - 1) + (b2 - 1)` samples on each side.
Zero-phase filtering makes an output sample depend on inputs within `taps - 1`
either way and the broadband chain applies two filters in series, so that
margin is exactly what makes every epoched sample identical to filtering the
recording whole — a stretch of recording no trial reaches cannot influence one
that does. `tests/test_filter_crop.py` pins both that equality and the fact
that cropping cannot change which trials survive the recording-boundary check.
Each recording's kept span is reported under `recordings[].filter_crop`.

**The time-frequency transform** dominates `compute_ft`. Its frequency axis is
separable, so `klh/superlet.py` can split it across worker processes; the
`superlet_workers` setting (or `--superlet-workers`, or
`KLH_SUPERLET_WORKERS`) controls how many, and 0 or 1 keeps it in one process.
This is a scheduling choice with no effect on the result: every worker is given
the orders and the block geometry of the *whole* spectrum, so the split is
bit-identical rather than merely close, which `tests/test_superlet_workers.py`
asserts with `array_equal`. It is not free — each worker is a fresh interpreter
that re-imports the application, costing roughly 170 MB of resident memory —
and past four workers on a six-core machine the gain reverses, because
hyperthread siblings contend for one FPU. Filtering does not parallelise at
all: it streams far more memory than the cache holds, and is bandwidth-bound
whether cropped or not.
3. **`topoplot`** uses the same `v4` biharmonic interpolation and orientation,
   but head/nose/ear cosmetics are drawn locally and are not pixel-identical.
4. `matlab_prctile`, median, MAD and nearest-index helpers follow MATLAB
   semantics, but floating-point rounding can still flip a borderline strict
   threshold.

Historical H5 scaling is retained: overt/quasi are used as stored and imagery
is multiplied by `1e6`. XDF records all three conditions in volts, so all three
are converted by `1e6`. The raw candidate rule remains
`median(diff) + 3·1.4826·MAD`; log-odds is an explicit alternative. The latent
double-trial branch of `trigDetector` remains guarded and is not used by the
single-trial data path.

## Tests and validation

Install the test runner separately, then run the complete synthetic/unit suite:

```powershell
python -m pip install pytest
python -m pytest -q
```

Useful focused commands are:

```powershell
python -m pytest -q tests/test_smoke.py
python -m pytest -q tests/test_core_datapath_v2.py tests/test_scoring_v2_reconciled.py tests/test_scoring_v3_independent_erd.py tests/test_scoring_v4_rolling_peak.py
python -m pytest -q tests/test_qc.py tests/test_csp_hardening.py
python -m pytest -q tests/test_ui_cli_reporting.py tests/test_app_settings.py tests/test_integration_qc_and_options.py
```

The EMG half has its own suite (`tests/test_emg_core.py`,
`tests/test_emg_tab.py`); `tests/test_marker_shift.py` and
`tests/test_recordings.py` cover the code both halves share.

The tests cover H5 trigger/EEG separation, preparation exclusion, recovery
extension, marker padding on active and rest epochs, baseline modes and
transition exclusion, window bounds, candidate
counts/scales, score-version edge cases, independent sustained-ERD scoring and
diagnostics, QC, covariance conditioning/shrinkage,
component-count guards, GUI persistence, the settings file, score-gauge
cutoffs, component-selector alignment, first-pass component selection and its
fallback message, and JSON conversion. They do not
constitute cohort calibration or a MATLAB cross-check on matched
overt/quasi/imagery recordings.

## Performance

Long-recording FIR filtering and superlet transforms dominate runtime (often
about 1–3 minutes per subject). The superlet uses FFT convolution; direct
`filtfilt` is retained for MATLAB fidelity. JSON/provenance reporting and the
synthetic test suite do not require a full data analysis.
