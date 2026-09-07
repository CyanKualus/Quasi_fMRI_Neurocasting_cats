# NeuroCasting — Kitties' Little Helper for Quasi-fMRI

[![Tests](https://github.com/CyanKualus/Quasi_fMRI_Neurocasting_cats/actions/workflows/tests.yml/badge.svg)](https://github.com/CyanKualus/Quasi_fMRI_Neurocasting_cats/actions/workflows/tests.yml)

One desktop application for both analyses of a Quasi-fMRI session:

| Half | Question | Where it is documented |
|---|---|---|
| **EEG overt-template transfer score** | Does a CSP component learned from overt movement retain the historical pattern and transfer to quasi-movement and imagery? | [`docs/eeg_transfer_score.md`](docs/eeg_transfer_score.md) |
| **Hand EMG screening** | How many trials of each recording carry real hand EMG, in each hand separately? | [`docs/emg_analysis.md`](docs/emg_analysis.md) |

It merges two programs that already read the same recordings: `KLH_fMRI_Quasi`
(“KITTIES' LIL HELPER”, the `klh` package) and `EMGcasting` (the `emgcasting`
package). They discovered the same files, in the same folders, and corrected the
same amplifier timestamp lag, in two separate copies of the same code. Here they
share one Start tab and one copy of that code (`shared/`), so a session is
described once and analysed both ways.

The two results stay separate. EMG contamination scores no points in the
transfer score; it says whether a recording's score should be trusted at all.

Original MATLAB algorithm by **A.N. Vasilyev (2020)**; superlet transform by
Harald Barzan; app “KITTIES' LIL HELPER” (KLH_100).

## Install and run

Requires Python 3.10 or newer; Python 3.12 is recommended for the Windows
launcher. Clone the repository and create an isolated environment:

```powershell
git clone https://github.com/CyanKualus/Quasi_fMRI_Neurocasting_cats.git
cd Quasi_fMRI_Neurocasting_cats
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

After installation, Windows users can double-click `launch_NeuroCasting.bat`.
It prefers this folder's `.venv`, then Python 3.12 via `py`, then `python`
from PATH. The launcher starts the app; it does not install dependencies.

On Linux or macOS, use `python3 -m venv .venv`, then
`.venv/bin/python -m pip install -r requirements.txt` and
`.venv/bin/python app.py`. The GUI needs a graphical desktop. Automated CI
currently covers Windows with Python 3.10 and 3.12.

On the **Start** tab, click **Browse...** to select your recording folder,
check the detected overt/quasi/imagery filenames and EMG channel labels,
then click **START**. The initial folder text is an example from the original
study; replace it with your own path. Supply your own XDF recordings for EEG
and EMG, or H5 recordings for EEG only. Participant recordings and generated
diagnostics are not included in this repository.

## Portable Windows executable

Run `build_windows.bat` to build `dist/NeuroCasting.exe` with the bundled
kitten-with-cables picture as its Explorer, window, and taskbar icon. The build
uses Python 3.12 from the Windows launcher when `.venv` does not yet exist,
and installs `requirements-build.txt` into that environment. Alternatively:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm NeuroCasting.spec
```

Copy the single executable to a writable folder and double-click it. Python
does not need to be installed on the destination computer. The executable
includes Qt, the numerical libraries, and the application assets. First launch
can take a little longer while the bundle extracts. Settings are created as
`neurocasting_settings.json` beside the executable, and relative output folders
are resolved from that same location. Keep the settings and `output/` folder
with the executable when moving an existing installation.

The build is unsigned. Generated binaries stay in the Git-ignored `dist/`
folder. To check the compiled GUI, assets, settings paths, H5 support, and an
actual two-process EEG transform without participant data:

```powershell
$exe = (Resolve-Path .\dist\NeuroCasting.exe).Path
$report = Join-Path (Split-Path $exe) "smoke-test.json"
$process = Start-Process -FilePath $exe -ArgumentList "--smoke-test `"$report`"" -WindowStyle Hidden -Wait -PassThru
Get-Content $report
if ($process.ExitCode -ne 0) { throw "Executable smoke test failed" }
```

This check uses an isolated Qt profile and creates the default settings beside
the executable if they do not exist.

## The four tabs

**Start** — everything that changes from participant to participant, for both
analyses: folder template and subject name, the three filename lists (overt,
quasi, imagery), which hand the EEG half analyses, the marker time shift, and
the two bipolar EMG derivations. Setting the folder or the subject fills the
filename lists from that folder; the lists stay editable. You can also click
**Browse...** beside the folder field to choose the recording folder directly.
This fills in the path and subject name and detects the recordings. When
selecting a session subfolder such as `008TST/D1`, the subject remains `008TST`.

**START runs both halves.** The EMG screening runs first and opens the EMG
Analysis tab; the EEG pipeline then runs on the same recordings and opens the
EEG Components tab when its CSP components are ready. EMG goes first because it
finishes in a fraction of the time the CSP fit needs, so its verdict is on
screen while the EEG half is still going. They are chained rather than run at
once so that one session is not loaded into memory twice over, and either may
fail without withholding the other's result. An H5 run reports the EMG half as
skipped: H5 files carry no auxiliary EMG channels.

**EEG Components** — the CSP eigenvalue spectrum, up to five forward patterns
with one preselected, the three condition time-frequency maps, and
*Calculate FT*.

**EEG Score** — the six criterion gauges, the 15-point total and the score
image.

A panel with no result behind it carries no scale: the eigenvalue, map, gauge
and score panels are blank until the step that fills them has finished, and are
blanked again when a new START begins. An axis is drawn only over numbers a
recording produced — the time axis in particular comes from the marker
durations found in the files, so there is nothing truthful to draw before they
are read.

**EMG Analysis** — one card per recording: the mean left- and right-hand
envelopes side by side, the share of trials carrying high EMG in each hand,
and a button per hand that opens every trial of that hand. The detector's
false-positive rate on the recording's own rest is measured but not shown on
the card; it is in `emg_summary.csv` and in the `run_emg_cli.py` table. The trial
browser shows all trials at once as thumbnails — pink for high EMG — and
enlarges any of them into a zoomable panel with the numbers behind its verdict.
Nothing is written to disk until **Save figures + CSV** is pressed.

## Settings

Two files, with different jobs:

- **`neurocasting_settings.json`**, next to `app.py`, holds every analysis
  parameter that is not on the Start tab — for both halves. It is created with
  the built-in defaults on first run. A missing key, an unusable value or an
  unknown key is reported on the Start tab and repaired in place, so the file
  always states what is actually in force. EEG keys are unprefixed; every EMG
  key starts with `emg_`.
- The **Qt profile** remembers the Start fields between sessions (folder,
  subject, the three filename lists, the EMG electrode labels, the selected
  hand, the marker shift, window geometry). Analysis parameters are deliberately
  not mirrored there: a stale profile must not be able to outrank the file.

The generated settings file is ignored by Git, so local analysis choices stay
local. The application creates it automatically; no template needs to be copied.

`superlet_workers` sets how many
processes split the time-frequency transform across cores (0 or 1 keeps it in
one). Every worker uses the whole spectrum's wavelet orders and block geometry,
so any setting produces bit-identical output; it costs about 170 MB per worker,
and on a six-core machine four is the fastest setting.

`analysis_workers` controls threads for independent CSP covariance fits,
cross-validation folds, bootstrap batches, and broadband/alpha filtering.
The default is **2**; **0 or 1** runs these steps serially. Counts are capped
to the available CPUs and reduced when estimated working memory exceeds
available RAM (with 256 MB headroom where the platform exposes it).
This is a conservative scheduling estimate, not a hard memory limit.
More workers increase temporary memory use and may
not be faster. Recordings still load one at a time. Concurrent covariance
fits use one BLAS thread each, and the previous BLAS setting is restored when
the step finishes or fails. These threads do not wrap the superlet process
pool or change the number of folds or bootstrap resamples.

The pipeline caches one configuration's CV covariance matrices and complete
CSP decompositions, then matches and scales the chosen component separately.
Sensor-space QC is also reused across component selections. Content keys
detect changes to the input windows and relevant settings, and START clears
both caches. Caches stay in memory; `KLHPipeline.clear_analysis_cache()` can
release them explicitly. Recording-level QC reuses per-trial Welch powers
from the corresponding condition, without changing the median reductions.
Epoch stores are concatenated once per condition, and XDF montage discovery
reads stream headers while seeking over sample payloads.

Bootstrap draws and reductions retain their original order across worker
counts. Different BLAS thread counts can introduce floating-point rounding
in CSP fitting; serial/parallel regression tests check tight numerical
agreement, held-out trial assignment, and scores. No sampling rates, filters,
frequency grids, normalization rules, or scoring thresholds are relaxed.
Install the added `threadpoolctl` dependency with the usual
`python -m pip install -r requirements.txt` command.

For the EEG CLI, use `--analysis-workers 2 --superlet-workers 4`, or set both
options to `1` to disable the application-level parallel workers. Native
numerical libraries may still use their own threads on the serial path.

The score formula is locked to `scoring-v4-rolling-peak-independent-nonreal-erd`
in this build. It is written to the settings file so a result can be traced back
to the formula that produced it; editing it there is reported and corrected.
Use `run_cli.py --score-version` to compare formulas explicitly.

## Command line

```powershell
python run_cli.py --help          # EEG transfer score, JSON + figure output
python run_emg_cli.py --help      # hand EMG, full figure and CSV output
```

Both take a folder and comma-separated filenames without extensions. The EMG
runner analyses and writes in one call, which is the difference from the
application, where saving is a separate button.

## Layout

```
app.py                     the four-tab application
emg_view.py                the EMG Analysis tab and its trial browser
run_cli.py                 headless EEG transfer score
run_emg_cli.py             headless hand EMG
neurocasting_settings.json analysis parameters for both halves (created on first run)
klh/                       EEG: IO, filters, robust covariance, CSP, superlets,
                           scoring, QC, topographies, pipeline
emgcasting/                EMG: loading, envelope, epoching, trial classification,
                           drawing and output
shared/                    what both halves need: recording discovery by
                           filename, the per-recording marker time shift, and
                           the one palette and typeface both tabs are drawn in
assets/                    filter coefficients, ideal pattern, score images,
                           the 64-channel .ced layout
docs/                      the two analysis documents
tests/                     synthetic and unit tests for all of the above
output/                    EMG figures and tables, once you save them; always
                           inside this folder, so a copy keeps its own results
```

`emgcasting.core` separates analysis from output: `analyze_batch` produces the
epochs and verdicts and writes nothing, `save_batch_outputs` turns that same
result into files, and `process_batch` does both for the CLI. The application
draws its panels through the same `draw_hand_mean` and `draw_trial` functions
that write the saved figures, so the screen and a saved file cannot disagree.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The suite is synthetic throughout; no recordings are needed.
Use the virtual environment's Python for these commands (activate it first,
or substitute `.\.venv\Scripts\python.exe` on Windows). GUI tests use Qt's
offscreen platform, so they do not need an interactive window.

```powershell
python -m pytest -q tests/test_emg_core.py tests/test_emg_tab.py       # EMG half
python -m pytest -q tests/test_marker_shift.py tests/test_recordings.py # shared
python -m pytest -q tests/test_settings_file.py tests/test_app_settings.py
```

They do not constitute cohort calibration, or a MATLAB cross-check on matched
overt/quasi/imagery recordings.

## What was left out of the merge

Ad-hoc scripts that analysed particular recordings — hard-coded participant
tables, one-off timeline plots, EEGLAB video-trial conversions, saved
diagnostic figures and their JSON — were not carried over. Everything here is
reached from the application or from one of the two runners. The KitEX
single-file alpha screening tab was replaced by the EMG Analysis tab.

## Contributing

Install `requirements-dev.txt` and run the tests before opening a pull request.
For analysis changes, include a synthetic regression case and describe any
effect on scores or saved results. Keep participant data, generated reports,
and local settings out of commits; the standard folders and recording formats
are covered by `.gitignore`. Check `git status` before committing outputs in
custom locations. Include the command, Python version, and traceback in bug
reports, using synthetic inputs where possible.
