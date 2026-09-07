"""Finding a participant's overt / quasi / imagery recordings in a folder.

Filenames in this study carry their condition as a prefix, but not one spelling
of it: overt movement has been called ``om``, ``origmov`` and ``full``, imagery
``im`` and ``mi``, quasi movement ``qm`` and ``quasi``. Sequence tagging then
appends to the stem (``om1_order_01``), and older runs carry ad-hoc suffixes
(``mi1_finished``). Every known spelling is listed below rather than inferred,
so an unrecognised name is reported as unmatched instead of being filed under a
guess.

A prefix only counts when a digit follows it immediately. That single rule is
what keeps ``important_notes.xdf`` out of the imagery list while still matching
every real name in the study, all of which are a condition prefix followed by a
run number.

Detection fills the file lists in; it does not own them. The lists stay
editable, and nothing here runs again unless the folder changes or the operator
asks for a rescan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Ordered longest-first within each condition so that a stem matching both a
# long and a short spelling is attributed by the more specific one.
CONDITION_PREFIXES: dict[str, tuple[str, ...]] = {
    "overt": ("origmov", "overt", "full", "real", "om"),
    "quasi": ("quasi", "qm"),
    "imagery": ("imag", "im", "mi"),
}

# The order the three conditions are presented in, and the order a single
# combined list is built in.
CONDITION_ORDER = ("overt", "quasi", "imagery")

# A recording folder may sit one level below the participant, in a per-day
# subfolder: ``.../008TST/D1``. Such a name is a session, never a participant.
SESSION_FOLDER = re.compile(r"^d(?:ay)?\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveredRecordings:
    """Recording stems found in one folder, split by condition."""

    folder: str
    extension: str
    overt: list[str] = field(default_factory=list)
    quasi: list[str] = field(default_factory=list)
    imagery: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)

    def __getitem__(self, condition: str) -> list[str]:
        return getattr(self, condition)

    def all(self) -> list[str]:
        """Every matched stem, overt then quasi then imagery."""
        return [stem for condition in CONDITION_ORDER for stem in self[condition]]

    @property
    def total(self) -> int:
        return len(self.all())

    def summary(self) -> str:
        parts = [f"{condition} {len(self[condition])}"
                 for condition in CONDITION_ORDER]
        text = f"{self.total} recording(s) found: " + ", ".join(parts)
        if self.unmatched:
            shown = ", ".join(self.unmatched[:4])
            more = "" if len(self.unmatched) <= 4 else f", +{len(self.unmatched) - 4} more"
            text += (f"; {len(self.unmatched)} name(s) not recognised as a "
                     f"condition and left out ({shown}{more})")
        return text


def _natural_key(stem: str):
    """Sort ``om2`` before ``om10`` rather than after it."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", stem)]


def classify_recording(stem: str) -> str | None:
    """Which condition a recording stem belongs to, or None if unrecognised."""
    lowered = stem.strip().lower()
    best_condition, best_length = None, 0
    for condition, prefixes in CONDITION_PREFIXES.items():
        for prefix in prefixes:
            if len(prefix) <= best_length or not lowered.startswith(prefix):
                continue
            # The run number is what separates a condition prefix from a word
            # that merely begins with the same letters.
            rest = lowered[len(prefix):]
            if rest[:1].isdigit():
                best_condition, best_length = condition, len(prefix)
    return best_condition


def discover_recordings(folder, extension: str = ".xdf") -> DiscoveredRecordings:
    """Split the recordings directly inside *folder* by condition.

    Only that folder is read; per-day subfolders are recording folders in their
    own right and are selected as such. A folder that does not exist yields an
    empty result rather than an error, because this runs while an operator is
    still typing a path.
    """
    directory = Path(str(folder).strip().strip('"'))
    found: dict[str, list[str]] = {name: [] for name in CONDITION_ORDER}
    unmatched: list[str] = []
    if directory.is_dir():
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() != extension.lower():
                continue
            condition = classify_recording(path.stem)
            if condition is None:
                unmatched.append(path.stem)
            else:
                found[condition].append(path.stem)
    return DiscoveredRecordings(
        folder=str(directory),
        extension=extension,
        overt=sorted(found["overt"], key=_natural_key),
        quasi=sorted(found["quasi"], key=_natural_key),
        imagery=sorted(found["imagery"], key=_natural_key),
        unmatched=sorted(unmatched, key=_natural_key),
    )


def participant_code(folder) -> str:
    """The participant a recording folder belongs to.

    ``.../new/008TST/D1`` is participant ``008TST``'s first day, not a
    participant called ``D1``, so trailing session folders are skipped. Anything
    else is taken as typed; guessing further would be inventing structure that
    the folder layout does not state.
    """
    directory = Path(str(folder).strip().strip('"'))
    while (SESSION_FOLDER.fullmatch(directory.name)
           and directory.parent != directory
           and directory.parent.name):
        directory = directory.parent
    return directory.name
