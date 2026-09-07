"""Study-level code shared by the EEG (``klh``) and EMG (``emgcasting``) halves.

Both analyses read the same recordings, in the same folder layout, with the
same amplifier timestamp lag.  Finding a participant's files and measuring that
lag are therefore properties of the *recording*, not of a modality, and live
here so the two pipelines cannot drift apart on either.

:mod:`shared.theme` is here for the same reason on the other side of the
application: the two halves are shown in one window, so they are drawn from one
palette. It is the only module here that needs Qt, and nothing imports it
unless a window is being built.
"""
__all__ = ["marker_shift", "recordings", "theme"]
