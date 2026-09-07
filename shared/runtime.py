"""Locations for writable files in source and portable executable builds."""
from pathlib import Path
import sys


def application_dir() -> Path:
    """Keep settings and results beside the app, outside bundle extraction."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
