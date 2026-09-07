"""Portable builds must not save user files into temporary bundle storage."""
from pathlib import Path
import sys

from shared.runtime import application_dir


def test_source_application_dir(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert application_dir() == Path(__file__).resolve().parent.parent


def test_frozen_application_dir_uses_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "unpacked"), raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "portable" / "NeuroCasting.exe"))
    monkeypatch.chdir(tmp_path)
    assert application_dir() == tmp_path / "portable"
