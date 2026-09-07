"""Opt-in executable check using synthetic data and an isolated Qt profile."""
import json
import os
from pathlib import Path
import tempfile
import traceback


def run(report_path: str) -> int:
    report = {}
    try:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        import h5py
        import numpy as np
        import pyxdf
        from PyQt6 import QtCore, QtWidgets
        from app import NeuroCastingApp, SETTINGS_FILE_NAME
        from emgcasting.core import resolve_output_root
        from klh import superlet
        from klh.assets import load_ced, load_helper
        from shared.runtime import application_dir

        qapp = QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory(prefix="neurocasting-smoke-") as scratch:
            settings = QtCore.QSettings(
                str(Path(scratch) / "profile.ini"),
                QtCore.QSettings.Format.IniFormat)
            window = NeuroCastingApp(settings=settings)
            try:
                window.show()
                qapp.processEvents()
                assert not window.windowIcon().isNull(), "Window icon missing"
                assert Path(window.config_path) == application_dir() / SETTINGS_FILE_NAME
                assert Path(window.config_path).is_file(), "Settings were not saved"
                assert resolve_output_root("output") == application_dir() / "output"
                assert len(load_ced()["labels"]) == 64
                assert load_helper()["score_img"].shape[-1] == 15
                assert callable(pyxdf.load_xdf)
                with h5py.File(Path(scratch) / "synthetic.h5", "w") as handle:
                    handle["samples"] = np.arange(10)
                report["gui_assets_settings_and_io"] = "passed"

                signal = np.sin(2 * np.pi * 10 * np.arange(24_000) / 500)
                frequencies = np.array([8., 10., 12., 14.])
                superlet.configure_workers(1)
                serial = superlet.aslt(signal, 500, frequencies, 3, (1, 2), 0)
                superlet.configure_workers(2)
                parallel = superlet.aslt(signal, 500, frequencies, 3, (1, 2), 0)
                assert superlet._POOL is not None, "Worker pool was not exercised"
                np.testing.assert_array_equal(serial, parallel)
                report["frozen_worker_transform"] = "passed"
            finally:
                superlet.shutdown_workers()
                window.close()
                qapp.processEvents()
        report["status"] = "passed"
        code = 0
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        code = 1
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return code
