"""Access to bundled assets exported from helper_files.mat and the .ced layout."""
from __future__ import annotations

import os
import functools
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


@functools.lru_cache(maxsize=1)
def load_helper() -> dict:
    """Load the exported helper arrays (filters, ideal pattern, images, refs)."""
    with np.load(os.path.join(ASSETS_DIR, "helper.npz")) as z:
        return {k: z[k] for k in z.files}


def asset_path(name: str) -> str:
    return os.path.join(ASSETS_DIR, name)


def load_ced(path: str | None = None):
    """Parse the EEGLAB .ced channel layout.

    Returns a dict with labels (list[str]) and numpy arrays theta, radius,
    x, y, z (all length n_channels, in file order).
    """
    if path is None:
        path = asset_path("mumeg_mks64.ced")
    labels, theta, radius, X, Y, Z = [], [], [], [], [], []
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            labels.append(parts[col["labels"]])
            theta.append(float(parts[col["theta"]]))
            radius.append(float(parts[col["radius"]]))
            X.append(float(parts[col["X"]]))
            Y.append(float(parts[col["Y"]]))
            Z.append(float(parts[col["Z"]]))
    return {
        "labels": labels,
        "theta": np.array(theta),
        "radius": np.array(radius),
        "x": np.array(X),
        "y": np.array(Y),
        "z": np.array(Z),
    }
