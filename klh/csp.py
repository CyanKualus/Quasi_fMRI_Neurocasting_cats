"""CSP via generalized eigendecomposition — port of ``calcCSP_cov.m``.

Coded originally by A.N. Vasilyev, 2020. Inputs are two covariance matrices
S1 (e.g. active/movement) and S2 (e.g. rest). Because the columns of the
returned filters are only ever used up to sign and scale (data projection is
later normalised to dB; patterns are compared with correlation and plotted
with 'absmax'), eigenvector normalisation does not affect downstream results.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, pinv


def _sign_fix(W, R1):
    """Enforce the MATLAB sign convention on eigenvectors.

    The largest-magnitude element of each forward-projection column is made
    positive. Assumes square W (n_channels == n_components), as in the app.
    """
    fProj = (W.T @ R1).T                     # (ch x comp)
    maxind = np.argmax(np.abs(fProj), axis=0)  # per column
    rowsign = np.sign(fProj[maxind, np.arange(fProj.shape[1])])
    rowsign[rowsign == 0] = 1.0
    return W * rowsign[None, :]


def _normalize_cols(W):
    norms = np.linalg.norm(W, axis=0)
    norms[norms == 0] = 1.0
    return W / norms[None, :]


def _shrink_covariance(S, gamma):
    """Shrink ``S`` toward its scaled identity (gamma=0 is an exact no-op)."""
    S = np.asarray(S, dtype=np.float64)
    gamma = float(gamma)
    if not 0.0 <= gamma < 1.0:
        raise ValueError("shrinkage must satisfy 0 <= gamma < 1")
    if gamma == 0.0:
        return S
    target = (np.trace(S) / S.shape[0]) * np.eye(S.shape[0])
    return (1.0 - gamma) * S + gamma * target


def covariance_condition(S1, S2):
    """Condition number of the trace-normalised CSP denominator ``R1+R2``."""
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    if S1.ndim != 2 or S1.shape[0] != S1.shape[1] or S2.shape != S1.shape:
        raise ValueError("S1 and S2 must be square covariance matrices of equal shape")
    t1, t2 = float(np.trace(S1)), float(np.trace(S2))
    if not np.isfinite(t1) or not np.isfinite(t2) or t1 <= 0 or t2 <= 0:
        return np.inf
    return float(np.linalg.cond(S1 / t1 + S2 / t2))


def calc_csp_cov(S1, S2, *, shrinkage=0.0, include_ged=True):
    """Return (projInverse, projForward, projInverse_GED, projForward_GED, evals).

    * projInverse / projForward come from eig(R1, R1+R2) (ascending) — these
      are what the app uses. ``evals`` are the corresponding sorted eigenvalues
      in [0, 1].
    * The *_GED pair come from eig(R1, R2) and are provided for completeness.
      Set ``include_ged=False`` when only the bounded CSP solution is needed;
      this also permits a singular ``R2`` when ``R1 + R2`` is well posed.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    if S1.ndim != 2 or S1.shape[0] != S1.shape[1] or S2.shape != S1.shape:
        raise ValueError("S1 and S2 must be square covariance matrices of equal shape")
    S1 = _shrink_covariance(S1, shrinkage)
    S2 = _shrink_covariance(S2, shrinkage)
    if np.trace(S1) <= 0 or np.trace(S2) <= 0:
        raise ValueError("covariance matrices must have positive trace")
    R1 = S1 / np.trace(S1)
    R2 = S2 / np.trace(S2)

    # --- eig(R1, R1+R2): eigenvalues in [0,1], ascending ---
    evals, W = eigh(R1, R1 + R2)            # eigh returns ascending real evals
    W = _normalize_cols(W)
    W_fixed = _sign_fix(W, R1)
    projInverse = W_fixed
    projForward = pinv(W_fixed).T
    L1 = evals.copy()

    # --- eig(R1, R2): optional generalized (GED), ascending ---
    # The application never consumes this second decomposition.  Keeping it
    # optional prevents a singular R2 from rejecting an otherwise valid
    # bounded CSP pencil (R1, R1 + R2).
    if include_ged:
        _, Wg = eigh(R1, R2)
        Wg = _normalize_cols(Wg)
        Wg_fixed = _sign_fix(Wg, R1)
        projInverse_GED = Wg_fixed
        projForward_GED = pinv(Wg_fixed).T
    else:
        projInverse_GED = None
        projForward_GED = None

    return projInverse, projForward, projInverse_GED, projForward_GED, L1
