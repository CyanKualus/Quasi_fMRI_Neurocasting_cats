"""Olive-Hawkins robust covariance — MATLAB ``robustcov(..., 'olivehawkins')``.

This delegates to the vendored :mod:`klh._ohcov` implementation, which has been
verified by the lab to produce **identical** output to MATLAB's ``robustcov``.
It is kept unmodified for that reason; this module only adapts the call/return
signature to what the rest of the port expects (``(C, T)``).
"""
from __future__ import annotations

import numpy as np

from ._ohcov import olivehawkins_robustcov


def olivehawkins(X, reweight="rfch", csteps=10, outlier_fraction=0.5, **kw):
    """Return (C, T): robust covariance and location of X (n x p).

    Parameters mirror the MATLAB options actually used by the app; extra
    keyword arguments are forwarded to the vendored implementation (e.g.
    ``start_method``, ``num_trials``, ``random_state``).
    """
    Sig, Mu, *_ = olivehawkins_robustcov(
        np.asarray(X, dtype=np.float64),
        outlier_fraction=outlier_fraction,
        reweighting_method=reweight,
        num_concentration_steps=csteps,
        **kw,
    )
    return Sig, Mu


def robustcov(X, method="olivehawkins", **kw):
    """Convenience wrapper matching the MATLAB call the app uses.

    Returns just the covariance matrix.
    """
    if method != "olivehawkins":
        raise NotImplementedError(f"method {method!r} not ported")
    C, _ = olivehawkins(X, **kw)
    return C
