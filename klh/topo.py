"""Scalp topographic maps, mimicking EEGLAB ``topoplot``.

EEGLAB's default interpolation is the MATLAB 'v4' biharmonic spline (Sandwell,
1987). We reproduce it here so the maps match the app's look, using the
electrode positions from the .ced polar coordinates. Rendering uses matplotlib
so the maps can be embedded directly in the Qt canvases.
"""
from __future__ import annotations

import numpy as np

from .assets import load_ced

# EEGLAB topoplot defaults used by the app.
HEADRAD = 0.5
INTRAD = 0.65
GRIDSCALE = 200


def _positions(ced=None):
    """2-D projected electrode coordinates matching topoplot's orientation.

    topoplot places theta=0 (front) pointing up and increasing theta clockwise.
    It computes x = radius*cos(theta), y = radius*sin(theta) then plots (y, x)
    with x pointing up, i.e. the nose is at the top.
    """
    if ced is None:
        ced = load_ced()
    th = np.deg2rad(ced["theta"])
    rd = ced["radius"]
    x = rd * np.cos(th)   # "front-back"
    y = rd * np.sin(th)   # "left-right"
    # Return in plot coordinates: horizontal = y (left-right), vertical = x.
    return y, x, ced["labels"]


def _select_labels(ced, labels):
    """Return a CED layout subset in ``labels`` order."""
    if labels is None:
        return ced
    lut = {str(label).casefold(): i for i, label in enumerate(ced["labels"])}
    indices = [lut[str(label).casefold()] for label in labels]
    return {
        "labels": [ced["labels"][i] for i in indices],
        **{key: np.asarray(ced[key])[indices]
           for key in ("theta", "radius", "x", "y", "z")},
    }


def _griddata_v4(xk, yk, vk, xq, yq):
    """MATLAB 'v4' biharmonic spline interpolation (Sandwell 1987)."""
    xy = xk + 1j * yk
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(xy[:, None] - xy[None, :])
        g = (d ** 2) * (np.log(d) - 1.0)
    g[np.isnan(g) | ~np.isfinite(g)] = 0.0  # diagonal (d==0)
    weights = np.linalg.solve(g, vk)

    xyq = (xq + 1j * yq).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        dq = np.abs(xyq[:, None] - xy[None, :])
        gq = (dq ** 2) * (np.log(dq) - 1.0)
    gq[np.isnan(gq) | ~np.isfinite(gq)] = 0.0
    vq = gq @ weights
    return vq.reshape(xq.shape)


def interpolate(values, ced=None, gridscale=GRIDSCALE, intrad=INTRAD):
    """Return (Xi, Yi, Zi) interpolated scalp map with points outside the head
    interpolation radius set to NaN (so they are not drawn)."""
    hx, hy, _ = _positions(ced)
    values = np.asarray(values, dtype=np.float64).ravel()
    # Support channel counts other than the full montage: map only the leading
    # channels present in both the values and the electrode layout.
    n = min(values.size, hx.size)
    hx, hy, values = hx[:n], hy[:n], values[:n]

    lim = intrad
    xi = np.linspace(-lim, lim, gridscale)
    yi = np.linspace(-lim, lim, gridscale)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = _griddata_v4(hx, hy, values, Xi, Yi)

    mask = np.sqrt(Xi ** 2 + Yi ** 2) > intrad
    Zi[mask] = np.nan
    return Xi, Yi, Zi


def draw(ax, values, ced=None, cmap=None, contours=7, show_sensors=True,
         labels=None):
    """Draw a topographic map of ``values`` onto a matplotlib axis.

    Uses 'absmax' colour limits (symmetric about zero) like the app.
    """
    import matplotlib
    if ced is None:
        ced = load_ced()
    ced = _select_labels(ced, labels)
    if cmap is None:
        cmap = matplotlib.colormaps["RdBu_r"]

    Xi, Yi, Zi = interpolate(values, ced)
    amax = np.nanmax(np.abs(values))
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.contourf(Xi, Yi, Zi, 60, cmap=cmap, vmin=-amax, vmax=amax)
    if contours:
        ax.contour(Xi, Yi, Zi, contours, colors="k", linewidths=0.4, alpha=0.5)

    # Head, nose, ears at the interpolation/head radius.
    _draw_head(ax, HEADRAD)
    if show_sensors:
        hx, hy, _ = _positions(ced)
        n = min(np.asarray(values).ravel().size, hx.size)
        ax.plot(hx[:n], hy[:n], ".", color="k", markersize=2)
    # The map is interpolated out to INTRAD, so limits inside it slice the
    # coloured disc into a square and the map reads as cropped rather than
    # round. A little air past INTRAD also clears the nose and the ears.
    lim = INTRAD + 0.03
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return amax


def _draw_head(ax, r):
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(t), r * np.sin(t), "k", linewidth=1)
    # nose
    ax.plot([-0.1 * r, 0, 0.1 * r], [r * 0.995, r * 1.12, r * 0.995], "k", linewidth=1)
    # ears (simple lobes on each side)
    ear_y = np.array([0.18, 0.15, -0.15, -0.18]) * r / 0.5
    for sgn in (-1, 1):
        ax.plot(sgn * (r + np.array([0.0, 0.02, 0.02, 0.0])), ear_y, "k", linewidth=1)
