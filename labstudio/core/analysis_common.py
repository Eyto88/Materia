"""General-purpose x/y analysis helpers shared by the DSC, mechanical and cyclic analyses.

Pure numpy/scipy (no Qt). All functions accept array-likes and ignore non-finite points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

__all__ = [
    "LinearFit", "Integration", "XYPeak",
    "linear_fit", "integrate_xy", "find_peaks_xy", "smooth", "trapz",
]


def trapz(y, x) -> float:
    """Trapezoidal integral of ``y`` along ``x`` (path order, signed)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def _finite(*arrays):
    arrs = [np.asarray(a, dtype=float).ravel() for a in arrays]
    m = np.ones(arrs[0].shape, bool)
    for a in arrs:
        m &= np.isfinite(a)
    return [a[m] for a in arrs]


def smooth(y, window: int, polyorder: int = 2, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    """Savitzky–Golay smoothing (or derivative) with a window clipped to the data length.

    ``window`` is in points; it is made odd and at least ``polyorder + 2``.
    """
    y = np.asarray(y, float)
    n = y.size
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= polyorder + 1:
        if deriv == 0:
            return y.copy()
        return np.gradient(y, delta) if n > 1 else np.zeros_like(y)
    return signal.savgol_filter(y, window, polyorder, deriv=deriv, delta=delta, mode="interp")


# --------------------------------------------------------------------------- linear fit
@dataclass
class LinearFit:
    """Least-squares straight line ``y = slope * x + intercept`` over [xmin, xmax]."""
    slope: float
    intercept: float
    r2: float
    xmin: float
    xmax: float
    n: int

    def __call__(self, x):
        return self.slope * np.asarray(x, float) + self.intercept

    def x_at(self, y: float) -> float:
        """x where the line reaches ``y`` (nan for a zero slope)."""
        return (y - self.intercept) / self.slope if self.slope != 0 else float("nan")


def linear_fit(x, y, xmin: float | None = None, xmax: float | None = None) -> LinearFit:
    """Fit a straight line to the points with ``xmin <= x <= xmax``.

    Returns slope (y-unit per x-unit), intercept, coefficient of determination R².
    Raises ``ValueError`` if fewer than two points are in range.
    """
    x, y = _finite(x, y)
    m = np.ones(x.shape, bool)
    if xmin is not None:
        m &= x >= xmin
    if xmax is not None:
        m &= x <= xmax
    xs, ys = x[m], y[m]
    if xs.size < 2 or np.ptp(xs) == 0:
        raise ValueError("linear_fit needs at least two distinct x values in range")
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return LinearFit(float(slope), float(intercept), r2, float(xs.min()), float(xs.max()), int(xs.size))


# --------------------------------------------------------------------------- integration
@dataclass
class Integration:
    """Result of :func:`integrate_xy`. ``area`` has unit y_unit·x_unit."""
    area: float
    xmin: float
    xmax: float
    baseline: str
    x: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    y_base: np.ndarray = field(repr=False)


def integrate_xy(x, y, xmin: float | None = None, xmax: float | None = None,
                 baseline: str = "none", sort: bool = True) -> Integration:
    """Integrate ``y`` d``x`` over [xmin, xmax].

    baseline: ``"none"`` (area to y = 0) or ``"linear"`` (straight line between the
    curve values at the two ends of the range, e.g. for a DSC peak).
    With ``sort=True`` points are ordered by x first (recommended for single-valued
    curves); with ``sort=False`` the path order is kept (signed by direction of x).
    """
    if baseline not in ("none", "linear"):
        raise ValueError("baseline must be 'none' or 'linear'")
    x, y = _finite(x, y)
    if sort:
        o = np.argsort(x, kind="stable")
        x, y = x[o], y[o]
    lo = np.nanmin(x) if xmin is None else xmin
    hi = np.nanmax(x) if xmax is None else xmax
    m = (x >= lo) & (x <= hi)
    xs, ys = x[m], y[m]
    if xs.size < 2:
        raise ValueError("integrate_xy needs at least two points in range")
    if baseline == "linear":
        yb = ys[0] + (ys[-1] - ys[0]) * (xs - xs[0]) / (xs[-1] - xs[0]) if xs[-1] != xs[0] \
            else np.full_like(ys, ys[0])
    else:
        yb = np.zeros_like(ys)
    area = trapz(ys - yb, xs)
    return Integration(area, float(xs.min()), float(xs.max()), baseline, xs, ys, yb)


# --------------------------------------------------------------------------- peaks
@dataclass
class XYPeak:
    index: int          # index into the (finite) input arrays
    x: float
    y: float
    prominence: float
    fwhm: float         # full width at ``rel_height`` of the prominence, in x units
    left_x: float
    right_x: float


def find_peaks_xy(x, y, prominence: float | None = None, distance: float | None = None,
                  sign: int = 1, rel_height: float = 0.5) -> list[XYPeak]:
    """Find peaks (``sign=+1``) or valleys (``sign=-1``) of y(x).

    prominence: minimum prominence in y units (default 5 % of the y range).
    distance: minimum separation in x units (converted to samples via the median spacing).
    Widths are reported in x units (linear interpolation of x on the fractional index).
    """
    x, y = _finite(x, y)
    if x.size < 3:
        return []
    ys = y * (1 if sign >= 0 else -1)
    if prominence is None:
        prominence = 0.05 * float(np.ptp(ys)) if np.ptp(ys) > 0 else None
    kw = {"prominence": prominence if prominence is not None else 0}
    if distance:
        dx = float(np.median(np.abs(np.diff(x)))) or 1.0
        kw["distance"] = max(1, int(round(distance / dx)))
    idx, props = signal.find_peaks(ys, **kw)
    if idx.size == 0:
        return []
    widths, _, lips, rips = signal.peak_widths(ys, idx, rel_height=rel_height)
    fi = np.arange(x.size, dtype=float)
    lx = np.interp(lips, fi, x)
    rx = np.interp(rips, fi, x)
    out = []
    for k, i in enumerate(idx):
        out.append(XYPeak(int(i), float(x[i]), float(y[i]), float(props["prominences"][k]),
                          float(abs(rx[k] - lx[k])), float(min(lx[k], rx[k])), float(max(lx[k], rx[k]))))
    return out
