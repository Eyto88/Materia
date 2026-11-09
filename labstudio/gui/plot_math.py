"""Qt-free numerical helpers for the Plot tab (module B).

Everything here is pure numpy/scipy so it can be unit-tested without a display:

* :func:`integrate_span` – exact integral of the piece-wise linear curve over an
  x-span, optional linear baseline, works for monotonic *and* looping curves
  (e.g. superelastic hysteresis loops give the enclosed area).
* :func:`product_unit`, :func:`quotient_unit`, :func:`area_conversions` – unit
  bookkeeping for integrals / slopes, with physically meaningful conversions
  (MPa·% → MJ/m³, mW·s → J, W/g·s → J/g, mW·K ÷ heating rate → J …).
* :func:`find_peaks_xy` – scipy.signal.find_peaks wrapper returning x, y,
  prominence and FWHM in x units.
* :func:`linear_fit`, :func:`span_stats`.
* :func:`nearest_point` – nearest point in *display* (pixel) coordinates.
* :func:`display_indices` – view-aware min/max decimation used only for drawing.
* v1.1: :func:`monotonic_segments` / :func:`segment_names` (curves that double back
  in x, e.g. DSC heating+cooling or load/unload), :func:`average_curves`
  (mean ± SD across repeats), :func:`legend_groups` (collapsible legend groups)
  and :func:`place_labels` (non-overlapping annotation placement in pixels).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from labstudio.core import units as U

# --------------------------------------------------------------------------- units


def _base_unit(dimension: str) -> str:
    return U.units_for(dimension)[0]


def si_factor(dimension: str | None, unit: str | None) -> float | None:
    """Multiplicative factor taking a *difference/amount* in ``unit`` to the base unit
    of ``dimension`` (offsets ignored, so °C → K gives 1). ``None`` if unknown."""
    if not dimension or unit is None or dimension not in U.dimensions():
        return None
    try:
        base = _base_unit(dimension)
        return float(U.convert(1.0, unit, base, dimension) - U.convert(0.0, unit, base, dimension))
    except (ValueError, KeyError):
        return None


def product_unit(y_unit: str | None, x_unit: str | None) -> str:
    """Unit of ∫y dx, e.g. ``product_unit('MPa', '%') == 'MPa·%'``."""
    yu, xu = (y_unit or "").strip(), (x_unit or "").strip()
    if yu and xu:
        return f"{yu}·{xu}"
    return yu or xu


def quotient_unit(y_unit: str | None, x_unit: str | None) -> str:
    """Unit of dy/dx, e.g. ``quotient_unit('MPa', '%') == 'MPa/%'``."""
    yu, xu = (y_unit or "").strip(), (x_unit or "").strip()
    if yu and xu:
        return f"{yu}/{xu}"
    if xu:
        return f"1/{xu}"
    return yu


@dataclass
class Quantity:
    """One reported number: ``quantity`` name, ``value``, ``unit``."""
    quantity: str
    value: float
    unit: str


_TEMPS = ("temperature", "temperature_difference")


def area_conversions(area: float, y_dim: str | None, y_unit: str | None,
                     x_dim: str | None, x_unit: str | None, *,
                     heating_rate_K_per_min: float | None = None,
                     mass_mg: float | None = None) -> tuple[list[Quantity], list[str]]:
    """Express an integral ∫y dx in physically meaningful units.

    Returns ``(quantities, notes)``; the first quantity is always the raw area in
    ``y_unit·x_unit``. Recognised combinations:

    * stress × strain → energy density (MJ/m³); 1 MPa·% = 0.01 MJ/m³
    * force × displacement/length → work (J, mJ)
    * heat flow × time → energy (J, mJ) and, if ``mass_mg``, J/g
    * specific heat flow × time → J/g
    * (specific) heat flow × temperature → needs the heating rate β (K/min):
      ∫hf dt = ∫hf dT / β. Without β a note is returned instead.
    """
    out = [Quantity("Area", float(area), product_unit(y_unit, x_unit))]
    notes: list[str] = []
    fy, fx = si_factor(y_dim, y_unit), si_factor(x_dim, x_unit)
    if fy is None or fx is None or not np.isfinite(area):
        return out, notes
    si = area * fy * fx  # area in base units (e.g. Pa·1, N·m, W·s, W/g·s, W·K)
    mass_ok = mass_mg is not None and mass_mg > 0
    if y_dim == "stress" and x_dim == "strain":
        out.append(Quantity("Energy density", si / 1e6, "MJ/m³"))
    elif y_dim == "force" and x_dim in ("displacement", "length"):
        out += [Quantity("Work", si, "J"), Quantity("Work", si * 1e3, "mJ")]
    elif y_dim == "heat_flow" and x_dim == "time":
        out += [Quantity("Energy", si, "J"), Quantity("Energy", si * 1e3, "mJ")]
        if mass_ok:
            out.append(Quantity("Specific energy", si / (mass_mg * 1e-3), "J/g"))
        else:
            notes.append("Enter the sample mass (mg) to also get J/g.")
    elif y_dim == "specific_heat_flow" and x_dim == "time":
        out.append(Quantity("Specific energy", si, "J/g"))
    elif y_dim in ("heat_flow", "specific_heat_flow") and x_dim in _TEMPS:
        if heating_rate_K_per_min and heating_rate_K_per_min > 0:
            energy = si / (heating_rate_K_per_min / 60.0)  # W·K / (K/s) = J (or J/g)
            if y_dim == "heat_flow":
                out += [Quantity("Energy", energy, "J"), Quantity("Energy", energy * 1e3, "mJ")]
                if mass_ok:
                    out.append(Quantity("Specific energy", energy / (mass_mg * 1e-3), "J/g"))
                else:
                    notes.append("Enter the sample mass (mg) to also get J/g.")
            else:
                out.append(Quantity("Specific energy", energy, "J/g"))
        else:
            notes.append("Heat flow vs temperature: divide by the heating rate to get energy — "
                         "enter the heating rate (K/min) to convert to J or J/g.")
    return out, notes


def slope_conversions(slope: float, y_dim: str | None, y_unit: str | None,
                      x_dim: str | None, x_unit: str | None) -> list[Quantity]:
    """Slope with units; stress/strain slopes are also given as a modulus in GPa."""
    out = [Quantity("Slope", float(slope), quotient_unit(y_unit, x_unit))]
    fy, fx = si_factor(y_dim, y_unit), si_factor(x_dim, x_unit)
    if y_dim == "stress" and x_dim == "strain" and fy and fx:
        out.append(Quantity("Modulus E", slope * fy / fx / 1e9, "GPa"))
    return out


# --------------------------------------------------------------------------- integration


@dataclass
class IntegrationResult:
    """Result of :func:`integrate_span` (all in the units of the input arrays)."""
    area: float                 # ∫(y − baseline) dx, signed, following data order
    raw_area: float             # ∫y dx without baseline
    xmin: float
    xmax: float
    baseline: str
    xs: np.ndarray = field(repr=False)   # clipped curve (for shading)
    ys: np.ndarray = field(repr=False)
    base: np.ndarray = field(repr=False)  # baseline evaluated at xs
    monotonic: bool = True
    baseline_coeffs: tuple[float, float] = (0.0, 0.0)  # slope, intercept
    # v1.1 segment-aware integration (filled by the Integrate tool)
    warnings: list = field(default_factory=list)
    segment: str = ""                 # e.g. "heating 1", "whole path (loop area)"
    segment_index: int | None = None  # 0-based index into monotonic_segments, None = whole path
    loop: bool = False                # True when the result is an enclosed loop area


def _seg_integral(x0, y0, x1, y1, lo, hi):
    """Signed ∫ of linear segments clipped to [lo, hi] (vectorised).
    Returns integral and clipped endpoints (xa, ya, xb, yb, valid)."""
    dx = x1 - x0
    xa = np.clip(np.minimum(x0, x1), lo, hi)
    xb = np.clip(np.maximum(x0, x1), lo, hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(dx != 0, (y1 - y0) / np.where(dx != 0, dx, 1.0), 0.0)
    ya = y0 + slope * (xa - x0)
    yb = y0 + slope * (xb - x0)
    width = xb - xa
    integ = 0.5 * (ya + yb) * width * np.sign(dx)
    valid = (width > 0) & np.isfinite(integ)
    return np.where(valid, integ, 0.0), valid


def integrate_span(x, y, xmin: float | None = None, xmax: float | None = None,
                   baseline: str = "none") -> IntegrationResult:
    """Integrate ``y dx`` along the curve, restricted to ``xmin <= x <= xmax``.

    The data are treated as a piece-wise linear curve in their recorded order, so
    the result is exact for linear segments, endpoints are interpolated exactly at
    the span limits, and closed loops (load/unload cycles) return their enclosed
    (signed) area. ``baseline='linear'`` subtracts the straight line joining the
    curve at the two span ends (first and last in-span points in data order).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        raise ValueError("Need at least two finite points to integrate.")
    lo = float(np.nanmin(x)) if xmin is None else float(xmin)
    hi = float(np.nanmax(x)) if xmax is None else float(xmax)
    if lo > hi:
        lo, hi = hi, lo
    x0, x1, y0, y1 = x[:-1], x[1:], y[:-1], y[1:]
    integ, valid = _seg_integral(x0, y0, x1, y1, lo, hi)
    raw = float(integ.sum())

    # clipped polyline in data order (for shading and baseline anchors)
    xs_list, ys_list = [], []
    dx = x1 - x0
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(dx != 0, (y1 - y0) / np.where(dx != 0, dx, 1.0), 0.0)
    for i in np.flatnonzero(valid):
        a, b = (x0[i], x1[i])
        ca, cb = np.clip(a, lo, hi), np.clip(b, lo, hi)
        ya, yb = y0[i] + slope[i] * (ca - x0[i]), y0[i] + slope[i] * (cb - x0[i])
        if not xs_list or xs_list[-1] != ca or ys_list[-1] != ya:
            xs_list.append(ca)
            ys_list.append(ya)
        xs_list.append(cb)
        ys_list.append(yb)
    xs, ys = np.asarray(xs_list, float), np.asarray(ys_list, float)
    if xs.size == 0:
        raise ValueError("The span contains no data.")

    m, c = 0.0, 0.0
    area = raw
    if baseline == "linear":
        xa, xb, ya, yb = xs[0], xs[-1], ys[0], ys[-1]
        if xb != xa:
            m = (yb - ya) / (xb - xa)
            c = ya - m * xa
        else:
            m, c = 0.0, ya
        # ∫ (m x + c) dx over the same clipped segments (exact for a line)
        b0, b1 = m * x0 + c, m * x1 + c
        bint, _ = _seg_integral(x0, b0, x1, b1, lo, hi)
        area = raw - float(bint.sum())
    elif baseline != "none":
        raise ValueError(f"Unknown baseline '{baseline}' (use 'none' or 'linear').")
    base = m * xs + c if baseline == "linear" else np.zeros_like(xs)
    d = np.diff(xs)
    mono = bool(np.all(d >= 0) or np.all(d <= 0))
    return IntegrationResult(area, raw, lo, hi, baseline, xs, ys, base, mono, (m, c))


# --------------------------------------------------------------------------- monotonic segments


def monotonic_segments(x, tol_frac: float = 0.005) -> list[tuple[int, int, int]]:
    """Split a curve into runs where ``x`` moves in one direction.

    Returns ``[(start, stop, direction), …]`` (``stop`` exclusive, direction +1/−1,
    0 if x never moves). Consecutive segments share the turning point. Small
    reversals (noise) below ``tol_frac`` × range are ignored, isotherms/holds stay
    in the current segment. Fast on 10⁶ points (works on a strided copy, then
    refines each turning point on the full data).
    """
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return [(0, n, 0)]
    fin = np.isfinite(x)
    if not fin.any():
        return [(0, n, 0)]
    if not fin.all():  # fill gaps so they never create turning points
        idx = np.arange(n)
        x = np.interp(idx, idx[fin], x[fin])
    rng = float(x.max() - x.min())
    if rng == 0:
        return [(0, n, 0)]
    tol = tol_frac * rng
    stride = max(1, n // 20000)
    xs = x[::stride].tolist()
    d = 0
    hi = lo = xs[0]
    hi_i = lo_i = 0
    ext, ext_i = xs[0], 0
    first = 0
    turns: list[tuple[int, int]] = []  # (index in xs, +1 if maximum else -1)
    for i in range(1, len(xs)):
        v = xs[i]
        if d == 0:
            if v > hi:
                hi, hi_i = v, i
            if v < lo:
                lo, lo_i = v, i
            if hi - lo > tol:
                d = 1 if hi_i > lo_i else -1
                ext, ext_i = (hi, hi_i) if d == 1 else (lo, lo_i)
                first = d
        elif d == 1:
            if v > ext:
                ext, ext_i = v, i
            elif ext - v > tol:
                turns.append((ext_i, 1))
                d, ext, ext_i = -1, v, i
        else:
            if v < ext:
                ext, ext_i = v, i
            elif v - ext > tol:
                turns.append((ext_i, -1))
                d, ext, ext_i = 1, v, i
    if first == 0:
        return [(0, n, 0)]
    bounds = [0]
    for ti, kind in turns:
        c = ti * stride
        a, b = max(bounds[-1], c - stride), min(n, c + stride + 1)
        w = x[a:b]
        j = a + int(np.argmax(w) if kind == 1 else np.argmin(w))
        if j > bounds[-1]:
            bounds.append(j)
    segs = []
    dirn = first
    for k, a in enumerate(bounds):
        stop = bounds[k + 1] + 1 if k + 1 < len(bounds) else n
        segs.append((a, stop, dirn))
        dirn = -dirn
    return segs


def segment_names(segments: list[tuple[int, int, int]], x_dim: str | None) -> list[str]:
    """Human names for :func:`monotonic_segments` output: 'heating 1', 'cooling 1' for
    temperature, 'loading'/'unloading' for strain/displacement, else increasing/decreasing."""
    if x_dim in ("temperature", "temperature_difference"):
        words = {1: "heating", -1: "cooling", 0: "isothermal"}
    elif x_dim in ("strain", "displacement", "length", "force", "stress"):
        words = {1: "loading", -1: "unloading", 0: "hold"}
    else:
        words = {1: "increasing", -1: "decreasing", 0: "constant"}
    count: dict[int, int] = {}
    out = []
    for _a, _b, d in segments:
        count[d] = count.get(d, 0) + 1
        out.append(f"{words[d]} {count[d]}")
    return out


def segments_in_span(x, segments, lo: float, hi: float) -> list[int]:
    """Indices of segments that contain a non-zero x-width inside [lo, hi]."""
    x = np.asarray(x, float)
    out = []
    for k, (a, b, _d) in enumerate(segments):
        xx = x[a:b]
        xx = xx[np.isfinite(xx)]
        if xx.size < 2:
            continue
        if min(hi, float(xx.max())) - max(lo, float(xx.min())) > 0:
            out.append(k)
    return out


# --------------------------------------------------------------------------- averaging


@dataclass
class AverageResult:
    x: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    n: np.ndarray
    xmin: float
    xmax: float
    warnings: list = field(default_factory=list)


def average_curves(curves: list[tuple[np.ndarray, np.ndarray]], n_points: int | None = None) -> AverageResult:
    """Mean ± SD (ddof=1) of several y(x) curves on a common x grid over their overlap.

    Each curve is sorted by x (a warning is added if it doubles back in x — only
    its first monotonic segment is used then). ``n_points`` defaults to the median
    number of points the curves have inside the overlap (capped at 5000).
    """
    if len(curves) < 2:
        raise ValueError("Select at least two traces to average.")
    warnings: list[str] = []
    prepared = []
    for k, (x, y) in enumerate(curves, 1):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        segs = monotonic_segments(x)
        if len(segs) > 1:
            a, b, _ = segs[0]
            x, y = x[a:b], y[a:b]
            warnings.append(f"Curve {k} doubles back in x — only its first monotonic segment was averaged.")
        o = np.argsort(x, kind="stable")
        x, y = x[o], y[o]
        x, i = np.unique(x, return_index=True)
        y = y[i]
        if x.size < 2:
            raise ValueError(f"Curve {k} has fewer than two distinct x values.")
        prepared.append((x, y))
    lo = max(float(c[0][0]) for c in prepared)
    hi = min(float(c[0][-1]) for c in prepared)
    if not hi > lo:
        raise ValueError("The selected traces do not overlap in x.")
    full = min(float(c[0][0]) for c in prepared), max(float(c[0][-1]) for c in prepared)
    if (hi - lo) < 0.9 * (full[1] - full[0]):
        warnings.append(f"Averaged only over the common x range {fmt(lo)} – {fmt(hi)} "
                        f"(the traces span {fmt(full[0])} – {fmt(full[1])}).")
    if not n_points:
        counts = [int(np.sum((c[0] >= lo) & (c[0] <= hi))) for c in prepared]
        n_points = int(min(5000, max(2, np.median(counts))))
    grid = np.linspace(lo, hi, int(n_points))
    Y = np.vstack([np.interp(grid, x, y) for x, y in prepared])
    mean = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1)
    return AverageResult(grid, mean, sd, np.full(grid.size, len(prepared), float), lo, hi, warnings)


# --------------------------------------------------------------------------- legend grouping

_NUM_SUFFIX = re.compile(r"^(?P<prefix>.*?)(?P<num>[-+]?\d+(?:[.,]\d+)?)\s*(?P<unit>[^\d\s]{0,8})$")


def label_key(label: str) -> tuple[str, str, float] | None:
    """Split 'Cycle 12' / 'NiTi 450 °C' into (prefix, unit-suffix, number); None if no number."""
    m = _NUM_SUFFIX.match(str(label).strip())
    if not m:
        return None
    try:
        num = float(m.group("num").replace(",", "."))
    except ValueError:
        return None
    return m.group("prefix").strip(" _-=:#"), m.group("unit"), num


def legend_groups(labels: list[str], min_size: int = 13) -> list[list[int]]:
    """Groups (lists of indices) of at least ``min_size`` labels that share a common
    prefix and differ only by a trailing number (e.g. 'Cycle 1' … 'Cycle 50')."""
    by: dict[tuple[str, str], list[int]] = {}
    for i, lab in enumerate(labels):
        k = label_key(lab)
        if k is not None and k[0]:
            by.setdefault((k[0], k[1]), []).append(i)
    return [v for v in by.values() if len(v) >= min_size]


# --------------------------------------------------------------------------- label placement


def _overlap(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Pairwise overlap areas (K, M) of boxes (K, 4) and others (M, 4) as x0, y0, x1, y1."""
    if others.size == 0:
        return np.zeros((boxes.shape[0], 0))
    w = np.minimum(boxes[:, None, 2], others[None, :, 2]) - np.maximum(boxes[:, None, 0], others[None, :, 0])
    h = np.minimum(boxes[:, None, 3], others[None, :, 3]) - np.maximum(boxes[:, None, 1], others[None, :, 1])
    return np.clip(w, 0, None) * np.clip(h, 0, None)


def _candidates(max_r: float) -> list[tuple[float, float]]:
    out = [(1.0, 1.0, 0.0)]  # (cos, sin, radius) – default up-right
    radii = [6, 12, 20, 30, 42, 56, 72, 90, 110, 135, 165]
    angles = [45, 90, 0, 135, 315, 270, 180, 225, 67.5, 22.5, 112.5, 157.5, 337.5, 292.5, 247.5, 202.5]
    for r in radii:
        if r > max_r:
            break
        for a in angles:
            out.append((math.cos(math.radians(a)), math.sin(math.radians(a)), float(r)))
    return out


def place_labels(anchors, sizes, bounds=None, priorities=None, *, pad: float = 2.0,
                 marker_px: float = 4.0, max_radius: float = 165.0) -> np.ndarray:
    """Non-overlapping label positions in display (pixel) coordinates.

    ``anchors`` (N, 2) points the labels belong to, ``sizes`` (N, 2) label width and
    height, ``bounds`` (N, 4) x0, y0, x1, y1 box each label should stay inside (its
    axes), ``priorities`` (higher = placed first, i.e. closest to its anchor).
    Greedy: each label tries candidate positions on rings of growing radius around
    its anchor and takes the first that overlaps neither placed labels nor marker
    points; if none is free, the least-overlapping candidate. Returns (N, 2)
    offsets of each label's lower-left corner relative to its anchor (pixels).
    """
    A = np.asarray(anchors, float).reshape(-1, 2)
    S = np.asarray(sizes, float).reshape(-1, 2) + 2 * pad
    n = A.shape[0]
    out = np.zeros((n, 2))
    if n == 0:
        return out
    B = None if bounds is None else np.asarray(bounds, float).reshape(-1, 4)
    P = np.zeros(n) if priorities is None else np.asarray(priorities, float)
    order = sorted(range(n), key=lambda i: (-P[i], A[i, 0], -A[i, 1]))
    marks = np.column_stack([A[:, 0] - marker_px, A[:, 1] - marker_px, A[:, 0] + marker_px, A[:, 1] + marker_px])
    cand = np.array(_candidates(max_radius))
    placed: list[np.ndarray] = []
    for i in order:
        w, h = S[i]
        c, s_, r = cand[:, 0], cand[:, 1], cand[:, 2]
        # centre of the box: the box edge is ~r away from the anchor in direction (c, s)
        cx = A[i, 0] + c * (r + w / 2)
        cy = A[i, 1] + s_ * (r + h / 2)
        cx[0], cy[0] = A[i, 0] + 4 + w / 2, A[i, 1] + 4 + h / 2  # default: up-right of the marker
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        others = np.vstack(placed) if placed else np.zeros((0, 4))
        cost = _overlap(boxes, others).sum(axis=1) * 4.0
        cost += _overlap(boxes, marks).sum(axis=1)
        if B is not None and np.all(np.isfinite(B[i])):
            x0, y0, x1, y1 = B[i]
            out_w = np.clip(x0 - boxes[:, 0], 0, None) + np.clip(boxes[:, 2] - x1, 0, None)
            out_h = np.clip(y0 - boxes[:, 1], 0, None) + np.clip(boxes[:, 3] - y1, 0, None)
            cost += (out_w * h + out_h * w) * 2.0
        free = np.flatnonzero(cost <= 1e-9)
        k = int(free[0]) if free.size else int(np.argmin(cost + r * 1e-3))
        placed.append(boxes[k])
        out[i] = (boxes[k, 0] + pad - A[i, 0], boxes[k, 1] + pad - A[i, 1])
    return out


# --------------------------------------------------------------------------- peaks / fit / stats


@dataclass
class Peak:
    index: int
    x: float
    y: float
    prominence: float
    fwhm: float                 # in x units (nan if undefined)
    left_x: float               # half-height crossing points (for drawing)
    right_x: float
    half_y: float


def find_peaks_xy(x, y, *, prominence: float | None = None, min_distance: float | None = None,
                  sign: str = "peaks", rel_height: float = 0.5) -> list[Peak]:
    """Find peaks (``sign='peaks'``) or valleys (``'valleys'``) of y(x).

    ``prominence`` in y units (default 5 % of the y range), ``min_distance`` in x
    units (converted to samples using the median |dx|). FWHM is measured at
    ``rel_height`` of the prominence and returned in x units.
    """
    from scipy.signal import find_peaks, peak_widths

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return []
    s = -1.0 if sign.startswith("valley") else 1.0
    yy = s * y
    if prominence is None or prominence <= 0:
        rng = float(np.nanmax(y) - np.nanmin(y))
        prominence = 0.05 * rng if rng > 0 else None
    kw: dict = {"prominence": prominence}
    if min_distance and min_distance > 0:
        step = float(np.median(np.abs(np.diff(x)))) or 1.0
        kw["distance"] = max(1, int(round(min_distance / step)))
    idx, props = find_peaks(yy, **kw)
    if idx.size == 0:
        return []
    widths, heights, lips, rips = peak_widths(yy, idx, rel_height=rel_height)
    n = np.arange(x.size)
    lx = np.interp(lips, n, x)
    rx = np.interp(rips, n, x)
    out = []
    for k, i in enumerate(idx):
        out.append(Peak(int(i), float(x[i]), float(y[i]), float(props["prominences"][k]),
                        float(abs(rx[k] - lx[k])), float(lx[k]), float(rx[k]), float(s * heights[k])))
    return out


@dataclass
class LinearFit:
    slope: float
    intercept: float
    r2: float
    n: int
    xmin: float
    xmax: float


def linear_fit(x, y, xmin: float | None = None, xmax: float | None = None) -> LinearFit:
    """Least-squares straight line y = slope·x + intercept over the span."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    lo = -np.inf if xmin is None else min(xmin, xmax if xmax is not None else np.inf)
    hi = np.inf if xmax is None else max(xmax, xmin if xmin is not None else -np.inf)
    m &= (x >= lo) & (x <= hi)
    xs, ys = x[m], y[m]
    if xs.size < 2 or np.ptp(xs) == 0:
        raise ValueError("Need at least two distinct x values in the span for a fit.")
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return LinearFit(float(slope), float(intercept), r2, int(xs.size), float(xs.min()), float(xs.max()))


def span_stats(x, y, xmin: float | None = None, xmax: float | None = None) -> dict[str, float]:
    """min / max / mean / std of y (and x at min/max) for points inside the span."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if xmin is not None and xmax is not None:
        lo, hi = sorted((xmin, xmax))
        m &= (x >= lo) & (x <= hi)
    if not m.any():
        raise ValueError("The span contains no data.")
    xs, ys = x[m], y[m]
    i0, i1 = int(np.argmin(ys)), int(np.argmax(ys))
    return {"n": int(ys.size), "min": float(ys[i0]), "x_at_min": float(xs[i0]),
            "max": float(ys[i1]), "x_at_max": float(xs[i1]), "mean": float(ys.mean()),
            "std": float(ys.std(ddof=1)) if ys.size > 1 else 0.0, "range": float(ys[i1] - ys[i0])}


# --------------------------------------------------------------------------- cursor


def nearest_point(pixel_sets: list[np.ndarray], target: tuple[float, float],
                  max_dist: float = np.inf) -> tuple[int, int, float] | None:
    """Nearest point to ``target`` (pixels) over several ``(N, 2)`` pixel arrays.

    Returns ``(set_index, point_index, distance_px)`` or ``None`` if nothing is
    within ``max_dist`` pixels. NaN points are ignored.
    """
    best: tuple[int, int, float] | None = None
    tx, ty = target
    for k, pts in enumerate(pixel_sets):
        if pts is None or len(pts) == 0:
            continue
        d2 = (pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2
        if not np.isfinite(d2).any():
            continue
        i = int(np.nanargmin(d2))
        d = math.sqrt(float(d2[i]))
        if d <= max_dist and (best is None or d < best[2]):
            best = (k, i, d)
    return best


# --------------------------------------------------------------------------- display decimation


def _minmax_decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> np.ndarray:
    """Indices (sorted) keeping per-bucket extrema of x and y plus first/last."""
    n = x.size
    if n <= max_points:
        return np.arange(n)
    buckets = max(1, max_points // 4)
    per = int(math.ceil(n / buckets))
    buckets = int(math.ceil(n / per))
    pad = buckets * per - n

    def arg(a: np.ndarray, fn, fill):
        a2 = np.where(np.isfinite(a), a, fill)
        a2 = np.concatenate([a2, np.full(pad, fill)]).reshape(buckets, per)
        return fn(a2, axis=1) + np.arange(buckets) * per

    idx = np.concatenate([
        arg(y, np.argmin, np.inf), arg(y, np.argmax, -np.inf),
        arg(x, np.argmin, np.inf), arg(x, np.argmax, -np.inf), [0, n - 1]])
    idx = np.unique(np.clip(idx, 0, n - 1))
    return idx


def display_indices(x, y, xlim: tuple[float, float] | None = None,
                    ylim: tuple[float, float] | None = None,
                    max_points: int = 20000) -> np.ndarray:
    """Indices of points to *draw* (never used for analysis).

    Keeps only points inside the current view (plus their neighbours so lines run
    to the frame edge), then min/max-decimates down to ``max_points``. A ``-1``
    entry marks a break where a NaN should be inserted (non-contiguous runs).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n <= max_points:
        return np.arange(n)
    if xlim is not None and ylim is not None:
        x0, x1 = sorted(xlim)
        y0, y1 = sorted(ylim)
        mx, my = 0.02 * (x1 - x0), 0.02 * (y1 - y0)
        vis = (x >= x0 - mx) & (x <= x1 + mx) & (y >= y0 - my) & (y <= y1 + my)
        vis[1:] |= vis[:-1].copy()
        vis[:-1] |= vis[1:].copy()
        sub = np.flatnonzero(vis)
        if sub.size == 0:
            return sub
    else:
        sub = np.arange(n)
    if sub.size > max_points:
        keep = _minmax_decimate(x[sub], y[sub], max_points)
        gaps = np.flatnonzero(np.diff(sub) > 1)
        keep = np.unique(np.concatenate([keep, gaps, gaps + 1]))
        sel = sub[keep]
    else:
        sel = sub
    jumps = np.flatnonzero(np.diff(sel) > 1)
    if jumps.size and sub.size != n:
        # break lines only where the *visible subset* was non-contiguous
        orig_gap = np.zeros(sel.size - 1, bool)
        gap_starts = set(sub[np.flatnonzero(np.diff(sub) > 1)].tolist())
        for j in jumps:
            if int(sel[j]) in gap_starts:
                orig_gap[j] = True
        if orig_gap.any():
            sel = np.insert(sel, np.flatnonzero(orig_gap) + 1, -1)
    return sel


def take_with_breaks(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``a[idx]`` where ``idx == -1`` yields NaN (line break)."""
    a = np.asarray(a, float)
    out = a[np.where(idx < 0, 0, idx)] if a.size else np.zeros(idx.size)
    out = out.astype(float, copy=True)
    out[idx < 0] = np.nan
    return out


def fmt(v: float, sig: int = 6) -> str:
    """Compact number formatting for read-outs."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    try:
        return f"{float(v):.{sig}g}"
    except (TypeError, ValueError):
        return str(v)
