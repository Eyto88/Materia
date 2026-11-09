"""DSC analysis for shape memory alloys (transformation temperatures and latent heats).

Conventions (ASTM F2004 / ISO 11357 style):

* **Onset / endset** = intersection of the extrapolated baseline with the tangent drawn at the
  inflection point (steepest slope) of the leading / trailing edge of the peak.
* **Enthalpy** ΔH = ∫ (q − q_baseline) dt. On a temperature axis dt = dT / β with β the
  (measured or programmed) scan rate in K/s, so ΔH [J/g] = ∫ (q − q_b) [W/g] dT [K] / β [K/s].
  Enthalpies are reported as positive magnitudes; ``DSCPeak.endothermic`` gives the sign.
* Heating → endothermic reverse transformation (B19'→B2: As, Ap, Af; with an R-phase two-step
  B19'→R→B2 the lower peak is R'). Cooling → exothermic forward transformation(s)
  (B2→R: Rs, Rp, Rf; B2/R→B19': Ms, Mp, Mf).
* All characteristic temperatures of a peak are given in *process order*: ``start_T`` →
  ``onset_T`` → ``peak_T`` → ``endset_T`` → ``end_T``. On cooling this means decreasing T, so
  ``onset_T`` is Ms (or Rs) and ``endset_T`` is Mf (or Rf).

Heat flow may be specific (W/g, mW/mg, …) or absolute (mW, needs ``mass_mg`` to give J/g).
Sign convention (endo up / exo up) is a parameter or auto-detected from the dominant deviation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from . import units as U
from .analysis_common import linear_fit, smooth, trapz

__all__ = [
    "DSCPeak", "DSCSegments", "DSCSegmentResult", "DSCAnalysis",
    "split_dsc_segments", "analyse_dsc_segment", "analyse_dsc", "name_sma_peaks",
    "delta_T_ad_estimate", "detect_endo_up", "PeakList", "segment_labels", "select_segments",
    "dsc_segment_datasets", "estimate_noise", "DEFAULT_RATE_K_PER_MIN",
]

DEFAULT_RATE_K_PER_MIN = 10.0


# =========================================================================== segments
@dataclass
class DSCSegments:
    """Heating / cooling masks of a DSC run plus the list of dynamic segments.

    ``segments`` is a list of ``(start, stop, direction)`` with ``stop`` exclusive and direction
    +1 (heating) or −1 (cooling). Unpacks as ``heating, cooling = split_dsc_segments(T)``.
    """
    heating: np.ndarray
    cooling: np.ndarray
    segments: list[tuple[int, int, int]] = field(default_factory=list)

    def __iter__(self):
        yield self.heating
        yield self.cooling


def split_dsc_segments(T, t=None, *, min_rate_K_per_min: float = 0.5, smooth_points: int | None = None,
                       min_points: int = 20, min_span_K: float = 5.0) -> DSCSegments:
    """Split a DSC run into heating and cooling segments from the sign of dT/dt.

    T: temperature (°C or K). t: time (s); if omitted the sample index is used and the
    isotherm threshold is 20 % of the 90th percentile of |dT/di|.
    Isothermal holds (|dT/dt| below ``min_rate_K_per_min``), very short runs
    (< ``min_points``) and runs spanning less than ``min_span_K`` are ignored.
    """
    T = np.asarray(T, float)
    n = T.size
    heating = np.zeros(n, bool)
    cooling = np.zeros(n, bool)
    if n < 3:
        return DSCSegments(heating, cooling, [])
    Tf = np.where(np.isfinite(T), T, np.nan)
    if np.isnan(Tf).any():
        good = np.isfinite(Tf)
        Tf = np.interp(np.arange(n), np.flatnonzero(good), Tf[good])
    if t is None:
        tt = np.arange(n, dtype=float)
    else:
        tt = np.asarray(t, float)
    w = smooth_points or max(5, min(51, n // 200 * 2 + 5))
    # light smoothing for the rate, then a median filter on the sign: removes noise-induced flips
    # without smearing the ramp/isotherm boundaries
    Ts = smooth(Tf, 5, polyorder=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.gradient(Ts, tt)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    if t is None:
        thr = 0.2 * float(np.nanpercentile(np.abs(rate), 90))
    else:
        thr = min_rate_K_per_min / 60.0
    sgn = np.where(rate > thr, 1, np.where(rate < -thr, -1, 0))
    sgn = median_filter(sgn, size=w | 1, mode="nearest")
    segs: list[tuple[int, int, int]] = []
    i = 0
    while i < n:
        j = i
        while j < n and sgn[j] == sgn[i]:
            j += 1
        if sgn[i] != 0 and (j - i) >= min_points and np.ptp(Tf[i:j]) >= min_span_K:
            segs.append((i, j, int(sgn[i])))
            (heating if sgn[i] > 0 else cooling)[i:j] = True
        i = j
    return DSCSegments(heating, cooling, segs)


# =========================================================================== peaks
@dataclass
class DSCPeak:
    """One DSC transformation peak. Temperatures in °C (same as the input T axis)."""
    direction: str                 # "heating" | "cooling"
    start_T: float                 # integration limit where the transformation begins (process order)
    onset_T: float                 # tangent-construction onset (As, Ms, Rs …)
    peak_T: float                  # extremum of the excess heat flow (Ap, Mp, Rp …)
    endset_T: float                # tangent-construction endset (Af, Mf, Rf …)
    end_T: float                   # integration limit where the transformation ends
    enthalpy_J_per_g: float        # |ΔH| in J/g (nan if heat flow is absolute and no mass given)
    enthalpy_J: float = float("nan")   # |ΔH| in J when heat flow is absolute (mW)
    endothermic: bool = True
    peak_height: float = float("nan")  # excess heat flow at the peak, input hf units (>0)
    peak_hf: float = float("nan")      # raw heat flow value at the peak, input hf units
    fwhm_K: float = float("nan")
    onset_hf: float = float("nan")     # baseline value at onset (for markers), input hf units
    endset_hf: float = float("nan")
    phase: str = ""                    # "A", "M", "R", "R'", "P3" …
    tangent_onset: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    tangent_endset: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    region_T: np.ndarray | None = field(default=None, repr=False)
    region_hf: np.ndarray | None = field(default=None, repr=False)
    region_baseline: np.ndarray | None = field(default=None, repr=False)
    heating_rate_K_per_min: float = float("nan")
    truncated: bool = False            # region touches the data edge (incomplete transformation)
    area_frac: float = float("nan")    # |ΔH| / |ΔH| of the largest peak of the segment
    group: int = 0                     # peaks sharing one integration region (overlapping) share a group
    manual_limits: bool = False        # integration limits given by the user
    parts: list["DSCPeak"] | None = field(default=None, repr=False)   # combined peak: its components

    # SMA labels ---------------------------------------------------------------
    @property
    def label_start(self) -> str:
        return f"{self.phase}s" if self.phase else "onset"

    @property
    def label_peak(self) -> str:
        return f"{self.phase}p" if self.phase else "peak"

    @property
    def label_end(self) -> str:
        return f"{self.phase}f" if self.phase else "endset"

    @property
    def temperatures(self) -> dict[str, float]:
        """{'As': onset, 'Ap': peak, 'Af': endset} with this peak's phase letter."""
        return {self.label_start: self.onset_T, self.label_peak: self.peak_T,
                self.label_end: self.endset_T}


def name_sma_peaks(peaks: list[DSCPeak], direction: str) -> list[DSCPeak]:
    """Assign SMA phase letters to peaks given in process order (in place, also returned).

    heating: 1 peak → A; 2 peaks → R' (B19'→R, lower T) then A (R→B2); extra → P3, P4…
    cooling: 1 peak → M; 2 peaks → R (B2→R, higher T) then M (R→B19'); extra → P3, P4…
    """
    if direction == "heating":
        order = sorted(range(len(peaks)), key=lambda k: peaks[k].peak_T)
        names = {1: ["A"], 2: ["R'", "A"]}.get(len(peaks))
    else:
        order = sorted(range(len(peaks)), key=lambda k: -peaks[k].peak_T)
        names = {1: ["M"], 2: ["R", "M"]}.get(len(peaks))
    if names is None:
        names = [f"P{k + 1}" for k in range(len(peaks))]
    for k, idx in enumerate(order):
        peaks[idx].phase = names[k]
    return peaks


def _hf_scale(hf_unit: str, mass_mg: float | None) -> tuple[float, bool]:
    """Return (k, specific) such that q = hf / k is W/g (specific) or mW (absolute)."""
    u = U.normalize_symbol(hf_unit)
    if u in U.units_for("specific_heat_flow"):
        return float(U.convert(1.0, "W/g", u, "specific_heat_flow")), True
    if u in U.units_for("heat_flow"):
        per_mW = float(U.convert(1.0, "mW", u, "heat_flow"))
        if mass_mg and mass_mg > 0:
            return per_mW * float(mass_mg), True   # (mW/mg == W/g)
        return per_mW, False
    raise ValueError(f"Heat-flow unit '{hf_unit}' not recognised (use W/g, mW/mg, mW …)")


def _edge_baseline(T, q, frac: float = 0.06) -> tuple[float, float]:
    n = T.size
    k = max(3, int(frac * n))
    idx = np.r_[0:k, n - k:n]
    f = linear_fit(T[idx], q[idx])
    return f.slope, f.intercept


def detect_endo_up(T, hf, direction: str) -> bool:
    """Guess the sign convention: True if endothermic effects point up (positive).

    Uses the dominant deviation from an edge-anchored linear baseline: on heating the
    dominant effect is endothermic, on cooling exothermic.
    """
    T = np.asarray(T, float)
    q = np.asarray(hf, float)
    m = np.isfinite(T) & np.isfinite(q)
    T, q = T[m], q[m]
    o = np.argsort(T, kind="stable")
    T, q = T[o], q[o]
    a, b = _edge_baseline(T, q)
    r = smooth(q - (a * T + b), max(5, T.size // 100))
    positive_dominant = float(np.max(r)) > -float(np.min(r))
    return positive_dominant if direction == "heating" else not positive_dominant


def _regions(e_s: np.ndarray, thr: float, peaks: np.ndarray) -> list[tuple[int, int, list[int]]]:
    """Contiguous regions where e_s > thr around peaks: [(lo, hi, [peak indices])]."""
    regs: list[tuple[int, int, list[int]]] = []
    n = e_s.size
    for p in sorted(peaks):
        if regs and regs[-1][0] <= p <= regs[-1][1]:
            regs[-1][2].append(int(p))
            continue
        lo = p
        while lo > 0 and e_s[lo - 1] > thr:
            lo -= 1
        hi = p
        while hi < n - 1 and e_s[hi + 1] > thr:
            hi += 1
        regs.append((lo, hi, [int(p)]))
    return regs


class PeakList(list):
    """``list[DSCPeak]`` returned by :func:`analyse_dsc_segment` with extra diagnostics.

    ``warnings`` (human-readable, e.g. truncated peak), ``notes`` (informational, e.g. rejected
    noise peaks), ``rejected`` (peaks dropped by the area/width filters), ``combined`` (one peak per
    overlapping group when ``overlap="combined"``) and ``params`` (auto-chosen smoothing, noise …).
    Compares equal to a plain list, so ``peaks == []`` still works.
    """

    def __init__(self, items=(), *, warnings=None, notes=None, rejected=None, combined=None, params=None):
        super().__init__(items)
        self.warnings: list[str] = list(warnings or [])
        self.notes: list[str] = list(notes or [])
        self.rejected: list[DSCPeak] = list(rejected or [])
        self.combined: list[DSCPeak] = list(combined or [])
        self.params: dict = dict(params or {})


def estimate_noise(y) -> float:
    """Robust white-noise σ of ``y``: MAD of the residual of a 5-point quadratic Savitzky–Golay
    high-pass (residual std = 0.717 σ for white noise)."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 7:
        return 0.0
    r = y - smooth(y, 5, polyorder=2)
    return 1.4826 * float(np.median(np.abs(r - np.median(r)))) / 0.717


def _auto_smooth_K(Ta, q, pol, ppk, noise, base_K: float = 1.0, max_K: float = 5.0) -> float:
    """Adaptive smoothing window (K): large enough that the smoothed noise is ≲ 1 % of the signal
    excursion (σ_s ≈ 1.5 σ/√w for a quadratic SG window of w points), at least ``base_K`` and at
    most min(``max_K``, 0.35 × the estimated FWHM of the dominant peak)."""
    if noise <= 0 or Ta.size < 20:
        return base_K
    heavy = smooth(q, max(5, int(round(3.0 * ppk))), polyorder=2)
    a, b = _edge_baseline(Ta, heavy)
    e = pol * (heavy - (a * Ta + b))
    A = float(np.max(e))
    if A <= 0:
        return base_K
    w_pts = (150.0 * noise / A) ** 2
    w_K = w_pts / ppk
    imax = int(np.argmax(e))
    above = e >= 0.5 * A
    lo = imax
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = imax
    while hi < e.size - 1 and above[hi + 1]:
        hi += 1
    fwhm = float(Ta[hi] - Ta[lo]) if hi > lo else max_K
    return float(min(max(base_K, w_K), max(base_K, min(max_K, 0.35 * fwhm))))


def _edge_check(Ta, q_s, noise_s, a: float, b: float, frac: float = 0.3) -> list[tuple[str, float]]:
    """Detect a signal that is still changing steeply at an edge of the segment.

    In the outer 5 % of the T span at each end a straight line is fitted; if its slope differs from
    the baseline slope ``a`` by more than max(10 × smoothed noise, ``frac`` × the detrended signal
    range) over that window, the transformation is probably cut off by the scan range.
    Returns [("low"|"high", T_edge)].
    """
    out = []
    span = float(Ta[-1] - Ta[0])
    if span <= 0 or Ta.size < 40:
        return out
    w = 0.05 * span
    det = q_s - (a * Ta + b)
    thr = max(10.0 * noise_s, frac * float(np.ptp(det)))
    for side, m in (("low", Ta <= Ta[0] + w), ("high", Ta >= Ta[-1] - w)):
        if m.sum() < 10:
            continue
        try:
            f = linear_fit(Ta[m], q_s[m])
        except ValueError:
            continue
        rise = (f.slope - a) * w
        # signal rising (high end) / falling into the data (low end) in the peak direction
        if abs(rise) > thr:
            out.append((side, float(Ta[0] if side == "low" else Ta[-1])))
    return out


def analyse_dsc_segment(T, hf, heating_rate_K_per_min: float | None = None, mass_mg: float | None = None,
                        direction: str = "auto", baseline: str = "linear", prominence: float | None = None,
                        *, endo_up: bool | None = None, hf_unit: str = "W/g", time=None,
                        smooth_K: float | None = None, threshold_frac: float = 0.02,
                        min_peak_height_frac: float = 0.05, name_peaks: bool = True,
                        min_area_frac: float = 0.05, min_width_K: float = 1.0, overlap: str = "split",
                        limits: list[tuple[float, float]] | None = None) -> PeakList:
    """Analyse one heating or cooling DSC segment.

    Parameters
    ----------
    T, hf : temperature (°C) and heat flow arrays of a single dynamic segment.
    heating_rate_K_per_min : scan rate magnitude β. If ``time`` (s) is given the rate is measured
        from the data instead (a linear fit of T vs t) and this value is only a fallback.
    mass_mg : sample mass; required to obtain J/g when ``hf_unit`` is absolute (mW).
    direction : "heating", "cooling" or "auto" (from time order or T[-1] − T[0]).
    baseline : "linear" — straight line between the curve at the integration limits of each
        transformation (overlapping peaks share one line); "fit" — one least-squares line through
        all points outside the peaks; "none" — zero baseline.
    prominence : minimum peak prominence in the input hf units (default: auto, 5 % of the
        largest excess or 5 × noise).
    endo_up : sign convention; ``None`` = auto-detect (:func:`detect_endo_up`).
    smooth_K : Savitzky–Golay window (K) for peak detection; ``None`` = adaptive to the noise level
        (noise from the MAD of a high-pass residual; ≥ 1 K, ≤ 5 K / 0.35 FWHM).
    threshold_frac : fraction of the peak height that defines the extent of a transformation.
    min_peak_height_frac : candidate peaks smaller than this fraction of the largest are ignored.
    min_area_frac, min_width_K : peaks whose |ΔH| < ``min_area_frac`` × the largest |ΔH| or whose
        FWHM < max(``min_width_K``, ½ smoothing window) are rejected as noise (listed in ``notes``).
        SMA phase names are assigned to the remaining (dominant) peaks only.
    overlap : "split" — overlapping peaks share one baseline and are split at the minimum between
        them; "combined" — additionally report each overlapping group as one transformation
        (``PeakList.combined``, phase e.g. "R+M", with ``parts``).
    limits : optional user integration limits [(T_a, T_b), …] (°C), one window per peak; replaces
        the automatic peak detection (no rejection filters).

    Returns a :class:`PeakList` of peaks in process order (time order), named with SMA nomenclature.
    """
    T = np.asarray(T, float).ravel()
    hf = np.asarray(hf, float).ravel()
    tt = None if time is None else np.asarray(time, float).ravel()
    m = np.isfinite(T) & np.isfinite(hf)
    if tt is not None:
        m &= np.isfinite(tt)
        tt = tt[m]
    T, hf = T[m], hf[m]
    if T.size < 10:
        return PeakList(notes=["Segment has fewer than 10 valid points; skipped."])
    if direction == "auto":
        if tt is not None and tt.size > 2:
            direction = "heating" if linear_fit(tt, T).slope > 0 else "cooling"
        else:
            direction = "heating" if T[-1] >= T[0] else "cooling"
    if direction not in ("heating", "cooling"):
        raise ValueError("direction must be 'heating', 'cooling' or 'auto'")
    if baseline not in ("linear", "fit", "none"):
        raise ValueError("baseline must be 'linear', 'fit' or 'none'")
    if overlap not in ("split", "combined"):
        raise ValueError("overlap must be 'split' or 'combined'")

    # scan rate (K/s)
    beta = None
    if tt is not None and np.ptp(tt) > 0:
        beta = abs(linear_fit(tt, T).slope)
    if not beta:
        if not heating_rate_K_per_min:
            raise ValueError("Heating rate required (or pass a time column)")
        beta = abs(float(heating_rate_K_per_min)) / 60.0
    rate_K_min = beta * 60.0

    k_scale, specific = _hf_scale(hf_unit, mass_mg)
    o = np.argsort(T, kind="stable")
    Ta = T[o]
    q = hf[o] / k_scale
    hf_sorted = hf[o]
    if endo_up is None:
        endo_up = detect_endo_up(Ta, q, direction)
    pol = (1.0 if direction == "heating" else -1.0) * (1.0 if endo_up else -1.0)

    span = float(np.ptp(Ta)) or 1.0
    ppk = Ta.size / span
    noise = estimate_noise(q) + 1e-30
    auto_smooth = smooth_K is None or smooth_K <= 0
    sK = _auto_smooth_K(Ta, q, pol, ppk, noise) if auto_smooth else float(smooth_K)
    win = max(5, int(round(sK * ppk)))
    q_s = smooth(q, win, polyorder=2)
    noise_s = noise * 1.5 / np.sqrt(max(win, 1)) + 1e-30
    params = {"smoothing_K": sK, "smoothing_auto": auto_smooth, "noise_hf": noise * k_scale,
              "scan_rate_K_per_min": rate_K_min, "direction": direction, "endo_up": bool(endo_up),
              "baseline": baseline, "min_area_frac": min_area_frac, "min_width_K": min_width_K,
              "overlap": overlap}
    warnings: list[str] = []
    notes: list[str] = []
    nT = Ta.size

    # --- iterative global baseline + peak regions (automatic mode)
    a, b = _edge_baseline(Ta, q_s)
    groups: list[tuple[int, int, list[int], bool]] = []     # (lo, hi, peak idx, manual)
    manual_splits: dict[tuple[int, int], list[int]] = {}
    e_s = pol * (q_s - (a * Ta + b))
    hmax = 0.0
    if limits:
        for lo_T, hi_T in limits:
            lo_T, hi_T = sorted((float(lo_T), float(hi_T)))
            idx = np.flatnonzero((Ta >= lo_T) & (Ta <= hi_T))
            if idx.size < 5:
                warnings.append(f"Integration window {lo_T:.4g}–{hi_T:.4g} °C contains fewer than 5 points; "
                                "skipped.")
                continue
            ip = int(idx[0] + np.argmax(e_s[idx]))
            groups.append((int(idx[0]), int(idx[-1]), [ip], True))
        groups.sort(key=lambda g: g[0])
        params["limits"] = [[float(Ta[g[0]]), float(Ta[g[1]])] for g in groups]
        # windows that touch (shared limit) form one transformation region with one baseline, split at
        # the user's limit; genuinely overlapping windows are kept separate with a warning
        merged_m: list[list] = []
        for g in groups:
            if merged_m and abs(g[0] - merged_m[-1][1]) <= 1:
                merged_m[-1][3].append(merged_m[-1][1])
                merged_m[-1][1] = g[1]
                merged_m[-1][2] += g[2]
                continue
            if merged_m and g[0] < merged_m[-1][1]:
                warnings.append(f"Manual integration windows overlap ({Ta[merged_m[-1][0]]:.4g}–"
                                f"{Ta[merged_m[-1][1]]:.4g} and {Ta[g[0]]:.4g}–{Ta[g[1]]:.4g} °C): the overlap "
                                "is counted twice.")
            merged_m.append([g[0], g[1], list(g[2]), []])
        groups = [(g[0], g[1], g[2], True) for g in merged_m]
        manual_splits.update({(g[0], g[1]): g[3] for g in merged_m if g[3]})
    else:
        regs: list[tuple[int, int, list[int]]] = []
        for _ in range(4):
            e_s = pol * (q_s - (a * Ta + b))
            emax = float(np.max(e_s))
            if emax <= 0:
                regs = []
                break
            prom = prominence / k_scale if prominence else max(0.05 * emax, 5 * noise_s)
            pk, _props = signal.find_peaks(e_s, prominence=prom)
            if pk.size == 0:
                regs = []
                break
            hmax = float(np.max(e_s[pk]))
            pk = pk[e_s[pk] >= max(min_peak_height_frac * hmax, 3 * noise_s)]
            thr = max(threshold_frac * hmax, 3 * noise_s)
            regs = _regions(e_s, thr, pk)
            outside = np.ones(nT, bool)
            for lo, hi, _p in regs:
                marg = max(2.0, 0.3 * (Ta[hi] - Ta[lo]))
                outside &= ~((Ta >= Ta[lo] - marg) & (Ta <= Ta[hi] + marg))
            if outside.sum() < 6 or baseline == "none":
                break
            f = linear_fit(Ta[outside], q_s[outside])
            if abs(f.slope - a) * span < 1e-3 * hmax and abs(f.intercept - b) < 1e-3 * hmax:
                a, b = f.slope, f.intercept
                break
            a, b = f.slope, f.intercept
        e_s = pol * (q_s - (a * Ta + b))
        small = np.array([], int)
        if regs:
            cand, _ = signal.find_peaks(e_s, prominence=max(8 * noise_s, 0.01 * hmax))
            inside = np.zeros(nT, bool)
            for lo, hi, _p in regs:
                inside[lo:hi + 1] = True
            small = cand[~inside[cand]]
        if small.size:
            notes.append(f"{small.size} candidate {direction} peak(s) lower than {100 * min_peak_height_frac:.0f} % "
                         f"of the largest ignored (at {', '.join(f'{Ta[i]:.1f}' for i in small[:5])} °C).")
        # extend each region outward until the excess returns to the baseline (bounded), then merge
        # regions whose extended windows overlap: they share one baseline (no double counting)
        ext = []
        for lo, hi, plist in regs:
            width = Ta[hi] - Ta[lo]
            lo2, hi2 = lo, hi
            while lo2 > 0 and e_s[lo2 - 1] > 0 and Ta[lo] - Ta[lo2 - 1] < 0.5 * width + 1.0:
                lo2 -= 1
            while hi2 < nT - 1 and e_s[hi2 + 1] > 0 and Ta[hi2 + 1] - Ta[hi] < 0.5 * width + 1.0:
                hi2 += 1
            ext.append([lo2, hi2, list(plist)])
        ext.sort(key=lambda g: g[0])
        merged: list[list] = []
        for g in ext:
            if merged and g[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], g[1])
                merged[-1][2] += g[2]
            else:
                merged.append(g)
        groups = [(g[0], g[1], sorted(set(g[2])), False) for g in merged]
    params["global_baseline"] = [a * k_scale, b * k_scale]

    half = max(1, win // 2)

    def build(groups_) -> list[DSCPeak]:
        peaks_: list[DSCPeak] = []
        for gi, (lo2, hi2, plist, manual) in enumerate(groups_):
            if baseline == "linear":
                y_lo = float(np.mean(q_s[max(0, lo2 - half):lo2 + half + 1])) if not manual else float(q_s[lo2])
                y_hi = float(np.mean(q_s[max(0, hi2 - half):hi2 + half + 1])) if not manual else float(q_s[hi2])
                T_lo, T_hi = Ta[lo2], Ta[hi2]
                sl = (y_hi - y_lo) / (T_hi - T_lo) if T_hi > T_lo else 0.0
                ba, bb = sl, y_lo - sl * T_lo
            elif baseline == "fit":
                ba, bb = a, b
            else:
                ba, bb = 0.0, 0.0
            sl_ = slice(lo2, hi2 + 1)
            Tr = Ta[sl_]
            if Tr.size < 4:
                continue
            base_r = ba * Tr + bb
            e_r = pol * (q[sl_] - base_r)
            es_r = pol * (q_s[sl_] - base_r)
            dT_med = float(np.median(np.diff(Tr))) if Tr.size > 2 else 1.0
            dT_med = dT_med if dT_med > 0 else 1.0
            hh = float(np.max(es_r)) if es_r.size else 0.0
            above_all = np.flatnonzero(es_r >= 0.5 * hh) if hh > 0 else np.array([], int)
            fw = (Tr[above_all[-1]] - Tr[above_all[0]]) if above_all.size > 1 else sK
            wd = max(win, int(round(fw / 3.0 / dT_med)))
            if Tr.size > 7:
                es_r = smooth(e_r, wd, polyorder=3)
                de = smooth(e_r, wd, polyorder=3, deriv=1, delta=dT_med)
            else:
                de = np.gradient(es_r, Tr) if Tr.size > 2 else np.zeros_like(Tr)
            pl = sorted(p - lo2 for p in plist)
            bounds = [0]
            if (lo2, hi2) in manual_splits:
                bounds += [b_ - lo2 for b_ in manual_splits[(lo2, hi2)]]
            else:
                for p1, p2 in zip(pl[:-1], pl[1:]):
                    bounds.append(p1 + int(np.argmin(es_r[p1:p2 + 1])))
            bounds.append(Tr.size - 1)
            for kk, p in enumerate(pl):
                i0, i1 = bounds[kk], bounds[kk + 1]
                if i1 - i0 < 3:
                    continue
                seg = slice(i0, i1 + 1)
                area_K = trapz(e_r[seg], Tr[seg])
                dH = area_K / beta          # W/g·K / (K/s) = J/g ; or mW·K/(K/s) = mJ
                ip = i0 + int(np.argmax(es_r[seg]))
                Tp = Tr[ip]
                if 0 < ip < Tr.size - 1:
                    y0, y1, y2 = es_r[ip - 1], es_r[ip], es_r[ip + 1]
                    den = y0 - 2 * y1 + y2
                    if den < 0:
                        dx = 0.5 * (y0 - y2) / den
                        if abs(dx) <= 1:
                            Tp = Tr[ip] + dx * (Tr[ip + 1] - Tr[ip - 1]) / 2.0
                h = float(es_r[ip])
                il = i0 + int(np.argmax(de[i0:ip + 1])) if ip > i0 else i0
                ir = ip + int(np.argmin(de[ip:i1 + 1])) if i1 > ip else i1
                T_low = Tr[il] - es_r[il] / de[il] if de[il] > 0 else Tr[i0]
                T_high = Tr[ir] - es_r[ir] / de[ir] if de[ir] < 0 else Tr[i1]
                # tangents must not extrapolate beyond the integration window
                T_low = float(min(max(T_low, Tr[i0]), Tp))
                T_high = float(max(min(T_high, Tr[i1]), Tp))

                def to_hf(Tv, ev, ba=ba, bb=bb):
                    Tv = np.asarray(Tv, float)
                    return k_scale * (ba * Tv + bb + np.asarray(ev, float) / pol)

                tan_lo = None
                if de[il] > 0 and h > 0:
                    xs = np.array([T_low, T_low + 1.1 * h / de[il]])
                    tan_lo = (xs, to_hf(xs, de[il] * (xs - T_low)))
                tan_hi = None
                if de[ir] < 0 and h > 0:
                    xs = np.array([T_high + 1.1 * h / de[ir], T_high])
                    tan_hi = (xs, to_hf(xs, de[ir] * (xs - T_high)))
                above = np.flatnonzero(es_r[seg] >= 0.5 * h) if h > 0 else np.array([], int)
                fwhm = float(Tr[seg][above[-1]] - Tr[seg][above[0]]) if above.size else float("nan")
                if direction == "heating":
                    st, on, en, et = Tr[i0], T_low, T_high, Tr[i1]
                    tan_on, tan_en = tan_lo, tan_hi
                else:
                    st, on, en, et = Tr[i1], T_high, T_low, Tr[i0]
                    tan_on, tan_en = tan_hi, tan_lo
                trunc = bool((kk == 0 and lo2 == 0) or (kk == len(pl) - 1 and hi2 == nT - 1))
                peaks_.append(DSCPeak(
                    direction=direction, start_T=float(st), onset_T=float(on), peak_T=float(Tp),
                    endset_T=float(en), end_T=float(et),
                    enthalpy_J_per_g=float(abs(dH)) if specific else float("nan"),
                    enthalpy_J=float(abs(dH)) * 1e-3 if not specific else (
                        float(abs(dH)) * float(mass_mg) * 1e-3 if mass_mg else float("nan")),
                    endothermic=(direction == "heating"),
                    peak_height=float(h * k_scale), peak_hf=float(hf_sorted[lo2 + ip]),
                    fwhm_K=fwhm,
                    onset_hf=float(to_hf(on, 0.0)), endset_hf=float(to_hf(en, 0.0)),
                    tangent_onset=tan_on, tangent_endset=tan_en,
                    region_T=Tr[seg].copy(), region_hf=(q[sl_][seg] * k_scale).copy(),
                    region_baseline=(base_r[seg] * k_scale).copy(),
                    heating_rate_K_per_min=rate_K_min, truncated=trunc, group=gi, manual_limits=manual,
                ))
                peaks_[-1]._apex = lo2 + ip     # index into the sorted arrays (internal)
        big = max((abs(p.enthalpy_J_per_g if specific else p.enthalpy_J) for p in peaks_), default=0.0)
        for p in peaks_:
            v = abs(p.enthalpy_J_per_g if specific else p.enthalpy_J)
            p.area_frac = v / big if big > 0 else float("nan")
        return peaks_

    peaks = build(groups)
    rejected: list[DSCPeak] = []
    if not limits and peaks:
        w_min = max(min_width_K, 0.5 * sK)
        keep, drop = [], []
        for p in peaks:
            bad_area = np.isfinite(p.area_frac) and p.area_frac < min_area_frac
            bad_width = not np.isfinite(p.fwhm_K) or p.fwhm_K < w_min
            p._broad = bool(np.isfinite(p.fwhm_K) and p.fwhm_K > 0.5 * span)
            (drop if (bad_area or bad_width or p._broad) else keep).append(p)
        if drop:
            rejected = drop
            for p in drop:
                why = (f"FWHM {p.fwhm_K:.3g} K > half the scan range: baseline curvature, not a transformation"
                       if p._broad else f"{100 * p.area_frac:.1f} % of the largest peak area"
                       if p.area_frac < min_area_frac else f"FWHM {p.fwhm_K:.2g} K < {w_min:.2g} K")
                notes.append(f"Minor {direction} peak at {p.peak_T:.1f} °C ignored as noise ({why}).")
            apex_keep = {p._apex for p in keep}
            groups2 = []
            for lo2, hi2, plist, manual in groups:
                pk2 = [i for i in plist if any(abs(i - a_) <= max(2, win) for a_ in apex_keep)]
                if not pk2 and any(lo2 <= a_ <= hi2 for a_ in apex_keep):
                    pk2 = [a_ for a_ in apex_keep if lo2 <= a_ <= hi2]
                if pk2:
                    groups2.append((lo2, hi2, sorted(set(pk2)), manual))
            peaks = build(groups2)
            groups = groups2
    peaks.sort(key=lambda p: p.peak_T, reverse=(direction == "cooling"))
    if name_peaks:
        name_sma_peaks(peaks, direction)
    # combined transformations (overlapping groups)
    combined: list[DSCPeak] = []
    if overlap == "combined":
        by_g: dict[int, list[DSCPeak]] = {}
        for p in peaks:
            by_g.setdefault(p.group, []).append(p)
        for gi, parts in by_g.items():
            if len(parts) < 2:
                continue
            first, last = parts[0], parts[-1]          # process order
            big = max(parts, key=lambda p: p.peak_height)
            rT = np.concatenate([p.region_T for p in parts])
            oo = np.argsort(rT, kind="stable")
            c = DSCPeak(
                direction=direction, start_T=first.start_T, onset_T=first.onset_T, peak_T=big.peak_T,
                endset_T=last.endset_T, end_T=last.end_T,
                enthalpy_J_per_g=float(sum(p.enthalpy_J_per_g for p in parts)),
                enthalpy_J=float(sum(p.enthalpy_J for p in parts)), endothermic=first.endothermic,
                peak_height=big.peak_height, peak_hf=big.peak_hf, fwhm_K=float("nan"),
                onset_hf=first.onset_hf, endset_hf=last.endset_hf, phase="+".join(p.phase for p in parts),
                tangent_onset=first.tangent_onset, tangent_endset=last.tangent_endset,
                region_T=rT[oo], region_hf=np.concatenate([p.region_hf for p in parts])[oo],
                region_baseline=np.concatenate([p.region_baseline for p in parts])[oo],
                heating_rate_K_per_min=rate_K_min, truncated=any(p.truncated for p in parts), group=gi,
                parts=list(parts))
            combined.append(c)
    # warnings: truncated peaks / signal still changing at an edge
    for p in peaks:
        if p.truncated:
            warnings.append(f"{direction.capitalize()} peak {p.phase or ''} at {p.peak_T:.1f} °C touches the "
                            f"edge of the data: the transformation is truncated (incomplete) — ΔH and "
                            f"{'endset' if p is peaks[-1] else 'onset'} are underestimated.")
    if not limits:
        covered = [(p.region_T[0], p.region_T[-1]) for p in peaks if p.truncated]
        for side, T_edge in _edge_check(Ta, q_s, noise_s, a, b):
            if any(lo <= T_edge + 1e-9 and T_edge - 1e-9 <= hi for lo, hi in covered):
                continue
            warnings.append(f"Heat flow is still changing at the {side}-temperature end of the {direction} "
                            f"segment ({T_edge:.1f} °C): a transformation may be cut off by the scan range "
                            "(incomplete) — extend the temperature range or check the edge trim.")
    for p in peaks:
        if hasattr(p, "_apex"):
            del p._apex
    for p in rejected:
        for a_ in ("_apex", "_broad"):
            if hasattr(p, a_):
                delattr(p, a_)
    for p in peaks:
        if hasattr(p, "_broad"):
            del p._broad
    return PeakList(peaks, warnings=warnings, notes=notes, rejected=rejected, combined=combined,
                    params=params)


# =========================================================================== full run
@dataclass
class DSCSegmentResult:
    number: int
    direction: str
    indices: np.ndarray = field(repr=False)
    heating_rate_K_per_min: float
    peaks: list[DSCPeak]
    label: str = ""                         # "heating 1", "cooling 1", "heating 2" …
    combined: list[DSCPeak] = field(default_factory=list)
    rejected: list[DSCPeak] = field(default_factory=list)
    smoothing_K: float = float("nan")


@dataclass
class DSCAnalysis:
    """Result of :func:`analyse_dsc` for a whole heating/cooling run."""
    segments: list[DSCSegmentResult]
    endo_up: bool
    temperatures: dict[str, float]            # As, Af, Ap, Ms, Mf, Mp, Rs, Rf, Rp … (°C)
    enthalpies: dict[str, float]              # {"A": J/g, "M": J/g, …}
    hysteresis: dict[str, float]              # {"Ap − Mp": K, "Af − Ms": K, …}
    delta_T_ad: dict[str, float]              # adiabatic ΔT upper bound L/cp per phase (K)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)  # every parameter incl. auto-chosen ones (provenance)

    @property
    def peaks(self) -> list[DSCPeak]:
        return [p for s in self.segments for p in s.peaks]

    def rows(self) -> list[tuple[str, float, str]]:
        """Summary rows (quantity, value, unit) for tables/CSV."""
        rows: list[tuple[str, float, str]] = []
        for k, v in self.temperatures.items():
            rows.append((k, v, "°C"))
        for k, v in self.enthalpies.items():
            rows.append((f"ΔH {k}", v, "J/g"))
        for k, v in self.hysteresis.items():
            rows.append((f"Hysteresis {k}", v, "K"))
        for k, v in self.delta_T_ad.items():
            rows.append((f"ΔT_ad upper bound (L/c_p) {k}", v, "K"))
        return rows


def delta_T_ad_estimate(latent_J_per_g: float, cp_J_per_gK: float = 0.45, T_C: float | None = None,
                        T_transformation_C: float | None = None) -> float:
    """Theoretical UPPER BOUND of the adiabatic temperature change of a full transformation,
    ΔT_ad ≈ L / c_p (K). Measured elastocaloric ΔT of NiTi is typically much smaller (≈ 15–25 K)
    because of incomplete transformation, heat losses and hysteresis.

    If both ``T_C`` (operating temperature) and ``T_transformation_C`` (at which L was measured,
    e.g. the peak temperature) are given, the entropy-based form ΔT_ad ≈ T·ΔS/c_p with
    ΔS = L/T_tr is used: ΔT_ad = (T/T_tr)·L/c_p (temperatures in K).
    """
    dT = abs(float(latent_J_per_g)) / float(cp_J_per_gK)
    if T_C is not None and T_transformation_C is not None:
        dT *= (T_C + 273.15) / (T_transformation_C + 273.15)
    return dT


def segment_labels(segments) -> list[str]:
    """Labels 'heating 1', 'cooling 1', 'heating 2' … for ``(start, stop, direction)`` tuples."""
    n = {1: 0, -1: 0}
    out = []
    for *_r, d in segments:
        n[d] += 1
        out.append(f"{'heating' if d > 0 else 'cooling'} {n[d]}")
    return out


def select_segments(labels: list[str], which: str = "all") -> list[int]:
    """Indices of the segments selected by ``which``: 'all', 'heating 2', 'cooling 1',
    'last heating', 'last cooling', 'first heating', 'first cooling', 'heating' / 'cooling' (all of
    that direction). Unknown selections raise ``ValueError``."""
    w = (which or "all").strip().lower()
    if w == "all":
        return list(range(len(labels)))
    if w in ("heating", "cooling"):
        return [i for i, lb in enumerate(labels) if lb.startswith(w)]
    parts = w.split()
    if len(parts) == 2 and parts[0] in ("last", "first") and parts[1] in ("heating", "cooling"):
        idx = [i for i, lb in enumerate(labels) if lb.startswith(parts[1])]
        if not idx:
            return []
        return [idx[-1] if parts[0] == "last" else idx[0]]
    if w in labels:
        return [labels.index(w)]
    raise ValueError(f"Unknown DSC segment selection '{which}'")


def _rate_from_time(T, t) -> float:
    try:
        return abs(linear_fit(t, T).slope) * 60.0
    except ValueError:
        return float("nan")


def analyse_dsc(T, hf, heating_rate_K_per_min: float | None = None, mass_mg: float | None = None, *,
                time=None, hf_unit: str = "W/g", endo_up: bool | None = None, baseline: str = "linear",
                prominence: float | None = None, trim_K: float = 3.0, smooth_K: float | None = None,
                mode: str = "auto", summary_segment: str = "last", cp_J_per_gK: float = 0.45,
                threshold_frac: float = 0.02, min_area_frac: float = 0.05, min_width_K: float = 1.0,
                overlap: str = "split", limits: dict | None = None, segment: str = "all",
                rate_source: str | None = None, rate_tolerance: float = 0.15,
                dH_mismatch_tolerance: float = 0.20) -> DSCAnalysis:
    """Analyse a complete DSC run (any number of heating/cooling segments).

    mode: "auto" (split heating/cooling via :func:`split_dsc_segments`), "heating" / "cooling"
    (treat all data as one segment of that direction) or "single" (one segment, direction auto).
    trim_K: the first and last ``trim_K`` kelvin of every segment are dropped (start-up hooks).
    summary_segment: which heating / cooling segment feeds the summary ("first" or "last" —
    the second thermal cycle is the usual choice to remove thermal history).
    segment: analyse only some segments ('all', 'heating 2', 'last cooling' … see
    :func:`select_segments`).
    limits: manual integration windows per segment label, e.g. ``{"cooling 1": [(30, 5), (5, -35)]}``.
    Heating rate: measured from ``time`` when given (a warning names both values if it differs by
    more than ``rate_tolerance`` from ``heating_rate_K_per_min``); without time and without a rate
    the rate defaults to 10 K/min **with a warning** (``rate_source`` = "default" also warns).
    Other warnings: truncated peaks, heating vs cooling total ΔH differing by more than
    ``dH_mismatch_tolerance``, implausible ΔH (> 100 J/g or < 0.5 J/g).
    """
    T = np.asarray(T, float).ravel()
    hf = np.asarray(hf, float).ravel()
    tt = None if time is None else np.asarray(time, float).ravel()
    good = np.isfinite(T) & np.isfinite(hf)
    if int(good.sum()) < 10:
        raise ValueError(f"Not enough valid data points for a DSC analysis ({int(good.sum())} < 10).")
    notes: list[str] = []
    warnings: list[str] = []
    rate_in = float(heating_rate_K_per_min) if heating_rate_K_per_min else None
    if tt is not None and not np.isfinite(tt).any():
        tt = None
    if rate_in is None and tt is None:
        rate_in = DEFAULT_RATE_K_PER_MIN
        rate_source = "default"
    if rate_source == "default" and tt is None:
        warnings.append(f"Heating rate defaulted to {rate_in:g} K/min (no time column and no rate in the file): "
                        "ΔH scales with 1/β — enter the programmed rate.")
    if mode == "auto":
        segs = split_dsc_segments(T, tt).segments
        if not segs:
            d = "heating" if np.nanmean(np.diff(T)) >= 0 else "cooling"
            segs = [(0, T.size, 1 if d == "heating" else -1)]
            notes.append("No scan segments detected; data treated as a single segment.")
    elif mode in ("heating", "cooling", "single"):
        if mode == "single":
            d = 1 if (np.nanmean(np.diff(T)) >= 0) else -1
        else:
            d = 1 if mode == "heating" else -1
        segs = [(0, T.size, d)]
    else:
        raise ValueError("mode must be auto, heating, cooling or single")
    labels = segment_labels(segs)
    sel = select_segments(labels, segment)
    if not sel:
        raise ValueError(f"No '{segment}' segment in the data (segments: {', '.join(labels)})")
    segs = [segs[i] for i in sel]
    labels = [labels[i] for i in sel]

    # trimmed index sets
    seg_idx = []
    for (i0, i1, d) in segs:
        idx = np.arange(i0, i1)
        idx = idx[np.isfinite(T[idx]) & np.isfinite(hf[idx])]
        if idx.size == 0:
            seg_idx.append((idx, d))
            continue
        Ts = T[idx]
        lo, hi = np.nanmin(Ts), np.nanmax(Ts)
        keep = (Ts >= lo + trim_K) & (Ts <= hi - trim_K) if (hi - lo) > 4 * trim_K else np.ones(idx.size, bool)
        seg_idx.append((idx[keep], d))

    if endo_up is None:
        ref = next(((ix, d) for ix, d in seg_idx if d > 0 and ix.size > 10), None) or \
            next(((ix, d) for ix, d in seg_idx if ix.size > 10), seg_idx[0])
        k_scale, _ = _hf_scale(hf_unit, mass_mg)
        if ref[0].size > 10:
            endo_up = detect_endo_up(T[ref[0]], hf[ref[0]] / k_scale, "heating" if ref[1] > 0 else "cooling")
        else:
            endo_up = True
        notes.append(f"Sign convention auto-detected: {'endo up' if endo_up else 'exo up'}.")

    limits = limits or {}
    results: list[DSCSegmentResult] = []
    seg_params = {}
    for num, ((idx, d), label) in enumerate(zip(seg_idx, labels), start=1):
        direction = "heating" if d > 0 else "cooling"
        if tt is not None and idx.size > 2:
            meas = _rate_from_time(T[idx], tt[idx])
            if rate_in and np.isfinite(meas) and abs(meas - rate_in) > rate_tolerance * rate_in:
                warnings.append(f"Measured scan rate {meas:.3g} K/min ({label}, from the time column) differs from "
                                f"the header/entered rate {rate_in:.3g} K/min by {100 * (meas / rate_in - 1):+.0f} %; "
                                "the measured rate is used — check the time-column unit.")
        lim = limits.get(label)
        pks = analyse_dsc_segment(
            T[idx], hf[idx], rate_in, mass_mg, direction, baseline, prominence,
            endo_up=endo_up, hf_unit=hf_unit, time=None if tt is None else tt[idx],
            smooth_K=smooth_K, threshold_frac=threshold_frac, min_area_frac=min_area_frac,
            min_width_K=min_width_K, overlap=overlap, limits=lim)
        rate = pks[0].heating_rate_K_per_min if pks else (
            _rate_from_time(T[idx], tt[idx]) if tt is not None and idx.size > 2 else (rate_in or float("nan")))
        warnings += [f"[{label}] {w}" for w in pks.warnings]
        notes += [f"[{label}] {n}" for n in pks.notes]
        results.append(DSCSegmentResult(num, direction, idx, float(rate), list(pks), label=label,
                                        combined=list(pks.combined), rejected=list(pks.rejected),
                                        smoothing_K=float(pks.params.get("smoothing_K", float("nan")))))
        seg_params[label] = {k: pks.params.get(k) for k in ("smoothing_K", "smoothing_auto", "noise_hf",
                                                             "scan_rate_K_per_min", "limits")
                             if k in pks.params}
        seg_params[label]["peak_limits"] = {p.phase: [p.start_T, p.end_T] for p in pks}

    temps: dict[str, float] = {}
    enth: dict[str, float] = {}
    pick = (lambda lst: lst[-1]) if summary_segment == "last" else (lambda lst: lst[0])
    tot: dict[str, float] = {}
    for direction in ("heating", "cooling"):
        lst = [r for r in results if r.direction == direction and r.peaks]
        if not lst:
            continue
        r = pick(lst)
        for p in r.peaks:
            temps.update(p.temperatures)
            enth[p.phase] = p.enthalpy_J_per_g
        for c in r.combined:
            enth[f"{c.phase} (combined)"] = c.enthalpy_J_per_g
        tot[direction] = float(np.nansum([p.enthalpy_J_per_g for p in r.peaks]))
        if len(r.peaks) > 1:
            enth[f"total {direction}"] = tot[direction]
    hyst: dict[str, float] = {}
    if "Ap" in temps and "Mp" in temps:
        hyst["Ap − Mp"] = temps["Ap"] - temps["Mp"]
    if "Af" in temps and "Ms" in temps:
        hyst["Af − Ms"] = temps["Af"] - temps["Ms"]
    if "R'p" in temps and "Rp" in temps:
        hyst["R'p − Rp"] = temps["R'p"] - temps["Rp"]
    dTad = {}
    for ph, L in enth.items():
        if np.isfinite(L) and not ph.startswith("total") and not ph.endswith("(combined)"):
            dTad[ph] = delta_T_ad_estimate(L, cp_J_per_gK)
    # sanity checks
    h, c = tot.get("heating"), tot.get("cooling")
    if h and c and np.isfinite(h) and np.isfinite(c) and max(h, c) > 0:
        diff = abs(h - c) / max(h, c)
        if diff > dH_mismatch_tolerance:
            warnings.append(f"Heating and cooling total ΔH differ by {100 * diff:.0f} % ({h:.3g} vs {c:.3g} J/g; "
                            f"> {100 * dH_mismatch_tolerance:.0f} %): check overlapping peaks, integration limits, "
                            "baseline or an incomplete transformation.")
    for s in results:
        for p in s.peaks:
            v = p.enthalpy_J_per_g
            if np.isfinite(v) and v > 100:
                warnings.append(f"ΔH {p.phase} [{s.label}] = {v:.3g} J/g is implausibly large (> 100 J/g; NiTi ≈ "
                                "10–35 J/g): check the heating rate, mass and heat-flow unit.")
            elif np.isfinite(v) and v < 0.5:
                warnings.append(f"ΔH {p.phase} [{s.label}] = {v:.3g} J/g is implausibly small (< 0.5 J/g): check the "
                                "heating rate, mass and heat-flow unit.")
    params = {"heating_rate_K_per_min": rate_in, "heating_rate_source": rate_source or ("time column" if tt is not None
                                                                                       else "user"),
              "mass_mg": mass_mg, "hf_unit": hf_unit, "endo_up": bool(endo_up),
              "endo_up_auto": "Sign convention auto-detected" in " ".join(notes), "baseline": baseline,
              "prominence": prominence, "trim_K": trim_K, "smooth_K": smooth_K, "mode": mode,
              "summary_segment": summary_segment, "cp_J_per_gK": cp_J_per_gK, "threshold_frac": threshold_frac,
              "min_area_frac": min_area_frac, "min_width_K": min_width_K, "overlap": overlap,
              "limits": {k: [list(map(float, w)) for w in v] for k, v in limits.items()}, "segment": segment,
              "segments": seg_params}
    return DSCAnalysis(results, bool(endo_up), temps, enth, hyst, dTad, notes, warnings, params)


def dsc_segment_datasets(ds, T_col: str, t_col: str | None = None, *, kind: str | None = None) -> list:
    """Split a DSC Dataset into one Dataset per heating/cooling segment.

    Names are ``'<name> (heating 1)'``, ``'<name> (cooling 1)'`` …; ``metadata['segment']`` holds the
    label and, with a time column, ``heating_rate_K_per_min`` the measured rate of that segment
    (``heating_rate_source = 'measured (segment)'``).
    """
    T = ds.values(T_col)
    t = ds.values(t_col) if t_col else None
    segs = split_dsc_segments(T, t).segments
    out = []
    for (i0, i1, _d), label in zip(segs, segment_labels(segs)):
        mask = np.zeros(T.size, bool)
        mask[i0:i1] = True
        sub = ds.subset(mask, name=f"{ds.name} ({label})")
        if kind:
            sub.kind = kind
        sub.metadata["segment"] = label
        sub.metadata["derived_from"] = ds.name
        if t is not None and i1 - i0 > 2:
            r = _rate_from_time(T[i0:i1], t[i0:i1])
            if np.isfinite(r):
                sub.metadata["heating_rate_K_per_min"] = float(r)
                sub.metadata["heating_rate_source"] = "measured (segment)"
        out.append(sub)
    return out
