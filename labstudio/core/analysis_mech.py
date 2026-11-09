"""Monotonic mechanical test analysis (tension to failure, compression).

Units: strain in %, stress in MPa (canonical). Moduli are reported in GPa:
slope [MPa/%] × 100 / 1000 = slope / 10 GPa. Energy densities ∫σ dε use ε in mm/mm,
so MPa·(mm/mm) = MJ/m³ (1 MPa·% = 0.01 MJ/m³).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .analysis_common import LinearFit, linear_fit, smooth, trapz

__all__ = [
    "MechanicalResult", "TensileResult", "CompressionResult",
    "stress_strain", "tensile_properties", "compression_properties", "auto_modulus_fit",
    "compliance_correct", "estimate_compliance", "mechanical_sanity_warnings",
]


def compliance_correct(disp_mm, force_N, compliance_mm_per_N: float) -> np.ndarray:
    """Machine-compliance correction of a crosshead displacement: d_specimen = d − C·F (mm)."""
    return np.asarray(disp_mm, float) - float(compliance_mm_per_N or 0.0) * np.asarray(force_N, float)


def estimate_compliance(force_N, disp_mm, area_mm2: float, gauge_mm: float, E_true_GPa: float,
                        modulus_range: tuple[float, float] | None = None) -> dict:
    """Machine compliance C (mm/N) such that the corrected curve has the known modulus ``E_true_GPa``.

    The apparent elastic slope gives the total compliance C_tot = L0 / (E_app·A0); the specimen
    contributes L0 / (E_true·A0), so C = C_tot − L0/(E_true·A0) (E in MPa = N/mm²).
    Returns {'C_mm_per_N', 'E_apparent_GPa', 'E_true_GPa', 'fit'}.
    """
    e, s = stress_strain(force_N, disp_mm, area_mm2, gauge_mm)
    m = np.isfinite(e) & np.isfinite(s)
    e, s = e[m], s[m]
    if e.size < 6:
        raise ValueError("Not enough points to estimate the machine compliance")
    if np.nanmax(s) < -np.nanmin(s):
        e, s = -e, -s
    i = int(np.argmax(s))
    if modulus_range is not None:
        fit = linear_fit(e[: i + 1], s[: i + 1], *sorted(modulus_range))
    else:
        fit = auto_modulus_fit(e[: i + 1], s[: i + 1])
    E_app = fit.slope * 100.0                     # MPa/% → MPa
    if E_app <= 0 or E_true_GPa <= 0:
        raise ValueError("Apparent and true modulus must be positive")
    C_tot = gauge_mm / (E_app * area_mm2)
    C = C_tot - gauge_mm / (E_true_GPa * 1000.0 * area_mm2)
    return {"C_mm_per_N": float(C), "E_apparent_GPa": float(E_app / 1000.0), "E_true_GPa": float(E_true_GPa),
            "fit": fit}


def stress_strain(force_N, disp_mm, area_mm2: float, gauge_mm: float, compression: bool = False,
                  zero_displacement: bool = True, compliance_mm_per_N: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Engineering strain (%) and stress (MPa) from force (N) and displacement (mm).

    σ = F / A0 (N/mm² = MPa); ε = ΔL / L0 × 100 %. With ``zero_displacement`` the first
    displacement value is taken as ΔL = 0. With ``compression=True`` the signs are normalised so
    that compressive stress and strain are positive (whatever sign the machine records).
    ``compliance_mm_per_N`` (machine compliance C) corrects the crosshead displacement: d − C·F.
    """
    if not area_mm2 or area_mm2 <= 0:
        raise ValueError("Cross-section area must be > 0 mm²")
    if not gauge_mm or gauge_mm <= 0:
        raise ValueError("Gauge length must be > 0 mm")
    F = np.asarray(force_N, float)
    d = np.asarray(disp_mm, float)
    if compliance_mm_per_N:
        d = compliance_correct(d, F, compliance_mm_per_N)
    if zero_displacement and d.size:
        first = d[np.isfinite(d)]
        d = d - (first[0] if first.size else 0.0)
    if compression:
        if np.nanmax(F) < -np.nanmin(F):
            F = -F
        if np.nanmax(d) < -np.nanmin(d):
            d = -d
    return d / gauge_mm * 100.0, F / area_mm2


# --------------------------------------------------------------------------- result
@dataclass
class MechanicalResult:
    """Properties of a monotonic stress–strain curve (tension or compression, magnitudes)."""
    mode: str                                   # "tensile" | "compression"
    E_GPa: float
    modulus_fit: LinearFit | None
    modulus_range: tuple[float, float]          # strain range (%) of the modulus fit
    strain_intercept_pct: float                 # toe correction: strain where the modulus line hits σ = 0
    offset_pct: float
    yield_strength_MPa: float                   # offset (proof) strength
    yield_strain_pct: float
    uts_MPa: float                              # max stress (UTS in tension, max compressive stress)
    strain_at_uts_pct: float
    fracture_detected: bool
    fracture_strain_pct: float                  # total strain at fracture (or last point)
    fracture_stress_MPa: float
    elongation_at_break_pct: float              # fracture strain minus elastic recovery σ_f/E
    toughness_MJ_m3: float                      # ∫σ dε up to fracture (or end)
    superelastic: bool = False
    sigma_SIM_MPa: float = float("nan")        # forward transformation start (two-tangent)
    strain_SIM_pct: float = float("nan")
    plateau_stress_MPa: float = float("nan")   # stress at mid-plateau
    plateau_slope_MPa_per_pct: float = float("nan")
    plateau_end_strain_pct: float = float("nan")
    transformation_strain_pct: float = float("nan")
    i_uts: int = 0
    i_fracture: int = 0
    modulus_line: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    offset_line: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    plateau_line: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)       # all parameters incl. auto-chosen modulus range

    @property
    def max_stress_MPa(self) -> float:
        return self.uts_MPa

    @property
    def strain_at_max_pct(self) -> float:
        return self.strain_at_uts_pct

    def rows(self) -> list[tuple[str, float, str]]:
        """Summary rows (quantity, value, unit)."""
        smax = "UTS" if self.mode == "tensile" else "Max compressive stress"
        r = [
            ("Young's modulus E", self.E_GPa, "GPa"),
            ("Modulus fit R²", self.modulus_fit.r2 if self.modulus_fit else float("nan"), ""),
            ("Modulus range start", self.modulus_range[0], "%"),
            ("Modulus range end", self.modulus_range[1], "%"),
            (f"{self.offset_pct:g} % offset yield strength", self.yield_strength_MPa, "MPa"),
            ("Strain at yield", self.yield_strain_pct, "%"),
            (smax, self.uts_MPa, "MPa"),
            (f"Strain at {'UTS' if self.mode == 'tensile' else 'max stress'}", self.strain_at_uts_pct, "%"),
        ]
        if self.mode == "tensile" or self.fracture_detected:
            r += [("Fracture strain" if self.fracture_detected else "Final strain (no fracture detected)",
                   self.fracture_strain_pct, "%"),
                  ("Fracture stress", self.fracture_stress_MPa, "MPa"),
                  ("Elongation at break (plastic)", self.elongation_at_break_pct, "%")]
        r.append(("Toughness ∫σdε" if self.mode == "tensile" else "Absorbed energy ∫σdε",
                  self.toughness_MJ_m3, "MJ/m³"))
        if self.superelastic:
            r += [("σ_SIM (transformation start)", self.sigma_SIM_MPa, "MPa"),
                  ("Strain at σ_SIM", self.strain_SIM_pct, "%"),
                  ("Plateau stress (mid-plateau)", self.plateau_stress_MPa, "MPa"),
                  ("Plateau slope", self.plateau_slope_MPa_per_pct, "MPa/%"),
                  ("Plateau end strain", self.plateau_end_strain_pct, "%"),
                  ("Transformation strain", self.transformation_strain_pct, "%")]
        return r


def mechanical_sanity_warnings(E_GPa: float, uts_MPa: float, max_strain_pct: float) -> list[str]:
    """Plausibility checks: E outside 1–300 GPa, UTS > 5 GPa, strain > 100 %."""
    w = []
    if np.isfinite(E_GPa) and not (1.0 <= E_GPa <= 300.0):
        w.append(f"E = {E_GPa:.4g} GPa is outside the plausible range 1–300 GPa (NiTi austenite ≈ 40–80 GPa): "
                 "check the strain unit (% vs mm/mm), gauge length, area or the modulus range.")
    if np.isfinite(uts_MPa) and uts_MPa > 5000.0:
        w.append(f"Maximum stress {uts_MPa:.4g} MPa exceeds 5 GPa: check the force unit and cross-section area.")
    if np.isfinite(max_strain_pct) and max_strain_pct > 100.0:
        w.append(f"Maximum strain {max_strain_pct:.4g} % exceeds 100 %: check the strain/displacement unit and "
                 "gauge length.")
    return w


TensileResult = MechanicalResult
CompressionResult = MechanicalResult


# --------------------------------------------------------------------------- helpers
def auto_modulus_fit(strain_pct, stress_MPa, *, band_frac: float = 0.15, r2_min: float = 0.995,
                     min_points: int = 6) -> LinearFit:
    """Automatic elastic-region fit on a loading branch (data up to the maximum stress).

    Sliding windows spanning ``band_frac`` × σ_max in stress are fitted; among windows with
    R² ≥ ``r2_min`` the steepest is taken (the initial elastic line is the steepest linear part of
    a loading curve; toe regions, plateaux and hardening are less steep or non-linear). If no
    window reaches ``r2_min`` the window with the highest R² is used.
    """
    e = np.asarray(strain_pct, float)
    s = np.asarray(stress_MPa, float)
    m = np.isfinite(e) & np.isfinite(s)
    e, s = e[m], s[m]
    if e.size < min_points:
        return linear_fit(e, s)
    smax = float(np.max(s))
    band = band_frac * smax
    run_max = np.maximum.accumulate(s)
    best: LinearFit | None = None
    best_r2: LinearFit | None = None
    for L in np.linspace(0.02, max(0.03, 0.95 - band_frac), 60) * smax:
        i0 = int(np.searchsorted(run_max, L))
        i1 = int(np.searchsorted(run_max, L + band))
        if i0 >= e.size:
            break
        if i1 >= e.size:
            continue
        i1 = max(i1, i0 + min_points - 1)
        if i1 >= e.size:
            continue
        xs, ys = e[i0:i1 + 1], s[i0:i1 + 1]
        if np.ptp(xs) <= 0:
            continue
        try:
            f = linear_fit(xs, ys)
        except ValueError:
            continue
        if f.slope <= 0:
            continue
        f.xmin, f.xmax = float(xs.min()), float(xs.max())
        if f.r2 >= r2_min and (best is None or f.slope > best.slope):
            best = f
        if best_r2 is None or f.r2 > best_r2.r2:
            best_r2 = f
    out = best or best_r2
    if out is None:
        out = linear_fit(e, s)
    return out


def _detect_fracture(e, s, i_uts, drop_frac, window_pct):
    """First sudden drop after the maximum: stress falls by ≥ drop_frac·σ_max within window_pct strain.

    Returns (index of the last point before the drop, detected?)."""
    smax = s[i_uts]
    need = drop_frac * smax
    dq: deque[int] = deque()      # indices with decreasing stress (sliding max)
    left = i_uts
    for j in range(i_uts, e.size):
        while dq and s[dq[-1]] <= s[j]:
            dq.pop()
        dq.append(j)
        while left < j and e[j] - e[left] > window_pct and abs(j - left) > 1:
            left += 1
            while dq and dq[0] < left:
                dq.popleft()
        imax = dq[0]
        if s[imax] - s[j] >= need:
            # the break = the point just before the largest single-step drop in the window
            lo = min(imax, j - 1)
            steps = s[lo:j] - s[lo + 1:j + 1]
            return int(lo + int(np.argmax(steps))), True
    # no drop: last point
    return int(e.size - 1), False


def _envelope(e, s):
    """Uniform-strain resampling of the monotonic loading envelope (running max of strain)."""
    keep = np.r_[True, e[1:] > np.maximum.accumulate(e)[:-1]]
    ee, ss = e[keep], s[keep]
    if ee.size < 5:
        return ee, ss, 1.0
    step = max(float(np.ptp(ee)) / 4000.0, 1e-4)
    grid = np.arange(ee[0], ee[-1], step)
    return grid, np.interp(grid, ee, ss), step


def _superelastic(e, s, i_uts, fit: LinearFit, plateau_frac: float, min_plateau_pct: float):
    """Detect a forward-transformation (SIM) plateau on the loading branch.

    Returns dict or None."""
    E = fit.slope
    ge, gs, step = _envelope(e[: i_uts + 1], s[: i_uts + 1])
    if ge.size < 50:
        return None
    win = max(7, int(round(0.25 / step)) | 1)
    slope = smooth(gs, win, polyorder=2, deriv=1, delta=step)
    start = int(np.searchsorted(ge, fit.xmax))
    low = slope < plateau_frac * E
    p0 = start + int(np.argmax(low[start:])) if low[start:].any() else None
    if p0 is None:
        return None
    p1 = p0
    while p1 < ge.size - 1 and low[p1 + 1]:
        p1 += 1
    L = ge[p1] - ge[p0]
    if L < min_plateau_pct or p1 >= ge.size - 1:
        return None
    mid = slice(p0 + int(0.25 * (p1 - p0)), p0 + int(0.75 * (p1 - p0)) + 1)
    pfit = linear_fit(ge[mid], gs[mid])
    # re-hardening after the plateau (detwinned martensite elastic loading)
    after = slope[p1 + 1:]
    harden_len = (after > max(3 * max(pfit.slope, 0), 0.1 * E)).sum() * step
    if harden_len < 0.5:
        return None
    if pfit.slope >= E:
        return None
    eps_sim = (pfit.intercept - fit.intercept) / (fit.slope - pfit.slope)
    sig_sim = fit(eps_sim)
    # martensite elastic fit just after the plateau → end of transformation
    j0 = p1 + 1
    j1 = min(ge.size - 1, j0 + int(round(1.0 / step)))
    eps_end = float(ge[p1])
    if j1 - j0 > 5:
        mfit = linear_fit(ge[j0:j1 + 1], gs[j0:j1 + 1])
        if mfit.slope > pfit.slope:
            eps_end = (pfit.intercept - mfit.intercept) / (mfit.slope - pfit.slope)
    e_mid = 0.5 * (eps_sim + eps_end)
    return dict(sigma=float(sig_sim), eps=float(eps_sim), pfit=pfit,
                plateau_stress=float(np.interp(e_mid, ge, gs)), eps_end=float(eps_end))


def _mechanical(strain_pct, stress_MPa, mode, modulus_range, offset_pct, fracture_drop_pct,
                fracture_window_pct, superelastic, r2_min, plateau_slope_frac, min_plateau_pct,
                band_frac) -> MechanicalResult:
    e = np.asarray(strain_pct, float).ravel()
    s = np.asarray(stress_MPa, float).ravel()
    m = np.isfinite(e) & np.isfinite(s)
    e, s = e[m], s[m]
    if e.size < 10:
        raise ValueError("Not enough points for a mechanical analysis")
    notes = []
    if mode == "compression":
        big = int(np.argmax(np.abs(s)))
        if s[big] < 0:
            s = -s
            notes.append("Compressive stress was negative; signs flipped.")
        if np.nanmax(e) < -np.nanmin(e):
            e = -e
    i_uts = int(np.argmax(s))
    uts = float(s[i_uts])
    eL, sL = e[: i_uts + 1], s[: i_uts + 1]

    if modulus_range is not None:
        lo, hi = sorted(modulus_range)
        fit = linear_fit(eL, sL, lo, hi)
    else:
        fit = auto_modulus_fit(eL, sL, band_frac=band_frac, r2_min=r2_min)
    E_GPa = fit.slope / 10.0
    eps0 = -fit.intercept / fit.slope if fit.slope else 0.0

    # offset yield
    i_start = int(np.searchsorted(np.maximum.accumulate(eL), fit.xmin))
    off = fit.slope * (eL - eps0 - offset_pct)
    f = sL - off
    ys_val, ye_val = float("nan"), float("nan")
    for i in range(max(i_start, 1), eL.size):
        if f[i] <= 0 < f[i - 1]:
            t = f[i - 1] / (f[i - 1] - f[i])
            ye_val = float(eL[i - 1] + t * (eL[i] - eL[i - 1]))
            ys_val = float(sL[i - 1] + t * (sL[i] - sL[i - 1]))
            break
    if not np.isfinite(ys_val):
        notes.append("Offset line does not intersect the curve before the maximum.")

    # fracture
    i_f, detected = _detect_fracture(e, s, i_uts, fracture_drop_pct / 100.0, fracture_window_pct)
    ef, sf = float(e[i_f]), float(s[i_f])
    elong = ef - sf / fit.slope if fit.slope else float("nan")
    tough = trapz(s[: i_f + 1], e[: i_f + 1] / 100.0)

    # superelastic plateau
    se = None
    if superelastic is not False:
        se = _superelastic(e, s, i_uts, fit, plateau_slope_frac, min_plateau_pct)
        if superelastic is True and se is None:
            notes.append("No transformation plateau detected.")
    res = MechanicalResult(
        mode=mode, E_GPa=E_GPa, modulus_fit=fit, modulus_range=(fit.xmin, fit.xmax),
        strain_intercept_pct=float(eps0), offset_pct=offset_pct,
        yield_strength_MPa=ys_val, yield_strain_pct=ye_val, uts_MPa=uts,
        strain_at_uts_pct=float(e[i_uts]), fracture_detected=detected, fracture_strain_pct=ef,
        fracture_stress_MPa=sf, elongation_at_break_pct=float(elong), toughness_MJ_m3=float(tough),
        i_uts=i_uts, i_fracture=i_f, notes=notes)
    res.warnings = mechanical_sanity_warnings(E_GPa, uts, float(np.nanmax(e)))
    res.params = {"mode": mode, "modulus_range_user": list(modulus_range) if modulus_range is not None else None,
                  "modulus_range_used": [fit.xmin, fit.xmax], "modulus_auto": modulus_range is None,
                  "offset_pct": offset_pct, "fracture_drop_pct": fracture_drop_pct,
                  "fracture_window_pct": fracture_window_pct, "superelastic": superelastic, "r2_min": r2_min,
                  "plateau_slope_frac": plateau_slope_frac, "min_plateau_pct": min_plateau_pct,
                  "band_frac": band_frac, "signs_flipped": any("flipped" in n for n in notes)}
    top = ys_val if np.isfinite(ys_val) else 0.6 * uts
    if se is not None:
        res.superelastic = True
        res.sigma_SIM_MPa = se["sigma"]
        res.strain_SIM_pct = se["eps"]
        res.plateau_stress_MPa = se["plateau_stress"]
        res.plateau_slope_MPa_per_pct = se["pfit"].slope
        res.plateau_end_strain_pct = se["eps_end"]
        res.transformation_strain_pct = se["eps_end"] - se["eps"]
        px = np.array([se["eps"], se["eps_end"]])
        res.plateau_line = (px, se["pfit"](px))
        top = max(top, se["sigma"])
    y_top = min(1.15 * top, uts)
    mx = np.array([eps0, fit.x_at(y_top)])
    res.modulus_line = (mx, fit(mx))
    if np.isfinite(ye_val):
        ox = np.array([eps0 + offset_pct, ye_val + 0.3 * (ye_val - eps0 - offset_pct) + 0.05])
        res.offset_line = (ox, fit.slope * (ox - eps0 - offset_pct))
    return res


def tensile_properties(strain_pct, stress_MPa, modulus_range: tuple[float, float] | None = None, *,
                       offset_pct: float = 0.2, fracture_drop_pct: float = 40.0,
                       fracture_window_pct: float = 0.5, superelastic: bool | None = None,
                       r2_min: float = 0.995, plateau_slope_frac: float = 0.15,
                       min_plateau_pct: float = 1.0, band_frac: float = 0.15) -> MechanicalResult:
    """Tensile properties from an engineering stress–strain curve (strain %, stress MPa).

    * E: linear fit over ``modulus_range`` (strain %, on the loading branch) or auto
      (:func:`auto_modulus_fit`). Reported in GPa.
    * Offset yield: intersection of the curve with σ = E (ε − ε0 − offset), ε0 = toe-corrected
      strain intercept of the modulus line.
    * UTS = max stress; fracture = last point before the first drop of ≥ ``fracture_drop_pct`` % of
      UTS within ``fracture_window_pct`` strain after the UTS (else the last point).
    * Toughness = ∫σ dε (ε in mm/mm) from the start to fracture, MJ/m³.
    * Superelastic NiTi (``superelastic=None`` auto-detects): σ_SIM is the intersection of the
      elastic line with a line fitted to the middle half of the transformation plateau
      (plateau = local slope < ``plateau_slope_frac`` × E for ≥ ``min_plateau_pct`` strain,
      followed by re-hardening); the end of transformation is the intersection of the plateau line
      with the detwinned-martensite elastic line.
    """
    return _mechanical(strain_pct, stress_MPa, "tensile", modulus_range, offset_pct, fracture_drop_pct,
                       fracture_window_pct, superelastic, r2_min, plateau_slope_frac, min_plateau_pct,
                       band_frac)


def compression_properties(strain_pct, stress_MPa, modulus_range: tuple[float, float] | None = None, *,
                           offset_pct: float = 0.2, fracture_drop_pct: float = 40.0,
                           fracture_window_pct: float = 0.5, superelastic: bool | None = None,
                           r2_min: float = 0.995, plateau_slope_frac: float = 0.15,
                           min_plateau_pct: float = 1.0, band_frac: float = 0.15) -> MechanicalResult:
    """Compression properties (magnitudes; negative machine signs are flipped automatically).

    Same definitions as :func:`tensile_properties`; ``uts_MPa`` is the maximum compressive stress
    and ``toughness_MJ_m3`` the absorbed energy up to failure (if a drop is detected) or the end.
    """
    return _mechanical(strain_pct, stress_MPa, "compression", modulus_range, offset_pct,
                       fracture_drop_pct, fracture_window_pct, superelastic, r2_min,
                       plateau_slope_frac, min_plateau_pct, band_frac)
