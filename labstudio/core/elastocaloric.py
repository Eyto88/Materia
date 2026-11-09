"""Elastocaloric analyses for NiTi: Clausius–Clapeyron and measured adiabatic ΔT / material COP.

Pure Python (no Qt).

Clausius–Clapeyron (stress-induced transformation)::

    dσ/dT = −ΔS·ρ / ε_tr      ⇒   ΔS = −(dσ/dT)·ε_tr / ρ          [J/(kg·K)]
    L = T·|ΔS|                                                    [J/kg → J/g]

with dσ/dT in Pa/K (= 10⁶ × MPa/K), ε_tr the transformation strain (mm/mm), ρ in kg/m³
(6.45 g/cm³ = 6450 kg/m³ for NiTi). L is compared with the DSC latent heat ΔH (J/g).

Measured elastocaloric effect (temperature log + cyclic stress–strain test, common time base
``t_mech = t_log + offset``) per cycle::

    T_base   = median T in the window before the cycle starts (pre-cycle baseline)
    ΔT_heat  = max T during loading  − T_base   (> 0)
    ΔT_cool  = min T during unloading − T_base  (< 0)
    W        = hysteresis loop area (MJ/m³) × 10⁶ / ρ          [J/kg]  (input work per cycle)
    Q_c      = c_p·|ΔT_cool|                                   [J/kg]
    COP_mat  = Q_c / W ;   specific cooling power = Q_c · f    [W/kg], f = 1 / cycle duration
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import analysis as A
from .analysis_common import linear_fit
from .dataset import Dataset

__all__ = [
    "RHO_NITI_G_CM3", "CP_NITI_J_GK", "TEMPERATURE_REGEX", "temperature_from", "CCPoint", "ClausiusClapeyron",
    "clausius_clapeyron", "cc_point_from_dataset", "CycleCaloric", "CaloricResult", "caloric_per_cycle",
    "caloric_dataset", "entropy_change", "temperature_log_columns",
]

RHO_NITI_G_CM3 = 6.45
CP_NITI_J_GK = 0.45
TEMPERATURE_REGEX = r"(-?\d+(?:[.,]\d+)?)\s*°?\s*C(?![a-zA-Z])"
_META_TEMP_KEYS = ("test_temperature_C", "temperature_C", "test_temperature", "T_test_C", "temperature")


# =========================================================================== temperature of a test
def _number(text: str) -> float | None:
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*(°\s*C|K\b|°\s*F)?", str(text))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    u = (m.group(2) or "").replace(" ", "")
    if u == "K":
        v -= 273.15
    elif u == "°F":
        v = (v - 32) * 5 / 9
    return v


def temperature_from(ds: Dataset, regex: str | None = None) -> float | None:
    """Test temperature (°C) of a dataset: metadata (``test_temperature_C`` …, or a raw header line
    containing 'temperature'), else ``regex`` (default :data:`TEMPERATURE_REGEX`) on the name / file."""
    md = ds.metadata or {}
    for k in _META_TEMP_KEYS:
        if md.get(k) is not None:
            v = md[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
            n = _number(v)
            if n is not None:
                return n
    raw = md.get("source_meta")
    items = raw.items() if isinstance(raw, dict) else ([] if raw is None else
                                                       [(str(x), str(x)) for x in (raw if isinstance(raw, (list, tuple)) else [raw])])
    for k, v in items:
        if "temp" in str(k).lower() and "rate" not in str(k).lower():
            n = _number(v if k != v else re.sub(r"^[^:]*:", "", str(v)))
            if n is not None:
                return n
    rx = regex or TEMPERATURE_REGEX
    import os
    for text in (ds.name, os.path.basename(ds.source_path or "")):
        if text:
            m = re.search(rx, text)
            if m:
                g = m.group(1) if m.groups() else m.group(0)
                try:
                    return float(g.replace(",", "."))
                except ValueError:
                    pass
    return None


# =========================================================================== Clausius–Clapeyron
def entropy_change(slope_MPa_per_K: float, eps_tr_pct: float, rho_g_cm3: float = RHO_NITI_G_CM3) -> float:
    """ΔS = −(dσ/dT)·ε_tr/ρ in J/(kg·K) (negative for the forward A→M transformation)."""
    return -float(slope_MPa_per_K) * 1e6 * float(eps_tr_pct) / 100.0 / (float(rho_g_cm3) * 1000.0)


@dataclass
class CCPoint:
    name: str
    T_C: float
    sigma_MPa: float
    eps_tr_pct: float = float("nan")
    source: str = ""            # "σ_SIM" | "σ_ms" | "user"
    ds_id: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ClausiusClapeyron:
    points: list[CCPoint]
    slope_MPa_per_K: float
    intercept_MPa: float
    r2: float
    slope_stderr: float
    eps_tr_pct: float
    rho_g_cm3: float
    T_ref_C: float
    dS_J_per_kgK: float
    latent_J_per_g: float          # L = T_ref·|ΔS|
    dsc_dH_J_per_g: float = float("nan")
    dsc_diff_pct: float = float("nan")
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def line(self, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
        T = np.array([p.T_C for p in self.points], float)
        x = np.linspace(T.min(), T.max(), n) if T.size else np.array([])
        return x, self.slope_MPa_per_K * x + self.intercept_MPa

    def rows(self) -> list[tuple[str, float, str]]:
        r = [("dσ/dT (Clausius–Clapeyron slope)", self.slope_MPa_per_K, "MPa/K"),
             ("dσ/dT standard error", self.slope_stderr, "MPa/K"),
             ("Fit R²", self.r2, ""), ("Intercept σ(0 °C)", self.intercept_MPa, "MPa"),
             ("Points", len(self.points), "#"),
             ("Transformation strain ε_tr", self.eps_tr_pct, "%"), ("Density ρ", self.rho_g_cm3, "g/cm³"),
             ("Reference temperature T", self.T_ref_C, "°C"),
             ("Entropy change ΔS", self.dS_J_per_kgK, "J/(kg·K)"),
             ("Latent heat L = T·|ΔS|", self.latent_J_per_g, "J/g")]
        if np.isfinite(self.dsc_dH_J_per_g):
            r += [("DSC ΔH", self.dsc_dH_J_per_g, "J/g"), ("L vs DSC ΔH difference", self.dsc_diff_pct, "%")]
        return r


def clausius_clapeyron(T_C, sigma_MPa, eps_tr_pct, rho_g_cm3: float = RHO_NITI_G_CM3,
                       T_ref_C: float | None = None, dsc_dH_J_per_g: float | None = None,
                       names: list[str] | None = None, sources: list[str] | None = None) -> ClausiusClapeyron:
    """Linear fit σ(T) and the derived entropy change / latent heat.

    eps_tr_pct: scalar or one value per point (the mean of the finite values is used).
    T_ref_C: temperature for L = T·|ΔS| (default: mean test temperature).
    dsc_dH_J_per_g: optional DSC latent heat to cross-check (% difference = (L − ΔH)/ΔH).
    """
    T = np.asarray(T_C, float).ravel()
    s = np.asarray(sigma_MPa, float).ravel()
    eps = np.broadcast_to(np.asarray(eps_tr_pct, float), T.shape).astype(float)
    names = names or [f"point {i + 1}" for i in range(T.size)]
    sources = sources or ["user"] * T.size
    pts = [CCPoint(n, float(t), float(v), float(e), src) for n, t, v, e, src in zip(names, T, s, eps, sources)]
    m = np.isfinite(T) & np.isfinite(s)
    warnings: list[str] = []
    if m.sum() < 2 or np.ptp(T[m]) == 0:
        raise ValueError("Clausius–Clapeyron needs at least two tests at different temperatures")
    if m.sum() == 2:
        warnings.append("Only two temperatures: the slope has no uncertainty estimate — test ≥ 3 temperatures.")
    f = linear_fit(T[m], s[m])
    n = int(m.sum())
    if n > 2:
        resid = s[m] - f(T[m])
        sxx = float(np.sum((T[m] - T[m].mean()) ** 2))
        se = float(np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx)) if sxx > 0 else float("nan")
    else:
        se = float("nan")
    e_ok = eps[np.isfinite(eps)]
    e_tr = float(np.mean(e_ok)) if e_ok.size else float("nan")
    if not np.isfinite(e_tr):
        warnings.append("Transformation strain ε_tr unknown: enter it to obtain ΔS and L.")
    Tref = float(np.mean(T[m])) if T_ref_C is None else float(T_ref_C)
    dS = entropy_change(f.slope, e_tr, rho_g_cm3) if np.isfinite(e_tr) else float("nan")
    L = (Tref + 273.15) * abs(dS) / 1000.0 if np.isfinite(dS) else float("nan")
    if f.slope <= 0:
        warnings.append(f"dσ/dT = {f.slope:.3g} MPa/K is not positive: superelastic NiTi gives ≈ 5–8 MPa/K — check the "
                        "test temperatures and the plateau stresses.")
    elif not (2.0 <= f.slope <= 20.0):
        warnings.append(f"dσ/dT = {f.slope:.3g} MPa/K is outside the usual NiTi range (≈ 5–8 MPa/K).")
    if n > 2 and f.r2 < 0.9:
        warnings.append(f"Poor linear fit (R² = {f.r2:.3f}).")
    dH = float(dsc_dH_J_per_g) if dsc_dH_J_per_g else float("nan")
    diff = 100.0 * (L - dH) / dH if np.isfinite(dH) and dH and np.isfinite(L) else float("nan")
    params = {"rho_g_cm3": rho_g_cm3, "eps_tr_pct": e_tr, "eps_tr_per_point": eps.tolist(), "T_ref_C": Tref,
              "T_ref_auto": T_ref_C is None, "dsc_dH_J_per_g": dH,
              "points": [{"name": p.name, "T_C": p.T_C, "sigma_MPa": p.sigma_MPa, "source": p.source} for p in pts]}
    return ClausiusClapeyron(pts, f.slope, f.intercept, f.r2, se, e_tr, rho_g_cm3, Tref, dS, L, dH, diff,
                             warnings, params)


def cc_point_from_dataset(ds: Dataset, *, source: str = "auto", T_C: float | None = None,
                          temperature_regex: str | None = None, compression: bool = False) -> CCPoint:
    """σ(T) point of one tensile/compression/cyclic test.

    source: "σ_SIM" (two-tangent transformation start of the (first) loading branch), "σ_ms"
    (upper plateau at mid-plateau strain, first complete cycle — cyclic data) or "auto" (σ_SIM if
    detected, else σ_ms). ε_tr from the plateau (transformation strain) when available.
    """
    from .batch import mech_arrays
    T = T_C if T_C is not None else temperature_from(ds, temperature_regex)
    warns: list[str] = []
    if T is None:
        warns.append("test temperature not found (enter it in the table)")
        T = float("nan")
    e, s, _ = mech_arrays(ds, compression=compression)
    e, s, _fe, _fs = A.normalise_signs(e, s, "auto")
    m = np.isfinite(e) & np.isfinite(s)
    e, s = e[m], s[m]
    sig, eps, src = float("nan"), float("nan"), ""
    # first loading branch: up to the first strain maximum
    cyc_col = ds.first_of("count")
    if ds.kind == "cyclic" or cyc_col:
        res = A.analyse_cyclic(e, s, cycle=ds.values(cyc_col)[m] if cyc_col else None)
        first = res.metrics[0] if res.metrics else None
        cm = res.complete_metrics[0] if res.complete_metrics else first
        if first is not None:
            idx = res.cycles[0].indices
            ipk = int(np.argmax(e[idx]))
            eL, sL = e[idx][: ipk + 1], s[idx][: ipk + 1]
        else:
            eL, sL = e, s
    else:
        cm = None
        ipk = int(np.argmax(s))
        eL, sL = e, s
    if source in ("auto", "σ_SIM", "sigma_SIM"):
        try:
            r = A.tensile_properties(eL, sL, superelastic=True)
            if r.superelastic:
                sig, eps, src = r.sigma_SIM_MPa, r.transformation_strain_pct, "σ_SIM"
        except ValueError:
            pass
    if not np.isfinite(sig) and source in ("auto", "σ_ms", "sigma_ms") and cm is not None:
        sig, src = cm.sigma_ms_MPa, "σ_ms"
        try:
            r = A.tensile_properties(eL, sL, superelastic=None)
            if r.superelastic:
                eps = r.transformation_strain_pct
        except ValueError:
            pass
    if not np.isfinite(sig):
        warns.append(f"no transformation plateau found ({source})")
    return CCPoint(ds.name, float(T), float(sig), float(eps), src, ds.id, warns)


# =========================================================================== measured ΔT and COP
@dataclass
class CycleCaloric:
    cycle: int
    t_start_s: float
    t_peak_s: float
    t_end_s: float
    T_base_C: float
    dT_heat_K: float
    dT_cool_K: float              # negative (cooling)
    work_MJ_m3: float             # loop area
    work_J_per_kg: float
    Qc_J_per_kg: float
    COP: float
    frequency_Hz: float
    cooling_power_W_per_kg: float
    complete: bool = True


@dataclass
class CaloricResult:
    cycles: list[CycleCaloric]
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def column(self, attr: str) -> np.ndarray:
        return np.array([getattr(c, attr) for c in self.cycles], float)

    HEADERS = ["Cycle", "ΔT heating (K)", "ΔT cooling (K)", "Baseline T (°C)", "Work W (MJ/m³)", "W (J/kg)",
               "Q_c (J/kg)", "COP_mat", "Frequency (Hz)", "Cooling power (W/kg)"]

    def table(self) -> list[list]:
        return [[c.cycle, c.dT_heat_K, c.dT_cool_K, c.T_base_C, c.work_MJ_m3, c.work_J_per_kg, c.Qc_J_per_kg,
                 c.COP, c.frequency_Hz, c.cooling_power_W_per_kg] for c in self.cycles]

    def rows(self) -> list[tuple[str, float, str]]:
        ok = [c for c in self.cycles if c.complete and np.isfinite(c.COP)]
        if not ok:
            return [("Cycles", len(self.cycles), "#")]
        a, b = ok[0], ok[-1]
        mean = lambda attr: float(np.nanmean([getattr(c, attr) for c in ok]))  # noqa: E731
        return [("Cycles", len(self.cycles), "#"),
                ("ΔT heating first cycle", a.dT_heat_K, "K"), ("ΔT heating last cycle", b.dT_heat_K, "K"),
                ("ΔT cooling first cycle", a.dT_cool_K, "K"), ("ΔT cooling last cycle", b.dT_cool_K, "K"),
                ("COP_mat first cycle", a.COP, ""), ("COP_mat last cycle", b.COP, ""),
                ("COP_mat mean", mean("COP"), ""),
                ("Specific cooling power mean", mean("cooling_power_W_per_kg"), "W/kg"),
                ("Frequency mean", mean("frequency_Hz"), "Hz")]


def temperature_log_columns(ds: Dataset) -> tuple[str, str]:
    """(time, temperature) columns of a temperature-log dataset; ValueError if missing."""
    t, T = ds.first_of("time"), ds.first_of("temperature")
    if not t or not T:
        raise ValueError(f"'{ds.name}' needs a time and a temperature column")
    return t, T


def caloric_per_cycle(temp_time_s, temp_C, mech_time_s, strain_pct, stress_MPa, *, cycle=None,
                      offset_s: float = 0.0, rho_g_cm3: float = RHO_NITI_G_CM3, cp_J_per_gK: float = CP_NITI_J_GK,
                      baseline_window_s: float | None = None, lag_frac: float = 0.1,
                      compression="auto") -> CaloricResult:
    """Per-cycle measured ΔT (heating on loading, cooling on unloading), input work, COP and cooling power.

    temp_time_s / temp_C: temperature log (thermocouple / IR). The log is shifted by ``offset_s``
    onto the mechanical time base (t_mech = t_log + offset). ``baseline_window_s`` (default 5 % of
    each cycle's duration) is the window before the cycle start used for T_base; ``lag_frac`` extends
    the loading/unloading windows by this fraction of the half-cycle (thermal lag of the sensor).
    """
    tl = np.asarray(temp_time_s, float).ravel() + float(offset_s)
    Tl = np.asarray(temp_C, float).ravel()
    ml = np.isfinite(tl) & np.isfinite(Tl)
    tl, Tl = tl[ml], Tl[ml]
    o = np.argsort(tl, kind="stable")
    tl, Tl = tl[o], Tl[o]
    tm = np.asarray(mech_time_s, float).ravel()
    if tl.size < 5:
        raise ValueError("Temperature log has fewer than 5 valid points")
    res = A.analyse_cyclic(strain_pct, stress_MPa, cycle=cycle, time=tm, compression=compression)
    warnings = list(res.warnings)
    rho_kg = rho_g_cm3 * 1000.0
    out: list[CycleCaloric] = []
    uncovered = []
    for cy, mt in zip(res.cycles, res.metrics):
        idx = cy.indices
        t0, t1 = float(np.nanmin(tm[idx])), float(np.nanmax(tm[idx]))
        tp = float(tm[mt.i_max])
        dur = t1 - t0
        w = baseline_window_s if baseline_window_s else 0.05 * dur
        sel = (tl >= t0 - w) & (tl <= t0)
        if sel.sum() < 2:
            sel = (tl >= t0) & (tl <= t0 + w)
        nan = float("nan")
        if sel.sum() < 1 or tl[-1] < t1 - 0.05 * dur or tl[0] > t0 + 0.05 * dur:
            uncovered.append(mt.cycle)
            out.append(CycleCaloric(mt.cycle, t0, tp, t1, nan, nan, nan, mt.dissipated_energy_MJ_m3, nan, nan, nan,
                                    1 / dur if dur > 0 else nan, nan, False))
            continue
        Tb = float(np.median(Tl[sel]))
        lagL = lag_frac * (tp - t0)
        lagU = lag_frac * (t1 - tp)
        mL = (tl >= t0) & (tl <= tp + lagL)
        mU = (tl >= tp) & (tl <= t1 + lagU)
        dTh = float(np.max(Tl[mL]) - Tb) if mL.any() else nan
        dTc = float(np.min(Tl[mU]) - Tb) if mU.any() else nan
        W = mt.dissipated_energy_MJ_m3
        Wkg = W * 1e6 / rho_kg if np.isfinite(W) else nan
        Qc = cp_J_per_gK * 1000.0 * abs(dTc) if np.isfinite(dTc) else nan
        cop = Qc / Wkg if np.isfinite(Wkg) and Wkg > 0 and np.isfinite(Qc) else nan
        f = 1.0 / dur if dur > 0 else nan
        out.append(CycleCaloric(mt.cycle, t0, tp, t1, Tb, dTh, dTc, W, Wkg, Qc, cop, f,
                                Qc * f if np.isfinite(Qc) else nan, bool(mt.complete and np.isfinite(cop))))
    if uncovered:
        warnings.append(f"Temperature log does not cover cycle(s) {', '.join(map(str, uncovered[:10]))}"
                        f"{' …' if len(uncovered) > 10 else ''}: check the time offset.")
    cops = np.array([c.COP for c in out if np.isfinite(c.COP)])
    if cops.size and np.nanmax(cops) > 50:
        warnings.append(f"COP_mat up to {np.nanmax(cops):.3g} is implausibly high: check ρ, c_p, the time offset and "
                        "the loop area.")
    if any(np.isfinite(c.dT_heat_K) and c.dT_heat_K <= 0 for c in out):
        warnings.append("No temperature rise on loading in some cycles: check the time offset / sensor channel.")
    params = {"offset_s": offset_s, "rho_g_cm3": rho_g_cm3, "cp_J_per_gK": cp_J_per_gK,
              "baseline_window_s": baseline_window_s, "baseline_window_auto": baseline_window_s is None,
              "lag_frac": lag_frac, **{f"cyclic_{k}": v for k, v in res.params.items()}}
    return CaloricResult(out, warnings, params)


def caloric_dataset(result: CaloricResult, name: str = "Elastocaloric per cycle") -> Dataset:
    """Derived Dataset: cycle vs ΔT heating/cooling, COP, cooling power, work."""
    c = result.column
    ds = Dataset.from_arrays(name, {
        "Cycle": (c("cycle"), "count", "#"),
        "ΔT heating": (c("dT_heat_K"), "temperature_difference", "K"),
        "ΔT cooling": (c("dT_cool_K"), "temperature_difference", "K"),
        "COP_mat": (c("COP"), "dimensionless", ""),
        "Cooling power": (c("cooling_power_W_per_kg"), "specific_heat_flow", "W/kg"),
        "Input work W": (c("work_J_per_kg"), "specific_energy", "J/kg"),
        "Q_c": (c("Qc_J_per_kg"), "specific_energy", "J/kg"),
    }, kind="generic", default_x="Cycle", default_y="ΔT heating")
    ds.metadata["derived_from"] = "elastocaloric ΔT/COP"
    return ds
