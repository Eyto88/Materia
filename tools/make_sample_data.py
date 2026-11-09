#!/usr/bin/env python3
"""Generate realistic synthetic NiTi test files in ``sample_data/``.

Files (instrument-style exports, parseable by LabStudio's auto import):

* ``NiTi_DSC_Netzsch.txt``          Netzsch ASCII export: ``#`` preamble (mass, rate), header
  ``##Temp./°C;Time/min;DSC/(mW/mg)``, semicolon separated, exo-up sign convention.
  Program: −80 °C iso 5 min → heat 10 K/min → 100 °C iso 5 min → cool 10 K/min → −80 °C.
  Heating: single B19'→B2 peak; cooling: two-step B2→R→B19'.
* ``NiTi_tensile_failure_Instron.csv`` Bluehill-style preamble, header + units row
  (``Time,Extension,Load`` / ``(s),(mm),(kN)``), Ø 1 mm wire, 50 mm gauge, to fracture.
* ``NiTi_compression.csv``          Ø 3 mm × 6 mm cylinder, machine sign convention (negative
  load/extension in compression), header with units in parentheses.
* ``NiTi_superelastic_cyclic_50cycles.csv`` ``Cycle,Time (s),Strain (%),Stress (MPa)``, 50
  strain-controlled cycles to 6 % with functional fatigue.
* ``generic_xy.txt``                tab separated, decimal comma (elastocaloric ΔT vs time).
* ``NiTi_tensile_25C.csv``, ``NiTi_tensile_40C.csv``, ``NiTi_tensile_55C.csv``  superelastic loading
  curves (Instron style, Ø 1 mm, 50 mm) at three test temperatures with σ_SIM = 400 + 6.5·(T − 25) MPa
  and ε_tr = 5.0 % (Clausius–Clapeyron demo: dσ/dT = 6.5 MPa/K → ΔS = −50.39 J/(kg·K)).
* ``NiTi_elastocaloric_temperature_log.csv``  thermocouple log (Time (s), Temperature (°C), 5 Hz) on
  the same clock as the 50-cycle test (offset 0 s): a sin² pulse of +ΔT_h(N) during loading and
  −ΔT_c(N) during unloading (occupying 15–85 % of each half-cycle, flat 22 °C in between),
  ΔT_h = 8.0 − 2.5 f, ΔT_c = 7.0 − 2.0 f, f = 1 − exp(−(N − 1)/10); see :func:`caloric_truth`.

The encoded "true" parameters are the module-level ``TRUE_*`` dicts / functions below; the tests
import them. Peak shapes are split Gaussians so the tangent onset is analytic: for a half-Gaussian
of width σ the inflection tangent meets the baseline at μ ∓ 2σ, and the area is
h·√(π/2)·(σ_left + σ_right).

Usage:  python tools/make_sample_data.py [output_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_data"

# =========================================================================== DSC
TRUE_DSC = {
    "mass_mg": 15.20,
    "heating_rate_K_per_min": 10.0,
    "T_min": -80.0, "T_max": 100.0, "iso_min": 5.0,
    # peaks: (mu °C, sigma_low K, sigma_high K, enthalpy J/g); onset/endset at mu ∓ 2 sigma
    "A": (15.0, 4.0, 3.0, 20.0),     # heating, endothermic   -> As 7, Ap 15, Af 21
    "R": (12.0, 3.0, 2.5, 6.0),      # cooling, exothermic    -> Rs 17, Rp 12, Rf 6
    "M": (-30.0, 6.0, 4.0, 14.0),    # cooling, exothermic    -> Ms -22, Mp -30, Mf -42
}


def dsc_true_temperatures() -> dict[str, float]:
    """Analytic SMA temperatures (°C) encoded in the DSC sample file."""
    A, R, M = TRUE_DSC["A"], TRUE_DSC["R"], TRUE_DSC["M"]
    return {
        "As": A[0] - 2 * A[1], "Ap": A[0], "Af": A[0] + 2 * A[2],
        "Rs": R[0] + 2 * R[2], "Rp": R[0], "Rf": R[0] - 2 * R[1],
        "Ms": M[0] + 2 * M[2], "Mp": M[0], "Mf": M[0] - 2 * M[1],
    }


def _split_gauss(T, mu, s_lo, s_hi, area_KWg):
    h = area_KWg / (np.sqrt(np.pi / 2) * (s_lo + s_hi))
    s = np.where(T < mu, s_lo, s_hi)
    return h * np.exp(-0.5 * ((T - mu) / s) ** 2)


def make_dsc(rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T °C, time min, DSC mW/mg) with the Netzsch exo-up sign convention."""
    p = TRUE_DSC
    beta = p["heating_rate_K_per_min"]          # K/min
    dt = 0.02                                    # min  (1.2 s -> 0.2 K)
    t_iso = np.arange(0, p["iso_min"], dt)
    span = p["T_max"] - p["T_min"]
    t_ramp = np.arange(0, span / beta, dt)
    segs = []
    t0 = 0.0
    # iso low
    segs.append((t0 + t_iso, np.full(t_iso.size, p["T_min"]), 0)); t0 += p["iso_min"]
    segs.append((t0 + t_ramp, p["T_min"] + beta * t_ramp, +1)); t0 += span / beta
    segs.append((t0 + t_iso, np.full(t_iso.size, p["T_max"]), 0)); t0 += p["iso_min"]
    segs.append((t0 + t_ramp, p["T_max"] - beta * t_ramp, -1)); t0 += span / beta
    Ts, ts, qs = [], [], []
    for t, T, d in segs:
        # endo-positive excess heat flow (W/g = mW/mg); ΔH[J/g] = ∫q dT / β[K/s]
        bs = beta / 60.0
        if d > 0:
            q_endo = _split_gauss(T, *p["A"][:3], p["A"][3] * bs)
            base = 0.020 + 0.00025 * (T - 20)        # heat-capacity-like sloping baseline
            hook = 0.08 * np.exp(-(t - t[0]) / 0.15)  # start-up transient (endo-like on heating)
        elif d < 0:
            q_endo = -(_split_gauss(T, *p["R"][:3], p["R"][3] * bs)
                       + _split_gauss(T, *p["M"][:3], p["M"][3] * bs))
            base = -0.025 - 0.0002 * (T - 20)
            hook = -0.08 * np.exp(-(t - t[0]) / 0.15)
        else:
            q_endo = np.zeros_like(T)
            base = np.full(T.size, 0.0)
            hook = 0.0
        q = q_endo + base + hook
        Tm = T + rng.normal(0, 0.005, T.size)
        Ts.append(Tm); ts.append(t); qs.append(q)
    T = np.concatenate(Ts); t = np.concatenate(ts); q = np.concatenate(qs)
    q = q + rng.normal(0, 0.0008, q.size)
    return T, t, -q        # Netzsch default sign: exo up (endotherms point down)


def write_dsc(path: Path, rng) -> None:
    T, t, q = make_dsc(rng)
    p = TRUE_DSC
    lines = [
        "#FORMAT:;NETZSCH5",
        "#FTYPE:;ANSI",
        "#IDENTITY:;NiTi-SE-01",
        "#DECIMAL:;POINT",
        "#SEPARATOR:;SEMICOLON",
        "#INSTRUMENT:;NETZSCH DSC 214 Polyma",
        "#PROJECT:;MEng elastocaloric NiTi",
        "#DATE/TIME:;06.11.2026 10:42",
        "#SAMPLE:;NiTi wire Ni50.8Ti49.2 (superelastic, annealed 500C 30min)",
        f"#SAMPLE MASS /mg:;{p['mass_mg']:.3f}",
        "#REFERENCE:;Empty Al pan",
        "#CRUCIBLE:;DSC/TG pan Al, pierced lid",
        "#ATMOSPHERE:;N2 / 40 ml/min",
        f"#HEATING RATE /(K/min):;{p['heating_rate_K_per_min']:.1f}",
        f"#RANGE:;{p['T_min']:.0f}°C/{p['heating_rate_K_per_min']:.1f}(K/min)/{p['T_max']:.0f}°C",
        "#SEGMENTS:;4",
        "#EXO:;up",
        "#TYPE OF DATA:;DSC",
        "##Temp./°C;Time/min;DSC/(mW/mg)",
    ]
    body = [f"{a:.3f};{b:.4f};{c:.5f}" for a, b, c in zip(T, t, q)]
    path.write_text("\n".join(lines + body) + "\n", encoding="utf-8")


# =========================================================================== tensile
TRUE_TENSILE = {
    "diameter_mm": 1.0, "gauge_length_mm": 50.0, "crosshead_mm_per_min": 2.0, "sample_hz": 10.0,
    "E_GPa": 60.0,                    # austenite elastic modulus (slope 600 MPa/%)
    "sigma_SIM_MPa": 450.0, "strain_SIM_pct": 0.75,
    "plateau_end": (6.5, 510.0),      # end of plateau (strain %, MPa)
    "UTS_MPa": 1300.0, "strain_at_UTS_pct": 12.6,
    "fracture_strain_pct": 14.0, "fracture_stress_MPa": 1270.0,
}
# knots of the clean engineering curve (strain %, MPa); lightly smoothed at the knots
_TENSILE_KNOTS = [(0.0, 0.0), (0.75, 450.0), (6.5, 510.0), (9.0, 1010.0), (10.0, 1150.0),
                  (11.0, 1240.0), (12.0, 1288.0), (12.6, 1300.0), (13.3, 1294.0), (14.0, 1270.0)]


def _smooth_knots(knots, eps, width):
    """Piecewise-linear knots convolved with a Gaussian of ``width`` (strain %)."""
    xk, yk = np.array(knots).T
    fine = np.arange(xk[0] - 4 * width, xk[-1] + 4 * width, width / 20)
    yf = np.interp(fine, xk, yk, left=np.nan, right=np.nan)
    # linear extrapolation beyond ends
    lo = fine < xk[0]
    yf[lo] = yk[0] + (fine[lo] - xk[0]) * (yk[1] - yk[0]) / (xk[1] - xk[0])
    hi = fine > xk[-1]
    yf[hi] = yk[-1] + (fine[hi] - xk[-1]) * (yk[-1] - yk[-2]) / (xk[-1] - xk[-2])
    k = np.arange(-80, 81) * (width / 20)
    g = np.exp(-0.5 * (k / width) ** 2); g /= g.sum()
    ys = np.convolve(yf, g, mode="same")
    return np.interp(eps, fine, ys)


def tensile_clean_curve(eps):
    return _smooth_knots(_TENSILE_KNOTS, eps, 0.08)


def tensile_true_derived() -> dict[str, float]:
    """Numerically exact values of the clean tensile curve (yield, toughness)."""
    e = np.linspace(0, TRUE_TENSILE["fracture_strain_pct"], 200001)
    s = tensile_clean_curve(e)
    s[0] = 0.0
    Ep = TRUE_TENSILE["E_GPa"] * 10
    f = s - Ep * (e - 0.2)
    i = np.flatnonzero((f[:-1] > 0) & (f[1:] <= 0))[0]
    tough = float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(e / 100)))
    return {"yield_MPa": float(s[i]), "yield_strain_pct": float(e[i]), "toughness_MJ_m3": tough,
            "UTS_MPa": float(s.max())}


def write_tensile(path: Path, rng) -> None:
    p = TRUE_TENSILE
    A = np.pi * p["diameter_mm"] ** 2 / 4
    L0 = p["gauge_length_mm"]
    rate = p["crosshead_mm_per_min"] / 60.0
    ef = p["fracture_strain_pct"]
    t_f = ef / 100 * L0 / rate
    t = np.arange(0, t_f + 1e-9, 1 / p["sample_hz"])
    ext = rate * t
    eps = ext / L0 * 100
    sig = tensile_clean_curve(eps)
    sig[0] = 0.0
    # fracture: a few points after the break with ~zero load
    t_after = t[-1] + np.arange(1, 16) / p["sample_hz"]
    ext_after = ext[-1] + rate * (t_after - t[-1])
    sig_after = np.r_[np.array([380.0, 25.0]), np.zeros(13)]
    t = np.r_[t, t_after]; ext = np.r_[ext, ext_after]; sig = np.r_[sig, sig_after]
    load_kN = sig * A / 1000.0 + rng.normal(0, 0.0004, sig.size)   # ±0.4 N noise
    ext = ext + rng.normal(0, 0.0005, ext.size)
    ext[0] = 0.0
    lines = [
        "Instron Bluehill Universal - Raw data export",
        "Method: NiTi_wire_tension_to_failure.im_tens",
        "Sample ID: NiTi-SE-wire-T01",
        "Material: NiTi Ni50.8Ti49.2 wire (SE grade, straight annealed)",
        f"Diameter: {p['diameter_mm']:.3f} mm",
        f"Gauge length: {L0:.2f} mm",
        f"Rate 1: {p['crosshead_mm_per_min']:.2f} mm/min",
        "Test temperature: 22 °C",
        "",
        "Time,Extension,Load",
        "(s),(mm),(kN)",
    ]
    body = [f"{a:.2f},{b:.5f},{c:.5f}" for a, b, c in zip(t, ext, load_kN)]
    path.write_text("\n".join(lines + body) + "\n", encoding="utf-8")


# =========================================================================== compression
TRUE_COMPRESSION = {
    "diameter_mm": 3.0, "length_mm": 6.0, "sample_hz": 10.0, "rate_mm_per_min": 0.36,
    "E_GPa": 55.0, "toe_pct": 0.05,
    "sigma_SIM_MPa": 650.0,
    "max_stress_MPa": 1650.0, "strain_at_max_pct": 10.0,
}
_COMP_KNOTS = [(0.0, 0.0), (0.05, 5.0), (0.05 + 650 / 550, 650.0), (4.0, 760.0), (5.5, 1050.0),
               (7.0, 1300.0), (8.5, 1500.0), (10.0, 1650.0)]


def compression_clean_curve(eps):
    return np.maximum(_smooth_knots(_COMP_KNOTS, eps, 0.03), 0.0)


def write_compression(path: Path, rng) -> None:
    p = TRUE_COMPRESSION
    A = np.pi * p["diameter_mm"] ** 2 / 4
    L0 = p["length_mm"]
    rate = p["rate_mm_per_min"] / 60
    t_end = p["strain_at_max_pct"] / 100 * L0 / rate
    t = np.arange(0, t_end + 1e-9, 1 / p["sample_hz"])
    ext = rate * t
    sig = compression_clean_curve(ext / L0 * 100)
    sig[0] = 0
    F = -(sig * A) + rng.normal(0, 0.8, t.size)    # N, compressive negative
    d = -(ext + rng.normal(0, 0.0002, t.size))
    d[0] = 0.0
    lines = [
        "Instron 5969 - Compression test export",
        "Specimen: NiTi-SE-C01 (cylinder)",
        f"Diameter (mm): {p['diameter_mm']:.3f}",
        f"Gauge length: {L0:.3f} mm",
        "Sign convention: compression negative",
        "",
        "Time (s),Extension (mm),Load (N)",
    ]
    body = [f"{a:.2f},{b:.5f},{c:.2f}" for a, b, c in zip(t, d, F)]
    path.write_text("\n".join(lines + body) + "\n", encoding="utf-8")


def compression_true_derived() -> dict[str, float]:
    p = TRUE_COMPRESSION
    e = np.linspace(0, p["strain_at_max_pct"], 200001)
    s = compression_clean_curve(e)
    Ep = p["E_GPa"] * 10
    eps0 = p["toe_pct"] - 5.0 / Ep
    f = s - Ep * (e - eps0 - 0.2)
    i = np.flatnonzero((f[:-1] > 0) & (f[1:] <= 0))[0]
    return {"yield_MPa": float(s[i]), "max_stress_MPa": float(s.max()),
            "strain_intercept_pct": float(eps0)}


# =========================================================================== cyclic
TRUE_CYCLIC = {
    "n_cycles": 50, "max_strain_pct": 6.0, "strain_rate_pct_per_s": 0.05, "sample_hz": 4.0,
    "E_A_MPa_per_pct": 500.0, "E_M_MPa_per_pct": 300.0, "plateau_slope_MPa_per_pct": 8.0,
    "res_first_pct": 0.15, "res_extra_pct": 1.05, "res_tau": 8.0,
    "sms0": 420.0, "sms_drop": 90.0, "sas0": 220.0, "sas_drop": 30.0, "plat_tau": 10.0,
    "smax0": 640.0, "smax_drop": 70.0,
}


def cyclic_truth(N: int) -> dict[str, float]:
    """Encoded per-cycle parameters (cycle N = 1..50)."""
    p = TRUE_CYCLIC
    f_res = 1 - np.exp(-(N - 1) / p["res_tau"])
    f_pl = 1 - np.exp(-(N - 1) / p["plat_tau"])
    res = p["res_first_pct"] + p["res_extra_pct"] * f_res
    start = 0.0 if N == 1 else cyclic_truth(N - 1)["residual_strain_pct"]
    sms = p["sms0"] - p["sms_drop"] * f_pl
    sas = p["sas0"] - p["sas_drop"] * f_pl
    return {"start_strain_pct": start, "residual_strain_pct": res, "sigma_ms_MPa": sms,
            "sigma_as_MPa": sas, "delta_sigma_MPa": sms - sas,
            "max_stress_MPa": p["smax0"] - p["smax_drop"] * f_pl}


def _smin(a, b, k=4.0):
    return -k * np.logaddexp(-a / k, -b / k)


def _smax(a, b, k=4.0):
    return k * np.logaddexp(a / k, b / k)


def cyclic_branches(N: int, eps_load, eps_unload):
    """Clean loading / unloading stresses (MPa) of cycle N on given strain arrays."""
    p = TRUE_CYCLIC
    tr = cyclic_truth(N)
    s0, r, emax = tr["start_strain_pct"], tr["residual_strain_pct"], p["max_strain_pct"]
    ep = max(s0, r) + 0.5 * (emax - max(s0, r))
    k = p["plateau_slope_MPa_per_pct"]
    EA, EM = p["E_A_MPa_per_pct"], p["E_M_MPa_per_pct"]

    def mart(e):
        return tr["max_stress_MPa"] - EM * (emax - e)

    sl = _smax(_smin(EA * (eps_load - s0), tr["sigma_ms_MPa"] + k * (eps_load - ep)), mart(eps_load))
    su = _smax(_smin(EA * (eps_unload - r), tr["sigma_as_MPa"] + k * (eps_unload - ep)), mart(eps_unload))
    return sl, su


def cyclic_true_energy(N: int) -> float:
    """Dissipated energy (MJ/m³) of the clean cycle N (W_load − W_unload)."""
    p = TRUE_CYCLIC
    tr = cyclic_truth(N)
    el = np.linspace(tr["start_strain_pct"], p["max_strain_pct"], 20001)
    eu = np.linspace(tr["residual_strain_pct"], p["max_strain_pct"], 20001)
    sl, su = cyclic_branches(N, el, eu)
    sl = sl - sl[0] * np.linspace(1, 0, el.size)      # pin to σ=0 at the start, as in the file
    su = su - su[0] * np.linspace(1, 0, eu.size)

    def tz(y, x):
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x / 100)))
    return tz(sl, el) - tz(su, eu)


def make_cyclic(rng):
    p = TRUE_CYCLIC
    rate = p["strain_rate_pct_per_s"]
    dt = 1 / p["sample_hz"]
    cyc, tt, ee, ss = [], [], [], []
    t0 = 0.0
    for N in range(1, p["n_cycles"] + 1):
        tr = cyclic_truth(N)
        s0, r, emax = tr["start_strain_pct"], tr["residual_strain_pct"], p["max_strain_pct"]
        n_l = int(round((emax - s0) / rate / dt))
        n_u = int(round((emax - r) / rate / dt))
        el = np.linspace(s0, emax, n_l + 1)
        eu = np.linspace(emax, r, n_u + 1)[1:]
        sl, su = cyclic_branches(N, el, eu)
        # pin exact zero stress at the end points (smooth min/max leave ~1 MPa offsets)
        sl = sl - sl[0] * np.linspace(1, 0, el.size)
        su = su - su[-1] * np.linspace(0, 1, eu.size)
        e = np.r_[el, eu] if N == 1 else np.r_[el[1:], eu]
        s = np.r_[sl, su] if N == 1 else np.r_[sl[1:], su]
        t = t0 + dt * np.arange(1 if N > 1 else 0, e.size + (1 if N > 1 else 0))
        t0 = t[-1]
        cyc.append(np.full(e.size, N)); tt.append(t); ee.append(e); ss.append(s)
    C = np.concatenate(cyc); t = np.concatenate(tt); e = np.concatenate(ee); s = np.concatenate(ss)
    e = e + rng.normal(0, 0.002, e.size)
    s = s + rng.normal(0, 1.0, s.size)
    return C, t, e, s


def write_cyclic(path: Path, rng) -> None:
    C, t, e, s = make_cyclic(rng)
    lines = [
        "# Superelastic cyclic test (functional fatigue), strain-controlled triangle 0 -> 6 %",
        "# Specimen: NiTi-SE-wire-C01, diameter 1.0 mm, gauge 50 mm, 22 °C",
        "# Strain rate: 5e-4 1/s",
        "Cycle,Time (s),Strain (%),Stress (MPa)",
    ]
    body = [f"{int(a)},{b:.2f},{c:.4f},{d:.2f}" for a, b, c, d in zip(C, t, e, s)]
    path.write_text("\n".join(lines + body) + "\n", encoding="utf-8")


def cyclic_timing() -> list[tuple[float, float, float]]:
    """(t_start, t_peak, t_end) in s of every cycle of the cyclic sample (same clock as the file)."""
    p = TRUE_CYCLIC
    rate = p["strain_rate_pct_per_s"]
    dt = 1 / p["sample_hz"]
    out = []
    t0 = 0.0
    for N in range(1, p["n_cycles"] + 1):
        tr = cyclic_truth(N)
        n_l = int(round((p["max_strain_pct"] - tr["start_strain_pct"]) / rate / dt))
        n_u = int(round((p["max_strain_pct"] - tr["residual_strain_pct"]) / rate / dt))
        out.append((t0, t0 + dt * n_l, t0 + dt * (n_l + n_u)))
        t0 = t0 + dt * (n_l + n_u)
    return out


# =========================================================================== elastocaloric
TRUE_CALORIC = {"T0_C": 22.0, "dTh0_K": 8.0, "dTh_drop_K": 2.5, "dTc0_K": 7.0, "dTc_drop_K": 2.0, "tau": 10.0,
                "sample_hz": 5.0, "noise_K": 0.01, "rho_g_cm3": 6.45, "cp_J_per_gK": 0.45, "offset_s": 0.0,
                "active": (0.15, 0.85)}   # T pulse occupies this fraction of each half-cycle (flat baseline between)


def caloric_truth(N: int) -> dict[str, float]:
    """Encoded per-cycle elastocaloric values of the temperature-log sample (cycle N = 1..50)."""
    p = TRUE_CALORIC
    f = 1 - np.exp(-(N - 1) / p["tau"])
    dTh = p["dTh0_K"] - p["dTh_drop_K"] * f
    dTc = p["dTc0_K"] - p["dTc_drop_K"] * f
    t0, tp, t1 = cyclic_timing()[N - 1]
    W = cyclic_true_energy(N)                                   # MJ/m³
    Wkg = W * 1e6 / (p["rho_g_cm3"] * 1000)
    Qc = p["cp_J_per_gK"] * 1000 * dTc
    return {"dT_heat_K": dTh, "dT_cool_K": -dTc, "work_MJ_m3": W, "work_J_per_kg": Wkg, "Qc_J_per_kg": Qc,
            "COP": Qc / Wkg, "frequency_Hz": 1 / (t1 - t0), "cooling_power_W_per_kg": Qc / (t1 - t0)}


def make_temperature_log(rng) -> tuple[np.ndarray, np.ndarray]:
    p = TRUE_CALORIC
    timing = cyclic_timing()
    t = np.arange(0, timing[-1][2] + 1e-9, 1 / p["sample_hz"])
    T = np.full(t.size, p["T0_C"])
    for N, (t0, tp, t1) in enumerate(timing, start=1):
        tr = caloric_truth(N)
        a, b = p["active"]
        u = (t - t0) / (tp - t0)
        mL = (u >= a) & (u <= b)
        T[mL] += tr["dT_heat_K"] * np.sin(np.pi * (u[mL] - a) / (b - a)) ** 2
        v = (t - tp) / (t1 - tp)
        mU = (v >= a) & (v <= b)
        T[mU] += tr["dT_cool_K"] * np.sin(np.pi * (v[mU] - a) / (b - a)) ** 2
    return t, T + rng.normal(0, p["noise_K"], t.size)


def write_temperature_log(path: Path, rng) -> None:
    t, T = make_temperature_log(rng)
    lines = ["# Thermocouple log (type K, 0.1 mm, spot-welded at mid-gauge) for NiTi-SE-wire-C01",
             "# Synchronised with NiTi_superelastic_cyclic_50cycles.csv (time offset 0 s), 5 Hz",
             "Time (s),Temperature (°C)"]
    lines += [f"{a:.2f},{b:.3f}" for a, b in zip(t, T)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


TRUE_CC = {"temperatures_C": (25.0, 40.0, 55.0), "sigma_SIM_25C_MPa": 400.0, "dsigma_dT_MPa_per_K": 6.5,
           "eps_tr_pct": 5.0, "E_GPa": 60.0, "plateau_slope_MPa_per_pct": 10.0, "E_M_MPa_per_pct": 200.0,
           "martensite_pct": 2.5, "rho_g_cm3": 6.45, "diameter_mm": 1.0, "gauge_length_mm": 50.0}


def cc_sigma_SIM(T_C: float) -> float:
    p = TRUE_CC
    return p["sigma_SIM_25C_MPa"] + p["dsigma_dT_MPa_per_K"] * (T_C - 25.0)


def cc_true_entropy() -> float:
    """ΔS = −(dσ/dT)·ε_tr/ρ in J/(kg·K) encoded in the three tensile files."""
    p = TRUE_CC
    return -p["dsigma_dT_MPa_per_K"] * 1e6 * p["eps_tr_pct"] / 100 / (p["rho_g_cm3"] * 1000)


def cc_true_latent(T_ref_C: float = 40.0) -> float:
    """L = T·|ΔS| in J/g at T_ref (default: mean test temperature 40 °C)."""
    return (T_ref_C + 273.15) * abs(cc_true_entropy()) / 1000


def _cc_knots(T_C: float):
    p = TRUE_CC
    sS = cc_sigma_SIM(T_C)
    eS = sS / (p["E_GPa"] * 10)
    e1, s1 = eS + p["eps_tr_pct"], sS + p["plateau_slope_MPa_per_pct"] * p["eps_tr_pct"]
    e2, s2 = e1 + p["martensite_pct"], s1 + p["E_M_MPa_per_pct"] * p["martensite_pct"]
    return [(0.0, 0.0), (eS, sS), (e1, s1), (e2, s2)]


def write_tensile_T(path: Path, rng, T_C: float) -> None:
    p = TRUE_CC
    A = np.pi * p["diameter_mm"] ** 2 / 4
    L0 = p["gauge_length_mm"]
    rate = 2.0 / 60.0                        # mm/s
    knots = _cc_knots(T_C)
    e_end = knots[-1][0]
    t = np.arange(0, e_end / 100 * L0 / rate + 1e-9, 0.1)
    ext = rate * t
    sig = _smooth_knots(knots, ext / L0 * 100, 0.03)
    sig[0] = 0.0
    load = sig * A / 1000 + rng.normal(0, 0.0003, t.size)
    ext = ext + rng.normal(0, 0.0003, t.size)
    ext[0] = 0.0
    lines = ["Instron Bluehill Universal - Raw data export",
             "Method: NiTi_wire_superelastic_T_series.im_tens",
             f"Sample ID: NiTi-SE-wire-CC{int(T_C)}",
             f"Diameter: {p['diameter_mm']:.3f} mm",
             f"Gauge length: {L0:.2f} mm",
             "Rate 1: 2.00 mm/min",
             f"Test temperature: {T_C:.0f} °C",
             "",
             "Time,Extension,Load",
             "(s),(mm),(kN)"]
    body = [f"{a:.2f},{b:.5f},{c:.5f}" for a, b, c in zip(t, ext, load)]
    path.write_text("\n".join(lines + body) + "\n", encoding="utf-8")


# =========================================================================== generic
def write_generic(path: Path, rng) -> None:
    """Elastocaloric ΔT response (tab separated, decimal comma)."""
    t = np.arange(0, 120, 0.1)
    T = 22.0 + np.zeros_like(t)
    for t_load, sgn in ((10, 1), (40, -1), (70, 1), (100, -1)):
        m = t >= t_load
        T[m] += sgn * 16.0 * np.exp(-(t[m] - t_load) / 4.0) * (1 - np.exp(-(t[m] - t_load) / 0.3))
    T += rng.normal(0, 0.03, t.size)
    lines = ["Time (s)\tTemperature (°C)"]
    lines += [f"{a:.1f}".replace(".", ",") + "\t" + f"{b:.3f}".replace(".", ",") for a, b in zip(t, T)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(out: Path = OUT, seed: int = 2026) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    files = {
        "NiTi_DSC_Netzsch.txt": write_dsc,
        "NiTi_tensile_failure_Instron.csv": write_tensile,
        "NiTi_compression.csv": write_compression,
        "NiTi_superelastic_cyclic_50cycles.csv": write_cyclic,
        "generic_xy.txt": write_generic,
    }
    paths = []
    for name, fn in files.items():
        pth = out / name
        fn(pth, rng)
        paths.append(pth)
    # v1.1 elastocaloric demo files (own generator so the files above stay byte-identical)
    rng2 = np.random.default_rng(seed + 11)
    for T_C in TRUE_CC["temperatures_C"]:
        pth = out / f"NiTi_tensile_{int(T_C)}C.csv"
        write_tensile_T(pth, rng2, T_C)
        paths.append(pth)
    pth = out / "NiTi_elastocaloric_temperature_log.csv"
    write_temperature_log(pth, rng2)
    paths.append(pth)
    return paths


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    for pth in main(target):
        print("wrote", pth)
