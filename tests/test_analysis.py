"""Tests for module C (analysis): synthetic curves with known answers, sample files, headless GUI."""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_sample_data as MSD  # noqa: E402

from labstudio.core import analysis as A  # noqa: E402
from labstudio.core.dataset import Dataset, DatasetStore  # noqa: E402

SQRT_E = np.exp(-0.5)


# =========================================================================== general
def test_linear_fit_exact_and_range():
    x = np.linspace(0, 10, 101)
    y = 3.0 * x - 2.0
    y[x > 5] = 100.0                     # outside the fit range
    f = A.linear_fit(x, y, 0, 5)
    assert f.slope == pytest.approx(3.0)
    assert f.intercept == pytest.approx(-2.0)
    assert f.r2 == pytest.approx(1.0)
    assert f.n == 51
    assert f.x_at(7.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        A.linear_fit([1.0], [2.0])


def test_integrate_xy_none_and_linear_baseline():
    x = np.linspace(0, np.pi, 2001)
    r = A.integrate_xy(x, np.sin(x))
    assert r.area == pytest.approx(2.0, rel=1e-5)
    # peak on a sloping background: linear baseline removes the trapezoid
    y = np.sin(x) + 1.0 + 0.5 * x
    r = A.integrate_xy(x, y, baseline="linear")
    assert r.area == pytest.approx(2.0, rel=1e-5)
    r = A.integrate_xy(x, np.ones_like(x), 1.0, 2.0)
    assert r.area == pytest.approx(1.0, abs=2e-3)
    # unsorted input is sorted by default
    o = np.random.default_rng(0).permutation(x.size)
    assert A.integrate_xy(x[o], np.sin(x)[o]).area == pytest.approx(2.0, rel=1e-5)


def test_find_peaks_xy_gaussian_fwhm_and_valleys():
    x = np.linspace(0, 100, 5001)
    y = 2.0 * np.exp(-0.5 * ((x - 30) / 3) ** 2) + 1.0 * np.exp(-0.5 * ((x - 70) / 5) ** 2)
    pk = A.find_peaks_xy(x, y)
    assert [round(p.x, 1) for p in pk] == [30.0, 70.0]
    assert pk[0].fwhm == pytest.approx(2.3548 * 3, rel=2e-3)
    assert pk[1].prominence == pytest.approx(1.0, rel=1e-2)
    v = A.find_peaks_xy(x, -y, sign=-1)
    assert len(v) == 2 and v[0].y == pytest.approx(-2.0, rel=1e-3)
    assert len(A.find_peaks_xy(x, y, prominence=1.5)) == 1


# =========================================================================== DSC
def _split_gauss(T, mu, s_lo, s_hi, h):
    s = np.where(T < mu, s_lo, s_hi)
    return h * np.exp(-0.5 * ((T - mu) / s) ** 2)


def test_dsc_gaussian_onset_endset_enthalpy_analytic():
    # Gaussian: inflection at mu ± sigma, tangent meets the baseline at mu ∓ 2 sigma exactly
    T = np.linspace(-60, 80, 2801)
    mu, sig, h, beta = 12.0, 4.0, 0.30, 10.0
    hf = 0.03 + 0.0008 * T + h * np.exp(-0.5 * ((T - mu) / sig) ** 2)
    pk = A.analyse_dsc_segment(T, hf, beta, direction="heating", endo_up=True)
    assert len(pk) == 1
    p = pk[0]
    assert p.phase == "A" and p.label_start == "As" and p.label_end == "Af"
    assert p.onset_T == pytest.approx(mu - 2 * sig, abs=0.05)
    assert p.endset_T == pytest.approx(mu + 2 * sig, abs=0.05)
    assert p.peak_T == pytest.approx(mu, abs=0.02)
    true_H = h * sig * np.sqrt(2 * np.pi) / (beta / 60.0)      # J/g
    assert p.enthalpy_J_per_g == pytest.approx(true_H, rel=0.01)
    assert p.endothermic
    assert p.start_T < p.onset_T < p.peak_T < p.endset_T < p.end_T
    assert p.fwhm_K == pytest.approx(2.3548 * sig, abs=0.1)
    assert p.heating_rate_K_per_min == pytest.approx(beta)
    # overlays present and in hf units
    assert p.tangent_onset is not None and p.region_T.size > 10
    assert p.onset_hf == pytest.approx(0.03 + 0.0008 * p.onset_T, abs=2e-3)


def test_dsc_sign_autodetect_mass_and_time_axis():
    rng = np.random.default_rng(1)
    T = np.linspace(-60, 80, 1401)
    mu, sig, h, beta, mass = 20.0, 3.0, 0.4, 20.0, 12.5
    q = h * np.exp(-0.5 * ((T - mu) / sig) ** 2)            # W/g, endothermic
    hf_mW_exo_up = -(q * mass) + 0.2 + rng.normal(0, 0.01, T.size)   # mW, endo DOWN
    t = (T - T[0]) / (beta / 60.0)
    pk = A.analyse_dsc_segment(T, hf_mW_exo_up, None, mass, hf_unit="mW", time=t)
    assert len(pk) == 1
    true_H = h * sig * np.sqrt(2 * np.pi) / (beta / 60.0)
    assert pk[0].enthalpy_J_per_g == pytest.approx(true_H, rel=0.02)
    assert pk[0].enthalpy_J == pytest.approx(true_H * mass * 1e-3, rel=0.02)
    assert pk[0].onset_T == pytest.approx(mu - 2 * sig, abs=0.25)
    assert pk[0].heating_rate_K_per_min == pytest.approx(beta, rel=1e-6)
    assert A.detect_endo_up(T, hf_mW_exo_up, "heating") is False
    # absolute heat flow without mass: only J reported
    pk2 = A.analyse_dsc_segment(T, hf_mW_exo_up, beta, None, hf_unit="mW", endo_up=False)
    assert np.isnan(pk2[0].enthalpy_J_per_g)
    assert pk2[0].enthalpy_J == pytest.approx(true_H * mass * 1e-3, rel=0.02)


def test_dsc_two_step_cooling_named_R_and_M():
    T = np.linspace(80, -80, 3201)                          # cooling, time order
    beta = 10.0
    exo = _split_gauss(T, 12, 3, 2.5, 0.12) + _split_gauss(T, -30, 6, 4, 0.20)
    hf = 0.01 + exo                                          # exo up file convention
    pk = A.analyse_dsc_segment(T, hf, beta, endo_up=False)
    assert [p.phase for p in pk] == ["R", "M"]              # process order (decreasing T)
    R, M = pk
    assert not R.endothermic
    assert R.onset_T == pytest.approx(12 + 2 * 2.5, abs=0.1)  # Rs
    assert R.endset_T == pytest.approx(12 - 2 * 3, abs=0.1)   # Rf
    assert M.onset_T == pytest.approx(-30 + 2 * 4, abs=0.1)   # Ms
    assert M.endset_T == pytest.approx(-30 - 2 * 6, abs=0.1)  # Mf
    assert M.start_T > M.onset_T > M.peak_T > M.endset_T > M.end_T
    area = lambda h, a, b: h * np.sqrt(np.pi / 2) * (a + b) / (beta / 60)  # noqa: E731
    assert R.enthalpy_J_per_g == pytest.approx(area(0.12, 3, 2.5), rel=0.02)
    assert M.enthalpy_J_per_g == pytest.approx(area(0.20, 6, 4), rel=0.02)
    assert set(M.temperatures) == {"Ms", "Mp", "Mf"}


def test_dsc_baseline_options_and_validation():
    T = np.linspace(-50, 50, 1001)
    hf = 0.1 * np.exp(-0.5 * (T / 3) ** 2)
    for b in ("linear", "fit", "none"):
        pk = A.analyse_dsc_segment(T, hf, 10, direction="heating", baseline=b, endo_up=True)
        assert pk[0].enthalpy_J_per_g == pytest.approx(0.1 * 3 * np.sqrt(2 * np.pi) * 6, rel=0.02)
    with pytest.raises(ValueError):
        A.analyse_dsc_segment(T, hf, 10, baseline="spline")
    with pytest.raises(ValueError):
        A.analyse_dsc_segment(T, hf, None, direction="heating")      # no rate, no time
    assert A.analyse_dsc_segment(T, np.zeros_like(T) + 0.01, 10, direction="heating", endo_up=True) == []


def test_split_dsc_segments_with_isotherms():
    rate = 10 / 60.0
    t1 = np.arange(0, 300, 1.0)
    t_ramp = np.arange(0, 180 / rate, 1.0)
    T = np.r_[np.full(t1.size, -80.0), -80 + rate * t_ramp, np.full(t1.size, 100.0), 100 - rate * t_ramp]
    t = np.arange(T.size, dtype=float)
    seg = A.split_dsc_segments(T, t)
    assert [d for *_, d in seg.segments] == [1, -1]
    heating, cooling = seg
    assert heating.sum() == pytest.approx(t_ramp.size, abs=12)
    assert cooling.sum() == pytest.approx(t_ramp.size, abs=12)
    assert not (heating & cooling).any()
    assert not heating[:250].any() and not cooling[:250].any()       # isotherm ignored
    # without time column the index is used
    seg2 = A.split_dsc_segments(T)
    assert [d for *_, d in seg2.segments] == [1, -1]


def test_analyse_dsc_full_run_hysteresis_and_dTad():
    rate = 10.0
    b = rate / 60
    T_h = np.linspace(-80, 100, 1801)
    T_c = T_h[::-1]
    h_A = 20 * b / (np.sqrt(np.pi / 2) * 7)
    q_h = 0.01 + _split_gauss(T_h, 15, 4, 3, h_A)
    q_c = -0.01 - _split_gauss(T_c, -30, 6, 4, 14 * b / (np.sqrt(np.pi / 2) * 10))
    T = np.r_[T_h, T_c]
    q = np.r_[q_h, q_c]                                       # endo up
    t = np.arange(T.size) * (0.1 / b)
    res = A.analyse_dsc(T, q, rate, time=t)
    assert res.endo_up is True
    assert res.temperatures["As"] == pytest.approx(7, abs=0.15)
    assert res.temperatures["Af"] == pytest.approx(21, abs=0.15)
    assert res.temperatures["Ms"] == pytest.approx(-22, abs=0.15)
    assert res.temperatures["Mf"] == pytest.approx(-42, abs=0.15)
    assert res.enthalpies["A"] == pytest.approx(20, rel=0.02)
    assert res.enthalpies["M"] == pytest.approx(14, rel=0.02)
    assert res.hysteresis["Ap − Mp"] == pytest.approx(45, abs=0.2)
    assert res.hysteresis["Af − Ms"] == pytest.approx(43, abs=0.3)
    assert res.delta_T_ad["A"] == pytest.approx(res.enthalpies["A"] / 0.45)
    rows = res.rows()
    assert ("As", res.temperatures["As"], "°C") in rows
    assert any(r[0] == "ΔH A" and r[2] == "J/g" for r in rows)


def test_delta_T_ad_estimate():
    assert A.delta_T_ad_estimate(20.0) == pytest.approx(20 / 0.45)
    assert A.delta_T_ad_estimate(-12.0, 0.5) == pytest.approx(24.0)
    v = A.delta_T_ad_estimate(20.0, 0.45, T_C=50.0, T_transformation_C=20.0)
    assert v == pytest.approx(20 / 0.45 * 323.15 / 293.15)


# =========================================================================== mechanical
def test_stress_strain_conversion_and_compression_signs():
    F = np.array([0.0, 100.0, 200.0])
    d = np.array([0.1, 0.6, 1.1])
    e, s = A.stress_strain(F, d, area_mm2=2.0, gauge_mm=50.0)
    np.testing.assert_allclose(e, [0, 1, 2])
    np.testing.assert_allclose(s, [0, 50, 100])
    e, s = A.stress_strain(-F, -d, 2.0, 50.0, compression=True)
    np.testing.assert_allclose(e, [0, 1, 2])
    np.testing.assert_allclose(s, [0, 50, 100])
    with pytest.raises(ValueError):
        A.stress_strain(F, d, 0, 50)


def _bilinear(E_GPa=200.0, sy=400.0, H_MPa_pct=20.0, e_end=10.0, n=4001, noise=0.0, seed=0):
    e = np.linspace(0, e_end, n)
    Ep = E_GPa * 10
    s = np.minimum(Ep * e, sy + H_MPa_pct * (e - sy / Ep))
    if noise:
        s = s + np.random.default_rng(seed).normal(0, noise, n)
    return e, s


def test_tensile_bilinear_known_E_yield_uts_fracture_toughness():
    e, s = _bilinear(noise=0.5)
    # fracture: sudden drop to ~0 after 10 %
    e = np.r_[e, 10.0025, 10.005, 10.0075]
    s = np.r_[s, 30.0, 1.0, 0.5]
    r = A.tensile_properties(e, s)
    assert r.E_GPa == pytest.approx(200.0, rel=0.01)
    # offset line 2000(ε−0.2) meets 400+20(ε−0.2): ε = 0.2 + 400/1980
    ey = 0.2 + 400 / 1980
    assert r.yield_strain_pct == pytest.approx(ey, abs=0.01)
    assert r.yield_strength_MPa == pytest.approx(400 + 20 * (ey - 0.2), abs=2.0)
    s_uts = 400 + 20 * (10 - 0.2)
    assert r.uts_MPa == pytest.approx(s_uts, abs=2.5)
    assert r.fracture_detected
    assert r.fracture_strain_pct == pytest.approx(10.0, abs=0.01)
    # toughness: triangle + trapezoid, MPa·(mm/mm) = MJ/m³
    tough = (0.5 * 0.2 * 400 + 0.5 * (400 + s_uts) * 9.8) / 100
    assert r.toughness_MJ_m3 == pytest.approx(tough, rel=0.005)
    assert r.elongation_at_break_pct == pytest.approx(10 - r.fracture_stress_MPa / 2000, abs=0.01)
    assert not r.superelastic
    assert r.modulus_line is not None and r.offset_line is not None
    assert any(q.startswith("Young") for q, *_ in r.rows())


def test_tensile_manual_modulus_range_and_toe():
    e, s = _bilinear(E_GPa=100.0, sy=300.0, H_MPa_pct=10.0)
    e = e + 0.05                                              # toe / slack: curve shifted by 0.05 %
    e = np.r_[0.0, e]
    s = np.r_[0.0, s]
    r = A.tensile_properties(e, s, modulus_range=(0.1, 0.25))
    assert r.E_GPa == pytest.approx(100.0, rel=1e-6)
    assert r.strain_intercept_pct == pytest.approx(0.05, abs=1e-6)
    assert r.modulus_range == (pytest.approx(0.1, abs=0.01), pytest.approx(0.25, abs=0.01))
    ey = 0.05 + 0.2 + 300 / 990
    assert r.yield_strain_pct == pytest.approx(ey, abs=0.005)
    assert not r.fracture_detected                            # no drop → last point
    assert r.fracture_strain_pct == pytest.approx(e[-1])


def test_superelastic_sigma_SIM_two_tangent():
    # elastic 600 MPa/% to (0.75, 450), plateau slope 10 MPa/% to 6.5 %, martensite 200 MPa/%, then hardening
    knots = [(0, 0), (0.75, 450), (6.5, 507.5), (9.0, 1007.5), (12.0, 1250)]
    e = np.linspace(0, 12, 6001)
    xk, yk = np.array(knots).T
    s = np.interp(e, xk, yk) + np.random.default_rng(3).normal(0, 0.5, e.size)
    r = A.tensile_properties(e, s)
    assert r.superelastic
    assert r.E_GPa == pytest.approx(60.0, rel=0.01)
    assert r.sigma_SIM_MPa == pytest.approx(450.0, abs=3.0)
    assert r.strain_SIM_pct == pytest.approx(0.75, abs=0.02)
    assert r.plateau_slope_MPa_per_pct == pytest.approx(10.0, abs=1.0)
    assert r.plateau_end_strain_pct == pytest.approx(6.5, abs=0.1)
    assert r.transformation_strain_pct == pytest.approx(5.75, abs=0.12)
    r_off = A.tensile_properties(e, s, superelastic=False)
    assert not r_off.superelastic and np.isnan(r_off.sigma_SIM_MPa)


def test_compression_properties_sign_flip():
    e, s = _bilinear(E_GPa=50.0, sy=600.0, H_MPa_pct=50.0, e_end=8.0)
    r = A.compression_properties(-e, -s)
    assert r.mode == "compression"
    assert r.E_GPa == pytest.approx(50.0, rel=0.01)
    assert r.max_stress_MPa == pytest.approx(600 + 50 * (8 - 1.2), abs=1)
    assert r.strain_at_max_pct == pytest.approx(8.0)
    ey = (600 - 50 * 1.2 + 500 * 0.2) / 450            # 500(ε−0.2) = 600 + 50(ε−1.2)
    assert r.yield_strain_pct == pytest.approx(ey, abs=0.01)
    assert any("flipped" in n for n in r.notes)
    assert any(q.startswith("Max compressive") for q, *_ in r.rows())


# =========================================================================== cyclic
def _parallelogram_cycle(s0, r, emax=6.0, sig_u=400.0, sig_l=200.0, n_seg=200):
    """Loading (s0,0)->(s0+1,σu)->(emax,σu); unloading (emax,σu)->(emax,σl)->(r+0.5,σl)->(r,0)."""
    pts_l = [(s0, 0.0), (s0 + 1.0, sig_u), (emax, sig_u)]
    pts_u = [(emax, sig_u), (emax, sig_l), (r + 0.5, sig_l), (r, 0.0)]

    def path(pts):
        es, ss = [], []
        for (e0, a0), (e1, a1) in zip(pts[:-1], pts[1:]):
            t = np.linspace(0, 1, n_seg, endpoint=False)
            es.append(e0 + t * (e1 - e0)); ss.append(a0 + t * (a1 - a0))
        return np.concatenate(es), np.concatenate(ss)

    el, sl = path(pts_l)
    eu, su = path(pts_u)
    e = np.r_[el, eu, r]
    s = np.r_[sl, su, 0.0]
    W_load = (0.5 * 1.0 * sig_u + (emax - s0 - 1.0) * sig_u) / 100
    W_unl = ((emax - r - 0.5) * sig_l + 0.5 * 0.5 * sig_l) / 100
    return e, s, W_load, W_unl


def test_cycle_metrics_single_parallelogram():
    e, s, Wl, Wu = _parallelogram_cycle(0.0, 0.0)
    m = A.cycle_metrics(e, s)
    assert len(m) == 1
    c = m[0]
    assert c.max_strain_pct == pytest.approx(6.0)
    assert c.max_stress_MPa == pytest.approx(400.0)
    assert c.residual_strain_pct == pytest.approx(0.0, abs=1e-9)
    assert c.recoverable_strain_pct == pytest.approx(6.0)
    assert c.loading_work_MJ_m3 == pytest.approx(Wl, rel=1e-3)
    assert c.recovered_work_MJ_m3 == pytest.approx(Wu, rel=1e-3)
    assert c.dissipated_energy_MJ_m3 == pytest.approx(Wl - Wu, rel=1e-3)
    assert c.dissipated_energy_MJ_m3 == pytest.approx(10.5, rel=1e-3)
    assert c.plateau_strain_pct == pytest.approx(3.0)
    assert c.sigma_ms_MPa == pytest.approx(400.0)
    assert c.sigma_as_MPa == pytest.approx(200.0)
    assert c.delta_sigma_MPa == pytest.approx(200.0)
    assert c.efficiency == pytest.approx(Wu / Wl)


def _drifting_test(n=8):
    E, S, C, truth = [], [], [], []
    prev = 0.0
    for N in range(1, n + 1):
        r = 0.1 * N
        e, s, Wl, Wu = _parallelogram_cycle(prev, r, sig_u=400 - 5 * N, sig_l=200 - 2 * N)
        if N > 1:
            e, s = e[1:], s[1:]
        E.append(e); S.append(s); C.append(np.full(e.size, N))
        truth.append(dict(start=prev, res=r, W=Wl - Wu, sms=400 - 5 * N, sas=200 - 2 * N))
        prev = r
    return np.concatenate(E), np.concatenate(S), np.concatenate(C), truth


def test_cycles_with_drifting_residual_strain_and_fatigue_summary():
    e, s, cyc, truth = _drifting_test()
    for kw in ({"cycle": cyc}, {}):
        cycles = A.split_cycles(e, s, **kw)
        assert len(cycles) == len(truth)
        m = A.cycle_metrics(e, s, cycles)
        for mi, tr in zip(m, truth):
            assert mi.residual_strain_pct == pytest.approx(tr["res"], abs=0.03)
            assert mi.start_strain_pct == pytest.approx(tr["start"], abs=0.03)
            assert mi.cycle_residual_pct == pytest.approx(tr["res"] - tr["start"], abs=0.04)
            assert mi.dissipated_energy_MJ_m3 == pytest.approx(tr["W"], rel=0.01)
            assert mi.sigma_ms_MPa == pytest.approx(tr["sms"], abs=0.5)
            assert mi.sigma_as_MPa == pytest.approx(tr["sas"], abs=0.5)
    df = A.fatigue_summary(m)
    assert list(df.columns) == list(A.FATIGUE_COLUMNS)
    assert df["Cycle"].tolist() == list(range(1, 9))
    assert df["Accumulated residual strain (%)"].iloc[-1] == pytest.approx(0.8, abs=0.03)
    assert np.all(np.diff(df["Stress hysteresis Δσ (MPa)"]) < 0)
    ov = dict((q, v) for q, v, _u in A.fatigue_overview(m))
    assert ov["Cycles analysed"] == 8
    assert ov["σ_ms change"] == pytest.approx(100 * (360 - 395) / 395, abs=0.3)
    ds = A.fatigue_dataset(m, "fat")
    assert ds.channels["Dissipated energy"].dimension == "energy_density"
    assert ds.channels["Upper plateau σ_ms"].unit == "MPa"
    assert ds.label("Dissipated energy") == "Dissipated energy (MJ/m³)"     # unit once (F10)
    assert ds.default_x == "Cycle" and len(ds.data) == 8


def test_split_cycles_auto_and_cycle_datasets():
    t = np.linspace(0, 10, 2001)
    strain = 3 * (1 - np.cos(2 * np.pi * t))                  # 10 cycles 0 → 6 → 0
    stress = 100 * strain
    cycles = A.split_cycles(strain, stress)
    assert len(cycles) == 10
    assert all(np.max(strain[c.indices]) == pytest.approx(6.0, abs=1e-3) for c in cycles)
    ds = Dataset.from_arrays("cyc", {"Strain": (strain, "strain", "%"), "Stress": (stress, "stress", "MPa")},
                             kind="cyclic")
    subs = A.cycle_datasets(ds, cycles, every=3)
    assert [d.metadata["cycle"] for d in subs] == [1, 4, 7, 10]
    assert subs[0].kind == "cyclic" and "cycle 1" in subs[0].name


# =========================================================================== sample files
@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("samples")
    MSD.main(d)
    return d


def test_sample_files_exist_and_parse(sample_dir):
    names = {p.name for p in sample_dir.iterdir()}
    assert names == {"NiTi_DSC_Netzsch.txt", "NiTi_tensile_failure_Instron.csv", "NiTi_compression.csv",
                     "NiTi_superelastic_cyclic_50cycles.csv", "generic_xy.txt", "NiTi_tensile_25C.csv",
                     "NiTi_tensile_40C.csv", "NiTi_tensile_55C.csv", "NiTi_elastocaloric_temperature_log.csv"}
    g = pd.read_csv(sample_dir / "generic_xy.txt", sep="\t", decimal=",")
    assert list(g.columns) == ["Time (s)", "Temperature (°C)"]
    assert g["Temperature (°C)"].dtype.kind == "f" and g["Temperature (°C)"].max() > 30


def _read_netzsch(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("##"))
    df = pd.read_csv(path, sep=";", skiprows=hdr)
    meta = {ln.split(":;")[0]: ln.split(":;")[1] for ln in lines[:hdr] if ":;" in ln}
    return df, meta


def test_sample_dsc_analysis_recovers_encoded_values(sample_dir):
    df, meta = _read_netzsch(sample_dir / "NiTi_DSC_Netzsch.txt")
    assert list(df.columns) == ["##Temp./°C", "Time/min", "DSC/(mW/mg)"]
    assert float(meta["#SAMPLE MASS /mg"]) == pytest.approx(15.2)
    T = df.iloc[:, 0].to_numpy()
    t = df.iloc[:, 1].to_numpy() * 60
    q = df.iloc[:, 2].to_numpy()
    res = A.analyse_dsc(T, q, 10.0, time=t, hf_unit="mW/mg")
    assert res.endo_up is False                              # Netzsch exo up
    true = MSD.dsc_true_temperatures()
    for k, v in true.items():
        assert res.temperatures[k] == pytest.approx(v, abs=0.5), k
    assert res.enthalpies["A"] == pytest.approx(MSD.TRUE_DSC["A"][3], rel=0.03)
    assert res.enthalpies["R"] == pytest.approx(MSD.TRUE_DSC["R"][3], rel=0.05)
    assert res.enthalpies["M"] == pytest.approx(MSD.TRUE_DSC["M"][3], rel=0.03)
    assert res.hysteresis["Ap − Mp"] == pytest.approx(45.0, abs=0.5)
    assert [s.direction for s in res.segments] == ["heating", "cooling"]
    assert res.segments[0].heating_rate_K_per_min == pytest.approx(10.0, rel=0.01)
    # also without the time column (rate from metadata)
    res2 = A.analyse_dsc(T, q, 10.0, hf_unit="W/g")
    assert res2.temperatures["Ms"] == pytest.approx(true["Ms"], abs=0.5)


def _read_instron(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Time,"))
    df = pd.read_csv(path, skiprows=hdr, header=0)
    units = df.iloc[0].tolist()
    return df.iloc[1:].astype(float).reset_index(drop=True), units


def test_sample_tensile_analysis(sample_dir):
    df, units = _read_instron(sample_dir / "NiTi_tensile_failure_Instron.csv")
    assert units == ["(s)", "(mm)", "(kN)"]
    p = MSD.TRUE_TENSILE
    area = np.pi * p["diameter_mm"] ** 2 / 4
    e, s = A.stress_strain(df["Load"] * 1000, df["Extension"], area, p["gauge_length_mm"])
    r = A.tensile_properties(e, s)
    truth = MSD.tensile_true_derived()
    assert r.E_GPa == pytest.approx(p["E_GPa"], rel=0.03)
    assert r.uts_MPa == pytest.approx(1300, abs=10)
    assert r.strain_at_uts_pct == pytest.approx(p["strain_at_UTS_pct"], abs=0.3)
    assert r.fracture_detected
    assert r.fracture_strain_pct == pytest.approx(p["fracture_strain_pct"], abs=0.05)
    assert r.yield_strength_MPa == pytest.approx(truth["yield_MPa"], abs=5)
    assert r.toughness_MJ_m3 == pytest.approx(truth["toughness_MJ_m3"], rel=0.01)
    assert r.superelastic
    assert r.sigma_SIM_MPa == pytest.approx(p["sigma_SIM_MPa"], abs=8)
    assert r.plateau_end_strain_pct == pytest.approx(p["plateau_end"][0], abs=0.2)


def test_sample_compression_analysis(sample_dir):
    path = sample_dir / "NiTi_compression.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.startswith("Time (s)"))
    df = pd.read_csv(path, skiprows=hdr)
    assert df["Load (N)"].min() < 0                          # machine sign convention
    p = MSD.TRUE_COMPRESSION
    area = np.pi * p["diameter_mm"] ** 2 / 4
    e, s = A.stress_strain(df["Load (N)"], df["Extension (mm)"], area, p["length_mm"], compression=True)
    r = A.compression_properties(e, s)
    truth = MSD.compression_true_derived()
    assert r.E_GPa == pytest.approx(p["E_GPa"], rel=0.03)
    assert r.max_stress_MPa == pytest.approx(p["max_stress_MPa"], abs=10)
    assert r.yield_strength_MPa == pytest.approx(truth["yield_MPa"], abs=8)
    # raw negative data gives the same answer
    r2 = A.compression_properties(-e, -s)
    assert r2.E_GPa == pytest.approx(r.E_GPa)


def test_sample_cyclic_functional_fatigue(sample_dir):
    path = sample_dir / "NiTi_superelastic_cyclic_50cycles.csv"
    df = pd.read_csv(path, comment="#")
    assert list(df.columns) == ["Cycle", "Time (s)", "Strain (%)", "Stress (MPa)"]
    e, s = df["Strain (%)"].to_numpy(), df["Stress (MPa)"].to_numpy()
    m = A.cycle_metrics(e, s, cycle=df["Cycle"].to_numpy())
    m_auto = A.cycle_metrics(e, s)
    assert len(m) == len(m_auto) == 50
    for N in (1, 2, 10, 25, 50):
        tr = MSD.cyclic_truth(N)
        c = m[N - 1]
        assert c.residual_strain_pct == pytest.approx(tr["residual_strain_pct"], abs=0.02)
        assert c.sigma_ms_MPa == pytest.approx(tr["sigma_ms_MPa"], abs=3)
        assert c.sigma_as_MPa == pytest.approx(tr["sigma_as_MPa"], abs=3)
        assert c.dissipated_energy_MJ_m3 == pytest.approx(MSD.cyclic_true_energy(N), rel=0.02)
        assert m_auto[N - 1].dissipated_energy_MJ_m3 == pytest.approx(c.dissipated_energy_MJ_m3, rel=0.02)
    # functional fatigue trends
    assert m[-1].delta_sigma_MPa < m[0].delta_sigma_MPa
    assert m[-1].dissipated_energy_MJ_m3 < m[0].dissipated_energy_MJ_m3
    assert m[-1].residual_strain_pct > m[0].residual_strain_pct


# =========================================================================== GUI
@dataclass
class FakeTrace:
    ds: Dataset
    x: str
    y: str
    label: str | None
    style: dict = field(default_factory=dict)
    secondary_y: bool = False

    def xy(self):
        return self.ds.values(self.x), self.ds.values(self.y)


class FakePlotPanel:
    """Records the PlotPanel API calls made by the AnalysisPanel."""

    def __init__(self):
        self.traces: list[FakeTrace] = []
        self.calls: list[tuple] = []
        self.results: list[dict] = []
        self.span = None

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))

    def add_trace(self, ds, x, y, label=None, *, secondary_y=False, **style):
        self._rec("add_trace", ds, x, y, label, **style)
        tr = FakeTrace(ds, x, y, label, style, secondary_y)
        self.traces.append(tr)
        return tr

    def add_marker(self, x, y, text="", *, color=None, marker="o"):
        assert np.isfinite(x) and np.isfinite(y)
        self._rec("add_marker", x, y, text, color=color, marker=marker)

    def add_vline(self, x, text="", **kw):
        self._rec("add_vline", x, text, **kw)

    def add_hline(self, y, text="", **kw):
        self._rec("add_hline", y, text, **kw)

    def add_line(self, xs, ys, label=None, **kw):
        assert len(xs) == len(ys)
        self._rec("add_line", xs, ys, label, **kw)

    def fill_between(self, x, y1, y2=0, **kw):
        self._rec("fill_between", x, y1, y2, **kw)

    def clear_overlays(self):
        self._rec("clear_overlays")

    def add_results(self, rows):
        for r in rows:
            assert set(r) == {"Tool", "Trace", "Quantity", "Value", "Unit"}
        self.results.extend(rows)
        self._rec("add_results", rows)

    def redraw(self):
        self._rec("redraw")

    def current_span(self):
        return self.span

    def clear(self):
        self.traces.clear()
        self._rec("clear")

    def names(self):
        return [c[0] for c in self.calls]

    def markers(self):
        return [c[1][2] for c in self.calls if c[0] == "add_marker"]


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _dsc_dataset():
    rate = 10.0
    b = rate / 60
    T_h = np.linspace(-80, 100, 1801)
    T = np.r_[T_h, T_h[::-1]]
    q = np.r_[0.01 + _split_gauss(T_h, 15, 4, 3, 0.4), -0.01 - _split_gauss(T_h[::-1], -30, 6, 4, 0.3)]
    t = np.arange(T.size) * (0.1 / b)
    return Dataset.from_arrays("DSC test", {"Temp.": (T, "temperature", "°C"), "Time": (t, "time", "s"),
                                            "DSC": (q, "specific_heat_flow", "W/g")},
                               kind="dsc", metadata={"mass_mg": 10.0, "heating_rate_K_per_min": rate})


def test_gui_dsc_tab(qapp, tmp_path):
    from labstudio.gui.analysis_panel import AnalysisPanel
    store = DatasetStore()
    fake = FakePlotPanel()
    panel = AnalysisPanel(fake, store)
    panel.resize(380, 800)
    ds = store.add(_dsc_dataset())
    tab = panel.dsc_tab
    assert tab.ds_combo.dataset() is ds
    assert tab.c_T.column() == "Temp." and tab.c_hf.column() == "DSC" and tab.c_t.column() == "Time"
    assert tab.mass.value() == pytest.approx(10.0)
    res = tab.run()
    assert res is not None and len(res.peaks) == 2
    names = fake.names()
    assert "add_trace" in names and "redraw" in names
    assert names.count("fill_between") == 2
    assert names.count("add_line") >= 4                          # baselines + tangents
    labels = " ".join(fake.markers())
    for k in ("As", "Af", "Ap", "Ms", "Mf", "Mp"):
        assert k in labels
    assert any(r["Quantity"] == "As" and r["Unit"] == "°C" for r in fake.results)
    assert tab.table.table.rowCount() > 5
    out = tmp_path / "dsc.csv"
    tab.table.export_csv(str(out))
    text = out.read_text(encoding="utf-8-sig").splitlines()
    assert text[0].startswith("# LabStudio") and any(ln.startswith("# parameters:") for ln in text)
    rows = list(csv.reader([ln for ln in text if not ln.startswith("#")]))
    assert rows[0] == ["Quantity", "Value", "Unit"] and len(rows) > 5
    panel.deleteLater()


def test_gui_mechanical_tab_force_displacement(qapp):
    from labstudio.gui.analysis_panel import AnalysisPanel
    store = DatasetStore()
    fake = FakePlotPanel()
    panel = AnalysisPanel(fake, store)
    e, s = _bilinear(noise=0.3)
    e = np.r_[e, 10.003, 10.006]
    s = np.r_[s, 5.0, 0.0]
    area, L0 = 0.7854, 50.0
    ds = Dataset.from_arrays("Tensile test", {"Extension": (e / 100 * L0, "displacement", "mm"),
                                              "Load": (s * area / 1000, "force", "kN")},
                             kind="tensile", metadata={"area_mm2": area, "gauge_length_mm": L0})
    store.add(ds)
    tab = panel.mech_tab
    panel.select_dataset(ds)
    assert panel.tabs.currentWidget() is tab
    assert tab.area.value() == pytest.approx(area) and tab.gauge.value() == pytest.approx(L0)
    assert tab.run() is None                                      # no stress/strain columns yet
    assert tab.compute_stress_strain()
    assert "Stress" in ds.channels and ds.channels["Stress"].unit == "MPa"
    assert tab.c_e.column() == "Strain" and tab.c_s.column() == "Stress"
    res = tab.run()
    assert res is not None
    assert res.E_GPa == pytest.approx(200, rel=0.02)
    assert res.fracture_detected
    assert fake.names().count("add_line") >= 2
    labels = " ".join(fake.markers())
    assert "UTS" in labels and "Fracture" in labels
    assert any(r["Quantity"].startswith("Young") and r["Unit"] == "GPa" for r in fake.results)
    # span → manual modulus range
    fake.span = (0.05, 0.15)
    tab.use_span()
    assert not tab.auto_mod.isChecked()
    res2 = tab.run()
    assert res2.modulus_range[0] >= 0.05 and res2.modulus_range[1] <= 0.15
    # missing geometry → status message, no modal dialog
    ds2 = Dataset.from_arrays("no geometry", {"d": (e, "displacement", "mm"), "F": (s, "force", "N")},
                              kind="tensile")
    store.add(ds2)
    tab.ds_combo.select(ds2)
    assert not tab.compute_stress_strain()
    assert "area" in tab.status.text().lower()
    panel.deleteLater()


def test_gui_cyclic_tab_and_fatigue_summary(qapp, tmp_path):
    from labstudio.gui.analysis_panel import AnalysisPanel
    store = DatasetStore()
    fake = FakePlotPanel()
    panel = AnalysisPanel(fake, store)
    e, s, cyc, truth = _drifting_test()
    ds = Dataset.from_arrays("Cyclic", {"Cycle": (cyc, "count", "#"), "Strain": (e, "strain", "%"),
                                        "Stress": (s, "stress", "MPa")}, kind="cyclic")
    store.add(ds)
    tab = panel.cyclic_tab
    assert tab.c_n.column() == "Cycle"
    tab.every.setValue(2)
    tab.add_cycles.setChecked(True)
    m = tab.run()
    assert m is not None and len(m) == 8
    traces = [c for c in fake.calls if c[0] == "add_trace"]
    assert len(traces) == 5                                        # cycles 1,3,5,7 + last
    colors = {c[2]["color"] for c in traces}
    assert len(colors) == 5                                        # viridis by cycle number
    assert len(store) == 1 + 5
    assert any("σ_ms" in t for t in fake.markers())
    assert tab.cycle_table.table.rowCount() == 8
    tab.cycle_table.export_csv(str(tmp_path / "cyc.csv"))
    hdr = next(csv.reader([ln for ln in (tmp_path / "cyc.csv").read_text(encoding="utf-8-sig").splitlines()
                           if not ln.startswith("#")]))
    assert hdr[0] == "Cycle" and "Dissipated energy (MJ/m³)" in hdr
    n_before = len(store)
    tab.metric.setCurrentText("Stress hysteresis Δσ (MPa)")
    fds = tab.plot_fatigue()
    assert fds is not None and len(store) == n_before + 1
    last = [c for c in fake.calls if c[0] == "add_trace"][-1]
    assert last[1][0] is fds and last[1][1] == "Cycle" and last[1][2] == "Stress hysteresis Δσ"
    tab.plot_fatigue()                                             # reuses the derived dataset
    assert len(store) == n_before + 1
    panel.deleteLater()


# =========================================================================== v1.1 — DSC fixes
def _gauss(T, Tp, s, h):
    return h * np.exp(-(T - Tp) ** 2 / (2 * s * s))


@pytest.mark.parametrize("seed", range(6))
def test_dsc_phantom_noise_peaks_rejected_QA_H1(seed):
    """QA repro defect_dsc_phantom_peak: SNR ≈ 17 must give one A peak with As ≈ 12 °C."""
    T = np.linspace(-80, 100, 3601)
    rng = np.random.default_rng(seed)
    hf = 0.1 + _gauss(T, 20, 4, 0.5) + rng.normal(0, 0.03, T.size)
    r = A.analyse_dsc(T, hf, 10.0, mode="heating", hf_unit="W/g", endo_up=True)
    assert [p.phase for p in r.peaks] == ["A"]
    assert r.temperatures["As"] == pytest.approx(12.0, abs=1.5)
    assert r.temperatures["Ap"] == pytest.approx(20.0, abs=1.0)
    assert r.temperatures["Af"] == pytest.approx(28.0, abs=1.5)
    assert r.enthalpies["A"] == pytest.approx(30.08, rel=0.05)
    assert r.params["segments"]["heating 1"]["smoothing_K"] >= 1.0


def test_dsc_adaptive_smoothing_and_rejected_peaks_reported():
    T = np.linspace(-80, 100, 3601)
    clean = A.analyse_dsc_segment(T, 0.1 + _gauss(T, 20, 4, 0.5), 10, direction="heating", endo_up=True)
    assert clean.params["smoothing_K"] == pytest.approx(1.0)
    noisy = A.analyse_dsc_segment(T, 0.1 + _gauss(T, 20, 4, 0.5) + np.random.default_rng(3).normal(0, 0.08, T.size),
                                  10, direction="heating", endo_up=True)
    assert noisy.params["smoothing_K"] > 1.0
    assert len(noisy) == 1 and noisy[0].phase == "A"
    # a small but real second peak is kept when the area threshold is lowered
    hf = 0.1 + _gauss(T, 20, 4, 0.5) + _gauss(T, -20, 2, 0.03)          # 3 % of the main area
    assert [p.phase for p in A.analyse_dsc_segment(T, hf, 10, direction="heating", endo_up=True)] == ["A"]
    pk = A.analyse_dsc_segment(T, hf, 10, direction="heating", endo_up=True, min_area_frac=0.01)
    assert [p.phase for p in pk] == ["R'", "A"]
    assert A.analyse_dsc_segment(T, hf, 10, direction="heating", endo_up=True).notes  # rejection reported


def test_dsc_estimate_noise():
    y = np.random.default_rng(0).normal(0, 0.05, 20000) + np.linspace(0, 3, 20000) ** 2
    assert A.estimate_noise(y) == pytest.approx(0.05, rel=0.05)


def test_dsc_truncated_peak_warnings_QA_M6():
    T = np.linspace(-80, 100, 3601)
    pk = A.analyse_dsc_segment(T, 0.1 + _gauss(T, 95, 4, 0.5), 10, direction="heating", endo_up=True)
    assert pk and pk[-1].truncated
    assert any("truncated" in w for w in pk.warnings)
    # peak cut at its maximum by the default edge trim (QA sci_dsc_edge case b): no peak, but a warning
    r = A.analyse_dsc(T, 0.1 + _gauss(T, 97, 4, 0.5), 10, mode="heating")
    assert any("cut off" in w for w in r.warnings)
    # curved baseline without a transformation: no peak and no warning
    hf = 0.1 + 2e-5 * (T - 10) ** 2 + np.random.default_rng(1).normal(0, 0.002, T.size)
    r = A.analyse_dsc(T, hf, 10, mode="heating")
    assert r.peaks == [] and r.warnings == []


def _two_step_run(rate=10.0, step=0.1):
    """TA-like run: heating peak 0.9 W/g σ 5 at 30 °C; cooling R (0.35, σ 3.5, 18 °C) + M (0.6, σ 5.5, −10 °C)."""
    Th = np.arange(-80, 100.0001, step)
    Tc = Th[::-1]
    hh = 0.02 + 0.0004 * Th + _gauss(Th, 30, 5, 0.9)
    hc = 0.02 + 0.0004 * Tc - _gauss(Tc, 18, 3.5, 0.35) - _gauss(Tc, -10, 5.5, 0.6)
    T = np.r_[Th, Tc]
    t = np.r_[(Th + 80) / rate, (Th[-1] + 80) / rate + 0.5 + (100 - Tc) / rate] * 60
    b = rate / 60
    truth = {"A": 0.9 * 5 * np.sqrt(2 * np.pi) / b, "R": 0.35 * 3.5 * np.sqrt(2 * np.pi) / b,
             "M": 0.6 * 5.5 * np.sqrt(2 * np.pi) / b}
    return T, np.r_[hh, hc], t, truth


def test_dsc_overlapping_two_step_cooling_non_overlapping_windows_customer_C3():
    T, q, t, truth = _two_step_run()
    r = A.analyse_dsc(T, q, None, time=t)
    cool = [s for s in r.segments if s.direction == "cooling"][0]
    R, M = cool.peaks
    assert (R.phase, M.phase) == ("R", "M")
    assert R.group == M.group                                  # one shared region / baseline
    assert R.end_T == pytest.approx(M.start_T, abs=0.2)        # split at the minimum, no overlap
    assert r.enthalpies["total cooling"] == pytest.approx(truth["R"] + truth["M"], rel=0.02)
    assert r.enthalpies["A"] == pytest.approx(truth["A"], rel=0.02)
    assert not any("differ" in w for w in r.warnings)
    rc = A.analyse_dsc(T, q, None, time=t, overlap="combined")
    assert rc.enthalpies["R+M (combined)"] == pytest.approx(truth["R"] + truth["M"], rel=0.02)
    assert rc.segments[1].combined[0].parts[0].phase == "R"
    assert rc.enthalpies["R"] == pytest.approx(r.enthalpies["R"])


def test_dsc_heating_cooling_mismatch_warning():
    T, q, t, truth = _two_step_run()
    q2 = q.copy()
    q2[T.size // 2:] = 0.02 + 0.0004 * T[T.size // 2:] - _gauss(T[T.size // 2:], -10, 5.5, 0.3)
    r = A.analyse_dsc(T, q2, None, time=t)
    assert any("differ by" in w and "ΔH" in w for w in r.warnings)


def test_dsc_heating_rate_default_and_mismatch_warnings_F1():
    T, q, t, truth = _two_step_run()
    # no time, no rate → defaulted with a warning
    r = A.analyse_dsc(T, q, None)
    assert any("defaulted to 10 K/min" in w for w in r.warnings)
    assert r.params["heating_rate_source"] == "default"
    assert r.enthalpies["A"] == pytest.approx(truth["A"], rel=0.02)    # truth rate is 10
    # header 20 K/min but the time column says 10 → warning naming both, measured rate used
    r = A.analyse_dsc(T, q, 20.0, time=t, rate_source="file header")
    w = [x for x in r.warnings if "Measured scan rate" in x]
    assert w and "10" in w[0] and "20" in w[0]
    assert r.enthalpies["A"] == pytest.approx(truth["A"], rel=0.02)
    # matching rates → no warning
    assert not any("Measured scan rate" in x for x in A.analyse_dsc(T, q, 10.5, time=t).warnings)
    # wrong rate without time → implausible ΔH warning
    r = A.analyse_dsc(T, q, 1.0)
    assert any("implausibly large" in x for x in r.warnings)


def test_dsc_manual_limits_and_segment_selection():
    T, q, t, truth = _two_step_run()
    r = A.analyse_dsc(T, q, None, time=t, limits={"cooling 1": [(40, 5), (5, -45)]}, segment="cooling 1")
    assert [s.label for s in r.segments] == ["cooling 1"]
    R, M = r.segments[0].peaks
    assert R.manual_limits and sorted([R.start_T, R.end_T]) == [pytest.approx(5, abs=0.11), pytest.approx(40, abs=0.11)]
    assert R.phase == "R" and M.phase == "M"
    assert R.enthalpy_J_per_g + M.enthalpy_J_per_g == pytest.approx(truth["R"] + truth["M"], rel=0.03)
    assert r.params["limits"] == {"cooling 1": [[40.0, 5.0], [5.0, -45.0]]}
    assert r.params["segments"]["cooling 1"]["peak_limits"]["R"][0] == pytest.approx(40, abs=0.11)
    # selections
    labels = ["heating 1", "cooling 1", "heating 2", "cooling 2"]
    assert A.select_segments(labels, "last heating") == [2]
    assert A.select_segments(labels, "cooling 1") == [1]
    assert A.select_segments(labels, "all") == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        A.select_segments(labels, "sideways")
    with pytest.raises(ValueError):
        A.analyse_dsc(T, q, None, time=t, segment="heating 3")


def test_dsc_segment_datasets_N1():
    T, q, t, truth = _two_step_run()
    T2, q2, t2 = np.r_[T, T], np.r_[q, q], np.r_[t, t + t[-1] + 30]
    ds = Dataset.from_arrays("Sample", {"T": (T2, "temperature", "°C"), "t": (t2, "time", "s"),
                                        "q": (q2, "specific_heat_flow", "W/g")}, kind="dsc")
    subs = A.dsc_segment_datasets(ds, "T", "t")
    assert [s.name for s in subs] == ["Sample (heating 1)", "Sample (cooling 1)", "Sample (heating 2)",
                                      "Sample (cooling 2)"]
    assert subs[2].metadata["heating_rate_K_per_min"] == pytest.approx(10, rel=0.01)
    assert subs[2].kind == "dsc" and np.all(np.diff(subs[2].values("T")) > 0)


def test_dsc_tiny_input_and_dTad_label_QA_L6():
    with pytest.raises(ValueError, match="Not enough"):
        A.analyse_dsc(np.array([1.0, 2, 3]), np.array([0.1, 0.2, np.nan]), 10)
    T, q, t, _ = _two_step_run()
    rows = A.analyse_dsc(T, q, None, time=t).rows()
    assert any(r[0].startswith("ΔT_ad upper bound (L/c_p)") for r in rows)


# =========================================================================== v1.1 — mechanical
def test_mechanical_sanity_warnings_QA_M9():
    e, s = _bilinear(E_GPa=70.0, sy=400.0)
    assert A.tensile_properties(e, s).warnings == []
    r = A.tensile_properties(e / 100, s, superelastic=False)          # strain mm/mm labelled % → 7000 GPa
    assert r.E_GPa == pytest.approx(7000, rel=0.02)
    assert any("outside the plausible range" in w for w in r.warnings)
    r = A.tensile_properties(e * 20, s * 20)
    assert any("exceeds 5 GPa" in w for w in r.warnings) and any("exceeds 100 %" in w for w in r.warnings)
    assert r.params["modulus_auto"] and len(r.params["modulus_range_used"]) == 2


def test_machine_compliance_correction_and_estimate_N5():
    area, L0, E_true, C = 0.7854, 50.0, 60.0, 2e-4      # mm², mm, GPa, mm/N
    e, s = _bilinear(E_GPa=E_true, sy=450.0, H_MPa_pct=10.0, e_end=4.0)
    F = s * area
    d = e / 100 * L0 + C * F                            # crosshead includes frame deflection
    e_app, s_app = A.stress_strain(F, d, area, L0)
    assert A.tensile_properties(e_app, s_app).E_GPa < 0.9 * E_true
    est = A.estimate_compliance(F, d, area, L0, E_true)
    assert est["C_mm_per_N"] == pytest.approx(C, rel=0.01)
    e_c, s_c = A.stress_strain(F, d, area, L0, compliance_mm_per_N=est["C_mm_per_N"])
    assert A.tensile_properties(e_c, s_c).E_GPa == pytest.approx(E_true, rel=0.01)
    np.testing.assert_allclose(A.compliance_correct([1.0, 2.0], [0.0, 1000.0], 1e-3), [1.0, 1.0])


# =========================================================================== v1.1 — cyclic
def _qa_loop():
    def seg(a, b, n=400):
        return np.linspace(a, b, n)
    e = np.r_[seg(0, .9), seg(.9, 6)[1:], seg(6, 5.5)[1:], seg(5.5, .4)[1:], seg(.4, 0)[1:]]
    s = np.r_[seg(0, 450), np.full(399, 450.), seg(450, 200)[1:], np.full(399, 200.), seg(200, 0)[1:]]
    return e, s


def test_cyclic_compression_signed_QA_H2():
    e, s = _qa_loop()
    for ee, ss in ((e, s), (-e, -s)):
        m = A.cycle_metrics(ee, ss)[0]
        assert m.dissipated_energy_MJ_m3 == pytest.approx(12.75, rel=1e-3)
        assert m.sigma_ms_MPa == pytest.approx(450) and m.sigma_as_MPa == pytest.approx(200)
    r = A.analyse_cyclic(-e, -s)
    assert r.strain_sign == -1 and r.stress_sign == -1
    assert any("sign inverted" in w for w in r.warnings)
    assert r.metrics[0].dissipated_energy_MJ_m3 == pytest.approx(12.75, rel=1e-3)
    assert A.analyse_cyclic(e, s).warnings == []
    # forced "no" keeps the wrong signs; "yes" inverts
    assert A.normalise_signs(-e, -s, "no")[2:] == (False, False)
    ee, ss, fe, fs = A.normalise_signs(e, s, "yes")
    assert fe and fs and ee.max() == 0


def test_cyclic_incomplete_cycles_flagged_QA_M7():
    e, s = _qa_loop()
    m = A.cycle_metrics(e[:799], s[:799])                       # loading only
    assert len(m) == 1 and not m[0].complete
    assert np.isnan(m[0].dissipated_energy_MJ_m3) and np.isnan(m[0].sigma_as_MPa)
    assert m[0].loading_work_MJ_m3 == pytest.approx(24.975, rel=1e-3)
    r = A.analyse_cyclic(e[:799], s[:799])
    assert any("incomplete" in w for w in r.warnings)
    # two full cycles + a cut-off third: excluded from the overview statistics
    E = np.r_[e, e[1:], e[1:799]]
    S = np.r_[s, s[1:], s[1:799]]
    r = A.analyse_cyclic(E, S)
    assert [x.complete for x in r.metrics] == [True, True, False]
    ov = dict((q, v) for q, v, _ in A.fatigue_overview(r.metrics))
    assert ov["Incomplete cycles (excluded)"] == 1
    assert ov["Dissipated energy last cycle"] == pytest.approx(12.75, rel=1e-3)
    assert ov["Cycles analysed"] == 3


def test_fit_saturating_recovers_tau_and_stabilisation_N6():
    N = np.arange(1, 51)
    y = 0.15 + 1.05 * (1 - np.exp(-(N - 1) / 8.0))
    f = A.fit_saturating(N, y, threshold=0.01)
    assert f.tau == pytest.approx(8.0, rel=1e-4)
    assert f.saturation == pytest.approx(1.2, rel=1e-5)
    # |dy/dN| = 1.05/8·exp(−(N−1)/8) < 0.01  ⇔  N − 1 > 8·ln(13.125)
    assert f.stabilisation_cycle == float(int(np.ceil(1 + 8 * np.log(1.05 / 8 / 0.01))))
    assert f(np.array([1.0]))[0] == pytest.approx(0.15)
    with pytest.raises(ValueError):
        A.fit_saturating([1, 2], [1, 2])


def test_sample_cyclic_fatigue_fits(sample_dir):
    df = pd.read_csv(sample_dir / "NiTi_superelastic_cyclic_50cycles.csv", comment="#")
    m = A.cycle_metrics(df["Strain (%)"].to_numpy(), df["Stress (MPa)"].to_numpy(), cycle=df["Cycle"].to_numpy())
    s = A.fatigue_summary(m)
    p = MSD.TRUE_CYCLIC
    f = A.fit_saturating(s["Cycle"], s["Residual strain (%)"])
    assert f.tau == pytest.approx(p["res_tau"], rel=0.05)
    assert f.saturation == pytest.approx(p["res_first_pct"] + p["res_extra_pct"], abs=0.02)
    f = A.fit_saturating(s["Cycle"], s["Stress hysteresis Δσ (MPa)"])
    assert f.tau == pytest.approx(p["plat_tau"], rel=0.15)
    assert f.saturation == pytest.approx(p["sms0"] - p["sas0"] - p["sms_drop"] + p["sas_drop"], abs=3)


def test_bare_column_name():
    assert A.bare_column_name("Dissipated energy (MJ/m³)") == "Dissipated energy"
    assert A.bare_column_name("Efficiency (-)") == "Efficiency"
    assert A.bare_column_name("Cycle") == "Cycle"


# =========================================================================== v1.1 — GUI
class FakePlotPanel2(FakePlotPanel):
    """FakePlotPanel with the optional add_label / add_colorbar extras of the v1.1 plot panel."""

    def add_label(self, x, y, text, **kw):
        assert np.isfinite(x) and np.isfinite(y)
        self._rec("add_label", x, y, text, **kw)

    def add_colorbar(self, cmap, vmin, vmax, label):
        self._rec("add_colorbar", cmap, vmin, vmax, label)

    def labels(self):
        return [c[1][2] for c in self.calls if c[0] == "add_label"]


def _panel(fake=None):
    from labstudio.gui.analysis_panel import AnalysisPanel
    store = DatasetStore()
    fake = fake or FakePlotPanel2()
    return AnalysisPanel(fake, store), store, fake


def test_gui_dock_fits_420px(qapp):
    panel, store, fake = _panel()
    store.add(_dsc_dataset())
    panel.resize(400, 680)
    assert panel.minimumSizeHint().width() <= 420
    assert panel.minimumSizeHint().height() <= 680
    for t in panel.all_tabs:
        assert t.minimumSizeHint().width() <= 420
    panel.deleteLater()


def test_gui_dsc_labels_warnings_rate_highlight_and_provenance(qapp, tmp_path):
    panel, store, fake = _panel()
    T, q, t, truth = _two_step_run()
    ds_nt = store.add(Dataset.from_arrays("no time", {"T": (T, "temperature", "°C"),
                                                       "q": (q, "specific_heat_flow", "W/g")}, kind="dsc"))
    tab = panel.dsc_tab
    tab.ds_combo.select(ds_nt)
    assert tab.rate_is_defaulted() and "defaulted" in tab.rate_src.text()
    assert "background" in tab.rate.styleSheet()                 # amber highlight
    res = tab.run()
    assert res is not None and any("defaulted" in w for w in res.warnings)
    assert tab.table.warn_rows() and "defaulted" in tab.table.rows()[0][1]
    assert "⚠" in tab.status.text()
    assert {"As", "Af", "Ms"} <= {lab.split()[0] for lab in fake.labels()}  # collision-free labels
    assert any(r["Quantity"] == "Warning" for r in fake.results)
    tab.rate.setValue(11.0)
    tab.rate.setValue(10.0)                                      # user confirms
    assert not tab.rate_is_defaulted() and tab.rate.styleSheet() == ""
    res = tab.run()
    assert not any("defaulted" in w for w in res.warnings)
    assert tab.last_provenance["parameters"]["heating_rate_source"] == "user"
    out = tmp_path / "p.csv"
    tab.table.export_csv(str(out))
    txt = out.read_text(encoding="utf-8-sig")
    assert txt.startswith("# LabStudio") and "min_area_frac" in txt
    # rate source from import metadata shown in results
    ds2 = store.add(Dataset.from_arrays("hdr", {"T": (T, "temperature", "°C"), "q": (q, "specific_heat_flow", "W/g")},
                                        kind="dsc", metadata={"heating_rate_K_per_min": 10.0,
                                                              "heating_rate_source": "file header"}))
    tab.ds_combo.select(ds2)
    assert "file header" in tab.rate_src.text()
    tab.run()
    assert tab.table.rows()[0] == ["Heating rate source", "file header", ""]
    # ΔT_ad tooltip
    i = next(i for i, r in enumerate(tab.table.rows()) if str(r[0]).startswith("ΔT_ad"))
    assert "UPPER BOUND" in tab.table.table.item(i, 0).toolTip()
    panel.deleteLater()


def test_gui_dsc_manual_limits_segments_and_split(qapp):
    panel, store, fake = _panel()
    T, q, t, truth = _two_step_run()
    ds = store.add(Dataset.from_arrays("Sample", {"T": (np.r_[T, T], "temperature", "°C"),
                                                   "t": (np.r_[t, t + t[-1] + 30], "time", "s"),
                                                   "q": (np.r_[q, q], "specific_heat_flow", "W/g")}, kind="dsc"))
    tab = panel.dsc_tab
    tab.ds_combo.select(ds)
    items = [tab.segment.itemText(i) for i in range(tab.segment.count())]
    assert items == ["all", "heating 1", "cooling 1", "heating 2", "cooling 2", "last heating", "last cooling"]
    tab.segment.setCurrentText("last cooling")
    res = tab.run()
    assert [s.label for s in res.segments] == ["cooling 2"]
    plotted = [c[1][0] for c in fake.calls if c[0] == "add_trace"][-1]
    assert plotted.name == "Sample (cooling 2)" and plotted in list(store)
    assert tab.peak_table.rowCount() == 2
    # edit the R window (process order: start > end on cooling) and re-run with manual limits
    tab.set_peak_limits(0, 35.0, 4.0)
    tab.set_peak_limits(1, 4.0, -40.0)
    res = tab.run_manual()
    R = res.segments[0].peaks[0]
    assert R.manual_limits and R.start_T == pytest.approx(35, abs=0.11)
    assert tab.last_provenance["parameters"]["limits"]["cooling 2"][0] == [35.0, 4.0]
    fake.span = (-45.0, 0.0)
    tab.peak_table.selectRow(1)
    tab.span_to_selected()
    assert tab.peak_table.cellWidget(1, 2).value() == pytest.approx(0.0)    # kept process order
    res = tab.run_auto()
    assert not res.segments[0].peaks[0].manual_limits
    n = len(store)
    added = tab.split_segments()
    assert len(store) == n + len(added) and len(added) == 3                # cooling 2 already existed
    panel.deleteLater()


def _ten_cycles(sign=1.0):
    e, s = _qa_loop()
    E = np.concatenate([e] + [e[1:]] * 9)
    S = np.concatenate([s] + [s[1:]] * 9)
    return sign * E, sign * S


def test_gui_cyclic_colorbar_compression_and_never_silent(qapp):
    panel, store, fake = _panel()
    e, s = _ten_cycles(-1.0)
    ds = store.add(Dataset.from_arrays("comp", {"Strain": (e, "strain", "%"), "Stress": (s, "stress", "MPa")},
                                       kind="cyclic"))
    tab = panel.cyclic_tab
    tab.ds_combo.select(ds)
    m = tab.run()
    assert m is not None and len(m) == 10
    assert m[0].dissipated_energy_MJ_m3 == pytest.approx(12.75, rel=1e-3)
    assert "sign inverted" in tab.status.text()
    cb = [c for c in fake.calls if c[0] == "add_colorbar"]
    assert cb and cb[0][1] == ("viridis", 1, 10, "Cycle number")
    assert all(c[1][3].startswith("_") for c in fake.calls if c[0] == "add_trace")     # no 10-entry legend
    # overlay markers are drawn at the recorded (negative) positions
    sms = [c for c in fake.calls if c[0] == "add_label" and c[1][2].startswith("σ_ms")]
    assert sms and sms[0][1][1] < 0
    # trend fits table
    assert tab.fit_table.table.rowCount() >= 1
    # fatigue plot: bare column, unit appears once in the label, fit line drawn
    tab.metric.setCurrentText("Dissipated energy (MJ/m³)")
    fds = tab.plot_fatigue()
    assert fds.label("Dissipated energy") == "Dissipated energy (MJ/m³)"
    # failure inside the analysis is shown, never silent
    bad = store.add(Dataset.from_arrays("bad", {"Strain": (np.array([1.0, np.nan]), "strain", "%"),
                                                "Stress": (np.array([1.0, 2.0]), "stress", "MPa")}, kind="cyclic"))
    tab.ds_combo.select(bad)
    assert tab.run() is None and "failed" in tab.status.text()
    panel.deleteLater()


def test_gui_cyclic_force_displacement_F8(qapp):
    panel, store, fake = _panel()
    e, s = _ten_cycles()
    area, L0 = np.pi / 4, 50.0
    ds = store.add(Dataset.from_arrays("kN_mm", {"Time": (np.arange(e.size) * 0.1, "time", "s"),
                                                 "Extension": (e / 100 * L0, "displacement", "mm"),
                                                 "Load": (s * area / 1000, "force", "kN")}, kind="cyclic",
                                       metadata={"diameter_mm": 1.0, "gauge_length_mm": L0}))
    tab = panel.cyclic_tab
    tab.ds_combo.select(ds)
    assert tab.c_e.column() is None and tab.c_s.column() is None       # never defaults to Time
    assert tab.gss.isVisibleTo(tab)
    assert tab.run() is None and "Compute stress" in tab.status.text()
    assert tab.area.value() == pytest.approx(area) and tab.gauge.value() == pytest.approx(L0)
    assert tab.compute_stress_strain()
    assert tab.c_e.column() == "Strain" and tab.c_s.column() == "Stress"
    m = tab.run()
    assert m is not None and len(m) == 10
    assert m[0].dissipated_energy_MJ_m3 == pytest.approx(12.75, rel=2e-3)
    panel.deleteLater()


def test_gui_mechanical_compliance_and_warnings(qapp):
    panel, store, fake = _panel()
    area, L0, E_true, C = 0.7854, 50.0, 60.0, 2e-4
    e, s = _bilinear(E_GPa=E_true, sy=450.0, H_MPa_pct=10.0, e_end=4.0)
    F = s * area
    ds = store.add(Dataset.from_arrays("frame", {"d": (e / 100 * L0 + C * F, "displacement", "mm"),
                                                 "F": (F, "force", "N")}, kind="tensile",
                                       metadata={"area_mm2": area, "gauge_length_mm": L0}))
    tab = panel.mech_tab
    tab.ds_combo.select(ds)
    tab.E_true.setValue(E_true)
    assert tab.estimate_compliance() == pytest.approx(C, rel=0.01)
    assert tab.compute_stress_strain()
    r = tab.run()
    assert r.E_GPa == pytest.approx(E_true, rel=0.01)
    assert tab.last_provenance["parameters"]["machine_compliance_mm_per_N"] == pytest.approx(C, rel=0.01)
    ds2 = store.add(Dataset.from_arrays("mmmm", {"Strain": (e / 100, "strain", "%"), "Stress": (s, "stress", "MPa")},
                                        kind="tensile"))
    tab.ds_combo.select(ds2)
    tab.run()
    assert "plausible range" in tab.status.text() and tab.table.warn_rows()
    panel.deleteLater()


def test_gui_number_formatting_uses_sig_figs(qapp, monkeypatch):
    from labstudio.gui import analysis_widgets as W
    monkeypatch.setattr(W, "format_value", lambda v: f"{v:.3g}")
    t = W.SummaryTable()
    t.set_rows([["x", 20.069457240309497, "°C"], ["n", 7, "#"]])
    assert t.table.item(0, 1).text() == "20.1" and t.table.item(1, 1).text() == "7"
    assert "20.1" in t.text()


def test_gui_state_roundtrip_and_defaults_Q4_Q6(qapp, tmp_path):
    from labstudio.core import session
    panel, store, fake = _panel()
    ds = store.add(_dsc_dataset())
    cyc_e, cyc_s = _ten_cycles()
    cds = store.add(Dataset.from_arrays("cyc", {"Strain": (cyc_e, "strain", "%"), "Stress": (cyc_s, "stress", "MPa")},
                                        kind="cyclic"))
    panel.dsc_tab.ds_combo.select(ds)
    panel.dsc_tab.min_area.setValue(3.0)
    panel.dsc_tab.overlap.setCurrentIndex(1)
    panel.dsc_tab.run()
    panel.cyclic_tab.ds_combo.select(cds)
    panel.cyclic_tab.pfrac.setValue(40)
    panel.cyclic_tab.run()
    panel.tabs.setCurrentWidget(panel.cyclic_tab)
    defaults = panel.current_defaults()
    assert set(defaults) == {"dsc", "mechanical", "cyclic"}
    assert defaults["dsc"]["min_area"] == 3.0 and defaults["cyclic"]["pfrac"] == 40
    import json
    json.dumps(panel.get_state())
    path = tmp_path / "s.lstudio"
    session.save_session(str(path), store, {"analysis": panel})
    panel2, store2, fake2 = _panel()
    session.load_session(str(path), store2, {"analysis": panel2})
    assert panel2.tabs.currentWidget() is panel2.cyclic_tab
    assert panel2.dsc_tab.ds_combo.dataset().name == ds.name
    assert panel2.dsc_tab.min_area.value() == 3.0 and panel2.dsc_tab.overlap.currentIndex() == 1
    assert panel2.dsc_tab.last_result is not None                         # re-run on restore
    assert panel2.cyclic_tab.last_result is not None and panel2.cyclic_tab.pfrac.value() == 40
    # analysis_defaults carried by a dataset (import preset) are applied when it is selected
    panel2.apply_defaults({"dsc": {"min_area": 5.0}})
    assert panel2.dsc_tab.min_area.value() == 5.0
    ds3 = _dsc_dataset()
    ds3.metadata["analysis_defaults"] = {"dsc": {"min_area": 12.0, "trim": 4.0}}
    store2.add(ds3)
    panel2.dsc_tab.ds_combo.select(ds3)
    assert panel2.dsc_tab.min_area.value() == 12.0 and panel2.dsc_tab.trim.value() == 4.0
    panel.deleteLater()
    panel2.deleteLater()
