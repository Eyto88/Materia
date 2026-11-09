"""Tests for core/elastocaloric.py and the Elastocaloric tab (known-answer synthetic data + sample files)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_sample_data as MSD  # noqa: E402

from labstudio.core import data_io  # noqa: E402
from labstudio.core import elastocaloric as EC  # noqa: E402
from labstudio.core.dataset import Dataset, DatasetStore  # noqa: E402
from tests.test_analysis import FakePlotPanel2, _qa_loop  # noqa: E402


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("samples_ec")
    MSD.main(d)
    return d


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------------------- Clausius–Clapeyron
def test_entropy_change_formula():
    # ΔS = −(dσ/dT)·ε/ρ = −6.5e6 Pa/K × 0.05 / 6450 kg/m³
    assert EC.entropy_change(6.5, 5.0, 6.45) == pytest.approx(-6.5e6 * 0.05 / 6450)


def test_clausius_clapeyron_recovers_encoded_slope():
    T = np.array([20.0, 30.0, 40.0, 50.0])
    sig = 300 + 6.2 * (T - 20) + np.array([0.3, -0.2, 0.1, -0.2])
    cc = EC.clausius_clapeyron(T, sig, 4.0, 6.45, dsc_dH_J_per_g=15.0)
    assert cc.slope_MPa_per_K == pytest.approx(6.2, abs=0.05)
    assert cc.r2 > 0.999 and np.isfinite(cc.slope_stderr)
    assert cc.T_ref_C == pytest.approx(35.0)
    dS = -cc.slope_MPa_per_K * 1e6 * 0.04 / 6450
    assert cc.dS_J_per_kgK == pytest.approx(dS)
    assert cc.latent_J_per_g == pytest.approx(308.15 * abs(dS) / 1000)
    assert cc.dsc_diff_pct == pytest.approx(100 * (cc.latent_J_per_g - 15) / 15)
    assert cc.warnings == []
    rows = dict((q, v) for q, v, _ in cc.rows())
    assert rows["Entropy change ΔS"] == pytest.approx(dS)
    x, y = cc.line()
    assert x[0] == 20 and x[-1] == 50
    # two points / negative slope / missing ε → warnings
    cc2 = EC.clausius_clapeyron([20, 40], [400, 300], float("nan"))
    assert any("two temperatures" in w for w in cc2.warnings)
    assert any("not positive" in w for w in cc2.warnings)
    assert any("unknown" in w for w in cc2.warnings) and np.isnan(cc2.latent_J_per_g)
    with pytest.raises(ValueError):
        EC.clausius_clapeyron([20, 20], [1, 2], 4)


def test_temperature_from_metadata_and_name():
    ds = Dataset.from_arrays("NiTi_tensile_40C", {"x": (np.arange(3.0), "strain", "%")})
    assert EC.temperature_from(ds) == 40.0
    ds.metadata["test_temperature_C"] = 37.5
    assert EC.temperature_from(ds) == 37.5
    ds2 = Dataset.from_arrays("sample A", {"x": (np.arange(3.0), "strain", "%")},
                              metadata={"source_meta": {"Test temperature": "313.15 K"}})
    assert EC.temperature_from(ds2) == pytest.approx(40.0)
    ds3 = Dataset.from_arrays("Cycle 3 test", {"x": (np.arange(3.0), "strain", "%")})
    assert EC.temperature_from(ds3) is None


def test_sample_files_clausius_clapeyron(sample_dir):
    dss = [data_io.load_file(str(sample_dir / f"NiTi_tensile_{T}C.csv")) for T in (25, 40, 55)]
    pts = [EC.cc_point_from_dataset(d) for d in dss]
    for p, T in zip(pts, MSD.TRUE_CC["temperatures_C"]):
        assert p.T_C == T and p.source == "σ_SIM"
        assert p.sigma_MPa == pytest.approx(MSD.cc_sigma_SIM(T), abs=2.0)
        assert p.eps_tr_pct == pytest.approx(MSD.TRUE_CC["eps_tr_pct"], abs=0.05)
    cc = EC.clausius_clapeyron([p.T_C for p in pts], [p.sigma_MPa for p in pts], [p.eps_tr_pct for p in pts])
    assert cc.slope_MPa_per_K == pytest.approx(MSD.TRUE_CC["dsigma_dT_MPa_per_K"], abs=0.05)
    assert cc.dS_J_per_kgK == pytest.approx(MSD.cc_true_entropy(), rel=0.01)
    assert cc.latent_J_per_g == pytest.approx(MSD.cc_true_latent(40.0), rel=0.01)


def test_cc_point_from_cyclic_dataset():
    e, s = _qa_loop()
    ds = Dataset.from_arrays("cyc 30C", {"Strain": (e, "strain", "%"), "Stress": (s, "stress", "MPa")},
                             kind="cyclic")
    p = EC.cc_point_from_dataset(ds, source="σ_ms")
    assert p.T_C == 30 and p.source == "σ_ms" and p.sigma_MPa == pytest.approx(450)


# --------------------------------------------------------------------------- measured ΔT / COP
def _caloric_synthetic(n_cycles=4, dTh=10.0, dTc=8.0, offset=0.0):
    """Cycles of the QA parallelogram loop (W = 12.75 MJ/m³), 1 point per 0.1 s, with a square ±ΔT pulse in the
    middle of loading (+dTh) and unloading (−dTc) on a 22 °C baseline. Log on its own clock (t_log = t − offset)."""
    e, s = _qa_loop()
    E = np.concatenate([e] + [e[1:]] * (n_cycles - 1))
    S = np.concatenate([s] + [s[1:]] * (n_cycles - 1))
    t = np.arange(E.size) * 0.1
    n = e.size - 1                                            # points per cycle
    tl = np.arange(0, t[-1] + 0.05, 0.05)
    Tl = np.full(tl.size, 22.0)
    for k in range(n_cycles):
        t0 = k * n * 0.1
        ipk = int(np.argmax(e))
        tp = t0 + ipk * 0.1
        t1 = t0 + n * 0.1
        Tl[(tl > t0 + 0.3 * (tp - t0)) & (tl < t0 + 0.7 * (tp - t0))] += dTh
        Tl[(tl > tp + 0.3 * (t1 - tp)) & (tl < tp + 0.7 * (t1 - tp))] -= dTc
    return tl - offset, Tl, t, E, S, n * 0.1


def test_caloric_known_answer():
    tl, Tl, t, E, S, dur = _caloric_synthetic()
    r = EC.caloric_per_cycle(tl, Tl, t, E, S, rho_g_cm3=6.45, cp_J_per_gK=0.45)
    assert len(r.cycles) == 4 and r.warnings == []
    W_kg = 12.75e6 / 6450
    for c in r.cycles:
        assert c.T_base_C == pytest.approx(22.0)
        assert c.dT_heat_K == pytest.approx(10.0)
        assert c.dT_cool_K == pytest.approx(-8.0)
        assert c.work_MJ_m3 == pytest.approx(12.75, rel=1e-3)
        assert c.work_J_per_kg == pytest.approx(W_kg, rel=1e-3)
        assert c.Qc_J_per_kg == pytest.approx(0.45 * 1000 * 8)
        assert c.COP == pytest.approx(3600 / W_kg, rel=1e-3)
        assert c.frequency_Hz == pytest.approx(1 / dur, rel=0.01)
        assert c.cooling_power_W_per_kg == pytest.approx(3600 / dur, rel=0.01)
    rows = dict((q, v) for q, v, _ in r.rows())
    assert rows["COP_mat mean"] == pytest.approx(3600 / W_kg, rel=1e-3)
    ds = EC.caloric_dataset(r)
    assert ds.values("COP_mat")[0] == pytest.approx(3600 / W_kg, rel=1e-3)
    assert ds.values("Cooling power", "W/kg")[0] == pytest.approx(3600 / dur, rel=0.01)


def test_caloric_time_offset_and_coverage_warning():
    tl, Tl, t, E, S, _ = _caloric_synthetic(offset=25.0)          # logger started 25 s before the test
    bad = EC.caloric_per_cycle(tl, Tl, t, E, S)
    assert any("does not cover" in w or "offset" in w for w in bad.warnings) or \
        abs(bad.cycles[0].dT_heat_K - 10) > 1
    good = EC.caloric_per_cycle(tl, Tl, t, E, S, offset_s=25.0)
    assert good.cycles[1].dT_heat_K == pytest.approx(10.0)
    # compression-signed mechanical data gives the same work
    r = EC.caloric_per_cycle(tl, Tl, t, -E, -S, offset_s=25.0)
    assert r.cycles[0].work_MJ_m3 == pytest.approx(12.75, rel=1e-3)


def test_sample_temperature_log_matches_encoded_truth(sample_dir):
    cyc = data_io.load_file(str(sample_dir / "NiTi_superelastic_cyclic_50cycles.csv"))
    log = data_io.load_file(str(sample_dir / "NiTi_elastocaloric_temperature_log.csv"))
    r = EC.caloric_per_cycle(log.values("Time"), log.values("Temperature"), cyc.values("Time"), cyc.values("Strain"),
                             cyc.values("Stress"), cycle=cyc.values("Cycle"))
    assert len(r.cycles) == 50
    for N in (1, 10, 50):
        c, tr = r.cycles[N - 1], MSD.caloric_truth(N)
        assert c.dT_heat_K == pytest.approx(tr["dT_heat_K"], abs=0.06)
        assert c.dT_cool_K == pytest.approx(tr["dT_cool_K"], abs=0.06)
        assert c.COP == pytest.approx(tr["COP"], rel=0.03)
        assert c.cooling_power_W_per_kg == pytest.approx(tr["cooling_power_W_per_kg"], rel=0.03)


# --------------------------------------------------------------------------- GUI
def test_gui_elastocaloric_tab(qapp, sample_dir, tmp_path):
    from labstudio.gui.analysis_panel import AnalysisPanel
    store = DatasetStore()
    fake = FakePlotPanel2()
    panel = AnalysisPanel(fake, store)
    tab = panel.ec_tab
    dss = [store.add(data_io.load_file(str(sample_dir / f"NiTi_tensile_{T}C.csv"))) for T in (25, 40, 55)]
    T = np.linspace(-80, 100, 1801)
    dsc = store.add(Dataset.from_arrays("DSC", {"T": (T, "temperature", "°C"), "q": (
        0.01 + 20 / 6 / (4 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((T - 15) / 4) ** 2), "specific_heat_flow", "W/g")},
        kind="dsc", metadata={"heating_rate_K_per_min": 10.0}))
    tab.set_cc_checked([d.id for d in dss])
    pts = tab.collect_points()
    assert len(pts) == 3 and tab.cc_table.rowCount() == 3
    tab.dsc_combo.select(dsc)
    cc = tab.run_cc()
    assert cc is not None and cc.slope_MPa_per_K == pytest.approx(6.5, abs=0.05)
    assert cc.dsc_dH_J_per_g == pytest.approx(20, rel=0.03)
    assert np.isfinite(cc.dsc_diff_pct)
    assert any(c[0] == "add_line" for c in fake.calls)
    assert tab.cc_ds in list(store)
    # hand-edited point is used
    tab.cc_table.item(2, 2).setText("700")
    cc2 = tab.run_cc()
    assert cc2.slope_MPa_per_K > 6.5
    # ΔT / COP
    cyc = store.add(data_io.load_file(str(sample_dir / "NiTi_superelastic_cyclic_50cycles.csv")))
    log = store.add(data_io.load_file(str(sample_dir / "NiTi_elastocaloric_temperature_log.csv")))
    tab.refresh()
    tab.log_combo.select(log)
    tab.mech_combo.select(cyc)
    res = tab.run_caloric()
    assert res is not None and len(res.cycles) == 50
    assert tab.cycle_table.table.rowCount() == 50
    n = len(store)
    d1 = tab.plot_caloric("dT")
    d2 = tab.plot_caloric("COP")
    assert d1 is d2 and len(store) == n + 1
    assert [c for c in fake.calls if c[0] == "add_trace"][-1][1][2] == "COP_mat"
    out = tmp_path / "cop.csv"
    tab.cycle_table.export_csv(str(out))
    assert out.read_text(encoding="utf-8-sig").startswith("# LabStudio")
    st = panel.get_state()
    assert st["elastocaloric"]["ran_caloric"] and len(st["elastocaloric"]["cc_points"]) == 3
    # missing selection → message, no exception
    tab.mech_combo.setCurrentIndex(-1)
    assert tab.run_caloric() is None and tab.status.text()
    panel.deleteLater()
