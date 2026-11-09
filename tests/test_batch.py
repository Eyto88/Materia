"""Tests for core/batch.py and the Batch tab."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from labstudio.core import batch as B  # noqa: E402
from labstudio.core.dataset import Dataset, DatasetStore  # noqa: E402
from tests.test_analysis import FakePlotPanel2, _bilinear, _qa_loop, _two_step_run  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_parse_parameter_and_presets():
    rx = B.PARAMETER_PRESETS["Temperature (°C)"]
    assert B.parse_parameter("NiTi annealed 450C 30min", rx) == 450
    assert B.parse_parameter("NiTi_500 °C", rx) == 500
    assert B.parse_parameter("amp 4,5 %", B.PARAMETER_PRESETS["Strain amplitude (%)"]) == 4.5
    assert B.parse_parameter("no number", rx) is None
    with pytest.raises(ValueError):
        B.parse_parameter("x", "(")
    ds = Dataset.from_arrays("S1", {"x": (np.arange(3.0), "strain", "%")}, metadata={"thickness_mm": 0.2,
                                                                                     "note": "t = 3 mm"})
    assert B.dataset_parameter(ds, meta_key="thickness_mm") == 0.2
    assert B.dataset_parameter(ds, meta_key="note") == 3
    assert B.dataset_parameter(ds, meta_key="missing") is None


def _tensile_ds(name, E, sy, force_disp=False):
    e, s = _bilinear(E_GPa=E, sy=sy, H_MPa_pct=20, e_end=6.0)
    if force_disp:
        return Dataset.from_arrays(name, {"d": (e / 100 * 50, "displacement", "mm"), "F": (s * 0.5, "force", "N")},
                                   kind="tensile", metadata={"area_mm2": 0.5, "gauge_length_mm": 50.0})
    return Dataset.from_arrays(name, {"e": (e, "strain", "%"), "s": (s, "stress", "MPa")}, kind="tensile")


def test_batch_tensile_groups_and_metric_dataset():
    dss = [_tensile_ds("T 25C a", 60, 400), _tensile_ds("T 25C b", 62, 410, force_disp=True),
           _tensile_ds("T 40C a", 64, 500), _tensile_ds("T 40C b", 66, 520)]
    res = B.batch_analyse(dss, "tensile", {}, parameter_regex=B.PARAMETER_PRESETS["Temperature (°C)"],
                          parameter_label="Temperature (°C)")
    assert [r.parameter for r in res.rows] == [25, 25, 40, 40]
    E = res.metric("E (GPa)")
    np.testing.assert_allclose(E, [60, 62, 64, 66], rtol=0.01)
    hdr, rows = res.table()
    assert hdr[:3] == ["Sample", "Temperature (°C)", "E (GPa)"] and hdr[-1] == "Warnings"
    sh, sr = B.group_stats(res, ["E (GPa)"])
    assert sh == ["Temperature (°C)", "n", "E (GPa) mean", "E (GPa) SD"]
    assert sr[0][:2] == [25, 2] and sr[0][2] == pytest.approx(61, rel=0.01)
    assert sr[0][3] == pytest.approx(np.std([E[0], E[1]], ddof=1))
    d = B.metric_vs_parameter_dataset(res, "E (GPa)")
    assert d.columns() == ["Temperature", "E (GPa)"] and d.channels["Temperature"].unit == "°C"
    np.testing.assert_allclose(d.values("E (GPa)"), E, rtol=1e-9)
    g = B.metric_vs_parameter_dataset(res, "UTS (MPa)", grouped=True)
    assert len(g.data) == 2 and "UTS SD" in g.columns()
    assert res.provenance["analysis"] == "Batch tensile"


def test_batch_errors_and_warnings_per_row():
    good = _tensile_ds("good 1C", 60, 400)
    e, s = _bilinear()
    bad_units = Dataset.from_arrays("mmmm 2C", {"e": (e / 100, "strain", "%"), "s": (s, "stress", "MPa")},
                                    kind="tensile")
    nogeo = Dataset.from_arrays("nogeo", {"d": (e, "displacement", "mm"), "F": (s, "force", "N")}, kind="tensile")
    res = B.batch_analyse([good, bad_units, nogeo], "tensile", parameter_regex=r"(\d+)C")
    assert res.rows[0].error is None and not res.rows[0].warnings
    assert any("plausible" in w for w in res.rows[1].warnings)
    assert res.rows[2].error and "area" in res.rows[2].error
    assert "ERROR" in res.table()[1][2][-1]


def test_batch_dsc_and_cyclic():
    T, q, t, truth = _two_step_run()
    dss = [Dataset.from_arrays(f"DSC {k}", {"T": (T, "temperature", "°C"), "t": (t, "time", "s"),
                                            "q": (q * k, "specific_heat_flow", "W/g")}, kind="dsc") for k in (1, 2)]
    res = B.batch_analyse(dss, "dsc", {"overlap": "split"})
    assert res.rows[0].values["ΔH cooling (J/g)"] == pytest.approx(truth["R"] + truth["M"], rel=0.02)
    assert res.rows[1].values["ΔH heating (J/g)"] == pytest.approx(2 * truth["A"], rel=0.02)
    assert res.rows[0].values["As (°C)"] == pytest.approx(res.rows[1].values["As (°C)"], abs=0.1)
    e, s = _qa_loop()
    E, S = np.r_[e, e[1:]], np.r_[s, s[1:]]
    cyc = Dataset.from_arrays("cyc", {"Strain": (-E, "strain", "%"), "Stress": (-S, "stress", "MPa")}, kind="cyclic")
    r = B.batch_analyse([cyc], "cyclic", {})
    assert r.rows[0].values["Dissipated first (MJ/m³)"] == pytest.approx(12.75, rel=1e-3)
    assert r.rows[0].values["Cycles (#)"] == 2
    assert any("sign inverted" in w for w in r.rows[0].warnings)


def test_gui_batch_tab(qapp, tmp_path):
    from labstudio.gui.analysis_panel import AnalysisPanel
    store = DatasetStore()
    fake = FakePlotPanel2()
    panel = AnalysisPanel(fake, store)
    tab = panel.batch_tab
    dss = [store.add(_tensile_ds(n, E, sy)) for n, E, sy in (("A 25C", 60, 400), ("B 25C", 62, 400),
                                                             ("C 40C", 70, 500))]
    store.add(Dataset.from_arrays("dsc", {"T": (np.arange(10.0), "temperature", "°C")}, kind="dsc"))
    tab.kind.setCurrentIndex(1)                                  # tensile
    assert tab.list.count() == 3
    tab._check_all(True)
    res = tab.run()
    assert res is not None and len(res.rows) == 3
    assert tab.table.table.rowCount() == 3 and tab.stats_table.table.rowCount() == 2
    tab.metric.setCurrentText("E (GPa)")
    tab.grouped.setChecked(True)
    n = len(store)
    d = tab.plot_metric()
    assert d is not None and len(store) == n + 1
    assert any(c[0] == "add_line" for c in fake.calls)            # SD bars
    out = tmp_path / "batch.csv"
    tab.table.export_csv(str(out))
    txt = out.read_text(encoding="utf-8-sig")
    assert txt.startswith("# LabStudio") and "Batch tensile" in txt
    st = panel.get_state()["batch"]
    assert sorted(st["checked"]) == sorted(d.id for d in dss)
    # nothing ticked → message
    tab._check_all(False)
    assert tab.run() is None and "Tick" in tab.status.text()
    panel.deleteLater()
