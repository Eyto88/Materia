"""Tests for module B — plotting (run headless: QT_QPA_PLATFORM=offscreen)."""
from __future__ import annotations

import os
import struct
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from labstudio.core.dataset import Dataset, DatasetStore
from labstudio.gui import plot_math as PM

# =========================================================================== pure math (no Qt)


def test_product_and_quotient_units():
    assert PM.product_unit("MPa", "%") == "MPa·%"
    assert PM.product_unit("mW", "") == "mW"
    assert PM.quotient_unit("MPa", "%") == "MPa/%"
    assert PM.si_factor("temperature", "°C") == pytest.approx(1.0)
    assert PM.si_factor("temperature", "°F") == pytest.approx(5 / 9)
    assert PM.si_factor("strain", "%") == pytest.approx(0.01)


def test_integrate_whole_and_span_analytic():
    x = np.linspace(0, 2, 20001)
    y = x ** 2
    assert PM.integrate_span(x, y).area == pytest.approx(8 / 3, rel=1e-7)
    # span limits fall between samples -> interpolated exactly
    r = PM.integrate_span(x, y, 0.50003, 1.50007)
    assert r.area == pytest.approx((1.50007 ** 3 - 0.50003 ** 3) / 3, rel=1e-6)
    assert r.xs[0] == pytest.approx(0.50003) and r.xs[-1] == pytest.approx(1.50007)
    # piece-wise linear data are integrated exactly
    assert PM.integrate_span([0, 1, 3], [0, 2, 2]).area == pytest.approx(1 + 4)
    # reversed span arguments are fine
    assert PM.integrate_span(x, y, 1.5, 0.5).area == pytest.approx((1.5 ** 3 - 0.5 ** 3) / 3, rel=1e-6)


def test_integrate_linear_baseline_dsc_peak():
    T = np.linspace(-50, 100, 30001)
    A, mu, sig = 3.0, 20.0, 4.0
    base = 0.5 + 0.01 * T
    hf = A * np.exp(-0.5 * ((T - mu) / sig) ** 2) + base
    r = PM.integrate_span(T, hf, -10, 50, baseline="linear")
    assert r.area == pytest.approx(A * sig * np.sqrt(2 * np.pi), rel=1e-4)
    # without baseline the trapezoid under the baseline is included
    raw = PM.integrate_span(T, hf, -10, 50, baseline="none").area
    lin = (0.5 + 0.01 * 20) * 60
    assert raw == pytest.approx(A * sig * np.sqrt(2 * np.pi) + lin, rel=1e-4)
    slope, icpt = r.baseline_coeffs
    assert slope == pytest.approx(0.01, rel=1e-3) and icpt == pytest.approx(0.5, rel=1e-3)


def test_integrate_closed_loop_gives_enclosed_area():
    # rectangle loop: load along y=10, unload along y=4 between x=0..5 -> area 30
    x = np.r_[np.linspace(0, 5, 51), np.linspace(5, 0, 51)]
    y = np.r_[np.full(51, 10.0), np.full(51, 4.0)]
    r = PM.integrate_span(x, y)
    assert r.area == pytest.approx(30.0)
    assert not r.monotonic
    # span [1, 3] of the loop: 2 * 6 = 12
    assert PM.integrate_span(x, y, 1, 3).area == pytest.approx(12.0)
    with pytest.raises(ValueError):
        PM.integrate_span(x, y, 10, 20)


def test_area_conversions():
    q, _ = PM.area_conversions(1000.0, "stress", "MPa", "strain", "%")
    assert q[0].unit == "MPa·%" and q[1].unit == "MJ/m³" and q[1].value == pytest.approx(10.0)
    q, _ = PM.area_conversions(250.0, "stress", "GPa", "strain", "-")  # same thing in other units
    assert q[1].value == pytest.approx(250e3)
    q, notes = PM.area_conversions(500.0, "heat_flow", "mW", "time", "s", mass_mg=10)
    d = {x.unit: x.value for x in q}
    assert d["J"] == pytest.approx(0.5) and d["mJ"] == pytest.approx(500) and d["J/g"] == pytest.approx(50)
    q, _ = PM.area_conversions(12.0, "specific_heat_flow", "W/g", "time", "s")
    assert q[-1].unit == "J/g" and q[-1].value == pytest.approx(12.0)
    # heat flow vs temperature: needs heating rate
    q, notes = PM.area_conversions(600.0, "heat_flow", "mW", "temperature", "°C")
    assert len(q) == 1 and "heating rate" in notes[0]
    q, _ = PM.area_conversions(600.0, "heat_flow", "mW", "temperature", "K", heating_rate_K_per_min=10,
                               mass_mg=20)
    d = {x.unit: x.value for x in q}
    assert d["J"] == pytest.approx(3.6) and d["J/g"] == pytest.approx(180.0)
    q, _ = PM.area_conversions(3.0, "specific_heat_flow", "W/g", "temperature", "°C", heating_rate_K_per_min=10)
    assert q[-1].unit == "J/g" and q[-1].value == pytest.approx(18.0)
    q, _ = PM.area_conversions(2000.0, "force", "N", "displacement", "mm")
    assert {x.unit: x.value for x in q}["J"] == pytest.approx(2.0)


def test_slope_and_linear_fit():
    x = np.linspace(0, 1, 101)
    y = 450 * x + 3 + np.sin(40 * x) * 0.01
    f = PM.linear_fit(x, y, 0.1, 0.8)
    assert f.slope == pytest.approx(450, rel=1e-4) and f.intercept == pytest.approx(3, abs=0.01)
    assert f.r2 > 0.9999 and f.xmin >= 0.1 and f.xmax <= 0.8
    q = PM.slope_conversions(f.slope, "stress", "MPa", "strain", "%")
    assert q[1].unit == "GPa" and q[1].value == pytest.approx(45.0, rel=1e-4)
    with pytest.raises(ValueError):
        PM.linear_fit(x, y, 5, 6)


def test_find_peaks_xy_fwhm_and_valleys():
    x = np.linspace(0, 100, 5001)
    sig = 3.0
    y = 2 * np.exp(-0.5 * ((x - 30) / sig) ** 2) + 1 * np.exp(-0.5 * ((x - 70) / sig) ** 2)
    peaks = PM.find_peaks_xy(x, y, prominence=0.2)
    assert [round(p.x) for p in peaks] == [30, 70]
    assert peaks[0].fwhm == pytest.approx(2 * np.sqrt(2 * np.log(2)) * sig, rel=1e-3)
    assert peaks[0].prominence == pytest.approx(2.0, rel=1e-2)
    assert len(PM.find_peaks_xy(x, y, prominence=1.5)) == 1
    v = PM.find_peaks_xy(x, -y, prominence=0.2, sign="valleys")
    assert [round(p.x) for p in v] == [30, 70] and v[0].y == pytest.approx(-2, rel=1e-3)
    # min distance in x units merges the two peaks
    assert len(PM.find_peaks_xy(x, y, prominence=0.2, min_distance=50)) == 1


def test_span_stats_and_nearest_point():
    x = np.arange(10.0)
    y = (x - 4) ** 2
    s = PM.span_stats(x, y, 2, 8)
    assert s["min"] == 0 and s["x_at_min"] == 4 and s["max"] == 16 and s["n"] == 7
    pts = [np.array([[0, 0], [10, 10]], float), np.array([[3, 4], [np.nan, np.nan]])]
    assert PM.nearest_point(pts, (2.5, 4.2)) == (1, 0, pytest.approx(np.hypot(0.5, 0.2)))
    assert PM.nearest_point(pts, (100, 100), max_dist=5) is None


def test_display_indices_decimation():
    n = 200_000
    x = np.linspace(0, 10, n)
    y = np.sin(x * 5)
    y[123_457] = 50.0  # a spike must survive decimation
    idx = PM.display_indices(x, y, max_points=10_000)
    assert idx.size <= 10_002 and 123_457 in idx and idx[0] == 0 and idx[-1] == n - 1
    # zoomed view keeps only visible points (full resolution when few)
    idx = PM.display_indices(x, y, (2.0, 2.05), (-2, 2), max_points=10_000)
    xs = x[idx[idx >= 0]]
    assert xs.min() >= 1.9 and xs.max() <= 2.2 and idx.size < 2000
    # loop that leaves and re-enters the view -> a NaN break (-1) is inserted
    t = np.linspace(0, 4 * np.pi, n)
    idx = PM.display_indices(np.cos(t), np.sin(t), (0.5, 1.5), (-1.5, 1.5), max_points=1000)
    assert (idx == -1).any()
    assert np.isnan(PM.take_with_breaks(np.cos(t), idx)).any()


# =========================================================================== Qt panel

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _loop_ds(name="NiTi loop", n=400):
    e1 = np.linspace(0, 6, n)
    up = np.where(e1 < 1, 450 * e1, 450 + 10 * (e1 - 1))
    dn = np.where(e1 < 0.8, 250 / 0.8 * e1, 250 + 10 * (e1 - 0.8))
    e = np.r_[e1, e1[::-1]]
    s = np.r_[up, np.minimum(dn, up)[::-1]]
    return Dataset.from_arrays(name, {"Strain": (e, "strain", "%"), "Stress": (s, "stress", "MPa")},
                               kind="cyclic", metadata={"area_mm2": 0.785, "gauge_length_mm": 50})


def _dsc_ds(n=20001, name="DSC heating"):
    T = np.linspace(-50, 100, n)
    hf = 3.0 * np.exp(-0.5 * ((T - 20) / 4) ** 2) + 0.5 + 0.01 * T
    t = (T + 50) / 10 * 60  # 10 K/min
    return Dataset.from_arrays(name, {"Temperature": (T, "temperature", "°C"),
                                      "Heat flow": (hf, "heat_flow", "mW"), "Time": (t, "time", "s")},
                               kind="dsc", metadata={"mass_mg": 15.0, "heating_rate_K_per_min": 10.0})


@pytest.fixture
def panel(qapp):
    from labstudio.gui.plot_widget import PlotPanel
    store = DatasetStore()
    p = PlotPanel(store)
    p.resize(1400, 850)
    p.show()
    qapp.processEvents()
    yield p
    p.close()
    p.deleteLater()
    qapp.processEvents()


def _png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(33)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", head[16:24])
    return w, h, head[25]  # colour type (6 = RGBA)


def test_add_traces_labels_styles(panel):
    store = panel.store
    loop = store.add(_loop_ds())
    loop2 = store.add(_loop_ds("cycle 50"))
    t1 = panel.add_trace(loop, "Strain", "Stress")
    t2 = panel.add_trace(loop2, "Strain", "Stress", color="#ff0000", linewidth=3, marker="o")
    assert panel.traces == [t1, t2] and panel.selected_trace() is t2
    assert t1.line.get_color() != t2.line.get_color()
    assert t2.style["linewidth"] == 3 and t2.style["marker"] == "o"
    assert panel.ax.get_xlabel() == "Strain (%)" and panel.ax.get_ylabel() == "Stress (MPa)"
    x, y = t1.xy()
    assert x.size == 800 and y.max() == pytest.approx(500)
    assert panel.ax.get_legend() is not None
    assert panel.trace_manager.list.count() == 2
    panel.set_trace_style(t1, linestyle="--", label="first")
    assert t1.line.get_linestyle() == "--" and t1.label == "first"
    panel.set_trace_visible(t1, False)
    assert not t1.visible
    panel.remove_trace(t2)
    assert panel.traces == [t1]
    panel.clear()
    assert panel.traces == [] and panel.selected_trace() is None


def test_many_distinct_styles_and_secondary_axis(panel):
    ds = panel.store.add(_dsc_ds(2001))
    traces = [panel.add_trace(ds, "Temperature", "Heat flow", label=f"t{i}") for i in range(10)]
    combos = {(t.style["color"], t.line.get_linestyle()) for t in traces}
    assert len(combos) == 10  # colour cycle then line-style cycle
    assert panel._ax2 is None
    t = panel.add_trace(ds, "Temperature", "Time", secondary_y=True)
    assert panel._ax2 is not None and t.line.axes is panel.ax2
    assert panel.ax2.get_ylabel() == "Time (s)"
    panel.set_trace_secondary(t, False)
    assert panel._ax2 is None and t.line.axes is panel.ax


def test_customisation_live(panel):
    from labstudio.gui.plot_style import PRESET_NAMES
    ds = panel.store.add(_dsc_ds(2001))
    panel.add_trace(ds, "Temperature", "Heat flow")
    panel.add_trace(panel.store.add(_dsc_ds(2001, "DSC 2")), "Temperature", "Heat flow")
    s = panel.settings
    s.title = "NiTi DSC"
    s.x.label = "T"
    s.label_size = 14
    s.tick_size = 9
    s.grid_major = False
    s.minor_ticks = True
    s.legend_loc = "upper left"
    s.legend_ncol = 2
    s.x.auto_limits = False
    s.x.vmin, s.x.vmax = 0, 50
    s.y.scale = "log"
    s.y.invert = True
    panel.apply_settings()
    ax = panel.ax
    assert ax.get_title() == "NiTi DSC" and ax.get_xlabel() == "T"
    assert ax.xaxis.label.get_fontsize() == 14
    assert ax.get_xlim() == (0, 50) and ax.get_yscale() == "log" and ax.yaxis_inverted()
    assert ax.get_legend()._loc == 2 and ax.get_legend()._ncols == 2
    assert panel.customize_panel.ed_title.text() == "NiTi DSC"
    # widgets drive settings live
    cp = panel.customize_panel
    cp.ed_title.setText("Edited")
    cp.ed_title.editingFinished.emit()
    assert ax.get_title() == "Edited"
    cp.chk_leg.setChecked(False)
    assert ax.get_legend() is None
    # presets
    panel.apply_style_preset(PRESET_NAMES[1])
    assert panel.settings.tick_direction == "in" and panel.settings.export_dpi == 300
    assert "serif" in panel.settings.font_family
    assert panel.traces[0].line.get_linewidth() == pytest.approx(1.2)
    assert ax.get_title() == "Edited"  # text kept
    panel.apply_style_preset(PRESET_NAMES[2])
    assert ax.xaxis.label.get_fontsize() == 18


def test_unit_override_and_overlays(panel):
    ds = panel.store.add(_dsc_ds(2001))
    t = panel.add_trace(ds, "Temperature", "Heat flow")
    m = panel.add_marker(20.0, 3.0, "As")
    vl = panel.add_vline(30.0, "Af")
    panel.add_hline(1.0, "base")
    panel.add_line([0, 40], [0.5, 0.9], label="tangent")
    panel.fill_between(np.array([10, 20, 30]), np.array([1, 3, 1]), 0.5)
    assert m.get_xdata()[0] == pytest.approx(20.0)
    assert len(panel.overlays) >= 8
    assert "tangent" in [x.get_text() for x in panel.ax.get_legend().get_texts()]
    panel.set_display_unit("x", "K")
    # data stay canonical; line + overlays converted
    assert t.xy()[0][0] == pytest.approx(-50.0)
    assert t.line.get_xdata()[0] == pytest.approx(223.15)
    assert panel.ax.get_xlabel() == "Temperature (K)"
    mk = [a for a in panel.overlays if hasattr(a, "get_marker") and a.get_marker() == "o"][0]
    assert mk.get_xdata()[0] == pytest.approx(293.15)
    vls = [a for a in panel.overlays if hasattr(a, "get_xdata") and len(a.get_xdata()) == 2
           and a.get_xdata()[0] == a.get_xdata()[1]]
    assert any(a.get_xdata()[0] == pytest.approx(303.15) for a in vls)
    panel.set_span(10.0, 30.0)  # canonical °C
    assert panel.display_span() == pytest.approx((283.15, 303.15))
    assert panel.current_span() == pytest.approx((10.0, 30.0))
    assert panel.trace_manager.cb_unit["x"].currentText() == "K"
    panel.clear_overlays()
    assert panel.overlays == []
    panel.add_text(0, 1, "note")
    panel.clear_overlays()
    assert len(panel.overlays) == 1  # annotations survive clear_overlays
    panel.set_display_unit("x", None)
    assert t.line.get_xdata()[0] == pytest.approx(-50.0)
    with pytest.raises(ValueError):
        panel.set_display_unit("y", "MPa")


def test_tools_computations(panel):
    rows_seen = []
    panel.resultsAdded.connect(lambda rows: rows_seen.extend(rows))
    ds = panel.store.add(_dsc_ds())
    t = panel.add_trace(ds, "Temperature", "Heat flow")
    tools = panel.tools
    peaks = tools.find_peaks(prominence=1.0)
    assert len(peaks) == 1 and peaks[0].x == pytest.approx(20.053, abs=0.01)  # sloped baseline shifts max
    # FWHM is measured at half prominence (scipy convention); the sloped baseline narrows it slightly
    assert peaks[0].fwhm == pytest.approx(2 * np.sqrt(2 * np.log(2)) * 4, rel=0.05)
    assert any(r["quantity"] == "Peak 1 FWHM" and r["unit"] == "°C" for r in rows_seen)
    # integrate over span with linear baseline: mW·°C, converted with 10 K/min and 15 mg
    panel.set_span(-10, 50)
    res, qs = tools.integrate(baseline="linear")
    area = 3.0 * 4 * np.sqrt(2 * np.pi)
    assert res.area == pytest.approx(area, rel=1e-4)
    d = {q.unit: q.value for q in qs}
    assert qs[0].unit == "mW·°C"
    assert d["J"] == pytest.approx(area * 1e-3 / (10 / 60), rel=1e-4)
    assert d["J/g"] == pytest.approx(area * 1e-3 / (10 / 60) / 0.015, rel=1e-4)
    # same in K display unit gives the same energy
    panel.set_display_unit("x", "K")
    res_k, qs_k = tools.integrate(baseline="linear")
    assert res_k.area == pytest.approx(area, rel=1e-4) and qs_k[0].unit == "mW·K"
    panel.set_display_unit("x", None)
    # whole trace without baseline
    res_all, _ = tools.integrate(baseline="none", use_span=False)
    T, hf = t.xy()
    assert res_all.area == pytest.approx(np.trapezoid(hf, T), rel=1e-9)
    # heat flow vs time -> J directly
    t2 = panel.add_trace(ds, "Time", "Heat flow")
    _, qs_t = tools.integrate(t2, baseline="none", use_span=False)
    assert {q.unit for q in qs_t} >= {"mW·s", "J", "mJ", "J/g"}
    panel.remove_trace(t2)
    # stats + fit
    s = tools.stats(t, xmin=-10, xmax=50)
    assert s["max"] == pytest.approx(3.0 + 0.5 + 0.2, abs=0.01)
    loop = panel.store.add(_loop_ds())
    tl = panel.add_trace(loop, "Strain", "Stress")
    loading = panel.store.add(loop.subset(np.arange(len(loop.data)) < 400, name="loading"))
    f = tools.fit(panel.add_trace(loading, "Strain", "Stress"), xmin=0.05, xmax=0.7)
    assert f.slope == pytest.approx(450, rel=1e-6)
    assert any(r["quantity"] == "Modulus E" and r["value"] == pytest.approx(45.0) for r in rows_seen)
    # hysteresis loop area via integrate -> MJ/m³
    _, ql = tools.integrate(tl, use_span=False)
    assert ql[1].unit == "MJ/m³" and ql[1].value > 0
    assert panel.results.table.rowCount() == len(panel.results.rows) == len(rows_seen)


def test_results_table_copy_csv_clear(panel, tmp_path):
    panel.add_results([{"Tool": "Analysis", "Trace": "x", "Quantity": "As", "Value": 12.5, "Unit": "°C"},
                       {"tool": "Analysis", "trace": "x", "quantity": "Note", "value": "ok", "unit": ""}])
    assert panel.results.table.rowCount() == 2
    assert panel.results.table.item(0, 3).text() == "12.5"
    txt = panel.results.as_text()
    assert txt.splitlines()[0] == "Tool\tTrace\tQuantity\tValue\tUnit" and "As\t12.5\t°C" in txt
    panel.results.copy_to_clipboard()
    assert "As" in QtWidgets.QApplication.clipboard().text()
    p = tmp_path / "r.csv"
    panel.results.export_csv(str(p))
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("# LabStudio") and "Plot tools" in lines[0]
    content = [ln for ln in lines if not ln.startswith("#")]
    assert content[0] == "Tool,Trace,Quantity,Value,Unit" and content[1].startswith("Analysis,x,As,12.5")
    panel.results.clear()
    assert panel.results.table.rowCount() == 0


def test_export_png_size_formats(panel, tmp_path):
    ds = panel.store.add(_dsc_ds(5001))
    panel.add_trace(ds, "Temperature", "Heat flow")
    size_before = tuple(panel.figure.get_size_inches())
    p = tmp_path / "fig.png"
    panel.export_png(str(p), dpi=300, size_in=(3.0, 2.0))
    assert _png_size(p)[:2] == (900, 600)
    assert tuple(panel.figure.get_size_inches()) == pytest.approx(size_before)
    p2 = tmp_path / "fig_t.png"
    panel.export_png(str(p2), dpi=150, transparent=True, size_in=(8.5 / 2.54, 6.5 / 2.54))
    w, h, ctype = _png_size(p2)
    assert (w, h) == (round(8.5 / 2.54 * 150), round(6.5 / 2.54 * 150)) and ctype == 6
    from matplotlib.image import imread
    assert imread(str(p2))[2, 2, 3] == 0.0  # transparent corner
    for ext in ("svg", "pdf"):
        q = tmp_path / f"fig.{ext}"
        panel.export_figure(str(q), ext, 300)
        assert q.stat().st_size > 1000
    assert (tmp_path / "fig.pdf").read_bytes()[:4] == b"%PDF"
    # export dialog without exec()
    from labstudio.gui.plot_customize import ExportDialog
    dlg = ExportDialog(panel)
    dlg.cb_fmt.setCurrentText("PNG")
    dlg.cb_dpi.setCurrentText("600")
    dlg.sp_w.setValue(5.08)
    dlg.sp_h.setValue(2.54)
    out = dlg.do_export(str(tmp_path / "dlg.png"))
    assert _png_size(out)[:2] == (1200, 600)
    assert "1200 × 600" in dlg.lbl_px.text()
    assert len(panel.figure_to_png_bytes(72)) > 1000


def _mouse(canvas, name, x, y, button=None):
    from matplotlib.backend_bases import MouseEvent
    MouseEvent(name, canvas, x, y, button=button)._process()


def test_data_cursor_mouse_snap_and_pin(panel, qapp):
    ds = panel.store.add(_dsc_ds(2001))
    t = panel.add_trace(ds, "Temperature", "Heat flow")
    loop = panel.store.add(_loop_ds())
    panel.add_trace(loop, "Strain", "Stress", secondary_y=True)  # a second trace on the twin axes
    panel.tools.set_mode("cursor")
    panel.canvas.draw()
    # pixel position of the DSC peak, offset a few pixels
    T, hf = t.xy()
    i = int(np.argmax(hf))
    px, py = panel.ax.transData.transform((T[i], hf[i]))
    _mouse(panel.canvas, "motion_notify_event", px + 3, py - 4)
    assert panel.tools._tip.get_visible()
    ht, hi = panel.tools._hover
    assert ht is t
    # snapped point is the true nearest point in pixel space
    pix = panel.ax.transData.transform(np.column_stack([T, hf]))
    d = np.hypot(pix[:, 0] - (px + 3), pix[:, 1] - (py - 4))
    assert d[hi] == pytest.approx(d.min()) and d[hi] < 5
    assert "°C" in panel.tools._tip.get_text() and "mW" in panel.tools._tip.get_text()
    n0 = len(panel.results.rows)
    _mouse(panel.canvas, "button_press_event", px + 3, py - 4, button=1)
    rows = panel.results.rows[n0:]
    assert [r["unit"] for r in rows] == ["°C", "mW"] and rows[0]["tool"] == "Cursor"
    assert rows[0]["value"] == pytest.approx(T[i], abs=1.0)
    # far from any curve: nothing within snap radius
    bb = panel.ax.bbox
    assert panel.tools.nearest(bb.x0 + 1, bb.y1 - 1, max_dist=3) is None
    panel.tools.set_mode("navigate")
    assert not panel.tools._tip.get_visible()


def test_span_selector_mouse_drag_integrates(panel, qapp):
    ds = panel.store.add(_dsc_ds(4001))
    panel.add_trace(ds, "Temperature", "Heat flow")
    panel.tools.set_mode("integrate")
    panel.tools.cb_base.setCurrentText("linear")
    panel.canvas.draw()
    y_mid = panel.ax.transData.transform((0, 2.0))[1]
    x0 = panel.ax.transData.transform((-10, 0))[0]
    x1 = panel.ax.transData.transform((50, 0))[0]
    n0 = len(panel.results.rows)
    _mouse(panel.canvas, "button_press_event", x0, y_mid, button=1)
    _mouse(panel.canvas, "motion_notify_event", (x0 + x1) / 2, y_mid, button=1)
    _mouse(panel.canvas, "motion_notify_event", x1, y_mid, button=1)
    _mouse(panel.canvas, "button_release_event", x1, y_mid, button=1)
    sp = panel.current_span()
    assert sp is not None and sp[0] == pytest.approx(-10, abs=0.5) and sp[1] == pytest.approx(50, abs=0.5)
    new = panel.results.rows[n0:]
    assert new and new[0]["tool"] == "Integrate"
    j = next(r for r in new if r["unit"] == "J/g")
    assert j["value"] == pytest.approx(3 * 4 * np.sqrt(2 * np.pi) * 1e-3 / (10 / 60) / 0.015, rel=0.02)


def test_store_removal_and_new_columns(panel):
    ds = panel.store.add(_loop_ds())
    panel.add_trace(ds, "Strain", "Stress")
    ds.add_column("Stress GPa-ish", ds.values("Stress") * 2, "stress")
    panel.store.notify_changed()
    assert panel.trace_manager.cb_y.findData("Stress GPa-ish") >= 0
    panel.store.remove(ds)
    assert panel.traces == []


def test_large_traces_responsive_and_full_res_analysis(panel, qapp):
    n = 100_000
    T = np.linspace(-80, 120, n)
    dsets = []
    for k in range(20):
        hf = np.exp(-0.5 * ((T - 10 - k) / 3) ** 2) + 0.002 * k + np.sin(T) * 1e-3
        dsets.append(panel.store.add(Dataset.from_arrays(f"run {k}", {
            "Temperature": (T, "temperature", "°C"), "Heat flow": (hf, "heat_flow", "mW")}, kind="dsc")))
    t0 = time.perf_counter()
    traces = [panel.add_trace(d, "Temperature", "Heat flow") for d in dsets]
    panel.canvas.draw()
    elapsed = time.perf_counter() - t0
    assert elapsed < 20, elapsed
    assert all(len(t.line.get_xdata()) <= panel.display_max_points + 2 for t in traces)
    # cursor lookup across 2M points stays fast
    panel.tools.set_mode("cursor")
    panel.canvas.draw()
    t0 = time.perf_counter()
    for k in range(5):
        _mouse(panel.canvas, "motion_notify_event", 400 + k, 300)
    assert (time.perf_counter() - t0) / 5 < 0.5
    # analysis uses all points
    res, _ = panel.tools.integrate(traces[0], use_span=False)
    T0, hf0 = traces[0].xy()
    assert T0.size == n and res.area == pytest.approx(np.trapezoid(hf0, T0), rel=1e-12)
    # zoom → view-aware re-decimation shows full resolution in the zoomed window
    panel.ax.set_xlim(10, 12)
    panel._redecimate()
    xd = np.asarray(traces[0].line.get_xdata(), float)
    xd = xd[np.isfinite(xd)]
    assert np.sum((xd >= 10) & (xd <= 12)) == np.sum((T0 >= 10) & (T0 <= 12))
