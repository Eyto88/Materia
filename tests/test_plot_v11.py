"""v1.1 tests for module B — plotting (QT_QPA_PLATFORM=offscreen)."""
from __future__ import annotations

import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LABSTUDIO_HOME", tempfile.mkdtemp(prefix="labstudio_plot_v11_"))

import numpy as np
import pytest

from labstudio.core.dataset import Dataset, DatasetStore
from labstudio.gui import plot_math as PM

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

PEAK_AREA = 3.0 * 4 * np.sqrt(2 * np.pi)   # heating endotherm, mW·°C


# =========================================================================== helpers


def _heat_cool_ds(n=3001, name="DSC heat+cool"):
    Th = np.linspace(-50, 100, n)
    Tc = Th[::-1]
    hf_h = 3.0 * np.exp(-0.5 * ((Th - 20) / 4) ** 2) + 0.5 + 0.01 * Th
    hf_c = -2.0 * np.exp(-0.5 * ((Tc - 5) / 4) ** 2) + 0.3 + 0.01 * Tc
    T = np.r_[Th, Tc[1:]]
    hf = np.r_[hf_h, hf_c[1:]]
    return Dataset.from_arrays(name, {"Temperature": (T, "temperature", "°C"), "Heat flow": (hf, "heat_flow", "mW")},
                               kind="dsc", metadata={"mass_mg": 15.0, "heating_rate_K_per_min": 10.0})


def _loop_ds(name="loop", n=400):
    x = np.r_[np.linspace(0, 5, n), np.linspace(5, 0, n)]
    y = np.r_[np.full(n, 10.0), np.full(n, 4.0)]
    return Dataset.from_arrays(name, {"Strain": (x, "strain", "%"), "Stress": (y, "stress", "MPa")}, kind="cyclic")


def _line_ds(name, slope=1.0, off=0.0, x0=0.0, x1=10.0, n=201, **md):
    x = np.linspace(x0, x1, n)
    return Dataset.from_arrays(name, {"Strain": (x, "strain", "%"), "Stress": (slope * x + off, "stress", "MPa")},
                               kind="tensile", metadata=md)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def panel(qapp):
    from labstudio.gui.plot_widget import PlotPanel
    p = PlotPanel(DatasetStore())
    p.resize(1400, 850)
    p.show()
    qapp.processEvents()
    yield p
    p.close()
    p.deleteLater()
    qapp.processEvents()


# =========================================================================== 1. segments


def test_monotonic_segments_pure():
    rng = np.random.default_rng(1)
    T = np.r_[np.linspace(-50, 100, 5000), np.full(200, 100.0), np.linspace(100, -50, 5000),
              np.linspace(-50, 100, 5000)] + rng.normal(0, 0.03, 15200)
    segs = PM.monotonic_segments(T)
    assert [d for *_, d in segs] == [1, -1, 1]
    assert PM.segment_names(segs, "temperature") == ["heating 1", "cooling 1", "heating 2"]
    assert PM.monotonic_segments(np.linspace(0, 1, 100)) == [(0, 100, 1)]
    assert PM.segment_names(PM.monotonic_segments(np.r_[0:5:0.1, 5:0:-0.1]), "strain") == ["loading 1",
                                                                                           "unloading 1"]


def test_integrate_heat_cool_gives_heating_only_with_warning(panel):
    ds = panel.store.add(_heat_cool_ds())
    t = panel.add_trace(ds, "Temperature", "Heat flow")
    panel.set_span(-10, 50)
    n0 = len(panel.results.rows)
    res, qs = panel.tools.integrate(t, baseline="linear")
    assert res.area == pytest.approx(PEAK_AREA, rel=2e-3)
    assert res.segment == "heating 1" and res.warnings
    rows = panel.results.rows[n0:]
    warn = [r for r in rows if r["warning"]]
    assert warn and "heating 1" in warn[0]["value"] and "cooling 1" in warn[0]["value"]
    assert panel.status_label.text().startswith("⚠")
    assert "FFE3A3" in panel.status_label.styleSheet()
    # amber row in the table
    rr = panel.results.rows.index(warn[0])
    assert panel.results.table.item(rr, 3).background().color().name().upper() == "#FFE3A3"
    jg = {q.unit: q.value for q in qs}["J/g"]
    assert jg == pytest.approx(PEAK_AREA * 1e-3 / (10 / 60) / 0.015, rel=2e-3)
    # provenance kept with the rows
    prov = rows[0]["provenance"]
    assert prov["parameters"]["segment_used"] == "heating 1" and prov["parameters"]["baseline"] == "linear"
    assert prov["parameters"]["heating_rate_K_per_min"] == 10.0 and prov["warnings"]
    # explicit segment 2 = cooling exotherm (sign-corrected to ∫hf dt < 0)
    res2, qs2 = panel.tools.integrate(t, baseline="linear", segment=2, xmin=-25, xmax=35)
    assert res2.segment == "cooling 1" and not res2.warnings
    assert {q.unit: q.value for q in qs2}["J"] == pytest.approx(-2 * 4 * np.sqrt(2 * np.pi) * 1e-3 / (10 / 60),
                                                                rel=5e-3)
    # whole path → mixed, warned
    res3, qs3 = panel.tools.integrate(t, baseline="none", segment="whole")
    assert res3.loop and res3.warnings and "(enclosed loop)" in qs3[0].quantity
    # combo offers the segments
    panel.select_trace(t)
    items = [panel.tools.cb_seg.itemText(i) for i in range(panel.tools.cb_seg.count())]
    assert items[0].startswith("auto") and any("cooling 1" in s for s in items)


def test_loop_area_default_unchanged_and_labelled(panel):
    ds = panel.store.add(_loop_ds())
    t = panel.add_trace(ds, "Strain", "Stress")
    res, qs = panel.tools.integrate(t, use_span=False)
    assert res.area == pytest.approx(30.0) and res.loop
    assert qs[1].unit == "MJ/m³" and qs[1].value == pytest.approx(0.3)
    assert "enclosed loop" in qs[0].quantity
    assert res.warnings  # policy: doubling back is always flagged


# =========================================================================== 2. labels


def test_labels_do_not_overlap(panel, qapp):
    ds = panel.store.add(_line_ds("l"))
    panel.add_trace(ds, "Strain", "Stress")
    anns = [panel.add_label(5.0 + 0.02 * k, 5.0 + 0.02 * k, f"Label number {k}", marker="o") for k in range(10)]
    panel.add_marker(5.0, 5.1, "As 12.0 °C")
    panel.canvas.draw()
    panel.relayout_labels()
    r = panel.canvas.get_renderer()
    from matplotlib.text import Text
    boxes = [Text.get_window_extent(a, r) for a in panel.labels]
    assert len(boxes) == 11
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (i, j)
    assert any(a.arrow_patch.get_visible() for a in anns)  # leader lines for displaced labels
    # still fine after a resize
    panel.resize(1100, 700)
    qapp.processEvents()
    panel.canvas.draw()
    panel.relayout_labels()
    boxes = [Text.get_window_extent(a, panel.canvas.get_renderer()) for a in panel.labels]
    assert not any(boxes[i].overlaps(boxes[j]) for i in range(11) for j in range(i + 1, 11))


# =========================================================================== 3. colorbar / legend


def test_colorbar_and_legend_modes(panel):
    for k in range(1, 21):
        panel.add_trace(panel.store.add(_line_ds(f"c{k}", slope=k)), "Strain", "Stress", label=f"Cycle {k}")
    texts = [t.get_text() for t in panel.ax.get_legend().get_texts()]
    assert texts[0] == "Cycle 1" and texts[-1] == "Cycle 20" and len(texts) == 3 and "more" in texts[1]
    panel.settings.legend_mode = "full"
    panel._apply_all()
    assert len(panel.ax.get_legend().get_texts()) == 20
    panel.settings.legend_mode = "colorbar"
    panel._apply_all()
    assert panel.ax.get_legend() is None and len(panel.colorbars) == 1
    panel.settings.legend_mode = "auto"
    panel._apply_all()
    n_axes = len(panel.figure.axes)
    cb = panel.add_colorbar("viridis", 1, 20, "Cycle number", ticks=[1, 10, 20])
    assert cb in panel.colorbars and len(panel.figure.axes) == n_axes + 1
    assert cb.ax.get_ylabel() == "Cycle number"
    assert panel.ax.get_legend() is None  # numbered series dropped while a colorbar shows it
    panel.remove_colorbars()
    assert panel.colorbars == []


# =========================================================================== 4. preview


def test_preview_at_export_size(panel, qapp, tmp_path):
    panel.add_trace(panel.store.add(_line_ds("l")), "Strain", "Stress")
    panel.set_preview_export_size(True, (8.5, 6.5))
    qapp.processEvents()
    w, h = panel.preview_size_px()
    assert (w, h) == (round(8.5 / 2.54 * 100), round(6.5 / 2.54 * 100))
    assert panel.canvas.width() == w and panel.canvas.height() == h
    assert tuple(panel.figure.get_size_inches()) == pytest.approx((8.5 / 2.54, 6.5 / 2.54), abs=0.02)
    assert panel.customize_panel.chk_preview.isChecked()
    p = tmp_path / "x.png"
    panel.export_png(str(p), dpi=100, size_in=(8.5 / 2.54, 6.5 / 2.54))
    panel.set_preview_export_size(False)
    qapp.processEvents()
    assert panel.canvas.maximumWidth() > 10000


# =========================================================================== 5. results formatting / csv


def test_results_format_value_and_csv(panel, tmp_path):
    from labstudio.gui.app_settings import settings
    panel.add_results([{"Tool": "T", "Trace": "t", "Quantity": "q", "Value": 20.069457240309497, "Unit": "J/g"}],
                      provenance_rec={"analysis": "X", "parameters": {"span": [1, 2]}, "warnings": ["w1"]})
    assert panel.results.table.item(0, 3).text() == "20.069"
    assert "20.069\tJ/g" in panel.results.as_text()
    p = tmp_path / "r.csv"
    settings().set("csv_full_precision", False)
    panel.results.export_csv(str(p))
    txt = p.read_text(encoding="utf-8-sig")
    assert "# parameters:" in txt and "# [1] X" in txt and "w1" in txt and ",20.069," in txt
    settings().set("csv_full_precision", True)
    panel.results.export_csv(str(p))
    assert "20.069457240309497" in p.read_text(encoding="utf-8-sig")
    settings().set("csv_full_precision", False)


# =========================================================================== 6. datasets


def test_dataset_management(panel, qapp):
    st = panel.store
    a, b, c = (st.add(_line_ds(n)) for n in "abc")
    ta = panel.add_trace(a, "Strain", "Stress")
    panel.rename_dataset(a, "alpha")
    assert a.name == "alpha" and ta.label == "alpha"
    assert panel.trace_manager.ds_list.item(0).text() == "alpha"
    panel.move_dataset(c, 0)
    assert [d.name for d in st] == ["c", "alpha", "b"]
    dup = panel.duplicate_dataset(b)
    assert dup.id != b.id and len(st) == 4 and dup.data is not b.data
    panel.remove_dataset(a)
    assert ta not in panel.traces and st.get(a.id) is None
    # inline rename through the list widget
    lst = panel.trace_manager.ds_list
    it = lst.item(0)
    assert it.data(0x0100) == c.id
    it.setText("gamma")
    qapp.processEvents()
    assert c.name == "gamma" and lst.item(0).text() == "gamma"


# =========================================================================== 7. responsive


def test_minimum_size_fits_small_window(panel, qapp):
    panel.resize(1200, 680)
    qapp.processEvents()
    hint = panel.minimumSizeHint()
    assert hint.width() <= 1200 and hint.height() <= 680, hint
    assert panel.left_scroll.minimumSizeHint().width() <= 420
    assert panel.right_scroll.minimumSizeHint().width() <= 420
    panel.tools.act_side.setChecked(True)
    assert not panel.side_panels_visible()
    panel.tools.act_side.setChecked(False)
    assert panel.side_panels_visible()


# =========================================================================== 8. state


def test_state_round_trip(panel, qapp):
    from labstudio.gui.plot_widget import PlotPanel
    dsc = panel.store.add(_heat_cool_ds(501))
    lp = panel.store.add(_loop_ds())
    panel.set_layout(1, 2)
    panel.add_trace(dsc, "Temperature", "Heat flow", label="DSC", color="#ff0000", linewidth=2.5)
    panel.add_trace(lp, "Strain", "Stress", label="loop", panel=1, linestyle="--")
    panel.set_display_unit("x", "K", panel=0)
    panel.settings.title = "Heating"
    panel.settings.panel(1).title = "Loop"
    panel.settings.panel(1).y.auto_limits = False
    panel.settings.panel(1).y.vmin, panel.settings.panel(1).y.vmax = 0, 20
    panel.settings.font_family = "serif"
    panel.settings.legend_mode = "full"
    panel.apply_settings()
    panel.set_span(0, 40)
    state = json.loads(json.dumps(panel.get_state()))
    p2 = PlotPanel(panel.store)
    try:
        p2.set_state(state, {dsc.id: dsc, lp.id: lp})
        assert p2.n_panels == 2 and [t.label for t in p2.traces] == ["DSC", "loop"]
        assert p2.traces[1].panel_index == 1 and p2.traces[1].line.axes is p2.panel_axes(1)
        assert p2.traces[0].style["color"] == "#ff0000" and p2.traces[0].style["linewidth"] == 2.5
        assert p2.display_unit("x", 0) == "K" and p2.panel_axes(0).get_xlabel() == "Temperature (K)"
        assert p2.panel_axes(0).get_title() == "Heating" and p2.panel_axes(1).get_title() == "Loop"
        assert p2.panel_axes(1).get_ylim() == (0, 20)
        assert p2.settings.font_family == "serif" and p2.settings.legend_mode == "full"
        assert p2.current_span() == pytest.approx((0, 40))
        assert "ui" in state and len(state["ui"]["h_split"]) == 3
    finally:
        p2.close()
        p2.deleteLater()


# =========================================================================== 9. templates


def test_templates_save_load_roundtrip(panel, tmp_path, monkeypatch):
    from labstudio.gui import plot_templates as TPL
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path))
    assert TPL.styles_dir() == str(tmp_path / "styles")
    names = TPL.list_templates()
    assert TPL.THESIS in names and TPL.JOURNAL in names and TPL.SLIDES in names
    panel.add_trace(panel.store.add(_line_ds("a")), "Strain", "Stress")
    panel.settings.title_size = 17
    panel.settings.color_cycle = ["#111111", "#222222"]
    panel.settings.export_w_cm, panel.settings.export_h_cm = 12.0, 9.0
    path = panel.save_template("Mine")
    assert os.path.isfile(path) and path.startswith(str(tmp_path / "styles"))
    t = TPL.get_template("Mine")
    assert t["settings"]["title_size"] == 17 and t["figure_size_cm"] == [12.0, 9.0]
    assert t["rc"]["axes.titlesize"] == 17
    # apply a builtin, then ours back
    panel.apply_template(TPL.JOURNAL)
    assert panel.settings.label_size == 8 and panel.settings.export_w_cm == 8.5
    assert panel.ax.xaxis.label.get_fontsize() == 8
    panel.apply_template("Mine")
    assert panel.settings.title_size == 17 and panel.traces[0].style["color"] == "#111111"
    # export / import
    out = tmp_path / "share.json"
    TPL.export_template("Mine", str(out))
    d = json.loads(out.read_text())
    d["name"] = "Labmate"
    out.write_text(json.dumps(d))
    assert TPL.import_template(str(out)) == "Labmate" and "Labmate" in TPL.list_templates()
    assert TPL.delete_template("Labmate") and "Labmate" not in TPL.list_templates()
    # default for new figures
    from labstudio.gui.plot_widget import PlotPanel
    panel.set_default_template(TPL.SLIDES)
    try:
        p2 = PlotPanel(DatasetStore())
        assert p2.settings.label_size == 18 and p2.settings.template == TPL.SLIDES
        p2.close()
        p2.deleteLater()
    finally:
        panel.set_default_template("")


# =========================================================================== 10. palettes


def test_palettes_and_colour_by(panel):
    from labstudio.gui.plot_palettes import PALETTES, palette_colors
    assert palette_colors("viridis", 5)[0].lower() == "#440154"
    assert len(palette_colors("Okabe–Ito", 12)) == 12
    trs = [panel.add_trace(panel.store.add(_line_ds(f"s{k}", slope=k, anneal_T_C=400 + 25 * k)), "Strain", "Stress")
           for k in range(4)]
    cols = panel.apply_palette("Tableau 10")
    assert [t.style["color"].upper() for t in trs] == [c.upper() for c in PALETTES["Tableau 10"][:4]]
    assert panel.settings.color_cycle == cols
    cm = panel.apply_palette("plasma")
    assert len(cm) == 4 and trs[0].style["color"] == cm[0]
    assert "anneal_T_C" in panel.metadata_keys()
    out = panel.color_traces_by("anneal_T_C", "viridis")
    assert out[0] != out[-1] and len(panel.colorbars) == 1
    assert panel.colorbars[0].ax.get_ylabel() == "anneal_T_C"
    panel.color_traces_by("order", "cividis")
    assert len(panel.colorbars) == 1  # replaced, not stacked


# =========================================================================== 11. multi-panel


def test_multipanel_traces_and_export(panel, tmp_path):
    panel.set_layout(2, 2)
    assert panel.n_panels == 4
    ds = [panel.store.add(_line_ds(f"p{k}", slope=k + 1)) for k in range(4)]
    trs = [panel.add_trace(d, "Strain", "Stress", panel=k) for k, d in enumerate(ds)]
    for k, t in enumerate(trs):
        assert t.line.axes is panel.panel_axes(k)
    panel.set_active_panel(2)
    assert panel.ax is panel.panel_axes(2)
    m = panel.add_marker(1, 1, "x")
    assert m.axes is panel.panel_axes(2)
    panel.settings.title = "first"
    panel.settings.panel(3).title = "fourth"
    panel._apply_all()
    assert panel.panel_axes(0).get_title() == "first" and panel.panel_axes(3).get_title() == "fourth"
    labels = [a.get_text() for a in panel._deco if a.get_gid() == "panel_label"]
    assert labels == ["(a)", "(b)", "(c)", "(d)"]
    panel.set_trace_panel(trs[0], 3)
    assert trs[0].line.axes is panel.panel_axes(3)
    panel.set_layout(2, 2, share_x=True)
    assert panel.panel_axes(1).get_shared_x_axes().joined(panel.panel_axes(0), panel.panel_axes(1))
    p = tmp_path / "grid.png"
    panel.export_png(str(p), dpi=100, size_in=(6, 5))
    assert p.stat().st_size > 5000
    panel.set_layout(1, 1)
    assert panel.n_panels == 1 and all(t.panel_index == 0 for t in panel.traces)
    assert all(t.line.axes is panel.ax for t in panel.traces)


# =========================================================================== 12. offsets


def test_offsets_display_only(panel):
    a = panel.add_trace(panel.store.add(_line_ds("a")), "Strain", "Stress")
    b = panel.add_trace(panel.store.add(_line_ds("b")), "Strain", "Stress")
    panel.set_offsets(5.0, labels=True)
    assert np.asarray(b.line.get_ydata())[0] == pytest.approx(5.0)
    assert np.asarray(a.line.get_ydata())[0] == pytest.approx(0.0)
    assert b.xy_display()[1][0] == pytest.approx(0.0) and b.xy()[1][0] == pytest.approx(0.0)
    # tools use original data
    s = panel.tools.stats(b, xmin=0, xmax=10)
    assert s["min"] == pytest.approx(0.0) and s["max"] == pytest.approx(10.0)
    # cursor snaps to the drawn (offset) curve but reports original values
    panel.canvas.draw()
    px, py = panel.ax.transData.transform((2.0, 2.0 + 5.0))
    tr, i, _ = panel.tools.nearest(px, py)
    assert tr is b
    assert panel.tools._readout(b, i)[1] == pytest.approx(2.0, abs=0.06)
    assert [x.get_text() for x in panel._deco if x.get_gid() == "offset_label"] == ["a", "b"]
    panel.set_offsets(50.0, mode="percent")
    assert b.offset[1] == pytest.approx(5.0)  # 50 % of the 0..10 range
    panel.set_offsets(0.0, labels=False)
    assert np.asarray(b.line.get_ydata())[0] == pytest.approx(0.0)


# =========================================================================== 13. averaging


def test_average_traces_mean_sd(panel):
    rng = np.random.default_rng(3)
    offs = [1.0, 2.0, 4.0]
    trs = [panel.add_trace(panel.store.add(_line_ds(f"r{k}", slope=2.0, off=o, x0=k * 0.5, x1=10 + k * 0.5,
                                                    n=150 + 37 * k, heating_rate_K_per_min=10)),
                           "Strain", "Stress") for k, o in enumerate(offs)]
    n_ds = len(panel.store)
    avg = panel.average_traces(trs, n_points=101)
    assert len(panel.store) == n_ds + 1
    ds = avg.ds
    x = ds.values("Strain")
    assert x[0] == pytest.approx(1.0) and x[-1] == pytest.approx(10.0) and x.size == 101
    assert np.allclose(ds.values("Stress"), 2 * x + np.mean(offs))
    assert np.allclose(ds.values("Stress SD"), np.std(offs, ddof=1))
    assert np.all(ds.values("n") == 3)
    assert avg.sd_column == "Stress SD" and avg.error_mode == "band" and avg._err
    assert ds.metadata["heating_rate_K_per_min"] == 10
    panel.set_trace_error(avg, "bars", every=10)
    assert len(avg._err[0].get_segments()) == 11
    assert any(r["warning"] for r in panel.results.rows)  # common range < 90 % → warned
    with pytest.raises(ValueError):
        panel.average_traces(trs[:1])
    _ = rng
