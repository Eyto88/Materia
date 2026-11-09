"""v1.1 tests for the Micrographs workstream: calibration sources, µ-capable burn-in font,
scale-bar / label exclusion, ImageJ perimeter, provenance, warnings, session state, presets,
responsive layout and (slow, opt-in) 8k performance."""
from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import pytest
import tifffile

from labstudio.core import image_ops as io
from labstudio.core import micro_analysis as ma
from labstudio.core.image_ops import Calibration


def disc(r, size=None, centre=None):
    size = size or int(2 * r + 10)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2 if centre is None else centre
    return (xx - c) ** 2 + (yy - c) ** 2 <= r * r


def lom_with_burned_bar(seed=7):
    """LOM-like image: bright matrix, dark discs, black box with white bar + '20 µm' bottom-right."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    H, W = 500, 700
    img = np.full((H, W), 190.0)
    yy, xx = np.mgrid[0:H, 0:W]
    centres = [(80 + 120 * i, 80 + 110 * j) for i in range(5) for j in range(3)]
    for (x, y) in centres:
        img[(xx - x) ** 2 + (yy - y) ** 2 <= 15 ** 2] = 60
    img += rng.normal(0, 8, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([W - 200, H - 70, W - 20, H - 15], fill=0)
    d.rectangle([W - 180, H - 58, W - 40, H - 52], fill=255)
    d.text((W - 140, H - 45), "20 µm", fill=255, font=io.annotation_font(18))
    return np.array(im), len(centres), (W - 200, H - 70, W - 19, H - 14)


# ====================================================================== calibration sources

def test_dpi_only_tiff_is_uncalibrated(tmp_path):
    img = np.full((60, 80), 120, np.uint8)
    for dpi, unit in ((144, "INCH"), (300, "INCH"), (57, "CENTIMETER")):
        p = tmp_path / f"dpi{dpi}.tif"
        tifffile.imwrite(p, img, resolution=(dpi, dpi), resolutionunit=unit)
        li = io.load_image(str(p))
        assert not li.calibration.calibrated
        assert li.calibration.source_label == "uncalibrated"
        assert li.metadata["calibration_source"] == "uncalibrated"
        assert li.metadata["print_dpi"] == pytest.approx(dpi * (2.54 if unit == "CENTIMETER" else 1))
    from PIL import Image

    q = tmp_path / "dpi.png"
    Image.fromarray(img).save(q, dpi=(144, 144))
    lq = io.load_image(str(q))
    assert not lq.calibration.calibrated and lq.metadata["print_dpi"] == pytest.approx(144, abs=0.1)
    cal = io.calibration_from_print_resolution(144)
    assert cal.calibrated and cal.from_print_resolution
    assert cal.pixel_size_m() == pytest.approx(0.0254 / 144)
    assert "144 dpi" in io.print_resolution_banner(144)


def test_trusted_sources(tmp_path):
    img = np.full((40, 50), 100, np.uint8)
    p = tmp_path / "fei.tif"
    tifffile.imwrite(p, img, resolution=(144, 144),
                     extratags=[(34682, "s", 0, "[Scan]\nPixelWidth=5e-08\n", True)])
    li = io.load_image(str(p))
    assert li.calibration.calibrated and li.calibration.source == "FEI tag 34682"
    assert li.calibration.to_unit("nm").unit_per_px == pytest.approx(50)
    q = tmp_path / "ij.tif"
    tifffile.imwrite(q, img, imagej=True, resolution=(4.0, 4.0), metadata={"unit": "micron"})
    lj = io.load_image(str(q))
    assert lj.calibration.source == io.SOURCE_IMAGEJ and lj.calibration.px_per_unit == pytest.approx(4)
    r = tmp_path / "ome.ome.tif"
    tifffile.imwrite(r, img, ome=True, metadata={"PhysicalSizeX": 0.25, "PhysicalSizeXUnit": "µm",
                                                 "PhysicalSizeY": 0.25, "PhysicalSizeYUnit": "µm"})
    lo = io.load_image(str(r))
    assert lo.calibration.source == io.SOURCE_OME
    assert lo.calibration.to_unit("µm").unit_per_px == pytest.approx(0.25)
    # Å and inch are rescaled (QA L1)
    assert Calibration(1.0, "Å").pixel_size_m() == pytest.approx(1e-10)
    assert Calibration(254.0, "in").to_unit("mm").px_per_unit == pytest.approx(10.0)
    assert Calibration().describe(with_source=True).endswith("uncalibrated")


def test_gui_dpi_banner_and_sources(ws, tmp_path):
    img = np.random.default_rng(0).integers(90, 160, (120, 160)).astype(np.uint8)
    p = tmp_path / "screenshot.tif"
    tifffile.imwrite(p, img, resolution=(144, 144), resolutionunit="INCH")
    doc = ws.open_image(str(p))
    assert not doc.calibration.calibrated
    assert "print resolution of 144 dpi" in ws.banner_text
    assert "Set scale from the scale bar" in ws.banner_text
    assert ws.banner_btn.text() == "Set scale" and ws.banner_btn2.text() == "Use dpi anyway"
    assert "uncalibrated" in ws.cal_source_label.text()
    ws._banner_secondary()                                     # "Use dpi anyway"
    assert doc.calibration.calibrated and doc.calibration.from_print_resolution
    assert "print resolution" in ws.cal_source_label.text()
    ws.set_mask(img < 100)
    res = ws.panels["particles"].run()
    assert any("print resolution" in w for w in res.warnings)
    assert all(r["Scale source"].startswith("print resolution") for r in ws.results.rows)
    # FEI image: calibrated, source shown
    q = tmp_path / "fei.tif"
    tifffile.imwrite(q, img, extratags=[(34682, "s", 0, "[Scan]\nPixelWidth=1e-07\n", True)])
    d2 = ws.open_image(str(q))
    assert d2.calibration.source == "FEI tag 34682"
    assert "FEI tag 34682" in ws.cal_source_label.text()
    # 'Set scale' records its source; LabStudio TIFF export round-trips with its source
    ws.set_calibration(Calibration.from_known_distance(100, 10, "µm"), global_=False)
    assert ws.cal.source == io.SOURCE_SCALE_BAR
    out = tmp_path / "exp.tif"
    ws.export_image(str(out), with_overlays=False)
    back = io.load_image(str(out))
    assert back.calibration.px_per_unit == pytest.approx(10)
    assert back.calibration.source.startswith(io.SOURCE_LABSTUDIO)
    assert "set from scale bar" in back.calibration.source


# ====================================================================== font

def test_burn_in_font_has_micro_sign():
    font = io.annotation_font(32)
    assert io.annotation_font_path() and io.annotation_font_path().lower().endswith(".ttf")
    for ch in "µ²°Å":
        assert io.font_has_glyph(font, ch), ch
    # pixel check: 'µ' must not render like the missing-glyph box
    def mask(c):
        m = font.getmask(c)
        return np.array(m, np.uint8).reshape(m.size[1], m.size[0])

    mu, tofu = mask("µ"), mask("\U0010fffd")
    assert mu.any()
    assert mu.shape != tofu.shape or not np.array_equal(mu, tofu)
    # the burned-in scale bar label: 'µm' region differs from a tofu rendering of '?m'
    img = np.full((200, 400), 100, np.uint8)
    cal = Calibration(4.0, "µm")
    a = io.burn_scale_bar(img, cal, 25, font_size=30)
    b = io.burn_scale_bar(img, cal, 25, font_size=30, unit_text="\U0010fffdm")
    assert not np.array_equal(a, b)
    from PIL import Image, ImageDraw, ImageFont

    # Pillow's built-in font has no µ: our output must differ from it
    try:
        old = ImageFont.load_default(30)
    except TypeError:
        old = ImageFont.load_default()
    im = Image.fromarray(np.full((60, 120), 0, np.uint8))
    ImageDraw.Draw(im).text((5, 5), "µm", fill=255, font=old)
    im2 = Image.fromarray(np.full((60, 120), 0, np.uint8))
    ImageDraw.Draw(im2).text((5, 5), "µm", fill=255, font=font)
    assert not np.array_equal(np.array(im), np.array(im2))
    t = io.draw_text(img, "ΔT = 5 °C, 2 µm²", (10, 10), 20, "yellow")
    assert t.ndim == 3 and (t != img[..., None]).any()


# ====================================================================== exclusion of burned-in bars

def test_scale_bar_box_excluded_from_particles():
    img, n_true, box = lom_with_burned_bar()
    boxes = io.detect_annotation_regions(img)
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 <= box[0] + 2 and y0 <= box[1] + 2 and x1 >= box[2] - 2 and y1 >= box[3] - 2
    assert (x1 - x0) * (y1 - y0) < 1.5 * (box[2] - box[0]) * (box[3] - box[1])
    mask = io.auto_threshold(img, "Otsu", dark_objects=True)[2]
    mask = io.remove_small(mask, 20)
    raw = ma.analyze_particles(mask)
    assert raw.count > n_true                            # box (+ label pieces) counted without exclusion
    ex = ma.analyze_particles(mask, exclude_regions=io.regions_mask(img.shape, boxes))
    assert ex.count == n_true
    assert ex.summary["Analysed area"] == pytest.approx(img.size - io.regions_mask(img.shape, boxes).sum())
    # clean images: nothing detected
    g, _, _ = ma.make_voronoi_grains((300, 300), 40, seed=1, noise=4)
    assert io.detect_annotation_regions(g) == []
    # data bar region is reported too
    assert io.detect_annotation_regions(g, data_bar_row=250)[0] == (0, 250, 300, 300)


def test_particles_touching_excluded_region_warn():
    m = np.zeros((100, 100), bool)
    m[40:60, 40:60] = True
    m[10:20, 10:20] = True
    excl = np.zeros_like(m)
    excl[50:70, 50:70] = True
    res = ma.analyze_particles(m, exclude_regions=excl)
    assert res.count == 2 and res.n_touching_excluded == 1
    assert any("touch an excluded region" in w for w in res.warnings)
    assert ma.analyze_particles(m, exclude_regions=excl, exclude_edges=True).count == 1


def test_gui_lom_particles_exclude_labels(ws):
    img, n_true, _ = lom_with_burned_bar()
    doc = ws.add_image(img, "lom.png", Calibration(10.0, "µm", io.SOURCE_SCALE_BAR))
    thr = ws.panels["threshold"]
    thr.dark.setChecked(True)
    thr.auto("Otsu")
    thr.apply()
    ws.mask_op("Remove small", lambda m: io.remove_small(m, 20))
    pp = ws.panels["particles"]
    assert pp.excl_labels.isChecked()                     # default on
    assert len(pp.preview_excluded()) == 1
    assert any(s.get("hatch") for s in doc.shapes)
    res = pp.run()
    assert res.count == n_true
    pp.excl_labels.setChecked(False)
    assert pp.run().count > n_true


# ====================================================================== ImageJ perimeter

def test_imagej_perimeter_values():
    # square of side n: traced polygon has 4 corners → P = 4n − 4(2 − √2) (ImageJ 1.54)
    for n in (20, 100):
        sq = np.zeros((n + 6, n + 6), bool)
        sq[3:-3, 3:-3] = True
        P = ma.perimeter_imagej(sq)
        assert P == pytest.approx(4 * n - 4 * (2 - math.sqrt(2)))
        circ = ma.analyze_particles(sq).rows[0]["Circ."]
        assert circ == pytest.approx(4 * math.pi * n * n / P ** 2)
    assert ma.analyze_particles(np.pad(np.ones((20, 20), bool), 3)).rows[0]["Circ."] == pytest.approx(0.8335, abs=1e-3)
    big = np.pad(np.ones((400, 400), bool), 3)
    assert ma.analyze_particles(big).rows[0]["Circ."] == pytest.approx(math.pi / 4, abs=0.01)
    # single pixel: 4 unit sides, 2 corners → 4 − 2(2 − √2) = 2√2 (ImageJ reports 2.828)
    sp = np.zeros((5, 5), bool)
    sp[2, 2] = True
    assert ma.perimeter_imagej(sp) == pytest.approx(2 * math.sqrt(2))
    # discs: ImageJ circularity ≈ 0.90–0.91 for large digital discs (QA reference values)
    ref = {10: 0.913, 50: 0.909}
    for r, c in ref.items():
        row = ma.analyze_particles(disc(r)).rows[0]
        assert row["Circ."] == pytest.approx(c, abs=0.002)
    r100 = ma.analyze_particles(disc(100)).rows[0]
    assert 0.88 <= r100["Circ."] <= 0.92
    assert ma.analyze_particles(disc(100), perimeter="crofton").rows[0]["Circ."] > 0.98
    # diagonal (8-connected) chain is one particle with a closed traced outline
    dg = np.eye(5, dtype=bool)
    assert ma.analyze_particles(dg).count == 1
    assert ma.perimeter_imagej(dg) > 0
    with pytest.raises(ValueError):
        ma.analyze_particles(sq, perimeter="bogus")


# ====================================================================== warnings

def test_low_statistics_and_uncalibrated_warnings():
    cal = Calibration(2.0, "µm")
    _, _, b = ma.make_voronoi_grains((200, 200), 12, seed=1)
    li = ma.linear_intercept(b, cal, n_lines=2)
    assert li.intersections < 50 and any("Low statistics" in w for w in li.warnings)
    pl = ma.planimetric(b, cal)
    assert pl.n_equivalent < 50 and any("Low statistics" in w for w in pl.warnings)
    gd = ma.grain_area_distribution(b, cal)
    assert any("Low statistics" in w for w in gd.warnings)
    # plenty of grains: no low-statistics warning
    _, _, b2 = ma.make_voronoi_grains((800, 800), 400, seed=2)
    assert not ma.linear_intercept(b2, cal, n_lines=8).warnings
    assert not ma.planimetric(b2, cal).warnings
    # no boundaries: G not reported (QA L2)
    empty = np.zeros((300, 300), bool)
    pe = ma.planimetric(empty, cal)
    assert pe.G is None and any("No grain boundaries" in w for w in pe.warnings)
    assert ma.linear_intercept(empty, cal).G is None
    # uncalibrated
    assert any("uncalibrated" in w for w in ma.planimetric(b2).warnings)
    assert any("uncalibrated" in w for w in ma.analyze_particles(disc(5)).warnings)


def test_gui_warning_rows_and_provenance_csv(ws, tmp_path):
    from labstudio.gui.image_viewer import Roi

    img, lab, b = ma.make_voronoi_grains((240, 240), 15, seed=4, noise=3)
    p = tmp_path / "few_grains.tif"
    tifffile.imwrite(p, img, extratags=[(34682, "s", 0, "[Scan]\nPixelWidth=1e-06\n", True)])
    doc = ws.open_image(str(p))
    ws.panels["grain"].run_all()
    warns = ws.summary.warnings
    assert any("Low statistics" in w for w in warns)
    assert "⚠" in ws.status_label.text()
    out = tmp_path / "summary.csv"
    ws.export_summary_csv(str(out))
    head = [ln for ln in open(out, encoding="utf-8-sig").read().splitlines() if ln.startswith("#")]
    joined = "\n".join(head)
    assert "Linear intercept (E112)" in joined and "FEI tag 34682" in joined
    assert "sha256=" in joined and "Low statistics" in joined
    # uncalibrated measurement → px + warning row
    d2 = ws.add_image(np.full((50, 50), 100, np.uint8), "raw")
    ws.add_roi(Roi("rect", [(0, 0), (10, 10)]))
    row = ws.measure_roi()[0]
    assert "Area (px²)" in row and row["Scale source"] == "uncalibrated"
    assert any("uncalibrated" in w for w in ws.results.warnings)
    # table text uses significant-figure formatting
    from labstudio.gui.app_settings import format_value as fv
    from labstudio.gui.image_results import format_value

    assert format_value(math.pi) == fv(math.pi)
    assert d2 is ws.doc and doc is not d2


def test_data_bar_excluded_from_grain_size(ws):
    img, lab, b = ma.make_voronoi_grains((300, 300), 60, seed=5, noise=3)
    bar = np.zeros((40, 300), np.uint8)
    bar[10:30, 20:150] = 255
    doc = ws.add_image(np.vstack([img, bar]), "sem", Calibration(5.0, "µm", "FEI tag 34682"))
    doc.data_bar_row = 300
    g = ws.panels["grain"]
    out = g.run_all()
    assert out["planimetric"].labels.shape == (300, 300)
    assert any("data bar" in w for w in ws.summary.warnings)


# ====================================================================== session

def test_session_round_trip(ws, qapp, tmp_path):
    from labstudio.gui.image_viewer import Roi
    from labstudio.gui.image_workspace import ImageWorkspace

    img = (np.arange(80 * 100) % 251).astype(np.uint8).reshape(80, 100)
    d1 = ws.add_image(img, "a.png", Calibration(2.0, "µm", "FEI tag 34682"))
    ws.add_roi(Roi("rect", [(5, 5), (40, 30)], name="box"))
    ws.add_roi(Roi("line", [(0, 0), (30, 40)]))
    ws.set_mask(img > 100, "thr")
    d1.annotations.append({"kind": "text", "text": "B2 µm", "pos": (10, 10), "size": 12, "color": "yellow"})
    d1.scale_bar = {"position": "lower left", "color": "black", "background": False, "length_rule": "auto"}
    ws.apply_op("Invert", io.invert)
    d2 = ws.add_image(np.zeros((30, 40), np.uint16) + 1000, "b.tif")
    ws.measure_roi()
    ws.set_current(0)
    state = ws.get_state()
    files = ws.get_state_files()
    import json

    json.dumps(state, allow_nan=True)                 # JSON-serialisable
    w2 = ImageWorkspace()
    w2.set_state_files(files)
    w2.set_state(json.loads(json.dumps(state)), {})
    assert len(w2.docs) == 2 and w2.doc.name == "a.png"
    a = w2.docs[0]
    assert np.array_equal(a.data, d1.data) and a.data.dtype == np.uint8
    assert np.array_equal(a.mask, d1.mask)
    assert a.calibration.px_per_unit == pytest.approx(2.0) and a.calibration.source == "FEI tag 34682"
    assert [r.name for r in a.rois] == ["box", d1.rois[1].name]
    assert a.rois[0].points == d1.rois[0].points
    assert a.annotations[0]["text"] == "B2 µm" and a.annotations[0]["pos"] == (10, 10)
    assert a.scale_bar["position"] == "lower left"
    assert w2.docs[1].data.dtype == np.uint16 and w2.docs[1].calibration.source_label == "uncalibrated"
    assert len(w2.results.rows) == len(ws.results.rows)
    # also through the lead's .lstudio session file
    from labstudio.core.dataset import DatasetStore
    from labstudio.core.session import load_session, save_session

    path = tmp_path / "s.lstudio"
    save_session(str(path), DatasetStore(), {"images": ws})
    w3 = ImageWorkspace()
    load_session(str(path), DatasetStore(), {"images": w3})
    assert len(w3.docs) == 2 and np.array_equal(w3.docs[0].mask, d1.mask)
    for w in (w2, w3):
        w.deleteLater()
    assert d2 is not None


# ====================================================================== presets

def test_scalebar_presets_round_trip(ws, tmp_path, monkeypatch):
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path))
    names = io.list_scalebar_presets()
    for n in ("Thesis (white bar, black box, bottom-right)", "Journal (black bar, no box, bottom-left)",
              "Minimal"):
        assert n in names
    st = {"position": "upper left", "length_rule": "fixed", "length": 50, "color": "yellow",
          "font_px": 22, "thickness": 6, "background": True, "box_color": "black",
          "unit_text": "μm", "show_label": True, "bogus": 1}
    path = io.save_scalebar_preset("My thesis µ style", st)
    assert os.path.dirname(path) == os.path.join(str(tmp_path), "styles")
    assert os.path.basename(path).startswith("scalebar_")
    back = io.load_scalebar_preset("My thesis µ style")
    assert back["length"] == 50 and back["unit_text"] == "μm" and "bogus" not in back
    assert "My thesis µ style" in io.list_scalebar_presets()
    # GUI: apply built-in, save, reload, apply to all open images
    cal = Calibration(4.0, "µm", io.SOURCE_SCALE_BAR)
    ws.add_image(np.full((200, 300), 120, np.uint8), "one", cal)
    ws.add_image(np.full((100, 150), 120, np.uint8), "two", cal)
    ann = ws.panels["annotate"]
    ann.refresh_presets()
    ann.apply_preset("Journal (black bar, no box, bottom-left)")
    assert ann.sb_pos.currentText() == "lower left" and not ann.sb_bg.isChecked()
    ann.save_preset("journal copy")
    assert io.load_scalebar_preset("journal copy")["position"] == "lower left"
    ann.apply_preset("Thesis (white bar, black box, bottom-right)")
    assert ann.apply_to_all() == len(ws.docs)
    assert all(d.scale_bar and d.scale_bar["position"] == "lower right" for d in ws.docs)
    spec = ws._scale_bar_spec(ws.doc)
    assert spec["label"].endswith("µm") and spec["length_px"] > 0
    assert io.delete_scalebar_preset("journal copy") and not io.delete_scalebar_preset("Minimal")


# ====================================================================== layout / perf helpers

def test_responsive_minimum_size(ws, qapp):
    ws.add_image(np.full((300, 400), 100, np.uint8), "x")
    ws.resize(1200, 680)
    ws.show()
    qapp.processEvents()
    hint = ws.minimumSizeHint()
    assert hint.width() <= 1200 and hint.height() <= 680, hint
    assert ws.left_panel.minimumSizeHint().width() <= 420
    assert ws.right_panel.minimumSizeHint().width() <= 420
    assert ws.right_panel.minimumWidth() <= 260
    ws._actions["toggle_right"].trigger()
    assert ws.right_panel.isHidden()
    ws._actions["toggle_right"].trigger()
    ws._actions["toggle_left"].trigger()
    assert ws.left_panel.isHidden()
    ws._actions["toggle_left"].trigger()
    ws.hide()


def test_display_pyramid_and_memory_bounded_undo(ws, qapp, monkeypatch):
    import labstudio.gui.image_workspace as W

    big = np.random.default_rng(0).integers(0, 255, (300, 9000)).astype(np.uint8)
    doc = ws.add_image(big, "wide")
    v = ws.viewer
    assert v.pyramid_step == 3
    assert v._pix.pixmap().width() == 3000
    out = v.render_image(False)
    assert out.width() == 9000                      # exports stay at native resolution
    v.zoom_100()
    v._update_detail()
    assert v._detail.isVisible()
    monkeypatch.setattr(W, "UNDO_MEMORY_LIMIT", 3 * big.nbytes)
    for i in range(6):
        ws.apply_op(f"Invert {i}", io.invert)
    assert 1 <= len(doc.undo_stack) <= 3


def test_fast_morphology_matches_reference():
    from skimage.morphology import closing, disk, opening

    g = np.random.default_rng(3).integers(0, 255, (120, 140)).astype(np.uint8)
    for r in (2, 6):
        ref_c, ref_o = closing(g, disk(r)), opening(g, disk(r))
        c, o = ma.grey_closing_disk(g, r), ma.grey_opening_disk(g, r)
        s = slice(2 * r, -2 * r)
        assert np.array_equal(ref_c[s, s], c[s, s]) and np.array_equal(ref_o[s, s], o[s, s])
    lab = np.zeros((6, 6), np.int32)
    lab[:, :3], lab[:, 4:] = 1, 2
    filled = ma.fill_label_gaps(lab)
    assert (filled > 0).all() and set(np.unique(filled)) == {1, 2}


@pytest.mark.skipif(os.environ.get("LABSTUDIO_SLOW") != "1", reason="slow: set LABSTUDIO_SLOW=1")
def test_8k_grain_size_time_and_memory(tmp_path):
    import resource

    img8, _, _ = ma.make_voronoi_grains((512, 512), 80, seed=1)
    big = np.tile(img8, (16, 16))
    cal = Calibration(5.0, "µm")
    t0 = time.perf_counter()
    b = ma.boundaries_from_image(big, sigma=0.5, thin=False, min_size=20)
    bt = ma.thin_boundaries(b)
    del b
    ma.grain_area_distribution(bt, cal, min_grain_px=10)
    ma.planimetric(bt, cal, 10, thin=False)
    ma.linear_intercept(bt, cal, 8, "grid", thin=False)
    dt = time.perf_counter() - t0
    mask = io.auto_threshold(big, "Otsu", True)[2]
    ma.analyze_particles(mask, cal)
    io.subtract_background(big, 50)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"8k grain size {dt:.1f} s, peak RSS {peak_mb:.0f} MB")
    assert dt < 15
    assert peak_mb < 1536


# ====================================================================== fixtures

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def ws(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path / "home"))
    from labstudio.gui.image_workspace import ImageWorkspace

    w = ImageWorkspace()
    w.resize(1280, 720)
    yield w
    w.deleteLater()
