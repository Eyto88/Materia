"""Tests for module D — Micrographs (core image ops, metallography, workspace GUI)."""
from __future__ import annotations

import csv
import math
import os

import numpy as np
import pytest
import tifffile

from labstudio.core import image_ops as io
from labstudio.core import micro_analysis as ma
from labstudio.core.image_ops import Calibration


# ======================================================================================
# helpers
# ======================================================================================

def discs(shape=(300, 400), centres=((80, 80), (250, 90), (120, 220), (300, 210)), r=25):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    m = np.zeros(shape, bool)
    for cx, cy in centres:
        m |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return m


def expected_intercept(n, area):
    """Poisson–Voronoi: P_L = 4√λ/π → l̄ = π/(4√λ)."""
    return math.pi / (4 * math.sqrt(n / area))


# ======================================================================================
# calibration & metadata
# ======================================================================================

def test_calibration_conversions():
    c = Calibration(2.0, "um")
    assert c.unit == "µm" and c.calibrated
    assert c.length(10) == pytest.approx(5.0)
    assert c.area(100) == pytest.approx(25.0)
    assert c.area_unit == "µm²"
    nm = c.to_unit("nm")
    assert nm.px_per_unit == pytest.approx(0.002)
    assert nm.length(10) == pytest.approx(5000.0)
    assert c.to_unit("mm").length(10) == pytest.approx(0.005)
    assert c.pixel_size_m() == pytest.approx(0.5e-6)
    assert c.to_px(5.0) == pytest.approx(10.0)
    s = Calibration.from_known_distance(200, 50, "µm")
    assert s.px_per_unit == pytest.approx(4.0)
    p = Calibration.from_pixel_size(2.345e-9, "m")
    assert p.unit == "nm" and p.unit_per_px == pytest.approx(2.345)
    p2 = Calibration.from_pixel_size(0.5e-6, "m")
    assert p2.unit == "µm" and p2.unit_per_px == pytest.approx(0.5)
    assert not Calibration().calibrated
    with pytest.raises(ValueError):
        Calibration.from_known_distance(0, 1, "µm")
    with pytest.raises(ValueError):
        Calibration().length_in(5, "µm")


FEI_TEXT = """[User]
Date=11/06/2026
[Beam]
HV=20000
[Scan]
InternalScan=1
PixelWidth=4.9609375e-09
PixelHeight=4.9609375e-09
HorFieldsize=5.08e-06
[EScan]
PixelWidth=1e-07
"""

ZEISS_TEXT = ("\r\nAP_PIXEL_SIZE\r\nImage Pixel Size = 2.345 nm\r\nAP_WD\r\nWD = 8.5 mm\r\n"
              "DP_DETECTOR_CHANNEL\r\nSignal A = SE2\r\n")


def test_parse_metadata_strings():
    assert io.parse_fei_metadata(FEI_TEXT) == pytest.approx(4.9609375e-09)
    assert io.parse_zeiss_metadata(ZEISS_TEXT) == pytest.approx(2.345e-9)
    assert io.parse_zeiss_metadata("Pixel Size = 12.3 µm") == pytest.approx(12.3e-6)
    assert io.parse_zeiss_metadata("nothing here") is None
    assert io.parse_sidecar_text("$CM_FORMAT JEOL/SEM\n$$SM_MICRON_BAR 200\n$$SM_MICRON_MARKER 10um\n") \
        == pytest.approx(10e-6 / 200)
    assert io.parse_sidecar_text("[SemImageFile]\nPixelSize=12.5\n") == pytest.approx(12.5e-9)


def test_fei_tiff_tag(tmp_path):
    img = (np.random.default_rng(0).random((64, 80)) * 255).astype(np.uint8)
    p = tmp_path / "fei.tif"
    tifffile.imwrite(p, img, extratags=[(34682, "s", 0, FEI_TEXT, True)])
    li = io.load_image(str(p))
    assert li.data.shape == (64, 80)
    assert li.calibration.unit == "nm"
    assert li.calibration.unit_per_px == pytest.approx(4.9609375)
    assert "34682" in li.calibration.source


def test_zeiss_tiff_tag(tmp_path):
    img = (np.random.default_rng(1).random((50, 60)) * 4000).astype(np.uint16)
    p = tmp_path / "zeiss.tif"
    tifffile.imwrite(p, img, extratags=[(34118, "s", 0, ZEISS_TEXT, True)])
    li = io.load_image(str(p))
    assert li.data.dtype == np.uint16
    assert li.calibration.unit == "nm"
    assert li.calibration.unit_per_px == pytest.approx(2.345)


def test_sidecar_and_resolution_fallback(tmp_path):
    from PIL import Image

    img = np.full((40, 50), 128, np.uint8)
    p = tmp_path / "jeol.bmp"
    Image.fromarray(img).save(p)
    (tmp_path / "jeol.txt").write_text("$$SM_MICRON_BAR 100\n$$SM_MICRON_MARKER 5um\n")
    li = io.load_image(str(p))
    assert li.calibration.unit == "nm" or li.calibration.unit == "µm"
    assert li.calibration.to_unit("µm").unit_per_px == pytest.approx(0.05)
    # ImageJ-style TIFF resolution
    q = tmp_path / "ij.tif"
    tifffile.imwrite(q, img, imagej=True, resolution=(4.0, 4.0), metadata={"unit": "micron"})
    lj = io.load_image(str(q))
    assert lj.calibration.unit == "µm"
    assert lj.calibration.px_per_unit == pytest.approx(4.0)
    # plain file without anything: uncalibrated
    r = tmp_path / "plain.png"
    Image.fromarray(img).save(r)
    assert not io.load_image(str(r)).calibration.calibrated


def test_multipage_and_rgb(tmp_path):
    stack = np.stack([np.full((20, 30), v, np.uint8) for v in (10, 20, 30)])
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, stack, photometric="minisblack")
    li = io.load_image(str(p), page=2)
    assert li.n_pages == 3 and li.page == 2 and int(li.data[0, 0]) == 30
    from PIL import Image

    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 0] = 200
    Image.fromarray(rgb).save(tmp_path / "c.png")
    assert io.load_image(str(tmp_path / "c.png")).data.shape == (10, 12, 3)
    Image.fromarray(np.dstack([rgb[..., 0]] * 3)).save(tmp_path / "g.png")
    assert io.load_image(str(tmp_path / "g.png")).data.ndim == 2  # grey stored as RGB → 2-D


def test_detect_data_bar():
    rng = np.random.default_rng(2)
    img = (rng.normal(128, 25, (400, 500))).clip(0, 255).astype(np.uint8)
    assert io.detect_data_bar(img) is None
    bar = np.zeros((48, 500), np.uint8)
    bar[12:30, 20:180] = 255          # "text"
    bar[12:30, 300:330] = 255
    full = np.vstack([img, bar])
    assert io.detect_data_bar(full) == 400
    white = np.full((40, 500), 255, np.uint8)
    white[10:25, 50:200] = 0
    assert io.detect_data_bar(np.vstack([img, white])) == 400


# ======================================================================================
# adjust / filters / background
# ======================================================================================

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_filters_keep_dtype(dtype):
    rng = np.random.default_rng(3)
    scale = 255 if dtype == np.uint8 else 4000 if dtype == np.uint16 else 1.0
    img = (rng.random((64, 64)) * scale).astype(dtype)
    for out in (io.gaussian_blur(img, 1.5), io.median_filter(img, 2), io.mean_filter(img, 2),
                io.unsharp_mask(img, 2, 0.5), io.sharpen(img), io.find_edges(img),
                io.fft_bandpass(img, 20, 2), io.gamma_correct(img, 0.8), io.invert(img),
                io.equalize_histogram(img), io.clahe(img, 0.02, 16), io.rotate(img, 10),
                io.flip(img), io.rotate90(img), io.subtract_background(img, 10)):
        assert out.dtype == img.dtype
    assert io.resize(img, 32, 16).shape == (16, 32)
    assert io.crop(img, 5, 6, 25, 16).shape == (10, 20)
    assert io.to_8bit(img).dtype == np.uint8
    assert io.median_filter(img, 2).shape == img.shape


def test_contrast_helpers():
    img = np.arange(10000, dtype=np.uint16).reshape(100, 100)
    lo, hi = io.auto_contrast_limits(img, 0.35)
    assert lo == pytest.approx(17.5, abs=1) and hi == pytest.approx(9981.5, abs=1)
    w = io.apply_window(img, 0, 9999)
    assert w.dtype == np.uint8 and w.min() == 0 and w.max() == 255
    assert io.invert(np.array([[0, 255]], np.uint8)).tolist() == [[255, 0]]
    a, b = io.brightness_contrast_window(0, 0, 0, 255)
    assert (a, b) == (0, 255)
    a, b = io.brightness_contrast_window(0, 0.25, 0, 255)
    assert b - a == pytest.approx(127.5)


def test_rolling_ball_removes_gradient():
    h, w = 200, 240
    yy, xx = np.mgrid[:h, :w]
    background = 40 + 0.5 * xx + 0.2 * yy               # strong illumination gradient
    feat = discs((h, w), ((60, 60), (170, 130), (100, 160)), 8)
    img = (background + 80 * feat).clip(0, 255).astype(np.uint8)
    out = io.subtract_background(img, radius=20)
    bg_region = ~io.binary_dilate(feat, 3)
    assert img[bg_region].std() > 30
    assert out[bg_region].mean() < 8                   # gradient gone (was ~120 ± 35)
    assert out[bg_region].std() < 2
    assert out[feat].mean() > 60                       # features kept
    # light background (bright-field LOM: dark features on bright, uneven field)
    img2 = (220 - 0.3 * xx - 80 * feat).clip(0, 255).astype(np.uint8)
    out2 = io.subtract_background(img2, radius=20, light_background=True)
    assert out2[bg_region].std() < 4
    assert out2[bg_region].mean() > 245
    assert out2[feat].mean() < 190


# ======================================================================================
# threshold & binary
# ======================================================================================

def bimodal(seed=4):
    rng = np.random.default_rng(seed)
    img = rng.normal(60, 10, (200, 200))
    m = discs((200, 200), ((60, 60), (140, 140), (60, 150)), 30)
    img[m] = rng.normal(180, 12, m.sum())
    return img.clip(0, 255).astype(np.uint8), m


@pytest.mark.parametrize("method", io.THRESHOLD_METHODS)
def test_threshold_methods(method):
    img, m = bimodal()
    t = io.threshold_value(img, method)
    assert 60 < t < 180
    lo, hi, mask = io.auto_threshold(img, method, dark_objects=False)
    assert hi == img.max()
    if method not in ("Mean", "Triangle", "Li", "MaxEntropy", "Yen"):
        assert (mask == m).mean() > 0.97


def test_threshold_errors_and_range():
    with pytest.raises(ValueError):
        io.threshold_value(np.zeros((10, 10)), "Otsu")
    with pytest.raises(ValueError):
        io.threshold_value(np.arange(100).reshape(10, 10), "Nope")
    img = np.arange(100).reshape(10, 10)
    assert io.threshold_range(img, 10, 19).sum() == 10


def test_binary_ops():
    m = discs((120, 160), ((50, 60), (90, 60)), 25)        # two overlapping discs
    assert ma.analyze_particles(m).count == 1
    sep = io.watershed_separate(m, 10)
    assert ma.analyze_particles(sep).count == 2
    ring = discs((80, 80), ((40, 40),), 30) & ~discs((80, 80), ((40, 40),), 12)
    assert io.fill_holes(ring).sum() > ring.sum()
    assert io.binary_erode(ring).sum() < ring.sum() < io.binary_dilate(ring).sum()
    noisy = ring.copy()
    noisy[2, 2] = noisy[5, 70] = True
    assert io.remove_small(noisy, 10).sum() == ring.sum()
    assert io.binary_open(noisy).sum() <= noisy.sum()
    assert io.binary_close(ring, 2, 1).sum() >= ring.sum()
    assert io.skeletonize(ring).sum() < ring.sum() / 5
    assert io.outline(ring).sum() < ring.sum() / 3


# ======================================================================================
# measurements
# ======================================================================================

def test_line_profile_length():
    img = np.tile(np.arange(100, dtype=float), (100, 1))      # value = x
    cal = Calibration(2.0, "µm")
    # ROI coordinates: pixel (i, j) centre is at (j + 0.5, i + 0.5) (ImageJ convention)
    d, v = io.line_profile(img, (10.5, 10.5), (40.5, 50.5), width=1, cal=cal)
    assert d[-1] == pytest.approx(25.0)                       # 50 px / 2 px per µm
    assert d[0] == 0 and len(d) >= 50
    assert v[0] == pytest.approx(10, abs=0.5) and v[-1] == pytest.approx(40, abs=0.5)
    d2, v2 = io.line_profile(img, (10.5, 50.5), (90.5, 50.5), width=5)
    assert np.allclose(v2, np.linspace(10, 90, len(v2)), atol=0.6)
    ml = io.measure_line(img, [(10, 10), (40, 50)], cal)
    assert ml["Length"] == pytest.approx(25.0)
    assert ml["Angle (°)"] == pytest.approx(-math.degrees(math.atan2(40, 30)))
    assert io.measure_line(img, [(0, 0), (30, 0), (30, 40)], cal)["Length"] == pytest.approx(35.0)


def test_polygon_measure_and_feret():
    img = np.zeros((100, 100), np.uint8)
    img[20:60, 10:50] = 100
    cal = Calibration(0.5, "mm")                               # 2 mm per px
    m = io.measure_polygon(img, io.rect_vertices(10, 20, 50, 60), cal)
    assert m["Area"] == pytest.approx(40 * 40 * 4)
    assert m["Perimeter"] == pytest.approx(160 * 2)
    assert m["Mean"] == pytest.approx(100)
    assert m["Feret"] == pytest.approx(math.sqrt(2) * 80)
    assert m["MinFeret"] == pytest.approx(80)
    assert (m["X"], m["Y"]) == (pytest.approx(60), pytest.approx(80))
    f = io.feret(np.array([[0, 0], [10, 0], [10, 2], [0, 2]], float))
    assert f["max"] == pytest.approx(math.hypot(10, 2)) and f["min"] == pytest.approx(2)
    assert io.angle_between((10, 0), (0, 0), (0, 10)) == pytest.approx(90)
    assert io.line_angle((0, 0), (10, -10)) == pytest.approx(45)
    ev = io.ellipse_vertices(0, 0, 40, 20, n=720)
    assert io.polygon_area(ev) == pytest.approx(math.pi * 20 * 10, rel=1e-3)


# ======================================================================================
# particle analysis
# ======================================================================================

def test_particles_discs():
    r = 25
    m = discs(r=r)
    cal = Calibration(2.0, "µm")                               # 0.5 µm per px
    res = ma.analyze_particles(m, cal, intensity=m.astype(np.uint8) * 200)
    assert res.count == 4
    for row in res.rows:
        assert row["Area"] == pytest.approx(math.pi * r ** 2 / 4, rel=0.02)
        assert row["ECD"] == pytest.approx(r, rel=0.02)        # 2r px × 0.5 µm/px
        assert 0.86 < row["Circ."] < 0.93                       # ImageJ traced perimeter (default)
        assert row["AR"] == pytest.approx(1.0, abs=0.03)
        assert row["Solidity"] > 0.95                           # area / pixel-outline hull (ImageJ)
        assert row["Feret"] == pytest.approx(r + 0.5, abs=1.0)
        assert row["MinFeret"] == pytest.approx(r + 0.5, abs=1.0)
        assert row["Mean"] == pytest.approx(200)
    for row in ma.analyze_particles(m, cal, perimeter="crofton").rows:
        assert row["Circ."] > 0.95                              # Crofton ≈ true perimeter
    s = res.summary
    assert s["Area fraction (%)"] == pytest.approx(100 * m.sum() / m.size, rel=1e-6)
    assert s["Total area"] == pytest.approx(cal.area(m.sum()))
    # size filter in calibrated units and exclude edges
    assert ma.analyze_particles(m, cal, min_area=1000).count == 0
    edge = m.copy()
    edge[:10, :10] = True
    assert ma.analyze_particles(edge, cal, exclude_edges=True).count == 4
    assert ma.analyze_particles(edge, cal, exclude_edges=False).count == 5
    # circularity filter removes a thin rectangle
    rect = m.copy()
    rect[280:285, 20:200] = True
    assert ma.analyze_particles(rect, cal, min_circ=0.5).count == 4


def test_particles_ellipse_and_holes():
    yy, xx = np.mgrid[:200, :200]
    u, v = xx - 100.0, yy - 100.0
    a = u * math.cos(math.pi / 4) - v * math.sin(math.pi / 4)   # axis pointing up-right
    b = u * math.sin(math.pi / 4) + v * math.cos(math.pi / 4)
    e = (a / 60) ** 2 + (b / 20) ** 2 <= 1
    row = ma.analyze_particles(e).rows[0]
    assert row["AR"] == pytest.approx(3.0, rel=0.03)
    assert row["Angle"] == pytest.approx(45, abs=1.0)
    assert row["Major"] == pytest.approx(120, rel=0.02)
    assert math.pi / 4 * row["Major"] * row["Minor"] == pytest.approx(row["Area"], rel=1e-6)
    ring = discs((100, 100), ((50, 50),), 30) & ~discs((100, 100), ((50, 50),), 15)
    a0 = ma.analyze_particles(ring).rows[0]["Area"]
    a1 = ma.analyze_particles(ring, include_holes=True).rows[0]["Area"]
    assert a1 == pytest.approx(math.pi * 30 ** 2, rel=0.02)
    assert a0 == pytest.approx(math.pi * (30 ** 2 - 15 ** 2), rel=0.03)


# ======================================================================================
# grain size (ASTM E112)
# ======================================================================================

def test_astm_formulae():
    # E112 Table: G = 7 ↔ l̄ = 0.0283 mm ↔ N_A = 992 /mm²
    assert ma.astm_g_from_intercept(0.0283) == pytest.approx(7.0, abs=0.01)
    assert ma.astm_g_from_na(992.0) == pytest.approx(7.0, abs=0.01)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_voronoi_planimetric_and_intercept(seed):
    shape, n = (512, 512), 100
    img, lab, b = ma.make_voronoi_grains(shape, n, seed=seed)
    cal = Calibration(1.0, "µm")
    pl = ma.planimetric(b, cal)
    assert pl.n_equivalent == pytest.approx(n, rel=0.10)
    assert pl.N_A == pytest.approx(n / (512 * 512), rel=0.10)
    assert pl.n_corner == 4
    lexp = expected_intercept(n, 512 * 512)
    for pattern in ("grid", "circles"):
        li = ma.linear_intercept(b, cal, n_lines=8, pattern=pattern)
        assert li.mean_intercept == pytest.approx(lexp, rel=0.12), pattern
        assert li.P_L == pytest.approx(li.intersections / li.total_length)
        assert li.G == pytest.approx(ma.astm_g_from_intercept(li.mean_intercept * 1e-3))
        assert len(li.hits) == round(li.intersections) or pattern == "circles"
    gd = ma.grain_area_distribution(b, cal)
    assert gd.areas.mean() == pytest.approx(512 * 512 / n, rel=0.15)


def test_grains_from_micrograph_and_units():
    img, lab, b = ma.make_voronoi_grains((600, 600), 150, seed=7, boundary_width=3, noise=10)
    cal = Calibration(1 / 0.2, "µm")                       # 0.2 µm per px
    bb = ma.boundaries_from_image(img)
    pl = ma.planimetric(bb, cal)
    assert pl.n_equivalent == pytest.approx(150, rel=0.10)
    field_mm2 = (600 * 0.2e-3) ** 2
    assert pl.N_A_mm2 == pytest.approx(pl.n_equivalent / field_mm2)
    assert pl.G == pytest.approx(ma.astm_g_from_na(pl.N_A_mm2))
    li = ma.linear_intercept(bb, cal, 10, "grid")
    assert li.mean_intercept == pytest.approx(expected_intercept(150, 120 ** 2), rel=0.12)
    assert li.mean_intercept_mm == pytest.approx(li.mean_intercept * 1e-3)
    # intercept and planimetric G agree within ~0.5 for equiaxed grains
    assert abs(li.G - pl.G) < 0.6
    # same image in nm gives the same G
    li_nm = ma.linear_intercept(bb, cal.to_unit("nm"), 10, "grid")
    assert li_nm.G == pytest.approx(li.G)
    assert li_nm.mean_intercept == pytest.approx(li.mean_intercept * 1000)
    # uncalibrated → no G
    assert ma.linear_intercept(bb, Calibration(), 4).G is None


# ======================================================================================
# phase fraction & histogram
# ======================================================================================

def test_phase_fraction_two_and_three_phases():
    rng = np.random.default_rng(5)
    img = np.full((200, 300), 70.0)
    img[:, :90] = 180                                         # 30 % bright phase
    img += rng.normal(0, 8, img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    r = ma.phase_fractions(img, 2, cal=Calibration(10, "µm"))
    assert r.fractions[1] == pytest.approx(30.0, abs=0.5)
    assert sum(r.fractions) == pytest.approx(100)
    assert r.areas[1] == pytest.approx(200 * 90 / 100, rel=0.02)
    img3 = np.zeros((100, 100), np.uint8)
    img3[:, :20] = 30
    img3[:, 20:50] = 120
    img3[:, 50:] = 220
    r3 = ma.phase_fractions(img3, 3)
    assert [round(f) for f in r3.fractions] == [20, 30, 50]
    rm = ma.phase_fractions(img3, thresholds=[200, 60])
    assert [round(f) for f in rm.fractions] == [20, 30, 50]
    roi = np.zeros_like(img3, bool)
    roi[:, :50] = True
    assert ma.phase_fractions(img3, thresholds=[60, 200], mask=roi).fractions[0] == pytest.approx(40)


def test_histogram_stats():
    img = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 4, axis=0)
    hs = ma.histogram_stats(img)
    assert hs.counts.sum() == img.size and len(hs.counts) == 256
    assert hs.mean == pytest.approx(127.5) and hs.min == 0 and hs.max == 255
    assert abs(hs.skewness) < 1e-9


def test_scale_bar_burn_in():
    img = np.full((300, 400), 100, np.uint8)
    cal = Calibration(4.0, "µm")
    assert io.auto_scale_bar_length(400, cal) == 20.0
    assert io.nice_number(73) == 50 and io.nice_number(0.034) == pytest.approx(0.02)
    out = io.burn_scale_bar(img, cal, 25, "lower right", "white")
    assert out.shape == img.shape
    assert (out == 255).sum() >= 100 * 3                      # 25 µm = 100 px bar
    assert (out[:150] == 100).all()                           # top half untouched
    rgb = io.burn_scale_bar(img, cal, None, "upper left", "yellow", background=False)
    assert rgb.ndim == 3 and (rgb[:150, :, 2] == 0).any()


# ======================================================================================
# GUI
# ======================================================================================

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def ws(qapp):
    from labstudio.gui.image_workspace import ImageWorkspace

    w = ImageWorkspace()
    w.resize(1400, 850)
    yield w
    w.deleteLater()


def test_workspace_api(ws):
    from labstudio.gui.image_workspace import IMAGE_EXTENSIONS, ImageWorkspace

    assert ".tif" in IMAGE_EXTENSIONS and ".png" in ImageWorkspace.IMAGE_EXTENSIONS
    assert callable(ws.open_image) and callable(ws.open_dialog)


def test_undo_redo_revert(ws):
    img = (np.arange(64 * 64) % 256).astype(np.uint8).reshape(64, 64)
    doc = ws.add_image(img, "ramp")
    ws.apply_op("Invert", io.invert)
    assert int(doc.data[0, 1]) == 254
    ws.undo()
    assert np.array_equal(doc.data, img)
    ws.redo()
    assert int(doc.data[0, 1]) == 254
    ws.apply_op("Rotate", lambda d: io.rotate90(d, 1))
    for i in range(30):
        ws.apply_op(f"Invert {i}", io.invert)
    assert 20 <= len(doc.undo_stack) <= 25
    for _ in range(22):
        ws.undo()
    assert len(doc.redo_stack) == 22
    ws.revert()
    assert np.array_equal(doc.data, img)
    ws.undo()                                                # revert itself is undoable
    assert not np.array_equal(doc.data, img)
    # mask operations are undoable too
    ws.revert()
    ws.set_mask(img > 128, "Threshold")
    ws.mask_op("Erode", io.binary_erode)
    n_eroded = doc.mask.sum()
    ws.undo()
    assert doc.mask.sum() > n_eroded
    ws.undo()
    assert doc.mask is None


def _write_disc_image(path, r=20):
    m = discs((300, 400), ((80, 80), (250, 90), (120, 220), (300, 210), (200, 160)), r)
    rng = np.random.default_rng(9)
    img = np.where(m, 200, 50) + rng.normal(0, 6, m.shape)
    tifffile.imwrite(path, img.clip(0, 255).astype(np.uint8),
                     extratags=[(34682, "s", 0, "[Scan]\nPixelWidth=5e-07\n", True)])
    return m


def test_gui_threshold_particles_export(ws, tmp_path):
    p = tmp_path / "particles.tif"
    m = _write_disc_image(str(p))
    doc = ws.open_image(str(p))
    assert doc.calibration.unit == "µm" and doc.calibration.unit_per_px == pytest.approx(0.5)
    thr = ws.panels["threshold"]
    thr.dark.setChecked(False)
    thr.method.setCurrentText("Otsu")
    lo, hi = thr.auto()
    assert 50 < lo < 200
    mask = thr.apply()
    assert (mask == m).mean() > 0.99
    assert doc.mask is not None
    part = ws.panels["particles"]
    res = part.run()
    assert res.count == 5
    assert len(ws.results.rows) == 5
    assert "Area (µm²)" in ws.results.columns
    assert ws.results.rows[0]["Area (µm²)"] == pytest.approx(math.pi * 400 * 0.25, rel=0.03)
    out = tmp_path / "results.csv"
    ws.export_results_csv(str(out))
    with open(out, encoding="utf-8-sig") as fh:
        text = fh.read().splitlines()
    assert text[0].startswith("# LabStudio")                 # provenance header
    rows = list(csv.DictReader([ln for ln in text if not ln.startswith("#")]))
    assert len(rows) == 5 and float(rows[0]["Circ."]) > 0.85
    assert any(s["Quantity"] == "Count" and s["Value"] == 5 for s in ws.summary.rows)
    # the size distribution went to the plot tab
    assert len(ws.plot.data) == 2 and ws.plot.data[1].sum() == 5


def test_gui_grain_size_roi_scale_export(ws, tmp_path):
    img, lab, b = ma.make_voronoi_grains((400, 500), 80, seed=3, boundary_width=2, noise=5)
    p = tmp_path / "grains.png"
    from PIL import Image

    Image.fromarray(img).save(p)
    doc = ws.open_image(str(p))
    assert not doc.calibration.calibrated
    # Set scale: a 250 px line = 50 µm
    from labstudio.gui.image_viewer import Roi

    ws._on_roi_created("scale", Roi("line", [(100, 300), (350, 300)]))
    assert ws.scale_px.value() == pytest.approx(250)
    ws.scale_dist.setValue(50)
    ws.scale_unit.setCurrentText("µm")
    ws.set_scale_from_fields()
    assert doc.calibration.px_per_unit == pytest.approx(5.0)
    g = ws.panels["grain"]
    g.run_all()
    assert g.last_planimetric.G is not None and g.last_intercept.G is not None
    assert g.last_planimetric.n_equivalent == pytest.approx(80, rel=0.15)
    assert abs(g.last_planimetric.G - g.last_intercept.G) < 0.7
    assert len(doc.shapes) > 0 and doc.overlay_rgba is not None
    # ROI measure (rectangle 100×50 px = 20×10 µm)
    ws.add_roi(Roi("rect", [(10, 10), (110, 60)]))
    row = ws.measure_roi()[0]
    assert row["Area (µm²)"] == pytest.approx(200.0)
    ws.add_roi(Roi("point", [(5, 5), (20, 20), (30, 40)]))
    assert len(ws.measure_roi()) == 3
    ws.add_roi(Roi("line", [(0, 0), (30, 40)]))
    x, y = ws.plot_profile()
    assert x[-1] == pytest.approx(10.0)                  # 50 px / 5 px/µm
    # phase fraction + histogram + scale bar
    assert ws.panels["phase"].run() is not None
    assert ws.panels["histogram"].run().n == img.size
    ann = ws.panels["annotate"]
    ann.sb_show.setChecked(True)
    assert doc.scale_bar is not None
    ann.place_text(50, 50)
    ann.place_arrow((10, 10), (60, 60))
    assert len(doc.annotations) == 2
    # exports
    ws.export_image(str(tmp_path / "ov.png"), with_overlays=True)
    ws.export_image(str(tmp_path / "raw.tif"), with_overlays=False)
    ov = np.array(Image.open(tmp_path / "ov.png"))
    assert ov.shape[:2] == img.shape and ov.ndim == 3
    raw = io.load_image(str(tmp_path / "raw.tif"))
    assert np.array_equal(raw.data, img)
    assert raw.calibration.px_per_unit == pytest.approx(5.0) and raw.calibration.unit == "µm"
    # burn in scale bar changes the data and is undoable
    before = doc.data.copy()
    ann.burn_in()
    assert not np.array_equal(doc.data, before)
    ws.undo()
    assert np.array_equal(doc.data, before)


def test_gui_data_bar_crop_and_process(ws, tmp_path):
    rng = np.random.default_rng(1)
    img = rng.normal(120, 20, (300, 300)).clip(0, 255).astype(np.uint8)
    bar = np.zeros((40, 300), np.uint8)
    bar[10:30, 20:150] = 255
    p = tmp_path / "sem.tif"
    tifffile.imwrite(p, np.vstack([img, bar]))
    doc = ws.open_image(str(p))
    assert not ws.banner.isHidden()
    ws._banner_accept()
    assert doc.shape == (300, 300)
    proc = ws.panels["process"]
    proc.subtract_background()
    proc.rs_w.setValue(150)
    doc.calibration = Calibration(2.0, "µm")
    proc.resize()
    assert doc.shape == (150, 150) and doc.calibration.px_per_unit == pytest.approx(1.0)
    from labstudio.gui.image_viewer import Roi

    ws.add_roi(Roi("rect", [(10, 20), (60, 50)]))
    proc.crop_to_roi()
    assert doc.shape == (30, 50)
    adj = ws.panels["adjust"]
    adj.auto()
    adj.apply_lut()
    assert doc.data.dtype == np.uint8
    ws.close_image(0)
    assert ws.doc is None


def test_viewer_mouse_tools(ws, qapp):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    ws.add_image(np.full((200, 300), 120, np.uint8), "flat", Calibration(1.0, "µm"))
    ws.show()
    qapp.processEvents()
    v = ws.viewer
    v.zoom_100()
    vp = v.viewport()

    def at(x, y):
        return v.mapFromScene(x, y)

    ws.set_tool("rect")
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, at(20, 30))
    QTest.mouseMove(vp, at(70, 80))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, at(70, 80))
    roi = ws.doc.rois[-1]
    assert roi.kind == "rect"
    assert roi.bounds() == pytest.approx((20, 30, 70, 80), abs=1.0)
    ws.set_tool("polygon")
    for x, y in ((100, 20), (180, 20), (180, 100)):
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, at(x, y))
    v.finish_current()
    assert ws.doc.rois[-1].kind == "polygon" and len(ws.doc.rois[-1].points) >= 3
    ws.set_tool("point")
    for x, y in ((10, 10), (20, 20)):
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, at(x, y))
    v.finish_current()
    assert ws.doc.rois[-1].kind == "point" and len(ws.doc.rois[-1].points) == 2
    ws.set_tool("profile")
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, at(10, 100))
    QTest.mouseMove(vp, at(110, 100))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, at(110, 100))
    assert ws.plot.data[0][-1] == pytest.approx(100, abs=2)
    assert ws.roi_list.count() == 4
    ws.delete_current_roi()
    assert len(ws.doc.rois) == 3
    ws.hide()
