"""Quantitative metallography on micrographs (pure numpy / scipy / scikit-image, no Qt).

* :func:`analyze_particles` — ImageJ 'Analyze Particles' equivalents (area, perimeter, ECD,
  fitted-ellipse major/minor, aspect ratio, circularity 4πA/P², solidity, Feret max/min, angle,
  centroid) with size/circularity filters, exclude-edges and include-holes options.
* :func:`linear_intercept` — ASTM E112 / ISO 643 mean lineal intercept with straight test lines
  (horizontal/vertical/grid) or three concentric circles (Abrams).
* :func:`planimetric` — ASTM E112 Jeffries planimetric grain count on a rectangular field.
* :func:`grain_area_distribution` — per-grain areas from segmented grains.
* :func:`phase_fractions` — area fractions by thresholds or multi-level Otsu.
* :func:`histogram_stats` — intensity histogram and statistics.

All lengths are returned in the calibration unit (nm/µm/mm) and areas in unit²; the ASTM
grain-size numbers are computed from mm-based quantities as required by the standard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .image_ops import Calibration, LENGTH_UNITS, feret, to_gray

# --------------------------------------------------------------------------------------
# Particle analysis
# --------------------------------------------------------------------------------------

PARTICLE_COLUMNS = ["Label", "Area", "Perimeter", "ECD", "Major", "Minor", "AR", "Circ.",
                    "Round", "Solidity", "Feret", "MinFeret", "FeretAngle", "Angle", "X", "Y",
                    "Mean"]

#: Perimeter estimators offered for particle analysis (GUI label → key).
PERIMETER_METHODS = {"ImageJ (traced polygon)": "imagej", "Crofton (unbiased)": "crofton"}

#: ASTM E112 guidance: a field / test pattern should give at least ~50 counts.
E112_MIN_COUNTS = 50


@dataclass
class ParticleResult:
    """Output of :func:`analyze_particles`."""

    rows: list[dict]                  # one dict per particle (keys = PARTICLE_COLUMNS)
    labels: np.ndarray                # int32 label image of the *kept* particles (0 = none)
    contours: list[np.ndarray]        # traced outer outline (N×2 x/y, pixel-corner coords) per particle
    summary: dict                     # count, total area, area fraction %, mean size ...
    unit: str = "px"
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    n_touching_excluded: int = 0      # particles touching an excluded region (scale bar, labels…)

    @property
    def count(self) -> int:
        return len(self.rows)

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.rows], dtype=float)


def _outer_contour(region_mask: np.ndarray) -> np.ndarray:
    """Longest marching-squares contour (x, y) of a padded binary crop."""
    from skimage.measure import find_contours

    padded = np.pad(region_mask.astype(float), 1)
    cs = find_contours(padded, 0.5, fully_connected="high")
    if not cs:
        return np.zeros((0, 2))
    c = max(cs, key=len)
    return np.column_stack([c[:, 1] - 1, c[:, 0] - 1])


def _contour_length(c: np.ndarray) -> float:
    if len(c) < 2:
        return 0.0
    d = np.diff(c, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


_STEP_DX = (1, 0, -1, 0)   # direction 0:+x, 1:+y (down), 2:−x, 3:−y — clockwise on screen
_STEP_DY = (0, 1, 0, -1)


def traced_outline(mask: np.ndarray) -> np.ndarray:
    """Pixel-edge outline of the outer boundary of a single 8-connected blob, as ImageJ's Wand
    traces it: vertices (x, y) in pixel-corner coordinates at every change of direction, clockwise
    on screen, starting at the top-left corner of the topmost pixel of the leftmost column.
    Holes are ignored (outer boundary only); diagonal pixel contacts are followed (8-connected).
    """
    m = ndi.binary_fill_holes(np.pad(np.asarray(mask, bool), 1))
    H, W = m.shape
    if not m.any():
        return np.zeros((0, 2))
    W1 = W + 1
    up = np.zeros_like(m); up[1:] = m[:-1]
    dn = np.zeros_like(m); dn[:-1] = m[1:]
    lf = np.zeros_like(m); lf[:, 1:] = m[:, :-1]
    rt = np.zeros_like(m); rt[:, :-1] = m[:, 1:]
    starts, dirs = [], []
    for d, sel, (oy, ox) in ((0, m & ~up, (0, 0)), (1, m & ~rt, (0, 1)),
                             (2, m & ~dn, (1, 1)), (3, m & ~lf, (1, 0))):
        r, c = np.nonzero(sel)
        starts.append((r + oy) * W1 + (c + ox))
        dirs.append(np.full(len(r), d, np.int8))
    st = np.concatenate(starts)
    dr = np.concatenate(dirs)
    order = np.argsort(st, kind="stable")
    st, dr = st[order], dr[order]
    dup = np.zeros(len(st), bool)
    dup[1:] = st[1:] == st[:-1]
    single = dict(zip(st[~dup].tolist(), dr[~dup].tolist()))
    second = dict(zip(st[dup].tolist(), dr[dup].tolist()))
    # start: leftmost column, topmost pixel → its top-left corner (always a corner, unambiguous)
    xs = st % W1
    cand = st[xs == xs.min()]
    v = int(cand.min())
    start = v
    n_edges = len(st)
    seq = []
    d_in = 3
    steps = (1, W1, -1, -W1)
    for _ in range(n_edges + 4):
        d = single[v]
        d2 = second.get(v)
        if d2 is not None:  # diagonal contact: take the left turn to stay 8-connected
            left = (d_in - 1) % 4
            d = left if left in (d, d2) else d
        seq.append(d)
        v += steps[d]
        d_in = d
        if v == start:
            break
    seq = np.asarray(seq, np.int8)
    # vertices = positions where the direction changes (seq[0] starts at the start corner)
    chg = np.flatnonzero(seq != np.roll(seq, 1))
    x0, y0 = start % W1, start // W1
    dx = np.array(_STEP_DX)[seq]
    dy = np.array(_STEP_DY)[seq]
    px = x0 + np.concatenate([[0], np.cumsum(dx)[:-1]])
    py = y0 + np.concatenate([[0], np.cumsum(dy)[:-1]])
    return np.column_stack([px[chg] - 1, py[chg] - 1]).astype(float)


def imagej_traced_perimeter(xp, yp) -> float:
    """ImageJ ``PolygonRoi.getTracedPerimeter()`` for a traced (Wand) outline.

    Each pixel edge counts 1; at every 'corner' the two unit edges are replaced by the diagonal,
    i.e. P = Σ|dx| + Σ|dy| − n_corners·(2 − √2). A vertex counts as a corner unless the side
    leading into it has length 1 and the previous vertex was a corner (so staircase runs of unit
    steps alternate). Ported from ImageJ 1.54 (``ij/gui/PolygonRoi.java``).
    """
    xp = np.asarray(xp, float)
    yp = np.asarray(yp, float)
    n = len(xp)
    if n == 0:
        return 0.0
    dxs = np.abs(np.diff(np.concatenate([xp[-1:], xp])))   # side ending at vertex i
    dys = np.abs(np.diff(np.concatenate([yp[-1:], yp])))
    side = (dxs + dys).tolist()
    corner = False
    n_corners = 0
    for s1 in side:
        if s1 > 1 or not corner:
            corner = True
            n_corners += 1
        else:
            corner = False
    return float(dxs.sum() + dys.sum() - n_corners * (2.0 - math.sqrt(2.0)))


def perimeter_imagej(mask: np.ndarray) -> float:
    """ImageJ 'Analyze Particles' perimeter (traced outer outline, corner-corrected) of a blob."""
    out = traced_outline(mask)
    if len(out) == 0:
        return 0.0
    return imagej_traced_perimeter(out[:, 0], out[:, 1])


def perimeter_crofton(mask: np.ndarray) -> float:
    """Crofton (4-direction) perimeter of the hole-filled blob: unbiased for smooth curved outlines
    (a digital disc of radius r gives ≈ 2πr), unlike ImageJ's traced perimeter."""
    from skimage.measure import perimeter_crofton as _pc

    crop = np.asarray(mask, bool)
    p = float(_pc(ndi.binary_fill_holes(np.pad(crop, 1)), directions=4))
    if crop.sum() < 9 or p <= 0:
        p = max(_contour_length(_outer_contour(crop)), 2 * math.sqrt(2))
    return p


def _pixel_corner_hull(coords_rc: np.ndarray) -> tuple[np.ndarray, float]:
    """Convex hull (x, y) of the pixel squares of a pixel set, and its area (px²)."""
    pts = coords_rc[:, ::-1].astype(float)
    offs = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    corners = (pts[:, None, :] + offs[None, :, :]).reshape(-1, 2)
    if len(corners) >= 3:
        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = ConvexHull(corners)
            return corners[hull.vertices], float(hull.volume)
        except (QhullError, ValueError):
            pass
    return corners, float(len(pts))


def _polygon_hull(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Convex hull vertices and area of a polygon's vertices (x, y)."""
    if len(pts) >= 3:
        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = ConvexHull(pts)
            return pts[hull.vertices], float(hull.volume)
        except (QhullError, ValueError):
            pass
    return pts, 0.0


def _pixel_corner_hull_points(coords_rc: np.ndarray) -> np.ndarray:
    """Corner points of pixel squares on the convex hull of a pixel set (x, y)."""
    return _pixel_corner_hull(coords_rc)[0]


def _boundary_coords(crop: np.ndarray, r0: int, c0: int) -> np.ndarray:
    """(row, col) of the boundary pixels of a crop (enough for convex hulls / Feret)."""
    edge = crop & ~ndi.binary_erosion(crop, structure=np.ones((3, 3), bool), border_value=0)
    r, c = np.nonzero(edge)
    return np.column_stack([r + r0, c + c0])


def analyze_particles(mask: np.ndarray, cal: Calibration | None = None,
                      intensity: np.ndarray | None = None,
                      min_area: float = 0.0, max_area: float = math.inf,
                      min_circ: float = 0.0, max_circ: float = 1.0,
                      exclude_edges: bool = False, include_holes: bool = False,
                      connectivity: int = 8, perimeter: str = "imagej",
                      exclude_regions: np.ndarray | None = None) -> ParticleResult:
    """Measure connected particles of a binary ``mask`` (True = particle), like ImageJ.

    Parameters
    ----------
    min_area, max_area : particle size range in calibrated unit² (px² if uncalibrated).
    min_circ, max_circ : circularity range (0–1).
    exclude_edges : drop particles touching the image border (and, if ``exclude_regions`` is
        given, particles touching an excluded region — they are truncated the same way).
    include_holes : interior holes are filled and counted as particle area (ImageJ 'Include holes').
    perimeter : ``"imagej"`` (default) — ImageJ's traced-polygon perimeter
        (:func:`imagej_traced_perimeter`), so circularity and circularity filters match ImageJ and
        the literature that uses it (a large digital disc gives Circ ≈ 0.90, a square ≈ 0.80–0.83);
        ``"crofton"`` — the Crofton estimator, closer to the true perimeter of smooth outlines
        (discs → Circ ≈ 1.0) but *not* what ImageJ reports.
    exclude_regions : boolean mask of pixels that are not part of the specimen (burned-in scale
        bar, label boxes, SEM data bar). Mask pixels there are ignored and the analysed area (for
        area fraction / number density) excludes them.

    Definitions (ImageJ): Area = pixel count; Perimeter as above (outer boundary, holes excluded);
    circularity = 4π·area/perimeter² (capped at 1); AR = major/minor of the best-fit ellipse with
    the same area and second moments; round = 4·area/(π·major²); solidity = area / area of the
    convex hull of the pixel outline; Feret = maximum caliper, MinFeret = minimum caliper (pixel
    corners); Angle = major-axis angle (0–180°, counter-clockwise from +x); ECD = √(4·area/π).
    """
    from skimage.measure import regionprops

    cal = cal or Calibration()
    pkey = PERIMETER_METHODS.get(perimeter, perimeter).lower()
    if pkey not in ("imagej", "crofton"):
        raise ValueError(f"Unknown perimeter method '{perimeter}'")
    m = mask.astype(bool)
    excl = None
    if exclude_regions is not None and np.any(exclude_regions):
        excl = np.asarray(exclude_regions, bool)
        m = m & ~excl
    if include_holes:
        m = ndi.binary_fill_holes(m)
    struct = np.ones((3, 3)) if connectivity == 8 else ndi.generate_binary_structure(2, 1)
    lab, n = ndi.label(m, structure=struct)
    h, w = m.shape
    touching = np.zeros(n + 1, bool)
    if excl is not None and n:
        ring = ndi.binary_dilation(excl, structure=np.ones((3, 3), bool)) & ~excl
        touching[np.unique(lab[ring])] = True
        touching[0] = False
    rows: list[dict] = []
    contours: list[np.ndarray] = []
    lut = np.zeros(n + 1, np.int32)
    gray = to_gray(intensity) if intensity is not None else None
    props = regionprops(lab, intensity_image=gray, cache=True)
    k = 0
    n_touch_kept = n_touch_dropped = 0
    for rp in props:
        r0, c0, r1, c1 = rp.bbox
        if exclude_edges and (r0 == 0 or c0 == 0 or r1 == h or c1 == w):
            continue
        if touching[rp.label] and exclude_edges:
            n_touch_dropped += 1
            continue
        a_px = float(rp.area)
        area = cal.area(a_px)
        if area < min_area or area > max_area:
            continue
        crop = rp.image
        outline = traced_outline(crop)          # pixel-corner polygon (crop coordinates)
        if pkey == "imagej":
            perim_px = imagej_traced_perimeter(outline[:, 0], outline[:, 1]) if len(outline) else 0.0
        else:
            perim_px = perimeter_crofton(crop)
        circ = min(1.0, 4 * math.pi * a_px / perim_px ** 2) if perim_px > 0 else 1.0
        if circ < min_circ or circ > max_circ:
            continue
        k += 1
        if touching[rp.label]:
            n_touch_kept += 1
        # Best-fit ellipse normalised to the particle area (ImageJ)
        maj_raw, min_raw = float(rp.axis_major_length), float(rp.axis_minor_length)
        if maj_raw <= 0:
            maj_raw = min_raw = 1.0
        ratio = maj_raw / min_raw if min_raw > 0 else np.inf
        if np.isfinite(ratio):
            minor = math.sqrt(4 * a_px / (math.pi * ratio))
            major = minor * ratio
        else:
            major, minor = maj_raw, 0.0
        angle = (math.degrees(rp.orientation) + 90.0) % 180.0
        hull_pts, hull_area = _polygon_hull(outline + np.array([c0, r0], float))
        fr = feret(hull_pts)
        cy, cx = rp.centroid
        cx, cy = cx + 0.5, cy + 0.5  # ImageJ corner convention: pixel centres at +0.5
        row = {
            "Label": k,
            "Area": area,
            "Perimeter": cal.length(perim_px),
            "ECD": cal.length(math.sqrt(4 * a_px / math.pi)),
            "Major": cal.length(major),
            "Minor": cal.length(minor),
            "AR": major / minor if minor > 0 else np.inf,
            "Circ.": circ,
            "Round": 4 * a_px / (math.pi * major ** 2) if major > 0 else 1.0,
            "Solidity": min(1.0, a_px / hull_area) if hull_area > 0 else 1.0,
            "Feret": cal.length(fr["max"]),
            "MinFeret": cal.length(fr["min"]),
            "FeretAngle": fr["angle"],
            "Angle": angle,
            "X": cal.length(cx),
            "Y": cal.length(cy),
            "Mean": float(rp.intensity_mean) if gray is not None else np.nan,
        }
        rows.append(row)
        lut[rp.label] = k
        contours.append(outline + np.array([c0, r0], float))
    kept = lut[lab]
    del lab
    total_area = float(sum(r["Area"] for r in rows))
    n_excl_px = int(excl.sum()) if excl is not None else 0
    img_area = cal.area(float(h * w - n_excl_px))
    summary = {
        "Count": len(rows),
        "Total area": total_area,
        "Average size": total_area / len(rows) if rows else 0.0,
        "Area fraction (%)": 100.0 * total_area / img_area if img_area else 0.0,
        "Mean ECD": float(np.mean([r["ECD"] for r in rows])) if rows else 0.0,
        "Mean circularity": float(np.mean([r["Circ."] for r in rows])) if rows else 0.0,
        "Mean AR": float(np.mean([r["AR"] for r in rows])) if rows else 0.0,
        "Number density (1/unit²)": len(rows) / img_area if img_area else 0.0,
        "Analysed area": img_area,
    }
    warnings: list[str] = []
    if not cal.calibrated:
        warnings.append("Image is uncalibrated: sizes are in pixels (px, px²). Set the scale from "
                        "the scale bar for physical units.")
    elif cal.from_print_resolution:
        warnings.append("Calibration comes from a print resolution (dpi), not from the microscope — "
                        "sizes are probably wrong.")
    if n_touch_kept:
        warnings.append(f"{n_touch_kept} particle(s) touch an excluded region (scale bar / label / "
                        "data bar) and may be truncated; tick 'Exclude on edges' to drop them.")
    if n_touch_dropped:
        warnings.append(f"{n_touch_dropped} particle(s) touching an excluded region were dropped "
                        "(treated like edge particles).")
    if not rows:
        warnings.append("No particles found with the current size / circularity limits.")
    params = {"min_area": min_area, "max_area": max_area, "min_circ": min_circ,
              "max_circ": max_circ, "exclude_edges": exclude_edges, "include_holes": include_holes,
              "connectivity": connectivity, "perimeter": pkey,
              "excluded_px": n_excl_px}
    return ParticleResult(rows=rows, labels=kept, contours=contours, summary=summary, unit=cal.unit,
                          warnings=warnings, params=params,
                          n_touching_excluded=n_touch_kept + n_touch_dropped)


# --------------------------------------------------------------------------------------
# Grain boundaries / segmentation helpers
# --------------------------------------------------------------------------------------


def _running_extreme(f: np.ndarray, size: int, axis: int, op) -> np.ndarray:
    """1-D running max/min over a centred window of odd ``size`` (edge-replicated borders), by
    log-doubling of shifted views — several times faster than ``ndi.maximum_filter1d`` on big
    images and identical in result."""
    r = size // 2
    if r <= 0:
        return f.copy()
    pad = [(0, 0)] * f.ndim
    pad[axis] = (r, r)
    out = np.pad(f, pad, mode="edge")
    n = f.shape[axis]
    span = 1
    L = out.shape[axis]
    while span * 2 <= size:
        dst = [slice(None)] * f.ndim
        src = [slice(None)] * f.ndim
        dst[axis] = slice(0, L - span)
        src[axis] = slice(span, L)
        tmp = out[tuple(dst)]
        op(tmp, out[tuple(src)], out=tmp)
        span *= 2
    a = [slice(None)] * f.ndim
    b = [slice(None)] * f.ndim
    a[axis] = slice(0, n)
    b[axis] = slice(size - span, size - span + n)
    return op(out[tuple(a)], out[tuple(b)])


def _disk_rectangles(radius: int) -> list[tuple[int, int]]:
    """Exact decomposition of skimage ``disk(radius)`` into centred rectangles (half-width, half-height)."""
    from skimage.morphology import disk

    fp = disk(radius)
    r = fp.shape[0] // 2
    hw = [int(np.flatnonzero(fp[r + dy])[-1] - r) for dy in range(r + 1)]
    return [(hw[dy], dy) for dy in range(r + 1) if dy == r or hw[dy + 1] < hw[dy]]


def _morph_disk(f: np.ndarray, radius: int, dilate: bool) -> np.ndarray:
    """Grey dilation / erosion by a disk = max / min over its rectangle decomposition."""
    op = np.maximum if dilate else np.minimum
    best = None
    for a, b in _disk_rectangles(radius):
        t = _running_extreme(_running_extreme(f, 2 * a + 1, 1, op), 2 * b + 1, 0, op)
        best = t if best is None else op(best, t, out=best)
    return best


def grey_closing_disk(f: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing with a disk of ``radius`` px (fast, exact away from the border)."""
    return _morph_disk(_morph_disk(f, radius, True), radius, False)


def grey_opening_disk(f: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening with a disk of ``radius`` px (fast, exact away from the border)."""
    return _morph_disk(_morph_disk(f, radius, False), radius, True)


def boundaries_from_image(img: np.ndarray, dark_boundaries: bool = True, method: str = "Otsu",
                          sigma: float = 1.0, tophat_radius: float = 6.0, thin: bool = True,
                          min_size: int = 20) -> np.ndarray:
    """Binary grain-boundary map from an etched micrograph.

    Smooths with a Gaussian (``sigma`` px), enhances thin dark (or bright) lines with a
    morphological black (white) top-hat of radius ``tophat_radius`` px — which removes the
    grain-to-grain shading of etched/EBSD-like contrast — thresholds the result with ``method``,
    removes specks smaller than ``min_size`` px and optionally thins to 1-px lines.
    ``tophat_radius`` should exceed the boundary width; 0 thresholds the raw intensities.
    Works on an 8-bit (8-bit input) or 16-bit quantised copy of the smoothed image to keep
    big images in memory.
    """
    from .image_ops import remove_small, threshold_value

    g0 = to_gray(img)
    eight = g0.dtype == np.uint8
    g = g0.astype(np.float32)
    del g0
    if sigma > 0:
        g = ndi.gaussian_filter(g, sigma)
    lo, hi = float(g.min()), float(g.max())
    if hi <= lo:
        raise ValueError("Image has a single intensity; cannot find boundaries")
    qmax = 255.0 if eight else 65535.0   # 8-bit input → 8-bit working copy (half the memory)
    q = np.empty(g.shape, np.uint8 if eight else np.uint16)
    np.multiply(g - lo, qmax / (hi - lo), out=g)
    np.rint(g, out=g)
    q[...] = g
    del g
    if tophat_radius and tophat_radius > 0:
        r = int(round(tophat_radius))
        if dark_boundaries:
            enh = grey_closing_disk(q, r)
            enh -= q            # closing ≥ image: no underflow
        else:
            enh = grey_opening_disk(q, r)
            np.subtract(q, enh, out=enh)
        t = threshold_value(enh, method)
        b = enh > t
        del enh
    else:
        t = threshold_value(q, method)
        b = q <= t if dark_boundaries else q > t
    del q
    if min_size > 0:
        b = remove_small(b, min_size)
    if thin:
        b = thin_boundaries(b)
    return b


def thin_boundaries(boundary: np.ndarray) -> np.ndarray:
    """Skeletonise a boundary map to 1-px lines without shortening lines that reach the image edge
    (the map is framed before thinning so boundary ends stay attached to the border)."""
    from skimage.morphology import skeletonize

    p = 4
    framed = np.pad(boundary.astype(bool), p, constant_values=True)
    return skeletonize(framed)[p:-p, p:-p]


def grains_from_boundaries(boundary: np.ndarray, min_grain_px: int = 5,
                           fill_boundaries: bool = False) -> np.ndarray:
    """Label grains = 4-connected regions not on a boundary; tiny regions are dropped.

    With ``fill_boundaries`` the boundary pixels are assigned to the nearest grain so the grains
    tile the image (useful for area distributions).
    """
    lab, n = ndi.label(~boundary.astype(bool), structure=ndi.generate_binary_structure(2, 1))
    lab = lab.astype(np.int32, copy=False)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_grain_px
        keep[0] = False
        lut = np.zeros(len(sizes), np.int32)
        lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
        lab = lut[lab]
    if fill_boundaries:
        lab = fill_label_gaps(lab)
    return lab


def fill_label_gaps(lab: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """Assign unlabelled (0) pixels to an adjacent label (largest label among the 8 neighbours),
    repeatedly, so grains tile the image (boundary pixels are split between neighbouring grains).
    Works only on the unlabelled pixels (gather of their neighbours), so it is fast and
    memory-light on big images — replacement for ``skimage.segmentation.expand_labels``."""
    lab = np.array(lab, dtype=np.int32, copy=True)
    if not lab.any():
        return lab
    h, w = lab.shape
    p = np.pad(lab, 1)
    W = w + 2
    flat = p.ravel()
    rr, cc = np.nonzero(lab == 0)
    idx = (rr + 1) * W + (cc + 1)
    offs = np.array([-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1])
    for _ in range(max_iter):
        if idx.size == 0:
            break
        best = np.zeros(idx.size, np.int32)
        for o in offs:
            np.maximum(best, flat[idx + o], out=best)
        hit = best > 0
        if not hit.any():
            break
        flat[idx[hit]] = best[hit]
        idx = idx[~hit]
    return p[1:-1, 1:-1].copy()


def _relabel(lab: np.ndarray):
    """Sequential relabelling (0 stays 0) → (labels, None, None); LUT based, no copies of the
    label image besides the output (replacement for skimage ``relabel_sequential``)."""
    ids = np.unique(lab)
    ids = ids[ids > 0]
    lut = np.zeros(int(lab.max()) + 1 if lab.size else 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[lab], None, None


def make_voronoi_grains(shape: tuple[int, int] = (512, 512), n_grains: int = 100, seed: int = 0,
                        boundary_width: int = 1, noise: float = 0.0, contrast: bool = True,
                        pad: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic equiaxed grain structure (Poisson–Voronoi tessellation).

    ``n_grains`` seeds are placed uniformly inside the field; seeds with the same density are also
    placed in a margin of ``pad``×size around it, so the field is an unbiased window into an
    infinite tessellation with N_A = n_grains / field area (no enlarged edge cells).
    Returns (image uint8 with dark boundaries and grain-to-grain shading, labels, boundary mask).
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    h, w = shape
    pts = np.column_stack([rng.uniform(0, w, n_grains), rng.uniform(0, h, n_grains)])
    if pad > 0:
        px, py = pad * w, pad * h
        big = (w + 2 * px) * (h + 2 * py)
        n_out = rng.poisson(n_grains * (big - w * h) / (w * h))
        cand = np.column_stack([rng.uniform(-px, w + px, n_out * 3), rng.uniform(-py, h + py, n_out * 3)])
        outside = ~((cand[:, 0] >= 0) & (cand[:, 0] < w) & (cand[:, 1] >= 0) & (cand[:, 1] < h))
        pts = np.vstack([pts, cand[outside][:n_out]])
    yy, xx = np.mgrid[0:h, 0:w]
    _, idx = cKDTree(pts).query(np.column_stack([xx.ravel(), yy.ravel()]))
    lab = idx.reshape(h, w).astype(np.int32) + 1
    lab, _, _ = _relabel(lab)
    b = np.zeros((h, w), bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    if boundary_width > 1:
        b = ndi.binary_dilation(b, iterations=boundary_width - 1)
    shade = rng.uniform(150, 230, len(pts) + 1) if contrast else np.full(len(pts) + 1, 200.0)
    img = shade[lab]
    img[b] = 40
    if noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), lab, b


# --------------------------------------------------------------------------------------
# ASTM E112 — linear intercept
# --------------------------------------------------------------------------------------


@dataclass
class InterceptResult:
    pattern: str
    n_lines: int
    total_length: float              # calibrated unit
    intersections: float             # P (triple points counted 1.5 when enabled)
    n_triple: int
    P_L: float                       # intersections per unit length (1/unit)
    mean_intercept: float            # l̄ (unit)
    unit: str
    mean_intercept_mm: float | None
    G: float | None                  # ASTM grain size number (needs calibration)
    lines: list[np.ndarray] = field(default_factory=list)   # test lines (N×2 x/y px)
    hits: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # intersection points px
    per_line: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def rows(self) -> list[dict]:
        u = self.unit
        out = [
            {"Quantity": "Test pattern", "Value": self.pattern, "Unit": ""},
            {"Quantity": "Number of test lines", "Value": self.n_lines, "Unit": ""},
            {"Quantity": "Total test line length L_T", "Value": self.total_length, "Unit": u},
            {"Quantity": "Boundary intersections P", "Value": self.intersections, "Unit": ""},
            {"Quantity": "Triple-point hits (×1.5)", "Value": self.n_triple, "Unit": ""},
            {"Quantity": "P_L = P / L_T", "Value": self.P_L, "Unit": f"1/{u}"},
            {"Quantity": "Mean lineal intercept l̄", "Value": self.mean_intercept, "Unit": u},
        ]
        if self.G is not None:
            out.append({"Quantity": "ASTM E112 grain size G (intercept)", "Value": self.G, "Unit": ""})
        return out


def _path_pixels(xs: np.ndarray, ys: np.ndarray, shape) -> np.ndarray:
    """Rasterise a densely sampled path to a 4-connected pixel sequence (rows of (x, y))."""
    h, w = shape
    xi = np.clip(np.rint(xs).astype(int), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, h - 1)
    keep = np.ones(len(xi), bool)
    keep[1:] = (xi[1:] != xi[:-1]) | (yi[1:] != yi[:-1])
    xi, yi = xi[keep], yi[keep]
    out_x, out_y = [xi[0]], [yi[0]]
    for x, y in zip(xi[1:], yi[1:]):
        px, py = out_x[-1], out_y[-1]
        if x != px and y != py:  # diagonal step → insert an intermediate pixel (4-connected)
            out_x.append(x)
            out_y.append(py)
        out_x.append(x)
        out_y.append(y)
    return np.column_stack([out_x, out_y])


def _sample_segment(p0, p1, shape):
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(2, int(L * 4) + 1)
    t = np.linspace(0, 1, n)
    return _path_pixels(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), shape), L


def _sample_circle(cx, cy, r, shape):
    n = max(16, int(2 * math.pi * r * 4))
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    xs, ys = cx + r * np.cos(t), cy + r * np.sin(t)
    xs, ys = np.append(xs, xs[0]), np.append(ys, ys[0])
    return _path_pixels(xs, ys, shape), 2 * math.pi * r


def _junctions(boundary: np.ndarray, already_thin: bool = False) -> np.ndarray:
    from skimage.morphology import skeletonize

    sk = boundary if already_thin else skeletonize(boundary)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    return sk & (nb >= 3)


def _count_runs(values: np.ndarray, closed: bool) -> list[tuple[int, int]]:
    """(start, end) index pairs of True runs along a path."""
    v = values.astype(np.int8)
    if not v.any():
        return []
    if closed and v.all():
        return [(0, len(v) - 1)]
    d = np.diff(np.concatenate([[0], v, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    runs = list(zip(starts, ends))
    if closed and len(runs) > 1 and v[0] and v[-1]:  # run wrapping around the closure point
        s0, e0 = runs[0]
        sL, eL = runs[-1]
        runs = [(sL, e0 + len(v))] + runs[1:-1]
    return runs


def linear_intercept(boundary: np.ndarray, cal: Calibration | None = None, n_lines: int = 5,
                     pattern: str = "grid", margin: float = 0.05,
                     triple_points: bool = False, thin: bool = True) -> InterceptResult:
    """ASTM E112 mean lineal intercept on a grain-boundary binary (True = boundary).

    Test patterns: ``horizontal``, ``vertical``, ``grid`` (``n_lines`` horizontal +
    ``n_lines`` vertical lines) or ``circles`` (three concentric circles, diameters 1 : 2/3 : 1/3,
    as in the Abrams three-circle procedure — orientation independent). Grains are labelled
    (4-connected regions between boundaries) and every change of grain label along a test line is
    one boundary intersection P, so a line running along or jittering on a boundary is not
    over-counted and a tangent hit counts 0 or 1. If ``triple_points``, a crossing through a
    triple-junction pixel counts 1.5 (E112 convention). Line ends inside a grain are not counted.
    ``thin`` skeletonises the boundary map first (recommended for etched, several-px-wide boundaries).

    P_L = P / L_T, l̄ = 1 / P_L, G = −3.2877 − 6.6439·log10(l̄ [mm]).
    """
    cal = cal or Calibration()
    b = boundary.astype(bool)
    if thin:  # thick etched boundaries would otherwise eat into the intercept lengths
        b = thin_boundaries(b)
    h, w = b.shape
    mx, my = margin * w, margin * h
    paths: list[tuple[np.ndarray, float, bool, np.ndarray]] = []
    p = pattern.lower()
    if p in ("horizontal", "grid", "both"):
        for y in np.linspace(my, h - 1 - my, n_lines + 2)[1:-1]:
            a, bb = (mx, y), (w - 1 - mx, y)
            px, L = _sample_segment(a, bb, b.shape)
            paths.append((px, L, False, np.array([a, bb])))
    if p in ("vertical", "grid", "both"):
        for x in np.linspace(mx, w - 1 - mx, n_lines + 2)[1:-1]:
            a, bb = (x, my), (x, h - 1 - my)
            px, L = _sample_segment(a, bb, b.shape)
            paths.append((px, L, False, np.array([a, bb])))
    if p == "circles":
        cx, cy = (w - 1) / 2, (h - 1) / 2
        rmax = (min(h, w) - 1) / 2 * (1 - 2 * margin)
        for f in (1.0, 2 / 3, 1 / 3):
            px, L = _sample_circle(cx, cy, rmax * f, b.shape)
            t = np.linspace(0, 2 * math.pi, 241)
            poly = np.column_stack([cx + rmax * f * np.cos(t), cy + rmax * f * np.sin(t)])
            paths.append((px, L, True, poly))
    if not paths:
        raise ValueError(f"Unknown test pattern '{pattern}'")
    junc = _junctions(b, already_thin=thin) if triple_points else None
    grains = grains_from_boundaries(b, min_grain_px=3)
    total_px = 0.0
    P = 0.0
    n_triple = 0
    hits = []
    per_line = []
    for px, L, closed, _poly in paths:
        total_px += L
        labs = grains[px[:, 1], px[:, 0]]
        idx = np.flatnonzero(labs > 0)
        cnt = 0
        if len(idx) >= 2:
            seq = labs[idx]
            pairs = list(zip(range(len(idx) - 1), range(1, len(idx))))
            if closed:
                pairs.append((len(idx) - 1, 0))
            for i, j in pairs:
                if seq[i] == seq[j]:
                    continue
                a, bnd = idx[i], idx[j]
                span = np.arange(a, bnd + 1) if bnd > a else np.r_[np.arange(a, len(px)), np.arange(0, bnd + 1)]
                weight = 1.0
                if junc is not None and junc[px[span, 1], px[span, 0]].any():
                    weight = 1.5
                    n_triple += 1
                P += weight
                cnt += 1
                hits.append(px[span[len(span) // 2]])
        per_line.append(cnt)
    L_T = cal.length(total_px)
    P_L = P / L_T if L_T > 0 else 0.0
    lbar = 1.0 / P_L if P_L > 0 else math.inf
    lbar_mm = G = None
    if cal.calibrated and np.isfinite(lbar):
        lbar_mm = lbar * LENGTH_UNITS[cal.unit] / 1e-3
        G = astm_g_from_intercept(lbar_mm)
    warnings = _calibration_warnings(cal)
    if P == 0:
        warnings.append("No grain-boundary intersections found: mean intercept and G are undefined.")
    elif P < E112_MIN_COUNTS:
        warnings.append(f"Low statistics: only {P:g} intercepts counted (ASTM E112 recommends a test "
                        f"pattern giving ≥ {E112_MIN_COUNTS} per field) — add lines or fields.")
    return InterceptResult(pattern=p, n_lines=len(paths), total_length=L_T, intersections=P,
                           n_triple=n_triple, P_L=P_L, mean_intercept=lbar, unit=cal.unit,
                           mean_intercept_mm=lbar_mm, G=G, lines=[q[3] for q in paths],
                           hits=np.array(hits, float).reshape(-1, 2), per_line=per_line,
                           warnings=warnings)


def _calibration_warnings(cal: Calibration) -> list[str]:
    if not cal.calibrated:
        return ["Image is uncalibrated: lengths are in pixels and no ASTM grain size number can be "
                "given. Set the scale from the scale bar."]
    if cal.from_print_resolution:
        return ["Calibration comes from a print resolution (dpi), not from the microscope — the "
                "grain size is probably wrong."]
    return []


def astm_g_from_intercept(lbar_mm: float) -> float:
    """ASTM E112: G = −3.2877 − 6.6439·log10(l̄), l̄ mean lineal intercept in mm."""
    return -3.2877 - 6.6439 * math.log10(lbar_mm)


def astm_g_from_na(na_per_mm2: float) -> float:
    """ASTM E112: G = 3.321928·log10(N_A) − 2.954, N_A grains per mm²."""
    return 3.321928 * math.log10(na_per_mm2) - 2.954


# --------------------------------------------------------------------------------------
# ASTM E112 — planimetric (Jeffries)
# --------------------------------------------------------------------------------------


@dataclass
class PlanimetricResult:
    n_inside: int
    n_intercepted: int
    n_corner: int
    n_equivalent: float          # N_inside + 0.5·N_intercepted + 0.25·N_corner
    area: float                  # field area in unit²
    unit: str
    N_A: float                   # grains per unit²
    N_A_mm2: float | None
    mean_grain_area: float       # 1/N_A in unit²
    G: float | None
    labels: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0), np.int32))
    inside_labels: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, int))
    edge_labels: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, int))
    warnings: list[str] = field(default_factory=list)

    def rows(self) -> list[dict]:
        u = self.unit
        out = [
            {"Quantity": "Grains completely inside", "Value": self.n_inside, "Unit": ""},
            {"Quantity": "Grains intercepted by border", "Value": self.n_intercepted, "Unit": ""},
            {"Quantity": "Corner grains (×0.25)", "Value": self.n_corner, "Unit": ""},
            {"Quantity": "Equivalent grain count N", "Value": self.n_equivalent, "Unit": ""},
            {"Quantity": "Field area", "Value": self.area, "Unit": f"{u}²"},
            {"Quantity": "N_A", "Value": self.N_A, "Unit": f"1/{u}²"},
            {"Quantity": "Mean grain area Ā = 1/N_A", "Value": self.mean_grain_area, "Unit": f"{u}²"},
        ]
        if self.N_A_mm2 is not None:
            out.append({"Quantity": "N_A", "Value": self.N_A_mm2, "Unit": "1/mm²"})
        if self.G is not None:
            out.append({"Quantity": "ASTM E112 grain size G (planimetric)", "Value": self.G, "Unit": ""})
        return out


def planimetric(grains: np.ndarray, cal: Calibration | None = None, min_grain_px: int = 5,
                thin: bool = True) -> PlanimetricResult:
    """ASTM E112 planimetric (Jeffries) count on the whole rectangular image field.

    ``grains`` is a label image (0 = boundary / ignored) **or** a boolean grain-boundary map
    (True = boundary), which is labelled with :func:`grains_from_boundaries`.
    For a rectangular field E112 counts grains completely inside as 1, grains intercepted by the
    border as ½ and the four corner grains as ¼ : N = N_inside + ½·N_intercepted + ¼·N_corner.
    N_A = N / A, G = 3.321928·log10(N_A [mm⁻²]) − 2.954.
    """
    cal = cal or Calibration()
    if grains.dtype == bool:
        b = thin_boundaries(grains) if thin else grains
        lab = grains_from_boundaries(b, min_grain_px)
    else:
        lab = grains.astype(np.int32)
    h, w = lab.shape
    ids = np.unique(lab)
    ids = ids[ids > 0]
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    corner = set()
    for (r, c) in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        win = lab[max(0, r - 2):r + 3, max(0, c - 2):c + 3]
        rr, cc = np.nonzero(win)
        if len(rr):  # grain nearest to the corner pixel
            r0, c0 = max(0, r - 2), max(0, c - 2)
            d = (rr + r0 - r) ** 2 + (cc + c0 - c) ** 2
            corner.add(int(win[rr[np.argmin(d)], cc[np.argmin(d)]]))
    inside = np.setdiff1d(ids, border)
    edge = np.setdiff1d(border, np.array(sorted(corner), int))
    n_corner = len(corner)
    n_eq = len(inside) + 0.5 * len(edge) + 0.25 * n_corner
    area = cal.area(float(h * w))
    NA = n_eq / area if area > 0 else 0.0
    NA_mm2 = G = None
    warnings = _calibration_warnings(cal)
    degenerate = len(ids) <= 1
    if cal.calibrated and NA > 0:
        unit_mm = LENGTH_UNITS[cal.unit] / 1e-3
        NA_mm2 = NA / unit_mm ** 2
        G = None if degenerate else astm_g_from_na(NA_mm2)
    if degenerate:
        warnings.append("No grain boundaries inside the field (≤ 1 grain): planimetric N_A and G are "
                        "meaningless and G is not reported.")
    elif n_eq < E112_MIN_COUNTS:
        warnings.append(f"Low statistics: only {n_eq:g} grains counted (ASTM E112 planimetric "
                        f"procedure asks for ≥ {E112_MIN_COUNTS} grains per field) — lower the "
                        "magnification or add fields.")
    return PlanimetricResult(n_inside=len(inside), n_intercepted=len(edge), n_corner=n_corner,
                             n_equivalent=n_eq, area=area, unit=cal.unit, N_A=NA, N_A_mm2=NA_mm2,
                             mean_grain_area=1.0 / NA if NA > 0 else math.inf, G=G, labels=lab,
                             inside_labels=inside, edge_labels=np.union1d(edge, list(corner)),
                             warnings=warnings)


# --------------------------------------------------------------------------------------
# Grain area distribution
# --------------------------------------------------------------------------------------


@dataclass
class GrainDistribution:
    areas: np.ndarray            # unit²
    ecd: np.ndarray              # unit
    unit: str
    labels: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0), np.int32))
    G_mean_area: float | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> list[dict]:
        u = self.unit
        if len(self.areas) == 0:
            return [{"Quantity": "Grains measured", "Value": 0, "Unit": ""}]
        out = [
            {"Quantity": "Grains measured (edge grains excluded)", "Value": len(self.areas), "Unit": ""},
            {"Quantity": "Mean grain area", "Value": float(self.areas.mean()), "Unit": f"{u}²"},
            {"Quantity": "Median grain area", "Value": float(np.median(self.areas)), "Unit": f"{u}²"},
            {"Quantity": "Std grain area", "Value": float(self.areas.std(ddof=1)) if len(self.areas) > 1 else 0.0,
             "Unit": f"{u}²"},
            {"Quantity": "Mean ECD", "Value": float(self.ecd.mean()), "Unit": u},
            {"Quantity": "Median ECD", "Value": float(np.median(self.ecd)), "Unit": u},
        ]
        if self.G_mean_area is not None:
            out.append({"Quantity": "ASTM G from mean area (N_A = 1/Ā)", "Value": self.G_mean_area,
                        "Unit": ""})
        return out


def grain_area_distribution(grains: np.ndarray, cal: Calibration | None = None,
                            exclude_edges: bool = True, min_grain_px: int = 5) -> GrainDistribution:
    """Areas and equivalent circle diameters of segmented grains.

    ``grains`` label image or boolean boundary map. Boundaries are split between neighbouring
    grains (labels expanded) so the grain areas tile the image; edge grains are excluded by default.
    """
    cal = cal or Calibration()
    if grains.dtype == bool:
        lab = grains_from_boundaries(grains, min_grain_px, fill_boundaries=True)
    else:
        lab = fill_label_gaps(grains.astype(np.int32))
    counts = np.bincount(lab.ravel())
    ids = np.flatnonzero(counts)
    ids = ids[ids > 0]
    if exclude_edges:
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        ids = np.setdiff1d(ids, border)
    a_px = counts[ids].astype(float)
    areas = cal.area(a_px) if len(a_px) else np.zeros(0)
    ecd = cal.length(np.sqrt(4 * a_px / math.pi)) if len(a_px) else np.zeros(0)
    G = None
    if cal.calibrated and len(areas):
        unit_mm = LENGTH_UNITS[cal.unit] / 1e-3
        G = astm_g_from_na(1.0 / (float(np.mean(areas)) * unit_mm ** 2))
    warnings = _calibration_warnings(cal)
    if len(a_px) < E112_MIN_COUNTS:
        warnings.append(f"Low statistics: only {len(a_px)} complete grains measured (< "
                        f"{E112_MIN_COUNTS}); the distribution and its mean are uncertain.")
    return GrainDistribution(areas=np.atleast_1d(areas), ecd=np.atleast_1d(ecd), unit=cal.unit,
                             labels=lab, G_mean_area=G, warnings=warnings)


# --------------------------------------------------------------------------------------
# Phase fraction
# --------------------------------------------------------------------------------------


@dataclass
class PhaseResult:
    thresholds: list[float]
    classes: np.ndarray          # int8 class map 0..n-1 (dark → bright)
    fractions: list[float]       # area % per class
    areas: list[float]           # unit²
    ranges: list[tuple[float, float]]
    unit: str

    def rows(self) -> list[dict]:
        out = []
        for i, (f, a, r) in enumerate(zip(self.fractions, self.areas, self.ranges)):
            out.append({"Phase": i + 1, "Intensity min": r[0], "Intensity max": r[1],
                        "Area fraction (%)": f, f"Area ({self.unit}²)": a})
        return out


def phase_fractions(img: np.ndarray, n_phases: int = 2, thresholds: list[float] | None = None,
                    cal: Calibration | None = None, mask: np.ndarray | None = None) -> PhaseResult:
    """Area fractions of intensity classes.

    Thresholds either given (ascending; n_phases = len + 1) or from multi-level Otsu
    (skimage ``threshold_multiotsu``, 2–5 classes). Class i contains t_{i-1} < I ≤ t_i.
    ``mask`` restricts the analysed region (fractions relative to the masked area).
    """
    from skimage.filters import threshold_multiotsu

    cal = cal or Calibration()
    g = to_gray(img).astype(float)
    vals = g[mask] if mask is not None else g.ravel()
    if thresholds is None:
        if n_phases < 2:
            raise ValueError("Need at least 2 phases")
        if n_phases == 2:
            from skimage.filters import threshold_otsu

            thresholds = [float(threshold_otsu(vals))]
        else:
            thresholds = [float(t) for t in threshold_multiotsu(vals, classes=n_phases)]
    thresholds = sorted(float(t) for t in thresholds)
    classes = np.digitize(g, thresholds, right=True).astype(np.int8)
    region = mask if mask is not None else np.ones(g.shape, bool)
    n_tot = int(region.sum())
    fr, ar, rng = [], [], []
    edges = [float(g.min())] + thresholds + [float(g.max())]
    for i in range(len(thresholds) + 1):
        n = int(((classes == i) & region).sum())
        fr.append(100.0 * n / n_tot if n_tot else 0.0)
        ar.append(cal.area(float(n)))
        rng.append((edges[i], edges[i + 1]))
    return PhaseResult(thresholds=thresholds, classes=classes, fractions=fr, areas=ar, ranges=rng,
                       unit=cal.unit)


# --------------------------------------------------------------------------------------
# Histogram
# --------------------------------------------------------------------------------------


@dataclass
class HistogramStats:
    counts: np.ndarray
    edges: np.ndarray
    n: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    mode: float
    skewness: float
    kurtosis: float

    def rows(self) -> list[dict]:
        return [{"Quantity": k, "Value": getattr(self, k), "Unit": ""}
                for k in ("n", "mean", "std", "min", "max", "median", "mode", "skewness", "kurtosis")]


def histogram_stats(img: np.ndarray, bins: int = 256, mask: np.ndarray | None = None,
                    value_range: tuple[float, float] | None = None) -> HistogramStats:
    """Intensity histogram and statistics (ImageJ Analyze ▸ Histogram)."""
    g = to_gray(img)
    vals = (g[mask] if mask is not None else g.ravel()).astype(float)
    if value_range is None:
        if np.issubdtype(g.dtype, np.integer) and g.dtype == np.uint8:
            value_range = (0, 256)
        else:
            value_range = (float(vals.min()), float(vals.max()) if vals.max() > vals.min()
                           else float(vals.min()) + 1)
    counts, edges = np.histogram(vals, bins=bins, range=value_range)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    sd0 = vals.std()
    skew = float(((vals - mean) ** 3).mean() / sd0 ** 3) if sd0 > 0 else 0.0
    kurt = float(((vals - mean) ** 4).mean() / sd0 ** 4 - 3.0) if sd0 > 0 else 0.0
    i = int(np.argmax(counts))
    return HistogramStats(counts=counts, edges=edges, n=int(vals.size), mean=mean, std=std,
                          min=float(vals.min()), max=float(vals.max()), median=float(np.median(vals)),
                          mode=float((edges[i] + edges[i + 1]) / 2), skewness=skew, kurtosis=kurt)
