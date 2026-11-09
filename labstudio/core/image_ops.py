"""Image operations for the Micrographs workspace (pure numpy / scipy / scikit-image / tifffile).

Everything here is Qt-free so it can be unit-tested headless. Images are numpy arrays,
either 2-D greyscale (uint8, uint16, float) or 3-D RGB(A) (H, W, 3|4). Functions return new arrays
(never modify their input) and, where sensible, keep the input dtype.

Spatial calibration is carried by :class:`Calibration` (pixels per unit + unit symbol).
Lengths reported to the user are ``pixels / px_per_unit`` in ``unit`` and areas in ``unit²``.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------

#: Length units supported for spatial calibration, with their size in metres.
LENGTH_UNITS: dict[str, float] = {"px": float("nan"), "nm": 1e-9, "µm": 1e-6, "mm": 1e-3,
                                  "cm": 1e-2, "m": 1.0}

_UNIT_ALIASES = {"um": "µm", "μm": "µm", "micron": "µm", "microns": "µm", "micrometer": "µm",
                 "micrometre": "µm", "nanometer": "nm", "nanometre": "nm", "millimeter": "mm",
                 "millimetre": "mm", "pixel": "px", "pixels": "px", "pixel(s)": "px",
                 "\\u00b5m": "µm", "µ": "µm", "u": "µm", "mum": "µm", "a": "Å"}


def normalize_length_unit(unit: str | None) -> str:
    """Normalise a length unit string (``um``, ``micron``, ``μm`` → ``µm``). Unknown → ``px``."""
    if not unit:
        return "px"
    u = unit.strip()
    u = _UNIT_ALIASES.get(u.lower(), u)
    if u == "Å":
        return "nm"  # caller must scale; handled in parse helpers
    return u if u in LENGTH_UNITS else "px"


#: Human-readable calibration sources (shown next to the scale and written into provenance).
SOURCE_UNCALIBRATED = "uncalibrated"
SOURCE_SCALE_BAR = "set from scale bar"
SOURCE_IMAGEJ = "ImageJ metadata"
SOURCE_OME = "OME metadata"
SOURCE_LABSTUDIO = "LabStudio saved calibration"
SOURCE_DPI = "print resolution (dpi) — user accepted, NOT a microscope scale"

#: Unit symbols accepted by :class:`Calibration` that are rescaled to a supported unit.
_RESCALED_UNITS = {"å": ("nm", 10.0), "angstrom": ("nm", 10.0), "in": ("mm", 1 / 25.4),
                   "inch": ("mm", 1 / 25.4), "inches": ("mm", 1 / 25.4)}


@dataclass
class Calibration:
    """Spatial calibration: ``px_per_unit`` pixels correspond to 1 ``unit``.

    ``unit == "px"`` (with ``px_per_unit == 1``) means uncalibrated. ``source`` records where the
    scale came from ("FEI tag 34682", "Zeiss tag 34118", "sidecar x.txt", "ImageJ metadata",
    "OME metadata", "set from scale bar", "LabStudio saved calibration" ...). It is shown next to
    the scale in the GUI and written into the provenance of every result.
    Å and inch are accepted and converted (px/Å → px/nm ×10, px/in → px/mm ÷25.4).
    """

    px_per_unit: float = 1.0
    unit: str = "px"
    source: str = ""

    def __post_init__(self) -> None:
        raw = (self.unit or "").strip()
        resc = _RESCALED_UNITS.get(raw.lower()) or (_RESCALED_UNITS["å"] if raw in ("Å", "Å") else None)
        if resc is not None:
            self.unit = resc[0]
            self.px_per_unit = float(self.px_per_unit) * resc[1]
        else:
            self.unit = normalize_length_unit(self.unit)
        if not (self.px_per_unit > 0 and math.isfinite(self.px_per_unit)):
            raise ValueError("px_per_unit must be positive and finite")

    # -- basic conversions -----------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.unit != "px"

    @property
    def source_label(self) -> str:
        """Where the scale came from; ``'uncalibrated'`` when there is no scale."""
        if not self.calibrated:
            return SOURCE_UNCALIBRATED
        return self.source or "unknown source"

    @property
    def from_print_resolution(self) -> bool:
        return self.calibrated and self.source.startswith("print resolution")

    @property
    def unit_per_px(self) -> float:
        """Size of one pixel in ``unit``."""
        return 1.0 / self.px_per_unit

    @property
    def area_unit(self) -> str:
        return f"{self.unit}²"

    def length(self, px):
        """Convert a length in pixels to calibrated units."""
        return np.asarray(px, dtype=float) / self.px_per_unit if not np.isscalar(px) \
            else float(px) / self.px_per_unit

    def area(self, px2):
        """Convert an area in pixels² to calibrated unit²."""
        return np.asarray(px2, dtype=float) / self.px_per_unit ** 2 if not np.isscalar(px2) \
            else float(px2) / self.px_per_unit ** 2

    def to_px(self, length):
        """Convert a calibrated length to pixels."""
        return length * self.px_per_unit

    def pixel_size_m(self) -> float | None:
        """Pixel size in metres, or None if uncalibrated."""
        if not self.calibrated:
            return None
        return self.unit_per_px * LENGTH_UNITS[self.unit]

    def to_unit(self, unit: str) -> "Calibration":
        """Same calibration expressed in another length unit (nm/µm/mm...)."""
        unit = normalize_length_unit(unit)
        if not self.calibrated or unit == "px":
            return Calibration(self.px_per_unit, self.unit, self.source)
        factor = LENGTH_UNITS[unit] / LENGTH_UNITS[self.unit]  # 1 new unit = factor old units
        return Calibration(self.px_per_unit * factor, unit, self.source)

    def length_in(self, px, unit: str):
        """Length in pixels → value in ``unit`` (must be calibrated unless unit == 'px')."""
        unit = normalize_length_unit(unit)
        if unit == "px":
            return px
        if not self.calibrated:
            raise ValueError("Image is not spatially calibrated")
        return self.to_unit(unit).length(px)

    def describe(self, with_source: bool = False) -> str:
        if not self.calibrated:
            return "Uncalibrated (pixels)" + (f" · source: {SOURCE_UNCALIBRATED}" if with_source else "")
        s = f"{self.px_per_unit:.6g} px/{self.unit}  (1 px = {self.unit_per_px:.6g} {self.unit})"
        return s + (f" · source: {self.source_label}" if with_source else "")

    def short(self) -> str:
        """Compact scale string for result rows, e.g. '0.05 µm/px' or 'uncalibrated (px)'."""
        if not self.calibrated:
            return "uncalibrated (px)"
        return f"{self.unit_per_px:.6g} {self.unit}/px"

    def to_dict(self) -> dict:
        return {"px_per_unit": float(self.px_per_unit), "unit": self.unit, "source": self.source}

    @staticmethod
    def from_dict(d: dict | None) -> "Calibration":
        if not d:
            return Calibration()
        return Calibration(float(d.get("px_per_unit", 1.0)), d.get("unit", "px"), d.get("source", ""))

    @staticmethod
    def from_pixel_size(size: float, unit: str, source: str = "") -> "Calibration":
        """Build from a pixel size (e.g. 2.345 nm per pixel), choosing a readable unit."""
        unit = normalize_length_unit(unit)
        if unit == "px" or size <= 0:
            return Calibration()
        size_m = size * LENGTH_UNITS[unit]
        best = choose_unit_for_pixel_size(size_m)
        per_px = size_m / LENGTH_UNITS[best]
        return Calibration(1.0 / per_px, best, source)

    @staticmethod
    def from_known_distance(pixels: float, distance: float, unit: str,
                            source: str = SOURCE_SCALE_BAR) -> "Calibration":
        """ImageJ 'Set Scale': ``pixels`` pixels measured along a scale bar equal ``distance`` ``unit``."""
        if pixels <= 0 or distance <= 0:
            raise ValueError("Distance in pixels and known distance must be > 0")
        return Calibration(pixels / distance, unit, source)


def choose_unit_for_pixel_size(size_m: float) -> str:
    """Pick nm for SEM-like pixel sizes (< 10 nm/px), µm for typical LOM/SEM, mm for macro."""
    if size_m < 1e-8:
        return "nm"
    if size_m < 1e-4:
        return "µm"
    return "mm"


# --------------------------------------------------------------------------------------
# Loading & SEM metadata
# --------------------------------------------------------------------------------------

IMAGE_EXTENSIONS: list[str] = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]


@dataclass
class LoadedImage:
    data: np.ndarray
    calibration: Calibration
    metadata: dict = field(default_factory=dict)
    n_pages: int = 1
    page: int = 0
    path: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path) if self.path else "untitled"


def parse_fei_metadata(text: str) -> float | None:
    """Pixel size in metres from FEI/Thermo ini metadata (tag 34682 / 34680).

    Looks for ``PixelWidth=`` in the ``[Scan]`` section first, then ``[EScan]``/``[IBeam]``,
    then anywhere.
    """
    if not text:
        return None
    sections: dict[str, dict[str, str]] = {}
    current = ""
    for raw in text.replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\[(.+)]$", line)
        if m:
            current = m.group(1).strip().lower()
            sections.setdefault(current, {})
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            sections.setdefault(current, {})[k.strip().lower()] = v.strip()
    for sec in ("scan", "escan", "ibeam", "ebeam", ""):
        v = sections.get(sec, {}).get("pixelwidth")
        if v:
            try:
                val = float(v)
                if val > 0:
                    return val
            except ValueError:
                pass
    for sec in sections.values():
        if "pixelwidth" in sec:
            try:
                val = float(sec["pixelwidth"])
                if val > 0:
                    return val
            except ValueError:
                pass
    return None


_SIZE_RE = r"([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(pm|nm|µm|μm|um|mm|m)\b"


def _value_unit_to_m(value: str, unit: str) -> float:
    v = float(value.replace(",", "."))
    u = unit.lower().replace("μ", "µ")
    scale = {"pm": 1e-12, "nm": 1e-9, "µm": 1e-6, "um": 1e-6, "mm": 1e-3, "m": 1.0}[u]
    return v * scale


def parse_zeiss_metadata(text: str) -> float | None:
    """Pixel size in metres from Zeiss SmartSEM metadata (tag 34118).

    Matches ``Image Pixel Size = 2.345 nm`` (preferred) or ``Pixel Size = 2.345 nm``.
    """
    if not text:
        return None
    t = text.replace("\x00", "\n")
    for key in (r"Image\s+Pixel\s+Size", r"Pixel\s+Size"):
        m = re.search(key + r"\s*[=:]\s*" + _SIZE_RE, t, flags=re.IGNORECASE)
        if m:
            val = _value_unit_to_m(m.group(1), m.group(2))
            if val > 0:
                return val
    return None


def parse_sidecar_text(text: str, image_width_px: int | None = None) -> float | None:
    """Pixel size in metres from a JEOL / Hitachi sidecar ``.txt`` file.

    * Hitachi: ``PixelSize=12.34`` (nm per pixel).
    * JEOL: ``$$SM_MICRON_BAR 156`` (scale-bar length in pixels) with ``$$SM_MICRON_MARKER 10um``
      (its physical length), or ``$CM_PIXEL_SIZE``/``PixelSize`` style keys.
    """
    if not text:
        return None
    m = re.search(r"^\s*\$*\s*PixelSize\s*[=\s]\s*([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(\w+)?",
                  text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        val = float(m.group(1).replace(",", "."))
        unit = (m.group(2) or "nm").lower()
        if unit in ("pm", "nm", "um", "µm", "mm", "m"):
            return _value_unit_to_m(str(val), unit)
        return val * 1e-9  # Hitachi convention: nm
    bar = re.search(r"\$*SM_MICRON_BAR\s+(\d+(?:\.\d+)?)", text)
    marker = re.search(r"\$*SM_MICRON_MARKER\s+" + _SIZE_RE, text, flags=re.IGNORECASE)
    if bar and marker:
        bar_px = float(bar.group(1))
        length_m = _value_unit_to_m(marker.group(1), marker.group(2))
        if bar_px > 0:
            return length_m / bar_px
    return None


def _resolution_info(page) -> dict:
    """Raw TIFF XResolution / ResolutionUnit as ``{'print_dpi', 'resolution_unit', 'px_per'}`` (or {})."""
    tags = page.tags
    if "XResolution" not in tags:
        return {}
    try:
        num, den = tags["XResolution"].value
    except (TypeError, ValueError):
        return {}
    if not den or not num:
        return {}
    px_per = num / den
    unit_tag = int(tags["ResolutionUnit"].value) if "ResolutionUnit" in tags else 2
    info = {"px_per": float(px_per), "resolution_unit": {1: "none", 2: "inch", 3: "cm"}.get(unit_tag, "inch")}
    if unit_tag == 3:
        info["print_dpi"] = float(px_per * 2.54)
    elif unit_tag == 2:
        info["print_dpi"] = float(px_per)
    return info


def _imagej_calibration(page, description: str) -> Calibration | None:
    """ImageJ hyperstack description (``ImageJ=…`` + ``unit=…``): XResolution is px per ``unit``.

    A description that also carries ``labstudio_cal_source=…`` (written by LabStudio's TIFF export)
    is reported as a previously saved LabStudio calibration.
    """
    desc = description or ""
    if "ImageJ=" not in desc:
        return None
    m = re.search(r"^unit=(.+)$", desc, flags=re.MULTILINE)
    if not m:
        return None
    unit = m.group(1).strip().replace("\\u00B5", "µ").replace("\\u00b5", "µ")
    raw = unit.lower()
    if normalize_length_unit(unit) == "px" and raw not in _RESCALED_UNITS and unit not in ("Å",):
        return None
    info = _resolution_info(page)
    px_per = info.get("px_per", 0.0)
    if not px_per or px_per <= 0:
        return None
    src = SOURCE_IMAGEJ
    m2 = re.search(r"^labstudio_cal_source=(.+)$", desc, flags=re.MULTILINE)
    if m2:
        src = f"{SOURCE_LABSTUDIO} (originally: {m2.group(1).strip()})"
    try:
        return Calibration(px_per, unit, src)
    except ValueError:
        return None


_OME_UNITS = {"nm": 1e-9, "µm": 1e-6, "um": 1e-6, "μm": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0,
              "pm": 1e-12, "Å": 1e-10, "nanometer": 1e-9, "micrometer": 1e-6, "millimeter": 1e-3}


def parse_ome_xml(xml: str) -> float | None:
    """Pixel size in metres from OME-XML ``<Pixels PhysicalSizeX=… PhysicalSizeXUnit=…>`` (default µm)."""
    if not xml:
        return None
    m = re.search(r"<(?:\w+:)?Pixels\b[^>]*>", xml)
    if not m:
        return None
    tag = m.group(0)
    v = re.search(r'PhysicalSizeX="([^"]+)"', tag)
    if not v:
        return None
    u = re.search(r'PhysicalSizeXUnit="([^"]+)"', tag)
    unit = u.group(1) if u else "µm"
    try:
        val = float(v.group(1))
    except ValueError:
        return None
    f = _OME_UNITS.get(unit, _OME_UNITS.get(unit.lower()))
    if f is None or val <= 0:
        return None
    return val * f


def read_tiff_calibration(path: str) -> tuple[Calibration, dict]:
    """Read spatial calibration and raw instrument metadata from a TIFF file.

    Trusted sources only: FEI/Thermo (tag 34682/34680), Zeiss (34118), OME-XML, and ImageJ
    descriptions with ``unit=`` (incl. LabStudio's own exports). Generic TIFF ``XResolution`` /
    ``ResolutionUnit`` tags are a *print* resolution (dpi) set by scanners, screenshots and
    figure exports — never a microscope scale — so they are only reported in
    ``meta['print_dpi']`` and the image opens uncalibrated (see :func:`calibration_from_print_resolution`).
    """
    import tifffile

    meta: dict = {}
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        tags = page.tags
        desc = tags["ImageDescription"].value if "ImageDescription" in tags else ""
        if isinstance(desc, bytes):
            desc = desc.decode("latin-1", "replace")
        for code, label in ((34682, "FEI"), (34680, "FEI")):
            if code in tags:
                text = tags[code].value
                if isinstance(text, bytes):
                    text = text.decode("latin-1", "replace")
                if isinstance(text, dict):  # tifffile may already parse FEI metadata
                    scan = {k.lower(): v for k, v in text.items()}
                    sec = scan.get("scan") or scan.get("Scan") or {}
                    pw = {k.lower(): v for k, v in sec.items()}.get("pixelwidth") if sec else None
                    meta["instrument"] = "FEI/Thermo"
                    if pw:
                        return Calibration.from_pixel_size(float(pw), "m", f"{label} tag {code}"), meta
                    continue
                meta["instrument"] = "FEI/Thermo"
                meta["raw"] = str(text)[:20000]
                pw = parse_fei_metadata(str(text))
                if pw:
                    return Calibration.from_pixel_size(pw, "m", f"{label} tag {code}"), meta
        if 34118 in tags:
            raw = tags[34118].value
            meta["instrument"] = "Zeiss"
            pw = None
            if isinstance(raw, dict):  # tifffile parses CZ_SEM into {key: (name, value, unit)}
                items = sorted(raw.items(), key=lambda kv: kv[0] != "ap_pixel_size")
                for _k, v in items:
                    if (isinstance(v, (tuple, list)) and len(v) >= 3
                            and "pixel size" in str(v[0]).lower()):
                        try:
                            pw = _value_unit_to_m(str(v[1]), str(v[2]))
                            break
                        except (ValueError, KeyError):
                            continue
                meta["raw"] = "\n".join(f"{k}: {v}" for k, v in raw.items())[:20000]
            else:
                text = raw.decode("latin-1", "replace") if isinstance(raw, bytes) else str(raw)
                meta["raw"] = text[:20000]
                pw = parse_zeiss_metadata(text)
            if pw:
                return Calibration.from_pixel_size(pw, "m", "Zeiss tag 34118"), meta
        if getattr(tf, "is_ome", False) or "<OME" in (desc or "")[:2000]:
            try:
                pw = parse_ome_xml(tf.ome_metadata or desc)
            except Exception:  # noqa: BLE001 - malformed OME must not prevent opening
                pw = None
            if pw:
                return Calibration.from_pixel_size(pw, "m", SOURCE_OME), meta
        cal = _imagej_calibration(page, desc)
        if cal is not None:
            return cal, meta
        info = _resolution_info(page)
        if info.get("print_dpi"):
            meta["print_dpi"] = info["print_dpi"]
            meta["resolution_unit"] = info["resolution_unit"]
    return Calibration(), meta


def calibration_from_print_resolution(dpi: float) -> Calibration:
    """'Use dpi anyway': treat a print resolution of ``dpi`` dots per inch as the pixel size.

    Only meaningful for flat-bed scans of prints/negatives; the source is flagged so every
    result carries a warning.
    """
    if not dpi or dpi <= 0:
        return Calibration()
    return Calibration.from_pixel_size(0.0254 / float(dpi), "m", f"print resolution {dpi:g} dpi — "
                                                                 "user accepted, NOT a microscope scale")


def print_resolution_banner(dpi: float) -> str:
    """Banner text for a dpi-only image."""
    return (f"Image has a print resolution of {dpi:g} dpi — not a microscope scale. "
            "Set scale from the scale bar.")


def find_sidecar(path: str) -> str | None:
    """Return a JEOL/Hitachi-style sidecar text file next to the image (same stem), if any."""
    stem, _ = os.path.splitext(path)
    for ext in (".txt", ".TXT", ".hdr", ".HDR"):
        p = stem + ext
        if os.path.isfile(p):
            return p
    return None


def tiff_page_count(path: str) -> int:
    import tifffile

    with tifffile.TiffFile(path) as tf:
        return len(tf.pages)


def load_image(path: str, page: int = 0) -> LoadedImage:
    """Load an image file and its spatial calibration.

    TIFF (8/16/32-bit, multi-page → ``page``) via tifffile; PNG/JPG/BMP via Pillow.
    Calibration from SEM metadata (FEI/Thermo 34682, Zeiss 34118), OME-XML, ImageJ ``unit=``
    metadata (incl. LabStudio exports) or a sidecar .txt (JEOL/Hitachi), else uncalibrated.
    Generic dpi resolution tags (TIFF/PNG/JPEG) are *not* a scale: they are reported in
    ``metadata['print_dpi']`` and the image stays uncalibrated. ``metadata['calibration_source']``
    always names the source ('uncalibrated' if none).
    """
    ext = os.path.splitext(path)[1].lower()
    meta: dict = {}
    cal = Calibration()
    n_pages = 1
    if ext in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tf:
            n_pages = len(tf.pages)
            page = int(np.clip(page, 0, n_pages - 1))
            data = tf.pages[page].asarray()
        try:
            cal, meta = read_tiff_calibration(path)
        except Exception as exc:  # metadata problems must not prevent opening
            meta["metadata_error"] = str(exc)
    else:
        from PIL import Image

        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            if dpi:
                try:
                    d0 = float(dpi[0] if isinstance(dpi, (tuple, list)) else dpi)
                    if d0 > 0:
                        meta["print_dpi"] = round(d0, 4)
                        meta["resolution_unit"] = "inch"
                except (TypeError, ValueError):
                    pass
            if im.mode in ("I;16", "I;16B", "I;16L", "I"):
                data = np.array(im)
            elif im.mode in ("RGB", "RGBA", "P", "CMYK", "YCbCr", "LA"):
                data = np.array(im.convert("RGB"))
            elif im.mode == "1":
                data = np.array(im.convert("L"))
            else:
                data = np.array(im.convert("L")) if im.mode != "L" else np.array(im)
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[-1] not in (3, 4) and data.shape[0] in (3, 4):
        data = np.moveaxis(data, 0, -1)  # planar RGB
    if data.ndim == 3 and data.shape[-1] == 4:
        data = data[..., :3]
    if data.ndim == 3 and data.shape[-1] == 3 and _is_effectively_gray(data):
        data = data[..., 0].copy()
    if data.dtype == bool:
        data = data.astype(np.uint8) * 255
    if not cal.calibrated:
        side = find_sidecar(path)
        if side:
            try:
                with open(side, "r", encoding="latin-1") as fh:
                    pw = parse_sidecar_text(fh.read(), data.shape[1])
                if pw:
                    cal = Calibration.from_pixel_size(pw, "m", f"sidecar {os.path.basename(side)}")
                    meta.setdefault("instrument", "JEOL/Hitachi (sidecar)")
            except OSError:
                pass
    meta["calibration_source"] = cal.source_label
    return LoadedImage(data=data, calibration=cal, metadata=meta, n_pages=n_pages, page=page,
                       path=path)


def _is_effectively_gray(rgb: np.ndarray) -> bool:
    sub = rgb[::max(1, rgb.shape[0] // 64), ::max(1, rgb.shape[1] // 64)]
    return bool(np.array_equal(sub[..., 0], sub[..., 1]) and np.array_equal(sub[..., 0], sub[..., 2])
                and np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]))


def detect_data_bar(img: np.ndarray, min_frac: float = 0.03, max_frac: float = 0.35) -> int | None:
    """Detect the SEM information bar at the bottom of a micrograph.

    SEM data bars are strips of nearly uniform background (often black or white) with text,
    with very different statistics from the noisy image above. Returns the first row index of the
    bar (crop with ``img[:row]``) or ``None`` if no bar is found.
    """
    g = to_gray(img)
    h = g.shape[0]
    if h < 30:
        return None
    lo, hi = float(g.min()), float(g.max())
    if hi <= lo:
        return None
    if g.dtype in (np.uint8, np.uint16):  # 64-level quantisation by lookup table (no float copy)
        lut = np.clip(np.round((np.arange(int(hi) + 1, dtype=np.float64) - lo) / (hi - lo) * 63), 0, 63)
        q = lut.astype(np.uint8)[g]
    else:
        q = np.round((g.astype(np.float32) - lo) * np.float32(63.0 / (hi - lo))).astype(np.uint8)
    # fraction of pixels in each row equal to the row's most frequent level
    mode_frac = np.empty(h)
    for r in range(h):
        counts = np.bincount(q[r], minlength=64)
        mode_frac[r] = counts.max() / q.shape[1]
    # Image rows are "busy"; bar rows are dominated by one level.
    bar_like = mode_frac > 0.45
    # Typical image: row-to-row variation small, bar rows almost constant background
    min_rows = max(3, int(min_frac * h))
    max_rows = int(max_frac * h)
    if bar_like[-min_rows:].mean() < 0.7:
        return None
    # walk upward from the bottom until a run of non-bar rows
    run = 0
    boundary = h
    need = max(4, int(0.02 * h))
    r = h - 1
    while r >= 0:
        if bar_like[r]:
            boundary = r
            run = 0
        else:
            run += 1
            if run >= need:
                break
        r -= 1
    bar_rows = h - boundary
    if bar_rows < min_rows or bar_rows > max_rows:
        return None
    # The image above must actually differ: check that most of the image rows are not bar-like
    if bar_like[: boundary].mean() > 0.5:
        return None
    return int(boundary)


def detect_annotation_regions(img: np.ndarray, data_bar_row: int | None = None,
                              edge_frac: float = 0.3, max_area_frac: float = 0.12,
                              pad: int = 3) -> list[tuple[int, int, int, int]]:
    """Find burned-in scale bars, label boxes and text near the image corners/edges.

    Burned-in annotations are drawn with *flat* pure black/white (or saturated colour) pixels,
    whereas a micrograph is noisy. Candidate pixels are extreme (≤ 3 % or ≥ 97 % of the 8-bit
    range) *and* locally flat (3×3 range ≤ 2 grey levels); connected candidates whose bounding box
    (i) lies within ``edge_frac`` of an image edge in both directions of a corner, or along an
    edge, (ii) is fairly rectangular (filled area / box ≥ 0.55) and (iii) covers < ``max_area_frac``
    of the image are kept (filled boxes, bars). Nearby flat-extreme specks (label text) are merged
    into a kept box when within 4 % of the image size. If ``data_bar_row`` is given, the SEM data
    bar (rows ≥ that index) is returned as an extra region.

    Returns boxes ``(x0, y0, x1, y1)`` in pixel indices (exclusive upper bounds).
    """
    g = to_gray(img)
    if g.dtype != np.uint8:
        g = to_8bit(g)
    h, w = g.shape
    boxes: list[tuple[int, int, int, int]] = []
    if data_bar_row is not None and 0 < data_bar_row < h:
        boxes.append((0, int(data_bar_row), w, h))
    if h < 16 or w < 16:
        return boxes
    hh = data_bar_row if data_bar_row is not None and 0 < data_bar_row < h else h
    gg = g[:hh]
    extreme = (gg <= 8) | (gg >= 247)
    if not extreme.any():
        return boxes
    loc_max = ndi.maximum_filter(gg, size=3)
    loc_min = ndi.minimum_filter(gg, size=3)
    cand = extreme & ((loc_max.astype(np.int16) - loc_min) <= 2)
    del loc_max, loc_min
    # bridge text strokes / anti-aliased glyph edges to the box they sit in
    cand = ndi.binary_closing(cand, structure=np.ones((3, 3), bool), iterations=2) | cand
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    if n == 0:
        return boxes
    objs = ndi.find_objects(lab)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    ex, ey = edge_frac * w, edge_frac * hh
    min_side = max(3, int(0.004 * min(h, w)))
    kept: list[list[int]] = []
    specks: list[tuple[int, int, int, int]] = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        bw, bh = x1 - x0, y1 - y0
        if sizes[i] < 4:
            continue
        near_x = x0 < ex or x1 > w - ex
        near_y = y0 < ey or y1 > hh - ey
        if not (near_x and near_y):
            # also accept full-width / full-height strips along an edge
            if not ((bw > 0.6 * w and near_y) or (bh > 0.6 * hh and near_x)):
                continue
        filled = ndi.binary_fill_holes(lab[sl] == i).sum()
        rect = filled / float(bw * bh)
        big = bw >= max(8, 0.02 * w) or bh >= max(8, 0.02 * hh)
        if big and rect >= 0.55 and min(bw, bh) >= min_side and bw * bh <= max_area_frac * h * w:
            kept.append([x0, y0, x1, y1])
        elif bw * bh <= 0.01 * h * w:
            specks.append((x0, y0, x1, y1))
    if not kept:
        return boxes
    near = 0.04 * max(h, w)
    changed = True
    while changed:  # merge text specks and touching boxes
        changed = False
        for k in kept:
            for sp in list(specks):
                dx = max(k[0] - sp[2], sp[0] - k[2], 0)
                dy = max(k[1] - sp[3], sp[1] - k[3], 0)
                if dx <= near and dy <= near:
                    k[0], k[1] = min(k[0], sp[0]), min(k[1], sp[1])
                    k[2], k[3] = max(k[2], sp[2]), max(k[3], sp[3])
                    specks.remove(sp)
                    changed = True
    for k in kept:
        x0, y0, x1, y1 = k
        boxes.append((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(hh, y1 + pad)))
    return boxes


def regions_mask(shape: tuple[int, int], boxes) -> np.ndarray:
    """Boolean mask (True = excluded) of rectangular boxes (x0, y0, x1, y1)."""
    m = np.zeros(shape[:2], bool)
    for x0, y0, x1, y1 in boxes or []:
        m[int(y0):int(y1), int(x0):int(x1)] = True
    return m


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def dtype_range(dtype) -> tuple[float, float]:
    """Nominal display range for a dtype (ImageJ-like)."""
    dt = np.dtype(dtype)
    if dt == np.uint8:
        return 0.0, 255.0
    if dt == np.uint16:
        return 0.0, 65535.0
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def to_gray(img: np.ndarray) -> np.ndarray:
    """Greyscale view: RGB → luminance (float, ITU-R 601 weights as in ImageJ 'weighted'); 2-D unchanged."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] >= 3:
        f = np.float64 if img.dtype == np.float64 else np.float32  # float32 halves memory for big RGB
        out = img[..., 0].astype(f) * f(0.299)
        out += img[..., 1].astype(f) * f(0.587)
        out += img[..., 2].astype(f) * f(0.114)
        return out
    return img[..., 0]


def _restore_dtype(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    if np.issubdtype(like.dtype, np.integer):
        lo, hi = dtype_range(like.dtype)
        out = np.rint(out, out=out) if out.dtype.kind == "f" else out
        np.clip(out, lo, hi, out=out)
        return out.astype(like.dtype)
    return out.astype(like.dtype if like.dtype.kind == "f" else np.float64, copy=False)


def _wf(img: np.ndarray) -> np.ndarray:
    """Working float copy: float32 for 8/16-bit integer data (enough precision, half the memory
    of float64), float64 kept for float64 input."""
    if img.dtype == np.float64 or (np.issubdtype(img.dtype, np.integer) and img.dtype.itemsize > 2):
        return img.astype(np.float64)
    return img.astype(np.float32)


def _per_channel(fn, img: np.ndarray, *args, **kwargs) -> np.ndarray:
    if img.ndim == 3:
        return np.stack([fn(img[..., c], *args, **kwargs) for c in range(img.shape[-1])], axis=-1)
    return fn(img, *args, **kwargs)


def _disk(radius: float) -> np.ndarray:
    r = max(0.5, float(radius))
    n = int(math.floor(r))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return (x * x + y * y) <= r * r + 1  # ImageJ-like 'circular' kernel (r² + 1)


# --------------------------------------------------------------------------------------
# Adjust
# --------------------------------------------------------------------------------------


def auto_contrast_limits(img: np.ndarray, saturated: float = 0.35) -> tuple[float, float]:
    """ImageJ 'Auto' B&C: limits that saturate ``saturated`` % of pixels in total (half each end)."""
    g = to_gray(img)
    lo, hi = np.percentile(g, [saturated / 2.0, 100.0 - saturated / 2.0])
    if hi <= lo:
        lo, hi = float(g.min()), float(g.max())
        if hi <= lo:
            hi = lo + 1
    return float(lo), float(hi)


def apply_window(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map [lo, hi] linearly to 0–255 (uint8), clipping outside — ImageJ display LUT / 'Apply'."""
    if hi <= lo:
        hi = lo + 1e-12
    if img.dtype in (np.uint8, np.uint16):  # lookup table: no float copy of the image
        lut = (np.arange(256 if img.dtype == np.uint8 else 65536, dtype=np.float64) - lo) * (255.0 / (hi - lo)) + 0.5
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return lut[img]
    out = (img.astype(np.float32) - np.float32(lo)) * np.float32(255.0 / (hi - lo)) + np.float32(0.5)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def brightness_contrast_window(brightness: float, contrast: float, lo0: float, hi0: float
                               ) -> tuple[float, float]:
    """Display window from ImageJ-style brightness/contrast sliders.

    ``brightness`` in [-1, 1] shifts the window centre by up to the full range; ``contrast`` in
    [-1, 1] scales the window width by 2**(−4·contrast) (positive = more contrast).
    """
    rng = hi0 - lo0
    centre = (lo0 + hi0) / 2.0 - brightness * rng
    width = rng * 2.0 ** (-4.0 * contrast)
    return centre - width / 2.0, centre + width / 2.0


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    """ImageJ Process ▸ Math ▸ Gamma: out = range·(in/range)^gamma, applied on the data range."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if img.dtype in (np.uint8, np.uint16):
        lo, hi = dtype_range(img.dtype)
        x = np.arange(int(hi) + 1, dtype=np.float64)
        lut = np.clip(np.rint(lo + (hi - lo) * np.clip((x - lo) / (hi - lo), 0, 1) ** gamma), lo, hi)
        return lut.astype(img.dtype)[img]
    f = _wf(img)
    lo, hi = (dtype_range(img.dtype) if np.issubdtype(img.dtype, np.integer)
              else (float(f.min()), float(f.max())))
    if hi <= lo:
        return img.copy()
    out = lo + (hi - lo) * np.clip((f - lo) / (hi - lo), 0, 1) ** gamma
    return _restore_dtype(out, img)


def invert(img: np.ndarray) -> np.ndarray:
    """Invert intensities (uint: max − v; float: min + max − v)."""
    if np.issubdtype(img.dtype, np.integer):
        lo, hi = dtype_range(img.dtype)
        if img.dtype == np.uint16 and img.max() <= 4095 and img.max() > 255:
            pass  # still invert on the full 16-bit range like ImageJ ≥1.52
        return (hi + lo - img.astype(np.int64)).astype(img.dtype)
    return (float(img.min()) + float(img.max()) - img).astype(img.dtype)


def to_8bit(img: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Convert to 8-bit greyscale scaling [lo, hi] (default: data min–max) to 0–255 (ImageJ)."""
    g = to_gray(img)
    if g.dtype == np.uint8 and lo is None and hi is None:
        return g.copy()
    if lo is None or hi is None:
        lo, hi = float(g.min()), float(g.max())
    return apply_window(g, lo, hi)


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation (keeps dtype)."""
    from skimage import exposure

    def _eq(ch):
        lo, hi = dtype_range(ch.dtype) if np.issubdtype(ch.dtype, np.integer) else (ch.min(), ch.max())
        out = exposure.equalize_hist(ch, nbins=256 if ch.dtype == np.uint8 else 4096)
        return _restore_dtype(lo + out * (hi - lo), ch)

    return _per_channel(_eq, img)


def clahe(img: np.ndarray, clip_limit: float = 0.01, tile: int = 64) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation (skimage), ``tile`` = block size in px."""
    from skimage import exposure

    def _c(ch):
        lo, hi = dtype_range(ch.dtype) if np.issubdtype(ch.dtype, np.integer) else (ch.min(), ch.max())
        norm = (ch.astype(float) - lo) / max(hi - lo, 1e-12)
        out = exposure.equalize_adapthist(np.clip(norm, 0, 1), kernel_size=max(8, int(tile)),
                                          clip_limit=clip_limit)
        return _restore_dtype(lo + out * (hi - lo), ch)

    return _per_channel(_c, img)


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with standard deviation ``sigma`` px."""
    return _per_channel(lambda c: _restore_dtype(ndi.gaussian_filter(_wf(c), sigma), c), img)


def median_filter(img: np.ndarray, radius: float) -> np.ndarray:
    """Median filter over a circular neighbourhood of ``radius`` px (ImageJ 'Median')."""
    fp = _disk(radius)
    return _per_channel(lambda c: ndi.median_filter(c, footprint=fp), img)


def mean_filter(img: np.ndarray, radius: float) -> np.ndarray:
    """Mean filter over a circular neighbourhood of ``radius`` px (ImageJ 'Mean')."""
    k = _disk(radius).astype(np.float32)
    k /= k.sum()
    return _per_channel(lambda c: _restore_dtype(ndi.convolve(_wf(c), k.astype(_wf(c[:1, :1]).dtype),
                                                              mode="reflect"), c), img)


def unsharp_mask(img: np.ndarray, radius: float = 2.0, amount: float = 0.6) -> np.ndarray:
    """ImageJ unsharp mask: (in − w·gauss(in, σ)) / (1 − w), w = ``amount`` in [0, 0.9]."""
    w = float(np.clip(amount, 0.0, 0.95))

    def _u(c):
        f = _wf(c)
        out = (f - w * ndi.gaussian_filter(f, radius)) / (1.0 - w)
        return _restore_dtype(out, c)

    return _per_channel(_u, img)


def sharpen(img: np.ndarray) -> np.ndarray:
    """ImageJ Process ▸ Sharpen (3×3 kernel −1…12…−1, /4)."""
    k = np.array([[-1, -1, -1], [-1, 12, -1], [-1, -1, -1]], float) / 4.0
    return _per_channel(lambda c: _restore_dtype(ndi.convolve(_wf(c), k.astype(_wf(c[:1, :1]).dtype),
                                                              mode="reflect"), c), img)


def find_edges(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude (ImageJ 'Find Edges')."""
    def _e(c):
        f = _wf(c)
        mag = np.hypot(ndi.sobel(f, axis=0), ndi.sobel(f, axis=1))
        return _restore_dtype(mag, c)

    return _per_channel(_e, img)


def fft_bandpass(img: np.ndarray, large: float = 40.0, small: float = 3.0,
                 autoscale: bool = True) -> np.ndarray:
    """FFT band-pass: remove structures larger than ``large`` px and smaller than ``small`` px.

    Gaussian filters in the frequency domain (like ImageJ 'FFT ▸ Bandpass Filter'); the mean
    intensity is kept, and with ``autoscale`` the result is stretched back to the input range.
    """
    from scipy import fft as sfft

    def _bp(c):
        f = _wf(c)
        h, w = f.shape
        fy = np.fft.fftfreq(h).astype(f.dtype)[:, None]
        fx = np.fft.rfftfreq(w).astype(f.dtype)[None, :]
        fr2 = fx ** 2 + fy ** 2
        H = np.ones_like(fr2)
        if small and small > 0:  # Gaussian low-pass, sigma = small/2 px
            H *= np.exp(-2.0 * np.pi ** 2 * (small / 2.0) ** 2 * fr2)
        if large and large > 0:  # Gaussian high-pass, sigma = large/2 px
            H *= 1.0 - np.exp(-2.0 * np.pi ** 2 * (large / 2.0) ** 2 * fr2)
        H[0, 0] = 1.0  # keep mean
        del fr2
        spec = sfft.rfft2(f, workers=-1)
        spec *= H
        del H
        out = sfft.irfft2(spec, s=(h, w), workers=-1).astype(f.dtype, copy=False)
        del spec
        if autoscale:
            lo, hi = float(f.min()), float(f.max())
            olo, ohi = float(out.min()), float(out.max())
            if ohi > olo:
                out = lo + (out - olo) * ((hi - lo) / (ohi - olo))
        return _restore_dtype(out, c)

    return _per_channel(_bp, img)


def subtract_background(img: np.ndarray, radius: float = 50.0, light_background: bool = False,
                        return_background: bool = False, shrink: bool = True):
    """Rolling-ball background subtraction (skimage ``restoration.rolling_ball``).

    ``light_background``: the background is brighter than the features (e.g. LOM bright field).
    Like ImageJ, large radii are computed on a shrunk image for speed and the background is
    interpolated back. Returns the corrected image (same dtype; for light background the
    background is set to the maximum) and optionally the background estimate.
    """
    from skimage import restoration

    g = _wf(to_gray(img))
    work = -g if light_background else g
    del g
    offset = work.min()
    work -= offset
    factor = 1
    if shrink:
        factor = 1 if radius <= 10 else 2 if radius <= 30 else 4 if radius <= 100 else 8
        if work.size > 4_000_000:  # big images: keep the shrunk ball radius ≲ 12 px (speed)
            factor = max(factor, int(math.ceil(radius / 12.0)))
        factor = max(1, min(factor, int(min(work.shape) // 16) or 1))
    if factor > 1:
        # ImageJ shrinks with a minimum filter so the ball cannot rise above the features
        hh, ww = (work.shape[0] // factor) * factor, (work.shape[1] // factor) * factor
        small = work[:hh, :ww].reshape(hh // factor, factor, ww // factor, factor).min(axis=(1, 3))
        bg_small = restoration.rolling_ball(small, radius=radius / factor).astype(work.dtype)
        zoom = (work.shape[0] / bg_small.shape[0], work.shape[1] / bg_small.shape[1])
        bg = ndi.zoom(bg_small, zoom, order=1, mode="nearest", grid_mode=True)
        bg = bg[:work.shape[0], :work.shape[1]]
        if bg.shape != work.shape:  # rounding: pad with edge values
            bg = np.pad(bg, ((0, work.shape[0] - bg.shape[0]), (0, work.shape[1] - bg.shape[1])), mode="edge")
        np.minimum(bg, work, out=bg)
    else:
        bg = restoration.rolling_ball(work, radius=radius).astype(work.dtype)
    corrected = work
    corrected -= bg          # in place: no extra full-size float copy
    if light_background:
        lo, hi = (dtype_range(img.dtype) if np.issubdtype(img.dtype, np.integer)
                  else (0.0, float(-offset)))
        out = hi - corrected
        background = -(bg + offset)
    else:
        out = corrected
        background = bg + offset
    like = img if img.ndim == 2 else img[..., 0]
    out = _restore_dtype(out, like)
    if return_background:
        return out, background
    return out


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


def crop(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Crop to the rectangle [x0, x1) × [y0, y1) (clipped to the image)."""
    h, w = img.shape[:2]
    xa, xb = sorted((int(round(x0)), int(round(x1))))
    ya, yb = sorted((int(round(y0)), int(round(y1))))
    xa, xb = max(0, xa), min(w, xb)
    ya, yb = max(0, ya), min(h, yb)
    if xb <= xa or yb <= ya:
        raise ValueError("Crop rectangle is empty")
    return img[ya:yb, xa:xb].copy()


def rotate90(img: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate by k×90° counter-clockwise."""
    return np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))


def rotate(img: np.ndarray, angle_deg: float, enlarge: bool = False, order: int = 1) -> np.ndarray:
    """Rotate by an arbitrary angle (degrees, counter-clockwise), bilinear by default."""
    from skimage import transform

    f = img.astype(float)
    out = transform.rotate(f, angle_deg, resize=enlarge, order=order, mode="constant",
                           cval=float(f.min()), preserve_range=True)
    return _restore_dtype(out, img)


def flip(img: np.ndarray, horizontal: bool = True) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1] if horizontal else img[::-1])


def resize(img: np.ndarray, width: int, height: int, order: int = 1) -> np.ndarray:
    """Resize to ``width``×``height`` px (anti-aliased when shrinking)."""
    from skimage import transform

    shape = (int(height), int(width)) + img.shape[2:]
    shrinking = height < img.shape[0] or width < img.shape[1]
    out = transform.resize(img.astype(float), shape, order=order, preserve_range=True,
                           anti_aliasing=shrinking)
    return _restore_dtype(out, img)


# --------------------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------------------

THRESHOLD_METHODS = ["Otsu", "Li", "Yen", "Triangle", "IsoData", "Mean", "Minimum", "MaxEntropy"]


def _max_entropy_threshold(img: np.ndarray, nbins: int = 256) -> float:
    """Kapur–Sahoo–Wong maximum-entropy threshold (ImageJ 'MaxEntropy')."""
    hist, edges = np.histogram(img.ravel(), bins=nbins)
    p = hist.astype(float) / max(hist.sum(), 1)
    P = np.cumsum(p)
    best_t, best_h = 0, -np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
        cum_plogp = np.cumsum(plogp)
        total_plogp = cum_plogp[-1]
        for t in range(nbins - 1):
            pb, pf = P[t], 1.0 - P[t]
            if pb <= 0 or pf <= 0:
                continue
            hb = -cum_plogp[t] / pb + math.log(pb)
            hf = -(total_plogp - cum_plogp[t]) / pf + math.log(pf)
            if hb + hf > best_h:
                best_h, best_t = hb + hf, t
    return float(edges[best_t + 1])


def _otsu_threshold(vals: np.ndarray) -> float:
    """Otsu threshold; when the between-class variance has a flat maximum (well separated
    modes with an empty gap) the *middle* of the plateau is returned instead of its first bin."""
    vals = vals.ravel()
    lo, hi = float(vals.min()), float(vals.max())
    is_int = np.issubdtype(vals.dtype, np.integer) or bool(np.all(np.mod(vals[:1000], 1) == 0))
    nb = int(hi - lo) + 1 if is_int and hi - lo < 4096 else 1024
    if vals.dtype in (np.uint8, np.uint16) and hi - lo < 4096:   # exact & fast integer histogram
        hist = np.bincount(vals, minlength=int(hi) + 1)[int(lo):int(hi) + 1]
        edges = np.arange(lo, hi + 2, dtype=np.float64)
    else:
        hist, edges = np.histogram(vals, bins=nb, range=(lo, hi + (1 if is_int else 0)))
    centres = (edges[:-1] + edges[1:]) / 2 if not is_int else edges[:-1]
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centres)
    m0 = np.divide(s0, w0, out=np.zeros_like(s0), where=w0 > 0)
    m1 = np.divide(s0[-1] - s0, w1, out=np.zeros_like(s0), where=w1 > 0)
    var = w0 * w1 * (m0 - m1) ** 2
    vmax = var.max()
    best = np.flatnonzero(var >= vmax * (1 - 1e-9))
    # contiguous run containing the first maximum
    run = [best[0]]
    for b in best[1:]:
        if b == run[-1] + 1:
            run.append(b)
        else:
            break
    i = run[len(run) // 2] if len(run) > 1 else run[0]
    # threshold t: class 0 = values <= t
    return float(centres[i] if is_int else edges[i + 1])


def threshold_value(img: np.ndarray, method: str = "Otsu", mask: np.ndarray | None = None) -> float:
    """Automatic threshold (intensity value) by ``method`` in :data:`THRESHOLD_METHODS`.

    Pixels ``> t`` are 'bright', ``<= t`` 'dark'.
    """
    from skimage import filters

    g = to_gray(img)
    vals = g[mask] if mask is not None else g.ravel()
    if vals.size == 0 or vals.min() == vals.max():
        raise ValueError("Image has a single intensity; cannot threshold")
    m = method.lower()
    fn = {"otsu": filters.threshold_otsu, "li": filters.threshold_li, "yen": filters.threshold_yen,
          "triangle": filters.threshold_triangle, "isodata": filters.threshold_isodata,
          "mean": filters.threshold_mean, "minimum": filters.threshold_minimum}.get(m)
    if m == "maxentropy":
        return _max_entropy_threshold(vals)
    if m not in ("otsu", "mean"):
        vals = vals.astype(np.float32 if vals.dtype.itemsize <= 4 else np.float64, copy=False)
    if m == "otsu":
        return _otsu_threshold(g[mask] if mask is not None else g)
    if fn is None:
        raise ValueError(f"Unknown threshold method '{method}'")
    try:
        return float(fn(vals))
    except RuntimeError as exc:  # threshold_minimum on unimodal histograms
        raise ValueError(f"{method} threshold failed: {exc}") from exc


def threshold_range(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Binary mask of pixels with lo ≤ value ≤ hi (ImageJ manual threshold)."""
    g = to_gray(img)
    return (g >= lo) & (g <= hi)


def auto_threshold(img: np.ndarray, method: str = "Otsu", dark_objects: bool = True
                   ) -> tuple[float, float, np.ndarray]:
    """Auto threshold → (lo, hi, mask). ``dark_objects`` selects pixels ≤ t, else > t."""
    g = to_gray(img)
    t = threshold_value(g, method)
    gmin, gmax = float(g.min()), float(g.max())
    if dark_objects:
        lo, hi = gmin, t
        mask = g <= t
    else:
        lo, hi = t, gmax
        mask = g > t
    return lo, hi, mask


# --------------------------------------------------------------------------------------
# Binary operations
# --------------------------------------------------------------------------------------


def _fp(radius: int) -> np.ndarray:
    return ndi.generate_binary_structure(2, 2) if radius <= 1 else _disk(radius)


def binary_erode(mask: np.ndarray, iterations: int = 1, radius: int = 1) -> np.ndarray:
    return ndi.binary_erosion(mask, structure=_fp(radius), iterations=max(1, iterations),
                              border_value=1)


def binary_dilate(mask: np.ndarray, iterations: int = 1, radius: int = 1) -> np.ndarray:
    return ndi.binary_dilation(mask, structure=_fp(radius), iterations=max(1, iterations))


def binary_open(mask: np.ndarray, iterations: int = 1, radius: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask, iterations, radius), iterations, radius)


def binary_close(mask: np.ndarray, iterations: int = 1, radius: int = 1) -> np.ndarray:
    m = np.pad(mask, iterations * radius + 1, mode="edge")
    out = ndi.binary_erosion(ndi.binary_dilation(m, structure=_fp(radius), iterations=iterations),
                             structure=_fp(radius), iterations=iterations, border_value=1)
    p = iterations * radius + 1
    return out[p:-p, p:-p]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(mask)


def remove_small(mask: np.ndarray, min_size_px: int) -> np.ndarray:
    """Remove connected objects (8-connectivity) smaller than ``min_size_px`` pixels."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask.copy()
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size_px
    keep[0] = False
    return keep[lab]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize as _sk

    return _sk(mask.astype(bool))


def outline(mask: np.ndarray) -> np.ndarray:
    """One-pixel outlines of objects (ImageJ 'Outline')."""
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def watershed_separate(mask: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """Separate touching particles (ImageJ 'Watershed'): distance-transform watershed with
    1-px separating lines. Returns a new binary mask."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    m = mask.astype(bool)
    if not m.any():
        return m.copy()
    dist = ndi.distance_transform_edt(m)
    dist_s = ndi.gaussian_filter(dist, 0.7)
    coords = peak_local_max(dist_s, min_distance=max(1, int(min_distance)), labels=ndi.label(m)[0],
                            exclude_border=False)
    markers = np.zeros(m.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    lab = watershed(-dist_s, markers, mask=m, watershed_line=True)
    return m & (lab > 0)


# --------------------------------------------------------------------------------------
# ROI masks and measurements
# --------------------------------------------------------------------------------------


def polygon_mask(shape: tuple[int, int], xs, ys) -> np.ndarray:
    """Boolean mask of a polygon given vertex x/y coordinates.

    ROI coordinates follow the ImageJ convention: (0, 0) is the top-left *corner* of the first
    pixel, so pixel (i, j) has its centre at (j + 0.5, i + 0.5); a pixel is inside when its centre is.
    """
    from skimage.draw import polygon2mask

    pts = np.column_stack([np.asarray(ys, float) - 0.5, np.asarray(xs, float) - 0.5])
    return polygon2mask(shape[:2], pts)


def ellipse_vertices(x0: float, y0: float, x1: float, y1: float, n: int = 180) -> np.ndarray:
    """Vertices (N×2, x/y) of the ellipse inscribed in the bounding box (x0,y0)–(x1,y1)."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = abs(x1 - x0) / 2, abs(y1 - y0) / 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


def rect_vertices(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]], float)


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(pts: np.ndarray, closed: bool = True) -> float:
    d = np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def polygon_centroid(pts: np.ndarray) -> tuple[float, float]:
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    return float(((x + xn) * cross).sum() / (6 * a)), float(((y + yn) * cross).sum() / (6 * a))


def feret(points: np.ndarray) -> dict:
    """Feret (caliper) diameters of a point set (N×2, x/y).

    Returns ``max``, ``min`` (minimum caliper width, rotating calipers on the convex hull),
    ``angle`` (of the max Feret, degrees 0–180 counter-clockwise from +x with y up, as in ImageJ)
    and ``max_pts`` endpoints.
    """
    pts = np.unique(np.asarray(points, float), axis=0)
    if len(pts) == 1:
        return {"max": 0.0, "min": 0.0, "angle": 0.0, "max_pts": (pts[0], pts[0])}
    if len(pts) >= 3:
        from scipy.spatial import ConvexHull, QhullError

        try:
            hull = pts[ConvexHull(pts).vertices]
        except (QhullError, ValueError):
            hull = pts
    else:
        hull = pts
    d = hull[:, None, :] - hull[None, :, :]
    dist2 = (d ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    fmax = float(math.sqrt(dist2[i, j]))
    p, q = hull[i], hull[j]
    if p[0] > q[0]:
        p, q = q, p
    ang = math.degrees(math.atan2(-(q[1] - p[1]), q[0] - p[0])) % 180.0
    if len(hull) < 3:
        fmin = 0.0
    else:
        fmin = np.inf
        n = len(hull)
        for k in range(n):
            a, b = hull[k], hull[(k + 1) % n]
            e = b - a
            L = math.hypot(*e)
            if L == 0:
                continue
            dists = np.abs(e[0] * (hull[:, 1] - a[1]) - e[1] * (hull[:, 0] - a[0])) / L
            fmin = min(fmin, float(dists.max()))
    return {"max": fmax, "min": float(fmin), "angle": ang, "max_pts": (p, q)}


def intensity_stats(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Mean / std / min / max / median / integrated density of grey values inside ``mask``."""
    g = to_gray(img)
    vals = g[mask].astype(float) if mask is not None else g.astype(float).ravel()
    if vals.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "median": np.nan,
                "n_px": 0, "intden": 0.0}
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()), "median": float(np.median(vals)),
            "n_px": int(vals.size), "intden": float(vals.sum())}


def line_angle(p0, p1) -> float:
    """ImageJ line angle: degrees counter-clockwise from +x, image y pointing down (−180…180]."""
    return math.degrees(math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]))


def angle_between(p_a, vertex, p_b) -> float:
    """Angle (degrees, 0–180) at ``vertex`` between rays to ``p_a`` and ``p_b`` (ImageJ angle tool)."""
    v1 = np.asarray(p_a, float) - np.asarray(vertex, float)
    v2 = np.asarray(p_b, float) - np.asarray(vertex, float)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return 0.0
    return float(math.degrees(math.acos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def measure_line(img: np.ndarray, pts, cal: Calibration, width: float = 1.0) -> dict:
    """Measure a straight (2 points) or segmented (≥3 points) line: length, angle, intensity."""
    pts = np.asarray(pts, float)
    L_px = polygon_perimeter(pts, closed=False)
    out = {"Length": cal.length(L_px), "Length (px)": L_px}
    if len(pts) == 2:
        out["Angle (°)"] = line_angle(pts[0], pts[1])
    vals = []
    for a, b in zip(pts[:-1], pts[1:]):
        _, v = line_profile(img, a, b, width=width, cal=cal)
        vals.append(v)
    v = np.concatenate(vals) if vals else np.array([np.nan])
    out.update({"Mean": float(np.nanmean(v)), "StdDev": float(np.nanstd(v)),
                "Min": float(np.nanmin(v)), "Max": float(np.nanmax(v))})
    out["BX"] = cal.length(float(pts[:, 0].min()))
    out["BY"] = cal.length(float(pts[:, 1].min()))
    return out


def measure_polygon(img: np.ndarray, pts, cal: Calibration) -> dict:
    """ImageJ 'Measure' for an area ROI given as polygon vertices (x, y px).

    Area/perimeter from the vertex geometry; intensity statistics from the rasterised mask;
    centroid; Feret max/min/angle; circularity 4πA/P².
    """
    pts = np.asarray(pts, float)
    A = polygon_area(pts)
    P = polygon_perimeter(pts, closed=True)
    mask = polygon_mask(img.shape[:2], pts[:, 0], pts[:, 1])
    st = intensity_stats(img, mask)
    cx, cy = polygon_centroid(pts)
    fr = feret(pts)
    circ = min(1.0, 4 * math.pi * A / P ** 2) if P > 0 else 0.0
    return {"Area": cal.area(A), "Perimeter": cal.length(P), "Mean": st["mean"],
            "StdDev": st["std"], "Min": st["min"], "Max": st["max"], "Median": st["median"],
            "IntDen": st["intden"], "X": cal.length(cx), "Y": cal.length(cy),
            "Feret": cal.length(fr["max"]), "MinFeret": cal.length(fr["min"]),
            "FeretAngle (°)": fr["angle"], "Circ.": circ, "Area (px²)": A}


def line_profile(img: np.ndarray, p0, p1, width: float = 1.0, cal: Calibration | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Intensity profile along a line from p0=(x0, y0) to p1=(x1, y1).

    ``width`` px: values are averaged perpendicular to the line (ImageJ wide line profile).
    Returns (distance in calibrated units, values). One sample per pixel of length, the last
    distance equals the line length. Coordinates use the ImageJ corner convention (pixel centres
    at +0.5), as for ROIs.
    """
    from skimage.measure import profile_line

    g = to_gray(img).astype(float)
    src = (float(p0[1]) - 0.5, float(p0[0]) - 0.5)
    dst = (float(p1[1]) - 0.5, float(p1[0]) - 0.5)
    vals = profile_line(g, src, dst, linewidth=max(1, int(round(width))), order=1,
                        mode="reflect", reduce_func=np.mean)
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    dist_px = np.linspace(0.0, L, len(vals))
    cal = cal or Calibration()
    return cal.length(dist_px), vals


# --------------------------------------------------------------------------------------
# Scale bar
# --------------------------------------------------------------------------------------


def nice_number(x: float) -> float:
    """Largest 'nice' value (1, 2, 5 × 10^n) ≤ x."""
    if x <= 0:
        return 1.0
    e = math.floor(math.log10(x))
    for m in (5, 2, 1):
        if m * 10 ** e <= x:
            return float(m * 10 ** e)
    return float(10 ** e)


def auto_scale_bar_length(image_width_px: int, cal: Calibration, fraction: float = 0.2) -> float:
    """A nice scale-bar length (calibrated units) of about ``fraction`` of the image width."""
    return nice_number(cal.length(image_width_px * fraction))


def format_length(value: float, unit: str) -> str:
    """Pretty label like '10 µm' or '2.5 mm'."""
    if value == int(value):
        return f"{int(value)} {unit}"
    return f"{value:g} {unit}"


# ---- fonts --------------------------------------------------------------------------

_FONT_CACHE: dict = {}


def annotation_font_path() -> str | None:
    """Path of a TrueType font with µ, ², °, Å … for burned-in text: DejaVu Sans (ships with
    matplotlib, so no extra dependency). None if it cannot be found."""
    if "path" not in _FONT_CACHE:
        path = None
        try:
            from matplotlib import font_manager

            path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"),
                                         fallback_to_default=True)
            if not path or not os.path.isfile(path):
                path = None
        except Exception:  # noqa: BLE001 - matplotlib missing/broken → Pillow default
            path = None
        _FONT_CACHE["path"] = path
    return _FONT_CACHE["path"]


def annotation_font(size: int):
    """PIL font of ``size`` px for burned-in labels (DejaVu Sans; Pillow default as last resort)."""
    from PIL import ImageFont

    size = max(4, int(size))
    key = ("font", size)
    if key not in _FONT_CACHE:
        path = annotation_font_path()
        font = None
        if path:
            try:
                font = ImageFont.truetype(path, size)
            except OSError:
                font = None
        if font is None:
            try:
                font = ImageFont.load_default(size)
            except TypeError:  # very old Pillow
                font = ImageFont.load_default()
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


def font_has_glyph(font, ch: str) -> bool:
    """True if ``font`` renders ``ch`` with a real glyph (not the .notdef 'tofu' box)."""
    try:
        from fontTools.ttLib import TTFont  # optional, exact

        path = getattr(font, "path", None)
        if path:
            cmap = TTFont(path, lazy=True).getBestCmap()
            return ord(ch) in cmap
    except Exception:  # noqa: BLE001
        pass
    import numpy as _np

    def mask_of(c):
        m = font.getmask(c)
        return _np.array(m, _np.uint8).reshape(m.size[1], m.size[0]) if m.size[0] * m.size[1] else \
            _np.zeros((0, 0), _np.uint8)

    m = mask_of(ch)
    if m.size == 0 or not m.any():
        return False
    notdef = mask_of("\U0010fffd")  # private-use code point: no font maps it
    return not (m.shape == notdef.shape and _np.array_equal(m, notdef))


def draw_text(img: np.ndarray, text: str, pos: tuple[float, float], size: int = 24,
              color: str = "yellow", outline: bool = True) -> np.ndarray:
    """Burn ``text`` into a copy of ``img`` (8-bit gray or RGB) at ``pos`` (x, y top-left) using
    the µ-capable annotation font."""
    from PIL import Image, ImageColor, ImageDraw

    base = img if img.dtype == np.uint8 else to_8bit(img)
    im = Image.fromarray(base[..., :3] if base.ndim == 3 else base).convert("RGB")
    d = ImageDraw.Draw(im)
    col = ImageColor.getrgb(color)
    lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
    kw = {}
    if outline:
        kw = {"stroke_width": max(1, int(size) // 16), "stroke_fill": (0, 0, 0) if lum > 127 else (255, 255, 255)}
    d.text((float(pos[0]), float(pos[1])), text, fill=col, font=annotation_font(size), **kw)
    return np.array(im)


# ---- scale-bar style presets -----------------------------------------------------------

#: Keys of a scale-bar / annotation style (see :func:`burn_scale_bar` for meaning).
SCALEBAR_STYLE_KEYS = ("position", "length_rule", "length", "fraction", "color", "font_px",
                       "font_frac", "thickness", "thickness_frac", "background", "box_color",
                       "unit_text", "show_label", "margin")

BUILTIN_SCALEBAR_PRESETS: dict[str, dict] = {
    "Thesis (white bar, black box, bottom-right)": {
        "position": "lower right", "length_rule": "auto", "fraction": 0.2, "color": "white",
        "font_frac": 0.035, "thickness_frac": 0.012, "background": True, "box_color": "black",
        "unit_text": "", "show_label": True},
    "Journal (black bar, no box, bottom-left)": {
        "position": "lower left", "length_rule": "auto", "fraction": 0.15, "color": "black",
        "font_frac": 0.03, "thickness_frac": 0.01, "background": False, "box_color": "white",
        "unit_text": "", "show_label": True},
    "Minimal": {
        "position": "lower right", "length_rule": "auto", "fraction": 0.1, "color": "white",
        "font_frac": 0.025, "thickness_frac": 0.006, "background": False, "box_color": "black",
        "unit_text": "", "show_label": True},
}


def styles_dir() -> str:
    """Folder for user style presets: ``$LABSTUDIO_HOME/styles`` or ``~/.labstudio/styles``."""
    home = os.environ.get("LABSTUDIO_HOME") or os.path.join(os.path.expanduser("~"), ".labstudio")
    d = os.path.join(home, "styles")
    os.makedirs(d, exist_ok=True)
    return d


def _preset_file(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("_") or "preset"
    return os.path.join(styles_dir(), f"scalebar_{safe}.json")


def save_scalebar_preset(name: str, style: dict) -> str:
    """Save a named scale-bar style as JSON; returns the file path."""
    import json

    if not name.strip():
        raise ValueError("Preset name is empty")
    data = {"name": name, "kind": "scalebar",
            "style": {k: style[k] for k in SCALEBAR_STYLE_KEYS if k in style}}
    path = _preset_file(name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    return path


def list_scalebar_presets() -> list[str]:
    """Built-in preset names followed by user presets (sorted)."""
    import glob
    import json

    users = []
    for p in sorted(glob.glob(os.path.join(styles_dir(), "scalebar_*.json"))):
        try:
            with open(p, encoding="utf-8") as fh:
                users.append(json.load(fh).get("name") or os.path.basename(p)[9:-5])
        except (OSError, ValueError):
            continue
    return list(BUILTIN_SCALEBAR_PRESETS) + [u for u in users if u not in BUILTIN_SCALEBAR_PRESETS]


def load_scalebar_preset(name: str) -> dict:
    """Style dict of a built-in or user preset (KeyError if unknown)."""
    import json

    if name in BUILTIN_SCALEBAR_PRESETS:
        return dict(BUILTIN_SCALEBAR_PRESETS[name])
    path = _preset_file(name)
    if not os.path.isfile(path):
        raise KeyError(f"No scale-bar preset named '{name}'")
    with open(path, encoding="utf-8") as fh:
        return dict(json.load(fh).get("style", {}))


def delete_scalebar_preset(name: str) -> bool:
    path = _preset_file(name)
    if name in BUILTIN_SCALEBAR_PRESETS or not os.path.isfile(path):
        return False
    os.remove(path)
    return True


def resolve_scalebar_style(style: dict, shape: tuple[int, int], cal: "Calibration") -> dict:
    """Concrete pixel geometry for a style on an image of ``shape``: length (unit), length_px,
    label, font_px, thickness, margin, colours, position."""
    h, w = shape[:2]
    length = style.get("length") or None
    if length and style.get("length_rule", "fixed" if length else "auto") != "auto":
        L = float(length)
    else:  # 'auto': nice length ≈ fraction × image width
        L = auto_scale_bar_length(w, cal, float(style.get("fraction") or 0.2))
    font_px = style.get("font_px") or max(10, int(round(h * float(style.get("font_frac") or 0.035))))
    thick = style.get("thickness") or max(2, int(round(h * float(style.get("thickness_frac") or 0.012))))
    unit_text = style.get("unit_text") or cal.unit
    return {"length": L, "length_px": cal.to_px(L), "position": style.get("position", "lower right"),
            "color": style.get("color", "white"), "font_px": int(font_px), "thickness": int(thick),
            "background": bool(style.get("background", True)),
            "box_color": style.get("box_color") or None,
            "label": format_length(L, unit_text) if style.get("show_label", True) else "",
            "margin": style.get("margin") or max(4, int(min(h, w) * 0.025))}


def burn_scale_bar(img: np.ndarray, cal: Calibration, length: float | None = None,
                   position: str = "lower right", color: str = "white", font_size: int = 0,
                   thickness: int = 0, background: bool = True, label: bool = True,
                   margin: int = 0, box_color: str | None = None, unit_text: str | None = None
                   ) -> np.ndarray:
    """Return a copy of ``img`` (as 8-bit gray or RGB) with a scale bar drawn into the pixels.

    ``length`` in calibrated units (auto 'nice' length if None); ``position`` one of
    'lower right', 'lower left', 'upper right', 'upper left'; ``color`` a PIL colour name;
    ``box_color`` background box colour (default: black for light bars, white for dark ones);
    ``unit_text`` overrides the unit in the label (e.g. 'μm' or 'micron').
    The label is rendered with DejaVu Sans (:func:`annotation_font`) so µ, ², ° render correctly.
    """
    from PIL import Image, ImageDraw, ImageColor

    h, w = img.shape[:2]
    if length is None or length <= 0:
        length = auto_scale_bar_length(w, cal)
    bar_px = max(1, int(round(cal.to_px(length))))
    font_size = font_size or max(10, int(h * 0.035))
    thickness = thickness or max(2, int(h * 0.012))
    margin = margin or max(4, int(min(h, w) * 0.025))
    rgb_col = ImageColor.getrgb(color)
    box_rgb = ImageColor.getrgb(box_color) if box_color else None
    is_gray_col = rgb_col[0] == rgb_col[1] == rgb_col[2] and (
        box_rgb is None or box_rgb[0] == box_rgb[1] == box_rgb[2])
    if img.ndim == 2:
        base = img if img.dtype == np.uint8 else to_8bit(img)
        im = Image.fromarray(base, "L" if is_gray_col else None)
        if not is_gray_col:
            im = im.convert("RGB")
    else:
        base = img if img.dtype == np.uint8 else apply_window(img, float(img.min()), float(img.max()))
        im = Image.fromarray(base[..., :3], "RGB")
    draw = ImageDraw.Draw(im)
    font = annotation_font(font_size)
    text = format_length(length, unit_text or cal.unit) if label else ""
    if text:
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    else:
        tw = th = 0
        tb = (0, 0, 0, 0)
    gap = max(2, thickness // 2)
    block_w = max(bar_px, tw)
    block_h = thickness + (gap + th if text else 0)
    right = "right" in position
    lower = "lower" in position
    bx0 = w - margin - block_w if right else margin
    by0 = h - margin - block_h if lower else margin
    fill = rgb_col if im.mode == "RGB" else rgb_col[0]
    if background:
        lum = 0.299 * rgb_col[0] + 0.587 * rgb_col[1] + 0.114 * rgb_col[2]
        bgc = box_rgb or ((0, 0, 0) if lum > 127 else (255, 255, 255))
        pad = max(3, thickness)
        draw.rectangle([bx0 - pad, by0 - pad, bx0 + block_w + pad, by0 + block_h + pad],
                       fill=bgc if im.mode == "RGB" else bgc[0])
    bar_x0 = bx0 + (block_w - bar_px) // 2
    draw.rectangle([bar_x0, by0, bar_x0 + bar_px - 1, by0 + thickness - 1], fill=fill)
    if text:
        tx = bx0 + (block_w - tw) // 2 - tb[0]
        ty = by0 + thickness + gap - tb[1]
        draw.text((tx, ty), text, fill=fill, font=font)
    return np.array(im)
