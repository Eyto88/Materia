"""Data import for LabStudio (pure Python, no Qt).

Reads text/CSV exports from thermal-analysis and mechanical-testing instruments
(Netzsch Proteus, TA Instruments TRIOS / Universal Analysis, Mettler STARe,
Instron Bluehill, Zwick testXpert, Shimadzu Trapezium, generic whitespace
tables), guesses what every column is, and builds a :class:`Dataset` whose
values are converted to the canonical units of :mod:`labstudio.core.units`.

Pipeline::

    raw = read_table(path)            # sniff encoding/delimiter/decimal/preamble/header/units row
    specs = guess_mapping(raw)        # list[ColumnSpec] (name, dimension, unit, include)
    kind = guess_kind(raw)            # generic | dsc | tensile | compression | cyclic
    ds = build_dataset(raw, specs, name, kind, metadata)

or simply ``load_file(path)``.  Column mappings can be stored as JSON presets
(``save_preset`` / ``load_preset`` / ``list_presets``) in
``$LABSTUDIO_HOME/presets`` (default ``~/.labstudio/presets``); a preset may also carry
``analysis_defaults`` (per-analysis parameter dicts) that end up in
``Dataset.metadata['analysis_defaults']``.

Spreadsheets (.xlsx/.xlsm/.xls/.ods) are read with pandas (openpyxl / xlrd / odfpy) and fed
through the same sniffing pipeline (``read_table(path, sheet=...)``, ``list_sheets(path)``).

No silent guesses: every :class:`ColumnSpec` has ``guessed``/``reason``; ``build_dataset``
lists unconfirmed guesses (and a defaulted DSC heating rate, header-vs-measured scan-rate
mismatch, ...) in ``Dataset.metadata['warnings']``.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import units as U
from .dataset import DATASET_KINDS, Channel, Dataset

__all__ = [
    "RawTable", "ColumnSpec", "Preset", "read_table", "read_text", "guess_mapping", "guess_kind",
    "build_dataset", "default_axes", "metadata_from_raw", "save_preset", "load_preset",
    "list_presets", "delete_preset", "presets_dir", "apply_preset", "load_file",
    "export_dataset_csv", "cross_section_area", "MECHANICAL_KINDS", "list_sheets",
    "TEXT_EXTENSIONS", "SPREADSHEET_EXTENSIONS", "DATA_EXTENSIONS", "UnsupportedFileError",
    "DEFAULT_HEATING_RATE_K_PER_MIN", "measured_heating_rate", "import_warnings",
]

MECHANICAL_KINDS = ("tensile", "compression", "cyclic")

#: Plain-text exports read by the sniffing parser.
TEXT_EXTENSIONS = (".csv", ".txt", ".tsv", ".dat", ".asc", ".exp", ".prn", ".tab", ".text")
#: Spreadsheets read through pandas.
SPREADSHEET_EXTENSIONS = (".xlsx", ".xlsm", ".xls", ".ods")
#: Everything the importer accepts (use for file dialogs / drag-and-drop filters).
DATA_EXTENSIONS = TEXT_EXTENSIONS + SPREADSHEET_EXTENSIONS
#: Known instrument/binary formats that cannot be read directly (export them as text/CSV first).
_BINARY_EXTENSIONS = {
    ".xlsb": "Excel binary workbook (.xlsb) — save it as .xlsx or CSV",
    ".ngb": "Netzsch Proteus binary file — use File ▸ Export data ▸ ASCII in Proteus",
    ".ngb-dsc": "Netzsch Proteus binary file — use File ▸ Export data ▸ ASCII in Proteus",
    ".tri": "TA TRIOS project — export the run as text/CSV in TRIOS",
    ".001": "TA Universal Analysis binary file — export it as text (File ▸ Export data)",
    ".is_tens": "Instron Bluehill binary method/sample — export the raw data as CSV",
    ".is_comp": "Instron Bluehill binary method/sample — export the raw data as CSV",
    ".is_cyclic": "Instron Bluehill binary method/sample — export the raw data as CSV",
    ".zs2": "Zwick testXpert binary file — export the series as text/CSV",
    ".zp2": "Zwick testXpert binary file — export the series as text/CSV",
    ".docx": "Word document", ".doc": "Word document", ".pdf": "PDF document", ".pptx": "PowerPoint file",
    ".png": "image — open it in the Micrographs tab", ".jpg": "image — open it in the Micrographs tab",
    ".jpeg": "image — open it in the Micrographs tab", ".tif": "image — open it in the Micrographs tab",
    ".tiff": "image — open it in the Micrographs tab", ".bmp": "image — open it in the Micrographs tab",
    ".zip": "zip archive — extract it first", ".7z": "archive — extract it first", ".lstudio": "LabStudio session — use File ▸ Open session",
}
DEFAULT_HEATING_RATE_K_PER_MIN = 10.0
#: tolerated relative difference between the header heating rate and the one measured from time
RATE_MISMATCH_TOL = 0.10


class UnsupportedFileError(ValueError):
    """Raised for files the importer cannot read (binary, unknown format, missing optional package)."""

_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")
_DELIMITERS = ("\t", ";", ",", "|", " ")   # " " means "any run of whitespace"
_SNIFF_LINES = 3000


# --------------------------------------------------------------------------- data classes
@dataclass
class RawTable:
    """A parsed-but-uninterpreted instrument table.

    ``df`` has one column per header (named exactly as in ``headers``) holding floats
    (non-numeric cells -> NaN). ``units`` holds the units row (if the file had one)
    aligned with ``headers``. ``meta`` holds metadata pairs found in the preamble plus
    derived canonical entries (``mass_mg``, ``heating_rate_K_per_min``, ``sample_name``,
    ``diameter_mm``, ``width_mm``, ``thickness_mm``, ``gauge_length_mm``, ``area_mm2``).
    """
    df: pd.DataFrame
    headers: list[str]
    meta_lines: list[str] = field(default_factory=list)
    delimiter: str = ","
    decimal: str = "."
    encoding: str = "utf-8"
    header_row: int | None = None      # 0-based line index of the header row, None if absent
    units: list[str | None] = field(default_factory=list)
    units_row: int | None = None
    data_start: int = 0                 # 0-based line index of the first data row
    meta: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    sheet: str | None = None            # spreadsheet: sheet that was read
    sheets: list[str] = field(default_factory=list)   # spreadsheet: all sheet names
    #: per source column: "confirmed" (unit read from file), "guessed" (needs confirmation) or
    #: "none" (dimensionless/count, nothing to confirm). Filled by :func:`guess_mapping`.
    unit_confidence: dict[str, str] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.df)


@dataclass
class ColumnSpec:
    """How to interpret one source column."""
    source: str          # header in RawTable.headers
    name: str            # column name in the Dataset
    dimension: str       # key of units.CANONICAL
    unit: str            # unit the SOURCE values are in
    include: bool = True
    guessed: bool = False   # dimension/unit was guessed (spelled-out word, no unit, ambiguous) -> confirm
    reason: str = ""        # human-readable explanation of the guess


@dataclass
class Preset:
    """A saved column mapping (e.g. for one machine's export format)."""
    name: str
    kind: str
    specs: list[ColumnSpec]
    metadata: dict[str, Any] = field(default_factory=dict)
    #: optional per-analysis defaults, e.g. {"dsc": {"prominence": 0.02, "baseline": "linear"}}
    analysis_defaults: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- text / encoding
def read_text(path: str | os.PathLike) -> tuple[str, str]:
    """Return ``(text, encoding)`` trying BOMs, then utf-8, utf-16 (NUL heuristic), cp1252, latin-1."""
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace"), "utf-16"
    head = data[:4000]
    if head and head.count(b"\x00") > len(head) // 4:
        # UTF-16 without BOM: NULs on odd positions -> LE, even -> BE
        odd = head[1::2].count(b"\x00")
        even = head[0::2].count(b"\x00")
        enc = "utf-16-le" if odd >= even else "utf-16-be"
        try:
            return data.decode(enc), "utf-16"
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8", "cp1252"):
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1"), "latin-1"


# --------------------------------------------------------------------------- numbers
_NUM_RE = re.compile(
    r"^[+-]?(?:\d{1,3}(?:[.,'   ]\d{3})+(?:[.,]\d*)?|\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?$")
_NAN_WORDS = {"nan", "na", "n/a", "inf", "-inf", "+inf", "infinity", "-infinity"}


def _clean_cell(s: str) -> str:
    return s.strip().strip('"').strip("'").strip().replace("−", "-")


def _is_num(s: str) -> bool:
    return bool(_NUM_RE.match(s))


def _cell_kind(s: str) -> int:
    """1 numeric, 0 empty/neutral, -1 text."""
    s = _clean_cell(s)
    if not s or s.lower() in _NAN_WORDS or s in ("-", "--", "---"):
        return 0
    return 1 if _is_num(s) else -1


def _to_numeric(col: pd.Series, decimal: str) -> pd.Series:
    """Vectorised string -> float with given decimal mark and thousands separators removed."""
    s = col.astype("string").str.strip().str.strip("\"'").str.strip()
    s = s.str.replace("−", "-", regex=False)
    s = s.str.replace(r"(?<=\d)[ '  ](?=\d{3}(?:\D|$))", "", regex=True)
    if decimal == ",":
        thousands = s.str.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+(?:,\d*)?(?:[eE][+-]?\d+)?", na=False)
        s = s.where(~thousands, s.str.replace(".", "", regex=False))
        s = s.str.replace(",", ".", regex=False)
    else:
        thousands = s.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?", na=False)
        s = s.where(~thousands, s.str.replace(",", "", regex=False))
    return pd.to_numeric(s, errors="coerce").astype(float)


def _parse_float(text: str) -> float | None:
    """Parse the first number in ``text`` (accepting decimal comma)."""
    m = re.search(r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?", text.replace("−", "-"))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


# --------------------------------------------------------------------------- splitting
def _split(line: str, delim: str) -> list[str]:
    if delim == " ":
        return line.split()
    if '"' not in line:
        return line.split(delim)
    try:
        return next(csv.reader([line], delimiter=delim, quotechar='"', skipinitialspace=True))
    except (csv.Error, StopIteration):
        return line.split(delim)


def _strip_hash(line: str) -> str:
    return line.lstrip("#").strip()


@dataclass
class _Run:
    start: int
    end: int          # exclusive
    score: float
    ncols: int


def _best_run(lines: list[str], delim: str) -> _Run | None:
    """Best run of consecutive numeric rows for delimiter ``delim``.

    *Strong* rows are mostly numeric. *Weak* rows have at least one number but also text
    (e.g. ``0,ERR,x`` or a sample-label column); they never start a run on their own, but are
    tolerated inside a run and adopted at its start when they have the run's field count, so a
    first data row containing text does not push the header detection one row down.
    """
    best: _Run | None = None
    cur_start: int | None = None
    cur_score = 0.0
    cur_counts: Counter = Counter()
    pending: list[tuple[int, int]] = []      # contiguous weak rows just before the current line

    def close(end: int) -> None:
        nonlocal best
        if cur_start is None or not cur_counts:
            return
        ncols = cur_counts.most_common(1)[0][0]
        n = end - cur_start
        score = cur_score * (1.0 if n >= 3 else 0.3)
        if best is None or score > best.score:
            best = _Run(cur_start, end, score, ncols)

    for i, line in enumerate(lines):
        stripped = line.strip()
        cls = 0
        fields: list[str] = []
        row_score = 0.0
        if stripped and not stripped.startswith("#"):
            fields = _split(stripped if delim != "\t" else line.strip("\r\n "), delim)
            kinds = [_cell_kind(f) for f in fields]
            n_num = kinds.count(1)
            n_txt = kinds.count(-1)
            if n_num >= 1 and n_num >= 0.6 * (n_num + n_txt):
                cls = 1
                row_score = n_num - 2 * n_txt
            elif n_num >= 1:
                cls = 2
        if cls == 1:
            if cur_start is None:
                start = i
                for j, nf in reversed(pending):
                    if nf != len(fields):
                        break
                    start = j
                cur_start, cur_score, cur_counts = start, 0.0, Counter()
            cur_score += row_score
            cur_counts[len(fields)] += 1
            pending = []
        elif cls == 2:
            if cur_start is not None and len(fields) == cur_counts.most_common(1)[0][0]:
                continue                     # tolerated inside the run
            if cur_start is not None:
                close(i)
                cur_start = None
            pending.append((i, len(fields)))
        else:
            close(i)
            cur_start = None
            pending = []
    close(len(lines))
    return best


def _detect_decimal(rows: Iterable[list[str]], delim: str) -> str:
    if delim == ",":
        return "."
    n_comma = n_dot = 0
    for fields in rows:
        for f in fields:
            f = _clean_cell(f)
            if not _is_num(f):
                continue
            ic, idot = f.rfind(","), f.rfind(".")
            if ic > idot:
                n_comma += 1
            elif idot > ic:
                n_dot += 1
    return "," if n_comma > n_dot else "."


_UNIT_TOKEN_RE = re.compile(r"^[\(\[].*[\)\]]$")


def _is_unit_token(tok: str) -> bool:
    t = _clean_cell(tok)
    if not t:
        return True
    if _UNIT_TOKEN_RE.match(t):
        return True
    return U.dimension_of(t) is not None or t.lower() in ("sec", "1", "a.u.", "au")


def _looks_like_units_row(fields: list[str]) -> bool:
    toks = [_clean_cell(f) for f in fields]
    nonempty = [t for t in toks if t]
    if not nonempty:
        return False
    if any(_is_num(t) for t in nonempty):
        return False
    good = sum(_is_unit_token(t) for t in nonempty)
    return good == len(nonempty)


def _looks_like_header(fields: list[str], ncols: int) -> bool:
    toks = [_clean_cell(f) for f in fields]
    nonempty = [t for t in toks if t]
    if not nonempty:
        return False
    if len(nonempty) < max(1, min(2, ncols), math.ceil(ncols * 0.5)):
        return False
    n_num = sum(_is_num(t) for t in nonempty)
    return n_num <= 0.34 * len(nonempty)


def _token_spans(line: str, pattern: str = r"\S+") -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(pattern, line)]


def _split_ws_header(line: str, ncols: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Split a whitespace-delimited header line into ``ncols`` names when possible."""
    line = _strip_hash(line) if line.lstrip().startswith("#") else line
    simple = _token_spans(line)
    if len(simple) == ncols:
        return [t for _, _, t in simple], [(a, b) for a, b, _ in simple]
    wide = _token_spans(line, r"\S+(?: \S+)*")
    if len(wide) == ncols:
        return [t for _, _, t in wide], [(a, b) for a, b, _ in wide]
    # merge unit tokens '(s)' / '[mm]' into the preceding name
    merged: list[list] = []
    for a, b, t in simple:
        if merged and _UNIT_TOKEN_RE.match(t):
            merged[-1][1] = b
            merged[-1][2] += " " + t
        else:
            merged.append([a, b, t])
    if len(merged) == ncols:
        return [m[2] for m in merged], [(m[0], m[1]) for m in merged]
    best = wide if abs(len(wide) - ncols) < abs(len(simple) - ncols) else simple
    return [t for _, _, t in best], [(a, b) for a, b, _ in best]


def _align_ws_units(unit_line: str, spans: list[tuple[int, int]], ncols: int) -> list[str | None]:
    toks = _token_spans(unit_line)
    out: list[str | None] = [None] * ncols
    if len(toks) == ncols or not spans:
        for i, (_, _, t) in enumerate(toks[:ncols]):
            out[i] = t
        return out
    centres = [(a + b) / 2 for a, b in spans]
    for a, b, t in toks:
        c = (a + b) / 2
        j = int(np.argmin([abs(c - x) for x in centres]))
        if j < ncols and out[j] is None:
            out[j] = t
    return out


def _dedupe(names: list[str]) -> list[str]:
    seen: Counter = Counter()
    out = []
    for n in names:
        base = n
        seen[base] += 1
        if seen[base] > 1:
            n = f"{base}_{seen[base]}"
            while n in seen:
                seen[base] += 1
                n = f"{base}_{seen[base]}"
            seen[n] += 1
        out.append(n)
    return out


# --------------------------------------------------------------------------- metadata
def _meta_pairs(lines: list[str], delim: str) -> list[tuple[str, str, str]]:
    """(key, value, unit) triples from preamble lines."""
    out = []
    for raw in lines:
        line = _strip_hash(raw)
        if not line:
            continue
        key = val = unit = ""
        fields = _split(line, delim) if delim != " " else [line]
        fields = [_clean_cell(f) for f in fields]
        while fields and not fields[-1]:
            fields.pop()
        if len(fields) >= 2 and ":" not in fields[0]:
            key, val = fields[0], fields[1]
            unit = fields[2] if len(fields) > 2 else ""
        elif ":" in line:
            key, _, val = line.partition(":")
        elif "=" in line:
            key, _, val = line.partition("=")
        elif "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) >= 2:
                key, val = parts[0], parts[1]
                unit = parts[2] if len(parts) > 2 else ""
        key, val, unit = (_clean_cell(key).rstrip(":").strip(),
                          _clean_cell(val).lstrip(";,\t ").strip(), _clean_cell(unit))
        if key:
            out.append((key, val, unit))
    return out


_KEY_UNIT_PATTERNS = (
    r"/\s*\(([^()]+)\)\s*$",              # 'HEATING RATE /(K/min)'
    r"\(([^()]+)\)\s*$",                   # 'Diameter (mm)'
    r"\[([^\[\]]+)\]\s*$",                 # 'Mass [mg]'
    r"/\s*([^/\s()]+(?:/[^/\s()]+)?)\s*$",  # 'SAMPLE MASS /mg', 'Rate/K/min' (not 'dT/dt')
    r"\s((?:K|°C|mg|g|mm|µm|um)(?:/(?:min|s))?)\s*$",   # 'Heating rate K/min', 'Mass mg'
)


def _key_unit(key: str) -> tuple[str, str | None]:
    """Split 'SAMPLE MASS /mg', 'HEATING RATE /(K/min)' or 'Diameter (mm)' into (key, unit)."""
    k = key.strip()
    for pat in _KEY_UNIT_PATTERNS:
        m = re.search(pat, k)
        if m and U.dimension_of(m.group(1)) is not None and m.start() > 0:
            return k[: m.start()].strip(" /").lower(), U.normalize_symbol(m.group(1))
    return k.lower(), None


def _value_unit(val: str, unit_field: str, key_unit: str | None) -> tuple[float | None, str | None]:
    num = _parse_float(val)
    unit = None
    if unit_field and U.dimension_of(unit_field) is not None:
        unit = U.normalize_symbol(unit_field)
    if unit is None:
        m = re.search(r"[\d.,]\s*([A-Za-zµμ°²/%]+[A-Za-z²]*)\s*$", val)
        if m and U.dimension_of(m.group(1)) is not None:
            unit = U.normalize_symbol(m.group(1))
    return num, unit or key_unit


def _conv(v: float, unit: str | None, default: str, target: str, dim: str) -> float | None:
    unit = unit or default
    try:
        return float(U.convert(v, unit, target, dim))
    except (ValueError, KeyError):
        try:
            return float(U.convert(v, default, target, dim))
        except (ValueError, KeyError):
            return None


_MECH_KEYS = [
    ("area_mm2", r"^(?:cross[\s-]*section(?:al)?(?:\s*area)?|area|csa|s0|s_0|so|querschnitt|specimen area)$",
     "area", "mm²"),
    ("diameter_mm", r"^(?:diameter|dia|d0|d_0|ø|⌀|durchmesser|specimen diameter|wire diameter)$", "length", "mm"),
    ("width_mm", r"^(?:width|b0|b_0|breite|specimen width)$", "length", "mm"),
    ("thickness_mm", r"^(?:thickness|a0|a_0|dicke|depth|specimen thickness)$", "length", "mm"),
    ("gauge_length_mm", r"^(?:gauge\s*length|gage\s*length|gauge|gl|l0|l_0|lo|messlänge|"
                        r"extensometer gauge length|initial gauge length)$", "length", "mm"),
]

_RATE_RE = re.compile(
    r"([+-]?\d+(?:[.,]\d+)?)\s*\(?\s*(K|°C|ºC|C|deg\s*C|℃)\s*(?:/\s*(min|s|h)\b|·?\s*(min|s)\s*(?:-1|⁻¹))",
    re.I)
#: (priority, pattern) for metadata keys that hold the DSC heating rate
_RATE_KEYS: list[tuple[int, str]] = [
    (4, r"heat(ing)?\s*rate|heizrate|aufheiz"),
    (3, r"scan(ning)?\s*rate|^ramp(\s*rate)?$|^dt\s*/\s*dt$|^β$|^beta$|^rate\s*β$|heat(ing)?\s*/\s*cool(ing)?\s*rate"),
    (2, r"^(temperature\s*)?rate(\s*\d+)?$"),
    (1, r"cool(ing)?\s*rate|abk(ü|ue)hl"),
]


def _rate_to_k_per_min(num: float, unit: str | None) -> float | None:
    if unit is None or U.dimension_of(unit) != "heating_rate":
        return None
    try:
        return abs(float(U.convert(num, unit, "K/min", "heating_rate")))
    except ValueError:
        return None


def _derive_meta(pairs: list[tuple[str, str, str]], lines: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    mass_candidates: list[tuple[int, float]] = []
    rate_candidates: list[tuple[int, float, str, str]] = []
    canonical_keys = {"mass_mg", "heating_rate_K_per_min", "area_mm2", "gauge_length_mm", "diameter_mm",
                      "width_mm", "thickness_mm"}
    for key, val, unit in pairs:
        meta.setdefault(key, f"{val} {unit}".strip())
        if key in canonical_keys:
            v = _parse_float(val)
            if v is not None:
                meta[key] = v
                if key == "mass_mg":
                    mass_candidates.append((3, v))
            continue
        if key == "sample_name" and val:
            meta["sample_name"] = val
            continue
        k, kunit = _key_unit(key)
        kl = k.lower()
        num, vunit = _value_unit(val, unit, kunit)
        # -- mass
        if re.search(r"\b(mass|weight|einwaage|masse)\b", kl) or kl in ("size", "sample size"):
            if num is not None and not re.search(r"ref|crucible|pan|tara|tiegel", kl):
                mg = _conv(num, vunit, "mg", "mg", "mass")
                if mg is not None:
                    mass_candidates.append((2 if "sample" in kl else 1, mg))
        elif re.search(r"\bsample\b", kl):
            m = re.search(r"([\d.,]+)\s*(mg|g)\b", val)
            if m:
                mg = _conv(float(m.group(1).replace(",", ".")), m.group(2), "mg", "mg", "mass")
                if mg is not None:
                    mass_candidates.append((1, mg))
        # -- sample name
        if "sample_name" not in meta and re.fullmatch(
                r"(sample|sample\s*name|sample\s*id|specimen|specimen\s*(label|name|id)|probe|"
                r"probenname|name)", kl) and val:
            name = re.split(r",\s*[\d.,]+\s*(?:mg|g)\b", val)[0].strip(" ,")
            if name:
                meta["sample_name"] = name
        # -- heating rate (Netzsch '#HEATING RATE /(K/min)', TA 'Heating rate', Mettler 'dT/dt', 'Rate')
        if num is not None:
            prio = next((p for p, pat in _RATE_KEYS if re.search(pat, kl.strip())), None)
            if prio is not None:
                r = _rate_to_k_per_min(num, vunit) or _rate_to_k_per_min(num, kunit)
                note = ""
                if r is None and prio >= 3 and vunit is None and kunit is None and 0 < abs(num) <= 1000:
                    r = abs(num)
                    note = f"heating rate '{key}: {val}' has no unit in the file — assumed K/min"
                if r is not None and r > 0:
                    rate_candidates.append((prio, r, f"{key}: {val} {unit}".strip(), note))
        # -- mechanical geometry
        kclean = re.sub(r"\s+", " ", kl).strip()
        for mkey, pattern, dim, target in _MECH_KEYS:
            if mkey not in meta and num is not None and re.fullmatch(pattern, kclean):
                v = _conv(num, vunit, target, target, dim)
                if v is not None and v > 0:
                    meta[mkey] = v
    if mass_candidates:
        meta["mass_mg"] = max(mass_candidates, key=lambda t: t[0])[1]
    if "heating_rate_K_per_min" in meta:          # canonical key written by LabStudio's own export
        meta.setdefault("heating_rate_text", f"heating_rate_K_per_min: {meta['heating_rate_K_per_min']}")
    elif rate_candidates:
        prio, r, text, note = max(rate_candidates, key=lambda t: t[0])
        meta["heating_rate_K_per_min"] = r
        meta["heating_rate_text"] = text
        if note:
            meta["heating_rate_note"] = note
    else:
        for line in lines:
            m = _RATE_RE.search(line)
            if m:
                unit = {"c": "°C", "℃": "°C", "ºc": "°C"}.get(m.group(2).lower().replace(" ", ""), m.group(2))
                if unit.lower().startswith("deg"):
                    unit = "°C"
                per = m.group(3) or m.group(4) or "min"
                r = _rate_to_k_per_min(float(m.group(1).replace(",", ".")),
                                       f"{'K' if unit.upper() == 'K' else '°C'}/{per.lower()}")
                if r:
                    meta["heating_rate_K_per_min"] = r
                    meta["heating_rate_text"] = _strip_hash(line).strip()
                    break
    if "area_mm2" not in meta:
        a = cross_section_area(diameter=meta.get("diameter_mm"), width=meta.get("width_mm"),
                               thickness=meta.get("thickness_mm"))
        if a:
            meta["area_mm2"] = a
    return meta


def cross_section_area(*, area: float | None = None, diameter: float | None = None,
                       width: float | None = None, thickness: float | None = None) -> float | None:
    """Cross-section in mm² from area, round Ø, or rectangular width × thickness (all mm)."""
    if area:
        return float(area)
    if diameter:
        return math.pi * float(diameter) ** 2 / 4.0
    if width and thickness:
        return float(width) * float(thickness)
    return None


# --------------------------------------------------------------------------- spreadsheets
_SHEET_ENGINES = {".xlsx": ("openpyxl", "openpyxl"), ".xlsm": ("openpyxl", "openpyxl"),
                  ".xls": ("xlrd", "xlrd"), ".ods": ("odf", "odfpy")}


def _excel_engine(path: Path) -> str:
    ext = path.suffix.lower()
    module, package = _SHEET_ENGINES[ext]
    try:
        __import__(module)
    except ImportError as exc:
        raise UnsupportedFileError(
            f"Reading {ext} files needs the Python package '{package}', which is not installed.\n"
            f"Install it with:  python -m pip install {package}\n"
            f"(or re-run the LabStudio launcher, which installs requirements.txt), "
            f"or save the sheet as CSV.") from exc
    return module


def _open_workbook(path: Path):
    engine = _excel_engine(path)
    try:
        return pd.ExcelFile(path, engine=engine)
    except Exception as exc:  # zipfile.BadZipFile, xlrd errors, ...
        raise UnsupportedFileError(
            f"Could not read '{path.name}' as a {path.suffix.lower()} workbook ({type(exc).__name__}: {exc}). "
            f"The file may be damaged or not really a spreadsheet; try opening it in Excel and "
            f"saving it again, or export the data as CSV.") from exc


def list_sheets(path: str | os.PathLike) -> list[str]:
    """Sheet names of a spreadsheet (``[]`` for text files)."""
    p = Path(path)
    if p.suffix.lower() not in SPREADSHEET_EXTENSIONS:
        return []
    with _open_workbook(p) as xl:
        return [str(n) for n in xl.sheet_names]


def _cell_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return repr(v)
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, np.floating):
        return "" if np.isnan(v) else repr(float(v))
    return str(v).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _sheet_lines(xl, sheet: str) -> list[str]:
    df = xl.parse(sheet, header=None, dtype=object)
    df = df.dropna(axis=1, how="all")
    return ["\t".join(_cell_text(v) for v in row) for row in df.itertuples(index=False, name=None)]


def _read_spreadsheet(path: Path, sheet: str | int | None) -> RawTable:
    with _open_workbook(path) as xl:
        names = [str(n) for n in xl.sheet_names]
        if not names:
            raise UnsupportedFileError(f"'{path.name}' contains no sheets")
        if sheet is not None:
            if isinstance(sheet, int):
                if not 0 <= sheet < len(names):
                    raise ValueError(f"Sheet index {sheet} out of range; sheets: {names}")
                chosen = names[sheet]
            elif str(sheet) in names:
                chosen = str(sheet)
            else:
                raise ValueError(f"Sheet {sheet!r} not found in '{path.name}'; sheets: {names}")
            lines = _sheet_lines(xl, chosen)
        else:
            # pick the sheet with the strongest numeric table
            chosen, lines, best_score = names[0], None, -math.inf
            for n in names:
                ls = _sheet_lines(xl, n)
                run = _best_run(ls[:_SNIFF_LINES], "\t")
                score = run.score if run else -math.inf
                if lines is None or score > best_score:
                    chosen, lines, best_score = n, ls, score
    try:
        raw = _parse_lines(lines or [], encoding=path.suffix.lower().lstrip("."), path=path,
                           delimiters=("\t",))
    except ValueError as exc:
        raise ValueError(f"No numeric data found in sheet '{chosen}' of {path.name}"
                         + (f" (sheets: {', '.join(names)})" if len(names) > 1 else "")) from exc
    raw.sheet, raw.sheets = chosen, names
    return raw


# --------------------------------------------------------------------------- read_table
def _check_text(text: str, path: Path) -> None:
    head = text[:8000]
    if not head.strip():
        raise ValueError(f"No numeric data found in {path} (the file is empty)")
    bad = sum(1 for ch in head if (ord(ch) < 32 and ch not in "\t\r\n\f\v") or ch == "\ufffd")
    if bad > max(3, 0.02 * len(head)):
        raise UnsupportedFileError(
            f"'{path.name}' does not look like a text data file (binary content). "
            f"Export the data from the instrument software as text/CSV or Excel.")


def read_table(path: str | os.PathLike, sheet: str | int | None = None) -> RawTable:
    """Parse an instrument export into a :class:`RawTable`.

    Text files: sniffs encoding, delimiter (tab ; , | or whitespace), decimal mark, preamble
    lines, the header row, an optional units row under it, trailing non-numeric junk and
    thousands separators. Spreadsheets (.xlsx/.xlsm/.xls/.ods): ``sheet`` selects a sheet by
    name or index (default: the sheet with the strongest numeric table); see :func:`list_sheets`.

    Raises :class:`UnsupportedFileError` (a ``ValueError``) for binary / unsupported files.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    if ext in SPREADSHEET_EXTENSIONS:
        return _read_spreadsheet(p, sheet)
    if ext in _BINARY_EXTENSIONS:
        raise UnsupportedFileError(
            f"'{p.name}': {ext} files cannot be imported ({_BINARY_EXTENSIONS[ext]}). "
            f"Supported: {', '.join(DATA_EXTENSIONS)}.")
    text, encoding = read_text(p)
    _check_text(text, p)
    return _parse_lines(text.splitlines(), encoding=encoding, path=p)


def _parse_lines(lines: list[str], *, encoding: str, path: Path | str | None,
                 delimiters: tuple[str, ...] = _DELIMITERS) -> RawTable:
    sample = lines[:_SNIFF_LINES]

    best: tuple[str, _Run] | None = None
    for d in delimiters:
        run = _best_run(sample, d)
        if run and (best is None or run.score > best[1].score):
            best = (d, run)
    if best is None:
        raise ValueError(f"No numeric data found in {path}")
    delim, run = best

    def split_line(line: str) -> list[str]:
        return _split(line.strip("\r\n ") if delim == "\t" else line.strip(), delim)

    data_start = run.start
    ncols = run.ncols
    sample_rows = [split_line(l) for l in lines[run.start:min(run.end, run.start + 500)]]
    decimal = _detect_decimal(sample_rows, delim)

    # ---- header / units rows (look back over at most 2 blank lines)
    def prev_nonblank(i: int) -> int | None:
        j, blanks = i - 1, 0
        while j >= 0 and not lines[j].strip():
            j -= 1
            blanks += 1
            if blanks > 2:
                return None
        return j if j >= 0 else None

    header_row = units_row = None
    headers: list[str] | None = None
    units_list: list[str | None] = [None] * ncols
    spans: list[tuple[int, int]] = []
    cand = prev_nonblank(data_start)
    if cand is not None:
        fields = split_line(_strip_hash(lines[cand]) if lines[cand].lstrip().startswith("#") else lines[cand])
        if _looks_like_units_row(fields):
            h = prev_nonblank(cand)
            if h is not None:
                hline = _strip_hash(lines[h]) if lines[h].lstrip().startswith("#") else lines[h]
                hfields = (split_line(hline) if delim != " " else _split_ws_header(hline, ncols)[0])
                if _looks_like_header(hfields, ncols):
                    units_row, header_row = cand, h
        if header_row is None and _looks_like_header(
                fields if delim != " " else _split_ws_header(lines[cand], ncols)[0], ncols):
            header_row = cand
    if header_row is not None:
        hline = lines[header_row]
        hline = _strip_hash(hline) if hline.lstrip().startswith("#") else hline
        if delim == " ":
            headers, spans = _split_ws_header(hline, ncols)
        else:
            headers = [_clean_cell(f) for f in split_line(hline)]
    if units_row is not None:
        uline = lines[units_row]
        if delim == " ":
            units_list = _align_ws_units(uline, spans, ncols)
        else:
            units_list = [_clean_cell(f) or None for f in split_line(uline)]

    meta_end = header_row if header_row is not None else (units_row if units_row is not None else data_start)
    meta_lines = [l.rstrip("\r\n") for l in lines[:meta_end] if l.strip()]
    pairs = _meta_pairs(meta_lines, delim)
    meta = _derive_meta(pairs, meta_lines)

    # ---- data rows (rest of file; junk filtered below)
    rows = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(split_line(line))
    # drop a leading empty column (e.g. TA Universal Analysis lines start with a tab)
    if rows and sum(1 for r in rows[:200] if r and not r[0].strip()) >= 0.9 * min(len(rows), 200):
        rows = [r[1:] if r and not r[0].strip() else r for r in rows]
        ncols = max(1, ncols - 1)
        if headers and len(headers) > ncols and not headers[0]:
            headers = headers[1:]
        if units_row is not None and len(units_list) > ncols and not units_list[0]:
            units_list = units_list[1:]
    width = max(ncols, len(headers or []))
    rows = [(r + [""] * width)[:width] for r in rows]

    # header names: from header row, else TA 'SigN' metadata, else generic
    sig = [(int(m.group(1)), v) for k, v, _ in pairs if (m := re.fullmatch(r"sig(\d+)", k.strip().lower()))]
    if sig and (headers is None or len(sig) == ncols):
        sig.sort()
        headers = [v for _, v in sig]
    if headers is None:
        headers = []
    headers = [h if h else f"Column {i + 1}" for i, h in enumerate((headers + [""] * width)[:width])]
    headers = _dedupe(headers)
    units_list = (list(units_list) + [None] * width)[:width]

    df = pd.DataFrame(rows, columns=headers, dtype="string") if rows else pd.DataFrame(columns=headers)
    num = pd.DataFrame({h: _to_numeric(df[h], decimal) for h in headers}) if rows else \
        pd.DataFrame({h: pd.Series(dtype=float) for h in headers})
    # filter trailing junk / repeated headers / footers. A row is kept when enough numeric
    # columns hold numbers ("full" row); a partial row (e.g. one NaN, or 'ERR' in one cell) is
    # kept too when it has no more text cells than numbers and lies before the last full row.
    numeric_cols = [h for h in headers if num[h].notna().mean() > 0.5] if len(num) else []
    if numeric_cols:
        need = max(1, math.ceil(0.6 * len(numeric_cols)))
        n_ok = num[numeric_cols].notna().sum(axis=1)
        full = n_ok >= need
        keep = full.copy()
        partial = (~full) & (n_ok >= 1)
        if partial.any() and full.any():
            idx = partial[partial].index
            last_full = full[full].index.max()
            idx = idx[idx < last_full]
            if len(idx):
                sub = df.loc[idx, numeric_cols]
                n_text = pd.Series(0, index=idx)
                for h in numeric_cols:
                    c = sub[h].str.strip().str.strip("\"'").str.strip().str.lower()
                    neutral = c.isna() | (c == "") | c.isin(_NAN_WORDS | {"-", "--", "---"})
                    n_text += (~neutral & num.loc[idx, h].isna()).astype(int)
                ok = n_text <= n_ok.loc[idx]
                keep.loc[ok[ok].index] = True
        num = num.loc[keep].reset_index(drop=True)
    # drop empty columns that also have no header (e.g. trailing delimiters)
    drop = [h for h in headers if h.startswith("Column ") and num[h].isna().all()
            and len(headers) > 1]
    if drop:
        idx = [headers.index(h) for h in drop]
        num = num.drop(columns=drop)
        units_list = [u for i, u in enumerate(units_list) if i not in idx]
        headers = [h for h in headers if h not in drop]

    return RawTable(df=num, headers=headers, meta_lines=meta_lines, delimiter=delim, decimal=decimal,
                    encoding=encoding, header_row=header_row, units=units_list, units_row=units_row,
                    data_start=data_start, meta=meta, path=None if path is None else str(path))


# --------------------------------------------------------------------------- mapping
_EXTRA_HINTS: list[tuple[str, str]] = [
    (r"kraft", "force"),
    (r"spannung", "stress"),
    (r"dehnung", "strain"),
    (r"\bweg\b|standardweg|traverse|verformung|deform", "displacement"),
    (r"zeit|^t$", "time"),
    (r"temperatur|^t[sr]$|^tr?ef", "temperature"),
    (r"w(ä|ae)rmestrom|^value$", "heat_flow"),
    (r"zyklus|cycle|^n$|count", "count"),
    (r"^(index|nr\.?|no\.?|point|#|row)$", "count"),
]

#: parenthesised words that qualify a quantity rather than give its unit ('Heat Flow (Normalized)')
_QUALIFIERS = {
    "normalized", "normalised", "corrected", "raw", "smoothed", "filtered", "sample", "reference", "ref",
    "program", "programme", "furnace", "total", "average", "avg", "mean", "abs", "absolute", "true",
    "eng", "engineering", "nominal", "measured", "set", "setpoint", "actual", "extension", "extensometer",
    "crosshead", "channel", "left", "right", "upper", "lower", "calculated", "calc", "derivative",
    "reversing", "nonreversing", "rev", "nonrev", "tension", "compression", "tensile", "compressive",
    "load", "position", "displacement", "strain", "stress", "time", "temperature", "heat", "flow",
    "cycle", "segment", "step", "dsc", "tga", "dta", "signal", "sensor", "cell", "specimen", "unit",
    "units", "peak", "baseline", "gauge", "gage", "video", "laser", "lvdt", "dic", "local", "global",
}


@dataclass
class _ColumnGuess:
    name: str
    dimension: str
    unit: str
    guessed: bool = False
    reason: str = ""


def _is_qualifier(text: str) -> bool:
    t = text.strip().lower()
    if " " in t or "," in t:
        return True                      # multi-word: 'Channel 1', 'Heat Flow (T4P)'...
    t = t.strip(".")
    return t in _QUALIFIERS or bool(re.fullmatch(r"(ch|channel|sensor|s)?\d+|[a-z]\d+", t))


def _analyse_column_ex(header: str, unit_row: str | None) -> _ColumnGuess:
    """Interpret one source column: name, dimension, unit and whether that is a guess."""
    h = header.strip()
    info = U.parse_header_ex(h)
    name, unit, dim = info.name, info.unit, info.dimension
    reasons: list[str] = []
    guessed = False
    unknown_unit = None
    if info.unit_text and unit is None:
        if _is_qualifier(info.unit_text):
            # e.g. 'Heat Flow (Normalized)': parenthesised text is not a unit
            name, unit = h, None
            dim = U.hint_dimension(h)
        else:
            unknown_unit = info.unit_text          # unit-like but not convertible, e.g. 'fortnights'
    elif info.guessed:
        guessed = True
        reasons.append(info.reason)
    if unit_row:
        um = U.resolve_unit(unit_row)
        if um.symbol is not None and um.symbol != "":
            info2 = U.parse_header_ex(f"{name} ({um.symbol})")
            name, unit, dim = info2.name, info2.unit, info2.dimension
            unknown_unit = None
            guessed = um.guessed
            reasons = [f"units row: {um.reason}"] if um.guessed else []
        elif um.symbol is None and unit is None and _clean_cell(unit_row).strip("()[] ") and \
                _clean_cell(unit_row).strip("()[] ").lower() not in ("1", "a.u.", "au", "-", "#"):
            unknown_unit = _clean_cell(unit_row).strip("()[] ")
    if unknown_unit and unit is None:
        return _ColumnGuess(f"{name} [{unknown_unit}]", "dimensionless", "", True,
                            f"unit '{unknown_unit}' not recognised — imported as dimensionless, "
                            f"no unit conversion")
    if dim is None:
        low = name.lower()
        for pat, d in _EXTRA_HINTS:
            if re.search(pat, low):
                dim = d
                break
    if dim is None and unit:
        dim = U.dimension_of(unit)
    if dim is None:
        return _ColumnGuess(name or h, "dimensionless", "")
    if unit is None:
        unit = U.CANONICAL[dim]
        if dim not in ("count", "dimensionless"):
            guessed = True
            reasons.append(f"no unit in the file — {U.DIMENSION_LABELS.get(dim, dim).lower()} "
                           f"guessed from the header '{h}', unit assumed {unit}")
    if unit not in U.units_for(dim):
        udim = U.dimension_of(unit)
        if udim is not None:
            dim = udim
        else:
            return _ColumnGuess(f"{name} [{unit}]", "dimensionless", "", True,
                                f"unit '{unit}' not recognised — imported as dimensionless")
    return _ColumnGuess(name or h, dim, unit, guessed, "; ".join(r for r in reasons if r))


def _analyse_column(header: str, unit_row: str | None) -> tuple[str, str, str]:
    """Return (name, dimension, unit) for one source column (see :func:`_analyse_column_ex`)."""
    g = _analyse_column_ex(header, unit_row)
    return g.name, g.dimension, g.unit


def guess_mapping(raw: RawTable) -> list[ColumnSpec]:
    """Guess a :class:`ColumnSpec` for every column from header text and units row.

    Specs whose dimension/unit could not be read unambiguously from the file have
    ``guessed=True`` and a ``reason``; ``raw.unit_confidence`` is updated accordingly.
    """
    specs = []
    names_seen: Counter = Counter()
    for i, h in enumerate(raw.headers):
        ur = raw.units[i] if i < len(raw.units) else None
        base = re.sub(r"_\d+$", "", h)
        g = _analyse_column_ex(base if base != h and base in raw.headers else h, ur)
        name = g.name
        names_seen[name] += 1
        if names_seen[name] > 1:
            name = f"{name}_{names_seen[name]}"
        col = raw.df[h] if h in raw.df else pd.Series(dtype=float)
        include = bool(col.notna().any())
        specs.append(ColumnSpec(source=h, name=name, dimension=g.dimension, unit=g.unit, include=include,
                                guessed=g.guessed, reason=g.reason))
        raw.unit_confidence[h] = ("guessed" if g.guessed else
                                  "none" if g.dimension in ("count", "dimensionless") else "confirmed")
    return specs


def _count_turns(x: np.ndarray, frac: float = 0.3) -> int:
    """Number of reversals of ``x`` larger than ``frac`` of its range (hysteresis counting)."""
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return 0
    rng = float(np.nanmax(x) - np.nanmin(x))
    if rng <= 0:
        return 0
    thr = frac * rng
    turns = 0
    direction = 0
    ext = x[0]
    for v in x[1:]:
        if direction >= 0:
            if v > ext:
                ext = v
                direction = 1 if v - x[0] > thr or direction == 1 else direction
            elif ext - v > thr:
                if direction == 1:
                    turns += 1
                direction, ext = -1, v
        if direction == -1:
            if v < ext:
                ext = v
            elif v - ext > thr:
                turns += 1
                direction, ext = 1, v
    return turns


def guess_kind(raw: RawTable, specs: list[ColumnSpec] | None = None) -> str:
    """Guess dataset kind: ``dsc``, ``tensile``, ``compression``, ``cyclic`` or ``generic``."""
    specs = specs if specs is not None else guess_mapping(raw)
    inc = [s for s in specs if s.include]
    dims = {s.dimension for s in inc}
    if str(raw.meta.get("kind", "")).strip() in DATASET_KINDS and raw.meta.get("kind") != "generic":
        return str(raw.meta["kind"]).strip()
    hint = " ".join([Path(raw.path).stem if raw.path else ""] + raw.meta_lines[:60]).lower()
    if "temperature" in dims and dims & {"heat_flow", "specific_heat_flow"}:
        return "dsc"
    if dims & {"heat_flow", "specific_heat_flow"}:
        return "dsc"
    if not dims & {"force", "stress"}:
        return "generic"
    if any(s.dimension == "count" and re.search(r"cycle|zyklus", s.name.lower()) for s in inc):
        return "cyclic"
    ycol = next((s for s in inc if s.dimension == "stress"), None) or \
        next(s for s in inc if s.dimension == "force")
    xcol = next((s for s in inc if s.dimension == "strain"), None) or \
        next((s for s in inc if s.dimension == "displacement"), None)
    if xcol is not None:
        x = raw.df[xcol.source].to_numpy(float)
        if _count_turns(x) >= 1:
            return "cyclic"
    if re.search(r"compress|druck|compr\b", hint):
        return "compression"
    if re.search(r"cycl|fatigue", hint):
        return "cyclic"
    y = raw.df[ycol.source].to_numpy(float)
    y = y[np.isfinite(y)]
    if len(y) and np.nanmedian(y) < 0 and abs(np.nanmin(y)) > abs(np.nanmax(y)):
        return "compression"
    return "tensile"


# --------------------------------------------------------------------------- build
def default_axes(channels: dict[str, Channel], kind: str) -> tuple[str | None, str | None]:
    """Sensible default (x, y) column names for a kind."""
    def first(*dims: str) -> str | None:
        for d in dims:
            for c, ch in channels.items():
                if ch.dimension == d:
                    return c
        return None

    cols = list(channels)
    x = y = None
    if kind == "dsc":
        x = first("temperature", "time")
        y = first("specific_heat_flow", "heat_flow")
    elif kind in MECHANICAL_KINDS:
        if first("stress") and first("strain"):
            x, y = first("strain"), first("stress")
        else:
            x = first("displacement", "strain", "length")
            y = first("force", "stress")
    if x is None or y is None:
        non_time = [c for c in cols if channels[c].dimension not in ("time", "count")]
        if x is None:
            x = first("time") if len(cols) > 1 and first("time") else (cols[0] if cols else None)
        if y is None:
            y = next((c for c in non_time if c != x), None) or next((c for c in cols if c != x), None)
    return x, y


def metadata_from_raw(raw: RawTable, kind: str) -> dict[str, Any]:
    """Canonical metadata keys relevant to ``kind`` that were found in the file preamble."""
    keys = ["sample_name"]
    if kind == "dsc":
        keys += ["mass_mg", "heating_rate_K_per_min"]
    elif kind in MECHANICAL_KINDS:
        keys += ["area_mm2", "gauge_length_mm", "diameter_mm", "width_mm", "thickness_mm"]
    else:
        keys += ["mass_mg", "heating_rate_K_per_min", "area_mm2", "gauge_length_mm"]
    return {k: raw.meta[k] for k in keys if k in raw.meta}


def measured_heating_rate(T_degC: np.ndarray, t_s: np.ndarray) -> float | None:
    """Typical |dT/dt| in K/min measured from temperature (°C/K) and time (s) arrays.

    Robust to isotherms and heating+cooling programmes: median of the scan rates over the
    non-isothermal parts. None if it cannot be measured.
    """
    T = np.asarray(T_degC, float)
    t = np.asarray(t_s, float)
    ok = np.isfinite(T) & np.isfinite(t)
    T, t = T[ok], t[ok]
    if len(T) < 10:
        return None
    k = max(1, len(T) // 400)
    dT, dt = T[k:] - T[:-k], t[k:] - t[:-k]
    good = dt > 0
    if good.sum() < 5:
        return None
    r = np.abs(dT[good] / dt[good]) * 60.0
    hi = np.percentile(r, 95)
    if not np.isfinite(hi) or hi <= 0:
        return None
    r = r[r > 0.25 * hi]
    return float(np.median(r)) if len(r) else None


def import_warnings(specs: list[ColumnSpec]) -> list[str]:
    """Human-readable warnings for included columns whose unit/dimension is an unconfirmed guess."""
    out = []
    for s in specs:
        if s.include and s.guessed:
            unit = f" [{s.unit}]" if s.unit else ""
            out.append(f"Column '{s.name}'{unit}: unit not confirmed — {s.reason or 'guessed'}")
    return out


def _heating_rate_meta(raw: RawTable, meta: dict[str, Any], given: dict[str, Any], kind: str,
                       channels: dict[str, Channel], df: pd.DataFrame, warnings: list[str]) -> None:
    file_rate = raw.meta.get("heating_rate_K_per_min")
    user_rate = given.get("heating_rate_K_per_min")
    if user_rate is not None:
        src = given.get("heating_rate_source")
        if src not in ("header", "user", "default"):
            src = "header" if file_rate and math.isclose(float(user_rate), float(file_rate), rel_tol=1e-6) \
                else "user"
        meta["heating_rate_K_per_min"] = float(user_rate)
        meta["heating_rate_source"] = src
        if src == "default":
            warnings.append(f"heating rate not found in file — defaulted to "
                            f"{float(user_rate):g} K/min")
    elif file_rate:
        meta["heating_rate_K_per_min"] = float(file_rate)
        meta["heating_rate_source"] = "header"
    elif kind == "dsc":
        meta["heating_rate_K_per_min"] = DEFAULT_HEATING_RATE_K_PER_MIN
        meta["heating_rate_source"] = "default"
        warnings.append(f"heating rate not found in file — defaulted to {DEFAULT_HEATING_RATE_K_PER_MIN:g} K/min")
    if meta.get("heating_rate_source") == "header" and raw.meta.get("heating_rate_note"):
        warnings.append(str(raw.meta["heating_rate_note"]))
    if kind != "dsc":
        return
    tcol = next((c for c, ch in channels.items() if ch.dimension == "time"), None)
    Tcol = next((c for c, ch in channels.items() if ch.dimension == "temperature"), None)
    if tcol is None or Tcol is None:
        return
    measured = measured_heating_rate(df[Tcol].to_numpy(float), df[tcol].to_numpy(float))
    if measured is None:
        return
    meta["measured_heating_rate_K_per_min"] = measured
    rate = meta.get("heating_rate_K_per_min")
    if rate and abs(measured - rate) > RATE_MISMATCH_TOL * rate:
        src = meta.get("heating_rate_source")
        what = {"header": "in the file header", "user": "entered", "default": "defaulted"}.get(src, "")
        tunit = channels[tcol].original_unit or "s"
        warnings.append(
            f"heating rate {what} ({rate:g} K/min) differs from the scan rate measured from the time "
            f"column '{tcol}' ({measured:.3g} K/min) by more than {RATE_MISMATCH_TOL:.0%} — check the "
            f"time unit (read as {tunit}) and the heating rate")


def build_dataset(raw: RawTable, specs: list[ColumnSpec], name: str | None = None,
                  kind: str | None = None, metadata: dict[str, Any] | None = None) -> Dataset:
    """Build a :class:`Dataset` from included specs, converting values to canonical units.

    ``metadata`` (canonical units) overrides values found in the file preamble. Also records:

    * ``metadata['warnings']`` — unconfirmed unit guesses (``ColumnSpec.guessed``), a defaulted DSC
      heating rate, header-vs-measured scan-rate mismatch (plus any ``warnings`` passed in);
    * ``metadata['heating_rate_source']`` — ``'header' | 'user' | 'default'`` (when a rate is set);
    * ``metadata['source_sha256']`` and ``metadata['import']`` (delimiter, decimal, encoding,
      header row, sheet, column specs) for provenance.
    """
    kind = kind or guess_kind(raw, specs)
    if kind not in DATASET_KINDS:
        raise ValueError(f"Unknown dataset kind {kind!r}")
    data: dict[str, np.ndarray] = {}
    channels: dict[str, Channel] = {}
    used: Counter = Counter()
    for s in specs:
        if not s.include:
            continue
        if s.source not in raw.df.columns:
            raise KeyError(f"Column {s.source!r} not in file (columns: {raw.headers})")
        dim = s.dimension if s.dimension in U.CANONICAL else "dimensionless"
        vals = raw.df[s.source].to_numpy(dtype=float)
        unit = s.unit if s.unit is not None else U.CANONICAL[dim]
        if unit != U.CANONICAL[dim]:
            vals = np.asarray(U.to_canonical(vals, unit, dim), dtype=float)
        cname = (s.name or s.source).strip() or s.source
        used[cname] += 1
        if used[cname] > 1:
            cname = f"{cname}_{used[cname]}"
        data[cname] = vals
        channels[cname] = Channel(cname, dim, U.CANONICAL[dim], original_unit=unit)
    if not data:
        raise ValueError("No columns selected for import")
    df = pd.DataFrame(data)
    df = df.loc[df.notna().any(axis=1)].reset_index(drop=True)
    given = dict(metadata or {})
    warnings: list[str] = [str(w) for w in (given.pop("warnings", None) or [])]
    meta = metadata_from_raw(raw, kind)
    meta.pop("heating_rate_K_per_min", None)
    meta.update({k: v for k, v in given.items() if v is not None and k not in
                 ("heating_rate_K_per_min", "heating_rate_source")})
    meta.setdefault("source_meta", {k: v for k, v in raw.meta.items()
                                    if isinstance(v, str)})
    warnings.extend(w for w in import_warnings(specs) if w not in warnings)
    if kind == "dsc" or given.get("heating_rate_K_per_min") is not None or \
            (raw.meta.get("heating_rate_K_per_min") and kind not in MECHANICAL_KINDS):
        _heating_rate_meta(raw, meta, given, kind, channels, df, warnings)
    meta["warnings"] = warnings
    sha = _file_sha256(raw.path)
    if sha:
        meta["source_sha256"] = sha
    meta["import"] = {
        "delimiter": raw.delimiter, "decimal": raw.decimal, "encoding": raw.encoding,
        "header_row": raw.header_row, "units_row": raw.units_row, "data_start": raw.data_start,
        "sheet": raw.sheet,
        "specs": [{"source": s.source, "name": s.name, "dimension": s.dimension, "unit": s.unit,
                   "include": s.include, "guessed": bool(s.guessed)} for s in specs],
    }
    dx, dy = default_axes(channels, kind)
    ds_name = name or (meta.get("sample_name") if not raw.path else Path(raw.path).stem) or "Dataset"
    return Dataset(name=ds_name, data=df, channels=channels, kind=kind, source_path=raw.path,
                   metadata=meta, default_x=dx, default_y=dy)


def _file_sha256(path: str | None) -> str | None:
    try:
        from .provenance import file_sha256
        return file_sha256(path)
    except Exception:  # pragma: no cover - provenance is best effort
        return None


# --------------------------------------------------------------------------- presets
def presets_dir(base: str | os.PathLike | None = None) -> Path:
    """Preset directory: ``base`` if given, else ``$LABSTUDIO_HOME/presets``, else ``~/.labstudio/presets``."""
    if base is not None:
        p = Path(base)
    else:
        home = os.environ.get("LABSTUDIO_HOME")
        p = (Path(home) if home else Path.home() / ".labstudio") / "presets"
    return p


def _safe_name(name: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", name).strip(" .")
    return s or "preset"


def _preset_path(path_or_name: str | os.PathLike, directory: str | os.PathLike | None) -> Path:
    p = Path(path_or_name)
    s = str(path_or_name)
    if p.suffix.lower() == ".json" or os.sep in s or (os.altsep and os.altsep in s):
        return p
    return presets_dir(directory) / f"{_safe_name(s)}.json"


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def save_preset(path_or_name: str | os.PathLike, specs: list[ColumnSpec], kind: str,
                metadata: dict[str, Any] | None = None, *,
                directory: str | os.PathLike | None = None,
                analysis_defaults: dict[str, Any] | None = None) -> Path:
    """Save a column mapping as JSON; a bare name goes into :func:`presets_dir`. Returns the file path.

    ``analysis_defaults`` (optional): per-analysis parameter dicts, e.g.
    ``{"dsc": {"prominence": 0.02, "baseline": "linear"}, "mechanical": {...}, "cyclic": {...}}``.
    Saving a preset counts as confirming its units, so specs are stored with ``guessed=False``.
    """
    path = _preset_path(path_or_name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = path.stem if (Path(str(path_or_name)).suffix.lower() == ".json" or os.sep in str(path_or_name)) \
        else str(path_or_name)
    spec_dicts = []
    for sp in specs:
        d = asdict(sp)
        d["guessed"], d["reason"] = False, ""
        spec_dicts.append(d)
    payload = {"labstudio_preset": 2, "name": name, "kind": kind,
               "specs": spec_dicts, "metadata": _jsonable(dict(metadata or {}))}
    if analysis_defaults:
        payload["analysis_defaults"] = _jsonable(dict(analysis_defaults))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_preset(path_or_name: str | os.PathLike, *, directory: str | os.PathLike | None = None) -> Preset:
    """Load a preset by name (from :func:`presets_dir`) or file path."""
    path = _preset_path(path_or_name, directory)
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    specs = [ColumnSpec(**{k: s[k] for k in ("source", "name", "dimension", "unit", "include") if k in s})
             for s in d.get("specs", [])]
    return Preset(name=d.get("name", path.stem), kind=d.get("kind", "generic"), specs=specs,
                  metadata=d.get("metadata", {}) or {},
                  analysis_defaults=d.get("analysis_defaults", {}) or {})


def list_presets(directory: str | os.PathLike | None = None) -> list[str]:
    """Names of saved presets (sorted)."""
    d = presets_dir(directory)
    if not d.is_dir():
        return []
    names = []
    for f in sorted(d.glob("*.json")):
        try:
            names.append(json.loads(f.read_text(encoding="utf-8")).get("name", f.stem))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(names, key=str.lower)


def delete_preset(name: str, *, directory: str | os.PathLike | None = None) -> bool:
    path = _preset_path(name, directory)
    if path.exists():
        path.unlink()
        return True
    return False


def apply_preset(raw: RawTable, preset: Preset | list[ColumnSpec]) -> list[ColumnSpec]:
    """Map a preset's specs onto ``raw``'s columns (by source header, else by position).

    Columns not covered by the preset keep their guessed mapping.
    """
    pspecs = preset.specs if isinstance(preset, Preset) else list(preset)
    guessed = guess_mapping(raw)
    by_source = {s.source: s for s in pspecs}

    def mapped(g: ColumnSpec, p: ColumnSpec) -> ColumnSpec:
        # a saved preset is a confirmed mapping; a plain spec list keeps its own guess flags
        return ColumnSpec(g.source, p.name, p.dimension, p.unit, p.include,
                          guessed=bool(getattr(p, "guessed", False)), reason=getattr(p, "reason", ""))

    if any(h in by_source for h in raw.headers):
        return [mapped(g, by_source[g.source]) if g.source in by_source else g for g in guessed]
    if len(pspecs) == len(guessed):
        return [mapped(g, p) for g, p in zip(guessed, pspecs)]
    return guessed


# --------------------------------------------------------------------------- convenience
def load_file(path: str | os.PathLike, preset: Preset | str | None = None, *, kind: str | None = None,
              metadata: dict[str, Any] | None = None, name: str | None = None,
              directory: str | os.PathLike | None = None, sheet: str | int | None = None) -> Dataset:
    """Read, auto-map (or apply ``preset``) and build a Dataset without any GUI.

    Specimen geometry found in the file wins over geometry stored in a preset (a preset for
    'Instron wire' must not give specimen 2 the diameter of specimen 1); explicit ``metadata``
    wins over both. A preset's ``analysis_defaults`` go to ``metadata['analysis_defaults']``.
    For DSC without a heating rate, 10 K/min is used and a warning is recorded.
    """
    raw = read_table(path, sheet=sheet)
    meta: dict[str, Any] = {}
    if preset is not None:
        if not isinstance(preset, Preset):
            preset = load_preset(preset, directory=directory)
        specs = apply_preset(raw, preset)
        kind = kind or preset.kind
        kind = kind or guess_kind(raw, specs)
        file_meta = metadata_from_raw(raw, kind)
        pmeta = dict(preset.metadata)
        geom = ("area_mm2", "diameter_mm", "width_mm", "thickness_mm")
        if any(k in file_meta for k in geom):
            for k in geom:
                pmeta.pop(k, None)
        meta.update({k: v for k, v in pmeta.items() if k not in file_meta})
        if preset.analysis_defaults:
            meta["analysis_defaults"] = dict(preset.analysis_defaults)
    else:
        specs = guess_mapping(raw)
    meta.update(metadata or {})
    return build_dataset(raw, specs, name=name, kind=kind or guess_kind(raw, specs), metadata=meta)


def export_dataset_csv(ds: Dataset, path: str | os.PathLike, columns: list[str] | None = None,
                       include_metadata: bool = True) -> Path:
    """Write ``ds`` as CSV with unit-bearing headers ('Force (N)'); metadata as '# key: value' lines."""
    path = Path(path)
    cols = columns or ds.columns()
    out = pd.DataFrame({ds.label(c): ds.values(c) for c in cols})
    with path.open("w", encoding="utf-8", newline="") as fh:
        if include_metadata:
            fh.write(f"# name: {ds.name}\n# kind: {ds.kind}\n")
            for k, v in ds.metadata.items():
                if isinstance(v, (int, float, str)) and "\n" not in str(v):
                    fh.write(f"# {k}: {v}\n")
            warns = ds.metadata.get("warnings") or []
            if warns:
                fh.write("# warnings: " + " | ".join(str(w).replace("\n", " ") for w in warns) + "\n")
        out.to_csv(fh, index=False, lineterminator="\n")
    return path
