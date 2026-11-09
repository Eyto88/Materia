"""Batch analysis: run the same DSC / tensile / compression / cyclic analysis on many datasets.

Pure Python (no Qt). One :class:`BatchRow` per dataset with the key metrics (unit-bearing column
names), its warnings and an optional *parameter* (e.g. anneal temperature, strain amplitude,
thickness) parsed from the dataset name with a regex or read from a metadata key. Repeats with the
same parameter can be summarised as mean ± SD (:func:`group_stats`) and any metric can be turned into
a derived "metric vs parameter" Dataset (:func:`metric_vs_parameter_dataset`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from . import analysis as A
from .dataset import Dataset
from .provenance import provenance

__all__ = [
    "BATCH_KINDS", "PARAMETER_PRESETS", "BatchRow", "BatchResult", "batch_analyse", "group_stats",
    "metric_vs_parameter_dataset", "parse_parameter", "dataset_parameter", "mech_arrays", "dsc_columns",
    "geometry_from_metadata",
]

BATCH_KINDS = ("dsc", "tensile", "compression", "cyclic")

#: ready-made regexes (first capture group = the number)
PARAMETER_PRESETS: dict[str, str] = {
    "Temperature (°C)": r"(-?\d+(?:[.,]\d+)?)\s*°?\s*C(?![a-zA-Z])",
    "Strain amplitude (%)": r"(\d+(?:[.,]\d+)?)\s*(?:%|pct)",
    "Thickness (mm)": r"(\d+(?:[.,]\d+)?)\s*mm",
    "Thickness (µm)": r"(\d+(?:[.,]\d+)?)\s*(?:µm|um)",
    "Rate (mm/min)": r"(\d+(?:[.,]\d+)?)\s*mm\s*/\s*min",
    "First number": r"(-?\d+(?:[.,]\d+)?)",
}


# --------------------------------------------------------------------------- parameters
def parse_parameter(text: str, regex: str) -> float | None:
    """First number matched by ``regex`` in ``text`` (group 1 if present, decimal comma accepted)."""
    if not text or not regex:
        return None
    try:
        m = re.search(regex, text)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc
    if not m:
        return None
    g = m.group(1) if m.groups() else m.group(0)
    try:
        return float(str(g).replace(",", "."))
    except (TypeError, ValueError):
        num = re.search(r"-?\d+(?:[.,]\d+)?", str(g))
        return float(num.group(0).replace(",", ".")) if num else None


def dataset_parameter(ds: Dataset, regex: str | None = None, meta_key: str | None = None) -> float | None:
    """Parameter of a dataset: ``metadata[meta_key]`` if given (number or text with a number), else the
    regex applied to the dataset name (then to the source file name)."""
    if meta_key:
        v = (ds.metadata or {}).get(meta_key)
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return parse_parameter(str(v), regex or PARAMETER_PRESETS["First number"])
    if regex:
        v = parse_parameter(ds.name, regex)
        if v is None and ds.source_path:
            import os
            v = parse_parameter(os.path.basename(ds.source_path), regex)
        return v
    return None


# --------------------------------------------------------------------------- column helpers
def dsc_columns(ds: Dataset) -> tuple[str, str, str | None]:
    """(temperature, heat flow, time or None) columns of a DSC dataset; ValueError if missing."""
    T = ds.first_of("temperature")
    hf = ds.first_of("specific_heat_flow", "heat_flow")
    if not T or not hf:
        raise ValueError("needs temperature and heat-flow columns")
    return T, hf, ds.first_of("time")


def geometry_from_metadata(md: dict) -> tuple[float | None, float | None]:
    """(area mm², gauge length mm) from metadata (area, diameter or width × thickness)."""
    md = md or {}
    area = md.get("area_mm2")
    if not area and md.get("diameter_mm"):
        area = math.pi * float(md["diameter_mm"]) ** 2 / 4
    if not area and md.get("width_mm") and md.get("thickness_mm"):
        area = float(md["width_mm"]) * float(md["thickness_mm"])
    g = md.get("gauge_length_mm")
    return (float(area) if area else None), (float(g) if g else None)


def mech_arrays(ds: Dataset, *, compression: bool = False, compliance_mm_per_N: float = 0.0,
                area_mm2: float | None = None, gauge_mm: float | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """(strain %, stress MPa, source) of a mechanical dataset.

    Uses existing strain/stress columns; otherwise computes them from force/displacement and the
    geometry (arguments or metadata). Raises ``ValueError`` naming what is missing.
    """
    ec, sc = ds.first_of("strain"), ds.first_of("stress")
    if ec and sc and not compliance_mm_per_N:
        return ds.values(ec, "%"), ds.values(sc, "MPa"), "strain/stress columns"
    Fc, dc = ds.first_of("force"), ds.first_of("displacement", "length")
    if not Fc or not dc:
        if ec and sc:
            return ds.values(ec, "%"), ds.values(sc, "MPa"), "strain/stress columns"
        raise ValueError("needs strain/stress columns or force/displacement columns")
    a, g = geometry_from_metadata(ds.metadata)
    a = area_mm2 or a
    g = gauge_mm or g
    if not a or not g:
        raise ValueError("force/displacement data needs the cross-section area and gauge length "
                         "(metadata area_mm2/diameter_mm and gauge_length_mm)")
    e, s = A.stress_strain(ds.values(Fc, "N"), ds.values(dc, "mm"), a, g, compression=compression,
                           compliance_mm_per_N=compliance_mm_per_N)
    return e, s, f"computed from force/displacement (A0 = {a:.4g} mm², L0 = {g:.4g} mm)"


# --------------------------------------------------------------------------- result
@dataclass
class BatchRow:
    name: str
    ds_id: int
    values: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    parameter: float | None = None
    result: Any = field(default=None, repr=False)


@dataclass
class BatchResult:
    kind: str
    rows: list[BatchRow]
    columns: list[str]
    params: dict
    parameter_label: str = "Parameter"
    provenance: dict = field(default_factory=dict)

    def table(self) -> tuple[list[str], list[list]]:
        """(headers, rows) for display / CSV: Sample, parameter, metrics…, Warnings."""
        has_p = any(r.parameter is not None for r in self.rows)
        headers = ["Sample"] + ([self.parameter_label] if has_p else []) + self.columns + ["Warnings"]
        out = []
        for r in self.rows:
            row: list = [r.name] + ([r.parameter if r.parameter is not None else float("nan")] if has_p else [])
            row += [r.values.get(c, float("nan")) for c in self.columns]
            msgs = ([f"ERROR: {r.error}"] if r.error else []) + list(r.warnings)
            row.append(" | ".join(msgs))
            out.append(row)
        return headers, out

    def metric(self, column: str) -> np.ndarray:
        return np.array([r.values.get(column, float("nan")) for r in self.rows], float)


# --------------------------------------------------------------------------- per kind
_DSC_TEMPS = ["As", "Ap", "Af", "R's", "R'p", "R'f", "Rs", "Rp", "Rf", "Ms", "Mp", "Mf"]


def _dsc_row(ds: Dataset, p: dict) -> tuple[dict, list[str], Any]:
    Tc, hc, tc = dsc_columns(ds)
    md = ds.metadata or {}
    rate = p.get("heating_rate_K_per_min") or md.get("heating_rate_K_per_min")
    src = "user (batch)" if p.get("heating_rate_K_per_min") else md.get("heating_rate_source", "metadata")
    if not rate and not tc:
        src = "default"
    mass = p.get("mass_mg") or md.get("mass_mg")
    kw = {k: v for k, v in p.items() if k not in ("heating_rate_K_per_min", "mass_mg")}
    res = A.analyse_dsc(ds.values(Tc), ds.values(hc), rate or None, mass or None,
                        time=ds.values(tc) if tc else None, hf_unit=ds.channel(hc).unit or "W/g",
                        rate_source=src, **kw)
    vals = {}
    for k in _DSC_TEMPS:
        if k in res.temperatures:
            vals[f"{k} (°C)"] = res.temperatures[k]
    tot = {}
    for direction in ("heating", "cooling"):
        segs = [s for s in res.segments if s.direction == direction and s.peaks]
        if segs:
            s = segs[-1] if kw.get("summary_segment", "last") == "last" else segs[0]
            tot[direction] = float(np.nansum([pk.enthalpy_J_per_g for pk in s.peaks]))
    if "heating" in tot:
        vals["ΔH heating (J/g)"] = tot["heating"]
    if "cooling" in tot:
        vals["ΔH cooling (J/g)"] = tot["cooling"]
    for k, v in res.hysteresis.items():
        vals[f"Hysteresis {k} (K)"] = v
    vals["Peaks (#)"] = len(res.peaks)
    return vals, list(res.warnings), res


def _mech_row(ds: Dataset, p: dict, mode: str) -> tuple[dict, list[str], Any]:
    comp = mode == "compression"
    e, s, _src = mech_arrays(ds, compression=comp, compliance_mm_per_N=p.get("compliance_mm_per_N", 0.0))
    kw = {k: v for k, v in p.items() if k != "compliance_mm_per_N"}
    fn = A.compression_properties if comp else A.tensile_properties
    r = fn(e, s, **kw)
    top = "UTS (MPa)" if not comp else "Max stress (MPa)"
    vals = {"E (GPa)": r.E_GPa, f"{r.offset_pct:g} % offset yield (MPa)": r.yield_strength_MPa, top: r.uts_MPa,
            "Strain at max (%)": r.strain_at_uts_pct,
            ("Fracture strain (%)" if r.fracture_detected else "Final strain (%)"): r.fracture_strain_pct,
            "Toughness (MJ/m³)": r.toughness_MJ_m3}
    if r.superelastic:
        vals.update({"σ_SIM (MPa)": r.sigma_SIM_MPa, "Plateau stress (MPa)": r.plateau_stress_MPa,
                     "Transformation strain (%)": r.transformation_strain_pct})
    return vals, list(r.warnings), r


def _cyclic_row(ds: Dataset, p: dict) -> tuple[dict, list[str], Any]:
    e, s, _src = mech_arrays(ds)
    nc = ds.first_of("count")
    tc = ds.first_of("time")
    res = A.analyse_cyclic(e, s, cycle=ds.values(nc) if nc else None, time=ds.values(tc) if tc else None, **p)
    m = res.complete_metrics or res.metrics
    if not m:
        raise ValueError("no cycles found")
    a, b = m[0], m[-1]
    vals = {"Cycles (#)": len(res.metrics), "σ_ms first (MPa)": a.sigma_ms_MPa, "σ_ms last (MPa)": b.sigma_ms_MPa,
            "Δσ first (MPa)": a.delta_sigma_MPa, "Δσ last (MPa)": b.delta_sigma_MPa,
            "Dissipated first (MJ/m³)": a.dissipated_energy_MJ_m3,
            "Dissipated last (MJ/m³)": b.dissipated_energy_MJ_m3,
            "Accumulated residual strain (%)": b.residual_strain_pct - res.metrics[0].start_strain_pct,
            "Recoverable strain last (%)": b.recoverable_strain_pct}
    ov = dict((q, v) for q, v, _u in A.fatigue_overview(res.metrics))
    vals["Stabilisation cycle (#)"] = ov.get("Stabilisation cycle (Δε_res < 0.01 %/cycle)", float("nan"))
    return vals, list(res.warnings), res


def batch_analyse(datasets: Iterable[Dataset], kind: str, params: dict | None = None, *,
                  parameter_regex: str | None = None, parameter_key: str | None = None,
                  parameter_label: str = "Parameter",
                  progress: Callable[[int, int], None] | None = None) -> BatchResult:
    """Run one analysis with the same parameters on every dataset (errors are caught per row).

    kind: "dsc" (``params`` → :func:`analyse_dsc` keywords; mass and rate from each dataset's
    metadata unless given), "tensile" / "compression" (→ :func:`tensile_properties` /
    :func:`compression_properties` keywords, plus ``compliance_mm_per_N``), "cyclic"
    (→ :func:`analyse_cyclic` keywords).
    """
    if kind not in BATCH_KINDS:
        raise ValueError(f"kind must be one of {BATCH_KINDS}")
    p = dict(params or {})
    dss = list(datasets)
    rows: list[BatchRow] = []
    cols: list[str] = []
    for i, ds in enumerate(dss):
        row = BatchRow(ds.name, ds.id)
        try:
            row.parameter = dataset_parameter(ds, parameter_regex, parameter_key)
        except ValueError as exc:
            row.warnings.append(str(exc))
        try:
            if kind == "dsc":
                vals, warns, res = _dsc_row(ds, p)
            elif kind == "cyclic":
                vals, warns, res = _cyclic_row(ds, p)
            else:
                vals, warns, res = _mech_row(ds, p, kind)
            row.values, row.result = vals, res
            row.warnings += warns
            for c in vals:
                if c not in cols:
                    cols.append(c)
        except Exception as exc:  # noqa: BLE001 - reported per row
            row.error = str(exc) or exc.__class__.__name__
        if (parameter_regex or parameter_key) and row.parameter is None and row.error is None:
            row.warnings.append("parameter not found in the name/metadata")
        rows.append(row)
        if progress:
            progress(i + 1, len(dss))
    all_w = [f"{r.name}: {w}" for r in rows for w in r.warnings] + [f"{r.name}: ERROR {r.error}" for r in rows
                                                                   if r.error]
    prov = provenance(f"Batch {kind}", {"kind": kind, **p, "parameter_regex": parameter_regex,
                                        "parameter_key": parameter_key,
                                        "datasets": [{"name": d.name, "source": d.source_path} for d in dss]},
                      extra={"warnings": all_w})
    return BatchResult(kind, rows, cols, p, parameter_label, prov)


# --------------------------------------------------------------------------- statistics
def group_stats(result: BatchResult, metrics: list[str] | None = None) -> tuple[list[str], list[list]]:
    """Mean ± SD (sample SD, ddof = 1) of each metric across repeats with the same parameter.

    Rows without a parameter form one group labelled 'all' when no row has a parameter, else they are
    skipped. Returns (headers, rows): parameter, n, then '<metric> mean' / '<metric> SD' pairs.
    """
    metrics = metrics or list(result.columns)
    groups: dict[Any, list[BatchRow]] = {}
    any_p = any(r.parameter is not None for r in result.rows)
    for r in result.rows:
        if r.error:
            continue
        key = r.parameter if any_p else "all"
        if key is None:
            continue
        groups.setdefault(key, []).append(r)
    headers = [result.parameter_label, "n"]
    for m in metrics:
        headers += [f"{m} mean", f"{m} SD"]
    out = []
    for key in sorted(groups, key=lambda k: (isinstance(k, str), k)):
        rs = groups[key]
        row: list = [key, len(rs)]
        for m in metrics:
            v = np.array([r.values.get(m, np.nan) for r in rs], float)
            v = v[np.isfinite(v)]
            row += [float(v.mean()) if v.size else float("nan"),
                    float(v.std(ddof=1)) if v.size > 1 else float("nan")]
        out.append(row)
    return headers, out


_UNIT_DIM = {"°C": "temperature", "K": "temperature_difference", "J/g": "specific_energy", "GPa": "stress",
             "MPa": "stress", "%": "strain", "MJ/m³": "energy_density", "#": "count", "mm": "length",
             "µm": "length", "mm/min": "velocity", "s": "time"}


def _split_unit(col: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", col)
    return (m.group(1), m.group(2)) if m else (col, "")


def metric_vs_parameter_dataset(result: BatchResult, metric: str, name: str | None = None, *,
                                grouped: bool = False) -> Dataset:
    """Derived Dataset: parameter (x) vs metric (y); ``grouped`` = mean with an SD column."""
    pname, punit = _split_unit(result.parameter_label)
    mname, munit = _split_unit(metric)
    pdim = _UNIT_DIM.get(punit, "dimensionless")
    mdim = _UNIT_DIM.get(munit, "dimensionless")
    if munit == "GPa":                     # keep moduli readable (canonical stress would be MPa)
        mname, mdim = f"{mname} (GPa)", "dimensionless"
    if punit == "GPa":
        pname, pdim = f"{pname} (GPa)", "dimensionless"
    if pdim == "dimensionless" and punit and punit != "GPa":
        pname = f"{pname} ({punit})"
    if mdim == "dimensionless" and munit and munit != "GPa":
        mname = f"{mname} ({munit})"
    pu = punit if pdim != "dimensionless" else ""
    mu = munit if mdim != "dimensionless" else ""
    if grouped:
        hdr, rows = group_stats(result, [metric])
        rows = [r for r in rows if not isinstance(r[0], str)]
        x = np.array([r[0] for r in rows], float)
        y = np.array([r[2] for r in rows], float)
        sd = np.array([r[3] for r in rows], float)
        n = np.array([r[1] for r in rows], float)
        cols = {pname: (x, pdim, pu), f"{mname} mean": (y, mdim, mu), f"{mname} SD": (sd, mdim, mu),
                "n": (n, "count", "#")}
    else:
        pairs = [(r.parameter, r.values.get(metric, np.nan)) for r in result.rows
                 if r.parameter is not None and not r.error]
        pairs.sort(key=lambda t: t[0])
        x = np.array([p for p, _ in pairs], float)
        y = np.array([v for _, v in pairs], float)
        cols = {pname: (x, pdim, pu), mname: (y, mdim, mu)}
    ds = Dataset.from_arrays(name or f"{mname} vs {pname}", cols, kind="generic")
    ds.metadata["derived_from"] = "batch " + result.kind
    ds.metadata["provenance"] = result.provenance
    return ds
