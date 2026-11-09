"""Analysis dock: DSC | Mechanical | Cyclic | Batch | Elastocaloric.

``AnalysisPanel(plot_panel, store)`` drives the pure functions in :mod:`labstudio.core.analysis`
(+ :mod:`~labstudio.core.batch`, :mod:`~labstudio.core.elastocaloric`) and draws overlays through the
public PlotPanel API (add_trace, add_marker, add_line, fill_between, clear_overlays, add_results,
redraw, current_span, clear, traces; optional add_label / add_colorbar when the plot offers them).

v1.1: every result carries a provenance record (CSV exports start with it) and a warnings list
(amber status + amber table rows); numbers use ``format_value``; long analyses run through
``gui.worker.run_task``; the dock fits ≤ 420 px (scrollable tabs, wrapping/short buttons);
``get_state``/``set_state`` for sessions; ``current_defaults``/``apply_defaults`` for presets.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHeaderView,
                               QLabel, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from ..core import analysis as A
from ..core.batch import geometry_from_metadata
from ..core.dataset import Dataset, DatasetStore
from .analysis_widgets import (AMBER, NONE, ColumnCombo, DatasetCombo, SummaryTable, TabBase, button, combo,
                               dspin, fmt, hrow, log, scroll)
from .worker import run_task

__all__ = ["AnalysisPanel", "DSCTab", "MechanicalTab", "CyclicTab", "SummaryTable", "DatasetCombo",
           "ColumnCombo", "HEAT_COLOR", "COOL_COLOR", "NONE"]

HEAT_COLOR = "#d62728"
COOL_COLOR = "#1f77b4"
BIG_CYCLIC_POINTS = 200_000
_fmt = fmt          # backwards compatibility

DTAD_TIP = ("Theoretical UPPER BOUND ΔT_ad ≈ L/c_p for a complete, perfectly adiabatic transformation. "
            "Measured elastocaloric ΔT of NiTi is much smaller (≈ 15–25 K): incomplete transformation, "
            "heat losses, hysteresis. Do not quote it as the adiabatic temperature change.")


def _form(gb: QGroupBox) -> QFormLayout:
    f = QFormLayout(gb)
    f.setRowWrapPolicy(QFormLayout.WrapLongRows)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    return f


# =========================================================================== DSC
class DSCTab(TabBase):
    KINDS = ("dsc",)
    TOOL = "DSC"
    KEY = "dsc"
    STATE_WIDGETS = ("mass", "rate", "mode", "sign", "baseline", "prom", "trim", "smoothK", "summary_seg", "cp",
                     "clear_cb", "overlap", "min_area", "min_width", "segment")
    DEFAULT_WIDGETS = ("mode", "sign", "baseline", "prom", "trim", "smoothK", "summary_seg", "cp", "overlap",
                       "min_area", "min_width")

    def __init__(self, panel):
        super().__init__(panel)
        self.manual_limits: dict[str, list[tuple[float, float]]] = {}
        self._rate_src: str | None = None
        self._rate_prog = False
        inner = QWidget()
        lay = QVBoxLayout(inner)
        gb = QGroupBox("Data")
        f = _form(gb)
        self.ds_combo = DatasetCombo(self.store, self.KINDS)
        self.c_T = ColumnCombo(("temperature",), tip="Sample temperature column")
        self.c_hf = ColumnCombo(("specific_heat_flow", "heat_flow"),
                                tip="Heat flow: W/g (mW/mg) or absolute mW (then the mass is needed)")
        self.c_t = ColumnCombo(("time",), optional=True,
                               tip="Time column (optional). If given, the actual scan rate is measured and "
                                   "heating/cooling segments are detected from dT/dt.")
        f.addRow("Dataset", self.ds_combo)
        f.addRow("Temperature", self.c_T)
        f.addRow("Heat flow", self.c_hf)
        f.addRow("Time", self.c_t)
        self.mass = dspin(0, 0, 10000, 3, 0.1, "mg", "Sample mass. Needed to convert an absolute heat flow (mW) "
                                                      "into J/g. 0 = not given.", special="not given")
        self.rate = dspin(10, 0, 1000, 2, 1, "K/min", "Scan rate β used to convert the temperature axis to "
                                                       "time (dt = dT/β) when no time column is given; with a "
                                                       "time column it is only cross-checked (warning if "
                                                       "> 15 % different from the measured rate).")
        self.rate_src = QLabel("")
        self.rate_src.setWordWrap(True)
        self.rate_src.setStyleSheet("color: #666; font-size: 9pt;")
        f.addRow("Mass", self.mass)
        f.addRow("Heating rate", self.rate)
        f.addRow("", self.rate_src)
        self.segment = combo(["all"], "Segment to analyse and plot: all, one heating/cooling scan, or the last "
                                      "heating / cooling (2nd cycle removes thermal history)")
        f.addRow("Segment", self.segment)
        self.btn_split = button("Split segments → datasets",
                                "Create one dataset per heating/cooling scan in the store, e.g. "
                                "'Sample (heating 1)', 'Sample (cooling 1)', 'Sample (heating 2)'",
                                self.split_segments)
        f.addRow(self.btn_split)
        lay.addWidget(gb)

        gp = QGroupBox("Parameters")
        g = _form(gp)
        self.mode = combo(["Auto (split heating/cooling)", "Heating only", "Cooling only", "Single segment"],
                          "How to treat the data: auto-split by the sign of dT/dt, or force one segment.")
        self.sign = combo(["Auto-detect", "Endo up", "Exo up"],
                          "Sign convention of the heat-flow signal. Netzsch default: exo up; TA: endo down.")
        self.baseline = combo(["Linear (between limits)", "Linear fit (outside peaks)", "None"],
                              "Linear: straight line between the curve at the start and end of each "
                              "transformation (overlapping peaks share one line). Linear fit: one least-squares "
                              "line through all points outside the peaks. None: zero baseline.")
        self.overlap = combo(["Split at minimum", "Combined R+M (+ parts)"],
                             "Overlapping peaks (e.g. two-step B2→R→B19' on cooling) always share one baseline "
                             "and non-overlapping windows split at the minimum between them. 'Combined' also "
                             "reports the group as one transformation (ΔH of R+M).")
        self.prom = dspin(0, 0, 1e6, 4, 0.01, "", "Minimum peak prominence in heat-flow units "
                                                    "(0 = automatic: 5 % of the largest peak or 5× noise)",
                          special="auto")
        self.min_area = dspin(5, 0, 100, 1, 1, "% of largest", "Peaks with a smaller area than this fraction of "
                                                               "the largest peak are rejected as noise (listed in "
                                                               "the notes). SMA names go to the dominant peaks.")
        self.min_width = dspin(1.0, 0, 50, 2, 0.25, "K", "Peaks narrower (FWHM) than this — or than half the "
                                                         "smoothing window — are rejected as noise")
        self.trim = dspin(3, 0, 50, 1, 0.5, "K", "Ignore the first/last kelvins of every scan segment "
                                                 "(start-up hooks after isotherms)")
        self.smoothK = dspin(0, 0, 20, 2, 0.25, "K", "Savitzky–Golay smoothing window for peak detection and "
                                                     "tangent slopes (auto: adapted to the noise level, ≥ 1 K)",
                             special="auto")
        self.summary_seg = combo(["Last cycle", "First cycle"],
                                 "Which heating/cooling segment feeds the summary when there are several "
                                 "(the 2nd cycle is usual, to erase thermal history)")
        self.cp = dspin(0.45, 0.01, 10, 3, 0.01, "J/(g·K)", "Specific heat capacity for the theoretical upper "
                                                            "bound ΔT_ad ≤ L/c_p (NiTi ≈ 0.45 J/g·K)")
        self.clear_cb = QCheckBox("Clear plot before drawing")
        self.clear_cb.setChecked(True)
        self.clear_cb.setToolTip("Otherwise only previous overlays are removed and the trace is reused")
        g.addRow("Segments", self.mode)
        g.addRow("Sign", self.sign)
        g.addRow("Baseline", self.baseline)
        g.addRow("Overlap", self.overlap)
        g.addRow("Prominence", self.prom)
        g.addRow("Min. peak area", self.min_area)
        g.addRow("Min. width", self.min_width)
        g.addRow("Edge trim", self.trim)
        g.addRow("Smoothing", self.smoothK)
        g.addRow("Summary", self.summary_seg)
        g.addRow("c_p", self.cp)
        g.addRow(self.clear_cb)
        lay.addWidget(gp)

        self.btn_run = button("Run DSC analysis", "Onset/endset by tangent construction, peak temperatures, ΔH, "
                                                  "hysteresis (automatic integration limits)", self.run)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.status)
        self.table = SummaryTable()
        lay.addWidget(self.table)

        gl = QGroupBox("Integration limits (per peak)")
        gll = QVBoxLayout(gl)
        self.peak_table = QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(["Segment", "Peak", "Start (°C)", "End (°C)"])
        self.peak_table.verticalHeader().setVisible(False)
        self.peak_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.peak_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.peak_table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.peak_table.setMinimumHeight(110)
        self.peak_table.setToolTip("Edit the start/end temperature of any peak, then 'Re-run (manual)'.")
        gll.addWidget(self.peak_table)
        self.btn_span_peak = button("Span → selected", "Set the selected peak's limits from the plot span",
                                    self.span_to_selected)
        self.btn_add_peak = button("Add from span", "Add an integration window (selected row's segment) from the "
                                                    "plot span", self.add_from_span)
        self.btn_manual = button("Re-run (manual)", "Re-run the analysis with the limits in this table",
                                 self.run_manual)
        self.btn_auto = button("Auto limits", "Discard manual limits and re-run with automatic detection",
                               self.run_auto)
        gll.addWidget(hrow(self.btn_span_peak, self.btn_add_peak))
        gll.addWidget(hrow(self.btn_manual, self.btn_auto))
        lay.addWidget(gl)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(inner))
        self.ds_combo.currentIndexChanged.connect(self._on_ds)
        self.c_t.currentIndexChanged.connect(lambda *_: (self._update_rate_state(), self._refresh_segments()))
        self.rate.valueChanged.connect(self._on_rate_edited)
        self.rate.editingFinished.connect(self._on_rate_edited)     # Enter on the shown value = confirm

    # ------------------------------------------------------------------ data selection
    def refresh(self):
        self.ds_combo.refresh()
        self._on_ds()

    def _set_rate(self, v: float) -> None:
        self._rate_prog = True
        self.rate.setValue(float(v))
        self._rate_prog = False

    def _on_rate_edited(self, *_):
        if not self._rate_prog:
            self._rate_src = "user"
            self._update_rate_state()

    def _update_rate_state(self) -> None:
        has_time = self.c_t.column() is not None
        src = self._rate_src
        amber = False
        if self.ds_combo.dataset() is None:
            txt = "Select a DSC dataset."
        elif src is None and not has_time:
            txt = (f"⚠ No time column and no rate in the file: defaulted to {self.rate.value():g} K/min — "
                   "please confirm (ΔH ∝ 1/β).")
            amber = True
        elif src is None:
            txt = "Measured from the time column."
        elif src == "user":
            txt = "Entered by user" + (" (cross-checked with the time column)." if has_time else ".")
        else:
            txt = f"From {src}" + (" (cross-checked with the measured rate)." if has_time else ".")
        self.rate_src.setText(txt)
        self.rate.setStyleSheet(f"background: {AMBER};" if amber else "")
        self.rate_src.setStyleSheet("color: #8a5a00; font-size: 9pt;" if amber else "color: #666; font-size: 9pt;")

    def rate_is_defaulted(self) -> bool:
        return self._rate_src is None and self.c_t.column() is None

    def rate_source(self) -> str:
        if self.c_t.column() is not None:
            return "time column (measured)" if self._rate_src is None else f"time column (measured); {self._rate_src}"
        return "default" if self._rate_src is None else self._rate_src

    def _on_ds(self, *_):
        ds = self.ds_combo.dataset()
        self.c_T.populate(ds)
        self.c_hf.populate(ds)
        self.c_t.populate(ds)
        if ds is not None and ds.id != getattr(self, "_meta_ds", None):
            self._meta_ds = ds.id
            self.manual_limits = {}
            m = ds.metadata
            if m.get("mass_mg"):
                self.mass.setValue(float(m["mass_mg"]))
            if m.get("heating_rate_K_per_min"):
                self._set_rate(float(m["heating_rate_K_per_min"]))
                self._rate_src = str(m.get("heating_rate_source") or "file header")
            else:
                self._set_rate(A.DEFAULT_RATE_K_PER_MIN)
                self._rate_src = None
            self.panel._apply_ds_defaults(self, ds)
        self._update_rate_state()
        self._refresh_segments()

    def _segments_info(self, ds=None):
        ds = ds or self.ds_combo.dataset()
        Tc, tc = self.c_T.column(), self.c_t.column()
        if ds is None or not Tc:
            return []
        try:
            segs = A.split_dsc_segments(ds.values(Tc), ds.values(tc) if tc else None).segments
        except Exception:  # noqa: BLE001
            return []
        return A.segment_labels(segs)

    def _refresh_segments(self) -> None:
        cur = self.segment.currentText()
        labels = self._segments_info()
        items = ["all"] + labels
        if any(lb.startswith("heating") for lb in labels):
            items.append("last heating")
        if any(lb.startswith("cooling") for lb in labels):
            items.append("last cooling")
        self.segment.blockSignals(True)
        self.segment.clear()
        self.segment.addItems(items)
        i = self.segment.findText(cur)
        self.segment.setCurrentIndex(i if i >= 0 else 0)
        self.segment.blockSignals(False)

    # ------------------------------------------------------------------ N1 segments → datasets
    def split_segments(self) -> list[Dataset]:
        ds = self.ds_combo.dataset()
        Tc, tc = self.c_T.column(), self.c_t.column()
        if ds is None or not Tc:
            self._msg("Select a DSC dataset with a temperature column.", True)
            return []
        subs = self.safe("Split segments", A.dsc_segment_datasets, ds, Tc, tc) or []
        if not subs:
            if not self.status.text():
                self._msg("No heating/cooling segments detected.", True)
            return []
        existing = {d.name for d in self.store}
        added = []
        for sub in subs:
            if sub.name in existing:
                continue
            added.append(sub)
        for sub in added:
            self.store.add(sub)
        self.ds_combo.select(ds)
        self._msg(f"{len(added)} segment dataset(s) added: " + ", ".join(s.name for s in added)
                  if added else "Segment datasets already exist in the store.")
        return added

    def _segment_dataset(self, ds: Dataset, label: str) -> Dataset | None:
        for d in self.store:
            if d.metadata.get("derived_from") == ds.name and d.metadata.get("segment") == label:
                return d
        Tc, tc = self.c_T.column(), self.c_t.column()
        for sub in A.dsc_segment_datasets(ds, Tc, tc):
            if sub.metadata.get("segment") == label:
                self.store.add(sub)
                self.ds_combo.select(ds)
                return sub
        return None

    # ------------------------------------------------------------------ parameters
    def params(self) -> dict:
        """analyse_dsc keyword parameters from the widgets (without data, mass and rate)."""
        return dict(endo_up=[None, True, False][self.sign.currentIndex()],
                    baseline=["linear", "fit", "none"][self.baseline.currentIndex()],
                    prominence=self.prom.value() or None, trim_K=self.trim.value(),
                    smooth_K=self.smoothK.value() or None,
                    mode=["auto", "heating", "cooling", "single"][self.mode.currentIndex()],
                    summary_segment="last" if self.summary_seg.currentIndex() == 0 else "first",
                    cp_J_per_gK=self.cp.value(), min_area_frac=self.min_area.value() / 100.0,
                    min_width_K=self.min_width.value(),
                    overlap="split" if self.overlap.currentIndex() == 0 else "combined")

    # ------------------------------------------------------------------ run
    def run(self) -> A.DSCAnalysis | None:
        return self.safe("DSC analysis", self._run)

    def run_manual(self) -> A.DSCAnalysis | None:
        self.manual_limits = self._limits_from_table()
        if not self.manual_limits:
            self._msg("The limits table is empty: run the analysis first.", True)
            return None
        return self.run()

    def run_auto(self) -> A.DSCAnalysis | None:
        self.manual_limits = {}
        return self.run()

    def _run(self) -> A.DSCAnalysis | None:
        ds = self.ds_combo.dataset()
        Tc, hc, tc = self.c_T.column(), self.c_hf.column(), self.c_t.column()
        if ds is None or not Tc or not hc:
            self._msg("Select a dataset with temperature and heat-flow columns.", True)
            return None
        unit = ds.channel(hc).unit or "W/g"
        mass = self.mass.value() or None
        seg = self.segment.currentText() or "all"
        p = self.params()
        if p["mode"] != "auto":
            seg = "all"
        rate_src = self.rate_source()
        res = A.analyse_dsc(ds.values(Tc), ds.values(hc), self.rate.value() or None, mass,
                            time=ds.values(tc) if tc else None, hf_unit=unit, segment=seg,
                            limits=dict(self.manual_limits) or None,
                            rate_source="default" if self.rate_is_defaulted() else rate_src, **p)
        self.last_result = res
        self._ran = True
        warnings = list(res.warnings)
        params = {**res.params, "columns": {"temperature": Tc, "heat_flow": hc, "time": tc},
                  "heating_rate_field_K_per_min": self.rate.value(), "heating_rate_source": rate_src}
        rows = self._rows(res)
        rows.insert(0, ("Heating rate source", rate_src, ""))
        trows, warn_idx = self.table_rows(rows, warnings)
        tips = {i: DTAD_TIP for i, r in enumerate(trows) if str(r[0]).startswith("ΔT_ad")}
        self.table.set_rows(trows, ["Quantity", "Value", "Unit"], warn_rows=warn_idx, tooltips=tips)
        self._record("DSC", params, ds, warnings, [self.table])
        self._fill_peak_table(res)
        self._draw(ds, Tc, hc, res, seg)
        self._results(ds, [r for r in rows if not isinstance(r[1], str)], warnings)
        n = len(res.peaks)
        note = " ".join(n_ for n_ in res.notes if not n_.startswith("Sign convention"))
        if n and not np.isfinite(res.peaks[0].enthalpy_J_per_g):
            note += " Heat flow is absolute (mW) and no mass was given: ΔH reported in J."
        manual = " (manual limits)" if self.manual_limits else ""
        self._msg(f"{n} transformation peak(s) found{manual}. {note}".strip(), n == 0 and not warnings, warnings)
        return res

    def _rows(self, res: A.DSCAnalysis) -> list[tuple[str, float, str]]:
        rows = list(res.rows())
        multi = len(res.segments) > 2
        for s in res.segments:
            tagb = f" [{s.label}]" if multi or s.label.endswith(("2", "3", "4")) else f" [{s.direction}]"
            for p in s.peaks:
                tag = tagb
                rows += [(f"{p.label_start} onset{tag}", p.onset_T, "°C"),
                         (f"{p.label_peak} peak{tag}", p.peak_T, "°C"),
                         (f"{p.label_end} endset{tag}", p.endset_T, "°C"),
                         (f"{p.phase} start (integration){tag}", p.start_T, "°C"),
                         (f"{p.phase} end (integration){tag}", p.end_T, "°C")]
                if np.isfinite(p.enthalpy_J_per_g):
                    rows.append((f"ΔH {p.phase}{tag}", p.enthalpy_J_per_g, "J/g"))
                else:
                    rows.append((f"ΔH {p.phase}{tag}", p.enthalpy_J, "J"))
                rows.append((f"FWHM {p.phase}{tag}", p.fwhm_K, "K"))
            for c in s.combined:
                rows.append((f"ΔH {c.phase} combined{tagb}", c.enthalpy_J_per_g, "J/g"))
            rows.append((f"Scan rate [{s.label}]", s.heating_rate_K_per_min, "K/min"))
            if np.isfinite(s.smoothing_K):
                rows.append((f"Smoothing window [{s.label}]", s.smoothing_K, "K"))
        return rows

    # ------------------------------------------------------------------ limits table
    def _fill_peak_table(self, res: A.DSCAnalysis) -> None:
        self.peak_table.setRowCount(0)
        for s in res.segments:
            for p in s.peaks:
                self._add_limit_row(s.label, p.phase, p.start_T, p.end_T)

    def _add_limit_row(self, seg: str, phase: str, a: float, b: float) -> None:
        i = self.peak_table.rowCount()
        self.peak_table.insertRow(i)
        self.peak_table.setItem(i, 0, QTableWidgetItem(seg))
        self.peak_table.setItem(i, 1, QTableWidgetItem(phase))
        for j, v in ((2, a), (3, b)):
            sp = QDoubleSpinBox()
            sp.setRange(-300, 2000)
            sp.setDecimals(2)
            sp.setSingleStep(0.5)
            sp.setValue(float(v))
            sp.setKeyboardTracking(False)
            self.peak_table.setCellWidget(i, j, sp)
        self.peak_table.resizeColumnsToContents()

    def _limits_from_table(self) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for i in range(self.peak_table.rowCount()):
            seg = self.peak_table.item(i, 0).text()
            a = self.peak_table.cellWidget(i, 2).value()
            b = self.peak_table.cellWidget(i, 3).value()
            out.setdefault(seg, []).append((a, b))
        return out

    def set_peak_limits(self, row: int, start: float, end: float) -> None:
        self.peak_table.cellWidget(row, 2).setValue(start)
        self.peak_table.cellWidget(row, 3).setValue(end)

    def span_to_selected(self) -> None:
        span = self.plot.current_span()
        rows = sorted({ix.row() for ix in self.peak_table.selectedIndexes()})
        if not span:
            self._msg("No span selected on the plot (use the Span tool).", True)
            return
        if not rows:
            self._msg("Select a peak row in the limits table first.", True)
            return
        r = rows[0]
        a, b = sorted(span)
        cur_a, cur_b = self.peak_table.cellWidget(r, 2).value(), self.peak_table.cellWidget(r, 3).value()
        if cur_a > cur_b:
            a, b = b, a                         # keep process order (cooling: start > end)
        self.set_peak_limits(r, a, b)
        self._msg(f"Limits of row {r + 1} set to {a:.4g} … {b:.4g} °C — press 'Re-run (manual)'.")

    def add_from_span(self) -> None:
        span = self.plot.current_span()
        if not span:
            self._msg("No span selected on the plot (use the Span tool).", True)
            return
        rows = sorted({ix.row() for ix in self.peak_table.selectedIndexes()})
        seg = self.peak_table.item(rows[0], 0).text() if rows else (
            self.segment.currentText() if self.segment.currentText() not in ("all", "") else "heating 1")
        if seg.startswith("last"):
            labels = self._segments_info()
            seg = [lb for lb in labels if lb.startswith(seg.split()[1])][-1]
        self._add_limit_row(seg, "new", *sorted(span))
        self._msg(f"Window {min(span):.4g} … {max(span):.4g} °C added for {seg} — press 'Re-run (manual)'.")

    # ------------------------------------------------------------------ drawing
    def _draw(self, ds, Tc, hc, res, seg: str = "all"):
        plot_ds = ds
        if seg not in ("all", ""):
            labels = [s.label for s in res.segments]
            if len(labels) == 1:
                sub = self._segment_dataset(ds, labels[0])
                if sub is not None:
                    plot_ds = sub
        self._ensure_trace(plot_ds, Tc, hc, self.clear_cb.isChecked(), color="black", linewidth=1.0)
        for s in res.segments:
            combined_parts = {id(p) for c in s.combined for p in (c.parts or [])}
            for c in s.combined:
                col = HEAT_COLOR if c.direction == "heating" else COOL_COLOR
                self.plot.fill_between(c.region_T, c.region_hf, c.region_baseline, color=col, alpha=0.25)
                xm = float(np.mean(c.region_T))
                self.label(xm, float(np.interp(xm, c.region_T, c.region_baseline)),
                           f"ΔH {c.phase} {fmt(c.enthalpy_J_per_g)} J/g", col, None)
            for p in s.peaks:
                col = HEAT_COLOR if p.direction == "heating" else COOL_COLOR
                if p.region_T is not None and p.region_T.size > 1:
                    if id(p) not in combined_parts:
                        self.plot.fill_between(p.region_T, p.region_hf, p.region_baseline, color=col, alpha=0.25)
                    self.plot.add_line(p.region_T, p.region_baseline, None, color="gray", linestyle="--",
                                       linewidth=0.8)
                for tan in (p.tangent_onset, p.tangent_endset):
                    if tan is not None:
                        self.plot.add_line(tan[0], tan[1], None, color=col, linestyle=":", linewidth=1.0)
                self.label(p.onset_T, p.onset_hf, f"{p.label_start} {p.onset_T:.1f} °C", col, "o")
                self.label(p.endset_T, p.endset_hf, f"{p.label_end} {p.endset_T:.1f} °C", col, "s")
                self.label(p.peak_T, p.peak_hf, f"{p.label_peak} {p.peak_T:.1f} °C", col, "v")
        self.plot.redraw()

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        st = super().get_state()
        st["manual_limits"] = {k: [list(map(float, w)) for w in v] for k, v in self.manual_limits.items()}
        st["rate_source"] = self._rate_src
        return st

    def set_state(self, state: dict, id_map: dict) -> None:
        super().set_state(state, id_map)
        self._refresh_segments()
        self.apply_values({"segment": (state.get("params") or {}).get("segment")})
        if "rate_source" in state:
            self._rate_src = state.get("rate_source")
            self._update_rate_state()
        self.manual_limits = {k: [tuple(w) for w in v] for k, v in (state.get("manual_limits") or {}).items()}


# =========================================================================== mechanical
class MechanicalTab(TabBase):
    KINDS = ("tensile", "compression")
    TOOL = "Mechanical"
    KEY = "mechanical"
    STATE_WIDGETS = ("test_mode", "area", "diam", "gauge", "compliance", "E_true", "auto_mod", "mod_lo", "mod_hi",
                     "r2min", "offset", "drop", "drop_win", "se_mode", "shade", "clear_cb")
    DEFAULT_WIDGETS = ("test_mode", "compliance", "auto_mod", "mod_lo", "mod_hi", "r2min", "offset", "drop",
                       "drop_win", "se_mode")

    def __init__(self, panel):
        super().__init__(panel)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        gb = QGroupBox("Data")
        f = _form(gb)
        self.ds_combo = DatasetCombo(self.store, self.KINDS)
        self.test_mode = combo(["Tensile", "Compression"],
                               "Compression: compressive stress/strain are reported as positive values "
                               "(negative machine signs are flipped)")
        f.addRow("Dataset", self.ds_combo)
        f.addRow("Test", self.test_mode)
        lay.addWidget(gb)

        gs = QGroupBox("Stress–strain from force / displacement")
        g = _form(gs)
        self.c_F = ColumnCombo(("force",), tip="Force column (N)", strict=True)
        self.c_d = ColumnCombo(("displacement", "length"), tip="Crosshead / extensometer displacement (mm)",
                               strict=True)
        self.area = dspin(0, 0, 1e6, 6, 0.01, "mm²", "Initial cross-section A0 (σ = F/A0)", special="missing")
        self.diam = dspin(0, 0, 1e4, 3, 0.1, "mm", "Round specimen diameter; sets A0 = πd²/4", special="–")
        self.gauge = dspin(0, 0, 1e5, 3, 1, "mm", "Initial gauge length L0 (ε = ΔL/L0 × 100 %)",
                           special="missing")
        self.compliance = dspin(0, 0, 1.0, 7, 1e-5, "mm/N", "Machine compliance C (frame + grips): the crosshead "
                                                            "displacement is corrected to d − C·F before computing "
                                                            "the strain. 0 = no correction.", special="none")
        self.E_true = dspin(60, 1, 1000, 2, 1, "GPa", "Known (true) Young's modulus of the specimen used to "
                                                      "estimate C (e.g. NiTi austenite ≈ 60–75 GPa from an "
                                                      "extensometer or literature)")
        self.btn_C = button("Estimate C from known E", "Solve C so that the corrected elastic slope equals the "
                                                        "known modulus (uses the auto/selected modulus range)",
                            self.estimate_compliance)
        self.diam.valueChanged.connect(lambda v: v > 0 and self.area.setValue(math.pi * v * v / 4))
        self.btn_ss = button("Compute stress–strain", "Adds 'Stress' (MPa) and 'Strain' (%) columns to the dataset "
                                                      "and plots them", self.compute_stress_strain)
        g.addRow("Force", self.c_F)
        g.addRow("Displacement", self.c_d)
        g.addRow("Diameter", self.diam)
        g.addRow("Area A0", self.area)
        g.addRow("Gauge L0", self.gauge)
        g.addRow("Compliance C", self.compliance)
        g.addRow("Known E", self.E_true)
        g.addRow(self.btn_C)
        g.addRow(self.btn_ss)
        lay.addWidget(gs)

        gp = QGroupBox("Properties")
        h = _form(gp)
        self.c_e = ColumnCombo(("strain",), tip="Engineering strain column (%)", strict=True)
        self.c_s = ColumnCombo(("stress",), tip="Engineering stress column (MPa)", strict=True)
        self.auto_mod = QCheckBox("Auto modulus range")
        self.auto_mod.setChecked(True)
        self.auto_mod.setToolTip("Auto: steepest sliding window with R² ≥ R²min on the loading branch")
        self.mod_lo = dspin(0.1, -100, 1000, 3, 0.05, "%", "Modulus fit start strain")
        self.mod_hi = dspin(0.4, -100, 1000, 3, 0.05, "%", "Modulus fit end strain")
        self.btn_span = button("Use plot span", "Take the modulus strain range from the span selected on the plot",
                               self.use_span)
        rw = hrow(self.mod_lo, self.mod_hi)
        self.r2min = dspin(0.995, 0.5, 1.0, 4, 0.001, "", "Minimum R² of an auto modulus window")
        self.offset = dspin(0.2, 0.0, 10, 3, 0.05, "%", "Offset strain of the proof (yield) strength line")
        self.drop = dspin(40, 1, 100, 1, 5, "% of UTS", "Fracture = first drop of at least this fraction of "
                                                        "the max stress after the maximum …")
        self.drop_win = dspin(0.5, 0.001, 50, 3, 0.1, "%", "… occurring within this strain interval")
        self.se_mode = combo(["Auto", "Yes", "No"], "Detect the superelastic (stress-induced martensite) "
                                                    "plateau and σ_SIM by the two-tangent method")
        self.shade = QCheckBox("Shade toughness area")
        self.shade.setToolTip("Shade ∫σdε up to fracture")
        self.clear_cb = QCheckBox("Clear plot before drawing")
        self.clear_cb.setChecked(True)
        h.addRow("Strain", self.c_e)
        h.addRow("Stress", self.c_s)
        h.addRow(self.auto_mod)
        h.addRow("Modulus range", rw)
        h.addRow("", self.btn_span)
        h.addRow("R² min", self.r2min)
        h.addRow("Yield offset", self.offset)
        h.addRow("Fracture drop", self.drop)
        h.addRow("Drop within", self.drop_win)
        h.addRow("Superelastic", self.se_mode)
        h.addRow(self.shade)
        h.addRow(self.clear_cb)
        lay.addWidget(gp)
        self.btn_run = button("Run mechanical analysis", "E, offset yield, UTS, fracture strain, toughness, σ_SIM",
                              self.run)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.status)
        self.table = SummaryTable()
        lay.addWidget(self.table)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(inner))
        self.auto_mod.toggled.connect(self._toggle_mod)
        self._toggle_mod(True)
        self.ds_combo.currentIndexChanged.connect(self._on_ds)

    def _toggle_mod(self, auto: bool):
        for w in (self.mod_lo, self.mod_hi, self.btn_span):
            w.setEnabled(not auto)
        self.r2min.setEnabled(auto)

    def refresh(self):
        self.ds_combo.refresh()
        self._on_ds()

    def _on_ds(self, *_):
        ds = self.ds_combo.dataset()
        for c in (self.c_F, self.c_d, self.c_e, self.c_s):
            c.populate(ds)
        if ds is None or ds.id == getattr(self, "_meta_ds", None):
            return
        self._meta_ds = ds.id
        self.test_mode.setCurrentIndex(1 if ds.kind == "compression" else 0)
        m = ds.metadata
        self.diam.blockSignals(True)
        self.diam.setValue(float(m.get("diameter_mm") or 0))
        self.diam.blockSignals(False)
        area, gauge = geometry_from_metadata(m)
        self.area.setValue(float(area or 0))
        self.gauge.setValue(float(gauge or 0))
        self.panel._apply_ds_defaults(self, ds)

    def use_span(self):
        span = self.plot.current_span()
        if span:
            self.auto_mod.setChecked(False)
            self.mod_lo.setValue(min(span))
            self.mod_hi.setValue(max(span))
            self._msg(f"Modulus range {min(span):.3g}–{max(span):.3g} % from plot span.")
        else:
            self._msg("No span selected on the plot (use the Span tool).", True)

    def _geometry_ok(self) -> bool:
        if self.area.value() <= 0 or self.gauge.value() <= 0:
            self._msg("Enter the cross-section area (or diameter) and the gauge length first.", True)
            (self.area if self.area.value() <= 0 else self.gauge).setFocus()
            return False
        return True

    def estimate_compliance(self) -> float | None:
        ds = self.ds_combo.dataset()
        Fc, dc = self.c_F.column(), self.c_d.column()
        if ds is None or not Fc or not dc:
            self._msg("Select force and displacement columns.", True)
            return None
        if not self._geometry_ok():
            return None
        mr = None if self.auto_mod.isChecked() else (self.mod_lo.value(), self.mod_hi.value())
        r = self.safe("Compliance estimate", A.estimate_compliance, ds.values(Fc, "N"), ds.values(dc, "mm"),
                      self.area.value(), self.gauge.value(), self.E_true.value(), mr)
        if r is None:
            return None
        C = r["C_mm_per_N"]
        if C < 0:
            self._msg(f"Apparent E = {fmt(r['E_apparent_GPa'])} GPa is already ≥ the known E: no compliance "
                      "correction possible (C would be negative).", False,
                      ["Estimated compliance is negative — check the known E."])
            return C
        self.compliance.setValue(C)
        self._msg(f"Machine compliance C = {fmt(C)} mm/N (apparent E = {fmt(r['E_apparent_GPa'])} GPa → "
                  f"{fmt(self.E_true.value())} GPa). Press 'Compute stress–strain' to apply.")
        return C

    def compute_stress_strain(self) -> bool:
        return bool(self.safe("Stress–strain", self._compute_stress_strain))

    def _compute_stress_strain(self) -> bool:
        ds = self.ds_combo.dataset()
        Fc, dc = self.c_F.column(), self.c_d.column()
        if ds is None or not Fc or not dc:
            self._msg("Select force and displacement columns.", True)
            return False
        if not self._geometry_ok():
            return False
        comp = self.test_mode.currentIndex() == 1
        C = self.compliance.value()
        e, s = A.stress_strain(ds.values(Fc, "N"), ds.values(dc, "mm"), self.area.value(), self.gauge.value(),
                               compression=comp, compliance_mm_per_N=C)
        ds.add_column("Strain", e, "strain", "%")
        ds.add_column("Stress", s, "stress", "MPa")
        ds.metadata["area_mm2"] = self.area.value()
        ds.metadata["gauge_length_mm"] = self.gauge.value()
        if C:
            ds.metadata["machine_compliance_mm_per_N"] = C
        if self.diam.value() > 0:
            ds.metadata["diameter_mm"] = self.diam.value()
        ds.default_x, ds.default_y = "Strain", "Stress"
        self.store.notify_changed()          # may refresh the combos
        self.ds_combo.select(ds)
        self.c_e.populate(ds, "Strain")
        self.c_s.populate(ds, "Stress")
        self.plot.clear()
        self.plot.add_trace(ds, "Strain", "Stress", label=ds.name)
        self.plot.redraw()
        self._msg("Added Stress (MPa) and Strain (%) columns" + (f" (compliance-corrected, C = {fmt(C)} mm/N)."
                                                                 if C else "."))
        return True

    def params(self) -> dict:
        return dict(modulus_range=None if self.auto_mod.isChecked() else (self.mod_lo.value(), self.mod_hi.value()),
                    offset_pct=self.offset.value(), fracture_drop_pct=self.drop.value(),
                    fracture_window_pct=self.drop_win.value(),
                    superelastic=[None, True, False][self.se_mode.currentIndex()], r2_min=self.r2min.value())

    def run(self) -> A.MechanicalResult | None:
        return self.safe("Mechanical analysis", self._run)

    def _run(self) -> A.MechanicalResult | None:
        ds = self.ds_combo.dataset()
        ec, sc = self.c_e.column(), self.c_s.column()
        if ds is None or not ec or not sc:
            self._msg("Select strain and stress columns (or compute them from force/displacement).", True)
            return None
        p = self.params()
        mr = p.pop("modulus_range")
        fn = A.compression_properties if self.test_mode.currentIndex() == 1 else A.tensile_properties
        res = fn(ds.values(ec, "%"), ds.values(sc, "MPa"), mr, **p)
        self.last_result = res
        self._ran = True
        warnings = list(res.warnings)
        params = {**res.params, "columns": {"strain": ec, "stress": sc},
                  "machine_compliance_mm_per_N": ds.metadata.get("machine_compliance_mm_per_N", 0.0),
                  "area_mm2": ds.metadata.get("area_mm2"), "gauge_length_mm": ds.metadata.get("gauge_length_mm")}
        self._draw(ds, ec, sc, res)
        rows = res.rows()
        trows, widx = self.table_rows(rows, warnings)
        self.table.set_rows(trows, ["Quantity", "Value", "Unit"], warn_rows=widx)
        self._record("Tensile" if res.mode == "tensile" else "Compression", params, ds, warnings, [self.table])
        self._results(ds, rows, warnings)
        self._msg(("Done. " + " ".join(res.notes)).strip(), False, warnings)
        return res

    def _draw(self, ds, ec, sc, r: A.MechanicalResult):
        self._ensure_trace(ds, ec, sc, self.clear_cb.isChecked())
        sign = 1.0
        esign = 1.0
        if r.mode == "compression":
            s = ds.values(sc)
            sign = -1.0 if s[np.nanargmax(np.abs(s))] < 0 else 1.0
            e = ds.values(ec)
            esign = -1.0 if np.nanmax(e) < -np.nanmin(e) else 1.0
        if self.shade.isChecked():
            e = ds.values(ec)[: r.i_fracture + 1]
            s = ds.values(sc)[: r.i_fracture + 1]
            self.plot.fill_between(e, s, 0, color="#ff7f0e", alpha=0.15)
        if r.modulus_line is not None:
            self.plot.add_line(esign * r.modulus_line[0], sign * r.modulus_line[1], f"E = {fmt(r.E_GPa)} GPa",
                               color="#2ca02c", linestyle="--", linewidth=1.2)
        if r.offset_line is not None:
            self.plot.add_line(esign * r.offset_line[0], sign * r.offset_line[1], f"{r.offset_pct:g} % offset",
                               color="#9467bd", linestyle=":", linewidth=1.2)
        if r.plateau_line is not None:
            self.plot.add_line(esign * r.plateau_line[0], sign * r.plateau_line[1], "SIM plateau",
                               color="#8c564b", linestyle="-.", linewidth=1.2)
        if np.isfinite(r.yield_strength_MPa):
            self.label(esign * r.yield_strain_pct, sign * r.yield_strength_MPa,
                       f"σ{r.offset_pct:g} = {r.yield_strength_MPa:.0f} MPa", "#9467bd", "o")
        lab = "UTS" if r.mode == "tensile" else "σmax"
        self.label(esign * r.strain_at_uts_pct, sign * r.uts_MPa, f"{lab} = {r.uts_MPa:.0f} MPa", "#d62728", "^")
        if r.fracture_detected:
            self.label(esign * r.fracture_strain_pct, sign * r.fracture_stress_MPa,
                       f"Fracture ε = {r.fracture_strain_pct:.2f} %", "black", "x")
        if r.superelastic:
            self.label(esign * r.strain_SIM_pct, sign * r.sigma_SIM_MPa, f"σ_SIM = {r.sigma_SIM_MPa:.0f} MPa",
                       "#8c564b", "D")
        self.plot.redraw()


# =========================================================================== cyclic
_FIT_METRICS = ("Accumulated residual strain (%)", "Stress hysteresis Δσ (MPa)", "Dissipated energy (MJ/m³)")


class CyclicTab(TabBase):
    KINDS = ("cyclic",)
    TOOL = "Cyclic"
    KEY = "cyclic"
    STATE_WIDGETS = ("compression", "prom", "pfrac", "pwin", "pref", "every", "add_cycles", "metric", "fit_cb",
                     "area", "diam", "gauge", "stab_thr")
    DEFAULT_WIDGETS = ("compression", "prom", "pfrac", "pwin", "pref", "every")

    def __init__(self, panel):
        super().__init__(panel)
        self.metrics: list[A.CycleMetrics] = []
        self.cycles: list[A.Cycle] = []
        self.result: A.CyclicResult | None = None
        self.fits: dict[str, A.SaturationFit] = {}
        self.fatigue_ds: Dataset | None = None
        inner = QWidget()
        lay = QVBoxLayout(inner)
        gb = QGroupBox("Data")
        f = _form(gb)
        self.ds_combo = DatasetCombo(self.store, self.KINDS)
        self.c_e = ColumnCombo(("strain",), tip="Strain column (%)", strict=True)
        self.c_s = ColumnCombo(("stress",), tip="Stress column (MPa)", strict=True)
        self.c_n = ColumnCombo(("count",), optional=True,
                               tip="Cycle-number column (optional). Without it, cycles are split at strain "
                                   "minima between load reversals.")
        self.compression = combo(["Auto", "Yes", "No"],
                                 "Compression (invert signs): Auto flips strain/stress whose largest excursion "
                                 "is negative (compression recorded as negative); Yes inverts both; No keeps "
                                 "the recorded signs. Loops, plateaus and areas use positive magnitudes.")
        f.addRow("Dataset", self.ds_combo)
        f.addRow("Strain", self.c_e)
        f.addRow("Stress", self.c_s)
        f.addRow("Cycle no.", self.c_n)
        f.addRow("Compression", self.compression)
        lay.addWidget(gb)

        self.gss = QGroupBox("No stress/strain? Compute from force / displacement")
        gs = _form(self.gss)
        self.c_F = ColumnCombo(("force",), tip="Force column", strict=True)
        self.c_d = ColumnCombo(("displacement", "length"), tip="Displacement column", strict=True)
        self.diam = dspin(0, 0, 1e4, 3, 0.1, "mm", "Round specimen diameter; sets A0 = πd²/4", special="–")
        self.area = dspin(0, 0, 1e6, 6, 0.01, "mm²", "Cross-section A0", special="missing")
        self.gauge = dspin(0, 0, 1e5, 3, 1, "mm", "Gauge length L0", special="missing")
        self.diam.valueChanged.connect(lambda v: v > 0 and self.area.setValue(math.pi * v * v / 4))
        self.btn_ss = button("Compute stress–strain", "Adds Stress (MPa) / Strain (%) columns from force and "
                                                      "displacement (geometry from the file metadata or these "
                                                      "fields)", self.compute_stress_strain)
        gs.addRow("Force", self.c_F)
        gs.addRow("Displacement", self.c_d)
        gs.addRow("Diameter", self.diam)
        gs.addRow("Area A0", self.area)
        gs.addRow("Gauge L0", self.gauge)
        gs.addRow(self.btn_ss)
        lay.addWidget(self.gss)

        gp = QGroupBox("Parameters")
        g = _form(gp)
        self.prom = dspin(0, 0, 1000, 3, 0.1, "%", "Minimum strain prominence of a load reversal "
                                                    "(0 = 30 % of the strain range)", special="auto")
        self.pfrac = dspin(50, 5, 95, 1, 5, "%", "Plateau stresses are read at this fraction of the cycle's "
                                                 "strain range (mid-plateau)")
        self.pwin = dspin(4, 0.1, 50, 1, 1, "%", "Width of the averaging window around the plateau strain, "
                                                 "as % of the strain range")
        self.pref = combo(["Cycle strain range", "Absolute strain"],
                          "Range: ε_p = ε_lo + f·(ε_max − ε_lo) with ε_lo = max(start, residual) — robust to "
                          "accumulated residual strain. Absolute: ε_p = f·ε_max.")
        self.every = QSpinBox()
        self.every.setRange(1, 1000)
        self.every.setValue(1)
        self.every.setToolTip("Plot every N-th cycle (first and last always plotted)")
        self.add_cycles = QCheckBox("Add per-cycle datasets to the store")
        self.add_cycles.setToolTip("Export each plotted cycle as a separate dataset")
        g.addRow("Reversal prominence", self.prom)
        g.addRow("Plateau at", self.pfrac)
        g.addRow("Plateau window", self.pwin)
        g.addRow("Plateau ref.", self.pref)
        g.addRow("Plot every", self.every)
        g.addRow(self.add_cycles)
        lay.addWidget(gp)
        self.btn_run = button("Run cyclic analysis", "Per-cycle residual strain, plateau stresses, Δσ, dissipated "
                                                     "energy (incomplete cycles flagged and excluded)", self.run)
        lay.addWidget(self.btn_run)

        gf = QGroupBox("Fatigue summary")
        ff = _form(gf)
        self.metric = combo([c for c in A.FATIGUE_COLUMNS if c != "Cycle"], "Metric to plot against cycle number")
        self.metric.setCurrentText("Accumulated residual strain (%)")
        self.fit_cb = QCheckBox("Fit saturating trend a + b(1 − e^(−(N−1)/τ))")
        self.fit_cb.setChecked(True)
        self.fit_cb.setToolTip("Least-squares fit drawn as a line; τ, saturation value and stabilisation cycle "
                               "are reported")
        self.stab_thr = dspin(0, 0, 1e6, 4, 0.01, "/cycle", "Stabilisation = first cycle whose fitted change per "
                                                            "cycle is below this (0 = defaults: 0.01 % strain, "
                                                            "0.5 MPa, 0.05 MJ/m³ per cycle)", special="default")
        self.btn_fat = button("Plot fatigue summary", "Create a derived dataset (cycle vs metrics), add it to the "
                                                      "store and plot the selected metric", self.plot_fatigue)
        ff.addRow("Metric", self.metric)
        ff.addRow(self.fit_cb)
        ff.addRow("Stab. threshold", self.stab_thr)
        ff.addRow(self.btn_fat)
        lay.addWidget(gf)
        lay.addWidget(self.status)
        lay.addWidget(QLabel("Overview"))
        self.table = SummaryTable()
        lay.addWidget(self.table)
        lay.addWidget(QLabel("Trend fits"))
        self.fit_table = SummaryTable(["Metric", "a (N=1)", "Saturation a+b", "τ (cycles)", "R²",
                                       "Stabilisation cycle"], min_height=90)
        lay.addWidget(self.fit_table)
        lay.addWidget(QLabel("Per cycle"))
        self.cycle_table = SummaryTable(list(A.FATIGUE_COLUMNS))
        lay.addWidget(self.cycle_table)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(inner))
        self.ds_combo.currentIndexChanged.connect(self._on_ds)

    def refresh(self):
        self.ds_combo.refresh()
        self._on_ds()

    def _on_ds(self, *_):
        ds = self.ds_combo.dataset()
        for c in (self.c_e, self.c_s, self.c_n, self.c_F, self.c_d):
            c.populate(ds)
        has_ss = self.c_e.column() is not None and self.c_s.column() is not None
        self.gss.setVisible(not has_ss)
        if ds is not None and ds.id != getattr(self, "_meta_ds", None):
            self._meta_ds = ds.id
            m = ds.metadata
            self.diam.blockSignals(True)
            self.diam.setValue(float(m.get("diameter_mm") or 0))
            self.diam.blockSignals(False)
            area, gauge = geometry_from_metadata(m)
            self.area.setValue(float(area or 0))
            self.gauge.setValue(float(gauge or 0))
            self.panel._apply_ds_defaults(self, ds)

    def compute_stress_strain(self) -> bool:
        return bool(self.safe("Stress–strain", self._compute_ss))

    def _compute_ss(self) -> bool:
        ds = self.ds_combo.dataset()
        Fc, dc = self.c_F.column(), self.c_d.column()
        if ds is None or not Fc or not dc:
            self._msg("This dataset has no force/displacement columns either.", True)
            return False
        if self.area.value() <= 0 or self.gauge.value() <= 0:
            self._msg("Enter the cross-section area (or diameter) and the gauge length.", True)
            (self.area if self.area.value() <= 0 else self.gauge).setFocus()
            return False
        e, s = A.stress_strain(ds.values(Fc, "N"), ds.values(dc, "mm"), self.area.value(), self.gauge.value())
        ds.add_column("Strain", e, "strain", "%")
        ds.add_column("Stress", s, "stress", "MPa")
        ds.metadata["area_mm2"] = self.area.value()
        ds.metadata["gauge_length_mm"] = self.gauge.value()
        ds.default_x, ds.default_y = "Strain", "Stress"
        self.store.notify_changed()
        self.ds_combo.select(ds)
        self._on_ds()
        self.c_e.set_column("Strain")
        self.c_s.set_column("Stress")
        self._msg("Added Stress (MPa) and Strain (%) columns — now press 'Run cyclic analysis'.")
        return True

    def params(self) -> dict:
        return dict(compression=["auto", "yes", "no"][self.compression.currentIndex()],
                    prominence=self.prom.value() or None, plateau_frac=self.pfrac.value() / 100,
                    plateau_window_frac=self.pwin.value() / 100,
                    plateau_ref="range" if self.pref.currentIndex() == 0 else "absolute")

    def run(self) -> list[A.CycleMetrics] | None:
        return self.safe("Cyclic analysis", self._run)

    def _run(self) -> list[A.CycleMetrics] | None:
        ds = self.ds_combo.dataset()
        ec, sc, nc = self.c_e.column(), self.c_s.column(), self.c_n.column()
        if ds is None:
            self._msg("Select a cyclic dataset.", True)
            return None
        if not ec or not sc:
            if self.c_F.column() and self.c_d.column():
                self._msg("No stress/strain columns: use 'Compute stress–strain' (force/displacement + geometry) "
                          "first.", True)
            else:
                self._msg("Select strain and stress columns.", True)
            return None
        for col, dim in ((ec, "strain"), (sc, "stress")):
            if ds.channel(col).dimension != dim:
                self._msg(f"Column '{col}' is {ds.channel(col).dimension}, not {dim}.", True)
                return None
        e, s = ds.values(ec, "%"), ds.values(sc, "MPa")
        cyc = ds.values(nc) if nc else None
        p = self.params()
        tcol = ds.first_of("time")
        tt = ds.values(tcol) if tcol else None
        fn = lambda: A.analyse_cyclic(e, s, cycle=cyc, time=tt, **p)  # noqa: E731
        if e.size > BIG_CYCLIC_POINTS:
            out = {}
            run_task(self, f"Analysing {e.size:,} points…", fn,
                     on_done=lambda r: out.setdefault("r", self._finish(ds, ec, sc, nc, r)),
                     on_error=lambda tb, exc: self._fail("Cyclic analysis", exc))
            return out.get("r")
        return self._finish(ds, ec, sc, nc, fn())

    def _finish(self, ds, ec, sc, nc, res: A.CyclicResult):
        return self.safe("Cyclic analysis", self._finish_, ds, ec, sc, nc, res)

    def _finish_(self, ds, ec, sc, nc, res: A.CyclicResult):
        metrics = res.metrics
        if not metrics:
            self._msg("No cycles found.", True, res.warnings)
            return None
        self.result, self.cycles, self.metrics, self.last_result = res, res.cycles, metrics, metrics
        self._ran = True
        self.fatigue_ds = None
        self.source_ds = ds
        warnings = list(res.warnings)
        df = A.fatigue_summary(metrics)
        rows = df.values.tolist()
        for r in rows:
            r[0] = int(r[0])
        inc = {i for i, m in enumerate(metrics) if not m.complete}
        self.cycle_table.set_rows(rows, list(df.columns), warn_rows=inc)
        ov = A.fatigue_overview(metrics)
        self._compute_fits(metrics)
        trows, widx = self.table_rows(ov, warnings)
        self.table.set_rows(trows, ["Quantity", "Value", "Unit"], warn_rows=widx)
        params = {**res.params, "columns": {"strain": ec, "stress": sc, "cycle": nc}}
        self._record("Cyclic", params, ds, warnings, [self.table, self.cycle_table, self.fit_table])
        self._results(ds, ov, warnings)
        self._draw(ds, ec, sc)
        self._msg(f"{len(metrics)} cycle(s) analysed.", False, warnings)
        return metrics

    def _compute_fits(self, metrics) -> None:
        self.fits = {}
        df = A.fatigue_summary([m for m in metrics if m.complete] or metrics)
        rows = []
        thr_user = self.stab_thr.value() or None
        for col in _FIT_METRICS:
            dim = A.FATIGUE_COLUMNS[col][1]
            try:
                fit = A.fit_saturating(df["Cycle"].to_numpy(float), df[col].to_numpy(float),
                                       thr_user or A.FIT_THRESHOLDS.get(dim))
            except ValueError as exc:
                log.info("fatigue fit %s: %s", col, exc)
                continue
            self.fits[col] = fit
            st = int(fit.stabilisation_cycle) if np.isfinite(fit.stabilisation_cycle) else "not reached"
            rows.append([col, fit.a, fit.saturation, fit.tau, fit.r2, st])
        self.fit_table.set_rows(rows)

    def _draw(self, ds, ec, sc):
        self.plot.clear()
        subs = A.cycle_datasets(ds, self.cycles, every=self.every.value())
        cmap = colormaps["viridis"]
        nums = [c.number for c in self.cycles]
        lo, hi = min(nums), max(nums)
        many = len(self.cycles) > 8
        use_cbar = many and hasattr(self.plot, "add_colorbar")
        for sub in subs:
            n = sub.metadata["cycle"]
            col = to_hex(cmap(0.0 if hi == lo else (n - lo) / (hi - lo)))
            lab = f"Cycle {n}"
            if use_cbar or (many and n not in (lo, hi)):
                lab = f"_Cycle {n}"           # matplotlib: leading '_' → not in the legend
            self.plot.add_trace(sub, ec, sc, label=lab, color=col, linewidth=1.0)
            if self.add_cycles.isChecked():
                self.store.add(sub)
        if use_cbar:
            self.plot.add_colorbar("viridis", lo, hi, "Cycle number")
        r = self.result
        es = r.strain_sign if r is not None else 1.0
        ss = r.stress_sign if r is not None else 1.0
        sel = (self.metrics[0], self.metrics[-1]) if len(self.metrics) > 1 else (self.metrics[0],)
        for m in sel:
            self.label(es * m.plateau_strain_pct, ss * m.sigma_ms_MPa, f"σ_ms {m.sigma_ms_MPa:.0f} MPa (N={m.cycle})",
                       "#d62728", "^")
            if np.isfinite(m.sigma_as_MPa):
                self.label(es * m.plateau_strain_pct, ss * m.sigma_as_MPa,
                           f"σ_as {m.sigma_as_MPa:.0f} MPa (N={m.cycle})", "#1f77b4", "v")
            if np.isfinite(m.residual_strain_pct):
                self.label(es * m.residual_strain_pct, 0.0, f"ε_res {m.residual_strain_pct:.2f} % (N={m.cycle})",
                           "black", "x")
        self.plot.redraw()

    def plot_fatigue(self) -> Dataset | None:
        return self.safe("Fatigue summary", self._plot_fatigue)

    def _plot_fatigue(self) -> Dataset | None:
        if not self.metrics:
            if self.run() is None:
                return None
        if self.fatigue_ds is None:
            src = getattr(self, "source_ds", None)
            name = f"{src.name} – fatigue summary" if src is not None else "Fatigue summary"
            self.fatigue_ds = A.fatigue_dataset(self.metrics, name, source=src)
            if self.last_provenance:
                self.fatigue_ds.metadata["provenance"] = self.last_provenance
            self.store.add(self.fatigue_ds)
        header = self.metric.currentText()
        col = A.bare_column_name(header)
        self.plot.clear()
        self.plot.add_trace(self.fatigue_ds, "Cycle", col, label=col, marker="o", markersize=4, linewidth=1.0)
        msg = f"Fatigue summary dataset '{self.fatigue_ds.name}' added to the store."
        if self.fit_cb.isChecked():
            fit = self.fits.get(header)
            if fit is None:
                try:
                    dim = A.FATIGUE_COLUMNS[header][1]
                    df = A.fatigue_summary([m for m in self.metrics if m.complete] or self.metrics)
                    fit = A.fit_saturating(df["Cycle"].to_numpy(float), df[header].to_numpy(float),
                                           self.stab_thr.value() or A.FIT_THRESHOLDS.get(dim))
                    self.fits[header] = fit
                except (ValueError, KeyError) as exc:
                    fit = None
                    msg += f" (no trend fit: {exc})"
            if fit is not None:
                self.plot.add_line(fit.x_fit, fit.y_fit, f"fit: τ = {fmt(fit.tau)} cycles", color="#d62728",
                                   linestyle="--", linewidth=1.2)
                st = (f"stabilised at N = {int(fit.stabilisation_cycle)}" if np.isfinite(fit.stabilisation_cycle)
                      else "not stabilised")
                msg += f" Fit: saturation {fmt(fit.saturation)}, τ = {fmt(fit.tau)} cycles, {st}."
                unit = A.FATIGUE_COLUMNS[header][1]
                u = {"strain": "%", "stress": "MPa", "energy_density": "MJ/m³"}.get(unit, "")
                self._results(self.fatigue_ds, [(f"{col} fit saturation", fit.saturation, u),
                                                (f"{col} fit τ", fit.tau, "cycles"),
                                                (f"{col} stabilisation cycle", fit.stabilisation_cycle, "#")])
        self.plot.redraw()
        self._msg(msg)
        return self.fatigue_ds


# =========================================================================== panel
class AnalysisPanel(QWidget):
    """Right-hand analysis dock with DSC, mechanical, cyclic, batch and elastocaloric tabs."""

    analysisFinished = Signal(str, object)
    statusMessage = Signal(str)

    def __init__(self, plot_panel, store: DatasetStore, parent=None):
        super().__init__(parent)
        from .analysis_batch import BatchTab
        from .analysis_elastocaloric import ElastocaloricTab
        self.plot_panel = plot_panel
        self.store = store
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.ElideRight)
        self.dsc_tab = DSCTab(self)
        self.mech_tab = MechanicalTab(self)
        self.cyclic_tab = CyclicTab(self)
        self.batch_tab = BatchTab(self)
        self.ec_tab = ElastocaloricTab(self)
        for t, name, tip in ((self.dsc_tab, "DSC", "DSC: transformation temperatures and latent heats"),
                             (self.mech_tab, "Mechanical", "Tensile / compression: E, yield, UTS, σ_SIM"),
                             (self.cyclic_tab, "Cyclic", "Cyclic superelastic tests and functional fatigue"),
                             (self.batch_tab, "Batch", "Same analysis on many datasets + comparison table"),
                             (self.ec_tab, "eCaloric", "Elastocaloric: Clausius–Clapeyron, measured ΔT, COP")):
            i = self.tabs.addTab(t, name)
            self.tabs.setTabToolTip(i, tip)
        lay.addWidget(self.tabs)
        for t in (self.dsc_tab, self.mech_tab, self.cyclic_tab):
            t.btn_run.clicked.connect(lambda _=False, tab=t: self.analysisFinished.emit(tab.TOOL, tab.last_result))
        self._refreshing = False
        store.subscribe(self._on_store_changed)
        self.refresh()

    @property
    def all_tabs(self):
        return (self.dsc_tab, self.mech_tab, self.cyclic_tab, self.batch_tab, self.ec_tab)

    def _on_store_changed(self) -> None:
        try:
            self.refresh()
        except RuntimeError:     # widget already deleted
            pass

    def refresh(self) -> None:
        """Re-populate dataset/column pickers from the store."""
        if self._refreshing:
            return
        self._refreshing = True
        try:
            for t in (self.dsc_tab, self.mech_tab, self.cyclic_tab):
                cur = {c: c.column() for c in t.findChildren(ColumnCombo)}
                prev = t.ds_combo.currentData()
                t.ds_combo.refresh()
                t._on_ds()
                if t.ds_combo.currentData() == prev:
                    for c, col in cur.items():
                        if col is not None:
                            c.set_column(col)
                if t is self.cyclic_tab:
                    t.gss.setVisible(not (t.c_e.column() and t.c_s.column()))
            self.batch_tab.refresh()
            self.ec_tab.refresh()
        finally:
            self._refreshing = False

    def select_dataset(self, ds: Dataset) -> None:
        """Select ``ds`` in the tab matching its kind and show that tab."""
        tab = {"dsc": self.dsc_tab, "tensile": self.mech_tab, "compression": self.mech_tab,
               "cyclic": self.cyclic_tab}.get(ds.kind)
        for t in (self.dsc_tab, self.mech_tab, self.cyclic_tab):
            t.ds_combo.select(ds)
        if tab is not None:
            self.tabs.setCurrentWidget(tab)

    # ------------------------------------------------------------------ Q6 defaults / presets
    def current_defaults(self) -> dict:
        """Analysis parameters of the DSC / mechanical / cyclic tabs (JSON-serialisable), e.g. for an
        import preset: ``{'dsc': {...}, 'mechanical': {...}, 'cyclic': {...}}``."""
        return {"dsc": self.dsc_tab.defaults(), "mechanical": self.mech_tab.defaults(),
                "cyclic": self.cyclic_tab.defaults()}

    def apply_defaults(self, defaults: dict) -> None:
        """Apply a dict as produced by :meth:`current_defaults` (missing keys are left unchanged)."""
        for key, tab in (("dsc", self.dsc_tab), ("mechanical", self.mech_tab), ("cyclic", self.cyclic_tab)):
            if isinstance((defaults or {}).get(key), dict):
                tab.apply_values({k: v for k, v in defaults[key].items() if k in tab.STATE_WIDGETS})

    def _apply_ds_defaults(self, tab: TabBase, ds: Dataset) -> None:
        d = (ds.metadata or {}).get("analysis_defaults")
        if isinstance(d, dict) and isinstance(d.get(tab.KEY), dict):
            tab.apply_values({k: v for k, v in d[tab.KEY].items() if k in tab.STATE_WIDGETS})
            tab._msg(f"Analysis defaults from the import preset of '{ds.name}' applied.")

    # ------------------------------------------------------------------ Q4 session state
    def get_state(self) -> dict:
        st = {"current_tab": self.tabs.currentIndex()}
        for t in self.all_tabs:
            try:
                st[t.KEY] = t.get_state()
            except Exception:  # noqa: BLE001
                log.exception("get_state of %s failed", t.KEY)
        return st

    def set_state(self, state: dict, id_map: dict) -> None:
        state = state or {}
        self.refresh()
        for t in self.all_tabs:
            sub = state.get(t.KEY)
            if not isinstance(sub, dict):
                continue
            try:
                t.set_state(sub, id_map)
            except Exception:  # noqa: BLE001
                log.exception("set_state of %s failed", t.KEY)
        for t in self.all_tabs:
            sub = state.get(t.KEY) or {}
            if sub.get("ran") and hasattr(t, "rerun_from_state"):
                t.safe(f"Restoring {t.TOOL}", t.rerun_from_state)
            elif sub.get("ran"):
                t.safe(f"Restoring {t.TOOL}", t.run)
        if isinstance(state.get("current_tab"), int):
            self.tabs.setCurrentIndex(state["current_tab"])
