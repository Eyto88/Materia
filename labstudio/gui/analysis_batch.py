"""Batch tab: same analysis on many datasets → one comparison table (+ mean ± SD, metric vs parameter)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..core import batch as B
from .analysis_widgets import SummaryTable, TabBase, button, combo, hrow, scroll
from .worker import run_task

KIND_LABELS = ["DSC", "Tensile", "Compression", "Cyclic"]
KIND_KEYS = ["dsc", "tensile", "compression", "cyclic"]


class BatchTab(TabBase):
    TOOL = "Batch"
    KEY = "batch"
    STATE_WIDGETS = ("kind", "only_kind", "psource", "preset", "regex", "meta_key", "plabel", "metric", "grouped")

    def __init__(self, panel):
        super().__init__(panel)
        self.result: B.BatchResult | None = None
        self.derived = None
        inner = QWidget()
        lay = QVBoxLayout(inner)
        gd = QGroupBox("Datasets")
        f = QFormLayout(gd)
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.kind = combo(KIND_LABELS, "Analysis to run on every checked dataset, with the parameters currently "
                                       "set in the corresponding tab (DSC / Mechanical / Cyclic)")
        self.only_kind = QCheckBox("Only datasets of this kind")
        self.only_kind.setChecked(True)
        self.only_kind.setToolTip("Filter the list by dataset kind")
        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        self.list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.list.setToolTip("Tick the datasets to analyse")
        f.addRow("Analysis", self.kind)
        f.addRow(self.only_kind)
        f.addRow(self.list)
        f.addRow(hrow(button("All", "Check all listed datasets", lambda: self._check_all(True)),
                      button("None", "Uncheck all", lambda: self._check_all(False))))
        lay.addWidget(gd)

        gp = QGroupBox("Parameter (for grouping / metric vs parameter)")
        g = QFormLayout(gp)
        g.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.psource = combo(["Name regex", "Metadata key", "None"],
                             "Where the parameter comes from: a regular expression on the dataset name (first "
                             "capture group = the number) or a metadata key")
        self.preset = combo(list(B.PARAMETER_PRESETS), "Ready-made regular expressions")
        self.regex = QLineEdit(B.PARAMETER_PRESETS["Temperature (°C)"])
        self.regex.setToolTip("Regular expression; group 1 = the number (decimal comma accepted)")
        self.meta_key = QLineEdit("")
        self.meta_key.setPlaceholderText("e.g. anneal_temperature_C, thickness_mm")
        self.plabel = QLineEdit("Temperature (°C)")
        self.plabel.setToolTip("Parameter name with unit in parentheses, e.g. 'Strain amplitude (%)'")
        self.preset.currentTextChanged.connect(self._on_preset)
        g.addRow("Source", self.psource)
        g.addRow("Preset", self.preset)
        g.addRow("Regex", self.regex)
        g.addRow("Metadata key", self.meta_key)
        g.addRow("Label", self.plabel)
        lay.addWidget(gp)

        self.btn_run = button("Run batch analysis", "Analyse every checked dataset (errors are reported per row)",
                              self.run)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.status)
        lay.addWidget(QLabel("Comparison (one row per sample)"))
        self.table = SummaryTable(["Sample"])
        lay.addWidget(self.table)
        lay.addWidget(QLabel("Mean ± SD by parameter (repeats)"))
        self.stats_table = SummaryTable(["Parameter"], min_height=90)
        lay.addWidget(self.stats_table)
        gm = QGroupBox("Plot metric vs parameter")
        m = QFormLayout(gm)
        m.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.metric = combo([], "Metric (y) to plot against the parameter (x)")
        self.grouped = QCheckBox("Mean ± SD of repeats")
        self.grouped.setToolTip("Plot the group means with ±SD bars instead of every sample")
        self.btn_plot = button("Plot metric vs parameter", "Creates a derived dataset in the store and plots it",
                               self.plot_metric)
        m.addRow("Metric", self.metric)
        m.addRow(self.grouped)
        m.addRow(self.btn_plot)
        lay.addWidget(gm)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(inner))
        self.kind.currentIndexChanged.connect(lambda *_: self.refresh())
        self.only_kind.toggled.connect(lambda *_: self.refresh())

    # ------------------------------------------------------------------ list
    def _on_preset(self, name: str) -> None:
        if name in B.PARAMETER_PRESETS:
            self.regex.setText(B.PARAMETER_PRESETS[name])
            self.plabel.setText(name if name != "First number" else "Parameter")

    def kind_key(self) -> str:
        return KIND_KEYS[self.kind.currentIndex()]

    def refresh(self) -> None:
        checked = set(self.checked_ids())
        k = self.kind_key()
        kinds = {"dsc": ("dsc",), "tensile": ("tensile",), "compression": ("compression",),
                 "cyclic": ("cyclic",)}[k]
        self.list.blockSignals(True)
        self.list.clear()
        for ds in self.store:
            if self.only_kind.isChecked() and ds.kind not in kinds:
                continue
            it = QListWidgetItem(f"{ds.name}  [{ds.kind}]")
            it.setData(Qt.UserRole, ds.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if ds.id in checked else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)

    def _check_all(self, on: bool) -> None:
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def checked_ids(self) -> list[int]:
        return [self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked]

    def set_checked(self, ids) -> None:
        ids = set(ids)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ids else Qt.Unchecked)

    # ------------------------------------------------------------------ run
    def params(self) -> dict:
        k = self.kind_key()
        p = self.panel
        if k == "dsc":
            return p.dsc_tab.params()
        if k == "cyclic":
            return p.cyclic_tab.params()
        mp = p.mech_tab.params()
        mp["compliance_mm_per_N"] = p.mech_tab.compliance.value()
        return mp

    def run(self):
        return self.safe("Batch analysis", self._run)

    def _run(self):
        dss = [d for d in (self.store.get(i) for i in self.checked_ids()) if d is not None]
        if not dss:
            self._msg("Tick at least one dataset.", True)
            return None
        src = self.psource.currentIndex()
        regex = self.regex.text().strip() if src == 0 else None
        key = self.meta_key.text().strip() if src == 1 else None
        label = self.plabel.text().strip() or "Parameter"
        k, params = self.kind_key(), self.params()
        fn = lambda: B.batch_analyse(dss, k, params, parameter_regex=regex or None,  # noqa: E731
                                     parameter_key=key or None, parameter_label=label)
        out = {}
        run_task(self, f"Batch analysis of {len(dss)} datasets…", fn,
                 on_done=lambda r: out.setdefault("r", self.safe("Batch analysis", self._show, r)),
                 on_error=lambda tb, exc: self._fail("Batch analysis", exc))
        return out.get("r")

    def _show(self, res: B.BatchResult) -> B.BatchResult:
        self.result = self.last_result = res
        self._ran = True
        hdr, rows = res.table()
        warn = {i for i, r in enumerate(res.rows) if r.warnings or r.error}
        self.table.set_rows(rows, hdr, warn_rows=warn)
        shdr, srows = B.group_stats(res)
        self.stats_table.set_rows(srows, shdr)
        self.last_provenance = res.provenance
        self.table.set_provenance(res.provenance)
        self.stats_table.set_provenance(res.provenance)
        cur = self.metric.currentText()
        self.metric.blockSignals(True)
        self.metric.clear()
        self.metric.addItems(res.columns)
        i = self.metric.findText(cur)
        self.metric.setCurrentIndex(i if i >= 0 else 0)
        self.metric.blockSignals(False)
        n_err = sum(1 for r in res.rows if r.error)
        n_w = sum(1 for r in res.rows if r.warnings)
        warns = [f"{r.name}: {r.error or '; '.join(r.warnings)}" for r in res.rows if r.error or r.warnings]
        self._msg(f"{len(res.rows)} dataset(s) analysed ({n_err} error(s), {n_w} with warnings).",
                  n_err == len(res.rows), warns[:8])
        return res

    def plot_metric(self):
        return self.safe("Plot metric vs parameter", self._plot_metric)

    def _plot_metric(self):
        if self.result is None:
            self._msg("Run the batch analysis first.", True)
            return None
        metric = self.metric.currentText()
        if not metric:
            self._msg("No metric available.", True)
            return None
        if not any(r.parameter is not None for r in self.result.rows):
            self._msg("No parameter values: set a name regex or a metadata key and re-run.", True)
            return None
        grouped = self.grouped.isChecked()
        ds = B.metric_vs_parameter_dataset(self.result, metric, grouped=grouped)
        self.store.add(ds)
        self.derived = ds
        cols = ds.columns()
        x, y = cols[0], cols[1]
        self.plot.clear()
        self.plot.add_trace(ds, x, y, label=y, marker="o", markersize=5, linewidth=1.0)
        if grouped:
            xs, ys, sd = ds.values(x), ds.values(y), ds.values(cols[2])
            for xi, yi, si in zip(xs, ys, sd):
                if si == si:   # finite SD → error bar
                    self.plot.add_line([xi, xi], [yi - si, yi + si], None, color="black", linewidth=1.0)
        self.plot.redraw()
        self._msg(f"Derived dataset '{ds.name}' added to the store and plotted.")
        return ds

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        st = super().get_state()
        st["checked"] = self.checked_ids()
        return st

    def set_state(self, state: dict, id_map: dict) -> None:
        self.apply_values({k: v for k, v in (state.get("params") or {}).items() if k != "metric"})
        self.refresh()
        self.set_checked([id_map[int(i)].id for i in state.get("checked", []) if int(i) in id_map])
        self._pending_metric = (state.get("params") or {}).get("metric")

    def rerun_from_state(self):
        r = self.run()
        if getattr(self, "_pending_metric", None):
            self.apply_values({"metric": self._pending_metric})
        return r
