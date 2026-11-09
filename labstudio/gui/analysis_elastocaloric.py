"""Elastocaloric tab: Clausius–Clapeyron (σ vs T, ΔS, L vs DSC ΔH) and measured ΔT / material COP."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFormLayout, QGroupBox, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QSizePolicy, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..core import analysis as A
from ..core import elastocaloric as EC
from ..core.batch import dsc_columns, mech_arrays
from ..core.dataset import Dataset
from .analysis_widgets import DatasetCombo, SummaryTable, TabBase, button, combo, dspin, fmt, hrow, scroll

CC_HEADERS = ["Sample", "T (°C)", "σ (MPa)", "ε_tr (%)", "Source"]


def _has_temp_log(ds: Dataset) -> bool:
    return bool(ds.first_of("time") and ds.first_of("temperature"))


def _is_mech(ds: Dataset) -> bool:
    return bool((ds.first_of("strain") and ds.first_of("stress")) or (ds.first_of("force") and
                                                                      ds.first_of("displacement", "length")))


class ElastocaloricTab(TabBase):
    TOOL = "Elastocaloric"
    KEY = "elastocaloric"
    STATE_WIDGETS = ("sig_source", "t_regex", "rho", "eps_override", "T_ref", "dsc_which", "offset", "rho2", "cp",
                     "base_win", "lag")

    def __init__(self, panel):
        super().__init__(panel)
        self.cc: EC.ClausiusClapeyron | None = None
        self.caloric: EC.CaloricResult | None = None
        self.cc_ds: Dataset | None = None
        self.caloric_ds: Dataset | None = None
        inner = QWidget()
        lay = QVBoxLayout(inner)

        # ------------------------------------------------------- Clausius–Clapeyron
        gc = QGroupBox("Clausius–Clapeyron (σ vs T)")
        c = QVBoxLayout(gc)
        self.cc_list = QListWidget()
        self.cc_list.setMinimumHeight(90)
        self.cc_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.cc_list.setToolTip("Tick the tensile / cyclic tests made at different temperatures")
        c.addWidget(QLabel("Tests at different temperatures:"))
        c.addWidget(self.cc_list)
        f = QFormLayout()
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.sig_source = combo(["Auto (σ_SIM, else σ_ms)", "σ_SIM (two-tangent)", "σ_ms (mid-plateau)"],
                                "Stress taken from every test: transformation start σ_SIM on the (first) loading "
                                "branch, or the upper plateau σ_ms at mid-plateau of the first complete cycle")
        self.t_regex = QLineEdit(EC.TEMPERATURE_REGEX)
        self.t_regex.setToolTip("Regex for the test temperature in the dataset name when it is not in the "
                                "metadata (group 1 = number in °C)")
        f.addRow("Stress", self.sig_source)
        f.addRow("T regex", self.t_regex)
        c.addLayout(f)
        self.btn_collect = button("Collect points", "Analyse the ticked tests: temperature, σ and ε_tr into the "
                                                    "editable table below", self.collect_points)
        c.addWidget(self.btn_collect)
        self.cc_table = QTableWidget(0, len(CC_HEADERS))
        self.cc_table.setHorizontalHeaderLabels(CC_HEADERS)
        self.cc_table.verticalHeader().setVisible(False)
        self.cc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.cc_table.horizontalHeader().setStretchLastSection(True)
        self.cc_table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.cc_table.setMinimumHeight(110)
        self.cc_table.setToolTip("Editable: correct a temperature, stress or ε_tr, or add rows typed by hand")
        c.addWidget(self.cc_table)
        c.addWidget(hrow(button("Add row", "Add a hand-entered point", self.add_cc_row),
                         button("Remove row", "Remove the selected point", self.remove_cc_row)))
        f2 = QFormLayout()
        f2.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.rho = dspin(EC.RHO_NITI_G_CM3, 0.1, 30, 3, 0.01, "g/cm³", "Density ρ (NiTi ≈ 6.45 g/cm³)")
        self.eps_override = dspin(0, 0, 20, 3, 0.1, "%", "Transformation strain ε_tr used for ΔS (0 = mean of the "
                                                          "plateau lengths found in the tests / table)",
                                  special="from data")
        self.T_ref = dspin(-274, -274, 1000, 1, 1, "°C", "Temperature for L = T·|ΔS| (auto = mean test "
                                                          "temperature)", special="auto")
        self.dsc_combo = DatasetCombo(self.store, ("dsc",), optional=True,
                                      accept=lambda d: d.first_of("temperature") is not None and
                                      d.first_of("specific_heat_flow", "heat_flow") is not None)
        self.dsc_combo.setToolTip("Optional DSC dataset to cross-check L against ΔH (analysed with the DSC tab's "
                                  "parameters)")
        self.dsc_which = combo(["Heating total ΔH", "Cooling total ΔH"], "Which DSC latent heat to compare with")
        f2.addRow("ρ", self.rho)
        f2.addRow("ε_tr", self.eps_override)
        f2.addRow("T for L", self.T_ref)
        f2.addRow("DSC check", self.dsc_combo)
        f2.addRow("", self.dsc_which)
        c.addLayout(f2)
        self.btn_cc = button("Fit dσ/dT → ΔS, L", "Linear fit of σ vs T (R²), ΔS = −(dσ/dT)·ε_tr/ρ, L = T·|ΔS|, "
                                                 "cross-check with DSC; plots σ vs T with the fit line",
                             self.run_cc)
        c.addWidget(self.btn_cc)
        self.cc_status = QLabel("")
        self.cc_status.setWordWrap(True)
        self.cc_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        c.addWidget(self.cc_status)
        self.cc_summary = SummaryTable(min_height=110)
        c.addWidget(self.cc_summary)
        lay.addWidget(gc)

        # ------------------------------------------------------- measured ΔT / COP
        gt = QGroupBox("Measured ΔT and material COP")
        t = QFormLayout(gt)
        t.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.log_combo = DatasetCombo(self.store, ("generic",), accept=_has_temp_log)
        self.log_combo.setToolTip("Temperature log (thermocouple / IR): any dataset with time + temperature")
        self.mech_combo = DatasetCombo(self.store, ("cyclic",), accept=lambda d: _is_mech(d) and
                                       d.first_of("time") is not None)
        self.mech_combo.setToolTip("Cyclic stress–strain test with a time column")
        self.offset = dspin(0, -1e6, 1e6, 2, 0.5, "s", "Time offset: t_mechanical = t_log + offset")
        self.rho2 = dspin(EC.RHO_NITI_G_CM3, 0.1, 30, 3, 0.01, "g/cm³", "Density ρ to convert W (MJ/m³) to J/kg")
        self.cp = dspin(EC.CP_NITI_J_GK, 0.01, 10, 3, 0.01, "J/(g·K)", "Specific heat c_p for Q_c = c_p·|ΔT_cool|")
        self.base_win = dspin(0, 0, 1e5, 2, 1, "s", "Window before each cycle for the baseline temperature "
                                                    "(auto = 5 % of the cycle duration)", special="auto")
        self.lag = dspin(10, 0, 100, 0, 5, "% of half-cycle", "Extend the loading/unloading windows by this "
                                                              "fraction (sensor thermal lag)")
        t.addRow("Temperature log", self.log_combo)
        t.addRow("Cyclic test", self.mech_combo)
        t.addRow("Time offset", self.offset)
        t.addRow("ρ", self.rho2)
        t.addRow("c_p", self.cp)
        t.addRow("Baseline window", self.base_win)
        t.addRow("Thermal lag", self.lag)
        self.btn_cop = button("Compute ΔT and COP", "Per cycle: ΔT on loading/unloading vs the pre-cycle baseline, "
                                                    "work W = loop area, COP = c_p·|ΔT_cool| / W, cooling power",
                              self.run_caloric)
        t.addRow(self.btn_cop)
        t.addRow(hrow(button("Plot ΔT vs cycle", "Derived dataset (cycle vs ΔT) → store + plot",
                             lambda: self.plot_caloric("dT")),
                      button("Plot COP vs cycle", "Derived dataset (cycle vs COP) → store + plot",
                             lambda: self.plot_caloric("COP"))))
        lay.addWidget(gt)
        lay.addWidget(self.status)
        self.table = SummaryTable(min_height=110)
        lay.addWidget(self.table)
        lay.addWidget(QLabel("Per cycle"))
        self.cycle_table = SummaryTable(EC.CaloricResult.HEADERS)
        lay.addWidget(self.cycle_table)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll(inner))

    # ------------------------------------------------------------------ store
    def refresh(self) -> None:
        checked = set(self.cc_checked_ids())
        self.cc_list.clear()
        for ds in self.store:
            if ds.kind in ("tensile", "compression", "cyclic") and _is_mech(ds):
                it = QListWidgetItem(f"{ds.name}  [{ds.kind}]")
                it.setData(Qt.UserRole, ds.id)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if ds.id in checked else Qt.Unchecked)
                self.cc_list.addItem(it)
        for cb in (self.dsc_combo, self.log_combo, self.mech_combo):
            cb.refresh()

    def cc_checked_ids(self) -> list[int]:
        return [self.cc_list.item(i).data(Qt.UserRole) for i in range(self.cc_list.count())
                if self.cc_list.item(i).checkState() == Qt.Checked]

    def set_cc_checked(self, ids) -> None:
        ids = set(ids)
        for i in range(self.cc_list.count()):
            it = self.cc_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in ids else Qt.Unchecked)

    # ------------------------------------------------------------------ CC table
    def _set_cc_row(self, i: int, vals: list) -> None:
        for j, v in enumerate(vals):
            txt = v if isinstance(v, str) else ("" if v is None or not np.isfinite(v) else fmt(v))
            self.cc_table.setItem(i, j, QTableWidgetItem(txt))

    def add_cc_row(self, vals: list | None = None) -> None:
        i = self.cc_table.rowCount()
        self.cc_table.insertRow(i)
        self._set_cc_row(i, vals or [f"point {i + 1}", float("nan"), float("nan"), float("nan"), "user"])

    def remove_cc_row(self) -> None:
        rows = sorted({ix.row() for ix in self.cc_table.selectedIndexes()}, reverse=True)
        for r in rows or [self.cc_table.rowCount() - 1]:
            if r >= 0:
                self.cc_table.removeRow(r)

    def cc_points(self) -> list[list]:
        out = []
        for i in range(self.cc_table.rowCount()):
            row = []
            for j in range(len(CC_HEADERS)):
                it = self.cc_table.item(i, j)
                txt = it.text().strip() if it else ""
                if j in (1, 2, 3):
                    try:
                        row.append(float(txt.replace(",", ".")))
                    except ValueError:
                        row.append(float("nan"))
                else:
                    row.append(txt)
            out.append(row)
        return out

    def collect_points(self):
        return self.safe("Collect points", self._collect)

    def _collect(self):
        dss = [d for d in (self.store.get(i) for i in self.cc_checked_ids()) if d is not None]
        if not dss:
            self._cc_msg("Tick at least two tests at different temperatures.", True)
            return None
        src = ["auto", "σ_SIM", "σ_ms"][self.sig_source.currentIndex()]
        rx = self.t_regex.text().strip() or None
        self.cc_table.setRowCount(0)
        warns = []
        pts = []
        for ds in dss:
            try:
                p = EC.cc_point_from_dataset(ds, source=src, temperature_regex=rx,
                                             compression=ds.kind == "compression")
            except Exception as exc:  # noqa: BLE001
                warns.append(f"{ds.name}: {exc}")
                continue
            pts.append(p)
            warns += [f"{ds.name}: {w}" for w in p.warnings]
            self.add_cc_row([p.name, p.T_C, p.sigma_MPa, p.eps_tr_pct, p.source])
        self._cc_msg(f"{len(pts)} point(s) collected — check/edit the table, then 'Fit'.", False, warns)
        return pts

    def _cc_msg(self, text, error=False, warnings=()):
        self._msg(text, error, list(warnings))
        self.cc_status.setText(self.status.text())
        self.cc_status.setStyleSheet(self.status.styleSheet())

    def _dsc_dH(self) -> float | None:
        ds = self.dsc_combo.dataset()
        if ds is None:
            return None
        Tc, hc, tc = dsc_columns(ds)
        dt = self.panel.dsc_tab
        md = ds.metadata
        res = A.analyse_dsc(ds.values(Tc), ds.values(hc), md.get("heating_rate_K_per_min"), md.get("mass_mg"),
                            time=ds.values(tc) if tc else None, hf_unit=ds.channel(hc).unit or "W/g",
                            rate_source=md.get("heating_rate_source"), **dt.params())
        key = "heating" if self.dsc_which.currentIndex() == 0 else "cooling"
        segs = [s for s in res.segments if s.direction == key and s.peaks]
        if not segs:
            return None
        return float(np.nansum([p.enthalpy_J_per_g for p in segs[-1].peaks]))

    def run_cc(self):
        return self.safe("Clausius–Clapeyron", self._run_cc)

    def _run_cc(self):
        if self.cc_table.rowCount() == 0:
            self._collect()
        pts = [p for p in self.cc_points() if np.isfinite(p[1]) and np.isfinite(p[2])]
        if len(pts) < 2:
            self._cc_msg("Need at least two points with temperature and stress.", True)
            return None
        eps = self.eps_override.value() or [p[3] for p in pts]
        Tref = None if self.T_ref.value() <= -273.15 else self.T_ref.value()
        dH = self._dsc_dH()
        cc = EC.clausius_clapeyron([p[1] for p in pts], [p[2] for p in pts], eps, self.rho.value(), Tref, dH,
                                   names=[p[0] for p in pts], sources=[p[4] for p in pts])
        self.cc = self.last_result = cc
        self._ran = True
        warnings = list(cc.warnings)
        if self.dsc_combo.dataset() is not None and dH is None:
            warnings.append("No DSC peaks found in the selected DSC dataset for the cross-check.")
        rows = cc.rows()
        trows, widx = self.table_rows(rows, warnings)
        self.cc_summary.set_rows(trows, ["Quantity", "Value", "Unit"], warn_rows=widx)
        self._record("Clausius–Clapeyron", {**cc.params, "stress_source": self.sig_source.currentText(),
                                            "dsc_dataset": getattr(self.dsc_combo.dataset(), "name", None)},
                     None, warnings, [self.cc_summary])
        # σ vs T derived dataset + fit line
        T = np.array([p.T_C for p in cc.points])
        o = np.argsort(T)
        ds = Dataset.from_arrays("σ vs T (Clausius–Clapeyron)",
                                 {"Temperature": (T[o], "temperature", "°C"),
                                  "Transformation stress": (np.array([p.sigma_MPa for p in cc.points])[o],
                                                            "stress", "MPa")}, kind="generic")
        ds.metadata["provenance"] = self.last_provenance
        if self.cc_ds is not None and self.store.get(self.cc_ds.id) is not None:
            self.store.remove(self.cc_ds)
        self.cc_ds = ds
        self.store.add(ds)
        self.plot.clear()
        self.plot.add_trace(ds, "Temperature", "Transformation stress", label="σ (tests)", marker="o",
                            markersize=6, linestyle="none")
        x, y = cc.line()
        self.plot.add_line(x, y, f"fit: dσ/dT = {fmt(cc.slope_MPa_per_K)} MPa/K (R² = {fmt(cc.r2)})",
                           color="#d62728", linestyle="--", linewidth=1.2)
        self.plot.redraw()
        self.plot.add_results([{"Tool": "Clausius–Clapeyron", "Trace": ds.name, "Quantity": q, "Value": v,
                                "Unit": u} for q, v, u in rows])
        msg = (f"dσ/dT = {fmt(cc.slope_MPa_per_K)} MPa/K, ΔS = {fmt(cc.dS_J_per_kgK)} J/(kg·K), "
               f"L = {fmt(cc.latent_J_per_g)} J/g")
        if np.isfinite(cc.dsc_dH_J_per_g):
            msg += f" (DSC ΔH {fmt(cc.dsc_dH_J_per_g)} J/g, {cc.dsc_diff_pct:+.0f} %)"
        self._cc_msg(msg + ".", False, warnings)
        return cc

    # ------------------------------------------------------------------ ΔT / COP
    def run_caloric(self):
        return self.safe("ΔT / COP", self._run_caloric)

    def _run_caloric(self):
        lg, mds = self.log_combo.dataset(), self.mech_combo.dataset()
        if lg is None or mds is None:
            self._msg("Select a temperature log (time + temperature) and a cyclic test with a time column.", True)
            return None
        tcol, Tcol = EC.temperature_log_columns(lg)
        e, s, _src = mech_arrays(mds)
        nc = mds.first_of("count")
        res = EC.caloric_per_cycle(lg.values(tcol, "s"), lg.values(Tcol, "°C"), mds.values(mds.first_of("time"), "s"),
                                   e, s, cycle=mds.values(nc) if nc else None, offset_s=self.offset.value(),
                                   rho_g_cm3=self.rho2.value(), cp_J_per_gK=self.cp.value(),
                                   baseline_window_s=self.base_win.value() or None, lag_frac=self.lag.value() / 100,
                                   compression=self.panel.cyclic_tab.params()["compression"])
        self.caloric = self.last_result = res
        self.caloric_ds = None
        self._ran = True
        warnings = list(res.warnings)
        rows = res.rows()
        trows, widx = self.table_rows(rows, warnings)
        self.table.set_rows(trows, ["Quantity", "Value", "Unit"], warn_rows=widx)
        crows = res.table()
        for r in crows:
            r[0] = int(r[0])
        self.cycle_table.set_rows(crows, EC.CaloricResult.HEADERS,
                                  warn_rows={i for i, c in enumerate(res.cycles) if not c.complete})
        self._record("Elastocaloric ΔT/COP", {**res.params, "temperature_log": lg.name, "mechanical": mds.name},
                     mds, warnings, [self.table, self.cycle_table])
        self._results(mds, rows, warnings)
        self._msg(f"{len(res.cycles)} cycle(s): ΔT and COP computed.", False, warnings)
        return res

    def plot_caloric(self, what: str = "dT"):
        return self.safe("Plot ΔT/COP", self._plot_caloric, what)

    def _plot_caloric(self, what: str):
        if self.caloric is None and self.run_caloric() is None:
            return None
        if self.caloric_ds is None:
            mds = self.mech_combo.dataset()
            self.caloric_ds = EC.caloric_dataset(self.caloric, f"{mds.name if mds else 'Test'} – elastocaloric")
            self.caloric_ds.metadata["provenance"] = self.last_provenance
            self.store.add(self.caloric_ds)
        ds = self.caloric_ds
        self.plot.clear()
        if what == "COP":
            self.plot.add_trace(ds, "Cycle", "COP_mat", label="COP_mat", marker="o", markersize=4)
        else:
            self.plot.add_trace(ds, "Cycle", "ΔT heating", label="ΔT heating (loading)", marker="^", markersize=4,
                                color="#d62728")
            self.plot.add_trace(ds, "Cycle", "ΔT cooling", label="ΔT cooling (unloading)", marker="v",
                                markersize=4, color="#1f77b4")
        self.plot.redraw()
        self._msg(f"Derived dataset '{ds.name}' added to the store and plotted.")
        return ds

    # ------------------------------------------------------------------ state
    def get_state(self) -> dict:
        st = super().get_state()
        st["cc_checked"] = self.cc_checked_ids()
        st["cc_points"] = self.cc_points()
        st["log"] = self.log_combo.currentData()
        st["mech"] = self.mech_combo.currentData()
        st["dsc"] = self.dsc_combo.currentData()
        st["ran_cc"] = self.cc is not None
        st["ran_caloric"] = self.caloric is not None
        st["ran"] = False                       # restored by rerun_from_state
        st["rerun"] = bool(self.cc is not None or self.caloric is not None)
        return st

    def set_state(self, state: dict, id_map: dict) -> None:
        self.apply_values(state.get("params") or {})
        self.refresh()
        self.set_cc_checked([id_map[int(i)].id for i in state.get("cc_checked", []) if int(i) in id_map])
        self.cc_table.setRowCount(0)
        for row in state.get("cc_points") or []:
            self.add_cc_row(list(row))
        for key, cb in (("log", self.log_combo), ("mech", self.mech_combo), ("dsc", self.dsc_combo)):
            v = state.get(key)
            if v is not None and int(v) in id_map:
                cb.select(id_map[int(v)])
        self._restore = (bool(state.get("ran_cc")), bool(state.get("ran_caloric")))
        if any(self._restore):
            state["ran"] = True

    def rerun_from_state(self):
        cc, cal = getattr(self, "_restore", (False, False))
        if cc:
            self.run_cc()
        if cal:
            self.run_caloric()
