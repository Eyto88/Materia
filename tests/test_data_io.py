"""Tests for module A — data import (core/data_io.py and gui/import_dialog.py)."""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from labstudio.core import data_io as D

FIX = Path(__file__).parent / "fixtures_data_io"


def _spec(specs, source):
    return next(s for s in specs if s.source == source)


def _pipeline(name):
    raw = D.read_table(FIX / name)
    specs = D.guess_mapping(raw)
    kind = D.guess_kind(raw, specs)
    ds = D.build_dataset(raw, specs, kind=kind)
    return raw, specs, kind, ds


# --------------------------------------------------------------------------- instrument fixtures
def test_netzsch_proteus():
    raw, specs, kind, ds = _pipeline("netzsch_proteus.txt")
    assert raw.delimiter == ";" and raw.decimal == "." and raw.encoding == "cp1252"
    assert raw.headers == ["Temp./°C", "Time/min", "DSC/(mW/mg)"]
    assert len(raw.df) == 50
    assert any(l.startswith("#SAMPLE MASS") for l in raw.meta_lines)
    assert raw.meta["mass_mg"] == pytest.approx(15.23)
    assert raw.meta["heating_rate_K_per_min"] == pytest.approx(10.0)
    assert raw.meta["sample_name"].startswith("NiTi wire")
    assert [(s.dimension, s.unit) for s in specs] == [
        ("temperature", "°C"), ("time", "min"), ("specific_heat_flow", "mW/mg")]
    assert kind == "dsc"
    assert ds.channels["Time"].unit == "s"
    assert ds.values("Time")[-1] == pytest.approx(49 * 0.1 * 60)
    assert ds.values("Temp")[0] == pytest.approx(-80.0)
    assert ds.channels["DSC"].unit == "W/g"
    assert (ds.default_x, ds.default_y) == ("Temp", "DSC")
    assert ds.metadata["mass_mg"] == pytest.approx(15.23)
    assert ds.metadata["heating_rate_K_per_min"] == pytest.approx(10.0)


def test_ta_universal_analysis_utf16_sig_headers():
    raw, specs, kind, ds = _pipeline("ta_universal.txt")
    assert raw.encoding == "utf-16" and raw.delimiter == "\t"
    assert raw.headers == ["Time (min)", "Temperature (°C)", "Heat Flow (W/g)"]   # from Sig1..Sig3
    assert len(raw.df) == 40
    assert raw.meta["mass_mg"] == pytest.approx(12.34)
    assert raw.meta["heating_rate_K_per_min"] == pytest.approx(5.0)
    assert kind == "dsc"
    assert ds.values("Time")[1] == pytest.approx(0.2 * 60)
    assert ds.channels["Heat Flow"].dimension == "specific_heat_flow"
    assert ds.values("Temperature")[-1] == pytest.approx(120.0)


def test_ta_trios_kelvin_and_bare_units_row():
    raw, specs, kind, ds = _pipeline("ta_trios.txt")
    assert raw.units == ["s", "K", "W/g", "mW"]
    t = _spec(specs, "Temperature")
    assert (t.dimension, t.unit) == ("temperature", "K")
    hfn = _spec(specs, "Heat Flow (Normalized)")
    assert hfn.name == "Heat Flow (Normalized)" and hfn.dimension == "specific_heat_flow"
    hf = _spec(specs, "Heat Flow")
    assert (hf.dimension, hf.unit) == ("heat_flow", "mW")
    assert kind == "dsc"
    assert ds.values("Temperature")[0] == pytest.approx(200 - 273.15)     # K -> °C
    assert ds.channels["Temperature"].unit == "°C"
    assert ds.channels["Temperature"].original_unit == "K"
    assert raw.meta["mass_mg"] == pytest.approx(8.5)
    assert raw.meta["heating_rate_K_per_min"] == pytest.approx(10.0)
    assert raw.meta["sample_name"] == "NiTi_TRIOS"
    assert ds.default_y == "Heat Flow (Normalized)"


def test_mettler_stare_whitespace_aligned_units():
    raw, specs, kind, ds = _pipeline("mettler_stare.txt")
    assert raw.delimiter == " "
    assert raw.headers == ["Index", "t", "Ts", "Tr", "Value"]
    assert raw.units == [None, "[s]", "[°C]", "[°C]", "[mW]"]
    assert len(raw.df) == 30          # trailing "Results:" junk dropped
    assert _spec(specs, "t").dimension == "time"
    assert (_spec(specs, "Ts").dimension, _spec(specs, "Ts").unit) == ("temperature", "°C")
    assert (_spec(specs, "Value").dimension, _spec(specs, "Value").unit) == ("heat_flow", "mW")
    assert kind == "dsc"
    assert raw.meta["mass_mg"] == pytest.approx(10.5)
    assert raw.meta["sample_name"] == "NiTi foil"
    assert raw.meta["heating_rate_K_per_min"] == pytest.approx(10.0)
    assert ds.default_x in ("Ts", "Tr") and ds.default_y == "Value"


def test_instron_bluehill_kN_to_N_and_thousands():
    raw, specs, kind, ds = _pipeline("instron_bluehill.csv")
    assert raw.encoding == "utf-8-sig" and raw.delimiter == ","
    assert raw.headers == ["Time", "Extension", "Load"]
    assert raw.units == ["(s)", "(mm)", "(kN)"]
    assert len(raw.df) == 41           # 40 rows + thousands row; 'Maximum Load' summary dropped
    load = _spec(specs, "Load")
    assert (load.dimension, load.unit) == ("force", "kN")
    assert _spec(specs, "Extension").dimension == "displacement"
    assert kind == "tensile"
    assert ds.values("Load")[4] == pytest.approx(4 * 0.025 * 1000)      # kN -> N
    assert ds.channels["Load"].unit == "N"
    assert ds.values("Time")[-1] == pytest.approx(1234.5)               # "1,234.500"
    assert (ds.default_x, ds.default_y) == ("Extension", "Load")
    assert ds.metadata["area_mm2"] == pytest.approx(math.pi / 4)
    assert ds.metadata["gauge_length_mm"] == pytest.approx(50.0)
    assert ds.metadata["sample_name"] == "NiTi_wire_01"


def test_zwick_tab_decimal_comma_german():
    raw, specs, kind, ds = _pipeline("zwick_testxpert.txt")
    assert raw.delimiter == "\t" and raw.decimal == ","
    assert raw.headers == ["Standardkraft", "Dehnung", "Standardweg", "Prüfzeit"]
    assert [(s.dimension, s.unit) for s in specs] == [
        ("force", "N"), ("strain", "%"), ("displacement", "mm"), ("time", "s")]
    assert ds.values("Standardkraft")[1] == pytest.approx(1012.5)
    assert ds.values("Dehnung")[2] == pytest.approx(0.5)
    assert kind == "tensile"
    assert raw.meta["thickness_mm"] == pytest.approx(0.5)
    assert raw.meta["width_mm"] == pytest.approx(2.0)
    assert raw.meta["gauge_length_mm"] == pytest.approx(25.0)
    assert ds.metadata["area_mm2"] == pytest.approx(1.0)


def test_shimadzu_trapezium_compression():
    raw, specs, kind, ds = _pipeline("shimadzu_trapezium.csv")
    assert raw.units == ["sec", "N", "mm"]
    assert [(s.dimension, s.unit) for s in specs] == [("time", "s"), ("force", "N"), ("displacement", "mm")]
    assert kind == "compression"
    assert ds.values("Force")[-1] == pytest.approx(-1450.0)
    assert ds.metadata["area_mm2"] == pytest.approx(math.pi * 9 / 4)
    assert ds.metadata["gauge_length_mm"] == pytest.approx(6.0)
    assert (ds.default_x, ds.default_y) == ("Stroke", "Force")


def test_generic_whitespace_cyclic():
    raw, specs, kind, ds = _pipeline("generic_whitespace.txt")
    assert raw.delimiter == " "
    assert [(s.name, s.dimension, s.unit) for s in specs] == [("Strain", "strain", "%"),
                                                              ("Stress", "stress", "MPa")]
    assert kind == "cyclic"
    assert (ds.default_x, ds.default_y) == ("Strain", "Stress")
    assert len(ds.data) == 80


# --------------------------------------------------------------------------- edge cases (tmp files)
def test_decimal_comma_tab_generic(tmp_path):
    p = tmp_path / "generic_xy.txt"
    p.write_text("x\ty\n" + "\n".join(f"{i},5\t{i * 2},25" for i in range(10)), encoding="utf-8")
    raw = D.read_table(p)
    assert raw.decimal == "," and raw.delimiter == "\t"
    assert raw.df["x"].tolist()[:2] == [0.5, 1.5]
    assert raw.df["y"].iloc[3] == pytest.approx(6.25)


def test_semicolon_decimal_comma_with_thousands(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("Time [s];Force [kN]\n0,0;1.234,5\n1,0;2.000,25\n2,0;3,5\n3,0;4,0\n", encoding="utf-8")
    ds = D.load_file(p)
    assert ds.values("Force").tolist() == pytest.approx([1234500.0, 2000250.0, 3500.0, 4000.0])


def test_whitespace_header_with_unit_tokens_and_temperature_K(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("Time (s)   Temperature (K)   Heat Flow (mW)\n" +
                 "\n".join(f"{i}  {300 + i}  {0.1 * i}" for i in range(6)), encoding="utf-8")
    raw = D.read_table(p)
    assert raw.headers == ["Time (s)", "Temperature (K)", "Heat Flow (mW)"]
    ds = D.load_file(p)
    assert ds.kind == "dsc"
    assert ds.values("Temperature")[0] == pytest.approx(300 - 273.15)


def test_no_header_and_utf16_without_bom(tmp_path):
    p = tmp_path / "nohdr.txt"
    p.write_bytes("1.0,2.0\n2.0,4.0\n3.0,6.0\n".encode("utf-16-le"))
    raw = D.read_table(p)
    assert raw.header_row is None
    assert raw.headers == ["Column 1", "Column 2"]
    assert raw.df["Column 2"].tolist() == [2.0, 4.0, 6.0]


def test_trailing_delimiters_and_repeated_header(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Strain (%),Stress (MPa),\n0,0,\n1,100,\nStrain (%),Stress (MPa),\n2,200,\n"
                 "End of test\n", encoding="utf-8")
    raw = D.read_table(p)
    assert raw.headers == ["Strain (%)", "Stress (MPa)"]
    assert raw.df["Stress (MPa)"].tolist() == [0.0, 100.0, 200.0]


def test_unknown_unit_becomes_dimensionless(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("Temp./°C;DSC/(furlongs)\n1;2\n2;3\n3;4\n", encoding="utf-8")
    specs = D.guess_mapping(D.read_table(p))
    assert specs[1].dimension == "dimensionless" and "furlongs" in specs[1].name
    assert specs[1].guessed and "not recognised" in specs[1].reason


def test_raw_dsc_signal_in_microvolt_is_voltage(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("Temp./°C;DSC/(uV)\n1;2\n2;3\n3;4\n", encoding="utf-8")
    specs = D.guess_mapping(D.read_table(p))
    assert (specs[1].dimension, specs[1].unit) == ("voltage", "µV") and not specs[1].guessed


def test_build_dataset_respects_specs_and_metadata_override():
    raw = D.read_table(FIX / "instron_bluehill.csv")
    specs = D.guess_mapping(raw)
    _spec(specs, "Time").include = False
    _spec(specs, "Load").name = "Force"
    ds = D.build_dataset(raw, specs, name="w1", kind="tensile", metadata={"gauge_length_mm": 40.0})
    assert ds.columns() == ["Extension", "Force"]
    assert ds.name == "w1"
    assert ds.metadata["gauge_length_mm"] == 40.0
    assert ds.channels["Force"].dimension == "force"
    with pytest.raises(ValueError):
        D.build_dataset(raw, [D.ColumnSpec("Time", "Time", "time", "s", include=False)], kind="generic")


# --------------------------------------------------------------------------- presets
def test_presets_save_load_list(tmp_path, monkeypatch):
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path / "home"))
    assert D.presets_dir() == tmp_path / "home" / "presets"
    assert D.list_presets() == []
    raw = D.read_table(FIX / "instron_bluehill.csv")
    specs = D.guess_mapping(raw)
    _spec(specs, "Load").name = "Force"
    path = D.save_preset("Instron 5967", specs, "tensile", {"gauge_length_mm": 50.0})
    assert path.parent == tmp_path / "home" / "presets" and path.exists()
    assert D.list_presets() == ["Instron 5967"]
    p = D.load_preset("Instron 5967")
    assert p.kind == "tensile" and p.metadata["gauge_length_mm"] == 50.0
    assert [s.name for s in p.specs] == ["Time", "Extension", "Force"]
    # explicit directory argument and file path
    other = tmp_path / "elsewhere"
    D.save_preset("x", specs, "generic", directory=other)
    assert D.list_presets(other) == ["x"]
    fp = D.save_preset(tmp_path / "my.json", specs, "cyclic")
    assert D.load_preset(fp).kind == "cyclic"
    assert D.delete_preset("x", directory=other) and D.list_presets(other) == []
    # load_file with a preset name
    ds = D.load_file(FIX / "instron_bluehill.csv", preset="Instron 5967")
    assert "Force" in ds.columns() and ds.kind == "tensile"
    assert ds.metadata["gauge_length_mm"] == 50.0


def test_apply_preset_by_position():
    raw = D.read_table(FIX / "shimadzu_trapezium.csv")
    preset = D.Preset("p", "compression", [D.ColumnSpec("a", "t", "time", "s"),
                                           D.ColumnSpec("b", "F", "force", "kN"),
                                           D.ColumnSpec("c", "d", "displacement", "mm", include=False)])
    specs = D.apply_preset(raw, preset)
    assert [s.source for s in specs] == raw.headers
    assert [s.name for s in specs] == ["t", "F", "d"] and specs[1].unit == "kN"


# --------------------------------------------------------------------------- load_file / export
@pytest.mark.parametrize("name,kind", [
    ("netzsch_proteus.txt", "dsc"), ("ta_universal.txt", "dsc"), ("ta_trios.txt", "dsc"),
    ("mettler_stare.txt", "dsc"), ("instron_bluehill.csv", "tensile"), ("zwick_testxpert.txt", "tensile"),
    ("shimadzu_trapezium.csv", "compression"), ("generic_whitespace.txt", "cyclic")])
def test_load_file_end_to_end(name, kind):
    ds = D.load_file(FIX / name)
    assert ds.kind == kind
    assert ds.source_path.endswith(name)
    assert ds.name == Path(name).stem
    assert ds.default_x in ds.columns() and ds.default_y in ds.columns()
    assert ds.default_x != ds.default_y
    for c, ch in ds.channels.items():
        assert ch.unit == D.U.CANONICAL[ch.dimension]
        assert np.isfinite(ds.values(c)).any()


def test_export_roundtrip(tmp_path):
    ds = D.load_file(FIX / "instron_bluehill.csv")
    out = D.export_dataset_csv(ds, tmp_path / "out.csv")
    text = out.read_text(encoding="utf-8")
    assert "Load (N)" in text and "Extension (mm)" in text
    ds2 = D.load_file(out)
    assert ds2.kind == "tensile"
    assert ds2.values("Load") == pytest.approx(ds.values("Load"))
    assert ds2.metadata["area_mm2"] == pytest.approx(ds.metadata["area_mm2"])


# --------------------------------------------------------------------------- dialog (headless)
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_import_dialog_mechanical(qapp, tmp_path):
    from labstudio.gui.import_dialog import SECTION_RECT, SECTION_ROUND, ImportDialog
    dlg = ImportDialog(FIX / "instron_bluehill.csv", presets_dir=tmp_path)
    assert dlg.kind() == "tensile"
    assert dlg.preview_table.rowCount() == 41
    # prefilled from preamble: Ø 1 mm, gauge 50 mm
    assert dlg.section_mode.currentIndex() == SECTION_ROUND
    assert dlg.metadata()["area_mm2"] == pytest.approx(0.785, abs=5e-4)
    # change the Load unit from kN to N (user says the file is actually in N)
    i = dlg.raw.headers.index("Load")
    dlg.unit_combos[i].setCurrentText("N")
    # set geometry explicitly: round Ø 1000 µm == 1 mm, gauge 25 mm
    dlg.diam_unit.setCurrentText("µm")
    dlg.diam_spin.setValue(1000.0)
    dlg.gauge_spin.setValue(25.0)
    assert "0.7854" in dlg.area_label.text()
    dlg.accept()                                   # programmatic accept: builds, no exec()
    ds = dlg.dataset()
    assert ds is not None and ds.kind == "tensile"
    assert ds.metadata["area_mm2"] == pytest.approx(0.785, abs=5e-4)
    assert ds.metadata["diameter_mm"] == pytest.approx(1.0)
    assert ds.metadata["gauge_length_mm"] == pytest.approx(25.0)
    assert ds.values("Load")[4] == pytest.approx(0.1)          # now treated as N, not kN
    # rectangular section
    dlg.section_mode.setCurrentIndex(SECTION_RECT)
    dlg.width_spin.setValue(2.0)
    dlg.thick_unit.setCurrentText("µm")
    dlg.thick_spin.setValue(500.0)
    assert dlg.build().metadata["area_mm2"] == pytest.approx(1.0)
    # dimension change repopulates unit combo
    dlg.dim_combos[i].setCurrentIndex(D.U.dimensions().index("stress"))
    assert dlg.unit_combos[i].currentText() == "MPa"
    # presets from the dialog
    dlg.save_preset_as("bluehill")
    assert D.list_presets(tmp_path) == ["bluehill"]
    dlg.dim_combos[i].setCurrentIndex(D.U.dimensions().index("force"))
    dlg.unit_combos[i].setCurrentText("kN")
    dlg.apply_preset("bluehill")
    assert dlg.dim_combos[i].currentData() == "stress"
    dlg.deleteLater()


def test_import_dialog_dsc_prefill(qapp, tmp_path):
    from labstudio.gui.import_dialog import ImportDialog
    dlg = ImportDialog(FIX / "netzsch_proteus.txt", presets_dir=tmp_path)
    assert dlg.kind() == "dsc"
    assert dlg.meta_stack.currentIndex() == 2
    assert dlg.mass_spin.value() == pytest.approx(15.23)
    assert dlg.rate_spin.value() == pytest.approx(10.0)
    dlg.mass_unit.setCurrentText("g")
    dlg.mass_spin.setValue(0.0152)
    ds = dlg.build()
    assert ds.metadata["mass_mg"] == pytest.approx(15.2)
    assert ds.metadata["heating_rate_K_per_min"] == pytest.approx(10.0)
    assert ds.channels["Time"].unit == "s"
    assert dlg.dataset() is None                   # not accepted yet
    dlg.set_kind("generic")
    assert dlg.meta_stack.currentIndex() == 0
    dlg.reject()
    assert dlg.dataset() is None
    dlg.deleteLater()


def test_import_files_apply_all(qapp, tmp_path, monkeypatch):
    """import_files: first dialog accepted with 'apply to all', rest imported silently."""
    from labstudio.gui import import_dialog as M

    calls = []

    def fake_exec(self):
        calls.append(self.path)
        self.apply_all_check.setChecked(True)
        self.accept()
        return M.QDialog.Accepted

    monkeypatch.setattr(M.ImportDialog, "exec", fake_exec)
    paths = [str(FIX / "instron_bluehill.csv"), str(FIX / "instron_bluehill.csv")]
    out = M.ImportDialog.import_files(paths, presets_dir=tmp_path)
    assert len(out) == 2 and len(calls) == 1
    assert all(ds.kind == "tensile" for ds in out)
    assert out[1].values("Load")[4] == pytest.approx(100.0)


# =========================================================================== v1.1
import hashlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


def _dsc_body(n=60, rate=10.0, minutes=False):
    T = np.linspace(-50, 100, n)
    t = (T - T[0]) / rate * (1 if minutes else 60)
    hf = 0.1 + 0.5 * np.exp(-(T - 20) ** 2 / 32)
    return T, t, hf


# --------------------------------------------------------------------------- 1. heating rate / time units
@pytest.mark.parametrize("pre,expected", [
    ("#HEATING RATE /(K/min):;20.0\n", 20.0),                       # Netzsch key with unit in key
    ("#RANGE:-80°C/10.0(K/min)/100°C\n", 10.0),                    # Netzsch range line
    ("Heating rate\t5.0\tK/min\n", 5.0),                           # TA key/value/unit
    ("Heating rate: 15 °C/min\n", 15.0),
    ("2: Ramp 10.00 °C/min to 100.00 °C\n", 10.0),                 # TA procedure line
    ("dT/dt: 2 K/min\n", 2.0),                                     # Mettler
    ("Rate: 0.5 K/s\n", 30.0),                                     # bare 'Rate' with a heating-rate unit
    ("Method: -80..100°C 10K/min N2\n", 10.0),                     # Mettler method string
    ("Scan rate: 40\n", 40.0),                                     # explicit key, no unit -> K/min + note
])
def test_heating_rate_keys(tmp_path, pre, expected):
    T, _, hf = _dsc_body()
    body = "Temperature (°C);Heat Flow (W/g)\n" + "".join(f"{a:.3f};{b:.5f}\n" for a, b in zip(T, hf))
    raw = D.read_table(_write(tmp_path, "d.txt", pre + "Sample mass: 10 mg\n" + body))
    assert raw.meta["heating_rate_K_per_min"] == pytest.approx(expected)
    ds = D.build_dataset(raw, D.guess_mapping(raw), kind="dsc")
    assert ds.metadata["heating_rate_K_per_min"] == pytest.approx(expected)
    assert ds.metadata["heating_rate_source"] == "header"
    if pre.startswith("Scan rate"):
        assert any("assumed K/min" in w for w in ds.metadata["warnings"])
    else:
        assert ds.metadata["warnings"] == []


def test_instron_crosshead_rate_is_not_a_heating_rate(tmp_path):
    p = _write(tmp_path, "c.csv", "Rate 1: 15.0 mm/min\nTime,Extension,Load\n(s),(mm),(kN)\n"
               + "".join(f"{i},{i * 0.1},{i * 0.01}\n" for i in range(20)))
    assert "heating_rate_K_per_min" not in D.read_table(p).meta


def test_netzsch_heating_rate_key_without_time_column():
    """Customer C1: '#HEATING RATE /(K/min):;20.0' was ignored -> ΔH 2x too large."""
    ds = D.load_file(FIX / "netzsch_rate_key_notime.txt")
    assert ds.kind == "dsc"
    assert ds.metadata["heating_rate_K_per_min"] == pytest.approx(20.0)
    assert ds.metadata["heating_rate_source"] == "header"
    assert ds.metadata["mass_mg"] == pytest.approx(15.0)
    assert ds.metadata["warnings"] == []


def test_dsc_without_rate_defaults_with_warning_and_minutes_time():
    """QA H3: 'Time (minutes)' must be min (not s); a missing rate is defaulted *and* flagged."""
    ds = D.load_file(FIX / "dsc_minutes_norate.csv")
    assert ds.channels["Time"].original_unit == "min"
    assert ds.values("Time")[1] == pytest.approx(2.0 / 10 * 60)          # 2 K step at 10 K/min = 12 s
    assert ds.metadata["heating_rate_K_per_min"] == 10.0
    assert ds.metadata["heating_rate_source"] == "default"
    assert ds.metadata["measured_heating_rate_K_per_min"] == pytest.approx(10.0, rel=1e-3)
    w = ds.metadata["warnings"]
    assert "heating rate not found in file — defaulted to 10 K/min" in w
    assert any("Time" in x and "minutes" in x for x in w)               # spelled-out unit not confirmed


def test_user_rate_and_header_measured_mismatch(tmp_path):
    T, t, hf = _dsc_body(rate=10.0, minutes=True)
    text = ("# Heating rate: 10 K/min\nTime (s),Temperature (°C),Heat Flow (W/g)\n"     # wrong: values are min
            + "".join(f"{a:.5f},{b:.3f},{c:.5f}\n" for a, b, c in zip(t, T, hf)))
    ds = D.load_file(_write(tmp_path, "m.csv", text))
    assert ds.metadata["heating_rate_source"] == "header"
    assert ds.metadata["measured_heating_rate_K_per_min"] == pytest.approx(600, rel=1e-3)
    assert any("differs from the scan rate measured" in w for w in ds.metadata["warnings"])
    # explicit user rate
    ds2 = D.load_file(_write(tmp_path, "m2.csv", text.replace("# Heating rate: 10 K/min\n", "")),
                      metadata={"heating_rate_K_per_min": 600.0})
    assert ds2.metadata["heating_rate_source"] == "user"
    assert not any("differs" in w for w in ds2.metadata["warnings"])


@pytest.mark.parametrize("header,dim,unit,guessed", [
    ("Time (minutes)", "time", "min", True), ("Time (mins)", "time", "min", False),
    ("Time (seconds)", "time", "s", True), ("Time (sec)", "time", "s", False),
    ("Time (hours)", "time", "h", True), ("Temp (degC)", "temperature", "°C", False),
    ("Temp (deg C)", "temperature", "°C", False), ("Temp (Celsius)", "temperature", "°C", True),
    ("Temp (Kelvin)", "temperature", "K", True), ("Load (newtons)", "force", "N", True),
    ("Load (kilonewtons)", "force", "kN", True), ("Ext (millimetres)", "displacement", "mm", True),
    ("Ext (millimeters)", "displacement", "mm", True), ("Stress (megapascals)", "stress", "MPa", True),
    ("Energy (MJ)", "energy", "MJ", False), ("Force (MN)", "force", "MN", False),
    ("Heat flow (MW)", "heat_flow", "MW", False), ("Heat flow (mW)", "heat_flow", "mW", False),
    ("Stress (mpa)", "stress", "MPa", True), ("Time (fortnights)", "dimensionless", "", True),
    ("Strain", "strain", "%", True), ("Time", "time", "s", True), ("Cycle", "count", "#", False),
])
def test_spelled_out_and_guessed_units(header, dim, unit, guessed):
    g = D._analyse_column_ex(header, None)
    assert (g.dimension, g.unit, g.guessed) == (dim, unit, guessed)
    if guessed:
        assert g.reason


def test_energy_MJ_column_is_mega(tmp_path):
    """QA M1 via import: 'Energy (MJ)' 1.5 -> 1.5e6 J (was 0.0015 J)."""
    p = _write(tmp_path, "e.csv", "Cycle,Energy (MJ),Work (MJ/m3)\n" + "".join(f"{i},1.5,2\n" for i in range(1, 20)))
    ds = D.load_file(p)
    assert ds.values("Energy")[0] == pytest.approx(1.5e6)
    assert ds.channels["Work"].dimension == "energy_density" and ds.values("Work")[0] == pytest.approx(2)


def test_unit_confidence_and_build_warnings_for_unconfirmed(tmp_path):
    p = _write(tmp_path, "g.csv", "Time (minutes),Load (kN),Ext\n" + "".join(f"{i},{i},{i}\n" for i in range(10)))
    raw = D.read_table(p)
    specs = D.guess_mapping(raw)
    assert raw.unit_confidence == {"Time (minutes)": "guessed", "Load (kN)": "confirmed", "Ext": "guessed"}
    ds = D.build_dataset(raw, specs, kind="tensile")
    assert len(ds.metadata["warnings"]) == 2
    # confirming (guessed=False) removes the warnings; excluded columns never warn
    for s in specs:
        s.guessed = False
    _spec(specs, "Ext").include = False
    assert D.build_dataset(raw, specs, kind="tensile").metadata["warnings"] == []


# --------------------------------------------------------------------------- 3. header guesses / header row
def test_header_guesses_point_weight_ext(tmp_path):
    p = _write(tmp_path, "h.csv", "Point,Index,No.,Weight,Mass,Ext,Ext.,Extensometer,Strain gauge,Crosshead,"
               "Position,Standard force,Kraft,Dehnung,Weg,Zeit,Temp.,Ts,Tr\n"
               + "".join(",".join([str(i)] * 19) + "\n" for i in range(10)))
    dims = [s.dimension for s in D.guess_mapping(D.read_table(p))]
    assert dims == ["count", "count", "count", "mass", "mass", "displacement", "displacement", "displacement",
                    "strain", "displacement", "displacement", "force", "force", "strain", "displacement",
                    "time", "temperature", "temperature", "temperature"]


def test_ext_with_strain_unit_is_strain(tmp_path):
    p = _write(tmp_path, "x.csv", "Ext,Load\n(%),(N)\n" + "".join(f"{i},{i}\n" for i in range(10)))
    s = D.guess_mapping(D.read_table(p))
    assert (s[0].dimension, s[0].unit, s[0].guessed) == ("strain", "%", False)


def test_first_data_row_with_text_keeps_real_header(tmp_path):
    """QA M10: 'Time (s),Load (N),Note' + first row '0,ERR,x' -> header must not be lost."""
    p = _write(tmp_path, "mixed.csv", "Time (s),Load (N),Note\n"
               + "".join(f"{i},{i * 2 if i % 7 else 'ERR'},{'ok' if i % 3 else 'x'}\n" for i in range(100)))
    raw = D.read_table(p)
    assert raw.headers == ["Time (s)", "Load (N)", "Note"]
    assert len(raw.df) == 100                         # rows with 'ERR' kept (Load = NaN)
    assert np.isnan(raw.df["Load (N)"][0]) and raw.df["Load (N)"][1] == 2.0
    specs = D.guess_mapping(raw)
    assert [(s.dimension, s.unit) for s in specs[:2]] == [("time", "s"), ("force", "N")]


def test_sample_label_column_with_units_row(tmp_path):
    p = _write(tmp_path, "lab.csv", "Specimen: W1\n\nSample,Time,Extension,Load\n,(s),(mm),(kN)\n"
               + "W1-a,0,0,0\n" + "".join(f",{i},{i * 0.1},{i * 0.01}\n" for i in range(1, 30)))
    raw = D.read_table(p)
    assert raw.headers == ["Sample", "Time", "Extension", "Load"] and raw.units_row is not None
    assert len(raw.df) == 30


def test_nan_rows_kept_and_footer_dropped(tmp_path):
    """QA L5 (NaN rows lost) + Zwick 'Results' footer rows not taken as data."""
    p = _write(tmp_path, "n.csv", "Time (s),Load (N)\n" + "".join(f"{i},{'NaN' if i % 5 == 0 else i * 2}\n"
                                                                   for i in range(100)))
    assert len(D.read_table(p).df) == 100
    p2 = _write(tmp_path, "z.csv", "Travel [mm];Force [N];Time [s]\n"
                + "".join(f"{i},5;{i * 3};{i}\n" for i in range(50))
                + "Results\nE-Modulus;45,1;GPa\nRm;1300;MPa\nEnd of report\n")
    raw = D.read_table(p2)
    assert len(raw.df) == 50 and raw.decimal == ","


def test_binary_and_unsupported_files_raise_clear_errors(tmp_path):
    rng = np.random.default_rng(0)
    b = tmp_path / "binary.csv"
    b.write_bytes(bytes(rng.integers(0, 256, 5000, dtype=np.uint8)))
    with pytest.raises(D.UnsupportedFileError, match="text data file"):
        D.read_table(b)
    for ext in (".ngb", ".png", ".xlsb", ".is_tens"):
        f = tmp_path / f"x{ext}"
        f.write_bytes(b"\x00\x01")
        with pytest.raises(D.UnsupportedFileError, match="cannot be imported"):
            D.read_table(f)
    with pytest.raises(FileNotFoundError):
        D.read_table(tmp_path / "missing.csv")
    assert ".xlsx" in D.DATA_EXTENSIONS and ".csv" in D.DATA_EXTENSIONS


# --------------------------------------------------------------------------- 4. spreadsheets
def test_xlsx_multisheet_picks_data_sheet():
    path = FIX / "instron_multisheet.xlsx"
    assert D.list_sheets(path) == ["Results", "Raw Data", "Second test"]
    raw = D.read_table(path)
    assert raw.sheet == "Raw Data" and raw.sheets == ["Results", "Raw Data", "Second test"]
    assert raw.headers == ["Time", "Extension", "Load", "Label"]
    assert raw.units[:3] == ["(s)", "(mm)", "(kN)"] and len(raw.df) == 40
    assert raw.meta["diameter_mm"] == pytest.approx(1.0) and raw.meta["gauge_length_mm"] == 50
    ds = D.load_file(path)
    assert ds.kind == "tensile" and ds.values("Load")[2] == pytest.approx(10.0)
    assert ds.metadata["import"]["sheet"] == "Raw Data"
    raw3 = D.read_table(path, sheet="Second test")
    assert raw3.headers == ["Time (s)", "Load (N)"] and len(raw3.df) == 10
    assert D.read_table(path, sheet=2).sheet == "Second test"
    with pytest.raises(ValueError, match="No numeric data found in sheet 'Results'"):
        D.read_table(path, sheet="Results")
    with pytest.raises(ValueError, match="not found"):
        D.read_table(path, sheet="nope")


def test_ods_and_bad_workbook_and_missing_engine(tmp_path, monkeypatch):
    pytest.importorskip("odf")
    import pandas as pd
    p = tmp_path / "d.ods"
    pd.DataFrame({"Temperature (°C)": [1.0, 2, 3, 4, 5], "Heat flow (mW)": [0.1, 0.2, 0.3, 0.2, 0.1]}) \
        .to_excel(p, engine="odf", index=False)
    ds = D.load_file(p)
    assert ds.kind == "dsc" and ds.channels["Heat flow"].unit == "mW"
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not a zip")
    with pytest.raises(D.UnsupportedFileError, match="workbook"):
        D.read_table(bad)
    monkeypatch.setitem(D._SHEET_ENGINES, ".xlsx", ("labstudio_no_such_module", "openpyxl"))
    with pytest.raises(D.UnsupportedFileError, match="pip install openpyxl"):
        D.read_table(FIX / "instron_multisheet.xlsx")


# --------------------------------------------------------------------------- 5. analysis defaults in presets
def test_preset_analysis_defaults_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path / "home"))
    raw = D.read_table(FIX / "netzsch_proteus.txt")
    ad = {"dsc": {"prominence": 0.02, "baseline": "linear"}, "cyclic": {"n_cycles": np.int64(3)}}
    D.save_preset("Netzsch", D.guess_mapping(raw), "dsc", analysis_defaults=ad)
    p = D.load_preset("Netzsch")
    assert p.analysis_defaults == {"dsc": {"prominence": 0.02, "baseline": "linear"}, "cyclic": {"n_cycles": 3}}
    assert all(not s.guessed for s in p.specs)
    ds = D.load_file(FIX / "netzsch_proteus.txt", preset="Netzsch")
    assert ds.metadata["analysis_defaults"]["dsc"]["prominence"] == 0.02
    D.save_preset("plain", D.guess_mapping(raw), "dsc")
    assert D.load_preset("plain").analysis_defaults == {}
    assert "analysis_defaults" not in D.load_file(FIX / "netzsch_proteus.txt", preset="plain").metadata


def test_preset_geometry_does_not_override_file(tmp_path, monkeypatch):
    """QA L4: specimen geometry from the file wins over geometry stored in a preset."""
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path / "home"))
    raw = D.read_table(FIX / "instron_bluehill.csv")
    D.save_preset("wire", D.guess_mapping(raw), "tensile", {"area_mm2": 0.7854, "gauge_length_mm": 50})
    p = _write(tmp_path, "s2.csv", "Diameter: 2 mm\nGauge length: 25 mm\nTime,Extension,Load\n(s),(mm),(kN)\n"
               + "".join(f"{i},{i * 0.1},{i * 0.01}\n" for i in range(20)))
    ds = D.load_file(p, preset="wire")
    assert ds.metadata["area_mm2"] == pytest.approx(math.pi)
    assert ds.metadata["gauge_length_mm"] == pytest.approx(25)


# --------------------------------------------------------------------------- 6. provenance
def test_provenance_sha256_and_import_record():
    path = FIX / "instron_bluehill.csv"
    ds = D.load_file(path)
    assert ds.metadata["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    imp = ds.metadata["import"]
    assert imp["delimiter"] == "," and imp["decimal"] == "." and imp["header_row"] is not None
    assert imp["encoding"] and imp["sheet"] is None
    assert {"source", "name", "dimension", "unit", "guessed"} <= set(imp["specs"][0])
    assert [s["source"] for s in imp["specs"]] == ["Time", "Extension", "Load"]
    import json
    json.dumps(ds.metadata["import"])          # JSON-serialisable for sessions/provenance


def test_export_writes_warnings(tmp_path):
    ds = D.load_file(FIX / "dsc_minutes_norate.csv")
    text = D.export_dataset_csv(ds, tmp_path / "o.csv").read_text(encoding="utf-8")
    assert "# warnings: " in text and "defaulted to 10 K/min" in text


# --------------------------------------------------------------------------- 7. install files
def test_requirements_pinned_and_launchers_use_venv():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for line in ("numpy>=1.26,<3", "scipy>=1.11", "pandas>=2.1", "matplotlib>=3.8", "PySide6>=6.6",
                 "scikit-image>=0.22", "tifffile>=2023.7", "Pillow>=10", "openpyxl>=3.1"):
        assert line in req
    pkgs = [l.split(">")[0].split("<")[0].split("=")[0].strip() for l in req.splitlines()
            if l.strip() and not l.startswith("#")]
    assert all(">=" in l for l in req.splitlines() if l.strip() and not l.startswith("#")), pkgs
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "pytest" in dev and "pytest-cov" in dev and "-r requirements.txt" in dev
    mac_p = ROOT / "Start Materia (Mac).command"
    mac = mac_p.read_text(encoding="utf-8")
    assert "launcher.pyw" in mac and "python.org" in mac
    assert subprocess.run(["bash", "-n", str(mac_p)]).returncode == 0
    assert os.access(mac_p, os.X_OK)
    win = (ROOT / "Start Materia (Windows).bat").read_bytes()
    assert b"\r\n" in win
    w = win.decode("ascii")
    assert "py -3" in w and "python -c" in w and "launcher.pyw" in w and "pyw" in w and "python.org" in w
    assert "for /f" not in w  # quoting trap that broke v1.1 on Windows
    lp = (ROOT / "launcher.pyw").read_text(encoding="utf-8")
    assert ".venv" in lp and "requirements.sha256" in lp and "Scripts" in lp and "pythonw.exe" in lp
    compile(lp, "launcher.pyw", "exec")


# --------------------------------------------------------------------------- dialog (v1.1)
def test_dialog_guessed_columns_need_confirmation(qapp, tmp_path):
    from labstudio.gui.import_dialog import AMBER, ImportDialog
    p = _write(tmp_path, "g.csv", "Time (minutes),Load (kN),Ext\n" + "".join(f"{i},{i},{i}\n" for i in range(10)))
    dlg = ImportDialog(p, presets_dir=tmp_path)
    assert [c.isChecked() for c in dlg.confirm_checks] == [False, True, False]
    item = dlg.map_table.item(0, dlg.COL_SOURCE)
    assert item.background().color().name() == AMBER
    assert "minutes" in item.toolTip()
    assert dlg.unconfirmed() == ["Time (minutes)", "Ext"]
    assert "2 column unit(s) guessed" in dlg.status_label.text()
    assert dlg.unit_combos[0].currentText() == "min"
    ds = dlg.build()                                   # import still works, but warns
    assert len(ds.metadata["warnings"]) == 2
    assert ds.metadata["import"]["specs"][0]["guessed"] is True
    dlg.confirm_checks[0].setChecked(True)
    assert item.background().style() == Qt_NoBrush()
    dlg.unit_combos[2].setCurrentText("µm")            # editing a unit confirms that row
    assert dlg.confirm_checks[2].isChecked()
    assert dlg.unconfirmed() == [] and dlg.status_label.text() == ""
    assert dlg.build().metadata["warnings"] == []
    dlg.deleteLater()


def Qt_NoBrush():
    from PySide6.QtCore import Qt
    return Qt.NoBrush


def test_dialog_dsc_missing_rate_highlighted(qapp, tmp_path):
    from labstudio.gui.import_dialog import AMBER, ImportDialog
    dlg = ImportDialog(FIX / "dsc_minutes_norate.csv", presets_dir=tmp_path)
    assert dlg.kind() == "dsc" and not dlg.rate_from_file
    assert dlg.rate_needs_confirmation
    assert AMBER in dlg.rate_spin.styleSheet()
    assert not dlg.rate_confirm_check.isHidden()
    assert "not found" in dlg.dsc_note.text()
    assert "heating_rate_K_per_min" not in dlg.metadata()
    assert any("defaulted to 10 K/min" in w for w in dlg.pending_warnings())
    dlg.confirm_all()
    ds = dlg.build()
    assert ds.metadata["heating_rate_source"] == "default"
    assert ds.metadata["warnings"] == ["heating rate not found in file — defaulted to 10 K/min"]
    dlg.rate_spin.setValue(5.0)                        # typing a rate confirms it
    assert dlg.rate_confirm_check.isChecked() and not dlg.rate_needs_confirmation
    assert dlg.rate_spin.styleSheet() == ""
    ds = dlg.build()
    assert ds.metadata["heating_rate_K_per_min"] == 5.0 and ds.metadata["heating_rate_source"] == "user"
    assert ds.metadata["warnings"] != [] and any("differs" in w for w in ds.metadata["warnings"])
    dlg.deleteLater()
    # header rate: no highlight, source 'header'
    dlg = ImportDialog(FIX / "netzsch_rate_key_notime.txt", presets_dir=tmp_path)
    assert dlg.rate_spin.value() == pytest.approx(20.0) and not dlg.rate_needs_confirmation
    assert dlg.rate_confirm_check.isHidden()
    assert "HEATING RATE" in dlg.dsc_note.text()
    ds = dlg.build()
    assert ds.metadata["heating_rate_source"] == "header" and ds.metadata["warnings"] == []
    dlg.rate_spin.setValue(10.0)
    assert dlg.build().metadata["heating_rate_source"] == "user"
    dlg.deleteLater()


def test_dialog_sheet_picker(qapp, tmp_path):
    from labstudio.gui.import_dialog import ImportDialog
    path = FIX / "instron_multisheet.xlsx"
    dlg = ImportDialog(path, presets_dir=tmp_path)
    assert dlg.sheet == "Raw Data" and not dlg.sheet_combo.isHidden()
    assert [dlg.sheet_combo.itemText(i) for i in range(dlg.sheet_combo.count())] == \
        ["Results", "Raw Data", "Second test"]
    assert dlg.kind() == "tensile" and dlg.diam_spin.value() == pytest.approx(1.0)
    dlg.sheet_combo.setCurrentText("Second test")
    assert dlg.sheet == "Second test" and dlg.raw.headers == ["Time (s)", "Load (N)"]
    assert dlg.preview_table.rowCount() == 10 and len(dlg.dim_combos) == 2
    dlg.sheet_combo.setCurrentText("Results")          # no data: stays on previous sheet, shows why
    assert dlg.sheet == "Second test" and "Results" in dlg.status_label.text()
    dlg.deleteLater()
    dlg = ImportDialog(path, presets_dir=tmp_path, sheet="Second test")
    assert dlg.sheet == "Second test"
    assert dlg.build().metadata["import"]["sheet"] == "Second test"
    dlg.deleteLater()
    dlg = ImportDialog(FIX / "instron_bluehill.csv", presets_dir=tmp_path)
    assert dlg.sheet_combo.isHidden()
    dlg.deleteLater()


def test_dialog_analysis_defaults_with_preset(qapp, tmp_path):
    from labstudio.gui.import_dialog import ImportDialog
    dlg = ImportDialog(FIX / "netzsch_proteus.txt", presets_dir=tmp_path)
    assert not dlg.save_defaults_check.isEnabled()
    provided = {"dsc": {"prominence": 0.02, "baseline": "linear"}}
    dlg.set_analysis_defaults_provider(lambda: provided)
    assert dlg.save_defaults_check.isEnabled()
    dlg.save_defaults_check.setChecked(True)
    dlg.save_preset_as("netzsch+defaults")
    dlg.save_defaults_check.setChecked(False)
    dlg.save_preset_as("netzsch-plain")
    assert D.load_preset("netzsch+defaults", directory=tmp_path).analysis_defaults == provided
    assert D.load_preset("netzsch-plain", directory=tmp_path).analysis_defaults == {}
    dlg.deleteLater()
    dlg = ImportDialog(FIX / "netzsch_proteus.txt", presets_dir=tmp_path)
    assert "analysis_defaults" not in dlg.metadata()
    dlg.apply_preset("netzsch+defaults")
    assert dlg.analysis_defaults() == provided and "dsc" in dlg.preset_defaults_label.text()
    assert dlg.build().metadata["analysis_defaults"] == provided
    dlg.deleteLater()
    # application-wide provider
    ImportDialog.set_default_analysis_defaults_provider(lambda: {"cyclic": {"x": 1}})
    try:
        dlg = ImportDialog(FIX / "netzsch_proteus.txt", presets_dir=tmp_path)
        assert dlg.save_defaults_check.isEnabled()
        dlg.save_preset_as("global", include_analysis_defaults=True)
        assert D.load_preset("global", directory=tmp_path).analysis_defaults == {"cyclic": {"x": 1}}
        dlg.deleteLater()
    finally:
        ImportDialog.set_default_analysis_defaults_provider(None)


def test_dialog_fits_small_screen(qapp, tmp_path):
    from labstudio.gui.import_dialog import ImportDialog
    for f in ("instron_bluehill.csv", "netzsch_proteus.txt", "instron_multisheet.xlsx"):
        dlg = ImportDialog(FIX / f, presets_dir=tmp_path)
        dlg.resize(1100, 650)
        hint = dlg.minimumSizeHint()
        assert hint.width() <= 1100 and hint.height() <= 650, (f, hint)
        assert dlg.preview_table.verticalScrollBarPolicy() != 1      # Qt.ScrollBarAlwaysOff == 1
        dlg.deleteLater()
    # a wide, long file still fits (tables scroll instead of growing the dialog)
    p = _write(tmp_path, "wide.csv", ",".join(f"Col {i} (N)" for i in range(30)) + "\n"
               + "".join(",".join(str(j) for j in range(30)) + "\n" for _ in range(500)))
    dlg = ImportDialog(p, presets_dir=tmp_path)
    hint = dlg.minimumSizeHint()
    assert hint.width() <= 1100 and hint.height() <= 650
    assert dlg.preview_table.rowCount() == 200
    dlg.deleteLater()


def test_import_files_apply_all_does_not_leak_header_rate(qapp, tmp_path, monkeypatch):
    from labstudio.gui import import_dialog as M

    def fake_exec(self):
        self.apply_all_check.setChecked(True)
        self.confirm_all()
        self.accept()
        return M.QDialog.Accepted

    monkeypatch.setattr(M.ImportDialog, "exec", fake_exec)
    paths = [str(FIX / "netzsch_rate_key_notime.txt"), str(FIX / "dsc_minutes_norate.csv")]
    out = M.ImportDialog.import_files(paths, presets_dir=tmp_path,
                                      analysis_defaults_provider=lambda: {"dsc": {}})
    assert len(out) == 2
    assert out[0].metadata["heating_rate_K_per_min"] == 20.0
    assert out[1].metadata["heating_rate_source"] == "default"      # not file 1's 20 K/min
    assert out[1].metadata["heating_rate_K_per_min"] == 10.0
