"""Tests for core/units.py: symbol resolution (case-sensitive SI prefixes, aliases, spelled-out
words), conversions incl. temperature offsets, header parsing and header keyword hints."""
from __future__ import annotations

import numpy as np
import pytest

from labstudio.core import units as U

ALL_UNITS = [(dim, sym) for dim in U.dimensions() for sym in U.units_for(dim)]


# --------------------------------------------------------------------------- registry sanity
def test_every_dimension_has_canonical_and_label():
    for dim in U.dimensions():
        assert dim in U.CANONICAL and U.CANONICAL[dim] in U.units_for(dim)
        assert dim in U.DIMENSION_LABELS
    assert U.CANONICAL["voltage"] == "µV"


@pytest.mark.parametrize("dim,sym", ALL_UNITS)
def test_registered_symbol_round_trip(dim, sym):
    m = U.resolve_unit(sym)
    assert m.symbol == sym and m.how == "exact" and not m.guessed
    assert U.normalize_symbol(sym) == sym
    assert dim in U.dimensions_of(sym)
    canon = U.CANONICAL[dim]
    v = np.array([-3.5, 0.0, 1.0, 1234.5])
    there = U.convert(v, sym, canon, dim)
    back = U.convert(there, canon, sym, dim)
    assert np.allclose(back, v, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("alias,target", sorted(U._ALIASES.items()))
def test_every_alias_resolves_to_a_registered_symbol(alias, target):
    assert U.normalize_symbol(alias) == target
    assert U.dimension_of(alias) is not None
    assert not U.resolve_unit(alias).guessed


@pytest.mark.parametrize("word,target", sorted(U._SPELLED.items()))
def test_every_spelled_word_resolves_and_is_flagged(word, target):
    for w in (word, word.title(), word.upper()):
        m = U.resolve_unit(w)
        assert m.symbol == target, w
        assert m.guessed and w in m.reason
    assert U.dimension_of(word) is not None


# --------------------------------------------------------------------------- QA M1: SI prefix case
@pytest.mark.parametrize("text,expected", [
    ("MJ", "MJ"), ("mJ", "mJ"), ("MN", "MN"), ("mN", "mN"), ("MW", "MW"), ("mW", "mW"),
    ("MPa", "MPa"), ("mPa", "mPa"), ("ms", "ms"), ("mm", "mm"), ("m", "m"),
    ("Mpa", "MPa"), ("MPA", "MPa"), ("GPA", "GPa"), ("Kn", "kN"), ("KN", "kN"), ("kn", "kN"),
    ("Min", "min"), ("S", "s"), ("C", "°C"), ("c", "°C"), ("K", "K"), ("k", "K"),
    ("mw", "mW"), ("Mw", "MW"), ("mn", "mN"), ("um", "µm"), ("μm", "µm"), ("N/mm2", "N/mm²"),
    ("MJ/m3", "MJ/m³"), ("MJ m-3", "MJ/m³"), ("J/cm3", "J/cm³"), ("uV", "µV"), ("μV", "µV"),
    ("mV", "mV"), ("mJ/g", "mJ/g"), ("J/mg", "J/mg"), ("[kN]", "kN"), ("(mm)", "mm"),
    ("degC", "°C"), ("deg C", "°C"), ("K/min", "K/min"), ("K min-1", "K/min"), ("°C/s", "°C/s"),
])
def test_case_sensitive_resolution(text, expected):
    assert U.normalize_symbol(text) == expected


@pytest.mark.parametrize("text", ["Ms", "M", "MM", "fortnights", "furlong-lbf", "a.u.", "Mm"])
def test_unsafe_or_unknown_symbols_are_not_resolved(text):
    m = U.resolve_unit(text)
    assert m.symbol is None and m.how == "unknown"
    assert U.dimension_of(text) is None


def test_ambiguous_lowercase_mpa_is_flagged():
    m = U.resolve_unit("mpa")
    assert m.symbol == "MPa" and m.how == "ambiguous" and m.guessed
    assert "mPa" in m.candidates and "ambiguous" in m.reason


@pytest.mark.parametrize("value,frm,to,expected", [
    (1.5, "MJ", "J", 1.5e6), (1.5, "mJ", "J", 1.5e-3), (2, "MN", "N", 2e6), (2, "mN", "N", 2e-3),
    (3, "MW", "mW", 3e9), (3, "mW", "W", 3e-3), (1, "mPa", "MPa", 1e-9), (1, "MPa", "Pa", 1e6),
    (1, "Ms", "s", None), (500, "mJ/g", "J/g", 0.5), (2, "J/mg", "J/g", 2000), (1500, "µV", "mV", 1.5),
    (1, "MJ/m3", "J/m³", 1e6), (1, "J/cm³", "MJ/m³", 1.0), (1, "K/s", "K/min", 60), (1, "°C/s", "K/min", 60),
    (120, "K/h", "K/min", 2), (1, "%/s", "1/s", 0.01), (1, "µg", "mg", 1e-3), (2, "minutes", "s", 120),
    (1, "kilonewtons", "N", 1000), (25.4, "millimetres", "in", 1.0), (1, "megapascals", "MPa", 1),
])
def test_conversions(value, frm, to, expected):
    if expected is None:
        with pytest.raises(ValueError):
            U.convert(value, frm, to)
        return
    assert U.convert(value, frm, to) == pytest.approx(expected)


# --------------------------------------------------------------------------- temperature offsets
@pytest.mark.parametrize("value,frm,to,expected", [
    (25, "°C", "K", 298.15), (0, "K", "°C", -273.15), (-40, "°C", "°F", -40), (212, "°F", "°C", 100),
    (32, "°F", "K", 273.15), (300, "K", "°F", 80.33), (100, "degC", "K", 373.15), (300, "Kelvin", "°C", 26.85),
    (50, "celsius", "°F", 122),
])
def test_temperature_offsets(value, frm, to, expected):
    assert U.convert(value, frm, to, "temperature") == pytest.approx(expected, abs=1e-9 + 1e-4 * abs(expected))


def test_temperature_difference_has_no_offset_and_arrays_round_trip():
    assert U.convert(10, "°C", "K", "temperature_difference") == pytest.approx(10)
    T = np.linspace(-150, 900, 101)
    assert np.allclose(U.convert(U.convert(T, "°C", "°F"), "°F", "°C"), T)
    assert np.allclose(U.to_canonical(T + 273.15, "K", "temperature"), T)
    assert U.si_factor("temperature", "°F") == pytest.approx(5 / 9)
    assert U.si_factor("energy", "MJ") == 1e6 and U.si_factor("energy", "mJ") == 1e-3


# --------------------------------------------------------------------------- headers
@pytest.mark.parametrize("header,name,unit,dim", [
    ("Energy (MJ)", "Energy", "MJ", "energy"),
    ("Force (MN)", "Force", "MN", "force"),
    ("Work (MJ/m3)", "Work", "MJ/m³", "energy_density"),
    ("Time (minutes)", "Time", "min", "time"),
    ("Time (mins)", "Time", "min", "time"),
    ("Time (sec)", "Time", "s", "time"),
    ("Time, s", "Time", "s", "time"),
    ("Load (kilonewtons)", "Load", "kN", "force"),
    ("Extension (millimetres)", "Extension", "mm", "displacement"),
    ("Extension (millimeters)", "Extension", "mm", "displacement"),
    ("Temperature (Kelvin)", "Temperature", "K", "temperature"),
    ("Temperature (deg C)", "Temperature", "°C", "temperature"),
    ("Stress (megapascals)", "Stress", "MPa", "stress"),
    ("Displacement (inch)", "Displacement", "in", "displacement"),
    ("Temp./°C", "Temp", "°C", "temperature"),
    ("DSC/(mW/mg)", "DSC", "mW/mg", "specific_heat_flow"),
    ("DSC/(uV)", "DSC", "µV", "voltage"),
    ("Heat Flow (Normalized) (W/g)", "Heat Flow (Normalized)", "W/g", "specific_heat_flow"),
    ("Ext (%)", "Ext", "%", "strain"),
    ("Ext (mm)", "Ext", "mm", "displacement"),
    ("Standard force [N]", "Standard force", "N", "force"),
    ("Ts (°C)", "Ts", "°C", "temperature"),
    ("Strain %", "Strain", "%", "strain"),
    ("Weight (g)", "Weight", "g", "mass"),
])
def test_parse_header(header, name, unit, dim):
    assert U.parse_header(header) == (name, unit, dim)


def test_parse_header_ex_flags_spelled_units():
    info = U.parse_header_ex("Time (minutes)")
    assert info.guessed and info.unit == "min" and "minutes" in info.reason
    assert not U.parse_header_ex("Time (min)").guessed
    assert U.parse_header_ex("dT/dt").unit is None


@pytest.mark.parametrize("header,dim", [
    ("Point", "count"), ("Index", "count"), ("No.", "count"), ("Count", "count"), ("Cycle", "count"),
    ("Weight", "mass"), ("Mass", "mass"), ("Ext", "displacement"), ("Ext.", "displacement"),
    ("Extensometer", "displacement"), ("Crosshead", "displacement"), ("Position", "displacement"),
    ("Strain gauge", "strain"), ("Standard force", "force"), ("Kraft", "force"), ("Dehnung", "strain"),
    ("Weg", "displacement"), ("Zeit", "time"), ("Temp.", "temperature"), ("Ts", "temperature"),
    ("Tr", "temperature"), ("Elapsed time", "time"), ("Heat flow", "heat_flow"), ("t", "time"),
])
def test_header_keyword_hints(header, dim):
    assert U.hint_dimension(header) == dim
    assert U.parse_header(header)[2] == dim


@pytest.mark.parametrize("header", ["Point", "Count", "Weight", "Ext", "Timestamp", "Last cycle note"])
def test_words_ending_in_t_are_not_time(header):
    assert U.hint_dimension(header) != "time"
