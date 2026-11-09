"""Unit handling for LabStudio.

Every imported column is tagged with a *dimension* (force, stress, temperature, ...)
and a *unit*. On import, values are converted to the CANONICAL unit of their
dimension, so everything downstream (plots, analysis, readouts) is in consistent,
materials-science-friendly units. The user can still display in another unit via
``convert``.

Linear units: value_SI = value * factor.
Temperature uses an offset: value_K = value * factor + offset.

Symbol resolution (:func:`resolve_unit`) is **case-sensitive first**: ``MJ`` (mega) and
``mJ`` (milli) are different units, as are ``MN``/``mN``, ``MW``/``mW``, ``MPa``/``mPa``,
``Ms``/``ms``. A case-insensitive fallback is only used when it cannot swap an SI
``M``/``m`` prefix and the folded spelling has a single meaning. Spelled-out words
(``minutes``, ``kilonewtons``, ``millimetres``, ``Celsius`` ...) are recognised too, but
are reported as *guessed* so the import dialog can ask the user to confirm them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Unit:
    symbol: str
    factor: float          # multiply to get the dimension's base (SI-ish) unit
    offset: float = 0.0    # added after factor (temperature only)


# dimension -> {symbol: Unit}; first listed base is factor 1.
_UNITS: dict[str, dict[str, Unit]] = {}


def _reg(dim: str, *units: Unit) -> None:
    _UNITS.setdefault(dim, {})
    for u in units:
        _UNITS[dim][u.symbol] = u


_reg("force", Unit("N", 1), Unit("kN", 1e3), Unit("mN", 1e-3), Unit("MN", 1e6), Unit("lbf", 4.4482216152605),
     Unit("kgf", 9.80665))
_reg("displacement", Unit("m", 1), Unit("mm", 1e-3), Unit("µm", 1e-6), Unit("nm", 1e-9),
     Unit("cm", 1e-2), Unit("in", 0.0254))
_reg("length", Unit("m", 1), Unit("mm", 1e-3), Unit("µm", 1e-6), Unit("nm", 1e-9),
     Unit("cm", 1e-2), Unit("in", 0.0254))
_reg("area", Unit("m²", 1), Unit("mm²", 1e-6), Unit("µm²", 1e-12), Unit("cm²", 1e-4),
     Unit("in²", 0.00064516))
_reg("stress", Unit("Pa", 1), Unit("kPa", 1e3), Unit("MPa", 1e6), Unit("GPa", 1e9),
     Unit("N/mm²", 1e6), Unit("mPa", 1e-3), Unit("psi", 6894.757293168), Unit("ksi", 6894757.293168))
_reg("strain", Unit("-", 1), Unit("%", 1e-2), Unit("mm/mm", 1), Unit("µε", 1e-6))
_reg("temperature", Unit("K", 1, 0), Unit("°C", 1, 273.15), Unit("°F", 5 / 9, 273.15 - 32 * 5 / 9))
_reg("temperature_difference", Unit("K", 1), Unit("°C", 1))
_reg("time", Unit("s", 1), Unit("ms", 1e-3), Unit("min", 60), Unit("h", 3600))
_reg("heat_flow", Unit("W", 1), Unit("mW", 1e-3), Unit("µW", 1e-6), Unit("MW", 1e6))
_reg("specific_heat_flow", Unit("W/g", 1), Unit("mW/mg", 1), Unit("mW/g", 1e-3), Unit("W/kg", 1e-3))
_reg("energy", Unit("J", 1), Unit("mJ", 1e-3), Unit("kJ", 1e3), Unit("MJ", 1e6))
_reg("specific_energy", Unit("J/g", 1), Unit("kJ/kg", 1), Unit("mJ/mg", 1), Unit("J/kg", 1e-3),
     Unit("mJ/g", 1e-3), Unit("J/mg", 1e3))
_reg("energy_density", Unit("J/m³", 1), Unit("MJ/m³", 1e6), Unit("mJ/mm³", 1e6), Unit("kJ/m³", 1e3),
     Unit("J/cm³", 1e6))
_reg("heating_rate", Unit("K/s", 1), Unit("K/min", 1 / 60), Unit("°C/min", 1 / 60), Unit("°C/s", 1),
     Unit("K/h", 1 / 3600))
_reg("strain_rate", Unit("1/s", 1), Unit("1/min", 1 / 60), Unit("%/s", 1e-2), Unit("%/min", 1e-2 / 60))
_reg("mass", Unit("kg", 1), Unit("g", 1e-3), Unit("mg", 1e-6), Unit("µg", 1e-9))
_reg("velocity", Unit("m/s", 1), Unit("mm/s", 1e-3), Unit("mm/min", 1e-3 / 60))
_reg("voltage", Unit("V", 1), Unit("mV", 1e-3), Unit("µV", 1e-6))
_reg("count", Unit("#", 1))
_reg("dimensionless", Unit("", 1))

# Canonical (display + storage) unit per dimension.
CANONICAL: dict[str, str] = {
    "force": "N",
    "displacement": "mm",
    "length": "mm",
    "area": "mm²",
    "stress": "MPa",
    "strain": "%",
    "temperature": "°C",
    "temperature_difference": "K",
    "time": "s",
    "heat_flow": "mW",
    "specific_heat_flow": "W/g",
    "energy": "J",
    "specific_energy": "J/g",
    "energy_density": "MJ/m³",
    "heating_rate": "K/min",
    "strain_rate": "1/s",
    "mass": "mg",
    "velocity": "mm/min",
    "voltage": "µV",
    "count": "#",
    "dimensionless": "",
}

DIMENSION_LABELS: dict[str, str] = {
    "force": "Force", "displacement": "Displacement", "length": "Length", "area": "Area",
    "stress": "Stress", "strain": "Strain", "temperature": "Temperature",
    "temperature_difference": "Temperature difference", "time": "Time",
    "heat_flow": "Heat flow", "specific_heat_flow": "Specific heat flow", "energy": "Energy",
    "specific_energy": "Specific energy", "energy_density": "Energy density",
    "heating_rate": "Heating rate", "strain_rate": "Strain rate", "mass": "Mass",
    "velocity": "Velocity", "voltage": "Voltage (raw signal)", "count": "Count",
    "dimensionless": "Dimensionless",
}

# Abbreviations / ASCII spellings typed in instrument headers -> registered symbol.
# Keys are matched EXACTLY (case-sensitive) first; see resolve_unit for the fallback rules.
_ALIASES: dict[str, str] = {
    # length / area
    "um": "µm", "μm": "µm", "micron": "µm", "microns": "µm", "mu m": "µm",
    "um2": "µm²", "μm²": "µm²", "µm2": "µm²", "um^2": "µm²", "µm^2": "µm²",
    "mm2": "mm²", "mm^2": "mm²", "sq mm": "mm²", "sq. mm": "mm²", "m2": "m²", "m^2": "m²",
    "cm2": "cm²", "cm^2": "cm²", "in2": "in²", "in^2": "in²", "sq in": "in²", "inch": "in", "inches": "in",
    "\"": "in",
    # temperature
    "degC": "°C", "deg C": "°C", "deg. C": "°C", "°c": "°C", "C": "°C", "℃": "°C", "oC": "°C", "º C": "°C",
    "ºC": "°C", "° C": "°C", "degF": "°F", "deg F": "°F", "°f": "°F", "℉": "°F", "ºF": "°F", "° F": "°F",
    "degK": "K", "deg K": "K", "°K": "K",
    # force
    "kn": "kN", "lbs": "lbf", "lbf.": "lbf", "kp": "kgf",
    # stress
    "n/mm2": "N/mm²", "N/mm2": "N/mm²", "N/mm^2": "N/mm²", "n/mm^2": "N/mm²", "N mm-2": "N/mm²",
    "Mpa": "MPa", "MPA": "MPa", "GPA": "GPa", "Gpa": "GPa", "KPa": "kPa", "kpa": "kPa", "Kpa": "kPa",
    # time
    "sec": "s", "secs": "s", "Sec": "s", "S": "s", "msec": "ms", "Min": "min", "MIN": "min", "mins": "min",
    "min.": "min", "hr": "h", "hrs": "h", "Hr": "h", "H": "h",
    # power / heat flow / voltage
    "uW": "µW", "μW": "µW", "uV": "µV", "μV": "µV",
    "mw/mg": "mW/mg", "MW/MG": "mW/mg", "mW mg-1": "mW/mg", "W g-1": "W/g", "W/G": "W/g", "mw/g": "mW/g",
    # energies
    "j/g": "J/g", "J g-1": "J/g", "kj/kg": "kJ/kg", "j": "J", "kj": "kJ",
    "MJ/m3": "MJ/m³", "MJ/m^3": "MJ/m³", "MJ m-3": "MJ/m³", "MJ·m⁻³": "MJ/m³", "MJm-3": "MJ/m³",
    "J/m3": "J/m³", "J/m^3": "J/m³", "mJ/mm3": "mJ/mm³", "mJ/mm^3": "mJ/mm³", "kJ/m3": "kJ/m³",
    "J/cm3": "J/cm³", "J/cm^3": "J/cm³",
    # rates
    "k/min": "K/min", "K min-1": "K/min", "K·min⁻¹": "K/min", "K min^-1": "K/min", "K.min-1": "K/min",
    "°c/min": "°C/min", "c/min": "°C/min", "C/min": "°C/min", "degC/min": "°C/min", "deg C/min": "°C/min",
    "°C min-1": "°C/min", "°C·min⁻¹": "°C/min", "ºC/min": "°C/min", "K/sec": "K/s", "°C/sec": "°C/s",
    "C/s": "°C/s", "s-1": "1/s", "s^-1": "1/s", "/s": "1/s", "min-1": "1/min", "/min": "1/min",
    "mm min-1": "mm/min", "mm/sec": "mm/s",
    # strain
    "percent": "%", "pct": "%", "mm/mm": "mm/mm", "m/m": "mm/mm", "ue": "µε", "με": "µε", "microstrain": "µε",
    "µstrain": "µε",
    # misc
    "No.": "#", "no.": "#", "count": "#", "counts": "#",
}

# Spelled-out words (matched case-insensitively, after whitespace/punctuation clean-up).
# These are recognised but flagged as *guessed* (see UnitMatch.guessed).
_SPELLED: dict[str, str] = {}


def _spell(symbol: str, *words: str) -> None:
    for w in words:
        _SPELLED[w] = symbol


_spell("s", "second", "seconds")
_spell("ms", "millisecond", "milliseconds")
_spell("min", "minute", "minutes")
_spell("h", "hour", "hours")
_spell("°C", "celsius", "degree celsius", "degrees celsius", "deg celsius", "degrees c", "degree c",
       "centigrade", "degrees centigrade", "grad c", "grad celsius")
_spell("°F", "fahrenheit", "degree fahrenheit", "degrees fahrenheit", "degrees f")
_spell("K", "kelvin", "kelvins", "degrees kelvin")
_spell("N", "newton", "newtons")
_spell("kN", "kilonewton", "kilonewtons", "kilo newton", "kilo newtons")
_spell("mN", "millinewton", "millinewtons")
_spell("MN", "meganewton", "meganewtons")
_spell("lbf", "pound-force", "pounds-force", "pound force", "pounds force")
_spell("kgf", "kilogram-force", "kilogram force", "kilopond")
_spell("m", "metre", "metres", "meter", "meters")
_spell("mm", "millimetre", "millimetres", "millimeter", "millimeters")
_spell("cm", "centimetre", "centimetres", "centimeter", "centimeters")
_spell("µm", "micrometre", "micrometres", "micrometer", "micrometers")
_spell("nm", "nanometre", "nanometres", "nanometer", "nanometers")
_spell("mm²", "square millimetre", "square millimetres", "square millimeter", "square millimeters")
_spell("Pa", "pascal", "pascals")
_spell("kPa", "kilopascal", "kilopascals")
_spell("MPa", "megapascal", "megapascals", "mega pascal", "mega pascals")
_spell("GPa", "gigapascal", "gigapascals")
_spell("%", "per cent", "percentage")
_spell("W", "watt", "watts")
_spell("mW", "milliwatt", "milliwatts")
_spell("µW", "microwatt", "microwatts")
_spell("V", "volt", "volts")
_spell("mV", "millivolt", "millivolts")
_spell("µV", "microvolt", "microvolts")
_spell("J", "joule", "joules")
_spell("mJ", "millijoule", "millijoules")
_spell("kJ", "kilojoule", "kilojoules")
_spell("MJ", "megajoule", "megajoules")
_spell("g", "gram", "grams", "gramme", "grammes")
_spell("mg", "milligram", "milligrams", "milligramme", "milligrammes")
_spell("kg", "kilogram", "kilograms", "kilogramme", "kilogrammes")
_spell("µg", "microgram", "micrograms")
_spell("K/min", "kelvin per minute", "kelvin/minute", "k per min")
_spell("°C/min", "degrees per minute", "degc per min", "celsius per minute", "deg c per min")
_spell("mm/min", "millimetres per minute", "millimeters per minute", "mm per min")

# Case-folded spellings that are ambiguous between an M (mega) and m (milli) prefix and for
# which only one reading is physically plausible in a materials lab: folded -> preferred.
_AMBIGUOUS_PREFERRED: dict[str, str] = {"mpa": "MPa"}

# Header keyword -> dimension guess (checked in order, on the lower-cased header name).
_HEADER_HINTS: list[tuple[str, str]] = [
    (r"heat\s*flow|\bdsc\b|^hf\b|w(ä|ae)rmestrom", "heat_flow"),
    (r"stress|σ|spannung", "stress"),
    (r"strain\s*gau?ge|strain|ε|dehnung", "strain"),
    (r"\btemp|temperatur|^t\s*[\(\[/]|^t[sr]\b|°c|°f|kelvin|celsius", "temperature"),
    (r"force|load|kraft", "force"),
    (r"disp|extens|^ext\b|position|crosshead|stroke|elongation|travel|\bweg\b|standardweg|traverse|"
     r"verformung|deform|verl(ä|ae)ngerung", "displacement"),
    (r"\btime\b|^t$|zeit|elapsed|dauer", "time"),
    (r"cycle|zyklus", "count"),
    (r"^(point|points|index|idx|no\.?|nr\.?|number|count|counter|#|row|n|step|sample\s*no\.?|"
     r"data\s*point|record)$", "count"),
    (r"^(weight|mass|masse|einwaage|sample\s*mass|sample\s*weight)\b", "mass"),
]


def dimensions() -> list[str]:
    return list(_UNITS.keys())


def units_for(dimension: str) -> list[str]:
    return list(_UNITS[dimension].keys())


def canonical_unit(dimension: str) -> str:
    return CANONICAL[dimension]


# --------------------------------------------------------------------------- symbol resolution
@dataclass(frozen=True)
class UnitMatch:
    """Result of :func:`resolve_unit`.

    ``how``: ``exact`` (registered symbol), ``alias`` (known abbreviation), ``case``
    (case-insensitive, unambiguous), ``spelled`` (spelled-out word), ``ambiguous`` (several
    readings, the preferred one was chosen) or ``unknown`` (``symbol`` is None).
    """
    text: str
    symbol: str | None
    how: str
    candidates: tuple[str, ...] = ()

    @property
    def guessed(self) -> bool:
        """True when the user should confirm the interpretation."""
        return self.how in ("spelled", "ambiguous")

    @property
    def reason(self) -> str:
        if self.how == "spelled":
            return f"unit read from the word '{self.text}' → {self.symbol}"
        if self.how == "ambiguous":
            alts = ", ".join(c for c in self.candidates if c != self.symbol)
            return f"'{self.text}' is ambiguous (could be {alts}); assumed {self.symbol}"
        if self.how == "unknown":
            return f"unit '{self.text}' not recognised"
        return ""


def _all_symbols() -> set[str]:
    return {s for table in _UNITS.values() for s in table}


_SYMBOLS: set[str] = _all_symbols()


def _fold_index() -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for s in _SYMBOLS:
        idx.setdefault(s.lower(), set()).add(s)
    for a, s in _ALIASES.items():
        idx.setdefault(a.lower(), set()).add(s)
    return idx


_FOLD: dict[str, set[str]] = _fold_index()


def _swaps_m_prefix(text: str, target: str) -> bool:
    """True if matching ``text`` to ``target`` would change an M (mega) into m (milli) or back."""
    if not text or not target or text[0] == target[0]:
        return False
    if {text[0], target[0]} != {"M", "m"}:
        return False
    # 'Min' -> 'min' is a capitalised word, not a prefix swap
    return not target.lower().startswith("min")


def _clean_unit_text(symbol: str) -> str:
    s = str(symbol).strip()
    # strip one level of surrounding brackets: '(kN)', '[mm]', '{s}'
    while len(s) >= 2 and s[0] in "([{" and s[-1] in ")]}":
        s = s[1:-1].strip()
    s = s.strip("[]()").strip()
    return s.replace(" ", " ")


def resolve_unit(symbol: str | None) -> UnitMatch:
    """Resolve a unit string from a file to a registered symbol (see module docstring)."""
    if symbol is None:
        return UnitMatch("", None, "unknown")
    s = _clean_unit_text(symbol)
    if s in _SYMBOLS:
        return UnitMatch(s, s, "exact")
    if s in _ALIASES:
        return UnitMatch(s, _ALIASES[s], "alias")
    s2 = re.sub(r"\s+", " ", s)
    if s2 != s and (s2 in _SYMBOLS or s2 in _ALIASES):
        return UnitMatch(s, _ALIASES.get(s2, s2), "alias")
    low = s2.lower()
    if low in _AMBIGUOUS_PREFERRED:
        pref = _AMBIGUOUS_PREFERRED[low]
        cands = tuple(sorted(_FOLD.get(low, {pref}) | {pref}))
        if s2 == pref:
            return UnitMatch(s, pref, "exact")
        return UnitMatch(s, pref, "ambiguous", cands)
    targets = _FOLD.get(low)
    if targets:
        ok = [t for t in targets if not _swaps_m_prefix(s2, t)]
        if len(targets) == 1 and ok:
            return UnitMatch(s, ok[0], "case")
        if len(ok) == 1:
            # several readings, only one keeps the M/m prefix as written (e.g. 'mn' -> mN, 'Mw' -> MW)
            return UnitMatch(s, ok[0], "case", tuple(sorted(targets)))
    word = re.sub(r"[\s_]+", " ", low).strip(" .")
    if word != low and word in _FOLD and len(_FOLD[word]) == 1:      # 'min.' -> min, 'K .' -> K
        t = next(iter(_FOLD[word]))
        if not _swaps_m_prefix(s2.strip(" ."), t):
            return UnitMatch(s, t, "case")
    for w in (word, word[:-1] if word.endswith("s") and len(word) > 4 else None):
        if w and w in _SPELLED:
            return UnitMatch(s, _SPELLED[w], "spelled")
    return UnitMatch(s, None, "unknown")


def normalize_symbol(symbol: str) -> str:
    """Registered symbol for ``symbol`` (e.g. 'kn' -> 'kN', 'minutes' -> 'min'); unchanged if unknown.

    Case-sensitive for SI prefixes: 'MJ' stays mega-joule, 'mJ' milli-joule.
    """
    m = resolve_unit(symbol)
    return m.symbol if m.symbol is not None else _clean_unit_text(symbol)


def dimension_of(symbol: str) -> str | None:
    """Return the first dimension that has this unit symbol (ambiguous ones like mm -> displacement)."""
    s = resolve_unit(symbol).symbol
    if s is None:
        return None
    for dim, table in _UNITS.items():
        if s in table:
            return dim
    return None


def dimensions_of(symbol: str) -> list[str]:
    """All dimensions that contain this unit symbol (e.g. 'mm' -> displacement, length)."""
    s = resolve_unit(symbol).symbol
    return [d for d, t in _UNITS.items() if s is not None and s in t]


def convert(values, from_unit: str, to_unit: str, dimension: str | None = None):
    """Convert scalar/array ``values`` between units of the same dimension."""
    fu, tu = normalize_symbol(from_unit), normalize_symbol(to_unit)
    if fu == tu:
        return values
    dim = dimension or dimension_of(fu)
    if dim is None or dim not in _UNITS or fu not in _UNITS[dim] or tu not in _UNITS[dim]:
        raise ValueError(f"Cannot convert '{from_unit}' -> '{to_unit}' (dimension {dim})")
    a, b = _UNITS[dim][fu], _UNITS[dim][tu]
    v = np.asarray(values, dtype=float) if not np.isscalar(values) else float(values)
    base = v * a.factor + a.offset
    return (base - b.offset) / b.factor


def to_canonical(values, unit: str, dimension: str):
    return convert(values, unit, CANONICAL[dimension], dimension)


# --------------------------------------------------------------------------- headers
@dataclass(frozen=True)
class HeaderInfo:
    """Detailed result of :func:`parse_header_ex`."""
    name: str
    unit: str | None              # registered symbol, None if absent/unrecognised
    dimension: str | None
    unit_text: str | None         # the unit text exactly as found in the header
    unit_how: str                 # UnitMatch.how, or 'none' when the header has no unit part
    reason: str = ""

    @property
    def guessed(self) -> bool:
        return self.unit_how in ("spelled", "ambiguous")


_TRAILING_UNIT_RE = re.compile(
    r"\s(%|mm|N|kN|MPa|GPa|°C|K|s|min|mW|W/g|mg|µm|um|µε)$")


def hint_dimension(name: str) -> str | None:
    """Dimension suggested by header keywords only (no unit), e.g. 'Standard force' -> force."""
    low = name.strip().lower()
    for pattern, d in _HEADER_HINTS:
        if re.search(pattern, low):
            return d
    return None


def parse_header_ex(header: str) -> HeaderInfo:
    """Split an instrument column header into name / unit / dimension, with diagnostics.

    Handles 'Force (kN)', 'Force [N]', 'Temp./°C', 'DSC/(mW/mg)', 'Strain %', 'Time, s',
    'Time (minutes)'.
    """
    h = header.strip()
    unit_text = None
    m = re.search(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$", h)
    if m:
        unit_text = m.group(1)
        name = h[: m.start()].strip(" /,")
    elif "/" in h and not h.lower().startswith(("mm/", "n/")) and not re.fullmatch(r"d\s*T\s*/\s*dt", h, re.I):
        name, _, unit_text = h.partition("/")
        unit_text = unit_text.strip("() ")
        name = name.strip(" .")
        if not unit_text:
            name, unit_text = h, None
    elif "," in h and resolve_unit(h.rpartition(",")[2]).symbol is not None:
        name, _, unit_text = h.rpartition(",")
        name, unit_text = name.strip(), unit_text.strip()
    else:
        m2 = _TRAILING_UNIT_RE.search(h)
        if m2:
            unit_text = m2.group(1)
            name = h[: m2.start()].strip()
        else:
            name = h
    match = resolve_unit(unit_text) if unit_text else UnitMatch("", None, "none")
    unit_n = match.symbol
    dim = hint_dimension(name) or hint_dimension(h)
    if unit_n:
        udims = dimensions_of(unit_n)
        if udims and (dim is None or unit_n not in _UNITS.get(dim, {})):
            # e.g. 'Extension (mm)' -> displacement; 'DSC (mW/mg)' -> specific_heat_flow;
            # 'Ext (%)' -> strain
            dim = udims[0] if not (dim == "displacement" and udims[0] == "length") else dim
        if dim == "heat_flow" and unit_n in _UNITS["specific_heat_flow"]:
            dim = "specific_heat_flow"
    reason = match.reason if (match.guessed or match.how == "unknown") else ""
    return HeaderInfo(name or header, unit_n, dim, unit_text, match.how, reason)


def parse_header(header: str) -> tuple[str, str | None, str | None]:
    """Split an instrument column header into (name, unit, dimension guess).

    Handles 'Force (kN)', 'Force [N]', 'Temp./°C', 'DSC/(mW/mg)', 'Strain %'. Unrecognised
    unit text yields ``unit`` = the cleaned text (not a registered symbol).
    """
    info = parse_header_ex(header)
    unit = info.unit if info.unit is not None else (_clean_unit_text(info.unit_text) if info.unit_text else None)
    return info.name, unit, info.dimension


def axis_label(name: str, unit: str | None) -> str:
    return f"{name} ({unit})" if unit else name


def si_factor(dimension: str, unit: str) -> float:
    """Multiplicative factor from ``unit`` to the dimension's base unit (offset units: K for temperature)."""
    return _UNITS[dimension][normalize_symbol(unit)].factor
