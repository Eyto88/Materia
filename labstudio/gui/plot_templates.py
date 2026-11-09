"""User plot style templates (module B).

A template is the *look* of a figure — :data:`plot_style.LOOK_FIELDS` (fonts,
sizes, ticks, grid, legend style, line width, colour cycle, mathtext font, panel
labels…), a matching rcParams subset (``"rc"``, informative and also accepted
on import), and the figure size in cm. Titles, labels, limits, units, data and
layout are never part of a template, so a template can be applied to any figure.

Stored as JSON in ``$LABSTUDIO_HOME/styles`` (or ``~/.labstudio/styles``); three
read-only built-ins are always available (a user file with the same name wins).
The default template for new figures is the app setting ``default_style_template``.
"""
from __future__ import annotations

import copy
import json
import os
import re

from .plot_style import LOOK_FIELDS, PlotSettings

FORMAT = "labstudio-style-template"
VERSION = 1
#: settings saved in templates in addition to LOOK_FIELDS
EXTRA_FIELDS = ("legend_loc", "legend_mode", "legend_on")

THESIS = "Thesis (Imperial, serif 11 pt, 15 cm wide)"
JOURNAL = "Journal single column (8.5 cm)"
SLIDES = "Slides (16:9, large)"


def styles_dir(create: bool = False) -> str:
    """Folder holding user templates: ``$LABSTUDIO_HOME/styles`` or ``~/.labstudio/styles``."""
    home = os.environ.get("LABSTUDIO_HOME") or os.path.join(os.path.expanduser("~"), ".labstudio")
    d = os.path.join(home, "styles")
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def _rc_from(vals: dict) -> dict:
    from .plot_style import resolve_family
    return {
        "font.family": resolve_family(vals.get("font_family", "sans-serif")),
        "axes.titlesize": vals.get("title_size"), "axes.labelsize": vals.get("label_size"),
        "xtick.labelsize": vals.get("tick_size"), "ytick.labelsize": vals.get("tick_size"),
        "xtick.direction": vals.get("tick_direction"), "ytick.direction": vals.get("tick_direction"),
        "xtick.major.size": vals.get("tick_major_len"), "ytick.major.size": vals.get("tick_major_len"),
        "xtick.major.width": vals.get("tick_major_width"), "ytick.major.width": vals.get("tick_major_width"),
        "axes.linewidth": vals.get("axes_linewidth"), "lines.linewidth": vals.get("linewidth"),
        "lines.markersize": vals.get("markersize"), "legend.fontsize": vals.get("legend_size"),
        "legend.frameon": vals.get("legend_frame"), "mathtext.fontset": vals.get("mathtext_fontset"),
        "savefig.dpi": vals.get("export_dpi"), "axes.grid": vals.get("grid_major"),
    }


#: rc key -> settings field (used when importing a hand-written template that only has "rc")
_RC_TO_FIELD = {"axes.titlesize": "title_size", "axes.labelsize": "label_size", "xtick.labelsize": "tick_size",
                "xtick.direction": "tick_direction", "xtick.major.size": "tick_major_len",
                "xtick.major.width": "tick_major_width", "axes.linewidth": "axes_linewidth",
                "lines.linewidth": "linewidth", "lines.markersize": "markersize", "legend.fontsize": "legend_size",
                "legend.frameon": "legend_frame", "mathtext.fontset": "mathtext_fontset",
                "savefig.dpi": "export_dpi", "axes.grid": "grid_major", "font.family": "font_family"}


def template_from_settings(s: PlotSettings, name: str, figure_size_cm: tuple[float, float] | None = None) -> dict:
    """Template dict capturing the look of ``s`` (figure size defaults to its export size)."""
    vals = {k: copy.deepcopy(getattr(s, k)) for k in LOOK_FIELDS + EXTRA_FIELDS}
    w, h = figure_size_cm or (s.export_w_cm, s.export_h_cm)
    return {"format": FORMAT, "version": VERSION, "name": str(name), "settings": vals,
            "rc": _rc_from(vals), "figure_size_cm": [float(w), float(h)]}


def _builtin(name: str, size_cm, **vals) -> dict:
    base = PlotSettings()
    s = {k: copy.deepcopy(getattr(base, k)) for k in LOOK_FIELDS + EXTRA_FIELDS}
    s.update(vals)
    return {"format": FORMAT, "version": VERSION, "name": name, "builtin": True, "settings": s,
            "rc": _rc_from(s), "figure_size_cm": list(size_cm)}


BUILTIN_TEMPLATES: dict[str, dict] = {
    THESIS: _builtin(THESIS, (15.0, 10.0), font_family="serif", title_size=11, label_size=11, tick_size=10,
                     legend_size=9, tick_direction="in", ticks_all_sides=True, minor_ticks=True, grid_major=False,
                     grid_minor=False, legend_frame=False, linewidth=1.2, export_dpi=300, markersize=4.0,
                     axes_linewidth=0.8, tick_major_len=4.0, tick_major_width=0.8, mathtext_fontset="stix",
                     panel_labels=True, panel_label_pos="top left", panel_label_bold=True),
    JOURNAL: _builtin(JOURNAL, (8.5, 6.5), font_family="sans-serif", title_size=8, label_size=8, tick_size=7,
                      legend_size=7, tick_direction="in", ticks_all_sides=True, minor_ticks=True, grid_major=False,
                      grid_minor=False, legend_frame=False, linewidth=1.0, export_dpi=600, markersize=3.0,
                      axes_linewidth=0.6, tick_major_len=3.0, tick_major_width=0.6, mathtext_fontset="dejavusans",
                      panel_labels=True, panel_label_pos="top left", panel_label_bold=True),
    SLIDES: _builtin(SLIDES, (25.4, 14.29), font_family="sans-serif", title_size=20, label_size=18, tick_size=15,
                     legend_size=14, tick_direction="out", ticks_all_sides=False, minor_ticks=False, grid_major=True,
                     grid_minor=False, legend_frame=True, linewidth=2.5, export_dpi=150, markersize=7.0,
                     axes_linewidth=1.2, tick_major_len=6.0, tick_major_width=1.2, mathtext_fontset="dejavusans",
                     panel_labels=True, panel_label_pos="top left", panel_label_bold=True),
}


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\-. ]+", "_", name, flags=re.UNICODE).strip().replace(" ", "_")
    return (s or "template")[:80]


def _user_files() -> dict[str, str]:
    d = styles_dir()
    out: dict[str, str] = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".json"):
            continue
        p = os.path.join(d, fn)
        try:
            with open(p, encoding="utf-8") as fh:
                name = json.load(fh).get("name") or fn[:-5]
        except (OSError, ValueError, AttributeError):
            continue
        out[str(name)] = p
    return out


def list_templates() -> list[str]:
    """Built-in template names followed by the user's saved templates."""
    names = list(BUILTIN_TEMPLATES)
    names += [n for n in _user_files() if n not in names]
    return names


def is_builtin(name: str) -> bool:
    return name in BUILTIN_TEMPLATES and name not in _user_files()


def validate_template(t: dict) -> dict:
    """Normalise a template dict (fills ``settings`` from ``rc`` if needed); raises ValueError."""
    if not isinstance(t, dict) or not t.get("name"):
        raise ValueError("Not a LabStudio style template (missing 'name').")
    t = copy.deepcopy(t)
    vals = dict(t.get("settings") or {})
    for rc, f in _RC_TO_FIELD.items():
        if f not in vals and (t.get("rc") or {}).get(rc) is not None:
            vals[f] = t["rc"][rc]
    vals = {k: v for k, v in vals.items() if k in LOOK_FIELDS + EXTRA_FIELDS}
    if not vals:
        raise ValueError("The template contains no style settings.")
    t["settings"] = vals
    t.setdefault("format", FORMAT)
    t.setdefault("version", VERSION)
    size = t.get("figure_size_cm")
    if size is not None:
        w, h = (float(v) for v in size)
        if not (1 <= w <= 200 and 1 <= h <= 200):
            raise ValueError("figure_size_cm out of range.")
        t["figure_size_cm"] = [w, h]
    return t


def get_template(name: str) -> dict:
    """Template by name (user file first, then built-in). KeyError if unknown."""
    files = _user_files()
    if name in files:
        with open(files[name], encoding="utf-8") as fh:
            return validate_template(json.load(fh))
    if name in BUILTIN_TEMPLATES:
        return copy.deepcopy(BUILTIN_TEMPLATES[name])
    raise KeyError(f"No style template named '{name}'")


def save_template(t: dict) -> str:
    """Write template ``t`` to the styles folder; returns the file path."""
    t = validate_template(t)
    t.pop("builtin", None)
    path = _user_files().get(t["name"]) or os.path.join(styles_dir(create=True), _slug(t["name"]) + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(t, fh, indent=2, ensure_ascii=False)
    return path


def delete_template(name: str) -> bool:
    """Delete a user template (built-ins cannot be deleted). Returns True if a file was removed."""
    p = _user_files().get(name)
    if p and os.path.isfile(p):
        os.remove(p)
        return True
    return False


def export_template(name: str, path: str) -> str:
    """Write template ``name`` to ``path`` (to share with labmates)."""
    t = get_template(name)
    t.pop("builtin", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(t, fh, indent=2, ensure_ascii=False)
    return path


def import_template(path: str) -> str:
    """Copy a template file into the styles folder; returns its name."""
    with open(path, encoding="utf-8") as fh:
        t = validate_template(json.load(fh))
    save_template(t)
    return t["name"]


def apply_to_settings(t: dict, s: PlotSettings) -> PlotSettings:
    """Copy of ``s`` with the look of template ``t`` (titles, labels, limits, units, layout kept)."""
    t = validate_template(t)
    out = s.copy()
    for k, v in t["settings"].items():
        setattr(out, k, copy.deepcopy(v))
    if t.get("figure_size_cm"):
        out.export_w_cm, out.export_h_cm = t["figure_size_cm"]
    out.template = t["name"]
    return out
