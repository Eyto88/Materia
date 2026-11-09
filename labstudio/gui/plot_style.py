"""Plot styling for LabStudio (module B): colour cycle, style presets and the
Qt-free function that applies a :class:`PlotSettings` to matplotlib axes.

Styling is applied *directly to artists* (not only via global rcParams) so changes
in the customisation panel update the live figure immediately and never leak into
other matplotlib users in the application. ``rc_for_preset`` gives the matching
rcParams for code that wants to build figures with the same look.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache

# Okabe–Ito based categorical palette, reordered (and yellow → wine) so every
# adjacent pair stays separable under deutan/protan/tritan colour-vision deficiency.
COLOR_CYCLE: list[str] = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#882255"]
# After the colours run out, the line style changes too (composite encoding),
# so 20+ traces remain distinguishable without inventing new hues.
LINESTYLE_CYCLE: list[str] = ["-", "--", "-.", ":"]

LINESTYLES: dict[str, str] = {"Solid": "-", "Dashed": "--", "Dash-dot": "-.", "Dotted": ":", "None": "None"}
MARKERS: dict[str, str] = {"None": "None", "Circle": "o", "Square": "s", "Triangle": "^", "Diamond": "D",
                           "Cross": "x", "Plus": "+", "Point": ".", "Star": "*", "Down triangle": "v"}
LEGEND_LOCATIONS: list[str] = ["best", "upper right", "upper left", "lower left", "lower right", "right",
                               "center left", "center right", "lower center", "upper center", "center",
                               "outside right"]
TICK_DIRECTIONS: list[str] = ["out", "in", "inout"]

PRESET_NAMES: list[str] = ["Default", "Publication (serif, 300 dpi, inward ticks)", "Presentation (large fonts)"]
LEGEND_MODES: list[str] = ["auto", "full", "colorbar", "off"]
LAYOUT_PRESETS: dict[str, tuple[int, int]] = {"1×1": (1, 1), "1×2": (1, 2), "2×1": (2, 1), "2×2": (2, 2),
                                              "1×3": (1, 3), "3×1": (3, 1)}
PANEL_LABEL_POSITIONS: list[str] = ["top left", "top left outside", "top right", "bottom left"]
MATHTEXT_FONTSETS: list[str] = ["dejavusans", "dejavuserif", "cm", "stix", "stixsans"]

_SERIF_PREF = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "STIX Two Text",
               "STIXGeneral", "DejaVu Serif"]
_SANS_PREF = ["Arial", "Helvetica", "Liberation Sans", "Segoe UI", "DejaVu Sans"]


@lru_cache(maxsize=1)
def available_fonts() -> list[str]:
    """Installed font family names (sorted), as matplotlib sees them."""
    from matplotlib import font_manager
    return sorted({f.name for f in font_manager.fontManager.ttflist})


def resolve_family(family: str) -> str:
    """Map generic names ('serif', 'sans-serif') to a concrete installed family."""
    fonts = set(available_fonts())
    pref = {"serif": _SERIF_PREF, "sans-serif": _SANS_PREF, "sans": _SANS_PREF}.get(family.lower())
    if pref is None:
        return family if family in fonts else "DejaVu Sans"
    return next((f for f in pref if f in fonts), "DejaVu Sans")


def font_choices() -> list[str]:
    """Short list for the font combo: generic families first, then common installed ones."""
    fonts = available_fonts()
    common = [f for f in _SANS_PREF + _SERIF_PREF + ["DejaVu Sans Mono", "Calibri", "Cambria", "Georgia",
                                                     "Verdana", "Computer Modern", "CMU Serif"] if f in fonts]
    seen: list[str] = ["sans-serif", "serif"]
    for f in common + fonts:
        if f not in seen:
            seen.append(f)
    return seen


@dataclass
class AxisSettings:
    """Per-axis options. ``label`` empty → automatic from channel labels."""
    label: str = ""
    auto_limits: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    scale: str = "linear"        # 'linear' | 'log'
    invert: bool = False
    unit: str | None = None      # display-unit override (None = canonical)


@dataclass
class PanelSettings:
    """Per-panel texts/axes for panels 2…n of a multi-panel figure (panel 1 uses
    the fields of :class:`PlotSettings` itself, so single-panel code is unchanged)."""
    title: str = ""
    x: AxisSettings = field(default_factory=AxisSettings)
    y: AxisSettings = field(default_factory=AxisSettings)
    y2: AxisSettings = field(default_factory=AxisSettings)


@dataclass
class PlotSettings:
    """All customisation state of a PlotPanel (sizes in points).

    ``title``/``x``/``y``/``y2`` belong to panel 0; further panels keep theirs in
    ``subpanels`` (index ``i - 1``)."""
    preset: str = "Default"
    title: str = ""
    x: AxisSettings = field(default_factory=AxisSettings)
    y: AxisSettings = field(default_factory=AxisSettings)
    y2: AxisSettings = field(default_factory=AxisSettings)
    font_family: str = "sans-serif"
    title_size: float = 12.0
    label_size: float = 11.0
    tick_size: float = 10.0
    tick_direction: str = "out"
    ticks_all_sides: bool = False
    minor_ticks: bool = False
    grid_major: bool = True
    grid_minor: bool = False
    legend_on: bool = True
    legend_loc: str = "best"
    legend_ncol: int = 1
    legend_frame: bool = True
    legend_size: float = 9.0
    fig_facecolor: str = "#ffffff"
    ax_facecolor: str = "#ffffff"
    linewidth: float = 1.5          # default width for new traces / applied by presets
    export_dpi: int = 300
    # ---- v1.1 look
    color_cycle: list = field(default_factory=lambda: list(COLOR_CYCLE))
    markersize: float = 4.0         # default marker size for new traces
    axes_linewidth: float = 0.8     # spine width
    tick_major_len: float = 4.5
    tick_major_width: float = 0.8
    mathtext_fontset: str = "dejavusans"
    legend_mode: str = "auto"       # auto | full | colorbar | off
    # ---- v1.1 layout (multi-panel)
    layout_rows: int = 1
    layout_cols: int = 1
    share_x: bool = False
    share_y: bool = False
    panel_labels: bool = True       # (a), (b)… when there is more than one panel
    panel_label_pos: str = "top left"
    panel_label_bold: bool = True
    subpanels: list = field(default_factory=list)   # list[PanelSettings] for panels 1..n-1
    # ---- v1.1 waterfall (display only)
    offset_y: float = 0.0
    offset_mode: str = "absolute"   # absolute (y display units) | percent (of the y range)
    offset_x: float = 0.0
    offset_labels: bool = False     # trace labels at the right end of each trace
    # ---- v1.1 export / preview
    export_w_cm: float = 16.0
    export_h_cm: float = 11.0
    preview_export: bool = False
    template: str = ""              # name of the last applied style template

    def copy(self) -> "PlotSettings":
        return copy.deepcopy(self)

    def panel(self, i: int):
        """Settings object (``.title/.x/.y/.y2``) of panel ``i`` (created on demand)."""
        if i <= 0:
            return self
        while len(self.subpanels) < i:
            self.subpanels.append(PanelSettings())
        return self.subpanels[i - 1]

    def to_dict(self) -> dict:
        """JSON-serialisable dict (see :func:`settings_from_dict`)."""
        return asdict(self)


#: Fields that define the *look* (saved in style templates, reset by the Default preset).
LOOK_FIELDS: tuple[str, ...] = (
    "font_family", "title_size", "label_size", "tick_size", "tick_direction", "ticks_all_sides", "minor_ticks",
    "grid_major", "grid_minor", "legend_frame", "legend_size", "fig_facecolor", "ax_facecolor", "linewidth",
    "export_dpi", "color_cycle", "markersize", "axes_linewidth", "tick_major_len", "tick_major_width",
    "mathtext_fontset", "panel_labels", "panel_label_pos", "panel_label_bold")


def _axis_from(d) -> AxisSettings:
    a = AxisSettings()
    for k, v in (d or {}).items():
        if hasattr(a, k):
            setattr(a, k, v)
    return a


def settings_from_dict(d: dict, base: PlotSettings | None = None) -> PlotSettings:
    """Inverse of :meth:`PlotSettings.to_dict` (unknown keys ignored, missing keep ``base``)."""
    s = (base or PlotSettings()).copy()
    names = {f.name for f in fields(PlotSettings)}
    for k, v in (d or {}).items():
        if k not in names:
            continue
        if k in ("x", "y", "y2"):
            setattr(s, k, _axis_from(v))
        elif k == "subpanels":
            s.subpanels = [PanelSettings(p.get("title", ""), _axis_from(p.get("x")), _axis_from(p.get("y")),
                                         _axis_from(p.get("y2"))) for p in (v or [])]
        elif k == "color_cycle":
            s.color_cycle = [str(c) for c in v] or list(COLOR_CYCLE)
        else:
            setattr(s, k, v)
    return s


def preset_settings(name: str, base: PlotSettings | None = None) -> PlotSettings:
    """Return a copy of ``base`` with the look of preset ``name`` applied
    (titles, labels, limits and unit overrides are kept)."""
    s = (base or PlotSettings()).copy()
    s.preset = name
    common = dict(fig_facecolor="#ffffff", ax_facecolor="#ffffff", legend_ncol=s.legend_ncol)
    vals: dict
    if name.startswith("Publication"):
        vals = dict(font_family="serif", title_size=11, label_size=11, tick_size=10, legend_size=9,
                    tick_direction="in", ticks_all_sides=True, minor_ticks=True, grid_major=False,
                    grid_minor=False, legend_frame=False, linewidth=1.2, export_dpi=300)
    elif name.startswith("Presentation"):
        vals = dict(font_family="sans-serif", title_size=20, label_size=18, tick_size=15, legend_size=14,
                    tick_direction="out", ticks_all_sides=False, minor_ticks=False, grid_major=True,
                    grid_minor=False, legend_frame=True, linewidth=2.5, export_dpi=150)
    else:
        d = PlotSettings()
        vals = {k: copy.deepcopy(getattr(d, k)) for k in LOOK_FIELDS}
    for k, v in {**common, **vals}.items():
        setattr(s, k, v)
    return s


def rc_for_preset(name: str) -> dict:
    """rcParams equivalent of a preset (for figures created outside PlotPanel)."""
    s = preset_settings(name)
    return {
        "font.family": resolve_family(s.font_family), "axes.titlesize": s.title_size,
        "axes.labelsize": s.label_size, "xtick.labelsize": s.tick_size, "ytick.labelsize": s.tick_size,
        "xtick.direction": s.tick_direction, "ytick.direction": s.tick_direction,
        "xtick.top": s.ticks_all_sides, "ytick.right": s.ticks_all_sides,
        "xtick.minor.visible": s.minor_ticks, "ytick.minor.visible": s.minor_ticks,
        "axes.grid": s.grid_major, "legend.frameon": s.legend_frame, "legend.fontsize": s.legend_size,
        "lines.linewidth": s.linewidth, "savefig.dpi": s.export_dpi,
        "axes.prop_cycle": __import__("matplotlib").cycler(color=COLOR_CYCLE),
        "figure.constrained_layout.use": True,
    }


def cycle_style(i: int, colors: list[str] | None = None) -> tuple[str, str]:
    """Colour and line style for the i-th trace (``colors`` = palette, default :data:`COLOR_CYCLE`)."""
    colors = list(colors) if colors else COLOR_CYCLE
    return colors[i % len(colors)], LINESTYLE_CYCLE[(i // len(colors)) % len(LINESTYLE_CYCLE)]


def _style_ticks(ax, s: PlotSettings, family: str, *, secondary: bool = False) -> None:
    common = dict(direction=s.tick_direction, labelsize=s.tick_size, labelfontfamily=family)
    if secondary:
        ax.tick_params(axis="y", which="both", right=True, left=False, labelright=True, labelleft=False,
                       **common)
        ax.tick_params(axis="y", which="major", length=s.tick_major_len * (1.33 if s.tick_direction == "inout"
                                                                             else 1.0), width=s.tick_major_width)
        return
    ax.tick_params(axis="both", which="both", **common)
    ax.tick_params(axis="x", which="both", top=s.ticks_all_sides, bottom=True)
    ax.tick_params(axis="y", which="both", left=True)
    ax.tick_params(which="major", length=s.tick_major_len * (1.33 if s.tick_direction == "inout" else 1.0),
                   width=s.tick_major_width)
    ax.tick_params(which="minor", length=s.tick_major_len * 0.55, width=s.tick_major_width * 0.75)


def style_axes(fig, ax, ax2, s: PlotSettings, *, auto_labels: dict[str, str] | None = None,
               data_limits: dict[str, tuple[float, float] | None] | None = None,
               legend_handles: list | None = None, autoscale: bool = True) -> None:
    """Apply ``s`` to ``fig``/``ax``/``ax2`` (``ax2`` may be None). Qt-free.

    ``auto_labels`` maps 'x'/'y'/'y2' to automatic axis labels used when the user
    label is empty. ``data_limits`` maps the same keys to the (min, max) of the
    *data* used for automatic limits (so overlays such as long tangent lines never
    blow up the scale); missing keys fall back to matplotlib autoscaling.
    ``legend_handles`` fixes legend order; default = labelled lines of both axes.
    ``autoscale=False`` leaves automatic axes at their current (e.g. zoomed) view.
    """
    data_limits = data_limits or {}
    family = resolve_family(s.font_family)
    auto_labels = auto_labels or {}
    fig.set_facecolor(s.fig_facecolor)
    ax.set_facecolor(s.ax_facecolor)
    ax.set_title(s.title, fontsize=s.title_size, fontfamily=family)
    ax.set_xlabel(s.x.label or auto_labels.get("x", ""), fontsize=s.label_size, fontfamily=family)
    ax.set_ylabel(s.y.label or auto_labels.get("y", ""), fontsize=s.label_size, fontfamily=family)

    pairs = [(ax, "x", s.x, "x"), (ax, "y", s.y, "y")]
    if ax2 is not None:
        ax2.set_ylabel(s.y2.label or auto_labels.get("y2", ""), fontsize=s.label_size, fontfamily=family)
        pairs.append((ax2, "y", s.y2, "y2"))
    for a, which, axs, key in pairs:
        _apply_axis(a, which, axs, data_limits.get(key, "missing"), autoscale)

    _style_ticks(ax, s, family)
    if s.ticks_all_sides and ax2 is None:
        ax.tick_params(axis="y", which="both", right=True)
    else:
        ax.tick_params(axis="y", which="both", right=False)
    if ax2 is not None:
        _style_ticks(ax2, s, family, secondary=True)
    for a in [ax] + ([ax2] if ax2 is not None else []):
        if s.minor_ticks:
            a.minorticks_on()
        else:
            a.minorticks_off()
    # grid only on the primary axes; recessive styling
    ax.grid(False, which="both")
    if s.grid_major:
        ax.grid(True, which="major", color="#b0b0b0", alpha=0.45, linewidth=0.6, linestyle="-")
    if s.grid_minor:
        if not s.minor_ticks:
            ax.minorticks_on()
        ax.grid(True, which="minor", color="#c8c8c8", alpha=0.35, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    for a in [ax] + ([ax2] if ax2 is not None else []):
        for sp in a.spines.values():
            sp.set_linewidth(s.axes_linewidth)

    _apply_legend(ax, ax2, s, family, legend_handles)
    _apply_mathtext(ax, ax2, s)


def _apply_mathtext(ax, ax2, s: PlotSettings) -> None:
    """Per-artist mathtext font set (no global rcParams change)."""
    fs = s.mathtext_fontset if s.mathtext_fontset in MATHTEXT_FONTSETS else "dejavusans"
    texts = []
    for a in [ax] + ([ax2] if ax2 is not None else []):
        texts += [a.title, a.xaxis.label, a.yaxis.label]
        leg = a.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
    for t in texts:
        try:
            t.set_math_fontfamily(fs)
        except (AttributeError, ValueError):  # pragma: no cover - old matplotlib
            pass


def padded_limits(lo: float, hi: float, scale: str = "linear", margin: float = 0.04):
    """Limits with a fractional margin (multiplicative on log axes)."""
    import math
    if scale == "log":
        if hi <= 0:
            return None
        lo = lo if lo > 0 else hi / 1e3
        a, b = math.log10(lo), math.log10(hi)
        d = (b - a) * margin or 0.5
        return 10 ** (a - d), 10 ** (b + d)
    d = (hi - lo) * margin
    if d == 0:
        d = abs(hi) * 0.05 or 1.0
    return lo - d, hi + d


def _apply_axis(a, which: str, axs: AxisSettings, lims="missing", autoscale: bool = True) -> None:
    set_scale = a.set_xscale if which == "x" else a.set_yscale
    get_scale = a.get_xscale if which == "x" else a.get_yscale
    if get_scale() != axs.scale:
        set_scale(axs.scale)
        autoscale = True
    if axs.auto_limits and not autoscale:
        pass
    elif axs.auto_limits and lims != "missing":
        pl = padded_limits(lims[0], lims[1], axs.scale, 0.02 if which == "x" else 0.05) if lims else None
        if pl is not None:
            (a.set_xlim if which == "x" else a.set_ylim)(*pl)
    elif axs.auto_limits:
        a.autoscale(True, axis=which)
        a.relim(visible_only=True)
        a.autoscale_view(scalex=which == "x", scaley=which == "y")
    else:
        lo, hi = sorted((axs.vmin, axs.vmax))
        if axs.scale == "log" and lo <= 0:
            lo = None
        (a.set_xlim if which == "x" else a.set_ylim)(lo, hi if hi != lo else None)
    inverted = a.xaxis_inverted() if which == "x" else a.yaxis_inverted()
    if inverted != axs.invert:
        (a.invert_xaxis if which == "x" else a.invert_yaxis)()


def _apply_legend(ax, ax2, s: PlotSettings, family: str, legend_handles: list | None = None) -> None:
    for a in (ax, ax2):
        if a is not None and a.get_legend() is not None:
            a.get_legend().remove()
    if not s.legend_on or s.legend_mode == "off":
        return
    handles, labels = [], []
    pool = legend_handles if legend_handles is not None else [
        h for a in (ax, ax2) if a is not None for h in a.get_lines()]
    for h in pool:
        lab = h.get_label()
        if lab and not lab.startswith("_") and h.get_visible():
            handles.append(h)
            labels.append(lab)
    if not handles:
        return
    host = ax2 if ax2 is not None else ax
    ncol = max(1, int(s.legend_ncol))
    if ncol == 1 and len(handles) > 15 and s.legend_mode == "full":  # e.g. 50 fatigue cycles: wrap instead of overflowing
        ncol = -(-len(handles) // 15)
    kw = dict(ncols=ncol, frameon=s.legend_frame,
              prop={"size": s.legend_size, "family": family})
    if s.legend_frame:
        kw.update(framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    if s.legend_loc == "outside right":
        kw.update(loc="center left", bbox_to_anchor=(1.12 if ax2 is not None else 1.02, 0.5))
    else:
        kw["loc"] = s.legend_loc
    leg = host.legend(handles, labels, **kw)
    leg.set_draggable(True)
    # a legend inside the axes must not shrink the axes in constrained layout
    leg.set_in_layout(s.legend_loc == "outside right")
