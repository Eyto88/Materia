"""Colour palettes and colormap sampling for the Plot tab (module B, Qt-free).

* :data:`PALETTES` – categorical palettes (colour-blind-safe default, Tableau 10,
  Okabe–Ito, greys).
* :data:`COLORMAPS` – sequential / diverging matplotlib colormaps offered for
  "single-hue sequential" palettes and colour-by-value.
* :func:`palette_colors` – N colours from a palette name or a colormap name.
"""
from __future__ import annotations

import numpy as np

from .plot_style import COLOR_CYCLE

PALETTES: dict[str, list[str]] = {
    "Colour-blind safe (default)": list(COLOR_CYCLE),
    "Okabe–Ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"],
    "Tableau 10": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
                   "#9C755F", "#BAB0AC"],
    "Greys": ["#000000", "#555555", "#888888", "#AAAAAA", "#333333", "#777777"],
}

#: colormap name -> (lo, hi) fraction of the map used (avoids near-white ends on white paper)
COLORMAPS: dict[str, tuple[float, float]] = {
    "viridis": (0.0, 0.9), "plasma": (0.0, 0.88), "cividis": (0.0, 0.92), "inferno": (0.0, 0.85),
    "magma": (0.0, 0.85), "coolwarm": (0.0, 1.0), "RdBu_r": (0.0, 1.0), "Blues": (0.35, 1.0),
    "Reds": (0.35, 1.0), "Greens": (0.35, 1.0), "Greys": (0.35, 1.0), "turbo": (0.05, 0.95),
}


def colormap_colors(cmap: str, n: int, lo: float | None = None, hi: float | None = None) -> list[str]:
    """``n`` evenly spaced hex colours from matplotlib colormap ``cmap``."""
    import matplotlib
    from matplotlib.colors import to_hex
    cm = matplotlib.colormaps[cmap]
    dlo, dhi = COLORMAPS.get(cmap, (0.0, 1.0))
    lo = dlo if lo is None else lo
    hi = dhi if hi is None else hi
    n = max(1, int(n))
    pos = np.array([0.5 * (lo + hi)]) if n == 1 else np.linspace(lo, hi, n)
    return [to_hex(cm(float(p))) for p in pos]


def palette_colors(name: str, n: int | None = None) -> list[str]:
    """Colours of palette ``name`` (or ``n`` steps of colormap ``name``).

    For a categorical palette and ``n`` larger than the palette, the palette repeats.
    """
    if name in PALETTES:
        cols = list(PALETTES[name])
        if n:
            cols = [cols[i % len(cols)] for i in range(int(n))]
        return cols
    try:
        return colormap_colors(name, n or 7)
    except KeyError as e:
        raise ValueError(f"Unknown palette or colormap '{name}'") from e


def palette_names() -> list[str]:
    """Palette combo entries: categorical palettes first, then colormaps."""
    return list(PALETTES) + list(COLORMAPS)
