"""Application-wide user settings (persisted with QSettings) + value formatting.

    from .app_settings import settings, format_value
    format_value(20.069457240309497)  -> '20.07'   (significant figures from Preferences)
    settings().sig_figs / settings().set("sig_figs", 5)
Listeners: settings().changed (Qt signal, emits the key that changed).
In tests set LABSTUDIO_HOME to a temp dir to isolate (QSettings stored as INI there).
"""
from __future__ import annotations

import math
import os

from PySide6.QtCore import QObject, QSettings, Signal

DEFAULTS = {
    "sig_figs": 5,            # significant figures for copy / CSV / tables
    "csv_full_precision": False,  # if True, CSV export ignores sig_figs
    "default_style_template": "",  # name of user plot style template applied to new figures
    "font_family": "",        # preferred figure font ('' = preset default)
}


class AppSettings(QObject):
    changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        home = os.environ.get("LABSTUDIO_HOME")
        if home:
            os.makedirs(home, exist_ok=True)
            self._qs = QSettings(os.path.join(home, "settings.ini"), QSettings.IniFormat)
        else:
            self._qs = QSettings("LabStudio", "LabStudio")

    def get(self, key: str):
        default = DEFAULTS.get(key)
        v = self._qs.value(key, default)
        if isinstance(default, bool):
            return v in (True, "true", "True", 1, "1")
        if isinstance(default, int):
            try:
                return int(v)
            except (TypeError, ValueError):
                return default
        return v

    def set(self, key: str, value) -> None:
        self._qs.setValue(key, value)
        self.changed.emit(key)

    @property
    def sig_figs(self) -> int:
        return max(1, min(15, self.get("sig_figs")))


_inst: AppSettings | None = None


def settings() -> AppSettings:
    global _inst
    if _inst is None:
        _inst = AppSettings()
    return _inst


def format_value(v, sig: int | None = None) -> str:
    """Format a number to N significant figures (no scientific notation for 1e-4..1e6)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if isinstance(v, (bool,)):
        return str(v)
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) < 1e15:
        return str(v)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    sig = sig or settings().sig_figs
    if x == 0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    if -4 <= mag < 6:
        decimals = max(0, sig - 1 - mag)
        s = f"{x:.{decimals}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return f"{x:.{sig - 1}e}"
