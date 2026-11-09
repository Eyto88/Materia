"""Logging to a rotating file in the LabStudio home folder (Help ▸ Show log / Report a problem)."""
from __future__ import annotations

import logging
import os
import platform
import sys
from logging.handlers import RotatingFileHandler

from .. import __version__


def home_dir() -> str:
    d = os.environ.get("LABSTUDIO_HOME") or os.path.join(os.path.expanduser("~"), ".labstudio")
    os.makedirs(d, exist_ok=True)
    return d


def log_path() -> str:
    return os.path.join(home_dir(), "labstudio.log")


_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    global _configured
    root = logging.getLogger("labstudio")
    if _configured:
        return root
    root.setLevel(level)
    try:
        fh = RotatingFileHandler(log_path(), maxBytes=2_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)
    except OSError:  # pragma: no cover - read-only home
        pass
    root.info("LabStudio %s started (Python %s, %s)", __version__, platform.python_version(), platform.platform())

    def excepthook(t, v, tb):  # log uncaught exceptions instead of losing them
        root.error("Uncaught exception", exc_info=(t, v, tb))
        sys.__excepthook__(t, v, tb)

    sys.excepthook = excepthook
    _configured = True
    return root


def environment_report() -> str:
    lines = [f"LabStudio {__version__}", f"Python {platform.python_version()} ({sys.executable})",
             f"Platform {platform.platform()}"]
    for mod in ("numpy", "scipy", "pandas", "matplotlib", "PySide6", "skimage", "tifffile", "PIL", "openpyxl"):
        try:
            m = __import__(mod)
            lines.append(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception:
            lines.append(f"{mod} (not installed)")
    return "\n".join(lines)
