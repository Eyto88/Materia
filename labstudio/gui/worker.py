"""Run long computations off the GUI thread with a busy indicator.

Usage (from any panel):
    from .worker import run_task
    run_task(self, "Analysing 1M points…", compute_fn, on_done=self._show, on_error=None)

compute_fn() must be pure (no Qt widget access). on_done(result) runs back on the GUI thread.
In tests (QT_QPA_PLATFORM=offscreen) or when LABSTUDIO_SYNC=1, or for parent=None, tasks run synchronously so
results are available immediately after run_task returns.
"""
from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, QTimer, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

log = logging.getLogger("labstudio.worker")


def _sync_mode() -> bool:
    return os.environ.get("LABSTUDIO_SYNC") == "1" or os.environ.get("QT_QPA_PLATFORM") == "offscreen"


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str, object)


class _Job(QRunnable):
    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self) -> None:  # pragma: no cover - thread
        try:
            res = self.fn()
        except Exception as exc:
            self.signals.failed.emit(traceback.format_exc(), exc)
        else:
            self.signals.done.emit(res)


_alive: set = set()


def default_error(parent: QWidget | None, tb: str, exc: Exception) -> None:
    log.error("Background task failed: %s", tb)
    if parent is not None and not _sync_mode():
        QMessageBox.warning(parent, "Task failed", f"{exc}\n\nSee Help ▸ Show log for details.")


def run_task(parent: QWidget | None, message: str, fn: Callable[[], Any],
             on_done: Callable[[Any], None] | None = None,
             on_error: Callable[[str, Exception], None] | None = None,
             *, min_duration_ms: int = 400) -> Any:
    """Run ``fn`` in a worker thread; show a busy dialog if it takes longer than ``min_duration_ms``.

    Returns the result immediately in sync mode, else None.
    """
    err = on_error or (lambda tb, e: default_error(parent, tb, e))
    if parent is None or _sync_mode():
        try:
            res = fn()
        except Exception as exc:
            err(traceback.format_exc(), exc)
            return None
        if on_done:
            on_done(res)
        return res

    dlg = QProgressDialog(message, None, 0, 0, parent)
    dlg.setWindowTitle("Working…")
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(min_duration_ms)
    dlg.setCancelButton(None)
    job = _Job(fn)
    _alive.add(job)

    def finish():
        dlg.reset()
        dlg.deleteLater()
        _alive.discard(job)

    def ok(res):
        finish()
        if on_done:
            on_done(res)

    def bad(tb, exc):
        finish()
        err(tb, exc)

    job.signals.done.connect(ok, Qt.QueuedConnection)
    job.signals.failed.connect(bad, Qt.QueuedConnection)
    QTimer.singleShot(0, lambda: QThreadPool.globalInstance().start(job))
    return None
