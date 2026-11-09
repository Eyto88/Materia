"""Materia main window: integrates import, plotting, analysis and micrograph modules.

v1.1: sessions (.lstudio), preferences, logging / problem report, responsive sizing, recent files.
"""
from __future__ import annotations

import logging
import os
import traceback
import zipfile

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QDialog, QDockWidget, QFileDialog, QInputDialog, QMainWindow, QMessageBox,
    QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget,
)

from .. import APP_NAME, __version__
from ..core.dataset import DatasetStore
from ..core.logsetup import environment_report, log_path, setup_logging
from ..core import session as session_io

from ..core.data_io import DATA_EXTENSIONS  # noqa: E402
SESSION_FILTER = f"Materia session (*{session_io.EXTENSION})"
MAX_RECENT = 8

log = logging.getLogger("labstudio.main")


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        setup_logging()
        self.setAcceptDrops(True)
        home = os.environ.get("LABSTUDIO_HOME")
        self.settings = (QSettings(os.path.join(home, "window.ini"), QSettings.IniFormat) if home
                         else QSettings("LabStudio", "LabStudio"))
        self.store = DatasetStore()
        self.session_path: str | None = None
        self._dirty = False
        self._restoring = False

        from .plot_widget import PlotPanel
        from .analysis_panel import AnalysisPanel
        from .image_workspace import ImageWorkspace

        self.tabs = QTabWidget(self)
        self.tabs.setDocumentMode(True)
        self.plot_panel = PlotPanel(self.store, self)
        self.image_ws = ImageWorkspace(self)
        self.tabs.addTab(self.plot_panel, "Plot && Analyse")
        self.tabs.addTab(self.image_ws, "Micrographs (SEM / LOM)")
        self.setCentralWidget(self.tabs)

        self.analysis_panel = AnalysisPanel(self.plot_panel, self.store, self)
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setObjectName("AnalysisDock")
        self.analysis_dock.setWidget(self.analysis_panel)
        self.analysis_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.analysis_dock)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        for src in (self.image_ws, self.plot_panel, self.analysis_panel):
            sig = getattr(src, "statusMessage", None)
            if sig is not None:
                sig.connect(lambda m: self.statusBar().showMessage(m, 6000))
        try:  # presets can store the analysis panel's current parameters
            from .import_dialog import ImportDialog
            ImportDialog.set_default_analysis_defaults_provider(self.analysis_panel.current_defaults)
        except Exception:
            log.exception("analysis defaults provider")
        self.store.subscribe(self._mark_dirty)
        self._build_menus()
        self._update_title()
        self.statusBar().showMessage("Drop CSV/TXT/XLSX data, micrographs or a .lstudio session onto the window, "
                                     "or use File ▸ Import.")
        self._initial_geometry()
        self._compact_if_small()

    def _compact_if_small(self) -> None:
        """On laptop-size windows start with the customise panel collapsed so the plot gets the room."""
        if self.width() < 1500 and hasattr(self, "act_right_scroll"):
            self.act_right_scroll.setChecked(False)
            self.statusBar().showMessage("Small screen: Customise panel hidden — View ▸ Customise panel to show it.",
                                         8000)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if not getattr(self, "_compact_checked", False) and self.isVisible():
            self._compact_checked = True
            self._compact_if_small()

    # ------------------------------------------------------------------ geometry
    def _initial_geometry(self) -> None:
        self.setMinimumSize(800, 500)
        geo = self.settings.value("geometry")
        if geo is not None and self.restoreGeometry(geo):
            return
        scr = QGuiApplication.primaryScreen()
        if scr is not None:
            av = scr.availableGeometry()
            w, h = min(1600, int(av.width() * 0.92)), min(950, int(av.height() * 0.9))
            self.resize(w, h)
            self.move(av.x() + (av.width() - w) // 2, av.y() + (av.height() - h) // 2)
        else:  # pragma: no cover
            self.resize(1280, 760)

    # ------------------------------------------------------------------ menus
    def _act(self, menu, text, slot, shortcut=None, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setStatusTip(tip)
            a.setToolTip(tip)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        self._act(f, "New session", self.new_session, "Ctrl+N", "Close everything and start an empty session")
        self._act(f, "Open session…", self.open_session_dialog, "Ctrl+Alt+O",
                  "Open a saved .lstudio session (datasets, figures, analyses, images)")
        self.recent_menu = f.addMenu("Open recent")
        self._act(f, "Save session", self.save_session, "Ctrl+S")
        self._act(f, "Save session as…", self.save_session_as, "Ctrl+Shift+S")
        f.addSeparator()
        self._act(f, "Import data (CSV/TXT/XLSX)…", self.import_data_dialog, "Ctrl+O",
                  "Import DSC / tensile / compression / cyclic test exports")
        self._act(f, "Open micrograph…", self.open_image_dialog, "Ctrl+Shift+O",
                  "Open SEM / optical micrograph (TIFF, PNG, JPG, BMP)")
        f.addSeparator()
        self._act(f, "Export plot…", self.export_plot, "Ctrl+E")
        self._act(f, "Export dataset as CSV…", self.export_dataset)
        f.addSeparator()
        self._act(f, "Load sample NiTi data", self.load_samples,
                  tip="Load the synthetic NiTi demo files to try every feature")
        f.addSeparator()
        self._act(f, "Preferences…", self.show_preferences, "Ctrl+,")
        f.addSeparator()
        self._act(f, "Quit", self.close, "Ctrl+Q")
        self._rebuild_recent()

        v = mb.addMenu("&View")
        v.addAction(self.analysis_dock.toggleViewAction())
        self._act(v, "Plot && Analyse tab", lambda: self.tabs.setCurrentIndex(0), "Ctrl+1")
        self._act(v, "Micrographs tab", lambda: self.tabs.setCurrentIndex(1), "Ctrl+2")
        self._act(v, "Fit window to screen", self.fit_to_screen)
        v.addSeparator()
        for label, attr in (("Trace manager panel", "left_scroll"), ("Customise panel", "right_scroll")):
            w = getattr(self.plot_panel, attr, None)
            if w is None:
                continue
            a = QAction(label, self, checkable=True)
            a.setChecked(True)
            a.toggled.connect(w.setVisible)
            v.addAction(a)
            setattr(self, f"act_{attr}", a)

        h = mb.addMenu("&Help")
        self._act(h, "Quick tutorial (PDF)", self.open_tutorial, "F1")
        self._act(h, "Check for updates…", self.check_updates)
        self._act(h, "Restore previous version…", self.restore_previous_version)
        h.addSeparator()
        self._act(h, "Show log", self.show_log)
        self._act(h, "Report a problem…", self.report_problem,
                  tip="Save a zip with the log and environment details to send with a bug report")
        h.addSeparator()
        self._act(h, f"About {APP_NAME}", self.about)

    def _on_tab_changed(self, idx: int) -> None:
        self.analysis_dock.setVisible(idx == 0)

    # ------------------------------------------------------------------ dirty / title
    def _mark_dirty(self) -> None:
        if not self._restoring:
            self._dirty = True
            self._update_title()

    def _update_title(self) -> None:
        name = os.path.basename(self.session_path) if self.session_path else "Untitled session"
        self.setWindowTitle(f"{'• ' if self._dirty else ''}{name} — {APP_NAME} {__version__}")

    def _panels(self) -> dict:
        return {"plot": self.plot_panel, "analysis": self.analysis_panel, "images": self.image_ws}

    # ------------------------------------------------------------------ sessions
    def new_session(self) -> None:
        if not self._confirm_discard():
            return
        self._restoring = True
        try:
            for ds in list(self.store):
                self.store.remove(ds)
            try:
                self.plot_panel.clear()
            except Exception:
                log.exception("clear plot")
            for fn in ("close_all", "clear_all"):
                if hasattr(self.image_ws, fn):
                    getattr(self.image_ws, fn)()
                    break
        finally:
            self._restoring = False
        self.session_path, self._dirty = None, False
        self._update_title()

    def save_session(self) -> bool:
        if not self.session_path:
            return self.save_session_as()
        return self._write_session(self.session_path)

    def save_session_as(self) -> bool:
        start = self.session_path or os.path.join(self.settings.value("last_session_dir", os.path.expanduser("~")),
                                                  "session" + session_io.EXTENSION)
        path, _ = QFileDialog.getSaveFileName(self, "Save session", start, SESSION_FILTER)
        if not path:
            return False
        if not path.endswith(session_io.EXTENSION):
            path += session_io.EXTENSION
        return self._write_session(path)

    def _write_session(self, path: str) -> bool:
        try:
            session_io.save_session(path, self.store, self._panels())
        except Exception as exc:
            log.exception("save session")
            self._error("Could not save session", exc)
            return False
        self.session_path, self._dirty = path, False
        self.settings.setValue("last_session_dir", os.path.dirname(path))
        self._add_recent(path)
        self._update_title()
        self.statusBar().showMessage(f"Session saved: {path}", 5000)
        return True

    def open_session_dialog(self) -> None:
        if not self._confirm_discard():
            return
        start = self.settings.value("last_session_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Open session", start, SESSION_FILTER + ";;All files (*)")
        if path:
            self.open_session(path, confirm=False)

    def open_session(self, path: str, confirm: bool = True) -> bool:
        if confirm and not self._confirm_discard():
            return False
        self._restoring = True
        try:
            try:
                self.plot_panel.clear()
            except Exception:
                log.exception("clear plot")
            session_io.load_session(path, self.store, self._panels())
        except Exception as exc:
            log.exception("open session")
            self._error("Could not open session", exc)
            return False
        finally:
            self._restoring = False
        self.session_path, self._dirty = path, False
        self.settings.setValue("last_session_dir", os.path.dirname(path))
        self._add_recent(path)
        self._update_title()
        self.statusBar().showMessage(f"Opened session {os.path.basename(path)} ({len(self.store)} datasets)", 6000)
        return True

    def _confirm_discard(self) -> bool:
        if not self._dirty or not len(self.store) or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save the current session first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Save:
            return self.save_session()
        return True

    # ------------------------------------------------------------------ recent files
    def _recent(self) -> list[str]:
        v = self.settings.value("recent_sessions", []) or []
        if isinstance(v, str):
            v = [v]
        return [p for p in v if os.path.isfile(p)]

    def _add_recent(self, path: str) -> None:
        lst = [path] + [p for p in self._recent() if p != path]
        self.settings.setValue("recent_sessions", lst[:MAX_RECENT])
        self._rebuild_recent()

    def _rebuild_recent(self) -> None:
        self.recent_menu.clear()
        rec = self._recent()
        if not rec:
            a = self.recent_menu.addAction("(none)")
            a.setEnabled(False)
        for p in rec:
            a = self.recent_menu.addAction(os.path.basename(p))
            a.setToolTip(p)
            a.triggered.connect(lambda _=False, q=p: self.open_session(q))

    # ------------------------------------------------------------------ data
    def import_data_dialog(self) -> None:
        start = self.settings.value("last_data_dir", os.path.expanduser("~"))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import data files", start,
            "Data files (" + " ".join("*" + e for e in DATA_EXTENSIONS) + ");;All files (*)")
        if paths:
            self.settings.setValue("last_data_dir", os.path.dirname(paths[0]))
            self.import_files(paths)

    def import_files(self, paths: list[str]) -> list:
        from .import_dialog import ImportDialog
        try:
            datasets = ImportDialog.import_files(paths, self)
        except Exception as exc:  # pragma: no cover - surfaced to user
            log.exception("import")
            self._error("Import failed", exc)
            return []
        for ds in datasets:
            self.store.add(ds)
            if ds.default_x and ds.default_y:
                self.plot_panel.add_trace(ds, ds.default_x, ds.default_y)
            for w in (ds.metadata or {}).get("warnings", []) or []:
                log.warning("%s: %s", ds.name, w)
        if datasets:
            self.tabs.setCurrentIndex(0)
            n_warn = sum(len((d.metadata or {}).get("warnings", []) or []) for d in datasets)
            msg = f"Imported {len(datasets)} dataset(s)."
            if n_warn:
                msg += f"  ⚠ {n_warn} import warning(s) — check units / heating rate."
            self.statusBar().showMessage(msg, 8000)
        return datasets

    def _app_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def load_samples(self) -> None:
        from ..core.data_io import load_file
        root = os.path.join(self._app_root(), "sample_data")
        if not os.path.isdir(root):
            QMessageBox.information(self, "Samples", "sample_data folder not found.")
            return
        n = 0
        for fn in sorted(os.listdir(root)):
            p = os.path.join(root, fn)
            if fn.lower().endswith(DATA_EXTENSIONS):
                try:
                    self.store.add(load_file(p))
                    n += 1
                except Exception as exc:
                    log.exception("sample %s", fn)
                    self._error(f"Could not load {fn}", exc)
            elif self._is_image(p):
                self.image_ws.open_image(p)
        self.statusBar().showMessage(f"Loaded {n} sample dataset(s). Pick X/Y in the trace manager.", 8000)

    def export_dataset(self) -> None:
        from ..core.data_io import export_dataset_csv
        items = list(self.store)
        if not items:
            QMessageBox.information(self, "Export", "No datasets loaded.")
            return
        names = [f"{d.id}: {d.name}" for d in items]
        choice, ok = QInputDialog.getItem(self, "Export dataset", "Dataset:", names, 0, False)
        if not ok:
            return
        ds = items[names.index(choice)]
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{ds.name}.csv", "CSV (*.csv)")
        if path:
            export_dataset_csv(ds, path)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def export_plot(self) -> None:
        if hasattr(self.plot_panel, "open_export_dialog"):
            self.plot_panel.open_export_dialog()
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export plot", "plot.png", "PNG (*.png)")
        if path:
            self.plot_panel.export_png(path, dpi=300)

    # ------------------------------------------------------------------ images
    def _is_image(self, path: str) -> bool:
        from .image_workspace import IMAGE_EXTENSIONS
        return path.lower().endswith(tuple(IMAGE_EXTENSIONS))

    def open_image_dialog(self) -> None:
        self.tabs.setCurrentIndex(1)
        self.image_ws.open_dialog()

    # ------------------------------------------------------------------ drag & drop
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        self.open_paths(paths)

    def open_paths(self, paths: list[str]) -> None:
        sessions = [p for p in paths if p.lower().endswith(session_io.EXTENSION)]
        if sessions:
            self.open_session(sessions[0])
        data = [p for p in paths if p.lower().endswith(DATA_EXTENSIONS)]
        imgs = [p for p in paths if self._is_image(p)]
        other = [p for p in paths if p not in sessions + data + imgs]
        for p in imgs:
            self.image_ws.open_image(p)
        if imgs and not data:
            self.tabs.setCurrentIndex(1)
        if data:
            self.import_files(data)
        if other:
            QMessageBox.warning(self, "Unsupported files",
                                "These files were not opened (unsupported type):\n" +
                                "\n".join(os.path.basename(p) for p in other))

    # ------------------------------------------------------------------ misc
    def show_preferences(self) -> None:
        from .preferences import PreferencesDialog
        dlg = PreferencesDialog(self)
        if dlg.exec():
            for p in self._panels().values():
                if hasattr(p, "refresh_formatting"):
                    p.refresh_formatting()

    def fit_to_screen(self) -> None:
        scr = self.screen() or QGuiApplication.primaryScreen()
        if scr is not None:
            av = scr.availableGeometry()
            self.setGeometry(av.adjusted(10, 10, -10, -10))

    def open_tutorial(self) -> None:
        pdf = os.path.join(self._app_root(), "Materia_Quick_Tutorial.pdf")
        if os.path.isfile(pdf):
            QDesktopServices.openUrl(QUrl.fromLocalFile(pdf))
        else:
            QMessageBox.information(self, "Tutorial", "Materia_Quick_Tutorial.pdf not found next to the app.")

    # ------------------------------------------------------------------ updates
    def check_updates(self, *, quiet: bool = False):
        """Ask GitHub for a newer release; offer to install it (only changed files; settings kept)."""
        from ..core import updater as U
        from .worker import run_task
        root = self._app_root()
        if not U.read_config(root).get("repo"):
            if not quiet:
                QMessageBox.information(self, "Updates", "Updates are not configured for this copy "
                                        "(update.json has no repository).")
            return None
        result = {}

        def done(info):
            result["info"] = info
            if info is None:
                if not quiet:
                    QMessageBox.information(self, "Updates",
                                            f"You have the latest version ({U.local_version(root)}).")
                return
            self._offer_update(info)

        def failed(tb, exc):
            log.warning("update check failed: %s", exc)
            if not quiet:
                QMessageBox.warning(self, "Updates", f"Could not check for updates:\n{exc}")

        run_task(self, "Checking for updates…", lambda: U.check_for_update(root, timeout=8), done, failed)
        return result.get("info")

    def _offer_update(self, info) -> None:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        box = QMessageBox(self)
        box.setWindowTitle("Update available")
        box.setIcon(QMessageBox.Information)
        box.setText(f"<b>Materia {info.version}</b> is available (you have {info.current}).")
        box.setInformativeText("Only changed files are replaced. Your settings, presets, styles and sessions "
                               "are kept. Materia restarts after updating.")
        box.setDetailedText(info.notes or "(no release notes)")
        upd = box.addButton("Update and restart", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is upd:
            self.install_update(info)

    def install_update(self, info) -> bool:
        from ..core import updater as U
        from .worker import run_task
        if not self._confirm_discard():
            return False
        root = self._app_root()
        res = run_task(self, f"Downloading and installing Materia {info.version}…",
                       lambda: U.apply_update(root, info), on_done=lambda r: self._after_update(r))
        return res is not None

    def _after_update(self, res) -> None:
        import subprocess
        import sys
        log.info("updated %s -> %s (%d written, %d removed)", res.old_version, res.new_version,
                 len(res.written), len(res.removed))
        launcher = os.path.join(self._app_root(), "launcher.pyw")
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        QMessageBox.information(self, "Updated",
                                f"Materia was updated to {res.new_version} ({len(res.written)} files changed). "
                                "It will now restart.")
        if os.path.isfile(launcher):
            flags = 0x08000000 if os.name == "nt" else 0
            subprocess.Popen([sys.executable, launcher, "--no-update"], cwd=self._app_root(), creationflags=flags)
        self._dirty = False
        self.close()

    def restore_previous_version(self) -> None:
        from ..core import updater as U
        root = self._app_root()
        vers = U.list_backups(root)
        if not vers:
            QMessageBox.information(self, "Restore previous version", "No earlier version is stored.")
            return
        v, ok = QInputDialog.getItem(self, "Restore previous version",
                                     "Restore the files of version:", list(reversed(vers)), 0, False)
        if ok and v:
            U.rollback(root, v)
            QMessageBox.information(self, "Restored", f"Version {v} restored. Please restart Materia.")

    def show_log(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Materia log")
        dlg.resize(900, 500)
        lay = QVBoxLayout(dlg)
        te = QPlainTextEdit()
        te.setReadOnly(True)
        try:
            with open(log_path(), encoding="utf-8", errors="replace") as fh:
                te.setPlainText(fh.read()[-200_000:])
        except OSError:
            te.setPlainText("(no log yet)")
        te.moveCursor(te.textCursor().End)
        lay.addWidget(te)
        dlg.exec()

    def report_problem(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save problem report", "labstudio_report.zip", "Zip (*.zip)")
        if not path:
            return None
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("environment.txt", environment_report())
            if os.path.isfile(log_path()):
                z.write(log_path(), "labstudio.log")
        self.statusBar().showMessage(f"Problem report saved: {path}", 6000)
        return path

    def _error(self, title: str, exc: Exception) -> None:
        log.error("%s: %s", title, exc)
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        QMessageBox.critical(self, title, f"{exc}\n\n{traceback.format_exc(limit=3)}")

    def about(self) -> None:
        QMessageBox.about(self, f"About {APP_NAME}",
                          f"<b>{APP_NAME} {__version__}</b><br>Data plotting, DSC / mechanical / cyclic / elastocaloric "
                          "analysis and ImageJ-style micrograph analysis for shape memory alloy research."
                          "<br><br>All values are stored in canonical units (N, mm, MPa, %, °C, s, mW, W/g)."
                          f"<br>Log file: {log_path()}")

    def closeEvent(self, e) -> None:
        if not self._confirm_discard():
            e.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(e)


RESOURCES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")


def app_icon():
    from PySide6.QtGui import QIcon
    icon = QIcon()
    for fn in ("materia.ico", "materia_256.png"):
        p = os.path.join(RESOURCES, fn)
        if os.path.isfile(p):
            icon.addFile(p)
    return icon


def _set_windows_app_id() -> None:
    """Give Materia its own taskbar identity (otherwise Windows groups it under python.exe's icon)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Materia.App")
    except Exception:
        pass


def _signal_ready() -> None:
    """Tell the launcher's loading screen that the main window is up."""
    path = os.environ.get("MATERIA_READY_FILE")
    if path:
        try:
            with open(path, "w") as fh:
                fh.write(str(os.getpid()))
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    import sys
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QSplashScreen
    argv = argv or sys.argv
    _set_windows_app_id()
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setStyle("Fusion")
    app.setWindowIcon(app_icon())
    splash = None
    if not os.environ.get("MATERIA_LAUNCHER"):  # started directly: show our own splash while loading
        pm = QPixmap(os.path.join(RESOURCES, "splash.png"))
        if not pm.isNull():
            splash = QSplashScreen(pm)
            splash.show()
            splash.showMessage(f"Loading {APP_NAME}…", Qt.AlignBottom | Qt.AlignHCenter, Qt.white)
            app.processEvents()
    win = MainWindow()
    win.setWindowIcon(app_icon())
    win.show()
    if splash is not None:
        splash.finish(win)
    QTimer.singleShot(0, _signal_ready)
    files = [p for p in argv[1:] if os.path.isfile(p)]
    if files:
        QTimer.singleShot(0, lambda: win.open_paths(files))
    return app.exec()
