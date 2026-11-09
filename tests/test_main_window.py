"""Integration tests for the main window (sessions, preferences, problem report, sizing)."""
import os
import zipfile

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def win(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path / "home"))
    import labstudio.gui.app_settings as aps
    aps._inst = None
    from labstudio.gui.main_window import MainWindow
    w = MainWindow()
    w.resize(1280, 720)
    w.show()
    qapp.processEvents()
    yield w
    w._dirty = False
    w.close()
    aps._inst = None  # don't leak this test's settings (e.g. sig figs) into other tests


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_starts_and_loads_samples(win, qapp):
    win.load_samples()
    qapp.processEvents()
    kinds = {d.kind for d in win.store}
    assert {"dsc", "tensile", "cyclic", "compression"} <= kinds
    assert win._dirty


def test_fits_small_laptop(win):
    hint = win.minimumSizeHint()
    assert hint.width() <= 1280 and hint.height() <= 720, hint


def test_session_round_trip(win, qapp, tmp_path):
    win.load_samples()
    dsc = next(d for d in win.store if d.kind == "dsc")
    win.plot_panel.add_trace(dsc, dsc.default_x, dsc.default_y)
    n = len(win.store)
    p = str(tmp_path / "s.lstudio")
    assert win._write_session(p)
    assert not win._dirty and win.session_path == p
    win.new_session()
    assert len(win.store) == 0
    assert win.open_session(p, confirm=False)
    assert len(win.store) == n
    names = sorted(d.name for d in win.store)
    assert "NiTi_DSC_Netzsch" in names
    if hasattr(win.plot_panel, "get_state"):
        assert len(win.plot_panel.traces) >= 1


def test_open_paths_dispatch(win, qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a))
    bad = tmp_path / "x.docx"
    bad.write_bytes(b"x")
    img = os.path.join(ROOT, "sample_data", "NiTi_SEM_grains_synthetic.tif")
    win.open_paths([str(bad), img])
    assert shown, "unsupported file must be reported, not silently ignored"
    assert win.tabs.currentIndex() == 1


def test_problem_report(win, tmp_path):
    p = win.report_problem(str(tmp_path / "r.zip"))
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        assert "environment.txt" in names
        assert b"LabStudio" in z.read("environment.txt")


def test_preferences_sig_figs(win):
    from labstudio.gui.preferences import PreferencesDialog
    from labstudio.gui.app_settings import format_value, settings
    d = PreferencesDialog(win)
    d.sig.setValue(3)
    d.apply()
    assert settings().sig_figs == 3
    assert format_value(20.069457) == "20.1"
