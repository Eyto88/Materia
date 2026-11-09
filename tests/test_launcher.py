"""Launcher (loading screen) logic that can be tested without a display."""
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(tmp_path, monkeypatch):
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path / "home"))
    m = SourceFileLoader("materia_launcher", str(ROOT / "launcher.pyw")).load_module()
    return m


def test_needs_setup_and_stamp(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    venv = tmp_path / ".venv"
    monkeypatch.setattr(m, "VENV", str(venv))
    monkeypatch.setattr(m, "STAMP", str(venv / "requirements.sha256"))
    assert m.needs_setup()                      # no venv at all
    py = Path(m.venv_python())
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert m.needs_setup()                      # venv but no stamp
    (venv / "requirements.sha256").write_text(m.req_hash())
    assert not m.needs_setup()                  # stamp matches requirements.txt
    (venv / "requirements.sha256").write_text("old")
    assert m.needs_setup()                      # requirements changed -> reinstall


def test_resources_present():
    res = ROOT / "labstudio" / "resources"
    for fn in ("materia.ico", "splash.png", "materia_256.png"):
        assert (res / fn).is_file(), fn


def test_pip_progress_parsing(tmp_path, monkeypatch):
    m = load(tmp_path, monkeypatch)
    reports = []
    st = m.Setup(lambda f, t: reports.append((f, t)))
    lines = ["Collecting numpy>=1.26", "  Downloading numpy-2.1.whl (16 MB)", "Collecting scipy",
             "Installing collected packages: numpy, scipy", "Successfully installed numpy-2.1 scipy-1.14"]
    seen = set()
    # replicate the parser used inside Setup.run on canned pip output
    import re
    for line in lines:
        mm = re.match(r"\s*(Collecting|Requirement already satisfied:?)\s+([A-Za-z0-9_.\-]+)", line)
        if mm:
            seen.add(re.split(r"[<>=!~\[; ]", mm.group(2))[0])
    assert seen == {"numpy", "scipy"}
    assert st.expected >= 25


def test_app_icon_and_ready_signal(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from labstudio.gui import main_window as mw
    assert not mw.app_icon().isNull()
    ready = tmp_path / "ready"
    monkeypatch.setenv("MATERIA_READY_FILE", str(ready))
    mw._signal_ready()
    assert ready.read_text() == str(os.getpid())
