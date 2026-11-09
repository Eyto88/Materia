"""Self-updater: only changed release files are touched; user data, .venv and own files survive."""
import io
import json
import os
import zipfile

import pytest

from labstudio.core import updater as U


def make_app(root, version, files):
    os.makedirs(root, exist_ok=True)
    files = dict(files)
    files["labstudio/__init__.py"] = f'__version__ = "{version}"\nAPP_NAME = "Materia"\n'
    files.setdefault("launcher.pyw", "print('launcher')\n")
    files.setdefault("requirements.txt", "numpy>=1.26\n")
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
    return root


def release_zip(src_root):
    """Same layout as tools/make_release.py: Materia/<files> + MANIFEST.json (+ update.json)."""
    man = U.build_manifest(src_root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for rel in man["files"]:
            z.write(os.path.join(src_root, rel), "Materia/" + rel)
        z.writestr("Materia/MANIFEST.json", json.dumps(man))
        z.writestr("Materia/update.json", json.dumps({"repo": "someone/else"}))
    return buf.getvalue(), man


@pytest.fixture()
def installed(tmp_path, monkeypatch):
    monkeypatch.setenv("LABSTUDIO_HOME", str(tmp_path / "userhome"))
    app = make_app(str(tmp_path / "Materia"), "1.2.0",
                   {"labstudio/a.py": "A = 1\n", "labstudio/old_only.py": "gone in 1.3\n",
                    "README.md": "readme 1.2\n"})
    U.write_manifest(app, U.build_manifest(app))
    with open(os.path.join(app, "update.json"), "w") as fh:
        json.dump({"repo": "Eyto88/materia"}, fh)
    # things an update must never touch
    os.makedirs(os.path.join(app, ".venv", "bin"))
    open(os.path.join(app, ".venv", "bin", "python"), "w").write("venv")
    open(os.path.join(app, "my_notes.txt"), "w").write("mine")
    open(os.path.join(app, "chapter3.lstudio"), "w").write("session")
    home = U.user_home()
    open(os.path.join(home, "settings.ini"), "w").write("sig_figs=4")
    os.makedirs(os.path.join(home, "presets"))
    open(os.path.join(home, "presets", "instron.json"), "w").write("{}")
    return app


def new_release(tmp_path, requirements="numpy>=1.26\n"):
    return make_app(str(tmp_path / "build13"), "1.3.0",
                    {"labstudio/a.py": "A = 2  # changed\n", "labstudio/new_feature.py": "NEW = True\n",
                     "README.md": "readme 1.2\n", "requirements.txt": requirements})


def test_versions():
    assert U.is_newer("1.3.0", "1.2.0") and U.is_newer("v1.10", "1.9.9") and not U.is_newer("1.2", "1.2.0")


def test_apply_only_changes_and_keeps_user_data(installed, tmp_path):
    data, _ = release_zip(new_release(tmp_path))
    with __import__("tempfile").TemporaryDirectory() as tmp:
        res = U.apply_update_from_folder(installed, U.extract_release(data, tmp))
    assert res.old_version == "1.2.0" and res.new_version == "1.3.0"
    assert set(res.written) == {"labstudio/__init__.py", "labstudio/a.py", "labstudio/new_feature.py"}
    assert res.removed == ["labstudio/old_only.py"]
    assert res.unchanged >= 3 and not res.requirements_changed
    rd = lambda rel: open(os.path.join(installed, rel), encoding="utf-8").read()
    assert "changed" in rd("labstudio/a.py") and rd("labstudio/new_feature.py") == "NEW = True\n"
    assert not os.path.exists(os.path.join(installed, "labstudio/old_only.py"))
    assert U.local_version(installed) == "1.3.0"
    # untouched
    assert rd("my_notes.txt") == "mine" and rd("chapter3.lstudio") == "session"
    assert rd(".venv/bin/python") == "venv"
    assert json.loads(rd("update.json"))["repo"] == "Eyto88/materia"  # our config kept, not the zip's
    home = U.user_home()
    assert open(os.path.join(home, "settings.ini")).read() == "sig_figs=4"
    assert os.path.isfile(os.path.join(home, "presets", "instron.json"))


def test_requirements_change_detected(installed, tmp_path):
    data, _ = release_zip(new_release(tmp_path, requirements="numpy>=1.26\nopenpyxl>=3.1\n"))
    with __import__("tempfile").TemporaryDirectory() as tmp:
        res = U.apply_update_from_folder(installed, U.extract_release(data, tmp))
    assert res.requirements_changed


def test_rollback_restores_previous_version(installed, tmp_path):
    data, _ = release_zip(new_release(tmp_path))
    with __import__("tempfile").TemporaryDirectory() as tmp:
        U.apply_update_from_folder(installed, U.extract_release(data, tmp))
    assert U.list_backups(installed) == ["1.2.0"]
    U.rollback(installed)
    rd = lambda rel: open(os.path.join(installed, rel), encoding="utf-8").read()
    assert U.local_version(installed) == "1.2.0"
    assert rd("labstudio/a.py") == "A = 1\n" and rd("labstudio/old_only.py") == "gone in 1.3\n"
    assert not os.path.exists(os.path.join(installed, "labstudio/new_feature.py"))
    assert rd("my_notes.txt") == "mine"


def test_corrupt_update_rejected_and_nothing_changed(installed, tmp_path):
    src = new_release(tmp_path)
    data, man = release_zip(src)
    z = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for n in z.namelist():
            body = z.read(n)
            if n.endswith("labstudio/a.py"):
                body = b"tampered"
            out.writestr(n, body)
    with __import__("tempfile").TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="corrupt"):
            U.apply_update_from_folder(installed, U.extract_release(buf.getvalue(), tmp))
    assert U.local_version(installed) == "1.2.0"


def test_zip_slip_and_protected_paths_rejected(installed, tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="Unsafe"):
        U.extract_release(buf.getvalue(), str(tmp_path / "x"))
    src = new_release(tmp_path)
    man = U.build_manifest(src)
    man["files"][".venv/bin/python"] = "0" * 64
    U.write_manifest(src, man)
    with pytest.raises(ValueError, match="protected"):
        U.verify_release(src)


def test_check_and_apply_via_github_api(installed, tmp_path, monkeypatch):
    data, _ = release_zip(new_release(tmp_path))
    rel = {"tag_name": "v1.3.0", "body": "- Faster DSC\n- New eCaloric plot", "published_at": "2026-11-20",
           "assets": [{"name": "Materia_v1.3.0.zip", "browser_download_url": "https://example/zip",
                       "url": "https://api.github.com/x", "size": len(data)}]}
    monkeypatch.setattr(U, "_request", lambda url, **k: json.dumps(rel).encode())

    class Resp(io.BytesIO):
        headers = {"Content-Length": str(len(data))}
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(U.urllib.request, "urlopen", lambda req, **k: Resp(data))
    info = U.check_for_update(installed)
    assert info and info.version == "1.3.0" and info.current == "1.2.0" and "eCaloric" in info.notes
    seen = []
    res = U.apply_update(installed, info, progress=lambda d, t: seen.append((d, t)))
    assert res.new_version == "1.3.0" and seen and seen[-1][0] == len(data)
    assert U.check_for_update(installed) is None  # now up to date


def test_skip_version(installed, monkeypatch):
    U.skip_version("1.3.0")
    assert U.read_skip() == "1.3.0"


def test_no_repo_configured(tmp_path):
    app = make_app(str(tmp_path / "A"), "1.0.0", {})
    assert U.check_for_update(app) is None


def test_real_tree_manifest_excludes_user_and_build_files():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    man = U.build_manifest(root)
    files = man["files"]
    assert "launcher.pyw" in files and "labstudio/core/updater.py" in files
    assert not any(f.startswith((".venv/", ".git/", "dist/")) or "__pycache__" in f or f == "update.json"
                   for f in files)
