"""Materia self-updater (standard library only — the launcher uses it before any packages exist).

How it works
------------
* Every release zip contains ``MANIFEST.json``: version + the relative path and SHA-256 of every file the
  release ships.
* ``check_for_update()`` asks GitHub Releases (``update.json`` → ``"repo": "owner/name"``) for the latest
  release and compares versions.
* ``apply_update()`` downloads the release zip, verifies it, and then touches ONLY files that the release
  ships and that actually changed:
    - changed / new files are written,
    - files shipped by the OLD release but no longer by the new one are removed,
    - everything else is left alone: ``.venv``, your own files in the folder, and all user data, which lives
      outside the app folder anyway (``~/.labstudio``: settings, presets, styles, logs; sessions wherever you
      saved them; Windows registry settings).
  Replaced files are backed up to ``.update_backup/<old version>/`` so ``rollback()`` can restore them.
* Packages are reinstalled by the launcher only if ``requirements.txt`` changed (hash stamp).

Private repositories: put a GitHub token (read access to the repo) in ``~/.labstudio/github_token``.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import ssl
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field

MANIFEST = "MANIFEST.json"
CONFIG = "update.json"
RELEASE_INFO = "release.json"   # "repo" channel: announces the newest version on the main branch
BACKUP_DIR = ".update_backup"
# never shipped/overwritten/deleted by an update, even if a manifest lists them
PROTECTED_PREFIXES = (".venv/", ".git/", BACKUP_DIR + "/", "update.json")
# files that are never part of a release manifest
EXCLUDE_PARTS = {".venv", ".git", "__pycache__", ".pytest_cache", BACKUP_DIR, "dist"}
EXCLUDE_SUFFIXES = (".pyc", ".lstudio")
UA = "Materia-updater"


# ------------------------------------------------------------------------------------------ versions
def parse_version(v: str) -> tuple:
    nums = re.findall(r"\d+", v or "")
    return tuple(int(n) for n in nums[:4]) or (0,)


def is_newer(remote: str, local: str) -> bool:
    return parse_version(remote) > parse_version(local)


def local_version(app_root: str) -> str:
    p = os.path.join(app_root, "labstudio", "__init__.py")
    try:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', open(p, encoding="utf-8").read())
        return m.group(1) if m else "0"
    except OSError:
        return "0"


# ------------------------------------------------------------------------------------------ manifest
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def iter_release_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_PARTS]
        for fn in filenames:
            if fn.endswith(EXCLUDE_SUFFIXES) or fn == MANIFEST:
                continue
            full = os.path.join(dirpath, fn)
            rel = _rel(full, root)
            if rel in (CONFIG, RELEASE_INFO):  # shipped/served, but never part of a release's file set
                continue
            yield rel, full


def build_manifest(root: str, version: str | None = None) -> dict:
    files = {rel: _sha256_file(full) for rel, full in sorted(iter_release_files(root))}
    return {"app": "Materia", "version": version or local_version(root),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files}


def read_manifest(root: str) -> dict | None:
    try:
        with open(os.path.join(root, MANIFEST), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_manifest(root: str, manifest: dict) -> None:
    with open(os.path.join(root, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)


# ------------------------------------------------------------------------------------------ config / net
def read_config(app_root: str) -> dict:
    try:
        with open(os.path.join(app_root, CONFIG), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def user_home() -> str:
    d = os.environ.get("LABSTUDIO_HOME") or os.path.join(os.path.expanduser("~"), ".labstudio")
    os.makedirs(d, exist_ok=True)
    return d


def _token() -> str | None:
    tok = os.environ.get("MATERIA_GITHUB_TOKEN")
    if tok:
        return tok.strip()
    p = os.path.join(user_home(), "github_token")
    try:
        return open(p, encoding="utf-8").read().strip() or None
    except OSError:
        return None


def _request(url: str, accept: str = "application/vnd.github+json", timeout: float = 10.0) -> bytes:
    headers = {"User-Agent": UA, "Accept": accept}
    tok = _token()
    if tok and "github" in url:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read()


@dataclass
class UpdateInfo:
    version: str
    tag: str
    notes: str
    zip_url: str
    asset_api_url: str | None = None
    published: str = ""
    size: int = 0
    current: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return f"Materia {self.version}"


def check_for_update(app_root: str, *, timeout: float = 6.0, include_prerelease: bool = False) -> UpdateInfo | None:
    """Return UpdateInfo if a newer release exists, None if up to date or not configured. Raises on network
    errors (callers decide whether to stay silent).

    Channels (update.json "channel"):
      "repo"     – read release.json from the main branch; download that commit's snapshot (default here),
      "releases" – GitHub Releases (latest release + zip asset); falls back to "repo" if there are none.
    """
    cfg = read_config(app_root)
    repo = cfg.get("repo")
    if not repo:
        return None
    current = local_version(app_root)
    channel = cfg.get("channel", "releases")
    rel_info = None
    if channel != "repo":
        try:
            rel_info = _latest_from_releases(cfg, repo, timeout, include_prerelease)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
    if rel_info is None:
        rel_info = _latest_from_repo(cfg, repo, timeout)
    if rel_info is None or not is_newer(rel_info.version, current):
        return None
    rel_info.current = current
    rel_info.extra["skipped"] = read_skip() == rel_info.version
    return rel_info


def _latest_from_releases(cfg, repo, timeout, include_prerelease) -> UpdateInfo | None:
    api = (cfg.get("api") or "https://api.github.com").rstrip("/")  # "api" override is for testing only
    if include_prerelease or cfg.get("prerelease"):
        rels = json.loads(_request(f"{api}/repos/{repo}/releases?per_page=10", timeout=timeout))
        rel = next((r for r in rels if not r.get("draft")), None)
    else:
        rel = json.loads(_request(f"{api}/repos/{repo}/releases/latest", timeout=timeout))
    if not rel:
        return None
    tag = rel.get("tag_name") or ""
    asset = next((a for a in rel.get("assets", []) if a.get("name", "").lower().endswith(".zip")), None)
    return UpdateInfo(version=tag.lstrip("vV"), tag=tag, notes=(rel.get("body") or "").strip(),
                      zip_url=(asset or {}).get("browser_download_url") or rel.get("zipball_url"),
                      asset_api_url=(asset or {}).get("url"), published=rel.get("published_at", ""),
                      size=(asset or {}).get("size", 0))


def _latest_from_repo(cfg, repo, timeout) -> UpdateInfo | None:
    raw = (cfg.get("raw") or "https://raw.githubusercontent.com").rstrip("/")
    code = (cfg.get("codeload") or "https://codeload.github.com").rstrip("/")
    branch = cfg.get("branch", "main")
    try:
        data = _request(f"{raw}/{repo}/{branch}/{RELEASE_INFO}?t={int(time.time())}",
                        accept="application/json", timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    info = json.loads(data)
    commit = info.get("commit") or branch
    return UpdateInfo(version=str(info.get("version", "0")), tag=f"v{info.get('version')}",
                      notes=(info.get("notes") or "").strip(), zip_url=f"{code}/{repo}/zip/{commit}",
                      published=info.get("published", ""), size=int(info.get("size") or 0),
                      extra={"commit": commit})


def _skip_file() -> str:
    return os.path.join(user_home(), "update_skip.txt")


def read_skip(app_root: str | None = None) -> str:
    try:
        return open(_skip_file(), encoding="utf-8").read().strip()
    except OSError:
        return ""


def skip_version(version: str) -> None:
    with open(_skip_file(), "w", encoding="utf-8") as fh:
        fh.write(version)


def download(info: UpdateInfo, progress=None) -> bytes:
    """Download the release zip (asset via API when a token is needed for private repos)."""
    url, accept = info.zip_url, "application/octet-stream"
    if _token() and info.asset_api_url:
        url = info.asset_api_url
    headers = {"User-Agent": UA, "Accept": accept}
    tok = _token()
    if tok and "github" in url:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context()) as r:
        total = int(r.headers.get("Content-Length") or info.size or 0)
        buf = io.BytesIO()
        done = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            buf.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    return buf.getvalue()


# ------------------------------------------------------------------------------------------ apply
@dataclass
class UpdateResult:
    old_version: str
    new_version: str
    written: list
    removed: list
    unchanged: int
    requirements_changed: bool
    backup_dir: str


def _is_protected(rel: str) -> bool:
    return rel.startswith(PROTECTED_PREFIXES) or ".." in rel.split("/") or rel.startswith("/")


def extract_release(zip_bytes: bytes, dest: str) -> str:
    """Extract and return the folder that holds the release (handles a single top-level folder)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for n in z.namelist():  # zip-slip guard
            if n.startswith("/") or ".." in n.replace("\\", "/").split("/"):
                raise ValueError(f"Unsafe path in update: {n}")
        z.extractall(dest)
    for dirpath, dirnames, filenames in os.walk(dest):
        if MANIFEST in filenames and "launcher.pyw" in filenames:
            return dirpath
    raise ValueError("The downloaded update is not a Materia release (no MANIFEST.json / launcher.pyw).")


def verify_release(root: str) -> dict:
    man = read_manifest(root)
    if not man or "files" not in man:
        raise ValueError("Update has no valid manifest.")
    for rel, digest in man["files"].items():
        if _is_protected(rel):
            raise ValueError(f"Update tries to change a protected path: {rel}")
        p = os.path.join(root, rel)
        if not os.path.isfile(p) or _sha256_file(p) != digest:
            raise ValueError(f"Update file is missing or corrupt: {rel}")
    return man


def plan_update(app_root: str, new_root: str, new_man: dict) -> tuple[list, list, int]:
    """(files to write, files to remove, unchanged count). Only touches release files."""
    old_man = read_manifest(app_root) or {"files": {}}
    write, unchanged = [], 0
    for rel, digest in new_man["files"].items():
        cur = os.path.join(app_root, rel)
        if os.path.isfile(cur) and _sha256_file(cur) == digest:
            unchanged += 1
        else:
            write.append(rel)
    remove = [rel for rel in old_man.get("files", {})
              if rel not in new_man["files"] and not _is_protected(rel)
              and os.path.isfile(os.path.join(app_root, rel))]
    return write, remove, unchanged


def apply_update_from_folder(app_root: str, new_root: str) -> UpdateResult:
    new_man = verify_release(new_root)
    old_version = local_version(app_root)
    write, remove, unchanged = plan_update(app_root, new_root, new_man)
    backup = os.path.join(app_root, BACKUP_DIR, old_version)
    if os.path.isdir(backup):
        shutil.rmtree(backup, ignore_errors=True)
    os.makedirs(backup, exist_ok=True)
    req_before = _sha256_file(os.path.join(app_root, "requirements.txt")) \
        if os.path.isfile(os.path.join(app_root, "requirements.txt")) else ""
    done: list[str] = []
    try:
        for rel in write + remove:  # back up everything we will touch
            cur = os.path.join(app_root, rel)
            if os.path.isfile(cur):
                b = os.path.join(backup, rel)
                os.makedirs(os.path.dirname(b), exist_ok=True)
                shutil.copy2(cur, b)
        old_man = read_manifest(app_root)
        if old_man:
            write_manifest(backup, old_man)
        with open(os.path.join(backup, "_new_files.json"), "w", encoding="utf-8") as fh:
            json.dump([r for r in write if not os.path.isfile(os.path.join(app_root, r))], fh)
        for rel in write:
            src, dst = os.path.join(new_root, rel), os.path.join(app_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".updating"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            done.append(rel)
        for rel in remove:
            try:
                os.remove(os.path.join(app_root, rel))
            except OSError:
                pass
        write_manifest(app_root, new_man)
    except Exception:
        _restore(app_root, backup)
        raise
    req_after = _sha256_file(os.path.join(app_root, "requirements.txt"))
    return UpdateResult(old_version, new_man.get("version", "?"), write, remove, unchanged,
                        req_before != req_after, backup)


def apply_update(app_root: str, info: UpdateInfo, progress=None) -> UpdateResult:
    data = download(info, progress)
    with tempfile.TemporaryDirectory(prefix="materia_update_") as tmp:
        new_root = extract_release(data, tmp)
        return apply_update_from_folder(app_root, new_root)


def _restore(app_root: str, backup: str) -> None:
    try:
        new_files = json.load(open(os.path.join(backup, "_new_files.json"), encoding="utf-8"))
    except (OSError, ValueError):
        new_files = []
    for rel in new_files:
        try:
            os.remove(os.path.join(app_root, rel))
        except OSError:
            pass
    for dirpath, _, filenames in os.walk(backup):
        for fn in filenames:
            if fn == "_new_files.json":
                continue
            src = os.path.join(dirpath, fn)
            rel = _rel(src, backup)
            dst = os.path.join(app_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def list_backups(app_root: str) -> list[str]:
    d = os.path.join(app_root, BACKUP_DIR)
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d), key=parse_version)


def rollback(app_root: str, version: str | None = None) -> str:
    """Restore the files backed up before the update away from ``version`` (default: newest backup)."""
    vers = list_backups(app_root)
    if not vers:
        raise ValueError("No update backups available.")
    v = version or vers[-1]
    _restore(app_root, os.path.join(app_root, BACKUP_DIR, v))
    return v
