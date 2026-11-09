"""Build (and optionally publish) a Materia release.

    python tools/make_release.py                  # dist/Materia_v<version>.zip with MANIFEST.json
    python tools/make_release.py --publish --notes "What's new…"   # also create the GitHub release (gh CLI)
    python tools/make_release.py --publish-repo ../materia --notes "- What's new"
        # "repo" channel (no GitHub Releases needed): sync files + MANIFEST.json into the git clone, commit,
        # push, then commit release.json (version, commit, notes) and push — installed copies update from it.

The version comes from labstudio/__init__.py. The zip contains one top-level folder "Materia/" with the app,
MANIFEST.json (sha256 of every shipped file) and update.json (where to look for updates).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from labstudio.core import updater as U  # noqa: E402


def build(out_dir: str) -> tuple[str, dict]:
    version = U.local_version(ROOT)
    man = U.build_manifest(ROOT, version)
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, f"Materia_v{version}.zip")
    cfg = os.path.join(ROOT, U.CONFIG)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in man["files"]:
            full = os.path.join(ROOT, rel)
            zi = zipfile.ZipInfo.from_file(full, f"Materia/{rel}")
            if rel.endswith((".command", ".sh")):
                zi.external_attr = 0o100755 << 16
            with open(full, "rb") as fh:
                z.writestr(zi, fh.read(), compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("Materia/" + U.MANIFEST, json.dumps(man, indent=1, sort_keys=True))
        if os.path.isfile(cfg):
            z.write(cfg, "Materia/" + U.CONFIG)
    return zpath, man


KEEP_IN_REPO = {".gitignore", U.CONFIG, U.MANIFEST, U.RELEASE_INFO}


def _git(repo: str, *args: str) -> str:
    out = subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def publish_repo(repo: str, notes: str, trailers: list[str], name: str, email: str) -> str:
    """Sync the release into a git clone, commit + push, then announce it in release.json."""
    import datetime
    import shutil
    version = U.local_version(ROOT)
    man = U.build_manifest(ROOT, version)
    tracked = set(_git(repo, "ls-files").splitlines()) if os.path.isdir(os.path.join(repo, ".git")) else set()
    for rel in tracked:
        if rel not in man["files"] and rel not in KEEP_IN_REPO:
            os.remove(os.path.join(repo, rel))
    for rel in man["files"]:
        dst = os.path.join(repo, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
    shutil.copy2(os.path.join(ROOT, U.CONFIG), os.path.join(repo, U.CONFIG))
    U.write_manifest(repo, man)
    ident = ["-c", f"user.name={name}", "-c", f"user.email={email}"]
    msg_tail = ("\n\n" + "\n".join(trailers)) if trailers else ""
    _git(repo, "add", "-A")
    _git(repo, *ident, "commit", "-m", f"Materia {version}{msg_tail}")
    _git(repo, "push", "origin", "HEAD:main")
    sha = _git(repo, "rev-parse", "HEAD")
    info = {"version": version, "commit": sha, "notes": notes or f"Materia {version}",
            "published": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(repo, U.RELEASE_INFO), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    _git(repo, "add", U.RELEASE_INFO)
    _git(repo, *ident, "commit", "-m", f"Release {version}{msg_tail}")
    _git(repo, "push", "origin", "HEAD:main")
    return sha


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--publish", action="store_true", help="create the GitHub release with the gh CLI")
    ap.add_argument("--notes", default="", help="release notes (markdown); shown in the update prompt")
    ap.add_argument("--notes-file", default="")
    ap.add_argument("--publish-repo", default="", help="path of the git clone of the update repository")
    ap.add_argument("--trailer", action="append", default=[], help="commit message trailer line (repeatable)")
    ap.add_argument("--author", default="Eytan Valansi")
    ap.add_argument("--email", default="eytanvalansi@icloud.com")
    args = ap.parse_args()
    zpath, man = build(args.out)
    print(f"Built {zpath}  ({len(man['files'])} files, version {man['version']})")
    notes_txt = open(args.notes_file, encoding="utf-8").read() if args.notes_file else (args.notes or "")
    if args.publish_repo:
        sha = publish_repo(args.publish_repo, notes_txt, args.trailer, args.author, args.email)
        print(f"Published {man['version']} to the repo channel (commit {sha[:10]}).")
        return 0
    if args.publish:
        repo = U.read_config(ROOT).get("repo")
        if not repo:
            print("update.json has no 'repo' — cannot publish.")
            return 1
        notes = open(args.notes_file, encoding="utf-8").read() if args.notes_file else (args.notes or "")
        tag = f"v{man['version']}"
        cmd = ["gh", "release", "create", tag, zpath, "--repo", repo, "--title", f"Materia {man['version']}",
               "--notes", notes or f"Materia {man['version']}"]
        print(" ".join(cmd[:6]), "…")
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
