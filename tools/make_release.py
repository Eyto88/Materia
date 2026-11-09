"""Build (and optionally publish) a Materia release.

    python tools/make_release.py                  # dist/Materia_v<version>.zip with MANIFEST.json
    python tools/make_release.py --publish --notes "What's new…"   # also create the GitHub release (gh CLI)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--publish", action="store_true", help="create the GitHub release with the gh CLI")
    ap.add_argument("--notes", default="", help="release notes (markdown); shown in the update prompt")
    ap.add_argument("--notes-file", default="")
    args = ap.parse_args()
    zpath, man = build(args.out)
    print(f"Built {zpath}  ({len(man['files'])} files, version {man['version']})")
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
