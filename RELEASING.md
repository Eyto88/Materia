# Releasing a new Materia version (for whoever builds the next version)

Updates are delivered from **Eyto88/materia** (public) through the *repo channel*: the main branch holds the
code + MANIFEST.json, and `release.json` announces the newest version and the commit to download
(GitHub Releases also work if they can be created; installed copies try them first only when
`update.json` says `"channel": "releases"`). Installed copies check for a newer
release every time the launcher starts (Help ▸ Check for updates… does the same inside the app), download the
release zip and replace **only files that changed**. User data is never touched: settings, presets, styles,
logs live in `~/.labstudio` (or the registry); sessions are wherever the user saved them; `.venv`,
`update.json` and any user files in the app folder are left alone. Packages are reinstalled only when
`requirements.txt` changes. The previous version's files are kept in `.update_backup/<version>/`
(Help ▸ Restore previous version…).

Steps for version X.Y.Z
1. Make the changes; run `QT_QPA_PLATFORM=offscreen python -m pytest -q` (all green).
2. Bump `__version__` in `labstudio/__init__.py` (semantic: 1.3.0, 1.3.1 …).
3. Clone the repo next to the dev tree (`git clone https://github.com/Eyto88/materia ../materia`).
4. `python tools/make_release.py --publish-repo ../materia --notes "- what changed, one line each" \
      --trailer "Co-Authored-By: …"`
   → syncs exactly the release files + MANIFEST.json into the clone, commits "Materia X.Y.Z", pushes,
   then commits `release.json` (version, commit, notes) and pushes. The notes are what users see in the
   update prompt. It also builds `dist/Materia_vX.Y.Z.zip` for a fresh manual install.
5. Done: every installed copy offers the update at its next start.

Rules
- Never rename/move user-data locations or the session format without a migration.
- Keep `launcher.pyw` and `labstudio/core/updater.py` standard-library only (they run before packages exist).
- If the launcher itself changes, it restarts itself after updating — test the update path from the previous
  release (see tests/test_updater.py) before publishing.
