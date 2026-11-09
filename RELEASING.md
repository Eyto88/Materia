# Releasing a new Materia version (for whoever builds the next version)

Updates are delivered as GitHub Releases on **Eyto88/materia** (public). Installed copies check for a newer
release every time the launcher starts (Help ▸ Check for updates… does the same inside the app), download the
release zip and replace **only files that changed**. User data is never touched: settings, presets, styles,
logs live in `~/.labstudio` (or the registry); sessions are wherever the user saved them; `.venv`,
`update.json` and any user files in the app folder are left alone. Packages are reinstalled only when
`requirements.txt` changes. The previous version's files are kept in `.update_backup/<version>/`
(Help ▸ Restore previous version…).

Steps for version X.Y.Z
1. Make the changes; run `QT_QPA_PLATFORM=offscreen python -m pytest -q` (all green).
2. Bump `__version__` in `labstudio/__init__.py` (semantic: 1.3.0, 1.3.1 …).
3. Commit and push to `main` of Eyto88/materia.
4. `python tools/make_release.py --publish --notes "- what changed, one line each"`
   → builds `dist/Materia_vX.Y.Z.zip` (with MANIFEST.json + update.json) and creates release `vX.Y.Z`
   with the zip attached. The release notes are what users see in the update prompt.
5. Done: every installed copy offers the update at its next start.

Rules
- Never rename/move user-data locations or the session format without a migration.
- Keep `launcher.pyw` and `labstudio/core/updater.py` standard-library only (they run before packages exist).
- If the launcher itself changes, it restarts itself after updating — test the update path from the previous
  release (see tests/test_updater.py) before publishing.
