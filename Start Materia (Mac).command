#!/bin/bash
# Materia launcher for macOS: opens the Materia loading screen (launcher.pyw).
# The first start sets up a private environment (.venv) with a progress bar.
cd "$(dirname "$0")" || exit 1
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Materia needs Python 3.10 or newer: install it from https://www.python.org/downloads/macos/"
  read -n 1 -s -r -p "Press any key to close"
  exit 1
fi
# run the loading screen detached so this Terminal window can be closed
nohup "$PY" launcher.pyw "$@" >/dev/null 2>&1 &
disown
echo "Materia is starting - you can close this window."
sleep 2
osascript -e 'tell application "Terminal" to close (every window whose name contains "Start Materia")' >/dev/null 2>&1 &
exit 0
