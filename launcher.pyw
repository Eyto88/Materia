"""Materia launcher: branded loading screen, first-run setup and app start.

Runs with the system Python using only the standard library (tkinter), so it can show the Materia
splash *before* any packages are installed. It

  1. shows the Materia splash with a progress bar,
  2. creates/updates the private environment ``.venv`` when needed (pip output is parsed into friendly
     progress messages instead of a console full of downloads),
  3. starts Materia with the environment's Python (no console window) and keeps the splash up until the
     main window reports it is ready,
  0. checks GitHub for a newer Materia release and offers to install it (only changed files; settings kept),
  4. on Windows, creates a "Materia" shortcut (with the Materia icon) in this folder and on the Desktop.

Without tkinter (e.g. some Homebrew Pythons) it falls back to plain text progress in the terminal.
"""
from __future__ import annotations

import hashlib
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback

APP_NAME = "Materia"
ROOT = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(ROOT, ".venv")
REQ = os.path.join(ROOT, "requirements.txt")
STAMP = os.path.join(VENV, "requirements.sha256")
RES = os.path.join(ROOT, "labstudio", "resources")
SPLASH_PNG = os.path.join(RES, "splash.png")
ICON_ICO = os.path.join(RES, "materia.ico")
ICON_PNG = os.path.join(RES, "materia_256.png")
IS_WIN = os.name == "nt"
NO_WINDOW = 0x08000000 if IS_WIN else 0  # CREATE_NO_WINDOW
READY_TIMEOUT_S = 180

BG = "#030c1b"
FG = "#e6ebf5"
MUTED = "#8a96ad"
ACCENT = "#3d6bff"
ERR = "#ff8a80"


def home_dir() -> str:
    d = os.environ.get("LABSTUDIO_HOME") or os.path.join(os.path.expanduser("~"), ".labstudio")
    os.makedirs(d, exist_ok=True)
    return d


INSTALL_LOG = os.path.join(home_dir(), "install.log")
APP_LOG = os.path.join(home_dir(), "app_console.log")


def venv_python(windowed: bool = False) -> str:
    if IS_WIN:
        return os.path.join(VENV, "Scripts", "pythonw.exe" if windowed else "python.exe")
    return os.path.join(VENV, "bin", "python")


def req_hash() -> str:
    with open(REQ, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def needs_setup() -> bool:
    if not os.path.isfile(venv_python()):
        return True
    try:
        with open(STAMP, encoding="ascii") as fh:
            return fh.read().strip() != req_hash()
    except OSError:
        return True


def base_python() -> str:
    """The interpreter running this launcher, but the console one (python.exe) for building the venv."""
    exe = sys.executable
    if IS_WIN and exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.isfile(cand):
            return cand
    return exe


def updater():
    """Import labstudio.core.updater (standard library only) without needing the packages."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from labstudio.core import updater as U
    return U


# ----------------------------------------------------------------------------------------- setup worker
class Setup:
    """Creates the venv and installs requirements, reporting (fraction, message) through a callback."""

    def __init__(self, report) -> None:
        self.report = report
        self.n_reqs = sum(1 for line in open(REQ, encoding="utf-8")
                          if line.strip() and not line.lstrip().startswith("#"))
        self.expected = max(25, self.n_reqs * 3)  # rough number of packages incl. dependencies

    def _run(self, args: list[str], log, on_line=None) -> int:
        log.write("\n$ " + " ".join(args) + "\n")
        log.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
        p = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", creationflags=NO_WINDOW, env=env)
        for line in p.stdout:
            log.write(line)
            if on_line:
                on_line(line.rstrip())
        log.flush()
        return p.wait()

    def run(self) -> None:
        with open(INSTALL_LOG, "w", encoding="utf-8") as log:
            log.write(f"{APP_NAME} setup {time.ctime()}\nPython {sys.version}\n")
            if not os.path.isfile(venv_python()):
                self.report(0.03, "Preparing a private Python environment (first start only)…")
                rc = self._run([base_python(), "-m", "venv", VENV], log)
                if rc != 0 or not os.path.isfile(venv_python()):
                    raise RuntimeError("Could not create the Python environment (.venv).")
            self.report(0.08, "Updating the package installer…")
            self._run([venv_python(), "-m", "pip", "install", "-q", "--upgrade", "pip"], log)

            seen: set[str] = set()
            state = {"phase": "collect"}

            def on_line(line: str) -> None:
                m = re.match(r"\s*(Collecting|Requirement already satisfied:?)\s+([A-Za-z0-9_.\-]+)", line)
                if m:
                    name = re.split(r"[<>=!~\[; ]", m.group(2))[0]
                    if name.lower() not in seen:
                        seen.add(name.lower())
                        frac = 0.10 + 0.55 * min(1.0, len(seen) / self.expected)
                        verb = "Downloading" if m.group(1) == "Collecting" else "Checking"
                        self.report(frac, f"{verb} components… {name}")
                    return
                if line.startswith("Installing collected packages"):
                    state["phase"] = "install"
                    pk = line.split(":", 1)[-1].strip().split(", ")
                    self.report(0.70, f"Installing {len(pk)} components — this can take a few minutes…")
                elif line.startswith("Successfully installed"):
                    self.report(0.97, "Components installed.")
                elif "error" in line.lower() and "ERROR" in line:
                    self.report(None, line.strip()[:120])

            rc = self._run([venv_python(), "-m", "pip", "install", "--progress-bar", "off", "-r", REQ], log,
                           on_line)
            if rc != 0:
                raise RuntimeError("Installing the components failed. Check the internet connection and try "
                                   "again.")
            with open(STAMP, "w", encoding="ascii") as fh:
                fh.write(req_hash())
            self.report(1.0, "Setup complete.")


# ----------------------------------------------------------------------------------------- app start
def start_app(ready_file: str) -> subprocess.Popen:
    env = dict(os.environ, MATERIA_READY_FILE=ready_file, MATERIA_LAUNCHER="1")
    out = open(APP_LOG, "w", encoding="utf-8", errors="replace")
    args = [venv_python(windowed=True), os.path.join(ROOT, "run_labstudio.py")] + sys.argv[1:]
    return subprocess.Popen(args, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                            creationflags=NO_WINDOW)


def tail(path: str, n: int = 12) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-n:]).strip()
    except OSError:
        return ""


def create_windows_shortcuts() -> None:
    """Materia.lnk (with icon, no console) in the app folder and on the Desktop — once."""
    if not IS_WIN:
        return
    flag = os.path.join(VENV, "shortcuts.done")
    if os.path.exists(flag):
        return
    pyw = sys.executable if sys.executable.lower().endswith("pythonw.exe") else base_python().replace(
        "python.exe", "pythonw.exe")

    def q(text: str) -> str:  # PowerShell single-quoted literal (handles apostrophes in folder names)
        return "'" + text.replace("'", "''") + "'"

    target_args = '"' + os.path.join(ROOT, "launcher.pyw") + '"'
    ps = []
    for dest in (q(os.path.join(ROOT, APP_NAME + ".lnk")),
                 f"(Join-Path ([Environment]::GetFolderPath('Desktop')) {q(APP_NAME + '.lnk')})"):
        ps.append(
            f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut({dest});"
            f"$s.TargetPath={q(pyw)};$s.Arguments={q(target_args)};$s.WorkingDirectory={q(ROOT)};"
            f"$s.IconLocation={q(ICON_ICO + ',0')};$s.Description={q(APP_NAME)};$s.Save();")
    try:
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", " ".join(ps)],
                       creationflags=NO_WINDOW, timeout=30, capture_output=True)
        open(flag, "w").close()
    except Exception:
        pass


# ----------------------------------------------------------------------------------------- Tk splash
class Splash:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        if IS_WIN:
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Materia.App")
            except Exception:
                pass
        self.root = root = tk.Tk()
        root.title(APP_NAME)
        root.configure(bg=BG)
        root.overrideredirect(True)
        try:
            if IS_WIN and os.path.isfile(ICON_ICO):
                root.iconbitmap(ICON_ICO)
            elif os.path.isfile(ICON_PNG):
                root.iconphoto(True, tk.PhotoImage(file=ICON_PNG))
        except Exception:
            pass
        self.img = tk.PhotoImage(file=SPLASH_PNG) if os.path.isfile(SPLASH_PNG) else None
        w = self.img.width() if self.img else 720
        frame = tk.Frame(root, bg=BG, highlightthickness=1, highlightbackground="#1b2a4a")
        frame.pack(fill="both", expand=True)
        if self.img:
            tk.Label(frame, image=self.img, bg=BG, bd=0).pack()
        else:
            tk.Label(frame, text=APP_NAME, fg=FG, bg=BG, font=("Segoe UI", 40, "bold")).pack(pady=60)
        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("M.Horizontal.TProgressbar", troughcolor="#0d1830", background=ACCENT,
                        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)
        self.bar = ttk.Progressbar(frame, style="M.Horizontal.TProgressbar", orient="horizontal",
                                   length=w - 120, mode="determinate", maximum=1000)
        self.bar.pack(pady=(4, 10))
        self.msg = tk.Label(frame, text="Starting…", fg=MUTED, bg=BG, font=("Segoe UI", 10))
        self.msg.pack(pady=(0, 6))
        self.buttons = tk.Frame(frame, bg=BG)
        self.buttons.pack(pady=(0, 14))
        self._fit()
        root.attributes("-topmost", True)
        root.after(1500, lambda: root.attributes("-topmost", False))
        self.q: queue.Queue = queue.Queue()
        self.frac = 0.0
        self.pulse = False
        root.after(50, self._poll)

    def _fit(self) -> None:
        """Size the window to its content and centre it (re-run when an error adds text/buttons)."""
        root = self.root
        root.update_idletasks()
        W, H = root.winfo_reqwidth(), root.winfo_reqheight()
        x = (root.winfo_screenwidth() - W) // 2
        y = (root.winfo_screenheight() - H) // 3
        root.geometry(f"{W}x{H}+{x}+{y}")

    # thread-safe progress
    def report(self, frac, text) -> None:
        self.q.put((frac, text))

    def _poll(self) -> None:
        try:
            while True:
                frac, text = self.q.get_nowait()
                if frac is not None:
                    self.frac = max(self.frac, frac)
                if text:
                    self.msg.configure(text=text, fg=MUTED)
        except queue.Empty:
            pass
        if self.pulse and self.frac < 0.95:  # slow creep during long silent phases
            self.frac = min(0.95, self.frac + 0.0008)
        self.bar["value"] = int(self.frac * 1000)
        self.root.after(80, self._poll)

    def error(self, text: str, details: str = "", retry=None) -> None:
        tk = self.tk
        self.pulse = False
        self.msg.configure(text=text, fg=ERR, wraplength=640, justify="center")
        for c in self.buttons.winfo_children():
            c.destroy()
        def btn(label, cmd):
            tk.Button(self.buttons, text=label, command=cmd, bg="#14244a", fg=FG, activebackground=ACCENT,
                      activeforeground="white", relief="flat", padx=14, pady=4, bd=0).pack(side="left", padx=6)
        if details:
            btn("Show details", lambda: self._open(details))
        if retry:
            btn("Try again", retry)
        btn("Close", self.root.destroy)
        self._fit()

    def prompt(self, text: str, buttons: list) -> None:
        """Show a message with custom (label, callback) buttons on the splash."""
        tk = self.tk
        self.pulse = False
        self.msg.configure(text=text, fg=FG, wraplength=640, justify="left")
        for c in self.buttons.winfo_children():
            c.destroy()
        for i, (label, cmd) in enumerate(buttons):
            tk.Button(self.buttons, text=label, command=cmd, bg=ACCENT if i == 0 else "#14244a", fg="white",
                      activebackground=ACCENT, activeforeground="white", relief="flat", padx=14, pady=4,
                      bd=0).pack(side="left", padx=6)
        self._fit()

    def clear_buttons(self) -> None:
        for c in self.buttons.winfo_children():
            c.destroy()
        self.msg.configure(fg=MUTED)
        self._fit()

    @staticmethod
    def _open(path: str) -> None:
        try:
            if IS_WIN:
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass


def run_gui() -> int:
    s = Splash()
    state: dict = {"proc": None, "ready": None, "t0": 0.0}

    def launch() -> None:
        s.clear_buttons()
        s.report(max(s.frac, 0.97), f"Starting {APP_NAME}…")
        s.pulse = True
        fd, ready = tempfile.mkstemp(prefix="materia_ready_")
        os.close(fd)
        os.remove(ready)
        state.update(ready=ready, t0=time.time())
        try:
            state["proc"] = start_app(ready)
        except Exception as exc:
            s.error(f"Could not start {APP_NAME}: {exc}", APP_LOG, retry=launch)
            return
        s.root.after(200, watch)

    def watch() -> None:
        proc, ready = state["proc"], state["ready"]
        if os.path.exists(ready):
            try:
                os.remove(ready)
            except OSError:
                pass
            s.report(1.0, "Ready")
            s.root.after(250, s.root.destroy)
            return
        rc = proc.poll()
        if rc is not None:
            if rc == 0:  # closed before signalling ready (e.g. user quit quickly)
                s.root.destroy()
                return
            last = [l for l in tail(APP_LOG, 15).splitlines() if l.strip()]
            hint = last[-1][:160] if last else ""
            s.error(f"{APP_NAME} stopped while starting.\n{hint}", APP_LOG, retry=launch)
            return
        if time.time() - state["t0"] > READY_TIMEOUT_S:
            s.root.destroy()  # app is still running; just stop showing the splash
            return
        s.root.after(200, watch)

    def setup_then_launch() -> None:
        s.clear_buttons()
        s.msg.configure(fg=MUTED)
        s.pulse = True

        def work():
            try:
                Setup(s.report).run()
            except Exception as exc:
                s.root.after(0, lambda e=exc: s.error(str(e), INSTALL_LOG, retry=setup_then_launch))
                with open(INSTALL_LOG, "a", encoding="utf-8") as log:
                    log.write(traceback.format_exc())
                return
            create_windows_shortcuts()
            s.root.after(0, launch)

        threading.Thread(target=work, daemon=True).start()

    def continue_start() -> None:
        s.clear_buttons()
        s.msg.configure(justify="center")
        if needs_setup():
            s.report(0.01, "Setting up Materia's components (a few minutes, needs internet)…")
            s.root.after(300, setup_then_launch)
        else:
            s.report(0.6, f"Loading {APP_NAME}…")
            s.root.after(100, launch)

    def do_update(info) -> None:
        s.clear_buttons()
        s.frac = 0.0
        s.report(0.02, f"Downloading {APP_NAME} {info.version}…")

        def work():
            U = updater()
            try:
                def prog(done, total):
                    if total:
                        s.report(0.02 + 0.6 * done / total, f"Downloading {APP_NAME} {info.version}… "
                                 f"{done / 1e6:.1f} / {total / 1e6:.1f} MB")
                res = U.apply_update(ROOT, info, prog)
            except Exception as exc:
                with open(INSTALL_LOG, "a", encoding="utf-8") as log:
                    log.write("\nUpdate failed:\n" + traceback.format_exc())
                s.root.after(0, lambda e=exc: s.prompt(
                    f"The update could not be installed ({e}).\nYour current version is unchanged.",
                    [("Start current version", continue_start), ("Show details", lambda: s._open(INSTALL_LOG))]))
                return
            s.report(0.65, f"Updated to {APP_NAME} {res.new_version}: {len(res.written)} files changed, "
                           f"{res.unchanged} unchanged. Your settings are untouched.")
            if "launcher.pyw" in res.written:  # the launcher itself changed: restart it
                s.root.after(1200, restart_launcher)
            else:
                s.root.after(1200, continue_start)

        threading.Thread(target=work, daemon=True).start()

    def restart_launcher() -> None:
        subprocess.Popen([sys.executable, os.path.abspath(__file__)] + sys.argv[1:], cwd=ROOT,
                         creationflags=NO_WINDOW)
        s.root.destroy()

    def offer(info) -> None:
        notes = "\n".join(l for l in info.notes.splitlines() if l.strip())[:600]
        more = "…" if len(info.notes) > 600 else ""
        text = (f"{APP_NAME} {info.version} is available (you have {info.current}).\n\n"
                f"{notes}{more}\n\nOnly changed files are replaced; your settings, presets, styles and "
                f"sessions are kept.")
        s.prompt(text, [("Update now", lambda: do_update(info)), ("Later", continue_start),
                        ("Skip this version", lambda: (updater().skip_version(info.version), continue_start()))])

    def check_updates() -> None:
        s.report(0.01, "Checking for updates…")
        result: dict = {}

        def work():
            try:
                result["info"] = updater().check_for_update(ROOT, timeout=4)
            except Exception as exc:  # offline, rate limited… -> start normally
                result["error"] = exc

        t = threading.Thread(target=work, daemon=True)
        t.start()

        def wait(n=0):
            if t.is_alive() and n < 60:
                s.root.after(100, lambda: wait(n + 1))
                return
            info = result.get("info")
            if info is not None and not info.extra.get("skipped") and "--no-update" not in sys.argv:
                offer(info)
            else:
                continue_start()

        wait()

    if os.path.isfile(os.path.join(ROOT, "update.json")) and not os.environ.get("MATERIA_NO_UPDATE_CHECK"):
        s.root.after(150, check_updates)
    else:
        continue_start()
    s.root.mainloop()
    return 0


def run_text() -> int:
    """Fallback without tkinter: readable progress in the terminal."""
    print(f"{APP_NAME}: starting…")
    if needs_setup():
        print("First start: setting up the private environment (a few minutes, needs internet)…")
        last = [""]

        def report(frac, text):
            if text and text != last[0]:
                last[0] = text
                pct = f"{int(frac * 100):3d}% " if frac is not None else "     "
                print(pct + text, flush=True)

        try:
            Setup(report).run()
        except Exception as exc:
            print(f"\nSetup failed: {exc}\nDetails: {INSTALL_LOG}")
            return 1
    fd, ready = tempfile.mkstemp(prefix="materia_ready_")
    os.close(fd)
    os.remove(ready)
    proc = start_app(ready)
    rc = proc.wait()
    if rc != 0:
        print(f"{APP_NAME} exited with an error:\n{tail(APP_LOG)}")
    return rc


def main() -> int:
    if not os.path.isfile(os.path.join(ROOT, "run_labstudio.py")):
        print("Materia's files were not found. Extract the zip file first, then start the launcher from the "
              "extracted folder.")
        return 1
    try:
        import tkinter  # noqa: F401
    except Exception:
        return run_text()
    return run_gui()


if __name__ == "__main__":
    sys.exit(main())
