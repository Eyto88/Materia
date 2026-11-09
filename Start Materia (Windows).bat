@echo off
rem Materia launcher for Windows: finds Python and opens the Materia loading screen (launcher.pyw).
rem The first start sets up a private environment (.venv) with a progress bar, then creates a
rem "Materia" shortcut with the Materia icon here and on the Desktop - use that from then on.
setlocal EnableExtensions
cd /d "%~dp0"

if exist "launcher.pyw" goto files_ok
echo Materia's files were not found next to this launcher.
echo.
echo You probably opened it directly from inside the zip file.
echo Right-click the zip, choose "Extract All...", open the extracted folder and
echo double-click "Start Materia (Windows).bat" there.
echo.
pause
exit /b 1
:files_ok

rem ---- Python 3.10+ : the "py" launcher first, then "python" (skips the Microsoft Store stub)
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto use_py
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 goto use_python
echo Materia needs Python 3.10 or newer, but none was found.
echo.
echo Install Python from https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" in the installer, then start Materia again.
echo.
pause
exit /b 1

:use_py
rem pyw = windowed Python (no console window)
where pyw >nul 2>&1
if errorlevel 1 (
    start "" py -3 "%~dp0launcher.pyw" %*
) else (
    start "" pyw -3 "%~dp0launcher.pyw" %*
)
exit /b 0

:use_python
where pythonw >nul 2>&1
if errorlevel 1 (
    start "" python "%~dp0launcher.pyw" %*
) else (
    start "" pythonw "%~dp0launcher.pyw" %*
)
exit /b 0
