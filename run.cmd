@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [filler-detector] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :error
)

echo [filler-detector] Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q --disable-pip-version-check
if errorlevel 1 goto :error

start "" /min ".venv\Scripts\python.exe" -c "import time,webbrowser;time.sleep(2);webbrowser.open('http://127.0.0.1:8020')"
echo [filler-detector] Starting server at http://127.0.0.1:8020
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8020
goto :eof

:error
echo [filler-detector] Setup failed. Check that Python 3.11+ is on PATH and network is available.
pause
exit /b 1