@echo off
setlocal

rem Always launch relative to this file, including when it is double-clicked.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.12 app.py
    ) else (
        python app.py
    )
)

if errorlevel 1 (
    echo.
    echo NeuroCasting failed to start.
    pause
)

endlocal
