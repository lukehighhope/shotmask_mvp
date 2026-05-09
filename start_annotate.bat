@echo off
REM shotmask_mvp annotate launcher — 2026-05-08
REM All behavior (video /video, HTML toolbar, cache) lives in annotate_shots.py next to this file.

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ---- Opening video --------------------------------------------------------
REM Double-click this bat        = file picker (no args).
REM Drag-drop an .mp4 on this bat  = open that file directly, no picker.
REM To ALWAYS use the picker (ignore dropped path / shortcut --video), set ONE of:
REM   set SHOTMASK_ALWAYS_PICK=1
REM before starting, or run:  py -3 annotate_shots.py --pick
REM ----------------------------------------------------------------------------

set "FVIDEO=%~1"
if "!SHOTMASK_ALWAYS_PICK!"=="1" (
  set "FVIDEO="
  echo SHOTMASK_ALWAYS_PICK=1 — will open file picker ^(ignoring dropped/path argument^).
  echo.
)

echo [%date% %time%] Stopping anything on port 8765 ...
powershell -NoProfile -Command "$o=@(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue ^| Select-Object -ExpandProperty OwningProcess -Unique); foreach ($id in $o) { Stop-Process -Id $id -Force -EA SilentlyContinue; Write-Host ('  stopped PID ' + $id + ' ') }; if (-not $o) { Write-Host '  port 8765 was already free.' }"
REM Fallback (some setups miss the PowerShell stop): LISTENING PIDs on 8765
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8765" ^| findstr LISTENING') do (
  echo   Fallback stop PID %%P (port 8765^)
  taskkill /F /PID %%P >nul 2>&1
)

timeout /t 1 /nobreak >nul

echo [%date% %time%] Starting annotate...
echo.
echo   Working folder: %CD%
echo   Script file: %~dp0annotate_shots.py
echo.

set "ANNOTATE_PY=%~dp0annotate_shots.py"
if not exist "%ANNOTATE_PY%" (
  echo ERROR: annotate_shots.py not found next to this bat:
  echo   "%ANNOTATE_PY%"
  pause
  exit /b 1
)

findstr /C:"_video_src_cache_bust_param" "%ANNOTATE_PY%" >nul 2>&1
if errorlevel 1 (
  echo WARNING: annotate_shots.py does not contain recent video-session markers.
  echo   Save the latest annotate_shots.py from your editor into this folder, then retry.
  echo.
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
  echo Using: py -3
  py -3 -c "import sys; print('Python:', sys.executable)" 2>nul
  if "!FVIDEO!"=="" (
    py -3 "%ANNOTATE_PY%"
  ) else (
    echo   Video: "!FVIDEO!"
    echo.
    py -3 "%ANNOTATE_PY%" --video "!FVIDEO!"
  )
  goto :done
)

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
  echo Using: python
  python -c "import sys; print('Python:', sys.executable)" 2>nul
  if "!FVIDEO!"=="" (
    python "%ANNOTATE_PY%"
  ) else (
    echo   Video: "!FVIDEO!"
    echo.
    python "%ANNOTATE_PY%" --video "!FVIDEO!"
  )
  goto :done
)

echo ERROR: Neither "py" nor "python" was found in PATH.
echo Install Python 3 and ensure the launcher or python.exe is on PATH.
pause
exit /b 1

:done
echo.
pause
