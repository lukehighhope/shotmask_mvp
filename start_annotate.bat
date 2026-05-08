@echo off
setlocal
cd /d "%~dp0"

REM Optional: drag an .mp4 onto this bat or run: start_annotate.bat "path with spaces\video.mp4"
REM No argument opens annotate_shots built-in video file picker.

echo [%date% %time%] Stop anything listening on port 8765 ...
powershell -NoProfile -Command "$o=@(Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue ^| Select-Object -ExpandProperty OwningProcess -Unique); foreach ($id in $o) { Stop-Process -Id $id -Force -EA SilentlyContinue; Write-Host ('  stopped PID ' + $id + ' ') }; if (-not $o) { Write-Host '  port 8765 was already free.' }"

timeout /t 1 /nobreak >nul

echo [%date% %time%] Starting annotate...
echo.

if "%~1"=="" (
  python annotate_shots.py
) else (
  echo   Video: "%~1"
  echo.
  python annotate_shots.py --video "%~1"
)
echo.
pause
