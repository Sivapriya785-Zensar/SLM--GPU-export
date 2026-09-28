@echo off
REM ============================================================
REM  Zensar Dispute Reason-Code Resolver - start everything
REM  1) Ollama server   2) FastAPI backend   3) Frontend + browser
REM  (the frontend is served by the backend - same process/port)
REM ============================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "PORT=8010"
set "OLLAMA_MODEL=dispute-phi3-4ep:latest"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo.
echo [1/4] Ollama server...
where ollama >nul 2>&1
if errorlevel 1 (
  echo       ollama not on PATH - skipping ^(install from https://ollama.com^)
  goto :afterollama
)
netstat -ano | findstr ":11434" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo       already running on port 11434 - opening a live status window
  start "Ollama" "%~dp0ollama_status.bat"
) else (
  echo       launching "ollama serve" in a new window
  start "Ollama" cmd /k "ollama serve"
  call :sleep 3
)
echo       warming model %OLLAMA_MODEL%
start "Ollama warm" /min cmd /c "ollama run %OLLAMA_MODEL% ping >nul 2>&1"
:afterollama

echo.
echo [2/4] Python dependencies...
%PY% -c "import fastapi, uvicorn, sklearn, joblib" >nul 2>&1
if errorlevel 1 (
  echo       installing from requirements.txt
  %PY% -m pip install -r requirements.txt
) else (
  echo       ok
)

echo.
echo [3/4] Trained models...
if not exist "ml\models\network.joblib" (
  echo       missing - training now ^(~45s^)
  %PY% ml\train.py
) else (
  echo       found ml\models\*.joblib
)

echo.
echo [4/4] Backend + frontend on http://localhost:%PORT%
start "Dispute Resolver API" cmd /k "%PY% -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%"
call :sleep 4
start "" "http://localhost:%PORT%"

echo.
echo ============================================================
echo  Running. Windows opened:
echo    - "Ollama"                model server ^(port 11434^)
echo    - "Dispute Resolver API"  backend + UI ^(port %PORT%^)
echo    - browser                 http://localhost:%PORT%
echo.
echo  Run  stop.bat  to shut everything down.
echo ============================================================
echo.
pause
endlocal
goto :eof

:sleep
REM sleep N seconds without needing console stdin (timeout fails when piped)
ping -n %1 127.0.0.1 >nul 2>&1
goto :eof
