@echo off
REM Stop the backend / Ollama windows started by run.bat
setlocal
echo Stopping Dispute Resolver backend (uvicorn on 8010)...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":8010" ^| findstr "LISTENING"') do taskkill /f /pid %%p >nul 2>&1

echo Stopping the "ollama serve" window started by run.bat (Ollama desktop app is left running)...
taskkill /f /fi "WINDOWTITLE eq Ollama*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Dispute Resolver API*" >nul 2>&1

echo Done.
endlocal
