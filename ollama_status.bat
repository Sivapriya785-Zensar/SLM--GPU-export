@echo off
REM Live-refreshing Ollama status view, used by run.bat when Ollama is already
REM running in the background (so there's still something visible to point at).
:loop
cls
echo ============================================================
echo  Ollama live status - refreshes every 3s (close window to stop)
echo ============================================================
echo.
echo === installed models ===
ollama list
echo.
echo === currently loaded in memory ===
ollama ps
echo.
echo (Loaded list is empty until a request is made. Send a message in the
echo  Dispute Assistant chat, or wait for the warm-up call, and the model
echo  will appear here with its CPU/GPU split and how long it stays resident.)
ping -n 4 127.0.0.1 >nul
goto loop
