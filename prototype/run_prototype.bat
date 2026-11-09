@echo off
rem Starts the layer prototype UI at http://127.0.0.1:7870 (Ollama must be running: it starts with Windows after install)
cd /d "%~dp0.."
set HF_HUB_DISABLE_XET=1
"fastsdcpu\env\Scripts\python.exe" "prototype\app.py"
pause
