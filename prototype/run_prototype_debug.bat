@echo off
rem Same as run_prototype.bat but prints every step, so we can see which path is "not found".
echo Script folder: "%~dp0"
cd /d "%~dp0.."
echo Working folder now: "%CD%"
if not exist "fastsdcpu\env\Scripts\python.exe" echo MISSING: fastsdcpu\env\Scripts\python.exe
if not exist "prototype\app.py" echo MISSING: prototype\app.py
if not exist "prototype\config.json" echo NOTE: prototype\config.json not found
set HF_HUB_DISABLE_XET=1
echo Starting the app, then open http://127.0.0.1:7870
"fastsdcpu\env\Scripts\python.exe" "prototype\app.py"
echo Python exited with code %ERRORLEVEL%
pause
