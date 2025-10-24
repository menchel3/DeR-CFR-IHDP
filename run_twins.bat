@echo off
echo ================================================================================
echo Running DeR-CFR TWINS Dataset Experiment
echo ================================================================================
echo.

cd /d "%~dp0"
C:\tool\Anaconda3\envs\DeR_CFR_tf1\python.exe run_twins.py

echo.
echo ================================================================================
echo Experiment finished. Press any key to exit...
echo ================================================================================
pause
