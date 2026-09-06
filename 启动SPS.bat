@echo off
title SPS Stock Pattern System
cd /d "%~dp0"
set "SPS_DATA_DIR=%~dp0data"

if exist "%~dp0SPS.exe" goto root_exe
if exist "%~dp0release\SPS\SPS.exe" goto release_exe
if exist "%~dp0dist\SPS\SPS.exe" goto dist_exe
if exist "%~dp0.venv\Scripts\python.exe" goto dev_mode

echo [ERROR] SPS.exe and the project Python environment were not found.
echo Re-extract the complete package or install the development environment.
pause
exit /b 1

:root_exe
"%~dp0SPS.exe"
exit /b %errorlevel%

:release_exe
"%~dp0release\SPS\SPS.exe"
exit /b %errorlevel%

:dist_exe
"%~dp0dist\SPS\SPS.exe"
exit /b %errorlevel%

:dev_mode
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\app.py"
exit /b %errorlevel%
