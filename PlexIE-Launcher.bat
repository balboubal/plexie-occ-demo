@echo off
title PlexIE OCC Demo Launcher
cd /d "%~dp0"

echo.
echo  PlexIE OCC Demo Launcher
echo  ========================
echo.

node --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Node.js not found!
    echo  Install from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

echo  Starting launcher...
echo.

node launcher.cjs

echo.
pause
