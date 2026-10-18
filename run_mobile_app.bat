@echo off
title FairSkin AI - Mobile Application Launcher
echo ============================================================
echo      FairSkin AI - Flutter Mobile Application Launcher
echo ============================================================
echo.

:: 1. Ensure Flutter bin is in PATH for this session
set "PATH=C:\src\flutter\bin;%PATH%"

:: 2. Check if Python backend is already running
powershell -Command "$c = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }"
if %ERRORLEVEL% EQU 0 (
    echo [OK] FairSkin AI backend is already active on port 5000.
) else (
    echo [*] Starting local FairSkin AI inference backend on port 5000...
    start /min "FairSkin AI Backend" python app.py
    timeout /t 4 /nobreak >nul
)

:: 3. Setup USB reverse port forwarding if ADB is available
where adb >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] Setting up USB reverse tethering (port 5000)...
    adb reverse tcp:5000 tcp:5000
) else (
    if exist "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" (
        echo [*] Setting up USB reverse tethering with Android SDK...
        "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" reverse tcp:5000 tcp:5000
    )
)

:: 4. Launch Flutter Application
echo.
echo [*] Launching FairSkin Mobile App...
cd /d "%~dp0mobile_flutter"

echo.
echo Select target device:
echo   [1] USB-Connected Android Phone (or default device)
echo   [2] Google Chrome (Web Mobile Preview)
echo   [3] Microsoft Edge
set /p choice="Enter choice [1/2/3] (default 1): "

if "%choice%"=="2" (
    flutter run -d chrome
) else if "%choice%"=="3" (
    flutter run -d edge
) else (
    flutter run
)

pause
