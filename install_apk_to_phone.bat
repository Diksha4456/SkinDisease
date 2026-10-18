@echo off
setlocal
echo ========================================================
echo  FairSkin AI - One-Click Phone APK Installer
echo ========================================================
echo.
set "PATH=%LOCALAPPDATA%\Android\Sdk\platform-tools;%PATH%"

echo [1/3] Checking connected Android devices via USB...
adb devices
echo.

echo [2/3] Setting up USB reverse port forwarding (tcp:5000)...
adb reverse tcp:5000 tcp:5000

echo [3/3] Installing FairSkin_AI_Mobile.apk to your phone...
adb install -r "%~dp0FairSkin_AI_Mobile.apk"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================================
    echo  SUCCESS! FairSkin AI is installed on your mobile phone!
    echo ========================================================
    echo You can now open "FairSkin AI" from your phone's app drawer.
) else (
    echo.
    echo [NOTE] If adb install did not find your device:
    echo 1. Connect your phone via USB cable.
    echo 2. Enable "USB Debugging" in phone Settings ^> Developer Options.
    echo 3. Alternatively, copy "FairSkin_AI_Mobile.apk" directly from your Desktop to your phone and tap it to install!
)
echo.
pause
