@echo off
title CIS v2 — Criminal Identification System
color 0B
cls

echo.
echo  =========================================================================
echo   CIS v2  ^|  Criminal Identification System  ^|  Team BioFuse  ^|  IGNISIA
echo  =========================================================================
echo.

:: Change to the folder where this bat lives
cd /d "%~dp0"

:: ── [1/5] Python check ───────────────────────────────────────────────────────
echo  [1/5] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo  [ERROR] Python not found!
    echo         Install Python 3.8+ from: https://www.python.org/downloads/
    echo         Tick "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo  [OK]  Python %PYVER%

:: ── [2/5] Core packages ──────────────────────────────────────────────────────
echo.
echo  [2/5] Checking core packages...
python -c "import cv2, PIL, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo  [*]   Installing: opencv-python pillow numpy ...
    pip install opencv-python pillow numpy --quiet
    if %errorlevel% neq 0 (
        color 0C
        echo  [ERROR] Could not install packages. Run manually:
        echo         pip install opencv-python pillow numpy
        echo.
        pause
        exit /b 1
    )
    echo  [OK]  Packages installed
) else (
    echo  [OK]  opencv + pillow + numpy ready
)

:: ── [3/5] Optional mediapipe ─────────────────────────────────────────────────
echo.
echo  [3/5] Optional: mediapipe (gait analysis)...
python -c "import mediapipe as mp; assert hasattr(mp,'solutions') and hasattr(mp.solutions,'pose'); print('[OK]  mediapipe ready (old API compatible)')" 2>nul
if %errorlevel% neq 0 (
    echo  [INFO] mediapipe not available or version incompatible (0.10+ changed API)
    echo  [INFO] Gait analysis will use OpenCV fallback - no action needed
)

:: ── [4/5] Camera probe ───────────────────────────────────────────────────────
echo.
echo  [4/5] Camera probe...
python -c "import cv2; cap=cv2.VideoCapture(0,cv2.CAP_DSHOW); r,f=cap.read(); cap.release(); print('[OK]  Built-in camera OK' if r else '[WARN] Camera 0 not detected')" 2>nul
if %errorlevel% neq 0 (
    echo  [WARN] Camera check skipped
)

:: ── [5/5] Fresh database ─────────────────────────────────────────────────────
echo.
echo  [5/5] Initializing fresh database (clears old session)...
python -c "from database.db_manager import init_db,get_stats; init_db(); s=get_stats(); print(f'[OK]  {s[chr(34)+chr(116)+chr(111)+chr(116)+chr(97)+chr(108)+chr(95)+chr(99)+chr(114)+chr(105)+chr(109)+chr(105)+chr(110)+chr(97)+chr(108)+chr(115)+chr(34)]} criminals seeded, 0 prior detections')" 2>nul
if %errorlevel% neq 0 (
    python -c "from database.db_manager import init_db; init_db(); print('[OK]  Database ready')" 2>nul
    if %errorlevel% neq 0 (
        echo  [WARN] DB pre-init skipped (will auto-init on launch)
    )
)

:: ── Launch ───────────────────────────────────────────────────────────────────
echo.
echo  =========================================================================
echo   Tab 1 - Image Upload  : Photo analysis + DB criminal name match
echo   Tab 2 - Video Upload  : Frame-by-frame CCTV processing
echo   Tab 3 - Live Webcam   : 3-scan pipeline + Add-to-DB popup
echo   Tab 4 - Database      : Registry (30 criminals) + session log
echo  =========================================================================
echo.
echo  Launching CIS v2 ... (this window stays open while the app runs)
echo.

python cis_v2.py

:: ── Post-exit ────────────────────────────────────────────────────────────────
echo.
if %errorlevel% neq 0 (
    color 0C
    echo  =========================================================================
    echo  [ERROR] CIS v2 crashed (exit code %errorlevel%)
    echo  =========================================================================
    echo.
    echo  Quick fixes:
    echo    1. pip install --upgrade opencv-python pillow numpy
    echo    2. Close Teams / Zoom before starting (camera conflict)
    echo    3. Run as Administrator
    echo.
) else (
    color 0A
    echo  [OK] CIS v2 closed normally.
)
echo.
pause
