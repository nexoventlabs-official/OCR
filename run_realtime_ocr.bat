@echo off
REM Run Real-Time OCR (Production Mode)
echo.
echo ============================================
echo    REAL-TIME OCR - PRODUCTION MODE
echo ============================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Check command line arguments
if "%1"=="" (
    echo Usage: run_realtime_ocr.bat [options]
    echo.
    echo Options:
    echo   --view 1-4    View mode ^(1=high conf, 2=color, 3=gradient, 4=all^)
    echo   --crop W H    Crop area in pixels
    echo   --lang CODE   Language code ^(eng, chi_sim, etc.^)
    echo   --easyocr     Use EasyOCR instead of Tesseract
    echo.
    echo Starting with defaults...
    echo.
)

REM Run the real-time OCR module
python -m modules.realtime_ocr %*

echo.
echo Real-Time OCR stopped.
pause
