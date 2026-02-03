@echo off
echo ============================================
echo   Assessment OCR Scanner - Setup Script
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [WARNING] Could not create virtual environment, continuing with global Python
) else (
    echo [OK] Virtual environment created
    call venv\Scripts\activate.bat
)

echo.
echo [2/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo.
echo [3/3] Checking Tesseract OCR...
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo [OK] Tesseract found at default location
) else (
    echo [WARNING] Tesseract not found at default location
    echo Please install Tesseract OCR from:
    echo https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo After installation, update TESSERACT_PATH in .env file
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo To start the application, run:
echo   python app.py
echo.
echo Or double-click: run.bat
echo.
pause
