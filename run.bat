@echo off
echo Starting Assessment OCR Scanner...
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run the application
python app.py

pause
