@echo off
REM Windows Batch Script for FLIMera Analysis

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Activate the virtual environment if it exists
if exist "%SCRIPT_DIR%..\venv\Scripts\activate.bat" (
    call "%SCRIPT_DIR%..\venv\Scripts\activate.bat"
) else (
    echo Warning: Virtual environment not found at .venv
    echo Please ensure you have activated the correct Python environment
)

REM Launch the UI
python "%SCRIPT_DIR%flim_gui.py"

pause
