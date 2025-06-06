@echo off
TITLE Image-to-Prompt AI - Installer
COLOR 0A

ECHO.
ECHO  =======================================================
ECHO      Image-to-Prompt AI Application Installer
ECHO  =======================================================
ECHO.
ECHO  This script will set up the necessary environment and
ECHO  install the required Python packages.
ECHO.

:: Change directory to the script's location to ensure paths are correct
cd /d "%~dp0"

:: Step 1: Check if Python is installed and available in PATH
ECHO [1/4] Checking for Python installation...
python --version >nul 2>nul
IF %errorlevel% NEQ 0 (
    COLOR 0C
    ECHO.
    ECHO  ERROR: Python is not found in your system's PATH.
    ECHO  Please install Python from https://www.python.org/
    ECHO  and make sure to check "Add Python to PATH" during installation.
    ECHO.
    PAUSE
    EXIT /B
)
ECHO  Python found!
ECHO.

:: Step 2: Create a virtual environment
ECHO [2/4] Creating virtual environment folder ('venv')...
IF EXIST venv (
    ECHO  'venv' folder already exists. Skipping creation.
) ELSE (
    python -m venv venv
    IF %errorlevel% NEQ 0 (
        COLOR 0C
        ECHO.
        ECHO  ERROR: Failed to create the virtual environment.
        ECHO.
        PAUSE
        EXIT /B
    )
    ECHO  Virtual environment created successfully.
)
ECHO.

:: Step 3: Activate the virtual environment and install requirements
ECHO [3/4] Activating environment and installing packages from requirements.txt...
ECHO  This may take a few moments.
ECHO.
CALL venv\Scripts\activate.bat
pip install -r requirements.txt
IF %errorlevel% NEQ 0 (
    COLOR 0C
    ECHO.
    ECHO  ERROR: Failed to install required packages.
    ECHO  Please check your internet connection and the 'requirements.txt' file.
    ECHO.
    PAUSE
    EXIT /B
)
ECHO.

:: Step 4: Finalization
ECHO [4/4] Installation complete!
ECHO.
ECHO  =======================================================
ECHO      SETUP SUCCESSFUL!
ECHO  =======================================================
ECHO.
ECHO  You can now run the application by double-clicking
ECHO  the 'run.bat' file.
ECHO.
PAUSE