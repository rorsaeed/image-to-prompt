@echo off
TITLE Image-to-Prompt AI - Updater
COLOR 0B

ECHO.
ECHO  =======================================================
ECHO      Image-to-Prompt AI Application Updater
ECHO  =======================================================
ECHO.
ECHO  This script will download the latest version of the app
ECHO  from GitHub and update any required packages.
ECHO.

:: Change directory to the script's location
cd /d "%~dp0"

:: Step 1: Verify this is a Git repository
ECHO [1/5] Verifying Git repository...
IF NOT EXIST .git (
    COLOR 0C
    ECHO.
    ECHO  ERROR: This does not appear to be a Git repository.
    ECHO  This script can only be used if you originally cloned the
    ECHO  project using 'git clone'.
    ECHO.
    ECHO  Please download a fresh copy from the GitHub page.
    ECHO.
    PAUSE
    EXIT /B
)
ECHO   Repository found.
ECHO.

:: Step 2: Check if Git is installed
ECHO [2/5] Checking for Git installation...
git --version >nul 2>nul
IF %errorlevel% NEQ 0 (
    COLOR 0C
    ECHO.
    ECHO  ERROR: Git is not found in your system's PATH.
    ECHO  Please install Git from https://git-scm.com/downloads
    ECHO  and ensure it's added to your PATH during installation.
    ECHO.
    PAUSE
    EXIT /B
)
ECHO   Git is installed.
ECHO.

:: Step 3: Pull the latest changes from the repository
ECHO [3/5] Pulling latest changes from the repository...
git pull
IF %errorlevel% NEQ 0 (
    COLOR 0C
    ECHO.
    ECHO  ERROR: 'git pull' failed.
    ECHO  This can happen if you have made local changes to the files.
    ECHO  Please resolve the conflicts manually or download a fresh copy.
    ECHO  You may also want to check your internet connection.
    ECHO.
    PAUSE
    EXIT /B
)
ECHO   Successfully pulled updates.
ECHO.

:: Step 4: Check if the virtual environment exists
ECHO [4/5] Checking for virtual environment...
IF NOT EXIST venv\Scripts\activate.bat (
    COLOR 0C
    ECHO.
    ECHO  ERROR: Virtual environment not found.
    ECHO  Please run 'install.bat' first to set up the application.
    ECHO.
    PAUSE
    EXIT /B
)
ECHO   Virtual environment found.
ECHO.

:: Step 5: Update Python packages
ECHO [5/5] Activating environment and updating Python packages...
ECHO   This may take a few moments if new packages were added.
ECHO.
CALL venv\Scripts\activate.bat
pip install -r requirements.txt
IF %errorlevel% NEQ 0 (
    COLOR 0C
    ECHO.
    ECHO  ERROR: Failed to update required packages.
    ECHO  Please check your internet connection and the repository for issues.
    ECHO.
    PAUSE
    EXIT /B
)
ECHO.

:: Finalization
COLOR 0A
ECHO  =======================================================
ECHO      UPDATE SUCCESSFUL!
ECHO  =======================================================
ECHO.
ECHO  The application is now up-to-date. You can run it
ECHO  using the 'run.bat' file.
ECHO.
PAUSE