@echo off
TITLE Image-to-Prompt AI - Runner
COLOR 0B

ECHO.
ECHO  ==================================================
ECHO      Starting Image-to-Prompt AI Application
ECHO  ==================================================
ECHO.

:: Change directory to the script's location
cd /d "%~dp0"

:: Check if the virtual environment exists. If not, instruct user to run install.bat
IF NOT EXIST venv\Scripts\activate.bat (
    COLOR 0C
    ECHO.
    ECHO  ERROR: Virtual environment not found.
    ECHO  Please run 'install.bat' first to set up the application.
    ECHO.
    PAUSE
    EXIT /B
)

:: Activate the virtual environment
ECHO [1/2] Activating virtual environment...
CALL venv\Scripts\activate.bat
ECHO.

:: Run the Streamlit app
ECHO [2/2] Starting the Streamlit server...
ECHO  Your web browser should open with the application shortly.
ECHO.
ECHO  To stop the application, close the browser tab and
ECHO  press CTRL+C in this window.
ECHO.

streamlit run app.py

ECHO.
ECHO  Application server has been stopped.
ECHO.
PAUSE