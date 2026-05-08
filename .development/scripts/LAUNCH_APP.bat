@echo off
REM Quick launcher for HOGE web interface
echo.
echo ========================================================================
echo HOGE Framework - Interactive Web Interface
echo ========================================================================
echo.
echo Starting Streamlit app at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Launch Streamlit
streamlit run app.py

pause
