@echo off
title Market Research Engine
echo ========================================
echo   Market Research Engine - Launcher
echo ========================================
echo.

REM Start the trading engine in a separate window
echo [1/2] Starting trading engine...
start "Trading Engine" cmd /k "cd /d C:\Users\dezon\market-research-engine && python main.py"

REM Wait 3 seconds for engine to begin initializing
timeout /t 3 /nobreak >nul

REM Start Streamlit dashboard
echo [2/2] Starting web dashboard...
start "Dashboard" cmd /k "cd /d C:\Users\dezon\market-research-engine && streamlit run streamlit_app.py"

REM Wait for Streamlit to start then open browser
timeout /t 4 /nobreak >nul
echo Opening browser...
start http://localhost:8501

echo.
echo Both windows are running.
echo - Trading engine: see the "Trading Engine" terminal window
echo - Dashboard:      http://localhost:8501 in your browser
echo.
echo Close both terminal windows to shut everything down.
pause
