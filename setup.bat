@echo off
REM MDT AI Copilot — Setup Script (Windows)
REM Run this from the project root directory.

echo ============================================================
echo   MDT AI Copilot — Setup
echo ============================================================
echo.

REM Install Python dependencies
echo [1/5] Installing Python dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: pip install failed. Make sure Python 3.11+ is in PATH.
    pause
    exit /b 1
)
echo       Done.

REM Create data directory
echo [2/5] Creating data directory...
if not exist "data" mkdir data
echo       Done.

REM Init DB and generate synthetic data
echo [3/5] Initialising database and generating synthetic data...
python scripts\demo_phase1.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Phase 1 demo failed.
    pause
    exit /b 1
)

REM Run ML evaluation
echo [4/5] Running ML evaluation (Phase 2)...
python scripts\demo_phase2.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Phase 2 demo failed.
    pause
    exit /b 1
)

REM Run tests
echo [5/5] Running unit tests...
python -m pytest tests\ -v
if %ERRORLEVEL% neq 0 (
    echo WARNING: Some tests failed. Check output above.
)

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Copy .env.example to .env and add your API keys
echo   2. Run: python -m rag.build_index     (build RAG index)
echo   3. Run: python -m copilot.cli          (CLI chat)
echo   4. Run: streamlit run dashboard\app.py (Streamlit dashboard)
echo   5. Or:  python scripts\demo_full.py   (full demo)
echo.
pause
