@echo off
REM Setup script for CO2 Reduction AI Agent (Windows)
REM This script checks prerequisites, creates virtual environment, installs dependencies,
REM and initializes the vector store.

echo ============================================================
echo CO2 Reduction AI Agent - Setup Script
echo ============================================================
echo.

REM Check Python version
echo [1/6] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://www.python.org/
    exit /b 1
)

REM Get Python version and check if it's 3.9+
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% LSS 3 (
    echo ERROR: Python 3.9+ is required, found %PYTHON_VERSION%
    exit /b 1
)
if %MAJOR% EQU 3 if %MINOR% LSS 9 (
    echo ERROR: Python 3.9+ is required, found %PYTHON_VERSION%
    exit /b 1
)

echo OK: Python %PYTHON_VERSION% meets requirements (3.9+)
echo.

REM Check if virtual environment already exists
echo [2/6] Setting up virtual environment...
if exist venv (
    echo Virtual environment already exists
    set /p RECREATE="Do you want to recreate it? (y/n): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
        echo Creating new virtual environment...
        python -m venv venv
    ) else (
        echo Using existing virtual environment
    )
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        exit /b 1
    )
)
echo OK: Virtual environment ready
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)
echo OK: Virtual environment activated
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
) else (
    echo OK: pip upgraded
)
echo.

REM Install dependencies
echo [5/6] Installing dependencies from requirements.txt...
if not exist requirements.txt (
    echo ERROR: requirements.txt not found
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check requirements.txt and try again
    exit /b 1
)
echo OK: Dependencies installed
echo.

REM Initialize vector store
echo [6/6] Initializing vector store...
python scripts\init_vector_store.py
if errorlevel 1 (
    echo ERROR: Failed to initialize vector store
    echo Please check that data/sustainability_tips.txt exists
    exit /b 1
)
echo OK: Vector store initialized
echo.

REM Check Ollama installation
echo ============================================================
echo Checking Ollama installation...
echo ============================================================
ollama --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Ollama is not installed or not in PATH
    echo.
    echo Ollama is required to run the AI agent with local LLMs.
    echo Please install Ollama from: https://ollama.ai/
    echo.
    echo After installing Ollama, run:
    echo   ollama pull llama3
    echo or
    echo   ollama pull mistral
    echo.
) else (
    for /f "tokens=*" %%i in ('ollama --version 2^>^&1') do set OLLAMA_VERSION=%%i
    echo OK: Ollama is installed - %OLLAMA_VERSION%
    echo.
    
    REM Check if llama3 or mistral model is available
    echo Checking for available models...
    ollama list >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Could not check Ollama models
    ) else (
        ollama list | findstr /i "llama3" >nul 2>&1
        if not errorlevel 1 (
            echo OK: llama3 model is available
        ) else (
            ollama list | findstr /i "mistral" >nul 2>&1
            if not errorlevel 1 (
                echo OK: mistral model is available
            ) else (
                echo WARNING: No compatible model found
                echo Please pull a model with: ollama pull llama3
            )
        )
    )
)
echo.

REM Setup complete
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To start the application:
echo   1. Activate the virtual environment: venv\Scripts\activate.bat
echo   2. Run the Streamlit app: streamlit run app.py
echo.
echo Make sure Ollama is running before starting the app.
echo.

pause
