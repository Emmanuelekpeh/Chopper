@echo off
REM Setup script for Chopper Audio Generator on Windows

REM Check Python version
python --version
FOR /F "tokens=2" %%G IN ('python -c "import sys; print(sys.version_info.major)"') DO SET PY_MAJOR=%%G
FOR /F "tokens=2" %%G IN ('python -c "import sys; print(sys.version_info.minor)"') DO SET PY_MINOR=%%G
echo Python version: %PY_MAJOR%.%PY_MINOR%

REM Set up virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies with special handling for Python 3.12
echo Installing dependencies...
pip install --upgrade pip setuptools wheel

REM Handle greenlet for Python 3.12
if "%PY_MAJOR%.%PY_MINOR%"=="3.12" (
    echo Python 3.12 detected, using special installation procedure...
    
    REM Install key packages from wheels
    pip install --only-binary=:all: numpy matplotlib
    pip install --only-binary=:all: torch torchaudio
    
    REM Install SQLAlchemy without greenlet
    pip install sqlalchemy --no-deps
    
    REM Install remaining requirements
    pip install -r requirements.txt
) else (
    REM Normal installation for Python 3.11 and below
    pip install -r requirements.txt -c constraints.txt
)

REM Create necessary directories
mkdir data\raw data\processed data\processed\mel data\processed\segments models output 2>nul

echo.
echo Setup complete! Run the application with:
echo call venv\Scripts\activate.bat ^&^& streamlit run streamlit_app.py 