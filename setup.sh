#!/bin/bash
# Setup script for Chopper Audio Generator

# Make script directory the working directory
cd "$(dirname "$0")"

# Check Python version
python_version=$(python3 --version)
echo "Using $python_version"
py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $py_version"

# Set up virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies based on Python version
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Handle greenlet for Python 3.12
if [ "$py_version" = "3.12" ]; then
    echo "Python 3.12 detected, using special installation procedure..."
    
    # Install key packages from wheels
    pip install --only-binary=:all: numpy matplotlib
    pip install --only-binary=:all: torch torchaudio
    
    # Install SQLAlchemy without greenlet
    pip install sqlalchemy --no-deps
    
    # Install remaining requirements
    pip install -r requirements.txt
else
    # Normal installation for Python 3.11 and below
    pip install -r requirements.txt -c constraints.txt
fi

# Create necessary directories
mkdir -p data/raw data/processed data/processed/mel data/processed/segments models output

echo "Setup complete! Run the application with:"
echo "source venv/bin/activate && streamlit run streamlit_app.py" 