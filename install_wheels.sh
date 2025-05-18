#!/bin/bash
# Script to install pre-built wheels for Python 3.12

# Upgrade pip
pip install --upgrade pip

# Install key packages from wheels
pip install --only-binary=:all: numpy matplotlib
pip install --only-binary=:all: torch torchaudio

# Skip greenlet on Python 3.12
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
  echo "Python 3.12 detected, skipping greenlet installation"
  # Install SQLAlchemy without greenlet
  pip install sqlalchemy --no-deps
else
  pip install greenlet==2.0.2
  pip install sqlalchemy==2.0.23
fi

# Install the rest of the requirements
pip install -r requirements.txt --no-deps || pip install -r requirements.txt

echo "Installation complete" 