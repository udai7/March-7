#!/bin/bash
# Setup script for CO2 Reduction AI Agent (Linux/Mac)
# This script checks prerequisites, creates virtual environment, installs dependencies,
# and initializes the vector store.

set -e  # Exit on error

echo "============================================================"
echo "CO2 Reduction AI Agent - Setup Script"
echo "============================================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Check if version is 3.9+
MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9+ is required, found $PYTHON_VERSION"
    exit 1
fi

echo "OK: Python $PYTHON_VERSION meets requirements (3.9+)"
echo ""

# Check if virtual environment already exists
echo "[2/6] Setting up virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n): " RECREATE
    if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
        echo "Creating new virtual environment..."
        python3 -m venv venv
    else
        echo "Using existing virtual environment"
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
echo "OK: Virtual environment ready"
echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "OK: Virtual environment activated"
echo ""

# Upgrade pip
echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "OK: pip upgraded"
echo ""

# Install dependencies
echo "[5/6] Installing dependencies from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found"
    exit 1
fi

pip install -r requirements.txt
echo "OK: Dependencies installed"
echo ""

# Initialize vector store
echo "[6/6] Initializing vector store..."
python scripts/init_vector_store.py
echo "OK: Vector store initialized"
echo ""

# Check Ollama installation
echo "============================================================"
echo "Checking Ollama installation..."
echo "============================================================"
if ! command -v ollama &> /dev/null; then
    echo "WARNING: Ollama is not installed or not in PATH"
    echo ""
    echo "Ollama is required to run the AI agent with local LLMs."
    echo "Please install Ollama from: https://ollama.ai/"
    echo ""
    echo "After installing Ollama, run:"
    echo "  ollama pull llama3"
    echo "or"
    echo "  ollama pull mistral"
    echo ""
else
    OLLAMA_VERSION=$(ollama --version 2>&1)
    echo "OK: Ollama is installed - $OLLAMA_VERSION"
    echo ""
    
    # Check if llama3 or mistral model is available
    echo "Checking for available models..."
    if ollama list | grep -qi "llama3"; then
        echo "OK: llama3 model is available"
    elif ollama list | grep -qi "mistral"; then
        echo "OK: mistral model is available"
    else
        echo "WARNING: No compatible model found"
        echo "Please pull a model with: ollama pull llama3"
    fi
fi
echo ""

# Setup complete
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the Streamlit app: streamlit run app.py"
echo ""
echo "Make sure Ollama is running before starting the app."
echo ""
