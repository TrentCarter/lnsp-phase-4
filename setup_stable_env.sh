#!/bin/bash
# Create stable training environment to avoid macOS ARM crashes
# Based on: PyTorch 2.5.0 + Python 3.11 (avoid 2.7.x + 3.13 edge cases)

set -e

echo "=========================================="
echo "STABLE TRAINING ENVIRONMENT SETUP"
echo "=========================================="
echo ""
echo "Target: Python 3.11 + PyTorch 2.5.0 + FAISS 1.7.4"
echo "Reason: Avoid PyTorch 2.7.1 + Python 3.13 ARM64 bugs"
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda detected, using conda for clean environment"
    echo ""

    # Create new environment
    echo "Creating conda environment: twotower311..."
    conda create -n twotower311 python=3.11 -y

    echo ""
    echo "Installing packages..."
    conda run -n twotower311 pip install \
        "torch==2.5.0" \
        "faiss-cpu==1.7.4" \
        "numpy==2.0.2" \
        "tqdm>=4.65.0" \
        "pyyaml>=6.0"

    echo ""
    echo "✓ Conda environment created: twotower311"
    echo ""
    echo "To activate:"
    echo "  conda activate twotower311"

else
    echo "⚠️  Conda not found, using venv with specific Python version"
    echo ""

    # Check if python3.11 is available
    if ! command -v python3.11 &> /dev/null; then
        echo "✗ Python 3.11 not found!"
        echo ""
        echo "Install Python 3.11:"
        echo "  brew install python@3.11"
        echo ""
        exit 1
    fi

    echo "Creating venv: .venv311..."
    python3.11 -m venv .venv311

    echo ""
    echo "Installing packages..."
    .venv311/bin/pip install --upgrade pip
    .venv311/bin/pip install \
        "torch==2.5.0" \
        "faiss-cpu==1.7.4" \
        "numpy==2.0.2" \
        "tqdm>=4.65.0" \
        "pyyaml>=6.0"

    echo ""
    echo "✓ Virtual environment created: .venv311"
    echo ""
    echo "To activate:"
    echo "  source .venv311/bin/activate"
fi

echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Activate environment (see above)"
echo "2. Run stable training:"
echo "   bash launch_v4_stable.sh"
echo ""
