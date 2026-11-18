#!/usr/bin/env bash
# ==============================================================================
# CS24B057 - Ch Venkata Sai Pranav Tej
# EE506L PRML 2026 Assignment 4 (1st Place Winning Solution)
# Automated Cross-Platform Environment Setup & Reproduction Script (Linux/macOS)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "CS24B057 | Setting up reproducible Python environment..."
echo "=============================================================================="

if [ -d "gpu/.venv" ]; then
    echo "Found existing virtual environment at gpu/.venv"
    source gpu/.venv/bin/activate
elif [ -d ".venv_cs24b057" ]; then
    echo "Found existing virtual environment at .venv_cs24b057"
    source .venv_cs24b057/bin/activate
elif command -v uv >/dev/null 2>&1; then
    echo "Creating Python 3.11 virtual environment using uv..."
    uv venv .venv_cs24b057 --python 3.11
    source .venv_cs24b057/bin/activate
    if command -v nvidia-smi >/dev/null 2>&1; then
        uv pip install -r gpu/requirements-gpu.txt || uv pip install -r requirements.txt
    else
        uv pip install -r requirements.txt
    fi
else
    echo "Creating Python virtual environment using python3 -m venv..."
    python3 -m venv .venv_cs24b057
    source .venv_cs24b057/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo "=============================================================================="
echo "CS24B057 | Running Reproducible Pipeline..."
echo "=============================================================================="
python CS24B057.py "$@"
