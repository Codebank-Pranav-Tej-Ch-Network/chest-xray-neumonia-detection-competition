# ==============================================================================
# CS24B057 - Ch Venkata Sai Pranav Tej
# EE506L PRML 2026 Assignment 4 (1st Place Winning Solution)
# Automated Cross-Platform Environment Setup & Reproduction Script (Windows)
# ==============================================================================
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=============================================================================="
Write-Host "CS24B057 | Setting up reproducible Python environment..."
Write-Host "=============================================================================="

if (Test-Path ".venv_cs24b057\Scripts\Activate.ps1") {
    Write-Host "Activating existing environment at .venv_cs24b057..."
    . .\.venv_cs24b057\Scripts\Activate.ps1
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Creating Python 3.11 virtual environment via uv..."
    uv venv .venv_cs24b057 --python 3.11
    . .\.venv_cs24b057\Scripts\Activate.ps1
    uv pip install -r requirements.txt
} else {
    Write-Host "Creating Python virtual environment via python -m venv..."
    python -m venv .venv_cs24b057
    . .\.venv_cs24b057\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
}

Write-Host "=============================================================================="
Write-Host "CS24B057 | Running Reproducible Pipeline..."
Write-Host "=============================================================================="
python CS24B057.py $args
