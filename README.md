# PRML 2026: Pediatric Chest X-Ray Pneumonia Detection (CS24B057)

**Student:** Ch Venkata Sai Pranav Tej  
**Roll Number:** CS24B057  

## Directory Structure

```text
Kaggle-comp/
|-- cpu/
|   |-- CS24B057_cpu.py          # Local CPU training and inference pipeline
|   |-- CS24B057_cpu.ipynb       # Runnable CPU notebook
|   |-- requirements-cpu.txt     # Pinned CPU pip dependencies
|   |-- pyproject.toml           # uv project config (CPU)
|-- gpu/
|   |-- CS24B057_gpu.py          # High-resolution AMP GPU pipeline (Colab L4/A100 & SPCV Lab)
|   |-- CS24B057_gpu.ipynb       # Runnable Colab Pro / SPCV Lab GPU notebook
|   |-- requirements-gpu.txt     # CUDA 12.4 pip requirements
|   |-- environment-gpu.yml      # Conda environment specification
|   |-- pyproject.toml           # uv project config (CUDA 12.4)
|-- submissions/
|   |-- submission_iter1_full.csv
|   |-- submission_iter1_99pct.csv
|   |-- submission_iter1_per_image.csv
|-- sample_submission.csv
|-- submission.csv               # Primary submission file ready for Kaggle upload
|-- PRML 2026 - Pneumonia Detection.pdf
```

## 1. Running the CPU Version Locally

Using standard Python or `uv`:
```powershell
cd cpu
uv venv
uv pip install -r requirements-cpu.txt
uv run python CS24B057_cpu.py
```

## 2. Running the GPU Version (SPCV Lab 16GB GPU or Google Colab Pro)

### Option A: SPCV Lab Workstation with `uv`
```bash
cd gpu
uv venv --python 3.11
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -r requirements-gpu.txt
python CS24B057_gpu.py --arch ensemble --img-size 224 --batch-size 64 --epochs 12 --lr 3e-4
```

### Option B: SPCV Lab Workstation with `conda`
```bash
cd gpu
conda env create -f environment-gpu.yml
conda activate prml-pa4-gpu
python CS24B057_gpu.py --arch ensemble --img-size 224 --batch-size 64 --epochs 12 --lr 3e-4
```

### Option C: Google Colab Pro (L4 / A100)
1. Upload `gpu/CS24B057_gpu.ipynb` and `gpu/CS24B057_gpu.py` to Colab (or clone this private GitHub repository inside Colab).
2. Select **Runtime -> Change runtime type -> GPU (L4 or A100)**.
3. Run all cells in `CS24B057_gpu.ipynb`.
