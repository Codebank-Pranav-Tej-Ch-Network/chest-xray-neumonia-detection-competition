# High-Resolution AMP GPU Pipeline (`CS24B057_gpu.py` & `CS24B057_gpu.ipynb`)

**Author:** Ch Venkata Sai Pranav Tej | **Roll No:** CS24B057  
**Competition Result:** **#1 on Public Leaderboard (`0.99640`) & #1 on Final Private Leaderboard (`0.99029`)**

## Overview
This directory contains the winning **High-Resolution Automatic Mixed Precision (AMP) GPU Pipeline**, verified on an **NVIDIA RTX A4000 (16GB VRAM)** workstation and compatible with **Google Colab Pro (L4 / A100)**.

### Architecture & Anti-Overfitting Design
1. **3-Backbone High-Resolution Visual Ensemble (`320x320` Resolution):**
   - **`DenseNet121`** (`0.45` weight): Dense feature reuse across convolutional blocks captures both fine interstitial markings and coarse lobar consolidation (`98.46%` standalone Val Acc at `320x320`).
   - **`ResNet50`** (`0.35` weight): Residual bottleneck architecture providing complementary high-level semantic features (`97.44%` standalone Val Acc).
   - **`CustomResAttNet`** (`0.20` weight): Custom 4-stage Residual CNN with Squeeze-and-Excitation (`SEBlock`) channel attention trained from scratch (`96.75%` Val Acc at `224x224`, `95.73%` at `320x320`).
   - **3-View Test-Time Augmentation (TTA):** Averages predictions across the center view and $\pm 4$-pixel horizontal shifts.
2. **Regularization & Checkpoint Calibration:**
   - **Label Smoothing ($\epsilon = 0.05$):** Targets $0.95 y + 0.025$ inside class-weighted `BCEWithLogitsLoss` (`pos_weight = 0.3704`) prevent late-epoch logit saturation.
   - **Differential Learning Rates:** `0.25x` base learning rate on pretrained `DenseNet121`/`ResNet50` convolutional trunks and `1.0x` on projection/classification heads (`AdamW`, `weight_decay = 5e-3`, `CosineAnnealingLR`).
   - **Validation-Loss Tie-Breaking:** Selects the checkpoint maximizing `(round(va_acc, 5), -va_loss)` so early well-calibrated checkpoints (`Epoch 9`) are never overwritten by late-epoch checkpoints.
   - **Automatic Per-Backbone Caching (`gpu_cache_{arch}_{img_size}.npz`):** Saves out-of-sample probabilities and embeddings immediately after each backbone finishes training.
3. **Leak-Free Out-of-Fold Stacking & Three Output Modes:**
   - **`submission_gpu_pure_visual.csv` (Pure Pixels Only):** Combines `320x320` GPU CNNs (`0.72`) + pure-pixel SciPy HOG/Contrast/Symmetry `PCA + RBF-SVM + LR` (`0.28`) with zero header metadata and zero sequence smoothing (`98.63%` Val Acc, `0.97482` Public LB).
   - **`submission_gpu_per_image.csv` (Per-Image Hybrid, `alpha = 0.0`):** Combines `320x320` GPU CNNs (`0.55`) + 5-Fold Stratified CV Radiographic/Geometry Stacker (`0.45`, `99.05%` OOF Acc across all 5,269 images) with zero sequence smoothing (`99.15%` Val Acc, `0.98201` Public LB).
   - **`submission.csv` / `submission_gpu_calibrated_v2.csv` (Prevalence-Aware Calibrated Hybrid):** Applies an **Automatic Lag-1 Autocorrelation Gate (`r1 > 0.50`)** and prevalence-aware prior weights (`w = 0.36` when `prior > 0.75`, `w = 0.28` otherwise), achieving **`99.49%` Val Acc (`100.0%` Recall, `0.9965` F1)**, **`0.99640` (`277/278`) on the Public Leaderboard**, and **`0.99029` (`306/309`) on the Final Private Leaderboard**.

### Execution
```bash
# Activate Python 3.11 GPU environment
source gpu/.venv/bin/activate

# Run 320x320 ensemble (pass --retrain to force retraining CNNs from scratch)
python gpu/CS24B057_gpu.py --data-dir . --arch ensemble --img-size 320 --epochs 15 --batch-size 32
```
