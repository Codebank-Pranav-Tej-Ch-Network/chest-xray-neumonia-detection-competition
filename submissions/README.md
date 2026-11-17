# Complete Index of Generated Submissions & Leaderboard Log (`CS24B057`)

**Author:** Ch Venkata Sai Pranav Tej | **Roll No:** CS24B057

This directory archives all **13 submission CSV files** (`587` test rows each) generated across our classical PRML baselines, CPU iterations, and high-resolution RTX A4000 GPU runs:

| File Name | Pipeline Stage / Description | Predicted `Normal` (`0`) | Predicted `Pneumonia` (`1`) | Val Accuracy | Val F1 | Public LB (`278` imgs) | Private LB (`309` imgs) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `submission_baseline1_lr.csv` | Stage 1: L2 Logistic Regression on 4,096-D 64x64 pixels | 142 | 445 | `0.9436` | `0.9617` | Baseline | Baseline |
| `submission_baseline2_pca_lr.csv` | Stage 2: PCA ($k=237$, 95%+ var) + L2 Logistic Regression | 143 | 444 | `0.9436` | `0.9617` | Baseline | Baseline |
| `submission_baseline3_pca_svm.csv` | Stage 3: PCA ($k=237$) + Class-Weighted Soft-Margin RBF SVM | 144 | 443 | `0.9453` | `0.9631` | Baseline | Baseline |
| `submission_baseline4_hog_pca_svm.csv` | Stage 4: SciPy HOG + Local Contrast + Symmetry + PCA + RBF SVM | 156 | 431 | `0.9829` | `0.9883` | Baseline | Baseline |
| `submission_iter1_full.csv` | Iter 1 CPU: Sequence step baseline (`alpha = 0.62`) | 159 | 428 | `0.9829` | `0.9883` | `0.98201` (273/278) | -- |
| `submission_iter1_99pct.csv` | Iter 1 CPU: 64x64 CNN + PCA + RBF SVM Hybrid | 153 | 434 | `0.9829` | `0.9883` | `0.98201` (273/278) | -- |
| `submission_iter1_per_image.csv` | Iter 1 CPU: Per-Image only (`alpha = 0.0`) | 150 | 437 | `0.9675` | `0.9778` | -- | -- |
| `submission_iter2_hybrid.csv` | Iter 2 CPU: 5-Fold Stratified CV Stacker (`alpha = 0.32`) | 153 | 434 | `0.9846` | `0.9895` | `0.98201` (273/278) | -- |
| `submission_iter2_per_image.csv` | Iter 2 CPU: 5-Fold Stratified CV Per-Image (`alpha = 0.0`) | 151 | 436 | `0.9761` | `0.9836` | `0.97482` (271/278) | -- |
| `submission_gpu_pure_visual.csv` | Iter 3 GPU (`320x320`): Pure Pixels Only, Zero Metadata | 162 | 425 | `0.9863` | `0.9906` | `0.97482` (271/278) | -- |
| `submission_gpu_per_image.csv` | Iter 3 GPU (`320x320`): Per-Image Hybrid (`alpha = 0.0`) | 157 | 430 | `0.9915` | `0.9942` | `0.98201` (273/278) | Selected #2 |
| `submission_gpu_full.csv` | Iter 3 GPU (`320x320`): Symmetric Prior Hybrid (`alpha = 0.28`) | 157 | 430 | `0.9949` | `0.9965` | `0.99280` (276/278) | -- |
| **`submission_gpu_calibrated_v2.csv`** | **Final Winning GPU Hybrid (`320x320` + Prevalence Prior)** | **155** | **432** | **`0.9949`** | **`0.9965`** | **`0.99640` (277/278, #1)** | **`0.99029` (306/309, #1)** |
