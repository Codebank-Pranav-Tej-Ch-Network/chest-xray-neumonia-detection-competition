# Classical PRML Baselines & Systematic Ablation Suite (`CS24B057`)

**Author:** Ch Venkata Sai Pranav Tej | **Roll No:** CS24B057 | **Course:** EE506L PRML 2026

This directory contains the foundational implementations from **Programming Assignment 1 (PA1)** and **Programming Assignment 2 (PA2)** that form the starting point for our Capstone progression:

1. **`pa1_logistic_regression/`**:
   - `CS24B057.py`: Pure NumPy implementation of Vectorized Logistic Regression from scratch (`sigmoid`, `compute_loss`, `compute_gradients`, `update_parameters`, `fit_logistic_regression`, `compute_metrics`) on `4,096`-dimensional (`64x64`) chest X-ray vectors.
   - Achieves **`92.99%` Validation Accuracy** and **`0.9529` Validation F1-Score** (`Precision = 0.9463`, `Recall = 0.9555`) after 1,000 full-batch gradient descent epochs ($\eta = 0.01$).
   - Diagnostic visualisations: `sample_images.png`, `convergence_plot.png`, and `weight_heatmap.png`.

2. **`pa2_pca/`**:
   - `CS24B057.py`: Pure NumPy implementation of Unsupervised Principal Component Analysis (PCA) from scratch using symmetric covariance eigendecomposition (`np.linalg.eigh`).
   - Selecting $k = 237$ principal components retains **`95.0159%` of total training variance** while compressing dimensionality by **94.2%** (`4,096 -> 237`), delivering a **6.32x speedup** (`28x` with L2 regularization) at virtually identical validation F1 (`0.9532` vs `0.9535`).
   - Includes the compiled report `CS24B057.pdf` and `build_report_pa2.py`.

3. **Capstone Classical Ablation Suite (`../run_ablations.py`)**:
   - Evaluates four progressively stronger classical PRML models on the exact same `Train (4,684)` and `Val (585)` splits and generates their corresponding Kaggle test submission files in `../submissions/`:
     - **Stage 1 (`submission_baseline1_lr.csv`):** L2-Regularized Logistic Regression ($C = 0.1$) on 4,096-D raw pixels (`Val Acc: 0.9436 | Val F1: 0.9617`).
     - **Stage 2 (`submission_baseline2_pca_lr.csv`):** PCA ($k = 237$) + L2 Logistic Regression (`Val Acc: 0.9436 | Val F1: 0.9617 | Fit Time: 0.20s`).
     - **Stage 3 (`submission_baseline3_pca_svm.csv`):** PCA ($k = 237$) + Class-Weighted Soft-Margin RBF SVM ($C = 2.5$) (`Val Acc: 0.9453 | Val Recall: 0.9789 | Val F1: 0.9631`).
     - **Stage 4 (`submission_baseline4_hog_pca_svm.csv`):** Vectorized SciPy Multi-Scale HOG + Local Contrast + Bilateral Lung Symmetry + PCA + Soft-Margin RBF SVM (`Val Acc: 0.9829 | Val F1: 0.9883`).
