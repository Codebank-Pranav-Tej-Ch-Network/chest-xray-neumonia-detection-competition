# Local CPU Pipeline (`CS24B057_cpu.py` & `CS24B057_cpu.ipynb`)

**Author:** Ch Venkata Sai Pranav Tej | **Roll No:** CS24B057

## Overview
This folder contains the standalone **CPU Pipeline** engineered to run rapidly on any standard laptop without a discrete GPU (`~48-190s` end-to-end runtime).

### Key Technical Components
1. **Multi-Threaded Image & Header Extraction (`load_split`):**
   - Uses `ThreadPoolExecutor(max_workers=16)` and Pillow's fast decoder (`im.draft("L", (128, 128))`) to extract `64x64` normalized pixel matrices alongside 15 image geometry and quantization features, caching them to `../features_cache.npz`.
2. **Custom 3-Block PyTorch CNN (`ChestXRayCNN`):**
   - Lightweight convolutional feature extractor (`Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> AdaptiveAvgPool2d -> 64-D Embedding`) trained with on-the-fly spatial shift, contrast scaling, and Gaussian noise augmentation.
   - Standalone Validation Accuracy: **`95.73%` (`F1 = 0.9706`)**.
3. **Vectorized Pure-SciPy Radiographic Descriptor (`extract_multiscale_features`):**
   - Zero OpenCV (`cv2`) dependency.
   - Extracts `16x16` local adaptive contrast normalization, multi-scale `8x8` and `16x16` cell Histogram of Oriented Gradients (HOG) with `2x2` L2 block normalization, Sobel gradient magnitude grids, 24-bin intensity histograms, and bilateral left-vs-right lung symmetry ratios (`2,441` total features).
4. **5-Fold Stratified Out-of-Fold Stacking (`SVC + LogisticRegression + MLP + HistGradientBoosting + CNN`):**
   - Pools all `5,269` labeled radiographs (`train + val`) under 5-Fold Stratified Cross-Validation.
   - **5-Fold Out-of-Fold Per-Image Accuracy (All 5,269 images):** **`98.14%` (`F1 = 0.9873`)**
   - **Kaggle Public Leaderboard (Per-Image Only, `alpha = 0.0`):** **`0.97482` (`271 / 278`)**
   - **Kaggle Public Leaderboard (Calibrated, `alpha = 0.32`):** **`0.98201` (`273 / 278`)**

### Execution
```bash
python cpu/CS24B057_cpu.py
```
