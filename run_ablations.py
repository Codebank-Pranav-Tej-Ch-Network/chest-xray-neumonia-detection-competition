"""
CS24B057
PRML 2026 Assignment 4 - Systematic Ablation Study & Classical Baselines
Generates validation metrics and Kaggle submission CSVs for:
  1. Baseline 1: Raw Pixels (4096-D) + Logistic Regression (PA1 Baseline)
  2. Baseline 2: PCA (k=237, 95% variance) + Logistic Regression (PA2 Baseline)
  3. Baseline 3: PCA (k=237, 95% variance) + Class-Weighted Soft-Margin RBF SVM
  4. Baseline 4: Multi-Scale HOG + Local Contrast + PCA + Soft-Margin SVM (Per-Image Visual)
"""

import os
import csv
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu"))
from CS24B057_cpu import resolve_paths, load_split, extract_multiscale_features, compute_metrics, save_submission_csv


def measure_model(name, clf, X_tr, y_tr, X_va, y_va, X_te):
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    tr_preds = clf.predict(X_tr)
    va_preds = clf.predict(X_va)
    te_preds = clf.predict(X_te)

    m_tr = compute_metrics(y_tr, tr_preds)
    m_va = compute_metrics(y_va, va_preds)
    print(
        f"{name:<42} | Time: {fit_time:5.2f}s | "
        f"Train Acc: {m_tr['accuracy']:.4f} | "
        f"Val Acc: {m_va['accuracy']:.4f} | "
        f"Val Prec: {m_va['precision']:.4f} | "
        f"Val Rec: {m_va['recall']:.4f} | "
        f"Val F1: {m_va['f1']:.4f}"
    )
    return m_tr, m_va, te_preds, fit_time


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, "features_cache.npz")
    sub_dir = os.path.join(base_dir, "submissions")
    os.makedirs(sub_dir, exist_ok=True)

    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        M_tr, P_tr, y_tr = d["M_tr"], d["P_tr"], d["y_tr"]
        M_va, P_va, y_va = d["M_va"], d["P_va"], d["y_va"]
        M_te, P_te, test_names = d["M_te"], d["P_te"], list(d["test_names"])
    else:
        print("Scanning dataset splits to build features_cache.npz...")
        train_dir, val_dir, test_dir = resolve_paths(base_dir)
        M_tr, P_tr, y_tr, _ = load_split(train_dir, is_labeled=True)
        M_va, P_va, y_va, _ = load_split(val_dir, is_labeled=True)
        M_te, P_te, _, test_names = load_split(test_dir, is_labeled=False)
        np.savez(
            cache_path,
            M_tr=M_tr, P_tr=P_tr, y_tr=y_tr,
            M_va=M_va, P_va=P_va, y_va=y_va,
            M_te=M_te, P_te=P_te, test_names=np.array(test_names)
        )

    print("=" * 125)
    print("PRML 2026 PA4 - Step-by-Step Ablation Progression (Train=4684, Val=585, Test=587)")
    print("=" * 125)

    # 1. Baseline 1: Standard Logistic Regression on 4096-D Raw 64x64 Pixels (PA1 evolution)
    lr_raw = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    _, _, preds_b1, _ = measure_model(
        "1. Basic Logistic Regression (4096-D Raw)",
        lr_raw, P_tr, y_tr, P_va, y_va, P_te
    )
    save_submission_csv(os.path.join(sub_dir, "submission_baseline1_lr.csv"), test_names, preds_b1)

    # 2. Baseline 2: PCA (k=237, >=95% variance) + Logistic Regression (PA2 evolution)
    t_pca0 = time.time()
    mu = P_tr.mean(axis=0, keepdims=True)
    P_tr_c = P_tr - mu
    P_va_c = P_va - mu
    P_te_c = P_te - mu
    pca_237 = PCA(n_components=237, random_state=42)
    Z_tr_237 = pca_237.fit_transform(P_tr_c)
    Z_va_237 = pca_237.transform(P_va_c)
    Z_te_237 = pca_237.transform(P_te_c)
    var_retained = float(np.sum(pca_237.explained_variance_ratio_) * 100.0)
    print(f"   [PCA Info] k=237 retains {var_retained:.4f}% variance (computed in {time.time()-t_pca0:.2f}s)")

    lr_pca = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    _, _, preds_b2, _ = measure_model(
        "2. Basic PCA (k=237) + Logistic Regression",
        lr_pca, Z_tr_237, y_tr, Z_va_237, y_va, Z_te_237
    )
    save_submission_csv(os.path.join(sub_dir, "submission_baseline2_pca_lr.csv"), test_names, preds_b2)

    # 3. Baseline 2b: Class-Weighted PCA + Logistic Regression (handling 73%/27% imbalance)
    lr_pca_bal = LogisticRegression(C=0.1, class_weight={0: 2.0, 1: 1.0}, max_iter=500, random_state=42)
    measure_model(
        "2b. PCA (k=237) + Class-Weighted LR",
        lr_pca_bal, Z_tr_237, y_tr, Z_va_237, y_va, Z_te_237
    )

    # 4. Baseline 3: PCA (k=237) + Soft-Margin RBF SVM
    scaler_z = StandardScaler()
    Z_tr_s = scaler_z.fit_transform(Z_tr_237)
    Z_va_s = scaler_z.transform(Z_va_237)
    Z_te_s = scaler_z.transform(Z_te_237)

    svm_pca = SVC(C=2.5, kernel="rbf", class_weight={0: 2.0, 1: 1.0}, random_state=42)
    _, _, preds_b3, _ = measure_model(
        "3. Basic PCA (k=237) + Soft-Margin RBF SVM",
        svm_pca, Z_tr_s, y_tr, Z_va_s, y_va, Z_te_s
    )
    save_submission_csv(os.path.join(sub_dir, "submission_baseline3_pca_svm.csv"), test_names, preds_b3)

    # 5. Baseline 4: Multi-Scale HOG + Local Contrast + Lung Symmetry + PCA + Soft-Margin RBF SVM
    F_tr = extract_multiscale_features(M_tr, P_tr)
    F_va = extract_multiscale_features(M_va, P_va)
    F_te = extract_multiscale_features(M_te, P_te)

    sc_f = StandardScaler()
    F_tr_s = sc_f.fit_transform(F_tr)
    F_va_s = sc_f.transform(F_va)
    F_te_s = sc_f.transform(F_te)

    pca_f = PCA(n_components=180, random_state=42)
    ZF_tr = np.hstack([F_tr_s[:, :17], pca_f.fit_transform(F_tr_s[:, 17:])])
    ZF_va = np.hstack([F_va_s[:, :17], pca_f.transform(F_va_s[:, 17:])])
    ZF_te = np.hstack([F_te_s[:, :17], pca_f.transform(F_te_s[:, 17:])])

    svm_hog = SVC(C=2.2, kernel="rbf", class_weight={0: 2.0, 1: 1.0}, random_state=42)
    _, _, preds_b4, _ = measure_model(
        "4. HOG + Local Contrast + PCA + RBF SVM",
        svm_hog, ZF_tr, y_tr, ZF_va, y_va, ZF_te
    )
    save_submission_csv(os.path.join(sub_dir, "submission_baseline4_hog_pca_svm.csv"), test_names, preds_b4)

    print("=" * 125)
    print("Saved baseline submission files in submissions/:")
    for fname, p in [
        ("submission_baseline1_lr.csv", preds_b1),
        ("submission_baseline2_pca_lr.csv", preds_b2),
        ("submission_baseline3_pca_svm.csv", preds_b3),
        ("submission_baseline4_hog_pca_svm.csv", preds_b4),
    ]:
        print(f"  - {fname:<38} -> Normal={int((p==0).sum()):3d}, Pneumonia={int((p==1).sum()):3d}")


if __name__ == "__main__":
    main()
