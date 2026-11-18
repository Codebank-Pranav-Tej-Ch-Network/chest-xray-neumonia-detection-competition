"""
CS24B057
EE506L Programming Assignment 4
Ablation study comparing classical models:
  1. Logistic Regression on 4096-D raw pixels
  2. PCA (k=237) + Logistic Regression
  3. PCA (k=237) + Soft-Margin RBF SVM
  4. Multi-scale HOG + Local Contrast + PCA + Soft-Margin RBF SVM
"""

import os
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from CS24B057 import resolve_paths, scan_split, extract_handcrafted_features, compute_metrics


def evaluate_baseline(name, clf, X_tr, y_tr, X_va, y_va):
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    tr_preds = clf.predict(X_tr)
    va_preds = clf.predict(X_va)

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
    return m_tr, m_va, fit_time


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = os.path.join(base_dir, "features_cache.npz")

    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        M_tr, P_tr, y_tr = d["M_tr"], d["P_tr"], d["y_tr"]
        M_va, P_va, y_va = d["M_va"], d["P_va"], d["y_va"]
        M_te, P_te, test_names = d["M_te"], d["P_te"], list(d["test_names"])
    else:
        print("Reading dataset splits...")
        train_dir, val_dir, test_dir = resolve_paths(base_dir)
        _, M_tr, P_tr, y_tr, _ = scan_split(train_dir, is_labeled=True)
        _, M_va, P_va, y_va, _ = scan_split(val_dir, is_labeled=True)
        _, M_te, P_te, _, test_names = scan_split(test_dir, is_labeled=False)
        np.savez(
            cache_path,
            M_tr=M_tr, P_tr=P_tr, y_tr=y_tr,
            M_va=M_va, P_va=P_va, y_va=y_va,
            M_te=M_te, P_te=P_te, test_names=np.array(test_names)
        )

    print("-" * 120)
    print("EE506L PA4 Ablation Results (Train=4684, Val=585)")
    print("-" * 120)

    # 1. Logistic Regression on 4096-D raw 64x64 pixels
    lr_raw = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    evaluate_baseline("1. Logistic Regression (4096-D Raw)", lr_raw, P_tr, y_tr, P_va, y_va)

    # 2. PCA (k=237) + Logistic Regression
    mu = P_tr.mean(axis=0, keepdims=True)
    pca_237 = PCA(n_components=237, random_state=42)
    Z_tr_237 = pca_237.fit_transform(P_tr - mu)
    Z_va_237 = pca_237.transform(P_va - mu)

    lr_pca = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    evaluate_baseline("2. PCA (k=237) + Logistic Regression", lr_pca, Z_tr_237, y_tr, Z_va_237, y_va)

    # 2b. PCA (k=237) + Class-Weighted Logistic Regression
    lr_pca_bal = LogisticRegression(C=0.1, class_weight={0: 2.0, 1: 1.0}, max_iter=500, random_state=42)
    evaluate_baseline("2b. PCA (k=237) + Weighted LogReg", lr_pca_bal, Z_tr_237, y_tr, Z_va_237, y_va)

    # 3. PCA (k=237) + Soft-Margin RBF SVM
    scaler_z = StandardScaler()
    Z_tr_s = scaler_z.fit_transform(Z_tr_237)
    Z_va_s = scaler_z.transform(Z_va_237)

    svm_pca = SVC(C=2.5, kernel="rbf", class_weight={0: 2.0, 1: 1.0}, random_state=42)
    evaluate_baseline("3. PCA (k=237) + Soft-Margin RBF SVM", svm_pca, Z_tr_s, y_tr, Z_va_s, y_va)

    # 4. HOG + Local Contrast + Lung Symmetry + PCA + Soft-Margin RBF SVM
    F_tr = extract_handcrafted_features(M_tr, P_tr)
    F_va = extract_handcrafted_features(M_va, P_va)

    sc_f = StandardScaler()
    F_tr_s = sc_f.fit_transform(F_tr)
    F_va_s = sc_f.transform(F_va)

    pca_f = PCA(n_components=180, random_state=42)
    ZF_tr = np.hstack([F_tr_s[:, :17], pca_f.fit_transform(F_tr_s[:, 17:])])
    ZF_va = np.hstack([F_va_s[:, :17], pca_f.transform(F_va_s[:, 17:])])

    svm_hog = SVC(C=2.2, kernel="rbf", class_weight={0: 2.0, 1: 1.0}, random_state=42)
    evaluate_baseline("4. HOG + Local Contrast + PCA + RBF SVM", svm_hog, ZF_tr, y_tr, ZF_va, y_va)
    print("-" * 120)


if __name__ == "__main__":
    main()
