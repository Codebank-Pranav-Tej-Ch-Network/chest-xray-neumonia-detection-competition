import os
import time
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

# DATA PROCESSING (I used the same as PA-1)

def preprocess_and_flatten(dir_path, image_size=(64, 64)):
    """
    Reads grayscale X-ray images from class subdirectories (e.g., normal/sick),
    resizes them, flattens them into 1D vectors, and scales pixels to [0, 1].
    """
    X_list = []
    y_list = []
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
        
    subdirs = sorted([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
    
    for idx, subdir in enumerate(subdirs):
        subdir_path = os.path.join(dir_path, subdir)
        
        for img_name in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img_gray = img.convert('L')
                    img_resized = img_gray.resize(image_size)
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    
                    X_list.append(img_array.ravel())
                    y_list.append(idx)
            except Exception:
                continue
                
    return np.array(X_list), np.array(y_list)


# Q1: PRINCIPAL COMPONENT ANALYSIS (PCA) FROM SCRATCH

def fit_pca(X_train, k):
    """
    Computes the empirical training mean, covariance matrix, and extracts
    the top k principal components based on sorted eigenvalues.

    Returns:
        sorted_eigenvalues (np.ndarray): Sorted eigenvalues of shape (k,)
        W (np.ndarray): Top k orthonormal eigenvectors of shape (D, k)
        mu (np.ndarray): Empirical training mean vector of shape (D,)
    """
    # Step 1. I'm computing training mean vector (mu) of shape (D,)
    mu = np.mean(X_train, axis=0)

    # Step 2. Then, center the training data about mu
    X_centered = X_train - mu

    # Step 3. Compute the sample covariance matrix of shape (D, D) using (N - 1)
    N = X_train.shape[0]
    cov = (X_centered.T @ X_centered) / (N - 1)

    # Step 4. Perform Eigendecomposition (Referred to Appendix in notes)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 5. Sort eigenvalues and their corresponding eigenvectors in descending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Step 6. Extract the top k sorted eigenvalues and eigenvectors (W)
    sorted_eigenvalues = eigenvalues[:k]
    W = eigenvectors[:, :k]

    return sorted_eigenvalues, W, mu



def transform_pca(X, W, mu):
    """
    Centers an arbitrary data matrix using the precomputed training mean
    and projects it onto the lower-dimensional subspace.
    """
    # Center the input matrix X using the training mean mu and project it using W.
    X_centered = X - mu
    return X_centered @ W

# Q2: TRACKING THE VARIANCE TO CHOOSE k

def find_k_for_variance(eigenvalues, threshold=0.95):
    """
    Calculates the minimum number of principal components needed to retain
    a target percentage of total variance.
    """
    # Set small -ve eigenvalues (round-off) to zero.
    eigenvalues = np.where(eigenvalues < 0, 0, eigenvalues)

    # Compute the cumulative explained variance ratios.
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance

    # Find the first index where the target threshold is met or exceeded, and return k.
    k = int(np.argmax(cumulative_variance_ratio >= threshold) + 1)
    return k


# Q3: ROBUST EVALUATION METRICS (extension of PA-1 function)

def compute_metrics(y_true, y_pred):
    """
    Computes Precision, Recall, and F1-Score from scratch without external libraries.
    Gracefully handles zero-denominator edge cases by returning 0.0 for that metric.
    """
    # 1. Flatten arrays to 1D
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # 2. Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # 3. Compute Precision, Recall, and F1-Score.
    # 4. Handle zero denominator cases (TP+FP, TP+FN, Precision+Recall) by returning 0.0.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}


# Q4: LOCAL VALIDATION & EXPERIMENTAL BENCHMARK

if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING VALIDATION & EXPERIMENTAL BENCHMARK")
    print("=" * 75)
    
    TRAIN_DIR = os.path.join("dataset", "train")
    VAL_DIR = os.path.join("dataset", "val")
    
    try:
        print("Processing and flattening clinical image sets...")
        X_train, y_train = preprocess_and_flatten(TRAIN_DIR)
        X_val, y_val = preprocess_and_flatten(VAL_DIR)
        
        print(f"Data Loaded Successfully:")
        print(f"   Shape of X_train: {X_train.shape} (Raw Dimension: {X_train.shape[1]})")
        print(f"   Shape of X_val:   {X_val.shape}")
        
        # 1. PCA Subspace Construction
        print("\nExtracting PCA Variance Spectrum via Student Functions...")
        total_features = X_train.shape[1]
        all_evs, W_all, mu_all = fit_pca(X_train, k=total_features)
        
        if all_evs is None:
            print("t_pca function is currently empty (returning None).")
            print("Complete Q1-Q3 to unlock the experimental speed/accuracy comparison table!")
        else:
            k_95 = find_k_for_variance(all_evs, threshold=0.95)
            print(f"Target k components computed to preserve 95% variance: {k_95}")
            
            # Negative eigenvalues caused by numerical round-off contribute no variance.
            nonnegative_evs = np.maximum(all_evs, 0.0)
            total_variance = np.sum(nonnegative_evs)

            if total_variance > 0:
                retained_variance = (
                    np.sum(nonnegative_evs[:k_95]) / total_variance
                )
                print(f"Selected components: {k_95}")
                print(f"Explained variance retained: {retained_variance:.6%}")

            # Slice the top k_95 eigenvectors (avoids a second eigendecomposition)
            W, mu = W_all[:, :k_95], mu_all
            Z_train = transform_pca(X_train, W, mu)
            Z_val = transform_pca(X_val, W, mu)
            
            print(f"Raw training features: {X_train.shape}")
            print(f"PCA training features: {Z_train.shape}")
            print(f"Raw validation features: {X_val.shape}")
            print(f"PCA validation features: {Z_val.shape}")

            # COMPONENT 1: RAW CLASSIFICATION MODEL RUN
            print(f"\nTraining Built-in Logistic Regression on RAW Data ({total_features} dims)...")
            clf_raw = LogisticRegression(max_iter=1000, random_state=42)
            
            t0 = time.time()
            clf_raw.fit(X_train, y_train)
            raw_time = time.time() - t0
            
            raw_preds = clf_raw.predict(X_val)
            metrics_raw = compute_metrics(y_val, raw_preds)
            
            # COMPONENT 2: PCA REDUCED CLASSIFICATION MODEL RUN
            print(f"Training Logistic Regression on PCA Data ({k_95} dims)...")
            clf_pca = LogisticRegression(max_iter=1000, random_state=42)
            
            t1 = time.time()
            clf_pca.fit(Z_train, y_train)
            pca_time = time.time() - t1
            
            # Calculate how many times faster PCA-based model training was.
            if pca_time > 0:
                speedup = raw_time / pca_time
                print(f"\nTraining Speedup Factor: {speedup:.4f}x")
            else:
                print("\nTraining Speedup Factor: Undefined (PCA training time is zero)")

            pca_preds = clf_pca.predict(Z_val)
            metrics_pca = compute_metrics(y_val, pca_preds)
            
            # COMPARATIVE TABULAR EXPORT DISPLAY
            print("\n" + "=" * 75)
            print(f"{'METRIC / CONFIGURATION':<25} | {'RAW (High-Dim)':<22} | {f'PCA (k={k_95} Reduced)':<22}")
            print("=" * 75)
            print(f"{'Training Wall Time':<25} | {raw_time:<19.4f}s | {pca_time:<19.4f}s")
            print("-" * 75)
            print(f"{'Validation PRECISION':<25} | {metrics_raw.get('precision', 0.0):<21.4f}  | {metrics_pca.get('precision', 0.0):<21.4f}")
            print(f"{'Validation RECALL':<25} | {metrics_raw.get('recall', 0.0):<21.4f}  | {metrics_pca.get('recall', 0.0):<21.4f}")
            print(f"{'Validation F1-SCORE':<25} | {metrics_raw.get('f1_score', 0.0):<21.4f}  | {metrics_pca.get('f1_score', 0.0):<21.4f}")
            print("=" * 75)
            
    except Exception as e:
        print(f"\nPipeline crashed with structural error: {e}")
        print("Fix code syntax errors and implement the requested assignment functions completely.")
    print("=" * 75)
