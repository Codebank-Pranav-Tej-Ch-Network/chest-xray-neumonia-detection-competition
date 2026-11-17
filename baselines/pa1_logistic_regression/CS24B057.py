"""
CS24B057
EE506L Programming Assignment 1

Vectorized Logistic Regression from Scratch -- Pediatric Chest X-Ray
Pneumonia Detection

Objective: The objective of this programming assignment is to provide
hands-on practice in formulating, implementing, and evaluating a
foundational vectorized Logistic Regression model optimized via vanilla
Gradient Descent using pure NumPy.

Run:
    python CS24B057.py

Expects the Kaggle competition dataset to be extracted under
./dataset/train and ./dataset/val, each containing NORMAL/ and
PNEUMONIA/ subfolders of images (same layout as the provided skeleton).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_and_flatten(image_dir, target_size=(64, 64)):
    """
    Loads images from 'Normal' and 'Pneumonia' subdirectories, resizes them,
    normalizes pixel values, flattens them, and compiles the data matrix.

    Args:
        image_dir (str): Path to the dataset split directory (train or val).
        target_size (tuple): Destination resolution (height, width).
    Returns:
        X (np.ndarray): Data matrix of shape (N, D) where D = target_size[0] * target_size[1].
        y (np.ndarray): Binary labels vector of shape (N,) where 0=Normal, 1=Pneumonia.
    """
    X_list = []
    y_list = []

    # Define class mappings
    class_mapping = {'NORMAL': 0, 'PNEUMONIA': 1}

    for class_name, label in class_mapping.items():
        dir_path = os.path.join(image_dir, class_name)
        if not os.path.exists(dir_path):
            continue

        for img_name in sorted(os.listdir(dir_path)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_path, img_name)

                # ------------------------------------------------------------------
                # TASK 1
                # 1. Load image in grayscale using Pillow (PIL)
                # 2. Resize image to target_size
                # 3. Convert to numpy array and normalize pixel intensities to [0.0, 1.0]
                # 4. Flatten the 2D image matrix into a 1D vector
                # ------------------------------------------------------------------
                img = Image.open(img_path).convert('L')
                img = img.resize(target_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                flat = arr.flatten()

                X_list.append(flat)
                y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y


def plot_sample_images(image_dir, target_size=(64, 64), save_path="sample_images.png"):
    """
    EDA helper (Task 1): saves a grid of a few normalized sample images from
    each class (Normal, Pneumonia) so they can be visually inspected.
    """
    class_mapping = {'NORMAL': 0, 'PNEUMONIA': 1}
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))

    for row, class_name in enumerate(class_mapping.keys()):
        dir_path = os.path.join(image_dir, class_name)
        if not os.path.exists(dir_path):
            continue
        files = [f for f in sorted(os.listdir(dir_path))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3]
        for col, fname in enumerate(files):
            img = Image.open(os.path.join(dir_path, fname)).convert('L').resize(target_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            axes[row, col].imshow(arr, cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(class_name)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved sample image grid to {save_path}")


def sigmoid(z):
    """
    Computes the element-wise logistic sigmoid function.
    """
    # ----------------------------------------------------------------------
    # TASK 2a
    # Clip z before exponentiating to guard against numerical overflow.
    # ----------------------------------------------------------------------
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(X, y, w, b):
    """
    Computes the average Binary Cross-Entropy (Negative Log-Likelihood) loss.
    """
    N = X.shape[0]
    # ----------------------------------------------------------------------
    # TASK 2c
    # E(w, b) = -(1/N) * sum_i [ y_i*log(y_hat_i) + (1-y_i)*log(1-y_hat_i) ]
    # eps guards against log(0).
    # ----------------------------------------------------------------------
    eps = 1e-15
    z = X @ w + b
    y_hat = sigmoid(z)
    y_hat = np.clip(y_hat, eps, 1 - eps)
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss


def compute_gradients(X, y, w, b):
    """
    Computes analytical gradients of the loss function with respect to w and b.
    """
    N = X.shape[0]
    # ----------------------------------------------------------------------
    # TASK 2d
    # dE/dw = (1/N) * X^T (y_hat - y)      shape (D,)
    # dE/db = (1/N) * sum_i (y_hat_i - y_i) scalar
    # ----------------------------------------------------------------------
    z = X @ w + b
    y_hat = sigmoid(z)
    error = y_hat - y
    dw = (X.T @ error) / N
    db = np.sum(error) / N
    return dw, db


def compute_evaluation_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Calculates binary classification metrics from scratch without sklearn.
    """
    # ----------------------------------------------------------------------
    # TASK 4
    # 1. Threshold probabilities to get binary 0 or 1 predictions.
    # 2. Calculate True Positives, False Positives, True Negatives, False Negatives.
    # 3. Compute accuracy, precision, recall, and F1-score.
    # ----------------------------------------------------------------------
    y_pred = (y_pred_probs >= threshold).astype(int)
    y_true = y_true.astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    return metrics


def train_model(X_train, y_train, X_val, y_val, epochs=1000, learning_rate=0.01):
    """
    Executes the full Gradient Descent optimization loop and logs performance.
    """
    N, D = X_train.shape

    # ----------------------------------------------------------------------
    # FIX HYPERPARAMETERS & INITIALIZATION
    # Initialize weights to zeros and bias to zero.
    # ----------------------------------------------------------------------
    w = np.zeros(D)
    b = 0.0

    train_losses = []
    val_losses = []

    print("Starting optimization loop...")
    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------
        # TASK 3
        # 1. Compute gradients using compute_gradients()
        # 2. Update w and b using vanilla Gradient Descent rule
        # 3. Compute and record training and validation losses
        # ------------------------------------------------------------------
        dw, db = compute_gradients(X_train, y_train, w, b)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        train_loss = compute_loss(X_train, y_train, w, b)
        val_loss = compute_loss(X_val, y_val, w, b)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Logging progress every 100 epochs
        if epoch % 100 == 0 or epoch == 1:
            train_loss = compute_loss(X_train, y_train, w, b)
            val_loss = compute_loss(X_val, y_val, w, b)
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return w, b, train_losses, val_losses


if __name__ == "__main__":
    # Define relative paths to your local dataset folders downloaded from Kaggle
    TRAIN_DIR = "./dataset/train"
    VAL_DIR = "./dataset/val"

    print("Loading and preprocessing data...")
    X_train, y_train = preprocess_and_flatten(TRAIN_DIR)
    X_val, y_val = preprocess_and_flatten(VAL_DIR)

    print(f"Training Matrix Shape: {X_train.shape} | Labels: {y_train.shape}")
    print(f"Validation Matrix Shape: {X_val.shape} | Labels: {y_val.shape}")

    print(f"Train - Normal: {int(np.sum(y_train==0))}, Pneumonia: {int(np.sum(y_train==1))}")
    print(f"Val   - Normal: {int(np.sum(y_val==0))}, Pneumonia: {int(np.sum(y_val==1))}")
    # ------------------------------------------------------------------
    # Task 1: EDA sample visualizations (both classes), before training
    # ------------------------------------------------------------------
    plot_sample_images(TRAIN_DIR)

    # Train the model with frozen assignment hyperparameters
    EPOCHS = 1000
    LR = 0.01
    w, b, train_losses, val_losses = train_model(X_train, y_train, X_val, y_val, epochs=EPOCHS, learning_rate=LR)

    # Generate Plots and Metrics evaluation below...

    # ------------------------------------------------------------------
    # Task 3(c): convergence plot -- training vs validation loss
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=150)
    plt.close()
    print("Saved convergence plot to convergence_plot.png")

    # ------------------------------------------------------------------
    # Task 4: quantitative performance metrics (train + validation)
    # ------------------------------------------------------------------
    train_probs = sigmoid(X_train @ w + b)
    val_probs = sigmoid(X_val @ w + b)
    train_metrics = compute_evaluation_metrics(y_train, train_probs)
    val_metrics = compute_evaluation_metrics(y_val, val_probs)

    print("\n{:<12} {:<10} {:<10}".format("Metric", "Train", "Validation"))
    print("-" * 34)
    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
        print("{:<12} {:<10.4f} {:<10.4f}".format(key, train_metrics[key], val_metrics[key]))

    # ------------------------------------------------------------------
    # Task 5: weight heatmap (model interpretability)
    # ------------------------------------------------------------------
    w_grid = w.reshape(64, 64)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(w_grid, cmap='coolwarm')
    plt.colorbar(im, label="Weight magnitude")
    plt.title("Learned Pixel Weights (Model Interpretability)")
    plt.tight_layout()
    plt.savefig("weight_heatmap.png", dpi=150)
    plt.close()
    print("Saved weight heatmap to weight_heatmap.png")
