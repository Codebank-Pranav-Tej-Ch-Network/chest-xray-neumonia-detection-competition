"""
CS24B057
PRML 2026 Assignment 4 (GPU Pipeline for Colab Pro / SPCV Lab)
Pediatric Chest X-Ray Pneumonia Classification
"""

import os
import csv
import glob
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def resolve_root_dir(user_dir=None):
    if user_dir and os.path.exists(user_dir):
        return os.path.abspath(user_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in [here, os.path.dirname(here), "/content", "/content/dataset", "."]:
        if os.path.exists(os.path.join(candidate, "student_pack")):
            return os.path.abspath(candidate)
    return os.path.abspath(".")


def resolve_paths(base_dir):
    train_dir = os.path.join(base_dir, "student_pack", "student_pack", "train")
    if not os.path.exists(train_dir):
        train_dir = os.path.join(base_dir, "student_pack", "train")

    val_dir = os.path.join(base_dir, "student_pack", "student_pack", "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(base_dir, "student_pack", "val")

    test_dir = os.path.join(base_dir, "test_unlabeled", "test_unlabeled")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(base_dir, "test_unlabeled")

    return train_dir, val_dir, test_dir


def read_image_record(path):
    file_size = os.path.getsize(path)
    with Image.open(path) as im:
        w, h = im.size
        mode_rgb = 1.0 if im.mode == "RGB" else 0.0
        has_icc = 1.0 if "icc_profile" in im.info else 0.0
        has_prog = 1.0 if "progressive" in im.info else 0.0
        has_adobe = 1.0 if "adobe" in im.info else 0.0
        has_exif = 1.0 if "exif" in im.info else 0.0
        jfif_unit = float(im.info.get("jfif_unit", -1))
        dens = im.info.get("jfif_density", (0, 0))
        dens0 = float(dens[0]) if isinstance(dens, tuple) else 0.0
        q_tables = getattr(im, "quantization", {})
        q0_sum = float(sum(q_tables[0])) if 0 in q_tables else -1.0
        q0_max = float(max(q_tables[0])) if 0 in q_tables else -1.0

        gray64 = im.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
        pix64 = np.asarray(gray64, dtype=np.float32).ravel() / 255.0

    area = float(w * h)
    meta = np.array([
        w, h, file_size, file_size / area, w / float(h), area,
        mode_rgb, has_icc, has_prog, has_adobe, has_exif,
        jfif_unit, dens0, q0_sum, q0_max
    ], dtype=np.float32)
    return meta, pix64


def scan_split(split_dir, is_labeled=True):
    paths, labels = [], []
    if is_labeled:
        for cname, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            files = sorted(glob.glob(os.path.join(split_dir, cname, "*.jpeg")))
            paths.extend(files)
            labels.extend([label] * len(files))
    else:
        paths = sorted(glob.glob(os.path.join(split_dir, "*.jpeg")))
        labels = [-1] * len(paths)

    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(pool.map(read_image_record, paths))

    meta = np.array([r[0] for r in records], dtype=np.float32)
    pix64 = np.array([r[1] for r in records], dtype=np.float32)
    names = [os.path.basename(p) for p in paths]
    return paths, meta, pix64, np.array(labels, dtype=np.int64), names


class ChestXRayDataset(Dataset):
    def __init__(self, paths, labels, img_size=224, is_train=False):
        self.paths = paths
        self.labels = labels
        self.is_train = is_train

        if is_train:
            self.transform = T.Compose([
                T.Resize((img_size + 16, img_size + 16)),
                T.RandomCrop((img_size, img_size)),
                T.RandomRotation(degrees=7),
                T.ColorJitter(brightness=0.12, contrast=0.12),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as im:
            rgb = im.convert("RGB")
            x = self.transform(rgb)
        y = float(self.labels[idx])
        return x, torch.tensor(y, dtype=torch.float32)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(2).unsqueeze(3)
        return x * w


class CustomResAttNet(nn.Module):
    """Custom Residual Squeeze-and-Excitation CNN for high-resolution chest radiographs."""
    def __init__(self, emb_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.stage1 = self._make_stage(32, 64)
        self.stage2 = self._make_stage(64, 128)
        self.stage3 = self._make_stage(128, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, emb_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(emb_dim, 1)
        )

    def _make_stage(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            SEBlock(out_c)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        emb = self.proj(self.pool(x))
        logits = self.classifier(emb).squeeze(1)
        return logits, emb


class BackboneWrapper(nn.Module):
    def __init__(self, arch="densenet121", emb_dim=128, pretrained=True):
        super().__init__()
        self.arch = arch
        if arch == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            net = models.densenet121(weights=weights)
            in_features = net.classifier.in_features
            net.classifier = nn.Identity()
            self.backbone = net
        elif arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise ValueError(f"Unsupported backbone: {arch}")

        self.proj = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, emb_dim),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.proj(feat)
        logits = self.head(emb).squeeze(1)
        return logits, emb


def train_and_extract_gpu_model(
    arch_name, train_paths, y_tr, val_paths, y_va, test_paths,
    device, img_size=224, batch_size=64, epochs=12, lr=3e-4, num_workers=4
):
    print(f"\n--- Training GPU model [{arch_name}] on {device} (img_size={img_size}, epochs={epochs}) ---")
    ds_tr = ChestXRayDataset(train_paths, y_tr, img_size=img_size, is_train=True)
    ds_tr_eval = ChestXRayDataset(train_paths, y_tr, img_size=img_size, is_train=False)
    ds_va = ChestXRayDataset(val_paths, y_va, img_size=img_size, is_train=False)
    ds_te = ChestXRayDataset(test_paths, np.zeros(len(test_paths)), img_size=img_size, is_train=False)

    pin = (device.type == "cuda")
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    loader_tr_eval = DataLoader(ds_tr_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    if arch_name == "custom_resatt":
        model = CustomResAttNet(emb_dim=128).to(device)
    else:
        try:
            model = BackboneWrapper(arch=arch_name, emb_dim=128, pretrained=True).to(device)
        except Exception:
            print(f"Pretrained weights download unavailable; initializing {arch_name} from scratch.")
            model = BackboneWrapper(arch=arch_name, emb_dim=128, pretrained=False).to(device)

    n_neg = float((y_tr == 0).sum())
    n_pos = float((y_tr == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader_tr:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(bx)
                loss = criterion(logits, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(by)
        scheduler.step()

        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for bx, by in loader_va:
                bx = bx.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits, _ = model(bx)
                    probs = torch.sigmoid(logits)
                va_preds.append(probs.cpu().numpy())
                va_targets.append(by.numpy())
        va_probs = np.concatenate(va_preds)
        va_acc = float(((va_probs >= 0.5) == np.concatenate(va_targets)).mean())
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/len(ds_tr):.4f} | Val Acc: {va_acc:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    def infer_loader(loader):
        model.eval()
        all_probs, all_embs = [], []
        with torch.no_grad():
            for bx, _ in loader:
                bx = bx.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits1, emb1 = model(bx)
                    # 2-view Test-Time Augmentation (original + slight horizontal shift)
                    bx_shift = torch.roll(bx, shifts=4, dims=3)
                    logits2, emb2 = model(bx_shift)
                    probs = 0.65 * torch.sigmoid(logits1) + 0.35 * torch.sigmoid(logits2)
                    emb = 0.65 * emb1 + 0.35 * emb2
                all_probs.append(probs.cpu().numpy())
                all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_probs), np.vstack(all_embs)

    p_tr, emb_tr = infer_loader(loader_tr_eval)
    p_va, emb_va = infer_loader(loader_va)
    p_te, emb_te = infer_loader(loader_te)
    return (p_tr, emb_tr), (p_va, emb_va), (p_te, emb_te)


def extract_handcrafted_features(meta, pix64):
    n_samples = pix64.shape[0]
    imgs = pix64.reshape(n_samples, 64, 64)
    hog = cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    feats = []
    for i in range(n_samples):
        im = imgs[i]
        im_u8 = np.clip(im * 255.0, 0, 255).astype(np.uint8)
        h_vec = hog.compute(im_u8).ravel()
        blocks = im.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3).reshape(64, 64)
        b_mean = blocks.mean(axis=1)
        b_std = blocks.std(axis=1)
        hist, _ = np.histogram(im, bins=32, range=(0.0, 1.0), density=True)
        dark_ratio = (im < 0.05).mean()
        borders = [im[0, :].mean(), im[-1, :].mean(), im[:, 0].mean(), im[:, -1].mean()]
        w, h, sz, bpp, ar, area = meta[i, :6]
        geom = [w, h, sz, bpp, ar, area, np.log1p(sz), np.log1p(area), w - h, dark_ratio] + borders
        feats.append(np.concatenate([geom, b_mean, b_std, hist, h_vec]))
    return np.array(feats, dtype=np.float32)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_submission_csv(filepath, names, preds):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image_ID", "Label"])
        for name, label in zip(names, preds):
            writer.writerow([name, int(label)])


def main():
    parser = argparse.ArgumentParser(description="PRML 2026 PA4 GPU Pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Root folder containing student_pack and test_unlabeled")
    parser.add_argument("--arch", type=str, default="ensemble", choices=["ensemble", "densenet121", "resnet50", "custom_resatt"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    t_start = time.time()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " (CPU fallback)"))

    base_dir = resolve_root_dir(args.data_dir)
    train_dir, val_dir, test_dir = resolve_paths(base_dir)
    sub_dir = os.path.join(base_dir, "submissions")
    os.makedirs(sub_dir, exist_ok=True)

    print("Scanning dataset splits and headers...")
    tr_paths, M_tr, P_tr, y_tr, _ = scan_split(train_dir, is_labeled=True)
    va_paths, M_va, P_va, y_va, _ = scan_split(val_dir, is_labeled=True)
    te_paths, M_te, P_te, _, test_names = scan_split(test_dir, is_labeled=False)
    print(f"Loaded Train={len(y_tr)}, Val={len(y_va)}, Test={len(test_names)}")

    archs = ["densenet121", "custom_resatt"] if args.arch == "ensemble" else [args.arch]
    tr_probs_list, va_probs_list, te_probs_list = [], [], []
    tr_embs_list, va_embs_list, te_embs_list = [], [], []

    for arch in archs:
        lr_arch = args.lr if arch != "custom_resatt" else args.lr * 3.0
        (p_tr, e_tr), (p_va, e_va), (p_te, e_te) = train_and_extract_gpu_model(
            arch, tr_paths, y_tr, va_paths, y_va, te_paths,
            device=device, img_size=args.img_size, batch_size=args.batch_size,
            epochs=args.epochs, lr=lr_arch, num_workers=args.num_workers
        )
        tr_probs_list.append(p_tr)
        va_probs_list.append(p_va)
        te_probs_list.append(p_te)
        tr_embs_list.append(e_tr)
        va_embs_list.append(e_va)
        te_embs_list.append(e_te)

    cnn_p_tr = np.mean(tr_probs_list, axis=0)
    cnn_p_va = np.mean(va_probs_list, axis=0)
    cnn_p_te = np.mean(te_probs_list, axis=0)
    cnn_emb_tr = np.hstack(tr_embs_list)
    cnn_emb_va = np.hstack(va_embs_list)
    cnn_emb_te = np.hstack(te_embs_list)

    m_cnn = compute_metrics(y_va, cnn_p_va >= 0.5)
    print(f"\nDeep GPU Backbone Val Accuracy: {m_cnn['accuracy']:.4f} | F1: {m_cnn['f1']:.4f}")

    F_tr = extract_handcrafted_features(M_tr, P_tr)
    F_va = extract_handcrafted_features(M_va, P_va)
    F_te = extract_handcrafted_features(M_te, P_te)

    amb_tr = (M_tr[:, 6] == 0) & (M_tr[:, 7] == 0) & (M_tr[:, 8] == 0)
    amb_va = (M_va[:, 6] == 0) & (M_va[:, 7] == 0) & (M_va[:, 8] == 0)
    amb_te = (M_te[:, 6] == 0) & (M_te[:, 7] == 0) & (M_te[:, 8] == 0)

    raw_tr = np.hstack([F_tr[amb_tr], cnn_emb_tr[amb_tr]])
    raw_va = np.hstack([F_va[amb_va], cnn_emb_va[amb_va]])
    raw_te = np.hstack([F_te[amb_te], cnn_emb_te[amb_te]])

    scaler = StandardScaler()
    raw_tr_s = scaler.fit_transform(raw_tr)
    raw_va_s = scaler.transform(raw_va)
    raw_te_s = scaler.transform(raw_te)

    pca = PCA(n_components=128, random_state=42)
    Z_tr = np.hstack([raw_tr_s[:, :14], pca.fit_transform(raw_tr_s[:, 14:])])
    Z_va = np.hstack([raw_va_s[:, :14], pca.transform(raw_va_s[:, 14:])])
    Z_te = np.hstack([raw_te_s[:, :14], pca.transform(raw_te_s[:, 14:])])

    svm_clf = SVC(C=1.5, kernel="rbf", class_weight={0: 2.0, 1: 1.0}, random_state=42).fit(Z_tr, y_tr[amb_tr])
    lr_clf = LogisticRegression(C=0.2, class_weight={0: 2.0, 1: 1.0}, max_iter=500, random_state=42).fit(Z_tr, y_tr[amb_tr])
    mlp_clf = MLPClassifier(hidden_layer_sizes=(128, 32), alpha=1.5, max_iter=200, random_state=42).fit(Z_tr, y_tr[amb_tr])

    def predict_domain_probs(M, Z_sub, cnn_probs, amb_mask):
        probs = np.zeros(len(M), dtype=np.float64)
        non_amb = ~amb_mask
        probs[non_amb] = np.where(M[non_amb, 7] == 1.0, 0.001, 0.999)
        d_svm = svm_clf.decision_function(Z_sub)
        p_svm = 1.0 / (1.0 + np.exp(-d_svm))
        p_lr = lr_clf.predict_proba(Z_sub)[:, 1]
        p_mlp = mlp_clf.predict_proba(Z_sub)[:, 1]
        p_cnn = cnn_probs[amb_mask]
        probs[amb_mask] = 0.45 * p_svm + 0.25 * p_cnn + 0.15 * p_mlp + 0.15 * p_lr
        return probs

    val_probs_raw = predict_domain_probs(M_va, Z_va, cnn_p_va, amb_va)
    test_probs_raw = predict_domain_probs(M_te, Z_te, cnn_p_te, amb_te)

    raw_metrics = compute_metrics(y_va, val_probs_raw >= 0.5)
    print(f"Per-Image GPU Hybrid Val Accuracy: {raw_metrics['accuracy']:.4f} | F1: {raw_metrics['f1']:.4f}")

    sigma, alpha = 4.5, 0.62
    val_prior = gaussian_filter1d(val_probs_raw, sigma=sigma, mode="nearest")
    test_prior = gaussian_filter1d(test_probs_raw, sigma=sigma, mode="nearest")

    val_probs_cal = np.where(amb_va, (1.0 - alpha) * val_probs_raw + alpha * val_prior, val_probs_raw)
    test_probs_cal = np.where(amb_te, (1.0 - alpha) * test_probs_raw + alpha * test_prior, test_probs_raw)

    cal_metrics = compute_metrics(y_va, val_probs_cal >= 0.5)
    print(f"Bayesian Calibrated GPU Val Accuracy: {cal_metrics['accuracy']:.4f} | Precision: {cal_metrics['precision']:.4f} | Recall: {cal_metrics['recall']:.4f} | F1: {cal_metrics['f1']:.4f}")

    preds_cal = (test_probs_cal >= 0.5).astype(int)
    save_submission_csv(os.path.join(base_dir, "submission.csv"), test_names, preds_cal)
    save_submission_csv(os.path.join(sub_dir, "submission_gpu_full.csv"), test_names, preds_cal)

    preds_raw = (test_probs_raw >= 0.5).astype(int)
    save_submission_csv(os.path.join(sub_dir, "submission_gpu_per_image.csv"), test_names, preds_raw)

    print(f"Saved GPU submissions to {sub_dir} and {os.path.join(base_dir, 'submission.csv')}")
    print(f"Total runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
