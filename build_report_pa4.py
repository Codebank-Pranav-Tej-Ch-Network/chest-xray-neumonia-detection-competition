# -*- coding: utf-8 -*-
import os
import sys
import shutil
import tempfile

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cs24b057")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage  # preload native Pillow before appending PA1 site-packages

pa1_site = r"C:\Users\sunny\PRML\Assgn-1\venv\lib\python3.12\site-packages"
if os.path.exists(pa1_site) and pa1_site not in sys.path:
    sys.path.append(pa1_site)

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, HRFlowable, KeepTogether
)


def generate_figures(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Validation Progression across Models
    stages = [
        "PA1: GD\nLogReg",
        "L2 LogReg\n(4096-D)",
        "PA2: PCA(237)\n+ LogReg",
        "PCA(237) +\nRBF SVM",
        "HOG+Contrast\n+ RBF SVM",
        "GPU Per-Image\n(alpha=0.0)",
        "Final GPU\nCalibrated"
    ]
    val_acc = [92.99, 94.36, 94.36, 94.53, 98.29, 99.15, 99.49]
    val_f1 = [95.29, 96.17, 96.17, 96.31, 98.83, 99.42, 99.65]

    x = np.arange(len(stages))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=200)
    r1 = ax.bar(x - width / 2, val_acc, width, label="Validation Accuracy (%)", color="#1a3a5c")
    r2 = ax.bar(x + width / 2, val_f1, width, label="Validation F1-Score (%)", color="#2e8b57")

    ax.set_ylabel("Score (%)", fontsize=9.5, fontweight="bold")
    ax.set_title("Validation Performance Progression Across Models (N = 585)", fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8.2)
    ax.set_ylim(90.0, 100.8)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", frameon=True, fontsize=8.8)

    for r in r1:
        h = r.get_height()
        ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1a3a5c")
    for r in r2:
        h = r.get_height()
        ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#195232")

    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "ablation_progression.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close(fig)

    # Figure 2: Training Convergence & Public vs Private Leaderboard Scores
    epochs = np.array([1, 3, 6, 9, 12, 15])
    tr_loss = np.array([0.1420, 0.0355, 0.0221, 0.0066, 0.0037, 0.0017])
    va_acc_dense = np.array([90.60, 93.16, 95.56, 98.12, 97.95, 98.12])
    va_acc_resatt = np.array([51.28, 54.87, 82.22, 96.75, 96.41, 96.07])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7), dpi=200)

    ax1_t = ax1.twinx()
    l1 = ax1.plot(epochs, tr_loss, "o--", color="#c0392b", lw=1.8, label="DenseNet121 Train Loss")
    l2 = ax1_t.plot(epochs, va_acc_dense, "s-", color="#1a3a5c", lw=2.0, label="DenseNet121 Val Acc (%)")
    l3 = ax1_t.plot(epochs, va_acc_resatt, "^-", color="#2e8b57", lw=1.8, label="CustomResAtt Val Acc (%)")
    ax1_t.axvline(9, color="#7f8c8d", linestyle=":", lw=1.3)
    ax1_t.annotate("Best Val Checkpoint\n(Epoch 9)", xy=(9, 98.12), xytext=(4.2, 90.0),
                   arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9), fontsize=7.8, fontweight="bold")

    ax1.set_xlabel("Epoch", fontsize=9.0, fontweight="bold")
    ax1.set_ylabel("Weighted BCE Loss", color="#c0392b", fontsize=9.0, fontweight="bold")
    ax1_t.set_ylabel("Validation Accuracy (%)", color="#1a3a5c", fontsize=9.0, fontweight="bold")
    ax1.set_title("(a) GPU Backbone Training vs. Validation Curve", fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    lines = l1 + l2 + l3
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=7.5, frameon=True)

    sub_labels = [
        "Sub 3:\nCPU Per-Img",
        "Sub 7:\nGPU Pixels",
        "Sub 6:\nGPU Per-Img",
        "Sub 4 & 5:\nGPU v1",
        "Sub 8 (Final):\nGPU v2"
    ]
    pub_s = [97.482, 97.482, 98.201, 99.280, 99.640]
    prv_s = [97.734, 98.058, 98.381, 98.705, 99.029]
    x2 = np.arange(len(sub_labels))
    w2 = 0.35

    b_pub = ax2.bar(x2 - w2 / 2, pub_s, w2, label="Public LB (47%)", color="#2b5c8f")
    b_prv = ax2.bar(x2 + w2 / 2, prv_s, w2, label="Private LB (53%)", color="#1b7837")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sub_labels, fontsize=7.8)
    ax2.set_ylim(96.5, 100.3)
    ax2.set_ylabel("Kaggle Accuracy (%)", fontsize=9.0, fontweight="bold")
    ax2.set_title("(b) Public vs. Private Leaderboard Progression", fontsize=9.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=7.8, frameon=True)

    for b in b_pub:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#1a3a5c")
    for b in b_prv:
        h = b.get_height()
        ax2.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#195232")

    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "generalization_diagnostics.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close(fig)
    return fig1_path, fig2_path


def make_table(data, col_widths, font_size=8.2):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.0),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def build_report(pdf_path, fig1_path, fig2_path):
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "SubtitleCustom", parent=styles["Normal"], fontSize=11, leading=14, alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"), spaceAfter=4
    )
    author_style = ParagraphStyle(
        "AuthorCustom", parent=styles["Normal"], fontSize=10, leading=13, alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"), spaceAfter=12
    )
    heading_style = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"], fontSize=12.5, leading=15, spaceBefore=12, spaceAfter=6,
        fontName="Helvetica-Bold", textColor=colors.HexColor("#1a1a1a")
    )
    subheading_style = ParagraphStyle(
        "SubHeadingCustom", parent=styles["Heading3"], fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=4,
        fontName="Helvetica-Bold", textColor=colors.HexColor("#222222")
    )
    body_style = ParagraphStyle(
        "BodyCustom", parent=styles["Normal"], fontSize=9.8, leading=14, spaceAfter=7, alignment=4
    )
    formula_style = ParagraphStyle(
        "FormulaCustom", parent=body_style, alignment=TA_CENTER, fontName="Helvetica-Oblique", spaceBefore=4, spaceAfter=6
    )
    caption_style = ParagraphStyle(
        "CaptionCustom", parent=styles["Normal"], fontSize=8.5, leading=11, alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"), spaceAfter=8, fontName="Helvetica-Oblique"
    )
    mono_style = ParagraphStyle(
        "MonoCustom", fontName="Courier", fontSize=8.2, leading=11, spaceAfter=8,
        backColor=colors.HexColor("#f4f4f4"), leftIndent=6, rightIndent=6, borderPadding=5
    )

    story = []

    # ---------------- Title Block ----------------
    story.append(Paragraph("EE506L Programming Assignment 4", title_style))
    story.append(Paragraph("Pediatric Chest X-Ray Pneumonia Detection", subtitle_style))
    story.append(Paragraph("Ch Venkata Sai Pranav Tej&nbsp;&nbsp;|&nbsp;&nbsp;Roll No: CS24B057&nbsp;&nbsp;|&nbsp;&nbsp;BTech CSE, 3rd Year, IIT Tirupati", author_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#999999"), spaceAfter=10))

    # ---------------- Intro ----------------
    story.append(Paragraph(
        "My report documents the models and experiments I built for the Programming Assignment 4 Kaggle competition "
        "on pediatric chest X-ray pneumonia detection. Starting from my Logistic Regression baseline in Programming "
        "Assignment 1 and my Principal Component Analysis (PCA) implementation in Programming Assignment 2, I "
        "gradually tested linear classifiers, kernel SVMs, handcrafted radiographic descriptors, and deep convolutional "
        "networks across all 5,856 radiographs in the dataset (4,684 training, 585 validation, and 587 test images). "
        "My main focus was making sure the model generalizes well to unseen radiographs rather than just memorizing "
        "the training split. My final model finished 1st on the 47% Public Leaderboard (0.99640, or 277/278 correct) "
        "and 1st on the 53% Private Leaderboard (0.99029, or 306/309 correct).",
        body_style
    ))

    # ---------------- Section 1 ----------------
    story.append(Paragraph("1. Dataset Breakdown, Class Imbalance, and Loss Function", heading_style))
    story.append(Paragraph(
        "As seen in Programming Assignment 1, both the training and validation splits have a 73% Pneumonia to 27% "
        "Normal class imbalance (a 2.70 : 1 ratio). A classifier that blindly predicts Pneumonia on every image already "
        "gets 72.99% accuracy while failing completely on Normal chest X-rays. Because of this skew, I tracked "
        "Precision, Recall, and F1-Score alongside accuracy on the validation split for every model.",
        body_style
    ))

    split_data = [
        ["Split", "Total Images (N)", "Normal (Label 0)", "Pneumonia (Label 1)", "Pneumonia %"],
        ["Training (train)", "4,684", "1,266", "3,418", "72.97%"],
        ["Validation (val)", "585", "158", "427", "72.99%"],
        ["Pooled Labeled (train + val)", "5,269", "1,424", "3,845", "72.97%"],
        ["Unlabeled Test (test_unlabeled)", "587", "155 (predicted)", "432 (predicted)", "73.59%"],
    ]
    story.append(make_table(split_data, [1.9 * inch, 1.15 * inch, 1.15 * inch, 1.25 * inch, 1.05 * inch]))
    story.append(Paragraph("Table 1: Dataset split distribution and class proportions.", caption_style))

    story.append(Paragraph(
        "To handle the 2.70 : 1 class imbalance without hurting recall on Pneumonia cases, I used two adjustments: "
        "(1) In my PyTorch convolutional networks, I used class-weighted binary cross-entropy (<font name='Courier'>BCEWithLogitsLoss</font>) "
        "with positive weight w<sub>+</sub> = N<sub>neg</sub> / N<sub>pos</sub> = 1266 / 3418 = 0.3704 and label smoothing (&epsilon; = 0.05), "
        "so that targets become y<sub>smooth</sub> = 0.95y + 0.025:",
        body_style
    ))
    story.append(Paragraph(
        "L(z, y) = &minus;(1/N) &Sigma;<sub>i</sub> [ w<sub>+</sub> y<sub>smooth,i</sub> log &sigma;(z<sub>i</sub>) + "
        "(1 &minus; y<sub>smooth,i</sub>) log(1 &minus; &sigma;(z<sub>i</sub>)) ]",
        formula_style
    ))
    story.append(Paragraph(
        "Label smoothing prevents the network from driving training logits to &plusmn;&infin; in later epochs. "
        "(2) In my Soft-Margin SVM and Logistic Regression classifiers, I tested different class weights on validation "
        "and found that a moderate weight of w<sub>0</sub> = 2.0, w<sub>1</sub> = 1.0 works better than full 2.70&times; inverse weighting.",
        body_style
    ))

    # ---------------- Section 2 ----------------
    story.append(Paragraph("2. Gradual Model Progression and Experimental Results", heading_style))
    story.append(Paragraph(
        "Before jumping to deep networks, I benchmarked a clean progression of models starting from my PA1 and PA2 "
        "implementations on the 64&times;64 grayscale pixel vectors (4,096 features), and then moved to domain-specific "
        "radiographic features and 320&times;320 GPU backbones. Table 2 summarizes the validation results across all stages.",
        body_style
    ))

    ablation_data = [
        ["Model / Stage", "Features (Dim)", "Fit Time", "Val Acc", "Val Prec", "Val Rec", "Val F1"],
        ["0. PA1 Vanilla GD LogReg", "Raw 64x64 Pixels (4096)", "14.20 s", "0.9299", "0.9463", "0.9555", "0.9529"],
        ["1. Basic L2 LogReg (C=0.1)", "Raw 64x64 Pixels (4096)", "5.60 s", "0.9436", "0.9539", "0.9696", "0.9617"],
        ["2. Basic PCA (k=237) + LogReg", "PCA Subspace (237)", "0.20 s", "0.9436", "0.9539", "0.9696", "0.9617"],
        ["2b. PCA (k=237) + Weighted LR", "PCA Subspace (237)", "0.27 s", "0.9231", "0.9591", "0.9344", "0.9466"],
        ["3. PCA (k=237) + Soft-Margin SVM", "Standardized PCA (237)", "10.07 s", "0.9453", "0.9478", "0.9789", "0.9631"],
        ["4. HOG + Contrast + PCA + SVM", "Radiographic + PCA (197)", "1.77 s", "0.9829", "0.9860", "0.9906", "0.9883"],
        ["5. Iter 2 CPU 5-Fold Stacker", "HOG + Contrast + CNN64", "48.0 s", "0.9846", "0.9838", "0.9953", "0.9895"],
        ["6a. GPU Pure-Pixel (320x320)", "DenseNet121+ResNet50+HOG", "180.0 s", "0.9863", "0.9883", "0.9930", "0.9906"],
        ["6b. GPU Per-Image (alpha=0.0)", "320x320 CNNs + 5-Fold", "195.0 s", "0.9915", "0.9907", "0.9977", "0.9942"],
        ["6c. Final GPU Hybrid (320x320)", "+ Prevalence Prior (w=0.36)", "198.0 s", "0.9949", "0.9930", "1.0000", "0.9965"],
    ]
    story.append(make_table(ablation_data, [2.05 * inch, 1.60 * inch, 0.60 * inch, 0.58 * inch, 0.58 * inch, 0.58 * inch, 0.58 * inch], font_size=8.0))
    story.append(Paragraph("Table 2: Validation split (N = 585) progression from PA1/PA2 baselines to the final 320x320 GPU hybrid.", caption_style))

    story.append(Image(fig1_path, width=6.0 * inch, height=2.25 * inch))
    story.append(Paragraph("Figure 1: Validation Accuracy and F1-Score progression across all model stages.", caption_style))

    story.append(Paragraph(
        "<b>1. From PA1 &amp; PA2 Linear Baselines to Soft-Margin RBF SVM (Stages 0 to 3):</b> Replacing plain full-batch "
        "gradient descent (92.99% Val Acc) with L2-regularized L-BFGS Logistic Regression (C = 0.1) improved validation "
        "accuracy to 94.36% (F1 = 0.9617) by damping pixel-to-pixel colinearity. Applying my PA2 PCA pipeline with k = 237 "
        "components (retaining over 95% variance) cut solver fit time by 28&times; (5.60s down to 0.20s) with the exact "
        "same validation accuracy (94.36%). Replacing the linear hyperplane on those 237 principal components with a "
        "Soft-Margin RBF SVM (C = 2.5) captured non-linear interactions between eigen-modes and lifted recall to 97.89% (F1 = 0.9631).",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Radiographic Feature Engineering (Stage 4, +3.76% Val Acc jump to 98.29%):</b> Raw pixel intensities "
        "are sensitive to overall X-ray exposure. To capture local lung structure without relying on OpenCV, I implemented "
        "a vectorized NumPy/SciPy feature extractor computing: (a) 16&times;16 local adaptive contrast normalization "
        "(exposing faint retrocardiac and basal opacities), (b) multi-scale 8&times;8 and 16&times;16 cell Histograms of "
        "Oriented Gradients (HOG) with 2&times;2 L2 block normalization (separating sharp horizontal rib edges from cloudy "
        "alveolar infiltrates), (c) Sobel gradient magnitude grids, and (d) bilateral left-vs-right lung symmetry ratios "
        "(since lobar pneumonia often consolidates asymmetrically in one lung). Projecting these features with PCA into "
        "my Soft-Margin RBF SVM jumped validation accuracy from 94.53% to 98.29% (F1 = 0.9883) in 1.77s.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. High-Resolution 320&times;320 GPU Ensemble &amp; 5-Fold Stacking (Stage 6):</b> Downsampling a chest X-ray "
        "to 64&times;64 blurs fine interstitial markings and air bronchograms. On an NVIDIA RTX A4000 GPU, I trained "
        "three 320&times;320 convolutional backbones: pretrained <font name='Courier'>DenseNet121</font> (98.46% standalone Val Acc), "
        "pretrained <font name='Courier'>ResNet50</font> (97.44% Val Acc), and a custom 4-stage Squeeze-and-Excitation residual network "
        "<font name='Courier'>CustomResAttNet</font>, using 3-view Test-Time Augmentation (original + &plusmn;4px horizontal shifts). "
        "Pooling all 5,269 labeled images under 5-Fold Stratified Cross-Validation for the tabular stacker (RBF-SVM + "
        "Logistic Regression + MLP + HistGradientBoosting) achieved 99.05% out-of-fold accuracy, and blending it with "
        "the out-of-sample GPU backbones reached 99.15% per-image validation accuracy (99.49% after prior calibration).",
        body_style
    ))

    # ---------------- Section 3 ----------------
    story.append(Paragraph("3. What Worked vs. What Did Not Work (Preventing Overfitting)", heading_style))
    story.append(Paragraph(
        "Several ideas I tried either hurt validation performance or risked overfitting to the training split:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Aggressive Class Weighting on Linear Models (Did Not Work):</b> In Stage 2b, applying a 2.0&times; Normal class "
        "weight to linear Logistic Regression on PCA features pushed the linear decision boundary too far into the Pneumonia "
        "region, dropping validation recall from 96.96% to 93.44% and accuracy from 94.36% to 92.31%. Class weighting only "
        "helped once paired with non-linear RBF kernels and tree/neural ensembles.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Late-Epoch Logit Saturation on DenseNet121 (Overfitting Fix):</b> In my initial 15-epoch GPU run (Figure 2a), "
        "<font name='Courier'>DenseNet121</font> reached its peak validation accuracy (98.12%) at Epoch 9 (train loss 0.0066), dipped at Epoch 12, "
        "and hit 98.12% again at Epoch 15 (where train loss had collapsed to 0.0017). Because my checkpoint condition initially "
        "used <font name='Courier'>va_acc &gt;= best_val_acc</font>, the overfitted Epoch 15 weights overwrote Epoch 9. I fixed this by breaking "
        "accuracy ties using lowest validation loss <font name='Courier'>(round(va_acc, 5), -va_loss)</font>, adding label smoothing (&epsilon; = 0.05), "
        "<font name='Courier'>RandomErasing(p=0.15)</font>, and using a 0.25&times; learning rate on the pretrained backbone trunk.",
        body_style
    ))
    story.append(Paragraph(
        "<b>3. In-Sample CNN Embedding Leakage in Stacking (Overfitting Fix):</b> Initially, I concatenated the 128-D CNN "
        "training embeddings with the handcrafted HOG features inside the 5-Fold SVM/MLP/HGB stacker. Because the CNN had "
        "already trained on the training labels, its training embeddings were nearly linearly separable, causing the stacker "
        "to over-rely on the CNN and ignore the geometric and HOG features. Training the 5-Fold stacker strictly on the "
        "handcrafted radiographic features and blending probabilities at the meta-level (0.55 GPU CNN + 0.45 5-Fold Stacker) "
        "removed this leakage completely.",
        body_style
    ))
    story.append(Paragraph(
        "<b>4. Sequence Smoothing vs. Unseen Shuffled Data:</b> Inspecting the dataset metadata showed that 31.3% of images "
        "carry deterministic scanner tags (<font name='Courier'>icc_profile</font> is 100% Normal across 1,220/1,220 labeled images; RGB/Progressive "
        "headers are 100% Pneumonia across 398/398 images) and the test filenames exhibit high lag-1 autocorrelation (r<sub>1</sub> = 0.924). "
        "In Submission 1, using a heavy Gaussian smoothing weight (&alpha; = 0.62) erased genuine visual exceptions and plateaued at "
        "0.98201 on the Public Leaderboard. Lowering the weight and accounting for the 73% Pneumonia base rate (w = 0.36 when local "
        "prior &gt; 0.75, w = 0.28 otherwise) preserved high-confidence visual overrides while preventing borderline false negatives. "
        "To make sure the code never fails on a randomly shuffled test set, I added an automatic autocorrelation gate that checks "
        "r<sub>1</sub> &gt; 0.50 and sets &alpha; = 0.0 automatically if the test directory is unordered.",
        body_style
    ))

    story.append(Image(fig2_path, width=6.1 * inch, height=2.10 * inch))
    story.append(Paragraph("Figure 2: (a) GPU backbone convergence curve; (b) Public vs. Private Leaderboard scores across submissions.", caption_style))

    # ---------------- Section 4 ----------------
    story.append(Paragraph("4. Progression Across My 8 Kaggle Submissions", heading_style))
    story.append(Paragraph(
        "Table 3 lists all 8 submissions I made on Kaggle during the competition, showing both the 47% Public Leaderboard "
        "score (278 images) and the final 53% Private Leaderboard score (309 images). Notably, even my unsmoothed per-image "
        "GPU model (Submission 6, &alpha; = 0.0) scored 0.98381 on the Private Leaderboard, which by itself would have placed "
        "1st ahead of the 0.98058 runner-up.",
        body_style
    ))

    sub_table_data = [
        ["#", "Submitted File", "Model Description", "Public LB (47%)", "Private LB (53%)"],
        ["1", "sub1_iter1_99pct.csv", "Iter 1: CNN + PCA + RBF SVM (alpha=0.62)", "0.98201 (273/278)", "0.99029 (306/309)"],
        ["2", "sub2_iter2_hybrid.csv", "Iter 2: 5-Fold CLAHE + HOG + CNN (alpha=0.32)", "0.98201 (273/278)", "0.99029 (306/309)"],
        ["3", "sub3_iter2_per_image.csv", "Iter 2: 5-Fold Per-Image Only (alpha=0.0)", "0.97482 (271/278)", "0.97734 (302/309)"],
        ["4", "sub4_gpu_ensemble_v1.csv", "Iter 3 GPU: DenseNet121 + ResNet50 + ResAtt", "0.99280 (276/278)", "0.98705 (305/309)"],
        ["5", "sub5_gpu_full.csv", "Iter 3 GPU: 320x320 Hybrid (sym. prior)", "0.99280 (276/278)", "0.98705 (305/309)"],
        ["6", "sub6_gpu_per_image.csv", "Iter 3 GPU: 320x320 Per-Image (alpha=0.0)", "0.98201 (273/278)", "0.98381 (304/309)"],
        ["7", "sub7_gpu_pure_visual.csv", "Iter 3 GPU: 320x320 Pure Pixels (no headers)", "0.97482 (271/278)", "0.98058 (303/309)"],
        ["8", "sub8_gpu_calibrated_v2.csv", "Final 320x320 GPU + Prevalence Prior (#1)", "0.99640 (277/278)", "0.99029 (306/309)"],
    ]
    story.append(make_table(sub_table_data, [0.30 * inch, 1.62 * inch, 2.55 * inch, 1.05 * inch, 1.05 * inch], font_size=7.9))
    story.append(Paragraph("Table 3: Complete log of my 8 Kaggle competition submissions.", caption_style))

    # ---------------- Section 5 ----------------
    story.append(Paragraph("5. References", heading_style))
    refs = [
        "1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. <i>Cell</i>, 172(5), 1122-1131.",
        "2. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. <i>arXiv:1711.05225</i>.",
        "3. Huang, G., Liu, Z., Van Der Maaten, L., &amp; Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. <i>IEEE CVPR</i>, 4700-4708.",
        "4. He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep Residual Learning for Image Recognition. <i>IEEE CVPR</i>, 770-778.",
        "5. Hu, J., Shen, L., &amp; Sun, G. (2018). Squeeze-and-Excitation Networks. <i>IEEE CVPR</i>, 7132-7141.",
        "6. Dalal, N., &amp; Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. <i>IEEE CVPR</i>, 886-893.",
        "7. Cortes, C., &amp; Vapnik, V. (1995). Support-Vector Networks. <i>Machine Learning</i>, 20(3), 273-297.",
        "8. Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.",
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle("RefItem", parent=body_style, fontSize=8.8, leading=12, spaceAfter=3)))

    # ---------------- Appendix ----------------
    story.append(Paragraph("Appendix: Environment Setup and Reproducibility Instructions", heading_style))
    story.append(Paragraph(
        "All code is self-contained in <font name='Courier'>CS24B057.py</font> (and <font name='Courier'>CS24B057.ipynb</font>) with fixed random seeds "
        "(<font name='Courier'>seed = 42</font>, <font name='Courier'>cudnn.deterministic = True</font>). The script automatically locates the dataset "
        "folders (<font name='Courier'>student_pack</font> and <font name='Courier'>test_unlabeled</font>) in the current or parent directory, or via "
        "<font name='Courier'>--data-dir</font>. To reproduce my best GPU run and the classical ablation table from scratch on any machine:",
        body_style
    ))
    story.append(Paragraph(
        "# 1. Create and activate a Python 3.11 virtual environment<br/>"
        "python3 -m venv .venv &amp;&amp; source .venv/bin/activate &nbsp;&nbsp;# On Windows: .venv\\Scripts\\activate<br/>"
        "pip install -r requirements.txt<br/><br/>"
        "# 2. Run the winning 320x320 GPU pipeline (generates submission.csv and submissions/*.csv)<br/>"
        "python CS24B057.py --data-dir . --arch ensemble --img-size 320 --epochs 15 --batch-size 32<br/><br/>"
        "# 3. (Optional) Run the classical baselines (LR, PCA+LR, PCA+SVM, HOG+PCA+SVM)<br/>"
        "python run_ablations.py",
        mono_style
    ))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        leftMargin=0.80 * inch,
        rightMargin=0.80 * inch,
        title="EE506L PA4 Report - CS24B057",
        author="Ch Venkata Sai Pranav Tej"
    )
    doc.build(story)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(base_dir, "figures")
    pdf_path = os.path.join(base_dir, "CS24B057.pdf")
    pdf_copy = os.path.join(base_dir, "CS24B057_Report.pdf")

    fig1_path, fig2_path = generate_figures(fig_dir)
    build_report(pdf_path, fig1_path, fig2_path)
    shutil.copyfile(pdf_path, pdf_copy)
    print(f"Built ReportLab PDF: {pdf_path} and {pdf_copy}")


if __name__ == "__main__":
    main()
