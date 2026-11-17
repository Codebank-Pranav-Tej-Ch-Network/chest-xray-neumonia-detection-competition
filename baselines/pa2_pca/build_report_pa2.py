# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "TitleCustom", parent=styles["Title"], fontSize=16, spaceAfter=4
)
subtitle_style = ParagraphStyle(
    "SubtitleCustom", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER,
    textColor=colors.HexColor("#333333"), spaceAfter=4
)
author_style = ParagraphStyle(
    "AuthorCustom", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), spaceAfter=14
)
heading_style = ParagraphStyle(
    "HeadingCustom", parent=styles["Heading2"], fontSize=13, spaceBefore=16, spaceAfter=8,
    textColor=colors.HexColor("#1a1a1a")
)
body_style = ParagraphStyle(
    "BodyCustom", parent=styles["Normal"], fontSize=10.3, leading=15, spaceAfter=10,
    alignment=4  # justify
)
caption_style = ParagraphStyle(
    "CaptionCustom", parent=styles["Normal"], fontSize=8.5, alignment=TA_CENTER,
    textColor=colors.HexColor("#555555"), spaceAfter=10, fontName="Helvetica-Oblique"
)

story = []

# ---------------- Title block ----------------
story.append(Paragraph("EE506L Programming Assignment 2", title_style))
story.append(Paragraph(
    "PCA from Scratch and Evaluation for Pediatric Chest X-Ray Pneumonia Detection",
    subtitle_style
))
story.append(Paragraph("Ch Venkata Sai Pranav Tej&nbsp;&nbsp;|&nbsp;&nbsp;Roll No: CS24B057&nbsp;&nbsp;|&nbsp;&nbsp;BTech CSE, 3rd Year, IIT Tirupati", author_style))
story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#999999"), spaceAfter=10))

story.append(Paragraph(
    "My report documents a from-scratch Principal Component Analysis (PCA) pipeline built in pure NumPy, "
    "applied to the same preprocessed chest X-ray dataset used in Programming Assignment 1 (64\u00d764 grayscale "
    "images, flattened to 4096-dimensional vectors). The mean, covariance, eigendecomposition and projection are "
    "all implemented by hand; scikit-learn is used only for the downstream Logistic Regression benchmark that "
    "compares training time and validation performance on raw pixels versus PCA-reduced features, as asked "
    "in the assignment.", body_style
))

# ---------------- Section 1: PCA ----------------
story.append(Paragraph("1. Principal Component Analysis (PCA) from Scratch", heading_style))
story.append(Paragraph(
    "The training mean &mu; &isin; R<super>4096</super> is computed over all 4684 training images, and both the "
    "training and validation matrices are centered using this same training mean (the validation split is never "
    "used to recompute &mu;, as specified). Let X<sub>c,train</sub> = X<sub>train</sub> &minus; &mu; denote the "
    "centered training data. The unbiased 4096\u00d74096 covariance matrix is formed directly as "
    "&Sigma; = X<sub>c,train</sub><super>T</super>X<sub>c,train</sub> / (N&minus;1), and its eigenvalues/"
    "eigenvectors are extracted with <font name='Courier'>np.linalg.eigh</font>, which exploits the symmetry of "
    "the covariance matrix. Since eigh returns eigenvalues in ascending order, they are reversed along with their "
    "corresponding eigenvectors to obtain a descending spectrum, from which the top-k projection matrix "
    "W &isin; R<super>4096\u00d7k</super> is sliced. Both splits are then projected as Z = X<sub>c</sub>W, using the "
    "same training-derived W and &mu; in both cases.", body_style
))

# ---------------- Section 2: k selection ----------------
story.append(Paragraph("2. Tracking the Variance Spectrum and Choosing k", heading_style))
story.append(Paragraph(
    "Before computing the cumulative variance ratio, any small negative eigenvalues produced by "
    "floating-point round-off (expected here since N = 4684 &lt; D = 4096) are clipped to zero, so they "
    "contribute nothing to the variance total. Walking the cumulative ratio "
    "&sum;<sub>i=1</sub><super>k</super>&lambda;<sub>i</sub> / &sum;<sub>j=1</sub><super>D</super>&lambda;<sub>j</sub> "
    "up from the largest eigenvalue, the minimum no. of components needed to reach or exceed 95% of the total "
    "training variance is:", body_style
))

k_table_data = [
    ["Quantity", "Value"],
    ["Minimum components (k) for \u226595% variance", "237"],
    ["Explained variance retained at k = 237", "95.015883%"],
    ["Dimensionality reduction", "4096 \u2192 237 (\u224894.2% fewer features)"],
]
k_table = Table(k_table_data, colWidths=[3.6*inch, 2.6*inch])
k_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(k_table)
story.append(Spacer(1, 10))
story.append(Paragraph(
    "A model built on just 237 principal components (under 6% of the original 4096 raw pixel features) "
    "therefore already captures over 95% of the variance in the training images, which quite a lot of"
    "pixel-to-pixel redundancy in the raw 64\u00d764 representation; unsurprisingly, since neighboring pixels "
    "in an X-ray are highly correlated with each other.", body_style
))

# ---------------- Section 3: Robust metrics ----------------
story.append(Paragraph("3. Robust Evaluation Metrics", heading_style))
story.append(Paragraph(
    "Precision, Recall and F1-Score are reused from Programming Assignment 1, with Pneumonia kept as the "
    "positive class, and extended to handle every zero-denominator edge case explicitly: "
    "TP + FP = 0 (no positive predictions at all), TP + FN = 0 (no true positives present in the ground truth), "
    "and Precision + Recall = 0 (both metrics are zero). In every such case the function returns 0.0 for that "
    "metric instead of raising a division error, so the evaluation pipeline cannot crash regardless of how "
    "degenerate a classifier's predictions are.", body_style
))

# ---------------- Section 4: Benchmarking ----------------
story.append(Paragraph("4. Experimental Benchmarking and Empirical Reflection", heading_style))
story.append(Paragraph(
    "Running the local validation block trains two scikit-learn "
    "<font name='Courier'>LogisticRegression</font> models on the training split; one on the full "
    "4096-dimensional raw pixel features, and one on the 237-dimensional PCA-projected features. I"
    "evaluate both on the (unseen) validation split. Training wall time measures the classifier fit only; "
    "the PCA computation itself is excluded, as asked.", body_style
))

scorecard_data = [
    ["Metric / Configuration", "RAW (4096 Dimensions)", "PCA (k = 237 Dimensions)"],
    ["Training Wall Time", "8.6268 s", "1.3641 s"],
    ["Validation Precision", "0.9469", "0.9532"],
    ["Validation Recall", "0.9602", "0.9532"],
    ["Validation F1-Score", "0.9535", "0.9532"],
]
scorecard = Table(scorecard_data, colWidths=[2.3*inch, 2.0*inch, 2.0*inch])
scorecard.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(scorecard)
story.append(Paragraph("Table 1: Raw-feature vs. PCA-reduced Logistic Regression, validation split.", caption_style))

story.append(Paragraph("<b>Analysis</b>", body_style))
story.append(Paragraph(
    "The measured training speedup factor is <b>6.3242\u00d7</b> (8.6268s on raw 4096-dim features versus "
    "1.3641s on 237-dim PCA features); reducing the feature dimensionality by roughly 94% translated "
    "almost directly into a proportionally large cut in training wall time, since scikit-learn's solver cost "
    "scales with the number of input features.", body_style
))
story.append(Paragraph(
    "On validation performance, the two configurations are effectively tied: F1-score drops by only 0.0003 "
    "(0.9535 \u2192 0.9532) when moving from raw pixels to the 237-dimensional PCA representation. Precision "
    "actually improves slightly (0.9469 \u2192 0.9532) while recall drops slightly (0.9602 \u2192 0.9532), so the "
    "small shift is a minor rebalancing between the two rather than a genuine loss in distinguishing ability. "
    "Discarding the lowest-variance 5% of the pixel space therefore neither meaningfully helps nor hurts "
    "classification quality here; it is best read as making no practical difference to performance, while "
    "delivering a large, unambiguous speed benefit.", body_style
))
story.append(Paragraph(
    "This result is worth pausing on precisely because PCA is unsupervised: the top-237 directions were chosen "
    "purely to maximize retained pixel-intensity variance, with no access to the Normal/Pneumonia labels during "
    "fitting. The fact that validation performance barely moves implies that the directions of highest variance "
    "in this dataset happen to align well with the directions that are actually distinguishable for pneumonia "
    "classification; i.e., the lowest-variance 5% that was discarded was largely redundant or near-constant "
    "pixel information (e.g. uniform background and low-variation border regions) rather than information the "
    "classifier depended on. This may not hold in general; had the class-distinguishing signal lived "
    "predominantly in a low-variance direction, PCA could have discarded exactly the information the classifier "
    "needed and performance would have dropped. Here, that risk did not materialize, and PCA offers "
    "a clearly favorable tradeoff: a 6.3\u00d7 faster classifier at statistically negligible cost to validation F1.",
    body_style
))

# ---------------- Conclusion ----------------
story.append(Paragraph("Conclusion", heading_style))
story.append(Paragraph(
    "A from-scratch PCA implementation reduced the chest X-ray feature space from 4096 raw pixel dimensions to "
    "237 principal components while retaining just over 95% of the training variance. Feeding this compressed "
    "representation into an out-of-the-box Logistic Regression classifier trained 6.32\u00d7 faster than the "
    "raw-pixel baseline, with validation F1-score essentially unchanged (0.9532 vs. 0.9535). The robust Precision/"
    "Recall/F1 functions handle every zero-denominator edge case without crashing, and the overall experiment "
    "demonstrates that, for this dataset, PCA is an effective and nearly cost-free dimensionality reduction step "
    "ahead of a linear classifier.", body_style
))

story.append(Spacer(1, 14))
story.append(Paragraph("Appendix", heading_style))
story.append(Paragraph(
    "Requirements to run: install numpy, pillow and scikit-learn via pip in a Python virtual environment "
    "(same environment as PA1, with scikit-learn added). Directory structure expected by the provided skeleton: "
    "the script sits inside a working folder with <font name='Courier'>dataset/train</font> and "
    "<font name='Courier'>dataset/val</font> one level up (<font name='Courier'>../dataset/train</font>, "
    "<font name='Courier'>../dataset/val</font>), each containing class subdirectories as in PA1. "
    "Function names and signatures (<font name='Courier'>fit_pca</font>, <font name='Courier'>transform_pca</font>, "
    "<font name='Courier'>find_k_for_variance</font>, <font name='Courier'>compute_metrics</font>) are unchanged "
    "from the provided skeleton, per the assignment's constraints.", body_style
))

doc = SimpleDocTemplate(
    "CS24B057.pdf",
    pagesize=letter,
    topMargin=0.85*inch, bottomMargin=0.85*inch,
    leftMargin=0.9*inch, rightMargin=0.9*inch,
    title="EE506L PA2 Report - CS24B057",
    author="Ch Venkata Sai Pranav Tej"
)
doc.build(story)
print("done")
