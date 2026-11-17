from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib import colors

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom', fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=4, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='SubTitle', fontSize=11, leading=14, alignment=TA_CENTER, textColor=colors.HexColor('#444444'), spaceAfter=14))
styles.add(ParagraphStyle(name='SectionHeading', fontSize=13, leading=16, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold', textColor=colors.HexColor('#1a1a1a')))
styles.add(ParagraphStyle(name='SubHeading', fontSize=11, leading=14, spaceBefore=8, spaceAfter=4, fontName='Helvetica-Bold'))
styles.add(ParagraphStyle(name='BodyCustom', fontSize=10.3, leading=14.5, spaceAfter=8, alignment=4))  # justified
styles.add(ParagraphStyle(name='Caption', fontSize=9, leading=11, alignment=TA_CENTER, textColor=colors.HexColor('#555555'), spaceAfter=10, fontName='Helvetica-Oblique'))
styles.add(ParagraphStyle(name='Mono', fontName='Courier', fontSize=8.5, leading=11, spaceAfter=8, backColor=colors.HexColor('#f2f2f2')))

doc = SimpleDocTemplate(
    "CS24B057.pdf", pagesize=letter,
    topMargin=0.75*inch, bottomMargin=0.75*inch,
    leftMargin=0.85*inch, rightMargin=0.85*inch
)

story = []

# ---------------- Title ----------------
story.append(Paragraph("EE506L Programming Assignment 1", styles['TitleCustom']))
story.append(Paragraph("Vectorized Logistic Regression from Scratch for Pediatric Chest X-Ray Pneumonia Detection", styles['SubTitle']))
story.append(Paragraph("Ch Venkata Sai Pranav Tej &nbsp;|&nbsp; Roll No: CS24B057 &nbsp;|&nbsp; BTech CSE, 3rd Year, IIT Tirupati", styles['SubTitle']))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#999999'), spaceAfter=10))

# ---------------- Intro ----------------
story.append(Paragraph(
    "My report documents a logistic regression classifier built entirely from first principles using NumPy, "
    "trained to distinguish pneumonia-positive chest X-rays from normal ones, per the Kaggle competition dataset "
    "provided for this assignment. Every component including the sigmoid prediction, the binary cross-entropy loss, "
    "its gradient, and the gradient-descent update rule; is implemented by hand rather than pulled from a "
    "library, per the assignment's requirement to use pure NumPy only.",
    styles['BodyCustom']))

# ================= Part 1 =================
story.append(Paragraph("1. Exploratory Data Analysis (EDA) and Image Preprocessing", styles['SectionHeading']))
story.append(Paragraph(
    "Each raw chest X-ray is converted to grayscale, resized to 64&times;64 pixels, and its intensities are "
    "rescaled from the original [0, 255] range down to [0.0, 1.0]. The resized image is then flattened into a "
    "single 4096-dimensional vector, so that every pixel becomes one input feature to the logistic regression "
    "model. Running this pipeline over the provided train/validation split produced the following data matrices:",
    styles['BodyCustom']))

data1 = [
        ["Split", "Samples (N)", "Features (D)", "Normal", "Pneumonia"],
    ["Train", "4684", "4096", "1266", "3418"],
    ["Validation", "585", "4096", "158", "427"],
]
t1 = Table(data1, colWidths=[1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
t1.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 9.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t1)
story.append(Spacer(1, 10))

story.append(Image("sample_images.png", width=5.4*inch, height=3.6*inch))
story.append(Paragraph("Figure 1: Sample preprocessed 64&times;64 grayscale images from each class (top row: Normal, bottom row: Pneumonia).", styles['Caption']))

story.append(Paragraph(
    "Both splits are dominated by the pneumonia class in almost exactly the same proportion; about 73% "
    "pneumonia in training and about 73% in validation; so the two splits are internally consistent with "
    "each other, but the class distribution itself is imbalanced. This matters for how the results should be read: "
    "a model that blindly predicted &ldquo;Pneumonia&rdquo; for every single image would already score close to "
    "73% accuracy without having learned anything. That single fact is the reason accuracy alone cannot be trusted "
    "as the main performance measure here, and why precision, recall, and F1-score (computed in Part 4) matter "
    "more for judging whether the model has actually learned something useful. ",
    styles['BodyCustom']))

# ================= Part 2 =================
story.append(Paragraph("2. Vectorized Logistic Regression from Scratch", styles['SectionHeading']))
story.append(Paragraph(
    "The model predicts the probability of pneumonia as p = &sigma;(Xw + b), where &sigma;(z) = 1 / (1 + "
    "e<super>&minus;z</super>) is the logistic sigmoid, applied element-wise, and p plays the role of "
    "&ldquo;y_hat&rdquo; in the code. Training minimizes the average "
    "binary cross-entropy loss over all N training examples:",
    styles['BodyCustom']))
story.append(Paragraph(
    "E(w, b) = &minus;(1/N) &Sigma;<sub>i</sub> [ y<sub>i</sub> log(p<sub>i</sub>) + (1&minus;y<sub>i</sub>) "
    "log(1&minus;p<sub>i</sub>) ]",
    ParagraphStyle(name='Formula', parent=styles['BodyCustom'], alignment=TA_CENTER, fontName='Helvetica-Oblique')))
story.append(Paragraph(
    "Differentiating this loss with respect to the parameters gives the two gradients used in every update step "
    "of gradient descent:",
    styles['BodyCustom']))
story.append(Paragraph(
    "&part;E/&part;w = (1/N) X<super>T</super>(p &minus; y), &nbsp;&nbsp;&nbsp; "
    "&part;E/&part;b = (1/N) &Sigma;<sub>i</sub> (p<sub>i</sub> &minus; y<sub>i</sub>)",
    ParagraphStyle(name='Formula2', parent=styles['BodyCustom'], alignment=TA_CENTER, fontName='Helvetica-Oblique')))
story.append(Paragraph(
    "All of this is implemented directly as matrix operations on the full 4684&times;4096 training matrix at once "
    "(vectorized, i.e. no explicit per-sample Python loop over training examples), which is what keeps each epoch "
    "computationally cheap despite the 4096-dimensional input. A small numerical clip is applied inside both the "
    "sigmoid (bounding z to &plusmn;500) and the loss (bounding p away from exactly 0 or 1) purely as a guard "
    "against floating-point overflow/underflow in log(&middot;) and exp(&middot;). This does not change the "
    "math used, only increases its numerical stability. ",
    styles['BodyCustom']))

# ================= Part 3 =================
story.append(Paragraph("3. Optimization via Gradient Descent and Tracking", styles['SectionHeading']))
story.append(Paragraph(
    "Weights are initialized to an all-zero vector and the bias to zero. With the frozen hyperparameters specified "
    "for the assignment (learning rate = 0.01, 1000 epochs), plain (vanilla, full-batch) gradient descent update "
    "w &larr; w &minus; &eta;&nabla;<sub>w</sub>E and b &larr; b &minus; &eta;&nabla;<sub>b</sub>E was run for the "
    "full 1000 epochs, recomputing both the training and validation loss after every single update so that the "
    "convergence curve below is not just a sparse snapshot every 100 epochs, but the true per-epoch trajectory. Also, "
    "I had tried putting 10000 epochs as an experiment. Both losses decreased slightly but the heatmap seemed incorrect "
    " to my naked eye, and hence, as the assignment asks, I am submitting documentation related to 1000 epochs only ",
    styles['BodyCustom']))

loss_table_data = [
    ["Epoch", "Train Loss", "Val Loss"],
    ["1", "0.7533", "0.7508"],
    ["100", "0.3430", "0.3430"],
    ["200", "0.2825", "0.2831"],
    ["300", "0.2530", "0.2548"],
    ["400", "0.2346", "0.2378"],
    ["500", "0.2219", "0.2264"],
    ["600", "0.2124", "0.2181"],
    ["700", "0.2050", "0.2118"],
    ["800", "0.1990", "0.2068"],
    ["900", "0.1941", "0.2028"],
    ["1000", "0.1900", "0.1994"],
]
t2 = Table(loss_table_data, colWidths=[1.4*inch, 1.6*inch, 1.6*inch])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 9.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t2)
story.append(Spacer(1, 8))

story.append(Image("convergence_plot.png", width=4.6*inch, height=3.3*inch))
story.append(Paragraph("Figure 2: Training and validation binary cross-entropy loss vs. epoch.", styles['Caption']))

story.append(Paragraph(
    "Both curves fall smoothly and almost on top of each other for the entire run, dropping from about 0.75 at "
    "epoch 1 to about 0.19&ndash;0.20 by epoch 1000, with no sign of the validation curve turning upward while the "
    "training curve keeps falling; which is the usual visual signature of overfitting, and it is not "
    "present here. Since both losses are still decreasing at epoch 1000 rather than having flattened out, the "
    "model has not fully converged yet within the 1000 epochs specified by the assignment; training for more "
    "epochs at this same learning rate would likely push both losses down further before diminishing returns set "
    "in. This is what I had tried out, and the model seems closer to convergence after 10000 epochs, i.e., the "
    "change observed between any two losses in both training and validation is small after every 100 epochs too. ",
    styles['BodyCustom']))

# ================= Part 4 =================
story.append(Paragraph("4. Quantitative Performance Metrics", styles['SectionHeading']))
story.append(Paragraph(
    "Because of the roughly 73%/27% class imbalance noted in Part 1, accuracy alone was treated as an insufficient "
    "measure, so precision, recall, and F1-score were also computed directly from the confusion-matrix counts "
    "(true/false positives and negatives), without using any library function: ",
    styles['BodyCustom']))

metrics_data = [
    ["Metric", "Train", "Validation"],
    ["Accuracy", "0.9317", "0.9299"],
    ["Precision", "0.9406", "0.9347"],
    ["Recall", "0.9675", "0.9719"],
    ["F1-score", "0.9539", "0.9529"],
]
t3 = Table(metrics_data, colWidths=[1.6*inch, 1.6*inch, 1.6*inch])
t3.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,-1), 9.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bbbbbb')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
    ('TOPPADDING', (0,0), (-1,-1), 5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
]))
story.append(t3)
story.append(Spacer(1, 8))

story.append(Paragraph(
    "The validation recall of 0.9719 means the model catches roughly 97% of true pneumonia cases; with 427 "
    "pneumonia images in the validation set, that corresponds to only about 12 missed cases (false negatives). On "
    "the training split, recall is very slightly lower at 0.9675, i.e. roughly 111 missed cases out of 3418. Each "
    "of these false negatives represents a genuinely sick child whom the model labels healthy; a missed "
    "diagnosis that delays treatment for a disease that remains a leading cause of childhood mortality. By "
    "contrast, the model's false positives (reflected in the somewhat lower precision of 0.9347&ndash;0.9406) only "
    "cost a clinician an extra review or a repeat scan; a far cheaper mistake than a missed diagnosis. As far "
    "as I've reas, in a screening context like this one, it is therefore preferable for the model to lean toward "
    "high recall even atsome expense to precision, and the fact that recall exceeds precision on both splits here "
    "suggests the model has landed on a clinically sensible side of that tradeoff, purely as a consequence of the "
    "class imbalance in the training data rather than any explicit re-weighting. ",
    styles['BodyCustom']))

# ================= Part 5 =================
story.append(Paragraph("5. Model Interpretability (The Weight Heatmap)", styles['SectionHeading']))
story.append(Paragraph(
    "Because every one of the 4096 learned weights corresponds to one specific pixel position, the weight vector "
    "w can be reshaped back into a 64&times;64 grid and viewed as an image in its own right, which turns this "
    "otherwise opaque linear model into something directly interpretable.",
    styles['BodyCustom']))

story.append(Image("weight_heatmap.png", width=4.3*inch, height=3.6*inch))
story.append(Paragraph("Figure 3: Learned pixel weights reshaped to 64&times;64, overlaid with a diverging colour map (red = pushes toward Pneumonia, blue = pushes toward Normal).", styles['Caption']))

story.append(Paragraph(
    "Strong positive weights (red) are concentrated inside both lung fields, specifically in the mid-to-lower "
    "region around rows 30&ndash;50; roughly where the two darkest red hotspots sit just left and "
    "right of the spine. This location seems consistent with where pneumonia-related opacities and consolidation "
    "commonly appear on a chest X-ray, so higher pixel intensity in these zones pushes the model's prediction "
    "toward &ldquo;Pneumonia.&rdquo; Strong negative weights (blue), by contrast, trace the outer rib-cage margins "
    "and the space between the lungs and the chest wall; most visibly two blue patches near rows "
    "45&ndash;60 at the lateral edges of each lung; where clear, well-aerated lung tissue in normal X-rays "
    "produces a consistently low-intensity (dark) pattern that the model associates with &ldquo;Normal.&rdquo; "
    "There are also small positive patches near the top corners (shoulder region) and near row 0 at the "
    "top-centre, which more likely reflect incidental patient-positioning or collarbone artifacts in the dataset "
    "rather than any genuine medical reason. This is a known limitation of a purely pixel-wise linear model: it can "
    "only learn a fixed weight per pixel location and cannot distinguish genuine anatomical structure from"
    "positional correlations the way a spatially-aware model such as a convolutional neural network could. ",
    styles['BodyCustom']))

# ================= Conclusion =================
story.append(Paragraph("Conclusion", styles['SectionHeading']))
story.append(Paragraph(
    "A from-scratch, fully vectorized logistic regression classifier; using only raw pixel intensities as "
    "features; reaches roughly 93% accuracy, 93&ndash;94% precision, and about 97% recall on pneumonia "
    "detection from chest X-rays, without having converged completely within the assigned 1000 training epochs. "
    "Both the loss curves and the metrics table suggest the model is underfitting slightly rather than "
    "overfitting, so more epochs (or a slightly higher learning rate, checked against divergence) would likely "
    "improve results further. The weight heatmap shows that, even as a simple linear model, it has picked up on "
    "spatial patterns broadly consistent with real radiological signs of pneumonia, alongside a few artifacts tied "
    "to incidental patterns in the dataset rather than to the disease itself. ",
    styles['BodyCustom']))

# ================ Appendix ======================
story.append(Paragraph("Appendix", styles['SectionHeading']))
story.append(Paragraph(
    "Requirements to run, install numpy, matplotlib and pillow libraries through pip in a python virtual environment "
    "Followed directory structure, put CS24B057.py in the root, with the data in a new directory 'dictionary', with paths " 
    "dataset/train/NORMAL, dataset/train/PNEUMONIA, dataset/val/NORMAL and dataset/val/PNEUMONIA. "
    "EPOCHs, etc are hardcoded and can be changed in CS24B057.py itself with ease ",
    styles['BodyCustom']))

doc.build(story)
print("Report built: CS24B057.pdf")
