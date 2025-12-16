CBIS‑DDSM Mammogram Classification with Image Enhancement

This repository contains a CNN‑based pipeline for benign vs malignant classification on the CBIS‑DDSM mammography dataset. The focus is on how classical preprocessing filters (bilateral, CLAHE, histogram equalization, gamma correction, Gaussian, Wiener, etc.) affect downstream classification performance and image quality.
1. Project Overview

Breast cancer screening mammograms are low‑contrast and noisy, which can make automated detection challenging.
This project builds a reproducible framework to:

    Preprocess CBIS‑DDSM patches with different enhancement/denoising filters.

    Train and evaluate a CNN classifier on fixed patient‑wise splits.

    Compare accuracy, ROC AUC, per‑class metrics, and image quality scores (PSNR, SSIM, optionally BRISQUE) across preprocessing pipelines.

The code is structured so each experiment (baseline, single filter, composite filters) is fully reproducible from saved file lists.

Preprocessing Pipelines

All image filters are implemented in src/preprocessing/pipelines.py. Examples include:

    bilateral: edge‑preserving denoising.

    clahe: local contrast enhancement (adaptive histogram equalization).

    hist_eq: global histogram equalization.

    gamma_08 / gamma_12: gamma correction using LUTs.

    median: median filtering for speckle noise.

    gaussian: Gaussian blur.

    unsharp: unsharp masking for mild sharpening.

    nlm: non‑local means denoising.

    wiener: frequency‑domain Wiener filter.

    histeq_median: composite filter (histogram equalization + median denoise).

Each pipeline is registered in a PIPELINES dict so it can be called by name.

Model Training

The baseline model is a small CNN for binary classification on grayscale mammogram patches (e.g., 128×128).

Evaluation:
The evaluation script typically:

    Computes accuracy, precision, recall, F1.

    Computes ROC AUC.

    Saves a confusion matrix and ROC curve plots under results/.

    Writes a classification report (per‑class metrics) to a text file.

These artifacts can then be copied into experiments/expX_<pipeline>/results/ for archiving.

To quantify how much each filter alters images, src/utils/image_quality.py provides metrics:

    PSNR (Peak Signal‑to‑Noise Ratio): reference‑based, measures pixel‑wise distortion.

    SSIM (Structural Similarity Index): reference‑based, emphasizes structural similarity.
Typical Experiments

For each pipeline:

    Apply pipeline to preprocessed data into data/enhanced/<pipeline>/.

    Rebuild splits from lists with source_root=data/enhanced/<pipeline>/.

    Train CNN and save model under experiments/exp_<pipeline>/models/.

    Evaluate on test set and save metrics/plots under experiments/exp_<pipeline>/results/.

    Optionally, compute PSNR/SSIM stats between data/preprocessed and data/enhanced/<pipeline>.

Comparisons in the report typically include:

    Baseline vs. bilateral vs. CLAHE vs. hist_eq vs. gamma vs. gaussian/wiener.

    Accuracy, ROC AUC, per‑class recall (especially malignant), confusion matrices.

    PSNR and SSIM to discuss how much each filter distorts the original image.
Possible Extensions

    Transfer‑learning baselines (e.g., EfficientNet, ResNet) at higher resolutions.

    Multi‑class classification (normal vs benign vs malignant) on CBIS‑DDSM or INbreast.

    More advanced quality metrics (e.g., learned no‑reference IQA models) and correlation analysis between filter quality and CNN performance.

    Calibration and threshold tuning for clinically meaningful sensitivity/specificity trade‑offs.
Dataset used: CBIS-DDSM Dataset
