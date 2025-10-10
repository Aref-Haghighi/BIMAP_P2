BIMAP-P2: Characterizing Bacteria Using Shape Descriptors

This repository contains a fully reproducible pipeline for Structured-Illumination Microscopy (SIM) analysis of Streptococcus pneumoniae. It implements Noise2Void denoising, segmentation with Cellpose and a U-Net trained from scratch (LOIO), a unified post-processing chain applied identically to both methods, and quantitative evaluation using Dice/IoU (overlap-only). The code generates paper-ready overlays, tables, figures, and training audits.

TABLE OF CONTENTS

Project Overview

Technologies Used

Scripts
3.1 noise2v_train.py
3.2 Cellpose Segmentation with evaluation metrics.py(.py)
3.3 train_UNet_from_scratch.py
3.4 loss_performance metrics.py

Installation

Usage
5.1 Denoising (Noise2Void)
5.2 Segmentation with Cellpose
5.3 Segmentation with U-Net (Train From Scratch + LOIO)
5.4 Training Audits

Unified Post-Processing and Evaluation

Scalebar Convention

Inputs, Outputs, and Naming

Repository Layout

Reproducibility Notes

Troubleshooting

License

Citation

1) PROJECT OVERVIEW

Goal: robustly segment, quantify, and visualize S. pneumoniae in SIM images acquired under THY and NHS conditions.

Research questions addressed:
• How do Cellpose (cyto, cyto2) and a U-Net trained from scratch compare when both use the same post-processing?
• Does Noise2Void (N2V) self-supervised denoising improve downstream segmentation?

Core deliverables: denoised channel caches, segmentation overlays with 2 µm scalebars, standardized CSV/XLSX metrics, and publication-ready figures/tables.

2) TECHNOLOGIES USED

• Python 3.10+; NumPy, SciPy, scikit-image, pandas, matplotlib
• Noise2Void (n2v) for self-supervised denoising
• Cellpose (cyto, cyto2) for pre-trained segmentation
• PyTorch for U-Net training and inference
• czifile for .czi image support (optional)
• Optional QA: BRISQUE/NIQE

3) SCRIPTS

3.1 noise2v_train.py
Folder: “Noise2Void Denoising/”
Function: per-channel N2V denoising with QA. Caches saved beside inputs as:
*_denoised_R.npy, *_denoised_G.npy, *_denoised_B.npy
Also writes QA plots (with µm scalebars) and psnr_ssim_3ch.csv. No-reference metrics (BRISQUE/NIQE) run if available.

3.2 Cellpose Segmentation with evaluation metrics.py(.py)
Folder: “Segmentation with Cellpose/”
Function: runs Cellpose (cyto & cyto2) on raw or denoised inputs, applies the unified post-processing, and exports:
• Overlays (PNG) with 2 µm scalebars
• segmentation_metrics_overlay_3ch.csv (Dice/IoU; overlap-only)
Note: If the file shows as “.py.py” it still runs; you may rename to a single “.py”.

3.3 train_UNet_from_scratch.py
Folder: “Segmentation with U-Net Train From Scratch/”
Function: U-Net training with Leave-One-Image-Out (LOIO); evaluation uses the same post-processing as Cellpose. Exports overlays and metrics.xlsx.

3.4 loss_performance metrics.py
Folder: “Segmentation with U-Net Train From Scratch/”
Function: parses training logs, produces learning-curve PNGs, and writes results/audits/training_audit.csv (plus errors.txt if anomalies are found).

4) INSTALLATION

Create environment (Conda recommended)
conda env create -f environment.yml
conda activate bimap-seg

Install PyTorch matching your CUDA (see the PyTorch website)
Example (CUDA 12.1):
pip install --index-url https://download.pytorch.org/whl/cu121
 torch torchvision torchaudio

(Optional) Enable Git LFS for large binaries
git lfs install
git lfs track ".czi" ".tif" ".tiff" "experiments/**/.pt" "models/**/*.h5"
git add .gitattributes

Windows note: when a path contains spaces, quote it (e.g., "Segmentation with Cellpose\...py").

5) USAGE

Windows paths with spaces must be quoted, e.g.:
python "Segmentation with Cellpose\Cellpose Segmentation with evaluation metrics.py.py"

5.1 Denoising (Noise2Void)
macOS/Linux:
python "Noise2Void Denoising/noise2v_train.py"
Windows:
python "Noise2Void Denoising\noise2v_train.py"
Outputs: *denoised[R|G|B].npy caches, QA plots (with scalebars), psnr_ssim_3ch.csv.

5.2 Segmentation with Cellpose
macOS/Linux:
python "Segmentation with Cellpose/Cellpose Segmentation with evaluation metrics.py.py"
Windows:
python "Segmentation with Cellpose\Cellpose Segmentation with evaluation metrics.py.py"
Select images, optional ROI zips (GT), and the output folder.
Outputs: overlays with 2 µm scalebars and segmentation_metrics_overlay_3ch.csv.

5.3 Segmentation with U-Net (Train From Scratch + LOIO)
macOS/Linux:
python "Segmentation with U-Net Train From Scratch/train_UNet_from_scratch.py" --modes both --in-mode rgb --epochs 50 --batch 8
Windows:
python "Segmentation with U-Net Train From Scratch\train_UNet_from_scratch.py" --modes both --in-mode rgb --epochs 50 --batch 8
Auto-trains missing folds; evaluates existing checkpoints.
Outputs: per-fold overlays/metrics and aggregated metrics.xlsx.

5.4 Training Audits
macOS/Linux:
python "Segmentation with U-Net Train From Scratch/loss_performance metrics.py"
Windows:
python "Segmentation with U-Net Train From Scratch\loss_performance metrics.py"
Outputs: results/audits/training_audit.csv, learning-curve PNGs, and errors.txt (if any).

6) UNIFIED POST-PROCESSING AND EVALUATION

Applied identically to Cellpose and U-Net masks:
• Gaussian-smoothed Euclidean Distance Transform (EDT)
• Peak detection with min distance = 10 px
• Marker-controlled watershed split
• Remove small objects: area ≥ 10 px
• Shape filter: solidity ≥ 0.30
