# BIMAP_P2 — SIM Bacterial Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](#)
[![Cellpose](https://img.shields.io/badge/Segmentation-Cellpose-orange)](#)
[![U-Net](https://img.shields.io/badge/Model-U--Net-purple)](#)

End-to-end, paper-grade pipeline for **Streptococcus pneumoniae** segmentation in **Structured-Illumination Microscopy (SIM)**.

It answers two questions:

1. How do **Cellpose** and a **U-Net trained from scratch (LOIO)** compare under **identical post-processing**?
2. Does **self-supervised denoising (Noise2Void, N2V)** improve segmentation quality?

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [A) Denoising (Noise2Void)](#a-denoising-noise2void)
  - [B) Segmentation with Cellpose](#b-segmentation-with-cellpose)
  - [C) Segmentation with U-Net (Train From Scratch + LOIO)](#c-segmentation-with-u-net-train-from-scratch--loio)
  - [D) Training Audits](#d-training-audits)
- [Unified Post-Processing & Evaluation](#unified-post-processing--evaluation)
- [Scalebar Convention](#scalebar-convention)
- [Inputs, Outputs & Naming](#inputs-outputs--naming)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features

- **Noise2Void (N2V)** denoising per channel (405/488/561 nm) with QA metrics and **µm scalebars**  
- **Cellpose (cyto, cyto2)** inference with a **unified post-processing** identical to U-Net  
- **U-Net from scratch** with **Leave-One-Image-Out (LOIO)** validation + overlays & metrics  
- Standardized **Dice / IoU (overlap-only)**, CSV/XLSX exports, and paper-ready figures/tables

---

## Repository Structure

> Folder and file names below reflect your current GitHub layout (including spaces).  
> On Windows, always **quote paths** that contain spaces.





---

## Installation

```bash
# 1) Create the environment
conda env create -f environment.yml
conda activate bimap-seg

# 2) Install PyTorch that matches your CUDA (see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.1:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 3) (Optional) Track large binaries with Git LFS
git lfs install
git lfs track "*.czi" "*.tif" "*.tiff" "experiments/**/*.pt" "models/**/*.h5"
git add .gitattributes




A) Denoising (Noise2Void)
# Windows
python "Noise2Void Denoising\noise2v_train.py"

# macOS/Linux (quotes still recommended)
python "Noise2Void Denoising/noise2v_train.py"


GUI pickers ask for input images and an output directory

Processes each channel independently; saves denoised caches next to inputs:

*_denoised_R.npy, *_denoised_G.npy, *_denoised_B.npy

Outputs: QA plots with scalebars + psnr_ssim_3ch.csv (+ BRISQUE/NIQE if available)




B) Segmentation with Cellpose
# Use the filename exactly as it appears on GitHub.
# If your file ends with ".py.py", keep it or rename to ".py" first.

# Windows (as-is)
python "Segmentation with Cellpose\Cellpose Segmentation with evaluation metrics.py.py"

# macOS/Linux (as-is)
python "Segmentation with Cellpose/Cellpose Segmentation with evaluation metrics.py.py"


Select images, optional ROI zips (ground truth), and an output folder

Uses denoised caches if present; otherwise raw inputs

Runs cyto and cyto2, then applies the same post-processing as U-Net

Outputs: overlays with 2 µm scalebars + segmentation_metrics_overlay_3ch.csv




C) Segmentation with U-Net (Train From Scratch + LOIO)
# Windows
python "Segmentation with U-Net Train From Scratch\train_UNet_from_scratch.py" --modes both --in-mode rgb --epochs 50 --batch 8

# macOS/Linux
python "Segmentation with U-Net Train From Scratch/train_UNet_from_scratch.py" --modes both --in-mode rgb --epochs 50 --batch 8


Auto-trains folds if missing; evaluates when checkpoints exist

Outputs: fold overlays, per-ROI metrics, aggregated metrics.xlsx in results/

D) Training Audits
# Windows
python "Segmentation with U-Net Train From Scratch\loss_performance metrics.py"

# macOS/Linux
python "Segmentation with U-Net Train From Scratch/loss_performance metrics.py"


Scans experiments/unet_cv/*/fold_*/train_log.txt

Outputs: results/audits/training_audit.csv, errors.txt, learning-curve PNGs

Unified Post-Processing & Evaluation

Applied identically to Cellpose & U-Net masks

Gaussian-smoothed Euclidean Distance Transform (EDT)

Peak detection with min distance = 10 px

Marker-controlled watershed split

Remove small objects: area ≥ 10 px

Shape filter: solidity ≥ 0.30

Evaluation protocol (overlap-only)

Metrics computed only where ground truth exists (partial GT respected)

Report Dice and IoU consistently across methods

Scalebar Convention

All qualitative panels include a 2 µm scalebar

Default pixel size: 0.0322 µm/px (override if your metadata differs)

If the scalebar length looks off, set pixel size explicitly in script/CLI

Inputs, Outputs & Naming

Inputs

Images: .czi, .tif/.tiff, .jpg/.png

Optional GT: ROI zips (ImageJ/Fiji)

Denoised caches (created by N2V)

<basename>_denoised_R.npy
<basename>_denoised_G.npy
<basename>_denoised_B.npy


Typical outputs

results/
  figs/                       # paper-ready figures (PNG, with scalebars)
  tables/                     # CSV/XLSX tables
  audits/                     # training curves, audit CSVs
  overlays_<image>.png        # colored masks on raw/denoised inputs
  segmentation_metrics_overlay_3ch.csv
  metrics.xlsx                # aggregated U-Net results


Suggested figure names (drop-in for Overleaf)

fig1_raw_vs_n2v_<COND>_<ROI>.png

fig2_cellpose_thy_roi3_[raw|n2v]_[cyto|cyto2].png

fig3_thy_roi3_raw_vs_n2v.png

Suggested table names

n2v_indicators_per_roi.csv (Δσ_bg, ΔFWHM, ΔBRISQUE, ΔCNR_gm)

cellpose_metrics_per_roi.csv (Dice/IoU, overlap-only)

unet_loio_raw_vs_n2v_per_roi.csv (per-ROI + deltas)

summary_means.csv (method-level means)




