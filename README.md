# Bacterial Segmentation for SIM Microscopy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](#)
[![Cellpose](https://img.shields.io/badge/Segmentation-Cellpose-orange)](#)
[![U-Net](https://img.shields.io/badge/Model-U--Net-purple)](#)

End-to-end, paper-grade pipeline for **Streptococcus pneumoniae** segmentation in **structured-illumination microscopy (SIM)**.  
This repo reproduces the study’s two questions:

1. How do **Cellpose** and a **U-Net trained from scratch (LOIO)** compare under **identical** post-processing?
2. Does **self-supervised denoising (Noise2Void, N2V)** help segmentation quality?

---

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [A) Denoising (Noise2Void)](#a-denoising-noise2void)
  - [B) Cellpose Pipeline](#b-cellpose-pipeline)
  - [C) U-Net LOIO Training & Evaluation](#c-u-net-loio-training--evaluation)
  - [D) Training Audits](#d-training-audits)
- [Unified Post-Processing & Evaluation](#unified-post-processing--evaluation)
- [Scalebar Convention](#scalebar-convention)
- [Inputs, Outputs & File Naming](#inputs-outputs--file-naming)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features
- **Noise2Void denoising** per channel (405/488/561 nm) with QA metrics & **µm scalebars**.
- **Cellpose (cyto, cyto2)** inference, followed by a **unified post-processing** identical to U-Net.
- **U-Net from scratch**, **leave-one-image-out (LOIO)** validation, with overlays and metrics.
- Standardized **Dice / IoU (overlap-only)**, plus CSV/XLSX exports.
- Paper-ready **figures** and **tables** saved in consistent folders and names.

---


> Filenames match the working scripts; you can later normalize (e.g., `full_pipeline.py`, `train_unet.py`) without changing behavior.

---

## Installation

```bash
# 1) Create the environment
conda env create -f environment.yml
conda activate bimap-seg

# 2) Install PyTorch matching your CUDA (see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.1:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 3) (Optional) Track big files with Git LFS
git lfs install
git lfs track "*.czi" "*.tif" "*.tiff" "experiments/**/*.pt" "models/**/*.h5"
git add .gitattributes
Windows note: Use double quotes around paths with spaces, e.g.
python "src\full_pipeline\Full pipeline_segmentation.py"

Quick Start
A) Denoising (Noise2Void)
bash
Copy code
python src/denoising/noise2v_train.py
GUI pickers ask for input images and an output directory.

Processes each channel independently; saves denoised caches next to inputs:

*_denoised_R.npy, *_denoised_G.npy, *_denoised_B.npy

Outputs: QA plots with scalebars + psnr_ssim_3ch.csv (+ BRISQUE/NIQE if available).

B) Cellpose Pipeline
bash
Copy code
python "src/full_pipeline/Full pipeline_segmentation.py"
Select images, optional ROI zips (ground truth), and an output folder.

Uses denoised caches if present; otherwise raw inputs.

Runs cyto and cyto2, then applies the same post-processing as U-Net.

Outputs: overlays with scalebars + segmentation_metrics_overlay_3ch.csv.

C) U-Net LOIO Training & Evaluation
bash
Copy code
python src/unet/train_UNet_from_scratch.py --modes both --in-mode rgb --epochs 50 --batch 8
Auto-trains folds if missing; evaluates when checkpoints exist.

Outputs: fold overlays, per-ROI metrics, aggregated metrics.xlsx.

D) Training Audits
bash
Copy code
python "src/unet/loss_performance metrics.py"
Scans experiments/unet_cv/*/fold_*/train_log.txt.

Outputs: results/audits/training_audit.csv, errors.txt, learning-curve PNGs.

Unified Post-Processing & Evaluation
Applied identically to Cellpose & U-Net masks

Gaussian-smoothed Euclidean Distance Transform (EDT)

Peak detection with min distance = 10 px

Marker-controlled watershed split

Remove small objects: area ≥ 10 px

Shape filter: solidity ≥ 0.30

Evaluation protocol (overlap-only)

Metrics computed only where GT exists (partial GT respected).

Report Dice and IoU consistently across methods.

Why overlap-only?
SIM datasets often have incomplete annotations; overlap-only guards against penalizing predictions in unlabelled regions.

Scalebar Convention
All qualitative panels include a 2 µm scalebar.

Default pixel size: 0.0322 µm/px (override if your metadata differs).

If scalebar length looks off, set pixel size explicitly in the script/CLI.

Inputs, Outputs & File Naming
Inputs

Images: .czi, .tif/.tiff, .jpg/.png

Optional GT: ROI zips (ImageJ/FIJI)

Denoised caches (created by N2V)

bash
Copy code
<basename>_denoised_R.npy
<basename>_denoised_G.npy
<basename>_denoised_B.npy
Typical outputs

bash
Copy code
results/
  figs/                       # Paper-ready figures (PNG, with scalebars)
  tables/                     # CSV/XLSX tables
  audits/                     # Training curves, audit CSVs
  overlays_<image>.png        # Colored masks over raw/denoised inputs
  segmentation_metrics_overlay_3ch.csv
  metrics.xlsx                # Aggregated U-Net results
Suggested figure names (drop-in for Overleaf)

fig1_raw_vs_n2v_<COND>_<ROI>.png

fig2_cellpose_thy_roi3_[raw|n2v]_[cyto|cyto2].png

fig3_thy_roi3_raw_vs_n2v.png

Suggested table names

n2v_indicators_per_roi.csv (Δσ_bg, ΔFWHM, ΔBRISQUE, ΔCNR_gm)

cellpose_metrics_per_roi.csv (Dice/IoU, overlap-only)

unet_loio_raw_vs_n2v_per_roi.csv (per-ROI + deltas)

summary_means.csv (method-level means)

Reproducibility
environment.yml provides the scientific stack; install PyTorch for your CUDA.

Seeds set where possible; exact reproducibility can still vary with CUDA/cuDNN.

Post-processing is parametrized and shared across methods to ensure fair comparison.

Troubleshooting
Big files rejected by GitHub
Use Git LFS or host data externally and link via DATA.md.

Cannot read .czi
Ensure czifile is installed; otherwise convert to TIFF/PNG.

BRISQUE/NIQE missing
QA gracefully skips if the optional packages aren’t available.

Scalebar length wrong
Verify pixel pitch in metadata or pass the value explicitly.

Windows path & spaces
Wrap script paths in quotes:
python "src\full_pipeline\Full pipeline_segmentation.py"

Contributing
Contributions are welcome! Please:

Keep PRs focused and documented.

Add before/after overlays or metric diffs when altering segmentation.

Avoid committing large binaries; prefer cloud links or LFS.

See CONTRIBUTING.md for details.

License
Released under the MIT License. See LICENSE









