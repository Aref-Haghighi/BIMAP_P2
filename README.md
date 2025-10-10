# BIMAP-P2: Characterizing Bacteria Using Shape Descriptors

End-to-end, **reproducible** pipeline for SIM microscopy of *Streptococcus pneumoniae*.  
It covers **Noise2Void denoising**, **Cellpose & U-Net (LOIO) segmentation**, **unified post-processing**, and evaluation with **Dice/IoU (overlap-only)**. Paper-ready overlays, tables, and figures are produced automatically.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Scripts](#scripts)
  - [noise2v_train.py](#noise2v_trainpy)
  - [Cellpose Segmentation with evaluation metrics.py(.py)](#cellpose-segmentation-with-evaluation-metricspypy)
  - [train_UNet_from_scratch.py](#train_unet_from_scratchpy)
  - [loss_performance metrics.py](#loss_performance-metricspy)
- [Installation](#installation)
- [Usage](#usage)
  - [1) Denoising (Noise2Void)](#1-denoising-noise2void)
  - [2) Segmentation with Cellpose](#2-segmentation-with-cellpose)
  - [3) Segmentation with U-Net (Train From Scratch + LOIO)](#3-segmentation-with-u-net-train-from-scratch--loio)
  - [4) Training Audits](#4-training-audits)
- [Unified Post-Processing & Evaluation](#unified-post-processing--evaluation)
- [Scalebar Convention](#scalebar-convention)
- [Inputs, Outputs & Naming](#inputs-outputs--naming)
- [Repository Layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)

---

## Project Overview
This project provides an automated pipeline to **segment, quantify, and visualize** bacteria in SIM images under THY/NHS conditions.  
We directly compare **Cellpose (cyto, cyto2)** with a **U-Net trained from scratch** while holding **post-processing identical**. We also test if **Noise2Void denoising** helps segmentation.

---

## Technologies Used
- Python 3.10+, NumPy, SciPy, scikit-image, pandas, matplotlib  
- **Noise2Void (n2v)** for self-supervised denoising  
- **Cellpose** for pre-trained segmentation  
- **PyTorch** for U-Net training  
- **czifile** for `.czi` reading (optional)  
- Optional QA: **BRISQUE/NIQE**

---

## Scripts

### `noise2v_train.py`
**Folder:** `Noise2Void Denoising/`  
Denoises each channel independently and creates cached arrays next to the input:
*_denoised_R.npy
*_denoised_G.npy
*_denoised_B.npy

csharp
Copy code
Also saves QA plots (with µm scalebars) and `psnr_ssim_3ch.csv`.

### `Cellpose Segmentation with evaluation metrics.py(.py)`
**Folder:** `Segmentation with Cellpose/`  
Runs Cellpose (**cyto**, **cyto2**) on raw/denoised inputs, applies **unified post-processing**, and exports:
- Overlays (PNG) with **2 µm** scalebars  
- `segmentation_metrics_overlay_3ch.csv` (Dice/IoU, overlap-only)

> On Windows the file may appear as `...metrics.py.py`. It works as-is; you can rename to `.py` later.

### `train_UNet_from_scratch.py`
**Folder:** `Segmentation with U-Net Train From Scratch/`  
Trains a U-Net with **Leave-One-Image-Out (LOIO)**; evaluates with the **same post-processing** as Cellpose.  
Outputs overlays and aggregated `metrics.xlsx`.

### `loss_performance metrics.py`
**Folder:** `Segmentation with U-Net Train From Scratch/`  
Parses training logs and exports learning-curve PNGs plus `results/audits/training_audit.csv`.

---

## Installation
```bash
# 1) Create environment (Conda recommended)
conda env create -f environment.yml
conda activate bimap-seg

# 2) Install PyTorch that matches your CUDA (see https://pytorch.org/get-started/locally/)
# Example for CUDA 12.1:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# 3) (Optional) Track large binaries with Git LFS
git lfs install
git lfs track "*.czi" "*.tif" "*.tiff" "experiments/**/*.pt" "models/**/*.h5"
git add .gitattributes
