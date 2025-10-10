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





