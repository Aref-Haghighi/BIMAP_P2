# BIMAP_P2
Automated Characterization of Bacteria Using Shape Descriptors and Deep Learning Segmentation
# Bacterial Segmentation for SIM Microscopy

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#)
[![Cellpose](https://img.shields.io/badge/segmentation-cellpose-orange.svg)](#)
[![U-Net](https://img.shields.io/badge/model-U--Net-purple.svg)](#)

End-to-end, paper-grade pipeline for **Streptococcus pneumoniae** segmentation in **structured-illumination microscopy (SIM)**.  
It answers two questions:

1. How do **Cellpose** and a **U-Net trained from scratch (LOIO)** compare under **identical post-processing**?
2. Does **self-supervised denoising (Noise2Void)** improve segmentation quality?

---

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [A) Denoising (Noise2Void)](#a-denoising-noise2void)
  - [B) Cellpose Pipeline (Unified Post-Processing)](#b-cellpose-pipeline-unified-post-processing)
  - [C) U-Net LOIO Training & Evaluation](#c-u-net-loio-training--evaluation)
  - [D) Training Audits](#d-training-audits)
- [Post-Processing & Evaluation](#post-processing--evaluation)
- [Scalebar Convention](#scalebar-convention)
- [Outputs & Paper Artifacts](#outputs--paper-artifacts)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Features
- **Noise2Void (N2V) denoising** per channel (405/488/561 nm) with QA metrics & **µm scalebars**.
- **Cellpose (cyto, cyto2)** inference with a **unified post-processing** identical to U-Net.
- **U-Net from scratch** with **leave-one-image-out (LOIO)** validation and training audits.
- Standardized **Dice/IoU (overlap-only)**, overlays, and CSV/XLSX exports.
- Paper-ready **figure and table** naming to drop into Overleaf.

---

## Repository Structure
