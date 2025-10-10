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

## Repository Structure
