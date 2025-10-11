# BIMAP-P2: SIM Bacterial Segmentation

Automated segmentation and denoising of *Streptococcus pneumoniae* structured-illumination microscopy (SIM) images under THY/NHS conditions. The repository compares three strategies—Cellpose (cyto, cyto2), Cellpose after Noise2Void, and a U-Net trained from scratch with leave-one-image-out (LOIO). All methods share one unified post-processing so results are directly comparable. The pipelines save publication-ready overlays with a 2 µm scale bar and per-image metrics (Dice, IoU, PSNR/SSIM, BRISQUE/NIQE, Δσ_bg, ΔCNR_gm, ΔFWHM).

---

## Table of Contents

- [Overview](#overview)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Scripts](#scripts)
- [Reproduce Paper Figures](#reproduce-paper-figures)
- [Troubleshooting](#troubleshooting)
- [License and Citation](#license-and-citation)

---

## Overview

This repository provides an **end-to-end, reproducible workflow** for SIM images of *S. pneumoniae* captured in laboratory (THY) and host-derived (NHS) media.

### What problem this solves
SIM images can be noisy; bacterial boundaries and septa are subtle. Papers often mix different post-processing and metrics, making comparisons unfair. You need a pipeline you can rerun on new ROIs with identical settings.

### What this repository offers
A fair comparison of three strategies:
- **Cellpose pretrained models** (`cyto`, `cyto2`)
- **Cellpose after self-supervised Noise2Void** (per-channel caches)
- **U-Net trained from scratch** with LOIO and 8× test-time augmentation

Every method uses the same post-processing:  
Gaussian-smoothed distance transform → watershed split (fixed peak distance) → small-object removal → solidity filtering.

### Consistent outputs
PNG overlays with a fixed 2 µm scale bar, per-image CSV/Excel metrics (Dice/IoU), denoising QA (PSNR/SSIM, BRISQUE/NIQE, Δσ_bg, ΔCNR_gm, ΔFWHM), and plots for quick review.

### Inputs and ground truth
- Images in `.czi`/`.tif`/`.tiff`/`.png`/`.jpg`
- Optional ImageJ `RoiSet.zip` ground truth
- Optional per-channel N2V caches stored next to the image as `*_denoised_R.npy`, `*_denoised_G.npy`, `*_denoised_B.npy`

### Who this is for
Researchers who want a transparent, rerunnable baseline with figures and tables that drop into manuscripts.

---

## Technologies

- Python 3.10+
- Cellpose, PyTorch
- Noise2Void, CSBDeep, TensorFlow
- scikit-image, OpenCV, NumPy, SciPy
- Matplotlib, Pandas
- czifile, read-roi

---

## Installation

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### Noise2Void denoising and QA

```sh
python scripts/noise2v_train.py --pixel-size 0.0322 --bar-length 2 --bar-loc "lower right"
```

Select images and an output folder in the dialogs. If `*_denoised_R.npy`, `*_denoised_G.npy`, `*_denoised_B.npy` exist next to an image, they are reused (no retraining needed). Saves RAW vs DENOISED panels, difference maps, and QA CSV.

---

### Cellpose segmentation

```sh
python scripts/full_pipeline_cellpose.py
# or, if your filename differs:
python "scripts/Full pipeline_segmentation.py"
```

Pick images, optional ImageJ `RoiSet.zip` ground truth, and an output folder. If all three denoised caches exist next to an image, they are used; otherwise RAW is used. Applies unified post-processing, then saves overlays with a 2 µm scale bar and a CSV of Dice/IoU.

---

### U-Net train and evaluate (LOIO)

```sh
python scripts/train_UNet_from_scratch.py --train --modes both --in-mode g --no-show
```

Runs LOIO training and 8× TTA inference with the same post-processing as Cellpose. Writes `metrics.xlsx`, plots, and overlays under `results/<experiment>/`.

---

## Configuration

- **Pixel size (µm/px):** `--pixel-size 0.0322`
- **Scale-bar length (µm):** `--bar-length 2`
- **Scale-bar location:** `--bar-loc "lower right"` (e.g., "lower right", "lower left")
- **U-Net input mode:** `--in-mode g|rgb|lum`
- **U-Net evaluation modes:** `--modes raw|n2v|both`
- **Suppress windows during training/eval:** `--no-show`

---

## Outputs

**Noise2Void:**  
`results/psnr_ssim_3ch.csv`, RAW/DENOISED panels, absolute and signed difference maps, optional line-profiles/power spectra.

**Cellpose:**  
`results/overlays_cases/*.png`, `segmentation_metrics_overlay_3ch.csv`.

**U-Net:**  
`results/<experiment>/metrics.xlsx`, `results/<experiment>/plots/*.png`, per-image overlays.

**Scale bar:**  
2 µm by default (0.0322 µm/px). Override via CLI flags shown above.

---

## Scripts

**noise2v_train.py**  
Trains or reuses per-channel N2V; writes overlays and QA tables. Looks for \*denoised[RGB].npy next to each image.

**full_pipeline_cellpose.py**  
Segments with cyto and cyto2; if all three denoised caches are present beside an image, they’re used; otherwise RAW is used. Applies the unified post-processing; saves overlays and Dice/IoU CSV. Pairs images ↔ `RoiSet.zip` by ROI number and condition tokens (e.g., WT, THY, NHS).

**train_UNet_from_scratch.py**  
LOIO training/validation/testing with 8× TTA, identical post-processing, per-image Dice/IoU to `metrics.xlsx`, plus plots and overlays.

**loss_performance metrics.py**  
Parses training logs, plots learning curves, flags plateaus/jumps or missing checkpoints, and saves an audit CSV.

---

## Reproduce Paper Figures

- **Denoising panels and QA:** run `noise2v_train.py`; use the saved overlays and `psnr_ssim_3ch.csv`.
- **Cellpose overlays and metrics:** run `full_pipeline_cellpose.py`; use `results/overlays_cases/*.png` and `segmentation_metrics_overlay_3ch.csv`.
- **U-Net overlays, metrics, and plots:** run `train_UNet_from_scratch.py`; use `results/<experiment>/metrics.xlsx` and `results/<experiment>/plots/*.png`.

---

## Troubleshooting

- **BRISQUE/NIQE not installed:**  
  Scripts still run; those columns will be written as missing.

- **CUDA not detected for Cellpose/PyTorch:**  
  Falls back to CPU. Install GPU drivers/CUDA if you want acceleration.

- **Scale bar missing or wrong:**  
  Pass `--pixel-size` and `--bar-length` explicitly to ensure correct units.

---

## License and Citation

**License:** MIT  
Please cite the included paper in `paper/` and this repository (see `CITATION.cff` if present).
