BIMAP-P2: Characterizing Bacteria Using Shape Descriptors

This project is a reproducible pipeline for SIM microscopy of Streptococcus pneumoniae. It covers Noise2Void denoising, Cellpose and U-Net (trained from scratch with LOIO) segmentation, a unified post-processing chain, and quantitative evaluation with Dice/IoU (overlap-only). The code produces paper-ready overlays, tables, and figures.

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

PROJECT OVERVIEW

Goal: Segment, quantify, and visualize Streptococcus pneumoniae in SIM images under THY and NHS conditions.

We address two questions:

Comparison between Cellpose (cyto, cyto2) and a U-Net trained from scratch when both use the exact same post-processing.

Whether self-supervised denoising with Noise2Void (N2V) improves downstream segmentation.

Core deliverables:

Denoised channel caches, segmentation overlays with 2 µm scalebars, and CSV/XLSX metrics.

Figures and tables saved with consistent names for easy inclusion in Overleaf.

TECHNOLOGIES USED

Python 3.10+

NumPy, SciPy, scikit-image, pandas, matplotlib

Noise2Void (n2v)

Cellpose (cyto, cyto2)

PyTorch (for U-Net training)

czifile (optional, for .czi)

Optional QA: BRISQUE/NIQE

SCRIPTS

3.1 noise2v_train.py
Location: “Noise2Void Denoising/”
Function: Self-supervised denoising per channel. Saves cached arrays next to inputs:
*_denoised_R.npy
*_denoised_G.npy
*_denoised_B.npy
Also writes QA plots (with µm scalebars) and psnr_ssim_3ch.csv.
No-reference metrics (BRISQUE/NIQE) are included when available.

3.2 Cellpose Segmentation with evaluation metrics.py(.py)
Location: “Segmentation with Cellpose/”
Function: Runs Cellpose (cyto and cyto2) on raw or denoised inputs, applies the unified post-processing chain, and exports:

Overlays with 2 µm scalebars

segmentation_metrics_overlay_3ch.csv (Dice/IoU, overlap-only)
Note: If your file appears as “.py.py” it still runs; feel free to rename to a single “.py”.

3.3 train_UNet_from_scratch.py
Location: “Segmentation with U-Net Train From Scratch/”
Function: Trains a U-Net with Leave-One-Image-Out (LOIO) and evaluates using the exact same post-processing as Cellpose. Exports overlays and aggregated metrics.xlsx.

3.4 loss_performance metrics.py
Location: “Segmentation with U-Net Train From Scratch/”
Function: Parses training logs and exports learning-curve PNGs and results/audits/training_audit.csv. Also writes errors.txt if anomalies are found.

INSTALLATION

Create the environment (Conda recommended)
conda env create -f environment.yml
conda activate bimap-seg

Install PyTorch that matches your CUDA (see pytorch.org/get-started/locally/)
Example (CUDA 12.1):
pip install --index-url https://download.pytorch.org/whl/cu121
 torch torchvision torchaudio

(Optional) Large files with Git LFS
git lfs install
git lfs track ".czi" ".tif" ".tiff" "experiments/**/.pt" "models/**/*.h5"
git add .gitattributes

Windows note: When a folder or file name contains spaces, wrap the path in double quotes.

USAGE

Windows paths with spaces must be quoted, for example:
python "Segmentation with Cellpose\Cellpose Segmentation with evaluation metrics.py.py"

5.1 Denoising (Noise2Void)
macOS/Linux:
python "Noise2Void Denoising/noise2v_train.py"
Windows:
python "Noise2Void Denoising\noise2v_train.py"

Steps:

Choose input images and output directory via GUI pickers.

Outputs: *denoised[R|G|B].npy caches, QA plots (with scalebars), psnr_ssim_3ch.csv.

5.2 Segmentation with Cellpose
macOS/Linux:
python "Segmentation with Cellpose/Cellpose Segmentation with evaluation metrics.py.py"
Windows:
python "Segmentation with Cellpose\Cellpose Segmentation with evaluation metrics.py.py"

Steps:

Select images, optional ROI zips (ground truth), and an output folder.

Uses denoised caches if present; otherwise runs on raw inputs.

Outputs: overlays with 2 µm scalebars and segmentation_metrics_overlay_3ch.csv.

5.3 Segmentation with U-Net (Train From Scratch + LOIO)
macOS/Linux:
python "Segmentation with U-Net Train From Scratch/train_UNet_from_scratch.py" --modes both --in-mode rgb --epochs 50 --batch 8
Windows:
python "Segmentation with U-Net Train From Scratch\train_UNet_from_scratch.py" --modes both --in-mode rgb --epochs 50 --batch 8

Steps:

Auto-trains any missing folds and evaluates when checkpoints exist.

Outputs: per-fold overlays/metrics and aggregated metrics.xlsx.

5.4 Training Audits
macOS/Linux:
python "Segmentation with U-Net Train From Scratch/loss_performance metrics.py"
Windows:
python "Segmentation with U-Net Train From Scratch\loss_performance metrics.py"

Outputs:

results/audits/training_audit.csv

results/audits/*.png learning curves

results/audits/errors.txt (if any)

UNIFIED POST-PROCESSING AND EVALUATION

Applied identically to Cellpose and U-Net masks:

Gaussian-smoothed Euclidean Distance Transform (EDT)

Peak detection with minimum distance = 10 pixels

Marker-controlled watershed split

Remove small objects: area >= 10 pixels

Shape filter: solidity >= 0.30

Evaluation protocol (overlap-only):

Compute Dice and IoU only in regions covered by ground truth (to respect partial GT).

Metrics reported consistently for all methods.

SCALEBAR CONVENTION

All qualitative panels include a 2 µm scalebar.

Default pixel size: 0.0322 µm/px unless a per-file value is available from metadata.

If your microscope pixel pitch differs, set it explicitly in script arguments or constants.

INPUTS, OUTPUTS, AND NAMING

Inputs:

Images: .czi, .tif/.tiff, .jpg/.png

Optional ground truth: ROI zips (Fiji/ImageJ)

Denoised caches produced by N2V:

<basename>_denoised_R.npy

<basename>_denoised_G.npy

<basename>_denoised_B.npy

Typical outputs (created during runs):

results/figs/ paper-ready figures (PNG, with scalebars)

results/tables/ CSV/XLSX tables

results/audits/ training curves and audit CSVs

overlays_<image>.png colored masks on raw/denoised inputs

segmentation_metrics_overlay_3ch.csv

metrics.xlsx aggregated U-Net results

Suggested figure names (for Overleaf):

fig1_raw_vs_n2v_<COND>_<ROI>.png

fig2_cellpose_thy_roi3_[raw|n2v]_[cyto|cyto2].png

fig3_thy_roi3_raw_vs_n2v.png

Suggested table names:

n2v_indicators_per_roi.csv (Δσ_bg, ΔFWHM, ΔBRISQUE, ΔCNR_gm)

cellpose_metrics_per_roi.csv (Dice/IoU, overlap-only)

unet_loio_raw_vs_n2v_per_roi.csv (per-ROI + deltas)

summary_means.csv (method-level means)

REPOSITORY LAYOUT

BIMAP_P2/
Noise2Void Denoising/
noise2v_train.py
Segmentation with Cellpose/
Cellpose Segmentation with evaluation metrics.py(.py)
Segmentation with U-Net Train From Scratch/
train_UNet_from_scratch.py
loss_performance metrics.py
results/ created at runtime
experiments/ created at runtime
environment.yml
.gitignore
LICENSE
README.txt (this file)

REPRODUCIBILITY NOTES

environment.yml specifies the scientific stack. Install PyTorch to match your CUDA.

Seeds are set where possible; exact replication may still vary across GPUs/CUDA/cuDNN.

Post-processing parameters are shared and centralized so Cellpose and U-Net are compared fairly.

TROUBLESHOOTING

“.py.py” script name:

Works as-is; rename to “.py” for cleanliness if desired.

Cannot read .czi files:

Ensure the “czifile” package is installed or convert data to TIFF/PNG.

Big files rejected by GitHub:

Use Git LFS for .czi/.tif and model checkpoints or host the data externally and link in a DATA.md.

BRISQUE/NIQE not installed:

QA plots and core metrics still run; those two are optional and will be skipped if unavailable.

Scalebar length looks wrong:

Verify pixel pitch from metadata or set the pixel size explicitly in the script.

Windows paths with spaces:

Always wrap script paths in double quotes.

LICENSE

This project is released under the MIT License. See the LICENSE file.
