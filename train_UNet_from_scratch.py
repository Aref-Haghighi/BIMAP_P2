#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval-only U-Net LOIO CV (RAW vs N2V) with thin contour overlays, professional 2 µm scalebar,
overlap-only metrics (fair for partial GT), CSV+optional Excel outputs.

Default scalebar is FIXED to 2 µm (like your full pipeline). Change via --bar-um if needed.
"""

import os, json, random, warnings
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as pe
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch import amp

from czifile import imread
from read_roi import read_roi_zip
from skimage.draw import polygon2mask
from skimage import morphology
from skimage.segmentation import find_boundaries, watershed
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.morphology import disk

# ---------------- CLI ----------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--source-exp", type=str, default="unet_cv",
                    help="Folder under ./experiments with trained checkpoints")
parser.add_argument("--exp-name", type=str, default="unet_eval_rgb",
                    help="Destination experiment name (results saved here)")
parser.add_argument("--modes", type=str, default="both", choices=["raw","n2v","both"],
                    help="Which pipelines to evaluate")
parser.add_argument("--folds", type=str, default="", help="Comma list of folds (1-based). Empty=all")
parser.add_argument("--no-tta", action="store_true", help="Disable 8x TTA")
parser.add_argument("--base", type=int, default=32,
                    help="Fallback base channels if missing in source config")
parser.add_argument("--n2v-cache", type=str, default="./n2v_cache",
                    help="Folder with <czi>_denoised_G.npy and (optionally) _R/_G/_B")
# display / overlays
parser.add_argument("--show", dest="show", action="store_true", help="Show panels while saving")
parser.add_argument("--no-show", dest="show", action="store_false", help="Save-only, do not show")
parser.set_defaults(show=True)
parser.add_argument("--gt-linew",   type=float, default=0.9, help="GT contour linewidth")
parser.add_argument("--pred-linew", type=float, default=0.9, help="Prediction contour linewidth")
# scalebar (FIXED LENGTH by default)
parser.add_argument("--bar-um",  type=float, default=2.0,
                    help="Scalebar length in µm (default 2). Use a positive number; no auto in this version.")
parser.add_argument("--bar-loc", type=str, default="lower right",
                    choices=["lower right","lower left","upper right","upper left"],
                    help="Scalebar corner")
parser.add_argument("--bar-text", type=int, default=14, help="Scalebar text size")
# brightness (luminance-based stretch)
parser.add_argument("--lum-low",  type=float, default=2.0,   help="Lower luminance percentile")
parser.add_argument("--lum-high", type=float, default=99.7,  help="Upper luminance percentile")
parser.add_argument("--gain",     type=float, default=1.15,  help="Global gain after stretch")
parser.add_argument("--gamma",    type=float, default=0.90,  help="Gamma (<1 brightens mids)")
# optional: skip excel entirely
parser.add_argument("--no-excel", action="store_true",
                    help="Skip creating .xlsx files (CSVs only)")
args, _ = parser.parse_known_args()

# ---------------- Paths & constants ----------------
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_ROOT   = Path("./experiments") / args.source_exp
DEST_ROOT  = Path("./experiments") / args.exp_name
MODES_TO_RUN = ["raw","n2v"] if args.modes == "both" else [args.modes]
N2V_CACHE    = Path(args.n2v_cache)

# Your dataset lists
CZI_FILES = [
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI3_SIM.czi",
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI2_SIM.czi",
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI1_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI2_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI3_SIM.czi",
]
ROI_FILES = [
    "RoiSet_2D_WT_NADA_THY3.zip",
    "RoiSet_2D_WT_NADA_THY2.zip",
    "RoiSet_2D_WT_NADA_THY1.zip",
    "RoiSet_Contour_bacteria_ROI1.zip",
    "RoiSet_Contour_bacteria_ROI2.zip",
    "RoiSet_Contour_bacteria_ROI3.zip",
]

# Imaging scale
MICRONS_PER_PIXEL = 0.0322  # µm / px

# Sliding window
PATCH_SIZE   = 512
SLIDE_STRIDE = 256
THR_GRID     = np.linspace(0.45, 0.65, 9)

# Post-processing
PROBA_SMOOTH_SIGMA     = 1.0
REMOVE_SMALL_PRED_AREA = 50
OPENING_RADIUS         = 1
WS_MIN_DISTANCE        = 10
FILTER_MIN_AREA        = 10
FILTER_MIN_SOLIDITY    = 0.30

# ---------------- Utils ----------------
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn)

def extract_czyx_from_czi(filename: str) -> np.ndarray:
    """Return (C,Y,X) float [0,1] from CZI (first Z and T)."""
    arr = np.squeeze(imread(filename))
    if arr.ndim == 5:   # T,C,Z,Y,X
        arr = arr[0]
    if arr.ndim == 4:
        if arr.shape[0] < 3 and arr.shape[1] >= 3:
            arr = np.moveaxis(arr, 0, 1)
        arr = arr[:, 0]  # Z=0
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected CZI shape: {arr.shape}")
    return normalize01(arr.astype(np.float32))

def czi_to_green01(filename: str) -> np.ndarray:
    cyx = extract_czyx_from_czi(filename)
    g_idx = 1 if cyx.shape[0] >= 2 else 0
    return normalize01(cyx[g_idx])

def czi_to_rgb01(filename: str) -> np.ndarray:
    """RGB visualization from (B,G,R) (NO enhancement)."""
    cyx = extract_czyx_from_czi(filename)
    c = cyx.shape[0]
    b_idx = 0 if c >= 1 else 0
    g_idx = 1 if c >= 2 else 0
    r_idx = 2 if c >= 3 else c - 1
    R = normalize01(cyx[r_idx]); G = normalize01(cyx[g_idx]); B = normalize01(cyx[b_idx])
    return np.stack([R, G, B], axis=-1)

def roi_zip_to_binary_mask(roi_zip: str, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    rois = read_roi_zip(roi_zip)
    for r in rois.values():
        if 'x' not in r or 'y' not in r or len(r['x']) < 3:
            continue
        y, x = r['y'], r['x']
        poly = polygon2mask(shape, np.column_stack([y, x]))
        mask[poly] = 1
    return mask

def _try_load(path: Path) -> Optional[np.ndarray]:
    try:
        if path.exists():
            return np.load(path).astype(np.float32)
    except Exception:
        pass
    return None

def get_green_by_mode(czi_path: str, mode: str) -> np.ndarray:
    """Return GREEN [0..1] for RAW or cached N2V (no N2V training)."""
    g = czi_to_green01(czi_path)
    if mode == "raw":
        return g
    if mode == "n2v":
        base = Path(czi_path).name
        candidates = [
            N2V_CACHE / f"{base}_denoised_G.npy",
            Path(czi_path).with_suffix("").with_name(base + "_denoised_G.npy"),
        ]
        for p in candidates:
            arr = _try_load(p)
            if arr is not None:
                return normalize01(arr)
        raise FileNotFoundError(f"Missing N2V cache for {base}")
    raise ValueError(mode)

# -------- Brightness (color-faithful) --------
def enhance_rgb_for_display(rgb01: np.ndarray,
                            low_pct: float,
                            high_pct: float,
                            gain: float,
                            gamma: float) -> np.ndarray:
    """Luminance-based global stretch; preserves relative channel ratios."""
    x = rgb01.clip(0, 1).astype(np.float32)
    Y = 0.2126 * x[...,0] + 0.7152 * x[...,1] + 0.0722 * x[...,2]
    lo, hi = np.percentile(Y, low_pct), np.percentile(Y, high_pct)
    if hi <= lo:
        lo, hi = float(Y.min()), float(Y.max() if Y.max() > Y.min() else 1.0)
    x = (x - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0, 1)
    if gain != 1.0:
        x = np.clip(x * gain, 0, 1)
    if gamma != 1.0:
        x = np.power(x, gamma)
    return x

def get_rgb_display(czi_path: str, mode_for_visual: str) -> np.ndarray:
    stem = Path(czi_path).name
    if mode_for_visual == "n2v":
        cand_sets = [
            (N2V_CACHE / f"{stem}_denoised_R.npy",
             N2V_CACHE / f"{stem}_denoised_G.npy",
             N2V_CACHE / f"{stem}_denoised_B.npy"),
            (Path(czi_path).with_suffix("").with_name(stem + "_denoised_R.npy"),
             Path(czi_path).with_suffix("").with_name(stem + "_denoised_G.npy"),
             Path(czi_path).with_suffix("").with_name(stem + "_denoised_B.npy")),
        ]
        for r, g, b in cand_sets:
            R = _try_load(r); G = _try_load(g); B = _try_load(b)
            if R is not None and G is not None and B is not None:
                rgb = np.stack([normalize01(R), normalize01(G), normalize01(B)], axis=-1)
                return enhance_rgb_for_display(rgb, args.lum_low, args.lum_high, args.gain, args.gamma)
    return enhance_rgb_for_display(czi_to_rgb01(czi_path), args.lum_low, args.lum_high, args.gain, args.gamma)

# ---------------- Model ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.bottleneck = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2); self.c4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2);  self.c3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2);  self.c2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2);    self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.pool(x1)); x3 = self.d3(self.pool(x2)); x4 = self.d4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))
        y4 = self.c4(torch.cat([self.up4(x5), x4], dim=1))
        y3 = self.c3(torch.cat([self.up3(y4), x3], dim=1))
        y2 = self.c2(torch.cat([self.up2(y3), x2], dim=1))
        y1 = self.c1(torch.cat([self.up1(y2), x1], dim=1))
        return self.out(y1)

# ---------------- Inference ----------------
def _tukey2d(h, w, alpha=0.6, eps=0.10):
    def tukey(n, a):
        if a <= 0:  return np.ones(n, np.float32)
        if a >= 1:  return np.hanning(n).astype(np.float32)
        win = np.ones(n, np.float32); p=a/2; x=np.linspace(0,1,n)
        m1 = (x < p); m2 = (x > 1 - p)
        win[m1] = 0.5*(1+np.cos(np.pi*(2*x[m1]/a - 1)))
        win[m2] = 0.5*(1+np.cos(np.pi*(2*x[m2]/a - 2/a + 1)))
        return win
    wy = tukey(h, alpha); wx = tukey(w, alpha)
    return (eps + (1.0 - eps) * np.outer(wy, wx)).astype(np.float32)

@torch.no_grad()
def sliding_proba_seamless(model, image: np.ndarray, patch=PATCH_SIZE, stride=SLIDE_STRIDE) -> np.ndarray:
    model.eval()
    H, W = image.shape
    acc  = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    win_full = _tukey2d(patch, patch, alpha=0.6, eps=0.10)
    use_amp = (DEVICE.type == "cuda")
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0 = min(y, H - patch); x0 = min(x, W - patch)
            tile = image[y0:y0+patch, x0:x0+patch]; h0, w0 = tile.shape
            if h0 < patch or w0 < patch:
                tile = np.pad(tile, ((0, patch-h0), (0, patch-w0)), mode='reflect')
            t = torch.from_numpy(tile[None, None].copy()).float().to(DEVICE)
            with amp.autocast('cuda', enabled=use_amp):
                p = torch.sigmoid(model(t))[0,0].detach().cpu().numpy()
            p = p[:h0, :w0]; win = win_full[:h0, :w0]
            acc [y0:y0+h0, x0:x0+w0] += p * win
            wsum[y0:y0+h0, x0:x0+w0] += win
    return acc / (wsum + 1e-8)

@torch.no_grad()
def tta_proba_8x(model, image: np.ndarray) -> np.ndarray:
    rots = [0,1,2,3]; flips = [False, True]
    ps = []
    for r in rots:
        img_r = np.rot90(image, k=r)
        for f in flips:
            img_rf = img_r[:, ::-1] if f else img_r
            prob = sliding_proba_seamless(model, img_rf)
            prob = prob[:, ::-1] if f else prob
            ps.append(np.rot90(prob, k=4-r))
    return np.mean(ps, axis=0)

# -------- Metrics (OVERLAP-ONLY) --------
def dice_iou_overlap_only(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Evaluate only where GT exists: compare (pred ∧ GT) vs GT."""
    pred_eff = np.logical_and(pred_bin > 0, gt_bin > 0)
    gt_bool  = (gt_bin > 0)
    inter = np.logical_and(pred_eff, gt_bool).sum()
    pred_sum = pred_eff.sum()
    gt_sum   = gt_bool.sum()
    dice = 2.0 * inter / (pred_sum + gt_sum + 1e-8)
    iou  = inter / (gt_sum + 1e-8)  # pred_eff ⊆ gt
    return float(dice), float(iou)

def pick_best_threshold_overlap(prob: np.ndarray, gt: np.ndarray, grid=THR_GRID):
    gt_bool = (gt > 0)
    best_t, best_d = grid[0], -1.0
    for t in grid:
        m = (prob >= t).astype(np.uint8)
        m = morphology.remove_small_objects(m.astype(bool), 1).astype(np.uint8)
        d, _ = dice_iou_overlap_only(m, gt_bool.astype(np.uint8))
        if d > best_d:
            best_d, best_t = d, t
    return best_t, best_d

# -------- Postprocess --------
def debug_filter(lbl, min_area=FILTER_MIN_AREA, min_solidity=FILTER_MIN_SOLIDITY):
    props = regionprops(lbl)
    out = np.zeros_like(lbl, dtype=np.int32); k = 1
    for r in props:
        if r.area >= min_area and r.solidity >= min_solidity:
            out[lbl == r.label] = k; k += 1
    return out

def split_touching_cells(mask, min_distance=WS_MIN_DISTANCE):
    proc = mask.astype(bool)
    if proc.sum() == 0:
        return np.zeros_like(mask, dtype=np.int32)
    dist = gaussian(ndi.distance_transform_edt(proc), sigma=0.7)
    coords = peak_local_max(dist, min_distance=min_distance, labels=proc)
    peaks = np.zeros_like(dist, dtype=bool)
    if coords.size: peaks[tuple(coords.T)] = True
    else: peaks[np.unravel_index(np.argmax(dist), dist.shape)] = True
    markers, _ = ndi.label(peaks)
    labels = watershed(-dist, markers, mask=proc)
    labels = morphology.remove_small_objects(labels, min_size=FILTER_MIN_AREA)
    labels = debug_filter(labels, min_area=FILTER_MIN_AREA, min_solidity=FILTER_MIN_SOLIDITY)
    return labels

# -------- Checkpoints --------
def try_load_state(model: nn.Module, path: Path) -> bool:
    if not path.exists():
        print(f"[ckpt] missing: {path}")
        return False
    try:
        try:
            ckpt = torch.load(str(path), map_location=DEVICE, weights_only=True)
        except TypeError:
            ckpt = torch.load(str(path), map_location=DEVICE)
        model.load_state_dict(ckpt, strict=True)
        print(f"[ckpt] loaded: {path}")
        return True
    except Exception as e:
        print(f"[ckpt] failed: {e}")
        return False

# -------- Professional fixed-length µm scalebar --------
def _style_asb_label(asb, color='white', fontsize=12):
    """Safely style the scalebar label across Matplotlib versions."""
    for attr in ('txt_label', 'label_txt'):
        obj = getattr(asb, attr, None)
        if obj is None:
            continue
        try:
            obj.set_color(color)
            obj.set_fontsize(fontsize)
            obj.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])
            return
        except Exception:
            pass
        try:
            for ch in obj.get_children():
                try:
                    ch.set_color(color)
                    ch.set_fontsize(fontsize)
                    ch.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])
                except Exception:
                    pass
            return
        except Exception:
            pass

def add_scale_bar_um(ax,
                     image_shape,
                     microns_per_pixel: float,
                     bar_um: float,
                     loc: str = "lower right",
                     text_size: int = 12):
    """AnchoredSizeBar in data units with WHITE bar + haloed label. FIXED bar length."""
    H, W = image_shape[:2]
    if bar_um is None or bar_um <= 0:
        bar_um = 2.0  # fallback to 2 µm if someone passes <=0
    bar_px = bar_um / microns_per_pixel

    loc_map = {'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4}
    asb = AnchoredSizeBar(ax.transData,
                          size=bar_px,
                          label=f'{bar_um:g} µm',
                          loc=loc_map.get(loc, 4),
                          pad=0.4, sep=4, borderpad=0.8,
                          frameon=False,
                          size_vertical=max(2, int(0.006 * H)),
                          color='white')
    _style_asb_label(asb, color='white', fontsize=text_size)
    try:
        asb.size_bar.set_edgecolor('black')
        asb.size_bar.set_linewidth(max(1, int(0.002 * H)))
    except Exception:
        pass
    ax.add_artist(asb)

# -------- Overlay & save (thin contours + single scalebar) --------
def save_panel_with_contours(base_rgb01: np.ndarray, title: str, out_path: Path,
                             gt_bound: np.ndarray, pred_bound: np.ndarray,
                             gt_linew: float, pred_linew: float,
                             show: bool,
                             microns_per_pixel: float,
                             bar_um: float,
                             bar_loc: str,
                             bar_text_size: int):
    H, W, _ = base_rgb01.shape
    fig, ax = plt.subplots(1, 1, figsize=(10, 10*H/W), facecolor='black')
    ax.imshow((base_rgb01 * 255).astype(np.uint8), interpolation='nearest')
    ax.set_title(title, color='white', fontsize=18, pad=14)
    ax.axis('off')

    if gt_bound is not None and gt_bound.any():
        ax.contour(gt_bound.astype(float), levels=[0.5], colors='deepskyblue', linewidths=gt_linew)
    if pred_bound is not None and pred_bound.any():
        ax.contour(pred_bound.astype(float), levels=[0.5], colors='yellow', linewidths=pred_linew)

    # Draw scalebar ONCE (FIXED length)
    add_scale_bar_um(ax, (H, W), microns_per_pixel,
                     bar_um=bar_um, loc=bar_loc, text_size=bar_text_size)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, facecolor='black', bbox_inches='tight', pad_inches=0.02)
    if show: plt.show()
    plt.close(fig)

# -------- Excel helper --------
def try_make_excel_writer(path: Path, skip: bool = False) -> Optional[pd.ExcelWriter]:
    if skip:
        return None
    for engine in ("openpyxl", "xlsxwriter"):
        try:
            return pd.ExcelWriter(path, engine=engine)
        except Exception:
            continue
    print(f"[excel] NOTE: No Excel engine found. Skipping: {path}")
    return None

# -------- High-level runners --------
def read_source_base(mode_root: Path, fallback_base: int) -> int:
    cfg = mode_root / "run_config.json"
    if cfg.exists():
        try:
            j = json.loads(cfg.read_text())
            return int(j.get("base", fallback_base))
        except Exception:
            pass
    return fallback_base

def load_images_masks_for_mode(mode: str):
    imgs, msks = [], []
    for czi, roi in zip(CZI_FILES, ROI_FILES):
        g = get_green_by_mode(czi, mode)
        m = roi_zip_to_binary_mask(roi, g.shape)
        imgs.append(g); msks.append(m)
    return imgs, msks

@torch.no_grad()
def eval_fold(model: nn.Module, fold_dir: Path,
              val_img, val_msk, test_img, test_msk,
              czi_test_path: str, mode_label: str,
              gt_linew: float, pred_linew: float,
              tta=True):
    # validation proba (+ smoothing)
    val_prob = tta_proba_8x(model, val_img) if tta else sliding_proba_seamless(model, val_img)
    if PROBA_SMOOTH_SIGMA > 0:
        val_prob = gaussian(val_prob, sigma=PROBA_SMOOTH_SIGMA, preserve_range=True)

    # threshold chosen by maximizing OVERLAP-ONLY Dice on validation set
    thr, _  = pick_best_threshold_overlap(val_prob, (val_msk > 0).astype(np.uint8), THR_GRID)
    (fold_dir / "val_threshold.txt").write_text(f"thr={thr:.4f}  (overlap-only)\n")

    # test proba (+ smoothing)
    test_prob = tta_proba_8x(model, test_img) if tta else sliding_proba_seamless(model, test_img)
    if PROBA_SMOOTH_SIGMA > 0:
        test_prob = gaussian(test_prob, sigma=PROBA_SMOOTH_SIGMA, preserve_range=True)

    # threshold + morphology
    pred_bin = (test_prob >= thr).astype(np.uint8)
    if REMOVE_SMALL_PRED_AREA > 0:
        pred_bin = morphology.remove_small_objects(pred_bin.astype(bool), min_size=REMOVE_SMALL_PRED_AREA).astype(np.uint8)
    if OPENING_RADIUS > 0:
        pred_bin = morphology.binary_opening(pred_bin.astype(bool), footprint=disk(OPENING_RADIUS)).astype(np.uint8)

    # instance split + filter
    pred_lbl   = split_touching_cells(pred_bin, min_distance=WS_MIN_DISTANCE)
    pred_binPP = (pred_lbl > 0).astype(np.uint8)
    gt_bin     = (test_msk > 0).astype(np.uint8)

    # === OVERLAP-ONLY METRICS ===
    dice, iou = dice_iou_overlap_only(pred_binPP, gt_bin)

    # boundaries for visualization
    gt_bound   = find_boundaries(gt_bin.astype(bool),   mode='outer')
    pred_bound = find_boundaries(pred_binPP.astype(bool), mode='outer')

    # overlay on ENHANCED RGB
    base_rgb = get_rgb_display(czi_test_path, mode_for_visual=mode_label)

    # save figure (thin contours + fixed 2 µm scalebar unless overridden)
    title = f"{Path(czi_test_path).name} — {mode_label.upper()} (Dice {dice:.3f}, IoU {iou:.3f}, thr={thr:.3f} | overlap-only)"
    save_panel_with_contours(base_rgb, title, fold_dir / "test_overlay.png",
                             gt_bound, pred_bound, args.gt_linew, args.pred_linew,
                             show=args.show,
                             microns_per_pixel=MICRONS_PER_PIXEL,
                             bar_um=args.bar_um,
                             bar_loc=args.bar_loc,
                             bar_text_size=args.bar_text)

    # also save masks
    cv2.imwrite(str(fold_dir / "pred_mask.png"), (pred_binPP * 255).astype(np.uint8))
    cv2.imwrite(str(fold_dir / "pred_labels.png"), pred_lbl.astype(np.uint16))

    # save metrics json
    with open(fold_dir / "test_metrics.json", "w") as f:
        json.dump({
            "threshold": float(thr),
            "dice_overlap_only": float(dice),
            "iou_overlap_only": float(iou),
            "note": "Metrics computed only on prediction overlapped with GT.",
            "gt_linewidth": float(gt_linew),
            "pred_linewidth": float(pred_linew),
            "scalebar_um": float(args.bar_um),
            "scalebar_loc": args.bar_loc
        }, f, indent=2)

    return dice, iou, float(thr)

def run_eval_for_mode(mode: str, folds: List[int]):
    src_mode_root  = SRC_ROOT  / mode
    dest_mode_root = DEST_ROOT / mode
    dest_mode_root.mkdir(parents=True, exist_ok=True)

    base = read_source_base(src_mode_root, args.base)

    # tta config
    tta = True
    cfg_src = src_mode_root / "run_config.json"
    if cfg_src.exists():
        try:
            j = json.loads(cfg_src.read_text()); tta = bool(j.get("tta_8x", True))
        except Exception:
            pass
    if args.no_tta: tta = False

    # persist minimal run config
    (dest_mode_root / "run_config.json").write_text(json.dumps({
        "mode": mode, "base": base, "tta_8x": tta, "source_exp": str(SRC_ROOT.name),
        "display": {"lum_low": args.lum_low, "lum_high": args.lum_high, "gain": args.gain, "gamma": args.gamma},
        "evaluation": "overlap-only (prediction clipped to GT) for Dice/IoU",
        "contours": {"gt_linew": args.gt_linew, "pred_linew": args.pred_linew},
        "scalebar": {"um": args.bar_um, "loc": args.bar_loc, "text": args.bar_text}
    }, indent=2))

    # load data
    IMGS, MSKS = load_images_masks_for_mode(mode)
    N = len(IMGS)

    # CSV path (overlap-only)
    csv_path = dest_mode_root / "cv_results.csv"
    rows: List[List[object]] = []

    # evaluate folds
    for test_id in folds:
        val_id = (test_id + 1) % N
        print(f"\n== {mode.upper()} | FOLD {test_id+1}/{N} ==")

        model = UNet(1,1,base).to(DEVICE)
        ckpt_path = src_mode_root / f"fold_{test_id+1}" / "best.pt"
        if not try_load_state(model, ckpt_path):
            print(f"[skip] fold {test_id+1} (missing/incompatible checkpoint)")
            continue

        fold_dir = dest_mode_root / f"fold_{test_id+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "used_checkpoint.txt").write_text(str(ckpt_path))

        dice, iou, thr = eval_fold(
            model, fold_dir,
            val_img=IMGS[val_id],   val_msk=MSKS[val_id],
            test_img=IMGS[test_id], test_msk=MSKS[test_id],
            czi_test_path=CZI_FILES[test_id], mode_label=mode,
            gt_linew=args.gt_linew, pred_linew=args.pred_linew,
            tta=tta
        )

        rows.append([test_id+1, os.path.basename(CZI_FILES[test_id]), dice, iou, thr])

    # write CSV (always)
    pd.DataFrame(rows, columns=["Fold","TestImage","Dice","IoU","Threshold"]).to_csv(csv_path, index=False)

    # write Excel (per mode) only if engine available / not skipped
    xlw = try_make_excel_writer(dest_mode_root / "cv_results.xlsx", skip=args.no_excel)
    if xlw is not None:
        with xlw:
            pd.DataFrame(rows, columns=["Fold","TestImage","Dice","IoU","Threshold"]).to_excel(
                xlw, index=False, sheet_name="overlap_only"
            )

    # summaries (text)
    def summarize(rows):
        if not rows: return "no folds evaluated"
        a = np.array([[r[2], r[3]] for r in rows], dtype=np.float32)  # Dice, IoU
        if len(a) > 1:
            return f"Dice mean±SD = {a[:,0].mean():.4f} ± {a[:,0].std(ddof=1):.4f} | IoU mean±SD = {a[:,1].mean():.4f} ± {a[:,1].std(ddof=1):.4f}"
        else:
            return f"Dice = {a[0,0]:.4f} | IoU = {a[0,1]:.4f}"
    (dest_mode_root / "cv_summary.txt").write_text(summarize(rows) + "\n(overlap-only metrics)\n")

    return rows

def top_level_excel_summary(raw_rows, n2v_rows):
    out_xlsx = DEST_ROOT / "cv_compare.xlsx"
    xlw = try_make_excel_writer(out_xlsx, skip=args.no_excel)
    if xlw is None:
        print(f"[excel] Skipping top-level Excel summary (no engine). CSVs are still saved per mode.")
        return
    with xlw:
        if raw_rows:
            pd.DataFrame(raw_rows, columns=["Fold","TestImage","Dice","IoU","Threshold"]).to_excel(
                xlw, index=False, sheet_name="raw_overlap_only")
        if n2v_rows:
            pd.DataFrame(n2v_rows, columns=["Fold","TestImage","Dice","IoU","Threshold"]).to_excel(
                xlw, index=False, sheet_name="n2v_overlap_only")

        def agg(rows):
            if not rows: return {"Dice_mean": np.nan, "Dice_sd": np.nan, "IoU_mean": np.nan, "IoU_sd": np.nan, "N": 0}
            a = np.array([[r[2], r[3]] for r in rows], dtype=np.float32)
            return {"Dice_mean": float(a[:,0].mean()),
                    "Dice_sd": float(a[:,0].std(ddof=1) if len(a)>1 else 0.0),
                    "IoU_mean": float(a[:,1].mean()),
                    "IoU_sd": float(a[:,1].std(ddof=1) if len(a)>1 else 0.0),
                    "N": len(a)}
        summary_rows = []
        summary_rows.append({"Mode":"RAW overlap-only", **agg(raw_rows)})
        summary_rows.append({"Mode":"N2V overlap-only", **agg(n2v_rows)})
        pd.DataFrame(summary_rows).to_excel(xlw, index=False, sheet_name="summary")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    (DEST_ROOT / "experiment.json").write_text(json.dumps({
        "source_exp": str(SRC_ROOT.name),
        "dest_exp": str(DEST_ROOT.name),
        "modes": MODES_TO_RUN,
        "tta_8x": not args.no_tta,
        "postprocess": {
            "proba_gaussian_sigma": PROBA_SMOOTH_SIGMA,
            "remove_small_area": REMOVE_SMALL_PRED_AREA,
            "binary_opening_radius": OPENING_RADIUS,
            "min_solidity": FILTER_MIN_SOLIDITY,
            "min_area": FILTER_MIN_AREA,
            "ws_min_distance": WS_MIN_DISTANCE
        },
        "display": {"lum_low": args.lum_low, "lum_high": args.lum_high, "gain": args.gain, "gamma": args.gamma},
        "thr_grid": list(map(float, THR_GRID)),
        "microns_per_pixel": MICRONS_PER_PIXEL,
        "scalebar": {"um": args.bar_um, "loc": args.bar_loc, "text": args.bar_text},
        "evaluation": "Dice/IoU computed only on prediction overlapped with GT (clipped-to-GT). Threshold maximizes overlap-only Dice.",
        "contours": {"gt_linew": args.gt_linew, "pred_linew": args.pred_linew}
    }, indent=2))

    # folds
    N = len(CZI_FILES)
    if args.folds.strip():
        wanted = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
        folds = [i-1 for i in wanted if 1 <= i <= N]
    else:
        folds = list(range(N))

    raw_rows: List[List[object]] = []
    n2v_rows: List[List[object]] = []

    if "raw" in MODES_TO_RUN:
        raw_rows = run_eval_for_mode("raw", folds)
    if "n2v" in MODES_TO_RUN:
        n2v_rows = run_eval_for_mode("n2v", folds)

    top_level_excel_summary(raw_rows, n2v_rows)

if __name__ == "__main__":
    main()
