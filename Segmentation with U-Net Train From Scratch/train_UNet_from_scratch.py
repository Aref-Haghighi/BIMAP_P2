#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net LOIO training + evaluation (RAW vs N2V) with 8× TTA.
Post-processing is now IDENTICAL to your full-pipeline file:

    • Threshold → watershed split (DT σ=0.7, min_distance=10)
    • Remove small labels: min_size = 10
    • Region filter: keep labels with area ≥ 10 and solidity ≥ 0.30
    (No probability smoothing, no pre-split small-object removal, no opening.)

Interactive I/O:
- If you don't pass --images / --rois / --out-dir, GUI pickers open:
  1) pick input images (CZI/JPG/PNG/TIF) — multi-select
  2) pick GT ROI .zip files — multi-select
  3) pick output directory

Supervisor-facing training evidence:
- train_log.csv (epoch, train_loss, val_dice, time_s)
- learning_curves.png
- train_log.txt
- Early stopping (--patience, --min-epochs) + optional cosine LR (--cosine)

Flexible:
- CZI + JPG/PNG/TIF support
- --in-mode {g,rgb,lum} to use 1-channel green, 3-channel RGB, or luminance
- N2V caches per channel (_denoised_R/G/B.npy; G-only also supported)
- Auto-train missing folds during evaluation
- Paper metrics (per-image Dice/IoU) to a single Excel (two locations) + overlays
"""

import os, json, random, warnings, time, csv
from pathlib import Path
from typing import Tuple, List, Optional, Sequence, Dict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as pe
import pandas as pd
import cv2

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset, DataLoader

from czifile import imread as czi_imread
from read_roi import read_roi_zip
from skimage.draw import polygon2mask
from skimage import morphology
from skimage.segmentation import find_boundaries, watershed
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.morphology import disk

# ---------- CLI ----------
import argparse
parser = argparse.ArgumentParser()

# Interactive inputs (if omitted, GUI pickers appear)
parser.add_argument("--images", type=str, default="", help="Comma-separated image paths (CZI/JPG/PNG/TIF). If empty => ask.")
parser.add_argument("--rois", type=str, default="", help="Comma-separated ROI .zip paths. If empty => ask.")
parser.add_argument("--out-dir", type=str, default="", help="Output directory. If empty => ask.")

# Experiment naming
parser.add_argument("--source-exp", type=str, default="unet_cv", help="Folder under ./experiments for checkpoints")
parser.add_argument("--exp-name", type=str, default="unet_eval_rgb", help="Destination experiment name (subfolder)")

# Pipeline modes
parser.add_argument("--modes", type=str, default="both", choices=["raw","n2v","both"], help="Pipelines to evaluate")
parser.add_argument("--no-tta", action="store_true", help="Disable 8x TTA")
parser.add_argument("--base", type=int, default=32, help="UNet base channels")
parser.add_argument("--n2v-cache", type=str, default="./n2v_cache", help="Folder with *_denoised_[RGB].npy")
parser.add_argument("--in-mode", type=str, default="g", choices=["g","rgb","lum"], help="Model input: g=green, rgb=3ch, lum=luminance")

# Train controls
parser.add_argument("--train", action="store_true", help="Enable LOIO training before evaluation")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--patch", type=int, default=256)
parser.add_argument("--samples-per-epoch", type=int, default=2000)
parser.add_argument("--val-every", type=int, default=2)
parser.add_argument("--bce-weight", type=float, default=0.5)
parser.add_argument("--pos-frac", type=float, default=0.5)
parser.add_argument("--aug-rot", action="store_true")
parser.add_argument("--seed", type=int, default=42)

# Supervisor-facing knobs
parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (validations without improvement)")
parser.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs before early stop")
parser.add_argument("--cosine", action="store_true", help="Use CosineAnnealingLR scheduler")

# Display/overlays
parser.add_argument("--show", dest="show", action="store_true", help="Show overlays while saving")
parser.add_argument("--no-show", dest="show", action="store_false", help="Save-only, do not show")
parser.set_defaults(show=True)
parser.add_argument("--gt-linew",   type=float, default=0.9, help="GT contour linewidth")
parser.add_argument("--pred-linew", type=float, default=0.9, help="Prediction contour linewidth")
parser.add_argument("--bar-um",  type=float, default=2.0, help="Scalebar length in µm (default 2)")
parser.add_argument("--bar-loc", type=str, default="lower right",
                    choices=["lower right","lower left","upper right","upper left"], help="Scalebar corner")
parser.add_argument("--bar-text", type=int, default=14, help="Scalebar text size")
parser.add_argument("--lum-low",  type=float, default=2.0,   help="Lower luminance percentile")
parser.add_argument("--lum-high", type=float, default=99.7,  help="Upper luminance percentile")
parser.add_argument("--gain",     type=float, default=1.15,  help="Global gain after stretch")
parser.add_argument("--gamma",    type=float, default=0.90,  help="Gamma (<1 brightens mids)")

args, _ = parser.parse_known_args()

# ---------- Constants ----------
SEED   = args.seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_ROOT   = Path("./experiments") / args.source_exp
N2V_CACHE  = Path(args.n2v_cache)

MICRONS_PER_PIXEL = 0.0322  # µm / px
PATCH_SIZE_EVAL   = 512
SLIDE_STRIDE      = 256
THR_GRID          = np.linspace(0.45, 0.65, 9)

# IMPORTANT: match full-pipeline post-processing
PROBA_SMOOTH_SIGMA     = 0.0   # no prob smoothing
REMOVE_SMALL_PRED_AREA = 0     # no pre-split small-object removal
OPENING_RADIUS         = 0     # no pre-split opening
WS_MIN_DISTANCE        = 10
FILTER_MIN_AREA        = 10
FILTER_MIN_SOLIDITY    = 0.30

# ---------- Utils ----------
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32); mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn + 1e-8)

# IO: CZI + standard images
def _read_any_image_float01(path: str) -> np.ndarray:
    """
    Return (C,Y,X) float[0,1].
    - .czi via czifile: squeeze to C,Y,X using Z=0,T=0
    - jpg/png/tif via cv2 -> RGB -> (C,Y,X)
    """
    ext = Path(path).suffix.lower()
    if ext == ".czi":
        arr = np.squeeze(czi_imread(path))
        if arr.ndim == 5: arr = arr[0]           # T,C,Z,Y,X -> C,Z,Y,X
        if arr.ndim == 4:
            if arr.shape[0] < 3 and arr.shape[1] >= 3:
                arr = np.moveaxis(arr, 0, 1)     # Z,C,Y,X -> C,Z,Y,X
            arr = arr[:, 0]                      # Z=0 => (C,Y,X)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected CZI shape {arr.shape} for {path}")
        return normalize01(arr.astype(np.float32))
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise ValueError(f"Cannot read image: {path}")
    if img.ndim == 2:
        g = normalize01(img)
        return np.stack([g,g,g], axis=0)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        if img.max() > 1.0: img /= 255.0
        return np.moveaxis(img, 2, 0)
    raise ValueError(f"Unsupported image ndim for {path}")

def czi_to_green01(filename: str) -> np.ndarray:
    cyx = _read_any_image_float01(filename)
    g_idx = 1 if cyx.shape[0] >= 2 else 0
    return normalize01(cyx[g_idx])

def get_rgb01_any(path: str) -> np.ndarray:
    cyx = _read_any_image_float01(path)
    c = cyx.shape[0]
    r = 0 if c>=1 else c-1
    g = 1 if c>=2 else r
    b = 2 if c>=3 else g
    return np.stack([normalize01(cyx[r]), normalize01(cyx[g]), normalize01(cyx[b])], axis=0)  # (3,H,W)

def roi_zip_to_binary_mask(roi_zip: str, shape: Tuple[int,int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    rois = read_roi_zip(roi_zip)
    for r in rois.values():
        if 'x' not in r or 'y' not in r or len(r['x']) < 3: continue
        y, x = r['y'], r['x']
        poly = polygon2mask(shape, np.column_stack([y, x]))
        mask[poly] = 1
    return mask

def _try_load(path: Path) -> Optional[np.ndarray]:
    try:
        if path.exists(): return np.load(path).astype(np.float32)
    except Exception: pass
    return None

# Model input by pipeline mode + in-mode
def get_input_by_mode(img_path: str, pipe_mode: str, in_mode: str) -> np.ndarray:
    if in_mode == "g":
        g = czi_to_green01(img_path)  # (H,W)
        if pipe_mode == "raw":
            return g[None,...]
        # n2v
        base = Path(img_path).name
        for p in [N2V_CACHE/f"{base}_denoised_G.npy", Path(img_path).with_suffix("").with_name(base+"_denoised_G.npy")]:
            arr = _try_load(p)
            if arr is not None: return normalize01(arr)[None,...]
        print(f"[n2v] WARNING missing denoised G for {base}; fallback RAW.")
        return g[None,...]

    # RGB or luminance
    def _rgb_from(mode):
        stem = Path(img_path).name
        if mode == "n2v":
            cand = [
                (N2V_CACHE/f"{stem}_denoised_R.npy", N2V_CACHE/f"{stem}_denoised_G.npy", N2V_CACHE/f"{stem}_denoised_B.npy"),
                (Path(img_path).with_suffix("").with_name(stem+"_denoised_R.npy"),
                 Path(img_path).with_suffix("").with_name(stem+"_denoised_G.npy"),
                 Path(img_path).with_suffix("").with_name(stem+"_denoised_B.npy")),
            ]
            for r,g,b in cand:
                R=_try_load(r); G=_try_load(g); B=_try_load(b)
                if R is not None and G is not None and B is not None:
                    return np.stack([normalize01(R), normalize01(G), normalize01(B)], axis=0)
            print(f"[n2v] WARNING missing RGB caches for {stem}; fallback RAW.")
        return get_rgb01_any(img_path)

    RGB = _rgb_from(pipe_mode)  # (3,H,W)
    if in_mode == "rgb":
        return RGB
    # luminance
    Y = (0.2126*RGB[0] + 0.7152*RGB[1] + 0.0722*RGB[2]).astype(np.float32)
    return Y[None,...]

# Display enhancer (visual only)
def enhance_rgb_for_display(rgb01_hwc: np.ndarray, low_pct=2.0, high_pct=99.7, gain=1.15, gamma=0.90):
    x = rgb01_hwc.clip(0,1).astype(np.float32)
    Y = 0.2126*x[...,0]+0.7152*x[...,1]+0.0722*x[...,2]
    lo,hi = np.percentile(Y,low_pct), np.percentile(Y,high_pct)
    if hi<=lo: lo,hi = float(Y.min()), float(max(Y.max(), Y.min()+1e-6))
    x = (x-lo)/(hi-lo+1e-8); x = np.clip(x,0,1)
    if gain!=1.0: x = np.clip(x*gain,0,1)
    if gamma!=1.0: x = np.power(x,gamma)
    return x

def get_rgb_display(img_path: str, pipe_mode: str, lum_low=2.0, lum_high=99.7, gain=1.15, gamma=0.90) -> np.ndarray:
    # try n2v RGB first
    stem = Path(img_path).name
    for r,g,b in [
        (N2V_CACHE/f"{stem}_denoised_R.npy", N2V_CACHE/f"{stem}_denoised_G.npy", N2V_CACHE/f"{stem}_denoised_B.npy"),
        (Path(img_path).with_suffix("").with_name(stem+"_denoised_R.npy"),
         Path(img_path).with_suffix("").with_name(stem+"_denoised_G.npy"),
         Path(img_path).with_suffix("").with_name(stem+"_denoised_B.npy")),
    ]:
        R=_try_load(r); G=_try_load(g); B=_try_load(b)
        if pipe_mode=="n2v" and R is not None and G is not None and B is not None:
            rgb = np.stack([normalize01(R), normalize01(G), normalize01(B)], axis=-1)
            return enhance_rgb_for_display(rgb, lum_low, lum_high, gain, gamma)
    rgb = np.moveaxis(get_rgb01_any(img_path), 0, -1)
    return enhance_rgb_for_display(rgb, lum_low, lum_high, gain, gamma)

# ---------- Pairing images ↔ ROI ZIPs ----------
def _tokens(s: str) -> set:
    base = Path(s).stem.lower().replace("_"," ").replace("-"," ")
    return set([t for t in base.split() if t])

def pair_images_to_rois(image_paths: List[str], roi_paths: List[str]) -> List[Tuple[str,str]]:
    """One-to-one pairing by filename token Jaccard similarity. Keeps original image order for folds."""
    from itertools import product
    I = list(map(str, image_paths)); R = list(map(str, roi_paths))
    ti = [(_tokens(p), p) for p in I]
    tr = [(_tokens(p), p) for p in R]
    scores = []
    for (ti_tokens, ip), (tr_tokens, rp) in product(ti, tr):
        inter = len(ti_tokens & tr_tokens)
        union = len(ti_tokens | tr_tokens) or 1
        scores.append((-(inter/union), ip, rp))  # negative for ascending sort -> max first
    scores.sort()
    used_i, used_r = set(), set()
    pairs = []
    for _, ip, rp in scores:
        if ip in used_i or rp in used_r: continue
        pairs.append((ip, rp))
        used_i.add(ip); used_r.add(rp)
        if len(pairs) == min(len(I), len(R)): break
    if len(pairs) < len(I) or len(pairs) < len(R):
        print(f"[pair] WARNING: unmatched items (images={len(I)}, rois={len(R)}, pairs={len(pairs)})")
    ip2rp = {ip: rp for ip, rp in pairs}
    ordered = [(ip, ip2rp.get(ip, R[0])) for ip in I]  # fallback to first ROI if missing
    return ordered

# ---------- Model ----------
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

# ---------- Inference ----------
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
def sliding_proba_seamless(model, image_chw: np.ndarray, patch=PATCH_SIZE_EVAL, stride=SLIDE_STRIDE) -> np.ndarray:
    # image_chw: (C,H,W)
    C,H,W = image_chw.shape
    acc  = np.zeros((H,W), np.float32); wsum = np.zeros((H,W), np.float32)
    win_full = _tukey2d(patch, patch, alpha=0.6, eps=0.10)
    use_amp = (DEVICE.type == "cuda")
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0 = min(y, H-patch); x0 = min(x, W-patch)
            tile = image_chw[:, y0:y0+patch, x0:x0+patch]  # (C,h0,w0)
            h0,w0 = tile.shape[-2:]
            if h0<patch or w0<patch:
                tile = np.pad(tile, ((0,0),(0,patch-h0),(0,patch-w0)), mode='reflect')
            t = torch.from_numpy(tile[None].copy()).float().to(DEVICE)
            with amp.autocast('cuda', enabled=use_amp):
                p = torch.sigmoid(model(t))[0,0].detach().cpu().numpy()
            p = p[:h0,:w0]; win = win_full[:h0,:w0]
            acc [y0:y0+h0, x0:x0+w0] += p*win
            wsum[y0:y0+h0, x0:x0+w0] += win
    return acc/(wsum+1e-8)

@torch.no_grad()
def tta_proba_8x(model, image_chw: np.ndarray) -> np.ndarray:
    rots=[0,1,2,3]; flips=[False,True]
    ps=[]
    for r in rots:
        img_r = np.rot90(image_chw, k=r, axes=(-2,-1))
        for f in flips:
            img_rf = img_r[..., ::-1] if f else img_r
            prob = sliding_proba_seamless(model, img_rf)
            prob = prob[:, ::-1] if f else prob
            ps.append(np.rot90(prob, k=4-r))
    return np.mean(ps, axis=0)

# ---------- Metrics (overlap-only) ----------
def dice_iou_overlap_only(pred_bin: np.ndarray, gt_bin: np.ndarray):
    pred_eff = np.logical_and(pred_bin>0, gt_bin>0)
    gt_bool  = (gt_bin>0)
    inter = np.logical_and(pred_eff, gt_bool).sum()
    pred_sum = pred_eff.sum(); gt_sum = gt_bool.sum()
    dice = 2.0*inter/(pred_sum+gt_sum+1e-8)
    iou  = inter/(gt_sum+1e-8)
    return float(dice), float(iou)

def pick_best_threshold_overlap(prob: np.ndarray, gt: np.ndarray, grid=THR_GRID):
    gt_bool = (gt>0); best_t, best_d = grid[0], -1.0
    for t in grid:
        m = (prob>=t).astype(np.uint8)
        m = morphology.remove_small_objects(m.astype(bool), 1).astype(np.uint8)
        d,_ = dice_iou_overlap_only(m, gt_bool.astype(np.uint8))
        if d>best_d: best_d, best_t = d, t
    return best_t, best_d

# ---------- Postprocess (MATCHES full pipeline) ----------
def split_touching_cells(mask, min_distance=WS_MIN_DISTANCE):
    proc = mask.astype(bool)
    if proc.sum()==0: return np.zeros_like(mask, dtype=np.int32)
    # DT + light smoothing (as in full pipeline)
    dist = gaussian(ndi.distance_transform_edt(proc), sigma=0.7)
    # seed peaks
    coords = peak_local_max(dist, min_distance=min_distance, labels=proc)
    peaks = np.zeros_like(dist, dtype=bool)
    if coords.size: peaks[tuple(coords.T)] = True
    else: peaks[np.unravel_index(np.argmax(dist), dist.shape)] = True
    markers,_ = ndi.label(peaks)
    # watershed
    labels = watershed(-dist, markers, mask=proc)
    # remove tiny labels, then region filtering
    labels = morphology.remove_small_objects(labels, min_size=FILTER_MIN_AREA)
    from skimage.measure import regionprops
    out = np.zeros_like(labels, dtype=np.int32); k=1
    for r in regionprops(labels):
        if r.area>=FILTER_MIN_AREA and r.solidity>=FILTER_MIN_SOLIDITY:
            out[labels==r.label] = k; k+=1
    return out

# ---------- Checkpoints ----------
def try_load_state(model: nn.Module, path: Path) -> bool:
    if not path.exists():
        print(f"[ckpt] missing: {path}"); return False
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

# ---------- Scalebar + overlay ----------
def _style_asb_label(asb, color='white', fontsize=12):
    for attr in ('txt_label','label_txt'):
        obj = getattr(asb, attr, None)
        if obj is None: continue
        try:
            obj.set_color(color); obj.set_fontsize(fontsize)
            obj.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])
            return
        except Exception: pass

def add_scale_bar_um(ax, image_shape, microns_per_pixel, bar_um, loc, text_size):
    H,W = image_shape[:2]
    bar_px = (bar_um if bar_um and bar_um>0 else 2.0)/microns_per_pixel
    loc_map = {'upper right':1,'upper left':2,'lower left':3,'lower right':4}
    asb = AnchoredSizeBar(ax.transData, size=bar_px, label=f'{bar_um:g} µm',
                          loc=loc_map.get(loc,4), pad=0.4, sep=4, borderpad=0.8,
                          frameon=False, size_vertical=max(2,int(0.006*H)), color='white')
    _style_asb_label(asb, 'white', text_size)
    try:
        asb.size_bar.set_edgecolor('black'); asb.size_bar.set_linewidth(max(1,int(0.002*H)))
    except Exception: pass
    ax.add_artist(asb)

def save_panel_with_contours(base_rgb01: np.ndarray, title: str, out_path: Path,
                             gt_bound: np.ndarray, pred_bound: np.ndarray,
                             gt_linew: float, pred_linew: float,
                             microns_per_pixel: float, bar_um: float, bar_loc: str, bar_text_size: int,
                             show: bool):
    H,W,_ = base_rgb01.shape
    fig, ax = plt.subplots(1,1, figsize=(10,10*H/W), facecolor='black')
    ax.imshow((base_rgb01*255).astype(np.uint8), interpolation='nearest')
    ax.set_title(title, color='white', fontsize=18, pad=14); ax.axis('off')
    if gt_bound is not None and gt_bound.any():
        ax.contour(gt_bound.astype(float), levels=[0.5], colors='deepskyblue', linewidths=gt_linew)
    if pred_bound is not None and pred_bound.any():
        ax.contour(pred_bound.astype(float), levels=[0.5], colors='yellow', linewidths=pred_linew)
    add_scale_bar_um(ax, (H,W), microns_per_pixel, bar_um, bar_loc, bar_text_size)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, facecolor='black', bbox_inches='tight', pad_inches=0.02)
    if show: plt.show()
    plt.close(fig)

# ---------- Training ----------
class PatchDataset(Dataset):
    def __init__(self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray],
                 patch: int=256, pos_frac: float=0.5, aug_rot90: bool=False):
        self.images = [img.astype(np.float32) for img in images]  # (C,H,W)
        self.masks  = [msk.astype(np.uint8)   for msk in masks]   # (H,W)
        self.patch = patch; self.pos_frac = np.clip(pos_frac,0,1); self.aug_rot90 = aug_rot90
        self.fg_coords: List[np.ndarray] = []
        for m in self.masks:
            ys,xs = np.where(m>0)
            self.fg_coords.append(np.stack([ys,xs],1) if ys.size else np.zeros((0,2),int))
    def __len__(self): return 10_000_000
    def __getitem__(self, idx):
        k = random.randrange(len(self.images))
        img = self.images[k]; msk = self.masks[k]
        C,H,W = img.shape
        if random.random() < self.pos_frac and self.fg_coords[k].shape[0] > 0:
            y,x = self.fg_coords[k][random.randrange(self.fg_coords[k].shape[0])]
        else:
            y = random.randrange(H); x = random.randrange(W)
        p = self.patch
        y0 = np.clip(y - p//2, 0, H - p); x0 = np.clip(x - p//2, 0, W - p)
        crop_img = img[:, y0:y0+p, x0:x0+p]               # (C,p,p)
        crop_msk = msk[y0:y0+p, x0:x0+p]                  # (p,p)
        if crop_img.shape[-2:] != (p,p):
            pad_h = p - crop_img.shape[-2]; pad_w = p - crop_img.shape[-1]
            crop_img = np.pad(crop_img, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
            crop_msk = np.pad(crop_msk, ((0,pad_h),(0,pad_w)), mode='constant')
        if random.random()<0.5:
            crop_img = crop_img[..., ::-1]; crop_msk = crop_msk[:, ::-1]
        if random.random()<0.5:
            crop_img = crop_img[..., ::-1, :]; crop_msk = crop_msk[::-1, :]
        if self.aug_rot90 and random.random()<0.5:
            krot = random.choice([1,2,3])
            crop_img = np.rot90(crop_img, krot, axes=(-2,-1)); crop_msk = np.rot90(crop_msk, krot)
        x_t = torch.from_numpy(crop_img.copy()).float()
        y_t = torch.from_numpy((crop_msk>0).astype(np.float32)[None].copy())
        return x_t, y_t

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float=0.5, smooth: float=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(); self.w = float(np.clip(bce_weight,0,1)); self.smooth = smooth
    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        inter = (probs*target).sum(dim=(2,3))
        den   = (probs+target).sum(dim=(2,3))
        dice = (2.0*inter + self.smooth)/(den + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.w*bce + (1.0-self.w)*dice_loss

@torch.no_grad()
def eval_on_val_image(model, val_img_chw: np.ndarray, val_msk: np.ndarray, use_tta=True):
    prob = tta_proba_8x(model, val_img_chw) if use_tta else sliding_proba_seamless(model, val_img_chw)
    # keep PROBA_SMOOTH_SIGMA=0.0 to match full pipeline
    if PROBA_SMOOTH_SIGMA>0:  # currently disabled
        prob = gaussian(prob, sigma=PROBA_SMOOTH_SIGMA, preserve_range=True)
    thr, dice_val = pick_best_threshold_overlap(prob, (val_msk>0).astype(np.uint8), THR_GRID)
    return float(dice_val), float(thr)

def train_fold_loio(model: nn.Module,
                    train_imgs: Sequence[np.ndarray],
                    train_msks: Sequence[np.ndarray],
                    val_img: np.ndarray, val_msk: np.ndarray,
                    out_fold_dir: Path,
                    lr: float, epochs: int, batch: int, patch: int,
                    samples_per_epoch: int, val_every: int,
                    bce_weight: float, pos_frac: float, aug_rot90: bool,
                    use_tta_for_val: bool):
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    (out_fold_dir / "train_config.json").write_text(json.dumps({
        "lr": lr, "epochs": epochs, "batch": batch, "patch": patch, "samples_per_epoch": samples_per_epoch,
        "val_every": val_every, "bce_weight": bce_weight, "pos_frac": pos_frac, "aug_rot90": aug_rot90,
        "tta_val": use_tta_for_val
    }, indent=2))

    # Logs
    log_txt = out_fold_dir / "train_log.txt"
    log_csv = out_fold_dir / "train_log.csv"
    curves_png = out_fold_dir / "learning_curves.png"
    if not log_csv.exists():
        log_csv.write_text("epoch,train_loss,val_dice,time_s\n")

    ds = PatchDataset(train_imgs, train_msks, patch=patch, pos_frac=pos_frac, aug_rot90=aug_rot90)
    steps_per_epoch = max(1, samples_per_epoch // batch)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = BCEDiceLoss(bce_weight=bce_weight)
    scaler = amp.GradScaler('cuda') if DEVICE.type=="cuda" else None
    scheduler = None
    if args.cosine and epochs>=5:
        t_max = max(1, int(epochs*0.8))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=lr*0.05)

    best_val = -1.0; no_improve = 0
    best_path = out_fold_dir / "best.pt"; last_path = out_fold_dir / "last.pt"

    model.to(DEVICE); model.train()
    it = iter(loader)

    for epoch in range(1, epochs+1):
        model.train(); epoch_loss=0.0; t0=time.time()
        for _ in range(steps_per_epoch):
            try: x,y = next(it)
            except StopIteration: it=iter(loader); x,y=next(it)
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with amp.autocast('cuda'):
                    logits = model(x); loss = loss_fn(logits, y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                logits = model(x); loss = loss_fn(logits, y)
                loss.backward(); opt.step()
            epoch_loss += float(loss.detach().cpu().item())
        dur = time.time()-t0
        torch.save(model.state_dict(), last_path)

        cur_val = None
        if (epoch % val_every == 0) or (epoch == epochs):
            model.eval()
            cur_val,_ = eval_on_val_image(model, val_img, val_msk, use_tta=use_tta_for_val)
            if cur_val > best_val:
                best_val = cur_val; torch.save(model.state_dict(), best_path); no_improve = 0
            else:
                no_improve += 1
            log_txt.open("a").write(
                f"epoch {epoch:03d} | loss {epoch_loss/steps_per_epoch:.4f} | valDice(overlap-only) {cur_val:.4f} | best {best_val:.4f} | {dur:.1f}s\n"
            )
            log_csv.open("a").write(f"{epoch},{epoch_loss/steps_per_epoch:.6f},{cur_val:.6f},{dur:.3f}\n")
            if epoch >= args.min_epochs and no_improve >= args.patience:
                print(f"[early-stop] no improvement for {no_improve} validations. Best valDice={best_val:.4f}")
                break
        if scheduler is not None: scheduler.step()

    # Plot curves
    try:
        df = pd.read_csv(log_csv)
        if not df.empty:
            fig = plt.figure(figsize=(7,4.5))
            ax1 = plt.gca(); ax1.plot(df["epoch"], df["train_loss"], label="train_loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
            ax2 = ax1.twinx(); ax2.plot(df["epoch"], df["val_dice"], label="valDice (overlap-only)"); ax2.set_ylabel("valDice")
            ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
            fig.tight_layout(); fig.savefig(curves_png, dpi=160); plt.close(fig)
    except Exception as e:
        print(f"[plot] skip: {e}")

    return best_path if best_path.exists() else last_path

# ---------- Evaluation ----------
@torch.no_grad()
def eval_fold(model: nn.Module, fold_dir: Path,
              val_img, val_msk, test_img, test_msk,
              img_test_path: str, mode_label: str,
              gt_linew: float, pred_linew: float,
              tta=True, show=True):
    # threshold chosen on validation (overlap-only)
    val_prob = tta_proba_8x(model, val_img) if tta else sliding_proba_seamless(model, val_img)
    if PROBA_SMOOTH_SIGMA>0:  # disabled to match full pipeline
        val_prob = gaussian(val_prob, sigma=PROBA_SMOOTH_SIGMA, preserve_range=True)
    thr,_ = pick_best_threshold_overlap(val_prob, (val_msk>0).astype(np.uint8), THR_GRID)
    (fold_dir/"val_threshold.txt").write_text(f"thr={thr:.4f}  (overlap-only)\n")

    # test prob → threshold (no pre-split cleaning)
    test_prob = tta_proba_8x(model, test_img) if tta else sliding_proba_seamless(model, test_img)
    if PROBA_SMOOTH_SIGMA>0:  # disabled
        test_prob = gaussian(test_prob, sigma=PROBA_SMOOTH_SIGMA, preserve_range=True)
    pred_bin = (test_prob>=thr).astype(np.uint8)

    # watershed split + post label filtering (same as full pipeline)
    pred_lbl   = split_touching_cells(pred_bin, min_distance=WS_MIN_DISTANCE)
    pred_binPP = (pred_lbl>0).astype(np.uint8)
    gt_bin     = (test_msk>0).astype(np.uint8)

    # metrics (overlap-only)
    dice_ol, iou_ol = dice_iou_overlap_only(pred_binPP, gt_bin)

    # overlay
    gt_bound   = find_boundaries(gt_bin.astype(bool),   mode='outer')
    pred_bound = find_boundaries(pred_binPP.astype(bool), mode='outer')
    base_rgb = get_rgb_display(img_test_path, mode_label, lum_low=args.lum_low, lum_high=args.lum_high, gain=args.gain, gamma=args.gamma)
    title = f"{Path(img_test_path).name} — {mode_label.upper()} (Dice {dice_ol:.3f}, IoU {iou_ol:.3f}, thr={thr:.3f} | overlap-only)"
    save_panel_with_contours(base_rgb, title, fold_dir/"test_overlay.png",
                             gt_bound, pred_bound, gt_linew, pred_linew,
                             MICRONS_PER_PIXEL, args.bar_um, args.bar_loc, args.bar_text,
                             show=show)

    with open(fold_dir/"test_metrics.json","w") as f:
        json.dump({"threshold":float(thr),
                   "dice_overlap_only":float(dice_ol),
                   "iou_overlap_only":float(iou_ol),
                   "note":"Overlap-only compares (pred ∧ GT) vs GT."}, f, indent=2)
    return dice_ol, iou_ol, float(thr)

def read_source_base(mode_root: Path, fallback_base: int) -> int:
    cfg = mode_root/"run_config.json"
    if cfg.exists():
        try:
            j=json.loads(cfg.read_text()); return int(j.get("base", fallback_base))
        except Exception: pass
    return fallback_base

def load_images_masks_for_mode(pairs: List[Tuple[str,str]], pipe_mode: str, in_mode: str):
    imgs, msks, img_paths = [], [], []
    for img_path, roi_zip in pairs:
        x = get_input_by_mode(img_path, pipe_mode, in_mode)  # (C,H,W)
        m = roi_zip_to_binary_mask(roi_zip, x.shape[1:])
        imgs.append(x); msks.append(m); img_paths.append(img_path)
    return imgs, msks, img_paths

def run_eval_for_mode(mode: str, folds: List[int], pairs: List[Tuple[str,str]],
                      dest_root: Path, base: int, in_ch: int, tta: bool, show: bool):
    src_mode_root  = SRC_ROOT / mode
    dest_mode_root = dest_root / mode
    dest_mode_root.mkdir(parents=True, exist_ok=True)

    (dest_mode_root/"run_config.json").write_text(json.dumps({
        "mode": mode, "base": base, "tta_8x": tta, "source_exp": str(SRC_ROOT.name),
        "evaluation": "overlap-only (prediction clipped to GT) for Dice/IoU.",
        "postprocess_like_full_pipeline": True
    }, indent=2))

    IMGS, MSKS, IMG_PATHS = load_images_masks_for_mode(pairs, mode, args.in_mode)
    N = len(IMGS)
    rows: List[List[object]] = []

    for test_idx in folds:
        val_idx = (test_idx + 1) % N
        print(f"\n== {mode.upper()} | FOLD {test_idx+1}/{N} ==")

        fold_ckpt_dir = src_mode_root / f"fold_{test_idx+1}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model = UNet(in_ch,1,base).to(DEVICE)
        ckpt_path = fold_ckpt_dir/"best.pt"
        if not try_load_state(model, ckpt_path):
            print(f"[auto-train] fold {test_idx+1} checkpoint missing -> training this fold now...")
            train_ids = [k for k in range(N) if k not in (test_idx, val_idx)]
            best_ckpt = train_fold_loio(
                model=UNet(in_ch,1,base).to(DEVICE),
                train_imgs=[IMGS[k] for k in train_ids],
                train_msks=[MSKS[k] for k in train_ids],
                val_img=IMGS[val_idx], val_msk=MSKS[val_idx],
                out_fold_dir=fold_ckpt_dir,
                lr=args.lr, epochs=args.epochs, batch=args.batch, patch=args.patch,
                samples_per_epoch=args.samples_per_epoch, val_every=args.val_every,
                bce_weight=args.bce_weight, pos_frac=args.pos_frac,
                aug_rot90=args.aug_rot, use_tta_for_val=True
            )
            ckpt_path = Path(best_ckpt)
            model = UNet(in_ch,1,base).to(DEVICE)
            if not try_load_state(model, ckpt_path):
                print(f"[skip] fold {test_idx+1} still missing/incompatible after training.")
                continue

        fold_dir = dest_mode_root / f"fold_{test_idx+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir/"used_checkpoint.txt").write_text(str(ckpt_path))

        dice_ol, iou_ol, thr = eval_fold(
            model, fold_dir,
            val_img=IMGS[val_idx],   val_msk=MSKS[val_idx],
            test_img=IMGS[test_idx], test_msk=MSKS[test_idx],
            img_test_path=IMG_PATHS[test_idx], mode_label=mode,
            gt_linew=args.gt_linew, pred_linew=args.pred_linew, tta=tta, show=show
        )
        rows.append([test_idx+1, os.path.basename(IMG_PATHS[test_idx]), dice_ol, iou_ol, thr])
    return rows

# ---------- Excel + print ----------
def build_needed_rows(raw_rows, n2v_rows):
    out=[]
    for r in raw_rows: out.append({"Image":r[1], "Mode":"RAW", "Dice":float(r[2]), "IoU":float(r[3])})
    for r in n2v_rows: out.append({"Image":r[1], "Mode":"DENOISED", "Dice":float(r[2]), "IoU":float(r[3])})
    return out

# === NEW: results plots ===
def _annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.3f}", (p.get_x()+p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9, rotation=0, xytext=(0,3), textcoords="offset points")

def plot_results(df: pd.DataFrame, dest_root: Path, show: bool):
    if df.empty: return
    plots_dir = dest_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Per-image Dice bars (RAW vs DENOISED) with IoU text
    for img in sorted(df['Image'].unique()):
        sub = df[df['Image']==img].copy()
        sub = sub.sort_values("Mode")
        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()
        ax.bar(sub["Mode"], sub["Dice"])
        _annotate_bars(ax)
        # Add IoU as text underneath bars for clarity
        for i,(mode, iou) in enumerate(zip(sub["Mode"], sub["IoU"])):
            ax.text(i, max(sub["Dice"])*0.02, f"IoU {iou:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.05*max(0.8, sub["Dice"].max()))
        ax.set_ylabel("Dice (overlap-only)")
        ax.set_title(f"{img} — RAW vs DENOISED")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{Path(img).stem}_dice_bars.png", dpi=180)
        if show: plt.show()
        plt.close(fig)

    # Mean Dice / IoU by mode
    mean_df = df.groupby("Mode", as_index=False).agg({"Dice":"mean", "IoU":"mean"})
    for metric in ("Dice","IoU"):
        fig = plt.figure(figsize=(5,4))
        ax = plt.gca()
        ax.bar(mean_df["Mode"], mean_df[metric])
        _annotate_bars(ax)
        ax.set_ylim(0, 1.05*mean_df[metric].max())
        ax.set_ylabel(f"Mean {metric} (overlap-only)")
        ax.set_title(f"Mean {metric} by Mode")
        fig.tight_layout()
        fig.savefig(plots_dir / f"mean_{metric.lower()}_by_mode.png", dpi=180)
        if show: plt.show()
        plt.close(fig)
# === END NEW: results plots ===

def save_paper_metrics_excel_and_print(raw_rows, n2v_rows, dest_root: Path):
    df = pd.DataFrame(build_needed_rows(raw_rows, n2v_rows))
    path_a = dest_root / "metrics.xlsx"
    path_b = dest_root / "metric_evaluation_results" / "metrics.xlsx"
    path_b.parent.mkdir(parents=True, exist_ok=True)

    def _write_xlsx(path: Path):
        for engine in ("openpyxl", "xlsxwriter"):
            try:
                with pd.ExcelWriter(path, engine=engine) as w:
                    df.to_excel(w, index=False, sheet_name="per_image")
                return True
            except Exception: continue
        print(f"[excel] NOTE: no Excel engine; skipped: {path}")
        return False

    ok_a=_write_xlsx(path_a); ok_b=_write_xlsx(path_b)

    print("\n=== Per-image Dice / IoU (RAW vs DENOISED) ===")
    if df.empty: print("No rows to report.")
    else:
        for img in sorted(df['Image'].unique()):
            sub = df[df['Image']==img]
            for _,row in sub.iterrows():
                print(f"{img:35s} | {row['Mode']:9s} | Dice={row['Dice']:.4f}  IoU={row['IoU']:.4f}")
        if ok_a: print(f"Saved Excel: {path_a}")
        if ok_b: print(f"Saved Excel: {path_b}")
    print("")

    # === NEW: kick off plots ===
    try:
        plot_results(df, dest_root, show=args.show)
    except Exception as e:
        print(f"[plots] Skipped making results plots due to: {e}")

# ---------- Interactive pickers ----------
def ask_user_files_if_needed() -> Tuple[List[str], List[str], Path]:
    imgs = [p for p in map(str.strip, args.images.split(",")) if p] if args.images else []
    rois = [p for p in map(str.strip, args.rois.split(",")) if p] if args.rois else []
    out_dir = Path(args.out_dir) if args.out_dir else None

    if not (imgs and rois and out_dir):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            if not imgs:
                imgs = list(filedialog.askopenfilenames(
                    title="Select input images (CZI/JPG/PNG/TIF)",
                    filetypes=[("Microscopy/Images","*.czi *.tif *.tiff *.png *.jpg *.jpeg"), ("All","*.*")]
                ))
            if not rois:
                rois = list(filedialog.askopenfilenames(
                    title="Select Ground Truth ROI .zip files",
                    filetypes=[("ImageJ ROI Zip","*.zip"), ("All","*.*")]
                ))
            if out_dir is None or not str(out_dir):
                od = filedialog.askdirectory(title="Select OUTPUT directory")
                out_dir = Path(od) if od else Path("./results")
        except Exception as e:
            print(f"[ui] picker fallback due to: {e}")
            if not imgs:
                raise RuntimeError("No images provided and GUI picker failed. Use --images")
            if not rois:
                raise RuntimeError("No ROI zips provided and GUI picker failed. Use --rois")
            if out_dir is None:
                out_dir = Path("./results")
    if not imgs: raise RuntimeError("No image files selected.")
    if not rois: raise RuntimeError("No ROI zip files selected.")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    return imgs, rois, out_dir

# ---------- MAIN ----------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    seed_all(SEED)

    # Get files/folders from user if needed
    image_paths, roi_paths, OUT_DIR = ask_user_files_if_needed()
    pairs = pair_images_to_rois(image_paths, roi_paths)  # [(img,roi), …]

    # Experiment destination rooted at chosen output folder
    DEST_ROOT = OUT_DIR / args.exp_name
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # Persist experiment metadata
    (DEST_ROOT / "experiment.json").write_text(json.dumps({
        "source_exp": str(SRC_ROOT.name),
        "dest_exp": str(DEST_ROOT.name),
        "modes": (["raw","n2v"] if args.modes=="both" else [args.modes]),
        "tta_8x": not args.no_tta,
        "in_mode": args.in_mode,
        "microns_per_pixel": MICRONS_PER_PIXEL,
        "thr_grid": list(map(float, THR_GRID)),
        "postprocess": {
            "match_full_pipeline": True,
            "proba_gaussian_sigma": PROBA_SMOOTH_SIGMA,
            "remove_small_area_pre_split": REMOVE_SMALL_PRED_AREA,
            "binary_opening_radius": OPENING_RADIUS,
            "min_solidity": FILTER_MIN_SOLIDITY,
            "min_area": FILTER_MIN_AREA,
            "ws_min_distance": WS_MIN_DISTANCE
        },
        "display": {"lum_low": args.lum_low, "lum_high": args.lum_high, "gain": args.gain, "gamma": args.gamma},
        "scalebar": {"um": args.bar_um, "loc": args.bar_loc, "text": args.bar_text},
        "evaluation": "Dice/IoU computed only on prediction overlapped with GT (clipped-to-GT)."
    }, indent=2))

    # Folds (all images participate)
    N = len(pairs)
    folds = list(range(N))

    # In-channels by input mode
    in_ch = 3 if args.in_mode=="rgb" else 1

    # TRAIN (optional)
    if args.train:
        print("\n=== TRAINING FROM SCRATCH (LOIO) ===")
        for mode in (["raw","n2v"] if args.modes=="both" else [args.modes]):
            src_mode_root = SRC_ROOT / mode
            src_mode_root.mkdir(parents=True, exist_ok=True)
            (src_mode_root/"run_config.json").write_text(json.dumps({
                "mode": mode, "base": args.base, "tta_8x": True,
                "note": "Training hyperparams per-fold in train_config.json."
            }, indent=2))
            IMGS, MSKS, _ = load_images_masks_for_mode(pairs, mode, args.in_mode)
            for test_id in folds:
                val_id = (test_id+1)%N
                train_ids = [k for k in range(N) if k not in (test_id, val_id)]
                fold_dir = src_mode_root / f"fold_{test_id+1}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                best_ckpt = train_fold_loio(
                    model=UNet(in_ch,1,args.base).to(DEVICE),
                    train_imgs=[IMGS[k] for k in train_ids],
                    train_msks=[MSKS[k] for k in train_ids],
                    val_img=IMGS[val_id], val_msk=MSKS[val_id],
                    out_fold_dir=fold_dir,
                    lr=args.lr, epochs=args.epochs, batch=args.batch, patch=args.patch,
                    samples_per_epoch=args.samples_per_epoch, val_every=args.val_every,
                    bce_weight=args.bce_weight, pos_frac=args.pos_frac,
                    aug_rot90=args.aug_rot, use_tta_for_val=True
                )
                print(f"[{mode}] fold {test_id+1} best: {best_ckpt}")

    # EVAL
    MODES_TO_RUN = (["raw","n2v"] if args.modes=="both" else [args.modes])
    tta = not args.no_tta

    raw_rows: List[List[object]] = []
    n2v_rows: List[List[object]] = []

    base = args.base
    for mode in MODES_TO_RUN:
        src_mode_root = SRC_ROOT / mode
        base = read_source_base(src_mode_root, args.base)
        rows = run_eval_for_mode(mode, folds, pairs, DEST_ROOT, base, in_ch, tta, show=args.show)
        if mode=="raw": raw_rows = rows
        if mode=="n2v": n2v_rows = rows

    # Save Excel + console print + NEW plots
    save_paper_metrics_excel_and_print(raw_rows, n2v_rows, DEST_ROOT)

if __name__ == "__main__":
    main()
