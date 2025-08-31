#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full pipeline:
- N2V per-channel denoising (trains only if cache is missing)
- Cellpose segmentation (cyto, cyto2) on RAW & N2V
- Post-processing + evaluation (Dice/IoU restricted to GT support)
- VIS: show & save each case individually (RAW/Denoised × cyto/cyto2) with a professional µm scale bar

Outputs:
- <image>_denoised_R.npy/_G.npy/_B.npy
- psnr_ssim_3ch.csv
- segmentation_metrics_overlay_3ch.csv
- overlays_cases/<stem>__<Original|Denoised>__<cyto|cyto2>.png
"""

import os, sys, platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as pe

from czifile import imread
from read_roi import read_roi_zip
from skimage.draw import polygon2mask
from skimage.util import img_as_ubyte
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed
from skimage.measure import regionprops, label as sk_label
from skimage.filters import gaussian
from skimage import exposure
from skimage.feature import peak_local_max
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy import ndimage as ndi
import pandas as pd
from cellpose import models

# Optional libs
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from n2v.models import N2V, N2VConfig
    N2V_AVAIL = True
except Exception:
    N2V_AVAIL = False

# ============== Your files ==============
image_files = [
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI3_SIM.czi",
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI2_SIM.czi",
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI1_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI2_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI3_SIM.czi",
]
roi_files = [
    "RoiSet_2D_WT_NADA_THY3.zip",
    "RoiSet_2D_WT_NADA_THY2.zip",
    "RoiSet_2D_WT_NADA_THY1.zip",
    "RoiSet_Contour_bacteria_ROI1.zip",
    "RoiSet_Contour_bacteria_ROI2.zip",
    "RoiSet_Contour_bacteria_ROI3.zip",
]

# ============== Config ==============
MICRONS_PER_PIXEL = 0.0322      # µm per pixel (your imaging scale)
BAR_MICRONS = 2                 # scalebar length in µm (set None to auto-pick)
SCALEBAR_LOC = "lower right"    # 'lower right'|'lower left'|'upper right'|'upper left'

MODEL_TYPES = ['cyto', 'cyto2']
BASE_MODEL_DIR = './models'     # where N2V logs go

# N2V
PATCH_SIZE = (64, 64)
N_PATCHES = 512
VAL_SPLIT = 0.10
TRAIN_EPOCHS = 20
STEPS_PER_EPOCH = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
REDUCE_LR = {'patience': 3, 'factor': 0.5}

# display enhancer
LUM_LOW_PCT, LUM_HIGH_PCT = 2.0, 99.7
DISPLAY_GAIN, DISPLAY_GAMMA = 1.15, 0.90

# ============== Utils ==============
def setup_gpu_and_log():
    print("="*60)
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    if TORCH_AVAILABLE:
        print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try: print("CUDA device:", torch.cuda.get_device_name(0))
            except Exception: pass
    else:
        print("Torch: not installed")
    if TF_AVAILABLE:
        print("TensorFlow:", tf.__version__)
        gpus = tf.config.list_physical_devices('GPU')
        print("TF GPUs:", gpus)
        if gpus:
            for g in gpus:
                try: tf.config.experimental.set_memory_growth(g, True)
                except Exception: pass
    else:
        print("TensorFlow: not installed")
    try:
        import cellpose as _cp
        print("Cellpose:", getattr(_cp, "__version__", "unknown"))
    except Exception:
        pass
    print("="*60)

def normalize(x):
    x = x.astype(np.float32)
    mn, mx = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x, np.float32) if mx <= mn else (x - mn) / (mx - mn)

def auto_contrast(img_rgb, low_pct=1, high_pct=99, gamma=1.0):
    p_low = np.percentile(img_rgb, low_pct)
    p_high = np.percentile(img_rgb, high_pct)
    img = np.clip((img_rgb - p_low) / (p_high - p_low + 1e-8), 0, 1)
    return exposure.adjust_gamma(img, gamma)

def enhance_rgb_for_display(rgb_u8: np.ndarray) -> np.ndarray:
    x = rgb_u8.astype(np.float32) / 255.0
    Y = 0.2126 * x[...,0] + 0.7152 * x[...,1] + 0.0722 * x[...,2]
    lo = np.percentile(Y, LUM_LOW_PCT)
    hi = np.percentile(Y, LUM_HIGH_PCT)
    if hi <= lo:
        lo, hi = (Y.min(), Y.max()) if Y.max() > Y.min() else (0.0, 1.0)
    x = (x - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0, 1)
    x = np.clip(x * DISPLAY_GAIN, 0, 1)
    x = np.power(x, DISPLAY_GAMMA)
    return np.clip(x, 0, 1)

def read_valid_rois(zip_path, shape):
    if not zip_path or not os.path.exists(zip_path):
        return np.zeros(shape, dtype=np.uint16)
    try:
        rois = read_roi_zip(zip_path)
    except Exception as e:
        print(f"✖ ROI read error {zip_path}: {e}")
        return np.zeros(shape, dtype=np.uint16)
    mask = np.zeros(shape, dtype=np.uint16)
    for i, r in enumerate(rois.values(), 1):
        if 'x' not in r or 'y' not in r or len(r['x']) < 3:
            continue
        y, x = r['y'], r['x']
        try:
            poly = polygon2mask(shape, np.column_stack((y, x)))
            mask[poly] = i
        except Exception:
            pass
    mask = remove_small_objects(mask, min_size=5)
    return sk_label(mask > 0)

def dice_score(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    return 2.0 * inter / (gt.sum() + pred.sum() + 1e-8)

def iou_score(gt, pred):
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return inter / (union + 1e-8)

def debug_filter(lbl, min_area=10, min_solidity=0.3):
    props = regionprops(lbl)
    out = np.zeros_like(lbl)
    k = 1
    for r in props:
        if r.area >= min_area and r.solidity >= min_solidity:
            out[lbl == r.label] = k; k += 1
    return out

def split_touching_cells(mask, min_distance=10):
    proc = mask > 0
    dist = gaussian(ndi.distance_transform_edt(proc), sigma=0.7)
    coords = peak_local_max(dist, min_distance=min_distance, labels=proc)
    peaks = np.zeros_like(dist, bool)
    if coords.size:
        for y, x in coords: peaks[y, x] = True
    else:
        peaks[np.unravel_index(np.argmax(dist), dist.shape)] = True
    markers, _ = ndi.label(peaks)
    labels = watershed(-dist, markers, mask=proc)
    labels = remove_small_objects(labels, min_size=10)
    return debug_filter(labels)

def extract_czyx_channels_from_czi(raw):
    arr = np.squeeze(raw)
    if arr.ndim == 5:
        arr = arr[0]
        if arr.shape[0] < 3 and arr.shape[1] >= 3:
            arr = np.moveaxis(arr, 0, 1)
        arr = arr[:, 0]
    elif arr.ndim == 4:
        if arr.shape[0] < 3 and arr.shape[1] >= 3:
            arr = np.moveaxis(arr, 0, 1)
        arr = arr[:, 0]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        raise ValueError(f"Unexpected CZI shape: {arr.shape}")
    return normalize(arr.astype(np.float32))

def psnr_ssim(a, b):
    a, b = normalize(a), normalize(b)
    return (peak_signal_noise_ratio(a, b, data_range=1.0),
            structural_similarity(a, b, data_range=1.0))

def _fmt_metric(x):
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.3f}"

# ---------- Professional µm scale bar (version-agnostic styling) ----------
def _nice_bar_um(max_um: float) -> float:
    if max_um <= 0: return 1.0
    nice = [1, 2, 5]
    power = -6
    best = 1.0
    while True:
        any_fit = False
        for b in nice:
            cand = b * (10 ** power)
            if cand <= max_um:
                best = cand; any_fit = True
            else:
                return best if any_fit else max_um
        power += 1

def _style_asb_label(asb, color='white', fontsize=12):
    """Safely style the scalebar label across Matplotlib versions."""
    # Try common attributes first
    for attr in ('txt_label', 'label_txt'):
        obj = getattr(asb, attr, None)
        if obj is None:
            continue
        # If it's a Text object
        try:
            obj.set_color(color)
            obj.set_fontsize(fontsize)
            obj.set_path_effects([pe.withStroke(linewidth=2.5, foreground='black')])
            return
        except Exception:
            pass
        # If it's a container/ TextArea: style its children
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
    # Fallback: do nothing (label remains default)

def add_scale_bar_um(ax,
                     image_shape,
                     microns_per_pixel: float,
                     bar_um: float | None = None,
                     loc: str = "lower right",
                     text_size: int = 12):
    """AnchoredSizeBar in data units with white bar + haloed label (portable across Matplotlib versions)."""
    H, W = image_shape[:2]
    img_width_um = W * microns_per_pixel
    if bar_um is None:
        bar_um = _nice_bar_um(img_width_um * 0.25)
    bar_px = bar_um / microns_per_pixel

    loc_map = {'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4}
    asb = AnchoredSizeBar(ax.transData,
                          size=bar_px,
                          label=f'{bar_um:g} µm',
                          loc=loc_map.get(loc, 4),
                          pad=0.4, sep=4, borderpad=0.8,
                          frameon=False,
                          size_vertical=max(2, int(0.006 * H)),
                          color='white')  # bar color
    # Safely style the label
    _style_asb_label(asb, color='white', fontsize=text_size)

    # Edge on the bar for contrast (portable: 'size_bar' exists widely)
    try:
        asb.size_bar.set_edgecolor('black')
        asb.size_bar.set_linewidth(max(1, int(0.002 * H)))
    except Exception:
        pass

    ax.add_artist(asb)

def _freeze_image_limits(ax, H, W):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # origin='upper'
    ax.set_aspect('equal', adjustable='box')

# ============== N2V ==============
def _pad_to_multiple(img2d: np.ndarray, m: int = 4):
    H, W = img2d.shape
    Hn = ((H + m - 1) // m) * m
    Wn = ((W + m - 1) // m) * m
    pad_y, pad_x = Hn - H, Wn - W
    t, l = pad_y // 2, pad_x // 2
    b, r = pad_y - t, pad_x - l
    if pad_y or pad_x:
        pad_img = np.pad(img2d, ((t, b), (l, r)), mode='reflect')
        return pad_img, (slice(t, t + H), slice(l, l + W))
    return img2d, (slice(0, H), slice(0, W))

def train_n2v_for_channel(img2d, model_name, basedir):
    if not N2V_AVAIL:
        raise RuntimeError("Noise2Void (n2v) is not installed. `pip install n2v`")
    os.makedirs(basedir, exist_ok=True)
    rng = np.random.default_rng(42)
    h, w = img2d.shape
    ph, pw = PATCH_SIZE
    patches = []
    for _ in range(N_PATCHES):
        top = rng.integers(0, max(1, h - ph))
        left = rng.integers(0, max(1, w - pw))
        patches.append(img2d[top:top+ph, left:left+pw][..., None])
    patches = np.stack(patches).astype(np.float32)
    n_val = int(len(patches) * VAL_SPLIT)
    val_X, tr_X = patches[:n_val], patches[n_val:]
    cfg = N2VConfig(
        X=patches, axes='YXC',
        unet_kern_size=3,
        train_steps_per_epoch=STEPS_PER_EPOCH,
        train_epochs=TRAIN_EPOCHS, train_loss='mse',
        batch_norm=True, train_batch_size=BATCH_SIZE,
        n2v_perc_pix=0.198, n2v_patch_shape=PATCH_SIZE,
        train_learning_rate=LEARNING_RATE,
        train_reduce_lr=REDUCE_LR
    )
    n2v = N2V(config=cfg, name=model_name, basedir=basedir)
    n2v.train(tr_X, val_X, epochs=TRAIN_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    pad_img, crop = _pad_to_multiple(img2d, 4)
    pred_full = n2v.predict(pad_img, axes='YX').astype(np.float32)
    pred = pred_full[crop]
    return normalize(pred)

# ============== Overlays (individual) ==============
def show_case_overlay(img_path: str,
                      version_label: str,
                      model_name: str,
                      pred_mask: np.ndarray,
                      roi_path: str | None,
                      rgb_display_base_u8: np.ndarray,
                      dice: float,
                      iou: float):
    rgb_vis = enhance_rgb_for_display(rgb_display_base_u8)
    H, W = rgb_vis.shape[:2]
    pred_label = split_touching_cells(pred_mask)
    pred_bound = find_boundaries(pred_label, mode='outer')

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.imshow(rgb_vis, interpolation='nearest')
    ax.axis('off')
    _freeze_image_limits(ax, H, W)

    if roi_path and os.path.exists(roi_path):
        gt_label = read_valid_rois(roi_path, rgb_vis.shape[:2])
        gt_bound = find_boundaries(gt_label, mode='outer')
        ax.contour(gt_bound, colors='deepskyblue', linewidths=0.9)
    ax.contour(pred_bound, colors='yellow', linewidths=0.9)

    add_scale_bar_um(ax, rgb_vis.shape, MICRONS_PER_PIXEL,
                     bar_um=BAR_MICRONS, loc=SCALEBAR_LOC, text_size=12)

    ax.set_title(
        f"{Path(img_path).name} — {version_label} · {model_name} "
        f"(Dice: {_fmt_metric(dice)}, IoU: {_fmt_metric(iou)})",
        fontsize=12, color='black'
    )
    plt.show()
    plt.close(fig)

def save_case_overlay(img_path: str,
                      version_label: str,
                      model_name: str,
                      pred_mask: np.ndarray,
                      roi_path: str | None,
                      rgb_display_base_u8: np.ndarray,
                      out_dir: Path,
                      dice: float,
                      iou: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    out_png = out_dir / f"{stem}__{version_label}__{model_name}.png"

    rgb_vis = enhance_rgb_for_display(rgb_display_base_u8)
    H, W = rgb_vis.shape[:2]
    pred_label = split_touching_cells(pred_mask)
    pred_bound = find_boundaries(pred_label, mode='outer')

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.imshow(rgb_vis, interpolation='nearest')
    ax.axis('off')
    _freeze_image_limits(ax, H, W)

    if roi_path and os.path.exists(roi_path):
        gt_label = read_valid_rois(roi_path, rgb_vis.shape[:2])
        gt_bound = find_boundaries(gt_label, mode='outer')
        ax.contour(gt_bound, colors='deepskyblue', linewidths=0.9)
    ax.contour(pred_bound, colors='yellow', linewidths=0.9)

    add_scale_bar_um(ax, rgb_vis.shape, MICRONS_PER_PIXEL,
                     bar_um=BAR_MICRONS, loc=SCALEBAR_LOC, text_size=12)

    ax.set_title(
        f"{Path(img_path).name} — {version_label} · {model_name} "
        f"(Dice: {_fmt_metric(dice)}, IoU: {_fmt_metric(iou)})",
        fontsize=12, color='black'
    )
    plt.savefig(out_png, dpi=160, bbox_inches=None, pad_inches=0.1)
    plt.close(fig)
    print(f"[Saved] {out_png}")

# ============== Main ==============
def run_pipeline():
    setup_gpu_and_log()
    if len(image_files) != len(roi_files):
        print("Warning: image/ROI count mismatch; pairing by index.")

    OVER_CASES = Path("./overlays_cases"); OVER_CASES.mkdir(exist_ok=True)

    # 1) N2V + PSNR/SSIM
    psnr_rows = []
    for i,(img_path, roi_path) in enumerate(zip(image_files, roi_files), 1):
        if not os.path.exists(img_path): raise FileNotFoundError(img_path)
        raw = imread(img_path)
        cyx = extract_czyx_channels_from_czi(raw)
        c = cyx.shape[0]
        b_idx = 0 if c>=1 else 0
        g_idx = 1 if c>=2 else 0
        r_idx = 2 if c>=3 else c-1
        base = img_path[:-4] if img_path.lower().endswith('.czi') else img_path
        fR, fG, fB = f"{base}_denoised_R.npy", f"{base}_denoised_G.npy", f"{base}_denoised_B.npy"

        need = not (os.path.exists(fR) and os.path.exists(fG) and os.path.exists(fB))
        if need:
            if not N2V_AVAIL: raise RuntimeError("n2v missing & no cached denoised files.")
            print(f"\n[TRAIN N2V] {img_path}")
            R = normalize(cyx[r_idx]); G = normalize(cyx[g_idx]); B = normalize(cyx[b_idx])
            denR = train_n2v_for_channel(R, f"N2V_{i}_R", BASE_MODEL_DIR); np.save(fR, denR)
            denG = train_n2v_for_channel(G, f"N2V_{i}_G", BASE_MODEL_DIR); np.save(fG, denG)
            denB = train_n2v_for_channel(B, f"N2V_{i}_B", BASE_MODEL_DIR); np.save(fB, denB)
        else:
            print(f"[SKIP] Found denoised files for {img_path}")
            denR, denG, denB = np.load(fR), np.load(fG), np.load(fB)

        for ch_name, raw_ch, den_ch in [('R', cyx[r_idx], denR), ('G', cyx[g_idx], denG), ('B', cyx[b_idx], denB)]:
            p, s = psnr_ssim(raw_ch, den_ch)
            psnr_rows.append({'Image': os.path.basename(img_path), 'Channel': ch_name, 'PSNR': p, 'SSIM': s})

    pd.DataFrame(psnr_rows).to_csv("psnr_ssim_3ch.csv", index=False)
    print("Saved psnr_ssim_3ch.csv")

    # 2) Segmentation + overlays + metrics
    results = []
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    print("[Cellpose] Device:", "CUDA" if use_gpu else "CPU")
    models_dict = {mt: models.CellposeModel(gpu=use_gpu, device=device, model_type=mt) for mt in MODEL_TYPES}

    for i,(img_path, roi_path) in enumerate(zip(image_files, roi_files), 1):
        print(f"\n[SEGMENTATION] {img_path}")
        raw = imread(img_path)
        cyx = extract_czyx_channels_from_czi(raw)
        c = cyx.shape[0]
        b_idx = 0 if c>=1 else 0
        g_idx = 1 if c>=2 else 0
        r_idx = 2 if c>=3 else c-1

        R_raw, G_raw, B_raw = normalize(cyx[r_idx]), normalize(cyx[g_idx]), normalize(cyx[b_idx])
        rgb_display_base_u8 = img_as_ubyte(np.stack([R_raw, G_raw, B_raw], axis=-1))

        rgb_original_u8 = img_as_ubyte(auto_contrast(np.stack([R_raw, G_raw, B_raw], axis=-1), 1, 99, 1.0))

        base = img_path[:-4] if img_path.lower().endswith('.czi') else img_path
        denR = normalize(np.load(f"{base}_denoised_R.npy"))
        denG = normalize(np.load(f"{base}_denoised_G.npy"))
        denB = normalize(np.load(f"{base}_denoised_B.npy"))
        rgb_denoised_u8 = img_as_ubyte(auto_contrast(np.stack([denR, denG, denB], axis=-1), 1, 99, 1.0))

        for version_label, rgb_u8_in in [('Original', rgb_original_u8), ('Denoised', rgb_denoised_u8)]:
            for mt in MODEL_TYPES:
                pred_mask, _, _ = models_dict[mt].eval(rgb_u8_in, diameter=None, channels=[0, 1])
                pred_label = split_touching_cells(pred_mask)
                if roi_path and os.path.exists(roi_path):
                    gt = read_valid_rois(roi_path, rgb_u8_in.shape[:2])
                    dice = dice_score(gt > 0, (pred_label > 0) & (gt > 0))
                    iou  = iou_score (gt > 0, (pred_label > 0) & (gt > 0))
                else:
                    dice, iou = np.nan, np.nan

                results.append({'Image': Path(img_path).name,
                                'ROI': Path(roi_path).name if roi_path and os.path.exists(roi_path) else None,
                                'Type': version_label, 'Model': mt,
                                'Dice': dice, 'IoU': iou})

                # show + save each case with µm bar
                show_case_overlay(img_path, version_label, mt, pred_mask, roi_path,
                                  rgb_display_base_u8, dice, iou)
                save_case_overlay(img_path, version_label, mt, pred_mask, roi_path,
                                  rgb_display_base_u8, Path("./overlays_cases"), dice, iou)

    df = pd.DataFrame(results)
    df.to_csv("segmentation_metrics_overlay_3ch.csv", index=False)
    print("\nSaved segmentation_metrics_overlay_3ch.csv")
    if not df.empty:
        print(df.to_string(index=False))

if __name__ == "__main__":
    run_pipeline()
