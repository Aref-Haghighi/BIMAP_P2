#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full pipeline (interactive, strict GT pairing):

- Select image files (.czi, .tif/.tiff, .jpg/.jpeg/.png)
- Optionally select ground truth ROI files (.zip)
- Select output folder

Pairs images ↔ GTs by:
✅ Same ROI number (ROI1, ROI2, etc.)
✅ Same strain/condition keyword (WT, DpspA, THY, NHS, etc.)
✅ Falls back to fuzzy matching only if no match found.

Then:
- ONLY checks if Noise2Void denoised caches exist NEXT TO each image
- Runs Cellpose (cyto, cyto2)
- Post-processes (watershed split, small object removal)
- Saves overlays + Dice/IoU CSV
- Shows one figure at a time; after you close it, the next one appears.
"""

import os, sys, platform, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
# import matplotlib.patheffects as pe  # (not used)
from read_roi import read_roi_zip
from skimage.draw import polygon2mask
from skimage.util import img_as_ubyte
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries, watershed
from skimage.measure import regionprops, label as sk_label
from skimage.filters import gaussian
from skimage import exposure
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import pandas as pd
from cellpose import models
from difflib import SequenceMatcher

# ---- Torch availability (for GPU info) ----
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ============== Config ==============
MICRONS_PER_PIXEL = 0.0322
BAR_MICRONS = 2
SCALEBAR_LOC = "lower right"
MODEL_TYPES = ['cyto', 'cyto2']
LUM_LOW_PCT, LUM_HIGH_PCT = 2.0, 99.7
DISPLAY_GAIN, DISPLAY_GAMMA = 1.15, 0.90

# ============== Universal Image Reader ==============
try:
    from czifile import imread as czi_imread
    CZI_OK = True
except Exception:
    CZI_OK = False
    def czi_imread(path):
        raise ValueError("czifile not available")

from skimage.io import imread as sk_imread

def imread_any(path):
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext == ".czi" and CZI_OK:
        print(f"[READ] CZI image: {Path(path).name}")
        return czi_imread(path)
    else:
        print(f"[READ] Generic image: {Path(path).name}")
        return sk_imread(path)

# ============== File Selection ==============
def select_image_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    files = filedialog.askopenfilenames(
        title="Select image files (.czi/.tif/.tiff/.jpg/.jpeg/.png)",
        filetypes=[("Microscopy/Images", "*.czi *.tif *.tiff *.jpg *.jpeg *.png")]
    )
    root.update(); root.destroy()
    return list(files)

def select_gt_files_optional():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    root = tk.Tk(); root.withdraw()
    answer = messagebox.askyesno("Ground Truth", "Do you want to select ground truth ROI files (.zip)?")
    gt_files = []
    if answer:
        gt_files = filedialog.askopenfilenames(
            title="Select ground truth ROI files (.zip)",
            filetypes=[("ROIset ZIP", "*.zip"), ("All files", "*.*")]
        )
    root.update(); root.destroy()
    return list(gt_files)

def select_output_folder(default="./results"):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    outdir = filedialog.askdirectory(title="Select output folder")
    root.update(); root.destroy()
    return outdir or default

# ============== Strict Smart Pairing ==============
def smart_pair_images_and_rois(images, rois):
    """
    Strict pairing: requires same strain keyword + same ROI number.
    """
    def get_roi_num(name):
        m = re.search(r'roi[_\- ]?(\d+)', name.lower())
        return int(m.group(1)) if m else None

    def get_tokens(name):
        tokens = ['wt', 'dpspa', 'thy', 'nhs', 'nada', 'rada', 'hada', '40min']
        return [t for t in tokens if t in name.lower()]

    pairs = []
    for img in images:
        stem_img = Path(img).stem.lower()
        roi_img = get_roi_num(stem_img)
        toks_img = set(get_tokens(stem_img))

        best_gt, best_score = None, 0.0
        for roi in rois:
            stem_roi = Path(roi).stem.lower()
            roi_gt = get_roi_num(stem_roi)
            toks_gt = set(get_tokens(stem_roi))

            if roi_img is not None and roi_gt is not None and roi_img != roi_gt:
                continue
            if toks_img and toks_gt:
                if ('wt' in toks_img and 'dpspa' in toks_gt) or ('dpspa' in toks_img and 'wt' in toks_gt):
                    continue
                if not (toks_img & toks_gt):
                    continue

            score = SequenceMatcher(None, stem_img, stem_roi).ratio()
            if score > best_score:
                best_score, best_gt = score, roi

        if best_gt:
            print(f"[PAIR] {Path(img).name} ↔ {Path(best_gt).name} (score={best_score:.2f})")
        else:
            print(f"[PAIR] {Path(img).name} ↔ No matching GT found")
        pairs.append((img, best_gt))

    print("\n=== Final confirmed image ↔ GT pairs ===")
    for i, (img, gt) in enumerate(pairs):
        print(f"{i+1:02d}. {Path(img).name} ↔ {Path(gt).name if gt else 'None'}")
    return pairs

# ============== Utilities (same as before) ==============
def setup_gpu_and_log():
    print("="*60)
    print("Python:", sys.version)
    print("Platform:", platform.platform())
    if TORCH_AVAILABLE:
        print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            try:
                print("CUDA device:", torch.cuda.get_device_name(0))
            except Exception: pass
    else:
        print("Torch: not installed")
    try:
        import cellpose as _cp
        print("Cellpose:", getattr(_cp, "__version__", "unknown"))
    except Exception: pass
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

def enhance_rgb_for_display(rgb_u8):
    x = rgb_u8.astype(np.float32) / 255.0
    Y = 0.2126*x[...,0]+0.7152*x[...,1]+0.0722*x[...,2]
    lo, hi = np.percentile(Y, LUM_LOW_PCT), np.percentile(Y, LUM_HIGH_PCT)
    if hi <= lo: lo, hi = (Y.min(), Y.max()) if Y.max() > Y.min() else (0,1)
    x = np.clip((x - lo)/(hi-lo+1e-8), 0, 1)
    x = np.clip(x*DISPLAY_GAIN, 0, 1)
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
        if 'x' not in r or 'y' not in r or len(r['x']) < 3: continue
        y, x = r['y'], r['x']
        try:
            poly = polygon2mask(shape, np.column_stack((y, x)))
            mask[poly] = i
        except Exception: pass
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
    k=1
    for r in props:
        if r.area>=min_area and r.solidity>=min_solidity:
            out[lbl==r.label]=k; k+=1
    return out

def split_touching_cells(mask, min_distance=10):
    proc = mask > 0
    dist = gaussian(ndi.distance_transform_edt(proc), sigma=0.7)
    coords = peak_local_max(dist, min_distance=min_distance, labels=proc)
    peaks = np.zeros_like(dist, bool)
    if coords.size:
        for y,x in coords: peaks[y,x]=True
    else:
        peaks[np.unravel_index(np.argmax(dist), dist.shape)] = True
    markers,_ = ndi.label(peaks)
    labels = watershed(-dist, markers, mask=proc)
    labels = remove_small_objects(labels, min_size=10)
    return debug_filter(labels)

def extract_czyx_channels_from_any(raw):
    arr = np.squeeze(raw)
    if arr.ndim == 2:
        return arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[0] in [3,4]:
            return arr
        else:
            return np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

def add_scale_bar_um(ax, shape, um_per_px, bar_um, loc="lower right", text_size=12):
    H,W=shape[:2]
    bar_px=bar_um/um_per_px
    asb=AnchoredSizeBar(ax.transData,bar_px,label=f'{bar_um:g} µm',
                        loc=4,pad=0.4,sep=4,borderpad=0.8,
                        frameon=False,size_vertical=max(2,int(0.006*H)),color='white')
    ax.add_artist(asb)

# ============== Denoised-triplet check (same folder) ==============
def find_denoised_triplet_next_to_image(img_path: str):
    """
    Looks for <stem>_denoised_R.npy/G.npy/B.npy in the SAME directory as img_path.
    Returns (fR, fG, fB) if all exist; otherwise prints what's missing and returns None.
    """
    p = Path(img_path)
    fR = p.with_name(f"{p.stem}_denoised_R.npy")
    fG = p.with_name(f"{p.stem}_denoised_G.npy")
    fB = p.with_name(f"{p.stem}_denoised_B.npy")

    missing = [str(x.name) for x in [fR, fG, fB] if not x.exists()]
    if missing:
        print(f"[N2V] Missing in same folder for {p.name}: {', '.join(missing)}")
        return None
    return fR, fG, fB

# ---------- Main ----------
def run_pipeline():
    image_files = select_image_files()
    gt_files = select_gt_files_optional()
    out_dir = select_output_folder("./results")

    if not image_files:
        print("✖ No image files selected. Exiting."); return

    pairs = smart_pair_images_and_rois(image_files, gt_files)
    setup_gpu_and_log()
    OVER_CASES = Path(out_dir) / "overlays_cases"; OVER_CASES.mkdir(parents=True, exist_ok=True)

    results=[]
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu") if TORCH_AVAILABLE else None
    print("[Cellpose] Device:", "CUDA" if use_gpu else "CPU")
    models_dict={mt:models.CellposeModel(gpu=use_gpu,device=device,model_type=mt) for mt in MODEL_TYPES}

    for img_path, roi_path in pairs:
        print(f"\n[SEGMENTATION] {img_path}")
        if not os.path.exists(img_path):
            print(f"✖ Missing image: {img_path}"); continue

        # Read raw → RGB
        raw = imread_any(img_path)
        cyx = extract_czyx_channels_from_any(raw)
        c = cyx.shape[0]
        R,G,B = normalize(cyx[min(0,c-3)]), normalize(cyx[min(1,c-2)]), normalize(cyx[min(2,c-1)])
        rgb_base = img_as_ubyte(np.stack([R,G,B],axis=-1))
        rgb_orig = img_as_ubyte(auto_contrast(np.stack([R,G,B],axis=-1)))

        # Check for denoised triplet in the SAME folder
        den_paths = find_denoised_triplet_next_to_image(img_path)
        variants = [('Original', rgb_orig)]

        if den_paths is not None:
            denR_path, denG_path, denB_path = den_paths
            denR = normalize(np.load(denR_path)); denG = normalize(np.load(denG_path)); denB = normalize(np.load(denB_path))
            rgb_den = img_as_ubyte(auto_contrast(np.stack([denR,denG,denB],axis=-1)))
            variants.append(('Denoised', rgb_den))

        for vlabel, rgb_in in variants:
            for mt in MODEL_TYPES:
                pred_mask,_,_=models_dict[mt].eval(rgb_in,diameter=None,channels=[0,1])
                pred_label=split_touching_cells(pred_mask)

                if roi_path and os.path.exists(roi_path):
                    gt=read_valid_rois(roi_path,rgb_in.shape[:2])
                    dice=dice_score(gt>0,(pred_label>0)&(gt>0))
                    iou=iou_score(gt>0,(pred_label>0)&(gt>0))
                else:
                    dice,iou=np.nan,np.nan

                results.append({'Image':Path(img_path).name,'ROI':Path(roi_path).name if roi_path else None,
                                'Type':vlabel,'Model':mt,'Dice':dice,'IoU':iou})

                fig,ax=plt.subplots(figsize=(10,7),facecolor='white')
                ax.imshow(enhance_rgb_for_display(rgb_base)); ax.axis('off')
                pred_bound=find_boundaries(pred_label,mode='outer')
                if roi_path and os.path.exists(roi_path):
                    gt_label=read_valid_rois(roi_path,rgb_in.shape[:2])
                    gt_bound=find_boundaries(gt_label,mode='outer')
                    ax.contour(gt_bound,colors='deepskyblue',linewidths=0.9)
                ax.contour(pred_bound,colors='yellow',linewidths=0.9)
                add_scale_bar_um(ax,rgb_in.shape,MICRONS_PER_PIXEL,BAR_MICRONS,SCALEBAR_LOC,12)
                ax.set_title(f"{Path(img_path).name} — {vlabel} · {mt}\n(Dice {dice:.3f} | IoU {iou:.3f})",
                             fontsize=12,color='black')
                out_png=OVER_CASES/f"{Path(img_path).stem}__{vlabel}__{mt}.png"
                fig.savefig(out_png,dpi=160,bbox_inches='tight')
                print(f"[Saved] {out_png}")

                # >>> Show ONE figure at a time (blocking) <<<
                try:
                    plt.show()   # blocks until you close this figure window
                finally:
                    plt.close(fig)

    # Save metrics
    df=pd.DataFrame(results)
    out_csv=Path(out_dir)/"segmentation_metrics_overlay_3ch.csv"
    df.to_csv(out_csv,index=False)
    print(f"\nSaved metrics: {out_csv}")
    if not df.empty:
        print(df.to_string(index=False))

if __name__=="__main__":
    run_pipeline()
