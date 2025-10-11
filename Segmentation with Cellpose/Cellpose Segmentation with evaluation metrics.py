#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full pipeline (interactive, strict GT pairing):

- Select image files (.czi, .tif/.tiff, .jpg/.jpeg/.png)
- Optionally select ground truth ROI files (.zip)
- Select output folder

Pairs images ↔ GTs by:
Same ROI number (ROI1, ROI2, etc.)
Same strain/condition keyword (WT, DpspA, THY, NHS, etc.)
Falls back to fuzzy matching only if no match found.

Then:
- ONLY checks if Noise2Void denoised caches exist NEXT TO each image
- Runs Cellpose (cyto, cyto2)
- Post-processes (watershed split, small object removal)
- Saves overlays + Dice/IoU CSV
- Shows one figure at a time; after you close it, the next one appears.
"""

# --- Standard libs and core scientific/plotting stack ---
import os, sys, platform, re                          # OS utilities, interpreter/platform info, regex
from pathlib import Path                               # Path handling (cross-platform)
import numpy as np                                     # Numeric arrays
import matplotlib.pyplot as plt                        # Plotting and figure generation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # Scale bar overlay artist
# import matplotlib.patheffects as pe  # (not used)       # Left commented intentionally (matches your original)
from read_roi import read_roi_zip                      # Parse ImageJ ROI .zip
from skimage.draw import polygon2mask                  # Convert ROI polygons → binary masks
from skimage.util import img_as_ubyte                  # Convert floats → uint8 for display
from skimage.morphology import remove_small_objects    # Remove tiny labels/regions
from skimage.segmentation import find_boundaries, watershed  # Boundaries + watershed
from skimage.measure import regionprops, label as sk_label   # Region properties + relabeling
from skimage.filters import gaussian                   # Gaussian smoothing
from skimage import exposure                           # Contrast/gamma ops
from skimage.feature import peak_local_max             # Peak detection for seeds
from scipy import ndimage as ndi                       # Distance transform, labeling
import pandas as pd                                    # Tabular results
from cellpose import models                            # Cellpose model API
from difflib import SequenceMatcher                    # Fuzzy matching for pairing fallback

# ---- Torch availability (for GPU info) ----
try:
    import torch                                       # Try to import PyTorch for GPU capabilities
    TORCH_AVAILABLE = True                             # Flag: torch import succeeded
except Exception:
    TORCH_AVAILABLE = False                            # Flag: torch import failed

# ============== Config ==============
MICRONS_PER_PIXEL = 0.0322                             # Physical scale (µm/px) for scale bar
BAR_MICRONS = 2                                        # Default scale bar length (µm)
SCALEBAR_LOC = "lower right"                           # Scale bar location string (not used for mapping here)
MODEL_TYPES = ['cyto', 'cyto2']                        # Cellpose model types to evaluate
LUM_LOW_PCT, LUM_HIGH_PCT = 2.0, 99.7                 # Display stretch percentiles
DISPLAY_GAIN, DISPLAY_GAMMA = 1.15, 0.90              # Gain and gamma for display enhancement

# ============== Universal Image Reader ==============
try:
    from czifile import imread as czi_imread           # CZI reader (if available)
    CZI_OK = True                                      # Flag for CZI support
except Exception:
    CZI_OK = False                                     # No czifile installed/available
    def czi_imread(path):                              # Stub to keep symbol defined
        raise ValueError("czifile not available")

from skimage.io import imread as sk_imread             # Generic image reader

def imread_any(path):
    path = str(path)                                   # Ensure string path
    ext = Path(path).suffix.lower()                    # File extension
    if ext == ".czi" and CZI_OK:                       # Use CZI reader when available
        print(f"[READ] CZI image: {Path(path).name}")
        return czi_imread(path)                        # Returns ndarray per czifile convention
    else:
        print(f"[READ] Generic image: {Path(path).name}")
        return sk_imread(path)                         # Fallback to skimage reader

# ============== File Selection ==============
def select_image_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()                    # Hide root window
    files = filedialog.askopenfilenames(               # Multi-select dialog for images
        title="Select image files (.czi/.tif/.tiff/.jpg/.jpeg/.png)",
        filetypes=[("Microscopy/Images", "*.czi *.tif *.tiff *.jpg *.jpeg *.png")]
    )
    root.update(); root.destroy()                      # Cleanly close Tk
    return list(files)                                 # Return list of selected paths

def select_gt_files_optional():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    root = tk.Tk(); root.withdraw()                    # Hide root window
    answer = messagebox.askyesno("Ground Truth", "Do you want to select ground truth ROI files (.zip)?")
    gt_files = []
    if answer:                                         # If user wants to provide GTs
        gt_files = filedialog.askopenfilenames(
            title="Select ground truth ROI files (.zip)",
            filetypes=[("ROIset ZIP", "*.zip"), ("All files", "*.*")]
        )
    root.update(); root.destroy()                      # Close Tk
    return list(gt_files)                              # Possibly empty list if user chose "No"

def select_output_folder(default="./results"):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()                    # Hide root window
    outdir = filedialog.askdirectory(title="Select output folder")  # Directory chooser
    root.update(); root.destroy()                      # Close Tk
    return outdir or default                           # Default if user cancels

# ============== Strict Smart Pairing ==============
def smart_pair_images_and_rois(images, rois):
    """
    Strict pairing: requires same strain keyword + same ROI number.
    """
    def get_roi_num(name):
        m = re.search(r'roi[_\- ]?(\d+)', name.lower()) # Extract "ROI<number>" pattern
        return int(m.group(1)) if m else None           # Return int or None

    def get_tokens(name):
        tokens = ['wt', 'dpspa', 'thy', 'nhs', 'nada', 'rada', 'hada', '40min']  # Domain-specific tokens
        return [t for t in tokens if t in name.lower()]  # Keep tokens present in name

    pairs = []                                          # (image, matched_roi_or_None)
    for img in images:
        stem_img = Path(img).stem.lower()               # Lowercase stem for matching
        roi_img = get_roi_num(stem_img)                 # Extract ROI index from image name
        toks_img = set(get_tokens(stem_img))            # Extract condition tokens from image name

        best_gt, best_score = None, 0.0                 # Track best fuzzy match within constraints
        for roi in rois:
            stem_roi = Path(roi).stem.lower()
            roi_gt = get_roi_num(stem_roi)              # Extract ROI index from GT name
            toks_gt = set(get_tokens(stem_roi))         # Extract condition tokens from GT name

            if roi_img is not None and roi_gt is not None and roi_img != roi_gt:
                continue                                # ROI number mismatch: skip

            if toks_img and toks_gt:                    # If both sides have condition tokens
                # Disallow pairing WT↔DpspA mismatched condition
                if ('wt' in toks_img and 'dpspa' in toks_gt) or ('dpspa' in toks_img and 'wt' in toks_gt):
                    continue
                if not (toks_img & toks_gt):            # Require at least one shared token
                    continue

            score = SequenceMatcher(None, stem_img, stem_roi).ratio()  # Fuzzy similarity
            if score > best_score:
                best_score, best_gt = score, roi        # Keep best-so-far

        if best_gt:
            print(f"[PAIR] {Path(img).name} ↔ {Path(best_gt).name} (score={best_score:.2f})")
        else:
            print(f"[PAIR] {Path(img).name} ↔ No matching GT found")
        pairs.append((img, best_gt))                    # Pair may contain None for GT

    print("\n=== Final confirmed image ↔ GT pairs ===")
    for i, (img, gt) in enumerate(pairs):
        print(f"{i+1:02d}. {Path(img).name} ↔ {Path(gt).name if gt else 'None'}")
    return pairs                                        # Return list of (image, roi_or_None)

# ============== Utilities (same as before) ==============
def setup_gpu_and_log():
    print("="*60)
    print("Python:", sys.version)                       # Interpreter version
    print("Platform:", platform.platform())             # OS/platform info
    if TORCH_AVAILABLE:
        print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True       # Enable cuDNN autotune for perf
            try:
                print("CUDA device:", torch.cuda.get_device_name(0))  # Device name
            except Exception: pass                      # Optional info; ignore errors
    else:
        print("Torch: not installed")                   # CPU-only or torch unavailable
    try:
        import cellpose as _cp
        print("Cellpose:", getattr(_cp, "__version__", "unknown"))  # Cellpose version if available
    except Exception: pass
    print("="*60)

def normalize(x):
    x = x.astype(np.float32)                            # Work in float32
    mn, mx = float(np.min(x)), float(np.max(x))         # Range for normalization
    return np.zeros_like(x, np.float32) if mx <= mn else (x - mn) / (mx - mn)

def auto_contrast(img_rgb, low_pct=1, high_pct=99, gamma=1.0):
    p_low = np.percentile(img_rgb, low_pct)             # Lower percentile
    p_high = np.percentile(img_rgb, high_pct)           # Upper percentile
    img = np.clip((img_rgb - p_low) / (p_high - p_low + 1e-8), 0, 1)  # Stretch to [0,1]
    return exposure.adjust_gamma(img, gamma)            # Optional gamma correction

def enhance_rgb_for_display(rgb_u8):
    x = rgb_u8.astype(np.float32) / 255.0               # Convert uint8→float in [0,1]
    Y = 0.2126*x[...,0]+0.7152*x[...,1]+0.0722*x[...,2] # Luminance plane
    lo, hi = np.percentile(Y, LUM_LOW_PCT), np.percentile(Y, LUM_HIGH_PCT)  # Percentile stretch
    if hi <= lo: lo, hi = (Y.min(), Y.max()) if Y.max() > Y.min() else (0,1)
    x = np.clip((x - lo)/(hi-lo+1e-8), 0, 1)            # Stretch to [0,1]
    x = np.clip(x*DISPLAY_GAIN, 0, 1)                   # Global gain
    x = np.power(x, DISPLAY_GAMMA)                      # Gamma curve
    return np.clip(x, 0, 1)                             # Final clip

def read_valid_rois(zip_path, shape):
    if not zip_path or not os.path.exists(zip_path):    # Handle missing GT gracefully
        return np.zeros(shape, dtype=np.uint16)
    try:
        rois = read_roi_zip(zip_path)                   # Load ROI dict from ImageJ zip
    except Exception as e:
        print(f"✖ ROI read error {zip_path}: {e}")      # Report parse error
        return np.zeros(shape, dtype=np.uint16)
    mask = np.zeros(shape, dtype=np.uint16)             # Start empty label mask
    for i, r in enumerate(rois.values(), 1):            # Enumerate polygons starting at 1
        if 'x' not in r or 'y' not in r or len(r['x']) < 3: continue  # Skip invalid polygons
        y, x = r['y'], r['x']                           # Coordinates are arrays
        try:
            poly = polygon2mask(shape, np.column_stack((y, x)))  # Rasterize polygon
            mask[poly] = i                               # Assign label id
        except Exception: pass                           # Skip malformed ROI
    mask = remove_small_objects(mask, min_size=5)        # Remove tiny blobs
    return sk_label(mask > 0)                            # Binarize and relabel (1..N)

def dice_score(gt, pred):
    inter = np.logical_and(gt, pred).sum()              # Intersection count
    return 2.0 * inter / (gt.sum() + pred.sum() + 1e-8) # Dice = 2|A∩B| / (|A|+|B|)

def iou_score(gt, pred):
    inter = np.logical_and(gt, pred).sum()              # Intersection count
    union = np.logical_or(gt, pred).sum()               # Union count
    return inter / (union + 1e-8)                       # IoU = |A∩B| / |A∪B|

def debug_filter(lbl, min_area=10, min_solidity=0.3):
    props = regionprops(lbl)                             # Region props from labeled mask
    out = np.zeros_like(lbl)                             # Output relabeled image
    k=1
    for r in props:
        if r.area>=min_area and r.solidity>=min_solidity:  # Keep by size and solidity
            out[lbl==r.label]=k; k+=1                   # Assign compact labels
    return out

def split_touching_cells(mask, min_distance=10):
    proc = mask > 0                                     # Binary foreground
    dist = gaussian(ndi.distance_transform_edt(proc), sigma=0.7)  # Smooth DT for nice peaks
    coords = peak_local_max(dist, min_distance=min_distance, labels=proc)  # Seed peaks
    peaks = np.zeros_like(dist, bool)                   # Peak map
    if coords.size:
        for y,x in coords: peaks[y,x]=True              # Mark peaks
    else:
        peaks[np.unravel_index(np.argmax(dist), dist.shape)] = True  # Fallback: single max peak
    markers,_ = ndi.label(peaks)                        # Label seeds
    labels = watershed(-dist, markers, mask=proc)       # Watershed split on inverted DT
    labels = remove_small_objects(labels, min_size=10)  # Remove tiny labels
    return debug_filter(labels)                         # Apply solidity/area filter

def extract_czyx_channels_from_any(raw):
    arr = np.squeeze(raw)                               # Remove singleton axes
    if arr.ndim == 2:
        return arr[None, :, :]                          # Promote to (C=1,H,W)
    elif arr.ndim == 3:
        if arr.shape[0] in [3,4]:
            return arr                                  # Already (C,H,W)
        else:
            return np.moveaxis(arr, -1, 0)              # If (H,W,C) → (C,H,W)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")  # Unexpected dimensionality

def add_scale_bar_um(ax, shape, um_per_px, bar_um, loc="lower right", text_size=12):
    H,W=shape[:2]                                       # Image height/width
    bar_px=bar_um/um_per_px                             # Convert microns → pixels
    asb=AnchoredSizeBar(ax.transData,bar_px,label=f'{bar_um:g} µm',
                        loc=4,pad=0.4,sep=4,borderpad=0.8,
                        frameon=False,size_vertical=max(2,int(0.006*H)),color='white')
    ax.add_artist(asb)                                  # Add anchored scale bar artist

# ============== Denoised-triplet check (same folder) ==============
def find_denoised_triplet_next_to_image(img_path: str):
    """
    Looks for <stem>_denoised_R.npy/G.npy/B.npy in the SAME directory as img_path.
    Returns (fR, fG, fB) if all exist; otherwise prints what's missing and returns None.
    """
    p = Path(img_path)                                  # Path object for input image
    fR = p.with_name(f"{p.stem}_denoised_R.npy")        # Expected R cache path
    fG = p.with_name(f"{p.stem}_denoised_G.npy")        # Expected G cache path
    fB = p.with_name(f"{p.stem}_denoised_B.npy")        # Expected B cache path

    missing = [str(x.name) for x in [fR, fG, fB] if not x.exists()]  # List missing caches
    if missing:
        print(f"[N2V] Missing in same folder for {p.name}: {', '.join(missing)}")
        return None                                     # Return None if any channel missing
    return fR, fG, fB                                   # Return the triplet paths

# ---------- Main ----------
def run_pipeline():
    image_files = select_image_files()                  # Ask user to choose images
    gt_files = select_gt_files_optional()               # Optionally choose GT ROI zips
    out_dir = select_output_folder("./results")         # Choose output directory (default ./results)

    if not image_files:
        print("✖ No image files selected. Exiting."); return  # Abort if no inputs

    pairs = smart_pair_images_and_rois(image_files, gt_files)  # Pair each image to a GT (or None)
    setup_gpu_and_log()                                # Print environment + GPU/Cellpose info
    OVER_CASES = Path(out_dir) / "overlays_cases"; OVER_CASES.mkdir(parents=True, exist_ok=True)  # Overlay output dir

    results=[]                                         # Accumulate metrics per case
    use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()  # Decide if GPU is available
    device = torch.device("cuda") if use_gpu else torch.device("cpu") if TORCH_AVAILABLE else None  # Preferred device
    print("[Cellpose] Device:", "CUDA" if use_gpu else "CPU")
    models_dict={mt:models.CellposeModel(gpu=use_gpu,device=device,model_type=mt) for mt in MODEL_TYPES}  # Load models

    for img_path, roi_path in pairs:                   # Iterate image/GT pairs (GT may be None)
        print(f"\n[SEGMENTATION] {img_path}")
        if not os.path.exists(img_path):
            print(f"✖ Missing image: {img_path}"); continue  # Skip unavailable image

        # Read raw → RGB
        raw = imread_any(img_path)                     # Read CZI or generic format
        cyx = extract_czyx_channels_from_any(raw)      # Ensure (C,H,W)
        c = cyx.shape[0]                               # Number of channels available
        R,G,B = normalize(cyx[min(0,c-3)]), normalize(cyx[min(1,c-2)]), normalize(cyx[min(2,c-1)])  # Pick 3 channels
        rgb_base = img_as_ubyte(np.stack([R,G,B],axis=-1))     # Base RGB for overlay background
        rgb_orig = img_as_ubyte(auto_contrast(np.stack([R,G,B],axis=-1)))  # Contrast-stretched RGB for inference

        # Check for denoised triplet in the SAME folder
        den_paths = find_denoised_triplet_next_to_image(img_path)  # Look for *_denoised_[RGB].npy
        variants = [('Original', rgb_orig)]              # Start with original variant

        if den_paths is not None:
            denR_path, denG_path, denB_path = den_paths
            denR = normalize(np.load(denR_path)); denG = normalize(np.load(denG_path)); denB = normalize(np.load(denB_path))  # Load N2V caches
            rgb_den = img_as_ubyte(auto_contrast(np.stack([denR,denG,denB],axis=-1)))  # Contrast-stretched denoised
            variants.append(('Denoised', rgb_den))       # Add denoised variant

        for vlabel, rgb_in in variants:                  # For Original/Denoised variants
            for mt in MODEL_TYPES:                       # For each Cellpose model type
                pred_mask,_,_=models_dict[mt].eval(rgb_in,diameter=None,channels=[0,1])  # Run Cellpose; channel map [0,1]
                pred_label=split_touching_cells(pred_mask)  # Post-process with watershed splitting

                if roi_path and os.path.exists(roi_path):     # If we have GT for this image
                    gt=read_valid_rois(roi_path,rgb_in.shape[:2])                 # Read GT labels
                    dice=dice_score(gt>0,(pred_label>0)&(gt>0))                   # Dice on overlap-only regime
                    iou=iou_score(gt>0,(pred_label>0)&(gt>0))                     # IoU on overlap-only regime
                else:
                    dice,iou=np.nan,np.nan                                       # No GT → NaN metrics

                results.append({'Image':Path(img_path).name,'ROI':Path(roi_path).name if roi_path else None,
                                'Type':vlabel,'Model':mt,'Dice':dice,'IoU':iou})  # Append to results list

                fig,ax=plt.subplots(figsize=(10,7),facecolor='white')            # Create overlay figure
                ax.imshow(enhance_rgb_for_display(rgb_base)); ax.axis('off')     # Show base RGB and hide axes
                pred_bound=find_boundaries(pred_label,mode='outer')              # Predicted contour
                if roi_path and os.path.exists(roi_path):                         # If GT exists, overlay GT contour
                    gt_label=read_valid_rois(roi_path,rgb_in.shape[:2])
                    gt_bound=find_boundaries(gt_label,mode='outer')
                    ax.contour(gt_bound,colors='deepskyblue',linewidths=0.9)     # GT contour in blue
                ax.contour(pred_bound,colors='yellow',linewidths=0.9)            # Prediction contour in yellow
                add_scale_bar_um(ax,rgb_in.shape,MICRONS_PER_PIXEL,BAR_MICRONS,SCALEBAR_LOC,12)  # Add scale bar
                ax.set_title(f"{Path(img_path).name} — {vlabel} · {mt}\n(Dice {dice:.3f} | IoU {iou:.3f})",
                             fontsize=12,color='black')                           # Title with metrics
                out_png=OVER_CASES/f"{Path(img_path).stem}__{vlabel}__{mt}.png"  # Output path for overlay PNG
                fig.savefig(out_png,dpi=160,bbox_inches='tight')                 # Save figure
                print(f"[Saved] {out_png}")

                # >>> Show ONE figure at a time (blocking) <<<
                try:
                    plt.show()   # Display until user closes the window (interactive review)
                finally:
                    plt.close(fig)  # Ensure figure is closed to free memory

    # Save metrics
    df=pd.DataFrame(results)                            # Convert results list → DataFrame
    out_csv=Path(out_dir)/"segmentation_metrics_overlay_3ch.csv"  # CSV path
    df.to_csv(out_csv,index=False)                      # Write CSV
    print(f"\nSaved metrics: {out_csv}")
    if not df.empty:
        print(df.to_string(index=False))                # Pretty-print metrics to console

if __name__=="__main__":
    run_pipeline()                                      # Entry point: run interactive pipeline
