#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise2Void per-channel denoising with paper-grade QA plots/metrics and scalebars.

Scalebar:
- Drawn with AnchoredSizeBar in data units (pixels).
- Shown on ALL image panels (RAW, DENOISED, ABS-DIFF, SIGNED-DIFF, BG-DIFF).
- Defaults ensure the bar appears even if no CLI flags are provided.

Run example:
  python noise2v_train.py --pixel-size 0.0322 --bar-length 2 --bar-loc "lower right"
"""

import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from skimage.io import imread as sk_imread
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import threshold_otsu, sobel_h, sobel_v
from skimage.measure import profile_line
from skimage import img_as_ubyte

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(divide="ignore", invalid="ignore")

# ---------- Optional CZI loader ----------
try:
    from czifile import imread as czi_imread
    CZI_OK = True
except Exception:
    CZI_OK = False

def imread_any(path):
    path = str(path); ext = Path(path).suffix.lower()
    if ext == ".czi" and CZI_OK:
        print(f"[READ] CZI image: {Path(path).name}")
        return czi_imread(path)
    print(f"[READ] Generic image: {Path(path).name}")
    return sk_imread(path)

# ---------- helpers ----------
def normalize(x):
    x = np.asarray(x, np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    return np.zeros_like(x) if mx <= mn else (x - mn) / (mx - mn)

def match_shape(a, b):
    return b if a.shape == b.shape else resize(
        b, a.shape, order=1, preserve_range=True, anti_aliasing=True
    ).astype(np.float32)

def psnr_ssim(a, b):
    b = match_shape(a, b)
    a_n, b_n = normalize(a), normalize(b)
    psnr = peak_signal_noise_ratio(a_n, b_n, data_range=1.0)
    H, W = a_n.shape
    win = min(H, W, 7)
    if win % 2 == 0: win -= 1
    ssim = structural_similarity(a_n, b_n, data_range=1.0, win_size=max(3, win))
    return psnr, ssim

def _to_u8_gray_for_qas(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 3:
        if x.shape[-1] in (3, 4):
            x = 0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]
        else:
            x = x[...,0]
    vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
    if vmax <= vmin:
        return np.zeros((256,256), np.uint8)
    x = (x - vmin) / (vmax - vmin)
    if min(x.shape) < 256:
        scale = 256/min(x.shape)
        x = resize(x, (int(x.shape[0]*scale), int(x.shape[1]*scale)),
                   order=1, preserve_range=True, anti_aliasing=True)
    return img_as_ubyte(np.clip(x, 0, 1))

# --- pyBRISQUE shims (Windows libsvm import) ---
try:
    import sys as _sys, libsvm.svmutil as _svm
    _sys.modules.setdefault("svmutil", _svm)
except Exception:
    pass

# NIQE (skimage >=0.21). If missing, returns NaN.
try:
    from skimage.metrics._niqe import niqe as _niqe
    def niqe_score(x):
        try: return float(_niqe(_to_u8_gray_for_qas(x)))
        except Exception: return np.nan
except Exception:
    def niqe_score(x): return np.nan

# BRISQUE (pybrisque variants)
try:
    from brisque import BRISQUE as _PYBRISQUE
    import cv2
    _brisque_model = _PYBRISQUE()
    def brisque_score(x):
        try:
            u8 = _to_u8_gray_for_qas(x)
            bgr = cv2.cvtColor(np.stack([u8,u8,u8], -1), cv2.COLOR_RGB2BGR)
            if hasattr(_brisque_model, "get_score"):
                return float(_brisque_model.get_score(bgr))
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                cv2.imwrite(tmp.name, bgr)
                return float(_brisque_model.score(tmp.name))
        except Exception:
            return np.nan
except Exception:
    def brisque_score(x): return np.nan

# ---------- N2V (optional; cached outputs reused if present) ----------
try:
    from n2v.models import N2V, N2VConfig
    N2V_AVAILABLE = True
except Exception:
    N2V_AVAILABLE = False
    def N2V(*a, **k): raise RuntimeError("Install N2V to denoise, or preload *_denoised_*.npy")
    def N2VConfig(*a, **k): raise RuntimeError("Install N2V")

def train_or_load_n2v(img2d_norm, model_name, basedir):
    path = Path(basedir) / model_name
    ckpt = path / "weights_best.h5"
    if ckpt.exists():
        print(f"   ✓ Using checkpoint: {ckpt}")
        n2v = N2V(config=None, name=model_name, basedir=str(basedir))
        n2v.load_weights(name="weights_best.h5")
        return n2v
    print(f"   → Training new model for {model_name}")
    from numpy.random import default_rng
    rng = default_rng(42)
    H, W = img2d_norm.shape
    patches = [img2d_norm[
                   rng.integers(0, max(1, H-64)):rng.integers(0, max(1, H-64))+64,
                   rng.integers(0, max(1, W-64)):rng.integers(0, max(1, W-64))+64
               ][..., None] for _ in range(512)]
    patches = np.stack(patches).astype(np.float32)
    val_X, tr_X = patches[:50], patches[50:]
    cfg = N2VConfig(
        X=patches, axes='YXC',
        unet_kern_size=3, train_steps_per_epoch=100, train_epochs=20,
        train_loss='mse', batch_norm=True, train_batch_size=16,
        n2v_perc_pix=0.198, n2v_patch_shape=(64, 64),
        train_learning_rate=1e-3, train_reduce_lr={"patience":3, "factor":0.5}
    )
    n2v = N2V(config=cfg, name=model_name, basedir=str(basedir))
    n2v.train(tr_X, val_X, epochs=20, steps_per_epoch=100)
    return n2v

def predict_n2v_in_raw_units(img_raw, n2v):
    mn = float(np.nanmin(img_raw)); mx = float(np.nanmax(img_raw)); rng = max(1e-8, mx - mn)
    img_norm = (img_raw.astype(np.float32) - mn) / rng
    pad = np.pad(img_norm, 8, mode="reflect")
    pred_norm = n2v.predict(pad, axes='YX').astype(np.float32)
    pred_norm = np.clip(pred_norm[8:-8, 8:-8], 0, 1)
    pred_raw = pred_norm * rng + mn
    return pred_raw

# ---------- analysis ----------
def to_raw_scale_triplet(R,G,B, denR,denG,denB):
    mn = float(np.nanmin([R,G,B])); mx = float(np.nanmax([R,G,B])); rng = max(1e-8, mx - mn)
    s = lambda x: np.clip((x.astype(np.float32) - mn) / rng, 0, 1)
    raw_rgb = np.stack([s(R), s(G), s(B)], axis=-1)
    den_rgb = np.stack([s(match_shape(R, denR)), s(match_shape(G, denG)), s(match_shape(B, denB))], axis=-1)
    raw_luma = 0.2126*raw_rgb[...,0] + 0.7152*raw_rgb[...,1] + 0.0722*raw_rgb[...,2]
    den_luma = 0.2126*den_rgb[...,0] + 0.7152*den_rgb[...,1] + 0.0722*den_rgb[...,2]
    return raw_rgb, den_rgb, raw_luma, den_luma

def fixed_masks_from_raw_luma(raw_luma):
    t = threshold_otsu(raw_luma); bg = raw_luma < t; fg = ~bg
    return fg, bg

def residuals_luma(raw_rgb, den_rgb):
    luma = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    diff_rgb = den_rgb - raw_rgb
    signed = (diff_rgb * luma).sum(axis=-1)
    absdiff = np.abs(signed)
    return signed, absdiff

def radial_power_spectrum(img2d):
    f = np.fft.fftshift(np.fft.fft2(img2d.astype(np.float32)))
    p = np.abs(f)**2
    h, w = p.shape
    cy, cx = h/2.0, w/2.0
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(np.int32)
    tbin = np.bincount(r.ravel(), p.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(1, nr)

def auto_line_profile(raw_luma, den_luma, length=160):
    gx = sobel_h(raw_luma); gy = sobel_v(raw_luma)
    mag = np.hypot(gx, gy)
    y0, x0 = np.unravel_index(np.argmax(mag), mag.shape)
    vx, vy = gx[y0, x0], gy[y0, x0]
    if vx == 0 and vy == 0: vx, vy = 1.0, 0.0
    norm = np.hypot(vx, vy); vx, vy = vx/norm, vy/norm
    half = length // 2
    x1, y1 = float(x0 - vx*half), float(y0 - vy*half)
    x2, y2 = float(x0 + vx*half), float(y0 + vy*half)
    raw_prof = profile_line(raw_luma, (y1, x1), (y2, x2), mode='reflect')
    den_prof = profile_line(den_luma, (y1, x1), (y2, x2), mode='reflect')
    return (x1,y1,x2,y2), raw_prof, den_prof

def fwhm_1d(y):
    y = y.astype(np.float32)
    yb = y - np.min(y)
    if np.max(yb) <= 0: return np.nan
    peak = int(np.argmax(yb))
    half = 0.5 * float(np.max(yb))
    x = np.arange(len(yb), dtype=np.float32)  # FIXED dtype
    left_idx = np.where(yb[:peak] <= half)[0]
    right_idx = np.where(yb[peak:] <= half)[0]
    if len(left_idx) == 0 or len(right_idx) == 0: return np.nan
    li = left_idx[-1]; l2 = min(li+1, peak)
    ri = right_idx[0] + peak; r1 = max(ri-1, peak)
    xL = np.interp(half, [yb[li], yb[l2]], [x[li], x[l2]])
    xR = np.interp(half, [yb[r1], yb[ri]], [x[r1], x[ri]])
    return float(xR - xL)

# ---------- SCALE BAR (exact style) ----------
def _loc_to_id(loc_str):
    m = {"upper right":1, "upper left":2, "lower left":3, "lower right":4}
    return m.get(str(loc_str).lower(), 4)

def add_scale_bar_um(ax, shape, um_per_px, bar_um, loc="lower right"):
    """
    Exact style as your reference:
    white bar, AnchoredSizeBar in data units, label 'X µm',
    thickness tied to image height.
    """
    if um_per_px is None or bar_um is None or um_per_px <= 0 or bar_um <= 0:
        return
    H, W = shape[:2]
    bar_px = bar_um / um_per_px
    asb = AnchoredSizeBar(
        ax.transData, bar_px, label=f"{bar_um:g} µm",
        loc=_loc_to_id(loc), pad=0.4, sep=4, borderpad=0.8,
        frameon=False, size_vertical=max(2, int(0.006*H)), color='white'
    )
    ax.add_artist(asb)

# ---------- CSV writer ----------
def _safe_write_csv(df: pd.DataFrame, target: Path):
    try:
        df.to_csv(target, index=False); return target
    except PermissionError:
        base = target.with_suffix("")
        for k in range(1,999):
            alt = Path(f"{base}__{k}").with_suffix(".csv")
            try: df.to_csv(alt, index=False); print(f"[WARN] CSV locked: wrote {alt}"); return alt
            except PermissionError: continue
        raise

# ---------- main pipeline ----------
def run_pipeline(in_dir, out_dir, files_explicit, force=False, model_dir="./models",
                 pixel_size=0.0322, bar_length=2.0, bar_loc="lower right"):
    """
    pixel_size (µm/px) and bar_length (µm) have defaults so the scalebar always appears.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"; overlay_dir.mkdir(exist_ok=True)

    rows = []
    files = [str(f) for f in files_explicit]

    for i, f in enumerate(files, 1):
        p = Path(f)
        print(f"\n[{i}/{len(files)}] Processing {p.name}")
        arr = np.squeeze(imread_any(p))

        # Basic channel handling
        if arr.ndim == 5:
            arr = arr[0]
            arr = np.moveaxis(arr, 0, 1) if arr.shape[1] < arr.shape[0] else arr
            arr = arr[:, 0]
        elif arr.ndim == 4:
            arr = arr[:, 0]
        elif arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > 8:
            arr = np.moveaxis(arr, 0, -1)

        if arr.ndim == 3 and arr.shape[-1] >= 3:
            R,G,B = arr[...,0], arr[...,1], arr[...,2]
        else:
            R=G=B=arr

        # Denoise or load cached
        fR,fG,fB = [out_dir/f"{p.stem}_denoised_{c}.npy" for c in ("R","G","B")]
        denR=denG=denB=None
        need = force or not all(x.exists() for x in (fR,fG,fB))
        if not need:
            denR, denG, denB = np.load(fR), np.load(fG), np.load(fB)
        else:
            if not N2V_AVAILABLE:
                raise RuntimeError("N2V not available; cannot recompute denoised outputs.")
            print("   → Denoising channels in RAW units")
            for ch, img, tag in (("R",R,"_R"),("G",G,"_G"),("B",B,"_B")):
                n2v = train_or_load_n2v(normalize(img), f"N2V_{i}{tag}", model_dir)
                pred = predict_n2v_in_raw_units(img, n2v)
                if ch=="R": denR=pred
                elif ch=="G": denG=pred
                else: denB=pred
            np.save(fR, denR); np.save(fG, denG); np.save(fB, denB)

        # RAW-scale composites
        raw_rgb, den_rgb, raw_luma, den_luma = to_raw_scale_triplet(R,G,B, denR,denG,denB)
        fg_mask, bg_mask = fixed_masks_from_raw_luma(raw_luma)

        # Metrics
        psnrR,ssimR = psnr_ssim(R,denR); psnrG,ssimG = psnr_ssim(G,denG); psnrB,ssimB = psnr_ssim(B,denB)
        brisque_raw = float(np.mean([brisque_score(R), brisque_score(G), brisque_score(B)]))
        brisque_den = float(np.mean([brisque_score(denR), brisque_score(denG), brisque_score(denB)]))
        delta_brisque = brisque_den - brisque_raw
        niqe_raw = float(np.mean([niqe_score(R), niqe_score(G), niqe_score(B)]))
        niqe_den = float(np.mean([niqe_score(denR), niqe_score(denG), niqe_score(denB)]))
        delta_niqe = niqe_den - niqe_raw

        # Residuals / noise / CNR
        signed_res, abs_res = residuals_luma(raw_rgb, den_rgb)
        sigma_bg_raw = float(np.std(raw_luma[bg_mask])); sigma_bg_den = float(np.std(den_luma[bg_mask]))
        delta_sigma_bg_pct = 100.0*(sigma_bg_den - sigma_bg_raw) / (sigma_bg_raw + 1e-8)
        p95_raw = float(np.percentile(raw_luma[fg_mask], 95))
        p95_den = float(np.percentile(den_luma[fg_mask], 95)); alpha = p95_raw / max(1e-8, p95_den)
        den_luma_gm = np.clip(den_luma*alpha, 0, 1)
        sigma_bg_den_gm = float(np.std(den_luma_gm[bg_mask]))
        cnr_raw = (np.median(raw_luma[fg_mask]) - np.median(raw_luma[bg_mask])) / (sigma_bg_raw + 1e-8)
        cnr_den_gm = (np.median(den_luma_gm[fg_mask]) - np.median(den_luma_gm[bg_mask])) / (sigma_bg_den_gm + 1e-8)
        delta_cnr_gm = float(cnr_den_gm - cnr_raw)

        # Edge profile / FWHM
        (x1,y1,x2,y2), raw_prof, den_prof = auto_line_profile(raw_luma, den_luma, length=160)
        fwhm_raw = fwhm_1d(raw_prof); fwhm_den = fwhm_1d(den_prof)
        delta_fwhm = (fwhm_den - fwhm_raw) if (np.isfinite(fwhm_raw) and np.isfinite(fwhm_den)) else np.nan

        # ---------- Figures (all with scalebar) ----------
        caption = "Foreground from RAW luma via Otsu; all displays on RAW min–max."
        um_per_px = pixel_size; bar_um = bar_length

        # (1) RAW vs DENOISED
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].imshow(raw_rgb); ax[0].set_title("RAW (RAW-scale)"); ax[0].axis("off")
        add_scale_bar_um(ax[0], raw_rgb.shape, um_per_px, bar_um, bar_loc)
        ax[1].imshow(den_rgb); ax[1].set_title("DENOISED (RAW-scale)"); ax[1].axis("off")
        add_scale_bar_um(ax[1], den_rgb.shape, um_per_px, bar_um, bar_loc)
        fig.text(0.01, 0.01, caption, fontsize=8)
        plt.tight_layout(rect=[0,0.03,1,1]); fig.savefig(overlay_dir/f"{p.stem}_overlay.png", dpi=300); plt.close(fig)

        # (2) ABS-DIFF (single)
        vmax_abs = float(np.clip(np.percentile(abs_res, 99.5), 1e-6, 1.0))
        fig = plt.figure(figsize=(5.6,5.2)); ax = fig.add_subplot(1,1,1)
        im = ax.imshow(abs_res, cmap='inferno', vmin=0, vmax=vmax_abs)
        ax.set_title("ABSOLUTE DIFFERENCE (luma, RAW-scale)"); ax.axis("off")
        add_scale_bar_um(ax, abs_res.shape, um_per_px, bar_um, bar_loc)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("Abs Δ (0–1)")
        fig.text(0.01, 0.01, caption, fontsize=8)
        plt.tight_layout(); plt.savefig(overlay_dir/f"{p.stem}_difference.png", dpi=300); plt.close(fig)

        # (3) SIGNED-DIFF (single)
        amax = float(np.nanpercentile(np.abs(signed_res), 99.5)); lim = max(1e-6, amax)
        fig = plt.figure(figsize=(5.6,5.2)); ax = fig.add_subplot(1,1,1)
        im = ax.imshow(signed_res, cmap='seismic', vmin=-lim, vmax=+lim)
        ax.set_title("SIGNED DIFFERENCE (denoised − raw), luma"); ax.axis("off")
        add_scale_bar_um(ax, signed_res.shape, um_per_px, bar_um, bar_loc)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("Δ (− to +)")
        fig.text(0.01, 0.01, caption, fontsize=8)
        plt.tight_layout(); plt.savefig(overlay_dir/f"{p.stem}_difference_signed.png", dpi=300); plt.close(fig)

        # (4) BG-only residual
        bg_only = abs_res.copy(); bg_only[fg_mask] = np.nan
        vmax_bg = float(np.nanpercentile(bg_only, 99.5))
        fig = plt.figure(figsize=(5.6,5.2)); ax = fig.add_subplot(1,1,1)
        im = ax.imshow(bg_only, cmap='inferno', vmin=0, vmax=np.clip(vmax_bg, 1e-6, 1.0))
        ax.set_title("BACKGROUND DIFFERENCE (abs, luma) — noise removal"); ax.axis("off")
        add_scale_bar_um(ax, bg_only.shape, um_per_px, bar_um, bar_loc)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04); cbar.set_label("Abs Δ (0–1)")
        txt = (f"σ_bg RAW: {sigma_bg_raw:.4f}\n"
               f"σ_bg DEN: {sigma_bg_den:.4f}\n"
               f"Δσ_bg: {delta_sigma_bg_pct:.2f}%\n"
               f"ΔCNR (gm): {delta_cnr_gm:.3f}\n"
               f"ΔFWHM: {delta_fwhm:.2f}px\n"
               f"ΔBRISQUE: {delta_brisque:.2f}\n"
               f"ΔNIQE: {delta_niqe if np.isfinite(delta_niqe) else np.nan}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='black', alpha=0.55, pad=6), color='w', fontsize=9)
        fig.text(0.01, 0.01, caption, fontsize=8)
        plt.tight_layout(); plt.savefig(overlay_dir/f"{p.stem}_difference_background.png", dpi=300); plt.close(fig)

        # (5) Triptych with metrics; colorbar only on ABS-DIFF
        fig, ax = plt.subplots(1,3, figsize=(14,5))
        ax[0].imshow(raw_rgb); ax[0].set_title("RAW (RAW-scale)"); ax[0].axis("off")
        add_scale_bar_um(ax[0], raw_rgb.shape, um_per_px, bar_um, bar_loc)
        ax[1].imshow(den_rgb); ax[1].set_title("DENOISED (RAW-scale)"); ax[1].axis("off")
        add_scale_bar_um(ax[1], den_rgb.shape, um_per_px, bar_um, bar_loc)
        im = ax[2].imshow(abs_res, cmap='inferno', vmin=0, vmax=vmax_abs)
        ax[2].set_title("ABS-DIFF (luma)"); ax[2].axis("off")
        cbar = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04); cbar.set_label("Abs Δ (0–1)")
        txt = (f"Δσ_bg: {delta_sigma_bg_pct:.2f}%\n"
               f"ΔCNR (gm): {delta_cnr_gm:.3f}\n"
               f"ΔFWHM: {delta_fwhm:.2f}px\n"
               f"ΔBRISQUE: {delta_brisque:.2f}\n"
               f"ΔNIQE: {delta_niqe if np.isfinite(delta_niqe) else np.nan}")
        ax[1].text(0.02, 0.98, txt, transform=ax[1].transAxes, va='top', ha='left',
                   bbox=dict(facecolor='black', alpha=0.55, pad=6), color='w', fontsize=9)
        fig.text(0.01, 0.01, caption, fontsize=8)
        plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig(overlay_dir/f"{p.stem}_overlay_diff.png", dpi=300); plt.close(fig)

        # (6) Line profile
        fig = plt.figure(figsize=(7,4.2)); ax = fig.add_subplot(1,1,1)
        ax.plot(raw_prof, label="RAW (RAW-scale)"); ax.plot(den_prof, label="DENOISED (RAW-scale)")
        ax.set_title(f"Edge line profile (auto)  ΔFWHM={delta_fwhm:.2f}px")
        ax.set_xlabel("Distance (px)"); ax.set_ylabel("Intensity (0–1)"); ax.legend()
        plt.tight_layout(); plt.savefig(overlay_dir/f"{p.stem}_line_profile.png", dpi=300); plt.close(fig)

        # (7) Radial power spectrum
        raw_ps = radial_power_spectrum(raw_luma + 1e-8); den_ps = radial_power_spectrum(den_luma + 1e-8)
        r = np.arange(min(len(raw_ps), len(den_ps)))
        fig = plt.figure(figsize=(6.6,4.6)); ax = fig.add_subplot(1,1,1)
        ax.plot(r, np.log10(raw_ps[:len(r)]), label="RAW"); ax.plot(r, np.log10(den_ps[:len(r)]), label="DENOISED")
        ax.set_title("Radial power spectrum (log scale)"); ax.set_xlabel("Spatial frequency (radius)"); ax.set_ylabel("log10 Power")
        ax.legend(); plt.tight_layout(); plt.savefig(overlay_dir/f"{p.stem}_radial_power.png", dpi=300); plt.close(fig)

        # Row to CSV
        rows.append({
            "Image": p.name,
            "PSNR_R": psnrR, "SSIM_R": ssimR,
            "PSNR_G": psnrG, "SSIM_G": ssimG,
            "PSNR_B": psnrB, "SSIM_B": ssimB,
            "BRISQUE_raw": brisque_raw, "BRISQUE_den": brisque_den, "delta_brisque": delta_brisque,
            "NIQE_raw": niqe_raw, "NIQE_den": niqe_den, "delta_niqe": delta_niqe,
            "sigma_bg_raw": sigma_bg_raw, "sigma_bg_den": sigma_bg_den, "delta_sigma_bg_pct": float(delta_sigma_bg_pct),
            "cnr_raw": float(cnr_raw), "cnr_den_gm": float(cnr_den_gm), "delta_cnr_gm": float(delta_cnr_gm),
            "fwhm_raw_px": float(fwhm_raw) if np.isfinite(fwhm_raw) else np.nan,
            "fwhm_den_px": float(fwhm_den) if np.isfinite(fwhm_den) else np.nan,
            "delta_fwhm_px": float(delta_fwhm) if np.isfinite(delta_fwhm) else np.nan,
        })

    # Save CSV + quick summaries
    df = pd.DataFrame(rows)
    csv_path = _safe_write_csv(df, out_dir/"psnr_ssim_3ch.csv")

    if len(df) > 0:
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].boxplot([df["PSNR_R"],df["PSNR_G"],df["PSNR_B"]], tick_labels=["R","G","B"]); ax[0].set_title("PSNR (Denoised vs Raw)")
        ax[1].boxplot([df["SSIM_R"],df["SSIM_G"],df["SSIM_B"]], tick_labels=["R","G","B"]); ax[1].set_title("SSIM (Denoised vs Raw)")
        plt.tight_layout(); plt.savefig(out_dir/"metrics_summary.png", dpi=300); plt.close(fig)

    if len(df) >= 3:
        fig, ax = plt.subplots(1,5, figsize=(14,4))
        def bx(a, series, title, ref=0.0):
            a.boxplot([series.dropna()], tick_labels=["1"]); a.set_title(title); a.axhline(ref, ls="--", lw=1, color="tab:blue")
        bx(ax[0], df["delta_brisque"], "ΔBRISQUE (DEN−RAW)\n<0 better", 0)
        bx(ax[1], df["delta_niqe"],    "ΔNIQE (DEN−RAW)\n<0 better", 0)
        bx(ax[2], df["delta_sigma_bg_pct"], "Δσ_bg (%)\n<0 less noise", 0)
        bx(ax[3], df["delta_cnr_gm"],  "ΔCNR (gm)\n>0 better", 0)
        bx(ax[4], df["delta_fwhm_px"], "ΔFWHM (px)\n≈0 good", 0)
        plt.tight_layout(); plt.savefig(out_dir/"summary_deltas.png", dpi=300); plt.close(fig)

        tbl = pd.DataFrame({
            "Metric": ["ΔBRISQUE", "ΔNIQE", "Δσ_bg (%)", "ΔCNR (gm)", "ΔFWHM (px)"],
            "Mean ± SD": [
                f"{df['delta_brisque'].mean():.2f} ± {df['delta_brisque'].std(ddof=1):.2f}",
                f"{df['delta_niqe'].mean():.2f} ± {df['delta_niqe'].std(ddof=1):.2f}",
                f"{df['delta_sigma_bg_pct'].mean():.2f} ± {df['delta_sigma_bg_pct'].std(ddof=1):.2f}",
                f"{df['delta_cnr_gm'].mean():.3f} ± {df['delta_cnr_gm'].std(ddof=1):.3f}",
                f"{df['delta_fwhm_px'].mean():.3f} ± {df['delta_fwhm_px'].std(ddof=1):.3f}",
            ],
            "N": [df['delta_brisque'].dropna().shape[0],
                  df['delta_niqe'].dropna().shape[0],
                  df['delta_sigma_bg_pct'].dropna().shape[0],
                  df['delta_cnr_gm'].dropna().shape[0],
                  df['delta_fwhm_px'].dropna().shape[0]]
        })
        fig,ax=plt.subplots(figsize=(6.5,1.2+0.35*len(tbl))); ax.axis('off')
        t=ax.table(cellText=tbl.values, colLabels=tbl.columns, loc='center')
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1,1.2)
        plt.tight_layout(); plt.savefig(out_dir/"summary_table_deltas.png", dpi=300); plt.close(fig)

    print(f"\n✅ Metrics saved to: {csv_path}")
    print(f"✅ Overlays saved in: {overlay_dir}")
    print(f"✅ Summary plots saved to: {out_dir}")

# ---------- simple file pickers ----------
def select_image_files():
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    files=filedialog.askopenfilenames(
        title="Select input images (.czi/.tif/.jpg/.jpeg/.png)",
        filetypes=[("Microscopy/Images","*.czi *.tif *.tiff *.jpg *.jpeg *.png")])
    root.destroy(); return list(files)

def select_outdir_popup(default="./results"):
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    out=filedialog.askdirectory(title="Select output folder")
    root.destroy(); return out or default

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(description="N2V per-channel denoising with scalebars + QA visuals")
    ap.add_argument("--force", action="store_true", help="Retrain/redo even if caches exist")
    ap.add_argument("--models", type=str, default="./models", help="Directory for N2V models")
    ap.add_argument("--pixel-size", type=float, default=0.0322, help="Pixel size in µm/px (default 0.0322)")
    ap.add_argument("--bar-length", type=float, default=2.0, help="Scale bar length in µm (default 2)")
    ap.add_argument("--bar-loc", type=str, default="lower right", help="Scale bar location")
    args=ap.parse_args()

    imgs = select_image_files()
    if not imgs: sys.exit("✖ No images selected.")
    outdir = select_outdir_popup("./final results")

    run_pipeline(str(Path(imgs[0]).parent), outdir, imgs, args.force, args.models,
                 pixel_size=args.pixel_size, bar_length=args.bar_length, bar_loc=args.bar_loc)
