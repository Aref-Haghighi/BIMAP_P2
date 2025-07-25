import numpy as np
import matplotlib.pyplot as plt
from czifile import imread
from n2v.models import N2V, N2VConfig
from cellpose import models
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
from matplotlib.patches import Rectangle
import os
import pandas as pd

MICRONS_PER_PIXEL = 0.0322
BAR_MICRONS = 2
MODEL_TYPES = ['cyto', 'cyto2']
base_model_dir = './models'

image_files = [
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI3_SIM.czi",
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI2_SIM.czi",
    "2D_WT_NADA_RADA_HADA_THY_40min_ROI1_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI2_SIM.czi",
    "WT_NADA_RADA_HADA_NHS_40min_ROI3_SIM.czi"
]
roi_files = [
    "RoiSet_2D_WT_NADA_THY3.zip",
    "RoiSet_2D_WT_NADA_THY2.zip",
    "RoiSet_2D_WT_NADA_THY1.zip",
    "RoiSet_Contour_bacteria_ROI1.zip",
    "RoiSet_Contour_bacteria_ROI2.zip",
    "RoiSet_Contour_bacteria_ROI3.zip"
]

def add_scale_bar(ax, image_shape, microns_per_pixel, bar_microns=2, width_frac=0.013,
                  position=(0.07, 0.08), color='white', font_size=14):
    img_width_px = image_shape[1]
    bar_pixels = bar_microns / microns_per_pixel
    bar_length_frac = bar_pixels / img_width_px
    ax.add_patch(
        Rectangle(
            position, bar_length_frac, width_frac,
            color=color, transform=ax.transAxes, clip_on=False
        )
    )
    ax.text(
        position[0] + bar_length_frac/2,
        position[1] + width_frac*2.1,
        f"{bar_microns} μm",
        color=color, fontsize=font_size,
        ha='center', va='bottom', transform=ax.transAxes, clip_on=False
    )

def normalize(channel):
    return (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
def auto_contrast(img_rgb, low_pct=1, high_pct=99, gamma=1.0):
    img = img_rgb.copy()
    flat = img.reshape(-1, 3) if img.ndim == 3 else img.flatten()
    p_low = np.percentile(flat, low_pct)
    p_high = np.percentile(flat, high_pct)
    img = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
    img = exposure.adjust_gamma(img, gamma)
    return img
def read_valid_rois(zip_path, shape):
    try:
        rois = read_roi_zip(zip_path)
    except Exception as e:
        print(f"\u274c Error reading ROI zip {zip_path}: {e}")
        return np.zeros(shape, dtype=np.uint16)
    mask = np.zeros(shape, dtype=np.uint16)
    for i, roi in enumerate(rois.values(), 1):
        if 'x' not in roi or 'y' not in roi or len(roi['x']) < 3:
            continue
        try:
            y, x = roi['y'], roi['x']
            poly_mask = polygon2mask(shape, np.column_stack((y, x)))
            mask[poly_mask] = i
        except Exception:
            continue
    mask = remove_small_objects(mask, min_size=5)
    return sk_label(mask > 0)
def dice_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    return 2.0 * intersection / (gt.sum() + pred.sum() + 1e-8)
def iou_score(gt, pred):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return intersection / (union + 1e-8)
def debug_filter(lbl, min_area=10, min_solidity=0.3):
    props = regionprops(lbl)
    out = np.zeros_like(lbl)
    count = 1
    for r in props:
        if r.area < min_area:
            continue
        if r.solidity < min_solidity:
            continue
        out[lbl == r.label] = count
        count += 1
    return out
def split_touching_cells(mask, min_distance=10):
    proc_mask = mask > 0
    distance = ndi.distance_transform_edt(proc_mask)
    distance = gaussian(distance, sigma=0.7)
    coords = peak_local_max(distance, min_distance=min_distance, labels=proc_mask)
    mask_peaks = np.zeros_like(distance, dtype=bool)
    if coords.shape[0] > 0:
        for y, x in coords:
            mask_peaks[y, x] = True
    if not np.any(mask_peaks):
        max_dist = np.unravel_index(np.argmax(distance), distance.shape)
        mask_peaks[max_dist] = True
    markers, _ = ndi.label(mask_peaks)
    labels_ws = watershed(-distance, markers, mask=proc_mask)
    labels_ws = remove_small_objects(labels_ws, min_size=10)
    labels_ws = debug_filter(labels_ws)
    return labels_ws

# === STEP 1: Train and denoise per image ===
patch_size = (64, 64)
n_patches = 512

psnr_ssim_results = []  # To store PSNR/SSIM results

for idx, img_path in enumerate(image_files):
    denoised_path = img_path.replace('.czi', '_denoised.npy')
    raw = imread(img_path)
    img_stack = np.squeeze(raw)[0:3, :, :]
    green = img_stack[1]
    green_norm = normalize(green)

    if os.path.exists(denoised_path):
        print(f"[SKIP] {denoised_path} exists. Skipping N2V training and prediction.")
        denoised_green = np.load(denoised_path)
    else:
        model_name = f'N2V_MODEL_{idx+1}'
        model_dir = os.path.join(base_model_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print(f"\n[TRAINING N2V] Image: {img_path}")
        img = green_norm
        # Patch extraction
        patches = []
        np.random.seed(42)
        h, w = img.shape
        for _ in range(n_patches):
            top = np.random.randint(0, h - patch_size[0])
            left = np.random.randint(0, w - patch_size[1])
            patch = img[top:top+patch_size[0], left:left+patch_size[1]]
            patches.append(patch)
        patches = np.stack(patches, axis=0)[..., np.newaxis]
        val_split = 0.1
        n_val = int(len(patches) * val_split)
        validation_X = patches[:n_val]
        train_X = patches[n_val:]
        # Config and train
        config = N2VConfig(
            X=patches,
            unet_kern_size=3,
            train_steps_per_epoch=100,
            train_epochs=20,
            train_loss='mse',
            batch_norm=True,
            train_batch_size=16,
            n2v_perc_pix=0.198,
            n2v_patch_shape=patch_size,
            train_learning_rate=1e-3,
            train_reduce_lr={'patience': 3, 'factor': 0.5},
            train_epochs_per_step=1
        )
        n2v = N2V(config=config, name=model_name, basedir=base_model_dir)
        n2v.train(train_X, validation_X, epochs=20, steps_per_epoch=100)
        print(f"[TRAINED] Saved at {model_dir}")

        # Denoise and save denoised image
        denoised_green = n2v.predict(img, axes='YX')
        np.save(denoised_path, denoised_green)
        print(f"[DENOISED] Saved as {denoised_path}")

    denoised_green_norm = normalize(denoised_green)
    # Calculate PSNR and SSIM between original and denoised green channel
    psnr = peak_signal_noise_ratio(green_norm, denoised_green_norm, data_range=1.0)
    ssim = structural_similarity(green_norm, denoised_green_norm, data_range=1.0)
    psnr_ssim_results.append({
        'Image': os.path.basename(img_path),
        'PSNR': psnr,
        'SSIM': ssim
    })

# Show PSNR/SSIM summary
df_psnr_ssim = pd.DataFrame(psnr_ssim_results)
print("\n====== PSNR and SSIM for each image (green channel, original vs. denoised) ======")
print(df_psnr_ssim.to_string(index=False))

# === STEP 2: Segment all images (original/denoised) ===
results = []
for idx, (img_path, roi_path) in enumerate(zip(image_files, roi_files)):
    print(f"\n[SEGMENTATION] Processing: {img_path}")
    raw = imread(img_path)
    img_stack = np.squeeze(raw)[0:3, :, :]
    green = img_stack[1]
    green = normalize(green)
    denoised_path = img_path.replace('.czi', '_denoised.npy')
    denoised_green = np.load(denoised_path)
    denoised_green = np.clip(denoised_green, 0, None)
    denoised_green = normalize(denoised_green)

    models_dict = {model_type: models.CellposeModel(gpu=True, model_type=model_type) for model_type in MODEL_TYPES}
    for version_label, input_img in zip(['Original', 'Denoised'], [green, denoised_green]):
        rgb = np.stack([normalize(img_stack[2]), normalize(input_img), normalize(img_stack[0])], axis=-1)
        rgb_auto = auto_contrast(rgb, low_pct=1, high_pct=99, gamma=1.0)
        rgb_uint8 = img_as_ubyte(rgb_auto)
        best_dice = -1
        best_iou = -1
        best_model = None
        best_pred_mask = None
        for model_type in MODEL_TYPES:
            pred_mask, _, _ = models_dict[model_type].eval(rgb_uint8, diameter=None, channels=[0, 1])
            pred_label = split_touching_cells(pred_mask)
            n_pred_cells = np.max(pred_label)
            if roi_path is not None:
                gt_label = read_valid_rois(roi_path, rgb.shape[:2])
                dice = dice_score(gt_label > 0, (pred_label > 0) & (gt_label > 0))
                iou = iou_score(gt_label > 0, (pred_label > 0) & (gt_label > 0))
            else:
                dice, iou = np.nan, np.nan
            results.append({
                'Image': os.path.basename(img_path),
                'ROI': os.path.basename(roi_path) if roi_path else None,
                'Type': version_label,
                'Model': model_type,
                'Dice': dice,
                'IoU': iou
            })
            if dice > best_dice:
                best_dice = dice
                best_iou = iou
                best_model = model_type
                best_pred_mask = pred_mask

        bar_position = (0.88, 0.08) if img_path.startswith("WT_NADA_RADA_HADA_NHS_40min_ROI") else (0.07, 0.08)
        low_brightness = exposure.adjust_gamma(rgb_auto, gamma=2.0)
        pred_label = split_touching_cells(best_pred_mask)
        pred_bound = find_boundaries(pred_label, mode='outer')
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        axes[0].imshow(low_brightness)
        axes[0].contour(pred_bound, colors='yellow', linewidths=0.9)
        axes[0].set_title(f"(Cells={np.max(pred_label)})\nBest Model: {best_model}\nDice: {best_dice:.3f}, IoU: {best_iou:.3f}")
        axes[0].axis('off')
        add_scale_bar(axes[0], rgb_auto.shape, MICRONS_PER_PIXEL, bar_microns=BAR_MICRONS,
                      width_frac=0.013, position=bar_position, color='white', font_size=14)
        if roi_path is not None:
            gt_label = read_valid_rois(roi_path, rgb.shape[:2])
            gt_bound = find_boundaries(gt_label, mode='outer')
            n_gt_cells = np.max(gt_label)
            axes[1].imshow(low_brightness)
            axes[1].contour(gt_bound, colors='deepskyblue', linewidths=0.9)
            axes[1].contour(pred_bound, colors='yellow', linewidths=0.9)
            axes[1].set_title(f"GT={n_gt_cells} | Seg={np.max(pred_label)}")
            axes[1].axis('off')
        else:
            axes[1].imshow(low_brightness)
            axes[1].contour(pred_bound, colors='yellow', linewidths=0.9)
            axes[1].set_title("Segmented Only")
            axes[1].axis('off')
        add_scale_bar(axes[1], rgb_auto.shape, MICRONS_PER_PIXEL, bar_microns=BAR_MICRONS,
                      width_frac=0.013, position=bar_position, color='white', font_size=14)
        plt.suptitle(f"{os.path.basename(img_path)} ({version_label}) - Best: {best_model}", fontsize=14)
        plt.tight_layout()
        plt.show()

# ------- Print full metrics summary as table -------
df = pd.DataFrame(results)
print("\n====== All segmentation metrics (each image, model, and type) ======")
print(df.to_string(index=False))
