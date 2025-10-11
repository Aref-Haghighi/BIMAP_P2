#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Audit & visualize U-Net training (LOIO) runs.

Automatically uses:
    C:\Users\arefm\Desktop\Bacteria\experiments\unet_cv

Prompts user to select where to save audit results, then:
  • Parses train_log.txt in each fold (train_loss, valDice, etc.)
  • Checks for missing checkpoints or anomalies
  • Produces per-fold and grid learning curves
  • Saves results CSV, readable errors list, and plots
  • Opens the output folder automatically after finishing

Requirements:
    pip install pandas matplotlib scipy openpyxl
"""

# --- Standard library imports ---
import math, re, os, subprocess                # math for grid layout, re for log parsing, os/subprocess for system ops
from pathlib import Path                       # pathlib for robust path handling
from typing import Dict, List, Optional, Tuple # type hints for readability

# --- GUI for selecting output folder ---
import tkinter as tk
from tkinter import filedialog

# --- Numerics / data / plotting ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
# Root directory containing per-mode subfolders (e.g., raw/, n2v/), each with fold_* subfolders.
CKROOT = Path(r"C:\Users\arefm\Desktop\Bacteria\experiments\unet_cv")

# Ask user for output directory using a simple Tk dialog (no visible main window)
root = tk.Tk(); root.withdraw()
OUTDIR = Path(filedialog.askdirectory(title="Select output folder to save audit results"))
if not OUTDIR:
    # Abort if nothing selected to avoid writing to an unintended location
    raise SystemExit(" No output folder selected. Exiting.")
OUTDIR.mkdir(parents=True, exist_ok=True)              # Ensure output folder exists
LCDIR = OUTDIR / "learning_curves"; LCDIR.mkdir(parents=True, exist_ok=True)  # Folder for generated curve plots

# Minimum number of epochs expected per fold; used for sanity checks
MIN_EPOCHS = 15
# Plotting resolution for all saved figures
DPI = 150

# ---------------- Helpers ----------------
def savefig(path: Path):
    """
    Save current Matplotlib figure to 'path' with consistent DPI and tight layout,
    then close the figure to free memory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def parse_train_log(path: Path) -> Optional[pd.DataFrame]:
    """
    Parse a fold's train_log.txt into a tidy DataFrame with columns:
      epoch, train_loss, valDice, best_so_far
    Returns None if file is missing or no lines matched the expected pattern.
    """
    if not path.exists():
        return None
    rows = []
    # Regex captures: epoch number, training loss, valDice (overlap-only), and "best so far" valDice.
    pat = re.compile(
        r"epoch\s+(\d+)\s*\|\s*loss\s*([0-9.]+)\s*\|\s*valDice\(overlap-only\)\s*([0-9.]+)\s*\|\s*best\s*([0-9.]+)",
        re.IGNORECASE
    )
    # Open with tolerant encoding/error handling in case of non-ASCII characters
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                # Append the parsed tuple as numeric types
                rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4])))
    if not rows:
        return None
    # Build DataFrame and sort by epoch to ensure monotonic order
    df = pd.DataFrame(rows, columns=["epoch", "train_loss", "valDice", "best_so_far"])
    return df.sort_values("epoch").reset_index(drop=True)

def slope_last_k(x, y, k=10):
    """
    Compute the linear slope over the last k points of a learning curve.
    Returns NaN if not enough points are available.
    """
    if len(x) < 2: return np.nan
    xk, yk = x[-k:], y[-k:]
    if len(xk) < 2: return np.nan
    # Polyfit degree 1 => slope; cast to float for JSON/CSV friendliness
    return float(np.polyfit(xk, yk, 1)[0])

def detect_warnings(df: pd.DataFrame, min_epochs: int) -> List[str]:
    """
    Heuristics to detect training anomalies:
      - too few epochs
      - NaN values in loss/Dice
      - non-increasing epoch sequence (log corruption)
      - large jumps in validation Dice
      - validation Dice near zero at the end
      - plateau over the last 10 epochs
      - training loss not decreasing overall
    Returns a list of warning strings.
    """
    w = []
    if len(df) < min_epochs: w.append(f"few_epochs:{len(df)}")
    if df[["train_loss", "valDice"]].isna().any().any(): w.append("nan_values")
    if not np.all(np.diff(df["epoch"].values) > 0): w.append("non_monotonic_epochs")
    diffs = np.abs(np.diff(df["valDice"].values))
    if len(diffs) and np.nanmax(diffs) > 0.25: w.append(f"valDice_big_jump:{np.nanmax(diffs):.2f}")
    if len(df) >= 5 and np.nanmean(df["valDice"].values[-5:]) < 0.05: w.append("valDice_near_zero_end")
    sl = slope_last_k(df["epoch"].values, df["valDice"].values, 10)
    if not np.isnan(sl) and abs(sl) < 1e-3: w.append("valDice_plateau_last10")
    if len(df) >= 6:
        a, b = df["train_loss"].values[:3], df["train_loss"].values[-3:]
        if np.nanmedian(b) > np.nanmedian(a) - 1e-3: w.append("loss_not_decreasing")
    return w

def plot_learning_curve(df, title, outpath):
    """
    Plot a single fold's learning curve:
      - train_loss on left y-axis
      - valDice on right y-axis
    Save the figure to 'outpath'.
    """
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    # Training loss over epochs
    ax1.plot(df["epoch"], df["train_loss"], label="train_loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    # Validation Dice (overlap-only) on twin axis
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["valDice"], label="valDice", color="tab:orange")
    ax2.set_ylabel("valDice")
    # Merge legends from both axes into one legend placed upper right
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")
    ax1.set_title(title)
    fig.tight_layout()
    savefig(outpath)

def grid_learning_curves(per_fold, mode, outpath):
    """
    Plot a grid of learning curves for all folds of a given mode.
    Each panel shows train_loss (left axis) and valDice (right axis) over epochs.
    """
    if not per_fold: return
    # Arrange up to 4 columns; compute required rows
    cols, rows = 4, math.ceil(len(per_fold)/4)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6*cols,3.2*rows))
    # Ensure axes is a flat array for consistent indexing
    axes = np.array(axes).reshape(-1)
    # Start with all axes hidden (turn on only those we actually plot into)
    for ax in axes: ax.axis("off")
    # Fill panels fold-by-fold
    for i, (fid, df) in enumerate(per_fold):
        ax=axes[i]; ax.axis("on")
        ax.plot(df["epoch"], df["train_loss"], linewidth=1.4)
        ax2=ax.twinx(); ax2.plot(df["epoch"], df["valDice"], color="tab:orange", linewidth=1.4)
        ax.set_title(f"F{fid}", fontsize=10); ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    # Overall title and compact layout
    fig.suptitle(f"Learning curves – {mode}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96])
    savefig(outpath)

# ---------------- Main audit ----------------
def audit_mode(mode_dir: Path, min_epochs: int) -> List[dict]:
    """
    Audit a single mode subdirectory (e.g., CKROOT/raw or CKROOT/n2v).
    For each fold_*:
      - parse train_log.txt (if available)
      - collect stats and warnings
      - save per-fold learning curve plot
    Also saves a mode-level grid plot of all folds.
    Returns a list of row dicts for aggregation.
    """
    mode = mode_dir.name
    rows, per_fold_for_grid = [], []
    # Iterate fold directories sorted lexicographically (fold_1, fold_2, ...)
    for fold_dir in sorted(mode_dir.glob("fold_*")):
        try:
            # Extract numeric id from "fold_X"
            fold_id = int(fold_dir.name.split("_")[-1])
        except:
            # Skip any non-standard folder
            continue

        # Expected artifacts within each fold
        log_path, best_path, last_path = fold_dir/"train_log.txt", fold_dir/"best.pt", fold_dir/"last.pt"
        df = parse_train_log(log_path)                # Parse the training log to DataFrame or None
        has_log = df is not None and not df.empty     # Flag for presence of parsable log content
        has_best, has_last = best_path.exists(), last_path.exists()  # Checkpoint presence flags

        # Initialize defaults (NaN for numeric fields when missing)
        warnings=[]; n_epochs=0; best_epoch=np.nan; best_val=np.nan
        last_loss=np.nan; last_val=np.nan; last10_slope=np.nan

        if df is None:
            # Could not parse or missing log: record a warning
            warnings.append("no_log_or_unparsable")
        else:
            # Basic stats derived from the log table
            n_epochs=len(df)
            best_idx=int(df["valDice"].idxmax())
            best_epoch=int(df.loc[best_idx,"epoch"])
            best_val=float(df.loc[best_idx,"valDice"])
            last_loss=float(df["train_loss"].values[-1])
            last_val=float(df["valDice"].values[-1])
            last10_slope=slope_last_k(df["epoch"].values, df["valDice"].values,10)
            # Add heuristic warnings (few epochs, NaNs, plateaus, etc.)
            warnings.extend(detect_warnings(df,min_epochs))
            # Accumulate for mode-level grid plot
            per_fold_for_grid.append((fold_id,df))
            # Save per-fold learning curve image
            plot_learning_curve(df, f"{mode} – fold {fold_id}", LCDIR/f"{mode}_fold_{fold_id}.png")

        # Check for presence of checkpoints and warn if missing
        if not has_best: warnings.append("missing_best.pt")
        if not has_last: warnings.append("missing_last.pt")

        # Collect a row for the master CSV
        rows.append({
            "mode": mode, "fold": fold_id,
            "epochs": n_epochs, "best_epoch": best_epoch,
            "best_valDice": best_val, "last_train_loss": last_loss,
            "last_valDice": last_val, "last10_valDice_slope": last10_slope,
            "has_log": has_log, "has_best_pt": has_best, "has_last_pt": has_last,
            "warnings": ";".join(warnings)
        })
    # After scanning all folds for this mode, create a grid figure if we have any logs
    if per_fold_for_grid:
        grid_learning_curves(per_fold_for_grid, mode, LCDIR/f"grid_{mode}.png")
    return rows

# ---------------- Run audit ----------------
all_rows=[]
# Iterate subfolders in CKROOT; treat any folder with fold_* children as a mode to audit
for mode_dir in CKROOT.iterdir():
    if mode_dir.is_dir() and any(mode_dir.glob("fold_*")):
        print(f" Auditing mode: {mode_dir.name}")
        all_rows.extend(audit_mode(mode_dir, MIN_EPOCHS))

if not all_rows:
    # Friendly guidance if nothing was found, including an example expected path layout
    print("\n  No valid folds found under:", CKROOT)
    print("Make sure paths look like:")
    print("  C:\\Users\\arefm\\Desktop\\Bacteria\\experiments\\unet_cv\\raw\\fold_1\\train_log.txt")
else:
    # Build a single DataFrame across all modes/folds and save to CSV
    audit_df=pd.DataFrame(all_rows)
    audit_df.sort_values(["mode","fold"], inplace=True)
    audit_df.to_csv(OUTDIR/"training_audit.csv", index=False)

    # Also emit a human-readable errors file with grouped warnings per fold
    with (OUTDIR/"errors.txt").open("w",encoding="utf-8") as f:
        for _,r in audit_df.iterrows():
            ws=[w for w in str(r["warnings"]).split(";") if w]
            if ws:
                f.write(f"[{r['mode'].upper()}] fold {int(r['fold'])}\n")
                for w in ws: f.write(f"  - {w}\n")
                f.write("\n")

    # Final status + where to find plots
    print(f"\n Audit complete!\nResults saved to: {OUTDIR}\nFigures under: {LCDIR}")

    # Attempt to open the output folder automatically (Windows Explorer)
    try:
        subprocess.Popen(f'explorer "{OUTDIR}"')
    except Exception as e:
        print(f"(Could not open folder automatically: {e})")
