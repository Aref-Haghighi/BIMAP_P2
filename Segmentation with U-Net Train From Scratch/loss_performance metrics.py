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

import math, re, os, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
CKROOT = Path(r"C:\Users\arefm\Desktop\Bacteria\experiments\unet_cv")

# Ask user for output directory
root = tk.Tk(); root.withdraw()
OUTDIR = Path(filedialog.askdirectory(title="Select output folder to save audit results"))
if not OUTDIR:
    raise SystemExit("❌ No output folder selected. Exiting.")
OUTDIR.mkdir(parents=True, exist_ok=True)
LCDIR = OUTDIR / "learning_curves"; LCDIR.mkdir(parents=True, exist_ok=True)

MIN_EPOCHS = 15
DPI = 150

# ---------------- Helpers ----------------
def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()

def parse_train_log(path: Path) -> Optional[pd.DataFrame]:
    """Parse train_log.txt into a DataFrame."""
    if not path.exists():
        return None
    rows = []
    pat = re.compile(
        r"epoch\s+(\d+)\s*\|\s*loss\s*([0-9.]+)\s*\|\s*valDice\(overlap-only\)\s*([0-9.]+)\s*\|\s*best\s*([0-9.]+)",
        re.IGNORECASE
    )
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4])))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["epoch", "train_loss", "valDice", "best_so_far"])
    return df.sort_values("epoch").reset_index(drop=True)

def slope_last_k(x, y, k=10):
    """Linear slope over the last k epochs."""
    if len(x) < 2: return np.nan
    xk, yk = x[-k:], y[-k:]
    if len(xk) < 2: return np.nan
    return float(np.polyfit(xk, yk, 1)[0])

def detect_warnings(df: pd.DataFrame, min_epochs: int) -> List[str]:
    """Detect anomalies or early stops."""
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
    """Plot per-fold loss and Dice."""
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    ax1.plot(df["epoch"], df["train_loss"], label="train_loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["valDice"], label="valDice", color="tab:orange")
    ax2.set_ylabel("valDice")
    h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")
    ax1.set_title(title)
    fig.tight_layout()
    savefig(outpath)

def grid_learning_curves(per_fold, mode, outpath):
    """Plot grid of learning curves for all folds."""
    if not per_fold: return
    cols, rows = 4, math.ceil(len(per_fold)/4)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6*cols,3.2*rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for i, (fid, df) in enumerate(per_fold):
        ax=axes[i]; ax.axis("on")
        ax.plot(df["epoch"], df["train_loss"], linewidth=1.4)
        ax2=ax.twinx(); ax2.plot(df["epoch"], df["valDice"], color="tab:orange", linewidth=1.4)
        ax.set_title(f"F{fid}", fontsize=10); ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
    fig.suptitle(f"Learning curves – {mode}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0,0,1,0.96])
    savefig(outpath)

# ---------------- Main audit ----------------
def audit_mode(mode_dir: Path, min_epochs: int) -> List[dict]:
    """Audit a single mode folder (raw/n2v/etc.)."""
    mode = mode_dir.name
    rows, per_fold_for_grid = [], []
    for fold_dir in sorted(mode_dir.glob("fold_*")):
        try:
            fold_id = int(fold_dir.name.split("_")[-1])
        except:
            continue

        log_path, best_path, last_path = fold_dir/"train_log.txt", fold_dir/"best.pt", fold_dir/"last.pt"
        df = parse_train_log(log_path)
        has_log = df is not None and not df.empty
        has_best, has_last = best_path.exists(), last_path.exists()

        warnings=[]; n_epochs=0; best_epoch=np.nan; best_val=np.nan
        last_loss=np.nan; last_val=np.nan; last10_slope=np.nan

        if df is None:
            warnings.append("no_log_or_unparsable")
        else:
            n_epochs=len(df)
            best_idx=int(df["valDice"].idxmax())
            best_epoch=int(df.loc[best_idx,"epoch"])
            best_val=float(df.loc[best_idx,"valDice"])
            last_loss=float(df["train_loss"].values[-1])
            last_val=float(df["valDice"].values[-1])
            last10_slope=slope_last_k(df["epoch"].values, df["valDice"].values,10)
            warnings.extend(detect_warnings(df,min_epochs))
            per_fold_for_grid.append((fold_id,df))
            plot_learning_curve(df, f"{mode} – fold {fold_id}", LCDIR/f"{mode}_fold_{fold_id}.png")

        if not has_best: warnings.append("missing_best.pt")
        if not has_last: warnings.append("missing_last.pt")

        rows.append({
            "mode": mode, "fold": fold_id,
            "epochs": n_epochs, "best_epoch": best_epoch,
            "best_valDice": best_val, "last_train_loss": last_loss,
            "last_valDice": last_val, "last10_valDice_slope": last10_slope,
            "has_log": has_log, "has_best_pt": has_best, "has_last_pt": has_last,
            "warnings": ";".join(warnings)
        })
    if per_fold_for_grid:
        grid_learning_curves(per_fold_for_grid, mode, LCDIR/f"grid_{mode}.png")
    return rows

# ---------------- Run audit ----------------
all_rows=[]
for mode_dir in CKROOT.iterdir():
    if mode_dir.is_dir() and any(mode_dir.glob("fold_*")):
        print(f"🔍 Auditing mode: {mode_dir.name}")
        all_rows.extend(audit_mode(mode_dir, MIN_EPOCHS))

if not all_rows:
    print("\n⚠️  No valid folds found under:", CKROOT)
    print("Make sure paths look like:")
    print("  C:\\Users\\arefm\\Desktop\\Bacteria\\experiments\\unet_cv\\raw\\fold_1\\train_log.txt")
else:
    audit_df=pd.DataFrame(all_rows)
    audit_df.sort_values(["mode","fold"], inplace=True)
    audit_df.to_csv(OUTDIR/"training_audit.csv", index=False)

    with (OUTDIR/"errors.txt").open("w",encoding="utf-8") as f:
        for _,r in audit_df.iterrows():
            ws=[w for w in str(r["warnings"]).split(";") if w]
            if ws:
                f.write(f"[{r['mode'].upper()}] fold {int(r['fold'])}\n")
                for w in ws: f.write(f"  - {w}\n")
                f.write("\n")

    print(f"\n✅ Audit complete!\nResults saved to: {OUTDIR}\nFigures under: {LCDIR}")

    # Automatically open folder in Windows Explorer
    try:
        subprocess.Popen(f'explorer "{OUTDIR}"')
    except Exception as e:
        print(f"(Could not open folder automatically: {e})")
