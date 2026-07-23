#!/usr/bin/env python3
"""Top-down (x-z) trajectory plots for the swap-ablation results.

Reads results/swap_ablation/<model>/<mode>/<name>_{pred,gt}.npz (poses = (T,4,4) w2c,
OpenCV). Camera center = inv(w2c)[:3,3]. Plots pred (red) vs gt (blue) x-z top-down,
one subplot per sample, one figure per model+mode.

  python topdown_swap.py [--root results/swap_ablation] [--mode a_normal]
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "swap_ablation")


def anchored_centers(poses, ref_poses):
    """(T,4,4) w2c -> (T,3) camera centers in the ref-first-camera frame (inv(c2w[0]) @ c2w).
    Anchoring to the ref (GT) first cam gives the OpenGL cam0 axes (up=+Y, right=+X, forward=-Z)
    and keeps GT/pred comparable. Top-down then drops up(+Y); plot X horizontal, -Z vertical."""
    c2w = np.linalg.inv(poses)
    Rinv = np.linalg.inv(np.linalg.inv(ref_poses)[0])   # inv(ref_c2w[0])
    return (Rinv @ c2w)[:, :3, 3]


def plot_model_mode(model_dir, model, mode, out_path):
    mdir = os.path.join(model_dir, mode)
    preds = sorted(glob.glob(os.path.join(mdir, "*_pred.npz")))
    if not preds:
        print(f"  no preds in {mdir}"); return
    n = len(preds)
    cols = min(5, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for k, pf in enumerate(preds):
        name = os.path.basename(pf)[:-9]
        gt = np.load(pf.replace("_pred.npz", "_gt.npz"))["poses"]
        pr = np.load(pf)["poses"]
        gc = anchored_centers(gt, gt)     # GT anchored to its own first cam
        pc = anchored_centers(pr, gt)     # pred anchored to the SAME (GT) first cam -> comparable
        ax = axes[k // cols][k % cols]
        ax.plot(gc[:, 0], -gc[:, 2], "-o", ms=2.5, lw=1.2, c="tab:blue", label="GT")
        ax.plot(pc[:, 0], -pc[:, 2], "-x", ms=3, lw=1.2, c="tab:red", label="pred")
        ax.scatter(gc[0, 0], -gc[0, 2], c="k", s=45, marker="*", zorder=5)   # start
        ax.set_title(name[:22], fontsize=7)
        ax.set_aspect("equal", "datalim"); ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{model} / {mode}  (top-down, anchored to GT first cam; X vs -Z)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path} ({n} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--mode", default="a_normal")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    models = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for model in models:
        mdir = os.path.join(root, model)
        out = os.path.join(mdir, f"{args.mode}_topdown.png")
        print(f"[{model}]")
        plot_model_mode(mdir, model, args.mode, out)


if __name__ == "__main__":
    main()
