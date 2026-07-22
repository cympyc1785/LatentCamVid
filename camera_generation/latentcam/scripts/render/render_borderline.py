"""Render specific (scene, segment) pairs — e.g. the borderline survivors just above a
filtering tau — with coverage-aware out-of-segment context (frustum_cover longer side) +
posed cam_token. Produces per-segment [GT|LagerNVS] mp4, a mid-frame montage, and a
top-down context-selection viz. Helps judge whether a tau threshold is strict enough.

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_borderline.py --segs "chunk1:seg,chunk2:seg" \
      --k 6 --stride 3 --out-dir /tmp/render_border
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SCR = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/scripts"
sys.path.insert(0, SCR)
from render_scene_stitched import (load_scene, scene_wh, select_context, render_segment,
                                   viewpoint_coverage, ROOT, CKPT)
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
from models.encoder_decoder import EncDec_VitB8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', required=True, help='comma list of chunk:segkey')
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--radius', type=float, default=2.5)
    ap.add_argument('--out-dir', default='/tmp/render_border')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16

    pairs = [p.rsplit(':', 1) for p in args.segs.split(',') if p.strip()]
    model = EncDec_VitB8(pretrained_vggt=False,
                         attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.to(device).eval()

    mids, viz_items = [], []
    for i, (chunk, seg_key) in enumerate(pairs):
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd)
        w, h = scene_wh(sd)
        K = np.array([[K_frac[0] * w, 0, K_frac[2] * w],
                      [0, K_frac[1] * h, K_frac[3] * h], [0, 0, 1]], float)
        seg = json.load(open(osp.join(sd, 'prompts.json')))[seg_key]
        s, e = int(seg['frame_idx'][0]), int(seg['frame_idx'][1])
        ctx_idx, side, side_name, seg_scale = select_context(c2w_all, K, w, h, s, e,
                                                             args.k, args.radius)
        centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]
        cov = viewpoint_coverage(centers, faxis, ctx_idx, s, e, seg_scale)
        gt, ren = render_segment(model, sd, c2w_all, fnames, K_frac, w, h, s, e,
                                 ctx_idx, args.stride, args.res, device, dtype)
        if gt is None:
            print(f"[{i}] skip {chunk} {seg_key}"); continue
        H, W = gt.shape[1], gt.shape[2]
        frames = []
        for t in range(gt.shape[0]):
            canvas = np.concatenate([gt[t], ren[t]], axis=1)
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            d.text((4, 4), "GT", fill=(255, 255, 0)); d.text((W + 4, 4), "LagerNVS", fill=(255, 255, 0))
            d.text((4, H - 14), f"cov={cov:.2f} {side_name} K={len(ctx_idx)} f{s + t * args.stride}",
                   fill=(0, 255, 0))
            frames.append(np.asarray(im))
        name = f"{i:02d}_cov{cov:.2f}_{chunk.split('/')[-1][:10]}_{seg_key}"
        imageio.mimwrite(osp.join(args.out_dir, name + ".mp4"), frames, fps=8,
                         quality=8, macro_block_size=1)
        mids.append(frames[len(frames) // 2])
        viz_items.append((chunk, c2w_all, s, e, ctx_idx, side_name, cov))
        print(f"[{i}] cov={cov:.2f} {side_name} {chunk.split('/')[-1][:10]} seg{seg_key} "
              f"ctx={ctx_idx} -> {name}.mp4")

    if mids:
        w0 = min(m.shape[1] for m in mids)
        Image.fromarray(np.concatenate([m[:, :w0] for m in mids], 0)).save(
            osp.join(args.out_dir, "montage_mid.png"))
    # context viz grid
    if viz_items:
        n = len(viz_items); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
        for i, (chunk, c2w_all, s, e, ctx_idx, side_name, cov) in enumerate(viz_items):
            centers = c2w_all[:, :3, 3]; ax0, ax1 = np.argsort(centers.var(0))[-2:]
            a = axes[i // ncol][i % ncol]
            a.scatter(centers[:, ax0], centers[:, ax1], s=6, c='lightgray')
            a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=0.5, alpha=0.5)
            a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=14, c='tab:blue', label='target')
            a.scatter(centers[list(ctx_idx), ax0], centers[list(ctx_idx), ax1], s=70,
                      c='tab:red', marker='*', edgecolors='k', label='context')
            a.scatter([centers[s, ax0]], [centers[s, ax1]], s=40, c='tab:green', marker='s')
            a.set_title(f"cov={cov:.2f} {side_name}\n{chunk.split('/')[-1][:10]}", fontsize=9)
            a.set_aspect('equal', 'datalim')
            if i == 0:
                a.legend(fontsize=7)
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis('off')
        fig.suptitle("Borderline survivors: context selection (per-panel cov = selected-K coverage)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(osp.join(args.out_dir, "context_viz.png"), dpi=110)
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
