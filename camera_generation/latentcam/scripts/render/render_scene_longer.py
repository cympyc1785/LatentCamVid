"""Whole-scene render, NON-AR: for each 49-frame segment the context is K=6 views retrieved
from the WHOLE out-of-segment region — restricted to the LONGER side (before[0:s] or
after[e:N], whichever is longer). frustum_cover with the coverage ball centered at the
segment's first camera s (honest ball@s selection). Context = out-of-segment only (no target
frame). Render each segment and concat all segments per scene -> one long video.

Contrast with render_ar_scene.py (rolling causal memory). Here every segment uses the full
longer out-of-seg side independently (no memory accumulation).

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_scene_longer.py --scenes "chunk1,chunk2,..." --k 6
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np, torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SCR = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, SCR)
from render_scene_stitched import load_scene, scene_wh, frustum_cover_select, ROOT, CKPT
from render_ar_continuity import render_frames
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
from models.encoder_decoder import EncDec_VitB8


def ctx_scale(centers, side, T=49):
    chunks = [side[i:i+T] for i in range(0, len(side)-T+1, T)]
    if not chunks and len(side) >= 2: chunks = [side]
    if not chunks: return 1e-3
    return max(float(np.mean([np.linalg.norm(centers[ch]-centers[ch[0]], axis=1).mean() for ch in chunks])), 1e-5)


def pick_longer(centers, faxis, w2c, K, w, h, s, e, N, k, radius):
    """K views from the LONGER out-of-seg side (ball@s, frustum_cover). Out-of-seg only."""
    before, after = list(range(0, s)), list(range(e, N))
    side = before if len(before) >= len(after) else after
    side_name = 'before' if len(before) >= len(after) else 'after'
    if not side:
        return [], side_name
    scale = ctx_scale(centers, side)
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=s, seg_scale=scale,
                                 k=k, radius=radius, allowed=side, ball_center=centers[s])
    if len(picks) < k:
        for j in [side[int(round(x))] for x in np.linspace(0, len(side)-1, k)]:
            if j not in picks: picks.append(j)
            if len(picks) >= k: break
    while len(picks) < k and picks: picks.append(picks[-1])
    return picks[:k], side_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True); ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--res', type=int, default=512); ap.add_argument('--radius', type=float, default=2.5)
    ap.add_argument('--num-frames', type=int, default=49); ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--out-dir', default='/tmp/render_scene_longer')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    for si, chunk in enumerate(args.scenes.split(',')):
        chunk = chunk.strip(); sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd); w, h = scene_wh(sd)
        K = np.array([[K_frac[0]*w, 0, K_frac[2]*w], [0, K_frac[1]*h, K_frac[3]*h], [0, 0, 1]], float)
        N = c2w_all.shape[0]; centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]; w2c = np.linalg.inv(c2w_all)
        prompts = json.load(open(osp.join(sd, 'prompts.json')))
        segs = sorted([(int(v['frame_idx'][0]), int(v['frame_idx'][1]), kk) for kk, v in prompts.items()
                       if v.get('frame_idx') and len(v['frame_idx']) == 2 and int(v['frame_idx'][1]) <= N
                       and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= args.num_frames], key=lambda x: x[0])
        if not segs: print(f"scene{si}: no segments"); continue

        stitched = []; seg_info = []
        for (s, e, sk) in segs:
            tgt = list(range(s, e))[::args.stride]
            ctx, side_name = pick_longer(centers, faxis, w2c, K, w, h, s, e, N, args.k, args.radius)
            if not ctx: continue
            gt, ren = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx, args.res, device, dtype)
            if gt is None: continue
            H, W = gt.shape[1], gt.shape[2]
            for t in range(len(tgt)):
                canvas = np.concatenate([gt[t], ren[t]], axis=1)
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                d.text((4, 4), "GT", fill=(255, 255, 0)); d.text((W+4, 4), "LagerNVS (longer out-of-seg)", fill=(255, 255, 0))
                d.text((4, H-14), f"seg{sk}[{s}:{e}] side={side_name} K={len(ctx)}", fill=(0, 255, 0))
                stitched.append(np.asarray(im))
            seg_info.append((s, e, sk, ctx, side_name))
            print(f"  scene{si} seg{sk}[{s}:{e}] side={side_name} ctx={ctx}")
        name = f"longer_{si:02d}_{chunk.split('/')[-1][:10]}"
        if stitched:
            imageio.mimwrite(osp.join(args.out_dir, name+".mp4"), stitched, fps=10, quality=8, macro_block_size=1)
            print(f"scene{si} -> {name}.mp4 ({len(stitched)} frames, {len(segs)} segments)")
        if seg_info:
            ax0, ax1 = np.argsort(centers.var(0))[-2:]; Lc = {0: 'X', 1: 'Y', 2: 'Z'}
            ns = len(seg_info); ncol = min(4, ns); nrow = int(np.ceil(ns / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.7*nrow), squeeze=False)
            for ci, (s, e, sk, ctx, side_name) in enumerate(seg_info):
                a = axes[ci//ncol][ci%ncol]
                a.scatter(centers[:, ax0], centers[:, ax1], s=5, c='lightgray')
                a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=.5, alpha=.5)
                a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=16, c='tab:blue', label='cur seg')
                a.scatter(centers[ctx, ax0], centers[ctx, ax1], s=85, marker='*', c='tab:orange', edgecolors='k', linewidths=.5, label='ctx (longer out-of-seg)')
                a.scatter([centers[s, ax0]], [centers[s, ax1]], s=60, c='k', marker='s', label='seg start s (ball center)')
                a.set_title(f"seg{sk}[{s}:{e}] side={side_name}", fontsize=8)
                a.set_aspect('equal', 'datalim'); a.set_xlabel(Lc[ax0]); a.set_ylabel(Lc[ax1])
                if ci == 0: a.legend(fontsize=6)
            for j in range(ns, nrow*ncol): axes[j//ncol][j%ncol].axis('off')
            fig.suptitle(f"per-segment context — longer out-of-seg (ball@s) — {chunk.split('/')[-1][:12]}", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(osp.join(args.out_dir, name+"_ctxviz.png"), dpi=110)
            print(f"       viz -> {name}_ctxviz.png")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
