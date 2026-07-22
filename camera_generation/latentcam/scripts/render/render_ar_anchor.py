"""AR continuity coordinate-anchor ablation.

Force LagerNVS's context view 0 (= its reconstruction reference / coord anchor) to a chosen
camera, so the render frame aligns with a chosen origin (you can't disable VGGT's first-view
anchoring, but you can MAKE view0 be that camera by feeding it first).

Two variants per chunk (context = [view0] + 5 retrieved from rolling memory near current cs):
  A "reanchor" : view0 = the CURRENT chunk's first camera (cs)          -> frame shifts per chunk
  B "fixed"    : view0 = the FIRST chunk's first camera (segment s)     -> frame fixed to start
Both retrieve the other 5 based on the current chunk position cs. Output [GT | A | B] video.

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_ar_anchor.py --segs "chunk:seg,..." --chunk 7 --k 6
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


def retrieve(centers, faxis, w2c, K, w, h, pool, cur, k, radius, exclude):
    """frustum_cover over pool (minus exclude), ball centered at cur. returns up to k idx."""
    pool = [j for j in pool if j not in exclude]
    if not pool:
        return []
    if len(pool) <= k:
        return pool[:k]
    seg_scale = max(float(np.linalg.norm(centers[pool] - centers[cur], axis=1).mean()), 1e-5)
    anchor = min(pool, key=lambda j: np.linalg.norm(centers[j] - centers[cur]))
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=anchor, seg_scale=seg_scale,
                                 k=k, radius=radius, allowed=pool, ball_center=centers[cur])
    picks = [j for j in picks if j not in exclude]
    for j in [pool[int(round(x))] for x in np.linspace(0, len(pool) - 1, k)]:
        if len(picks) >= k: break
        if j not in picks and j not in exclude: picks.append(j)
    return picks[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', required=True); ap.add_argument('--chunk', type=int, default=7)
    ap.add_argument('--k', type=int, default=6); ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--radius', type=float, default=2.5); ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--out-dir', default='/tmp/render_ar_anchor')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    for si, pr in enumerate(args.segs.split(',')):
        chunk, seg_key = pr.rsplit(':', 1)
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd); w, h = scene_wh(sd)
        K = np.array([[K_frac[0]*w, 0, K_frac[2]*w], [0, K_frac[1]*h, K_frac[3]*h], [0, 0, 1]], float)
        seg = json.load(open(osp.join(sd, 'prompts.json')))[seg_key]
        s, e = int(seg['frame_idx'][0]), int(seg['frame_idx'][1])
        N = c2w_all.shape[0]; centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]; w2c = np.linalg.inv(c2w_all)
        before, after = list(range(0, s)), list(range(e, N))
        source = before if len(before) >= len(after) else after

        stitched = []; chunk_info = []
        for cs in range(s, e, args.chunk):
            ce = min(cs + args.chunk, e); tgt = list(range(cs, ce))
            mem = list(source) + list(range(s, cs))                  # rolling causal memory
            retr = retrieve(centers, faxis, w2c, K, w, h, mem, cs, args.k - 1, args.radius, exclude={s, cs})
            ctx_A = [cs] + retr                                       # view0 = current chunk first cam
            ctx_B = [s] + retr                                       # view0 = first chunk first cam (frame s)
            chunk_info.append((cs, ce, retr, set(source)))
            gt, ren_A = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_A, args.res, device, dtype)
            _, ren_B = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_B, args.res, device, dtype)
            if gt is None: continue
            H, W = gt.shape[1], gt.shape[2]
            for t in range(len(tgt)):
                canvas = np.concatenate([gt[t], ren_A[t], ren_B[t]], axis=1)
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                d.text((4, 4), "GT", fill=(255, 255, 0))
                d.text((W+4, 4), "A: reanchor(view0=chunk start)", fill=(255, 255, 0))
                d.text((2*W+4, 4), "B: fixed(view0=frame s)", fill=(255, 255, 0))
                d.text((4, H-14), f"chunk@{cs} f{tgt[t]} view0_A={cs} view0_B={s}", fill=(0, 255, 0))
                stitched.append(np.asarray(im))
            print(f"  scene{si} chunk@{cs} ctx_A={ctx_A} ctx_B={ctx_B}")
        name = f"aranchor_{si:02d}_{chunk.split('/')[-1][:10]}_{seg_key}"
        if stitched:
            imageio.mimwrite(osp.join(args.out_dir, name+".mp4"), stitched, fps=6, quality=8, macro_block_size=1)
            print(f"scene{si} -> {name}.mp4 ({len(stitched)} frames)")
        # per-chunk context viz
        if chunk_info:
            ax0, ax1 = np.argsort(centers.var(0))[-2:]; Lc = {0: 'X', 1: 'Y', 2: 'Z'}
            nc = len(chunk_info); ncol = min(4, nc); nrow = int(np.ceil(nc / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.7*nrow), squeeze=False)
            for ci, (cs, ce, retr, srcset) in enumerate(chunk_info):
                rs = [j for j in retr if j in srcset]; rg = [j for j in retr if j not in srcset]
                a = axes[ci//ncol][ci%ncol]
                a.scatter(centers[:, ax0], centers[:, ax1], s=5, c='lightgray')
                a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=.5, alpha=.5)
                a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=10, c='#bcd', label='target')
                a.scatter(centers[cs:ce, ax0], centers[cs:ce, ax1], s=22, c='tab:blue', label='cur chunk')
                if rs: a.scatter(centers[rs, ax0], centers[rs, ax1], s=70, marker='X', c='tab:orange', edgecolors='k', linewidths=.5, label='retr: source')
                if rg: a.scatter(centers[rg, ax0], centers[rg, ax1], s=95, marker='*', c='tab:red', edgecolors='k', linewidths=.5, label='retr: gen-past')
                a.scatter([centers[cs, ax0]], [centers[cs, ax1]], s=70, c='k', marker='s', label='view0_A=cs(anchor)')
                a.scatter([centers[s, ax0]], [centers[s, ax1]], s=70, c='tab:green', marker='P', label='view0_B=frame s')
                a.set_title(f"chunk@{cs}[{cs}:{ce}] src{len(rs)}/gen{len(rg)}", fontsize=8)
                a.set_aspect('equal', 'datalim'); a.set_xlabel(Lc[ax0]); a.set_ylabel(Lc[ax1])
                if ci == 0: a.legend(fontsize=6, loc='best')
            for j in range(nc, nrow*ncol): axes[j//ncol][j%ncol].axis('off')
            fig.suptitle(f"per-chunk context — {chunk.split('/')[-1][:12]} seg{seg_key} "
                         f"(view0_A=cs black■ / view0_B=frame s green✚ / retrieved 5)", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(osp.join(args.out_dir, name+"_ctxviz.png"), dpi=110)
            print(f"       viz -> {name}_ctxviz.png")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
