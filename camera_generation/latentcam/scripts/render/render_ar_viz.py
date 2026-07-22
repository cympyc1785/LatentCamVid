"""Visualize, per chunk of the AR continuity render, the ANCHOR and the selected CONTEXT
views (matching render_ar_continuity.pick_ctx). Distinguishes context drawn from the SOURCE
(out-of-seg) vs the GENERATED-so-far past (target[s:cs]) so the growing AR memory is visible.
Pure geometry (CPU).

Run: python scripts/render_ar_viz.py --segs "chunk:seg,..." --chunk 7 --k 6 --out-dir /tmp/render_ar
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SCR = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, SCR)
from render_scene_stitched import load_scene, scene_wh, frustum_cover_select, ROOT


def pick_ctx(centers, faxis, w2c, K, w, h, memory, cur, k, radius):
    """same as render_ar_continuity.pick_ctx"""
    if len(memory) <= k:
        idx = list(memory)
        while len(idx) < k and idx:
            idx.append(idx[-1])
        return idx[:k] if idx else [cur]
    seg_scale = max(float(np.linalg.norm(centers[list(memory)] - centers[cur], axis=1).mean()), 1e-5)
    anchor = min(memory, key=lambda j: np.linalg.norm(centers[j] - centers[cur]))
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=anchor, seg_scale=seg_scale,
                                 k=k, radius=radius, allowed=list(memory), ball_center=centers[cur])
    if len(picks) < k:
        for j in [list(memory)[int(round(x))] for x in np.linspace(0, len(memory) - 1, k)]:
            if j not in picks: picks.append(j)
            if len(picks) >= k: break
    return picks[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', required=True)
    ap.add_argument('--chunk', type=int, default=7); ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--radius', type=float, default=2.5); ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--out-dir', default='/tmp/render_ar')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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
        source_set = set(source)
        ax0, ax1 = np.argsort(centers.var(0))[-2:]; L = {0: 'X', 1: 'Y', 2: 'Z'}

        chunk_starts = list(range(s, e, args.chunk))
        nc = len(chunk_starts); ncol = min(4, nc); nrow = int(np.ceil(nc / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.7*nrow), squeeze=False)
        for ci, cs in enumerate(chunk_starts):
            ce = min(cs + args.chunk, e); cur = cs if cs > s else s
            mem = list(source) + list(range(s, cs))
            ctx = pick_ctx(centers, faxis, w2c, K, w, h, mem, cur, args.k, args.radius)
            ctx_src = [j for j in ctx if j in source_set]        # context from SOURCE
            ctx_gen = [j for j in ctx if j not in source_set]    # context from generated-past
            a = axes[ci//ncol][ci%ncol]
            a.scatter(centers[:, ax0], centers[:, ax1], s=5, c='lightgray')
            a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=.5, alpha=.5)
            a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=10, c='#bcd', label='target[s:e]')
            a.scatter(centers[cs:ce, ax0], centers[cs:ce, ax1], s=22, c='tab:blue', label='current chunk')
            if ctx_src:
                a.scatter(centers[ctx_src, ax0], centers[ctx_src, ax1], s=70, marker='X',
                          c='tab:orange', edgecolors='k', linewidths=.5, label='ctx: source')
            if ctx_gen:
                a.scatter(centers[ctx_gen, ax0], centers[ctx_gen, ax1], s=95, marker='*',
                          c='tab:red', edgecolors='k', linewidths=.5, label='ctx: gen-past')
            a.scatter([centers[cur, ax0]], [centers[cur, ax1]], s=60, c='k', marker='s', label='anchor=cs')
            a.set_title(f"chunk@{cs}[{cs}:{ce}] |mem|={len(mem)} (src{len(ctx_src)}/gen{len(ctx_gen)})", fontsize=8)
            a.set_aspect('equal', 'datalim'); a.set_xlabel(L[ax0]); a.set_ylabel(L[ax1])
            if ci == 0: a.legend(fontsize=6, loc='best')
        for j in range(nc, nrow*ncol): axes[j//ncol][j%ncol].axis('off')
        fig.suptitle(f"AR continuity — per-chunk anchor & context  |  {chunk.split('/')[-1][:12]} seg{seg_key}",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = osp.join(args.out_dir, f"ar_{si:02d}_{chunk.split('/')[-1][:10]}_{seg_key}_ctxviz.png")
        fig.savefig(p, dpi=115); print("wrote", p)


if __name__ == '__main__':
    main()
