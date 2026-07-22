"""Per-segment context+target viz for render_scene_longer_anchor (no GPU). Shows, per segment:
target (cur seg), retrieved context (longer out-of-seg), view0_A=seg start, view0_B=scene start.
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SCR = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, SCR)
from render_scene_stitched import frustum_cover_select, ROOT
from render_scene_longer import ctx_scale
from blacklist_by_coverage import load_scene   # numpy: -> centers, faxis, w2c, K, w, h


def retr_longer(centers, faxis, w2c, K, w, h, s, e, N, k, radius, exclude):
    before, after = list(range(0, s)), list(range(e, N))
    side = before if len(before) >= len(after) else after
    side_name = 'before' if len(before) >= len(after) else 'after'
    side = [j for j in side if j not in exclude]
    if not side: return [], side_name
    scale = ctx_scale(centers, side)
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=s, seg_scale=scale,
                                 k=k, radius=radius, allowed=side, ball_center=centers[s])
    picks = [j for j in picks if j not in exclude]
    for j in [side[int(round(x))] for x in np.linspace(0, len(side)-1, k)]:
        if len(picks) >= k: break
        if j not in picks: picks.append(j)
    return picks[:k], side_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True); ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--radius', type=float, default=2.5); ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--out-dir', default='/tmp/render_scene_longer_anchor')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for si, chunk in enumerate(args.scenes.split(',')):
        chunk = chunk.strip(); sd = osp.join(ROOT, chunk)
        centers, faxis, w2c, K, w, h = load_scene(osp.join(sd, 'transforms.json'))
        N = centers.shape[0]
        prompts = json.load(open(osp.join(sd, 'prompts.json')))
        segs = sorted([(int(v['frame_idx'][0]), int(v['frame_idx'][1]), kk) for kk, v in prompts.items()
                       if v.get('frame_idx') and len(v['frame_idx']) == 2 and int(v['frame_idx'][1]) <= N
                       and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= args.num_frames], key=lambda x: x[0])
        if not segs: continue
        s0 = segs[0][0]
        ax0, ax1 = np.argsort(centers.var(0))[-2:]; Lc = {0: 'X', 1: 'Y', 2: 'Z'}
        ns = len(segs); ncol = min(4, ns); nrow = int(np.ceil(ns / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.7*nrow), squeeze=False)
        for ci, (s, e, sk) in enumerate(segs):
            retr, side_name = retr_longer(centers, faxis, w2c, K, w, h, s, e, N, args.k-1, args.radius, exclude={s, s0})
            a = axes[ci//ncol][ci%ncol]
            a.scatter(centers[:, ax0], centers[:, ax1], s=5, c='lightgray')
            a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=.5, alpha=.5)
            a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=16, c='tab:blue', label='target (cur seg)')
            if retr: a.scatter(centers[retr, ax0], centers[retr, ax1], s=85, marker='*', c='tab:orange', edgecolors='k', linewidths=.5, label='ctx (longer out-of-seg)')
            a.scatter([centers[s, ax0]], [centers[s, ax1]], s=70, c='k', marker='s', label='view0_A=seg start')
            a.scatter([centers[s0, ax0]], [centers[s0, ax1]], s=70, c='tab:green', marker='P', label='view0_B=scene start')
            a.set_title(f"seg{sk}[{s}:{e}] side={side_name}", fontsize=8)
            a.set_aspect('equal', 'datalim'); a.set_xlabel(Lc[ax0]); a.set_ylabel(Lc[ax1])
            if ci == 0: a.legend(fontsize=6)
        for j in range(ns, nrow*ncol): axes[j//ncol][j%ncol].axis('off')
        fig.suptitle(f"per-segment context+target (longer out-of-seg + anchor ablation) — {chunk.split('/')[-1][:12]}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = osp.join(args.out_dir, f"longeranchor_{si:02d}_{chunk.split('/')[-1][:10]}_ctxviz.png")
        fig.savefig(p, dpi=110); print("wrote", p)


if __name__ == '__main__':
    main()
