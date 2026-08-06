"""Explanatory viz of START-based, coverage-based context retrieval (the geo_viewS selection):
  - view0 = frame s (target segment's FIRST/start camera; the known anchor),
  - candidate pool = the LONGER out-of-segment side (before[0:s] or after[e:N]),
  - a coverage BALL of radius (radius * context_scale) is centered at frame s,
  - frustum_cover greedily picks (k-1) out-of-seg views that maximally COVER that ball,
  - final geo context = [frame s] + (k-1) retrieved.
Top-down figure per scene with the ball circle, candidate pool, and selected views annotated.

Run: python scripts/viz_start_coverage_retrieval.py --segs "chunk:seg,..." --out /tmp/start_cov.png
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle

SCR = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, SCR)
from render_scene_stitched import frustum_cover_select
from render_scene_longer import ctx_scale
from blacklist_by_coverage import load_scene   # numpy: -> centers, faxis, w2c, K, w, h
ROOT = '/data1/cympyc1785/data/DL3DV/scenes'


def select(centers, faxis, w2c, K, w, h, s, e, N, k, radius):
    before, after = list(range(0, s)), list(range(e, N))
    side = before if len(before) >= len(after) else after
    side_name = 'before' if len(before) >= len(after) else 'after'
    scale = ctx_scale(centers, side)
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=s, seg_scale=scale,
                                 k=k - 1, radius=radius, allowed=side, ball_center=centers[s])
    picks = [p for p in picks if p != s][:k - 1]
    ctx = [s] + picks
    return ctx, side, side_name, radius * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', required=True); ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--radius', type=float, default=2.5); ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--out', default='/tmp/start_cov.png')
    args = ap.parse_args()
    pairs = [p.rsplit(':', 1) for p in args.segs.split(',')]
    n = len(pairs); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), squeeze=False)
    for i, (chunk, sk) in enumerate(pairs):
        sd = osp.join(ROOT, chunk)
        centers, faxis, w2c, K, w, h = load_scene(osp.join(sd, 'transforms.json'))
        N = centers.shape[0]
        seg = json.load(open(osp.join(sd, 'prompts.json')))[sk]
        s, e = int(seg['frame_idx'][0]), int(seg['frame_idx'][1])
        ctx, side, side_name, rad = select(centers, faxis, w2c, K, w, h, s, e, N, args.k, args.radius)
        retr = ctx[1:]
        ax0, ax1 = np.argsort(centers.var(0))[-2:]; Lc = {0: 'X', 1: 'Y', 2: 'Z'}
        a = axes[i // ncol][i % ncol]
        # all frames + non-selected candidate pool
        a.scatter(centers[:, ax0], centers[:, ax1], s=6, c='lightgray', label='all frames', zorder=1)
        a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=.5, alpha=.5, zorder=1)
        a.scatter(centers[side, ax0], centers[side, ax1], s=14, c='#f0c890', edgecolors='none',
                  label='candidate pool (longer out-of-seg)', zorder=2)
        # target segment (to GENERATE) + start frame s
        a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=18, c='tab:blue', label='target segment [s:e]', zorder=3)
        # coverage ball around frame s
        a.add_patch(Circle((centers[s, ax0], centers[s, ax1]), rad, fill=False, ls='--',
                            ec='tab:purple', lw=1.5, zorder=4, label=f'coverage ball (r={args.radius}·ctx_scale)'))
        # viewing-direction (optical axis, OpenCV +Z = faxis) arrows
        alen = 0.28 * rad
        tsub = list(range(s, e, max(1, (e - s) // 6)))          # a few target frames
        a.quiver(centers[tsub, ax0], centers[tsub, ax1], faxis[tsub, ax0] * alen, faxis[tsub, ax1] * alen,
                 angles='xy', scale_units='xy', scale=1, color='tab:blue', width=.004, alpha=.5, zorder=3)
        a.quiver(centers[retr, ax0], centers[retr, ax1], faxis[retr, ax0] * alen, faxis[retr, ax1] * alen,
                 angles='xy', scale_units='xy', scale=1, color='darkred', width=.006, zorder=5)
        a.quiver([centers[s, ax0]], [centers[s, ax1]], [faxis[s, ax0] * alen], [faxis[s, ax1] * alen],
                 angles='xy', scale_units='xy', scale=1, color='darkgreen', width=.007, zorder=6)
        # selected retrieved context + view0=start
        a.scatter(centers[retr, ax0], centers[retr, ax1], s=140, marker='*', c='tab:red',
                  edgecolors='k', linewidths=.6, label='retrieved context (frustum_cover)', zorder=5)
        a.scatter([centers[s, ax0]], [centers[s, ax1]], s=150, marker='P', c='tab:green',
                  edgecolors='k', linewidths=.6, label='view0 = start (frame s)', zorder=6)
        a.plot([], [], color='darkred', lw=1.5, label='context look dir (optical axis)')
        a.plot([], [], color='tab:blue', lw=1.5, alpha=.5, label='target look dir')
        a.set_title(f"{chunk.split('/')[-1][:10]} seg{sk} [{s}:{e}]  side={side_name}", fontsize=10)
        a.set_aspect('equal', 'datalim'); a.set_xlabel(Lc[ax0]); a.set_ylabel(Lc[ax1])
        if i == n - 1:                                # legend box on the rightmost subplot
            a.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5),
                     framealpha=0.95, title='markers')
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle("Start-based coverage retrieval: view0 = start frame s; retrieve (k-1) out-of-seg "
                 "views that maximally cover the ball around s (frustum_cover)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print("wrote", args.out)


if __name__ == '__main__':
    main()
