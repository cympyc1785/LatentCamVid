"""Compare context-view selection: ours (frustum_cover, region+direction, deterministic grid)
vs I3DM-style (Monte-Carlo FOV-overlap greedy, query = segment FIRST camera, binary in-FOV).
Both restricted to the LONGER out-of-segment side, K=6, target excluded. Pure geometry (CPU).

I3DM-style (fov_overlap_retrieval.retrieve_top_k_cameras, learned parts dropped):
  - sample n_pts random points in a sphere around the QUERY (frame s) camera center,
  - keep points in the query FOV,
  - greedily pick the candidate covering the most still-uncovered query-FOV points
    (binary in-FOV; no viewing-direction diversity term; no FPS fallback).
Sphere radius = radius * seg_scale (scene-relative; I3DM uses an absolute 30).

Usage: python scripts/select_compare.py --n 200 --k 6 [--viz-n 6 --out /tmp/select_cmp.png]
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, csv, json, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SCR = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, SCR)
from blacklist_by_coverage import read_meta, read_blacklist, load_scene
from dump_coverage_selk import frustum_cover_select, select_context, viewpoint_coverage

ROOT = '/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K'


def rand_in_sphere(n, r, rng):
    v = rng.normal(size=(n, 3)); v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    u = rng.uniform(size=n) ** (1.0 / 3.0)
    return v * (u * r)[:, None]


def in_fov(P, w2c_j, K, w, h):
    Xc = (w2c_j[:3, :3] @ P.T).T + w2c_j[:3, 3]
    z = Xc[:, 2]
    uv = (K @ (Xc / np.clip(z, 1e-6, None)[:, None]).T).T
    return (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)


def i3dm_select(centers, w2c, K, w, h, s, e, side, k, radius, rng, n_pts=10000):
    """Monte-Carlo FOV-overlap greedy, query = frame s. Returns picked indices."""
    seg_scale = max(float(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean()), 1e-5)
    qc = centers[s]
    pts = qc[None] + rand_in_sphere(n_pts, radius * seg_scale, rng)
    inq = in_fov(pts, w2c[s], K, w, h)
    P = pts[inq]
    if len(P) == 0 or not side:
        return [side[int(round(x))] for x in np.linspace(0, len(side) - 1, k)] if side else []
    cov = {j: set(np.where(in_fov(P, w2c[j], K, w, h))[0]) for j in side}
    remaining = set(range(len(P))); picked = []
    for _ in range(min(k, len(side))):
        best = max((j for j in side if j not in picked),
                   key=lambda x: len(cov[x] & remaining), default=None)
        if best is None or len(cov[best] & remaining) == 0:
            break
        picked.append(best); remaining -= cov[best]
    # pad by even spacing if greedy saturated early
    if len(picked) < k:
        for j in [side[int(round(x))] for x in np.linspace(0, len(side) - 1, k)]:
            if j not in picked:
                picked.append(j)
            if len(picked) >= k:
                break
    return picked[:k]


def longer_side(N, s, e):
    before, after = list(range(0, s)), list(range(e, N))
    return (before, 'before') if len(before) >= len(after) else (after, 'after')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200, help='#segments for aggregate stats')
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--radius', type=float, default=2.5)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--viz-n', type=int, default=6)
    ap.add_argument('--out', default='/tmp/select_cmp.png')
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    scenes = read_meta(ROOT); blocked = read_blacklist(ROOT)
    ours_cov, i3dm_cov, overlap_frac = [], [], []
    viz = []; n_scene = 0
    for chunk in scenes:
        if len(ours_cov) >= args.n:
            break
        if chunk.split('/')[-1] in blocked:
            continue
        sd = osp.join(ROOT, chunk)
        tj, pj = osp.join(sd, 'transforms.json'), osp.join(sd, 'prompts.json')
        if not (osp.isfile(tj) and osp.isfile(pj)):
            continue
        try:
            centers, faxis, w2c, K, w, h = load_scene(tj)
            prompts = json.load(open(pj))
        except Exception:
            continue
        N = centers.shape[0]
        for seg_key, seg in prompts.items():
            fi = seg.get('frame_idx')
            if not fi or len(fi) != 2:
                continue
            s, e = int(fi[0]), int(fi[1])
            if e > N or (e - s) < args.num_frames:
                continue
            side, side_name = longer_side(N, s, e)
            if not side:
                continue
            seg_scale = max(float(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean()), 1e-5)
            ours, _, _ = select_context(centers, faxis, w2c, K, w, h, s, e, args.k, args.radius)
            i3d = i3dm_select(centers, w2c, K, w, h, s, e, side, args.k, args.radius, rng)
            ov = len(set(ours) & set(i3d)) / max(len(ours), 1)
            ours_cov.append(viewpoint_coverage(centers, faxis, ours, s, e, seg_scale))
            i3dm_cov.append(viewpoint_coverage(centers, faxis, i3d, s, e, seg_scale))
            overlap_frac.append(ov)
            if len(viz) < args.viz_n:
                viz.append((chunk, centers, s, e, ours, i3d, side_name))
            if len(ours_cov) >= args.n:
                break
        n_scene += 1

    oc, ic, of = np.array(ours_cov), np.array(i3dm_cov), np.array(overlap_frac)
    print(f"segments={len(oc)} scenes={n_scene} K={args.k}")
    print(f"viewpoint-coverage of SELECTED views (our metric):")
    print(f"  ours  (frustum_cover+dir): mean={oc.mean():.3f} median={np.median(oc):.3f}")
    print(f"  i3dm  (MC FOV-overlap)   : mean={ic.mean():.3f} median={np.median(ic):.3f}")
    print(f"  ours>i3dm: {100*(oc>ic).mean():.1f}%  equal: {100*(oc==ic).mean():.1f}%  "
          f"i3dm>ours: {100*(ic>oc).mean():.1f}%")
    print(f"  mean picked-index overlap ours vs i3dm: {of.mean():.2f} "
          f"({of.mean()*args.k:.1f}/{args.k} views shared)")

    # viz: per-segment top-down, ours vs i3dm picks
    n = len(viz); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
    for i, (chunk, centers, s, e, ours, i3d, side_name) in enumerate(viz):
        a = axes[i // ncol][i % ncol]
        ax0, ax1 = np.argsort(centers.var(0))[-2:]
        a.scatter(centers[:, ax0], centers[:, ax1], s=5, c='lightgray')
        a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=0.5, alpha=0.5)
        a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=12, c='tab:blue', label='target')
        a.scatter(centers[list(ours), ax0], centers[list(ours), ax1], s=110, marker='*',
                  facecolors='none', edgecolors='tab:red', linewidths=1.6, label='ours')
        a.scatter(centers[list(i3d), ax0], centers[list(i3d), ax1], s=42, marker='x',
                  c='tab:green', linewidths=1.8, label='i3dm')
        a.scatter([centers[s, ax0]], [centers[s, ax1]], s=45, c='k', marker='s', label='query=s')
        a.set_title(f"{chunk.split('/')[-1][:8]} {side_name}", fontsize=9)
        a.set_aspect('equal', 'datalim')
        if i == 0:
            a.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle("Context selection: ours (red ★ frustum_cover+dir) vs I3DM-style "
                 "(green × MC FOV-overlap, query=first frame)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(args.out, dpi=115)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
