"""Coverage dump using the K frustum_cover-SELECTED out-of-segment context views (what the
geo encoder actually sees), NOT the full out-of-segment union. Matches render_scene_stitched:
  context = frustum_cover(allowed=longer side) -> K views; coverage = viewpoint coverage of
  the target frames by those K views (position rho + optical-axis angle).

Pure camera geometry (numpy), CPU. Writes coverage_selk.csv (scene, segment, coverage, side).

Usage: python scripts/dump_coverage_selk.py --k 6 --radius 2.5 --out /tmp/cov_dump_selk/coverage_all.csv
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, csv, json, argparse
import numpy as np
from tqdm import tqdm

SCR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCR)
from blacklist_by_coverage import read_meta, read_blacklist, load_scene  # numpy-only


def frustum_cover_select(centers, faxis, w2c, K, w, h, anchor, seg_scale,
                         k=6, radius=2.5, allowed=None, look_centroid=None, ball_center=None):
    N = centers.shape[0]; c_a = centers[anchor]; rad = radius * seg_scale
    pool = range(N) if allowed is None else list(allowed)
    cand = [j for j in pool if np.linalg.norm(centers[j] - c_a) <= rad]
    if not cand:
        cand = list(pool) if allowed is not None and len(list(pool)) else [anchor]
    if ball_center is not None:
        ball_c = np.asarray(ball_center, dtype=centers.dtype)
    else:
        scene_c = centers.mean(0) if look_centroid is None else look_centroid
        look = np.stack([centers[j] + max(float(np.dot(scene_c - centers[j], faxis[j])),
                                          0.3 * seg_scale) * faxis[j] for j in cand])
        ball_c = look.mean(0)
    g = np.linspace(-rad, rad, 16); gx, gy, gz = np.meshgrid(g, g, g, indexing='ij')
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1) + ball_c
    P = P[np.linalg.norm(P - ball_c, axis=1) <= rad]
    if P.shape[0] == 0:
        P = ball_c[None]
    zmax = float(np.linalg.norm(centers[cand] - ball_c, axis=1).max()) + rad

    def covered(j):
        Xc = (w2c[j, :3, :3] @ P.T).T + w2c[j, :3, 3]; z = Xc[:, 2]
        uv = (K @ (Xc / np.clip(z, 1e-6, None)[:, None]).T).T
        ok = (z > 1e-6) & (z < zmax) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return set()
        dirv = centers[j] - P[idx]; dirv = dirv / (np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-9)
        b = np.round(dirv * 1.5).astype(int)
        return set((int(i), int(b[t, 0]), int(b[t, 1]), int(b[t, 2])) for t, i in enumerate(idx))

    cov = {j: covered(j) for j in cand}
    picked, union = [], set()
    while len(picked) < min(k, len(cand)):
        best_j = max((j for j in cand if j not in picked),
                     key=lambda x: len(cov[x] - union), default=None)
        if best_j is None:
            break
        if len(cov[best_j] - union) == 0:
            feat = {x: np.concatenate([centers[x] / max(seg_scale, 1e-5), faxis[x]]) for x in cand}
            remaining = [x for x in cand if x not in picked]
            if not picked and remaining:
                picked.append(min(remaining, key=lambda x: np.linalg.norm(centers[x] - c_a)))
                remaining.remove(picked[-1])
            while len(picked) < min(k, len(cand)) and remaining:
                nxt = max(remaining, key=lambda x: min(np.linalg.norm(feat[x] - feat[p]) for p in picked))
                picked.append(nxt); remaining.remove(nxt)
            break
        picked.append(best_j); union |= cov[best_j]
    return picked


def _context_scale(centers, side, num_frames):
    """Mean per-window camera movement over context `side` (num_frames chunks). Target-free."""
    T = num_frames
    chunks = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]
    if not chunks and len(side) >= 2:
        chunks = [side]
    if not chunks:
        return None
    sc = [float(np.linalg.norm(centers[ch] - centers[ch[0]], axis=1).mean()) for ch in chunks]
    return max(float(np.mean(sc)), 1e-5)


def select_context(centers, faxis, w2c, K, w, h, s, e, k, radius, first_frame=True,
                   num_frames=49, first_view_s=False):
    """Honest (first_frame=True): anchor = target's FIRST frame s only; radius from CONTEXT
    scale; look_centroid = candidate mean -> selection never uses target [s+1:e].
    first_view_s=True: view0 = frame s + (k-1) out-of-seg retrieved (matches training's
    geo_first_view_target_s). Legacy (False): anchor near target midpoint."""
    N = centers.shape[0]
    before, after = list(range(0, s)), list(range(e, N))
    side = before if len(before) >= len(after) else after
    side_name = 'before' if len(before) >= len(after) else 'after'
    seg_scale = max(float(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean()), 1e-5)
    if not side:
        return ([s] if first_view_s else []), side_name, seg_scale
    k_retr = (k - 1) if first_view_s else k
    if first_frame:
        anchor = s
        cs = _context_scale(centers, side, num_frames)
        scale = cs if cs is not None else seg_scale
        look_centroid, ball_center = None, centers[s]   # omnidirectional ball around frame s
    else:
        mid = (s + e) // 2
        anchor = min(side, key=lambda j: np.linalg.norm(centers[j] - centers[mid]))
        scale, look_centroid, ball_center = seg_scale, None, None
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor, scale,
                                 k=k_retr, radius=radius, allowed=side, look_centroid=look_centroid,
                                 ball_center=ball_center)
    if len(picks) < k_retr:
        for j in [side[int(round(x))] for x in np.linspace(0, len(side) - 1, k_retr)]:
            if j not in picks:
                picks.append(j)
            if len(picks) >= k_retr:
                break
    picks = picks[:k_retr]
    if first_view_s:
        picks = [s] + picks
    return picks[:k], side_name, seg_scale


def viewpoint_coverage(centers, faxis, ctx_idx, s, e, seg_scale, rho=1.0, angle=60.0):
    if not len(ctx_idx):
        return 0.0
    tgt_c, tgt_f = centers[s:e], faxis[s:e]
    oc, of = centers[list(ctx_idx)], faxis[list(ctx_idx)]
    dist = np.linalg.norm(tgt_c[:, None] - oc[None], axis=-1) / seg_scale
    ang = tgt_f @ of.T
    return float(((dist <= rho) & (ang >= np.cos(np.deg2rad(angle)))).any(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/scenes')
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--radius', type=float, default=2.5)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--first-frame', type=int, default=1,
                    help='1 = honest (anchor=first frame + context scale); 0 = legacy target-mid')
    ap.add_argument('--first-view-s', type=int, default=0,
                    help='1 = view0 = frame s + (k-1) out-of-seg; coverage measured over [s+1:e]')
    ap.add_argument('--max-scenes', type=int, default=None)
    ap.add_argument('--out', default='/tmp/cov_dump_selk/coverage_all.csv')
    args = ap.parse_args()
    os.makedirs(osp.dirname(args.out), exist_ok=True)

    scenes = read_meta(args.root); blocked = read_blacklist(args.root)
    rows = []; n_scene = 0
    for chunk in tqdm(scenes, desc='scenes'):
        if args.max_scenes and n_scene >= args.max_scenes:
            break
        if chunk.split('/')[-1] in blocked:
            continue
        sd = osp.join(args.root, chunk)
        tj, pj = osp.join(sd, 'transforms.json'), osp.join(sd, 'prompts.json')
        if not (osp.isfile(tj) and osp.isfile(pj)):
            continue
        try:
            centers, faxis, w2c, K, w, h = load_scene(tj)
            prompts = json.load(open(pj))
        except Exception:
            continue
        N = centers.shape[0]; used = False
        for seg_key, seg in prompts.items():
            fi = seg.get('frame_idx')
            if not fi or len(fi) != 2:
                continue
            s, e = int(fi[0]), int(fi[1])
            if e > N or (e - s) < args.num_frames:
                continue
            ctx, side_name, seg_scale = select_context(centers, faxis, w2c, K, w, h,
                                                       s, e, args.k, args.radius,
                                                       first_frame=bool(args.first_frame),
                                                       num_frames=args.num_frames,
                                                       first_view_s=bool(args.first_view_s))
            cov_start = (s + 1) if args.first_view_s else s   # target: exclude self-covered s
            # covering set: when first_view_s, exclude frame s too so coverage reflects only the
            # OUT-OF-SEG context (frame s trivially covers adjacent frames -> inflates otherwise).
            cover_ctx = ctx[1:] if (args.first_view_s and len(ctx) > 1) else ctx
            cov = viewpoint_coverage(centers, faxis, cover_ctx, cov_start, e, seg_scale)
            rows.append((chunk, seg_key, cov, side_name)); used = True
        if used:
            n_scene += 1

    with open(args.out, 'w', newline='') as f:
        wtr = csv.writer(f); wtr.writerow(['scene', 'segment', 'coverage', 'side'])
        wtr.writerows(rows)
    covs = np.array([r[2] for r in rows])
    print(f"\nscenes={n_scene} segments={len(rows)} k={args.k} radius={args.radius}")
    print(f"coverage(selK): mean={covs.mean():.4f} median={np.median(covs):.4f} "
          f"min={covs.min():.4f} std={covs.std():.4f}  -> {args.out}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"  tau={t:.1f} -> blacklist {float((covs < t).mean())*100:5.2f}% "
              f"({int((covs < t).sum())} seg)")


if __name__ == '__main__':
    main()
