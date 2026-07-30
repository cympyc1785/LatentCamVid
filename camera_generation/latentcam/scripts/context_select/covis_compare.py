"""Compare covis retrieval: CURRENT (FPS on center, tight gate) vs IMPROVED
(direction-around-X FPS + widened radius + relaxed axis) over N scenes.

Clustering cause: candidates are gated to see one anchor look-at X, look the same
way, within a tight radius -> the candidate POOL is already clustered, so FPS on
raw center can't spread it. Fix: (1) widen radius / axis so the pool spans more
viewpoints, (2) diversify on the *viewing direction toward X* (azimuth around the
scene point) so picks surround the scene point instead of bunching.
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, csv, json, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# image dir: 'images_4' (DL3DV-960, 960x540) or 'images_8' (DL3DV-480, 480x270)
def _img_dir(sd, names=('images_4', 'images_8', 'images')):
    for c in names:
        p = osp.join(sd, c)
        if osp.isdir(p):
            return p
    return None


_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def read_meta(root):
    with open(osp.join(root, 'meta.csv'), newline='') as f:
        return [r['chunk'].strip() for r in csv.DictReader(f)]


def read_blacklist(root):
    p = osp.join(root, 'blacklist.csv'); s = set()
    if osp.isfile(p):
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                s.add(r['scene'].strip())
    return s


def load_scene(scene_dir):
    tj = json.load(open(osp.join(scene_dir, 'transforms.json')))
    w, h = int(tj['w']), int(tj['h'])
    K = np.array([[tj['fl_x'], 0, tj['cx']], [0, tj['fl_y'], tj['cy']], [0, 0, 1]])
    frames = sorted(tj['frames'], key=lambda fr: fr['file_path'])
    c2w = np.array([fr['transform_matrix'] for fr in frames], dtype=np.float64) @ _GL2CV
    _d = _img_dir(scene_dir)
    files = [osp.join(_d, osp.basename(fr['file_path'])) for fr in frames]
    return dict(centers=c2w[:, :3, 3], faxis=c2w[:, :3, 2], w2c=np.linalg.inv(c2w),
                K=K, w=w, h=h, files=files)


def pw_gauss(th, t0=10.0, s1=5.0, s2=15.0):
    d = th - t0; sig = np.where(th <= t0, s1, s2)
    return np.exp(-(d * d) / (2 * sig * sig))


def gate_and_score(sc, s, e, radius, max_axis_deg):
    """return list of (score, idx, dir_unit_toward_camera_from_X) for gated candidates."""
    centers, faxis, w2c, K, w, h = sc['centers'], sc['faxis'], sc['w2c'], sc['K'], sc['w'], sc['h']
    c_s, f_s = centers[s], faxis[s]
    seg_scale = np.linalg.norm(centers[s:e] - c_s, axis=1).mean()
    centroid = centers.mean(0)
    d = max(float(np.dot(centroid - c_s, f_s)), 0.5 * seg_scale)
    X = c_s + d * f_s
    cos_max = np.cos(np.deg2rad(max_axis_deg))
    out = []
    for j in range(centers.shape[0]):
        if s <= j < e:
            continue
        if np.linalg.norm(centers[j] - c_s) > radius * seg_scale:
            continue
        if np.dot(f_s, faxis[j]) < cos_max:
            continue
        Xc = w2c[j, :3, :3] @ X + w2c[j, :3, 3]
        if Xc[2] <= 1e-6:
            continue
        uv = K @ (Xc / Xc[2])
        if not (0 <= uv[0] < w and 0 <= uv[1] < h):
            continue
        v1, v2 = c_s - X, centers[j] - X
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        theta = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
        dir_from_X = (centers[j] - X) / (np.linalg.norm(centers[j] - X) + 1e-9)
        out.append([float(pw_gauss(theta, 10.0)), j, dir_from_X, centers[j], seg_scale])
    return X, seg_scale, out


def fps(feats, k, start=0):
    picks = [start]
    while len(picks) < min(k, len(feats)):
        dmin = np.min([np.linalg.norm(feats - feats[p], axis=1) for p in picks], axis=0)
        for p in picks:
            dmin[p] = -1.0
        picks.append(int(np.argmax(dmin)))
    return picks


def select_current(cand, k, topM=32):
    cand = sorted(cand, key=lambda x: -x[0])[:topM]
    if not cand:
        return []
    feats = np.stack([np.concatenate([c[3] / c[4], c[2]]) for c in cand])   # center + axis
    return [cand[p][1] for p in fps(feats, k)]


def select_improved(cand, k):
    if not cand:
        return []
    cand = sorted(cand, key=lambda x: -x[0])          # start from best score
    dirs = np.stack([c[2] for c in cand])             # direction from X to camera (azimuth)
    return [cand[p][1] for p in fps(dirs, k, start=0)]


def min_pair_angle(sc, X, idxs):
    """min pairwise angular separation (deg) of viewing directions toward X — spread metric."""
    if len(idxs) < 2:
        return float('nan')
    dv = sc['centers'][idxs] - X
    dv = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
    ang = []
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            ang.append(np.rad2deg(np.arccos(np.clip(dv[i] @ dv[j], -1, 1))))
    return float(np.min(ang))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K')
    ap.add_argument('--n-scenes', type=int, default=3)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--out', default='/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/scripts/covis_compare.png')
    args = ap.parse_args()

    scenes = read_meta(args.root); blk = read_blacklist(args.root)
    picked = []
    for chunk in scenes:
        if len(picked) >= args.n_scenes:
            break
        if chunk.split('/')[-1] in blk:
            continue
        sd = osp.join(args.root, chunk)
        if not (osp.isfile(osp.join(sd, 'transforms.json')) and osp.isfile(osp.join(sd, 'prompts.json'))):
            continue
        pr = json.load(open(osp.join(sd, 'prompts.json')))
        sc = load_scene(sd); N = sc['centers'].shape[0]
        for kkey, seg in pr.items():
            fi = seg.get('frame_idx')
            if fi and len(fi) == 2 and int(fi[1]) <= N and int(fi[1]) - int(fi[0]) >= args.num_frames:
                picked.append((chunk, sc, kkey, int(fi[0]), int(fi[1]))); break

    fig, axes = plt.subplots(args.n_scenes, 2, figsize=(13, 5 * args.n_scenes))
    if args.n_scenes == 1:
        axes = axes[None, :]

    for row, (chunk, sc, kkey, s, e) in enumerate(picked):
        span = min(args.num_frames // 8, e - 1 - s)
        inseg = [s + int(round(i)) for i in np.linspace(0, span, args.k)]
        # CURRENT: tight gate (R=1.0, axis 60), FPS on center
        Xc, scale_c, cand_c = gate_and_score(sc, s, e, radius=1.0, max_axis_deg=60.0)
        cur = select_current(cand_c, args.k)
        # IMPROVED: wide gate (R=2.0, axis 80), FPS on direction-around-X
        Xi, scale_i, cand_i = gate_and_score(sc, s, e, radius=2.0, max_axis_deg=80.0)
        imp = select_improved(cand_i, args.k)

        print(f"[{row}] {chunk}  seg[{s}:{e}]  cand(cur/imp)={len(cand_c)}/{len(cand_i)}")
        print(f"     in-seg {inseg}")
        print(f"     CURRENT  covis {cur}  minPairAngle={min_pair_angle(sc, Xc, cur):.1f}deg")
        print(f"     IMPROVED covis {imp}  minPairAngle={min_pair_angle(sc, Xi, imp):.1f}deg")

        centers = sc['centers']
        C0 = centers - centers.mean(0)
        Vt = np.linalg.svd(C0, full_matrices=False)[2]
        P = C0 @ Vt[:2].T
        for col, (title, idxs, Xpt) in enumerate([
                ('CURRENT (tight+center-FPS)', cur, Xc),
                ('IMPROVED (wide+dir-FPS)', imp, Xi)]):
            ax = axes[row, col]
            ax.scatter(P[:, 0], P[:, 1], s=6, c='lightgray')
            ax.plot(P[s:e, 0], P[s:e, 1], '-o', ms=2, c='tab:blue')
            ax.scatter(P[inseg, 0], P[inseg, 1], s=90, marker='s', c='tab:purple', zorder=4)
            ax.scatter(P[idxs, 0], P[idxs, 1], s=130, marker='X', c='tab:red', zorder=5)
            ax.scatter(*P[s], s=230, marker='*', c='tab:green', zorder=6)
            Xp = (Xpt - centers.mean(0)) @ Vt[:2].T
            ax.scatter(*Xp, s=90, marker='P', c='orange', zorder=5)
            for j in idxs:
                ax.annotate(str(j), P[j], fontsize=8, color='tab:red')
            ang = min_pair_angle(sc, Xpt, idxs)
            ax.set_title(f"scene{row} seg[{s}:{e}] {title}\nminPairAngle={ang:.0f}deg",
                         fontsize=9)
            ax.set_aspect('equal', 'datalim')

    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    print("saved:", args.out)


if __name__ == '__main__':
    main()
