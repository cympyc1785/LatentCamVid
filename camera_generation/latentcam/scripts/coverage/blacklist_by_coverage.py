"""Per-SEGMENT out-of-target coverage → blacklist.

Task-honest setup: context views must come from OUTSIDE the target segment [s:e] (no
cheating). This measures, for each 49-frame prompt segment, how much of the region the
TARGET trajectory observes can be covered by out-of-segment context camera frustums.
Segments whose max out-of-segment coverage < tau are blacklisted (uncoverable without
peeking at the target -> would create a train/test gap or force cheating).

Coverage (frustum-based, camera geometry only — no learned model):
  - target region P: for each target frame, its look-at point c+d*f at scene depth d
    (the scene the target actually observes).
  - candidates: frames OUTSIDE [s:e], within radius*seg_scale of the segment centroid.
  - cover(P) = P points inside a candidate frustum (project w2c+K, in image, 0<z<zmax).
  - coverage_union = |P covered by ANY out-of-seg candidate| / |P|   (max achievable)
  - coverage_k = |P covered by greedy top-K (=geo_num_views) out-of-seg views| / |P|
  blacklist if coverage_union < tau.

Usage: python scripts/blacklist_by_coverage.py --max-segments 10 [--tau 0.5 --k 6 --radius 2.5]
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, csv, json, argparse
import numpy as np
from tqdm import tqdm

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


def load_scene(tj_path):
    tj = json.load(open(tj_path))
    w, h = int(tj['w']), int(tj['h'])
    K = np.array([[tj['fl_x'], 0, tj['cx']], [0, tj['fl_y'], tj['cy']], [0, 0, 1]], float)
    fr = sorted(tj['frames'], key=lambda f: f['file_path'])
    c2w = np.array([f['transform_matrix'] for f in fr], float) @ _GL2CV   # OpenGL c2w -> OpenCV c2w
    centers = c2w[:, :3, 3]
    faxis = c2w[:, :3, 2]                                                  # OpenCV forward (+Z)
    w2c = np.linalg.inv(c2w)
    return centers, faxis, w2c, K, w, h


def seg_coverage(centers, faxis, w2c, K, w, h, s, e, rho, angle_deg):
    """Per-target-frame viewpoint coverage by OUT-OF-SEGMENT cameras.
    A target frame is 'covered' if some out-of-segment camera views the scene from a
    similar viewpoint — center within rho*seg_scale AND optical axis within angle_deg.
    (Frustum-containment saturates under wide FoV, so we require viewpoint proximity.)
    Returns (coverage_fraction, #covered, #target_frames, #out_of_seg_cands)."""
    N = centers.shape[0]
    seg_scale = max(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean(), 1e-5)
    tgt_c, tgt_f = centers[s:e], faxis[s:e]                     # (T,3)
    out = np.array([j for j in range(N) if not (s <= j < e)], dtype=int)
    if len(out) == 0:
        return 0.0, 0, tgt_c.shape[0], 0
    out_c, out_f = centers[out], faxis[out]                     # (M,3)
    cos_th = np.cos(np.deg2rad(angle_deg))

    dist = np.linalg.norm(tgt_c[:, None, :] - out_c[None, :, :], axis=-1) / seg_scale  # (T,M)
    ang = tgt_f @ out_f.T                                        # (T,M) cos between axes
    ok = (dist <= rho) & (ang >= cos_th)                        # (T,M)
    covered_mask = ok.any(axis=1)                              # (T,) frame covered?
    coverage = float(covered_mask.mean())
    return coverage, int(covered_mask.sum()), tgt_c.shape[0], len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K')
    ap.add_argument('--max-segments', type=int, default=10)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--rho', type=float, default=1.0)       # position radius (x seg_scale)
    ap.add_argument('--angle', type=float, default=60.0)    # max optical-axis angle (deg)
    ap.add_argument('--tau', type=float, default=0.6)       # min covered-frame fraction
    ap.add_argument('--out', default=None)   # write blacklist csv if given
    args = ap.parse_args()

    scenes = read_meta(args.root)
    blocked = read_blacklist(args.root)
    rows, black = [], []
    done = 0
    print(f"{'scene/seg':>28} | {'coverage':>8} {'cov/T':>7} {'#out':>5} | blacklist")
    for chunk in scenes:
        if done >= args.max_segments:
            break
        if chunk.split('/')[-1] in blocked:
            continue
        sd = osp.join(args.root, chunk)
        tj = osp.join(sd, 'transforms.json'); pj = osp.join(sd, 'prompts.json')
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
            cov, ncov, ntgt, nout = seg_coverage(centers, faxis, w2c, K, w, h, s, e, args.rho, args.angle)
            is_black = cov < args.tau
            name = f"{chunk.split('/')[-1][:16]}_{seg_key}"
            print(f"{name:>28} | {cov:>8.2f} {ncov:>3}/{ntgt:<3} {nout:>5} | {'YES' if is_black else ''}")
            rows.append((chunk, seg_key, cov))
            if is_black:
                black.append((chunk, seg_key))
            done += 1
            if done >= args.max_segments:
                break

    cov_all = np.array([r[2] for r in rows])
    print("-" * 70)
    print(f"segments={len(rows)} | coverage mean={cov_all.mean():.2f} | "
          f"blacklisted (coverage<{args.tau})={len(black)}  (rho={args.rho}, angle={args.angle})")
    if args.out and black:
        with open(args.out, 'w', newline='') as f:
            wtr = csv.writer(f); wtr.writerow(['scene', 'segment'])
            for chunk, sk in black:
                wtr.writerow([chunk, sk])
        print(f"wrote blacklist: {args.out} ({len(black)} segments)")


if __name__ == '__main__':
    main()
