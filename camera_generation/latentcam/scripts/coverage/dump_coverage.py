"""Dump out-of-segment coverage for ALL DL3DV segments once, so tau can be chosen
statistically afterward (no re-scan needed). Pure camera geometry, CPU-only.

Writes:
  coverage_all.csv  : scene, segment, coverage, ncov, ntgt, nout, seg_scale
  coverage_all.npz  : per_frame (S,49) uint8 covered masks, index-aligned to csv rows
                      (only rows with ntgt>=49; others get an all-zero row + flag)

Usage: python scripts/dump_coverage.py --rho 1.0 --angle 60 --out-dir /tmp/cov_dump
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, csv, json, argparse
import numpy as np
from tqdm import tqdm

from blacklist_by_coverage import read_meta, read_blacklist, load_scene
from viz_coverage import seg_coverage_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/scenes')
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--rho', type=float, default=1.0)
    ap.add_argument('--angle', type=float, default=60.0)
    ap.add_argument('--side', default='longer', choices=['both', 'longer'],
                    help="context source: 'both'=앞+뒤(interpolation/cheat), "
                         "'longer'=긴 쪽만(extrapolation 강제)")
    ap.add_argument('--out-dir', default='/tmp/cov_dump')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenes = read_meta(args.root)
    blocked = read_blacklist(args.root)

    rows, masks = [], []
    n_scene = n_skip = 0
    for chunk in tqdm(scenes, desc='scenes'):
        if chunk.split('/')[-1] in blocked:
            continue
        sd = osp.join(args.root, chunk)
        tj, pj = osp.join(sd, 'transforms.json'), osp.join(sd, 'prompts.json')
        if not (osp.isfile(tj) and osp.isfile(pj)):
            n_skip += 1; continue
        try:
            centers, faxis, w2c, K, w, h = load_scene(tj)
            prompts = json.load(open(pj))
        except Exception:
            n_skip += 1; continue
        N = centers.shape[0]
        used = False
        for seg_key, seg in prompts.items():
            fi = seg.get('frame_idx')
            if not fi or len(fi) != 2:
                continue
            s, e = int(fi[0]), int(fi[1])
            if e > N or (e - s) < args.num_frames:
                continue
            seg_scale = float(max(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean(), 1e-5))
            cov, mask, nout = seg_coverage_full(centers, faxis, s, e, args.rho, args.angle, args.side)
            rows.append((chunk, seg_key, cov, int(mask.sum()), int(mask.shape[0]), nout, seg_scale))
            m = np.zeros(args.num_frames, np.uint8)
            if mask.shape[0] >= args.num_frames:
                m[:] = mask[:args.num_frames]
            masks.append(m)
            used = True
        if used:
            n_scene += 1

    csv_path = osp.join(args.out_dir, 'coverage_all.csv')
    with open(csv_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['scene', 'segment', 'coverage', 'ncov', 'ntgt', 'nout', 'seg_scale'])
        for r in rows:
            wtr.writerow(r)
    npz_path = osp.join(args.out_dir, 'coverage_all.npz')
    np.savez_compressed(npz_path, per_frame=np.stack(masks),
                        rho=args.rho, angle=args.angle, side=args.side)

    covs = np.array([r[2] for r in rows])
    print(f"\nside={args.side} scenes_used={n_scene} skipped={n_skip} segments={len(rows)}")
    print(f"csv={csv_path} npz={npz_path}")
    print(f"coverage: mean={covs.mean():.4f} median={np.median(covs):.4f} "
          f"min={covs.min():.4f} max={covs.max():.4f} std={covs.std():.4f}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"  tau={t:.1f} -> blacklist {float((covs < t).mean())*100:5.2f}%  "
              f"({int((covs < t).sum())} segments)")


if __name__ == '__main__':
    main()
