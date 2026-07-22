"""Visualize out-of-target coverage distribution over many DL3DV segments.

Reuses blacklist_by_coverage.{load_scene, seg_coverage}. Sweeps a large sample of
prompt segments (out-of-segment viewpoint coverage), then renders:
  (1) histogram of per-segment coverage,
  (2) blacklist-rate vs tau curve,
  (3) coverage vs #covered-target-frames scatter,
  (4) per-frame covered heat: fraction of segments where target-frame t is covered.

Usage: python scripts/viz_coverage.py --max-scenes 300 --out /tmp/coverage_viz.png
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os.path as osp, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from blacklist_by_coverage import read_meta, read_blacklist, load_scene


def _context_idx(N, s, e, side):
    """Indices usable as context (outside target [s:e]).
      'both'   : everything outside [s:e] (앞+뒤, interpolation 가능 -> cheating).
      'longer' : only the LONGER of before([0:s]) / after([e:N]); forces extrapolation."""
    before = np.arange(0, s)
    after = np.arange(e, N)
    if side == 'both':
        return np.concatenate([before, after])
    return before if len(before) >= len(after) else after


def seg_coverage_full(centers, faxis, s, e, rho, angle_deg, side='both'):
    """Same metric as blacklist_by_coverage.seg_coverage but also returns the
    per-target-frame covered mask (T,) for the per-frame heat plot.
    side: 'both' | 'longer' (see _context_idx)."""
    seg_scale = max(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean(), 1e-5)
    tgt_c, tgt_f = centers[s:e], faxis[s:e]
    out = _context_idx(centers.shape[0], s, e, side)
    if len(out) == 0:
        return 0.0, np.zeros(tgt_c.shape[0], bool), 0
    out_c, out_f = centers[out], faxis[out]
    cos_th = np.cos(np.deg2rad(angle_deg))
    dist = np.linalg.norm(tgt_c[:, None, :] - out_c[None, :, :], axis=-1) / seg_scale
    ang = tgt_f @ out_f.T
    ok = (dist <= rho) & (ang >= cos_th)
    covered = ok.any(axis=1)
    return float(covered.mean()), covered, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K')
    ap.add_argument('--max-scenes', type=int, default=300)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--rho', type=float, default=1.0)
    ap.add_argument('--angle', type=float, default=60.0)
    ap.add_argument('--tau', type=float, default=0.6)
    ap.add_argument('--out', default='/tmp/coverage_viz.png')
    args = ap.parse_args()

    scenes = read_meta(args.root)
    blocked = read_blacklist(args.root)

    covs, ncovs, ntgts = [], [], []
    per_frame = np.zeros(args.num_frames)   # sum of covered masks
    per_frame_n = 0
    n_scene = 0
    for chunk in tqdm(scenes, desc='scenes'):
        if n_scene >= args.max_scenes:
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
        N = centers.shape[0]
        used = False
        for seg_key, seg in prompts.items():
            fi = seg.get('frame_idx')
            if not fi or len(fi) != 2:
                continue
            s, e = int(fi[0]), int(fi[1])
            if e > N or (e - s) < args.num_frames:
                continue
            cov, mask, nout = seg_coverage_full(centers, faxis, s, e, args.rho, args.angle)
            covs.append(cov); ncovs.append(int(mask.sum())); ntgts.append(mask.shape[0])
            if mask.shape[0] >= args.num_frames:
                per_frame += mask[:args.num_frames]; per_frame_n += 1
            used = True
        if used:
            n_scene += 1

    covs = np.array(covs)
    print(f"segments={len(covs)} scenes={n_scene} | coverage mean={covs.mean():.3f} "
          f"median={np.median(covs):.3f} min={covs.min():.3f} "
          f"frac<{args.tau}={float((covs < args.tau).mean()):.3f}")

    taus = np.linspace(0, 1, 101)
    black_rate = np.array([(covs < t).mean() for t in taus])

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"DL3DV out-of-segment coverage  (rho={args.rho}, angle={args.angle}deg, "
                 f"{len(covs)} segments / {n_scene} scenes)", fontsize=13)

    # (1) histogram
    a = ax[0, 0]
    a.hist(covs, bins=25, range=(0, 1), color='#4C72B0', edgecolor='white')
    a.axvline(args.tau, color='crimson', ls='--', lw=1.5, label=f'tau={args.tau}')
    a.set_title(f'Coverage histogram (mean={covs.mean():.3f}, median={np.median(covs):.3f})')
    a.set_xlabel('out-of-segment coverage'); a.set_ylabel('# segments'); a.legend()

    # (2) blacklist rate vs tau
    a = ax[0, 1]
    a.plot(taus, black_rate, color='#C44E52', lw=2)
    a.axvline(args.tau, color='gray', ls='--', lw=1)
    br = float((covs < args.tau).mean())
    a.scatter([args.tau], [br], color='crimson', zorder=5,
              label=f'tau={args.tau} -> {br*100:.1f}% blacklisted')
    a.set_title('Blacklist rate vs tau'); a.set_xlabel('tau (min coverage)')
    a.set_ylabel('fraction blacklisted'); a.set_ylim(0, 1); a.legend()

    # (3) coverage cdf
    a = ax[1, 0]
    xs = np.sort(covs); ys = np.arange(1, len(xs) + 1) / len(xs)
    a.plot(xs, ys, color='#55A868', lw=2)
    a.axvline(args.tau, color='crimson', ls='--', lw=1.5, label=f'tau={args.tau}')
    a.set_title('Coverage CDF'); a.set_xlabel('out-of-segment coverage')
    a.set_ylabel('cumulative fraction'); a.set_xlim(0, 1); a.legend()

    # (4) per target-frame covered fraction
    a = ax[1, 1]
    if per_frame_n > 0:
        pf = per_frame / per_frame_n
        a.plot(np.arange(args.num_frames), pf, color='#8172B3', lw=2)
        a.fill_between(np.arange(args.num_frames), pf, alpha=0.2, color='#8172B3')
        a.set_ylim(0, 1)
    a.set_title(f'Per-frame covered fraction ({per_frame_n} segments)')
    a.set_xlabel('target-frame index (0..48)'); a.set_ylabel('frac segments covered')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
