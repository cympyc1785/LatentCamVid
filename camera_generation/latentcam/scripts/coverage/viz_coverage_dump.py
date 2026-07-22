"""Visualize the precomputed coverage dump (dump_coverage.py output). No re-scan.

Usage: python scripts/viz_coverage_dump.py --dump-dir /tmp/cov_dump --tau 0.6 --out /tmp/cov_stats.png
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os.path as osp, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', default='/tmp/cov_dump')
    ap.add_argument('--tau', type=float, default=0.6)
    ap.add_argument('--out', default='/tmp/cov_stats.png')
    args = ap.parse_args()

    covs, nout = [], []
    with open(osp.join(args.dump_dir, 'coverage_all.csv'), newline='') as f:
        for r in csv.DictReader(f):
            covs.append(float(r['coverage'])); nout.append(int(r['nout']))
    covs = np.array(covs); nout = np.array(nout)
    npz = np.load(osp.join(args.dump_dir, 'coverage_all.npz'))
    per_frame = npz['per_frame']                       # (S,49) uint8
    valid = per_frame.sum(1) >= 0                      # all rows
    pf = per_frame[valid].mean(0)                      # per-frame covered fraction

    taus = np.linspace(0, 1, 101)
    black_rate = np.array([(covs < t).mean() for t in taus])

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"DL3DV out-of-segment coverage  ({len(covs)} segments, "
                 f"rho={float(npz['rho'])}, angle={float(npz['angle'])}deg)", fontsize=13)

    a = ax[0, 0]
    a.hist(covs, bins=40, range=(0, 1), color='#4C72B0', edgecolor='white')
    a.axvline(args.tau, color='crimson', ls='--', lw=1.5, label=f'tau={args.tau}')
    a.set_yscale('log')
    a.set_title(f'Coverage histogram (log y; mean={covs.mean():.3f}, median={np.median(covs):.3f})')
    a.set_xlabel('out-of-segment coverage'); a.set_ylabel('# segments (log)'); a.legend()

    a = ax[0, 1]
    a.plot(taus, black_rate, color='#C44E52', lw=2)
    for t in [0.5, 0.6, 0.7, 0.8]:
        br = float((covs < t).mean())
        a.scatter([t], [br], color='gray', zorder=4, s=18)
        a.annotate(f'{br*100:.1f}%', (t, br), textcoords='offset points',
                   xytext=(4, 6), fontsize=8)
    a.axvline(args.tau, color='crimson', ls='--', lw=1)
    a.set_title('Blacklist rate vs tau'); a.set_xlabel('tau (min coverage)')
    a.set_ylabel('fraction blacklisted'); a.set_ylim(0, max(0.25, black_rate.max())); a.legend()

    a = ax[1, 0]
    xs = np.sort(covs); ys = np.arange(1, len(xs) + 1) / len(xs)
    a.plot(xs, ys, color='#55A868', lw=2)
    a.axvline(args.tau, color='crimson', ls='--', lw=1.5, label=f'tau={args.tau}')
    a.set_title('Coverage CDF'); a.set_xlabel('out-of-segment coverage')
    a.set_ylabel('cumulative fraction'); a.set_xlim(0, 1); a.set_ylim(0, 1); a.legend()

    a = ax[1, 1]
    a.plot(np.arange(len(pf)), pf, color='#8172B3', lw=2)
    a.fill_between(np.arange(len(pf)), pf, alpha=0.2, color='#8172B3')
    a.set_ylim(0.7, 1.0)
    a.set_title(f'Per-frame covered fraction ({len(per_frame)} segments)')
    a.set_xlabel('target-frame index (0..48)'); a.set_ylabel('frac segments covered')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
