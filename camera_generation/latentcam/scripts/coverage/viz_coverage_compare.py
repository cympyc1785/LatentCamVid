"""Compare 'both' vs 'longer' context-side coverage dumps.

Usage: python scripts/viz_coverage_compare.py --both /tmp/cov_dump --longer /tmp/cov_dump_longer --out /tmp/cov_compare.png
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


def load(dd):
    covs = []
    with open(osp.join(dd, 'coverage_all.csv'), newline='') as f:
        for r in csv.DictReader(f):
            covs.append(float(r['coverage']))
    npz = np.load(osp.join(dd, 'coverage_all.npz'))
    return np.array(covs), npz['per_frame'].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--both', default='/tmp/cov_dump')
    ap.add_argument('--longer', default='/tmp/cov_dump_longer')
    ap.add_argument('--tau', type=float, default=0.6)
    ap.add_argument('--out', default='/tmp/cov_compare.png')
    args = ap.parse_args()

    cb, pfb = load(args.both)
    cl, pfl = load(args.longer)
    taus = np.linspace(0, 1, 101)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Context side: both (before+after, interpolation) vs '
                 f'longer (longer side only, extrapolation)  |  {len(cb)} segments', fontsize=13)

    a = ax[0, 0]
    bins = np.linspace(0, 1, 41)
    a.hist(cb, bins=bins, color='#4C72B0', alpha=0.6, label=f'both (mean={cb.mean():.3f})')
    a.hist(cl, bins=bins, color='#C44E52', alpha=0.6, label=f'longer (mean={cl.mean():.3f})')
    a.set_yscale('log'); a.axvline(args.tau, color='k', ls='--', lw=1)
    a.set_title('Coverage histogram (log y)'); a.set_xlabel('coverage')
    a.set_ylabel('# segments (log)'); a.legend()

    a = ax[0, 1]
    a.plot(taus, [(cb < t).mean() for t in taus], color='#4C72B0', lw=2, label='both')
    a.plot(taus, [(cl < t).mean() for t in taus], color='#C44E52', lw=2, label='longer')
    for c, col in [(cb, '#4C72B0'), (cl, '#C44E52')]:
        br = float((c < args.tau).mean())
        a.scatter([args.tau], [br], color=col, zorder=5)
        a.annotate(f'{br*100:.1f}%', (args.tau, br), textcoords='offset points',
                   xytext=(6, 0), fontsize=9, color=col)
    a.axvline(args.tau, color='k', ls='--', lw=1)
    a.set_title('Blacklist rate vs tau'); a.set_xlabel('tau'); a.set_ylabel('frac blacklisted')
    a.set_ylim(0, 0.35); a.legend()

    a = ax[1, 0]
    for c, col, lab in [(cb, '#4C72B0', 'both'), (cl, '#C44E52', 'longer')]:
        xs = np.sort(c); ys = np.arange(1, len(xs) + 1) / len(xs)
        a.plot(xs, ys, color=col, lw=2, label=lab)
    a.axvline(args.tau, color='k', ls='--', lw=1)
    a.set_title('Coverage CDF'); a.set_xlabel('coverage'); a.set_ylabel('cumulative fraction')
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.legend()

    a = ax[1, 1]
    a.plot(np.arange(len(pfb)), pfb, color='#4C72B0', lw=2, label='both')
    a.plot(np.arange(len(pfl)), pfl, color='#C44E52', lw=2, label='longer')
    a.set_title('Per-frame covered fraction'); a.set_xlabel('target-frame index (0..48)')
    a.set_ylabel('frac segments covered'); a.set_ylim(0.4, 1.0); a.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
