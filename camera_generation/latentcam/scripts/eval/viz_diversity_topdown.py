"""traj_diversity.py 가 낸 숫자를 top-down 으로 그린다.

두 종류의 패널을 만든다.

  --fig seeds : 같은 context 에서 seed 만 바꾼 8개 궤적 + GT + context 카메라.
                "spread 가 GT 이동거리의 36%" 가 눈으로 어떻게 보이는지.
  --fig swap  : context 를 같은 scene 의 다른 segment 로 옮겼을 때
                base(주황) vs swap(자홍) 궤적 + 원래 context(초록 사각) vs
                donor context(빨강 삼각). 궤적이 donor 쪽으로 끌려가는지.

좌표계: target 첫 프레임 s 의 카메라 좌표계로 anchor 후 X vs -Z 투영
        (viz_avgscale_context_topdown.py / topdown_swap.py 와 같은 관례).
거리 단위: 그 segment 의 GT reach L = max_t||T_gt(t)-T_gt(0)|| 로 나눠 무차원화.
        점선 원 r=1 이 GT 가 닿는 최대 반경이다.

  python scripts/eval/viz_diversity_topdown.py --fig seeds \
      --seeds eval_my/divers/seed0 ... --out eval_my/divers/viz
  python scripts/eval/viz_diversity_topdown.py --fig swap \
      --base eval_my/divers/swap_base --swap eval_my/divers/swap_inscene \
      --out eval_my/divers/viz
"""
import argparse
import glob
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _tr(run_dir, name, kind):
    p = osp.join(run_dir, 'test', f'{name}_transforms_{kind}.json')
    if not osp.exists(p):
        return None
    return np.asarray([f['transform_matrix']
                       for f in json.load(open(p))['frames']], dtype=np.float64)


def _ctx(run_dir, name):
    p = osp.join(run_dir, 'geo_ctx', f'{name}.json')
    if not osp.exists(p):
        return None
    return np.asarray(json.load(open(p))['c2w'], dtype=np.float64)


def _names(run_dir):
    suf = '_transforms_pred.json'
    return sorted(osp.basename(p)[:-len(suf)]
                  for p in glob.glob(osp.join(run_dir, 'test', '*' + suf)))


def _frame(gt):
    """GT 첫 카메라 좌표계로 보내는 4x4 와, 그 프레임에서 잰 GT reach."""
    ref = np.linalg.inv(gt[0])
    L = float(np.linalg.norm((ref @ gt)[:, :3, 3], axis=-1).max())
    return ref, max(L, 1e-6)


def _xz(mats, ref, L):
    """(N,4,4) -> (N,2) top-down 좌표 (X, -Z), GT reach 로 정규화."""
    c = np.einsum('ij,njk->nik', ref, mats)[:, :3, 3] / L
    return np.stack([c[:, 0], -c[:, 2]], axis=-1)


def _finish(ax, title):
    ax.add_artist(plt.Circle((0, 0), 1.0, fill=False, ls=':', lw=1.0, color='0.55'))
    ax.scatter([0], [0], c='k', s=80, marker='*', zorder=7)      # frame s (anchor)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    # 정사각 bbox 를 직접 잡는다. aspect='datalim' 은 set_xlim 을 무시하므로 'box' 를 쓴다.
    # r=1 원이 항상 보이도록 최소 범위를 보장 (GT reach 를 눈금으로 쓰기 위해).
    bb = ax.dataLim
    x0, x1 = min(bb.x0, -1.12), max(bb.x1, 1.12)
    y0, y1 = min(bb.y0, -1.12), max(bb.y1, 1.12)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    r = 0.53 * max(x1 - x0, y1 - y0)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect('equal', adjustable='box')


def fig_seeds(run_dirs, names, out_path, ctx_from=None, style='cloud'):
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    cmap = plt.get_cmap('turbo')
    for k, nm in enumerate(names):
        ax = axes[k // cols][k % cols]
        gt = _tr(run_dirs[0], nm, 'ref')
        ref, L = _frame(gt)
        for j, d in enumerate(run_dirs):
            p = _tr(d, nm, 'pred')
            if p is None:
                continue
            xy = _xz(p, ref, L)
            if style == 'cloud':
                col, lw, alpha = 'tab:red', 1.1, 0.5
                lab = 'pred (8 seeds)' if (k == 0 and j == 0) else None
            else:
                col = cmap(0.08 + 0.84 * j / max(len(run_dirs) - 1, 1))
                lw, alpha = 1.3, 0.95
                lab = f'seed{j}' if k == 0 else None
            ax.plot(xy[:, 0], xy[:, 1], '-', lw=lw, alpha=alpha, color=col, label=lab)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=16, color=col, alpha=alpha, zorder=6)
        g = _xz(gt, ref, L)
        ax.plot(g[:, 0], g[:, 1], '-', lw=3.0, color='tab:blue',
                label='GT' if k == 0 else None, zorder=5)
        ax.scatter(g[-1, 0], g[-1, 1], s=55, color='tab:blue', marker='o', zorder=6)
        if ctx_from is not None:
            C = _ctx(ctx_from, nm)
            if C is not None:
                c = _xz(C, ref, L)
                ax.scatter(c[:, 0], c[:, 1], marker='s', s=52, facecolors='none',
                           edgecolors='tab:green', lw=1.6, zorder=7,
                           label='context view' if k == 0 else None)
        _finish(ax, nm[:26])
        if k == 0:
            ax.legend(fontsize=8, loc='best')
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle('seed diversity @ fixed context — top-down (anchored to frame s; X vs -Z; '
                 'GT reach = 1, dotted circle)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


def fig_swap(base, swap, names, out_path):
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    for k, nm in enumerate(names):
        ax = axes[k // cols][k % cols]
        gt = _tr(base, nm, 'ref')
        ref, L = _frame(gt)
        for C, mk, col, lab in ((_ctx(base, nm), 's', 'tab:green', 'context (orig)'),
                                (_ctx(swap, nm), '^', 'tab:red', 'context (donor)')):
            if C is None:
                continue
            c = _xz(C, ref, L)
            ax.scatter(c[:, 0], c[:, 1], marker=mk, s=58, facecolors='none',
                       edgecolors=col, lw=1.7, zorder=7, label=lab if k == 0 else None)
        for d, col, lab in ((base, 'tab:orange', 'pred (orig ctx)'),
                            (swap, 'tab:purple', 'pred (donor ctx)')):
            p = _tr(d, nm, 'pred')
            if p is None:
                continue
            xy = _xz(p, ref, L)
            ax.plot(xy[:, 0], xy[:, 1], '-', lw=2.0, color=col,
                    label=lab if k == 0 else None)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=34, color=col, zorder=6)
        g = _xz(gt, ref, L)
        ax.plot(g[:, 0], g[:, 1], '-', lw=3.0, color='tab:blue',
                label='GT' if k == 0 else None, zorder=5)
        ax.scatter(g[-1, 0], g[-1, 1], s=55, color='tab:blue', zorder=6)
        pb, ps = _tr(base, nm, 'pred'), _tr(swap, nm, 'pred')
        dsw = float(np.linalg.norm(_xz(pb, ref, L) - _xz(ps, ref, L), axis=-1).mean())
        _finish(ax, f'{nm[:22]}   d_swap(2D)={dsw:.3f}')
        if k == 0:
            ax.legend(fontsize=8, loc='best')
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle('context swap (same scene, other segment) — top-down (anchored to frame s; '
                 'X vs -Z; GT reach = 1)', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


def pick(names, mode, n, base=None, swap=None):
    if mode == 'even':
        idx = np.linspace(0, len(names) - 1, n).round().astype(int)
        return [names[i] for i in sorted(set(idx.tolist()))]
    if mode == 'dswap-top' or mode == 'dswap-low':
        sc = []
        for nm in names:
            gt, pb, ps = _tr(base, nm, 'ref'), _tr(base, nm, 'pred'), _tr(swap, nm, 'pred')
            if gt is None or pb is None or ps is None:
                continue
            ref, L = _frame(gt)
            sc.append((float(np.linalg.norm((pb[:, :3, 3] - ps[:, :3, 3]) / L, axis=-1).mean()), nm))
        sc.sort(reverse=(mode == 'dswap-top'))
        return [nm for _, nm in sc[:n]]
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig', choices=['seeds', 'swap'], required=True)
    ap.add_argument('--seeds', nargs='*', default=[])
    ap.add_argument('--base', default=None)
    ap.add_argument('--swap', default=None)
    ap.add_argument('--ctx-from', default=None,
                    help='--fig seeds 에서 context 카메라를 가져올 run (geo_ctx/ 있는 것)')
    ap.add_argument('--pick', default='even', choices=['even', 'dswap-top', 'dswap-low'])
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--style', default='cloud', choices=['cloud', 'rainbow'])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.fig == 'seeds':
        names = _names(args.seeds[0])
        for d in args.seeds[1:]:
            names = [n for n in names if n in set(_names(d))]
        sel = pick(names, args.pick, args.n, args.base, args.swap)
        fig_seeds(args.seeds, sel, osp.join(args.out, f'seeds_{args.pick}_{args.style}{args.tag}.png'),
                  ctx_from=args.ctx_from, style=args.style)
    else:
        names = [n for n in _names(args.base) if n in set(_names(args.swap))]
        sel = pick(names, args.pick, args.n, args.base, args.swap)
        fig_swap(args.base, args.swap, sel,
                 osp.join(args.out, f'swap_{args.pick}{args.tag}.png'))


if __name__ == '__main__':
    main()
