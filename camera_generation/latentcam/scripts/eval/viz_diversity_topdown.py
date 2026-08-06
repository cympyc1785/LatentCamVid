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


def _fwd_xz(mats, ref):
    """(N,4,4) -> (N,2) 카메라 forward 를 같은 top-down 평면에 투영.

    dump 된 transform_matrix 는 OpenGL c2w 이므로 forward = -R[:,2].
    XZ 평면에 그대로 투영하므로 위/아래를 보는 카메라일수록 화살표가 짧아진다
    (tilt 가 눈에 보이도록 일부러 정규화하지 않는다).
    """
    R = np.einsum('ij,njk->nik', ref, mats)[:, :3, :3]
    d = -R[:, :, 2]
    return np.stack([d[:, 0], -d[:, 2]], axis=-1)


def _arrows(ax, xy, mats, ref, color, every, scale, alpha=1.0, zorder=9):
    """궤적 위 `every` 프레임마다 카메라 시선 화살표."""
    if not every:
        return
    idx = np.arange(0, len(xy), every)
    if idx[-1] != len(xy) - 1:
        idx = np.append(idx, len(xy) - 1)
    d = _fwd_xz(mats, ref)[idx]
    ax.quiver(xy[idx, 0], xy[idx, 1], d[:, 0], d[:, 1], color=color, alpha=alpha,
              angles='xy', scale_units='xy', scale=1.0 / scale, width=0.006,
              headwidth=3.2, headlength=4.0, zorder=zorder)


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


def fig_seeds(run_dirs, names, out_path, ctx_from=None, style='cloud', ctx_leak_k=0):
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
                # geo_test_inseg_k 를 켰으면 _mix_inseg_context 가 누수 view 를 앞쪽 K 장에
                # 놓으므로(inseg + rest), 앞 K 장만 다른 마커로 구분해 그린다.
                kk = max(0, min(int(ctx_leak_k), c.shape[0]))
                if kk:
                    ax.scatter(c[:kk, 0], c[:kk, 1], marker='D', s=62, facecolors='none',
                               edgecolors='tab:green', lw=2.0, zorder=8,
                               label=f'context: IN-segment (leaked, K={kk})' if k == 0 else None)
                ax.scatter(c[kk:, 0], c[kk:, 1], marker='s', s=52, facecolors='none',
                           edgecolors='0.35', lw=1.6, zorder=7,
                           label='context: out-of-segment' if k == 0 else None)
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


def fig_swap(base, swap, names, out_path, arrow_every=0, arrow_scale=0.16,
             label_set='swap', swap_leak_k=0):
    """두 arm(base / swap)을 겹쳐 그린다. 같은 seed, context 만 다른 쌍을 보는 용도.

    label_set='swap' : arm C (같은 scene 다른 segment 로 donor context).
    label_set='leak' : geo_test_inseg_k 누수 arm. swap 쪽 context 앞 `swap_leak_k` 장이
                       target segment 안 카메라이므로 마커를 갈라 그린다.
    """
    LAB = {
        'swap': ('context (orig)', 'context (donor)',
                 'pred (orig ctx)', 'pred (donor ctx)'),
        'leak': ('context: out-of-segment', f'context: IN-segment (leaked, K={swap_leak_k})',
                 'pred (out-of-seg ctx)', f'pred (in-seg ctx, K={swap_leak_k})'),
    }[label_set]
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    for k, nm in enumerate(names):
        ax = axes[k // cols][k % cols]
        gt = _tr(base, nm, 'ref')
        ref, L = _frame(gt)
        for C, mk, col, lab, kk in ((_ctx(base, nm), 's', 'tab:green', LAB[0], 0),
                                    (_ctx(swap, nm), '^', 'tab:red', LAB[1], swap_leak_k)):
            if C is None:
                continue
            c = _xz(C, ref, L)
            kk = max(0, min(int(kk), c.shape[0]))
            if kk:
                # 누수 view 는 앞쪽 K 장 (_mix_inseg_context = inseg + rest)
                ax.scatter(c[:kk, 0], c[:kk, 1], marker='D', s=64, facecolors='none',
                           edgecolors=col, lw=2.0, zorder=8, label=lab if k == 0 else None)
                ax.scatter(c[kk:, 0], c[kk:, 1], marker=mk, s=58, facecolors='none',
                           edgecolors=col, lw=1.7, zorder=7, alpha=0.55,
                           label='context: out-of-segment (kept)' if k == 0 else None)
            else:
                ax.scatter(c[:, 0], c[:, 1], marker=mk, s=58, facecolors='none',
                           edgecolors=col, lw=1.7, zorder=7, label=lab if k == 0 else None)
            # context 카메라는 6장뿐이므로 every=1 로 전부 시선을 그린다.
            _arrows(ax, c, C, ref, col, 1 if arrow_every else 0, arrow_scale, alpha=0.85)
        for d, col, lab in ((base, 'tab:orange', LAB[2]),
                            (swap, 'tab:purple', LAB[3])):
            p = _tr(d, nm, 'pred')
            if p is None:
                continue
            xy = _xz(p, ref, L)
            ax.plot(xy[:, 0], xy[:, 1], '-', lw=2.0, color=col,
                    label=lab if k == 0 else None)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=34, color=col, zorder=6)
            _arrows(ax, xy, p, ref, col, arrow_every, arrow_scale, alpha=0.9)
        g = _xz(gt, ref, L)
        ax.plot(g[:, 0], g[:, 1], '-', lw=3.0, color='tab:blue',
                label='GT' if k == 0 else None, zorder=5)
        ax.scatter(g[-1, 0], g[-1, 1], s=55, color='tab:blue', zorder=6)
        _arrows(ax, g, gt, ref, 'tab:blue', arrow_every, arrow_scale, alpha=0.9, zorder=8)
        pb, ps = _tr(base, nm, 'pred'), _tr(swap, nm, 'pred')
        dsw = float(np.linalg.norm(_xz(pb, ref, L) - _xz(ps, ref, L), axis=-1).mean())
        _finish(ax, f'{nm[:22]}   d_swap(2D)={dsw:.3f}')
        if k == 0:
            ax.legend(fontsize=8, loc='best')
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    sub = (f'; arrows = camera forward every {arrow_every} frames, XZ-projected '
           f'(shorter = looking up/down)') if arrow_every else ''
    head = ('context swap (same scene, other segment)' if label_set == 'swap' else
            f'out-of-segment context  vs  geo_test_inseg_k={swap_leak_k} leakage (same seed)')
    fig.suptitle(f'{head} — top-down (anchored to frame s; '
                 f'X vs -Z; GT reach = 1{sub})', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches='tight')
    plt.close(fig)
    print('saved', out_path)


def fig_inseg(arms, names, out_path):
    """arms = [(K, run_dir), ...]  K=0 은 없다; K=1 이 학습 조건과 동일한 대조군.

    누수된 context 카메라는 저장돼 있지 않지만 결정적으로 복원된다 --
    `_mix_inseg_context` 가 target 프레임을 `np.linspace(0, T-1, K).round()` 로 균등 분할하므로
    GT 궤적의 그 인덱스가 곧 context 로 들어간 카메라다. GT 위에 마커로 찍는다.
    """
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False)
    style = {1: ('0.45', 'o', 'pred K=1 (= training condition)'),
             3: ('tab:orange', 's', 'pred K=3'),
             5: ('tab:purple', '^', 'pred K=5')}
    for k, nm in enumerate(names):
        ax = axes[k // cols][k % cols]
        gt = _tr(arms[0][1], nm, 'ref')
        ref, L = _frame(gt)
        g = _xz(gt, ref, L)
        T = g.shape[0]
        ax.plot(g[:, 0], g[:, 1], '-', lw=3.0, color='tab:blue',
                label='GT' if k == 0 else None, zorder=4)
        ax.scatter(g[-1, 0], g[-1, 1], s=55, color='tab:blue', zorder=5)
        ds = {}
        p1 = _xz(_tr(arms[0][1], nm, 'pred'), ref, L)
        for K, d in arms:
            p = _tr(d, nm, 'pred')
            if p is None:
                continue
            xy = _xz(p, ref, L)
            col, mk, lab = style[K]
            ax.plot(xy[:, 0], xy[:, 1], '-', lw=2.0, color=col,
                    label=lab if k == 0 else None, zorder=6)
            ax.scatter(xy[-1, 0], xy[-1, 1], s=32, color=col, zorder=7)
            if K != 1:
                ds[K] = float(np.linalg.norm(xy - p1, axis=-1).mean())
                leak = np.linspace(0, T - 1, K).round().astype(int)
                ax.scatter(g[leak, 0], g[leak, 1], marker=mk, s=95, facecolors='none',
                           edgecolors=col, lw=1.8, zorder=8,
                           label=f'leaked ctx (K={K})' if k == 0 else None)
        _finish(ax, f"{nm[:20]}  d3={ds.get(3, float('nan')):.3f} d5={ds.get(5, float('nan')):.3f}")
        if k == 0:
            ax.legend(fontsize=7.5, loc='best')
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis('off')
    fig.suptitle('geo_test_inseg_k leakage — K of the 6 context views replaced by TARGET cameras '
                 '(top-down; anchored to frame s; X vs -Z; GT reach = 1)', fontsize=11)
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
    ap.add_argument('--fig', choices=['seeds', 'swap', 'inseg'], required=True)
    ap.add_argument('--seeds', nargs='*', default=[])
    ap.add_argument('--base', default=None)
    ap.add_argument('--swap', default=None)
    ap.add_argument('--inseg', nargs='*', default=[],
                    help="--fig inseg 용. 'DIR:K' 쌍들, K=1 을 먼저. 예: a:1 b:3 c:5")
    ap.add_argument('--names-from', default=None,
                    help='패널 후보 이름을 이 run 의 것으로 제한 (다른 그림과 같은 segment 를 뽑을 때)')
    ap.add_argument('--ctx-from', default=None,
                    help='--fig seeds 에서 context 카메라를 가져올 run (geo_ctx/ 있는 것)')
    ap.add_argument('--pick', default='even', choices=['even', 'dswap-top', 'dswap-low'])
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--style', default='cloud', choices=['cloud', 'rainbow'])
    ap.add_argument('--ctx-leak-k', type=int, default=0,
                    help='geo_test_inseg_k 값. context 앞 K장을 in-segment 누수 view 로 표시')
    ap.add_argument('--only', nargs='*', default=None,
                    help='패널로 그릴 segment 이름을 직접 지정 (--pick/--n 무시)')
    ap.add_argument('--arrow-every', type=int, default=0,
                    help='--fig swap 용. N 프레임마다 카메라 forward 화살표 (0=끄기)')
    ap.add_argument('--arrow-scale', type=float, default=0.16,
                    help='화살표 길이 (GT reach 단위)')
    ap.add_argument('--label-set', default='swap', choices=['swap', 'leak'],
                    help="--fig swap 의 범례. 'leak' 이면 --swap 을 in-segment 누수 arm 으로 표기")
    ap.add_argument('--swap-leak-k', type=int, default=0,
                    help="--label-set leak 에서 --swap 쪽 context 앞 K 장을 누수 view 로 표시")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.fig == 'seeds':
        names = _names(args.seeds[0])
        for d in args.seeds[1:]:
            names = [n for n in names if n in set(_names(d))]
    elif args.fig == 'swap':
        names = [n for n in _names(args.base) if n in set(_names(args.swap))]
    else:
        arms = [(int(x.rsplit(':', 1)[1]), x.rsplit(':', 1)[0]) for x in args.inseg]
        names = _names(arms[0][1])
        for _, d in arms[1:]:
            names = [n for n in names if n in set(_names(d))]
    if args.names_from:
        keep = set(_names(args.names_from))
        names = [n for n in names if n in keep]

    if args.only:
        miss = [n for n in args.only if n not in set(names)]
        if miss:
            raise SystemExit(f'--only 에 없는 segment: {miss}')
        sel = list(args.only)
    else:
        sel = pick(names, args.pick, args.n, args.base, args.swap)
    if args.fig == 'seeds':
        fig_seeds(args.seeds, sel, osp.join(args.out, f'seeds_{args.pick}_{args.style}{args.tag}.png'),
                  ctx_from=args.ctx_from, style=args.style, ctx_leak_k=args.ctx_leak_k)
    elif args.fig == 'swap':
        fig_swap(args.base, args.swap, sel,
                 osp.join(args.out, f'swap_{args.pick}{args.tag}.png'),
                 arrow_every=args.arrow_every, arrow_scale=args.arrow_scale,
                 label_set=args.label_set, swap_leak_k=args.swap_leak_k)
    else:
        fig_inseg(arms, sel, osp.join(args.out, f'inseg_{args.pick}{args.tag}.png'))


if __name__ == '__main__':
    main()
