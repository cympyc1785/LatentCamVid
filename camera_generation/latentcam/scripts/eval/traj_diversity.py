"""Conditional-diversity / context-sensitivity probe for sampled trajectories.

두 가지 질문에 답한다.

  (A) 같은 context 조건에서 seed 만 바꿨을 때 궤적이 얼마나 흩어지는가?
      -> seed 간 분산(within)과 segment 간 분산(between)을 분해해
         ICC  R = var_between / (var_between + var_within)  를 낸다.
         R -> 1 : 궤적이 조건(context/caption)에 의해 결정됨, seed 는 거의 무의미.
         R -> 0 : 궤적이 사실상 seed noise. 조건부 붕괴의 반대편 실패.

  (C) context 를 같은 scene 의 다른 segment 로 바꿔치기하면 궤적이 얼마나 움직이는가?
      -> d_swap 을 (A) 의 seed noise 단위로 표현한다.
         d_swap / seed_noise >> 1 : context 내용이 실제로 궤적을 끈다.
         d_swap / seed_noise ~ 1  : context 를 바꿔도 seed 를 바꾼 정도밖에 안 변한다
                                    = 모델이 context 의 '내용'은 거의 안 보고 있다.

궤적 좌표계: eval_testset.py 가 덤프한 *_transforms_{pred,ref}.json 은 DL3DV world
frame 의 OpenGL c2w 다. seed 간/arm 간 frame 이 공유되므로 직접 뺄셈이 가능하다.
segment 마다 스케일이 다르므로, 모든 거리는 그 segment 의 GT reach
  L = max_t || T_gt[t] - T_gt[0] ||
로 나눠 무차원화한 뒤에만 segment 를 가로질러 평균낸다.

Usage
-----
  # (A) seed 분산 분해
  python scripts/eval/traj_diversity.py --seeds eval_my/divers/seed*

  # (C) context swap 효과를 seed noise 단위로
  python scripts/eval/traj_diversity.py --seeds eval_my/divers/seed* \
      --base eval_my/divers/swap_base --swap eval_my/divers/swap_inscene
"""
import argparse
import glob
import json
import os
import os.path as osp

import numpy as np


def _load_traj(run_dir, name, kind):
    p = osp.join(run_dir, 'test', f'{name}_transforms_{kind}.json')
    if not osp.exists(p):
        return None
    frames = json.load(open(p))['frames']
    return np.asarray([f['transform_matrix'] for f in frames], dtype=np.float64)


def _names(run_dir):
    suf = '_transforms_pred.json'
    return sorted(osp.basename(p)[:-len(suf)]
                  for p in glob.glob(osp.join(run_dir, 'test', '*' + suf)))


def _load_geo_ctx(run_dir, name):
    p = osp.join(run_dir, 'geo_ctx', f'{name}.json')
    if not osp.exists(p):
        return None
    d = json.load(open(p))
    d['c2w'] = np.asarray(d['c2w'], dtype=np.float64)
    return d


def _collect(run_dirs, names):
    """-> P[n_seg, n_run, T, 3] (GT frame0 기준, GT reach 로 나눔), G[n_seg, T, 3], L[n_seg]"""
    P, G, L, kept = [], [], [], []
    for nm in names:
        g = _load_traj(run_dirs[0], nm, 'ref')
        if g is None:
            continue
        o = g[0, :3, 3]
        reach = float(np.linalg.norm(g[:, :3, 3] - o, axis=-1).max())
        if reach < 1e-6:
            continue
        rows = [_load_traj(d, nm, 'pred') for d in run_dirs]
        if any(r is None for r in rows) or len({r.shape[0] for r in rows}) != 1:
            continue
        if rows[0].shape[0] != g.shape[0]:
            continue
        P.append(np.stack([(r[:, :3, 3] - o) / reach for r in rows]))
        G.append((g[:, :3, 3] - o) / reach)
        L.append(reach)
        kept.append(nm)
    return np.asarray(P), np.asarray(G), np.asarray(L), kept


def arm_a(P, G):
    """seed 분산 분해. P[N, S, T, 3] (S = seed 수)."""
    N, S, T, _ = P.shape
    mu_seg = P.mean(axis=1)                                    # [N,T,3] segment 평균 궤적
    var_within = ((P - mu_seg[:, None]) ** 2).sum(-1).mean()   # seed 분산 (mm^2 아님, 무차원^2)
    var_between = ((mu_seg - mu_seg.mean(axis=0)) ** 2).sum(-1).mean()
    icc = var_between / (var_between + var_within + 1e-12)

    # 궤적 쌍거리 (APD) 와 GT 오차 (ADE) — 둘 다 프레임 평균 유클리드
    d = np.linalg.norm(P[:, :, None] - P[:, None, :], axis=-1).mean(-1)   # [N,S,S]
    iu = np.triu_indices(S, 1)
    apd = d[:, iu[0], iu[1]].mean(-1)                                     # [N]
    ade = np.linalg.norm(P - G[:, None], axis=-1).mean(-1)                # [N,S]

    return dict(
        n_segments=N, n_seeds=S, n_frames=T,
        var_within=float(var_within), var_between=float(var_between), icc=float(icc),
        rms_within=float(np.sqrt(var_within)), rms_between=float(np.sqrt(var_between)),
        apd_mean=float(apd.mean()), apd_median=float(np.median(apd)),
        ade_mean=float(ade.mean()), ade_best_of_S=float(ade.min(axis=1).mean()),
        ade_worst_of_S=float(ade.max(axis=1).mean()),
        apd_over_ade=float(apd.mean() / (ade.mean() + 1e-12)),
        gt_reach_norm=float(np.linalg.norm(G, axis=-1).max(axis=-1).mean()),
    )


def per_seg_within(P):
    """segment 별 seed noise 스케일 = seed 쌍거리 평균. -> [N]"""
    S = P.shape[1]
    d = np.linalg.norm(P[:, :, None] - P[:, None, :], axis=-1).mean(-1)
    iu = np.triu_indices(S, 1)
    return d[:, iu[0], iu[1]].mean(-1)


def arm_c(base_dir, swap_dir, seed_names, seed_within, names_index):
    nm_b = set(_names(base_dir))
    nm_s = set(_names(swap_dir))
    common = sorted(nm_b & nm_s)
    rows, n_swapped, n_nochange = [], 0, 0
    for nm in common:
        g = _load_traj(base_dir, nm, 'ref')
        b = _load_traj(base_dir, nm, 'pred')
        s = _load_traj(swap_dir, nm, 'pred')
        if g is None or b is None or s is None or b.shape != s.shape:
            continue
        o = g[0, :3, 3]
        reach = float(np.linalg.norm(g[:, :3, 3] - o, axis=-1).max())
        if reach < 1e-6:
            continue
        B = (b[:, :3, 3] - o) / reach
        Sw = (s[:, :3, 3] - o) / reach
        Gn = (g[:, :3, 3] - o) / reach
        gc = _load_geo_ctx(swap_dir, nm)
        was_swapped = bool(gc['geo_swapped']) if gc and 'geo_swapped' in gc else None
        if was_swapped is True:
            n_swapped += 1
        elif was_swapped is False:
            n_nochange += 1
        r = dict(
            name=nm,
            d_swap=float(np.linalg.norm(B - Sw, axis=-1).mean()),
            ade_base=float(np.linalg.norm(B - Gn, axis=-1).mean()),
            ade_swap=float(np.linalg.norm(Sw - Gn, axis=-1).mean()),
            swapped=was_swapped,
        )
        # context camera 에 얼마나 붙는가
        gcb = _load_geo_ctx(base_dir, nm)
        for tag, ctx, traj in (('base', gcb, B), ('swap', gc, Sw), ('gt', gcb, Gn)):
            if ctx is None:
                continue
            C = (ctx['c2w'][:, :3, 3] - o) / reach
            dmin = np.linalg.norm(traj[:, None] - C[None], axis=-1).min(-1)
            r[f'nnctx_{tag}'] = float(dmin.mean())
        j = names_index.get(nm)
        if j is not None:
            r['seed_noise'] = float(seed_within[j])
            r['ratio'] = r['d_swap'] / (seed_within[j] + 1e-12)
        rows.append(r)

    def m(k):
        v = [r[k] for r in rows if k in r and r[k] is not None]
        return float(np.mean(v)) if v else None

    out = dict(
        n_common=len(rows), n_swapped=n_swapped, n_not_swapped=n_nochange,
        d_swap_mean=m('d_swap'),
        ade_base_mean=m('ade_base'), ade_swap_mean=m('ade_swap'),
        seed_noise_mean=m('seed_noise'),
        d_swap_over_seed_noise=m('ratio'),
        nnctx_base=m('nnctx_base'), nnctx_swap=m('nnctx_swap'), nnctx_gt=m('nnctx_gt'),
    )
    return out, rows


def retrieval_r1(base_dir, swap_dir, names):
    """context retrieval R@1: 예측 궤적이 '자기 context' 를 다른 context 보다 가까이 두는가.

    후보 = 평가에 등장한 모든 segment 의 context camera 집합.
    점수 = 예측 궤적 -> 후보 context 의 nearest-neighbour 평균거리 (작을수록 매칭).
    """
    pool = []
    for nm in names:
        gc = _load_geo_ctx(base_dir, nm)
        g = _load_traj(base_dir, nm, 'ref')
        if gc is None or g is None:
            continue
        o = g[0, :3, 3]
        reach = float(np.linalg.norm(g[:, :3, 3] - o, axis=-1).max())
        if reach < 1e-6:
            continue
        pool.append((nm, (gc['c2w'][:, :3, 3] - o) / reach, o, reach))
    if len(pool) < 2:
        return None
    hit = 0
    for nm, C, o, reach in pool:
        b = _load_traj(base_dir, nm, 'pred')
        if b is None:
            continue
        B = (b[:, :3, 3] - o) / reach
        scores = [np.linalg.norm(B[:, None] - Ck[None], axis=-1).min(-1).mean()
                  for _, Ck, _, _ in pool]
        if int(np.argmin(scores)) == [p[0] for p in pool].index(nm):
            hit += 1
    return dict(n_pool=len(pool), r_at_1=hit / len(pool),
                chance=1.0 / len(pool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='*', default=[],
                    help='seed 별 eval 출력 디렉토리들 (>=2 개)')
    ap.add_argument('--base', default=None, help='context swap 안 한 arm')
    ap.add_argument('--swap', default=None, help='context swap 한 arm')
    ap.add_argument('--only', nargs='*', default=None,
                    help='이 segment 이름들만 쓴다 (run 마다 평가 segment 수가 다를 때 맞춰 비교용)')
    ap.add_argument('--per-seg', action='store_true',
                    help='segment 별 seed 통계(APD/ADE)도 출력. segment 가 몇 개뿐일 때 유용')
    ap.add_argument('--retrieval', action='store_true',
                    help='context retrieval R@1 도 계산 (O(N^2), geo_ctx 필요)')
    ap.add_argument('--out', default=None, help='결과 json 저장 경로')
    args = ap.parse_args()

    report = {}

    seed_within, names_index = None, {}
    if len(args.seeds) >= 2:
        names = _names(args.seeds[0])
        for d in args.seeds[1:]:
            names = [n for n in names if n in set(_names(d))]
        if args.only:
            names = [n for n in names if n in set(args.only)]
            miss = sorted(set(args.only) - set(names))
            if miss:
                print(f'[warn] --only 인데 없는 segment: {miss}')
        P, G, L, kept = _collect(args.seeds, names)
        report['A'] = arm_a(P, G)
        report['A']['runs'] = [osp.basename(d.rstrip('/')) for d in args.seeds]
        seed_within = per_seg_within(P)
        names_index = {nm: j for j, nm in enumerate(kept)}
        print('=== A: seed diversity (fixed context) ===')
        for k, v in report['A'].items():
            print(f'  {k:20s} {v}')
        if args.per_seg:
            # segment 가 2~3개뿐이면 var_between / icc 는 표본이 없는 거나 마찬가지라
            # segment 별 APD(seed 쌍거리) / ADE 를 그대로 본다. 단위는 그 segment 의 GT reach.
            ade = np.linalg.norm(P - G[:, None], axis=-1).mean(-1)          # [N,S]
            report['A']['per_segment'] = []
            print('  --- per segment (unit = that segment GT reach) ---')
            for j, nm in enumerate(kept):
                row = dict(name=nm, gt_reach=float(L[j]), apd=float(seed_within[j]),
                           ade_mean=float(ade[j].mean()), ade_best=float(ade[j].min()),
                           ade_worst=float(ade[j].max()),
                           pred_reach_over_gt=float(
                               (np.linalg.norm(P[j], axis=-1).max(-1)).mean()))
                report['A']['per_segment'].append(row)
                print(f"  {nm[:24]:24s} APD {row['apd']:.4f}  ADE {row['ade_mean']:.4f} "
                      f"(best {row['ade_best']:.4f} / worst {row['ade_worst']:.4f})  "
                      f"pred_reach/GT {row['pred_reach_over_gt']:.4f}  |L| {row['gt_reach']:.3f}")

    if args.base and args.swap:
        if seed_within is None:
            seed_within, names_index = np.zeros(0), {}
        c, rows = arm_c(args.base, args.swap, None, seed_within, names_index)
        report['C'] = c
        print('=== C: context swap (same scene, other segment) ===')
        for k, v in c.items():
            print(f'  {k:24s} {v}')
        if args.retrieval:
            report['C']['retrieval'] = retrieval_r1(args.base, args.swap,
                                                    sorted(names_index))
            print('  retrieval', report['C']['retrieval'])

    if args.out:
        os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
        json.dump(report, open(args.out, 'w'), indent=2)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
