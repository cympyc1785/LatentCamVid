"""GT 궤적 분포가 얼마나 '단순'한지를, density/coverage 가 실제로 문턱으로 쓰는 양으로 잰다.

왜 필요한가
-----------
`val/clatr/{density,coverage}` 는 CLaTr **임베딩 공간**에서
(`main/evaluate/eval/src/metrics/modules/prdc.py`, `manifold_k=3`)

    r_i      = (real 집합 안에서) i 번째 real 의 3-NN 거리
    density  = (1/k) * mean_j #{ i : ||real_i - fake_j|| < r_i }
    coverage = mean_i  [ min_j ||real_i - fake_j|| < r_i ]

로 계산되고, **real 집합은 그 validation 배치의 GT 궤적 160 개 그 자체**다 (고정된 외부
reference 가 아니다). 즉 문턱 r_i 는 코퍼스가 스스로 정한다. GT 궤적들이 서로 거의
같은 모양이면 r_i 가 작아지고, 모델이 아무리 그럴듯한 궤적을 내놔도 density/coverage 가
같이 무너진다 -- **모델이 나빠서가 아니라 분포가 좁아서** 생기는 붕괴다.
이 스크립트는 그 r_i 를 직접 재서 두 코퍼스를 비교한다.

같이 재는 것
------------
(1) caption tag 분포. `val/captions/*` 는 텍스트가 아니라 프레임간 상대 pose 에서 뽑은
    27 translation x 7 rotation = 189 클래스 태그의 weighted F1 이다
    (`metrics/modules/caption.py`). GT 태그가 한 클래스로 쏠려 있으면 F1 은 거저 올라간다.
    -> 태그 엔트로피 / 최빈 태그 비중 / 한 궤적 안에서 태그가 안 바뀌는 비율.
(2) 궤적 기하. straightness = |끝점-시작점| / 경로길이 (1.0 = 완전 직선),
    카메라 중심 PCA 의 sv2/sv1, sv3/sv1 (0 에 가까우면 직선/평면).
(3) 임베딩 매니폴드. 3-NN 반경을 전체 구름 크기로 나눠 무차원화하고, 참여비
    (participation ratio) 로 유효 차원을 잰다.
(4) **천장(ceiling)**. real 임베딩을 두 조각으로 갈라 한쪽을 fake 인 척 넣고 PRDC 를 돌린다.
    "참분포에서 뽑은 표본"이 받는 점수 = 이 지표로 받을 수 있는 상한이다. 코퍼스가 좁아서
    지표가 눌리는 것인지, 모델이 못하는 것인지를 가르는 유일한 대조군.

실측 결론 (2026-08-10, SD whuman textonly ep21 vs DL3DV da3_7k textonly ep54, n=80 맞춤)
------------------------------------------------------------------------------------
천장은 **두 코퍼스가 같다**: density 1.0016+-0.1240 / coverage 0.8877+-0.0843 (SD) vs
1.0206+-0.1395 / 0.8830+-0.0675 (DL3DV). 즉 분포가 좁다고 지표가 기계적으로 눌리지는 않는다.
대신 눌리는 것은 **허용 반경**이다. r(real 3-NN) 이 15.71 vs 26.25 로 SD 가 0.60 배인데
모델 오차(real -> 최근접 fake)는 17.49 vs 22.90 으로 0.76 배밖에 안 줄었다. 절대 거리로는
SD 쪽 모델이 **더 정확한데** 문턱이 더 빨리 좁아져서 비율이 1.121(>1, 탈락) vs 0.885(<1, 통과)
로 뒤집힌다. coverage 는 지시함수라 이 27% 차이가 0.225 vs 0.9125 로 증폭된다.

입력은 학습이 이미 남긴 산출물뿐이라 재추론이 필요 없다:
  <result_dir>/preds.npy       ref_matrices (N,49,4,4) / m_ref_latents (N,D)
  <result_dir>/preds_pcf.csv   target_segments (GT 태그 리스트)

usage
-----
  python scripts/eval/corpus_traj_manifold.py \
      --run SD=results/20260810_010756_sd_whuman_textonly \
      --run DL3DV=results/20260809_212844_da3_7k_textonly
out -> stdout + results/compare/corpus_traj_manifold/summary.md
"""
import argparse
import ast
import csv
import json
import os
import os.path as osp
from collections import Counter

import numpy as np

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
OUT = osp.join(REPO, 'results', 'compare', 'corpus_traj_manifold')


def _pct(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f'p{q}': float('nan') for q in qs}
    return {f'p{q}': float(np.percentile(a, q)) for q in qs}


def tag_stats(run_dir):
    """preds_pcf.csv 의 target_segments = GT 태그. 궤적당 48 개 (프레임간 상대 pose)."""
    p = osp.join(run_dir, 'preds_pcf.csv')
    per_traj_tags, all_tags, const_traj = [], [], 0
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            tags = ast.literal_eval(row['target_segments'])
            per_traj_tags.append(tags)
            all_tags.extend(tags)
            if len(set(tags)) == 1:
                const_traj += 1
    c = Counter(all_tags)
    n = sum(c.values())
    probs = np.array([v / n for v in c.values()])
    ent = float(-(probs * np.log2(probs)).sum())
    # 궤적을 하나의 '모션 문자열'로 봤을 때 서로 다른 것이 몇 종류인가
    sigs = Counter(tuple(t) for t in per_traj_tags)
    return {
        'n_traj': len(per_traj_tags),
        'n_tag_tokens': n,
        'n_distinct_tags': len(c),
        'tag_entropy_bits': ent,
        'tag_entropy_max_bits': float(np.log2(189)),   # 27 x 7 클래스
        'top1_tag': c.most_common(1)[0][0],
        'top1_share': c.most_common(1)[0][1] / n,
        'top3_share': sum(v for _, v in c.most_common(3)) / n,
        'const_traj_frac': const_traj / len(per_traj_tags),
        'n_distinct_traj_signatures': len(sigs),
        'top1_signature_share': sigs.most_common(1)[0][1] / len(per_traj_tags),
    }


def traj_geometry(ref_matrices):
    """ref_matrices (N,49,4,4) c2w -> straightness 와 중심 PCA 특이값 비."""
    C = np.asarray(ref_matrices, dtype=np.float64)[..., :3, 3]      # (N,T,3)
    d = np.diff(C, axis=1)
    pathlen = np.linalg.norm(d, axis=-1).sum(axis=1)                # (N,)
    chord = np.linalg.norm(C[:, -1] - C[:, 0], axis=-1)
    reach = np.linalg.norm(C - C[:, :1], axis=-1).max(axis=1)
    ok = pathlen > 1e-9
    straight = np.where(ok, chord / np.maximum(pathlen, 1e-12), np.nan)
    sv21, sv31 = [], []
    for c in C:
        x = c - c.mean(axis=0, keepdims=True)
        s = np.linalg.svd(x, compute_uv=False)
        if s[0] < 1e-12:
            sv21.append(np.nan); sv31.append(np.nan); continue
        sv21.append(s[1] / s[0]); sv31.append(s[2] / s[0])
    return {
        'straightness': _pct(straight),
        'sv2_over_sv1': _pct(sv21),
        'sv3_over_sv1': _pct(sv31),
        'pathlen': _pct(pathlen),
        'reach': _pct(reach),
        'frac_straightness_gt_0.99': float(np.nanmean(np.asarray(straight) > 0.99)),
        'frac_sv2_over_sv1_lt_0.01': float(np.nanmean(np.asarray(sv21) < 0.01)),
    }


def manifold_stats(lat, k=3):
    """density/coverage 가 문턱으로 쓰는 real 3-NN 반경을 직접 잰다."""
    X = np.asarray(lat, dtype=np.float64)
    n = X.shape[0]
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    D = np.sqrt(np.maximum(d2, 0))
    np.fill_diagonal(D, np.inf)
    knn = np.sort(D, axis=1)[:, k - 1]                       # k 번째 이웃까지의 거리 = r_i
    iu = np.triu_indices(n, 1)
    pair = D[iu]
    pair = pair[np.isfinite(pair)]
    # 구름 자체의 크기 (평균으로부터의 RMS 반경) 로 나눠 무차원화
    R = float(np.sqrt(((X - X.mean(0)) ** 2).sum(-1).mean()))
    ev = np.linalg.eigvalsh(np.cov(X.T))
    ev = np.clip(ev, 0, None)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum()) if (ev ** 2).sum() > 0 else float('nan')
    return {
        'n': n, 'dim': int(X.shape[1]),
        'cloud_rms_radius': R,
        'knn3_radius': _pct(knn),
        'knn3_radius_over_cloud_radius': _pct(knn / max(R, 1e-12)),
        'pairwise_dist': _pct(pair),
        'knn3_over_median_pairwise': float(np.median(knn) / max(np.median(pair), 1e-12)),
        'participation_ratio': pr,
    }


def _prdc(real, fake, k=3):
    """prdc.py 의 density/coverage 를 그대로 옮긴 것 (manifold_k 기본 3)."""
    dr = np.sqrt(((real[:, None] - real[None]) ** 2).sum(-1))
    np.fill_diagonal(dr, np.inf)
    r = np.sort(dr, 1)[:, k - 1]
    d = np.sqrt(((real[:, None] - fake[None]) ** 2).sum(-1))
    density = (1.0 / k) * (d < r[:, None]).sum(0).astype(float).mean()
    coverage = (d.min(1) < r).astype(float).mean()
    return float(density), float(coverage)


def ceiling(lat, n, k=3, trials=200, seed=0):
    """real 을 반으로 갈라 한쪽을 fake 취급 -> 참분포 표본이 받는 점수 = 지표의 상한."""
    X = np.asarray(lat, dtype=np.float64)
    rng = np.random.default_rng(seed)
    half = n // 2
    ds, cs = [], []
    for _ in range(trials):
        idx = rng.permutation(len(X))[:n]
        d, c = _prdc(X[idx[:half]], X[idx[half:]], k=k)
        ds.append(d); cs.append(c)
    return {'n_matched': n, 'half': half, 'trials': trials,
            'density_mean': float(np.mean(ds)), 'density_std': float(np.std(ds)),
            'coverage_mean': float(np.mean(cs)), 'coverage_std': float(np.std(cs))}


def threshold_budget(ref_lat, pred_lat, k=3):
    """coverage 가 걸리는 곳: 허용 반경 r 대비 실제 오차가 얼마인가."""
    R = np.asarray(ref_lat, dtype=np.float64)
    F = np.asarray(pred_lat, dtype=np.float64)
    dr = np.sqrt(((R[:, None] - R[None]) ** 2).sum(-1))
    np.fill_diagonal(dr, np.inf)
    r = np.sort(dr, 1)[:, k - 1]
    d = np.sqrt(((R[:, None] - F[None]) ** 2).sum(-1))
    return {
        'real_knn_radius_median': float(np.median(r)),
        'real_to_nearest_fake_median': float(np.median(d.min(1))),
        # 이 값이 1 을 넘으면 그 real 은 coverage 에서 탈락한다
        'ratio_median': float(np.median(d.min(1) / r)),
        'paired_ref_pred_over_r_median': float(np.median(np.diag(d) / r)),
        'fake_cloud_over_real_cloud_radius': float(
            np.sqrt(((F - F.mean(0)) ** 2).sum(-1).mean())
            / np.sqrt(((R - R.mean(0)) ** 2).sum(-1).mean())),
        'centroid_sep_over_real_radius': float(
            np.linalg.norm(R.mean(0) - F.mean(0))
            / np.sqrt(((R - R.mean(0)) ** 2).sum(-1).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True,
                    help='LABEL=<result_dir>  (여러 번)')
    ap.add_argument('--k', type=int, default=3, help='prdc 의 manifold_k')
    a = ap.parse_args()

    out, lats = {}, {}
    for spec in a.run:
        label, run_dir = spec.split('=', 1)
        run_dir = run_dir if osp.isabs(run_dir) else osp.join(REPO, run_dir)
        preds = np.load(osp.join(run_dir, 'preds.npy'), allow_pickle=True).item()
        lats[label] = preds['m_ref_latents']
        out[label] = {
            'run_dir': run_dir,
            'tags': tag_stats(run_dir),
            'geometry': traj_geometry(preds['ref_matrices']),
            'manifold_ref': manifold_stats(preds['m_ref_latents'], k=a.k),
            'threshold_budget': threshold_budget(preds['m_ref_latents'],
                                                 preds['m_pred_latents'], k=a.k),
            'metrics_json': json.load(open(osp.join(run_dir, 'metrics.json'))),
        }
    # 천장은 run 마다 n 이 다르면 비교가 안 된다 (batch_size 가 다르면 val 표본 수가 다르다)
    n_match = min(len(v) for v in lats.values())
    for label, x in lats.items():
        out[label]['ceiling_gt_vs_gt'] = ceiling(x, n_match, k=a.k)

    os.makedirs(OUT, exist_ok=True)
    with open(osp.join(OUT, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f'\nout -> {OUT}/summary.json')


if __name__ == '__main__':
    main()
