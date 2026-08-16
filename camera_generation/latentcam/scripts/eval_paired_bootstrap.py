#!/usr/bin/env python
"""공통 testset eval 9 arm 의 **짝지은(paired) 부트스트랩**.

전 arm 이 `--sample-seed 42` 로 같은 3263 세그먼트 · 같은 x_T 를 썼으므로 (common random
numbers) arm 간 차이는 세그먼트 단위로 짝지어진다. 세그먼트를 리샘플해 평균차의 95% CI 를
낸다 -> "arm A 가 B 보다 낫다"가 잡음인지 아닌지 갈린다 ([[match-requires-seed-variance]]).

**할 수 있는 지표만 한다.** `preds_scores.csv` 에서 per-sample 인 열은
    captions/{precision,recall,fscore}, clatr/{clatr_score,pred_ref_cosine}
뿐이다. `clatr/{precision,recall,density,coverage,fcd}` 는 전 행에 **전역 값이 복제**돼 있어
(집합 단위 지표) 여기서는 CI 를 못 낸다 -- 그건 trajectory embedding 을 다시 뽑아야 한다
(`test/*_transforms_{pred,ref}.json` -> CLaTr encoder). 그래서 PRDC/FCD 는 점추정만 보고한다.
"""
import argparse
import csv
import json
import os
import os.path as osp

import numpy as np

ROOT = '/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/eval_my/common100ep'
PER_SAMPLE = ['clatr/clatr_score', 'clatr/pred_ref_cosine',
              'captions/precision', 'captions/recall', 'captions/fscore']
GLOBAL_ONLY = ['clatr/precision', 'clatr/recall', 'clatr/density', 'clatr/coverage', 'clatr/fcd']

# 짝 비교: (라벨, arm_A, arm_B) -> A - B 를 잰다. 같은 geo context 계열 안에서만 짝짓는다.
PAIRS = [
    ('A frustum_cover: front_first_anchor - context_first_cam',
     '20260813_113851_da3_7k_da3geo_frontanchor',
     '20260813_113852_da3_7k_da3geo_asfirstcam'),
    ('B front_uniform: front_first_anchor_same_len - da3latent',
     '20260813_113852_da3_7k_da3geo_frontanchor_samelen',
     '20260813_152327_da3_7k_da3geo_latentscale'),
]


def load(run):
    """preds_scores.csv -> {filename: {metric: value}} + 전역 지표 dict."""
    p = osp.join(ROOT, run, 'preds_scores.csv')
    per, glob = {}, None
    with open(p) as f:
        for row in csv.DictReader(f):
            fn = row['filename']
            per[fn] = {m: float(row[m]) for m in PER_SAMPLE}
            if glob is None:
                glob = {m: float(row[m]) for m in GLOBAL_ONLY}
    return per, glob


def boot(diff, n_boot, seed):
    """세그먼트 단위 부트스트랩 -> (mean, lo, hi, P(diff>0))."""
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), float((means > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=osp.join(ROOT, 'paired_bootstrap.md'))
    a = ap.parse_args()

    runs = sorted(d for d in os.listdir(ROOT)
                  if osp.isfile(osp.join(ROOT, d, 'preds_scores.csv')))
    data = {r: load(r) for r in runs}
    print(f'[load] {len(runs)} runs')

    # 전 arm 의 filename 집합이 정말 같은지 -- 짝 비교의 전제
    keysets = {r: set(data[r][0]) for r in runs}
    base = keysets[runs[0]]
    for r in runs:
        assert keysets[r] == base, f'{r}: filename 집합이 다르다 (paired 비교 불가)'
    names = sorted(base)
    print(f'[check] 전 arm 공통 filename {len(names)} 개 (집합 동일)')

    lines = [f'# paired bootstrap (n_seg={len(names)}, n_boot={a.n_boot}, seed={a.seed})', '']
    lines += ['전 arm 이 `--sample-seed 42` 로 같은 세그먼트·같은 x_T 를 썼다 -> 세그먼트 단위 paired.',
              '`clatr/{precision,recall,density,coverage,fcd}` 는 per-sample 이 아니라 CI 없음.', '']

    results = {}
    for label, A, B in PAIRS:
        lines += [f'## {label}', '',
                  f'- A = `{A}`', f'- B = `{B}`', '',
                  '| metric | A | B | A-B | 95% CI | P(A>B) |',
                  '|---|---|---|---|---|---|']
        rows = {}
        for m in PER_SAMPLE:
            va = np.array([data[A][0][n][m] for n in names])
            vb = np.array([data[B][0][n][m] for n in names])
            d, lo, hi, p = boot(va - vb, a.n_boot, a.seed)
            sig = '' if lo <= 0 <= hi else ' **'
            lines.append(f'| {m} | {va.mean():.4f} | {vb.mean():.4f} | '
                         f'{d:+.4f}{sig} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} |')
            rows[m] = dict(a=va.mean(), b=vb.mean(), diff=d, lo=lo, hi=hi, p_gt=p)
        lines += ['', '집합 단위 지표 (CI 없음, 점추정만):', '',
                  '| metric | A | B | A-B |', '|---|---|---|---|']
        for m in GLOBAL_ONLY:
            ga, gb = data[A][1][m], data[B][1][m]
            lines.append(f'| {m} | {ga:.4f} | {gb:.4f} | {ga - gb:+.4f} |')
            rows[m] = dict(a=ga, b=gb, diff=ga - gb, lo=None, hi=None, p_gt=None)
        lines.append('')
        results[label] = dict(arm_a=A, arm_b=B, metrics=rows)

    txt = '\n'.join(lines)
    print(txt)
    with open(a.out, 'w') as f:
        f.write(txt + '\n')
    with open(a.out.replace('.md', '.json'), 'w') as f:
        json.dump(dict(n_seg=len(names), n_boot=a.n_boot, seed=a.seed, pairs=results), f, indent=1)
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
