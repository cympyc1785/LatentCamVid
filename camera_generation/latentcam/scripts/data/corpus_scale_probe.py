"""코퍼스별 translation 레벨을 **데이터셋 객체에서 직접** 잰다.

왜 이 스크립트가 따로 필요한가: 지금까지의 레벨 표들은 전부 파일(pose.npz + avg_scale/*.json)을
다시 읽어 손으로 재현한 값이라, `normalize_camera_extrinsics_and_points` / `max_trans_norm` /
`_even_indices` / `norm_scale_gain` 같은 학습 경로의 마지막 단계가 빠져 있었다. 여기서는
`__getitem__` 이 뱉은 **cam_param 그 자체**를 잰다:

    m = mean_t || cam_param[t, 6:9] ||          (6:9 = translation 채널)

이 값이 세 코퍼스에서 비슷해야 한다는 것이 혼합 학습의 스케일 요건이고, `scale_gain` 은 이
숫자를 보고 정한다. 파일 기반 표와 어긋나면 **이쪽이 맞다** (모델이 보는 것이 이것이다).

usage:
  PYTHONPATH=.:main python scripts/data/corpus_scale_probe.py \
      --config da3_7k_da3geo_frontanchor --n 300
  PYTHONPATH=.:main python scripts/data/corpus_scale_probe.py \
      --config mix_smoke --n 300 --out results/mixed/scale_probe
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path[:0] = ['.', 'main']


def describe(m):
    m = np.asarray(m, np.float64)
    m = m[np.isfinite(m) & (m > 0)]
    if m.size == 0:
        return None
    p = np.percentile(m, [5, 25, 50, 75, 95, 99])
    return {'n': int(m.size), 'p05': p[0], 'p25': p[1], 'med': p[2], 'p75': p[3],
            'p95': p[4], 'p99': p[5], 'log10sd': float(np.std(np.log10(m)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config', help='hydra experiment 이름')
    ap.add_argument('--n', type=int, default=300, help='코퍼스당 표본 sample 수')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/mixed/scale_probe')
    ap.add_argument('--overrides', nargs='*', default=[])
    a = ap.parse_args()

    from hydra_cfg import load_cfg
    ov = list(a.overrides)
    if a.config and a.config != 'config':
        ov.append(f'experiment={a.config}')
    cfg, _ = load_cfg('config', overrides=ov)

    from base import build_dataset
    ds = build_dataset(cfg)

    # 코퍼스 이름 -> (dataset, 로컬 idx 리스트)
    if getattr(cfg, 'dataset_name', None) == 'mixed':
        groups = {}
        for ci, name in enumerate(ds.names):
            groups[name] = (ds.sub[ci], list(range(len(ds.sub[ci]))))
    else:
        groups = {str(getattr(cfg, 'dataset_name', 'dl3dv') or 'dl3dv'):
                  (ds, list(range(len(ds))))}

    rng = np.random.default_rng(a.seed)
    stats, rows = {}, []
    for name, (sub, idxs) in groups.items():
        pick = idxs if len(idxs) <= a.n else [idxs[i] for i in
                                              sorted(rng.choice(len(idxs), a.n, replace=False))]
        vals, nsc, fails = [], [], 0
        for i in pick:
            try:
                out = sub[i]
            except Exception as e:                      # noqa: BLE001
                fails += 1
                if fails <= 3:
                    print(f'  [{name}] idx {i} 실패: {type(e).__name__}: {e}')
                continue
            t = out['cam_param'][:, 6:9]
            m = float(torch.linalg.norm(t, dim=-1).mean())
            vals.append(m)
            nsc.append(float(out['norm_scale'].reshape(-1)[0]))
            rows.append((name, out.get('data_name', ''), m, nsc[-1]))
        st = describe(vals)
        st['fails'] = fails
        st['norm_scale_med'] = float(np.median(nsc)) if nsc else float('nan')
        stats[name] = st
        print(f"[{name}] n={st['n']} med={st['med']:.4f} p95={st['p95']:.4f} "
              f"sd(log10)={st['log10sd']:.3f} norm_scale_med={st['norm_scale_med']:.4g} "
              f"fails={fails}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, f'{a.config}_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    with open(osp.join(a.out, f'{a.config}_per_sample.csv'), 'w') as f:
        f.write('corpus,data_name,m,norm_scale\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]:.8g},{r[3]:.8g}\n')

    lines = ['| 코퍼스 | n | med | p05 | p25 | p75 | p95 | p99 | sd(log10 m) | norm_scale med |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for k, s in stats.items():
        lines.append(f"| {k} | {s['n']} | {s['med']:.4f} | {s['p05']:.4f} | {s['p25']:.4f} | "
                     f"{s['p75']:.4f} | {s['p95']:.4f} | {s['p99']:.4f} | {s['log10sd']:.3f} | "
                     f"{s['norm_scale_med']:.4g} |")
    table = '\n'.join(lines)
    print('\n' + table)
    with open(osp.join(a.out, f'{a.config}_summary.md'), 'w') as f:
        f.write(f'# m = mean_t||cam_param[t,6:9]|| (config={a.config}, n<={a.n}/코퍼스)\n\n'
                f'{table}\n')
    print(f"\n-> {osp.join(a.out, f'{a.config}_summary.md')}")


if __name__ == '__main__':
    main()
