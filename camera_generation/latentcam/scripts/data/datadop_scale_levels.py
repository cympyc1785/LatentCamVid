"""DataDoP 전 코퍼스의 translation 레벨 m 을 실측해 저장한다.

왜: 혼합 학습에서 세 코퍼스가 **모델이 실제로 보는** translation 크기
`m = mean_t||C_t − C_0|| / norm_scale` 를 비슷한 범위로 가져야 하는데, DataDoP 의 레벨은
20 scene 표본에서 med 0.176, 이전 측정에서 med 0.0727 로 2배 넘게 어긋나 있었다.
표본 크기/scene 편중 중 무엇인지 가르려면 전량을 같은 정의로 한 번 재야 한다.

정의는 `main/dataset_datadop.py` 의 인덱스 필터와 **글자 그대로 같은 코드**(`_shot_stats`)를
재사용한다 — 여기서 따로 구현하면 두 정의가 조용히 갈라진다.
  분자 mean_disp = mean over **120 프레임** ||C_t − C_0||   (GL c2w 의 4열 = 카메라 중심)
  분모 meanray   = mean(depth * ||K^-1 [u+.5, v+.5, 1]||)   (frame0 한 장)

usage:
  PYTHONPATH=.:main python scripts/data/datadop_scale_levels.py --workers 24
"""
import argparse
import json
import os
import os.path as osp
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

sys.path[:0] = ['.', 'main']
from dataset_datadop import (DATADOP_ROOT_DEFAULT, DATADOP_VALID_TXT,   # noqa: E402
                             _shot_stats)


def _job(args):
    root, rel = args
    ok, payload = _shot_stats(osp.join(root, rel))
    if not ok:
        return rel, None, payload
    return rel, (payload['mean_disp'], payload['meanray'], payload['hw']), None


def describe(m):
    m = np.asarray(m, np.float64)
    m = m[np.isfinite(m) & (m > 0)]
    p = np.percentile(m, [5, 25, 50, 75, 95, 99])
    return {'n': int(m.size), 'p05': p[0], 'p25': p[1], 'med': p[2], 'p75': p[3],
            'p95': p[4], 'p99': p[5], 'log10sd': float(np.std(np.log10(m)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DATADOP_ROOT_DEFAULT)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--trans-min', type=float, default=0.005)
    ap.add_argument('--trans-max', type=float, default=1.0)
    ap.add_argument('--out', default='results/datadop/scale_levels')
    a = ap.parse_args()

    valid = None
    if osp.isfile(DATADOP_VALID_TXT):
        with open(DATADOP_VALID_TXT) as f:
            valid = {ln.strip() for ln in f if ln.strip()}
    rels = []
    for sc in sorted(d for d in os.listdir(a.root)
                     if osp.isdir(osp.join(a.root, d)) and not d.startswith('.')):
        for fn in sorted(os.listdir(osp.join(a.root, sc))):
            if fn.endswith('_transforms_cleaning.json'):
                rel = f"{sc}/{fn[:-len('_transforms_cleaning.json')]}"
                if valid is None or rel in valid:
                    rels.append(rel)
    print(f'candidate shots: {len(rels)} (DataDoP_valid.txt {"on" if valid else "off"})',
          flush=True)

    rows, drops = [], {}
    jobs = [(a.root, r) for r in rels]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rel, payload, why in tqdm(ex.map(_job, jobs, chunksize=32), total=len(jobs)):
            if payload is None:
                k = str(why).split(':')[0]
                drops[k] = drops.get(k, 0) + 1
                continue
            disp, meanray, hw = payload
            rows.append((rel, disp, meanray, disp / meanray, hw[0], hw[1]))
    print(f'scanned ok: {len(rows)} | dropped {drops}', flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, 'per_shot.csv'), 'w') as f:
        f.write('shot,mean_disp,meanray,m,h,w\n')
        for rel, disp, mr, m, h, w in rows:
            f.write(f'{rel},{disp:.8g},{mr:.8g},{m:.8g},{h},{w}\n')

    m_all = [r[3] for r in rows]
    m_keep = [x for x in m_all if a.trans_min <= x <= a.trans_max]
    # scene 편중을 보려면 scene 당 1 shot 만 뽑은 것도 같이 본다 (한 scene 이 최대 수십 shot).
    seen, m_scene = set(), []
    for rel, _, _, m, _, _ in rows:
        sc = rel.split('/')[0]
        if sc not in seen and a.trans_min <= m <= a.trans_max:
            seen.add(sc)
            m_scene.append(m)
    stats = {'all': describe(m_all), 'filtered': describe(m_keep),
             'filtered_one_shot_per_scene': describe(m_scene),
             'keep_frac': len(m_keep) / max(len(m_all), 1),
             'filter': [a.trans_min, a.trans_max]}
    with open(osp.join(a.out, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    lines = ['| 집합 | n | med | p05 | p25 | p75 | p95 | p99 | sd(log10 m) |',
             '|---|---|---|---|---|---|---|---|---|']
    for k in ('all', 'filtered', 'filtered_one_shot_per_scene'):
        s = stats[k]
        lines.append(f"| {k} | {s['n']} | {s['med']:.4f} | {s['p05']:.4f} | {s['p25']:.4f} | "
                     f"{s['p75']:.4f} | {s['p95']:.4f} | {s['p99']:.4f} | {s['log10sd']:.3f} |")
    table = '\n'.join(lines)
    print(f"\nfilter [{a.trans_min}, {a.trans_max}] keeps {stats['keep_frac']*100:.1f}%\n{table}")
    with open(osp.join(a.out, 'summary.md'), 'w') as f:
        f.write(f'# DataDoP m = mean_t||C_t − C_0|| / meanray (frame0 mean ray, D=1)\n\n'
                f'filter [{a.trans_min}, {a.trans_max}] keeps '
                f"{stats['keep_frac']*100:.1f}%\n\n{table}\n")
    print(f"\n-> {osp.join(a.out, 'per_shot.csv')}\n-> {osp.join(a.out, 'summary.md')}")


if __name__ == '__main__':
    main()
