"""DataDoP 의 "divisor 를 아예 안 쓴다(D=1)" 후보를 혼합 학습 관점에서 실측한다.

배경: scripts/data/datadop_divisor_spread.py 가 이미 후보 12종의 **산포**(sd(log10 m))를 쟀고
(results/datadop/datadop_divisor_spread/summary.md), 거기서 raw(D=1) 0.655 가 최소였다.
MonST3R 가 shot 단위로 이미 정규화하기 때문이다 (dust3r/losses.py:178-181 norm_mode='avg_dis',
cloud_opt/base_opt.py:66 base_scale=0.5).

이 스크립트가 추가로 재는 것은 **레벨과 꼬리**다. 산포가 작아도 다른 코퍼스와 중앙값이 어긋나면
모델이 "이 스케일 = DataDoP" 지름길을 배운다. 그래서:
  1) D=1 / meanray 각각의 m = mean_disp/D 분위수 + sd(log10 m)
  2) DL3DV / Scene-Decoupled 중앙값에 맞추는 상수 gain g 와, gain 적용 후 꼬리 위치
  3) index 필터 [min,max] 를 m 에 걸었을 때 남는 비율
  4) scene 내부 sd(log10 m)  <- shot 간 world 스케일 불일치의 하한 추정 (주의: 같은 scene 의
     shot 들은 궤적 자체가 달라서 모션 차이와 스케일 차이가 섞여 있다. 하한이지 순수 스케일 아님)

입력은 datadop_divisor_spread.py 가 남긴 per_shot.csv 하나뿐이라 재스캔이 없다 (~1초).
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import pandas as pd

# 이전 실측치 (계획서 표 / results 기록). 재측정이 아니라 **참조 상수**로만 쓴다.
REF_LEVELS = {
    'dl3dv (avg_scale_ref=centroid)':          {'med': 0.141, 'p95': 0.456, 'log10sd': 0.317},
    'dl3dv (avg_scale_ref=front_first_anchor)': {'med': 0.431, 'p95': 0.991, 'log10sd': None},
    'scene_decoupled (avg_scale_align)':       {'med': 0.147, 'p95': 0.416, 'log10sd': None},
}

QS = [1, 5, 25, 50, 75, 95, 99]


def describe(m):
    m = np.asarray(m, dtype=np.float64)
    m = m[np.isfinite(m) & (m > 0)]
    q = np.percentile(m, QS)
    return {'n': int(m.size),
            **{f'p{p}': float(v) for p, v in zip(QS, q)},
            'log10sd': float(np.std(np.log10(m)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/datadop/datadop_divisor_spread/per_shot.csv')
    ap.add_argument('--out', default='results/datadop/datadop_nodivisor_levels')
    ap.add_argument('--filter-min', type=float, default=0.01)
    ap.add_argument('--filter-max', type=float, default=2.0)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df['scene'] = df['shot'].str.split('/').str[0]
    print(f"[load] {len(df)} shots / {df['scene'].nunique()} scenes <- {args.csv}")

    cands = {'none (D=1)': np.ones(len(df)), 'meanray': df['meanray'].to_numpy()}
    out = {'candidates': {}, 'ref_levels': REF_LEVELS,
           'filter': [args.filter_min, args.filter_max]}

    for name, D in cands.items():
        m = df['mean_disp'].to_numpy() / D
        st = describe(m)
        keep = float(np.mean((m >= args.filter_min) & (m <= args.filter_max)))
        # scene 내부 산포 (shot >= 3 인 scene 만)
        tmp = pd.DataFrame({'scene': df['scene'], 'lm': np.log10(np.clip(m, 1e-12, None))})
        g = tmp.groupby('scene')['lm']
        within = g.std(ddof=1)[g.size() >= 3]
        st['keep_frac'] = keep
        st['within_scene_log10sd_med'] = float(np.nanmedian(within))
        st['within_scene_n_scenes'] = int(within.notna().sum())
        # 다른 코퍼스 중앙값에 맞추는 상수 gain 과, 그 gain 적용 후 꼬리
        st['gain_to'] = {}
        for ref, rv in REF_LEVELS.items():
            gain = rv['med'] / st['p50']
            st['gain_to'][ref] = {'gain': float(gain),
                                  'p95_after': float(st['p95'] * gain),
                                  'p99_after': float(st['p99'] * gain),
                                  'ref_p95': rv['p95']}
        out['candidates'][name] = st

    def row(name, st):
        return (f"{name:<12} n={st['n']:>6}  med={st['p50']:.4f}  "
                f"p05={st['p5']:.4f} p95={st['p95']:.4f} p99={st['p99']:.4f}  "
                f"log10sd={st['log10sd']:.3f}  keep[{args.filter_min},{args.filter_max}]="
                f"{st['keep_frac']*100:.1f}%  within-scene log10sd(med)="
                f"{st['within_scene_log10sd_med']:.3f}")

    print("\n== m = mean_disp / D  (모델이 실제로 보는 translation 크기) ==")
    for name, st in out['candidates'].items():
        print(row(name, st))

    print("\n== 참조 코퍼스 레벨 (이전 실측, 재측정 아님) ==")
    for ref, rv in REF_LEVELS.items():
        sd = f"{rv['log10sd']:.3f}" if rv['log10sd'] is not None else "-"
        print(f"{ref:<42} med={rv['med']:.3f}  p95={rv['p95']:.3f}  log10sd={sd}")

    print("\n== 중앙값 정렬용 상수 gain 과 정렬 후 꼬리 ==")
    for name, st in out['candidates'].items():
        for ref, gv in st['gain_to'].items():
            print(f"{name:<12} -> {ref:<42} gain={gv['gain']:.3f}  "
                  f"p95_after={gv['p95_after']:.3f} (ref p95 {gv['ref_p95']:.3f})  "
                  f"p99_after={gv['p99_after']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(osp.join(args.out, 'stats.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {osp.join(args.out, 'stats.json')}")


if __name__ == '__main__':
    main()
