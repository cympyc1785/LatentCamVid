"""SD 앵커 잔차를 **전 pair 에서** 재고, 어떤 필터로 잘리는지 본다.

`viz_sd_anchor_error.py` 가 200 sample 을 그림으로 보여 준 것을 여기서는 전수로 잰다.
핵심은 앵커 잔차가 **모델 없이 계산되는 pair 속성**이라는 것 — 그러니 인덱스 빌드 단계에서
그대로 필터가 된다. 이미지 디코딩 없이 pose.npz + avg_scale_align 만 읽으므로 빠르다.

  A = w2c_target[0] @ inv(w2c_ctx[0])
  rot   = angle(A[:3,:3])
  trans = ||A[:3,3]|| / norm_scale(ctx)      # 모델 단위
  m     = mean_t ||cam_param[t,6:9]||        # _target_out 을 직접 호출 (이미지 불필요)
  ratio = trans / m

같이 붙이는 clip 속성은 `sd_build_seg_lists.py` 가 이미 남긴
`<lists>/sd_<split>_clip_stats.csv` (moving / resid_rmse_over_rad / rot_spread_deg / keep).
static clip(moving=false, 7682/23408)은 리스트 단계에서 이미 빠져 있어서 여기 안 나온다 —
따라서 "안 움직이는 카메라"가 남아 있다면 그건 static 플래그가 아니라 **moving 인데 거의
안 움직이는** clip 이다. 그것도 같이 잰다 (path_len / radius).

usage:
  PYTHONPATH=.:main python scripts/data/sd_anchor_error_filters.py \
      --config mix_dl3dv_sd_datadop_v1 --out results/mixed/sd_anchor_filters
"""
import argparse
import csv
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path[:0] = ['.', 'main']


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def load_clip_stats(lists_dir, split):
    p = osp.join(lists_dir, f'sd_{split}_clip_stats.csv')
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            # csv 의 scene 은 '<split>/<scene>' 인데 dataset 의 relpath 는 '<scene>/<clip>' 이라
            # split 접두사를 떼야 붙는다 (안 떼면 조용히 전부 NaN 이 된다).
            d[f"{r['scene'].split('/')[-1]}/{r['clip']}"] = {
                'moving': int(r['moving']), 'keep': int(r['keep']),
                'resid': float(r['resid']) if r['resid'] not in ('', 'nan') else float('nan'),
                'rot_spread': float(r['rot_spread'])
                if r['rot_spread'] not in ('', 'nan') else float('nan')}
    return d


def pct(v, q):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='mix_dl3dv_sd_datadop_v1')
    ap.add_argument('--lists_dir',
                    default='/data1/cympyc1785/data/Scene-Decoupled-Video-dataset/latentcam_lists')
    ap.add_argument('--split', default='whuman')
    ap.add_argument('--max_targets', type=int, default=0, help='0 = 전부')
    ap.add_argument('--out', default='results/mixed/sd_anchor_filters')
    ap.add_argument('--overrides', nargs='*', default=[])
    ap.add_argument('--from_csv', default='',
                    help='이미 만든 pairs.csv 로 집계만 다시 (dataset 로딩 생략)')
    a = ap.parse_args()

    if a.from_csv:
        stats = load_clip_stats(a.lists_dir, a.split)
        rows = []
        with open(a.from_csv) as f:
            for r in csv.DictReader(f):
                d = {k: (v if k in ('tgt', 'ctx') else float(v)) for k, v in r.items()}
                st, sc = stats.get(d['tgt'], {}), stats.get(d['ctx'], {})
                d['tgt_resid'] = st.get('resid', float('nan'))
                d['ctx_resid'] = sc.get('resid', float('nan'))
                d['tgt_rotspread'] = st.get('rot_spread', float('nan'))
                d['ctx_rotspread'] = sc.get('rot_spread', float('nan'))
                rows.append(d)
        n_hit = sum(1 for r in rows if np.isfinite(r['tgt_resid']))
        print(f'[from_csv] {len(rows)} pairs, clip_stats 매칭 {n_hit}')
        return report(rows, a)

    from hydra_cfg import load_cfg
    ov = list(a.overrides)
    if a.config and a.config != 'config':
        ov.append(f'experiment={a.config}')
    cfg, _ = load_cfg('config', overrides=ov)
    from base import build_dataset
    ds = build_dataset(cfg)
    sd = ds.sub[list(ds.names).index('scene_decoupled')] \
        if getattr(cfg, 'dataset_name', None) == 'mixed' else ds

    stats = load_clip_stats(a.lists_dir, a.split)
    print(f'[clip_stats] {len(stats)} clips '
          f'(moving {sum(v["moving"] for v in stats.values())}, '
          f'keep {sum(v["keep"] for v in stats.values())})')

    # ---- clip 별 캐시: frame0 w2c, 자체 운동량, norm_scale
    clip_key = [osp.relpath(d, sd.root) for d in sd.scene_dir_list]
    n_clip = len(clip_key)
    w0 = np.zeros((n_clip, 4, 4))
    plen = np.zeros(n_clip)      # 궤적 경로 길이 (m)
    rad = np.zeros(n_clip)       # 중심에서의 최대 반경 (m)
    nsc = np.zeros(n_clip)
    for i in range(n_clip):
        w = sd.extrinsics_list[i].double().numpy()
        R, t = w[:, :3, :3], w[:, :3, 3]
        C = -np.einsum('tji,tj->ti', R, t)
        w0[i] = w[0]
        plen[i] = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
        rad[i] = float(np.linalg.norm(C - C.mean(0), axis=1).max())
        nsc[i] = float(sd._ctx_norm_scale(i).item())
        if (i + 1) % 2000 == 0:
            print(f'  clip {i + 1}/{n_clip}', flush=True)

    # ---- target 별 m (이미지 없이 _target_out 만)
    tgts = list(range(len(sd.samples)))
    if a.max_targets:
        tgts = tgts[:a.max_targets]
    rows = []
    for k, si in enumerate(tgts):
        tgt, s, e, caption, data_name = sd.samples[si]
        E = sd.extrinsics_list[tgt][s:e]
        K = sd.intrinsics_list[tgt][s:e]
        if E.shape[0] > sd.num_frames:
            sel = sd._even_indices(E.shape[0], sd.num_frames)
            E, K = E[sel], K[sel]
        h, w = sd.hw_list[tgt]
        for ctx in sd.ctx_pool_of[si]:
            ns = nsc[ctx]
            out = sd._target_out(E, K, torch.tensor([ns]).clamp(min=1e-5), h, w, caption, data_name)
            m = float(torch.linalg.norm(out['cam_param'][:, 6:9], dim=-1).mean())
            A = w0[tgt] @ np.linalg.inv(w0[ctx])
            tr = float(np.linalg.norm(A[:3, 3]) / max(ns, 1e-8))
            st, sc = stats.get(clip_key[tgt], {}), stats.get(clip_key[ctx], {})
            rows.append({'tgt': clip_key[tgt], 'ctx': clip_key[ctx], 'm': m,
                         'trans': tr, 'rot': float(rot_angle_deg(A[:3, :3])),
                         'ratio': tr / max(m, 1e-8), 'norm_scale': ns,
                         'tgt_plen': plen[tgt], 'tgt_rad': rad[tgt],
                         'ctx_plen': plen[ctx], 'ctx_rad': rad[ctx],
                         'tgt_resid': st.get('resid', float('nan')),
                         'ctx_resid': sc.get('resid', float('nan')),
                         'tgt_rotspread': st.get('rot_spread', float('nan')),
                         'ctx_rotspread': sc.get('rot_spread', float('nan'))})
        if (k + 1) % 2000 == 0:
            print(f'  target {k + 1}/{len(tgts)} -> {len(rows)} pairs', flush=True)

    os.makedirs(a.out, exist_ok=True)
    keys = list(rows[0].keys())
    with open(osp.join(a.out, 'pairs.csv'), 'w', newline='') as f:
        w_ = csv.DictWriter(f, keys)
        w_.writeheader()
        w_.writerows(rows)
    print(f'{len(rows)} pairs -> {osp.join(a.out, "pairs.csv")}')
    return report(rows, a)


def report(rows, a):
    os.makedirs(a.out, exist_ok=True)
    keys = list(rows[0].keys())
    R = {k: np.array([r[k] for r in rows], np.float64) for k in keys if k not in ('tgt', 'ctx')}
    R['resid_max'] = np.fmax(R['tgt_resid'], R['ctx_resid'])      # 두 clip 중 나쁜 쪽
    R['rotspread_min'] = np.fmin(R['tgt_rotspread'], R['ctx_rotspread'])
    rep = []

    def line(s=''):
        print(s)
        rep.append(s)

    line(f'# SD 앵커 잔차 전수 ({len(rows)} pairs)\n')
    line('## 분포')
    line('| 항목 | p50 | p90 | p95 | p99 | max |')
    line('|---|---|---|---|---|---|')
    for k in ['ratio', 'trans', 'rot', 'm', 'tgt_plen', 'tgt_rad',
              'tgt_resid', 'resid_max', 'tgt_rotspread']:
        line(f'| {k} | ' + ' | '.join(f'{pct(R[k], q):.4f}'
                                      for q in (50, 90, 95, 99)) +
             f' | {np.nanmax(R[k]):.4f} |')
    bad = R['ratio'] > 0.5
    line(f'\n`ratio>0.5` **{bad.sum()} / {len(rows)} = {100 * bad.mean():.2f}%**, '
         f'`>1.0` {100 * (R["ratio"] > 1).mean():.2f}%')

    line('\n## 상관 (log-log)')
    for k in ['m', 'trans', 'tgt_resid', 'ctx_resid', 'resid_max', 'tgt_plen', 'ctx_plen',
              'tgt_rad', 'tgt_rotspread', 'ctx_rotspread', 'norm_scale']:
        v = R[k]
        ok = np.isfinite(v) & (v > 0) & np.isfinite(R['ratio']) & (R['ratio'] > 0)
        c = np.corrcoef(np.log(v[ok]), np.log(R['ratio'][ok]))[0, 1] if ok.sum() > 10 else np.nan
        c2 = np.corrcoef(np.log(v[ok]), np.log(R['trans'][ok]))[0, 1] if ok.sum() > 10 else np.nan
        line(f'- `{k}` vs ratio {c:+.3f} | vs trans(절대오차) {c2:+.3f}  (n={int(ok.sum())})')

    def rowline(label, k):
        r = R['ratio'][k]
        line(f'| {label} | {100 * k.mean():.1f}% ({int(k.sum())}) | {pct(r, 95):.4f} | '
             f'{pct(r, 99):.4f} | {r.max():.4f} | {100 * (r > 0.5).mean():.2f}% | '
             f'{pct(R["trans"][k], 99):.4f} |')

    def head(name):
        line(f'\n## 필터: {name}')
        line('| 컷 | 남는 pair | ratio p95 | ratio p99 | ratio max | >0.5 | trans p99 |')
        line('|---|---|---|---|---|---|---|')

    def sweep(name, col, cuts, keep_below=True):
        head(name)
        for c in cuts:
            k = (R[col] <= c) if keep_below else (R[col] >= c)
            k &= np.isfinite(R[col])
            if k.sum() < 10:
                continue
            rowline(f'{c:.4g}', k)

    fr = R['tgt_resid'][np.isfinite(R['tgt_resid'])]
    qs = (99, 95, 90, 75, 50)
    sweep('target clip sim3 resid (현재 컷 = 코퍼스 p99 0.3273)', 'tgt_resid',
          [np.percentile(fr, q) for q in qs])
    sweep('두 clip 중 나쁜 쪽 resid (max)', 'resid_max',
          [np.percentile(R['resid_max'][np.isfinite(R['resid_max'])], q) for q in qs])
    sweep('target motion m 하한 (안 움직이는 target 제거)', 'm',
          [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1], keep_below=False)
    sweep('target 궤적 경로 길이 하한 (m)', 'tgt_plen', [0.1, 0.25, 0.5, 1.0, 2.0], keep_below=False)
    sweep('앵커 절대 오차 trans 상한 (pair 단위 직접 컷)', 'trans',
          [0.05, 0.03, 0.02, 0.015, 0.01, 0.005])
    sweep('앵커 회전 오차 상한 (deg)', 'rot', [8, 4, 2, 1, 0.5])
    sweep('ratio 직접 컷 (분자/분모 동시)', 'ratio', [1.0, 0.5, 0.3, 0.2, 0.1])

    head('조합: resid_max 컷 + m 하한')
    for rc in [np.percentile(R['resid_max'][np.isfinite(R['resid_max'])], q) for q in (99, 90, 75)]:
        for mc in [0.0, 0.02, 0.05]:
            k = np.isfinite(R['resid_max']) & (R['resid_max'] <= rc) & (R['m'] >= mc)
            rowline(f'resid<={rc:.3f}, m>={mc:.2f}', k)

    with open(osp.join(a.out, 'report.md'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print(f'\n-> {osp.join(a.out, "report.md")}')


if __name__ == '__main__':
    main()
