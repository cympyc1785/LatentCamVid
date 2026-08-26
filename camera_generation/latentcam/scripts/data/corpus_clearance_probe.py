"""코퍼스 카메라가 **표면에 얼마나 가까이 갔는지**를 잰다 (scene affordance 신호가 있나).

왜 필요한가: `corpus_scale_probe.py` 는 이동량 `‖t‖/norm_scale` 만 재는데, "이 씬에서 이만큼밖에
못 움직인다"를 모델이 배우려면 이동량이 아니라 **여유 거리(clearance)** 가 데이터에 들어 있어야
한다. DL3DV 는 실촬영이라 모든 궤적이 사람이 실제로 안 부딪히고 지나간 경로다 — 그 경로가 표면을
스치듯 지나가면 affordance 신호가 실재하는 것이고, 넉넉한 거리를 유지하면 신호가 없는 것이다.
합성 뱅크(LBM-Lite)의 collision solve 가 유일한 출처인지 여부가 여기서 갈린다.

재는 값 (세그먼트 하나당):
    점군 X = ∪_{f∈seg} unproject(da3 depth[f], K[f], c2w[f])      # pixel_stride 로 솎음
    clearance(f) = min_{x∈X} ‖center(f) − x‖
    -> min_f, med_f 를 **`norm_scale` 로 나눠서** 보고한다. target translation 과 같은 분모라
       `corpus_scale_probe.py` 의 `med` 와 바로 같은 축에서 비교된다.

norm_scale 은 재현하지 않고 `__getitem__` 이 뱉은 값을 그대로 쓴다 (scale_mode 분기가 8종이라
손으로 다시 짜면 어긋난다).

깊이 불연속 픽셀은 뺀다 (|∇log z| > edge_thr). 안 빼면 물체 경계에서 두 표면 사이에 붕 뜬 점이
생겨 clearance 를 실제보다 작게 만든다.

가정: da3 depth 는 **z-depth** (ray length 아님). `scene_graph/scale.py` 가 z·‖K⁻¹[u,v,1]‖ 로
ray length 를 따로 만드는 것이 근거. da3 extrinsics 는 **w2c** (`_parse_da3` docstring).

usage:
  PYTHONPATH=.:main CUDA_VISIBLE_DEVICES=7 python scripts/data/corpus_clearance_probe.py \
      --config da3_7k_da3geo_frontanchor --n 300
  PYTHONPATH=.:main python scripts/data/corpus_clearance_probe.py \
      --config mix_dl3dv_sd_datadop_v1 --n 300 --out results/mixed/clearance_probe
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
    p = np.percentile(m, [1, 5, 25, 50, 75, 95])
    return {'n': int(m.size), 'p01': p[0], 'p05': p[1], 'p25': p[2], 'med': p[3],
            'p75': p[4], 'p95': p[5], 'log10sd': float(np.std(np.log10(m)))}


def unproject(depth, K, c2w, stride, edge_thr, device):
    """(F,H,W) z-depth + (F,3,3) da3 격자 K + (F,4,4) c2w -> (M,3) world 점.

    K 는 **da3 depth 격자 해상도 기준**이어야 한다 (호출부에서 스케일해 넘긴다)."""
    F, H, W = depth.shape
    d = depth.to(device)
    ok = torch.isfinite(d) & (d > 0)
    d = torch.where(ok, d, torch.ones_like(d))

    # 깊이 불연속 마스크: |∇ log z| 가 큰 픽셀은 경계라 붕 뜬 점을 만든다.
    lz = torch.log(d.clamp(min=1e-6))
    g = torch.zeros_like(lz)
    g[:, 1:, :] = torch.maximum(g[:, 1:, :], (lz[:, 1:, :] - lz[:, :-1, :]).abs())
    g[:, :-1, :] = torch.maximum(g[:, :-1, :], (lz[:, 1:, :] - lz[:, :-1, :]).abs())
    g[:, :, 1:] = torch.maximum(g[:, :, 1:], (lz[:, :, 1:] - lz[:, :, :-1]).abs())
    g[:, :, :-1] = torch.maximum(g[:, :, :-1], (lz[:, :, 1:] - lz[:, :, :-1]).abs())
    ok = ok & (g <= edge_thr)

    vs = torch.arange(0, H, stride, device=device)
    us = torch.arange(0, W, stride, device=device)
    vv, uu = torch.meshgrid(vs, us, indexing='ij')
    pix = torch.stack([uu.reshape(-1) + 0.5, vv.reshape(-1) + 0.5,
                       torch.ones(uu.numel(), device=device)], -1)          # (P,3)

    dd = d[:, vs][:, :, us].reshape(F, -1)                                  # (F,P)
    mm = ok[:, vs][:, :, us].reshape(F, -1)
    Kinv = torch.linalg.inv(K.to(device))                                   # (F,3,3)
    ray = torch.einsum('fij,pj->fpi', Kinv, pix)                            # (F,P,3) z=1 평면
    pc = ray * dd.unsqueeze(-1)                                             # z-depth
    R = c2w[:, :3, :3].to(device)
    t = c2w[:, :3, 3].to(device)
    pw = torch.einsum('fij,fpj->fpi', R, pc) + t.unsqueeze(1)               # (F,P,3)
    return pw.reshape(-1, 3)[mm.reshape(-1)]


def min_dist(centers, pts, chunk=200000):
    """(F,3) 카메라 중심 -> 각 f 에 대한 min_x ‖c_f − x‖. 점군을 나눠 훑는다."""
    best = torch.full((centers.shape[0],), float('inf'), device=centers.device)
    for i in range(0, pts.shape[0], chunk):
        d = torch.cdist(centers, pts[i:i + chunk])                          # (F,c)
        best = torch.minimum(best, d.min(1).values)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='da3_7k_da3geo_frontanchor')   # hydra experiment 이름
    ap.add_argument('--n', type=int, default=300)                      # 코퍼스당 표본 수
    ap.add_argument('--seed', type=int, default=0)                     # scale_probe 와 같은 표본
    ap.add_argument('--out', default='results/mixed/clearance_probe')
    ap.add_argument('--stride', type=int, default=4)                   # depth 격자 솎음 간격
    ap.add_argument('--edge_thr', type=float, default=0.05)            # |∇log z| 컷
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--overrides', nargs='*', default=[])
    a = ap.parse_args()

    from hydra_cfg import load_cfg
    ov = list(a.overrides)
    if a.config and a.config != 'config':
        ov.append(f'experiment={a.config}')
    cfg, _ = load_cfg('config', overrides=ov)

    from base import build_dataset
    ds = build_dataset(cfg)

    if getattr(cfg, 'dataset_name', None) == 'mixed':
        groups = {name: ds.sub[ci] for ci, name in enumerate(ds.names)}
    else:
        groups = {str(getattr(cfg, 'dataset_name', 'dl3dv') or 'dl3dv'): ds}

    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(a.seed)
    stats, rows = {}, []
    for name, sub in groups.items():
        if not hasattr(sub, 'samples') or not hasattr(sub, '_da3_depth'):
            print(f'[{name}] da3 depth 접근자가 없다 -> 건너뜀', flush=True)
            continue
        idxs = list(range(len(sub)))
        pick = idxs if len(idxs) <= a.n else [idxs[i] for i in
                                              sorted(rng.choice(len(idxs), a.n, replace=False))]
        cmin, cmed, dmin, fails = [], [], [], 0
        for i in pick:
            try:
                scene_idx, s, e, _cap, _dn = sub.samples[i]
                w2c = sub.extrinsics_list[scene_idx][s:e].clone()           # (T,4,4)
                K_full = sub.intrinsics_list[scene_idx][s:e].clone()        # (T,3,3)
                if w2c.shape[0] > sub.num_frames:
                    sel = sub._even_indices(w2c.shape[0], sub.num_frames)
                    w2c, K_full = w2c[sel], K_full[sel]
                    fidx = [s + int(k) for k in sel]
                else:
                    fidx = list(range(s, s + w2c.shape[0]))

                dep = sub._da3_depth(scene_idx, fidx)                       # (F,H0,W0)
                h0, w0 = int(dep.shape[1]), int(dep.shape[2])
                hf, wf = sub.hw_list[scene_idx]
                K = K_full.clone()
                K[:, 0, :] *= (w0 / float(wf))                              # da3 격자로 스케일
                K[:, 1, :] *= (h0 / float(hf))

                c2w = torch.linalg.inv(w2c.double()).float()
                pts = unproject(dep, K, c2w, a.stride, a.edge_thr, dev)
                if pts.shape[0] < 1000:
                    raise ValueError(f'유효 점 {pts.shape[0]}개')
                cen = c2w[:, :3, 3].to(dev)
                cl = min_dist(cen, pts).cpu().numpy()

                ns = float(sub[i]['norm_scale'].reshape(-1)[0])             # 모델이 쓰는 그 분모
                cmin.append(float(cl.min()) / ns)
                cmed.append(float(np.median(cl)) / ns)
                dmin.append(float(dep[torch.isfinite(dep) & (dep > 0)].min()) / ns)
                rows.append((name, _dn, cmin[-1], cmed[-1], dmin[-1], ns))
            except Exception as ex:                                         # noqa: BLE001
                fails += 1
                if fails <= 3:
                    print(f'  [{name}] idx {i} 실패: {type(ex).__name__}: {ex}', flush=True)
                continue
            if len(cmin) % 25 == 0:
                print(f'  [{name}] {len(cmin)}/{len(pick)}', flush=True)

        st = {'clearance_min': describe(cmin), 'clearance_med': describe(cmed),
              'depth_min': describe(dmin), 'fails': fails}
        stats[name] = st
        cm = st['clearance_min']
        print(f"[{name}] n={cm['n']} clearance_min: p01={cm['p01']:.4f} p05={cm['p05']:.4f} "
              f"med={cm['med']:.4f} p95={cm['p95']:.4f}  fails={fails}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, f'{a.config}_clearance.json'), 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(osp.join(a.out, f'{a.config}_clearance_per_sample.csv'), 'w') as f:
        f.write('corpus,data_name,clearance_min,clearance_med,depth_min,norm_scale\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},{r[2]:.8g},{r[3]:.8g},{r[4]:.8g},{r[5]:.8g}\n')

    lines = ['| 코퍼스 | 지표 | n | p01 | p05 | p25 | med | p75 | p95 | sd(log10) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for k, s in stats.items():
        for key in ('clearance_min', 'clearance_med', 'depth_min'):
            d = s[key]
            if d is None:
                continue
            lines.append(f"| {k} | {key} | {d['n']} | {d['p01']:.4f} | {d['p05']:.4f} | "
                         f"{d['p25']:.4f} | {d['med']:.4f} | {d['p75']:.4f} | {d['p95']:.4f} | "
                         f"{d['log10sd']:.3f} |")
    table = '\n'.join(lines)
    print('\n' + table)
    with open(osp.join(a.out, f'{a.config}_clearance_summary.md'), 'w') as f:
        f.write(f'# clearance / norm_scale (config={a.config}, n<={a.n}, stride={a.stride}, '
                f'edge_thr={a.edge_thr})\n\n{table}\n')
    print(f"\n-> {osp.join(a.out, f'{a.config}_clearance_summary.md')}")


if __name__ == '__main__':
    main()
