"""DL3DV avg_scale 변형별로 **모델이 실제로 보는 translation 크기** m 을 재서 비교한다.

왜: 다중 코퍼스 혼합에서 DL3DV 의 레벨을 DataDoP/Scene-Decoupled 와 맞춰야 하는데,
분모(avg_scale) 변형만 바꿔도 레벨이 3배 넘게 움직인다. 어느 변형을 쓸지 정하려면
같은 정의로 잰 표가 필요하다.

m 의 정의 (학습 경로와 일치):
  cam_param 의 translation 은 `E_t @ inv(E_s)` 의 t 열이고, 그 노름은 대수적으로
  ||C_t − C_s|| (카메라 중심 거리) 다.  m = mean_t ||C_t − C_s|| / avg_scale.
  t 는 target segment [s,e) 를 num_frames 개로 균등 샘플한 프레임 (dataset 의 _even_indices).

변형 축은 2x2 다 — **range**(어느 프레임에서 점을 모으나 = 인과성)와 **origin**(어디서 거리를 재나):
  avg_scale                          (긴 쪽,   context centroid)   비인과 ~50%
  avg_scale_context_first_cam        (긴 쪽,   context 첫 카메라)   비인과 ~50%
  avg_scale_front_centroid           ([0,s),   context centroid)   인과   <- [new 2026-08-16]
  avg_scale_front_centroid_same_len  ([s-L,s), context centroid)   인과   <- [new 2026-08-16]
  avg_scale_front_first_anchor       ([0,s),   target 첫 카메라 s)  인과
  avg_scale_front_first_anchor_same_len ([s-L,s), target 첫 카메라) 인과
  avg_scale_da3latent                (front_uniform view, DA3 latent 스케일)
그리고 pose_source='transforms' 의 <scene>/avg_scale 은 **정의가 아예 다르다**:
  기준점 = target 첫 카메라 s, 점 = scene.ply **전체** (context range 개념 없음).
  포즈도 COLMAP 이라 분자까지 다르므로 별도 행으로 뺀다.

usage:
  python scripts/data/avgscale_variant_levels.py --splits 1K 2K 3K 4K 5K 6K 7K --workers 16
"""
import argparse
import json
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

ROOT = "/data1/cympyc1785/data/DL3DV/scenes"

# da3 포즈로 재는 변형들: 표시이름 -> <scene>/da3/<dir>
DA3_VARIANTS = {
    'centroid (긴 쪽)':            'avg_scale',
    'context_first_cam (긴 쪽)':   'avg_scale_context_first_cam',
    'front_centroid':              'avg_scale_front_centroid',
    'front_centroid_same_len':     'avg_scale_front_centroid_same_len',
    'front_first_anchor':          'avg_scale_front_first_anchor',
    'front_first_anchor_same_len': 'avg_scale_front_first_anchor_same_len',
    'da3latent':                   'avg_scale_da3latent',
}
TRANSFORMS_VARIANT = 'transforms avg_scale (scene.ply 전체, s 앵커)'


def even_indices(n_total, n_pick):
    """dataset_dl3dv.CamDataset._even_indices 와 같은 식."""
    return np.linspace(0, n_total - 1, n_pick).round().astype(int)


def _centers_from_w2c(ext):
    """(N,3,4) or (N,4,4) w2c -> (N,3) 카메라 중심."""
    R, t = ext[:, :3, :3], ext[:, :3, 3]
    return -np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)), t)


def _read_scalar(path):
    try:
        with open(path) as f:
            v = float(json.load(f))
        return v if np.isfinite(v) and v > 0 else None
    except Exception:
        return None


def scene_job(job):
    scene_dir, num_frames = job
    name = osp.relpath(scene_dir, ROOT).replace('/', '_')
    rows = []          # (variant, data_name, num, denom, m)

    # ---- da3 포즈 ----
    da3 = osp.join(scene_dir, 'da3')
    pz_path, pr_path = osp.join(da3, 'pose.npz'), osp.join(da3, 'prompts.json')
    if osp.isfile(pz_path) and osp.isfile(pr_path):
        try:
            C = _centers_from_w2c(np.load(pz_path)['extrinsics'].astype(np.float64))
            prompts = json.load(open(pr_path))
            for k in sorted(prompts, key=int):
                fi = prompts[k].get('frame_idx')
                if not fi or len(fi) != 2:
                    continue
                s, e = int(fi[0]), int(fi[1])
                if e > len(C) or (e - s) < num_frames:
                    continue
                idxs = s + even_indices(e - s, num_frames)
                num = float(np.linalg.norm(C[idxs] - C[s][None], axis=1).mean())
                for disp, sub in DA3_VARIANTS.items():
                    d = _read_scalar(osp.join(da3, sub, f'{int(k)}.json'))
                    if d is not None:
                        rows.append((disp, f'{name}_{k}', num, d, num / d))
        except Exception:
            pass

    # ---- transforms(COLMAP) 포즈 ----
    tj_path, pr2 = osp.join(scene_dir, 'transforms.json'), osp.join(scene_dir, 'prompts.json')
    if osp.isfile(tj_path) and osp.isfile(pr2):
        try:
            tj = json.load(open(tj_path))
            frames = sorted(tj['frames'], key=lambda fr: fr['file_path'])
            # c2w 의 translation 열 = 카메라 중심. GL->CV flip 은 오른쪽 곱이라 4열을 안 바꾼다.
            C = np.asarray([fr['transform_matrix'] for fr in frames], np.float64)[:, :3, 3]
            prompts = json.load(open(pr2))
            for k in sorted(prompts, key=int):
                fi = prompts[k].get('frame_idx')
                if not fi or len(fi) != 2:
                    continue
                s, e = int(fi[0]), int(fi[1])
                if e > len(C) or (e - s) < num_frames:
                    continue
                d = _read_scalar(osp.join(scene_dir, 'avg_scale', f'{int(k)}.json'))
                if d is None:
                    continue
                idxs = s + even_indices(e - s, num_frames)
                num = float(np.linalg.norm(C[idxs] - C[s][None], axis=1).mean())
                rows.append((TRANSFORMS_VARIANT, f'{name}_{k}', num, d, num / d))
        except Exception:
            pass
    return rows


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
    ap.add_argument('--splits', nargs='+', default=['1K', '2K', '3K', '4K', '5K', '6K', '7K'])
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--sample', type=int, default=0, help='무작위 표본 scene 수 (0=전량)')
    ap.add_argument('--sample-seed', type=int, default=0)
    ap.add_argument('--out', default='results/dl3dv/avgscale_variant_levels')
    a = ap.parse_args()

    scenes = []
    for sp in a.splits:
        sd = osp.join(ROOT, sp)
        if not osp.isdir(sd):
            continue
        scenes += [osp.join(sd, s) for s in sorted(os.listdir(sd))
                   if osp.isdir(osp.join(sd, s))]
    if a.sample and a.sample < len(scenes):
        rng = np.random.default_rng(a.sample_seed)
        scenes = [scenes[i] for i in sorted(rng.choice(len(scenes), a.sample, replace=False))]
    print(f'scenes: {len(scenes)}  num_frames={a.num_frames}', flush=True)

    per_variant = {}
    jobs = [(s, a.num_frames) for s in scenes]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for rows in tqdm(ex.map(scene_job, jobs, chunksize=8), total=len(jobs)):
            for var, dn, num, den, m in rows:
                per_variant.setdefault(var, []).append((dn, num, den, m))

    os.makedirs(a.out, exist_ok=True)
    csv_path = osp.join(a.out, 'per_seg.csv')
    with open(csv_path, 'w') as f:
        f.write('variant,data_name,mean_disp,avg_scale,m\n')
        for var, rows in per_variant.items():
            for dn, num, den, m in rows:
                f.write(f'"{var}",{dn},{num:.8g},{den:.8g},{m:.8g}\n')

    order = [TRANSFORMS_VARIANT] + list(DA3_VARIANTS)
    stats = {}
    lines = ['| 변형 | n | med | p05 | p25 | p75 | p95 | p99 | sd(log10 m) |',
             '|---|---|---|---|---|---|---|---|---|']
    for var in order:
        rows = per_variant.get(var)
        if not rows:
            continue
        st = describe([r[3] for r in rows])
        stats[var] = st
        lines.append(f"| {var} | {st['n']} | {st['med']:.4f} | {st['p05']:.4f} | "
                     f"{st['p25']:.4f} | {st['p75']:.4f} | {st['p95']:.4f} | "
                     f"{st['p99']:.4f} | {st['log10sd']:.3f} |")
    table = '\n'.join(lines)
    print('\n' + table)
    with open(osp.join(a.out, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    with open(osp.join(a.out, 'summary.md'), 'w') as f:
        f.write(f'# DL3DV avg_scale 변형별 m = mean_t||C_t − C_s|| / avg_scale '
                f'(num_frames={a.num_frames}, scenes={len(scenes)})\n\n{table}\n')
    print(f'\n-> {csv_path}\n-> {osp.join(a.out, "summary.md")}')


if __name__ == '__main__':
    main()
