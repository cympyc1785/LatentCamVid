"""한 scene 안에서 chunk 가 바뀔 때 arm B 의 divisor D 가 얼마나 흔들리는지 top-down 으로 본다.

배경. arm B = `scale_mode: geo_lagernvs` 는 target chunk 의 translation 을
    D = 1.35 * max || c_geo - c_anchor ||        (c_geo = 검색된 geo context 카메라들)
로 나눈다. geo context 는 chunk 마다 새로 검색(frustum max-coverage)되므로, **같은 scene 이라도
chunk 가 다르면 D 가 다르다**. 즉 B 는 scene 단위 canonical scale 이 아니라 chunk 단위 scale 이고,
"한 군데서 맴도는" 경로와 "일직선으로 뻗는" 경로에서 그 흔들림의 성격이 다르다.
이 스크립트는 그 D 를 실제 학습 경로 그대로(CamDataset._sample_geo_frustum_cover +
_geo_lagernvs_scale) 계산해서 chunk 별로 그린다.

용어 (약어는 첫 등장에 정의):
  D        divisor. norm_scale. dataset_dl3dv._geo_lagernvs_scale 이 돌려주는 값.
  maxd     max || c_target - c_s ||. target chunk 가 첫 프레임에서 얼마나 멀어지는지.
  m        maxd / D. 모델이 실제로 회귀해야 하는 normalized reach. 이게 chunk 마다 들쭉날쭉하면
           canonicalization 이 실패한 것.
  tok      camera_scale = max||c_geo - c_geo[0]|| / D. LagerNVS native 값은 1/1.35 = 0.7407.
  straight (직진성) = max_i||c_i - c_0|| / pathlen. 1 = 완전 직선, 0 에 가까울수록 제자리 배회.

두 가지 모드:
  --rank            scene 들을 직진성으로 정렬해서 후보를 출력만 한다 (그림 없음).
  --scene <name>    그 scene 의 모든 chunk 를 top-down 으로 그린다.

  # 후보 고르기 (chunk 5개 이상인 scene 만)
  PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
  $PY scripts/data/viz_scene_chunk_scale.py --rank --min-chunks 5 --top 15

  # 그리기
  $PY scripts/data/viz_scene_chunk_scale.py --scene 1K_abcdef... --out results/compare/chunk_scale

투영면: DL3DV world 축은 COLMAP 이 정한 임의 축이라 "z 가 위" 가 아니다. 카메라 center 들의
PCA 상위 2축(=이동이 가장 큰 평면)에 투영하는 것을 top-down 으로 본다(--plane pca, 기본).
--plane xz 를 주면 world X vs -Z 로 그리던 기존 스크립트 방식을 그대로 쓴다.
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg                                     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', default='geo_worldtraj_lagernvsnorm_scale96',
                   help='arm B 계열 experiment (scale_mode: geo_lagernvs)')
    p.add_argument('--no-arm-a', action='store_true',
                   help='비교용 arm A (ctx_longer_135max) 곡선을 D 패널에서 뺀다. '
                        'arm A 의 divisor 는 순수 기하 함수 _first_farthest_context 라 '
                        'CamDataset 을 새로 만들 필요 없이 같은 ds 에서 그대로 계산된다')
    p.add_argument('--split', default='train')
    p.add_argument('--rank', action='store_true', help='scene 직진성 랭킹만 출력')
    p.add_argument('--min-chunks', type=int, default=5)
    p.add_argument('--top', type=int, default=15)
    p.add_argument('--scene', default=None, help='data_name 의 scene 부분 (<batch>_<hash>)')
    p.add_argument('--plane', default='pca', choices=['pca', 'xz'])
    p.add_argument('--out', default=os.path.join(HERE, 'results/compare/chunk_scale'))
    p.add_argument('--tag', default=None, help='출력 파일 이름 앞에 붙일 라벨 (wander/straight 등)')
    return p.parse_args()


def scene_key(data_name):
    """'1K_<hash>_<seg>' -> '1K_<hash>'"""
    return data_name.rsplit('_', 1)[0]


def centers_of(ds, scene_idx):
    c2w = torch.linalg.inv(ds.extrinsics_list[scene_idx].float()).numpy()
    return c2w[:, :3, 3], c2w[:, :3, 2]        # centers, OpenCV forward axis


def straightness(centers):
    path = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    reach = float(np.linalg.norm(centers - centers[0], axis=1).max())
    return (reach / path if path > 1e-9 else 0.0), reach, path


def project(centers, plane):
    if plane == 'xz':
        return np.stack([centers[:, 0], -centers[:, 2]], 1), ('X', '-Z')
    c = centers - centers.mean(0)
    # PCA 상위 2축. 부호는 첫 프레임이 왼쪽 아래에 오도록 고정해 그림들끼리 방향이 안 뒤집히게.
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    P = c @ vt[:2].T
    for k in range(2):
        if P[0, k] > P[:, k].mean():
            P[:, k] *= -1
    return P, ('PC1', 'PC2')


def allowed_side(n, s, e):
    before, after = list(range(0, s)), list(range(e, n))
    return (before, 'before') if len(before) >= len(after) else (after, 'after')


def main():
    a = parse_args()
    cfg, _ = load_cfg("config", overrides=[f"experiment={a.exp}"])
    from dataset_dl3dv import CamDataset, resolve_scale_mode
    assert resolve_scale_mode(cfg) == 'geo_lagernvs', \
        f"--exp 는 scale_mode geo_lagernvs 여야 한다 (지금: {resolve_scale_mode(cfg)})"
    ds = CamDataset(cfg, a.split)

    by_scene = {}
    for i, (scene_idx, s, e, _cap, data_name) in enumerate(ds.samples):
        by_scene.setdefault(scene_key(data_name), []).append(i)

    if a.rank:
        rows = []
        for key, idxs in by_scene.items():
            if len(idxs) < a.min_chunks:
                continue
            centers, _ = centers_of(ds, ds.samples[idxs[0]][0])
            st, reach, path = straightness(centers)
            rows.append((st, key, len(idxs), centers.shape[0], reach, path))
        rows.sort()
        print(f"\n{len(rows)} scenes with >= {a.min_chunks} chunks "
              f"(exp={a.exp} split={a.split})")
        sv = np.array([r[0] for r in rows])
        print("straight percentiles: " + "  ".join(
            f"p{q}={np.percentile(sv, q):.3f}" for q in (1, 5, 25, 50, 75, 95, 99)))
        print("--- 중앙값 근처 (전형적 DL3DV) ---")
        mid = len(rows) // 2
        for r in rows[mid - a.top // 2:mid + a.top // 2]:
            print(f"{r[0]:8.4f} {r[2]:6d} {r[3]:6d} {r[4]:8.3f} {r[5]:9.3f}  {r[1]}")
        hdr = f"{'straight':>8s} {'chunks':>6s} {'frames':>6s} {'reach':>8s} {'path':>9s}  scene"
        print("\n--- 배회형 (straight 낮음) ---\n" + hdr)
        for r in rows[:a.top]:
            print(f"{r[0]:8.4f} {r[2]:6d} {r[3]:6d} {r[4]:8.3f} {r[5]:9.3f}  {r[1]}")
        print("\n--- 직선형 (straight 높음) ---\n" + hdr)
        for r in rows[-a.top:][::-1]:
            print(f"{r[0]:8.4f} {r[2]:6d} {r[3]:6d} {r[4]:8.3f} {r[5]:9.3f}  {r[1]}")
        return

    assert a.scene, "--scene 또는 --rank 중 하나는 줘야 한다"
    idxs = sorted(by_scene[a.scene], key=lambda i: ds.samples[i][1])
    scene_idx = ds.samples[idxs[0]][0]
    centers, _faxis = centers_of(ds, scene_idx)
    P, (axl0, axl1) = project(centers, a.plane)
    st, reach, path = straightness(centers)

    chunks = []
    for i in idxs:
        _si, s, e, _cap, data_name = ds.samples[i]
        gi = ds._sample_geo_frustum_cover(scene_idx, s, e)
        D = float(ds._geo_lagernvs_scale(scene_idx, s, e))
        tgt = centers[s:e]
        maxd = float(np.linalg.norm(tgt - centers[s], axis=1).max())
        c_geo = centers[gi]
        tok = float(np.linalg.norm(c_geo - c_geo[0], axis=1).max() / D)
        pool, side = allowed_side(centers.shape[0], s, e)
        DA = None
        if not a.no_arm_a:
            da = ds._first_farthest_context(scene_idx, s, e)     # arm A 의 divisor
            DA = None if da is None else float(da)
        chunks.append(dict(seg=data_name, s=s, e=e, gi=gi, D=D, DA=DA, maxd=maxd,
                           m=maxd / D, tok=tok, pool=pool, side=side))

    Ds = np.array([c['D'] for c in chunks])
    ms = np.array([c['m'] for c in chunks])
    DAs = (np.array([c['DA'] for c in chunks])
           if all(c['DA'] is not None for c in chunks) else None)
    n = len(chunks)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(5.2 * ncol, 5.2 * nrow + 4.0))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1] * nrow + [0.72],
                          hspace=0.30, wspace=0.18)

    # 모든 패널이 같은 xlim/ylim/aspect 를 쓴다 -- 그래야 chunk 사이에서 D 원의 크기 차이가
    # 그림 위에서 바로 비교된다. 정사각 범위는 데이터 bbox 와 가장 큰 D 원을 둘 다 담게 잡는다.
    lo, hi = P.min(0), P.max(0)
    ctr = (lo + hi) / 2
    half = max((hi - lo).max() / 2,
               max(np.abs(P[c['s']] - ctr).max() + c['D'] for c in chunks)) * 1.06

    handles = None
    for j, c in enumerate(chunks):
        ax = fig.add_subplot(gs[j // ncol, j % ncol])
        ax.plot(P[:, 0], P[:, 1], '-', c='0.85', lw=0.9, zorder=1, label='full trajectory')
        ax.scatter(P[c['pool'], 0], P[c['pool'], 1], s=6, c='#e8a33d', alpha=0.55, zorder=2,
                   label='candidate pool (out-of-segment longer side)')
        ax.plot(P[c['s']:c['e'], 0], P[c['s']:c['e'], 1], '-', c='tab:blue', lw=3.2, zorder=3,
                solid_capstyle='round', label='target chunk [s:e]')
        ax.scatter(P[c['e'] - 1, 0], P[c['e'] - 1, 1], s=34, marker='o', c='tab:blue',
                   ec='white', lw=0.6, zorder=4, label='target last frame')
        ax.scatter(P[c['gi'], 0], P[c['gi'], 1], s=150, marker='*', c='red', ec='white', lw=0.5,
                   zorder=6, label='geo context (retrieved)')
        ax.scatter(P[c['s'], 0], P[c['s'], 1], s=110, marker='P', c='green', ec='white', lw=0.6,
                   zorder=7, label='view0 = frame s (anchor)')
        # D 원: 반지름 D 를 anchor(frame s) 중심으로 그린다. chunk 별 divisor 크기 비교용.
        # 정의상 가장 먼 geo context 별이 반지름 D/1.35 지점에 놓인다.
        ax.add_patch(Circle(P[c['s']], c['D'], fill=False, ls='--', ec='purple', lw=1.6,
                            zorder=5, label='circle of radius D'))
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_aspect('equal')
        ax.set_title(f"chunk {j}  [{c['s']}:{c['e']}]\n"
                     f"D={c['D']:.3f}   maxd={c['maxd']:.3f}   m={c['m']:.3f}   tok={c['tok']:.4f}",
                     fontsize=10)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_xlabel(axl0, fontsize=8)
            ax.set_ylabel(axl1, fontsize=8)
            handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=10, ncol=7, loc='upper center',
                   bbox_to_anchor=(0.5, 0.955), frameon=False)

    axd = fig.add_subplot(gs[nrow, :])
    x = np.arange(n)
    axd.plot(x, Ds, 'o-', c='purple', label='D (arm B, geo_lagernvs)')
    if chunks[0]['DA'] is not None:
        axd.plot(x, [c['DA'] for c in chunks], 's--', c='tab:orange',
                 label='D (arm A, ctx_longer_135max)')
    axd.plot(x, [c['maxd'] for c in chunks], '^:', c='tab:blue', label='maxd (target reach)')
    axd.set_xticks(x)
    axd.set_xticklabels([f"{c['s']}:{c['e']}" for c in chunks], fontsize=8)
    axd.set_ylabel('length (world units)')
    axd.set_xlabel('chunk  [s:e]')
    axd.grid(alpha=0.3)
    axd.set_ylim(0, max(Ds.max(), max(c['maxd'] for c in chunks)) * 1.35)
    ax2 = axd.twinx()
    ax2.plot(x, ms, 'd-', c='crimson', label='m = maxd / D  (right axis)')
    ax2.set_ylabel('m = maxd / D', color='crimson')
    ax2.tick_params(axis='y', colors='crimson')
    ax2.set_ylim(0, ms.max() * 1.35)
    h1, l1 = axd.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axd.legend(h1 + h2, l1 + l2, fontsize=9, ncol=4, loc='upper center', framealpha=0.9)

    spread = float(Ds.max() / max(Ds.min(), 1e-9))
    mspread = float(ms.max() / max(ms.min(), 1e-9))
    fig.suptitle(
        f"{a.scene}   straight={st:.3f} (reach {reach:.2f} / path {path:.2f})   "
        f"{n} chunks, {centers.shape[0]} frames\n"
        f"D: min {Ds.min():.3f}  max {Ds.max():.3f}  max/min {spread:.2f}x  "
        f"sd(log10 D) {np.log10(Ds).std():.3f}   |   "
        f"m: min {ms.min():.3f}  max {ms.max():.3f}  max/min {mspread:.2f}x  "
        f"sd(log10 m) {np.log10(ms).std():.3f}"
        + (f"   |   arm A D: max/min {DAs.max() / DAs.min():.2f}x  "
           f"sd(log10 D) {np.log10(DAs).std():.3f}" if DAs is not None else ""),
        fontsize=13, y=0.995)

    os.makedirs(a.out, exist_ok=True)
    name = f"{a.tag + '_' if a.tag else ''}{a.scene}_{a.plane}.png"
    fp = os.path.join(a.out, name)
    fig.savefig(fp, dpi=130, bbox_inches='tight')
    print(f"saved {fp}")
    print(f"{'chunk':>12s} {'D(B)':>9s} {'D(A)':>9s} {'maxd':>9s} {'m':>7s} {'tok':>7s}  geo_idxs")
    for c in chunks:
        da = f"{c['DA']:9.4f}" if c['DA'] is not None else f"{'-':>9s}"
        print(f"{c['s']:5d}:{c['e']:<6d} {c['D']:9.4f} {da} {c['maxd']:9.4f} {c['m']:7.4f} "
              f"{c['tok']:7.4f}  {c['gi']}")


if __name__ == '__main__':
    main()
