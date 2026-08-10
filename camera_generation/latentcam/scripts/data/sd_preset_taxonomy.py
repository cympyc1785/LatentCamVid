"""Scene-Decoupled 의 카메라 움직임 '프리셋' 이 실제로 몇 종류인지 데이터로 센다.

배경: SD 는 scene 당 `<scene>_01_24mm` ~ `_07_24mm` 7 개 clip 을 갖고, README 는
"7 distinct camera trajectories per scene" 이라고만 적어 놓았다. 인덱스 01~07 이
**scene 을 가로질러 같은 움직임**을 뜻하는 고정 프리셋인지, 아니면 scene 마다 새로
뽑은 7 개인지가 안 적혀 있다. 그 둘은 학습/평가 해석이 완전히 다르다
(전자면 GT 궤적 다양성이 사실상 7 종, 후자면 46816 종).

방법
  1. camera/<split>/<scene>/<scene>_cam.json 을 읽는다. 값은 프레임별 4x4 행렬을
     행 벡터 규약으로 적은 문자열이다 (마지막 행이 translation, 단위는 cm 로 보인다).
  2. 프레임 0 카메라 좌표계로 옮긴다:  p'_i = R0^T (p_i - p_0),  R'_i = R0^T R_i.
     -> scene 배치(위치/방향)를 제거하고 '움직임' 만 남긴다.
  3. 경로 길이 L = sum ||p_{i+1} - p_i|| 로 나눠 크기까지 제거한다 (L=0 이면 static).
  4. 같은 인덱스끼리(scene 간) RMSD 와 다른 인덱스끼리 RMSD 를 비교한다.
     같은 인덱스가 확연히 작으면 인덱스 = 고정 프리셋이다.
  5. 각 인덱스의 움직임을 요약: frame-0 카메라축 기준 순 이동 방향(전/우/상),
     yaw/pitch/roll 변화량, static 비율.

사용
  python scripts/data/sd_preset_taxonomy.py [--split whuman] [--max-scenes 400]
"""
import argparse
import glob
import json
import os

import numpy as np

SD_ROOT = '/data1/cympyc1785/data/Scene-Decoupled-Video-dataset'


def parse_mat(s):
    """'[a b c d] [e f g h] ...' -> (4,4) 행 벡터 규약 행렬."""
    vals = [float(x) for x in s.replace('[', ' ').replace(']', ' ').split()]
    return np.asarray(vals, dtype=np.float64).reshape(4, 4)


def load_scene(cam_json):
    """-> {preset: (R (T,3,3) c2w, p (T,3))}. 프레임 키는 정수 순서로 정렬."""
    j = json.load(open(cam_json))
    frames = sorted(j.keys(), key=int)
    presets = sorted(j[frames[0]].keys())
    out = {}
    for k in presets:
        M = np.stack([parse_mat(j[f][k]) for f in frames])          # (T,4,4)
        # 행 벡터 규약: 각 행이 카메라 축의 world 표현 -> c2w 회전은 그 전치
        R = np.transpose(M[:, :3, :3], (0, 2, 1))                   # (T,3,3)
        p = M[:, 3, :3]                                             # (T,3)
        out[k] = (R, p)
    return out


def canon(R, p):
    """frame-0 카메라 좌표계로 옮기고 경로 길이로 정규화. -> (p_hat (T,3), R_rel (T,3,3), L)"""
    R0 = R[0]
    p_rel = (p - p[0]) @ R0                                         # R0^T (p-p0)
    R_rel = np.einsum('ji,tjk->tik', R0, R)                         # R0^T R_t
    L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return (p_rel / L if L > 1e-9 else p_rel), R_rel, L


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(-1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='whuman')
    ap.add_argument('--max-scenes', type=int, default=400)
    ap.add_argument('--static-thresh', type=float, default=1e-6,
                    help='경로 길이 L 이 이보다 작으면 static 으로 센다 (cm)')
    args = ap.parse_args()

    root = os.path.join(SD_ROOT, 'camera', args.split)
    scenes = sorted(os.listdir(root))[:args.max_scenes]
    print(f'{args.split}: {len(scenes)} scenes')

    shapes, lengths, rots, presets = {}, {}, {}, None
    for s in scenes:
        g = glob.glob(os.path.join(root, s, '*_cam.json'))
        if not g:
            continue
        try:
            sc = load_scene(g[0])
        except Exception as e:                                      # 깨진 json 은 건너뛴다
            print(f'  skip {s}: {e}')
            continue
        presets = presets or sorted(sc.keys())
        for k, (R, p) in sc.items():
            ph, Rr, L = canon(R, p)
            shapes.setdefault(k, []).append(ph)
            lengths.setdefault(k, []).append(L)
            # frame-0 대비 최대 회전각 (deg)
            tr = np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1)
            rots.setdefault(k, []).append(float(np.degrees(np.arccos(tr)).max()))

    print(f'presets: {presets}')
    T = shapes[presets[0]][0].shape[0]
    print(f'frames per clip: {T}\n')

    # ---- static 비율 / 경로 길이
    print('preset   n   static%   L median(cm)   L p05        L p95')
    for k in presets:
        L = np.asarray(lengths[k])
        st = float((L < args.static_thresh).mean())
        print(f'{k}  {len(L):4d}   {st * 100:6.2f}   {np.median(L):12.3f}  '
              f'{np.percentile(L, 5):10.3f}  {np.percentile(L, 95):10.3f}')

    # ---- 같은 인덱스 vs 다른 인덱스 RMSD (정규화된 shape 기준, static 제외)
    A = {}
    for k in presets:
        arr = np.stack(shapes[k])
        keep = np.asarray(lengths[k]) >= args.static_thresh
        A[k] = arr[keep]
    print('\n정규화 shape RMSD (static 제외). within = 같은 인덱스의 scene 간, '
          'mean = 그 인덱스 평균궤적과의 거리')
    print('preset   n_dyn   within-RMSD(mean shape 대비)')
    means = {}
    for k in presets:
        if len(A[k]) == 0:
            print(f'{k}      0   (전부 static)')
            means[k] = None
            continue
        m = A[k].mean(0)
        means[k] = m
        w = float(np.sqrt(((A[k] - m[None]) ** 2).sum(-1).mean()))
        print(f'{k}  {len(A[k]):6d}   {w:.6f}')

    ks = [k for k in presets if means[k] is not None]
    print('\n인덱스 평균궤적 사이의 RMSD (between)')
    print('        ' + '  '.join(f'{k[:2]:>8}' for k in ks))
    for a in ks:
        row = '  '.join(f'{rmsd(means[a], means[b]):8.5f}' for b in ks)
        print(f'{a[:2]:>6}  {row}')

    # ---- 각 인덱스가 무슨 움직임인지 요약 (frame-0 카메라축 기준)
    #      p_rel 축 = (x,y,z) 를 그대로 쓰되, 부호 규약은 데이터로 확인해야 하므로
    #      '축별 순 이동 / 총 이동' 비율과 회전각으로만 기술한다.
    print('\n움직임 요약 (frame-0 카메라 좌표, 정규화 shape; net = 마지막 프레임 위치, '
          'rot = frame-0 대비 최대 회전각 deg)')
    # NOTE 평균궤적의 |net| 을 straightness 로 쓰면 안 된다 — 방향이 scene 마다 다른
    # 프리셋은 평균에서 서로 상쇄돼 0 쪽으로 줄어든다. clip 별로 재고 중앙값을 쓴다.
    print('preset   mean net(x,y,z)          |net(mean)|   clip별 straightness med   '
          'clip별 net 방향 일치도(=|mean unit|)   rot_max med(deg)  p05    p95')
    for k in presets:
        if means[k] is None:
            continue
        n = means[k][-1]
        keep = np.asarray(lengths[k]) >= args.static_thresh
        nets = np.stack([sh[-1] for sh in shapes[k]])[keep]          # 정규화 shape 의 끝점
        straight = np.linalg.norm(nets, axis=1)                      # 경로 길이 1 기준
        unit = nets / np.maximum(np.linalg.norm(nets, axis=1, keepdims=True), 1e-12)
        coh = float(np.linalg.norm(unit.mean(0)))                    # 1 = 전부 같은 방향
        r = np.asarray(rots[k])[keep]
        print(f'{k}  ({n[0]:7.4f},{n[1]:8.4f},{n[2]:8.4f})  {np.linalg.norm(n):11.4f}  '
              f'{np.median(straight):22.4f}  {coh:33.4f}  '
              f'{np.median(r):15.2f}  {np.percentile(r, 5):6.2f} {np.percentile(r, 95):6.2f}')


if __name__ == '__main__':
    main()
