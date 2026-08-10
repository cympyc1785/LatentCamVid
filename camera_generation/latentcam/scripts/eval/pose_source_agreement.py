"""같은 test segment 를 두 pose_source(COLMAP transforms vs DA3) 로 뽑았을 때
**GT 궤적 자체가 얼마나 다른가 / 얼마나 지저분한가**를 잰다.

왜 필요한가
-----------
da3_7k_da3pose 와 da3_7k_textonly 는 `pose_source` 한 줄만 다르고 seg list / blacklist /
meta_csv / vae_latent_scale 이 전부 같다. 그런데 같은 epoch 에서 da3pose 가 모든 지표에서
진다. 후보 원인은 둘뿐이다:

  (A) 지표 문제 — CLaTr 임베딩이 DA3 pose 를 못 다룬다 (OOD).
  (B) 데이터 문제 — DA3 GT 가 COLMAP GT 보다 **예측 불가능한 성분**(프레임간 지터)을 더 많이
      갖고 있어서, 텍스트로부터 학습 가능한 신호 대비 노이즈 비가 나쁘다.

(A) 는 corpus_traj_manifold.py 가 이미 기각했다 (두 GT 의 CLaTr 구름 반경 28.99 vs 29.15,
3-NN 반경 25.58 vs 26.05, GT-vs-GT 천장 density 1.0002/0.9999 coverage 0.8740/0.8770 —
구분이 안 된다). 이 스크립트는 (B) 를 잰다.

두 arm 의 preds.npy 는 **같은 160 segment 를 같은 순서로** 담고 있어서 (filename 완전 일치)
paired 비교가 된다. ref_matrices 는 frame-0-relative c2w 이고 arm 마다 분모(avg_scale)가
다르므로, 모든 양은 **경로길이로 무차원화**하거나 sim3 정렬 후에 잰다.

재는 것
-------
1. jitter_pos : 중심 경로를 길이 5 창의 2차 다항식으로 국소 적합했을 때의 잔차 중앙값을
                스텝 길이 중앙값으로 나눈 값. 클수록 프레임간 위치가 튄다.
2. jitter_rot : 회전의 2차 차분 각도(deg) 중앙값. 등속 회전이면 0 이다.
3. cross_resid: DA3 중심을 COLMAP 중심에 umeyama sim3 정렬한 뒤의 RMSE / reach.
                두 추정기가 같은 장면을 얼마나 다르게 본다고 말하는지.
4. cross_rot  : 정렬 후 프레임별 회전 측지 거리(deg) 중앙값.

usage
-----
  python scripts/eval/pose_source_agreement.py \
      --da3 results/20260808_140209_da3_7k_da3pose \
      --colmap results/20260809_212844_da3_7k_textonly
out -> stdout + results/compare/pose_source_agreement/summary.json
"""
import argparse
import csv
import json
import os
import os.path as osp

import numpy as np


def _pct(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return {f'p{q}': float(np.percentile(a, q)) for q in qs} if a.size else {}


def load(run_dir):
    d = np.load(osp.join(run_dir, 'preds.npy'), allow_pickle=True).item()
    M = np.asarray(d['ref_matrices'], dtype=np.float64)      # (N,T,4,4) frame-0-relative c2w
    m = np.asarray(d['masks'], dtype=bool)                   # (N,T)
    with open(osp.join(run_dir, 'preds_pcf.csv')) as f:
        rows = list(csv.DictReader(f))
    names = [r['filename'] for r in rows]
    f1 = np.array([float(r['captions/f1']) for r in rows], dtype=np.float64)
    return M, m, names, f1


def _local_poly_resid(C, win=5, deg=2):
    """중심 경로 C (T,3) 를 길이 win 창의 deg 차 다항식으로 국소 적합한 잔차 (T,).

    '부드러운 궤적'은 국소적으로 2차로 잘 맞는다. 잘 안 맞는 성분이 곧 지터다.
    창이 안 잡히는 양 끝 (win//2) 프레임은 버린다 -- 경계 효과를 지터로 오해하지 않기 위해.
    """
    T = C.shape[0]
    h = win // 2
    if T < win:
        return np.zeros(0)
    t = np.arange(-h, h + 1, dtype=np.float64)
    A = np.vander(t, deg + 1)                      # (win, deg+1)
    pinv = np.linalg.pinv(A)
    out = []
    for i in range(h, T - h):
        seg = C[i - h:i + h + 1]                   # (win,3)
        fit = A @ (pinv @ seg)                     # (win,3)
        out.append(np.linalg.norm(seg[h] - fit[h]))
    return np.asarray(out)


def _rot_angle(Ra, Rb):
    """두 회전행렬 배치 사이 측지 각도(deg)."""
    tr = np.einsum('...ij,...ij->...', Ra, Rb)     # trace(Ra^T Rb)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def jitter(M, m):
    """arm 하나의 GT 지터. 분모(avg_scale)에 안 걸리도록 스텝 길이로 무차원화한다."""
    pos, rot, step = [], [], []
    for Mi, mi in zip(M, m):
        C = Mi[mi][:, :3, 3]
        R = Mi[mi][:, :3, :3]
        d = np.linalg.norm(np.diff(C, axis=0), axis=1)
        s = float(np.median(d))
        if s <= 0:
            continue
        r = _local_poly_resid(C)
        if r.size:
            pos.append(float(np.median(r)) / s)
        # 회전 2차 차분: R_{t-1}^T R_t 두 개를 비교 -> 각속도가 얼마나 튀는지
        rel = np.einsum('tji,tjk->tik', R[:-1], R[1:])          # (T-1,3,3)
        if rel.shape[0] >= 2:
            rot.append(float(np.median(_rot_angle(rel[:-1], rel[1:]))))
        step.append(s)
    return {
        'jitter_pos_over_step': _pct(pos),
        'jitter_rot_2nd_diff_deg': _pct(rot),
        'median_step_len': _pct(step),
    }


def umeyama(X, Y):
    """X (n,3) -> Y (n,3) 의 sim3 (s,R,t). 반환은 (s, R, t, rmse)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    S = Yc.T @ Xc / len(X)
    U, D, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(U @ Vt))
    W = np.diag([1.0, 1.0, d])
    R = U @ W @ Vt
    varx = (Xc ** 2).sum() / len(X)
    s = float((np.diag(D) @ W).trace() / varx) if varx > 0 else 1.0
    t = my - s * R @ mx
    res = Y - (s * (R @ X.T).T + t)
    return s, R, t, float(np.sqrt((res ** 2).sum(1).mean()))


def cross(Ma, ma, Mb, mb):
    """DA3(a) 를 COLMAP(b) 에 sim3 정렬한 뒤의 불일치."""
    resid, rot, reach_ratio = [], [], []
    for Mai, mai, Mbi, mbi in zip(Ma, ma, Mb, mb):
        A, B = Mai[mai], Mbi[mbi]
        n = min(len(A), len(B))
        Ca, Cb = A[:n, :3, 3], B[:n, :3, 3]
        reach_b = float(np.linalg.norm(Cb[-1] - Cb[0]))
        pathlen_b = float(np.linalg.norm(np.diff(Cb, axis=0), axis=1).sum())
        if pathlen_b <= 0:
            continue
        s, R, t, rmse = umeyama(Ca, Cb)
        resid.append(rmse / pathlen_b)
        Ra = np.einsum('ij,tjk->tik', R, A[:n, :3, :3])
        rot.append(float(np.median(_rot_angle(Ra, B[:n, :3, :3]))))
        ra = float(np.linalg.norm(Ca[-1] - Ca[0]))
        if reach_b > 0:
            reach_ratio.append(s * ra / reach_b)
    return {
        'sim3_rmse_over_pathlen': _pct(resid),
        'rot_disagree_deg': _pct(rot),
        'reach_ratio_after_s': _pct(reach_ratio),
        'n': len(resid),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--da3', required=True)
    ap.add_argument('--colmap', required=True)
    ap.add_argument('--out', default='results/compare/pose_source_agreement')
    a = ap.parse_args()

    Ma, ma, na, f1a = load(a.da3)
    Mb, mb, nb, f1b = load(a.colmap)
    assert na == nb, 'val segment 집합/순서가 다르다 -- paired 비교 불가'

    out = {
        'n_segments': len(na),
        'DA3': {'run_dir': a.da3, **jitter(Ma, ma)},
        'COLMAP': {'run_dir': a.colmap, **jitter(Mb, mb)},
        'cross_DA3_to_COLMAP': cross(Ma, ma, Mb, mb),
        'caption_f1_per_sample': {
            'DA3_mean': float(f1a.mean()), 'COLMAP_mean': float(f1b.mean()),
            'paired_diff_mean': float((f1a - f1b).mean()),
            'note': 'ckpt epoch 이 다르면(DA3 ep149 vs COLMAP ep59) 이 항목은 비교 불가',
        },
    }
    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print('\nout ->', osp.join(a.out, 'summary.json'))


if __name__ == '__main__':
    main()
