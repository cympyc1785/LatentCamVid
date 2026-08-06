"""Step 0 진단: cam_param 의 translation 을 w2c 로 둘 때와 c2w 로 둘 때 학습 신호가 실제로
얼마나 다른가 — DL3DV 학습 세그먼트 통계만으로 (모델/GPU 없이) 잰다.

배경. dataset_dl3dv.py:1085-1090 의 cam_param 은 `normalized_extrinsics[:, :3, 3]`,
즉 **w2c 상대 translation t = -R c** 를 쓴다 (c = 첫 카메라 기준 카메라 중심).
외부 궤적 생성 모델(GenDoP / Director3D / E.T.)은 전부 c(= c2w)를 쓴다. 어느 쪽이 나은지
논문 근거는 없으므로, 먼저 "차이가 생길 전제가 이 데이터에서 성립하는가"부터 확인한다.

용어 (약어는 첫 등장에 정의):
  t        w2c relative translation. cam_param[:, 6:9] 가 지금 담고 있는 값.
  c        camera center. 첫 카메라 기준 카메라 위치. c = -Rᵀ t, 그리고 항상 ‖t‖ = ‖c‖.
  s        divisor(norm_scale). 여기서는 avg_scale 규약인 mean‖c‖ 를 쓴다.
           ‖t‖ = ‖c‖ 라 divisor 는 두 규약에서 동일하다 — 스케일 정규화는 convention 불변.
  bleed    rotation bleed. Δt = -R_i Δc - ΔR c_{i+1} 로 분해했을 때, 위치 변화와 무관하게
           **회전 때문에** t 채널이 움직이는 몫:  mean‖ΔR c‖ / mean‖Δc‖.
           이게 0 에 가까우면 w2c/c2w 는 사실상 같은 신호다. 1 이면 t 의 프레임간 변화 중
           절반이 회전 성분이다.
  curv     비매끄러움. 2차 차분 크기를 1차 차분 크기로 나눈 scale-free 값
           (mean‖x_{i+1} - 2x_i + x_{i-1}‖ / mean‖x_{i+1} - x_i‖). 클수록 시간축으로 덜 매끄럽다.
  ratio    curv(t) / curv(c). > 1 이면 w2c 쪽이 덜 매끄럽다 = diffusion 이 맞히기 어렵다.
  th_path  총 회전 경로. 인접 프레임 geodesic 각의 합 (deg).
  th_net   시작-끝 geodesic 각 (deg).
  reach    max‖c‖ / s. anchor(frame 0)에서 얼마나 멀리 뻗는가.

판정. bleed 와 ratio 가 둘 다 1 에 한참 못 미치면(예: bleed p95 < 0.2, ratio p95 < 1.1)
"회전이 translation 채널로 샌다"는 전제가 DL3DV 에서 안 먹는 것이므로 VAE/DM 재학습까지
가는 ablation 은 접는다. 반대로 bleed 가 크고 th_path / reach 와 상관이 있으면 Step 1
(VAE 만 두 규약으로 학습해 recon 비교)로 올린다.

Run:
  python scripts/data/cam_repr_w2c_vs_c2w.py
  python scripts/data/cam_repr_w2c_vs_c2w.py --max-scenes 500      # 빠른 확인
out -> results/compare/cam_repr_w2c_vs_c2w/  (png + summary.json + per_seg.npz)
per_seg.npz 는 Step 0b(기존 arm 의 per-segment 오차를 th_path/reach 와 회귀)에서 재사용한다.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data1/cympyc1785/data/DL3DV/scenes"
OUT = ("/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results/compare/"
       "cam_repr_w2c_vs_c2w")
SEG_LIST = os.path.join(ROOT, "latentcam_train_seg_list_c4d2k5y4.txt")
NUM_FRAMES = 49
GL2CV = np.diag([1., -1., -1., 1.])      # DL3DV transforms.json 은 OpenGL c2w (nerfstudio)


def even_idx(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def geodesic_deg(Ra, Rb):
    """(N,3,3) x (N,3,3) -> (N,) 두 회전 사이 각 (deg)."""
    tr = np.einsum('nij,nij->n', Ra, Rb)          # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))


def seg_metrics(w2c):
    """(T,4,4) w2c -> dict. 실패하면 None."""
    nr = w2c @ np.linalg.inv(w2c[0])[None]        # 첫 카메라 기준 상대 w2c
    R = nr[:, :3, :3]
    t = nr[:, :3, 3]                              # w2c translation
    c = -np.einsum('tji,tj->ti', R, t)            # camera center = -Rᵀ t

    s = float(np.linalg.norm(c, axis=1).mean())   # avg_scale 규약의 divisor
    if s < 1e-3:                                  # 사실상 정지 카메라 -> divisor 가 0 에 붙는다
        return None
    t, c = t / s, c / s

    dt = np.diff(t, axis=0)
    dc = np.diff(c, axis=0)
    mdc = float(np.linalg.norm(dc, axis=1).mean())
    mdt = float(np.linalg.norm(dt, axis=1).mean())
    if mdc < 1e-6 or mdt < 1e-6:
        return None

    # Δt = -R_i Δc - ΔR c_{i+1}  중 두 번째 항이 순수 회전 성분
    dR = R[1:] - R[:-1]
    bleed_vec = np.einsum('nij,nj->ni', dR, c[1:])
    bleed = float(np.linalg.norm(bleed_vec, axis=1).mean() / mdc)

    d2t = t[2:] - 2 * t[1:-1] + t[:-2]
    d2c = c[2:] - 2 * c[1:-1] + c[:-2]
    curv_t = float(np.linalg.norm(d2t, axis=1).mean() / mdt)
    curv_c = float(np.linalg.norm(d2c, axis=1).mean() / mdc)

    return dict(
        bleed=bleed, curv_t=curv_t, curv_c=curv_c, ratio=curv_t / max(curv_c, 1e-9),
        th_path=float(geodesic_deg(R[:-1], R[1:]).sum()),
        th_net=float(geodesic_deg(R[:1], R[-1:])[0]),
        reach=float(np.linalg.norm(c, axis=1).max()),
    )


def rank(a):
    o = np.empty(len(a), float)
    o[np.argsort(a)] = np.arange(len(a))
    return o


def corr(a, b):
    """(pearson, spearman)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = float(np.corrcoef(a, b)[0, 1])
    s = float(np.corrcoef(rank(a), rank(b))[0, 1])
    return p, s


def stat(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), std=float(a.std()), p05=float(np.percentile(a, 5)),
                p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(a.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-list', default=SEG_LIST)
    ap.add_argument('--max-scenes', type=int, default=0, help='0 = 전부 (기본). 부분 표본은 '
                    'scene 편중이 생기므로 빠른 확인용으로만 쓸 것')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # seg list 를 scene 단위로 묶어 transforms.json 을 scene 당 한 번만 읽는다
    by_scene = {}
    with open(args.seg_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch, scene, seg = line.split('/')
            by_scene.setdefault(f'{batch}/{scene}', []).append(seg)
    scenes = sorted(by_scene)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    print(f'seg_list {args.seg_list}\nscenes {len(scenes)}  segs {sum(len(by_scene[s]) for s in scenes)}')

    keys = ['bleed', 'curv_t', 'curv_c', 'ratio', 'th_path', 'th_net', 'reach']
    acc = {k: [] for k in keys}
    names = []
    n_skip = 0
    for si, sc in enumerate(scenes):
        if si % 500 == 0:
            print(f'  {si}/{len(scenes)}  segs={len(names)}', flush=True)
        tj = os.path.join(ROOT, sc, 'transforms.json')
        pj = os.path.join(ROOT, sc, 'prompts.json')
        try:
            d = json.load(open(tj))
            pr = json.load(open(pj))
            c2w = np.array([f['transform_matrix'] for f in d['frames']], float) @ GL2CV
            w2c_all = np.linalg.inv(c2w)
        except Exception:
            n_skip += len(by_scene[sc])
            continue
        n = w2c_all.shape[0]
        for seg in by_scene[sc]:
            fi = pr.get(seg, {}).get('frame_idx')
            if not fi:
                n_skip += 1
                continue
            s0, e0 = int(fi[0]), int(fi[1])
            if e0 > n or (e0 - s0) < NUM_FRAMES:
                n_skip += 1
                continue
            w2c = w2c_all[s0:e0]
            if w2c.shape[0] > NUM_FRAMES:
                w2c = w2c[even_idx(w2c.shape[0], NUM_FRAMES)]
            m = seg_metrics(w2c)
            if m is None:
                n_skip += 1
                continue
            for k in keys:
                acc[k].append(m[k])
            names.append(f'{sc}/{seg}')

    nseg = len(names)
    print(f'\nsegments={nseg}  skipped={n_skip}')
    if not nseg:
        raise SystemExit('세그먼트를 하나도 못 읽었다')

    arr = {k: np.asarray(v, float) for k, v in acc.items()}
    summary = {k: stat(arr[k]) for k in keys}

    hdr = f"{'':9}" + ''.join(f'{h:>9}' for h in ['mean', 'std', 'p05', 'p50', 'p95', 'p99', 'max'])
    print('\n== per-segment 통계 ==')
    print(hdr)
    for k in keys:
        s = summary[k]
        print(f'{k:9}' + ''.join(f'{s[h]:>9.3f}' for h in ['mean', 'std', 'p05', 'p50', 'p95', 'p99', 'max']))

    frac = {f'ratio>{th}': float((arr['ratio'] > th).mean()) for th in (1.05, 1.1, 1.5, 2.0)}
    frac.update({f'bleed>{th}': float((arr['bleed'] > th).mean()) for th in (0.1, 0.2, 0.5, 1.0)})
    print('\n== 비율 ==')
    for k, v in frac.items():
        print(f'  {k:12} {v * 100:6.2f}%')

    cors = {}
    for x in ('th_path', 'th_net', 'reach'):
        for y in ('bleed', 'ratio'):
            p, s = corr(arr[x], arr[y])
            cors[f'{y}~{x}'] = dict(pearson=p, spearman=s)
    print('\n== 상관 (mechanism 이 실제로 작동하는지) ==')
    for k, v in cors.items():
        print(f'  {k:16} pearson {v["pearson"]:+.3f}   spearman {v["spearman"]:+.3f}')

    out_json = dict(n_segments=nseg, n_skipped=n_skip, seg_list=args.seg_list,
                    num_frames=NUM_FRAMES, divisor='avg_scale (mean||c||)',
                    summary=summary, fractions=frac, correlations=cors)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(out_json, f, indent=2)
    np.savez(os.path.join(args.out, 'per_seg.npz'), names=np.array(names), **arr)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    a = axs[0][0]
    a.hist(arr['bleed'], bins=80, color='tab:red', alpha=0.75,
           range=(0, np.percentile(arr['bleed'], 99)))
    a.axvline(np.median(arr['bleed']), color='k', ls='--',
              label=f"median {np.median(arr['bleed']):.3f}")
    a.set_title('rotation bleed  mean‖ΔR·c‖ / mean‖Δc‖\n(w2c t 채널 변화 중 회전이 만드는 몫)')
    a.set_xlabel('bleed'); a.legend()

    a = axs[0][1]
    a.hist(arr['ratio'], bins=80, color='tab:purple', alpha=0.75,
           range=(np.percentile(arr['ratio'], 1), np.percentile(arr['ratio'], 99)))
    a.axvline(1.0, color='k', lw=2, label='1.0 (동일)')
    a.axvline(np.median(arr['ratio']), color='tab:orange', ls='--',
              label=f"median {np.median(arr['ratio']):.3f}")
    a.set_title('비매끄러움 비 curv(t)/curv(c)\n>1 이면 w2c 가 덜 매끄럽다')
    a.set_xlabel('ratio'); a.legend()

    a = axs[1][0]
    a.scatter(arr['th_path'], arr['bleed'], s=3, alpha=0.15, color='tab:blue')
    p, s = corr(arr['th_path'], arr['bleed'])
    a.set_title(f'bleed vs 총 회전량   pearson {p:+.3f} / spearman {s:+.3f}')
    a.set_xlabel('th_path (deg)'); a.set_ylabel('bleed')
    a.set_xlim(0, np.percentile(arr['th_path'], 99))
    a.set_ylim(0, np.percentile(arr['bleed'], 99))

    a = axs[1][1]
    a.scatter(arr['reach'], arr['bleed'], s=3, alpha=0.15, color='tab:green')
    p, s = corr(arr['reach'], arr['bleed'])
    a.set_title(f'bleed vs reach(anchor 거리)   pearson {p:+.3f} / spearman {s:+.3f}')
    a.set_xlabel('reach = max‖c‖/s'); a.set_ylabel('bleed')
    a.set_xlim(0, np.percentile(arr['reach'], 99))
    a.set_ylim(0, np.percentile(arr['bleed'], 99))

    fig.suptitle(f'cam_param translation: w2c(t) vs c2w(c) — DL3DV 학습 세그먼트 {nseg}개', fontsize=13)
    fig.tight_layout()
    png = os.path.join(args.out, 'cam_repr_w2c_vs_c2w.png')
    fig.savefig(png, dpi=120, bbox_inches='tight')
    print('\nsaved', png)
    print('saved', os.path.join(args.out, 'summary.json'))
    print('saved', os.path.join(args.out, 'per_seg.npz'))


if __name__ == '__main__':
    main()
