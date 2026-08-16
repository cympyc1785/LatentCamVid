"""Scene-Decoupled 의 **앵커 잔차**를 눈으로 확인한다.

문제. `geo_encoder='da3'` + `geo_posed=true` 에서 DA3 `build_cam_token` 은 context view0 을
기준으로 pose 를 재고정한다 (`models/da3_geo_encoder.py:279-282`, `w2c = w2c @ c2w[:, :1]`).
반면 target `cam_param` 의 기준은 target clip 의 첫 프레임 s 다 (`_target_out` 의
`rel = w2c_t @ inv(w2c_s)`). SD 는 같은 scene 을 7개 궤적으로 렌더한 clip 을 쓰므로 두 clip 의
frame0 카메라가 **같아야** 이 두 기준이 일치하는데, clip 마다 da3 를 따로 돌린 뒤 umeyama sim3
로 GT 미터에 올리기 때문에 정렬 잔차가 남는다.

잔차 정의 (한 sample):
    A = w2c_target[s] @ inv(w2c_ctx[view0])        # 두 앵커 사이의 rigid 변환
    rot  = angle(A[:3,:3])                          # deg
    tr   = ||A[:3,3]|| / norm_scale                 # 모델 단위 (cam_param 과 같은 자)
    m    = mean_t ||cam_param[t,6:9]||              # 예측해야 할 translation 크기
    ratio = tr / m                                  # 신호 대비 앵커 오차

DL3DV / DataDoP 는 view0 == target 첫 프레임이 **구조적으로** 참이라 ratio == 0 이다.

usage:
  PYTHONPATH=.:main python scripts/data/viz_sd_anchor_error.py \
      --config mix_dl3dv_sd_datadop_v1 --n 300 --out results/mixed/sd_anchor_viz
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path[:0] = ['.', 'main']


def rot_angle_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def cam_centers(w2c):
    """w2c (T,4,4) -> camera centers (T,3)."""
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return -np.einsum('tji,tj->ti', R, t)


def measure(sd, idxs):
    rows = []
    for i in idxs:
        tgt, s, e, caption, data_name = sd.samples[i]
        ctx = sd._pick_ctx(i) if sd.pair_mode == 'random_scene' else sd.ctx_of[i]
        try:
            out = sd[i]
        except Exception as ex:                                   # noqa: BLE001
            print(f'  idx {i} 실패: {type(ex).__name__}: {ex}')
            continue
        ns = float(out['norm_scale'].reshape(-1)[0])
        m = float(torch.linalg.norm(out['cam_param'][:, 6:9], dim=-1).mean())
        gi = [int(v) for v in out['geo_idxs']]
        w2c_t = sd.extrinsics_list[tgt][s].double().numpy()
        w2c_c = sd.extrinsics_list[ctx][gi[0]].double().numpy()
        A = w2c_t @ np.linalg.inv(w2c_c)
        tr = float(np.linalg.norm(A[:3, 3]) / max(ns, 1e-8))
        rows.append({'idx': int(i), 'tgt': int(tgt), 'ctx': int(ctx), 's': int(s), 'e': int(e),
                     'data_name': data_name, 'caption': caption, 'geo_idxs': gi,
                     'norm_scale': ns, 'm': m, 'rot_deg': rot_angle_deg(A[:3, :3]),
                     'trans': tr, 'ratio': tr / max(m, 1e-8)})
    return rows


def render_case(sd, r, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tgt, ctx, s = r['tgt'], r['ctx'], r['s']
    it = sd._load_images(sd.frame_files_list[tgt], [s])[0].permute(1, 2, 0).numpy()
    ic = sd._load_images(sd.frame_files_list[ctx], [r['geo_idxs'][0]])[0].permute(1, 2, 0).numpy()
    diff = np.abs(it - ic).mean(-1)

    wt = sd.extrinsics_list[tgt][s:r['e']].double().numpy()
    wc = sd.extrinsics_list[ctx][s:r['e']].double().numpy()
    Ct, Cc = cam_centers(wt), cam_centers(wc)
    ns = r['norm_scale']

    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.22, wspace=0.16)

    for k, (img, name) in enumerate([(it, f"target clip frame {s}\n{osp.basename(_clip(sd, tgt))}"),
                                     (ic, f"context clip frame {r['geo_idxs'][0]} (= DA3 view0)\n"
                                          f"{osp.basename(_clip(sd, ctx))}")]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    ax = fig.add_subplot(gs[0, 2])
    # vmax 를 max 로 잡으면 몇 픽셀짜리 outlier 가 전체를 까맣게 만든다 -> p99.5 로 자른다.
    im = ax.imshow(diff, cmap='inferno', vmin=0,
                   vmax=max(1e-3, float(np.percentile(diff, 99.5))))
    ax.set_title(f'|difference|  mean={diff.mean():.4f}  p99.5={np.percentile(diff, 99.5):.4f}',
                 fontsize=9)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.035)

    # ---- 3D: 두 clip 의 카메라 궤적을 정렬 후 world(미터) 에서
    ax3 = fig.add_subplot(gs[1, 0:2], projection='3d')
    ax3.plot(Ct[:, 0], Ct[:, 1], Ct[:, 2], '-', color='tab:red', lw=1.6, label='target clip')
    ax3.plot(Cc[:, 0], Cc[:, 1], Cc[:, 2], '-', color='tab:blue', lw=1.6, label='context clip')
    ax3.scatter(*Ct[0], color='tab:red', s=70, marker='o', label='target anchor (frame s)')
    ax3.scatter(*Cc[r['geo_idxs'][0]], color='tab:blue', s=70, marker='^',
                label='context anchor (DA3 view0)')
    ax3.plot([Ct[0, 0], Cc[r['geo_idxs'][0], 0]], [Ct[0, 1], Cc[r['geo_idxs'][0], 1]],
             [Ct[0, 2], Cc[r['geo_idxs'][0], 2]], '--', color='k', lw=2.0)
    _axis_equal(ax3, np.concatenate([Ct, Cc], 0))
    ax3.set_title('camera centers (aligned world, meters) - black dashed = anchor residual',
                  fontsize=9)
    ax3.legend(fontsize=7, loc='upper left')

    # ---- 숫자 패널
    axt = fig.add_subplot(gs[1, 2])
    axt.axis('off')
    txt = (f"data_name: {r['data_name']}\n\n"
           f"anchor rot err     {r['rot_deg']:.3f} deg\n"
           f"anchor trans err   {r['trans']:.4f}  (model unit)\n"
           f"                   {r['trans'] * ns:.4f} m\n"
           f"target motion m    {r['m']:.4f}\n"
           f"ratio = err / m    {r['ratio']:.3f}\n\n"
           f"norm_scale         {ns:.4f} m\n"
           f"geo_idxs           {r['geo_idxs']}\n\n"
           f"caption:\n{_wrap(r['caption'], 46)}")
    axt.text(0.0, 1.0, txt, va='top', ha='left', fontsize=8.5, family='monospace')

    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def _clip(sd, i):
    return osp.relpath(sd.scene_dir_list[i], sd.root)


def _wrap(s, w):
    s = str(s or '')
    out, line = [], ''
    for word in s.split():
        if len(line) + len(word) + 1 > w:
            out.append(line)
            line = word
        else:
            line = (line + ' ' + word).strip()
    out.append(line)
    return '\n'.join(out[:8])


def _axis_equal(ax, P):
    c = P.mean(0)
    r = max(float(np.abs(P - c).max()), 1e-3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel('x', fontsize=7)
    ax.set_ylabel('y', fontsize=7)
    ax.set_zlabel('z', fontsize=7)
    ax.tick_params(labelsize=6)


def render_hist(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ratio = np.array([r['ratio'] for r in rows])
    rot = np.array([r['rot_deg'] for r in rows])
    m = np.array([r['m'] for r in rows])
    tr = np.array([r['trans'] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    axes[0].hist(ratio, bins=40, color='tab:purple')
    axes[0].axvline(0.5, color='k', ls='--', lw=1)
    axes[0].set_title(f'anchor trans err / target motion m  (n={len(rows)}, '
                      f'>0.5: {100 * (ratio > 0.5).mean():.1f}%)', fontsize=9)
    axes[0].set_xlabel('ratio')
    axes[1].hist(rot, bins=40, color='tab:orange')
    axes[1].set_title(f'anchor rotation err (med {np.median(rot):.2f} deg, '
                      f'max {rot.max():.2f} deg)', fontsize=9)
    axes[1].set_xlabel('deg')
    # 꼬리가 "정렬이 나빠서"인지 "분모가 작아서"인지 가르는 패널.
    sc = axes[2].scatter(m, tr, c=np.log10(np.maximum(ratio, 1e-4)), cmap='coolwarm', s=14)
    x = np.linspace(m.min(), m.max(), 50)
    axes[2].plot(x, 0.5 * x, 'k--', lw=1, label='ratio = 0.5')
    axes[2].plot(x, 1.0 * x, 'k-', lw=1, label='ratio = 1.0')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('m = target motion magnitude')
    axes[2].set_ylabel('anchor trans err')
    # 꼬리의 원인을 분자/분모로 분해해서 제목에 그대로 쓴다 (한쪽으로 단정하지 않는다).
    hi = ratio > 0.5
    if hi.any():
        axes[2].set_title(f'ratio>0.5 (n={int(hi.sum())}): m x{np.median(m) / np.median(m[hi]):.1f} '
                          f'below med, err x{np.median(tr[hi]) / np.median(tr):.1f} above med',
                          fontsize=9)
    else:
        axes[2].set_title('no sample with ratio > 0.5', fontsize=9)
    axes[2].legend(fontsize=7)
    fig.colorbar(sc, ax=axes[2], label='log10 ratio')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='mix_dl3dv_sd_datadop_v1')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/mixed/sd_anchor_viz')
    ap.add_argument('--overrides', nargs='*', default=[])
    ap.add_argument('--reuse', action='store_true',
                    help='out/per_sample.json 이 있으면 재측정 없이 그림만 다시 그린다')
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for f in ['NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR', 'DejaVu Sans']:
        if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = f
            break
    plt.rcParams['axes.unicode_minus'] = False

    from hydra_cfg import load_cfg
    ov = list(a.overrides)
    if a.config and a.config != 'config':
        ov.append(f'experiment={a.config}')
    cfg, _ = load_cfg('config', overrides=ov)

    from base import build_dataset
    ds = build_dataset(cfg)
    if getattr(cfg, 'dataset_name', None) == 'mixed':
        sd = ds.sub[list(ds.names).index('scene_decoupled')]
    else:
        sd = ds

    cache = osp.join(a.out, 'per_sample.json')
    if a.reuse and osp.isfile(cache):
        rows = json.load(open(cache))
        print(f'[reuse] {cache} 에서 {len(rows)} 행 재사용 (측정 생략)')
    else:
        rng = np.random.default_rng(a.seed)
        pick = sorted(rng.choice(len(sd), min(a.n, len(sd)), replace=False).tolist())
        rows = measure(sd, pick)
    rows.sort(key=lambda r: r['ratio'])
    ratio = np.array([r['ratio'] for r in rows])
    print(f"n={len(rows)}  ratio med={np.median(ratio):.4f} p95={np.percentile(ratio, 95):.4f} "
          f"max={ratio.max():.4f}  >0.5 {100 * (ratio > 0.5).mean():.2f}%")

    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, 'per_sample.json'), 'w') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    render_hist(rows, osp.join(a.out, 'hist.png'))

    n = len(rows)
    cases = [('worst', rows[-1]), ('p99', rows[int(0.99 * (n - 1))]),
             ('p95', rows[int(0.95 * (n - 1))]), ('median', rows[n // 2]),
             ('best', rows[0])]
    for tag, r in cases:
        p = osp.join(a.out, f'{tag}_ratio{r["ratio"]:.3f}.png')
        render_case(sd, r, p, f"[{tag}] SD anchor residual  ratio={r['ratio']:.3f}  "
                              f"rot={r['rot_deg']:.2f} deg  trans={r['trans']:.4f}")
        print(f'  -> {p}')


if __name__ == '__main__':
    main()
