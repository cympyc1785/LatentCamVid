"""DA3 latent 이 실제로 쓰는 카메라 스케일과 우리가 넣어 준 스케일의 비율 sigma 를 잰다.

문제
----
`geo_posed=True` 로 `cam_enc` 에 pose 를 넣어 줘도, DA3 가 latent 에서 뽑는 카메라의
**translation 은 항상 예측값**이다 (`cam_dec.py:35` 의 `out_t = self.fc_t(feat)` 에는 echo
경로가 없고, rotation/fov 만 `camera_encoding` 이 주어질 때 echo 된다. 게다가 `da3.py:216` 은
`camera_encoding` 없이 부른다). 즉 latent 이 표현하는 카메라 공간의 스케일은 우리가 넣어 준
스케일(view0 재고정 + per-sample median camera distance M)과 다르다.

  c_fed   = (GT 카메라 center, view0 기준) / M          <- cam_enc 에 넣은 것
  c_pred  = cam_dec(feats[-1][1]) -> c2w 의 center       <- latent 이 실제로 보는 것
  sigma   = Umeyama(source=c_pred, target=c_fed).scale
  => c_fed ~ sigma * R * c_pred + t,  즉 GT 상대좌표 = M * c_fed = (M*sigma) * c_pred
  => **latent 의 1 단위 = M*sigma GT 미터**. target translation 을 M*sigma 로 나누면
     target 궤적이 latent 과 같은 공간에 놓인다.

이 스크립트는 그 sigma / M*sigma 의 분포를 재기만 한다 (파일을 쓰지 않는다).
sigma 가 1 근처에 몰려 있으면 재정규화 arm 은 no-op 이므로 여기서 멈춰야 한다.

usage (main/ 에서):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<latentcam>:. \
    python ../scripts/data/da3_latent_scale_probe.py \
      experiment=da3_7k_da3geo_frontanchor_samelen --n 200 --max-scenes 60
"""
import argparse
import sys

import numpy as np
import torch

from hydra_cfg import load_cfg


def umeyama_scale(src, dst):
    """Umeyama similarity 의 scale 만. src/dst (N,3). 회전까지 푼 뒤의 optimal scale 이다."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    var_s = (s0 ** 2).sum() / len(src)
    if var_s < 1e-20:
        return np.nan, np.nan
    cov = (d0.T @ s0) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / var_s)
    resid = float(np.sqrt(((d0 - scale * (R @ s0.T).T) ** 2).sum(1)).mean())
    return scale, resid


def stats(name, v):
    v = np.asarray([x for x in v if np.isfinite(x)], np.float64)
    if not len(v):
        print(f"  {name:22s} (empty)")
        return
    q = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"  {name:22s} n={len(v):5d} mean {v.mean():10.5f} med {q[2]:10.5f} "
          f"p05 {q[0]:10.5f} p25 {q[1]:9.5f} p75 {q[3]:9.5f} p95 {q[4]:10.5f} "
          f"min {v.min():9.5f} max {v.max():10.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--max-scenes', type=int, default=60)
    # hydra 토큰(experiment=...)은 load_cfg 가 sys.argv 에서 직접 읽으므로 여기선 무시한다.
    args, _ = ap.parse_known_args()

    cfg, _ = load_cfg('config')
    if str(getattr(cfg, 'geo_encoder', None)) != 'da3':
        raise SystemExit(f"geo_encoder must be 'da3' (got {cfg.geo_encoder!r})")
    cfg.max_scenes = args.max_scenes

    from dataset_dl3dv import CamDataset
    from models.da3_geo_encoder import DA3SceneEncoder

    device = 'cuda'
    torch.set_grad_enabled(False)
    enc = DA3SceneEncoder(
        repo_path=cfg.da3_geo_repo_path, model_name=cfg.da3_geo_model,
        ckpt_path=getattr(cfg, 'da3_geo_ckpt_path', None),
        hf_home=getattr(cfg, 'da3_geo_hf_home', None),
        input_hw=getattr(cfg, 'da3_geo_input_hw', None),
        layers=getattr(cfg, 'da3_geo_layers', 'last'),
        keep_cam_dec=True,                     # <- cam_dec 를 살린다 (기본은 삭제)
    ).to(device).eval()

    ds = CamDataset(cfg, 'train')
    n = min(args.n, len(ds))
    print(f"dataset {len(ds)} samples -> probing {n} | view_sampling="
          f"{getattr(cfg, 'geo_view_sampling', None)} avg_scale_ref={cfg.avg_scale_ref}", flush=True)

    sig, med_M, latent_div, ratio_vs_avg, resids, avgs = [], [], [], [], [], []
    for i in range(n):
        d = ds[i]
        if 'geo_c2w' not in d:
            raise SystemExit("geo_c2w 가 없다 — geo_posed=True 여야 한다")
        images = d['images'].to(device).unsqueeze(0)
        c2w = d['geo_c2w'].to(device).unsqueeze(0).float()
        fx = d['geo_fxfycxcy'].to(device).unsqueeze(0).float()
        hw = d['geo_hw'].to(device).unsqueeze(0)

        # cam_enc 에 실제로 들어가는 정규화 카메라 center 를 encoder 와 **같은 식**으로 재현한다.
        from depth_anything_3.utils.geometry import affine_inverse
        w2c = affine_inverse(c2w) @ c2w[:, :1]
        c2w_n = affine_inverse(w2c)
        M = float(c2w_n[..., :3, 3].norm(dim=-1).median(dim=1).values.clamp(min=1e-1)[0])
        c_fed = (c2w_n[0, :, :3, 3] / M).cpu().numpy()

        tok = enc.build_cam_token(c2w, fx, hw)
        c2w_p, _ = enc.predict_cameras(images, cam_token=tok)
        c_pred = c2w_p[0, :, :3, 3].cpu().numpy()

        s, r = umeyama_scale(c_pred, c_fed)
        a = float(d['norm_scale']) if 'norm_scale' in d else float('nan')
        sig.append(s); med_M.append(M); resids.append(r); avgs.append(a)
        latent_div.append(M * s)
        ratio_vs_avg.append((M * s) / a if a and np.isfinite(a) else np.nan)
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{n}", flush=True)

    print("\n=== sigma = Umeyama scale (pred -> fed). 1.0 이면 latent 이 넣어 준 스케일 그대로 ===")
    stats("sigma", sig)
    stats("umeyama resid (fed 단위)", resids)
    print("\n=== 분모 후보 ===")
    stats("M (median cam dist)", med_M)
    stats("M*sigma (= 새 분모)", latent_div)
    stats("avg_scale (현 분모)", avgs)
    stats("M*sigma / avg_scale", ratio_vs_avg)
    v = np.asarray([x for x in sig if np.isfinite(x)])
    print(f"\nsigma 가 [0.95,1.05] 안인 비율: {float((np.abs(v - 1) <= 0.05).mean()):.4f}  "
          f"(1 에 가까울수록 재정규화 arm 은 no-op)")


if __name__ == '__main__':
    main()
