"""val/loss_traj, val/loss_latent 의 **"평균 궤적만 내놓는 모델"** 기준선을 코퍼스별로 잰다.

왜 필요한가
-----------
train_latent_cam_dm.py 는

    val_loss_latent = F.mse_loss(out, traj_latents)        # VAE latent / vae_latent_scale 공간
    val_loss_traj   = F.mse_loss(traj_pred, traj)          # cam_param (T,11) 공간

이라 **타깃 텐서의 분산이 코퍼스마다 다르면 두 수는 서로 비교 불가**다. Scene-Decoupled(SD)
처럼 카메라 프리셋이 5 종뿐인 코퍼스는 타깃 분산 자체가 작아서, 아무것도 학습하지 않고
"코퍼스 평균 궤적"만 출력해도 loss 가 작게 나온다.

그래서 각 코퍼스의 val 집합에서

    baseline_traj = mean_{n,t,c} ( x[n,t,c] - mean_n x[n,t,c] )^2      (per-(t,c) 평균 궤적)

을 재고, 실제 run 의 val/loss_traj 를 이 기준선으로 나눈 비율(=1 - R^2)을 본다. 비율이
1 에 가까우면 "평균만 맞히는 중", 0 에 가까우면 실제로 궤적을 맞히는 중이다.

geo 는 필요 없으므로 cfg.geo_encoder 를 None 으로 눌러 이미지 I/O 를 건너뛴다
(dataset_dl3dv.py:296 geo_enabled). cam_param 경로는 geo 와 무관하다.

usage
-----
  python scripts/eval/mean_traj_baseline.py \
      --cfg SD=results/20260810_010756_sd_whuman_textonly/config.yaml \
      --cfg DA3=results/20260808_140209_da3_7k_da3pose/config.yaml \
      --n 600
out -> stdout + results/compare/mean_traj_baseline/summary.json
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, '..', '..'))          # .../latentcam
sys.path[:0] = [osp.join(ROOT, 'main'), ROOT, osp.join(ROOT, 'data')]


def collect(cfg, n, seed=0):
    """test_seg_list(=val 집합) 에서 최대 n 개 segment 의 cam_param 을 모은다."""
    from base import build_dataset
    ds = build_dataset(cfg)
    seg_key = type(ds).seg_key
    id2idx = {}
    for i, s in enumerate(ds.samples):
        id2idx.setdefault(seg_key(s[4]), i)
    with open(cfg.test_seg_list) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    idx = [id2idx[x] for x in ids if x in id2idx]          # val 순서 = 파일 순서
    n_val = len(idx)
    if n and n < len(idx):
        rng = np.random.default_rng(seed)
        idx = [idx[i] for i in sorted(rng.choice(len(idx), n, replace=False))]
    traj, names = [], []
    for i in idx:
        traj.append(ds[i]['cam_param'])
        names.append(seg_key(ds.samples[i][4]))
    return torch.stack(traj), names, n_val


def vae_latents(cfg, traj, device='cuda', bs=64):
    from models.vae_intr_large import CameraVAE
    vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
    vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
    vae.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(traj), bs):
            out.append(vae.encode(traj[i:i + bs].to(device)).float().cpu())
    del vae
    torch.cuda.empty_cache()
    return torch.cat(out)


def baseline(x):
    """x (N, ...) -> 평균-예측 MSE 두 가지."""
    x = np.asarray(x, dtype=np.float64)
    per_elem = float(((x - x.mean(0, keepdims=True)) ** 2).mean())   # per-(t,c) 평균 궤적
    scalar = float(((x - x.mean()) ** 2).mean())                     # 전역 스칼라 평균
    return {'mse_vs_mean_traj': per_elem, 'mse_vs_scalar_mean': scalar,
            'rmsd_vs_mean_traj': float(np.sqrt(per_elem))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', action='append', required=True,
                    help='LABEL=path/to/config.yaml (반복 가능)')
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--no-vae', action='store_true')
    ap.add_argument('--out', default='results/compare/mean_traj_baseline')
    a = ap.parse_args()

    from omegaconf import OmegaConf
    out = {}
    for spec in a.cfg:
        lab, p = spec.split('=', 1)
        cfg = OmegaConf.load(p)
        OmegaConf.set_struct(cfg, False)
        cfg.geo_encoder = None                      # 이미지/depth I/O 생략
        traj, names, n_val = collect(cfg, a.n)
        rec = {
            'config': p,
            'dataset_name': str(cfg.get('dataset_name', 'dl3dv')),
            'pose_source': str(cfg.get('pose_source', 'transforms')),
            'vae_latent_scale': float(cfg.vae_latent_scale),
            'n_val_total': n_val, 'n_sampled': len(names),
            'traj_shape': list(traj.shape),
            'cam_param': baseline(traj.numpy()),
        }
        if not a.no_vae:
            z = vae_latents(cfg, traj).numpy() / float(cfg.vae_latent_scale)
            rec['vae_target'] = {**baseline(z), 'std_all': float(z.std())}
        out[lab] = rec
        print(f'[{lab}] {len(names)}/{n_val} segments  '
              f'cam_param baseline={rec["cam_param"]["mse_vs_mean_traj"]:.6f}', flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print('\nout ->', osp.join(a.out, 'summary.json'))


if __name__ == '__main__':
    main()
