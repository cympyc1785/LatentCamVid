"""diffusion **타깃 그 자체**의 통계를 pose_source 별로 잰다.

왜 필요한가
-----------
pose_source_agreement.py 는 궤적 요약량(reach)만 봤다. 실제 학습이 맞추는 것은

    traj    = data['cam_param']                       (T,11) rot6d(6)+trans(3)+intr(2)
              trans 는 이미 avg_scale 로 나눈 값이다 (dataset_dl3dv._target_out)
    target  = camera_vae.encode(traj) / cfg.vae_latent_scale     (train_latent_cam_dm.py:600)

이므로, "avg_scale 이 원인이냐"는 질문은 **이 두 텐서의 분포**로 답해야 한다.
특히 `vae_latent_scale` 은 두 arm 모두 0.96032625 로 **고정된 상수**다. 이 상수의 역할은
latent 를 std≈1 로 만들어 DDPM 노이즈 스케줄의 SNR 을 맞추는 것이므로, arm 별 latent std 가
1 에서 얼마나 벗어나는지가 곧 스케줄 오정합의 크기다.

두 arm 은 같은 seg list 를 쓰므로 **같은 segment 집합**에서 잰다 (paired).

usage
-----
  python scripts/eval/target_scale_stats.py --n 2000
  python scripts/eval/target_scale_stats.py --cfg-a <runA>/config.yaml --cfg-b <runB>/config.yaml
out -> stdout + results/compare/target_scale_stats/summary.json
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


def _pct(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    return {f'p{q}': float(np.percentile(a, q)) for q in qs} if a.size else {}


def collect(cfg, n, seed=0):
    """test_seg_list 의 앞 n 개 segment 에서 cam_param / norm_scale 을 모은다."""
    from base import build_dataset
    ds = build_dataset(cfg)
    seg_key = type(ds).seg_key
    id2idx = {}
    for i, s in enumerate(ds.samples):
        id2idx.setdefault(seg_key(s[4]), i)
    with open(cfg.test_seg_list) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    idx = [id2idx[x] for x in ids if x in id2idx]
    rng = np.random.default_rng(seed)
    if n and n < len(idx):
        idx = [idx[i] for i in sorted(rng.choice(len(idx), n, replace=False))]
    traj, ns, names = [], [], []
    for i in idx:
        d = ds[i]
        traj.append(d['cam_param'])
        ns.append(float(d['norm_scale']))
        names.append(seg_key(ds.samples[i][4]))
    return torch.stack(traj), np.asarray(ns), names


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


def stats(traj, ns, lat, scale):
    """traj (N,T,11), lat (N,L,C). scale = cfg.vae_latent_scale."""
    t = traj.numpy().astype(np.float64)
    rot6d, trans, intr = t[..., :6], t[..., 6:9], t[..., 9:11]
    tn = np.linalg.norm(trans, axis=-1)                    # (N,T) 프레임별 |t|
    z = lat.numpy().astype(np.float64) / scale             # 실제 diffusion 타깃
    return {
        'n_segments': int(t.shape[0]),
        'norm_scale(avg_scale)': {'mean': float(ns.mean()), **_pct(ns)},
        'cam_param/trans': {
            'std_all': float(trans.std()),
            'std_per_channel': [float(trans[..., c].std()) for c in range(3)],
            'abs_mean': float(np.abs(trans).mean()),
            'norm_per_frame': _pct(tn),
            'norm_last_frame': _pct(tn[:, -1]),
        },
        'cam_param/rot6d': {'std_all': float(rot6d.std()), 'abs_mean': float(np.abs(rot6d).mean())},
        'cam_param/intr': {'std_all': float(intr.std()), 'mean': float(intr.mean())},
        'vae_target (encode/vae_latent_scale)': {
            'vae_latent_scale': float(scale),
            'std_all': float(z.std()),
            'mean_all': float(z.mean()),
            'abs_mean': float(np.abs(z).mean()),
            'std_per_channel': _pct([z[..., c].std() for c in range(z.shape[-1])]),
            'implied_vae_latent_scale': float((z * scale).std()),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg-a', default='results/20260808_140209_da3_7k_da3pose/config.yaml')
    ap.add_argument('--cfg-b', default='results/20260809_212844_da3_7k_textonly/config.yaml')
    ap.add_argument('--label-a', default='DA3')
    ap.add_argument('--label-b', default='COLMAP')
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--out', default='results/compare/target_scale_stats')
    a = ap.parse_args()

    from omegaconf import OmegaConf
    out = {}
    keep = {}
    for lab, p in [(a.label_a, a.cfg_a), (a.label_b, a.cfg_b)]:
        cfg = OmegaConf.load(p)
        traj, ns, names = collect(cfg, a.n)
        lat = vae_latents(cfg, traj)
        out[lab] = {'config': p, 'pose_source': str(cfg.get('pose_source', 'transforms')),
                    **stats(traj, ns, lat, float(cfg.vae_latent_scale))}
        keep[lab] = names
        print(f'[{lab}] {len(names)} segments done', flush=True)

    ka, kb = keep[a.label_a], keep[a.label_b]
    out['paired'] = {'same_segments': ka == kb, 'n_common': len(set(ka) & set(kb))}

    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print('\nout ->', osp.join(a.out, 'summary.json'))


if __name__ == '__main__':
    main()
