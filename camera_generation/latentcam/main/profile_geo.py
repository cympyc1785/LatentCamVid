"""Latency breakdown for the GEO (LagerNVS) conditioning path — why per-segment gen/train is
slow when context changes each segment (VGGT latent re-extracted every time, no cache).

Per sample times (CUDA-synced):
  (1) ds[idx]            : dataset getitem = frustum_cover view selection + load 6 images (CPU/disk)
  (2) build_cam_token    : VGGT pose-encoding of the 6 context cams
  (3) reconstructor(...) : VGGT forward over 6 images  <-- the "latent 매번 새로 뽑기"
  (4) proj               : geo token projection
  (5) DiT forward        : one denoiser step (reference)

Run: CUDA_VISIBLE_DEVICES=3 LATENTCAM_CONFIG=config_geo_viewS \
     PYTHONPATH=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam python profile_geo.py --n 8
"""
import os, importlib, argparse, time
import numpy as np, torch

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config_geo_viewS', overrides=[])
from models.geo_encoder import build_geo_encoder
from models.camera_diffusion_model_latent import CameraDiffusionModel
from dataset_dl3dv import CamDataset


def sync(): torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8, help='#samples to time')
    ap.add_argument('--max-scenes', type=int, default=6)
    args = ap.parse_args()
    device = 'cuda'; torch.set_grad_enabled(False)

    geo = build_geo_encoder(cfg).to(device).eval()
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim, geo_encoder=cfg.geo_encoder).to(device).eval()
    cfg.max_scenes = args.max_scenes
    ds = CamDataset(cfg, 'train')
    N = min(args.n, len(ds))

    stg = {k: [] for k in ['getitem', 'cam_token', 'vggt', 'proj', 'dit']}
    for i in range(N + 2):                      # +2 warmup
        rec = (i >= 2)
        t = time.time(); d = ds[i % len(ds)]; dt_get = time.time() - t
        images = d['images'].to(device).unsqueeze(0)
        has_pose = 'geo_c2w' in d and getattr(cfg, 'geo_posed', False)

        sync(); t = time.time()
        cam_token = None
        if has_pose:
            cam_token = geo.build_cam_token(d['geo_c2w'].to(device).unsqueeze(0),
                                            d['geo_fxfycxcy'].to(device).unsqueeze(0),
                                            d['geo_hw'].to(device).unsqueeze(0))
        sync(); dt_tok = time.time() - t

        sync(); t = time.time()
        tokens, mask = geo.backend(images, cam_token)
        sync(); dt_vggt = time.time() - t

        sync(); t = time.time(); geo_emb = geo.proj(tokens); geo_mask = mask; sync(); dt_proj = time.time() - t

        B = 1; W, Dd = cfg.num_cam, cfg.cam_dim
        z = torch.randn(B, W, Dd, device=device); ts = torch.rand(B, device=device) * 1000
        te = torch.randn(B, cfg.text_len, 4096, device=device); tm = torch.ones(B, cfg.text_len, dtype=torch.bool, device=device)
        sync(); t = time.time()
        _ = model(z, ts, te, tm, geo_emb, geo_mask)
        sync(); dt_dit = time.time() - t

        if rec:
            stg['getitem'].append(dt_get); stg['cam_token'].append(dt_tok)
            stg['vggt'].append(dt_vggt); stg['proj'].append(dt_proj); stg['dit'].append(dt_dit)

    print(f"\n=== geo path latency, mean of {N} samples (V={cfg.geo_cover_k} imgs @ {cfg.geo_image_hw}) ===")
    tot = sum(np.mean(v) for v in stg.values())
    for k in ['getitem', 'cam_token', 'vggt', 'proj', 'dit']:
        m = np.mean(stg[k]) * 1000
        print(f"  {k:10s}: {m:8.1f} ms  ({100*np.mean(stg[k])/tot:4.1f}%)")
    print(f"  {'TOTAL':10s}: {tot*1000:8.1f} ms/sample")
    print(f"\nVGGT re-encode is {100*np.mean(stg['vggt'])/tot:.0f}% of per-sample cost; it re-runs "
          f"every segment because context views change (no cache).")


if __name__ == '__main__':
    main()
