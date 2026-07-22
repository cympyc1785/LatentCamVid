"""Verify the camera VAE currently used in training (vae_intr_large.CameraVAE,
cfg.vae_ckpt_path, cfg.vae_latent_scale):
  1) reconstruction quality on REAL current-pipeline cam_param (dataset_dl3dv) —
     per-component error + rotation geodesic + latent-scale calibration,
  2) latent interpolation smoothness/validity,
  3) scene-scale normalization sanity (is it applied, camera-based, re-applied at decode).

Run:  python scripts/verify_vae.py  (from latentcam/, env latentcam)
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..', 'main'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as C
cfg = C.cfg
cfg.geo_encoder = None            # cam_param is geo-independent; skip VGGT
cfg.max_scenes = 6
from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE
from utils.data_utils import out_to_trajectory
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = osp.join(osp.dirname(__file__), 'vae_verify.png')


def centers_from_camparam(cp, scale=None, e0=None):
    """cp: (B,T,11) -> camera centers (B,T,3). Uses out_to_trajectory (w2c) then inverts.
    Default normalized space (scale=1, e0=identity)."""
    B, T, _ = cp.shape
    if scale is None:
        scale = torch.ones(B, 1)
    if e0 is None:
        e0 = torch.eye(4)[None].repeat(B, 1, 1)
    w2c = out_to_trajectory(cp[:, :, :9], scale, e0, device='cpu')   # (B,T,4,4)
    c2w = torch.linalg.inv(w2c)
    return c2w[:, :, :3, 3]


def geodesic_deg(r6_a, r6_b):
    Ra = compute_rotation_matrix_from_ortho6d(r6_a.reshape(-1, 6))
    Rb = compute_rotation_matrix_from_ortho6d(r6_b.reshape(-1, 6))
    tr = torch.diagonal(Ra.transpose(1, 2) @ Rb, dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.arccos(torch.clamp((tr - 1) / 2, -1, 1)))


def main():
    ds = CamDataset(cfg)
    n = min(48, len(ds))
    cams = torch.stack([ds[i]['cam_param'] for i in range(n)])          # (n,49,11)
    scales = torch.stack([ds[i]['avg_scale'] for i in range(n)]).squeeze(-1)  # (n,)

    vae = CameraVAE(latent_dim=cfg.cam_dim).to(dev)
    sd = torch.load(cfg.vae_ckpt_path, map_location=dev)
    miss, unexp = vae.load_state_dict(sd, strict=False)
    vae.eval()
    print(f"VAE: {osp.basename(cfg.vae_ckpt_path)}  latent_dim={cfg.cam_dim} "
          f"(missing={len(miss)}, unexpected={len(unexp)}) | vae_latent_scale={cfg.vae_latent_scale}")

    with torch.no_grad():
        x = cams.to(dev)
        z = vae.encode(x)                       # (n,13,64)
        z_scaled = z / cfg.vae_latent_scale     # what training actually uses
        rec = vae.decode(z)                     # (n,49,11)
    x, rec, z = x.cpu(), rec.cpu(), z.cpu()

    # ---- (1) reconstruction metrics ----
    mse = lambda a, b: (a - b).pow(2).mean().item()
    rot_mse = mse(x[..., :6], rec[..., :6])
    tr_mse = mse(x[..., 6:9], rec[..., 6:9])
    intr_mse = mse(x[..., 9:11], rec[..., 9:11])
    geo = geodesic_deg(x[..., :6], rec[..., :6])
    tr_l2 = (x[..., 6:9] - rec[..., 6:9]).norm(dim=-1)          # normalized units
    tr_l2_real = (tr_l2 * scales[:, None]).mean().item()        # real units (×avg_scale)
    print("\n== (1) Reconstruction (n=%d segments, 49 frames) ==" % n)
    print(f"  overall MSE      : {mse(x, rec):.6e}")
    print(f"  rot6d   MSE      : {rot_mse:.6e}")
    print(f"  trans   MSE(norm): {tr_mse:.6e}   | mean |Δtrans| norm={tr_l2.mean():.4f}  real={tr_l2_real:.4f}")
    print(f"  intr    MSE      : {intr_mse:.6e}")
    print(f"  rotation geodesic: mean={geo.mean():.3f} deg  median={geo.median():.3f}  p95={geo.quantile(0.95):.3f}")
    print(f"  latent stats: mean|z|={z.abs().mean():.4f}  std(z)={z.std():.4f}  "
          f"(vae_latent_scale={cfg.vae_latent_scale}; scaled std={ (z/cfg.vae_latent_scale).std():.4f} -> want ~1)")

    # ---- (3) scene-scale sanity ----
    print("\n== (3) Scene-scale normalization ==")
    print(f"  dataset avg_scale (camera-based, per-seg): mean={scales.mean():.3f} "
          f"min={scales.min():.3f} max={scales.max():.3f}")
    print(f"  normalized |trans| fed to VAE: mean={x[...,6:9].norm(dim=-1).mean():.4f} "
          f"(O(1) => scale IS applied in dataset)")
    print("  flow: dataset divides trans by camera-based avg_scale -> VAE encode ->"
          " decode -> out_to_trajectory multiplies trans by avg_scale (re-applied).")

    # ---- (2) latent interpolation ----
    i, j = 0, 1
    with torch.no_grad():
        z1 = vae.encode(cams[i:i+1].to(dev)); z2 = vae.encode(cams[j:j+1].to(dev))
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        interp = [vae.decode((1 - a) * z1 + a * z2).cpu() for a in alphas]

    # ---- figure ----
    fig = plt.figure(figsize=(16, 8))
    # row 1: GT vs recon (4 samples), row 2: interpolation morph
    gt_c = centers_from_camparam(x[:4])
    rc_c = centers_from_camparam(rec[:4])
    for k in range(4):
        ax = fig.add_subplot(2, 4, k + 1)
        g, r = gt_c[k].numpy(), rc_c[k].numpy()
        ax.plot(g[:, 0], g[:, 2], '-o', ms=2, c='tab:blue', label='GT')
        ax.plot(r[:, 0], r[:, 2], '-x', ms=3, c='tab:red', label='recon')
        ax.set_title(f'recon #{k} (x-z top-down)', fontsize=9)
        if k == 0:
            ax.legend(fontsize=7)
        ax.set_aspect('equal', 'datalim')
    for c, (a, cp) in enumerate(zip(alphas, interp)):
        ax = fig.add_subplot(2, 5, 5 + c + 1)
        cc = centers_from_camparam(cp)[0].numpy()
        ax.plot(cc[:, 0], cc[:, 2], '-o', ms=2, c=plt.cm.viridis(a))
        ax.set_title(f'interp a={a}', fontsize=9)
        ax.set_aspect('equal', 'datalim')
    fig.suptitle('VAE verify: top row = GT(blue) vs recon(red); bottom = latent interpolation #0->#1')
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    print("\nsaved:", OUT)


if __name__ == '__main__':
    main()
