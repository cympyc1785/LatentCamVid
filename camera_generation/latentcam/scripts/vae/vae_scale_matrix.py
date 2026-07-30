"""Measure the camera-VAE latent std + reconstruction over the FULL (ckpt x scale_mode x
intrinsics-convention) matrix, to pin down which (vae_ckpt, vae_latent_scale) pair a config
value like SCVideo's 0.4467666 / 0.96032625 actually belongs to.

Why the intrinsics axis is separate: dataset_dl3dv couples the intrinsics encoding to
scale_mode ('avg_scale' -> SCVideo's frame0-relative intr, everything else -> fx/w,fy/h),
but in SCVideo those are INDEPENDENT choices (dataset_large.py:313 `normalized_intrinsics /
normalized_intrinsics[0:1]` was added at some point, so older VAE ckpts saw the raw fx/w,fy/h
under the SAME avg_scale translation normalization). So we take cam_param from the dataset and
overwrite channels 9:11 with each convention:
  rel = (fx/(2cx), fy/(2cy)) / frame0   -> exactly 1.0 for DL3DV (constant intrinsics/scene)
  raw = fx/(2cx), fy/(2cy)              -> (0.448, 0.796) for DL3DV
`vae_latent_scale` must equal the reported latent std for the pair (train divides by it), and
the recon L1 columns say whether the ckpt is in-distribution at all.

Run:  python scripts/vae/vae_scale_matrix.py            (from latentcam/, env latentcam)
env:  MAX_SCENES (default 200), MODES, CKPTS, META (default meta_worldtraj.csv)
"""

import os, os.path as osp, sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..', 'main'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
import torch

import config as C
cfg = C.cfg
cfg.geo_encoder = None                 # cam_param is geo-independent; skip VGGT
cfg.max_scenes = int(os.environ.get('MAX_SCENES', 200))
cfg.meta_csv = os.environ.get('META', 'meta_worldtraj.csv')
cfg.lazy_dataset = True
from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = osp.join(osp.dirname(__file__), '..', '..', 'checkpoints')
CKPTS = {                              # name -> (file relative to checkpoints/, latent_dim)
    'vae_20260302_300': ('vae_20260302_300.pth', 64),
    'vae_20260202_065659_400': ('vae_20260202_065659_400.pth', 32),
    # 1K-7K re-fit of the 32-dim ckpt above (experiment=vae_dl3dv_1_7k) -- drop-in, same
    # architecture/keys, so it belongs in the same matrix for a like-for-like comparison
    'vae_dl3dv_1_7k': ('../my_checkpoints/vae_dl3dv_1_7k/last.pth', 32),
}
MODES = os.environ.get(
    'MODES', 'avg_scale,cam_dist_mean,first_farthest_135,context_longer').split(',')
ONLY = [c for c in os.environ.get('CKPTS', '').split(',') if c]
if ONLY:
    CKPTS = {k: v for k, v in CKPTS.items() if k in ONLY}


def load_vae(fname, dim):
    m = CameraVAE(latent_dim=dim).to(dev).eval()
    sd = torch.load(osp.join(CKPT_DIR, fname), map_location='cpu')
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd)
    return m


def intr_variants(intrinsics):
    """intrinsics (B,T,3,3) -> {'rel': (B,T,2), 'raw': (B,T,2)} exactly as SCVideo builds them
    (width/height from the principal point, i.e. 2cx/2cy -- not the transforms.json w/h)."""
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    w, h = intrinsics[..., 0, 2] * 2, intrinsics[..., 1, 2] * 2
    raw = torch.stack([fx / w, fy / h], dim=-1)
    return {'rel': raw / raw[:, 0:1, :], 'raw': raw}


@torch.no_grad()
def main():
    vaes = {k: load_vae(*v) for k, v in CKPTS.items()}
    rows = []
    for mode in MODES:
        cfg.scale_mode = mode
        ds = CamDataset(cfg=cfg, type='train')
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
        # accumulate per (ckpt, intr convention)
        acc = {(c, iv): {'n': 0, 'sq': 0.0, 'rot': 0.0, 'tr': 0.0, 'intr': 0.0, 'el': 0}
               for c in vaes for iv in ('rel', 'raw')}
        n_samples = 0
        for batch in dl:
            cp = batch['cam_param'].to(dev)                  # (B,T,11)
            ivs = intr_variants(batch['intrinsics'].to(dev))
            n_samples += cp.shape[0]
            for iv, intr in ivs.items():
                x = torch.cat([cp[..., :9], intr], dim=-1)
                for cname, vae in vaes.items():
                    z = vae.encode(x)                        # (B, num_cam, D)
                    xr = vae.decode(z)                       # (B, T, 11)
                    a = acc[(cname, iv)]
                    a['sq'] += (z.double() ** 2).sum().item(); a['el'] += z.numel()
                    d = (xr - x).abs()
                    a['rot'] += d[..., :6].mean().item() * cp.shape[0]
                    a['tr'] += d[..., 6:9].mean().item() * cp.shape[0]
                    a['intr'] += d[..., 9:11].mean().item() * cp.shape[0]
                    a['n'] += cp.shape[0]
        for (cname, iv), a in acc.items():
            rows.append((cname, CKPTS[cname][1], mode, iv,
                         (a['sq'] / a['el']) ** 0.5,          # latent std (mean 0 assumed)
                         a['rot'] / a['n'], a['tr'] / a['n'], a['intr'] / a['n']))
        print(f"[{mode}] {n_samples} samples", flush=True)

    print(f"\n{'ckpt':<26} {'dim':>3} {'scale_mode':<19} {'intr':<4} "
          f"{'lat.std':>8} {'/0.96033':>8} {'/0.44677':>8} {'rot':>8} {'trans':>8} {'intr_L1':>8}")
    for c, dim, mode, iv, std, rot, tr, it in rows:
        print(f"{c:<26} {dim:>3} {mode:<19} {iv:<4} {std:>8.5f} {std/0.96032625:>8.4f} "
              f"{std/0.4467666:>8.4f} {rot:>8.5f} {tr:>8.5f} {it:>8.5f}")


if __name__ == '__main__':
    main()
