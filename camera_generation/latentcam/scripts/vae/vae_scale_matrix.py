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
      [new] CFG=k=v,k=v  arbitrary config.py overrides applied before the dataset is built.
      Needed for the retrieval-DEPENDENT scale_modes ('geo_lagernvs' calls
      _sample_geo_frustum_cover, so its divisor depends on the geo_cover_* flags): without it
      config.py's defaults (all off) give a DIFFERENT D than the experiment configs, e.g.
        CFG=geo_view_sampling=frustum_cover,geo_cover_out_of_seg=true,geo_first_view_target_s=true,
            geo_cover_subtract_first=true,geo_cover_centered_at_s=true
      Leakage-free modes ('avg_scale', 'ctx_longer_135max', ...) are unaffected by these.
      [new] DATASET=dl3dv (default, unchanged) | dynamicverse | both, DV_CHUNKS (default 3).
      'both' concatenates the DL3DV CamDataset and the dynamicverse_shim, so the reported
      latent std is the pooled one. Caveat: DynamicVerse samples are n_chunks NON-overlapping
      [k*T,(k+1)*T) chunks per scene, not DL3DV's sliding-window segments, so 'both' weights
      the two corpora by those counts. (Its stored avg_scale IS keyed by that same chunk cut,
      so the 'avg_scale' row is a real measurement -- see the shim's "avg_scale hit/total".)
"""

import os, os.path as osp, sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..', 'main'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
import torch

import config as C
cfg = C.cfg
cfg.geo_encoder = None                 # cam_param is geo-independent; skip VGGT
_ms = os.environ.get('MAX_SCENES', '200')          # [new] 'none'/'0' -> FULL corpus (권장)
cfg.max_scenes = None if _ms.lower() in ('none', 'null', '0', '') else int(_ms)
cfg.meta_csv = os.environ.get('META', 'meta_worldtraj.csv')
cfg.lazy_dataset = True
for _kv in os.environ.get('CFG', '').split(','):        # [new] see docstring
    if _kv.strip():
        _k, _v = _kv.strip().split('=', 1)
        _lv = _v.lower()
        setattr(cfg, _k, True if _lv == 'true' else False if _lv == 'false'
                else None if _lv in ('none', 'null')
                else int(_v) if _v.lstrip('-').isdigit()
                else float(_v) if _v.replace('.', '', 1).lstrip('-').isdigit() else _v)
        print(f"[CFG] {_k} = {getattr(cfg, _k)!r}")
from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE

DATASET = os.environ.get('DATASET', 'dl3dv')       # [new] dl3dv | dynamicverse | both
DV_CHUNKS = int(os.environ.get('DV_CHUNKS', 3))
if DATASET not in ('dl3dv', 'dynamicverse', 'both'):
    raise SystemExit(f"DATASET must be dl3dv|dynamicverse|both, got {DATASET!r}")


def build_dataset(mode):
    """[new] DATASET 분기. 'dl3dv' 는 기존 동작 그대로(CamDataset 하나)."""
    cfg.scale_mode = mode
    parts = []
    if DATASET in ('dl3dv', 'both'):
        parts.append(CamDataset(cfg=cfg, type='train'))
    if DATASET in ('dynamicverse', 'both'):
        from dynamicverse_shim import load_dynamicverse
        # [2026-08-02] DynamicVerse 도 avg_scale/<k>.json 이 있고 chunk 와 1:1 이다.
        # 커버리지("avg_scale hit/total")는 load_dynamicverse 가 찍는다 -- 하나라도 비면
        # 그 chunk 는 조용히 _cam_dist_mean_scale 로 fallback 된다.
        parts.append(load_dynamicverse(cfg, n_chunks=DV_CHUNKS))
    return parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = osp.join(osp.dirname(__file__), '..', '..', 'checkpoints')
CKPTS = {                              # name -> (file relative to checkpoints/, latent_dim)
    'vae_20260302_300': ('vae_20260302_300.pth', 64),
    'vae_20260202_065659_400': ('vae_20260202_065659_400.pth', 32),
    # 1K-7K re-fit of the 32-dim ckpt above (experiment=vae_dl3dv_1_7k) -- drop-in, same
    # architecture/keys, so it belongs in the same matrix for a like-for-like comparison
    'vae_dl3dv_1_7k': ('../my_checkpoints/vae_dl3dv_1_7k/last.pth', 32),
    # [new] the two per-scale_mode re-fits (intr_norm raw, train_frac 1.0) that arms A/B train
    # against -- included so the shared-ckpt option can be judged against the best achievable
    # trans recon AT THAT scale_mode, not in the abstract.
    'vae_ctxlonger135': ('../my_checkpoints/vae_ctxlonger135/last.pth', 64),
    'vae_geolagernvs_wt': ('../my_checkpoints/vae_geolagernvs_wt/last.pth', 64),
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
        ds = build_dataset(mode)
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
        print(f"[{mode}] {n_samples} samples (DATASET={DATASET})", flush=True)

    print(f"\nDATASET={DATASET}  META={cfg.meta_csv}  MAX_SCENES={cfg.max_scenes}")
    print(f"{'ckpt':<26} {'dim':>3} {'scale_mode':<19} {'intr':<4} "
          f"{'lat.std':>8} {'/0.96033':>8} {'/0.44677':>8} {'rot':>8} {'trans':>8} {'intr_L1':>8}")
    for c, dim, mode, iv, std, rot, tr, it in rows:
        print(f"{c:<26} {dim:>3} {mode:<19} {iv:<4} {std:>8.5f} {std/0.96032625:>8.4f} "
              f"{std/0.4467666:>8.4f} {rot:>8.5f} {tr:>8.5f} {it:>8.5f}")


if __name__ == '__main__':
    main()
