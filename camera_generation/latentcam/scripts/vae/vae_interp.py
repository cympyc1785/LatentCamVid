"""Latent interpolation between the TWO MOST DIFFERENT trajectories (by camera-center
path shape) among sampled DL3DV segments. Shows endpoints overlay + interpolation morph."""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..', 'main'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', '..'))
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

import config as C
cfg = C.cfg; cfg.geo_encoder = None; cfg.max_scenes = 16
from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE
from utils.data_utils import out_to_trajectory

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = osp.join(osp.dirname(__file__), 'vae_interp.png')


def centers(cp):   # (B,T,11) -> (B,T,3) camera centers in normalized first-frame space
    B = cp.shape[0]
    w2c = out_to_trajectory(cp[:, :, :9], torch.ones(B, 1), torch.eye(4)[None].repeat(B, 1, 1), device='cpu')
    return torch.linalg.inv(w2c)[:, :, :3, 3]


def path_dissim(ca, cb):
    """scale-normalized shape distance between two center paths."""
    a = (ca - ca[0]); b = (cb - cb[0])
    a = a / (a.norm(dim=-1).mean() + 1e-6); b = b / (b.norm(dim=-1).mean() + 1e-6)
    return (a - b).norm(dim=-1).mean().item()


def main():
    ds = CamDataset(cfg)
    # one segment per distinct scene for motion variety
    seen, picks = set(), []
    for i in range(len(ds)):
        sc = ds.samples[i][0]
        if sc not in seen:
            seen.add(sc); picks.append(i)
    cams = torch.stack([ds[i]['cam_param'] for i in picks])
    caps = [ds.samples[i][3] for i in picks]
    cen = centers(cams)

    # most-different pair by path shape
    best, bi, bj = -1, 0, 1
    for a in range(len(picks)):
        for b in range(a + 1, len(picks)):
            d = path_dissim(cen[a], cen[b])
            if d > best:
                best, bi, bj = d, a, b
    print(f"picked most-different pair: #{bi} vs #{bj}  (path dissim={best:.3f})")
    print(f"  A cap: {caps[bi][:90]}")
    print(f"  B cap: {caps[bj][:90]}")

    vae = CameraVAE(latent_dim=cfg.cam_dim).to(dev)
    vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=dev)); vae.eval()
    with torch.no_grad():
        z1 = vae.encode(cams[bi:bi+1].to(dev)); z2 = vae.encode(cams[bj:bj+1].to(dev))
        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        interp = [vae.decode((1 - a) * z1 + a * z2).cpu() for a in alphas]

    fig = plt.figure(figsize=(18, 6))
    # panel 0: endpoints overlay
    ax = fig.add_subplot(2, 4, 1)
    A, B = cen[bi].numpy(), cen[bj].numpy()
    ax.plot(A[:, 0], A[:, 2], '-o', ms=2, c='tab:blue', label=f'A #{bi}')
    ax.plot(B[:, 0], B[:, 2], '-o', ms=2, c='tab:red', label=f'B #{bj}')
    ax.scatter(*A[0, [0, 2]], c='k', s=40, marker='*'); ax.scatter(*B[0, [0, 2]], c='k', s=40, marker='*')
    ax.set_title(f'endpoints (dissim={best:.2f})', fontsize=9); ax.legend(fontsize=7); ax.set_aspect('equal', 'datalim')
    # overlay of all interpolations
    axo = fig.add_subplot(2, 4, 5)
    for a, cp in zip(alphas, interp):
        cc = centers(cp)[0].numpy()
        axo.plot(cc[:, 0], cc[:, 2], '-', lw=1.5, c=plt.cm.plasma(a), label=f'{a}')
    axo.set_title('all interp overlay', fontsize=9); axo.legend(fontsize=6, ncol=2); axo.set_aspect('equal', 'datalim')
    # individual steps
    for c, (a, cp) in enumerate(zip(alphas, interp)):
        ax = fig.add_subplot(2, 4, (2, 3, 4, 6, 7, 8)[c])
        cc = centers(cp)[0].numpy()
        ax.plot(cc[:, 0], cc[:, 2], '-o', ms=2, c=plt.cm.plasma(a))
        ax.scatter(*cc[0, [0, 2]], c='k', s=30, marker='*')
        ax.set_title(f'a={a}', fontsize=9); ax.set_aspect('equal', 'datalim')
    fig.suptitle(f'VAE latent interpolation A#{bi} -> B#{bj} (camera centers, x-z top-down)')
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    print('saved:', OUT)


if __name__ == '__main__':
    main()
