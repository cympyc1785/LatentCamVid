"""3-mode inference for the per-token (Diffusion-Forcing) rectified-flow camera model, in the
CAUSAL-VAE latent space (W=13 tokens, D=64). One trained model -> full_sequence / chunk_ar /
rolling by tau schedule only. Latent is denoised, then decoded ONCE -> 49 frames.

Flow: v=model(z, tau*1000, text); z_tau = z0 + tau*v  =>  Euler denoise: z -= v*dt as tau-=dt.

Also: jerk periodogram (period-4 = latent-token boundary, period-12 = 3-token chunk boundary),
per-frame velocity Wasserstein-1 vs a reference, and VAE round-trip jerk baseline (noise floor).

CLI: python rolling_sampler.py --ckpt best.pth --mode all --text "..." --seed 0 --out /tmp/roll
"""
import os, argparse, importlib
import numpy as np, torch

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config_rolling', overrides=[])
from models.camera_diffusion_model_latent import CameraDiffusionModel
from models.t5 import T5EncoderModel
from utils.data_utils import out_to_trajectory

W_DEFAULT = 13


def _load_causal_vae(device):
    from types import SimpleNamespace
    from models.cam_causal_vae import CameraVAE
    vae = CameraVAE(SimpleNamespace(in_dim=11, out_dim=11, hidden_size=64,
                                    latent_dim=cfg.cam_dim, scale_factor=cfg.vae_latent_scale)).to(device)
    vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device)); vae.eval()
    return vae


@torch.no_grad()
def _v(model, z, tau, temb, tmask):
    """model velocity at per-token tau (tau: (1,W))."""
    return model(z, tau * 1000.0, temb, tmask, None, None)


@torch.no_grad()
def gen_full(model, temb, tmask, W, D, N, device):
    z = torch.randn(1, W, D, device=device)
    taus = torch.linspace(1, 0, N + 1, device=device)
    for i in range(N):
        cur = taus[i].expand(1, W)
        z = z - _v(model, z, cur, temb, tmask) * (taus[i] - taus[i + 1])
    return z


@torch.no_grad()
def gen_chunk_ar(model, temb, tmask, W, D, N, device, chunks=(1, 3, 3, 3, 3)):
    z = torch.randn(1, W, D, device=device)
    tau = torch.ones(1, W, device=device)
    start = 0
    for cs in chunks:
        e = start + cs
        taus = torch.linspace(1, 0, N + 1, device=device)
        for i in range(N):
            tau[:, :start] = 0.0                      # history fixed (clean, tau=0)
            tau[:, start:e] = taus[i]                 # current chunk denoise
            v = _v(model, z, tau, temb, tmask)
            z[:, start:e] = z[:, start:e] - v[:, start:e] * (taus[i] - taus[i + 1])
        tau[:, start:e] = 0.0
        start = e
    return z


@torch.no_grad()
def gen_rolling(model, temb, tmask, W, D, n_tokens, device, spt=8):
    """Steady-ramp emit/reinject. ds = 1/(W-1); each outer step advances all tokens by ds
    (via spt sub-steps), emits front token when tau<=0, injects tau=1 noise at the back.
    Boot-up: start from a full ramp on an all-noise window (front tokens converge first)."""
    z = torch.randn(1, W, D, device=device)
    tau = torch.linspace(1.0, 1.0 / (W), W, device=device).flip(0).unsqueeze(0)  # front low, back high
    ds = 1.0 / (W - 1)
    out = []
    guard = 0
    while len(out) < n_tokens and guard < n_tokens + W + 5:
        guard += 1
        sub = ds / spt
        for _ in range(spt):                          # denoise all tokens by ds total
            step = torch.clamp(tau, max=sub)
            z = z - _v(model, z, tau, temb, tmask) * step.unsqueeze(-1)
            tau = (tau - sub).clamp(min=0)
        if tau[0, 0].item() <= 1e-4:                  # front clean -> emit, shift, inject
            out.append(z[0, 0].clone())
            z = torch.cat([z[:, 1:], torch.randn(1, 1, D, device=device)], dim=1)
            tau = torch.cat([tau[:, 1:], torch.ones(1, 1, device=device)], dim=1)
    zc = torch.stack(out[:n_tokens]).unsqueeze(0) if out else z
    return zc


@torch.no_grad()
def gen_rolling_scene(model, seg_embs, tokens_per_seg, W, D, device, spt=8):
    """CONTINUOUS rolling over a WHOLE scene: one sliding window (W tokens), emit/reinject
    token-by-token; the text condition switches to the segment of the token currently at the
    FRONT (emit position). Returns the full emitted latent (1, sum(tokens_per_seg), D) -> decode
    ONCE for a seam-free trajectory. seg_embs = [(temb,tmask), ...] per segment."""
    total = int(sum(tokens_per_seg))
    seg_of_tok = [si for si, n in enumerate(tokens_per_seg) for _ in range(n)]
    nseg = len(seg_embs)
    z = torch.randn(1, W, D, device=device)
    tau = torch.linspace(1.0, 1.0 / W, W, device=device).flip(0).unsqueeze(0)  # front low, back high
    ds = 1.0 / (W - 1)
    out, guard = [], 0
    while len(out) < total and guard < total + W + 5:
        guard += 1
        temb, tmask = seg_embs[min(seg_of_tok[len(out)], nseg - 1)]   # front token's segment text
        sub = ds / spt
        for _ in range(spt):
            step = torch.clamp(tau, max=sub)
            z = z - _v(model, z, tau, temb, tmask) * step.unsqueeze(-1)
            tau = (tau - sub).clamp(min=0)
        if tau[0, 0].item() <= 1e-4:                  # front clean -> emit, shift, inject
            out.append(z[0, 0].clone())
            z = torch.cat([z[:, 1:], torch.randn(1, 1, D, device=device)], dim=1)
            tau = torch.cat([tau[:, 1:], torch.ones(1, 1, device=device)], dim=1)
    return torch.stack(out[:total]).unsqueeze(0) if out else z


def latent_to_traj(vae, z, device):
    """z (1,W,D) latent -> (T,11) cam_param -> (T,4,4) w2c (relative: scale=1, E0=I)."""
    cam = vae.decode(z * cfg.vae_latent_scale)                  # (1,T,11)
    E0 = torch.eye(4, device=device).unsqueeze(0)
    w2c = out_to_trajectory(cam, torch.ones(1, 1, device=device), E0, device)[0]
    return cam[0].cpu().numpy(), w2c.cpu().numpy()


def jerk_spectrum(traj_w2c):
    """translation 3rd-difference norm -> periodogram; return power at period 4 & 12."""
    c2w = np.linalg.inv(traj_w2c); pos = c2w[:, :3, 3]
    jerk = np.diff(pos, n=3, axis=0)
    j = np.linalg.norm(jerk, axis=1)
    if len(j) < 8:
        return {}
    f = np.abs(np.fft.rfft(j - j.mean())) ** 2
    freqs = np.fft.rfftfreq(len(j))
    def power_at(period):
        fi = np.argmin(np.abs(freqs - 1.0 / period))
        lo, hi = max(1, fi - 3), min(len(f), fi + 4)
        med = np.median(np.concatenate([f[1:lo], f[hi:]])) + 1e-9
        return float(f[fi] / med)
    return {'jerk_mean': float(j.mean()), 'peak_p4': power_at(4), 'peak_p12': power_at(12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mode', default='all', choices=['full', 'chunk_ar', 'rolling', 'all'])
    ap.add_argument('--text', default='the camera moves forward')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--n-tokens', type=int, default=W_DEFAULT)
    ap.add_argument('--out', default='/tmp/roll')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda'; torch.manual_seed(args.seed)
    W, D = args.n_tokens, cfg.cam_dim

    model = CameraDiffusionModel(cam_dim=cfg.cam_dim).to(device)
    sd = torch.load(args.ckpt, map_location=device); model.load_state_dict(sd.get('model', sd)); model.eval()
    vae = _load_causal_vae(device)
    te = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
                        checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
                        tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)
    temb, tmask = te([args.text], device); temb = temb.float(); tmask = tmask.bool()

    modes = ['full', 'chunk_ar', 'rolling'] if args.mode == 'all' else [args.mode]
    fns = {'full': lambda: gen_full(model, temb, tmask, W, D, args.steps, device),
           'chunk_ar': lambda: gen_chunk_ar(model, temb, tmask, W, D, max(4, args.steps // W), device),
           'rolling': lambda: gen_rolling(model, temb, tmask, W, D, args.n_tokens, device)}
    for m in modes:
        torch.manual_seed(args.seed)
        z = fns[m]()
        cam, w2c = latent_to_traj(vae, z, device)
        np.save(os.path.join(args.out, f"{m}_pose9.npy"), cam[:, :9])   # (T,9) rot6d+trans
        np.save(os.path.join(args.out, f"{m}_w2c.npy"), w2c)            # (T,4,4)
        print(f"[{m}] latent {tuple(z.shape)} -> traj {cam.shape} | {jerk_spectrum(w2c)}")
    print(f"out: {args.out}")


if __name__ == '__main__':
    main()
