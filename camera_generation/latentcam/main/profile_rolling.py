"""Latency breakdown for rolling scene generation. Times, with CUDA sync:
  - T5 text encode (per segment)
  - rolling denoise loop = model forwards (n_tokens x spt), and a single model-forward cost
  - VAE decode (once, whole token seq)
Also reports #model-forwards and per-forward ms, so we see whether "context 바꿔가며 latent
매번 새로 뽑기"(= re-running the DiT every rolling sub-step, no caching) is the cost.

Run: CUDA_VISIBLE_DEVICES=3 LATENTCAM_CONFIG=config_rolling PYTHONPATH=<latentcam> \
     python profile_rolling.py --ckpt <best.pth> --scene-skip 1
"""
import os, importlib, argparse, time
import numpy as np, torch
from collections import defaultdict

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config_rolling', overrides=[])
from models.camera_diffusion_model_latent import CameraDiffusionModel
from models.t5 import T5EncoderModel
from dataset_dl3dv import CamDataset
import rolling_sampler as RS


def sync(): torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--scene-skip', type=int, default=1)
    ap.add_argument('--spt', type=int, default=8)
    args = ap.parse_args()
    device = 'cuda'; torch.set_grad_enabled(False)
    W, D = cfg.num_cam, cfg.cam_dim

    t0 = time.time()
    text_encoder = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
                                  checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
                                  tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim).to(device)
    sd = torch.load(args.ckpt, map_location=device); model.load_state_dict(sd.get('model', sd)); model.eval()
    vae = RS._load_causal_vae(device)
    print(f"[load] models+vae+t5: {time.time()-t0:.2f}s")

    cfg.max_scenes = args.scene_skip + 1
    ds = CamDataset(cfg, 'train')
    by_scene = defaultdict(list)
    for idx, (sidx, s, e, cap, name) in enumerate(ds.samples):
        by_scene[sidx].append((s, e, idx))
    sc = sorted(by_scene.keys())[args.scene_skip]
    segs = sorted(by_scene[sc], key=lambda x: x[0])
    nseg = len(segs)

    # --- T5 text encode (per segment) ---
    sync(); t = time.time(); seg_embs = []
    for (s, e, idx) in segs:
        d = ds[idx]
        te, tm = text_encoder([d['text_prompt']], device)
        seg_embs.append((te.float(), tm.bool()))
    sync(); t_text = time.time() - t
    print(f"[text] T5 encode x{nseg} segs: {t_text*1000:.1f} ms total, {t_text/nseg*1000:.1f} ms/seg")

    # --- single model forward cost (13-token window, with text cross-attn) ---
    z = torch.randn(1, W, D, device=device); tau = torch.rand(1, W, device=device)
    te, tm = seg_embs[0]
    for _ in range(3): RS._v(model, z, tau, te, tm)     # warmup
    sync(); t = time.time(); NIT = 30
    for _ in range(NIT): RS._v(model, z, tau, te, tm)
    sync(); ms_fwd = (time.time() - t) / NIT * 1000
    print(f"[fwd] single DiT forward (W={W} tokens, +text cross-attn): {ms_fwd:.2f} ms")

    # --- full rolling denoise loop ---
    sync(); t = time.time()
    ztok = RS.gen_rolling_scene(model, seg_embs, [W] * nseg, W, D, device, spt=args.spt)
    sync(); t_roll = time.time() - t
    total_tok = ztok.shape[1]
    # emit loop runs ~ (total_tok + boot) outer steps, each spt forwards
    approx_fwd = total_tok * args.spt
    print(f"[roll] gen_rolling_scene: {t_roll:.2f}s for {total_tok} tokens "
          f"(spt={args.spt}) ~= {approx_fwd} forwards -> {t_roll/approx_fwd*1000:.2f} ms/fwd")

    # --- VAE decode (once, whole token seq) ---
    sync(); t = time.time()
    traj = vae.decode(ztok * cfg.vae_latent_scale)
    sync(); t_dec = time.time() - t
    print(f"[dec] causal VAE decode ({total_tok} tok -> {traj.shape[1]} frames): {t_dec*1000:.1f} ms")

    print(f"\n=== scene total (gen only) ~= text {t_text:.2f}s + roll {t_roll:.2f}s + dec {t_dec:.3f}s "
          f"= {t_text+t_roll+t_dec:.2f}s | rolling loop is {100*t_roll/(t_text+t_roll+t_dec):.0f}% ===")


if __name__ == '__main__':
    main()
