"""Swap-ablation inference for the geo camera-DM models.

For a fixed set of N validation samples, run 3 inference modes and save each:
  a) normal      : each sample's own geo context + own text (original training setting)
  b) ctxswap     : keep the ANCHOR context view (view0 = frame s) per sample, but take the
                   remaining context views from another sample (cyclic i -> (i+1)%N).
  c) textswap    : keep each sample's geo context, but take the text from another sample.

Model / VAE / geo-encoder are built per the run's Hydra experiment config (pass
`experiment=<name>` as a CLI override). Checkpoint + output dir via env:
  SWAP_CKPT=<best.pth>  SWAP_OUT=<dir>  SWAP_N=<num samples>  SWAP_TAG=<model tag>
"""
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config')

from base import Trainer
from models.vae_intr_large import CameraVAE
from models.t5 import T5EncoderModel
from models.camera_diffusion_model_latent import CameraDiffusionModel
from models.geo_encoder import build_geo_encoder
from utils.data_utils import out_to_trajectory, make_intrinsics
from diffusers import DDPMScheduler, DDIMScheduler

DEVICE = 'cuda:0'
CKPT = os.environ['SWAP_CKPT']
OUT = os.environ['SWAP_OUT']
N = int(os.environ.get('SWAP_N', '10'))
TAG = os.environ.get('SWAP_TAG', 'model')

torch.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)


@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, geo_emb, geo_mask):
    B = text_emb.size(0)
    x_t = torch.randn(B, traj_len, cfg.cam_dim, device=text_emb.device)
    scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=text_emb.device)
    for t in scheduler.timesteps:
        ts = torch.full((B,), t, device=text_emb.device)
        noise_pred = model(x_t, ts.float(), text_emb, text_masks, geo_emb, geo_mask)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


def geo_encode_from(geo_encoder, d):
    """geo emb from a (already device) data dict, mirroring train_latent_cam_dm.geo_encode."""
    images = d['images'].to(DEVICE)
    cam_token = None
    if getattr(cfg, 'geo_posed', False) and 'geo_c2w' in d:
        override = (d['avg_scale'].to(DEVICE)
                    if getattr(cfg, 'geo_lagernvs_skip_ctx_norm', False) else None)
        cam_token = geo_encoder.build_cam_token(
            d['geo_c2w'].to(DEVICE), d['geo_fxfycxcy'].to(DEVICE),
            d['geo_hw'].to(DEVICE), override_scale=override)
    return geo_encoder(images, cam_token)


def swap_context_keep_anchor(di, dj):
    """New data dict = di, but geo context views 1: taken from dj (view0/anchor kept from di)."""
    d = copy.copy(di)
    for k in ('images', 'geo_c2w', 'geo_fxfycxcy'):
        if k in di and k in dj:
            vi, vj = di[k], dj[k]                     # (B, V, ...)
            V = min(vi.shape[1], vj.shape[1])
            d[k] = torch.cat([vi[:, :1], vj[:, 1:V]], dim=1)
    # geo_hw: per-view -> swap same way; per-sample -> keep di's
    if 'geo_hw' in di and 'geo_hw' in dj and di['geo_hw'].dim() >= 2 and di['geo_hw'].shape[1] > 1:
        vi, vj = di['geo_hw'], dj['geo_hw']
        V = min(vi.shape[1], vj.shape[1])
        d['geo_hw'] = torch.cat([vi[:, :1], vj[:, 1:V]], dim=1)
    return d


def main():
    traj_len = (cfg.num_frames - 1) // 4 + 1
    cfg.batch_size = 1
    cfg.num_gpus = 1

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator(include_train=False, include_val=True)

    geo_encoder = build_geo_encoder(cfg).to(DEVICE) if getattr(cfg, 'geo_encoder', None) else None
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len, dtype=cfg.t5_dtype, device=DEVICE,
        checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
        tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)

    model = CameraDiffusionModel(cam_dim=cfg.cam_dim)
    model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    model = model.to(DEVICE).eval()

    camera_vae = CameraVAE(latent_dim=cfg.cam_dim).to(DEVICE)
    camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=DEVICE))
    camera_vae.eval()

    sched_cls = DDPMScheduler if cfg.sampling_type == 'ddpm' else DDIMScheduler
    noise_scheduler = sched_cls(
        num_train_timesteps=cfg.diffusion_max_step, prediction_type=cfg.prediction_type,
        beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
        set_alpha_to_one=cfg.set_alpha_to_one, steps_offset=cfg.steps_offset,
        beta_start=cfg.beta_start, beta_end=cfg.beta_end)

    # collect N fixed val samples (deterministic split, shuffle=False)
    samples = []
    for step, data in enumerate(trainer.valid_batch_generator):
        samples.append(data)
        if len(samples) >= N:
            break
    print(f"[{TAG}] collected {len(samples)} val samples")

    modes = ['a_normal', 'b_ctxswap', 'c_textswap']
    torch.manual_seed(cfg.random_seed)   # fix sampling noise across modes for comparability
    for mode in modes:
        mdir = os.path.join(OUT, TAG, mode)
        os.makedirs(mdir, exist_ok=True)
        summary = []
        for i, di in enumerate(samples):
            j = (i + 1) % len(samples)
            dj = samples[j]
            name_i = os.path.basename(di['data_name'][0])
            name_j = os.path.basename(dj['data_name'][0])

            traj = di['cam_param'].to(DEVICE)
            E0 = di['first_extrinsic'].to(DEVICE)
            scale = di['avg_scale'].to(DEVICE)
            width = di['width'].to(DEVICE)
            height = di['height'].to(DEVICE)
            intrinsics = di['intrinsics'].to(DEVICE)

            # ---- build geo + text per mode ----
            if mode == 'b_ctxswap':
                geo_src = swap_context_keep_anchor(di, dj)
            else:
                geo_src = di
            text = dj['text_prompt'] if mode == 'c_textswap' else di['text_prompt']

            geo_emb, geo_mask = (geo_encode_from(geo_encoder, geo_src)
                                 if geo_encoder is not None else (None, None))
            text_emb, text_mask = text_encoder(text, DEVICE)
            text_emb = text_emb.float(); text_mask = text_mask.bool()

            # fixed noise per sample index (same across modes) for fair comparison
            g = torch.Generator(device=DEVICE).manual_seed(cfg.random_seed + i)
            x_t = torch.randn(1, traj_len, cfg.cam_dim, device=DEVICE, generator=g)
            noise_scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=DEVICE)
            with torch.no_grad():
                for t in noise_scheduler.timesteps:
                    ts = torch.full((1,), t, device=DEVICE)
                    noise_pred = model(x_t, ts.float(), text_emb, text_mask, geo_emb, geo_mask)
                    x_t = noise_scheduler.step(noise_pred, t, x_t).prev_sample
                out = x_t
                traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                loss_latent = F.mse_loss(out, traj_latents).item()
                traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)
                loss_traj = F.mse_loss(traj_pred, traj).item()

            pred_intr = make_intrinsics(traj_pred[:, :, -2:], width, height, intrinsics)
            gt_intr = make_intrinsics(traj[:, :, -2:], width, height, intrinsics)
            pred_traj = out_to_trajectory(traj_pred, scale, E0, DEVICE)
            gt_traj = out_to_trajectory(traj, scale, E0, DEVICE)

            np.savez(os.path.join(mdir, f'{name_i}_pred.npz'),
                     poses=pred_traj[0].cpu().numpy(), intrinsics=pred_intr[0].cpu().numpy())
            np.savez(os.path.join(mdir, f'{name_i}_gt.npz'),
                     poses=gt_traj[0].cpu().numpy(), intrinsics=gt_intr[0].cpu().numpy())
            summary.append({'idx': i, 'name': name_i, 'paired_with': name_j,
                            'loss_latent': loss_latent, 'loss_traj': loss_traj,
                            'text_used': text[0]})
            print(f"[{TAG}/{mode}] {i}: {name_i} (pair {name_j}) loss_traj={loss_traj:.4f}")
        import json
        json.dump(summary, open(os.path.join(mdir, 'summary.json'), 'w'), indent=2)
    print(f"[{TAG}] DONE -> {os.path.join(OUT, TAG)}")


if __name__ == '__main__':
    main()
