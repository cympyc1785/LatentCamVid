import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import wandb
import datetime
import time
import json
import random
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object, tqdm, broadcast_object_list
from torch_scatter import scatter_mean
import torch.nn as nn

import importlib as _importlib
from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config')
from base import Trainer  # ported to main/base.py (DL3DV dataset)
from models.vae_intr_large import CameraVAE
from models.t5 import T5EncoderModel
from utils.data_utils import out_to_trajectory, make_intrinsics, inverse_camera_matrix
from utils.eval_utils import run_command_in_dir
from utils.pc_utils import get_ray_sim_per_point
from diffusers import DDPMScheduler, DDIMScheduler

# CLaTr feature extraction (OpenAI clip) is optional — only needed for CLaTr eval.
try:
    from evaluate.CLaTr.clip_extraction import load_clip_model, encode_text, save_feats_custom
    _HAS_CLIP = True
except Exception as _e:
    print(f"(train) clip_extraction unavailable ({_e}); CLaTr feature saving disabled")
    _HAS_CLIP = False

torch.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)
random.seed(cfg.random_seed)

# Camera diffusion denoiser: use the latent (geo-conditioned) variant.
from models.camera_diffusion_model_latent import CameraDiffusionModel

# Geo encoder (image-based scene encoder feeding the latent model's geo latent).
from models.geo_encoder import build_geo_encoder

# Legacy point-cloud encoder — only imported when the pc path is used (load_points).
# The geo path (lagernvs) does not need concerto, so we skip the import otherwise.
PCEncoder = None
if cfg.load_points:
    if cfg.point_encoder == 'concerto':
        from models.pc_encoder import PCEncoder
    elif cfg.point_encoder == 'custom':
        from models.pc_encoder_custom import PCEncoder
    elif cfg.point_encoder == 'mosaic':
        from models.pc_encoder_mosaic import PCEncoder

def geo_emb_from_cache(data, device):
    """Precomputed geo path: the dataset already read the frozen geo_emb off disk, so there is no
    encoder forward. Mirrors GeoEncoder.forward's contract -> (B, M, 768), (B, M) bool all-ones
    (the cache stores no mask because lagernvs marks every token valid, geo_encoder.py:139)."""
    emb = data['geo_emb'].to(device)
    mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=device)
    return emb, mask


def geo_encode(geo_encoder, data, device):
    """Run the geo encoder on a batch. When cfg.geo_posed and the batch carries geo-view
    geometry, feed the posed lagernvs cam_token; otherwise unposed (cam_token=None)."""
    images = data['images'].to(device)
    cam_token = None
    if getattr(cfg, 'geo_posed', False) and 'geo_c2w' in data:
        # skip LagerNVS's own 1.35*ctx_max normalization -> reuse the target's initial
        # 1.35*max_dist (avg_scale) so context & target share one frame+scale.
        override = (data['avg_scale'].to(device)
                    if getattr(cfg, 'geo_lagernvs_skip_ctx_norm', False) else None)
        cam_token = geo_encoder.build_cam_token(
            data['geo_c2w'].to(device), data['geo_fxfycxcy'].to(device),
            data['geo_hw'].to(device), override_scale=override)
    return geo_encoder(images, cam_token)


def attach_geo_cam(emb, data, device):
    """[new] cfg.geo_cam_embed: append a raw camera embedding to every geo patch token.

      'relfirst' -- (B, V, 11) each context view's pose relative to the target's FIRST camera
                    (dataset_dl3dv._geo_cam_param), broadcast over that view's P patches.
      'plucker'  -- (B, V, P, 6) a Plücker ray PER patch token in the same frame/units
                    (dataset_dl3dv._geo_cam_plucker); no broadcast, one value per token.

      emb (B, M, C) with M = V*P  ->  (B, M, C + 11) or (B, M, C + 6)

    The raw dims ride along inside geo_emb so no call site's signature changes; the model splits
    them off and lifts them with a trainable MLP (CameraDiffusionModel._lift_geo_cam). No-op when
    geo_cam_embed is off or the batch carries no geo_cam_param."""
    if not getattr(cfg, 'geo_cam_embed', None) or 'geo_cam_param' not in data:
        return emb
    cam = data['geo_cam_param'].to(device).to(emb.dtype)   # (B,V,11) | (B,V,P,6)
    B, M, _ = emb.shape
    V = cam.shape[1]
    if M % V:
        raise ValueError(f"geo token count {M} not divisible by #views {V}")
    P = M // V
    if cam.dim() == 4:            # plucker: per-patch, must line up with the token grid
        if cam.shape[2] != P:
            raise ValueError(f"plucker grid {cam.shape[2]} != tokens per view {P} "
                             f"(M={M}, V={V}); check geo_image_hw vs _geo_patch_grid()")
        cam = cam.reshape(B, M, cam.shape[-1])
    else:                         # relfirst: per-view, broadcast over the view's patches
        cam = cam.unsqueeze(2).expand(B, V, P, cam.shape[-1]).reshape(B, M, cam.shape[-1])
    return torch.cat([emb, cam], dim=-1)


@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, point_emb, point_mask):
    B = text_emb.size(0)
    device = text_emb.device
    x_t = torch.randn(B, traj_len, cfg.cam_dim).to(device)

    scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)
    for t in scheduler.timesteps:
        timesteps = torch.full(
            (B,),
            t,
            device=device,
        )
        noise_pred = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


from tau_sampler import sample_tau


def per_token_flow_loss(model, z, text_emb, text_mask, geo_emb, geo_mask, return_stats=False):
    """Diffusion-Forcing rectified-flow loss on the causal-VAE latent z (B, W, D).
      eps~N(0,1); z_noisy=(1-tau)z+tau*eps; v_target=eps-z; model predicts v (per-token tau).
      tokens with tau < cfg.loss_tau_min are context -> excluded from loss.
    Batch-level 10% text CFG dropout (null embedding)."""
    B, W, _ = z.shape
    device = z.device
    eps = torch.randn_like(z)
    tau, labels = sample_tau(B, W, device)                       # (B,W), (B,)
    tau_e = tau.unsqueeze(-1)
    z_noisy = (1 - tau_e) * z + tau_e * eps
    v_target = eps - z
    if torch.rand(1).item() < getattr(cfg, 'cfg_dropout_p', 0.1):  # batch-level CFG dropout
        text_emb = torch.zeros_like(text_emb)
        text_mask = torch.ones_like(text_mask)
    v_pred = model(z_noisy, tau * 1000.0, text_emb, text_mask, geo_emb, geo_mask)  # per-token time
    mask = (tau >= getattr(cfg, 'loss_tau_min', 0.02)).float()   # (B,W)
    per_tok = ((v_pred - v_target) ** 2).mean(-1)                 # (B,W)
    denom = mask.sum().clamp(min=1.0)                            # zero-denom guard
    loss = (mask * per_tok).sum() / denom
    if return_stats:
        return loss, tau.detach(), labels.detach(), per_tok.detach()
    return loss


def _ar_chunk_bounds(T, chunk_size):
    """[(s,e), ...] covering [0,T) in chunks of chunk_size (last may be smaller)."""
    bounds, s = [], 0
    while s < T:
        bounds.append((s, min(s + chunk_size, T)))
        s += chunk_size
    return bounds


def ar_train_loss(raw_model, scheduler, traj_latents, text_emb, text_mask, geo_emb, geo_mask, chunk_size):
    """Chunk-wise AR DDPM loss: for each chunk, past chunks are CLEAN (teacher-forcing),
    current chunk gets fresh DDPM noise at a per-chunk timestep; causal self-attn conditions
    the current chunk on the clean past + text + geo. Loss = mean MSE(pred_noise, noise)."""
    B, T, _ = traj_latents.shape
    device = traj_latents.device
    total, n = 0.0, 0
    for s, e in _ar_chunk_bounds(T, chunk_size):
        cur = traj_latents[:, s:e]
        history = traj_latents[:, :s] if s > 0 else None
        noise = torch.randn_like(cur)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy = scheduler.add_noise(cur, noise, t)
        pred = raw_model.forward_ar(noisy, t.float(), text_emb, text_mask, geo_emb, geo_mask, history=history)
        total = total + F.mse_loss(pred, noise)
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def ar_sample(raw_model, scheduler, traj_len, text_emb, text_masks, geo_emb, geo_mask, chunk_size, cam_dim):
    """Autoregressive generation: denoise chunk 0 from noise, then each next chunk
    conditioned on the previously generated (clean) chunks via causal self-attn."""
    B, device = text_emb.size(0), text_emb.device
    scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)
    generated = None
    for s, e in _ar_chunk_bounds(traj_len, chunk_size):
        x = torch.randn(B, e - s, cam_dim, device=device)
        for tt in scheduler.timesteps:
            t = torch.full((B,), tt, device=device).float()
            noise_pred = raw_model.forward_ar(x, t, text_emb, text_masks, geo_emb, geo_mask, history=generated)
            x = scheduler.step(noise_pred, tt, x).prev_sample
        generated = x if generated is None else torch.cat([generated, x], dim=1)
    return generated

def train():
    T = cfg.diffusion_max_step 
    traj_len = cfg.num_cam
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # ---- resume: peek checkpoint. full = {model,opt,global_step,epoch,best_val,...};
    #      a pure state_dict (old last.pth) -> weights-only warm-start. ----
    _resume = None
    if getattr(cfg, 'load_ckpt_path', None):
        _resume = torch.load(cfg.load_ckpt_path, map_location='cpu')
        if not (isinstance(_resume, dict) and 'model' in _resume):
            _resume = {'model': _resume, '_weights_only': True}
    _full_resume = _resume is not None and not _resume.get('_weights_only', False)

    if accelerator.is_main_process:
        if _full_resume and _resume.get('exp_name'):
            exp_name = _resume['exp_name']                 # reuse result dir + wandb run
        else:
            exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f'_{cfg.exp_name}'
    else:
        exp_name = None
    exp_name = broadcast_object_list([exp_name])[0]
    result_dir = os.path.abspath(os.path.join('../results', exp_name))
    os.makedirs(result_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_data_dir = os.path.join(result_dir, 'test')
    os.makedirs(eval_data_dir, exist_ok=True)

    _wandb_init = {"name": f"{exp_name}"}
    if _full_resume and _resume.get('wandb_id'):           # continue the same wandb run
        _wandb_init = {"id": _resume['wandb_id'], "resume": "allow", "name": f"{exp_name}"}
    accelerator.init_trackers(
        project_name="camera-diffusion-training",
        config=cfg_dict,
        init_kwargs={"wandb": _wandb_init},
    )
    try:
        _wandb_id = accelerator.get_tracker("wandb", unwrap=True).id
    except Exception:
        _wandb_id = None

    if accelerator.is_main_process:            # full resolved config as YAML (run + wandb dir)
        from hydra_cfg import save_cfg_yaml
        save_cfg_yaml(cfg_dict, os.path.join(result_dir, 'config.yaml'))
        try:
            save_cfg_yaml(cfg_dict, os.path.join(wandb.run.dir, 'config.yaml'))
        except Exception as _e:
            print(f"(train) wandb config.yaml save skipped: {_e}")

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    # NOTE: trainer._make_model() removed — it allocated an unused DataParallel model
    # on GPU (get_model() defaults to cam_dim=11); training builds its own `model` below.

    if cfg.sampling_type == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=T,
            prediction_type=cfg.prediction_type,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            set_alpha_to_one=cfg.set_alpha_to_one,
            steps_offset=cfg.steps_offset,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )
    elif cfg.sampling_type == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=T,
            prediction_type=cfg.prediction_type,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            set_alpha_to_one=cfg.set_alpha_to_one,
            steps_offset=cfg.steps_offset,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )    

    pc_encoder = PCEncoder().to(device) if PCEncoder is not None else None
    # Geo encoder (image-based, frozen, on-the-fly). Feeds the latent model's geo latent.
    # TODO(C-4): switch the training loop below to compute geo_emb/geo_mask via
    #   geo_emb, geo_mask = geo_encoder(images, cam_token)
    # (needs the dataset to provide multi-view images) instead of pc_embeds/pc_masks.
    geo_encoder = build_geo_encoder(cfg).to(device) if getattr(cfg, 'geo_encoder', None) else None
    if cfg.text_encoder == 'T5':
        text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=device,
            checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
            tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path),
            shard_fn=None)
    elif cfg.text_encoder == 'CLIP':
        from core_pkg.common.utils.clip_utils import get_model, encode_text_clip
        text_encoder, tokenizer = get_model(cfg.clip_repo_name, device=device)
    
    # [new] cfg.geo_cam_embed widens the geo cross-attention input: geo_proj sees
    # geo_latent_dim + geo_cam_embed_dim instead of geo_latent_dim. Off (0) -> identical model.
    _geo_kw = {}
    if getattr(cfg, 'geo_cam_embed', None):
        # relfirst = 11 raw dims per view (rot6d + trans + fx/2cx, fy/2cy);
        # plucker  = 6 raw dims per patch token (direction + moment).
        _raw = 6 if cfg.geo_cam_embed == 'plucker' else 11
        _geo_kw = dict(geo_latent_dim=getattr(cfg, 'geo_latent_dim', 768),
                       geo_cam_raw_dim=_raw,
                       geo_cam_embed_dim=getattr(cfg, 'geo_cam_embed_dim', 128))
        print(f"(model) geo_cam_embed={cfg.geo_cam_embed}: geo_proj input = "
              f"{_geo_kw['geo_latent_dim']} + {_geo_kw['geo_cam_embed_dim']}")
    if cfg.point_encoder != 'custom':
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim, **_geo_kw)
    else:
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim, pc_encoder=pc_encoder, **_geo_kw)
    
    # Load weights from the (peeked) resume checkpoint. Optimizer/step/epoch are
    # restored after accelerator.prepare() below (full resume only).
    if _resume is not None:
        _mm, _uu = model.load_state_dict(_resume['model'], strict=False)
        _kind = "full-resume" if _full_resume else "weights warm-start"
        print(f"(resume) {_kind} model from {cfg.load_ckpt_path} "
              f"(missing={len(_mm)}, unexpected={len(_uu)})")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if cfg.use_vae:
        if getattr(cfg, 'causal_vae', False):
            # AR: causal VAE (past-only conv) — reuse CamVLA causal_vae_v1 ckpt (49->13, 11-dim)
            from types import SimpleNamespace
            from models.cam_causal_vae import CameraVAE as CausalCameraVAE
            camera_vae = CausalCameraVAE(SimpleNamespace(
                in_dim=11, out_dim=11, hidden_size=64, latent_dim=cfg.cam_dim,
                scale_factor=cfg.vae_latent_scale)).to(device)
        else:
            camera_vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
        camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
        camera_vae.eval()

    # For evaluation (CLaTr) — optional
    clip_model = load_clip_model(cfg.clip_version, device=device) if _HAS_CLIP else None

    global_step = 0
    start_epoch = 0

    model, opt, train_dataloader, valid_dataloader = accelerator.prepare(
        model, opt, trainer.batch_generator, trainer.valid_batch_generator
    )

    best_val = float('inf')   # best val/loss_traj so far (for best.pth)
    if _full_resume:          # restore optimizer + counters for seamless continuation
        opt.load_state_dict(_resume['opt'])
        _real_opt = getattr(opt, 'optimizer', opt)
        for _st in _real_opt.state.values():              # move AdamW moments to device
            for _k, _v in _st.items():
                if torch.is_tensor(_v):
                    _st[_k] = _v.to(device)
        global_step = int(_resume.get('global_step', 0))
        start_epoch = int(_resume.get('epoch', -1)) + 1
        best_val = float(_resume.get('best_val', float('inf')))
        print(f"(resume) continuing at epoch {start_epoch}, global_step {global_step}, "
              f"best_val {best_val:.6f}")

    def run_validation(epoch, global_step):
        """Per-epoch validation: sample trajectories, log val losses (+ optional
        CLaTr metrics) to wandb, and keep only last.pth + best.pth (by val/loss_traj)."""
        nonlocal best_val
        model.eval()
        with torch.no_grad():
            pbar = tqdm(valid_dataloader)
            total_loss_latent = torch.tensor(0.0, device=accelerator.device)
            total_loss_traj = torch.tensor(0.0, device=accelerator.device)
            total_samples = torch.tensor(0, device=accelerator.device)
            data_name_list = []
            _pt = getattr(cfg, 'per_token_noise', False)   # Diffusion-Forcing val monitoring
            tau_bin_sum = torch.zeros(10, device=accelerator.device)
            tau_bin_cnt = torch.zeros(10, device=accelerator.device)
            pat_sum = torch.zeros(4, device=accelerator.device)
            pat_cnt = torch.zeros(4, device=accelerator.device)
            for step, data in enumerate(pbar):
                if step >= cfg.val_max_batches:   # cap validation cost
                    break
                data_name = data['data_name']
                text_prompt = data['text_prompt']
                traj = data['cam_param'].to(device)
                E0 = data['first_extrinsic'].to(device)
                scale = data['avg_scale'].to(device)
                width = data['width'].to(device)
                height = data['height'].to(device)
                intrinsics = data['intrinsics'].to(device)

                B = traj.shape[0]

                if cfg.load_points:
                    if 'pc' in data:
                        point = data['pc']
                        if cfg.point_encoder != 'custom':
                            pc_embeds, pc_masks, point_obj = pc_encoder(point)
                        else:
                            pc_embeds, pc_masks, point_obj = model.pc_encoder(point)
                    elif 'pc_embeds' in data:
                        pc_embeds = data['pc_embeds'].to(device)
                        pc_masks = data['pc_masks'].to(device)

                else:
                    pc_embeds, pc_masks = None, None
                # Geo latent conditioning (image-based, frozen) — mirrors the train loop.
                if 'geo_emb' in data:
                    pc_embeds, pc_masks = geo_emb_from_cache(data, device)
                elif geo_encoder is not None and 'images' in data:
                    pc_embeds, pc_masks = geo_encode(geo_encoder, data, device)
                if pc_embeds is not None:
                    pc_embeds = attach_geo_cam(pc_embeds, data, device)
                if cfg.text_encoder == 'T5':
                    text_embeds, text_masks = text_encoder(text_prompt, device)
                elif cfg.text_encoder == 'CLIP':
                    text_embeds, text_masks = encode_text_clip(text_encoder, tokenizer, text_prompt, device=device)
                text_embeds = text_embeds.float()
                text_masks = text_masks.bool()
                if cfg.use_vae:
                    traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                else:
                    traj_latents = traj

                if _pt:   # per-token flow: masked val loss + tau-bin/pattern monitoring
                    vloss, tau, labels, per_tok = per_token_flow_loss(
                        model, traj_latents, text_embeds, text_masks, pc_embeds, pc_masks, return_stats=True)
                    total_loss_traj += vloss * B
                    total_samples += B
                    bins = (tau * 10).long().clamp(0, 9)
                    for bi in range(10):
                        m = bins == bi
                        if m.any():
                            tau_bin_sum[bi] += per_tok[m].sum(); tau_bin_cnt[bi] += m.sum()
                    for p in range(4):
                        m = labels == p
                        if m.any():
                            pat_sum[p] += per_tok[m].mean(); pat_cnt[p] += 1
                    continue

                if getattr(cfg, 'is_ar', False):
                    out = ar_sample(accelerator.unwrap_model(model), noise_scheduler, traj_len,
                                    text_embeds, text_masks, pc_embeds, pc_masks, cfg.ar_chunk_size, cfg.cam_dim)
                else:
                    out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks)

                val_loss_latent = F.mse_loss(out, traj_latents, reduction='mean')
                total_loss_latent += val_loss_latent * B

                if cfg.use_vae:
                    traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)
                else:
                    traj_pred = out
                val_loss_traj = F.mse_loss(traj_pred, traj, reduction='mean')
                total_loss_traj += val_loss_traj * B
                total_samples += B

                ref_intrinsics = make_intrinsics(traj[:, :, -2:], width, height, intrinsics).cpu().tolist()
                pred_intrinsics = make_intrinsics(traj_pred[:, :, -2:], width, height, intrinsics).cpu().tolist()
                width = width.cpu().tolist()
                height = height.tolist()

                traj = out_to_trajectory(traj, scale, E0, device)
                traj_pred = out_to_trajectory(traj_pred, scale, E0, device)
                matrix_traj_ref = inverse_camera_matrix(traj)
                matrix_traj_pred = inverse_camera_matrix(traj_pred)
                matrix_traj_ref[:, :, :3, 1:3] *= -1
                matrix_traj_pred[:, :, :3, 1:3] *= -1
                matrix_traj_ref = matrix_traj_ref.cpu().tolist()
                matrix_traj_pred = matrix_traj_pred.cpu().tolist()

                if _HAS_CLIP and clip_model is not None:
                    seq_embeds, tok_embeds = encode_text(text_prompt, clip_model, max_token_length=None, device=device)
                    save_feats_custom(seq_embeds, data_name, Path(os.path.join(result_dir, 'seq')))
                    save_feats_custom(tok_embeds, data_name, Path(os.path.join(result_dir, 'token')))
                for i in range(len(data_name)):
                    text_path = os.path.join(eval_data_dir, f"{data_name[i]}_caption.json")
                    ref_path = os.path.join(eval_data_dir, f"{data_name[i]}_transforms_ref.json")
                    pred_path = os.path.join(eval_data_dir, f"{data_name[i]}_transforms_pred.json")
                    text_json = {
                        "Concise Interaction": text_prompt[i]
                    }
                    ref_json = {
                        "w": width[i][0],
                        "h": height[i][0],
                        "fl_x": ref_intrinsics[i][0][0][0],
                        "fl_y": ref_intrinsics[i][0][1][1],
                        "cx": ref_intrinsics[i][0][0][2],
                        "cy": ref_intrinsics[i][0][1][2],
                        "frames": [{
                            "transform_matrix": matrix_traj_ref[i][frame_idx],
                            "monst3r_im_id": frame_idx + 1}
                            for frame_idx in range(len(matrix_traj_ref[i]))]
                    }
                    pred_json = {
                        "w": width[i][0],
                        "h": height[i][0],
                        "fl_x": pred_intrinsics[i][0][0][0],
                        "fl_y": pred_intrinsics[i][0][1][1],
                        "cx": pred_intrinsics[i][0][0][2],
                        "cy": pred_intrinsics[i][0][1][2],
                        "frames": [{
                            "transform_matrix": matrix_traj_pred[i][frame_idx],
                            "monst3r_im_id": frame_idx + 1}
                            for frame_idx in range(len(matrix_traj_pred[i]))]
                    }
                    with open(text_path, 'w') as f:
                        json.dump(text_json, f, indent=4)
                    with open(ref_path, 'w') as f:
                        json.dump(ref_json, f, indent=4)
                    with open(pred_path, 'w') as f:
                        json.dump(pred_json, f, indent=4)
                data_name_list += data_name
            all_data_name_list = sorted(set(gather_object(data_name_list)))
            if accelerator.is_main_process:
                valid_txt_path = os.path.join(result_dir, 'test_valid.txt')
                with open(valid_txt_path, "w", encoding="utf-8") as f:
                    for name in all_data_name_list:
                        f.write(name + "\n")

                # CLaTr metric eval (subprocess) needs clip + hydra + the eval
                # subpackages; skipped when clip_extraction is unavailable or per-token
                # (per-token val saves no transforms; uses flow val loss + tau/pattern bins).
                if _HAS_CLIP and not _pt:
                    out, err, ret = run_command_in_dir([sys.executable, '-m', 'src.extraction', f'checkpoint_path={cfg.clatr_ckpt_path}', f'data_dir={result_dir}'], \
                                                    'evaluate/CLaTr')
                    print(out)
                    print(err)
                    out, err, ret = run_command_in_dir([sys.executable, '-m', 'src.eval_only', '--pred_path', f"{os.path.join(result_dir, 'preds.npy')}"], \
                                                    'evaluate/eval')
                    print(out)
                    print(err)
                    # robust: if the CLaTr subprocess failed, don't crash training
                    metric_json_path = os.path.join(result_dir, 'metrics.json')
                    if os.path.exists(metric_json_path):
                        with open(metric_json_path, 'r') as f:
                            metrics = json.load(f)
                        accelerator.log(metrics, step=global_step,)
                    else:
                        print("(CLaTr) metrics.json not produced; skipping CLaTr log this eval")

            total_loss_latent = accelerator.reduce(total_loss_latent, reduction='sum')
            total_loss_traj = accelerator.reduce(total_loss_traj, reduction='sum')
            total_samples = accelerator.reduce(total_samples, reduction='sum')
            print("valid samples:", total_samples)
            val_loss_latent_mean = (total_loss_latent / total_samples).item()
            val_loss_traj_mean = (total_loss_traj / total_samples).item()
            accelerator.log(
                {"val/loss_latent": val_loss_latent_mean,
                "val/loss_traj": val_loss_traj_mean},
                step=global_step,
            )
            if _pt:   # tau-bin + pattern val loss (Diffusion-Forcing monitoring)
                tbs = accelerator.reduce(tau_bin_sum, 'sum'); tbc = accelerator.reduce(tau_bin_cnt, 'sum')
                ps = accelerator.reduce(pat_sum, 'sum'); pc = accelerator.reduce(pat_cnt, 'sum')
                if accelerator.is_main_process:
                    logd = {f"val/tau_bin_{bi}": (tbs[bi] / tbc[bi]).item() for bi in range(10) if tbc[bi] > 0}
                    for p, nm in enumerate(['iid', 'ramp', 'bootup', 'full']):
                        if pc[p] > 0:
                            logd[f"val/pat_{nm}"] = (ps[p] / pc[p]).item()
                    accelerator.log(logd, step=global_step)

        # Keep only 2 checkpoints: last.pth (always) + best.pth (lowest val/loss_traj).
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
            if val_loss_traj_mean < best_val:
                best_val = val_loss_traj_mean
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
                print(f"(ckpt) new best val/loss_traj={best_val:.6f} @ epoch {epoch} -> best.pth")
            # full checkpoint for seamless resume (model+opt+step+epoch+best_val+ids)
            torch.save({
                'model': unwrapped_model.state_dict(),
                'opt': opt.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
                'best_val': best_val,
                'exp_name': exp_name,
                'wandb_id': _wandb_id,
            }, os.path.join(ckpt_dir, "resume.pth"))
        model.train()

    for epoch in range(start_epoch, cfg.epochs):
        pbar = tqdm(train_dataloader)
        model.train()
        total_loss = torch.tensor(0.0, device=accelerator.device)
        total_samples = torch.tensor(0, device=accelerator.device)
        t0 = time.time()
        for step, data in enumerate(pbar):
            t1 = time.time()
            data_name = data['data_name']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].to(accelerator.device)
            B = traj.shape[0]

            t2 = time.time()
            with torch.no_grad():
                if cfg.load_points:
                    if 'pc' in data:
                        point = data['pc']
                        if cfg.point_encoder != 'custom':
                            pc_embeds, pc_masks, point_obj = pc_encoder(point)
                    elif 'pc_embeds' in data:
                        pc_embeds = data['pc_embeds'].to(device)
                        pc_masks = data['pc_masks'].to(device)
                    t3 = time.time()
                    # target_dir = '/home/ckd248/data/SCVideo/camera_generation/dataset/pc_embeds_datadop'
                    # for name, embed, points_mask in zip(data_name, pc_embeds, pc_masks):
                    #     file_path = os.path.join(target_dir, name)
                    #     valid_mask = points_mask > 0
                    #     # print(data_name[0])
                    #     # print(embed[valid_mask].shape)
                    #     # print(sim[valid_mask].shape)
                    #     # print(p2v_map.shape)
                    #     # print(inv.shape)
                    #     # print(points_single.shape)
                    #     # print(scatter_mean(points[0], inv, dim=0).shape)
                    #     # torch.save(torch.cat([points_single.cpu(), torch.ones(points_single.shape[0], 3)], dim=1), 'example_scene.pt')
                    #     # input()
                    #     # print(p2v_map[inv].shape)
                    #     # print(points_mask.shape)
                    #     # print(points_mask.sum())
                    #     # print(points_single.shape)
                    #     # print(p2v_map[inv].max())
                    #     # print(embed[valid_mask].shape)
                    #     # input()
                    #     # p2v_map = p2v_map[inv]
                    #     # voxel_coord = scatter_mean(points_single, p2v_map, dim=0) # (V, 3)
                    #     # print(voxel_coord.shape)
                    #     embed_combined = embed[valid_mask] # (V, D)
                    #     # embed_combined = torch.cat([voxel_coord, embed_combined], dim=-1)
                    #     embed_data = {
                    #         "embed": embed_combined.cpu(),
                    #         # "sim": sim[valid_mask].cpu(),
                    #         # "p2v_map": p2v_map.cpu(),
                    #         "mask": points_mask.cpu()
                    #     }
                    #     torch.save(embed_data, file_path)
                    # continue
                else:
                    pc_embeds, pc_masks = None, None
                if cfg.text_encoder == 'T5':
                    text_embeds, text_masks = text_encoder(text_prompt, device)
                elif cfg.text_encoder == 'CLIP':
                    text_embeds, text_masks = encode_text_clip(text_encoder, tokenizer, text_prompt, device=device)
                text_embeds = text_embeds.float()
                text_masks = text_masks.bool()
                if cfg.use_vae:
                    traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                else:
                    traj_latents = traj
                t4 = time.time()
            
            if 'pc' in data and cfg.point_encoder == 'custom':
                pc_embeds, pc_masks, point_obj = model.pc_encoder(point)

            # Geo latent conditioning (image-based, frozen, on-the-fly). Replaces the
            # point-cloud path: feeds the latent model's geo_emb/geo_mask (5th/6th args).
            # 'geo_emb' present = the dataset served a precomputed cache hit (cfg.
            # geo_latent_cache_dir); otherwise fall back to the original LagerNVS forward.
            if 'geo_emb' in data:
                pc_embeds, pc_masks = geo_emb_from_cache(data, device)
            elif geo_encoder is not None and 'images' in data:
                with torch.no_grad():
                    pc_embeds, pc_masks = geo_encode(geo_encoder, data, device)
            if pc_embeds is not None:
                pc_embeds = attach_geo_cam(pc_embeds, data, device)

            noise = torch.randn_like(traj_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (traj_latents.shape[0],),
                device=device,
            ).long()
            if getattr(cfg, 'is_ar', False):
                # chunk-wise AR: teacher-forced causal self-attn over past clean latents
                _raw = accelerator.unwrap_model(model)
                loss = ar_train_loss(_raw, noise_scheduler, traj_latents, text_embeds, text_masks,
                                     pc_embeds, pc_masks, cfg.ar_chunk_size)
            elif getattr(cfg, 'per_token_noise', False):
                # Diffusion Forcing: per-token tau rectified flow on the causal-VAE latent
                loss = per_token_flow_loss(model, traj_latents, text_embeds, text_masks,
                                           pc_embeds, pc_masks)
            else:
                noisy_x = noise_scheduler.add_noise(traj_latents, noise, timesteps)
                noise_pred = model(noisy_x, timesteps.float(), text_embeds, text_masks, pc_embeds, pc_masks)
                loss = F.mse_loss(noise_pred, noise)
                
            t5 = time.time()
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            t6 = time.time()
            accelerator.log({"train/loss": loss.item()}, step=global_step)
            pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f}")

            total_loss += loss.detach() * B
            total_samples += B
            # print(
            #     f"data {t1-t0:.3f} | "
            #     f"ray_sim {t2-t1:.3f} | "
            #     f"point {t3-t2:.3f} | "
            #     f"text {t4-t3:.3f} | "
            #     f"model {t5-t4:.3f} | "
            #     f"loss {t6-t5:.3f} | "
            #     f"total {t6-t0:.3f} | "
            # )
            t0 = time.time()

        total_loss = accelerator.reduce(total_loss, reduction='sum')
        total_samples = accelerator.reduce(total_samples, reduction='sum')
        accelerator.log(
            {"train/epoch_loss": total_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch": epoch + 1},
            step=global_step
        )

        # Validation once per epoch (wandb curve). run_validation restores train mode
        # and keeps only last.pth + best.pth (by val/loss_traj).
        run_validation(epoch, global_step)

if __name__ == "__main__":
    train()