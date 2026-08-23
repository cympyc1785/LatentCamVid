import torch
import torch.nn.functional as F
import numpy as np
import os
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

from config_large import cfg, cfg_dict
from core_pkg.common.base import Trainer
from core_pkg.models.vae_intr_large import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.data_utils import out_to_trajectory, make_intrinsics, inverse_camera_matrix
from core_pkg.common.utils.eval_utils import run_command_in_dir
from core_pkg.common.utils.pc_utils import get_ray_sim_per_point
from diffusers import DDPMScheduler, DDIMScheduler

from evaluate.CLaTr.clip_extraction import load_clip_model, encode_text, save_feats_custom

torch.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)
random.seed(cfg.random_seed)

if cfg.model_type == 'baseline':
    from core_pkg.models.camera_diffusion_model_base import CameraDiffusionModel
if cfg.model_type == 'director':
    from core_pkg.models.camera_diffusion_model_director import CameraDiffusionModel
if cfg.model_type == 'baseline_mmdit':
    from core_pkg.models.camera_diffusion_model_mmdit import CameraDiffusionModel
if cfg.model_type == 'baseline_attn_sup':
    from core_pkg.models.camera_diffusion_model_attn_sup import CameraDiffusionModel

if cfg.point_encoder == 'concerto':
    from core_pkg.models.pc_encoder import PCEncoder
if cfg.point_encoder == 'custom':
    from core_pkg.models.pc_encoder_custom import PCEncoder
if cfg.point_encoder == 'mosaic':
    from core_pkg.models.pc_encoder_mosaic import PCEncoder

@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, point_emb, point_mask, null_text_embeds=None, null_text_masks=None, from_t=None, traj_latents=None, cfg_weight=0.0):
    B = text_emb.size(0)
    device = text_emb.device
        
    x_t = torch.randn(B, traj_len, cfg.cam_dim).to(device)
    noise = x_t.clone()
    if from_t is not None and traj_latents is not None:
        from_t = torch.tensor(from_t).to(device)
        x_t = scheduler.add_noise(traj_latents, x_t, from_t)

    scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)
    for t in scheduler.timesteps:
        if from_t is not None and t > from_t:
            continue
        timesteps = torch.full(
            (B,),
            t,
            device=device,
        )
        if cfg.model_type == 'baseline_attn_sup':
            noise_pred, attn_weight = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
            noise_uncond, _ = model(x_t, timesteps.float(), null_text_embeds, null_text_masks, point_emb, point_mask)
        else:
            noise_pred = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
            noise_uncond = model(x_t, timesteps.float(), null_text_embeds, null_text_masks, point_emb, point_mask)
        if cfg_weight != 0:
            noise_pred = noise_pred + cfg_weight * (noise_pred - noise_uncond)
        output = scheduler.step(noise_pred, t, x_t)
        x_t = output.prev_sample
        clean_x = output.pred_original_sample
    return x_t

def train():
    T = cfg.diffusion_max_step 
    traj_len = cfg.num_cam
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    if accelerator.is_main_process:
        exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f'_{cfg.exp_name}'
    else:
        exp_name = None
    exp_name = broadcast_object_list([exp_name])[0]
    result_dir = os.path.abspath(os.path.join('../results_valid_acc', exp_name))
    os.makedirs(result_dir, exist_ok=True)
    eval_data_dir = os.path.join(result_dir, 'test')
    os.makedirs(eval_data_dir, exist_ok=True)

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator(include_train=False)
    trainer._make_model()

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

    pc_encoder = PCEncoder().to(device)
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
    
    if cfg.point_encoder != 'custom':
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim)
    else:
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim, pc_encoder=pc_encoder)
    ckpt = torch.load(cfg.val_ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    
    model.eval()

    camera_vae = CameraVAE().to(device)
    camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
    camera_vae.eval()

    # For evaluation
    clip_model = load_clip_model(cfg.clip_version, device=device)

    global_step = 0

    model, valid_dataloader = accelerator.prepare(
        model, trainer.valid_batch_generator
    )
    val_info = {}
    with torch.no_grad():
        pbar = tqdm(valid_dataloader)
        total_loss_latent = torch.tensor(0.0, device=accelerator.device)
        total_loss_traj = torch.tensor(0.0, device=accelerator.device)
        total_samples = torch.tensor(0, device=accelerator.device)
        data_name_list = []
        for step, data in enumerate(pbar):
            data_name = data['data_name']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].to(device)
            E0 = data['first_extrinsic'].to(device)
            scale = data['avg_scale'].to(device)
            width = data['width'].to(device)
            height = data['height'].to(device)
            extrinsics = data['extrinsics']
            intrinsics = data['intrinsics']

            B = traj.shape[0]

            if cfg.load_points:
                if 'pc' in data:
                    point = data['pc']
                    if cfg.point_encoder != 'custom':
                        points = data['points']
                        inverse = data['inverse']
                        ray_info_list = [get_ray_sim_per_point(w2c, K, scatter_mean(points_single, inv, dim=0), device=device) for w2c, K, points_single, inv in zip(extrinsics, intrinsics, points, inverse)]
                        pc_embeds, ray_sims, pc_masks, point_obj = pc_encoder(point, ray_info_list)
                    else:
                        pc_embeds, pc_masks, point_obj = model.pc_encoder(point)
                elif 'pc_embeds' in data:
                    pc_embeds = data['pc_embeds'].to(device)
                    pc_masks = data['pc_masks'].to(device)
                    # Ablation for binary and coord concat
                    # remove binary
                    # pc_embeds = torch.cat([pc_embeds[:, :, :3], pc_embeds[:, :, 4:]], dim=2)
                    # remove coord
                    # pc_embeds = pc_embeds[:, :, 3:]
                    # remove both
                    # pc_embeds = pc_embeds[:, :, 4:]
            else:
                pc_embeds, pc_masks = None, None
            if cfg.text_encoder == 'T5':
                text_embeds, text_masks = text_encoder(text_prompt, device)
            elif cfg.text_encoder == 'CLIP':
                text_embeds, text_masks = encode_text_clip(text_encoder, tokenizer, text_prompt, device=device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()


            null_text_embeds, null_text_masks = text_encoder([""] * len(text_prompt), device)
            null_text_embeds = null_text_embeds.float()
            null_text_masks = null_text_masks.bool()

            if cfg.use_vae:
                traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                # traj_latents, mu, log_var = camera_vae(traj)
                # traj_latents = camera_vae.reparameterize(mu, log_var).transpose(-1, -2) / cfg.vae_latent_scale
            else:
                traj_latents = traj
        
            out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks, null_text_embeds, null_text_masks, cfg_weight=0.0)
        
            val_loss_latent = F.mse_loss(out, traj_latents, reduction='mean')
            total_loss_latent += val_loss_latent * B

            if cfg.use_vae:
                traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)
            else:
                traj_pred = out
            
            val_loss_traj = F.mse_loss(traj_pred, traj, reduction='mean')
            total_loss_traj += val_loss_traj * B
            total_samples += B

            if cfg.max_trans_norm:
                ref_intrinsics = make_intrinsics(traj[:, :-1, -2:], width, height, intrinsics).cpu().tolist()
                pred_intrinsics = make_intrinsics(traj_pred[:, :-1, -2:], width, height, intrinsics).cpu().tolist()
            else:
                ref_intrinsics = make_intrinsics(traj[:, :, -2:], width, height, intrinsics).cpu().tolist()
                pred_intrinsics = make_intrinsics(traj_pred[:, :, -2:], width, height, intrinsics).cpu().tolist()
            width = width.cpu().tolist()
            height = height.tolist()
            
            # anchor_pred_frame0: target 첫 카메라를 given 으로 (rel[0]=I 강제). 기본 False = 기존 동작.
            _a0 = bool(getattr(cfg, 'anchor_pred_frame0', False))
            traj = out_to_trajectory(traj, scale, E0, device, anchor_frame0=_a0)
            traj_pred = out_to_trajectory(traj_pred, scale, E0, device, anchor_frame0=_a0)
            matrix_traj_ref = inverse_camera_matrix(traj)
            matrix_traj_pred = inverse_camera_matrix(traj_pred)
            matrix_traj_ref[:, :, :3, 1:3] *= -1
            matrix_traj_pred[:, :, :3, 1:3] *= -1
            matrix_traj_ref = matrix_traj_ref.cpu().tolist()
            matrix_traj_pred = matrix_traj_pred.cpu().tolist()

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

        total_loss_latent = accelerator.reduce(total_loss_latent, reduction='sum')
        total_loss_traj = accelerator.reduce(total_loss_traj, reduction='sum')
        total_samples = accelerator.reduce(total_samples, reduction='sum')
        all_data_name_list = sorted(set(gather_object(data_name_list)))
        if accelerator.is_main_process:
            valid_txt_path = os.path.join(result_dir, 'test_valid.txt')
            with open(valid_txt_path, "w", encoding="utf-8") as f:
                for name in all_data_name_list:
                    f.write(name + "\n")
            
            out, err, ret = run_command_in_dir(['python', '-m', 'src.extraction', f'checkpoint_path={cfg.clatr_ckpt_path}', f'data_dir={result_dir}'], \
                                            'evaluate/CLaTr')
            print(out)
            print(err)
            out, err, ret = run_command_in_dir(['python', '-m', 'src.eval_only', '--pred_path', f"{os.path.join(result_dir, 'preds.npy')}"], \
                                            'evaluate/eval')
            print(out)
            print(err)
            metric_json_path = os.path.join(result_dir, 'metrics.json')
            with open(metric_json_path, 'r') as f:
                metrics = json.load(f)
            print(metrics)
            print("valid samples:", total_samples)
            print(
                {"val/loss_latent": total_loss_latent / total_samples, 
                "val/loss_traj": total_loss_traj / total_samples}
            )

if __name__ == "__main__":
    train()