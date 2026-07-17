import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime
import time
import json
from pathlib import Path

from config import cfg, cfg_dict
from core_pkg.common.base import Trainer
from core_pkg.models.vae_intr import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.data_utils import out_to_trajectory, make_intrinsics, inverse_camera_matrix
from core_pkg.common.utils.eval_utils import run_command_in_dir
# from core_pkg.common.utils.render_utils import render_3dgs
from diffusers import DDPMScheduler, DDIMScheduler

from evaluate.CLaTr.clip_extraction import load_clip_model, encode_text, save_feats_custom

torch.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)

if cfg.model_type == 'baseline':
    from core_pkg.models.camera_diffusion_model import CameraDiffusionModel
if cfg.model_type == 'director':
    from core_pkg.models.camera_diffusion_model_director import CameraDiffusionModel
if cfg.model_type == 'baseline_mmdit':
    from core_pkg.models.camera_diffusion_model_mmdit import CameraDiffusionModel

if cfg.point_encoder == 'conerto':
    from core_pkg.models.pc_encoder import PCEncoder
if cfg.point_encoder == 'custom':
    from core_pkg.models.pc_encoder_custom import PCEncoder

@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, point_emb, point_mask):
    B = text_emb.size(0)
    device = text_emb.device
    x_t = torch.randn(B, traj_len, 32).to(device)

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

def valid():
    T = cfg.diffusion_max_step 
    traj_len = (cfg.num_frames - 1) // 4 + 1
    device = 'cuda:0'
    exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.abspath(os.path.join('../results_valid', exp_name))
    os.makedirs(result_dir, exist_ok=True)
    eval_data_dir = os.path.join(result_dir, 'test')
    os.makedirs(eval_data_dir, exist_ok=True)

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
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
    
    pc_encoder = PCEncoder()
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
        tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path),
        shard_fn=None)
    
    if cfg.point_encoder == 'concerto':
        model = CameraDiffusionModel()
    elif cfg.point_encoder == 'custom':
        model = CameraDiffusionModel(pc_encoder=pc_encoder)
    model.load_state_dict(torch.load(cfg.val_ckpt_path, weights_only=True))
    model = model.to(device)
    model.eval()

    camera_vae = CameraVAE().to(device)
    camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
    camera_vae.eval()

    # For evaluation
    clip_model = load_clip_model(cfg.clip_version, device=device)
    with torch.no_grad():
        pbar = tqdm(trainer.valid_batch_generator)
        val_info = {"val/loss_latent": 0, "val/loss_traj": 0}
        loss_latent_cnt = 0
        loss_traj_cnt = 0
        data_name_list = []
        for step, data in enumerate(pbar):
            data_name = data['data_name']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].to(device)
            E0 = data['first_extrinsic'].to(device)
            scale = data['avg_scale'].to(device)
            width = data['width'].to(device)
            height = data['height'].to(device)

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
            text_embeds, text_masks = text_encoder(text_prompt, device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()
            traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
        
            out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks)
        
            val_loss_latent = F.mse_loss(out, traj_latents, reduction='sum')
            val_info["val/loss_latent"] += val_loss_latent.item()
            loss_latent_cnt += out.numel()

            traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)
            val_loss_traj = F.mse_loss(traj_pred, traj, reduction='sum')
            val_info["val/loss_traj"] += val_loss_traj.item()
            loss_traj_cnt += traj_pred.numel()

            ref_intrinsics = make_intrinsics(traj[:, :, -2:], width, height).cpu().tolist()
            pred_intrinsics = make_intrinsics(traj_pred[:, :, -2:], width, height).cpu().tolist()
            width = width.cpu().tolist()
            height = height.tolist()

            traj = out_to_trajectory(traj, scale, E0, device)
            traj_pred = out_to_trajectory(traj_pred, scale, E0, device)
            # render_3dgs('/data1/cympyc1785/SceneData/DL3DV/scenes/7K/19758c1aa9e5b35f94a798c8941f39f4d4a46d4cf96e13734b811f99e7bf7e95/scene.ply', \
            #             traj[0], ref_intrinsics[0], 'test_render.mp4')
            # print("XX")
            # input()
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
                        "transform_matrix": matrix_traj_ref[i][frame_idx], 
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
        valid_txt_path = os.path.join(result_dir, 'test_valid.txt')
        with open(valid_txt_path, "w", encoding="utf-8") as f:
            for name in data_name_list:
                f.write(name + "\n")
        
        cmd = f'python -m src.extraction checkpoint_path={cfg.clatr_ckpt_path} data_dir={result_dir}'
        print(cmd)
        out, err, ret = run_command_in_dir(cmd.split(' '), \
                                            'evaluate/CLaTr')
        print(out, err)

        cmd = f"python -m src.eval_only --pred_path {os.path.join(result_dir, 'preds.npy')}"
        print(cmd)
        out, err, ret = run_command_in_dir(cmd.split(' '), \
                                            'evaluate/eval')
        print(out, err)
        metric_json_path = os.path.join(result_dir, 'metrics.json')
        with open(metric_json_path, 'r') as f:
            metrics = json.load(f)
        val_info['val/loss_latent'] /= loss_latent_cnt
        val_info['val/loss_traj'] /= loss_traj_cnt
        val_info.update(metrics)
        print(val_info)

if __name__ == "__main__":
    valid()