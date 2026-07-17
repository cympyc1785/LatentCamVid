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
        if cfg.model_type == 'baseline_attn_sup':
            noise_pred, attn_weight = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
        else:
            noise_pred = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t

def train():
    T = cfg.diffusion_max_step 
    traj_len = cfg.num_cam
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    if accelerator.is_main_process:
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

    accelerator.init_trackers(
        project_name="camera-diffusion-training", 
        config=cfg_dict,
        init_kwargs={
            "wandb": {
                "name": f"{exp_name}"
            }
        }
    )

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
    if cfg.load_ckpt_path is not None:
        ckpt = torch.load(cfg.load_ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
    
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    camera_vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
    camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
    camera_vae.eval()

    # For evaluation
    clip_model = load_clip_model(cfg.clip_version, device=device)

    global_step = 0

    model, opt, train_dataloader, valid_dataloader = accelerator.prepare(
        model, opt, trainer.batch_generator, trainer.valid_batch_generator
    )


    for epoch in range(cfg.epochs):
        pbar = tqdm(train_dataloader)
        model.train()
        total_loss = torch.tensor(0.0, device=accelerator.device)
        total_recon_loss = torch.tensor(0.0, device=accelerator.device)
        total_kl_loss = torch.tensor(0.0, device=accelerator.device)
        total_samples = torch.tensor(0, device=accelerator.device)
        t0 = time.time()
        for step, data in enumerate(pbar):
            t1 = time.time()
            data_name = data['data_name']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].to(accelerator.device)
            # extrinsics = data['extrinsics']
            # intrinsics = data['intrinsics']
            # points = data['points']
            # inverse = data['inverse']

            B = traj.shape[0]

            with torch.no_grad():
                if cfg.load_points:
                    if 'pc' in data:
                        point = data['pc']
                        if cfg.point_encoder != 'custom':
                            ray_info_list = [get_ray_sim_per_point(w2c, K, scatter_mean(points_single, inv, dim=0), c2w=False, device=device) for w2c, K, points_single, inv in zip(extrinsics, intrinsics, points, inverse)]
                            pc_embeds, ray_sims, p2v_map_list, pc_masks, point_obj = pc_encoder(point, ray_info_list)
                            # target_dir = '/home/ckd248/data/SCVideo/camera_generation/dataset/pc_embeds_inv'
                            # for name, embed, sim, mask, p2v_map, inv, points_single, points_mask in zip(data_name, pc_embeds, ray_sims, pc_masks, p2v_map_list, inverse, points, data['points_mask']):
                            #     file_path = os.path.join(target_dir, name)
                            #     valid_mask = mask > 0
                            #     p2v_map = p2v_map[inv]
                            #     voxel_coord = scatter_mean(points_single, p2v_map, dim=0) # (V, 3)
                            #     # print(voxel_coord.shape)
                            #     embed_combined = embed[valid_mask] # (V, D)
                            #     embed_combined = torch.cat([voxel_coord, embed_combined], dim=-1)
                            #     embed_data = {
                            #         "embed": embed_combined.cpu(),
                            #         "sim": sim[valid_mask].cpu(),
                            #         "p2v_map": p2v_map.cpu(),
                            #         "mask": points_mask.cpu()
                            #     }
                            #     torch.save(embed_data, file_path)
                            # continue
                    elif 'pc_embeds' in data:
                        pc_embeds = data['pc_embeds'].to(device)
                        pc_masks = data['pc_masks'].to(device)
                        pc_sims = data['pc_sim'] # (B, P, T)
                    t2 = time.time()
                    # PC Embeds preprocessing
                    # # print(data_name)
                    # data_path = os.path.join(trainer.batch_generator.dataset.dataset_path, data_name[0])
                    # # pc_feat_path = os.path.join(data_path, 'pc_embeds.pt')
                    # pc_decoded_feat_path = os.path.join(data_path, 'pc_embeds_decoded.pt')
                    # pc_feat = pc_embeds[0][pc_masks[0]]
                    # torch.save(pc_feat, pc_decoded_feat_path)
                    # # print(pc_masks[0].sum())
                    # # print(pc_embeds.shape)
                    # # input()

                    # # decoded_feat, point_obj[0] = pc_encoder.decode(point_obj, num_iter=1)
                    # # print(decoded_feat.shape)
                    
                    # # torch.save(decoded_feat, pc_decoded_feat_path)
                    # continue
                else:
                    pc_embeds, pc_masks = None, None
                if cfg.text_encoder == 'T5':
                    text_embeds, text_masks = text_encoder(text_prompt, device)
                elif cfg.text_encoder == 'CLIP':
                    text_embeds, text_masks = encode_text_clip(text_encoder, tokenizer, text_prompt, device=device)
                text_embeds = text_embeds.float()
                text_masks = text_masks.bool()
                traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                t3 = time.time()
            
            if 'pc' in data and cfg.point_encoder == 'custom':
                pc_embeds, pc_masks, point_obj = model.pc_encoder(point)

            noise = torch.randn_like(traj_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (traj_latents.shape[0],),
                device=device,
            ).long()

            loss = 0.0

            noisy_x = noise_scheduler.add_noise(traj_latents, noise, timesteps)
            if cfg.model_type == 'baseline_attn_sup':
                noise_pred, point_ca_weights = model(noisy_x, timesteps.float(), text_embeds, text_masks, pc_embeds, pc_masks)
            else:
                noise_pred = model(noisy_x, timesteps.float(), text_embeds, text_masks, pc_embeds, pc_masks)
            recon_loss = F.mse_loss(noise_pred, noise)
            loss += recon_loss

            if cfg.model_type == 'baseline_attn_sup':
                eps = 1e-8
                _, B, N_c, N_p = point_ca_weights.shape # (N_layer, B, N_c, N_p)
                # ray_sims = ray_sims.squeeze(-1)
                # target = ray_sims.masked_fill(~pc_masks, -1e9)
                # target = F.softmax(target, dim=-1) * pc_masks
                # target = target / (target.sum(dim=-1, keepdim=True) + eps) # (B, N_p)
                # target = target.unsqueeze(1).expand(B, N_c, N_p).unsqueeze(0) # (1, B, N_c, N_p)

                pc_sims = pc_sims.unsqueeze(0).permute(0, 1, 3, 2) # (1, B, N_c, N_p)
                masked_sims = pc_sims.masked_fill(
                    ~pc_masks.unsqueeze(0).unsqueeze(2),
                    -1e9
                )
                target = torch.softmax(masked_sims, dim=-1) * pc_masks.unsqueeze(0).unsqueeze(2)
                target = target / (target.sum(dim=-1, keepdim=True) + eps) # (1, B, N_c, N_p)

                pred = point_ca_weights * pc_masks.unsqueeze(0).unsqueeze(2)
                pred = pred / (pred.sum(dim=-1, keepdim=True) + eps)
                log_pred = torch.log(pred + eps)

                kl_loss = F.kl_div(
                    log_pred,
                    target,
                    reduction="none"
                )  # (L,B,T,P)
                kl_loss = kl_loss * pc_masks.unsqueeze(0).unsqueeze(2)
                kl_loss = kl_loss.sum(dim=-1).mean()
                loss += cfg.attn_loss_weight * kl_loss
            else:
                kl_loss = torch.tensor(0.0).to(device)
                
            t4 = time.time()
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            t5 = time.time()
            accelerator.log({"train/loss": loss.item()}, step=global_step)
            accelerator.log({"train/recon_loss": recon_loss.item()}, step=global_step)
            accelerator.log({"train/kl_loss": kl_loss.item()}, step=global_step)
            pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f}")

            total_loss += loss.detach() * B
            total_recon_loss += recon_loss.detach() * B
            total_kl_loss += kl_loss.detach() * B
            total_samples += B
            # print(
            #     f"data {t1-t0:.3f} | "
            #     f"pc {t2-t1:.3f} | "
            #     f"text {t3-t2:.3f} | "
            #     f"model {t4-t3:.3f} | "
            #     f"loss {t5-t4:.3f} | "
            #     f"total {t5-t0:.3f} | "
            # )
            t0 = time.time()

        total_loss = accelerator.reduce(total_loss, reduction='sum')
        total_recon_loss = accelerator.reduce(total_recon_loss, reduction='sum')
        total_kl_loss = accelerator.reduce(total_kl_loss, reduction='sum')
        total_samples = accelerator.reduce(total_samples, reduction='sum')
        accelerator.log(
            {"train/epoch_loss": total_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch_recon_loss": total_recon_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch_kl_loss": total_kl_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch": epoch + 1},
            step=global_step
        )

        if epoch % cfg.save_epoch == 0:
            model.eval()
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
                    # extrinsics = data['extrinsics']
                    intrinsics = data['intrinsics']
                    # points = data['points']
                    # inverse = data['inverse']

                    # ray_info_list = [get_ray_sim_per_point(w2c, K, scatter_mean(points_single, inv, dim=0), device=device) for w2c, K, points_single, inv in zip(extrinsics, intrinsics, points, inverse)]
                    B = traj.shape[0]

                    if cfg.load_points:
                        if 'pc' in data:
                            point = data['pc']
                            if cfg.point_encoder != 'custom':
                                pc_embeds, ray_sims, pc_masks, point_obj = pc_encoder(point)
                            else:
                                pc_embeds, pc_masks, point_obj = model.pc_encoder(point)
                        elif 'pc_embeds' in data:
                            pc_embeds = data['pc_embeds'].to(device)
                            pc_masks = data['pc_masks'].to(device)

                    else:
                        pc_embeds, pc_masks = None, None
                    if cfg.text_encoder == 'T5':
                        text_embeds, text_masks = text_encoder(text_prompt, device)
                    elif cfg.text_encoder == 'CLIP':
                        text_embeds, text_masks = encode_text_clip(text_encoder, tokenizer, text_prompt, device=device)
                    text_embeds = text_embeds.float()
                    text_masks = text_masks.bool()
                    traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                
                    out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks)
                
                    val_loss_latent = F.mse_loss(out, traj_latents, reduction='mean')
                    total_loss_latent += val_loss_latent * B

                    traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)
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
                    
                    traj = out_to_trajectory(traj, scale, E0, device, max_trans_norm=cfg.max_trans_norm)
                    traj_pred = out_to_trajectory(traj_pred, scale, E0, device, max_trans_norm=cfg.max_trans_norm)
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
                    accelerator.log(metrics, step=global_step,)

                total_loss_latent = accelerator.reduce(total_loss_latent, reduction='sum')
                total_loss_traj = accelerator.reduce(total_loss_traj, reduction='sum')
                total_samples = accelerator.reduce(total_samples, reduction='sum')
                print("valid samples:", total_samples)
                accelerator.log(
                    {"val/loss_latent": total_loss_latent / total_samples, 
                    "val/loss_traj": total_loss_traj / total_samples},
                    step=global_step,
                )
                
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))

if __name__ == "__main__":
    train()