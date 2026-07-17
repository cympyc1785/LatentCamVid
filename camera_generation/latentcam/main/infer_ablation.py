import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import datetime

from config_large import cfg
from core_pkg.common.base import Trainer
from core_pkg.models.camera_diffusion_model import CameraDiffusionModel
from core_pkg.models.pc_encoder_custom import PCEncoder
from core_pkg.models.vae_intr_large import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.vis_utils import attention_bar, attn_to_color
from core_pkg.common.utils.data_utils import out_to_trajectory, make_intrinsics
from core_pkg.common.utils.pc_utils import get_ray_sim_per_point
from torch_scatter import scatter_mean

from diffusers import DDPMScheduler, DDIMScheduler
import copy

torch.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)

# ['t', 'ts', 'tsv', 'tsva', 'tsvac]
ablation_type = 5

if ablation_type == 0:
    cfg.val_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results/20260302_035935_only_text/ckpts/200.pth'
    cfg.model_type = 'baseline'
    cfg.load_points = False
    cfg.load_saved_pc_embeds = False
    cfg.use_vae = True
    cfg.num_cam = 13
    cfg.cam_dim = 64
if ablation_type == 1:
    cfg.val_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results/20260301_081801_wo_vae_w_bin_wo_attnloss/ckpts/250.pth'
    cfg.model_type = 'baseline'
    cfg.load_points = True
    cfg.load_saved_pc_embeds = False
    cfg.use_vae = False
    cfg.num_cam = 49
    cfg.cam_dim = 11
if ablation_type == 2:
    cfg.val_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results/20260301_164439_w_vae_w_bin_wo_attnloss/ckpts/300.pth'
    cfg.model_type = 'baseline'
    cfg.load_points = True
    cfg.load_saved_pc_embeds = False
    cfg.use_vae = True
    cfg.num_cam = 13
    cfg.cam_dim = 64
if ablation_type == 3:
    cfg.val_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results/20260301_164408_w_vae_w_bin_w_attnloss/ckpts/400.pth'
    cfg.model_type = 'baseline_attn_sup'
    cfg.load_points = True
    cfg.load_saved_pc_embeds = False
    cfg.use_vae = True
    cfg.num_cam = 13
    cfg.cam_dim = 64
if ablation_type == 4:
    cfg.val_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/evaluation/20260302_202052_w_vae_w_bin_w_attnloss_w_coord_2/ckpts/225.pth'
    cfg.model_type = 'baseline_attn_sup'
    cfg.load_points = True
    cfg.load_saved_pc_embeds = True
    cfg.use_vae = True
    cfg.num_cam = 13
    cfg.cam_dim = 64
    cfg.embed_path = '/home/ckd248/data/SCVideo/camera_generation/dataset/pc_embeds_inv'

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
    # hyperparams
    T = cfg.diffusion_max_step
    traj_len = cfg.num_cam
    cfg.batch_size = 1

    device = 'cuda:0'
    ckpt_path = cfg.val_ckpt_path
    result_dir = os.path.abspath('./ablation')
    os.makedirs(result_dir, exist_ok=True)

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
    noise_scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)

    # model
    pc_encoder = PCEncoder()
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
        tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path),
        shard_fn=None)
    
    if cfg.point_encoder != 'custom':
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim)
    else:
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim, pc_encoder=pc_encoder)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model = model.to(device)
    model.eval()

    camera_vae = CameraVAE().cuda()
    camera_vae.load_state_dict(
        torch.load(
            cfg.vae_ckpt_path,
            map_location="cuda"
        )
    )

    camera_vae.eval()
    pbar = tqdm(trainer.valid_batch_generator)
    with torch.no_grad():
        for step, data in enumerate(pbar):
            data_name = data['data_name']
            text = data['text_prompt']
            traj = data['cam_param'].cuda()
            E0 = data['first_extrinsic'].cuda()
            scale = data['avg_scale'].cuda()
            width = data['width'].cuda()
            height = data['height'].cuda()
            extrinsics = data['extrinsics']
            intrinsics = data['intrinsics']
            # points = data['points']
            # inverse = data['inverse']

            if cfg.load_points and cfg.load_saved_pc_embeds == False:
                ray_info_list = [get_ray_sim_per_point(w2c, K, scatter_mean(points_single, inv, dim=0), c2w=False, device=device) for w2c, K, points_single, inv in zip(extrinsics, intrinsics, points, inverse)]
            else:
                ray_info_list = None

            # if '19758c1aa9e5b35f94a798c8941f39f4d4a46d4cf96e13734b811f99e7bf7e95' not in data_name[0]:
            #     continue
            # text = ['The camera captures the basketball court.']

            if cfg.load_points:
                if 'pc' in data:
                    point = data['pc']
                    if cfg.point_encoder != 'custom':
                        pc_embeds, ray_sims, p2v_map_list, pc_masks, point_obj = pc_encoder(point, ray_info_list)
                    else:
                        pc_embeds, pc_masks, point_obj = model.pc_encoder(point)
                elif 'pc_embeds' in data:
                    pc_embeds = data['pc_embeds'].to(device)
                    pc_masks = data['pc_masks'].to(device)
            else:
                pc_embeds, pc_masks = None, None
                
            text_embeds, text_masks = text_encoder(text, device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()

            traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale

            # while True:
            #     noise = torch.randn_like(traj_latents)
                
            #     timesteps = torch.randint(
            #         0,
            #         noise_scheduler.config.num_train_timesteps,
            #         (traj_latents.shape[0],),
            #         device=device,
            #     ).long()
            #     # timesteps[0] = 0
            #     noisy_x = noise_scheduler.add_noise(traj_latents, noise, timesteps)
            #     noise_pred = model(noisy_x, timesteps.float(), text_embeds, text_masks, pc_embeds, pc_masks)
            #     loss = F.mse_loss(noise_pred, noise)
            #     print(timesteps)
            #     print(loss)
            #     input()
            flag = 1
            while flag: 
                out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks)

                if cfg.use_vae:
                    traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                else:
                    traj_latents = traj
                val_loss = F.mse_loss(out, traj_latents)
                print("Loss (Latent):", val_loss.item())
                if cfg.use_vae:
                    out = camera_vae.decode(out * cfg.vae_latent_scale)
                else:
                    out = out
                traj_loss = F.mse_loss(out, traj)
                print("Loss (Traj):", traj_loss.item())

                print("Intrinsics (pred, gt)")
                pred_intrinsics = make_intrinsics(out[:, :, -2:], width, height, intrinsics.to(device))
                gt_intrinsics = make_intrinsics(traj[:, :, -2:], width, height, intrinsics.to(device))
                print(pred_intrinsics[0][0])
                print(gt_intrinsics[0][0])

                pred_traj = out_to_trajectory(out, scale, E0, device)
                gt_traj = out_to_trajectory(traj, scale, E0, device)
                if cfg.use_vae:
                    print("GT - GT: ", F.mse_loss(camera_vae.decode(traj_latents * cfg.vae_latent_scale), traj))
                else:
                    transform = torch.linalg.inv(pred_traj[0][0])
                    pred_traj = E0[0] @ transform @ pred_traj
                print("after denormalize")
                print(pred_traj[0][0])
                print(gt_traj[0][0])

                pred_path = os.path.join(result_dir, f'{ablation_type}_{os.path.basename(data_name[0])}_pred.npz')
                np.savez(pred_path, poses=pred_traj[0].cpu().numpy(), intrinsics=pred_intrinsics[0].cpu().numpy())
                gt_path = os.path.join(result_dir, f'{os.path.basename(data_name[0])}_gt.npz')
                np.savez(gt_path, poses=gt_traj[0].cpu().numpy(), intrinsics=gt_intrinsics[0].cpu().numpy())

                # normalized_traj = out_to_trajectory(traj, torch.ones(1)[None].to(device), torch.eye(4)[None].to(device), device)
                # np.savez(gt_path, poses=normalized_traj[0].cpu().numpy(), intrinsics=gt_intrinsics[0].cpu().numpy())

                print("Data_name")
                print(data_name)
                print("Prediction, GT path")
                print(pred_path)
                print(gt_path)
                print("Text Prompt")
                print(text)

                t = input()
                if t == 'a':
                    flag = 0

            # print("Scene save")
            # scene_point = points[0][:, :3].cuda()
            # # scene_point = scene_point * scale[0]
            # # point[0][:, :3] = scene_point @ E0[0, :3, :3].T + E0[0, :3, 3]
            # torch.save(torch.cat([points[0].cpu(), point['color'][inverse[0]].cpu()], dim=1).cpu(), 'scene.pt')

            # gt_path = os.path.join(f'gt.npz')
            # gt_traj = out_to_trajectory(traj, torch.tensor([1.0]).cuda(), torch.eye(4)[None].cuda(), device)
            # np.savez(gt_path, poses=gt_traj[0].cpu().numpy(), intrinsics=gt_intrinsics[0].cpu().numpy())
            # gt_traj = extrinsics
            # np.savez(gt_path, poses=extrinsics[0].cpu().numpy(), intrinsics=intrinsics[0].cpu().numpy())
            
            # print(gt_path)

            if hasattr(model, "text_cross_attn_weight") and model.text_cross_attn_weight is not None:
                # print(model.text_cross_attn_weight.shape)
                cross_attn_weight = model.text_cross_attn_weight[0]
                assert cross_attn_weight.shape[0] == traj_latents.shape[1] and cross_attn_weight.shape[1] == text_embeds.shape[1], f"{cross_attn_weight.shape[0]} == {traj_latents.shape[1]} , {cross_attn_weight.shape[1]} == {text_embeds.shape[1]}"

                text_weight = cross_attn_weight
                text_max, text_max_idx = text_weight.max(dim=-1)

                text_tokens = text_encoder.tokenizer.tokenizer.tokenize(text[0])
                attention_path = os.path.join(result_dir, "text_attention.png")
                # print(len(text_tokens))
                # print(text_weight.mean(dim=0)[:len(text_tokens)].sum())
                # print(text_weight.mean(dim=0)[len(text_tokens)])
                attention_bar(tokens=text_tokens, attn=text_weight.mean(dim=0)[:len(text_tokens)], save_path=attention_path)

                print("Text attention saved in : ", attention_path)
                # print(text_tokens)

            # if hasattr(model, "point_cross_attn_weight") and model.point_cross_attn_weight is not None:
            #     print(model.point_cross_attn_weight.shape)
            #     cross_attn_weight = model.point_cross_attn_weight[0]
            #     assert cross_attn_weight.shape[0] == traj_latents.shape[1] and cross_attn_weight.shape[1] == pc_embeds.shape[1], f"{cross_attn_weight.shape[0]} == {traj_latents.shape[1]} , {cross_attn_weight.shape[1]} == {pc_embeds.shape[1]}"
                
            #     point_weight = cross_attn_weight
                
            #     point_max, point_max_idx = point_weight.max(dim=-1)

            #     p2v_map = pc_encoder.get_p2v_map(copy.deepcopy(point_obj))[0] # (N_points)
            #     point_attention = cross_attn_weight.mean(dim=0)

            #     # print(point_attention.shape)
            #     # print(points[0].shape)
            #     # print(p2v_map.shape)
            #     # print(p2v_map[inverse].shape)

            #     point_attn_vis = attn_to_color(point_attention.detach().cpu().numpy())
            #     point_attention_path = os.path.join(result_dir, f"{ablation_type}_{data_name[0]}_point_attention.pt")
            #     point_attn_vis = point_attn_vis[p2v_map[inverse[0]].cpu().numpy()]
            #     # print(point_attn_vis.shape)

            #     points[0] = points[0].cuda()
            #     pc_pos = points[0][:, :3]
            #     pc_pos *= scale[0]
            #     pc_pos = (pc_pos - E0[0:1, :3, 3]) @ E0[0, :3, :3]
            #     points[0][:, :3] = pc_pos
            #     ray_sims = ray_sims[0][p2v_map[inverse[0]], 0]
            #     is_vis = pc_embeds[0, :, 0][p2v_map[inverse[0]]]
                
            #     scene = torch.cat([points[0][:, :3], torch.tensor(point_attn_vis).to(points[0])], dim=1)
            #     # print(scene.shape)
            #     torch.save(scene.cpu(), point_attention_path)
            #     scene = torch.cat([points[0][:, :3], ray_sims.unsqueeze(1).repeat(1, 3)], dim=1)
            #     torch.save(scene.cpu(), 'ray_sim.pt')
            #     scene = torch.cat([points[0][:, :3], is_vis.unsqueeze(1).repeat(1, 3)], dim=1)
            #     torch.save(scene.cpu(), 'is_vis.pt')

            #     print('Point attention saved in : ', point_attention_path)
            input()

if __name__ == "__main__":
    train()
