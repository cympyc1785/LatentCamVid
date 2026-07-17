import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import datetime

from config import cfg
from core_pkg.common.base import Trainer
from core_pkg.models.camera_diffusion_model import CameraDiffusionModel
from core_pkg.models.pc_encoder import PCEncoder
from core_pkg.models.vae_intr import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.vis_utils import attention_bar, attn_to_color
from core_pkg.common.utils.data_utils import out_to_trajectory, make_intrinsics, normalize_camera_extrinsics_and_points

from diffusers import DDPMScheduler, DDIMScheduler
import copy
import json

if cfg.model_type == 'baseline':
    from core_pkg.models.camera_diffusion_model import CameraDiffusionModel
if cfg.model_type == 'director':
    from core_pkg.models.camera_diffusion_model_director import CameraDiffusionModel

@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, point_emb, point_mask, from_t=None, traj_latents=None):
    B = text_emb.size(0)
    device = text_emb.device

    x_t = torch.randn(B, traj_len, 32).to(device)
    noise = x_t.clone()
    if from_t is not None and traj_latents is not None:
        from_t = torch.tensor(from_t).to(device)
        x_t = scheduler.add_noise(traj_latents, x_t, from_t)

    for t in tqdm(scheduler.timesteps):
        if from_t is not None and t > from_t:
            continue
        timesteps = torch.full(
            (B,),
            t,
            device=device,
        )
        noise_pred = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask)
        output = scheduler.step(noise_pred, t, x_t)
        x_t = output.prev_sample
    return x_t

# hyperparams
T = cfg.diffusion_max_step 
traj_len = (cfg.num_frames - 1) // 4 + 1

data_dir_path = '/data1/cympyc1785/SceneData/DL3DV/scenes/2K/00f32fbf46f60a718942a417962d42d998317725fb59248ed728ee69ac3a3e8f'
scene_path = os.path.join(data_dir_path, 'scene.pt')
camera_path = os.path.join(data_dir_path, 'e0.json')
text = 'A camera moves right and backward while caputring the ground.'
width = 832
height = 480

device = 'cuda:0'
ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results/20260212_083947/ckpts/700.pth'
exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = os.path.abspath(os.path.join('../infer_ddpm', exp_name))
os.makedirs(result_dir, exist_ok=True)

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
model = CameraDiffusionModel().cuda()
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()

pc_encoder = PCEncoder()
text_encoder = T5EncoderModel(
    text_len=cfg.text_len,
    dtype=cfg.t5_dtype,
    device=device,
    checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
    tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path),
    shard_fn=None)

camera_vae = CameraVAE().cuda()
camera_vae.load_state_dict(
    torch.load(
        cfg.vae_ckpt_path,
        map_location="cuda"
    )
)
camera_vae.eval()
with torch.no_grad():
    with open(camera_path, 'r') as f:
        E0 = torch.tensor(json.load(f))[None].to(device) # (1, 4, 4), w2c
    width = torch.tensor([width]).unsqueeze(1).repeat(1, cfg.num_frames).to(device)
    height = torch.tensor([height]).unsqueeze(1).repeat(1, cfg.num_frames).to(device)
    text_prompt = [text]
    point = [torch.load(scene_path).to(device)] # (N, 6), coordinates (3) + color (3)
    normalized_extrinsics, normalized_scene_pc, scale, _ = normalize_camera_extrinsics_and_points(E0, point[0][:, :3])
    point[0][:, :3] = normalized_scene_pc
    pc_embeds, pc_masks, point_obj = pc_encoder(point)
            
    text_embeds, text_masks = text_encoder(text_prompt, device)
    text_embeds = text_embeds.float()
    text_masks = text_masks.bool()

    out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks, pc_embeds, pc_masks)
    gen_traj = camera_vae.decode(out * cfg.vae_latent_scale)
    gen_intrinsics = make_intrinsics(gen_traj[:, :, -2:], width, height) 
    gen_traj = out_to_trajectory(gen_traj, scale, E0, device)

    gen_path = os.path.join(result_dir, f'{text_prompt[0].replace(' ', '_').replace('.', '')}.npz')
    np.savez(gen_path, poses=gen_traj[0].cpu().numpy(), intrinsics=gen_intrinsics[0].cpu().numpy())

    print("Generation path")
    print(gen_path)
    print("Text Prompt")
    print(text_prompt)

    if hasattr(model, "text_cross_attn_weight"):
        cross_attn_weight = model.text_cross_attn_weight[0]
        assert cross_attn_weight.shape[0] == out.shape[1] and cross_attn_weight.shape[1] == text_embeds.shape[1], f"{cross_attn_weight.shape[0]} == {out.shape[1]} , {cross_attn_weight.shape[1]} == {text_embeds.shape[1]}"

        print('Text max attention')
        text_weight = cross_attn_weight
        text_max, text_max_idx = text_weight.max(dim=-1)
        print(text_max, text_max_idx)

        text_tokens = text_encoder.tokenizer.tokenizer.tokenize(text_prompt[0])
        attention_path = os.path.join(result_dir, "text_attention.png")
        print(len(text_tokens))
        print(text_weight.mean(dim=0)[:len(text_tokens)].sum())
        attention_bar(tokens=text_tokens, attn=text_weight.mean(dim=0)[:len(text_tokens)], save_path=attention_path)

        print("Text attention saved in : ", attention_path)
        print(text_tokens)

    if hasattr(model, "point_cross_attn_weight"):
        cross_attn_weight = model.point_cross_attn_weight[0]
        assert cross_attn_weight.shape[0] == out.shape[1] and cross_attn_weight.shape[1] == pc_embeds.shape[1], f"{cross_attn_weight.shape[0]} == {out.shape[1]} , {cross_attn_weight.shape[1]} == {pc_embeds.shape[1]}"
        
        print('Point max attention')
        point_weight = cross_attn_weight
        point_max, point_max_idx = point_weight.max(dim=-1)
        print(point_max, point_max_idx)

        recovered, _, point_attention = pc_encoder.decode_with_attention(copy.deepcopy(point_obj), point_weight.mean(dim=0)[:, None], recover_origin_scale=True, num_iter=1) # (N_points, 1)
        print(pc_embeds.shape)
        print(recovered.shape)
        print(point[0].shape)

        point_attn_vis = attn_to_color(point_attention[:, 0].detach().cpu().numpy())
        point_attention_path = os.path.join(result_dir, "point_attention.pt")

        point[0] = point[0].cuda()
        pc_pos = point[0][:, :3]
        pc_pos *= scale[0]
        pc_pos = (pc_pos - E0[0:1, :3, 3]) @ E0[0, :3, :3]
        point[0][:, :3] = pc_pos
        point[0][:, 3:] = torch.tensor(point_attn_vis).to(point[0])
        torch.save(point[0].cpu(), point_attention_path)

        print('Point attention saved in : ', point_attention_path)

    if hasattr(model, "mm_cross_attn_weight"):
        point_text_weight = model.mm_cross_attn_weight[0]
        assert point_text_weight.shape[0] == pc_embeds.shape[1] and point_text_weight.shape[1] == text_embeds.shape[1], f"{point_text_weight.shape[0]} == {pc_embeds.shape[1]} , {point_text_weight.shape[1]} == {text_embeds.shape[1]}"
        total = 0
        text_attention_path =  os.path.join(result_dir, f"pc_text_attention.png")
        attention_bar(tokens=text_tokens, attn=point_text_weight.mean(dim=0)[:len(text_tokens)], save_path=text_attention_path)
        print(sum(point_text_weight[0]))

        print(text_attention_path)
        input()