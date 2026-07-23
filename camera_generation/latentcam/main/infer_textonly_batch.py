"""Run the TEXT-ONLY model on the worldtraj validation targets (same GT), so text-only vs
worldtraj can be compared PER-PAIR on identical targets (both target_cam norm; diff = geo).

No CamDataset load: inputs are reconstructed from the saved worldtraj test files
(<tid>_transforms_ref.json = denormalized-world OpenGL c2w; <tid>_caption.json = text):
  ref -> OpenCV w2c -> E0, target_cam avg_scale, intrinsics ; text from caption.
Then text-only DDPM sample -> vae decode -> out_to_trajectory -> OpenGL c2w pred.

env: TO_CKPT (textonly ckpt) ; reads worldtraj test dir ; writes textonly preds dir.
"""
import os, json, glob
import numpy as np
import torch
import torch.nn.functional as F

from hydra_cfg import load_cfg
cfg, _ = load_cfg('config', overrides=['experiment=textonly'])

from models.vae_intr_large import CameraVAE
from models.t5 import T5EncoderModel
from models.camera_diffusion_model_latent import CameraDiffusionModel
from utils.data_utils import out_to_trajectory, make_intrinsics, inverse_camera_matrix
from diffusers import DDPMScheduler, DDIMScheduler

DEVICE = 'cuda:0'
CKPT = os.environ['TO_CKPT']
WT = '/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results/20260721_213927_dl3dv_geo_worldtraj/test'
OUT = '/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results/compare/text-only_vs_worldtraj/textonly_preds'
os.makedirs(OUT, exist_ok=True)
GL2CV = torch.diag(torch.tensor([1., -1., -1., 1.])).float()


def load_ref(p):
    d = json.load(open(p))
    c2w = torch.tensor([f['transform_matrix'] for f in d['frames']], dtype=torch.float32)
    return c2w, d


def main():
    traj_len = (cfg.num_frames - 1) // 4 + 1
    text_encoder = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=DEVICE,
                                  checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
                                  tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim)
    model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    model = model.to(DEVICE).eval()
    vae = CameraVAE(latent_dim=cfg.cam_dim).to(DEVICE)
    vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=DEVICE)); vae.eval()

    sched_cls = DDPMScheduler if cfg.sampling_type == 'ddpm' else DDIMScheduler
    sch = sched_cls(num_train_timesteps=cfg.diffusion_max_step, prediction_type=cfg.prediction_type,
                    beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
                    set_alpha_to_one=cfg.set_alpha_to_one, steps_offset=cfg.steps_offset,
                    beta_start=cfg.beta_start, beta_end=cfg.beta_end)

    refs = sorted(glob.glob(os.path.join(WT, '*_transforms_ref.json')))
    print(f"{len(refs)} targets")
    for n, rf in enumerate(refs):
        tid = os.path.basename(rf)[:-len('_transforms_ref.json')]
        c2w_gl, meta = load_ref(rf)                       # (T,4,4) OpenGL c2w world
        w2c = torch.linalg.inv(c2w_gl @ GL2CV)            # OpenCV w2c
        E0 = w2c[0].clone()
        e0i = torch.linalg.inv(w2c[0]); nr = w2c @ e0i[None]
        centers = torch.linalg.inv(nr)[:, :3, 3]
        avg_scale = centers.norm(dim=-1).mean().clamp(min=1e-5).unsqueeze(0)   # target_cam
        T = w2c.shape[0]
        W = float(meta['w']); H = float(meta['h'])
        K = torch.tensor([[meta['fl_x'], 0, meta['cx']], [0, meta['fl_y'], meta['cy']], [0, 0, 1.]]).float()
        intr = K[None].repeat(T, 1, 1)
        cap = json.load(open(rf.replace('_transforms_ref.json', '_caption.json')))
        text = [cap.get('Concise Interaction') or list(cap.values())[0]]

        E0 = E0[None].to(DEVICE); scale = avg_scale[None].to(DEVICE)
        width = torch.full((1, T), W).to(DEVICE); height = torch.full((1, T), H).to(DEVICE)
        intrinsics = intr[None].to(DEVICE)
        text_emb, text_mask = text_encoder(text, DEVICE); text_emb = text_emb.float(); text_mask = text_mask.bool()

        torch.manual_seed(cfg.random_seed + n)
        with torch.no_grad():
            x_t = torch.randn(1, traj_len, cfg.cam_dim, device=DEVICE)
            sch.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=DEVICE)
            for t in sch.timesteps:
                ts = torch.full((1,), t, device=DEVICE)
                x_t = sch.step(model(x_t, ts.float(), text_emb, text_mask, None, None), t, x_t).prev_sample
            traj_pred = vae.decode(x_t * cfg.vae_latent_scale)

        pred_intr = make_intrinsics(traj_pred[:, :, -2:], width, height, intrinsics).cpu().tolist()
        pred_w2c = out_to_trajectory(traj_pred, scale, E0, DEVICE)
        m = inverse_camera_matrix(pred_w2c); m[:, :, :3, 1:3] *= -1; m = m.cpu().tolist()
        out_json = {"w": int(W), "h": int(H), "fl_x": pred_intr[0][0][0], "fl_y": pred_intr[0][1][1],
                    "cx": pred_intr[0][0][2], "cy": pred_intr[0][1][2],
                    "frames": [{"transform_matrix": m[0][k], "monst3r_im_id": k + 1} for k in range(len(m[0]))]}
        json.dump(out_json, open(os.path.join(OUT, f"{tid}_transforms_pred.json"), 'w'))
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/{len(refs)}")
    print("saved textonly preds ->", OUT)


if __name__ == '__main__':
    main()
