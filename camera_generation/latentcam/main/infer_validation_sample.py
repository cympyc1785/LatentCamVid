"""Run the validation-style inference on ONE specific segment for a geo model, and save
predicted vs GT trajectory (.npz), metrics (.json), and a top-down (x-z) plot with GT.

env: VAL_CKPT=<best.pth>  VAL_OUT=<dir>  VAL_TAG=<model tag>  VAL_SEG=<batch>/<hash>/<seg>
     + `experiment=<name>` Hydra override (so geo/vae/scale match the trained model).
Mirrors train_latent_cam_dm.run_validation (geo_encode -> text -> DDPM sample -> vae decode
-> out_to_trajectory), on the single target segment.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hydra_cfg import load_cfg
cfg, _ = load_cfg('config')

from base import collate_fn
from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE
from models.t5 import T5EncoderModel
from models.camera_diffusion_model_latent import CameraDiffusionModel
from models.geo_encoder import build_geo_encoder
from utils.data_utils import out_to_trajectory, make_intrinsics, inverse_camera_matrix
from diffusers import DDPMScheduler, DDIMScheduler
from torch.utils.data import Subset, DataLoader

DEVICE = 'cuda:0'
CKPT = os.environ['VAL_CKPT']; OUT = os.environ['VAL_OUT']
TAG = os.environ['VAL_TAG']; SEG = os.environ['VAL_SEG']
torch.manual_seed(cfg.random_seed); np.random.seed(cfg.random_seed)


def seg_key(sid):
    bh, seg = sid.rsplit('_', 1); b, h = bh.split('_', 1); return f"{b}/{h}/{seg}"


def geo_encode(geo_encoder, data):
    images = data['images'].to(DEVICE)
    cam_token = None
    if getattr(cfg, 'geo_posed', False) and 'geo_c2w' in data:
        override = (data['avg_scale'].to(DEVICE)
                    if getattr(cfg, 'geo_lagernvs_skip_ctx_norm', False) else None)
        cam_token = geo_encoder.build_cam_token(
            data['geo_c2w'].to(DEVICE), data['geo_fxfycxcy'].to(DEVICE),
            data['geo_hw'].to(DEVICE), override_scale=override)
    return geo_encoder(images, cam_token)


def w2c_centers(traj_w2c):   # (T,4,4) w2c -> (T,3) camera centers in world
    c2w = np.linalg.inv(traj_w2c)
    return c2w[:, :3, 3]


def main():
    traj_len = (cfg.num_frames - 1) // 4 + 1
    os.makedirs(os.path.join(OUT, TAG), exist_ok=True)

    ds = CamDataset(cfg, 'train')
    idx = next((i for i, s in enumerate(ds.samples) if seg_key(s[4]) == SEG), None)
    assert idx is not None, f"segment {SEG} not found in dataset ({cfg.meta_csv})"
    loader = DataLoader(Subset(ds, [idx]), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    data = next(iter(loader))
    print(f"[{TAG}] target seg {SEG} -> sample idx {idx}, name {data['data_name'][0]}")

    geo_encoder = build_geo_encoder(cfg).to(DEVICE) if getattr(cfg, 'geo_encoder', None) else None
    text_encoder = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=DEVICE,
                                  checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
                                  tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim)
    model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    model = model.to(DEVICE).eval()
    camera_vae = CameraVAE(latent_dim=cfg.cam_dim).to(DEVICE)
    camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=DEVICE)); camera_vae.eval()

    sched_cls = DDPMScheduler if cfg.sampling_type == 'ddpm' else DDIMScheduler
    sch = sched_cls(num_train_timesteps=cfg.diffusion_max_step, prediction_type=cfg.prediction_type,
                    beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
                    set_alpha_to_one=cfg.set_alpha_to_one, steps_offset=cfg.steps_offset,
                    beta_start=cfg.beta_start, beta_end=cfg.beta_end)

    traj = data['cam_param'].to(DEVICE)
    E0 = data['first_extrinsic'].to(DEVICE); scale = data['avg_scale'].to(DEVICE)
    width = data['width'].to(DEVICE); height = data['height'].to(DEVICE); intrinsics = data['intrinsics'].to(DEVICE)
    text = data['text_prompt']

    geo_emb, geo_mask = (geo_encode(geo_encoder, data) if geo_encoder is not None else (None, None))
    text_emb, text_mask = text_encoder(text, DEVICE); text_emb = text_emb.float(); text_mask = text_mask.bool()

    with torch.no_grad():
        x_t = torch.randn(1, traj_len, cfg.cam_dim, device=DEVICE)
        sch.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=DEVICE)
        for t in sch.timesteps:
            ts = torch.full((1,), t, device=DEVICE)
            x_t = sch.step(model(x_t, ts.float(), text_emb, text_mask, geo_emb, geo_mask), t, x_t).prev_sample
        out = x_t
        traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
        loss_latent = F.mse_loss(out, traj_latents).item()
        traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)
        loss_traj = F.mse_loss(traj_pred, traj).item()

    # intrinsics from the camera params (before out_to_trajectory), exactly like run_validation
    ref_intr = make_intrinsics(traj[:, :, -2:], width, height, intrinsics).cpu().tolist()
    pred_intr = make_intrinsics(traj_pred[:, :, -2:], width, height, intrinsics).cpu().tolist()
    Wv = int(width.flatten()[0].item()); Hv = int(height.flatten()[0].item())

    # out_to_trajectory -> w2c; inverse -> c2w; flip y,z axes -> OpenGL c2w (monst3r/transforms convention)
    gt_w2c = out_to_trajectory(traj, scale, E0, DEVICE)
    pred_w2c = out_to_trajectory(traj_pred, scale, E0, DEVICE)
    m_ref = inverse_camera_matrix(gt_w2c); m_pred = inverse_camera_matrix(pred_w2c)
    m_ref[:, :, :3, 1:3] *= -1; m_pred[:, :, :3, 1:3] *= -1
    m_ref = m_ref.cpu().tolist(); m_pred = m_pred.cpu().tolist()

    def build_json(intr, mats):
        return {"w": Wv, "h": Hv,
                "fl_x": intr[0][0][0], "fl_y": intr[0][1][1],
                "cx": intr[0][0][2], "cy": intr[0][1][2],
                "frames": [{"transform_matrix": mats[0][k], "monst3r_im_id": k + 1}
                           for k in range(len(mats[0]))]}

    tag_dir = os.path.join(OUT, TAG)
    base = SEG.replace('/', '_')
    json.dump({"Concise Interaction": text[0]}, open(os.path.join(tag_dir, f"{base}_caption.json"), "w"), indent=4)
    json.dump(build_json(ref_intr, m_ref), open(os.path.join(tag_dir, f"{base}_transforms_ref.json"), "w"), indent=4)
    json.dump(build_json(pred_intr, m_pred), open(os.path.join(tag_dir, f"{base}_transforms_pred.json"), "w"), indent=4)

    # top-down from the c2w matrices (camera center = transform_matrix[:3,3])
    ref_c2w = np.array([f["transform_matrix"] for f in build_json(ref_intr, m_ref)["frames"]])
    gc = ref_c2w[:, :3, 3]
    pc = np.array([f["transform_matrix"] for f in build_json(pred_intr, m_pred)["frames"]])[:, :3, 3]
    pos_err = np.linalg.norm(pc - gc, axis=1)
    metrics = {"model": TAG, "segment": SEG, "experiment": cfg.exp_name,
               "loss_latent": loss_latent, "loss_traj": loss_traj,
               "pos_rmse": float(np.sqrt((pos_err ** 2).mean())),
               "pos_err_mean": float(pos_err.mean()), "pos_err_max": float(pos_err.max()),
               "ckpt": CKPT, "text": text[0]}
    json.dump(metrics, open(os.path.join(tag_dir, f"{base}_metrics.json"), "w"), indent=2)
    print(f"[{TAG}] loss_latent={loss_latent:.4f} loss_traj={loss_traj:.4f} pos_rmse={metrics['pos_rmse']:.4f}")

    # true top-down: look DOWN the world up-axis (from the +up side, looking -up). up-axis =
    # dominant component of mean camera-up (OpenGL c2w col1); drop it, plot the two ground-plane
    # axes. Horizontal sign is chosen right-handed so R x U = +world-up (else it mirrors into a
    # "down-top" bottom-up view): hsign = up_sign * levi_civita(ga0, ga1, up_axis). (latentcam
    # up=-X -> plots (-Y, Z); plain x-z was a front/back view, not top-down.)
    up_vec = ref_c2w[:, :3, 1].mean(0)
    up_axis = int(np.argmax(np.abs(up_vec)))                       # 0=X 1=Y 2=Z
    up_sign = 1.0 if up_vec[up_axis] >= 0 else -1.0
    ga = [a for a in (0, 1, 2) if a != up_axis]                    # ground-plane axes
    parity = 1.0 if (ga[0], ga[1], up_axis) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)} else -1.0
    hsign = up_sign * parity
    an = "XYZ"
    hlab = ("-" if hsign < 0 else "") + an[ga[0]]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(hsign * gc[:, ga[0]], gc[:, ga[1]], '-o', ms=3, lw=1.4, c='tab:blue', label='GT')
    ax.plot(hsign * pc[:, ga[0]], pc[:, ga[1]], '-x', ms=4, lw=1.4, c='tab:red', label='pred')
    ax.scatter(hsign * gc[0, ga[0]], gc[0, ga[1]], c='k', s=70, marker='*', zorder=5, label='start')
    ax.set_aspect('equal', 'datalim'); ax.legend(fontsize=9)
    ax.set_xlabel(hlab); ax.set_ylabel(an[ga[1]])
    ax.set_title(f"{TAG}  {SEG}\nloss_traj={loss_traj:.4f}  pos_rmse={metrics['pos_rmse']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(tag_dir, f"{base}_topdown.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[{TAG}] saved transforms_ref/pred.json + caption.json + metrics.json + topdown.png -> {tag_dir}")


if __name__ == '__main__':
    main()
