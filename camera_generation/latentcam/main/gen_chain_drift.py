"""Chunk-chained camera generation + drift, using the text-only baseline model.

For each scene, generate each 49-frame segment ("chunk") from its text with the trained
model, then CHAIN: place chunk i so its FIRST camera coincides with chunk (i-1)'s LAST
camera. Chunk 0 is anchored at the GT scene-start pose. -> one long generated sequence per
scene. Saved (OpenGL c2w transforms.json) as {scene}_transforms_pred.json (generated) and
{scene}_transforms_ref.json (GT full scene, same world) so scripts/viser_val_cameras.py can
show it. Drift = per-frame gen-vs-GT camera-center distance / scene scale.

Per-chunk scale = the GT segment's target_cam avg_scale (metric per chunk). Chaining is done
in c2w; each chunk's start is snapped exactly to the running anchor.

Run: CUDA_VISIBLE_DEVICES=1 LATENTCAM_CONFIG=config_textonly PYTHONPATH=..:. \
     python gen_chain_drift.py --ckpt results/20260719_210144_dl3dv_textonly/ckpts/best.pth \
     --n-scenes 20 --out-dir results/20260719_210144_dl3dv_textonly/chain_drift
"""
import os, importlib, json, argparse
import numpy as np, torch
from collections import defaultdict

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config_textonly', overrides=[])
from diffusers import DDIMScheduler, DDPMScheduler
from models.camera_diffusion_model_latent import CameraDiffusionModel
from models.vae_intr_large import CameraVAE
from models.t5 import T5EncoderModel
from dataset_dl3dv import CamDataset
from utils.data_utils import out_to_trajectory, inverse_camera_matrix

_GL_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV c2w -> OpenGL c2w (save convention)


@torch.no_grad()
def sample(model, sch, traj_len, temb, tmask, device):
    x = torch.randn(1, traj_len, cfg.cam_dim, device=device)
    sch.set_timesteps(cfg.diffusion_inference_step, device=device)
    for t in sch.timesteps:
        ts = torch.full((1,), t, device=device)
        x = sch.step(model(x, ts.float(), temb, tmask, None, None), t, x).prev_sample
    return x


def to_transforms(c2w_gl, fx, fy, w, h):
    return {"w": w, "h": h, "fl_x": fx, "fl_y": fy, "cx": w / 2, "cy": h / 2,
            "frames": [{"transform_matrix": c2w_gl[i].tolist(), "monst3r_im_id": i + 1}
                       for i in range(len(c2w_gl))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n-scenes', type=int, default=20)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda'
    torch.set_grad_enabled(False)

    sch = (DDIMScheduler if cfg.sampling_type == 'ddim' else DDPMScheduler)(
        num_train_timesteps=cfg.diffusion_max_step, prediction_type=cfg.prediction_type,
        beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
        set_alpha_to_one=cfg.set_alpha_to_one, steps_offset=cfg.steps_offset,
        beta_start=cfg.beta_start, beta_end=cfg.beta_end)
    text_encoder = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
                                  checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
                                  tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim).to(device)
    sd = torch.load(args.ckpt, map_location=device); model.load_state_dict(sd.get('model', sd)); model.eval()
    vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
    vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device)); vae.eval()

    cfg.max_scenes = args.n_scenes           # limit dataset load to the scenes we need
    ds = CamDataset(cfg, 'train')
    by_scene = defaultdict(list)
    for idx, (sidx, s, e, cap, name) in enumerate(ds.samples):
        by_scene[sidx].append((s, e, idx))
    scene_ids = sorted(by_scene.keys())[:args.n_scenes]

    summary = []
    for sc in scene_ids:
        segs = sorted(by_scene[sc], key=lambda x: x[0])
        gt_w2c = ds.extrinsics_list[sc].numpy().astype(np.float64)     # (Nfull,4,4) OpenCV w2c
        gt_c2w = np.linalg.inv(gt_w2c)
        scene_scale = float(np.linalg.norm(gt_c2w[:, :3, 3] - gt_c2w[0, :3, 3], axis=1).mean() + 1e-6)
        W = gt_c2w[segs[0][0]].copy()                                  # anchor = GT first-seg start (c2w)
        chain_c2w, chain_gtidx = [], []
        for si, (s, e, idx) in enumerate(segs):
            d = ds[idx]
            temb, tmask = text_encoder([d['text_prompt']], device)
            out = sample(model, sch, cfg.num_cam, temb.float(), tmask.bool(), device)
            traj_pred = vae.decode(out * cfg.vae_latent_scale) if cfg.use_vae else out   # (1,49,11)
            scale = d['avg_scale'].to(device).view(1, 1)
            E0 = torch.eye(4, device=device).unsqueeze(0)
            rel_w2c = out_to_trajectory(traj_pred, scale, E0, device)[0].cpu().numpy()    # (49,4,4)
            rel_c2w = np.linalg.inv(rel_w2c)                            # (49,4,4) frame0~=I
            T = W @ np.linalg.inv(rel_c2w[0])                          # snap chunk start to anchor exactly
            world_c2w = T @ rel_c2w                                     # (49,4,4)
            sel = list(range(1, len(world_c2w))) if si > 0 else list(range(len(world_c2w)))  # drop dup seam
            for k in sel:
                chain_c2w.append(world_c2w[k])
            # GT frame indices this segment covers (even-subsampled to 49, like the dataset)
            gsel = np.linspace(s, e - 1, len(world_c2w)).round().astype(int)
            for k in sel:
                chain_gtidx.append(int(gsel[k]))
            W = world_c2w[-1].copy()

        chain_c2w = np.stack(chain_c2w)                               # (M,4,4) OpenCV c2w
        # drift vs GT (same world; anchored at GT seg0 start)
        gt_pos = gt_c2w[np.array(chain_gtidx), :3, 3]
        gen_pos = chain_c2w[:, :3, 3]
        drift = np.linalg.norm(gen_pos - gt_pos, axis=1) / scene_scale
        name = ds.samples[segs[0][2]][4].rsplit('_', 1)[0]           # scene data_name (drop seg suffix)

        # intrinsics from GT (for saving)
        K = ds.intrinsics_list[sc][0].numpy(); h, w = ds.hw_list[sc]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        json.dump(to_transforms(chain_c2w @ _GL_FLIP, fx, fy, float(w), float(h)),
                  open(os.path.join(args.out_dir, f"{name}_transforms_pred.json"), 'w'))
        json.dump(to_transforms(gt_c2w @ _GL_FLIP, fx, fy, float(w), float(h)),
                  open(os.path.join(args.out_dir, f"{name}_transforms_ref.json"), 'w'))
        json.dump({"Concise Interaction": f"{len(segs)} chained segments"},
                  open(os.path.join(args.out_dir, f"{name}_caption.json"), 'w'))
        summary.append((name, len(segs), len(chain_c2w), float(drift.mean()), float(drift.max()), float(drift[-1])))
        print(f"{name}: {len(segs)} segs, {len(chain_c2w)} frames | drift(/scale) mean={drift.mean():.3f} "
              f"max={drift.max():.3f} final={drift[-1]:.3f}", flush=True)

    print("\n=== drift summary (position error / scene_scale) ===")
    dm = np.array([s[3] for s in summary]); dmax = np.array([s[4] for s in summary]); dfin = np.array([s[5] for s in summary])
    print(f"scenes={len(summary)} | mean_drift avg={dm.mean():.3f} | max_drift avg={dmax.mean():.3f} | final_drift avg={dfin.mean():.3f}")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
