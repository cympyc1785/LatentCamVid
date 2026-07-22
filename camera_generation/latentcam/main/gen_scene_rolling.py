"""Generate a FULL scene end-to-end with the ROLLING (per-token flow) model.

For each text-prompt segment of a scene, generate 49 frames with the rolling model (mode:
full-sequence or chunk_ar, in the causal-VAE latent), then CHAIN segments (chunk i's first
camera snapped onto chunk (i-1)'s last camera). chunk 0 anchored at the GT scene-start pose.
-> one long generated trajectory per scene. Saves transforms_pred/ref.json (viser) + a
top-down pred-vs-GT plot. Drift = per-frame gen-vs-GT center distance / scene scale
(same metric as gen_chain_drift.py, so comparable to the full-seq baseline).

Run: CUDA_VISIBLE_DEVICES=3 LATENTCAM_CONFIG=config_rolling \
     PYTHONPATH=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam \
     python gen_scene_rolling.py --ckpt results/20260721_032652_dl3dv_rolling/ckpts/best.pth \
     --n-scenes 1 --mode full --out-dir /tmp/roll_scene
"""
import os, importlib, json, argparse
import numpy as np, torch
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config_rolling', overrides=[])
from models.camera_diffusion_model_latent import CameraDiffusionModel
from models.t5 import T5EncoderModel
from dataset_dl3dv import CamDataset
from utils.data_utils import out_to_trajectory
import rolling_sampler as RS

_GL_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenCV c2w -> OpenGL c2w (save convention)


def to_transforms(c2w_gl, fx, fy, w, h):
    return {"w": w, "h": h, "fl_x": fx, "fl_y": fy, "cx": w / 2, "cy": h / 2,
            "frames": [{"transform_matrix": c2w_gl[i].tolist(), "monst3r_im_id": i + 1}
                       for i in range(len(c2w_gl))]}


@torch.no_grad()
def gen_latent(model, temb, tmask, mode, W, D, steps, device):
    if mode == 'chunk_ar':
        return RS.gen_chunk_ar(model, temb, tmask, W, D, max(4, steps // W), device)
    return RS.gen_full(model, temb, tmask, W, D, steps, device)


def top_down_plot(gen_c2w, gt_c2w, gtidx, seams, name, drift, out_png):
    gp, gtp = gen_c2w[:, :3, 3], gt_c2w[np.array(gtidx), :3, 3]
    allc = np.concatenate([gp, gt_c2w[:, :3, 3]], 0)
    ax0, ax1 = np.argsort(allc.var(0))[-2:]
    L = {0: 'X', 1: 'Y', 2: 'Z'}
    fig, a = plt.subplots(figsize=(7, 6))
    a.plot(gt_c2w[:, ax0], gt_c2w[:, ax1], c='tab:blue', lw=1.2, alpha=0.6, label='GT full scene')
    a.plot(gp[:, ax0], gp[:, ax1], c='tab:red', lw=1.4, label='generated')
    a.scatter([gp[0, ax0]], [gp[0, ax1]], c='k', s=50, marker='s', label='start (anchor)', zorder=5)
    for sm in seams:
        a.scatter([gp[sm, ax0]], [gp[sm, ax1]], c='orange', s=30, marker='x', zorder=5)
    if seams:
        a.scatter([], [], c='orange', marker='x', label='segment seams')
    a.set_aspect('equal', 'datalim'); a.set_xlabel(L[ax0]); a.set_ylabel(L[ax1])
    a.set_title(f"{name}\ndrift/scale mean={drift.mean():.3f} "
                f"max={drift.max():.3f} final={drift[-1]:.3f}", fontsize=10)
    a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n-scenes', type=int, default=1)
    ap.add_argument('--scene-skip', type=int, default=0, help='skip the first N scenes (pick a different one)')
    ap.add_argument('--mode', default='full', choices=['full', 'chunk_ar', 'rolling'])
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda'; torch.set_grad_enabled(False)
    W, D = cfg.num_cam, cfg.cam_dim

    text_encoder = T5EncoderModel(text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
                                  checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
                                  tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim).to(device)
    sd = torch.load(args.ckpt, map_location=device); model.load_state_dict(sd.get('model', sd)); model.eval()
    vae = RS._load_causal_vae(device)

    cfg.max_scenes = args.scene_skip + args.n_scenes
    ds = CamDataset(cfg, 'train')
    by_scene = defaultdict(list)
    for idx, (sidx, s, e, cap, name) in enumerate(ds.samples):
        by_scene[sidx].append((s, e, idx))
    scene_ids = sorted(by_scene.keys())[args.scene_skip:args.scene_skip + args.n_scenes]

    summary = []
    for sc in scene_ids:
        segs = sorted(by_scene[sc], key=lambda x: x[0])
        gt_w2c = ds.extrinsics_list[sc].numpy().astype(np.float64)
        gt_c2w = np.linalg.inv(gt_w2c)
        scene_scale = float(np.linalg.norm(gt_c2w[:, :3, 3] - gt_c2w[0, :3, 3], axis=1).mean() + 1e-6)
        Wanchor = gt_c2w[segs[0][0]].copy()

        if args.mode == 'rolling':
            # CONTINUOUS rolling over the whole scene: one sliding window, text switches per
            # emitted token's segment, decode ONCE -> seam-free. Single scene scale (mean of
            # per-segment avg_scale) since one continuous trajectory can't carry per-seg scales.
            seg_embs, scales = [], []
            for (s, e, idx) in segs:
                d = ds[idx]
                te, tm = text_encoder([d['text_prompt']], device)
                seg_embs.append((te.float(), tm.bool())); scales.append(float(d['avg_scale']))
            scale_val = float(np.mean(scales))
            z = RS.gen_rolling_scene(model, seg_embs, [W] * len(segs), W, D, device)   # (1,total,D)
            traj_pred = vae.decode(z * cfg.vae_latent_scale)                            # (1,L,11)
            scale = torch.tensor([[scale_val]], device=device)
            E0 = torch.eye(4, device=device).unsqueeze(0)
            rel_w2c = out_to_trajectory(traj_pred, scale, E0, device)[0].cpu().numpy()
            rel_c2w = np.linalg.inv(rel_w2c)
            world_c2w = (Wanchor @ np.linalg.inv(rel_c2w[0])) @ rel_c2w
            chain_c2w = world_c2w
            gsel = np.linspace(segs[0][0], segs[-1][1] - 1, len(world_c2w)).round().astype(int)
            chain_gtidx = list(gsel)
            seams = []                                                                 # no seams
        else:
          chain_c2w, chain_gtidx, seams = [], [], []
          for si, (s, e, idx) in enumerate(segs):
            d = ds[idx]
            temb, tmask = text_encoder([d['text_prompt']], device)
            z = gen_latent(model, temb.float(), tmask.bool(), args.mode, W, D, args.steps, device)
            traj_pred = vae.decode(z * cfg.vae_latent_scale)                 # (1,49,11)
            scale = d['avg_scale'].to(device).view(1, 1)
            E0 = torch.eye(4, device=device).unsqueeze(0)
            rel_w2c = out_to_trajectory(traj_pred, scale, E0, device)[0].cpu().numpy()
            rel_c2w = np.linalg.inv(rel_w2c)
            T = Wanchor @ np.linalg.inv(rel_c2w[0])
            world_c2w = T @ rel_c2w
            sel = list(range(1, len(world_c2w))) if si > 0 else list(range(len(world_c2w)))
            if si > 0:
                seams.append(len(chain_c2w))
            gsel = np.linspace(s, e - 1, len(world_c2w)).round().astype(int)
            for k in sel:
                chain_c2w.append(world_c2w[k]); chain_gtidx.append(int(gsel[k]))
            Wanchor = world_c2w[-1].copy()

        chain_c2w = np.stack(chain_c2w)
        gt_pos = gt_c2w[np.array(chain_gtidx), :3, 3]; gen_pos = chain_c2w[:, :3, 3]
        drift = np.linalg.norm(gen_pos - gt_pos, axis=1) / scene_scale
        name = ds.samples[segs[0][2]][4].rsplit('_', 1)[0]

        K = ds.intrinsics_list[sc][0].numpy(); h, w = ds.hw_list[sc]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        tag = f"{name}_{args.mode}"
        json.dump(to_transforms(chain_c2w @ _GL_FLIP, fx, fy, float(w), float(h)),
                  open(os.path.join(args.out_dir, f"{tag}_transforms_pred.json"), 'w'))
        json.dump(to_transforms(gt_c2w @ _GL_FLIP, fx, fy, float(w), float(h)),
                  open(os.path.join(args.out_dir, f"{tag}_transforms_ref.json"), 'w'))
        json.dump({"Concise Interaction": f"{len(segs)} chained segments ({args.mode})"},
                  open(os.path.join(args.out_dir, f"{tag}_caption.json"), 'w'))
        top_down_plot(chain_c2w, gt_c2w, chain_gtidx, seams, tag, drift,
                      os.path.join(args.out_dir, f"{tag}_topdown.png"))
        summary.append((tag, len(segs), len(chain_c2w), float(drift.mean()), float(drift.max()), float(drift[-1])))
        print(f"{tag}: {len(segs)} segs, {len(chain_c2w)} frames | drift(/scale) mean={drift.mean():.3f} "
              f"max={drift.max():.3f} final={drift[-1]:.3f}", flush=True)

    print(f"\nout-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
