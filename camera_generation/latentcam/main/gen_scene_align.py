"""Per-segment inference placed TWO ways, plotted with GT, top-down, 3 scenes -> one image.
Text-only baseline (config_textonly, vae_intr_large, DDPM).

For each segment the model predicts a 49-frame relative trajectory rel_c2w (frame0 ~ I). Then:
  (A) per-seg aligned : each segment placed so its FIRST camera == GT camera at that segment's
                        start (anchor RESET to GT every segment) -> shows per-segment SHAPE
                        without accumulated drift.
  (B) chained         : segment i's first camera snapped onto segment (i-1)'s LAST camera
                        (running anchor) -> accumulates drift. (== gen_chain_drift)
GT = full scene. One row of 3 subplots.

Run: CUDA_VISIBLE_DEVICES=3 LATENTCAM_CONFIG=config_textonly \
     PYTHONPATH=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam \
     python gen_scene_align.py --ckpt <best.pth> --out /tmp/align3.png
"""
import os, importlib, argparse
import numpy as np, torch
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config_textonly', overrides=[])
from diffusers import DDIMScheduler, DDPMScheduler
from models.camera_diffusion_model_latent import CameraDiffusionModel
from models.vae_intr_large import CameraVAE
from models.t5 import T5EncoderModel
from dataset_dl3dv import CamDataset
from utils.data_utils import out_to_trajectory

TARGETS = [
    "1K_a4c20f668ce179db83200fc38f610d2e0aae2633e4462084397f7c390f07cb97",
    "1K_aeea987ca3dea5b7b55abeef9d9807629a091fbc0b78db65041ab4c71d9298b4",
    "1K_97f72cff0be96647eeb2fe17ac49752c739af5d1cda656b52e83917a4b2bc17d",
]


@torch.no_grad()
def sample(model, sch, traj_len, temb, tmask, device):
    x = torch.randn(1, traj_len, cfg.cam_dim, device=device)
    sch.set_timesteps(cfg.diffusion_inference_step, device=device)
    for t in sch.timesteps:
        ts = torch.full((1,), t, device=device)
        x = sch.step(model(x, ts.float(), temb, tmask, None, None), t, x).prev_sample
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--max-scenes', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/tmp/align3.png')
    ap.add_argument('--auto-best', type=int, default=0,
                    help='scan all loaded scenes, pick the N with lowest mean per-segment err '
                         '(0 = use the fixed TARGETS instead)')
    args = ap.parse_args()
    device = 'cuda'; torch.set_grad_enabled(False)

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

    cfg.max_scenes = args.max_scenes
    ds = CamDataset(cfg, 'train')
    by_scene = defaultdict(list)
    name_of = {}
    for idx, (sidx, s, e, cap, nm) in enumerate(ds.samples):
        by_scene[sidx].append((s, e, idx)); name_of[sidx] = nm.rsplit('_', 1)[0]

    def gen_scene(sc):
        torch.manual_seed(args.seed)
        segs = sorted(by_scene[sc], key=lambda x: x[0])
        gt_c2w = np.linalg.inv(ds.extrinsics_list[sc].numpy().astype(np.float64))
        gtc = gt_c2w[:, :3, 3]
        ss = float(np.linalg.norm(gtc - gtc[0], axis=1).mean() + 1e-6)
        Wchain = gt_c2w[segs[0][0]].copy()
        aligned_segs, chain_c2w, chain_gt, seams, seg_records, seg_errs = [], [], [], [], [], []
        for si, (s, e, idx) in enumerate(segs):
            d = ds[idx]
            temb, tmask = text_encoder([d['text_prompt']], device)
            out = sample(model, sch, cfg.num_cam, temb.float(), tmask.bool(), device)
            traj = vae.decode(out * cfg.vae_latent_scale) if cfg.use_vae else out
            scale = d['avg_scale'].to(device).view(1, 1)
            E0 = torch.eye(4, device=device).unsqueeze(0)
            rel_c2w = np.linalg.inv(out_to_trajectory(traj, scale, E0, device)[0].cpu().numpy())
            Ta = gt_c2w[s] @ np.linalg.inv(rel_c2w[0])       # (A) reset to GT segment start
            aln = Ta @ rel_c2w
            gt_seg = gt_c2w[np.linspace(s, e - 1, len(aln)).round().astype(int)]
            ssg = float(np.linalg.norm(gt_seg[:, :3, 3] - gt_seg[0, :3, 3], axis=1).mean() + 1e-6)
            seg_errs.append(float(np.linalg.norm(aln[:, :3, 3] - gt_seg[:, :3, 3], axis=1).mean() / ssg))
            aligned_segs.append(aln); seg_records.append((s, e, aln, gt_seg))
            Tc = Wchain @ np.linalg.inv(rel_c2w[0])          # (B) chained (running anchor)
            world = Tc @ rel_c2w
            sel = range(1, len(world)) if si > 0 else range(len(world))
            gsel = np.linspace(s, e - 1, len(world)).round().astype(int)
            if si > 0:
                seams.append(len(chain_c2w))
            for k in sel:
                chain_c2w.append(world[k]); chain_gt.append(int(gsel[k]))
            Wchain = world[-1].copy()
        chain_c2w = np.stack(chain_c2w)
        d_chain = np.linalg.norm(chain_c2w[:, :3, 3] - gtc[np.array(chain_gt)], axis=1) / ss
        return dict(name=name_of[sc][3:11], segs=segs, gt_c2w=gt_c2w, aligned_segs=aligned_segs,
                    chain_c2w=chain_c2w, seams=seams, seg_records=seg_records,
                    mean_seg_err=float(np.mean(seg_errs)), d_chain=d_chain)

    # candidate scenes: scan all loaded (auto-best) or just the TARGETS
    if args.auto_best > 0:
        cand = list(by_scene.keys())
        print(f"scanning {len(cand)} scenes for best mean per-segment err ...")
        results = [gen_scene(sc) for sc in cand]
        results.sort(key=lambda r: r['mean_seg_err'])
        for r in results:
            print(f"  {r['name']}: mean_seg_err={r['mean_seg_err']:.3f} ({len(r['segs'])} segs)")
        results = results[:args.auto_best]
    else:
        want = {t: sc for sc in by_scene for t in TARGETS if name_of[sc] == t}
        results = [gen_scene(want[t]) for t in TARGETS if t in want]
    print(f"selected {len(results)} scene(s): {[r['name'] for r in results]}")

    # --- 3-column layout per scene: GT | per-seg aligned | chained ---
    nrow = len(results)
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 4.7 * nrow), squeeze=False)
    for r, res in enumerate(results):
        gtc = res['gt_c2w'][:, :3, 3]
        ax0, ax1 = np.argsort(gtc.var(0))[-2:]; L = {0: 'X', 1: 'Y', 2: 'Z'}
        start = res['segs'][0][0]
        for c in range(3):
            a = axes[r][c]
            a.plot(gtc[:, ax0], gtc[:, ax1], c='tab:blue', lw=1.5,
                   alpha=(0.9 if c == 0 else 0.35), label='GT')
            a.scatter([gtc[start, ax0]], [gtc[start, ax1]], c='k', s=40, marker='s', zorder=7)
            if c == 1:
                for j, seg in enumerate(res['aligned_segs']):
                    p = seg[:, :3, 3]
                    a.plot(p[:, ax0], p[:, ax1], c='tab:red', lw=1.4, alpha=0.9,
                           label='per-seg aligned' if j == 0 else None)
                    a.scatter([p[0, ax0]], [p[0, ax1]], c='tab:red', s=22, marker='o', zorder=5)
            if c == 2:
                cp = res['chain_c2w'][:, :3, 3]
                a.plot(cp[:, ax0], cp[:, ax1], c='tab:green', lw=1.5, label='chained')
                for sm in res['seams']:
                    a.scatter([cp[sm, ax0]], [cp[sm, ax1]], c='orange', s=40, marker='x', zorder=6)
                a.scatter([], [], c='orange', marker='x', label='seams')
            a.set_aspect('equal', 'datalim'); a.set_xticks([]); a.set_yticks([])
            a.legend(fontsize=8, loc='best')
        axes[r][0].set_ylabel(f"{res['name']}\nmean_seg_err={res['mean_seg_err']:.2f}", fontsize=10)
        axes[r][0].set_title('GT', fontsize=11)
        axes[r][1].set_title(f"per-seg aligned  (mean_seg_err={res['mean_seg_err']:.2f})", fontsize=11)
        axes[r][2].set_title(f"chained  (drift mean={res['d_chain'].mean():.2f} "
                             f"final={res['d_chain'][-1]:.2f})", fontsize=11)
    fig.suptitle("Text-only baseline, best-average scenes: GT | per-seg aligned | chained (top-down)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
