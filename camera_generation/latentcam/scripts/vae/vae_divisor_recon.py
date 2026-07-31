"""Does the FROZEN camera VAE still work under each candidate normalization divisor?

The divisor comparison (scripts/data/norm_divisor_compare.py) scores divisors on distribution
shape. This adds the constraint that actually binds in practice: cfg.vae_ckpt_path is frozen and
was trained on ONE normalization, so a divisor that shifts the trans-channel scale away from what
the VAE saw degrades reconstruction -- and every trajectory the diffusion model produces has to
survive that decode.

For the same segments, cam_param is rebuilt under each divisor (identical to
dataset_dl3dv.__getitem__: rel = w2c_t @ inv(w2c_s), trans /= D, rot6d, intr per cfg.intr_norm)
and pushed through encode->decode. Reported per divisor:
  in_trans_std     std of the 3 translation channels (what the VAE is fed)
  lat_std_raw      latent std BEFORE /cfg.vae_latent_scale
  lat_std_scaled   latent std AFTER  /cfg.vae_latent_scale  <- the diffusion target's std
  vae_scale_for_1  the vae_latent_scale value that would make that exactly 1.0
  rec_trans_norm   recon MAE on the trans channels, in NORMALIZED units
  rec_trans_world  the same MAE * D  -> comparable ACROSS divisors (this is the one that matters)
  rec_rot6d        recon MAE on the 6 rotation channels (divisor-independent sanity check)

env: N (segments, default 400), CACHE_EXP (default geo_worldtraj), SPLIT (train), SEED (0)
out -> results/compare/norm_divisor_compare/vae_recon.json  (+ printed table)
"""
import os, sys, json, random
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

N = int(os.environ.get("N", "400"))
EXP = os.environ.get("CACHE_EXP", "geo_worldtraj")
SPLIT = os.environ.get("SPLIT", "train")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.path.join(HERE, "results", "compare", "norm_divisor_compare")
os.makedirs(OUT, exist_ok=True)

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE

device = "cuda"
vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
vae.eval()
print(f"VAE {cfg.vae_ckpt_path} latent_dim={cfg.cam_dim} vae_latent_scale={cfg.vae_latent_scale}")

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
ds = CamDataset(cfg, SPLIT)
T = cfg.num_frames


def even(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def longer_side(N_all, s, e):
    return list(range(0, s)) if s >= (N_all - e) else list(range(e, N_all))


def windows(side, T):
    ch = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]
    if not ch and len(side) >= 2:
        ch = [side]
    return ch


NAMES = ["geo_lagernvs", "ctx_longer_135max", "ctx_side_135max", "context_longer",
         "first_farthest_135*", "cam_dist_mean*", "avg_scale(excl)"]
acc = {n: dict(D=[], instd=[], lat=[], rt=[], rtw=[], rr=[]) for n in NAMES}

order = list(range(len(ds.samples)))
random.shuffle(order)
seen, done = set(), 0
for idx in order:
    if done >= N:
        break
    scene_idx, s, e, caption, data_name = ds.samples[idx]
    if scene_idx in seen:
        continue
    w2c_all = ds.extrinsics_list[scene_idx].float()
    K_all = ds.intrinsics_list[scene_idx].float()
    N_all = w2c_all.shape[0]
    tgt = list(range(s, e))
    if len(tgt) > T:
        tgt = [tgt[i] for i in even(len(tgt), T)]
    centers = torch.linalg.inv(w2c_all)[:, :3, 3].numpy()
    d_tgt = np.linalg.norm(centers[tgt] - centers[s], axis=1)
    if d_tgt.max() < 1e-6:
        continue
    gi = ds._sample_geo_frustum_cover(scene_idx, s, e)
    side = longer_side(N_all, s, e)
    chs = windows(side, T)

    D = {"geo_lagernvs": 1.35 * float(np.linalg.norm(centers[gi] - centers[s], axis=1).max()),
         "first_farthest_135*": 1.35 * float(d_tgt.max()),
         "cam_dist_mean*": float(d_tgt.mean())}
    if chs:
        D["ctx_longer_135max"] = float(np.mean(
            [1.35 * float(np.linalg.norm(centers[c] - centers[c[0]], axis=1).max()) for c in chs]))
        D["context_longer"] = float(np.mean(
            [float(np.linalg.norm(centers[c] - centers[c[0]], axis=1).mean()) for c in chs]))
    if len(side) >= 2:
        D["ctx_side_135max"] = 1.35 * float(
            np.linalg.norm(centers[side] - centers[side[0]], axis=1).max())
    a = ds._avg_scale(scene_idx, data_name.split("_")[-1])
    if a is not None:
        D["avg_scale(excl)"] = float(a)

    # cam_param, exactly as dataset_dl3dv.__getitem__ builds it (max_trans_norm is false)
    w2c = w2c_all[tgt]
    K = K_all[tgt]
    rel = w2c @ torch.linalg.inv(w2c[0]).unsqueeze(0)                     # (T,4,4)
    intr = torch.stack([K[:, 0, 0] / (K[:, 0, 2] * 2), K[:, 1, 1] / (K[:, 1, 2] * 2)], -1)
    if (getattr(cfg, "intr_norm", "auto") == "rel"
            or (getattr(cfg, "intr_norm", "auto") == "auto" and cfg.scale_mode == "avg_scale")):
        intr = intr / intr[0:1]

    for nm, dv in D.items():
        cp = torch.cat([rel[:, :3, 0], rel[:, :3, 1], rel[:, :3, 3] / dv, intr], -1)  # (T,11)
        x = cp.unsqueeze(0).to(device)
        with torch.no_grad():
            lat = vae.encode(x)
            lat = lat.sample() if hasattr(lat, "sample") else lat
            rec = vae.decode(lat)
        acc[nm]["D"].append(dv)
        acc[nm]["instd"].append(float(cp[:, 6:9].std()))
        acc[nm]["lat"].append(float(lat.std()))
        e_t = float((rec[0, :, 6:9].cpu() - cp[:, 6:9]).abs().mean())
        acc[nm]["rt"].append(e_t)
        acc[nm]["rtw"].append(e_t * dv)
        acc[nm]["rr"].append(float((rec[0, :, :6].cpu() - cp[:, :6]).abs().mean()))
    seen.add(scene_idx); done += 1
    if done % 100 == 0:
        print(f"  {done}/{N}", flush=True)

print(f"\n{done} segments / distinct scenes (exp={EXP} split={SPLIT}) "
      f"vae_latent_scale={cfg.vae_latent_scale}")
hdr = (f"{'divisor':22s} {'n':>5s} {'D med':>7s} {'in_std':>7s} {'lat_raw':>8s} {'lat_scl':>8s} "
       f"{'vae_s@1':>8s} {'rec_t_n':>8s} {'rec_t_w':>8s} {'rec_r':>7s}")
print(hdr); print("-" * len(hdr))
res = {}
for nm in NAMES:
    a = acc[nm]
    if len(a["D"]) < 10:
        continue
    lr = float(np.mean(a["lat"]))
    res[nm] = dict(n=len(a["D"]), D_median=float(np.median(a["D"])),
                   in_trans_std=float(np.mean(a["instd"])), lat_std_raw=lr,
                   lat_std_scaled=lr / cfg.vae_latent_scale, vae_scale_for_1=lr,
                   rec_trans_norm=float(np.mean(a["rt"])),
                   rec_trans_world=float(np.mean(a["rtw"])),
                   rec_rot6d=float(np.mean(a["rr"])))
    v = res[nm]
    print(f"{nm:22s} {v['n']:5d} {v['D_median']:7.3f} {v['in_trans_std']:7.4f} "
          f"{v['lat_std_raw']:8.4f} {v['lat_std_scaled']:8.4f} {v['vae_scale_for_1']:8.4f} "
          f"{v['rec_trans_norm']:8.5f} {v['rec_trans_world']:8.5f} {v['rec_rot6d']:7.5f}")
json.dump(dict(exp=EXP, split=SPLIT, n=done, vae_ckpt=cfg.vae_ckpt_path,
               vae_latent_scale=cfg.vae_latent_scale, cam_dim=cfg.cam_dim, stats=res),
          open(os.path.join(OUT, "vae_recon.json"), "w"), indent=2)
print("\nsaved", os.path.join(OUT, "vae_recon.json"))
