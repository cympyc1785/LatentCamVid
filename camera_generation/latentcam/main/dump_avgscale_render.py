"""Step 1 (latentcam): dump N segments for the 'avg_scale-only' LagerNVS render test.

For each of N target segments (geo_worldtraj config -> frustum_cover context = target's first
frame s + out-of-target coverage views, EXACTLY as in training): dump the GT target trajectory
(segment cameras), the geo-context cameras/images/intrinsics, and the STORED point-cloud avg_scale.
Step 2 renders with avg_scale normalization (NOT the usual 1.35*max(context)).

Writes results/lagernvs_avgscale_test/<seg_flat>/render_inputs.pt with:
  image_paths, ctx_c2w (V,4,4 OpenCV world), ctx_K (V,3,3), hw_full,
  tgt_c2w (T,4,4 OpenCV world = GT), avg_scale (float), seg
env: N (default 10), CACHE_EXP (default geo_worldtraj),
     SEGS (optional comma-separated data_names, e.g. '1K_<hash>_0,...' -> only those scenes are
     parsed via CamDataset.from_segments; without it the corpus index is built/loaded first)
"""
import os, sys, json, random
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

N = int(os.environ.get("N", "10"))
EXP = os.environ.get("CACHE_EXP", "geo_worldtraj")
cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
from dataset_dl3dv import CamDataset

OUT = os.path.join(HERE, "results", "lagernvs_avgscale_test")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(cfg.random_seed); np.random.seed(cfg.random_seed); random.seed(cfg.random_seed)
SEGS = [x for x in os.environ.get("SEGS", "").split(",") if x]
ds = CamDataset.from_segments(cfg, SEGS) if SEGS else CamDataset(cfg, "train")


def geo_idxs_for(scene_idx, s, e):
    if cfg.geo_view_sampling == "frustum_cover":
        gi = ds._sample_geo_frustum_cover(scene_idx, s, e)
    elif cfg.geo_view_sampling == "hybrid":
        gi = ds._sample_geo_hybrid(scene_idx, s, e)
    else:
        gi = [s + i for i in ds._even_indices(e - s, ds.geo_num_views)]
    return list(gi)


def even(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


done = 0
for idx in range(len(ds.samples)):
    if done >= N:
        break
    scene_idx, s, e, caption, data_name = ds.samples[idx]
    seg_key = data_name.split("_")[-1]
    avg = ds._saved_avg_scale(scene_idx, seg_key)
    if avg is None:                       # need the stored point-cloud avg_scale
        continue
    gi = geo_idxs_for(scene_idx, s, e)
    w2c = ds.extrinsics_list[scene_idx]
    K = ds.intrinsics_list[scene_idx]
    h, w = ds.hw_list[scene_idx]
    frame_files = ds.frame_files_list[scene_idx]
    # target GT trajectory (segment, subsampled to num_frames)
    tgt = w2c[s:e].float()
    if tgt.shape[0] > cfg.num_frames:
        tgt = tgt[even(tgt.shape[0], cfg.num_frames)]
    tgt_c2w = torch.linalg.inv(tgt)
    ctx_c2w = torch.linalg.inv(w2c[gi].float())
    d = {"image_paths": [frame_files[i] for i in gi], "ctx_c2w": ctx_c2w,
         "ctx_K": K[gi].float(), "hw_full": torch.tensor([float(h), float(w)]),
         "tgt_c2w": tgt_c2w, "avg_scale": float(avg), "seg": data_name}
    od = os.path.join(OUT, data_name); os.makedirs(od, exist_ok=True)
    torch.save(d, os.path.join(od, "render_inputs.pt"))
    print(f"[{done}] {data_name}: ctx {tuple(ctx_c2w.shape)} tgt {tuple(tgt_c2w.shape)} avg_scale={float(avg):.2f} geo_idxs={gi}")
    done += 1
print(f"dumped {done} segments -> {OUT}")
