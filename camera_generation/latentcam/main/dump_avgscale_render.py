"""Step 1 (latentcam): dump N segments for the 'avg_scale-only' LagerNVS render test.

For each of N target segments (geo_worldtraj config -> frustum_cover context = target's first
frame s + out-of-target coverage views, EXACTLY as in training): dump the GT target trajectory
(segment cameras), the geo-context cameras/images/intrinsics, and the STORED point-cloud avg_scale.
Step 2 renders with avg_scale normalization (NOT the usual 1.35*max(context)).

Writes results/<OUT_NAME>/<seg_flat>/render_inputs.pt with:
  image_paths, ctx_c2w (V,4,4 OpenCV world), ctx_K (V,3,3), hw_full,
  tgt_c2w (T,4,4 OpenCV world = GT), avg_scale (float), seg
  [new] tgt_image_paths: the GT frame files for the target cameras -> lets step 2 compute PSNR
  [new] scales: {name: divisor} for the normalization ablation. All are "view0(=frame s)-relative
        translations / divisor", they differ only in what the divisor is:
          lagernvs          1.35*max||ctx center - ctx0||  == LagerNVS's OWN normalize_extrinsics
                            (data/normalization.py:37-50); the control, and identical to
                            dataset_dl3dv._geo_lagernvs_scale when geo_first_view_target_s.
          avg_scale         stored point-cloud avg_scale (what latentcam currently trains on)
          maxd_seg          1.35*max||tgt center - tgt0||  (target-derived -> leaks, upper bound)
          ctx_longer_135max longer out-of-segment side chunked into num_frames windows, mean over
                            windows of 1.35*max||c - c_win0||  (leakage-free)
          ctx_side_135max   [new] the SAME longer out-of-segment side taken as ONE range, no
                            windowing: 1.35*max||c - c_side0||  (leakage-free; measures the
                            scene's spatial extent rather than per-num_frames motion)
env: N (default 10), CACHE_EXP (default geo_worldtraj), OUT_NAME (default lagernvs_avgscale_test),
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

OUT = os.path.join(HERE, "results", os.environ.get("OUT_NAME", "lagernvs_avgscale_test"))
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


def ctx_longer_135max(centers, s, e, T):
    """[new] mean over num_frames windows of 1.35*max||c - c_win0||, taken on the LONGER
    out-of-segment side -- the same window construction as dataset_dl3dv._cam_dist_mean_context
    (line 735), but aggregating 1.35*max instead of mean so the units match LagerNVS's."""
    N = centers.shape[0]
    side = list(range(0, s)) if s >= (N - e) else list(range(e, N))
    chunks = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]
    if not chunks and len(side) >= 2:
        chunks = [side]
    if not chunks:
        return None
    return float(np.mean([1.35 * float(np.linalg.norm(centers[c] - centers[c[0]], axis=1).max())
                          for c in chunks]))


def ctx_side_135max(centers, s, e):
    """[new] no windowing: 1.35*max||c - c_side0|| over the WHOLE longer out-of-segment side."""
    N = centers.shape[0]
    side = list(range(0, s)) if s >= (N - e) else list(range(e, N))
    if len(side) < 2:
        return None
    return 1.35 * float(np.linalg.norm(centers[side] - centers[side[0]], axis=1).max())


ONE_PER_SCENE = os.environ.get("ONE_PER_SCENE", "0") == "1"   # [new] N distinct scenes, not N
seen_scenes = set()                                            # consecutive segments of scene 0

done = 0
for idx in range(len(ds.samples)):
    if done >= N:
        break
    scene_idx, s, e, caption, data_name = ds.samples[idx]
    if ONE_PER_SCENE and scene_idx in seen_scenes:
        continue
    seg_key = data_name.split("_")[-1]
    avg = ds._avg_scale(scene_idx, seg_key)
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
    # [new] the GT frames matching tgt_c2w (same subsample) -> step 2 can compute PSNR
    tgt_idxs = list(range(s, e))
    if len(tgt_idxs) > cfg.num_frames:
        tgt_idxs = [tgt_idxs[i] for i in even(len(tgt_idxs), cfg.num_frames)]
    # [new] the normalization ablation's divisors (all view0=frame s relative)
    centers_all = torch.linalg.inv(w2c.float())[:, :3, 3].numpy()
    scales = {
        "lagernvs": 1.35 * float(np.linalg.norm(centers_all[gi] - centers_all[gi[0]], axis=1).max()),
        "avg_scale": float(avg),
        "maxd_seg": 1.35 * float(np.linalg.norm(
            centers_all[tgt_idxs] - centers_all[tgt_idxs[0]], axis=1).max()),
    }
    d2 = ctx_longer_135max(centers_all, s, e, cfg.num_frames)
    if d2 is not None:
        scales["ctx_longer_135max"] = d2
    f = ctx_side_135max(centers_all, s, e)
    if f is not None:
        scales["ctx_side_135max"] = f
    d = {"image_paths": [frame_files[i] for i in gi], "ctx_c2w": ctx_c2w,
         "ctx_K": K[gi].float(), "hw_full": torch.tensor([float(h), float(w)]),
         "tgt_c2w": tgt_c2w, "avg_scale": float(avg), "seg": data_name,
         "tgt_image_paths": [frame_files[i] for i in tgt_idxs], "scales": scales}
    od = os.path.join(OUT, data_name); os.makedirs(od, exist_ok=True)
    torch.save(d, os.path.join(od, "render_inputs.pt"))
    print(f"[{done}] {data_name}: ctx {tuple(ctx_c2w.shape)} tgt {tuple(tgt_c2w.shape)} "
          f"geo_idxs={gi} scales=" + " ".join(f"{k}={v:.3f}" for k, v in scales.items()))
    seen_scenes.add(scene_idx)
    done += 1
print(f"dumped {done} segments -> {OUT}")
