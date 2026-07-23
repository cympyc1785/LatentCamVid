"""Step 1 (latentcam env): dump the exact geo-context (frames used at inference) + the predicted
target trajectory for the results/validation target, so lagernvs can render the pred trajectory.

Writes <result_dir>/render_inputs.pt with:
  image_paths : list[str]      geo-context image files (the conditioning views used at inference)
  ctx_c2w     : (V,4,4)        context camera c2w, OpenCV, WORLD frame
  ctx_K       : (V,3,3)        context intrinsics (px, images_4 full res)
  hw_full     : (2,)           images_4 full-res (H,W) the intrinsics are in
  pred_c2w    : (T,4,4)        predicted target cameras, OpenCV, WORLD frame
env: RI_EXP (hydra experiment), RI_RESULT (result subdir under results/validation), RI_SEG.
"""
import os, sys, json
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
import random
from hydra_cfg import load_cfg

EXP = os.environ["RI_EXP"]; RESULT = os.environ["RI_RESULT"]
SEG = os.environ.get("RI_SEG", "1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f/0")
cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
from dataset_dl3dv import CamDataset

VAL = os.path.join(HERE, "results", "validation", RESULT)
GL2CV = torch.diag(torch.tensor([1., -1., -1., 1.])).float()


def seg_key(sid):
    bh, seg = sid.rsplit("_", 1); b, h = bh.split("_", 1); return f"{b}/{h}/{seg}"


torch.manual_seed(cfg.random_seed); np.random.seed(cfg.random_seed); random.seed(cfg.random_seed)
ds = CamDataset(cfg, "train")
idx = next(i for i, s in enumerate(ds.samples) if seg_key(s[4]) == SEG)
scene_idx, s, e, _, _ = ds.samples[idx]

# mirror __getitem__ geo view selection
if cfg.geo_view_sampling == "frustum_cover":
    geo_idxs = ds._sample_geo_frustum_cover(scene_idx, s, e)
elif cfg.geo_view_sampling == "hybrid":
    geo_idxs = ds._sample_geo_hybrid(scene_idx, s, e)
elif cfg.geo_view_sampling == "random_inseg":
    geo_idxs = ds._sample_geo_random_inseg(s, e)
else:
    geo_idxs = [s + i for i in ds._even_indices(e - s, ds.geo_num_views)]
if getattr(ds, "geo_shuffle_order", False):
    geo_idxs = list(geo_idxs)
    if getattr(cfg, "geo_shuffle_keep_first", False) and len(geo_idxs) > 1:
        rest = geo_idxs[1:]; random.shuffle(rest); geo_idxs = [geo_idxs[0]] + rest
    else:
        random.shuffle(geo_idxs)
geo_idxs = list(geo_idxs)
print("geo_idxs:", geo_idxs)

w2c = ds.extrinsics_list[scene_idx]           # (N,4,4) OpenCV w2c world
K = ds.intrinsics_list[scene_idx]             # (N,3,3)
h, w = ds.hw_list[scene_idx]
frame_files = ds.frame_files_list[scene_idx]
ctx_c2w = torch.linalg.inv(w2c[geo_idxs].float())     # OpenCV c2w world
ctx_K = K[geo_idxs].float()
image_paths = [frame_files[i] for i in geo_idxs]

# predicted target trajectory (OpenGL c2w world) -> OpenCV c2w world
pred = json.load(open(os.path.join(VAL, f"{SEG.replace('/', '_')}_transforms_pred.json")))
pred_c2w_gl = torch.tensor([f["transform_matrix"] for f in pred["frames"]], dtype=torch.float32)
pred_c2w = pred_c2w_gl @ GL2CV                        # OpenGL -> OpenCV c2w

out = {"image_paths": image_paths, "ctx_c2w": ctx_c2w, "ctx_K": ctx_K,
       "hw_full": torch.tensor([float(h), float(w)]), "pred_c2w": pred_c2w, "seg": SEG}
torch.save(out, os.path.join(VAL, "render_inputs.pt"))
print("saved", os.path.join(VAL, "render_inputs.pt"), "| ctx", tuple(ctx_c2w.shape), "| pred", tuple(pred_c2w.shape))
