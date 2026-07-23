"""Extract the geo-context camera world centers for the 160 validation targets (geo_worldtraj
config; context selection is deterministic + identical for align). Saves a JSON cache
{target_id: [[x,y,z], ...]} of context camera centers in DENORMALIZED world (same frame as the
saved transforms_ref.json), so the comparison viz can overlay them as stars.

Run:  python scripts/data/extract_geo_context.py
"""
import os, sys, glob, json
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))

from hydra_cfg import load_cfg
cfg, _ = load_cfg("config", overrides=["experiment=geo_worldtraj"])
from dataset_dl3dv import CamDataset

TESTDIR = os.path.join(HERE, "results", "20260721_213927_dl3dv_geo_worldtraj", "test")
OUT = os.path.join(HERE, "results", "compare", "normalization_point_vs_dist", "_geo_context.json")


def seg_key(sid):
    bh, seg = sid.rsplit("_", 1); b, h = bh.split("_", 1); return f"{b}/{h}/{seg}"


ids = sorted({os.path.basename(f)[:-len("_transforms_ref.json")]
              for f in glob.glob(os.path.join(TESTDIR, "*_transforms_ref.json"))})
print(f"{len(ids)} targets")

ds = CamDataset(cfg, "train")
key2idx = {seg_key(s[4]): i for i, s in enumerate(ds.samples)}

out = {}
for n, tid in enumerate(ids):
    sk = seg_key(tid)
    idx = key2idx.get(sk)
    if idx is None:
        print("MISSING", tid); continue
    d = ds[idx]
    if "geo_c2w" not in d:
        print("no geo_c2w", tid); continue
    centers = d["geo_c2w"][:, :3, 3].cpu().numpy()   # (V,3) world camera centers (OpenCV c2w == OpenGL centers)
    out[tid] = centers.tolist()
    if (n + 1) % 40 == 0:
        print(f"  {n+1}/{len(ids)}")

json.dump(out, open(OUT, "w"))
print("saved", OUT, "| targets:", len(out), "| views/target:", len(next(iter(out.values()))))
