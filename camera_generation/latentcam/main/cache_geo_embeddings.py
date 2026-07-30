"""Precompute + cache frozen geo embeddings (fp16) for DL3DV segments, so training can skip the
per-step LagerNVS forward (frozen + deterministic context -> geo_emb is constant across epochs).

Pilot: BATCH_FILTER='1K' -> only the 1K batch. Saves one file per segment:
  <OUT>/<data_name>.pt  = geo_emb (M, 768) fp16   (mask is all-ones -> not stored)
env: CACHE_EXP (default geo_worldtraj), CACHE_BATCH (default 1K), CACHE_OUT.
"""
import os, sys
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

EXP = os.environ.get("CACHE_EXP", "geo_worldtraj")
BATCH = os.environ.get("CACHE_BATCH", "1K")
OUT = os.environ.get("CACHE_OUT", "/data1/cympyc1785/data/DL3DV/latent_cache")
DEVICE = "cuda:0"
os.makedirs(OUT, exist_ok=True)

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
from dataset_dl3dv import CamDataset
from base import collate_fn
from models.geo_encoder import build_geo_encoder
from torch.utils.data import Subset, DataLoader


def geo_encode(ge, data):
    images = data["images"].to(DEVICE)
    override = (data["avg_scale"].to(DEVICE)
                if getattr(cfg, "geo_lagernvs_skip_ctx_norm", False) else None)
    cam_token = ge.build_cam_token(data["geo_c2w"].to(DEVICE), data["geo_fxfycxcy"].to(DEVICE),
                                   data["geo_hw"].to(DEVICE), override_scale=override)
    return ge(images, cam_token)


def main():
    ds = CamDataset(cfg, "train")
    idxs = [i for i, s in enumerate(ds.samples) if s[4].startswith(BATCH + "_") or s[4].startswith(BATCH + "/")]
    print(f"{EXP}: {len(idxs)} segments in batch {BATCH} (of {len(ds.samples)} total)")
    ge = build_geo_encoder(cfg).to(DEVICE).eval()

    loader = DataLoader(Subset(ds, idxs), batch_size=4, shuffle=False, num_workers=8, collate_fn=collate_fn)
    done, saved, bytes_ = 0, 0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for data in loader:
            names = data["data_name"]
            # skip if all already cached
            todo = [n for n in names if not os.path.exists(os.path.join(OUT, f"{n}.pt"))]
            if todo:
                emb, _ = geo_encode(ge, data)              # (B, M, 768)
                emb = emb.to(torch.float16).cpu()
                for b, n in enumerate(names):
                    p = os.path.join(OUT, f"{n}.pt")
                    if not os.path.exists(p):
                        torch.save(emb[b].clone(), p); saved += 1; bytes_ += emb[b].numel() * 2
            done += len(names)
            if done % 200 < 4:
                print(f"  {done}/{len(idxs)} | saved {saved} | {bytes_/1e9:.2f} GB", flush=True)
    print(f"DONE batch {BATCH}: {saved} new files, ~{bytes_/1e9:.2f} GB -> {OUT}")


if __name__ == "__main__":
    main()
