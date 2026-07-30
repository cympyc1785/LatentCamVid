"""Precompute + cache frozen geo embeddings (fp16) for DL3DV segments, so training can skip the
per-step LagerNVS forward (frozen + deterministic context -> geo_emb is constant across epochs).

Saves one file per segment. Two layouts, selected by CACHE_LAYOUT:
  'batch' (default) -> <OUT>/<first_cam_included|first_cam_not_included>/<iK>/<data_name>.pt
  'flat'            -> <OUT>/<data_name>.pt        (the original pilot layout, unchanged)
The 'batch' layout mirrors the DL3DV scene root, which is now split into per-1000 batch dirs
(<dl3dv_root>/{1K..7K}/<scene_hash>/), and additionally splits on geo_first_view_target_s --
that flag changes which context views are encoded, so the two settings must not share files.
Each file is {'emb': geo_emb (M, 768) fp16, 'geo_idxs': (V,) int16} (the mask is all-ones -> not
stored), ~6.7 MB. geo_idxs is the selected context frame index per view; it lets a cache hit
rebuild the per-view camera embedding (cfg.geo_cam_embed) without redoing the greedy coverage
search. Files written before geo_idxs existed are bare (M, 768) tensors and are still read (the
dataset accepts both), but they cannot serve a geo_cam_embed run -- those fall back to on-the-fly.

CACHE_BATCH accepts one batch, a comma-separated list, or 'all' (every batch present in the
index). Batches are processed in order and each is resumable -- already-present files are
skipped without a forward pass.

env: CACHE_EXP (default geo_worldtraj), CACHE_BATCH (default 1K), CACHE_OUT,
     CACHE_LAYOUT (batch|flat), CACHE_SUBDIR (override the first_cam_* dir), CACHE_BS (default 4).
"""
import os, sys, time
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

EXP = os.environ.get("CACHE_EXP", "geo_worldtraj")
BATCH = os.environ.get("CACHE_BATCH", "1K")
OUT = os.environ.get("CACHE_OUT", "/data1/cympyc1785/data/DL3DV/latent_cache")
LAYOUT = os.environ.get("CACHE_LAYOUT", "batch")
BS = int(os.environ.get("CACHE_BS", "4"))
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


def batch_of(sample):
    """Batch dir ('1K'..'7K') a sample belongs to. data_name is '<batch>_<hash>_<seg>' and the
    chunk field s[4] is built from '<batch>/<hash>' with '/'->'_' (dataset_dl3dv.py:369)."""
    return str(sample[4]).split('/')[0].split('_')[0]


def out_dir_for(batch):
    """Where this batch's files live. 'flat' reproduces the original pilot layout exactly."""
    if LAYOUT == "flat":
        return OUT
    sub = os.environ.get("CACHE_SUBDIR") or (
        "first_cam_included" if getattr(cfg, "geo_first_view_target_s", False)
        else "first_cam_not_included")
    return os.path.join(OUT, sub, batch)


def run_batch(ds, ge, batch, idxs):
    odir = out_dir_for(batch)
    os.makedirs(odir, exist_ok=True)
    print(f"[{batch}] {len(idxs)} segments -> {odir}", flush=True)
    loader = DataLoader(Subset(ds, idxs), batch_size=BS, shuffle=False, num_workers=8,
                        collate_fn=collate_fn)
    done, saved, bytes_, t0 = 0, 0, 0, time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for data in loader:
            names = data["data_name"]
            paths = [os.path.join(odir, f"{n}.pt") for n in names]
            # skip the forward entirely if every segment in this mini-batch is already cached
            if any(not os.path.exists(p) for p in paths):
                emb, _ = geo_encode(ge, data)              # (B, M, 768)
                emb = emb.to(torch.float16).cpu()
                # NOT `idxs` -- that is this function's sample-index argument; shadowing it broke
                # the progress/ETA line (it printed done/<batch size>).
                gidxs = data["geo_idxs"].to(torch.int16).cpu()  # (B, V) selected context frames
                for b, p in enumerate(paths):
                    if not os.path.exists(p):
                        torch.save({"emb": emb[b].clone(), "geo_idxs": gidxs[b].clone()}, p)
                        saved += 1; bytes_ += emb[b].numel() * 2
            done += len(names)
            if done % 200 < BS:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(idxs) - done) / max(rate, 1e-9)
                print(f"  [{batch}] {done}/{len(idxs)} | saved {saved} | {bytes_/1e9:.2f} GB "
                      f"| {rate:.2f} seg/s | ETA {eta/60:.1f} min", flush=True)
    print(f"DONE batch {batch}: {saved} new files, ~{bytes_/1e9:.2f} GB -> {odir} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return saved, bytes_


def main():
    ds = CamDataset(cfg, "train")
    all_batches = sorted({batch_of(s) for s in ds.samples}, key=lambda x: (len(x), x))
    wanted = all_batches if BATCH.lower() == "all" else [b.strip() for b in BATCH.split(',') if b.strip()]
    missing = [b for b in wanted if b not in all_batches]
    if missing:
        raise SystemExit(f"batch(es) {missing} not in index (present: {all_batches})")
    per = {b: [i for i, s in enumerate(ds.samples) if batch_of(s) == b] for b in wanted}
    print(f"{EXP}: layout={LAYOUT} batches={wanted} "
          f"({sum(len(v) for v in per.values())} of {len(ds.samples)} segments)", flush=True)

    ge = build_geo_encoder(cfg).to(DEVICE).eval()
    tot_saved, tot_bytes = 0, 0
    for b in wanted:
        s, by = run_batch(ds, ge, b, per[b])
        tot_saved += s; tot_bytes += by
    print(f"ALL DONE: {tot_saved} new files, ~{tot_bytes/1e9:.2f} GB across {wanted}")


if __name__ == "__main__":
    main()
