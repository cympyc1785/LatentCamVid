#!/usr/bin/env python3
"""Dump the latentcam DL3DV splits to the DATA dir.

(1) SEGMENT split — reproduces exactly what geo_worldtraj uses: build CamDataset with the
    geo_worldtraj config (meta_worldtraj.csv, num_frames=49), then torch.random_split
    [0.9, 0.1] with generator manual_seed(cfg.random_seed). Writes segment lists
    (`<batch>/<hash>/<seg_key>` per line):
      latentcam_train_seg_list.txt / latentcam_test_seg_list.txt
    NOTE: segment-level split -> a scene's segments may fall in both -> scene leakage.

(2) SCENE split — explicit seen/unseen. Unique scenes (those with >=1 valid segment),
    sorted, 9:1 split with a fixed seed. Writes (`<batch>/<hash>` per line):
      latentcam_train_list.txt / latentcam_test_list.txt

Run: PYTHONPATH=<root> python scripts/data/make_latentcam_splits.py
"""
import os
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import random_split

from hydra_cfg import load_cfg
from dataset_dl3dv import CamDataset

OUT_DIR = "/data1/cympyc1785/data/DL3DV/scenes"

# [new 2026-08-03] FROM_CACHE: dump the split of a SPECIFIC index cache .pt instead of whatever
#   the live config resolves to. Needed because the live config now carries the frozen-pose
#   blacklist (pool 39817) while the reference run 20260731_001511 (c4d2k5y4) split a 39830 pool
#   -- random_split's partition depends on len(dataset), so the only way to reproduce that run's
#   exact train/val is to split its own cache. Empty = existing behavior (CamDataset from config).
# OUT_SUFFIX: appended before .txt so a new dump never clobbers the 2026-07-23 lists.
# NOTE: lists are written in random_split order, NOT sorted. base.py feeds validset_loader with
#   shuffle=False, so "val = first 160 of the test list" only reproduces a random_split run's
#   wandb val if the list keeps that order. Sorting makes the first 160 all-1K (measured: 160/160
#   vs 26/160 in random_split order) and breaks comparability with c4d2k5y4.
FROM_CACHE = os.environ.get("FROM_CACHE", "")
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")


def seg_id_to_chunk_seg(sample_id):
    """`<batch>_<hash>_<segkey>` -> ('<batch>/<hash>', '<segkey>')."""
    bh, seg = sample_id.rsplit("_", 1)
    batch, h = bh.split("_", 1)
    return f"{batch}/{h}", seg


def main():
    if FROM_CACHE:
        obj = torch.load(FROM_CACHE, weights_only=False)
        samples = obj["samples"] if isinstance(obj, dict) and "samples" in obj else obj
        seed = 42
        print(f"[FROM_CACHE] {FROM_CACHE}")
    else:
        cfg, _ = load_cfg("config", ["experiment=geo_worldtraj"])
        samples = CamDataset(cfg, "train").samples
        seed = int(cfg.random_seed)
    N = len(samples)
    print(f"dataset: {N} segments | seed={seed}")

    # ---- (1) segment split : EXACT reproduction of base.py _make_batch_generator ----
    train_size = int(0.9 * N)
    test_size = N - train_size
    gen = torch.Generator().manual_seed(seed)
    tr, te = random_split(range(N), [train_size, test_size], generator=gen)
    tr_idx, te_idx = list(tr), list(te)

    def seg_line(i):
        chunk, seg = seg_id_to_chunk_seg(samples[i][4])
        return f"{chunk}/{seg}"

    tr_name = f"latentcam_train_seg_list{OUT_SUFFIX}.txt"
    te_name = f"latentcam_test_seg_list{OUT_SUFFIX}.txt"
    with open(os.path.join(OUT_DIR, tr_name), "w") as f:
        f.write("\n".join(seg_line(i) for i in tr_idx) + "\n")
    with open(os.path.join(OUT_DIR, te_name), "w") as f:
        f.write("\n".join(seg_line(i) for i in te_idx) + "\n")
    print(f"(1) segment split: train={len(tr_idx)} test={len(te_idx)} -> {tr_name} / {te_name}")

    # ---- (2) scene split : explicit seen/unseen, 9:1 ----
    scenes = sorted({seg_id_to_chunk_seg(s[4])[0] for s in samples})
    ns = len(scenes)
    ntr = int(0.9 * ns)
    perm = torch.randperm(ns, generator=torch.Generator().manual_seed(seed)).tolist()
    train_scenes = sorted(scenes[i] for i in perm[:ntr])
    test_scenes = sorted(scenes[i] for i in perm[ntr:])
    with open(os.path.join(OUT_DIR, f"latentcam_train_list{OUT_SUFFIX}.txt"), "w") as f:
        f.write("\n".join(train_scenes) + "\n")
    with open(os.path.join(OUT_DIR, f"latentcam_test_list{OUT_SUFFIX}.txt"), "w") as f:
        f.write("\n".join(test_scenes) + "\n")
    # sanity: scene lists are disjoint
    overlap = set(train_scenes) & set(test_scenes)
    print(f"(2) scene split: {ns} scenes -> train={len(train_scenes)} test={len(test_scenes)} "
          f"overlap={len(overlap)} -> latentcam_{{train,test}}_list{OUT_SUFFIX}.txt")
    print("DONE ->", OUT_DIR)


if __name__ == "__main__":
    main()
