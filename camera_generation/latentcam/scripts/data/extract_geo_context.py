"""Extract the geo-context camera world centers for a set of eval targets (geo_worldtraj
config; context selection is deterministic + identical for align). Saves a JSON cache
{target_id: [[x,y,z], ...]} of context camera centers in DENORMALIZED world (same frame as the
saved transforms_ref.json), so the comparison viz can overlay them as stars.

Run (기본값 = 예전 160 target 그대로):
  python scripts/data/extract_geo_context.py

[new] target 집합과 seg_list 를 옵션으로 준다. eval_testset.py 로 뽑은 전체 held-out 집합처럼
기본 seg_list 가 아닌 split 을 쓸 때 필요하다:
  python scripts/data/extract_geo_context.py \
    --testdir eval_my/20260731_001511_dl3dv_geo_worldtraj/test \
    --out results/.../plots_geo_worldtraj/_geo_context.json --split test \
    --set test_seg_list=/data1/cympyc1785/data/DL3DV/scenes/latentcam_test_seg_list_c4d2k5y4.txt
"""
import argparse
import os, sys, glob, json
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))

ap = argparse.ArgumentParser()
ap.add_argument("--testdir", default=os.path.join(
    HERE, "results", "20260721_213927_dl3dv_geo_worldtraj", "test"),
    help="<tid>_transforms_ref.json 이 들어 있는 디렉토리 (HERE 기준 상대경로도 가능)")
ap.add_argument("--out", default=os.path.join(
    HERE, "results", "compare", "normalization_point_vs_dist", "_geo_context.json"))
ap.add_argument("--experiment", default="geo_worldtraj", help="hydra experiment (context 선택 규칙)")
ap.add_argument("--split", default="train", choices=["train", "test"],
                help="target 들이 들어 있는 CamDataset split")
ap.add_argument("--set", action="append", default=[], metavar="K=V",
                help="hydra override 추가 (예 test_seg_list=...). 반복 가능")
args = ap.parse_args()
TESTDIR = args.testdir if os.path.isabs(args.testdir) else os.path.join(HERE, args.testdir)
OUT = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)

os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg
cfg, _ = load_cfg("config", overrides=[f"experiment={args.experiment}"] + args.set)
from dataset_dl3dv import CamDataset


def seg_key(sid):
    bh, seg = sid.rsplit("_", 1); b, h = bh.split("_", 1); return f"{b}/{h}/{seg}"


ids = sorted({os.path.basename(f)[:-len("_transforms_ref.json")]
              for f in glob.glob(os.path.join(TESTDIR, "*_transforms_ref.json"))})
print(f"{len(ids)} targets | split={args.split} | out={OUT}")

ds = CamDataset(cfg, args.split)
key2idx = {seg_key(s[4]): i for i, s in enumerate(ds.samples)}
print(f"dataset split '{args.split}': {len(ds.samples)} samples")

# [fix 2026-08-03] 예전에는 ds[idx]['geo_c2w'] 를 읽었는데, geo latent cache 가 히트하면
# __getitem__ 이 dataset_dl3dv.py:960-976 에서 조기 return 해서 'geo_c2w' 가 아예 안 나온다
# (geo_c2w 는 on-the-fly 경로에서만 붙는다, :1011). 캐시가 다 채워진 트리에서는 전 target 이
# "no geo_c2w" 로 빠졌다. 어차피 필요한 건 context view 의 world center 뿐이라 sampler 를 직접
# 불러 extrinsics 에서 뽑는다 -- 이미지/latent 로딩도 건너뛰어 훨씬 빠르다.
SAMPLERS = {"frustum_cover": "_sample_geo_frustum_cover", "hybrid": "_sample_geo_hybrid"}
sampler_name = SAMPLERS.get(ds.geo_view_sampling)
if sampler_name is None:
    raise SystemExit(f"geo_view_sampling='{ds.geo_view_sampling}' 은 아직 지원 안 함 "
                     f"(지원: {sorted(SAMPLERS)})")
sampler = getattr(ds, sampler_name)
print(f"geo_view_sampling={ds.geo_view_sampling} -> {sampler_name}")

out, missing = {}, 0
for n, tid in enumerate(ids):
    sk = seg_key(tid)
    idx = key2idx.get(sk)
    if idx is None:
        missing += 1
        if missing <= 5:
            print("MISSING", tid)
        continue
    scene_idx, s, e = ds.samples[idx][:3]
    geo_idxs = list(sampler(scene_idx, s, e))
    if ds.geo_test_inseg_k:   # [new] --set geo_test_inseg_k=K 로 돌린 eval 과 같은 context
        geo_idxs = ds._mix_inseg_context(geo_idxs, s, e)
    gc2w = torch.linalg.inv(ds.extrinsics_list[scene_idx][geo_idxs].float())   # (V,4,4) c2w
    centers = gc2w[:, :3, 3].cpu().numpy()   # (V,3) world camera centers
    out[tid] = centers.tolist()
    if (n + 1) % 500 == 0:
        print(f"  {n+1}/{len(ids)}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print("saved", OUT, "| targets:", len(out), "| missing:", missing,
      "| views/target:", len(next(iter(out.values()))))
