"""geo latent cache 파일이 '어느 view index 들로' 만들어졌는지 매니페스트로 덤프한다.

배경
----
- v2 캐시(`first_cam_not_included/`)는 파일 안에 `{'emb', 'geo_idxs'}` 로 index 를 들고 있다.
- v1 캐시(`first_cam_included/`)는 bare `(M,768)` tensor 라 index 가 없다. 하지만
  `frustum_cover` 는 extrinsics 와 (s,e) 만 보는 deterministic greedy 라 같은 config 로
  `_sample_geo_frustum_cover` 를 다시 돌리면 캐시를 만든 그 선택이 그대로 재현된다.
  (2026-08-03 dataset_dl3dv.py 패치가 학습 경로에서 쓰는 것과 같은 재계산.)

그래서 두 트리 모두 **재계산**으로 매니페스트를 만들고, v2 는 파일에 든 값과 대조해서
재계산이 맞는지 검증까지 한다.

산출물 (subdir 당 2개 파일만 — inode 아끼려고 per-file sidecar 는 안 만든다)
  <cache_dir>/<subdir>__geo_idxs.csv   data_name,scene,s,e,n_view,geo_idxs(공백구분),cached
  <cache_dir>/<subdir>__geo_idxs.pt    {data_name: LongTensor(V)} + meta

사용
----
  EXP=geo_worldtraj SUBDIR=first_cam_included \
    python scripts/data/dump_geo_cache_idxs.py
  EXP=geo_worldtraj_camembed SUBDIR=first_cam_not_included VERIFY=200 \
    python scripts/data/dump_geo_cache_idxs.py

env
  EXP      experiment config (subdir 를 만든 그 config 여야 한다)
  SUBDIR   캐시 하위 트리 이름
  VERIFY   v2 트리에서 재계산 대조할 샘플 수 (0=끔, 기본 200)
  WORKERS  재계산 프로세스 수 (기본 16)
"""
import csv
import os
import os.path as osp
import random
import sys

import torch

HERE = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, osp.join(HERE, "main"))
os.chdir(osp.join(HERE, "main"))
from hydra_cfg import load_cfg  # noqa: E402

EXP = os.environ.get("EXP", "geo_worldtraj")
SUBDIR = os.environ.get("SUBDIR", "first_cam_included")
VERIFY = int(os.environ.get("VERIFY", "200"))
WORKERS = int(os.environ.get("WORKERS", "16"))

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
CACHE = cfg.geo_latent_cache_dir
assert CACHE, "geo_latent_cache_dir 가 config 에 없다"
ROOT = osp.join(CACHE, SUBDIR)
assert osp.isdir(ROOT), ROOT

from dataset_dl3dv import CamDataset  # noqa: E402

ds = CamDataset(cfg, "train")
print(f"exp={EXP} subdir={SUBDIR} samples={len(ds.samples)} "
      f"sampling={cfg.geo_view_sampling} first_view_target_s={cfg.geo_first_view_target_s} "
      f"subtract_first={cfg.geo_cover_subtract_first} centered_at_s={cfg.geo_cover_centered_at_s}",
      flush=True)


def cache_path(name):
    return osp.join(ROOT, name.split("_")[0], f"{name}.pt")


def row_for(i):
    scene_idx, s, e, _caption, name = ds.samples[i]
    idxs = list(ds._sample_geo_frustum_cover(scene_idx, s, e))
    return name, scene_idx, s, e, idxs, osp.exists(cache_path(name))


def main():
    from multiprocessing import Pool
    with Pool(WORKERS) as p:
        rows = p.map(row_for, range(len(ds.samples)), chunksize=64)

    csv_path = osp.join(CACHE, f"{SUBDIR}__geo_idxs.csv")
    pt_path = osp.join(CACHE, f"{SUBDIR}__geo_idxs.pt")
    n_cached = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data_name", "scene_idx", "s", "e", "n_view", "geo_idxs", "cached"])
        for name, sc, s, e, idxs, cached in rows:
            n_cached += cached
            w.writerow([name, sc, s, e, len(idxs), " ".join(map(str, idxs)), int(cached)])
    torch.save({
        "meta": {"exp": EXP, "subdir": SUBDIR, "sampling": cfg.geo_view_sampling,
                 "geo_first_view_target_s": bool(cfg.geo_first_view_target_s),
                 "geo_cover_subtract_first": bool(cfg.geo_cover_subtract_first),
                 "geo_cover_centered_at_s": bool(cfg.geo_cover_centered_at_s),
                 "geo_num_views": int(cfg.geo_num_views)},
        "idxs": {name: torch.tensor(idxs, dtype=torch.long) for name, _, _, _, idxs, _ in rows},
    }, pt_path)
    print(f"wrote {csv_path}\nwrote {pt_path}\n  rows={len(rows)}  cache 파일 존재={n_cached}",
          flush=True)

    if VERIFY:
        have = [r for r in rows if r[5]]
        random.seed(0)
        sample = random.sample(have, min(VERIFY, len(have)))
        ok = miss = novt = 0
        for name, _, _, _, idxs, _ in sample:
            c = torch.load(cache_path(name), map_location="cpu", weights_only=False)
            if not (isinstance(c, dict) and "geo_idxs" in c):
                novt += 1
                continue
            stored = [int(v) for v in c["geo_idxs"].tolist()]
            if stored == idxs:
                ok += 1
            else:
                miss += 1
                if miss <= 5:
                    print(f"  MISMATCH {name}: stored={stored} recomp={idxs}")
        print(f"verify: v2={ok + miss}개 중 일치 {ok} / 불일치 {miss}, "
              f"v1(geo_idxs 없음)={novt}", flush=True)


if __name__ == "__main__":
    main()
