"""video_onfly=molmo2_conn arm 의 frame 디코드를 매 스텝 안 하려면 씬당 무엇을 미리 구워 두나 (R58 / D286).

답: 씬 frame [s, e) 49장을 Molmo2 전처리 앞단(378x378 squash, bilinear, antialias=False) 까지 한
**uint8** 배열 `<OUT>/<scene_key>.npy` (T,378,378,3) ≈ 21 MB. ViT feature 캐시가 아니다 — ViT 는 학습
중 계속 frozen on-the-fly 로 돈다 (사용자 결정 2026-09-25: 27x27 feature 캐시 1.67 TB 는 안 함,
frame 캐시 + worker 증설은 채택). dataset 은 `video_onfly_frame_cache` 가 있고 파일이 있으면 이걸
mmap 으로 읽고, 없으면 PNG 에서 같은 함수(`load_molmo2_frames`)로 만든다 — 값이 비트 동일하다.

사용 (CPU 전용, GPU 불필요):
  EXP=dynpose_d200_siglip2conn_srccam SHARDS=8 SHARD=0 \
    /data1/cympyc1785/miniconda3/envs/latentcam/bin/python scripts/data/cache_molmo2_frames.py
"""
import os
import os.path as osp
import sys
import time

import numpy as np

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, osp.join(HERE, "main"))
os.chdir(osp.join(HERE, "main"))
from hydra_cfg import load_cfg                                    # noqa: E402

EXP = os.environ.get("EXP", "dynpose_d200_siglip2conn_srccam")
SPLITS = [s.strip() for s in os.environ.get("SPLITS", "train,test").split(',') if s.strip()]
SHARDS = max(1, int(os.environ.get("SHARDS", "1")))
SHARD = int(os.environ.get("SHARD", "0"))
assert 0 <= SHARD < SHARDS

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
OUT = os.environ.get("OUT") or getattr(cfg, 'video_onfly_frame_cache', None)
assert OUT, "OUT 또는 cfg.video_onfly_frame_cache 가 필요하다"
cfg.video_onfly_frame_cache = None          # 굽는 동안에는 PNG 경로를 타야 한다

from dataset_dl3dv import CamDataset                              # noqa: E402
from models.molmo2_video_connector import load_molmo2_frames      # noqa: E402

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n_new = 0
    for split in SPLITS:
        ds = CamDataset(cfg, split)
        plan = {}
        for scene_idx, s, e, _, data_name in ds.samples:
            key = ds.geo_raw_key(data_name)
            prev = plan.setdefault(key, (scene_idx, s, e))
            assert prev[1:] == (s, e), f"{key}: frame 창이 segment 마다 다르다 — 씬 단위 캐시 불가"
        keys = sorted(plan)
        nall = len(keys)
        if SHARDS > 1:
            keys = [k for i, k in enumerate(keys) if i % SHARDS == SHARD]
        todo = [k for k in keys if not osp.exists(osp.join(OUT, f'{k}.npy'))]
        print(f"[{split}] scenes {len(keys)}/{nall} (shard {SHARD}/{SHARDS}, "
              f"skip-done {len(keys) - len(todo)}) -> {OUT}", flush=True)
        t0 = time.time()
        for n, k in enumerate(todo):
            scene_idx, s, e = plan[k]
            arr = load_molmo2_frames(ds.frame_files_list[scene_idx], s, e).numpy()
            tmp = osp.join(OUT, f'{k}.tmp.npy')
            np.save(tmp, arr)
            os.replace(tmp, osp.join(OUT, f'{k}.npy'))      # 반쪽 파일이 skip-done 에 안 걸리게
            n_new += 1
            if (n + 1) % 100 == 0 or n + 1 == len(todo):
                print(f"  [{split}] {n + 1}/{len(todo)} | {(time.time() - t0) / (n + 1):.2f} s/scene",
                      flush=True)
    print(f"[frames] ALL DONE shard {SHARD}/{SHARDS}: wrote {n_new} -> {OUT}", flush=True)
