"""video_onfly=molmo2_conn arm 의 frame 디코드를 매 스텝 안 하려면 씬당 무엇을 미리 구워 두나 (R58 / D286).

답: 씬 frame [s, e) 49장을 Molmo2 전처리 앞단(378x378 squash, bilinear, antialias=False) 까지 한
**uint8** 배열 `<OUT>/<scene_key>.npy` (T,378,378,3) ≈ 21 MB. ViT feature 캐시가 아니다 — ViT 는 학습
중 계속 frozen on-the-fly 로 돈다 (사용자 결정 2026-09-25: 27x27 feature 캐시 1.67 TB 는 안 함,
frame 캐시 + worker 증설은 채택). dataset 은 `video_onfly_frame_cache` 가 있고 파일이 있으면 이걸
mmap 으로 읽고, 없으면 PNG 에서 같은 함수(`load_molmo2_frames`)로 만든다 — 값이 비트 동일하다.

사용 (CPU 전용, GPU 불필요):
  EXP=dynpose_d200_siglip2conn_srccam SHARDS=8 SHARD=0 \
    /data1/cympyc1785/miniconda3/envs/latentcam/bin/python scripts/data/cache_molmo2_frames.py

MODE=feat (2026-09-27, R58 사용자 지시 "R58을 caching해놓고 이어서 돌려줘"): frame 캐시를 읽어
frozen ViT 까지 돌린 출력 (49,729,2304) bf16 을 uint16 으로 `<OUT>/<scene_key>.npy` 에 굽는다
(≈164 MB/씬, 10,169 씬 ≈ 1.67 TB). 학습은 `video_onfly_feat_cache` 가 있으면 ViT 를 건너뛴다.
frame 캐시에 없는 씬은 PNG 경로로 만든다. GPU 필요:
  MODE=feat CUDA_VISIBLE_DEVICES=1 SHARDS=3 SHARD=0 .../python scripts/data/cache_molmo2_frames.py
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
MODE = os.environ.get("MODE", "frames")
assert 0 <= SHARD < SHARDS and MODE in ("frames", "feat")

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
FRAME_CACHE = getattr(cfg, 'video_onfly_frame_cache', None)
OUT = os.environ.get("OUT") or getattr(
    cfg, 'video_onfly_frame_cache' if MODE == "frames" else 'video_onfly_feat_cache', None)
assert OUT, "OUT 또는 cfg.video_onfly_{frame,feat}_cache 가 필요하다"
cfg.video_onfly_frame_cache = None          # 굽는 동안에는 PNG 경로를 타야 한다
cfg.video_onfly_feat_cache = None

from dataset_dl3dv import CamDataset                              # noqa: E402
from models.molmo2_video_connector import load_molmo2_frames      # noqa: E402

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    n_new = 0
    vit = None
    if MODE == "feat":
        import torch
        from models.molmo2_video_connector import FrozenMolmo2ViT
        vit = FrozenMolmo2ViT().cuda()
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
            if MODE == "frames":
                arr = load_molmo2_frames(ds.frame_files_list[scene_idx], s, e).numpy()
            else:
                fp = osp.join(FRAME_CACHE, f'{k}.npy') if FRAME_CACHE else None
                fr = (torch.from_numpy(np.load(fp)) if fp and osp.exists(fp)
                      else load_molmo2_frames(ds.frame_files_list[scene_idx], s, e))
                feat = vit(fr[None].cuda())[0]                      # (T,729,2304) bf16
                arr = feat.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            tmp = osp.join(OUT, f'{k}.tmp.npy')
            np.save(tmp, arr)
            os.replace(tmp, osp.join(OUT, f'{k}.npy'))      # 반쪽 파일이 skip-done 에 안 걸리게
            n_new += 1
            if (n + 1) % 100 == 0 or n + 1 == len(todo):
                print(f"  [{split}] {n + 1}/{len(todo)} | {(time.time() - t0) / (n + 1):.2f} s/scene",
                      flush=True)
    print(f"[{MODE}] ALL DONE shard {SHARD}/{SHARDS}: wrote {n_new} -> {OUT}", flush=True)
