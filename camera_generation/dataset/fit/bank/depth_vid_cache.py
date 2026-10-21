"""이 코드가 답하는 질문: 카메라 모델에 소스 영상 depth 를 voxel 화 없이 (동적 물체 포함) 시간 축 그대로 조건으로 주려면 무엇을 굽나?

R482 (사용자 2026-10-21 "v1 으로 voxelize 없이 dynamic 도 포함해서 그냥 depth 그 자체들에서 time 축 고려해서 conv 로 압축하고 resampling 해서
condition 으로"). 씬마다 소스 전 프레임 DA3 depth (`<eval_data>/recon_and_seg/<video>/depths`) 를 (T, h, w) 로 줄여
`<corpus>/dynpose/<video>/da3/<name>` (키 `depth`, float16, 하늘·무효 = 0) 에 둔다 — 코퍼스에는 파일을 더하기만 한다.
축소: 유효 픽셀 평균 (cv2.INTER_AREA 로 depth·valid 를 각각 줄이고 나눔), 칸의 유효 비율 < 0.5 면 0. 정규화 (/ norm_scale, log1p) 는 학습 쪽
(`dataset_dl3dv._depth_vid`) 이 세그먼트별로 한다.
env vista4d. 예: python fit/bank/depth_vid_cache.py --corpus /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 --workers 16
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np


def one(job):
    a, v = job
    dst = osp.join(a.corpus, "dynpose", v, "da3", a.name)
    if osp.isfile(dst) and not a.overwrite:
        return v, "skip"
    sys.path.insert(0, a.vista4d_root)
    import cv2
    from utils.media import load_depths, load_masks
    src = osp.join(a.eval_data, v)
    try:
        d = load_depths(osp.join(src, "depths"), dtype=np.float16, desc=None).astype(np.float32)
        sky = np.asarray(load_masks(osp.join(src, "sky_mask"), desc=None)).astype(bool) if osp.isdir(osp.join(src, "sky_mask")) else np.zeros(d.shape, bool)
    except Exception as e:  # noqa: BLE001
        return v, f"error {str(e)[:80]}"
    out = np.zeros((d.shape[0], a.h, a.w), np.float16)
    for t in range(d.shape[0]):
        ok = np.isfinite(d[t]) & (d[t] > 0) & ~sky[t]
        dv = np.where(ok, d[t], 0).astype(np.float32); m = ok.astype(np.float32)
        ds = cv2.resize(dv, (a.w, a.h), interpolation=cv2.INTER_AREA); ms = cv2.resize(m, (a.w, a.h), interpolation=cv2.INTER_AREA)
        out[t] = np.where(ms >= 0.5, ds / np.maximum(ms, 1e-6), 0).astype(np.float16)
    np.savez(dst, depth=out)
    return v, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--eval_data", default="/data1/cympyc1785/data/DynPose-100K/eval_data/recon_and_seg")
    ap.add_argument("--vista4d_root", default="/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    ap.add_argument("--name", default="depth_vid36x64.npz")
    ap.add_argument("--h", type=int, default=36)
    ap.add_argument("--w", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    vids = sorted(v for v in os.listdir(osp.join(a.corpus, "dynpose")) if osp.isdir(osp.join(a.corpus, "dynpose", v, "da3")))
    vids = vids[:a.limit] if a.limit else vids
    from multiprocessing import Pool
    t0 = time.time(); bad = 0
    with Pool(a.workers) as p:
        for i, (v, msg) in enumerate(p.imap_unordered(one, [(a, v) for v in vids], chunksize=4)):
            bad += msg.startswith("error")
            if i % 500 == 0 or msg.startswith("error"):
                print(f"[{i}/{len(vids)}] {v} {msg} {time.time() - t0:.0f}s", flush=True)
    print(f"=== depth_vid DONE {len(vids)} (error {bad}) ===", flush=True)


if __name__ == "__main__":
    main()
