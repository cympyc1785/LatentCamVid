"""이 코드가 답하는 질문: Molmo2 video 토큰 (프레임 f, 8x8 격자 r,c) 하나하나가 소스 3D 공간의 어디를 보고 있나? (토큰별 3D 점 캐시)

R291 (사용자 2026-10-10 OmniCam 참고 "Molmo2 의 feature 랑 같이 쓸 때 3d 위치가 더 잘 나오지 않을까?").
soft-argmax readout (`peav_readout_aux_mode: softargmax`) 이 subject 위치를 **이 점들의 가중평균**으로 낸다 —
semantic (어느 토큰이 subject 인가) 은 Molmo2 가, 거리는 소스 depth 가 맡는다.

토큰 → 원본 픽셀 (tmp/agent/reader-r291-molmo2-layout.md): 3136 = 49 프레임 (소스 0..48) x 8x8.
각 프레임을 378x378 로 늘린 27x27 패치 → 3x3 pool 9x9 → adaptive_avg_pool2d 8x8 (2x2 셀). 그래서 토큰 (r,c) 의 원본
사각형은 x ∈ [c·W/9, (c+2)·W/9), y ∈ [r·H/9, (r+2)·H/9). 여기서는 그 사각형 안 유효 depth (유한·>0·sky 아님) 의
**중앙값**을 사각형 중심 광선에 놓아 world 점으로 올린다 (경계 픽셀이 앞뒤 표면 사이에 떠도 중앙값은 덜 흔들린다).

입력: <eval_data>/eval_data/recon_and_seg/<video>/ (depths/*.exr z-depth, sky_mask, cameras.npz)
출력: <corpus>/dynpose/<video>/da3/patch_points_8x8.npz  points (49,64,3) float32 world · valid (49,64) bool.
world = recon `cameras.npz` world (target_track 과 같은 좌표계). dataset 이 frame-s 카메라 좌표 / norm_scale 로 바꾼다.
env: vista4d.  예) python scripts/data/cache_patch_points.py --corpus /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 --workers 24
"""
import argparse
import os
import os.path as osp
import sys
from multiprocessing import Pool

import numpy as np

EVAL = "/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K"
VISTA4D = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"
G = 8
A = None


def one(video):
    out = osp.join(A.corpus, "dynpose", video, "da3", "patch_points_8x8.npz")
    if osp.exists(out) and not A.overwrite:
        return video, "skip"
    sys.path.insert(0, VISTA4D)
    from utils.media import load_cameras, load_depths, load_masks   # 영상은 안 읽는다 (load_recon_and_seg 의 대부분 시간)
    folder = osp.join(A.eval_data, "eval_data", "recon_and_seg", video)
    try:
        depth = load_depths(osp.join(folder, "depths"), dtype=np.float16, desc=None).astype(np.float32)
        sky = np.asarray(load_masks(osp.join(folder, "sky_mask"), desc=None)).astype(bool)
        C, K = load_cameras(osp.join(folder, "cameras.npz"))
    except Exception as e:      # noqa: BLE001
        return video, f"error {str(e)[:60]}"
    C, K = np.asarray(C, dtype=np.float64), np.asarray(K, dtype=np.float64)
    F_, H, W = depth.shape
    pts = np.zeros((F_, G * G, 3), np.float32); val = np.zeros((F_, G * G), bool)
    for f in range(F_):
        fx, fy, cx, cy = (K[f][0], K[f][1], K[f][2], K[f][3]) if K.ndim == 2 else (K[f][0, 0], K[f][1, 1], K[f][0, 2], K[f][1, 2])
        d = depth[f]; ok = np.isfinite(d) & (d > 0) & ~sky[f]
        for r in range(G):
            y0, y1 = int(round(r * H / 9)), int(round((r + 2) * H / 9))
            for c in range(G):
                x0, x1 = int(round(c * W / 9)), int(round((c + 2) * W / 9))
                m = ok[y0:y1, x0:x1]
                if m.mean() < 0.25:
                    continue
                z = float(np.median(d[y0:y1, x0:x1][m]))
                u, v = (c + 1) * W / 9 - 0.5, (r + 1) * H / 9 - 0.5
                pc = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z, 1.0])
                pts[f, r * G + c] = (C[f] @ pc)[:3]; val[f, r * G + c] = True
    np.savez_compressed(out, points=pts, valid=val)
    return video, f"ok {int(val.sum())}/{val.size}"


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--eval_data", default=EVAL)
    ap.add_argument("--videos", default=None)       # 파일 (기본: <corpus>/dynpose 전부)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    A = ap.parse_args()
    vids = (open(A.videos).read().split() if A.videos else sorted(os.listdir(osp.join(A.corpus, "dynpose"))))
    bad = 0
    with Pool(A.workers) as p:
        for i, (v, msg) in enumerate(p.imap_unordered(one, vids, chunksize=4)):
            bad += msg.startswith("error")
            if i % 500 == 0 or msg.startswith("error"):
                print(f"[{i}/{len(vids)}] {v} {msg}", flush=True)
    print(f"=== done {len(vids)} (error {bad}) ===", flush=True)


if __name__ == "__main__":
    main()
