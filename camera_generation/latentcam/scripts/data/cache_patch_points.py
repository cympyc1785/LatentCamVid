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

[R582] `--mode scene4d` — 같은 소스 depth 로 **조건용 4D 점 구름** (사용자 10-27 "dynamic, static 분리 · 8x8 중앙값은 sparse · 동적도 dense").
  정적: 프레임마다 dynamic·sky 를 뺀 픽셀을 b x b 블록 **최소 depth** (얇은 장애물 보존) 로 올려 49 프레임을 world 에서 합치고,
        voxel (중앙 depth / vox_div) 에서 **2 프레임 이상** 맞은 칸만 남긴다 (최소값이 고르는 경계 flying pixel 은 시점마다 흩어져 빠진다) → FPS n_static.
  동적: 프레임마다 dynamic_mask 픽셀을 b x b 블록 중앙값 (mask 안 픽셀만) 으로 올려 FPS n_dyn (시간축 유지).
  출력: <corpus>/dynpose/<video>/da3/scene_points4d.npz  static (Ns,3) f16 · static_valid (Ns,) · dyn (49,Nd,3) f16 · dyn_valid (49,Nd) — world.
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


def _fps(p, n, rng):
    """farthest point sampling (numpy). 점이 n 이하면 그대로."""
    if len(p) <= n:
        return p
    if len(p) > 20 * n:
        p = p[rng.choice(len(p), 20 * n, replace=False)]
    idx = np.zeros(n, np.int64); idx[0] = rng.integers(len(p)); dist = np.full(len(p), np.inf)
    for i in range(1, n):
        dist = np.minimum(dist, ((p - p[idx[i - 1]]) ** 2).sum(1)); idx[i] = int(dist.argmax())
    return p[idx]


def _unproject(z, us, vs, Kf, Cf):
    fx, fy, cx, cy = (Kf[0], Kf[1], Kf[2], Kf[3]) if Kf.ndim == 1 else (Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2])
    pc = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z, np.ones_like(z)], 1)
    return (pc @ Cf.T)[:, :3]


def one_scene4d(video):
    out = osp.join(A.corpus, "dynpose", video, "da3", "scene_points4d.npz")
    if osp.exists(out) and not A.overwrite:
        return video, "skip"
    sys.path.insert(0, VISTA4D)
    from utils.media import load_cameras, load_depths, load_masks
    folder = osp.join(A.eval_data, "eval_data", "recon_and_seg", video)
    try:
        depth = load_depths(osp.join(folder, "depths"), dtype=np.float16, desc=None).astype(np.float32)
        sky = np.asarray(load_masks(osp.join(folder, "sky_mask"), desc=None)).astype(bool)
        dyn = np.asarray(load_masks(osp.join(folder, "dynamic_mask"), desc=None)).astype(bool)[:len(depth)]
        C, K = load_cameras(osp.join(folder, "cameras.npz"))
    except Exception as e:      # noqa: BLE001
        return video, f"error {str(e)[:60]}"
    C, K = np.asarray(C, dtype=np.float64), np.asarray(K, dtype=np.float64)
    F_, H, W = depth.shape; b = A.block; h, w = H // b, W // b
    rng = np.random.default_rng(0)
    ok = np.isfinite(depth) & (depth > 0) & ~sky
    vv, uu = np.meshgrid((np.arange(h) + 0.5) * b - 0.5, (np.arange(w) + 0.5) * b - 0.5, indexing="ij")
    st_pts, st_f = [], []
    dyn_out = np.zeros((F_, A.n_dyn, 3), np.float16); dyn_val = np.zeros((F_, A.n_dyn), bool)
    for f in range(F_):
        d = depth[f, :h * b, :w * b].reshape(h, b, w, b)
        ms = (ok[f] & ~dyn[f])[:h * b, :w * b].reshape(h, b, w, b)
        zmin = np.where(ms, d, np.inf).min((1, 3)); cnt = ms.sum((1, 3))
        sel = np.isfinite(zmin) & (cnt >= max(2, b * b // 8))
        if sel.any():
            st_pts.append(_unproject(zmin[sel], uu[sel], vv[sel], K[f], C[f])); st_f.append(np.full(int(sel.sum()), f))
        md = (ok[f] & dyn[f])[:h * b, :w * b].reshape(h, b, w, b)
        cd = md.sum((1, 3)); seld = cd >= max(2, b * b // 4)
        if seld.any():
            zd = np.nanmedian(np.where(md, d, np.nan).transpose(0, 2, 1, 3).reshape(h, w, -1)[seld], 1)
            pd = _fps(_unproject(zd, uu[seld], vv[seld], K[f], C[f]), A.n_dyn, rng)
            dyn_out[f, :len(pd)] = pd; dyn_val[f, :len(pd)] = True
    static = np.zeros((A.n_static, 3), np.float16); sval = np.zeros(A.n_static, bool); nvox = 0
    if st_pts:
        P = np.concatenate(st_pts); Fi = np.concatenate(st_f)
        vox = float(np.median(depth[ok])) / A.vox_div
        key = np.floor(P / vox).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.reshape(-1)
        nfr = np.zeros(inv.max() + 1, np.int64)
        u_pair = np.unique(np.stack([inv, Fi], 1), axis=0); np.add.at(nfr, u_pair[:, 0], 1)
        keep = nfr >= 2
        csum = np.zeros((len(nfr), 3)); np.add.at(csum, inv, P); cnum = np.bincount(inv, minlength=len(nfr))
        cent = (csum / cnum[:, None])[keep]; nvox = len(cent)
        ps = _fps(cent, A.n_static, rng); static[:len(ps)] = ps; sval[:len(ps)] = True
    np.savez_compressed(out, static=static, static_valid=sval, dyn=dyn_out, dyn_valid=dyn_val)
    return video, f"ok static {int(sval.sum())} (vox {nvox}) dyn/frame {dyn_val.sum(1).mean():.0f}"


def one(video):
    if A.mode == "scene4d":
        return one_scene4d(video)
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
    ap.add_argument("--mode", choices=["patch8", "scene4d"], default="patch8")   # [R582] scene4d = 조건용 정적·동적 점 구름
    ap.add_argument("--block", type=int, default=8)          # scene4d 픽셀 블록 (720x1280 → 90x160)
    ap.add_argument("--vox_div", type=float, default=100.0)  # 정적 voxel = 중앙 depth / vox_div
    ap.add_argument("--n_static", type=int, default=4096)
    ap.add_argument("--n_dyn", type=int, default=256)        # 프레임당
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
