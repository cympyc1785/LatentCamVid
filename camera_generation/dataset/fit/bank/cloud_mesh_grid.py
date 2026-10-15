"""이 코드가 답하는 질문: mesh 가 없는 in-the-wild 영상 (DynPose · Vista) 에서 뱅크 fit 의 충돌 게이트를 ESDF (부피) 로 걸려면?

R405 (사용자 2026-10-15 "ESDF 로 parkour 를 track orbit left preset fitting 해줘봐"). TRUMANS mesh 용 `lbm.mesh_collision` 격자
(`trumans_mesh_grid_v1`: 정적 EDT · 소스 카메라에서 닿는 free 성분 · 프레임별 동적 EDT) 와 **같은 형식**을 4D 점구름으로 굽는다.
그러면 fit 의 `--collision_source mesh|both` 경로가 코드 변경 없이 이 격자를 쓴다 (depth shell 이 놓치는 "표면 바로 앞 / 비스듬히 스침" 을 잡는다, R374·R404).

  정적 점유 = 소스 depth 점 중 하늘 아님 · 동적 mask (dilate `--dyn_dilate` px) 밖 · depth 불연속 (3x3 max/min > `--edge_ratio`) 아님 — R374 GT 오검출
              원인 (동적 경계 누출 · flying pixel) 을 뺀다
  동적 = 프레임마다 동적 mask 안 점 (불연속 제외) → 그 프레임 EDT
  좌표 = recon world (= 뱅크 world: `lbm.render.Renderer.cam_c2w_src` 가 recon cameras 그대로) → anchor_c2w = I, cam_centers = recon 카메라 중심
env vista4d.  예) python fit/bank/cloud_mesh_grid.py --video parkour --eval_data /data1/cympyc1785/data/Vista4D-Eval-Data --out_root out_r404
"""
import argparse
import json
import sys
from os import path

import numpy as np

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0, HERE)
from lbm.mesh_collision import GRID_FORMAT, _edt, reachable_from  # noqa: E402


def build(recon, S, out, voxel_frac=0.02, pix_stride=2, dyn_dilate=7, edge_ratio=1.15, pad_frac=0.5, clip_frac=1.0):
    import cv2
    D = np.asarray(recon["depths"], np.float32); T, H, W = D.shape
    sky = np.asarray(recon["sky_mask"], bool); dyn = np.asarray(recon["dynamic_mask"], bool)
    Cw = np.asarray(recon["cam_c2w"], np.float64); K = np.asarray(recon["intrinsics"], np.float64)
    if K.ndim == 2:
        K = np.stack([np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]) for k in K])
    vv, uu = np.mgrid[0:H:pix_stride, 0:W:pix_stride]; ker = np.ones((2 * dyn_dilate + 1,) * 2, np.uint8)
    stat, dpts = [], []
    for t in range(T):
        d = D[t]; ok = np.isfinite(d) & (d > 0) & ~sky[t]; dd = np.where(ok, d, np.nan)
        mx = cv2.dilate(np.nan_to_num(dd, nan=0), np.ones((3, 3))); mn = -cv2.dilate(-np.nan_to_num(dd, nan=1e9), np.ones((3, 3)))
        edge = (mx / np.maximum(mn, 1e-6)) > edge_ratio; nd = cv2.dilate(dyn[t].astype(np.uint8), ker).astype(bool)
        z = d[vv, uu]; base = ok[vv, uu] & ~edge[vv, uu]
        Xc = np.stack([(uu + 0.5 - K[t][0, 2]) / K[t][0, 0] * z, (vv + 0.5 - K[t][1, 2]) / K[t][1, 1] * z, z], -1)
        Xw = Xc @ Cw[t][:3, :3].T + Cw[t][:3, 3]
        stat.append(Xw[base & ~nd[vv, uu]]); dpts.append(Xw[base & dyn[t][vv, uu]])
    P = np.concatenate(stat); cams = Cw[:, :3, 3]; voxel = voxel_frac * S; pad = pad_frac * S; clip = clip_frac * S
    lo = np.minimum(np.percentile(P, 1, 0), cams.min(0)) - pad; hi = np.maximum(np.percentile(P, 99, 0), cams.max(0)) + pad
    voxel = max(voxel, float((hi - lo).max()) / 400)                   # 축 최대 400 칸
    s_origin = np.floor(lo / voxel) * voxel; s_shape = tuple(np.ceil((hi - s_origin) / voxel).astype(int) + 1)
    occ = np.zeros(s_shape, bool); ij = np.floor((P - s_origin) / voxel).astype(int); m = ((ij >= 0) & (ij < s_shape)).all(1); occ[tuple(ij[m].T)] = True
    free = ~occ; seeds = np.floor((cams - s_origin) / voxel).astype(np.int64)
    reach = reachable_from(free, seeds); edt_s = _edt(free, voxel, clip)
    Dall = np.concatenate([p for p in dpts if len(p)]) if any(len(p) for p in dpts) else np.zeros((0, 3))
    if len(Dall):
        dlo, dhi = np.percentile(Dall, 0.5, 0) - pad, np.percentile(Dall, 99.5, 0) + pad
        d_origin = np.floor(dlo / voxel) * voxel; d_shape = tuple(np.ceil((dhi - d_origin) / voxel).astype(int) + 1)
        edt_d = []
        for p in dpts:
            g = np.zeros(d_shape, bool)
            if len(p):
                ij = np.floor((p - d_origin) / voxel).astype(int); m = ((ij >= 0) & (ij < d_shape)).all(1); g[tuple(ij[m].T)] = True
            edt_d.append(_edt(~g, voxel, clip) if g.any() else np.full(d_shape, clip, np.float16))
        edt_d = np.stack(edt_d)
    else:
        d_origin, d_shape = s_origin.copy(), (1, 1, 1); edt_d = np.full((T, 1, 1, 1), clip, np.float16)
    np.savez_compressed(out, format=GRID_FORMAT, source="cloud_mesh_grid (4D 점구름)", voxel=float(voxel), pad=float(pad), clip=float(clip),
                        frame_list=np.arange(T, dtype=np.int32), static_origin=s_origin, static_shape=np.asarray(s_shape, np.int32),
                        static_edt=edt_s, reachable=np.packbits(reach), dyn_origin=d_origin, dyn_shape=np.asarray(d_shape, np.int32), dyn_edt=edt_d,
                        cam_centers=cams, anchor_c2w=np.eye(4))
    return dict(voxel_over_S=voxel / S, static_shape=s_shape, occupied=int(occ.sum()), reachable_frac=float(reach.mean()), dyn_shape=d_shape,
                src_clear_min_over_S=float(edt_s[tuple(np.clip(seeds, 0, np.array(s_shape) - 1).T)].astype(float).min() / S), out=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--eval_data", default="/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K")
    ap.add_argument("--out_root", required=True)                     # <out_root>/<video>/scene_graph.json (S) · mesh_grid.npz 저장
    ap.add_argument("--vista4d_root", default="/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    ap.add_argument("--voxel_frac", type=float, default=0.02)
    ap.add_argument("--pix_stride", type=int, default=2)
    ap.add_argument("--dyn_dilate", type=int, default=7)
    ap.add_argument("--edge_ratio", type=float, default=1.15)
    a = ap.parse_args()
    from scene_graph.io import import_vista4d
    v4 = import_vista4d(a.vista4d_root)
    recon = v4["load_recon_and_seg"](path.join(a.eval_data, "eval_data", "recon_and_seg", a.video))
    S = float(json.load(open(path.join(a.out_root, a.video, "scene_graph.json")))["scale"]["S"])
    info = build(recon, S, path.join(a.out_root, a.video, "mesh_grid.npz"), a.voxel_frac, a.pix_stride, a.dyn_dilate, a.edge_ratio)
    print(json.dumps(info, default=str))


if __name__ == "__main__":
    main()
