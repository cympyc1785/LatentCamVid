"""이 코드가 답하는 질문: 학습 코퍼스의 GT 카메라가 ESDF 로 충돌하나 (GT 정제 목록), 그리고 카메라 모델에 줄 3D 점유 격자 조건은 어떻게 굽나?

R425 (사용자 2026-10-16 "v1 으로 3D 점유 격자 방식으로 추천 세팅으로 Collision 개선 학습 해보자. 우선순위 높은거 두개 먼저").
씬 하나에 `eval/pred_collision_rate.esdf_scene` (R374 평가 정의 그대로: 정적 occupied = 동적·하늘 아닌 depth 점 voxel, free = 소스 ray 가
지난 voxel, unknown = 나머지, voxel = vox_frac·S) 를 **한 번** 돌려서 두 가지를 같이 낸다:

  judge  세그먼트 (prompts.json 키) 마다 GT 궤적 ESDF 판정 (coll_any · coll_frac · 정적/동적 · unk_frac) → <out>/judge/<video>.json
  grid   세그먼트마다 **frame-s (target frame 0) 카메라 좌표 / norm_scale** 의 정육면체 [-L, L]³ 를 N³ 칸으로 — 칸마다 occupied · free ·
         unknown 비율 (칸 안 sub³ 점 trilinear 평균, uint8 0..255) → <scene>/da3/<grid_name> (키 = 세그먼트, 값 (3,N,N,N) uint8).
         좌표·분모는 dataset_dl3dv._track_cond 와 같다 (q = (E_s @ p_w)[:3] / norm_scale) — 모델이 카메라 t · subject track 과 같은 자로 읽는다.
         norm_scale = <scene>/da3/<avg_scale_dir>/<seg>.json (학습 avg_scale_ref 와 같아야 한다, 기본 context_first_cam).
코퍼스에는 **파일을 더하기만** 한다 (기존 파일 안 건드림).

env vista4d (GPU: free 투영). 샤딩: --num_shards/--shard_id.
  python fit/bank/occ_grid_cache.py --corpus /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 --out /data1/.../tmp/r425 --shard_id 0 --num_shards 8
  python fit/bank/occ_grid_cache.py --stage summary --out /data1/.../tmp/r425      # judge 집계 + 정제 seg list
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

import numpy as np

HERE = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, HERE)


def seg_grids(occ, free, lo, vox, ext_list, ns_list, n=32, half=2.0, sub=3):
    """world 격자 (occ/free bool) → 세그먼트별 (3,n,n,n) uint8. ext = target w2c (T,4,4), ns = norm_scale."""
    from scipy.ndimage import map_coordinates
    occ_f, free_f = occ.astype(np.float32), free.astype(np.float32)
    cell = 2 * half / n
    o = (np.arange(n * sub) + 0.5) / (n * sub) * 2 * half - half                       # sub-sample 중심 (frame-s / ns 단위)
    Q = np.stack(np.meshgrid(o, o, o, indexing="ij"), -1).reshape(-1, 3)               # (n·sub)³ × 3, 축 순서 x,y,z
    out = []
    for E, ns in zip(ext_list, ns_list):
        Rs, ts = E[0][:3, :3], E[0][:3, 3]
        Pw = (Q * ns - ts) @ Rs                                                         # = Rsᵀ (q·ns − t_s)
        idx = ((Pw - lo) / vox - 0.5).T                                                 # voxel 중심 기준 연속 index
        a = map_coordinates(occ_f, idx, order=1, mode="constant", cval=0.0)
        b = map_coordinates(free_f, idx, order=1, mode="constant", cval=0.0)
        a = a.reshape(n, sub, n, sub, n, sub).mean((1, 3, 5)); b = b.reshape(n, sub, n, sub, n, sub).mean((1, 3, 5))
        c = np.clip(1.0 - a - b, 0, 1)
        out.append(np.rint(np.stack([a, b, c]) * 255).astype(np.uint8))
    return out, cell


def scene(video, a, M, v4):
    sd = osp.join(a.corpus, a.seg_prefix, video, "da3")
    gp = osp.join(sd, a.grid_name); jp = osp.join(a.out, "judge", f"{video}.json")
    if osp.isfile(jp) and (a.no_grid or osp.isfile(gp)) and not a.overwrite:
        return "skip"
    z = np.load(osp.join(sd, "target_poses.npz"), allow_pickle=True)
    keys = [str(k) for k in z["keys"]]; ext = z["extrinsics"].astype(np.float64)
    recon = v4["load_recon_and_seg"](osp.join(a.eval_data, "eval_data", "recon_and_seg", video))
    S = float(json.load(open(osp.join(a.graph_root, video, "scene_graph.json")))["scale"]["S"])
    K = np.asarray(recon["intrinsics"], np.float64)
    if K.ndim == 2:
        K = np.stack([np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]) for k in K])
    trajs = [("gt", k, np.linalg.inv(E)) for k, E in zip(keys, ext)]
    keep = {}
    rows = M.esdf_scene(recon, K, S, trajs, keep=keep)
    os.makedirs(osp.dirname(jp), exist_ok=True)
    json.dump({"video": video, "S": S, "rows": rows}, open(jp, "w"))
    if a.no_grid:
        return "judge"
    ns = []
    for k in keys:
        f = osp.join(sd, a.avg_scale_dir, f"{k}.json")
        ns.append(float(json.load(open(f))) if osp.isfile(f) else np.nan)
    ok = [i for i, x in enumerate(ns) if np.isfinite(x) and x > 0]
    grids, cell = seg_grids(keep["occ"], keep["free"], keep["lo"], keep["vox"], [ext[i] for i in ok], [ns[i] for i in ok],
                            a.grid_n, a.grid_half, a.grid_sub)
    np.savez_compressed(gp + ".tmp.npz", **{keys[i]: g for i, g in zip(ok, grids)},
                        __meta__=np.array(json.dumps({"n": a.grid_n, "half": a.grid_half, "cell": cell, "sub": a.grid_sub, "channels": ["occupied", "free", "unknown"],
                                                       "frame": "target frame-s camera (OpenCV) / norm_scale", "avg_scale_dir": a.avg_scale_dir,
                                                       "vox_over_S": keep["vox"] / S, "source": "occ_grid_cache.py (esdf_scene)"})))
    os.replace(gp + ".tmp.npz", gp)
    return f"ok {len(ok)}/{len(keys)}"


def summary(a):
    """judge 집계 → 정제 seg list (train/test 원 목록에서 ESDF 충돌 세그먼트 제외)."""
    rows = []
    for f in glob.glob(osp.join(a.out, "judge", "*.json")):
        d = json.load(open(f))
        for r in d["rows"]:
            if "error" not in r:
                rows.append({**r, "video": d["video"]})
    print(f"[summary] 세그먼트 {len(rows)} · 씬 {len({r['video'] for r in rows})}")
    cf = np.array([r["coll_frac"] for r in rows]); cs = np.array([r["coll_static_frac"] for r in rows]); cd = np.array([r["coll_dyn_frac"] for r in rows])
    for th in (1, 3, 5, 10):
        print(f"  coll 프레임 ≥ {th:2d}: {np.mean(cf * 49 >= th - 1e-6):.4f}  (정적 {np.mean(cs * 49 >= th - 1e-6):.4f} · 동적 {np.mean(cd * 49 >= th - 1e-6):.4f})")
    drop = {f"{a.seg_prefix}/{r['video']}/{r['entry']}" for r in rows if r["coll_frac"] * 49 >= a.drop_min_frames - 1e-6}
    for split in ("train", "test"):
        src = osp.join(a.corpus, f"seg_list_{a.seg_prefix}_{a.seg_list_tag}_{split}.txt")
        if not osp.isfile(src):
            continue
        L = [x for x in open(src).read().split("\n") if x.strip()]
        keepL = [x for x in L if x.strip() not in drop]
        dst = osp.join(a.out, f"seg_list_{a.seg_prefix}_{a.seg_list_tag}_esdf{a.drop_min_frames}_{split}.txt")
        open(dst, "w").write("\n".join(keepL) + "\n")
        print(f"  {split}: {len(L)} → {len(keepL)} (제외 {len(L) - len(keepL)}) → {dst}")
    json.dump(sorted(drop), open(osp.join(a.out, f"drop_esdf{a.drop_min_frames}.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="build", choices=["build", "summary"])
    ap.add_argument("--corpus", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200")
    ap.add_argument("--seg_prefix", default="dynpose")
    ap.add_argument("--seg_list_tag", default="s91")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_data", default="/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K")
    ap.add_argument("--graph_root", default=osp.join(HERE, "out_dynpose"))
    ap.add_argument("--vista4d_root", default="/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    ap.add_argument("--videos_file", default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--grid_name", default="occ_grid32.npz")
    ap.add_argument("--grid_n", type=int, default=32)
    ap.add_argument("--grid_half", type=float, default=2.0)       # 정육면체 반폭 (norm_scale 단위). v1 train 카메라 중심 |c| 세그먼트 최대 95% 1.49 · 99% 2.36
    ap.add_argument("--grid_sub", type=int, default=3)
    ap.add_argument("--avg_scale_dir", default="avg_scale_context_first_cam")   # 학습 avg_scale_ref=context_first_cam 과 같게
    ap.add_argument("--no_grid", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--drop_min_frames", type=int, default=3)    # summary: ESDF 충돌 프레임이 이 이상이면 정제 목록에서 뺀다
    # esdf_scene 인자 (pred_collision_rate 기본값 그대로)
    ap.add_argument("--vox_frac", type=float, default=0.02)
    ap.add_argument("--radius_frac", type=float, default=0.02)
    ap.add_argument("--max_dim", type=int, default=256)
    ap.add_argument("--pix_stride", type=int, default=2)
    ap.add_argument("--src_free", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "summary":
        return summary(a)
    import eval.pred_collision_rate as M
    M.ARGS = argparse.Namespace(vox_frac=a.vox_frac, radius_frac=a.radius_frac, max_dim=a.max_dim, pix_stride=a.pix_stride, src_free=a.src_free, knn_down=1)
    from scene_graph.io import import_vista4d
    v4 = import_vista4d(a.vista4d_root)
    vids = (open(a.videos_file).read().split() if a.videos_file else sorted(os.listdir(osp.join(a.corpus, a.seg_prefix))))
    vids = [v for i, v in enumerate(vids) if i % a.num_shards == a.shard_id]
    t0 = time.time()
    for i, v in enumerate(vids):
        try:
            st = scene(v, a, M, v4)
        except Exception as e:                                       # noqa: BLE001 — 씬 하나 실패는 그 씬만
            st = f"ERR {type(e).__name__}: {str(e)[:120]}"
        if i % 20 == 0 or st.startswith("ERR"):
            print(f"[{a.shard_id}] {i + 1}/{len(vids)} {v[:8]} {st}  {(time.time() - t0) / (i + 1):.1f}s/씬", flush=True)
    print(f"=== occ_grid shard {a.shard_id} DONE ===", flush=True)


if __name__ == "__main__":
    main()
