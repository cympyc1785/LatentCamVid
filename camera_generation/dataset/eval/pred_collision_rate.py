"""이 코드가 답하는 질문: 모델이 낸 카메라 궤적이 씬 표면을 얼마나 뚫고 들어가나? (예측 궤적 충돌률, GT 와 같은 식)

R286 (사용자 2026-10-09 "CLaTr 지표, 충돌률 같은 것도 같이 봐줘"). 뱅크 fit 의 collision 게이트와 **같은 함수**
(`lbm.gates.behind_profile`, depth shell G1) 를 eval 산출에 건다 — 예측 카메라 중심이 소스 49 프레임 depth 가 본
표면보다 뒤(= 물체 안)에 있으면 그 프레임은 충돌. 인자도 뱅크 기본값 그대로 (margin 0.02·S, min_zcam 0.02·S,
time_match: 정적 채널은 49 소스 프레임 전부·동적 픽셀 제외, 동적 채널은 시간이 맞는 소스 프레임 한 장).

입력: `eval_testset.py` 산출 `<eval_dir>/test/<name>_transforms_{pred,ref}.json` — scene world (= recon `cameras.npz`
world, `traj_time_metrics` 의 framing 이 같은 가정으로 subject 를 투영한다) 의 OpenGL c2w. OpenCV 로 바꿔서 쓴다.
씬: `<eval_data>/eval_data/recon_and_seg/<video>` (depth · sky · dynamic mask · 소스 카메라), S = graph `scale.S`.

출력: <out>/per_entry.csv (arm, entry, coll_any, coll_frac, coll_static_frac, coll_dyn_frac) + summary.json.
GT (`ref`) 도 같은 식으로 잰다 — 뱅크가 collision 게이트를 통과시킨 궤적이라 거의 0 이어야 하고, 아니면 좌표가 틀린 것.
env: vista4d.  예) python eval/pred_collision_rate.py --arm F2=<eval_dir> --arm MST=<eval_dir> --out <dir> [--workers 16]
"""
import argparse
import csv
import json
import os
import os.path as osp
import sys
from collections import defaultdict
from glob import glob
from multiprocessing import Pool

import numpy as np

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
EVAL_DATA = "/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K"
GRAPH = osp.join(HERE, "out_dynpose")
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
ARGS = None


def load_c2w(p):
    return np.array([f["transform_matrix"] for f in json.load(open(p))["frames"]], dtype=np.float64) @ GL2CV


def scene_job(job):
    """한 씬의 모든 (arm, entry) 를 잰다 — depth 를 한 번만 읽는다."""
    video, items = job
    from lbm.gates import behind_profile
    from scene_graph.io import import_vista4d
    v4 = import_vista4d(ARGS.vista4d_root)
    try:
        recon = v4["load_recon_and_seg"](osp.join(ARGS.eval_data, "eval_data", "recon_and_seg", video))
        S = float(json.load(open(osp.join(ARGS.graph_root, video, "scene_graph.json")))["scale"]["S"])
    except Exception as e:      # noqa: BLE001 — 씬 하나가 없으면 그 씬만 빠진다
        return [{"arm": a, "entry": n, "error": str(e)[:80]} for a, n, _ in items]
    K = np.asarray(recon["intrinsics"], dtype=np.float64)
    if K.ndim == 2:            # (N,4) fx fy cx cy → (N,3,3)
        K = np.stack([np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]) for k in K])
    out = []
    if ARGS.method == "knn":
        return knn_scene(recon, K, S, [(a, n, load_c2w(p)) for a, n, p in items])
    if ARGS.method == "esdf":
        return esdf_scene(recon, K, S, [(a, n, load_c2w(p)) for a, n, p in items])
    for arm, name, path in items:
        poses = load_c2w(path)
        bad, worst, ch = behind_profile(poses, recon["depths"], K=K, cam_c2w=recon["cam_c2w"],
                                        sky_mask=recon["sky_mask"], scale=S, frames=list(range(len(recon["cam_c2w"]))),
                                        margin_frac=ARGS.margin_frac, clear_frac=0.0, radius_px=0,
                                        dynamic_mask=recon["dynamic_mask"].astype(bool), time_match=True, detail=True,
                                        min_zcam_frac=ARGS.min_zcam_frac)
        T = len(poses)
        out.append({"arm": arm, "entry": name, "coll_any": float(bad > 0), "coll_frac": bad / T,
                    "coll_static_frac": ch["static"]["frames"] / T, "coll_dyn_frac": ch["dynamic"]["frames"] / T})
    return out


def _cloud(recon, K):
    """소스 depth → world 점 (정적: 동적·하늘 제외 전 프레임 / 동적: 프레임별), `--pix_stride` 간격."""
    D = np.asarray(recon["depths"], np.float32); T, H, W = D.shape
    sky = np.asarray(recon["sky_mask"], bool); dyn = np.asarray(recon["dynamic_mask"], bool)
    Cw = np.asarray(recon["cam_c2w"], np.float64); st = ARGS.pix_stride
    vv, uu = np.mgrid[0:H:st, 0:W:st]
    pts_s, pts_d = [], []
    for t in range(T):
        d = D[t][vv, uu]; ok = np.isfinite(d) & (d > 0) & ~sky[t][vv, uu]
        Xc = np.stack([(uu + 0.5 - K[t][0, 2]) / K[t][0, 0] * d, (vv + 0.5 - K[t][1, 2]) / K[t][1, 1] * d, d], -1)
        Xw = Xc @ Cw[t][:3, :3].T + Cw[t][:3, 3]
        dm = dyn[t][vv, uu]
        pts_s.append(Xw[ok & ~dm]); pts_d.append(Xw[ok & dm])
    P = np.concatenate(pts_s) if pts_s else np.zeros((0, 3))
    return D, sky, Cw, pts_s, pts_d, P


def knn_scene(recon, K, S, trajs):
    """[R374] `--method knn` — esdf 와 같은 3 상태 판정을 voxel 없이. occupied = 정적 점 KD-tree 최근접 거리 < r (동적은 esdf 와 같음),
    free = 질의 카메라 중심을 소스 프레임마다 투영해 `0 < z < depth − vox_frac·S` 인 프레임이 하나라도 있거나 소스 카메라 반경 r 안,
    나머지 unknown. 격자를 안 만들고 질의 점 (궤적 49 프레임) 만 재므로 esdf 의 voxel 양자화가 없다."""
    from scipy.spatial import cKDTree
    D, sky, Cw, pts_s, pts_d, P = _cloud(recon, K); T, H, W = D.shape
    r = ARGS.radius_frac * S; vox = ARGS.vox_frac * S
    if ARGS.knn_down:                       # 점을 vox 격자 대표점 하나로 줄여 KD-tree 를 짓는다 (원본 6M 점 build 가 병목)
        def dn(q):
            if not len(q):
                return q
            ij = np.floor(q / vox).astype(np.int64); ij -= ij.min(0); n = ij.max(0) + 1
            k = np.unique((ij[:, 0] * n[1] + ij[:, 1]) * n[2] + ij[:, 2])          # 1D 키 unique (axis=0 정렬보다 훨씬 빠름)
            return (np.stack([k // (n[1] * n[2]), k // n[2] % n[1], k % n[2]], 1) + np.floor(q / vox).astype(np.int64).min(0)) * vox + vox / 2
        P = dn(P); pts_d = [dn(q) for q in pts_d]
    tree = cKDTree(P) if len(P) else None; trees = [cKDTree(p) if len(p) else None for p in pts_d]
    Dinf = np.where(np.isfinite(D) & (D > 0) & ~sky, D, np.inf); w2c = np.linalg.inv(Cw)
    out = []
    for arm, name, poses in trajs:
        Tp = len(poses); C = poses[:, :3, 3]
        ds = tree.query(C, distance_upper_bound=r)[0] if tree is not None else np.full(Tp, np.inf)
        x = np.einsum("tij,pj->tpi", w2c[:, :3, :3], C) + w2c[:, None, :3, 3]          # (T_src, Tp, 3)
        z = x[..., 2]; zc = np.maximum(z, 1e-6)
        u = np.floor(K[:, None, 0, 0] * x[..., 0] / zc + K[:, None, 0, 2]).astype(int); v_ = np.floor(K[:, None, 1, 1] * x[..., 1] / zc + K[:, None, 1, 2]).astype(int)
        ins = (z > 1e-6) & (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)
        dd = np.full(z.shape, -1.0); ti, pi = np.nonzero(ins); dd[ti, pi] = Dinf[ti, v_[ti, pi], u[ti, pi]]
        free = (ins & (z < dd - vox)).any(0)
        if ARGS.src_free:
            free |= np.linalg.norm(C[:, None] - Cw[None, :, :3, 3], axis=-1).min(1) < r
        cs = ds < r; cd = np.zeros(Tp, bool)
        for f in range(Tp):
            tr = trees[int(round(f * (T - 1) / max(Tp - 1, 1)))]
            cd[f] = tr is not None and tr.query(C[f], distance_upper_bound=r)[0] < r
        nc = cs | cd; unk = ~nc & ~free; fr = ~nc & free
        out.append({"arm": arm, "entry": name, "coll_any": float(nc.any()), "coll_frac": nc.mean(), "coll_static_frac": cs.mean(),
                    "coll_dyn_frac": cd.mean(), "unk_frac": unk.mean(), "unk_any": float(unk.any()), "free_frac": fr.mean(),
                    "coll_or_unk_any": float((nc | unk).any()), "vox": vox / S, "n_vox": 0, "occ_frac": 0.0, "free_vox_frac": 0.0})
    return out


def esdf_scene(recon, K, S, trajs, keep=None):
    """[R374] `--method esdf` — 4D 점구름을 voxel 로 올려 occupied / free / unknown 세 상태로 판정 (최종 평가용, shell 보다 느림).

    - occupied (정적): 동적·하늘 아닌 depth 점 (전 소스 프레임, `--pix_stride` 간격) 이 든 voxel. ESDF = 가장 가까운 occupied voxel
      까지 거리 (scipy EDT). 카메라 중심의 ESDF < `--radius_frac`·S 면 정적 충돌.
    - occupied (동적): 시간이 맞는 소스 프레임 한 장의 동적 점까지 최근접 거리 < 반경 (shell 의 time_match 와 같은 규약, KD-tree).
    - free: 소스 카메라에서 depth 점까지 쏜 ray 가 지난 voxel. ray 를 하나씩 걷는 대신 voxel 중심을 소스 프레임마다 투영해
      `0 < z_cam < depth(u,v) − voxel` 이면 free (같은 집합의 쌍대 계산, 하늘·무효 depth 픽셀은 ray 가 끝없이 free). 동적 픽셀도
      그 시각의 앞쪽 공간은 비어 있었으므로 free 증거로 쓴다.
    - `--src_free 1` (기본): 소스 카메라 중심 반경 r 안 voxel 도 free (카메라가 실제로 지나간 자리).
    - unknown: 나머지 (어느 소스 프레임에서도 표면 앞으로 보인 적 없는 곳 + 격자 밖).
    판정 순서: 충돌 (정적/동적) > free > unknown. 격자 = 정적 점 축별 1~99% 구간 ∪ 소스·예측 카메라, voxel = `--vox_frac`·S
    (축 최대 `--max_dim` 넘으면 voxel 을 키운다)."""
    import torch
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import cKDTree
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D, sky, Cw, pts_s, pts_d, P = _cloud(recon, K); T, H, W = D.shape
    cams = np.concatenate([Cw[:, :3, 3]] + [p[:, :3, 3] for _, _, p in trajs])
    r = ARGS.radius_frac * S
    lo = np.minimum(np.percentile(P, 1, 0) if len(P) else cams.min(0), cams.min(0)) - 2 * r
    hi = np.maximum(np.percentile(P, 99, 0) if len(P) else cams.max(0), cams.max(0)) + 2 * r
    vox = max(ARGS.vox_frac * S, float((hi - lo).max()) / ARGS.max_dim)
    dims = np.ceil((hi - lo) / vox).astype(int) + 1
    occ = np.zeros(dims, bool)
    ij = np.floor((P - lo) / vox).astype(int); inb = ((ij >= 0) & (ij < dims)).all(1)
    occ[tuple(ij[inb].T)] = True
    esdf = distance_transform_edt(~occ) * vox if occ.any() else np.full(dims, np.inf)
    # free: voxel 중심을 소스 프레임마다 투영 (GPU)
    g = [torch.arange(n, device=dev, dtype=torch.float32) for n in dims]
    ctr = torch.stack(torch.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3) * vox + torch.tensor(lo + vox / 2, device=dev, dtype=torch.float32)
    free = torch.zeros(len(ctr), dtype=torch.bool, device=dev)
    Dt = torch.from_numpy(np.where(np.isfinite(D) & (D > 0) & ~sky, D, np.inf)).to(dev)
    w2c = torch.from_numpy(np.linalg.inv(Cw)).float().to(dev); Kt = torch.from_numpy(K).float().to(dev)
    for t in range(T):
        for c0 in range(0, len(ctr), 1 << 22):
            x = ctr[c0:c0 + (1 << 22)] @ w2c[t, :3, :3].T + w2c[t, :3, 3]
            z = x[:, 2]; zc = z.clamp(min=1e-6)
            u = (Kt[t, 0, 0] * x[:, 0] / zc + Kt[t, 0, 2]).floor().long(); v_ = (Kt[t, 1, 1] * x[:, 1] / zc + Kt[t, 1, 2]).floor().long()
            ins = (z > 1e-6) & (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)
            dd = torch.full_like(z, -1.0); dd[ins] = Dt[t, v_[ins], u[ins]]
            free[c0:c0 + (1 << 22)] |= ins & (z < dd - vox)
    free = free.reshape(*dims).cpu().numpy()
    if ARGS.src_free:                       # 소스 카메라가 지나간 자리 (반경 r) 는 비어 있었다
        n = int(np.ceil(r / vox)); o = np.stack(np.meshgrid(*[np.arange(-n, n + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        o = o[(np.linalg.norm(o, axis=1) * vox) <= r + vox / 2]
        kk = (np.floor((Cw[:, :3, 3] - lo) / vox).astype(int)[:, None] + o[None]).reshape(-1, 3)
        kk = kk[((kk >= 0) & (kk < dims)).all(1)]; free[tuple(kk.T)] = True
    free &= ~occ
    trees = [cKDTree(p) if len(p) else None for p in pts_d]
    out = []
    if keep is not None:                    # 시각화용 (viz_collision_esdf.py): 격자와 프레임별 상태
        keep.update(occ=occ, free=free, esdf=esdf, lo=lo, vox=vox, r=r, cls={})
    for arm, name, poses in trajs:
        Tp = len(poses); cs = cd = unk = fr = 0; cs_m = []; cls = []
        for f in range(Tp):
            c = poses[f][:3, 3]; k = np.floor((c - lo) / vox).astype(int); ing = bool(((k >= 0) & (k < dims)).all())
            s_hit = ing and esdf[tuple(k)] < r
            tr = trees[int(round(f * (T - 1) / max(Tp - 1, 1)))]
            d_hit = tr is not None and tr.query(c, distance_upper_bound=r)[0] < r
            cs += s_hit; cd += d_hit
            cls.append("coll" if (s_hit or d_hit) else ("free" if ing and free[tuple(k)] else "unk"))
            if not (s_hit or d_hit):
                if ing and free[tuple(k)]:
                    fr += 1
                else:
                    unk += 1
            cs_m.append(s_hit or d_hit)
        nc = sum(cs_m)
        if keep is not None:
            keep["cls"][(arm, name)] = cls
        out.append({"arm": arm, "entry": name, "coll_any": float(nc > 0), "coll_frac": nc / Tp, "coll_static_frac": cs / Tp,
                    "coll_dyn_frac": cd / Tp, "unk_frac": unk / Tp, "unk_any": float(unk > 0), "free_frac": fr / Tp,
                    "coll_or_unk_any": float(nc + unk > 0), "vox": vox / S, "n_vox": int(np.prod(dims)),
                    "occ_frac": float(occ.mean()), "free_vox_frac": float(free.mean())})
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)          # label=<eval_dir>
    ap.add_argument("--out", required=True)
    ap.add_argument("--entries", default=None)                         # 파일: data_name 목록 (기본 전부)
    ap.add_argument("--gt_from", default=None)                         # GT 를 잴 arm 라벨 (기본 첫 arm)
    ap.add_argument("--eval_data", default=EVAL_DATA)
    ap.add_argument("--graph_root", default=GRAPH)
    ap.add_argument("--vista4d_root", default="/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    ap.add_argument("--name_prefix", default="dynpose")
    ap.add_argument("--margin_frac", type=float, default=0.02)         # 뱅크 --behind_margin_frac 기본
    ap.add_argument("--min_zcam_frac", type=float, default=0.02)       # 뱅크 --behind_min_zcam
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--method", default="shell", choices=["shell", "esdf", "knn"])   # [R374] shell = 기존 depth shell (빠름, val 용)
    ap.add_argument("--vox_frac", type=float, default=0.02)            # esdf: voxel 크기 / S
    ap.add_argument("--radius_frac", type=float, default=0.02)         # esdf: 카메라 반경 / S (= shell margin 과 같은 값)
    ap.add_argument("--max_dim", type=int, default=256)                # esdf: 격자 축 최대 칸 수
    ap.add_argument("--src_free", type=int, default=1)                # esdf: 소스 카메라 경로 (반경 r) 를 free 로
    ap.add_argument("--knn_down", type=int, default=1)               # knn: 점을 vox_frac·S 격자로 줄인 뒤 KD-tree
    ap.add_argument("--pix_stride", type=int, default=2)               # esdf: occupied 점 픽셀 간격
    ARGS = ap.parse_args()
    arms = [s.split("=", 1) for s in ARGS.arm]
    gt_arm = ARGS.gt_from or arms[0][0]
    keep = set(open(ARGS.entries).read().split()) if ARGS.entries else None
    by_scene = defaultdict(list)
    for label, d in arms:
        for p in glob(osp.join(d, "test", "*_transforms_pred.json")):
            name = osp.basename(p)[:-len("_transforms_pred.json")]
            if keep and name not in keep:
                continue
            video = name[len(ARGS.name_prefix) + 1:].rsplit("_", 1)[0]
            by_scene[video].append((label, name, p))
            if label == gt_arm:
                by_scene[video].append(("GT", name, p.replace("_pred.json", "_ref.json")))
    os.makedirs(ARGS.out, exist_ok=True)
    rows = []
    with Pool(ARGS.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(scene_job, sorted(by_scene.items()), chunksize=2)):
            rows += r
            if i % 100 == 0:
                print(f"[{i}/{len(by_scene)}] scenes", flush=True)
    keys = ["coll_any", "coll_frac", "coll_static_frac", "coll_dyn_frac"]
    if ARGS.method in ("esdf", "knn"):
        keys += ["unk_frac", "unk_any", "free_frac", "coll_or_unk_any", "vox", "n_vox", "occ_frac", "free_vox_frac"]
    with open(osp.join(ARGS.out, "per_entry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "entry", *keys, "error"]); w.writeheader(); w.writerows(rows)
    summ = {}
    for label in ["GT"] + [a for a, _ in arms]:
        R = [r for r in rows if r["arm"] == label and "error" not in r]
        summ[label] = {"n": len(R), "n_error": sum(1 for r in rows if r["arm"] == label and "error" in r),
                       **{k: float(np.mean([r[k] for r in R])) if R else float("nan") for k in keys}}
        print(label, json.dumps(summ[label]), flush=True)
    json.dump(summ, open(osp.join(ARGS.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
