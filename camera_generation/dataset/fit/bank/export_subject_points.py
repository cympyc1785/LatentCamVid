"""이 코드가 답하는 질문: subject 를 점 하나 (중심) 대신 3D 점 여러 개 (N, T, 3) 로 주려면 GT 에서 어떻게 만드나?

R361 (사용자 2026-10-13 "CoTracker3 (N, T, 3) track 이 있으니 track3d 대신에 condition 으로"). anchor 마다 `<scene>/da3/subject_points.npz`:
  dyn anchor: R246 CoTracker3 2D track (tmp/r246/tracks/<scene>__<anchor>.npz: tr2d (T,600,2) · vis (T,600), GT SAM3 subject mask 안 query)
              → 프레임마다 visible · GT subject mask 안 · depth 유효 · 하늘 아님 인 점만 recon depth · K · c2w 로 world 3D
  stat anchor (CoTracker 없음): 프레임마다 GT subject mask 안 픽셀에서 무작위 --n_static 개 → world 3D (정지 물체라 점 대응 불필요)
저장: pts (A,T,P,3) float16 world · valid (A,T,P) bool · anchor_id (A,) — target_track.npz 의 anchor_id 로 세그먼트와 잇는다.
env vista4d. 예) python fit/bank/export_subject_points.py --workers 24
"""
import argparse, json, os, os.path as osp, sys
import numpy as np

C = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"
G = "/data1/cympyc1785/LatentCamVid/camera_generation/dataset/out_dynpose"
E = "/data1/cympyc1785/data/DynPose-100K/eval_data/recon_and_seg"
SEG = "/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K/eval_data"
TRK = "/data1/cympyc1785/LatentCamVid/tmp/r246/tracks"


def gt_masks(v, node, T, H, W):
    sub = "seg_instances" if node["id"].startswith("dyn") else "seg_instances_static"
    mp = f"{SEG}/{sub}/{v}"
    out = np.zeros((T, H, W), bool)
    if not osp.isfile(f"{mp}/meta.json"):
        return out
    meta = json.load(open(f"{mp}/meta.json")); mz = np.load(f"{mp}/masks.npz")
    ids = {node.get("track_id")} | set(node.get("merged_from") or [])
    for f in range(min(T, len(meta["frames"]))):
        k = [i for i, b in enumerate(meta["frames"][f]) if b.get("id") in ids]
        if k:
            m = np.unpackbits(mz[f"{f:05d}"][k], axis=-1, count=meta["width"]).astype(bool).any(0)
            out[f] = m if m.shape == (H, W) else m[(np.arange(H) * m.shape[0] // H)][:, (np.arange(W) * m.shape[1] // W)]
    return out


def one(job):
    a, v = job
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    from utils.media import load_cameras, load_depths, load_masks
    dst = f"{a.corpus}/dynpose/{v}/da3/{a.name}"
    if osp.isfile(dst) and not a.overwrite:
        return v, "skip"
    try:
        z = np.load(f"{a.corpus}/dynpose/{v}/da3/target_track.npz", allow_pickle=True)
        aids = sorted({str(x) for x in z["anchor_id"]})
        nodes = {n["id"]: n for n in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"]}
        d = load_depths(f"{E}/{v}/depths", dtype=np.float16, desc=None).astype(np.float32)
        sky = np.asarray(load_masks(f"{E}/{v}/sky_mask", desc=None)).astype(bool)
        Cw, K = load_cameras(f"{E}/{v}/cameras.npz"); Cw, K = np.asarray(Cw, float), np.asarray(K, float)
    except Exception as e:      # noqa: BLE001
        return v, f"error {str(e)[:60]}"
    T, H, W = d.shape; P = a.n_static
    PTS = np.zeros((len(aids), T, P, 3), np.float16); VAL = np.zeros((len(aids), T, P), bool); src = []
    rng = np.random.default_rng(0)
    for ai, aid in enumerate(aids):
        node = nodes.get(aid)
        if node is None:
            src.append("no_node"); continue
        M = gt_masks(v, node, T, H, W); tp = f"{TRK}/{v}__{aid}.npz"
        if osp.isfile(tp):
            t = np.load(tp); xy, vis = t["tr2d"][:T], t["vis"][:T].astype(bool)       # (T,N,2) 픽셀 (recon 해상도)
            N = min(P, xy.shape[1]); xy, vis = xy[:, :N], vis[:, :N]; src.append("cotracker")
        else:
            xy = np.zeros((T, P, 2), np.float32); vis = np.zeros((T, P), bool); src.append("mask")
            for f in range(T):
                ys, xs = np.nonzero(M[f])
                if len(ys):
                    k = rng.choice(len(ys), min(P, len(ys)), replace=False)
                    xy[f, :len(k)] = np.stack([xs[k], ys[k]], -1); vis[f, :len(k)] = True
            N = P
        for f in range(T):
            u = np.clip(np.round(xy[f, :, 0]).astype(int), 0, W - 1); vv = np.clip(np.round(xy[f, :, 1]).astype(int), 0, H - 1)
            zf = d[f][vv, u]
            ok = vis[f] & M[f][vv, u] & np.isfinite(zf) & (zf > 0) & ~sky[f][vv, u]
            fx, fy, cx, cy = (K[f][0], K[f][1], K[f][2], K[f][3]) if K.ndim == 2 else (K[f][0, 0], K[f][1, 1], K[f][0, 2], K[f][1, 2])
            Xc = np.stack([(u + 0.5 - cx) / fx * zf, (vv + 0.5 - cy) / fy * zf, zf, np.ones_like(zf)], 0)
            PTS[ai, f, :N] = (Cw[f] @ Xc)[:3].T.astype(np.float16); VAL[ai, f, :N] = ok
    np.savez_compressed(dst, pts=PTS, valid=VAL, anchor_id=np.array(aids), source=np.array(src))
    return v, f"ok {' '.join(src)} vis {VAL.mean():.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=C)
    ap.add_argument("--name", default="subject_points.npz")
    ap.add_argument("--n_static", type=int, default=600)      # 저장 점 수 상한 (= R246 CoTracker query 600)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--videos", default="")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    vids = [x for x in a.videos.split(",") if x] or sorted(os.listdir(f"{a.corpus}/dynpose"))
    from multiprocessing import Pool
    bad = 0
    with Pool(a.workers) as p:
        for i, (v, msg) in enumerate(p.imap_unordered(one, [(a, v) for v in vids], chunksize=4)):
            bad += msg.startswith("error")
            if i % 500 == 0 or msg.startswith("error"):
                print(f"[{i}/{len(vids)}] {v} {msg}", flush=True)
    print(f"=== export_subject_points DONE {len(vids)} (error {bad}) ===", flush=True)
