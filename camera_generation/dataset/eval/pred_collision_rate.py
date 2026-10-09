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
