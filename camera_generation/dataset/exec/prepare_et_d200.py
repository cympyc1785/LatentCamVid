"""이 코드가 답하는 질문: Dataset-v1 (d200 s91) 을 E.T.(DIRECTOR) 학습용 et-data 로 어떻게 바꾸나? (우리 데이터로 E.T. 를 학습해 대조군으로)

사용자 2026-10-08 "molmo2, srccam 없이 3D position 넣는거면 사실 E.T. 를 학습하는거랑 비슷할 것 같은데 대조군으로 … E.T. 는 우선 세팅만".
형식 근거: tmp/agent/reader-r258-et-format.md (DIRECTOR src/datasets/** · configs/dataset/**).

et-data (= --out) 구성, 파일 stem `<scene>__<idx>` (clatr_dynpose_d200_s91 과 같은 이름):
  mixed_{train,test}_split.txt   이름 목록 (끝 개행 없음 — 빈 이름이 생긴다)
  traj/<n>.txt                   KITTI 3x4 c2w (프레임당 12 값)
  char/<n>.npy, char_raw/<n>.npy (T,3) subject 중심
  caption/<n>.txt                prompts.json concise 캡션
  caption_clip/seq/<n>.npy       (L,512) CLIP ViT-B/32 seq — clatr_dynpose_d200_s91/seq 기존 산출물 심링크
  intrinsics/<n>.npy             (3,3) 소스 K (학습 미사용, 로드만)
  cam_segments/<n>.npy           (T,) int — DIRECTOR CaptionMetrics.segment_camera_trajectories (caption F1 GT, 자기일관)
좌표: target_poses (OpenCV w2c, frame0 = I) → c2w → E.T. 관례 (char0 원점, y-up): W = diag(1,-1,-1),
      t' = W (c − p0), R' = W R, char' = W (p − p0)  (p = target_track subject 중심, p0 = 그 frame0).
stats (--stage stats, train 만): shift = t'[0], norm = diff(t'), norm_h = char 의 [p'0; diff(p')] → mean/std
      → DIRECTOR configs/dataset/standardization/d200s91.yaml (num_cams 49, velocity true).
발산 컷 |c2w| ≥ 1e2 (prepare_clatr_vista.py 와 같다), subject invalid 는 제외.
env: GenDoP (torch · evo · DIRECTOR import). 예: python exec/prepare_et_d200.py --stage convert,stats
"""
import json
import os
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

C = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"
CLIP_SEQ = "/data1/cympyc1785/data/DynPose-LBM/clatr_dynpose_d200_s91/seq"
DIRECTOR = "/data1/cympyc1785/LatentCamVid/camera_generation/models/DIRECTOR"
W = np.diag([1.0, -1.0, -1.0])


def convert_one(scene, idx, cache):
    if scene not in cache:
        sd = f"{C}/dynpose/{scene}/da3"
        tp = np.load(f"{sd}/target_poses.npz", allow_pickle=True)
        tt = np.load(f"{sd}/target_track.npz", allow_pickle=True)
        K = np.load(f"{sd}/pose.npz")["intrinsics"][0]
        cache.clear()
        cache[scene] = dict(E=tp["extrinsics"], keys={str(k): i for i, k in enumerate(tp["keys"].tolist())},
                            track=tt["track_world"], tvalid=tt["valid"], tkeys={str(k): i for i, k in enumerate(tt["keys"].tolist())},
                            K=K, prompts=json.load(open(f"{sd}/prompts.json")))
    d = cache[scene]
    i, j = d["keys"].get(idx), d["tkeys"].get(idx)
    if i is None or j is None or not d["tvalid"][j]:
        return None, "no_track"
    c2w = np.linalg.inv(d["E"][i].astype(np.float64))
    if np.abs(c2w).max() >= 1e2:
        return None, "diverged_pose"
    p = d["track"][j].astype(np.float64)
    p0 = p[0]
    R = np.einsum("ij,tjk->tik", W, c2w[:, :3, :3])
    t = (c2w[:, :3, 3] - p0) @ W.T
    ch = (p - p0) @ W.T
    pose = np.concatenate([R, t[:, :, None]], -1)                  # (T,3,4)
    cap = d["prompts"][idx]["prompt_camera_with_scene_video"]
    cap = cap["concise"] if isinstance(cap, dict) and "concise" in cap else (list(cap.values())[0] if isinstance(cap, dict) else cap)
    return dict(pose=pose, char=ch.astype(np.float32), K=d["K"].astype(np.float32), caption=cap), None


def stage_convert(a):
    sys.path.insert(0, DIRECTOR)
    import torch
    from src.metrics.modules.caption import CaptionMetrics
    seg = CaptionMetrics(num_classes=27)
    for sub in ("traj", "char", "char_raw", "caption", "caption_clip/seq", "intrinsics", "cam_segments"):
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    cache, why = {}, {}
    for split in ("train", "test"):
        names = []
        for line in open(f"{C}/seg_list_dynpose_s91_{split}.txt"):
            line = line.strip()
            if not line:
                continue
            _, scene, idx = line.split("/")
            n = f"{scene}__{idx}"
            clip_src = f"{CLIP_SEQ}/{split}/{n}_caption.npy"
            if not path.isfile(clip_src):
                why["no_clip"] = why.get("no_clip", 0) + 1
                continue
            r, bad = convert_one(scene, idx, cache)
            if r is None:
                why[bad] = why.get(bad, 0) + 1
                continue
            np.savetxt(f"{a.out}/traj/{n}.txt", r["pose"].reshape(len(r["pose"]), 12), fmt="%.8f")
            np.save(f"{a.out}/char/{n}.npy", r["char"])
            if not path.lexists(f"{a.out}/char_raw/{n}.npy"):
                os.symlink(f"../char/{n}.npy", f"{a.out}/char_raw/{n}.npy")
            open(f"{a.out}/caption/{n}.txt", "w").write(r["caption"])
            if not path.lexists(f"{a.out}/caption_clip/seq/{n}.npy"):
                os.symlink(clip_src, f"{a.out}/caption_clip/seq/{n}.npy")
            np.save(f"{a.out}/intrinsics/{n}.npy", r["K"])
            c2w4 = np.tile(np.eye(4), (len(r["pose"]), 1, 1)); c2w4[:, :3] = r["pose"]
            s = seg.segment_camera_trajectories(torch.from_numpy(c2w4).float()).numpy()   # (T-1,)
            np.save(f"{a.out}/cam_segments/{n}.npy", np.concatenate([s, s[-1:]]).astype(np.int64))
            names.append(n)
        open(f"{a.out}/mixed_{split}_split.txt", "w").write("\n".join(names))     # 끝 개행 없음
        print(f"[convert] {split}: {len(names)}  제외 {why}", flush=True)


def stage_stats(a):
    names = open(f"{a.out}/mixed_train_split.txt").read().split("\n")
    sh, nv, nh = [], [], []
    for n in names:
        P = np.loadtxt(f"{a.out}/traj/{n}.txt").reshape(-1, 3, 4)
        t = P[:, :, 3]
        sh.append(t[0]); nv.append(np.diff(t, axis=0))
        ch = np.load(f"{a.out}/char/{n}.npy")
        nh.append(np.concatenate([ch[:1], np.diff(ch, axis=0)]))
    sh, nv, nh = np.stack(sh), np.concatenate(nv), np.concatenate(nh)
    f = lambda x: "[" + ", ".join(f"{v:.8g}" for v in x) + "]"
    y = (f"name: 'd200s91'\nnum_interframes: 0\nnum_cams: 49\n"
         "num_total_frames: ${eval:'${dataset.standardization.num_interframes} * (${dataset.standardization.num_cams} - 1) + ${dataset.standardization.num_cams} '}\n\n"
         f"# prepare_et_d200.py --stage stats (Dataset-v1 s91 train {len(names)} 샘플, E.T. 좌표 char0 원점 · y-up)\n"
         f"norm_mean: {f(nv.mean(0))}\nnorm_std: {f(nv.std(0))}\n\nshift_mean: {f(sh.mean(0))}\nshift_std: {f(sh.std(0))}\n\n"
         f"norm_mean_h: {f(nh.mean(0))}\nnorm_std_h: {f(nh.std(0))}\n\nvelocity: true\n")
    out = f"{DIRECTOR}/configs/dataset/standardization/d200s91.yaml"
    open(out, "w").write(y)
    print(f"[stats] → {out}\n{y}", flush=True)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--stage", default="convert,stats")
    ap.add_argument("--out", default="/data1/cympyc1785/data/DynPose-LBM/et_dynpose_d200_s91")
    a = ap.parse_args()
    for st in a.stage.split(","):
        globals()[f"stage_{st}"](a)
