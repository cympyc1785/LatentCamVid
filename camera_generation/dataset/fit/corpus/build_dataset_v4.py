"""이 코드가 답하는 질문: Dataset-v1 에서 subject 궤적만 CoTracker3 로 고친 코퍼스 (Dataset-v4) 를 기존 파일을 건드리지 않고 어떻게 만드나?

배경 (R245/R246): scene graph 의 subject 중심 = 보이는 점의 OBB 중심이라, 가림·화면 밖으로 일부만 보이면 중심이 떠돌아
정지 물체가 움직이는 것으로 (moving), 움직이는 subject 는 이동량이 부풀려져 잡힌다. tmp/r246/track.py 가 같은 물리 점을
CoTracker3 로 따라가 만든 이동 (tracks/<scene>__<anchor>.npz: trk_disp, n_pairs) 으로 바꾼다.
사용자 2026-10-07 "이전 데이터들 지우지는 말고 Dataset-v4로 필요한 파일만 새로 만드는걸로".

--stage link    v4/<corpus>/<scene>/ : images_4 → v1 symlink, da3/ 는 실디렉토리 + target_track.npz 외 전부 v1 symlink.
                루트의 meta / molmo2_cache / umt5_cache / geo_raw_cache 도 symlink (이름이 그대로라 캐시 재사용).
--stage track   da3/target_track.npz (v1 과 같은 키·포맷) + da3/subject_motion.npz 새로 씀:
                dyn anchor: 기준 프레임 r = n_pairs 최대 프레임, pos(t) = OBB center_smooth[r] + trk_disp[t] - trk_disp[r] (G → world T_wg)
                            n_pairs < --min_pairs 인 스텝은 trk_disp 가 이미 0 (hold) → frame_valid 0 으로 표시
                stat anchor · 추적 결과 없는 dyn: v1 target_track 그대로 (source='v1')
                moving_v2 = trk_path ≥ --moving_path (dyn 만; 추적 결과 없으면 v1 moving)
--stage lists   seg list: v1 목록에서 **track 계열 preset 인데 anchor 의 moving_v2 = False** 인 변이를 뺀다 (나머지 그대로).
--stage report  변경 통계.
env: 아무거나 (numpy). 예: python fit/corpus/build_dataset_v4.py --stage link,track,lists,report"""
import csv
import json
import os
import sys
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np

V1 = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"
V4 = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_v4"
TRK = "/data1/cympyc1785/LatentCamVid/tmp/r246/tracks"
GRAPH = "/data1/cympyc1785/LatentCamVid/camera_generation/dataset/out_dynpose"
LISTS = ["seg_list_dynpose_s91_train.txt", "seg_list_dynpose_s91_test.txt",
         "seg_list_dynpose_s91_lookat_nopull_train.txt", "seg_list_dynpose_s91_lookat_nopull_test.txt"]


def link(src, dst):
    if not path.lexists(dst):
        os.symlink(src, dst)


def stage_link(a):
    os.makedirs(f"{V4}/dynpose", exist_ok=True)
    for name in os.listdir(V1):
        if name != "dynpose" and not name.startswith("seg_list"):
            link(f"{V1}/{name}", f"{V4}/{name}")
    n = 0
    for sd in sorted(glob(f"{V1}/dynpose/*")):
        s = path.basename(sd)
        os.makedirs(f"{V4}/dynpose/{s}/da3", exist_ok=True)
        for x in os.listdir(sd):
            if x != "da3":
                link(f"{sd}/{x}", f"{V4}/dynpose/{s}/{x}")
        for x in os.listdir(f"{sd}/da3"):
            if x not in ("target_track.npz", "subject_motion.npz"):
                link(f"{sd}/da3/{x}", f"{V4}/dynpose/{s}/da3/{x}")
        n += 1
    print(f"[link] scenes {n}", flush=True)


def stage_track(a):
    stats = {"scenes": 0, "dyn_anchor": 0, "dyn_tracked": 0, "moving_v1": 0, "moving_v2": 0}
    for sd in sorted(glob(f"{V1}/dynpose/*")):
        s = path.basename(sd)
        z = np.load(f"{sd}/da3/target_track.npz", allow_pickle=True)
        track, valid, keys, anchors = z["track_world"].copy(), z["valid"], z["keys"], [str(x) for x in z["anchor_id"]]
        T = track.shape[1]
        fvalid = np.ones(track.shape[:2], dtype=bool)
        src = np.array(["v1"] * len(keys))
        mv1 = np.zeros(len(keys), dtype=bool); mv2 = np.zeros(len(keys), dtype=bool)
        trk_path = np.full(len(keys), np.nan)
        gp = f"{GRAPH}/{s}/scene_graph.json"
        g = json.load(open(gp)) if path.isfile(gp) else None
        nodes = {n["id"]: n for n in g["nodes"]} if g else {}
        T_wg = np.asarray(g["frames"]["T_wg"], dtype=np.float64) if g else None
        cache = {}
        for i, an in enumerate(anchors):
            node = nodes.get(an)
            if node is None:
                continue
            mv1[i] = bool(node.get("moving"))
            mv2[i] = mv1[i]
            if not an.startswith("dyn"):
                continue
            if an not in cache:
                f = f"{TRK}/{s}__{an}.npz"
                cache[an] = np.load(f) if path.isfile(f) else None
                stats["dyn_anchor"] += 1
            t = cache[an]
            if t is None or len(node["track"]["center_smooth"]) != T or t["trk_disp"].shape[0] != T:
                continue
            disp, npair = t["trk_disp"].astype(np.float64), t["n_pairs"]
            r = int(np.argmax(npair))
            cs = np.asarray(node["track"]["center_smooth"], dtype=np.float64)
            pos_g = cs[r] + disp - disp[r]
            pos_w = (np.c_[pos_g, np.ones(T)] @ T_wg.T)[:, :3]
            track[i] = pos_w.astype(np.float32)
            fv = np.ones(T, dtype=bool); fv[1:] = npair[1:] >= a.min_pairs
            fvalid[i] = fv
            src[i] = "cotracker3"
            trk_path[i] = float(np.linalg.norm(np.diff(disp, axis=0), axis=1).sum())
            mv2[i] = trk_path[i] >= a.moving_path
        out = f"{V4}/dynpose/{s}/da3"
        np.savez(f"{out}/target_track.npz", track_world=track, valid=valid, keys=keys, anchor_id=np.array(anchors))
        np.savez(f"{out}/subject_motion.npz", keys=keys, anchor_id=np.array(anchors), source=src, frame_valid=fvalid,
                 moving_v1=mv1, moving_v2=mv2, trk_path_u=trk_path)
        stats["scenes"] += 1
        stats["dyn_tracked"] += len({an for an, sr in zip(anchors, src) if sr == "cotracker3"})
    print("[track]", stats, flush=True)


def scene_motion(s):
    z = np.load(f"{V4}/dynpose/{s}/da3/subject_motion.npz", allow_pickle=True)
    pr = json.load(open(f"{V1}/dynpose/{s}/da3/prompts.json"))
    return {str(k): (bool(m2), pr[str(k)]["preset"]) for k, m2 in zip(z["keys"].tolist(), z["moving_v2"])}


def stage_lists(a):
    cache = {}
    for name in LISTS:
        keep, drop = [], 0
        for line in open(f"{V1}/{name}"):
            seg = line.strip()
            if not seg:
                continue
            _, s, k = seg.split("/")
            if s not in cache:
                cache[s] = scene_motion(s)
            mv, preset = cache[s].get(k, (True, ""))
            if preset.startswith("track") and not mv:
                drop += 1
                continue
            keep.append(seg)
        open(f"{V4}/{name}", "w").write("\n".join(keep) + "\n")
        print(f"[lists] {name}: keep {len(keep)} drop {drop}", flush=True)


def stage_report(a):
    import collections
    c = collections.Counter()
    for f in glob(f"{V4}/dynpose/*/da3/subject_motion.npz"):
        z = np.load(f, allow_pickle=True)
        seen = set()
        for an, sr, m1, m2 in zip(z["anchor_id"], z["source"], z["moving_v1"], z["moving_v2"]):
            if an in seen or not str(an).startswith("dyn"):
                continue
            seen.add(an)
            c[(str(sr), bool(m1), bool(m2))] += 1
    print("[report] (source, moving_v1, moving_v2) → anchor 수")
    for k, v in sorted(c.items()):
        print(" ", k, v)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--stage", default="link,track,lists,report")
    ap.add_argument("--min_pairs", type=int, default=5)
    ap.add_argument("--moving_path", type=float, default=0.1)   # trk 경로 (u) — 시험 202편: 가짜 후보 trk 변위 중앙 0.017
    a = ap.parse_args()
    for st in a.stage.split(","):
        globals()[f"stage_{st}"](a)
