"""소스 영상 필터 — VLM(영상 fps 2) + SAM3(9프레임 추적) 로 "깔끔한 3인칭 subject" 영상만 남긴다 (D282).

이 코드가 답하는 질문: "DynPose-100K(등 코퍼스) 영상 중 1인칭·셀피·subject 과대/과소·신체 일부만·
가림·컷/자막 같은 영상을 빼고 학습용으로 쓸 수 있는 영상은 무엇이고, 뺀 이유는 무엇인가".

사용자 결정 (2026-09-24, R41~R52):
  - VLM 은 **영상 전체를 fps 2** 로 (`lbm/vlm.py VLMClient.chat(video=, video_fps=2)`).
    R41 50편 대조에서 8장 이미지와 정확도가 비슷하고 프레임당 토큰이 ~절반(126 vs 222)이다.
  - SAM3 는 **영상 추적 9프레임** (영상 전체 균등 9장). 49프레임 추적 12.15 s/clip vs 2.38 s/clip,
    추적 ID 가 유지되어 사람이 여럿이어도 main 을 한 명으로 이어 볼 수 있다 (R45).
  - 판정은 **VLM = 의미(시점·셀피·사람 수·합성·자막), SAM3 = 기하(면적·가장자리 잘림·지속)** 로
    나눈다. VLM 단독 R41 에서 closeup 9편 중 7~8편을 third 로 봤다 — 크기·잘림은 마스크로 잰다.

단계 (`--stage`):
  list    코퍼스 영상 목록 -> <out>/scenes.txt
  vlm     영상마다 구조화 JSON (main_subject_noun 포함) -> <out>/vlm/<shard>.jsonl   [vLLM 필요]
  sam3    VLM 의 main_subject_noun 으로 9프레임 추적 -> <out>/sam3/<scene_key>.npz    [GPU]
  judge   VLM + SAM3 -> 판정·사유 코드 -> <out>/judge.jsonl  (근거 수치 전부 기록)
  export  pass.csv / fail.csv (사유 분류) + reason 분포
  reel    통과/탈락 4x4 concat mp4 (사유 라벨) -> <out>/reels/

    python exec/run_video_filter.py --stage list
    python exec/run_video_filter.py --stage vlm --workers 24                     # vLLM 22002
    CUDA_VISIBLE_DEVICES=0 python exec/run_video_filter.py --stage sam3 --num_shards 3 --shard_id 0
    python exec/run_video_filter.py --stage judge && python exec/run_video_filter.py --stage export
"""
import csv
import json
import os
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from os import path

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))                  # camera_generation/dataset
sys.path.insert(0, HERE)
DV = "/data1/cympyc1785/LatentCamVid/DATA/worldtraj/dynamicverse"
VISTA4D = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"
MASK_HW = (180, 320)                  # 저장 마스크 해상도 (H, W) — 면적·가장자리 판정엔 충분하다

SCHEMA = """{
  "scene": "<one sentence: where this is and what happens>",
  "objects": [{"name": "<object>", "screen_fraction": "tiny|small|medium|large|dominant"}],
  "main_subject": {
    "category": "person|animal|vehicle|object|none",
    "noun": "<ONE common noun for segmentation, e.g. person, dog, horse, car, bicycle>",
    "description": "<short, distinguishing: clothes/color/position>",
    "num_people": <int>,
    "screen_fraction": "tiny(<5%)|small(5-15%)|medium(15-40%)|large(40-70%)|dominant(>70%)",
    "whole_body_visible": true|false,
    "visible_body_parts": ["head","face","torso","arms","hands","legs","feet"],
    "cut_by_frame_edge": ["top","bottom","left","right"],
    "occluded": true|false
  },
  "viewpoint": "third_person|first_person|selfie|none",
  "viewpoint_evidence": "<what in the video shows this>",
  "camera_motion": "static|pan|tilt|forward|backward|sideways|orbit|tracking|handheld_shake|mixed",
  "quality": {"blurry": true|false, "text_overlay_or_watermark": true|false,
              "letterbox_or_vertical": true|false, "scene_cut": true|false,
              "synthetic_or_game": true|false},
  "clean_third_person_subject": true|false,
  "reason": "<one sentence>"
}"""
DEFS = """Definitions:
- first_person: the camera is the eyes/head/body of a person or is mounted on a vehicle the person drives/rides (egocentric, dashcam, helmet cam); hands or body of the camera wearer may appear from the bottom edge.
- selfie: the person holding the camera films themselves; face/upper body stays close and centered while the background moves.
- third_person: an external camera films a subject that is not the camera operator.
- clean_third_person_subject = viewpoint is third_person AND a clear main subject is visible, not dominant (> 70% of the frame), not only a body part, and not heavily cut or occluded.
- If several people appear, the main subject is the one the camera is most clearly filming."""
PROMPT = ("This is a video clip.\nDescribe the clip as a JSON object with exactly these fields "
          f"(choose one value where options are listed):\n{SCHEMA}\n{DEFS}\nAnswer with the JSON only.")

# ── 판정 임계 (1차값 — 손라벨로 보정 예정. 바꾸면 judge 를 다시 돌리면 된다, VLM/SAM3 재호출 없음)
TH = {"area_min": 0.01,        # main 면적 중앙값 하한 (화면 대비) — 너무 작음
      "area_max": 0.45,        # main 면적 중앙값 상한 — 너무 큼 (closeup)
      "area_peak_max": 0.70,   # 어느 프레임이든 이 이상이면 너무 큼
      "edge_frac_max": 0.5,    # 마주보는 두 변(위+아래 또는 좌+우)에 닿는 프레임 비율 상한 — 신체 일부
      "presence_min": 0.6,     # main 이 보이는 프레임 비율 하한 — 사라짐/가림
      "rival_ratio": 0.8}      # 2등/1등 점수비 이상이면 main 모호 (사람이 여럿)
# [R69, 2026-09-28] v2 판정 (`--judge v2`). 사용자 "필터링이 좀 안좋은 것 같은데" + R68 reel 검토:
#   - 클로즈업 통과: 면적 상한 0.45/0.70 -> 0.25/0.50
#   - 한 변 잘림 미판정: 큰 subject(면적 med > cut_area) 가 한 변이라도 닿는 프레임 > edge_any_max 면 탈락
#   - main 오선택(지나가는 사람): 점수를 "보인 프레임 면적 중앙값" -> "전 K 프레임 평균 면적(없으면 0)"
#   - synthetic_or_game 가 실제 스포츠 경기를 잡음(154 중 76 이 경기 단어): 탈락 사유에서 뺀다(기록만)
TH_V2 = {**TH, "area_max": 0.25, "area_peak_max": 0.50, "edge_any_max": 0.5, "cut_area": 0.10}
# [R72] v3 = v2 에서 subject_too_small · blurry 를 판정에서 뺀다 (사용자 "subject too small은 filter에서 제외",
#       "blurry도 필터에서 빼줘").
TH_V3 = {**TH_V2, "area_min": 0.0}
# [R76] v4 = v3 + main subject 49 프레임 연속성 (`--stage subject49` 산출물 필요). 사용자 "main subject 가 보이다
#   안보이는 경우나 잘 보이다가 화면에 잘리거나 occlusion 이 심한 경우를 걸러내고 싶어". SAM3 9프레임이 아니라
#   recon 단계 seg_instances(49 프레임 전량) 의 그 track 으로 잰다.
TH_V4 = {**TH_V3,
         "vis49_min": 0.9,       # 49 프레임 중 보이는 비율 하한 — 보이다 안 보임 / 늦게 등장
         "gap49_max": 2,         # 처음~마지막 등장 사이 안 보이는 프레임 수 상한 — 중간 끊김
         "edge49_max": 0.3,      # 보이는 프레임 중 화면 가장자리에 닿는 비율 상한 — 잘림
         "occl_ratio": 0.5,      # 면적 < 중앙값 x 이 값 이면 가림 프레임 (가장자리 프레임 제외)
         "occl49_max": 0.2}      # 가림 프레임 비율 상한


def jpath(a, base):
    """v1 은 기존 파일명 그대로, v2 는 `<stem>_v2.<ext>` — 기존 리스트를 덮어쓰지 않는다."""
    if a.judge == "v1":
        return path.join(a.out, base)
    stem, ext = path.splitext(base)
    return path.join(a.out, f"{stem}_{a.judge}{ext}")


def scene_key(rel):
    return rel.replace("/", "__")


# ───────────────────────────────────────────────────────────────── list
def stage_list(a):
    fs = sorted(glob(path.join(DV, a.corpus, "*", "*", "video_input.mp4")))
    rels = [path.relpath(path.dirname(f), DV) for f in fs]
    os.makedirs(a.out, exist_ok=True)
    open(path.join(a.out, "scenes.txt"), "w").write("\n".join(rels) + "\n")
    print(f"[list] {len(rels)} scenes -> {a.out}/scenes.txt")


def load_scenes(a):
    s = [l.strip() for l in open(path.join(a.out, "scenes.txt")) if l.strip()]
    return s[a.shard_id::a.num_shards] if a.num_shards > 1 else s


def load_vlm(a):
    out = {}
    for f in glob(path.join(a.out, "vlm", "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            out[r["scene"]] = r
    return out


# ───────────────────────────────────────────────────────────────── vlm
def stage_vlm(a):
    from threading import Lock
    from lbm.vlm import VLMClient, parse_json_candidates
    scenes = load_scenes(a)
    done = load_vlm(a)
    todo = [s for s in scenes if s not in done]
    os.makedirs(path.join(a.out, "vlm"), exist_ok=True)
    fout = open(path.join(a.out, "vlm", f"s{a.shard_id}_{int(time.time())}.jsonl"), "a")
    lock, stat, t0 = Lock(), {"n": 0, "fail": 0}, time.time()
    print(f"[vlm] todo {len(todo)} / {len(scenes)} (done {len(scenes) - len(todo)})", flush=True)

    def one(rel):
        client = VLMClient(temperature=0.0, max_tokens=1200, timeout=600)
        try:
            vp = path.join(DV, rel, "video_input.mp4")
            # 영상 전체 픽셀 예산을 **프레임당 640x360** 으로 맞춘다 — Qwen3-VL 비디오 프로세서의
            # `size.longest_edge` 는 프레임당이 아니라 영상 전체 예산이다(실측: 230400 이면 139 토큰).
            # 원본 1280x720 그대로면 중앙값 6,762 토큰이라 KV 캐시가 15 요청에서 찬다 (R49 첫 판).
            # R41 이 검증한 640x360 clip 과 같은 해상도가 된다.
            import cv2
            cap = cv2.VideoCapture(vp)
            nf, vfps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 10.0
            cap.release()
            n2 = max(2, int(np.ceil(nf / vfps * a.fps)))
            budget = {"size": {"longest_edge": int(n2 * a.frame_pixels), "shortest_edge": 4096}}
            text, info = client.chat(PROMPT, video=vp, video_fps=a.fps, video_mm=budget)
            parsed = next((c for c in parse_json_candidates(text) if isinstance(c, dict)), None)
            row = {"scene": rel, "parsed": parsed, "raw": text, "sec": info["seconds"],
                   "prompt_tokens": info.get("prompt_tokens")}
        except Exception as exc:  # noqa: BLE001
            row = {"scene": rel, "parsed": None, "raw": "", "error": f"{type(exc).__name__}: {exc}"[:300]}
        with lock:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            stat["n"] += 1
            stat["fail"] += int(row["parsed"] is None)
            if stat["n"] % 200 == 0:
                el = time.time() - t0
                print(f"[vlm] {stat['n']}/{len(todo)} fail {stat['fail']}  {el / stat['n']:.2f} s/scene "
                      f"eta {(len(todo) - stat['n']) * el / stat['n'] / 3600:.1f} h", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    print(f"[vlm] ALL DONE {stat}", flush=True)


# ───────────────────────────────────────────────────────────────── sam3
def sample_frames(video_path, k):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(n - 1, 0), k).round().astype(int)
    frames, want, i = [], set(idx.tolist()), 0
    while len(frames) < len(want):
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            frames.append(im[:, :, ::-1].copy())
        i += 1
    cap.release()
    return np.stack(frames), idx[:len(frames)], n


def stage_sam3(a):
    import cv2
    import torch
    sys.path.insert(0, VISTA4D)
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video
    scenes = load_scenes(a)
    os.makedirs(path.join(a.out, "sam3"), exist_ok=True)
    pred, t0, n_done = None, time.time(), 0
    idle = 0
    while True:
        vlm = load_vlm(a)
        todo = [s for s in scenes if s in vlm and not path.exists(path.join(a.out, "sam3", scene_key(s) + ".npz"))]
        if not todo:
            if len([s for s in scenes if s in vlm]) >= len(scenes) or idle >= a.wait_rounds:
                break
            idle += 1
            time.sleep(120)                 # VLM 이 앞서 가도록 기다린다 (파이프라인)
            continue
        idle = 0
        if pred is None:
            pred = init_sam3_video()
        for rel in todo:
            # 다른 프로세스가 그새 끝냈으면 건너뛴다 — todo 는 라운드 시작 때 한 번만 만든다.
            if path.exists(path.join(a.out, "sam3", scene_key(rel) + ".npz")):
                continue
            p = (vlm[rel].get("parsed") or {})
            ms = p.get("main_subject") or {}
            noun = str(ms.get("noun") or "").strip().lower() or (
                "person" if ms.get("category") == "person" else "")
            dst = path.join(a.out, "sam3", scene_key(rel) + ".npz")
            if not noun or ms.get("category") == "none":
                np.savez_compressed(dst, noun="", skipped="no_subject_noun")
                continue
            frames, fidx, nf = sample_frames(path.join(DV, rel, "video_input.mp4"), a.k)
            H, W = frames.shape[1:3]
            try:
                _, seg = run_sam3_video(frames, pred, [noun])
            except Exception as exc:  # noqa: BLE001 — 한 편이 죽어도 샤드는 간다
                np.savez_compressed(dst, noun=noun, skipped=f"error:{type(exc).__name__}")
                torch.cuda.empty_cache()
                continue
            ids = sorted({inst["id"] for f in seg for inst in f})
            K = len(frames)
            masks = np.zeros((len(ids), K) + MASK_HW, dtype=bool)
            boxes = np.full((len(ids), K, 4), np.nan, dtype=np.float32)
            scores = np.zeros((len(ids), K), dtype=np.float32)
            for t, f in enumerate(seg):
                for inst in f:
                    j = ids.index(inst["id"])
                    masks[j, t] = cv2.resize(inst["mask"].astype(np.uint8), MASK_HW[::-1],
                                             interpolation=cv2.INTER_NEAREST) > 0
                    boxes[j, t] = np.asarray(inst["box_xyxy"], dtype=np.float32) / [W, H, W, H]
                    scores[j, t] = inst["score"]
            np.savez_compressed(dst, noun=noun, ids=np.asarray(ids), frame_idx=fidx, num_frames=nf,
                                masks=np.packbits(masks, axis=-1), mask_hw=np.asarray(MASK_HW),
                                boxes_norm=boxes, scores=scores)
            n_done += 1
            if n_done % 100 == 0:
                el = time.time() - t0
                print(f"[sam3/s{a.shard_id}] {n_done} done  {el / n_done:.2f} s/scene", flush=True)
    print(f"[sam3/s{a.shard_id}] ALL DONE {n_done}", flush=True)


# ───────────────────────────────────────────────────────────────── subject49
def subject49_stats(seg, track_id, edge_px=2, occl_ratio=0.5):
    """seg_instances track 하나의 49 프레임 연속성 — 보임/끊김/가장자리/가림(면적 급감)."""
    F = seg.num_frames
    area = np.zeros(F); edge = np.zeros(F, bool)
    tr = seg.tracks[track_id]
    for f, slot in zip(tr["frames"], tr["slots"]):
        # frame_masks 는 그 프레임 인스턴스 전부를 푼다(~30장) — 필요한 slot 한 장만 푼다
        m = np.unpackbits(seg._npz[f"{f:05d}"][slot], axis=-1, count=seg.width).astype(bool)
        area[f] = m.mean()
        edge[f] = bool(m[:edge_px].any() or m[-edge_px:].any() or m[:, :edge_px].any() or m[:, -edge_px:].any())
    vis = area > 0
    if not vis.any():
        return {"vis_frac": 0.0, "gap_frames": F, "edge_frac": 0.0, "occl_frac": 0.0, "area_med": 0.0}
    idx = np.flatnonzero(vis)
    med = float(np.median(area[vis]))
    inner = vis & ~edge                                    # 가장자리 잘림은 따로 세므로 가림에서 뺀다
    occl = inner & (area < occl_ratio * med)
    return {"vis_frac": float(vis.mean()), "gap_frames": int((~vis[idx[0]:idx[-1] + 1]).sum()),
            "first": int(idx[0]), "last": int(idx[-1]), "edge_frac": float(edge[vis].mean()),
            "occl_frac": float(occl.sum() / max(inner.sum(), 1)), "area_med": med}


def main_node(graph_nodes, noun):
    """VLM 명사와 label 이 맞는 동적 노드 중 max_area_frac 최대, 없으면 동적 노드 전체에서 (R75 와 같은 규칙)."""
    dyn = [n for n in graph_nodes if n["id"].startswith("dyn")]
    if not dyn:
        return None
    noun = (noun or "").lower()
    hit = [n for n in dyn if noun and (noun in n["label"].lower() or n["label"].lower() in noun)]
    return max(hit or dyn, key=lambda n: n.get("max_area_frac", 0))


def stage_subject49(a):
    """`--judge` 판정의 통과 영상마다 main 동적 subject 의 49 프레임 연속성 -> <out>/subject49.jsonl.
    graph = <graph_root>/<uuid>/scene_graph.json, seg = <seg_root>/<uuid>/ (없으면 main=None)."""
    from scene_graph.io import SegInstances
    rows = [json.loads(l) for l in open(jpath(a, "judge.jsonl"))]
    # [R82] `--s49_scope all` 이면 판정 통과 여부와 무관하게 전량 (graph/seg 없는 영상은 main=None).
    todo = rows if a.s49_scope == "all" else [r for r in rows if r["keep"]]
    done = set()
    dst = path.join(a.out, "subject49.jsonl")
    if path.exists(dst):
        done = {json.loads(l)["scene"] for l in open(dst)}
    n = 0
    with open(dst, "a") as fo:
        for r in todo:
            if r["scene"] in done:
                continue
            u = r["scene"].split("/")[-1]
            gp, sp = path.join(a.graph_root, u, "scene_graph.json"), path.join(a.seg_root, u)
            out = {"scene": r["scene"], "main": None}
            if path.isfile(gp) and path.isdir(sp):
                nodes = json.load(open(gp))["nodes"]
                node = main_node(nodes, r["vlm"]["noun"])
                if node is not None and node.get("track_id") is not None:
                    seg = SegInstances(sp, "dyn")
                    if int(node["track_id"]) in seg.tracks:
                        out["main"] = {"node": node["id"], "label": node["label"],
                                       **subject49_stats(seg, int(node["track_id"]))}
            fo.write(json.dumps(out) + "\n")
            fo.flush()
            n += 1
            if n % 200 == 0:
                print(f"[subject49] {n}/{len(todo)}", flush=True)
    print(f"[subject49] {n} new -> {dst}")


# ───────────────────────────────────────────────────────────────── judge
def g(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def sam3_stats(npz_path, main_score="median"):
    z = np.load(npz_path, allow_pickle=False)
    if "skipped" in z.files:
        return {"skipped": str(z["skipped"])}
    ids = z["ids"]
    if len(ids) == 0:
        return {"n_inst": 0}
    hw = tuple(z["mask_hw"])
    masks = np.unpackbits(z["masks"], axis=-1)[..., :hw[1]].astype(bool)      # (N,K,H,W)
    area = masks.mean(axis=(2, 3))                                             # (N,K)
    present = area > 0
    top = masks[:, :, 0, :].any(-1); bot = masks[:, :, -1, :].any(-1)
    lef = masks[:, :, :, 0].any(-1); rig = masks[:, :, :, -1].any(-1)
    opposite = (top & bot) | (lef & rig)                                       # 마주보는 두 변 = 화면을 가로지름
    cx = np.nanmean((z["boxes_norm"][..., 0] + z["boxes_norm"][..., 2]) / 2, axis=1)
    cy = np.nanmean((z["boxes_norm"][..., 1] + z["boxes_norm"][..., 3]) / 2, axis=1)
    center = 1.0 - np.clip(np.hypot(np.nan_to_num(cx, nan=0.5) - 0.5, np.nan_to_num(cy, nan=0.5) - 0.5) / 0.707, 0, 1)
    pres = present.mean(1)
    if main_score == "mean":             # v2: 안 보인 프레임은 0 — 잠깐 크게 지나간 인스턴스가 못 이긴다
        score = area.mean(1) * (0.75 + 0.25 * center)
    else:
        score = np.array([np.median(area[j][present[j]]) if present[j].any() else 0.0
                          for j in range(len(ids))]) * pres * (0.75 + 0.25 * center)
    order = np.argsort(-score)
    m = int(order[0])
    pm = present[m]
    return {"n_inst": int(len(ids)), "main_id": int(ids[m]),
            "main_area_med": float(np.median(area[m][pm])) if pm.any() else 0.0,
            "main_area_max": float(area[m].max()), "main_presence": float(pres[m]),
            "main_edge_opposite_frac": float(opposite[m][pm].mean()) if pm.any() else 0.0,
            "main_edge_any_frac": float((top | bot | lef | rig)[m][pm].mean()) if pm.any() else 0.0,
            "rival_ratio": float(score[order[1]] / score[m]) if len(ids) > 1 and score[m] > 0 else 0.0,
            "n_inst_frame_max": int(present.sum(0).max())}


def judge_one(v, s, version="v1"):
    """-> (keep, [사유 코드], 근거 dict). 사유는 전부 모은다 (첫 번째만이 아니라)."""
    p = (v or {}).get("parsed") or {}
    why = []
    if not p:
        why.append("vlm_parse_fail")
    vp = p.get("viewpoint")
    if vp == "first_person":
        why.append("first_person")
    elif vp == "selfie":
        why.append("selfie")
    elif vp == "none" or g(p, "main_subject", "category") in ("none", None):
        why.append("no_subject")
    # 2차 판정 (첫 전량 판정의 PASS reel 16편 중 6편이 손만 나오는 탁상/물체 클로즈업이었다 —
    # 사람 크기·잘림 게이트는 main 이 물체면 아무것도 안 본다). subject 는 사람/동물/탈것만 둔다.
    cat, noun = g(p, "main_subject", "category"), str(g(p, "main_subject", "noun") or "").lower()
    parts = set(g(p, "main_subject", "visible_body_parts") or [])
    if cat == "object":
        why.append("subject_is_object")
    if noun in ("hand", "hands", "finger", "fingers", "arm", "arms") or \
            (cat == "person" and parts and parts <= {"hands", "arms"}):
        why.append("hands_only")
    q = p.get("quality") or {}
    for k, code in (("scene_cut", "scene_cut"), ("text_overlay_or_watermark", "overlay"),
                    ("letterbox_or_vertical", "letterbox"), ("synthetic_or_game", "synthetic"),
                    ("blurry", "blurry")):
        if q.get(k) is True and not (version in ("v2", "v3", "v4", "v5") and code == "synthetic") \
                and not (version in ("v3", "v4", "v5") and code == "blurry"):
            why.append(code)
    th = {"v2": TH_V2, "v3": TH_V3, "v4": TH_V4, "v5": TH_V4}.get(version, TH)
    # [R82] v5 = 전량 판정. 9 프레임 SAM3 의 연속성·잘림 검사(too_small/body_part/lost/cut_edge)는
    #   49 프레임 검사가 대신하므로 끈다. 면적 상한(too_large)·main 모호·인스턴스 없음은 남긴다.
    g9 = version != "v5"
    if s.get("skipped"):
        why.append("sam3_" + s["skipped"].split(":")[0])
    elif s.get("n_inst", 0) == 0:
        why.append("sam3_no_instance")
    else:
        if g9 and s["main_area_med"] < th["area_min"]:
            why.append("subject_too_small")
        if s["main_area_med"] > th["area_max"] or s["main_area_max"] > th["area_peak_max"]:
            why.append("subject_too_large")
        if g9 and s["main_edge_opposite_frac"] > th["edge_frac_max"]:
            why.append("body_part_or_cut")
        if g9 and s["main_presence"] < th["presence_min"]:
            why.append("subject_lost_or_occluded")
        if s["rival_ratio"] >= th["rival_ratio"]:
            why.append("main_ambiguous_multi")
        if version in ("v2", "v3", "v4") and s["main_area_med"] > th["cut_area"] and s["main_edge_any_frac"] > th["edge_any_max"]:
            why.append("large_subject_cut_edge")
    if version in ("v4", "v5"):
        m = s.get("subject49")
        if m is None:
            why.append("no_subject49")           # graph/seg_instances 없음 — recon 전 영상
        else:
            if m["vis_frac"] < th["vis49_min"]:
                why.append("subject_not_always_visible")
            if m["gap_frames"] > th["gap49_max"]:
                why.append("subject_gap")
            if m["edge_frac"] > th["edge49_max"]:
                why.append("subject_edge_cut")
            if m["occl_frac"] > th["occl49_max"]:
                why.append("subject_occluded")
    # 기록용: VLM 과 SAM3 가 어긋난 것 (판정엔 안 쓰고 검토 큐 용)
    disagree = []
    if p.get("clean_third_person_subject") is True and any(c in why for c in ("subject_too_large", "body_part_or_cut")):
        disagree.append("vlm_clean_but_sam3_geom_bad")
    if p.get("clean_third_person_subject") is False and not why:
        disagree.append("vlm_unclean_but_all_gates_pass")
    return (not why), why, disagree


def stage_judge(a):
    vlm = load_vlm(a)
    s49 = {}
    if a.judge in ("v4", "v5"):
        for l in open(path.join(a.out, "subject49.jsonl")):
            r = json.loads(l)
            s49[r["scene"]] = r.get("main")
    scenes = [l.strip() for l in open(path.join(a.out, "scenes.txt")) if l.strip()]
    rows = []
    for rel in scenes:
        f = path.join(a.out, "sam3", scene_key(rel) + ".npz")
        if rel not in vlm or not path.exists(f):
            continue
        s = sam3_stats(f, "mean" if a.judge in ("v2", "v3", "v4", "v5") else "median")
        if a.judge in ("v4", "v5"):
            s["subject49"] = s49.get(rel)
        keep, why, disagree = judge_one(vlm[rel], s, a.judge)
        p = vlm[rel].get("parsed") or {}
        rows.append({"scene": rel, "keep": keep, "reasons": why, "disagree": disagree,
                     "vlm": {"viewpoint": p.get("viewpoint"), "noun": g(p, "main_subject", "noun"),
                             "category": g(p, "main_subject", "category"),
                             "num_people": g(p, "main_subject", "num_people"),
                             "fraction": g(p, "main_subject", "screen_fraction"),
                             "clean": p.get("clean_third_person_subject"), "reason": p.get("reason")},
                     "sam3": s, "th": {"v2": TH_V2, "v3": TH_V3, "v4": TH_V4, "v5": TH_V4}.get(a.judge, TH), "judge": a.judge})
    if a.manual_keep:                    # [R70] 사람이 본 영상 강제 keep — 자동 사유는 기록으로 남긴다
        want = {l.strip() for l in open(a.manual_keep) if l.strip() and not l.startswith("#")}
        for r in rows:
            if r["scene"] in want:
                r.update(keep=True, manual_keep=True, auto_reasons=r["reasons"], reasons=[])
        print(f"[judge] manual_keep {sum(r.get('manual_keep', False) for r in rows)} / {len(want)} <- {a.manual_keep}")
    with open(jpath(a, "judge.jsonl"), "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[judge] {len(rows)} scenes, keep {sum(r['keep'] for r in rows)} -> {jpath(a, 'judge.jsonl')}")


def stage_export(a):
    from collections import Counter
    rows = [json.loads(l) for l in open(jpath(a, "judge.jsonl"))]
    cols = ["scene", "reasons", "vlm_viewpoint", "vlm_noun", "vlm_num_people", "vlm_fraction",
            "sam3_n_inst", "main_area_med", "main_area_max", "main_presence", "main_edge_opposite_frac",
            "rival_ratio", "disagree", "vlm_reason"]
    for name, keep in (("pass.csv", True), ("fail.csv", False)):
        with open(jpath(a, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                if r["keep"] != keep:
                    continue
                s, v = r["sam3"], r["vlm"]
                w.writerow([r["scene"], "|".join(r["reasons"]), v["viewpoint"], v["noun"], v["num_people"],
                            v["fraction"], s.get("n_inst"), s.get("main_area_med"), s.get("main_area_max"),
                            s.get("main_presence"), s.get("main_edge_opposite_frac"), s.get("rival_ratio"),
                            "|".join(r["disagree"]), v["reason"]])
    c = Counter(x for r in rows for x in r["reasons"])
    first = Counter(r["reasons"][0] for r in rows if r["reasons"])
    summ = {"n": len(rows), "pass": sum(r["keep"] for r in rows),
            "reason_any": dict(c.most_common()), "reason_first": dict(first.most_common()),
            "disagree": dict(Counter(x for r in rows for x in r["disagree"]))}
    json.dump(summ, open(jpath(a, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


SAM3_GEOM = ("subject_too_small", "subject_too_large", "body_part_or_cut", "subject_lost_or_occluded",
             "main_ambiguous_multi")
SUBJ49 = ("subject_not_always_visible", "subject_gap", "subject_edge_cut", "subject_occluded")
SAM3_GEOM_V2 = SAM3_GEOM[:3] + ("large_subject_cut_edge",) + SAM3_GEOM[3:]   # v1 reel 표본을 안 바꾸려고 따로 둔다


def plain_tile(r, TW, TH_, nf=49):
    """원본 영상 앞 nf 프레임 + 사유/수치 라벨 (마스크 없음)."""
    import cv2
    cap = cv2.VideoCapture(path.join(DV, r["scene"], "video_input.mp4"))
    s = r["sam3"]
    lab = "PASS" if r["keep"] else ",".join(r["reasons"])[:52]
    cap_ = (f"{r['vlm']['noun']} area med {s.get('main_area_med', 0):.3f} max {s.get('main_area_max', 0):.3f} "
            f"pres {s.get('main_presence', 0):.2f} edge {s.get('main_edge_opposite_frac', 0):.2f}")
    fr = []
    while len(fr) < nf:
        ok, im = cap.read()
        if not ok:
            break
        im = cv2.resize(im, (TW, TH_))
        cv2.rectangle(im, (0, 0), (TW, 18), (0, 0, 0), -1)
        cv2.putText(im, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (80, 255, 80) if r["keep"] else (80, 80, 255), 1, cv2.LINE_AA)
        cv2.rectangle(im, (0, TH_ - 16), (TW, TH_), (0, 0, 0), -1)
        cv2.putText(im, cap_, (3, TH_ - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        fr.append(im)
    cap.release()
    return fr or [np.zeros((TH_, TW, 3), np.uint8)]


def stage_reel_overlay(a, rows, ffmpeg):
    """[R68] 4x4 한 장 — 위 2행 = SAM3 기하 게이트로 탈락 8편(사유마다 고르게), 아래 2행 = 통과 8편.

    main subject 마스크(judge 의 `main_id`)를 초록으로, 나머지 인스턴스를 빨강으로 얹는다. SAM3 마스크는
    추적한 `frame_idx` K장(기본 9)에만 있으므로 타일도 그 K장만 재생한다 (보간하면 마스크가 어긋난다).
    """
    import cv2
    import subprocess
    rng = np.random.default_rng(a.seed)
    geom = SAM3_GEOM_V2 if a.judge in ("v2", "v3") else SAM3_GEOM
    def strat(codes, n, only):
        """사유별로 돌아가며 한 편씩. only 면 그 계열 사유로만 탈락한 영상, 아니면 사유가 하나뿐인 영상 우선."""
        fam = set(codes)
        qs = []
        for code in codes:
            if only:
                pool = [r for r in rows if not r["keep"] and r["reasons"][0] == code and set(r["reasons"]) <= fam]
            else:
                pool = [r for r in rows if not r["keep"] and r["reasons"] == [code]] or \
                    [r for r in rows if not r["keep"] and code in r["reasons"]]
            if pool:
                qs.append([pool[i] for i in rng.permutation(len(pool))])
        out, i = [], 0
        while len(out) < n and any(qs):
            if qs[i % len(qs)]:
                out.append(qs[i % len(qs)].pop())
            i += 1
        return out
    if a.fail_rows == "vlm_sam3":        # [R68c] 1행 = VLM 사유로만 탈락 4, 2행 = SAM3 기하로만 탈락 4
        vlm_codes = sorted({c for r in rows for c in r["reasons"]} - set(geom)
                           - {c for r in rows for c in r["reasons"] if c.startswith("sam3_")})
        # 4칸 < 사유 종류라 seed 마다 사유 순서를 섞는다 (알파벳순이면 늘 같은 4 사유만 나온다)
        pick_f = strat([vlm_codes[i] for i in rng.permutation(len(vlm_codes))], 4, True) + \
            strat([geom[i] for i in rng.permutation(len(geom))], 4, True)
    else:
        pick_f = strat(geom, 8, False)
    passes = [r for r in rows if r["keep"]]
    if a.pass_list:                      # [R71] 사람이 고른 통과 8편 (목록 순서대로)
        by = {r["scene"]: r for r in rows}
        pp = [by[l.strip()] for l in open(a.pass_list) if l.strip() and not l.startswith("#")][:8]
        assert all(r["keep"] for r in pp), "pass_list 에 keep=False 인 영상이 있다 (manual_keep 먼저)"
    else:
        pp = [passes[j] for j in rng.choice(len(passes), 8, replace=False)]
    pick = pick_f + pp
    TW, TH_ = 480, 270
    plain = a.reel_mode == "plain"           # [R68b] 마스크 없이 원본 영상 49프레임
    tiles = []
    for r in pick:
        if plain:
            tiles.append(plain_tile(r, TW, TH_))
            continue
        z = np.load(path.join(a.out, "sam3", scene_key(r["scene"]) + ".npz"), allow_pickle=False)
        frames, _, _ = sample_frames(path.join(DV, r["scene"], "video_input.mp4"), a.k)
        hw = tuple(z["mask_hw"])
        masks = np.unpackbits(z["masks"], axis=-1)[..., :hw[1]].astype(bool)          # (N,K,H,W)
        ids = list(z["ids"])
        main = ids.index(r["sam3"]["main_id"]) if r["sam3"].get("main_id") in ids else -1
        s = r["sam3"]
        lab = "PASS" if r["keep"] else ",".join(c for c in r["reasons"])[:52]
        cap_ = (f"{r['vlm']['noun']} area med {s.get('main_area_med', 0):.3f} max {s.get('main_area_max', 0):.3f} "
                f"pres {s.get('main_presence', 0):.2f} edge {s.get('main_edge_opposite_frac', 0):.2f}")
        fr = []
        for t in range(len(frames)):
            im = cv2.resize(frames[t][:, :, ::-1], (TW, TH_)).astype(np.float32)
            for j in range(len(ids)):
                m = cv2.resize(masks[j, t].astype(np.uint8), (TW, TH_), interpolation=cv2.INTER_NEAREST) > 0
                col = np.array((60, 220, 60) if j == main else (60, 60, 230), np.float32)
                im[m] = 0.45 * im[m] + 0.55 * col
            im = im.astype(np.uint8)
            cv2.rectangle(im, (0, 0), (TW, 18), (0, 0, 0), -1)
            cv2.putText(im, lab, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (80, 255, 80) if r["keep"] else (80, 80, 255), 1, cv2.LINE_AA)
            cv2.rectangle(im, (0, TH_ - 16), (TW, TH_), (0, 0, 0), -1)
            cv2.putText(im, cap_, (3, TH_ - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            fr.append(im)
        tiles.append(fr)
    K = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (K - len(t)) for t in tiles]
    tmpd = path.join(a.out, "reels" if a.judge == "v1" else f"reels_{a.judge}", "_overlay")
    os.makedirs(tmpd, exist_ok=True)
    for t in range(K):
        grid = np.vstack([np.hstack([tiles[r0 * 4 + c][t] for c in range(4)]) for r0 in range(4)])
        cv2.imwrite(path.join(tmpd, f"{t:03d}.jpg"), grid)
    tag = ("vlm4_sam3_4" if a.fail_rows == "vlm_sam3" else "fail8") + ("_picked" if a.pass_list else "")
    dst = path.join(a.out, "reels" if a.judge == "v1" else f"reels_{a.judge}", f"{a.reel_mode}_{tag}_pass8_seed{a.seed}.mp4")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "10" if plain else "2", "-i", path.join(tmpd, "%03d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
    for f in glob(path.join(tmpd, "*.jpg")):
        os.remove(f)
    os.rmdir(tmpd)
    with open(dst.replace(".mp4", ".txt"), "w") as f:
        for r in pick:
            f.write(f"{r['scene']}\t{'|'.join(r['reasons']) or 'PASS'}\t{r['vlm']['reason']}\n")
    print(f"[reel] {dst}")


def stage_reel(a):
    """4x4 concat (16편) mp4 — 통과 1장 + 탈락 사유별 1장씩. 타일마다 사유 라벨.
    `--reel_mode overlay` 면 탈락 8 + 통과 8 한 장에 SAM3 마스크 overlay (stage_reel_overlay)."""
    import cv2
    import subprocess
    ffmpeg = ("/data1/cympyc1785/miniconda3/envs/vista4d/lib/python3.12/site-packages/imageio_ffmpeg/"
              "binaries/ffmpeg-linux-x86_64-v7.0.2")
    rows = [json.loads(l) for l in open(jpath(a, "judge.jsonl"))]
    if a.reel_mode in ("overlay", "plain"):
        return stage_reel_overlay(a, rows, ffmpeg)
    rng = np.random.default_rng(0)
    groups = {"PASS": [r for r in rows if r["keep"]]}
    for r in rows:
        if not r["keep"]:
            groups.setdefault(r["reasons"][0], []).append(r)
    os.makedirs(path.join(a.out, "reels" if a.judge == "v1" else f"reels_{a.judge}"), exist_ok=True)
    TW, TH_, NF = 320, 180, 49
    for name, rs in groups.items():
        if not rs:
            continue
        pick = [rs[i] for i in rng.choice(len(rs), min(16, len(rs)), replace=False)]
        tiles = []
        for r in pick:
            cap = cv2.VideoCapture(path.join(DV, r["scene"], "video_input.mp4"))
            fr = []
            while len(fr) < NF:
                ok, im = cap.read()
                if not ok:
                    break
                im = cv2.resize(im, (TW, TH_))
                lab = "PASS" if r["keep"] else ",".join(r["reasons"])[:44]
                cv2.rectangle(im, (0, 0), (TW, 16), (0, 0, 0), -1)
                cv2.putText(im, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (80, 255, 80) if r["keep"] else (80, 80, 255), 1, cv2.LINE_AA)
                fr.append(im)
            cap.release()
            while len(fr) < NF:
                fr.append(fr[-1] if fr else np.zeros((TH_, TW, 3), np.uint8))
            tiles.append(fr)
        while len(tiles) < 16:
            tiles.append([np.zeros((TH_, TW, 3), np.uint8)] * NF)
        tmpd = path.join(a.out, "reels" if a.judge == "v1" else f"reels_{a.judge}", f"_{name}")
        os.makedirs(tmpd, exist_ok=True)
        for t in range(NF):
            grid = np.vstack([np.hstack([tiles[r0 * 4 + c][t] for c in range(4)]) for r0 in range(4)])
            cv2.imwrite(path.join(tmpd, f"{t:03d}.jpg"), grid)
        dst = path.join(a.out, "reels" if a.judge == "v1" else f"reels_{a.judge}", f"{name}_{len(rs)}.mp4")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "10", "-i",
                        path.join(tmpd, "%03d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
        for f in glob(path.join(tmpd, "*.jpg")):
            os.remove(f)
        os.rmdir(tmpd)
        with open(path.join(a.out, "reels" if a.judge == "v1" else f"reels_{a.judge}", f"{name}_{len(rs)}.txt"), "w") as f:
            for r in pick:
                f.write(f"{r['scene']}\t{'|'.join(r['reasons']) or 'PASS'}\t{r['vlm']['reason']}\n")
        print(f"[reel] {dst}")


if __name__ == "__main__":
    q = ArgumentParser()
    q.add_argument("--stage", required=True, choices=("list", "vlm", "sam3", "subject49", "judge", "export", "reel"))
    q.add_argument("--corpus", default="dynpose-100k")
    q.add_argument("--out", default=path.join(HERE, "out_filter", "d282_dynpose100k"))
    q.add_argument("--fps", default=2.0, type=float)
    q.add_argument("--frame_pixels", default=640 * 360, type=int)   # VLM 프레임당 픽셀 예산 (R41 과 같은 해상도)
    q.add_argument("--workers", default=24, type=int)
    q.add_argument("--k", default=9, type=int)                  # SAM3 추적 프레임 수 (영상 전체 균등)
    q.add_argument("--reel_mode", default="groups", choices=("groups", "overlay", "plain"))   # reel: 기존 사유별 / R68 overlay / 같은 표본 원본
    q.add_argument("--fail_rows", default="sam3", choices=("sam3", "vlm_sam3"))  # reel overlay/plain 탈락 2행 구성
    q.add_argument("--seed", default=0, type=int)
    q.add_argument("--s49_scope", default="pass", choices=("pass", "all"))                       # subject49
    q.add_argument("--graph_root", default=path.join(HERE, "out_dynpose"))                         # subject49
    q.add_argument("--seg_root", default="/data1/cympyc1785/data/DynPose-100K/eval_data/seg_instances")  # subject49
    q.add_argument("--pass_list", default=None)     # reel overlay/plain: 아래 2행 통과 8편을 이 목록으로
    q.add_argument("--manual_keep", default=None)   # judge: 한 줄에 scene 하나, 자동 판정과 무관하게 keep
    q.add_argument("--judge", default="v1", choices=("v1", "v2", "v3", "v4", "v5"))   # v2 = R69 판정, v3 = v2 - subject_too_small , v4 = v3 + subject49 (파일명 _vN, 기존 보존)
    q.add_argument("--num_shards", default=1, type=int)
    q.add_argument("--shard_id", default=0, type=int)
    q.add_argument("--wait_rounds", default=600, type=int)      # sam3: VLM 결과 기다리는 최대 라운드(x120 s)
    a = q.parse_args()
    {"list": stage_list, "vlm": stage_vlm, "sam3": stage_sam3, "subject49": stage_subject49, "judge": stage_judge,
     "export": stage_export, "reel": stage_reel}[a.stage](a)
