"""이 코드가 답하는 질문: 외부 baseline (GenDoP · E.T.) 이 학습한 캡션 분포 그대로 우리 test GT 카메라에 캡션을 달면 어떤 문장이 나오나?

R476 (사용자 2026-10-21 "공개 ckpt 돌리는건 해당 모델들이 받는 텍스트 분포대로 돌리는것도 필요해 … 캡션은 다 Qwen3-VL 로 대체").
각 모델의 **원 캡션 파이프라인을 그대로** 태우고 LLM 만 로컬 Qwen3-VL 로 바꾼다. 결과는 eval 폴더 모양
(`<out>/test/<prefix>_<scene>_<idx>_caption.json`, 키 `Concise Interaction`) 이라 `run_gendop_eval --text_dir`·
`run_director_batch --text_dir` 가 그대로 읽는다. tag/outline 은 `<out>/tags/` 에 남긴다.

--model gendop  (DataDoP, `models/GenDoP/dataset/scripts`)
  ① GT c2w (OpenGL) 를 120 pose 로 slerp 리샘플, fps 30, DataDoP 분절기 (27 이동 × 7 회전) — `caption_cameras_datadop.tag_trajectory`
  ② outline → LLM → `Movement` (`llm/cam+rotate.yaml` 그대로, 숫자 섞이면 3 회까지 재질의)
  ③ 4×4 grid + Movement → LLM → `**Detailed**` / `**Concise**` (`llm/relationship+image.json` 그대로) → `Concise Interaction`
  grid 는 `Filtering.py` 규칙 그대로: 앞뒤 2 프레임 버림, int(fps/6) 간격 저장, 그중 len//16 간격 16 장, 폭 900 px.
  영상 = 소스 영상 (target 카메라로 찍은 영상은 없다 — 장면·subject 를 보는 용도).
--model et  (E.T., `models/the-exceptional-trajectories/scripts/processing`)
  ① GT 카메라·subject 궤적을 E.T. 좌표 (run_director_batch 와 같은 변환: G frame → world_to_et, 키 1.7 m 게이지) 로,
     10 fps 49 프레임 → **25 fps** 로 리샘플 (E.T. 분절 창 56 · 최소 구간 25 프레임이 25 fps 기준)
  ② `segment_rigidbody_trajectories` (카메라, static 0.02 · diff 0.4) + `segment_translation_trajectories` (char, 0.22 · 0.32)
  ③ `caption_trajectories` 와 같은 outline·프롬프트 (`llm/prompt/cam+mesh.yaml` = cam+char 판), 샘플링 T 0.6 · top_p 0.95 · top_k 50 ·
     max 300, 숫자가 섞이면 "Remove frame indices." 로 한 번 되묻는다 → `Concise Interaction` 키에 저장.

env: GenDoP (evo · torchtyping · cv2). 예:
    PY=/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python
    $PY exec/native_captions.py --model gendop --out <dir> --limit 4
"""
import json
import os
import sys
import threading
import urllib.request
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from os import makedirs, path

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))     # .../dataset
ET_ROOT = path.join(path.dirname(HERE), "models", "the-exceptional-trajectories")
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
TEXT_KEY = "Concise Interaction"


def post(api, model, messages, **kw):
    req = urllib.request.Request(f"{api}/chat/completions", method="POST",
                                 data=json.dumps({"model": model, "messages": messages, **kw}).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
    return ""


def load_ref(ref_dir, prefix, scene, idx):
    p = path.join(ref_dir, "test", f"{prefix}_{scene}_{idx}_transforms_ref.json")
    with open(p, encoding="utf-8") as f:
        return np.asarray([fr["transform_matrix"] for fr in json.load(f)["frames"]], dtype=np.float64)   # OpenGL c2w


# ------------------------------------------------------------------------------------- GenDoP

def datadop_grid(video, out_png, all_width=900):
    """`Filtering.py` extract (skip 2/2, interval int(fps/6)) + combine (len//16 간격 16 장, 폭 900) 그대로."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(video)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(int(fps / 6), 1)
    kept, i = [], 0
    while cap.isOpened():
        ok, fr = cap.read()
        if not ok or i >= n - 2:
            break
        if i >= 2 and (i - 2) % interval == 0:
            kept.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if len(kept) < 16:
        raise ValueError(f"{video}: 저장 프레임 {len(kept)} < 16")
    imgs = [kept[j] for j in range(0, len(kept), len(kept) // 16)[:16]]
    w = all_width // 4
    hs = [int(w / (im.width / im.height)) for im in imgs]
    rows = [max(hs[r * 4:(r + 1) * 4]) for r in range(4)]
    canvas, y = Image.new("RGB", (all_width, sum(rows))), 0
    for r in range(4):
        for c in range(4):
            canvas.paste(imgs[r * 4 + c].resize((w, hs[r * 4 + c])), (c * w, y))
        y += rows[r]
    makedirs(path.dirname(out_png), exist_ok=True)
    tmp = f"{out_png}.{threading.get_ident()}.tmp.png"      # 같은 씬을 여러 스레드가 동시에 만들 때 반쯤 쓴 파일을 읽지 않게
    canvas.save(tmp)
    os.replace(tmp, out_png)
    return out_png


def gendop_one(a, job, ctx):
    import caption_cameras_datadop as CD      # main 에서 이미 import 됨 (GenDoP 경로 포함)
    scene, idx = job
    # 게이지: DataDoP 은 MonST3R 게이지 (정규화 없음, 장면 depth median ~0.323 = 릴리즈 case depth) 그대로 분절 임계를 쓴다.
    # 우리 GT 는 DA3 world (장면 평균 depth = S, 예 5.83) 라 그대로면 거의 전부 '움직임' 으로 잡힌다 →
    # GenDoP 입력 depth 를 median 정규화한 것 (run_gendop_eval depth_norm=median) 과 같은 뜻으로 위치에 0.323/S 를 곱한다.
    poses = load_ref(a.ref_eval_dir, a.prefix, scene, idx)
    with open(path.join(a.corpus, "dynpose", scene, "da3", "avg_scale_context_first_cam", f"{idx}.json")) as f:
        S = float(json.load(f))
    poses[:, :3, 3] *= a.datadop_depth / S
    tag = CD.tag_trajectory(poses, 120, None, False, convert=False)
    tag["gauge"] = {"S": S, "mul": a.datadop_depth / S}
    grid = path.join(a.out, "grids", f"{scene}.png")
    with ctx["grid_lock"]:
        if not path.isfile(grid):
            datadop_grid(path.join(a.recon_root, scene, "video.mp4"), grid)
    if a.no_llm:
        return tag, None
    pm, pr = ctx["prompts"]
    movement = CD.ask_movement(ctx["client"], pm, tag["outline"])
    detailed, concise = CD.ask_interaction(ctx["client"], pr, grid, movement)
    return tag, {"Movement": movement, "Detailed Interaction": detailed, TEXT_KEY: concise}


# ------------------------------------------------------------------------------------- E.T.

def et_tracks(a, scene, idx, cache):
    """GT 카메라 (OpenCV c2w world) · subject 중심 → E.T. 좌표 (m), run_director_batch.flush 의 역변환."""
    from run_director_vista import graph_basis, world_to_et
    if scene not in cache:
        cache.clear()
        cache[scene] = dict(graph=json.load(open(path.join(a.cloud_root, scene, "scene_graph.json"))),
                            prompts=json.load(open(path.join(a.corpus, "dynpose", scene, "da3", "prompts.json"))),
                            track=np.load(path.join(a.corpus, "dynpose", scene, "da3", "target_track.npz"), allow_pickle=True))
    c = cache[scene]
    graph, tz = c["graph"], c["track"]
    subject = c["prompts"][idx]["variant_id"].split("__")[0]
    node = next((n for n in graph["nodes"] if n["id"] == subject), None)
    ti = [str(k) for k in tz["keys"]].index(idx)
    if node is None or not bool(tz["valid"][ti]):
        return None
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=np.float64)
    S = float(np.linalg.norm(T_wg[:3, 0]))
    inv_A = np.linalg.inv(T_wg[:3, :3])
    centers_u = (tz["track_world"][ti].astype(np.float64) - T_wg[:3, 3]) @ inv_A.T
    mpu = 1.7 / float(node["obb"]["extent"][2])
    up_g, fwd_g = graph_basis(graph, "graph")
    char_m, R_et_w = world_to_et(centers_u, up_g, fwd_g, mpu)
    c2w = load_ref(a.ref_eval_dir, a.prefix, scene, idx) @ GL2CV                 # OpenCV c2w world
    t_g = (c2w[:, :3, 3] - T_wg[:3, 3]) @ inv_A.T
    R_g = np.einsum("ij,fjk->fik", (T_wg[:3, :3] / S).T, c2w[:, :3, :3])
    pose = np.tile(np.eye(4), (len(c2w), 1, 1))
    pose[:, :3, :3] = np.einsum("ij,fjk->fik", R_et_w, R_g)
    pose[:, :3, 3] = (t_g - centers_u[0]) @ R_et_w.T * mpu
    return pose, char_m


def resample_25(pose, char, src_fps=10.0, dst_fps=25.0):
    from scipy.spatial.transform import Rotation, Slerp
    t = np.arange(len(pose)) / src_fps
    tq = np.arange(0.0, t[-1] + 1e-9, 1.0 / dst_fps)
    out = np.tile(np.eye(4), (len(tq), 1, 1))
    out[:, :3, :3] = Slerp(t, Rotation.from_matrix(pose[:, :3, :3]))(tq).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(tq, t, pose[:, k, 3])
    ch = np.stack([np.interp(tq, t, char[:, k]) for k in range(3)], 1)
    return out, ch


def et_one(a, job, ctx):
    import torch
    from src.processing.captioning import find_consecutive_chunks, get_description, CAM_INDEX_TO_PATTERN, BODY_INDEX_TO_PATTERN
    from src.processing.segmentation import segment_rigidbody_trajectories, segment_translation_trajectories
    scene, idx = job
    got = et_tracks(a, scene, idx, ctx.setdefault("cache", {}))
    if got is None:
        return None, None
    pose, char = resample_25(*got)
    cam_seg = segment_rigidbody_trajectories(torch.from_numpy(pose), fps=25.0, cam_static_threshold=0.02, cam_diff_threshold=0.4,
                                             min_chunk_size=25, smoothing_window_size=56)
    char_seg = segment_translation_trajectories(torch.from_numpy(char), fps=25.0, char_static_threshold=0.22, char_diff_threshold=0.32,
                                                min_chunk_size=25, smoothing_window_size=56)
    cam_d = [get_description(i, s, e, CAM_INDEX_TO_PATTERN) for i, s, e in find_consecutive_chunks(list(cam_seg))]
    char_chunks = find_consecutive_chunks(list(char_seg))
    char_d = [get_description(i, s, e, BODY_INDEX_TO_PATTERN) for i, s, e in char_chunks]
    end = char_chunks[-1][2]
    # `caption_trajectories` 의 traj_description 조립 그대로 (Total frames {end} 는 char 마지막 chunk 끝 — 원본 동작)
    desc = (f"\n\nOutline: Total frames {end}; [Camera motion] " + ";".join(cam_d) + ". "
            + "[Main character motion] " + ";".join(char_d) + ".")
    tag = {"outline": desc, "num_frames_25fps": int(len(pose)), "cam_segments": [int(x) for x in cam_seg],
           "char_segments": [int(x) for x in char_seg]}
    if a.no_llm:
        return tag, None
    p = ctx["prompts"]
    msgs = [{"role": "user", "content": f"{p['context']}\n{p['instruction']}\n{p['constraint']}{p['demonstration']}{desc}\nDescription: "}]
    kw = dict(temperature=0.6, top_p=0.95, top_k=50, max_tokens=300)
    cap = post(a.api_base, a.model_name, msgs, **kw)
    if any(ch.isdigit() for ch in cap):                    # 원본: 첫 시도에 숫자면 한 번 되묻는다
        msgs += [{"role": "assistant", "content": cap}, {"role": "user", "content": "Remove frame indices."}]
        cap = post(a.api_base, a.model_name, msgs, **kw)
    cap = cap.replace("Description: ", "").strip()
    return tag, {TEXT_KEY: cap, "outline": desc}


# ------------------------------------------------------------------------------------- main

def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=["gendop", "et"])
    ap.add_argument("--split", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_v5f50k/seg_list_dynpose_s91_noroll_test.txt")
    ap.add_argument("--ref_eval_dir", default="/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/eval_my/"
                    "20261021_003902_dsv5f50kv2_molmo2_l21_noprobe_fm_vaeCrepenclin_zstd_track3d_box3d__last__seed42__v5test")
    ap.add_argument("--corpus", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_v5f50k")
    ap.add_argument("--cloud_root", default=path.join(HERE, "out_dynpose_v5"))
    ap.add_argument("--recon_root", default="/data1/cympyc1785/data/DynPose-100K/eval_data/recon_and_seg")
    ap.add_argument("--prefix", default="dynpose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api_base", default="http://127.0.0.1:22009/v1")
    ap.add_argument("--model_name", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_llm", action="store_true")
    ap.add_argument("--datadop_depth", type=float, default=0.323)   # GenDoP 릴리즈 case depth median (run_gendop_eval docstring)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    for p in (path.join(HERE, "fit", "caption"), path.join(HERE, "exec")):
        sys.path.insert(0, p)
    ctx = {}
    if a.model == "gendop":
        import caption_cameras_datadop as CD
        from lbm.vlm import VLMClient
        ctx["prompts"] = CD.load_prompts()
        ctx["grid_lock"] = threading.Lock()
        # 원본은 OpenAI 기본 샘플링 (temperature 1.0) — 지정이 없으므로 그 기본값을 따른다
        ctx["client"] = None if a.no_llm else VLMClient(api_base=a.api_base, model=a.model_name, temperature=1.0, max_tokens=700, timeout=300)
        fn = gendop_one
    else:
        import yaml
        sys.path.insert(0, ET_ROOT)
        # DIRECTOR 도 `src` 패키지를 갖고 있다 — E.T. 쪽을 먼저 import 해 캐시에 박아 둔다
        import src.processing.captioning  # noqa: F401
        import src.processing.segmentation  # noqa: F401
        ctx["prompts"] = yaml.safe_load(open(path.join(ET_ROOT, "scripts/processing/configs/captioning/llm/prompt/cam+mesh.yaml")))
        fn = et_one

    lines = [l.strip() for l in open(a.split, encoding="utf-8") if l.strip()]
    jobs = [tuple(l.split("/")[1:]) for l in lines][: a.limit or None]
    makedirs(path.join(a.out, "test"), exist_ok=True)
    makedirs(path.join(a.out, "tags"), exist_ok=True)

    def cap_path(s, i):
        return path.join(a.out, "test", f"{a.prefix}_{s}_{i}_caption.json")
    todo = [j for j in jobs if a.overwrite or a.no_llm or not path.isfile(cap_path(*j))]
    print(f"{a.model}: jobs {len(jobs)} · todo {len(todo)}", flush=True)
    stats = {"ok": 0, "skip": 0, "err": 0}

    def run(job, c):
        try:
            tag, cap = fn(a, job, c)
        except Exception as e:  # noqa: BLE001
            stats["err"] += 1
            print(f"[err] {job} {str(e)[:160]}", flush=True)
            return
        if tag is None:
            stats["skip"] += 1
            return
        json.dump(tag, open(path.join(a.out, "tags", f"{job[0]}_{job[1]}_tag.json"), "w"), ensure_ascii=False, indent=1)
        if cap is not None:
            json.dump(cap, open(cap_path(*job), "w"), ensure_ascii=False, indent=2)
        stats["ok"] += 1
        if stats["ok"] % 200 == 0:
            print(f"[{stats['ok']}/{len(todo)}] {stats}", flush=True)

    # E.T. 경로는 scene 캐시를 씬 단위로 쓰므로 같은 씬을 한 스레드에 몰아 준다
    if a.model == "et":
        by_scene = {}
        for j in todo:
            by_scene.setdefault(j[0], []).append(j)
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda js: (lambda c: [run(j, c) for j in js])({**ctx}), by_scene.values()))   # 씬 묶음마다 자기 캐시
    else:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda j: run(j, ctx), todo))
    print(f"=== native captions {a.model} DONE {stats} ===", flush=True)


if __name__ == "__main__":
    main()
