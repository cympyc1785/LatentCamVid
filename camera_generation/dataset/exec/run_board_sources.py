"""TRUMANS board 후보 시작 카메라에서 source video 를 만든다 — chunk 마다 **시작점 2개 x 움직임 2개**.

이 코드가 답하는 질문: "sweep66 board 의 usable 시작 pose 로 같은 시간 창(chunk)에 대해
시작점이 같은 클립 짝 / 시작점이 다른 클립 짝을 어떻게 뽑아 렌더하나".

사용자 지시 (2026-09-23, R22 재계획): "후보군 시작 카메라들로 source video" + "같은 시간에 대해서
시작점이 같은 카메라 2개씩 다른 시작 카메라로 2x2로 최소 4개씩은 pair되도록".

한 chunk 의 산출 (성공 시):

    start A ─┬─ clip p0 (preset X)      같은 시작점 · 다른 움직임  (A.p0, A.p1)
             └─ clip p1 (preset Y≠X)
    start B ─┬─ clip p0                 다른 시작점 · 같은 시간 창 (A.*, B.*)
             └─ clip p1

클립 하나 = `fit/ingest/trumans_to_recon.py --board_json ...` 한 번 (시작 pose 는 board 후보 그대로,
preset 호 합성 → 49프레임 raycast 재검증 → GT 렌더 → recon_and_seg). p1 은 `--exclude_presets
hold <p0 의 preset>` 로 부른다 — `hold` 을 빼는 이유는 짝의 두 클립이 둘 다 **움직여야** 해서다.
시작점에서 움직이는 preset 이 둘 이상 통과하지 못하면 그 시작점은 버리고 다음 후보로 간다
(p0 만 남은 클립은 지우지 않고 `unpaired` 로 기록한다).

**GPU.** EEVEE 는 EGL 로 장치를 잡아 `CUDA_VISIBLE_DEVICES` 를 무시한다 (2026-09-23 실측:
GPU 2 로 줬는데 GPU 0 에 떴다). 그래서 RGB 는 Cycles(`--rgb_engine cycles`, 기본 32 spp +
denoise) 로 렌더한다 — Cycles 는 CUDA 를 따른다.

    python exec/run_board_sources.py --stage plan
    python exec/run_board_sources.py --stage render --gpus 2 3 --workers_per_gpu 3
    python exec/run_board_sources.py --stage report
"""
import json
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from os import makedirs, path
from threading import Lock
from zlib import crc32

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))          # camera_generation/dataset
PY = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"
TO_RECON = path.join(HERE, "fit", "ingest", "trumans_to_recon.py")
# s3f0k6 뱅크(`out/trumans_lite_bank_s3f0k6/bank_manifest.json`)와 같은 합성 설정.
RECON_ARGS = ["--frame_step", "3", "--preset_scale", "3.0", "--aim_keyframes", "6",
              "--max_tries", "8"]


def plan_order(cands, seed_key):
    """usable 후보를 **연속한 두 개가 서로 다른 시작점**이 되게 줄 세운다.

    (반경, 90° 방위 구역) 버킷으로 나눠 버킷 순서를 섞고 라운드로빈으로 뽑는다 — 1지망 A 와
    2지망 B 가 같은 칸 이웃이면 "다른 시작 카메라" 짝이 사실상 같은 카메라가 된다.
    """
    rng = np.random.default_rng(crc32(seed_key.encode()))
    buckets = {}
    for i, c in cands:
        buckets.setdefault((round(float(c["radius"]), 2), int(c["azimuth_deg"] // 90)), []).append(i)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])
    order = []
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                order.append(int(buckets[k].pop()))
    return order


def stage_plan(a):
    rows = []
    for bj in sorted(glob(path.join(HERE, a.board_root, "*", "board.json"))):
        rec = path.basename(path.dirname(bj))
        board = json.load(open(bj, encoding="utf-8"))
        for ch in board["chunks"]:
            cands = [(i, c) for i, c in enumerate(ch["candidates"]) if c["usable"]]
            rows.append({"recording": rec, "board_json": bj, "chunk": ch["tag"],
                         "n_usable": len(cands),
                         "order": plan_order(cands, f"{rec}|{ch['tag']}|{a.seed}")})
    makedirs(path.dirname(a.plan), exist_ok=True)
    with open(a.plan, "w") as f:
        json.dump(rows, f)
    print(f"[plan] recordings {len({r['recording'] for r in rows})}  chunks {len(rows)}  "
          f"usable {sum(r['n_usable'] for r in rows)}  -> {a.plan}", flush=True)


def far_enough(board_cands, i, j, min_az):
    """두 시작점이 '다른 카메라' 인가 — 반경이 다르거나 방위각이 min_az 이상 떨어졌다."""
    ci, cj = board_cands[i], board_cands[j]
    daz = abs((float(ci["azimuth_deg"]) - float(cj["azimuth_deg"]) + 180.0) % 360.0 - 180.0)
    return abs(float(ci["radius"]) - float(cj["radius"])) > 1e-6 or daz >= min_az


def run_clip(a, row, cand, slot, exclude, gpu, log_dir):
    """clip 1개. -> (ok, preset, video)"""
    video = f"tru_{row['recording'][:8]}_{row['chunk']}_k{cand:03d}_p{slot}"
    man = path.join(HERE, "out", "trumans_recon", row["recording"],
                     f"manifest_{row['chunk']}_k{cand:03d}_p{slot}.json")
    if path.exists(man):                                   # 이미 만든 클립 (재실행 안전)
        m = json.load(open(man))
        return True, m["source_camera"]["preset"], video
    cmd = [PY, TO_RECON, "--recording", row["recording"], "--board_json", row["board_json"],
           "--board_chunk", row["chunk"], "--board_candidate", str(cand),
           "--board_slot", str(slot), "--exclude_presets", *exclude,
           "--rgb_engine", "cycles", "--rgb_samples", str(a.rgb_samples),
           # D276 (사용자 채택): RGB `--anim` 1 job + depth/index 별 job. 같은 조건 A/B 에서
           # 666.9 s -> 343.8 s (depth 비트 동일, rgb mean|Δ| 0.2~0.3/255 = 샘플 노이즈).
           "--rgb_anim",
           # D278 (사용자 지시 "바꿔줘"): probe(grid/verify) 를 recording 상주 Blender 서버에서.
           # grid probe JSON 이 새 Blender 판과 전체 일치 (00add26c c01 k029, 576 후보).
           "--probe_server",
           "--eval_data", a.eval_data] + RECON_ARGS
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8", "PATH": "/usr/bin:/bin"}
    import os
    env = {**os.environ, **env}
    with open(path.join(log_dir, f"{video}.log"), "w") as log:
        p = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=log, env=env, timeout=a.timeout)
    if p.returncode != 0 or not path.exists(man):
        return False, None, video
    return True, json.load(open(man))["source_camera"]["preset"], video


def do_chunk(a, row, gpu, log_dir, status_dir):
    sp = path.join(status_dir, f"{row['recording'][:8]}_{row['chunk']}.json")
    if path.exists(sp):
        st = json.load(open(sp))
        if st.get("done"):
            return st
    board = json.load(open(row["board_json"], encoding="utf-8"))
    bc = next(c for c in board["chunks"] if c["tag"] == row["chunk"])["candidates"]
    st = {"recording": row["recording"], "chunk": row["chunk"], "pairs": [], "unpaired": [],
          "failed_starts": [], "tried": 0}
    for cand in row["order"][:a.max_starts]:
        if len(st["pairs"]) >= a.starts:
            break
        if any(not far_enough(bc, cand, p["candidate"], a.min_az) for p in st["pairs"]):
            continue
        st["tried"] += 1
        ok0, pr0, v0 = run_clip(a, row, cand, 0, ["hold"], gpu, log_dir)
        if not ok0:
            st["failed_starts"].append(cand)
            continue
        clips = [{"video": v0, "preset": pr0}]
        excl = ["hold", pr0]
        for slot in range(1, a.per_start):
            ok, pr, v = run_clip(a, row, cand, slot, excl, gpu, log_dir)
            if not ok:
                break
            clips.append({"video": v, "preset": pr})
            excl.append(pr)
        if len(clips) >= a.per_start:
            st["pairs"].append({"candidate": cand, "clips": clips})
        else:
            st["unpaired"].append({"candidate": cand, "clips": clips})
    st["done"] = True
    st["complete"] = len(st["pairs"]) >= a.starts
    with open(sp, "w") as f:
        json.dump(st, f, indent=1)
    return st


def stage_render(a):
    rows = json.load(open(a.plan))
    rows = [r for i, r in enumerate(rows) if i % a.num_shards == a.shard_id]
    log_dir = path.join(a.work, "log")
    status_dir = path.join(a.work, "status")
    makedirs(log_dir, exist_ok=True)
    makedirs(status_dir, exist_ok=True)
    slots = [g for g in a.gpus for _ in range(a.workers_per_gpu)]
    lock, stat, t0 = Lock(), {"n": 0, "complete": 0, "clips": 0}, time.time()
    free = list(slots)

    def job(row):
        with lock:
            gpu = free.pop(0)
        try:
            st = do_chunk(a, row, gpu, log_dir, status_dir)
        finally:
            with lock:
                free.append(gpu)
        with lock:
            stat["n"] += 1
            stat["complete"] += int(st["complete"])
            stat["clips"] += sum(len(p["clips"]) for p in st["pairs"] + st["unpaired"])
            el = time.time() - t0
            print(f"[render] {stat['n']}/{len(rows)} {row['recording'][:8]} {row['chunk']} "
                  f"pairs {len(st['pairs'])} unpaired {len(st['unpaired'])} "
                  f"fail {len(st['failed_starts'])}  | complete {stat['complete']} "
                  f"clips {stat['clips']}  {el / 3600:.2f} h", flush=True)

    with ThreadPoolExecutor(len(slots)) as ex:
        list(ex.map(job, rows))
    print(f"[render] ALL DONE {stat}", flush=True)


def stage_report(a):
    sts = [json.load(open(p)) for p in sorted(glob(path.join(a.work, "status", "*.json")))]
    n = len(sts)
    comp = sum(s["complete"] for s in sts)
    clips = sum(len(p["clips"]) for s in sts for p in s["pairs"])
    unp = sum(len(p["clips"]) for s in sts for p in s["unpaired"])
    print(f"chunks {n}  complete(2x2) {comp}  paired clips {clips}  unpaired clips {unp}")


if __name__ == "__main__":
    q = ArgumentParser()
    q.add_argument("--stage", required=True, choices=("plan", "render", "report"))
    q.add_argument("--board_root", default="out/sweep66")
    q.add_argument("--work", default="/data1/cympyc1785/LatentCamVid/tmp/r22b")
    q.add_argument("--plan", default="/data1/cympyc1785/LatentCamVid/tmp/r22b/plan.json")
    q.add_argument("--eval_data", default="/data1/cympyc1785/data/TRUMANS-Lite")
    q.add_argument("--seed", default=0, type=int)
    q.add_argument("--starts", default=2, type=int)          # chunk 당 서로 다른 시작점 수
    q.add_argument("--per_start", default=2, type=int)       # 시작점 당 (서로 다른 preset) 클립 수
    q.add_argument("--max_starts", default=10, type=int)     # chunk 당 시도할 시작점 상한
    q.add_argument("--min_az", default=60.0, type=float)     # 같은 반경이면 방위각 이만큼은 떨어져야
    q.add_argument("--gpus", default=[2, 3], nargs="+", type=int)
    q.add_argument("--workers_per_gpu", default=3, type=int)
    q.add_argument("--rgb_samples", default=16, type=int)      # D281 (사용자 지시): 32 -> 16 (PSNR 42.3 -> 39.6 dB, RGB 133.7 -> 103.6 s)
    q.add_argument("--timeout", default=3600, type=int)
    q.add_argument("--num_shards", default=1, type=int)
    q.add_argument("--shard_id", default=0, type=int)
    a = q.parse_args()
    assert all(g in (0, 1, 2, 3) for g in a.gpus), f"GPU 는 0~3 만 (CLAUDE.md): {a.gpus}"
    {"plan": stage_plan, "render": stage_render, "report": stage_report}[a.stage](a)
