"""D169 — dynpose-100k 대량 ingest 드라이버 (recon_and_seg → metadata → SAM3 → dynamic_mask).

왜 새 드라이버인가. 기존 경로는 `recon_and_seg_single.py` 를 **영상 하나당 프로세스 하나**로
띄우는 bash 래퍼였다. DA3NESTED-GIANT-LARGE 는 로드에만 수십 초가 걸리는데 추론은 영상당 1분
남짓이라, 1만 편이면 모델 로딩이 전체 시간의 절반을 먹는다. 이 드라이버는 **프로세스당 DA3 를
한 번만 올리고 배정된 영상을 전부 도는** 구조로 바꾼다. 산출물은 `recon_and_seg_single.py` 를
같은 인자로 돌린 것과 동일하다 (본문을 그대로 옮겼다 — §recon_scene 주석 참조).

단계는 서로 **배리어**로 나뉜다. 사용자 지시(2026-09-08): "모델 올렸다 내렸다 하면 오래 걸리니까
단계 나눠서 최대한 병렬로 다 돌리고 다 되면 다음 단계로 넘어가는 식". recon 은 DA3 만, sam3 는
SAM3 만 올린 채로 끝까지 돈다. 두 모델이 한 GPU 에 같이 올라가는 구간이 없다.

    link      기존 DynPose-LBM 880편(shard 0000)을 새 루트에 symlink — 재계산하지 않는다
    recon     video_input.mp4 → recon_and_seg/<uuid>/{video.mp4,depths,conf,cameras.npz,*_mask} [GPU]
    nouns     extract_nouns_vlm.py (우리 프롬프트, 6프레임) → shard 별 vlm_nouns.json     [vLLM]
    metadata  명사 → metadata.csv  (`--noun_source vlm`(기본) / `category`)                 [CPU]
    sam3      scripts/sam3_seg_instances.py 위임 (SAM3 도 이미 프로세스당 1회 로드)          [GPU]
    dynmask   scripts/dynpose_dynamic_mask_from_seg.py 위임 — seg 인스턴스 합집합으로 덮어씀 [CPU]
    launch    위 단계 하나를 GPU 여러 장에 부채꼴로 띄우고 전부 끝날 때까지 기다린다

**입력은 `video_input.mp4` 뿐이다.** `inpaint_result.mp4` 는 쓰지 않는다 (사용자 지시
2026-09-08) — 그건 저자들이 이미 워프·인페인트를 돌린 결과물이라 소스 영상이 아니다.

**출력 루트를 `DATA/DynPose-LBM` 이 아니라 새로 판다.** 기존 루트의 `metadata.csv` 는 d157/d166/
d168 뱅크가 코퍼스 목록으로 읽는 파일이라, 여기에 9천 편을 더하면 그 뱅크들을 다시 굽지 못한다.
shard 0000 은 symlink 로 재사용하므로 디스크는 안 는다.

사용 예시:

    # 0) 기존 880편 재사용 + 1만 편 목록 확인
    python scripts/dynpose_ingest.py --stage link
    python scripts/dynpose_ingest.py --stage recon --dry_run

    # 1) recon 을 GPU 0~4 다섯 장에 부채꼴로 (screen 안에서 돌린다)
    python scripts/dynpose_ingest.py --stage launch --launch_stage recon --gpus 0,1,2,3,4

    # 2) 전부 끝나면 다음 단계. nouns 는 vLLM 서버를 **밖에서 먼저 띄운 뒤** 돌린다
    #    (vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --port 22002 ... ; 끝나면 내린다)
    python scripts/dynpose_ingest.py --stage launch --launch_stage nouns --workers 8
    python scripts/dynpose_ingest.py --stage metadata
    python scripts/dynpose_ingest.py --stage launch --launch_stage sam3 --gpus 0,1,2,3,4
    python scripts/dynpose_ingest.py --stage launch --launch_stage dynmask --workers 8
"""

import csv
import json
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from os import environ, listdir, makedirs, path, rename, symlink
from shutil import rmtree
from time import time

HERE = path.dirname(path.dirname(path.abspath(__file__)))                    # .../CinemaTraj
REPO = "/data1/cympyc1785/LatentCamVid"
VISTA4D_ROOT = path.join(REPO, "video_generation", "models", "Vista4D")
SAM3_SCRIPT = path.join(REPO, "video_generation", "scripts", "sam3_seg_instances.py")
DYNMASK_SCRIPT = path.join(HERE, "scripts", "dynpose_dynamic_mask_from_seg.py")
NOUNS_SCRIPT = path.join(HERE, "scripts", "extract_nouns_vlm.py")
VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
VLM_API_BASE = "http://127.0.0.1:22002/v1"
NOUN_FRAMES = 6          # 사용자 지시 2026-09-08 "우리 방식대로 뽑되 프레임은 6개로"

SRC_ROOT_DEFAULT = "/data1/cympyc1785/data/worldtraj/dynamicverse/dynpose-100k"
OUT_ROOT_DEFAULT = path.join(REPO, "DATA", "DynPose-100K")
# shard 0000 은 이미 DynPose-LBM 에 recon 이 있다. link 단계가 여기서 끌어다 쓴다.
DONE_ROOT_DEFAULT = path.join(REPO, "DATA", "DynPose-LBM")

# recon_and_seg_single.py 를 dynpose 코퍼스에 쓸 때의 인자 (D145 이후 고정)
RECON_METHOD = "da3"
DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
NUM_FRAMES = 49
HEIGHT, WIDTH = 720, 1280


# ---------------------------------------------------------------- scene 목록

def parse_shards(spec: str):
    """'0000-0011' / '0001,0003' / '0000-0002,0007' → ['0000', ...] (정렬·중복 제거)."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(f"{i:04d}" for i in range(int(lo), int(hi) + 1))
        else:
            out.add(f"{int(part):04d}")
    return sorted(out)


def list_scenes(src_root: str, shards):
    """[(video_id, scene_dir)] — video_input.mp4 가 있는 scene 만, shard→uuid 정렬 순."""
    scenes, missing = [], 0
    for shard in shards:
        shard_dir = path.join(src_root, f"dynpose-{shard}")
        assert path.isdir(shard_dir), f"shard 디렉토리가 없다: {shard_dir}"
        for vid in sorted(listdir(shard_dir)):
            scene_dir = path.join(shard_dir, vid)
            if not path.isdir(scene_dir):
                continue
            if not path.isfile(path.join(scene_dir, "video_input.mp4")):
                missing += 1
                continue
            scenes.append((vid, scene_dir))
    return scenes, missing


def recon_done(recon_dir: str, num_frames: int):
    """산출물이 온전한가. 중간에 죽은 폴더를 '완료'로 세지 않기 위한 검사."""
    if not path.isfile(path.join(recon_dir, "cameras.npz")):
        return False
    if not path.isfile(path.join(recon_dir, "video.mp4")):
        return False
    for sub in ("depths", "dynamic_mask", "sky_mask"):
        d = path.join(recon_dir, sub)
        if not path.isdir(d) or len(listdir(d)) != num_frames:
            return False
    # `conf` 는 **일부러 검사하지 않는다** (D169). 2026-09-09 에 conf 저장을 켰는데, 이걸
    # 완료 조건에 넣으면 그 전에 구운 6천여 편이 전부 '미완료'로 되살아나 재계산된다.
    # conf 는 있으면 쓰고 없으면 마는 부가 산출물이다 (`scene_graph/lift.py` 의 conf=None 분기).
    return True


# ---------------------------------------------------------------- stage: link

def stage_link(args):
    """기존 DynPose-LBM recon(shard 0000, 880편)을 새 루트에 symlink 한다."""
    src = path.join(args.done_root, "eval_data", "recon_and_seg")
    dst = path.join(args.out_root, "eval_data", "recon_and_seg")
    makedirs(dst, exist_ok=True)
    if not path.isdir(src):
        print(f"  건너뜀: {src} 가 없다")
        return
    linked = existing = 0
    for vid in sorted(listdir(src)):
        target, link = path.join(src, vid), path.join(dst, vid)
        if path.exists(link) or path.islink(link):
            existing += 1
            continue
        if not args.dry_run:
            symlink(target, link)
        linked += 1
    print(f"  symlink 생성 {linked}   이미 있음 {existing}   → {dst}")


# ---------------------------------------------------------------- stage: recon

def recon_scene(video_path: str, out_dir: str, da3_model, args, run_da3, media):
    """`recon_and_seg_single.main()` 에서 이 코퍼스가 타는 분기만 그대로 옮긴 것.

    고정된 값들: dse_video_path=None / recon_method="da3" / scene_scale=1.0 /
    seg_keywords=[] / keep_recon_sky=True / save_seg_instances=False / save_vis=False.
    그래서 원본의 DSE concat · pi3 · SAM3 · vis 분기는 여기서 죽은 코드다. 남은 순서는
    load → slice_center → crop_resize → run_da3 → (dynamic_mask=0, sky_mask 유지) → save 5종.
    """
    video, fps = media["load_video"](video_path)
    n_src = video.shape[0]
    assert n_src >= args.num_frames, f"{n_src} < num_frames={args.num_frames}"
    video = media["slice_center_frames"](video, args.num_frames)
    video = media["crop_and_resize_video"](video, args.height, args.width, resample="lanczos")

    out = run_da3(
        video, model=da3_model, process_res=args.da3_process_res, resize_output=True,
        return_conf=args.save_conf,
    )
    depths, sky_mask, cam_c2w, intrinsics = out[:4]
    conf = out[4] if args.save_conf else None
    media["cleanup"]()

    num_frames, height, width, _ = video.shape
    # seg_keywords 가 비었으므로 dynamic_mask 는 0. 나중에 dynmask 단계가 SAM3 인스턴스
    # 합집합으로 덮어쓴다. sky_mask 는 --keep_recon_sky 와 같이 DA3 것을 살린다.
    dynamic_mask = media["np"].zeros((num_frames, height, width), dtype=media["np"].bool_)
    media["cleanup"]()

    media["save_video"](path.join(out_dir, "video.mp4"), video, fps=fps, quality=9)
    media["save_depths"](path.join(out_dir, "depths"), depths, dtype=media["np"].float16)
    media["save_cameras"](path.join(out_dir, "cameras.npz"), cam_c2w, intrinsics)
    media["save_masks"](path.join(out_dir, "dynamic_mask"), dynamic_mask)
    media["save_masks"](path.join(out_dir, "sky_mask"), sky_mask)
    if conf is not None:
        # depths 와 같은 float16 EXR 포맷 — `load_depths(".../conf")` 로 그대로 읽힌다.
        # 값 범위는 [1, inf) 이지 [0,1] 이 아니다 (expp1). 임계는 반드시 상대값으로.
        media["save_depths"](path.join(out_dir, "conf"), conf, dtype=media["np"].float16)


def stage_recon(args, scenes):
    out_base = path.join(args.out_root, "eval_data", "recon_and_seg")
    makedirs(out_base, exist_ok=True)

    todo = []
    for vid, scene_dir in scenes:
        out_dir = path.join(out_base, vid)
        if args.skip_done and (path.islink(out_dir) or recon_done(out_dir, args.num_frames)):
            continue
        todo.append((vid, scene_dir, out_dir))

    print(f"  배정 {len(scenes)}   할 일 {len(todo)}   완료/링크 {len(scenes) - len(todo)}")
    if args.dry_run or not todo:
        return

    # DA3 를 **여기서 한 번만** 올린다. 원본은 영상마다 init_da3/del 을 반복했다.
    sys.path.insert(0, VISTA4D_ROOT)
    import numpy as np
    from utils.media import (
        crop_and_resize_video, load_video, save_cameras, save_depths, save_masks, save_video,
        slice_center_frames,
    )
    from utils.misc import cleanup
    from utils.recon_and_seg.recon_da3 import init_da3, run_da3
    import torch

    media = {
        "np": np, "load_video": load_video, "save_video": save_video, "save_depths": save_depths,
        "save_cameras": save_cameras, "save_masks": save_masks, "cleanup": cleanup,
        "slice_center_frames": slice_center_frames, "crop_and_resize_video": crop_and_resize_video,
    }

    t_load = time()
    da3_model = init_da3(model_id=args.da3_model_id, device="cuda")
    print(f"  DA3 로드 {time() - t_load:.1f}s  (이 프로세스에서 1회)", flush=True)

    ok = fail = 0
    t0 = time()
    for i, (vid, scene_dir, out_dir) in enumerate(todo):
        # 중간에 죽은 폴더가 '완료'로 보이지 않도록 임시 폴더에 쓰고 마지막에 rename 한다.
        tmp_dir = out_dir + ".partial"
        if path.isdir(tmp_dir):
            rmtree(tmp_dir)
        makedirs(tmp_dir, exist_ok=True)
        t_scene = time()
        try:
            with torch.no_grad():
                recon_scene(path.join(scene_dir, "video_input.mp4"), tmp_dir, da3_model,
                            args, run_da3, media)
            if path.isdir(out_dir):
                rmtree(out_dir)
            rename(tmp_dir, out_dir)
            ok += 1
            status = "OK"
        except Exception as exc:                                  # 한 편이 죽어도 계속 간다
            rmtree(tmp_dir, ignore_errors=True)
            fail += 1
            status = f"FAIL {type(exc).__name__}: {exc}"[:120]
        dt = time() - t_scene
        eta = (time() - t0) / (i + 1) * (len(todo) - i - 1) / 3600.0
        print(f"  [{i + 1}/{len(todo)}] {vid}  {status}  {dt:.1f}s  ETA {eta:.2f}h", flush=True)

    print(f"  ---- recon OK {ok} / FAIL {fail} / {len(todo)}  ({datetime.now():%H:%M:%S})",
          flush=True)


# ---------------------------------------------------------------- stage: metadata

def noun_records(args):
    """`--stage nouns` 가 남긴 shard 별 `vlm_nouns.json` 을 하나로 합친다 → {video: [명사]}.

    같은 video 가 여러 shard 에 있을 수 없으므로 충돌 처리는 없다. `frame_mode` 는 `multi`
    만 본다 — `single`(frame0 1장)은 detector 대조용이라 여기 쓰면 어휘가 얇아진다.
    """
    out = {}
    for blob_path in sorted(glob(path.join(args.nouns_dir, "*", "vlm_nouns.json"))):
        with open(blob_path, encoding="utf-8") as f:
            for rec in json.load(f).get("records", []):
                if rec.get("frame_mode") != "multi" or rec.get("source") != "vlm":
                    continue
                nouns = [str(n).strip() for n in (rec.get("dynamic") or []) if str(n).strip()]
                if nouns:
                    out[rec["video"]] = nouns
    return out


def stage_nouns(args, scenes):
    """recon 이 끝난 영상 → `extract_nouns_vlm.py`(우리 프롬프트) → shard 별 `vlm_nouns.json`.

    **왜 배포본 `category/category.json` 을 안 쓰나** (사용자 지시 2026-09-08 "우리 방식대로
    뽑되 프레임은 6개로 늘려줘"): 그쪽 명사는 같은 Qwen3-VL-30B 가 낸 것이지만 프롬프트가
    "[형용사] + [명사]" 구를 **강제**하고 `reasoning` 문장과 짝지어 낸다. 하류 세그멘터가
    referring-expression 을 먹는 Sa2VA 라서 그렇게 설계된 것이고, 우리 `sam3_seg_instances.py`
    는 metadata.csv 의 평명사만 읽어 SAM3 text PCS 에 넣는다 — `man holding dog` 같은 합성구는
    track 하나로 뭉치거나 아예 안 잡힌다. 게다가 기존 880편 코퍼스는 D145 우리 프롬프트로
    뽑은 평명사 4~5개라, 섞으면 shard 0000 과 나머지의 앵커 후보가 구조적으로 갈린다.

    vLLM 서버(포트 22002)는 **밖에서 미리 띄워 둔다** — 여기서 올렸다 내리면 샤드마다 로드가
    반복된다. 샤드 워커는 전부 같은 서버를 두드리므로 GPU 는 서버 것 한 장뿐이고, 워커 수는
    서버 배치 처리량에 맞춘 동시 요청 수다 (`--stage launch --launch_stage nouns --workers N`).
    """
    recon_root = path.join(args.out_root, "eval_data", "recon_and_seg")
    ready = [vid for vid, _ in scenes
             if path.isfile(path.join(recon_root, vid, "video.mp4"))]
    out_dir = path.join(args.nouns_dir, f"shard{args.shard_id}")
    have = set()
    if args.skip_done:
        blob = path.join(out_dir, "vlm_nouns.json")
        if path.isfile(blob):
            with open(blob, encoding="utf-8") as f:
                have = {r["video"] for r in json.load(f).get("records", [])
                        if r.get("frame_mode") == "multi" and r.get("source") == "vlm"}
    todo = [v for v in ready if v not in have]
    print(f"  배정 {len(scenes)}   recon 완료 {len(ready)}   이미 명사 있음 {len(have)}   "
          f"할 일 {len(todo)}")
    if args.dry_run or not todo:
        return 0
    makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-u", NOUNS_SCRIPT,
           "--eval_data", args.out_root, "--output", out_dir,
           "--frame_mode", "multi", "--num_frames", str(args.noun_frames),
           "--api_base", args.api_base, "--model", args.vlm_model, "--merge",
           "--videos"] + todo
    return run_child(cmd, args)


def stage_metadata(args, scenes):
    """dynamic 명사 → `<out_root>/metadata.csv`. 출처는 `--noun_source`.

    `vlm`(기본, D169) — `--stage nouns` 가 우리 프롬프트로 뽑은 `vlm_nouns.json`.
    `category` — 배포본 `category/category.json` (옛 동작). 어휘가 왜 다른지는 `stage_nouns`.

    어느 쪽이든 shard 0000 은 **기존 DynPose-LBM metadata.csv 행을 그대로** 쓴다. 이미 그
    명사로 SAM3 를 돌려 뱅크까지 구운 880편이고, `--stage link` 가 그 `recon_and_seg` 를
    symlink 로 재사용하고 있어서 지금 명사를 갈아끼우면 링크된 산출물과 어긋난다.
    """
    rows, src_counts = {}, {"lbm_csv": 0, "vlm": 0, "category": 0, "empty": 0}

    done_csv = path.join(args.done_root, "metadata.csv")
    if path.isfile(done_csv):
        with open(done_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["video"]] = r["dynamic"]
                src_counts["lbm_csv"] += 1

    from_vlm = noun_records(args) if args.noun_source == "vlm" else {}
    for vid, scene_dir in scenes:
        if vid in rows:
            continue
        nouns = []
        if args.noun_source == "vlm":
            nouns = from_vlm.get(vid, [])
            key = "vlm"
        else:
            cat_path = path.join(scene_dir, "category", "category.json")
            if path.isfile(cat_path):
                with open(cat_path, encoding="utf-8") as f:
                    nouns = [str(n).strip() for n in (json.load(f).get("dynamic") or [])
                             if str(n).strip()]
            key = "category"
        if not nouns:
            # 명사가 없으면 SAM3 가 돌 게 없다 — 행을 만들면 seg_instances 가 비어 하류가 죽는다.
            src_counts["empty"] += 1
            continue
        rows[vid] = ", ".join(nouns)
        src_counts[key] += 1

    out_csv = path.join(args.out_root, "metadata.csv")
    print(f"  기존 LBM csv {src_counts['lbm_csv']}   "
          f"{args.noun_source} 추가 {src_counts[args.noun_source]}   "
          f"명사 0개라 제외 {src_counts['empty']}   최종 {len(rows)}")
    if args.dry_run:
        return
    makedirs(args.out_root, exist_ok=True)
    if path.isfile(out_csv):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rename(out_csv, f"{out_csv}.bak_{stamp}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video", "dynamic"])
        w.writeheader()
        for vid in sorted(rows):
            w.writerow({"video": vid, "dynamic": rows[vid]})
    prov = path.join(args.out_root, "metadata_provenance_d169.json")
    with open(prov, "w", encoding="utf-8") as f:
        json.dump({"format": "dynpose100k_metadata_provenance_v1",
                   "updated": datetime.now().isoformat(timespec="seconds"),
                   "noun_source": args.noun_source,
                   # note 를 하드코딩하면 --noun_source 를 바꿔도 옛 설명이 남는다 (실제로 D169 에서
                   # vlm 으로 돌렸는데 "category.json" 이라고 적혔다). 분기해서 쓴다.
                   "note": ("shard 0000 은 DynPose-LBM metadata.csv(D145 VLM 명사) 그대로, "
                            + ("나머지는 우리 프롬프트로 뽑은 vlm_nouns.json 의 dynamic 배열."
                               if args.noun_source == "vlm"
                               else "나머지는 dynpose-100k category/category.json 의 dynamic 배열.")),
                   "counts": src_counts}, f, ensure_ascii=False, indent=2)
    print(f"  → {out_csv}")


# ---------------------------------------------------------------- stage: sam3 / dynmask (위임)

def stage_sam3(args, scenes):
    """`video_generation/scripts/sam3_seg_instances.py` 에 위임. 그쪽이 이미 SAM3 를 1회만 올린다."""
    cmd = [sys.executable, SAM3_SCRIPT,
           "--eval_data", args.out_root, "--vista4d_root", VISTA4D_ROOT,
           "--num_shards", str(args.num_shards), "--shard_id", str(args.shard_id)]
    cmd += ["--skip_done"] if args.skip_done else ["--no_skip_done"]
    return run_child(cmd, args)


def stage_dynmask(args, scenes):
    """seg 인스턴스 합집합으로 `dynamic_mask/` 를 덮어쓴다 (CPU)."""
    seg_root = path.join(args.out_root, "eval_data", "seg_instances")
    have = sorted(path.basename(p) for p in glob(path.join(seg_root, "*")))
    mine = have[args.shard_id::args.num_shards]
    print(f"  seg_instances {len(have)}   이 샤드 {len(mine)}")
    if args.dry_run or not mine:
        return 0
    cmd = [sys.executable, DYNMASK_SCRIPT, "--eval_data", args.out_root, "--videos"] + mine
    return run_child(cmd, args)


def run_child(cmd, args):
    print("  $ " + " ".join(cmd[:6]) + " ...", flush=True)
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


# ---------------------------------------------------------------- stage: launch

def stage_launch(args, scenes):
    """한 단계를 GPU 여러 장(또는 CPU 워커 여러 개)에 부채꼴로 띄우고 전부 끝날 때까지 기다린다.

    단계 사이에 배리어를 두는 게 목적이므로, 여기서는 **한 단계만** 띄운다. 다음 단계는
    이게 끝난 뒤 별도 호출로 간다 (사용자 지시: "다 되면 다음 단계로").
    """
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    for g in gpus:
        assert 0 <= int(g) <= 4, f"GPU 는 0~4 만 쓴다 (프로젝트 규칙). 받은 값: {g}"
    n = len(gpus) if gpus else args.workers
    assert n > 0, "--gpus 나 --workers 중 하나는 있어야 한다"

    log_dir = path.join(args.work_dir, f"{args.launch_stage}_logs")
    makedirs(log_dir, exist_ok=True)

    procs = []
    for i in range(n):
        cmd = [sys.executable, path.abspath(__file__),
               "--stage", args.launch_stage,
               "--src_root", args.src_root, "--out_root", args.out_root,
               "--done_root", args.done_root, "--shards", args.shards,
               "--num_shards", str(n), "--shard_id", str(i),
               "--num_frames", str(args.num_frames), "--work_dir", args.work_dir,
               "--nouns_dir", args.nouns_dir, "--noun_frames", str(args.noun_frames),
               "--api_base", args.api_base, "--vlm_model", args.vlm_model,
               "--noun_source", args.noun_source]
        cmd += ["--skip_done"] if args.skip_done else ["--no_skip_done"]
        cmd += ["--save_conf"] if args.save_conf else ["--no_save_conf"]
        env = dict(environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[i]
        log_path = path.join(log_dir, f"shard{i}.log")
        log = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        procs.append((i, p, log, log_path))
        print(f"  launch shard {i}/{n}  gpu={gpus[i] if gpus else '-'}  pid={p.pid}  {log_path}",
              flush=True)

    t0 = time()
    rcs = []
    for i, p, log, log_path in procs:
        rc = p.wait()
        log.close()
        rcs.append(rc)
        print(f"  shard {i} 종료 rc={rc}  ({(time() - t0) / 3600:.2f}h)", flush=True)
    print(f"  ---- launch {args.launch_stage} 완료  rc={rcs}  총 {(time() - t0) / 3600:.2f}h",
          flush=True)
    return 0 if all(rc == 0 for rc in rcs) else 1


# ---------------------------------------------------------------- main

def main():
    ap = ArgumentParser(description="dynpose-100k ingest 드라이버 (단계별 배리어 + 샤딩)")
    ap.add_argument("--stage", required=True,
                    choices=("link", "recon", "nouns", "metadata", "sam3", "dynmask", "launch"))
    ap.add_argument("--launch_stage", default="recon",
                    choices=("recon", "nouns", "sam3", "dynmask"))  # --stage launch 가 띄울 단계
    ap.add_argument("--src_root", default=SRC_ROOT_DEFAULT)        # dynpose-100k 원본
    ap.add_argument("--out_root", default=OUT_ROOT_DEFAULT)        # 새 eval_data 루트
    ap.add_argument("--done_root", default=DONE_ROOT_DEFAULT)      # 재사용할 기존 코퍼스
    ap.add_argument("--shards", default="0000-0011")               # dynpose-NNNN 범위
    ap.add_argument("--work_dir", default=path.join(REPO, "tmp", "d169"))

    ap.add_argument("--num_shards", default=1, type=int)           # 워커 수
    ap.add_argument("--shard_id", default=0, type=int)             # 이 워커 인덱스
    ap.add_argument("--gpus", default=None)                        # launch 용, 예 "0,1,2,3,4"
    ap.add_argument("--workers", default=1, type=int)              # launch 용 CPU 워커 수

    ap.add_argument("--num_frames", default=NUM_FRAMES, type=int)
    ap.add_argument("--height", default=HEIGHT, type=int)
    ap.add_argument("--width", default=WIDTH, type=int)
    ap.add_argument("--da3_model_id", default=DA3_MODEL_ID)
    ap.add_argument("--da3_process_res", default=-1, type=int)     # <=0 이면 width 로 맞춘다
    # DA3 depth confidence 를 conf/ 에 같이 남긴다 (사용자 지시 2026-09-09). 편당 +18 MB.
    # 값은 expp1 이라 [1, inf) — 절대 임계 금지, non-sky 중앙값 기준 상대 임계로 쓸 것.
    # 2026-09-09 진행 중이던 10.4k 굽기는 이 플래그 없이 돌았으므로 conf/ 가 없다.
    ap.add_argument("--save_conf", action="store_true", default=True)
    ap.add_argument("--no_save_conf", dest="save_conf", action="store_false")

    # D169 명사 추출. vLLM 서버는 밖에서 띄운다 (`vllm-start-on-demand-only`).
    ap.add_argument("--nouns_dir", default=None)                   # 기본 <out_root>/vlm_nouns
    ap.add_argument("--noun_frames", default=NOUN_FRAMES, type=int)
    ap.add_argument("--api_base", default=VLM_API_BASE)
    ap.add_argument("--vlm_model", default=VLM_MODEL)
    # metadata 의 명사 출처. vlm = 우리 프롬프트(기본), category = 배포본 category.json(옛 동작).
    ap.add_argument("--noun_source", default="vlm", choices=("vlm", "category"))

    ap.add_argument("--skip_done", action="store_true", default=True)
    ap.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    ap.add_argument("--dry_run", action="store_true", default=False)
    args = ap.parse_args()

    if args.da3_process_res <= 0:
        args.da3_process_res = args.width
    args.nouns_dir = args.nouns_dir or path.join(args.out_root, "vlm_nouns")
    makedirs(args.work_dir, exist_ok=True)

    shards = parse_shards(args.shards)
    scenes, missing = list_scenes(args.src_root, shards)
    mine = scenes[args.shard_id::args.num_shards]

    print()
    print(f"  stage              {args.stage}"
          f"{' -> ' + args.launch_stage if args.stage == 'launch' else ''}")
    print(f"  shards             {shards[0]}..{shards[-1]}  ({len(shards)}개)")
    print(f"  scene 총계          {len(scenes)}   video_input.mp4 없음 {missing}")
    print(f"  이 워커             {args.shard_id}/{args.num_shards}  →  {len(mine)}편")
    print(f"  out_root           {args.out_root}")
    print()

    rc = 0
    if args.stage == "link":
        stage_link(args)
    elif args.stage == "recon":
        stage_recon(args, mine)
    elif args.stage == "nouns":
        rc = stage_nouns(args, mine)
    elif args.stage == "metadata":
        stage_metadata(args, scenes)                                # csv 는 전체를 한 번에
    elif args.stage == "sam3":
        rc = stage_sam3(args, mine)
    elif args.stage == "dynmask":
        rc = stage_dynmask(args, mine)
    elif args.stage == "launch":
        rc = stage_launch(args, scenes)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
