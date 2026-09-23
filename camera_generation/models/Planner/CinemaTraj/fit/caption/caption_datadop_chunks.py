"""DataDoP 원본 shot 을 **49프레임 chunk 로 잘라 태깅만** 하고, 우리 plan-text 형식으로 낸다.

WHY: 우리 뱅크 캡션(`lbm_bank_captions_v1`)은 `target: <물체> motion: <카메라 움직임>` 두 절이다.
     DataDoP 22,314 shot 은 카메라 궤적만 있고 물체 anchor 가 없으므로 `target` 은 **`none`**
     으로 고정하고 `motion` 만 채운다. 그 `motion` 은 새로 짜지 않고 GenDoP 분절기가 내는 어휘를
     그대로 쓴다 — `CAM_INDEX_TO_PATTERN` 27종(`move forward`, `move left and up`, ...) ×
     `ANG_INDEX_TO_PATTERN` 7종(`yaw left`, `pitch up`, ...). 그래서 LLM/VLM 을 한 번도 안 부른다
     (사용자 지시: "태깅까지만").

세 가지 규약 (전부 코드 근거):
  * **좌표계**: `_transforms_cleaning.json` 의 `transform_matrix` 는 **이미 DataDoP(OpenGL) c2w**
    다. `caption_cameras_datadop.tag_trajectory` 는 기본이 OpenCV 입력이라 `[:3,1:3]*=-1` 을
    거는데, 여기서는 `convert=False` 로 그 flip 을 끈다. 한 번 더 걸면 up/forward 부호가 뒤집혀
    라벨이 **조용히 반대로** 나온다.
  * **리샘플 안 함 (`--num_poses 0`)**. shot 은 언제나 120 pose 고, 49프레임 chunk 는 그 중
    연속 49개다. 이 49개를 다시 120 으로 늘리면 프레임당 이동량이 (48/119) 배로 줄어
    `cam_static_threshold=0.02` 가 훨씬 많은 구간을 static 으로 찍는다 — 어휘 분포가 통째로
    바뀐다. 원본 pose 를 그대로 쓰면 속도 눈금이 DataDoP 와 **동일**하다.
  * **combine 단계 window 는 못 바꾼다**. `segment_rigidbody_trajectories` 는 인자와 무관하게
    combine 단계에서 `smoothing_window_size=15 / min_chunk_size=10` 으로 덮어쓰고 chunk 수가
    4 이하가 될 때까지 5씩 키운다 (`segmentation.py:392-403`). 48 velocity 샘플에 window 15 라
    chunk 는 최소 10프레임짜리 1~4개가 된다. GenDoP 리포는 **0줄 수정**한다.

chunk 자르기는 `latentcam/main/dataset_datadop.py:_segments` 와 같은 슬라이딩 윈도우다. 기본
`--chunks_per_shot 3` 은 `linspace(0, 120-49, 3)` = 시작 0/36/71 이라 120 pose 를 **빠짐없이**
덮는다 (stride 36 짜리 `range` 는 71 을 못 만들어 뒤 22프레임이 버려진다).

사용 예시:
    # 50 shot 만 (기본값 확인용)
    python fit/caption/caption_datadop_chunks.py --limit 50

    # 전량 8샤드
    for i in 0 1 2 3 4 5 6 7; do
      python fit/caption/caption_datadop_chunks.py --num_shards 8 --shard_id $i &
    done; wait
    python fit/caption/caption_datadop_chunks.py --num_shards 8 --merge
"""
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path
import json
import sys
import time

import numpy as np

HERE = path.dirname(path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from caption_cameras_datadop import SEG_DEFAULTS, tag_trajectory       # noqa: E402

CINEMATRAJ_ROOT = path.dirname(path.dirname(HERE))
DATADOP_ROOT_DEFAULT = "/data1/cympyc1785/data/DataDoP/DataDoP_with_scene"
# `dataset_datadop.py` 가 보는 화이트리스트. 있으면 교집합만 쓴다 (없으면 전량).
VALID_TXT_DEFAULT = "/data1/cympyc1785/data/DataDoP/DataDoP_valid.txt"
N_POSES = 120           # DataDoP 는 shot 길이와 무관하게 120 pose 로 정규화돼 있다
SUFFIX = "_transforms_cleaning.json"


def motion_of(chunks: list, join: str = ", then "):
    """분절 chunk 목록 -> `motion` 한 줄. 어휘는 GenDoP 표 그대로 손대지 않는다.

    `move` 와 `angular` 중 한쪽이 `static` 이면 그쪽 절을 빼고, 둘 다면 `static` 이다
    ("static and static" 같은 말을 만들지 않는다). 같은 라벨이 연속이면 합친다 —
    `remove_short_chunks` 가 chunk 를 합칠 때 같은 index 가 두 번 나올 수 있다.
    """
    parts = []
    for chunk in chunks:
        move, angular = chunk["move"], chunk["angular"]
        if move == "static" and angular == "static":
            text = "static"
        elif angular == "static":
            text = move
        elif move == "static":
            text = angular
        else:
            text = f"{move} and {angular}"
        if not parts or parts[-1] != text:
            parts.append(text)
    return join.join(parts) if parts else "static"


def chunk_starts(num_frames: int, chunks_per_shot: int, stride: int):
    """shot 안에서 chunk 시작 인덱스. `chunks_per_shot>0` 이면 균등, 아니면 stride 슬라이딩."""
    last = N_POSES - num_frames
    if last < 0:
        return []
    if chunks_per_shot > 0:
        if chunks_per_shot == 1:
            return [0]
        return sorted(set(int(s) for s in np.rint(
            np.linspace(0, last, chunks_per_shot)).astype(int)))
    return list(range(0, last + 1, max(1, stride)))


def collect_shots(root: str, valid_txt: str, max_scenes: int):
    """'<scene>/<shot>' 목록. 순서는 정렬 고정 (샤드가 재현 가능해야 한다)."""
    valid = None
    if valid_txt and path.isfile(valid_txt):
        with open(valid_txt, encoding="utf-8") as file:
            valid = {line.strip() for line in file if line.strip()}
    scenes = sorted(path.basename(d) for d in glob(path.join(root, "*"))
                    if path.isdir(d) and not path.basename(d).startswith("."))
    if max_scenes:
        scenes = scenes[:int(max_scenes)]
    shots = []
    for scene in scenes:
        for file_path in sorted(glob(path.join(root, scene, f"*{SUFFIX}"))):
            shot = path.basename(file_path)[:-len(SUFFIX)]
            rel = f"{scene}/{shot}"
            if valid is None or rel in valid:
                shots.append(rel)
    return shots, (None if valid is None else len(valid))


def load_shot_poses(root: str, rel: str):
    """OpenGL c2w (120,4,4). frame 순서는 `monst3r_im_id` 오름차순으로 못 박는다."""
    with open(path.join(root, f"{rel}{SUFFIX}"), encoding="utf-8") as file:
        data = json.load(file)
    frames = sorted(data["frames"], key=lambda f: int(f["monst3r_im_id"]))
    poses = np.asarray([f["transform_matrix"] for f in frames], dtype=np.float64)
    return poses, data


def main(args):
    root = args.datadop_root
    out_dir = args.out or path.join(CINEMATRAJ_ROOT, "out", "datadop_chunk_captions")
    makedirs(out_dir, exist_ok=True)
    tag = f"shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else "all"

    if args.merge:                      # 샤드 조각들을 하나로 합치기만 한다
        captions, files = {}, sorted(glob(path.join(out_dir, "captions.shard*.json")))
        for file_path in files:
            with open(file_path, encoding="utf-8") as file:
                captions.update(json.load(file)["captions"])
        payload = {"format": "lbm_bank_captions_v1", "video": "datadop",
                   "bank_dir": "datadop_chunk_captions", "prompt_fields": ["target", "motion"],
                   "source": "DataDoP 49-frame chunks (tagging only, no LLM)",
                   "merged_from": [path.basename(f) for f in files], "captions": captions}
        with open(path.join(out_dir, "captions.json"), "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=1)
        print(f"{'merged':<20}{len(files)} 샤드   {len(captions)} entry")
        print(f"{'wrote':<20}{path.join(out_dir, 'captions.json')}")
        return 0

    seg_kwargs = dict(cam_static_threshold=args.static_threshold,
                      cam_diff_threshold=args.diff_threshold,
                      angular_static_threshold=args.angular_static_threshold,
                      fps=args.fps, smoothing_window_size=args.smoothing_window_size,
                      min_chunk_size=args.min_chunk_size)
    shots, n_valid = collect_shots(root, args.valid_txt, args.max_scenes)
    if args.limit:
        shots = shots[:int(args.limit)]
    shots = [s for i, s in enumerate(shots) if i % args.num_shards == args.shard_id]
    starts = chunk_starts(args.num_frames, args.chunks_per_shot, args.stride)

    captions, bad, started = {}, [], time.time()
    move_hist, ang_hist, nchunk_hist = {}, {}, {}
    for index, rel in enumerate(shots):
        try:
            poses, meta = load_shot_poses(root, rel)
        except Exception as error:                              # noqa: BLE001
            bad.append(f"{rel}: {type(error).__name__} {error}")
            continue
        if poses.shape[0] != N_POSES:
            bad.append(f"{rel}: pose {poses.shape[0]} != {N_POSES}")
            continue
        scene, shot = rel.split("/")
        for start in starts:
            window = poses[start:start + args.num_frames]
            # convert=False: 이미 OpenGL c2w 다 (docstring 규약 ①).
            info = tag_trajectory(window, args.num_poses, seg_kwargs, False, convert=False)
            motion = motion_of(info["chunks"], args.join)
            caption = {"target": "none", "motion": motion,
                       "prompt": f"target: none motion: {motion}"}
            caption.update(scene=scene, shot=shot,
                           frames=[int(start), int(start + args.num_frames)],
                           chunks=info["chunks"], outline=info["outline"].strip(),
                           step_median=info["step_median"],
                           total_translation=info["total_translation"])
            captions[f"{rel}__f{start:03d}"] = caption
            for chunk in info["chunks"]:
                move_hist[chunk["move"]] = move_hist.get(chunk["move"], 0) + 1
                ang_hist[chunk["angular"]] = ang_hist.get(chunk["angular"], 0) + 1
            n = len(info["chunks"])
            nchunk_hist[n] = nchunk_hist.get(n, 0) + 1
        if args.progress and (index + 1) % args.progress == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  [{tag}] {index + 1}/{len(shots)} shot  {rate:.1f} shot/s  "
                  f"남은 {int((len(shots) - index - 1) / max(rate, 1e-9))}s", flush=True)

    payload = {"format": "lbm_bank_captions_v1", "video": "datadop",
               "bank_dir": "datadop_chunk_captions", "prompt_fields": ["target", "motion"],
               "source": "DataDoP 49-frame chunks (tagging only, no LLM)",
               "datadop_root": root, "num_frames": args.num_frames,
               "chunk_starts": [int(s) for s in starts], "num_poses": args.num_poses,
               "seg_kwargs": seg_kwargs, "join": args.join,
               "convention": "DataDoP OpenGL c2w 그대로 (flip 없음), 리샘플 없음, 스케일 D=1",
               "shard": [args.shard_id, args.num_shards], "captions": captions}
    out_path = path.join(out_dir, f"captions.{tag}.json" if args.num_shards > 1
                         else "captions.json")
    if not args.dry_run:
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=1)

    print()
    print(f"{'datadop_root':<20}{root}")
    print(f"{'valid.txt':<20}{args.valid_txt if n_valid else '없음 (전량)'}"
          f"{'' if n_valid is None else f'  ({n_valid} 항목)'}")
    print(f"{'shots (this shard)':<20}{len(shots)}")
    print(f"{'chunk starts':<20}{starts}  (num_frames {args.num_frames})")
    print(f"{'entries':<20}{len(captions)}")
    print(f"{'seg_kwargs':<20}{seg_kwargs}")
    print(f"{'num_poses':<20}{args.num_poses if args.num_poses > 0 else 'native (리샘플 없음)'}")
    print(f"{'elapsed':<20}{time.time() - started:.1f}s")
    print(f"{'wrote':<20}{'(dry run — 안 씀)' if args.dry_run else out_path}")
    print()
    print(f"{'chunk/entry':<20}{dict(sorted(nchunk_hist.items()))}")
    total_move = max(sum(move_hist.values()), 1)
    print(f"{'move 상위':<20}")
    for label, count in sorted(move_hist.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {label:<32}{count:>8}  {100 * count / total_move:5.1f}%")
    total_ang = max(sum(ang_hist.values()), 1)
    print(f"{'angular':<20}")
    for label, count in sorted(ang_hist.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<32}{count:>8}  {100 * count / total_ang:5.1f}%")
    if captions:
        sample = next(iter(captions.items()))
        print(f"\n예시  {sample[0]}\n      {sample[1]['prompt']}")
    if bad:
        print(f"\n⚠ 건너뛴 shot {len(bad)}: {bad[:5]}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--datadop_root", default=DATADOP_ROOT_DEFAULT, type=str)
    parser.add_argument("--valid_txt", default=VALID_TXT_DEFAULT, type=str)   # "" 면 전량
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--num_frames", default=49, type=int)      # chunk 길이
    # chunk 시작점: >0 이면 shot 당 균등 N개, 0 이면 --stride 슬라이딩 윈도우.
    parser.add_argument("--chunks_per_shot", default=3, type=int)
    parser.add_argument("--stride", default=36, type=int)
    parser.add_argument("--num_poses", default=0, type=int)        # 0 = 리샘플 안 함 (규약 ②)
    # 분절기 노브 — 기본값은 DataDoP 원본. combine 단계는 GenDoP 이 덮어쓴다 (규약 ③).
    parser.add_argument("--fps", default=SEG_DEFAULTS["fps"], type=float)
    parser.add_argument("--static_threshold", default=SEG_DEFAULTS["cam_static_threshold"],
                        type=float)
    parser.add_argument("--diff_threshold", default=SEG_DEFAULTS["cam_diff_threshold"], type=float)
    parser.add_argument("--angular_static_threshold",
                        default=SEG_DEFAULTS["angular_static_threshold"], type=float)
    parser.add_argument("--smoothing_window_size", default=SEG_DEFAULTS["smoothing_window_size"],
                        type=int)
    parser.add_argument("--min_chunk_size", default=SEG_DEFAULTS["min_chunk_size"], type=int)
    parser.add_argument("--join", default=", then ", type=str)     # chunk 사이 접속어
    parser.add_argument("--max_scenes", default=0, type=int)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--num_shards", default=1, type=int)
    parser.add_argument("--shard_id", default=0, type=int)
    parser.add_argument("--progress", default=200, type=int)       # 0 이면 조용히
    parser.add_argument("--merge", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    raise SystemExit(main(parser.parse_args()))
