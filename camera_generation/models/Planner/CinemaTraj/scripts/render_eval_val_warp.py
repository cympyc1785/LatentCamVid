"""latentcam eval 이 떨군 validation 전량을 **씬마다 1편**의 depth warp 비교영상으로 굽는다.

`render_pred_depth_warp.py` 는 `--video` 하나를 받는다. validation 은 씬이 수백 개라
셸에서 돌리면 씬마다 프로세스가 뜨고 임포트(~15 s)가 씬 수만큼 곱해진다. 여기서는
그 스크립트의 `main()` 을 **한 프로세스 안에서** 씬 루프로 돌리고, 씬당 산출물은
`all_entries.mp4` 하나만 `<scene>.mp4` 로 남긴다 (per-entry / reel_target 은 전부
그 안에 들어 있어 중복이다 — `--keep_entries` 로 예전처럼 다 남길 수 있다).

씬 목록은 `--entries_json` (`render_pred_depth_warp` 와 같은 코퍼스 규약의
`[{scene, idx}, ...]`) 으로 준다. 어떤 변이를 담을지(예: subject 가 움직이는 anchor 만)
고르는 일은 이 스크립트 바깥이다.

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=4 python scripts/render_eval_val_warp.py \
        --entries_json /data1/.../tmp/d201/val_moving_entries.json \
        --eval_dir d200=/data1/.../results/20260918_140904_dynpose_d200_da3 \
        --name_prefix dynpose --corpus_root /data1/.../latentcam_dynpose_d200 \
        --cloud_root .../out_dynpose --eval_data /data1/cympyc1785/data/DynPose-LBM \
        --cloud_source memory --out_dir results/20260919_d200_val_warp --shard 0/3
"""
import json
import sys
import traceback
from argparse import ArgumentParser
from os import listdir, makedirs, path, remove, rename, rmdir
from time import time

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from scripts import render_pred_depth_warp as rpdw                              # noqa: E402


def scene_entries(entries_json: str):
    """[{scene, idx}, ...] -> {scene: [idx ...]} (입력 순서 유지)."""
    out = {}
    with open(entries_json, encoding="utf-8") as file:
        for row in json.load(file):
            out.setdefault(row["scene"], []).append(str(row["idx"]))
    return out


def compact(scene_dir: str, scene: str, out_dir: str):
    """씬 폴더에서 릴 하나만 `<scene>.mp4` 로 끌어올리고 나머지는 지운다.

    WHY fallback: `render_pred_depth_warp` 는 entry 가 1개면 all_entries.mp4 를 쓰지
    않는다 (per-entry 파일과 내용이 같아서). val 99씬 중 11씬이 여기 걸려 조용히
    빠졌다 — 예외가 아니라 `done` 카운트만 줄어서 로그로는 안 보인다.
    """
    src = path.join(scene_dir, "all_entries.mp4")
    if not path.exists(src):
        single = [n for n in listdir(scene_dir) if n.endswith("__warp.mp4")]
        if len(single) != 1:
            return False
        src = path.join(scene_dir, single[0])
    rename(src, path.join(out_dir, f"{scene}.mp4"))
    for name in listdir(scene_dir):
        remove(path.join(scene_dir, name))
    rmdir(scene_dir)                                  # 빈 껍데기 디렉토리를 남기지 않는다
    return True


def main(args, passthrough):
    todo = scene_entries(args.entries_json)
    scenes = sorted(todo)
    shard, n_shard = (int(v) for v in args.shard.split("/"))
    scenes = [s for i, s in enumerate(scenes) if i % n_shard == shard]
    makedirs(args.out_dir, exist_ok=True)

    inner = rpdw.build_parser() if hasattr(rpdw, "build_parser") else None
    assert inner is not None, "render_pred_depth_warp.build_parser 가 필요하다"

    done, failed, t0 = 0, [], time()
    for i, scene in enumerate(scenes):
        final = path.join(args.out_dir, f"{scene}.mp4")
        if path.exists(final) and not args.overwrite:
            done += 1
            continue
        scene_dir = path.join(args.out_dir, f"_{scene}")
        argv = ["--video", scene, "--out_dir", scene_dir,
                "--entries", *todo[scene], *passthrough]
        try:
            main_args = inner.parse_args(argv)
            rpdw.main(main_args)
            ok = compact(scene_dir, scene, args.out_dir) if not args.keep_entries else True
            done += ok
        except Exception:                                        # 씬 하나가 죽어도 계속
            failed.append(scene)
            traceback.print_exc()
        elapsed = time() - t0
        print(f"[{shard}/{n_shard}] {i + 1}/{len(scenes)} {scene}  "
              f"{elapsed / 60:.1f} min  fail {len(failed)}", flush=True)

    print(f"\nshard {shard}/{n_shard}  scenes {len(scenes)}  done {done}  failed {len(failed)}")
    if failed:
        print("failed:", " ".join(failed))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--entries_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--shard", default="0/1")            # "i/n" — 씬을 n 갈래로
    parser.add_argument("--overwrite", action="store_true", default=False)
    # 씬 폴더를 통째로 남긴다 (per-entry warp + reel_target). 기본은 all_entries 만.
    parser.add_argument("--keep_entries", action="store_true", default=False)
    known, rest = parser.parse_known_args()
    main(known, rest)
