"""원본 LBM 실행의 **Blender 렌더 프레임**을 shot 별 mp4 + 라벨 릴 하나로 잇는다.

`lbm_preview_reel.py` 와 다른 것: 저건 Cinematographer 가 후보 판정에 쓴 **정지 프리뷰**를
이어 붙인다. 이건 VideoEngineer 가 실제로 렌더한 궤적 프레임 시퀀스 —
`<run>/renders/<shot>/<cam>/frames/frame_%04d.png` — 를 재생 가능한 영상으로 만든다.

## clip.mp4 를 왜 안 쓰나

`<run>/clips/<shot>/<cam>/clip.mp4` 가 이미 있는데 **1프레임짜리**다 (FIX: TRUMANS `.blend` 의
`frame_step=2` 가 그대로 새어 들어가 홀수 프레임만 렌더되고, `video_stage.encode_frames()` 의
`-i frame_%04d.png` 가 첫 구멍에서 멈춘다). `blender_render_worker.py --frame-step` 로 고친
뒤 실행분(`*_x15`)만 연속이고 그 전 실행분은 stride 2 다. 그래서 여기서는 **디스크에 있는 PNG
를 정렬해서 그대로 쓴다** — 구멍이 있든 없든 그 프레임들이 실제로 렌더된 전부다.

stride 2 실행분은 `--fps 12.5` 가 의도된 속도다 (렌더 fps 25, 프레임 절반).

라벨은 `trajectory_plan.json` 에서 온다: preset · travel_distance · visibility guard 적용 여부.
guard 가 걸린 shot 은 궤적이 0.75 배로 깎였다는 뜻이라 "왜 안 움직이나"의 답이 대개 여기 있다.

영상은 `imageio` + `libx264` (cv2 mp4v 는 VS Code 뷰어에서 안 열린다).

사용 예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY scripts/lbm_render_reel.py \
        --run_dir <LBM>/VideoEngineer/output/trumans_00add26c_x15 \
        --out_dir /tmp/lbm_reel --fps 25
"""
import json
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def shot_dirs(run_dir: str):
    """(shot_id, cam_id, frames_dir, plan) 들. 프레임이 0장인 카메라는 뺀다."""
    out = []
    for frames in sorted(glob(path.join(run_dir, "renders", "*", "*", "frames"))):
        cam_dir = path.dirname(frames)
        files = sorted(glob(path.join(frames, "frame_*.png")))
        if not files:
            continue
        plan_path = path.join(cam_dir, "trajectory_plan.json")
        plan = {}
        if path.exists(plan_path):
            with open(plan_path, encoding="utf-8") as file:
                plan = json.load(file)
        out.append((path.basename(path.dirname(cam_dir)), path.basename(cam_dir), files, plan))
    return out


def plan_label(plan: dict):
    """preset / travel / guard 를 한 줄로. 없는 값은 '?' 로 두고 죽지 않는다."""
    safety = plan.get("safety_report") or {}
    policy = ((plan.get("timing_policy") or {}).get("motion_speed_policy")) or {}
    preset = plan.get("preset_name") or "?"
    travel = safety.get("travel_distance")
    guard = policy.get("visibility_guard_applied")
    parts = [f"preset {preset}"]
    if travel is not None:
        parts.append(f"travel {float(travel):.3f}u")
    if policy.get("limited_travel_distance") is not None and guard:
        parts.append(f"guard x{policy.get('visibility_guard_motion_scale', '?')}"
                     f" -> {float(policy['limited_travel_distance']):.3f}u")
    vis = safety.get("start_visible_fraction")
    if vis is not None:
        parts.append(f"subj_vis {float(vis):.2f}")
    return "   ".join(parts)


def banner(width: int, height: int, lines, font_big, font_small):
    """상단 라벨 띠. 첫 줄이 크고 나머지는 작다."""
    strip = Image.new("RGB", (width, height), (16, 16, 20))
    draw = ImageDraw.Draw(strip)
    draw.text((10, 4), lines[0], fill=(255, 235, 120), font=font_big)
    for index, line in enumerate(lines[1:]):
        draw.text((10, 6 + (index + 1) * (font_big.size + 4) if hasattr(font_big, "size")
                   else 26 + index * 16), line, fill=(210, 210, 210), font=font_small)
    return np.asarray(strip)


def main(args):
    run_id = path.basename(path.normpath(args.run_dir))
    shots = shot_dirs(args.run_dir)
    if not shots:
        raise SystemExit(f"{args.run_dir}: 렌더 프레임이 없다")
    makedirs(args.out_dir, exist_ok=True)

    font_big, font_small = load_font(22), load_font(16)
    band = 56
    reel, table = [], []
    for shot_id, cam_id, files, plan in shots:
        frames = []
        for file in files:
            image = Image.open(file).convert("RGB")
            if args.width:
                image = image.resize((args.width, round(image.height * args.width / image.width)),
                                     Image.LANCZOS)
            frames.append(np.asarray(image))
        stack = np.stack(frames)
        height, width = stack.shape[1:3]

        # 프레임 번호 간격 — stride 2 면 이 실행분은 frame_step 수정 전이다.
        nums = [int(path.basename(f)[6:10]) for f in files]
        stride = int(np.median(np.diff(nums))) if len(nums) > 1 else 1

        out = path.join(args.out_dir, f"{run_id}__{shot_id}.mp4")
        imageio.mimwrite(out, stack, fps=args.fps, codec="libx264", quality=6,
                         macro_block_size=1)

        head = banner(width, band, [f"{run_id}  |  {shot_id}", plan_label(plan)],
                      font_big, font_small)
        labelled = np.concatenate(
            [np.repeat(head[None], len(stack), axis=0), stack], axis=1)
        reel.append(labelled)
        table.append((shot_id, plan.get("preset_name") or "?", len(files), stride,
                      (plan.get("safety_report") or {}).get("travel_distance"), out))

    reel_path = path.join(args.out_dir, f"{run_id}__reel.mp4")
    imageio.mimwrite(reel_path, np.concatenate(reel), fps=args.fps, codec="libx264",
                     quality=6, macro_block_size=1)

    print(f"run           {args.run_dir}")
    print(f"{'shot':22s} {'preset':20s} {'frames':>7s} {'stride':>7s} {'travel':>8s}")
    print("-" * 70)
    for shot_id, preset, n, stride, travel, _ in table:
        tstr = "?" if travel is None else f"{float(travel):.3f}"
        print(f"{shot_id:22s} {preset:20s} {n:7d} {stride:7d} {tstr:>8s}")
    print(f"\nreel          {len(table)} shots -> {reel_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="<LBM>/VideoEngineer/output/<run_id>")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps", default=25.0, type=float,
                        help="stride 2 실행분(frame_step 수정 전)은 12.5 가 의도된 속도")
    parser.add_argument("--width", default=0, type=int, help="0 = 원본 크기")
    main(parser.parse_args())
