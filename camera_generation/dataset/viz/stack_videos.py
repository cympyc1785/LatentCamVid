"""mp4 여러 개를 한 화면에 위아래(또는 좌우)로 붙인다 — before/after 를 같은 재생축에서 본다.

왜 필요한가: 같은 필터를 두 뱅크에 걸어 `render_bank_videos.py` 를 두 번 돌리면 mp4 가 두 개
나오는데, 따로 보면 "구멍이 줄었나 궤적이 짧아졌나"를 눈으로 못 가른다 — 프레임 f 를 나란히
놓아야 보인다. 이 호스트에는 **ffmpeg 이 없어서** `ffmpeg vstack` 을 못 쓴다. 그래서 python 에서
프레임을 읽어 붙인다.

프레임 수가 다르면 짧은 쪽에 맞춰 자르고, 폭이 다르면 가장 넓은 폭에 맞춰 좌우를 배경색으로
채운다 (리사이즈하면 두 줄의 배율이 달라져서 비교가 깨진다).

`--hold_short` 를 주면 자르는 대신 **짧은 타일의 마지막 프레임을 정지시켜** 가장 긴 타일까지
끌고 간다. before/after 두 줄은 프레임 대응이 생명이라 기본값은 예전대로 자르기지만, 길이가
제각각인 렌더를 격자로 볼 때는 자르기가 치명적이다 (실측: 19프레임짜리 하나가 6칸 격자 전체를
0.8초로 잘라낸다).

`--inputs` 는 mp4 뿐 아니라 **PNG 프레임 디렉토리**도 받는다 (`concat_videos.py` 와 같은 사정).
Blender GT 렌더(`trumans_gt_render.py`)는 mp4 를 안 만들고 `<preset>/rgb/frame_*.png` 만 떨구므로,
preset 격자를 만들려면 디렉토리를 직접 읽어야 한다.

출력: `--output` 경로 mp4 하나 (imageio + libx264; cv2 의 mp4v 는 VS Code 뷰어에서 안 열린다).

env: 아무 env (torch/GPU 불필요, imageio 만 있으면 된다)

예시:
    python viz/stack_videos.py --inputs out/camel/hole_bank_holeonly/g1_row.mp4 \
        out/camel/hole_bank/g1_row.mp4 --output out/camel/hole_bank/g1_before_after.mp4
"""
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)


def read_frame_dir(dir_path: str):
    """PNG 프레임 디렉토리를 읽는다. 정렬은 **파일명 안의 숫자**로 — LBM 렌더는 stride 2 라
    `frame_0001, 0003, ...` 처럼 띄엄띄엄이고, 문자열 정렬이면 0100 이 0099 앞으로 온다."""
    import re
    from glob import glob

    import imageio.v2 as imageio
    paths = glob(path.join(dir_path, "*.png")) + glob(path.join(dir_path, "*.jpg"))
    assert paths, f"프레임이 없다: {dir_path}"

    def index_of(p):
        digits = re.findall(r"\d+", path.basename(p))
        return int(digits[-1]) if digits else 0

    return [np.asarray(imageio.imread(p))[:, :, :3] for p in sorted(paths, key=index_of)]


def read_video(input_path: str):
    """mp4 든 PNG 디렉토리든 프레임 리스트로 돌려준다 (§docstring 의 `--inputs`)."""
    if path.isdir(input_path):
        return read_frame_dir(input_path)
    import imageio.v2 as imageio
    with imageio.get_reader(input_path) as reader:
        return [np.asarray(frame)[:, :, :3] for frame in reader]


def pad_to(frame: np.ndarray, height: int, width: int, background):
    """리사이즈가 아니라 **패딩**이다 — 두 줄의 배율이 달라지면 비교가 안 된다."""
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    top, left = (height - frame.shape[0]) // 2, (width - frame.shape[1]) // 2
    canvas[top:top + frame.shape[0], left:left + frame.shape[1]] = frame
    return canvas


def resize_to_width(frame: np.ndarray, width: int):
    """가로폭을 맞춘다. **`--columns` 를 쓸 때만** 부른다 — 기본 경로는 패딩 그대로다.

    타일이 8개쯤 되면 원본 1280px 를 그대로 붙일 수 없다(가로 5120px). 이때는 배율이 달라지는
    것보다 화면 밖으로 나가는 게 더 나쁘다. 대신 **모든 타일에 같은 폭**을 강제해서 타일 사이
    배율 차이는 만들지 않는다 (원본 해상도가 같은 코퍼스라 실제로는 배율이 하나다).
    """
    import cv2
    if frame.shape[1] == width:
        return frame
    height = max(1, int(round(frame.shape[0] * width / frame.shape[1])))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_label(frame: np.ndarray, text: str):
    """좌상단에 이름표. 어느 타일이 어느 영상인지 못 세면 8칸 격자는 그냥 벽지다."""
    import cv2
    scale = max(0.5, frame.shape[1] / 900.0)
    thickness = max(1, int(round(scale * 1.6)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (tw + 14, th + 16), (0, 0, 0), -1)
    cv2.putText(frame, text, (7, th + 7), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def main(args):
    clips = [read_video(p) for p in args.inputs]
    length = max(len(c) for c in clips) if args.hold_short else min(len(c) for c in clips)
    axis = 0 if args.direction == "vertical" else 1

    labels = args.labels or [None] * len(clips)
    assert len(labels) == len(clips), f"labels {len(labels)} != inputs {len(clips)}"
    if args.tile_width:
        clips = [[resize_to_width(frame, args.tile_width) for frame in clip] for clip in clips]

    heights = [c[0].shape[0] for c in clips]
    widths = [c[0].shape[1] for c in clips]
    box = (max(heights), max(widths))

    def row(parts, width):
        """한 줄을 가로로 잇고 부족한 칸은 배경으로 채운다."""
        blocks = []
        for index, part in enumerate(parts):
            blocks.append(part)
            if index + 1 < width and args.gap > 0:
                blocks.append(np.full((box[0], args.gap, 3), args.background, dtype=np.uint8))
        while len(parts) < width:                     # 마지막 줄의 빈 칸
            blocks.append(np.full((box[0], box[1], 3), args.background, dtype=np.uint8))
            parts = parts + [None]
        return np.concatenate(blocks, axis=1)

    frames = []
    for f in range(length):
        tiles = []
        for clip, label in zip(clips, labels):
            # --hold_short 면 f 가 이 타일의 길이를 넘어설 수 있다 → 마지막 프레임에서 정지.
            tile = pad_to(clip[min(f, len(clip) - 1)], box[0], box[1], args.background)
            tiles.append(draw_label(tile, label) if label else tile)

        if args.columns:                              # 격자 — 줄로 자른 뒤 세로로 잇는다
            rows, gap_h = [], None
            for start in range(0, len(tiles), args.columns):
                rows.append(row(tiles[start:start + args.columns], args.columns))
            if args.gap > 0 and len(rows) > 1:
                gap_h = np.full((args.gap, rows[0].shape[1], 3), args.background, dtype=np.uint8)
                rows = [x for r in rows for x in (r, gap_h)][:-1]
            frames.append(np.concatenate(rows, axis=0))
            continue

        parts = []
        for index, tile in enumerate(tiles):
            parts.append(tile)
            if index + 1 < len(tiles) and args.gap > 0:
                bar = ((args.gap, box[1], 3) if axis == 0 else (box[0], args.gap, 3))
                parts.append(np.full(bar, args.background, dtype=np.uint8))
        frames.append(np.concatenate(parts, axis=axis))

    import imageio.v2 as imageio
    imageio.mimwrite(args.output, frames, fps=args.fps, codec="libx264",
                     quality=6, macro_block_size=1)

    print(f"{'inputs':<12}{len(clips)}   direction {args.direction}   gap {args.gap}")
    for input_path, clip in zip(args.inputs, clips):
        print(f"{'':<12}{len(clip):>5} frames  {clip[0].shape[1]}x{clip[0].shape[0]}  {input_path}")
    print(f"{'output':<12}{length} frames  {frames[0].shape[1]}x{frames[0].shape[0]}  fps {args.fps}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    # 위아래가 기본 — before/after 는 같은 타일이 세로로 겹쳐야 프레임 대응이 보인다.
    parser.add_argument("--direction", default="vertical", choices=("vertical", "horizontal"))
    parser.add_argument("--gap", default=6, type=int)          # 줄 사이 구분선 두께 (px)
    parser.add_argument("--background", default=20, type=int)  # 패딩/구분선 회색값
    parser.add_argument("--fps", default=12.0, type=float)
    # 격자. None 이면 예전 동작 그대로 한 줄(direction) 이다.
    parser.add_argument("--columns", default=None, type=int)
    parser.add_argument("--tile_width", default=None, type=int)   # None = 원본 해상도 유지
    parser.add_argument("--labels", nargs="*", default=None)      # 타일 좌상단 이름표
    # 짧은 타일을 자를지(기본, 예전 동작) 마지막 프레임으로 정지시켜 늘일지.
    parser.add_argument("--hold_short", dest="hold_short", action="store_true")
    parser.add_argument("--no_hold_short", dest="hold_short", action="store_false")
    parser.set_defaults(hold_short=False)
    main(parser.parse_args())
