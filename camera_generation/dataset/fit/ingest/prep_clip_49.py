"""임의 mp4 → Lite 규약(49 프레임) 클립 mp4. `recon_and_seg_single.py` 앞단에 쓴다.

왜 필요한가: `recon_and_seg_single.py` 는 항상 **center-slice** 한다
(`utils/media.py:233 slice_center_frames`). 30fps 소스를 그대로 먹이면 49 프레임 = 1.6 초라
피사체가 거의 안 움직이고, TRUMANS-Lite 뱅크가 쓰는 `frame_step 3` (10fps, 4.9 초) 과 구간이
어긋난다. 그래서 여기서 먼저 stride 로 솎아 49 프레임을 만들어 넘긴다.

`--stride 0` 이면 전 구간에 균등 배분한다 (프레임 수가 49·stride 에 못 미칠 때 안전).

env: `vista4d` (cv2 · imageio 만 쓴다)

예시:
    python fit/ingest/prep_clip_49.py --input "/data1/.../jogging woman.mp4" \
        --output /data1/.../jogging_woman_49.mp4 --stride 3 --out_fps 10
"""
from argparse import ArgumentParser
from os import makedirs, path

import cv2
import imageio
import numpy as np


def read_frames(video_path: str):
    capture = cv2.VideoCapture(video_path)
    assert capture.isOpened(), f"영상을 못 연다: {video_path}"
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(frames), fps


def pick_indices(total: int, num_frames: int, stride: int, start: int):
    """stride 로 뽑되, 모자라면 전 구간 균등으로 되돌린다."""
    if stride > 0 and start + (num_frames - 1) * stride < total:
        return np.arange(num_frames) * stride + start, f"stride {stride} @ start {start}"
    return (np.rint(np.linspace(0, total - 1, num_frames)).astype(int),
            f"균등 (stride {stride} 로는 {num_frames} 프레임을 못 채운다)")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--num_frames", default=49, type=int)
    parser.add_argument("--stride", default=3, type=int)      # 0 이면 전 구간 균등
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--out_fps", default=10.0, type=float)
    args = parser.parse_args()

    frames, src_fps = read_frames(args.input)
    indices, how = pick_indices(len(frames), args.num_frames, args.stride, args.start)
    clip = frames[indices]

    makedirs(path.dirname(path.abspath(args.output)), exist_ok=True)
    #    cv2 의 mp4v 는 VS Code 뷰어에서 안 열린다 — 집 규칙대로 imageio + libx264.
    imageio.mimwrite(args.output, list(clip), fps=args.out_fps, codec="libx264",
                     quality=6, macro_block_size=1)

    print(f"{'소스':12} {args.input}")
    print(f"{'소스 프레임':12} {len(frames)}  @ {src_fps:.3f} fps  {frames.shape[2]}x{frames.shape[1]}")
    print(f"{'샘플링':12} {how}")
    print(f"{'구간':12} frame {indices[0]}..{indices[-1]}  = {(indices[-1]-indices[0])/src_fps:.2f} s")
    print(f"{'출력':12} {args.output}  {len(clip)} 프레임 @ {args.out_fps:g} fps")


if __name__ == "__main__":
    main()
