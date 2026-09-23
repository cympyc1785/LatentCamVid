"""mp4 여러 개를 **시간축으로** 이어 붙인다 (`stack_videos.py` 는 공간축으로 붙인다).

왜 따로 필요한가: LBM 의 VideoEngineer 는 shot 마다 `clips/<shot>/<cam>/clip.mp4` 를 따로
떨군다. 한 recording 의 카메라들이 실제로 어떻게 움직이는지 보려면 그 clip 들을 **한 편으로**
이어야 하는데, Editor 가 만드는 `exports/final_edit_v1.mp4` 는 편집본이라 각 shot 을 잘라낸다
(실측: 00add26c 8 clip 합계 ~600 프레임 → final_edit 218 프레임). 카메라 움직임을 판정하려면
잘리지 않은 원본 clip 이 필요하다.

이 호스트에는 ffmpeg 이 없어서 `ffmpeg concat` 을 못 쓴다 (`stack_videos.py` 와 같은 사정).
그래서 프레임을 읽어 잇는다. 해상도가 다르면 가장 넓은 폭에 맞춰 패딩한다 — 리사이즈하면
clip 마다 배율이 달라져 이동량 비교가 깨진다.

`--inputs` 는 mp4 뿐 아니라 **PNG 프레임 디렉토리**도 받는다. LBM 이 떨구는 `clip.mp4` 는
실측 결과 프레임이 1장뿐이라(포스터와 다를 게 없다) 실제 렌더를 보려면 그 옆의
`renders/<shot>/<cam>/frames/frame_*.png` 를 읽어야 한다. 파일명 숫자 순으로 정렬한다.

`--labels` 를 주면 각 구간 좌상단에 이름표를 박는다 (몇 번째 shot 인지 못 세면 못 읽는다).
`--gap_frames` 는 구간 사이에 끼우는 검은 프레임 수다.

출력: `--output` mp4 하나 (imageio + libx264; cv2 의 mp4v 는 VS Code 뷰어에서 안 열린다).

env: 아무 env (torch/GPU 불필요, imageio 만 있으면 된다)

예시:
    python viz/concat_videos.py \
        --inputs <LBM>/VideoEngineer/output/<run>/clips/*/*/clip.mp4 \
        --labels shot_2 shot_3 shot_4 --output /tmp/run_render.mp4
"""
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

#    `read_frame_dir` 은 `stack_videos.py` 로 옮겼다 — 거기 `read_video` 도 디렉토리를 받게
#    되면서 두 곳에 같은 구현이 남으면 정렬 규칙이 갈라진다.
from viz.stack_videos import draw_label, pad_to, read_frame_dir, read_video  # noqa: E402


def main(args):
    clips = [read_frame_dir(p) if path.isdir(p) else read_video(p) for p in args.inputs]
    labels = args.labels or [None] * len(clips)
    assert len(labels) == len(clips), f"labels {len(labels)} != inputs {len(clips)}"

    height = max(c[0].shape[0] for c in clips)
    width = max(c[0].shape[1] for c in clips)

    frames = []
    for clip, label in zip(clips, labels):
        for frame in clip:
            tile = pad_to(frame, height, width, args.background)
            frames.append(draw_label(tile, label) if label else tile)
        for _ in range(args.gap_frames):
            frames.append(np.zeros((height, width, 3), dtype=np.uint8))

    import imageio.v2 as imageio
    imageio.mimwrite(args.output, frames, fps=args.fps, codec="libx264",
                     quality=6, macro_block_size=1)

    print(f"{'input':60s} {'frames':>7s}")
    print("-" * 68)
    for source, clip in zip(args.inputs, clips):
        print(f"{source[-60:]:60s} {len(clip):7d}")
    print(f"\n출력  {len(frames)} 프레임  {width}x{height}  fps {args.fps}")
    print(f"->    {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    parser.add_argument("--labels", nargs="*", default=None)      # 구간별 좌상단 이름표
    parser.add_argument("--gap_frames", default=0, type=int)      # 구간 사이 검은 프레임 수
    parser.add_argument("--background", default=20, type=int)     # 패딩 회색값
    parser.add_argument("--fps", default=25.0, type=float)
    main(parser.parse_args())
