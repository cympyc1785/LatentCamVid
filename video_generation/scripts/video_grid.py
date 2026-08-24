"""mp4 들을 라벨 붙여 R×C 격자로 붙인다.

`tools/recammaster/halfsplit_compare_grid.py` 는 `<src>_<model>_<cam_src>.mp4` 라는 **고정된
파일명 규칙**에 묶여 있어서 임의 경로 조합엔 못 쓴다. 이건 경로를 그대로 받는다.

행 라벨(왼쪽 세로 띠)과 열 라벨(위 가로 띠)을 따로 준다 — "위=depth warp / 아래=생성" 처럼
행이 렌더 종류이고 열이 카메라인 배치가 기본형이다. **출력만 보면 hole 이 warp 탓인지 생성
탓인지 못 가르므로 depth warp 를 같은 그림에 넣는 게 규칙이다.**

프레임 수가 다르면 최소값으로 자르고, 해상도가 다르면 첫 타일 크기로 맞춘다.
영상은 `imageio` + `libx264` (cv2 mp4v 는 VS Code 뷰어에서 안 열린다).

사용 예시:
    python scripts/video_grid.py --out /tmp/g.mp4 --cols 4 \
        --row_labels "depth warp" "Vista4D" \
        --col_labels A B C D \
        --videos r0c0.mp4 r0c1.mp4 r0c2.mp4 r0c3.mp4 \
                  r1c0.mp4 r1c1.mp4 r1c2.mp4 r1c3.mp4
"""
from argparse import ArgumentParser
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


def read_frames(file: str, size=None):
    reader = imageio.get_reader(file)
    frames = []
    for frame in reader:
        image = Image.fromarray(np.asarray(frame)).convert("RGB")
        if size and image.size != size:
            image = image.resize(size, Image.LANCZOS)
        frames.append(np.asarray(image))
    reader.close()
    if not frames:
        raise SystemExit(f"{file}: 프레임 0장")
    return np.stack(frames)


def text_strip(width: int, height: int, label: str, font, vertical=False):
    """라벨 띠 하나. `vertical` 이면 90도 돌려 왼쪽 세로 띠로 쓴다.

    가로로 그린 뒤 회전하므로 `width`/`height` 는 **회전 전** 기준이다 — 세로 띠를 원하면
    `width=타일 높이`, `height=띠 두께` 를 주면 회전 후 (타일 높이, 띠 두께) 가 된다.
    """
    box = (width, height)
    strip = Image.new("RGB", box, (16, 16, 20))
    draw = ImageDraw.Draw(strip)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    draw.text(((box[0] - (right - left)) / 2 - left, (box[1] - (bottom - top)) / 2 - top),
              label, fill=(255, 235, 120), font=font)
    if vertical:
        strip = strip.rotate(90, expand=True)
    return np.asarray(strip)


def main(args):
    cols = args.cols
    if len(args.videos) % cols:
        raise SystemExit(f"영상 {len(args.videos)}개가 열 {cols}로 안 나눠떨어진다")
    rows = len(args.videos) // cols

    clips = [read_frames(args.videos[0])]
    size = (clips[0].shape[2], clips[0].shape[1])
    if args.width:
        size = (args.width, round(size[1] * args.width / size[0]))
        clips = [read_frames(args.videos[0], size)]
    clips += [read_frames(file, size) for file in args.videos[1:]]
    n = min(len(clip) for clip in clips)
    tile_w, tile_h = size

    font_col, font_row = load_font(max(12, tile_w // 26)), load_font(max(12, tile_h // 14))
    band = font_col.size + 12
    gutter = font_row.size + 12 if args.row_labels else 0

    col_head = None
    if args.col_labels:
        if len(args.col_labels) != cols:
            raise SystemExit(f"col_labels {len(args.col_labels)}개 != 열 {cols}")
        head = [text_strip(tile_w, band, label, font_col) for label in args.col_labels]
        col_head = np.concatenate(head, axis=1)
        if gutter:
            col_head = np.concatenate(
                [np.zeros((band, gutter, 3), np.uint8), col_head], axis=1)

    out_frames = []
    for index in range(n):
        row_images = []
        for row in range(rows):
            tiles = [clips[row * cols + col][index] for col in range(cols)]
            band_row = np.concatenate(tiles, axis=1)
            if gutter:
                label = args.row_labels[row] if row < len(args.row_labels) else ""
                band_row = np.concatenate(
                    [text_strip(tile_h, gutter, label, font_row, vertical=True), band_row], axis=1)
            row_images.append(band_row)
        frame = np.concatenate(row_images, axis=0)
        if col_head is not None:
            frame = np.concatenate([col_head, frame], axis=0)
        out_frames.append(frame)

    # libx264 + yuv420p 는 가로/세로가 짝수여야 한다. 라벨 띠 두께가 폰트 크기에서 나오므로
    # 홀수가 흔하고, 그러면 ffmpeg 이 broken pipe 로 죽는다 — 오른쪽/아래를 1px 채운다.
    stack = np.stack(out_frames)
    pad_h, pad_w = stack.shape[1] % 2, stack.shape[2] % 2
    if pad_h or pad_w:
        stack = np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

    makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, stack, fps=args.fps, codec="libx264",
                     quality=6, macro_block_size=1)
    print(f"grid          {rows}x{cols}  frames {n}  tile {tile_w}x{tile_h}")
    for i, file in enumerate(args.videos):
        print(f"  r{i // cols}c{i % cols}  {len(clips[i]):3d}f  {file}")
    print(f"out           {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", required=True, help="행 우선(row-major) 순서")
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--row_labels", nargs="*", default=[], help="왼쪽 세로 띠")
    parser.add_argument("--col_labels", nargs="*", default=[], help="위쪽 가로 띠")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=0, help="타일 가로 (0 = 원본)")
    main(parser.parse_args())
