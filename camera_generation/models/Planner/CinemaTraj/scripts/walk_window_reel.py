"""`mine_walk_actions` 가 캐낸(또는 놓친) 보행 창을 TRUMANS 배포 렌더 영상에서 잘라 타일 릴로 만든다.

왜 필요한가: 보행 채굴이 7편 중 5편에서 **0건**으로 나왔을 때 표만 보면 "그 편엔 보행이 없다"와
"임계가 틀렸다"를 구분할 수 없다. 실제로는 후자였다 — `mine_walk_actions` 가 `step * fps > 0.4 m/s`
로 재는데 그 `fps` 를 **blend 의 render fps**(편마다 15 또는 25)에서 받아온다. 그런데
`video_render/<seq>.pkl.mp4` 는 7편 전부 **프레임 수가 시퀀스 길이와 1:1 이고 fps 가 30** 이다.
즉 모션 배열의 실제 레이트는 30 이고 blend 의 15/25 는 렌더 설정일 뿐이다. 같은 걸음이 blend
fps 15 편에서는 0.24 m/s 로 찍혀 임계 아래로 떨어졌다.

그래서 이 스크립트는 "지금 버려지는 49프레임 창이 정말 보행인가"를 **눈으로** 확인시킨다.
채굴은 `trumans_to_recon.mine_walk_actions` 를 그대로 import 해서 돌린다 — 여기서 로직을
다시 쓰면 스크립트와 본코드가 갈라져서 확인의 의미가 없다.

사용 예시:
  # 7편, 편당 창 1개, 현행 fps 와 30fps 를 같이 표시
  python scripts/walk_window_reel.py --out ../../../results/2026-08-23_trumans_walk/walk_windows.mp4

  # 특정 편만, 편당 창 2개
  python scripts/walk_window_reel.py --recordings 0aa05d5a 4ac2c1b3 --per_rec 2
"""
import sys
from argparse import ArgumentParser
from os import makedirs, path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, HERE)
from trumans_to_recon import NUM_FRAMES, mine_walk_actions   # noqa: E402

TRUMANS_DEFAULT = "/data1/cympyc1785/data/trumans/Data_release"
RESULTS_DEFAULT = "/data1/cympyc1785/LatentCamVid/results/2026-08-23_trumans_walk"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# blend 의 render fps 는 편마다 다르다 (실측). 모션 배열 레이트가 아니라 **렌더 설정**이라는 걸
# 보여주려고 같이 찍는다 — `video_render` mp4 는 7편 전부 30fps / 프레임수 1:1 이다.
BLEND_FPS = {"00add26c": 25, "0a761819": 15, "0aa05d5a": 15, "1d43e076": 15,
             "2b4c9b84": 15, "3a19c7bb": 15, "4ac2c1b3": 15}
SEQUENCE = {"00add26c": "2023-01-17@00-55-00", "0a761819": "2023-01-14@22-33-09",
            "0aa05d5a": "2023-01-15@00-11-55", "1d43e076": "2023-02-12@14-50-00",
            "2b4c9b84": "2023-02-13@15-48-32", "3a19c7bb": "2023-02-16@17-11-54",
            "4ac2c1b3": "2023-02-19@22-24-43"}


def read_window(video: str, start: int, count: int):
    """mp4 의 [start, start+count) 프레임. 전량 디코드는 2,600 프레임이라 순차로 건너뛴다."""
    reader = imageio.get_reader(video)
    frames = []
    for index, frame in enumerate(reader):
        if index >= start + count:
            break
        if index >= start:
            frames.append(np.asarray(frame))
    reader.close()
    assert len(frames) == count, f"{video} 에서 {start}..{start + count} 를 못 읽었다 ({len(frames)})"
    return np.stack(frames)


def label_tile(frame, lines, scale: float):
    """타일 축소 + 좌상단 반투명 띠에 텍스트. 띠를 깔지 않으면 밝은 바닥에서 글자가 안 보인다."""
    image = Image.fromarray(frame)
    image = image.resize((int(image.width * scale), int(image.height * scale)), Image.BILINEAR)
    draw = ImageDraw.Draw(image, "RGBA")
    size = max(10, int(image.width / 30))
    font = ImageFont.truetype(FONT, size)
    band = (len(lines) + 0.4) * (size + 2)
    draw.rectangle([0, 0, image.width, band], fill=(0, 0, 0, 170))
    for row, text in enumerate(lines):
        colour = (255, 90, 90) if text.startswith("DROPPED") else (
            (120, 255, 120) if text.startswith("KEPT") else (255, 255, 255))
        draw.text((4, 2 + row * (size + 2)), text, font=font, fill=colour)
    return np.asarray(image)


def main(args):
    tiles, rows = [], []
    for uuid in args.recordings:
        sequence = SEQUENCE[uuid]
        video = path.join(args.trumans, "video_render", f"{sequence}.pkl.mp4")
        assert path.isfile(video), f"배포 렌더가 없다: {video}"
        blend_fps = BLEND_FPS[uuid]
        #    같은 시퀀스를 두 fps 로 채굴한다. `mine_walk_actions` 는 fps 를 인자로 받으므로
        #    로직 복제 없이 "임계만 바뀌면 어떻게 되나"를 그대로 잰다.
        now = mine_walk_actions(args.trumans, sequence, args.walk_speed, args.walk_frac,
                                args.walk_smooth, blend_fps, args.walk_max)
        fixed = mine_walk_actions(args.trumans, sequence, args.walk_speed, args.walk_frac,
                                  args.walk_smooth, args.true_fps, args.walk_max)
        for action in fixed[:args.per_rec]:
            #    "지금도 잡히나"는 시작 인덱스 일치로 보면 안 된다 — 임계가 바뀌면 보행 마스크가
            #    바뀌고 창 열거 stride 도 어긋나서, 같은 걸음을 잡아도 start 가 몇 프레임 밀린다.
            #    겹침 비율로 본다.
            kept = any(min(action["end"], a["end"]) - max(action["start"], a["start"])
                       >= 0.5 * NUM_FRAMES for a in now)
            #    `mine_walk_actions` 의 start 는 **시퀀스 로컬** 인덱스이고 배포 mp4 도 시퀀스
            #    길이와 1:1 이라 그대로 프레임 번호다 (전역 오프셋 lo 를 더하면 안 된다).
            frames = read_window(video, action["start"], NUM_FRAMES)
            status = (f"KEPT now (blend fps {blend_fps})" if kept
                      else f"DROPPED now (blend fps {blend_fps})")
            lines = [f"{uuid}  f{action['start']}..{action['end']}",
                     f"net {action['walk_net_m']:.2f}m  frac {action['walk_frac']:.2f}",
                     status]
            tiles.append(np.stack([label_tile(f, lines, args.scale) for f in frames]))
            rows.append({"rec": uuid, "blend_fps": blend_fps, "start": action["start"],
                         "net_m": action["walk_net_m"], "frac": action["walk_frac"],
                         "kept_now": kept, "n_now": len(now), "n_fixed": len(fixed)})

    assert tiles, "창이 하나도 없다"
    #    격자 채우기. 마지막 줄이 비면 검은 타일로 메운다 (np.stack 이 직사각형을 요구한다).
    cols = args.cols
    height, width = tiles[0].shape[1:3]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = [np.concatenate(tiles[i:i + cols], axis=2) for i in range(0, len(tiles), cols)]
    reel = np.concatenate(grid, axis=1)

    makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
    imageio.mimwrite(args.out, reel, fps=args.fps, codec="libx264", quality=6,
                     macro_block_size=1)

    print(f"\n{'rec':10s} {'blend':>5s} {'start':>6s} {'net_m':>6s} {'frac':>5s} "
          f"{'kept':>5s} {'n_now':>5s} {'n_30':>5s}")
    print("-" * 56)
    for row in rows:
        print(f"{row['rec']:10s} {row['blend_fps']:5d} {row['start']:6d} {row['net_m']:6.2f} "
              f"{row['frac']:5.2f} {str(row['kept_now']):>5s} {row['n_now']:5d} {row['n_fixed']:5d}")
    print("-" * 56)
    print(f"{'tiles':10s} {len(rows)}  grid {len(grid)}x{cols}  {reel.shape[2]}x{reel.shape[1]}")
    print(f"{'->':10s} {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recordings", default=list(SEQUENCE), nargs="*", type=str)
    parser.add_argument("--per_rec", default=1, type=int)            # 편당 타일 수 (이동량 큰 순)
    parser.add_argument("--cols", default=3, type=int)               # 격자 열 수
    parser.add_argument("--scale", default=0.5, type=float)          # 타일 축소 배율
    parser.add_argument("--fps", default=15, type=float)             # 릴 재생 fps
    parser.add_argument("--true_fps", default=30.0, type=float)      # 모션 배열 실제 레이트 (실측)
    parser.add_argument("--walk_speed", default=0.4, type=float)
    parser.add_argument("--walk_frac", default=0.8, type=float)
    parser.add_argument("--walk_smooth", default=9, type=int)
    parser.add_argument("--walk_max", default=8, type=int)
    parser.add_argument("--trumans", default=TRUMANS_DEFAULT, type=str)
    parser.add_argument("--out", default=path.join(
        RESULTS_DEFAULT, "walk_windows.mp4"), type=str)
    main(parser.parse_args())
