"""`trumans_first_pose_board.py` 의 board.json + 렌더 → chunk 별 contact sheet 한 장.

왜 따로 두나: 게이트 임계(특히 `min_crop_keep`)를 숫자로 정당화할 방법이 없다. crop_keep 0.65 가
"발이 잘렸다"인지 "구도가 타이트하다"인지는 **보고** 정해야 하는 문제다. 그래서 board 를 일부러
느슨한 임계로 뽑아놓고, 여기서 crop_keep 오름차순으로 늘어놓아 어디서 선을 그을지 눈으로 찾는다.

타일 정렬이 crop_keep 순인 게 핵심이다 — 방위각 순으로 붙이면 잘린 것과 안 잘린 것이 섞여서
경계가 안 보인다.

사용 예시:
  python scripts/board_contact_sheet.py --board out/board_lens18_s3/board.json --columns 6
"""
from argparse import ArgumentParser
from json import load
from os import path, makedirs
import sys

import cv2

# `import sys` 로 받아야 한다. `from sys import path` 로 미리 묶어두면 **cv2 가 sys.path 를
# 새 리스트 객체로 갈아끼우기 때문에** 그 참조가 죽은 리스트를 가리키고 insert 가 무시된다.
sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))
from lbm.overlay import contact_sheet, label_tile  # noqa: E402


def main(args):
    with open(args.board) as handle:
        board = load(handle)
    root = path.dirname(path.abspath(args.board))
    out_dir = args.out or root
    makedirs(out_dir, exist_ok=True)

    rows = []
    for chunk in board["chunks"]:
        shots = [c for c in chunk["candidates"] if c.get("image")]
        # 기본은 crop_keep 오름차순 = "가장 심하게 잘린 것"부터. 시트를 위에서 아래로 훑으면서
        # 어디서부터 봐줄 만한지 찾으라는 배치다. shot/facing 으로 묶어 보려면 --sort_by.
        order = {"crop": lambda c: (c["crop_keep"], c["center_offset"]),
                 "shot": lambda c: (c.get("height_frac", 0.0),),
                 "facing": lambda c: (c.get("facing_deg") or 0.0, c.get("height_frac", 0.0)),
                 "angle": lambda c: (c.get("eye_angle_deg") or 0.0, c.get("height_frac", 0.0))}
        shots.sort(key=order[args.sort_by])
        tiles = []
        for i, shot in enumerate(shots):
            image = cv2.imread(path.join(root, shot["image"]))
            assert image is not None, f"렌더 없음: {shot['image']}"
            if args.tile_width:
                scale = args.tile_width / image.shape[1]
                image = cv2.resize(image, (args.tile_width, int(round(image.shape[0] * scale))))
            facing_deg = shot.get("facing_deg")
            eye_deg = shot.get("eye_angle_deg")
            caption = (f"{shot.get('shot', '?')} h{shot.get('height_frac', 0.0):.2f} | "
                       f"{shot.get('facing', '?')} "
                       f"{'--' if facing_deg is None else format(facing_deg, '.0f')}d | "
                       f"{shot.get('camera_angle', '?')} "
                       f"{'--' if eye_deg is None else format(eye_deg, '.0f')}d | "
                       f"crop {shot['crop_keep']:.2f} clr {shot['clearance']:.2f} "
                       f"off {shot['center_offset']:.2f} | "
                       f"az{shot['azimuth_deg']:.0f} el{shot['elevation_deg']:.0f} "
                       f"r{shot['radius']:.2f}")
            tiles.append(label_tile(image, f"{i + 1:02d}", caption))
        sheet = contact_sheet(tiles, args.columns)
        name = f"board_{chunk['tag']}.png"
        cv2.imwrite(path.join(out_dir, name), sheet)
        crops = [s["crop_keep"] for s in shots]
        rows.append((chunk["tag"], len(tiles), sheet.shape[1], sheet.shape[0],
                     min(crops), max(crops), name))

    print(f"{'chunk':<18}{'tiles':>6}{'sheet':>14}{'crop min':>10}{'crop max':>10}  file")
    for tag, n, w, h, lo, hi, name in rows:
        print(f"{tag:<18}{n:>6}{f'{w}x{h}':>14}{lo:>10.2f}{hi:>10.2f}  {path.join(out_dir, name)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--board", required=True)                # trumans_first_pose_board.py 산출 board.json
    parser.add_argument("--out", default="")                     # 비우면 board.json 옆에
    parser.add_argument("--columns", type=int, default=6)        # 시트 열 수
    parser.add_argument("--sort_by", default="crop",             # 타일 정렬 축
                        choices=("crop", "shot", "facing", "angle"))
    parser.add_argument("--tile_width", type=int, default=480)   # 0 이면 원본 해상도 그대로
    main(parser.parse_args())
