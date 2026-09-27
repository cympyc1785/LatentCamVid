"""시작 카메라 후보가 각각 어떻게 보이고 판정은 무엇인가 — frame 0 렌더 contact sheet (R62).

`fit_hole_ladder.py --start_screen only|filter` 가 쓴 `<out>/<video>/<bank_dir>/start_screen.csv` 와
τ 뱅크(`<tau_bank_dir>/poses.npz`)의 후보별 **frame 0 카메라**(= 시작 pose. 손잡이·preset 무관)를 렌더해
한 장으로 잇는다. 타일 제목 = 후보 이름(`source` = 소스 frame0 카메라, 나머지는 격자 `az_el_cov`)
+ `OK` / `X <사유>`, 캡션 = 가시비율·국소 지면 높이.

env: vista4d (렌더러가 GPU 를 쓴다)
예시:
    CUDA_VISIBLE_DEVICES=1 python viz/render_start_screen.py --video snowboard \
        --tau_bank_dir bank_r62ts2 --bank_dir hole_bank_r62ts2
"""
import csv
import sys
from argparse import ArgumentParser
from os import path

import cv2
import numpy as np

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.overlay import contact_sheet, label_tile, paint_holes                  # noqa: E402
from lbm.render import open_renderer                                            # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402


def cand_of(variant_id: str) -> str:
    """variant_id 꼬리 → 후보 이름. 격자 후보는 `az_el_cov` 세 토막, 없으면 소스 frame0 = `source`."""
    tail = variant_id.split("__")[-1]
    return tail if len(tail.split("_")) == 3 and tail.split("_")[-1] in ("close", "medium", "wide") \
        else "source"


def main(args):
    out_root = path.join(ROOT, args.output_root)
    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    renderer, _ = open_renderer(args, out_root, graph)
    z = np.load(path.join(out_root, args.video, args.tau_bank_dir, "poses.npz"), allow_pickle=True)
    screen = {}
    with open(path.join(out_root, args.video, args.bank_dir, "start_screen.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            screen[(row["anchor_id"], row["cand_id"])] = row
    seen, tiles = set(), []
    for i, vid in enumerate(z["variant_id"]):
        vid, anchor = str(vid), str(z["anchor_id"][i])
        cand = cand_of(vid)
        if (anchor, cand) in seen:
            continue                      # 같은 후보의 다른 손잡이 단 — frame 0 은 같다
        seen.add((anchor, cand))
        pose0 = np.asarray(z["cam_c2w"][i][0], dtype=np.float64)
        K = np.array(renderer.K_src[0], dtype=np.float64).copy()
        shot = renderer.render(pose0, K=K, frame=0, height=args.tile_height, width=args.tile_width)
        img = paint_holes(np.ascontiguousarray(shot["rgb"]), shot["valid"])
        row = screen.get((anchor, cand), {})
        ok = row.get("ok") == "1"
        label = f"{cand}  " + ("OK" if ok else f"X {row.get('reason', '?')}")
        cap = (f"{anchor} vis{float(row.get('subject_visible', 'nan')):.2f} "
               f"lg{float(row.get('local_ground', 'nan')):+.3f} "
               f"g{float(row.get('ground_clear', 'nan')):.3f}")
        tiles.append((0 if cand == "source" else 1, label_tile(img, label, cap)))
    tiles = [t for _, t in sorted(tiles, key=lambda x: x[0])]   # source 를 맨 앞으로
    sheet = contact_sheet(tiles, args.columns)
    dst = path.join(out_root, args.video, args.bank_dir, args.name)
    cv2.imwrite(dst, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR) if args.rgb_input else sheet)
    print(f"-> {dst}  ({len(tiles)} tiles)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--tau_bank_dir", required=True)
    parser.add_argument("--bank_dir", required=True)                # start_screen.csv 가 있는 곳
    parser.add_argument("--output_root", default="out")
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--seg_root", default=None)
    parser.add_argument("--seg_static_root", default=None)
    parser.add_argument("--cloud_source", default="memory", choices=["npz", "memory"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed_focal", action="store_true", default=True)
    parser.add_argument("--columns", default=6, type=int)
    parser.add_argument("--tile_height", default=270, type=int)
    parser.add_argument("--tile_width", default=480, type=int)
    parser.add_argument("--name", default="start_screen.jpg")
    parser.add_argument("--rgb_input", action="store_true", default=True)  # 렌더 rgb 는 RGB 순서
    main(parser.parse_args())
