"""이 코드가 답하는 질문: graph 설정을 바꾸면 (OBB trim · tracker 위치 · 기준 프레임) anchor 3D OBB 가 소스 영상에서 subject 에 제대로 붙나?

R267 (2026-10-08 사용자 "tracker 기반을 우선 fitting 전에 3D OBB overlay 까지만 검사받아서"). 같은 영상의 scene_graph.json 여럿을
열(column)로 나란히 놓고, route 가 실제로 고른 anchor (1·2순위, `preset_route_<gen>.json` 의 `anchor_ids`) 의 OBB 만
소스 프레임에 그린다 (사용자 규칙: 실제로 쓰는 1·2순위만). 상자 위치·yaw = `node_obb_at` (그 프레임), 크기 = `obb.extent`.
  예) python viz/viz_obb_compare.py --video <uuid> --graph old=out_dynpose --graph tracker=/…/tmp/r265/out_dynpose --out x.mp4
env: vista4d.
"""
import json
from argparse import ArgumentParser
from os import path

import cv2
import imageio.v2 as iio
import numpy as np

from scene_graph.obb import node_obb_at, obb_corners
from scene_graph.viz import PALETTE, draw_obb

EVAL = "/data1/cympyc1785/data/DynPose-100K/eval_data"


def grid_main(a):
    """[R271] `--videos` 여럿 + graph 하나 → 영상마다 한 칸, `--grid` x `--grid` 격자 (사용자 "OBB overlay 영상 4x4 로").
    anchor 는 각 영상의 `--route_name` (graph 루트 아래, 없으면 out_dynpose 의 preset_route_d343.json) 에서 1·2순위."""
    label, root = a.graph[0].split("=", 1)
    notes = json.load(open(a.tile_note)) if a.tile_note else {}
    tiles = []
    for v in a.videos:
        rp = path.join(root, v, a.route_name)
        rp = rp if path.exists(rp) else path.join("out_dynpose", v, "preset_route_d343.json")
        anchors = json.load(open(rp))["anchor_ids"]
        g = json.load(open(path.join(root, v, "scene_graph.json")))
        nodes = {n["id"]: n for n in g["nodes"]}
        T_wg = np.linalg.inv(np.asarray(g["frames"]["T_gw"], dtype=float))
        cam = np.load(f"{EVAL}/recon_and_seg/{v}/cameras.npz")
        C, I = cam["cam_c2w"], cam["intrinsics"]
        frames = []
        for t, im in enumerate(iio.get_reader(f"{EVAL}/recon_and_seg/{v}/video.mp4")):
            im = np.ascontiguousarray(np.asarray(im).copy())
            K = np.array([[I[t][0], 0, I[t][2]], [0, I[t][1], I[t][3]], [0, 0, 1]])
            for rank, aid in enumerate(anchors):
                if aid in nodes:
                    draw_obb(im, obb_corners(*node_obb_at(nodes[aid], t)), T_wg, K, C[t], PALETTE[rank % len(PALETTE)],
                             f"{rank + 1}:{nodes[aid]['label']}", thickness=4)
            w = a.width // 2 * 2
            im = cv2.resize(im, (w, int(w * 9 / 16) // 2 * 2))
            cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
            note = notes.get(v, "")
            note = note if isinstance(note, str) else " ".join(f"{x[0]}:{x[2]} head{x[3]}" for x in note)
            cv2.putText(im, f"{v[:8]} t{t} {note}"[:60], (4, 16), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(im)
        tiles.append(frames)
    T = min(len(x) for x in tiles)
    n = a.grid
    blank = np.zeros_like(tiles[0][0])
    out = []
    for t in range(T):
        cells = [x[t] for x in tiles] + [blank] * (n * n - len(tiles))
        out.append(np.vstack([np.hstack(cells[r * n:(r + 1) * n]) for r in range(n)]))
    iio.mimwrite(a.out, out, fps=a.fps, macro_block_size=1)
    print(f"{label}: {len(tiles)} 영상 -> {a.out}")


def main(a):
    cam = np.load(f"{EVAL}/recon_and_seg/{a.video}/cameras.npz")
    C, I = cam["cam_c2w"], cam["intrinsics"]
    video = [np.asarray(f) for f in iio.get_reader(f"{EVAL}/recon_and_seg/{a.video}/video.mp4")]
    route = json.load(open(a.route)) if a.route else None
    anchors = a.anchors or (route["anchor_ids"] if route else ["dyn_0", "dyn_1"])
    cols = []
    for spec in a.graph:
        label, root = spec.split("=", 1)
        g = json.load(open(path.join(root, a.video, "scene_graph.json")))
        nodes = {n["id"]: n for n in g["nodes"]}
        T_wg = np.linalg.inv(np.asarray(g["frames"]["T_gw"], dtype=float))
        frames = []
        for t, im in enumerate(video):
            im = np.ascontiguousarray(im.copy())
            K = np.array([[I[t][0], 0, I[t][2]], [0, I[t][1], I[t][3]], [0, 0, 1]]) if np.asarray(I[t]).ndim == 1 else np.asarray(I[t])
            for rank, aid in enumerate(anchors):
                n = nodes.get(aid)
                if n is None:
                    continue
                draw_obb(im, obb_corners(*node_obb_at(n, t)), T_wg, K, C[t], PALETTE[rank % len(PALETTE)],
                         f"{rank + 1}:{aid} {n['label']}", thickness=3)
            im = cv2.resize(im, (a.width // 2 * 2, int(a.width * im.shape[0] / im.shape[1]) // 2 * 2))   # yuv420p 는 짝수 크기
            cv2.rectangle(im, (0, 0), (im.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(im, f"{label} | t{t}", (6, 18), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            info = " ".join(f"{aid}:{nodes[aid].get('center_source', 'obb')} d{nodes[aid].get('center_drift_u', 0):.3f}"
                            for aid in anchors if aid in nodes)
            cv2.putText(im, info[:90], (6, 38), 0, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
            frames.append(im)
        cols.append(frames)
    T = min(len(c) for c in cols)
    iio.mimwrite(a.out, [np.hstack([c[t] for c in cols]) for t in range(T)], fps=a.fps, macro_block_size=1)
    print("anchors", anchors, "->", a.out)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--videos", nargs="*", default=None)       # [R271] 격자 모드 (graph 하나)
    ap.add_argument("--grid", default=4, type=int)
    ap.add_argument("--tile_note", default=None)               # json {video: 문자열 | [[rank, id, label, 값], ...]} — 칸 제목에 덧붙임
    ap.add_argument("--route_name", default="preset_route_d343.json")
    ap.add_argument("--graph", action="append", required=True)   # label=<output_root> (그 아래 <video>/scene_graph.json)
    ap.add_argument("--route", default=None)                      # anchor 를 읽을 preset_route json (없으면 --anchors / dyn_0,dyn_1)
    ap.add_argument("--anchors", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", default=480, type=int)
    ap.add_argument("--fps", default=8, type=int)
    args = ap.parse_args()
    grid_main(args) if args.videos else main(args)
