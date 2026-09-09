"""`seg_instances` (SAM3 track) 를 소스 영상 위에 색으로 덧씌운 영상 + track/anchor 표.

왜 필요한가: `obb_overlay.mp4` 는 **scene graph 가 살아남긴 노드의 박스**만 보여준다. 그런데
박스가 이상할 때 원인은 두 군데다 — ① SAM3 마스크 자체가 엉뚱한 걸 잡았거나 ② 마스크는
맞는데 lift/OBB 가 틀렸거나. 박스만 봐서는 못 가른다. 마스크를 원본 위에 직접 칠해 보면
그 자리에서 갈린다 (golf 의 `dyn_0` 는 프레임의 91% 를 덮는 마스크였다 — 박스만 봤을 때는
"박스가 크다" 로만 보였다).

`--drop` 은 `scene_graph.json` 의 `diagnostics.dropped_tracks` 까지 같이 칠한다. 노드로
안 남은 track 이 사실은 우리가 원하던 물체였는지 확인하는 용도다 (golf 의 `golf ball` 은
`max_area_frac 7e-05` 로 떨어졌다).

표에는 뱅크가 실제로 **anchor 로 쓴 노드**를 같이 적는다. track 목록만 보면 "이게 카메라를
만들 때 쓰였나"가 안 보인다.

입력  : `<eval_data>/eval_data/{recon_and_seg,seg_instances,seg_instances_static}/<video>/`
        `<out>/<video>/scene_graph.json` (선택 — 노드/기각 사유/anchor 라벨)
        `<out>/<video>/<bank>/bank.json` (선택 — anchor 사용 횟수)
출력  : `<out_dir>/<video>_seg.mp4` + stdout 표

예시 (env vista4d):
    python scripts/seg_overlay_video.py --video golf --banks hole_bank_k6_d77
    python scripts/seg_overlay_video.py --video golf --drop --fps 10
"""
import json
import sys
from argparse import ArgumentParser
from collections import Counter
from os import makedirs, path

import imageio.v2 as imageio
import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from scene_graph.io import load_scene                                               # noqa: E402

EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "..", "..", "video_generation", "models", "Vista4D"))

# track 색. 순서가 고정이어야 두 영상(예: dyn / dyn+stat)을 나란히 놓고 같은 track 을 찾는다.
PALETTE = np.array([[255, 60, 60], [60, 200, 255], [120, 255, 90], [255, 200, 40],
                    [220, 100, 255], [255, 140, 40], [80, 255, 220], [200, 200, 200],
                    [255, 90, 160], [140, 160, 255], [180, 255, 40], [255, 240, 140]],
                   dtype=np.uint8)


def draw_text(frame: np.ndarray, text: str, org, color, scale: float = 0.5):
    """cv2 는 여기서만 쓴다 — import 를 위로 올리면 `sys.path` 를 갈아끼워 다른 import 가 깨진다."""
    import cv2
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                tuple(int(c) for c in color), 1, cv2.LINE_AA)


def overlay_masks(frame: np.ndarray, layers, alpha: float):
    """(mask, color, label) 목록을 한 프레임에 합성. 겹치면 **뒤에 온 것이 이긴다** —
    반투명 누적으로 섞으면 두 track 이 겹친 자리가 제3의 색이 되어 어느 track 인지 못 읽는다."""
    out = frame.astype(np.float32)
    for mask, color, _ in layers:
        if not mask.any():
            continue
        out[mask] = (1.0 - alpha) * out[mask] + alpha * np.asarray(color, np.float32)
    out = out.astype(np.uint8)
    for mask, color, label in layers:
        if not label or not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        draw_text(out, label, (int(xs.min()) + 3, max(int(ys.min()) - 5, 14)), color)
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--out", default=path.join(CINEMATRAJ_ROOT, "out"), type=str)
    parser.add_argument("--out_dir", default="", type=str)   # 비우면 <out>/<video>/vis
    # 뱅크 이름들. anchor 로 실제 쓰인 노드를 표에 적는 데만 쓴다.
    parser.add_argument("--banks", nargs="*", default=["hole_bank_k6_d77"], type=str)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--alpha", default=0.55, type=float)
    # 기각된 track 도 칠할지. 기본은 끔 — golf 는 기각 track 이 30개라 화면이 안 보인다.
    parser.add_argument("--drop", dest="drop", action="store_true", default=False)
    parser.add_argument("--no_drop", dest="drop", action="store_false")
    parser.add_argument("--static", dest="static", action="store_true", default=True)
    parser.add_argument("--no_static", dest="static", action="store_false")
    args = parser.parse_args()

    recon = load_scene(args.eval_data, args.video, args.vista4d_root)
    rgb = recon["video"]
    num_frames = len(rgb)

    graph_path = path.join(args.out, args.video, "scene_graph.json")
    graph = None
    if path.isfile(graph_path):
        with open(graph_path, encoding="utf-8") as file:
            graph = json.load(file)

    # scene graph 노드가 어느 (kind, track_id) 에서 왔는지. `merged_from` 이 있으면 흡수된
    # track 도 같은 노드로 친다 — 안 그러면 흡수된 쪽이 "노드 없음"으로 찍힌다.
    node_of = {}
    if graph is not None:
        for node in graph["nodes"]:
            for key in [(node["kind"], int(node["track_id"]))] + \
                       [(node["kind"], int(t)) for t in node.get("merged_from", [])
                        if str(t).isdigit()]:
                node_of[key] = node["id"]
    dropped = {}
    if graph is not None:
        for row in graph.get("diagnostics", {}).get("dropped_tracks", []):
            dropped[(row["kind"], int(row["track_id"]))] = row

    anchors = Counter()
    for bank in args.banks:
        bank_json = path.join(args.out, args.video, bank, "bank.json")
        if not path.isfile(bank_json):
            continue
        with open(bank_json, encoding="utf-8") as file:
            for variant in json.load(file)["variants"]:
                anchors[str(variant.get("anchor_id"))] += 1

    # 그릴 track 목록. 노드로 살아남은 것 먼저, 기각은 뒤 (색 순서를 노드에 고정하려고).
    entries = []
    for seg in recon["segs"]:
        if seg.kind == "stat" and not args.static:
            continue
        for track_id in seg.track_ids():
            key = (seg.kind, track_id)
            node_id = node_of.get(key)
            if node_id is None and not args.drop:
                continue
            entries.append({"seg": seg, "kind": seg.kind, "track_id": track_id,
                            "keyword": seg.tracks[track_id]["keyword"], "node": node_id,
                            "drop": dropped.get(key)})
    entries.sort(key=lambda e: (e["node"] is None, e["kind"], e["track_id"]))
    for index, entry in enumerate(entries):
        entry["color"] = PALETTE[index % len(PALETTE)]

    frames = []
    for f in range(num_frames):
        layers = []
        for entry in entries:
            mask = entry["seg"].track_mask(entry["track_id"], f)
            tag = entry["node"] or f"({entry['kind']}#{entry['track_id']})"
            layers.append((mask, entry["color"], f"{tag} {entry['keyword']}"))
        frame = overlay_masks(rgb[f], layers, args.alpha)
        draw_text(frame, f"{args.video}  f{f:03d}", (8, 22), (255, 255, 255), 0.6)
        frames.append(frame)

    out_dir = args.out_dir or path.join(args.out, args.video, "vis")
    makedirs(out_dir, exist_ok=True)
    out_path = path.join(out_dir, f"{args.video}_seg.mp4")
    imageio.mimwrite(out_path, frames, fps=args.fps, codec="libx264", quality=6,
                     macro_block_size=1)

    header = ("track", "keyword", "node", "area_max", "score", "nvis", "anchor_use", "drop_reason")
    rows = []
    for entry in entries:
        seg, tid = entry["seg"], entry["track_id"]
        areas = [float(entry["seg"].track_mask(tid, f).mean()) for f in
                 seg.tracks[tid]["frames"][::max(1, len(seg.tracks[tid]["frames"]) // 12)]]
        rows.append((f"{entry['kind']}#{tid}", entry["keyword"], entry["node"] or "-",
                     f"{max(areas) if areas else 0:.4f}",
                     f"{np.mean(seg.tracks[tid]['scores']):.3f}",
                     str(len(seg.tracks[tid]["frames"])),
                     str(anchors.get(entry["node"], 0)),
                     "; ".join(entry["drop"]["reasons"]) if entry["drop"] else "-"))
    widths = [max(len(header[i]), max((len(r[i]) for r in rows), default=0)) for i in range(8)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    if graph is not None:
        gravity = graph["gravity"]
        print(f"\ngravity   {gravity['method']}  angle_to_cam_up "
              f"{gravity['angle_to_cam_up_deg']:.1f} deg  ground_z {graph['ground']['ground_z']}")
    print(f"anchors   {dict(anchors)}  (banks: {' '.join(args.banks)})")
    print(f"video     {out_path}  {num_frames} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
