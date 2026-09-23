"""F1(제자리형 subject 를 static 으로 강등) 이 어떤 물체를 걸러내는지 눈으로 보는 영상.

왜 필요한가: `fix.log` F1 은 `moving` 판정을 `center_drift_u > 0.05` 로 바꾸자는 제안인데,
숫자만 보면 "0.0125 는 정지" 라고 단정할 수 없다. 사람이 팔만 휘두르는 chunk 와 실제로
걸어가는 chunk 는 **track center 궤적**을 나란히 놓고 봐야 갈린다 (path_len 은 둘 다 크다 —
in-place 예시의 path_len 0.259 는 걸어가는 예시의 0.4 와 구별이 안 된다).

그래서 화면을 둘로 나눈다. 왼쪽은 소스 영상 위에 OBB + 2D bbox + **track center 투영 궤적**,
오른쪽은 그 궤적의 조감도(graph frame G, xy). 조감도는 두 칸이다:
  · 위  = 전 예시 **공통 눈금** (`--span_common`, 기본 3.0 u) — 클립끼리 직접 비교하라고
  · 아래 = 그 클립에 맞춘 확대 — 제자리 흔들림의 모양을 보라고
둘 다 `--drift_thresh`(0.05 u) 반경 원을 그린다. 원 안에서 안 벗어나면 F1 이 강등한다.

`center_drift_u` 는 **순변위** `|c[-1] - c[0]|` 다 (경로 길이 아님). `scene_graph.json` 이 이미
그 값을 노드에 싣고 있으므로 여기서 다시 계산하지 않고 읽는다 — 두 값이 갈리면 그게 버그다.

입력  : `<eval_data>/eval_data/recon_and_seg/<video>/`, `<out>/<video>/scene_graph.json`
출력  : `<out_dir>/<video>__<node>__f1.mp4` + stdout 표

예시 (env vista4d):
    python viz/viz_f1_inplace.py \
        --eval_data /data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM --out out_dynpose \
        --out_dir results/20260905_f1_inplace \
        --clips 14325a2c-e53c-4141-91bc-f76faa6519fe:dyn_0 \
                0af9210e-d41f-46f0-abc5-985f59645b28:dyn_0
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import cv2
import imageio.v2 as imageio
import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from scene_graph.io import load_scene                                              # noqa: E402
from scene_graph.lift import apply_transform, project_points                       # noqa: E402
from scene_graph.obb import node_obb_at, obb_corners                               # noqa: E402
from scene_graph.viz import draw_obb                                               # noqa: E402

EVAL_DATA_DEFAULT = "/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM"
VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "video_generation", "models", "Vista4D"))

# 프레임이 RGB numpy 라 cv2 의 BGR 규약이 안 통한다 — 여기 튜플은 전부 **RGB** 다
# (`scene_graph/viz.py` PALETTE 과 같은 규약). BGR 로 적었더니 "빨강" 이 파랗게 나왔다.
CYAN, YELLOW, WHITE = (0, 255, 255), (255, 220, 40), (255, 255, 255)
RED, GREEN = (255, 70, 70), (120, 230, 90)


def draw_text(image, text, org, color, scale=0.5, thick=1):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def topdown_panel(size: int, track_xy: np.ndarray, upto: int, span: float, thresh: float,
                  title: str, verdict_color):
    """track center 조감도 한 칸. `upto` 프레임까지만 실선, 나머지는 회색 예고선."""
    canvas = np.full((size, size, 3), 22, dtype=np.uint8)
    center = track_xy[0]
    to_px = lambda p: np.round((np.asarray(p, float) - center) / span * size
                               + size / 2).astype(int)

    for grid in np.arange(-span / 2, span / 2 + 1e-9, span / 6):        # 눈금
        g0, g1 = to_px([center[0] + grid, center[1] - span / 2]), to_px([center[0] + grid, center[1] + span / 2])
        cv2.line(canvas, tuple(g0), tuple(g1), (42, 42, 42), 1)
        g0, g1 = to_px([center[0] - span / 2, center[1] + grid]), to_px([center[0] + span / 2, center[1] + grid])
        cv2.line(canvas, tuple(g0), tuple(g1), (42, 42, 42), 1)

    radius_px = int(round(thresh / span * size))
    if radius_px >= 2:
        cv2.circle(canvas, tuple(to_px(center)), radius_px, verdict_color, 1, cv2.LINE_AA)

    future = to_px(track_xy).reshape(-1, 1, 2)
    cv2.polylines(canvas, [future], False, (70, 70, 70), 1, cv2.LINE_AA)
    past = to_px(track_xy[:upto + 1]).reshape(-1, 1, 2)
    cv2.polylines(canvas, [past], False, WHITE, 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(to_px(track_xy[0])), 4, GREEN, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(to_px(track_xy[upto])), 5, verdict_color, -1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, tuple(to_px(track_xy[0])), tuple(to_px(track_xy[-1])),
                    RED, 1, cv2.LINE_AA, tipLength=0.05)

    draw_text(canvas, title, (8, 20), (210, 210, 210), 0.5)
    draw_text(canvas, f"circle r={thresh:.2f}u (F1 thresh)", (8, size - 10), verdict_color, 0.5)
    return canvas


def render_clip(recon, graph, node, args):
    rgb = recon["video"]
    K, cam_c2w = recon["K"], recon["cam_c2w"]
    T_wg = np.asarray(graph["frames"]["T_wg"], float)
    track_xy = np.asarray(node["track"]["center_smooth"], float)[:, :2]
    centers_g = np.asarray(node["track"]["center_smooth"], float)

    drift = float(node["center_drift_u"])
    plen = float(node["path_len_u"])
    demote = drift <= args.drift_thresh
    verdict_color = RED if demote else GREEN
    verdict = (f"F1: moving -> STATIC  (drift {drift:.3f} <= {args.drift_thresh:.2f} u)" if demote
               else f"F1: stays MOVING  (drift {drift:.3f} > {args.drift_thresh:.2f} u)")

    height, width = rgb.shape[1:3]
    panel = height // 2
    zoom_span = max(float(np.abs(track_xy - track_xy[0]).max()) * 2.6, 0.12)

    frames = []
    for f in range(rgb.shape[0]):
        image = np.ascontiguousarray(rgb[f].copy())
        if f in node["track"]["frames"]:
            corners = obb_corners(*node_obb_at(node, f))
            draw_obb(image, corners, T_wg, K[f], cam_c2w[f], CYAN,
                     f"{node['id']} {node['label']}")
            # 2D bbox 는 `scene_graph.json` 에 없다 (`bbox_by_frame` 은 빌드 중 메모리에만 산다).
            # 투영 OBB 꼭짓점의 축정렬 bbox 로 대신한다 — 구도 판단엔 그걸로 충분하다.
            uv_obb, z_obb = project_points(apply_transform(T_wg, corners), K[f], cam_c2w[f])
            if (z_obb > 1e-6).all():
                x0, y0 = np.round(uv_obb.min(axis=0)).astype(int)
                x1, y1 = np.round(uv_obb.max(axis=0)).astype(int)
                cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), YELLOW, 1, cv2.LINE_AA)
        uv, z = project_points(apply_transform(T_wg, centers_g), K[f], cam_c2w[f])
        trail = [tuple(np.round(p).astype(int)) for p, zz in zip(uv[:f + 1], z[:f + 1]) if zz > 1e-6]
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(image, a, b, WHITE, 2, cv2.LINE_AA)
        if trail:
            cv2.circle(image, trail[0], 4, GREEN, -1, cv2.LINE_AA)
            cv2.circle(image, trail[-1], 5, verdict_color, -1, cv2.LINE_AA)

        draw_text(image, f"{graph['video'][:8]}  {node['id']} {node['label']}  f{f:03d}",
                  (10, 32), WHITE, 0.8, 2)
        draw_text(image, verdict, (10, 64), verdict_color, 0.8, 2)
        draw_text(image, f"drift {drift:.3f}u   path_len {plen:.3f}u   "
                         f"area {node['max_area_frac']:.3f}", (10, height - 14), (220, 220, 220),
                  0.7, 2)

        right = np.concatenate([
            topdown_panel(panel, track_xy, f, args.span_common, args.drift_thresh,
                          f"top-down (G xy)  span {args.span_common:.1f}u  COMMON", verdict_color),
            topdown_panel(panel, track_xy, f, zoom_span, args.drift_thresh,
                          f"zoom  span {zoom_span:.2f}u", verdict_color)], axis=0)
        if right.shape[0] != height:                    # 홀수 높이 보정
            right = np.concatenate([right, np.full((height - right.shape[0], panel, 3), 22, np.uint8)])
        frames.append(np.concatenate([image, right], axis=1))
    return frames, drift, plen, demote


def main():
    parser = ArgumentParser()
    # `<scene>:<node_id>` 목록. node_id 를 빼면 그 씬의 첫 동적 노드.
    parser.add_argument("--clips", nargs="+", required=True, type=str)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--out", default=path.join(CINEMATRAJ_ROOT, "out_dynpose"), type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--fps", default=10.0, type=float)
    parser.add_argument("--drift_thresh", default=0.05, type=float)   # fix.log F1 기준
    parser.add_argument("--span_common", default=3.0, type=float)     # 전 클립 공통 조감도 눈금
    args = parser.parse_args()

    makedirs(args.out_dir, exist_ok=True)
    rows = []
    for clip in args.clips:
        scene, _, node_id = clip.partition(":")
        with open(path.join(args.out, scene, "scene_graph.json"), encoding="utf-8") as file:
            graph = json.load(file)
        nodes = [n for n in graph["nodes"] if n["kind"] == "dyn"]
        node = next((n for n in nodes if n["id"] == node_id), None) if node_id else nodes[0]
        assert node is not None, f"{scene}: {node_id} 없음 (있는 것: {[n['id'] for n in nodes]})"
        node["track"]["frames"] = [int(f) for f in node["track"]["frames"]]

        recon = load_scene(args.eval_data, scene, args.vista4d_root)
        frames, drift, plen, demote = render_clip(recon, graph, node, args)
        out_path = path.join(args.out_dir, f"{scene[:8]}__{node['id']}__f1.mp4")
        imageio.mimwrite(out_path, frames, fps=int(round(args.fps)), codec="libx264",
                         quality=6, macro_block_size=1)
        rows.append((scene[:8], node["id"], node["label"], f"{drift:.4f}", f"{plen:.4f}",
                     f"{node['max_area_frac']:.3f}", "demote" if demote else "keep",
                     path.basename(out_path)))

    header = ("scene", "node", "label", "drift_u", "path_len_u", "area", "f1", "file")
    widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"\nthresh    center_drift_u <= {args.drift_thresh} -> static (fix.log F1)")
    print(f"out_dir   {args.out_dir}")


if __name__ == "__main__":
    main()
