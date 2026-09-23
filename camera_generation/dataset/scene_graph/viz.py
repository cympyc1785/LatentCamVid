"""사람이 승인하기 위한 그림. `obb_overlay.mp4` 를 **제일 먼저** 볼 것.

박스가 물체를 안 감싸면 하류가 전부 무효다 — 후보 카메라 원점도, framing 게이트도, VLM 에게
보여줄 bbox 오버레이도 전부 이 OBB 에서 나온다. reproj_px 표가 통과해도 박스 크기가 틀릴 수
있으므로(중심만 검사한다) 눈으로 봐야 한다.

영상은 `imageio` + libx264 로 쓴다. cv2 mp4v 는 파일은 멀쩡한데 VS Code 뷰어에서 안 열려서
조용히 지나간다.
"""
import cv2
import numpy as np

from .lift import apply_transform, project_points
from .obb import OBB_EDGES, node_obb_at, obb_corners

PALETTE = [(0, 255, 255), (255, 128, 0), (0, 255, 128), (255, 0, 255),
           (128, 200, 255), (255, 255, 0), (200, 100, 255), (100, 255, 200)]


def draw_obb(image: np.ndarray, corners_g: np.ndarray, T_wg: np.ndarray, K: np.ndarray,
             cam_c2w: np.ndarray, color, label: str = "", thickness: int = 2):
    """12 edge 투영. 카메라 뒤 꼭짓점이 섞인 edge 는 그리지 않는다 (투영이 뒤집혀 화면을 가로지른다)."""
    uv, z = project_points(apply_transform(T_wg, corners_g), K, cam_c2w)
    drawn = False
    for a, b in OBB_EDGES:
        if z[a] <= 1e-6 or z[b] <= 1e-6:
            continue
        cv2.line(image, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)),
                 color, thickness, cv2.LINE_AA)
        drawn = True
    if label and drawn and z.min() > 1e-6:
        anchor = uv[np.argmin(uv[:, 1])]
        cv2.putText(image, label, (int(anchor[0]), max(int(anchor[1]) - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return drawn


def obb_overlay_video(output_path: str, video: np.ndarray, nodes: list, T_wg: np.ndarray,
                      K: np.ndarray, cam_c2w: np.ndarray, fps: float, draw_bbox: bool = True,
                      kinds: tuple = ("dyn",)):
    """`kinds` 로 그릴 노드 종류를 고른다. **기본은 동적만** — 정적 노드를 같이 그리면 상자가
    6~11개가 되어 정작 봐야 할 subject OBB 가 안 보인다. 정적까지 보려면 `kinds=("dyn","stat")`
    로 부르고(그러면 `obb_overlay_all.mp4` 가 나온다) 정적은 1px 로 얇게, bbox 없이 그린다.
    """
    import imageio.v2 as imageio

    drawn_nodes = [node for node in nodes if node["kind"] in kinds]
    frames = []
    for f in range(video.shape[0]):
        image = np.ascontiguousarray(video[f].copy())
        for n, node in enumerate(drawn_nodes):
            color = PALETTE[n % len(PALETTE)]
            if f not in node["track"]["frames"]:
                continue
            static = node["kind"] != "dyn"
            corners = obb_corners(*node_obb_at(node, f))
            draw_obb(image, corners, T_wg, K[f], cam_c2w[f], color,
                     f"{node['id']} {node['label']}", thickness=1 if static else 2)
            if draw_bbox and not static and f in node["bbox_by_frame"]:
                x0, y0, x1, y1 = (int(round(v)) for v in node["bbox_by_frame"][f])
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        frames.append(image)

    # cv2 mp4v 는 VS Code 뷰어에서 안 열린다 -> imageio + libx264.
    imageio.mimwrite(output_path, frames, fps=int(round(fps)), codec="libx264",
                     quality=6, macro_block_size=1)
    return output_path


def topdown_png(output_path: str, static_points_g: np.ndarray, nodes: list,
                cam_centers_g: np.ndarray, size: int = 900, margin: float = 0.15):
    """G 의 xy 평면 조감도. 씬 규모와 카메라 경로가 한눈에 들어와야 한다."""
    points = static_points_g[:, :2]
    if len(points) > 200000:
        points = points[np.linspace(0, len(points) - 1, 200000).astype(int)]
    everything = [points, cam_centers_g[:, :2]] + [
        obb_corners(np.asarray(node["obb"]["center"]), np.asarray(node["obb"]["extent"]),
                    np.asarray(node["obb"]["R"]))[:, :2] for node in nodes]
    stacked = np.concatenate(everything, axis=0)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    span = max((hi - lo).max(), 1e-6) * (1 + 2 * margin)
    center = (lo + hi) / 2
    to_px = lambda p: np.round((p - center) / span * size + size / 2).astype(int)

    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    for x, y in to_px(points):
        if 0 <= x < size and 0 <= y < size:
            canvas[y, x] = (90, 90, 90)
    path_px = to_px(cam_centers_g[:, :2])
    for a, b in zip(path_px[:-1], path_px[1:]):
        cv2.line(canvas, tuple(a), tuple(b), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(path_px[0]), 6, (255, 255, 255), -1)

    for n, node in enumerate(nodes):
        color = PALETTE[n % len(PALETTE)]
        corners = obb_corners(np.asarray(node["obb"]["center"]), np.asarray(node["obb"]["extent"]),
                              np.asarray(node["obb"]["R"]))[:4, :2]
        cv2.polylines(canvas, [to_px(corners)[[0, 1, 3, 2]].reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        track = np.asarray(node["track"]["center_smooth"])[:, :2]
        cv2.polylines(canvas, [to_px(track).reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        cx, cy = to_px(np.asarray(node["obb"]["center"])[:2])
        cv2.putText(canvas, node["id"], (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"top-down (graph frame G, 1 unit = S)  span={span:.2f}u",
                (12, size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, canvas)
    return output_path
