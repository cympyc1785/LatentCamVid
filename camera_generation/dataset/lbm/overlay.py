"""VLM 이 실제로 보는 그림을 만든다. **이 설계의 전부가 여기 걸려 있다.**

look-before-move 의 주장은 "LLM 에게 공간을 설명하지 말고 보여줘라"다. 그러면 보여주는 그림이
사람 눈에도 말이 돼야 한다. 그래서 타일마다 세 가지를 겹친다:

  ① subject OBB 12-edge + 2D bbox  — "어느 게 subject 인지" 를 추측하지 않게. 렌더는 hole 이
     섞여 있어 물체 경계가 흐릿한데, 박스가 있으면 VLM 이 그걸 기준으로 구도를 말할 수 있다
  ② rule-of-thirds 3x3 + 중앙 80% safe frame — "구도가 맞나"를 눈금 없이 물으면 답이 안 나온다
  ③ 큰 board label (A1..C9) — 응답이 좌표가 아니라 **라벨**이어야 파싱이 안 깨진다

hole 은 마젠타로 칠한다. 검정으로 두면 VLM 이 "어두운 배경"으로 읽고 coverage 문제를 무시한다.

소스 프레임 패널을 따로 붙이는 이유(사용자 확정 사항): 후보 렌더만 보면 "원래 씬이 어떻게
생겼는지"의 근거가 없다. 점군 렌더는 전부 조금씩 뭉개져 있어서 VLM 이 렌더 아티팩트를 씬의
성질로 오해한다.
"""
import cv2
import numpy as np

HOLE_COLOR = (255, 0, 255)     # BGR 아님 — imageio 로 나가므로 RGB 로 통일한다
OBB_COLOR = (0, 255, 255)
BBOX_COLOR = (255, 255, 0)
GRID_COLOR = (255, 255, 255)


def paint_holes(rgb: np.ndarray, valid: np.ndarray, color=HOLE_COLOR):
    out = np.ascontiguousarray(rgb.copy())
    out[~valid] = color
    return out


def project_world(points_world: np.ndarray, K: np.ndarray, cam_c2w: np.ndarray):
    """(n,3) world → (uv (n,2), z_cam (n,))."""
    w2c = np.linalg.inv(np.asarray(cam_c2w, dtype=float))
    cam = np.asarray(points_world, dtype=float) @ w2c[:3, :3].T + w2c[:3, 3]
    uvz = cam @ np.asarray(K, dtype=float).T
    z = uvz[:, 2]
    return uvz[:, :2] / np.where(np.abs(z[:, None]) < 1e-9, 1e-9, z[:, None]), z


def draw_obb_world(image: np.ndarray, corners_world: np.ndarray, K, cam_c2w,
                   color=OBB_COLOR, thickness: int = 1, draw_bbox: bool = True):
    """OBB 12 edge + 그 투영의 2D bbox. 카메라 뒤 꼭짓점이 낀 edge 는 건너뛴다."""
    from scene_graph.obb import OBB_EDGES

    uv, z = project_world(corners_world, K, cam_c2w)
    for a, b in OBB_EDGES:
        if z[a] <= 1e-6 or z[b] <= 1e-6:
            continue
        cv2.line(image, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)),
                 color, thickness, cv2.LINE_AA)
    if draw_bbox and (z > 1e-6).all():
        x0, y0 = np.round(uv.min(axis=0)).astype(int)
        x1, y1 = np.round(uv.max(axis=0)).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), BBOX_COLOR, 1, cv2.LINE_AA)
    return image


def draw_thirds(image: np.ndarray, safe_frac: float = 0.80, alpha: float = 0.30):
    """3x3 그리드 + 중앙 safe frame. 알파 블렌딩이라 아래 그림을 안 지운다."""
    height, width = image.shape[:2]
    layer = np.zeros_like(image)
    for k in (1, 2):
        cv2.line(layer, (width * k // 3, 0), (width * k // 3, height), GRID_COLOR, 1)
        cv2.line(layer, (0, height * k // 3), (width, height * k // 3), GRID_COLOR, 1)
    margin_x, margin_y = int(width * (1 - safe_frac) / 2), int(height * (1 - safe_frac) / 2)
    cv2.rectangle(layer, (margin_x, margin_y), (width - margin_x, height - margin_y), GRID_COLOR, 1)
    return cv2.addWeighted(image, 1.0, layer, alpha, 0.0)


def label_tile(image: np.ndarray, label: str, caption: str = ""):
    """좌상단 큰 라벨(검은 배경 위 흰 글씨) + 하단 한 줄 캡션."""
    out = np.ascontiguousarray(image)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(out, (0, 0), (tw + 14, th + 14), (0, 0, 0), -1)
    cv2.putText(out, label, (7, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if caption:
        height = out.shape[0]
        cv2.rectangle(out, (0, height - 20), (out.shape[1], height), (0, 0, 0), -1)
        cv2.putText(out, caption, (5, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return out


def contact_sheet(tiles: list, columns: int, gap: int = 4, background=(20, 20, 20)):
    """타일 리스트 → 격자 한 장. 빈 칸은 배경색으로 채워 격자가 안 무너지게 한다."""
    assert tiles, "붙일 타일이 없다"
    height, width = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.full((rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap, 3),
                    background, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        y, x = r * (height + gap), c * (width + gap)
        sheet[y:y + height, x:x + width] = tile
    return sheet


def board_labels(count: int, columns: int):
    """A1..A9 / B1.. — 행이 알파벳, 열이 숫자. 좌표처럼 안 읽히는 라벨이어야 한다."""
    return [f"{chr(ord('A') + i // columns)}{i % columns + 1}" for i in range(count)]


def source_panel(video: np.ndarray, frames: list, node, T_wg, K, cam_c2w,
                 tile_height: int, tile_width: int, columns: int = 2):
    """소스 프레임 2x2 + 같은 OBB/bbox 오버레이. 캡션은 영문 (VLM 에게 나간다)."""
    from scene_graph.lift import apply_transform
    from scene_graph.obb import node_obb_at, obb_corners

    tiles = []
    for f in frames:
        image = cv2.resize(np.ascontiguousarray(video[f]), (tile_width, tile_height))
        scaled_K = np.asarray(K[f], dtype=float).copy()
        scaled_K[0] *= tile_width / video.shape[2]
        scaled_K[1] *= tile_height / video.shape[1]
        corners = apply_transform(T_wg, obb_corners(*node_obb_at(node, f)))
        draw_obb_world(image, corners, scaled_K, cam_c2w[f])
        tiles.append(label_tile(image, f"t={f}"))
    return contact_sheet(tiles, columns)


def contract_text(graph: dict, node: dict, rows: list, source_camera: str, budget: dict):
    """§B3 텍스트 블록. **라인 지향, 결정론적 순서, raw dict f-string 금지.**

    LBM 원본(`director_shot_contract.py:741-751`)은 python dict 를 그대로 f-string 에 박아서
    프롬프트에 `{'a': 1, ...}` 이 나간다. 그러면 키 순서가 바뀔 때마다 모델 출력이 흔들리고
    diff 도 안 읽힌다. 여기서는 한 줄 = 한 사실로 고정한다.
    """
    lines = [
        "## SCENE",
        f"video {graph['video']} | frames {graph['num_frames']} | fps {graph['fps']:.1f} | "
        f"image {graph['width']}x{graph['height']} | unit u (1 u = scene scale)",
        f"parallax_ratio {graph['scale']['parallax_ratio']:.4f} "
        f"(how much the source camera actually moved; small = nearly one viewpoint)",
        "",
        "## SUBJECT",
        f"id {node['id']} | label {node['label']} | size_lwh_u "
        f"{','.join(f'{v:.2f}' for v in node['obb']['extent'])} | "
        f"moving {'yes' if node['moving'] else 'no'} | "
        f"speed {node['speed_u_per_frame']:.4f} u/frame | path_len {node['path_len_u']:.3f} u",
        f"observed_azimuth_span {node['obs_az_span_deg']:.1f} deg | "
        f"observed_faces {','.join(node['observed_faces']) or 'none'} | "
        f"reference_distance {node['viewing_distance']['d_ref']:.2f} u",
    ]
    neighbors = [e for e in graph["edges"] if node["id"] in (e["src"], e["dst"])]
    if neighbors or node.get("supported_by") or node.get("against_wall"):
        lines += ["", "## NEIGHBORS"]
        for edge in neighbors:
            other = edge["dst"] if edge["src"] == node["id"] else edge["src"]
            label = next((n["label"] for n in graph["nodes"] if n["id"] == other), other)
            line = f"{other} {label} | {edge['rel']} | dist {edge['dist_u']:.2f} u"
            #    temporal 엣지면 "언제" 가까웠는지까지 준다 — `dist_u` 만으론 스쳐 지나간 이웃과
            #    내내 붙어 있던 이웃이 구분되지 않고, 카메라가 그 순간을 잡아야 할 수도 있다.
            if "dist_u_t" in edge:
                spans = ", ".join(f"frames {s}-{t}" for s, t in edge["near_intervals"]) or "never"
                line += (f" (at reference frame) | closest {edge['dist_u_min']:.2f} u | "
                         f"near during {edge['near_frac'] * 100:.0f}% of frames [{spans}]")
            lines.append(line)
        if node.get("supported_by"):
            lines.append(f"supported_by {node['supported_by']}")
        if node.get("against_wall"):
            lines.append("against_wall yes")
    lines += ["", "## SOURCE_CAMERA", source_camera, "", "## CANDIDATES"]
    # 각도는 **소스 카메라 기준 상대값**으로 준다. subject OBB 의 yaw 는 180° 대칭이라 절대
    # 방위각은 "정면"이 어딘지 정하지 못한다 — VLM 에게 az -146 은 아무 의미가 없다.
    relative = all("d_azimuth_deg" in row for row in rows) if rows else False
    if relative:
        lines.append("(angles are RELATIVE to the source camera, orbiting around the subject: "
                     "d_az > 0 = counter-clockwise seen from above, d_elev > 0 = higher. "
                     "dist = radius / source radius.)")
    for row in rows:
        angles = (f"d_az {row['d_azimuth_deg']:+.0f}  d_elev {row['d_elevation_deg']:+.0f}"
                  if relative else
                  f"az {row['azimuth_deg']:+.0f}  elev {row['elevation_deg']:.0f}")
        lines.append(
            f"{row['board_label']}  {angles}  "
            f"dist {row['distance_ratio']:.1f}x  coverage {row['coverage']:.2f}  "
            f"subject_area {row['subject_area']:.2f}  occlusion_pass {row['occlusion_pass']:.2f}  "
            f"tau {row['tau']:.2f}")
    lines += ["", "## BUDGET",
              f"max_view_angle {budget['max_view_angle_deg']:.0f} deg | "
              f"max_tau {budget['max_tau']:.2f} | min_coverage {budget['min_coverage']:.2f}",
              "(every candidate shown already satisfies these; rejected ones are not shown)"]
    return "\n".join(lines)
