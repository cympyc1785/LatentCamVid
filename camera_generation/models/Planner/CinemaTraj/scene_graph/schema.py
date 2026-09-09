"""`planner_scene_graph_v1` 스키마 · 불변조건 assert · save/load.

**여기 있는 assert 가 좌표 규약 버그를 잡는 유일한 그물이다.** 특히 `check_reprojection` —
OBB center 를 그 물체가 가장 잘 보이는 프레임의 카메라로 되쏘아 2D 마스크 bbox 안에 떨어지는지
본다. w2c↔c2w 뒤바뀜 · up 부호 · S 나눗셈 빠짐 · z_depth↔ray_depth 를 **동시에** 검사한다.
개별 단위 테스트로는 이 넷을 다 못 잡는다 (두 개가 서로 상쇄되는 조합이 있다).

CinemaTraj 계획의 `free_space` / `sdf.npz` 는 없다 — 렌더 게이트(G1)가 대체했다.
"""
import json
import re
from os import makedirs, path

import numpy as np

GRAPH_FORMAT = "planner_scene_graph_v1"


def _to_list(value):
    if isinstance(value, np.ndarray):
        return [_to_list(v) for v in value] if value.ndim else float(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_list(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_list(v) for v in value]
    return value


def check_reprojection(nodes: list, T_wg: np.ndarray, K: np.ndarray, cam_c2w: np.ndarray,
                       tolerance_frac: float = 0.2, max_size_ratio: float = 1.8):
    """노드별 재투영 검사. 실패는 반환만 하고 죽이지 않는다 — 표로 보여줄 것.

    **두 가지를 잰다.** ① center 재투영 오차 — 규약(w2c/c2w, up 부호, S, z vs ray)을 잡는다.
    ② OBB 12 꼭짓점을 투영해 만든 2D bbox 가 마스크 bbox 대비 몇 배인가 — **extent 를 잡는다**.
    ①만으로는 크기가 3배 부푼 박스가 조용히 통과한다 (중심은 맞으니까). 실제로 처음 돌렸을 때
    ①은 전부 통과인데 ②가 4.3배였다 (마스크 경계 depth 누출이 깊이축을 늘림).

    동적 노드는 `node_obb_at` 으로 **그 프레임의** center/yaw 를 써야 한다. ref 프레임 박스를
    다른 프레임 카메라로 쏘면 물체가 움직인 만큼 어긋난다.
    """
    from .lift import apply_transform, project_points
    from .obb import node_obb_at, obb_corners

    report = []
    for node in nodes:
        frame = int(node["best_frame"])
        center, extent, R = node_obb_at(node, frame)
        uv, z = project_points(apply_transform(T_wg, center[None]), K[frame], cam_c2w[frame])
        x0, y0, x1, y1 = node["bbox_xyxy_best"]
        pad_x, pad_y = tolerance_frac * (x1 - x0), tolerance_frac * (y1 - y0)
        inside = bool(z[0] > 0 and x0 - pad_x <= uv[0, 0] <= x1 + pad_x
                      and y0 - pad_y <= uv[0, 1] <= y1 + pad_y)
        center_2d = np.array([(x0 + x1) / 2, (y0 + y1) / 2])

        corners_uv, corners_z = project_points(
            apply_transform(T_wg, obb_corners(center, extent, R)), K[frame], cam_c2w[frame])
        front = corners_z > 1e-6
        if front.all():
            projected = corners_uv.max(axis=0) - corners_uv.min(axis=0)
            ratio = float(np.max(projected / np.maximum([x1 - x0, y1 - y0], 1e-6)))
        else:
            projected, ratio = np.array([np.nan, np.nan]), float("nan")

        report.append({"id": node["id"], "frame": frame,
                       "uv": [round(float(uv[0, 0]), 1), round(float(uv[0, 1]), 1)],
                       "bbox_center": [round(float(center_2d[0]), 1), round(float(center_2d[1]), 1)],
                       "reproj_px": round(float(np.linalg.norm(uv[0] - center_2d)), 1),
                       "z_cam": round(float(z[0]), 4), "inside_bbox": inside,
                       "obb_bbox_wh": [round(float(v), 1) for v in projected],
                       "mask_bbox_wh": [round(x1 - x0, 1), round(y1 - y0, 1)],
                       "size_ratio": round(ratio, 2),
                       "size_ok": bool(ratio == ratio and ratio <= max_size_ratio)})
    return report


def assert_invariants(graph: dict):
    """저장 직전 마지막 방어선."""
    assert graph["format"] == GRAPH_FORMAT
    for node in graph["nodes"]:
        extent = np.asarray(node["obb"]["extent"], dtype=float)
        assert (extent > 0).all(), f"{node['id']}: extent 에 0 이 있다 {extent}"
        R = np.asarray(node["obb"]["R"], dtype=float)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6, f"{node['id']}: det(R) != 1"
    up = np.asarray(graph["gravity"]["up_world"], dtype=float)
    assert abs(np.linalg.norm(up) - 1.0) < 1e-6, "up_world 가 단위벡터가 아니다"
    T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
    assert np.abs(T_gw @ T_wg - np.eye(4)).max() < 1e-6, "T_gw / T_wg 왕복이 안 맞는다"
    ids = [node["id"] for node in graph["nodes"]]
    assert len(ids) == len(set(ids)), f"노드 id 중복: {ids}"
    for edge in graph["edges"]:
        assert edge["src"] in ids and edge["dst"] in ids, f"댕글링 엣지: {edge}"
        #    temporal 축 길이가 어긋나면 하류가 조용히 엉뚱한 프레임의 거리를 읽는다
        #    (짧아도 IndexError 가 아니라 잘린 채로 돈다).
        if "dist_u_t" in edge:
            assert len(edge["dist_u_t"]) == graph["num_frames"], \
                f"dist_u_t 길이 != num_frames: {edge['src']}-{edge['dst']}"
    for node in graph["nodes"]:
        for key in ("supported_by_t", "against_wall_t"):
            if key in node:
                assert len(node[key]) == graph["num_frames"], \
                    f"{node['id']}.{key} 길이 != num_frames"


def save_graph(output_path: str, graph: dict):
    assert_invariants(graph)
    makedirs(path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_to_list(graph), file, ensure_ascii=False, indent=1)


def load_graph(input_path: str):
    with open(input_path, encoding="utf-8") as file:
        graph = json.load(file)
    assert graph["format"] == GRAPH_FORMAT, f"알 수 없는 graph format: {graph['format']}"
    return graph


# ── anchor(=target) 선별 (D127) ──────────────────────────────────────────────
# 두 소비자(`scripts/sample_camera_bank.py`, `scripts/route_presets.py`)가 같은 규칙을 써야
# 라우팅 표와 실제 뱅크가 안 어긋난다. 그래서 여기 한 벌만 둔다.

# **target 이 될 수 없는 라벨** — 벽/바닥/천장/울타리/난간/문/창 같은 "씬의 껍데기".
# 왜 빼는가 (사용자 지시): 이것들은 촬영의 **대상**이 아니라 **배경**이다. OBB 를 씌우면 씬을
# 가로지르는 판이 되고, 그 판을 anchor 로 잡으면 카메라가 벽면을 중심으로 orbit 을 돈다.
# `extract_static_nouns.SURFACE_NOUNS` 는 SAM3 **keyword** 만 걸러서 두 군데가 샜다:
#   ① `fence`/`railing`/`window`/`door` 가 그 목록에 없다 (Vista 53편 anchor 357개 중 32개)
#   ② TRUMANS 그래프는 mesh object index 이름을 그대로 쓴다 — SAM3 를 안 탄다
#      (191편 anchor 2,087개 중 1,113개 = 53% 가 `WallInner.022` / `Floor.002` / `door_door_02`
#      / `BayWindow.017` 류)
SURFACE_TOKENS = frozenset({
    "wall", "floor", "ceiling", "ground", "sky", "roof", "road", "street", "pavement",
    "sidewalk", "terrain", "court", "lawn", "grass", "carpet", "rug", "backdrop", "background",
    "window", "windowsill", "door", "doorway", "gate", "fence", "railing", "banister",
    "handrail", "curb", "stair", "staircase", "step", "pillar", "column", "beam", "partition",
    "baseboard", "skirting", "wallpaper", "panel", "curtain", "blind",
})


def label_tokens(label: str):
    """`WallInner.022` -> {wall, inner}, `door_door_02` -> {door}, `BayWindow.017` -> {bay, window}.

    camelCase 를 쪼개는 게 핵심이다 — TRUMANS 라벨은 mesh object 이름이라 `WallInner` 처럼
    붙어 있고, 소문자 변환만 하면 `wallinner` 라 어떤 토큰에도 안 걸린다.
    """
    return {t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+", label)}


def is_surface_node(node: dict):
    """라벨 또는 alias 중 하나라도 `SURFACE_TOKENS` 에 걸리면 True."""
    names = [str(node.get("label", ""))] + [str(a) for a in (node.get("aliases") or [])]
    return any(label_tokens(n) & SURFACE_TOKENS for n in names)


def anchor_sort_key(node: dict):
    """움직이는 것 우선 -> 화면을 크게 차지하는 것 -> 오래 보이는 것 (내림차순으로 쓴다)."""
    return (bool(node.get("moving")), float(node["max_area_frac"]),
            int(node.get("num_visible_frames", 0)))


def pick_main_anchors(nodes: list, min_area_frac: float = 0.01, max_dynamic: int = 3,
                      max_static: int = 3, drop_surfaces: bool = True, max_anchors: int = 0):
    """-> (anchors, dropped) — `dropped` 는 `(id, label, 사유)` 목록이라 표로 찍을 수 있다.

    **상한을 동적/정적 따로 센다** (D127b, 사용자 지시 "dynamic target, static target 각각 최대
    3개"). 합산 상한 하나로 두면(D127 최초안) 동적 노드가 정렬 1순위라 동적이 3개 있는 편에서
    정적이 **0개**가 되고, 반대로 동적이 없는 편은 정적만 3개가 된다 — 즉 코퍼스 전체에서
    "동적 subject 촬영"과 "정적 물체 촬영"의 비율이 씬 구성에 끌려다닌다. 따로 세면 편당
    상한이 3+3=6 이 되지만 실측 편당은 Vista 3.9 · dynpose 4.6 · TRUMANS 4.0 이다 (동적이 3개
    넘는 편이 드물어서).

    왜 상한이 필요한가: anchor 하나가 곧 뱅크 변이 수의 배수라 편당 anchor 가 많으면 굽는 시간이
    그만큼 늘고, 정작 늘어난 몫은 **작고 안 움직이는 배경 물체**다 (변경 전 편당 anchor
    Vista 6.7 · dynpose 4.6 · TRUMANS 10.9).

    `max_anchors` 는 두 상한을 적용한 **뒤에** 거는 총합 상한이다. 기본 `0` = 끔. 셋 다 `0` 이면
    D127 이전 동작(하한만 통과하면 전량)이 된다.
    """
    kept, dropped = [], []
    for node in nodes:
        if float(node["max_area_frac"]) < min_area_frac:
            dropped.append((node["id"], node.get("label", ""), "min_area_frac"))
        elif min(node["obb"]["extent"]) <= 0:
            dropped.append((node["id"], node.get("label", ""), "degenerate_obb"))
        elif drop_surfaces and is_surface_node(node):
            dropped.append((node["id"], node.get("label", ""), "surface"))
        else:
            kept.append(node)
    # 표면만 남은 씬에서 anchor 를 0 개로 만들면 그 영상이 통째로 빠진다. 그것보다는 제일 큰
    # 표면 하나를 되살리고 사유를 남기는 쪽이 낫다 (`dropped` 에 `surface_restored` 로 표시).
    if not kept and drop_surfaces:
        pool = [n for n in nodes
                if float(n["max_area_frac"]) >= min_area_frac and min(n["obb"]["extent"]) > 0]
        if pool:
            best = max(pool, key=anchor_sort_key)
            kept = [best]
            dropped = [d for d in dropped if d[0] != best["id"]]
            dropped.append((best["id"], best.get("label", ""), "surface_restored"))
    kept.sort(key=anchor_sort_key, reverse=True)

    # 동적/정적을 따로 자른다. `kept` 가 이미 (moving, area, frames) 내림차순이라 각 갈래에서
    # 앞에서부터 세면 그 갈래의 "가장 main 한" 것들이다.
    survivors, counts = [], {True: 0, False: 0}
    for node in kept:
        moving = bool(node.get("moving"))
        limit = max_dynamic if moving else max_static
        if limit > 0 and counts[moving] >= limit:
            dropped.append((node["id"], node.get("label", ""),
                            "max_dynamic" if moving else "max_static"))
            continue
        counts[moving] += 1
        survivors.append(node)
    if max_anchors > 0 and len(survivors) > max_anchors:
        for node in survivors[max_anchors:]:
            dropped.append((node["id"], node.get("label", ""), "max_anchors"))
        survivors = survivors[:max_anchors]
    return survivors, dropped
