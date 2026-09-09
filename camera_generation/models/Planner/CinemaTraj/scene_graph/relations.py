"""노드 사이 관계 엣지. **용도는 자유공간 추론이 아니라 VLM 이 물체를 말로 지칭하는 것이다.**

폐기된 CinemaTraj 계획에서는 관계가 카메라 배치 제약(벽 뒤로 못 감 등)을 담당했다. 지금은
그 역할을 렌더 게이트 G1(behind-surface)이 가져갔으므로, 관계는 프롬프트 텍스트에만 쓰인다:
`"the camel standing on the ground, near the fence"`. 그래서 정밀도보다 **틀린 관계를 안 내는
것**이 중요하다 — 임계를 보수적으로 잡고 애매하면 엣지를 안 만든다.

전부 기하 규칙이지 학습된 predicate 이 아니다.

`temporal=True` (D70) 면 **엣지도 시간을 갖는다**. 노드는 원래부터 동적이었지만
(`track.center_smooth` (F,3) + `obb.node_obb_at`) 엣지는 `dist_u` 스칼라 하나뿐이었고, 그 스칼라는
각 노드의 **ref 프레임**(첫 관측) 중심으로 쟀다. 구멍이 두 개다:
  ① 두 노드의 ref 프레임이 서로 다를 수 있다 — 다른 시각의 두 위치 사이 거리를 잰다.
  ② **도중에 가까워지는 쌍이 엣지로 안 올라온다.** jogging-woman 실측: 사람이 쓰레기통 옆을
     지나가는데 ref 프레임에선 멀어서 `edges` 가 **0개**로 나왔다.
그래서 temporal 은 `dist_u_t` (F,) 를 싣고, 한 프레임이라도 가까운 쌍을 전부 엣지로 올린다.
기본값은 off = 기존 동작 그대로 (`dist_u` 의 의미도 안 바꾼다 — 하류 `lbm/overlay.py` 가 읽는다).
"""
import numpy as np

from .gravity import fit_plane_ransac


def ground_height(points_g: np.ndarray, quantile: float = 0.02):
    """G 의 지면 높이(z). RANSAC 이 실패했어도 정적 점의 하위 분위수는 대체로 바닥이다."""
    return float(np.quantile(points_g[:, 2], quantile))


def find_wall_planes(points_g: np.ndarray, max_walls: int = 3, dist_threshold_u: float = 0.01,
                     min_inlier_ratio: float = 0.05, max_up_dot: float = 0.3, seed: int = 0):
    """중력축과 거의 수직인 큰 평면 = 벽 후보. [(normal(3,), d)]."""
    walls, remaining = [], points_g
    for peel in range(max_walls * 2):
        if len(remaining) < 1000 or len(walls) >= max_walls:
            break
        normal, d, inliers = fit_plane_ransac(remaining, dist_threshold_u, seed=seed + peel)
        if normal is None:
            break
        if inliers.mean() >= min_inlier_ratio and abs(normal[2]) < max_up_dot:
            walls.append((normal, d))
        remaining = remaining[~inliers]
    return walls


def pair_distance_over_time(a: dict, b: dict, num_frames: int):
    """(dist_u_t (F,), both_observed (F,)) — 프레임별 수평 거리와 "둘 다 관측된" 마스크.

    `center_smooth` 는 미관측 구간까지 채워져 있어(`build_scene_graph.smooth_centers`) 거리 자체는
    전 프레임 나오지만 관측 밖 값은 외삽이다. **판정은 마스크 안에서만** 한다 — 안 그러면
    한 노드가 사라진 뒤의 외삽 위치로 "가까워졌다"를 만들어낸다.
    """
    ca = np.asarray(a["track"]["center_smooth"], dtype=float)[:num_frames]
    cb = np.asarray(b["track"]["center_smooth"], dtype=float)[:num_frames]
    dist = np.linalg.norm(ca[:, :2] - cb[:, :2], axis=-1)
    observed = np.zeros(num_frames, dtype=bool)
    shared = sorted(set(a["track"]["frames"]) & set(b["track"]["frames"]))
    if shared:
        observed[shared] = True
    return dist, observed


def mask_intervals(mask: np.ndarray):
    """bool (F,) → 연속 True 구간 `[[start, end], ...]` (양 끝 포함)."""
    intervals, start = [], None
    for frame, on in enumerate(np.asarray(mask, dtype=bool)):
        if on and start is None:
            start = frame
        elif not on and start is not None:
            intervals.append([start, frame - 1])
            start = None
    if start is not None:
        intervals.append([start, len(mask) - 1])
    return intervals


def support_of(node: dict, z_lo: float, center_xy: np.ndarray, nodes: list,
               tops: list, centers_xy: list, contact_u: float, ground_z: float):
    """①② 지면/윗면 접촉 판정을 한 군데로 모은 것. `tops[i]`/`centers_xy[i]` 는 **그 시점** 값.

    ref 프레임과 temporal 이 같은 규칙을 쓰게 하려고 뽑았다. 갈라지면 "ref 에선 ground 인데
    프레임별로는 아니다" 같은 설명 불가능한 결과가 나온다. ref 경로는 저장된 `z_lo`/`z_hi` 를
    그대로 넘기므로 산술이 예전과 완전히 같다.
    """
    if abs(z_lo - ground_z) <= contact_u:
        return "ground"
    extent = np.asarray(node["obb"]["extent"], dtype=float)
    footprint = float(np.prod(extent[:2]))
    for other, top, other_xy in zip(nodes, tops, centers_xy):
        if other is node or abs(z_lo - top) > contact_u:
            continue
        other_extent = np.asarray(other["obb"]["extent"], dtype=float)
        if float(np.prod(other_extent[:2])) < footprint:
            continue
        gap = float(np.linalg.norm(center_xy - other_xy))
        if gap < 0.5 * (max(extent[:2]) + max(other_extent[:2])):
            return other["id"]
    return None


def build_relations(nodes: list, static_points_g: np.ndarray | None,
                    contact_u: float = 0.05, wall_u: float = 0.08, near_factor: float = 1.5,
                    wall_contact_frac: float = 0.15,
                    temporal: bool = False, num_frames: int | None = None,
                    ground_z_override: float | None = None):
    """노드 리스트 → 엣지 리스트. 노드에는 `supported_by` / `against_wall` 를 직접 채운다.

    `wall_contact_frac`: 벽 접촉 임계를 **물체 크기에 비례**하게 만든다 (§③). `1e9` 를 주면
    `wall_u` 절대 임계만 쓰던 예전 동작으로 돌아간다.
    `temporal`: §⑤ 참조. 기본 off = 기존 동작 비트 동일.
    `ground_z_override`: GT 지면(`scene_graph.gt_trumans`)이 있으면 점군 2% 분위수 대신 그걸
    쓴다. 분위수는 바닥이 화면에 조금만 나오거나 사람이 가구 위에 있으면 지면을 올려 잡는다.
    None = 기존 동작 비트 동일.
    """
    edges = []
    if ground_z_override is not None:
        ground_z = float(ground_z_override)
    elif static_points_g is not None and len(static_points_g):
        ground_z = ground_height(static_points_g)
    else:
        ground_z = min((node["obb"]["z_lo"] for node in nodes), default=0.0)
    walls = find_wall_planes(static_points_g) if static_points_g is not None and len(static_points_g)\
        else []

    # ①② 는 `support_of` 가 판정한다. `tops`/`centers_xy` 에 **저장된 ref 프레임 값**을 그대로
    # 넘기므로 (`z_hi` 는 deinflate 후 값일 수 있어 재계산하면 안 된다) 결과가 예전과 같다.
    ref_tops = [float(node["obb"]["z_hi"]) for node in nodes]
    ref_xy = [np.asarray(node["obb"]["center"], dtype=float)[:2] for node in nodes]

    for node in nodes:
        obb = node["obb"]
        node["supported_by"], node["against_wall"] = None, False

        # ① 지면 접촉  ② 다른 노드 윗면 접촉 — 수평 겹침(공중 부양 오인 방지) + **받침이 더
        # 커야 한다**. 크기 조건이 없으면 avocado-slice 에서 `person supported_by avocado` 가
        # 나왔다: 사람이 상반신만 보여 z_lo 가 조리대 높이라 아보카도 윗면과 0.05u 안에 들어온다.
        # 기하로는 참인데 말로는 거짓이고, 이 엣지는 VLM 프롬프트로 그대로 나간다.
        node["supported_by"] = support_of(node, float(obb["z_lo"]),
                                          np.asarray(obb["center"], dtype=float)[:2],
                                          nodes, ref_tops, ref_xy, contact_u, ground_z)

        # ③ 벽 접촉 — OBB 꼭짓점 중 하나라도 벽면 근처. 임계는 **물체 크기에 비례**한다.
        # 절대 임계 하나(예전의 wall_u=0.08)만 쓰면 camel 씬에서 6개 노드가 전부 True 가 됐다:
        # 실측 최근접 꼭짓점 거리가 fence 0.0010 / roof 0.0175 / tree 0.0063 인데 **낙타 본체도**
        # 0.0436, 0.0569 다. 낙타는 울타리 **앞에 서 있는** 것이지 붙어 있는 게 아니고, 전부
        # True 인 플래그는 VLM 프롬프트에 정보를 0 비트 넣는다. 크기 비례로 바꾸면
        # 낙타 임계 0.021/0.024 (< 실측) 로 떨어져 나가고 정적 4개만 남는다.
        from .obb import obb_corners
        corners = obb_corners(obb["center"], obb["extent"], obb["R"])
        threshold = min(wall_u, wall_contact_frac * float(max(obb["extent"][:2])))
        for normal, d in walls:
            if np.abs(corners @ normal + d).min() <= threshold:
                node["against_wall"] = True
                break

    # ④ near — 수평 거리가 두 반경 합의 near_factor 배 이내
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            dist = float(np.linalg.norm(a["obb"]["center"][:2] - b["obb"]["center"][:2]))
            radius = 0.5 * (max(a["obb"]["extent"][:2]) + max(b["obb"]["extent"][:2]))
            if dist <= near_factor * radius:
                edges.append({"src": a["id"], "dst": b["id"], "rel": "near",
                              "dist_u": round(dist, 4), "near_at_ref": True})

    diagnostics = {"ground_z": round(float(ground_z), 4), "num_wall_planes": len(walls)}
    if not temporal:
        for edge in edges:                      # 기존 스키마 그대로 — `near_at_ref` 도 안 남긴다
            edge.pop("near_at_ref", None)
        return edges, diagnostics

    return _add_temporal(nodes, edges, walls, ground_z, contact_u, near_factor,
                         wall_contact_frac, wall_u, num_frames, diagnostics)


def _add_temporal(nodes: list, edges: list, walls: list, ground_z: float, contact_u: float,
                  near_factor: float, wall_contact_frac: float, wall_u: float,
                  num_frames: int | None, diagnostics: dict):
    """⑤ 엣지에 `dist_u_t` 를 싣고, 노드에 프레임별 `supported_by_t`/`against_wall_t` 를 채운다.

    ref 프레임엔 안 붙어 있었지만 **도중에 가까워지는 쌍도 엣지로 올린다** — 이게 시간 불변
    엣지의 실제 구멍이다. `dist_u` 의 의미는 안 바꾼다(ref 프레임 거리). 하류가 이미 읽고 있고,
    "가장 가까웠던 거리"는 별도 필드 `dist_u_min` 이 나른다.
    """
    from .obb import node_obb_at, obb_corners

    assert num_frames, "temporal=True 면 num_frames 를 넘겨야 한다"
    by_pair = {(edge["src"], edge["dst"]): edge for edge in edges}

    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            dist_t, observed = pair_distance_over_time(a, b, num_frames)
            radius = 0.5 * (max(a["obb"]["extent"][:2]) + max(b["obb"]["extent"][:2]))
            near_t = (dist_t <= near_factor * radius) & observed
            edge = by_pair.get((a["id"], b["id"]))
            if edge is None:
                if not near_t.any():
                    continue
                #    ref 프레임 거리로 `dist_u` 를 채운다 — 기존 엣지와 **같은 양**이어야 한다.
                ref_dist = float(np.linalg.norm(np.asarray(a["obb"]["center"], dtype=float)[:2]
                                                - np.asarray(b["obb"]["center"], dtype=float)[:2]))
                edge = {"src": a["id"], "dst": b["id"], "rel": "near",
                        "dist_u": round(ref_dist, 4), "near_at_ref": False}
                edges.append(edge)
            seen = int(observed.sum())
            edge.update({
                "dist_u_t": [round(float(v), 4) for v in dist_t],
                "dist_u_min": round(float(dist_t[observed].min()), 4) if seen else None,
                "dist_u_max": round(float(dist_t[observed].max()), 4) if seen else None,
                "near_frac": round(float(near_t.sum()) / seen, 3) if seen else 0.0,
                "near_intervals": mask_intervals(near_t),
                #    `dist_u_t` 는 전 프레임 채워지지만 이 구간 **밖은 외삽**이다. 구간을 같이
                #    싣지 않으면 하류가 그걸 구분할 방법이 없다 (`observed_frames` 는 개수뿐).
                "observed_intervals": mask_intervals(observed),
                "observed_frames": seen,
                "near_radius_u": round(float(near_factor * radius), 4),
            })

    #    프레임별 지지/벽 접촉. 노드 중심이 프레임마다 움직이므로 ①②③ 를 매 프레임 다시 푼다.
    centers = [np.asarray(node["track"]["center_smooth"], dtype=float)[:num_frames]
               for node in nodes]
    heights = [0.5 * float(np.asarray(node["obb"]["extent"], dtype=float)[2]) for node in nodes]
    for index, node in enumerate(nodes):
        support_t, wall_t = [], []
        for frame in range(num_frames):
            tops = [centers[j][frame][2] + heights[j] for j in range(len(nodes))]
            xys = [centers[j][frame][:2] for j in range(len(nodes))]
            support_t.append(support_of(node, float(centers[index][frame][2] - heights[index]),
                                        centers[index][frame][:2], nodes, tops, xys,
                                        contact_u, ground_z))
            center, extent, rotation = node_obb_at(node, frame)
            corners = obb_corners(center, extent, rotation)
            threshold = min(wall_u, wall_contact_frac * float(max(extent[:2])))
            wall_t.append(any(np.abs(corners @ normal + d).min() <= threshold
                              for normal, d in walls))
        node["supported_by_t"] = support_t
        node["against_wall_t"] = wall_t

    diagnostics["temporal"] = True
    diagnostics["num_transient_edges"] = sum(1 for e in edges if not e.get("near_at_ref"))
    return edges, diagnostics
