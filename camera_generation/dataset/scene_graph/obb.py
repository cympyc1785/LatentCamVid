"""중력 정렬 OBB. G 좌표에서 up=e_z 이므로 xy 평면에 눌러 `cv2.minAreaRect` 를 쓴다.

왜 축을 하나 고정하나: 자유 3D OBB(PCA)는 얇고 긴 물체에서 축이 휙휙 뒤집힌다. 카메라 배치에
필요한 건 "물체가 어느 쪽을 향하나(yaw)" 와 "얼마나 큰가" 뿐이고, 그 둘은 중력축 고정으로
충분하다. 게다가 orbit 은 어차피 중력축 주위를 돈다.

**저시차 깊이축 부풀림 보정은 기본 off 다** (`--deinflate` 로만 켠다). 원래 의도는 "시차 0.005
인 씬에서 시선 방향 두께는 depth 노이즈 폭이지 물체 두께가 아니다"였는데, camel dyn_1 에서
실측해 보니 반대였다: 마스크 경계 누출을 `trim_depth_tail` 로 먼저 잡고 `fit_obb` 의 yaw 규약을
고치자 투영 bbox 비가 1.05 로 떨어졌다 — **깊이축이 부푼 게 아니라 낙타가 실제로 카메라를
정면으로 보고 서 있었다**. 그 상태에서 이 보정을 걸면 l,w,h 가 0.16,0.05,0.13 → 0.05,0.05,0.13
이 되어 몸통 길이를 통째로 날린다. 부풀림 여부는 이제 `schema.check_reprojection` 의
`size_ratio`(투영 OBB bbox / 마스크 bbox)가 판정한다 — 그게 직접 측정치고 이건 추측이다.
켜더라도 `extent_inflated=True` 만 남을 뿐 **고친 게 아니라 표시한 것**이다.
"""
import cv2
import numpy as np


def fit_obb(points_g: np.ndarray):
    """G 점군 → dict(center(3,), extent(3,)=[l,w,h], yaw_rad, R(3,3)).

    l = yaw 축(로컬 +x) 길이, w = 로컬 +y 길이, h = 중력축 길이.
    """
    assert len(points_g) >= 3, "OBB 를 세울 점이 3개 미만"
    xy = np.ascontiguousarray(points_g[:, :2], dtype=np.float32)
    rect = cv2.minAreaRect(xy)
    (cx, cy) = rect[0]
    # `rect[1]` 의 (width, height) 가 `rect[2]` 각도의 어느 축에 붙는지는 OpenCV 버전마다 다르다.
    # 실제로 그 규약을 짐작했다가 yaw 가 정확히 90° 틀렸고, OBB 장축이 시선과 수직으로 놓여
    # 투영 bbox 가 마스크의 2.95배가 됐다. 그래서 **꼭짓점에서 직접** 변을 재 규약을 안 쓴다.
    box = cv2.boxPoints(rect).astype(np.float64)          # (4,2), 인접 순서 보장
    edge_a, edge_b = box[1] - box[0], box[2] - box[1]
    len_a, len_b = np.linalg.norm(edge_a), np.linalg.norm(edge_b)
    long_edge, w0, w1 = (edge_a, len_a, len_b) if len_a >= len_b else (edge_b, len_b, len_a)
    yaw = float(np.arctan2(long_edge[1], long_edge[0]))
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2           # (-90°, 90°] — 박스는 180° 대칭

    z_lo, z_hi = float(points_g[:, 2].min()), float(points_g[:, 2].max())
    cos, sin = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return {"center": np.array([cx, cy, (z_lo + z_hi) / 2.0]),
            "extent": np.array([float(w0), float(w1), z_hi - z_lo]),
            "yaw_rad": float(yaw), "R": R, "z_lo": z_lo, "z_hi": z_hi}


def deinflate_depth_axis(obb: dict, view_dir_g: np.ndarray, parallax: float,
                         enabled: bool = False, parallax_threshold: float = 0.05,
                         inflate_ratio: float = 2.5):
    """시선 정렬 수평축이 과하게 길면 잘라내고 플래그를 세운다. **기본 off** (모듈 docstring 참고)."""
    if not enabled or parallax >= parallax_threshold:
        return obb, False
    view = view_dir_g.copy()
    view[2] = 0.0
    if np.linalg.norm(view) < 1e-9:
        return obb, False
    view /= np.linalg.norm(view)
    align = np.abs(obb["R"][:, :2].T @ view)          # 로컬 x,y 축이 시선과 얼마나 정렬됐나
    axis = int(np.argmax(align))
    other = 1 - axis
    if obb["extent"][axis] > inflate_ratio * obb["extent"][other] and align[axis] > 0.7:
        obb = {**obb, "extent": obb["extent"].copy()}
        obb["extent"][axis] = obb["extent"][other]
        return obb, True
    return obb, False


def unwrap_yaw(yaws: np.ndarray):
    """박스 yaw 는 180° 주기라 그냥 unwrap 하면 안 된다 — 주기 π 로 편다."""
    return np.unwrap(np.asarray(yaws) * 2.0) / 2.0


def obb_corners(center: np.ndarray, extent: np.ndarray, R: np.ndarray):
    """(8,3) G 좌표 꼭짓점. 순서는 아래 `OBB_EDGES` 와 짝."""
    signs = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)],
                     dtype=np.float64)
    return center + (signs * extent / 2.0) @ R.T


OBB_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0),      # 아랫면
             (4, 5), (5, 7), (7, 6), (6, 4),      # 윗면
             (0, 4), (1, 5), (2, 6), (3, 7)]      # 기둥


def yaw_to_R(yaw: float):
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def node_obb_at(node: dict, frame: int):
    """프레임 `frame` 에서의 (center, extent, R). **동적 노드는 반드시 이걸 써야 한다** —

    `node["obb"]` 는 ref 프레임(첫 관측) 값이라 움직이는 물체를 다른 프레임 카메라로 투영하면
    엉뚱한 데 떨어진다. extent 만 시간 불변으로 본다.
    """
    center = np.asarray(node["track"]["center_smooth"], dtype=float)[frame]
    yaw = float(np.asarray(node["track"]["yaw"], dtype=float)[frame])
    return center, np.asarray(node["obb"]["extent"], dtype=float), yaw_to_R(yaw)


def observation_azimuths(centers_g: np.ndarray, cam_centers_g: np.ndarray, yaws: np.ndarray):
    """프레임별 "카메라가 물체를 어느 방위에서 봤나"(로컬 yaw 기준, 라디안)."""
    delta = cam_centers_g - centers_g
    world_az = np.arctan2(delta[:, 1], delta[:, 0])
    return (world_az - yaws + np.pi) % (2 * np.pi) - np.pi


def azimuth_span(azimuths: np.ndarray):
    """관측 방위각이 덮은 범위(도). 최대 공백의 여집합 — 원형이라 max-min 이 아니다."""
    if len(azimuths) == 0:
        return 0.0
    sorted_az = np.sort(np.asarray(azimuths))
    gaps = np.diff(np.concatenate([sorted_az, sorted_az[:1] + 2 * np.pi]))
    return float(np.degrees(2 * np.pi - gaps.max()))


def observed_faces(azimuths: np.ndarray, tolerance_deg: float = 60.0):
    """front/back/left/right 중 실제로 본 면. 로컬 +x = front."""
    names = {"front": 0.0, "left": 90.0, "back": 180.0, "right": -90.0}
    degrees = np.degrees(azimuths)
    seen = []
    for name, center in names.items():
        delta = (degrees - center + 180) % 360 - 180
        if np.any(np.abs(delta) <= tolerance_deg):
            seen.append(name)
    return seen
