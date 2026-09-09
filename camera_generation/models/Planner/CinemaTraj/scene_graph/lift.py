"""depth + mask → world 3D 점. 그리고 world → graph frame G 변환.

왜 gsplat `depth_to_points` 를 안 쓰는가 (계획 대비 변경): 그 함수는 CUDA 확장이 빌드된 패키지
안에 있어서 `importlib.util.spec_from_file_location` 로 파일만 떼어와야 하고, 우리가 필요한 건
`p_cam = z · K⁻¹[u+0.5, v+0.5, 1]` 다섯 줄이다. 떼어오기 트릭은 gsplat 이 업데이트되면 조용히
깨지므로 여기서 직접 쓴다. **z_depth 규약**(ray length 아님)은 `render_frame` 과 같다.

edge-bleed 필터가 있는 이유: depth 불연속(물체 실루엣) 픽셀은 앞뒤 어느 표면에도 안 속하는
중간값을 갖는다. 그대로 lift 하면 물체에서 카메라 쪽으로 뻗는 **꼬리**가 생기고, OBB 가 그 꼬리를
감싸느라 깊이 방향으로 2~3배 부푼다. `|∇log z| > 0.05` 를 2px dilate 해서 잘라낸다 (log 를 쓰는
이유: 임계가 절대 depth 에 안 휘둘리게).
"""
import cv2
import numpy as np


def edge_bleed_mask(depth: np.ndarray, threshold: float = 0.05, dilate_px: int = 2):
    """depth 불연속 근처 픽셀 = True (버릴 곳)."""
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float64)
    log_z = np.log(np.clip(z, 1e-6, None))
    gy, gx = np.gradient(np.nan_to_num(log_z, nan=0.0))
    edge = (np.hypot(gx, gy) > threshold) | ~np.isfinite(z)
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), kernel, iterations=1) > 0
    return edge


def valid_pixels(depth: np.ndarray, sky: np.ndarray, conf: np.ndarray | None = None,
                 conf_threshold: float = 0.5, edge_threshold: float = 0.05, dilate_px: int = 2,
                 conf_mode: str = "absolute"):
    """finite ∧ z>0 ∧ ~sky ∧ ~edge-bleed [∧ conf 임계].

    `conf_mode` (D169) — **DA3 conf 는 [0,1] 이 아니다.** DPT 헤드가 `conf_activation="expp1"`
    이라 `exp(x)+1 ∈ [1, inf)` 다 (`depth_anything_3/model/dpt.py:49,296`). 그래서 옛 기본값인
    `conf > 0.5` 는 **전 픽셀을 통과시키는 no-op** 이다. 지금까지는 conf 를 저장 안 해서
    (conf=None) 이 분기 자체가 안 돌았고, 2026-09-09 부터 conf/ 를 저장하기 시작했으니
    켜는 순간 조용히 아무것도 안 거르는 상태가 될 자리였다.

      `absolute`(함수 기본값 = 옛 동작) `conf > conf_threshold` 를 그대로 쓴다.
      `relative`  DA3 자신이 쓰는 방식 — non-sky 픽셀의 **분위수** 를 임계로 삼는다
                  (`depth_anything_3/utils/alignment.py:93` 의 `conf >= median_conf`).
                  이때 `conf_threshold` 는 버릴 분위 (0.5 = 하위 절반 제거, 0 = 안 거름).
    """
    assert conf_mode in ("absolute", "relative"), f"conf_mode: {conf_mode}"
    valid = np.isfinite(depth) & (depth > 0) & ~sky & ~edge_bleed_mask(depth, edge_threshold, dilate_px)
    if conf is None:
        return valid
    if conf_mode == "absolute":
        return valid & (conf > conf_threshold)
    pool = conf[valid]
    if pool.size == 0 or conf_threshold <= 0:
        return valid
    return valid & (conf >= float(np.quantile(pool, min(conf_threshold, 0.99))))


def unproject_mask(depth: np.ndarray, K: np.ndarray, cam_c2w: np.ndarray, mask: np.ndarray):
    """mask 픽셀만 world 좌표로. 반환 (points_world (m,3), pixels_vu (m,2) int)."""
    v, u = np.nonzero(mask)
    if len(v) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64)
    z = depth[v, u].astype(np.float64)
    homo = np.stack([u + 0.5, v + 0.5, np.ones_like(u, dtype=np.float64)], axis=-1)  # m 3
    cam = (homo @ np.linalg.inv(K).T) * z[:, None]                                   # m 3, z_depth 규약
    world = cam @ cam_c2w[:3, :3].T + cam_c2w[:3, 3]
    return world, np.stack([v, u], axis=-1)


def project_points(points_world: np.ndarray, K: np.ndarray, cam_c2w: np.ndarray):
    """world → (uv (m,2), z_cam (m,)). z_cam ≤ 0 은 카메라 뒤."""
    w2c = np.linalg.inv(cam_c2w)
    cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    uv = (cam @ K.T)[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[:, None]
    return uv, z


def graph_frame(up_world: np.ndarray, cam_c2w0: np.ndarray, scale: float):
    """world → G. up = e_z, e_x = frame0 시선의 수평 성분, 원점 = frame0 카메라 중심, 단위 = u.

    G 를 두는 이유는 두 가지다. ① world 는 DA3 가 뱉은 기울어진 카메라 좌표계라 "위"가 없다 —
    roll=0 을 정의하려면 중력축이 축이어야 한다. ② 길이를 `S` 로 나눠 씬 무관 단위 `u` 로 만든다.
    `T_gw` / `T_wg` 를 JSON 에 실어 왕복 가능하게 한다.
    """
    up = up_world / np.linalg.norm(up_world)
    forward0 = cam_c2w0[:3, 2]                       # OpenCV +Z = 시선
    e_x = forward0 - np.dot(forward0, up) * up
    if np.linalg.norm(e_x) < 1e-6:                   # 정확히 수직으로 내려다보는 경우
        e_x = cam_c2w0[:3, 0] - np.dot(cam_c2w0[:3, 0], up) * up
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(up, e_x)
    R_gw = np.stack([e_x, e_y, up], axis=0)          # 행이 G 의 축 = world→G 회전
    origin = cam_c2w0[:3, 3]

    T_gw = np.eye(4)
    T_gw[:3, :3] = R_gw / scale
    T_gw[:3, 3] = -(R_gw @ origin) / scale
    return T_gw, np.linalg.inv(T_gw)


def apply_transform(T: np.ndarray, points: np.ndarray):
    return points @ T[:3, :3].T + T[:3, 3]


def radius_outlier_mask(points: np.ndarray, radius: float, min_neighbors: int = 6):
    """반경 내 이웃이 부족한 점 = 노이즈. sklearn DBSCAN 대신(env 에 없다) cKDTree 로 같은 일.

    depth 노이즈는 물체에서 멀리 떨어진 산발적 점으로 나타나고, minAreaRect 는 **최외곽 점 하나**
    에 끌려가므로 이걸 안 하면 OBB 가 통째로 늘어난다.
    """
    from scipy.spatial import cKDTree

    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    counts = cKDTree(points).query_ball_point(points, r=radius, return_length=True)
    return counts >= min_neighbors + 1  # 자기 자신 포함
