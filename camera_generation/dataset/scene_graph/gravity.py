"""중력축 `up_world` 추정. **scene graph 가 존재해야 하는 가장 중요한 이유가 이것이다.**

roll=0 을 world Y 로 정의하면 안 된다 — world 는 DA3 frame0 카메라 좌표계라 카메라가 기울어져
있으면 전 shot 에 Dutch angle 이 박힌다. 그래서 중력축을 따로 구해 그걸 기준으로 roll 을 0 으로
잡는다. OBB 도 이 축에 정렬한다.

두 경로:
  ① ground RANSAC — 정적 점군에서 평면을 3회까지 벗겨내며(peel) 지면 후보를 찾는다.
     accept 조건 3개를 **전부** 만족해야 한다: inlier_ratio ≥ 0.15 (진짜 큰 평면인가),
     ∠(n, u_cam) ≤ 45° (카메라가 짐작하는 위쪽과 크게 안 어긋나는가), 평면이 카메라보다 아래.
     벽/천장/책상 상판을 지면으로 오인하는 걸 막는 게 이 세 조건의 전부다.
  ①' weak ground (`weak_inlier_ratio > 0` 일 때만) — ①의 세 조건 중 **기하 조건 둘**
     (각도, 카메라가 위)은 통과했는데 inlier_ratio 만 모자란 후보를 살린다. 실내 씬에서
     바닥이 화면의 15% 를 못 넘는 경우가 실제로 있다: trumans-bedroom 은 0.142/12.8°,
     park-selfie 0.105/11.4°, basketball-four 0.132/28.2° 로 셋 다 조용히 fallback 으로
     떨어졌고, 그 fallback 이 실제 바닥과 12~28° 어긋난 up 을 내놓았다.
     confidence 는 `ratio/0.5` 를 절반으로 깎아 ②보다는 높고 ①보다는 낮게 준다.
     **기본값 0.0 = 꺼짐** — 51편 코퍼스 결과를 바꾸지 않기 위해.
  ② fallback `u_cam = normalize(Σ_t −R_c2w[t][:,1])` — OpenCV 는 +Y 가 아래라 −Y 가 위.
     confidence 0.3 으로 낮게 준다. 핸드헬드 영상은 카메라 up 이 대체로 중력축이라 꽤 맞지만,
     의도적으로 기울인 shot 에선 틀린다.

open3d `segment_plane` 을 안 쓴다(env 에 없다) — 같은 알고리즘을 numpy 로 쓴다.
"""
import numpy as np


def fit_plane_ransac(points: np.ndarray, dist_threshold: float, num_iterations: int = 2000,
                     seed: int = 0):
    """(normal(3,), d, inlier_mask). 평면은 `n·p + d = 0`, ‖n‖=1."""
    rng = np.random.default_rng(seed)
    best_inliers, best_plane = None, None
    if len(points) < 3:
        return None, None, np.zeros(len(points), dtype=bool)

    for _ in range(num_iterations):
        idx = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(normal @ p0)
        inliers = np.abs(points @ normal + d) < dist_threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_plane = inliers, (normal, d)

    if best_plane is None:
        return None, None, np.zeros(len(points), dtype=bool)
    # 최소제곱 재적합 — RANSAC 의 3점 표본은 방향이 거칠다.
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    d = -float(normal @ centroid)
    return normal, d, np.abs(points @ normal + d) < dist_threshold


def camera_up(cam_c2w: np.ndarray):
    """Σ_t −R_c2w[t][:,1] 정규화. OpenCV +Y = 아래이므로 −Y 가 위."""
    up = -cam_c2w[:, :3, 1].sum(axis=0)
    return up / np.linalg.norm(up)


def estimate_gravity(points_world: np.ndarray, cam_c2w: np.ndarray, scale: float,
                     dist_threshold_u: float = 0.01, min_inlier_ratio: float = 0.15,
                     max_angle_deg: float = 45.0, max_peels: int = 3, seed: int = 0,
                     weak_inlier_ratio: float = 0.0):
    """정적 점군 → up_world. 반환 dict(up_world, method, confidence, ...)."""
    u_cam = camera_up(cam_c2w)
    cam_centers = cam_c2w[:, :3, 3]
    attempts = []
    weak = None  # 기하 조건은 통과했는데 inlier_ratio 만 모자란 최선의 후보

    remaining = points_world
    for peel in range(max_peels):
        if len(remaining) < 1000:
            break
        normal, d, inliers = fit_plane_ransac(remaining, dist_threshold_u * scale, seed=seed + peel)
        if normal is None:
            break
        ratio = float(inliers.mean())
        # 법선 부호를 카메라 up 쪽으로 맞춘다 (평면은 부호가 없다).
        if normal @ u_cam < 0:
            normal, d = -normal, -d
        angle = float(np.degrees(np.arccos(np.clip(normal @ u_cam, -1, 1))))
        # "평면이 카메라 아래" = 카메라 중심을 평면 식에 넣으면 법선(위쪽) 방향으로 양수.
        below = float(np.mean(cam_centers @ normal + d > 0))
        attempts.append({"peel": peel, "inlier_ratio": ratio, "angle_to_cam_up_deg": angle,
                         "cams_above_frac": below})
        if ratio >= min_inlier_ratio and angle <= max_angle_deg and below > 0.9:
            return {
                "up_world": normal, "plane_d": d, "method": "ground_ransac",
                "confidence": float(min(1.0, ratio / 0.5)),
                "inlier_ratio": ratio, "angle_to_cam_up_deg": angle, "peel": peel,
                "attempts": attempts,
            }
        if angle <= max_angle_deg and below > 0.9 and (weak is None or ratio > weak[0]):
            weak = (ratio, normal, d, angle, peel)
        remaining = remaining[~inliers]  # 이 평면은 지면이 아니다 → 벗겨내고 다음 후보

    if weak is not None and weak[0] >= weak_inlier_ratio > 0:
        ratio, normal, d, angle, peel = weak
        return {
            "up_world": normal, "plane_d": d, "method": "ground_ransac_weak",
            "confidence": float(min(1.0, ratio / 0.5)) * 0.5,
            "inlier_ratio": ratio, "angle_to_cam_up_deg": angle, "peel": peel,
            "attempts": attempts,
        }

    return {"up_world": u_cam, "plane_d": None, "method": "camera_up_fallback",
            "confidence": 0.3, "inlier_ratio": float("nan"), "angle_to_cam_up_deg": 0.0,
            "peel": -1, "attempts": attempts}
