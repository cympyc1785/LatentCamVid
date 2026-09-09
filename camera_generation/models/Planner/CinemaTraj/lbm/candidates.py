"""subject 를 도는 후보 카메라 풀. **후보는 G 좌표에서 정의하고 world 로 내보낸다.**

왜 subject local frame 인가: LBM 의 9-turnaround 는 씬 원점 기준 방위각이었다. 우리 subject 는
움직이고(camel path_len 0.6u) 제 방향이 있으므로(OBB yaw), "정면 45도"가 말이 되려면 원점이
물체여야 한다. 그래서 후보는 `(azimuth, elevation, distance_ratio)` 세 숫자로만 기술되고, 그
셋이 그대로 VLM 에게 보여줄 텍스트가 된다 — **좌표는 프롬프트에 안 나간다**.

방위각 규약은 `scene_graph/obb.py:observation_azimuths` 와 같다: 로컬 +x(=OBB yaw 축)가 0°,
반시계(+z 중력축 기준)가 +. 그래야 `obs_az_span_deg` 와 후보 방위각을 같은 자로 비교할 수 있다.

**`obs_az_span` 하드 필터는 기본 off** (`--az_margin_deg 180`). 계획서에는 "관측 범위 밖 방위는
즉시 제거"였는데 camel dyn_1 의 `obs_az_span` 이 2.6° 다 — 시차 0.0046 이라 사실상 한 점에서 봤다.
그대로 걸면 9개 방위 중 1개만 남아 board 가 성립하지 않는다. 못 본 쪽은 점군이 비어 있어
**G2 coverage 가 알아서 떨어뜨린다**. 그게 추측이 아니라 측정이라 그쪽을 믿는다.

**풀은 기본적으로 τ 예산에서 역산한다** (`pool_mode="budget"`). 계획서의 고정 격자
(az {0,±45,±90,±135,180} × el {12,28,45} × r {0.7,1.0,1.4}·d_ref)를 camel 에서 그대로 돌렸더니
72개 중 **70개가 G4_tau 에서 죽었다**. 우연이 아니라 산수다: `max_tau 0.30`, `z_med 3.553`,
`S 4.6713` 이면 소스 카메라에서 움직여도 되는 거리가 `0.30·3.553/4.6713 = 0.228 u` 인데
`d_ref` 가 `0.64 u` 라, 반경만 1.4배로 늘려도(0.256 u) 이미 예산 초과다. 방위각은 ±20° 가 한계고
"정면 45도"는 도달 불가능하다. 고정 격자는 board 에 **볼 수 없는 후보 70개**를 만들고 2개만
남긴다 — look-before-move 의 요점이 "고를 수 있는 것들 중에서 고르게 한다"인데 그게 무너진다.

그래서 budget 모드는 소스 카메라의 (방위, 고도, 반경)에서 출발해 예산 안쪽으로만 격자를 편다.
씬이 실제로 허용하는 범위가 곧 board 가 되므로, 시차가 큰 씬(avocado-slice 0.129)에서는 같은
코드가 저절로 넓게 편다. 계획서 격자는 `pool_mode="absolute"` 로 그대로 남겨뒀다.
"""
import numpy as np

AZIMUTHS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
ELEVATIONS_DEG = (12.0, 28.0, 45.0)
DISTANCE_RATIOS = (0.7, 1.0, 1.4)


def subject_frame(node: dict, frame: int = 0):
    """프레임 `frame` 에서 subject 의 (center_g, R_j). R_j 열 = [fwd | left | up]."""
    center = np.asarray(node["track"]["center_smooth"], dtype=float)[frame]
    yaw = float(np.asarray(node["track"]["yaw"], dtype=float)[frame])
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    return center, np.stack([fwd, left, up], axis=1)


def build_pool(node: dict, azimuths_deg=AZIMUTHS_DEG, elevations_deg=ELEVATIONS_DEG,
               distance_ratios=DISTANCE_RATIOS, look_at_bias: float = 0.0,
               az_margin_deg: float = 180.0, frame: int = 0):
    """후보 리스트. 각 항목은 G 좌표 위치/타겟 + VLM 에게 보여줄 세 숫자.

    `look_at_bias` 는 OBB 높이의 몇 배만큼 시선을 위/아래로 옮길지다 (0 = OBB 중심).
    큰 동물은 +0.2 쯤이 머리 쪽으로 붙는데, 기본은 0 으로 두고 micro-adjust(pan_up/down)에 맡긴다.
    """
    center, R_j = subject_frame(node, frame)
    height = float(node["obb"]["extent"][2])
    d_ref = float(node["viewing_distance"]["d_ref"])
    look_at = center + look_at_bias * height * np.array([0.0, 0.0, 1.0])

    span = float(node["obs_az_span_deg"])
    allowed = span / 2.0 + az_margin_deg

    pool = []
    for azimuth in azimuths_deg:
        if abs((azimuth + 180) % 360 - 180) > allowed:
            continue
        for elevation in elevations_deg:
            for ratio in distance_ratios:
                phi, theta = np.radians(azimuth), np.radians(elevation)
                direction = np.array([np.cos(theta) * np.cos(phi),
                                      np.cos(theta) * np.sin(phi),
                                      np.sin(theta)])
                pool.append({
                    "cand_id": f"c{len(pool):03d}",
                    "azimuth_deg": azimuth, "elevation_deg": elevation, "distance_ratio": ratio,
                    "radius_u": ratio * d_ref,
                    "p_g": center + ratio * d_ref * (R_j @ direction),
                    "look_at_g": look_at,
                })
    return pool


def source_spherical(node: dict, cam_centers_g: np.ndarray, frame: int = 0):
    """소스 카메라를 subject local 구면좌표로: (azimuth_deg, elevation_deg, radius_u)."""
    center, R_j = subject_frame(node, frame)
    local = R_j.T @ (np.asarray(cam_centers_g, dtype=float)[frame] - center)
    radius = float(np.linalg.norm(local))
    return (float(np.degrees(np.arctan2(local[1], local[0]))),
            float(np.degrees(np.arcsin(np.clip(local[2] / max(radius, 1e-9), -1, 1)))),
            radius)


def budget_spans(radius_u: float, max_tau: float, z_med: float, scale: float,
                 azimuth_share: float = 0.8, elevation_share: float = 0.5,
                 radius_share: float = 0.5):
    """τ 예산 → (azimuth 반폭 deg, elevation 반폭 deg, 반경 비율 반폭).

    `budget_u = max_tau·z_med/S` 가 소스 카메라에서 떨어져도 되는 거리다. 방위/고도는 반경
    `radius_u` 원 위의 현(chord)이 그 예산의 `share` 배가 되는 각도로 잡는다. 셋을 동시에 최대로
    쓰면 코너에서 예산을 넘지만, 그건 G4 가 몇 개 떨어뜨리는 것으로 족하다 — share 를 합이 1이
    되게 쪼개면 격자가 예산 안쪽으로만 오그라들어 board 가 전부 비슷해진다.
    """
    budget_u = max_tau * z_med / max(scale, 1e-9)
    chord = lambda share: float(np.degrees(2 * np.arcsin(
        np.clip(share * budget_u / (2 * max(radius_u, 1e-9)), 0.0, 1.0))))
    return (chord(azimuth_share), chord(elevation_share),
            float(np.clip(radius_share * budget_u / max(radius_u, 1e-9), 0.0, 0.6)),
            budget_u)


def build_pool_budget(node: dict, cam_centers_g: np.ndarray, max_tau: float, z_med: float,
                      scale: float, num_azimuth: int = 5, num_elevation: int = 3,
                      num_radius: int = 3, look_at_bias: float = 0.0, frame: int = 0):
    """소스 카메라 구면좌표를 중심으로 τ 예산 안쪽에 격자를 편다. (pool, info)."""
    center, R_j = subject_frame(node, frame)
    height = float(node["obb"]["extent"][2])
    look_at = center + look_at_bias * height * np.array([0.0, 0.0, 1.0])
    az_src, el_src, radius_src = source_spherical(node, cam_centers_g, frame)
    az_span, el_span, radius_span, budget_u = budget_spans(radius_src, max_tau, z_med, scale)

    grid = lambda span, n: (np.array([0.0]) if n <= 1 else np.linspace(-span, span, n))
    pool = []
    for d_az in grid(az_span, num_azimuth):
        for d_el in grid(el_span, num_elevation):
            for d_r in grid(radius_span, num_radius):
                azimuth, elevation = az_src + d_az, el_src + d_el
                radius = radius_src * (1.0 + d_r)
                phi, theta = np.radians(azimuth), np.radians(elevation)
                direction = np.array([np.cos(theta) * np.cos(phi),
                                      np.cos(theta) * np.sin(phi),
                                      np.sin(theta)])
                pool.append({
                    "cand_id": f"c{len(pool):03d}",
                    "azimuth_deg": round(float(azimuth), 1),
                    "elevation_deg": round(float(elevation), 1),
                    # OBB yaw 는 180° 대칭이라 절대 방위각은 "정면"이 어딘지 못 정한다.
                    # VLM 이 실제로 쓸 수 있는 건 **소스 카메라에서 얼마나 돌았나**다.
                    "d_azimuth_deg": round(float(d_az), 1),
                    "d_elevation_deg": round(float(d_el), 1),
                    "distance_ratio": round(float(radius / max(radius_src, 1e-9)), 3),
                    "radius_u": float(radius),
                    "p_g": center + radius * (R_j @ direction),
                    "look_at_g": look_at,
                })
    info = {"mode": "budget", "budget_u": round(budget_u, 4),
            "source_azimuth_deg": round(az_src, 1), "source_elevation_deg": round(el_src, 1),
            "source_radius_u": round(radius_src, 4), "azimuth_span_deg": round(az_span, 1),
            "elevation_span_deg": round(el_span, 1), "radius_span": round(radius_span, 3)}
    return pool, info


def g_pose_to_world(p_g, look_at_g, T_wg, look_at_c2w):
    """G 의 (위치, 타겟) → world OpenCV c2w. up 은 **G 의 +z(중력축)를 world 로 돌린 것**.

    `T_wg` 는 스케일 S 를 품고 있으므로(`T_gw[:3,:3] = R_gw/S`) 회전에 그대로 곱하면 안 된다.
    위치·타겟만 변환하고 회전은 world 에서 다시 세운다 — 그래야 `det(R)=1` 이 보장된다.
    """
    to_world = lambda p: np.asarray(T_wg)[:3, :3] @ np.asarray(p, dtype=float) + np.asarray(T_wg)[:3, 3]
    up_world = np.asarray(T_wg)[:3, :3] @ np.array([0.0, 0.0, 1.0])
    up_world /= np.linalg.norm(up_world)
    return look_at_c2w(to_world(p_g), to_world(look_at_g), up_world)


def pick_subject(nodes: list, subject_id: str | None = None, min_area_frac: float = 0.01):
    """subject 선택: 명시 id > (화면 점유 `min_area_frac` 이상 중) 가장 많이 움직인 노드 > 최대 노드.

    **화면 점유 하한이 먼저 걸린다.** "가장 많이 움직인 노드"만 보면 avocado-slice 에서 화면의
    0.24% 짜리 아보카도 조각(path 0.111 u)이 subject 로 뽑히고, 사람(12.7%)이 밀린다. 그러면
    G3 area 게이트가 45개 후보를 전부 떨어뜨린다 — τ 예산 안에서는 그 조각을 화면의 3% 로 키울
    만큼 다가갈 수 없기 때문이다(실측 최대 0.43%). 움직임은 subject 를 **고르는** 기준이 아니라
    같은 급 후보들 사이의 **순위** 기준이어야 한다.

    Stage 2(query grounding)가 들어오면 이 함수가 `subgraph.json` 의 `subject_id` 로 대체된다.
    """
    if subject_id is not None:
        matched = [n for n in nodes if n["id"] == subject_id]
        assert matched, f"subject_id={subject_id} 가 graph 에 없다: {[n['id'] for n in nodes]}"
        return matched[0]
    eligible = [n for n in nodes if n["max_area_frac"] >= min_area_frac] or nodes
    moving = [n for n in eligible if n["moving"]]
    if moving:
        return max(moving, key=lambda n: n["path_len_u"])
    return max(eligible, key=lambda n: n["max_area_frac"])
