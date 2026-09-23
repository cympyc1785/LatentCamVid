"""TRUMANS 전용 **GT 중력축 / GT 지면 / GT subject**. 추정을 안 한다.

왜 이게 가능한가: TRUMANS 클립은 우리가 Blender 에서 직접 렌더한 것이라 카메라의 **blend
world pose 가 디스크에 남아 있다** — `render_a<NN>/cameras.json` 의 `c2w_opencv` 다. blend 씬은
z-up 이므로 중력축은 정확히 `+Z` 이고, `trumans_to_recon.py` 가 `inv(world[0]) @ world` 로
frame0 앵커만 거는 rigid 변환이므로 recon world 에서의 중력축은

    up_recon = R0ᵀ @ [0,0,1] = R0[2, :]          (R0 = c2w_opencv[0][:3,:3])

RANSAC 도, 카메라-up fallback 도, confidence 도 필요 없다.

왜 바꿔야 하는가 (191 chunk 실측, `estimate_gravity` 대비 각도 오차):

    method                  n   median    p90    max   >10°   >20°
    ground_ransac         142      0.1°   4.2°   9.7°     0%     0%
    camera_up_fallback     49     10.9°  19.1°  26.2°    59%    10%

RANSAC 이 물면 사실상 정확하다. 문제는 **4개 중 1개가 fallback 으로 떨어진다**는 것이고,
그 49편은 중앙값 10.9° 기울어 있다 — roll=0 이 중력 기준이 아니라 카메라 기준이 되므로
그 chunk 의 뱅크 전량에 Dutch angle 이 박히고, OBB 도 같은 각도로 기울어 fit 된다.

지면도 같이 GT 로 준다. `probe_a<NN>.json` 의 `floor_z` 는 anchor 프레임에서 사람 발 아래로
광선을 쏴서 잰 blend world 높이다 (`trumans_scene_probe.py:344-358`). 점군 2% 분위수
(`relations.ground_height`)와 달리 가구 위에 서 있어도 진짜 바닥을 준다.

subject 도 GT 로 준다 (D106). depth 점군 OBB 는 **보이는 면만** 있는 shell 이라 systematically
작다 — 9편 실측(프레임 정렬): 높이비 median 0.73 (min 0.42), 수평비 median 0.70,
조준점(track center) 오차 median 0.255 m = 키의 16%. `probe_a<NN>.json:human_track` 은 blend
에서 mesh 정점 union 으로 잰 프레임별 AABB (min/max/center) 라 진짜 크기·중심이다.

사용 예시:
    from scene_graph.gt_trumans import gt_gravity, gt_subject
    g = gt_gravity("tru_00add26c_a01_s3f0k6")       # None 이면 TRUMANS chunk 가 아니다
    g["up_world"], g["ground_z_g"]                   # 후자는 T_gw 를 넘겨야 채워진다
    s = gt_subject("tru_00add26c_a01_s3f0k6", T_gw, 49)
    s["centers_g"], s["extent_u"], s["yaw_g"]        # G frame, 단위 u
"""
import json
from glob import glob
from os import path

import numpy as np

RECON_ROOT_DEFAULT = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                               "out", "trumans_recon")


def parse_chunk(video: str):
    """`tru_00add26c_a01_s3f0k6` → (`00add26c`, `a01`, `s3f0k6`). TRUMANS 가 아니면 None."""
    parts = video.split("_", 3)
    if len(parts) != 4 or parts[0] != "tru":
        return None
    return parts[1], parts[2], parts[3]


def resolve_dirs(video: str, recon_root: str = RECON_ROOT_DEFAULT):
    """chunk → (render_dir, probe_json). 못 찾으면 (None, None).

    작업 디렉토리 이름은 `<uuid>_<suffix>` 이고 chunk 이름에는 uuid 앞 8자만 들어 있다.
    glob 이 2개 이상 물면 **모호한 것이므로 실패로 친다** — 조용히 다른 recording 의 GT 를
    집어오는 것보다 낫다.
    """
    parsed = parse_chunk(video)
    if parsed is None:
        return None, None
    prefix, action, suffix = parsed
    work = glob(path.join(recon_root, f"{prefix}-*_{suffix}"))
    if len(work) != 1:
        return None, None
    render = path.join(work[0], f"render_{action}")
    probe = path.join(work[0], f"probe_{action}.json")
    return (render if path.isfile(path.join(render, "cameras.json")) else None,
            probe if path.isfile(probe) else None)


def gt_gravity(video: str, recon_root: str = RECON_ROOT_DEFAULT, T_gw: np.ndarray = None):
    """chunk → GT 중력 dict. `estimate_gravity` 와 **같은 키**를 채워서 하류가 안 갈라지게 한다.

    `plane_d` 는 world 좌표 평면식 `up·p + d = 0` 의 d 다 (GT 지면이 있을 때만).
    `T_gw` 를 주면 `ground_z_g` 도 채운다 — G 프레임은 `e_z = up` 이므로 지면은 z 상수 평면이다.
    """
    render, probe = resolve_dirs(video, recon_root)
    if render is None:
        return None

    with open(path.join(render, "cameras.json"), encoding="utf-8") as file:
        cameras = json.load(file)
    world0 = np.asarray(cameras["cameras"][0]["c2w_opencv"], dtype=np.float64)
    up = world0[2, :3].copy()                               # R0ᵀ @ e_z(blend) = R0 의 3행
    up /= np.linalg.norm(up)

    floor_z = None
    if probe is not None:
        with open(probe, encoding="utf-8") as file:
            floor_z = json.load(file).get("floor_z")
    # blend 평면 `z = floor_z` 를 world 로: z_blend = up·p_world + t0_z 이므로 d = t0_z − floor_z.
    plane_d = None if floor_z is None else float(world0[2, 3] - floor_z)

    out = {
        "up_world": up, "plane_d": plane_d, "method": "blend_gt_z_up", "confidence": 1.0,
        "inlier_ratio": float("nan"), "attempts": [], "floor_z_blend": floor_z,
        # 진단용 — recon world 는 frame0 앵커라 frame0 카메라 up 이 정확히 −e_y 다.
        # 이 값이 크면 그 chunk 는 예전에 fallback 으로 떨어졌을 때 그만큼 기울었다.
        "angle_to_cam_up_deg": float(np.degrees(np.arccos(np.clip(-up[1], -1, 1)))),
        "source": path.relpath(render, recon_root),
    }
    if T_gw is not None and plane_d is not None:
        # 평면 위의 한 점을 G 로 옮기면 그 z 가 곧 ground_z (G 의 e_z 가 up 이므로 평면은 z 상수).
        on_plane = -plane_d * up
        out["ground_z_g"] = float(on_plane @ T_gw[2, :3] + T_gw[2, 3])
    return out


def gt_subject(video: str, T_gw: np.ndarray, num_frames: int,
               recon_root: str = RECON_ROOT_DEFAULT):
    """chunk → GT subject 궤적·크기 (G frame). 못 찾으면 None, **부분 커버리지는 assert**.

    blend AABB 를 G 의 OBB 로 옮기는 근거: blend → recon 은 `inv(world0)` (rigid),
    recon → G 는 `T_gw` (up = blend +Z 일 때 회전부가 z 를 z 로 보냄 + 1/S). 그래서 합성
    `A = T_gw[:3,:3] @ inv0[:3,:3]` 은 **순수 yaw × 1/S** 이고, blend 축 정렬 AABB 는 G 에서
    yaw 하나로 기술되는 OBB 가 된다. 이 전제는 `--gravity_source gt` 일 때만 성립하므로
    assert 로 지킨다 (RANSAC up 이면 z 가 섞여서 AABB 가 기울어진 평행육면체가 된다).

    프레임 매핑: recon 프레임 i 의 blend 절대 프레임은 `probe.frame_list[i]` 이고 그 값이
    `human_track[*].frame` 을 키한다. 한 프레임이라도 비면 조용히 절반 GT 를 만드는 대신 assert.
    """
    render, probe_path = resolve_dirs(video, recon_root)
    if render is None or probe_path is None:
        return None
    with open(probe_path, encoding="utf-8") as file:
        probe = json.load(file)
    track = probe.get("human_track") or []
    frame_list = probe.get("frame_list") or []
    if not track or len(frame_list) < num_frames:
        return None

    with open(path.join(render, "cameras.json"), encoding="utf-8") as file:
        cameras = json.load(file)
    inv0 = np.linalg.inv(np.asarray(cameras["cameras"][0]["c2w_opencv"], dtype=np.float64))
    A = T_gw[:3, :3] @ inv0[:3, :3]                    # blend → G 선형부 (yaw × 1/S)
    b = T_gw[:3, :3] @ inv0[:3, 3] + T_gw[:3, 3]
    inv_scale = float(np.linalg.norm(A[:, 0]))         # = 1/S
    # 순수 yaw 검증 — z 행/열의 비대각이 크면 G 의 up 이 blend +Z 가 아니다.
    off = max(abs(A[2, 0]), abs(A[2, 1]), abs(A[0, 2]), abs(A[1, 2])) / inv_scale
    assert off < 1e-3, (
        f"{video}: blend→G 가 순수 yaw 가 아니다 (off-diag {off:.4f}). "
        f"--gravity_source gt 와 함께 써야 한다.")
    yaw_g = float(np.arctan2(A[1, 0], A[0, 0]))

    by_frame = {int(e["frame"]): e for e in track}
    missing = [frame_list[i] for i in range(num_frames) if frame_list[i] not in by_frame]
    assert not missing, f"{video}: human_track 에 없는 프레임 {missing[:5]} (총 {len(missing)})"

    centers, extents = [], []
    for i in range(num_frames):
        entry = by_frame[frame_list[i]]
        centers.append(A @ np.asarray(entry["center"], dtype=np.float64) + b)
        extents.append((np.asarray(entry["max"], dtype=np.float64)
                        - np.asarray(entry["min"], dtype=np.float64)) * inv_scale)
    return {
        "centers_g": np.stack(centers), "extents_u": np.stack(extents),
        "extent_u": np.median(np.stack(extents), axis=0), "yaw_g": yaw_g,
        "human_height": probe.get("human_height"),
        "method": "blend_gt_mesh_aabb", "source": path.relpath(probe_path, recon_root),
    }
