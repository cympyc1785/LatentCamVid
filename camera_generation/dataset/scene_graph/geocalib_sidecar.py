"""GeoCalib 사이드카(`geocalib_gravity.json`) 로더. **추정을 안 하고 읽기만 한다.**

왜 파일로 주고받는가: GeoCalib 은 `kornia` 를 요구하는데 그건 env `geocalib` 에만 있고,
scene graph 는 env `vista4d` 에서 돈다. 새 패키지를 깔지 않기로 했으므로 두 단계로 쪼갠다 —
`fit/ingest/geocalib_gravity.py`(env `geocalib`) 가 사이드카를 굽고, 여기서 읽는다.

왜 GeoCalib 을 1순위로 쓰는가 (TRUMANS GT 65 chunk 실측, GT 대비 각도 오차):

    RANSAC 판정        n    ransac~GT (med/p90/max)   geocalib~GT (med/p90/max)
    camera_up_fallback 15    11.14 / 15.88 / 24.18      0.75 / 1.39 / 1.66
    ground_ransac      50     0.08 /  4.07 /  5.28      0.96 / 2.55 / 5.59
    전체               65     1.79 / 12.48 / 24.18      0.87 / 2.42 / 5.59

RANSAC 은 물었을 때만 정확하고(median 0.08°), **vista 53편 중 15편(28%)이 fallback 으로
떨어진다**. 게다가 "성공"으로 찍힌 것 중에도 각도 상한 45° 바로 아래에 앉은 것들이 있다
(goat 41.1°, rabbit-basket 38.4°). GeoCalib 은 최악이 5.59° 로 꼬리를 자른다.

그래서 정책은 **GeoCalib 주(主) · RANSAC 검증**이다. RANSAC 은 계속 돌리되 결과는
`angle_to_ransac_deg` 로 기록만 하고 자동 전환은 하지 않는다 — 어긋나면 숨기지 말고 드러낸다.

`confidence` 는 프레임간 up 방향의 최대 각도차(`spread_deg`)에서 나온 값이다. GT 실측에서
spread<5° 인 206 chunk 는 오차 p90 1.64°, spread≥5° 인 40 chunk 는 p90 7.05° 로 갈렸다.

사용 예시:
    from scene_graph.geocalib_sidecar import load_geocalib_gravity
    g = load_geocalib_gravity("camel", output_root="out")   # None 이면 사이드카가 없다
    g["up_world"], g["confidence"], g["spread_deg"]
"""
import json
from os import path

import numpy as np

FORMAT = "geocalib_gravity_v1"
SIDECAR_NAME = "geocalib_gravity.json"


def sidecar_path(video: str, output_root: str) -> str:
    return path.join(output_root, video, SIDECAR_NAME)


def load_geocalib_gravity(video: str, output_root: str):
    """사이드카 → `estimate_gravity` 와 **같은 키 집합**의 dict. 없으면 None.

    `gt_trumans.gt_gravity` 와 같은 계약이다 — 하류(`graph_frame` 이하)가 어느 소스에서
    왔는지 몰라도 되게. `plane_d` 는 None: GeoCalib 은 지면 높이를 모른다. vista 경로는
    `relations.ground_height`(G 프레임 z 2% 분위수)가 지면을 다시 뽑으므로 상관없다.
    """
    file_path = sidecar_path(video, output_root)
    if not path.isfile(file_path):
        return None
    with open(file_path, encoding="utf-8") as file:
        payload = json.load(file)
    assert payload.get("format") == FORMAT, (
        f"{file_path}: format 이 {payload.get('format')} 다 (기대 {FORMAT}). "
        f"fit/ingest/geocalib_gravity.py 로 다시 구울 것.")

    up = np.asarray(payload["up_world"], dtype=np.float64)
    up /= np.linalg.norm(up)
    return {
        "up_world": up,
        "plane_d": None,
        "method": "geocalib",
        "confidence": float(payload["confidence"]),
        "inlier_ratio": float("nan"),
        "angle_to_cam_up_deg": float(payload["angle_to_cam_up_deg"]),
        "attempts": [],
        # 진단 — 이 셋이 사이드카를 신뢰할지 판단하는 근거다.
        "spread_deg": float(payload["spread_deg"]),
        "num_frames": len(payload.get("frames", [])),
        "num_kept": int(payload.get("num_kept", 0)),
        "trim_shift_deg": float(payload.get("trim_shift_deg", float("nan"))),
        "prior_focal": bool(payload.get("prior_focal", False)),
        "source": path.relpath(file_path, output_root),
    }


def angle_between(a, b) -> float:
    """두 단위벡터 사이 각도(도). 검증 필드 기록용."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))
