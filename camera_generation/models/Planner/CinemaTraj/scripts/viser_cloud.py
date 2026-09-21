"""`lbm/cloud.py` 가 만든 4D point cloud (`cloud.npz`) 를 viser 로 띄운다.

왜 필요한가: 게이트·hole·τ 수치는 전부 `render_frame` 이 만든 2D 렌더에서 나오는데, 그 렌더가
이상할 때 "카메라가 이상한가 / 점군이 이상한가"를 2D 만 봐서는 못 가른다. 점군을 3D 로 직접
띄워 소스 카메라 궤적과 같이 보면 그 두 원인이 눈으로 갈린다 — 깊이 shell 이 찢어졌는지, 동적
물체가 프레임마다 다른 자리에 앉는지, plan 카메라가 shell 안쪽(=벽 속)에 들어갔는지.

4D 인 이유는 `visible (n, f)`: 정적 점은 여러 프레임에서 보이고 동적 점은 **자기 프레임 하나**
에서만 보인다 (`visible.sum(1) == 1`). 그래서 정적 점은 한 번에 다 띄우고, 동적 점은 프레임
슬라이더로 하나씩 갈아끼운다. 둘을 섞어 띄우면 움직이는 물체가 49개 복사본으로 번져서 아무것도
안 보인다.

좌표 규약: `cloud.npz` 의 world 는 **OpenCV** c2w (X right / Y down / Z forward). viser 프러스텀
헬퍼(`latentcam/scripts/viewer/viser_val_cameras.py:add_frustums`)는 OpenGL c2w 를 먹으므로
`c2w_cv @ _GL2CV` 로 넘긴다 (`_GL2CV` 는 자기 역행렬이라 헬퍼 안에서 되돌려진다). 규약의 단일
출처를 그 파일 하나로 유지하려고 재구현하지 않고 import 한다.

## scene graph 오버레이 (`--obb` 기본 켬 / `--no_obb` 로 예전 동작)

`scene_graph.json` 이 있으면 노드 OBB(**동적=하늘색**, 프레임 슬라이더 따라감 / 정적=청록)와
**G 의 `z = ground_z` 지면 격자**를 같이 그린다. 격자가 필요한 이유: `gravity.up_world` 는
숫자 3개라 축이 41° 기울어도 JSON 만 봐서는 안 보인다. 격자가 실제 바닥과 어긋나 있으면
그 씬의 elevation·ground 게이트 판정이 전부 기울어진 축 위에서 났다는 뜻이다.

GUI `obb` 폴더에서 동적/정적/라벨을 따로 끄고, 색(rgb 피커)과 선 굵기를 바꾼다. 색은
`--obb_color_dyn`/`--obb_color_static` 으로 처음부터 지정할 수도 있다. 굵기가 색과 **다른
경로**로 도는 이유: viser line handle 은 `thickness` 는 갱신되는 프로퍼티인데 `colors` 는
아니라, 색을 바꾸려면 remove 후 다시 그리는 수밖에 없다. 동적 OBB 는 노드×프레임이라
snowboard 만 해도 249개고, 색을 바꿀 때마다 전부 다시 그리면 브라우저가 멎는다. 그래서
**버전 도장(`obb_version`)** 을 찍어 두고 `obb_show` 가 "보이게 되는 순간"에만 빚을 갚는다 —
`current frame` 모드면 매번 9개뿐이다.

## view 폴더 — 어느 프레임을 띄우나

`dynamic frames (points+OBB)` 가 세 갈래다: `current frame`(기본, 한 장) / `interval`(range
start~end 를 `frame interval (every N)` 간격으로) / `all frames`(전 구간을 그 간격으로).
동적 점군과 동적 OBB 가 **같은 집합**을 쓴다 — 궤적을 한 화면에 겹쳐 보려고 프레임을 띄엄띄엄
켜는 게 목적인데 점만 켜지고 박스는 한 장만 나오면 대응이 안 보이기 때문이다.

굵기 손잡이는 셋이고 서로 단위가 다르다: `camera thickness`(프러스텀, world 절대값) ·
`path thickness x`(경로선, 선마다 다른 기준값에 곱하는 **배율** — 소스 0.08 / plan 0.10 비율이
슬라이더를 밀어도 유지된다) · `OBB thickness`(cam_scale 배율 — 씬마다 scale 이 100배 다르다).

굵기를 코드에서 바꿀 때는 **`handle.thickness`** 에 넣는다. `handle.line_width` 는 viser 1.1.0
에서 그것의 폐기된 별칭이고, 대입하면 `thickness_units` 가 `"screen"` 으로 못 박혀 world 값이
픽셀로 재해석된다 — 0.02 world 프러스텀에 0.02 를 넣으면 0.02 **픽셀**이 돼 화면에서 사라진다.

`paths`(경로선 전체)와 `probe camera` 는 **기본 꺼짐**이다. 둘 다 프러스텀보다 눈에 먼저 들어와
정작 보려는 카메라 자세를 덮는다 (경로선은 49프레임 꺾은선, probe 는 gizmo 화살표). 예전처럼
처음부터 켜려면 `--paths_on` / `--probe_on`.

## motion 브라우저 (`--banks`)

뱅크 여러 개를 통째로 올려두고 **슬라이더로 motion 을 갈아끼운다**. 이게 필요한 이유는
"카메라가 떨린다"를 2D 렌더로는 못 가르기 때문이다 — 렌더가 떠는 원인이 셋이나 된다:
① plan 카메라 위치의 고주파(`follow_gain` × subject track 잔여 jitter), ② 조준 회전,
③ **점군 자체**(프레임마다 depth 가 조금씩 달라 정적 배경이 숨쉰다). 3D 에서 카메라 경로를
직접 보면 ①/②는 선이 지그재그로 보이고, 선이 매끈한데 렌더가 떨면 남는 건 ③뿐이다.

경로는 catmull-rom 스플라인이 아니라 **`add_line_segments` 생꺾은선**으로 그린다. 스플라인은
지금 보려는 그 jitter 를 부드럽게 만들어 버린다.

## target 카메라 여러 대 동시에 (`--pin` / GUI `pin current`)

슬라이더는 **한 번에 한 대**라 "A 가 B 보다 더 도나 / 둘이 같은 자리에서 시작하나"를 못 본다 —
갈아끼우는 순간 비교 대상이 사라지기 때문이다. pin 은 그 motion 을 고정 색으로 남겨서 활성
motion(주황) 을 계속 바꿔도 화면에 살아 있게 한다. 각 pin 은 경로 선 + 전 프레임 프러스텀 +
**현재 프레임 프러스텀(1.6배)** 셋을 같은 색으로 그리므로, frame 슬라이더를 밀면 pin 들이
동시에 움직여 시점 차이가 눈에 보인다. 색은 `PIN_COLORS` 순환이고 `pinned` 패널이 색↔라벨
대응을 적어 준다. 상한은 `--max_pins` (기본 8) — 프러스텀이 변이당 49/cam_stride 개라
무제한으로 켜면 브라우저가 느려진다.

같이 그리는 것: 소스 카메라 경로(회색) · plan 경로(주황) · subject track(초록, follow 의
입력이라 여기가 떨면 카메라도 떤다). GUI 에 preset/gain/smooth/k 와 `jerk p95`(px/frame³,
`|Δ³p|/z_med·fx` — 화면에서 실제로 몇 px 흔들리는지)를 같이 띄운다.

입력  : `<out>/<video>/cloud.npz` (format `lbm_cloud_v1`)
        선택: `<out>/<video>/<bank>/poses.npz` + `bank.json` (plan 카메라 오버레이)
출력  : 브라우저 (`http://localhost:<port>`)

예시 (env vista4d, screen viser1~4 에서):
    python scripts/viser_cloud.py --video snowboard --port 8084
    python scripts/viser_cloud.py --video snowboard --port 8084 --no_cloud \
        --banks follow_smooth_bank notrack_bank worldaim_bank
    python scripts/viser_cloud.py --video snowboard --port 8084 \
        --bank follow_kf_bank --variant dyn_0__orbit_left_arc__tau0.35   # 예전 동작(1개 고정)
    python scripts/viser_cloud.py --video snowboard --port 8084 --no_cloud \
        --bank hole_bank_k6_d151 --pin orbit_left dolly_in truck_left    # 여러 대 동시에
"""
import json
import sys
import time
from argparse import ArgumentParser
from os import path

import numpy as np
import viser

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
VIEWER_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "..", "latentcam", "scripts", "viewer"))

# world up 문자열 -> world 축 벡터. cloud world 는 OpenCV(=Y down) 라 기본이 '-y' 다.
UP_VECTORS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
              "+z": (0, 0, 1), "-z": (0, 0, -1)}

# probe 프러스텀의 기본 세로 화각. 소스 intrinsics 를 안 쓰는 이유는 그게 씬마다 망원이기
# 때문이다 — camel 은 vfov 16.4° 라 프러스텀이 바늘처럼 길어져 자리를 가늠할 수가 없다.
PROBE_FOV_DEG = 60.0

# probe 를 여러 대 띄우면 전부 같은 색이라 어느 게 어느 건지 못 고른다. 새로 만들 때마다 이
# 팔레트를 돌려 쓰고, `color` 피커로 활성 probe 만 따로 바꾼다. 소스(회색)·플랜(주황)·현재
# 프레임(초록) 과 겹치지 않는 색만 골랐다.
PROBE_COLORS = [(255, 60, 220), (60, 200, 255), (255, 210, 60), (150, 255, 120),
                (255, 120, 60), (180, 140, 255)]

# pin 된 target 카메라 색. 활성 motion(주황 255,140,40)·소스(회색)·subject track(초록)과
# 겹치지 않는 색만 고른다 — pin 의 목적이 "여러 대를 한 화면에서 **구분**하는 것"이라
# 팔레트가 겹치면 기능 자체가 무의미해진다. 개수를 넘기면 순환한다.
PIN_COLORS = [(60, 200, 255), (255, 60, 220), (255, 230, 60), (150, 255, 120),
              (180, 140, 255), (0, 160, 255), (255, 100, 100), (120, 255, 220)]

# [2026-09-21] 카메라 두 계열의 기본색. 여기 상수로 올려 둔 이유는 GUI `camera colors` 피커가
# 이 값을 초기값으로 읽고, `--src_color` / `--plan_color` 가 기동 시 덮기 때문이다 — 세 군데가
# 같은 숫자를 각자 적어 두면 피커가 화면과 다른 색을 가리킨다.
#   gt   = 소스 카메라 (씬이 준 궤적)
#   pred = 활성 motion 의 target 카메라 (뱅크가 낸 궤적)
SRC_COLOR = (140, 140, 140)          # 소스 프러스텀 + 경로선
SRC_NOW_COLOR = (60, 200, 90)        # 소스의 **현재 프레임** 프러스텀 (1.6배)
PLAN_COLOR = (255, 140, 40)          # 활성 target 프러스텀 + 경로선
PLAN_NOW_COLOR = (255, 80, 0)        # target 의 현재 프레임 프러스텀 (1.6배)

# scene graph OBB. **동적이 하늘색**이다 — 보는 대상이 거의 항상 동적 subject 라서, 눈이 먼저
# 가야 하는 쪽에 원래 쓰던 색을 준다. 정적은 겹치지 않게 청록으로 민다 (둘 다 하늘색이면
# "이 박스가 따라 움직여야 하는가"를 못 가른다). `--obb_color_dyn/_static` 로 덮는다.
OBB_DYN_COLOR = (80, 200, 255)       # 하늘색
OBB_STATIC_COLOR = (0, 200, 170)     # 청록


def parse_rgb(text: str, fallback):
    """`"80,200,255"` -> (80, 200, 255). 빈 문자열이면 기본색 그대로."""
    if not text:
        return tuple(int(c) for c in fallback)
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 3 or not all(0 <= v <= 255 for v in parts):
        raise SystemExit(f"색은 0~255 세 개여야 한다: {text!r}")
    return tuple(parts)


def import_frustum_helpers(viewer_root: str):
    """규약(GL↔CV)의 단일 출처는 latentcam 뷰어다 — 여기서 재구현하지 않고 그대로 가져온다."""
    if viewer_root not in sys.path:
        sys.path.insert(0, viewer_root)
    from viser_val_cameras import _GL2CV, add_frustums
    return add_frustums, _GL2CV


def visible_counts(visible_packed: np.ndarray):
    """점별 가시 프레임 수. `unpackbits` 로 (N,49) bool 을 만들면 43M 점에서 2 GB 라 popcount 로 센다."""
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)
    return popcount[visible_packed].sum(axis=1)  # 패딩 비트는 0 이라 그냥 더해도 된다


def subsample(count: int, limit: int, seed: int = 0):
    """limit 이하면 그대로, 넘으면 균등 랜덤 추출 인덱스."""
    if limit <= 0 or count <= limit:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, size=limit, replace=False))


def load_motions(out_root: str, video: str, banks, variant_filter=()):
    """뱅크들의 `poses.npz` 를 한 목록으로 합친다 -> [(label, c2w (F,4,4), row dict)].

    라벨에 뱅크 이름을 접두로 붙이는 이유: 서로 다른 뱅크에 **같은 variant_id** 가 흔하다
    (같은 preset·τ 를 축만 바꿔 되풀기 때문). 접두 없이 합치면 슬라이더에서 어느 뱅크 것인지
    못 가른다. `bank.json` 은 실측치(hole/τ/path)를 붙이려고 같이 읽되, 없으면 그냥 비워둔다 —
    `poses.npz` 만 있는 옛 뱅크도 열려야 한다.

    `variant_filter` 는 **부분문자열 AND** 다 (`["s9", "k3"]` = 둘 다 든 것만). 하나만 주면
    예전 단일 필터와 완전히 같다. 축이 5개(preset·τ·follow·smooth·keyframe)라 한 축만 잡아서는
    목록이 안 줄어든다 — s9 만 걸면 14개, k3 까지 걸어야 실제로 보고 싶은 것만 남는다.
    """
    if isinstance(variant_filter, str):
        variant_filter = [variant_filter] if variant_filter else []
    motions = []
    for bank in banks:
        folder = path.join(out_root, video, bank)
        poses = np.load(path.join(folder, "poses.npz"))
        rows = {}
        bank_json = path.join(folder, "bank.json")
        if path.isfile(bank_json):
            with open(bank_json, encoding="utf-8") as file:
                rows = {v["variant_id"]: v for v in json.load(file)["variants"]}
        for index, variant in enumerate([str(v) for v in poses["variant_id"]]):
            if not all(token in variant for token in variant_filter):
                continue
            row = dict(rows.get(variant, {}))
            # subject track 을 그리려면 어느 노드가 anchor 인지 알아야 한다. bank.json 이 없는
            # 옛 뱅크도 poses.npz 안에는 anchor_id 를 들고 있다.
            row.setdefault("anchor_id", str(poses["anchor_id"][index]))
            motions.append((f"{bank}/{variant}", poses["cam_c2w"][index].astype(np.float64), row))
    assert motions, f"뱅크 {list(banks)} 에서 {list(variant_filter)} 에 맞는 변이가 0개다"
    return motions


def subject_tracks_world(graph_path: str):
    """scene_graph 의 `track.center_smooth` (graph frame G) -> world -> {node_id: (F,3)}.

    이걸 그리는 이유: `follow_gain` 은 이 곡선의 차분을 그대로 카메라 위치에 더한다 — 여기가
    떨면 카메라도 정확히 g 배로 떤다. 카메라 선만 보면 "원래 떨리는 입력"인지 "우리가 만든
    떨림"인지 못 가른다. 없으면 그냥 빈 dict (track 없이도 뷰어는 돌아야 한다).
    """
    if not path.isfile(graph_path):
        return {}
    with open(graph_path, encoding="utf-8") as file:
        graph = json.load(file)
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
    tracks = {}
    for node in graph["nodes"]:
        centers = np.asarray((node.get("track") or {}).get("center_smooth", []), dtype=float)
        if centers.ndim == 2 and len(centers):
            tracks[node["id"]] = centers @ T_wg[:3, :3].T + T_wg[:3, 3]
    return tracks


# 8 코너를 (sx,sy,sz) 비트로 인덱싱하면, 한 비트만 다른 쌍이 정확히 12 모서리다.
OBB_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
_OBB_SIGNS = np.array([[(i >> 2 & 1) * 2 - 1, (i >> 1 & 1) * 2 - 1, (i & 1) * 2 - 1]
                       for i in range(8)], dtype=float)


def obb_segments_world(center_g, extent, R_g, T_wg):
    """G 프레임 OBB -> world 12 모서리 (12,2,3).

    `extent` 는 **전체 변 길이**다 (`z_lo == center_z - extent_z/2` 로 확인). 반쪽으로 오해하면
    박스가 2배로 그려지는데, 3D 로 보면 "물체보다 크다" 정도로만 보여서 조용히 지나간다.
    """
    corners = np.asarray(center_g, float) + (_OBB_SIGNS * (np.asarray(extent, float) / 2.0)
                                             ) @ np.asarray(R_g, float).T
    world = corners @ np.asarray(T_wg, float)[:3, :3].T + np.asarray(T_wg, float)[:3, 3]
    return np.stack([world[[a for a, _ in OBB_EDGES]], world[[b for _, b in OBB_EDGES]]],
                    axis=1).astype(np.float32)


def yaw_to_R(yaw: float):
    """G 의 중력축(+z) 둘레 회전. `scene_graph/obb.py` 와 같은 정의 — 뷰어가 그 파일을 import
    하면 numpy 외의 의존이 딸려오므로 3줄짜리 이 함수만 되풀어 쓴다."""
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ground_grid_world(ground_z: float, T_wg, center_xy, half: float, lines: int = 13):
    """G 의 `z = ground_z` 평면 격자 -> world 선분.

    이게 있어야 중력축이 틀어졌는지 **눈으로** 판정된다. `up_world` 는 숫자 3개라 41° 기울어도
    JSON 만 봐서는 안 보이고, 지면 격자가 실제 바닥과 어긋나 있으면 한눈에 보인다
    (`angle_to_cam_up_deg` 가 큰 씬들이 여기서 갈렸다).
    """
    ticks = np.linspace(-half, half, lines)
    segs = []
    for t in ticks:
        segs.append([[center_xy[0] + t, center_xy[1] - half, ground_z],
                     [center_xy[0] + t, center_xy[1] + half, ground_z]])
        segs.append([[center_xy[0] - half, center_xy[1] + t, ground_z],
                     [center_xy[0] + half, center_xy[1] + t, ground_z]])
    pts = np.asarray(segs, dtype=float)
    T_wg = np.asarray(T_wg, float)
    return (pts.reshape(-1, 3) @ T_wg[:3, :3].T + T_wg[:3, 3]).reshape(-1, 2, 3).astype(np.float32)


def path_segments(positions: np.ndarray):
    """(F,3) 궤적 -> `add_line_segments` 가 먹는 (F-1, 2, 3). 스플라인이 아니라 생꺾은선이다."""
    points = np.asarray(positions, dtype=np.float32)
    return np.stack([points[:-1], points[1:]], axis=1)


def jerk_px(positions: np.ndarray, z_med: float, focal: float):
    """|Δ³p| p95 를 **화면 픽셀/frame³** 으로. world 단위 jerk 는 씬마다 눈금이 달라 못 읽는다.

    작은 각도에서 화면 변위 ≈ (world 변위 / 깊이) · f 이므로 z_med 로 나누고 fx 를 곱한다.
    실측 기준선: snowboard `follow_gain` 0.91 · smoothing 없음이 10.6 px/frame³ 였다.
    """
    if len(positions) < 4:
        return 0.0
    third = np.linalg.norm(np.diff(np.asarray(positions, dtype=float), n=3, axis=0), axis=-1)
    return float(np.percentile(third, 95) / max(z_med, 1e-9) * focal)


def motion_report(label: str, row: dict, plan_c2w: np.ndarray, track, z_med: float,
                  focal: float, src_jerk: float):
    """GUI 우측에 띄울 텍스트. **소스와 subject track 의 jerk 를 같이** 적는다.

    plan 의 jerk 만 적으면 큰지 작은지 알 수가 없다. `follow_gain` 이 켜져 있으면 plan jerk 는
    거의 정확히 `g × (subject track jerk)` 라, 세 숫자를 나란히 놓으면 떨림이 어디서 왔는지가
    표 하나로 갈린다 (D73 근거).
    """
    gain = float(row.get("follow_gain", 0) or 0)
    lines = [label,
             f"preset      {row.get('preset', '?')}  {row.get('speed', '')} "
             f"{row.get('tracking', '')} b{row.get('look_at_bias', 0)}",
             # g0 이면 offset 이 통째로 0 이라 창 크기가 궤적을 못 바꾼다 — 그때 창을 적으면
             # 안 먹은 설정을 먹은 것처럼 읽힌다.
             f"follow      g{gain:g}  smooth "
             f"{('w' + str(row.get('follow_smooth', '-'))) if gain else '-'} "
             f"({row.get('follow_kind', '-')})",
             f"keyframes   k{row.get('aim_keyframes', 0)}  "
             f"aim_err {row.get('keyframe_aim_err_deg', '-')} deg",
             f"tau         max {row.get('tau_max', '-')}  scale {row.get('tau_scale', '-')}",
             f"path        {row.get('path_len_u', '-')} u  "
             f"view_angle {row.get('view_angle_max_deg', '-')} deg",
             f"hole        {row.get('hole_fraction', '-')}  max {row.get('hole_max', '-')}  "
             f"subj_in_frame {row.get('subject_in_frame', '-')}",
             f"jerk p95    plan {jerk_px(plan_c2w[:, :3, 3], z_med, focal):.2f} px/f3   "
             f"source {src_jerk:.2f}"]
    if track is not None and len(track) > 3:
        lines[-1] += f"   subj_track {jerk_px(track, z_med, focal):.2f}"
    return "\n".join(lines)


def main():
    parser = ArgumentParser()
    parser.add_argument("--video", default="snowboard", type=str)
    # cloud.npz 가 들어있는 상위 폴더. `--video` 와 합쳐 <out>/<video>/cloud.npz 를 연다.
    parser.add_argument("--out", default=path.join(CINEMATRAJ_ROOT, "out"), type=str)
    parser.add_argument("--port", default=8084, type=int)
    # 정적 점 표시 상한. 37M 을 다 보내면 브라우저가 죽는다.
    parser.add_argument("--max_static", default=1_500_000, type=int)
    # 프레임당 동적 점 상한 (0 = 전부). 49프레임을 미리 다 올려두고 슬라이더로 가린다.
    parser.add_argument("--max_dyn_per_frame", default=60_000, type=int)
    # 0 이면 z_med 기준 자동 (0.002·z_med).
    parser.add_argument("--point_size", default=0.0, type=float)
    parser.add_argument("--cam_stride", default=2, type=int)
    # 0 = 0.03·z_med. 슬라이더 초기값일 뿐이고, 띄운 뒤 GUI `view > camera size` 로 바꾼다.
    parser.add_argument("--cam_scale", default=0.0, type=float)
    parser.add_argument("--up", default="-y", choices=sorted(UP_VECTORS))
    # plan 카메라 오버레이 (선택). <out>/<video>/<bank>/poses.npz 의 variant_id 부분일치.
    parser.add_argument("--bank", default="", type=str)
    # 부분문자열 AND. `--variant s9 k3` = smoothing 켜고 keyframe 3개인 것만. 하나만 주면
    # 예전 단일 필터와 같다.
    parser.add_argument("--variant", nargs="*", default=[], type=str)
    # motion 브라우저. 뱅크 여러 개를 통째로 올리고 슬라이더로 갈아끼운다. 비우면 --bank 한 개만
    # 고정으로 그리는 기존 동작 그대로 (--variant 필터는 두 경로 모두에 걸린다).
    parser.add_argument("--banks", nargs="*", default=[], type=str)
    # 기동 직후부터 **동시에** 그려둘 target 카메라. 라벨 부분일치(OR)라 `--pin orbit_left
    # dolly_in` 처럼 주면 맞는 변이를 전부 pin 한다. 띄운 뒤에는 GUI `motion > pin current`
    # 로 늘리고 줄인다. 활성 motion(주황) 은 pin 과 별개로 계속 그려진다.
    parser.add_argument("--pin", nargs="*", default=[], type=str)
    # pin 상한. 프러스텀이 변이당 (49/cam_stride) 개라 무제한으로 켜면 브라우저가 느려진다.
    parser.add_argument("--max_pins", default=8, type=int)
    # 점군을 빼고 카메라 선만 본다 — 떨림만 볼 때는 점군이 시야를 가리고 로딩도 느리다.
    parser.add_argument("--no_cloud", action="store_true")
    # scene_graph.json 의 OBB / 지면 격자 오버레이. `--no_obb` 를 주면 예전 동작 그대로다.
    parser.add_argument("--obb", dest="obb", action="store_true", default=True)
    parser.add_argument("--no_obb", dest="obb", action="store_false")
    parser.add_argument("--ground_grid", dest="ground_grid", action="store_true", default=True)
    parser.add_argument("--no_ground_grid", dest="ground_grid", action="store_false")
    # 색 (전부 `"R,G,B"` 0~255, 빈 문자열이면 위 상수). 카메라 두 계열은 띄운 뒤 GUI
    # `camera colors` 로도 바꾼다 — 여기 플래그는 **기동 시 기본값**이고 피커가 그 값을 읽는다.
    # OBB 는 GUI 를 안 준다: 동적 박스가 노드×프레임이라 snowboard 만 245개고, viser 1.1.0
    # 선 handle 은 색을 바꿔 끼울 수가 없어 드래그 한 번에 245개를 다시 그려야 한다.
    parser.add_argument("--src_color", default="", type=str)
    parser.add_argument("--src_now_color", default="", type=str)
    parser.add_argument("--plan_color", default="", type=str)
    parser.add_argument("--plan_now_color", default="", type=str)
    parser.add_argument("--obb_color_dyn", default="", type=str)
    parser.add_argument("--obb_color_static", default="", type=str)
    # 경로선·probe 는 기본 꺼짐이다. 둘 다 프러스텀보다 눈에 먼저 들어와서, 정작 보려는
    # "카메라가 어디서 어디를 보나"를 가린다. 예전 동작(둘 다 켜짐)은 이 플래그로 돌아온다.
    parser.add_argument("--paths_on", action="store_true",
                        help="카메라 경로선을 처음부터 켠다 (기본 꺼짐, GUI view > paths)")
    parser.add_argument("--probe_on", action="store_true",
                        help="probe 카메라(프러스텀+gizmo)를 처음부터 켠다 (기본 꺼짐)")
    parser.add_argument("--viewer_root", default=VIEWER_ROOT_DEFAULT, type=str)
    args = parser.parse_args()

    src_color = parse_rgb(args.src_color, SRC_COLOR)
    src_now_color = parse_rgb(args.src_now_color, SRC_NOW_COLOR)
    plan_color = parse_rgb(args.plan_color, PLAN_COLOR)
    plan_now_color = parse_rgb(args.plan_now_color, PLAN_NOW_COLOR)
    obb_dyn_color = parse_rgb(args.obb_color_dyn, OBB_DYN_COLOR)
    obb_static_color = parse_rgb(args.obb_color_static, OBB_STATIC_COLOR)

    add_frustums, gl2cv = import_frustum_helpers(args.viewer_root)

    cloud_path = path.join(args.out, args.video, "cloud.npz")
    data = np.load(cloud_path)
    num_frames = int(data["visible_num_frames"])
    # npz 는 lazy 라 --no_cloud 일 때는 아예 안 꺼낸다 (37M 점 = 로딩 수십 초).
    points = np.zeros((0, 3), np.float32) if args.no_cloud else data["points_world"]
    colors = np.zeros((0, 3), np.uint8) if args.no_cloud else data["colors"]
    frame_of = np.zeros(0, np.int32) if args.no_cloud else data["indices"][:, 0]
    cam_c2w = data["meta_cam_c2w"].astype(np.float64)
    intrinsics = data["meta_K"].astype(np.float64)
    width, height = int(data["meta_width"]), int(data["meta_height"])
    z_med = float(data["meta_z_med_frame0"])

    n_visible = (np.zeros(0, np.int32) if args.no_cloud
                 else visible_counts(data["visible_packed"]))
    is_dynamic = n_visible == 1
    static_idx = np.flatnonzero(~is_dynamic)
    static_pick = static_idx[subsample(len(static_idx), args.max_static)]

    point_size = args.point_size if args.point_size > 0 else 0.002 * z_med
    cam_scale = args.cam_scale if args.cam_scale > 0 else 0.03 * z_med

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction(np.array(UP_VECTORS[args.up], dtype=np.float32))

    static_handle, dyn_handles, dyn_counts = None, [], []
    if not args.no_cloud:
        static_handle = server.scene.add_point_cloud(
            "/static", points=points[static_pick], colors=colors[static_pick],
            point_size=point_size)

        # 동적 점은 프레임별로 따로 올린다 — 한 덩어리로 올리면 움직이는 물체가 49겹으로 번진다.
        dyn_order = np.argsort(frame_of[is_dynamic], kind="stable")
        dyn_idx_sorted = np.flatnonzero(is_dynamic)[dyn_order]
        bounds = np.searchsorted(frame_of[dyn_idx_sorted], np.arange(num_frames + 1))
        for f in range(num_frames):
            idx = dyn_idx_sorted[bounds[f]:bounds[f + 1]]
            idx = idx[subsample(len(idx), args.max_dyn_per_frame, seed=f)]
            dyn_counts.append(len(idx))
            dyn_handles.append(server.scene.add_point_cloud(
                f"/dynamic/f{f:03d}", points=points[idx], colors=colors[idx],
                point_size=point_size, visible=(f == 0)))

    src_gl = cam_c2w @ gl2cv  # CV c2w -> add_frustums 가 먹는 GL c2w
    src_cams = [(h, 1.0) for h in add_frustums(
        server, "/cam_source", src_gl, float(intrinsics[0, 0, 0]),
        float(intrinsics[0, 1, 1]), width, height, src_color, cam_scale,
        downsample=args.cam_stride)]
    src_now = add_frustums(server, "/cam_source_now", src_gl[:1], float(intrinsics[0, 0, 0]),
                           float(intrinsics[0, 1, 1]), width, height, src_now_color,
                           cam_scale * 1.6)
    src_cams += [(h, 1.6) for h in src_now]

    focal = float(intrinsics[0, 0, 0])

    # 경로선은 **색을 갈아끼울 수가 없다** — viser 1.1.0 의 LineSegmentsHandle 이 내놓는 건
    # thickness/visible/wxyz/position 뿐이고 colors 는 없다. 그래서 색 피커가 움직이면 같은
    # 이름으로 remove 후 다시 add 한다 (프러스텀은 `.color` 대입이 먹으니 그쪽은 그대로 둔다).
    # 이름이 같으므로 재생성해도 씬 트리에 중복 노드가 남지 않는다.
    # 굵기는 반대다 — `thickness` 는 갱신되는 프로퍼티라 다시 그릴 필요가 없다. 그래서 선마다
    # **기준 굵기**를 따로 들고(`line_base`) 슬라이더는 거기 곱하는 배율로 둔다. 소스 0.08 /
    # plan 0.10 처럼 원래 다른 값을 쓰던 비율이 슬라이더를 움직여도 유지된다.
    lines, line_base = {}, {}
    # 배율·표시 여부를 GUI handle 이 아니라 dict 로 들고 있는 이유: 소스 경로선은 GUI 를 만들기
    # **전에** 그려지는데, 그때 위젯을 읽으려 하면 아직 없다. 위젯 콜백이 이 값을 갱신한다.
    # `show` 기본이 False 인 이유: 49프레임 경로선 3~4개가 프러스텀보다 굵게 화면을 덮어
    # 정작 봐야 할 카메라 자세가 안 보인다. `--paths_on` 이나 GUI 체크박스로 켠다.
    lw = {"path": 1.0, "show": bool(args.paths_on)}

    def draw_line(key, name, segments, color, thickness):
        drop_line(key)
        line_base[key] = float(thickness)
        lines[key] = server.scene.add_line_segments(
            name, segments, colors=color, thickness=float(thickness) * path_lw(),
            visible=bool(lw["show"]))
        return lines[key]

    def drop_line(key):
        old = lines.pop(key, None)
        line_base.pop(key, None)
        if old is not None:
            old.remove()

    def path_lw():
        return float(lw["path"])

    # `handle.line_width = v` 를 쓰면 안 된다 — viser 1.1.0 에서 그건 `thickness` 의 **폐기된
    # 별칭**이고, 대입하는 순간 `thickness_units` 를 `"screen"` 으로 못 박는다 (예전 line_width
    # 가 픽셀이었으니 그 뜻을 지키려고). world 0.02 로 만든 선에 0.02 를 넣으면 0.02 **픽셀**이
    # 돼서 화면에서 사라진다. `thickness` 에 직접 넣으면 단위가 world 그대로다.
    def apply_path_lw():
        for key, handle in lines.items():
            handle.thickness = line_base[key] * path_lw()

    def apply_path_vis():
        for handle in lines.values():
            handle.visible = bool(lw["show"])

    # 소스 카메라 경로도 선으로 — plan 이 떠는지 판단하려면 "원래 소스는 얼마나 떠는가"가
    # 있어야 한다. snowboard 소스는 |jerk| p95 가 plan 의 8배다.
    draw_line("src", "/src_path", path_segments(cam_c2w[:, :3, 3]), src_color, cam_scale * 0.08)
    src_jerk = jerk_px(cam_c2w[:, :3, 3], z_med, focal)

    banks = list(args.banks) or ([args.bank] if args.bank else [])
    motions = load_motions(args.out, args.video, banks, args.variant) if banks else []
    graph_path = path.join(args.out, args.video, "scene_graph.json")
    tracks = subject_tracks_world(graph_path)

    # ---- probe 카메라: 끌어서 옮기는 프러스텀 ----
    # 소스 카메라 자리는 씬이 정해준 것이라 "여기서 보면 어떻게 보이나"를 물어볼 수가 없다.
    # gizmo 를 띄우고 프러스텀을 그 자세에 맞춘다 — gizmo 의 (wxyz, position) 이 곧 그 카메라의
    # OpenCV c2w 라서, 마음에 드는 자리를 찾았을 때 읽은 숫자를 그대로 카메라 pose 로 쓸 수 있다
    # (add_frustums 가 `c2w_gl @ _GL2CV` 로 만드는 것과 같은 규약).
    #
    # 프러스텀은 gizmo 의 **자식이 아니라 형제**다. 자식으로 붙이면 부모를 숨길 때 자식도 같이
    # 사라져서 "화살표만 끄고 프러스텀만 보기"가 안 된다. 대신 드래그마다 pose 를 복사한다.
    #
    # fov 는 소스 intrinsics 를 안 쓴다. camel 소스는 vfov 16.4°(fy 2499 @ 720p) 짜리 망원이라
    # 그 화각으로 그리면 프러스텀이 바늘처럼 길어져 자리를 가늠할 수가 없다. 기본은 보통 렌즈
    # 화각(vfov 60°)이고 슬라이더로 소스 값까지 되돌릴 수 있다.
    src_vfov_deg = float(np.degrees(2 * np.arctan2(height / 2, float(intrinsics[0, 1, 1]))))
    probes = []                      # [{name, gizmo, cam}] — `add camera` 로 늘어난다
    # 번호는 리스트 길이가 아니라 **단조 증가 카운터**로 붙인다. 길이를 쓰면 가운데 것을 지운 뒤
    # 새로 만들 때 살아있는 노드와 이름이 겹쳐 그 노드를 덮어쓴다.
    probe_seq = {"n": 0}

    def make_probe(wxyz, position, fov_deg: float, size: float, show_cam: bool, show_giz: bool):
        index = probe_seq["n"]
        probe_seq["n"] += 1
        gizmo = server.scene.add_transform_controls(
            f"/probe{index}", scale=cam_scale * 2.5, line_width=2.0,
            wxyz=wxyz, position=position, visible=show_giz)
        cam = server.scene.add_camera_frustum(
            f"/probe{index}_cam", fov=float(np.radians(fov_deg)), aspect=width / height,
            scale=size, color=PROBE_COLORS[index % len(PROBE_COLORS)],
            wxyz=wxyz, position=position, visible=show_cam)
        probes.append({"name": f"probe {index}", "gizmo": gizmo, "cam": cam})
        return probes[-1]

    def sync_probe(probe):
        """gizmo -> 프러스텀. 형제라서 자동으로 안 따라온다."""
        probe["cam"].wxyz = probe["gizmo"].wxyz
        probe["cam"].position = probe["gizmo"].position

    aim_track = tracks[sorted(tracks)[0]] if tracks else None
    cloud_center = {"value": None}     # 1.5M 점 median 은 버튼 누를 때 한 번만

    def aim_target(frame: int):
        """probe 가 바라볼 점. 동적 subject 가 있으면 그 프레임 위치, 없으면 점군 중앙값."""
        if aim_track is not None and frame < len(aim_track):
            return np.asarray(aim_track[frame], dtype=float)
        if cloud_center["value"] is None:
            cloud_center["value"] = (np.median(points[static_pick], axis=0).astype(float)
                                     if len(static_pick) else np.zeros(3))
        return cloud_center["value"]

    def look_at_R(position: np.ndarray, target: np.ndarray):
        """OpenCV c2w 회전 [right | down | fwd]. roll 은 world up 기준 0 이다."""
        fwd = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
        norm = float(np.linalg.norm(fwd))
        if norm < 1e-9:
            return np.eye(3)
        fwd /= norm
        down_hint = -np.asarray(UP_VECTORS[args.up], dtype=float)
        right = np.cross(down_hint, fwd)
        if np.linalg.norm(right) < 1e-6:            # 정확히 위/아래를 볼 때 축이 무너진다
            right = np.cross(np.array([1.0, 0.0, 0.0]), fwd)
        right /= np.linalg.norm(right)
        return np.stack([right, np.cross(fwd, right), fwd], axis=1)

    # ---- scene graph 오버레이: OBB(정적 1개 / 동적 프레임별) + 지면 격자 ----
    # 동적 OBB 를 프레임별 handle 로 쪼개는 이유는 동적 점군과 같다 — 49개를 한꺼번에 그리면
    # 움직이는 박스가 겹쳐서 아무것도 안 보인다. refresh() 가 점군과 같은 슬라이더로 껐다 켠다.
    #
    # handle 을 그대로 들지 않고 **entry dict** 로 감싸는 이유는 색 때문이다. viser 1.1.0 선
    # handle 은 `line_width` 만 갱신되고 `colors` 는 없어서, 색을 바꾸려면 지웠다 다시 그려야
    # 한다. snowboard 는 동적 5노드 x 49프레임 + 정적 4 = 249개라 피커를 끌 때마다 249번
    # 다시 그리면 브라우저가 멎는다. 그래서 선분 좌표를 entry 에 들고 있다가 **보이는 순간에만**
    # 다시 그린다 (`obb_version` 이 뒤처진 entry 가 stale). `current frame` 에서 실제로 보이는
    # 건 9개뿐이라 드래그가 가볍다.
    obb_static, obb_dyn, obb_labels, ground_handle, graph_rows = [], [[] for _ in range(num_frames)], [], None, []
    obb_version = {"n": 0}                    # 색/굵기가 바뀔 때마다 +1

    def obb_entry(name, segments, moving, visible):
        return {"name": name, "seg": segments, "moving": moving, "handle": None,
                "version": -1, "visible": bool(visible)}

    if args.obb and path.isfile(graph_path):
        with open(graph_path, encoding="utf-8") as file:
            graph = json.load(file)
        T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
        T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
        for node in graph["nodes"]:
            moving = bool(node.get("moving")) and bool(node.get("track"))
            extent = node["obb"]["extent"]
            if moving:
                centers = np.asarray(node["track"]["center_smooth"], dtype=float)
                yaws = np.asarray(node["track"]["yaw"], dtype=float)
                for f in range(min(num_frames, len(centers))):
                    obb_dyn[f].append(obb_entry(
                        f"/obb/{node['id']}/f{f:03d}",
                        obb_segments_world(centers[f], extent, yaw_to_R(yaws[f]), T_wg),
                        True, f == 0))
                anchor_g = centers[0]
            else:
                obb_static.append(obb_entry(
                    f"/obb/{node['id']}",
                    obb_segments_world(node["obb"]["center"], extent,
                                       np.asarray(node["obb"]["R"], dtype=float), T_wg),
                    False, True))
                anchor_g = np.asarray(node["obb"]["center"], dtype=float)
            top = np.asarray(anchor_g, float) + np.array([0.0, 0.0, float(extent[2]) / 2 + 0.01])
            obb_labels.append(server.scene.add_label(
                f"/obb_label/{node['id']}", f"{node['id']} {node.get('label', '')}",
                position=(top @ T_wg[:3, :3].T + T_wg[:3, 3]).astype(np.float32)))
            graph_rows.append((node["id"], node.get("label", ""), moving))
        if args.ground_grid:
            cams_g = cam_c2w[:, :3, 3] @ T_gw[:3, :3].T + T_gw[:3, 3]
            nodes_g = np.asarray([n["obb"]["center"] for n in graph["nodes"]], dtype=float)
            span = np.concatenate([cams_g[:, :2], nodes_g[:, :2]], axis=0)
            half = float(max(np.ptp(span, axis=0).max(), 1e-3)) * 1.2
            ground_handle = server.scene.add_line_segments(
                "/ground_grid",
                ground_grid_world(float(graph["ground"]["ground_z"]), T_wg,
                                  span.mean(axis=0), half),
                colors=(110, 120, 140), thickness=cam_scale * 0.03)

    def obb_apply(entry):
        """entry 를 현재 색·굵기로 (다시) 그린다. 색을 못 갈아끼우니 remove 후 add."""
        if entry["handle"] is not None:
            entry["handle"].remove()
        color = (tuple(int(c) for c in gui_obb_dyn_color.value) if entry["moving"]
                 else tuple(int(c) for c in gui_obb_stat_color.value))
        entry["handle"] = server.scene.add_line_segments(
            entry["name"], entry["seg"], colors=color,
            thickness=cam_scale * float(gui_obb_lw.value), visible=entry["visible"])
        entry["version"] = obb_version["n"]

    def obb_show(entry, visible: bool):
        """보이게 하는 순간에만 stale 을 갚는다 — 249개를 매 드래그마다 다시 그리지 않으려고."""
        entry["visible"] = bool(visible)
        if visible and entry["version"] != obb_version["n"]:
            obb_apply(entry)
        elif entry["handle"] is not None:
            entry["handle"].visible = bool(visible)

    # 현재 선택된 motion 의 handle 들. 갈아끼울 때 통째로 remove 한다 — 같은 이름으로 덮어쓰면
    # 프러스텀 수가 줄어들 때(다른 F) 이전 것이 남는다.
    state = {"handles": [], "now": [], "c2w": None, "label": "", "cams": []}
    # pin 된 target 카메라들. {label: {"handles", "now", "c2w", "cams", "color"}}.
    # 활성 motion(state) 과 **따로** 들고 있는 이유: 활성은 슬라이더로 계속 갈아끼우는 자리라
    # 거기에 얹으면 pin 이 매번 지워진다. pin 은 갈아끼워도 남는 것이 존재 이유다.
    pins = {}
    pin_seq = {"n": 0}                       # 색 순환 카운터. 지웠다 다시 켜도 색이 안 겹치게.

    # 프러스텀 크기는 씬마다 맞는 값이 다르다 — camel 소스는 49프레임 경로가 0.167 u 뿐이라
    # 기본 0.03·z_med 로는 점군에 묻히고, 카메라가 크게 도는 씬에선 같은 값이 화면을 덮는다.
    # handle 을 지웠다 다시 만들면 깜빡이므로 `scale` prop 만 갈아끼운다. 배율(now=1.6)을 같이
    # 들고 있어야 "현재 프레임" 프러스텀이 큰 구분이 슬라이더를 움직여도 유지된다.
    def apply_cam_scale():
        value = float(gui_cam.value)
        pin_cams = [pair for pin in pins.values() for pair in pin["cams"]]
        for handle, ratio in src_cams + state["cams"] + pin_cams:
            handle.scale = value * ratio

    def apply_cam_lw():
        """프러스텀 선 굵기. 크기(scale)와 **다른 축**이다 — 작은 프러스텀을 굵게 그려야
        점군에 안 묻히는 씬이 있고, 반대로 큰 프러스텀은 가늘어야 뒤가 보인다.
        `thickness` 는 갱신되는 프로퍼티라 다시 그릴 필요가 없다 (색과 다르다).
        **`line_width` 에 넣으면 안 된다** — 폐기된 별칭이라 단위가 screen(픽셀) 로 못 박혀
        world 0.02 가 0.02 픽셀이 되고, 프러스텀이 통째로 안 보인다."""
        value = float(gui_cam_lw.value)
        pin_cams = [pair for pin in pins.values() for pair in pin["cams"]]
        for handle, _ in src_cams + state["cams"] + pin_cams:
            handle.thickness = value
        for probe in probes:
            probe["cam"].thickness = value

    def plan_colors():
        """(전 프레임, 현재 프레임) target 색. 피커가 단일 출처다."""
        return (tuple(int(c) for c in gui_plan_color.value),
                tuple(int(c) for c in gui_plan_now_color.value))

    def select(index: int):
        label, plan_c2w, row = motions[int(index)]
        for handle in state["handles"]:
            handle.remove()
        plan_gl = plan_c2w @ gl2cv
        # 색은 상수가 아니라 **피커의 현재 값**을 읽는다 — 안 그러면 motion 을 갈아끼우는 순간
        # 사용자가 고른 색이 기본색으로 되돌아간다.
        color, now_color = plan_colors()
        plan_cams = list(add_frustums(server, "/cam_plan", plan_gl, focal,
                                      float(intrinsics[0, 1, 1]), width, height, color,
                                      cam_scale, downsample=args.cam_stride))
        handles = list(plan_cams)
        # 경로선은 handles 에 안 담는다 — draw_line 이 키 하나로 이전 것을 지우므로, 여기에도
        # 담으면 갈아끼울 때 같은 handle 을 두 번 remove 하게 된다.
        draw_line("plan", "/plan_path", path_segments(plan_c2w[:, :3, 3]), color,
                  cam_scale * 0.10)
        now = list(add_frustums(server, "/cam_plan_now", plan_gl[:1], focal,
                                float(intrinsics[0, 1, 1]), width, height, now_color,
                                cam_scale * 1.6))
        track = tracks.get(str(row.get("anchor_id", "")))
        drop_line("track")
        if track is not None and len(track) > 1:
            draw_line("track", "/subject_track", path_segments(track), (60, 220, 90),
                      cam_scale * 0.08)
        state.update(handles=handles + now, now=now, c2w=plan_c2w, label=label,
                     cams=[(h, 1.0) for h in plan_cams] + [(h, 1.6) for h in now])
        gui_info.value = motion_report(label, row, plan_c2w, track, z_med, focal, src_jerk)
        apply_cam_scale()
        apply_cam_lw()
        refresh()

    def pin_node(label: str):
        """라벨 -> viser 노드 경로. `/` 가 계층 구분자라 라벨의 `bank/variant` 를 그대로 못 쓴다."""
        return "/pin/" + "".join(c if c.isalnum() or c in "_-" else "_" for c in label)

    def add_pin(label: str):
        """motion 하나를 고정 색으로 그려 두고 활성 motion 이 바뀌어도 남긴다."""
        if label in pins or len(pins) >= int(args.max_pins):
            return
        index = [m[0] for m in motions].index(label)
        _, plan_c2w, row = motions[index]
        color = PIN_COLORS[pin_seq["n"] % len(PIN_COLORS)]
        pin_seq["n"] += 1
        node, plan_gl = pin_node(label), plan_c2w @ gl2cv
        cams = list(add_frustums(server, node + "/cam", plan_gl, focal,
                                 float(intrinsics[0, 1, 1]), width, height, color,
                                 cam_scale, downsample=args.cam_stride))
        # 현재 프레임 프러스텀만 1.6배로 크게 — pin 을 여러 대 켜면 선이 엉켜서 "이 변이가 지금
        # 어디를 보고 있나"를 경로만으로는 못 읽는다. 같은 색이라 소속은 유지된다.
        now = list(add_frustums(server, node + "/now", plan_gl[:1], focal,
                                float(intrinsics[0, 1, 1]), width, height, color, cam_scale * 1.6))
        # 경로선은 `lines` 로 관리한다 (굵기 슬라이더가 여기만 본다). pin["handles"] 에는 안
        # 담는다 — 담으면 remove_pin 이 drop_line 과 겹쳐 같은 handle 을 두 번 지운다.
        draw_line(("pin", label), node + "/path", path_segments(plan_c2w[:, :3, 3]), color,
                  cam_scale * 0.10)
        pins[label] = {"handles": cams + now, "now": now, "c2w": plan_c2w,
                       "cams": [(h, 1.0) for h in cams] + [(h, 1.6) for h in now],
                       "color": color, "row": row}
        apply_cam_scale()
        apply_cam_lw()

    def remove_pin(label: str):
        pin = pins.pop(label, None)
        if pin is None:
            return
        for handle in pin["handles"]:
            handle.remove()
        drop_line(("pin", label))

    def pin_report():
        if not pins:
            return "(pin 없음 — pin current)"
        return "\n".join(f"#{c[0]:3d},{c[1]:3d},{c[2]:3d}  {lab}"
                         for lab, c in ((k, v["color"]) for k, v in pins.items()))

    with server.gui.add_folder("motion"):
        # 슬라이더와 드롭다운은 같은 select() 를 부른다. 서로를 갱신하므로 재진입 가드를 둔다 —
        # 안 두면 슬라이더->드롭다운->슬라이더로 콜백이 한 번 더 돈다.
        gui_motion = server.gui.add_slider("motion", min=0, max=max(len(motions) - 1, 0), step=1,
                                           initial_value=0, disabled=not motions)
        gui_pick = server.gui.add_dropdown("variant", options=[m[0] for m in motions] or ["-"],
                                           initial_value=(motions[0][0] if motions else "-"),
                                           disabled=not motions)
        gui_info = server.gui.add_text("info", initial_value="", multiline=True, disabled=True)
        # pin: 활성 motion 을 갈아끼워도 남는 target 카메라. 슬라이더 하나로는 "A 와 B 중 어느
        # 쪽이 더 도나"를 못 본다 — 갈아끼우는 순간 비교 대상이 사라지기 때문이다.
        gui_pin_add = server.gui.add_button("pin current")
        gui_pin_del = server.gui.add_button("unpin current")
        gui_pin_clear = server.gui.add_button("clear pins")
        gui_pin_info = server.gui.add_text("pinned", initial_value="", multiline=True,
                                           disabled=True)

    with server.gui.add_folder("view"):
        gui_frame = server.gui.add_slider("frame", min=0, max=num_frames - 1, step=1, initial_value=0)
        gui_play = server.gui.add_checkbox("play", initial_value=False)
        # 재생 속도. 예전에는 루프가 0.08 s 로 박혀 있어서 "빨라서 못 보겠다"를 못 고쳤다.
        gui_fps = server.gui.add_slider("fps", min=1, max=30, step=1, initial_value=12)
        gui_static = server.gui.add_checkbox("static cloud", initial_value=True)
        gui_dyn = server.gui.add_checkbox("dynamic cloud", initial_value=True)
        # 동적 점군 + 동적 OBB 를 **몇 프레임 띄울지**. 체크박스 하나(현재/전부)로는 가운데가
        # 없었다 — 전부 띄우면 49겹이라 아무것도 안 보이고, 한 장만 띄우면 "이 물체가 어디로
        # 갔나"가 안 보인다. 구간 + 간격이 그 사이를 만든다 (잔상처럼 떨어져 보인다).
        # 기본은 `current frame` = 예전 동작 그대로.
        gui_dyn_mode = server.gui.add_dropdown(
            "dynamic frames (points+OBB)", options=["current frame", "interval", "all frames"],
            initial_value="current frame")
        gui_span_lo = server.gui.add_slider("range start", min=0, max=num_frames - 1, step=1,
                                            initial_value=0)
        gui_span_hi = server.gui.add_slider("range end", min=0, max=num_frames - 1, step=1,
                                            initial_value=num_frames - 1)
        # 간격은 `all frames` 에서도 먹는다 — 전 프레임 잔상을 실제로 읽게 만드는 손잡이가 이거다.
        gui_frame_step = server.gui.add_slider("frame interval (every N)", min=1,
                                               max=max(2, num_frames // 2), step=1,
                                               initial_value=5)
        gui_ground = server.gui.add_checkbox("ground grid", initial_value=True,
                                             disabled=ground_handle is None)
        gui_size = server.gui.add_slider("point size", min=point_size * 0.25, max=point_size * 4.0,
                                         step=point_size * 0.05, initial_value=point_size)
        gui_cam = server.gui.add_slider("camera size", min=cam_scale * 0.1, max=cam_scale * 10.0,
                                        step=cam_scale * 0.05, initial_value=cam_scale)
        # 프러스텀 선 굵기(world 단위). viser 기본 0.02 를 초기값으로 둔다.
        gui_cam_lw = server.gui.add_slider("camera thickness", min=0.002,
                                           max=max(0.05, cam_scale), step=0.002,
                                           initial_value=min(0.02, max(0.05, cam_scale)))
        # 경로선 굵기 **배율**. 소스 0.08 / plan 0.10 처럼 원래 다른 값을 쓰던 비율을 유지한다.
        # 경로선 전체(소스·target·pin·subject track). 기본 꺼짐 — 49프레임 꺾은선이
        # 프러스텀보다 굵어 카메라 자세를 덮는다. 굵기 슬라이더는 켠 상태에서만 의미가 있다.
        gui_path = server.gui.add_checkbox("paths", initial_value=bool(args.paths_on))
        gui_path_lw = server.gui.add_slider("path thickness x", min=0.2, max=5.0, step=0.1,
                                            initial_value=1.0)

    with server.gui.add_folder("obb"):
        # 예전엔 체크박스 하나로 OBB·라벨을 통째로 껐다 켰다. 동적/정적/라벨을 따로 끄는 게
        # 필요한 이유: 정적 박스가 씬을 덮어 동적 박스를 가리는 씬이 있고, 라벨은 노드가 9개만
        # 돼도 화면 글자가 겹친다.
        _has_obb = bool(obb_static or any(obb_dyn))
        gui_obb = server.gui.add_checkbox("scene graph OBB", initial_value=True,
                                          disabled=not _has_obb)
        gui_obb_dyn_on = server.gui.add_checkbox("dynamic OBB", initial_value=True,
                                                 disabled=not any(obb_dyn))
        gui_obb_stat_on = server.gui.add_checkbox("static OBB", initial_value=True,
                                                  disabled=not obb_static)
        gui_obb_labels = server.gui.add_checkbox("labels", initial_value=True,
                                                 disabled=not obb_labels)
        gui_obb_dyn_color = server.gui.add_rgb("dynamic color", initial_value=obb_dyn_color,
                                               disabled=not any(obb_dyn))
        gui_obb_stat_color = server.gui.add_rgb("static color", initial_value=obb_static_color,
                                                disabled=not obb_static)
        # cam_scale 배율이다 (world 절대값이 아니라) — 씬마다 scale 이 100배 다르다.
        gui_obb_lw = server.gui.add_slider("OBB thickness", min=0.01, max=0.60, step=0.01,
                                           initial_value=0.06, disabled=not _has_obb)

    with server.gui.add_folder("camera colors"):
        # gt(소스) / pred(활성 target) 두 계열. 기본색이 씬에 따라 안 보일 때가 있어서 손잡이를
        # 둔다 — 회색 소스는 밝은 점군에 묻히고, 주황 target 은 노을·모래 씬에서 배경과 붙는다.
        # 계열마다 피커가 둘인 이유: "현재 프레임" 프러스텀이 크기(1.6배)**와 색**으로 구분되는데,
        # 하나로 합치면 frame 슬라이더를 밀 때 어느 것이 지금인지 다시 못 읽는다.
        # 초기값은 `--src_color` 등 기동 플래그를 그대로 받는다.
        gui_src_color = server.gui.add_rgb("source (gt)", initial_value=src_color)
        gui_src_now_color = server.gui.add_rgb("source now", initial_value=src_now_color)
        gui_plan_color = server.gui.add_rgb("target (pred)", initial_value=plan_color,
                                            disabled=not motions)
        gui_plan_now_color = server.gui.add_rgb("target now", initial_value=plan_now_color,
                                                disabled=not motions)

    with server.gui.add_folder("probe camera"):
        # frustum 과 gizmo 를 따로 끈다. 자리를 정하고 나면 화살표가 프러스텀을 가려서
        # "이 카메라가 뭘 보나"를 확인할 수가 없다.
        # 기본 꺼짐 (`--probe_on` 으로 예전 동작). probe 는 "여기서 보면 어떻게 보이나"를
        # 물을 때만 쓰는 도구인데, gizmo 화살표가 원점 근처 소스 프러스텀을 통째로 가린다.
        gui_probe = server.gui.add_checkbox("show frustum", initial_value=bool(args.probe_on))
        gui_probe_giz = server.gui.add_checkbox("show gizmo", initial_value=bool(args.probe_on))
        gui_probe_size = server.gui.add_slider(
            "probe size", min=cam_scale * 0.1, max=cam_scale * 10.0, step=cam_scale * 0.05,
            initial_value=cam_scale * 1.6)
        # 소스 화각(camel 16.4°)도 슬라이더 범위 안에 들어오게 하한을 10° 로 둔다.
        gui_probe_fov = server.gui.add_slider("probe vfov (deg)", min=10.0, max=120.0, step=1.0,
                                              initial_value=PROBE_FOV_DEG)
        gui_probe_pick = server.gui.add_dropdown("active", options=["probe 0"],
                                                 initial_value="probe 0")
        # 크기·화각은 전 probe 공통이지만 **색만 활성 probe 한 대**에 걸린다 — 여러 대를
        # 구분하려고 두는 손잡이라 다 같은 색으로 칠하면 의미가 없다.
        gui_probe_color = server.gui.add_rgb("color (active)", initial_value=PROBE_COLORS[0])
        gui_probe_add = server.gui.add_button("add camera")
        gui_probe_del = server.gui.add_button("remove active")
        gui_probe_snap = server.gui.add_button("snap to source frame")
        gui_probe_aim = server.gui.add_button("aim at subject")
        gui_probe_info = server.gui.add_text("pose", initial_value="", multiline=True,
                                             disabled=True)

    def shown_frames():
        """지금 띄울 프레임 집합. `current frame` / `interval` / `all frames` 세 갈래.

        `interval` 은 구간 `[lo, hi]` 를 `step` 칸마다 — 구간은 "어디부터 어디까지", 간격은
        "그중 몇 장 건너뛰고"로 **다른 축**이다. 현재 프레임은 구간 밖이어도 늘 포함한다
        (frame 슬라이더를 밀었는데 화면이 비면 슬라이더가 고장난 것처럼 보인다).
        """
        f = int(gui_frame.value)
        mode = str(gui_dyn_mode.value)
        step = max(1, int(gui_frame_step.value))
        if mode == "current frame":
            return {f}
        lo, hi = (0, num_frames - 1) if mode == "all frames" else \
            (min(int(gui_span_lo.value), int(gui_span_hi.value)),
             max(int(gui_span_lo.value), int(gui_span_hi.value)))
        return {i for i in range(lo, hi + 1) if (i - lo) % step == 0} | {f}

    def refresh():
        f = int(gui_frame.value)
        frames = shown_frames()
        show_dyn = bool(gui_dyn.value)
        for i, handle in enumerate(dyn_handles):
            handle.visible = show_dyn and (i in frames)
        if static_handle is not None:
            static_handle.visible = bool(gui_static.value)
        show_obb = bool(gui_obb.value)
        for entry in obb_static:
            obb_show(entry, show_obb and bool(gui_obb_stat_on.value))
        for handle in obb_labels:
            handle.visible = show_obb and bool(gui_obb_labels.value)
        # 동적 OBB 는 점군과 **같은 규칙**으로 켠다 — 박스만 전 프레임 켜두면 박스가 점군보다
        # 앞선 프레임에 있어도 어긋난 걸 못 알아챈다.
        show_dyn_obb = show_obb and bool(gui_obb_dyn_on.value)
        for i, entries in enumerate(obb_dyn):
            for entry in entries:
                obb_show(entry, show_dyn_obb and i in frames)
        if ground_handle is not None:
            ground_handle.visible = bool(gui_ground.value)
        src_now[0].wxyz, src_now[0].position = _pose(src_gl[f], gl2cv)
        if state["now"]:
            state["now"][0].wxyz, state["now"][0].position = _pose(
                state["c2w"][f] @ gl2cv, gl2cv)
        for pin in pins.values():
            pin["now"][0].wxyz, pin["now"][0].position = _pose(pin["c2w"][f] @ gl2cv, gl2cv)

    guard = {"busy": False}

    def switch(index: int):
        if guard["busy"] or not motions:
            return
        guard["busy"] = True
        try:
            index = int(index) % len(motions)
            gui_motion.value, gui_pick.value = index, motions[index][0]
            select(index)
        finally:
            guard["busy"] = False

    def pin_current(_event=None):
        if motions:
            add_pin(motions[int(gui_motion.value) % len(motions)][0])
            gui_pin_info.value = pin_report()
            refresh()

    def unpin_current(_event=None):
        if motions:
            remove_pin(motions[int(gui_motion.value) % len(motions)][0])
            gui_pin_info.value = pin_report()

    def clear_pins(_event=None):
        for label in list(pins):
            remove_pin(label)
        gui_pin_info.value = pin_report()

    gui_pin_add.on_click(pin_current)
    gui_pin_del.on_click(unpin_current)
    gui_pin_clear.on_click(clear_pins)

    gui_motion.on_update(lambda _: switch(gui_motion.value))
    gui_pick.on_update(lambda _: switch([m[0] for m in motions].index(gui_pick.value)))
    for widget in (gui_frame, gui_static, gui_dyn, gui_dyn_mode, gui_span_lo, gui_span_hi,
                   gui_frame_step, gui_obb, gui_obb_dyn_on, gui_obb_stat_on, gui_obb_labels,
                   gui_ground):
        widget.on_update(lambda _: refresh())

    # OBB 색·굵기: version 을 올리고 refresh() 를 부르면 **지금 보이는 것만** 다시 그려진다.
    # 숨어 있는 것들은 다음에 켜질 때 갚는다 (obb_show). 전 프레임을 켜 둔 상태에서 피커를
    # 끌면 그때는 249개가 다 보이므로 실제로 249번 다시 그린다 — 느린 건 그 조합뿐이다.
    def repaint_obb(_event=None):
        obb_version["n"] += 1
        refresh()

    gui_obb_dyn_color.on_update(repaint_obb)
    gui_obb_stat_color.on_update(repaint_obb)
    gui_obb_lw.on_update(repaint_obb)

    @gui_size.on_update
    def _(_event):
        if static_handle is not None:
            static_handle.point_size = float(gui_size.value)
        for handle in dyn_handles:
            handle.point_size = float(gui_size.value)

    gui_cam.on_update(lambda _: apply_cam_scale())
    gui_cam_lw.on_update(lambda _: apply_cam_lw())

    @gui_path_lw.on_update
    def _(_event):
        lw["path"] = float(gui_path_lw.value)
        apply_path_lw()

    @gui_path.on_update
    def _(_event):
        lw["show"] = bool(gui_path.value)
        apply_path_vis()

    # 색 갈아끼우기. 프러스텀은 `.color` 대입, 경로선은 remove + re-add (draw_line).
    # 계열 안에서 "현재 프레임"인지는 `(handle, ratio)` 의 ratio 로 가른다 — 크기 배율과 색
    # 구분이 같은 한 곳에서 나와야 둘이 어긋나지 않는다.
    def repaint_src(_event=None):
        color = tuple(int(c) for c in gui_src_color.value)
        now_color = tuple(int(c) for c in gui_src_now_color.value)
        for handle, ratio in src_cams:
            handle.color = now_color if ratio > 1.0 else color
        draw_line("src", "/src_path", path_segments(cam_c2w[:, :3, 3]), color, cam_scale * 0.08)

    def repaint_plan(_event=None):
        color, now_color = plan_colors()
        for handle, ratio in state["cams"]:
            handle.color = now_color if ratio > 1.0 else color
        if state["c2w"] is not None:
            draw_line("plan", "/plan_path", path_segments(state["c2w"][:, :3, 3]), color,
                      cam_scale * 0.10)

    gui_src_color.on_update(repaint_src)
    gui_src_now_color.on_update(repaint_src)
    gui_plan_color.on_update(repaint_plan)
    gui_plan_now_color.on_update(repaint_plan)

    def active_probe():
        """드롭다운이 가리키는 probe. 지워진 이름이 남아 있을 수 있으니 없으면 마지막 것."""
        for probe in probes:
            if probe["name"] == gui_probe_pick.value:
                return probe
        return probes[-1] if probes else None

    def probe_report():
        """gizmo 자세를 그대로 숫자로 — 이 값이 곧 쓸 수 있는 OpenCV c2w 다."""
        probe = active_probe()
        if probe is None:
            gui_probe_info.value = "(probe 없음 — add camera)"
            return
        p = np.asarray(probe["gizmo"].position, dtype=float)
        R = _R_of(probe["gizmo"].wxyz)
        frame = int(gui_frame.value)
        target = aim_target(frame)
        # 피커를 활성 probe 색으로 되돌린다. 이 대입이 on_update 를 다시 부르지만 같은 색을
        # 같은 probe 에 칠하는 것이라 무해하다.
        gui_probe_color.value = tuple(int(c) for c in probe["cam"].color)
        gui_probe_info.value = "\n".join([
            f"{probe['name']}   vfov {np.degrees(float(probe['cam'].fov)):.1f} deg",
            f"pos     {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}",
            f"fwd     {R[0, 2]:+.4f} {R[1, 2]:+.4f} {R[2, 2]:+.4f}",
            f"up      {-R[0, 1]:+.4f} {-R[1, 1]:+.4f} {-R[2, 1]:+.4f}",
            f"dist to subject  {float(np.linalg.norm(target - p)):.4f} u",
            f"dist to src[{frame}]   "
            f"{float(np.linalg.norm(cam_c2w[frame, :3, 3] - p)):.4f} u",
        ])

    def refresh_probe_list(select: str = ""):
        """드롭다운 옵션을 현재 probe 목록으로 맞춘다. 비면 placeholder 한 줄을 둔다 —
        viser 드롭다운은 빈 options 를 못 받는다."""
        names = [probe["name"] for probe in probes] or ["-"]
        gui_probe_pick.options = names
        gui_probe_pick.value = select if select in names else names[-1]
        probe_report()

    def bind_probe(probe):
        """드래그마다 프러스텀을 gizmo 자세로 따라오게 한다. 형제 노드라 이 복사가 없으면
        화살표만 움직이고 프러스텀은 제자리에 남는다."""
        probe["gizmo"].on_update(lambda _: (sync_probe(probe), probe_report()))

    def probe_snap(_event=None):
        probe = active_probe()
        if probe is None:
            return
        probe["gizmo"].wxyz, probe["gizmo"].position = _pose(src_gl[int(gui_frame.value)], gl2cv)
        sync_probe(probe)
        probe_report()

    def probe_aim(_event=None):
        probe = active_probe()
        if probe is None:
            return
        p = np.asarray(probe["gizmo"].position, dtype=float)
        probe["gizmo"].wxyz = _wxyz(look_at_R(p, aim_target(int(gui_frame.value))))
        sync_probe(probe)
        probe_report()

    def probe_add(_event=None):
        # 새 카메라는 **활성 probe 자리에서 한 발짝 옆**에 둔다. 같은 자리에 겹쳐 놓으면 방금
        # 만든 게 어느 것인지 못 고르고, 원점에 두면 씬 밖일 수 있다.
        base = active_probe()
        if base is None:
            wxyz, position = _pose(src_gl[int(gui_frame.value)], gl2cv)
        else:
            wxyz = np.asarray(base["gizmo"].wxyz, dtype=float)
            position = (np.asarray(base["gizmo"].position, dtype=float)
                        + _R_of(wxyz)[:, 0] * cam_scale * 3.0)
        probe = make_probe(wxyz, np.asarray(position, dtype=np.float32),
                           float(gui_probe_fov.value), float(gui_probe_size.value),
                           bool(gui_probe.value), bool(gui_probe_giz.value))
        bind_probe(probe)
        refresh_probe_list(probe["name"])

    def probe_remove(_event=None):
        probe = active_probe()
        if probe is None:
            return
        probe["cam"].remove()
        probe["gizmo"].remove()
        probes.remove(probe)
        refresh_probe_list()

    @gui_probe.on_update
    def _(_event):
        for probe in probes:
            probe["cam"].visible = bool(gui_probe.value)

    @gui_probe_giz.on_update
    def _(_event):
        for probe in probes:
            probe["gizmo"].visible = bool(gui_probe_giz.value)

    @gui_probe_color.on_update
    def _(_event):
        probe = active_probe()
        if probe is not None:
            probe["cam"].color = tuple(int(c) for c in gui_probe_color.value)

    @gui_probe_size.on_update
    def _(_event):
        for probe in probes:
            probe["cam"].scale = float(gui_probe_size.value)

    @gui_probe_fov.on_update
    def _(_event):
        for probe in probes:
            probe["cam"].fov = float(np.radians(gui_probe_fov.value))
        probe_report()

    gui_probe_pick.on_update(lambda _: probe_report())
    gui_probe_add.on_click(probe_add)
    gui_probe_del.on_click(probe_remove)
    gui_probe_snap.on_click(probe_snap)
    gui_probe_aim.on_click(probe_aim)

    probe_0 = make_probe(*_pose(src_gl[0], gl2cv), PROBE_FOV_DEG, cam_scale * 1.6,
                         bool(args.probe_on), bool(args.probe_on))
    bind_probe(probe_0)
    refresh_probe_list(probe_0["name"])

    if motions:
        select(0)
    else:
        refresh()

    # `--pin` 은 라벨 부분일치 **OR** 다 (--variant 의 AND 와 다르다). 여기서 OR 인 이유는
    # pin 의 쓰임이 "서로 다른 것 여러 개를 한 화면에" 라서다 — AND 로 걸면 한 종류만 남는다.
    if args.pin:
        picked = [m[0] for m in motions
                  if any(token in m[0] for token in args.pin)][:int(args.max_pins)]
        for label in picked:
            add_pin(label)
        gui_pin_info.value = pin_report()
        refresh()
        if not picked:
            print(f"[pin] {args.pin} 에 맞는 변이가 0개 — pin 없이 띄운다")

    rows = [("video", args.video), ("frames", num_frames), ("points total", len(points)),
            ("static shown", f"{len(static_pick):,} / {len(static_idx):,}"),
            ("dynamic/frame", f"{int(np.median(dyn_counts) if dyn_counts else 0):,} (median)"),
            ("z_med frame0", f"{z_med:.4f}"), ("point size", f"{point_size:.5f}"),
            ("camera size", f"{cam_scale:.4f}  (GUI view > camera size 로 조절)"),
            ("probe vfov", f"{PROBE_FOV_DEG:.0f} deg  (소스는 {src_vfov_deg:.1f} deg)"),
            ("up", args.up), ("banks", " ".join(banks) or "-"),
            ("obb nodes", " ".join(f"{i}({'dyn' if m else 'stat'})" for i, _, m in graph_rows)
             or "-"),
            ("motions", len(motions)),
            ("pinned", f"{len(pins)} / {args.max_pins}  (GUI motion > pin current)"),
            ("source jerk p95", f"{src_jerk:.2f} px/f3"),
            ("colors", f"gt {src_color}/now {src_now_color}  "
                       f"pred {plan_color}/now {plan_now_color}  (GUI camera colors)"),
            ("obb colors", f"dyn {obb_dyn_color}  static {obb_static_color}  "
                           f"(GUI obb / --obb_color_dyn/_static)"),
            ("obb handles", f"{sum(len(e) for e in obb_dyn)} dyn + {len(obb_static)} static  "
                            f"(GUI obb > OBB thickness / dynamic·static color)"),
            ("frame set", f"GUI view > dynamic frames = current frame  "
                          f"(interval / all frames + frame interval)"),
            ("paths", f"{'on' if args.paths_on else 'off'}  (GUI view > paths / --paths_on)"),
            ("probe", f"{'on' if args.probe_on else 'off'}  (GUI probe camera / --probe_on)"),
            ("url", f"http://localhost:{args.port}")]
    width_key = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key:<{width_key}}  {value}")

    while True:
        if gui_play.value:
            gui_frame.value = (int(gui_frame.value) + 1) % num_frames
        # 재생 중이 아닐 때까지 fps 를 따르면 손잡이를 놓을 때 반응이 굼뜬다 — 멈춰 있을 때는
        # 고정 간격으로 돌고 GUI 콜백에 맡긴다.
        time.sleep(1.0 / max(1.0, float(gui_fps.value)) if gui_play.value else 0.08)


def _pose(c2w_gl: np.ndarray, gl2cv: np.ndarray):
    """add_frustums 와 같은 변환을 슬라이더 갱신에도 적용 — 프러스텀을 지웠다 다시 만들면 깜빡인다."""
    import viser.transforms as vtf
    c2w_cv = c2w_gl @ gl2cv
    return vtf.SO3.from_matrix(c2w_cv[:3, :3]).wxyz, c2w_cv[:3, 3].astype(np.float32)


def _wxyz(rotation: np.ndarray):
    """OpenCV c2w 회전 -> viser quaternion."""
    import viser.transforms as vtf
    return vtf.SO3.from_matrix(np.asarray(rotation, dtype=float)).wxyz


def _R_of(wxyz) -> np.ndarray:
    """viser quaternion -> OpenCV c2w 회전. gizmo 를 끌고 난 자세를 숫자로 읽을 때 쓴다."""
    import viser.transforms as vtf
    return vtf.SO3(np.asarray(wxyz, dtype=float)).as_matrix()


if __name__ == "__main__":
    main()
