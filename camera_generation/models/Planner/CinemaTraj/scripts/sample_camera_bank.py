"""VLM 선택자 대신 **열거 + 실측**으로 카메라를 만든다 — augmentation 용 bank.

왜 VLM 을 뺐나: 목표가 "이 씬에 가장 좋은 카메라 1개"가 아니라 "다양한 카메라 N개"로 바뀌었다.
그런데 traj 턴은 실측상 **prior** 다 — 54 draw 중 non-`orbit_left_arc` 가 5개뿐이고, 숫자 제거 ·
이미지 제거 · 숫자 반전 · 가짜 마젠타 · 라벨 중립화 · temp 1.0 어느 것도 못 움직였다
(DECISIONS.md D36/D37). augmentation 에 mode collapse 하는 선택자를 쓸 이유가 없다. 그래서
**선택은 열거가 하고, VLM 은 나중에 caption 쪽으로 옮긴다.**

## 축 4개

    anchor    scene_graph 노드 전량 (`max_area_frac >= --min_area_frac`).
              anchor 가 바뀌면 look_at 표적과 `radius_world` 가 같이 바뀌므로 **회전 프로파일까지**
              달라진다. `start_mode source_frame0` 에서도 살아남는 축이다.
    preset    `lbm.presets.PRESETS` 13종 (zoom 계열은 `--allow_zoom` 없으면 제외).
    tau       `--tau_ladder` 사다리. **이게 강도 축이다.**
    speed / tracking / look_at_bias   기본은 각 1개. 늘리면 곱해진다.

## τ 사다리와 hole — 게이트가 아니라 지표다

기존 파이프라인은 `max_tau 0.30` / `min_coverage 0.55` 를 **게이트**로 썼다. 그건 "렌더가 GT 에
가까워야 한다"는 전제였고, 학습 pair 생성에는 그 전제가 없다 — 하류 video model 이 hole 을 채우는
게 일이다. 그래서 여기서는 τ 를 사다리로 훑고 `hole_fraction` 을 **각 단에서 실측해 기록만** 한다.
어디서 무너지는지가 데이터로 남고, 용도별로 나중에 잘라 쓰면 된다.

지키는 제약은 하나뿐이다 — anchor 노드가 소스 frame 0 에서 실제로 보여야 한다 (`--min_subject_area`).
"카메라가 표면 뒤"(G1)는 여기서 안 본다: 시작 pose 가 소스 frame 0 이라 정의상 표면 앞이고,
궤적이 뚫고 들어가는지는 `hole_fraction` 이 대신 말해준다.

## τ 가 크기를 못 정하는 두 경우 (1차 실행에서 실측으로 드러난 것)

    회전 전용 `pan_left/right`   이동이 0 이라 어떤 s 를 곱해도 τ 가 같다 → `fit_tau` 가 항상
        `max_scale` 로 튀어 사다리 5단이 **전부 같은 궤적**이었다 (camel `path_len_u 0.0000`,
        `hole 0.740` × 5). 이것들만 사다리를 회전 각도로 옮긴다 (`--pan_deg_at_max`).
    tau_start > rung   시작 pose(=소스 frame 0)의 자기 이동만으로 이미 예산 초과. `fit_tau` 가
        움직임을 0 으로 눌러 `static_hold` 복제본이 나온다 (avocado τ0.1 에서 98건 전량).
        `--drop_saturated`(기본) 로 빼고 `bank.json`/요약표에 몇 건인지 남긴다. 회전 전용 preset
        도 같이 걸린다 — τ 와 무관한데도 rel 이 통째로 눌리기 때문이다.

## 좁은 obs_az_span 노드의 orbit

`shape_context` 는 orbit sweep 을 `min(45, orbit_span_frac·obs_az_span)` 으로 자른다. camel
`stat_0` 는 span 이 3.0° 라 sweep 2.4° — orbit 이 직선으로 무너진다. 여기서는 노드마다
`orbit_span_frac = max(--orbit_span_frac, --min_sweep_deg / span)` 으로 **바닥을 깐다**. 소스가
못 본 면을 보게 되지만, 그게 정확히 이 사다리가 재려는 것이다.

출력:
    <out>/<video>/bank/bank.json    변이 전량 + 실측치 (`lbm_camera_bank_v1`)
    <out>/<video>/bank/bank.csv     사람이 읽는 표
    <out>/<video>/bank/poses.npz    cam_c2w (V,49,4,4) + look_at + subject_track
    <out>/<video>/bank/preview.png  변이 일부의 중간 프레임 (hole 마젠타)

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=0 python scripts/sample_camera_bank.py --video camel
    CUDA_VISIBLE_DEVICES=0 python scripts/sample_camera_bank.py --video camel \
        --tau_ladder 0.2 0.6 --trackings world lock --num_samples 40 --seed 0
"""
import json
import sys
from argparse import ArgumentParser
from inspect import signature
from os import makedirs, path
from time import perf_counter

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import build_poses                                      # noqa: E402
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import subject_point_mask                                        # noqa: E402
from lbm.gates import (approach_profile, behind_profile, elevation_profile,     # noqa: E402
                       obb_clearance)
from lbm.mesh_collision import (MeshClearance, mesh_behind_profile,             # noqa: E402
                                resolve_mesh_grid)
from lbm.overlay import contact_sheet, label_tile, paint_holes                  # noqa: E402
from lbm.presets import (LEGACY_ONLY_PRESETS, PRESETS, STATIC_PRESETS,          # noqa: E402
                         TAU_REF_CHOICES, fit_tau, load_external_shapes,
                         resolve_aim)
from lbm.render import CloudRenderer, add_cloud_source_args, open_renderer      # noqa: E402,F401
from scene_graph.io import load_scene                                           # noqa: E402
from scene_graph.scale import (TAU_DENOM_MODES, assert_scale_mode,             # noqa: E402
                               tau_denominator)
from scene_graph.schema import load_graph, pick_main_anchors                    # noqa: E402
from scripts.build_candidate_board import subject_track_volume                  # noqa: E402

# 강도 사다리 기본값. 0.20 은 지금 파이프라인 기본, 0.30 은 예전 게이트 상한, 그 위는 미개척.
TAU_LADDER = (0.10, 0.20, 0.35, 0.60, 1.00)

# **회전만 하는 preset 은 τ 로 크기를 못 정한다.** τ = |Δp|/z_med 인데 이동이 0 이라 어떤 s 를
# 곱해도 τ 가 같고, `fit_tau` 의 `tau_hi <= target_tau` 가지가 걸려 항상 `max_scale`(4.0)로
# 튄다 — 실측: camel pan_left 가 사다리 5단 전부 `path_len_u 0.0000`, `hole 0.740` 동일값이었다.
# 그래서 이것들만 사다리를 **회전 각도**로 옮긴다 (`--pan_deg_at_max` × rung/max_rung).
# `pan_right_zoom_out` 도 이동이 0 이다 — intrinsic zoom 은 시차를 안 만든다.
# **tuple 이 아니라 list 다 (D79)** — `--external_shapes` 로 등록한 DataDoP 궤적 중 이동이 0 인
# 것들이 `.extend()` 로 여기 들어온다. tuple 로 두고 재대입하면 `fit_hole_ladder` 가 import 해
# 둔 이름이 옛 객체를 계속 가리켜 조용히 어긋난다.
# D90. tilt 2종 추가 — pan 과 똑같이 이동이 정확히 0 이라 τ 사다리가 안 먹는다 (`T.tilt` 는
# 제자리 pitch 회전이다). 사다리 손잡이는 pan 과 같은 `pan_deg` 를 쓴다.
ROTATION_ONLY_PRESETS = ["pan_left", "pan_right", "pan_right_zoom_out",
                         "tilt_up", "tilt_down"]

# 그 `max_scale` 을 되돌려야 요청한 각도가 그대로 나온다 (안 나누면 4배로 커진다). 기본값이
# 바뀌어도 따라가도록 시그니처에서 읽는다.
FIT_TAU_MAX_SCALE = float(signature(fit_tau).parameters["max_scale"].default)


def register_external(shapes_json: str | None, aim: str = "traj"):
    """`--external_shapes` 처리. 등록 안 하면 `PRESETS` 가 손도 안 닿으므로 기존 동작 그대로다.

    **`sample_camera_bank` 와 `fit_hole_ladder` 가 같은 파일을 같은 인자로 불러야 한다** —
    fit 은 뱅크 행의 preset 이름으로 궤적을 다시 짓기 때문에, 등록을 빼먹으면 `모르는 preset`
    으로 죽는다 (조용히 틀리지는 않는다).
    """
    if not shapes_json:
        return []
    names, rot_only = load_external_shapes(shapes_json, aim=aim)
    ROTATION_ONLY_PRESETS.extend(n for n in rot_only if n not in ROTATION_ONLY_PRESETS)
    return names


def usable_presets(allow_zoom: bool):
    """zoom 이 필요한 preset 은 emit 에 intrinsics 채널이 없어 하류를 못 통과한다 (§emit).

    D90. `LEGACY_ONLY_PRESETS` 는 **새 뱅크에서 뺀다** — 옛 뱅크 행(`dolly_out_dont_look`)이
    그때 나오던 카메라를 되만들기 위해서만 존재하는 슬롯이라, 새로 샘플링하면 `dolly_out` 과
    이름만 다른 같은 궤적이 하나 더 생긴다.
    """
    return [name for name, (_, _, needs_zoom) in PRESETS.items()
            if (allow_zoom or not needs_zoom) and name not in LEGACY_ONLY_PRESETS]


def variant_tier(preset: str) -> int:
    """예비 변이의 층 (D168 ②). 본 슬롯은 호출자가 0 으로 덮는다.

    사용자가 준 순서 그대로다 ("… 다른 track 없는 preset 쪽을 본다던지 target anchor 를
    포기하고 free-moving 으로 대체한다던지"):

      1  조준은 하되 추종 안 함 (`aim != free`, 이름에 `track_` 없음)
      2  추종 (`track_*`) — anchor 를 따라다니므로 실패 원인이 본 슬롯과 겹치기 쉽다
      3  targetless (`aim == "free"`) — **target anchor 자체를 포기**하는 최후 수단

    2 를 1 보다 뒤에 두는 이유: d166 실측에서 `behind > 0` 인 10행이 **전부** `track_*` 였다.
    본 슬롯의 track 이 충돌로 죽었으면 같은 anchor 의 다른 track 도 같은 벽에 부딪힌다.
    """
    if resolve_aim(preset) == "free":
        return 3
    return 2 if preset.startswith("track_") else 1


def plan_variants(live, route_presets: dict, route_backfill: dict, free_preset,
                  object_budget: int, target_count: int, presets, track_ok,
                  variant_pool: str = "budget"):
    """살아남은 anchor 에 scene 예산을 배분한다 (D166). -> (plan, tiers).

    `plan` 은 `[(node, subject_points, [preset])]`, `tiers` 는 `{(anchor_id, preset): 층}`.

    `variant_pool` (D168 ②, 사용자 지시 2026-09-08 "가능한 preset 들 최대한 돌리고"):
      `budget` 예산만큼만 계획한다 = D167 이전 동작, 비트 단위로 같다.
      `full`   예산 밖 풀도 **예비**로 같이 계획한다. τ 뱅크가 커지는 대신 (풀이 anchor 당
               ~5 개라 변이가 대략 2배) `fit_hole_ladder --fallback_ladder` 가 본 슬롯이
               `clamped_low` 로 죽었을 때 갈아끼울 데가 생긴다. fit 은 예산이 차면 멈추므로
               fit 시간은 실패한 만큼만 는다.

    `target_count` 는 **상한**이지 요구가 아니다 (사용자 지시 2026-09-08 "최대 5개"). 못 채우면
    채운 만큼만 낸다 — 억지로 맞추려고 슬롯을 복제하면 같은 궤적이 캡션만 바꿔 두 번 들어간다.

    왜 라우팅이 아니라 여기서 정하나: anchor 가 소스 frame 0 에서 안 보이면(`min_subject_points`
    / `min_subject_area`) 뱅크가 통째로 버리는데, 라우팅은 렌더를 안 하므로 그걸 **예측할 수
    없다**. 파일럿 실측으로 짧은 6편 중 5편이 여기서 anchor 하나를 잃었고, 그중 `023615b3` /
    `01d32f88` 은 잃은 쪽이 `dyn_0` 이라 free-moving 까지 같이 사라졌다.

    배분 규칙 (사용자 확정):
      anchor 2개 이상 + 움직임  ->  {a, b} x 슬롯 2       = 4  (target 축과 motion 축 동시 대조)
      그 외(정지 / anchor 1개) ->  a x 슬롯 4            = 4  (target 이 상수라 motion 만 벌린다)
      + free-moving 1 (살아남은 **첫** anchor 에 붙인다 — 원래 anchor 가 죽어도 안 죽는다)
    모자란 만큼 `route_backfill`(keep_pair 밖 슬롯)에서 순서대로 채운다. 품질 게이트
    (hole/sif/behind) 탈락은 여기서 안 센다 — 그건 코퍼스 필터가 나중에 거른다 (사용자 확정).

    `target_count` 가 0 이면(=`--preset_route` 없음 또는 옛 JSON) **옛 동작 그대로**:
    anchor 전량 x `presets` 합집합.
    """
    if not target_count:
        return ([(n, sp, [p for p in route_presets.get(n["id"], presets) if track_ok(n, p)])
                 for n, sp in live], {})
    if not live:
        return [], {}
    moving = any(n.get("moving") for n, _ in live[:2])
    used = live[:2] if (len(live) >= 2 and moving) else live[:1]
    per = max(object_budget // len(used), 1)

    # anchor 별 후보 풀: keep_pair 슬롯 먼저, 그다음 backfill. track 게이트를 여기서 태운다 —
    # `track_*` 이 빠진 자리도 결손이라 backfill 로 채워야 한다 (파일럿 1차의 3개 원인).
    pool = {n["id"]: [p for p in (list(route_presets.get(n["id"], presets))
                                  + list(route_backfill.get(n["id"], [])))
                      if track_ok(n, p)] for n, _ in used}
    picked = {n["id"]: pool[n["id"]][:per] for n, _ in used}

    # 부족분 top-up: 쓰던 anchor 의 남은 풀을 round-robin 으로 돈다. anchor 를 늘리지 않는 이유는
    # 3번째 anchor 를 끌어오면 그 씬만 target 축이 3-way 라 2x2 대조가 깨지기 때문.
    def total():
        return sum(len(v) for v in picked.values()) + (1 if free_preset else 0)
    added = True
    while total() < target_count and added:
        added = False
        for n, _ in used:
            if total() >= target_count:
                break
            rest = [p for p in pool[n["id"]] if p not in picked[n["id"]]]
            if rest:
                picked[n["id"]].append(rest[0])
                added = True

    plan, tiers = [], {}
    for i, (node, sp) in enumerate(live):
        node_presets = list(picked.get(node["id"], []))
        # free-moving 은 anchor 를 안 쓰는 변이(caption `targetless`)라 살아남은 아무 anchor 에나
        # 붙으면 된다. 첫 anchor 로 고정해 결정론을 유지한다.
        if free_preset and i == 0:
            node_presets.append(free_preset)
        for p in node_presets:
            tiers[(node["id"], p)] = 0          # 예산 안 = 본 슬롯
        # D168 ②. 예비. 풀에서 안 뽑힌 나머지 + (첫 anchor 가 아니어서 free 를 못 받은
        # anchor 에는) free-moving 도 붙인다 — "target anchor 를 포기"하는 선택지가 그 anchor
        # 에서도 있어야 한다. 정렬은 (층, 풀 순서) 라 결정론이다.
        if variant_pool == "full":
            rest = [p for p in pool.get(node["id"], []) if p not in node_presets]
            if free_preset and free_preset not in node_presets:
                rest.append(free_preset)
            for p in sorted(rest, key=lambda q: (variant_tier(q),
                                                 pool.get(node["id"], []).index(q)
                                                 if q in pool.get(node["id"], []) else 1e9)):
                tiers[(node["id"], p)] = variant_tier(p)
                node_presets.append(p)
        if node_presets:
            plan.append((node, sp, node_presets))
    return plan, tiers


def anchor_nodes(graph: dict, node_ids, min_area_frac: float, max_dynamic: int = 3,
                 max_static: int = 3, drop_surfaces: bool = True, max_anchors: int = 0):
    """anchor 후보 -> (nodes, dropped). 명시 목록(`--nodes`)이 1순위 — 그건 필터를 안 탄다.

    D127. 예전엔 `max_area_frac` 하한만 넘으면 **노드 전량**이 anchor 였다. 그 결과 편당
    anchor 가 Vista 6.7 / dynpose 4.6 / TRUMANS 10.9 개였고, 늘어난 몫은 대부분 벽·바닥·문 같은
    배경 표면이었다 (TRUMANS anchor 2,087개 중 1,113개 = 53%). anchor 수는 변이 수의 배수라
    그대로 굽는 시간이다. §schema.pick_main_anchors 가 표면을 빼고 **동적/정적 각각** 상위
    `max_dynamic`/`max_static` 개만 남긴다 (D127b, 사용자 지시).
    """
    nodes = graph["nodes"]
    if node_ids:
        by_id = {n["id"]: n for n in nodes}
        missing = [i for i in node_ids if i not in by_id]
        assert not missing, f"graph 에 없는 node: {missing} (있는 것: {[n['id'] for n in nodes]})"
        return [by_id[i] for i in node_ids], []
    return pick_main_anchors(nodes, min_area_frac, max_dynamic, max_static,
                             drop_surfaces, max_anchors)


def rung_shape(preset: str, target_tau: float, ladder, pan_deg_at_max: float,
               sweep_deg: float = 0.0, pan_deg: float = 0.0, use_fit_tau: bool = True):
    """회전 전용 preset 의 사다리를 **회전 각도**로 옮긴다. 나머지는 `None` (기본 모양).

    `sweep_deg > 0` 이면 `DEFAULT_SHAPE["sweep_deg"]`(45) 를 덮는다. **`true_orbit` 계열엔
    거의 안 먹는다** — `fit_tau` 가 SE(3) 로그를 통째로 스케일하는데 순수 나선(screw)에서는
    sweep 과 scale 이 서로 상쇄돼서 같은 τ 면 같은 궤적이 나온다 (실측 snowboard dyn_0
    orbit_left τ0.6: sweep 45 → 53.5° / sweep 135 → 53.5°, 소수점까지 동일).
    `arc + dolly` 합성 preset 에서는 두 성분의 비가 바뀌므로 실제로 먹는다
    (`pull_out_arc_left` τ0.6: 34.8° → 46.2°). orbit 을 더 돌리는 손잡이는 τ 다.
    `shape_context` 가 `min(sweep_deg, orbit_span_frac·obs_az_span)` 로 다시 자르므로 소스가
    못 본 면까지 돌지는 않는다 (snowboard dyn_0 은 span 186.2° · frac 0.8 → 상한 149°).
    """
    shape = {"sweep_deg": float(sweep_deg)} if sweep_deg > 0 else {}
    if pan_deg > 0:
        # 각도를 직접 준다 — 사다리 매핑을 무시한다. `fit_tau` 가 켜져 있으면 회전 전용 preset 은
        # 아래와 같은 이유로 `max_scale` 로 튀므로 그만큼 미리 나눠 둔다.
        divisor = FIT_TAU_MAX_SCALE if (use_fit_tau and preset in ROTATION_ONLY_PRESETS) else 1.0
        return {**shape, "pan_deg": float(pan_deg) / divisor}
    if preset not in ROTATION_ONLY_PRESETS:
        return shape or None
    frac = float(target_tau) / max(ladder)
    return {**shape, "pan_deg": pan_deg_at_max * frac / (FIT_TAU_MAX_SCALE if use_fit_tau else 1.0)}


def make_decision(graph: dict, node: dict, preset: str, target_tau: float, speed: str,
                  tracking: str, look_at_bias: float, start_mode: str, shape: dict | None = None,
                  tau_refine: bool = False, aim: str | None = None,
                  tau_ref: str = "source"):
    """`lbm_decision_v1` 한 건. `source` 는 `sampler` — VLM 도 fallback 도 아니다.

    `tau_refine` (D53) 은 `fit_tau` 이분법의 첫 눈금(0.0156) 아래를 한 번 더 훑으라는 뜻이다.
    **켤 때만** 키를 넣는다 — 예전 decision 의 JSON 과 지문(`decision_fingerprint`)을 안 바꾸려고.

    `tau_ref` (D97) 도 같은 규칙이다 — 기본값 `"source"` 일 때만 키를 뺀다. 그래야 예전 뱅크의
    JSON·지문이 한 글자도 안 바뀐다. 새로 굽는 쪽은 `"auto"` 를 넘겨 `track_*` 만 follow 기준이
    되게 한다. **resolve 한 값이 아니라 요청값을 싣는다** — `build_poses` 가 preset 을 보고 다시
    resolve 하므로, 요청값만 있으면 재현에 충분하고 preset 별칭이 섞여도 한 군데서만 갈린다.

    `aim` (D94) 도 같은 규칙으로 **줄 때만** 넣는다. 디스크의 뱅크 행을 다시 디코드하는 쪽
    (`emit_bank.decision_from_variant`)이 `variant["aim"]` 을 실어 주면 `build_poses` 의
    `resolve_aim` 이 옛 이름·옛 aim 을 **그때 나오던 카메라**로 되돌린다. 새로 굽는 쪽은 안
    넘기므로 `None` → preset 의 현재 기본값이고 JSON 도 그대로다 (`decision_fingerprint` 는
    `aim` 을 안 보므로 지문도 안 바뀐다).
    """
    p_g = np.asarray(graph["cameras"]["cam_centers_g"], dtype=float)[0]
    look_at_g = np.asarray(node["track"]["center_smooth"], dtype=float)[0]
    return {
        "format": "lbm_decision_v1", "video": graph["video"], "subject_id": node["id"],
        "source": "sampler", "model": None, "start_mode": start_mode,
        "keyframes": [{
            "t": 0, "target": node["id"],
            "composition": {"azimuth_deg": None, "elevation_deg": None,
                            "d_azimuth_deg": 0.0, "d_elevation_deg": 0.0,
                            "distance_ratio": 1.0, "thirds_anchor": None,
                            "p_G": p_g.tolist(), "look_at_G": look_at_g.tolist(),
                            "focal_scale": 1.0},
            "visibility": None, "relation": None, "motion_preference": None,
            "candidate_id": None, "micro_ops": []}],
        "trajectory": {"preset": preset, "params": {}, "speed": speed, "tracking": tracking,
                       "look_at_bias": look_at_bias, "target_tau": target_tau,
                       # None 이면 `shape_context` 가 `DEFAULT_SHAPE` 를 그대로 쓴다.
                       "shape": shape,
                       **({"tau_refine": True} if tau_refine else {}),
                       **({"tau_ref": tau_ref} if tau_ref != "source" else {}),
                       **({"aim": aim} if aim else {})},
        "gates": None,
        "vlm": {"observation": None, "reasoning": "enumerated by sample_camera_bank.py",
                "confidence": None, "turns": 0, "repairs": 0},
    }


def behind_context(renderer, recon: dict, scale: float, num_src_frames: int, margin_frac: float,
                   clear_frac: float = 0.0, radius_px: int = 0, near_pct: float = 1.0,
                   measure_standoff: bool = False, graph: dict | None = None,
                   obb_margins: dict | None = None, time_match: bool = False,
                   mesh_grid: str = "", src_bank_c2w=None, collision_source: str = "depth",
                   mesh_margin_frac: float | None = None, min_zcam_frac: float = 0.0):
    """G1 판정에 필요한 소스 관측 묶음. `measure_trajectory(behind=...)` 에 그대로 넣는다.

    되쏘아 볼 소스 프레임은 board 경로(`--behind_frames 7`)와 같은 방식으로 균등 추출한다.
    `clear_frac`(표면 앞에 요구하는 여유)과 `radius_px`(패치 최소 depth)는 `gates.py` 설명 참조.
    `measure_standoff` 를 켜면 `renderer.standoff` (3D 최소거리)도 같이 잰다 — 기본은 꺼서
    예전 열 구성 그대로다. `graph` 를 주면 **OBB clearance**(G5, D49)도 붙는다 — 노드와 `T_gw`
    만 필요하고 렌더가 0회다. `obb_margins` (`gates.node_margins()`) 는 노드별 마진 — 없으면
    slack = raw 거리라 예전과 같다 (D51).

    `ground_z` 를 주면 **G6**(고도각 / 지면 여유)도 잴 자리가 열린다. 다만 고도각은 anchor 마다
    기준이 달라서 여기서 못 정한다 — 호출자가 anchor 루프에서 `behind["elev_node"] = node` 를
    꽂아 준다. 안 꽂으면 열이 안 붙고 예전과 똑같이 돈다.

    `time_match` 를 켜면 G1 이 **동적 픽셀을 시간이 맞는 소스 프레임에서만** 증거로 쓴다
    (`gates.behind_profile`). 2026-09-02 부터 기본 True (사용자 지시 "일단 켜줘").

    `mesh_grid` (D116) 를 주면 G1 을 **mesh 부피**로도 잰다 (`lbm.mesh_collision`). TRUMANS
    전용 — `.blend` 가 있는 코퍼스만 격자를 구울 수 있다. `collision_source` 가 어느 판을
    쓸지 고른다 (`depth` / `mesh` / `both`); 갈리는 곳은 `geometry_stats` 하나뿐이고 **열
    구성은 셋 다 같다**. `mesh_grid` 가 비면 예전 경로와 비트 동일.
    `src_bank_c2w` (뱅크 world 소스 카메라) 를 같이 주면 앵커 규약을 매번 다시 검증한다.

    `mesh_margin_frac` (D120) 는 **mesh 판 전용** 임계다. 두 판이 같은 `margin_frac` 을 쓰던
    동안 실효 여유가 4배 어긋나 있었다: depth 판은 `cam_z + clear > z + margin` 이라 실효
    standoff 가 `(clear_frac − margin_frac)·S` = 0.08·S 인데, mesh 판은 `clear_frac` 을 아예
    안 받아서 `margin_frac·S` = 0.02·S 였다. `None` 이면 `margin_frac` 을 그대로 써서 예전
    뱅크와 비트 동일이고, 0.08 을 주면 두 판이 같은 거리에서 걸린다.

    `min_zcam_frac` (D123) 는 **depth 판 전용** degenerate 투영 하한 (`gates.behind_surface_frames`
    참조). 0.0 이면 예전 경로와 비트 동일.
    """
    assert collision_source == "depth" or mesh_grid, \
        f"--collision_source {collision_source} 인데 mesh 격자가 없다"
    mesh_margin_frac = float(margin_frac if mesh_margin_frac is None else mesh_margin_frac)
    frames = np.unique(np.linspace(0, renderer.num_frames - 1, num_src_frames)
                       .round().astype(int)).tolist()
    mesh = MeshClearance(mesh_grid) if mesh_grid else None
    if mesh is not None:
        if src_bank_c2w is not None:
            mesh.assert_anchor(src_bank_c2w)
        #    D47 legality — 소스 카메라를 기각하는 임계는 충돌 판정이 아니라 버그다.
        #    `margin_frac·S` 는 뱅크 단위인데 뱅크 단위가 곧 blend metre 다 (앵커가 강체).
        src_clear, src_reach = mesh.static_clearance(mesh.cam_centers)
        assert src_reach.all(), \
            "소스 카메라가 mesh 격자에서 unreachable 이다 — 좌표계나 flood-fill 씨앗이 틀렸다"
        assert src_clear.min() > mesh_margin_frac * scale, (
            f"mesh 임계 {mesh_margin_frac * scale:.4f} m 가 소스 카메라 자신의 최소 여유 "
            f"{src_clear.min():.4f} m 보다 크다 (D47) — `--mesh_margin_frac` 를 낮출 것")
    return {"mesh": mesh, "collision_source": str(collision_source),
            "mesh_margin_frac": float(mesh_margin_frac),
            "depths": recon["depths"], "K": renderer.K_src, "cam_c2w": renderer.cam_c2w_src,
            "sky_mask": recon["sky_mask"], "scale": float(scale), "frames": frames,
            "dynamic_mask": (recon["dynamic_mask"].astype(bool) if time_match else None),
            "time_match": bool(time_match),
            "margin_frac": float(margin_frac), "clear_frac": float(clear_frac),
            "min_zcam_frac": float(min_zcam_frac),
            "radius_px": int(radius_px), "near_pct": float(near_pct),
            "standoff": bool(measure_standoff),
            "nodes": (graph["nodes"] if graph else None),
            "T_gw": (np.asarray(graph["frames"]["T_gw"], dtype=float) if graph else None),
            "obb_margins": obb_margins,
            "ground_z": (float(graph["ground"]["ground_z"]) if graph else None),
            "elev_node": None,
            # G7 (2026-08-22). `elev_node` 와 같은 자리에 anchor 노드를 꽂아 쓴다.
            "approach_node": None, "min_approach": 0.0}


def source_elevation(graph: dict, node: dict):
    """소스 카메라 **자신의** (|고도각| 최대 deg, 지면 위 최소 여유 u) — G6 임계의 legality 바닥.

    `source_standoff`/`source_obb_clear` 와 같은 이유다 (D47/D48/D51): 소스 카메라를 기각하는
    임계는 충돌 판정이 아니라 버그다. 실측 — 고도각은 anchor 마다 달라서 camel −5.10°~+4.24°,
    avocado-slice −6.32°~**+35.09°**(`stat_4` chair, 소스가 코앞 0.033 u 에서 내려다본다).
    지면 위 높이는 camel 0.023 u / avocado 0.078 u 로 **3.4배** 벌어진다 — 그래서 지면 임계는
    절대값이 아니라 이 값의 배수로 잡는다.
    """
    elev, clear = elevation_profile(
        np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float), node,
        np.asarray(graph["frames"]["T_gw"], dtype=float), float(graph["ground"]["ground_z"]))
    return float(np.abs(elev).max()), float(clear.min())


def source_approach(graph: dict, node: dict):
    """소스 카메라 **자신의** 시선축 전진 여유 최소값 (u) — G7 임계의 legality 바닥.

    `source_elevation`/`source_obb_clear` 와 같은 이유다 (D47/D48/D51/D55): 소스 카메라를
    기각하는 임계는 버그다. 실측 (2026-08-22) — 소스는 **어느 노드의 뒷면도 통과하지 않는다**.
    노드별 최소 여유는 camel 0.4950 u (`stat_0` fence) / avocado-slice 0.1097 u (`stat_4`
    chair) 로 **4.5배** 벌어지는데, 이건 raw OBB 여유의 4.4배(0.5107 / 0.1165)와 거의 같은
    비율이다 — 두 값이 정적 소스에서 사실상 같은 것을 재고 있다는 뜻이라, 임계를 소스 배수로
    잡으면 G5 마진과 자동으로 같은 눈금에 놓인다.
    """
    gap, _ = approach_profile(np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float),
                              node, np.asarray(graph["frames"]["T_gw"], dtype=float))
    return float(np.nanmin(gap))


def source_obb_clear(graph: dict, margins=None):
    """소스 카메라 **자신의** OBB slack 최소값 (u 단위) + 그 노드 id + 그 지점의 raw 거리.

    standoff 와 같은 이유로 남긴다 (D48): slack 이 음수면 **소스 카메라조차** 그 마진을 통과
    못 한다 — 원본 촬영이 기각되는 마진은 충돌 판정이 아니라 버그다. 마진 없는 raw 거리 실측은
    camel +0.5107 (`fence`) / avocado-slice +0.1165 (`chair`) 로 **4.4배** 벌어지고, 이게
    절대 마진 하나로 두 씬을 못 덮는 이유 중 하나다 (D51).
    """
    dists, ids, slacks = obb_clearance(
        np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float), graph["nodes"],
        np.asarray(graph["frames"]["T_gw"], dtype=float), margins=margins)
    f = int(slacks.argmin())
    return float(slacks[f]), ids[f], float(dists[f])


def source_g1_clear(behind: dict, pct: float = 10.0):
    """소스 카메라 **자신의** G1 여유 `min_t (z_surf − z_cam)/S` 의 `pct` 분위 (u) — G1 임계의
    legality 바닥. (D123)

    `source_standoff`/`source_obb_clear`/`source_approach` 와 같은 이유다 (D47/D48/D51/D55):
    소스 카메라를 기각하는 임계는 충돌 판정이 아니라 버그다. **G5/G6/G7 은 전부 이 바닥의 β 배로
    임계를 잡는데 G1 만 절대 분수(`clear_frac=0.10`)였고**, 그래서 실제로 소스를 기각한다 —
    18편 실측에서 parkour 1편이 배포 0.10 에서 위반, goat 0.1199 / hike 0.1029 로 아슬아슬하다.

    **분위수를 쓰는 이유.** min 은 못 쓴다. 사용자가 지시한 "일관적 기준 거리 조합" 탐색
    (`scripts/probe_g1_reference.py`, 18편)에서 소스 카메라 거리 · 최소 scene-camera 거리 ·
    피사체 shot scale · 피사체까지 거리와 그 곱/기하평균 조합의 CV 가 전부 0.89~2.10 으로
    게이지가 못 되고, 유일하게 안정적인 건 **G1 자기 분포**였다 — `g1_p10` CV 0.371 /
    `g1_2nd` 0.396 / `g1_p25` 0.418 vs `g1_p50` 0.534. p10 이 최저라 기본값으로 둔다.
    (min 이 오염돼 있던 이유는 `gates.behind_surface_frames` 의 `min_zcam_frac` 설명 참조.)

    `behind` 는 `behind_context()` 가 돌려준 그대로 — 판정에 쓰는 것과 **같은** depth/K/c2w/
    sky/dynamic/frames/`min_zcam_frac` 을 쓴다. 판정과 다른 재료로 바닥을 재면 β 가 뜻을 잃는다.
    """
    depths, K, cam_c2w = behind["depths"], behind["K"], behind["cam_c2w"]
    sky, dyn = behind["sky_mask"], behind.get("dynamic_mask")
    scale, radius = float(behind["scale"]), int(behind["radius_px"])
    z_floor = max(float(behind.get("min_zcam_frac", 0.0)) * scale, 1e-6)
    height, width = depths.shape[-2:]
    gaps = []
    for pose in np.asarray(cam_c2w, dtype=float):
        p_world = pose[:3, 3]
        best = float("inf")
        for t in behind["frames"]:
            w2c = np.linalg.inv(cam_c2w[t])
            cam = w2c[:3, :3] @ p_world + w2c[:3, 3]
            if cam[2] <= z_floor:
                continue
            uv = (K[t] @ cam)[:2] / cam[2]
            u, v = int(np.floor(uv[0])), int(np.floor(uv[1]))
            if not (0 <= u < width and 0 <= v < height):
                continue
            u0, u1 = max(u - radius, 0), min(u + radius + 1, width)
            v0, v1 = max(v - radius, 0), min(v + radius + 1, height)
            patch = depths[t][v0:v1, u0:u1]
            usable = np.isfinite(patch) & (patch > 0) & ~sky[t][v0:v1, u0:u1]
            if dyn is not None:                       # static 채널 — 판정과 같은 규약
                usable &= ~dyn[t][v0:v1, u0:u1]
            if not usable.any():
                continue
            best = min(best, (float(patch[usable].min()) - float(cam[2])) / scale)
        if np.isfinite(best):
            gaps.append(best)
    assert gaps, "소스 카메라 전부에서 G1 증거가 안 잡혔다 — frames/mask 배선을 볼 것"
    return float(np.percentile(np.asarray(gaps, dtype=float), float(pct)))


def source_standoff(renderer, scale: float):
    """소스 카메라 **자신의** 3D standoff 최소값 (S 단위) — clearance 임계의 물리적 바닥.

    절대값(`0.15·S`)으로 임계를 걸면 안 되는 이유다 (D47): 실측이 camel 0.087·S /
    avocado-slice 0.137·S 라, 0.15 는 **소스 카메라조차 통과 못 하는** 기준이었다. 씬마다
    기하가 카메라에 얼마나 붙어 있는지가 다르므로 임계는 이 값의 배수로 잡는다.
    """
    return float(renderer.standoff(renderer.cam_c2w_src).min()) / float(scale)


def geometry_stats(renderer, poses: np.ndarray, behind: dict):
    """**렌더가 0회인** 기하 열 전부 — G1 / G5 / G6 / G7 + standoff. (D113)

    `measure_trajectory` 안에 있던 블록을 그대로 떼어냈다 (`near_depth` 만 남겼다 — 그건
    렌더한 depth 에서 나오므로 여기 못 온다). **키 순서까지 원래 그대로**라 `bank.json` 이
    바뀌지 않는다.

    왜 떼어내나 (사용자 지시 D113): "물리 판정이 실패하면 사실 렌더할 필요가 없잖아". G1 은
    재투영, G5/G6/G7 은 프레임당 3×3 곱 몇 개라 hole 렌더보다 서너 자릿수 싸다. 그런데
    `measure_trajectory` 는 렌더를 **먼저** 돌고 이것들을 뒤에 붙였다 — `over()` 의 if/elif
    체인이 이 넷을 hole 앞에서 보므로, 넷 중 하나라도 걸리면 그 렌더는 통째로 버려진 것이다.
    이제 이걸 먼저 재고 판정을 통과한 손잡이만 렌더한다 (`measure_trajectory(gate_check=...)`).

    판정 결과가 바뀌지는 않는다 — 같은 값을 같은 순서로 보는 것이고, 걸린 손잡이의 hole 은
    애초에 아무도 안 읽었다 (`fit_hole_ladder` 의 최종 행은 `verify_frames` 로 **다시** 잰다).
    """
    stats = {}
    if behind.get("standoff"):
        # 렌더한 몇 프레임이 아니라 전 프레임 — 이것도 재투영 수준으로 싸다.
        stats["standoff"] = round(float(renderer.standoff(poses).min()) / behind["scale"], 4)
    # 렌더한 몇 프레임이 아니라 **궤적 전 프레임**을 본다 — 공짜이고, 충돌은 한 프레임만
    # 뚫려도 충돌이다.
    time_match = behind.get("time_match", False)
    mesh = behind.get("mesh")
    source = behind.get("collision_source", "depth")

    def _depth_profile():
        return behind_profile(poses, behind["depths"], K=behind["K"],
                              cam_c2w=behind["cam_c2w"], sky_mask=behind["sky_mask"],
                              scale=behind["scale"], frames=behind["frames"],
                              margin_frac=behind["margin_frac"],
                              clear_frac=behind["clear_frac"],
                              radius_px=behind["radius_px"],
                              dynamic_mask=behind.get("dynamic_mask"),
                              time_match=time_match, detail=time_match,
                              min_zcam_frac=behind.get("min_zcam_frac", 0.0))

    def _mesh_profile():
        # D116. depth shell 은 **소스가 본 표면**만 알아서 뒤로 물러나는 궤적의 벽을 통째로
        # 놓친다 (실측 a00: mesh 357/836 위반, binding `none` 이던 103건 포함).
        # `.blend` 가 부피를 아는데 shell 로 잴 이유가 없다. 반환 규약이 같아서 열 구성은
        # 한 줄도 안 바뀐다.
        # D120. depth 판의 실효 standoff 는 `(clear_frac − margin_frac)·S` 인데 mesh 는
        # `clear_frac` 을 안 받는다. 별도 손잡이로 같은 거리를 주도록 맞춘다 — 없으면
        # `margin_frac` 으로 떨어져서 예전 뱅크와 비트 동일.
        return mesh_behind_profile(mesh.to_blend(poses), mesh,
                                   behind.get("mesh_margin_frac", behind["margin_frac"])
                                   * behind["scale"], detail=time_match)

    if source == "both":
        # 두 판이 **서로 다른 것**을 잡는다 (a00 실측 836변이: depth 130 / mesh 357 / 겹침 0).
        # depth 는 소스가 본 표면을 정확히 알고, mesh 는 소스가 못 본 부피를 안다 — 어느 한쪽만
        # 쓰면 나머지 절반을 놓친다. 프레임 마스크로 **합집합**을 센다 (개수 합은 중복 계산,
        # max 는 과소보고라 게이트가 느슨해진다).
        assert time_match, "--collision_source both 는 --collision_time_match 가 필요하다 (채널 마스크)"
        dep, msh = _depth_profile(), _mesh_profile()
        per_ch = {}
        for name in ("static", "dynamic"):
            union = [a or b for a, b in zip(dep[2][name]["mask"], msh[2][name]["mask"])]
            per_ch[name] = {"frames": int(sum(union)),
                            "worst": max(dep[2][name]["worst"], msh[2][name]["worst"]),
                            "mask": union}
        any_bad = [a or b for a, b in zip(per_ch["static"]["mask"], per_ch["dynamic"]["mask"])]
        out = (int(sum(any_bad)), max(dep[1], msh[1]), per_ch)
    elif source == "mesh":
        out = _mesh_profile()
    else:
        out = _depth_profile()
    bad, worst = out[0], out[1]
    stats.update({"behind_frames": bad, "behind_frac": round(bad / len(poses), 4),
                  "behind_worst_src": worst})
    if time_match:
        # 정적/동적을 **따로 판정**한다 (사용자 지시 2026-09-02: "collision 을 static 도 하고
        # dynamic 은 따로 해서 양쪽 다 판정"). `*_frac` 두 개가 각자의 예산과 비교되고, 게이트는
        # 둘의 OR 다 (`physical_verdict`). 예산이 같으면 합집합 `behind_frac` 과 수학적으로
        # 동일하지만, 채널별로 다른 예산을 줄 수 있게 열을 분리해 둔다.
        # 동적이 잡은 건 그 시각의 소스 프레임 한 장에서만 나오므로 `behind_dyn_worst ≤ 1`.
        num = len(poses)
        stats.update({"behind_static_frames": out[2]["static"]["frames"],
                      "behind_static_frac": round(out[2]["static"]["frames"] / num, 4),
                      "behind_static_worst": out[2]["static"]["worst"],
                      "behind_dyn_frames": out[2]["dynamic"]["frames"],
                      "behind_dyn_frac": round(out[2]["dynamic"]["frames"] / num, 4),
                      "behind_dyn_worst": out[2]["dynamic"]["worst"]})
    if behind.get("nodes"):
        # G5 (D49). 부호 거리라 음수면 카메라가 **노드 OBB 안**이다. 어느 노드였는지도 남긴다
        # — "충돌했다"보다 "fence 를 뚫었다"가 고칠 수 있는 진단이다.
        # 판정량은 `obb_slack = obb_clear − m_j` (D51). 마진이 전 노드 상수면 둘의 argmin 이
        # 같아서 `obb_clear` 열은 예전 뱅크와 그대로 비교된다.
        dists, ids, slacks = obb_clearance(poses, behind["nodes"], behind["T_gw"],
                                           margins=behind.get("obb_margins"))
        worst_f = int(slacks.argmin())
        stats.update({"obb_clear": round(float(dists[worst_f]), 4),
                      "obb_slack": round(float(slacks[worst_f]), 4),
                      "obb_node": ids[worst_f]})
    if behind.get("elev_node") is not None and behind.get("ground_z") is not None:
        # G6 (2026-08-22). 역시 전 프레임 — 재투영보다도 싸다 (프레임당 3×3 곱 하나).
        # `elev_abs_max` 가 판정량이다: 위든 아래든 물체 **바로 위/아래**면 같은 문제다.
        elev, clear = elevation_profile(poses, behind["elev_node"], behind["T_gw"],
                                        behind["ground_z"])
        stats.update({"elev_max": round(float(elev.max()), 2),
                      "elev_min": round(float(elev.min()), 2),
                      "elev_abs_max": round(float(np.abs(elev).max()), 2),
                      "ground_clear": round(float(clear.min()), 4),
                      "below_ground_frames": int((clear < 0.0).sum())})
    if behind.get("approach_node") is not None:
        # G7 (2026-08-22). 부호 있는 시선축 전진량 — G5 가 구조적으로 못 보는 "지나침".
        # 역시 전 프레임이고 렌더가 0회다.
        gap, past = approach_profile(poses, behind["approach_node"], behind["T_gw"])
        stats.update({"approach_gap": round(float(np.nanmin(gap)), 4),
                      "approach_frames": int((gap < behind.get("min_approach", 0.0)).sum()),
                      "past_frames": int((past > 0.0).sum())})
    return stats


def composition_stats(renderer, poses: np.ndarray, picks, comp: dict):
    """subject **말고 또 무엇이 화면에 담기는가** — 렌더 0회. (D121)

    왜 필요한가: 캡션의 framing 절이 지금은 `subject_area_med` 버킷 하나(shot scale)뿐이라
    "무엇과 같이 담기는지 / 무엇이 프레임 밖으로 나가는지"를 말할 수 없다. 사용자가 요청한
    `while keeping [framing]` 의 framing 은 shot scale + composition 둘이다.

    어떻게: 노드 OBB 8꼭짓점을 그 프레임 카메라로 투영해 2D bbox 를 만들고, **이미지와 겹치는
    면적비**가 `min_area` 이상이면 "담겼다"로 본다. `obb_clearance` 와 같은 좌표 처리다 —
    `T_gw @ pose` 로 카메라를 G 로 옮기면 OBB(G 좌표)와 같은 프레임이 되고, 스케일은
    `inv(cam) @ point` 에서 상쇄되므로 K 를 손댈 필요가 없다. 면적 **비율**이라 렌더 해상도와도
    무관하다 (소스 해상도로 계산한다).

    **가림은 안 본다.** 가림까지 보려면 노드마다 실루엣을 한 번 더 그려야 해서 렌더가 노드 수
    배로 늘어난다 (camel 5노드 / avocado 6노드). 캡션이 말하는 건 "프레임에 들어오나"이지
    "앞이 뚫려 있나"가 아니므로 투영으로 충분하다. 대신 그 한계를 열 이름으로 남긴다
    (`in_frame_ids` 지 `visible_ids` 가 아니다).

    돌려주는 것 — 전부 노드 id 리스트다 (라벨이 아니라). 캡션 문구는 `instance_desc.json` 이
    갖고 있고 그건 뱅크를 다시 굽지 않고 바꿀 수 있어야 한다:
        `in_frame_ids`  잰 프레임의 절반 이상에서 담긴 노드 (면적 큰 순, `max_nodes` 개까지)
        `enter_ids`     앞 1/3 에는 없다가 뒤 1/3 에 들어온 노드
        `exit_ids`      앞 1/3 에는 있다가 뒤 1/3 에 없어진 노드
    """
    from lbm.overlay import project_world                          # 순환 import 회피
    from scene_graph.obb import node_obb_at, obb_corners

    T_gw = np.asarray(comp["T_gw"], dtype=float)
    keep = [n for n in comp["nodes"] if n["id"] != comp.get("anchor_id")]
    min_area, max_nodes = float(comp["min_area"]), int(comp["max_nodes"])
    height, width = int(renderer.height), int(renderer.width)
    areas = {n["id"]: [] for n in keep}
    for f in picks:
        cam = T_gw @ np.asarray(poses[int(f)], dtype=float)
        K = np.asarray(renderer.K_src[int(f)], dtype=float)
        for node in keep:
            if node.get("moving"):
                center, extent, R = node_obb_at(node, int(f))
            else:
                center = np.asarray(node["obb"]["center"], dtype=float)
                extent = np.asarray(node["obb"]["extent"], dtype=float)
                R = np.asarray(node["obb"]["R"], dtype=float)
            uv, z = project_world(obb_corners(center, extent, R), K, cam)
            front = z > 1e-6
            # 꼭짓점 절반이 카메라 뒤면 bbox 가 화면 전체로 번진다 — 그때는 "담겼다"로 안 센다.
            if int(front.sum()) < 4:
                areas[node["id"]].append(0.0)
                continue
            (u0, v0), (u1, v1) = uv[front].min(axis=0), uv[front].max(axis=0)
            overlap = (max(0.0, min(u1, width) - max(u0, 0.0))
                       * max(0.0, min(v1, height) - max(v0, 0.0)))
            areas[node["id"]].append(overlap / float(width * height))
    seen = {i: np.asarray(a, dtype=float) >= min_area for i, a in areas.items()}
    edge = max(1, len(picks) // 3)
    ranked = sorted(keep, key=lambda n: -float(np.mean(areas[n["id"]])))
    return {"in_frame_ids": [n["id"] for n in ranked
                             if seen[n["id"]].mean() >= 0.5][:max_nodes],
            "enter_ids": [n["id"] for n in ranked
                          if not seen[n["id"]][:edge].any() and seen[n["id"]][-edge:].all()],
            "exit_ids": [n["id"] for n in ranked
                         if seen[n["id"]][:edge].all() and not seen[n["id"]][-edge:].any()]}


def measure_trajectory(renderer, poses: np.ndarray, num_frames: int, measure_frames: int,
                       height: int, width: int, center_box: float, behind: dict | None = None,
                       focal=None, subject_points=None, metric_only: bool = False,
                       gate_check=None, area_timeline: bool = False,
                       composition: dict | None = None):
    """궤적 위 몇 프레임을 렌더해 hole / subject 지표. (지표 dict, 중간 프레임 렌더).

    `subject_points` (n,) bool 을 주면 **G3 가림**(`subject_visible_frac`)까지 잰다. 안 주면 열이
    안 붙고 예전 뱅크와 비트 단위로 같다. D81.

    `area_timeline` 을 켜면 `subject_area_seq`(잰 프레임별 면적비 전량) + `subject_area_start` /
    `subject_area_end` 가 붙는다. **렌더가 안 늘어난다** — 이미 `areas` 로 모으고 있던 값을
    `nanmedian` 으로 접기 전에 그대로 싣는 것뿐이다 (D119). 왜 필요한가: shot scale 캡션은
    `subject_area_med` 한 숫자에서 나오는데, median 은 push-in 과 pull-out 과 정지를 **같은
    값으로 만든다**. 시간축이 캡션에 들어가려면 접기 전 배열이 남아 있어야 한다.
    start/end 는 배열의 양 끝이라 median 의 이상치 보호를 못 받는다 — 소비하는 쪽이
    (`build_bank_captions.py`) 양 끝 몇 개를 다시 median 으로 접는 이유다.

    왜 이게 따로 필요한가: 기존 `subject_area_med` 는 **그려진 실루엣의 화면 면적비**라 가림과
    구분이 안 된다. 책상 아래로 내려간 카메라는 subject 가 상판에 가려 안 보이는데도 면적비는
    멀쩡히 나온다 (실측: `pedestal_down` 의 `subject_area_med` 중앙값 0.029 로 `orbit_left`
    0.031 과 사실상 같다). `render.CloudRenderer.measure` 의 `num_subject_points` 판은 픽셀수를
    점 개수로 나눈 **밀도**라 비율이 아니다 (그 함수 docstring 이 직접 경고한다). 제대로 된 값은
    `lbm.gates.evaluate:338-349` 가 쓰는 **두 번 렌더**다 — subject 만 그린 실루엣(`alone`)과
    전체 렌더의 depth 를 비교해 "그려졌어야 하는데 앞에 뭔가 있는" 픽셀을 센다.

    렌더가 2배가 되므로 **이분법에는 절대 안 넘긴다** — 판정(verify) 패스에서만 준다. 이분법이
    푸는 답은 물리 게이트와 hole 이지 가림이 아니다.

    `focal` (n,) 은 프레임별 **intrinsic zoom** 배율이다. 안 넘기면 hole 이 조용히 낮게 찍힌다 —
    zoom out 은 화각을 넓혀 소스가 안 본 영역을 프레임에 끌어들이므로 구멍이 늘어난다
    (실측 snowboard `zoom_out_pan_right` frame 48: valid 0.538 → 0.374).

    `behind` 를 주면 **G1(표면 뒤)** 도 같이 잰다 — `behind_frames`/`behind_frac`/
    `behind_worst_src` 열이 붙는다. 안 주면 열이 안 붙고 기존과 똑같이 돈다. hole 은 충돌을 못
    잡는다 (벽을 통과하면 벽 너머 관측이 그대로 그려져 `valid_mask` 가 멀쩡하다) — 그래서
    같이 재야 한다. 렌더가 안 늘어난다 (G1 은 재투영뿐).

    같이 붙는 `near_depth` 는 **렌더한 depth 의 하위 백분위 최소값(S 단위)** 이다. G1 은 "표면
    뒤냐"만 보므로 표면 **앞 1 cm** 는 통과시킨다 — 실측에서 camel `dyn_0 straight_ease` 가
    G1 을 다 통과하고도 코앞 0.037·S 까지 들어갔다. 그게 눈에는 충돌로 보인다. 이 값은 이미
    돌린 렌더에서 나오므로 역시 렌더가 안 늘어난다. **다만 판정용으로는 못 쓴다** — 방향에
    의존한다 (D47, `render.CloudRenderer.standoff` 설명). 진단용으로만 남긴다.

    `behind["standoff"]` 를 켜면 `standoff` (카메라 중심 ↔ 그 시각 점군 3D 최소거리, S 단위)가
    붙는다. 코앞 판정은 이쪽을 쓴다.

    `behind["elev_node"]` 를 꽂으면 **G6** 열이 붙는다 (`elev_max`/`elev_min`/`elev_abs_max`/
    `ground_clear`/`below_ground_frames`). 안 꽂으면 열이 없어 예전 뱅크와 그대로 비교된다.

    `composition` (D121) 을 주면 `in_frame_ids`/`enter_ids`/`exit_ids` 가 붙는다 —
    subject 말고 또 무엇이 화면에 담기는지 (`composition_stats`). 렌더가 0회이고, 안 주면 열이
    안 붙어 예전 뱅크와 비트 단위로 같다.

    `gate_check` (D113) 를 주면 **렌더 전에** 기하 열(`geometry_stats`)을 먼저 재고 그걸 넘긴다.
    사유 문자열을 돌려주면 렌더를 통째로 건너뛰고 `{"gated": 사유, hole=nan, ...}` 로 나간다 —
    `None` 이면 예전대로 전부 렌더한다. `behind=None` 이면 잴 게 없으므로 무시된다.
    """
    if gate_check is not None and behind is not None:
        geo = geometry_stats(renderer, poses, behind)
        gated = gate_check(geo)
        if gated is not None:
            # hole 계열은 **안 쟀다**는 뜻으로 nan 이다. 0.0 을 넣으면 "구멍이 없다"로 읽혀서
            # 판정이 조용히 뒤집힌다. 호출자(`over()`)는 이 손잡이를 이미 `gated` 사유로
            # 기각하므로 hole 을 안 본다.
            return ({"hole_fraction": float("nan"), "hole_max": float("nan"),
                     "subject_area_med": float("nan"), "subject_in_frame": float("nan"),
                     "measured_frames": [], "gated": gated,
                     "near_depth": float("nan"), **geo}, None)

    picks = np.unique(np.linspace(0, num_frames - 1, measure_frames).round().astype(int))
    holes, areas, in_frame, nears, seen = [], [], [], [], []
    # depth 비교 여유. `behind` 가 있으면 그 S 를 쓴다 (같은 게이지여야 `gates.evaluate` 와
    # 같은 판정이 된다). 없으면 렌더러가 들고 있는 값으로 떨어진다.
    occl_scale = float(behind["scale"]) if behind is not None else float(
        getattr(renderer, "scale", 1.0) or 1.0)
    middle = None
    lo, hi = (1 - center_box) / 2, 1 - (1 - center_box) / 2
    for f in picks:
        K = None
        if focal is not None and abs(float(focal[f]) - 1.0) > 1e-9:
            K = np.array(renderer.K_src[f], dtype=np.float64).copy()
            K[0, 0] *= float(focal[f])
            K[1, 1] *= float(focal[f])   # cx,cy 는 안 건드린다 — zoom 은 주점을 안 옮긴다
        # Bisection does not consume RGB or per-pixel arrays.  Reduce the raster on GPU and
        # transfer scalars only.  Verification/preview keeps the old full-render path.
        if metric_only:
            assert subject_points is None, "metric_only은 subject visibility 2-pass와 함께 쓸 수 없다"
            stats = renderer.render_metrics(
                poses[f], K=K, frame=int(f), height=height, width=width,
                near_percentile=(behind["near_pct"] if behind is not None else None))
            rendered = None
        else:
            rendered = renderer.render(poses[f], K=K, frame=int(f), height=height, width=width)
            stats = renderer.measure(rendered)
        holes.append(stats["hole_fraction"])
        areas.append(stats.get("subject_area", float("nan")))
        center = stats.get("subject_center")
        in_frame.append(bool(center is not None and lo <= center[0] <= hi and lo <= center[1] <= hi))
        if subject_points is not None:
            # G3 가림. subject 만 그린 실루엣과 전체 렌더의 depth 를 비교한다 (`lbm.gates.evaluate`
            # 와 같은 식). `drawn` 은 실루엣 중 전체 렌더에서도 유효한 픽셀, `occluded` 는 그중
            # 앞에 다른 게 온 픽셀. 실루엣이 비면(화면 밖) 0 으로 본다 — 가려서 안 보이는 것과
            # 프레임을 벗어난 건 다르지만, 둘 다 "subject 가 안 보인다"는 같은 결론이다.
            alone = renderer.render(poses[f], K=K, frame=int(f), height=height, width=width,
                                    subset=subject_points)
            silhouette = alone["valid"]
            n_sil = int(silhouette.sum())
            if n_sil == 0:
                seen.append(0.0)
            else:
                drawn = silhouette & rendered["valid"]
                occluded = drawn & (rendered["depth"] < alone["depth"] - 0.02 * occl_scale)
                seen.append(1.0 - float(occluded.sum()) / n_sil)
        if behind is not None:
            if metric_only:
                if stats["near_depth"] == stats["near_depth"]:
                    nears.append(stats["near_depth"] / behind["scale"])
            else:
                visible = rendered["depth"][rendered["valid"]]
                if visible.size:
                # 최소값이 아니라 백분위다 — 튀는 splat 하나에 상한이 끌려가면 못 쓴다.
                    nears.append(float(np.percentile(visible, behind["near_pct"])) / behind["scale"])
        if not metric_only and middle is None and f >= picks[len(picks) // 2]:
            middle = rendered
    stats = {"hole_fraction": round(float(np.mean(holes)), 4),
             "hole_max": round(float(np.max(holes)), 4),
             "subject_area_med": round(float(np.nanmedian(areas)), 4),
             "subject_in_frame": round(float(np.mean(in_frame)), 3),
             "measured_frames": picks.tolist()}
    if area_timeline:
        # 접기 전 배열. `measured_frames` 와 길이가 같고 인덱스가 1:1 로 맞는다 — CSV 로 나갈 때
        # 쉼표를 못 쓰므로(`bank.csv` 는 따옴표 없는 writer 다) `|` 로 잇는다.
        stats["subject_area_seq"] = [round(float(a), 4) for a in areas]
        stats["subject_area_start"] = round(float(areas[0]), 4)
        stats["subject_area_end"] = round(float(areas[-1]), 4)
    if composition is not None:
        # D121. anchor 말고 무엇이 화면에 담기나. 렌더가 0회라 (OBB 투영뿐) 여기 끼워도 공짜다.
        stats.update(composition_stats(renderer, poses, picks, composition))
    if seen:
        # 평균이 아니라 median — 한 프레임 스치는 가림에 열이 끌려가면 못 쓴다. 최악 프레임은
        # 따로 남긴다 (책상 밑 shot 은 중간이 멀쩡해도 바닥 근처에서 통째로 가린다).
        stats["subject_visible_frac"] = round(float(np.median(seen)), 4)
        stats["subject_visible_min"] = round(float(np.min(seen)), 4)
    if behind is not None:
        stats["near_depth"] = round(min(nears), 4) if nears else float("nan")
        stats.update(geometry_stats(renderer, poses, behind))
    return (stats, middle)


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    scale_mode = assert_scale_mode(graph, args.allow_legacy_scale)   # F2
    bank_folder = path.join(out_root, args.video, args.bank_dir)
    makedirs(bank_folder, exist_ok=True)

    #    `--cloud_source npz`(기본) 는 예전과 같이 cloud.npz 를 읽고, `memory` 는 recon 에서
    #    그 자리에 굽는다 (`lbm/render.py:open_renderer`).
    renderer, recon = open_renderer(args, out_root, graph)
    num_frames = int(graph["num_frames"])
    external = register_external(args.external_shapes, args.external_aim)
    presets = args.presets or usable_presets(args.allow_zoom)
    ladder = list(args.tau_ladder)

    #    G1 을 안 재면(`--measure_behind` 기본 off) 격자를 읽을 이유도, 없다고 죽을 이유도 없다.
    mesh_grid = resolve_mesh_grid(args, out_root) if args.measure_behind else ""
    # G1 은 렌더가 안 들지만 열이 늘면 기존 뱅크와 스키마가 달라진다 — 기본은 끄고 옵션으로 켠다.
    behind = (behind_context(renderer, recon, graph["scale"]["S"], args.behind_src_frames,
                             args.behind_margin_frac, args.behind_clear_frac,
                             args.behind_radius_px, args.near_pct,
                             args.measure_standoff,
                             graph=(graph if args.measure_obb else None),
                             time_match=args.collision_time_match,
                             mesh_grid=mesh_grid, src_bank_c2w=renderer.cam_c2w_src,
                             collision_source=args.collision_source,
                             mesh_margin_frac=args.mesh_margin_frac,
                             min_zcam_frac=args.behind_min_zcam)
              if args.measure_behind else None)

    nodes, dropped_anchors = anchor_nodes(graph, args.nodes, args.min_area_frac,
                                          args.max_dynamic_anchors, args.max_static_anchors,
                                          args.anchor_drop_surfaces, args.max_anchors)
    n_dyn = sum(1 for n in nodes if n.get("moving"))
    print(f"{'video':<16}{args.video}   nodes {len(graph['nodes'])} -> anchors {len(nodes)}"
          f" (dyn {n_dyn} / stat {len(nodes) - n_dyn})"
          f"   ({', '.join(n['id'] + ':' + str(n.get('label', '')) for n in nodes)})")
    if dropped_anchors:
        print(f"{'dropped':<16}" + "  ".join(f"{i}:{l}[{why}]" for i, l, why in dropped_anchors))
    # D166. anchor 별 preset 목록 (`--preset_route`). 없으면 전 anchor 가 `presets` 합집합.
    route_presets, route_backfill = {}, {}
    free_preset, target_count, object_budget = None, 0, 0
    if args.preset_route:
        with open(args.preset_route, encoding="utf-8") as file:
            blob = json.load(file)
        assert blob.get("format") == "preset_route_v1", f"모르는 포맷: {blob.get('format')}"
        route_presets = {a["anchor_id"]: [s["preset"] for s in a["slots"]]
                         for a in blob["anchors"]}
        # D166 backfill. 라우팅이 keep_pair 밖으로 밀어 둔 슬롯. anchor 가 아래 frame 0 가시성
        # 게이트에서 죽거나 track 게이트에서 슬롯이 빠져 `target_count` 에 못 미치면 여기서 채운다.
        route_backfill = {a["anchor_id"]: [s["preset"] for s in a.get("backfill", [])]
                          for a in blob["anchors"]}
        free_preset = (blob.get("free_moving") or {}).get("preset")
        target_count = args.target_variants or int(blob.get("target_count", 0))
        object_budget = max(target_count - (1 if free_preset else 0), 1) if target_count else 0
        print(f"{'preset_route':<16}" + "  ".join(f"{k}:{len(v)}"
                                                  for k, v in route_presets.items())
              + f"   free {free_preset}  target {target_count} (object {object_budget})")
        # 라우팅이 뺀 anchor 는 여기서도 뺀다. `route_presets.py --min_anchor_sep` 가 부속물
        # anchor(모자/선글라스)를 버리는데, 이쪽은 `pick_main_anchors` 를 따로 불러 anchor 를
        # 고르므로 그대로 두면 그 노드가 **`presets` 합집합으로 되살아난다** (D127 이 경고한
        # "라우팅 표에 없는 anchor 가 뱅크엔 있는" 상태). `--preset_route` 를 주면 그 JSON 이
        # anchor 목록의 정본이다.
        cut = [n["id"] for n in nodes if n["id"] not in route_presets]
        nodes = [n for n in nodes if n["id"] in route_presets]
        if cut:
            print(f"{'route_cut':<16}" + " ".join(cut) + "  (preset_route 에 없는 anchor)")
        # D198. anchor **집합은 라우팅 JSON 그대로** 두고 preset 목록만 갈아끼운다
        # (사용자 지시 2026-09-15 "시도해본 preset 말고 남는 preset 으로 돌려볼 수 있어").
        # 라우팅은 슬롯마다 preset 을 하나씩 박으므로 어휘 43종 중 anchor 가 실제로 시도하는 건
        # 8종 이하다 — d185 코퍼스 전체에서 22종이 **한 번도** 안 나왔다. 그 잔량을 구우려면
        # anchor 는 같아야 한다 (캡션·instance_desc 가 anchor_id 로 붙어 있다).
        # 왜 `--presets` 로 안 되나: `plan_variants` 는 `route_presets.get(id, presets)` 라
        # anchor 가 라우팅 표에 있으면 `--presets` 를 아예 안 본다.
        # `--preset_route` 없이 `--presets` 만 주는 옛 경로는 그대로다 (분기 안 탄다).
        if args.route_preset_override:
            route_presets = {a: list(args.route_preset_override) for a in route_presets}
            route_backfill = {a: [] for a in route_presets}
            free_preset, target_count, object_budget = None, 0, 0   # 예산 없이 전량 굽는다
            # `presets` 도 같이 갈아끼워야 한다. 이건 `--presets` 인데, 라우팅이 `--emit args`
            # 로 **자기 슬롯 합집합**을 적어 보내므로(`route_presets.py:691`) 안 바꾸면 옛 8종이
            # 그대로 남는다. 그 값이 `bank.json` 의 `axes.presets` 로 실리고, `fit_hole_ladder`
            # 는 그 축을 필터로 쓴다(`:805`) — τ 뱅크엔 새 preset 이 30개 들어 있는데 fit 이
            # 0개로 읽고 `skipped.json` 만 쓰고 나간다 (D198 파일럿 24편 전량이 이렇게 비었다).
            presets = list(args.route_preset_override)
            print(f"{'route_override':<16}{' '.join(args.route_preset_override)}"
                  f"   (anchor {len(route_presets)}개, 예산/free-moving 해제)")
    print(f"{'presets':<16}{len(presets)}   tau ladder {ladder}")
    print(f"{'speeds':<16}{args.speeds}   trackings {args.trackings}   "
          f"look_at_bias {args.look_at_biases}\n")

    # `focal_all` 은 프레임별 intrinsic zoom 배율이다. SE(3) 밖이라 `cam_c2w` 로는 못 나른다 —
    # 렌더러가 K 를 프레임마다 바꿔 줘야 화각 변화가 실제로 보인다.
    rows, poses_all, focal_all, previews, skipped, saturated = [], [], [], [], [], []
    track_skipped = []
    timing = []                          # D165. (anchor, preset) 별 소요초 — `--timing_json` 전용
    header = (f"{'variant':<52}{'tau*':>6}{'tau_max':>9}{'path_u':>8}"
              f"{'vang':>7}{'hole':>7}{'subj':>7}{'inFr':>6}")
    print(header)
    print("-" * len(header))

    # ── 1단계: anchor 가 소스 frame 0 에서 실제로 보이는지 (여기서만 거른다) ──────────────
    # D166 이전에는 이 검사가 preset 루프와 한 몸이라, 죽은 anchor 는 그 자리에서 `continue` 로
    # 사라지고 **슬롯도 같이 사라졌다**. grid2x2 (anchor 2 x 슬롯 2 + free 1 = 5) 에서는 이게
    # 치명적이다 — 파일럿 16편 중 5편이 anchor 하나를 잃어 5개가 3개(또는 2개)로 떨어졌다.
    # 그래서 가시성 판정을 **먼저 전량 돌려** 살아남은 anchor 를 확정한 뒤 예산을 배분한다.
    # `subject_points` 는 여기서 캐시한다 (재계산하면 anchor 당 point mask 를 두 번 만든다).
    live = []
    for node in nodes:
        volume = subject_track_volume(recon, node)
        subject_points = subject_point_mask(renderer.indices, volume).cpu().numpy()
        renderer.set_subject(subject_points)
        base = renderer.render(renderer.cam_c2w_src[0], frame=0,
                               height=args.tile_height, width=args.tile_width)
        base_area = float(renderer.measure(base).get("subject_area", 0.0))
        if int(subject_points.sum()) < args.min_subject_points or base_area < args.min_subject_area:
            skipped.append((node["id"], node["label"], int(subject_points.sum()), base_area))
            continue
        live.append((node, subject_points))

    # ── 2단계: 예산 배분 ───────────────────────────────────────────────────────────────
    def track_ok(node, preset):
        """`track_*` 게이트. 떨어지면 `track_skipped` 에 남기고 False (D77 / D128 근거는 아래)."""
        if not preset.startswith("track_"):
            return True
        if args.track_dynamic_only and not node.get("moving"):
            track_skipped.append((node["id"], preset))
            return False
        if (args.track_min_drift_u > 0
                and float(node.get("center_drift_u", 0.0)) <= args.track_min_drift_u):
            track_skipped.append((node["id"], preset))
            return False
        return True

    plan, plan_tiers = plan_variants(live, route_presets, route_backfill, free_preset,
                                     object_budget, target_count, presets, track_ok,
                                     args.variant_pool)
    if target_count:
        n_var = sum(len(p) for _, _, p in plan)
        n_primary = sum(1 for t in plan_tiers.values() if t == 0)
        print(f"{'plan':<16}" + "  ".join(f"{n['id']}:{len(p)}" for n, _, p in plan)
              + f"   총 {n_var}/{target_count}"
              + ("" if n_primary >= target_count else "  (backfill 소진)")
              # D168 ②. 예비를 같이 구웠으면 층별 수를 찍는다 — fit 이 여기서만 갈아끼울 수 있다.
              + ("" if args.variant_pool == "budget" else
                 "   예비 " + " ".join(
                     f"t{t}:{sum(1 for v in plan_tiers.values() if v == t)}"
                     for t in sorted({v for v in plan_tiers.values() if v}))))

    # ── 3단계: 실제 굽기 ──────────────────────────────────────────────────────────────
    for node, subject_points, node_presets in plan:
        renderer.set_subject(subject_points)
        span = max(float(node["obs_az_span_deg"]), 1e-6)
        span_frac = max(args.orbit_span_frac, args.min_sweep_deg / span)

        for preset in node_presets:
            # D77. `track_*` 의 유일한 차이는 `PRESET_FOLLOW` 가 켜는 follow_gain 1.0 —
            # anchor 의 **world 변위**를 카메라 위치에 더한다. 안 움직이는 anchor 는 변위가 0 이라
            # 궤적이 비-track 짝과 비트 단위로 같아진다. 그대로 두면 "같은 카메라 / 다른 캡션"
            # 쌍이 대량 생겨 텍스트 조건이 오염되므로(모델이 track 어휘를 궤적과 못 묶는다)
            # 여기서 자른다. 판정은 `node["moving"]` — 이미 graph 가 실측해 둔 플래그다
            # (실측: 484 노드 중 moving=True 124개, 전부 kind="dyn"; dyn 이어도 52개는 안 움직인다.
            #  `path_len_u` 로 직접 자르지 않는 이유는 stat 노드의 OBB 중심이 재적합 지터로
            #  최대 21 u 까지 흔들려 이동과 구분이 안 되기 때문).
            #
            # **D128.** 위 게이트는 `moving` 하나만 보는데, 그 `moving` 은 `path_len_u > 0.05`
            # 라 **제자리 흔들림**(춤·손짓·그네)을 통과시킨다. 순변위가 0.05 u 이하인 49 노드가
            # d121 뱅크에서 전부 게이트를 통과했고(path_len_u min 0.053 / med 0.108), 그 위에
            # track_* 2040 쌍이 실렸다. 그 쌍들의 track vs 비-track 차이를 재보면
            # |Δpath_len| med 0.0336 / |Δτ| med 0.0210 — 진짜 이동하는 anchor(0.1285 / 0.2071)
            # 대비 10배 약하다. 비트 동일은 아니지만 "follows the subject" 캡션을 떠받치기엔
            # 얇다. 그래서 `path_len` 이 아니라 **순변위** `center_drift_u` 로 한 번 더 자른다.
            #
            # `moving` 자체를 순변위로 바꾸지 않는 이유(F1 기각): `moving` 은 `schema.py:207` 의
            # anchor split key 라, 강등하면 그 dyn 노드가 `max_static` 버킷으로 넘어가 벽·바닥과
            # 3칸을 다툰다 (`swing` 은 dyn 6개가 전부 넘어가 dynamic anchor 가 0 이 된다).
            # 제자리에서 움직여도 dyn 인 건 맞으므로 라우팅에만 건다.
            # `--track_min_drift_u 0` 이면 d121 과 비트 동일.
            #
            # **D166.** 위 두 게이트의 집행부는 `track_ok()` 로 올라갔다 (`plan_variants` 가
            # 호출한다). 여기서 `continue` 하면 슬롯이 조용히 비는데, 배분 단계에서 걸러야
            # 그 자리를 backfill 로 채울 수 있기 때문이다. 판정 자체는 한 글자도 안 바뀌었다.
            # 정지 preset 은 τ 에 반응하지 않는다 (어떤 s 든 같은 궤적) — 사다리 첫 단만 돈다.
            rungs = [ladder[0]] if preset in STATIC_PRESETS else ladder
            # D165. preset 별 실측 시간 (τ 단계). `--timing_json` 없으면 아무 데도 안 쓴다.
            _t_preset, _rows0 = perf_counter(), len(rows)
            for target_tau in rungs:
                for speed in args.speeds:
                    for tracking in args.trackings:
                        for bias in args.look_at_biases:
                          for follow in args.follow_gains:
                           for smooth in args.follow_smooths:
                            # gain 0 이면 offset 이 통째로 0 이라 창 크기가 궤적을 못 바꾼다.
                            # 안 자르면 완전히 같은 변이가 창 수만큼 복제된다 (D73).
                            if str(follow) == "0" and smooth != args.follow_smooths[0]:
                                continue
                            for keyframes in args.aim_keyframes:
                             decision = make_decision(graph, node, preset, target_tau, speed,
                                                      tracking, bias, args.start_mode,
                                                      shape=rung_shape(preset, target_tau, ladder,
                                                                       args.pan_deg_at_max,
                                                                       args.sweep_deg,
                                                                       args.pan_deg,
                                                                       args.fit_tau),
                                                      tau_ref=args.tau_ref)
                             poses, extra = build_poses(
                                 decision, graph, board=None, num_frames=num_frames,
                                 orbit_span_frac=span_frac, start_mode=args.start_mode,
                                 aim_anchor=args.aim_anchor, aim_ramp_frames=args.aim_ramp_frames,
                                 traj_basis=args.traj_basis, aim_keyframes=keyframes,
                                 keyframe_aim=args.keyframe_aim, keyframe_ease=args.keyframe_ease,
                                 follow_gain=follow, follow_smooth=smooth,
                                 use_fit_tau=args.fit_tau,
                                 preset_tracking=args.preset_tracking, deroll=args.deroll,
                                 orbit_fixed_sweep=args.orbit_fixed_sweep,
                                 tau_denom=args.tau_denom)
                             info = extra["info"]
                             # 시작 pose 만으로 예산을 다 쓴 단 — `fit_tau` 가 움직임을 0 으로 눌러서
                             # 나오는 건 `static_hold` 의 복제본이다. 세어만 두고 뱅크에서 뺀다.
                             # 정지 preset 자신은 예외다 — 거기서는 identity 가 원래 의도한 결과다.
                             if (info["tau"]["tau_saturated"] and args.drop_saturated
                                     and preset not in STATIC_PRESETS):
                                 # follow 를 같이 남긴다 — 안 그러면 "몇 개가 왜 죽었나" 가
                                 # gain 별로 안 갈려서 진단이 안 된다 (D72).
                                 saturated.append((node["id"], preset, target_tau,
                                                   info["tau"]["tau_start"],
                                                   info["follow"]["gain"]))
                                 continue
                             stats, middle = measure_trajectory(
                                 renderer, poses, num_frames, args.measure_frames,
                                 args.tile_height, args.tile_width, args.center_box,
                                 behind=behind, focal=extra["focal_scale"],
                                 subject_points=subject_points if args.subject_visible else None)
                             # follow 를 안 쓰면 변이 이름을 예전 그대로 둔다 (이미 만든 뱅크와
                             # variant_id 로 맞대조할 수 있게).
                             # D93. `track_*` 은 `PRESET_TRACKING` 이 조준을 lock 으로 푼다 —
                             # 이름과 행에 **실제 쓴 값**을 써야 뱅크로 되만들 때 일치한다.
                             tracking_used = info["tracking"]
                             variant = (f"{node['id']}__{preset}__tau{target_tau:g}"
                                        f"__{speed}__{tracking_used}__b{bias:g}"
                                        + (f"__f{follow}" if str(follow) != "0" else "")
                                        # follow 를 안 쓰면 창 크기는 궤적에 영향이 없다 — 이름에
                                        # 넣으면 없는 축으로 변이가 갈라진다 (D73).
                                        + (f"__s{smooth}" if str(follow) != "0" else "")
                                        + (f"__k{keyframes}" if int(keyframes) else ""))
                             rows.append({
                                 "variant_id": variant, "anchor_id": node["id"],
                                 "anchor_label": node["label"], "preset": preset,
                                 # D168 ②. 0 = 라우팅이 준 본 슬롯, 1/2/3 = fallback 예비.
                                 # `--variant_pool budget` 이면 전부 0 이라 예전과 같은 뜻이다.
                                 "plan_tier": plan_tiers.get((node["id"], preset), 0),
                                 "target_tau": target_tau, "speed": speed,
                                 "tracking": tracking_used,
                                 "tracking_requested": tracking,
                                 "look_at_bias": bias, "aim": info["aim"],
                                 # D72. `follow_gain` 은 요청값이 아니라 **실제 쓴 값**이다
                                 # ("auto" 면 여기서 풀린 값). kind 는 CameraBench 갈래.
                                 "follow_gain": info["follow"]["gain"],
                                 "follow_kind": info["follow"].get("kind", "none"),
                                 # D73. follow 위치 채널 저역통과 창 (1 = 끔).
                                 "follow_smooth": int(smooth),
                                 # D71. 조준 keyframe 축. 0 이면 나머지 두 열은 없다 (조준이
                                 # 매 프레임이거나 아예 없다) — 뱅크를 열별로 거를 수 있게 남긴다.
                                 "aim_keyframes": int(keyframes),
                                 "keyframe_turn_deg": ((info["aim_keyframes"] or {})
                                                       .get("turn_deg", 0.0)),
                                 "keyframe_aim_err_deg": ((info["aim_keyframes"] or {})
                                                          .get("aim_err_max_deg", 0.0)),
                                 "tau_scale": info["tau"]["scale"],
                                 "tau_start": info["tau"]["tau_start"],
                                 "tau_saturated": info["tau"]["tau_saturated"],
                                 # D97. `tau_max` 는 **언제나 소스 기준** 시차다 (hole 예산과
                                 # 같은 눈금이라 게이트·verify 가 이걸 읽는다). 이분법이 실제로
                                 # 맞춘 값은 `tau_ref` 가 가리키는 기준 위의 `tau_ref_max` 다 —
                                 # `tau_ref="source"` 면 둘이 같고, `"follow"` 면 후자만
                                 # `target_tau` 와 맞는다. 두 열을 안 나누면 "target 0.35 인데
                                 # tau_max 0.9" 가 버그로 보인다.
                                 "tau_ref": info["tau"]["tau_ref"],
                                 "tau_ref_max": info["tau"]["tau_max"],
                                 "tau_max": info["tau"]["tau_max_final"],
                                 "path_len_u": info["path_len_u"],
                                 "view_angle_max_deg": info["view_angle_max_deg"],
                                 "radius_u": info["radius_u"],
                                 "sweep_deg": info["shape_context"]["sweep"],
                                 # 실제로 나간 회전각 (`fit_tau` 의 scale 이 곱해진 뒤).
                                 "pan_deg": round(info["shape_context"]["pan"]
                                                  * info["tau"]["scale"], 3),
                                 "tracking_ignored": info["tracking_ignored"],
                                 # 이동이 없어 τ 가 안 움직이는 preset. 사다리를 각도로 타거나(회전 전용)
                                 # 아예 안 탄다(정지) — τ 단별 요약 평균에서 빼야 흐려지지 않는다.
                                 "tau_invariant": (preset in STATIC_PRESETS
                                                   or preset in ROTATION_ONLY_PRESETS),
                                 # 마지막 프레임 focal 배율 (1.0 = zoom 없음). `fit_tau` 와 무관.
                                 "focal_end": info["zoom"]["focal_end"],
                                 "fit_tau": info["tau"]["fit_tau"], **stats})
                             poses_all.append(poses)
                             focal_all.append(extra["focal_scale"])
                             previews.append((variant, middle))
                             print(f"{variant:<52}{target_tau:>6.2f}"
                                   f"{info['tau']['tau_max_final']:>9.4f}{info['path_len_u']:>8.4f}"
                                   f"{info['view_angle_max_deg']:>7.1f}"
                                   f"{stats['hole_fraction']:>7.3f}"
                                   f"{stats['subject_area_med']:>7.3f}"
                                   f"{stats['subject_in_frame']:>6.2f}")

            timing.append({"anchor_id": node["id"], "preset": preset, "rungs": len(rungs),
                           "variants": len(rows) - _rows0,
                           "seconds": round(perf_counter() - _t_preset, 3)})

    # 변이 0 은 "이 영상은 뱅크를 못 만든다"는 정상적인 결말이기도 하다 (anchor 가 소스 frame 0
    # 에서 전부 안 보이는 경우). 기본값은 예전처럼 assert 로 죽지만, 대량 러너는 이걸 실패로
    # 받으면 산출물이 안 남아 "덜 된 영상"으로 보고 매 pass 마다 graph→cloud→bank 를 다시 돈다.
    # `--skip_on_empty` 를 주면 skipped.json 을 남기고 정상 종료해서 그 영구 재시도를 끊는다.
    if not rows and args.skip_on_empty:
        skip_doc = {"format": "lbm_camera_bank_skipped_v1", "video": args.video,
                    "reason": "no_surviving_anchors",
                    "detail": "변이가 0 — anchor 가 전부 걸러졌다 (소스 frame 0 에서 안 보이거나 "
                              "화면 점유가 --min_area_frac 미만).",
                    "min_area_frac": args.min_area_frac,
                    "skipped_anchors": [{"id": i, "label": l, "points": p, "subject_area": round(a, 5)}
                                        for i, l, p, a in skipped]}
        with open(path.join(bank_folder, "skipped.json"), "w", encoding="utf-8") as file:
            json.dump(skip_doc, file, ensure_ascii=False, indent=1)
        print(f"\nSKIP  {args.video}  변이 0 (anchor {len(skipped)} 전량 탈락) "
              f"-> {path.join(bank_folder, 'skipped.json')}")
        return

    assert rows, "변이가 0 — anchor 가 전부 걸러졌다. skipped 목록과 --min_subject_area 를 볼 것"

    # 부분 추출: τ 단으로 층화해 골고루 뽑는다 (그냥 균등추출하면 preset 수가 많은 단이 이긴다).
    keep = list(range(len(rows)))
    if args.num_samples and args.num_samples < len(rows):
        rng = np.random.default_rng(args.seed)
        by_rung = {}
        for i, row in enumerate(rows):
            by_rung.setdefault(row["target_tau"], []).append(i)
        per = max(1, args.num_samples // len(by_rung))
        keep = []
        for rung in sorted(by_rung):
            pool = by_rung[rung]
            keep.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
        rest = [i for i in range(len(rows)) if i not in set(keep)]
        if len(keep) < args.num_samples and rest:
            keep.extend(rng.choice(rest, size=min(args.num_samples - len(keep), len(rest)),
                                   replace=False).tolist())
        keep = sorted(keep)

    selected = [rows[i] for i in keep]
    bank = {"format": "lbm_camera_bank_v1", "video": args.video, "num_frames": num_frames,
            "start_mode": args.start_mode, "aim_anchor": args.aim_anchor,
            "aim_ramp_frames": args.aim_ramp_frames,
            "axes": {"anchors": [n["id"] for n in nodes], "presets": presets,
                     "tau_ladder": ladder, "speeds": args.speeds, "trackings": args.trackings,
                     "look_at_biases": args.look_at_biases,
                     "follow_gains": list(args.follow_gains),
                     "follow_smooths": list(args.follow_smooths),
                     "aim_keyframes": list(args.aim_keyframes)},
            "traj_basis": args.traj_basis,
            # D168 ②. `"full"` 이면 예산 밖 preset 까지 변이로 깔아 뒀다는 뜻이고, 그때만
            # `fit_hole_ladder.py --fallback_ladder` 가 층을 나눠 돌 수 있다. `"budget"` 이면
            # 전 행의 `plan_tier` 가 0 이라 사다리를 켜도 no-op 이다.
            "variant_pool": str(args.variant_pool),
            "plan_tiers": {str(t): sum(1 for v in plan_tiers.values() if v == t)
                           for t in sorted(set(plan_tiers.values()))},
            "keyframe_aim": args.keyframe_aim, "keyframe_ease": args.keyframe_ease,
            # D99. 중력 기준 roll 보정 여부. τ 뱅크와 hole 사다리가 어긋나면 손잡이가
            # 다른 궤적 위에서 풀린다 — 두 스크립트에 같은 값을 줄 것.
            "deroll": bool(args.deroll),
            # F9. 이분법이 sweep 을 깎았는지(False) 반경을 깎았는지(True). deroll 과 같은 이유로
            # hole 사다리와 **같은 값**이어야 한다.
            "orbit_fixed_sweep": bool(args.orbit_fixed_sweep),
            # F2. `S` 의 정의. 게이트 임계가 전부 S 배율이라 이 값이 다르면 같은
            # `behind_margin_frac` 이 다른 거리를 뜻한다.
            "scale_mode": str(scale_mode),
            # D150. τ 의 분모가 어느 정의인가 (`scale_mode` 와 같은 이유).
            "tau_denom": str(args.tau_denom),
            # D116. G1 증거 소스와 시간축 정합. deroll 과 같은 이유로 hole 사다리와 **같은 값**
            # 이어야 한다 — 다르면 사다리가 τ 뱅크에서 이미 걸러진 변이를 되살린다.
            # `"off"` = G1 을 아예 안 쟀다 (`--no_measure_behind`, τ 뱅크 기본).
            "collision": {"source": (args.collision_source if args.measure_behind else "off"),
                          "mesh_grid": (mesh_grid or None),
                          "time_match": bool(args.collision_time_match),
                          # D123. 이것도 사다리와 같은 값이어야 한다 (같은 이유).
                          "min_zcam_frac": float(args.behind_min_zcam)},
            "external_shapes": {"path": args.external_shapes, "aim": args.external_aim,
                                "presets": external},
            "shape": {"orbit_span_frac": args.orbit_span_frac,
                      "min_sweep_deg": args.min_sweep_deg,
                      "sweep_deg": args.sweep_deg,
                      "rotation_only_presets": list(ROTATION_ONLY_PRESETS),
                      "pan_deg": args.pan_deg,
                      "pan_deg_at_max": args.pan_deg_at_max},
            "fit_tau": args.fit_tau,
            # D93. `track_*` 의 조준 추종률을 위치 추종률에 맞춘다 (§presets.py `PRESET_TRACKING`).
            # 이 키가 **없는** 뱅크는 D93 이전이라 소비자가 False 로 읽어 옛 궤적을 그대로 되푼다.
            "preset_tracking": args.preset_tracking,
            # D97. τ 를 무엇에 대해 잴지의 **요청값**이다 (resolve 전). 이 키가 없는 뱅크는
            # D97 이전이라 소비자가 "source" 로 읽어 옛 궤적을 그대로 되푼다.
            "tau_ref": args.tau_ref,
            # 렌더 때도 같은 값을 줘야 hole 측정과 영상이 어긋나지 않는다.
            "fixed_focal": args.fixed_focal,
            "dropped_saturated": [{"anchor_id": a, "preset": p, "target_tau": t, "tau_start": s,
                                   "follow_gain": g}
                                  for a, p, t, s, g in saturated],
            "measure": {"frames": args.measure_frames, "height": args.tile_height,
                        "width": args.tile_width, "center_box": args.center_box,
                        "behind": (behind and {"src_frames": behind["frames"],
                                               "margin_frac": behind["margin_frac"],
                                               "clear_frac": behind["clear_frac"],
                                               "radius_px": behind["radius_px"],
                                               "near_pct": behind["near_pct"],
                                               "standoff": behind["standoff"],
                                               "obb": bool(behind["nodes"])}) or None,
                        "note": "hole_fraction 은 게이트가 아니라 기록용 지표다"},
            "sampling": {"num_samples": args.num_samples, "seed": args.seed,
                         "enumerated": len(rows), "selected": len(selected)},
            "skipped_anchors": [{"id": i, "label": l, "points": p, "subject_area": round(a, 5)}
                                for i, l, p, a in skipped],
            # `z_med` 는 이름만 예전 것이고 실제로는 **τ 의 분모**다 (D150).
            "S": float(graph["scale"]["S"]),
            "z_med": tau_denominator(graph, args.tau_denom),
            "variants": selected}
    with open(path.join(bank_folder, "bank.json"), "w", encoding="utf-8") as file:
        json.dump(bank, file, ensure_ascii=False, indent=1)

    columns = ["variant_id", "anchor_id", "anchor_label", "preset", "plan_tier",
               "target_tau", "speed",
               "tracking", "look_at_bias", "follow_gain", "follow_kind", "follow_smooth",
               "aim_keyframes", "keyframe_turn_deg", "keyframe_aim_err_deg",
               "aim", "tau_scale", "tau_start", "tau_saturated",
               "tau_ref", "tau_ref_max", "tau_max", "path_len_u", "view_angle_max_deg", "radius_u", "sweep_deg", "pan_deg",
               "tracking_ignored", "tau_invariant", "focal_end", "fit_tau",
               "hole_fraction", "hole_max",
               "subject_area_med", "subject_in_frame",
               # `--no_subject_visible` 이면 빈 칸 (D81).
               "subject_visible_frac", "subject_visible_min",
               # `--measure_behind` 일 때만 채워진다 (아니면 빈 칸).
               "behind_frames", "behind_frac", "behind_worst_src",
               # `--collision_time_match` 일 때만 채워지는 열 (2026-09-02). `*_frac` 둘은
               # 진단이 아니라 하류 `fit_hole_ladder` 가 **판정에 쓰는 값**이다.
               "behind_static_frames", "behind_static_frac", "behind_static_worst",
               "behind_dyn_frames", "behind_dyn_frac", "behind_dyn_worst", "standoff",
               "obb_clear", "obb_slack", "obb_node"]
    with open(path.join(bank_folder, "bank.csv"), "w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")
        for row in selected:
            file.write(",".join(str(row.get(c, "")) for c in columns) + "\n")

    np.savez_compressed(path.join(bank_folder, "poses.npz"),
                        cam_c2w=np.stack([poses_all[i] for i in keep]),
                        # (V, n) intrinsic zoom. zoom 없는 변이는 전부 1.0 이라 렌더러가 그냥
                        # K_src 를 쓰는 것과 같다 — 옛 뱅크엔 이 키가 없으므로 소비자는 optional.
                        focal_scale=np.stack([focal_all[i] for i in keep]),
                        variant_id=np.array([r["variant_id"] for r in selected]),
                        anchor_id=np.array([r["anchor_id"] for r in selected]),
                        target_tau=np.array([r["target_tau"] for r in selected]))

    if args.preview and previews:
        import imageio.v2 as imageio
        step = max(1, len(keep) // args.preview_max)
        tiles = []
        for i in keep[::step][:args.preview_max]:
            variant, rendered = previews[i]
            row = rows[i]
            tiles.append(label_tile(paint_holes(rendered["rgb"], rendered["valid"]),
                                    row["preset"][:18],
                                    f"{row['anchor_id']} tau{row['target_tau']:g} "
                                    f"hole{row['hole_fraction']:.2f} "
                                    f"path{row['path_len_u']:.2f}u"))
        imageio.imwrite(path.join(bank_folder, "preview.png"),
                        contact_sheet(tiles, args.preview_columns))

    # τ 단별 요약 — 사다리의 요점이 이 표다.
    moving = [r for r in selected if not r["tau_invariant"]]
    print(f"\n{'tau*':<8}{'n':>5}{'tau_max':>10}{'path_u':>9}{'vang':>8}{'hole':>8}"
          f"{'hole_max':>10}{'inFrame':>9}{'saturated':>11}   (정지 preset 제외)")
    for rung in sorted({r["target_tau"] for r in moving}):
        group = [r for r in moving if r["target_tau"] == rung]
        mean = lambda key: float(np.mean([r[key] for r in group]))
        print(f"{rung:<8.2f}{len(group):>5}{mean('tau_max'):>10.4f}{mean('path_len_u'):>9.4f}"
              f"{mean('view_angle_max_deg'):>8.1f}{mean('hole_fraction'):>8.3f}"
              f"{mean('hole_max'):>10.3f}{mean('subject_in_frame'):>9.2f}"
              f"{sum(r['tau_saturated'] for r in group):>11}")

    # 회전 전용 preset 은 같은 사다리를 **각도**로 탄다 — 위 표와 단위가 달라 따로 찍는다.
    rotating = [r for r in selected if r["preset"] in ROTATION_ONLY_PRESETS]
    if rotating:
        print(f"\n{'tau*':<8}{'n':>5}{'pan_deg':>10}{'hole':>8}{'hole_max':>10}{'inFrame':>9}"
              f"   (회전 전용: 사다리를 각도로)")
        for rung in sorted({r["target_tau"] for r in rotating}):
            group = [r for r in rotating if r["target_tau"] == rung]
            mean = lambda key: float(np.mean([r[key] for r in group]))
            print(f"{rung:<8.2f}{len(group):>5}{mean('pan_deg'):>10.1f}{mean('hole_fraction'):>8.3f}"
                  f"{mean('hole_max'):>10.3f}{mean('subject_in_frame'):>9.2f}")

    if saturated:
        # 시작 pose 가 이미 예산을 넘긴 단. 게이트가 아니라 **중복 제거**다 (전부 static 복제본).
        # gain 별로 갈라서 센다 — follow 가 사다리를 얼마나 되살렸는지가 여기서만 보인다 (D72).
        by_rung = {}
        for _, _, rung, tau_start, gain in saturated:
            by_rung.setdefault((gain, rung), []).append(tau_start)
        print("\ndropped (tau_saturated — 시작 pose 만으로 예산 초과, static_hold 복제본):")
        for gain, rung in sorted(by_rung):
            starts = by_rung[(gain, rung)]
            print(f"  follow {gain:<7.3f}tau* {rung:<6.2f}n {len(starts):<6}"
                  f"tau_start {float(np.mean(starts)):.4f}")

    if track_skipped:
        # 정지 anchor 에서 잘린 track_* 변이. anchor 별로 세면 "이 씬에 움직이는 게 없다" 가 보인다.
        by_anchor = {}
        for node_id, preset in track_skipped:
            by_anchor.setdefault(node_id, set()).add(preset)
        print(f"\ndropped track_* on static anchors (follow 변위 0 = 비-track 짝과 동일 궤적): "
              f"{len(track_skipped)} pairs / {len(by_anchor)} anchors")
        for node_id in sorted(by_anchor):
            print(f"  {node_id:<10}presets {len(by_anchor[node_id])}")

    if skipped:
        print("\nskipped anchors (소스 frame 0 에서 안 보인다):")
        for node_id, label, points, area in skipped:
            print(f"  {node_id:<10}{label:<18}points {points:>9,}  subject_area {area:.5f}")
    print(f"\nenumerated {len(rows)}  selected {len(selected)}")
    print(f"-> {bank_folder}")

    # D165. preset 별 소요시간. 안 주면 아무것도 안 찍고 안 쓴다 (기존 출력 비트 동일).
    if args.timing_json:
        makedirs(path.dirname(path.abspath(args.timing_json)), exist_ok=True)
        with open(args.timing_json, "w", encoding="utf-8") as file:
            json.dump({"format": "sample_camera_bank_timing_v1", "video": args.video,
                       "stage": "tau",
                       "total_seconds": round(sum(t["seconds"] for t in timing), 2),
                       "entries": timing}, file, ensure_ascii=False, indent=1)
        agg = {}
        for t in timing:
            a = agg.setdefault(t["preset"], {"n": 0, "sec": 0.0, "var": 0})
            a["n"] += 1; a["sec"] += t["seconds"]; a["var"] += t["variants"]
        print(f"\npreset 별 소요 (tau, anchor 합산)")
        print(f"{'preset':<26}{'n':>4}{'sec':>9}{'sec/anchor':>12}{'변이':>6}{'sec/변이':>10}")
        for pname, a in sorted(agg.items(), key=lambda kv: -kv[1]["sec"]):
            print(f"{pname:<26}{a['n']:>4}{a['sec']:>9.1f}{a['sec'] / a['n']:>12.2f}"
                  f"{a['var']:>6}{a['sec'] / max(a['var'], 1):>10.3f}")
        print(f"-> {args.timing_json}")


def build_parser():
    """파서를 **함수로** 꺼내 둔 이유: `run_bank.py --exec inproc` 이 이 스크립트를 서브프로세스가
    아니라 같은 프로세스에서 부른다 (D180). 파서가 `__main__` 블록 안에 있으면 import 로는
    만들 수 없다. CLI 동작은 그대로다 — 아래 `__main__` 이 이 함수를 쓴다.
    """
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)
    add_cloud_source_args(parser)

    parser.add_argument("--video", required=True, type=str)
    # 변이가 0 일 때 assert 로 죽는 대신 skipped.json 을 남기고 정상 종료한다. 대량 러너용 —
    # 안 주면 예전과 똑같이 AssertionError 로 죽는다.
    parser.add_argument("--skip_on_empty", action="store_true")
    # 출력 폴더. 이미 돌려둔 뱅크를 덮어쓰지 않으려면 새 이름을 준다.
    parser.add_argument("--bank_dir", default="bank", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    # D165. preset 별 소요시간을 JSON 으로. 안 주면 재기는 하되 아무 데도 안 쓴다 (출력 동일).
    parser.add_argument("--timing_json", default=None, type=str)

    # ── 축
    parser.add_argument("--nodes", nargs="*", default=None)             # None = 자동 (아래 하한)
    parser.add_argument("--min_area_frac", default=0.01, type=float)    # anchor 화면 점유 하한
    # D127b (사용자 지시). anchor = 촬영 **대상**이므로 편당 소수만 둔다. 동적/정적을 **따로**
    # 세야 동적 3개인 편에서 정적이 0 이 되지 않는다. 0 = 그 갈래 상한 없음(옛 동작).
    parser.add_argument("--max_dynamic_anchors", default=3, type=int)
    parser.add_argument("--max_static_anchors", default=3, type=int)
    parser.add_argument("--max_anchors", default=0, type=int)   # 둘을 적용한 뒤 거는 총합 상한
    # 벽/바닥/천장/울타리/난간/문/창을 anchor 에서 뺀다 (§schema.SURFACE_TOKENS).
    parser.add_argument("--anchor_drop_surfaces", action="store_true", default=True)
    parser.add_argument("--no_anchor_drop_surfaces", dest="anchor_drop_surfaces",
                        action="store_false")
    parser.add_argument("--presets", nargs="*", default=None)           # None = 전량
    # D166. `route_presets.py --out` 이 낸 `preset_route_v1` JSON. 주면 preset 을 **anchor 마다
    # 따로** 쓴다 — (nodes x presets) 합집합 격자를 안 돈다. 안 주면 기존 동작 그대로.
    # 왜 필요한가: `--slot_plan grid2x2` 는 anchor 두 개가 같은 *슬롯* 쌍을 쓰지만 preset 이름은
    # away side / track 여부 때문에 anchor 마다 다를 수 있다(`truck_left` vs `truck_right`).
    # 합집합으로 넘기면 그게 4개가 되어 격자가 2x4=8 로 부푼다. free-moving 슬롯도 같은 이유로
    # 합집합에 두면 anchor 전부에 걸려 scene 당 5개가 안 된다.
    parser.add_argument("--preset_route", default=None, type=str)
    # D198. `--preset_route` 의 **anchor 집합만** 쓰고 preset 목록은 이 값으로 갈아끼운다.
    # 라우팅이 이미 다녀간 씬에서 "안 써 본 preset" 만 추가로 구울 때 쓴다 (§route_override).
    # 주면 free-moving 슬롯과 scene 예산(`target_count`)이 같이 풀린다 — 예산을 남겨 두면
    # anchor 당 앞 몇 개만 굽고 나머지를 예비로 미뤄서 "잔량을 다 돌린다"는 목적을 못 지킨다.
    parser.add_argument("--route_preset_override", nargs="*", default=None)
    # D166. scene 당 변이 **상한**. 0 이면 `--preset_route` JSON 의 `target_count` 를 쓰고,
    # 그것도 없으면(옛 JSON) 배분 자체를 안 한다 = 옛 동작(anchor 전량 x preset 전량).
    # 여기서 덮어쓰는 건 라우팅을 다시 안 돌리고 상한만 바꿔 보고 싶을 때뿐이다.
    parser.add_argument("--target_variants", default=0, type=int)
    # D168 ② (사용자 지시 2026-09-08 "가능한 preset 들 최대한 돌리고 … 안전장치"). `full` 이면
    # 예산 밖 풀도 **예비 층**으로 같이 굽는다 (`plan_tier` 1/2/3). τ 뱅크가 대략 2배가 되는
    # 대신 `fit_hole_ladder --fallback_ladder` 가 죽은 본 슬롯을 갈아끼울 데가 생긴다.
    # `budget` 이 기본이고 그건 D167 뱅크와 비트 단위로 같다.
    parser.add_argument("--variant_pool", default="budget", choices=("budget", "full"))
    # D79. DataDoP 등 외부 코퍼스에서 검색한 실제 촬영 궤적을 preset 어휘에 끼워 넣는다.
    # `retrieve_datadop_shapes.py --out` 이 낸 `datadop_shapes_v1` 파일. **fit_hole_ladder 에도
    # 같은 파일을 줘야 한다** (뱅크 행에서 궤적을 다시 짓는다).
    parser.add_argument("--external_shapes", default=None, type=str)
    parser.add_argument("--external_aim", default="traj", type=str)      # traj|look_at
    parser.add_argument("--tau_ladder", nargs="*", default=list(TAU_LADDER), type=float)
    parser.add_argument("--speeds", nargs="*", default=["steady"])      # steady|accel|decel|ease
    # D127. `drift`(0.6) 삭제 (§presets.LEGACY_TRACKING). 기본값도 `lock` 으로 옮긴다 —
    # 조준 preset 은 어차피 `PRESET_TRACKING` 이 덮으므로 실제로 바뀌는 건 `aim="free"`
    # preset 의 기록값뿐이고, 그쪽은 `tracking_ignored` 라 궤적이 안 바뀐다.
    parser.add_argument("--trackings", nargs="*", default=["lock"],
                        choices=["world", "lock"])
    parser.add_argument("--look_at_biases", nargs="*", default=[0.0], type=float)
    # D72. subject 변위를 카메라 위치에 싣는 비율 = CameraBench tail/lead/side/aerial tracking.
    # 문자열인 이유는 "auto"(τ 최소 gain 을 풀어서 사용)를 같은 축에 섞기 위해서다.
    parser.add_argument("--follow_gains", nargs="*", default=["0"], type=str)
    # D73. follow 위치 채널 저역통과 창 (savgol p=2). 1 = 끔(D72 원래 동작). subject track 의
    # 잔여 고주파가 g 배로 카메라 위치에 실려 흔들림이 된다 — 실측 |jerk| p95 가 정확히 g×track.
    parser.add_argument("--follow_smooths", nargs="*", default=[9], type=int)
    # D93 -> D127. `aim="look_at"` preset 전부의 조준 추종률(`tracking`)을 `lock` 으로 강제한다.
    # 끄면 `--trackings` 값이 그대로 간다 (이제 `world` 를 주고 싶을 때만 의미가 있다).
    parser.add_argument("--preset_tracking", action="store_true", default=True)
    parser.add_argument("--no_preset_tracking", dest="preset_tracking", action="store_false")
    # D97. τ 를 무엇에 대한 변위로 잴지 (§presets.py `PRESET_TAU_REF`).
    #   source — 소스 카메라 기준 (D97 이전의 유일한 정의)
    #   follow — 추종 궤적 기준. τ 가 preset 모양의 **상대** 이동만 재므로 `track_*` 에서
    #            추종이 예산을 안 먹는다
    #   auto   — `track_*` 만 follow, 나머지는 source  (d115~d121 뱅크가 이것)
    # D128 에서 기본을 **`follow` 로 뒤집었다** (사용자 지시 2026-09-05 "tau_ref도 적용해서").
    # D125 parkour 프로브 실측: `auto` -> `follow` 로 stat_2/3/6 세 클립의 G1 위반 프레임 비율
    # (`behind`)이 0.347/0.286/0.163 -> 전부 0.000, 대신 `path_len_u` 가 0.254->0.159,
    # 0.334->0.143 로 줄었다 (stat_6 은 0.346->0.349 로 유지). 정확성을 사고 다양성을 판 것이다.
    parser.add_argument("--tau_ref", default="follow", choices=list(TAU_REF_CHOICES))
    # D150. τ 의 **분모**. 근거·배율은 `scene_graph.scale.tau_denominator` docstring.
    # 기본은 예전 정의 — 뱅크 정체성이라 기본을 뒤집으면 굽기 중간에 정의가 갈린다.
    parser.add_argument("--tau_denom", default="z_med_frame0", choices=list(TAU_DENOM_MODES))
    parser.add_argument("--allow_zoom", action="store_true", default=False)
    parser.add_argument("--no_allow_zoom", dest="allow_zoom", action="store_false")
    # D77. `track_*` 를 **움직이는 anchor** 에만 건다. 정지 anchor 에서는 follow 변위가 0 이라
    # 궤적이 비-track 짝과 완전히 같아지고, 캡션만 다른 쌍이 생겨 텍스트 조건이 오염된다.
    # `--no_track_dynamic_only` 로 예전 동작(전 anchor × 전 preset)을 그대로 되살릴 수 있다.
    parser.add_argument("--track_dynamic_only", action="store_true", default=True)
    # D128. `moving`(=`path_len_u > 0.05`) 위에 **순변위** 하한을 하나 더 건다. 근거는 위
    # 게이트 옆 주석 (d121 실측: 순변위 0.05 이하인 49 노드가 track_* 2040 쌍을 만들었고
    # track vs 비-track |Δτ| median 이 0.0210 — 진짜 이동 anchor 의 10분의 1). 0 이면 끈다.
    parser.add_argument("--track_min_drift_u", default=0.05, type=float)
    parser.add_argument("--no_track_dynamic_only", dest="track_dynamic_only",
                        action="store_false")
    # 소스 K 를 frame 0 에 고정한다. DA3 는 프레임마다 focal 을 다시 추정해서 정지 카메라에서도
    # 드리프트한다 — 의도한 zoom preset 과 그 드리프트가 섞이면 화각 변화를 못 가른다.
    # `PRESET_ZOOM_END` 배율은 이 고정된 K 위에 곱해진다.
    parser.add_argument("--fixed_focal", action="store_true", default=False)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")

    # ── 궤적 모양
    parser.add_argument("--start_mode", default="source_frame0", type=str)   # source_frame0|board
    parser.add_argument("--aim_anchor", default="subject", type=str)
    parser.add_argument("--aim_ramp_frames", default=12, type=int)
    # source|subject (D69) | auto (D99: look_at -> subject, free -> source. §presets 참조).
    # 기본값은 `fit_hole_ladder.py` 와 **항상 같아야 한다** — 다르면 같은 뱅크가 두 규약으로 섞인다.
    parser.add_argument("--traj_basis", default="source", type=str,
                        choices=["source", "subject", "auto"])
    # D71. 조준 keyframe 수 (뱅크 축). 0 = 기존 동작(매 프레임 조준 또는 조준 없음).
    # N>=2 면 frame 0 은 소스 회전 그대로, 나머지 N-1 개 keyframe 만 anchor 를 보고
    # 사이 49-N 프레임은 SO(3) 측지선으로 보간한다.
    parser.add_argument("--aim_keyframes", nargs="*", default=[0], type=int)
    parser.add_argument("--keyframe_aim", default="auto", type=str)      # auto|target|preset_rel
    # D96. 기본값이 `smooth_kf` 다 (D71~D94 는 `smoothstep`). 근거는 `build_poses.py:965` 의
    # 714변이 실측 — 각속도 맥동비 21.65 -> 4.58, 49프레임 조준오차 median 0.668° -> 0.418°.
    # 평활 인자(`smooth_passes`/`smooth_lambda`)는 여기서 안 넘긴다 = `build_poses` **서명**
    # 기본값 12/0.5 이고, 이건 `fit_hole_ladder.SHAPE_DEFAULTS` 와 **같은 값이어야 한다**.
    # 어긋나면 τ 뱅크와 hole 사다리의 회전만 조용히 달라진다 (위치는 정확히 맞는다 — D90).
    parser.add_argument("--keyframe_ease", default="smooth_kf", type=str)
    # D99. 중력 기준 roll 보정. **`fit_hole_ladder` 의 기본값과 같아야 한다** — 여기가 τ 뱅크고
    # 저기가 사다리라, 갈리면 같은 손잡이가 다른 궤적 위에서 풀린다 (D90 의 smooth_passes 사고).
    # D105. 기본값 True (d98 까지는 False). 되만들 때는 `--no_deroll`.
    parser.add_argument("--deroll", action="store_true", default=True)
    parser.add_argument("--no_deroll", dest="deroll", action="store_false")
    parser.add_argument("--orbit_span_frac", default=0.8, type=float)
    # 좁은 obs_az_span 노드에서 orbit 이 직선으로 무너지는 걸 막는 바닥 (모듈 docstring 참고).
    # 2026-09-02: 15 -> 20 (사용자 지시). `fit_hole_ladder.py` 와 **같은 값**이어야 한다 —
    # 갈리면 같은 손잡이가 다른 모양 위에서 풀린다 (D90 smooth_passes 사고).
    parser.add_argument("--min_sweep_deg", default=20.0, type=float)
    # F9. 이분법이 sweep 대신 반경을 깎게 한다 (§presets.shape_resizer). 위와 같은 이유로
    # `fit_hole_ladder.py` 와 같은 값이어야 한다.
    parser.add_argument("--orbit_fixed_sweep", action="store_true", default=True)
    parser.add_argument("--no_orbit_fixed_sweep", dest="orbit_fixed_sweep", action="store_false")
    # F2. 옛 게이지(`frame0_ray`)로 구운 그래프를 일부러 쓸 때만 (§scene_graph.scale.assert_scale_mode).
    parser.add_argument("--allow_legacy_scale", action="store_true", default=False)
    # 회전 전용 preset 이 사다리 맨 윗단에서 쓸 회전각. 아랫단은 rung 비율로 줄인다.
    parser.add_argument("--pan_deg_at_max", default=60.0, type=float)
    # orbit/arc sweep 상한을 DEFAULT_SHAPE 의 45° 대신 직접 준다. 0 = 기존 동작.
    # true_orbit 엔 거의 안 먹고 arc+dolly 합성 preset 에만 먹는다 (`rung_shape` docstring).
    parser.add_argument("--sweep_deg", default=0.0, type=float)
    # pan 각도를 사다리 매핑 대신 직접 준다. 0 = 기존 동작(사다리).
    parser.add_argument("--pan_deg", default=0.0, type=float)
    # τ 이분법을 끄면 preset 모양이 정의된 각도 그대로 나간다 (scale 1.0). "정확히 45도" 처럼
    # 각도가 요구사항일 때만 쓴다 — τ 는 결과값이 되고 예산을 안 지킨다.
    parser.add_argument("--fit_tau", action="store_true", default=True)
    parser.add_argument("--no_fit_tau", dest="fit_tau", action="store_false")
    # 시작 pose 만으로 예산을 넘긴 변이(전부 static_hold 복제본)를 뱅크에서 뺀다.
    parser.add_argument("--drop_saturated", action="store_true", default=True)
    parser.add_argument("--no_drop_saturated", dest="drop_saturated", action="store_false")

    # ── anchor 가시성 (유일한 하드 컷)
    parser.add_argument("--min_subject_points", default=100, type=int)
    parser.add_argument("--min_subject_area", default=0.005, type=float)

    # ── 실측
    parser.add_argument("--measure_frames", default=5, type=int)
    # G3 가림(`subject_visible_frac`). 렌더가 2배가 되지만 `subject_area_med` 로는 가림이 안
    # 보인다 (`measure_trajectory` docstring). 끄면 열이 빠지고 예전 뱅크와 비트 단위로 같다.
    parser.add_argument("--subject_visible", action="store_true", default=True)
    parser.add_argument("--no_subject_visible", dest="subject_visible", action="store_false")
    # G1(표면 뒤)을 궤적 전 프레임에 걸어 열로 남긴다. 렌더가 안 늘지만 뱅크 스키마가 바뀌므로
    # 기본은 off — 이미 만든 뱅크는 `scripts/audit_bank_geometry.py` 로 사후에 잰다.
    parser.add_argument("--measure_behind", action="store_true", default=False)
    parser.add_argument("--no_measure_behind", dest="measure_behind", action="store_false")
    # F11 (2026-09-02, 7 -> 49). §`fit_hole_ladder.py` 같은 인자 주석. 균등 서브샘플이라 그 사이
    # 프레임에만 있던 표면이 증거에서 빠졌다. G1 은 numpy 투영이라 렌더 수가 안 늘어난다.
    parser.add_argument("--behind_src_frames", default=49, type=int)
    parser.add_argument("--behind_margin_frac", default=0.02, type=float)   # 관통을 봐주는 여유
    parser.add_argument("--behind_clear_frac", default=0.0, type=float)     # 표면 앞에 요구하는 여유
    parser.add_argument("--behind_radius_px", default=0, type=int)          # 패치 최소 depth 반경
    # D123. degenerate 투영 하한 (S 비율). `gates.behind_surface_frames` 주석 참조 — 사다리와
    # **같은 값**이어야 두 단계가 같은 G1 을 본다. 0.0 이 예전 동작.
    parser.add_argument("--behind_min_zcam", default=0.02, type=float)
    # D120. mesh 판 전용 임계. `fit_hole_ladder.py` 의 같은 인자 주석 참조 — 거기는 사다리라
    # `behind_clear_frac 0.10` 이고 여기 τ 뱅크는 0.0 이라, **같은 0.08 을 주면 여기서는 mesh
    # 가 depth 보다 엄해진다**. 그게 맞는 방향이다: τ 뱅크가 통과시킨 변이를 사다리가 다시
    # 거르므로, τ 단계에서 미리 걸러 두면 사다리 렌더가 줄어든다. 예전 뱅크는 0.02.
    parser.add_argument("--mesh_margin_frac", default=0.08, type=float)
    # G1 시간축 정합 (2026-09-02). 켜면 `t != plan_frame` 인 소스 프레임의 동적 픽셀을 증거에서
    # 뺀다. **D116 부터 기본 True** (사용자 지시 "일단 켜줘"). 순수하게 느슨해지는 방향이고
    # (parkour 4/664, snowboard 6/196 변이 변화, 감소 0건) 채널별 예산을 나눌 수 있게 된다.
    # `--no_collision_time_match` 로 옛 경로(합집합 한 열) 그대로 돌릴 수 있다.
    parser.add_argument("--collision_time_match", action="store_true", default=True)
    parser.add_argument("--no_collision_time_match", dest="collision_time_match",
                        action="store_false")
    # D116. G1 의 **증거 소스**.
    #   `depth` — 소스 RGBD shell 재투영 (기본, in-the-wild 는 이것뿐).
    #   `mesh`  — `.blend` 삼각형 점유격자 + EDT (`lbm/mesh_collision.py`). TRUMANS 만 가능.
    #   `both`  — 둘의 **합집합**. TRUMANS 는 이걸 쓸 것.
    # 두 판이 서로 다른 것을 잡는다 (a00 836변이 실측: depth 130 / mesh 357 / 겹침 **0건**).
    # depth 는 소스가 본 표면을 정확히 알고 mesh 는 소스가 못 본 부피를 안다 — `mesh` 단독은
    # depth 가 잡던 130건을 놓친다.
    parser.add_argument("--collision_source", default="depth",
                        choices=["depth", "mesh", "both"])
    # 비면 `<output_root>/<video>/mesh_grid.npz`. `--collision_source mesh` 일 때만 읽는다.
    parser.add_argument("--mesh_grid", default="", type=str)
    parser.add_argument("--near_pct", default=1.0, type=float)              # near_depth 백분위
    # 3D standoff (카메라 중심 ↔ 그 시각 점군 최소거리). 렌더 불필요, 기본은 꺼서 열 구성 유지.
    parser.add_argument("--measure_standoff", action="store_true", default=False)
    parser.add_argument("--no_measure_standoff", dest="measure_standoff", action="store_false")
    # OBB clearance (G5, D49). 역시 렌더 0회. 기본은 꺼서 열 구성 유지.
    parser.add_argument("--measure_obb", action="store_true", default=False)
    parser.add_argument("--no_measure_obb", dest="measure_obb", action="store_false")
    parser.add_argument("--tile_width", default=640, type=int)
    parser.add_argument("--tile_height", default=360, type=int)
    parser.add_argument("--center_box", default=0.80, type=float)

    # ── 추출 / 프리뷰
    parser.add_argument("--num_samples", default=0, type=int)     # 0 = 열거한 전량
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--preview", action="store_true", default=True)
    parser.add_argument("--no_preview", dest="preview", action="store_false")
    parser.add_argument("--preview_max", default=24, type=int)
    parser.add_argument("--preview_columns", default=6, type=int)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
