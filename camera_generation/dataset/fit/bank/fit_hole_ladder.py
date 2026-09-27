"""**preset 마다 다른 크기 상한**을 실측으로 푼다 — τ 사다리를 hole 사다리로 갈아끼운다.

왜: `sample_camera_bank.py` 는 모든 preset 에 같은 τ 를 물렸다. 그런데 실측상 **강도보다 모양이
구멍을 더 좌우한다** — camel `dyn_0`, 같은 τ=0.35 에서 `straight_ease` 0.015 vs
`pedestal_down` 0.651 로 43배 벌어진다 (τ 를 10배 올려도 2~4배다). 같은 τ 로 묶으면 뱅크의
난이도가 preset 마다 제멋대로가 되고, 그게 하류 학습에는 "어떤 preset 은 항상 쉽고 어떤 건 항상
불가능"으로 들어간다.

그래서 축을 바꾼다. **τ 를 고정하고 hole 을 재는 대신, hole 을 고정하고 τ 를 푼다.**

    사다리 눈금        기존: τ ∈ {0.10 … 1.00}      →  지금: hole ∈ {0.10 … 0.50}
    preset 별 상한     없음 (전부 같은 τ)            →  hole 예산을 만족하는 τ (preset·anchor 마다 다름)

이건 D39("hole 은 게이트가 아니라 지표다")를 뒤집는 게 아니다. τ 뱅크는 그대로 남고, 여기서
나오는 건 그 위에 얹는 **선택 층**이다. hole 은 여전히 실측치고, 달라진 건 어느 축을 고정하느냐뿐.

## 크기 손잡이가 preset 마다 다르다

    보통 preset      손잡이 = `target_tau`. `fit_tau` 이분법이 SE(3) 로그로 크기를 맞춘다.
    pan_left/right   손잡이 = `pan_deg`. 이동이 0 이라 τ 가 스케일에 불변이라서 τ 로는 못 민다
                     (DECISIONS.md D40). `shape` 오버라이드로 각도를 직접 준다.
    static_hold*     손잡이가 없다. 크기가 정의상 0 이므로 사다리에서 뺀다.

## 어떻게 푸나

hole(k) 는 k 에 대해 단조다 (실측: 사다리 5단 전부 단조 증가). 그래서 이분법이 먹는다.
bracket 은 기존 `bank.csv` 의 사다리에서 가져온다 — 이미 5점을 재놨으니 처음부터 넓게 훑을
이유가 없다. 사다리 밖이면 `--knob_range` 끝까지 벌리고 `clamped` 로 찍는다.

    hole 예산이 사다리 최저점보다 낮다   → k 를 아래로 밀다 하한에 닿음. `clamped_low`
    사다리 최고점으로도 예산에 못 미친다 → 위로 밀다 상한에 닿음. `unreached` (그 preset 은
                                          이 anchor 에서 그만큼 어려워질 수 없다는 뜻)

## 예산이 여럿이다 — hole 과 충돌 (D46 옵션 1). 기본으로 켜진 건 hole · G1 · OBB 셋

**hole 은 충돌을 못 잡는다.** 카메라가 벽을 통과해도 벽 너머 관측이 그대로 그려져 `valid_mask`
가 멀쩡할 수 있다. 실측이 그렇다 — `corr(hole, behind_frac)` = −0.014(camel) / +0.059(avocado)
로 사실상 무상관이고, camel 에서 G1 위반 변이의 hole 평균(0.280)이 멀쩡한 변이(0.337)보다
**오히려 낮았다**. 그래서 hole 만 보는 이분법은 "예산 안의 최대 크기"를 풀 뿐 그게 통과 가능한
카메라라는 보장이 없다.

**G1 만으로도 부족하다.** G1 은 "표면 **뒤**냐"만 묻는다. 표면 앞 1 cm 는 통과다 — 실측에서
camel `dyn_0 straight_ease` 는 G1 을 전부 통과하고도 코앞까지 들어갔다. 눈에는 그게 충돌로
보인다. **표면 판정은 부피도 모른다 (D49)** — 표면은 물체의 껍질이라 카메라가 물체 **안**을
지나가도 반대쪽 껍질이 뒤에 남아 G1 은 "표면 앞"으로 읽는다. 그래서 두 번째 충돌 예산이
**OBB clearance (G5)** — scene graph 노드 OBB 표면까지의 부호 거리(음수면 박스 **안**, 단위 u)
다. 부피 판정이라 그 구멍을 막고, 덤으로 **무엇과 부딪혔나**를 노드 라벨로 말해준다
(`obb_node` 열). 렌더 0회. 실측: 이 게이트가 없던 뱅크에서 camel 최소 −0.0258 u / 21변이,
avocado −0.0351 u / 22변이가 노드 **안**에 있었다.

**그 마진은 절대값이면 안 된다 (D51).** 절대 마진 `m` 은 모든 박스를 축마다 똑같이 `2m` 부풀린다.
`m=0.15` 에서 camel 낙타(0.137×0.060×0.122)의 얇은 축은 6배가 되고 avocado 의자 조각
(0.056×0.005×0.022)의 얇은 축은 **61배**가 되어, 두 유령 박스가 거의 같은 물리적 크기로
수렴한다 — 물체가 낙타 한 마리든 의자 모서리든 금지구역이 같아진다는 뜻이다. 실제로 절대
마진으로 스윕하면 camel 은 0.15 까지 path 합이 12.84→10.35 로 완만한데 avocado 는
6.18→0.89 로 무너지고, 6변이 중 4개가 `path_len` 0.000 이 된다(전부 `stat_4` chair).

**크기 비례도 시도했다가 실측으로 기각했다.** 눈으로 고른 마진을 지배 노드 `max(extent)` 로
나누면 1.09/1.07 로 2% 안에 겹쳐서 `m_j = ratio·max_ext_j` 를 구현했는데, 돌려보니 camel 은
재현(10.66)했지만 avocado 는 **1.19** 로 절대 0.06(2.25)보다 나빴다. avocado 의 실제 binding
노드가 `stat_0`(테이블 0.376×0.354×0.024)이라 `max_ext` 가 두께가 아니라 **너비**를 집는다.
`min_ext` 로 바꾸면 `stat_0` 은 맞는 대신 `stat_4` 가 12배로 튄다 — 어느 척도든 binding 노드
셋 중 둘만 맞고 그 둘이 척도마다 바뀐다. 2% 일치는 우연이었다.

**그래서 마진은 소스 대비 배수 하나다** — `m = β × (소스 카메라 자신의 최소 OBB 거리)`,
`--obb_clear_src_ratio β`(기본 0.3). β<1 이면 **소스 카메라가 정의상 통과**하므로, 절대 마진이
반복해서 원본 촬영을 기각하던 사고를 구조적으로 막는다. 실측 마진은 camel 0.1532 /
avocado 0.0350 이고, camel 값은 사용자가 영상 보고 고른 0.15 를 path 합 10.35 까지 재현한다.
판정량은 `slack = 거리 − m ≥ 0`, 노드 선택도 거리가 아니라 **slack 최소**다.
소스 카메라 자신의 여유는 camel +0.5107 u(`fence`) / avocado-slice +0.1165 u(`chair`).

**표면 근접(standoff)과 렌더 depth(near_depth)는 판정에서 삭제됐다 (D47/D50 → D57).**
`near_depth`(렌더 depth 하위 백분위)는 D47 이 두 방향으로 틀렸음을 실측했다 — ① 걸리는
픽셀의 세로 위치 median 이 0.98 로 **화면 맨 아래**, 카메라 밑 바닥이지 장애물이 아니고
② 화면 **밖**의 가까운 기하는 못 본다(`dyn_0 truck_left` near_depth 0.687 인데 실제 3D 거리
0.090). 3D `standoff` 는 그 방향 의존을 고쳤지만 OBB 와 같은 걸 두 번 쟀고(OBB 를 침범한
33변이의 standoff 가 33개 전부 임계 아래), 붙여두면 항상 먼저 물어 OBB 가 `binding` 에 안
잡혔다. 게다가 소스 대비 배수라 **"조금 키운다"가 안 되는 축**이다 (ratio 0.80↔0.90 에서
camel `push_in_arc` 가 0.939→0.020 으로 불연속 붕괴). D50 이 끈 뒤 한 번도 안 켰으므로
D57 에서 손잡이 넷(`--min_standoff_ratio` `--min_standoff` `--min_near_depth` `--near_pct`)을
지웠다. **두 양은 계속 측정되어 `standoff`/`near_depth` 열로 뱅크에 남는다** — 사라진 건
판정뿐이다.

**충돌 셋을 전부 통과하는 구멍이 하나 더 있다 — 수직 (G6, D55).** 위 셋은 전부 **수평 근접**을
잰다. 수직 이동은 hole 을 거의 안 늘려서(바닥에도 천장에도 점이 있다) `crane` 계열은 shape 배증이
상한(`--shape_headroom 2 × --shape_doublings 4` = 16배)까지 다 돌아 `lateral_frac 0.35 × 16 = 5.6`
→ `atan(5.6) = 80°` 가 된다. 실측 고도각: `rise_reveal` p95 가 camel **89.83°** / avocado
**85.27°**, `drop_reveal` p05 가 **−83.99°** / **−84.58°** — 물체 **바로 위/아래**다.
`pedestal_up` 도 avocado 에서 77.63° 라 crane 만의 문제가 아니다.

아래쪽은 **지면 관통**까지 간다 (camel 14 변이 / avocado 5). G1 이 이걸 못 잡는 건 임계 문제가
아니라 구조다 — G1 은 카메라 중심을 소스 프레임에 투영해 채점하는데 바닥 밑 카메라는 **화면
밖**으로 투영돼 아예 채점 대상이 안 된다 (해당 변이의 `behind_frac` 이 전부 정확히 0.0000).
**가림 지표로도 커버 안 된다**: 바닥은 scene graph 노드가 아니라 OBB 가리기가 구조적으로 못
보고(avocado `stat_0__drop_reveal__hole0.5` 는 바닥 아래 0.66 u, 고도각 −85.98° 인데
`obb_occl_pass` 0.984), 렌더 가리기는 바닥을 밑에서 보면 **구멍**이 나는데 구멍은 "가려짐"이
아니라 "보임"으로 세어진다.

그래서 세 번째·네 번째 예산이 **고도각 상한**(`elev`)과 **지면 여유 하한**(`ground`)이다.
`lbm/gates.py:elevation_profile` — 렌더 0회, 프레임당 3×3 곱 하나. 임계는 여기서도 소스 대비다:
실효 상한 `max(--max_elev_deg, 소스 자신 + --elev_src_margin_deg)` (avocado `stat_4` chair 는
소스가 이미 35.09° 로 내려다본다), 실효 바닥 `--min_ground_clear_ratio × 소스 카메라 자신의 높이`
(camel 0.023 u / avocado 0.078 u 로 3.4배 벌어져 절대값이 한쪽을 반드시 기각한다).
`--no_elev_gate --no_ground_gate` 면 열만 남고 예전과 비트 동일하게 돈다.

## 앞뒤 이동은 거리가 아니라 **지나침**이 문제다 (G7, D56)

`obb_signed_distance` 는 **부호 없는** 거리다. 그래서 dolly 가 anchor 박스를 **옆으로 스쳐
지나가** 뒤쪽에 서도 거리만 멀면 G5 를 통과한다 — 사용자가 지적한 "앞 뒤로 움직이는 것도
bbox 를 지나치기 전까지만"이 이 구멍이다. 다섯 번째 예산 `approach` 는 축을 하나 고정해서
**부호**를 준다: `a` = normalize(anchor 중심(frame 0) − 플랜 시작 카메라 위치), 프레임마다
`s(f) = (p_g(f) − c_j(f))·a`, 근접면 여유 `gap = −half_a − s`, 뒷면 통과량 `past = s − half_a`.
임계는 여기서도 소스 대비 — `--approach_src_ratio β` × (소스 카메라 자신의 시선축 여유),
G5 와 같은 β=0.3. anchor 노드에만 건다 (축을 anchor 가 정의하므로 다른 노드엔 뜻이 없다).
`--no_approach_gate` 면 열만 남는다.

기본값(`--collision_free`)에서 "너무 크다"의 판정은 여섯이다 — **hole ≥ 예산** /
**G1 위반 프레임 비율 > `--max_behind_frac`** (`collision`) / **OBB slack < 0** (`obb`) /
**|고도각| > 상한** (`elev`) / **지면 여유 < 바닥** (`ground`) /
**시선축 전진 여유 < 임계** (`approach`).
이분법이 푸는 답이 **충돌 없는 최대 크기**로 바뀐다. 어느 쪽이 상한을 정했는지는 `binding` 열과
`collision_limited`/`obb_limited`/`elev_limited`/`ground_limited`/`approach_limited` status 로
남는다. `--no_collision_free` 면 예전처럼 hole 만 보고, `--no_obb_gate` 면 OBB 만 열로 남기고
판정에서 뺀다.

## 사다리의 바닥은 0 이 아니다 — 정지 hole 과 소스 시차 (D53)

고시차 씬에서 뱅크 절반이 **조용히 정지 카메라**가 됐다 (avocado 392 중 `translation_degenerate`
204, pan 계열을 빼도 148). 원인이 둘인데 둘 다 "사다리의 바닥"이다.

**① τ 손잡이 하한이 도달 불가능했다.** τ 는 plan 과 **소스** 의 프레임별 간격이라, 카메라를 시작
pose 에 얼려놔도 소스가 움직인 만큼 τ 가 쌓인다. 그 값 `tau_start` 는 anchor 와 무관한 **씬
상수**다 — 실측 avocado **0.1286** / camel **0.0042** (각 씬 `parallax_ratio` 0.129 / 0.0046 과
일치). `KNOB_RANGE["tau"]` 하한이 0.02 라 avocado 에서는 **하한 자체가 τ 를 못 만족**하고,
`presets.fit_tau` 의 "예산을 이미 다 썼다" 가지로 들어가 궤적이 통째로 0 이 된다.
→ `--tau_floor_src`(기본 켬): τ 하한을 `tau_start + KNOB_RANGE["tau"][0]` 로 올린다. 하한이
**소스 자신의 시차 위에 얹은 여유**가 되어 원래의 0.02 가 다시 의미를 갖는다 (camel 은
0.02→0.0242 로 사실상 그대로, avocado 는 0.02→0.1486 으로 도달 가능해진다). D47/D51/D55 가
쓰는 "임계는 소스 대비"와 같은 꼴이다. 하한에 닿으면 status 에 `tau_floor` 를 붙인다.

**①-b 하한만 고치면 여전히 정지였다.** 하한을 `tau_start` 바로 위로 올리면 그걸 맞추는 배율이
`fit_tau` 이분법의 **첫 눈금**(`max_scale/2**iterations` = 4/256 = 0.0156)보다 작아진다. 이분법은
`lo=0` 에서 시작하므로 모든 mid 가 목표를 넘으면 `lo` 가 **0 에 남고** — 하한을 고쳤는데도 궤적이
0 이다 (실측: avocado `stat_1 pull_out_arc` 목표 τ 0.1486, knob 0.149 인데 `path_len_u 0.000`;
같은 목표에서 `dyn_0` 는 반경이 작아 첫 눈금이 맞아떨어져 0.010 이 나왔다 — 어느 anchor 냐에
따라 갈리는 **해상도** 문제지 물리가 아니다). → `presets.fit_tau(refine_zero=True)`: `lo` 가 0 이면
`[0, hi]` 에서 이분법을 한 번 더 돌려 눈금을 2^8 배 잘게 만든다. `tau_of` 는 렌더가 아니라 numpy
라 비용이 0 이다. 플래그는 decision 의 `trajectory.tau_refine` 에 실려 `emit_bank` 재현까지 간다.

**② 사다리 눈금이 anchor 의 정지 hole 보다 낮았다.** 카메라를 얼려도 hole 은 0 이 아니다 —
시작 pose 가 소스와 다르면 그 자체로 안 본 영역이 생긴다. 실측 avocado `stat_1` **0.5895** /
`stat_4` **0.6384**, camel `stat_3` **0.6482**. 그런 anchor 에서는 0.10/0.20/0.35/0.50 단이
**어떤 손잡이 값으로도** 안 나오고, 이분법은 손잡이를 하한까지 밀다 ①에 걸린다. 궤적을 버려서
얻은 것도 없다 — `stat_1__pull_out_arc` 는 4단 전부 path 0.0000 인데 측정 hole 은 0.5895 로
hole 0.5 단조차 못 맞췄다.
→ `--hole_mode excess`(기본): 단을 **정지 hole 대비 초과분**으로 잡는다. anchor·preset 마다
손잡이 0 으로 한 번 재서(`hole_static`) 목표를 `hole_static + Δ` 로 둔다. Δ 는
`--hole_ladder` 값 그대로다. 부수 효과로 "이 anchor 는 정지만으로 이미 hole 0.59"가 `hstat`
열에 드러난다 — 예전엔 그게 `clamped_low` 로 찍혀 **작지만 정상인 카메라와 구분이 안 됐다**.
`--hole_mode absolute --no_tau_floor_src` 면 예전과 비트 동일하게 돈다.

**③ 실패 판정이 종착점이었다** (D168, 사용자 지시 2026-09-08 "가능한 preset 들 최대한 돌리고
결과적으로 안나오면 …"). `(anchor, preset)` 하나가 `clamped_low` 로 끝나면 그 행이 그대로
뱅크에 남고 끝이었다 — 다른 preset 을 대신 시도하는 경로가 파이프라인 어디에도 없었다
(preset 대체는 τ 단계 `plan_variants` backfill 에만 있고, 그건 fit 이 돌기 전이라 fit 판정을
못 본다). d166 dynpose 21편 105행 중 `clamped_low*` 23 + `static` 5 = **26.7%** 가 그렇게
남았다.
→ `--fallback_ladder`: τ 가 `--variant_pool full` 로 예비 preset 까지 깔아 두면, fit 이 층을
나눠 돌면서 실패한 자리를 다음 층 preset 으로 메운다 (0 본 슬롯 → 1 non-track 조준 →
2 `track_*` → 3 target 포기). **게이트는 한 개도 안 늘렸다** — `solve_knob` 과 사다리 목표는
D166 과 글자 그대로 같다. 실패 판정은 이미 있는 `status` / `suspect`(D140) 두 열을
`--retry_status` / `--retry_suspect` 로 읽을 뿐이다. 이름을 코퍼스 단계의
`--drop_status` / `--drop_suspect` 와 맞춘 이유는 판정 어휘를 두 벌 만들지 않기 위해서다.

출력:
    <out>/<video>/<bank_dir>/{bank.json,bank.csv,poses.npz}   hole 사다리 뱅크
    <out>/<video>/<bank_dir>/tau_caps.json                    preset×anchor -> 크기 상한

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=1 python fit/bank/fit_hole_ladder.py --video camel
    CUDA_VISIBLE_DEVICES=1 python fit/bank/fit_hole_ladder.py --video camel \
        --hole_ladder 0.2 0.4 --anchors dyn_0 --verify_frames 0
    CUDA_VISIBLE_DEVICES=3 python fit/bank/fit_hole_ladder.py --video camel \
        --aim_anchor source_frame0 --bank_dir hole_bank_f0share
"""
import json
import sys
from argparse import ArgumentParser
from math import log
from os import makedirs, path
from time import perf_counter

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import build_poses                                      # noqa: E402
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import subject_point_mask                                        # noqa: E402
from lbm.presets import (DEFAULT_SHAPE, STATIC_PRESETS, SUSPECT_TAGS,           # noqa: E402
                         TAU_REF_CHOICES)
from lbm.render import CloudRenderer, add_cloud_source_args, open_renderer      # noqa: E402,F401
from scene_graph.io import load_scene                                           # noqa: E402
from scene_graph.scale import (TAU_DENOM_MODES, assert_scale_mode,             # noqa: E402
                               tau_denominator)
from scene_graph.schema import load_graph                                       # noqa: E402
from fit.bank.build_candidate_board import subject_track_volume                  # noqa: E402
from lbm.gates import node_margins, obb_gate_nodes                             # noqa: E402
from lbm.mesh_collision import resolve_mesh_grid                               # noqa: E402
from fit.bank.sample_camera_bank import register_external                       # noqa: E402
from fit.bank.sample_camera_bank import (FIT_TAU_MAX_SCALE, ROTATION_ONLY_PRESETS,  # noqa: E402
                                        behind_context, geometry_stats, make_decision,
                                        measure_trajectory,
                                        source_approach, source_elevation, source_g1_clear,
                                        source_obb_clear, source_standoff)

# hole 예산 사다리 기본값. 0.10 은 지금 파이프라인이 실제로 내던 수준, 0.50 은 하류 모델이
# 절반을 지어내야 하는 지점 — 그 위는 "생성"이지 "카메라 이동"이 아니다.
HOLE_LADDER = (0.10, 0.20, 0.35, 0.50)

# `near_depth` 열(렌더 depth 하위 백분위)을 잴 때 쓰는 백분위. 판정에는 안 쓴다 —
# D47 이 이 양을 기각했고 D57 이 손잡이(`--near_pct`)를 지웠다. 열은 진단용으로 남긴다.
NEAR_PCT = 1.0

# 손잡이 탐색 범위. τ 는 사다리(0.10~1.00) 밖으로도 벌려야 예산을 맞출 수 있다
# (`straight_ease` 는 τ=1.0 에서도 hole 이 0.05 대다). pan 은 180° 넘으면 뒤를 본다.
#
# F5. τ 하한 0.02 는 **`--tau_knob_min` 으로 덮어쓸 수 있다**. 기본값이 여기 그대로라 인자를
# 안 주면 비트 동일이다. `main()` 이 이 dict 의 `"tau"` 를 **한 번** 갈아끼우면 `solve_knob`
# 의 탐색 경계 · `lo_override` · `knob_floor` 열 · `bank.json` 의 `knob_range` /
# `tau_floor_rule` 이 전부 같은 값을 읽는다 — 값이 여기 한 곳에만 있어 출처가 갈릴 자리가 없다.
KNOB_RANGE = {"tau": (0.02, 3.0), "pan_deg": (2.0, 180.0)}

# `main()` 이 위 dict 를 갈아끼우므로 `--tau_knob_min` 의 기본값은 **갈리기 전 값**을 여기
# 따로 잡아둔다. 안 그러면 한 프로세스에서 파서를 두 번 만들 때 기본값이 앞 실행의 인자로
# 흘러가서, 같은 명령이 실행 순서에 따라 다른 뱅크를 굽는다.
TAU_KNOB_MIN_DEFAULT = KNOB_RANGE["tau"][0]

# 궤적 모양에만 영향을 주고 사다리 탐색에는 안 들어가는 인자들. **여기가 유일한 출처다** —
# `emit_bank.py` 가 뱅크를 canonical 로 되풀 때 `fixed` 블록에 이 키들이 없는 예전 뱅크는
# 이 값으로 재현한다. 두 곳에 따로 적었다가 `min_sweep_deg` 가 15↔30 으로 어긋나 재현이
# 깨진 적이 있다 (pose 대조 assert 가 잡았다).
SHAPE_DEFAULTS = {"aim_ramp_frames": 12, "orbit_span_frac": 0.8, "min_sweep_deg": 15.0,
                  "traj_basis": "source",
                  # D71. `aim_keyframes: 0` 이 예전 동작(매 프레임 조준)이라, 이 키가 없는
                  # 예전 뱅크를 emit 할 때 0 으로 떨어져야 재현이 맞는다.
                  "aim_keyframes": 0, "keyframe_aim": "auto", "keyframe_ease": "smoothstep",
                  # D90. `smooth_kf` 스케줄만 소비하는 인자라, ease 가 `smoothstep` 이던 동안은
                  # 어긋나도 아무 데도 안 나타났다 (`build_poses.py:679`). 여기 값은
                  # **`build_poses` 의 서명 기본값 12/0.5 여야 한다** — 이 키가 없는 예전 뱅크는
                  # 이 스크립트가 인자를 아예 안 넘겨서 그 기본값으로 구워졌기 때문이다.
                  "smooth_passes": 12, "smooth_lambda": 0.5,
                  # D97. `keyframe_ease` 와 같은 이유로 **`"source"` 여야 한다** — 이 키가 없는
                  # 뱅크는 D97 이전에 구워졌고 그때 τ 는 언제나 소스 기준이었다. 새로 굽는
                  # 기본값(`"auto"`)은 CLI 쪽에 따로 있고, 새 뱅크는 `fixed.tau_ref` 에 값을
                  # 명시적으로 적으므로 이 폴백을 안 탄다.
                  "tau_ref": "source",
                  # D99. `keyframe_ease` 와 같은 두 층 구조. **`False` 여야 한다** — 이 키가 없는
                  # 뱅크는 D99 이전이고 그때는 중력 기준 roll 보정이 아예 없었다. 새로 굽는
                  # 기본값은 CLI 쪽에 따로 있다.
                  "deroll": False,
                  # F9. 같은 두 층 구조. **`False` 여야 한다** — 이 키가 없는 뱅크는 F9 이전이고
                  # 그때 `fit_tau` 는 언제나 SE(3) 로그 전체를 깎았다 (orbit 반경 유지 + sweep 축소).
                  "orbit_fixed_sweep": False,
                  # F7. 같은 두 층 구조. **`False` 여야 한다** — 이 키가 없는 뱅크는 F7 이전이고
                  # 그때는 절단이 아예 없었다. 되만들 때 실제로 읽는 건 행의 `hold_from` 이고
                  # (이 키는 "이 뱅크가 절단을 켜고 구워졌나"의 표식), 그 열이 없거나 비어 있으면
                  # `emit_bank` 는 절단을 안 한다 — 그래서 옛 뱅크는 비트 단위로 그대로다.
                  "time_truncate": False,
                  # D171. follow 위치 채널 keyframe 보간. **`0` / `"cubic"` 이어야 한다** — 이 키가
                  # 없는 뱅크는 D171 이전이고 그때 위치 채널은 언제나 savgol 저역통과였다
                  # (`0` = 끔). `interp` 는 켰을 때만 pose 를 바꾸므로 폴백 값은 아무거나 상관없지만
                  # `build_poses` 서명 기본값과 같은 `"cubic"` 으로 맞춰 둔다.
                  "follow_keyframes": 0, "follow_kf_interp": "cubic",
                  # D150. τ 의 분모. **`"z_med_frame0"` 이어야 한다** — 이 키가 없는 뱅크는
                  # D150 이전이고 그때 분모는 언제나 frame0 z-depth 중앙값이었다. 새 정의(`"S"`)
                  # 는 CLI `--tau_denom S` 로 명시적으로 켠다.
                  "tau_denom": "z_med_frame0"}


# D275 (사용자 지시 2026-09-24 "static을 포함한 anchor가 의미가 없는 free-moving은 shot scale
# 적용을 안하는게 좋을 것 같은데"). targetless = 캡션이 대상을 안 부르는 preset — 조준도 추종도
# 안 하므로 "subject 가 얼마나 커지나" 가 크기 기준이 될 수 없다. 정의는 캡션 설정의 `targetless`
# 플래그 한 곳에서 읽는다 (build_bank_captions 가 target 절을 빼는 판정과 같은 출처).
def _targetless_presets():
    cfg_path = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),
                         "configs", "caption_presets.json")
    with open(cfg_path, encoding="utf-8") as file:
        presets = json.load(file)["presets"]
    return frozenset(k for k, v in presets.items() if v.get("targetless"))


TARGETLESS_PRESETS = _targetless_presets()


def knob_kind(preset: str):
    """이 preset 의 크기 손잡이가 무엇인가. `None` 이면 손잡이가 없다 (정지)."""
    if preset in STATIC_PRESETS:
        return None
    return "pan_deg" if preset in ROTATION_ONLY_PRESETS else "tau"


def write_start_screen(folder: str, rows: list):
    """`--start_screen` 결과 → `<bank_dir>/start_screen.csv` (시작 pose 마다 한 행: ok/reason + 게이트 측정값)."""
    if not rows:
        return
    makedirs(folder, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path.join(folder, "start_screen.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"[start_screen] {sum(r['ok'] for r in rows)}/{len(rows)} 시작 pose 통과 → {folder}/start_screen.csv")


def start_cands_for(tau_bank: dict, anchor: str, preset: str) -> list:
    """이 (anchor, preset) 의 시작 pose 후보 목록 (D259). 격자 없는 뱅크는 `[None]` 하나다.

    후보를 **다시 유도하지 않고 τ 뱅크 행에서 읽는다.** 격자 정의(`build_pool_front`)가
    바뀌면 같은 `cand_id` 가 다른 pose 를 뜻하게 되는데, 재유도하면 그게 조용히 통과한다.
    행이 후보 전문을 들고 있으므로 여기서는 중복만 제거하면 된다.
    """
    out, ids = [], set()
    for row in tau_bank["variants"]:
        if row["anchor_id"] != anchor or row["preset"] != preset:
            continue
        cand = row.get("start_cand")
        key = (cand or {}).get("cand_id")
        if key not in ids:
            ids.add(key)
            out.append(cand)
    return out or [None]


def decision_at(graph, node, preset: str, knob: float, kind: str, args, mult: float = 1.0,
                start_cand: dict | None = None):
    """손잡이 값 하나 → `lbm_decision_v1`. τ 손잡이면 target_tau, 각도 손잡이면 shape 로 들어간다.

    `mult` 는 기본 모양(`DEFAULT_SHAPE`)의 배율이다. 왜 필요한가: τ 를 아무리 올려도
    `fit_tau` 는 `max_scale`(4.0) 이상 못 키우므로 **모양 자체가 천장**이 된다 — camel `dyn_0`
    `straight_ease` 는 dolly = 0.35·radius 라 4배 해도 τ 1.15 가 끝이었다. 모양을 키워서 그
    천장을 올린다 (`--shape_headroom` 배씩 최대 `--shape_doublings` 번).
    """
    if kind == "pan_deg":
        # 회전 전용은 `fit_tau` 가 항상 max_scale 로 되곱하므로 미리 나눠 준다 (D40).
        # target_tau 는 그 가지를 타게만 하면 되므로 탐색 상한을 그대로 쓴다.
        shape = {"pan_deg": knob / FIT_TAU_MAX_SCALE}
        target_tau = KNOB_RANGE["tau"][1]
    else:
        shape = None if mult == 1.0 else {"dolly_frac": DEFAULT_SHAPE["dolly_frac"] * mult,
                                          "lateral_frac": DEFAULT_SHAPE["lateral_frac"] * mult}
        target_tau = knob
    return make_decision(graph, node, preset, target_tau, args.speed, args.tracking,
                         args.look_at_bias, args.start_mode, shape=shape,
                         tau_refine=bool(getattr(args, "tau_refine", False)),
                         # D97. `getattr` 인 이유는 `emit_bank` 가 `SimpleNamespace` 를 만들어
                         # 넘기기 때문이다 — 옛 뱅크에는 이 키가 없고, 그때는 "source" 가 맞다.
                         tau_ref=str(getattr(args, "tau_ref", "source")),
                         # D259. `None` 이면 `make_decision` 이 예전대로 소스 frame0 구도를
                         # 쓴다. 후보가 있으면 decision 의 `start_mode` 가 "board" 가 되는데,
                         # `build_poses` 는 그걸 **인자로** 받으므로 호출부도 같이 바꿔야 한다.
                         start_cand=start_cand)


def behind_over(stats: dict, max_behind: float, max_behind_dyn: float) -> bool:
    """G1 판정. `--collision_time_match` 면 **정적·동적을 각자의 예산과 따로** 비교한 뒤 OR.

    사용자 지시 2026-09-02: "collision 을 static 도 하고 dynamic 은 따로 해서 양쪽 다 판정".
    두 예산이 같으면 합집합 `behind_frac` 과 수학적으로 동일하다 (프레임 하나가 어느 채널에서든
    걸리면 양쪽 식이 같이 넘는다). 다르게 주면 그때부터 갈라진다 — 예컨대 벽은 한 프레임도
    허용 안 하되 지나가는 사람은 조금 봐주는 식.

    `time_match` 가 꺼져 있으면 `behind_static_frac` 이 없으므로 예전과 같이 `behind_frac` 하나만 본다.
    """
    if max_behind < 0.0:                     # `--no_collision_free`: G1 자체를 안 본다
        return False
    if "behind_static_frac" not in stats:
        return stats.get("behind_frac", 0.0) > max_behind
    return (stats.get("behind_static_frac", 0.0) > max_behind
            or stats.get("behind_dyn_frac", 0.0) > max_behind_dyn)


# D277 (R37). fitting **중** GT mesh raycast 게이트. `--ray_gate server` 면 main 이 임계를 채운다.
# None 이면 꺼짐 — 아래 두 판정 함수가 아무것도 안 해서 옛 동작과 비트 동일하다.
# 판정 정의·임계는 `bank_to_blender_poses.py --raycast` 사후 판정과 같다 (서버가 Blender 판정과
# 188씬 전량 일치 — `eval/compare_mesh_gates.py --backend server`).
RAY_GATE = None
RAY_CLIENT = None


def ray_verdict(stats: dict):
    """raycast 열(`ray_*`)로 판정. 걸린 사유 또는 None. 열이 없으면(게이트 꺼짐) None."""
    if RAY_GATE is None or "ray_min_clearance" not in stats:
        return None
    if stats["ray_min_clearance"] < RAY_GATE["min_clearance"]:
        return "wall"
    if stats["ray_min_floor_drop"] < RAY_GATE["min_floor_drop"]:
        return "floor"
    if stats["ray_min_subject_dist"] < RAY_GATE["min_subject_dist"]:
        return "subject"
    if stats["ray_clear_frac"] < RAY_GATE["min_clear_frac"]:
        return "occluded"
    return None


def physical_verdict(stats: dict, max_behind: float, min_obb: float, max_elev: float,
                     min_ground: float, min_approach: float,
                     max_behind_dyn: float | None = None):
    """렌더 **없이** 나오는 게이트만 판정. 걸렸으면 사유 문자열, 아니면 `None`. (D113)

    `over()` 의 if/elif 체인 **앞 다섯 가지와 글자 그대로 같은 식·같은 순서**다. 두 곳에 적힌
    이유는 하나는 이분법의 최종 판정(hole·가림까지 본다)이고 하나는 렌더 **전** 예선이기
    때문이다 — 그래서 여기서 걸린 사유는 반드시 `over()` 에서도 같은 사유로 걸린다.
    식이 바뀌면 **두 곳을 같이** 고쳐야 한다.

    사용자 지시(D113): "앞에서 물리 판정이 실패하면 사실 렌더할 필요가 없잖아". 여기서 걸리면
    `measure_trajectory` 가 렌더를 통째로 건너뛴다. 판정 결과는 안 바뀐다 (`over()` 는 hole 을
    이 다섯 뒤에서 보므로 애초에 안 읽던 값이다).
    """
    obb = stats.get("obb_slack", float("nan"))
    ground = stats.get("ground_clear", float("nan"))
    elev = stats.get("elev_abs_max", float("nan"))
    approach = stats.get("approach_gap", float("nan"))
    if behind_over(stats, max_behind,
                   max_behind if max_behind_dyn is None else max_behind_dyn):
        return "collision"
    if obb == obb and obb < min_obb:
        return "obb"
    if ground == ground and ground < min_ground:
        return "ground"
    if elev == elev and elev > max_elev:
        return "elev"
    if approach == approach and approach < min_approach:
        return "approach"
    return ray_verdict(stats)


def truncate_hold(poses: np.ndarray, hold_from: int) -> np.ndarray:
    """`hold_from` 프레임부터 **위치만** 얼린다. 회전(조준)은 그대로 둔다. (F7)

    왜 위치만인가: G1/G5/G6/G7 **넷 다 카메라 위치만 본다** (`geometry_stats` — 재투영도
    OBB 부호거리도 고도도 시선축 전진량도 전부 `poses[:, :3, 3]`). 회전은 어느 게이트도 안
    보므로 얼릴 이유가 없고, 얼리면 동적 subject 를 놓친다. 남는 그림은 "밀고 들어가다
    멈춰서 계속 따라본다"로, 정지 궤적으로 눌리는 것보다 훨씬 낫다.

    `hold_from >= len(poses)` 면 절단이 없다 — 그때 이 함수는 **입력을 그대로** 돌려주므로
    (사본조차 안 만든다) `--no_time_truncate` 경로와 비트 단위로 같다.
    """
    if hold_from >= len(poses):
        return poses
    out = np.array(poses, dtype=float, copy=True)
    out[hold_from:, :3, 3] = out[max(hold_from - 1, 0), :3, 3]
    return out


def retime_info(extra: dict, poses: np.ndarray, src_c2w: np.ndarray, z_med: float):
    """절단된 궤적에 맞춰 **위치에서 나오는 두 열**만 다시 잰다 (제자리 수정). (F7)

    왜 필요한가: `build_poses` 가 낸 `info` 는 **절단 전** 궤적의 값이다. 그중 `path_len_u` 와
    `tau_max_final` 은 카메라 위치에서 직접 나오므로 절단하면 틀린다 — 특히 `path_len_u` 는
    `emit_bank --min_path_len` 이 읽는 필터라, 안 고치면 절단으로 사실상 정지가 된 궤적이
    "길이 0.4 u" 를 달고 통과한다. 둘 다 `build_poses` 와 **글자 그대로 같은 식**을 쓴다
    (`decode/build_poses.py:889, :924`).

    안 고치는 것들: `radius_u`(시작 pose 에서 나온다) · `view_angle_max_deg` / `roll_*`
    (회전만 보는데 절단은 회전을 안 건드린다) · `tau_max`(= `tau_ref_max`, **이분법이 맞춘**
    값이라 절단 전 궤적의 성질이다) · `shape_context`.
    """
    info = extra["info"]
    tau = np.linalg.norm(poses[:, :3, 3] - src_c2w[:len(poses), :3, 3], axis=-1) / max(z_med, 1e-9)
    extra["tau_per_frame"] = tau
    info["tau"] = {**info["tau"], "tau_max_final": round(float(tau.max()), 4)}
    info["path_len_u"] = round(float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                                    axis=-1).sum() / info["S"]), 5)
    return extra


def solve_hold_from(renderer, poses: np.ndarray, behind: dict, verdict, min_hold: int) -> int:
    """물리 게이트를 통과하는 **가장 늦은** hold 시작 프레임. 통과 못 하면 `None`. 렌더 0회. (F7)

    사용자 지시(2026-09-04, fix.log F7): "게이트에 걸리는 프레임 앞까지만 움직이고 그 뒤는
    hold". 지금 `solve_knob` 은 **크기**만 줄이므로 마지막 1프레임이 벽에 닿으면 전 구간이
    같이 눌려 정지 궤적이 된다 — 48프레임이 멀쩡한데 49번째 때문에 전부 버리는 셈이다.

    단조성: `hold_from` 을 줄이면 카메라가 **덜** 움직이므로 게이트가 느슨해진다. 엄밀히
    단조는 아니지만(움직이다 멈춘 자리가 다른 프레임의 동적 표면과 겹칠 수 있다) 이분법이
    푸는 답은 어차피 마지막에 `verdict` 로 다시 확인되므로 틀린 답을 통과시키지 않는다 —
    최악의 경우 최적보다 이른 hold 를 고른다.

    비용: 게이트 넷 다 렌더가 0회다 (재투영 + 3×3 곱). 49프레임이면 이분법 6회 = 렌더 0장.
    """
    n = len(poses)
    if verdict(geometry_stats(renderer, poses, behind)) is None:
        return n                             # 절단이 필요 없다
    if verdict(geometry_stats(renderer, truncate_hold(poses, min_hold), behind)) is not None:
        return None                          # 최소 이동조차 못 한다 → 기존 축소로 떨어진다
    lo, hi = min_hold, n                     # lo 는 통과, hi 는 실패
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if verdict(geometry_stats(renderer, truncate_hold(poses, mid), behind)) is None:
            lo = mid
        else:
            hi = mid
    return lo


def tag_suspects(rows: list, args) -> dict:
    """D140. **게이트가 막은 게 아닌데** 수치가 나쁜 행에 `suspect` 태그를 단다.

    사용자 지시 2026-09-06: "충돌, subject in frame 같은 수치가 의도와 다르게 preset 과 첫
    카메라가 놓인 상황을 봤을 때 불가능한 상황이 아닌데도 안좋게 나오면 일단 clamped_low 처럼
    인식할 수 있게끔 해두고".

    `clamped_low` 와 **성격이 다르다**. `clamped_low` 는 이분법이 하한에서도 게이트를 못 넘긴
    것 = 판정 결과다. 여기 태그는 판정을 통과했는데도(대개 `status == "solved"`) 결과 기하가
    캡션과 어긋나는 것 = **설정 탓**이다. 그래서 `status` 를 덮어쓰지 않고 **별도 열**로 둔다 —
    덮어쓰면 "게이트 어디서 걸렸나"라는 원래 정보가 사라지고, `binding` 과도 어긋난다.
    소비처(`vista4d_bank_to_dl3dv.py --drop_status`)가 자를지 말지를 정한다.

    ## 태그 3종과 그 근거 (2026-09-06 실측, export 대상 = clamped_low* 제외)

    ① `aim_free_subject_lost` — `aim == "free"` 인데 `subject_in_frame` 이 낮다.
       "불가능하지 않다"의 근거는 **같은 뱅크 안의 대조군**이다. aim 만 다른 행들이 있는데,

           TRUMANS d132  aim=look_at n=3241  p10=1.000  <0.85 =  0.0%
                         aim=free    n=2681  p10=0.462  <0.85 = 36.2%
           VISTA   d128  aim=look_at n=9007  p10=0.923  <0.85 =  4.5%
                         aim=free    n=7924  p10=0.077  <0.85 = 54.1%

       즉 씬이 어려워서가 아니라 **겨냥을 안 해서** 나가는 것이다. TRUMANS 970행 중 572행이
       `status == "solved"` (게이트 전부 통과), 그중 213행이 `track_*` preset 이라 캡션은
       "follows the target" 인데 피사체는 화면 밖이다 (goals.md 중기1 / task #134).

    ② `static_start_collision` — 이동량이 0인데 `behind_frac > 0`.
       `track_hold`/`static_hold` 계열은 손잡이가 inert 라 이분법이 돌릴 게 없다. 충돌은
       **시작 pose 자체**에서 오고 사다리는 그걸 못 고친다. TRUMANS 24행 전부 `knob` 이 하한
       0.1000 에 `status == "static"`, `behind_static_frac == behind_frac` (동적 0.000) —
       순수 벽. `tru_00add26c_a18` static_hold 는 `behind_frac == 1.000`, 49프레임 내내 벽 속인데
       정상 변이로 나간다. 고치는 법은 시작 pose 재배치라 fit 단계 밖이다.

    ③ `motion_preset_no_motion` — 이동 preset 인데 `path_len_u` 가 0에 가깝다. **게이트가
       하나도 안 걸린 행만** 잡는다 (`binding` 이 hole/none/빈칸). collision_limited 등은
       진짜로 막힌 것이라 정직한 결과이므로 뺀다. 남는 건 "사다리 아랫단이 너무 쉽게
       충족돼서 안 움직였다" = 캡션은 dolly_in 인데 카메라가 정지. VISTA 242행 중 79행이
       `solved`. `clamped_low` 필터가 잡는 것과 같은 종류의 해악인데 status 가 달라서 샌다.

    ④ `aim_target_subject_lost` — ① 의 **여집합**이다. `aim != "free"` 인데 `subject_in_frame`
       이 낮다 = 조준하겠다고 해 놓고 피사체를 놓쳤다. ① 과 임계(`--suspect_in_frame`)를
       공유한다 — 같은 것을 재는데 aim 만 다르므로 눈금을 따로 둘 이유가 없다.
       ① 을 만들 때 대조군으로 쓴 그 표가 이쪽 근거이기도 하다: `aim=look_at` 의 `<0.85` 비율이
       TRUMANS d132 0.0% / VISTA d128 **4.5%** 로 작지만 0 이 아니다. dynpose d166 21편에서는
       105행 중 **10행** 이고, 그중 7행은 `status` 도 `suspect` 도 안 붙어 조용히 통과했다
       (`pull_out_arc_right` solved, `crane_up` elev_limited …). ① 과 달리 이건 preset 선택
       실수가 아니라 **그 anchor 에서 그 궤적이 안 되는 것**이라, D168 의 retry 가 다른 preset
       으로 갈아탈 근거가 된다 (사용자 지시 2026-09-08 "target 이 있는 경우에만 sif").

    ⑤ `hole_over_budget` — `hole_fraction > --suspect_hole`. `hole_mode excess` 의 목표는
       `hole_static + Δ` 인데 `hole_static` 자체에 상한이 없다. d166 실측으로 `hole_static`
       p100 = 0.474 → `target_hole` p100 = **0.674** 이고, 그 예산이면 이분법이 `tau_max`
       **2.246** 짜리 궤적도 "예산 안"으로 통과시킨다. `status` 는 `solved` 로 찍힌다 —
       이분법 입장에서는 실제로 목표를 맞췄기 때문이다. 105행 중 `hole_fraction > 0.35` 가
       14행인데 기존 어휘로 잡히는 건 1행뿐이었다 (9행이 `solved`).
       기본값 0.35 는 d166 config `_rung` 이 이미 합격 기준으로 쓰던 그 숫자다
       (`hole<=0.35 & sif>=0.85 & behind=0`) — 새로 정한 눈금이 아니다.

    pan/tilt 는 제자리 회전이라 `path_len_u == 0` 이 정상 — ③ 에서 뺀다. hold/look_at 계열도
    정지가 의도이므로 뺀다 (그쪽은 ② 가 본다).

    돌려주는 값은 태그별 개수 dict (요약표용). `rows` 는 제자리에서 고친다.
    """
    hold_like = ("hold", "look_at")          # 정지가 의도인 preset 접미사
    turn_like = ("pan", "tilt")              # 제자리 회전 — path_len_u 0 이 정상
    # 목록은 `lbm.presets.SUSPECT_TAGS` 한 벌만 둔다 — 자르는 쪽(`--drop_suspect`)이 같은 걸
    # 읽어야 태그를 늘렸을 때 조용히 어긋나지 않는다.
    counts = {tag: 0 for tag in SUSPECT_TAGS}
    for row in rows:
        tags = []
        preset, status = str(row["preset"]), str(row["status"])
        in_frame = row.get("subject_in_frame")
        behind = row.get("behind_frac")
        path_len = row.get("path_len_u")
        hole = row.get("hole_fraction")
        # ① 조준을 안 해서 피사체가 나간 경우.
        if (str(row.get("aim")) == "free" and isinstance(in_frame, (int, float))
                and in_frame < args.suspect_in_frame):
            tags.append("aim_free_subject_lost")
        # ② 손잡이가 inert 인데 시작 pose 가 이미 벽 안.
        if (isinstance(behind, (int, float)) and behind > args.suspect_behind
                and (status == "static"
                     or (isinstance(path_len, (int, float))
                         and path_len < args.suspect_path_len))):
            tags.append("static_start_collision")
        # ③ 이동 preset 인데 안 움직였다 — 단, 게이트가 안 막은 경우만.
        if (isinstance(path_len, (int, float)) and path_len < args.suspect_path_len
                and not preset.endswith(hold_like) and not any(t in preset for t in turn_like)
                and str(row.get("binding")) in ("hole", "none", "")):
            tags.append("motion_preset_no_motion")
        # ④ 조준하는데도 피사체가 나간 경우 (① 의 여집합, 임계 공유).
        if (str(row.get("aim")) != "free" and isinstance(in_frame, (int, float))
                and in_frame == in_frame and in_frame < args.suspect_in_frame):
            tags.append("aim_target_subject_lost")
        # ⑤ 사다리 목표가 부풀어 예산을 넘긴 채로 "풀린" 경우.
        if (args.suspect_hole > 0 and isinstance(hole, (int, float))
                and hole == hole and hole > args.suspect_hole):
            tags.append("hole_over_budget")
        for tag in tags:
            counts[tag] += 1
        row["suspect"] = "|".join(tags)
    return counts


def bracket_from_bank(rows: list, kind: str):
    """기존 τ 사다리에서 (손잡이, hole) 점들을 뽑아 정렬. 이분법 bracket 의 출발점이다."""
    key = "pan_deg" if kind == "pan_deg" else "target_tau"
    points = sorted((float(r[key]), float(r["hole_fraction"])) for r in rows)
    # 같은 손잡이 값이 여러 개면(anchor 가 같으므로 없어야 정상) 평균으로 접는다.
    folded = {}
    for knob, hole in points:
        folded.setdefault(knob, []).append(hole)
    return sorted((k, float(np.mean(v))) for k, v in folded.items())


def solve_knob(probe, target_hole: float, points: list, kind: str, iterations: int,
               max_behind: float, min_obb: float,
               max_elev: float = float("inf"), min_ground: float = float("-inf"),
               lo_override: float | None = None, min_approach: float = float("-inf"),
               min_seen: float = float("-inf"), max_behind_dyn: float | None = None,
               metric_fn=None, min_in_frame: float = float("-inf")):
    """"예산 안의 최대 크기"를 이분법으로 푼다. (k, status, binding, 실측 횟수).

    `max_behind < 0` 이면 예산은 hole 하나다 (기존 동작 그대로). `>= 0` 이면 **충돌 예산 두 개**가
    붙어, 푸는 답이 "충돌 없는 최대 크기"로 바뀐다 — hole 은 충돌을 못 잡기 때문이다
    (D46: `corr(hole, behind_frac)` = −0.014 / +0.059, 사실상 무상관):

      `collision` — G1 위반 프레임 비율 > `max_behind`. 카메라가 관측된 표면 **뒤**.
                    `--collision_time_match` 면 정적/동적 두 채널을 **각자의 예산과 따로** 보고
                    OR 한다 (`behind_over`, `max_behind_dyn`).
      `obb`       — 노드 OBB 부호 거리 < `min_obb` u. G1 은 **표면**만 봐서 물체 내부를
                    통과하는 경우를 놓친다 (D49). 이건 부피 판정이라 그걸 막는다. 끄면
                    `-inf` 가 들어와 조건이 영원히 거짓이다 — 임계 0 이 "박스 표면까지 허용"
                    이라는 **켠 상태**여서, 끄는 값을 0 으로 두면 안 된다.

    (`clearance` — standoff / near_depth 판정 — 은 **삭제됐다**, D57. D50 이 껐고 그 뒤
     한 번도 안 켰다. `standoff`/`near_depth` 열은 계속 측정된다.)

    여기에 **G6 예산 둘**이 더 붙는다 (2026-08-22, 사용자 지적). 위 셋은 전부 "카메라가 무엇에
    부딪히나"만 보므로, 아무것과도 안 부딪히면서 물체 **바로 위/아래**로 올라가는 건 못 막는다:

      `elev`      — |subject 대비 고도각| > `max_elev` deg. 수직 이동은 hole 을 잘 안 늘려서
                    shape doubling 이 끝까지 배가된다 — `rise_reveal` 이 camel 89.96° /
                    avocado 87.13° 로 hole 0.32 · occlusion 0.95 를 통과한다.
      `ground`    — 지면 위 여유 < `min_ground` u. `drop_reveal`/`pedestal_down` 이 바닥을
                    뚫는데 (camel 14 변이, avocado 5) `behind_frac` 이 **전부 0.0000** 이다 —
                    바닥 밑 카메라는 소스 화면 밖으로 투영돼 G1 채점 자체가 안 된다. 바닥은
                    노드가 아니라 G5 도, 가림 지표도 못 본다 (`gates.elevation_profile` 참조).

    둘 다 끄는 값은 각각 `inf` / `-inf` 다 (`obb` 와 같은 이유 — 0 은 켠 상태다).

    그리고 **G7 예산 하나**가 더 붙는다 (2026-08-22, 사용자 지적 "앞 뒤로 움직이는 것도 bbox 를
    지나치기 전까지"):

      `approach`  — 시선축 위 전진 여유 < `min_approach` u. `obb` 는 **부호 없는 거리**라 박스
                    옆을 안전거리로 스쳐 지나가는 궤적을 통과시킨다 — 실측에서 camel
                    `dyn_0 push_in_arc` 가 근접면을 0.1972 u 뚫고 뒷면까지 0.0467 u 나갔는데
                    `obb_clear` 는 0.1536 이상이었다 (`gates.approach_profile` 참조).
                    끄는 값은 `-inf` 다.

    `binding` 은 답 바로 위에서 무엇이 막았나. bracket 은 τ 뱅크의 hole 사다리에서 오므로
    **충돌은 모른다**; 그래서 아래끝이 충돌이면 하한까지 다시 벌리고 그 아래끝을 새 위끝으로 쓴다.

    그리고 **G3 예산 하나**가 마지막에 붙는다 (D112, 사용자 지시 "가려지지 않고 보이는 point 들을
    실제 subject point 들로 나누면 … hole 뒤에 추가로 판정 기준"):

      `occlusion` — subject 가시 비율(`subject_visible_frac`) < `min_seen`. **hole 판정 뒤**에
                    둔다 — 순서를 앞으로 당기면 기존 뱅크의 `binding` 귀속이 바뀐다. 끄는 값은
                    `-inf` 다 (0 은 "전부 가려도 통과"라 사실상 꺼짐이지만, `nan` 방어를 위해
                    음수 하나로 통일한다). 켜면 이분법이 **렌더를 2배**로 쓴다 — subject 만 그린
                    실루엣이 분모라 두 번 그려야 한다 (`measure_trajectory` docstring).

    D266. `metric_fn` 을 주면 **사다리 예산의 양 자체가 바뀐다** — `hole >= target_hole` 자리에
    `metric_fn(stats) >= target_hole` 이 들어간다 (`binding` 이름은 `hole` 그대로 둔다: 하류
    `emit_bank`/`route_presets`/`probe_tau_divisor` 가 이 토큰으로 "예산이 상한을 정했다"를
    읽는다). 안 주면 `None` 이라 예전 식이 그대로 돌아 비트 단위로 같다. 호출자가 shot scale
    (`|log(subject_area_med / area_static)|`) 을 넣는 데 쓴다 — hole 을 안 보는 뱅크용이다.

    `min_in_frame` 은 subject 중심이 중앙 박스 안에 있던 프레임 비율(`subject_in_frame`)의
    하한이다. **가림이 아니라 프레이밍**이다 — 가림은 raycast 로 옮겼고(D266), 이건 손잡이를
    키우다 subject 가 화면 밖으로 나가는 것만 막는다 (d207 실측 `subject_area_end` p10 =
    0.0000 = 끝 프레임에 subject 가 아예 없다). 끄는 값은 `-inf` 다.

    `lo_override` 는 손잡이 하한을 올린다 (D53 ①: τ 는 `tau_start` 아래로는 도달 자체가 안 되고,
    거기로 밀면 `fit_tau` 가 궤적을 0 으로 만든다). 하한을 올리면 τ 뱅크에서 온 bracket 점 중
    그 아래 것들은 **재현 불가능**하므로 같이 버린다 — 안 버리면 이분법이 도달 못 하는 구간을
    탐색해 답이 하한으로 내려앉는다.
    """
    lo_bound, hi_bound = KNOB_RANGE[kind]
    if max_behind_dyn is None:               # 안 주면 정적과 같은 예산 = 합집합과 동일
        max_behind_dyn = max_behind
    if lo_override is not None:
        lo_bound = float(lo_override)
        points = [(k, h) for k, h in points if k >= lo_bound]
    verdicts, calls = {}, 0

    def over(knob):
        """이 크기가 예산을 넘겼나. (bool, 넘긴 이유)."""
        nonlocal calls
        if knob not in verdicts:
            # `near`/`standoff` 는 열로만 남는다 (판정 삭제, D57) — 여기선 자리만 받는다.
            (hole, geo, _near, _standoff, obb, elev, ground, approach,
             seen) = probe(knob)
            calls += 1
            # `physical_verdict` 와 **같은 함수**를 쓴다 — 예전엔 여기만 스칼라 비교라
            # `--gate_before_render` 의 렌더 생략 판정과 최종 판정이 갈릴 수 있었다.
            if behind_over(geo, max_behind, max_behind_dyn):
                verdicts[knob] = (True, "collision")
            elif obb == obb and obb < min_obb:
                verdicts[knob] = (True, "obb")
            elif ground == ground and ground < min_ground:
                verdicts[knob] = (True, "ground")
            elif elev == elev and elev > max_elev:
                verdicts[knob] = (True, "elev")
            # G7 은 기존 게이트들 **뒤**에 둔다 — 순서를 앞으로 당기면 여러 게이트가 동시에
            # 물릴 때 `binding` 귀속이 예전 뱅크와 달라진다.
            elif approach == approach and approach < min_approach:
                verdicts[knob] = (True, "approach")
            # D277. GT mesh raycast (벽/바닥/피사체 거리/시선). approach 뒤, 프레이밍 앞 —
            # 물리 게이트 무리 끝에 둔다. 꺼져 있으면 `ray_verdict` 가 None 이라 no-op.
            elif ray_verdict(geo) is not None:
                verdicts[knob] = (True, ray_verdict(geo))
            # D266. 프레이밍은 예산(hole/shot) **앞**에 둔다 — subject 가 화면을 벗어난 크기는
            # shot scale 이 얼마든 못 쓰는 카메라다. 끄면 `-inf` 라 조건이 영원히 거짓이다.
            elif (lambda f: f == f and f < min_in_frame)(
                    geo.get("subject_in_frame", float("nan"))):
                verdicts[knob] = (True, "framing")
            elif (metric_fn(geo) if metric_fn is not None else hole) >= target_hole:
                verdicts[knob] = (True, "hole")
            # D112. hole **뒤**. 여기 오면 hole 은 이미 예산 안이므로, 기존 뱅크에서 `(False,
            # "hole")` 이던 자리만 갈라진다 — `min_seen = -inf` 면 조건이 영원히 거짓이라
            # 비트 단위로 같다.
            elif seen == seen and seen < min_seen:
                verdicts[knob] = (True, "occlusion")
            else:
                verdicts[knob] = (False, "hole")
        return verdicts[knob]

    # bracket: 사다리에서 target 을 사이에 끼는 두 점. 없으면 범위 끝까지 벌린다.
    lo = next((k for k, h in reversed(points) if h <= target_hole), None)
    hi = next((k for k, h in points if h >= target_hole), None)
    binding = "hole"
    if lo is None:                       # 제일 작은 크기에서도 이미 예산 초과
        lo = lo_bound
        is_over, binding = over(lo)
        if is_over:
            return lo, "clamped_low", binding, calls
    elif (max_behind >= 0.0 or max_elev < float("inf") or min_ground > float("-inf")
          or min_seen > float("-inf") or min_in_frame > float("-inf")):
        is_over, why = over(lo)
        if is_over:                      # 사다리는 hole 만 봤다 — 충돌/G6 기준으로는 여기도 크다
            hi, binding = lo, why
            lo = lo_bound
            is_over, why = over(lo)
            if is_over:
                return lo, "clamped_low", why, calls
    if hi is None:
        hi = hi_bound
        is_over, binding = over(hi)
        if not is_over:                  # 제일 큰 크기로도 예산에 못 미친다
            return hi, "unreached", "none", calls
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        is_over, why = over(mid)
        if is_over:
            hi, binding = mid, why
        else:
            lo = mid
    # 마지막으로 통과한 쪽(lo)을 답으로 쓴다 — 예산을 넘기지 않는 쪽이다. 호출자가 캐시에서
    # 꺼내 쓰므로 반드시 한 번은 실측되어 있어야 한다.
    over(lo)
    return lo, "solved", binding, calls


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    scale_mode = assert_scale_mode(graph, args.allow_legacy_scale)   # F2
    # D77. τ 뱅크 위치를 인자로 뺀다. 기본값 "bank" 는 예전 동작 그대로다 — 새 preset 축(`track_*`)
    # 을 넣은 τ 뱅크를 별도 폴더에 두고 k6 소스를 안 덮어쓰기 위해서만 바꾼다.
    register_external(args.external_shapes, args.external_aim)
    bank_path = path.join(out_root, args.video, args.tau_bank_dir, "bank.json")
    assert path.isfile(bank_path), f"τ 뱅크가 먼저다: python fit/bank/sample_camera_bank.py --video {args.video}"
    with open(bank_path, encoding="utf-8") as file:
        tau_bank = json.load(file)
    folder = path.join(out_root, args.video, args.bank_dir)
    makedirs(folder, exist_ok=True)

    #    `--cloud_source npz`(기본) 는 예전과 같이 cloud.npz 를 읽고, `memory` 는 recon 에서
    #    그 자리에 굽는다 (`lbm/render.py:open_renderer`).
    renderer, recon = open_renderer(args, out_root, graph)
    global RAY_GATE, RAY_CLIENT
    RAY_CLIENT = None
    if args.ray_gate == "server":
        from lbm.blender_raycast import RaycastClient
        grid = np.load(path.join(out_root, args.video, "mesh_grid.npz"), allow_pickle=True)
        RAY_GATE = {"min_clearance": args.ray_min_clearance, "min_floor_drop": args.ray_min_floor_drop,
                    "min_subject_dist": args.ray_min_subject_dist,
                    "min_clear_frac": args.ray_min_clear_frac}
        client = RaycastClient(args.video.split("_")[1])
        client.load_clip(args.video, np.asarray(grid["frame_list"]).tolist())
        RAY_CLIENT = (client, args.video, np.asarray(grid["anchor_c2w"], dtype=np.float64))
        print(f"[ray] GT mesh raycast gate ON  {RAY_GATE}", flush=True)
    num_frames = int(graph["num_frames"])
    nodes = {n["id"]: n for n in graph["nodes"]}

    # D121. 캡션 framing 절의 composition 재료 — anchor 말고 무엇이 화면에 담기나. 렌더가 0회고
    # (OBB 투영뿐) `anchor_id` 만 anchor 루프에서 갈아 끼운다. 끄면 열이 안 붙어 예전과 같다.
    composition = ({"nodes": graph["nodes"],
                    "T_gw": np.asarray(graph["frames"]["T_gw"], dtype=float),
                    "min_area": float(args.composition_min_area),
                    "max_nodes": int(args.composition_max_nodes)}
                   if args.composition else None)

    # 충돌 예산. 끄면(`--no_collision_free`) `max_behind < 0` 이 되어 이분법이 hole 만 본다.
    # `graph` 를 주면 `measure_trajectory` 가 노드 OBB clearance 까지 잰다 (G5, D49). 판정에서
    # 빼더라도(`--no_obb_gate`) 열은 남기고 싶으므로 게이트가 아니라 **측정** 쪽에 붙는다.
    # 노드별 마진 (D51). 마진 = β × (소스 카메라 자신의 최소 OBB 거리). β<1 이면 **정의상 소스
    # 카메라가 통과**하고, 씬마다 기하가 카메라에 얼마나 붙어 있는지를 실측으로 반영한다.
    # 크기 비례항(`--obb_clear_ratio`)·상한(`--obb_clear_cap`)·절대 바닥(`--min_obb_clear`)은
    # 삭제했다 (D57) — 셋 다 기본값에서 no-op 였고 D51 이 실측으로 기각한 축이다.
    assert not args.measure_obb or args.obb_clear_src_ratio > 0.0, \
        "--obb_clear_src_ratio 는 0 보다 커야 한다 (절대 마진 모드는 D57 에서 삭제)"
    # [new 2026-09-27, R62] `--obb_skip_flat R`: 바닥처럼 넓고 납작한 노드를 G5·소스 floor 에서 뺀다
    # (`lbm.gates.obb_gate_nodes` 주석). 0 (기본) 이면 obb_graph 가 graph 그대로 = 기존 동작.
    _obb_nodes, _obb_skipped = obb_gate_nodes(graph["nodes"], args.obb_skip_flat,
                                              args.obb_skip_flat_min_extent)
    obb_graph = dict(graph, nodes=_obb_nodes) if _obb_skipped else graph
    if _obb_skipped:
        print(f"[obb] 납작·광역 노드 {len(_obb_skipped)}개를 G5 에서 제외: {_obb_skipped}")
    obb_floor = (args.obb_clear_src_ratio * source_obb_clear(obb_graph)[2]
                 if args.measure_obb else 0.0)
    obb_margins = (node_margins(obb_graph["nodes"], 0.0, obb_floor, float("inf"))
                   if args.measure_obb else None)
    # D116. G1 증거 소스. τ 뱅크(`sample_camera_bank.py`)와 **같은 규칙**으로 격자를 찾는다 —
    # 두 단계가 다른 G1 으로 굽히면 사다리가 τ 뱅크에서 이미 걸러진 변이를 되살린다.
    #    `--no_collision_free` 면 G1 자체를 안 쓰므로 격자가 없다고 죽을 이유도 없다.
    mesh_grid = resolve_mesh_grid(args, out_root) if args.collision_free else ""
    behind = (behind_context(renderer, recon, graph["scale"]["S"], args.behind_src_frames,
                             args.behind_margin_frac, args.behind_clear_frac,
                             args.behind_radius_px, NEAR_PCT,
                             measure_standoff=True,
                             graph=(obb_graph if args.measure_obb else None),
                             obb_margins=obb_margins,
                             time_match=args.collision_time_match,
                             mesh_grid=mesh_grid, src_bank_c2w=renderer.cam_c2w_src,
                             collision_source=args.collision_source,
                             mesh_margin_frac=args.mesh_margin_frac,
                             mesh_margin_autoclamp=args.mesh_margin_autoclamp,
                             min_zcam_frac=args.behind_min_zcam)
              if args.collision_free else None)
    # D123 ②. G1 임계를 소스 실측 위에 얹는다. `behind` 를 먼저 지은 뒤 `clear_frac` 만 갈아
    # 끼우는 이유는 바닥을 **판정과 같은 재료**로 재야 하기 때문 (같은 frames/mask/min_zcam).
    # `clear_frac` 은 판정 시점에만 읽히므로 여기서 덮어도 순서 문제가 없다.
    src_g1 = float("nan")
    if behind and args.behind_clear_src_ratio > 0.0:
        src_g1 = source_g1_clear(behind, args.behind_clear_src_pct)
        behind["clear_frac"] = float(behind["margin_frac"]
                                     + args.behind_clear_src_ratio * src_g1)
    max_behind = args.max_behind_frac if args.collision_free else -1.0
    # 동적 채널 예산. `--collision_time_match` 일 때만 갈라진다 (아니면 `behind_static_frac` 이
    # 아예 없어서 `behind_over` 가 합집합 한 열만 본다). 기본은 정적과 같은 값이라 **판정이
    # 예전과 비트 단위로 같다** — 프레임 하나가 어느 채널에서든 걸리면 두 식이 같이 넘는다.
    max_behind_dyn = (args.max_behind_frac if args.max_behind_frac_dyn is None
                      else args.max_behind_frac_dyn)
    # 진단 열로만 쓴다 — standoff 판정은 D50 에서 끄고 D57 에서 삭제했다.
    src_standoff = source_standoff(renderer, graph["scale"]["S"]) if behind else float("nan")
    # OBB 예산 (D49). 판정량은 slack = 거리 − m_j 라 임계가 **항상 0** 이고, 끄는 값은
    # `-inf` 다 — 0 은 "박스 표면 + 마진까지 허용"이라 켠 상태다.
    obb_on = bool(behind) and args.measure_obb and args.obb_gate
    min_obb = 0.0 if obb_on else float("-inf")
    src_obb, src_obb_node, src_obb_raw = (source_obb_clear(graph, obb_margins)
                                          if behind and args.measure_obb
                                          else (float("nan"), "", float("nan")))
    # G6 예산. 둘 다 anchor 마다 기준이 달라서 여기선 켜고 끄기만 정하고, 임계는 anchor 루프에서
    # `source_elevation` 으로 잡는다 — **소스 카메라를 기각하는 임계는 버그**라는 규칙(D47/D51)이
    # 여기에도 그대로 걸린다. avocado `stat_4` chair 는 소스 자신이 이미 35.09° 다.
    elev_on = bool(behind) and args.elev_gate
    ground_on = bool(behind) and args.ground_gate
    # G7 예산 (D56). 이것도 anchor 마다 기준이 다르다 (축을 anchor 가 정의한다).
    approach_on = bool(behind) and args.approach_gate
    # G3 가림 예산 (D112). 이건 anchor 무관 **절대 비율**이다 — 소스 배수로 잡을 수가 없다:
    # 소스 카메라는 subject 를 정의상 잘 보고 있고(그래서 track 이 있다), 그 값을 β 배 하면
    # 임계가 0.9 근처로 올라가 거의 전량을 기각한다. 그래서 사용자가 준 뜻 그대로
    # "보이는 subject 점 / 전체 subject 점" 의 하한 하나다. 끄면 `-inf`.
    occl_on = bool(args.min_subject_visible) and float(args.min_subject_visible) > 0.0
    min_seen = float(args.min_subject_visible) if occl_on else float("-inf")
    assert not (occl_on and not args.subject_visible), \
        "--min_subject_visible 은 --subject_visible 이 켜져 있어야 한다 (실루엣 렌더가 분모다)"
    # D168 ②. 재시도를 부르는 어휘. 코퍼스 단계의 `--drop_status` / `--drop_suspect` 와
    # 같은 이름·같은 토큰 매칭이다 (변이 판정을 두 벌 만들지 않기 위해).
    retry_status = tuple(args.retry_status)
    retry_suspect = tuple(args.retry_suspect)
    assert not (retry_suspect and not args.suspect), \
        "--retry_suspect 는 --suspect 가 켜져 있어야 한다 (태그를 다는 쪽이 꺼져 있다)"
    # D168 ② fallback 사다리의 층 표. τ 뱅크가 `plan_tier` 를 실어 줬을 때만 켜진다
    # (`sample_camera_bank.py --variant_pool full`) — 전부 0 이면 층이 하나뿐이라 no-op 이고,
    # 그때는 아예 끈 것으로 찍어 예전 동작과 구분되게 한다.
    tier_of = {(r["anchor_id"], r["preset"]): int(r.get("plan_tier", 0) or 0)
               for r in tau_bank["variants"]}
    fallback_on = bool(args.fallback_ladder) and any(tier_of.values())
    tiers = sorted(set(tier_of.values())) if fallback_on else [None]

    anchors = args.anchors or tau_bank["axes"]["anchors"]
    # D78. `knob_kind(p)` 가 None 인 정지 preset(`STATIC_PRESETS`)도 넣는다. 예전에는 여기서
    # 통째로 빠져서, τ 뱅크에는 있는 정지 카메라가 emit 뱅크에 **0 행**이었다 (실측: k6 뱅크
    # 에서 static_hold 273 / track_hold 76 행이 τ 뱅크에만 있고 emit 은 0). 사다리를 못 만든다는
    # 이유로 preset 자체를 버리면 학습 어휘에서 "카메라가 가만히 있는다"와 `track_hold`(움직임이
    # 전부 follow offset 에서 나오는 순수 추종)가 사라진다. `--no_static_rung` 이 예전 동작이다.
    presets = args.presets or [p for p in tau_bank["axes"]["presets"]
                               if knob_kind(p) or args.static_rung]
    ladder = list(args.hole_ladder)
    # D53. `excess` 면 단이 anchor 의 **정지 hole 대비 초과분**이라 목표가 anchor·preset 마다
    # 달라진다. 사다리 값(Δ)은 그대로 `hole_delta` 로 남고 `variant_id` 도 Δ 로 매긴다 —
    # 그래야 씬이 달라져도 같은 이름의 단이 같은 뜻이다.
    excess = args.hole_mode == "excess"
    # D266. 사다리가 재는 양. `hole` 이 기본이고 그 경로는 비트 단위로 예전과 같다.
    # `shot_scale` 은 hole 을 **아예 안 본다** — 예산이 |log(면적/바닥면적)| 로 바뀐다.
    # TRUMANS 파일럿용: 충돌·가림은 depth shell 이 아니라 raycast(`bank_to_blender_poses.py
    # --raycast`)가 보고, 뱅크는 크기 사다리만 만든다 (사용자 지시 2026-09-23).
    shot_ladder = args.ladder_metric == "shot_scale"
    # 프레이밍 하한. 가림 게이트와 **다른 양**이다 — 이건 subject 중심이 중앙 박스 안이었나다.
    min_in_frame = (float(args.min_subject_in_frame) if args.min_subject_in_frame > 0.0
                    else float("-inf"))
    # [new 2026-09-27, R62] `--start_screen`: 시작 pose(격자 후보)의 **frame 0 카메라만** 같은 게이트로
    # 먼저 판정한다 (사용자 "애초에 시작 카메라가 gate 위반이면 fitting을 할 필요가 없잖아").
    # frame 0 은 손잡이·preset 과 무관하게 시작 pose 이므로 (anchor, cand_id) 마다 한 번만 잰다.
    start_verdicts, start_screen_rows = {}, []
    # F5. τ 손잡이 하한을 여기서 **한 번만** 갈아끼운다. 아래 전부가 `KNOB_RANGE` 를 읽으므로
    # (탐색 경계 `solve_knob:554` / `lo_override` / `knob_floor` 열 / `bank.json`) 값의 출처가
    # 하나로 남는다. 기본값이면 튜플이 그대로라 예전 뱅크와 비트 동일이다.
    # D266. 상한도 같이 뺀다. hole/충돌 게이트를 다 끄면 손잡이를 위에서 막는 게 사다리
    # 예산뿐인데, shot scale 이 안 변하는 preset(orbit/truck/pan)은 예산에 영원히 안 닿아
    # `unreached` 로 상한까지 벌어진다. 기본값은 지금 값 그대로라 안 주면 비트 동일이다.
    KNOB_RANGE["tau"] = (float(args.tau_knob_min),
                         float(args.tau_knob_max or KNOB_RANGE["tau"][1]))
    tau_floor_on = bool(args.tau_floor_src)
    # D53 ①-b. 하한을 `tau_start` 바로 위로 올리면 필요한 배율이 `fit_tau` 이분법의 첫 눈금
    # (max_scale/2^8 = 0.0156)보다 작아져서 `lo` 가 0 에 남는다 — 하한을 고쳐도 궤적이 여전히
    # 정지다 (avocado `stat_1 pull_out_arc`: knob 0.149 인데 path 0.000). 두 D53 모드 중
    # 하나라도 켜지면 눈금을 잘게 다시 훑게 한다. `decision_at` 이 이 값을 결정에 실어 보낸다.
    args.tau_refine = excess or tau_floor_on
    print(f"{'video':<14}{args.video}   anchors {len(anchors)}   presets {len(presets)}")
    print(f"{'hole ladder':<14}{ladder}   bisect {args.iterations}회 x {args.bisect_frames}프레임")
    print(f"{'hole mode':<14}"
          + ("excess — 목표 = anchor 정지 hole + Δ (사다리 값이 Δ)" if excess
             else "absolute — 목표 = 사다리 값 그대로 (D53 이전 동작)"))
    print(f"{'tau floor':<14}"
          + (f"tau_start + {KNOB_RANGE['tau'][0]:g}  (소스 자신의 시차 위에 얹는다)" if tau_floor_on
             else f"{KNOB_RANGE['tau'][0]:g} 고정 (D53 이전 동작)"))
    print(f"{'verify':<14}{'전 프레임' if args.verify_frames == 0 else args.verify_frames}")
    print(f"{'collision':<14}"
          + (f"G1 max_behind_frac {max_behind:g}  소스 프레임 {len(behind['frames'])}  "
             f"margin {behind['margin_frac']:g}·S  clear {behind['clear_frac']:.4f}·S  "
             f"patch r{behind['radius_px']}px  min_zcam {behind['min_zcam_frac']:g}·S\n"
             f"{'':<14}" + (f"clear = margin + {args.behind_clear_src_ratio:g} x "
                            f"(소스 자신의 G1 여유 p{args.behind_clear_src_pct:g} = {src_g1:.4f} u)"
                            f"   [D123]"
                            if args.behind_clear_src_ratio > 0.0
                            else f"clear 는 절대값 {args.behind_clear_frac:g}·S (D123 이전 동작)")
             if behind else "off (hole 만 본다)"))
    print(f"{'standoff':<14}"
          + (f"측정만 (판정 삭제, D57)  소스 카메라 자신 {src_standoff:.4f}·S"
             if behind else "off"))
    if args.measure_obb:
        m = np.asarray(sorted(obb_margins.values()), dtype=float)
        print(f"{'obb':<14}"
              + (f"slack = 거리 − m >= 0   "
                 f"m = {args.obb_clear_src_ratio:g} x (소스 카메라 자신의 최소 OBB 거리) "
                 f"= {obb_floor:.4f} u   [{m.min():.3f}, {m.max():.3f}] u\n"
                 f"{'':<14}소스 카메라 자신 slack {src_obb:+.4f} u "
                 f"(거리 {src_obb_raw:+.4f}, {src_obb_node})" if obb_on else "측정만 (판정 off)"))
    else:
        print(f"{'obb':<14}off")
    print(f"{'elev':<14}"
          + (f"|고도각| <= max({args.max_elev_deg:g}°, 소스 자신 + {args.elev_src_margin_deg:g}°)"
             if elev_on else "측정만 (판정 off)" if behind else "off"))
    print(f"{'ground':<14}"
          + (f"지면 위 여유 >= {args.min_ground_clear_ratio:g} x (소스 카메라 자신의 높이)"
             if ground_on else "측정만 (판정 off)" if behind else "off"))
    print(f"{'approach':<14}"
          + (f"시선축 근접면 여유 >= {args.approach_src_ratio:g} x (소스 카메라 자신의 여유)"
             if approach_on else "측정만 (판정 off)" if behind else "off"))
    print(f"{'occlusion':<14}"
          + (f"subject 가시비율 >= {min_seen:g}  (hole **뒤**, 이분법 렌더 2배)" if occl_on
             else "측정만 (판정 off, D81)" if args.subject_visible else "off"))
    print(f"{'retry':<14}"
          + (f"status {sorted(retry_status)} 또는 suspect {sorted(retry_suspect)} 면 "
             f"다음 preset 으로 재시도" if fallback_on
             else "off — τ 뱅크에 plan_tier 가 없다 (--variant_pool full 로 구울 것)"
             if args.fallback_ladder else "off (D168 이전 동작)"))
    print(f"{'fallback':<14}"
          + (f"층 {tiers} 순서로 (0=본 슬롯 / 1=non-track / 2=track / 3=free-moving), "
             f"쓸 만한 변이 {args.fallback_target} 개 모이면 중단" if fallback_on
             else "off"))
    print(f"{'gate order':<14}"
          + ("물리 게이트(G1/G5/G6/G7) 먼저 → 통과분만 렌더 (D113)"
             if args.gate_before_render and behind else "렌더 먼저 (D113 이전 동작)"))
    print()

    header = (f"{'anchor  preset':<34}{'hstat':>7}{'hole*':>7}{'knob':>9}{'hole':>7}"
              f"{'tau_max':>9}{'path_u':>8}{'inFr':>6}{'G1':>6}{'stdof':>7}{'obb':>8}"
              f"{'elev':>7}{'grnd':>7}{'appr':>7}  status")
    print(header)
    print("-" * len(header))

    rows, poses_all, calls_total = [], [], 0
    timing = []                          # D165. (anchor, preset) 별 소요초 — `--timing_json` 전용
    # D113. 물리 게이트에 걸려 렌더를 건너뛴 probe 수. `calls_total` 은 이분법 **호출** 수라
    # 실제 렌더 수와 다르므로 따로 센다 — 절감량이 안 보이면 이 최적화가 켜졌는지도 모른다.
    gated_probes = [0]
    # ── D168 ② fallback 사다리 (사용자 지시 2026-09-08 "가능한 preset 들 최대한 돌리고
    #    결과적으로 안 나오면 … track 없는 preset … target anchor 를 포기하고 free-moving").
    #
    # 여태 구조: (anchor, preset) 하나가 `clamped_low` 로 끝나면 **그걸로 끝**이었다. 이분법이
    # 하한에서도 게이트를 못 넘겼다는 판정이 그대로 뱅크 행이 되고, 다른 preset 을 대신 시도하는
    # 경로가 없었다 (사용자 질문 ④ 의 답이 "아니다"인 이유). d166 21편 실측으로 105행 중
    # 27% 가 `clamped_low*` / `static` 이었고, 그 자리는 그냥 비었다.
    #
    # 여기서 층을 만든다. τ 뱅크가 `plan_tier` 를 실어 주면 (`sample_camera_bank --variant_pool
    # full`) 층별로 **패스를 나눠** 돈다:
    #   0  라우팅이 준 본 슬롯 (grid2x2 + free)          ← 예전과 같은 것들
    #   1  같은 anchor 의 남은 슬롯 중 **track 아닌** 조준 preset
    #   2  같은 anchor 의 남은 `track_*`
    #   3  free-moving (**target anchor 를 포기**한 targetless 변이)
    # 층 0 을 전 anchor 에 대해 다 돌고 나서야 층 1 로 내려간다 — anchor 별로 내려가면 anchor A
    # 의 예비가 anchor B 의 본 슬롯보다 먼저 들어와 2x2 격자가 깨진다.
    # 쓸 만한 변이가 `--fallback_target` 개 모이면 그 자리에서 멈춘다 (게으름) — 안 그러면
    # 풀 전체를 굽게 되어 fit 시간이 2배다.
    # 재시도 판정은 **새 기준을 만들지 않는다**. 뱅크가 이미 갖고 있는 두 열을 읽는다:
    #   `status`   이분법 판정 (`clamped_low` / `*_limited` / `solved` / `static` …)
    #   `suspect`  D140 `tag_suspects` — 게이트는 통과했는데 수치가 캡션과 어긋나는 행
    # 코퍼스 단계가 행을 자를 때 쓰는 어휘가 정확히 이 둘이다
    # (`vista4d_bank_to_dl3dv.py --drop_status / --drop_suspect`). 그래서 인자 이름도
    # `--retry_status` / `--retry_suspect` 로 맞췄다 — "코퍼스에서 잘릴 행"과 "재시도를 부르는
    # 행"이 같은 어휘로 쓰이면 둘이 어긋날 수가 없고, 여기서 새 임계를 발명할 일도 없다.
    #
    # `tag_suspects` 를 **행 하나에 그대로** 호출한다 (함수가 행 단위로 순수하고 제자리에서
    # 고친다). 나중에 전체 행에 다시 부르는 호출은 같은 결과를 다시 써서 멱등이다. 재구현하지
    # 않는 이유가 이것 — 재구현하면 retry 기준과 코퍼스 기준이 갈라진다.
    usable_variants = set()

    def variant_usable(row):
        tag_suspects([row], args)
        if any(str(row["status"]).startswith(t) for t in retry_status):
            return False
        tags = set(str(row.get("suspect", "") or "").split("|")) - {""}
        return not (tags & set(retry_suspect))

    for anchor, tier_pass in [(a, t) for t in tiers for a in anchors]:
        if fallback_on and len(usable_variants) >= int(args.fallback_target):
            break
        # `tier_pass is None` = fallback 끔 → 예전처럼 `presets` 전량을 한 번에 돈다.
        tier_presets = ([p for p in presets if tier_of.get((anchor, p), 0) == tier_pass]
                        if fallback_on else presets)
        if not tier_presets:
            continue
        node = nodes[anchor]
        subject_points = subject_point_mask(
            renderer.indices, subject_track_volume(recon, node)).cpu().numpy()
        renderer.set_subject(subject_points)
        # G6 는 anchor 기준이라 여기서 임계를 다시 잡는다. 고도각 상한은 절대값과 "소스 자신 +
        # 여유" 중 **큰 쪽** — 소스가 이미 가파른 anchor(avocado `stat_4` chair 35.09°)에서
        # 절대값이 원본을 기각하는 걸 막는다. 지면은 소스 높이의 배수다 (camel 0.023 u /
        # avocado 0.078 u 로 3.4배 벌어져서 절대값이 두 씬을 못 덮는다).
        src_elev, src_ground = (source_elevation(graph, node) if behind
                                else (float("nan"), float("nan")))
        max_elev = (max(float(args.max_elev_deg), src_elev + float(args.elev_src_margin_deg))
                    if elev_on else float("inf"))
        min_ground = (float(args.min_ground_clear_ratio) * src_ground if ground_on
                      else float("-inf"))
        # G7 (D56). 임계는 β × (소스 카메라 자신의 시선축 여유) — G5 의 `obb_clear_src_ratio` 와
        # 같은 꼴이고 같은 β 를 쓴다. 소스 실측이 camel 0.4950 u / avocado 0.1097 u 로 4.5배
        # 벌어지는데 이건 raw OBB 여유의 4.4배와 거의 같아, 소스 배수로 잡으면 G5 마진과 자동으로
        # 같은 눈금에 놓인다. **anchor 노드만** 본다 — 축이 shot 의 subject 로 정의되므로.
        src_approach = (source_approach(graph, node) if behind else float("nan"))
        min_approach = (float(args.approach_src_ratio) * src_approach if approach_on
                        else float("-inf"))
        if behind is not None:
            behind["elev_node"] = node          # 이게 있어야 `measure_trajectory` 가 G6 를 잰다
            behind["approach_node"] = node      # 〃 G7
            behind["min_approach"] = (min_approach if approach_on else float("-inf"))
        # D259. preset 하나가 시작 pose 후보 수만큼 갈라진다. 중첩을 안 만드는 이유는
        # 루프 몸통(이분법·게이트·행 쓰기)이 후보를 **하나씩** 다루기 때문 — 짝의 목록으로
        # 펴 두면 예전 코드가 그대로 돈다. 격자 없는 뱅크는 후보가 `[None]` 하나라 짝 목록이
        # `tier_presets` 와 같고, 루프 횟수·순서가 비트 단위로 예전과 같다.
        tier_pairs = [(p, c) for p in tier_presets for c in start_cands_for(tau_bank, anchor, p)]
        for preset, start_cand in tier_pairs:
            if fallback_on and len(usable_variants) >= int(args.fallback_target):
                break
            tier = tier_of.get((anchor, preset), 0)
            kind = knob_kind(preset)
            cand_id = (start_cand or {}).get("cand_id")
            seen = [r for r in tau_bank["variants"]
                    if r["anchor_id"] == anchor and r["preset"] == preset
                    and (r.get("start_cand") or {}).get("cand_id") == cand_id]
            if not seen:
                continue                 # τ 뱅크에서 saturated 로 빠진 조합 (D41)
            points = bracket_from_bank(seen, kind)
            # D165. preset 별 실측 시간. `--timing_json` 을 안 주면 dict 만 쌓이고 아무 데도
            # 안 쓰인다 (출력 비트 동일). 재는 단위는 (anchor, preset) 한 덩어리 — 이분법
            # 렌더가 여기 전부 들어 있어서, "어느 preset 이 비싼가"는 이 값으로만 갈린다.
            _t_preset, _calls0 = perf_counter(), calls_total

            # 손잡이 → hole. 이분법 중에는 몇 프레임만 본다 (판정은 뒤에서 다시 한다).
            cache = {}

            span_frac = max(args.orbit_span_frac,
                            args.min_sweep_deg / max(float(node["obs_az_span_deg"]), 1e-6))

            def probe(knob, force_render=False, _preset=preset, _node=node, _kind=kind,
                      _cache=cache, _span=span_frac, _subject=subject_points,
                      _max_elev=max_elev, _min_ground=min_ground, _min_approach=min_approach,
                      # D259. 후보가 있으면 시작 pose 가 소스 frame0 이 아니다.
                      _cand=start_cand,
                      _smode=("board" if start_cand else args.start_mode)):
                """손잡이 → (hole, stats, near_depth, standoff, obb, |elev|max, ground,
                approach, subject 가시비율, subject 화면중앙 프레임비율).

                충돌을 안 보면 뒤 일곱은 0/nan 이다 (G6 는 `behind["elev_node"]`, G7 은
                `behind["approach_node"]` 가 없으면 nan).

                D112. 가시비율은 `--min_subject_visible` 을 켰을 때만 잰다 — subject 실루엣을
                따로 한 번 더 그려야 해서 **이분법 렌더가 2배**가 된다. 끄면 `metric_only` 고속
                경로(GPU 위에서 스칼라만 환원) 그대로라 예전 뱅크와 비트 단위로 같다.

                D113 (`--gate_before_render`, 기본 켬). 물리 게이트 다섯(G1/G5/G6×2/G7)은
                렌더가 0회다. 그중 하나라도 걸리면 `over()` 가 hole 을 **안 보므로** 그 손잡이의
                렌더는 통째로 버려진다 — 그래서 기하를 먼저 재고 걸린 손잡이는 렌더를 건너뛴다.
                그때 hole 은 `nan` 이 되는데, 최종 행의 hole 열은 이 캐시가 아니라 뒤에서
                `verify_frames` 로 **다시 재므로** 뱅크 CSV 는 바뀌지 않는다.

                `force_render=True` 는 그 예외 하나다: D53 의 `hole_static`(손잡이 0) 은 hole
                값 **자체가** 사다리 목표를 정하므로 게이트에 걸려도 반드시 재야 한다.

                F7 (`--time_truncate`, 기본 끔). 게이트에 걸리면 **크기를 줄이기 전에** 시간축
                으로 자른다 (`solve_hold_from`). 캐시에 `hold_from` 이 같이 들어가고, 그 값이
                행의 열로 나가 `emit_bank` 가 같은 절단을 되풀 수 있게 한다.
                """
                if knob not in _cache:
                    mult = 1.0
                    for attempt in range(args.shape_doublings + 1):
                        decision = decision_at(graph, _node, _preset, knob, _kind, args, mult,
                                               start_cand=_cand)
                        poses, extra = build_poses(
                            decision, graph, board=None, num_frames=num_frames,
                            orbit_span_frac=_span, start_mode=_smode,
                            aim_anchor=args.aim_anchor, aim_ramp_frames=args.aim_ramp_frames,
                            traj_basis=args.traj_basis, aim_keyframes=args.aim_keyframes,
                            keyframe_aim=args.keyframe_aim, keyframe_ease=args.keyframe_ease,
                            # D90. 안 넘기면 `build_poses` 서명 기본값(12/0.5)이 쓰이는데
                            # `emit_bank` 는 자기 CLI 기본값(4)으로 되만들어 회전만 어긋난다
                            # (위치는 정확히 일치해서 더 안 보인다). 명시해서 `fixed` 에 싣는다.
                            smooth_passes=args.smooth_passes,
                            smooth_lambda=args.smooth_lambda,
                            preset_tracking=args.preset_tracking, deroll=args.deroll,
                            orbit_fixed_sweep=args.orbit_fixed_sweep,
                            # D171. 0 이면 `build_poses` 가 예전 저역통과 경로를 그대로 탄다.
                            follow_keyframes=args.follow_keyframes,
                            follow_kf_interp=args.follow_kf_interp,
                            tau_denom=args.tau_denom)
                        # `fit_tau` 가 max_scale 에 붙었으면 τ 를 못 맞춘 것이다 — 모양을 키워 다시.
                        # 마지막 시도에서는 **곱하지 않고** 끝낸다: 여기서 곱하면 `_cache` 에
                        # 남는 `mult` 가 방금 만든 `poses` 를 만든 값보다 headroom 배 크고,
                        # `emit_bank` 가 그 배율로 되만들어 궤적이 2배로 벌어진다 (실측:
                        # basketball-four `dyn_0__pull_out_arc__hole0.5`, 최대 8.918 어긋남).
                        # 이 가지는 `status == "shape_limited"` 행에서만 밟힌다.
                        if (_kind != "tau"
                                or extra["info"]["tau"]["scale"] < FIT_TAU_MAX_SCALE - 1e-9
                                or attempt == args.shape_doublings):
                            break
                        mult *= args.shape_headroom
                    verdict = (lambda geo: physical_verdict(
                        geo, max_behind, min_obb, _max_elev, _min_ground,
                        _min_approach, max_behind_dyn))
                    # F7. 크기를 줄이기 **전에** 시간축으로 잘라 본다 — 게이트에 걸리는 프레임
                    # 앞까지만 움직이고 그 뒤는 위치를 얼린다. 렌더 0회 (`geometry_stats`).
                    # 절단이 필요 없거나(통과) 최소 이동조차 못 하면 `hold_from = num_frames`
                    # 로 떨어져 `truncate_hold` 가 입력을 그대로 돌려주므로, 이 가지는
                    # `--no_time_truncate` 와 **비트 단위로 같다**.
                    hold_from = num_frames
                    if args.time_truncate and behind:
                        min_hold = max(2, int(round(args.min_move_frac * num_frames)))
                        solved = solve_hold_from(renderer, poses, behind, verdict, min_hold)
                        hold_from = num_frames if solved is None else int(solved)
                        if hold_from < num_frames:
                            poses = truncate_hold(poses, hold_from)
                            retime_info(extra, poses, behind["cam_c2w"],
                                        float(extra["info"]["z_med"]))
                    gate_check = (None if (force_render or not args.gate_before_render)
                                  else verdict)
                    stats, _ = measure_trajectory(renderer, poses, num_frames, args.bisect_frames,
                                                  args.tile_height, args.tile_width,
                                                  args.center_box, behind=behind,
                                                  subject_points=(_subject if occl_on else None),
                                                  metric_only=not occl_on,
                                                  gate_check=gate_check)
                    if RAY_CLIENT is not None:
                        # D277. 이 손잡이의 궤적(뱅크 world) 을 blend world 로 옮겨 서버에 묻는다.
                        _row = RAY_CLIENT[0].profile(RAY_CLIENT[1],
                                                     [RAY_CLIENT[2] @ np.asarray(poses)])[0]
                        stats.update({f"ray_{k}": float(v) for k, v in _row.items()})
                    if stats.get("gated"):
                        gated_probes[0] += 1
                    stats["_knob"] = float(knob)     # D275 motion 사다리의 metric (손잡이 자체)
                    _cache[knob] = (stats, poses, extra["info"], mult, hold_from)
                stats = _cache[knob][0]
                # 두 번째 자리는 예전엔 `behind_frac` 스칼라였는데 **stats 통째**로 바꿨다
                # (2026-09-02): G1 이 정적·동적 두 채널을 각자의 예산과 따로 보게 되면서
                # 스칼라 하나로는 판정이 안 된다. 판정식은 `behind_over` 한 군데에만 있다.
                return (stats["hole_fraction"], stats,
                        stats.get("near_depth", float("nan")),
                        stats.get("standoff", float("nan")),
                        stats.get("obb_slack", float("nan")),
                        stats.get("elev_abs_max", float("nan")),
                        stats.get("ground_clear", float("nan")),
                        stats.get("approach_gap", float("nan")),
                        stats.get("subject_visible_frac", float("nan")))

            # [new 2026-09-27, R62] 시작 카메라 선판정. `off`(기본)면 이 블록이 안 돌아 예전과 비트 동일.
            #   only   : 판정만 하고 fit 은 전부 건너뛴다 → start_screen.csv
            #   filter : 탈락한 시작 pose 는 fit 을 건너뛰고(행 없음), 통과한 것만 예전처럼 fit
            if args.start_screen != "off" and start_cand is not None:
                _key = (anchor, cand_id)
                if _key not in start_verdicts:
                    probe(0.0, force_render=True)          # 손잡이 0 = 시작 pose 에 얼린 궤적
                    _pose0 = np.asarray(cache[0.0][1])[:1]  # frame 0 카메라 하나
                    _st, _ = measure_trajectory(renderer, _pose0, 1, 1, args.tile_height,
                                                args.tile_width, args.center_box, behind=behind,
                                                subject_points=(subject_points if occl_on else None),
                                                metric_only=not occl_on)
                    _why = physical_verdict(_st, max_behind, min_obb, max_elev, min_ground,
                                            min_approach, max_behind_dyn)
                    _vis = float(_st.get("subject_visible_frac", float("nan")))
                    if _why is None and _vis == _vis and _vis < min_seen:
                        _why = "occlusion"
                    _inf = float(_st.get("subject_in_frame", float("nan")))
                    if _why is None and _inf == _inf and _inf < min_in_frame:
                        _why = "framing"
                    start_verdicts[_key] = _why
                    start_screen_rows.append({
                        "anchor_id": anchor, "cand_id": cand_id, "ok": int(_why is None),
                        "reason": _why or "", "subject_visible": _vis,
                        "subject_area": _st.get("subject_area_med", float("nan")),
                        "obb_slack": _st.get("obb_slack", float("nan")),
                        "obb_node": _st.get("obb_node", ""),
                        "ground_clear": _st.get("ground_clear", float("nan")),
                        "elev_abs": _st.get("elev_abs_max", float("nan")),
                        "behind_static": _st.get("behind_static_frac", float("nan")),
                        "behind_dyn": _st.get("behind_dyn_frac", float("nan")),
                        "hole": _st.get("hole_fraction", float("nan"))})
                if args.start_screen == "only" or start_verdicts[_key] is not None:
                    continue
            # D53. 사다리의 바닥을 먼저 실측한다. 손잡이 0 은 `fit_tau` 의 "예산을 이미 다 썼다"
            # 가지를 그대로 타서 **시작 pose 에 얼린** 궤적을 준다 — 그게 정지 hole 이고, 같은
            # 호출에서 `tau_start`(씬 상수, 소스 자신의 시차)도 나온다. 렌더 1회.
            hole_static, tau_start_src = 0.0, float("nan")
            # D266. shot scale 사다리의 바닥. `hole_static` 과 **같은 probe 한 번**에서 나온다 —
            # `render_metrics` 가 이분법용 싼 경로에서도 `subject_area` 를 돌려주므로
            # (`lbm/render.py:196`) 면적 측정에 렌더가 추가로 들지 않는다. 가림(2-pass)만
            # 비싼 것이고 면적은 공짜다. 그래서 `--min_subject_visible 0` 으로 가림 게이트를
            # 꺼도 shot scale 은 그대로 측정된다.
            area_static = float("nan")
            if excess or tau_floor_on or shot_ladder:
                # D113. `force_render` — 이 hole 은 사다리 목표(`hole_static + Δ`)를 정하는
                # 값이라 게이트에 걸려도 반드시 재야 한다. 손잡이 0 은 시작 pose 에 얼린
                # 궤적이라 보통 게이트를 다 통과하지만, 시작 pose 자체가 지면 아래거나
                # anchor 를 지나쳐 있으면 걸린다 (그러면 사다리 전 단이 nan 이 된다).
                _p0 = probe(0.0, force_render=True)
                hole_static = float(_p0[0])
                area_static = float(_p0[1].get("subject_area_med", float("nan")))
                tau_start_src = float(cache[0.0][2]["tau"]["tau_start"])
            # |log(면적 / 바닥면적)|. **비율의 로그**인 이유 둘. ① 손잡이에 대해 단조다 —
            # push_in 은 면적이 커지고 pull_out 은 작아지는데 로그 절대값은 양쪽 다 0 에서
            # 단조증가라 이분법이 성립한다 (생면적 목표는 방향마다 부등호가 뒤집혀 안 된다).
            # ② anchor 크기에 불변이다 — 사람 하나와 방 하나가 같은 Δ 를 같은 뜻으로 쓴다.
            # Δ 0.2/0.4/0.7/1.1 ≈ 배율 ×1.22/×1.49/×2.0/×3.0.
            def shot_dev(stats, _base=area_static):
                area = float(stats.get("subject_area_med", float("nan")))
                if not (area == area and _base == _base) or area <= 0.0 or _base <= 0.0:
                    return float("nan")      # nan 은 `>=` 가 거짓이라 "예산 안"으로 떨어진다
                return abs(log(area / _base))
            # τ 하한을 소스 시차 위로 올린다. pan 은 손잡이가 각도라 이 병이 없다 (τ 가 스케일에
            # 불변이라 애초에 `fit_tau` 를 안 탄다, D40).
            lo_override = (tau_start_src + KNOB_RANGE["tau"][0]
                           if tau_floor_on and kind == "tau" and tau_start_src == tau_start_src
                           else None)

            # D78. 정지 preset 은 손잡이가 없으니 사다리도 없다 — rung 1개로 통과시킨다.
            # `sample_camera_bank.py` 가 τ 뱅크에서 하는 처리와 같은 예외다. 손잡이 값은 τ 뱅크
            # 행의 `target_tau` 를 그대로 쓴다: 정지·순수추종은 궤적이 스케일에 반응하지 않아
            # 값 자체는 무의미하지만, 같은 값을 실어야 `emit_bank` 가 τ 뱅크와 **같은 결정**을
            # 되만든다 (pose 대조 assert 가 이 일치를 검사한다).
            rungs = [ladder[0]] if kind is None else ladder
            # D275. targetless preset 은 shot scale 대신 **이동량 사다리** — 단 i 의 목표는
            # `--motion_ladder[i]` (τ). 회전(pan_deg)은 τ 뱅크와 같은 60°×τ 로 바꾼다
            # (`sample_camera_bank.py` 의 pan_deg = 60 × τ / 1.00). 이분법·물리 게이트는 그대로라
            # 게이트가 물리면 작게 풀린다. 프레이밍 게이트는 끈다 (subject 를 안 겨눈다).
            motion = bool(shot_ladder and args.shot_exempt == "targetless"
                          and preset in TARGETLESS_PRESETS and kind is not None)
            for rung_i, delta in enumerate(rungs):
                # shot 사다리에서 Δ 는 **바닥 대비 배율의 로그**라 이미 상대량이다 —
                # `excess` 처럼 바닥을 더하면 이중으로 상대화된다. 그래서 그대로 쓴다.
                target = delta if shot_ladder else (hole_static + delta if excess else delta)
                if kind is None:
                    knob, status, binding, calls = float(seen[0]["target_tau"]), "static", "", 0
                    probed = probe(knob)         # 캐시에 poses/info 를 채운다 (렌더 1회)
                    # D192. 정지 preset 은 손잡이가 없어 `solve_knob` 을 건너뛰는데, 그러면
                    # **게이트 판정도 같이 건너뛴다**. 열은 위 `probe` 가 정상적으로 채우는데
                    # 아무도 안 읽는다. d185/d183/d179 의 `track_look_at` 3,205행 실측:
                    #     obb_slack < 0               11.3%   (카메라가 노드 OBB 안)
                    #     behind_frac > 0             21.7%   (G1 — 표면 뒤)
                    #     subject_visible_frac < 0.6  16.7%   (subject 가 가려짐)
                    #     합집합                       44.6%
                    # status 가 `static` 하나뿐이라 `--pick_budget` 의 `ok` 필터도 이걸 못 가른다
                    # (status 로만 거른다) — 그 행이 씬의 유일한 카메라로 뽑힌다.
                    # 켜면 `over()` 와 **같은 식·같은 순서**로 한 번 판정해 `f"{사유}_blocked"`
                    # 로 적는다. 그러면 `--retry_status ..._blocked` 가 사다리를 다음 층(다른
                    # anchor)으로 내리고 pick 의 `ok` 에서도 빠진다.
                    # `_limited` 가 아니라 `_blocked` 인 이유: `_limited` 는 "게이트가 크기의
                    # 천장을 정했지만 수렴한 카메라"라 쓸 수 있는 행이고 (`fit_hole_ladder`
                    # :1122), 이쪽은 게이트를 **위반한** 행이라 뜻이 반대다. 같은 접미사를 쓰면
                    # 기존 `--retry_status *_limited` 설정이 둘을 한꺼번에 집어간다.
                    # 기본 off — 안 주면 위 세 줄만 남아 옛 동작과 비트 동일하다.
                    if args.gate_static:
                        _geo, _seen_frac = probed[1], probed[8]
                        why = physical_verdict(_geo, max_behind, min_obb, max_elev,
                                               min_ground, min_approach, max_behind_dyn)
                        if why is None and _seen_frac == _seen_frac and _seen_frac < min_seen:
                            why = "occlusion"
                        # D266. 프레이밍도 같은 판정에 넣는다 — `over()` 와 같은 식이다.
                        _inf = probed[1].get("subject_in_frame", float("nan"))
                        if why is None and _inf == _inf and _inf < min_in_frame:
                            why = "framing"
                        if why is not None:
                            status, binding = f"{why}_blocked", why
                elif motion:
                    goal = float(args.motion_ladder[min(rung_i, len(args.motion_ladder) - 1)])
                    goal_k = goal * 60.0 if kind == "pan_deg" else goal
                    # 목표를 goal 바로 위에 둔다 — `over()` 가 `metric >= target` 이라 손잡이가
                    # 정확히 goal 이면 "넘었다" 로 판정되어 한 단 아래로 내려앉는다 (실측 12°→11.4°).
                    target = goal_k * (1.0 + 1e-6)
                    # metric 이 손잡이 자체라 bracket 을 목표점 하나로 준다 — 게이트가 허락하면
                    # 정확히 goal 에 앉고, 물리면 [하한, goal] 에서 이분법으로 내려간다.
                    # (빈 bracket 은 [2°,180°] 전 구간 4회 이분이라 12° 목표가 2° 로 떨어졌다.)
                    knob, status, binding, calls = solve_knob(probe, target,
                                                              [(goal_k, goal_k)], kind,
                                                              args.iterations, max_behind, min_obb,
                                                              max_elev, min_ground, lo_override,
                                                              min_approach, min_seen,
                                                              max_behind_dyn,
                                                              metric_fn=lambda st: st["_knob"],
                                                              min_in_frame=0.0)
                else:
                    # D266. shot 사다리에서는 bracket 을 **안 쓴다** — `points` 는 τ 뱅크가 hole
                    # 로 매긴 (손잡이, hole) 쌍이라 shot scale 의 bracket 이 아니다. 빈 리스트를
                    # 주면 범위 양끝을 재고 이분법으로 들어간다 (probe 2회 추가).
                    knob, status, binding, calls = solve_knob(probe, target,
                                                              [] if shot_ladder else points, kind,
                                                              args.iterations, max_behind, min_obb,
                                                              max_elev, min_ground, lo_override,
                                                              min_approach, min_seen,
                                                              max_behind_dyn,
                                                              metric_fn=(shot_dev if shot_ladder
                                                                         else None),
                                                              min_in_frame=min_in_frame)
                calls_total += calls
                _, poses, info, mult, hold_from = cache[knob]
                # 모양을 최대치까지 키우고도 max_scale 에 붙어 있으면 knob 이 아니라 **모양이 천장**이다.
                if status == "unreached" and info["tau"]["scale"] >= FIT_TAU_MAX_SCALE - 1e-9:
                    status = "shape_limited"
                # 판정은 더 촘촘히 다시 잰다 — 이분법용 5프레임은 사다리를 고르는 데만 쓴다.
                # 충돌이 상한을 정했으면 status 로 남긴다 — hole 만 보면 "덜 큰 이유"가 안 보인다.
                if status == "solved" and binding in ("wall", "floor", "subject", "occluded"):
                    status = f"{binding}_limited"      # D277 raycast 게이트가 크기를 정했다
                if status == "solved" and binding in ("collision", "clearance", "obb",
                                                      "elev", "ground", "approach", "occlusion",
                                                      "framing"):
                    status = f"{binding}_limited"
                # D53 ①. 올린 하한에 닿았으면 그렇게 찍는다 — 예전엔 이게 `clamped_low` 로만
                # 나와서 "작지만 정상"과 "손잡이가 도달 불가능한 구간에 있다"가 구분이 안 됐다.
                # 구분자가 쉼표면 안 된다 — `bank.csv` 는 따옴표 없이 `",".join` 으로 쓰므로
                # `solved,tau_floor` 가 두 칸으로 쪼개져 뒤 열이 통째로 밀린다 (실측: avocado
                # 10 행에서 `binding` 이 `tau_floor` 로 찍혔다).
                if lo_override is not None and knob <= lo_override + 1e-9:
                    status = f"{status}+tau_floor"
                frames = num_frames if args.verify_frames == 0 else args.verify_frames
                # 가림은 **여기서만** 잰다 — 이분법(위 `probe`)에 넘기면 렌더가 2배가 되는데,
                # 이분법이 푸는 답은 물리 게이트와 hole 이지 가림이 아니다 (D81).
                # 시간축(`subject_area_seq`)도 **여기서만** 붙인다 — 이분법 5프레임으로는
                # push-in 인지 pull-out 인지 못 가르고, 어차피 캡션이 읽는 건 이 행이다 (D119).
                # composition (D121) 도 같은 이유로 여기만 — 렌더는 0회지만 이분법 5프레임에서
                # 재면 "들어왔다/나갔다"의 앞뒤 1/3 이 두 프레임씩밖에 안 된다.
                stats, _ = measure_trajectory(renderer, poses, num_frames, frames,
                                              args.tile_height, args.tile_width, args.center_box,
                                              behind=behind,
                                              subject_points=(subject_points
                                                              if args.subject_visible else None),
                                              area_timeline=args.area_timeline,
                                              composition=(dict(composition, anchor_id=anchor)
                                                           if composition else None))
                calls_total += len(stats["measured_frames"])
                rows.append({
                    # 이름은 **Δ** 로 매긴다 — excess 모드에서 `target_hole` 은 anchor 마다
                    # 달라지므로 이름에 쓰면 씬 간 같은 단이 다른 이름이 된다.
                    # D259. 시작 pose 후보는 **뒤**에 붙인다 (`split("__")[0]` 로 anchor 를 읽는
                    # 하류 5곳이 안 깨진다). 격자를 끄면 접미사가 없어 예전 이름 그대로다.
                    # D266. shot 사다리는 접두를 `shot` 으로 바꾼다 — 같은 `hole0.2` 이름이
                    # 두 가지 눈금을 뜻하면 세대 간 대조가 조용히 어긋난다. 하류가 단을 읽는
                    # 정식 경로는 `hole_delta` **열**이고(`emit_bank.rung_of`), 그 열은 그대로
                    # 채운다 — 이름을 파싱하는 곳은 릴 도구 하나뿐이다
                    # (`render_target_swap_warp.py:40`, 못 읽으면 inf 로 뒤로 민다).
                    "variant_id": (f"{anchor}__{preset}__"
                                   f"{'motion' if motion else ('shot' if shot_ladder else 'hole')}{delta:g}"
                                   + (f"__{cand_id}" if cand_id else "")),
                    "anchor_id": anchor, "anchor_label": node["label"], "preset": preset,
                    # D259. 후보 전문을 그대로 옮겨 싣는다 — `emit_bank` 가 이 값으로 같은
                    # 카메라를 되만든다. 격자를 끄면 키 자체가 없다 (예전 뱅크와 같은 JSON).
                    **({"start_cand": start_cand} if start_cand else {}),
                    "target_hole": round(float(target), 5), "hole_delta": delta,
                    "hole_static": round(hole_static, 4),
                    # D266. 사다리가 무엇을 재고 있나 + shot 모드의 바닥 면적. `hole` 이면
                    # 두 열이 비어 예전 뱅크와 같은 스키마다.
                    **({"ladder_metric": "motion" if motion else "shot_scale",
                        "area_static": round(area_static, 5),
                        "shot_dev": (lambda d: round(d, 4) if d == d else "")(shot_dev(stats))}
                       if shot_ladder else {}),
                    # D277. 이분법이 본 raycast 값 (그 손잡이 캐시에서). 게이트 꺼지면 키 없음.
                    **{k: round(float(v), 4) for k, v in cache[knob][0].items()
                       if k.startswith("ray_")},
                    "knob_floor": (round(float(lo_override), 5) if lo_override is not None
                                   else KNOB_RANGE[kind][0] if kind else 0.0),
                    # D78. 정지 preset 은 `"none"` 으로 찍는다 — None 이면 CSV 에 "None" 으로
                    # 나가고 요약표의 `:>7` 포맷이 TypeError 로 죽는다. `emit_bank` 는
                    # `== "pan_deg"` 만 보므로 "none" 은 τ 가지로 떨어져 재현이 맞는다.
                    "knob_kind": kind or "none", "knob": round(float(knob), 5),
                    # `knob` 은 **표시용 반올림**이라 되만들기에 쓰면 안 된다. `fit_tau` 이분법이
                    # 스케일을 계단으로 양자화해서 손잡이→궤적이 불연속이다 — 5자리 반올림이
                    # 계단 경계를 1.8e-6 넘기면 궤적이 통째로 한 칸 커진다 (실측: snow-dog
                    # `stat_0__truck_right__hole0.1`, 참값 0.228128185878 → 0.22813 →
                    # scale 0.01428→0.01434, 궤적 0.43% 확대, 최대 1.793e-03 어긋남).
                    # `emit_bank` 는 이 열을 우선해서 읽는다.
                    "knob_raw": float(knob),
                    # `emit_bank` 가 이 행으로 결정을 되만들 때 같은 배율이 나와야 한다.
                    "tau_refine": bool(args.tau_refine),
                    "status": status, "binding": binding, "shape_mult": mult,
                    # F7. 위치를 얼리기 시작한 프레임. `num_frames` 면 절단 없음. 빈 칸이면
                    # `--no_time_truncate` 로 구운 뱅크 = F7 이전과 같다 (`emit_bank` 가
                    # 빈 칸을 "절단 없음"으로 읽는다).
                    "hold_from": (int(hold_from) if args.time_truncate else ""),
                    # D93. 요청값이 아니라 **실제 쓴 값** (`track_*` 은 lock 으로 풀린다).
                    "speed": args.speed, "tracking": info["tracking"],
                    "tracking_requested": args.tracking,
                    "look_at_bias": args.look_at_bias, "aim": info["aim"],
                    # D97. `tau_max` 는 언제나 **소스 기준** (hole 예산과 같은 눈금).
                    # `tau_ref_max` 는 `tau_ref` 가 가리키는 기준 위의 값 = 이분법이 실제로
                    # 맞춘 것. hole 손잡이 뱅크에서는 둘 다 결과값이라 판정에는 안 쓰이지만,
                    # `track_*` 이 얼마나 상대 이동했는지는 후자로만 읽힌다.
                    "tau_ref": info["tau"]["tau_ref"], "tau_ref_max": info["tau"]["tau_max"],
                    "tau_max": info["tau"]["tau_max_final"], "tau_start": info["tau"]["tau_start"],
                    "path_len_u": info["path_len_u"], "radius_u": info["radius_u"],
                    "view_angle_max_deg": info["view_angle_max_deg"],
                    "sweep_deg": info["shape_context"]["sweep"],
                    "pan_deg": round(info["shape_context"]["pan"] * info["tau"]["scale"], 3),
                    "src_elev_abs_max": round(src_elev, 2), "max_elev_deg": round(max_elev, 2),
                    "src_ground_clear": round(src_ground, 4),
                    "min_ground_clear": round(min_ground, 4),
                    "src_approach": round(src_approach, 4),
                    "min_approach": round(min_approach, 4) if approach_on else "",
                    # D112. 판정에 쓴 가림 하한. 빈 칸이면 측정만 한 뱅크(D81~D111)와 같다.
                    "min_subject_visible": round(min_seen, 4) if occl_on else "",
                    # D168 ②. 이 변이가 어느 층에서 왔나 (0 = 라우팅이 준 본 슬롯,
                    # 1/2/3 = fallback 예비). `--no_fallback_ladder` 면 전부 0.
                    "plan_tier": tier,
                    **stats})
                poses_all.append(poses)
                # D168 ②. 사다리가 여러 단이면 **한 단이라도** 풀리면 그 변이는 쓸 만한 것으로
                # 센다 (집합이라 중복 안 됨). 예산 판정은 변이 단위지 행 단위가 아니다.
                if variant_usable(rows[-1]):
                    usable_variants.add((anchor, preset))
                print(f"{anchor + '  ' + preset:<34}{hole_static:>7.3f}{target:>7.2f}{knob:>9.3f}"
                      f"{stats['hole_fraction']:>7.3f}{info['tau']['tau_max_final']:>9.4f}"
                      f"{info['path_len_u']:>8.3f}{stats['subject_in_frame']:>6.2f}"
                      f"{stats.get('behind_frac', float('nan')):>6.2f}"
                      f"{stats.get('standoff', float('nan')):>7.3f}"
                      f"{stats.get('obb_slack', float('nan')):>8.3f}"
                      f"{stats.get('elev_abs_max', float('nan')):>7.1f}"
                      f"{stats.get('ground_clear', float('nan')):>7.3f}"
                      f"{stats.get('approach_gap', float('nan')):>7.3f}"
                      # F7. 켰을 때만 찍는다 — 끄면 예전 출력 그대로다.
                      f"{f'{hold_from:>5d}' if args.time_truncate else ''}  {status}")

            timing.append({"anchor_id": anchor, "preset": preset,
                           "rungs": len(rungs), "solve_calls": calls_total - _calls0,
                           "seconds": round(perf_counter() - _t_preset, 3)})

    if not rows:
        # D67. 여기 오는 경우가 둘이라 구분해야 한다.
        #
        # (a) **소스 자신이 τ 예산을 넘겼다.** `tau_start`(정지 플랜의 τ — 플랜은 anchor 시작
        #     pose 에 가만히 있는데 소스 카메라가 날아가서 생기는 시차)가 사다리 꼭대기보다
        #     크면 D53 의 "τ 하한 = tau_start + 0.02" 가 사다리 전 단을 saturated 로 밀어낸다.
        #     `sample_camera_bank` 가 움직이는 조합을 통째로 `dropped_saturated` 로 빼고 정지
        #     preset 만 남기는데 그마저 saturated 표시라 `:513 continue` 가 전부 건너뛴다.
        #     실측: snowboard 1.9441 / snow-bike 1.3765 (사다리 꼭대기 1.0). 52편 중 2편.
        #     이건 **버그가 아니라 그 소스로는 만들 카메라가 없다는 결론**이므로 크래시가 아니라
        #     사유를 적고 rc=0 으로 나간다 — 안 그러면 배치 런너가 한 편 때문에 죽고, 죽지
        #     않더라도 "실패"와 "해당 없음"이 rc 로 구분이 안 된다.
        #
        # (b) `--anchors`/`--presets` 필터를 잘못 줘서 아무것도 안 걸린 것. 이건 진짜 실수라
        #     예전처럼 죽어야 한다. 둘을 가르는 건 "τ 뱅크에 생존 변이가 있었나"다.
        filtered = [r for r in tau_bank["variants"]
                    if r["anchor_id"] in anchors and r["preset"] in presets]
        if args.start_screen == "only":       # [new 2026-09-27, R62] 판정만 — fit 행이 0 인 게 정상
            write_start_screen(folder, start_screen_rows)
            return
        assert not filtered, (
            f"푼 게 0 인데 τ 뱅크에는 조합이 {len(filtered)} 개 남아 있다 — "
            "anchors/presets 필터가 아니라 이분법 쪽을 볼 것")
        dropped = tau_bank.get("dropped_saturated") or []
        tau_start = max((float(d.get("tau_start", 0.0)) for d in dropped), default=0.0)
        skipped = {
            "format": "lbm_hole_bank_skipped_v1", "video": args.video,
            "reason": "tau_start_exceeds_ladder",
            "detail": ("소스 카메라 자신의 시차가 τ 사다리 꼭대기를 넘어 "
                       "움직이는 변이가 전부 saturated 로 빠졌다 (D67)."),
            "tau_start": tau_start, "tau_ladder": tau_bank["axes"]["tau_ladder"],
            "hole_ladder": ladder, "hole_mode": args.hole_mode,
            "num_dropped_saturated": len(dropped),
            "num_surviving_variants": len(tau_bank["variants"]),
            "surviving_presets": sorted({r["preset"] for r in tau_bank["variants"]}),
            "S": tau_bank.get("S"), "z_med": tau_bank.get("z_med"),
        }
        with open(path.join(folder, "skipped.json"), "w", encoding="utf-8") as file:
            json.dump(skipped, file, ensure_ascii=False, indent=1)
        print(f"\n건너뜀 — {skipped['reason']}")
        print(f"  tau_start        {tau_start:.4f}   사다리 꼭대기 {max(tau_bank['axes']['tau_ladder']):.4f}")
        print(f"  saturated 조합   {len(dropped)}")
        print(f"  남은 변이        {len(tau_bank['variants'])}  {skipped['surviving_presets']}")
        print(f"  기록             {path.join(folder, 'skipped.json')}")
        return

    # D140. 게이트가 막은 게 아닌데 수치가 나쁜 행에 `suspect` 태그. `status` 는 안 건드린다.
    # `--no_suspect` 면 열이 빈 칸이라 D139 이전 뱅크와 같다.
    suspect_counts = tag_suspects(rows, args) if args.suspect else {}
    if not args.suspect:
        for row in rows:
            row["suspect"] = ""

    # ── D188 ③ 씬당 예산. 사다리는 **어디까지 내려갈지**를 정하지 무엇을 내보낼지는 안 정한다.
    #
    # `--fallback_target 1` 이 "카메라 1대"를 뜻하지 않는다는 게 d188 파일럿 58편의 결론이다.
    # 그 인자는 `variant_usable()` 이 참인 변이를 세는데, 그 함수는 `--retry_suspect` 도 보므로
    # `hole_over_budget`(hole > `--suspect_hole` 0.35) 이 붙은 solved 행은 **안 세어진다**.
    # 실측: solved 92행 중 57행이 그 태그였고(`hole_mode excess` 라 `hole_static` 이 크면 예산
    # 자체가 0.35 를 넘는다), 23/58 편이 `usable` 0 으로 끝나며 사다리가 3층을 다 돌았다 —
    # 그 결과 뱅크에 solved 가 씬당 최대 6행 쌓였다. 태그를 무시하게 고치면 반대로 "hole 0.6
    # 짜리 tier0" 가 "hole 0.2 짜리 tier1" 를 이겨서 어휘는 맞고 품질이 무너진다.
    #
    # 그래서 **탐색과 선택을 분리**한다. 사다리는 지금처럼 깨끗한(usable) 변이를 찾아 계속
    # 내려가고, 다 돌고 나서 여기서 씬당 `--pick_budget` 개를 고른다:
    #     track 먼저  →  `plan_tier` 오름차순  →  등급(usable 먼저)  →  `hole_fraction` 오름차순
    # 행은 **하나도 안 지운다** — `picked` 열만 단다 (뱅크는 재고 목록이라는 D39/D45 규칙).
    # `emit_bank.py --picked_only` 가 그 열을 소비한다. 기본값 0 = 열이 빈 칸, 예전 뱅크와 동일.
    if args.pick_budget:
        def pick_key(row):
            # `variant_usable()` 을 다시 부르지 않는다 — 그 함수는 `tag_suspects` 를 호출해
            # `suspect` 열을 **다시 쓴다**. `--no_suspect` 면 위에서 비운 열이 되살아난다.
            # 여기서는 이미 확정된 열만 읽는다 (같은 판정, 부작용 없음).
            tags = set(str(row.get("suspect", "") or "").split("|")) - {""}
            grade = 1 if tags & set(retry_suspect) else 0
            hole = row.get("hole_fraction")
            # **`plan_tier` 가 아니라 track 여부가 1순위다.** 지시는 "움직이는 dynamic anchor 면
            # track+object-centric, 아니면 object-centric" 인데 `plan_tier` 는 그 뜻이 아니다 —
            # tier 0 은 "라우터가 예산 안에 넣은 슬롯"이고 (`sample_camera_bank.plan_variants`
            # :226 이 예산 안이면 무조건 0 을 준다) 그 슬롯은 anchor 가 움직여도 평범 preset 일
            # 수 있다. tier 를 1순위로 두면 track 은 예비층(tier 2)에 있다는 이유만으로 진다.
            #
            # 실측(뱅크 82편, `tmp/d188/project_pick_keys.py`): tier 먼저는 track 39.7%,
            # track 먼저는 57.4%. 수율은 **양쪽 다 68대 / 0대 14편으로 동일**하고 hole 은 오히려
            # 좋아진다 (mean 0.3516 -> 0.3463). 갈린 12편 중 8편 개선 / 4편 악화. 즉 이건
            # 어휘↔품질 맞교환이 아니라 tier 가 그냥 틀린 키였던 것이다.
            #
            # track 행은 `track_ok()` 가 `--track_dynamic_only` + `--track_min_drift_u` 로 이미
            # 걸러 **충분히 움직이는 anchor 에만 존재**한다. 그래서 여기서 `moving` 을 다시 볼
            # 필요가 없다 — track 행이 있다는 것 자체가 "움직인다"의 증거다. 없으면 자동으로
            # object-centric 이 1순위가 되므로 지시의 "아니면" 갈래도 그대로 성립한다.
            #
            # tier 는 2순위로 남긴다 (같은 track 끼리는 라우터가 고른 슬롯이 먼저). grade 는 그
            # 다음 — 거짓 경보(`hole_over_budget`)가 어휘를 덮지 않게 하려던 fc56306 의 의도는
            # 유지된다. hole 은 마지막 동점 처리.
            return (0 if row["preset"].startswith("track_") else 1,
                    int(row.get("plan_tier", 0) or 0), grade,
                    float(hole) if isinstance(hole, (int, float)) else float("inf"))
        ok = [r for r in rows
              if not any(str(r["status"]).startswith(t) for t in retry_status)]
        for row in rows:
            row["picked"] = ""
        for row in sorted(ok, key=pick_key)[:int(args.pick_budget)]:
            row["picked"] = "1"
    picked_rows = [r for r in rows if r.get("picked")]

    # 상한표: hole 사다리 맨 윗단이 그 preset·anchor 의 크기 상한이다.
    cap_hole = args.cap_hole if args.cap_hole else max(ladder)
    caps = {}
    for row in rows:
        # excess 모드에서 `target_hole` 은 anchor 마다 다르므로 사다리 눈금(Δ)으로 고른다.
        # absolute 모드에서는 둘이 같은 값이라 예전과 동일하다.
        # D78. 정지 preset 은 rung 이 사다리 맨 아랫단 하나뿐이라 이 조건에 안 걸린다 —
        # 그러면 상한표에서 조용히 사라지므로 status 로 따로 넣는다 (그 1개가 곧 상한이다).
        if abs(row["hole_delta"] - cap_hole) < 1e-9 or row["status"] == "static":
            caps.setdefault(row["preset"], {})[row["anchor_id"]] = {
                "knob_kind": row["knob_kind"], "knob": row["knob"], "status": row["status"],
                "binding": row["binding"], "tau_max": row["tau_max"],
                "path_len_u": row["path_len_u"], "hole_fraction": row["hole_fraction"],
                "behind_frac": row.get("behind_frac"), "near_depth": row.get("near_depth"),
                "standoff": row.get("standoff"), "obb_clear": row.get("obb_clear"),
                "obb_slack": row.get("obb_slack"), "obb_node": row.get("obb_node"),
                "elev_abs_max": row.get("elev_abs_max"), "ground_clear": row.get("ground_clear"),
                "hole_static": row.get("hole_static"), "target_hole": row.get("target_hole")}

    bank = {"format": "lbm_hole_bank_v1", "video": args.video, "num_frames": num_frames,
            # `shape_mult` 열이 **poses 를 실제로 만든 배율**임을 뜻하는 표식. 이게 없는 뱅크는
            # `shape_limited` 행의 배율이 headroom 배 부풀어 있어서 `emit_bank` 가 못 되만든다
            # → `fit/bank/patch_bank_shape_mult.py` 로 고치고 이 표식을 남긴다.
            "shape_mult_semantics": "as_built",
            "source_bank": f"{args.tau_bank_dir}/bank.json",
            "hole_ladder": ladder, "cap_hole": cap_hole,
            "knob_range": KNOB_RANGE, "iterations": args.iterations,
            # D53. 사다리 눈금의 뜻과 손잡이 하한의 출처. `excess` 면 `hole_ladder` 는 Δ 고
            # 실제 목표는 행의 `target_hole`(= `hole_static` + Δ)이다.
            # D266. 사다리 눈금(`hole`/`shot_scale`)과 프레이밍 하한. 되만들기에 필요하다.
            "ladder_metric": args.ladder_metric,
            "min_subject_in_frame": (float(args.min_subject_in_frame)
                                     if min_in_frame > float("-inf") else None),
            "ladder_base": {"hole_mode": args.hole_mode,
                            "tau_floor_src": tau_floor_on,
                            "tau_floor_rule": (f"tau_start + {KNOB_RANGE['tau'][0]:g}"
                                               if tau_floor_on else "KNOB_RANGE['tau'][0]"),
                            "target_rule": ("hole_static + delta" if excess else "delta"),
                            # 두 모드 중 하나라도 켜지면 `fit_tau` 이분법을 첫 눈금 아래까지
                            # 다시 훑는다 — 안 하면 하한을 올려도 궤적이 정지로 남는다.
                            "tau_refine": bool(args.tau_refine)},
            # D168 ②. routing 층 + retry 판정. `enabled: false` 면 D167 뱅크와 같은 뜻이다.
            # 재시도 판정에 쓴 어휘를 그대로 싣는다 — 코퍼스 단계
            # (`vista4d_bank_to_dl3dv.py --drop_status/--drop_suspect`) 와 **같은 어휘**라,
            # 여기 값과 거기 값을 나란히 놓고 "왜 이 변이가 남았나"를 볼 수 있어야 한다.
            "fallback": {"enabled": fallback_on, "tiers": tiers if fallback_on else None,
                         "target": int(args.fallback_target),
                         "retry_status": sorted(retry_status),
                         "retry_suspect": sorted(retry_suspect),
                         "usable": sorted(f"{a}__{p}" for a, p in usable_variants)},
            # D188 ③ 씬당 예산. `budget: 0` 이면 안 골랐다는 뜻이고 `picked` 열은 전부 빈 칸이다.
            # `grade` 는 고른 행이 깨끗한 것(usable)인지 best-effort 인지 — 뱅크만 보고
            # "이 씬은 suspect 밖에 없었다"를 알 수 있어야 한다.
            "pick": {"budget": int(args.pick_budget),
                     "picked": [r["variant_id"] for r in picked_rows],
                     "tiers": [int(r.get("plan_tier", 0) or 0) for r in picked_rows],
                     # `track` 이 1순위 키다 (§pick_key). false 인데 이 anchor 에 track 행이
                     # 있었다면 그건 전부 `retry_status` 로 탈락했다는 뜻 = fallback 정상 작동.
                     "track": [r["preset"].startswith("track_") for r in picked_rows],
                     "grade": ["usable" if not (set(str(r.get("suspect", "") or "").split("|"))
                                                - {""}) & set(retry_suspect) else "best_effort"
                               for r in picked_rows]},
            "measure": {"bisect_frames": args.bisect_frames, "verify_frames": args.verify_frames,
                        "height": args.tile_height, "width": args.tile_width,
                        "center_box": args.center_box,
                        # D119. 캡션이 shot scale 시간축을 쓸 수 있는 뱅크인지 여기로 판별한다.
                        "area_timeline": bool(args.area_timeline),
                        # D121. 〃 composition 절. 임계를 같이 남긴다 — `in_frame_ids` 의 뜻이
                        # 이 숫자에 통째로 달려 있어서, 없으면 두 뱅크를 비교할 수 없다.
                        "composition": bool(args.composition),
                        "composition_min_area": float(args.composition_min_area),
                        "composition_max_nodes": int(args.composition_max_nodes)},
            # 이분법이 푼 게 "hole 예산 안의 최대"인지 "충돌까지 없는 최대"인지 — 뱅크를
            # 읽는 쪽이 이걸 봐야 status 의 `collision_limited` 를 해석할 수 있다.
            "collision": ({"gate": "G1_behind", "max_behind_frac": max_behind,
                           # 채널별 예산 (2026-09-02). `time_match` 가 꺼져 있으면 정적 열이
                           # 아예 없어서 이 값은 안 쓰인다 — 그래도 뱅크만 보고 재현할 수 있게 싣는다.
                           "max_behind_frac_dyn": max_behind_dyn,
                           "src_frames": behind["frames"],
                           "margin_frac": behind["margin_frac"],
                           "clear_frac": behind["clear_frac"],
                           # D123. `clear_frac` 이 이제 유도량일 수 있어서, 그 유도에 쓴 재료를
                           # 같이 남기지 않으면 뱅크만 보고 재현이 안 된다.
                           "min_zcam_frac": behind["min_zcam_frac"],
                           "clear_src_ratio": float(args.behind_clear_src_ratio),
                           "clear_src_pct": float(args.behind_clear_src_pct),
                           "source_g1_clear": src_g1,
                           "radius_px": behind["radius_px"],
                           # D115: 이게 없으면 뱅크만 보고 두 arm 을 구분할 수 없다 —
                           # 진단 4열의 유무로 역추정해야 했다 (2026-09-02 A/B 에서 실제로 겪음).
                           "time_match": bool(behind.get("time_match", False)),
                           # D116: 증거를 depth shell 에서 잰 건지 `.blend` 부피에서 잰 건지.
                           # 열 이름이 같아서(`behind_*`) 이게 없으면 사후에 구분이 안 된다.
                           "source": str(args.collision_source),
                           "mesh_grid": (mesh_grid or None),
                           # D50/D57: standoff·near_depth 는 판정이 아니라 **열**이다. 뱅크를
                           # 읽는 쪽이 열의 의미를 오해하지 않게 소스 실측만 남긴다.
                           "standoff_column": {"gate": False, "source_standoff": src_standoff,
                                               "near_pct": NEAR_PCT},
                           # D49: 표면이 아니라 **부피**. 판정을 껐어도 열은 남으므로 `gate` 를 같이 남긴다.
                           # D51: 마진이 노드별이라 임계(0)가 아니라 **마진 표**를 남긴다.
                           "obb_gate": {"gate": obb_on, "measured": bool(args.measure_obb),
                                        "obb_clear_src_ratio": float(args.obb_clear_src_ratio),
                                        "obb_clear_floor": float(obb_floor),
                                        "node_margins": (obb_margins if obb_on else None),
                                        "source_obb_slack": src_obb,
                                        "source_obb_clear": src_obb_raw,
                                        "source_obb_node": src_obb_node},
                           # D55: 고도/지면. 임계가 **anchor 마다** 다르므로(소스 자신을 기준으로
                           # 잡는다) 여기엔 규칙만 남기고 실효값은 행의 `max_elev_deg` /
                           # `min_ground_clear` 열에 anchor 별로 들어간다.
                           "elev_gate": {"gate": elev_on,
                                         "max_elev_deg": float(args.max_elev_deg),
                                         "src_margin_deg": float(args.elev_src_margin_deg),
                                         "rule": "max(max_elev_deg, src_elev_abs_max + src_margin_deg)"},
                           "ground_gate": {"gate": ground_on,
                                           "min_ground_clear_ratio": float(args.min_ground_clear_ratio),
                                           "rule": "ratio * src_ground_clear"},
                           # D56: 시선축 전진 한계. G5 가 부호 없는 거리라 못 보는 "지나침".
                           # 임계는 anchor 별이라 규칙만 남기고 실효값은 행의 `min_approach` 에.
                           "approach_gate": {"gate": approach_on,
                                             "approach_src_ratio": float(args.approach_src_ratio),
                                             "rule": "ratio * src_approach (anchor node only)"}}
                          if behind else None),
            # `emit_bank.py` 가 이 블록만으로 `build_poses` 호출을 **정확히 재현**해야 한다.
            # `orbit_span_frac`/`min_sweep_deg`/`aim_ramp_frames` 를 빼먹으면 재현이 조용히
            # 어긋난다 (poses.npz 대조 assert 로 잡히긴 하지만 원인이 안 보인다).
            "fixed": {"speed": args.speed, "tracking": args.tracking,
                      # D93. 이 키가 없는 뱅크는 D93 이전 = `emit_bank` 가 False 로 되푼다.
                      "preset_tracking": bool(args.preset_tracking),
                      "look_at_bias": args.look_at_bias, "start_mode": args.start_mode,
                      "aim_anchor": args.aim_anchor,
                      "aim_ramp_frames": int(args.aim_ramp_frames),
                      "traj_basis": args.traj_basis,
                      # D71 keyframe 조준. 0 이면 예전(매 프레임 조준) 동작 그대로다.
                      "aim_keyframes": int(args.aim_keyframes),
                      "keyframe_aim": args.keyframe_aim,
                      "keyframe_ease": args.keyframe_ease,
                      # D90. `smooth_kf` 가지만 소비한다. ease 가 `smoothstep` 이던 동안은
                      # 빠져 있어도 아무 데도 안 나타났다.
                      "smooth_passes": int(args.smooth_passes),
                      "smooth_lambda": float(args.smooth_lambda),
                      # D97. **요청값**이다 (resolve 전). `build_poses` 가 preset 을 보고 다시
                      # resolve 하므로 재현에는 이것만 있으면 된다. 이 키가 없는 뱅크는
                      # D97 이전 = `emit_bank` 가 `SHAPE_DEFAULTS` 를 타 `"source"` 로 되푼다.
                      "tau_ref": str(args.tau_ref),
                      # D99. 중력 기준 roll 보정. 키가 없는 뱅크는 D99 이전 = 보정 없음.
                      "deroll": bool(args.deroll),
                      # F9. 이분법이 sweep 을 깎았는지(False) 반경을 깎았는지(True).
                      "orbit_fixed_sweep": bool(args.orbit_fixed_sweep),
                      # D171. follow 위치 채널 keyframe 보간. 0 = 저역통과(D171 이전 동작).
                      # 키가 없는 옛 뱅크는 `SHAPE_DEFAULTS` 가 같은 0 으로 되푼다.
                      "follow_keyframes": int(args.follow_keyframes),
                      "follow_kf_interp": str(args.follow_kf_interp),
                      # F7. 이 뱅크가 시간축 절단을 켜고 구워졌나. 되만들 때 실제로 읽는 건
                      # 행의 `hold_from` 이고 이건 표식이다 (키가 없으면 F7 이전 = False).
                      "time_truncate": bool(args.time_truncate),
                      "min_move_frac": float(args.min_move_frac),
                      "orbit_span_frac": float(args.orbit_span_frac),
                      "min_sweep_deg": float(args.min_sweep_deg),
                      # F2. `S` 가 어느 정의로 구워졌는가. 게이트 임계가 전부 S 배율이라 이 값이
                      # 다르면 같은 `behind_margin_frac` 이 다른 거리를 뜻한다.
                      "scale_mode": str(scale_mode),
                      # D150. τ 의 분모가 어느 정의인가. `scale_mode` 와 같은 이유로 남긴다 —
                      # `knob_range.tau` 와 `tau_floor_src` 하한이 이 게이지 위의 숫자다.
                      "tau_denom": str(args.tau_denom),
                      "num_frames": int(num_frames)},
            # 렌더러 K 를 frame0 에 고정했는지. `emit_bank` 가 렌더를 다시 하지는 않지만,
            # 하류 export 가 intrinsics 를 어디서 가져와야 하는지의 유일한 근거다.
            "fixed_focal": bool(args.fixed_focal),
            # 외부(DataDoP) 궤적을 썼다면 그 파일 경로. `emit_bank.py` 가 이걸 보고 **알아서**
            # 다시 등록한다 — 인자를 세 스크립트에 손으로 맞춰 주다 하나를 빠뜨리면 emit 에서
            # "모르는 preset" 으로 죽는다. None 이면 예전 뱅크와 완전히 같은 필드 구성이다.
            "external_shapes": ({"path": args.external_shapes, "aim": args.external_aim}
                                if args.external_shapes else None),
            # `z_med` 는 이름만 예전 것이고 실제로는 **τ 의 분모**다 (D150). 어느 정의인지는
            # `fixed.tau_denom` 이 말해준다.
            "S": float(graph["scale"]["S"]),
            "z_med": tau_denominator(graph, args.tau_denom),
            "tau_caps": caps, "variants": rows}
    with open(path.join(folder, "bank.json"), "w", encoding="utf-8") as file:
        json.dump(bank, file, ensure_ascii=False, indent=1)
    with open(path.join(folder, "tau_caps.json"), "w", encoding="utf-8") as file:
        json.dump({"format": "lbm_tau_caps_v1", "video": args.video, "cap_hole": cap_hole,
                   "caps": caps}, file, ensure_ascii=False, indent=1)

    columns = ["variant_id", "anchor_id", "anchor_label", "preset", "target_hole", "hole_delta",
               "hole_static", "knob_floor", "knob_kind", "tau_refine",
               "knob", "knob_raw", "status", "binding", "shape_mult",
               # D140. 게이트가 아니라 설정 탓으로 수치가 나쁜 행 (`|` 로 이은 태그, 빈 칸 = 정상).
               "suspect",
               # F7. 위치를 얼리기 시작한 프레임 (빈 칸 = 절단 없이 구운 뱅크).
               "hold_from", "speed", "tracking",
               "look_at_bias",
               "aim", "tau_ref", "tau_ref_max",
               "tau_max", "tau_start", "path_len_u", "radius_u", "view_angle_max_deg",
               "sweep_deg", "pan_deg", "hole_fraction", "hole_max", "subject_area_med",
               "subject_in_frame",
               # `--no_subject_visible` 이면 빈 칸 (D81).
               "subject_visible_frac", "subject_visible_min",
               "behind_frames", "behind_frac", "behind_worst_src",
               # `--collision_time_match` 일 때만 채워지는 열 (2026-09-02). `*_frac` 둘은
               # 진단이 아니라 **판정에 쓰는 값**이다 (`behind_over`).
               "behind_static_frames", "behind_static_frac", "behind_static_worst",
               "behind_dyn_frames", "behind_dyn_frac", "behind_dyn_worst",
               "near_depth", "standoff", "obb_clear", "obb_slack", "obb_node",
               "elev_max", "elev_min", "elev_abs_max", "ground_clear", "below_ground_frames",
               "src_elev_abs_max", "max_elev_deg", "src_ground_clear", "min_ground_clear",
               "approach_gap", "approach_frames", "past_frames", "src_approach", "min_approach",
               # D112. 판정으로 쓴 가림 하한 (빈 칸이면 측정만 한 예전 뱅크).
               "min_subject_visible",
               # D266. 사다리 눈금. 빈 칸 = hole 사다리 (예전 뱅크 전부).
               "ladder_metric", "area_static", "shot_dev",
               # D168 ②. 이 변이가 온 층 (0 = 본 슬롯, 1/2/3 = retry 로 열린 예비).
               "plan_tier",
               # D188 ③. 씬당 예산이 고른 행 (`--pick_budget`). 빈 칸 = 안 골랐거나 탈락.
               "picked",
               # D119. shot scale 시간축. `--no_area_timeline` 이면 빈 칸이고 예전 뱅크와 같다.
               # `subject_area_seq` 는 `measured_frames` 와 인덱스가 1:1 인 배열이다.
               "subject_area_start", "subject_area_end", "subject_area_seq",
               # D121. composition. **노드 id 리스트**다 (라벨이 아니라) — 문구는
               # `instance_desc.json` 이 갖고 있고, 그건 뱅크를 다시 안 굽고 바꿀 수 있어야 한다.
               # `--no_composition` 이면 빈 칸이고 예전 뱅크와 같다.
               "in_frame_ids", "enter_ids", "exit_ids"]
    if RAY_GATE is not None:          # D277. 게이트를 켰을 때만 열을 늘린다 — 끄면 스키마 비트 동일
        columns = list(columns) + ["ray_clear_frac", "ray_min_clearance",
                                   "ray_min_subject_dist", "ray_min_floor_drop"]
    write_start_screen(folder, start_screen_rows)   # [new 2026-09-27, R62] --start_screen 일 때만 파일이 생긴다
    with open(path.join(folder, "bank.csv"), "w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")
        for row in rows:
            fields = [str(row.get(c, "")) for c in columns]
            # 배열 열은 `|` 로 잇는다 — 이 writer 는 따옴표를 안 쓰므로 `[0.1, 0.2]` 를 그대로
            # 흘리면 아래 쉼표 assert 에 걸린다. JSON 쪽은 리스트 그대로 남는다.
            for i, name in enumerate(columns):
                if isinstance(row.get(name), list):
                    fields[i] = "|".join(str(v) for v in row[name])
            # 따옴표를 안 쓰는 writer 라 값에 쉼표가 들어가면 뒤 열이 통째로 밀린다. 조용히
            # 밀리면 `binding` 이 `tau_floor` 로 읽히는 식으로 **틀린 채로 파싱된다**.
            assert not any("," in f for f in fields), \
                f"쉼표가 든 값: {[(c, f) for c, f in zip(columns, fields) if ',' in f]}"
            file.write(",".join(fields) + "\n")

    np.savez_compressed(path.join(folder, "poses.npz"),
                        cam_c2w=np.stack(poses_all),
                        variant_id=np.array([r["variant_id"] for r in rows]),
                        anchor_id=np.array([r["anchor_id"] for r in rows]),
                        target_hole=np.array([r["target_hole"] for r in rows]),
                        hole_delta=np.array([r["hole_delta"] for r in rows]))

    # preset 별 상한표 — 이 스크립트의 요점이다.
    print(f"\npreset 별 크기 상한 (hole {cap_hole:g} 기준, anchor 평균)")
    print(f"{'preset':<18}{'knob':>7}{'knob 값':>12}{'tau_max':>9}{'path_u':>8}{'hole':>7}"
          f"{'G1':>6}{'stdof':>7}{'obb':>8}{'충돌상한':>9}{'n':>4}  status")
    for preset in presets:
        group = list(caps.get(preset, {}).values())
        if not group:
            continue
        mean = lambda key: float(np.mean([g[key] for g in group]))
        status = ",".join(sorted({g["status"] for g in group}))
        bound = sum(g["binding"] in ("collision", "clearance", "obb") for g in group)
        print(f"{preset:<18}{group[0]['knob_kind']:>7}{mean('knob'):>12.3f}{mean('tau_max'):>9.4f}"
              f"{mean('path_len_u'):>8.3f}{mean('hole_fraction'):>7.3f}"
              f"{mean('behind_frac') if behind else float('nan'):>6.2f}"
              f"{mean('standoff') if behind else float('nan'):>7.3f}"
              f"{mean('obb_slack') if args.measure_obb and behind else float('nan'):>8.3f}"
              f"{bound:>9}{len(group):>4}  {status}")

    if behind:
        # 무엇이 상한을 정했나. 이게 D46 옵션 1 의 요점이다 — hole 만 보던 이분법이 이제 벽에도
        # 걸린다. `collision` 이 많으면 그 preset 은 구멍이 아니라 기하가 막고 있는 것이다.
        collided = [r for r in rows if r["binding"] == "collision"]
        boxed = [r for r in rows if r["binding"] == "obb"]
        # "남은 위반" 은 판정식과 같은 함수로 세야 한다 — 채널이 갈리면 합집합 열과 답이 다르다.
        left = [r for r in rows if behind_over(r, max_behind, max_behind_dyn)]
        # standoff 는 판정이 아니라 열이다 (D57). "소스 카메라보다 더 붙었나"만 세어 둔다 —
        # 임계가 아니라 참고선이라 `남은 위반` 이 아니라 별도 줄로 뺀다.
        tight = [r for r in rows if r.get("standoff", float("inf")) < src_standoff]
        # 판정을 껐어도 **관통 수**는 센다 — 그게 D49 를 켤지 말지의 근거다.
        inside = [r for r in rows if args.measure_obb
                  and r.get("obb_slack", float("nan")) < (min_obb if obb_on else 0.0)]
        # D55. 지면 관통은 판정을 껐어도 센다 — G1 이 구조적으로 못 잡는 위반이라 "0 이 아니다"
        # 자체가 게이트를 켤 근거다.
        steep = [r for r in rows if r["binding"] == "elev"]
        sunk = [r for r in rows if r["binding"] == "ground"]
        under = [r for r in rows if r.get("below_ground_frames", 0)]
        # D56. 근접면 침범과 **뒷면 통과**를 따로 센다 — 후자가 사용자가 말한 "지나침"이다.
        crossed = [r for r in rows if r["binding"] == "approach"]
        passed = [r for r in rows if r.get("past_frames", 0)]
        poked = [r for r in rows if r.get("approach_frames", 0)]
        print(f"\n{'binding':<14}collision {len(collided)}   "
              f"obb {len(boxed)}   elev {len(steep)}   ground {len(sunk)}   "
              f"approach {len(crossed)}   "
              f"hole {sum(r['binding'] == 'hole' for r in rows)}   / {len(rows)}")
        print(f"{'남은 위반':<14}G1 {len(left)}   obb {len(inside)}   "
              f"지면아래 {len(under)}   근접면침범 {len(poked)}   뒷면통과 {len(passed)}   "
              f"/ {len(rows)}   (이분법이 못 막은 것 — 손잡이 하한에서도 뚫리는 경우)")
        print(f"{'standoff 열':<14}소스({src_standoff:.4f}·S)보다 붙은 변이 {len(tight)} / {len(rows)}"
              f"   (판정 아님 — D57)")
        if passed:
            far = max(passed, key=lambda r: -r.get("approach_gap", float("inf")))
            print(f"{'':<14}최악 전진 여유 {far.get('approach_gap', float('nan')):+.4f} u  "
                  f"{far['variant_id']}  ({far.get('past_frames', 0)} 프레임 뒷면 통과)")
        if under:
            deep = min(under, key=lambda r: r.get("ground_clear", float("inf")))
            print(f"{'':<14}최악 지면 여유 {deep.get('ground_clear', float('nan')):+.4f} u  "
                  f"{deep['variant_id']}  ({deep.get('below_ground_frames', 0)} 프레임, "
                  f"고도각 {deep.get('elev_min', float('nan')):+.1f}°)")
        if args.measure_obb and inside:
            worst = min(inside, key=lambda r: r["obb_slack"])
            hit = {}
            for row in inside:
                hit[row.get("obb_node", "")] = hit.get(row.get("obb_node", ""), 0) + 1
            print(f"{'':<14}최악 slack {worst['obb_slack']:+.4f} u  {worst['variant_id']} "
                  f"({worst.get('obb_node', '')})   노드별 "
                  + "  ".join(f"{k} {v}" for k, v in sorted(hit.items(), key=lambda x: -x[1])))
        # 충돌 천장이 hole 예산보다 낮으면 그 위 단들은 **같은 궤적**이 된다 — 사다리가 접힌다.
        seen_knob, folded = {}, 0
        for row in rows:
            knobs = seen_knob.setdefault((row["anchor_id"], row["preset"]), set())
            folded += float(row["knob"]) in knobs
            knobs.add(float(row["knob"]))
        print(f"{'접힌 단':<14}{folded} / {len(rows)}   "
              f"(충돌 천장이 hole 예산보다 낮아 윗단이 같은 궤적)")

    # D53. 사다리 바닥 진단. `정지 카메라` 가 0 이 아니면 아직 궤적이 사라지고 있는 것이다.
    frozen = [r for r in rows if r["path_len_u"] < 1e-6 and r["preset"] not in ROTATION_ONLY_PRESETS]
    floored = [r for r in rows if r["status"].endswith("tau_floor")]
    # 정지 hole 이 목표보다 높은 단 — absolute 모드면 "어떤 손잡이로도 못 맞추는 단"이고,
    # excess 모드면 정의상 0 이어야 한다 (목표 = 정지 hole + Δ).
    unreachable = [r for r in rows if r["hole_static"] > r["target_hole"] + 1e-9]
    print(f"{'사다리 바닥':<14}정지 카메라 {len(frozen)}   tau 하한 {len(floored)}   "
          f"목표<정지hole {len(unreachable)}   / {len(rows)}")
    if excess or tau_floor_on:               # 안 재는 모드에서는 `hole_static` 이 0 이라 안 찍는다
        static_by_anchor = {}
        for row in rows:
            static_by_anchor.setdefault(row["anchor_id"], set()).add(row["hole_static"])
        worst = sorted(((max(v), k) for k, v in static_by_anchor.items()), reverse=True)[:3]
        print(f"{'':<14}정지 hole 최악 anchor  "
              + "   ".join(f"{k} {v:.3f}" for v, k in worst))

    # D140. suspect 태그 요약. 게이트에서 걸린 게 아니라 **설정 탓**으로 나쁜 행들이라
    # `status` 분포에는 안 보인다 — 여기서만 보인다.
    if args.suspect:
        flagged = sum(1 for r in rows if r["suspect"])
        print(f"{'suspect':<14}{flagged} / {len(rows)}   "
              + "   ".join(f"{k} {v}" for k, v in suspect_counts.items() if v))

    # `calls_total` 은 이분법 **호출** 수 + 최종 검증 프레임 수다. 그중 `gated_probes` 만큼은
    # 물리 게이트에서 잘려 실제 래스터가 0회였다 — 이 숫자가 안 보이면 D113 이 켜졌는지도 모른다.
    # D188 ③. 씬당 예산이 무엇을 골랐나. 이게 곧 `emit_bank --picked_only` 가 낼 카메라다.
    if args.pick_budget:
        for row in picked_rows:
            tags = set(str(row.get("suspect", "") or "").split("|")) - {""}
            print(f"{'pick':<14}{row['variant_id']}   tier {row.get('plan_tier')}   "
                  f"hole {row.get('hole_fraction')}   "
                  f"{'best_effort(' + '|'.join(sorted(tags & set(retry_suspect))) + ')' if tags & set(retry_suspect) else 'usable'}")
        if not picked_rows:
            print(f"{'pick':<14}0 / 예산 {args.pick_budget} — status 를 통과한 행이 없다")

    gated = (f"   게이트 선차단 {gated_probes[0]:,}" if args.gate_before_render else "")
    print(f"\n변이 {len(rows)}   렌더 {calls_total:,}{gated}   -> {folder}")

    # D165. preset 별 소요시간. 안 주면 아무것도 안 찍고 안 쓴다 (기존 출력 비트 동일).
    if args.timing_json:
        makedirs(path.dirname(path.abspath(args.timing_json)), exist_ok=True)
        with open(args.timing_json, "w", encoding="utf-8") as file:
            json.dump({"format": "fit_hole_ladder_timing_v1", "video": args.video,
                       "stage": "fit", "anchors": len(anchors), "presets": len(presets),
                       "total_seconds": round(sum(t["seconds"] for t in timing), 2),
                       "entries": timing}, file, ensure_ascii=False, indent=1)
        agg = {}
        for t in timing:
            a = agg.setdefault(t["preset"], {"n": 0, "sec": 0.0, "calls": 0})
            a["n"] += 1; a["sec"] += t["seconds"]; a["calls"] += t["solve_calls"]
        print(f"\npreset 별 소요 (fit, anchor 합산)")
        print(f"{'preset':<26}{'n':>4}{'sec':>9}{'sec/anchor':>12}{'calls':>8}{'sec/call':>10}")
        for p, a in sorted(agg.items(), key=lambda kv: -kv[1]["sec"]):
            print(f"{p:<26}{a['n']:>4}{a['sec']:>9.1f}{a['sec'] / a['n']:>12.2f}"
                  f"{a['calls']:>8}{a['sec'] / max(a['calls'], 1):>10.3f}")
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
    # 출력 폴더 이름. `emit_bank.py --bank_dir` 과 같은 값을 쓴다. 예전엔 여기가 하드코딩이라
    # 변종 뱅크를 만들려면 사후에 손으로 이름을 바꿔야 했다 (hole_bank_r90 등이 그렇게 생겼다).
    parser.add_argument("--bank_dir", default="hole_bank", type=str)
    # D77. 읽어들일 τ 뱅크 폴더 (`sample_camera_bank.py --bank_dir` 과 같은 값). 기본 "bank" 가
    # 예전 하드코딩과 동일하다 — 새 preset 축을 넣은 τ 뱅크를 옆에 두려고만 노출한다.
    parser.add_argument("--tau_bank_dir", default="bank", type=str)
    # D79. τ 뱅크를 만들 때 준 것과 **같은 파일**이어야 한다 — 뱅크 행의 preset 이름으로 궤적을
    # 다시 짓기 때문에, 빼먹으면 `모르는 preset: dd_*` 로 죽는다.
    parser.add_argument("--external_shapes", default=None, type=str)
    parser.add_argument("--external_aim", default="traj", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    # D165. preset 별 소요시간을 JSON 으로. 안 주면 재기는 하되 아무 데도 안 쓴다 (출력 동일).
    parser.add_argument("--timing_json", default=None, type=str)

    # ── 사다리
    parser.add_argument("--hole_ladder", nargs="*", default=list(HOLE_LADDER), type=float)
    parser.add_argument("--cap_hole", default=0.0, type=float)   # 0 = 사다리 맨 윗단
    # D53 ②. `excess` 면 사다리 값이 anchor 정지 hole 위에 얹는 **초과분**이다. avocado `stat_1`
    # 은 정지만으로 hole 0.5895 라 absolute 사다리(0.10~0.50)는 4단 전부 도달 불가능했다.
    parser.add_argument("--hole_mode", default="excess", choices=("absolute", "excess"))
    # D266. 사다리 눈금의 정체. 기본 `hole` = 예전 동작 그대로.
    #   `shot_scale` — Δ 는 **정지(손잡이 0) 대비 subject 화면면적 배율의 로그 절대값**이다.
    #                  Δ 0.2/0.4/0.7/1.1 ≈ ×1.22/×1.49/×2.0/×3.0. hole 은 열로만 남는다.
    parser.add_argument("--ladder_metric", default="hole", choices=("hole", "shot_scale"))
    # D275. shot_scale 사다리에서 빼는 preset 군. targetless = 캡션 `targetless` 플래그
    # (pan/tilt/roll/free_pedestal) → 이동량 사다리 `--motion_ladder` (τ; 회전은 ×60°). none = 옛 동작.
    parser.add_argument("--shot_exempt", default="none", choices=("none", "targetless"))
    # D277. fitting 중 GT mesh raycast 게이트 (Blender 상주 서버). off = 옛 동작.
    parser.add_argument("--ray_gate", default="off", choices=("off", "server"))
    parser.add_argument("--ray_min_clearance", default=0.20, type=float)
    parser.add_argument("--ray_min_floor_drop", default=0.30, type=float)
    parser.add_argument("--ray_min_subject_dist", default=0.80, type=float)
    parser.add_argument("--ray_min_clear_frac", default=0.90, type=float)
    parser.add_argument("--motion_ladder", default=[0.20, 0.35, 0.60], nargs="+", type=float)
    # D266. subject 중심이 중앙 박스(`--center_box`) 안이던 프레임 비율의 하한. 0 = 끔(기본,
    # 예전과 비트 동일). 가림(`--min_subject_visible`)과 달리 렌더가 안 늘어난다.
    parser.add_argument("--min_subject_in_frame", default=0.0, type=float)
    # D53 ①. τ 손잡이 하한을 `tau_start`(소스 자신의 시차, 씬 상수) 위로 올린다. 끄면 하한이
    # 0.02 로 고정돼 avocado(tau_start 0.1286)에서 궤적이 통째로 0 이 되던 예전 동작이다.
    parser.add_argument("--tau_floor_src", action="store_true", default=True)
    parser.add_argument("--no_tau_floor_src", dest="tau_floor_src", action="store_false")
    parser.add_argument("--anchors", nargs="*", default=None)    # None = τ 뱅크의 anchor 전량
    parser.add_argument("--presets", nargs="*", default=None)    # None = τ 뱅크 preset 전량
    # D78. 정지 preset(손잡이 없음)을 rung 1개로 통과시킨다. 끄면 예전처럼 통째로 빠진다.
    parser.add_argument("--static_rung", action="store_true", default=True)
    parser.add_argument("--no_static_rung", dest="static_rung", action="store_false")
    parser.add_argument("--iterations", default=4, type=int)     # bracket 이 이미 좁다
    # `fit_tau` 가 max_scale 에 붙으면 모양을 이 배율로 키워 다시 푼다 (docstring 의 천장 문제).
    parser.add_argument("--shape_headroom", default=2.0, type=float)
    parser.add_argument("--shape_doublings", default=4, type=int)

    # ── 충돌 (G1: 카메라가 관측된 표면 뒤인가). 렌더가 안 늘어 기본으로 켠다 — 끄면 hole 만 본다.
    parser.add_argument("--collision_free", action="store_true", default=True)
    parser.add_argument("--no_collision_free", dest="collision_free", action="store_false")
    # 궤적 49프레임 중 표면 뒤가 허용되는 비율. 0 = 한 프레임도 안 된다.
    parser.add_argument("--max_behind_frac", default=0.0, type=float)
    # F11 (2026-09-02, 13 -> 49). 증거로 쓰는 소스 프레임 수. 13 은 균등 서브샘플이라 그 사이
    # 프레임에만 있던 표면이 증거에서 빠졌다 — parkour 실측 39건(5.9%)의 binding 이 hole/approach
    # 에서 **collision 으로 옮겨간다**(= 원래 뚫고 있었는데 못 잡던 것). 렌더 수는 안 늘어난다
    # (G1 은 numpy 투영이라 사실상 공짜). snowboard 는 0건 = 소스 카메라가 정지에 가까운 씬에선
    # 13 으로도 충분했다는 뜻이고, 그래서 이 값은 씬에 따라 조용히 새는 종류였다.
    parser.add_argument("--behind_src_frames", default=49, type=int)
    parser.add_argument("--behind_margin_frac", default=0.02, type=float)  # 관통을 봐주는 여유
    # 표면 **앞**에 요구하는 여유. 0 이면 표면에 붙어 스쳐도 통과한다 — 실제로 그렇게 보였다.
    # `--behind_clear_src_ratio > 0` 이면 이 값은 안 쓰이고 소스 실측에서 유도된다.
    parser.add_argument("--behind_clear_frac", default=0.10, type=float)
    # D123 ①. degenerate 투영 하한 (S 비율). 플랜 위치가 소스 카메라 t 의 광학 중심에 겹치면
    # `uv = (K·cam)/cam[2]` 가 0 에 가까운 수로 나뉘어 투영 픽셀이 임의로 튀고, 판정식이
    # `clear > z_surf + margin` 으로 붕괴해 **소스 카메라 t 자기 주변**에 대한 진술이 된다.
    # parkour 배포 뱅크 실측: G1 히트 9,888 중 12.2%(0.005) / 31.9%(0.02) 가 이 축에 있다.
    # 0.02 = `behind_margin_frac` 과 같은 눈금이라 자기정합적이다. 0.0 이 예전 동작.
    parser.add_argument("--behind_min_zcam", default=0.02, type=float)
    # D123 ②. G1 임계를 **소스 카메라 자신의 G1 여유**의 β 배로 잡는다 (G5 `--obb_clear_src_ratio`
    # / G7 `--approach_src_ratio` 와 같은 형태). 게이트 5종 중 G1 만 절대 분수라 D47 legality
    # ("소스 카메라를 기각하는 임계는 버그다")가 안 걸려 있었고, 18편 중 parkour 가 실제로
    # 배포 0.10 에서 소스를 기각한다. `clear_frac = margin_frac + β·g1_src(pct)`.
    # 0.0 이면 `--behind_clear_frac` 절대값 그대로 = 예전 동작.
    #
    # **D128 에서 기본을 0.3 -> 0.0 으로 되돌렸다 (2026-09-05).** 두 가지가 겹쳤다.
    #  ① 사용자 판단으로 이 축의 채택은 **보류** 상태다 (게이트: `probe_g1_reference.py` 에
    #     `min_zcam` 을 넣고 52편에서 `g1_min` 을 다시 재는 것). 그런데 기본값이 0.3 이라
    #     d128 뱅크가 선언하지 않은 7번째 축을 조용히 물고 구워지고 있었다 — D105 가 말하는
    #     바로 그 사고(뱅크 정체성을 argparse 기본값에 맡기기)다. d121 manifest 에는
    #     `clear_src_ratio` 키 자체가 없다.
    #  ② `--behind_min_zcam 0.02` 와 같이 켜면 **터진다**. `source_g1_clear` 는 소스 pose 를
    #     다른 소스 프레임에 투영해 여유를 재는데, 소스 카메라가 거의 lateral 로만 움직이는
    #     클립은 그 쌍의 `z_cam` 이 전부 `0.02·S` 아래라 증언이 한 개도 안 남는다
    #     (avocado-slice 실측: `AssertionError: 소스 카메라 전부에서 G1 증거가 안 잡혔다`).
    #     즉 이 두 손잡이는 현재 코드에서 같이 못 켠다. ① 이 풀릴 때 같이 고칠 것.
    parser.add_argument("--behind_clear_src_ratio", default=0.0, type=float)
    # 바닥으로 쓸 분위수. min 은 못 쓴다 — 18편 CV 가 p10 0.371 로 최저 (probe_g1_reference.py).
    parser.add_argument("--behind_clear_src_pct", default=10.0, type=float)
    # D120. mesh 판 전용 임계(씬 스케일 비율). depth 판의 실효 standoff 는
    # `(behind_clear_frac − behind_margin_frac)·S` = 0.08·S 인데 mesh 판은 `clear_frac` 을
    # 안 받아 0.02·S 로 4배 느슨했다. 0.08 이면 두 판이 같은 거리에서 걸린다 (TRUMANS S 중앙값
    # 1.81 기준 0.145 m). 예전 뱅크 재현은 `--mesh_margin_frac 0.02`.
    parser.add_argument("--mesh_margin_frac", default=0.08, type=float)
    # D207. 벽에 붙어 찍은 chunk 는 소스 카메라 자신의 여유가 위 임계보다 작아 D47 assert 로
    # chunk 째 죽는다 (TRUMANS 191 편 중 27 편). 값을 주면 **그런 chunk 에서만** 임계를
    # `값 x 소스 최소여유` 로 내린다. 이미 합법인 chunk 는 건드리지 않으므로 0.0(기본)이든
    # 0.9 든 나머지 chunk 는 비트 동일이다. 느슨해진 chunk 는 stdout 에 `autoclamp` 로 찍힌다.
    parser.add_argument("--mesh_margin_autoclamp", default=0.0, type=float)
    # 투영 픽셀 하나만 보면 얇은 물체 가장자리를 스칠 때 옆 배경 depth 로 판정돼 안 걸린다.
    parser.add_argument("--behind_radius_px", default=2, type=int)
    # G1 시간축 정합 (2026-09-02). 켜면 `t != plan_frame` 인 소스 프레임의 **동적 픽셀**을 증거에서
    # 뺀다 — 정적 표면은 시간 불변이라 아무 프레임이나 증거지만 동적 표면은 그 시각에만 거기 있었다.
    # **D116 부터 기본 True** (사용자 지시 "일단 켜줘"). 순수하게 느슨해지는 방향이고
    # (parkour 4/664, snowboard 6/196 변이 변화, 증가 0건) 채널별 예산을 나눌 수 있게 된다.
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
    # 동적 채널 전용 예산 (사용자 지시 2026-09-02 "collision 을 static 도 하고 dynamic 은 따로
    # 해서 양쪽 다 판정"). `--collision_time_match` 일 때만 의미가 있다. None 이면
    # `--max_behind_frac` 과 같아서 두 채널 OR = 합집합, 즉 예전 판정 그대로다.
    # 예: 벽은 한 프레임도 허용 안 하되(0.0) 지나가는 사람은 조금 봐준다(--max_behind_frac_dyn 0.1).
    parser.add_argument("--max_behind_frac_dyn", default=None, type=float)
    # (`--min_standoff_ratio` / `--min_standoff` / `--min_near_depth` / `--near_pct` 는
    #  D57 에서 삭제. D50 이 standoff 판정을 껐고 D47 이 near_depth 를 기각한 뒤 한 번도 안 켰다.
    #  두 열은 계속 측정되고 bank 에 남는다 — 사라진 건 판정과 손잡이뿐이다.)
    # ── 물체 부피 (G5, D49). G1/standoff 는 **표면**만 봐서 물체 안을 지나가는 걸 놓친다.
    # `--no_measure_obb` 면 열까지 사라지고 예전 출력 그대로다. `--no_obb_gate` 는 열만 남기고
    # 판정에서 뺀다 (두 뱅크를 같은 열로 비교하고 싶을 때).
    parser.add_argument("--measure_obb", action="store_true", default=True)
    parser.add_argument("--no_measure_obb", dest="measure_obb", action="store_false")
    parser.add_argument("--obb_gate", action="store_true", default=True)
    parser.add_argument("--no_obb_gate", dest="obb_gate", action="store_false")
    # 마진은 **소스 대비 배수** 하나다 (D51): m = β × (소스 카메라 자신의 최소 OBB 거리).
    # β<1 이면 소스 카메라가 정의상 통과한다 — 절대 마진이 계속 원본 촬영을 기각했던 사고를
    # 구조적으로 막는다 (D47 이 standoff 에서 쓴 것과 같은 꼴).
    # β=0.3 인 이유: 아무것도 안 알려줬는데 camel 마진 0.153 이 나와, 사용자가 영상 보고 직접
    # 고른 0.15 를 path 합 10.35 / obb binding 14 까지 그대로 재현한다. avocado 는 0.035 로
    # 직접 고른 0.06 보다 느슨해서 움직임이 더 산다 (path 2.25 -> 6.17).
    # (크기 비례 `--obb_clear_ratio` · 상한 `--obb_clear_cap` · 절대 바닥 `--min_obb_clear` 는
    #  D57 에서 삭제. 셋 다 기본값에서 no-op 였고 비례항은 D51 이 실측으로 기각했다.)
    parser.add_argument("--obb_clear_src_ratio", default=0.3, type=float)
    # [new 2026-09-27, R62] 높이 < R x 수평 최대 extent 이고 수평 extent >= min 인 노드를 G5 에서 뺀다.
    # 0 = 끔 (기본, 기존 동작). 지면으로 분할된 노드(예 golf "golf" 코스 2.34x1.76x0.11 u) 대응.
    parser.add_argument("--obb_skip_flat", default=0.0, type=float)
    parser.add_argument("--obb_skip_flat_min_extent", default=1.0, type=float)
    # ── 고도 / 지면 (G6, D55). 위 게이트들이 **전부** 통과시키는 구멍이다. 수직 이동은 hole 을
    # 거의 안 늘리므로(바닥에도 천장에도 점이 있다) shape 배증이 상한 16배까지 다 돌아
    # `lateral_frac 0.35 × 16 = 5.6` → `atan(5.6) = 80°`. 실측: `rise_reveal` 고도각 p95 가
    # camel 89.83° / avocado 85.27°, `drop_reveal` p05 가 −83.99° / −84.58°. 물체 **바로 위/아래**다.
    # 아래쪽은 지면 관통까지 간다 (camel 14 변이 / avocado 5). G1 이 못 잡는 이유는 구조적이다 —
    # 바닥 밑 카메라는 소스 뷰에 **화면 밖**으로 투영돼 아예 채점이 안 된다 (behind_frac 전부 0.0000).
    # 가림으로도 커버 안 된다: 바닥은 노드가 아니라 G5/OBB 가리기 지표가 구조적으로 못 본다
    # (avocado `stat_0__drop_reveal__hole0.5` 는 바닥 아래 0.66 u 인데 obb_occl_pass 0.984).
    parser.add_argument("--elev_gate", action="store_true", default=True)
    parser.add_argument("--no_elev_gate", dest="elev_gate", action="store_false")
    # 절대 상한. 45° 인 이유는 후보 풀의 elevation 열거 상한과 같기 때문이다 — 시작 pose 에서
    # 허용하는 고도를 궤적이 넘어갈 이유가 없다. 소스 카메라는 −6.32°~+35.09° 라 아무것도 안 걸린다.
    parser.add_argument("--max_elev_deg", default=45.0, type=float)
    # 다만 절대값 하나로는 못 덮는다 — avocado `stat_4` chair 는 **소스 자신이** 35.09° 로 이미
    # 내려다보는 앵커다 (코앞 0.033 u). 실효 상한 = max(절대값, 소스 자신 + 이 여유).
    # 소스를 기각하는 임계는 버그라는 규칙(D47/D51)이 여기에도 그대로 걸린다.
    parser.add_argument("--elev_src_margin_deg", default=10.0, type=float)
    parser.add_argument("--ground_gate", action="store_true", default=True)
    parser.add_argument("--no_ground_gate", dest="ground_gate", action="store_false")
    # 지면 위 여유의 하한. **비율**인 이유는 소스 카메라 높이가 씬마다 camel 0.1069 u /
    # avocado 0.3641 u 로 3.4배 벌어져서 절대값 하나가 한쪽을 반드시 기각하기 때문이다
    # (두 씬 모두 49프레임 내내 높이가 ±0.0005 u 로 일정하다 — 드론이 아니라 고정 높이 촬영).
    # 0.2 인 이유는 실측이다 (camel dyn_0, hole 0.35). ratio 를 0/0.2/0.5 로 스윕하면
    #   drop_reveal   path 0.106 → 0.059 → 0.014 u,  실측 지면여유 0.003 → 0.050 → 0.094 u
    #   pedestal_down path 0.092 → 0.082 → 0.051 u
    # 0.5 는 drop 을 τ 하한(0.019)까지 눌러 **정지 클립**을 만들고(D53 과 같은 실패),
    # 0.0 은 바닥을 0.003 u 로 스친다. 0.2 만 실제 하강(path 0.059)과 여유(0.050)를 동시에 남긴다.
    # avocado 는 카메라가 0.364 u 로 높아 세 값 전부 같은 답을 낸다 — 이 축을 정하는 건 camel 이다.
    parser.add_argument("--min_ground_clear_ratio", default=0.2, type=float)
    # ── 전진 한계 (G7, D56). 사용자 지적: "앞 뒤로 움직이는 것도 bbox 를 지나치기 전까지 적당한
    # 거리까지만". G5 는 **부호 없는 거리**라 박스 옆을 안전거리로 스쳐 지나가는 궤적을 통과시킨다
    # — camel `dyn_0 push_in_arc` 4단 전부 근접면을 0.1972 u 뚫고 뒷면까지 0.0467 u 나갔는데
    # `obb_clear` 는 0.1536 이상이었다. avocado 는 dolly 84 중 16 침범 / 7 뒷면 통과.
    parser.add_argument("--approach_gate", action="store_true", default=True)
    parser.add_argument("--no_approach_gate", dest="approach_gate", action="store_false")
    # β. `--obb_clear_src_ratio` 와 **같은 값**을 기본으로 둔다 — 소스 실측(camel 0.4950 /
    # avocado 0.1097 u)의 비율 4.5배가 raw OBB 여유의 4.4배와 거의 같아서, 같은 β 를 쓰면 두
    # 게이트의 금지구역이 자동으로 같은 눈금에 놓인다 (0.3 → camel 0.149 / avocado 0.033,
    # 실제 G5 마진 0.153 / 0.035 와 일치). β<1 이라 소스 카메라는 정의상 통과한다.
    parser.add_argument("--approach_src_ratio", default=0.3, type=float)
    # ── 게이트 순서 (D113). 물리 게이트 다섯(G1 충돌 / G5 obb / G6 ground·elev / G7 approach)은
    # **렌더가 0회**다 — 재투영과 3×3 곱 몇 번이 전부다. 반면 hole 은 점군 래스터라 수백 배 비싸다.
    # 그런데 `over()` 의 if/elif 체인에서 hole 은 이 다섯 **뒤**에 있으므로, 물리 게이트가 하나라도
    # 걸린 knob 의 hole 값은 **애초에 소비되지 않는다**. 그래서 켜면 렌더를 건너뛰어도 결과가 같다.
    # 안전 근거: 최종 CSV 행은 `verify_frames` 로 다시 재고(:690 부근), 이분법 캐시에서 읽는 건
    # `poses`/`info`/`mult` 뿐이다. 유일한 예외가 사다리 기준점 `hole_static = probe(0.0)` 이라
    # 거기만 `force_render=True` 로 강제한다. 끄면 D113 이전과 완전히 동일하게 동작한다.
    parser.add_argument("--gate_before_render", action="store_true", default=True)
    parser.add_argument("--no_gate_before_render", dest="gate_before_render",
                        action="store_false")
    # ── 시간축 절단 (F7, fix.log 우선순위 1). 사용자 지시 2026-09-04.
    # 지금 `solve_knob` 은 게이트에 걸리면 **크기**만 줄인다 — 전 구간을 균일하게 깎으므로
    # 마지막 1프레임이 벽에 닿으면 48프레임이 멀쩡해도 궤적 전체가 정지로 눌린다. 켜면
    # 크기를 줄이기 **전에** "걸리는 프레임 앞까지만 움직이고 그 뒤는 위치를 얼린다"를
    # 먼저 시도한다 (`solve_hold_from`, 렌더 0회). 회전은 안 얼리므로 조준·추종은 계속된다.
    # **기본 False** — 켜면 뱅크의 궤적이 바뀌므로 기존 재현이 깨진다. 켜고 구운 뱅크는
    # 행의 `hold_from` 열로 `emit_bank` 가 같은 절단을 되푼다.
    parser.add_argument("--time_truncate", action="store_true", default=False)
    parser.add_argument("--no_time_truncate", dest="time_truncate", action="store_false")
    # 절단 후에도 남아야 하는 **이동 구간 비율**. 0.5 = 최소 절반은 움직여야 한다. 이 아래로
    # 잘라야 통과하는 궤적은 절단을 포기하고 예전대로 크기를 줄인다 — 안 그러면 "frame 2 에서
    # 얼어붙은 49프레임"이 나오는데, 그건 정지 궤적을 다른 이름으로 부르는 것뿐이다.
    parser.add_argument("--min_move_frac", default=0.5, type=float)

    # ── 궤적 (여기서는 고정한다 — 축이 이미 3개다)
    parser.add_argument("--speed", default="steady", type=str)
    parser.add_argument("--tracking", default="lock", type=str,      # D127: drift 삭제
                        choices=["world", "lock"])
    # D93 -> D127. `aim="look_at"` preset 은 조준 추종률을 `lock` 으로. 끄면 요청값 그대로.
    parser.add_argument("--preset_tracking", action="store_true", default=True)
    parser.add_argument("--no_preset_tracking", dest="preset_tracking", action="store_false")
    parser.add_argument("--look_at_bias", default=0.0, type=float)
    # D171. follow 위치 채널을 균등 keyframe N개로 줄였다가 다시 채운다. 0 = 끔 = D171 이전
    # 동작(`--follow_smooth` savgol 저역통과). 기본값을 0 으로 두는 이유는 `build_poses` 쪽
    # 주석과 같다 — 아직 육안 대조(D171-c) 전이고, 궤적 모양을 바꾸는 손잡이는 굽기 도중에
    # 기본이 바뀌면 코퍼스가 두 규약으로 갈린다.
    parser.add_argument("--follow_keyframes", default=SHAPE_DEFAULTS["follow_keyframes"], type=int)
    parser.add_argument("--follow_kf_interp", default=SHAPE_DEFAULTS["follow_kf_interp"],
                        choices=["linear", "savgol", "cubic"])
    parser.add_argument("--start_mode", default="source_frame0", type=str)
    parser.add_argument("--aim_anchor", default="subject", type=str)
    # 기본값은 `SHAPE_DEFAULTS` 한 군데서만 온다 (emit_bank 가 예전 뱅크를 읽을 때 같은 값을
    # 써야 하기 때문 — 실제로 여기서 어긋나 재현이 깨진 적이 있다).
    parser.add_argument("--aim_ramp_frames", default=SHAPE_DEFAULTS["aim_ramp_frames"], type=int)
    # preset 모양을 얹을 기준 회전 (D69). `subject` 면 pan/truck/pedestal 계열도 anchor 를 향해
    # 세운 회전 위에서 움직인다 — 소스 광축이 anchor 를 안 볼 때 subject 가 프레임을 벗어나는 걸 막는다.
    # D99. `auto` 는 조준 방식이 정한다 (look_at -> subject, free -> source) — object-centric
    # preset 의 선회 중심을 OBB 중심에 앉힌다 (§presets.resolve_traj_basis).
    # **기본값은 아직 `source`** 이고, 바꿀 땐 `sample_camera_bank.py` 와 **같이** 바꿔야 한다 —
    # 여기만 켜면 같은 뱅크를 두 규약으로 굽게 된다 (D90 smooth_passes 사고).
    parser.add_argument("--traj_basis", default=SHAPE_DEFAULTS["traj_basis"], type=str,
                        choices=["source", "subject", "auto"])
    parser.add_argument("--orbit_span_frac", default=SHAPE_DEFAULTS["orbit_span_frac"], type=float)
    # 2026-09-02: 15 -> 20 (사용자 지시, "일단 20 으로 놔둬보고 돌려보고 결정"). 이건 **하한**이다 —
    # `span_frac = max(orbit_span_frac, min_sweep_deg / obs_az_span)` 이라 관측 방위폭이 좁은 씬에서
    # orbit 이 몇 도짜리로 쪼그라드는 걸 막는다. `SHAPE_DEFAULTS` 쪽(15.0)은 키가 없는 예전 뱅크의
    # 재현 폴백이라 **안 건드린다** — 두 층 구조.
    parser.add_argument("--min_sweep_deg", default=20.0, type=float)
    # F9. 이분법이 sweep 이 아니라 반경을 깎게 한다 (§presets.shape_resizer). sweep 을 안 쓰는
    # preset 은 `shape_resizer` 가 None 을 줘서 옛 경로 그대로다.
    parser.add_argument("--orbit_fixed_sweep", action="store_true", default=True)
    parser.add_argument("--no_orbit_fixed_sweep", dest="orbit_fixed_sweep", action="store_false")
    # F2. 옛 게이지(`frame0_ray`)로 구운 그래프를 일부러 쓸 때만 (§scene_graph.scale.assert_scale_mode).
    parser.add_argument("--allow_legacy_scale", action="store_true", default=False)
    # D71 keyframe 조준. `sample_camera_bank.py` 는 사다리(`nargs="*"`)로 훑지만 여기서는 축을
    # 늘리지 않는다 — hole 이분법이 keyframe 수마다 따로 돌아야 해서 비용이 그대로 배가 된다.
    # 0 = 예전 동작(매 프레임 조준).
    parser.add_argument("--aim_keyframes", default=SHAPE_DEFAULTS["aim_keyframes"], type=int)
    parser.add_argument("--keyframe_aim", default=SHAPE_DEFAULTS["keyframe_aim"], type=str,
                        choices=["auto", "target", "preset_rel"])
    # D89. 기본값이 `SHAPE_DEFAULTS` 가 아니다 — 그쪽은 **키가 없는 예전 뱅크를 재현할 때의
    # 폴백**이라 영원히 `smoothstep` 이어야 하고, 여기는 **새로 굽는 뱅크의 기본값**이다.
    # 새 뱅크는 `fixed.keyframe_ease` 에 값을 명시적으로 적으므로 폴백을 타지 않는다.
    # D173. `cubic`/`savgol` 은 위치 채널 `--follow_kf_interp` 와 같은 이름 = 같은 규약이다.
    parser.add_argument("--keyframe_ease", default="smooth_kf", type=str,
                        choices=["smoothstep", "linear", "arclen", "arclen_kf", "smooth_kf",
                                 "cubic", "savgol"])
    # D90. `smooth_kf` 스케줄 전용 인자. 여기 기본값은 `build_poses` **서명** 기본값(12/0.5)과
    # 같아야 한다 — 예전에는 이 스크립트가 인자를 아예 안 넘겨서 서명 기본값으로 구웠고,
    # `emit_bank` 는 자기 CLI 기본값(4)으로 되만들어 회전만 조용히 어긋났다. 이제 `fixed` 에 싣는다.
    parser.add_argument("--smooth_passes", default=SHAPE_DEFAULTS["smooth_passes"], type=int)
    parser.add_argument("--smooth_lambda", default=SHAPE_DEFAULTS["smooth_lambda"], type=float)
    # D97. `keyframe_ease` 와 같은 두 층 구조다 — `SHAPE_DEFAULTS["tau_ref"]` 는 **키가 없는
    # 예전 뱅크의 폴백**이라 영원히 `"source"` 고, 여기는 **새로 굽는 뱅크의 기본값**이다.
    # `"auto"` = `track_*` 만 follow 기준 (§presets.py `PRESET_TAU_REF`).
    # D128. 기본을 `follow` 로 뒤집었다 — 근거는 `sample_camera_bank.py` 의 같은 인자 주석
    # (D125 parkour 프로브: behind 0.347/0.286/0.163 -> 0.000).
    parser.add_argument("--tau_ref", default="follow", type=str, choices=list(TAU_REF_CHOICES))
    # D150. τ 의 **분모**. `--tau_ref` 가 "무엇으로부터의 이동인가"라면 이건 "무엇으로 나누나"다.
    # `S` = 전 프레임 non-sky 점의 첫 카메라 거리 평균 (사용자 지시 2026-09-06),
    # `z_med_frame0` = 예전 frame0 z-depth 중앙값. **기본은 예전 것** — 이 값이 뱅크 정체성이라
    # 기본을 뒤집으면 진행 중인 굽기가 코퍼스 중간에 정의를 갈아탄다. 정의·배율은
    # `scene_graph.scale.tau_denominator` docstring.
    parser.add_argument("--tau_denom", default="z_med_frame0", type=str,
                        choices=list(TAU_DENOM_MODES))
    # F5. τ 손잡이 **하한** (`--tau_floor_src` 가 켜져 있으면 `tau_start + 이 값`). 예전엔
    # `KNOB_RANGE["tau"][0]` 상수 0.02 였고 기본값은 그 값 그대로다 — 안 주면 비트 동일.
    #
    # 0.005 를 쓰는 근거 (vista 6편 x 4 preset x hole 4단, 하한 7값 스윕, tmp/d151):
    # 채택 기준은 "물리 위반(`behind_frames>0 ∨ obb_slack<0 ∨ below_ground_frames>0`)이 0 이
    # 되는 **가장 큰** 하한". 하한이 높으면 이분법이 게이트를 만족하는 크기까지 못 내려가서
    # `clamped_low` 로 남는다. 위반이 0 이 되는 지점은 martian-flag 0.005 / camera-lens 0.01 /
    # snow-dog 0.01 / parkour 0.01 / hike 이미 0 이라, 최악 씬을 덮는 값이 0.005 다.
    # 그 아래(0.002~0)로 내려도 **어느 씬도 새로 안 풀리고** `path_len_u` p50 이 하한을 그대로
    # 따라 줄면서 `motion_preset_no_motion` suspect 만 는다 (snow-dog 3 -> 6@0.005 -> 8@0.002)
    # — 즉 정지 궤적을 정지가 아닌 척 내보내는 구간이다.
    # 임계 0.005/0.01/0.01/0.01 이 S/z_med 29.85/12.68/9.26/5.69 와 무상관이라 **씬 스케일
    # 함수가 아니라 단일 상수**다 (`0.02·z_med/S` 안은 이걸로 기각).
    # 안전성: "위반 행이 export 가능해진 사례"가 7x6 셀 전부에서 0 — 하한을 내려도 나쁜 행이
    # `DROP=clamped_low` export 필터를 빠져나가지 않는다.
    # 못 고치는 것: woman-phone 형(48행 전량)은 `src_ground_clear = -0.0082` 로 **소스 카메라
    # 자체가 추정 지면 아래**라 지면 게이트 기준선이 음수다. τ 하한으로는 안 풀린다.
    parser.add_argument("--tau_knob_min", default=TAU_KNOB_MIN_DEFAULT, type=float)
    # D266. 0 = 안 건드림 (기본, 예전 상한 3.0 그대로).
    parser.add_argument("--tau_knob_max", default=0.0, type=float)
    # D99. 중력 기준 roll 보정 — 광축은 그대로 두고 지평선만 세운다. 두 층 구조는 위와 같다
    # (`SHAPE_DEFAULTS["deroll"]` 는 영원히 False — 키가 없는 예전 뱅크의 재현 폴백).
    # D105. **굽는 기본값을 True 로 뒤집었다.** off 로 구운 d98 뱅크는 `tilt_*` 가 roll 163.88°,
    # `pan_*` 가 18.12° 를 흘려서 3,280행(15.7%)이 Dutch angle 로 학습 데이터에 들어간다.
    # 예전 뱅크를 되만들 때는 `--no_deroll` 을 명시할 것.
    parser.add_argument("--deroll", action="store_true", default=True)
    parser.add_argument("--no_deroll", dest="deroll", action="store_false")
    # 렌더러 K 를 frame0 에 고정 (DA3 per-frame focal drift 상쇄; snowboard +6.99%).
    parser.add_argument("--fixed_focal", action="store_true", default=False)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")

    # ── D140. suspect 태그. **판정이 아니라 진단**이라 궤적/게이트에 영향이 없다 (`status` 를
    #    안 건드리고 열만 하나 는다). `--no_suspect` 면 열이 빈 칸이라 D139 이전 뱅크와 같다.
    # D168 ⑤. `hole_fraction` 이 이걸 넘으면 태그. 0 이면 꺼짐 = D167 뱅크와 같다.
    # 0.35 는 d166 config `_rung` 이 이미 합격 기준으로 쓰던 숫자다 (새 눈금이 아니다).
    parser.add_argument("--suspect_hole", default=0.0, type=float)
    parser.add_argument("--suspect", action="store_true", default=True)
    parser.add_argument("--no_suspect", dest="suspect", action="store_false")
    # 임계 셋. 기본값 근거는 `tag_suspects` docstring 참고.
    parser.add_argument("--suspect_in_frame", default=0.85, type=float)   # ① 이 밑이면 태그
    parser.add_argument("--suspect_behind", default=0.0, type=float)      # ② 이 위면 태그
    parser.add_argument("--suspect_path_len", default=0.02, type=float)   # ②③ 이 밑이면 정지로 본다

    # ── 실측
    parser.add_argument("--bisect_frames", default=5, type=int)
    parser.add_argument("--verify_frames", default=13, type=int)   # 0 = 전 프레임
    # G3 가림(`subject_visible_frac`). 판정 패스에서만 재므로 이분법 횟수는 안 늘고 verify 렌더만
    # 2배가 된다. 끄면 열이 빠지고 예전 뱅크와 비트 단위로 같다 (D81).
    parser.add_argument("--subject_visible", action="store_true", default=True)
    parser.add_argument("--no_subject_visible", dest="subject_visible", action="store_false")
    # D112. 가림을 **판정**으로 올린다 (사용자 지시). 0 이하면 예전 동작 그대로 (측정만).
    # 켜면 이분법이 subject 실루엣을 따로 그려야 해서 fit 렌더가 2배다.
    #
    # D171 (사용자 지시 2026-09-10 "어차피 hole 볼 때 랜더링하니까 subject_visible_frac 를
    # 판정으로 올려줘"). 기본값을 0.0(측정만) → **0.6(판정)** 으로 올린다. `0` 을 주면 예전
    # 동작으로 정확히 되돌아간다.
    #
    # 왜 0.6 인가 (d169 dynpose 21편 164행 실측, `subject_visible_frac` = 잰 프레임의 median):
    #
    #     분위수   p05 0.508   p10 0.562   p25 0.827   p50 0.971
    #     임계     0.30 →   7행(4.3%)   0.50 →  8행(4.9%)   0.60 → 20행(12.2%)
    #              0.70 →  30행(18.3%)  0.80 → 36행(22.0%)  0.85 → 44행(26.8%)
    #
    # 0.5 이하는 분포의 바닥만 긁어서 **놓치는 사례가 남는다** — `027514bb orbit_right` 은
    # `subject_in_frame 1.0`(프레임 안에 멀쩡히 있다) 인데 `frac 0.542 / min 0.007` 로 벽에
    # 가려 사실상 안 보이는데도 태그가 하나도 안 붙었다. 0.7 이상은 usable 60행 중 8행(13%)의
    # 손잡이를 깎기 시작해서 대가가 커진다 (0.6 은 2행). 그 사이가 0.6 이다.
    #
    # **이 게이트는 행을 버리지 않는다** — 이분법이 가시비율이 임계를 넘을 때까지 손잡이를
    # 줄인다. 손잡이 하한에서도 못 넘기면 `status=clamped_low` + `binding=occlusion` 이 되고,
    # 그건 이미 `--retry_status clamped_low` 가 잡아 다음 층 preset 으로 넘긴다.
    parser.add_argument("--min_subject_visible", default=0.6, type=float)
    # D192. 정지 preset(`STATIC_PRESETS` — `static_look_at` / `track_look_at` 등)에도 위 게이트
    # 전부를 **한 번** 적용한다. 손잡이가 없어 이분법을 건너뛰는 preset 이라 여태 게이트 판정도
    # 같이 건너뛰었다 (측정 열은 채워지는데 아무도 안 읽는다 — 근거는 본문 `args.gate_static`
    # 주석의 3,205행 실측). 위반하면 `status = f"{사유}_blocked"` 가 되므로
    # `--retry_status collision_blocked obb_blocked ...` 로 사다리와 pick 에서 뺄 수 있다.
    # 기본 off = 옛 동작 비트 동일.
    # [new 2026-09-27, R62] 시작 pose 선판정 (frame 0 카메라만, fit 과 같은 게이트). off = 기존 동작.
    parser.add_argument("--start_screen", default="off", choices=["off", "only", "filter"])
    parser.add_argument("--gate_static", dest="gate_static", action="store_true", default=False)
    parser.add_argument("--no_gate_static", dest="gate_static", action="store_false")
    # ── D168 (사용자 지시 2026-09-08). **게이트는 한 개도 안 늘린다** — 이분법(`solve_knob`)과
    #    사다리 목표는 D166 과 글자 그대로 같다. 새로 짜는 건 routing 과 retry 둘뿐이다.
    #    끄면(기본) 출력이 D167 뱅크와 **비트 단위로 같다**.
    #
    #    판정 어휘를 새로 만들지 않는 게 핵심이다. "이 변이는 못 쓴다"는 판정은 이미
    #    `status`(이분법 판정) 와 `suspect`(D140 `tag_suspects`) 두 열에 있고, 코퍼스 단계
    #    (`vista4d_bank_to_dl3dv.py --drop_status/--drop_suspect`) 가 그걸 소비한다.
    #    retry 는 **같은 두 열을 같은 이름의 인자로** 읽는다 — 그래서 "코퍼스에서 잘릴 행"과
    #    "재시도를 부르는 행"이 정의상 같은 집합이고, 둘이 어긋날 수가 없다.
    parser.add_argument("--fallback_ladder", action="store_true", default=False)
    parser.add_argument("--no_fallback_ladder", dest="fallback_ladder", action="store_false")
    parser.add_argument("--fallback_target", default=5, type=int)      # 이만큼 모이면 멈춘다
    # `--drop_status` / `--drop_suspect` 와 같은 이름·같은 토큰 매칭. 접두사로 본다
    # (`clamped_low` 가 `clamped_low+tau_floor` 도 잡는다).
    parser.add_argument("--retry_status", nargs="*", default=["clamped_low"])
    parser.add_argument("--retry_suspect", nargs="*", default=[])
    # D188 ③. 씬당 내보낼 변이 수. 0 = 안 고른다 (기본, 예전 뱅크와 비트 동일).
    # `--fallback_target` 과 **다른 축**이다 — 저건 사다리를 어디서 멈출지, 이건 다 돌고 나서
    # 무엇을 내보낼지. 사다리가 suspect 때문에 끝까지 내려가도 여기서 1개로 접힌다.
    parser.add_argument("--pick_budget", default=0, type=int)
    # D119. shot scale 시간축. `verify_frames` 프레임에서 이미 재고 있던 면적비를 median 으로
    # 접기 전에 그대로 싣는다 — **렌더가 안 늘어난다**. 끄면 세 열이 빠지고 예전 뱅크와 같다.
    parser.add_argument("--area_timeline", action="store_true", default=True)
    parser.add_argument("--no_area_timeline", dest="area_timeline", action="store_false")
    # D121. composition — anchor 말고 **또 무엇이 화면에 담기나** (`in_frame_ids`/`enter_ids`/
    # `exit_ids`). 캡션 framing 절이 shot scale 한 축뿐이던 걸 늘리는 재료다. OBB 투영이라
    # 렌더가 0회. 끄면 세 열이 빠지고 예전 뱅크와 같다.
    parser.add_argument("--composition", action="store_true", default=True)
    parser.add_argument("--no_composition", dest="composition", action="store_false")
    # 화면 면적비 임계. 0.004 = 640x360 에서 약 920 px — 이보다 작으면 문장에 적을 만큼
    # 보이는 게 아니다. camel `tree`(배경) 가 이 근처라 값을 뱅크 메타에 같이 남긴다.
    parser.add_argument("--composition_min_area", default=0.004, type=float)
    parser.add_argument("--composition_max_nodes", default=3, type=int)
    parser.add_argument("--tile_width", default=640, type=int)
    parser.add_argument("--tile_height", default=360, type=int)
    parser.add_argument("--center_box", default=0.80, type=float)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
