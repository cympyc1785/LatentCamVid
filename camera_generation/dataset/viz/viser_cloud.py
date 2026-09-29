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

**지면 격자도 기본 꺼짐**이다 (`ground grid` / `--ground_on`). 격자는 카메라와 노드를 다 감싸게
`half` 를 잡으므로 씬 전체를 덮는데, 정작 보려는 카메라·OBB 위에 얹혀 화면을 가린다. 중력축이
틀어졌는지 확인할 때만 켠다 — 그때는 격자가 실제 바닥과 어긋나 보이는지가 판정 근거다.
`--no_ground_grid` 는 아예 **만들지 않는** 것이고(그러면 GUI 체크박스도 비활성), `--ground_on`
은 만든 것을 **처음부터 보이게** 하는 것이다.

**소스 카메라도 기본 꺼짐**이다 (`source cameras` / `--source_on`). 보려는 건 거의 항상 target
궤적인데 소스 49대가 원점 근처에 뭉쳐 앉아 그걸 덮는다. 이 체크박스는 소스 프러스텀과 소스
경로선을 **같이** 끈다 — `paths` 하나로만 묶여 있으면 소스를 끈 상태에서 paths 를 켰을 때 "끈
카메라의 궤적"만 화면에 남는다. 비교 기준선으로 소스가 필요할 때(떨림 판정)만 켠다.

## target 색은 시간축이다 (`camera colors` 폴더)

target 프러스텀·경로선이 **frame 0 빨강 -> 마지막 프레임 파랑** 으로 섞인다. 단색이면 49대가
전부 같은 색이라 시작과 끝이 구분되지 않아서, 궤적을 눈으로 보고도 dolly_in 인지 dolly_out
인지(= 어느 쪽으로 흐르는지)를 판정할 수가 없었다. `target start`/`target end` 피커로 두 끝
색을 바꾸고, `target gradation` 을 끄면 start 하나만 단색으로 쓰인다 — 그때 초기값은
`--plan_color`(예전 주황) 라 `--no_plan_gradient` 로 띄우면 예전 화면이 그대로 재현된다.
기동 시 지정은 `--plan_color_start` / `--plan_color_end`.

프러스텀과 경로선이 **같은 램프**를 쓴다. 따로 계산하면 `--cam_stride` downsample 탓에 프러스텀
i 의 색과 그 자리를 지나는 선의 색이 어긋나고, 그러면 "색 = 시간" 이라는 약속이 깨진다.

**pin 은 gradation 을 안 받는다.** 색의 두 쓰임이 다른 축이기 때문이다 — gradation 은 "궤적의
어디쯤"(시간), pin 색은 "어느 변이"(정체) 다. pin 까지 램프로 칠하면 여러 대를 구분한다는 pin
의 존재 이유가 사라진다.

## obb 폴더 — 어느 박스를 띄우나

`node` 드롭다운이 **한 노드만** 남긴다 (`all` 이면 전부). 기본이 **`dyn_0`** 다 — 씬 하나에
노드가 6~16개씩 있어 전부 켜면 어느 박스가 지금 보는 subject 인지 고를 수가 없고, 큰 정적
박스(snow-dog `stat_0 forest` 가 6.36 m)가 작은 subject 박스를 통째로 덮는다. 예전처럼 전부
켜려면 `--obb_node all`, 다른 노드는 `--obb_node dyn_1`. 그 id 가 그래프에 없으면 `all` 로
떨어진다 (노드 이름이 씬마다 달라 죽지 않게).
`labels` 는 **기본 꺼짐** — 노드가 여럿이면 글자가 서로 겹쳐 읽히지도 않으면서 박스를 가린다.

색은 handle 에 바꿔 끼울 수 없어 remove + re-add 다 (`LineSegmentsHandle` 에 `colors` 프로퍼티가
없다). snowboard 는 동적 5노드 × 49프레임 + 정적 4 = 249개라 매번 다 다시 그리면 슬라이더가
끊긴다. 그래서 `obb_version` 을 찍어 두고 **보이게 되는 순간에만** 밀린 것을 갚는다 —
`current frame` 모드면 실제로 다시 그리는 건 9개뿐이다.

## motion 브라우저 (`--banks`)

뱅크 여러 개를 통째로 올려두고 **슬라이더로 motion 을 갈아끼운다**. 이게 필요한 이유는
"카메라가 떨린다"를 2D 렌더로는 못 가르기 때문이다 — 렌더가 떠는 원인이 셋이나 된다:
① plan 카메라 위치의 고주파(`follow_gain` × subject track 잔여 jitter), ② 조준 회전,
③ **점군 자체**(프레임마다 depth 가 조금씩 달라 정적 배경이 숨쉰다). 3D 에서 카메라 경로를
직접 보면 ①/②는 선이 지그재그로 보이고, 선이 매끈한데 렌더가 떨면 남는 건 ③뿐이다.

경로는 catmull-rom 스플라인이 아니라 **`add_line_segments` 생꺾은선**으로 그린다. 스플라인은
지금 보려는 그 jitter 를 부드럽게 만들어 버린다.

## d221 번들 브라우저 (`--bundle`)

`results/20260921_d221_bundles/<scene>/` 를 통째로 올린다. 뱅크 대신 **모델 예측**을 씬과 같이
보는 경로다 — 레이아웃이 `<preset>/cameras/{gt,s42,s1234,s2026,gendop}.npz` 라, preset x arm
을 전부 motion 으로 펴서 같은 슬라이더에 꽂는다 (라벨 `<preset>/<arm>`). `cloud.npz` 와
`scene_graph.json` 도 **번들 폴더에서** 읽는다 — 번들의 graph 는 그 세대의 스냅샷이라 `out/` 을
다시 구운 뒤에도 그 카메라를 낳은 씬과 같이 볼 수 있다.

**좌표 변환이 없다.** `cam_c2w` 는 이미 recon world 의 절대 pose·절대 미터이고 recon world
원점이 소스 frame0 카메라다 (bmx-bumps 실측 `meta_cam_c2w[0] == I`, 8.9e-8). 그 증거로
`gt.npz` 가 뱅크 변이 `hole_bank_d215/dyn_0__crane_up__hole0.2` 와 위치 2.2e-7 로 일치한다.
앵커를 한 번 더 곱하면 궤적이 두 번 옮겨져 점군과 어긋난다 (`bank_to_vista4d_cams.py` 의
docstring 이 이 가정의 단일 출처다).

프러스텀 화각은 **arm 자기 focal** 로 그린다 (`intrinsics[0]`). 예측은 focal 이 조금씩 다른데
(s42 fx 2287.88 vs recon 2293.41) 소스 K 로 통일하면 그 zoom 차이가 화면에서 사라진다.
반대로 `jerk p95` 는 소스 focal 로 정규화한다 — arm 마다 분모가 다르면 숫자를 못 비교한다.

info 패널은 `caption.json` 을 그대로 읽는다. 그 파일이 `variant_id`/`tau_max`/`hole_fraction`/
`subject_in_frame` 을 들고 있어 뱅크의 `bank.json` 자리를 메운다. `anchor_id` 는
`variant_id` 앞머리(`dyn_0__...`)에서 뽑는다 — 없으면 subject track 초록선이 안 그려진다.

`--bundle_presets` / `--bundle_arms` 로 목록을 줄인다 (3 preset x 5 arm = 15개라 한 preset 의
5 arm 만 보려면 preset 을 걸어야 한다).

## target 카메라 여러 대 동시에 (`--pin` / GUI `pin current`)

슬라이더는 **한 번에 한 대**라 "A 가 B 보다 더 도나 / 둘이 같은 자리에서 시작하나"를 못 본다 —
갈아끼우는 순간 비교 대상이 사라지기 때문이다. pin 은 그 motion 을 고정 색으로 남겨서 활성
motion(주황) 을 계속 바꿔도 화면에 살아 있게 한다. pin 색은 **pin 하는 순간 `target (pred)`
피커에 들어있던 색**을 스냅샷으로 뜬다 — 방금 그 색으로 보던 궤적이 pin 하는 순간 팔레트
색으로 튀면 대조가 끊기기 때문이다. 여러 대를 구분하려면 피커를 바꿔 가며 pin 한다.
예전 `PIN_COLORS` 순환은 `--pin_palette` 로 남겨 뒀다. `pinned` 패널이 색↔라벨 대응을
적어 준다. 상한은 `--max_pins` (기본 8) — 프러스텀이 변이당 49/cam_stride 개라 무제한으로
켜면 브라우저가 느려진다.

"현재 프레임" 을 1.6배 프러스텀으로 따로 그리는 기능은 **기본 꺼짐**(`--now_cams` 로 켠다).
같은 자리의 일반 프러스텀 위에 큰 것이 겹쳐 앉아, 궤적을 훑을 때 카메라가 두 대인 것처럼
보였다. 지금 프레임은 frame 슬라이더와 `dynamic frames` 로 읽는다.

같이 그리는 것: 소스 카메라 경로(회색) · plan 경로(주황) · subject track(초록, follow 의
입력이라 여기가 떨면 카메라도 떤다). GUI 에 preset/gain/smooth/k 와 `jerk p95`(px/frame³,
`|Δ³p|/z_med·fx` — 화면에서 실제로 몇 px 흔들리는지)를 같이 띄운다.

env   : **`vista4d`** (viser >= 1.1.0). `latentcam` 의 viser 1.0.30 은 `thickness` 가 없어
        기동 직후 멈춘다 (`require_viser_thickness`) — 그쪽은 `line_width` 가 screen 픽셀이라
        이름만 바꿔 끼우면 선과 프러스텀이 통째로 안 보인다.
입력  : `<out>/<video>/cloud.npz` (format `lbm_cloud_v1`) — `--bundle` 이면 `<bundle>/cloud.npz`
        선택: `<out>/<video>/<bank>/poses.npz` + `bank.json` (plan 카메라 오버레이)
출력  : 브라우저 (`http://localhost:<port>`)

예시 (env vista4d, screen viser1~4 에서):
    python -u viz/viser_cloud.py --video snowboard --port 8084
    python -u viz/viser_cloud.py --video snowboard --port 8084 \
        --source_on --no_plan_gradient --plan_color 255,140,40     # 예전 동작 그대로
    python -u viz/viser_cloud.py --port 8090 \
        --bundle results/20260921_d221_bundles/bmx-bumps          # 예측 15개 (3 preset x 5 arm)
    python -u viz/viser_cloud.py --port 8090 --no_cloud \
        --bundle results/20260921_d221_bundles/bmx-bumps --bundle_presets crane_up \
        --pin crane_up/gt crane_up/s42 crane_up/s1234 crane_up/s2026  # 4대 동시 대조
    python viz/viser_cloud.py --video snowboard --port 8084 --no_cloud \
        --banks follow_smooth_bank notrack_bank worldaim_bank
    python viz/viser_cloud.py --video snowboard --port 8084 \
        --bank follow_kf_bank --variant dyn_0__orbit_left_arc__tau0.35   # 예전 동작(1개 고정)
    python viz/viser_cloud.py --video snowboard --port 8084 --no_cloud \
        --bank hole_bank_k6_d151 --pin orbit_left dolly_in truck_left    # 여러 대 동시에
"""
import json
import sys
import time
from argparse import ArgumentParser
from inspect import signature
from os import listdir, path

import numpy as np
import viser

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
VIEWER_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "latentcam", "scripts", "viewer"))

# world up 문자열 -> world 축 벡터. cloud world 는 OpenCV(=Y down) 라 기본이 '-y' 다.
UP_VECTORS = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
              "+z": (0, 0, 1), "-z": (0, 0, -1)}

# probe 프러스텀의 기본 세로 화각. 소스 intrinsics 를 안 쓰는 이유는 그게 씬마다 망원이기
# 때문이다 — camel 은 vfov 16.4° 라 프러스텀이 바늘처럼 길어져 자리를 가늠할 수가 없다.
PROBE_FOV_DEG = 60.0

# [2026-09-21] 소스·target 프러스텀의 세로 화각도 **고정값**이다. 카메라 자기 화각을 쓰면 씬마다
# 프러스텀이 바늘처럼 길어져 "이 카메라가 어디에 있나"를 가늠할 수가 없다 — bmx-bumps 소스는
# vfov 17.8°(fy 2293 @ 720p), camel 은 16.4° 다. probe 가 이미 같은 이유로 위 상수를 쓰는데,
# 정작 본편 프러스텀은 소스 K 를 쓰고 있었다. 화각 차이(arm 별 focal)는 info 패널의 `fx` 로
# 읽는다 — 그림에서 자리를 못 읽는 대가로 얻을 정보가 아니다. 예전 동작은 `--cam_fov_src`.
FRUSTUM_FOV_DEG = 60.0

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
PLAN_COLOR = (255, 140, 40)          # 활성 target 단색 (gradation 을 끌 때만)
PLAN_NOW_COLOR = (255, 80, 0)        # target 의 현재 프레임 프러스텀 (1.6배)

# [2026-09-21] target 궤적의 **시간 gradation**. 단색이면 "이 프러스텀이 궤적의 어디쯤인가"를
# 읽을 수가 없다 — 49대가 전부 같은 주황이라 시작과 끝이 구분되지 않아서, 궤적을 보고도
# dolly_in 인지 dolly_out 인지(= 어느 쪽으로 흐르는지) 판정이 안 됐다. frame 0 = 빨강 ->
# frame F-1 = 파랑 으로 섞어 시간축을 색에 얹는다. 예전 단색은 `--no_plan_gradient`.
PLAN_COLOR_START = (255, 40, 40)     # frame 0
PLAN_COLOR_END = (40, 80, 255)       # frame F-1

# scene graph OBB. **동적이 하늘색**이다 — 보는 대상이 거의 항상 동적 subject 라서, 눈이 먼저
# 가야 하는 쪽에 원래 쓰던 색을 준다. 정적은 겹치지 않게 청록으로 민다 (둘 다 하늘색이면
# "이 박스가 따라 움직여야 하는가"를 못 가른다). `--obb_color_dyn/_static` 로 덮는다.
OBB_DYN_COLOR = (80, 200, 255)       # 하늘색
OBB_STATIC_COLOR = (0, 200, 170)     # 청록


VISTA4D_PY = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"


def require_viser_thickness():
    """`add_line_segments(thickness=...)` 가 있는지 **cloud 를 읽기 전에** 확인한다.

    이 스크립트는 env `vista4d` (viser 1.1.0) 용이다. `latentcam` 의 viser 1.0.30 은 그 인자가
    `line_width` 이고 단위가 **screen 픽셀**이라, 이름만 바꿔 끼우면 world 0.007 이 0.007
    픽셀이 돼 선과 프러스텀이 통째로 안 보인다 (CHANGELOG 의 "프러스텀이 통째로 안 보이던
    것" 이 그 증상) — 조용히 깨진 화면보다 여기서 멈추는 게 낫다.

    확인을 함수 맨 앞으로 당긴 이유: 원래 실패 지점이 `draw_line` 이라 cloud.npz(1.4 GB)를
    다 읽은 **뒤**에 죽었다. 잘못된 env 로 띄우면 몇 분 기다린 끝에 TypeError 를 본다.
    """
    if "thickness" in signature(viser.SceneApi.add_line_segments).parameters:
        return
    raise SystemExit(
        f"[env] viser {viser.__version__} 은 `thickness` 를 안 받는다 (1.1.0 이상 필요).\n"
        f"      이 스크립트는 env vista4d 용이다. 아래로 다시 띄운다:\n"
        f"        {VISTA4D_PY} -u " + " ".join(sys.argv))


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


# 번들 arm 표시 순서. `gt` 를 맨 앞에 두는 이유는 슬라이더 0번이 곧 기준선이어야 하기
# 때문이다 — 예측부터 보면 "이게 GT 대비 얼마나 틀어졌나"를 볼 때마다 슬라이더를 되감아야 한다.
# 여기 없는 이름의 npz 가 생기면 뒤에 사전순으로 붙는다 (목록에서 빠지지 않게).
BUNDLE_ARMS = ("gt", "s42", "s1234", "s2026", "gendop")


def _bundle_caption(preset_dir: str, preset: str):
    """번들 preset 폴더 -> info 패널용 row 초안.

    두 레이아웃을 받는다. vista 번들(`run_d221.py`)은 변이 하나를 뽑아 둔 `caption.json` 을
    쓰고, dynpose 번들(`run_d219.py`)은 뱅크 캡션 파일을 통째로 복사한 `captions.json`
    (`{"captions": {variant_id: {...}}}`)을 쓴다. 후자는 preset 여러 개가 한 파일에 있으므로
    `__<preset>__` 로 그 preset 의 변이를 고른다 — 변이 키(`dyn_0__track_crane_up__hole0.2`)가
    `anchor_id` 의 출처라, 여기서 못 찾으면 subject track 초록선이 안 그려진다.
    """
    cap_path = path.join(preset_dir, "caption.json")
    if path.isfile(cap_path):
        with open(cap_path, encoding="utf-8") as file:
            return json.load(file)
    caps_path = path.join(preset_dir, "captions.json")
    if not path.isfile(caps_path):
        return {}
    with open(caps_path, encoding="utf-8") as file:
        entries = json.load(file).get("captions", {})
    key = next((k for k in entries if f"__{preset}__" in k), None)
    if key is None:
        return {}
    row = dict(entries[key])
    row["variant_id"] = key
    row.setdefault("caption_fields", entries[key])
    return row


def _bundle_arm_files(cam_dir: str):
    """`cameras/` -> {arm: 파일 경로}. npz 가 있으면 npz 를 쓴다.

    dynpose 번들에는 `cameras/*.npz` 가 없다 — `run_d219.py:stage_bundle` 이 Vista4D recon
    이 있는 씬에만 npz 를 굽기 때문에 예측 JSON(`<arm>_transforms.json`)만 복사된다. 뷰어가
    npz 만 읽으면 그 번들은 통째로 "arm 이 0개"로 떨어진다. 둘 다 있으면 npz 가 이긴다 —
    그게 Vista4D 가 실제로 소비한 파일이라, 규약 변환이 한 번 더 끼는 JSON 경로보다 원본에
    가깝다.
    """
    files = listdir(cam_dir)
    srcs = {f[:-4]: path.join(cam_dir, f) for f in sorted(files) if f.endswith(".npz")}
    for f in sorted(files):
        if f.endswith("_transforms.json"):
            srcs.setdefault(f[:-len("_transforms.json")], path.join(cam_dir, f))
    return srcs


def _bundle_arm_poses(file_path: str, cx_recon=None):
    """arm 파일 -> (c2w OpenCV recon world (F,4,4), fx, fy) — fx/fy 는 없으면 None.

    JSON 경로의 변환식은 `bank_to_vista4d_cams.py:load_pred` 와 **같아야 한다**. 거기가
    nerfstudio(OpenGL c2w)를 `diag(1,-1,-1,1)` 로 OpenCV 로 돌리고, JSON 이 절반 해상도로
    적혀 있어 `fl_x` 를 `recon cx / json cx` 배 해서 recon 픽셀로 되돌린다. 두 경로가 갈리면
    같은 궤적이 뷰어에서와 렌더에서 다르게 보인다.
    """
    if file_path.endswith(".npz"):
        data = np.load(file_path)
        intr = data["intrinsics"] if "intrinsics" in data.files else []
        fx, fy = ((float(intr[0][0]), float(intr[0][1])) if len(intr) else (None, None))
        return data["cam_c2w"].astype(np.float64), fx, fy
    with open(file_path, encoding="utf-8") as file:
        blob = json.load(file)
    c2w = (np.array([f["transform_matrix"] for f in blob["frames"]], dtype=np.float64)
           @ np.diag([1.0, -1.0, -1.0, 1.0]))
    scale = float(cx_recon) / float(blob["cx"]) if cx_recon else 1.0
    return c2w, float(blob["fl_x"]) * scale, float(blob["fl_y"]) * scale


def load_bundle(bundle_root: str, presets=(), arms=(), variant_filter=(), cx_recon=None):
    """d221 번들 -> `load_motions` 와 **같은 모양** [(label, c2w (F,4,4), row)].

    레이아웃: `<bundle>/<preset>/cameras/{gt,s42,s1234,s2026,gendop}.npz`, 라벨은
    `<preset>/<arm>`. npz 가 없는 dynpose 번들은 `<arm>_transforms.json` 으로 떨어진다
    (`_bundle_arm_files`).

    **좌표 변환이 없다.** `cam_c2w` 는 이미 recon world 의 절대 pose·절대 미터이고, recon
    world 원점이 소스 frame0 카메라다 (bmx-bumps 실측 `meta_cam_c2w[0] == I`, 8.9e-8). 그
    증거로 `gt.npz` 가 뱅크 변이 `hole_bank_d215/dyn_0__crane_up__hole0.2` 와 위치 2.2e-7 로
    일치한다. 앵커를 한 번 더 곱하면 궤적이 그만큼 두 번 옮겨져 점군과 어긋난다
    (`bank_to_vista4d_cams.py` 도 같은 가정으로 굽는다 — 그 파일 docstring 이 단일 출처다).
    JSON arm 만 규약 변환이 있다 — 그건 nerfstudio 판본이라서다.

    캡션을 row 에 합친다. 그 파일이 `variant_id`/`tau_max`/`hole_fraction`/
    `subject_in_frame` 을 들고 있어서 뱅크의 `bank.json` 자리를 그대로 메운다 (info 패널이
    같은 키를 읽는다). `anchor_id` 는 `variant_id` 앞머리(`dyn_0__...`)에서 뽑는다 — 이게
    없으면 follow 의 입력인 subject track 초록선이 안 그려진다.
    """
    if isinstance(variant_filter, str):
        variant_filter = [variant_filter] if variant_filter else []
    want_presets, want_arms = set(presets or ()), set(arms or ())
    motions = []
    for preset in sorted(d for d in listdir(bundle_root)
                         if path.isdir(path.join(bundle_root, d, "cameras"))):
        if want_presets and preset not in want_presets:
            continue
        cam_dir = path.join(bundle_root, preset, "cameras")
        caption = _bundle_caption(path.join(bundle_root, preset), preset)
        srcs = _bundle_arm_files(cam_dir)
        found = sorted(srcs)
        order = [a for a in BUNDLE_ARMS if a in found] + [a for a in found
                                                          if a not in BUNDLE_ARMS]
        for arm in order:
            if want_arms and arm not in want_arms:
                continue
            label = f"{preset}/{arm}"
            if not all(token in label for token in variant_filter):
                continue
            c2w, fx, fy = _bundle_arm_poses(srcs[arm], cx_recon)
            row = dict(caption)
            row["bundle"], row["preset"], row["arm"] = True, preset, arm
            # 예측은 자기 focal 을 들고 있다 (s42 fx 2287.88 vs recon 2293.41) — 프러스텀을
            # 소스 K 로 통일해 그리면 그 zoom 차이가 화면에서 사라진다.
            if fx is not None:
                row["fx"], row["fy"] = fx, fy
            variant = str(caption.get("variant_id", ""))
            row["anchor_id"] = variant.split("__")[0] if "__" in variant else ""
            row["path_len_u"] = round(float(np.linalg.norm(
                np.diff(c2w[:, :3, 3], axis=0), axis=-1).sum()), 4)
            motions.append((label, c2w, row))
    assert motions, f"번들 {bundle_root} 에서 {list(variant_filter)} 에 맞는 arm 이 0개다"
    return motions


def load_eval_dirs(specs, entry: str, cx_recon=None):
    """latentcam eval 폴더들 -> `load_bundle` 과 같은 모양 [(label, c2w (F,4,4), row)].

    GT 와 모델 출력을 점군 위에 같이 보려는 경로다 (R103). `specs` 는 `label=<eval_dir>` 목록,
    `entry` 는 data_name (예 `dynpose_<video>_0`). 라벨 `gt` 는 **첫 폴더**의
    `test/<entry>_transforms_ref.json`, 나머지는 폴더마다 `test/<entry>_transforms_pred.json`.
    변환은 번들 JSON arm 과 같은 `_bundle_arm_poses` 를 쓴다 — eval JSON 은 scene world 의
    OpenGL c2w 라 (`render_pred_depth_warp.py` 가 같은 파일을 같은 점군 위에 렌더한다)
    `diag(1,-1,-1,1)` 만 곱하면 cloud.npz 와 같은 OpenCV world 에 앉는다.
    """
    motions = []
    for k, spec in enumerate(specs):
        label, folder = spec.split("=", 1) if "=" in spec else (path.basename(spec.rstrip("/")), spec)
        test = path.join(folder, "test")
        cap = {}
        cap_path = path.join(test, f"{entry}_caption.json")
        if path.isfile(cap_path):
            with open(cap_path, encoding="utf-8") as file:
                cap = {"prompt_camera_with_scene_video": {"concise": str(next(iter(json.load(file).values()), ""))}}
        files = ([("gt", f"{entry}_transforms_ref.json")] if k == 0 else []) + \
            [(label, f"{entry}_transforms_pred.json")]
        for arm, name in files:
            fpath = path.join(test, name)
            if not path.isfile(fpath):
                print(f"[eval_dir] 없음: {fpath}")
                continue
            c2w, fx, fy = _bundle_arm_poses(fpath, cx_recon)
            row = dict(cap)
            row["bundle"], row["preset"], row["arm"] = True, "eval", arm
            row["fx"], row["fy"] = fx, fy
            row["anchor_id"] = "dyn_0"
            row["path_len_u"] = round(float(np.linalg.norm(
                np.diff(c2w[:, :3, 3], axis=0), axis=-1).sum()), 4)
            motions.append((arm, c2w, row))
    assert motions, f"eval 폴더 {list(specs)} 에서 {entry} 를 하나도 못 읽었다"
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


def node_animates(node: dict, mode: str) -> bool:
    """이 노드의 OBB 를 **프레임마다 갈아끼울지**.

    `dyn`(기본) = id/kind 가 `dyn` 이고 track 이 있으면 애니메이션. `moving` = 그래프의
    `moving` 플래그만 믿는 예전 동작.

    기본을 `dyn` 으로 둔 이유: `moving` 은 **절대 임계**(`--track_min_drift_u` 0.05)로 정해져서
    작은 물체가 자기 몸 길이의 몇 배를 움직여도 static 으로 떨어진다. snow-dog 의 dog 은
    track drift 0.0318 u 인데 자기 extent 가 0.0103 이라 **3.09 배**를 움직였는데도
    `moving=False` 다. 그러면 뷰어가 시간 median OBB 한 개만 그리고, "박스가 개를 따라가는가"
    를 아예 볼 수 없다 — 그걸 보려고 띄우는 화면인데.

    실측(번들 19 씬): `moving` 노드가 0개인 씬이 5개다 —
    avocado-slice / camera-lens / hike / mountain-hike / snow-dog.
    """
    if not (node.get("track") or {}).get("center_smooth"):
        return False
    if mode == "moving":
        return bool(node.get("moving"))
    return str(node.get("kind", "")) == "dyn" or str(node.get("id", "")).startswith("dyn")


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


def color_ramp(start, end, count: int):
    """start -> end 선형 보간 (count, 3) uint8.

    프러스텀과 경로선이 **같은 램프**를 써야 한다 — 따로 계산하면 downsample(cam_stride) 탓에
    프러스텀 i 의 색과 그 자리를 지나는 선의 색이 어긋나고, 그러면 색이 시간축을 가리킨다는
    약속 자체가 깨진다.
    """
    if count <= 1:
        return np.asarray([start], dtype=np.uint8)
    t = np.linspace(0.0, 1.0, int(count))[:, None]
    mix = np.asarray(start, float)[None] * (1.0 - t) + np.asarray(end, float)[None] * t
    return np.clip(np.rint(mix), 0, 255).astype(np.uint8)


def ramp_segments(ramp: np.ndarray):
    """(F,3) 램프 -> `add_line_segments(colors=...)` 가 먹는 (F-1, 2, 3) per-point 색.

    viser 1.1.0 의 `colors` 는 (N,2,3) 을 받아 선분 양 끝을 각각 칠한다 — 그래서 꺾은선
    하나로 gradation 이 나온다 (선을 프레임별로 쪼갤 필요가 없다).
    """
    return np.stack([ramp[:-1], ramp[1:]], axis=1)


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
    # 번들 arm 은 preset/τ/follow 손잡이가 아니라 **어느 arm 인가**가 첫 정보다. focal 을 같이
    # 적는 이유: 예측마다 focal 이 달라 프러스텀 화각이 다른데, 그걸 모르면 화각 차이가
    # 궤적 차이로 잘못 읽힌다.
    head = ([f"arm         {row.get('arm', '?')}  preset {row.get('preset', '?')}  "
             f"fx {row.get('fx', 0):.1f}",
             f"caption     {(row.get('prompt_camera_with_scene_video') or {}).get('concise', '-')[:96]}"]
            if row.get("bundle") else [])
    lines = [label, *head,
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
    # 프러스텀을 **몇 프레임마다 만들지**. 표시 간격은 GUI `camera frames`/`camera interval`
    # 이 따로 고른다 — 동적 점군이 49프레임을 다 올려두고 슬라이더로 가리는 것과 같은 구조다
    # (만들기를 줄이면 슬라이더가 그보다 촘촘해질 수 없으므로 기본은 1 = 전부 만든다).
    # 기본이 2 였던 시절의 화면은 `camera interval` 초기값 2 가 그대로 재현한다.
    parser.add_argument("--cam_stride", default=1, type=int)
    # 프러스텀 세로 화각(도). 카메라 K 와 무관한 고정값이다 (위 상수 주석 참고).
    parser.add_argument("--cam_fov_deg", default=FRUSTUM_FOV_DEG, type=float,
                        help=f"프러스텀 세로 화각 (기본 {FRUSTUM_FOV_DEG:.0f}도, GUI view > camera vfov)")
    parser.add_argument("--cam_fov_src", action="store_true",
                        help="프러스텀 화각을 카메라 자기 K 로 (예전 동작, 망원 씬에서 바늘이 된다)")
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
    # d221 번들 scene 폴더. 주면 cloud.npz / scene_graph.json 도 **그 폴더에서** 읽는다
    # (번들은 cloud 를 symlink, graph 는 스냅샷으로 들고 있어 그 세대의 씬이 재현된다).
    # `--banks` 와 같이 줄 수 있다 — 그때 뱅크는 여전히 `--out/<video>` 에서 찾는다.
    parser.add_argument("--bundle", default="", type=str,
                        help="d221 번들 scene 폴더 (예: results/20260921_d221_bundles/bmx-bumps)")
    parser.add_argument("--bundle_presets", nargs="*", default=[], type=str,
                        help="이 preset 만 (기본 전부)")
    parser.add_argument("--bundle_arms", nargs="*", default=[], type=str,
                        help="이 arm 만 (gt s42 s1234 s2026 gendop, 기본 전부)")
    # [R103] latentcam eval 폴더 (`label=<dir>`, 여러 개). 첫 폴더의 ref 가 `gt`, 폴더마다 pred.
    # `--entry` 는 data_name (예 dynpose_<video>_0). 둘 다 없으면 예전 동작.
    parser.add_argument("--eval_dir", nargs="*", default=[], type=str)
    parser.add_argument("--entry", default="", type=str)
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
    # 지면 격자는 **만들기**(`--no_ground_grid` 로 끔)와 **처음 보이기**(`--ground_on`)가
    # 다른 손잡이다. 기본은 "만들되 숨김" — 격자는 씬 전체를 덮는 선 26개라 (`half` 가 카메라
    # 와 노드를 다 포함하게 잡힌다) 정작 보려는 카메라·OBB 위에 얹혀 화면을 가린다. 중력축이
    # 틀어졌는지 확인할 때만 켠다. 아예 안 만들면 GUI 체크박스도 비활성이 되므로 기본은 만든다.
    parser.add_argument("--ground_grid", dest="ground_grid", action="store_true", default=True)
    parser.add_argument("--no_ground_grid", dest="ground_grid", action="store_false")
    parser.add_argument("--ground_on", action="store_true",
                        help="지면 격자를 처음부터 켠다 (기본 꺼짐, GUI view > ground grid)")
    # 색 (전부 `"R,G,B"` 0~255, 빈 문자열이면 위 상수). 카메라 두 계열은 띄운 뒤 GUI
    # `camera colors` 로도 바꾼다 — 여기 플래그는 **기동 시 기본값**이고 피커가 그 값을 읽는다.
    # OBB 는 GUI 를 안 준다: 동적 박스가 노드×프레임이라 snowboard 만 245개고, viser 1.1.0
    # 선 handle 은 색을 바꿔 끼울 수가 없어 드래그 한 번에 245개를 다시 그려야 한다.
    parser.add_argument("--src_color", default="", type=str)
    parser.add_argument("--src_now_color", default="", type=str)
    parser.add_argument("--plan_color", default="", type=str,
                        help="target 단색 (gradation 을 끈 경우에만 쓰인다)")
    parser.add_argument("--plan_now_color", default="", type=str)
    # target 궤적 gradation (기본 켬). 색이 시간축을 가리키므로 `target start` 가 frame 0,
    # `target end` 가 마지막 프레임이다. 예전 단색 동작은 `--no_plan_gradient` + `--plan_color`.
    parser.add_argument("--plan_gradient", dest="plan_gradient", action="store_true", default=True)
    parser.add_argument("--no_plan_gradient", dest="plan_gradient", action="store_false")
    parser.add_argument("--plan_color_start", default="", type=str,
                        help="target gradation 의 frame 0 색 (기본 빨강)")
    parser.add_argument("--plan_color_end", default="", type=str,
                        help="target gradation 의 마지막 프레임 색 (기본 파랑)")
    parser.add_argument("--obb_color_dyn", default="", type=str)
    parser.add_argument("--obb_color_static", default="", type=str)
    # 경로선·probe 는 기본 꺼짐이다. 둘 다 프러스텀보다 눈에 먼저 들어와서, 정작 보려는
    # "카메라가 어디서 어디를 보나"를 가린다. 예전 동작(둘 다 켜짐)은 이 플래그로 돌아온다.
    parser.add_argument("--paths_on", action="store_true",
                        help="카메라 경로선을 처음부터 켠다 (기본 꺼짐, GUI view > paths)")
    parser.add_argument("--probe_on", action="store_true",
                        help="probe 카메라(프러스텀+gizmo)를 처음부터 켠다 (기본 꺼짐)")
    parser.add_argument("--now_cams", action="store_true",
                        help="'현재 프레임' 1.6배 프러스텀을 그린다 (기본 꺼짐)")
    # 소스 카메라(프러스텀 + 경로선)는 **기본 꺼짐**. 보려는 건 거의 항상 target 궤적인데,
    # 소스 49대가 원점 근처에 뭉쳐 앉아 target 프러스텀을 덮는다. 예전처럼 처음부터 켜려면
    # `--source_on` (띄운 뒤에는 GUI view > source cameras).
    parser.add_argument("--source_on", action="store_true",
                        help="소스 카메라를 처음부터 켠다 (기본 꺼짐, GUI view > source cameras)")
    parser.add_argument("--pin_palette", action="store_true",
                        help="pin 색을 PIN_COLORS 순환으로 (기본: target 피커의 현재 색)")
    # 어떤 노드를 프레임마다 갈아끼울지 (위 `node_animates` 주석이 근거다).
    parser.add_argument("--obb_anim", default="dyn", choices=("dyn", "moving"),
                        help="dyn(기본)=dyn_* 이고 track 있으면 애니메이션 / "
                             "moving=그래프 moving 플래그만 (예전 동작)")
    # 기본이 `dyn_0` 이다 (예전엔 전부). 씬 하나에 노드가 6~16개씩 있어 전부 켜면 어느 박스가
    # 지금 보는 subject 인지 못 고르고, 큰 정적 박스(snow-dog forest 6.36 m)가 작은 subject
    # 박스를 통째로 덮는다. `--obb_node all` 로 예전 동작, `--obb_node dyn_1` 로 다른 노드.
    # 그 id 가 그래프에 없으면 `all` 로 떨어진다 (노드 이름이 씬마다 다르므로 죽지 않게).
    parser.add_argument("--obb_node", default="dyn_0", type=str,
                        help="이 노드 id 의 OBB 만 띄운다 (기본 dyn_0, 전부는 `all`)")
    parser.add_argument("--viewer_root", default=VIEWER_ROOT_DEFAULT, type=str)
    args = parser.parse_args()
    require_viser_thickness()          # cloud.npz(1.4 GB) 를 읽기 전에 env 를 가른다

    src_color = parse_rgb(args.src_color, SRC_COLOR)
    src_now_color = parse_rgb(args.src_now_color, SRC_NOW_COLOR)
    plan_color = parse_rgb(args.plan_color, PLAN_COLOR)
    plan_now_color = parse_rgb(args.plan_now_color, PLAN_NOW_COLOR)
    plan_start = parse_rgb(args.plan_color_start, PLAN_COLOR_START)
    plan_end = parse_rgb(args.plan_color_end, PLAN_COLOR_END)
    obb_dyn_color = parse_rgb(args.obb_color_dyn, OBB_DYN_COLOR)
    obb_static_color = parse_rgb(args.obb_color_static, OBB_STATIC_COLOR)

    add_frustums, gl2cv = import_frustum_helpers(args.viewer_root)

    # 번들을 주면 씬도 번들에서 읽는다 — 번들의 graph 는 그 세대의 **스냅샷**이라, out/ 을
    # 다시 구운 뒤에도 그때 카메라를 낳은 그래프와 같이 볼 수 있다.
    bundle_root = path.abspath(args.bundle) if args.bundle else ""
    if bundle_root and args.video == parser.get_default("video"):
        # 번들 폴더 이름이 곧 씬 이름이다. `--video` 를 안 준 채 뱅크까지 얹으려 할 때
        # 기본값(snowboard)으로 엉뚱한 씬을 열지 않게 여기서 맞춘다.
        args.video = path.basename(bundle_root)
    scene_root = bundle_root or path.join(args.out, args.video)
    cloud_path = path.join(scene_root, "cloud.npz")
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
    # 세 번째 원소가 **프레임 번호**다. handle 순서만으로는 downsample 때문에 어느 프레임인지
    # 알 수 없고, 그러면 표시 간격·색 램프를 어긋나게 칠한다. `-1` 은 "현재 프레임" 프러스텀
    # 처럼 프레임이 고정되지 않은 것 (간격 필터를 통과시킨다).
    src_cams = [(h, 1.0, f) for h, f in zip(add_frustums(
        server, "/cam_source", src_gl, float(intrinsics[0, 0, 0]),
        float(intrinsics[0, 1, 1]), width, height, src_color, cam_scale,
        downsample=args.cam_stride), range(0, num_frames, max(1, int(args.cam_stride))))]
    # "현재 프레임" 1.6배 프러스텀은 기본으로 안 그린다 (`--now_cams` 로 켠다). 프레임을
    # 가리키는 큰 프러스텀이 같은 자리의 일반 프러스텀 위에 겹쳐 앉아, 궤적을 훑을 때
    # 카메라가 두 대인 것처럼 보였다. 빈 리스트로 두면 아래 갱신·색칠이 전부 no-op 이 된다.
    src_now = list(add_frustums(
        server, "/cam_source_now", src_gl[:1], float(intrinsics[0, 0, 0]),
        float(intrinsics[0, 1, 1]), width, height, src_now_color,
        cam_scale * 1.6)) if args.now_cams else []
    src_cams += [(h, 1.6, -1) for h in src_now]

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
    # `src` 는 소스 카메라 계열 전체(프러스텀 + 경로선)의 게이트다. 경로선을 `show` 하나로만
    # 묶으면 소스 카메라를 끈 상태에서 `paths` 를 켰을 때 **끈 카메라의 궤적**만 화면에 남는다.
    lw = {"path": 1.0, "show": bool(args.paths_on), "src": bool(args.source_on)}

    def draw_line(key, name, segments, color, thickness):
        drop_line(key)
        line_base[key] = float(thickness)
        lines[key] = server.scene.add_line_segments(
            name, segments, colors=color, thickness=float(thickness) * path_lw(),
            visible=line_vis(key))
        return lines[key]

    def drop_line(key):
        old = lines.pop(key, None)
        line_base.pop(key, None)
        if old is not None:
            old.remove()

    def path_lw():
        return float(lw["path"])

    def line_vis(key):
        """경로선 하나의 표시 여부 = `paths` 체크박스 AND 그 계열 게이트."""
        return bool(lw["show"]) and (bool(lw["src"]) if key == "src" else True)

    # `handle.line_width = v` 를 쓰면 안 된다 — viser 1.1.0 에서 그건 `thickness` 의 **폐기된
    # 별칭**이고, 대입하는 순간 `thickness_units` 를 `"screen"` 으로 못 박는다 (예전 line_width
    # 가 픽셀이었으니 그 뜻을 지키려고). world 0.02 로 만든 선에 0.02 를 넣으면 0.02 **픽셀**이
    # 돼서 화면에서 사라진다. `thickness` 에 직접 넣으면 단위가 world 그대로다.
    def apply_path_lw():
        for key, handle in lines.items():
            handle.thickness = line_base[key] * path_lw()

    def apply_path_vis():
        for key, handle in lines.items():
            handle.visible = line_vis(key)

    def shown_cam_frames():
        """프러스텀을 띄울 프레임 집합. 동적 점군/OBB 와 **같은 모양의 손잡이**다.

        `range start`/`range end` 는 OBB 쪽과 **공유**한다 — "지금 보는 구간"은 카메라와 물체에
        같은 뜻이고, 따로 두면 두 슬라이더를 매번 맞춰야 한다. 간격만 카메라용으로 따로 둔다
        (프러스텀은 49개가 겹치면 점군보다 먼저 화면을 덮어서 OBB 보다 더 성기게 보는 일이 많다).
        현재 프레임은 구간 밖이어도 늘 포함한다 (frame 슬라이더를 밀었는데 화면이 비면 슬라이더가
        고장난 것처럼 보인다).
        """
        mode = str(gui_cam_mode.value)
        frame = int(gui_frame.value)
        if mode == "current frame":
            return {frame}
        step = max(1, int(gui_cam_step.value))
        lo, hi = (0, num_frames - 1) if mode == "all frames" else \
            (min(int(gui_span_lo.value), int(gui_span_hi.value)),
             max(int(gui_span_lo.value), int(gui_span_hi.value)))
        return {i for i in range(lo, hi + 1) if (i - lo) % step == 0} | {frame}

    def apply_cam_frames():
        """프러스텀 표시 여부 = (간격 필터) AND (계열 게이트).

        프레임이 `-1` 인 handle("현재 프레임" 1.6배 프러스텀)은 간격을 통과시킨다 — 그건 frame
        슬라이더가 가리키는 자리를 표시하는 것이라 간격과 무관하다.
        """
        frames = shown_cam_frames()
        for handle, _ratio, frame in src_cams:
            handle.visible = bool(lw["src"]) and (frame < 0 or frame in frames)
        pin_cams = [pair for pin in pins.values() for pair in pin["cams"]]
        for handle, _ratio, frame in state["cams"] + pin_cams:
            handle.visible = frame < 0 or frame in frames

    def apply_src_vis():
        """소스 프러스텀 + 소스 경로선을 한 손잡이로 껐다 켠다 (`--source_on` / GUI 체크박스)."""
        apply_cam_frames()
        apply_path_vis()

    # 소스 카메라 경로도 선으로 — plan 이 떠는지 판단하려면 "원래 소스는 얼마나 떠는가"가
    # 있어야 한다. snowboard 소스는 |jerk| p95 가 plan 의 8배다.
    draw_line("src", "/src_path", path_segments(cam_c2w[:, :3, 3]), src_color, cam_scale * 0.08)
    # 기본 꺼짐 (`add_frustums` 는 visible 인자를 안 받는다). `apply_cam_frames()` 를 못 쓰는
    # 이유는 그게 GUI 위젯을 읽는데 위젯이 아직 없기 때문이다 — 간격 필터는 아래 GUI 생성
    # 뒤에 한 번 더 돈다.
    for _h, _r, _f in src_cams:
        _h.visible = bool(lw["src"])
    src_jerk = jerk_px(cam_c2w[:, :3, 3], z_med, focal)

    banks = list(args.banks) or ([args.bank] if args.bank else [])
    # 번들을 앞에 둔다 — 둘을 같이 올리는 건 "예측이 뱅크 변이와 얼마나 다른가"를 볼 때이고,
    # 그때 슬라이더 0번은 번들 gt 여야 한다.
    # `cx_recon` 은 JSON arm 전용이다 — 예측 JSON 이 절반 해상도로 적혀 있어 focal 을 recon
    # 픽셀로 되돌려야 프러스텀 화각이 npz arm 과 같은 눈금에 앉는다.
    motions = load_bundle(bundle_root, args.bundle_presets, args.bundle_arms,
                          args.variant, float(intrinsics[0, 0, 2])) if bundle_root else []
    motions += (load_eval_dirs(args.eval_dir, args.entry, float(intrinsics[0, 0, 2]))
                if args.eval_dir else [])
    motions += load_motions(args.out, args.video, banks, args.variant) if banks else []
    # `scene_root` 를 쓴다 (`out/<video>` 를 직접 조립하면 안 된다) — `--bundle` 만 주고 띄우면
    # `--video` 는 기본값 그대로라 없는 경로가 나오고, OBB·subject track·지면 격자가 전부
    # 조용히 사라진다. `--bundle` 없이 쓰면 `scene_root == out/<video>` 라 예전과 같다.
    graph_path = path.join(scene_root, "scene_graph.json")
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

    def obb_entry(node_id, name, segments, moving, visible):
        return {"node": str(node_id), "name": name, "seg": segments, "moving": moving,
                "handle": None, "version": -1, "visible": bool(visible)}

    if args.obb and path.isfile(graph_path):
        with open(graph_path, encoding="utf-8") as file:
            graph = json.load(file)
        T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
        T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
        for node in graph["nodes"]:
            moving = node_animates(node, args.obb_anim)
            extent = node["obb"]["extent"]
            if moving:
                centers = np.asarray(node["track"]["center_smooth"], dtype=float)
                yaws = np.asarray(node["track"]["yaw"], dtype=float)
                for f in range(min(num_frames, len(centers))):
                    obb_dyn[f].append(obb_entry(
                        node["id"], f"/obb/{node['id']}/f{f:03d}",
                        obb_segments_world(centers[f], extent, yaw_to_R(yaws[f]), T_wg),
                        True, f == 0))
                anchor_g = centers[0]
            else:
                obb_static.append(obb_entry(
                    node["id"], f"/obb/{node['id']}",
                    obb_segments_world(node["obb"]["center"], extent,
                                       np.asarray(node["obb"]["R"], dtype=float), T_wg),
                    False, True))
                anchor_g = np.asarray(node["obb"]["center"], dtype=float)
            top = np.asarray(anchor_g, float) + np.array([0.0, 0.0, float(extent[2]) / 2 + 0.01])
            obb_labels.append((str(node["id"]), server.scene.add_label(
                f"/obb_label/{node['id']}", f"{node['id']} {node.get('label', '')}",
                position=(top @ T_wg[:3, :3].T + T_wg[:3, 3]).astype(np.float32))))
            graph_rows.append((node["id"], node.get("label", ""), moving,
                               float(np.max(np.abs(np.asarray(extent, float))))))
        if args.ground_grid:
            cams_g = cam_c2w[:, :3, 3] @ T_gw[:3, :3].T + T_gw[:3, 3]
            nodes_g = np.asarray([n["obb"]["center"] for n in graph["nodes"]], dtype=float)
            span = np.concatenate([cams_g[:, :2], nodes_g[:, :2]], axis=0)
            half = float(max(np.ptp(span, axis=0).max(), 1e-3)) * 1.2
            ground_handle = server.scene.add_line_segments(
                "/ground_grid",
                ground_grid_world(float(graph["ground"]["ground_z"]), T_wg,
                                  span.mean(axis=0), half),
                colors=(110, 120, 140), thickness=cam_scale * 0.03,
                visible=bool(args.ground_on))

    # OBB 가 선 굵기보다 작으면 박스가 아니라 점으로 보인다. snow-dog 의 dog 은 extent
    # 0.0103 인데 기본 굵기가 0.0110 이라 **자기 크기의 1.07 배** 굵기로 그려졌다 — 그러면
    # "박스가 안 보인다" 로 읽히고 원인이 데이터(cm 단위 OBB)라는 게 안 드러난다.
    _obb_lw0 = cam_scale * 0.20
    _thin = [(nid, ext) for nid, _, _, ext in graph_rows if ext < _obb_lw0]
    if _thin:
        print(f"[obb] 선 굵기({_obb_lw0:.4f} world) 보다 작은 OBB {len(_thin)}개 — 박스가 점으로 "
              f"보인다: " + ", ".join(f"{nid} extent {ext:.4f}" for nid, ext in _thin)
              + "\n      GUI obb > OBB thickness 를 내리거나 labels 로 위치를 읽는다.")

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

    def node_on(node_id: str):
        """`obb > node` 드롭다운이 고른 노드인가. `all` 이면 전부 통과."""
        pick = str(gui_obb_node.value)
        return pick == "all" or pick == str(node_id)

    def obb_show(entry, visible: bool):
        """보이게 하는 순간에만 stale 을 갚는다 — 249개를 매 드래그마다 다시 그리지 않으려고."""
        visible = bool(visible) and node_on(entry["node"])
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
        for handle, ratio, _f in src_cams + state["cams"] + pin_cams:
            handle.scale = value * ratio

    def apply_cam_fov():
        """프러스텀 화각을 고정값으로 덮는다.

        `add_frustums` 를 고치지 않고 **만든 뒤 `.fov` 를 대입**한다 — 그 함수는 GL↔CV 규약의
        단일 출처(latentcam 뷰어)라 시그니처를 늘리고 싶지 않고, `fov` 는 갱신되는 프로퍼티라
        다시 그릴 필요도 없다 (probe 가 이미 같은 방식으로 슬라이더를 먹는다).
        `--cam_fov_src` 면 아무것도 하지 않아 add_frustums 가 계산한 자기 화각이 남는다.
        """
        if args.cam_fov_src:
            return
        fov = float(np.radians(float(gui_cam_fov.value)))
        pin_cams = [pair for pin in pins.values() for pair in pin["cams"]]
        for handle, _r, _f in src_cams + state["cams"] + pin_cams:
            handle.fov = fov

    def apply_cam_lw():
        """프러스텀 선 굵기. 크기(scale)와 **다른 축**이다 — 작은 프러스텀을 굵게 그려야
        점군에 안 묻히는 씬이 있고, 반대로 큰 프러스텀은 가늘어야 뒤가 보인다.
        `thickness` 는 갱신되는 프로퍼티라 다시 그릴 필요가 없다 (색과 다르다).
        **`line_width` 에 넣으면 안 된다** — 폐기된 별칭이라 단위가 screen(픽셀) 로 못 박혀
        world 0.02 가 0.02 픽셀이 되고, 프러스텀이 통째로 안 보인다."""
        value = float(gui_cam_lw.value)
        pin_cams = [pair for pin in pins.values() for pair in pin["cams"]]
        for handle, _r, _f in src_cams + state["cams"] + pin_cams:
            handle.thickness = value
        for probe in probes:
            probe["cam"].thickness = value

    def cam_fx_fy(row):
        """프러스텀 화각. 번들 arm 은 자기 focal 을 쓰고, 뱅크는 소스 K 를 쓴다."""
        return (float(row.get("fx") or focal),
                float(row.get("fy") or intrinsics[0, 1, 1]))

    def plan_colors():
        """(frame 0 색, 현재 프레임 색) target 색. 피커가 단일 출처다.

        gradation 이 꺼져 있으면 첫 값이 **단색**으로 쓰인다 — 그래서 `--no_plan_gradient` 로
        띄우면 `target start` 피커가 예전 `target (pred)` 와 정확히 같은 역할을 한다.
        """
        return (tuple(int(c) for c in gui_plan_color.value),
                tuple(int(c) for c in gui_plan_now_color.value))

    def plan_ramp(count: int):
        """target 프레임 수만큼의 색 (count, 3). gradation 이 꺼지면 start 색 단색."""
        start = plan_colors()[0]
        if not bool(gui_plan_grad.value):
            return np.repeat(np.asarray([start], np.uint8), max(int(count), 1), axis=0)
        return color_ramp(start, tuple(int(c) for c in gui_plan_end_color.value), count)

    def stride_frames(count: int):
        """add_frustums 가 실제로 만든 프러스텀의 **프레임 번호**. downsample 탓에 0,2,4,... 라
        handle 순서만으로는 어느 프레임인지 알 수 없고, 그러면 램프를 어긋나게 칠한다."""
        return list(range(0, int(count), max(1, int(args.cam_stride))))

    def paint_ramp(pairs, ramp):
        """`[(handle, frame)]` 을 램프 색으로. `add_frustums` 는 색을 하나만 받으므로 만든 뒤
        칠한다 (`.color` 대입은 갱신되는 프로퍼티라 다시 그릴 필요가 없다 — 경로선과 반대다).
        프레임을 handle 과 **같이** 받는 이유: 순서로 되짚으면 downsample 에서 어긋난다."""
        for handle, frame in pairs:
            handle.color = tuple(int(c) for c in ramp[min(max(frame, 0), len(ramp) - 1)])

    def select(index: int):
        label, plan_c2w, row = motions[int(index)]
        for handle in state["handles"]:
            handle.remove()
        plan_gl = plan_c2w @ gl2cv
        # 색은 상수가 아니라 **피커의 현재 값**을 읽는다 — 안 그러면 motion 을 갈아끼우는 순간
        # 사용자가 고른 색이 기본색으로 되돌아간다.
        color, now_color = plan_colors()
        # 램프는 **변이의 프레임 수**로 만든다 (num_frames 로 굳히면 F 가 다른 뱅크에서 색이
        # 끝까지 안 가거나 잘린다).
        ramp = plan_ramp(len(plan_c2w))
        fx_v, fy_v = cam_fx_fy(row)
        plan_cams = list(zip(add_frustums(server, "/cam_plan", plan_gl, fx_v, fy_v,
                                          width, height, color,
                                          cam_scale, downsample=args.cam_stride),
                             stride_frames(len(plan_c2w))))
        paint_ramp(plan_cams, ramp)
        handles = [h for h, _f in plan_cams]
        # 경로선은 handles 에 안 담는다 — draw_line 이 키 하나로 이전 것을 지우므로, 여기에도
        # 담으면 갈아끼울 때 같은 handle 을 두 번 remove 하게 된다.
        draw_line("plan", "/plan_path", path_segments(plan_c2w[:, :3, 3]),
                  ramp_segments(ramp), cam_scale * 0.10)
        now = list(add_frustums(server, "/cam_plan_now", plan_gl[:1], fx_v, fy_v,
                                width, height, now_color,
                                cam_scale * 1.6)) if args.now_cams else []
        track = tracks.get(str(row.get("anchor_id", "")))
        drop_line("track")
        if track is not None and len(track) > 1:
            draw_line("track", "/subject_track", path_segments(track), (60, 220, 90),
                      cam_scale * 0.08)
        state.update(handles=handles + now, now=now, c2w=plan_c2w, label=label,
                     cams=[(h, 1.0, f) for h, f in plan_cams] + [(h, 1.6, -1) for h in now])
        gui_info.value = motion_report(label, row, plan_c2w, track, z_med, focal, src_jerk)
        apply_cam_scale()
        apply_cam_lw()
        apply_cam_fov()
        apply_cam_frames()
        refresh()

    def pin_node(label: str):
        """라벨 -> viser 노드 경로. `/` 가 계층 구분자라 라벨의 `bank/variant` 를 그대로 못 쓴다."""
        return "/pin/" + "".join(c if c.isalnum() or c in "_-" else "_" for c in label)

    def add_pin(label: str, palette: bool = False):
        """motion 하나를 고정 색으로 그려 두고 활성 motion 이 바뀌어도 남긴다.

        `palette` 는 **기동 `--pin` 전용**이다. 피커 스냅샷은 "방금 그 색으로 보던 궤적이 pin
        하는 순간 튀지 않게" 하려는 것인데, 기동 시엔 피커를 바꿀 틈이 없어 여러 개를 주면
        전부 같은 색이 된다 (실측: `--pin crane_up/gt crane_up/s42` 둘 다 (255,40,40)) —
        그러면 여러 대를 구분한다는 pin 의 목적 자체가 깨진다. GUI 버튼은 스냅샷 그대로다.
        """
        if label in pins or len(pins) >= int(args.max_pins):
            return
        index = [m[0] for m in motions].index(label)
        _, plan_c2w, row = motions[index]
        # pin 은 **단색으로 남긴다** — gradation 은 "궤적의 어디쯤"(시간)을 가리키는 축이고
        # pin 색은 "어느 변이"(정체)를 가리키는 축이다. pin 까지 램프로 칠하면 여러 대를
        # 구분한다는 pin 의 존재 이유가 사라진다 (전부 빨강->파랑으로 똑같이 보인다).
        # 기본은 **지금 target start 피커에 들어있는 색**이다 (`--pin_palette` 로 순환 팔레트).
        # 순환은 "여러 대를 구분한다"가 목적이었는데, 실제로는 방금 고른 색으로 보던 궤적이
        # pin 하는 순간 엉뚱한 색으로 바뀌어 대조가 끊겼다. 스냅샷이라 피커를 바꾼 뒤 다시
        # pin 하면 그 pin 만 새 색을 갖는다 — 구분은 사용자가 직접 준다.
        color = (PIN_COLORS[pin_seq["n"] % len(PIN_COLORS)]
                 if (args.pin_palette or palette) else plan_colors()[0])
        pin_seq["n"] += 1
        node, plan_gl = pin_node(label), plan_c2w @ gl2cv
        fx_v, fy_v = cam_fx_fy(row)
        cams = list(zip(add_frustums(server, node + "/cam", plan_gl, fx_v, fy_v,
                                     width, height, color,
                                     cam_scale, downsample=args.cam_stride),
                        stride_frames(len(plan_c2w))))
        # 현재 프레임 프러스텀만 1.6배로 크게 — pin 을 여러 대 켜면 선이 엉켜서 "이 변이가 지금
        # 어디를 보고 있나"를 경로만으로는 못 읽는다. 같은 색이라 소속은 유지된다.
        now = list(add_frustums(
            server, node + "/now", plan_gl[:1], fx_v, fy_v,
            width, height, color, cam_scale * 1.6)) if args.now_cams else []
        # 경로선은 `lines` 로 관리한다 (굵기 슬라이더가 여기만 본다). pin["handles"] 에는 안
        # 담는다 — 담으면 remove_pin 이 drop_line 과 겹쳐 같은 handle 을 두 번 지운다.
        draw_line(("pin", label), node + "/path", path_segments(plan_c2w[:, :3, 3]), color,
                  cam_scale * 0.10)
        pins[label] = {"handles": [h for h, _f in cams] + now, "now": now, "c2w": plan_c2w,
                       "cams": [(h, 1.0, f) for h, f in cams] + [(h, 1.6, -1) for h in now],
                       "color": color, "row": row}
        apply_cam_scale()
        apply_cam_lw()
        apply_cam_fov()
        apply_cam_frames()

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
        # 프러스텀 표시 간격. OBB 와 같은 모양의 손잡이지만 **간격만 따로**다 — 프러스텀은
        # 49개가 겹치면 점군보다 먼저 화면을 덮어 OBB 보다 성기게 보는 일이 많다.
        # 초기값 `interval` + 2 는 `--cam_stride 2` 가 기본이던 시절의 화면과 같다.
        gui_cam_mode = server.gui.add_dropdown(
            "camera frames", options=["interval", "all frames", "current frame"],
            initial_value="interval")
        gui_cam_step = server.gui.add_slider("camera interval (every N)", min=1,
                                             max=max(2, num_frames // 2), step=1,
                                             initial_value=2)
        gui_ground = server.gui.add_checkbox("ground grid", initial_value=bool(args.ground_on),
                                             disabled=ground_handle is None)
        gui_size = server.gui.add_slider("point size", min=point_size * 0.25, max=point_size * 4.0,
                                         step=point_size * 0.05, initial_value=point_size)
        gui_cam = server.gui.add_slider("camera size", min=cam_scale * 0.1, max=cam_scale * 10.0,
                                        step=cam_scale * 0.05, initial_value=cam_scale)
        # 프러스텀 선 굵기(world 단위). viser 기본 0.02 를 초기값으로 둔다.
        # 화각은 크기(scale)·굵기(thickness)와 **또 다른 축**이다 — 작은 프러스텀도 화각이
        # 좁으면 바늘이라 자리를 못 읽는다. `--cam_fov_src` 면 카메라 자기 화각을 쓰므로 비활성.
        gui_cam_fov = server.gui.add_slider("camera vfov (deg)", min=10.0, max=120.0, step=1.0,
                                            initial_value=float(args.cam_fov_deg),
                                            disabled=bool(args.cam_fov_src))
        gui_cam_lw = server.gui.add_slider("camera thickness", min=0.002,
                                           max=max(0.05, cam_scale), step=0.002,
                                           initial_value=min(0.02, max(0.05, cam_scale)))
        # 경로선 굵기 **배율**. 소스 0.08 / plan 0.10 처럼 원래 다른 값을 쓰던 비율을 유지한다.
        # 경로선 전체(소스·target·pin·subject track). 기본 꺼짐 — 49프레임 꺾은선이
        # 프러스텀보다 굵어 카메라 자세를 덮는다. 굵기 슬라이더는 켠 상태에서만 의미가 있다.
        # 소스 카메라 계열(프러스텀 + 경로선) 전체 게이트. 기본 꺼짐 — 소스 49대가 원점
        # 근처에 뭉쳐 앉아 정작 보려는 target 궤적을 덮는다 (`--source_on` 으로 예전 동작).
        gui_src_on = server.gui.add_checkbox("source cameras",
                                             initial_value=bool(args.source_on))
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
        # 노드 하나만 보기. 씬 하나에 노드가 9개씩 있어 전부 켜면 어느 박스가 지금 보는
        # subject 인지 못 고른다. `--obb_node dyn_0` 으로 기동 시 고정할 수도 있다.
        _node_ids = [row[0] for row in graph_rows]
        gui_obb_node = server.gui.add_dropdown(
            "node", options=["all"] + _node_ids,
            initial_value=(args.obb_node if args.obb_node in _node_ids else "all"),
            disabled=not _node_ids)
        # 라벨 기본 꺼짐 — 노드가 9개면 글자가 서로 겹쳐 읽히지도 않으면서 박스를 가린다.
        gui_obb_labels = server.gui.add_checkbox("labels", initial_value=False,
                                                 disabled=not obb_labels)
        gui_obb_dyn_color = server.gui.add_rgb("dynamic color", initial_value=obb_dyn_color,
                                               disabled=not any(obb_dyn))
        gui_obb_stat_color = server.gui.add_rgb("static color", initial_value=obb_static_color,
                                                disabled=not obb_static)
        # cam_scale 배율이다 (world 절대값이 아니라) — 씬마다 scale 이 100배 다르다.
        gui_obb_lw = server.gui.add_slider("OBB thickness", min=0.01, max=0.60, step=0.01,
                                           initial_value=0.20, disabled=not _has_obb)

    with server.gui.add_folder("camera colors"):
        # gt(소스) / pred(활성 target) 두 계열. 기본색이 씬에 따라 안 보일 때가 있어서 손잡이를
        # 둔다 — 회색 소스는 밝은 점군에 묻히고, 주황 target 은 노을·모래 씬에서 배경과 붙는다.
        # 계열마다 피커가 둘인 이유: "현재 프레임" 프러스텀이 크기(1.6배)**와 색**으로 구분되는데,
        # 하나로 합치면 frame 슬라이더를 밀 때 어느 것이 지금인지 다시 못 읽는다.
        # 초기값은 `--src_color` 등 기동 플래그를 그대로 받는다. now 피커는 `--now_cams`
        # 없이는 가리킬 프러스텀이 없어 비활성이다 (켜 두면 "눌러도 아무 일도 안 난다").
        gui_src_color = server.gui.add_rgb("source (gt)", initial_value=src_color)
        gui_src_now_color = server.gui.add_rgb("source now", initial_value=src_now_color,
                                               disabled=not args.now_cams)
        # target 은 피커가 둘이다 — 색이 **시간축**을 가리키기 때문이다 (start = frame 0,
        # end = 마지막 프레임). `target gradation` 을 끄면 start 하나만 단색으로 쓰이고,
        # 그때 초기값은 `--plan_color` (예전 주황) 라 예전 화면이 그대로 재현된다.
        gui_plan_grad = server.gui.add_checkbox("target gradation",
                                                initial_value=bool(args.plan_gradient),
                                                disabled=not motions)
        gui_plan_color = server.gui.add_rgb(
            "target start", initial_value=(plan_start if args.plan_gradient else plan_color),
            disabled=not motions)
        gui_plan_end_color = server.gui.add_rgb(
            "target end", initial_value=plan_end,
            disabled=not (motions and args.plan_gradient))
        gui_plan_now_color = server.gui.add_rgb("target now", initial_value=plan_now_color,
                                                disabled=not (motions and args.now_cams))

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
        for node_id, handle in obb_labels:
            handle.visible = (show_obb and bool(gui_obb_labels.value) and node_on(node_id))
        # 동적 OBB 는 점군과 **같은 규칙**으로 켠다 — 박스만 전 프레임 켜두면 박스가 점군보다
        # 앞선 프레임에 있어도 어긋난 걸 못 알아챈다.
        show_dyn_obb = show_obb and bool(gui_obb_dyn_on.value)
        for i, entries in enumerate(obb_dyn):
            for entry in entries:
                obb_show(entry, show_dyn_obb and i in frames)
        if ground_handle is not None:
            ground_handle.visible = bool(gui_ground.value)
        # 프러스텀도 frame / range 슬라이더를 따른다 (`current frame`·`interval` 모드).
        apply_cam_frames()
        if src_now:
            src_now[0].wxyz, src_now[0].position = _pose(src_gl[f], gl2cv)
        if state["now"]:
            state["now"][0].wxyz, state["now"][0].position = _pose(
                state["c2w"][f] @ gl2cv, gl2cv)
        for pin in pins.values():
            if pin["now"]:
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
                   gui_obb_node, gui_ground):
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
    gui_cam_fov.on_update(lambda _: apply_cam_fov())
    gui_cam_mode.on_update(lambda _: apply_cam_frames())
    gui_cam_step.on_update(lambda _: apply_cam_frames())
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
    # 구분이 같은 한 곳에서 나와야 둘이 어긋나지 않는다. `--now_cams` 가 꺼져 있으면
    # ratio > 1.0 인 handle 이 애초에 없어 now_color 분기가 그냥 안 타는 것뿐이다.
    def repaint_src(_event=None):
        color = tuple(int(c) for c in gui_src_color.value)
        now_color = tuple(int(c) for c in gui_src_now_color.value)
        for handle, ratio, _f in src_cams:
            handle.color = now_color if ratio > 1.0 else color
        draw_line("src", "/src_path", path_segments(cam_c2w[:, :3, 3]), color, cam_scale * 0.08)

    def repaint_plan(_event=None):
        now_color = plan_colors()[1]
        # end 피커는 gradation 이 켜져 있을 때만 의미가 있다 — 켜 두면 "눌러도 아무 일도
        # 안 난다" 가 되고, 그게 색이 안 먹는 버그처럼 읽힌다.
        gui_plan_end_color.disabled = not (motions and bool(gui_plan_grad.value))
        count = len(state["c2w"]) if state["c2w"] is not None else num_frames
        ramp = plan_ramp(count)
        paint_ramp([(h, f) for h, ratio, f in state["cams"] if ratio <= 1.0], ramp)
        for handle, ratio, _f in state["cams"]:
            if ratio > 1.0:
                handle.color = now_color
        if state["c2w"] is not None:
            draw_line("plan", "/plan_path", path_segments(state["c2w"][:, :3, 3]),
                      ramp_segments(ramp), cam_scale * 0.10)

    gui_src_color.on_update(repaint_src)
    gui_src_now_color.on_update(repaint_src)
    gui_plan_color.on_update(repaint_plan)
    gui_plan_end_color.on_update(repaint_plan)
    gui_plan_grad.on_update(repaint_plan)
    gui_plan_now_color.on_update(repaint_plan)

    @gui_src_on.on_update
    def _(_event):
        lw["src"] = bool(gui_src_on.value)
        apply_src_vis()

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

    # 소스 프러스텀은 GUI 보다 **먼저** 만들어지므로 여기서 한 번 덮어야 한다 — motions 가
    # 없는 씬에서는 select() 가 안 돌아 소스만 예전 화각으로 남았다.
    apply_cam_fov()
    apply_cam_frames()
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
            add_pin(label, palette=True)
        gui_pin_info.value = pin_report()
        refresh()
        if not picked:
            print(f"[pin] {args.pin} 에 맞는 변이가 0개 — pin 없이 띄운다")

    rows = [("video", args.video), ("frames", num_frames), ("points total", len(points)),
            ("static shown", f"{len(static_pick):,} / {len(static_idx):,}"),
            ("dynamic/frame", f"{int(np.median(dyn_counts) if dyn_counts else 0):,} (median)"),
            ("z_med frame0", f"{z_med:.4f}"), ("point size", f"{point_size:.5f}"),
            ("camera size", f"{cam_scale:.4f}  (GUI view > camera size 로 조절)"),
            ("camera vfov", (f"카메라 자기 K  (소스 {src_vfov_deg:.1f} deg, --cam_fov_src)"
                             if args.cam_fov_src else
                             f"{float(args.cam_fov_deg):.0f} deg 고정  "
                             f"(소스 K 는 {src_vfov_deg:.1f} deg, GUI view > camera vfov)")),
            ("probe vfov", f"{PROBE_FOV_DEG:.0f} deg  (소스는 {src_vfov_deg:.1f} deg)"),
            ("up", args.up), ("banks", " ".join(banks) or "-"),
            ("bundle", (f"{bundle_root}  (arm 표시순 {'/'.join(BUNDLE_ARMS)})")
                       if bundle_root else "-"),
            ("scene root", scene_root),
            ("obb nodes", " ".join(f"{i}({'dyn' if m else 'stat'})"
                                   for i, _, m, _e in graph_rows) or "-"),
            ("obb anim", f"{args.obb_anim}  "
                         f"({sum(1 for r in graph_rows if r[2])}/{len(graph_rows)} 노드 애니메이션, "
                         f"--obb_anim dyn|moving)"),
            ("motions", len(motions)),
            ("pinned", f"{len(pins)} / {args.max_pins}  (GUI motion > pin current)"),
            ("source jerk p95", f"{src_jerk:.2f} px/f3"),
            ("source cams", f"{'on' if args.source_on else 'off'}  "
                            f"(GUI view > source cameras / --source_on)"),
            ("colors", f"gt {src_color}/now {src_now_color}  "
                       f"pred now {plan_now_color}  (GUI camera colors)"),
            ("target color", (f"gradation {plan_start} -> {plan_end}  "
                              f"(frame 0 -> {num_frames - 1}, --plan_color_start/_end)")
                             if args.plan_gradient
                             else f"단색 {plan_color}  (--no_plan_gradient)"),
            ("obb colors", f"dyn {obb_dyn_color}  static {obb_static_color}  "
                           f"(GUI obb / --obb_color_dyn/_static)"),
            ("obb handles", f"{sum(len(e) for e in obb_dyn)} dyn + {len(obb_static)} static  "
                            f"(GUI obb > OBB thickness / dynamic·static color)"),
            ("obb node", f"{gui_obb_node.value}  (GUI obb > node / --obb_node), "
                         f"labels {'on' if gui_obb_labels.value else 'off'}"),
            ("frame set", f"GUI view > dynamic frames = current frame  "
                          f"(interval / all frames + frame interval)"),
            ("camera frames", f"interval every 2  (만든 것은 stride {max(1, int(args.cam_stride))} "
                              f"= {len(range(0, num_frames, max(1, int(args.cam_stride))))}개/궤적, "
                              f"GUI view > camera frames / --cam_stride)"),
            ("paths", f"{'on' if args.paths_on else 'off'}  (GUI view > paths / --paths_on)"),
            ("ground grid", ("없음 (--no_ground_grid 또는 graph 없음)" if ground_handle is None
                             else f"{'on' if args.ground_on else 'off'}  "
                                  f"(GUI view > ground grid / --ground_on)")),
            ("probe", f"{'on' if args.probe_on else 'off'}  (GUI probe camera / --probe_on)"),
            ("now cams", f"{'on' if args.now_cams else 'off'}  "
                         f"(현재 프레임 1.6배 프러스텀 / --now_cams)"),
            ("pin color", "PIN_COLORS 순환 (--pin_palette)" if args.pin_palette
                          else "GUI 버튼 = target start 피커 스냅샷 / 기동 --pin = "
                               "PIN_COLORS 순환 (gradation 없음)"),
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
