"""씬 특성에 따라 **anchor 1개 + preset 8슬롯**만 고른다 — 렌더 없이 `scene_graph.json` 만 보고.

왜: 지금 뱅크는 anchor 전량 × preset 36 × τ 5 × hole 4 를 다 돈다 (dynpose 실측 339 변이/편).
영상당 비용이 `9.7분 + 8.80초 × 변이수` (R=0.887, 84편 회귀)라 변이가 곧 시간이다. 그런데
`fit_hole_ladder` 가 실제로 요청한 hole 을 맞춰 주는 비율(`binding == "hole"`)은 preset 마다
40배 차이가 난다 — dynpose 뱅크 232편 실측:

    pan_left/right      77%      dolly_out 54%    pull_out_arc 48~50%   truck 40~43%
    pedestal/crane   12~21%      orbit     15~17%
    push_in_arc      2.5~3.1%    dolly_in  0.3~1.3%   (전부 obb/collision 이 먼저 잡는다)

전진 계열은 anchor 로 돌진하다 OBB 에 막혀 요청 크기가 안 나온다. 그렇다고 빼면 캡션 어휘에서
"move forward" 가 통째로 사라지므로 **슬롯 1개만 남긴다** (DataDoP 검색 궤적이 같은 어휘를 실제
촬영본으로 보충한다 — `retrieve_datadop_shapes.py`).

**선택은 `fit` 전에 끝나야 한다.** 다 만들어 놓고 고르면 시간이 하나도 안 줄기 때문에,
여기서 쓰는 신호는 전부 `sample_camera_bank` 보다 앞서 나오는 것들이다:

    node.moving            track_* 로 갈지 (안 움직이는 anchor 에 track_ 을 붙이면 궤적이 비-track
                           짝과 비트 단위로 같아진다 — `sample_camera_bank` D77 주석)
    node.obs_az_span_deg   orbit 을 돌릴 수 있는지 (실측 중앙값 72°, 120° 이상은 25%뿐)
    graph.gravity.method   `camera_up_fallback`(257편 중 141편)이면 "위"가 world up 이 아니라
                           카메라 up 이다 — pedestal/crane 의 방향이 의미를 잃으므로 세로 슬롯을 뺀다
    cameras.cam_c2w_world  소스가 이미 어느 쪽으로 트럭했는지 → **반대쪽**을 고른다 (소스가 안 본
                           면을 보여주는 쪽이 hole 이 크고, 그게 이 데이터의 값어치다)

예시:
    python fit/bank/route_presets.py --video <id> --output_root out_dynpose            # 표
    python fit/bank/route_presets.py --video <id> --output_root out_dynpose --emit args
"""
import json
import sys
from argparse import ArgumentParser
from os import path
from zlib import crc32

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.presets import PRESETS as _PRESETS  # noqa: E402  (sys.path 조작 뒤여야 한다)
from scene_graph.schema import pick_main_anchors  # noqa: E402

# 옛 orbit span 게이트의 문턱. **D174 에서 기본 off(0)** 로 내렸다 — 실측이 전제를 기각했다
# (`--orbit_min_span 120` 으로 되살리면 d166~d171 재현). 근거는 route() docstring.
ORBIT_MIN_SPAN_DEG = 120.0
# 소스 자체 횡이동이 이보다 작으면 방향을 못 정한다 (world 단위, z_med 로 나눈 값).
LATERAL_DEADBAND = 0.02

# DataDoP 궤적을 어느 슬롯부터 채울지. **preset 이 약한 슬롯 순서**다:
# advance 계열 preset 은 hole-bound 가 0.3~1.3% 라 요청한 크기가 사실상 안 나오고(anchor OBB 가
# 먼저 막는다), vertical 은 gravity 가 `camera_up_fallback` 이면 슬롯 자체가 빠진다. 실제 촬영
# 궤적은 그 자리를 "카메라가 실제로 그렇게 움직인 적 있는 모양"으로 메운다.
EXTERNAL_SLOT_ORDER = ("advance", "vertical", "arc", "rotate", "lateral", "recede")

# ── `--slot_plan grid2x2` (D166) ────────────────────────────────────────────────────────
# scene 당 카메라를 144.4 → 5 로 줄인다. 구성은 `anchor{a,b} × slot{P1,P2}` 2x2 + free-moving 1.
#
# **슬롯 쌍을 anchor 가 아니라 video 로 해싱하는 이유**: 두 anchor 가 같은 슬롯 쌍을 써야
# "무엇을 보느냐"만 다른 대조쌍이 된다. 슬롯까지 anchor 마다 다르면 target 축과 motion 축이
# 섞여서 4칸 중 어느 것도 서로의 대조군이 아니게 된다 (D84 의 target 절 상수 문제와 같은 종류).
#
# 제외한 슬롯 두 개:
#   `rotate`(pan_*)  — targetless 라 target 절이 없다. 2x2 에 넣으면 두 anchor 의 캡션이 글자
#                      단위로 같아져 target 대조가 죽는다. 대신 **free-moving 슬롯**으로 옮겼다.
#   `lateral`(truck_*) — aim="free" 인데 캡션엔 target 절이 있다. D157 실측 sif-only 탈락
#                      7,910 행 중 3,024(38.2%)가 여기다. 조준을 고치기 전엔 안 쓴다 (task #134).
GRID_ALLOWED_SLOTS = ("advance", "recede", "arc", "orbit", "vertical", "static")
# 앞에서부터 **둘 다 이 씬에 존재하는** 첫 쌍을 쓴다. video 해시로 목록을 회전시켜 코퍼스가
# 한 쌍으로 쏠리지 않게 한다. 각 쌍은 서로 다른 이동 축이어야 motion 절이 구별력을 갖는다.
# `static` 이 든 쌍이 **하나뿐**인 이유: 쌍은 균등 추출이라 2개면 scene 의 25% 에서 4칸 중 2칸이
# 정지 hold 가 되고, 거기에 free-moving `pan_*`(회전만, path_len 0)까지 겹치면 5개 중 3개가
# 이동량 0 인 씬이 생긴다. 1개면 정지 변이 비중이 코퍼스의 6.25% 로 D157 실측 5.4% 와 붙는다.
GRID_SLOT_PAIRS = (("advance", "arc"), ("recede", "orbit"), ("advance", "vertical"),
                   ("recede", "arc"), ("orbit", "static"), ("advance", "orbit"),
                   ("recede", "vertical"), ("arc", "vertical"))
# grid2x2 에서 `recede` 를 보너스 track 후보에서 뺀다. `track_dolly_out` 은 track_* 인데
# aim="free" 인 조합이고(D157 실측 1,844행), 사용자 지시로 이번 라운드에선 제외한다.
GRID_TRACK_BONUS_SLOTS = ("advance", "vertical")


def stable_hash(text: str):
    """`hash()` 는 PYTHONHASHSEED 로 소금이 쳐져 실행마다 값이 바뀐다 — 뱅크가 재현이 안 된다."""
    return crc32(text.encode("utf-8"))


def obb_center_sep(a: dict, b: dict):
    """b 의 OBB 중심이 a 의 OBB 반-extent 몇 배 거리에 있나 (a 의 로컬 축 기준 L∞).

    <=1 이면 b 의 중심이 a 상자 **안**이고, 2 근처면 상자에 붙어 있다. 큰 쪽을 기준 상자로
    삼아야 대칭이 된다 — 모자 기준으로 재면 사람은 항상 멀다.
    """
    vol = lambda n: float(np.prod(n["obb"]["extent"]))
    big, small = (a, b) if vol(a) >= vol(b) else (b, a)
    rot = np.asarray(big["obb"]["R"], dtype=float)
    ext = np.asarray(big["obb"]["extent"], dtype=float)
    delta = rot @ (np.asarray(small["obb"]["center"], dtype=float)
                   - np.asarray(big["obb"]["center"], dtype=float))
    return float(np.max(np.abs(delta) / np.maximum(ext / 2, 1e-9)))


def pick_anchors(graph: dict, min_area_frac: float, max_dynamic: int = 3, max_static: int = 3,
                 max_anchors: int = 0, min_anchor_sep: float = 0.0, min_drift_u: float = 0.0,
                 require_frame0: bool = False):
    """anchor = 동적 상위 `max_dynamic` 개 + 정적 상위 `max_static` 개.

    각 갈래 안에서는 화면을 제일 크게 차지하는 것부터, 동률이면 오래 보이는 것부터.

    움직이는 노드를 먼저 보는 이유는 `track_*` 슬롯이 거기서만 의미가 있어서다 (정지 anchor 의
    follow_gain 은 변위가 0 이라 비-track 과 같은 궤적을 낸다).

    anchor 가 편당 1개면 안 되는 이유 (D84): 그 씬의 **모든** 변이가 같은 target 문장을 갖는다 —
    `target:` 절이 씬 안에서 상수라 "무엇을 보느냐"의 학습 신호가 0 이다 (정적 subject 안을
    미뤄 둔 것과 같은 종류의 구멍).

    D127b. 상한을 동적/정적 **따로** 세는 이유: 합산 상한 하나면 동적이 3개인 편에서 정적
    anchor 가 0 이 되어 `track_*` 슬롯만 있는 씬이 생긴다.

    D127. 표면 라벨(벽/바닥/울타리/문/창)은 여기서 빠진다 — `sample_camera_bank.anchor_nodes`
    와 **같은 함수**(`schema.pick_main_anchors`)를 쓴다. 두 곳이 다른 규칙을 쓰면 라우팅 표에
    있는 anchor 가 뱅크엔 없는(또는 그 반대) 상태가 조용히 생긴다.

    D166. `max_anchors` 는 두 상한을 적용한 **뒤에** 거는 총합 상한이다 (기본 0 = 끔,
    `sample_camera_bank.py` 와 같은 이름·같은 의미). `pick_main_anchors` 가 동적 우선으로
    정렬해 두므로 총합 2 = "dyn+dyn 우선, 부족분만 stat" 이 그냥 나온다.

    D166. `min_anchor_sep` — **부속물 anchor 를 뺀다.** 면적 순으로만 고르면 2등이 1등에 달린
    물건이 되기 쉽다: 파일럿 10편에서 person/hands 1.08, man/sunglasses 1.07, man/hat 1.53,
    man/bowl 1.91 로 4편(40%)이 그랬다. 이러면 2x2 의 target 축이 죽는다 — 모자를 겨눈 카메라와
    그걸 쓴 사람을 겨눈 카메라는 look_at 이 몇 cm 다를 뿐이라 픽셀로는 같은 shot 인데 캡션만
    다르게 붙어, 학습이 "target 절과 화면은 무관하다"를 배운다. 진짜 별개인 쌍은 훨씬 멀다
    (dog/person 8.44, hand/person 11.14, jacket/shelf 6.38) — 2.0 이 그 사이의 빈 구간이다.
    0 이면 옛 동작(안 거른다). 상한(`max_anchors`)보다 **먼저** 걸러야 부속물이 자리를 차지한
    채로 잘리는 일이 없다 — 그래서 `max_anchors` 를 `pick_main_anchors` 에 넘기지 않고
    여기서 자른다.

    D181. `min_drift_u` — **면적이 아니라 이동량으로** 먼저 자른다 (0 = 끔, 옛 동작).
    `--track_min_drift_u` 와는 거는 자리가 다르다: 그건 이미 뽑힌 anchor 의 슬롯에 `track_`
    접두사를 붙일지만 정하고, 못 미치면 같은 anchor 로 **비-track** preset 을 굽는다.
    "일정 이상 움직이는 대상만"을 원할 때 그 문턱으로는 안 되는 이유가 이것이다 — anchor 선택
    자체는 여전히 `max_area_frac` 순서라, 화면을 크게 차지하는 정지 물체가 1등으로 뽑히고
    이동량이 큰 물체는 상한 밖으로 밀린다. 여기서 자르면 애초에 후보에 안 들어온다.
    캡을 걸기 **전에** 필터해야 이동량 상위 노드가 면적 순위에 밀려 잘리지 않는다.

    D181. `require_frame0` — `track.frames` 에 프레임 0 이 없는 노드를 뺀다 (기본 off = 옛 동작).
    뱅크가 anchor 를 **소스 frame 0 가시성**으로 한 번 더 떨어뜨리기 때문이다
    (`skipped.json` 의 `no_surviving_anchors`, `subject_area: 0.0`). anchor 가 여럿이면 하나
    죽어도 씬이 살지만 anchor 1개짜리 파일럿에서는 씬이 통째로 날아간다 — 스모크 3편 중 2편이
    그랬다(f0=13 car, f0=2 hand). 이동량이 큰 물체일수록 나중에 프레임에 들어오므로
    `min_drift_u` 와 **같이 걸릴 때 특히** 겹친다. 필요조건일 뿐 충분조건은 아니다: 프레임 0 에
    보여도 그때 면적이 `min_area_frac` 미만이면 뱅크가 여전히 버린다.
    """
    cap = 0 if min_anchor_sep > 0 else max_anchors
    nodes = graph["nodes"]
    if min_drift_u > 0:
        nodes = [n for n in nodes if n.get("moving")
                 and float(n.get("center_drift_u", 0.0)) > min_drift_u]
    if require_frame0:
        nodes = [n for n in nodes if 0 in (n.get("track") or {}).get("frames", [0])]
    anchors, _ = pick_main_anchors(nodes, min_area_frac, max_dynamic, max_static,
                                   max_anchors=cap)
    if min_anchor_sep > 0:
        kept = []
        for node in anchors:
            if all(obb_center_sep(node, k) > min_anchor_sep for k in kept):
                kept.append(node)
        # 전부 붙어 있으면(단일 물체 클로즈업) 1등이라도 남긴다 — anchor 0 은 씬 전체를 버린다.
        anchors = kept or anchors[:1]
        anchors = anchors[:max_anchors] if max_anchors > 0 else anchors
    return anchors


def pick_anchor(graph: dict, min_area_frac: float):
    """`pick_anchors(..., 1)` 의 옛 이름. 후보가 없으면 `None` (호출부 계약 유지)."""
    got = pick_anchors(graph, min_area_frac, 1)
    return got[0] if got else None


def source_lateral(graph: dict):
    """소스 카메라의 **frame0 카메라 좌표계 기준 횡이동**을 z_med 로 나눈 값. +면 오른쪽.

    OpenCV 라 카메라 x 축이 오른쪽이다. `cam_c2w_world[0][:3,0]` 에 전 프레임 중심 변위를
    투영해 최댓값을 쓴다 (마지막 프레임만 보면 왕복 궤적에서 0 이 된다).
    """
    c2w = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64)
    right = c2w[0, :3, 0]
    disp = (c2w[:, :3, 3] - c2w[0, :3, 3]) @ right
    z_med = float(graph["scale"].get("z_med_frame0") or 1.0)
    return float(disp[np.argmax(np.abs(disp))] / max(z_med, 1e-9))


# `track_mode="add"` 에서 **항상 추종으로 두는** 슬롯. 추종이 궤적을 실제로 바꾸는 건 카메라가
# subject 를 옆에서/돌아서 보거나 제자리에서 지켜보는 동안이다 — 이때 anchor 변위가 카메라
# 위치·조준각으로 곧장 들어간다. `arc` 도 같은 부류다(곡선으로 물러나며 재조준하므로 subject 가
# 움직이면 곡률 중심이 같이 끌려가야 한다) — `track_pull_out_arc_*` 를 D82 에서 추가하며 편입.
# 나머지(recede/advance/vertical)는 시선축·중력축을 따라 움직여 추종을 걸어도 차이가 작고,
# 그 슬롯을 비-track 으로 남겨야 같은 씬 안에 두 어휘가 공존한다 (D82).
TRACK_KEEP_SLOTS = ("lateral", "arc", "orbit", "static")
# `add` 에서 추가로 하나 더 track 을 붙일 후보. 영상 id 로 골라 코퍼스 전체가 한 슬롯만
# 반복하지 않게 한다.
TRACK_BONUS_SLOTS = ("recede", "advance", "vertical")

# `track_` 짝이 **실재하는** preset 만 접두사를 받는다. `pan_*`(rotate)·`s_curve`(span 좁을 때의
# orbit 대체) 는 `lbm/presets.py` 에 track 판이 아예 없어서 `track_mode` 와 무관하게 항상
# 비-track 이다. 목록을 여기 복사하지 않고 표에서 직접 읽는 이유는, preset 이 추가됐는데 여기가
# 안 따라와 조용히 track 이 빠지는 걸 막기 위함 — `track_pull_out_arc_*` 가 실제로 그 경우였다.
PRESET_NAMES = frozenset(_PRESETS)

# D273 `--slot_plan pool` / `--free_moving pool` 의 후보. 표에서 직접 읽는다 (위 PRESET_NAMES 와
# 같은 이유 — preset 이 늘면 자동으로 따라온다). anchor 풀에서 빼는 것:
#   targetless 회전(pan/tilt/roll) · free_*   → free-moving 풀로
#   s_curve (D167 사용자 지시로 제거) · *_legacy · static_zoom_in (zoom 은 카메라 이동이 아니다)
_ROTATION = ("pan_", "tilt_", "roll_")
POOL_PRESETS = tuple(sorted(p for p in _PRESETS
                            if not p.startswith(_ROTATION + ("free_",))
                            and p not in ("s_curve", "static_zoom_in", "pan_right_zoom_out")
                            and not p.endswith("_legacy")))
FREE_POOL_PRESETS = tuple(sorted(p for p in _PRESETS
                                 if p.startswith(_ROTATION + ("free_",)) and "zoom" not in p))


def route(graph: dict, min_area_frac: float, allow_vertical_fallback: bool = False,
          track_mode: str = "add", node: dict = None,
          vertical_gravity: tuple = ("ground_ransac",),
          bonus_slots: tuple = TRACK_BONUS_SLOTS,
          track_min_drift_u: float = 0.0,
          orbit_fallback: str = "s_curve",
          orbit_min_span: float = 0.0):
    """-> (anchor, [(slot, preset), ...], reasons dict). preset 이름은 그대로 캡션 어휘가 된다.

    `track_mode` (D82) — anchor 가 **움직일 때만** 의미가 있다 (`sample_camera_bank` D77 이
    non-moving anchor 의 `track_*` 를 버린다: 변위가 0 이라 비-track 쌍둥이와 궤적이 bit-identical).

      `replace` 예전 동작. `track_` 짝이 있는 슬롯 전부에 접두사 — 한 영상이 track 5~6개거나 0개다.
      `add`(기본) 추종이 궤적을 바꾸는 슬롯(`TRACK_KEEP_SLOTS`)만 track 으로 두고 나머지는
              비-track 으로 남긴 뒤, `TRACK_BONUS_SLOTS` 중 하나를 영상별로 골라 track 을 더한다.
              → 한 씬 안에 track 3~4 + 비-track 4~5 가 섞인다.
      `off`   `track_` 을 아예 안 쓴다 (대조군).

    `vertical_gravity` (D147) — 세로 슬롯(crane/pedestal)을 **어떤 gravity 방법에서 신뢰하는가**.
    이 게이트가 있는 이유는 중력축이 틀리면 "위로 올라간다"가 카메라 up 방향이 되어 캡션과
    궤적이 어긋나기 때문이고, 원래 신뢰 목록은 `ground_ransac` 하나였다. D98 이 gravity 를
    GeoCalib 로 옮기면서 그 이름이 **한 번도 안 나오게** 됐다 — dynpose 280/280, vista 52/53 이
    `geocalib` 이다. vista 는 라우팅을 안 타고 preset 을 전량 열거해서 안 걸렸지만, dynpose 는
    이 함수를 타므로 세로 슬롯이 전량 조용히 빠졌다 (dd10 코퍼스 10,857 행에 crane/pedestal
    **0건**). GeoCalib 는 이미지에서 중력을 직접 추정하는 것이라 `camera_up_fallback` 과 달리
    "위"가 진짜 world up 이고, 신뢰 목록에 들어가는 게 맞다.
    함수 기본값은 **옛 동작을 그대로 재현**하고 새 동작은 CLI 기본값이다 (D143 과 같은 규약).

    `track_min_drift_u` (D166) — `sample_camera_bank --track_min_drift_u` 와 **같은 문턱**으로
    여기서도 track 을 거른다. 두 곳이 다른 판정을 쓰던 것이 버그였다: 라우팅은 `moving`
    불리언만 보는데 뱅크는 `center_drift_u > 0.05` 를 요구해서, `moving=True` 인데 변위가
    작은 anchor(사람 0.021 / 모자 0.015 실측)의 `track_*` 슬롯이 뱅크에서 조용히 사라졌다.
    `--slot_plan full` 에서는 슬롯이 많아 티가 안 났지만 grid2x2 에서는 2칸 중 1칸이
    통째로 날아가 scene 당 5개가 3개가 된다. 0 이면 옛 동작(=`moving` 만 본다).

    `orbit_fallback` (D167, 사용자 지시 2026-09-08 "preset 에서 s_curve 는 제거해줘") —
    `obs_az_span < ORBIT_MIN_SPAN_DEG` 일 때 orbit 슬롯을 무엇으로 대체하나.
      `s_curve` 함수 기본값 = 옛 동작. `drop` 은 CLI 기본값이고 **슬롯을 아예 안 만든다**.
    s_curve 를 뺀 자리를 다른 preset 으로 메우지 않는 이유: orbit 슬롯의 뜻이 "곡선으로
    선회한다"인데 좁은 span 에서 그걸 할 수 있는 preset 이 s_curve 말고 없다. 억지로
    `arc`/`recede` 를 넣으면 이미 그 슬롯을 쓰는 쌍과 겹쳐 2x2 의 motion 축이 죽는다.
    슬롯이 빠진 몫은 D166 backfill 이 그 anchor 의 남은 슬롯에서 채운다.
    d166 실측: 40 anchor 중 29 개(72.5%)가 `orbit_ok=False` 라 이 갈래를 탄다.

    `orbit_min_span` (D170 도입 → **D174 에서 기본 0 = off**, 사용자 지시 2026-09-10
    "orbit_min_span 은 별로인 것 같아 빼줘") — 위 갈래를 가르는 문턱. `0` 이면 게이트가
    통째로 꺼져 **span 과 무관하게 orbit 슬롯을 만든다**. `120` 을 주면 옛 동작(d166~d171).

    끈 이유는 전제가 실측에서 관측되지 않아서다. 문턱의 근거는 "좁은 span 에서 선회하면
    점군에 자료 없는 면으로 넘어간다"는 추론이었는데, 게이트 도입 **이전** 세대인 d157
    (384편 / Δ0.2 단 orbit 807행)로 재보니 `hole_fraction`·`hole_max`·`hole_max/hole`·
    `subject_visible_frac`·`subject_area_med` 가 span 1.5°~315° 에서 평평하다. 구멍은 물체
    주위 방위각이 아니라 근접 가림물과의 시차에서 나오고, 그건 hole 사다리가 이미 직접
    재서 손잡이로 통제한다. 반면 대가는 컸다 — d169 파일럿에서 게이트가 anchor 의 72.5% 를
    막고 그 자리를 backfill 이 떠맡으면서 `pull_out_arc_right` 통과율이 100%(4/4) →
    16.7%(1/6) 로 무너졌다.

    span 이 실제로 바꾸는 것은 sweep clamp 하나다 — `span_frac = max(orbit_span_frac,
    min_sweep_deg/span)` 이라 span<56 이면 sweep 이 nominal 45°에 못 미치고 span<20 이면
    바닥 20°에 붙는다. 남는 비용은 캡션 충실도(span 1.5° 인데 "dramatically orbits")이고,
    그건 라우팅이 아니라 캡션이 realized sweep 을 읽게 하는 쪽에서 고친다.
    """
    assert orbit_fallback in ("s_curve", "drop"), f"orbit_fallback: {orbit_fallback}"
    assert orbit_min_span >= 0.0, f"orbit_min_span: {orbit_min_span}"
    assert track_mode in ("add", "replace", "off"), f"track_mode: {track_mode}"
    # `node` 를 주면 그 노드로 라우팅한다 (`--num_anchors > 1` 이 anchor 마다 부른다).
    node = node if node is not None else pick_anchor(graph, min_area_frac)
    assert node is not None, "anchor 후보가 없다 (max_area_frac 하한을 낮추거나 graph 를 확인)"
    dyn = bool(node.get("moving"))
    if dyn and track_min_drift_u > 0:
        dyn = float(node.get("center_drift_u", 0.0)) > track_min_drift_u

    span = float(node["obs_az_span_deg"])
    grav = graph.get("gravity", {}).get("method")
    lat = source_lateral(graph)
    # 소스가 오른쪽으로 갔으면 우리는 왼쪽. deadband 안이면 노드 id 로 결정론적으로 가른다.
    away = "left" if lat > LATERAL_DEADBAND else ("right" if lat < -LATERAL_DEADBAND
                                                  else ("left" if stable_hash(node["id"]) % 2 else "right"))
    toward = "right" if away == "left" else "left"

    # 보너스 track 슬롯. **영상 id 로 해싱한다** — `node["id"]` 는 `dyn_0`, `dyn_1` 처럼 씬 안에서만
    # 유일한 이름이라 코퍼스 전체가 같은 슬롯 하나로 쏠린다 (263편 실측 확인). 슬롯 목록은
    # 씬마다 다르므로(vertical 이 빠질 수 있다) 실제 반영 여부는 아래에서 다시 확인한다.
    seed = f"{graph.get('video', '')}/{node['id']}"
    bonus = (bonus_slots[stable_hash(seed) % len(bonus_slots)]
             if (dyn and track_mode == "add" and bonus_slots) else None)

    def want_track(slot: str):
        """이 슬롯을 추종으로 둘 것인가 (preset 이름과 무관한 라우팅 판정)."""
        if not dyn or track_mode == "off":
            return False
        return track_mode == "replace" or slot in TRACK_KEEP_SLOTS or slot == bonus

    def tp(slot: str, base: str):
        """슬롯 하나의 최종 preset 이름. `track_` 짝이 없으면 접두사를 조용히 버린다."""
        name = f"track_{base}"
        return name if (want_track(slot) and name in PRESET_NAMES) else base

    slots = [("recede", tp("recede", "dolly_out")),
             # advance 는 **재조준하는 판**을 쓴다 (D134, 사용자 승인). `dolly_in` 은 aim="free"
             # 라 전진하는 동안 subject 를 다시 안 본다 — 접근할수록 화각 안에서 대상이 커지며
             # 가장자리로 밀려나는데도 카메라가 가만히 있어서, 전진 preset 이 정작 "대상에게
             # 다가간다"는 캡션 문구를 못 지킨다. `dolly_in_look_at` 은 같은 궤적에 매 프레임
             # 재조준만 붙인 것이라(`lbm/presets.py:148,150` — traj lambda 가 글자 그대로 같고
             # aim 만 free/look_at) 이동량·τ 사다리·hole 예산은 그대로 간다.
             # recede 를 같이 안 바꾸는 이유: 물러나는 동안은 대상이 화면 중앙에 남아서 aim 을
             # 안 걸어도 안 놓친다. 여기서 하나만 바꿔야 같은 씬에 두 조준 어휘가 공존한다.
             ("advance", tp("advance", "dolly_in_look_at")),
             ("lateral", tp("lateral", f"truck_{away}")),
             ("rotate", tp("rotate", f"pan_{away}")),
             ("arc", tp("arc", f"pull_out_arc_{toward}"))]

    if span >= orbit_min_span:
        slots.append(("orbit", tp("orbit", f"orbit_{away}")))
    elif orbit_fallback == "s_curve":
        # 관측 폭이 좁으면 orbit 은 점군에 자료가 없는 면으로 넘어간다. 같은 "곡선 이동"을
        # 소스가 본 범위 안에서 하는 게 s_curve 다 (실측 hole-bound 9.9%, 대부분 approach).
        slots.append(("orbit", tp("orbit", "s_curve")))

    if grav in vertical_gravity:
        slots.append(("vertical", tp("vertical", "crane_up")))
    elif allow_vertical_fallback:
        slots.append(("vertical", tp("vertical", "pedestal_up")))

    # hold 만 이름 규칙이 다르다: 비-track 은 `static_*`, track 은 `track_*` (`lbm/presets.py`
    # D75 주석) — 접두사가 아니라 다른 이름이라 `tp()` 를 태우지 않고 여기서 가른다.
    # **양쪽 다 `_look_at`** 이다: 이 슬롯이 뽑아야 하는 건 "제자리에서 subject 를 눈으로 쫓는"
    # 카메라이고, D94 부터 그건 `static_look_at` / `track_look_at` 이라는 이름을 갖는다.
    # D94 이전에 이 자리에 있던 `static_hold`/`track_hold` 가 바로 그 카메라였는데, 지금 그
    # 두 이름은 완전 고정(aim="free")이라 이름만 그대로 두면 카메라가 조용히 바뀐다.
    slots.append(("static", "track_look_at" if want_track("static") else "static_look_at"))

    have = {s for s, _ in slots}
    # `anchor_moving` 은 graph 의 원래 불리언을 그대로 둔다 (옛 독자 계약). track 판정에 실제로
    # 쓰인 값은 `anchor_track_eligible` 이고, 둘이 다르면 drift 문턱에서 갈린 것이다.
    reasons = {"anchor_moving": bool(node.get("moving")),
               "anchor_track_eligible": dyn,
               "center_drift_u": round(float(node.get("center_drift_u", 0.0)), 4),
               "obs_az_span_deg": round(span, 1),
               "gravity_method": grav, "source_lateral": round(lat, 4),
               "away_side": away, "orbit_ok": span >= orbit_min_span,
               # D167. `orbit_ok=False` 인데 슬롯이 없으면 여기가 `drop` 이다 (옛 JSON 엔 없는 키).
               "orbit_fallback": orbit_fallback,
               # D170. `orbit_ok` 는 이 문턱에 대한 상대값이다 — 문턱을 같이 안 적으면 옛 JSON 과
               # 비교할 때 span 이 변한 건지 게이트가 변한 건지 못 가른다.
               "orbit_min_span_deg": round(float(orbit_min_span), 1),
               "vertical_gravity_ok": grav in vertical_gravity,
               "vertical_dropped": grav not in vertical_gravity and not allow_vertical_fallback,
               "track_mode": track_mode,
               # 뽑힌 보너스 슬롯이 이 씬에 없으면(vertical 이 빠진 경우) track 이 하나 줄어든다.
               # 그걸 `preset_route.json` 만 보고도 알 수 있게 실제 반영 여부를 같이 적는다.
               "track_bonus_slot": bonus if (bonus in have) else None,
               "track_bonus_dropped": bool(bonus) and bonus not in have,
               "num_track": sum(1 for _, p in slots if p.startswith("track_"))}
    return node, slots, reasons


def grid_slot_pair(video: str, available: set, fill: str = "rotate"):
    """`--slot_plan grid2x2` 의 슬롯 쌍 (D166). -> (slot_a, slot_b).

    `available` 은 이 씬에 **실제로 존재하는** 슬롯 집합이다 (orbit 은 obs_az_span, vertical 은
    gravity 로 빠질 수 있다). `GRID_SLOT_PAIRS` 를 video 해시만큼 회전시켜 씨앗 쌍을 정한다 —
    씨앗이 video 라 두 anchor 가 같은 쌍을 받는다(§GRID_SLOT_PAIRS 주석).

    `fill` (D169, 사용자 지시 2026-09-08 "가능한 preset 후보풀을 뽑아두고 5개가 안나오면
    후보풀에서 다시 뽑으면 되잖아") — 씨앗 쌍의 한 짝이 이 씬에 없을 때 무엇을 하나.
      `rotate`(함수 기본값 = 옛 동작) 쌍을 통째로 버리고 **다음 쌍**으로 넘어간다.
      `substitute`(CLI 기본값) 살아남은 짝은 그대로 두고 **빠진 자리만** 후보풀에서 메운다.

    `rotate` 를 기본에서 뺀 이유는 D167 이 orbit 슬롯을 `drop` 하게 만든 뒤 실측된 것이다:
    `GRID_SLOT_PAIRS` 8쌍 중 3쌍이 orbit 을 물고 있는데 d166 파일럿 40 anchor 중 29개
    (72.5%)가 `orbit_ok=False` 라, 그 3쌍에 걸린 씬은 **남은 짝까지 같이** 버려지고 목록
    뒤쪽의 전혀 다른 쌍으로 착지했다. 21편 파일럿에서 6편(28.6%)의 슬롯 쌍이 바뀌었고
    preset 분포가 crane 계열 11→23 행, `static_look_at` 5→2 행으로 쏠렸다. `substitute` 는
    `("recede","orbit")` 에서 orbit 만 잃고 recede 를 유지하므로 그 쏠림이 안 생긴다.

    후보풀 순서도 video 해시로 회전시킨다 — `GRID_ALLOWED_SLOTS` 를 고정 순서로 쓰면 빠진
    자리가 전부 목록 앞쪽(`advance`)으로 몰려 코퍼스가 한 슬롯으로 쏠린다. s_curve 는 애초에
    후보풀에 없다: 여기서 다루는 건 preset 이 아니라 **슬롯**이고, s_curve 는 `--orbit_fallback
    drop` 이 orbit 슬롯을 안 만드는 것으로 이미 빠져 있다.

    한 쌍도 못 맞추면(`rotate`) / 후보풀이 1개뿐이면(`substitute`) 하나만 돌려준다 — 그 씬은
    카메라가 5개가 아니라 3개(2x1 + free)가 된다. 죽이지 않는 이유는 슬롯이 하나뿐인 씬
    (span 좁고 gravity 불신)도 target 대조는 여전히 성립하기 때문.
    """
    assert fill in ("rotate", "substitute"), f"slot_pair_fill: {fill}"
    n = len(GRID_SLOT_PAIRS)
    start = stable_hash(video) % n
    if fill == "rotate":
        for i in range(n):
            pair = GRID_SLOT_PAIRS[(start + i) % n]
            if pair[0] in available and pair[1] in available:
                return pair
        left = [s for s in GRID_ALLOWED_SLOTS if s in available]
        return tuple(left[:2])

    # 씨앗 쌍의 생존자를 먼저 앉히고, 빈 자리만 회전된 후보풀에서 채운다. 둘 다 살아 있으면
    # `rotate` 와 글자 단위로 같은 답이 나온다 (씨앗 쌍이 곧 rotate 의 첫 후보라서).
    kept = [s for s in GRID_SLOT_PAIRS[start] if s in available]
    width = len(GRID_ALLOWED_SLOTS)
    offset = stable_hash(video) % width
    for i in range(width):
        if len(kept) >= 2:
            break
        slot = GRID_ALLOWED_SLOTS[(offset + i) % width]
        if slot in available and slot not in kept:
            kept.append(slot)
    return tuple(kept[:2])


def pick_external(shapes_json: str, video: str, num: int, taken_slots):
    """DataDoP 궤적을 **슬롯에 맞춰** 고른다 → [(slot, name), ...].

    슬롯당 후보가 여러 개(`dd_fwd__static_0`, `dd_fwd__static_1` ...)라 어느 하나를 고정하면
    코퍼스 전체가 같은 영화 한 컷만 반복한다. 영상 id 를 씨앗으로 **무작위 추출**한다 (D83).
    예전에는 `names[(stable_hash(video) + i) % len(names)]` 였는데, 그건 정렬된 이름 목록 위의
    회전이라 슬롯 안에서 **결합 라벨이 섞이지 않는다** — 이름이 `dd_<move>__<angular>_<k>` 로
    정렬되므로 회전은 사실상 `move` 알파벳 순서를 따라 도는 것이고, 한 슬롯 안의 라벨 종류가
    늘어나도(D83 에서 lateral 40개 / arc 64개) 영상마다 인접한 이름만 뽑힌다.

    `taken_slots` 는 preset 이 이미 채운 슬롯 — 같은 슬롯이라도 preset 은 합성 모양,
    DataDoP 는 촬영 모양이라 겹쳐도 되지만 **못 채운 슬롯을 먼저** 준다. `num` 이 슬롯 수를
    넘으면 슬롯을 한 바퀴 더 돈다 (같은 슬롯에서 서로 다른 shape 두 개).
    """
    with open(shapes_json, encoding="utf-8") as file:
        blob = json.load(file)
    by_slot = {}
    for shape in blob["shapes"]:
        by_slot.setdefault(shape.get("slot") or "misc", []).append(shape["name"])
    rng = np.random.default_rng(stable_hash(video))

    def shuffled(group):
        """슬롯 순서도 영상마다 섞는다 (D83). `EXTERNAL_SLOT_ORDER` 를 고정 순서로 쓰면
        `num_external=4` 일 때 코퍼스 전체가 그 목록의 앞 4칸만 뽑아, 모양 뱅크의 라벨 47종 중
        그 4슬롯에 속한 32종만 쓰인다 (268편 실측). `EXTERNAL_SLOT_ORDER` 는 이제 "어느 슬롯을
        아는가"의 목록이자 tie-break 순서일 뿐이다."""
        group = [s for s in group if s in by_slot]
        return [group[i] for i in rng.permutation(len(group))]

    order = shuffled([s for s in EXTERNAL_SLOT_ORDER if s not in taken_slots])
    order += shuffled([s for s in EXTERNAL_SLOT_ORDER if s in taken_slots])
    order += [s for s in by_slot if s not in order]
    out, used = [], set()
    for i in range(num):
        slot = order[i % len(order)]
        names = [n for n in sorted(by_slot[slot]) if n not in used]
        if not names:
            continue
        name = names[int(rng.integers(len(names)))]
        used.add(name)
        out.append((slot, name))
    return out


def load_anchor_ids(file_path: str, video: str):
    """`--anchor_ids_file` -> 이 영상에 못 박은 anchor id 목록. 안 주면 `None` (= 옛 동작).

    D215 (사용자 지시 2026-09-21 "keep된 scene들 전부 main dynamic subject를 target으로").
    `pick_anchors` 는 **면적 순**으로 고르는데 사람이 릴을 보고 고른 피사체는 **이동량 순**
    1등이다. keeper 17씬 실측: 9편(53%)에서 둘이 다른 노드를 가리켰고, 그중 5편은 면적 1등이
    아예 `stat_*`(정지 배경)였다. 문턱(`--anchor_min_drift_u`)으로는 못 맞춘다 — 그건 후보를
    자를 뿐 남은 것들의 **순서**는 여전히 면적이라, 문턱을 넘긴 큰 정지 물체가 그대로 1등이다.

    파일 형식은 `{"<video>": ["dyn_1", ...], ...}` 한 장이다. 영상마다 값이 다른데
    `run_bank.py` 의 `route.args` 는 세대 config 하나에서 전 영상에 공통으로 붙으므로,
    per-video 값을 인자로는 못 넘긴다 — 그래서 경로 하나를 주고 안에서 영상을 찾는다.
    목록에 없는 영상은 `None` 을 돌려줘 그 씬만 기존 선택 규칙으로 돈다.
    """
    if not file_path:
        return None
    with open(file_path, encoding="utf-8") as file:
        table = json.load(file)
    got = table.get(video)
    return list(got) if got else None


def main(args):
    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    graph_path = path.join(out_root, args.video, "scene_graph.json")
    with open(graph_path, encoding="utf-8") as file:
        graph = json.load(file)
    forced = load_anchor_ids(args.anchor_ids_file, args.video)
    if forced is None:
        anchors = pick_anchors(graph, args.min_area_frac,
                               args.max_dynamic_anchors, args.max_static_anchors,
                               args.max_anchors, args.min_anchor_sep,
                               min_drift_u=args.anchor_min_drift_u,
                               require_frame0=args.anchor_require_frame0)
        # D273 (사용자 지시 2026-09-24 "anchor는 그냥 사람만"). `--anchor_labels` 에 적힌 라벨의
        # 노드만 anchor 가 된다. 빈 문자열이면 끔 = 옛 동작. 캡 **뒤에** 거르면 사람이 면적 순위에
        # 밀려 잘린 뒤라 0개가 될 수 있으므로, 후보 노드 자체를 먼저 거른 graph 로 다시 뽑는다.
        labels = {x.strip() for x in args.anchor_labels.split(",") if x.strip()}
        if labels:
            only = dict(graph, nodes=[n for n in graph["nodes"] if n.get("label") in labels])
            anchors = pick_anchors(only, args.min_area_frac,
                                   args.max_dynamic_anchors, args.max_static_anchors,
                                   args.max_anchors, args.min_anchor_sep,
                                   min_drift_u=args.anchor_min_drift_u,
                                   require_frame0=args.anchor_require_frame0)
        why = ("anchor 후보가 없다 (max_area_frac 하한을 낮추거나 graph 를 확인"
               + (f"; --anchor_min_drift_u {args.anchor_min_drift_u} 로 걸렀다)"
                  if args.anchor_min_drift_u > 0 else ")"))
    else:
        by_id = {n["id"]: n for n in graph["nodes"]}
        anchors = [by_id[i] for i in forced if i in by_id]
        why = (f"--anchor_ids_file 가 지정한 {forced} 가 이 씬의 노드에 없다 "
               f"(그래프를 다시 구웠으면 id 가 밀렸을 수 있다)")
    if not anchors and args.skip_if_empty:
        print(why, file=sys.stderr)
        raise SystemExit(3)
    assert anchors, why
    # D181. 슬롯 화이트리스트. 빈 문자열이면 끔 = 옛 동작.
    whitelist = tuple(s.strip() for s in args.slot_whitelist.split(",") if s.strip())
    # D192. anchor 가 추종 가능(`center_drift_u > --track_min_drift_u`)일 때만 쓰는 **두 번째**
    # 화이트리스트. 빈 문자열이면 위 하나를 양쪽에 똑같이 써서 옛 동작과 비트 동일하다.
    whitelist_track = tuple(s.strip() for s in args.slot_whitelist_track.split(",") if s.strip())
    # 쉼표 목록 -> tuple. 빈 문자열이면 세로 슬롯을 아예 안 넣는다(`--vertical_fallback` 만 남는다).
    vertical_gravity = tuple(m.strip() for m in args.vertical_gravity.split(",") if m.strip())
    grid = args.slot_plan == "grid2x2"
    bonus_slots = GRID_TRACK_BONUS_SLOTS if grid else TRACK_BONUS_SLOTS

    # D224. `--force_presets` 검증. 슬롯 표가 아니라 사람이 이름을 직접 적는 경로라
    # 오타가 조용히 "그 preset 만 빠진 뱅크"가 되는 걸 여기서 막는다.
    forced_presets = tuple(p.strip() for p in args.force_presets.split(",") if p.strip())
    unknown = [p for p in forced_presets if p not in PRESET_NAMES]
    assert not unknown, f"--force_presets 에 없는 preset: {unknown}"
    assert not (forced_presets and grid), \
        "--force_presets 는 --slot_plan full 전용이다 (grid2x2 는 슬롯 쌍으로 2개를 고른다)"
    if forced_presets and args.target_variants < len(forced_presets):
        # 예산이 모자라면 `plan_variants` 가 앞쪽 몇 개만 뽑아 요청한 preset 이 조용히 빠진다.
        print(f"--target_variants {args.target_variants} -> {len(forced_presets)} "
              f"(--force_presets 개수에 맞춘다)", file=sys.stderr)
        args.target_variants = len(forced_presets)

    # anchor 마다 따로 라우팅한다 — away side 도 track 여부도 그 노드의 성질에서 나온다.
    # `--slot_plan full`(기본): `sample_camera_bank` 가 (nodes x presets) 격자를 돌므로 preset 은
    # **합집합**으로 넘긴다. 정지 anchor 에 track_ 이 걸리면 D77 이 알아서 버린다.
    # `grid2x2`: 합집합을 쓰면 anchor 마다 away side 가 달라 격자가 부풀어 5개가 안 나온다.
    # 그래서 `--out` JSON 의 anchor 별 `slots` 를 `sample_camera_bank --preset_route` 가 직접
    # 읽는다 (`--presets` 합집합은 상위집합으로 그대로 남긴다 — 옛 독자 계약 유지).
    routed, presets = [], []
    keep_pair, full_first = None, None
    for node in anchors:
        _, slots, reasons = route(graph, args.min_area_frac, args.vertical_fallback,
                                  args.track_mode, node=node,
                                  vertical_gravity=vertical_gravity,
                                  bonus_slots=bonus_slots,
                                  track_min_drift_u=args.track_min_drift_u,
                                  orbit_fallback=args.orbit_fallback,
                                  orbit_min_span=args.orbit_min_span)
        full_first = full_first if full_first is not None else dict(slots)
        # D224 (사용자 지시 2026-09-21 — mountain-hike "push in / track push in", parkour
        # "track push in", lady-running "track+orbit right / track+orbit left / track push
        # out arc right"). `route()` 의 슬롯 표는 슬롯당 preset 을 하나로 못 박아 둬서 이
        # 요청들이 라우팅으로는 안 나온다: `advance` 는 `dolly_in_look_at` 고정이라 맨
        # `dolly_in` 이 없고, `arc` 는 `pull_out_arc_*` 고정이라 `push_in_arc_*` 가 없으며,
        # `orbit` 은 away 쪽 한 방향뿐이라 left/right 를 같이 못 낸다.
        # **anchor 선택·track 판정·reasons 는 그대로 두고 슬롯 목록만** 갈아끼운다 —
        # 여기서 갈아야 화이트리스트/rotate/grid 가 뒤이어 걸리지 않는다. 슬롯 이름은
        # 하류에서 안 쓴다 (`sample_camera_bank.py:904` 가 `preset` 만 읽는다).
        # 빈 문자열이면 no-op = 옛 동작 비트 동일.
        if args.slot_plan == "pool":
            # D273 (사용자 지시 2026-09-24 "기본적으로 모든 조합의 preset을 쓸 수 있되 track은
            # 일정 이상 움직일때만"). 슬롯 표(슬롯당 preset 1개 고정) 대신 **preset 전량**이
            # 후보다. track 판정은 `route()` 가 이미 낸 `anchor_track_eligible`
            # (= moving ∧ center_drift_u > --track_min_drift_u) 를 그대로 쓴다 — 문턱이 두 곳으로
            # 갈리면 D166 버그가 돌아온다. 세로(crane/pedestal)는 gravity 를 믿을 때만.
            # 제자리 회전(pan/tilt/roll)·free_ 는 targetless 라 여기가 아니라 free-moving 몫이다.
            track_ok = bool(reasons.get("anchor_track_eligible"))
            grav_ok = graph.get("gravity", {}).get("method") in vertical_gravity
            pool = [p for p in POOL_PRESETS
                    if (track_ok or not p.startswith("track_"))
                    and (grav_ok or not any(v in p for v in ("crane", "pedestal")))]
            rng = np.random.default_rng(stable_hash(f"{args.video}/{node['id']}/pool"))
            pool = [pool[i] for i in rng.permutation(len(pool))]
            reasons["pool_size"] = len(pool)
            # `--pool_size N` — 섞은 풀의 앞 N 개만. τ 뱅크(`--variant_pool full`)가 풀 전량을
            # 예비로 굽기 때문에 70 개면 tau 가 19 s -> 168 s 가 된다 (d273 스모크 1편). 씬마다
            # 순서가 다르므로 코퍼스 전체로는 전량이 고르게 쓰인다. 0 = 전량.
            if args.pool_size > 0:
                pool = pool[:args.pool_size]
            reasons["pool_track_ok"] = track_ok
            reasons["pool_vertical_ok"] = grav_ok
            slots = [(f"pool:{p}", p) for p in pool]
        if forced_presets:
            reasons["forced_presets"] = list(forced_presets)
            reasons["forced_replaced"] = [p for _, p in slots]
            slots = [(f"forced:{p}", p) for p in forced_presets]
        # D192 (사용자 지시 2026-09-14 "일정 이상 움직이는 subject일 경우 track hold, 많이
        # 안움직이는 subject일 경우 dolly in look at으로만"). 어휘를 두 개로 고정하려면 슬롯을
        # anchor 의 운동량으로 갈라야 한다 — 하나의 화이트리스트로는 못 한다. `advance` 만
        # 남기면 움직이는 anchor 가 `track_dolly_in_look_at` 을 받고, `static` 만 남기면 정지
        # anchor 가 `static_look_at` 을 받아 둘 다 요청과 다른 preset 이 된다.
        # 판정량은 `route()` 가 이미 계산해 `reasons` 에 실어 둔 `anchor_track_eligible` 이다
        # (= `moving` ∧ `center_drift_u > --track_min_drift_u`). 여기서 다시 계산하면 문턱이
        # 두 곳으로 갈려 D166 이 고친 버그가 되돌아온다.
        # `whitelist` 를 덮으면 다음 anchor 가 앞 anchor 의 목록을 물려받는다 — 루프 안에서만
        # 사는 이름으로 받는다.
        keep = whitelist
        if whitelist_track:
            track_ok = bool(reasons.get("anchor_track_eligible"))
            keep = whitelist_track if track_ok else whitelist
            reasons["slot_whitelist_source"] = "track" if track_ok else "static"
        if keep:
            # D181. `route()` 의 슬롯 구성·방향(away/toward)·track 판정·reasons 를 그대로 쓰고
            # **마지막에 골라내기만** 한다. 여기서 거르는 이유는 route() 안에서 거르면 bonus
            # track 슬롯 해싱과 `num_track` 집계가 화이트리스트에 따라 달라져, 같은 씬의
            # 전량 라우팅과 파일럿 라우팅이 서로 다른 preset 을 내기 때문 — 그러면 파일럿이
            # 코퍼스의 부분집합이 아니게 된다.
            reasons["slot_whitelist"] = list(keep)
            reasons["slot_whitelist_dropped"] = [s for s, _ in slots if s not in keep]
            slots = [(s, p) for s, p in slots if s in keep]
        # D188. 슬롯 **순서**를 video 해시로 회전한다. 예산이 1일 때 `plan_variants` 가 뽑는 건
        # `pool[0]` 하나뿐인데, `route()` 의 슬롯 순서는 (recede, advance, lateral, rotate, arc,
        # orbit, vertical, static) 로 고정이라 화이트리스트를 씌우면 **살아남은 것 중 가장 앞**이
        # 코퍼스 전량에서 같은 슬롯이 된다. 400편 실측: `advance` 계열이 tier0 의 60.8%,
        # `vertical` 은 0건 (grid2x2 + substitute 를 써도 keep_pair 안에서 다시 route 순서로
        # 정렬되기 때문에 같은 쏠림이 남는다). 회전하면 4슬롯이 각 ~25% 로 갈린다.
        # 회전 씨앗이 `node["id"]` 가 아니라 video 인 이유는 D166 과 같다 — `dyn_0` 같은 이름은
        # 씬 안에서만 유일해서 코퍼스 전체가 한 슬롯으로 쏠린다.
        if args.slot_rotate and slots:
            off = stable_hash(args.video) % len(slots)
            slots = slots[off:] + slots[:off]
            reasons["slot_rotate"] = off
        # D188 (사용자 지시 2026-09-13 "움직이는 dynamic anchor면 track+object centric 아니면
        # object centric으로 나오도록 **fallback도 넣어서**"). `--track_mode replace` 는 슬롯을
        # track 판으로 **갈아끼우기만** 해서 비-track 짝이 후보 풀에서 통째로 사라진다 —
        # `sample_camera_bank.plan_variants` 의 pool 은 여기서 낸 slots+backfill 이 전부라,
        # 그 상태로 `--fallback_ladder` 를 켜도 1층(비-track 조준)이 비어 있어 추종이 실패하면
        # 다른 추종으로만 떨어진다 (같은 anchor 변위에 같은 벽 — D166 이 층 2를 1보다 뒤에 둔
        # 바로 그 이유). `--track_pair` 는 갈아끼운 track 슬롯의 **평범한 짝을 backfill 에**
        # 같이 실어 그 1층을 채운다. `variant_tier` 가 비-track look_at 을 1, track 을 2로
        # 매기므로 사다리는 자동으로 [track(0) → 평범(1) → 남은 track(2) → free(3)] 이 된다.
        # 정지 anchor 에는 애초에 `track_` 이 안 붙으므로 이 블록이 no-op 이고, 그 씬은 0층이
        # 곧 object-centric 이다. 기본값 off = 옛 동작 비트 동일.
        pair_backfill = []
        if args.track_pair:
            for s, p in slots:
                plain = p[len("track_"):] if p.startswith("track_") else None
                if plain and plain in PRESET_NAMES:
                    pair_backfill.append((s, plain))
            reasons["track_pair"] = [p for _, p in pair_backfill]
        backfill = []
        if grid:
            if keep_pair is None:
                keep_pair = grid_slot_pair(args.video, {s for s, _ in slots},
                                           fill=args.slot_pair_fill)
            # D166. keep_pair 밖 슬롯은 버리지 않고 **backfill 풀**로 남긴다. 뱅크가 anchor 를
            # frame 0 가시성으로 더 떨어뜨리므로(파일럿 16편 중 5편이 anchor 하나를 잃었다)
            # 라우팅이 미리 5개를 맞춰 놔도 소용이 없다 — 부족분을 그때 채울 재료가 필요하다.
            # 순서는 `GRID_ALLOWED_SLOTS` 고정: 영상마다 뒤집히면 backfill 이 들어간 씬과
            # 안 들어간 씬의 preset 분포가 서로 다른 이유로 갈린다.
            order = {s: i for i, s in enumerate(GRID_ALLOWED_SLOTS)}
            backfill = sorted(((s, p) for s, p in slots if s not in keep_pair),
                              key=lambda sp: order.get(sp[0], len(order)))
            slots = [(s, p) for s, p in slots if s in keep_pair]
            reasons["grid_slot_pair"] = list(keep_pair)
            # D169. 씨앗 쌍을 같이 남긴다 — `keep_pair` 만 보면 대체가 일어났는지 못 본다.
            reasons["grid_seed_pair"] = list(GRID_SLOT_PAIRS[stable_hash(args.video)
                                                             % len(GRID_SLOT_PAIRS)])
            reasons["slot_pair_fill"] = args.slot_pair_fill
        # 평범한 짝은 **항상 backfill 뒤쪽**이다 (grid 를 안 쓰면 backfill 이 이것뿐). keep_pair
        # 밖 슬롯(전부 track)보다 뒤에 두는 건 순서 때문이 아니라 — fit 은 `variant_tier` 로
        # 다시 정렬한다 — `pool.index` 가 같은 층 안의 동점을 가를 때 쓰이기 때문이다.
        backfill = backfill + [sp for sp in pair_backfill
                               if sp[1] not in {p for _, p in backfill}
                               and sp[1] not in {p for _, p in slots}]
        if args.external_shapes and args.num_external > 0:
            picked = pick_external(args.external_shapes, f"{args.video}/{node['id']}",
                                   args.num_external, {s for s, _ in slots})
            slots = slots + [(f"datadop:{s}", p) for s, p in picked]
        routed.append({"anchor_id": node["id"], "anchor_label": node["label"],
                       "anchor_moving": bool(node.get("moving")),
                       "anchor_area_frac": round(float(node["max_area_frac"]), 4),
                       "slots": [{"slot": s, "preset": p} for s, p in slots],
                       "backfill": [{"slot": s, "preset": p} for s, p in backfill],
                       "reasons": reasons})
        presets += [p for _, p in slots if p not in presets]
        presets += [p for _, p in backfill if p not in presets]

    # ── free-moving 슬롯 (D166, 사용자 지시 2026-09-08) ─────────────────────────────────
    # target 절이 **없는** 변이를 scene 당 정확히 1개 얹는다 (2x2 + 1 = 5). 격자에 넣으면
    # anchor 두 개가 같은 캡션을 갖게 되므로(target 절이 없어서 글자 단위로 같다) 격자 밖에 둔다.
    # **scene 단위 키**로 낸다 (`free_moving`) — 첫 anchor 의 `slots` 에 넣던 것을 옮겼다.
    # 그 anchor 가 뱅크의 frame 0 가시성 게이트에서 죽으면 free-moving 도 같이 사라졌는데
    # (파일럿 `023615b3`, `01d32f88`), free-moving 은 anchor 를 안 쓰는 변이라 같이 죽을
    # 이유가 없다. 이제 뱅크가 **살아남은 첫 anchor** 에 붙인다.
    # 캡션 쪽 처리는 이미 있다:
    #   `rotate`  -> `pan_*`  : `configs/caption_presets.json` 의 `targetless` 플래그
    #   `datadop` -> `dd_*`   : `build_bank_captions.FREE_MOVING_PREFIX`
    # 둘 다 target/framing 을 빈 문자열로 두고 motion 절만 남긴다.
    #
    # **track_ 은 안 붙인다** (사용자 지시). 구조적으로도 안 붙는다 — `lbm/presets.py` 에
    # `track_pan_*` 가 없어서 `tp()` 가 접두사를 버리고, `dd_*` 는 애초에 접두사 대상이 아니다.
    # 그래도 assert 로 못 박는다: preset 표가 바뀌어 `track_pan_left` 가 생기면 조용히
    # "추종하는데 target 을 안 부르는" 변이가 되기 때문.
    free_moving = None
    if args.free_moving != "off":
        if args.free_moving == "rotate":
            free = full_first.get("rotate")
        elif args.free_moving == "pool":
            # D273 (사용자 지시 "free moving vertical도 돌아가게"). 제자리 회전 + 세로 병진
            # (`free_pedestal_*`, caption targetless). 세로는 gravity 를 믿을 때만.
            grav_ok = graph.get("gravity", {}).get("method") in vertical_gravity
            cands = [p for p in FREE_POOL_PRESETS if grav_ok or "pedestal" not in p]
            free = cands[stable_hash(f"{args.video}/free") % len(cands)]
        else:
            picked = pick_external(args.external_shapes, f"{args.video}/free",
                                   1, {s["slot"] for s in routed[0]["slots"]})
            free = picked[0][1] if picked else None
        assert args.free_moving != "datadop" or args.external_shapes, \
            "--free_moving datadop 은 --external_shapes 가 있어야 한다"
        if free:
            assert not free.startswith("track_"), \
                f"free-moving 슬롯에 track_ 이 붙었다: {free} (target 절이 없는데 추종한다)"
            free_moving = {"slot": "free", "preset": free}
            routed[0]["reasons"]["free_moving"] = free
            if free not in presets:
                presets.append(free)

    # D184. 화이트리스트가 anchor 의 슬롯을 전부 걷어내면 preset 이 0개다 — anchor 0 과 같은
    # 결말(변이 0)이므로 같은 rc=3 으로 내보낸다. 켜지 않으면 옛 동작대로 빈 `--presets` 를
    # 그대로 출력한다.
    if args.skip_if_empty and not presets:
        print(f"슬롯이 0개 — --slot_whitelist '{args.slot_whitelist}'"
              + (f" / --slot_whitelist_track '{args.slot_whitelist_track}'"
                 if args.slot_whitelist_track else "")
              + " 뒤에 남은 슬롯이 없다", file=sys.stderr)
        raise SystemExit(3)

    # D166. scene 당 변이 예산. `target_count` 가 상한(요구가 아니다 — 사용자 지시
    # "최대 5개"), `object_budget` 은 그중 anchor 를 쓰는 몫이다. 뱅크가 이 둘을 읽어
    # 살아남은 anchor 수에 맞춰 배분한다(anchor 2+ 이고 움직이면 2x2, 아니면 1 anchor x 4).
    object_budget = max(args.target_variants - (1 if free_moving else 0), 1)

    node = anchors[0]
    slots = [(s["slot"], s["preset"]) for s in routed[0]["slots"]]
    if free_moving:
        # 최상위 `slots` 는 옛 독자(파일럿 러너)용 뷰라 free 를 예전처럼 붙여 둔다.
        # 뱅크가 읽는 정본은 `anchors[*].slots` + scene 단위 `free_moving` 이다.
        slots = slots + [(free_moving["slot"], free_moving["preset"])]
    reasons = routed[0]["reasons"]

    # `--out` 은 `--emit` 보다 먼저 쓴다 — `--emit presets` 로 쓰면서 동시에 anchor 를
    # 파일로 받아 가는 호출부(파일럿 러너)가 있다.
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            # 최상위 `anchor_id` / `slots` / `reasons` 는 첫 anchor 의 것 — `--num_anchors 1`
            # 이던 시절의 독자(파일럿 러너)가 그대로 읽는다. 전량은 `anchors` 에 있다.
            json.dump({"format": "preset_route_v1", "video": args.video,
                       "anchor_id": node["id"], "anchor_label": node["label"],
                       "anchor_ids": [a["anchor_id"] for a in routed],
                       "anchors": routed,
                       "free_moving": free_moving,
                       "target_count": args.target_variants,
                       "object_budget": object_budget,
                       "slots": [{"slot": s, "preset": p} for s, p in slots],
                       "presets": presets, "reasons": reasons},
                      file, ensure_ascii=False, indent=1)

    if args.emit == "args":
        # anchor 마다 preset 목록이 다르면 합집합만으론 격자가 부풀므로 JSON 경로를 같이 넘긴다.
        # `--free_moving` 만 켜도 그렇다 — 그 preset 이 합집합에 들어가면 anchor 전부에 걸린다.
        # D188. `--slot_plan full` + `--target_variants N` 조합에서는 위 조건이 **둘 다 거짓**이라
        # `--preset_route` 가 안 나갔다. 그러면 `sample_camera_bank` 의 `target_count` 가 0 이고
        # (`:897` 은 그 JSON 에서만 읽는다), `plan_variants` 가 `if not target_count` 가지로
        # 빠져 tiers 를 `{}` 로 돌려주며, 그 결과 `fit_hole_ladder` 의
        # `fallback_on = ... and any(tier_of.values())` 가 False 가 된다 — **예산도 사다리도
        # 통째로 무력화**된다. d188 파일럿 60편이 씬당 카메라 1대 대신 최대 9대를 냈고
        # (solved 175행/58편) track 비중도 42.9% 로 주저앉은 게 이 경로다.
        # `--emit_route` 가 그 경로를 연다. 기본 off = 옛 동작 비트 동일.
        per_anchor = grid or args.free_moving != "off" or args.emit_route
        assert not per_anchor or args.out, \
            "--slot_plan grid2x2 / --free_moving / --emit_route 는 --out 이 있어야 한다 (--preset_route 로 넘긴다)"
        print(f"--nodes {' '.join(a['anchor_id'] for a in routed)} "
              f"--presets {' '.join(presets)}"
              + (f" --preset_route {args.out}" if per_anchor else ""))
        return
    if args.emit == "presets":
        print(" ".join(presets))
        return
    if args.emit == "nodes":
        print(" ".join(a["anchor_id"] for a in routed))
        return

    header = f"{'slot':<18}{'preset':<24}"
    print(f"{'video':<16}{args.video}")
    for entry in routed:
        print(f"{'anchor':<16}{entry['anchor_id']}  {entry['anchor_label']}  "
              f"moving={entry['anchor_moving']}  area={entry['anchor_area_frac']:.3f}")
        for k, v in entry["reasons"].items():
            print(f"  {k:<22}{v}")
        print(f"\n{header}\n" + "-" * len(header))
        for slot in entry["slots"]:
            print(f"{slot['slot']:<18}{slot['preset']:<24}")
        print()
    total = sum(len(a["slots"]) for a in routed) + (1 if free_moving else 0)
    if grid or args.free_moving != "off":
        # 격자가 아니라 anchor 별 슬롯 수의 합이다 (`--preset_route` 를 넘기므로).
        # 실제 행 수는 뱅크가 정한다 — 여기 수는 **backfill 전** 값이고, anchor 가 뱅크의
        # frame 0 가시성 게이트에서 죽으면 줄고 backfill 이 다시 채운다.
        pool = sum(len(a["backfill"]) for a in routed)
        print(f"변이 수(backfill 전) = sum(anchor 별 슬롯) + free x rung 1 = {total}   "
              f"(anchor {len(routed)}, 슬롯쌍 {keep_pair}, free {args.free_moving})")
        print(f"예산: target {args.target_variants} = object {object_budget} + "
              f"free {1 if free_moving else 0}   backfill 풀 {pool}")
    else:
        print(f"변이 수 <= anchor {len(routed)} x preset {len(presets)} x rung 1 = "
              f"{len(routed) * len(presets)}")
    if args.out:
        print(f"-> {args.out}")


def build_parser():
    """파서를 **함수로** 꺼내 둔 이유: `run_bank.py --exec inproc` 이 이 스크립트를 서브프로세스가
    아니라 같은 프로세스에서 부른다 (D180). 파서가 `__main__` 블록 안에 있으면 import 로는
    만들 수 없다. CLI 동작은 그대로다 — 아래 `__main__` 이 이 함수를 쓴다.
    """
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--min_area_frac", default=0.01, type=float)
    # D84: anchor 를 몇 개까지. 편당 2 이상이어야 씬 안에 target 이 둘 이상이라 `target:` 절이
    # 비로소 구별력을 갖는다 (`pick_anchors` 주석).
    # D127b: 동적/정적 각각 상한 (사용자 지시). `sample_camera_bank.py` 와 같은 이름·같은 기본값.
    parser.add_argument("--max_dynamic_anchors", default=3, type=int)
    parser.add_argument("--max_static_anchors", default=3, type=int)
    # D166: 두 상한을 적용한 **뒤** 총합 상한. 0 = 끔(옛 동작). `pick_main_anchors` 가 동적
    # 우선으로 정렬하므로 `--max_dynamic_anchors 2 --max_static_anchors 2 --max_anchors 2`
    # 가 곧 "dyn+dyn 우선, 부족분만 stat" 이다 (사용자 지시).
    parser.add_argument("--max_anchors", default=0, type=int)
    # D166: 앞선 anchor 의 OBB 반-extent 이 배수 안에 중심이 들어오면 **부속물로 보고 버린다**
    # (모자/선글라스/손). 0 이면 옛 동작. 근거와 실측 분포는 `pick_anchors` docstring.
    parser.add_argument("--min_anchor_sep", default=2.0, type=float)
    # D166: 슬롯 구성. full = 8슬롯 전량(옛 동작). grid2x2 = anchor 두 개가 **같은 슬롯 쌍**을
    # 써서 4칸을 만든다 (§GRID_SLOT_PAIRS). `sample_camera_bank --preset_route` 와 짝이다.
    # D273: pool = 슬롯 표 대신 preset 전량 (`POOL_PRESETS`, track_ 은 추종 가능 anchor 만).
    parser.add_argument("--slot_plan", default="full", type=str, choices=("full", "grid2x2", "pool"))
    # D273: anchor 로 허용할 노드 라벨 (쉼표). 빈 문자열 = 끔 (옛 동작).
    parser.add_argument("--anchor_labels", default="", type=str)
    parser.add_argument("--pool_size", default=0, type=int)          # D273: --slot_plan pool 앞 N 개
    # D166: target 절 없는 변이를 scene 당 1개. off = 안 넣음(옛 동작).
    # rotate = 라우팅된 `pan_*`(caption `targetless`), datadop = `dd_*` 1개(`--external_shapes` 필요).
    parser.add_argument("--free_moving", default="off", type=str,
                        choices=("off", "rotate", "datadop", "pool"))
    # D166 (사용자 지시 2026-09-08): scene 당 변이 **상한**. 요구가 아니다 — 못 채우면 그대로 낸다.
    # `object_budget = target_variants - (free 1)` 이 anchor 를 쓰는 몫이고, 뱅크가 살아남은
    # anchor 수에 맞춰 나눈다 (2+ 이고 움직이면 2x2, 아니면 1 anchor x 4). JSON 에만 실린다.
    parser.add_argument("--target_variants", default=5, type=int)
    # D167 (사용자 지시 2026-09-08): `s_curve` 를 라우팅 어휘에서 뺀다. 좁은 span 의 orbit
    # 슬롯이 s_curve 로 가던 **유일한 입구**가 여기라, `drop` 이면 코퍼스에서 s_curve 가 0 이
    # 된다. `lbm/presets.py` 의 builder 자체는 안 지운다 — 이미 나간 뱅크·decision.json 이
    # 그 이름을 참조하고, 지우면 재현과 캡션 조회가 통째로 깨진다.
    # 옛 뱅크(d157/d166)를 재현하려면 `--orbit_fallback s_curve`.
    parser.add_argument("--orbit_fallback", default="drop", type=str,
                        choices=("drop", "s_curve"))
    # D174 (사용자 지시 2026-09-10 "orbit_min_span 은 별로인 것 같아 빼줘"): 게이트 **기본 off**.
    # D170 릴 + d157 807행 실측에서 hole·hole_max·subject_visible_frac 이 span 1.5°~315° 에서
    # 평평했다 — 문턱의 전제("좁은 span 에서 선회하면 미관측 면으로 넘어간다")가 관측되지 않는다.
    # 인자 자체는 남긴다: `--orbit_min_span 120` 이면 d166~d171 재현이고, 스윕도 이걸로 한다.
    parser.add_argument("--orbit_min_span", default=0.0, type=float)
    # D169 (사용자 지시 2026-09-08): 씨앗 슬롯 쌍의 한 짝이 없을 때. rotate = 쌍을 통째로 버리고
    # 다음 쌍으로(옛 동작, d166/d168 재현용). substitute = 살아남은 짝을 유지하고 빠진 자리만
    # 후보풀에서 메운다. 근거와 실측은 `grid_slot_pair` docstring.
    parser.add_argument("--slot_pair_fill", default="substitute", type=str,
                        choices=("rotate", "substitute"))
    # gravity 가 camera_up_fallback 이어도 세로 슬롯을 넣고 싶으면 켠다 (방향이 카메라 up 이다).
    parser.add_argument("--vertical_fallback", action="store_true", default=False)
    # D147: 세로 슬롯(crane/pedestal)을 신뢰할 gravity 방법(쉼표 목록). D98 이 gravity 를 GeoCalib
    # 으로 옮긴 뒤 실제로 나오는 이름은 `geocalib` 뿐이라 기본값에 넣는다 — 안 넣으면 세로 슬롯이
    # 조용히 전량 빠진다(dd10 10,857 행에 crane/pedestal 0건). 옛 동작 재현은
    # `--vertical_gravity ground_ransac`.
    parser.add_argument("--vertical_gravity", default="ground_ransac,geocalib", type=str)
    # add = 추종 슬롯만 track (한 씬에 track/비-track 공존), replace = 예전 전량 track, off = 대조군
    parser.add_argument("--track_mode", default="add", type=str,
                        choices=("add", "replace", "off"))
    # D166: `sample_camera_bank --track_min_drift_u` 와 **같은 값**을 줘야 한다. 라우팅이 track 을
    # 붙였는데 뱅크가 그 문턱으로 버리면 그 슬롯은 통째로 사라진다 (grid2x2 에서 2칸 중 1칸).
    # 0 이면 옛 동작(`moving` 불리언만 본다).
    parser.add_argument("--track_min_drift_u", default=0.05, type=float)
    # D188: track 슬롯의 **비-track 짝**을 backfill 에 같이 싣는다 (fallback 사다리 1층 채우기).
    parser.add_argument("--track_pair", dest="track_pair", action="store_true", default=False)
    parser.add_argument("--no_track_pair", dest="track_pair", action="store_false")
    # D188: 슬롯 순서를 video 해시로 회전 (예산 1일 때 tier0 가 한 슬롯으로 쏠리는 것을 막는다).
    parser.add_argument("--slot_rotate", dest="slot_rotate", action="store_true", default=False)
    parser.add_argument("--no_slot_rotate", dest="slot_rotate", action="store_false")
    # D181 (사용자 지시 2026-09-11 "일정 이상 움직이는 dynamic target 있는 경우만"):
    # anchor **후보 자체**를 이동량으로 자른다. 0 = 끔(옛 동작). `--track_min_drift_u` 와
    # 다른 자리에 걸리는 이유는 `pick_anchors` docstring 에 있다. 이 문턱을 넘는 노드가 하나도
    # 없으면 그 씬은 route 에서 죽는다 — 씬 목록을 미리 같은 문턱으로 거르고 쓰는 인자다.
    parser.add_argument("--anchor_min_drift_u", default=0.0, type=float)
    # D181: `track.frames` 에 프레임 0 이 없는 노드를 anchor 후보에서 뺀다 (기본 off = 옛 동작).
    # 뱅크의 frame 0 가시성 게이트를 라우팅이 미리 아는 것 — 근거는 `pick_anchors` docstring.
    parser.add_argument("--anchor_require_frame0", action="store_true", default=False)
    # D215: `{"<video>": ["dyn_1"]}` JSON 으로 anchor 를 못 박는다 (빈 값 = 끔 = 옛 동작).
    # 목록에 있는 영상은 `--max_*_anchors` / `--min_anchor_sep` / `--anchor_min_drift_u` /
    # `--anchor_require_frame0` 를 **전부 우회**한다 — 사람이 이미 고른 노드라 다시 거를 이유가
    # 없고, 걸러지면 그 씬이 조용히 다른 피사체로 구워진다. 근거는 `load_anchor_ids`.
    parser.add_argument("--anchor_ids_file", default="", type=str)
    # D181: 라우팅된 슬롯 중 **이것만** 남긴다 (쉼표 목록, 빈 문자열 = 끔 = 옛 동작).
    # 슬롯 이름이지 preset 이름이 아니다 — `orbit` 은 away side 와 track 여부에 따라
    # `track_orbit_left` / `orbit_right` 등으로 풀린다. preset 이름을 직접 적게 하지 않는 이유:
    # 방향은 소스 카메라의 횡이동에서 나오는 씬별 값이라 고정하면 소스와 같은 쪽으로 도는
    # 변이가 섞인다 (§route 의 away/toward).
    parser.add_argument("--slot_whitelist", default="", type=str)
    # D192 (사용자 지시 2026-09-14): anchor 가 **추종 가능**할 때만 쓰는 두 번째 목록. 비어
    # 있으면 위 목록을 양쪽에 써서 옛 동작과 비트 동일하다. 판정은 `route()` 의
    # `anchor_track_eligible` (= `moving` ∧ `center_drift_u > --track_min_drift_u`) 하나.
    #   예) `--slot_whitelist advance --slot_whitelist_track static --track_mode replace`
    #       움직이는 anchor -> static 슬롯 -> `track_look_at` (따라가며 재조준)
    #       정지   anchor -> advance 슬롯 -> `dolly_in_look_at` (다가가며 재조준)
    # 한 목록으로는 못 하는 이유: `advance` 만 남기면 움직이는 anchor 가
    # `track_dolly_in_look_at`, `static` 만 남기면 정지 anchor 가 `static_look_at` 이 된다.
    parser.add_argument("--slot_whitelist_track", default="", type=str)
    # D224 (사용자 지시 2026-09-21). 슬롯 표를 건너뛰고 **preset 이름을 직접** 적는다.
    # `--slot_whitelist` 는 슬롯 표에 있는 조합 중에서 고르는 것이라, 슬롯당 하나로 못 박힌
    # preset(advance=`dolly_in_look_at`, arc=`pull_out_arc_*`, orbit=away 한 방향)은 못 낸다.
    # 사용자가 특정 카메라를 지목하는 일회성 요청 전용이고, 코퍼스 굽기에는 쓰지 않는다 —
    # 방향(left/right)이 씬별 away/toward 가 아니라 고정이라 소스와 같은 쪽으로 도는 변이가
    # 섞인다(§route 의 away/toward). 빈 문자열 = 끔 = 옛 동작 비트 동일.
    #   예) --force_presets dolly_in,track_dolly_in
    parser.add_argument("--force_presets", default="", type=str)
    parser.add_argument("--external_shapes", default=None, type=str)
    # D83: 2 → 4. 모양 뱅크가 12개(6라벨)에서 188개(47라벨)로 늘었는데 영상당 2개만 뽑으면
    # 코퍼스가 그 다양성을 못 본다. 4면 변이의 약 1/3 이 free-moving(실제 촬영 궤적)이 된다.
    #
    # [2026-09-05] 4 → 1 (사용자 지시). **fit 예산이 dd 로 새고 있었다.** D129 118편 실측:
    # `dd_*` 가 전체 20,658행 중 12,528행(60.6%)인데 최종 학습 리스트는 dd10 서브샘플이라
    # `dd_*` 가 10% 다. 그 10% 의 절대 개수는 **비-dd 행 수**가 정하므로
    # (`round(N_nondd·0.1/0.9)`, `filter_seg_list_by_preset.py:subsample_to_frac`)
    # dd 를 더 구워도 쓰는 양은 안 늘고 버리는 양만 는다 — 280편 외삽으로 필요량 약 920행 대
    # 굽는 양 약 10,700행, 12배 과잉.
    #
    # 비용이 4배가 아니라 그 이상 붙는 이유: `pick_external` 은 anchor 마다 num 개를 뽑는데
    # `sample_camera_bank` 는 preset 을 **합집합**으로 받아 (anchor × preset) 격자를 전부 돈다.
    # anchor 6개 씬은 dd preset 24종 × anchor 6 × 사다리 4단이 통째로 fit 에 들어간다.
    #
    # 1 로 내려도 다양성은 안 죽는다 — `pick_external` 의 씨앗이 `(video, node_id)` 라
    # anchor·씬마다 다른 shape 을 뽑고 라벨 47종 커버리지는 코퍼스 전체에서 유지된다.
    # 남는 dd 도 여전히 dd10 필요량의 약 3배라 stride 서브샘플이 고를 여지가 있다.
    # 옛 굽기 스크립트(`run_dynpose_d{107,122,129}_shard.sh`, `probe_dynpose_scale_mode.sh`)는
    # `--num_external 4` 를 **명시**로 넘긴다 — 그 run 이 실제로 무엇으로 돌았는지의 기록이라
    # 안 건드렸다. 따라서 이 기본값 변경은 **진행 중인 D129 에 영향이 없다.**
    parser.add_argument("--num_external", default=1, type=int)
    # args = `--nodes X --presets ...` 한 줄, presets/nodes = 이름만, table = 사람용 표
    parser.add_argument("--emit", default="table", type=str,
                        choices=("table", "args", "presets", "nodes"))
    parser.add_argument("--out", default=None, type=str)
    # D188: `--emit args` 가 `--preset_route <out>` 을 **항상** 붙인다 (기본은 grid2x2/free 일 때만).
    # 예산(`--target_variants`)과 fallback 사다리는 그 JSON 을 통해서만 하류에 전달된다.
    parser.add_argument("--emit_route", dest="emit_route", action="store_true", default=False)
    parser.add_argument("--no_emit_route", dest="emit_route", action="store_false")
    # D184: anchor 가 0개(또는 화이트리스트 뒤 슬롯이 0개)면 assert 로 죽는 대신 **rc=3** 으로
    # 나간다. 기본 off = 옛 동작(assert). 전량 굽기에서만 켠다 — `--anchor_min_drift_u` 를
    # 코퍼스 중앙값으로 두면 절반 이상의 씬이 여기서 걸리는데, 크래시로 처리하면 `skipped.json`
    # 이 안 남아 라운드마다 같은 씬을 영원히 다시 집는다. `run_bank.py` 가 rc=3 을 받아
    # `skipped.json(no_surviving_anchors)` 를 남긴다.
    parser.add_argument("--skip_if_empty", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
