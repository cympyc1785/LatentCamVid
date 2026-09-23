"""LBM 의 trajectory preset 17종을 우리 `traj.py` primitive 로 옮긴다.

왜 preset 인가: 49프레임을 VLM 이 프레임 단위로 정의하게 두면 §계획서가 폐기한 그 문제로 돌아간다
— 모델이 공간 제약을 모른 채 궤적을 짓는다. preset 은 **어휘를 유한하게** 만들고, 크기는 τ 예산이
정하고, 시작점은 후보 board 가 정한다. VLM 이 고르는 건 "어떤 종류의 움직임인가" 하나다.

**preset 은 카메라 로컬 상대 궤적**(`rel[0]=I`, OpenCV 축)이다. world 로 올리는 건
`decode/build_poses.py` 가 `traj.start_at(c2w0, rel)` 로 한다. 그래야 시작 pose 와 움직임이
분리되고, 같은 preset 을 다른 후보 위에 그대로 얹을 수 있다.

## 세 가지 조준 방식 (`aim`) — 이 파일의 유일한 설계 결정

같은 preset 이라도 "카메라가 어디를 보는가"는 세 갈래다:

  `aim="look_at"`  위치만 궤적에서 가져오고 **방향은 매 프레임 subject 를 향해 다시 세운다**.
                   arc / orbit / crane / s_curve / *_hold 처럼 **정의 자체가 subject 중심**인
                   움직임이 여기 속한다. 이래야 §B5 의 `tracking` (world/drift/lock) 손잡이가
                   의미를 갖는다.
  `aim="free"`     **조준을 아예 안 한다** — 회전은 소스 frame0 회전 위에 궤적 회전만 얹은
                   `basis @ rel_local[f]` 다. free-moving primitive(truck / dolly / pedestal /
                   pan / tilt)가 전부 여기 속하고, **이쪽이 그 primitive 들의 기본값**이라 이름에
                   표시가 없다 (D90).
  `aim="traj"`     조준 위에 궤적 회전을 얹는다 (`look_at(anchor) @ rel_local[f]`).
                   **현재 이 값을 쓰는 preset 은 없다** — 순수 병진 preset 은 `rel_local` 회전이
                   항등이라 `traj` 가 `look_at` 으로 붕괴하고(§D90), 회전이 있는 preset 은
                   조준과 겹치면 안 되므로 `free` 다. 규약만 남겨 둔다.

free-moving 인데 **target 이 있는** 특수 케이스는 이름에 `_look_at` 을 붙인다 (D90). 사용자 지시로
`dolly_in` 계열만 만든다 — `dolly_in_look_at` / `track_dolly_in_look_at` 둘뿐이고 `aim="look_at"`
이다 (병진 preset 이라 `traj` 와 같은 카메라가 나온다).

### D90 — 왜 `free` 를 새로 만들었나 (실측)

D76~D89 의 `aim="traj"` 는 free-moving primitive 에서 **no-op 이었다**. `keyframe_aim="preset_rel"`
이 `look_at(anchor) @ rel_local[f]` 인데 순수 병진 preset 은 `rel_local` 회전이 **정확히 항등**
이라(`truck`/`dolly`/`pedestal` 전부 0.000°) `target` 으로 붕괴한다. `tru_0ac97866_a08` 49프레임
실측:

    dolly_in    vs dolly_in_dont_look       |Δt| 0.0   |ΔR| 0.022°   ← 같은 카메라
    dolly_out   vs dolly_out_dont_look      |Δt| 0.0   |ΔR| 0.023°
    static_hold vs static_hold_dont_look    |Δt| 0.0   |ΔR| 0.017°   (D94 이후 이름:
                                            `static_look_at` vs `static_hold`)

`truck_left` 는 frame0→48 총 회전 71.3° 인데 subject 조준오차가 3.03° — 그 71° 가 전부 조준이라
**이름만 truck 이고 실제로는 orbit** 이었다. pan 은 `rel_local` 회전이 살아 있어 조준오차 12.1°
로 다르지만, 기준이 소스 방향이 아니라 조준이므로 여전히 subject 가 앵커다.

즉 `_dont_look` 이라는 어휘가 학습 캡션에는 나가는데 카메라는 안 바뀌고 있었다. D90 은 그
어휘를 없애고 **free-moving 은 기본이 targetless**, target 이 붙는 쪽에만 `_look_at` 을 적는다.

즉 `tracking` 은 `aim="look_at"` preset 에만 적용된다. `aim="free"`/`aim="traj"` 인데 tracking 을
주면 decode 가 무시하고 `decision.json` 에 `tracking_ignored: true` 를 남긴다.

## 크기는 τ 로 닫는다

각 preset 은 "모양"만 정의하고 이동량은 `fit_tau` 가 이분법으로 맞춘다 — `se3.scale_traj` 로
전체를 s 배 해가며 `max_f |p(f) − p_src(f)| / z_med` 가 `target_tau` 가 되는 s 를 찾는다.
회전은 SE(3) 로그 공간에서 같이 줄어든다(스케일 불변이 아니다) — orbit 의 sweep 이 τ 를 통해
자동으로 제한된다는 뜻이고, 그게 의도다. 시작 pose 자체가 이미 τ 를 먹고 있으므로
`s=0` 에서의 τ 가 target 을 넘으면 s=0 으로 두고 `tau_saturated` 를 세운다.

예시:
    python -m lbm.presets --list
"""
import sys
from os import path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
RECAMMASTER_ROOT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "video_generation", "tools", "recammaster"))
if RECAMMASTER_ROOT not in sys.path:
    sys.path.insert(0, RECAMMASTER_ROOT)

import se3  # noqa: E402
import traj as T  # noqa: E402

NUM_FRAMES = 49

# 이동/회전의 "1x" 크기. 실제 값은 fit_tau 가 다시 정하므로 여기 숫자는 **모양의 비율**일 뿐이다.
# d 는 subject 까지 거리(radius) 대비 비율, sweep 은 degree.
DEFAULT_SHAPE = {"dolly_frac": 0.35, "lateral_frac": 0.35, "sweep_deg": 45.0, "pan_deg": 20.0}


def _dense(builder, num_frames: int, dense_mult: int = 8):
    """primitive 를 조밀하게 만들어 둔다 (speed 재파라미터화용)."""
    return builder((num_frames - 1) * dense_mult + 1)


def _resample(dense: np.ndarray, num_frames: int, speed: str = "steady"):
    """`speed` 에 따라 조밀 궤적에서 프레임을 골라낸다. **보간 없이 index pick.**

    행렬을 보간하면(lerp) SE(3) 를 벗어나므로 `canonical_cams.build_dl3dv:128` 과 같은 방식으로
    조밀하게 만들어 놓고 인덱스만 고른다. 8배 조밀이면 ease 곡선의 양자화 오차가 프레임의 1/8 이다.
    """
    a = np.linspace(0.0, 1.0, num_frames)
    if speed == "accel":
        a = a ** 2
    elif speed == "decel":
        a = 1.0 - (1.0 - a) ** 2
    elif speed == "ease":
        a = a * a * (3.0 - 2.0 * a)  # smoothstep
    elif speed != "steady":
        raise ValueError(f"모르는 speed: {speed}")
    return dense[np.rint(a * (len(dense) - 1)).astype(int)]


def _s_curve(sweep_deg: float, radius: float, num_frames: int):
    """전반부 arc(+σ/2) → 후반부 arc(−σ/2). 두 번째를 첫 끝점에 이어 붙인다."""
    half = (num_frames + 1) // 2
    first = T.arc(sweep_deg / 2, radius, half)
    second = T.start_at(first[-1], T.arc(-sweep_deg / 2, radius, num_frames - half + 1))
    return np.concatenate([first, second[1:]], axis=0)


# preset 이름 -> (모양 builder, aim, zoom 필요 여부).
# builder(n, ctx) -> (n,4,4) 카메라 로컬 rel c2w. ctx = {"radius", "dolly", "lateral", "sweep", "pan"}
# 부호 규약 (traj.py 실측): rot_y(+) = 오른쪽으로 yaw, arc(+) = 카메라가 왼쪽으로 가며 오른쪽을 봄,
#                          truck(+) = 오른쪽, pedestal(+) = 위, dolly(+) = 전진.
#
# ── 이름 규칙 (D75 → D76 개정). 어휘가 학습 캡션에 그대로 나가므로 이름이 곧 텍스트 축이다:
#     [track_]<primitive>_<direction>[_<primitive2>_<direction2>][_dont_look]
#   primitive  dolly(광축) / truck(좌우) / pedestal(상하) / pan(제자리 yaw) / arc(곡선 접근) /
#              orbit(등거리 선회) / crane(호를 그리며 상하) / hold(정지) / s_curve
#   direction  in·out / left·right / up·down
#   track_     `PRESET_FOLLOW` 로 subject 변위를 카메라 위치에 싣는다 (gain 1.0)
#   _look_at   **비대칭 표기 (D90)**. free-moving primitive(truck/dolly/pedestal/pan/tilt)는
#              **기본이 targetless**(aim="free") 이고 그건 이름에 안 적는다. 그 기본에서 벗어나
#              subject 를 조준하는 쪽에만 `_look_at` 을 붙인다. D76 의 `_dont_look` 은 정확히
#              반대 방향의 비대칭이었는데, 그 기본값(=조준)이 free-moving 이라는 말과 모순이라
#              뒤집었다. arc/orbit/crane/hold 는 정의가 subject 중심이라 접미사가 없다.
#   _dont_look D76 어휘. 전부 `PRESET_ALIASES` 로 남았다 — **이름이 뜻하던 동작을 코드가 한 적이
#              없으므로**(위 실측) 별칭 대상은 `free` 쪽이 아니라 그때 실제로 나오던 카메라다.
# 규칙에 안 맞던 옛 이름(`straight_ease` 는 speed 어휘, `orbit_*_arc` 는 arc 가 아니라 true_orbit,
# `*_reveal` 은 동작이 아니라 의도, `static_subtle_zoom` 의 "subtle" 은 캡션에서 뺀 크기 부사,
# D75 의 `_aimed`/`_locked`)은 전부 `PRESET_ALIASES` 에 남겨 뒀다.
# **단 하나 의미가 뒤집힌 이름이 `dolly_in`/`dolly_out` 이다** — D75 에서는 "locked"(aim="traj")
# 의 별칭이었는데 D76 에서 정식 이름(aim="look_at")이 됐다. 이미 디스크에 있는 뱅크 4,065행이
# 옛 뜻으로 이 문자열을 쓰고 있으므로 `LEGACY_AIM_COLLISIONS` + `resolve_preset(name, aim=...)`
# 로 **행이 들고 있는 `aim` 필드를 보고** 가른다 (뱅크를 다시 쓰지 않는다).
PRESETS = {
    # ── dolly (직선, 광축). free-moving 이라 기본이 targetless.
    #   (접미사 없음) 소스 frame0 방향 그대로 전진/후진 — 순수 dolly
    #   _look_at      매 프레임 subject 로 재조준. **`dolly_in` 에만 둔다** (사용자 지시) —
    #                 "다가가면서 대상을 놓치지 않는" push-in 이 실제로 쓰이는 유일한 조합이다.
    #                 `dolly_out_look_at` 은 pull_out_arc 계열이 이미 덮는다.
    "dolly_in":     (lambda n, c: T.dolly(c["dolly"], n), "free", False),
    "dolly_out":    (lambda n, c: T.dolly(-c["dolly"], n), "free", False),
    "dolly_in_look_at":  (lambda n, c: T.dolly(c["dolly"], n), "look_at", False),
    # ── arc (전후진 + 곡선). 방향을 이름에 다 적는다 (예전엔 4칸 중 2칸만 적혀 있었다).
    "push_in_arc_left":   (lambda n, c: T.compose(T.arc(c["sweep"] / 2, c["radius"], n),
                                                  T.dolly(c["dolly"], n)), "look_at", False),
    "push_in_arc_right":  (lambda n, c: T.compose(T.arc(-c["sweep"] / 2, c["radius"], n),
                                                  T.dolly(c["dolly"], n)), "look_at", False),
    "pull_out_arc_left":  (lambda n, c: T.compose(T.arc(c["sweep"] / 2, c["radius"], n),
                                                  T.dolly(-c["dolly"], n)), "look_at", False),
    "pull_out_arc_right": (lambda n, c: T.compose(T.arc(-c["sweep"] / 2, c["radius"], n),
                                                  T.dolly(-c["dolly"], n)), "look_at", False),
    # ── orbit (등거리 선회). `T.true_orbit` 이라 `_arc` 접미사는 틀린 말이었다.
    "orbit_left":         (lambda n, c: T.true_orbit(c["sweep"], c["radius"], n), "look_at", False),
    "orbit_right":        (lambda n, c: T.true_orbit(-c["sweep"], c["radius"], n), "look_at", False),
    # ── pan / tilt / truck / pedestal — free-moving primitive. 전부 aim="free" (D90).
    # tilt 부호는 **실측**이다 (추론 금지): `T.tilt(+20°)` 의 forward 가 [0, -0.342, 0.94] 인데
    # OpenCV +Y 가 아래이므로 forward_y < 0 = 위를 본다. 즉 `T.tilt(+) = tilt up`.
    # (같은 방식으로 확인한 것: `T.pan(+) = yaw right`.)
    "pan_left":           (lambda n, c: T.pan(-c["pan"], n), "free", False),
    "pan_right":          (lambda n, c: T.pan(c["pan"], n), "free", False),
    "tilt_up":            (lambda n, c: T.tilt(c["pan"], n), "free", False),
    "tilt_down":          (lambda n, c: T.tilt(-c["pan"], n), "free", False),
    "truck_left":         (lambda n, c: T.truck(-c["lateral"], n), "free", False),
    "truck_right":        (lambda n, c: T.truck(c["lateral"], n), "free", False),
    "pedestal_up":        (lambda n, c: T.pedestal(c["lateral"], n), "free", False),
    "pedestal_down":      (lambda n, c: T.pedestal(-c["lateral"], n), "free", False),
    # ── crane (호를 그리며 상하 + 재조준). pedestal 과 달리 subject 를 계속 본다.
    "crane_up":           (lambda n, c: T.crane(c["lateral"], c["radius"], n), "look_at", False),
    "crane_down":         (lambda n, c: T.crane(-c["lateral"], c["radius"], n), "look_at", False),
    "s_curve":            (lambda n, c: _s_curve(c["sweep"], c["radius"], n), "look_at", False),
    # ── 합성 2종. 이름은 `<주동작>_<부동작>` 순서로 통일했다.
    #   pan_right_zoom_out     — 조준 없이(aim="free") 오른쪽 pan + **intrinsic** zoom out.
    #                            이동이 0 이라 τ 로 크기를 못 정한다 (ROTATION_ONLY_PRESETS).
    #   orbit_left_pedestal_up — 왼쪽 궤도 + 상승을 동시에. aim="look_at" 이라 tracking="world"
    #                            를 주면 **frame0 subject 중심**을 계속 본다.
    "pan_right_zoom_out":     (lambda n, c: T.pan(c["pan"], n), "free", True),
    "orbit_left_pedestal_up": (lambda n, c: T.compose(T.true_orbit(c["sweep"], c["radius"], n),
                                                      T.pedestal(c["lateral"], n)),
                               "look_at", False),
    #   orbit_left_pedestal_down — 위의 하강 짝 (사용자 지시 2026-09-21 "hike도 orbit left,
    #                            pull out orbit left, orbit left pedestal down"). 부호만 뒤집는다
    #                            (`pedestal_down_dolly_in` 이 쓰는 `-c["lateral"]` 과 같은 규약).
    #                            aim 은 `_up` 과 같은 `look_at` — 궤도가 주동작이라 조준이 필요하다.
    "orbit_left_pedestal_down": (lambda n, c: T.compose(T.true_orbit(c["sweep"], c["radius"], n),
                                                        T.pedestal(-c["lateral"], n)),
                                 "look_at", False),
    #   pedestal_down_dolly_in — 하강하면서 전진 (사용자 지시 2026-09-21 "vista golf scene을
    #                            pedestal down, pedestal down and push in으로 돌려봐줘").
    #                            `orbit_left_pedestal_up` 과 같은 `<주동작>_<부동작>` 표기.
    #                            aim 은 두 구성 primitive 와 같은 `free` 다 — pedestal 도
    #                            dolly_in 도 조준을 안 하므로(조준은 `_look_at` 으로만 표기,
    #                            D90 비대칭) 여기만 `look_at` 으로 두면 어휘 규칙이 깨진다.
    "pedestal_down_dolly_in": (lambda n, c: T.compose(T.pedestal(-c["lateral"], n),
                                                      T.dolly(c["dolly"], n)),
                               "free", False),
    # ── static (카메라가 안 움직인다). free-moving 이 아니라 `_look_at` 규칙 밖이지만, `hold` 의
    # `rel_local` 회전도 항등이라 D89 까지 `_dont_look` 이 no-op 이었다 (실측 |ΔR| 0.017°).
    # D90 은 이름을 그대로 두고 aim 만 `free` 로 고쳤는데, 그러면 `static_hold` 가 "제자리에서
    # 눈으로 subject 를 쫓는" 카메라를 뜻하게 된다 — **`hold` 는 안 움직이고 재조준도 안 한다는
    # 뜻이어야 한다** (사용자 지시, D94). 그래서 이름 두 개를 맞바꾼다:
    #   `static_hold`(옛 `static_hold_dont_look`)  진짜 locked-off. 위치·조준 둘 다 고정
    #   `static_look_at`(옛 `static_hold`)         제자리, 조준만 subject 추종
    # 디스크의 옛 뱅크가 `static_hold` 를 옛 뜻(aim="look_at")으로 들고 있으므로 `dolly_in` 때와
    # 같은 방식 — `LEGACY_AIM_COLLISIONS` 로 **행이 기록한 aim 을 보고** 가른다 (뱅크 안 고침).
    "static_hold":  (lambda n, c: T.hold(np.eye(4), n), "free", False),
    "static_look_at": (lambda n, c: T.hold(np.eye(4), n), "look_at", False),
    "static_zoom_in":     (lambda n, c: T.hold(np.eye(4), n), "free", True),
    # ── tracking shot 12종 (D74/D75). 지금까지 preset 은 "카메라가 어떻게 움직이나"만 정의했고
    # **subject 를 따라가는 축은 `--follow_gains` CLI 에만** 있었다 — 그래서 배포 뱅크 51편
    # 16,748 변이의 `follow_gain` 이 전량 0 이고 학습 캡션 어휘에 track/follow 가 0건이다.
    # 여기서는 그 축을 preset 이름에 묶어 **추종을 어휘로** 만든다. 모양은 위 preset 과 같고
    # 다른 건 `PRESET_FOLLOW` 가 gain 을 켠다는 점뿐이다 (gain 1.0 = 카메라 변위 = subject
    # 변위 = 상대 위치 유지, 이게 tracking shot 의 정의다).
    # **이름은 `track_` + 위 비-track 이름 그대로**다. 유일한 예외가 hold 로, 비-track 쪽은
    # `static_`(안 움직인다) 이고 track 쪽은 `track_`(subject 와 같이 움직인다) — 접두어 자체가
    # 그 차이를 말한다.
    # D94 는 static 과 **똑같은 맞바꿈**을 여기에도 건다 (사용자 지시). `hold` = 위치·조준 둘 다
    # 고정, 조준이 붙는 쪽은 `_look_at`:
    #   `track_hold`(옛 `track_hold_dont_look`)  subject 변위만 싣고 재조준 없음
    #   `track_look_at`(옛 `track_hold`)         subject 변위 + 매 프레임 재조준
    # 이제 어휘 전체가 한 규칙이다 — **조준은 `_look_at` 으로만 표기**하고(D90 비대칭),
    # `hold` 는 어디서든 "안 움직이고 재조준도 안 한다"를 뜻한다. arc/orbit/crane/s_curve 는
    # 정의 자체가 subject 중심이라 여전히 접미사가 없다.
    "track_hold":     (lambda n, c: T.hold(np.eye(4), n), "free", False),
    "track_look_at":  (lambda n, c: T.hold(np.eye(4), n), "look_at", False),
    "track_truck_left":     (lambda n, c: T.truck(-c["lateral"], n), "free", False),
    "track_truck_right":    (lambda n, c: T.truck(c["lateral"], n), "free", False),
    "track_dolly_in":  (lambda n, c: T.dolly(c["dolly"], n), "free", False),
    "track_dolly_out": (lambda n, c: T.dolly(-c["dolly"], n), "free", False),
    # `_look_at` 은 비-track 과 같은 규칙으로 `dolly_in` 에만 (위 dolly 블록 주석 참조).
    "track_dolly_in_look_at": (lambda n, c: T.dolly(c["dolly"], n), "look_at", False),
    "track_orbit_left":     (lambda n, c: T.true_orbit(c["sweep"], c["radius"], n),
                             "look_at", False),
    "track_orbit_right":    (lambda n, c: T.true_orbit(-c["sweep"], c["radius"], n),
                             "look_at", False),
    # 수직축 4칸. 비-track 어휘에는 pedestal(순수 수직, aim="traj") 과 crane(재조준, aim="look_at")
    # 이 각각 위/아래 짝을 다 갖고 있는데 track 쪽은 crane-up 하나뿐이었다 — 여기서 나머지를 채운다.
    # 아래로 가는 둘은 위 짝보다 **자주 게이트에서 탈락한다**: micro-adjust 의 elevation clamp
    # [5°, 70°] 와 "지면 아래 금지"가 하강에만 걸린다. 그래서 뱅크 탈락률은 재보는 게 맞다 (가정 X).
    "track_crane_up":       (lambda n, c: T.crane(c["lateral"], c["radius"], n), "look_at", False),
    "track_crane_down":     (lambda n, c: T.crane(-c["lateral"], c["radius"], n), "look_at", False),
    "track_pedestal_up":    (lambda n, c: T.pedestal(c["lateral"], n), "free", False),
    "track_pedestal_down":  (lambda n, c: T.pedestal(-c["lateral"], n), "free", False),
    # 합성 1칸 (사용자 지시 2026-09-22 "588도 track dolly in, track pedestal down,
    # track dolly in pedestal down"). 비-track 짝 `pedestal_down_dolly_in` 의 track 판이라
    # 이름은 `track_` + 그 이름 그대로다 — 요청 문장은 "dolly in pedestal down" 순서지만
    # 어휘 규칙이 `<주동작>_<부동작>` 이고 비-track 쪽이 pedestal 을 주동작으로 이미 잡아
    # 뒀으므로, 같은 동작에 이름이 둘 생기지 않게 그 순서를 따른다 (동작은 동일: 하강+전진).
    # aim 은 비-track 짝과 같은 `free` — pedestal 도 dolly_in 도 재조준을 안 한다 (D90 비대칭).
    # `PRESET_FOLLOW` 가 접두어로 gain 을 켜므로 추종 배선은 여기 말고 손댈 곳이 없다.
    "track_pedestal_down_dolly_in": (lambda n, c: T.compose(T.pedestal(-c["lateral"], n),
                                                            T.dolly(c["dolly"], n)),
                                     "free", False),
    # arc 4칸 (D82). 이름 규칙상 `track_` + 비-track 이름이므로 DSL 에 새 어휘가 생기는 게 아니라
    # 기존 `arc` primitive 의 빈 칸이 채워지는 것이다. 추종이 arc 에서 특히 의미가 있는 이유:
    # arc 는 **곡선으로 접근/후퇴하며 재조준**(aim="look_at")하는 동작이라, subject 가 움직이면
    # 곡률 중심이 같이 끌려가야 "따라 돌며 다가간다"가 된다. 추종을 안 걸면 subject 가 호 밖으로
    # 빠져나가고 카메라만 빈 공간을 돈다. 라우터의 `arc` 슬롯이 이 4칸을 쓴다.
    "track_push_in_arc_left":   (lambda n, c: T.compose(T.arc(c["sweep"] / 2, c["radius"], n),
                                                        T.dolly(c["dolly"], n)), "look_at", False),
    "track_push_in_arc_right":  (lambda n, c: T.compose(T.arc(-c["sweep"] / 2, c["radius"], n),
                                                        T.dolly(c["dolly"], n)), "look_at", False),
    "track_pull_out_arc_left":  (lambda n, c: T.compose(T.arc(c["sweep"] / 2, c["radius"], n),
                                                        T.dolly(-c["dolly"], n)), "look_at", False),
    "track_pull_out_arc_right": (lambda n, c: T.compose(T.arc(-c["sweep"] / 2, c["radius"], n),
                                                        T.dolly(-c["dolly"], n)), "look_at", False),
}

# **모양의 크기가 0** 인 preset — τ 이분법이 의미 없다 (어떤 s 든 같은 궤적이라 τ 가 안 변한다).
# `track_hold*` 도 여기 든다: 모양은 identity 고 움직임은 전부 follow offset 에서 나오므로
# 스케일 s 가 궤적을 못 바꾼다. 이름은 "정지"지만 판정 기준은 "τ 가 s 에 반응하나"다.
STATIC_PRESETS = ("static_hold", "static_look_at", "static_zoom_in",
                  "track_hold", "track_look_at",
                  # D94 이전 이름 — 옛 뱅크 행이 (resolve 전 문자열로도) 그대로 걸리게 남긴다
                  "static_hold_dont_look", "track_hold_dont_look")

# preset 이름 -> `follow_gain` 기본값 (D74). subject 변위를 카메라 **위치**에 싣는 비율.
# 값이 없는 preset 은 예전 그대로 CLI / `decision.json` 이 정한다 (기본 0 = 추종 없음) —
# 즉 이 dict 는 기존 preset 의 동작을 한 톨도 안 바꾼다. `PRESET_ZOOM_END` 와 같은 형태다.
# 1.0 을 쓰는 이유: `auto` 는 τ 를 **최소화**하는 gain 을 풀기 때문에 정지 subject 나 소스가
# 이미 따라가고 있는 구간에서 0 에 가깝게 나온다 — 그러면 캡션은 "tracks" 인데 궤적은 추종이
# 아니게 된다. 예산이 모자라면 `fit_tau` 가 모양만 깎고, 그래도 넘치면 `tau_saturated` 로
# 뱅크에서 빠진다 (조용히 추종을 끄지는 않는다).
PRESET_FOLLOW = {name: 1.0 for name in PRESETS if name.startswith("track_")}

# preset 이름 -> `tracking` 기본값 (D93 -> D127). `PRESET_FOLLOW` 가 **위치** 추종률이라면
# 이쪽은 **조준점** 추종률이다 (`build_poses.TRACKING_GAIN` = world 0.0 / lock 1.0).
#
# D93 이 `track_*` 만 덮은 이유 — 둘이 따로 놀았다. `track_*` 은 `PRESET_FOLLOW` 로 위치를 100%
# 추종시키는데 뱅크 드라이버(`run_k6_d77_shard.sh`)는 `--trackings` 를 안 넘겨 조준이 기본
# `drift`(60%) 로 갔다. 위치 100% / 조준 60% 로 어긋나면 subject 가 프레임 밖으로 밀린다 —
# 실측: snowboard `dyn_0` 에서 `track_pull_out_arc_left` `subject_in_frame` 0.40,
# `track_crane_up` 0.20, 둘 다 `subject_area_med` 0.0000. 같은 행을 `lock` 으로만 바꾸면 전부
# 1.00 / 0.067·0.122 다 (`results/20260901_snowboard_track_lock_vs_drift/`).
#
# D127 은 그걸 **조준하는 preset 전체**로 넓힌다 (사용자 확정 라우팅 규칙):
#   정지 target : track 없음 + object-centric,  `look_at` 은 lock
#   동적 target : track + object-centric,       `look_at` 은 lock
#                 예외: `aim="free"` (free-moving) 는 조준 자체가 없어 tracking 이 무시된다
#                 (`build_poses` 의 `tracking_ignored`)
# 즉 판정 기준은 이름(`track_`)이 아니라 **aim 이 look_at 인가**다. object-centric 12종
# (crane/orbit/arc/s_curve/dolly_in_look_at/static_look_at/orbit_left_pedestal_up)이 D126 까지
# `drift` 로 굽혀 있었다. parkour 7-preset 실측(`results/20260904_tracking_drift_vs_lock/`,
# 172행): 동적 `dyn_0` 28행 `subject_in_frame` 0.9670 -> 1.0000 (`s_curve` 0.769 -> 1.000),
# 정지 `stat_*` 144행 Δ 0.0000 — 정지 anchor 는 변위가 0 이라 정의상 no-op 이다.
#
# `PRESET_FOLLOW` 와 달리 **요청값을 덮어쓴다**. sentinel 이 없기 때문이다 — 뱅크 행은 전부
# `tracking` 을 명시적으로 들고 있어서 "안 준 경우"를 가릴 수가 없다. 끄는 스위치는 남긴다
# (`--no_preset_tracking`) — 조준 preset 에 굳이 `world`(frame0 고정 조준)를 주고 싶을 때만.
PRESET_TRACKING = {name: "lock" for name, (_, aim, _) in PRESETS.items() if aim == "look_at"}

# D127. `drift` 를 어휘에서 지웠으므로(§build_poses.TRACKING_GAIN) 디스크의 옛 뱅크 행이 들고
# 있는 문자열을 읽는 시점에 승격시킨다. `world`(0.0) 로 내리지 않고 `lock`(1.0) 으로 올리는 쪽을
# 고른 이유: drift 는 "따라가되 뒤처진다" 였고, 규칙이 정한 답은 "따라간다" 다. 옛 뱅크를 그대로
# 재현할 방법은 이제 없다 — 그게 "아예 지운다"의 뜻이다 (사용자 지시).
LEGACY_TRACKING = {"drift": "lock"}


def resolve_tracking(preset: str, tracking: str, enabled: bool = True) -> str:
    """`aim="look_at"` preset 이면 조준 추종률을 `lock` 으로. `enabled=False` 면 요청값 그대로.

    어느 쪽이든 삭제된 `drift` 는 `lock` 으로 승격시킨다 (§LEGACY_TRACKING) — `enabled=False` 가
    옛 값을 되살리는 통로가 되면 "지웠다"가 아니게 된다.
    """
    tracking = LEGACY_TRACKING.get(tracking, tracking)
    if not enabled:
        return tracking
    return PRESET_TRACKING.get(PRESET_ALIASES.get(preset, preset), tracking)


# preset 이름 -> `tau_ref` 기본값 (D97). τ 를 **무엇에 대한** 변위로 잴지 고르는 축이다.
#   "source" — τ(f) = |p_plan(f) − p_src(f)| / z_med   (D97 이전의 유일한 정의)
#   "follow" — τ(f) = |p_shape(f) − p_start| / z_med   (추종 궤적 위에서의 **상대** 변위)
#
# 왜 필요한가 — `track_*` 은 위치를 100% 추종(`PRESET_FOLLOW`)하므로 카메라가 subject 를 따라
# 세계를 가로지른다. 그런데 τ 는 그 변위 전체를 소스 카메라 기준으로 재기 때문에, **추종 자체가
# 예산을 먹고** 정작 사람이 보라고 준 preset 모양(truck_left 의 왼쪽 미끄러짐)에 남는 몫이 없다.
# snowboard `bank_d94` 실측: `track_truck_left` 는 `tau_start` 0.1992~0.2211 인데 사다리 아랫단
# `target_tau` 가 0.20/0.35 라 모양에 0.14 밖에 안 남는다.
#
# 게다가 두 성분은 서로 **상쇄**한다. `hole_bank_k6_d94` 12행 실측에서 follow 순변위와 shape
# 순변위의 사잇각이 145.6~146.1° 였다 (`|follow|` 3.70~3.75 u, `|shape|` 4.53~5.44 u,
# 합쳐진 `|plan|` 2.53~3.15 u — 합보다 작다). subject 는 화면 오른쪽으로 가는데 truck_left 는
# 왼쪽으로 밀어서, 세계에서는 거의 제자리인 카메라가 "추종도 하고 트럭도 하는" 것처럼 기록된다.
#
# `follow` 로 재면 τ 가 `|p_shape − p_start|` 하나만 보므로 (a) 추종이 예산을 안 먹고
# (b) `tau_start` 가 정확히 0 이라 `tau_saturated` 가 원리적으로 안 뜬다. 즉 τ 손잡이가
# "얼마나 빠르게 따라붙거나 뒤처지나"라는 **상대 속도**를 직접 돌리게 된다.
#
# `PRESET_FOLLOW` 와 같은 sentinel 방식이다 — 요청값이 `"auto"` 일 때만 여기가 이긴다.
# 명시적으로 `--tau_ref source` 를 주면 D97 이전과 비트 동일하다.
PRESET_TAU_REF = {name: "follow" for name in PRESETS if name.startswith("track_")}

TAU_REF_CHOICES = ("source", "follow", "auto")

# D140 의 `suspect` 태그 전체 목록. 다는 쪽(`fit_hole_ladder.tag_suspects`)과 자르는 쪽
# (`vista4d_bank_to_dl3dv.py --drop_suspect`) 이 서로 다른 파일이라, 목록을 각자 들고 있으면
# 조용히 어긋난다 — 실제로 ④⑤ 가 추가된 뒤 emit 쪽 `choices` 는 ①②③ 에 머물러 있었고,
# d192 뱅크에서 실제로 붙은 태그(④ 518행 / ⑤ 500행)를 **지정할 방법이 없었다**.
# 그래서 두 파일이 이미 같이 import 하는 이 모듈에 한 벌만 둔다.
SUSPECT_TAGS = ("aim_free_subject_lost", "static_start_collision", "motion_preset_no_motion",
                "aim_target_subject_lost", "hole_over_budget")


def resolve_tau_ref(preset: str, tau_ref: str) -> str:
    """`"auto"` 일 때만 preset 기본값으로. 그 외에는 요청값 그대로 (D97 이전 동작 보존)."""
    tau_ref = str(tau_ref)
    assert tau_ref in TAU_REF_CHOICES, f"tau_ref: {tau_ref} (가능: {TAU_REF_CHOICES})"
    if tau_ref != "auto":
        return tau_ref
    return PRESET_TAU_REF.get(PRESET_ALIASES.get(preset, preset), "source")


# preset 이름 -> **마지막 프레임의 focal 배율** (frame0 = 1.0). < 1 이 zoom out(화각 확대).
# focal 은 SE(3) 밖이라 `traj.py` primitive 가 아니라 여기서 따로 든다 — `fit_tau` 의 스케일도
# 안 먹는다(광학 zoom 은 시차를 안 만든다). 값이 없는 preset 은 1.0 = 고정.
PRESET_ZOOM_END = {"static_zoom_in": 1.15, "pan_right_zoom_out": 1.0 / 1.5}

# 옛 이름 -> 새 이름. **전부 의미 보존**이다 (같은 builder·aim 을 가리킨다) — 그래서 이미 나간
# 뱅크 16,748 변이와 `decision.json`, VLM 이 옛 어휘로 답하는 경우가 전부 그대로 돈다.
# `build_shape`/`focal_track` 이 조회 전에 여기를 먼저 통과시킨다.
# **예외 2개**(`dolly_in`, `dolly_out`)는 D76 에서 정식 이름이 되어 여기 없다 — `LEGACY_AIM_COLLISIONS`.
PRESET_ALIASES = {
    # 짧은 별칭 (VLM 이 자주 이렇게 답한다)
    # `static` 은 D94 이전에 "제자리 + 조준 추종"을 가리켰다 — 뜻을 보존하려고 새 이름으로 보낸다.
    "static": "static_look_at", "locked_off": "static_hold",
    "push_in": "push_in_arc_left", "pull_out": "pull_out_arc_right",
    "zoom_in": "push_in_arc_left", "zoom_out": "pull_out_arc_right",
    # D75 이름 정리 이전의 정식 이름
    "straight_ease": "dolly_in",                # "ease" 는 speed 축 어휘였다
    "push_in_arc": "push_in_arc_left", "pull_out_arc": "pull_out_arc_right",
    "orbit_left_arc": "orbit_left", "orbit_right_arc": "orbit_right",  # arc 가 아니라 true_orbit
    "rise_reveal": "crane_up", "drop_reveal": "crane_down",            # "reveal" 은 의도지 동작이 아니다
    "zoom_out_pan_right": "pan_right_zoom_out",  # 합성은 <주동작>_<부동작> 순서
    "static_subtle_zoom": "static_zoom_in",      # "subtle" 은 캡션에서 뺀 크기 부사였다
    "track_follow": "track_look_at", "track_follow_locked": "track_hold",
    "track_side_left": "track_truck_left", "track_side_right": "track_truck_right",
    "track_push_in": "track_dolly_in", "track_pull_out": "track_dolly_out",
    "track_rise": "track_crane_up", "track_drop": "track_crane_down",
    # D75 의 `_aimed`/`_locked` 대칭 표기 -> D76 비대칭 표기 -> D90 반대 방향 비대칭.
    "dolly_in_aimed": "dolly_in_look_at", "dolly_out_aimed": "dolly_out",
    "static_hold_aimed": "static_look_at", "static_hold_locked": "static_hold",
    "track_hold_aimed": "track_look_at", "track_hold_locked": "track_hold",
    "track_dolly_in_aimed": "track_dolly_in_look_at", "track_dolly_out_aimed": "track_dolly_out",
    # ── D90. `_dont_look` 어휘 은퇴. **별칭 대상은 "이름이 뜻하던 카메라"가 아니라 "그때 실제로
    # 나오던 카메라"** 다 — `dolly_in_dont_look` 은 aim="traj" 였지만 `rel_local` 회전이 항등이라
    # 조준된 카메라를 냈다 (실측 |ΔR| 0.022°). 그러니 옛 뱅크를 재현하려면 `_look_at` 으로
    # 보내야 pose 가 보존된다. `free` 로 보내면 71° 짜리 회전 차이가 조용히 생긴다.
    "dolly_in_dont_look": "dolly_in_look_at",
    "dolly_out_dont_look": "dolly_out_look_at_legacy",
    "dolly_in_locked": "dolly_in_look_at", "dolly_out_locked": "dolly_out_look_at_legacy",
    "track_dolly_in_dont_look": "track_dolly_in_look_at",
    # ── D94. `*_hold_dont_look` 이 쓰던 카메라가 이제 `*_hold` 라는 이름을 갖는다.
    # 여기 없으면 옛 뱅크 행이 `PRESETS[...]` 에서 KeyError 로 죽는다 (`PRESETS` 키에서 빠졌다).
    "static_hold_dont_look": "static_hold",
    "track_hold_dont_look": "track_hold"}

# D90 은퇴 이름 전용 재현 슬롯. `dolly_out_look_at` 은 **어휘에서 뺐지만**(사용자 지시: `_look_at`
# 은 dolly_in 만) 옛 뱅크 재현에는 필요하다 — VLM/캡션에 나가지 않도록 `PRESETS` 밖에 둔다.
LEGACY_ONLY_PRESETS = {
    "dolly_out_look_at_legacy": (lambda n, c: T.dolly(-c["dolly"], n), "look_at", False)}
PRESETS.update(LEGACY_ONLY_PRESETS)

# **의미가 뒤집힌 이름** (D75 -> D76). 키는 `(옛 이름, 그 행이 기록한 aim)`.
# `dolly_in` 은 D75 에서 `dolly_in_locked`(aim="traj") 의 별칭이었고 D76 에서 `aim="look_at"` 인
# 정식 이름이 됐다. 디스크의 뱅크 4,065행이 옛 뜻으로 이 문자열을 들고 있는데, 다행히 뱅크
# `variants[]` 행은 **`aim` 을 같이 기록**한다 (실측: 해당 행 2,579개 전부 `aim="traj"`).
# 그래서 이름만 보지 말고 aim 을 같이 보면 옛 행과 새 행을 정확히 가를 수 있다 — 진행 중인
# DynPose 뱅크 생성을 멈추고 파일 4,065행을 다시 쓰는 것보다 이쪽이 안전하다.
#
# D90 이 같은 문제를 한 겹 더 쌓았다. free-moving 이름 전부가 뜻이 바뀌었는데(조준 -> targetless),
# 다행히 판별 규칙은 그대로다 — **옛 행은 `aim` 이 "traj"/"look_at" 이고 새 행은 "free"** 다.
# `("truck_left", "traj")` 처럼 옛 aim 을 든 행은 그때 실제로 나오던 카메라(= 조준된 것)로
# 보내야 pose 가 보존된다. `_look_at` 어휘가 dolly_in 에만 있으므로 나머지는 `LEGACY_ONLY_PRESETS`
# 대신 `keyframe_aim="preset_rel"` 을 강제하는 `*_traj_legacy` 슬롯으로 받는다 (아래).
#
# D94 가 `static_hold` / `track_hold` 에 같은 문제를 만든다 — 옛 뜻은 "제자리 + 조준 추종"
# (aim="look_at"), 새 뜻은 "완전 고정"(aim="free"). 판별은 여기서도 `aim` 하나로 끝난다:
# **옛 행은 "look_at"(D75 별칭 시절 행은 "traj"), 새 행은 "free"**. 옛 행을 `*_look_at` 으로 보낸다.
# `track_hold` 쪽이 물량이 크다 — 배포 d77 뱅크에만 3,282 track 행이 있고 그 중 `track_hold` 는
# 전부 `aim="look_at"` 이다 (D93 까지 그게 이 preset 의 기본값이었다).
LEGACY_AIM_COLLISIONS = {("dolly_in", "traj"): "dolly_in_look_at",
                         ("dolly_out", "traj"): "dolly_out_look_at_legacy",
                         ("dolly_in", "look_at"): "dolly_in_look_at",
                         ("dolly_out", "look_at"): "dolly_out_look_at_legacy",
                         ("track_dolly_in", "look_at"): "track_dolly_in_look_at",
                         ("static_hold", "look_at"): "static_look_at",
                         ("static_hold", "traj"): "static_look_at",
                         ("track_hold", "look_at"): "track_look_at",
                         ("track_hold", "traj"): "track_look_at"}

# D90 에서 뜻이 뒤집힌 free-moving 이름 — 옛 행(`aim` 이 "traj")을 만나면 조준 동작으로 되돌린다.
# 이름은 그대로 두고 aim 만 갈아끼우면 되므로 별도 preset 슬롯이 필요 없다. `resolve_aim` 이
# 디코더에 넘길 aim 을 정한다.
D90_FLIPPED = ("pan_left", "pan_right", "truck_left", "truck_right",
               "pedestal_up", "pedestal_down", "pan_right_zoom_out",
               "static_hold_dont_look", "static_zoom_in", "track_hold_dont_look",
               "track_truck_left", "track_truck_right",
               "track_pedestal_up", "track_pedestal_down",
               "track_dolly_in", "track_dolly_out")


def resolve_aim(name: str, recorded_aim: str | None = None):
    """이 preset 을 디코드할 때 쓸 `aim`. 뱅크 행을 읽을 땐 **항상 그 행의 `aim` 을 넘길 것**.

    `recorded_aim` 이 D90 이전 값("traj"/"look_at")이면 그 값을 그대로 돌려준다 — 옛 뱅크를
    비트 단위로 재현하기 위해서다. 새로 굽는 뱅크는 `recorded_aim=None` 이므로 `PRESETS` 의
    새 기본값(`free`)을 탄다.
    """
    canonical = PRESETS[resolve_preset(name, recorded_aim)][1]
    if recorded_aim in ("traj", "look_at") and name in D90_FLIPPED:
        return recorded_aim          # 옛 행 — 그때 나오던 카메라를 그대로 재현
    return canonical


def resolve_traj_basis(traj_basis: str, aim: str) -> str:
    """`--traj_basis auto` 를 **조준 방식으로** 푼다 (D99). 명시값이면 그대로 = 기존 동작.

    왜 aim 이 기준인가 — `traj.true_orbit`/`arc`/`crane` 는 전부 "**카메라 로컬 +Z 로 `radius`
    앞의 점**"을 중심으로 정의돼 있다. 그 로컬 프레임이 `basis_c2w` 이므로 `traj_basis="source"`
    면 선회 중심이 소스 frame0 광축 위에 앉는다. 소스가 subject 를 정면으로 안 보고 있으면
    (off-axis θ) 중심이 OBB 중심에서 `2·r·sin(θ/2)` 만큼 밀린다. 조준은 `aim="look_at"` 이 매
    프레임 고쳐 주므로 **화면상 subject 는 맞는데 궤적 모양만 틀어진다** — 그래서 안 들켰다.

    코퍼스 실측 off-axis 중앙값: TRUMANS 22.43° (n=2101) / dynpose 16.89° (n=1413). 즉 선회
    중심이 반경의 0.39 / 0.29 배만큼 밀려 있다. `out/breakdance dyn_1`(off-axis 22.39°, τ 0.25):

        preset        source(d_cv, |az|max)     subject(d_cv, |az|max)
        orbit_left      0.0323   16.07°           0.0000   18.29°
        orbit_right     0.0386   18.00°           0.0000   18.29°
        crane_up        0.0656   10.03°           0.0528    0.00°
        crane_down      0.0550   23.72°           0.0526    0.00°
        s_curve         0.0170    8.23°           0.0002    9.14°

    (`d_cv` = |p(f) − c_obb(f)| 의 변동계수, `|az|max` = 중력축 둘레 방위각 최대 이동)

    source 에서는 ① orbit 이 등거리가 아니고 (16° 도는 동안 subject 와의 거리가 11.6% 흔들린다)
    ② crane 이 수직이 아니라 방위각으로 10~24° 끌려가며 up/down 이 비대칭이다. subject 로 세우면
    orbit 은 d_cv 정확히 0, crane 은 방위각 정확히 0 = "OBB 위치 기준 패턴".

    `aim="free"` 를 안 건드리는 이유: free 는 조준을 안 하므로 basis 회전이 곧 **frame0 회전**
    이다. subject 로 세우면 frame0 이 소스 카메라와 달라져 뱅크의 frame0 공유 규약(D83)과
    `build_poses` 의 frame0 assert 가 같이 깨진다.
    """
    if traj_basis != "auto":
        return traj_basis
    return "subject" if aim == "look_at" else "source"


def resolve_preset(name: str, aim: str | None = None):
    """옛 이름/별칭 -> 정식 preset 이름. 모르는 이름은 **그대로** 돌려준다(호출부가 판정).

    `aim` 을 주면 D76 에서 의미가 뒤집힌 `dolly_in`/`dolly_out` 을 옛 뜻으로 되돌린다
    (`LEGACY_AIM_COLLISIONS`). 뱅크 행을 읽는 쪽은 **항상 `variant["aim"]` 을 같이 넘길 것** —
    안 넘기면 옛 행이 조용히 새 뜻(재조준)으로 읽혀 캡션이 궤적과 어긋난다.
    """
    if aim is not None and (name, aim) in LEGACY_AIM_COLLISIONS:
        return LEGACY_AIM_COLLISIONS[(name, aim)]
    return PRESET_ALIASES.get(name, name)


def row_preset(row: dict, raw: bool = False):
    """뱅크 행 **하나가 실제로 만들어 낸** 카메라의 정식 preset 이름.

    WHY: 디스크의 뱅크는 그때그때의 어휘로 `preset` 을 적어 뒀다. `bank.json` 을 읽어서 이름을
    화면에 찍거나 코퍼스로 내보내는 쪽이 `row["preset"]` 을 그대로 쓰면 **행이 들고 있는 `aim`
    을 무시**하게 되고, D76 에서 뜻이 뒤집힌 `dolly_in`/`dolly_out` 과 D75 별칭들이 궤적과 다른
    이름으로 나간다. 실측(snowboard `hole_bank_k6`): 228행 중 **108행(47%)** 이 어긋난다 —
    `straight_ease`→`dolly_in`, `orbit_left_arc`→`orbit_left`, `rise_reveal`→`crane_up`,
    `dolly_in`(aim=traj)→`dolly_in_look_at` 등. 캡션은 이미 `build_bank_captions.py:176` 이
    정식 이름으로 굽고 있어서, 안 고치면 **같은 행의 캡션과 라벨이 서로 다른 카메라를 가리킨다.**

    `raw=True` 면 적혀 있는 문자열을 그대로 돌려준다 (옛 산출물 재현용 — 각 CLI 의
    `--raw_preset_names`).
    """
    name = row.get("preset", "")
    return name if raw else resolve_preset(name, row.get("aim"))


def shape_context(radius_world: float, obs_az_span_deg: float, shape: dict | None = None,
                  orbit_span_frac: float = 0.8):
    """preset builder 가 먹는 ctx. **orbit sweep 은 관측된 방위각 폭으로 clamp** 한다.

    `obs_az_span` 밖으로 도는 건 "소스가 한 번도 못 본 면"을 보여달라는 뜻이라 점군에 자료가 없다.
    τ 이분법이 결국 줄이긴 하지만, 모양 단계에서 미리 잘라야 이분법이 회전이 아니라 **이동**을
    줄이는 쪽으로 수렴한다 (SE(3) 로그는 둘을 같은 비율로 깎는다).
    """
    shape = {**DEFAULT_SHAPE, **(shape or {})}
    return {"radius": float(radius_world),
            "dolly": float(shape["dolly_frac"] * radius_world),
            "lateral": float(shape["lateral_frac"] * radius_world),
            "sweep": float(min(shape["sweep_deg"], orbit_span_frac * max(obs_az_span_deg, 1e-6))),
            "pan": float(shape["pan_deg"])}


# ── 외부 궤적 preset (D79). 실제 영화에서 뽑은 모양을 preset 어휘에 **런타임에 끼워 넣는다**.
# 등록된 이름만 `PRESETS` 에 추가되므로, 등록을 안 하면 기존 동작이 비트 단위로 그대로다.
EXTERNAL_PRESETS: dict[str, dict] = {}          # name -> {"source", "move", "angular", ...} 출처 기록


def _project_so3(rotations: np.ndarray):
    """(f,3,3) 을 가장 가까운 회전행렬로 투영한다 (SVD, `R = U diag(1,1,det(UVt)) Vt`).

    WHY: 외부 궤적의 회전은 **우리가 만든 게 아니라 재구성기가 낸 추정치**다 — DataDoP 는
    MonST3R pose 를 float32 로 저장하고 그걸 JSON 으로 한 번 더 돌린 값이라 `det(R)` 가
    1 에서 median **4.8e-7** 만큼 떠 있다 (188 shape 전량). `build_poses` 의 마지막 assert 는
    `< 1e-9` 이라 500배 초과다. 그런데도 대부분 통과한 이유는 `fit_tau` 의 `se3.scale_traj` 가
    로그/지수를 거치며 **우연히** 재직교화해 주기 때문이고, 스케일이 1 이라 그 경로가 생략되는
    변이에서만 터진다 — 즉 조용히 있다가 코퍼스 중간에 영상을 통째로 날린다 (D84 실측:
    `AssertionError: frame 1: det(R) != 1`, 영상 3편 손실).

    보간이 못 고치는 것도 여기서 짚어 둔다: DataDoP shape 은 전부 m=49 라 `_se3_interp` 가
    `num_frames=49` 에서 `rel.copy()` 로 **그대로 통과**시킨다. 재직교화는 등록 시점에 한 번
    하는 게 맞다.
    """
    u, _s, vt = np.linalg.svd(np.asarray(rotations, dtype=np.float64))
    signs = np.sign(np.linalg.det(u @ vt))
    u = u.copy()
    u[:, :, 2] *= signs[:, None]                  # det=-1 (반사) 가 나오지 않도록 마지막 열만 뒤집는다
    return u @ vt


def _se3_interp(rel: np.ndarray, num_frames: int):
    """(m,4,4) rel c2w 를 (n,4,4) 로. **SE(3) 로그 공간 보간** — 행렬 lerp 는 SE(3) 를 벗어난다.

    `_resample` 의 index pick 을 못 쓰는 이유: 외부 궤적은 우리가 조밀하게 만든 게 아니라
    m 이 고정(DataDoP 120)이라 n>m 이면 계단이 생기고, 그 계단이 `fit_tau` 의 τ_max 를
    프레임 사이에서 과소평가하게 만든다. 구간 안에서는 screw 보간이 유일하게 자연스럽다.
    """
    rel = np.asarray(rel, dtype=np.float64)
    m = len(rel)
    if m == num_frames:
        return rel.copy()
    u = np.linspace(0.0, m - 1, num_frames)
    lo = np.clip(np.floor(u).astype(int), 0, m - 2)
    frac = u - lo
    out = np.empty((num_frames, 4, 4), dtype=np.float64)
    for i, (a, f) in enumerate(zip(lo, frac)):
        if f <= 1e-12:
            out[i] = rel[a]
            continue
        step = np.linalg.inv(rel[a]) @ rel[a + 1]
        out[i] = rel[a] @ se3.scale_pose(step, float(f))
    return out


def register_external_presets(shapes, aim: str = "traj", prefix: str = ""):
    """외부 궤적을 `PRESETS` 에 등록한다. `shapes` = [{"name", "rel"(m,4,4), ...}, ...].

    `rel[0] = I` 이고 `max_f |t(f)| = 1` 로 정규화된 **모양만** 받는다 — 크기는 우리 τ 사다리와
    hole 사다리가 다시 푼다. 그래서 출처 코퍼스의 게이지(DataDoP = MonST3R, D=1)와 우리
    게이지(DA3 frame0 `S`)가 달라도 상관없다 ([[datadop-world-scale-is-monst3r-gauge]]).

    `aim="traj"` 가 기본인 이유: 실제 촬영 궤적은 **회전이 궤적의 일부**다. `look_at` 으로
    바꾸면 우리 subject 를 향해 매 프레임 재조준하면서 원본의 회전이 통째로 지워진다 —
    그러면 "실제 카메라를 가져왔다"는 말이 궤적에 남지 않는다.
    """
    registered = []
    for s in shapes:
        name = f"{prefix}{s['name']}"
        rel = np.asarray(s["rel"], dtype=np.float64)
        assert np.abs(rel[0] - np.eye(4)).max() < 1e-9, f"{name}: rel[0] != I"
        rel[:, :3, :3] = _project_so3(rel[:, :3, :3])    # 추정 pose 의 비-직교 잔차 제거 (D85)
        # 이동이 0 인 shape 은 τ 로 크기를 못 정한다 — `pan_left/right` 와 같은 처지다
        # (`sample_camera_bank.ROTATION_ONLY_PRESETS` 주석). 그래서 회전량을 `ctx["pan"]` 에
        # 맞춰 다시 스케일해, 사다리가 각도로 들어오면 그 각도가 그대로 나오게 한다.
        rot0 = float(s.get("rot_deg_shape") or 0.0)
        rot_only = bool(s.get("rotation_only"))

        def _builder(n, c, _r=rel, _rot0=rot0, _ro=rot_only):
            out = _se3_interp(_r, n)
            if _ro and _rot0 > 1e-9:
                out = se3.scale_traj(out, float(c["pan"]) / _rot0)
            return out

        PRESETS[name] = (_builder, aim, False)
        EXTERNAL_PRESETS[name] = {k: v for k, v in s.items() if k != "rel"}
        registered.append(name)
    return registered


def load_external_shapes(json_path: str, aim: str = "traj", prefix: str = ""):
    """`retrieve_datadop_shapes.py` 가 낸 `datadop_shapes_v1` 파일을 읽어 등록한다.

    -> (등록된 이름, 회전 전용 이름). 호출부는 두 번째를 `ROTATION_ONLY_PRESETS` 에 더한다.
    """
    import json
    with open(json_path) as f:
        blob = json.load(f)
    assert blob.get("format") == "datadop_shapes_v1", f"모르는 포맷: {blob.get('format')}"
    names = register_external_presets(blob["shapes"], aim=aim, prefix=prefix)
    rot_only = [f"{prefix}{s['name']}" for s in blob["shapes"] if s.get("rotation_only")]
    return names, rot_only


def build_shape(name: str, ctx: dict, num_frames: int = NUM_FRAMES, speed: str = "steady"):
    """preset 이름 → (n,4,4) 카메라 로컬 rel c2w. 크기는 아직 안 맞춰져 있다."""
    name = resolve_preset(name)
    assert name in PRESETS, f"모르는 preset: {name} (있는 것: {sorted(PRESETS)})"
    builder, aim, needs_zoom = PRESETS[name]
    rel = _resample(_dense(lambda n: builder(n, ctx), num_frames), num_frames, speed)
    assert np.abs(rel[0] - np.eye(4)).max() < 1e-12, f"{name}: rel[0] != I"
    return np.asarray(rel, dtype=np.float64), aim, needs_zoom


def shape_resizer(name: str, ctx: dict, num_frames: int = NUM_FRAMES, speed: str = "steady"):
    """**sweep 을 고정한 채 반경만** s 배 해서 궤적을 다시 만드는 함수를 돌려준다. 안 쓰면 None.

    WHY (F9). `fit_tau` 의 기본 손잡이는 `se3.scale_traj` 인데 그건 SE(3) **로그 전체**를 s 배
    한다 — 이동과 회전을 같은 비율로 깎는다. `true_orbit` 은 정의상 `R = |t| / θ` 라 로그를
    깎으면 **반경 R 은 그대로고 sweep θ 만 줄어든다.** 즉 τ 예산이 빡빡한 씬에서 `orbit_left`
    를 시키면 반경 1.4·d_ref 짜리 큰 원 위에서 3° 만 도는 궤적이 나오고, 캡션은 여전히
    "orbit"이라고 적힌다. arc 계열(`push_in_arc_*`)도 같은 병이다.
    여기서는 반대로 **θ 를 상수로 두고 R(과 dolly/lateral)만** s 배 한다:
        τ ≈ 2·s·R·sin(θ/2) / z_med      — s 에 **선형**이라 이분법의 단조성이 그대로 유지된다.
    그래서 "sweep 45° 짜리 작은 궤도"가 나온다 — 이름과 궤적이 일치한다.

    sweep 을 안 쓰는 preset(`dolly_in`, `truck_left`, ...)은 로그 스케일과 결과가 같으므로
    None 을 돌려 옛 경로를 그대로 태운다. 의존 여부는 **추측하지 않고 찔러 본다** — 그래야
    `register_external_presets` 로 런타임에 들어온 DataDoP shape 까지 자동으로 걸러진다.
    """
    name = resolve_preset(name)
    assert name in PRESETS, f"모르는 preset: {name} (있는 것: {sorted(PRESETS)})"
    builder, _aim, _needs_zoom = PRESETS[name]
    probe_a = np.asarray(builder(num_frames, ctx), dtype=np.float64)
    probe_b = np.asarray(builder(num_frames, {**ctx, "sweep": ctx["sweep"] * 0.5 + 1e-3}),
                         dtype=np.float64)
    if np.allclose(probe_a, probe_b, atol=1e-12):
        return None

    def resize(s: float):
        if s <= 0.0:                       # 옛 `scale_traj(rel, 0)` 과 같은 "얼어붙음"
            return np.stack([np.eye(4)] * num_frames)
        scaled = {**ctx, "radius": ctx["radius"] * s,
                  "dolly": ctx["dolly"] * s, "lateral": ctx["lateral"] * s}
        rel = _resample(_dense(lambda n: builder(n, scaled), num_frames), num_frames, speed)
        return np.asarray(rel, dtype=np.float64)

    return resize


def focal_track(name: str, num_frames: int = NUM_FRAMES, speed: str = "steady"):
    """preset 이름 → 프레임별 focal 배율 (n,), `[0] == 1.0`. zoom 없는 preset 은 전부 1.0.

    **로그 선형**이다 (`exp(linspace(0, log(end)))`). focal 을 선형으로 흔들면 화각 변화가
    앞뒤로 안 고른다 — hfov 는 `atan(W/2f)` 라 f 의 역수 쪽에 붙어 있다. 배율을 로그에서 고르게
    가면 "매 프레임 같은 비율로 넓어지는" 느낌이 되고, 그게 실제 zoom 링의 눈금이다.
    `speed` 재파라미터화는 궤적과 **같은 인덱스 규칙**(`_resample`)을 써야 둘이 안 어긋난다.
    """
    name = resolve_preset(name)
    end = float(PRESET_ZOOM_END.get(name, 1.0))
    if end == 1.0:
        return np.ones(num_frames, dtype=np.float64)
    dense = np.exp(np.linspace(0.0, np.log(end), (num_frames - 1) * 8 + 1))
    track = _resample(dense, num_frames, speed)
    assert abs(track[0] - 1.0) < 1e-12, f"{name}: focal[0] != 1"
    return np.asarray(track, dtype=np.float64)


def tau_of(rel_local: np.ndarray, c2w_start: np.ndarray, src_centers: np.ndarray, z_med: float):
    """카메라 로컬 궤적을 시작 pose 에 얹었을 때의 프레임별 τ. (tau_max, tau_per_frame)."""
    world = T.start_at(np.asarray(c2w_start, dtype=np.float64), rel_local)
    src = np.asarray(src_centers, dtype=np.float64)
    n = len(world)
    src = src[np.rint(np.linspace(0, len(src) - 1, n)).astype(int)] if len(src) != n else src
    tau = np.linalg.norm(world[:, :3, 3] - src, axis=-1) / max(z_med, 1e-9)
    return float(tau.max()), tau


def fit_tau(rel_local: np.ndarray, c2w_start: np.ndarray, src_centers: np.ndarray, z_med: float,
            target_tau: float, iterations: int = 8, max_scale: float = 4.0,
            refine_zero: bool = False, resize=None):
    """이분법으로 τ_max 를 target 에 맞춘다. (scaled_rel, info).

    `refine_zero` (D53): 이분법의 최소 눈금은 `max_scale / 2**iterations` (기본 4/256 = 0.0156)
    이고 그보다 작은 답은 `lo` 가 **0 에 남는다** — 궤적이 통째로 얼어붙는다. `target_tau` 가
    `tau_start` 바로 위일 때 늘 이 구간에 떨어진다 (avocado `stat_1 pull_out_arc`: tau_start
    0.1286, target 0.1486 → 필요한 배율이 첫 눈금보다 작아 `path_len_u 0.000`). 켜면 `[0, hi]`
    에서 이분법을 한 번 더 돌려 눈금을 `2**iterations` 배 잘게 만든다. `tau_of` 는 렌더가 아니라
    numpy 라 비용이 사실상 0 이다. 기본은 꺼서 예전 뱅크와 비트 동일하다.

    `resize` (F9): 배율을 적용하는 방법 자체를 갈아끼운다. None 이면 예전 그대로
    `se3.scale_traj`(SE(3) 로그 전체 스케일 = 이동·회전 동시 축소), 함수면 그 함수가 배율에서
    궤적을 만든다 (`shape_resizer` = sweep 고정, 반경만 스케일). 이분법은 τ 가 배율에 대해
    단조이기만 하면 되고 두 경로 모두 그렇다 — §`shape_resizer` docstring.
    """
    def apply(s: float):
        return resize(s) if resize is not None else se3.scale_traj(rel_local, s)

    tau0, _ = tau_of(np.stack([np.eye(4)] * len(rel_local)), c2w_start, src_centers, z_med)
    if tau0 >= target_tau - 1e-6:
        # 시작 pose 만으로 이미 예산을 다 쓴 경우. 움직임을 0 으로 둔다 (예산을 넘기지 않는다).
        return (np.stack([np.eye(4)] * len(rel_local)),
                {"scale": 0.0, "tau_max": round(tau0, 4), "tau_start": round(tau0, 4),
                 "tau_saturated": True, "target_tau": target_tau, "tau_refined": False,
                 "tau_resize": bool(resize)})

    lo, hi = 0.0, max_scale
    tau_hi, _ = tau_of(apply(hi), c2w_start, src_centers, z_med)
    if tau_hi <= target_tau:                       # max_scale 로도 예산을 못 채운다 (정지 preset 등)
        lo = hi
    else:
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            tau_mid, _ = tau_of(apply(mid), c2w_start, src_centers, z_med)
            lo, hi = (mid, hi) if tau_mid <= target_tau else (lo, mid)
    refined = False
    if refine_zero and lo == 0.0 and hi > 0.0:
        # 답이 첫 눈금보다 작았다. 같은 이분법을 `[0, hi]` 에서 다시 — 구간이 이미 `2**iterations`
        # 배 좁으므로 여기서도 0 이 나오면 그건 정말 "움직이면 예산 초과"인 경우다.
        refined = True
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            tau_mid, _ = tau_of(apply(mid), c2w_start, src_centers, z_med)
            lo, hi = (mid, hi) if tau_mid <= target_tau else (lo, mid)
    scaled = apply(lo)
    tau_max, _ = tau_of(scaled, c2w_start, src_centers, z_med)
    return scaled, {"scale": round(float(lo), 5), "tau_max": round(tau_max, 4),
                    "tau_start": round(tau0, 4), "tau_saturated": False, "target_tau": target_tau,
                    "tau_refined": refined, "tau_resize": bool(resize)}


def describe(name: str, rel_local: np.ndarray):
    """preset 한 줄 요약 (영문 — board 캡션과 프롬프트로 나간다)."""
    d = T.describe(rel_local)
    return (f"{name} | move {d['move_path']:.3f} | rotation {d['rot_deg']:.1f} deg | "
            f"path/chord {d['path_over_chord'] if d['path_over_chord'] is not None else float('nan'):.2f}")


def main(args):
    ctx = shape_context(args.radius, args.obs_az_span)
    print(f"ctx radius={ctx['radius']:.3f} dolly={ctx['dolly']:.3f} lateral={ctx['lateral']:.3f} "
          f"sweep={ctx['sweep']:.1f} pan={ctx['pan']:.1f}\n")
    header = f"{'preset':<24}{'aim':<9}{'zoom':<6}{'move':>8}{'rot_deg':>9}{'p/c':>7}"
    print(header)
    print("-" * len(header))
    for name in PRESETS:
        rel, aim, needs_zoom = build_shape(name, ctx, args.num_frames, args.speed)
        d = T.describe(rel)
        pc = d["path_over_chord"]
        print(f"{name:<24}{aim:<9}{'yes' if needs_zoom else '-':<6}{d['move_path']:>8.3f}"
              f"{d['rot_deg']:>9.2f}{(pc if pc is not None else float('nan')):>7.2f}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--list", action="store_true", default=True)      # 표만 찍는다
    parser.add_argument("--radius", default=0.6, type=float)              # subject 까지 거리 (world 단위)
    parser.add_argument("--obs_az_span", default=45.0, type=float)        # 관측된 방위각 폭 (deg)
    parser.add_argument("--num_frames", default=NUM_FRAMES, type=int)
    parser.add_argument("--speed", default="steady", type=str)            # steady|accel|decel|ease
    main(parser.parse_args())
