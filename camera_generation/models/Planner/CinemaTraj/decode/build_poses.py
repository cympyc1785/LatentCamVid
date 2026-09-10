"""`decision.json` + `scene_graph.json` → world c2w (49,4,4). **결정론적이다 — VLM 이 없다.**

왜 디코더를 따로 두나: VLM 이 내놓는 건 라벨(`A3`) 과 어휘(`orbit_left_arc`) 뿐이고, 실제 행렬은
전부 여기서 만들어진다. 그래야 ① 같은 decision 이 항상 같은 궤적을 주고 ② fallback 결정만으로도
파이프라인 끝까지 뚫리고 ③ Phase 2 에서 keyframe 이 여러 개가 되어도 이 파일만 바뀐다.

조립 순서:
    시작 pose  board.json 의 후보 `c2w_world` (없으면 keyframe 의 `p_G`/`look_at_G` 로 재구성)
    모양      `lbm.presets.build_shape` → 카메라 로컬 rel c2w (49,4,4)
    크기      `lbm.presets.fit_tau` → τ_max 를 `target_tau` 에 맞춤 (se3 로그 이분법 8회)
    world     `traj.start_at(c2w_start, rel)` → 위치 확정
    조준      `aim="look_at"` 이면 매 프레임 subject 를 향해 회전을 **다시 세운다** (roll = 0 wrt g)
              `aim="traj"` 면 조준 위에 궤적 회전을 얹는다 (`_look_at` preset — §lbm/presets.py)
              `aim="free"` 면 조준을 **아예 안 한다** — 소스 frame0 회전 위에 궤적 회전만 (D90)
    앵커      `aim_anchor="source_frame0"` 이면 위 조준이 frame 0 을 덮어쓰지 못하게 되돌린다

`aim_anchor` (DECISIONS.md D26/D27):
    `subject`(**기본**) — frame 0 부터 subject 를 향한다. `start_mode=source_frame0` 이어도
        시작 pose 에서 살아남는 건 **위치뿐**이고 회전은 조준이 덮어쓴다. 즉 첫 프레임 그림이
        소스와 다르다 (실측 회전 camel 3.18° / avocado-slice 11.67°, 후자는 hole 0.0127 → 0.3648).
        사용자가 두 쪽을 다 렌더해서 보고 이쪽을 골랐다 (D27).
    `source_frame0` / `auto` — frame 0 을 소스 카메라와 **완전히** 같게 만든다. 조준 오차를
        frame 0 에서 100% 되돌리고 `aim_ramp_frames` 프레임에 걸쳐 smoothstep 으로 0 까지 푼다
        (한 프레임에 11° 를 끊어 붙이면 그게 더 눈에 띈다). `auto` 는 `start_mode` 를 따라간다.

`traj_basis` (D69) — **preset 모양을 어느 회전 위에 얹을 것인가**:
    `source`(기존) — 소스 frame0 회전 위에 얹는다. `aim="traj"` preset(pan/truck/pedestal/
        static_hold_dont_look)은 조준을 다시 안 세우므로, 소스 광축이 anchor 를 안 향하고 있으면
        **끝까지 anchor 를 안 본다**. camel 실측: anchor 를 frame0 에 투영하면 dyn_0 은 화면
        중앙(off-axis 0.47°)이지만 stat_2 는 (u/W, v/H) = (0.156, -0.004), off-axis 12.86° 로
        half-hfov 14.4° 의 테두리다. 그래서 `stat_2__pan_right` 는 subject_in_frame 0.000.
        aim=traj 120 변이 평균 0.615, 그중 20 개는 49프레임 내내 subject 가 0 픽셀이다.
    `subject` — preset 모양을 **anchor 를 향해 세운 frame0 회전** 위에 얹는다. LBM 이 하는 일과
        같다: LBM Cinematographer 는 후보를 렌더해 구도가 맞는 pose 를 고른 뒤 그 위에서
        VideoEngineer preset 을 돌린다 (preset 모양을 고치는 게 아니라 **기준 회전**을 고친다).
        우리는 frame0 을 소스에 묶어야 하므로 기준만 subject 로 세우고, 그 차이는 아래 `aim_anchor`
        의 smoothstep 이 그대로 되돌린다 — frame0 은 여전히 소스 카메라와 완전히 같다.
        `truck`/`pedestal` 의 이동 방향도 카메라 로컬이라 같이 돌아간다(의도한 것 — "subject 기준
        왼쪽으로 truck"). τ 는 world 이동량으로 재므로 `fit_tau` 도 이 기준 위에서 다시 푼다.

`aim_keyframes` (D71) — **조준을 매 프레임이 아니라 sparse keyframe 에서만 세우고 사이를 잇는다**:
    0(기본) = 위 동작 그대로. N>=2 면 프레임을 `linspace(0, F-1, N)` 로 잘라 그 지점에서만 회전을
    정하고, 사이는 SO(3) 측지선(=쿼터니언 slerp) 으로 보간한다. **위치는 preset 모양 그대로**라
    τ 도 그대로다 — 바뀌는 건 회전뿐이다.
        keyframe 0  = 소스 frame0 회전 (**첫 프레임이 소스와 완전히 같다**, `aim_anchor` 불필요)
        keyframe k>0 = 원하는 곳(=anchor + `look_at_bias`)을 향한 회전
    왜 이게 필요한가: `aim_anchor="source_frame0"` 은 첫 프레임을 맞추지만 그 뒤 회전은
    ① `aim="look_at"` 이면 **매 프레임** 재조준(경직된 추종)이고 ② `aim="traj"` 면 **영원히 재조준
    안 함**이라, "처음엔 소스를 보다가 이후엔 원하는 곳을 본다"가 표현이 안 됐다. keyframe 조준은
    둘 사이다 — 첫 구간에서 시선이 target 으로 넘어가고, 이후는 keyframe 간격만큼 느슨하게 따라간다.
    `keyframe_aim`:
        `target`     keyframe 회전 = look_at(anchor). preset 자체 회전은 버린다.
        `preset_rel` keyframe 회전 = look_at(anchor) @ (preset 의 frame0 대비 상대회전).
                     조준 위에 preset 회전을 얹는다 — 기준은 여전히 anchor 다.
        `auto`(기본) aim="look_at" preset 은 `target`, aim="traj" preset 은 `preset_rel`.

    **`aim="free"` 는 이 블록에 아예 안 들어온다 (D90).** 이유는 실측이다: `preset_rel` 은
    `look_at(anchor) @ rel_local[f]` 인데 순수 병진 preset(truck/dolly/pedestal)은 `rel_local`
    회전이 **정확히 항등**이라 `preset_rel` 이 `target` 으로 붕괴한다. 그래서 D76~D89 의
    `aim="traj"` 는 이 preset 들에서 **no-op** 이었다 — `tru_0ac97866_a08` 49프레임 실측으로
    `dolly_in` vs `dolly_in_dont_look` 이 |Δt| 0.0 / |ΔR| 0.022°, 즉 같은 카메라다.
    `truck_left` 는 frame0→48 총 회전 71.3° 인데 조준오차가 3.03° — 그 71° 가 전부 조준이라
    이름만 truck 이고 실제로는 orbit 이었다. `free` 는 조준 단계를 통째로 건너뛰어 회전을
    `basis @ rel_local[f]` (= traj_basis="source" 면 소스 frame0 회전) 로 둔다.

subject tracking (`aim="look_at"` 전용, `aim_keyframes>0` 이면 **전 preset 에 적용**):
    world  c_ref(f) = c_j(0) 고정      τ 최소. track 이 못 미더우면 이걸로
    drift  c_j(0) + 0.6·(c(f) − c_j(0))  기본
    lock   c(f)                        subject 고정 shot
track center 는 이미 `scene_graph` 가 conf 가중 Savitzky-Golay 로 폈다 (`track.center_smooth`).
안 펴면 look-at 을 통해 곧장 회전으로 들어가서 r=0.5u 에서 1% jitter ≈ 프레임당 1° yaw 가 된다.

예시:
    python decode/build_poses.py --video camel --decision out/camel/decision.json
"""
import json
import sys
from os import path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.presets import (PRESET_ALIASES, PRESET_FOLLOW, PRESETS,  # noqa: E402
                         STATIC_PRESETS, build_shape, fit_tau,
                         focal_track, resolve_aim, resolve_tau_ref, resolve_tracking,
                         resolve_traj_basis, shape_context, shape_resizer, tau_of)
from lbm.presets import RECAMMASTER_ROOT  # noqa: E402
from lbm.render import look_at_c2w  # noqa: E402
from scene_graph.lift import apply_transform  # noqa: E402
from scene_graph.scale import tau_denominator  # noqa: E402

if RECAMMASTER_ROOT not in sys.path:
    sys.path.insert(0, RECAMMASTER_ROOT)
import traj as T  # noqa: E402

# D127. `drift`(0.6) 를 **어휘에서 삭제**했다 (사용자 지시). 남은 건 두 갈래뿐이다:
#   `world` 0.0  frame0 subject 중심을 계속 본다 (재조준 없음)
#   `lock`  1.0  매 프레임 subject 를 다시 조준한다
# drift 는 "조준점이 subject 변위의 60% 만 따라간다" 인데, 그건 필터가 아니라 **영구 편향**이라
# 끝까지 40% 뒤처진 채로 남는다. 원래 이걸 넣은 이유였던 조준 jitter 는 지금
# `track.center_smooth`(savgol w=11 p=3) + `--aim_keyframes 6` + `--keyframe_ease smooth_kf` 가
# **올바른 축에서** 처리한다. parkour 7-preset 실측(`results/20260904_tracking_drift_vs_lock/`):
# 동적 anchor `dyn_0` 28행 `subject_in_frame` 0.9670 -> 1.0000, 정지 anchor 144행은 Δ 0.0000.
# 옛 뱅크 행은 전부 `tracking:"drift"` 를 명시적으로 들고 있으므로 `presets.resolve_tracking` 이
# 읽는 시점에 `lock` 으로 승격시킨다 (assert 로 죽이지 않는다 — §presets.LEGACY_TRACKING).
TRACKING_GAIN = {"world": 0.0, "lock": 1.0}


def decision_fingerprint(decision: dict, start_mode: str = "source_frame0",
                         aim_anchor: str = "subject", aim_ramp_frames: int = 12,
                         traj_basis: str = "source", aim_keyframes: int = 0,
                         keyframe_aim: str = "auto", keyframe_ease: str = "smoothstep",
                         # **`"0"` 이어야 한다 (`0.0` 아님).** 아래 `str(...)` 를 그대로 지문에
                         # 넣으므로 float 기본값은 `"0.0"` 으로 찍히는데, `build_poses` 의
                         # argparse 기본값(:761)은 문자열 `"0"` 이다. decision 에 `follow_gain`
                         # 키가 없으면 (= 보통의 경우) 두 쪽이 각자 기본값을 stringify 해서
                         # `"0"` vs `"0.0"` 로 갈리고, `emit.py:169` 가 **손대지도 않은 poses 를
                         # stale 로 오판**한다 (camel 실측).
                         follow_gain: float | str = "0", follow_smooth: int = 9,
                         follow_keyframes: int = 0, follow_kf_interp: str = "cubic",
                         deroll: bool = False):
    """decision 에서 pose 를 실제로 바꾸는 필드만 뽑아 만든 지문.

    왜: `emit.py` 는 `poses.npz` 를 읽지 `decision.json` 을 다시 디코드하지 않는다. 루프가
    decision 을 새로 쓰고 `build_poses` 를 다시 안 돌리면 **낡은 poses 가 조용히 emit 된다**
    (실측: avocado-slice 가 static_hold 결정으로 orbit_left_arc poses 를 내보냈다).
    타임스탬프 비교로는 못 잡는다 — 같은 초에 쓰이면 순서를 모른다.
    """
    keyframe = decision["keyframes"][0]
    composition = keyframe["composition"]
    trajectory = decision["trajectory"]
    payload = {"subject_id": decision["subject_id"],
               "p_G": [round(float(v), 9) for v in composition["p_G"]],
               "look_at_G": [round(float(v), 9) for v in composition["look_at_G"]],
               "focal_scale": round(float(composition.get("focal_scale", 1.0)), 9),
               "preset": trajectory["preset"], "speed": trajectory.get("speed"),
               "tracking": trajectory.get("tracking"),
               "look_at_bias": trajectory.get("look_at_bias"),
               "target_tau": trajectory.get("target_tau"),
               "params": trajectory.get("params") or {},
               # decision 에는 없지만 pose 를 바꾸는 CLI 인자들. 빼면 이것만 바꿔 재빌드했을 때
               # 지문이 같아서 emit/verify 가드가 못 잡는다.
               "start_mode": start_mode,
               "aim_anchor": resolve_aim_anchor(aim_anchor, start_mode),
               # 앵커를 안 쓰면 ramp 는 pose 에 영향이 없다 — 지문에 넣으면 무의미한 불일치가 난다.
               "aim_ramp_frames": (int(aim_ramp_frames)
                                   if resolve_aim_anchor(aim_anchor, start_mode) == "source_frame0"
                                   else 0),
               "traj_basis": traj_basis,
               "aim_keyframes": int(aim_keyframes),
               # keyframe 조준을 안 쓰면 두 손잡이는 pose 에 영향이 없다 (ramp 와 같은 이유).
               "keyframe_aim": keyframe_aim if aim_keyframes else None,
               "keyframe_ease": keyframe_ease if aim_keyframes else None,
               # decision 이 있으면 그쪽이 이기므로 (build_poses 와 같은 규칙) 여기서도 같게 푼다.
               "follow_gain": str(trajectory.get("follow_gain", follow_gain))}
    # follow 를 안 쓰면 offset 이 통째로 0 이라 창 크기가 pose 를 못 바꾼다 — 그때 지문에 넣으면
    # **D73 이전에 만든 뱅크 전량이 거짓으로 stale 판정**된다 (ramp/keyframe 과 같은 규칙).
    if payload["follow_gain"] not in ("0", "0.0"):
        payload["follow_smooth"] = int(trajectory.get("follow_smooth", follow_smooth))
        # D171. 같은 두 층 규칙 — keyframe 보간을 **켠 뱅크에서만** 지문에 들어간다. 항상 넣으면
        # D171 이전 뱅크 전량이 거짓 stale 이 된다. `kf_interp` 는 keyframe 이 켜졌을 때만
        # pose 를 바꾸므로 그때만 같이 싣는다.
        keyframes = int(trajectory.get("follow_keyframes", follow_keyframes))
        if keyframes > 1:
            payload["follow_keyframes"] = keyframes
            payload["follow_kf_interp"] = str(trajectory.get("follow_kf_interp", follow_kf_interp))
    # D99. 껐을 때만 키를 빼는 이유는 follow_smooth 와 같다 — 넣어 버리면 D99 이전 뱅크 전량이
    # 거짓 stale 이 된다. 켠 뱅크는 자기들끼리만 비교하면 되므로 그때만 지문에 들어간다.
    if deroll:
        payload["deroll"] = True
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def resolve_aim_anchor(cli_value: str = "subject", start_mode: str = "source_frame0"):
    """`auto` → `start_mode` 를 따라간다. 기본값은 `subject` — **사용자가 앵커 없는 쪽을 골랐다**
    (D27). `auto`/`source_frame0` 는 첫 프레임을 소스와 완전히 같게 만드는 쪽이다."""
    assert cli_value in ("auto", "source_frame0", "subject"), f"모르는 aim_anchor: {cli_value}"
    if cli_value != "auto":
        return cli_value
    return "source_frame0" if start_mode == "source_frame0" else "subject"


def rotation_log(rotation: np.ndarray):
    """SO(3) → (단위 axis, angle rad). scipy 없이 (env 에 없다)."""
    rotation = np.asarray(rotation, dtype=float)
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:        # angle ≈ π — 반대칭부가 0 이라 대칭부에서 축을 뽑아야 한다
        values, vectors = np.linalg.eigh((rotation + rotation.T) / 2.0)
        axis = vectors[:, int(np.argmax(values))]
        return axis / max(float(np.linalg.norm(axis)), 1e-12), angle
    return axis / norm, angle


def rotation_exp(axis: np.ndarray, angle: float):
    """Rodrigues. `rotation_log` 의 역."""
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def slerp_rotation(rotation_a: np.ndarray, rotation_b: np.ndarray, alpha: float):
    """SO(3) 측지선 보간. 쿼터니언 slerp 와 같은 곡선이고 부호 뒤집힘 처리가 필요 없다.

    행렬을 성분별로 lerp 하면 SO(3) 를 벗어난다 (`_resample` 이 index pick 을 쓰는 것과 같은 이유).
    여기서는 keyframe 이 sparse 해서 index pick 이 안 통하므로 회전만 로그/지수로 잇는다.
    """
    axis, angle = rotation_log(np.asarray(rotation_b) @ np.asarray(rotation_a).T)
    return rotation_exp(axis, angle * float(alpha)) @ np.asarray(rotation_a)


def keyframe_indices(num_frames: int, count: int):
    """`linspace` 로 고른 keyframe 프레임 번호. 항상 0 과 F-1 을 포함하고 중복은 없앤다."""
    assert count >= 2, f"aim_keyframes 는 2 이상이어야 한다 (받은 값 {count})"
    return sorted(set(np.rint(np.linspace(0, num_frames - 1,
                                          min(count, num_frames))).astype(int).tolist()))


def arclen_rotation_schedule(rotations: list, num_frames: int):
    """keyframe 회전을 **누적 호길이** 위에 펴고 전 구간에 ease 를 한 번만 건다 (D86).

    `keyframe_ease="smoothstep"` 은 ease 를 구간마다 독립으로 걸어서 각속도가 keyframe 마다
    0 으로 떨어졌다가 구간 중앙에서 최대가 된다 — 이동은 preset 이 정한 등속인데 회전만
    keyframe 수만큼 펌핑한다. 실측(`dyn_0__orbit_right`, F=49 / N=6): 구간 안에서 5.3배,
    배포 코퍼스 전량에서 프레임간 회전각 max/median 이 median 3.16×(Vista4D) / 3.70×(TRUMANS),
    keyframe 각속도가 최대치의 **11%**. 게다가 구간별 조준량이 다르면 구간 **사이**의 속도까지
    계단이 된다 (`tru_00add26c_a01/dyn_0__dolly_in`: 마지막 구간만 4.8배).

    여기서는 전 keyframe 회전을 하나의 곡선으로 보고 `s(f) = smoothstep(f/(F-1)) · L_total`
    만큼 진행시킨다. 각속도가 0 인 곳은 첫/끝 두 프레임뿐이고 구간 경계에서 꺾이지 않는다.
    대가는 keyframe 회전이 **자기 프레임 번호에 정확히 도착하지 않는다**는 것 — 호길이가
    구간마다 다르므로 짧게 도는 구간은 빨리 지나간다. keyframe 은 조준의 표본일 뿐이고
    프레임에 묶여 있어야 하는 건 frame 0 (소스 회전) 하나라 문제되지 않는다. 조준 오차는
    `aim_err_*_deg` 로 그대로 찍히므로 값으로 확인할 수 있다.
    """
    angles = [float(rotation_log(rotations[i + 1] @ rotations[i].T)[1])
              for i in range(len(rotations) - 1)]
    cumulative = np.concatenate([[0.0], np.cumsum(angles)])
    total = float(cumulative[-1])
    schedule = []
    for frame in range(num_frames):
        if total < 1e-12:                     # 회전이 없다 — 전부 첫 회전 그대로
            schedule.append(np.asarray(rotations[0], dtype=float).copy())
            continue
        u = frame / max(num_frames - 1, 1)
        u = u * u * (3.0 - 2.0 * u)           # 전체에 한 번만 걸리는 smoothstep
        distance = u * total
        index = int(np.searchsorted(cumulative[1:-1], distance, side="right"))
        span = angles[index]
        alpha = 0.0 if span < 1e-12 else (distance - cumulative[index]) / span
        schedule.append(slerp_rotation(rotations[index], rotations[index + 1],
                                       float(min(max(alpha, 0.0), 1.0))))
    return schedule


def _pchip_slopes(x: np.ndarray, y: np.ndarray):
    """Fritsch-Carlson 단조 보존 기울기. 양 끝은 0 (전체 ease-in/out)."""
    secants = np.diff(y) / np.diff(x)
    slopes = np.zeros(len(x))
    for i in range(1, len(x) - 1):
        left, right = secants[i - 1], secants[i]
        if left * right > 0:                       # 부호가 같을 때만 통과 — 단조성을 깨지 않는다
            h_left, h_right = x[i] - x[i - 1], x[i + 1] - x[i]
            weight_l, weight_r = 2 * h_right + h_left, h_right + 2 * h_left
            slopes[i] = (weight_l + weight_r) / (weight_l / left + weight_r / right)
    return slopes


def arclen_kf_schedule(rotations: list, frames: list, num_frames: int):
    """`arclen` 의 keyframe 시각 보존판 (D86). 각속도는 이어지되 keyframe 은 제 프레임에 온다.

    `arclen` 은 누적 호길이에 **전역 smoothstep** 을 걸어 각속도를 평탄하게 만드는 대신
    keyframe 회전이 자기 프레임 번호를 떠난다 — 실측으로 `aim_err_med` 가 0.36° → 3.29°
    (p90 13.76°) 로 9배 늘었다. 최댓값은 26.0 → 26.4° 로 사실상 그대로라 최악의 구도는
    안 나빠지지만, 중간 프레임에서 subject 가 중심을 벗어나는 양이 커진다.

    여기서는 누적 호길이 `Θ` 를 **(keyframe 프레임, 누적각) 매듭 위의 단조 3차(PCHIP)** 로 잇는다.
    `Θ(frame_i) = cum_i` 라 keyframe 은 정확히 제자리에 오고(→ aim_err 는 smoothstep 과 같다),
    `Θ` 가 C1 이라 구간 경계에서 각속도가 0 으로 떨어지지도 꺾이지도 않는다. 양 끝 기울기만
    0 으로 두어 첫/끝 프레임에서만 부드럽게 서고 멈춘다.
    """
    angles = [float(rotation_log(rotations[i + 1] @ rotations[i].T)[1])
              for i in range(len(rotations) - 1)]
    cumulative = np.concatenate([[0.0], np.cumsum(angles)])
    knots = np.asarray(frames, dtype=float)
    if float(cumulative[-1]) < 1e-12 or len(knots) < 2:
        return [np.asarray(rotations[0], dtype=float).copy() for _ in range(num_frames)]
    slopes = _pchip_slopes(knots, cumulative)
    schedule = []
    for frame in range(num_frames):
        i = int(np.clip(np.searchsorted(knots, frame, side="right") - 1, 0, len(knots) - 2))
        h = knots[i + 1] - knots[i]
        t = (frame - knots[i]) / h
        # Hermite 기저
        distance = ((2 * t ** 3 - 3 * t ** 2 + 1) * cumulative[i]
                    + (t ** 3 - 2 * t ** 2 + t) * h * slopes[i]
                    + (-2 * t ** 3 + 3 * t ** 2) * cumulative[i + 1]
                    + (t ** 3 - t ** 2) * h * slopes[i + 1])
        index = int(np.searchsorted(cumulative[1:-1], distance, side="right"))
        span = angles[index]
        alpha = 0.0 if span < 1e-12 else (distance - cumulative[index]) / span
        schedule.append(slerp_rotation(rotations[index], rotations[index + 1],
                                       float(min(max(alpha, 0.0), 1.0))))
    return schedule


def smooth_kf_schedule(rotations: list, frames: list, num_frames: int,
                       passes: int = 12, lam: float = 0.5):
    """선형 slerp 折れ線을 깐 뒤 SO(3) Laplacian 평활로 **모서리를 깎는다** (D89).

    `arclen_kf` 와 무엇이 다른가: 저쪽은 **재타이밍**이라 궤적이 keyframe 을 잇는 측지선 折れ線
    위에 그대로 남는다 — 각속도 정체는 사라져도 keyframe 에서 회전축이 꺾이는 것(각가속도의
    불연속)은 그대로다. 여기서는 이웃 두 프레임의 접선 평균으로 밀어 折れ線 자체를 깎는다.

        R_f  <-  R_f · exp( (lam/2)·( log(R_fᵀ R_{f-1}) + log(R_fᵀ R_{f+1}) ) )

    대가는 **keyframe 회전을 정확히 통과하지 않는다**는 것이다. 그 양은 `aim_err_*_deg` 로
    그대로 찍히므로 값으로 판정할 것 — keyframe 조준(D71)이 이 뱅크의 존재 이유라 공짜가 아니다.
    양 끝은 고정한다 (frame0 은 어차피 소스 회전으로 덮어써지고, 마지막은 preset 끝점이다).
    `lam` 은 pass 당 이동 비율, `passes` 는 반복 횟수 — 둘의 곱이 대략 평활 폭을 정한다.
    """
    schedule = [None] * num_frames
    for index in range(len(frames) - 1):
        a, b = frames[index], frames[index + 1]
        for f in range(a, b + 1):
            schedule[f] = slerp_rotation(rotations[index], rotations[index + 1], (f - a) / (b - a))
    for f in range(num_frames):                      # keyframe 이 F-1 에 못 미치는 경우 꼬리 채움
        if schedule[f] is None:
            schedule[f] = np.asarray(rotations[-1], dtype=float).copy()

    current = np.stack([np.asarray(r, dtype=float) for r in schedule])
    if num_frames < 3 or passes <= 0 or lam <= 0.0:
        return [current[f] for f in range(num_frames)]
    for _ in range(int(passes)):
        updated = current.copy()
        for f in range(1, num_frames - 1):
            step = np.zeros(3)
            for neighbour in (f - 1, f + 1):
                axis, angle = rotation_log(current[f].T @ current[neighbour])
                step += axis * angle
            axis = step / max(float(np.linalg.norm(step)), 1e-12)
            updated[f] = current[f] @ rotation_exp(axis, float(np.linalg.norm(step)) * lam / 2.0)
        current = updated
    return [current[f] for f in range(num_frames)]


def aim_keyframe_rotations(poses: np.ndarray, rel_local: np.ndarray, targets: np.ndarray,
                           up_world: np.ndarray, frames: list, start_rotation: np.ndarray,
                           mode: str):
    """keyframe 별 회전 (D71). frames[0]==0 은 **소스 회전 그대로** — 첫 프레임을 묶는 게 요점이다.

    `preset_rel` 은 preset 의 frame0 대비 상대회전을 조준 위에 얹는다. world 회전이
    `R_basis @ R_rel` 이므로 기준만 `look_at` 으로 갈아끼우면 preset 의 의미(pan 은 시선을 뗀다)가
    보존된다. `target` 은 그 상대회전을 버리고 조준만 남긴다.
    """
    assert mode in ("target", "preset_rel"), f"모르는 keyframe_aim: {mode}"
    rotations = []
    for frame in frames:
        if frame == 0:
            rotations.append(np.asarray(start_rotation, dtype=float).copy())
            continue
        aimed = look_at_c2w(poses[frame][:3, 3], targets[frame], up_world)[:3, :3]
        rotations.append(aimed if mode == "target" else aimed @ rel_local[frame][:3, :3])
    return rotations


def subject_centers_world(node: dict, T_wg: np.ndarray, num_frames: int):
    """subject track center (G) → world. 프레임 수가 다르면 균등 인덱스로 맞춘다."""
    centers = np.asarray(node["track"]["center_smooth"], dtype=float)
    if len(centers) != num_frames:
        centers = centers[np.rint(np.linspace(0, len(centers) - 1, num_frames)).astype(int)]
    return apply_transform(np.asarray(T_wg, dtype=float), centers)


def reference_centers(centers_world: np.ndarray, tracking: str):
    """`tracking` 에 따른 c_ref(f). 'world' 는 frame0 고정이라 τ 가 제일 작다."""
    gain = TRACKING_GAIN[tracking]
    return centers_world[0][None] + gain * (centers_world - centers_world[0][None])


def follow_offset(centers_world: np.ndarray, gain: float):
    """subject 변위를 카메라 **위치**에 싣는 양 (D72). gain=0 이면 정확히 0 벡터 = 기존 동작.

    `tracking`(=`reference_centers`) 과 헷갈리면 안 된다. 저건 **조준점**만 움직이고 위치는
    그대로라, CameraBench 분류로는 pan-/tilt-tracking(회전만)이다. 이 함수가 만드는 건 병진이라
    시차가 생긴다 — tail-/lead-/side-/aerial-tracking 이 그것이고, 논문이 둘을 가르는 기준도
    "가까운 물체가 더 빨리 흐르는가"다. frame 0 은 정의상 0 이라 시작 pose 는 안 건드린다.
    """
    return float(gain) * (np.asarray(centers_world, dtype=float) - centers_world[0][None])


def smooth_follow_centers(centers_world: np.ndarray, window: int = 9, polyorder: int = 2):
    """follow **위치 채널** 전용 저역통과. `window <= 1` 이면 그대로 = 기존 동작.

    왜 위치만 따로 한 번 더 미는가: `track.center_smooth` 는 이미 Savitzky-Golay(w=11, p=3)
    를 거쳤지만 그건 **조준(회전)** 눈금에 맞춘 것이다. follow 는 그 궤적을 g 배로 **위치**에
    싣기 때문에 남은 고주파가 그대로 카메라 흔들림이 된다. 실측 snowboard dyn_0 은
    subject |jerk| p95 = 0.01728 u 였고, plan 카메라 |jerk| p95 = 0.01572 u = **정확히
    그것의 0.91배**(=g) 였다 — preset 모양은 정의상 매끈하므로 흔들림은 100% 이 채널에서 온다.
    z_med 1.751 · f 1185 로 환산하면 프레임당 10.6 px 라 눈에 보인다.

    `tracking`(조준점)은 **안 건드린다**. 조준은 subject 를 실제로 따라가야 맞고, 흔들림이
    문제되는 건 병진 쪽이다.

    window 9 인 이유 (snowboard dyn_0, g 는 매번 다시 풀어서 측정):

        win   g*    tau*    jerk p95   px/f^3   subject 이탈
          1  0.91  0.0932    0.01572     10.6    0.0000
          9  0.91  0.0909    0.00357      2.4    0.0119
         21  0.92  0.0868    0.00336      2.3    0.0416
         31  0.92  0.0968    0.00254      1.7    0.0490

    9 가 무릎이다 — 흔들림을 4.4배 깎으면서 τ 는 오히려 조금 좋아지고(0.0932 → 0.0909)
    subject 이탈은 0.012u 뿐이다. 21 이상은 흔들림이 더 안 줄면서 이탈만 3~4배 커진다.
    """
    window = int(window)
    if window <= 1:
        return np.asarray(centers_world, dtype=float)
    centers = np.asarray(centers_world, dtype=float)
    window = min(window, len(centers) - (1 - len(centers) % 2))  # savgol 은 홀수 & <= N
    if window <= polyorder + 1:
        return centers
    return savgol_filter(centers, window, polyorder, axis=0, mode="interp")


def keyframe_follow_centers(centers_world: np.ndarray, keyframes: int = 0,
                            interp: str = "cubic", smooth_window: int = 9,
                            polyorder: int = 2):
    """follow **위치 채널**을 균등 keyframe 몇 개로 줄였다가 다시 채운다 (D171).
    `keyframes <= 1` 이면 아무것도 안 하고 `smooth_follow_centers` 로 넘긴다 = 기존 동작.

    왜 저역통과(`smooth_follow_centers`) 말고 이것인가 (사용자 지시 2026-09-10, "track 일 경우
    너무 물체를 따라가서 흔들리는데 recon 이 깔끔한 sparse keyframe 들을 기준으로 translation 도
    interpolate 해보는건 어떰"): savgol 은 **모든 프레임의 오차를 평균**하는 필터라, recon 이
    한두 프레임에서 크게 튀면 그 오차가 창 전체로 번진다. keyframe 방식은 반대로 **표본을
    버린다** — 49프레임 중 9개만 믿고 나머지 40개는 아예 안 본다. 조준(`reference_centers`)은
    여기서도 안 건드린다 (§`smooth_follow_centers`).

    `interp` 세 갈래. 셋 다 **같은 9개 control point** 를 지나므로 갈리는 건 그 사이를 어떻게
    채우냐 하나다:

        linear  keyframe 사이 직선. keyframe 에서 속도가 꺾여 6번 각진다 (대조군).
        savgol  linear 로 채운 뒤 기존 `smooth_follow_centers` 를 그대로 한 번 더 — 꺾인 자리를
                창 `smooth_window` 로 둥글린다. 기존 코드 경로를 재사용하는 쪽.
        cubic   9점을 지나는 natural cubic spline. 정의상 C² 라 꺾임이 없고 필터 창이 없다.
                단 control point 가 튀면 그 사이 구간이 **원자료보다 크게** 오버슈트한다.

    `bc_type="natural"` 인 이유: 양 끝 2차도함수를 0 으로 두면 첫/끝 프레임에서 가속이 0 이라
    시작·종료가 부드럽다. `not-a-knot`(scipy 기본)은 끝 구간을 이웃 구간과 같은 3차식으로
    이어 붙여서 끝에서 튀기 쉽다.
    """
    keyframes = int(keyframes)
    centers = np.asarray(centers_world, dtype=float)
    if keyframes <= 1 or len(centers) <= 2:
        return smooth_follow_centers(centers, smooth_window, polyorder)
    assert interp in ("linear", "savgol", "cubic"), f"모르는 follow_kf_interp: {interp}"
    n = len(centers)
    # 균등 인덱스. 반올림이 겹칠 수 있어(짧은 클립) unique 로 접는다 — 접히면 실제 keyframe 수가
    # 요청보다 적어지지만 궤적은 여전히 그 점들을 지난다.
    idx = np.unique(np.rint(np.linspace(0, n - 1, min(keyframes, n))).astype(int))
    if len(idx) < 2:
        return smooth_follow_centers(centers, smooth_window, polyorder)
    frames = np.arange(n, dtype=float)
    if interp == "cubic":
        if len(idx) < 4:                     # natural spline 은 4점 미만이면 못 푼다
            return np.stack([np.interp(frames, idx, centers[idx, k]) for k in range(3)], axis=1)
        return CubicSpline(idx.astype(float), centers[idx], axis=0, bc_type="natural")(frames)
    filled = np.stack([np.interp(frames, idx, centers[idx, k]) for k in range(3)], axis=1)
    return filled if interp == "linear" else smooth_follow_centers(filled, smooth_window, polyorder)


def solve_follow_gain(centers_world: np.ndarray, src_positions: np.ndarray,
                      start_position: np.ndarray, hi: float = 2.5, steps: int = 251,
                      min_benefit: float = 0.20, moving: bool = True):
    """τ 를 최소로 만드는 gain (격자 탐색, 반환 (gain, 그때의 τ·z_med)).

    **이분법이 아니라 격자**인 이유: 목적함수가 max_f 라 단봉이 보장되지 않는다 (subject 가
    방향을 꺾으면 국소 최소가 둘 이상 생긴다).

    왜 τ 최소인가: follow 는 예산을 쓰는 게 아니라 **되찾는** 쪽이다. 소스 카메라가 이미
    subject 를 따라간 영상에서는 plan 카메라가 가만히 있는 것만으로 예산을 넘긴다 (실측
    snowboard 1.9441, snow-bike 1.3765 — 뱅크 360 변이 중 350 이 `tau_saturated` 로 죽었다).
    gain 을 올리면 τ 가 내려가고 남은 예산이 preset 모양으로 간다. 소스가 회전으로만 따라간
    영상(parkour/truck-pose/trumans-bedroom)은 g*≈0 이 나와 저절로 기존 동작으로 돌아간다.

    `moving` 이 **1차 가드**다. tracking shot 은 정의상 *움직이는* subject 를 따라가는 것이므로
    정적 노드에는 애초에 걸리면 안 된다. 그런데 정적 노드의 `track.center_smooth` 는 가만히 있지
    않다 — 카메라가 큰 평면을 훑으면 **보이는 부분**이 옮겨가서 OBB 중심이 벽을 따라 미끄러진다.
    실측 truck-pose `stat_0`(graffiti, 벽) 은 0.535u 나 미끄러졌고 τ 를 26.6% 깎아 `min_benefit`
    을 통과했다. fashion-walk `stat_9`(tree) 는 path_len 13.6u / drift 4.69u 다. 즉 크기로는
    동적과 안 갈리므로 `node["moving"]` 플래그로 자른다.

    `min_benefit` 은 **2차 가드**다 (움직이는 노드인데 움직임이 너무 작은 경우). 그 떨림에도
    argmin 은 걸리므로 가드가 없으면 격자 끝까지 밀려가 **분할 노이즈를 증폭한 흔들리는 카메라**
    를 만든다 — snowboard 정적 anchor 에서 τ 가 1.9441 → 1.7798 (8.5%) 밖에 안 줄면서 그랬다.
    τ 가 `min_benefit` 만큼도 안 줄면 0 을 돌려 기존 동작으로 되돌린다.

    `hi` 가 2.5 인 이유: 1.5 는 실측에서 **상한이 물렸다**. fashion-walk 는 dyn_0 g*=1.96 /
    dyn_1 g*=1.88 / dyn_2 g*=2.05 라 1.5 에서 잘려 τ 가 0.165/0.110/0.233 에 멈췄고
    (진짜 최소는 0.135/0.073/0.206), dyn_1 은 τ*=0.10 사다리 칸을 그것 때문에 놓쳤다.
    """
    if not moving:
        return 0.0, float(np.linalg.norm(np.asarray(src_positions, dtype=float)
                                         - np.asarray(start_position, dtype=float), axis=-1).max())
    grid = np.linspace(0.0, hi, steps)
    delta = np.asarray(centers_world, dtype=float) - centers_world[0][None]
    residual = np.asarray(src_positions, dtype=float) - np.asarray(start_position, dtype=float)
    taus = np.linalg.norm(grid[:, None, None] * delta[None] - residual[None],
                          axis=-1).max(axis=1)
    best = int(np.argmin(taus))
    if taus[best] > (1.0 - min_benefit) * taus[0]:
        return 0.0, float(taus[0])
    return float(grid[best]), float(taus[best])


def follow_shot_kind(camera_position: np.ndarray, centers_world: np.ndarray,
                     up_world: np.ndarray):
    """CameraBench tracking shot 갈래를 **측정**한다 (고르는 게 아니다).

    frame 0 이 소스 카메라에 묶여 있어서, 카메라가 subject 진행방향의 어느 쪽에 서는지는 소스가
    이미 정해 놨다. 라벨만 붙여 두면 뱅크에서 갈래별로 골라 쓸 수 있다.
    각도 = ∠(카메라→subject, subject 속도): 0° 면 subject 가 멀어지니 카메라가 뒤(tail),
    180° 면 다가오니 앞(lead), 중간이면 옆(side). 고도 45° 이상은 aerial 로 따로 뺀다.
    """
    velocity = np.asarray(centers_world[-1], dtype=float) - centers_world[0]
    to_subject = np.asarray(centers_world[0], dtype=float) - np.asarray(camera_position, float)
    if np.linalg.norm(velocity) < 1e-9 or np.linalg.norm(to_subject) < 1e-9:
        return {"kind": "none", "angle_deg": 0.0, "elevation_deg": 0.0}
    velocity = velocity / np.linalg.norm(velocity)
    to_subject = to_subject / np.linalg.norm(to_subject)
    angle = float(np.degrees(np.arccos(np.clip(float(np.dot(to_subject, velocity)), -1.0, 1.0))))
    elevation = float(np.degrees(np.arcsin(np.clip(
        float(np.dot(-to_subject, np.asarray(up_world, dtype=float))), -1.0, 1.0))))
    kind = ("aerial-tracking" if elevation >= 45.0 else
            "tail-tracking" if angle < 40.0 else
            "lead-tracking" if angle > 140.0 else "side-tracking")
    return {"kind": kind, "angle_deg": round(angle, 2), "elevation_deg": round(elevation, 2)}


# `roll_about` 이 **도(deg)** 를 돌려주므로 아래 assert 들의 허용치도 도 단위다. 예전 값 1e-6 deg
# 는 1.7e-8 rad 로 float64 잡음보다 작아서 실제로 parkour 에서 1.14e-6 deg 로 터졌다 (조준은
# 멀쩡했고 `arctan2(cross, dot)` 누적 오차였다). 1e-4 deg = 지평선이 4K 폭에서 0.004 px 기우는
# 양이라 눈에 보이는 roll(>=0.01 deg)은 여전히 전부 잡는다.
ROLL_TOL_DEG = 1e-4


def roll_about(cam_c2w: np.ndarray, up_world: np.ndarray):
    """중력축 `up` 기준 roll (deg). look_at 으로 세운 카메라는 정의상 0 이어야 한다."""
    right, forward = cam_c2w[:3, 0], cam_c2w[:3, 2]
    up = np.asarray(up_world, dtype=float)
    horizontal = up - np.dot(up, forward) * forward
    if np.linalg.norm(horizontal) < 1e-6:      # 시선이 중력축과 평행 — roll 이 정의되지 않는다
        return 0.0
    horizontal /= np.linalg.norm(horizontal)
    expected_right = np.cross(forward, horizontal)  # look_at_c2w 와 같은 규약 (right = fwd x up)
    expected_right /= max(np.linalg.norm(expected_right), 1e-12)
    cos = float(np.clip(np.dot(right, expected_right), -1.0, 1.0))
    sin = float(np.dot(np.cross(expected_right, right), forward))
    return float(np.degrees(np.arctan2(sin, cos)))


def _roll_about_forward(rotation: np.ndarray, degrees: float):
    """광축(카메라 z) 둘레로 `degrees` 만큼 돌린 회전. `roll_about` 의 정확한 역연산이다."""
    angle = np.radians(float(degrees))
    cos, sin = np.cos(angle), np.sin(angle)
    return rotation @ np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def deroll_poses(poses: np.ndarray, up_world: np.ndarray, rolls_target: np.ndarray,
                 pole_deg: float = 5.0):
    """프레임마다 **광축은 그대로 두고** 중력 기준 roll 만 `rolls_target` 으로 다시 세운다 (D99).

    왜 필요한가 — 두 군데서 roll 이 샌다.
      ① `aim="free"` (pan/tilt/dolly/truck…): preset 회전이 **카메라 로컬** 축에서 정의돼 있다
         (`traj.pan` 은 `rot_y`). 카메라가 기울어 있으면 로컬 y 둘레 pan 은 중력 둘레 pan 이
         아니라 원뿔을 그리고, 지평선이 같이 돈다. 실측(TRUMANS `tru_0ac97866_a09_s3f0k6`,
         pan 45°): roll 이 f24 에서 16.0° 까지 갔다가 돌아온다. 삼각대 위 pan 은 중력 둘레다.
      ② `aim_keyframes` 측지선 보간: keyframe 회전이 전부 roll=0 이어도 **그 사이**는 아니다.
         SO(3) 측지선은 yaw 와 pitch 가 동시에 바뀌면 중간에서 roll 을 만든다 (구면 위 평행이동의
         holonomy). 실측 같은 chunk: `crane_up` 21.5°, `dolly_in_look_at` 13.4°.

    광축을 안 건드리므로 **조준 오차(`aim_err_*`)와 τ 는 한 자리도 안 변한다** — 바뀌는 건
    지평선의 기울기뿐이다. 그래서 이 보정은 preset 모양이나 예산과 독립이다.

    `pole_deg` — 시선이 중력축과 이 각도 안으로 들어오면 roll 이 정의되지 않는다
    (`roll_about` 도 거기서 0 을 돌려준다). 그 프레임은 **손대지 않고** 개수만 세서 돌려준다.
    """
    up = np.asarray(up_world, dtype=float)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    pole_sin = np.sin(np.radians(float(pole_deg)))
    skipped = 0
    for f in range(len(poses)):
        forward = poses[f][:3, 2]
        horizontal = up - float(np.dot(up, forward)) * forward
        if float(np.linalg.norm(horizontal)) < pole_sin:
            skipped += 1
            continue
        horizontal /= np.linalg.norm(horizontal)
        right = np.cross(forward, horizontal)      # roll_about 의 `expected_right` 와 같은 규약
        right /= max(float(np.linalg.norm(right)), 1e-12)
        level = np.stack([right, np.cross(forward, right), forward], axis=1)
        poses[f][:3, :3] = _roll_about_forward(level, rolls_target[f])
    return skipped


def orthonormalize(c2w: np.ndarray):
    """회전 블록을 가장 가까운 SO(3) 로 투영한다 (polar decomposition).

    `cameras.cam_c2w_world` 는 DA3 가 낸 w2c 를 뒤집어 저장한 것이라 열이 정확히 단위벡터가
    아니다 — 실측 `|R[:,2]|` 이 camel 1 + 1.71e-5, avocado-slice 1 − 2.19e-4. 그대로 시작
    pose 로 쓰면 아래 `det(R)=1` assert 가 frame 0 에서 터진다. **궤적이 틀린 게 아니라 입력이
    정규직교가 아닌 것**이므로 억누르지 말고 여기서 한 번 투영한다 (회전각 변화 ~1e-3°).
    """
    result = np.asarray(c2w, dtype=float).copy()
    U, _, Vt = np.linalg.svd(result[:3, :3])
    if np.linalg.det(U @ Vt) < 0:      # 반사가 섞이면 카메라가 좌우로 뒤집힌다
        U[:, -1] *= -1
    result[:3, :3] = U @ Vt
    return result


def build_poses(decision: dict, graph: dict, board: dict | None = None,
                num_frames: int | None = None, target_tau: float | None = None,
                orbit_span_frac: float = 0.8, start_mode: str = "source_frame0",
                aim_anchor: str = "subject", aim_ramp_frames: int = 12,
                traj_basis: str = "source", aim_keyframes: int = 0,
                keyframe_aim: str = "auto", keyframe_ease: str = "smoothstep",
                follow_gain: float | str = 0.0, follow_smooth: int = 9,
                follow_keyframes: int = 0, follow_kf_interp: str = "cubic",
                use_fit_tau: bool = True, preset_tracking: bool = True,
                smooth_passes: int = 12, smooth_lambda: float = 0.5,
                tau_ref: str = "source", deroll: bool = False,
                orbit_fixed_sweep: bool = False, tau_denom: str = "z_med_frame0"):
    """(poses (n,4,4) world c2w, info dict). 이 함수가 `decision.json` 의 유일한 해석자다.

    `tau_denom` (D150) 은 τ 의 분모를 고른다 — `"S"` 면 전 프레임 non-sky 점의 첫 카메라 거리
    평균, `"z_med_frame0"`(기본) 이면 예전 frame0 z-depth 중앙값. 정의와 배율 실측은
    `scene_graph.scale.tau_denominator` 의 docstring 에 있다. 기본값을 예전 것으로 둔 이유는
    이 값이 **뱅크 정체성**이라 기본을 바꾸면 진행 중인 굽기가 코퍼스 중간에 정의를 갈아타기
    때문이다. 새 정의는 호출자가 명시적으로(`--tau_denom S`) 켠다.
    """
    num_frames = num_frames or int(graph["num_frames"])
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
    up_world = np.asarray(graph["gravity"]["up_world"], dtype=float)
    up_world = up_world / np.linalg.norm(up_world)
    scale = float(graph["scale"]["S"])
    z_med = tau_denominator(graph, tau_denom)
    src_c2w = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float)

    node = next(n for n in graph["nodes"] if n["id"] == decision["subject_id"])
    keyframe = decision["keyframes"][0]
    trajectory = decision["trajectory"]
    target_tau = float(trajectory.get("target_tau", target_tau if target_tau is not None else 0.20))

    # ── 시작 pose.
    #
    # `source_frame0`(기본) — 소스 카메라의 frame 0 을 그대로 시작점으로 쓴다. 궤적만 결정 대상이다.
    #   왜 이게 기본인가:
    #   ① emit 의 기본 규약(`--no_free_start`)이 `rel = inv(P[0]) @ P` 라 **시작 pose 와 소스
    #      카메라 사이의 상수 offset 을 어차피 버린다**. ReCamMaster 는 `--free_start` 로 살려도
    #      픽셀 수준에서 무시한다(실측). 즉 select 산출물은 하류에 거의 도달하지 못했다.
    #   ② 그러면서 τ 예산은 먹었다. avocado-slice 는 시작 pose 가 예산 0.20 중 0.1549 를 써서
    #      preset 16종이 전부 `tau_scale 0` 으로 죽었다 (DECISIONS.md D20).
    #   ③ 시작 pose = 소스 frame0 이면 `τ(0) = 0` 이라 예산이 전부 궤적으로 간다.
    #
    # `board` — 예전 동작. VLM 이 고른 후보(+micro-adjust)를 시작점으로 쓴다.
    #   board.json 의 후보 행렬이 1순위다 (게이트가 통과시킨 바로 그 pose). micro_ops 가 있으면
    #   pose 가 후보에서 옮겨간 뒤라 board 행렬을 쓰면 안 된다 — 그때는 keyframe 의
    #   p_G/look_at_G(= micro-adjust 최종 상태)가 유일한 진실이다.
    assert start_mode in ("source_frame0", "board"), f"모르는 start_mode: {start_mode}"
    c2w_start = orthonormalize(src_c2w[0]) if start_mode == "source_frame0" else None
    start_source = "source_frame0" if c2w_start is not None else "keyframe_p_G"
    if c2w_start is None and board is not None and keyframe.get("candidate_id") \
            and not keyframe.get("micro_ops"):
        match = next((c for c in board["candidates"]
                      if keyframe["candidate_id"] in (c.get("board_label"), c.get("cand_id"))), None)
        if match is not None:
            c2w_start, start_source = np.asarray(match["c2w_world"], dtype=float), "board"
    if c2w_start is None:
        composition = keyframe["composition"]
        to_world = lambda p: T_wg[:3, :3] @ np.asarray(p, dtype=float) + T_wg[:3, 3]
        c2w_start = look_at_c2w(to_world(composition["p_G"]), to_world(composition["look_at_G"]),
                                T_wg[:3, :3] @ np.array([0.0, 0.0, 1.0]))

    centers_world = subject_centers_world(node, T_wg, num_frames)
    radius_world = float(np.linalg.norm(c2w_start[:3, 3] - centers_world[0]))

    # ── 모양 → 크기
    preset = trajectory["preset"]
    speed = trajectory.get("speed", "steady")
    context = shape_context(radius_world, float(node["obs_az_span_deg"]),
                            trajectory.get("shape"), orbit_span_frac)
    rel_local, aim, needs_zoom = build_shape(preset, context, num_frames, speed)
    # D90. 뱅크 행이 옛 `aim` 을 들고 있으면 그때 나오던 카메라를 그대로 재현한다 (§presets.py
    # `resolve_aim`). `decision.json` 에 `aim` 이 없으면 None → preset 의 새 기본값을 탄다.
    aim = resolve_aim(preset, trajectory.get("aim"))
    focal = focal_track(preset, num_frames, speed)   # (n,) intrinsic zoom, [0]=1.0

    tracking_requested = trajectory.get("tracking", "lock")
    # D93 -> D127. `aim="look_at"` preset 은 조준 추종률을 `lock` 으로 (§presets.PRESET_TRACKING).
    # 옛 뱅크의 `drift` 도 여기서 `lock` 으로 승격된다 (§presets.LEGACY_TRACKING).
    tracking = resolve_tracking(preset, tracking_requested, preset_tracking)
    assert tracking in TRACKING_GAIN, f"모르는 tracking: {tracking}"
    tracking_ignored = aim != "look_at"
    height = float(node["obb"]["extent"][2])
    bias = float(trajectory.get("look_at_bias", 0.0))
    # look_at 오프셋 방향은 중력축 위쪽(`gravity.up_world`). 크기는 OBB 높이가 u 단위라 S 를 곱한다
    # (T_wg[:3,:3] 로 벡터를 돌리면 스케일이 두 번 들어간다 — 그건 위치 변환용이다).
    bias_world = bias * height * scale * up_world

    # ── preset 모양을 얹을 기준 회전 (D69). 위치는 어느 쪽이든 소스 frame0 그대로다.
    # D99. `auto` 는 조준 방식이 정한다 (§presets.resolve_traj_basis) — object-centric preset
    # (`aim="look_at"`) 만 subject 로 세워 선회 중심을 OBB 중심에 앉히고, `aim="free"` 는
    # frame0 규약 때문에 source 그대로. 명시값이면 D99 이전과 한 톨도 안 다르다.
    traj_basis_requested = traj_basis
    traj_basis = resolve_traj_basis(traj_basis, aim)
    assert traj_basis in ("source", "subject"), f"모르는 traj_basis: {traj_basis}"
    basis_c2w = c2w_start
    if traj_basis == "subject":
        basis_c2w = look_at_c2w(c2w_start[:3, 3], centers_world[0] + bias_world, up_world)

    # `tau_refine` 은 CLI 가 아니라 **결정 자체**에 실려 온다 (D53). 뱅크를 되풀 때 같은 궤적이
    # 나와야 하는데, 플래그가 밖에 있으면 `emit_bank` 가 재현을 놓친다. 없으면 예전 동작.
    # τ 는 world 이동량이라 기준 회전이 바뀌면 같은 rel_local 도 다른 world 궤적을 준다 — 그래서
    # `basis_c2w` 로 푼다 (`c2w_start` 로 풀면 traj_basis=subject 에서 τ 가 목표를 빗나간다).
    # ── follow (D72). subject 변위를 **위치**에 실어 CameraBench 의 tail/lead/side/aerial
    # tracking shot 을 만든다. `decision.json` 이 이기고 없으면 CLI (target_tau 와 같은 규칙).
    #
    # 순서가 중요하다 — follow 를 `fit_tau` **앞**에서 정해야 한다. τ 는 world 이동량이고
    # follow 는 world 이동이므로, 나중에 더하면 `fit_tau` 가 맞춰 놓은 τ 가 그만큼 빗나간다.
    # 여기서는 follow 를 소스 위치에서 빼서 넘긴다: |p_shape + off − p_src| = |p_shape − (p_src − off)|
    # 이라 `fit_tau` 를 고치지 않고도 정확히 같은 값을 푼다.
    follow_gain = trajectory.get("follow_gain", follow_gain)
    # `track_*` preset 은 추종을 **이름에** 묶어 놨다 (D74). 명시 gain 이 0 일 때만 preset 기본값이
    # 이긴다 — 그래야 ① 기존 preset 은 dict 에 없으니 동작이 그대로고 ② track preset 이라도
    # `--follow_gain auto` 처럼 밖에서 준 값이 있으면 그게 이긴다.
    if str(follow_gain) in ("0", "0.0"):
        follow_gain = PRESET_FOLLOW.get(PRESET_ALIASES.get(preset, preset), follow_gain)
    follow_solved = str(follow_gain) == "auto"
    # 위치 채널만 한 번 더 편다 (D73). 조준(`reference_centers`)은 raw 를 그대로 쓴다 — 흔들림은
    # 병진 쪽이고, 조준은 subject 를 실제로 따라가야 맞다. gain 도 편 궤적 위에서 다시 푼다.
    follow_smooth = int(trajectory.get("follow_smooth", follow_smooth))
    # D171. keyframe 보간을 켜면 저역통과 대신 그쪽이 위치 채널을 만든다 (0 이면 예전 경로 그대로).
    follow_keyframes = int(trajectory.get("follow_keyframes", follow_keyframes))
    follow_kf_interp = str(trajectory.get("follow_kf_interp", follow_kf_interp))
    follow_centers = (keyframe_follow_centers(centers_world, follow_keyframes,
                                              follow_kf_interp, follow_smooth)
                      if follow_keyframes > 1
                      else smooth_follow_centers(centers_world, follow_smooth))
    if follow_solved:
        # `moving` 을 넘긴다 — 정적 노드의 OBB 중심 미끄러짐은 크기만으로 동적과 안 갈린다.
        follow_gain, _ = solve_follow_gain(follow_centers, src_c2w[:, :3, 3], c2w_start[:3, 3],
                                           moving=bool(node.get("moving", True)))
    follow_gain = float(follow_gain)
    offsets_world = follow_offset(follow_centers, follow_gain)

    # `fit_tau` 를 끄면 preset 모양이 **정의된 각도 그대로** 나간다 (scale 1.0). "orbit 정확히
    # 45도" 처럼 각도가 요구사항일 때 필요하다 — 이분법은 τ 예산에 맞춰 회전까지 같이 깎으므로
    # 요청한 각도를 지킬 방법이 없다. 대신 τ 는 결과값이 되고 예산 초과를 막아주지 않는다.
    # 결정 자체에 실려 오면 그게 이긴다 (`tau_refine` 과 같은 규칙 — 뱅크 재현용).
    #
    # τ 를 **무엇에 대해** 잴지 (D97, §`presets.PRESET_TAU_REF`). `fit_tau` 는 안 고친다 —
    # 바뀌는 건 `src_centers` 하나뿐이다.
    #   source: p_src(f) − off(f)          → τ = |p_shape + off − p_src| / z_med
    #   follow: p_start 상수 배열          → τ = |p_shape(f) − p_start| / z_med
    # `follow` 에서 상수인 이유는 기준 자체가 "추종만 하고 모양은 안 하는 카메라"이기 때문이다:
    # p_ref(f) = p_start + off(f) 이므로 |p_plan − p_ref| = |p_shape − p_start| 로 off 가 지워진다.
    # 그래서 `tau_start` 가 항상 0 이고 `tau_saturated` 가 원리적으로 안 뜬다.
    tau_ref = resolve_tau_ref(preset, trajectory.get("tau_ref", tau_ref))
    if tau_ref == "follow":
        tau_centers = np.repeat(basis_c2w[None, :3, 3], len(rel_local), axis=0)
    else:
        tau_centers = src_c2w[:, :3, 3] - offsets_world
    use_fit_tau = bool(trajectory.get("fit_tau", use_fit_tau))
    # F9. 이분법이 **무엇을 깎을지**. 기본(off)은 SE(3) 로그 전체 = 이동·회전 동시 축소인데,
    # `true_orbit` 은 `R = |t|/θ` 라 그러면 반경이 안 줄고 sweep 만 줄어든다 (§presets.shape_resizer).
    # 켜면 sweep 을 상수로 두고 반경/dolly/lateral 만 스케일한다. sweep 을 안 쓰는 preset 은
    # `shape_resizer` 가 None 을 줘서 옛 경로 그대로 — 그쪽 뱅크는 비트 단위로 안 바뀐다.
    orbit_fixed_sweep = bool(trajectory.get("orbit_fixed_sweep", orbit_fixed_sweep))
    resize = shape_resizer(preset, context, num_frames, speed) if orbit_fixed_sweep else None
    if use_fit_tau:
        rel_local, tau_info = fit_tau(rel_local, basis_c2w, tau_centers,
                                      z_med, target_tau,
                                      refine_zero=bool(trajectory.get("tau_refine", False)),
                                      resize=resize)
    else:
        tau0, _ = tau_of(np.stack([np.eye(4)] * len(rel_local)), basis_c2w, tau_centers, z_med)
        tau_now, _ = tau_of(rel_local, basis_c2w, tau_centers, z_med)
        tau_info = {"scale": 1.0, "tau_max": round(tau_now, 4), "tau_start": round(tau0, 4),
                    "tau_saturated": False, "target_tau": target_tau, "tau_refined": False,
                    "tau_resize": False}
    tau_info = {**tau_info, "fit_tau": bool(use_fit_tau), "tau_ref": tau_ref,
                "orbit_fixed_sweep": bool(orbit_fixed_sweep)}

    # ── world 로 올리고 조준
    poses = T.start_at(basis_c2w, rel_local)
    poses[:, :3, 3] += offsets_world      # frame 0 은 정의상 0 이라 시작 pose 는 그대로다

    look_at = np.zeros((num_frames, 3))
    keyframe_info = None
    deroll_skipped = 0      # 시선이 중력축과 평행해 roll 을 못 세운 프레임 수 (D99)
    # D90. `aim="free"` 는 조준 자체를 안 하므로 keyframe 블록을 통째로 건너뛴다. `poses` 는 이미
    # `T.start_at(basis_c2w, rel_local)` 라 회전이 `basis @ rel_local[f]` — traj_basis="source"
    # 면 소스 frame0 회전 위에 preset 회전만 얹힌 것이고, 그게 "첫 카메라 기준 움직임"이다.
    # 여기서 keyframe 을 태우면 `preset_rel` 이 조준을 도로 끌고 들어온다 (D90 이전 버그).
    if aim == "free":
        aim_keyframes = 0
    if aim_keyframes:
        # ── D71. 회전만 keyframe 에서 세우고 사이는 측지선으로 잇는다. 위치는 안 건드린다
        # (preset 모양 = τ 예산이 이미 닫아 놓은 것). keyframe 0 이 소스 회전이라 `aim_anchor`
        # 없이도 frame 0 이 소스와 완전히 같다 — 대신 "언제 target 을 보게 되나"가 keyframe
        # 간격으로 정해진다 (F=49, N=5 면 12프레임 안에 넘어간다).
        assert keyframe_aim in ("auto", "target", "preset_rel"), \
            f"모르는 keyframe_aim: {keyframe_aim}"
        assert keyframe_ease in ("smoothstep", "linear", "arclen", "arclen_kf", "smooth_kf"), \
            f"모르는 keyframe_ease: {keyframe_ease}"
        mode = keyframe_aim if keyframe_aim != "auto" else (
            "target" if aim == "look_at" else "preset_rel")
        reference = reference_centers(centers_world, tracking) + bias_world
        frames = keyframe_indices(num_frames, int(aim_keyframes))
        rotations = aim_keyframe_rotations(poses, rel_local, reference, up_world, frames,
                                           c2w_start[:3, :3], mode)
        if keyframe_ease in ("arclen", "arclen_kf", "smooth_kf"):
            # D86. 구간별 ease 를 버리고 누적 호길이 위에서 잇는다 — 각속도 펌핑 제거.
            # arclen 은 전역 smoothstep(keyframe 시각이 밀린다), arclen_kf 는 PCHIP(시각 보존).
            # D89 smooth_kf 는 재타이밍이 아니라 折れ線 자체를 깎는다 — aim_err 를 대가로 낸다.
            schedule = (arclen_rotation_schedule(rotations, num_frames)
                        if keyframe_ease == "arclen"
                        else smooth_kf_schedule(rotations, frames, num_frames,
                                                smooth_passes, smooth_lambda)
                        if keyframe_ease == "smooth_kf"
                        else arclen_kf_schedule(rotations, frames, num_frames))
            for f in range(num_frames):
                poses[f][:3, :3] = schedule[f]
        else:
            for index in range(len(frames) - 1):
                a, b = frames[index], frames[index + 1]
                for f in range(a, b + 1):
                    x = (f - a) / (b - a)
                    if keyframe_ease == "smoothstep":
                        x = x * x * (3.0 - 2.0 * x)   # keyframe 에서 각속도 0 — 이음매가 안 튄다
                    poses[f][:3, :3] = slerp_rotation(rotations[index], rotations[index + 1], x)
        poses[0][:3, :3] = np.asarray(c2w_start[:3, :3], dtype=float)   # 잔차 없이 완전히 같게
        # keyframe 회전 **자체**의 roll. `arclen` 은 회전이 자기 프레임 번호에 도착하지 않으므로
        # (호길이로 재타이밍했다) `poses[frames]` 에서 재면 아무 의미가 없다 — 조준이 맞는지는
        # 회전 자체에서 재야 한다. 나머지 ease 는 keyframe 에서 x=0/1 이라 둘이 같은 값이다.
        keyframe_rolls = []
        for rotation in rotations:
            probe = np.eye(4)
            probe[:3, :3] = rotation
            keyframe_rolls.append(float(roll_about(probe, up_world)))
        if deroll:
            # D99. keyframe 사이 측지선이 만든 roll 을 걷어낸다. 목표 profile 은 **keyframe roll 을
            # 프레임 위에서 선형보간**한 것 — frame 0 은 소스 roll 그대로라 `poses[0]==c2w_start`
            # assert 가 그대로 살고, keyframe k>0 이 0 인 `target` 모드면 첫 구간에서만 풀리고
            # 그 뒤는 정확히 0 이다. 광축을 안 건드리므로 `aim_err_*` 는 한 자리도 안 변한다.
            profile = np.interp(np.arange(num_frames), np.asarray(frames, dtype=float),
                                np.asarray(keyframe_rolls, dtype=float))
            deroll_skipped = deroll_poses(poses, up_world, profile)
            poses[0][:3, :3] = np.asarray(c2w_start[:3, :3], dtype=float)   # 극 근처여도 f0 은 고정
        for f in range(num_frames):
            distance = float(np.linalg.norm(reference[f] - poses[f][:3, 3]))
            look_at[f] = poses[f][:3, 3] + poses[f][:3, 2] * distance
        # 조준이 실제로 얼마나 빗나가는지 — `target` 이면 keyframe 에서 0 이고 사이에서만 커진다.
        # 이 숫자가 keyframe 을 몇 개 둘지의 유일한 근거다 (많을수록 look_at 매프레임에 가까워진다).
        offsets = _aim_offsets(poses, reference)
        keyframe_info = {"count": len(frames), "frames": frames, "mode": mode,
                         "ease": keyframe_ease,
                         "keyframe_rolls": [round(r, 4) for r in keyframe_rolls],
                         "turn_deg": round(float(np.degrees(rotation_log(
                             rotations[1] @ rotations[0].T)[1])), 3) if len(rotations) > 1 else 0.0,
                         "aim_err_max_deg": round(float(offsets.max()), 3),
                         "aim_err_med_deg": round(float(np.median(offsets)), 3),
                         "aim_err_frame0_deg": round(float(offsets[0]), 3),
                         # keyframe **위**에서의 조준 오차. 재타이밍(smoothstep/arclen_kf)은 여기가
                         # 0 이고, 折れ線을 깎는 smooth_kf 만 0 이 아니다 — 평활의 대가가 이 값이다.
                         "aim_err_at_kf_deg": round(float(max(offsets[f] for f in frames)), 3),
                         "deroll": bool(deroll), "deroll_skipped": int(deroll_skipped)}
    elif aim == "look_at":
        reference = reference_centers(centers_world, tracking)
        for f in range(num_frames):
            look_at[f] = reference[f] + bias_world
            poses[f] = look_at_c2w(poses[f][:3, 3], look_at[f], up_world)
    else:
        # D99. `aim="free"` 는 preset 회전이 **카메라 로컬** 축이라 (traj.pan = rot_y) 기울어진
        # 카메라에서 pan 하면 지평선이 같이 돈다. 삼각대 pan 은 중력 둘레다 — 광축은 그대로 두고
        # roll 만 **frame 0 값으로 고정**한다. 0 이 아니라 frame0 값인 이유: `poses[0]==c2w_start`
        # assert(:802) 가 소스 카메라의 기울기를 그대로 요구하기 때문이다.
        if deroll:
            probe = np.eye(4)
            probe[:3, :3] = poses[0][:3, :3]
            deroll_skipped = deroll_poses(poses, up_world,
                                          np.full(num_frames, roll_about(probe, up_world)))
            poses[0][:3, :3] = np.asarray(c2w_start[:3, :3], dtype=float)
        for f in range(num_frames):
            look_at[f] = poses[f][:3, 3] + poses[f][:3, 2] * radius_world

    # ── 조준 앵커 (D26). frame 0 회전이 소스와 어긋나 있을 때만 할 일이 있다. 그 경우는 둘이다:
    # ① `aim="look_at"` — 조준이 시작 회전을 덮어썼다. ② `traj_basis="subject"` — preset 을
    # subject 기준 위에 얹었다 (D69). `aim="traj"` + `traj_basis="source"` 는 poses[0] 이 이미
    # `c2w_start` 라 되돌릴 게 없다 (기존 동작).
    # keyframe 조준(D71)이면 keyframe 0 이 이미 소스 회전이라 앵커가 할 일이 없다 — 여기서 또
    # 되돌리면 이번엔 keyframe 1 쪽으로 두 번 감기고 첫 구간이 램프 두 개로 겹친다.
    anchor = resolve_aim_anchor(aim_anchor, start_mode)
    ramp = max(int(aim_ramp_frames), 1)
    anchor_deg, anchor_applied = 0.0, False
    if not aim_keyframes and (aim == "look_at" or traj_basis == "subject") \
            and anchor == "source_frame0":
        # frame 0 의 조준 오차를 world 회전 하나로 표현하고, 그걸 smoothstep 으로 풀어준다.
        # f=0 에서 100%(→ poses[0] == c2w_start 정확히), f>=ramp 에서 0%(→ 원래 조준 그대로).
        offset_axis, offset_angle = rotation_log(c2w_start[:3, :3] @ poses[0][:3, :3].T)
        anchor_deg, anchor_applied = float(np.degrees(offset_angle)), True
        for f in range(min(ramp, num_frames)):
            x = f / ramp
            weight = x * x * (3.0 - 2.0 * x)        # f=0 에서 기울기 0 — 첫 프레임이 튀지 않는다
            poses[f][:3, :3] = rotation_exp(offset_axis,
                                            offset_angle * (1.0 - weight)) @ poses[f][:3, :3]
        poses[0] = c2w_start.copy()                 # 수치 잔차 없이 **완전히** 같게
        # ramp 구간은 더 이상 subject 를 정확히 안 본다. `look_at` 을 그대로 두면 npz 와 plan_cam
        # 의 시선 화살표가 실제 회전과 어긋난다 — 실제 시선축 위의 같은 거리로 다시 찍는다.
        for f in range(min(ramp, num_frames)):
            distance = float(np.linalg.norm(look_at[f] - poses[f][:3, 3]))
            look_at[f] = poses[f][:3, 3] + poses[f][:3, 2] * distance

    for f in range(num_frames):
        assert abs(np.linalg.det(poses[f][:3, :3]) - 1.0) < 1e-9, f"frame {f}: det(R) != 1"
    rolls = np.array([roll_about(poses[f], up_world) for f in range(num_frames)])
    if aim_keyframes:
        # keyframe 사이는 측지선 보간이라 roll 이 정확히 0 은 아니다 (양 끝이 0 이어도 중간에
        # 샌다). 판정은 **keyframe 위**에서만 — 거기서 새면 조준 자체가 틀린 것이다. frame 0 은
        # 소스 카메라의 기울기를 그대로 물려받으므로 제외하고, `preset_rel` 은 preset 회전이
        # 얹혀 있어 정의상 0 이 아니다.
        if keyframe_info["mode"] == "target":
            if keyframe_ease in ("arclen", "smooth_kf"):
                # smooth_kf 는 折れ線을 깎느라 keyframe 회전을 정확히 통과하지 않는다 (의도된 것).
                # 그러니 `poses[frames]` 의 roll 은 조준 오류가 아니라 평활량이다 — 대신 그 양은
                # `aim_err_at_kf_deg` 로 따로 찍는다. 판정은 arclen 과 같이 회전 자체에서.
                checked = [abs(r) for r in keyframe_info["keyframe_rolls"][1:]]
            else:
                checked = [abs(rolls[f]) for f in keyframe_info["frames"] if f != 0]
            assert not checked or max(checked) < ROLL_TOL_DEG, \
                f"keyframe 조준인데 keyframe 에서 roll 이 남았다: {max(checked)} deg"
    elif aim == "look_at":
        # 앵커를 걸면 ramp 구간은 소스 카메라의 기울기를 일부러 되살린 것이라 roll 이 0 이 아니다.
        # 그래서 assert 는 ramp **바깥**에만 건다 — 거기서도 깨지면 조준 자체가 틀린 것이다.
        checked = rolls[ramp:] if anchor_applied else rolls
        if len(checked):
            assert np.abs(checked).max() < ROLL_TOL_DEG, \
                f"look_at 조준인데 roll 이 남았다: {np.abs(checked).max()} deg"
    # frame0 == 소스 카메라를 **의도한** 경우에만 건다. `traj_basis="subject"` + `anchor="subject"`
    # 는 일부러 subject 를 향해 시작하는 조합이라 여기서 걸면 안 된다.
    if start_mode == "source_frame0" and (
            anchor_applied or bool(aim_keyframes)
            or (aim != "look_at" and traj_basis == "source")):
        gap = float(np.abs(poses[0] - c2w_start).max())
        assert gap < 1e-9, f"frame 0 이 소스 카메라와 다르다 ({gap:.3e}) — 앵커가 안 먹었다"

    tau = np.linalg.norm(poses[:, :3, 3] - src_c2w[:, :3, 3], axis=-1) / max(z_med, 1e-9)
    view = _view_angles(poses[:, :3, 3], centers_world, src_c2w[:, :3, 3])
    info = {"preset": preset, "aim": aim, "speed": speed, "tracking": tracking,
            "tracking_ignored": bool(tracking_ignored), "needs_zoom": bool(needs_zoom),
            # D93. 요청값과 실제 쓴 값이 다를 수 있다 (`track_*` -> lock). 뱅크 행은 실제 값을 쓴다.
            "tracking_requested": tracking_requested,
            "tracking_from_preset": bool(tracking != tracking_requested),
            "aim_anchor": anchor, "aim_anchor_applied": bool(anchor_applied),
            # D99. 실제로 쓴 값을 적는다 (`tracking` 과 같은 규칙). `auto` 였을 때만 요청값도 남겨
            # 뱅크 행만 보고 "왜 subject 로 섰나"를 되짚을 수 있게 한다.
            "traj_basis": traj_basis, "aim_keyframes": keyframe_info,
            **({"traj_basis_requested": traj_basis_requested}
               if traj_basis_requested != traj_basis else {}),
            "follow": {"gain": round(follow_gain, 4), "solved": bool(follow_solved),
                       "smooth": int(follow_smooth),
                       # D171. 껐으면(0) 키를 안 남긴다 — 옛 행과 열이 같아야 diff 가 조용하다.
                       **({"keyframes": int(follow_keyframes),
                           "kf_interp": str(follow_kf_interp)} if follow_keyframes > 1 else {}),
                       # 이 궤적이 CameraBench 로 무슨 shot 인지 (gain=0 이면 tracking shot 이 아니다).
                       **({"kind": "none"} if follow_gain == 0.0 else
                          follow_shot_kind(c2w_start[:3, 3], centers_world, up_world)),
                       "offset_path_u": round(float(np.linalg.norm(
                           np.diff(offsets_world, axis=0), axis=-1).sum() / scale), 5)},
            "aim_ramp_frames": int(ramp) if anchor_applied else 0,
            "aim_anchor_deg": round(anchor_deg, 4),
            "start_source": start_source, "radius_world": round(radius_world, 6),
            "radius_u": round(radius_world / scale, 6), "look_at_bias": bias,
            "shape_context": {k: round(v, 5) for k, v in context.items()},
            "tau": {**tau_info, "tau_max_final": round(float(tau.max()), 4)},
            # intrinsic zoom. SE(3) 밖이라 τ 와 무관하고 `emit` 은 이걸 버린다
            # (`zoom_dropped`) — 소비자는 검증 렌더러뿐이다 (§emit).
            "zoom": {"focal_end": round(float(focal[-1]), 5), "needs_zoom": bool(needs_zoom)},
            "view_angle_max_deg": round(float(view.max()), 2),
            "roll_max_deg": round(float(np.abs(rolls).max()), 6),
            # D99. frame 0 은 소스 기울기를 물려받으므로 **보정 후에도 0 이 아니다** — 궤적이
            # 만든 roll 만 보려면 이 값을 볼 것 (deroll 이 켜져 있으면 극 근처 말고는 0 이다).
            "roll_vs_frame0_max_deg": round(float(np.abs(rolls - rolls[0]).max()), 6),
            "deroll": bool(deroll), "deroll_skipped": int(deroll_skipped),
            "path_len_u": round(float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0),
                                                     axis=-1).sum() / scale), 5),
            # `z_med` 는 이름만 예전 것이고 실제로는 **τ 의 분모**다. 어느 정의인지는
            # `tau_denom` 이 말해준다 (D150) — 이 키가 없는 옛 산출물은 `z_med_frame0` 이다.
            "S": scale, "z_med": z_med, "tau_denom": tau_denom}
    return poses, {"look_at": look_at, "subject_track": centers_world, "tau_per_frame": tau,
                   "view_angle_deg": view, "focal_scale": focal, "info": info}


def resolve_start_mode(decision: dict, cli_value: str = "source_frame0"):
    """`decision.json` 에 `start_mode` 가 있으면 **그게 진실**이다. CLI 값은 옛 파일용 기본값.

    루프가 `source_frame0` 로 돌았다는 것은 select/micro 를 아예 안 돌렸다는 뜻이라, 그 결정을
    `board` 로 디코드하면 존재하지도 않는 후보를 시작점으로 삼는다. 반대도 마찬가지다 — `board`
    로 고른 결정을 `source_frame0` 로 디코드하면 VLM 이 고른 pose 가 통째로 버려진다. 어느 쪽도
    에러 없이 "다른 궤적"만 내놓으므로, CLI 가 decision 을 이기게 두면 안 된다.
    """
    stored = decision.get("start_mode")
    if stored is None:
        return cli_value
    if stored != cli_value:
        print(f"!! start_mode: decision.json 은 '{stored}', CLI 는 '{cli_value}' — decision 을 따른다")
    return stored


def _aim_offsets(poses: np.ndarray, targets: np.ndarray):
    """프레임별 ∠(광축, target − p). 0 이면 target 이 화면 정중앙이다.

    `subject_in_frame` 과 다른 양이다 — 저건 렌더가 필요하고 이건 안 필요하다. keyframe 간격을
    고를 때 렌더 없이 볼 수 있는 유일한 눈금이라 info 에 싣는다 (half-hfov 와 비교할 것).
    """
    axis = poses[:, :3, 2]
    direction = np.asarray(targets, dtype=float) - poses[:, :3, 3]
    cos = np.sum(axis * direction, axis=-1) / np.maximum(
        np.linalg.norm(axis, axis=-1) * np.linalg.norm(direction, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _view_angles(positions: np.ndarray, targets: np.ndarray, src_centers: np.ndarray):
    """프레임별 ∠(target−p_plan, target−p_src). 소스가 본 방향에서 얼마나 돌아갔나."""
    plan = targets - positions
    source = targets - src_centers
    cos = np.sum(plan * source, axis=-1) / np.maximum(
        np.linalg.norm(plan, axis=-1) * np.linalg.norm(source, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def main(args):
    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    with open(path.join(out_root, args.video, "scene_graph.json"), encoding="utf-8") as file:
        graph = json.load(file)
    with open(args.decision or path.join(out_root, args.video, "decision.json"),
              encoding="utf-8") as file:
        decision = json.load(file)
    board_path = args.board or path.join(out_root, args.video, "board", "board.json")
    board = None
    if path.isfile(board_path):
        with open(board_path, encoding="utf-8") as file:
            board = json.load(file)

    start_mode = resolve_start_mode(decision, args.start_mode)
    poses, extra = build_poses(decision, graph, board, target_tau=args.target_tau,
                               orbit_span_frac=args.orbit_span_frac,
                               start_mode=start_mode, aim_anchor=args.aim_anchor,
                               aim_ramp_frames=args.aim_ramp_frames,
                               traj_basis=args.traj_basis,
                               aim_keyframes=args.aim_keyframes,
                               keyframe_aim=args.keyframe_aim,
                               keyframe_ease=args.keyframe_ease,
                               follow_gain=args.follow_gain,
                               follow_smooth=args.follow_smooth,
                               follow_keyframes=args.follow_keyframes,
                               follow_kf_interp=args.follow_kf_interp,
                               smooth_passes=args.smooth_passes,
                               smooth_lambda=args.smooth_lambda,
                               preset_tracking=args.preset_tracking,
                               deroll=args.deroll)
    output = args.output or path.join(out_root, args.video, "poses.npz")
    np.savez_compressed(output, cam_c2w=poses, look_at=extra["look_at"],
                        subject_track=extra["subject_track"], tau=extra["tau_per_frame"],
                        view_angle_deg=extra["view_angle_deg"],
                        info=json.dumps(extra["info"], ensure_ascii=False),
                        decision_fingerprint=decision_fingerprint(
                            decision, start_mode, args.aim_anchor, args.aim_ramp_frames,
                            args.traj_basis, args.aim_keyframes, args.keyframe_aim,
                            args.keyframe_ease, args.follow_gain, args.follow_smooth,
                            # D171 로 서명 중간에 두 인자가 늘었다 — 위치 인자로 넘기면
                            # `deroll` 이 `follow_keyframes` 자리로 밀린다. 키워드로 못 박는다.
                            follow_keyframes=args.follow_keyframes,
                            follow_kf_interp=args.follow_kf_interp,
                            deroll=args.deroll))

    info = extra["info"]
    print(f"{args.video}  subject {decision['subject_id']}  preset {info['preset']} "
          f"(aim {info['aim']}, speed {info['speed']}, tracking {info['tracking']}"
          f"{' [ignored]' if info['tracking_ignored'] else ''})")
    print(f"{'start_source':<20}{info['start_source']}")
    print(f"{'radius_u':<20}{info['radius_u']:.4f}")
    print(f"{'traj_scale':<20}{info['tau']['scale']:.5f}"
          f"{'  (saturated)' if info['tau']['tau_saturated'] else ''}")
    print(f"{'tau_start':<20}{info['tau']['tau_start']:.4f}")
    print(f"{'tau_max':<20}{info['tau']['tau_max_final']:.4f}  (target {info['tau']['target_tau']})")
    print(f"{'view_angle_max':<20}{info['view_angle_max_deg']:.2f} deg")
    print(f"{'path_len_u':<20}{info['path_len_u']:.4f}")
    print(f"{'roll_max':<20}{info['roll_max_deg']:.2e} deg")
    # 앵커가 안 걸리는 경우가 둘이다 — 껐거나(subject), preset 이 애초에 조준을 안 덮어쓰거나(traj).
    skipped = "" if info["aim_anchor_applied"] else (
        "  (off)" if info["aim_anchor"] == "subject" else "  (n/a — aim=traj)")
    print(f"{'aim_anchor':<20}{info['aim_anchor']}{skipped}")
    if info["aim_anchor_applied"]:
        print(f"{'aim_anchor_deg':<20}{info['aim_anchor_deg']:.4f} deg"
              f"  (ramp {info['aim_ramp_frames']} frames)")
    if info["follow"]["gain"]:
        follow = info["follow"]
        print(f"{'follow_gain':<20}{follow['gain']:.3f}"
              f"{'  (solved)' if follow['solved'] else ''}  {follow['kind']}"
              f"  (angle {follow['angle_deg']:.1f} deg, elev {follow['elevation_deg']:.1f} deg)")
        print(f"{'  offset_path_u':<20}{follow['offset_path_u']:.4f}"
              f"  (smooth w{follow['smooth']})")
    if info["aim_keyframes"]:
        # aim_err 는 렌더 없이 볼 수 있는 유일한 조준 눈금이다 — half-hfov 와 비교할 것.
        keys = info["aim_keyframes"]
        print(f"{'aim_keyframes':<20}{keys['count']} @ {keys['frames']}  "
              f"({keys['mode']}, {keys['ease']})")
        print(f"{'  turn_deg':<20}{keys['turn_deg']:.2f} deg  (frame 0 -> keyframe 1)")
        print(f"{'  aim_err':<20}max {keys['aim_err_max_deg']:.2f}  "
              f"med {keys['aim_err_med_deg']:.2f}  frame0 {keys['aim_err_frame0_deg']:.2f} deg")
    print(f"\n-> {output}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--decision", default=None, type=str)          # 없으면 out/<video>/decision.json
    parser.add_argument("--board", default=None, type=str)             # 후보 pose 출처 (없으면 p_G 재구성)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--target_tau", default=0.20, type=float)      # decision 에 있으면 그쪽이 우선
    parser.add_argument("--orbit_span_frac", default=0.8, type=float)  # sweep <= 이 값 x obs_az_span
    # 시작 pose 출처. source_frame0 = 소스 카메라 frame0 그대로(궤적만 결정), board = 예전 동작.
    # decision.json 에 start_mode 필드가 있으면 그쪽이 이긴다 (resolve_start_mode).
    parser.add_argument("--start_mode", default="source_frame0",
                        choices=["source_frame0", "board"])
    # 조준 앵커 (D26/D27). 기본 subject = frame 0 부터 subject 조준 (사용자 선택).
    # source_frame0 = 첫 프레임을 소스와 완전히 같게(ramp 로 품). auto = start_mode 를 따라감.
    parser.add_argument("--aim_anchor", default="subject",
                        choices=["auto", "source_frame0", "subject"])
    parser.add_argument("--aim_ramp_frames", default=12, type=int)      # 앵커를 푸는 데 쓰는 프레임 수
    # preset 모양을 얹을 기준 회전 (D69). source = 기존(소스 광축 기준), subject = anchor 를 향해
    # 세운 회전 기준. aim="traj" preset(pan/truck/pedestal)이 anchor 를 안 보는 문제를 고친다.
    # D99. auto = 조준 방식이 정한다 (look_at -> subject, free -> source). **기본값은 아직
    # `source`** — 여기만 바꾸면 굽는 중인 뱅크가 두 규약으로 섞인다 (D90 smooth_passes 사고).
    parser.add_argument("--traj_basis", default="source",
                        choices=["source", "subject", "auto"])
    # 조준 keyframe 수 (D71). 0 = 기존 동작(매 프레임 조준 또는 조준 없음). N>=2 면 frame 0 은
    # 소스 회전 그대로, 나머지 keyframe 은 anchor 를 보고, 사이는 SO(3) 측지선으로 잇는다.
    parser.add_argument("--aim_keyframes", default=0, type=int)
    parser.add_argument("--keyframe_aim", default="auto",
                        choices=["auto", "target", "preset_rel"])
    # D89. smooth_kf = 선형 slerp 折れ線 + SO(3) Laplacian 평활 (재타이밍이 아니라 모서리를
    # 깎는다) — **기본값**. 714 변이 실측에서 리타이밍만으로는 각속도 맥동비가 21.65 까지밖에
    # 안 내려가는데(keyframe 정체가 구조적) smooth_kf 는 4.58 로 떨어뜨린다. 대가는 keyframe
    # 회전을 정확히 통과하지 않는 것뿐이고, **49프레임 전체 조준오차 median 은 오히려 더 낮다**
    # (smooth p4 0.418° / arclen_kf 0.430° / smoothstep 0.668°). `aim_err_at_kf_deg` 로 확인.
    # D86. arclen_kf = 누적 호길이 PCHIP (각속도 평탄 + keyframe 시각 보존).
    # arclen = 누적 호길이 위 전역 smoothstep (더 평탄하지만 keyframe 시각이 밀린다).
    # smoothstep 은 D71~D84 기존 동작 (배포된 뱅크 전량), linear 는 구간별 등속.
    parser.add_argument("--keyframe_ease", default="smooth_kf",
                        choices=["smoothstep", "linear", "arclen", "arclen_kf", "smooth_kf"])
    # 맥동비는 p4 에서 이미 포화(4.58 -> p24 4.36)하는데 keyframe 조준오차는 계속 커진다
    # (kf_dev med 1.53° -> 3.85°). p4 가 효율 구간의 시작점이라 기본값이다.
    parser.add_argument("--smooth_passes", default=4, type=int)      # smooth_kf 반복 횟수
    parser.add_argument("--smooth_lambda", default=0.5, type=float)  # pass 당 이동 비율
    # subject 변위를 카메라 **위치**에 싣는 비율 (D72). 0 = 기존 동작, 1 = 완전 동행(CameraBench
    # tail/lead/side/aerial tracking), "auto" = τ 를 최소로 만드는 값을 풀어서 쓴다.
    parser.add_argument("--follow_gain", default="0", type=str)
    # follow 위치 채널 저역통과 창 (D73, savgol p=2). 1 = 끔(기존 동작). 9 가 실측 무릎이다 —
    # 흔들림(|jerk| p95)을 4.4배 깎으면서 τ 는 오히려 좋아지고 subject 이탈은 0.012u 다.
    parser.add_argument("--follow_smooth", default=9, type=int)
    # D171. follow 위치 채널을 균등 keyframe N개로 줄였다가 다시 채운다 (사용자 지시 2026-09-10
    # "recon 이 깔끔한 sparse keyframe 들을 기준으로 translation 도 interpolate"). 0/1 = 끔 =
    # `--follow_smooth` 저역통과 그대로 = 기존 동작. **기본값을 0 으로 두는 이유**: 이 손잡이는
    # 아직 육안 대조(D171-c 릴) 전이라 채택된 게 아니다. 사다리·게이트가 아니라 궤적 모양을
    # 바꾸므로, 기본을 켜면 진행 중인 굽기가 코퍼스 중간에 정의를 갈아탄다 (`tau_denom` 과 같은 이유).
    parser.add_argument("--follow_keyframes", default=0, type=int)
    parser.add_argument("--follow_kf_interp", default="cubic",
                        choices=["linear", "savgol", "cubic"])
    # D93. `track_*` preset 의 조준 추종률을 위치 추종률(`PRESET_FOLLOW`=1.0)에 맞춰 lock 으로.
    # 끄면 `decision.json` 의 `tracking` 이 그대로 간다 = D93 이전 동작.
    parser.add_argument("--preset_tracking", action="store_true", default=True)
    parser.add_argument("--no_preset_tracking", dest="preset_tracking", action="store_false")
    # D99. 중력축 기준 roll 재정렬. **기본값 False 는 `fit_hole_ladder` / `sample_camera_bank` 와
    # 같아야 한다** — 여기만 켜면 같은 뱅크를 두 규약으로 다시 굽게 된다 (D90 smooth_passes 사고).
    parser.add_argument("--deroll", action="store_true", default=False)
    parser.add_argument("--no_deroll", dest="deroll", action="store_false")
    main(parser.parse_args())
