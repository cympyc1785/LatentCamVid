"""micro-adjust 연산 — LBM 16종(`cinematographer_stage.py:3867-3871`)을 우리 단위로 옮긴다.

왜 후보 board 만으로 안 되나: board 는 τ 예산 안에서 편 격자라 눈금이 성기다 (camel 은 방위
±2.6°, 반경 ±20% 를 5x3x3 으로 쪼갠다). "subject 를 왼쪽 3분할선에 놓아라" 같은 구도 요구는
격자 눈금 사이에 있다. micro-adjust 는 그 사이를 메우는 연산이고, **매 연산 뒤에 게이트를 다시
돌려** 통과 못 하면 되돌린다 — 그게 "LLM 이 공간 제약을 모른다"에 대한 처방이다.

## 상태는 G 좌표의 (위치, 타겟) 두 점이다

행렬을 직접 굴리지 않는다. `(p_g, look_at_g, focal_scale)` 만 들고 다니고 world c2w 는 필요할 때
`candidates.g_pose_to_world` 로 다시 세운다. 이유는 `look_at_c2w` 가 매번 중력축 기준 roll=0 을
보장해 주기 때문이다 — 회전행렬을 누적해서 굴리면 10라운드쯤에서 Dutch angle 이 슬금슬금 낀다.

## 방위각 부호 (유도해서 못 박아 둔다)

subject 중심에서 방위각 φ 로 본 카메라의 right 벡터는 `fwd x up` 이고, φ=0 (즉 +x 쪽)에서
`fwd=(-1,0,0)`, `up=(0,0,1)` 이면 `right=(0,1,0)` = **φ 가 커지는 방향**이다. 즉 카메라가 제
왼쪽으로 도는 것은 **φ 가 작아지는 것**이다:

    orbit_left  -> azimuth -10 deg      orbit_right -> azimuth +10 deg
    pan_left    -> 시선을 +z 축 기준 +3 deg 회전 (같은 유도의 반대편)

`presets.py` 의 `arc(+) = 카메라가 왼쪽` 규약과 부호가 반대로 보이는데, 저긴 카메라 로컬 궤적
파라미터고 여긴 subject local 구면좌표라 서로 다른 자다. 둘을 섞지 말 것.

## orbit 은 회전행렬이 아니라 구면좌표로 한다

축 회전으로 짜면 elevation clamp(5~70°)를 걸 자리가 없다. 구면좌표에서 각도를 더하면 clamp 가
자명하고, 반경이 부동소수점으로 새지도 않는다.

예시:
    python -m lbm.ops --list
"""
import numpy as np

from .candidates import subject_frame

G_UP = np.array([0.0, 0.0, 1.0])

# 연산 -> (종류, 부호). 기본 14종 + zoom 2종(`--allow_zoom` 일 때만 메뉴에 오른다).
OPS = {
    "pan_left":     ("pan_yaw", +1.0),
    "pan_right":    ("pan_yaw", -1.0),
    "pan_up":       ("pan_pitch", +1.0),
    "pan_down":     ("pan_pitch", -1.0),
    "orbit_left":   ("orbit_az", -1.0),
    "orbit_right":  ("orbit_az", +1.0),
    "orbit_up":     ("orbit_el", +1.0),
    "orbit_down":   ("orbit_el", -1.0),
    "truck_left":   ("translate_right", -1.0),
    "truck_right":  ("translate_right", +1.0),
    "pedestal_up":  ("translate_up", +1.0),
    "pedestal_down": ("translate_up", -1.0),
    "dolly_in":     ("translate_fwd", +1.0),
    "dolly_out":    ("translate_fwd", -1.0),
    "zoom_in":      ("zoom", +1.0),
    "zoom_out":     ("zoom", -1.0),
}
ZOOM_OPS = ("zoom_in", "zoom_out")

# 스텝 크기 (§계획서 B4 표 그대로). 이동은 subject 까지 거리 `r` 에 대한 비율이다.
STEPS = {"pan_deg": 3.0, "orbit_az_deg": 10.0, "orbit_el_deg": 8.0,
         "lateral_frac": 0.10, "dolly_frac": 0.15, "zoom_ratio": 1.15}

# clamp. elevation 은 지평 아래로 내려가거나 수직 부감이 되는 걸 막고(look_at 특이점),
# radius 는 subject 를 뚫고 들어가거나 점처럼 작아지는 걸 막는다.
LIMITS = {"elevation_deg": (5.0, 70.0), "radius_ratio": (0.4, 2.0), "focal_scale": (0.5, 2.0)}

OP_MENU_HINTS = {
    "pan_left": "turn the camera left in place (subject slides right)",
    "pan_right": "turn the camera right in place (subject slides left)",
    "pan_up": "tilt the camera up in place (subject slides down)",
    "pan_down": "tilt the camera down in place (subject slides up)",
    "orbit_left": "move the camera left around the subject, staying aimed at it",
    "orbit_right": "move the camera right around the subject, staying aimed at it",
    "orbit_up": "raise the camera on its arc around the subject",
    "orbit_down": "lower the camera on its arc around the subject",
    "truck_left": "slide the camera left without turning",
    "truck_right": "slide the camera right without turning",
    "pedestal_up": "slide the camera straight up without turning",
    "pedestal_down": "slide the camera straight down without turning",
    "dolly_in": "move the camera toward the subject (subject gets bigger)",
    "dolly_out": "move the camera away from the subject (subject gets smaller)",
    "zoom_in": "narrow the lens (subject gets bigger, camera does not move)",
    "zoom_out": "widen the lens (subject gets smaller, camera does not move)",
}


def rodrigues(vectors, axis, degrees: float):
    """`axis` 둘레로 `degrees` 만큼 회전 (오른손). axis 는 정규화해서 받는다."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    theta = np.radians(degrees)
    v = np.asarray(vectors, dtype=float)
    return (v * np.cos(theta) + np.cross(axis, v) * np.sin(theta)
            + axis * np.dot(axis, v) * (1.0 - np.cos(theta)))


def make_state(p_g, look_at_g, focal_scale: float = 1.0):
    return {"p_g": np.asarray(p_g, dtype=float).copy(),
            "look_at_g": np.asarray(look_at_g, dtype=float).copy(),
            "focal_scale": float(focal_scale)}


def camera_basis(state: dict):
    """(fwd, right) — `look_at_c2w` 와 같은 규약 (right = fwd x up, up = G 의 +z)."""
    fwd = state["look_at_g"] - state["p_g"]
    fwd = fwd / max(np.linalg.norm(fwd), 1e-12)
    right = np.cross(fwd, G_UP)
    norm = np.linalg.norm(right)
    if norm < 1e-6:                       # 시선이 중력축과 평행 — 특이점 회피 (look_at_c2w 와 동일)
        right = np.cross(fwd, G_UP + 1e-3 * np.array([1.0, 0.0, 0.0]))
        norm = np.linalg.norm(right)
    return fwd, right / norm


def spherical(state: dict, node: dict, frame: int = 0):
    """subject local 구면좌표 (azimuth_deg, elevation_deg, radius). candidates 와 같은 자."""
    center, R_j = subject_frame(node, frame)
    local = R_j.T @ (state["p_g"] - center)
    radius = float(np.linalg.norm(local))
    return (float(np.degrees(np.arctan2(local[1], local[0]))),
            float(np.degrees(np.arcsin(np.clip(local[2] / max(radius, 1e-9), -1.0, 1.0)))),
            radius)


def from_spherical(azimuth_deg: float, elevation_deg: float, radius: float,
                   node: dict, frame: int = 0):
    center, R_j = subject_frame(node, frame)
    phi, theta = np.radians(azimuth_deg), np.radians(elevation_deg)
    direction = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)])
    return center + radius * (R_j @ direction)


def apply_op(state: dict, name: str, node: dict, d_ref: float, frame: int = 0,
             steps: dict | None = None):
    """(new_state, note). clamp 에 걸리면 note 에 남긴다 — 조용히 무시하지 않는다.

    `d_ref` 는 노드의 기준 관측거리(u)로, radius clamp 의 기준자다. 이동 스텝은 `d_ref` 가 아니라
    **현재 반경** 에 비례한다 — 멀리 있으면 크게, 가까이 있으면 작게 움직여야 화면 변화가 비슷하다.
    """
    assert name in OPS, f"모르는 연산: {name} (있는 것: {sorted(OPS)})"
    steps = {**STEPS, **(steps or {})}
    kind, sign = OPS[name]
    azimuth, elevation, radius = spherical(state, node, frame)
    fwd, right = camera_basis(state)
    new = make_state(state["p_g"], state["look_at_g"], state["focal_scale"])
    note = None

    if kind == "pan_yaw":
        new["look_at_g"] = state["p_g"] + rodrigues(
            state["look_at_g"] - state["p_g"], G_UP, sign * steps["pan_deg"])
    elif kind == "pan_pitch":
        new["look_at_g"] = state["p_g"] + rodrigues(
            state["look_at_g"] - state["p_g"], right, sign * steps["pan_deg"])
    elif kind == "orbit_az":
        new["p_g"] = from_spherical(azimuth + sign * steps["orbit_az_deg"], elevation, radius,
                                    node, frame)
    elif kind == "orbit_el":
        lo, hi = LIMITS["elevation_deg"]
        target = elevation + sign * steps["orbit_el_deg"]
        clamped = float(np.clip(target, lo, hi))
        if abs(clamped - target) > 1e-9:
            note = f"elevation clamped to {clamped:.1f} deg (limit {lo:.0f}..{hi:.0f})"
        new["p_g"] = from_spherical(azimuth, clamped, radius, node, frame)
    elif kind in ("translate_right", "translate_up", "translate_fwd"):
        axis = {"translate_right": right, "translate_up": G_UP, "translate_fwd": fwd}[kind]
        frac = steps["dolly_frac"] if kind == "translate_fwd" else steps["lateral_frac"]
        delta = sign * frac * radius * axis
        # 평행이동은 위치와 타겟을 **같이** 옮긴다. 타겟을 두면 회전이 생겨서 arc 가 되어버리고,
        # 그건 truck 이 아니라 orbit 이다 (§presets.py 의 aim 논의와 같은 구분).
        new["p_g"] = state["p_g"] + delta
        new["look_at_g"] = state["look_at_g"] + delta
    elif kind == "zoom":
        ratio = steps["zoom_ratio"] ** sign
        lo, hi = LIMITS["focal_scale"]
        new["focal_scale"] = float(np.clip(state["focal_scale"] * ratio, lo, hi))
        if abs(new["focal_scale"] - state["focal_scale"] * ratio) > 1e-9:
            note = f"focal_scale clamped to {new['focal_scale']:.2f}"
    else:
        raise AssertionError(f"미구현 연산 종류: {kind}")

    # dolly 로 반경이 한계를 넘으면 되돌린다 (clamp 가 아니라 거부 — 중간값은 의미가 없다).
    if kind == "translate_fwd":
        _, _, new_radius = spherical(new, node, frame)
        lo, hi = LIMITS["radius_ratio"]
        if not (lo * d_ref <= new_radius <= hi * d_ref):
            return None, (f"radius {new_radius / max(d_ref, 1e-9):.2f}x d_ref out of "
                          f"[{lo}, {hi}] — not applied")
    return new, note


def op_menu(allow_zoom: bool = False, rejected: dict | None = None):
    """VLM 에게 보여줄 연산 메뉴 (영문). `rejected` 는 {op: 사유} — 목록에 사유를 붙여 남긴다.

    거절된 연산을 메뉴에서 **빼지 않는** 이유: 빼면 모델이 왜 사라졌는지 모른 채 다른 걸 고르고,
    같은 실패를 다른 이름으로 반복한다. 사유를 붙여 남겨야 다음 선택이 그걸 피해 간다.
    """
    rejected = rejected or {}
    lines = []
    for name in OPS:
        if name in ZOOM_OPS and not allow_zoom:
            continue
        line = f"{name:<15} {OP_MENU_HINTS[name]}"
        if name in rejected:
            line += f"   [REJECTED EARLIER: {rejected[name]}]"
        lines.append(line)
    lines.append(f"{'none':<15} the framing is good; stop adjusting (set done=true)")
    return "\n".join(lines)


def main(args):
    print(f"{'op':<15}{'kind':<18}{'sign':>6}  hint")
    for name, (kind, sign) in OPS.items():
        mark = " (needs --allow_zoom)" if name in ZOOM_OPS else ""
        print(f"{name:<15}{kind:<18}{sign:>+6.0f}  {OP_MENU_HINTS[name]}{mark}")
    print(f"\nsteps  {STEPS}")
    print(f"limits {LIMITS}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--list", action="store_true", default=True)   # 표만 찍는다
    main(parser.parse_args())
