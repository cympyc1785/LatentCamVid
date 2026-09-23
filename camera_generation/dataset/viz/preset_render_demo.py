"""preset 이름만 바꿨을 때 렌더가 실제로 달라지는지 **통제 실험**용 handoff 를 만든다.

**왜 필요한가.** 기존 변종 렌더(`trumans_c49_w01_f0000_0048`)는 `expand_preset_variants.py`
가 `--vary_end` 로 **끝점도 같이 갈랐다**. 그래서 클립이 달라 보여도 그게 preset 때문인지
끝점 때문인지 못 가른다. 여기서는 `start_transform` 과 `end_transform` 을 **전 카메라 공통으로
못박고 `trajectory_plan.preset_name` 하나만** 바꾼다 — 그러면 화면 차이는 preset 이 만든 것뿐이다.

**대조군 하나 더.** 마지막 카메라는 preset 이 아니라 우리가 계산한 **49개 keyframe** 을
`trajectory_plan.keyframes` 에 직접 싣는다. `blender_render_worker.py:1038-1050` 이 그걸
우선으로 읽어 `_build_plan_from_explicit_trajectory` 로 가므로 LBM 의 3-keyframe 합성기를
통째로 우회한다 (LBM 수정 0줄). 이게 "preset 대신 keyframe 을 입력으로 줄 수 있나"의 답이다.

**주의 3가지 (전부 실측).**

1. `frame` 은 **1-based** 여야 한다. `blender_render_worker.py:968` 이
   `int(kf.get("frame") or 1)` 이라 frame 0 을 1 로 삼키고, `:964 zero_based` 가 다시 +1 해서
   전체가 한 프레임씩 밀린다 (45도 원호에서 2.15 cm / 0.94도). 1-based 면 max|dloc| 7e-7.
2. `_smooth_executable_keyframes`(`:328`)가 `|end-start|` 를 **preset 별 상한**으로 균일
   축소한다 (`_trajectory_travel_limit:305`): pan_*/static_* 0.0, pedestal 0.35, *_reveal 0.42,
   orbit 0.8, s_curve/push/pull 0.85, straight_ease 0.95, truck 1.05, **PRESET_NAMES 에 없는
   이름 1.4**. 그래서 keyframe arm 의 `preset_name` 은 일부러 `keyframed_external` 로 둔다.
3. 성긴 keyframe 은 `_subdivide_rotation_keyframes`(`:475`, 임계 0.35 rad)가 slerp 로 채운다.
   49개를 다 주면 안 건드린다. `apply_trajectory_plan` 이 LINEAR 를 강제하는 것도 이 때문에 무관.

사용 예시:

    # handoff 만 만든다
    python viz/preset_render_demo.py --run trumans_c49_w01_f0000_0048 --demo_run trumans_presetdemo

    # 만들고 Stage 3 명령까지 출력
    python viz/preset_render_demo.py --run trumans_c49_w01_f0000_0048 --demo_run trumans_presetdemo --print_cmd
"""

from argparse import ArgumentParser
from copy import deepcopy
from os import path, makedirs
import json
import math

LBM = "/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/Look-Before-Move"
BLENDER = "/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender"

#    `build_trajectory_plan`(video_runtime.py:244-296)이 실제로 분기하는 모양 8개.
#    straight_ease 는 push_in_arc / pull_out_arc / truck_* 와 같은 branch(기본 lerp)라 대표 1개만.
SHAPE_PRESETS = [
    "straight_ease", "orbit_left_arc", "orbit_right_arc", "pedestal_up",
    "pedestal_down", "pan_left", "pan_right", "s_curve",
]


def look_at_euler(location, target):
    """`video_runtime._look_at_euler:140-146` 과 같은 식. mathutils 없이 순수 python.

    Blender 카메라는 -Z 를 보고 +Y 가 위다. 그 정렬의 XYZ 오일러를 직접 만든다.
    """
    dx, dy, dz = (target[i] - location[i] for i in range(3))
    #    -Z 를 (dx,dy,dz) 에 맞추는 건 pitch = angle from +Z, yaw = atan2(dy,dx)+90도.
    pitch = math.atan2(math.hypot(dx, dy), -dz)
    yaw = math.atan2(dy, dx) + math.pi / 2.0
    return [pitch, 0.0, yaw]


def degrees_for_travel(start, focus, travel):
    """xy 반경 `r` 원호에서 시작-끝 직선거리(현)가 `travel` 이 되는 각도.

    `chord = 2 r sin(θ/2)` 를 뒤집는다. `travel > 2r` 면 원 지름을 넘으므로 180도로 자른다.
    """
    radius = math.hypot(start[0] - focus[0], start[1] - focus[1])
    return math.degrees(2.0 * math.asin(min(travel / (2.0 * radius), 1.0)))


def orbit_keyframes(start, focus, degrees, frames):
    """focus 둘레를 `degrees` 만큼 도는 1-based keyframe `frames` 개. 매 프레임 look_at 재조준."""
    rx, ry = start[0] - focus[0], start[1] - focus[1]
    radius, angle0 = math.hypot(rx, ry), math.atan2(ry, rx)
    out = []
    for index in range(frames):
        angle = angle0 + math.radians(degrees) * index / max(frames - 1, 1)
        location = [focus[0] + radius * math.cos(angle),
                    focus[1] + radius * math.sin(angle),
                    start[2]]
        out.append({
            "frame": index + 1,                       # ← 1-based (§docstring 주의 1)
            "location": [round(v, 6) for v in location],
            "rotation_euler": [round(v, 6) for v in look_at_euler(location, focus)],
            "lens_mm": None,                          # main() 에서 base lens 로 채운다
        })
    return out


def scaled_end(base, factor):
    """`end_transform.location` 을 start 기준으로 `factor` 배 밀어낸 사본.

    preset 경로에서 이동량을 키우는 **유일한 입력**이 이것이다 — preset 이름은 상한만 준다
    (`_trajectory_travel_limit:304`). `rotation_euler` 는 지운다: `build_trajectory_plan:234` 가
    비어 있으면 새 끝점으로 `_look_at_euler` 를 다시 계산하므로, 안 지우면 옛 끝점의 회전이
    남아 조준이 어긋난다.
    """
    end = deepcopy(base["end_transform"])
    start = base["start_transform"]["location"]
    end["location"] = [start[i] + (end["location"][i] - start[i]) * factor for i in range(3)]
    end.pop("rotation_euler", None)
    return end


def directed_end(base, focus, distance, axis="forward"):
    """`end_transform` 을 **카메라 축 방향으로** 직접 놓는다.

    `build_trajectory_plan`(video_runtime.py:244-296)의 if/elif 는 `orbit_*`/`pedestal_*`/
    `*_reveal`/`pan_*`/`s_curve`/`static_*` 만 다룬다. `push_in_arc`/`pull_out_arc`/
    `straight_ease`/`truck_*` 는 **어느 분기에도 안 걸려** `mid=lerp(start,end,0.5)` +
    `end=end_transform` 인 순수 직선 LERP 가 된다 — 즉 이 5개 preset 의 **방향은 100%
    `end_transform` 이 정하고 이름은 `_trajectory_travel_limit` 상한만 준다**.
    그래서 push-in 을 보려면 끝점을 focus 쪽으로 직접 밀어야 한다.

    `axis="forward"` 는 focus 를 향하는 단위벡터(음수 distance 면 pull-out),
    `axis="right"` 는 `fwd x up`(Blender world +Z 가 up) — 같은 크기의 lateral 대조군용.
    `rotation_euler` 를 지우는 이유는 `scaled_end` 와 같다(:234 재조준).
    """
    end = deepcopy(base["end_transform"])
    start = base["start_transform"]["location"]
    span = math.dist(start, focus)
    forward = [(focus[i] - start[i]) / span for i in range(3)]
    if axis == "right":
        #    cross(fwd, (0,0,1)) = (fy, -fx, 0)
        vector = [forward[1], -forward[0], 0.0]
        norm = math.hypot(vector[0], vector[1])
        vector = [v / norm for v in vector]
    else:
        vector = forward
    end["location"] = [start[i] + vector[i] * distance for i in range(3)]
    end.pop("rotation_euler", None)
    return end


#    (preset, 축, 이동량 m). push_in_arc/pull_out_arc 상한 0.85, straight_ease 0.95, truck 1.05.
FORWARD_ARMS = [
    ("push_in_arc", "forward", 0.30),
    ("push_in_arc", "forward", 0.60),
    ("push_in_arc", "forward", 0.85),
    ("push_in_arc", "forward", 1.20),    # 상한 0.85 에 깎이는지 본다
    ("straight_ease", "forward", 0.95),  # 같은 전진을 더 높은 상한으로
    ("pull_out_arc", "forward", -0.85),  # 뒤로 (부호만 반대)
    ("truck_right", "right", 0.85),      # 같은 크기의 lateral 대조군
]


def make_camera(base, preset, keyframes, suffix, end_transform=None):
    """base 카메라를 preset(또는 명시 keyframe)만 바꿔 복제한다. start 는 손대지 않는다."""
    camera = deepcopy(base)
    if end_transform is not None:
        camera["end_transform"] = end_transform
    camera["camera_name"] = f"{base['camera_name']}__{suffix}"
    camera["trajectory_plan"] = {
        "schema_version": "plan_a.video_trajectory.v1",
        "preset_name": preset,
        "keyframes": keyframes,
        "selection_reason": "preset_render_demo",
    }
    #    남겨두면 `:1039` 가 아니라 `:1038` 이 이기지만, 읽는 사람이 오해한다.
    camera.pop("trajectory_keyframes", None)
    camera["demo_arm"] = suffix
    return camera


def main():
    parser = ArgumentParser()
    parser.add_argument("--run", required=True, type=str)        # 원본 Cinematographer run 이름
    parser.add_argument("--demo_run", default="trumans_presetdemo", type=str)
    parser.add_argument("--camera_index", default=0, type=int)   # shots[0].cameras 중 몇 번째를 베이스로
    # keyframe arm 이 도는 각도. 이동량 상한 1.4 m 안에 들어가야 안 깎인다 (§docstring 주의 2).
    parser.add_argument("--orbit_deg", default=45.0, type=float)
    parser.add_argument("--frames", default=49, type=int)
    # 이동량 사다리 모드. preset 8종 대신 **같은 궤적을 크기만 바꿔** 렌더해서 실제 천장을 잰다.
    parser.add_argument("--ladder", dest="ladder", action="store_true")
    parser.add_argument("--no_ladder", dest="ladder", action="store_false")
    parser.set_defaults(ladder=False)
    # 사다리 각 칸의 목표 이동량(m, 시작-끝 직선거리). keyframe arm 상한은 1.4 m 다.
    parser.add_argument("--travels", default="0.23,0.4,0.6,0.8,1.0,1.2,1.4,2.0", type=str)
    # 사다리 마지막 칸에 넣을 preset 대조군. end_transform 을 이 배수만큼 밀어낸다.
    parser.add_argument("--preset_end_scale", default=8.0, type=float)
    parser.add_argument("--ladder_preset", default="straight_ease", type=str)
    # 전진(push-in) 모드. end_transform 을 focus 방향으로 직접 놓아 `FORWARD_ARMS` 를 만든다.
    parser.add_argument("--forward", dest="forward", action="store_true")
    parser.add_argument("--no_forward", dest="forward", action="store_false")
    parser.set_defaults(forward=False)
    parser.add_argument("--print_cmd", dest="print_cmd", action="store_true")
    parser.add_argument("--no_print_cmd", dest="print_cmd", action="store_false")
    parser.set_defaults(print_cmd=False)
    args = parser.parse_args()

    src = path.join(LBM, "Cinematographer", "output", args.run, "outputs", "camera_handoff_v1.json")
    assert path.exists(src), f"원본 handoff 가 없다: {src}"
    data = json.load(open(src, encoding="utf-8"))
    base = data["shots"][0]["cameras"][args.camera_index]
    #    변종이 아니라 원본 카메라를 베이스로 써야 한다 — 변종은 끝점이 이미 갈려 있다.
    assert not base.get("preset_variant_source"), "베이스가 변종이다. --camera_index 를 0 으로."

    start = base["start_transform"]["location"]
    #    focus_center 는 카메라 dict 에 없다 (worker 가 씬에서 계산한다). 렌더 리포트에서 가져온다.
    report_path = path.join(LBM, "VideoEngineer", "output", args.run, "outputs",
                            "blender_render_report_v1.json")
    focus = None
    if path.exists(report_path):
        entries = json.load(open(report_path, encoding="utf-8")).get("render_report") or []
        for entry in entries:
            focus = (entry.get("trajectory_plan") or {}).get("focus_center")
            if focus:
                break
    assert focus, f"focus_center 를 못 찾았다. Stage 3 리포트가 필요하다: {report_path}"

    lens = float(base.get("lens_mm") or 35.0)

    def keyframe_arm(degrees):
        keyframes = orbit_keyframes(start, focus, degrees, args.frames)
        for keyframe in keyframes:
            keyframe["lens_mm"] = lens
        return keyframes

    if args.forward:
        cameras = []
        for index, (preset, axis, distance) in enumerate(FORWARD_ARMS):
            cameras.append(make_camera(
                base, preset, [], f"F{index:02d}_{preset}_{axis}{distance:+.2f}m",
                end_transform=directed_end(base, focus, distance, axis)))
        keyframes = []
    elif args.ladder:
        #    같은 원호를 크기만 바꿔 올린다. preset 이름은 전부 `keyframed_external` 로 고정 —
        #    상한(1.4 m)을 최대로 열어두고 **입력 크기만** 변수로 남긴다.
        targets = [float(v) for v in args.travels.split(",") if v.strip()]
        cameras = []
        for index, target in enumerate(targets):
            degrees = degrees_for_travel(start, focus, target)
            cameras.append(make_camera(base, "keyframed_external", keyframe_arm(degrees),
                                       f"L{index:02d}_kf{target:.2f}m"))
        #    대조군: preset 경로에서 end_transform 만 밀어냈을 때. 상한이 물리는 걸 본다.
        cameras.append(make_camera(base, args.ladder_preset, [],
                                   f"L{len(targets):02d}_{args.ladder_preset}_x"
                                   f"{args.preset_end_scale:g}",
                                   end_transform=scaled_end(base, args.preset_end_scale)))
        keyframes = cameras[0]["trajectory_plan"]["keyframes"]
    else:
        keyframes = keyframe_arm(args.orbit_deg)
        cameras = [make_camera(base, preset, [], f"p{index:02d}_{preset}")
                   for index, preset in enumerate(SHAPE_PRESETS)]
        cameras.append(make_camera(base, "keyframed_external", keyframes,
                                   f"p{len(SHAPE_PRESETS):02d}_ourkeyframes"))
    travel = math.dist(keyframes[0]["location"], keyframes[-1]["location"]) if keyframes else 0.0

    data["shots"] = [deepcopy(data["shots"][0])]
    data["shots"][0]["cameras"] = cameras
    data["cameras"] = deepcopy(cameras)
    data["run_id"] = args.demo_run

    out_dir = path.join(LBM, "Cinematographer", "output", args.demo_run, "outputs")
    makedirs(out_dir, exist_ok=True)
    out = path.join(out_dir, "camera_handoff_v1.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

    print(f"{'베이스 카메라':22s} {base['camera_name']}")
    print(f"{'start':22s} {[round(v, 4) for v in start]}")
    print(f"{'end (전 arm 공통)':22s} {[round(v, 4) for v in base['end_transform']['location']]}")
    print(f"{'focus_center':22s} {[round(v, 4) for v in focus]}")
    print(f"{'lens_mm':22s} {lens}")
    print(f"{'distance_label':22s} {base.get('distance_label')!r}  closeup {base.get('closeup_required')}")
    if args.forward:
        print(f"{'모드':22s} forward  arm {len(cameras)}개  (focus 까지 {math.dist(start, focus):.4f} m)")
        for camera in cameras:
            end = camera["end_transform"]["location"]
            request = math.dist(start, end)
            print(f"{'':22s} {camera['demo_arm']:30s} 요청 {request:5.3f} m  "
                  f"focus 거리 {math.dist(start, focus):.3f} -> {math.dist(end, focus):.3f}")
    elif args.ladder:
        print(f"{'모드':22s} ladder  arm {len(cameras)}개")
        for camera in cameras:
            plan = camera["trajectory_plan"]
            if plan["keyframes"]:
                request = math.dist(plan["keyframes"][0]["location"],
                                    plan["keyframes"][-1]["location"])
                note = "통과" if request <= 1.4 else "1.4 m 로 깎인다"
            else:
                request = math.dist(base["start_transform"]["location"],
                                    camera["end_transform"]["location"])
                note = f"preset 상한이 정한다 ({plan['preset_name']})"
            print(f"{'':22s} {camera['demo_arm']:24s} 요청 {request:6.3f} m  {note}")
    else:
        print(f"{'arm 수':22s} {len(cameras)}  (preset {len(SHAPE_PRESETS)} + keyframe 1)")
        print(f"{'keyframe arm':22s} {args.orbit_deg}도 원호 {args.frames}개, travel {travel:.4f} m "
              f"(상한 1.4 m {'통과' if travel < 1.4 else '초과 -> 깎인다'})")
    print(f"{'->':22s} {out}")

    if args.print_cmd:
        blend = json.load(open(path.join(LBM, "VideoEngineer", "output", args.run,
                                         "manifest.json"), encoding="utf-8"))["blend_file"]
        output_root = path.join(LBM, "VideoEngineer", "output", args.demo_run)
        print()
        print(" ".join([
            BLENDER, "-b", "--python-use-system-env", blend,
            "--python", path.join(LBM, "VideoEngineer", "blender_render_worker.py"),
            "--", "--camera-handoff-path", out, "--output-root", output_root,
            "--fps", "25", "--resolution-x", "960", "--resolution-y", "540",
            "--render-engine", "BLENDER_EEVEE_NEXT", "--render-samples", "8"]))


if __name__ == "__main__":
    main()
