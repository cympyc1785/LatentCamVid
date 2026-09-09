"""Cinematographer 산출물의 카메라 길이를 chunk 프레임 수에 맞춘다.

**왜 필요한가.** 우리 TRUMANS chunk 는 49프레임인데 Stage 2 가 낸 `target_frame_count` 는
110 / 80 / 95 였다. Director 가 준 `duration_target_seconds` 도 아니다 — Cinematographer 의
`refresh_camera_trajectory` (`Cinematographer/cinematographer_stage.py:2820-2828`) 가 duration 을
**preset 에서** 다시 계산하고 상류 값을 버리기 때문이다:

    duration_seconds = trajectory_duration_seconds(preset, travel_distance=..., closeup=...)
    frame_count = max(24, int(round(duration_seconds * int(fps or 24))))

그 값이 그대로 Stage 3 로 간다 (`VideoEngineer/blender_render_worker.py:1030`). 그리고 워커는
렌더 직전에

    scene.frame_start = 1
    scene.frame_end = frame_count            # blender_render_worker.py:1141-1142

로 frame shift 훅이 걸어둔 `scene.frame_end = 49`
(`_frame_shift/startup/trumans_frame_shift_startup.py:101`) 를 **덮어쓴다**. 훅은 Blender 기동
시 한 번 돌고 워커는 그 뒤에 세팅하므로 워커가 이긴다. 결과적으로 w01 은 chunk 49프레임 +
**다음 chunk 영역 61프레임**을 이어서 렌더한다. 클램프도 루프도 아니라 그냥 chunk 밖 애니메이션이
섞인다.

**왜 `target_frame_count` 만 고치면 안 되는가.** 궤적 keyframe 이 절대 프레임 인덱스로 박혀
있다. `trajectory_plan.keyframes[].frame` 이 1 / 56 / 110 인데 `frame_count` 만 49 로 낮추면
`_build_plan_from_explicit_trajectory` 의

    frame_number = max(1, min(frame_number, frame_count))      # :1000 근처

가 1 / 49 / 49 로 클램프해서 **궤적 앞 절반만 재생하고 나머지는 정지**한다. 그래서 프레임 인덱스도
같이 다시 매겨야 한다. 두 군데다 — `trajectory_keyframes` (0-based, 정규화 `t` 동봉) 와
`trajectory_plan.keyframes` (1-based).

**시간을 압축하지 왜 궤적을 자르지 않는가.** 자르면(=원래 속도 유지, 49프레임 뒤에서 끊기)
orbit_right_arc 이 sweep 의 45% 만 돌고 끝난다 — preset 을 고른 이유 자체가 사라진다. 압축은
authored 구도를 통째로 유지하고 속도만 `old_fc/new_fc` 배 빨라진다. 실측 w01 은 0.231 m 를
4.4 s → 1.96 s 로, 0.053 → 0.118 m/s 다 (요약표의 `m/s` 열로 매 실행 확인할 것).

**손대지 않는 것.** `shot_contract.motion_contract.{start_frame,end_frame}` 은 Director 의 계약
프레임 공간이고 VideoEngineer 가 읽지 않는다 (`grep motion_contract VideoEngineer/*.py` → 0건).
`_smooth_executable_keyframes` 의 travel limit 은 거리 기준이라 프레임 수와 무관하다.

사용 예시:

    # 무엇이 바뀌는지만 보고 파일은 안 고친다
    python scripts/retime_camera_handoff.py \\
      --output_root ../Look-Before-Move/Cinematographer/output \\
      --run_glob 'trumans_c49_w*' --frame_count 49 --fps 25 --dry_run

    # 실제로 고친다 (원본은 <파일>.bak 로 남는다)
    python scripts/retime_camera_handoff.py \\
      --output_root ../Look-Before-Move/Cinematographer/output \\
      --run_glob 'trumans_c49_w*' --frame_count 49 --fps 25 --no_dry_run
"""

from argparse import ArgumentParser
from glob import glob
from math import hypot
from os import path
from shutil import copyfile
import json

#    Stage 3 이 실제로 읽는 파일은 이것 하나다 (`VideoEngineer/video_stage.py:46`,
#    `Engine/run_full_pipeline.py:315`). preview/quality input 사본은 Cinematographer 내부
#    중간산출물이라 건드리지 않는다 — 고쳐봐야 하류가 안 읽고, 원본 대조용으로 남겨두는 게 낫다.
HANDOFF_NAME = "camera_handoff_v1.json"


def monotonic(frames, lo: int, hi: int):
    """프레임 인덱스를 [lo, hi] 안에서 **순증가**하도록 민다.

    keyframe 수가 프레임 수보다 많으면 뒤쪽이 hi 로 뭉친다. 실측 3 chunk 는 keyframe 2~3개라
    일어나지 않지만, 뭉치면 요약표의 `collapsed` 열에 찍어 사람이 보게 한다.
    """
    out, prev = [], lo - 1
    for frame in frames:
        value = max(lo, min(hi, int(frame)))
        value = min(max(value, prev + 1), hi)
        out.append(value)
        prev = value
    return out


def path_length(keyframes):
    """keyframe location 을 이은 꺾은선 길이. 속도 보고용이라 회전은 안 본다."""
    total = 0.0
    previous = None
    for keyframe in keyframes:
        location = keyframe.get("location") or (keyframe.get("transform") or {}).get("location")
        if not location or len(location) < 3:
            continue
        if previous is not None:
            total += hypot(hypot(location[0] - previous[0], location[1] - previous[1]),
                           location[2] - previous[2])
        previous = location
    return total


def retime_camera(camera: dict, new_count: int, fps: int):
    """카메라 하나를 `new_count` 프레임으로 다시 매긴다. 제자리에서 고치고 요약 dict 를 돌려준다.

    이미 `new_count` 면 t 재계산이 항등이라 결과가 같다 — **여러 번 돌려도 안전**하다.
    """
    old_count = int(camera.get("target_frame_count") or 0)
    if old_count <= 1:
        return None
    duration = round(new_count / float(fps), 3)
    collapsed = 0

    camera["target_frame_count"] = int(new_count)
    camera["target_duration_seconds"] = duration

    #    ① trajectory_keyframes — 0-based (0 .. old_count-1). 정규화 `t` 가 같이 오므로 그걸 쓴다.
    #       `t` 가 없거나 범위 밖이면 프레임 인덱스에서 되돌린다.
    upstream = camera.get("trajectory_keyframes")
    if isinstance(upstream, list) and upstream:
        ratios = []
        for keyframe in upstream:
            ratio = keyframe.get("t")
            if not isinstance(ratio, (int, float)) or not 0.0 <= float(ratio) <= 1.0:
                ratio = int(keyframe.get("frame") or 0) / max(1, old_count - 1)
            ratios.append(max(0.0, min(1.0, float(ratio))))
        frames = monotonic([round(r * (new_count - 1)) for r in ratios], 0, new_count - 1)
        collapsed += len(frames) - len(set(frames))
        for keyframe, ratio, frame in zip(upstream, ratios, frames):
            keyframe["frame"] = int(frame)
            keyframe["t"] = round(ratio, 6)

    #    ② trajectory_plan.keyframes — 1-based (start_frame .. end_frame). `t` 가 없어서
    #       plan 이 신고한 span 으로 정규화한다.
    plan = camera.get("trajectory_plan")
    before_length = after_length = 0.0
    before_seconds = float(camera.get("target_duration_seconds") or 0.0)
    if isinstance(plan, dict):
        before_seconds = float(plan.get("duration_seconds") or old_count / float(fps))
        plan_keyframes = plan.get("keyframes")
        if isinstance(plan_keyframes, list) and plan_keyframes:
            before_length = after_length = path_length(plan_keyframes)
            start = int(plan.get("start_frame") or 1)
            span = max(1, int(plan.get("end_frame") or old_count) - start)
            ratios = [max(0.0, min(1.0, (int(k.get("frame") or start) - start) / span))
                      for k in plan_keyframes]
            frames = monotonic([round(r * (new_count - 1)) + 1 for r in ratios], 1, new_count)
            collapsed += len(frames) - len(set(frames))
            for keyframe, frame in zip(plan_keyframes, frames):
                keyframe["frame"] = int(frame)
        plan["start_frame"] = 1
        plan["end_frame"] = int(new_count)
        plan["duration_seconds"] = duration
        safety = plan.get("safety_report")
        if isinstance(safety, dict):
            safety["duration_seconds"] = duration

    safety = camera.get("trajectory_safety_report")
    if isinstance(safety, dict):
        safety["duration_seconds"] = duration

    return {
        "camera_name": str(camera.get("camera_name") or ""),
        "preset": str((plan or {}).get("preset_name") or ""),
        "old_count": old_count,
        "new_count": int(new_count),
        "travel_m": before_length,
        #    같은 거리를 더 짧은 시간에 간다. 이 값이 눈에 띄게 크면 압축 대신 preset 을
        #    다시 골라야 한다는 신호다.
        "speed_before": before_length / max(before_seconds, 1e-6),
        "speed_after": after_length / max(duration, 1e-6),
        "collapsed": collapsed,
    }


def walk(node, new_count: int, fps: int, rows: list):
    """JSON 트리에서 카메라 dict 를 찾아 전부 다시 매긴다.

    `camera_handoff_v1.json` 은 같은 카메라를 `/shots[i]/cameras[j]` 와 `/cameras[k]` 두 군데에
    **복사본으로** 들고 있다 (json.load 후에는 서로 다른 객체다). 그래서 키로 찾아 전부 훑는다.
    """
    if isinstance(node, dict):
        if "target_frame_count" in node and "camera_name" in node:
            row = retime_camera(node, new_count, fps)
            if row:
                rows.append(row)
            return
        for value in node.values():
            walk(value, new_count, fps, rows)
    elif isinstance(node, list):
        for item in node:
            walk(item, new_count, fps, rows)


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_root", required=True, type=str)   # Cinematographer/output
    parser.add_argument("--run_glob", default="trumans_c49_w*", type=str)
    parser.add_argument("--frame_count", default=49, type=int)      # chunk 프레임 수
    parser.add_argument("--fps", default=25, type=int)              # TRUMANS 씬 fps
    # 기본은 dry run — 원본을 고치려면 `--no_dry_run` 을 명시해야 한다.
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--backup_suffix", default=".bak", type=str)
    args = parser.parse_args()

    runs = sorted(glob(path.join(args.output_root, args.run_glob)))
    all_rows, touched, already = [], [], 0
    for run in runs:
        handoff = path.join(run, "outputs", HANDOFF_NAME)
        if not path.exists(handoff):
            continue
        data = json.load(open(handoff, encoding="utf-8"))
        rows = []
        walk(data, args.frame_count, args.fps, rows)
        if not rows:
            continue
        #    `/shots` 사본과 `/cameras` 사본이 같은 카메라라 두 번 잡힌다. 요약은 이름으로 접는다.
        unique = {row["camera_name"]: row for row in rows}
        for row in unique.values():
            row["run"] = path.basename(run)
        if all(row["old_count"] == args.frame_count for row in unique.values()):
            already += 1
            continue
        all_rows.extend(unique.values())
        touched.append(handoff)
        if args.dry_run:
            continue
        backup = handoff + args.backup_suffix
        if not path.exists(backup):
            copyfile(handoff, backup)
        with open(handoff, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    print(f"{'runs':22s} {len(runs)}")
    print(f"{'dry_run':22s} {args.dry_run}")
    print(f"{'target frame_count':22s} {args.frame_count} @ {args.fps} fps "
          f"({args.frame_count / args.fps:.2f} s)")
    print(f"{'고친 파일':22s} {len(touched)}")
    print(f"{'이미 맞음 (건너뜀)':22s} {already}")
    collapsed = sum(row["collapsed"] for row in all_rows)
    if collapsed:
        print(f"{'!! keyframe 뭉침':22s} {collapsed}  (프레임 수보다 keyframe 이 많다 — 확인 필요)")
    if all_rows:
        print()
        print(f"  {'run':26s} {'preset':18s} {'frames':>12s} {'travel_m':>9s} "
              f"{'m/s before':>11s} {'m/s after':>10s}")
        for row in sorted(all_rows, key=lambda item: item["run"]):
            print(f"  {row['run'][:26]:26s} {row['preset'][:18]:18s} "
                  f"{row['old_count']:5d} -> {row['new_count']:3d} "
                  f"{row['travel_m']:9.3f} {row['speed_before']:11.3f} {row['speed_after']:10.3f}")
        speeds = sorted(row["speed_after"] for row in all_rows)
        print()
        print(f"{'속도 after min/med/max':22s} {speeds[0]:.3f} / "
              f"{speeds[len(speeds) // 2]:.3f} / {speeds[-1]:.3f}  m/s")


if __name__ == "__main__":
    main()
