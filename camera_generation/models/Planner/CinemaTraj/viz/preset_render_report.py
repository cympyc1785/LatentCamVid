"""`preset_render_demo.py` 가 만든 통제 실험 렌더를 **실측**해서 표 + 타일 영상으로 낸다.

**무엇을 재는가.** start/end/focus/lens 를 전 arm 공통으로 못박았으므로, arm 사이의 차이는
전부 `trajectory_plan.preset_name` 이 만든 것이다. 그 차이를 세 층위에서 잰다:

1. **카메라** — camdump(`lbm_camera_dump_startup.py`)가 받아 적은 `matrix_world` 로 프레임별
   위치·시선을 뽑아, 기준 arm(`straight_ease`) 대비 `max|Δloc|` (m) 과 `max Δ시선각` (deg).
   plan 키프레임이 아니라 **렌더된 카메라**라야 `_smooth_executable_keyframes` 축소와
   가시성 폴백(`blender_render_worker.py:1094` motion_scale 사다리)까지 반영된다.
2. **픽셀** — 같은 프레임 인덱스끼리 mean|ΔRGB| (0~255). 카메라가 달라도 화면이 안 달라지면
   그 preset 은 실질적으로 같은 클립이다.
3. **keyframe arm 왕복** — 우리가 실은 49 keyframe 과 렌더된 카메라의 오차. 이게 작아야
   "LBM 에 임의 궤적을 입력으로 줄 수 있다"가 성립한다.

영상은 3x3 타일로 붙인다 (`imageio` + libx264 — cv2 mp4v 는 뷰어에서 안 열린다).

사용 예시:

    python viz/preset_render_report.py --demo_run trumans_presetdemo \
        --camdump out/preset_demo/_camdump --out out/preset_demo
"""

from argparse import ArgumentParser
from os import path, makedirs, listdir
import json
import math

import numpy as np

LBM = "/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/Look-Before-Move"


def load_camdump(camdump_dir):
    """`cam_<pid>.jsonl` 들을 읽어 `{arm: {frame: record}}`. arm 은 `filepath` 의 부모 디렉토리."""
    tracks = {}
    for name in sorted(listdir(camdump_dir)):
        if not name.endswith(".jsonl"):
            continue
        for line in open(path.join(camdump_dir, name), encoding="utf-8"):
            record = json.loads(line)
            #    `<...>/renders/<shot>/<camera_name>/frames/frame_` → `frames` 바로 앞이 arm.
            parts = [p for p in str(record["filepath"]).split("/") if p]
            arm = parts[parts.index("frames") - 1] if "frames" in parts else parts[-1]
            tracks.setdefault(arm, {})[int(record["frame"])] = record
    return tracks


def track_arrays(track):
    """`{frame: record}` → (frames, loc(N,3), fwd(N,3)). Blender GL 이라 시선은 -Z 열."""
    frames = sorted(track)
    matrices = np.array([track[f]["matrix_world"] for f in frames], dtype=np.float64)
    return np.array(frames), matrices[:, :3, 3], -matrices[:, :3, 2]


def angle_deg(a, b):
    """행 단위 두 방향벡터 사이 각(도)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def frame_paths(render_root, arm):
    """`renders/<shot>/<arm>/frames/*.png`. shot 디렉토리 이름은 모르므로 한 단계 훑는다."""
    root = path.join(render_root, "renders")
    if not path.isdir(root):
        return []
    for shot in sorted(listdir(root)):
        directory = path.join(root, shot, arm, "frames")
        if path.isdir(directory):
            return [path.join(directory, n) for n in sorted(listdir(directory))
                    if n.endswith(".png")]
    return []


def label_tile(image, text):
    """타일 좌상단에 검은 띠 + 흰 글씨. PIL 폰트가 없어도 죽지 않게 try 로 감싼다."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return image
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 22], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(canvas)


def main():
    parser = ArgumentParser()
    parser.add_argument("--demo_run", default="trumans_presetdemo", type=str)
    parser.add_argument("--camdump", required=True, type=str)     # `_camdump` 디렉토리
    parser.add_argument("--out", required=True, type=str)         # 표/영상 저장 위치
    parser.add_argument("--tile_w", default=320, type=int)        # 타일 가로 (3x3 이라 x3 이 최종 폭)
    parser.add_argument("--fps", default=12.5, type=float)
    parser.add_argument("--video", dest="video", action="store_true")
    parser.add_argument("--no_video", dest="video", action="store_false")
    parser.set_defaults(video=True)
    args = parser.parse_args()

    render_root = path.join(LBM, "VideoEngineer", "output", args.demo_run)
    handoff = json.load(open(path.join(LBM, "Cinematographer", "output", args.demo_run,
                                       "outputs", "camera_handoff_v1.json"), encoding="utf-8"))
    cameras = handoff["shots"][0]["cameras"]
    arms = [c["camera_name"] for c in cameras]
    tracks = load_camdump(args.camdump)
    makedirs(args.out, exist_ok=True)

    #    기준 arm. 나머지는 전부 이것과 비교한다.
    base_arm = arms[0]
    assert base_arm in tracks, f"기준 arm 덤프가 없다: {base_arm} / 있는 것 {sorted(tracks)}"
    _, base_loc, base_fwd = track_arrays(tracks[base_arm])

    #    렌더 리포트에서 arm 별 "요청 대비 실제" 흔적을 꺼낸다. 두 축소 경로가 서로 다른 곳에
    #    기록된다: travel limit 은 `motion_speed_policy`, 가시성 가드는 `safety_report` 다.
    plans = {}
    report_path = path.join(render_root, "outputs", "blender_render_report_v1.json")
    if path.exists(report_path):
        for entry in json.load(open(report_path, encoding="utf-8"))["render_report"]:
            plan = entry.get("trajectory_plan") or {}
            safety = plan.get("safety_report") or {}
            policy = safety.get("motion_speed_policy") or {}
            plans[str(entry.get("camera_name"))] = {
                "guard": safety.get("visibility_guard_motion_scale"),
                "limited": bool(policy.get("speed_limited")),
                "req_m": policy.get("original_travel_distance"),
                "cap_m": policy.get("max_travel"),
            }

    rows = []
    for camera in cameras:
        arm = camera["camera_name"]
        if arm not in tracks:
            rows.append({"arm": arm, "missing": True})
            continue
        frames, loc, fwd = track_arrays(tracks[arm])
        n = min(len(loc), len(base_loc))
        images = frame_paths(render_root, arm)
        rows.append({
            "arm": arm,
            "preset": camera["trajectory_plan"]["preset_name"],
            "n_frames": len(frames),
            "n_png": len(images),
            #    이 arm 자신의 움직임
            "net_m": float(np.linalg.norm(loc[-1] - loc[0])),
            "path_m": float(np.linalg.norm(np.diff(loc, axis=0), axis=1).sum()),
            "rot_deg": float(angle_deg(fwd[0], fwd[-1])),
            #    기준 arm 대비 차이
            "dloc_max_m": float(np.abs(loc[:n] - base_loc[:n]).max()),
            "dang_max_deg": float(angle_deg(fwd[:n], base_fwd[:n]).max()),
            **plans.get(arm, {}),
        })

    #    픽셀 차이 — 카메라가 달라도 화면이 같으면 실질적으로 같은 클립이다.
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    base_images = frame_paths(render_root, base_arm)
    if imageio is not None and base_images:
        base_stack = np.stack([imageio.imread(p)[..., :3].astype(np.float32) for p in base_images])
        for row in rows:
            if row.get("missing"):
                continue
            images = frame_paths(render_root, row["arm"])
            if not images:
                continue
            stack = np.stack([imageio.imread(p)[..., :3].astype(np.float32) for p in images])
            n = min(len(stack), len(base_stack))
            row["px_mean_abs"] = float(np.abs(stack[:n] - base_stack[:n]).mean())
            row["px_frac_gt8"] = float((np.abs(stack[:n] - base_stack[:n]).max(-1) > 8).mean())

    #    keyframe arm 왕복: 입력 keyframe vs 렌더된 카메라.
    keyframe_camera = next((c for c in cameras if c["trajectory_plan"]["keyframes"]), None)
    roundtrip = None
    if keyframe_camera and keyframe_camera["camera_name"] in tracks:
        track = tracks[keyframe_camera["camera_name"]]
        errors_loc, errors_ang = [], []
        for keyframe in keyframe_camera["trajectory_plan"]["keyframes"]:
            record = track.get(int(keyframe["frame"]))
            if record is None:
                continue
            matrix = np.array(record["matrix_world"], dtype=np.float64)
            errors_loc.append(np.abs(matrix[:3, 3] - np.array(keyframe["location"])).max())
            #    입력은 오일러라 회전은 렌더된 시선 vs 입력 오일러가 만드는 시선으로 비교한다.
            #    부호는 추측이 아니라 camdump 로 맞췄다 — 이 식으로 최종 plan 키프레임 vs
            #    렌더된 `matrix_world` 가 frame 1/25/49 에서 전부 0.000°.
            pitch, _, yaw = keyframe["rotation_euler"]
            want = np.array([math.sin(pitch) * math.sin(yaw), -math.sin(pitch) * math.cos(yaw),
                             -math.cos(pitch)])
            errors_ang.append(float(angle_deg(-matrix[:3, 2], want)))
        #    렌더 리포트가 가시성 가드로 플랜을 갈아치웠는지. 이게 켜지면 왕복이 안 닫힌다
        #    (`blender_render_worker.py:1094` motion_scale 사다리).
        guard = None
        report_path = path.join(render_root, "outputs", "blender_render_report_v1.json")
        if path.exists(report_path):
            for entry in json.load(open(report_path, encoding="utf-8"))["render_report"]:
                plan = entry.get("trajectory_plan") or {}
                if plan.get("preset_name") == keyframe_camera["trajectory_plan"]["preset_name"]:
                    guard = (plan.get("safety_report") or {}).get("visibility_guard_motion_scale")
        roundtrip = {"n": len(errors_loc), "max_dloc_m": max(errors_loc),
                     "max_dang_deg": max(errors_ang), "visibility_guard_motion_scale": guard}

    json.dump({"rows": rows, "roundtrip": roundtrip, "base_arm": base_arm},
              open(path.join(args.out, "preset_render_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    label_width = max(len(str(c.get("demo_arm") or c["camera_name"])) for c in cameras)
    header = (f"{'arm':{label_width}s} {'req_m':>6s} {'cap_m':>6s} {'guard':>5s} | "
              f"{'net_m':>7s} {'path_m':>7s} {'rot°':>6s} | "
              f"{'Δloc_max':>9s} {'Δang_max':>9s} | {'px|Δ|':>7s} {'px>8':>6s}")
    print(header)
    print("-" * len(header))
    for camera, row in zip(cameras, rows):
        label = str(camera.get("demo_arm") or camera["camera_name"])
        if row.get("missing"):
            print(f"{label:{label_width}s}  (덤프 없음)")
            continue
        show = lambda value, fmt: (format(value, fmt) if isinstance(value, (int, float))
                                   else "-")
        print(f"{label:{label_width}s} {show(row.get('req_m'), '6.3f'):>6s} "
              f"{show(row.get('cap_m'), '6.2f'):>6s} {show(row.get('guard'), '5.2f'):>5s} | "
              f"{row['net_m']:7.4f} {row['path_m']:7.4f} {row['rot_deg']:6.2f} | "
              f"{row['dloc_max_m']:9.4f} {row['dang_max_deg']:9.3f} | "
              f"{row.get('px_mean_abs', float('nan')):7.3f} {row.get('px_frac_gt8', float('nan')):6.3f}")
    print(f"\n기준 arm = {base_arm} (Δ 열은 전부 이것 대비)")
    if roundtrip:
        print(f"keyframe arm 왕복: n={roundtrip['n']}  max|Δloc| {roundtrip['max_dloc_m']:.2e} m  "
              f"max Δang {roundtrip['max_dang_deg']:.4f}°  "
              f"visibility_guard_motion_scale={roundtrip['visibility_guard_motion_scale']}")

    if args.video and imageio is not None:
        from PIL import Image
        columns = 3
        rows_n = int(math.ceil(len(arms) / columns))
        stacks = []
        for camera in cameras:
            images = frame_paths(render_root, camera["camera_name"])
            tiles = [np.asarray(Image.open(p).convert("RGB").resize(
                (args.tile_w, args.tile_w * 9 // 16))) for p in images]
            #    ladder 모드는 preset 이름이 전 arm 동일하므로 `demo_arm` 이 있으면 그걸 쓴다.
            caption = str(camera.get("demo_arm") or camera["trajectory_plan"]["preset_name"])
            tiles = [label_tile(t, caption) for t in tiles]
            stacks.append(tiles)
        length = min(len(s) for s in stacks if s)
        height, width = stacks[0][0].shape[:2]
        frames = []
        for index in range(length):
            grid = np.zeros((rows_n * height, columns * width, 3), np.uint8)
            for cell, tiles in enumerate(stacks):
                if not tiles:
                    continue
                r, c = divmod(cell, columns)
                grid[r * height:(r + 1) * height, c * width:(c + 1) * width] = tiles[index]
            frames.append(grid)
        video = path.join(args.out, "preset_tiles.mp4")
        imageio.mimwrite(video, frames, fps=args.fps, codec="libx264", quality=6,
                         macro_block_size=1)
        print(f"{'영상':20s} {video}  ({length} 프레임, {columns}x{rows_n})")


if __name__ == "__main__":
    main()
