"""LBM 이 렌더한 카메라 덤프 → `trumans_recon` 이 쓰는 `poses_a<NN>.npz` 규약.

**왜 필요한가.** LBM 을 TRUMANS 에 돌린 결과를 **Lite 뱅크와 같은 눈금으로** 재려면, LBM 카메라가
`audit_lite_framing.py` 가 읽는 형태(`cam_c2w (N,4,4)` OpenCV + `aim (N,3)`, TRUMANS Blender
world metre)로 있어야 한다. LBM 이 내놓는 건 `camera_packages/*/scene_N_shot_M_camK.json` 의
Blender `location` + `rotation_euler` 키프레임 3~5개뿐이고, 그 사이를 Blender 가 어떻게 채우는지는
`blender_render_worker.py` 의 slerp 재분할 + lens 램프 + fcurve easing 이 정한다. 그래서 키프레임을
numpy 로 다시 보간하면 **LBM 카메라가 아니라 그 근사치**를 평가하게 된다.

대신 `lbm_camera_dump_startup.py` 훅이 렌더 프로세스 안에서 프레임마다 받아 적은
`matrix_world` 를 읽는다 — 정의상 실제로 렌더된 카메라다.

**규약 변환 두 가지.**
1. Blender GL(X right / Y up / Z backward) → OpenCV(X right / Y down / Z forward):
   `c2w_cv = c2w_gl @ diag(1, -1, -1, 1)`. `trumans_gt_render.py:64 GL2CV` 와 같은 식이다.
   **재앵커(`cam_c2w[0] = I`)는 하지 않는다** — `trumans_to_recon.py` 의 `convert()` 가 depth 와
   짝을 맞춘 뒤 하류에서 하고, 여기서 미리 하면 Blender world 좌표를 잃어 사람 OBB track 과
   대조할 수 없게 된다 (`audit_lite_framing.py:222` 주석: "world (재앵커 전) — depth 와 짝이 맞는다").
2. 프레임 수. LBM 은 `target_frame_count`(movement 종류가 정하는 고정표) 만큼 렌더하므로 창
   길이와 무관하다 — w01 은 19프레임 창에 110프레임을 렌더했다. Lite 는 항상 49다. 여기서는
   **보간하지 않고 정수 step 으로 솎는다**: `step = max(1, round(N / num_frames))`. 보간은
   LBM 이 만든 easing 을 뭉개고, 그러면 지표 7번(jerk)이 실제보다 매끄럽게 나온다.

**시간축 매핑.** 덤프의 `frame` 은 LBM 이 1..count 로 덮어쓴 씬 프레임이다
(`blender_render_worker.py:1141-1142`). TRUMANS 원본 프레임으로 되돌리려면 창 오프셋을 빼면 된다:
`trumans_frame = scene_frame - TRUMANS_FRAME_OFFSET`. 오프셋은 demo root 의 `_window.json` 에
있다. 이 값이 있어야 GT 렌더가 **같은 애니메이션 순간**을 다시 그린다.

예시:
    python scripts/lbm_camera_to_poses.py \
        --dump_dir out/lbm_demos_w/_camdump/trumans_00add26c_w01_f0051_0069 \
        --window   out/lbm_demos_w/00add26c-..__w01_f0051_0069/_window.json \
        --out      out/trumans_recon_lbm/00add26c-.../poses_a00.npz
"""
import json
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

# Blender GL c2w -> OpenCV c2w. `trumans_gt_render.py:64 GL2CV` 와 같은 행렬이다.
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def read_dump(dump_dir):
    """`cam_*.jsonl` 전량 → {filepath: [record, ...]} (프레임 오름차순, 중복 프레임 제거).

    LBM 은 카메라마다 `scene.render.filepath` 를 `<...>/<camera_name>/frames/frame_` 로 놓으므로
    (`blender_render_worker.py:1173`) 그 문자열이 카메라를 가르는 유일한 키다. 같은 프로세스가
    여러 카메라를 연속 렌더해도 여기서 갈린다.
    """
    groups = {}
    for jsonl in sorted(glob(path.join(dump_dir, "cam_*.jsonl"))):
        with open(jsonl, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                groups.setdefault(record["filepath"], {})[int(record["frame"])] = record
    return {key: [value[f] for f in sorted(value)] for key, value in groups.items()}


def camera_name_of(filepath):
    """`<...>/scene_1_shot_1_cam1/frames/frame_` → `scene_1_shot_1_cam1`.

    끝에서부터 `frames` 를 찾아 그 부모를 쓴다. 못 찾으면 경로 마지막 디렉토리로 떨어진다.
    """
    parts = [p for p in filepath.replace("\\", "/").split("/") if p]
    if "frames" in parts:
        index = len(parts) - 1 - parts[::-1].index("frames")
        if index >= 1:
            return parts[index - 1]
    return parts[-2] if len(parts) >= 2 else filepath


def to_opencv(records):
    """덤프 레코드 목록 → (N,4,4) OpenCV c2w."""
    poses = np.zeros((len(records), 4, 4), dtype=np.float64)
    for i, record in enumerate(records):
        poses[i] = np.asarray(record["matrix_world"], dtype=np.float64) @ GL2CV
    determinants = np.linalg.det(poses[:, :3, :3])
    assert np.allclose(determinants, 1.0, atol=1e-6), \
        f"c2w 회전이 정규직교가 아니다: det 범위 {determinants.min():.6f}..{determinants.max():.6f}"
    return poses


def subsample(count, num_frames):
    """보간 없이 **인덱스만 골라** 솎는다. count <= num_frames 면 전량 그대로.

    고정 step(`arange(0, count, step)[:num_frames]`) 은 궤적 꼬리를 잘라먹는다 — w01 은 110프레임
    이라 step 2 로 0..96 만 남아 이동량이 0.323 m 대신 0.284 m (88%) 로 실측됐다. `linspace` 로
    양 끝을 고정하면 전 구간을 덮으면서 간격도 거의 균일하다. `canonical_cams.build_dl3dv:128` 의
    `np.rint(np.linspace(...))` 와 같은 관용구다.

    **보간은 여전히 안 한다.** 보간하면 LBM 이 얹은 fcurve easing 이 뭉개져 지표 7번(jerk)이
    실제보다 매끄럽게 나온다.
    """
    if count <= num_frames:
        return np.arange(count)
    return np.rint(np.linspace(0, count - 1, num_frames)).astype(int)


def aim_points(poses, target):
    """look-at 점 (N,3). LBM 은 per-frame target 을 안 남기므로 두 경로로 만든다.

    - camera package 에 `start_transform.target` 이 있으면 그 상수를 그대로 쓴다 (LBM 이 실제로
      겨눈 점이다).
    - 없으면 카메라 광축 위 |cam - target| 대신 **소스 카메라 거리 중앙값**을 못 쓰므로, 광축
      2 m 앞을 찍는다. `aim` 은 하류 지표에 안 들어가고 진단·시각화용이라 여기서 끝난다
      (`audit_lite_framing.py` 는 `cam_c2w` 만 읽는다).
    """
    if target is not None:
        return np.repeat(np.asarray(target, dtype=np.float64)[None, :], len(poses), axis=0)
    return poses[:, :3, 3] + 2.0 * poses[:, :3, 2]


def main(args):
    groups = read_dump(args.dump_dir)
    assert groups, f"카메라 덤프가 비었다: {args.dump_dir}/cam_*.jsonl"

    named = {camera_name_of(key): value for key, value in groups.items()}
    if args.camera:
        assert args.camera in named, f"{args.camera} 가 덤프에 없다. 있는 것: {sorted(named)}"
        camera_name, records = args.camera, named[args.camera]
    else:
        # 창당 카메라 1대가 정상이다. 여러 대면 프레임이 가장 많은 것을 집고 그 사실을 찍는다.
        camera_name = max(named, key=lambda k: len(named[k]))
        if len(named) > 1:
            print(f"  WARN 덤프에 카메라 {len(named)}대 — 최장 궤적 채택: {camera_name} "
                  f"({sorted((k, len(v)) for k, v in named.items())})")
        records = named[camera_name]

    poses_all = to_opencv(records)
    scene_frames_all = np.array([int(r["frame"]) for r in records], dtype=np.int64)
    lens_all = np.array([float(r["lens_mm"]) for r in records], dtype=np.float64)

    keep = subsample(len(records), int(args.num_frames))
    poses, scene_frames, lens = poses_all[keep], scene_frames_all[keep], lens_all[keep]

    # 씬 프레임 → TRUMANS 원본 프레임. 훅이 당긴 만큼 되돌린다.
    offset = int(args.frame_offset)
    if args.window:
        with open(args.window, encoding="utf-8") as file:
            offset = int(json.load(file)["frame_offset"])
    trumans_frames = scene_frames - offset

    target = None
    if args.camera_package and path.isfile(args.camera_package):
        with open(args.camera_package, encoding="utf-8") as file:
            package = json.load(file)
        target = (package.get("start_transform") or {}).get("target")
    aim = aim_points(poses, target)

    makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, cam_c2w=poses, aim=aim, lens=float(np.median(lens)),
             scene_frames=scene_frames, trumans_frames=trumans_frames)

    positions = poses[:, :3, 3]
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    print(f"{'camera':<22s}{camera_name}")
    print(f"{'dump frames':<22s}{len(records)}  (scene {scene_frames_all.min()}.."
          f"{scene_frames_all.max()})")
    steps = np.diff(keep) if len(keep) > 1 else np.array([1])
    print(f"{'kept':<22s}{len(keep)}  step {steps.min()}..{steps.max()}  "
          f"(dump idx {keep[0]}..{keep[-1]})")
    print(f"{'trumans frames':<22s}{trumans_frames.min()}..{trumans_frames.max()}")
    print(f"{'lens_mm':<22s}median {np.median(lens):.2f}  범위 {lens.min():.2f}..{lens.max():.2f}")
    print(f"{'path len (m)':<22s}{deltas.sum():.3f}   net {np.linalg.norm(positions[-1] - positions[0]):.3f}")
    print(f"{'dt median (m)':<22s}{np.median(deltas) if len(deltas) else 0.0:.4f}")
    print(f"{'out':<22s}{args.out}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True, type=str)        # cam_*.jsonl 이 있는 디렉토리
    parser.add_argument("--out", required=True, type=str)             # poses_a<NN>.npz 경로
    parser.add_argument("--camera", default="", type=str)             # 비우면 최장 궤적 자동 선택
    parser.add_argument("--camera_package", default="", type=str)     # aim(target) 을 여기서 읽는다
    parser.add_argument("--window", default="", type=str)             # demo root 의 _window.json
    parser.add_argument("--frame_offset", default=0, type=int)        # --window 가 없을 때만 쓴다
    parser.add_argument("--num_frames", default=49, type=int)         # Lite 코퍼스 고정 길이
    raise SystemExit(main(parser.parse_args()))
