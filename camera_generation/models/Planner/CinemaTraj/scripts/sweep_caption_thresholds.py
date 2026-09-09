"""Vista 카메라용으로 DataDoP 분절기 임계를 고르기 위한 sweep (LLM 호출 없음).

WHY: DataDoP 기본값은 **120 pose @ fps 30** 눈금에 맞춰진 값이다. 우리 Vista 카메라는
     **49 프레임 @ fps 10** 이라 같은 궤적이라도 `t_velocities = fps * Δt` (segmentation.py:92)
     가 통째로 다른 크기가 된다. 임계를 눈대중으로 옮기지 말고, preset 이름이 정답인 몇 개
     엔트리에서 config 별 outline 을 직접 찍어 보고 고른다.

읽을 때 알아야 할 것 (전부 `processing/segmentation.py` 실측):
  * `segment_rigidbody_trajectories` 는 넘긴 `smoothing_window_size` / `min_chunk_size` 를
    :391-393 에서 **15 / 10 으로 덮어쓴다.** 인자는 translation 분절(`perform_segmentation`)
    에만 먹는다. 그래서 `--min_chunk_size` 는 사실상 무효고, 아래 표에도 안 넣는다.
  * 그 뒤 while 루프가 세그먼트 수가 4 이하가 될 때까지 window/min_chunk 를 5씩 키운다.
    프레임 수가 줄면(120 -> 49) min_chunk 10 이 클립의 20% 를 먹는다.
  * fps 는 translation 임계에만 작용한다. angular 쪽 `to_euler_angles` 는 fps 를 안 쓴다.

사용 예시:
    conda run -n GenDoP python scripts/sweep_caption_thresholds.py
    conda run -n GenDoP python scripts/sweep_caption_thresholds.py --only camel/zoom-out \
        --configs native10_t016
"""
from argparse import ArgumentParser
from os import path
import sys

import numpy as np

sys.path.insert(0, path.dirname(path.abspath(__file__)))
from caption_cameras_datadop import (                                 # noqa: E402
    SEG_DEFAULTS, opencv_c2w_to_datadop, resample_poses, tag_trajectory,
)

ROOT = "/data1/cympyc1785/LatentCamVid/DATA/Vista4D-Eval-Data/eval_data"

# preset 이름 = 사람이 붙인 정답. 축이 명확한 것만 고른다 (zoom 계열은 "정지" 가 정답).
PROBES = [
    ("cameras", "camel",          "zoom-out"),      # 위치 고정 + focal ramp -> static 이 정답
    ("cameras", "camel",          "three-in-out"),  # 전후 왕복
    ("cameras", "bed-shopping",   "dolly-out"),
    ("cameras", "room-argue",     "pedestal-up"),
    ("cameras", "basketball-four", "right-left"),
    ("cameras", "swing",          "arc-left"),
    ("cameras", "car-roundabout", "side-truck"),
    ("recon",   "camel",          "source"),        # recon 소스 카메라 (합성 아님)
]

# name -> outline 에 있어야 하는 단어 (없으면 '' = 검사 안 함)
EXPECT = {"zoom-out": "static", "dolly-out": "backward", "pedestal-up": "up",
          "right-left": "left", "arc-left": "", "side-truck": "", "three-in-out": "",
          "source": ""}

CONFIGS = {
    # 이름:            (num_poses, fps,  static_thr, diff_thr, ang_thr, smooth_w)
    "datadop":         (120, 30.0, 0.02,  0.4, 0.005, 18),   # 현재 디스크에 있는 결과
    "resample120_f10": (120, 10.0, 0.02,  0.4, 0.005, 18),   # fps 만 10
    "native_f10":      (0,   10.0, 0.02,  0.4, 0.005, 18),   # 리샘플 되돌림 + fps 10
    "native_f10_w7":   (0,   10.0, 0.02,  0.4, 0.005, 7),    # 49프레임에 맞게 window 축소
    "native_f10_t008": (0,   10.0, 0.008, 0.4, 0.005, 7),    # static 임계 완화
    "native_f10_t004": (0,   10.0, 0.004, 0.4, 0.005, 7),
}


def npz_path_of(kind, scene, name):
    if kind == "recon":
        return path.join(ROOT, "recon_and_seg", scene, "cameras.npz")
    return path.join(ROOT, "cameras", scene, f"{name}.npz")


def compact(tag):
    """outline 을 한 줄로: 'forward+static(0-47)' 처럼."""
    parts = []
    for chunk in tag["chunks"]:
        move = chunk["move"]
        if chunk["angular"] != "static":
            move += "+" + chunk["angular"]
        parts.append(f"{move}({chunk['start']}-{chunk['end']})")
    return " | ".join(parts)


def velocity_stats(cam_c2w, num_poses, fps):
    """`compute_camera_dynamics` 와 같은 양 — 임계를 어디에 둘지 눈으로 보려고."""
    poses = resample_poses(opencv_c2w_to_datadop(cam_c2w), num_poses)
    rel = np.linalg.inv(poses[:-1]) @ poses[1:]
    speed = fps * np.abs(rel[:, :3, 3])          # 축별 속도 (임계와 같은 축별 비교)
    return np.percentile(speed.max(axis=1), [50, 90]), poses.shape[0]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", default=None)          # "scene/name" 필터
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = parser.parse_args()

    probes = PROBES
    if args.only:
        wanted = set(args.only)
        probes = [p for p in probes if f"{p[1]}/{p[2]}" in wanted or p[1] in wanted]

    rows = []
    for kind, scene, name in probes:
        data = np.load(npz_path_of(kind, scene, name))
        cam = data["cam_c2w"]
        focal = data["intrinsics"][:, 0] if "intrinsics" in data.files else None
        ratio = float(focal[-1] / focal[0]) if focal is not None else 1.0
        for key in args.configs:
            num_poses, fps, static_thr, diff_thr, ang_thr, smooth_w = CONFIGS[key]
            seg = dict(SEG_DEFAULTS, fps=fps, cam_static_threshold=static_thr,
                       cam_diff_threshold=diff_thr, angular_static_threshold=ang_thr,
                       smoothing_window_size=smooth_w)
            tag = tag_trajectory(cam, num_poses, seg)
            (v50, v90), n = velocity_stats(cam, num_poses, fps)
            rows.append(dict(entry=f"{scene}/{name}", config=key, n=n,
                             v50=v50, v90=v90, thr=static_thr, focal=ratio,
                             nseg=len(tag["chunks"]), outline=compact(tag),
                             expect=EXPECT.get(name, "")))

    header = (f"{'entry':<26}{'config':<18}{'n':>4}{'v50':>8}{'v90':>8}{'thr':>7}"
              f"{'seg':>4}  outline")
    print(header)
    print("-" * len(header))
    last = None
    for row in rows:
        if last is not None and row["entry"] != last:
            print()
        last = row["entry"]
        mark = ""
        if row["expect"]:
            mark = "  OK" if row["expect"] in row["outline"] else f"  <-- want {row['expect']}"
        print(f"{row['entry']:<26}{row['config']:<18}{row['n']:>4}{row['v50']:>8.4f}"
              f"{row['v90']:>8.4f}{row['thr']:>7.3f}{row['nseg']:>4}  "
              f"{row['outline'][:70]}{mark}")

    print()
    print(f"{'v50/v90':<22}축별 |Δt|·fps 의 프레임간 최대값, median / p90")
    print(f"{'thr':<22}cam_static_threshold — 이 아래면 그 축은 static")
    print(f"{'주의':<22}min_chunk_size 는 segmentation.py:392 가 10 으로 덮어써서 무효")
    print(f"{'주의':<22}세그먼트가 5개 이상이면 while 루프가 window 를 5씩 키워 다시 뭉갠다")


if __name__ == "__main__":
    main()
