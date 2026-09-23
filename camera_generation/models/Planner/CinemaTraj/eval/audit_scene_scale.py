"""씬 길이 단위 `S` 의 **정의 후보들을 같은 영상에서 나란히 재는** 감사 스크립트.

왜 필요한가 (사용자 지적 2026-09-02): "scene scale 은 카메라가 돌아서 다른 공간을 보면 달라질
수도 있을 것 같은데 sky 나 dynamic 을 제외한 나머지 depth 들을 다 unproject 해서 첫 카메라나
카메라 centroid 에서의 point 거리 median 이 더 적절하지 않음?"

지금 정의(`scene_graph/scale.py:scene_scale`)는 **frame 0 의 non-sky 픽셀 평균 ray length** 하나다.
frame 0 이 실내 벽을 보고 있다가 카메라가 돌아 창밖을 보면 그 뒤 프레임들의 실제 관측 거리는
몇 배가 되는데 `S` 는 안 따라간다. 그러면 `u` 단위로 표현된 모든 것(OBB extent, 후보 거리,
게이트 마진 `0.02·S`, τ 임계)이 그 프레임들에서만 조용히 어긋난다.

**이 스크립트는 고치지 않고 잰다.** 후보를 바꿀지는 숫자를 보고 정한다 (뱅크 정체성이 바뀌는
변경이라 재굽기가 따라온다).

## 재는 후보

| 키 | 정의 |
|---|---|
| `S_f0`          | **현행.** frame 0 non-sky 평균 ray length |
| `S_f0_nodyn`    | frame 0 non-sky **∧ non-dynamic** 평균 ray length |
| `S_frames_mean` | 프레임별 non-sky·non-dyn 평균 ray length 를 **전 프레임 평균** |
| `S_pts_first`   | 전 프레임 non-sky·non-dyn 픽셀을 world 로 unproject → `‖p − C_0‖` 의 **median** |
| `S_pts_centroid`| 같은 점군, `‖p − mean(C_t)‖` 의 median |
| `S_pts_own`     | 같은 점군, **자기를 찍은 카메라까지의** 거리 median (= ray length median 과 동일) |

`S_pts_*` 는 사용자가 제안한 것이고 `S_pts_own` 은 그 대조군이다 — 셋이 크게 다르면 그 차이는
"씬이 넓다"가 아니라 **카메라가 움직였다**에서 온다 (`parallax_ratio` 와 같이 봐야 한다).

## 카메라가 돌면 정말 달라지나 — 이게 핵심 열이다

`S_t_p05 / S_t_p50 / S_t_p95 / S_t_ratio(=max/min) / S_t_drift(=S_48/S_0)`.
`S_t_ratio` 가 1 근처면 frame 0 하나로 충분하다는 직접 증거이고, 크면 사용자의 우려가 실측된다.
`S_f0 / S_pts_first` 비율도 같이 낸다 — 후보를 갈아탈 때 기존 임계를 몇 배로 옮겨야 하는지가
그 숫자다.

env: `vista4d` (렌더 없음, CPU 만)

예시:
    python eval/audit_scene_scale.py --videos camel avocado-slice parkour snowboard
    python eval/audit_scene_scale.py --all --stride 4 --out /tmp/scene_scale.csv
"""
import csv
import sys
from argparse import ArgumentParser
from os import listdir, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from scene_graph.io import load_scene                                           # noqa: E402
from scene_graph.scale import parallax_ratio, scene_scale, z_median             # noqa: E402


def ray_length_grid(K_t: np.ndarray, height: int, width: int, stride: int):
    """`‖K⁻¹[u+0.5, v+0.5, 1]‖` 격자 + 그 픽셀의 정규화 카메라 방향.

    z-depth 를 ray length 로 바꾸는 배율이자, unproject 방향이기도 하다 — 같은 것을 두 번
    만들지 않으려고 한 함수에서 둘 다 돌려준다.
    """
    v, u = np.meshgrid(np.arange(0, height, stride), np.arange(0, width, stride), indexing="ij")
    pixels = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], axis=-1).astype(np.float64)
    rays = pixels @ np.linalg.inv(K_t).T.astype(np.float64)          # h w 3, z 성분이 1
    return np.linalg.norm(rays, axis=-1), rays


def measure(recon: dict, stride: int, max_points: int, seed: int):
    """한 영상의 후보 `S` 들 + 프레임별 산포."""
    # `recon["intrinsics"]` 는 (f,4) 원본이다. 3×3 로 편 것은 `io.load_scene` 이 `K` 로 넣어 둔다.
    depths, K, cam_c2w = recon["depths"], recon["K"], recon["cam_c2w"]
    sky, dyn = recon["sky_mask"], recon["dynamic_mask"]
    num_frames, height, width = depths.shape[:3]

    per_frame, chunks = [], []
    for t in range(num_frames):
        ray_len, rays = ray_length_grid(np.asarray(K[t], float), height, width, stride)
        z = np.asarray(depths[t], np.float64)[::stride, ::stride]
        valid = np.isfinite(z) & (z > 0) & ~sky[t][::stride, ::stride]
        valid &= ~dyn[t][::stride, ::stride]                 # 동적 표면은 씬 크기가 아니다
        if not valid.any():
            per_frame.append(np.nan)
            continue
        per_frame.append(float((z[valid] * ray_len[valid]).mean()))
        # world 로 올린다. rays 의 z 성분이 1 이라 `z·ray` 가 곧 카메라 좌표다.
        cam_pts = (z[..., None] * rays)[valid]
        c2w = np.asarray(cam_c2w[t], float)
        chunks.append(cam_pts @ c2w[:3, :3].T + c2w[:3, 3])

    points = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))
    # 평균은 **서브샘플 전에** 낸다 — 아래 `max_points` 추출은 median 용이다.
    origin = np.asarray(cam_c2w, float)[0][:3, 3]
    ship_nodyn = float(np.linalg.norm(points - origin, axis=1).mean()) if len(points) else np.nan
    if len(points) > max_points:                             # median 은 표본으로 충분하다
        points = points[np.random.default_rng(seed).choice(len(points), max_points, False)]
    centers = np.asarray(cam_c2w, float)[:, :3, 3]

    per_frame = np.asarray(per_frame, float)
    finite = per_frame[np.isfinite(per_frame)]
    z_med = z_median(depths, sky, frame=0)
    row = {
        "frames": num_frames, "points": len(points),
        # 옛 게이지. `mode` 를 명시해 둔다 — 기본값이 2026-09-02 에 바뀌었으므로 안 적으면
        # 이 열이 조용히 `S_ship` 과 같은 값이 되어 배율 비교가 통째로 1 이 된다.
        "S_f0": scene_scale(depths, K, sky, mode="frame0_ray"),
        # **현재 출고되는 정의**. 감사 스크립트가 자체 재구현을 들고 있으면 정의가 갈리므로
        # 출고 코드 그대로 부른다 (sky 만 제외 = dynamic 포함, 전 프레임, 첫 카메라 거리 평균).
        "S_ship": scene_scale(depths, K, sky, cam_c2w=cam_c2w, mode="points_first_cam",
                              stride=stride),
        "S_ship_nodyn": ship_nodyn,                          # 같은 식에서 동적만 뺀 것
        "S_f0_nodyn": float(per_frame[0]),
        "S_frames_mean": float(finite.mean()),
        "S_pts_first": float(np.median(np.linalg.norm(points - centers[0], axis=1))),
        "S_pts_centroid": float(np.median(np.linalg.norm(points - centers.mean(0), axis=1))),
        "S_t_p05": float(np.percentile(finite, 5)),
        "S_t_p50": float(np.percentile(finite, 50)),
        "S_t_p95": float(np.percentile(finite, 95)),
        "S_t_ratio": float(finite.max() / max(finite.min(), 1e-9)),
        "S_t_drift": float(per_frame[-1] / max(per_frame[0], 1e-9)),
        "z_med_f0": z_med,
        "parallax_ratio": parallax_ratio(np.asarray(cam_c2w, float), z_med),
    }
    # 재굽기 blast radius: 옛 뱅크의 `u` 눈금이 이 배율만큼 통째로 바뀐다.
    row["r_ship"] = row["S_ship"] / max(row["S_f0"], 1e-9)
    row["r_ship_dyn"] = row["S_ship"] / max(row["S_ship_nodyn"], 1e-9)   # 동적 포함이 준 차이
    row["r_pts_first"] = row["S_pts_first"] / max(row["S_f0"], 1e-9)
    row["r_pts_centroid"] = row["S_pts_centroid"] / max(row["S_f0"], 1e-9)
    row["r_frames_mean"] = row["S_frames_mean"] / max(row["S_f0"], 1e-9)
    return row


def main():
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="*", default=None, type=str)
    parser.add_argument("--all", action="store_true", default=False)       # recon 있는 전량
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)
    parser.add_argument("--stride", default=4, type=int)                   # 픽셀 서브샘플
    parser.add_argument("--max_points", default=2_000_000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--out", default="/data1/cympyc1785/LatentCamVid/tmp/scene_scale.csv", type=str)
    args = parser.parse_args()

    videos = args.videos
    if args.all or not videos:
        root = path.join(args.eval_data, "eval_data", "recon_and_seg")
        videos = sorted(v for v in listdir(root) if path.isdir(path.join(root, v)))

    rows = []
    for video in videos:
        try:
            recon = load_scene(args.eval_data, video, args.vista4d_root,
                               seg_root=args.seg_root, seg_static_root=args.seg_static_root)
        except Exception as error:                                          # noqa: BLE001
            print(f"{video:<22}SKIP  {type(error).__name__}: {error}")
            continue
        row = {"video": video}
        row.update(measure(recon, args.stride, args.max_points, args.seed))
        rows.append(row)
        print(f"{video:<22}S_f0 {row['S_f0']:8.4f}  S_ship {row['S_ship']:8.4f}"
              f"  ratio {row['r_ship']:6.3f}  S_t max/min {row['S_t_ratio']:7.3f}"
              f"  drift {row['S_t_drift']:6.3f}  plx {row['parallax_ratio']:.4f}")

    if not rows:
        print("측정된 영상이 없다")
        return
    with open(args.out, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def column(key):
        return np.asarray([r[key] for r in rows], float)

    print(f"\n{'':<18}{'p05':>9}{'p25':>9}{'p50':>9}{'p75':>9}{'p95':>9}{'max':>9}")
    for key in ("S_t_ratio", "S_t_drift", "r_ship", "r_ship_dyn", "r_pts_first",
                "r_pts_centroid", "r_frames_mean", "parallax_ratio"):
        values = column(key)
        print(f"{key:<18}" + "".join(f"{np.percentile(values, p):9.4f}"
                                     for p in (5, 25, 50, 75, 95)) + f"{values.max():9.4f}")
    ratio = column("S_t_ratio")
    print(f"\n{'S_t_ratio > 1.2':<18}{(ratio > 1.2).mean() * 100:6.1f}%"
          f"   {'> 1.5':<8}{(ratio > 1.5).mean() * 100:6.1f}%"
          f"   {'> 2.0':<8}{(ratio > 2.0).mean() * 100:6.1f}%   n={len(rows)}")
    print(f"{'CSV':<18}{args.out}")


if __name__ == "__main__":
    main()
