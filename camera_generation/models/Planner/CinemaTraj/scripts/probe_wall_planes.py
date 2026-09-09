"""CinemaTraj 식 "벽을 fence 로 쓰기"가 우리 씬에서 성립하는지 실측한다.

CinemaTraj 는 wall/floor/ceiling 박스를 **watertight enclosure** 로 스냅해 SDF 부호를 만든다
(§3.1.2). 우리는 `relations.find_wall_planes` 가 벽 평면을 이미 찾지만 `(normal, d)` 를 버리고
개수만 남긴다 (D55 옵션 5). 이 스크립트는 그 평면들을 되살려서 **반공간 게이트로 쓸 수 있는지**를
소스 카메라 합법성 규칙(D47/D49/D51/D55/D56 — 원본 footage 를 기각하는 임계는 버그다)으로 판정한다.

각 평면에 대해 재는 것:
  · inlier 비율, 법선, 원점 거리
  · 정적 점군이 양쪽에 어떻게 갈리는지  (한쪽이 거의 비어야 진짜 "경계"다)
  · **소스 카메라 49대**가 어느 쪽에 있는지  (safe side 가 정의되면 그쪽이 실내다)
  · 뱅크 카메라가 safe side 밖으로 나가는 비율  (게이트를 켜면 몇 %가 죽는지)

사용 예시:
    python scripts/probe_wall_planes.py --videos camel avocado-slice
"""
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))

from scene_graph.relations import find_wall_planes, ground_height  # noqa: E402


def apply_transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def main():
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="+", default=["camel", "avocado-slice"])
    parser.add_argument("--output_root", default="out")
    parser.add_argument("--bank_dir", default="hole_bank")
    parser.add_argument("--max_walls", default=3, type=int)
    parser.add_argument("--max_points", default=400000, type=int)
    args = parser.parse_args()

    import json
    for video in args.videos:
        root = path.join(args.output_root, video)
        graph = json.load(open(path.join(root, "scene_graph.json")))
        T_gw = np.array(graph["frames"]["T_gw"], dtype=np.float64)
        S = float(graph["scale"]["S"])
        cloud = np.load(path.join(root, "cloud.npz"))
        # 4천만 점을 다 쓸 필요가 없다 — RANSAC 은 비율만 보므로 균등 추출로 충분하다.
        stride = max(1, len(cloud["points_world"]) // args.max_points)
        pts_w = cloud["points_world"][::stride].astype(np.float64)
        vis = np.unpackbits(cloud["visible_packed"][::stride], axis=1,
                            count=int(cloud["visible_num_frames"]))
        static_w = pts_w[vis.sum(1) > 1]                     # 1프레임만 보인 점 = 동적
        # T_gw 가 1/S 를 이미 품고 있다 (graph_frame(up, c2w0, S)). 따로 나누지 않는다.
        static_g = apply_transform(T_gw, static_w)
        cams_g = np.array(graph["cameras"]["cam_centers_g"], dtype=np.float64)

        walls = find_wall_planes(static_g, max_walls=args.max_walls)
        gz = ground_height(static_g)
        print(f"\n===== {video}   정적 점 {len(static_g):,}   ground_z {gz:.4f}·S   "
              f"벽 평면 {len(walls)}")

        z = np.load(path.join(root, args.bank_dir, "poses.npz"))
        bank_g = apply_transform(T_gw, z["cam_c2w"][:, :, :3, 3].reshape(-1, 3) / S)
        n_var = z["cam_c2w"].shape[0]

        print(f"{'#':<3}{'법선(G)':<26}{'d':>9}{'inlier':>9}{'점 앞/뒤':>14}"
              f"{'소스캠 safe측':>14}{'뱅크캠 위반':>13}{'판정':>10}")
        for i, (n, d) in enumerate(walls):
            sd_pts = static_g @ n + d
            sd_src = cams_g @ n + d
            sd_bank = bank_g @ n + d
            # safe side = 소스 카메라 다수가 있는 쪽. 정의가 안 되면 fence 로 못 쓴다.
            sign = 1.0 if np.median(sd_src) > 0 else -1.0
            src_ok = float((sd_src * sign > 0).mean())
            bank_bad = float((sd_bank * sign <= 0).mean())
            front = float((sd_pts * sign > 0).mean())
            verdict = "쓸 수 있음" if src_ok == 1.0 and front > 0.9 else "못 씀"
            print(f"{i:<3}{str(np.round(n, 3)):<26}{d:>9.4f}{'':>9}"
                  f"{f'{front:.3f}/{1-front:.3f}':>14}{src_ok:>14.3f}{bank_bad:>13.3f}{verdict:>10}")
        print(f"    (뱅크 변이 {n_var} × 49프레임 = {len(bank_g):,} pose 로 위반율 계산)")


if __name__ == "__main__":
    main()
