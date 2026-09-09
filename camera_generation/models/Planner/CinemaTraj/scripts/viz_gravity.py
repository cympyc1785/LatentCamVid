"""GeoCalib 이 낸 **중력축을 눈으로 검증**하는 오버레이. 숫자로는 못 고르기 때문에 있다.

`scene_graph.json` 의 `gravity.up_world` 는 world(=DA3 frame0 카메라) 좌표의 단위벡터인데,
그 자체로는 맞는지 틀린지 알 수가 없다. `angle_to_ransac_deg` 는 "둘이 다르다"만 말하고
어느 쪽이 맞는지는 안 말한다 (camera-lens 는 18.3° 어긋나 있고 `disagrees_with_ransac=true`).
사람이 판정할 수 있는 유일한 형태는 **화면에 그린 수직선**이다 — 벽 모서리·문틀·사람과
나란하면 맞은 것이다.

두 가지 스타일을 낸다 (`--style`). 어느 쪽이 읽기 쉬운지는 씬마다 다르다:

  plumb  깊이가 유효한 픽셀 격자에서 3D 로 올린 뒤 `up_world` 방향으로 선분을 세운다.
         + 지평선(중력에 수직인 평면의 소실선). 씬 구조와 직접 비교되는 게 장점.
  grid   `ground_z` 높이에 중력 정렬 바닥 격자를 깔고 중앙에 큰 up 화살표.
         바닥이 실제 바닥과 붙는지로 판정한다.

`--compare cam_up` 을 주면 폴백(카메라 up 평균) 방향을 주황으로 같이 그린다 — GeoCalib 이
카메라 up 과 몇 도 다른지가 그림으로 보인다.

사용:
  python scripts/viz_gravity.py --video camera-lens --style plumb
  python scripts/viz_gravity.py --video camera-lens --style grid --compare cam_up
"""
import sys
from argparse import ArgumentParser
from json import load as json_load
from os import makedirs, path

HERE = path.dirname(path.dirname(path.abspath(__file__)))
# `from sys import path as sys_path` 로 미리 묶으면 안 된다 — cv2 import 가 sys.path 를 **새
# 리스트로 갈아끼워서** 죽은 리스트에 insert 하게 되고 ModuleNotFoundError 가 난다.
sys.path.insert(0, HERE)

import cv2                                                              # noqa: E402
import numpy as np                                                      # noqa: E402

from scene_graph.io import load_scene                                    # noqa: E402
from scene_graph.lift import project_points                              # noqa: E402

EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    HERE, "..", "..", "..", "..", "video_generation", "models", "Vista4D"))
GEO_COLOR = (255, 220, 0)        # BGR 하늘색 — GeoCalib
CMP_COLOR = (0, 150, 255)        # BGR 주황 — 비교 대상
HORIZON_COLOR = (0, 255, 255)


def draw_segment(image, uv, z, color, thickness=2):
    """두 끝점 중 하나라도 카메라 뒤면 안 그린다 (투영이 뒤집혀 화면을 가로지른다)."""
    if z[0] <= 1e-6 or z[1] <= 1e-6:
        return False
    a = tuple(np.round(uv[0]).astype(int))
    b = tuple(np.round(uv[1]).astype(int))
    cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
    return True


def draw_horizon(image, up_world, K, cam_c2w, color):
    """중력에 수직인 평면들의 **소실선**. `n_cam = R_w2c · up` 이 그 선의 법선이다.

    카메라 좌표에서 방향 `d` 의 소실점은 `K d / d_z` 다. up 에 수직인 방향 전체의 소실점 집합은
    직선 `(K^{-T} n_cam)·[u,v,1] = 0` — 이게 지평선이다. 중력이 맞으면 이 선이 실제 지평선/
    책상 모서리와 나란하다.
    """
    height, width = image.shape[:2]
    w2c = np.linalg.inv(cam_c2w)
    n_cam = w2c[:3, :3] @ np.asarray(up_world, dtype=float)
    line = np.linalg.inv(K).T @ n_cam                       # a u + b v + c = 0
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    if abs(a) < 1e-9 and abs(b) < 1e-9:
        return
    points = []
    if abs(b) > 1e-9:
        for u in (0.0, float(width - 1)):
            points.append((u, -(a * u + c) / b))
    if abs(a) > 1e-9:
        for v in (0.0, float(height - 1)):
            points.append((-(b * v + c) / a, v))
    inside = [p for p in points
              if -1 <= p[0] <= width and -1 <= p[1] <= height]
    if len(inside) < 2:
        return
    cv2.line(image, tuple(np.round(inside[0]).astype(int)),
             tuple(np.round(inside[1]).astype(int)), color, 1, cv2.LINE_AA)
    cv2.putText(image, "horizon", (8, max(int(inside[0][1]) - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def plumb_frame(image, depth, sky, K, cam_c2w, ups, length, grid=(7, 5)):
    """유효 depth 픽셀 격자에서 3D 로 올려 `up` 방향 선분을 세운다.

    격자 칸마다 **중앙 픽셀**이 아니라 그 칸에서 depth 가 유효하고 sky 가 아닌 픽셀 중
    중앙에 가장 가까운 것을 쓴다 — 중앙이 하늘이면 그 칸이 통째로 비어 그림이 성글어진다.
    """
    height, width = depth.shape
    K_inv = np.linalg.inv(K)
    cols = np.linspace(0, width - 1, grid[0] + 2)[1:-1]
    rows = np.linspace(0, height - 1, grid[1] + 2)[1:-1]
    half_u, half_v = width / (2 * (grid[0] + 2)), height / (2 * (grid[1] + 2))
    for cv_ in rows:
        for cu in cols:
            u0, u1 = int(max(cu - half_u, 0)), int(min(cu + half_u, width - 1)) + 1
            v0, v1 = int(max(cv_ - half_v, 0)), int(min(cv_ + half_v, height - 1)) + 1
            patch = depth[v0:v1, u0:u1]
            ok = np.isfinite(patch) & (patch > 0) & ~sky[v0:v1, u0:u1]
            if not ok.any():
                continue
            vv, uu = np.nonzero(ok)
            pick = np.argmin((uu + u0 - cu) ** 2 + (vv + v0 - cv_) ** 2)
            u, v = int(uu[pick] + u0), int(vv[pick] + v0)
            z = float(patch[vv[pick], uu[pick]])
            ray = K_inv @ np.array([u + 0.5, v + 0.5, 1.0])
            base_world = cam_c2w[:3, :3] @ (ray * z) + cam_c2w[:3, 3]
            for up, color in ups:
                tip = base_world + np.asarray(up, dtype=float) * length
                uv, zc = project_points(np.stack([base_world, tip]), K, cam_c2w)
                if draw_segment(image, uv, zc, color, 2):
                    cv2.circle(image, tuple(np.round(uv[0]).astype(int)), 3, color, -1,
                               cv2.LINE_AA)


def grid_frame(image, K, cam_c2w, up_world, cam0_c2w, ground_z, scale, ups,
               half_cells=5, cell=0.35):
    """중력 정렬 바닥 격자 + 중앙 up 화살표. `ground_z` 는 G 단위(u)라 world 로 되돌린다."""
    up = np.asarray(up_world, dtype=float)
    forward = cam0_c2w[:3, 2]
    e_x = forward - np.dot(forward, up) * up
    e_x = e_x / max(np.linalg.norm(e_x), 1e-9)
    e_y = np.cross(up, e_x)
    origin = cam0_c2w[:3, 3] + up * (ground_z * scale)     # 카메라 중심에서 지면 높이로
    step = cell * scale
    span = half_cells * step
    for i in range(-half_cells, half_cells + 1):
        for axis, other in ((e_x, e_y), (e_y, e_x)):
            a = origin + other * (i * step) - axis * span
            b = origin + other * (i * step) + axis * span
            uv, zc = project_points(np.stack([a, b]), K, cam_c2w)
            draw_segment(image, uv, zc, (90, 200, 90), 1)
    for up_vec, color in ups:
        tip = origin + np.asarray(up_vec, dtype=float) * (1.2 * scale)
        uv, zc = project_points(np.stack([origin, tip]), K, cam_c2w)
        if draw_segment(image, uv, zc, color, 3):
            cv2.circle(image, tuple(np.round(uv[1]).astype(int)), 6, color, -1, cv2.LINE_AA)


def legend(image, rows):
    for i, (text, color) in enumerate(rows):
        y = 22 + 22 * i
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1,
                    cv2.LINE_AA)


def main(args):
    import imageio.v2 as imageio

    out_root = args.output_root or path.join(HERE, "out")
    graph_path = path.join(out_root, args.video, "scene_graph.json")
    with open(graph_path, "r") as handle:
        graph = json_load(handle)
    up_world = np.asarray(graph["gravity"]["up_world"], dtype=float)
    scale = float(graph["scale"]["S"])
    ground_z = float(graph["ground"]["ground_z"])

    recon = load_scene(args.eval_data, args.video, args.vista4d_root)
    video, depths, K, cam_c2w = recon["video"], recon["depths"], recon["K"], recon["cam_c2w"]
    sky = recon["sky_mask"].astype(bool)

    ups = [(up_world, GEO_COLOR)]
    rows = [(f"GeoCalib up  conf {graph['gravity']['confidence']:.2f}"
             f"  spread {graph['gravity']['spread_deg']:.1f}deg", GEO_COLOR)]
    if args.compare == "cam_up":
        # OpenCV +Y = down 이므로 카메라 up 은 -R[:,1]. 전 프레임 평균이 D98 이전 폴백이다.
        cam_up = -cam_c2w[:, :3, 1].sum(axis=0)
        cam_up = cam_up / max(np.linalg.norm(cam_up), 1e-9)
        ups.append((cam_up, CMP_COLOR))
        angle = np.degrees(np.arccos(np.clip(float(np.dot(cam_up, up_world)), -1, 1)))
        rows.append((f"camera-up fallback  {angle:.1f}deg apart", CMP_COLOR))

    frames = []
    for f in range(video.shape[0]):
        image = np.ascontiguousarray(video[f].copy())
        if args.style == "plumb":
            plumb_frame(image, depths[f], sky[f], K[f], cam_c2w[f], ups,
                        args.plumb_len * scale, grid=(args.grid_cols, args.grid_rows))
        else:
            grid_frame(image, K[f], cam_c2w[f], up_world, cam_c2w[0], ground_z, scale, ups)
        if args.horizon:
            draw_horizon(image, up_world, K[f], cam_c2w[f], HORIZON_COLOR)
        legend(image, rows)
        frames.append(image)

    out_dir = path.join(out_root, args.video, "vis")
    makedirs(out_dir, exist_ok=True)
    suffix = f"_{args.compare}" if args.compare else ""
    output = path.join(out_dir, f"gravity_{args.style}{suffix}.mp4")
    # cv2 mp4v 는 VS Code 뷰어에서 안 열린다 -> imageio + libx264.
    imageio.mimwrite(output, frames, fps=int(round(recon["fps"])), codec="libx264",
                     quality=6, macro_block_size=1)

    print(f"{'video':<18}{args.video}")
    print(f"{'style':<18}{args.style}")
    print(f"{'up_world':<18}{np.round(up_world, 4).tolist()}")
    print(f"{'method':<18}{graph['gravity']['method']}")
    print(f"{'confidence':<18}{graph['gravity']['confidence']}")
    print(f"{'spread_deg':<18}{graph['gravity']['spread_deg']}")
    print(f"{'vs cam_up deg':<18}{graph['gravity']['angle_to_cam_up_deg']}")
    print(f"{'vs ransac deg':<18}{graph['gravity'].get('angle_to_ransac_deg')}")
    print(f"{'output':<18}{output}")
    return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)   # None = <CinemaTraj>/out
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--style", default="plumb", choices=("plumb", "grid"))
    parser.add_argument("--compare", default="", choices=("", "cam_up"))  # 폴백 방향 같이 그리기
    parser.add_argument("--plumb_len", default=0.30, type=float)   # u 단위 선분 길이
    parser.add_argument("--grid_cols", default=5, type=int)        # 촘촘하면 씬이 안 보인다
    parser.add_argument("--grid_rows", default=3, type=int)
    parser.add_argument("--horizon", action="store_true", default=True)
    parser.add_argument("--no_horizon", dest="horizon", action="store_false")
    main(parser.parse_args())
