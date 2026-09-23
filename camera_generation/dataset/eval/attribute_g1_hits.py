"""G1(behind-surface) 이 **무엇에 대해** 걸렸는지를 귀속한다.

`viz_g1_collision.py` 는 어느 (플랜 프레임 × 소스 프레임) 쌍이 pierce/tight 인지를 그려주지만
"그 픽셀이 무슨 표면이냐"는 안 알려준다. 게이트를 풀지(`clear_frac` 를 낮출지) 노드를 고칠지는
그 답에 달려 있다 — 바닥에 0.1·S 만큼 못 떨어져서 걸리는 것과 벽/사람에 붙어서 걸리는 것은
전혀 다른 처방이다.

귀속 두 축:
  ① **segmented** — 히트 픽셀(5×5 패치의 최소 depth 위치)이 어느 seg instance 안에 있나.
     `SegInstances.frame_masks` 로 dyn/stat 을 다 본다. 없으면 `unsegmented`.
  ② **기하** — 그 표면점을 world 로 되올린 뒤 G 프레임 z 와 `ground.ground_z` 를 비교한다.
     `ground` (지면 근처) / `below_cam` (카메라보다 아래지만 지면은 아님) / `level` (카메라 높이).
     ①이 unsegmented 일 때 바닥인지 벽인지를 가르는 게 이 축이다.

`judge_frame` 은 `viz_g1_collision` 에서 그대로 가져다 쓴다 — G1 식이 세 군데로 갈라지면 안 된다.

사용 예시:
    python eval/attribute_g1_hits.py --video parkour --bank_dir hole_bank_k6_d121 \
        --variant dyn_0__dolly_in__hole0.5 --behind_src_frames 49
    python eval/attribute_g1_hits.py --video parkour --bank_dir hole_bank_k6_d121 \
        --worst_n 20 --behind_src_frames 49        # 뱅크 상위 20 변이 집계
"""
import csv
import json
import sys
from argparse import ArgumentParser
from collections import Counter
from os import path

import numpy as np

sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from scene_graph.io import load_scene                                           # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402
from viz.viz_g1_collision import judge_frame                                # noqa: E402


def hit_pixel(depths, sky_mask, t, uv, radius_px):
    """5×5 패치에서 **실제로 최소 depth 를 준** 픽셀 (u, v). judge_frame 의 argmin 을 되짚는다."""
    height, width = depths.shape[-2:]
    u, v = int(np.floor(uv[0])), int(np.floor(uv[1]))
    if radius_px <= 0:
        return u, v
    u0, u1 = max(u - radius_px, 0), min(u + radius_px + 1, width)
    v0, v1 = max(v - radius_px, 0), min(v + radius_px + 1, height)
    patch = depths[t][v0:v1, u0:u1]
    usable = np.isfinite(patch) & (patch > 0) & ~sky_mask[t][v0:v1, u0:u1]
    masked = np.where(usable, patch, np.inf)
    dv, du = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return u0 + int(du), v0 + int(dv)


def label_at(segs, t, u, v):
    """히트 픽셀을 덮는 seg instance 의 keyword. 없으면 None."""
    for seg in segs:
        masks = seg.frame_masks(t)
        for slot, inst in enumerate(seg.frames[t]):
            if slot < len(masks) and masks[slot][v, u]:
                return f"{seg.kind}:{inst['keyword']}"
    return None


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    bank_folder = path.join(out_root, args.video, args.bank_dir)
    with open(path.join(bank_folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    variants = bank["variants"]
    cam_c2w_all = np.load(path.join(bank_folder, "poses.npz"), allow_pickle=False)["cam_c2w"]

    rows = {}
    csv_path = path.join(bank_folder, "bank.csv")
    if path.exists(csv_path):
        with open(csv_path, newline="") as file:
            rows = {r["variant_id"]: r for r in csv.DictReader(file)}

    targets = [args.variant] if args.variant else [
        vid for _, vid in sorted(((float(r.get("behind_frac") or 0.0), vid)
                                  for vid, r in rows.items()), reverse=True)[:args.worst_n]]
    assert targets, "--variant 또는 --worst_n 중 하나는 필요하다"

    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    recon = load_scene(args.eval_data, args.video, args.vista4d_root,
                       seg_root=args.seg_root, seg_static_root=args.seg_static_root)
    scale = float(graph["scale"]["S"])
    ground_z = float(graph["ground"]["ground_z"])
    T_gw = np.asarray(graph["frames"]["T_gw"], dtype=np.float64)
    depths, sky_mask, segs = recon["depths"], recon["sky_mask"], recon["segs"]
    K, cam_c2w = recon["K"], recon["cam_c2w"]
    num_src = len(depths)
    probe = np.unique(np.linspace(0, num_src - 1, args.behind_src_frames)
                      .round().astype(int)).tolist()

    tally = Counter()
    by_label, by_geom, deficit = Counter(), Counter(), []
    for vid in targets:
        index = next((i for i, v in enumerate(variants) if v["variant_id"] == vid), None)
        assert index is not None, f"변이를 못 찾았다: {vid}"
        poses = np.asarray(cam_c2w_all[index], dtype=np.float64)
        for pose in poses:
            p_world = pose[:3, 3]
            for t in probe:
                verdict, uv, z_cam, z_surf = judge_frame(
                    p_world, t, depths, K, cam_c2w, sky_mask, scale,
                    args.behind_margin_frac, args.behind_clear_frac, args.behind_radius_px)
                tally[verdict] += 1
                if verdict not in ("pierce", "tight"):
                    continue
                u, v = hit_pixel(depths, sky_mask, t, uv, args.behind_radius_px)
                by_label[label_at(segs, t, u, v) or "unsegmented"] += 1
                # 표면점을 world → G 로. depth 는 z-depth 이므로 ray 로 되올린다.
                ray = np.linalg.inv(K[t]) @ np.array([u + 0.5, v + 0.5, 1.0])
                surf_cam = ray * (z_surf / ray[2])
                surf_world = cam_c2w[t][:3, :3] @ surf_cam + cam_c2w[t][:3, 3]
                surf_g = T_gw[:3, :3] @ surf_world + T_gw[:3, 3]
                cam_g = T_gw[:3, :3] @ p_world + T_gw[:3, 3]
                drop = float(cam_g[2] - surf_g[2])          # 카메라가 표면보다 얼마나 위인가 (u)
                if abs(float(surf_g[2]) - ground_z) <= args.ground_band_u:
                    by_geom["ground"] += 1
                elif drop > args.ground_band_u:
                    by_geom["below_cam"] += 1
                else:
                    by_geom["level"] += 1
                # 얼마나 모자랐나. >0 이면 관통, <=0 이면 clear_frac 만큼의 여유가 부족한 것.
                deficit.append((z_cam - z_surf) / scale)

    total = sum(tally.values())
    print(f"video {args.video}   bank {args.bank_dir}   변이 {len(targets)}개   "
          f"probe {len(probe)}프레임   S={scale:.4f}   ground_z={ground_z:+.4f}u")
    print(f"{'판정':<14}{'수':>8}{'비율':>9}")
    for key in ("pierce", "tight", "clear", "skip"):
        print(f"  {key:<12}{tally[key]:>8}{100 * tally[key] / max(total, 1):>8.1f}%")
    hits = tally["pierce"] + tally["tight"]
    print(f"\n히트 {hits} 건의 귀속 — seg instance")
    for label, count in by_label.most_common():
        print(f"  {label:<26}{count:>8}{100 * count / max(hits, 1):>8.1f}%")
    print(f"\n히트 {hits} 건의 귀속 — 기하 (ground_band {args.ground_band_u}u)")
    for label, count in by_geom.most_common():
        print(f"  {label:<26}{count:>8}{100 * count / max(hits, 1):>8.1f}%")
    if deficit:
        arr = np.asarray(deficit)
        print(f"\n(z_cam − z_surf)/S,  u :  p10 {np.percentile(arr, 10):+.4f}  "
              f"p50 {np.median(arr):+.4f}  p90 {np.percentile(arr, 90):+.4f}  "
              f"max {arr.max():+.4f}   (>{args.behind_margin_frac} 이면 관통)")
        need = args.behind_clear_frac
        print(f"여유 요구 {need}u 를 {(arr > -need).mean() * 100:.1f}% 가 못 채웠다 "
              f"= 히트의 정의. 0.05u 로 낮추면 {(arr > -0.05).mean() * 100:.1f}%, "
              f"0.02u 면 {(arr > -0.02).mean() * 100:.1f}% 만 남는다")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_root", default=None)     # 기본 dataset/out
    parser.add_argument("--bank_dir", default="hole_bank")
    parser.add_argument("--variant", default=None)         # 단일 변이
    parser.add_argument("--worst_n", default=10, type=int)  # behind_frac 상위 N 변이 집계
    parser.add_argument("--behind_src_frames", default=49, type=int)
    parser.add_argument("--behind_margin_frac", default=0.02, type=float)
    parser.add_argument("--behind_clear_frac", default=0.10, type=float)
    parser.add_argument("--behind_radius_px", default=2, type=int)
    parser.add_argument("--ground_band_u", default=0.10, type=float)   # 지면으로 볼 z 밴드
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--seg_root", default=None)
    parser.add_argument("--seg_static_root", default=None)
    main(parser.parse_args())
