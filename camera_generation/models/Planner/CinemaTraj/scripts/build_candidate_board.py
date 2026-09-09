"""후보 풀 → 게이트 → `board_candidates.png`. **VLM 없이** 여기까지 먼저 사람이 본다.

왜 VLM 을 붙이기 전에 이 단계를 따로 두나: look-before-move 는 "모델에게 렌더를 보여준다"가
전부인 설계라, **보여줄 그림이 사람 눈에 말이 안 되면 프롬프트를 아무리 고쳐도 소용이 없다.**
게이트가 다 떨어뜨리는지, 남은 후보가 서로 구분되는지, OBB 박스가 렌더 위에서도 맞는지를
여기서 확인하고 넘어간다.

출력:
    <out>/<video>/board/board_candidates.png   VLM 이 볼 contact sheet 그대로
    <out>/<video>/board/source_frames.png      "원래 씬은 이렇게 생겼다" 패널
    <out>/<video>/board/contract.txt           같은 턴에 나갈 텍스트 블록
    <out>/<video>/board/gates.csv              **전 후보** 통과·탈락 + 사유

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=0 python scripts/build_candidate_board.py --video camel
    CUDA_VISIBLE_DEVICES=0 python scripts/build_candidate_board.py --video camel --subject_id dyn_0
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.candidates import (AZIMUTHS_DEG, DISTANCE_RATIOS, ELEVATIONS_DEG,      # noqa: E402
                            build_pool, build_pool_budget, g_pose_to_world, pick_subject)
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT, subject_point_mask  # noqa: E402
from lbm.gates import GATE_ORDER, evaluate, rank, write_csv                     # noqa: E402
from lbm.overlay import (board_labels, contact_sheet, contract_text,            # noqa: E402
                         draw_obb_world, draw_thirds, label_tile, paint_holes,
                         source_panel)
from lbm.render import CloudRenderer, look_at_c2w                               # noqa: E402
from scene_graph.io import load_scene                                           # noqa: E402
from scene_graph.lift import apply_transform                                    # noqa: E402
from scene_graph.obb import node_obb_at, obb_corners                            # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402


def subject_track_volume(recon: dict, node: dict):
    """노드가 흡수한 track 전부의 마스크 합집합 (F,H,W) bool.

    `merge_duplicates` 가 keyword 가 다른 track 을 3D IoU 로 합쳤으므로(avocado-slice 에서
    woman/person/human), 대표 track 하나만 쓰면 subject 점이 3분의 1로 줄어든다.
    """
    wanted = {(node["kind"], int(node["track_id"]))}
    for tag in node.get("merged_from", []):
        kind, track_id = tag.split("#")
        wanted.add((kind, int(track_id)))
    volume = None
    for seg in recon["segs"]:
        for kind, track_id in sorted(wanted):
            if seg.kind != kind or track_id not in seg.tracks:
                continue
            part = seg.track_volume(track_id)
            volume = part if volume is None else (volume | part)
    assert volume is not None, f"{node['id']}: subject track 마스크를 못 찾았다 ({wanted})"
    return volume


def source_camera_line(cam_c2w: np.ndarray, scale: float, z_med: float):
    """소스 카메라 움직임 한 줄 요약 (영문 — 프롬프트로 나간다)."""
    centers = cam_c2w[:, :3, 3]
    travel = float(np.linalg.norm(np.diff(centers, axis=0), axis=-1).sum()) / scale
    net = float(np.linalg.norm(centers[-1] - centers[0])) / scale
    relative = np.linalg.inv(cam_c2w[0]) @ cam_c2w[-1]
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))))
    right = float(np.dot(centers[-1] - centers[0], cam_c2w[0][:3, 0]) / scale)
    direction = "right" if right > 0 else "left"
    spread = float(np.linalg.norm(centers[:, None] - centers[None], axis=-1).max()) / max(z_med, 1e-9)
    return (f"path_length {travel:.3f} u | net_displacement {net:.3f} u ({direction}) | "
            f"rotation {rotation:.1f} deg | parallax_ratio {spread:.4f}")


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    cloud_path = path.join(out_root, args.video, "cloud.npz")
    assert path.isfile(cloud_path), f"cloud.npz 가 없다. 먼저 `python -m lbm.cloud --video {args.video}`"

    board_folder = path.join(out_root, args.video, "board")
    if path.isfile(path.join(board_folder, "gates.csv")) and args.skip_done:
        print(f"skip (done): {board_folder}")
        return
    makedirs(board_folder, exist_ok=True)

    renderer = CloudRenderer(cloud_path, vista4d_root=args.vista4d_root, device=args.device,
                             fixed_focal=args.fixed_focal)
    recon = load_scene(args.eval_data, args.video, args.vista4d_root,
                       seg_root=args.seg_root, seg_static_root=args.seg_static_root)
    node = pick_subject(graph["nodes"], args.subject_id, args.subject_min_area_frac)

    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
    scale, z_med = float(graph["scale"]["S"]), float(graph["scale"]["z_med_frame0"])

    volume = subject_track_volume(recon, node)
    subject_points = subject_point_mask(renderer.indices, volume).cpu().numpy()
    assert subject_points.sum() > 100, f"subject 점이 너무 적다: {int(subject_points.sum())}"

    if args.pool_mode == "budget":
        pool, pool_info = build_pool_budget(
            node, np.asarray(graph["cameras"]["cam_centers_g"], dtype=float),
            max_tau=args.max_tau, z_med=z_med, scale=scale, num_azimuth=args.num_azimuth,
            num_elevation=args.num_elevation, num_radius=args.num_radius,
            look_at_bias=args.look_at_bias, frame=args.frame)
    else:
        pool = build_pool(node, azimuths_deg=args.azimuths, elevations_deg=args.elevations,
                          distance_ratios=args.distances, look_at_bias=args.look_at_bias,
                          az_margin_deg=args.az_margin_deg, frame=args.frame)
        pool_info = {"mode": "absolute", "azimuths": args.azimuths,
                     "elevations": args.elevations, "distances": args.distances}
    for candidate in pool:
        candidate["c2w_world"] = g_pose_to_world(candidate["p_g"], candidate["look_at_g"],
                                                 T_wg, look_at_c2w)
        candidate["p_world"] = candidate["c2w_world"][:3, 3]
    pool_by_id = {candidate["cand_id"]: candidate for candidate in pool}

    subject_center_world = apply_transform(
        T_wg, np.asarray(node["track"]["center_smooth"], dtype=float)[args.frame][None])[0]
    behind_frames = np.unique(np.linspace(0, renderer.num_frames - 1, args.behind_frames)
                              .round().astype(int)).tolist()

    rows = [evaluate(candidate, renderer, subject_points,
                     obb_corners_world=None, depths=recon["depths"], sky_mask=recon["sky_mask"],
                     scale=scale, z_med=z_med, subject_center_world=subject_center_world,
                     src_center=renderer.cam_c2w_src[args.frame][:3, 3],
                     tile_height=args.tile_height, tile_width=args.tile_width,
                     behind_frames=behind_frames, max_tau=args.max_tau,
                     max_view_angle_deg=args.max_view_angle_deg, min_coverage=args.min_coverage,
                     subject_area_range=(args.min_subject_area, args.max_subject_area),
                     min_occlusion_pass=args.min_occlusion_pass, render_frame_index=args.frame)
            for candidate in pool]

    ordered = rank(rows)[:args.board_size]
    for label, row in zip(board_labels(len(ordered), args.board_columns), ordered):
        row["board_label"] = label

    # 타일 그리기 — 게이트가 이미 렌더해 둔 것을 재사용한다 (같은 그림을 두 번 렌더하지 않는다).
    corners_world = apply_transform(T_wg, obb_corners(*node_obb_at(node, args.frame)))
    scaled_K = renderer.K_src[args.frame].copy()
    scaled_K[0] *= args.tile_width / renderer.width
    scaled_K[1] *= args.tile_height / renderer.height

    tiles = []
    for row in ordered:
        rendered = row["rendered"]
        image = paint_holes(rendered["rgb"], rendered["valid"])
        draw_obb_world(image, corners_world, scaled_K, rendered["cam_c2w"])
        image = draw_thirds(image, safe_frac=args.center_box)
        # 캡션 각도도 소스 기준 상대값 — contract.txt 와 같은 수를 보여야 VLM 이 둘을 맞춰본다.
        angles = (f"daz{row['d_azimuth_deg']:+.0f} del{row['d_elevation_deg']:+.0f}"
                  if "d_azimuth_deg" in row else
                  f"az{row['azimuth_deg']:+.0f} el{row['elevation_deg']:.0f}")
        tiles.append(label_tile(image, row["board_label"],
                                f"{angles} d{row['distance_ratio']:.1f}x "
                                f"cov{row['coverage']:.2f} subj{row['subject_area']:.2f}"))
    board_path = path.join(board_folder, "board_candidates.png")
    if tiles:
        import imageio.v2 as imageio
        imageio.imwrite(board_path, contact_sheet(tiles, args.board_columns))

        # subject 가시 픽셀 상위 4프레임 — "물체가 제일 잘 보이는" 소스 근거를 붙인다.
        visible_per_frame = volume.reshape(volume.shape[0], -1).sum(axis=1)
        best = sorted(np.argsort(-visible_per_frame)[:4].tolist())
        imageio.imwrite(path.join(board_folder, "source_frames.png"),
                        source_panel(recon["video"], best, node, T_wg, recon["K"],
                                     recon["cam_c2w"], args.tile_height, args.tile_width))

    text = contract_text(graph, node, ordered,
                         source_camera_line(renderer.cam_c2w_src, scale, z_med),
                         {"max_view_angle_deg": args.max_view_angle_deg, "max_tau": args.max_tau,
                          "min_coverage": args.min_coverage})
    with open(path.join(board_folder, "contract.txt"), "w", encoding="utf-8") as file:
        file.write(text + "\n")
    for row in rows:
        row.pop("rendered", None)
    write_csv(path.join(board_folder, "gates.csv"), rows)

    # board.json — board 에 오른 후보의 **기하 전량**(G 좌표 + world c2w). decode 단계가 이걸 읽고
    # 후보 풀을 다시 렌더하지 않는다. gates.csv 는 사람이 읽는 진단표라 pose 를 안 싣는다.
    board = {"format": "lbm_board_v1", "video": args.video, "subject_id": node["id"],
             "subject_label": node["label"], "frame": args.frame,
             "pool": pool_info, "gate_thresholds": {
                 "max_tau": args.max_tau, "max_view_angle_deg": args.max_view_angle_deg,
                 "min_coverage": args.min_coverage, "center_box": args.center_box,
                 "subject_area_range": [args.min_subject_area, args.max_subject_area],
                 "min_occlusion_pass": args.min_occlusion_pass},
             "candidates": [{**{k: v for k, v in row.items()
                                if k not in ("rendered", "c2w_world", "p_world")},
                             "p_g": pool_by_id[row["cand_id"]]["p_g"].tolist(),
                             "look_at_g": pool_by_id[row["cand_id"]]["look_at_g"].tolist(),
                             "c2w_world": pool_by_id[row["cand_id"]]["c2w_world"].tolist()}
                            for row in ordered]}
    with open(path.join(board_folder, "board.json"), "w", encoding="utf-8") as file:
        json.dump(board, file, ensure_ascii=False, indent=1)

    counts = {name: sum(1 for r in rows if r["failed"] == name) for name in GATE_ORDER}
    print(f"\nsubject {node['id']} ({node['label']})  d_ref {node['viewing_distance']['d_ref']:.2f} u  "
          f"obs_az_span {node['obs_az_span_deg']:.1f} deg  subject_points {int(subject_points.sum()):,}")
    print("pool " + " ".join(f"{k}={v}" for k, v in pool_info.items()))
    print(f"pool {len(pool)} -> passed {len(rank(rows))} -> board {len(ordered)}")
    print("rejected: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    if not ordered:
        print("!! 통과 후보가 0 — subject 가 어느 방향에서도 안 보인다는 뜻이다. "
              "gates.csv 의 탈락 사유 분포를 먼저 볼 것 (폴백은 §B2)")
    else:
        print(f"\n{'lbl':<5}{'az':>6}{'el':>5}{'daz':>6}{'del':>6}{'dist':>6}{'cov':>7}{'subj':>7}"
              f"{'occl':>7}{'tau':>7}{'vang':>7}{'score':>7}")
        for row in ordered:
            print(f"{row['board_label']:<5}{row['azimuth_deg']:>6.0f}{row['elevation_deg']:>5.0f}"
                  f"{row.get('d_azimuth_deg', float('nan')):>6.0f}"
                  f"{row.get('d_elevation_deg', float('nan')):>6.0f}"
                  f"{row['distance_ratio']:>6.1f}{row['coverage']:>7.2f}{row['subject_area']:>7.3f}"
                  f"{row['occlusion_pass']:>7.2f}{row['tau']:>7.3f}{row['view_angle_deg']:>7.1f}"
                  f"{row['board_score']:>7.3f}")
    print(f"\n-> {board_folder}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)

    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    #    **기본 True.** board 는 뱅크와 같은 K 로 풀린 후보 pose 를 그리는 것이라 렌더도 frame0
    #    고정 K 여야 한다. 프레임별 DA3 K 를 쓰면 화각이 떨려(snowboard fx 진폭 6.99%) 타일 간
    #    구도 비교가 어긋난다. `--no_fixed_focal` 은 예전 동작(프레임별 K)이다.
    parser.add_argument("--fixed_focal", action="store_true", default=True)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    parser.add_argument("--subject_id", default=None, type=str)  # None = 자동 선택
    # subject 자동 선택의 화면 점유 하한. 이게 없으면 avocado-slice 에서 0.24% 짜리 조각이 뽑힌다.
    parser.add_argument("--subject_min_area_frac", default=0.01, type=float)
    parser.add_argument("--frame", default=0, type=int)          # 후보를 정의하는 기준 프레임
    parser.add_argument("--skip_done", action="store_true", default=False)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    # 후보 풀 (§B1). budget = τ 예산에서 역산(기본), absolute = 계획서 고정 격자.
    # camel 에서 absolute 는 72개 중 70개가 G4_tau 탈락이다 (§lbm/candidates.py docstring).
    parser.add_argument("--pool_mode", default="budget", choices=["budget", "absolute"])
    parser.add_argument("--num_azimuth", default=5, type=int)
    parser.add_argument("--num_elevation", default=3, type=int)
    parser.add_argument("--num_radius", default=3, type=int)
    parser.add_argument("--azimuths", nargs="*", type=float, default=list(AZIMUTHS_DEG))
    parser.add_argument("--elevations", nargs="*", type=float, default=list(ELEVATIONS_DEG))
    parser.add_argument("--distances", nargs="*", type=float, default=list(DISTANCE_RATIOS))
    parser.add_argument("--look_at_bias", default=0.0, type=float)
    # 관측 방위 밖을 자르는 하드 필터. 180 = off. camel obs_az_span 이 2.6도라 켜면 board 가 죽는다.
    parser.add_argument("--az_margin_deg", default=180.0, type=float)

    # 게이트 임계 (§B2)
    parser.add_argument("--max_tau", default=0.30, type=float)
    parser.add_argument("--max_view_angle_deg", default=40.0, type=float)
    parser.add_argument("--min_coverage", default=0.55, type=float)
    parser.add_argument("--center_box", default=0.80, type=float)
    parser.add_argument("--min_subject_area", default=0.03, type=float)
    parser.add_argument("--max_subject_area", default=0.50, type=float)
    parser.add_argument("--min_occlusion_pass", default=0.40, type=float)
    parser.add_argument("--behind_frames", default=7, type=int)

    # 9칸 3x3 이 기본이다 (DECISIONS D35/D36). 27칸은 4352x818 의 5.32:1 띠라 사람도 못 보고,
    # 실측상 VLM 도 못 본다 — select 턴 1샷 응답의 7/100 이 max_tokens 에서 잘렸고(9칸 0/100),
    # 숫자를 가리면 27칸에서는 최상위 타일을 아예 못 찾는다. `--board_size 27 --board_columns 9
    # --tile_width 480 --tile_height 270` 로 예전 동작을 그대로 재현할 수 있다.
    parser.add_argument("--board_size", default=9, type=int)
    parser.add_argument("--board_columns", default=3, type=int)
    parser.add_argument("--tile_width", default=640, type=int)
    parser.add_argument("--tile_height", default=360, type=int)

    main(parser.parse_args())
