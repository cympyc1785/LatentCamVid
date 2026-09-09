"""영상 1편 → `scene_graph.json` + 승인용 그림. Part A 전체를 여기서 조립한다.

왜 scene graph 가 필요한가 (렌더 게이트가 공간 제약을 다 집행하는데도): 렌더러는 "여기서 보면
이렇게 보인다"만 말할 수 있고, **어디를 봐야 하는지는 말해주지 못한다**. graph 가 주는 건 네 가지다.
  ① subject 지목 — 후보 카메라 풀의 원점이자 framing 게이트의 대상
  ② 중력축 — roll=0 의 기준. world 는 기울어진 DA3 frame0 카메라라 여기서 안 주면 Dutch angle
  ③ 단위 `S` 와 `d_ref` — "1.0배 거리" 같은 말이 성립하게
  ④ 라벨/관계 — VLM 이 물체를 이름으로 부를 수 있게

env: `vista4d` (cv2 · scipy · imageio · numpy 만 쓴다. torch 불필요, open3d/sklearn 불필요)

예시:
    python scripts/build_scene_graph.py --video camel
    python scripts/build_scene_graph.py --video avocado-slice --no_skip_done --topdown
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from scene_graph.geocalib_sidecar import angle_between, load_geocalib_gravity       # noqa: E402
from scene_graph.gravity import estimate_gravity                                    # noqa: E402
from scene_graph.gt_trumans import gt_gravity, gt_subject                           # noqa: E402
from scene_graph.instances import (build_instances, cap_instances,                  # noqa: E402
                                   filter_instances, merge_duplicates, smooth_centers,
                                   summarize_tracks)
from scene_graph.io import SegInstances, load_scene                                 # noqa: E402
from scene_graph.lift import (apply_transform, graph_frame, unproject_mask,         # noqa: E402
                              valid_pixels)
from scene_graph.obb import (azimuth_span, deinflate_depth_axis, fit_obb,           # noqa: E402
                             observation_azimuths, observed_faces, unwrap_yaw, yaw_to_R)
from scene_graph.relations import build_relations, ground_height                    # noqa: E402
from scene_graph.scale import (SCALE_MODES, parallax_ratio,                        # noqa: E402
                               scene_scale, z_median)
from scene_graph.schema import GRAPH_FORMAT, check_reprojection, save_graph         # noqa: E402
from scene_graph.viz import obb_overlay_video, topdown_png                          # noqa: E402

VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "..", "..", "video_generation", "models", "Vista4D"))
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
GT_ROOT_DEFAULT = path.join(CINEMATRAJ_ROOT, "out", "trumans_recon")


def sample_static_points(recon: dict, num_frames_sampled: int, max_points: int, seed: int = 0):
    """정적·비하늘 픽셀을 몇 프레임에서만 뽑아 world 점군으로. gravity / 지면 / 벽 추정용.

    전 프레임을 쓰면 4천만 점이라 RANSAC 이 못 돈다. 정적 구조는 프레임마다 거의 같으므로
    균등 표본 5프레임이면 충분하다.
    """
    rng = np.random.default_rng(seed)
    total = recon["depths"].shape[0]
    indices = np.unique(np.linspace(0, total - 1, num_frames_sampled).round().astype(int))
    chunks = []
    for f in indices:
        mask = recon["static_mask"][f] & valid_pixels(recon["depths"][f], recon["sky_mask"][f])
        points, _ = unproject_mask(recon["depths"][f], recon["K"][f], recon["cam_c2w"][f], mask)
        chunks.append(points)
    points = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3))
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    return points


def floor_ground_z(recon: dict, T_gw: np.ndarray, S: float, floor_seg_root: str, video: str,
                   num_frames_sampled: int = 5, quantile: float = 0.5, min_points: int = 2000):
    """SAM3 가 `floor`/`ground` 로 잡은 픽셀만 올려서 지면 높이(G 의 z)를 잰다. 없으면 None.

    왜 이게 `relations.ground_height` 의 2% 분위수보다 나은가: 분위수는 **정적 점 전체**를
    보므로 바닥이 화면에 조금만 나오면 다른 것(가구 아랫면, 계단 밑)이 하위 2% 를 차지하고,
    반대로 카메라가 아래를 많이 보면 바닥을 실제보다 낮게 잡는다. D122 280 scene 실측:
    최대 동적 노드 발밑 gap median 0.120u (= 그 노드 높이의 48%), 8.2% 는 gap 이 음수였다.
    "바닥"이라고 지목된 픽셀만 쓰면 분위수를 안 거쳐도 되므로 여기서는 **median** 을 쓴다.

    노드로는 올리지 않는다 (D62 재조준) — 바닥 OBB 는 씬 전체를 덮어 `near` 엣지가 전량
    걸리고 G5 clearance 가 모든 노드의 MIN 이라 후보가 전멸한다. 마스크의 용도는 이 스칼라
    하나뿐이다.
    """
    folder = path.join(floor_seg_root, video)
    if not path.isfile(path.join(folder, "masks.npz")):
        return None
    seg = SegInstances(folder, "floor")
    total = recon["depths"].shape[0]
    indices = np.unique(np.linspace(0, total - 1, num_frames_sampled).round().astype(int))
    chunks = []
    for f in indices:
        masks = seg.frame_masks(int(f))
        if not len(masks):
            continue
        # 이 프레임의 floor 인스턴스를 전부 합친다 — 바닥이 물체에 가려 조각날 수 있다.
        mask = masks.any(axis=0) & valid_pixels(recon["depths"][f], recon["sky_mask"][f])
        if not mask.any():
            continue
        points, _ = unproject_mask(recon["depths"][f], recon["K"][f], recon["cam_c2w"][f], mask)
        chunks.append(points)
    if not chunks:
        return None
    points_g = apply_transform(T_gw, np.concatenate(chunks, axis=0))
    if len(points_g) < min_points:
        return None
    return float(np.quantile(points_g[:, 2], quantile)), len(points_g)


def build_node(inst: dict, node_id: str, num_frames: int, cam_centers_g: np.ndarray,
               view_dir_g: np.ndarray, parallax: float, deinflate: bool = False,
               min_points_per_frame: int = 30,
               yaw_median_window: int = 5, moving_threshold_u: float = 0.05):
    """인스턴스 하나 → 노드 dict. 프레임별 OBB fit → yaw unwrap + median → 대표 OBB."""
    frames = sorted(f for f, points in inst["points_by_frame"].items()
                    if len(points) >= min_points_per_frame)
    if len(frames) < 1:
        return None

    per_frame = [fit_obb(inst["points_by_frame"][f]) for f in frames]
    centers = np.stack([obb["center"] for obb in per_frame])
    extents = np.stack([obb["extent"] for obb in per_frame])
    yaws = unwrap_yaw([obb["yaw_rad"] for obb in per_frame])

    # yaw 는 프레임별 fit 이 얇은 물체에서 튄다 → 이동 median 으로 편다 (median 은 outlier 에 강함).
    if len(yaws) >= yaw_median_window:
        padded = np.pad(yaws, yaw_median_window // 2, mode="edge")
        yaws = np.array([np.median(padded[i:i + yaw_median_window]) for i in range(len(yaws))])
    extent = np.median(extents, axis=0)

    centers_dense = smooth_centers(centers, frames, num_frames)
    yaws_dense = np.interp(np.arange(num_frames), frames, yaws)

    ref = frames[0]
    cos, sin = np.cos(yaws_dense[ref]), np.sin(yaws_dense[ref])
    obb = {"center": centers_dense[ref], "extent": extent, "yaw_rad": float(yaws_dense[ref]),
           "R": np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
           "z_lo": float(centers_dense[ref][2] - extent[2] / 2),
           "z_hi": float(centers_dense[ref][2] + extent[2] / 2)}
    obb, inflated = deinflate_depth_axis(obb, view_dir_g, parallax, enabled=deinflate)

    azimuths = observation_azimuths(centers_dense[frames], cam_centers_g[frames], yaws_dense[frames])
    distances = np.linalg.norm(cam_centers_g[frames] - centers_dense[frames], axis=-1)
    path_len = float(np.linalg.norm(np.diff(centers_dense[frames[0]:frames[-1] + 1], axis=0),
                                    axis=-1).sum()) if len(frames) > 1 else 0.0
    drift = float(np.linalg.norm(centers_dense[frames[-1]] - centers_dense[frames[0]]))
    span = frames[-1] - frames[0]

    # `moving` 은 **kind 로 정한다**. 임계로는 못 가른다 — 큰 정적 물체는 프레임마다 보이는
    # 부분이 달라져 OBB 중심이 떠돌고, 그 떠돎이 실제 운동보다 크다. camel 씬 실측
    # (path_len_u / center_drift_u): 낙타 0.0674/0.0546 · 0.0576/0.0290, 울타리 0.3217/0.0379,
    # 지붕 0.1495/0.0632, 나무 0.1545/0.0466 — 울타리가 낙타보다 4.8배 "움직인다". 경로길이든
    # 순변위든 두 무리가 겹쳐서 임계가 존재하지 않는다. 정적 인스턴스는 정적 명사에서 나왔고
    # `sam3_static_instances.py` 가 dynamic_mask 겹침으로 한 번 더 걸렀으므로 그걸 믿는다.
    # 측정치는 그대로 노드에 남긴다 — 숨기면 OBB 떠돎 자체를 진단할 수 없다.
    #
    # **D128 에서 이 잣대를 순변위로 바꾸려다 접었다 (F1 기각, 사용자 판단 2026-09-05).**
    # `moving` 은 `schema.py:207` 의 anchor split key 이기도 하다 — 여기서 강등하면 그 dyn
    # 노드가 `max_static` 버킷으로 넘어가 벽·바닥과 3칸을 다툰다 (실측: `swing` 은 dyn 6개가
    # 전부 넘어가 dynamic anchor 가 0 이 된다). 제자리에서 움직이는 물체도 dyn 인 건 맞으므로
    # `moving` 은 그대로 두고, 순변위 판정은 `sample_camera_bank.py --track_min_drift_u` 에서
    # `track_*` 라우팅에만 건다.

    best_pos = int(np.argmax([inst["area_frac_by_frame"].get(f, 0.0) for f in frames]))
    best_frame = frames[best_pos]
    bbox_by_frame = {f: box for f, box in zip(inst["frames"], inst["boxes_xyxy"])}

    return {
        "id": node_id, "kind": inst["kind"], "label": inst["label"],
        "aliases": inst.get("aliases", []), "track_id": int(inst["track_id"]),
        "merged_from": inst.get("merged_from", []),
        "obb": obb, "extent_inflated": bool(inflated),
        "moving": bool(inst["kind"] == "dyn" and path_len > moving_threshold_u),
        "path_len_u": round(path_len, 4),
        "center_drift_u": round(drift, 4),
        "speed_u_per_frame": round(path_len / max(span, 1), 5),
        "track": {"frames": frames, "center_smooth": centers_dense, "yaw": yaws_dense,
                  "conf": [round(float(s), 3) for s in inst["scores"]]},
        "obs_az_span_deg": round(azimuth_span(azimuths), 1),
        "observed_faces": observed_faces(azimuths),
        "viewing_distance": {"d_ref": round(float(np.median(distances)), 4),
                             "d_min": round(float(distances.min()), 4),
                             "d_max": round(float(distances.max()), 4)},
        "best_frame": int(best_frame), "bbox_xyxy_best": bbox_by_frame[best_frame],
        "bbox_by_frame": bbox_by_frame,
        "num_visible_frames": inst["num_visible_frames"],
        "mean_score": round(inst["mean_score"], 3),
        "max_area_frac": round(inst["max_area_frac"], 5),
    }


def apply_gt_subject(nodes: list, gt_sub: dict, cam_centers_g: np.ndarray, num_frames: int,
                     moving_threshold_u: float = 0.05):
    """dyn 노드 하나를 mesh GT 로 **제자리 교체** (D106). 교체된 노드를 돌려준다.

    어느 노드가 사람인가: GT 궤적과 track center 의 median 거리가 최소인 dyn 노드. TRUMANS 는
    사람 1명이 유일한 동적 subject 지만, SAM3 가 소품을 dyn 으로 잡는 chunk 가 있어 이름이 아니라
    거리로 고른다.

    무엇을 갈아끼우나 — 하류가 실제로 읽는 것 전부:
      `track.center_smooth` (뱅크 look_at/aim keyframe, follow, relations, export_target_track)
      `track.yaw` + `obb.extent/R` (`node_obb_at` 이 충돌 slack 을 이 셋으로 계산)
      `obb.center/z_lo/z_hi`, `viewing_distance.d_ref` (후보 거리 눈금)
    2D 필드(`bbox_by_frame`, `best_frame`)는 마스크에서 온 것이라 그대로 둔다. GT AABB 는 blend
    축 정렬이라 yaw 가 상수(씬 회전각)다 — 사람의 heading 이 아니므로 `observed_faces` 는 "blend
    축 기준 어느 면" 으로 뜻이 바뀐다 (span 크기는 상수 빼기라 안 변함).
    """
    dyn_nodes = [n for n in nodes if n["kind"] == "dyn"]
    assert dyn_nodes, "GT subject 를 붙일 dyn 노드가 없다"
    centers_gt = gt_sub["centers_g"]
    err = [float(np.median(np.linalg.norm(
        np.asarray(n["track"]["center_smooth"], dtype=float)[:num_frames] - centers_gt, axis=1)))
        for n in dyn_nodes]
    node = dyn_nodes[int(np.argmin(err))]
    est_aim_err = min(err)

    frames = list(range(num_frames))
    yaws = np.full(num_frames, gt_sub["yaw_g"])
    extent = np.asarray(gt_sub["extent_u"], dtype=float)
    center0 = centers_gt[0]
    node["obb"] = {"center": center0, "extent": extent, "yaw_rad": float(gt_sub["yaw_g"]),
                   "R": yaw_to_R(gt_sub["yaw_g"]),
                   "z_lo": float(center0[2] - extent[2] / 2),
                   "z_hi": float(center0[2] + extent[2] / 2)}
    node["extent_inflated"] = False
    node["track"] = {"frames": frames, "center_smooth": centers_gt, "yaw": yaws,
                     "conf": [1.0] * num_frames}

    path_len = float(np.linalg.norm(np.diff(centers_gt, axis=0), axis=-1).sum())
    drift = float(np.linalg.norm(centers_gt[-1] - centers_gt[0]))
    node["moving"] = bool(path_len > moving_threshold_u)
    node["path_len_u"] = round(path_len, 4)
    node["center_drift_u"] = round(drift, 4)
    node["speed_u_per_frame"] = round(path_len / max(num_frames - 1, 1), 5)
    node["num_visible_frames"] = num_frames

    azimuths = observation_azimuths(centers_gt, cam_centers_g[:num_frames], yaws)
    distances = np.linalg.norm(cam_centers_g[:num_frames] - centers_gt, axis=-1)
    node["obs_az_span_deg"] = round(azimuth_span(azimuths), 1)
    node["observed_faces"] = observed_faces(azimuths)
    node["viewing_distance"] = {"d_ref": round(float(np.median(distances)), 4),
                                "d_min": round(float(distances.min()), 4),
                                "d_max": round(float(distances.max()), 4)}
    node["subject_gt"] = {"method": gt_sub["method"], "source": gt_sub["source"],
                          "human_height_m": gt_sub["human_height"],
                          # 추정이 얼마나 틀렸었는지를 노드에 박아 둔다 (gravity 와 같은 규칙).
                          "estimated_aim_err_med_u": round(est_aim_err, 4)}
    return node


def main(args):
    recon = load_scene(args.eval_data, args.video, args.vista4d_root,
                       seg_root=args.seg_root, seg_static_root=args.seg_static_root)
    graph_output_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    output_folder = path.join(graph_output_root, args.video)
    output_path = path.join(output_folder, "scene_graph.json")
    if path.isfile(output_path) and args.skip_done:
        print(f"skip (done): {output_path}")
        return
    makedirs(path.join(output_folder, "vis"), exist_ok=True)

    num_frames, height, width, _ = recon["video"].shape
    K, cam_c2w = recon["K"], recon["cam_c2w"]

    # 게이지는 **전처리 없는 raw depth** 에서. sky 를 제외하는 것 말고는 아무것도 건드리지 않는다.
    S = scene_scale(recon["depths"], K, recon["sky_mask"], cam_c2w=cam_c2w,
                    mode=args.scene_scale_mode, stride=args.scene_scale_stride)
    z_med = z_median(recon["depths"], recon["sky_mask"])
    plx = parallax_ratio(cam_c2w, z_med)

    static_world = sample_static_points(recon, args.gravity_frames, args.gravity_max_points)
    assert len(static_world) > 1000, f"정적 점이 너무 적다: {len(static_world)}"
    gravity = estimate_gravity(static_world, cam_c2w, scale=S, seed=args.seed,
                               weak_inlier_ratio=args.weak_inlier_ratio)

    # GT 중력축 (TRUMANS 전용). 우리가 Blender 에서 렌더한 클립이라 blend world pose 가 남아
    # 있고 blend 는 z-up 이므로 중력축이 정확하다. 191 chunk 실측: RANSAC 이 물면 오차 median
    # 0.1° 로 사실상 정확하지만 **26% 가 camera_up_fallback 으로 떨어졌고** 그쪽은 median
    # 10.9° / max 26.2° 다. `--gravity_source ransac`(기본) 이면 예전 결과 비트 동일.
    gravity_estimated = gravity
    if args.gravity_source == "gt":
        gt = gt_gravity(args.video, recon_root=args.gt_root)
        assert gt is not None, (
            f"{args.video}: GT 중력축을 못 찾았다 (--gt_root {args.gt_root}). TRUMANS chunk 가 "
            f"아니거나 render_a<NN>/cameras.json 이 없다. 추정으로 넘기려면 "
            f"--gravity_source ransac.")
        gravity = gt

    # GeoCalib 사이드카 (in-the-wild 용). RANSAC 은 계속 돌리되 **검증으로만** 쓴다 —
    # 어긋나도 자동 전환하지 않고 `angle_to_ransac_deg` 로 드러낸다 (§geocalib_sidecar 실측표).
    # `auto`(기본) = 사이드카가 있으면 GeoCalib, 없으면 RANSAC → 사이드카 없는 코퍼스는 비트 동일.
    geocalib = None
    if args.gravity_source in ("geocalib", "auto"):
        geocalib = load_geocalib_gravity(args.video, output_root=graph_output_root)
        assert geocalib is not None or args.gravity_source != "geocalib", (
            f"{args.video}: GeoCalib 사이드카가 없다 "
            f"({path.join(graph_output_root, args.video, 'geocalib_gravity.json')}). "
            f"env geocalib 에서 scripts/geocalib_gravity.py 를 먼저 돌리거나 "
            f"--gravity_source ransac.")
        if geocalib is not None:
            gravity = geocalib

    T_gw, T_wg = graph_frame(gravity["up_world"], cam_c2w[0], S)
    static_g = apply_transform(T_gw, static_world)
    cam_centers_g = apply_transform(T_gw, cam_c2w[:, :3, 3])
    view_dir_g = T_gw[:3, :3] @ cam_c2w[0][:3, 2]
    view_dir_g /= np.linalg.norm(view_dir_g)

    # ① 사전 선별 — 2D 마스크만으로 판정되는 것을 **lift 전에** 거른다. car-roundabout 은
    #    track 이 123개라 이게 없으면 버릴 것까지 49프레임씩 unproject 한다.
    #    상한은 여기서 `prescreen_factor` 배로 넉넉히 준다: 최종 상한은 merge 뒤에 걸어야
    #    "면적 상위 K" 가 물체 단위로 세어지는데(같은 물체의 조각 5개가 상위 5칸을 먹으면 안 된다),
    #    lift 비용은 지금 깎아야 하므로 둘 사이의 타협이다.
    summaries = summarize_tracks(recon)
    pre_kept, dropped = filter_instances(summaries, args.min_area_frac, args.min_frames,
                                         args.min_score, min_points=0)
    factor = args.prescreen_factor
    pre_kept, pre_capped = cap_instances(pre_kept, args.per_keyword_top_k * factor,
                                         args.max_dyn_nodes * factor,
                                         args.max_stat_nodes * factor)
    dropped.extend(pre_capped)
    select = {(inst["kind"], inst["track_id"]) for inst in pre_kept}

    # ② 살아남은 track 만 3D 로 올린다.
    instances = build_instances(recon, T_gw, S, max_points_per_frame=args.max_points_per_frame,
                                depth_trim_quantile=args.depth_trim_quantile,
                                depth_mode_bins=args.depth_mode_bins,
                                depth_mode_gap_frac=args.depth_mode_gap_frac,
                                select=select, seed=args.seed)
    kept, lifted_dropped = filter_instances(instances, args.min_area_frac, args.min_frames,
                                            args.min_score)   # 여기서 새로 걸리는 건 min_points
    dropped.extend(lifted_dropped)
    merged, merge_log = merge_duplicates(kept, args.merge_voxel_u, args.merge_iou)
    merged, capped = cap_instances(merged, args.per_keyword_top_k,
                                   args.max_dyn_nodes, args.max_stat_nodes)
    dropped.extend(capped)

    nodes, counters = [], {}
    for inst in sorted(merged, key=lambda i: (i["kind"], -i["max_area_frac"])):
        counters[inst["kind"]] = counters.get(inst["kind"], -1) + 1
        node = build_node(inst, f"{inst['kind']}_{counters[inst['kind']]}", num_frames,
                          cam_centers_g, view_dir_g, plx, deinflate=args.deinflate)
        if node is None:
            dropped.append({**inst, "drop_reasons": ["OBB 를 세울 프레임이 없다"]})
            counters[inst["kind"]] -= 1
            continue
        nodes.append(node)
    assert nodes, f"{args.video}: 살아남은 노드가 없다. dropped={len(dropped)}"

    # GT subject (TRUMANS 전용, D106). depth 점군 OBB 는 보이는 면만 있는 shell 이라 systematically
    # 작고 조준점이 카메라 쪽으로 쏠린다 — 9편 실측: 높이비 median 0.73, 조준 오차 median
    # 0.255 m (키의 16%). `--subject_source pointcloud`(기본) 면 예전 결과 비트 동일.
    gt_subject_node = None
    if args.subject_source == "gt":
        assert args.gravity_source == "gt", (
            "--subject_source gt 는 --gravity_source gt 를 전제한다 "
            "(blend→G 가 순수 yaw 여야 AABB 가 OBB 로 옮겨진다)")
        gt_sub = gt_subject(args.video, T_gw, num_frames, recon_root=args.gt_root)
        assert gt_sub is not None, (
            f"{args.video}: GT subject 를 못 찾았다 (probe_a<NN>.json 의 human_track). "
            f"--subject_source pointcloud 로 넘길 수 있다.")
        gt_subject_node = apply_gt_subject(nodes, gt_sub, cam_centers_g, num_frames)

    # GT 지면. `probe_a<NN>.json` 의 `floor_z` 는 anchor 프레임 사람 발 아래로 광선을 쏴서 잰
    # blend world 높이라, 바닥이 화면에 조금만 나와도 흔들리는 점군 2% 분위수보다 정확하다.
    ground_z_gt = None
    if args.ground_source == "gt":
        gt = gt_gravity(args.video, recon_root=args.gt_root, T_gw=T_gw)
        assert gt is not None and gt.get("ground_z_g") is not None, (
            f"{args.video}: GT 지면을 못 찾았다 (probe_a<NN>.json 의 floor_z). "
            f"--ground_source pointcloud 로 넘길 수 있다.")
        ground_z_gt = gt["ground_z_g"]

    # floor 마스크 지면 (in-the-wild 용, D62 재조준). GT 가 없는 코퍼스에서 2% 분위수를 대신한다.
    # `--floor_seg_root` 를 안 주면 None → **기존 결과 비트 동일**. GT 가 있으면 GT 가 이긴다.
    ground_floor, floor_points = None, 0
    if args.floor_seg_root and ground_z_gt is None:
        measured = floor_ground_z(recon, T_gw, S, args.floor_seg_root, args.video,
                                  num_frames_sampled=args.gravity_frames,
                                  quantile=args.floor_quantile)
        if measured is not None:
            ground_floor, floor_points = measured

    ground_override = ground_z_gt if ground_z_gt is not None else ground_floor
    edges, ground = build_relations(nodes, static_g, args.contact_u, args.wall_u,
                                    args.near_factor, args.wall_contact_frac,
                                    temporal=args.temporal_edges, num_frames=num_frames,
                                    ground_z_override=ground_override)
    # 어느 소스로 지면을 잡았는지 + 분위수와 얼마나 어긋났는지를 그래프 안에 박아 둔다
    # (gravity 가 `angle_to_ransac_deg` 를 남기는 것과 같은 규칙 — 조용히 바뀌면 안 된다).
    ground["source"] = ("gt" if ground_z_gt is not None
                        else "floor_seg" if ground_floor is not None else "pointcloud")
    if ground_floor is not None or ground_z_gt is not None:
        quantile_z = (ground_height(static_g) if len(static_g)
                      else min((n["obb"]["z_lo"] for n in nodes), default=0.0))
        ground["pointcloud_ground_z"] = round(float(quantile_z), 4)
        ground["delta_to_pointcloud_u"] = round(float(ground["ground_z"] - quantile_z), 4)
    if ground_floor is not None:
        ground["floor_points"] = floor_points
        ground["floor_quantile"] = args.floor_quantile
    reproj = check_reprojection(nodes, T_wg, K, cam_c2w)

    graph = {
        "format": GRAPH_FORMAT, "video": args.video,
        "num_frames": num_frames, "height": height, "width": width, "fps": float(recon["fps"]),
        # `mode` 를 반드시 싣는다 — 정의가 둘이라 그래프만 보고 어느 게이지인지 알 수 없으면
        # 옛 뱅크와 새 뱅크를 같은 표에 올려놓고도 못 알아챈다.
        "scale": {"S": S, "z_med_frame0": z_med, "parallax_ratio": plx,
                  "mode": args.scene_scale_mode, "stride": args.scene_scale_stride,
                  "unit": ("1 u = S DA3 units ("
                           + ("all-frame non-sky points, mean distance from first source camera"
                              if args.scene_scale_mode == "points_first_cam"
                              else "frame0 non-sky mean ray length") + ")")},
        "gravity": {"up_world": gravity["up_world"], "method": gravity["method"],
                    "confidence": gravity["confidence"], "inlier_ratio": gravity["inlier_ratio"],
                    "angle_to_cam_up_deg": gravity["angle_to_cam_up_deg"],
                    "attempts": gravity["attempts"],
                    # GT 를 쓸 때만: 추정이 얼마나 틀렸는지를 그래프 안에 박아 둔다. 나중에
                    # "이 chunk 는 예전에 몇 도 기울어 있었나"를 파일 하나로 답할 수 있게.
                    **({"estimated_method": gravity_estimated["method"],
                        "estimated_error_deg": round(float(np.degrees(np.arccos(np.clip(
                            np.dot(np.asarray(gravity["up_world"]),
                                   np.asarray(gravity_estimated["up_world"])), -1, 1)))), 3),
                        "gt_source": gravity.get("source"),
                        "floor_z_blend": gravity.get("floor_z_blend")}
                       if args.gravity_source == "gt" else {}),
                    # GeoCalib 을 쓸 때만: RANSAC 은 **검증자**로 강등되어 여기에만 남는다.
                    # 자동 전환은 하지 않는다 — 어긋난 사실을 그래프에 박아 두고 사람이 본다.
                    **({"spread_deg": geocalib["spread_deg"],
                        "num_frames_used": geocalib["num_frames"],
                        "num_kept": geocalib["num_kept"],
                        "prior_focal": geocalib["prior_focal"],
                        "ransac_method": gravity_estimated["method"],
                        "ransac_inlier_ratio": gravity_estimated["inlier_ratio"],
                        "angle_to_ransac_deg": round(angle_between(
                            geocalib["up_world"], gravity_estimated["up_world"]), 3),
                        # RANSAC 이 물었는데도 크게 어긋나면 둘 중 하나가 틀린 것이다.
                        "disagrees_with_ransac": bool(
                            gravity_estimated["method"].startswith("ground_ransac")
                            and angle_between(geocalib["up_world"],
                                              gravity_estimated["up_world"]) > 10.0)}
                       if geocalib is not None else {})},
        "frames": {"T_gw": T_gw, "T_wg": T_wg,
                   "convention": "world = cameras.npz frame (OpenCV, frame0 NOT identity); "
                                 "G: up=e_z, e_x=frame0 view horizontal, origin=frame0 cam center, unit=u"},
        "cameras": {"cam_c2w_world": cam_c2w, "K": K, "cam_centers_g": cam_centers_g},
        "ground": ground,
        "nodes": [{k: v for k, v in node.items() if k != "bbox_by_frame"} for node in nodes],
        "edges": edges,
        "diagnostics": {
            "reprojection": reproj, "merge_log": merge_log,
            "dropped_tracks": [{"kind": d["kind"], "track_id": int(d["track_id"]),
                                "label": d["label"], "reasons": d["drop_reasons"],
                                "max_area_frac": round(d["max_area_frac"], 5),
                                "visible_frames": d["num_visible_frames"]} for d in dropped],
            "num_static_points_sampled": int(len(static_world)),
        },
    }
    save_graph(output_path, graph)

    if args.overlay:
        # `obb_overlay.mp4` 는 **동적만** — 이미 사람이 승인 중인 그림이라 정적 상자를 끼워
        # 넣으면 비교가 안 된다. 정적 검수는 별도 파일 `obb_overlay_all.mp4` 로 본다.
        obb_overlay_video(path.join(output_folder, "vis", "obb_overlay.mp4"), recon["video"],
                          nodes, T_wg, K, cam_c2w, recon["fps"])
        if any(node["kind"] != "dyn" for node in nodes):
            obb_overlay_video(path.join(output_folder, "vis", "obb_overlay_all.mp4"),
                              recon["video"], nodes, T_wg, K, cam_c2w, recon["fps"],
                              kinds=("dyn", "stat"))
    if args.topdown:
        topdown_png(path.join(output_folder, "vis", "topdown.png"), static_g, nodes, cam_centers_g)

    print(f"\n{'':<4}{'S':>9}{'z_med':>9}{'parallax':>10}{'gravity':>20}{'conf':>7}{'incl':>8}")
    print(f"{'':<4}{S:>9.4f}{z_med:>9.3f}{plx:>10.4f}{gravity['method']:>20}"
          f"{gravity['confidence']:>7.2f}{gravity['angle_to_cam_up_deg']:>8.1f}")
    if args.gravity_source == "gt":
        print(f"    GT 로 교체 — 추정({gravity_estimated['method']})은 "
              f"{graph['gravity']['estimated_error_deg']:.1f}° 틀렸다")
    if geocalib is not None:
        grav = graph["gravity"]
        flag = "  ** RANSAC 과 어긋남 **" if grav["disagrees_with_ransac"] else ""
        print(f"    GeoCalib {geocalib['num_kept']}/{geocalib['num_frames']}프레임 "
              f"spread {geocalib['spread_deg']:.1f}° — 검증: RANSAC({grav['ransac_method']}, "
              f"inlier {grav['ransac_inlier_ratio']:.2f}) 와 "
              f"{grav['angle_to_ransac_deg']:.1f}° 차이{flag}")
    if ground_z_gt is not None:
        print(f"    ground_z GT {ground_z_gt:+.4f} u  vs 점군 2% 분위수 "
              f"{ground_height(static_g):+.4f} u")
    if gt_subject_node is not None:
        meta = gt_subject_node["subject_gt"]
        print(f"    subject GT {gt_subject_node['id']} (키 {meta['human_height_m']:.2f} m) — "
              f"shell 조준 오차 median {meta['estimated_aim_err_med_u'] * S:.3f} m "
              f"({meta['estimated_aim_err_med_u']:.4f} u)")
    print(f"\n{'id':<9}{'label':<16}{'mov':>4}{'l,w,h (u)':>22}{'d_ref':>8}{'az_span':>8}"
          f"{'frms':>6}{'reproj_px':>10}{'in_bbox':>8}{'size_x':>8}{'size_ok':>8}{'infl':>6}")
    for node, rep in zip(nodes, reproj):
        extent = ",".join(f"{v:.2f}" for v in node["obb"]["extent"])
        print(f"{node['id']:<9}{node['label'][:15]:<16}{'Y' if node['moving'] else 'n':>4}"
              f"{extent:>22}{node['viewing_distance']['d_ref']:>8.2f}{node['obs_az_span_deg']:>8.1f}"
              f"{node['num_visible_frames']:>6}{rep['reproj_px']:>10.1f}"
              f"{'Y' if rep['inside_bbox'] else 'NO':>8}{rep['size_ratio']:>8.2f}"
              f"{'Y' if rep['size_ok'] else 'NO':>8}{'Y' if node['extent_inflated'] else '-':>6}")
    if edges and not args.temporal_edges:
        print("\nedges: " + ", ".join(f"{e['src']}-{e['rel']}-{e['dst']}({e['dist_u']:.2f}u)" for e in edges))
    elif edges:
        print(f"\n{'edge':24}{'ref?':>6}{'dist_u':>9}{'min':>8}{'max':>8}{'near':>7}  intervals")
        for e in edges:
            print(f"{e['src'] + '-' + e['dst']:<24}{'Y' if e['near_at_ref'] else 'no':>6}"
                  f"{e['dist_u']:>9.2f}{e['dist_u_min']:>8.2f}{e['dist_u_max']:>8.2f}"
                  f"{e['near_frac']:>7.2f}  "
                  + " ".join(f"[{s}..{t}]" for s, t in e["near_intervals"]))
    for node in nodes:
        if node["supported_by"] or node["against_wall"]:
            print(f"  {node['id']}: supported_by={node['supported_by']} against_wall={node['against_wall']}")
        #    프레임별 판정이 ref 와 갈리면 그것 자체가 볼거리다 (물체가 들리거나 놓이는 순간).
        if args.temporal_edges:
            changes = sorted({s for s in node["supported_by_t"] if s != node["supported_by"]})
            if changes:
                print(f"     supported_by_t 가 ref 와 다른 프레임 "
                      f"{sum(1 for s in node['supported_by_t'] if s != node['supported_by'])}"
                      f"/{num_frames} -> {changes}")
    if graph["diagnostics"]["dropped_tracks"]:
        print(f"\ndropped {len(dropped)} tracks:")
        for d in graph["diagnostics"]["dropped_tracks"]:
            print(f"  {d['kind']}#{d['track_id']} {d['label']}: {'; '.join(d['reasons'])}")
    if merge_log:
        print("\nmerged: " + ", ".join(f"{m['absorbed']}->{m['kept']}({m['iou_3d']})" for m in merge_log))
    print(f"\n-> {output_path}")
    if not all(rep["inside_bbox"] for rep in reproj):
        print("!! 재투영이 bbox 밖인 노드가 있다 — 좌표 규약을 의심할 것 (w2c/c2w, up 부호, S, z vs ray)")
    if not all(rep["size_ok"] for rep in reproj):
        print("!! OBB 투영 크기가 마스크 대비 과대 — depth 누출을 의심할 것 (--depth_trim_quantile)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)  # None = <CinemaTraj>/out
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)         # None = <eval_data>/eval_data/seg_instances
    parser.add_argument("--seg_static_root", default=None, type=str)  # 정적 인스턴스 (있으면 자동 사용)

    parser.add_argument("--video", required=True, type=str)
    # S 의 정의 (2026-09-02 사용자 지시로 기본값이 바뀌었다). `points_first_cam` = 전 프레임
    # non-sky 점을 world 로 올려 **첫 소스 카메라**까지 거리의 평균. `frame0_ray` = 예전 frame0
    # 한 장 정의로, 옛 뱅크를 재현·대조할 때만 쓴다. 정의가 바뀌면 `u` 단위가 통째로 바뀌므로
    # 뱅크 재굽기가 따라온다 (OBB extent · 후보 거리 · 게이트 마진 `0.02·S` 전부).
    parser.add_argument("--scene_scale_mode", default="points_first_cam", choices=SCALE_MODES)
    parser.add_argument("--scene_scale_stride", default=1, type=int)   # 픽셀 서브샘플 (1 = 전부)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    # 중력 추정 — 전 프레임 정적 점을 다 쓰면 RANSAC 이 못 돈다 (4천만 점).
    parser.add_argument("--gravity_frames", default=5, type=int)
    parser.add_argument("--gravity_max_points", default=200000, type=int)
    # 중력축 소스.
    #   geocalib — **기본값** (D112, 사용자 지시 "GeoCalib 이 더 쓸만하니까 이걸로 다 쓰도록
    #              하고 불일치하는 것만 기록만"). 사이드카 필수(없으면 assert). RANSAC 은 계속
    #              돌지만 **검증 필드로만** 남는다 (`ransac_method` / `angle_to_ransac_deg` /
    #              `disagrees_with_ransac`) — 어긋나도 RANSAC 으로 갈아타지 않는다.
    #   auto     — 사이드카가 있으면 GeoCalib, 없으면 RANSAC. **D112 이전 기본값**이고, 그래서
    #              vista 51편 중 `jogging-woman` 한 편만 조용히 RANSAC 으로 떨어졌다 (나머지
    #              50편 geocalib). 예전 코퍼스를 되만들 때만 쓸 것.
    #   ransac   — 점군 지면 RANSAC 만. 사이드카를 무시하므로 D98 이전 결과와 비트 동일
    #   gt       — TRUMANS blend world (z-up). 추정 없음
    parser.add_argument("--gravity_source", default="geocalib",
                        choices=("ransac", "gt", "geocalib", "auto"))
    parser.add_argument("--ground_source", default="pointcloud", choices=("pointcloud", "gt"))
    # SAM3 가 `floor`/`ground` 로 잡은 마스크 루트 (`sam3_static_instances.py --nouns_field surface`).
    # **노드로는 안 올린다** — 용도는 `ground.ground_z` 하나뿐이다 (§floor_ground_z).
    # None(기본) = 예전 결과 비트 동일. `--ground_source gt` 가 있으면 그쪽이 이긴다.
    parser.add_argument("--floor_seg_root", default=None, type=str)
    # floor 픽셀 z 의 분위수. 지목된 픽셀만 보므로 0.5(median) 로 충분하다.
    parser.add_argument("--floor_quantile", default=0.5, type=float)
    # subject OBB·조준점(track center) 소스. gt = probe 의 human_track (blend mesh AABB, D106).
    parser.add_argument("--subject_source", default="pointcloud", choices=("pointcloud", "gt"))
    parser.add_argument("--gt_root", default=GT_ROOT_DEFAULT, type=str)
    # 실내처럼 바닥이 화면의 15% 를 못 채우는 씬 구제용. 각도·카메라위치 조건은 그대로 걸고
    # inlier_ratio 하한만 이 값으로 낮춘다. 0 = 꺼짐(51편 코퍼스가 돈 그대로).
    parser.add_argument("--weak_inlier_ratio", default=0.0, type=float)

    # track 기각 — 셋 다 서로 다른 실패를 잡는다 (§instances.py docstring)
    parser.add_argument("--min_area_frac", default=0.002, type=float)
    parser.add_argument("--min_frames", default=5, type=int)
    parser.add_argument("--min_score", default=0.3, type=float)
    parser.add_argument("--max_points_per_frame", default=20000, type=int)
    # 마스크 경계 depth 누출 제거. 0 으로 두면 OBB 가 깊이축으로 4배 부푼다 (§instances.trim_depth_tail)
    parser.add_argument("--depth_trim_quantile", default=0.01, type=float)
    # 마스크 depth 가 두 덩어리일 때 중앙값 덩어리만 남긴다. bins 0 이면 끔 = 예전 동작.
    parser.add_argument("--depth_mode_bins", default=40, type=int)
    parser.add_argument("--depth_mode_gap_frac", default=0.005, type=float)
    # 저시차 깊이축 부풀림 보정. **기본 off** — camel dyn_1 에서 실제 몸통 길이(0.16u)를 0.05u 로
    # 날려버렸다. 부풀림 판정은 아래 표의 size_x(투영 OBB / 마스크 bbox) 로 한다.
    parser.add_argument("--deinflate", action="store_true", default=False)
    parser.add_argument("--no_deinflate", dest="deinflate", action="store_false")

    parser.add_argument("--merge_voxel_u", default=0.02, type=float)
    parser.add_argument("--merge_iou", default=0.5, type=float)

    # 인스턴스 폭발 상한 (`cap_instances`). 0 이하면 상한 없음 = 예전 동작.
    parser.add_argument("--per_keyword_top_k", default=5, type=int)   # goat 의 rock 31개용
    parser.add_argument("--max_dyn_nodes", default=6, type=int)       # car-roundabout 차 69대용
    parser.add_argument("--max_stat_nodes", default=10, type=int)     # parkour 정적 56개용
    parser.add_argument("--prescreen_factor", default=3, type=int)    # lift 전 상한 = 상한 x 이것

    parser.add_argument("--contact_u", default=0.05, type=float)
    parser.add_argument("--wall_u", default=0.08, type=float)
    # 벽 접촉 임계를 물체 크기에 비례하게. `1e9` 면 wall_u 절대 임계만 쓰던 예전 동작
    # (그 경우 camel 6개 노드가 전부 against_wall=True 다 — §relations.py ③)
    parser.add_argument("--wall_contact_frac", default=0.15, type=float)
    parser.add_argument("--near_factor", default=1.5, type=float)
    #    엣지에 프레임별 거리를 싣는다 (D70). off 면 예전 시간 불변 엣지 그대로.
    parser.add_argument("--temporal_edges", action="store_true", default=False)
    parser.add_argument("--no_temporal_edges", dest="temporal_edges", action="store_false")

    parser.add_argument("--overlay", action="store_true", default=True)
    parser.add_argument("--no_overlay", dest="overlay", action="store_false")
    parser.add_argument("--topdown", action="store_true", default=True)
    parser.add_argument("--no_topdown", dest="topdown", action="store_false")

    main(parser.parse_args())
