"""TRUMANS-Lite 뱅크의 **프레이밍**을 잰다 — 보이는 subject OBB 면적 / 전체 OBB 면적.

왜 필요한가 (사용자 지적): 앞서 뱅크 품질을 "피사체가 화면에서 차지하는 면적"으로 봤는데
그건 **shot size** 지 품질이 아니다. 멀리서 잡은 wide shot 은 면적이 작아도 좋은 그림이고,
문제는 반대쪽 — 카메라가 너무 붙어서 **사람이 화면 밖으로 잘리거나** 앞에 가구가 끼는 것이다.
그래서 판정 축을 절대 면적이 아니라 **보이는 비율**로 바꾼다:

    framing = (안 가려진 픽셀) / (**화면에 투영된** OBB 면적)          # = occl_keep

분모가 **화면 안** bbox 인 게 요점이다 (사용자 지정). 전체 hull 을 분모로 쓰면 close-up 에서
박스가 화면 밖으로 나간 것까지 "가려졌다"로 읽힌다 — 클로즈업은 원래 그런 그림이므로 그건
가림이 아니다. 잘림은 `crop_keep` 으로 **따로** 찍되 `framing` 에 곱하지 않는다.

## 두 열의 계산

- `framing` (= `occl_keep`) — hull 안 **화면 내** 픽셀마다 카메라 광선을 AABB 에 쏴 입사 깊이
  `t_enter` 를 구하고, 렌더 depth 가 `t_enter - eps` 보다 **앞**이면 가려진 것으로 센다. 사람
  자신의 표면은 박스 **안**에 있으므로 (depth >= t_enter) 자기 가림으로 안 잡힌다.
- `crop_keep` (진단 전용) — 이미지 사각형으로 Sutherland-Hodgman 클리핑한 면적 / 원본 hull 면적.
  **래스터가 아니라 해석적 면적**이라 화면 밖으로 크게 벗어난 hull 도 분모가 제대로 커진다
  (cv2.fillConvexPoly 로 세면 분모까지 화면에 잘려서 잘림이 안 보인다).
  `--include_crop` 이면 `framing = crop_keep x occl_keep` 이던 예전 정의로 되돌린다.

편향: OBB 는 사람 실루엣보다 크다 (팔 벌린 AABB 는 대부분 공기). 그래서 `crop_keep` 은 잘림을
**과대**, `occl_keep` 은 가림을 **과대** 보고하는 쪽이다. 순위 매기는 용도지 절대 기준이 아니다.
(`audit_bank_geometry.py` 의 OBB 가림과 같은 편향, 같은 단서.)

카메라 near plane 뒤로 박스가 걸치면 투영이 뒤집히므로 12 edge 를 `z=z_near` 로 3D 클리핑한 뒤
투영한다. 그런 프레임 수는 `nearclip` 열에 따로 센다 — 카메라가 사람 몸통에 박혔다는 신호다.

출력: `<out>/lite_framing.csv` (clip 당 1행) + `<out>/lite_framing_frames.csv` (프레임당 1행)
+ 요약표. **감사(audit)지 게이트가 아니다** — 행을 지우지 않는다.

env: `vista4d` (OpenEXR)

예시:
    python scripts/audit_lite_framing.py
    python scripts/audit_lite_framing.py --videos tru_00add26c_a00 tru_0aa05d5a_a03 --verbose
"""
import csv
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import cv2
import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)
VISTA4D_ROOT = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"
if VISTA4D_ROOT not in sys.path:
    sys.path.insert(0, VISTA4D_ROOT)

from scene_graph.lift import project_points                                     # noqa: E402
from utils.media import load_depths                                             # noqa: E402

EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/TRUMANS-Lite"
AABB_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def aabb_corners(lo: np.ndarray, hi: np.ndarray):
    """(8,3). `scene_graph.obb.obb_corners` 와 같은 부호 순서라 `AABB_EDGES` 가 짝이 맞는다."""
    signs = np.array([[sx, sy, sz] for sz in (0, 1) for sy in (0, 1) for sx in (0, 1)], dtype=float)
    return lo + signs * (hi - lo)


def near_clipped_corners(corners: np.ndarray, cam_c2w: np.ndarray, z_near: float):
    """박스를 `z_cam >= z_near` 반공간으로 자른 뒤 남는 꼭짓점들 (world).

    볼록 다면체 ∩ 반공간의 꼭짓점 = (조건을 만족하는 원래 꼭짓점) ∪ (경계를 지나는 edge 의 교점).
    이걸 안 하면 카메라 뒤 꼭짓점이 부호 반전돼 투영되어 hull 이 화면 전체를 덮는다.
    """
    w2c = np.linalg.inv(cam_c2w)
    z = corners @ w2c[2, :3] + w2c[2, 3]
    keep = [corners[i] for i in range(8) if z[i] >= z_near]
    for i, j in AABB_EDGES:
        if (z[i] >= z_near) != (z[j] >= z_near):
            alpha = (z_near - z[i]) / (z[j] - z[i])
            keep.append(corners[i] + alpha * (corners[j] - corners[i]))
    return np.asarray(keep, dtype=float) if keep else np.zeros((0, 3))


def polygon_area(poly: np.ndarray):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def clip_to_rect(poly: np.ndarray, width: int, height: int):
    """Sutherland-Hodgman. 볼록 다각형 ∩ 이미지 사각형 — 면적을 **해석적으로** 남기려고."""
    output = poly
    for axis, limit, keep_greater in ((0, 0.0, True), (0, float(width), False),
                                      (1, 0.0, True), (1, float(height), False)):
        if len(output) == 0:
            return np.zeros((0, 2))
        inside = (output[:, axis] >= limit) if keep_greater else (output[:, axis] <= limit)
        clipped = []
        for i in range(len(output)):
            j = (i + 1) % len(output)
            if inside[i]:
                clipped.append(output[i])
            if inside[i] != inside[j]:
                denominator = output[j, axis] - output[i, axis]
                alpha = (limit - output[i, axis]) / (denominator if abs(denominator) > 1e-12 else 1e-12)
                clipped.append(output[i] + alpha * (output[j] - output[i]))
        output = np.asarray(clipped, dtype=float)
    return output


def ray_aabb_enter(K: np.ndarray, cam_c2w: np.ndarray, uv: np.ndarray,
                   lo: np.ndarray, hi: np.ndarray):
    """픽셀 중심 광선의 AABB 입사 깊이 (=카메라 z, depth EXR 과 같은 z-planar 단위).

    카메라 좌표 방향을 `d_z = 1` 로 두면 `t` 가 그대로 z-depth 다 — 정규화하면 안 된다.
    """
    dirs_cam = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0],
                         (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], axis=1)
    dirs = dirs_cam @ cam_c2w[:3, :3].T
    origin = cam_c2w[:3, 3]
    safe = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    t_lo, t_hi = (lo - origin) / safe, (hi - origin) / safe
    return np.maximum(np.minimum(t_lo, t_hi), 0.0).max(axis=1)


def frame_framing(lo, hi, cam_c2w, K, depth, z_near, eps, include_crop=False):
    """한 프레임의 (crop_keep, occl_keep, framing, nearclip, area_total_px).

    `framing` 의 분모는 **화면에 투영된** bbox 다 — 화면 밖으로 나간 몫은 안 센다.
    `include_crop` 이면 예전 정의(`crop_keep x occl_keep`)로 되돌린다.
    화면 안에 bbox 가 한 픽셀도 없으면 `framing`/`occl_keep` 은 nan (분모가 없다).
    """
    height, width = depth.shape
    corners = near_clipped_corners(aabb_corners(lo, hi), cam_c2w, z_near)
    nearclip = int(len(corners) != 8)
    nan = float("nan")
    if len(corners) < 3:
        return 0.0, nan, nan, 1, 0.0                      # 박스가 통째로 카메라 뒤
    uv, _ = project_points(corners, K, cam_c2w)
    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2).astype(float)
    total = polygon_area(hull)
    if total <= 1e-6:
        return 0.0, nan, nan, nearclip, 0.0
    inside = clip_to_rect(hull, width, height)
    crop_keep = min(1.0, polygon_area(inside) / total)

    silhouette = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(silhouette, hull.round().astype(np.int32), 1)
    ys, xs = np.nonzero(silhouette)
    if len(ys) == 0:
        return crop_keep, nan, nan, nearclip, total       # 화면 안에 bbox 가 없다
    pixels = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(float)
    t_enter = ray_aabb_enter(K, cam_c2w, pixels, lo, hi)
    scene = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(scene) & (scene > 0)
    occluded = valid & (scene < t_enter - eps)
    occl_keep = 1.0 - float(occluded.sum()) / float(len(ys))
    return (crop_keep, occl_keep, (crop_keep * occl_keep if include_crop else occl_keep),
            nearclip, total)


def preview_clip(video, rgb, track, cam_c2w, intrinsics, depths, args, rows):
    """판정 근거를 **그림으로** 남긴다 — 청록 hull, 화면 밖으로 나간 변, 가려진 픽셀 빨강.

    숫자만 보면 crop 0.44 가 "머리가 잘렸다"인지 "발이 조금 나갔다"인지 구분이 안 된다.
    hull 을 화면 안으로 접어 그리므로(테두리에 붙는 변이 곧 잘린 곳) 어디가 나갔는지 보인다.
    """
    import imageio.v3 as iio

    frames = []
    for f in range(len(rows)):
        fx, fy, cx, cy = [float(v) for v in intrinsics[f]]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        lo = np.asarray(track[f]["min"], dtype=float)
        hi = np.asarray(track[f]["max"], dtype=float)
        corners = near_clipped_corners(aabb_corners(lo, hi), cam_c2w[f], args.z_near)
        image = rgb[f].copy()
        height, width = depths[f].shape
        if len(corners) >= 3:
            uv, _ = project_points(corners, K, cam_c2w[f])
            hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
            silhouette = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(silhouette, hull.round().astype(np.int32), 1)
            ys, xs = np.nonzero(silhouette)
            if len(ys):
                pixels = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(float)
                t_enter = ray_aabb_enter(K, cam_c2w[f], pixels, lo, hi)
                scene = depths[f][ys, xs].astype(np.float64)
                occluded = np.isfinite(scene) & (scene > 0) & (scene < t_enter - args.occlusion_eps)
                image[ys[occluded], xs[occluded]] = (
                    0.45 * image[ys[occluded], xs[occluded]]
                    + 0.55 * np.array([230, 40, 40])).astype(np.uint8)
            inside = clip_to_rect(hull.astype(float), width, height)
            if len(inside) >= 3:
                cv2.polylines(image, [inside.round().astype(np.int32)], True, (0, 235, 235), 2)
        row = rows[f]
        cv2.putText(image, f"{video}  f{f:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"framing {row['framing']:.2f}  (crop {row['crop_keep']:.2f})",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
        frames.append(image)
    makedirs(path.join(args.out, "framing_preview"), exist_ok=True)
    target = path.join(args.out, "framing_preview", f"{video}.mp4")
    iio.imwrite(target, np.stack(frames), fps=args.preview_fps, codec="libx264",
                quality=6, macro_block_size=1)
    return target


def audit_clip(entry, args):
    recording, action = entry["recording"], entry["action"]
    probe = path.join(args.work, recording, f"probe_a{action:02d}.json")
    poses = path.join(args.work, recording, f"poses_a{action:02d}.npz")
    scene = path.join(args.eval_data, "eval_data", "recon_and_seg", entry["video"])
    if not (path.exists(probe) and path.exists(poses) and path.isdir(scene)):
        return None, []

    with open(probe, encoding="utf-8") as file:
        track = json.load(file)["human_track"]
    cam_c2w = np.load(poses)["cam_c2w"]                   # world (재앵커 전) — depth 와 짝이 맞는다
    intrinsics = np.load(path.join(scene, "cameras.npz"))["intrinsics"]
    depths = load_depths(path.join(scene, "depths"), dtype=np.float32, desc=entry["video"])
    frames = min(len(track), len(cam_c2w), len(depths))

    rows = []
    for f in range(frames):
        fx, fy, cx, cy = [float(v) for v in intrinsics[f]]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        crop, occl, framing, nearclip, total = frame_framing(
            np.asarray(track[f]["min"], dtype=float), np.asarray(track[f]["max"], dtype=float),
            cam_c2w[f], K, depths[f], args.z_near, args.occlusion_eps, args.include_crop)
        rows.append({"video": entry["video"], "frame": f, "crop_keep": round(crop, 4),
                     "occl_keep": round(occl, 4), "framing": round(framing, 4),
                     "nearclip": nearclip,
                     "obb_area_frac": round(total / (depths.shape[1] * depths.shape[2]), 4)})

    framings = np.array([r["framing"] for r in rows])
    crops = np.array([r["crop_keep"] for r in rows])
    occls = np.array([r["occl_keep"] for r in rows])
    summary = {"video": entry["video"], "recording": recording[:8], "action": action,
               "preset": entry.get("preset", ""),
               "radius": round(float(entry.get("candidate", {}).get("radius", float("nan"))), 2),
               "azimuth": round(float(entry.get("candidate", {}).get("azimuth_deg",
                                                                     float("nan"))), 1),
               # 화면 안에 bbox 가 없는 프레임은 분모가 없어서 nan — 평균에서 뺀다.
               "framing_mean": round(float(np.nanmean(framings)), 4),
               "framing_min": round(float(np.nanmin(framings)), 4),
               "crop_mean": round(float(crops.mean()), 4),
               "occl_mean": round(float(np.nanmean(occls)), 4),
               "offscreen_frames": int(np.isnan(framings).sum()),
               "frames_below": int((framings < args.min_framing).sum()),
               "nearclip_frames": int(sum(r["nearclip"] for r in rows)),
               "obb_area_mean": round(float(np.mean([r["obb_area_frac"] for r in rows])), 4)}
    if args.preview:
        import imageio.v3 as iio
        rgb = np.stack(list(iio.imiter(path.join(scene, "video.mp4"))))[:frames]
        summary["preview"] = preview_clip(entry["video"], rgb, track, cam_c2w, intrinsics,
                                          depths, args, rows)
    return summary, rows


def main(args):
    with open(args.bank_manifest, encoding="utf-8") as file:
        manifest = json.load(file)
    entries = [e for e in manifest["entries"] if e.get("status") == "ok"]
    if args.videos:
        entries = [e for e in entries if e["video"] in set(args.videos)]
    assert entries, "감사할 clip 이 없다 (status=ok 인 항목 0개)"

    print(f"{'bank':<16}{args.bank_manifest}")
    print(f"{'clips':<16}{len(entries)}")
    print(f"{'min_framing':<16}{args.min_framing}   occlusion_eps {args.occlusion_eps} m\n")

    summaries, frame_rows = [], []
    for entry in entries:
        summary, rows = audit_clip(entry, args)
        if summary is None:
            print(f"  SKIP {entry['video']:24s} (probe/poses/scene 누락)")
            continue
        summaries.append(summary)
        frame_rows += rows
        if args.verbose:
            print(f"  {summary['video']:24s} framing {summary['framing_mean']:.3f} "
                  f"(crop {summary['crop_mean']:.3f} x occl {summary['occl_mean']:.3f})  "
                  f"r {summary['radius']:.1f}")

    makedirs(args.out, exist_ok=True)
    clip_csv = path.join(args.out, "lite_framing.csv")
    with open(clip_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    frame_csv = path.join(args.out, "lite_framing_frames.csv")
    with open(frame_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    framing = np.array([s["framing_mean"] for s in summaries])
    crop = np.array([s["crop_mean"] for s in summaries])
    occl = np.array([s["occl_mean"] for s in summaries])
    order = np.argsort(framing)
    print("\n" + "-" * 78)
    print(f"{'framing_mean':<20}median {np.median(framing):.3f}   "
          f"min {framing.min():.3f}   max {framing.max():.3f}")
    print(f"{'crop_keep':<20}median {np.median(crop):.3f}   min {crop.min():.3f}")
    print(f"{'occl_keep':<20}median {np.median(occl):.3f}   min {occl.min():.3f}")
    print(f"{'< min_framing':<20}{int((framing < args.min_framing).sum())} / {len(framing)} clips")
    print(f"{'nearclip 있는 clip':<19}"
          f"{sum(1 for s in summaries if s['nearclip_frames'] > 0)} / {len(summaries)}")
    print("\n최악 10:")
    for index in order[:10]:
        s = summaries[index]
        print(f"  {s['video']:24s} {s['framing_mean']:.3f}  crop {s['crop_mean']:.3f}  "
              f"occl {s['occl_mean']:.3f}  r {s['radius']:.1f}  {s['preset']}")
    print("최선 10:")
    for index in order[::-1][:10]:
        s = summaries[index]
        print(f"  {s['video']:24s} {s['framing_mean']:.3f}  crop {s['crop_mean']:.3f}  "
              f"occl {s['occl_mean']:.3f}  r {s['radius']:.1f}  {s['preset']}")
    print(f"\n{'->':<20}{clip_csv}")
    print(f"{'->':<20}{frame_csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bank_manifest",
                        default=path.join(CINEMATRAJ_ROOT, "out", "trumans_lite_bank",
                                          "bank_manifest.json"), type=str)
    parser.add_argument("--videos", default=[], nargs="*", type=str)   # 비우면 뱅크 전체
    parser.add_argument("--min_framing", default=0.70, type=float)     # 요약 카운트용 임계 (게이트 아님)
    parser.add_argument("--occlusion_eps", default=0.02, type=float)   # m. depth 노이즈 여유
    parser.add_argument("--z_near", default=0.05, type=float)          # m. near plane 3D 클리핑
    # framing 분모를 화면 안 bbox 로 (기본). 켜면 예전 정의 `crop_keep x occl_keep` 로 되돌린다 —
    # close-up 에서 화면 밖으로 나간 몫까지 가림으로 세므로 순위가 shot size 를 따라간다.
    parser.add_argument("--include_crop", action="store_true", default=False)
    parser.add_argument("--no_include_crop", dest="include_crop", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    # 판정 근거 영상. 전량에 켜면 95편이라 오래 걸린다 — `--videos` 로 골라서 쓰는 용도.
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview_fps", default=12, type=int)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--work", default=path.join(CINEMATRAJ_ROOT, "out", "trumans_recon"),
                        type=str)
    parser.add_argument("--out", default=path.join(CINEMATRAJ_ROOT, "out", "trumans_lite_bank"),
                        type=str)
    main(parser.parse_args())
