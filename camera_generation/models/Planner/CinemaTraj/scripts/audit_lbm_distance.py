"""원본 LBM 이 고른 카메라의 **거리**를 재서 Lite 뱅크의 반경과 같은 자에 올린다.

왜 필요한가: Lite 뱅크는 카메라를 사람 앵커에서 반경 `r ∈ {1.5, 2.2, 3.0, 4.0} m` 로 뽑는데,
LBM 출력에는 반경 개념이 없다 — 있는 건 `distance_label` ("medium shot"/"close-up") 과
후보 id 안의 `d0`/`d5` 같은 **버킷 토큰**뿐이고 미터 값이 아니다. 그래서 두 파이프라인의
"얼마나 멀리서 잡았나"를 직접 비교할 수가 없다. 여기서 실제 미터를 만들어 붙인다:

    d_focus = |cam_loc - primary_focus_center|        # LBM 이 실제로 겨눈 물체까지
    d_human = |cam_loc - human_center|                # 사람까지 (Lite 반경과 같은 양)

두 값을 **따로** 재는 게 요점이다. LBM 의 `primary_focus_id` 는 40/42 가 소품(book/oven/phone)
이라 `d_focus` 는 "책까지 몇 m" 이고, 우리가 궁금한 "사람이 왜 잘리나"는 `d_human` 이 답한다.
`d_focus` 만 보면 LBM 이 적당한 거리에 있는 것처럼 보인다.

좌표: 전부 **blend world (미터)**. LBM `start_transform.location` 과 Lite `poses_a*.npz` 의
`cam_c2w[:,:3,3]`, `probe_a*.json` 의 `human_track[*].center` 가 같은 프레임이다.

사람 위치는 shot 이 가리키는 action clip 49프레임의 **median center** 를 쓴다. LBM 의 shot_id 는
TRUMANS action id 와 같고 (`trumans_to_lbm_demo.py:277`), Lite 의 `manifest_a<NN>.json.action.id`
가 그 id 를 들고 있어 두 쪽을 이어준다.

주의: `d_human` 의 분모(사람 중심)는 앉은 자세면 z 가 낮아진다. 그래서 절대값보다 **분포와
Lite 대비 위치**를 보는 용도다.

출력: `<out>/lbm_distance.csv` (카메라 1대당 1행) + `<out>/lite_distance.csv` + 요약표.

예시:
    python scripts/audit_lbm_distance.py
    python scripts/audit_lbm_distance.py --no_lite --verbose
"""
import csv
import json
import re
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

LBM_OUT_DEFAULT = ("/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/"
                   "Look-Before-Move/Cinematographer/output")
RECON_DEFAULT = ("/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/"
                 "CinemaTraj/out/trumans_recon")
BUCKET_RE = re.compile(r"_d(\d+)_\d+$")          # 후보 id 꼬리의 거리 버킷 토큰
SENSOR_MM = 36.0                                 # Blender 기본값, LBM 은 sensor_* 를 안 건드린다
LITE_FX_PX, LITE_W, LITE_H = 666.6666666666666, 960, 540


def frame_extent(distance: float, lens_mm: float, width: int, height: int):
    """피사체 거리에서 화면이 덮는 실제 크기 (m). sensor_fit AUTO -> 긴 변에 36 mm 가 걸린다.

    거리만으로는 "가깝다/멀다"를 못 가린다 — 200 mm 로 3 m 에서 잡으면 1.5 m 에서 25 mm 로
    잡은 것보다 훨씬 타이트하다. 사람이 잘리는지는 이 세로 폭이 결정한다.
    """
    long_side = max(width, height)
    mm_per_px = SENSOR_MM / long_side
    return (distance * (width * mm_per_px) / lens_mm,
            distance * (height * mm_per_px) / lens_mm)


def human_center_by_action(recon_root: str, recording: str):
    """action id -> (human_center_median(3,), floor_z, human_height). probe/manifest 쌍에서."""
    table = {}
    for manifest_path in sorted(glob(path.join(recon_root, recording, "manifest_a*.json"))):
        manifest = json.load(open(manifest_path))
        tag = path.basename(manifest_path)[len("manifest_"):-len(".json")]
        probe_path = path.join(recon_root, recording, f"probe_{tag}.json")
        if not path.exists(probe_path):
            continue
        probe = json.load(open(probe_path))
        centers = np.array([entry["center"] for entry in probe["human_track"]], dtype=float)
        spans = np.array([np.array(e["max"]) - np.array(e["min"])
                          for e in probe["human_track"]], dtype=float)
        table[int(manifest["action"]["id"])] = {
            "center": np.median(centers, axis=0),
            "floor_z": float(probe["floor_z"]),
            "height": float(np.median(spans[:, 2])),      # 그 action 동안의 실제 세로 폭
            "tag": tag,
        }
    return table


def lbm_rows(lbm_root: str, recon_root: str, verbose: bool):
    rows = []
    for handoff_path in sorted(glob(path.join(lbm_root, "trumans_*_x15", "outputs",
                                              "camera_handoff_v1.json"))):
        run = path.basename(path.dirname(path.dirname(handoff_path)))
        stub = run[len("trumans_"):-len("_x15")]
        recordings = [path.basename(d) for d in glob(path.join(recon_root, f"{stub}-*"))]
        assert len(recordings) == 1, f"{stub} 에 대응하는 recon 폴더가 {recordings}"
        humans = human_center_by_action(recon_root, recordings[0])
        handoff = json.load(open(handoff_path))
        for shot in handoff["shots"]:
            for cam in shot["cameras"]:
                loc = np.array(cam["start_transform"]["location"], dtype=float)
                end = np.array(cam["end_transform"]["location"], dtype=float)
                contract = cam.get("shot_contract", {}).get("start_frame_contract", {})
                layout = contract.get("start_subject_layout") or {}
                focus_center = layout.get("center")
                focus_extent = layout.get("extent")
                d_focus = (float(np.linalg.norm(loc - np.array(focus_center, dtype=float)))
                           if focus_center else float("nan"))
                human = humans.get(int(cam["shot_id"]))
                d_human = (float(np.linalg.norm(loc - human["center"]))
                           if human is not None else float("nan"))
                bucket = BUCKET_RE.search(cam.get("final_render_source_candidate_id") or "")
                lens = float(cam.get("lens_mm") or 0.0) or float("nan")
                brief = cam.get("render_brief", {})
                frame_w, frame_h = frame_extent(d_human, lens,
                                                int(brief.get("frame_width", LITE_W)),
                                                int(brief.get("frame_height", LITE_H)))
                rows.append({
                    "run": stub,
                    "shot_id": cam["shot_id"],
                    "camera": cam["camera_name"],
                    "focus_id": cam.get("primary_focus_id", ""),
                    "distance_label": cam.get("distance_label", ""),
                    "preset": cam.get("trajectory_plan", {}).get("preset_name", ""),
                    "lens_mm": lens,
                    "d_focus": d_focus,
                    "d_human": d_human,
                    "frame_w_m": frame_w,
                    "frame_h_m": frame_h,
                    "human_h": human["height"] if human is not None else float("nan"),
                    "person_fill_v": ((human["height"] / frame_h)
                                      if human is not None and np.isfinite(frame_h) and frame_h > 0
                                      else float("nan")),
                    "d_human_end": (float(np.linalg.norm(end - human["center"]))
                                    if human is not None else float("nan")),
                    "focus_diag": (float(np.linalg.norm(focus_extent)) if focus_extent
                                   else float("nan")),
                    "cam_z": float(loc[2]),
                    "human_floor_z": human["floor_z"] if human is not None else float("nan"),
                    "bucket": int(bucket.group(1)) if bucket else -1,
                    "area_ratio": cam.get("quality_qc", {}).get("primary_area_ratio",
                                                                float("nan")),
                })
                if verbose:
                    print(f"  {stub} shot{cam['shot_id']:>3} {cam.get('primary_focus_id',''):<22}"
                          f" d_focus {d_focus:6.2f}  d_human {d_human:6.2f}")
    return rows


def lite_rows(recon_root: str):
    rows = []
    for manifest_path in sorted(glob(path.join(recon_root, "*", "manifest_a*.json"))):
        manifest = json.load(open(manifest_path))
        folder = path.dirname(manifest_path)
        tag = path.basename(manifest_path)[len("manifest_"):-len(".json")]
        probe = json.load(open(path.join(folder, f"probe_{tag}.json")))
        poses = np.load(path.join(folder, f"poses_{tag}.npz"))
        centers = np.array([entry["center"] for entry in probe["human_track"]], dtype=float)
        cams = poses["cam_c2w"][:, :3, 3]
        dist = np.linalg.norm(cams - centers, axis=1)
        candidate = manifest["source_camera"]["candidate"]
        lens = LITE_FX_PX * SENSOR_MM / max(LITE_W, LITE_H)     # fx(px) -> 35 mm 환산 초점거리
        frame_w, frame_h = frame_extent(float(dist.mean()), lens, LITE_W, LITE_H)
        spans = np.array([np.array(e["max"]) - np.array(e["min"])
                          for e in probe["human_track"]], dtype=float)
        human_h = float(np.median(spans[:, 2]))
        rows.append({
            "video": manifest["video"],
            "preset": manifest["source_camera"]["preset"],
            "radius": float(candidate["radius"]),
            "elevation_deg": float(candidate["elevation_deg"]),
            "d_human_mean": float(dist.mean()),
            "d_human_min": float(dist.min()),
            "d_human_max": float(dist.max()),
            "lens_mm": lens,
            "frame_w_m": frame_w,
            "frame_h_m": frame_h,
            "human_h": human_h,
            "person_fill_v": human_h / frame_h,
            "depth_p50": float(manifest["render"]["depth_p50"]),
        })
    return rows


def quantiles(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return (float("nan"),) * 5
    return (float(arr.min()), float(np.percentile(arr, 25)), float(np.median(arr)),
            float(np.percentile(arr, 75)), float(arr.max()))


def write_csv(rows, out_path):
    if not rows:
        return
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = ArgumentParser()
    parser.add_argument("--lbm_root", default=LBM_OUT_DEFAULT)     # Cinematographer/output
    parser.add_argument("--recon_root", default=RECON_DEFAULT)     # Lite 쪽 probe/manifest/poses
    parser.add_argument("--out", default=path.join(path.dirname(path.dirname(
        path.abspath(__file__))), "out", "trumans_lite_bank"))
    parser.add_argument("--lite", dest="lite", action="store_true", default=True)
    parser.add_argument("--no_lite", dest="lite", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    makedirs(args.out, exist_ok=True)
    lbm = lbm_rows(args.lbm_root, args.recon_root, args.verbose)
    write_csv(lbm, path.join(args.out, "lbm_distance.csv"))

    print(f"\nLBM 채택 카메라 {len(lbm)} 대")
    print(f"{'열':<16}{'min':>8}{'p25':>8}{'median':>8}{'p75':>8}{'max':>8}")
    for key in ("d_focus", "d_human", "lens_mm", "frame_h_m", "person_fill_v",
                "focus_diag", "area_ratio"):
        stats = quantiles([r[key] for r in lbm])
        print(f"{key:<16}" + "".join(f"{v:8.2f}" for v in stats))

    print(f"\n{'distance_label':<16}{'n':>4}{'d_focus_med':>13}{'d_human_med':>13}")
    labels = sorted({r["distance_label"] for r in lbm})
    for label in labels:
        sub = [r for r in lbm if r["distance_label"] == label]
        print(f"{label:<16}{len(sub):>4}{np.nanmedian([r['d_focus'] for r in sub]):13.2f}"
              f"{np.nanmedian([r['d_human'] for r in sub]):13.2f}")

    print(f"\n{'bucket(d?)':<16}{'n':>4}{'d_focus_med':>13}{'d_human_med':>13}")
    for bucket in sorted({r["bucket"] for r in lbm}):
        sub = [r for r in lbm if r["bucket"] == bucket]
        print(f"{'d' + str(bucket):<16}{len(sub):>4}{np.nanmedian([r['d_focus'] for r in sub]):13.2f}"
              f"{np.nanmedian([r['d_human'] for r in sub]):13.2f}")

    if args.lite:
        lite = lite_rows(args.recon_root)
        write_csv(lite, path.join(args.out, "lite_distance.csv"))
        print(f"\nLite 뱅크 clip {len(lite)} 편")
        print(f"{'열':<16}{'min':>8}{'p25':>8}{'median':>8}{'p75':>8}{'max':>8}")
        for key in ("radius", "d_human_mean", "lens_mm", "frame_h_m", "person_fill_v",
                    "depth_p50"):
            stats = quantiles([r[key] for r in lite])
            print(f"{key:<16}" + "".join(f"{v:8.2f}" for v in stats))
        print(f"\n{'radius':<16}{'n':>4}{'d_human_mean_med':>18}")
        for radius in sorted({r["radius"] for r in lite}):
            sub = [r for r in lite if r["radius"] == radius]
            print(f"{radius:<16.1f}{len(sub):>4}"
                  f"{np.median([r['d_human_mean'] for r in sub]):18.2f}")

    print(f"\n-> {path.join(args.out, 'lbm_distance.csv')}")


if __name__ == "__main__":
    main()
