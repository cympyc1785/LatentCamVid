"""GeoCalib + 소스 카메라 → 중력축 사이드카 `<out>/<video>/geocalib_gravity.json`.

왜 필요한가 — `scene_graph/gravity.py` 의 ground RANSAC 은 **바닥이 화면을 크게 차지하는 실외
씬에서만** 맞는다. vista 53편 실측: 15편(28%)이 `camera_up_fallback` 으로 떨어졌고(confidence
0.3), 통과한 38편 중에도 벽/책상 상판을 지면으로 문 게 여럿이다 (bed-shopping inlier 0.284 /
카메라 up 과 34.7°, rabbit-basket 0.284/38.4°, goat 0.540/41.1°). 중력축이 틀어지면 OBB yaw·
extent·center 가 전부 같이 틀어지고, roll=0 기준도 함께 어긋난다.

GeoCalib 은 단일 이미지에서 perspective field 로 중력을 직접 회귀한다 — 바닥이 안 보여도 된다.
소스 카메라를 두 군데에 쓴다:
  ① prior focal — `calibrate(priors={"focal": f_src})`. 안 주면 f 가 소스와 최대 40% 어긋난다
     (snow-dog 1802 → 1100). 중력 자체는 prior 유무로 거의 안 변하지만(실측 ≤0.5°, goat/
     rabbit-basket 만 4~5°) 공짜라 준다.
  ② camera→world — GeoCalib 의 `gravity.vec3d` 는 **카메라 프레임의 up 벡터**다(중력 방향이
     아니다). roll=pitch=0 에서 `from_rp` 가 `(0,-1,0)` 을 내는데 OpenCV 는 +Y 가 아래이므로
     그게 곧 위쪽이다. 실측으로도 확정: RANSAC 이 확실히 물린 10편에서 `R_c2w @ vec3d` 가
     1.2~3.7° (desert-park 만 12.4°), 부호를 뒤집으면 176~178° 다.
     따라서 `up_world = R_c2w[f] @ vec3d[f]`.

여러 프레임을 쓰는 이유: 한 장은 검증할 방법이 없다. 프레임 서로간 최대 각도(`spread_deg`)가
곧 자기 일관성이고, 그게 confidence 로 들어간다.

**기본 5장** — 지시는 "3프레임 정도" 였는데 3장이면 trim 이 아무것도 못 버린다
(`ceil(3·0.75)=3`). 5장이면 최악 1장을 버린다. n=3/5/9 스윕 실측(10편): 대부분 trim 결과가
n 에 무관하게 2° 안이고(camel 0.7, trumans-bedroom 0.1, truck-pose 0.3), snowboard 만
n=9 에서 spread 42.6° 까지 벌어진다 — 눈 덮인 사면이라 GeoCalib 이 프레임마다 다르게 본다.
그런 씬은 confidence 가 바닥으로 떨어지고 `angle_to_ransac_deg` 로 표시된다.

집계는 **trimmed mean**: 다른 표본과의 각도 median 이 최소인 표본(각도 median 방향)에서 먼
25% 를 버리고 평균. 단순 평균은 튄 프레임 1장에 5.1° 까지 끌려간다 (snowboard n=9).

ground RANSAC 은 **여기서 안 돌린다** — 검증은 `build_scene_graph.py --gravity_source geocalib`
쪽에서 두 축의 각도(`angle_to_ransac_deg`)로 기록한다. 이 스크립트는 사이드카만 만든다.

env: `geocalib` (kornia 가 필요한데 vista4d/da3 에 없다). cv2 로 video.mp4 만 읽으므로
Vista4D 로더는 안 쓴다.

예시:
    CUDA_VISIBLE_DEVICES=3 /data1/cympyc1785/miniconda3/envs/geocalib/bin/python \
        scripts/geocalib_gravity.py --videos camel snowboard
    CUDA_VISIBLE_DEVICES=3 ... scripts/geocalib_gravity.py --video_list out/vista_videos.txt
"""
import json
import sys
from argparse import ArgumentParser
from os import listdir, makedirs, path

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))
GEOCALIB_ROOT = path.normpath(path.join(HERE, "..", "..", "..", "tools", "GeoCalib"))
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
OUTPUT_ROOT_DEFAULT = path.join(HERE, "out")
FORMAT = "geocalib_gravity_v1"

# spread 가 이 각도면 confidence 0 — 3장이 이만큼 어긋나면 믿을 근거가 없다.
# RANSAC 이 확실히 물린 씬의 spread 가 0.5~4° 이고 흔들리는 씬이 7~9° 인 데서 잡은 값이다.
SPREAD_ZERO_DEG = 20.0
CONFIDENCE_FLOOR = 0.2


def read_frames(video_folder: str, indices):
    """video.mp4 에서 지정 프레임만 RGB 로. cv2 는 순차 디코드가 seek 보다 안전하다."""
    import cv2
    cap = cv2.VideoCapture(path.join(video_folder, "video.mp4"))
    assert cap.isOpened(), f"video.mp4 를 못 연다: {video_folder}"
    want, frames, f = set(int(i) for i in indices), {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f in want:
            frames[f] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f += 1
    cap.release()
    missing = want - set(frames)
    assert not missing, f"{video_folder}: 프레임 {sorted(missing)} 이 없다 (총 {f}장)"
    return [frames[int(i)] for i in indices]


def angle_deg(a: np.ndarray, b: np.ndarray):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def trimmed_direction(ups: np.ndarray, keep_frac: float = 0.75):
    """튄 프레임을 버린 평균 방향. (trimmed, mean, spread_deg, num_kept).

    각도 median 방향(다른 표본들과의 각도 median 이 최소인 표본)을 중심으로 잡고 거기서 먼
    `1-keep_frac` 을 버린다. 방향 데이터라 산술 median 이 없어서 이렇게 한다.
    """
    mean = ups.mean(axis=0)
    mean = mean / np.linalg.norm(mean)
    spread = max((angle_deg(ups[i], ups[j]) for i in range(len(ups)) for j in range(i)),
                 default=0.0)
    pairwise = np.array([[angle_deg(a, b) for b in ups] for a in ups])
    center = ups[int(np.argmin(np.median(pairwise, axis=1)))]
    num_kept = max(1, int(np.ceil(len(ups) * keep_frac)))
    keep = ups[np.argsort([angle_deg(u, center) for u in ups])[:num_kept]]
    trimmed = keep.mean(axis=0)
    trimmed = trimmed / np.linalg.norm(trimmed)
    return trimmed, mean, spread, num_kept


def estimate_video(model, video: str, eval_data: str, num_frames: int, device: str,
                   use_prior_focal: bool, args_keep_frac: float = 0.75):
    """영상 1편 → 사이드카 dict. GeoCalib 은 배치를 한 번에 받으므로 프레임 3장이 1 forward 다."""
    import torch

    folder = path.join(eval_data, "eval_data", "recon_and_seg", video)
    cameras = np.load(path.join(folder, "cameras.npz"))
    cam_c2w = cameras["cam_c2w"].astype(np.float64)
    intrinsics = cameras["intrinsics"].astype(np.float64)
    total = len(cam_c2w)
    indices = np.unique(np.linspace(0, total - 1, num_frames).round().astype(int))

    images = read_frames(folder, indices)
    batch = torch.stack([torch.from_numpy(im.transpose(2, 0, 1) / 255.0).float()
                         for im in images]).to(device)
    priors = None
    if use_prior_focal:
        priors = {"focal": torch.tensor([intrinsics[f][0] for f in indices],
                                        dtype=torch.float32, device=device)}
    with torch.no_grad():
        result = model.calibrate(batch, camera_model="pinhole", priors=priors)

    g_cam = result["gravity"].vec3d.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    f_geo = result["camera"].f.detach().cpu().numpy().reshape(-1, 2)[:, 0].astype(np.float64)

    def uncertainty(key):
        if key not in result:
            return None
        values = result[key].detach().cpu().numpy().reshape(len(indices), -1)
        return [float(v) for v in values[:, 0]]

    # 프레임별 up_world. `vec3d` 는 카메라 프레임 up 이므로 R_c2w 를 곱하면 world up 이다.
    per_frame = []
    ups = []
    for i, f in enumerate(indices):
        up = cam_c2w[f][:3, :3] @ g_cam[i]
        up = up / np.linalg.norm(up)
        ups.append(up)
        per_frame.append({
            "frame": int(f),
            "gravity_cam": [round(float(v), 6) for v in g_cam[i]],
            "up_world": [round(float(v), 6) for v in up],
            "focal_geocalib": round(float(f_geo[i]), 2),
            "focal_source": round(float(intrinsics[f][0]), 2),
        })
    ups = np.stack(ups)

    up_trimmed, up_mean, spread, num_kept = trimmed_direction(ups, args_keep_frac)
    confidence = float(np.clip(1.0 - spread / SPREAD_ZERO_DEG, CONFIDENCE_FLOOR, 1.0))

    # 카메라 up 과의 각도 — RANSAC fallback 이 쓰던 축과 얼마나 다른지. 크면 fallback 이
    # 그만큼 기울어 있었다는 뜻이다.
    up_cam = -cam_c2w[:, :3, 1].sum(axis=0)
    up_cam = up_cam / np.linalg.norm(up_cam)

    payload = {
        "format": FORMAT, "video": video,
        "num_frames_total": int(total), "frames": [int(f) for f in indices],
        "prior_focal": bool(use_prior_focal),
        "up_world": [float(v) for v in up_trimmed],          # trimmed mean — 하류가 쓰는 값
        "up_world_mean": [float(v) for v in up_mean],         # 진단용 (단순 평균)
        "trim_keep_frac": float(args_keep_frac), "num_kept": int(num_kept),
        "trim_shift_deg": round(angle_deg(up_trimmed, up_mean), 3),
        "spread_deg": round(spread, 3),
        "confidence": round(confidence, 3),
        "angle_to_cam_up_deg": round(angle_deg(up_trimmed, up_cam), 3),
        "per_frame": per_frame,
    }
    for key in ("gravity_uncertainty", "roll_uncertainty", "pitch_uncertainty",
                "focal_uncertainty"):
        values = uncertainty(key)
        if values is not None:
            payload[key] = [round(v, 6) for v in values]
    return payload


def main(args):
    sys.path.insert(0, args.geocalib_root)
    import torch
    from geocalib import GeoCalib

    videos = args.videos
    if videos is None and args.video_list:
        with open(args.video_list, encoding="utf-8") as file:
            videos = [line.strip() for line in file if line.strip()]
    if videos is None:
        root = path.join(args.eval_data, "eval_data", "recon_and_seg")
        videos = sorted(name for name in listdir(root)
                        if path.isfile(path.join(root, name, "cameras.npz")))
    videos = videos[args.shard_id::args.num_shards]

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available()
                                                        else "cpu")
    model = GeoCalib(weights=args.weights).to(device).eval()

    report = []
    for video in videos:
        output_path = path.join(args.output_root, video, "geocalib_gravity.json")
        if args.skip_done and path.isfile(output_path):
            print(f"SKIP {video} (이미 있음)")
            continue
        payload = estimate_video(model, video, args.eval_data, args.num_frames, device,
                                 args.prior_focal, args.trim_keep_frac)
        makedirs(path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        report.append(payload)
        print(f"{video:<24}spread {payload['spread_deg']:>6.2f}  "
              f"conf {payload['confidence']:.2f}  vs_camup "
              f"{payload['angle_to_cam_up_deg']:>6.2f}")

    if report:
        print(f"\n{'video':<24}{'spread':>8}{'conf':>7}{'vs_camup':>10}{'f_geo':>9}{'f_src':>9}")
        for payload in report:
            f_geo = np.mean([entry["focal_geocalib"] for entry in payload["per_frame"]])
            f_src = np.mean([entry["focal_source"] for entry in payload["per_frame"]])
            print(f"{payload['video']:<24}{payload['spread_deg']:>8.2f}"
                  f"{payload['confidence']:>7.2f}{payload['angle_to_cam_up_deg']:>10.2f}"
                  f"{f_geo:>9.0f}{f_src:>9.0f}")
        spreads = [p["spread_deg"] for p in report]
        print(f"\n{len(report)}편  spread median {np.median(spreads):.2f}  "
              f"max {np.max(spreads):.2f}  (>{SPREAD_ZERO_DEG:.0f}° 면 confidence "
              f"{CONFIDENCE_FLOOR})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT, type=str)
    parser.add_argument("--geocalib_root", default=GEOCALIB_ROOT, type=str)

    parser.add_argument("--videos", nargs="*", default=None)   # None = --video_list 또는 전체
    parser.add_argument("--video_list", default=None, type=str)
    parser.add_argument("--num_shards", default=1, type=int)
    parser.add_argument("--shard_id", default=0, type=int)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    parser.add_argument("--num_frames", default=5, type=int)   # 균등 5장 (§docstring)
    parser.add_argument("--trim_keep_frac", default=0.75, type=float)
    parser.add_argument("--weights", default="pinhole", choices=("pinhole", "distorted"))
    parser.add_argument("--device", default="auto", type=str)
    # 소스 K 를 prior 로 준다. 끄면 GeoCalib 이 focal 도 같이 추정한다.
    parser.add_argument("--prior_focal", action="store_true", default=True)
    parser.add_argument("--no_prior_focal", dest="prior_focal", action="store_false")

    main(parser.parse_args())
