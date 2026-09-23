"""TRUMANS Lite 클립들의 `avg_scale.json` 을 **이미 만들어 둔 recon 폴더에서** 채운다.

왜 별도 스크립트인가: `trumans_to_recon.py` 는 이제 클립을 만들 때 avg_scale 을 같이 쓰지만,
그 전에 만든 뱅크 96편에는 파일이 없다. 그렇다고 96편을 다시 렌더하는 건(편당 ~3분) 낭비다 —
avg_scale 은 depth 와 카메라만 있으면 되고 그 둘은 recon 폴더에 이미 들어 있다.

정의는 `trumans_to_recon.avg_scale_first_cam()` **하나뿐**이다. 여기서 다시 구현하지 않고
import 해서 쓴다 — 분모가 두 군데서 따로 계산되면 어긋나도 안 터지고, 어긋난 채로 학습되면
context 와 target 의 스케일 공간이 달라져 조용히 망가진다.

⚠ 정밀도: recon 폴더의 depth 는 **float16 EXR** 이다 (렌더 원본 `render_a*/depth/*.npy` 는
float32). 2 m 대에서 float16 간격이 ~1 mm 라 평균에는 영향이 없지만, 원본이 살아 있으면
`--from_render` 로 float32 를 쓰는 쪽이 정확하다. 두 경로의 실측 차이는 `--check` 로 잰다.

usage:
  # 뱅크 전량 (recon 폴더의 float16 depth 사용)
  python make_avg_scale_trumans.py --root /data1/cympyc1785/data/TRUMANS-Lite/eval_data

  # 렌더 원본 float32 로 (work 디렉토리가 살아 있을 때). manifest 가 두 경로를 이어 준다
  python make_avg_scale_trumans.py --manifests '<CinemaTraj>/out/trumans_recon/*/manifest_a*.json' \
      --from_render

  # 이미 있는 파일과 대조만 하고 쓰지 않는다
  python make_avg_scale_trumans.py --root ... --check --no_write
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
CINEMATRAJ_ROOT = path.dirname(path.dirname(HERE))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)
from fit.ingest.trumans_to_recon import AVG_SCALE_STRIDE, avg_scale_first_cam  # noqa: E402

VISTA4D_DEFAULT = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"


def load_from_recon(recon_dir: str, vista4d_root: str):
    """recon 폴더 → (depth, cam_c2w, K, sky_mask, dynamic_mask). depth 는 float16 EXR."""
    if vista4d_root not in sys.path:
        sys.path.insert(0, vista4d_root)
    from utils.media import intrinsics_to_K, load_cameras, load_depths, load_masks

    depth = load_depths(path.join(recon_dir, "depths"), dtype=np.float16).astype(np.float32)
    cam_c2w, intrinsics = load_cameras(path.join(recon_dir, "cameras.npz"))
    sky = load_masks(path.join(recon_dir, "sky_mask"))
    dyn = load_masks(path.join(recon_dir, "dynamic_mask"))
    #    intrinsics 는 프레임마다 같다 (Lite 는 렌즈 고정). 첫 프레임 K 를 대표로 쓴다.
    return depth, cam_c2w, intrinsics_to_K(intrinsics)[0], sky, dyn


def load_from_render(manifest: dict):
    """렌더 원본 → 같은 5종. depth 가 float32 라 EXR 왕복이 없다.

    sky/dynamic 은 recon 폴더의 PNG 대신 원본 index/depth 에서 **다시** 만든다 —
    `convert()` 가 쓰는 것과 같은 규칙이라야 픽셀 집합이 어긋나지 않는다.
    """
    render_dir = manifest["paths"]["render"]
    frames = sorted(int(path.basename(p)[6:11])
                    for p in glob(path.join(render_dir, "depth", "frame_*.npy")))
    depth = np.stack([np.load(path.join(render_dir, "depth", f"frame_{f:05d}.npy"))
                      for f in frames])
    index = np.stack([np.load(path.join(render_dir, "index", f"frame_{f:05d}.npy"))
                      for f in frames])
    with open(path.join(render_dir, "cameras.json"), encoding="utf-8") as file:
        cameras = json.load(file)
    poses = np.load(manifest["paths"]["poses"])["cam_c2w"].astype(np.float64)
    anchored = np.linalg.inv(poses[0])[None] @ poses
    #    convert() 와 같은 sky 규칙: sentinel 초과가 sky. 기본 sky_depth 는 1000 m.
    sky = depth > 1000.0
    return (np.minimum(depth, 1000.0).astype(np.float32), anchored,
            np.asarray(cameras["cameras"][0]["K"], dtype=np.float64), sky, index == 1)


def main(args):
    jobs, mpaths = [], {}
    if args.manifests:
        for pattern in args.manifests:
            for mpath in sorted(glob(pattern)):
                with open(mpath, encoding="utf-8") as file:
                    manifest = json.load(file)
                mpaths[manifest["video"]] = mpath
                jobs.append((manifest["video"], manifest["paths"]["recon"], manifest))
    else:
        jobs += [(path.basename(d.rstrip("/")), d.rstrip("/"), None)
                 for d in sorted(glob(path.join(args.root, "recon_and_seg", "*")))
                 if path.isdir(path.join(d, "depths"))]
    assert jobs, "대상 클립을 못 찾았다 (--root 또는 --manifests 확인)"

    rows, changed, skipped = [], 0, 0
    for video, recon_dir, manifest in jobs:
        out_path = path.join(recon_dir, "avg_scale.json")
        old = None
        if path.isfile(out_path):
            with open(out_path, encoding="utf-8") as file:
                old = json.load(file)
            if args.skip_done and not args.check:
                skipped += 1
                continue
        if args.from_render:
            assert manifest is not None, "--from_render 는 --manifests 가 있어야 한다"
            depth, cam_c2w, K, sky, dyn = load_from_render(manifest)
        else:
            depth, cam_c2w, K, sky, dyn = load_from_recon(recon_dir, args.vista4d_root)
        scale = avg_scale_first_cam(depth, cam_c2w, K, sky, dyn, stride=args.stride)
        if not args.no_write:
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump(scale, file, ensure_ascii=False, indent=1)
            changed += 1
        delta = (abs(scale["avg_scale"] - old["avg_scale"]) / old["avg_scale"] * 100.0
                 if old else float("nan"))
        #    manifest 의 `render` 블록에도 심는다. 새로 만든 클립은 convert() 가 report 로
        #    돌려주지만(trumans_to_recon:629,846), 그 전에 만든 뱅크는 이 경로로만 채워진다.
        #    top-level 이 아니라 render 아래인 이유: sky_frac/depth_p50 과 같은 "잰 값"이고,
        #    top-level 은 video/subject/caption 같은 신원 필드다.
        if args.patch_manifest and manifest is not None and not args.no_write:
            manifest.setdefault("render", {}).update(
                avg_scale=scale["avg_scale"], avg_scale_static=scale["avg_scale_static"])
            with open(mpaths[video], "w", encoding="utf-8") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=1)
        rows.append((video, scale["avg_scale"], scale["avg_scale_static"],
                     float(sky.mean()), delta))
        print(f"{video:24s} avg_scale {scale['avg_scale']:8.3f} m   "
              f"static {scale['avg_scale_static']:8.3f} m   sky {sky.mean() * 100:5.1f}%"
              + (f"   기존 대비 {delta:+.2f}%" if old else ""))

    values = np.array([r[1] for r in rows], dtype=np.float64)
    statics = np.array([r[2] for r in rows], dtype=np.float64)
    print(f"\n{'clips':22s} {len(rows)}  (written {changed}, skipped {skipped})")
    print(f"{'stride':22s} {args.stride}   source "
          f"{'render npy(float32)' if args.from_render else 'recon EXR(float16)'}")
    if len(rows):
        print(f"{'avg_scale min/med/max':22s} {values.min():.3f} / "
              f"{np.median(values):.3f} / {values.max():.3f} m")
        # 사람 포함/제외 차이. 크면 그 클립의 분모가 씬이 아니라 사람 거리를 재고 있다.
        ratio = statics / values
        print(f"{'static/all ratio':22s} {ratio.min():.4f} / "
              f"{np.median(ratio):.4f} / {ratio.max():.4f}")
        spread = values.max() / values.min()
        print(f"{'max/min spread':22s} {spread:.2f}x")


if __name__ == "__main__":
    parser = ArgumentParser()
    # 둘 중 하나. --root 는 recon 폴더를 직접 훑고, --manifests 는 렌더 원본까지 이어 준다.
    parser.add_argument("--root", default="", type=str)              # <...>/eval_data
    parser.add_argument("--manifests", default=[], nargs="*", type=str)   # glob 패턴
    parser.add_argument("--from_render", action="store_true")        # float32 원본 depth 사용
    parser.add_argument("--stride", default=AVG_SCALE_STRIDE, type=int)
    parser.add_argument("--skip_done", action="store_true")          # 이미 있으면 건너뛴다
    parser.add_argument("--no_write", action="store_true")           # 계산만 하고 안 쓴다
    parser.add_argument("--check", action="store_true")              # 기존 값과 대조해 출력
    parser.add_argument("--patch_manifest", action="store_true")     # manifest["render"] 에도 심는다
    parser.add_argument("--vista4d_root", default=VISTA4D_DEFAULT, type=str)
    parsed = parser.parse_args()
    assert parsed.root or parsed.manifests, "--root 또는 --manifests 중 하나는 필요하다"
    main(parsed)
