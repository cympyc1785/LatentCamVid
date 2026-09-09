"""학습 데이터(`latentcam_da3`)의 **target 궤적 그대로** depth 를 렌더해서 저장한다.

왜 뱅크(`out/<video>/hole_bank_k6/poses.npz`)가 아니라 `da3/target_poses.npz` 인가:
그 둘은 같지 않다. 내보내기(`vista4d_bank_to_dl3dv.py`)가 중간에 두 가지를 한다 —
① **dedup**: 사다리 4단이 게이트에 물려 같은 궤적으로 붕괴한 변이를 pose 배열 동일성으로 제거
   (camel 뱅크 → 178, avocado-slice → 191 만 남았다),
② **K 고정**: frame0 K 를 49프레임에 복사하고 이미지 배율(0.5)만큼 줄인다 (640x360).
그래서 "지금 학습이 실제로 보고 있는 카메라"는 `target_poses.npz` 쪽이고, 여기서 렌더해야
depth 가 학습 target 과 1:1 로 붙는다.

저장 규약
---------
    <out>/<video>/target_depth/<variant_id>.npz
        depth  (T,H,W) float16   — world(DA3 raw) 단위. **구멍은 0**
        (valid 마스크를 따로 안 둔다: 유효 depth 는 항상 > 0 이라 `depth > 0` 이 곧 valid 다.
         (T,H,W) bool 을 같이 두면 파일이 1.5배가 되는데 정보가 0 이다.)
    <out>/<video>/target_depth/index.json
        변이별 hole_fraction / depth 분위수 + 이 실행의 설정 전량

좌표계: `target_poses.npz:extrinsics` 는 **w2c** (V,T,4,4) 다 (`vista4d_bank_to_dl3dv.py:160`
`np.linalg.inv(bank_c2w)`). 렌더러는 c2w 를 받으므로 여기서 다시 뒤집는다. 부호를 잘못 잡으면
조용히 반대쪽을 렌더하므로, `--check` 가 렌더한 hole_fraction 을 `prompts.json` 에 저장된
뱅크 실측 `hole_fraction` 과 대조한다 — 규약이 어긋나면 이 값이 통째로 어긋난다.

`--fixed_focal` 기본 켬: 뱅크를 그렇게 만들었고 `target_poses.npz` 의 K 도 frame0 고정본이다.
끄면 렌더러가 프레임마다 흔들리는 DA3 focal 을 쓰게 되어 저장된 K 와 어긋난다.

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=6 python scripts/render_target_poses_depth.py \
        --videos camel avocado-slice --check --preview 3
    CUDA_VISIBLE_DEVICES=6 python scripts/render_target_poses_depth.py \
        --videos camel --variants 0 1 2 --preview 3 --dry_run
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import VISTA4D_ROOT_DEFAULT                                      # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.render import CloudRenderer                                            # noqa: E402
from scripts.render_bank_videos import colorize_depth, write_video              # noqa: E402

DL3DV_ROOT_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3"


def load_targets(dl3dv_root: str, chunk: str):
    """`da3/target_poses.npz` → (c2w (V,T,4,4), K (V,T,3,3), variant_id (V,), seg key (V,))."""
    npz = np.load(path.join(dl3dv_root, *chunk.split("/"), "da3", "target_poses.npz"),
                  allow_pickle=False)
    w2c = np.asarray(npz["extrinsics"], dtype=np.float64)          # (V,T,4,4) w2c
    assert w2c.ndim == 4 and w2c.shape[-2:] == (4, 4), f"extrinsics shape {w2c.shape}"
    return (np.linalg.inv(w2c), np.asarray(npz["intrinsics"], dtype=np.float64),
            [str(v) for v in npz["variant_id"].tolist()],
            [str(k) for k in npz["keys"].tolist()])


def stored_holes(dl3dv_root: str, chunk: str):
    """`prompts.json` 의 seg -> 뱅크가 실측한 hole_fraction. `--check` 의 대조군이다."""
    with open(path.join(dl3dv_root, *chunk.split("/"), "da3", "prompts.json"),
              encoding="utf-8") as file:
        prompts = json.load(file)
    return {seg: float(row["hole_fraction"]) for seg, row in prompts.items()
            if "hole_fraction" in row}


def render_variant_depth(renderer, c2w, K, height: int, width: int):
    """궤적 전 프레임 depth. 반환 (depth (T,H,W) float32, valid (T,H,W) bool)."""
    depth, valid = [], []
    for f in range(len(c2w)):
        shot = renderer.render(c2w[f], K=K[f], frame=int(f), height=height, width=width)
        good = shot["valid"] & np.isfinite(shot["depth"]) & (shot["depth"] > 0)
        depth.append(np.where(good, shot["depth"], 0.0).astype(np.float32))
        valid.append(good)
    return np.stack(depth), np.stack(valid)


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    rows, skipped = [], []

    for video in args.videos:
        chunk = f"{args.chunk_prefix}/{video}" if args.chunk_prefix else video
        target_npz = path.join(args.dl3dv_root, *chunk.split("/"), "da3", "target_poses.npz")
        cloud_path = path.join(out_root, video, "cloud.npz")
        if not path.isfile(target_npz):
            # 학습 코퍼스에 없는 영상. 뱅크가 통째로 빠진 경우가 있어(예: snowboard 는 소스
            # 자신의 시차가 τ 사다리 꼭대기를 넘어 350/360 변이가 saturated) 이유를 같이 낸다.
            why = "target_poses.npz 없음 (학습 코퍼스에 없는 영상)"
            skip_json = path.join(out_root, video, args.bank_dir, "skipped.json")
            if path.isfile(skip_json):
                with open(skip_json, encoding="utf-8") as file:
                    info = json.load(file)
                why += (f" — {args.bank_dir}/skipped.json: {info.get('reason')} "
                        f"(tau_start {info.get('tau_start')}, "
                        f"saturated {info.get('num_dropped_saturated')})")
            skipped.append((video, why))
            continue
        if not path.isfile(cloud_path):
            skipped.append((video, "cloud.npz 없음 (`python -m lbm.cloud --video <v>` 먼저)"))
            continue

        c2w, K, variant_ids, segs = load_targets(args.dl3dv_root, chunk)
        # 저장된 K 가 곧 데이터셋 해상도다 (cx*2, cy*2) — 내보내기가 `--image_scale 0.5` 로
        # 줄여서 640x360. 렌더 해상도는 여기에 `--scale` 을 더 곱한 것.
        store_w = int(round(float(K[0, 0, 0, 2]) * 2))
        store_h = int(round(float(K[0, 0, 1, 2]) * 2))
        width = int(round(store_w * args.scale))
        height = int(round(store_h * args.scale))
        pick = list(range(len(variant_ids))) if not args.variants else \
            [int(v) for v in args.variants]
        if args.limit > 0:
            pick = pick[:args.limit]

        nbytes = len(pick) * c2w.shape[1] * height * width * 2
        print(f"\n{'video':<16}{video}")
        print(f"{'variants':<16}{len(variant_ids)}  -> 렌더 {len(pick)}")
        print(f"{'frames':<16}{c2w.shape[1]}   render {width}x{height} (scale {args.scale})")
        print(f"{'예상 용량':<14}{nbytes / 2**30:.2f} GiB (float16)")
        if args.dry_run:
            for i in pick[:5]:
                print(f"   {segs[i]:>5}  {variant_ids[i]}")
            continue

        depth_dir = path.join(args.out_dir, video, "target_depth") if args.out_dir else \
            path.join(args.dl3dv_root, *chunk.split("/"), "da3", "target_depth")
        makedirs(depth_dir, exist_ok=True)
        renderer = CloudRenderer(cloud_path, vista4d_root=args.vista4d_root,
                                 device=args.device, fixed_focal=args.fixed_focal)
        # `CloudRenderer.render` 는 넘긴 K 를 **점군 원본 해상도 기준**으로 보고 렌더 해상도로
        # 다시 줄인다 (render.py:143-146). 저장된 K 는 이미 0.5배 줄여 둔 것이라 그대로 넘기면
        # 화각이 두 번 줄어든다 — 여기서 원본 해상도로 되돌려 넘긴다.
        K_native = K.copy()
        K_native[..., :2, :] *= renderer.width / store_w
        assert abs(renderer.height / store_h - renderer.width / store_w) < 1e-6, \
            f"가로/세로 배율이 다르다: {renderer.width}x{renderer.height} vs {store_w}x{store_h}"
        # frame0 고정 K 를 쓴다면 되돌린 K 는 렌더러 자신의 K_src 와 같아야 한다. 여기가
        # 어긋나면 뱅크와 다른 화각으로 렌더하는 것이고 hole/depth 가 통째로 달라진다.
        if args.fixed_focal:
            assert np.allclose(K_native[0, 0], renderer.K_src[0], rtol=1e-4, atol=1e-3), \
                f"target K != cloud frame0 K\n{K_native[0, 0]}\n{renderer.K_src[0]}"
        holes_ref = stored_holes(args.dl3dv_root, chunk) if args.check else {}

        index, preview_left, diffs = {}, args.preview, []
        for n, i in enumerate(pick):
            vid = variant_ids[i]
            depth, valid = render_variant_depth(renderer, c2w[i], K_native[i], height, width)
            # 구멍은 0 으로 저장한다 (위 docstring 참조). float16 은 world 단위 depth 에서
            # 상대오차 ~1e-3 이라 저장 손실이 렌더 자체의 splatting 오차보다 훨씬 작다.
            save = np.savez_compressed if args.compress else np.savez
            save(path.join(depth_dir, f"{vid}.npz"), depth=depth.astype(np.float16))
            hole = float(1.0 - valid.mean())
            pool = depth[valid]
            index[vid] = {
                "seg": segs[i], "variant_id": vid,
                "hole_fraction": round(hole, 6),
                "depth_p5": round(float(np.percentile(pool, 5)), 6) if pool.size else None,
                "depth_p50": round(float(np.percentile(pool, 50)), 6) if pool.size else None,
                "depth_p95": round(float(np.percentile(pool, 95)), 6) if pool.size else None,
            }
            if segs[i] in holes_ref:
                diffs.append(abs(hole - holes_ref[segs[i]]))
                index[vid]["hole_fraction_bank"] = round(holes_ref[segs[i]], 6)
            if preview_left > 0:
                lo, hi = (float(v) for v in np.percentile(pool, [5, 95]))
                write_video(path.join(depth_dir, f"{vid}__depth.mp4"),
                            [colorize_depth(depth[f], valid[f], lo, max(hi, lo * 1.01))
                             for f in range(len(depth))], args.fps)
                preview_left -= 1
            if (n + 1) % 20 == 0 or n + 1 == len(pick):
                print(f"   {n + 1}/{len(pick)}  {vid:<44} hole {hole:.3f}", flush=True)

        meta = {"format": "vista4d_target_depth_v1", "video": video, "chunk": chunk,
                "source": target_npz, "cloud": cloud_path,
                "height": height, "width": width, "scale": args.scale,
                "fixed_focal": args.fixed_focal, "hole_value": 0.0, "dtype": "float16",
                "units": "DA3 raw world units (S 로 안 나눔 — 분모는 avg_scale 로 따로)",
                "variants": index}
        with open(path.join(depth_dir, "index.json"), "w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=1)

        row = {"video": video, "n": len(pick), "dir": depth_dir,
               "hole_med": float(np.median([v["hole_fraction"] for v in index.values()]))}
        if diffs:
            # 규약 검사. w2c/c2w 를 뒤집었거나 K 가 어긋나면 hole 이 통째로 달라진다.
            row["check_med"], row["check_max"] = float(np.median(diffs)), float(np.max(diffs))
        rows.append(row)

    print(f"\n{'video':<18}{'variants':>10}{'hole_med':>10}"
          f"{'|Δhole| med':>13}{'max':>8}")
    for r in rows:
        chk = (f"{r['check_med']:>13.4f}{r['check_max']:>8.4f}"
               if "check_med" in r else f"{'-':>13}{'-':>8}")
        print(f"{r['video']:<18}{r['n']:>10}{r['hole_med']:>10.3f}{chk}")
    for r in rows:
        print(f"-> {r['dir']}")
    if skipped:
        print("\n[건너뜀]")
        for video, why in skipped:
            print(f"  - {video}: {why}")
    if any("check_max" in r for r in rows):
        print("\n대조: 렌더 hole_fraction vs prompts.json 에 저장된 뱅크 실측치. "
              "규약(w2c↔c2w, K)이 어긋나면 이 차이가 0.1 단위로 벌어진다.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dl3dv_root", default=DL3DV_ROOT_DEFAULT, type=str)
    parser.add_argument("--chunk_prefix", default="vista4d", type=str)
    parser.add_argument("--output_root", default=None, type=str)   # CinemaTraj out/ (cloud.npz)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--bank_dir", default="hole_bank_k6", type=str)  # skipped.json 조회용

    parser.add_argument("--videos", nargs="+", required=True)
    # 기본은 데이터셋 옆(`<chunk>/da3/target_depth/`) — target 궤적과 같은 자리에 둬야
    # 나중에 조건 입력으로 쓸 때 경로가 안 갈린다. 다른 데 쌓으려면 이 값을 준다.
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    # 뱅크·target_poses 의 K 가 frame0 고정본이라 기본 켬. 끄면 저장된 K 와 어긋난다.
    parser.add_argument("--fixed_focal", action="store_true", default=True)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")

    parser.add_argument("--variants", nargs="*", default=None)  # 변이 인덱스 (seg 키와 같다)
    parser.add_argument("--limit", default=0, type=int)         # 0 = 전량
    parser.add_argument("--scale", default=1.0, type=float)     # 1.0 = 저장된 K 그대로 (640x360)
    parser.add_argument("--compress", action="store_true", default=False)
    parser.add_argument("--no_compress", dest="compress", action="store_false")
    # 저장된 hole 실측치와 대조해 좌표 규약을 검증한다. 공짜라 기본 켬.
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--no_check", dest="check", action="store_false")
    parser.add_argument("--preview", default=0, type=int)       # 앞 N 변이는 컬러맵 mp4 도
    parser.add_argument("--fps", default=12.0, type=float)
    parser.add_argument("--dry_run", action="store_true", default=False)
    main(parser.parse_args())
