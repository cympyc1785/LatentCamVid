"""`near_depth` 열이 뱅크 재생성 사이에서 얼마나 흔들리는지 **뱅크와 똑같은 경로로** 잰다.

D57 정리 후 비트 동일성 확인에서 `poses.npz` 는 전량 비트 동일한데 `near_depth` 만 78/168 행이
달랐다 (최대 18.2%). 그게 렌더러 비결정성인지 내 재현 경로가 틀린 건지를 가리는 게 목적이다.
앞선 재현이 뱅크 값 근처에도 안 갔던 이유는 **렌더 해상도**를 안 맞췄기 때문일 가능성이 크다 —
`near_depth` 는 depth 의 **1 백분위**라 splat 밀도가 바뀌면 통째로 움직인다.

뱅크 경로 (`sample_camera_bank.measure_trajectory`) 그대로:
    picks   = unique(linspace(0, num_frames-1, verify_frames).round())
    near(f) = percentile(depth[valid], near_pct) / S
    열       = min over picks

사용 예시:
    python scripts/probe_near_depth_repeat.py --video camel \
        --variant_ids dyn_0__orbit_left_arc__hole0.2 --repeats 5
"""
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))

from lbm.render import CloudRenderer  # noqa: E402


def main():
    parser = ArgumentParser()
    parser.add_argument("--video", default="camel")
    parser.add_argument("--output_root", default="out")
    parser.add_argument("--bank_dir", default="hole_bank")
    parser.add_argument("--variant_ids", nargs="+", required=True)
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--verify_frames", default=13, type=int)   # 뱅크 manifest 와 같아야 한다
    parser.add_argument("--near_pct", default=1.0, type=float)
    parser.add_argument("--tile_height", default=360, type=int)    # ← 이걸 안 맞추면 값이 통째로 다르다
    parser.add_argument("--tile_width", default=640, type=int)
    args = parser.parse_args()

    import json
    root = path.join(args.output_root, args.video)
    bank = json.load(open(path.join(root, args.bank_dir, "bank.json")))
    scale = float(bank["S"])
    num_frames = int(bank["num_frames"])
    z = np.load(path.join(root, args.bank_dir, "poses.npz"))
    ids = [str(x) for x in z["variant_id"]]
    rows = {r["variant_id"]: r for r in bank["variants"]}

    #    **뱅크가 구워진 K 규약을 그대로 따라간다** (`fixed_focal` 은 bank.json top-level).
    #    이 프로브는 뱅크 수치를 재현하는 게 목적이라 K 가 다르면 애초에 비교가 성립하지 않는다.
    #    프레임별 DA3 K 는 화각이 떨린다 — snowboard fx 진폭 6.99%.
    renderer = CloudRenderer(path.join(root, "cloud.npz"), device="cuda",
                             fixed_focal=bool(bank.get("fixed_focal", False)))
    picks = np.unique(np.linspace(0, num_frames - 1, args.verify_frames).round().astype(int))
    print(f"picks {picks.tolist()}   S {scale:.6f}   near_pct {args.near_pct}   "
          f"render {args.tile_width}x{args.tile_height}")

    print(f"\n{'variant':<40}{'뱅크':>9}" + "".join(f"{f'재현{i+1}':>9}" for i in range(args.repeats))
          + f"{'재현 산포%':>11}{'뱅크 대비%':>11}")
    for vid in args.variant_ids:
        poses = z["cam_c2w"][ids.index(vid)]
        banked = rows[vid].get("near_depth", float("nan"))
        reps = []
        for _ in range(args.repeats):
            nears = []
            for f in picks:
                r = renderer.render(poses[f], frame=int(f),
                                    height=args.tile_height, width=args.tile_width)
                vis = r["depth"][r["valid"]]
                if vis.numel() if hasattr(vis, "numel") else vis.size:
                    v = vis.detach().cpu().numpy() if hasattr(vis, "detach") else vis
                    nears.append(float(np.percentile(v, args.near_pct)) / scale)
            reps.append(min(nears))
        spread = 100.0 * (max(reps) - min(reps)) / np.mean(reps)
        vs = 100.0 * (np.mean(reps) - banked) / banked
        print(f"{vid:<40}{banked:>9.4f}" + "".join(f"{r:>9.4f}" for r in reps)
              + f"{spread:>11.2f}{vs:>11.2f}")


if __name__ == "__main__":
    main()
