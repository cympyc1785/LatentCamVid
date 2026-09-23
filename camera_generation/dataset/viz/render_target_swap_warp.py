"""같은 preset 을 **target 만 바꿔** 돌린 entry 들을 한 화면에 나란히 놓고 depth warp 로 렌더한다.

왜 `render_pred_depth_warp.py` 로는 안 되는가: 그쪽은 entry 를 **시간축으로** 이어붙이므로
(entry 하나가 열, 여러 entry 는 순차 재생) "straight_ease 를 woman/table/window 로 돌리면 궤적이
어떻게 갈리나"를 눈으로 비교할 수가 없다. 여기서는 **entry 가 열**이고 위/아래 두 행이 GT / pred 다.
텍스트 조건에서 motion 문장은 같고 `target:` 만 다르므로, 열 사이의 차이가 곧 target 조건의 효과다.

첫 열은 소스 영상. 열 라벨은 `target: <anchor>`, 하단 캡션은 `<variant_id>  tau <tau_max>`.

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=7 python viz/render_target_swap_warp.py --video avocado-slice \
        --eval_dir /.../results/20260827_051423_vista4d_pgt_k6 \
        --preset straight_ease --out_dir results/20260828_target_swap
"""
import json
import re
import sys
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.overlay import contact_sheet, label_tile                               # noqa: E402
from lbm.render import add_cloud_source_args, open_renderer                     # noqa: E402
from viz.render_bank_videos import render_variant, write_video              # noqa: E402
from viz.render_pred_depth_warp import (load_entry_meta, load_transforms,   # noqa: E402
                                            split_target_motion)


def ladder_rung(variant_id: str):
    """`dyn_0__straight_ease__hole0.35` -> 0.35. 사다리 단을 못 읽으면 무한대(뒤로 밀어냄)."""
    hit = re.search(r"hole([0-9.]+)$", variant_id)
    return float(hit.group(1)) if hit else float("inf")


def pick_entries(eval_dir: str, corpus_root: str, video: str, preset: str, per_target: int):
    """그 preset 의 entry 를 anchor 별로 **사다리 단(variant_id 의 `hole<x>`) 낮은 순** 으로 고른다.

    측정 hole_fraction 으로 고르면 안 된다 — anchor 마다 실측 hole 이 달라서 woman 은 hole0.35 단,
    table 은 hole0.1 단이 뽑히고, 그러면 열 사이 차이에 target 효과와 τ 효과가 뒤엉킨다
    (2026-08-28 첫 실행에서 실제로 그렇게 나왔다: woman τ 0.52 vs table τ 0.20).
    """
    names = [path.basename(p)[:-len("_transforms_pred.json")]
             for p in glob(path.join(eval_dir, "test", f"vista4d_{video}_*_transforms_pred.json"))]
    rows = []
    for name in names:
        meta = load_entry_meta(corpus_root, video, name)
        if meta["preset"] == preset:
            rows.append((name, meta))
    picked, seen = [], {}
    for name, meta in sorted(rows, key=lambda r: (r[1]["anchor"], ladder_rung(r[1]["variant_id"]))):
        if seen.get(meta["anchor"], 0) < per_target:
            seen[meta["anchor"]] = seen.get(meta["anchor"], 0) + 1
            picked.append((name, meta))
    rungs = sorted({ladder_rung(m["variant_id"]) for _, m in picked})
    if len(rungs) > 1:                       # 조용히 섞이면 비교가 무의미해지므로 눈에 띄게 찍는다
        print(f"[warn] 사다리 단이 섞였다: {rungs} — anchor 별로 없는 단이 있다는 뜻", flush=True)
    return picked


def main(args):
    picked = pick_entries(args.eval_dir, args.corpus_root, args.video, args.preset, args.per_target)
    assert picked, f"{args.preset} 인 entry 가 {args.eval_dir}/test 에 없다"

    # `fixed_focal=True` 는 이 릴의 규약이다 (아래 K 일치 assert 가 frame0 K 를 전제한다).
    renderer, recon = open_renderer(args, path.join(CLOUD_ROOT, "out"), fixed_focal=True)
    import cv2
    source = recon["video"]

    makedirs(args.out_dir, exist_ok=True)
    cols, stats, motions = [], {}, set()
    for name, meta in picked:
        ref_c2w, (json_w, json_h, fl_x, _) = load_transforms(
            path.join(args.eval_dir, "test", f"{name}_transforms_ref.json"))
        assert abs(fl_x * renderer.width / json_w - float(renderer.K_src[0][0, 0])) < 1.0, \
            "eval JSON K 와 cloud frame0 K 가 다르다"
        pred_c2w, _ = load_transforms(path.join(args.eval_dir, "test", f"{name}_transforms_pred.json"))
        with open(path.join(args.eval_dir, "test", f"{name}_caption.json"), encoding="utf-8") as file:
            target, motion = split_target_motion(next(iter(json.load(file).values())))
        motions.add(motion)

        _, gt_frames = render_variant(renderer, ref_c2w, json_h, json_w, args.stride)
        _, pred_frames = render_variant(renderer, pred_c2w, json_h, json_w, args.stride)
        hole = {k: round(float(np.mean([(np.all(f == (255, 0, 255), axis=-1)).mean() for f in fr])), 4)
                for k, fr in (("gt", gt_frames), ("pred", pred_frames))}
        cols.append({"name": name, "meta": meta, "target": target,
                     "gt": gt_frames, "pred": pred_frames, "size": (json_w, json_h)})
        stats[name] = {"anchor": meta["anchor"], "variant_id": meta["variant_id"],
                       "tau_max": meta["tau_max"], "hole_magenta_frac": hole}
        print(f"{name:<26}{meta['anchor']:<8}{meta['variant_id']:<40}"
              f"gt {hole['gt']:.3f}  pred {hole['pred']:.3f}", flush=True)

    json_w, json_h = cols[0]["size"]
    length = min(min(len(c["gt"]), len(c["pred"])) for c in cols)
    sheet_frames = []
    for f in range(length):
        src = cv2.resize(source[f * max(args.stride, 1)], (json_w, json_h))
        top = [label_tile(src.copy(), "SOURCE", f"{args.preset}  frame {f * max(args.stride, 1)}")]
        bottom = [label_tile(np.zeros_like(src), "", "")]
        for col in cols:
            cap = f"{col['meta']['variant_id']}  tau {col['meta']['tau_max']}"
            top.append(label_tile(col["gt"][f].copy(), f"GT  target: {col['target']}", cap))
            bottom.append(label_tile(col["pred"][f].copy(), f"PRED  target: {col['target']}", cap))
        sheet_frames.append(contact_sheet(top + bottom, columns=len(top)))

    out_path = path.join(args.out_dir, f"{args.video}__{args.preset}__target_swap.mp4")
    write_video(out_path, sheet_frames, args.fps)
    with open(path.join(args.out_dir, f"{args.preset}__index.json"), "w", encoding="utf-8") as file:
        json.dump({"video": args.video, "preset": args.preset, "eval_dir": args.eval_dir,
                   "motion_texts": sorted(motions), "entries": stats,
                   "stride": args.stride, "fps": args.fps}, file, ensure_ascii=False, indent=2)
    print(f"\n{'columns':<14}{len(cols)}")
    print(f"{'motion uniq':<14}{len(motions)}")
    print(f"{'out':<14}{out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--eval_dir", required=True)            # latentcam eval run 폴더 (test/ 를 가짐)
    parser.add_argument("--preset", required=True)              # straight_ease / push_in_arc / ...
    parser.add_argument("--per_target", type=int, default=1)    # anchor 당 몇 개 (hole 낮은 순)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--corpus_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3")
    add_cloud_source_args(parser, eval_data_default=EVAL_DATA_DEFAULT)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=12.0)
    main(parser.parse_args())
