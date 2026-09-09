"""**target 을 고정하고 preset 여러 개**를 한 화면 격자에 놓고 depth warp 로 렌더한다.

왜 별도 스크립트인가: `render_pred_depth_warp.py` 의 reel 은 entry 를 **시간축으로** 이어붙인다
— `reel_target_woman.mp4` 는 preset 하나가 끝나야 다음 preset 이 나온다. 그래서 "같은 target 을
두고 preset 8개가 서로 얼마나 다른 궤적을 만드나"를 한눈에 못 본다. `render_target_swap_warp.py`
는 공간 배치를 하지만 축이 반대다 (preset 고정 + target 을 열로). 여기서는 **target 고정 +
preset 을 격자로** 놓는다.

배치: 4열 × 4행. 행이 GT / PRED 로 번갈아 나오고 **GT 바로 아래가 그 preset 의 PRED** 다.

    row0   GT   p1 p2 p3 p4
    row1   PRED p1 p2 p3 p4
    row2   GT   p5 p6 p7 p8
    row3   PRED p5 p6 p7 p8

preset 당 entry 는 `variant_id` 의 사다리 단(`hole<x>`)이 **가장 낮은 것 하나**를 쓴다 —
실측 hole 로 고르면 preset 마다 다른 τ 단이 뽑혀서 격자 사이 차이에 preset 효과와 τ 효과가
뒤엉킨다 (`render_target_swap_warp.pick_entries` 와 같은 이유).

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=7 python scripts/render_preset_grid_warp.py --video avocado-slice \
        --eval_dir /.../results/20260827_051423_vista4d_pgt_k6 \
        --anchor woman --columns 4 --out_dir results/20260828_woman_preset_grid
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import VISTA4D_ROOT_DEFAULT                                      # noqa: E402
from lbm.overlay import contact_sheet, label_tile                               # noqa: E402
from lbm.render import CloudRenderer                                            # noqa: E402
from scripts.render_bank_videos import render_variant, write_video              # noqa: E402
from scripts.render_pred_depth_warp import load_entry_meta, load_transforms     # noqa: E402
from scripts.render_target_swap_warp import ladder_rung                         # noqa: E402

# 이동 축이 서로 겹치지 않게 고른 기본 8종 (전/후·좌우 궤도·좌우 이동·상하). `--presets` 로 덮는다.
DEFAULT_PRESETS = ["straight_ease", "push_in_arc", "pull_out_arc", "orbit_left_arc",
                   "orbit_right_arc", "truck_left", "pan_right", "pedestal_up"]


def pick_entries(eval_dir: str, corpus_root: str, video: str, anchor: str, presets):
    """preset 마다 `anchor` 인 entry 를 사다리 단 낮은 순으로 1개씩. 없는 preset 은 건너뛴다."""
    names = [path.basename(p)[: -len("_transforms_pred.json")]
             for p in glob(path.join(eval_dir, "test", f"vista4d_{video}_*_transforms_pred.json"))]
    by_preset = {}
    for name in names:
        meta = load_entry_meta(corpus_root, video, name)
        if meta["anchor"] != anchor:
            continue
        by_preset.setdefault(meta["preset"], []).append((name, meta))
    picked, missing = [], []
    for preset in presets:
        rows = by_preset.get(preset)
        if not rows:
            missing.append(preset)
            continue
        picked.append(min(rows, key=lambda r: ladder_rung(r[1]["variant_id"])))
    return picked, missing, sorted(by_preset)


def main(args):
    presets = args.presets or DEFAULT_PRESETS
    picked, missing, available = pick_entries(args.eval_dir, args.corpus_root, args.video,
                                              args.anchor, presets)
    assert picked, (f"anchor={args.anchor} 인 entry 가 없다. "
                    f"그 target 에 있는 preset: {available}")
    if missing:      # 조용히 빠지면 격자가 어긋나므로 눈에 띄게 찍는다
        print(f"[warn] anchor={args.anchor} 에 없는 preset: {missing}", flush=True)

    renderer = CloudRenderer(path.join(CLOUD_ROOT, "out", args.video, "cloud.npz"),
                             vista4d_root=args.vista4d_root, device=args.device,
                             fixed_focal=True)
    makedirs(args.out_dir, exist_ok=True)

    cells, stats = [], {}
    for name, meta in picked:
        ref_c2w, (json_w, json_h, fl_x, _) = load_transforms(
            path.join(args.eval_dir, "test", f"{name}_transforms_ref.json"))
        assert abs(fl_x * renderer.width / json_w - float(renderer.K_src[0][0, 0])) < 1.0, \
            "eval JSON K 와 cloud frame0 K 가 다르다"
        pred_c2w, _ = load_transforms(
            path.join(args.eval_dir, "test", f"{name}_transforms_pred.json"))

        _, gt_frames = render_variant(renderer, ref_c2w, json_h, json_w, args.stride)
        _, pred_frames = render_variant(renderer, pred_c2w, json_h, json_w, args.stride)
        hole = {k: round(float(np.mean([(np.all(f == (255, 0, 255), axis=-1)).mean() for f in fr])), 4)
                for k, fr in (("gt", gt_frames), ("pred", pred_frames))}
        cells.append({"name": name, "meta": meta, "gt": gt_frames, "pred": pred_frames,
                      "size": (json_w, json_h), "hole": hole})
        stats[name] = {"preset": meta["preset"], "anchor": meta["anchor"],
                       "variant_id": meta["variant_id"], "tau_max": meta["tau_max"],
                       "hole_magenta_frac": hole}
        print(f"{name:<26}{meta['preset']:<18}{meta['variant_id']:<40}"
              f"gt {hole['gt']:.3f}  pred {hole['pred']:.3f}", flush=True)

    columns = args.columns
    json_w, json_h = cells[0]["size"]
    blank = np.zeros((json_h, json_w, 3), dtype=np.uint8)
    length = min(min(len(c["gt"]), len(c["pred"])) for c in cells)
    rungs = sorted({ladder_rung(c["meta"]["variant_id"]) for c in cells})

    sheet_frames = []
    for f in range(length):
        tiles = []
        # 한 밴드 = GT 행 + 그 바로 아래 PRED 행. 같은 열이 같은 preset 이 되게 묶는다.
        for start in range(0, len(cells), columns):
            band = cells[start:start + columns]
            for key, tag in (("gt", "GT"), ("pred", "PRED")):
                for col in range(columns):
                    if col >= len(band):
                        tiles.append(blank.copy())
                        continue
                    cell = band[col]
                    cap = f"{cell['meta']['variant_id']}  hole {cell['hole'][key]:.3f}"
                    tiles.append(label_tile(cell[key][f].copy(),
                                            f"{tag}  {cell['meta']['preset']}", cap))
        sheet_frames.append(contact_sheet(tiles, columns=columns))

    out_path = path.join(args.out_dir, f"{args.video}__{args.anchor}__preset_grid.mp4")
    write_video(out_path, sheet_frames, args.fps)
    with open(path.join(args.out_dir, f"{args.anchor}__index.json"), "w", encoding="utf-8") as file:
        json.dump({"video": args.video, "anchor": args.anchor, "eval_dir": args.eval_dir,
                   "presets": [c["meta"]["preset"] for c in cells], "missing_presets": missing,
                   "available_presets": available, "columns": columns,
                   "ladder_rungs": rungs, "entries": stats,
                   "layout": "밴드마다 GT 행 + PRED 행, 같은 열이 같은 preset",
                   "stride": args.stride, "fps": args.fps}, file, ensure_ascii=False, indent=2)

    print(f"\n{'target(anchor)':<18}{args.anchor}")
    print(f"{'presets':<18}{len(cells)}  {[c['meta']['preset'] for c in cells]}")
    print(f"{'missing':<18}{missing}")
    print(f"{'ladder rungs':<18}{rungs}" + ("   (섞였다 — τ 효과가 preset 효과에 섞인다)"
                                            if len(rungs) > 1 else ""))
    rows = 2 * ((len(cells) + columns - 1) // columns)      # 밴드마다 GT/PRED 두 행
    print(f"{'grid':<18}{columns} cols x {rows} rows,  {length} frames")
    print(f"{'out':<18}{out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--eval_dir", required=True)         # latentcam eval run 폴더 (test/ 를 가짐)
    parser.add_argument("--anchor", required=True)           # 고정할 target: woman / table / window
    parser.add_argument("--presets", nargs="*", default=None)   # None = DEFAULT_PRESETS 8종
    parser.add_argument("--columns", type=int, default=4)    # 한 밴드의 열 수 (preset 개수)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--corpus_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3")
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=12.0)
    main(parser.parse_args())
