"""Compose the normalization-ablation renders into ONE side-by-side comparison video per segment.

Inputs are the outputs of the two-step ablation pipeline:
  main/dump_avgscale_render.py       -> <seg>/render_inputs.pt   (tgt_image_paths, scales)
  tools/lagernvs/render_avgscale.py  -> <seg>/render_<mode>.mp4  + metrics.json

--layout grid (default): 2x3 tiles of the render resolution
                           GT | lagernvs | avg_scale
                           maxd_seg | ctx_longer_135max | PSNR curve w/ cursor
--layout pair: one video PER MODE, [GT | render] concatenated along width.
--layout rows: [new] R x (1+M) 격자. --rows 에 ';' 로 구분한 mode 목록을 주면 목록 하나가 한 줄이
               되고, 각 줄의 0번 칸은 GT 다. 같은 분모 집합을 context 구성만 바꿔 (예: 정지 복사
               context vs single view) 비교할 때 쓴다. --labels 로 열 이름을, --row-tags 로 줄
               이름을 직접 준다. PSNR 패널 열은 붙지 않는다.
Each render tile is labelled with mode / divisor / this-frame PSNR / mean PSNR.

Run (lagernvs env -- needs load_and_preprocess_images for GT frames identical to the PSNR ones):
  /data1/cympyc1785/miniconda3/envs/lagernvs/bin/python scripts/render/compare_norm_video.py \
      --root results/lagernvs_norm_compare [--layout pair]
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LAGERNVS = "/data1/cympyc1785/LatentCamVid/camera_generation/tools/lagernvs"
# render_avgscale.py's filename convention (only avg_scale is abbreviated; '_pt' = point-scale token)
SFX = {"avg_scale": "avgscale", "avg_scale_pt": "avgscale_pt"}


def _sfx(m):
    """metrics.json 의 mode key -> render_avgscale.py 가 실제로 쓴 파일명.

    render_avgscale.py:143 의 suffix() 는 mode 안의 'avg_scale' 을 'avgscale' 로 바꾼 뒤
    RD_TAG 를 덧붙인다. SFX dict 로만 찾으면 tag/flag 가 붙은 key('avg_scale_st',
    'avg_scale_ch9nat' ...)를 놓치므로 접두 치환으로 일반화한다.
    """
    if m in SFX:
        return SFX[m]
    return "avgscale" + m[len("avg_scale"):] if m.startswith("avg_scale") else m


ORDER = ["lagernvs", "lagernvs_pt", "avg_scale", "avg_scale_pt",
         "maxd_seg", "ctx_longer_135max", "ctx_side_135max"]


def _font(sz):
    try:
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), sz)
    except Exception:
        return ImageFont.load_default()


def load_gt(paths, size, hw):
    """GT frames preprocessed exactly like the ones PSNR was computed against."""
    sys.path.insert(0, LAGERNVS)
    from vggt.utils.load_fn import load_and_preprocess_images
    g = load_and_preprocess_images(paths, mode="resize", target_size=size, patch_size=8)
    g = (g.clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return np.stack([np.asarray(Image.fromarray(f).resize((hw[1], hw[0]))) for f in g])


def psnr_panel(metrics, modes, T, hw):
    """PSNR-vs-frame curves rendered once; returns (image, x_of_frame) for the moving cursor."""
    W, H = hw[1], hw[0]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0.11, 0.16, 0.86, 0.74])          # fixed box -> known pixel mapping
    for m in modes:
        ax.plot(metrics[m]["psnr_per_frame"], lw=1.4,
                label=f"{m} ({metrics[m]['psnr']:.2f})")
    ax.set_xlim(0, T - 1); ax.set_xlabel("frame", fontsize=7); ax.set_ylabel("PSNR", fontsize=7)
    ax.tick_params(labelsize=6); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="upper right")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x0, x1 = ax.get_window_extent().x0, ax.get_window_extent().x1
    plt.close(fig)
    img = np.asarray(Image.fromarray(img).resize((W, H)))
    sx = W / fig.get_size_inches()[0] / 100
    return img, (lambda t: (x0 + (x1 - x0) * t / max(T - 1, 1)) * sx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/"
                                      "results/lagernvs_norm_compare")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--layout", choices=["grid", "pair", "row", "rows"], default="grid",
                    help="grid: one 2x3 video; pair: one [GT|render] video per mode; "
                         "row: a single [GT | mode... | PSNR] strip, PSNR plots only --modes; "
                         "rows: R x (1+M) 격자, --rows 로 줄마다 mode 목록 지정")
    ap.add_argument("--modes", default=None,
                    help="row/grid: comma-separated subset of modes to show (default: all)")
    ap.add_argument("--rows", default=None,
                    help="layout=rows: ';' 로 줄을, ',' 로 열을 구분한 mode 목록 "
                         "(예 'one_st,avg_scale_st,maxd_tgt_st;one_sv,avg_scale_sv,maxd_tgt_sv')")
    ap.add_argument("--labels", default=None,
                    help="layout=rows: 열 이름 (GT 포함, ',' 구분). 주면 mode 키 대신 이걸 그린다")
    ap.add_argument("--row-tags", default=None, help="layout=rows: 줄 이름 (',' 구분)")
    ap.add_argument("--out", default=None, help="default: <seg>/norm_compare.mp4")
    ap.add_argument("--stack", default=None,
                    help="row layout only: also vstack every segment's strip into this ONE mp4, "
                         "ordered by --tags (segments absent from --tags keep alphabetical order)")
    ap.add_argument("--tags", default=None,
                    help="json {seg_name: caption}; caption is drawn on that segment's GT tile and "
                         "its key order drives --stack row order")
    args = ap.parse_args()
    want = [m for m in args.modes.split(",") if m] if args.modes else None
    grid_rows = [[m for m in r.split(",") if m] for r in args.rows.split(";")] if args.rows else None
    if args.layout == "rows":
        if not grid_rows:
            ap.error("--layout rows 는 --rows 가 필요하다")
        if len({len(r) for r in grid_rows}) != 1:
            ap.error(f"--rows 의 줄마다 열 수가 달라야 안 된다: {[len(r) for r in grid_rows]}")
        want = [m for r in grid_rows for m in r]      # 로드할 mode 전체 (중복 허용 안 함)
    tags = json.load(open(args.tags)) if args.tags else {}
    stacked = {}

    fnt, fnt_s = _font(15), _font(12)
    for seg_dir in sorted(_glob.glob(osp.join(args.root, "*"))):
        mj = osp.join(seg_dir, "metrics.json")
        if not osp.isdir(seg_dir) or not osp.exists(mj):
            continue
        metrics = json.load(open(mj))
        modes = [m for m in ORDER if m in metrics] + [m for m in metrics if m not in ORDER]
        if want:                                    # keep the caller's order for --modes
            missing = [m for m in want if m not in metrics]
            if missing:
                print(f"{osp.basename(seg_dir)}: no metrics for {missing} -- skip"); continue
            modes = want
        vids = {m: np.stack(imageio.mimread(
            osp.join(seg_dir, f"render_{_sfx(m)}.mp4"), memtest=False)) for m in modes}
        T = min(len(v) for v in vids.values())
        H, W = vids[modes[0]].shape[1:3]

        rip = __import__("torch").load(osp.join(seg_dir, "render_inputs.pt"),
                                       map_location="cpu", weights_only=False)
        gt = load_gt(rip["tgt_image_paths"], args.size, (H, W))[:T]

        if args.layout == "pair":          # [GT | render] width-concat, one video per mode
            for m in modes:
                mm, frames = metrics[m], []
                for t in range(T):
                    im = Image.fromarray(np.concatenate([gt[t], vids[m][t][:, :, :3]], axis=1))
                    d = ImageDraw.Draw(im)
                    d.rectangle([0, 0, 2 * W, 34], fill=(0, 0, 0))
                    d.text((5, 2), "GT", fill=(255, 255, 0), font=fnt)
                    d.text((5, 19), f"frame {t}/{T-1}", fill=(180, 255, 180), font=fnt_s)
                    d.text((W + 5, 2), f"{m}  div={mm['divisor']:.2f}", fill=(255, 255, 0), font=fnt)
                    d.text((W + 5, 19), f"PSNR {mm['psnr_per_frame'][t]:.2f}   mean {mm['psnr']:.2f}",
                           fill=(180, 255, 180), font=fnt_s)
                    frames.append(np.asarray(im))
                out = osp.join(seg_dir, f"gt_vs_{_sfx(m)}.mp4")
                imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
                print(f"{osp.basename(seg_dir)} [{m}]: {T} frames {frames[0].shape} -> {out}", flush=True)
            continue

        if args.layout == "rows":      # R x (1+M): 0번 열은 GT, 줄마다 다른 mode 목록
            cols = ["GT"] + list(grid_rows[0])
            labels = ([x for x in args.labels.split(",")] if args.labels else cols)
            if len(labels) != len(cols):
                print(f"--labels 개수({len(labels)}) != 열 개수({len(cols)}) -- skip"); continue
            rtags = args.row_tags.split(",") if args.row_tags else [""] * len(grid_rows)
            frames = []
            for t in range(T):
                strips = []
                for r, rmodes in enumerate(grid_rows):
                    strips.append(np.concatenate(
                        [gt[t]] + [vids[m][t][:, :, :3] for m in rmodes], axis=1))
                im = Image.fromarray(np.concatenate(strips, axis=0))
                d = ImageDraw.Draw(im)
                for r, rmodes in enumerate(grid_rows):
                    for c, name in enumerate(["GT"] + list(rmodes)):
                        x0, y0 = c * W, r * H
                        if name == "GT":
                            lab = labels[0] if not rtags[r] else f"{labels[0]}  [{rtags[r]}]"
                            sub = f"frame {t}/{T-1}"
                        else:
                            mm = metrics[name]
                            lab = f"{labels[c]}  div={mm['divisor']:.2f}"
                            sub = (f"PSNR {mm['psnr_per_frame'][t]:.2f}   "
                                   f"mean {mm['psnr']:.2f}")
                        d.rectangle([x0, y0, x0 + W, y0 + 34], fill=(0, 0, 0))
                        d.text((x0 + 5, y0 + 2), lab, fill=(255, 255, 0), font=fnt)
                        d.text((x0 + 5, y0 + 19), sub, fill=(180, 255, 180), font=fnt_s)
                frames.append(np.asarray(im))
            if args.stack:
                stacked[osp.basename(seg_dir)] = frames
            out = args.out or osp.join(seg_dir, "norm_rows.mp4")
            if args.out and len(_glob.glob(osp.join(args.root, "*", "metrics.json"))) > 1:
                out = f"{osp.splitext(args.out)[0]}_{osp.basename(seg_dir)[:16]}.mp4"
            imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
            print(f"{osp.basename(seg_dir)}: {T} frames {frames[0].shape} -> {out}", flush=True)
            continue

        curve, xat = psnr_panel(metrics, modes, T, (H, W))

        if args.layout == "row":     # [GT | mode... | PSNR] in ONE width-concat strip
            frames = []
            for t in range(T):
                cur = Image.fromarray(curve.copy()); dc = ImageDraw.Draw(cur)
                x = xat(t); dc.line([(x, 0), (x, H)], fill=(200, 0, 0), width=1)
                im = Image.fromarray(np.concatenate(
                    [gt[t]] + [vids[m][t][:, :, :3] for m in modes] + [np.asarray(cur)], axis=1))
                d = ImageDraw.Draw(im)
                for i, name in enumerate(["GT"] + modes):
                    x0 = i * W
                    d.rectangle([x0, 0, x0 + W, 34], fill=(0, 0, 0))
                    if name == "GT":
                        lab, sub = tags.get(osp.basename(seg_dir), "GT"), f"frame {t}/{T-1}"
                    else:
                        mm = metrics[name]
                        lab = f"{name}  div={mm['divisor']:.2f}"
                        sub = f"PSNR {mm['psnr_per_frame'][t]:.2f}   mean {mm['psnr']:.2f}"
                    d.text((x0 + 5, 2), lab, fill=(255, 255, 0), font=fnt)
                    d.text((x0 + 5, 19), sub, fill=(180, 255, 180), font=fnt_s)
                frames.append(np.asarray(im))
            if args.stack:
                stacked[osp.basename(seg_dir)] = frames
            out = args.out or osp.join(seg_dir, "norm_compare_row.mp4")
            imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
            print(f"{osp.basename(seg_dir)}: {T} frames {frames[0].shape} -> {out}", flush=True)
            continue

        tiles = ["GT"] + modes + ["PSNR"]
        tiles = tiles[:6] + [None] * (6 - len(tiles))
        frames = []
        for t in range(T):
            rows = []
            for r in range(2):
                row = []
                for c in range(3):
                    name = tiles[r * 3 + c]
                    if name is None:
                        row.append(np.zeros((H, W, 3), np.uint8)); continue
                    if name == "GT":
                        row.append(gt[t])
                    elif name == "PSNR":
                        im = Image.fromarray(curve.copy()); d = ImageDraw.Draw(im)
                        x = xat(t); d.line([(x, 0), (x, H)], fill=(200, 0, 0), width=1)
                        row.append(np.asarray(im))
                    else:
                        row.append(vids[name][t][:, :, :3])
                rows.append(np.concatenate(row, axis=1))
            im = Image.fromarray(np.concatenate(rows, axis=0)); d = ImageDraw.Draw(im)
            for i, name in enumerate(tiles):
                if name in (None, "PSNR"):
                    continue
                x0, y0 = (i % 3) * W, (i // 3) * H
                if name == "GT":
                    lab, sub = "GT", f"frame {t}/{T-1}"
                else:
                    mm = metrics[name]
                    lab = f"{name}  div={mm['divisor']:.2f}"
                    sub = f"PSNR {mm['psnr_per_frame'][t]:.2f}   mean {mm['psnr']:.2f}"
                d.rectangle([x0, y0, x0 + W, y0 + 34], fill=(0, 0, 0))
                d.text((x0 + 5, y0 + 2), lab, fill=(255, 255, 0), font=fnt)
                d.text((x0 + 5, y0 + 19), sub, fill=(180, 255, 180), font=fnt_s)
            frames.append(np.asarray(im))
        out = args.out or osp.join(seg_dir, "norm_compare.mp4")
        imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
        print(f"{osp.basename(seg_dir)}: {len(frames)} frames {frames[0].shape} -> {out}", flush=True)

    if args.stack and stacked:
        order = [s for s in tags if s in stacked] + [s for s in stacked if s not in tags]
        Tm = min(len(stacked[s]) for s in order)
        big = [np.concatenate([stacked[s][t] for s in order], axis=0) for t in range(Tm)]
        imageio.mimwrite(args.stack, big, fps=args.fps, quality=8, macro_block_size=1)
        print(f"stacked {len(order)} segments: {Tm} frames {big[0].shape} -> {args.stack}", flush=True)


if __name__ == "__main__":
    main()
