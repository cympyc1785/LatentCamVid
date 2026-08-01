"""Stitch consecutive per-segment LagerNVS renders into ONE continuous video per scene.

Question it answers: does LagerNVS stay consistent when the CHUNK changes? Each segment is
rendered independently (its own geo context views, its own divisor D), so any brightness /
geometry / scale jump at a segment boundary shows up as a seam in the stitched video.

Inputs = the two-step ablation pipeline's output, one dir per segment named <scene>_<segidx>:
  main/dump_avgscale_render.py       -> <seg>/render_inputs.pt   (tgt_image_paths, scales)
  tools/lagernvs/render_avgscale.py  -> <seg>/render_<mode>.mp4  + metrics.json
(cf. scripts/render/compare_norm_video.py, which does the same tiling but per segment only.)

Per scene it writes:
  <root>/_stitched/<scene>_<mode>.mp4        one arm, [GT | render], seams marked
  <root>/_stitched/<scene>_compare.mp4       [GT | mode1 | mode2 | ...] side by side
  <root>/_stitched/<scene>_psnr.png          PSNR vs global frame, segment boundaries marked
The first frame of every new segment gets a red border + "CHUNK N" banner, and each tile is
labelled with the segment's own divisor so a divisor jump is readable off the video.

Run (lagernvs env -- needs load_and_preprocess_images for GT frames identical to the PSNR ones):
  /data1/cympyc1785/miniconda3/envs/lagernvs/bin/python scripts/render/stitch_chunk_continuity.py \
      --root results/chunk_continuity --modes ctx_longer_135max,lagernvs
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os.path as osp, sys, json, re, argparse
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LAGERNVS = "/data1/cympyc1785/LatentCamVid/camera_generation/tools/lagernvs"
SFX = {"avg_scale": "avgscale"}          # render_avgscale.py's filename convention


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


def psnr_figure(segs, modes, out_png, scene):
    """PSNR vs GLOBAL frame index, with a vertical line at every chunk boundary."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.4))
    for m in modes:
        y, off = [], 0
        for sg in segs:
            y += sg["metrics"][m]["psnr_per_frame"][:sg["T"]]
        ax.plot(y, lw=1.3, label=m)
    off = 0
    for k, sg in enumerate(segs):
        if k:
            ax.axvline(off, color="k", ls="--", lw=1.0, alpha=0.7)
        for m in modes:
            ax.annotate(f"{m[:9]} D={sg['metrics'][m]['divisor']:.2f}",
                        (off + 2, ax.get_ylim()[0]), fontsize=6, rotation=90,
                        va="bottom", alpha=0.6)
        off += sg["T"]
    ax.set_xlabel("global frame"); ax.set_ylabel("PSNR"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); ax.set_title(f"{scene}: per-frame PSNR across {len(segs)} chunks", fontsize=9)
    fig.tight_layout(); fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


def seam_metric(segs, mode):
    """How much EXTRA temporal discontinuity the chunk boundary adds.

    Consecutive segments are contiguous in frame index (seg k ends at frame e-1, seg k+1 starts at
    e), so the GT itself already moves between them. Normalise by that:
        r(t) = mean|R[t+1]-R[t]| / mean|G[t+1]-G[t]|
    within-chunk r is the renderer's normal frame-to-frame behaviour; r at the boundary above that
    baseline is the seam. Returns (seam_ratios, baseline) with ratios in units of the baseline --
    1.0 = the boundary is indistinguishable from an ordinary frame step.
    """
    def r(a1, a0, g1, g0):
        dg = np.abs(g1.astype(np.float32) - g0.astype(np.float32)).mean()
        return float(np.abs(a1.astype(np.float32) - a0.astype(np.float32)).mean() / max(dg, 1e-6))
    within = []
    for sg in segs:
        V, G = sg["vids"][mode], sg["gt"]
        within += [r(V[t + 1], V[t], G[t + 1], G[t]) for t in range(sg["T"] - 1)]
    base = float(np.median(within))
    seams = []
    for a, b in zip(segs[:-1], segs[1:]):
        seams.append(r(b["vids"][mode][0], a["vids"][mode][a["T"] - 1],
                       b["gt"][0], a["gt"][a["T"] - 1]) / max(base, 1e-6))
    return seams, base


def banner(im, x0, y0, W, lab, sub, fnt, fnt_s):
    d = ImageDraw.Draw(im)
    d.rectangle([x0, y0, x0 + W, y0 + 34], fill=(0, 0, 0))
    d.text((x0 + 5, y0 + 2), lab, fill=(255, 255, 0), font=fnt)
    d.text((x0 + 5, y0 + 19), sub, fill=(180, 255, 180), font=fnt_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/"
                                      "results/chunk_continuity")
    ap.add_argument("--modes", default="ctx_longer_135max,lagernvs",
                    help="comma-separated. '<tag>:<mode>' pulls that mode from the SIBLING dir "
                         "<scene><tag>_<k> instead of <scene>_<k> -- needed when the arms live in "
                         "separate dirs because their TARGET trajectories differ "
                         "(scripts/render/inject_carry_divisor.py). Plain '<mode>' == tag ''.")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--seam-frames", type=int, default=3,
                    help="how many frames after a boundary keep the red border")
    args = ap.parse_args()
    specs = [m.split(":", 1) if ":" in m else ["", m] for m in args.modes.split(",") if m]
    TAG = {mode: tag for tag, mode in specs}          # mode -> sibling-dir tag
    modes = [mode for _, mode in specs]
    import torch

    outdir = osp.join(args.root, "_stitched"); _os.makedirs(outdir, exist_ok=True)
    fnt, fnt_s, fnt_b = _font(15), _font(12), _font(26)

    tag0 = TAG[modes[0]]
    scenes = defaultdict(list)
    for sd in sorted(_glob.glob(osp.join(args.root, f"*{tag0}_*"))):
        m = re.match(rf"^(.*){re.escape(tag0)}_(\d+)$", osp.basename(sd))
        if not osp.isdir(sd) or not m or not osp.exists(osp.join(sd, "metrics.json")):
            continue
        scenes[m.group(1)].append((int(m.group(2)), sd))

    def seg_dir(scene, idx, mode):
        return osp.join(args.root, f"{scene}{TAG[mode]}_{idx}")

    for scene, items in sorted(scenes.items()):
        items.sort()
        segs = []
        for idx, sd in items:
            met = {}
            for m in modes:
                mp = osp.join(seg_dir(scene, idx, m), "metrics.json")
                met.update({k: v for k, v in json.load(open(mp)).items()} if osp.exists(mp) else {})
            if any(m not in met for m in modes):
                print(f"{scene}_{idx}: missing {[m for m in modes if m not in met]} -- skip scene"); segs = None; break
            vids = {m: np.stack(imageio.mimread(
                osp.join(seg_dir(scene, idx, m), f"render_{SFX.get(m, m)}.mp4"),
                memtest=False))[:, :, :, :3] for m in modes}
            T = min(len(v) for v in vids.values())
            H, W = vids[modes[0]].shape[1:3]
            rip = torch.load(osp.join(sd, "render_inputs.pt"), map_location="cpu", weights_only=False)
            gt = load_gt(rip["tgt_image_paths"], args.size, (H, W))[:T]
            segs.append(dict(idx=idx, T=T, H=H, W=W, vids=vids, gt=gt, metrics=met,
                             frames=rip["seg"].get("frame_idx") if isinstance(rip.get("seg"), dict) else None))
        if not segs:
            continue
        H, W = segs[0]["H"], segs[0]["W"]

        psnr_figure(segs, modes, osp.join(outdir, f"{scene}_psnr.png"), scene)

        # ---- one video per mode: [GT | render]; and one all-mode comparison strip ----
        for tiles, name in [([m], f"{scene}_{SFX.get(m, m)}.mp4") for m in modes] + \
                           ([(modes, f"{scene}_compare.mp4")] if len(modes) > 1 else []):
            frames, gi = [], 0
            for k, sg in enumerate(segs):
                for t in range(sg["T"]):
                    im = Image.fromarray(np.concatenate(
                        [sg["gt"][t]] + [sg["vids"][m][t] for m in tiles], axis=1))
                    banner(im, 0, 0, W, "GT", f"chunk {sg['idx']}  f{t}/{sg['T']-1}  (global {gi})",
                           fnt, fnt_s)
                    for i, m in enumerate(tiles):
                        mm = sg["metrics"][m]
                        banner(im, (i + 1) * W, 0, W, f"{m}  D={mm['divisor']:.2f}",
                               f"PSNR {mm['psnr_per_frame'][t]:.2f}   chunk mean {mm['psnr']:.2f}",
                               fnt, fnt_s)
                    if t < args.seam_frames and k:          # chunk boundary marker
                        d = ImageDraw.Draw(im)
                        d.rectangle([0, 0, im.width - 1, im.height - 1], outline=(255, 0, 0), width=6)
                        d.text((im.width // 2 - 90, im.height - 44), f"CHUNK {sg['idx']}",
                               fill=(255, 60, 60), font=fnt_b)
                    frames.append(np.asarray(im)); gi += 1
            out = osp.join(outdir, name)
            imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
            print(f"{scene}: {len(frames)} frames {frames[0].shape} -> {out}", flush=True)

        summary = {}
        for m in modes:
            seams, base = seam_metric(segs, m)
            summary[m] = {"per_chunk": [{"idx": sg["idx"], "divisor": sg["metrics"][m]["divisor"],
                                         "psnr": sg["metrics"][m]["psnr"],
                                         "ssim": sg["metrics"][m]["ssim"],
                                         "lpips": sg["metrics"][m]["lpips"]} for sg in segs],
                          "seam_ratio": seams, "within_chunk_baseline": base}
            print(f"   {m}: " + "  ".join(
                f"chunk{sg['idx']} D={sg['metrics'][m]['divisor']:.3f} PSNR={sg['metrics'][m]['psnr']:.2f}"
                for sg in segs)
                + f" | seam {'/'.join(f'{s:.2f}' for s in seams)}x (baseline {base:.3f})", flush=True)
        json.dump(summary, open(osp.join(outdir, f"{scene}_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
