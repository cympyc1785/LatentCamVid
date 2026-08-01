"""Render what a moving normalization divisor costs when chunks are CHAINED.

Why this is not just "render each chunk with a different divisor": normalizing context AND target
by the same D is a global similarity transform of the scene, and `camera_scale` is a conditioning
TOKEN of LagerNVS (tools/lagernvs/data/normalization.py:116-121), so the renderer is scale-
EQUIVARIANT -- changing D alone leaves the image essentially unchanged (measured: PSNR moves
< 0.3 dB even for a 5.7x divisor change). D is not a renderer property, it is the units the
camera DM's output is expressed in.

Where it does bite: the DM emits a NORMALIZED trajectory t_hat_k for chunk k, and you place it in
the world by multiplying by a divisor. A chained/AR generator fixes the scale once (chunk 0's D_0)
and keeps emitting into it, so chunk k lands in the world as

    world_k = D_0 * t_hat_k = (D_0 / D_k) * GT_k          (relative to the chunk's frame s)

i.e. the camera travels D_0/D_k times too far. That ratio IS the spread metric of
scripts/data/chunk_divisor_stability.py. This script materialises exactly that trajectory:
target translations (relative to the chunk's own first pose) scaled by D_0/D_k, rotations and the
geo CONTEXT left untouched, so the render is against the true scene and PSNR vs the GT frames
measures the drift directly. Chunk 0 has ratio 1 by construction -> it is the control.

Writes <root>_drift/<scene>__<arm>_<k>/render_inputs.pt with `scales` reduced to that one arm, so

  /data1/cympyc1785/miniconda3/envs/latentcam/bin/python scripts/render/inject_carry_divisor.py \
      --root results/chunk_continuity_dv --arms ctx_longer_135max,lagernvs
  RD_ROOT=.../results/chunk_continuity_dv_drift RD_MODES=ctx_longer_135max,lagernvs \
      python tools/lagernvs/render_avgscale.py
"""
import argparse, glob, os, os.path as osp, re
from collections import defaultdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", default="ctx_longer_135max,lagernvs")
    ap.add_argument("--out", default=None, help="default <root>_drift")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    out_root = args.out or (args.root.rstrip("/") + "_drift")

    scenes = defaultdict(list)
    for sd in sorted(glob.glob(osp.join(args.root, "*"))):
        m = re.match(r"^(.*)_(\d+)$", osp.basename(sd))
        if m and osp.exists(osp.join(sd, "render_inputs.pt")):
            scenes[m.group(1)].append((int(m.group(2)), sd))

    for scene, items in sorted(scenes.items()):
        items.sort()
        d0 = torch.load(osp.join(items[0][1], "render_inputs.pt"),
                        map_location="cpu", weights_only=False)
        for arm in arms:
            D0 = float(d0["scales"][arm])
            ratios = []
            for k, sd in items:
                d = torch.load(osp.join(sd, "render_inputs.pt"), map_location="cpu",
                               weights_only=False)
                r = D0 / float(d["scales"][arm])
                ratios.append(r)
                tgt = d["tgt_c2w"].float()
                rel = torch.linalg.inv(tgt[0:1]) @ tgt          # relative to the chunk's frame s
                rel = rel.clone(); rel[:, :3, 3] = rel[:, :3, 3] * r
                d = dict(d)
                d["tgt_c2w"] = tgt[0:1] @ rel
                d["scales"] = {arm: float(d["scales"][arm])}
                d["carry_ratio"] = r
                od = osp.join(out_root, f"{scene}__{arm}_{k}")
                os.makedirs(od, exist_ok=True)
                torch.save(d, osp.join(od, "render_inputs.pt"))
            print(f"{scene} [{arm}] D_0={D0:.3f}  D_0/D_k = "
                  + " / ".join(f"{r:.2f}" for r in ratios), flush=True)
    print(f"-> {out_root}")


if __name__ == "__main__":
    main()
