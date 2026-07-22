"""Side-by-side comparison render: for each segment, select context TWO ways, render both with
LagerNVS, and build a 3-panel frame [GT | per-seg-scale | first-fixed+scene_span]. Stitch all
segments into ONE video per scene. GT kept on the LEFT.

  panel A "per-seg scale": original select_context — per-segment seg_scale re-computed each
                            segment, NO fixed first view (scale "context 뽑을 때마다 새로 구함").
  panel B "firstfix+span":  --first-view-fixed + --scale-mode scene_span (frame 0 always context,
                            coverage subtracted first; ball sized by scene-unified span).

Run (lagernvs env): CUDA_VISIBLE_DEVICES=3 python scripts/render_scene_compare.py \
      --n-scenes 3 --k 6 --stride 3 --radius-seg 2.5 --radius-span 0.5 --out-dir /tmp/render_cmp
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import render_scene_stitched as R
from render_target_from_context import load_scene, ROOT, CKPT
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
from models.encoder_decoder import EncDec_VitB8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-scenes', type=int, default=3)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--radius-seg', type=float, default=2.5, help='ball radius x seg_scale (panel A)')
    ap.add_argument('--radius-span', type=float, default=0.5, help='ball radius x scene_span (panel B)')
    ap.add_argument('--scenes', default=None, help='comma-separated scene chunks (overrides auto-pick)')
    ap.add_argument('--out-dir', default='/tmp/render_cmp')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16

    scenes = ([c.strip() for c in args.scenes.split(',') if c.strip()]
              if args.scenes else R.pick_scenes(args.n_scenes))
    print(f"{len(scenes)} scenes")
    model = EncDec_VitB8(pretrained_vggt=False,
                         attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.to(device).eval()

    for si, chunk in enumerate(scenes):
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd)
        w, h = R.scene_wh(sd)
        K = np.array([[K_frac[0] * w, 0, K_frac[2] * w],
                      [0, K_frac[1] * h, K_frac[3] * h], [0, 0, 1]], float)
        prompts = json.load(open(osp.join(sd, 'prompts.json')))
        N = c2w_all.shape[0]
        segs = sorted([(int(v['frame_idx'][0]), int(v['frame_idx'][1]), k)
                       for k, v in prompts.items()
                       if v.get('frame_idx') and len(v['frame_idx']) == 2
                       and int(v['frame_idx'][1]) <= N
                       and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= args.num_frames],
                      key=lambda x: x[0])
        centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]
        span_val = R.scene_span_scale(centers)

        stitched = []
        for (s, e, seg_key) in segs:
            ctxA, _, snA, ssA = R.select_context(c2w_all, K, w, h, s, e, args.k, args.radius_seg,
                                                 first_view_fixed=False, scale_mode='seg')
            ctxB, _, snB, ssB = R.select_context(c2w_all, K, w, h, s, e, args.k, args.radius_span,
                                                 first_view_fixed=True, scale_mode='scene_span',
                                                 scene_span_val=span_val)
            covA = R.viewpoint_coverage(centers, faxis, ctxA, s, e, ssA)
            covB = R.viewpoint_coverage(centers, faxis, ctxB, s, e, ssB)
            gt, renA = R.render_segment(model, sd, c2w_all, fnames, K_frac, w, h, s, e,
                                        ctxA, args.stride, args.res, device, dtype)
            _, renB = R.render_segment(model, sd, c2w_all, fnames, K_frac, w, h, s, e,
                                       ctxB, args.stride, args.res, device, dtype)
            if gt is None or renA is None or renB is None:
                continue
            H, W = gt.shape[1], gt.shape[2]
            for t in range(gt.shape[0]):
                canvas = np.concatenate([gt[t], renA[t], renB[t]], axis=1)
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                d.text((4, 4), f"GT seg{seg_key}", fill=(255, 255, 0))
                d.text((W + 4, 4), "per-seg scale", fill=(255, 255, 0))
                d.text((2 * W + 4, 4), "firstfix+span", fill=(255, 255, 0))
                d.text((W + 4, H - 14), f"cov={covA:.2f} {snA}", fill=(0, 255, 0))
                d.text((2 * W + 4, H - 14), f"cov={covB:.2f} {snB}", fill=(0, 255, 0))
                stitched.append(np.asarray(im))
            print(f"  scene{si} seg{seg_key} [{s}:{e}] A(cov {covA:.2f} ctx{sorted(ctxA)}) "
                  f"B(cov {covB:.2f} ctx{sorted(ctxB)})")
        if stitched:
            mp4 = osp.join(args.out_dir, f"cmp_scene{si:02d}_{chunk.split('/')[-1][:12]}.mp4")
            imageio.mimwrite(mp4, stitched, fps=8, quality=8, macro_block_size=1)
            print(f"scene{si}: {len(segs)} segs -> {osp.basename(mp4)} ({len(stitched)} frames)")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
