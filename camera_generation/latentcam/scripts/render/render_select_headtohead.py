"""Render head-to-head: ours (frustum_cover+dir) vs I3DM-style (MC FOV-overlap, query=frame s)
context selection, same segments, [GT | ours | i3dm]. Both out-of-segment (longer side), K=6,
posed. Quick (few segments) so it can run in a free GPU window.

Run (lagernvs env): CUDA_VISIBLE_DEVICES=3 python scripts/render_select_headtohead.py \
    --segs "chunk:seg,..." --k 6 --stride 4 --out-dir /tmp/render_sel_h2h
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np, torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio

SCR = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, SCR)
from render_scene_stitched import (load_scene, scene_wh, select_context, render_segment,
                                   viewpoint_coverage, ROOT, CKPT)
from select_compare import i3dm_select, longer_side
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
from models.encoder_decoder import EncDec_VitB8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', required=True)
    ap.add_argument('--k', type=int, default=6); ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--stride', type=int, default=4); ap.add_argument('--radius', type=float, default=2.5)
    ap.add_argument('--out-dir', default='/tmp/render_sel_h2h')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16
    rng = np.random.RandomState(0)
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    mids = []
    for i, pr in enumerate(args.segs.split(',')):
        chunk, seg_key = pr.rsplit(':', 1)
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd); w, h = scene_wh(sd)
        K = np.array([[K_frac[0]*w, 0, K_frac[2]*w], [0, K_frac[1]*h, K_frac[3]*h], [0, 0, 1]], float)
        seg = json.load(open(osp.join(sd, 'prompts.json')))[seg_key]
        s, e = int(seg['frame_idx'][0]), int(seg['frame_idx'][1])
        centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]; w2c = np.linalg.inv(c2w_all)
        side, side_name = longer_side(c2w_all.shape[0], s, e)
        seg_scale = max(float(np.linalg.norm(centers[s:e]-centers[s], axis=1).mean()), 1e-5)
        ours, _, _, _ = select_context(c2w_all, K, w, h, s, e, args.k, args.radius)
        i3d = i3dm_select(centers, w2c, K, w, h, s, e, side, args.k, args.radius, rng)
        cov_o = viewpoint_coverage(centers, faxis, ours, s, e, seg_scale)
        cov_i = viewpoint_coverage(centers, faxis, i3d, s, e, seg_scale)
        gt, ren_o = render_segment(model, sd, c2w_all, fnames, K_frac, w, h, s, e, ours, args.stride, args.res, device, dtype)
        _, ren_i = render_segment(model, sd, c2w_all, fnames, K_frac, w, h, s, e, i3d, args.stride, args.res, device, dtype)
        if gt is None or ren_i is None:
            print(f"[{i}] skip"); continue
        H, W = gt.shape[1], gt.shape[2]; frames = []
        for t in range(gt.shape[0]):
            canvas = np.concatenate([gt[t], ren_o[t], ren_i[t]], axis=1)
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            d.text((4, 4), "GT", fill=(255,255,0)); d.text((W+4, 4), f"ours cov{cov_o:.2f}", fill=(255,255,0))
            d.text((2*W+4, 4), f"i3dm cov{cov_i:.2f}", fill=(255,255,0))
            frames.append(np.asarray(im))
        name = f"{i:02d}_{chunk.split('/')[-1][:10]}_{seg_key}_o{cov_o:.2f}_i{cov_i:.2f}"
        imageio.mimwrite(osp.join(args.out_dir, name+".mp4"), frames, fps=8, quality=8, macro_block_size=1)
        mids.append(frames[len(frames)//2])
        print(f"[{i}] {chunk.split('/')[-1][:10]} seg{seg_key} ours_cov={cov_o:.2f} i3dm_cov={cov_i:.2f} "
              f"ours={ours} i3dm={i3d}")
    if mids:
        w0 = min(m.shape[1] for m in mids)
        Image.fromarray(np.concatenate([m[:, :w0] for m in mids], 0)).save(osp.join(args.out_dir, "montage.png"))
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
