"""Whole-scene NON-AR (longer out-of-seg context) WITH the view0 anchor ablation.
Per 49-frame segment: retrieve (k-1) views from the LONGER out-of-seg side (ball@s), then
prepend view0:
  A "reanchor": view0 = this segment's first camera s      -> frame shifts per segment
  B "fixed"   : view0 = the FIRST segment's first camera s0 -> frame fixed to scene start
Render all segments, concat -> [GT | A | B] per scene. (= render_scene_longer + the fixed-
first-camera ablation that render_ar_anchor did, but with longer out-of-seg retrieval.)

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_scene_longer_anchor.py --scenes "c1,c2,..." --k 6
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
from render_scene_stitched import load_scene, scene_wh, frustum_cover_select, ROOT, CKPT
from render_ar_continuity import render_frames
from render_scene_longer import ctx_scale
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
from models.encoder_decoder import EncDec_VitB8


def retr_longer(centers, faxis, w2c, K, w, h, s, e, N, k, radius, exclude):
    before, after = list(range(0, s)), list(range(e, N))
    side = before if len(before) >= len(after) else after
    side_name = 'before' if len(before) >= len(after) else 'after'
    side = [j for j in side if j not in exclude]
    if not side:
        return [], side_name
    scale = ctx_scale(centers, side)
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=s, seg_scale=scale,
                                 k=k, radius=radius, allowed=side, ball_center=centers[s])
    picks = [j for j in picks if j not in exclude]
    for j in [side[int(round(x))] for x in np.linspace(0, len(side)-1, k)]:
        if len(picks) >= k: break
        if j not in picks: picks.append(j)
    return picks[:k], side_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True); ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--res', type=int, default=512); ap.add_argument('--radius', type=float, default=2.5)
    ap.add_argument('--num-frames', type=int, default=49); ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--out-dir', default='/tmp/render_scene_longer_anchor')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    for si, chunk in enumerate(args.scenes.split(',')):
        chunk = chunk.strip(); sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd); w, h = scene_wh(sd)
        K = np.array([[K_frac[0]*w, 0, K_frac[2]*w], [0, K_frac[1]*h, K_frac[3]*h], [0, 0, 1]], float)
        N = c2w_all.shape[0]; centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]; w2c = np.linalg.inv(c2w_all)
        prompts = json.load(open(osp.join(sd, 'prompts.json')))
        segs = sorted([(int(v['frame_idx'][0]), int(v['frame_idx'][1]), kk) for kk, v in prompts.items()
                       if v.get('frame_idx') and len(v['frame_idx']) == 2 and int(v['frame_idx'][1]) <= N
                       and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= args.num_frames], key=lambda x: x[0])
        if not segs: print(f"scene{si}: no segments"); continue
        s0 = segs[0][0]

        stitched = []
        for (s, e, sk) in segs:
            tgt = list(range(s, e))[::args.stride]
            retr, side_name = retr_longer(centers, faxis, w2c, K, w, h, s, e, N, args.k - 1, args.radius, exclude={s, s0})
            ctx_A = [s] + retr            # view0 = current segment first camera
            ctx_B = [s0] + retr           # view0 = first segment first camera (scene start), fixed
            gt, ren_A = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_A, args.res, device, dtype)
            _, ren_B = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_B, args.res, device, dtype)
            if gt is None: continue
            H, W = gt.shape[1], gt.shape[2]
            for t in range(len(tgt)):
                canvas = np.concatenate([gt[t], ren_A[t], ren_B[t]], axis=1)
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                d.text((4, 4), "GT", fill=(255, 255, 0)); d.text((W+4, 4), "A: view0=seg start", fill=(255, 255, 0))
                d.text((2*W+4, 4), "B: view0=scene start (fixed)", fill=(255, 255, 0))
                d.text((4, H-14), f"seg{sk}[{s}:{e}] side={side_name} view0_A={s} view0_B={s0}", fill=(0, 255, 0))
                stitched.append(np.asarray(im))
            print(f"  scene{si} seg{sk}[{s}:{e}] side={side_name} ctx_A={ctx_A} ctx_B={ctx_B}")
        name = f"longeranchor_{si:02d}_{chunk.split('/')[-1][:10]}"
        if stitched:
            imageio.mimwrite(osp.join(args.out_dir, name+".mp4"), stitched, fps=10, quality=8, macro_block_size=1)
            print(f"scene{si} -> {name}.mp4 ({len(stitched)} frames, {len(segs)} segments)")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
