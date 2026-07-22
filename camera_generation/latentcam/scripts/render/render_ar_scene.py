"""Whole-scene AR continuity render: chunk = a full 49-frame segment. Render ALL segments of
a scene in order, AR-style (each segment's context from the memory of PREVIOUS segments), and
concat -> one long video per scene (clear cross-segment continuity).

Per segment (chunk) i:
  memory = all frames of previous segments [seg0_start : s_i]   (rolling causal; empty for i=0)
  context = [view0] + (k-1) retrieved from memory near s_i
    A "reanchor": view0 = this segment's first camera s_i     -> frame shifts per segment
    B "fixed"   : view0 = first segment's first camera s_0    -> frame fixed to scene start
  segment 0 (no past): bootstrap context = even-k frames within segment 0 (given initial clip).
Also writes a per-segment context viz.

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_ar_scene.py --scenes "chunk1,chunk2,..." --k 6
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse
import numpy as np, torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

SCR = osp.dirname(osp.abspath(__file__)); sys.path.insert(0, SCR)
from render_scene_stitched import load_scene, scene_wh, frustum_cover_select, ROOT, CKPT
from render_ar_continuity import render_frames
from render_ar_anchor import retrieve
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
from models.encoder_decoder import EncDec_VitB8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True, help='comma list of scene chunk paths')
    ap.add_argument('--k', type=int, default=6); ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--radius', type=float, default=2.5); ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--stride', type=int, default=2, help='frame stride within each segment')
    ap.add_argument('--out-dir', default='/tmp/render_ar_scene')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    for si, chunk in enumerate(args.scenes.split(',')):
        chunk = chunk.strip()
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd); w, h = scene_wh(sd)
        K = np.array([[K_frac[0]*w, 0, K_frac[2]*w], [0, K_frac[1]*h, K_frac[3]*h], [0, 0, 1]], float)
        N = c2w_all.shape[0]; centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]; w2c = np.linalg.inv(c2w_all)
        prompts = json.load(open(osp.join(sd, 'prompts.json')))
        segs = sorted([(int(v['frame_idx'][0]), int(v['frame_idx'][1]), kk)
                       for kk, v in prompts.items()
                       if v.get('frame_idx') and len(v['frame_idx']) == 2
                       and int(v['frame_idx'][1]) <= N and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= args.num_frames],
                      key=lambda x: x[0])
        if not segs:
            print(f"scene{si} {chunk[:16]}: no segments"); continue
        s0 = segs[0][0]

        stitched = []; seg_info = []
        for (s, e, sk) in segs:
            tgt = list(range(s, e))[::args.stride]
            mem = list(range(s0, s))                                  # previous segments (causal)
            if mem:
                retr = retrieve(centers, faxis, w2c, K, w, h, mem, s, args.k - 1, args.radius, exclude={s, s0})
                ctx_A = [s] + retr
                ctx_B = [s0] + retr
            else:                                                    # segment 0: bootstrap from itself
                boot = [s + int(round(x)) for x in np.linspace(0, e - 1 - s, args.k)]
                retr = boot[1:]; ctx_A = ctx_B = boot
            gt, ren_A = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_A, args.res, device, dtype)
            _, ren_B = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_B, args.res, device, dtype)
            if gt is None: continue
            H, W = gt.shape[1], gt.shape[2]
            for t in range(len(tgt)):
                canvas = np.concatenate([gt[t], ren_A[t], ren_B[t]], axis=1)
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                d.text((4, 4), "GT", fill=(255, 255, 0)); d.text((W+4, 4), "A: reanchor(view0=seg start)", fill=(255, 255, 0))
                d.text((2*W+4, 4), "B: fixed(view0=scene start)", fill=(255, 255, 0))
                d.text((4, H-14), f"seg{sk}[{s}:{e}] f{tgt[t]} |mem|={len(mem)}", fill=(0, 255, 0))
                stitched.append(np.asarray(im))
            seg_info.append((s, e, sk, retr, s0))
            print(f"  scene{si} seg{sk}[{s}:{e}] |mem|={len(mem)} ctx_A={ctx_A}")
        name = f"arscene_{si:02d}_{chunk.split('/')[-1][:10]}"
        if stitched:
            imageio.mimwrite(osp.join(args.out_dir, name+".mp4"), stitched, fps=10, quality=8, macro_block_size=1)
            print(f"scene{si} -> {name}.mp4 ({len(stitched)} frames, {len(segs)} segments)")
        # per-segment context viz
        if seg_info:
            ax0, ax1 = np.argsort(centers.var(0))[-2:]; Lc = {0: 'X', 1: 'Y', 2: 'Z'}
            ns = len(seg_info); ncol = min(4, ns); nrow = int(np.ceil(ns / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.7*nrow), squeeze=False)
            for ci, (s, e, sk, retr, _s0) in enumerate(seg_info):
                a = axes[ci//ncol][ci%ncol]
                a.scatter(centers[:, ax0], centers[:, ax1], s=5, c='lightgray')
                a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=.5, alpha=.5)
                a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=16, c='tab:blue', label='cur seg')
                if retr: a.scatter(centers[retr, ax0], centers[retr, ax1], s=80, marker='*', c='tab:red', edgecolors='k', linewidths=.5, label='retrieved (past)')
                a.scatter([centers[s, ax0]], [centers[s, ax1]], s=70, c='k', marker='s', label='view0_A=seg start')
                a.scatter([centers[_s0, ax0]], [centers[_s0, ax1]], s=70, c='tab:green', marker='P', label='view0_B=scene start')
                a.set_title(f"seg{sk}[{s}:{e}] |retr|={len(retr)}", fontsize=8)
                a.set_aspect('equal', 'datalim'); a.set_xlabel(Lc[ax0]); a.set_ylabel(Lc[ax1])
                if ci == 0: a.legend(fontsize=6)
            for j in range(ns, nrow*ncol): axes[j//ncol][j%ncol].axis('off')
            fig.suptitle(f"per-segment context (AR, chunk=segment) — {chunk.split('/')[-1][:12]}", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(osp.join(args.out_dir, name+"_ctxviz.png"), dpi=110)
            print(f"       viz -> {name}_ctxviz.png")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
