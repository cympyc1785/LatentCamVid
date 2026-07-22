"""A/B: does giving LagerNVS the CONTEXT camera cam_token change the render?

LagerNVS training (train_base.yaml) uses zero_out_cam_cond_p=0.4 -> it sees context
poses 60% of the time; eval configs default to zero_out_cam_cond_p=0.0 (POSED). The
dropout zeros ONLY the CONTEXT views' cam_token & rays (data/normalization.build_cam_cond
lines 104-105); TARGET views always get pose-token + rays. So "context cam_token on/off"
== posed vs unposed. This renders the SAME segments both ways for comparison.

cam_token (11-dim) = pose_enc.extri_intri_to_pose_encoding(c2w, fxfycxcy_px, hw) [9]
                     ++ [camera_scale, world_points_scale=0] [2].
  - unposed: context cam_token zeroed (context rays already zero).
  - posed  : context cam_token = pose encoding of GT context c2w.
  Both: target cam_token = pose encoding (matches build_cam_cond) + target rays.

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=0 python scripts/render_target_pose_ablation.py --n 10 --k 6 --stride 3
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, csv, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio

LAGER = "/data1/cympyc1785/lagernvs"
sys.path.insert(0, LAGER)
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vis import compute_plucker_coordinates, render_chunked
from render_target_from_context import (load_scene, longer_context, img_path,
                                        normalize, pick_segments, ROOT, CKPT, _GL2CV)


def build_cam_tokens(c2w_n, num_cond, camera_scale, fxfycxcy, hw, context_posed):
    """c2w_n: (1,V,4,4) normalized. Returns (1,V,11) cam_token.
    Target views (>=num_cond) always posed; context posed only if context_posed."""
    V = c2w_n.shape[1]
    fxfy = fxfycxcy.view(1, 1, 4).expand(1, V, 4)
    pose9 = extri_intri_to_pose_encoding(c2w_n, fxfy, image_size_hw=hw)   # (1,V,9)
    tok = torch.zeros(1, V, 11, device=c2w_n.device, dtype=pose9.dtype)
    tok[:, :, :9] = pose9
    if not context_posed:
        tok[:, :num_cond, :9] = 0.0                                       # drop context pose
    tok[:, :, 9] = camera_scale                                           # camera_scale
    tok[:, :, 10] = 0.0                                                   # world_points_scale
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/tmp/cov_dump_longer/coverage_all.csv')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--out-dir', default='/tmp/render_pose_ab')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16

    segs = pick_segments(args.dump, args.n)
    print(f"picked {len(segs)} segments")
    model = EncDec_VitB8(pretrained_vggt=False,
                         attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.to(device).eval()

    montage_rows = []
    for i, (scene, seg_key, cov) in enumerate(segs):
        sd = osp.join(ROOT, scene)
        try:
            c2w_all, fnames, K_frac = load_scene(sd)
            prompts = json.load(open(osp.join(sd, "prompts.json")))
            fi = prompts[seg_key]['frame_idx']; s, e = int(fi[0]), int(fi[1])
        except Exception as ex:
            print(f"[{i}] skip: {ex}"); continue
        N = c2w_all.shape[0]
        if e > N:
            continue
        ctx_idx, side = longer_context(N, s, e, args.k)
        tgt_idx = np.arange(s, e)[::args.stride]
        ctx_paths = [img_path(sd, fnames[j]) for j in ctx_idx]
        if any(p is None for p in ctx_paths):
            print(f"[{i}] skip: missing img"); continue

        images = load_and_preprocess_images(ctx_paths, mode="resize",
                                            target_size=args.res, patch_size=8)
        images = images.to(device).unsqueeze(0)
        H, W = images.shape[-2], images.shape[-1]
        c2w_sel = np.concatenate([c2w_all[ctx_idx], c2w_all[tgt_idx]], 0)
        c2w_n, camera_scale = normalize(c2w_sel, len(ctx_idx))
        c2w_n = c2w_n.to(device)
        T = len(tgt_idx)
        fxfycxcy = torch.tensor([K_frac[0] * W, K_frac[1] * H, K_frac[2] * W, K_frac[3] * H],
                                dtype=torch.float32, device=device)
        tgt_rays = compute_plucker_coordinates(c2w_n[:, len(ctx_idx):],
                                               fxfycxcy.view(1, 1, 4).expand(1, T, 4), (H, W))
        cond_rays = torch.zeros(1, len(ctx_idx), 6, H, W, device=device)
        rays = torch.cat([cond_rays, tgt_rays], dim=1)

        outs = {}
        for mode, posed in [('unposed', False), ('posed', True)]:
            cam_tokens = build_cam_tokens(c2w_n, len(ctx_idx), camera_scale,
                                          fxfycxcy, (H, W), posed).float()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
                vid = render_chunked(model, (images, rays, cam_tokens),
                                     num_cond_views=len(ctx_idx))
            outs[mode] = (vid.float().clamp(0, 1).cpu()[0].permute(0, 2, 3, 1).numpy()
                          * 255).astype(np.uint8)

        gt = []
        for j in tgt_idx:
            p = img_path(sd, fnames[j])
            gt.append(np.asarray(Image.open(p).convert("RGB").resize((W, H))))
        gt = np.stack(gt)

        # per-segment mp4: [GT | unposed | posed]
        name = f"{i:02d}_cov{cov:.2f}_{side}_{scene.split('/')[-1][:10]}_{seg_key}"
        frames = []
        for t in range(T):
            canvas = np.concatenate([gt[t], outs['unposed'][t], outs['posed'][t]], axis=1)
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            d.text((4, 4), "GT", fill=(255, 255, 0))
            d.text((W + 4, 4), "unposed ctx", fill=(255, 255, 0))
            d.text((2 * W + 4, 4), "posed ctx", fill=(255, 255, 0))
            d.text((4, H - 14), f"cov={cov:.2f} {side} K={len(ctx_idx)} f{tgt_idx[t]}",
                   fill=(0, 255, 0))
            frames.append(np.asarray(im))
        imageio.mimwrite(osp.join(args.out_dir, name + ".mp4"), frames, fps=8,
                         quality=8, macro_block_size=1)

        # per-segment mean abs diff unposed vs posed
        diff = float(np.abs(outs['unposed'].astype(np.int16) -
                            outs['posed'].astype(np.int16)).mean())
        montage_rows.append(frames[len(frames) // 2])
        print(f"[{i}] cov={cov:.2f} {side} K={len(ctx_idx)} T={T} | "
              f"|unposed-posed| mean={diff:.2f}/255  -> {name}.mp4")

    if montage_rows:
        w = min(r.shape[1] for r in montage_rows)
        mont = np.concatenate([r[:, :w] for r in montage_rows], axis=0)
        Image.fromarray(mont).save(osp.join(args.out_dir, "montage_ab.png"))
        print(f"montage: {osp.join(args.out_dir, 'montage_ab.png')}")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
