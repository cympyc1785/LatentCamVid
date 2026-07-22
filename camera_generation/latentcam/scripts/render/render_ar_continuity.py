"""LagerNVS AR continuity render.

Simulate chunk-wise AR *rendering* of a target trajectory and check cross-chunk continuity.
For each chunk the context is chosen from a GROWING causal memory (source out-of-seg frames
+ the already-'generated' past target chunks, using GT frames), anchored at the current
chunk position -> chunk i sees the frames right before it -> continuous boundaries.

Two modes rendered side by side for comparison:
  - AR (rolling memory): context from (source  ∪  target[s : chunk_start]), ball@current pos.
  - static (source-only): context from source only (no memory) — same as independent per-chunk.
Output per scene: [GT | AR | static] stitched mp4 (+ AR-only mp4).

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_ar_continuity.py --segs "chunk:seg,..." \
      --chunk 7 --k 6 --out-dir /tmp/render_ar
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
from render_scene_stitched import (load_scene, scene_wh, img_path, normalize,
                                   frustum_cover_select, ROOT, CKPT)
from render_target_pose_ablation import build_cam_tokens
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vis import compute_plucker_coordinates, render_chunked


def render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt_idx, ctx_idx, res, device, dtype):
    """Render GT poses tgt_idx conditioned on context views ctx_idx (posed)."""
    ctx_paths = [img_path(sd, fnames[j]) for j in ctx_idx]
    if any(p is None for p in ctx_paths) or len(tgt_idx) == 0:
        return None, None
    images = load_and_preprocess_images(ctx_paths, mode="resize", target_size=res, patch_size=8)
    images = images.to(device).unsqueeze(0)
    H, W = images.shape[-2], images.shape[-1]
    c2w_sel = np.concatenate([c2w_all[list(ctx_idx)], c2w_all[list(tgt_idx)]], 0)
    c2w_n, camera_scale = normalize(c2w_sel, len(ctx_idx)); c2w_n = c2w_n.to(device)
    T = len(tgt_idx)
    fxfy = torch.tensor([K_frac[0] * W, K_frac[1] * H, K_frac[2] * W, K_frac[3] * H],
                        dtype=torch.float32, device=device)
    tgt_rays = compute_plucker_coordinates(c2w_n[:, len(ctx_idx):], fxfy.view(1, 1, 4).expand(1, T, 4), (H, W))
    rays = torch.cat([torch.zeros(1, len(ctx_idx), 6, H, W, device=device), tgt_rays], dim=1)
    cam_tokens = build_cam_tokens(c2w_n, len(ctx_idx), camera_scale, fxfy, (H, W), context_posed=True).float()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        vid = render_chunked(model, (images, rays, cam_tokens), num_cond_views=len(ctx_idx))
    ren = (vid.float().clamp(0, 1).cpu()[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    gt = np.stack([np.asarray(Image.open(img_path(sd, fnames[j])).convert("RGB").resize((W, H))) for j in tgt_idx])
    return gt, ren


def pick_ctx(centers, faxis, w2c, K, w, h, memory, cur, k, radius):
    """frustum_cover over the memory pool, ball centered at the current position `cur`."""
    if len(memory) <= k:
        idx = list(memory)
        while len(idx) < k and idx:
            idx.append(idx[-1])
        return idx[:k] if idx else [cur]
    seg_scale = max(float(np.linalg.norm(centers[list(memory)] - centers[cur], axis=1).mean()), 1e-5)
    anchor = min(memory, key=lambda j: np.linalg.norm(centers[j] - centers[cur]))
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=anchor, seg_scale=seg_scale,
                                 k=k, radius=radius, allowed=list(memory), ball_center=centers[cur])
    if len(picks) < k:
        for j in [list(memory)[int(round(x))] for x in np.linspace(0, len(memory) - 1, k)]:
            if j not in picks: picks.append(j)
            if len(picks) >= k: break
    return picks[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segs', required=True, help='comma list chunk:segkey (target = that 49f segment)')
    ap.add_argument('--chunk', type=int, default=7)
    ap.add_argument('--k', type=int, default=6); ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--radius', type=float, default=2.5); ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--out-dir', default='/tmp/render_ar')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    for si, pr in enumerate(args.segs.split(',')):
        chunk, seg_key = pr.rsplit(':', 1)
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd); w, h = scene_wh(sd)
        K = np.array([[K_frac[0]*w, 0, K_frac[2]*w], [0, K_frac[1]*h, K_frac[3]*h], [0, 0, 1]], float)
        seg = json.load(open(osp.join(sd, 'prompts.json')))[seg_key]
        s, e = int(seg['frame_idx'][0]), int(seg['frame_idx'][1])
        N = c2w_all.shape[0]; centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]; w2c = np.linalg.inv(c2w_all)
        before, after = list(range(0, s)), list(range(e, N))
        source = before if len(before) >= len(after) else after     # out-of-seg source (initial memory)

        stitched = []
        chunk_starts = list(range(s, e, args.chunk))
        for cs in chunk_starts:
            ce = min(cs + args.chunk, e)
            tgt = list(range(cs, ce))
            cur = cs if cs > s else s
            mem_ar = list(source) + list(range(s, cs))              # source + generated-so-far (causal)
            ctx_ar = pick_ctx(centers, faxis, w2c, K, w, h, mem_ar, cur, args.k, args.radius)
            ctx_st = pick_ctx(centers, faxis, w2c, K, w, h, list(source), cur, args.k, args.radius)
            gt, ren_ar = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_ar, args.res, device, dtype)
            _, ren_st = render_frames(model, sd, c2w_all, fnames, K_frac, w, h, tgt, ctx_st, args.res, device, dtype)
            if gt is None: continue
            H, W = gt.shape[1], gt.shape[2]
            for t in range(len(tgt)):
                canvas = np.concatenate([gt[t], ren_ar[t], ren_st[t]], axis=1)
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                d.text((4, 4), "GT", fill=(255, 255, 0)); d.text((W+4, 4), "AR(rolling mem)", fill=(255, 255, 0))
                d.text((2*W+4, 4), "static(source)", fill=(255, 255, 0))
                d.text((4, H-14), f"chunk@{cs} f{tgt[t]} |mem_ar|={len(mem_ar)}", fill=(0, 255, 0))
                stitched.append(np.asarray(im))
            print(f"  scene{si} chunk@{cs}[{cs}:{ce}] mem_ar={len(mem_ar)} ctx_ar={ctx_ar}")
        if stitched:
            name = f"ar_{si:02d}_{chunk.split('/')[-1][:10]}_{seg_key}"
            imageio.mimwrite(osp.join(args.out_dir, name+".mp4"), stitched, fps=6, quality=8, macro_block_size=1)
            print(f"scene{si}: {len(chunk_starts)} chunks -> {name}.mp4 ({len(stitched)} frames)")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
