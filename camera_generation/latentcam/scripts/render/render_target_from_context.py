"""Render TARGET segment views with LagerNVS from OUT-OF-SEGMENT context only.

Task-honest NVS feasibility probe:
  - target segment [s:e] (49 frames) is EXCLUDED from context (no cheating).
  - context = K evenly-spaced views from the LONGER side (before[0:s] or after[e:N]).
  - render at the GT target camera poses; compare to GT target frames.
Picks ~N segments spanning the coverage spectrum (from the 'longer' coverage dump)
so render quality-vs-coverage is visible, and whether LagerNVS copes with FAR context.

LagerNVS specifics used:
  - context views take ZERO plucker rays (model infers their poses internally via VGGT);
    target poses are conveyed ONLY through target plucker rays.
  - cam_token = zeros except [9]=camera_scale, [10]=0 (camera-based normalization).
  - poses normalized: relative to first context cam, translations / (1.35*max||cond pos||).

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=0 python scripts/render_target_from_context.py --n 10 --k 6 --res 512
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
from vis import compute_plucker_coordinates, render_chunked

# image dir: 'images_4' (DL3DV-960, 960x540) or 'images_8' (DL3DV-480, 480x270)
def _img_dir(sd, names=('images_4', 'images_8', 'images')):
    for c in names:
        p = osp.join(sd, c)
        if osp.isdir(p):
            return p
    return None


_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
CKPT = osp.join(LAGER, "checkpoints/lagernvs_general_512/model.pt")
ROOT = "/data1/cympyc1785/data/DL3DV/scenes"


def load_scene(sd):
    tj = json.load(open(osp.join(sd, "transforms.json")))
    jw, jh = float(tj["w"]), float(tj["h"])
    fl_x, fl_y, cx, cy = tj["fl_x"], tj["fl_y"], tj["cx"], tj["cy"]
    fr = sorted(tj["frames"], key=lambda f: f["file_path"])
    c2w = np.array([f["transform_matrix"] for f in fr], float) @ _GL2CV
    fnames = [osp.basename(f["file_path"]) for f in fr]
    # intrinsics as fractions of image (resolution-independent)
    K_frac = (fl_x / jw, fl_y / jh, cx / jw, cy / jh)
    return c2w, fnames, K_frac


def longer_context(N, s, e, k):
    before, after = np.arange(0, s), np.arange(e, N)
    side = before if len(before) >= len(after) else after
    if len(side) < k:
        idx = side
    else:
        idx = side[[int(round(x)) for x in np.linspace(0, len(side) - 1, k)]]
    return idx, ('before' if side is before else 'after')


def img_path(sd, fname):
    d = _img_dir(sd) or osp.join(sd, "images_4")
    p = osp.join(d, fname)
    if osp.isfile(p):
        return p
    for ext in (".png", ".jpg", ".JPG"):
        q = osp.join(d, osp.splitext(fname)[0] + ext)
        if osp.isfile(q):
            return q
    return None


def normalize(c2w_all, num_cond):
    """relative to first cond cam + camera-based scale (create_target_camera_path)."""
    c2w = torch.tensor(c2w_all, dtype=torch.float32).unsqueeze(0)          # (1,V,4,4)
    first_inv = torch.linalg.inv(c2w[:, 0:1])
    c2w = first_inv @ c2w
    scene_scale = 1.35 * torch.max(torch.norm(c2w[:, :num_cond, :3, 3], dim=-1))
    scene_scale = torch.clamp(scene_scale, min=1e-6)
    c2w[:, :, :3, 3] /= scene_scale
    camera_scale = torch.max(torch.norm(c2w[:, :num_cond, :3, 3], dim=-1)).item()
    return c2w, camera_scale


def pick_segments(csv_path, n, seed=0):
    rows = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append((r['scene'], r['segment'], float(r['coverage'])))
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    per = max(1, n // len(bins))
    rng = np.random.RandomState(seed)
    picked = []
    for lo, hi in bins:
        cand = [r for r in rows if lo <= r[2] < hi]
        rng.shuffle(cand)
        picked_bin, tried = [], 0
        for r in cand:
            sd = osp.join(ROOT, r[0])
            if _img_dir(sd) is not None and osp.isfile(osp.join(sd, "prompts.json")):
                picked_bin.append(r)
            if len(picked_bin) >= per:
                break
            tried += 1
            if tried > 200:
                break
        picked += picked_bin
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='/tmp/cov_dump_longer/coverage_all.csv')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--stride', type=int, default=2, help='render every Nth target frame')
    ap.add_argument('--out-dir', default='/tmp/render_target')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16

    segs = pick_segments(args.dump, args.n)
    print(f"picked {len(segs)} segments across coverage bins")

    model = EncDec_VitB8(pretrained_vggt=False,
                         attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.to(device).eval()

    summary = []
    for i, (scene, seg_key, cov) in enumerate(segs):
        sd = osp.join(ROOT, scene)
        try:
            c2w_all, fnames, K_frac = load_scene(sd)
            prompts = json.load(open(osp.join(sd, "prompts.json")))
            fi = prompts[seg_key]['frame_idx']
            s, e = int(fi[0]), int(fi[1])
        except Exception as ex:
            print(f"[{i}] skip {scene[:20]} {seg_key}: {ex}"); continue
        N = c2w_all.shape[0]
        if e > N:
            print(f"[{i}] skip: e>N"); continue
        ctx_idx, side = longer_context(N, s, e, args.k)
        tgt_idx = np.arange(s, e)[::args.stride]

        ctx_paths = [img_path(sd, fnames[j]) for j in ctx_idx]
        if any(p is None for p in ctx_paths):
            print(f"[{i}] skip: missing ctx img"); continue
        images = load_and_preprocess_images(ctx_paths, mode="resize",
                                            target_size=args.res, patch_size=8)
        images = images.to(device).unsqueeze(0)              # (1,K,3,H,W)
        H, W = images.shape[-2], images.shape[-1]

        c2w_sel = np.concatenate([c2w_all[ctx_idx], c2w_all[tgt_idx]], 0)
        c2w_n, camera_scale = normalize(c2w_sel, len(ctx_idx))
        c2w_n = c2w_n.to(device)
        V = c2w_n.shape[1]; T = len(tgt_idx)
        fxfy = torch.tensor([K_frac[0] * W, K_frac[1] * H, K_frac[2] * W, K_frac[3] * H],
                            dtype=torch.float32, device=device)
        tgt_fxfycxcy = fxfy.view(1, 1, 4).expand(1, T, 4)
        tgt_rays = compute_plucker_coordinates(c2w_n[:, len(ctx_idx):], tgt_fxfycxcy, (H, W))
        cond_rays = torch.zeros(1, len(ctx_idx), 6, H, W, device=device)
        rays = torch.cat([cond_rays, tgt_rays], dim=1)
        cam_tokens = torch.zeros(1, V, 11, device=device)
        cam_tokens[:, :, 9] = camera_scale

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
            vid = render_chunked(model, (images, rays, cam_tokens),
                                 num_cond_views=len(ctx_idx))
        vid = vid.float().clamp(0, 1).cpu()[0]               # (T,3,H,W)

        # GT target frames at render res
        gt = []
        for j in tgt_idx:
            p = img_path(sd, fnames[j])
            g = Image.open(p).convert("RGB").resize((W, H)) if p else Image.new("RGB", (W, H))
            gt.append(np.asarray(g))
        gt = np.stack(gt)                                    # (T,H,W,3) uint8
        ren = (vid.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        # side-by-side [GT | render] video with burned label
        name = f"{i:02d}_cov{cov:.2f}_{side}_{scene.split('/')[-1][:10]}_{seg_key}"
        frames_out = []
        for t in range(T):
            canvas = np.concatenate([gt[t], ren[t]], axis=1)  # (H, 2W, 3)
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            d.text((4, 4), "GT", fill=(255, 255, 0))
            d.text((W + 4, 4), "LagerNVS", fill=(255, 255, 0))
            d.text((4, H - 14), f"cov={cov:.2f} side={side} K={len(ctx_idx)} f{tgt_idx[t]}",
                   fill=(0, 255, 0))
            frames_out.append(np.asarray(im))
        mp4 = osp.join(args.out_dir, name + ".mp4")
        imageio.mimwrite(mp4, frames_out, fps=8, quality=8, macro_block_size=1)

        # context strip
        ctx_np = (images.cpu()[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        strip = np.concatenate(list(ctx_np), axis=1)
        Image.fromarray(strip).save(osp.join(args.out_dir, name + "_ctx.png"))
        print(f"[{i}] cov={cov:.2f} side={side} s={s} e={e} #ctx={len(ctx_idx)} "
              f"#tgt={T} -> {osp.basename(mp4)}")
        summary.append((name, cov, side, len(ctx_idx), T))

    print("\n=== rendered ===")
    for nm, cov, side, kk, T in summary:
        print(f"  cov={cov:.2f} {side:6} K={kk} T={T}  {nm}")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
