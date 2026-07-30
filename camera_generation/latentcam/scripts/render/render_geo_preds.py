"""Render the GEO-model GENERATED cameras with LagerNVS. For each saved validation sample
(results/<run>/test/<name>_transforms_pred.json = generated OpenGL c2w, anchored at GT frame s,
in scene world), pick coverage context from the scene and render the generated camera path as
target views. Side by side: LagerNVS @ GT path (ref) | LagerNVS @ generated path (pred).

Run (lagernvs env): CUDA_VISIBLE_DEVICES=3 python scripts/render_geo_preds.py \
      --run results/20260720_181247_dl3dv_geo_viewS --n 10 --k 6 --stride 3 --out-dir /tmp/geo_preds
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, json, argparse, glob
import numpy as np, torch
from PIL import Image, ImageDraw
import imageio.v2 as imageio

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam")
import render_scene_stitched as RS
from render_target_from_context import load_scene, img_path, normalize, ROOT, CKPT
from render_target_pose_ablation import build_cam_tokens
from vggt.utils.load_fn import load_and_preprocess_images
from vis import compute_plucker_coordinates, render_chunked
from models.encoder_decoder import EncDec_VitB8

# image dir: 'images_4' (DL3DV-960, 960x540) or 'images_8' (DL3DV-480, 480x270)
def _img_dir(sd, names=('images_4', 'images_8', 'images')):
    for c in names:
        p = osp.join(sd, c)
        if osp.isdir(p):
            return p
    return None


_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])   # OpenGL c2w -> OpenCV c2w


def load_pred(path):
    j = json.load(open(path))
    m = np.array([f['transform_matrix'] for f in j['frames']], float) @ _GL2CV   # -> OpenCV c2w
    return m


@torch.no_grad()
def render_poses(model, sd, c2w_all, fnames, K_frac, w, h, ctx_idx, target_c2w, res, device, dtype):
    ctx_paths = [img_path(sd, fnames[j]) for j in ctx_idx]
    if any(p is None for p in ctx_paths):
        return None
    images = load_and_preprocess_images(ctx_paths, mode="resize", target_size=res, patch_size=8).to(device).unsqueeze(0)
    H, W = images.shape[-2], images.shape[-1]
    c2w_sel = np.concatenate([c2w_all[ctx_idx], target_c2w], 0)
    c2w_n, camera_scale = normalize(c2w_sel, len(ctx_idx)); c2w_n = c2w_n.to(device)
    T = len(target_c2w)
    fxfycxcy = torch.tensor([K_frac[0] * W, K_frac[1] * H, K_frac[2] * W, K_frac[3] * H],
                            dtype=torch.float32, device=device)
    tgt_rays = compute_plucker_coordinates(c2w_n[:, len(ctx_idx):], fxfycxcy.view(1, 1, 4).expand(1, T, 4), (H, W))
    rays = torch.cat([torch.zeros(1, len(ctx_idx), 6, H, W, device=device), tgt_rays], dim=1)
    cam_tokens = build_cam_tokens(c2w_n, len(ctx_idx), camera_scale, fxfycxcy, (H, W), context_posed=True).float()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        vid = render_chunked(model, (images, rays, cam_tokens), num_cond_views=len(ctx_idx))
    return (vid.float().clamp(0, 1).cpu()[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--names', default=None,
                    help='comma-separated sample names to render (overrides alphabetical first-N)')
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--radius', type=float, default=2.5)
    ap.add_argument('--out-dir', default='/tmp/geo_preds')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16

    if args.names:
        names = [n.strip() for n in args.names.split(',') if n.strip()]
    else:
        names = [osp.basename(p)[:-len('_transforms_pred.json')]
                 for p in sorted(glob.glob(osp.join(args.run, 'test', '*_transforms_pred.json')))]
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    done = 0
    for name in names:
        if done >= args.n:
            break
        parts = name.split('_')                       # ['1K', '<hash>', '<seg>']
        chunk = f"{parts[0]}/{parts[1]}"
        sd = osp.join(ROOT, chunk)
        if _img_dir(sd) is None:
            continue
        c2w_all, fnames, K_frac = load_scene(sd)
        w, h = RS.scene_wh(sd)
        K = np.array([[K_frac[0] * w, 0, K_frac[2] * w], [0, K_frac[1] * h, K_frac[3] * h], [0, 0, 1]], float)
        pred = load_pred(osp.join(args.run, 'test', f"{name}_transforms_pred.json"))
        ref = load_pred(osp.join(args.run, 'test', f"{name}_transforms_ref.json"))
        centers = c2w_all[:, :3, 3]
        # recover segment [s:e] by matching ref endpoints to scene camera centers
        s = int(np.argmin(np.linalg.norm(centers - ref[0, :3, 3], axis=1)))
        e = int(np.argmin(np.linalg.norm(centers - ref[-1, :3, 3], axis=1))) + 1
        if e <= s:
            s, e = min(s, e), max(s, e) + 1
        ctx_idx, side, sname, _ = RS.select_context(c2w_all, K, w, h, s, e, args.k, args.radius)
        sub = np.arange(0, len(pred), args.stride)
        ren_ref = render_poses(model, sd, c2w_all, fnames, K_frac, w, h, ctx_idx, ref[sub], args.res, device, dtype)
        ren_pred = render_poses(model, sd, c2w_all, fnames, K_frac, w, h, ctx_idx, pred[sub], args.res, device, dtype)
        if ren_ref is None or ren_pred is None:
            print(f"  {name}: skipped (missing ctx img)"); continue
        H, W = ren_pred.shape[1], ren_pred.shape[2]
        frames = []
        for t in range(len(sub)):
            canvas = np.concatenate([ren_ref[t], ren_pred[t]], axis=1)
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            d.text((4, 4), "LagerNVS @ GT path", fill=(255, 255, 0))
            d.text((W + 4, 4), "LagerNVS @ GENERATED path", fill=(255, 255, 0))
            d.text((4, H - 14), f"seg[{s}:{e}] ctx={sname} k={len(ctx_idx)}", fill=(0, 255, 0))
            frames.append(np.asarray(im))
        mp4 = osp.join(args.out_dir, f"{name[:28]}_seg{parts[2]}.mp4")
        imageio.mimwrite(mp4, frames, fps=8, quality=8, macro_block_size=1)
        print(f"{name[:28]} seg{parts[2]}: [{s}:{e}] {sname} ctx={ctx_idx} -> {osp.basename(mp4)}", flush=True)
        done += 1
    print(f"rendered {done} samples -> {args.out_dir}")


if __name__ == '__main__':
    main()
