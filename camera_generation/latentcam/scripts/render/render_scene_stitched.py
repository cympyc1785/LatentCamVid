"""Per-scene stitched NVS: for each prompt segment (temporal order), select COVERAGE-aware
context from the LONGER out-of-segment side (frustum_cover restricted to that side),
render the target segment (posed context cam_token), and concatenate all segments into
ONE video per scene. Also (for one scene) visualize which context views each segment used.

Context selection = frustum_cover_select(..., allowed=longer_side): greedy max frustum
coverage among out-of-segment (longer side) candidates -> the "coverage 고려" sampling.

Run (lagernvs env):
  CUDA_VISIBLE_DEVICES=2 python scripts/render_scene_stitched.py --n-scenes 10 --k 6 \
      --viz-scene 0 --stride 3
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
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

LAGER = "/data1/cympyc1785/lagernvs"
MAIN = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/main"
LATENTCAM = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, LAGER)
sys.path.insert(0, MAIN)
sys.path.insert(0, LATENTCAM)   # for `utils.data_utils` imported by dataset_dl3dv
from models.encoder_decoder import EncDec_VitB8
from vggt.utils.load_fn import load_and_preprocess_images
from vis import compute_plucker_coordinates, render_chunked
from render_target_from_context import load_scene, img_path, normalize, ROOT, CKPT
from render_target_pose_ablation import build_cam_tokens

# image dir: 'images_4' (DL3DV-960, 960x540) or 'images_8' (DL3DV-480, 480x270)
def _img_dir(sd, names=('images_4', 'images_8', 'images')):
    for c in names:
        p = osp.join(sd, c)
        if osp.isdir(p):
            return p
    return None


COV_RADIUS = 2.5


def frustum_cover_select(centers, faxis, w2c, K, w, h, anchor, seg_scale,
                         k=6, radius=2.0, n_depth=3, return_debug=False, allowed=None,
                         look_centroid=None, ball_center=None, prepicked=None):
    """Greedy max frustum-coverage view selection (copy of dataset_dl3dv.frustum_cover_select
    to avoid importing the heavy training deps). allowed = candidate frame indices;
    ball_center = explicit coverage-ball center (e.g. frame s / current pos).
    prepicked = frame indices FORCED into the context first (e.g. the always-context first
    view); their coverage is subtracted from the union before greedy picks the residual."""
    N = centers.shape[0]
    c_a = centers[anchor]
    rad = radius * seg_scale
    pool = range(N) if allowed is None else list(allowed)
    cand = [j for j in pool if np.linalg.norm(centers[j] - c_a) <= rad]
    if not cand:
        cand = list(pool) if allowed is not None and len(list(pool)) else [anchor]
    prepicked = list(prepicked) if prepicked else []
    for j in prepicked:                          # fixed context always a candidate (coverage counted)
        if j not in cand:
            cand.append(j)
    if ball_center is not None:
        ball_c = np.asarray(ball_center, dtype=centers.dtype)
    else:
        scene_c = centers.mean(0) if look_centroid is None else look_centroid
        look = np.stack([centers[j] + max(float(np.dot(scene_c - centers[j], faxis[j])),
                                          0.3 * seg_scale) * faxis[j] for j in cand])
        ball_c = look.mean(0)
    g = np.linspace(-rad, rad, 16)
    gx, gy, gz = np.meshgrid(g, g, g, indexing='ij')
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1) + ball_c
    P = P[np.linalg.norm(P - ball_c, axis=1) <= rad]
    if P.shape[0] == 0:
        P = ball_c[None]
    zmax = float(np.linalg.norm(centers[cand] - ball_c, axis=1).max()) + rad

    def covered(j):
        Xc = (w2c[j, :3, :3] @ P.T).T + w2c[j, :3, 3]
        z = Xc[:, 2]
        uv = (K @ (Xc / np.clip(z, 1e-6, None)[:, None]).T).T
        ok = (z > 1e-6) & (z < zmax) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return set()
        dirv = centers[j] - P[idx]
        dirv = dirv / (np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-9)
        b = np.round(dirv * 1.5).astype(int)
        return set((int(i), int(b[t, 0]), int(b[t, 1]), int(b[t, 2])) for t, i in enumerate(idx))

    cov = {j: covered(j) for j in cand}
    picked, union = [], set()
    for j in prepicked:                          # subtract fixed-context coverage first
        if j not in picked:
            picked.append(j); union |= cov[j]
    while len(picked) < min(k, len(cand)):
        best_j = max((j for j in cand if j not in picked),
                     key=lambda x: len(cov[x] - union), default=None)
        if best_j is None:
            break
        gain = len(cov[best_j] - union)
        if gain == 0:
            feat = {x: np.concatenate([centers[x] / max(seg_scale, 1e-5), faxis[x]]) for x in cand}
            remaining = [x for x in cand if x not in picked]
            if not picked and remaining:
                picked.append(min(remaining, key=lambda x: np.linalg.norm(centers[x] - c_a)))
                remaining.remove(picked[-1])
            while len(picked) < min(k, len(cand)) and remaining:
                nxt = max(remaining,
                          key=lambda x: min(np.linalg.norm(feat[x] - feat[p]) for p in picked))
                picked.append(nxt); remaining.remove(nxt)
            break
        picked.append(best_j); union |= cov[best_j]
    return picked


def scene_wh(sd):
    tj = json.load(open(osp.join(sd, "transforms.json")))
    return int(tj['w']), int(tj['h'])


def longer_side_idx(N, s, e):
    before, after = list(range(0, s)), list(range(e, N))
    return before if len(before) >= len(after) else after, ('before' if s >= (N - e) else 'after')


def scene_span_scale(centers):
    """scene-UNIFIED scale: max distance from the first camera to any camera (whole video)."""
    return max(float(np.linalg.norm(centers - centers[0], axis=1).max()), 1e-5)


def select_context(c2w_all, K, w, h, s, e, k, radius, first_view_fixed=False,
                   scale_mode='seg', scene_span_val=None):
    """frustum_cover over the LONGER out-of-segment side (coverage-aware).
    first_view_fixed: always include frame 0 as context (its coverage subtracted first, then
      the residual coverage drives the greedy picks over the longer side).
    scale_mode='scene_span': size the coverage ball by the scene-UNIFIED span (constant across
      segments) instead of the per-segment seg_scale. Coverage METRIC still uses seg_scale."""
    centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]
    w2c = np.linalg.inv(c2w_all)
    N = c2w_all.shape[0]
    side, side_name = longer_side_idx(N, s, e)
    seg_scale = float(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean())
    seg_scale = max(seg_scale, 1e-5)
    ball_scale = scene_span_val if (scale_mode == 'scene_span' and scene_span_val) else seg_scale
    anchor = side[np.argmin([np.linalg.norm(centers[j] - centers[(s + e) // 2]) for j in side])]
    ball_c = centers[s:e].mean(0)                # coverage ball centered on the target segment
    prepicked = [0] if first_view_fixed else None
    picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=anchor,
                                 seg_scale=ball_scale, k=k, radius=radius, allowed=side,
                                 ball_center=ball_c, prepicked=prepicked)
    if len(picks) < k:   # pad by even spacing over the side
        extra = [side[int(round(x))] for x in np.linspace(0, len(side) - 1, k)]
        for j in extra:
            if j not in picks:
                picks.append(j)
            if len(picks) >= k:
                break
    return picks[:k], side, side_name, seg_scale


def viewpoint_coverage(centers, faxis, ctx_idx, s, e, seg_scale, rho=1.0, angle=60.0):
    """fraction of target frames covered by the SELECTED context views (our metric)."""
    tgt_c, tgt_f = centers[s:e], faxis[s:e]
    oc, of = centers[ctx_idx], faxis[ctx_idx]
    dist = np.linalg.norm(tgt_c[:, None] - oc[None], axis=-1) / seg_scale
    ang = tgt_f @ of.T
    return float(((dist <= rho) & (ang >= np.cos(np.deg2rad(angle)))).any(1).mean())


def render_segment(model, sd, c2w_all, fnames, K_frac, w, h, s, e, ctx_idx, stride, res, device, dtype):
    tgt_idx = np.arange(s, e)[::stride]
    ctx_paths = [img_path(sd, fnames[j]) for j in ctx_idx]
    if any(p is None for p in ctx_paths):
        return None, None
    images = load_and_preprocess_images(ctx_paths, mode="resize", target_size=res, patch_size=8)
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
    rays = torch.cat([torch.zeros(1, len(ctx_idx), 6, H, W, device=device), tgt_rays], dim=1)
    cam_tokens = build_cam_tokens(c2w_n, len(ctx_idx), camera_scale, fxfycxcy, (H, W),
                                  context_posed=True).float()   # posed context
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        vid = render_chunked(model, (images, rays, cam_tokens), num_cond_views=len(ctx_idx))
    ren = (vid.float().clamp(0, 1).cpu()[0].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    gt = np.stack([np.asarray(Image.open(img_path(sd, fnames[j])).convert("RGB").resize((W, H)))
                   for j in tgt_idx])
    return gt, ren


def pick_scenes(n, min_segs=4):
    scenes = []
    with open(osp.join(ROOT, 'meta.csv'), newline='') as f:
        chunks = [r['chunk'].strip() for r in csv.DictReader(f)]
    for chunk in chunks:
        sd = osp.join(ROOT, chunk)
        pj = osp.join(sd, 'prompts.json')
        if not (_img_dir(sd) is not None and osp.isfile(pj)
                and osp.isfile(osp.join(sd, 'transforms.json'))):
            continue
        try:
            segs = [k for k, v in json.load(open(pj)).items()
                    if v.get('frame_idx') and len(v['frame_idx']) == 2]
        except Exception:
            continue
        if len(segs) >= min_segs:
            scenes.append(chunk)
        if len(scenes) >= n:
            break
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-scenes', type=int, default=10)
    ap.add_argument('--k', type=int, default=6)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--radius', type=float, default=COV_RADIUS,
                    help='coverage-ball radius (x scale). Use ~0.5 with --scale-mode scene_span.')
    ap.add_argument('--first-view-fixed', action='store_true',
                    help='always use frame 0 as context (coverage subtracted first)')
    ap.add_argument('--scale-mode', default='seg', choices=['seg', 'scene_span'],
                    help='seg: per-segment seg_scale (default). scene_span: scene-unified span.')
    ap.add_argument('--viz-scene', type=int, default=0, help='index into picked scenes to viz')
    ap.add_argument('--viz-all', action='store_true', help='context viz for every scene')
    ap.add_argument('--scenes', default=None, help='comma-separated scene chunks (overrides auto-pick)')
    ap.add_argument('--out-dir', default='/tmp/render_scene')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16

    if args.scenes:
        scenes = [c.strip() for c in args.scenes.split(',') if c.strip()]
        print(f"using {len(scenes)} explicit scenes")
    else:
        scenes = pick_scenes(args.n_scenes)
        print(f"picked {len(scenes)} scenes (>=4 segments)")
    model = EncDec_VitB8(pretrained_vggt=False,
                         attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.to(device).eval()

    for si, chunk in enumerate(scenes):
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd)
        w, h = scene_wh(sd)
        K = np.array([[K_frac[0] * w, 0, K_frac[2] * w],
                      [0, K_frac[1] * h, K_frac[3] * h], [0, 0, 1]], float)
        prompts = json.load(open(osp.join(sd, 'prompts.json')))
        segs = sorted([(int(v['frame_idx'][0]), int(v['frame_idx'][1]), k)
                       for k, v in prompts.items()
                       if v.get('frame_idx') and len(v['frame_idx']) == 2
                       and int(v['frame_idx'][1]) <= c2w_all.shape[0]
                       and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= args.num_frames],
                      key=lambda x: x[0])
        centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]
        span_val = scene_span_scale(centers)

        stitched, seg_meta = [], []
        for (s, e, seg_key) in segs:
            ctx_idx, side, side_name, seg_scale = select_context(
                c2w_all, K, w, h, s, e, args.k, args.radius,
                first_view_fixed=args.first_view_fixed, scale_mode=args.scale_mode,
                scene_span_val=span_val)
            cov = viewpoint_coverage(centers, faxis, ctx_idx, s, e, seg_scale)
            gt, ren = render_segment(model, sd, c2w_all, fnames, K_frac, w, h, s, e,
                                     ctx_idx, args.stride, args.res, device, dtype)
            if gt is None:
                continue
            H, W = gt.shape[1], gt.shape[2]
            for t in range(gt.shape[0]):
                canvas = np.concatenate([gt[t], ren[t]], axis=1)
                im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
                d.text((4, 4), f"GT seg{seg_key}", fill=(255, 255, 0))
                d.text((W + 4, 4), "LagerNVS", fill=(255, 255, 0))
                d.text((4, H - 14), f"cov={cov:.2f} {side_name} K={len(ctx_idx)}",
                       fill=(0, 255, 0))
                stitched.append(np.asarray(im))
            seg_meta.append((s, e, seg_key, ctx_idx, side_name, cov, seg_scale))
            print(f"  scene{si} seg{seg_key} [{s}:{e}] {side_name} cov={cov:.2f} "
                  f"ctx={ctx_idx}")

        if stitched:
            mp4 = osp.join(args.out_dir, f"scene{si:02d}_{chunk.split('/')[-1][:12]}.mp4")
            imageio.mimwrite(mp4, stitched, fps=8, quality=8, macro_block_size=1)
            print(f"scene{si}: {len(segs)} segs -> {osp.basename(mp4)} ({len(stitched)} frames)")

        if (args.viz_all or si == args.viz_scene) and seg_meta:
            viz_context(chunk, c2w_all, seg_meta, args.out_dir, si)

    print(f"out-dir: {args.out_dir}")


def viz_context(chunk, c2w_all, seg_meta, out_dir, si):
    """Per-segment top-down plot: full trajectory + target segment + selected context."""
    centers = c2w_all[:, :3, 3]
    var = centers.var(0)
    ax0, ax1 = np.argsort(var)[-2:]   # two most-varying axes
    labels = {0: 'X', 1: 'Y', 2: 'Z'}
    ns = len(seg_meta)
    ncol = min(4, ns); nrow = int(np.ceil(ns / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow), squeeze=False)
    for i, (s, e, seg_key, ctx_idx, side_name, cov, _) in enumerate(seg_meta):
        a = axes[i // ncol][i % ncol]
        a.scatter(centers[:, ax0], centers[:, ax1], s=6, c='lightgray', label='all frames')
        a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=0.5, alpha=0.5)
        a.scatter(centers[s:e, ax0], centers[s:e, ax1], s=14, c='tab:blue', label=f'target[{s}:{e}]')
        a.scatter(centers[list(ctx_idx), ax0], centers[list(ctx_idx), ax1], s=70,
                  c='tab:red', marker='*', edgecolors='k', label='context (frustum_cover)')
        a.scatter([centers[s, ax0]], [centers[s, ax1]], s=40, c='tab:green', marker='s',
                  label='target start')
        a.set_title(f"seg {seg_key}  {side_name}  cov={cov:.2f}", fontsize=9)
        a.set_xlabel(labels[ax0]); a.set_ylabel(labels[ax1]); a.set_aspect('equal', 'datalim')
        if i == 0:
            a.legend(fontsize=6, loc='best')
    for j in range(ns, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f"scene{si} {chunk.split('/')[-1][:16]}: context selection per segment "
                 f"(top-down {labels[ax0]}{labels[ax1]}); red* = selected out-of-segment context",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = osp.join(out_dir, f"scene{si:02d}_context_viz.png")
    fig.savefig(p, dpi=110); print(f"context viz: {p}")


if __name__ == '__main__':
    main()
