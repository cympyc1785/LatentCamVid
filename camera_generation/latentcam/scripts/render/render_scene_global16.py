"""Whole-scene LagerNVS render from GLOBAL context: select k=16 context views by COVERAGE over
Monte-Carlo points sampled INSIDE a sphere centered at the views' average look-at point, then
render ALL scene views (strided) as targets from those 16 context. ~3 scenes.

look-at center = least-squares intersection of all camera optical-axis rays.
sphere radius   = mean ||camera center - look-at center|| (scene scale).
coverage        = greedy max set-cover of (MC-point x view-direction bin) over all views.

Run (lagernvs env): CUDA_VISIBLE_DEVICES=3 python scripts/render_scene_global16.py \
      --n-scenes 3 --k 16 --stride 3 --mc 3000 --out-dir /tmp/render_global16
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

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/data1/cympyc1785/lagernvs")
sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/main")
sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam")
import render_scene_stitched as RS
from render_target_from_context import load_scene, img_path, ROOT, CKPT
from models.encoder_decoder import EncDec_VitB8


def lookat_center(centers, faxis):
    """Least-squares closest point to all optical-axis rays (p_i + t f_i)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, f in zip(centers, faxis):
        f = f / (np.linalg.norm(f) + 1e-9)
        M = np.eye(3) - np.outer(f, f)
        A += M; b += M @ c
    return np.linalg.solve(A + 1e-6 * np.eye(3), b)


def mc_sphere_cover_select(centers, faxis, w2c, K, w, h, center, radius, k=16, mc=3000, seed=0):
    """Greedy max-coverage view selection over Monte-Carlo points in a sphere (center, radius).
    coverage element = (point index, quantized view-direction) so a point seen from a new angle
    still counts (diversity). Returns k view indices."""
    rng = np.random.RandomState(seed)
    # uniform points inside the sphere
    u = rng.randn(mc, 3); u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
    r = radius * rng.rand(mc) ** (1.0 / 3.0)
    P = center[None] + u * r[:, None]
    N = centers.shape[0]
    zmax = float(np.linalg.norm(centers - center, axis=1).max()) + radius

    def covered(j):
        Xc = (w2c[j, :3, :3] @ P.T).T + w2c[j, :3, 3]
        z = Xc[:, 2]
        uv = (K @ (Xc / np.clip(z, 1e-6, None)[:, None]).T).T
        ok = (z > 1e-6) & (z < zmax) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return set()
        dv = centers[j] - P[idx]; dv /= (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
        bb = np.round(dv * 1.5).astype(int)
        return set((int(i), int(bb[t, 0]), int(bb[t, 1]), int(bb[t, 2])) for t, i in enumerate(idx))

    cov = {j: covered(j) for j in range(N)}
    picked, union = [], set()
    while len(picked) < min(k, N):
        best = max((j for j in range(N) if j not in picked),
                   key=lambda x: len(cov[x] - union), default=None)
        if best is None or len(cov[best] - union) == 0:
            break
        picked.append(best); union |= cov[best]
    # pad with farthest-point (viewpoint-diverse) if coverage saturates early
    if len(picked) < k:
        feat = {x: np.concatenate([centers[x] / (radius + 1e-9), faxis[x]]) for x in range(N)}
        rem = [x for x in range(N) if x not in picked]
        while len(picked) < k and rem:
            nxt = max(rem, key=lambda x: min(np.linalg.norm(feat[x] - feat[p]) for p in picked))
            picked.append(nxt); rem.remove(nxt)
    frac = len(union) / max(1, len(set((i,) for i in range(mc))))
    return picked, P, float(len(union))


def coverage_of_targets(centers, faxis, ctx_idx, seg_scale=None, rho=1.0, angle=60.0):
    """our viewpoint-coverage metric over ALL views as targets."""
    ss = seg_scale or float(np.linalg.norm(centers - centers.mean(0), axis=1).mean() + 1e-9)
    oc, of = centers[ctx_idx], faxis[ctx_idx]
    dist = np.linalg.norm(centers[:, None] - oc[None], axis=-1) / ss
    ang = faxis @ of.T
    return float(((dist <= rho) & (ang >= np.cos(np.deg2rad(angle)))).any(1).mean())


def viz(chunk, c2w_all, ctx_idx, center, radius, cov, out_png):
    centers = c2w_all[:, :3, 3]
    ax0, ax1 = np.argsort(centers.var(0))[-2:]; L = {0: 'X', 1: 'Y', 2: 'Z'}
    fig, a = plt.subplots(figsize=(7, 6))
    a.scatter(centers[:, ax0], centers[:, ax1], s=8, c='lightgray', label='all views')
    a.plot(centers[:, ax0], centers[:, ax1], c='lightgray', lw=0.5, alpha=0.5)
    a.scatter(centers[ctx_idx, ax0], centers[ctx_idx, ax1], s=80, c='tab:red', marker='*',
              edgecolors='k', label='16 context (MC-sphere coverage)')
    a.scatter([center[ax0]], [center[ax1]], c='tab:green', s=90, marker='P', label='look-at center')
    th = np.linspace(0, 2 * np.pi, 100)
    a.plot(center[ax0] + radius * np.cos(th), center[ax1] + radius * np.sin(th), 'g--', lw=1, label='sphere')
    a.set_aspect('equal', 'datalim'); a.set_xlabel(L[ax0]); a.set_ylabel(L[ax1])
    a.set_title(f"{chunk.split('/')[-1][:12]}: global 16-context selection  target-coverage={cov:.2f}")
    a.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-scenes', type=int, default=3)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--mc', type=int, default=3000)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--scenes', default=None)
    ap.add_argument('--out-dir', default='/tmp/render_global16')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device, dtype = 'cuda', torch.bfloat16

    scenes = ([c.strip() for c in args.scenes.split(',') if c.strip()]
              if args.scenes else RS.pick_scenes(args.n_scenes))
    print(f"{len(scenes)} scenes")
    model = EncDec_VitB8(pretrained_vggt=False, attention_to_features_type="bidirectional_cross_attention")
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"]); model.to(device).eval()

    for si, chunk in enumerate(scenes):
        sd = osp.join(ROOT, chunk)
        c2w_all, fnames, K_frac = load_scene(sd)
        w, h = RS.scene_wh(sd)
        K = np.array([[K_frac[0] * w, 0, K_frac[2] * w], [0, K_frac[1] * h, K_frac[3] * h], [0, 0, 1]], float)
        centers, faxis = c2w_all[:, :3, 3], c2w_all[:, :3, 2]
        w2c = np.linalg.inv(c2w_all)
        center = lookat_center(centers, faxis)
        radius = float(np.linalg.norm(centers - center, axis=1).mean())
        ctx_idx, P, ucov = mc_sphere_cover_select(centers, faxis, w2c, K, w, h, center, radius,
                                                  k=args.k, mc=args.mc)
        cov = coverage_of_targets(centers, faxis, ctx_idx)
        print(f"scene{si} {chunk.split('/')[-1][:12]}: N={len(centers)} ctx={sorted(ctx_idx)} "
              f"target-coverage={cov:.2f}")
        viz(chunk, c2w_all, ctx_idx, center, radius, cov, osp.join(args.out_dir, f"scene{si:02d}_ctx.png"))

        gt, ren = RS.render_segment(model, sd, c2w_all, fnames, K_frac, w, h, 0, len(centers),
                                    ctx_idx, args.stride, args.res, device, dtype)
        if gt is None:
            print(f"  scene{si}: render skipped (missing ctx image)"); continue
        H, W = gt.shape[1], gt.shape[2]
        frames = []
        for t in range(gt.shape[0]):
            canvas = np.concatenate([gt[t], ren[t]], axis=1)
            im = Image.fromarray(canvas); d = ImageDraw.Draw(im)
            d.text((4, 4), "GT", fill=(255, 255, 0)); d.text((W + 4, 4), "LagerNVS (16 ctx)", fill=(255, 255, 0))
            d.text((4, H - 14), f"cov={cov:.2f} k={len(ctx_idx)}", fill=(0, 255, 0))
            frames.append(np.asarray(im))
        mp4 = osp.join(args.out_dir, f"scene{si:02d}_{chunk.split('/')[-1][:12]}.mp4")
        imageio.mimwrite(mp4, frames, fps=8, quality=8, macro_block_size=1)
        print(f"  scene{si}: {gt.shape[0]} target views -> {osp.basename(mp4)}")
    print(f"out-dir: {args.out_dir}")


if __name__ == '__main__':
    main()
