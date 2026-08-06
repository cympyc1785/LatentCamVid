"""Demo: geometric co-visibility retrieval of geo-context views for ONE segment.

Method (non-learned, MVSNet-style view selection + FoV/frustum gate + FPS diversity):
  anchor = segment start frame s. look-at point X = c_s + d*f_s (d along optical axis
  to scene centroid). For each candidate frame OUTSIDE [s:e]:
    - frustum/FoV gate: X projects inside the candidate image with depth>0
    - optical-axis gate: f_s . f_cand > cos(max_axis_deg)
    - radius gate: |c_cand - c_s| <= R * seg_scale
    - score: piecewise-Gaussian on triangulation angle at X (favor ~theta0 deg)
  top-M by score -> farthest-point sampling (center+axis) -> K views.

Outputs selected indices and a PNG (top-down camera map + image thumbnails).
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, csv, json, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# image dir: 'images_4' (DL3DV-960, 960x540) or 'images_8' (DL3DV-480, 480x270)
def _img_dir(sd, names=('images_4', 'images_8', 'images')):
    for c in names:
        p = osp.join(sd, c)
        if osp.isdir(p):
            return p
    return None


_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def read_meta(root):
    with open(osp.join(root, 'meta.csv'), newline='') as f:
        return [r['chunk'].strip() for r in csv.DictReader(f)]


def read_blacklist(root):
    p = osp.join(root, 'blacklist.csv'); s = set()
    if osp.isfile(p):
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                s.add(r['scene'].strip())
    return s


def load_scene(scene_dir):
    tj = json.load(open(osp.join(scene_dir, 'transforms.json')))
    w, h = int(tj['w']), int(tj['h'])
    fx, fy, cx, cy = tj['fl_x'], tj['fl_y'], tj['cx'], tj['cy']
    frames = sorted(tj['frames'], key=lambda fr: fr['file_path'])
    c2w_gl = np.array([fr['transform_matrix'] for fr in frames], dtype=np.float64)
    c2w = c2w_gl @ _GL2CV                       # OpenGL -> OpenCV c2w
    centers = c2w[:, :3, 3]                      # (N,3)
    faxis = c2w[:, :3, 2]                        # OpenCV forward = +Z col
    _d = _img_dir(scene_dir)
    files = [osp.join(_d, osp.basename(fr['file_path'])) for fr in frames]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    w2c = np.linalg.inv(c2w)
    return dict(centers=centers, faxis=faxis, files=files, K=K, w=w, h=h, w2c=w2c)


def piecewise_gauss(theta_deg, t0=10.0, s1=5.0, s2=15.0):
    d = theta_deg - t0
    sig = np.where(theta_deg <= t0, s1, s2)
    return np.exp(-(d * d) / (2 * sig * sig))


def project(w2c_j, K, X):
    Xc = w2c_j[:3, :3] @ X + w2c_j[:3, 3]
    z = Xc[2]
    if z <= 1e-6:
        return None, z
    uv = K @ (Xc / z)
    return uv[:2], z


def retrieve(sc, s, e, K_views=4, radius=1.0, t0=10.0, max_axis_deg=60.0, topM=32):
    centers, faxis, K, w, h, w2c = sc['centers'], sc['faxis'], sc['K'], sc['w'], sc['h'], sc['w2c']
    N = centers.shape[0]
    c_s, f_s = centers[s], faxis[s]
    seg = centers[s:e]
    seg_scale = np.linalg.norm(seg - c_s, axis=1).mean()
    centroid = centers.mean(0)
    d = max(np.dot(centroid - c_s, f_s), 0.5 * seg_scale)   # look-at depth along axis
    X = c_s + d * f_s

    cand = [j for j in range(N) if not (s <= j < e)]
    scored = []
    for j in cand:
        if np.linalg.norm(centers[j] - c_s) > radius * seg_scale:
            continue
        if np.dot(f_s, faxis[j]) < np.cos(np.deg2rad(max_axis_deg)):
            continue
        uv, z = project(w2c[j], K, X)
        if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
            continue                                        # frustum/FoV gate
        v1 = c_s - X; v2 = centers[j] - X
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        theta = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
        scored.append((piecewise_gauss(theta, t0), j, theta,
                       np.linalg.norm(centers[j] - c_s) / seg_scale))
    scored.sort(key=lambda x: -x[0])
    scored = scored[:topM]
    if not scored:
        return X, seg_scale, [], []

    # farthest-point sampling over (center/seg_scale, axis) for diversity
    feats = np.array([np.concatenate([centers[j] / seg_scale, faxis[j]]) for _, j, _, _ in scored])
    picks = [0]
    while len(picks) < min(K_views, len(scored)):
        dmat = np.min([np.linalg.norm(feats - feats[p], axis=1) for p in picks], axis=0)
        for p in picks:
            dmat[p] = -1
        picks.append(int(np.argmax(dmat)))
    sel = [scored[p] for p in picks]
    return X, seg_scale, sel, scored


def even_indices(s, e, k):
    return [s + int(round(i)) for i in np.linspace(0, e - 1 - s, k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/scenes')
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--out', default='/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/scripts/covis_demo.png')
    args = ap.parse_args()

    scenes = read_meta(args.root); blk = read_blacklist(args.root)
    chosen = None
    for chunk in scenes:
        if chunk.split('/')[-1] in blk:
            continue
        sd = osp.join(args.root, chunk)
        if not (osp.isfile(osp.join(sd, 'transforms.json')) and osp.isfile(osp.join(sd, 'prompts.json'))):
            continue
        pr = json.load(open(osp.join(sd, 'prompts.json')))
        sc = load_scene(sd); N = sc['centers'].shape[0]
        for kkey, seg in pr.items():
            fi = seg.get('frame_idx')
            if fi and len(fi) == 2 and int(fi[1]) <= N and int(fi[1]) - int(fi[0]) >= args.num_frames:
                chosen = (chunk, sd, sc, kkey, int(fi[0]), int(fi[1]), seg); break
        if chosen:
            break

    chunk, sd, sc, kkey, s, e, seg = chosen
    # HYBRID 3+3: in-segment near-anchor (grounding, minimal leak) + out-of-segment covis
    n_inseg, n_covis = 3, 3
    span = min(args.num_frames // 8, e - 1 - s)
    inseg_idx = [s + int(round(i)) for i in np.linspace(0, span, n_inseg)]
    X, seg_scale, sel, scored = retrieve(sc, s, e, K_views=n_covis)
    covis_idx = [j for _, j, _, _ in sel]
    even_idx = even_indices(s, e, 4)

    print("=" * 64)
    print(f"scene   : {chunk}  (N={sc['centers'].shape[0]} frames)")
    print(f"segment : {kkey}  [s,e]=[{s},{e}]  len={e-s}")
    cap = seg.get('prompt_camera_with_scene_video', {})
    print(f"caption : {cap.get('concise','') if isinstance(cap,dict) else cap}")
    print(f"seg_scale (mean dist from anchor) = {seg_scale:.3f}")
    print("-" * 64)
    print(f"[baseline even (leaky)]   : {even_idx}")
    print(f"[hybrid in-segment  x{n_inseg}] : {inseg_idx}   (near-anchor, span={span}, minimal leak)")
    print(f"[hybrid covis out-of-seg x{n_covis}]: {covis_idx}   (decorrelated scene)")
    print("  covis idx |  score | tri-angle | dist/scale")
    for sco, j, th, dr in sel:
        print(f"      {j:>4} | {sco:>5.3f} |  {th:>6.2f}  |  {dr:>5.2f}")
    print(f"# valid covis candidates after gates: {len(scored)}")
    print(f"=> geo views fed to VGGT = {n_inseg}+{n_covis} = {len(inseg_idx)+len(covis_idx)}")
    print("=" * 64)

    # ---- visualize ----
    centers = sc['centers']
    # PCA to 2D for top-down-ish view
    C0 = centers - centers.mean(0)
    U, S, Vt = np.linalg.svd(C0, full_matrices=False)
    P2 = C0 @ Vt[:2].T
    Xp = (X - centers.mean(0)) @ Vt[:2].T

    fig = plt.figure(figsize=(16, 8))
    axm = fig.add_subplot(1, 2, 1)
    axm.scatter(P2[:, 0], P2[:, 1], s=8, c='lightgray', label='all cameras')
    axm.plot(P2[s:e, 0], P2[s:e, 1], '-o', ms=3, c='tab:blue', label=f'target segment [{s}:{e}]')
    axm.scatter(P2[even_idx, 0], P2[even_idx, 1], s=80, facecolors='none',
                edgecolors='gray', linewidths=1.5, label='baseline even (leaky)')
    axm.scatter(P2[inseg_idx, 0], P2[inseg_idx, 1], s=130, marker='s', c='tab:purple',
                zorder=5, label='hybrid in-segment x3 (near-anchor)')
    axm.scatter(P2[covis_idx, 0], P2[covis_idx, 1], s=150, marker='X', c='tab:red',
                zorder=5, label='hybrid covis x3 (out-of-seg)')
    axm.scatter(*P2[s], s=280, marker='*', c='tab:green', zorder=6, label='anchor (start s)')
    axm.scatter(*Xp, s=120, marker='P', c='orange', zorder=5, label='anchor look-at X')
    for j in covis_idx:
        axm.annotate(str(j), P2[j], fontsize=9, color='tab:red')
    for j in inseg_idx:
        axm.annotate(str(j), P2[j], fontsize=8, color='tab:purple')
    axm.set_title('Hybrid geo views (PCA top-down): 3 in-segment + 3 covis out-of-segment')
    axm.legend(loc='best', fontsize=8); axm.set_aspect('equal', 'datalim')

    # thumbnails: anchor + in-segment(3) + covis(3)
    show = [('anchor %d' % s, s)] + [('in-seg %d' % j, j) for j in inseg_idx] \
        + [('covis %d' % j, j) for j in covis_idx]
    gs = fig.add_gridspec(len(show), 2, left=0.55, right=0.99, top=0.95, bottom=0.05, hspace=0.25)
    for r, (title, idx) in enumerate(show):
        axi = fig.add_subplot(gs[r, :])
        try:
            im = Image.open(sc['files'][idx]).convert('RGB')
            im.thumbnail((360, 360))
            axi.imshow(im)
        except Exception as ex:
            axi.text(0.1, 0.5, f'(no image: {ex})')
        tcol = 'tab:green' if r == 0 else ('tab:purple' if r <= n_inseg else 'tab:red')
        axi.set_title(title, fontsize=10, color=tcol)
        axi.axis('off')

    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f"saved: {args.out}")


if __name__ == '__main__':
    main()
