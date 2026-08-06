"""Visualize 'frustum_cover' geo sampling: k views (in/out ignored) whose frustums
cover the most nearby observed space (greedy max coverage). Samples ONE segment."""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, csv, json
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'main'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dataset_dl3dv import frustum_cover_select

# image dir: 'images_4' (DL3DV-960, 960x540) or 'images_8' (DL3DV-480, 480x270)
def _img_dir(sd, names=('images_4', 'images_8', 'images')):
    for c in names:
        p = osp.join(sd, c)
        if osp.isdir(p):
            return p
    return None


ROOT = '/data1/cympyc1785/data/DL3DV/scenes'
OUT = osp.join(osp.dirname(__file__), 'frustum_cover_demo.png')
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
K_VIEWS, RADIUS = 6, 2.0


def load_scene(sd):
    tj = json.load(open(osp.join(sd, 'transforms.json')))
    w, h = int(tj['w']), int(tj['h'])
    K = np.array([[tj['fl_x'], 0, tj['cx']], [0, tj['fl_y'], tj['cy']], [0, 0, 1]])
    fr = sorted(tj['frames'], key=lambda f: f['file_path'])
    c2w = np.array([f['transform_matrix'] for f in fr], dtype=np.float64) @ _GL2CV
    _d = _img_dir(sd)
    files = [osp.join(_d, osp.basename(f['file_path'])) for f in fr]
    return c2w[:, :3, 3], c2w[:, :3, 2], np.linalg.inv(c2w), K, w, h, files


def main():
    scenes = [r['chunk'].strip() for r in csv.DictReader(open(osp.join(ROOT, 'meta.csv')))]
    chunk = next(c for c in scenes
                 if osp.isfile(osp.join(ROOT, c, 'transforms.json'))
                 and osp.isfile(osp.join(ROOT, c, 'prompts.json')))
    sd = osp.join(ROOT, chunk)
    centers, faxis, w2c, K, w, h, files = load_scene(sd)
    pr = json.load(open(osp.join(sd, 'prompts.json')))
    s, e = next((int(v['frame_idx'][0]), int(v['frame_idx'][1])) for v in pr.values()
                if v.get('frame_idx') and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= 49)
    seg_scale = np.linalg.norm(centers[s:e] - centers[s], axis=1).mean()

    picked, P, cov, union = frustum_cover_select(
        centers, faxis, w2c, K, w, h, anchor=s, seg_scale=seg_scale,
        k=K_VIEWS, radius=RADIUS, n_depth=3, return_debug=True)
    cand = [j for j in range(len(centers)) if np.linalg.norm(centers[j] - centers[s]) <= RADIUS * seg_scale]

    print(f"scene {chunk} (N={len(centers)}), seg[{s}:{e}], seg_scale={seg_scale:.2f}")
    print(f"candidates (near, in+out): {len(cand)} | proxy pts P={len(P)}")
    print(f"selected {len(picked)} views: {picked}  ({'in-seg' if any(s<=j<e for j in picked) else ''})")
    for r, j in enumerate(picked):
        print(f"  view{r}: frame {j}  {'[IN-seg]' if s<=j<e else '[out]'}  covers {len(cov[j])} pts")
    print(f"coverage: {len(union)}/{len(P)} = {100*len(union)/len(P):.1f}% of nearby space")

    # PCA top-down
    mu = centers.mean(0); Vt = np.linalg.svd(centers - mu, full_matrices=False)[2]
    def pca(X): return (X - mu) @ Vt[:2].T
    Pc, Pp, Cp = pca(centers), pca(P), pca(centers[picked])

    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(1, 2, 1)
    covered = np.array(sorted(union), dtype=int)
    unc = np.array([i for i in range(len(P)) if i not in union], dtype=int)
    if len(unc): ax.scatter(Pp[unc, 0], Pp[unc, 1], s=5, c='lightgray', label='space (uncovered)')
    if len(covered): ax.scatter(Pp[covered, 0], Pp[covered, 1], s=6, c='mediumseagreen', label='space (covered)')
    ax.scatter(Pc[:, 0], Pc[:, 1], s=6, c='silver')
    ax.scatter(Pc[cand, 0], Pc[cand, 1], s=14, c='steelblue', alpha=0.5, label='candidate cams (near)')
    ax.plot(Pc[s:e, 0], Pc[s:e, 1], '-', lw=1, c='tab:blue', label=f'segment [{s}:{e}]')
    ax.scatter(*Pc[s], s=240, marker='*', c='tab:green', zorder=6, label='anchor s')
    ax.scatter(Cp[:, 0], Cp[:, 1], s=150, marker='X', c='tab:red', zorder=6, label='selected 6')
    # view-direction arrows for selected
    for r, j in enumerate(picked):
        d2 = (faxis[j] @ Vt[:2].T); d2 = d2 / (np.linalg.norm(d2) + 1e-9) * seg_scale * 0.4
        ax.arrow(Pc[j, 0], Pc[j, 1], d2[0], d2[1], color='tab:red', width=seg_scale*0.008, alpha=0.7)
        ax.annotate(f'{r}:{j}', Pc[j], fontsize=8, color='tab:red')
    ax.set_title(f'frustum_cover: {len(picked)} views cover {100*len(union)/len(P):.0f}% nearby space')
    ax.legend(fontsize=7, loc='best'); ax.set_aspect('equal', 'datalim')

    # thumbnails
    gs = fig.add_gridspec(3, 2, left=0.55, right=0.99, top=0.95, bottom=0.05, hspace=0.3, wspace=0.05)
    for r, j in enumerate(picked[:6]):
        axi = fig.add_subplot(gs[r // 2, r % 2])
        try:
            im = Image.open(files[j]).convert('RGB'); im.thumbnail((300, 300)); axi.imshow(im)
        except Exception as ex:
            axi.text(0.1, 0.5, str(ex))
        axi.set_title(f"view{r} frame{j} {'[in]' if s<=j<e else '[out]'}", fontsize=8,
                      color='tab:green' if r == 0 else 'tab:red'); axi.axis('off')
    fig.suptitle('frustum_cover geo sampling (in/out ignored; view0 = max-coverage = VGGT ref)')
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    print('saved:', OUT)


if __name__ == '__main__':
    main()
