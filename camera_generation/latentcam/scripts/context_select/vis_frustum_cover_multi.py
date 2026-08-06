"""frustum_cover selection pattern across several DL3DV scenes (top-down coverage maps)."""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, csv, json
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'main'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dataset_dl3dv import frustum_cover_select

ROOT = '/data1/cympyc1785/data/DL3DV/scenes'
OUT = osp.join(osp.dirname(__file__), 'frustum_cover_multi.png')
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
N_SCENES, K_VIEWS, RADIUS = 4, 6, 2.0


def load_scene(sd):
    tj = json.load(open(osp.join(sd, 'transforms.json')))
    w, h = int(tj['w']), int(tj['h'])
    K = np.array([[tj['fl_x'], 0, tj['cx']], [0, tj['fl_y'], tj['cy']], [0, 0, 1]])
    fr = sorted(tj['frames'], key=lambda f: f['file_path'])
    c2w = np.array([f['transform_matrix'] for f in fr], dtype=np.float64) @ _GL2CV
    return c2w[:, :3, 3], c2w[:, :3, 2], np.linalg.inv(c2w), K, w, h


def main():
    scenes = [r['chunk'].strip() for r in csv.DictReader(open(osp.join(ROOT, 'meta.csv')))]
    picked_scenes = []
    for c in scenes:
        sd = osp.join(ROOT, c)
        if not (osp.isfile(osp.join(sd, 'transforms.json')) and osp.isfile(osp.join(sd, 'prompts.json'))):
            continue
        pr = json.load(open(osp.join(sd, 'prompts.json')))
        seg = next(((int(v['frame_idx'][0]), int(v['frame_idx'][1])) for v in pr.values()
                    if v.get('frame_idx') and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= 49), None)
        if seg:
            picked_scenes.append((c, sd, seg))
        if len(picked_scenes) >= N_SCENES:
            break

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    axes = axes.ravel()
    for ax, (chunk, sd, (s, e)) in zip(axes, picked_scenes):
        centers, faxis, w2c, K, w, h = load_scene(sd)
        seg_scale = max(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean(), 1e-5)
        picked, P, cov, union = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=s,
                                                     seg_scale=seg_scale, k=K_VIEWS, radius=RADIUS,
                                                     n_depth=3, return_debug=True)
        cand = [j for j in range(len(centers)) if np.linalg.norm(centers[j] - centers[s]) <= RADIUS * seg_scale]
        n_in = sum(1 for j in picked if s <= j < e)
        print(f"{chunk.split('/')[-1][:12]} N={len(centers)} seg[{s}:{e}] -> views {picked} "
              f"(in={n_in}/{len(picked)}) cover={100*len(union)/len(P):.0f}%")

        mu = centers.mean(0); Vt = np.linalg.svd(centers - mu, full_matrices=False)[2]
        pca = lambda X: (X - mu) @ Vt[:2].T
        Pc, Pp, Cp = pca(centers), pca(P), pca(centers[picked])
        covered = np.array(sorted(union), int); unc = np.array([i for i in range(len(P)) if i not in union], int)
        if len(unc): ax.scatter(Pp[unc, 0], Pp[unc, 1], s=4, c='lightgray')
        if len(covered): ax.scatter(Pp[covered, 0], Pp[covered, 1], s=5, c='mediumseagreen')
        ax.scatter(Pc[cand, 0], Pc[cand, 1], s=10, c='steelblue', alpha=0.4)
        ax.plot(Pc[s:e, 0], Pc[s:e, 1], '-', lw=1, c='tab:blue')
        ax.scatter(*Pc[s], s=180, marker='*', c='tab:green', zorder=6)
        ax.scatter(Cp[:, 0], Cp[:, 1], s=110, marker='X', c='tab:red', zorder=6)
        for r, j in enumerate(picked):
            d2 = faxis[j] @ Vt[:2].T; d2 = d2 / (np.linalg.norm(d2) + 1e-9) * seg_scale * 0.4
            ax.arrow(Pc[j, 0], Pc[j, 1], d2[0], d2[1], color='tab:red', width=seg_scale * 0.006, alpha=0.6)
            ax.annotate(f'{r}:{j}', Pc[j], fontsize=7, color='tab:red')
        ax.set_title(f"{chunk.split('/')[-1][:10]} seg[{s}:{e}] | cover {100*len(union)/len(P):.0f}% | in {n_in}/{K_VIEWS}", fontsize=9)
        ax.set_aspect('equal', 'datalim')
    fig.suptitle('frustum_cover selection across scenes (green=covered space, redX=6 selected views w/ direction)')
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    print('saved:', OUT)


if __name__ == '__main__':
    main()
