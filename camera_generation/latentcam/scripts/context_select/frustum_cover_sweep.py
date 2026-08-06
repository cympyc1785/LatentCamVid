"""Sweep geo_cover_radius over ~10 DL3DV scenes for the pose-only frustum_cover sampler.
Metrics per radius (mean over scenes):
  cover%      : (point x angle) coverage of the nearby grid ball
  spread      : mean pairwise selected-camera distance / seg_scale (low => padded/degenerate)
  dir_div(deg): mean pairwise angle between selected optical axes (viewpoint diversity)
  degen       : #scenes whose selected spread < 0.3 (near-padded)
Higher cover% + spread + dir_div is better; lower degen is better.
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, csv, json
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'main'))
import numpy as np
from dataset_dl3dv import frustum_cover_select

ROOT = '/data1/cympyc1785/data/DL3DV/scenes'
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
N_SCENES, K_VIEWS = 10, 6
RADII = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def load_scene(sd):
    tj = json.load(open(osp.join(sd, 'transforms.json')))
    w, h = int(tj['w']), int(tj['h'])
    K = np.array([[tj['fl_x'], 0, tj['cx']], [0, tj['fl_y'], tj['cy']], [0, 0, 1]])
    fr = sorted(tj['frames'], key=lambda f: f['file_path'])
    c2w = np.array([f['transform_matrix'] for f in fr], dtype=np.float64) @ _GL2CV
    return c2w[:, :3, 3], c2w[:, :3, 2], np.linalg.inv(c2w), K, w, h


def sel_metrics(centers, faxis, picked, seg_scale):
    C = centers[picked]; F = faxis[picked]
    n = len(picked); dists, angs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(C[i] - C[j]) / seg_scale)
            angs.append(np.rad2deg(np.arccos(np.clip(F[i] @ F[j], -1, 1))))
    return float(np.mean(dists)), float(np.mean(angs))


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
            picked_scenes.append((sd, seg))
        if len(picked_scenes) >= N_SCENES:
            break

    cache = [(load_scene(sd), seg) for sd, seg in picked_scenes]
    print(f"scenes={len(cache)}  k={K_VIEWS}")
    print(f"{'radius':>7} | {'cover%':>7} {'spread':>7} {'dirdeg':>7} {'degen':>6}")
    for R in RADII:
        covs, sprs, dirs, degen = [], [], [], 0
        for (centers, faxis, w2c, K, w, h), (s, e) in cache:
            ss = max(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean(), 1e-5)
            picked, P, pt_cov, pt_union = frustum_cover_select(
                centers, faxis, w2c, K, w, h, anchor=s, seg_scale=ss,
                k=K_VIEWS, radius=R, return_debug=True)
            covs.append(100 * len(pt_union) / max(len(P), 1))
            spr, dd = sel_metrics(centers, faxis, picked, ss)
            sprs.append(spr); dirs.append(dd)
            if spr < 0.3:
                degen += 1
        print(f"{R:>7.1f} | {np.mean(covs):>7.1f} {np.mean(sprs):>7.2f} {np.mean(dirs):>7.1f} {degen:>6}")


if __name__ == '__main__':
    main()
