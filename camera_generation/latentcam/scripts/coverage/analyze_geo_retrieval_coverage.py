"""Coverage analysis for I3DM-style geo-context retrieval on DL3DV-960.

Question: for each training segment [s:e], how many frames OUTSIDE the segment lie
spatially near the segment's start camera (by world camera-center distance)? If many
segments have >= geo_num_views such neighbors, world-position retrieval can supply
scene context that is anchored to the start but decorrelated from the target path
(no camera-trajectory leakage). If most segments have ~0, retrieval degenerates to
the current in-segment indexing (short/linear captures) and can't break leakage
without synthetic re-rendering.

Distances are measured in units of the segment's own scale (avg_scale = mean
camera-center distance from the start camera over [s:e]) — the same scale the dataset
uses to normalize. Camera centers are convention-independent (the OpenGL->OpenCV axis
flip does not move the center), so we read them straight from transform_matrix[:3,3].

Usage:
  python scripts/analyze_geo_retrieval_coverage.py [--max-scenes N] [--num-frames 49]
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os
import os.path as osp
import csv
import json
import argparse
import random

import numpy as np
from tqdm import tqdm


def read_meta_scenes(root):
    with open(osp.join(root, 'meta.csv'), newline='') as f:
        return [row['chunk'].strip() for row in csv.DictReader(f)]


def read_blacklist(root):
    p = osp.join(root, 'blacklist.csv')
    blocked = set()
    if osp.isfile(p):
        with open(p, newline='') as f:
            for row in csv.DictReader(f):
                blocked.add(row['scene'].strip())
    return blocked


def parse_centers(tj_path):
    with open(tj_path) as f:
        tj = json.load(f)
    frames = sorted(tj['frames'], key=lambda fr: fr['file_path'])
    C = np.array([fr['transform_matrix'] for fr in frames], dtype=np.float64)  # (N,4,4) c2w
    return C[:, :3, 3]  # (N,3) camera centers (convention-independent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K')
    ap.add_argument('--max-scenes', type=int, default=500)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    radii = [0.5, 1.0, 2.0]          # in units of segment avg_scale
    K_targets = [1, 2, 4]            # geo_num_views thresholds to report
    counts_by_radius = {r: [] for r in radii}  # per-segment #out-of-segment neighbors

    scenes = read_meta_scenes(args.root)
    blocked = read_blacklist(args.root)
    random.seed(args.seed)
    random.shuffle(scenes)           # sample diverse scenes, not just first K

    n_scenes_used = 0
    n_segments = 0
    seg_len_all = []
    frames_per_scene = []

    for chunk in tqdm(scenes):
        if n_scenes_used >= args.max_scenes:
            break
        scene_hash = chunk.split('/')[-1]
        if scene_hash in blocked:
            continue
        scene_dir = osp.join(args.root, chunk)
        tj_path = osp.join(scene_dir, 'transforms.json')
        pj_path = osp.join(scene_dir, 'prompts.json')
        if not (osp.isfile(tj_path) and osp.isfile(pj_path)):
            continue
        try:
            centers = parse_centers(tj_path)
            prompts = json.load(open(pj_path))
        except Exception:
            continue

        n = centers.shape[0]
        used_this_scene = False
        for seg_key, seg in prompts.items():
            fi = seg.get('frame_idx')
            if not fi or len(fi) != 2:
                continue
            s, e = int(fi[0]), int(fi[1])
            if e > n or (e - s) < args.num_frames:
                continue

            start = centers[s]
            seg_c = centers[s:e]
            avg_scale = np.linalg.norm(seg_c - start, axis=1).mean()
            if avg_scale < 1e-6:
                continue

            # out-of-segment candidate frames
            mask = np.ones(n, dtype=bool)
            mask[s:e] = False
            cand = centers[mask]
            if cand.shape[0] == 0:
                for r in radii:
                    counts_by_radius[r].append(0)
                n_segments += 1
                seg_len_all.append(e - s)
                used_this_scene = True
                continue

            d = np.linalg.norm(cand - start, axis=1) / avg_scale
            for r in radii:
                counts_by_radius[r].append(int((d <= r).sum()))
            n_segments += 1
            seg_len_all.append(e - s)
            used_this_scene = True

        if used_this_scene:
            n_scenes_used += 1
            frames_per_scene.append(n)

    # ---- report ----
    print("\n" + "=" * 70)
    print(f"DL3DV geo-retrieval coverage  (sampled {n_scenes_used} scenes, "
          f"{n_segments} segments, num_frames>={args.num_frames})")
    print(f"frames/scene: mean={np.mean(frames_per_scene):.0f} "
          f"median={np.median(frames_per_scene):.0f} "
          f"min={np.min(frames_per_scene):.0f} max={np.max(frames_per_scene):.0f}")
    print(f"segment len : mean={np.mean(seg_len_all):.0f} "
          f"median={np.median(seg_len_all):.0f} "
          f"min={np.min(seg_len_all):.0f} max={np.max(seg_len_all):.0f}")
    print("-" * 70)
    print("radius = R * (segment avg_scale = mean dist of seg cams from start)")
    print(f"{'R':>5} | {'mean#':>7} {'median':>7} | " +
          " ".join(f'%>= {k}v' % () if False else f'frac>={k}' for k in K_targets) +
          f" {'frac=0':>7}")
    for r in radii:
        c = np.array(counts_by_radius[r])
        row = f"{r:>5} | {c.mean():>7.1f} {np.median(c):>7.0f} | "
        row += " ".join(f"{(c >= k).mean():>7.2f}" for k in K_targets)
        row += f" {(c == 0).mean():>7.2f}"
        print(row)
    print("=" * 70)
    print("Read: frac>=4 at R=1.0 is the fraction of segments where I3DM-style "
          "retrieval\ncan supply >=4 out-of-segment neighbors within one segment-"
          "scale of the start\n(i.e., anchored + path-decorrelated context is "
          "available). frac=0 = degenerate\n(retrieval collapses to current "
          "in-segment indexing).")


if __name__ == '__main__':
    main()
