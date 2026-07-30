"""Statistical comparison of camera-normalization scales (Part A of the scene-unified scale study).

Three candidate scales, all from camera CENTER distances (frame-invariant, so computed from
transforms.json c2w directly; identical math to dataset_dl3dv):
  scene_span (NEW, scene-UNIFIED): max_i ||center[i] - center[0]|| over the WHOLE video.
                                   One value per scene (first camera is always context anchor).
  cam_dist_mean (per-segment):        mean_i ||center[s:e] - center[s]||  (== _cam_dist_mean_scale).
                                   Depends on the target -> changes every segment.
  context_longer (per-segment):    mean over num_frames-windows of the LONGER out-of-segment side
                                   of each window's mean center-norm (== _cam_dist_mean_context).
                                   Target-free but still changes every segment.

We quantify the user's concern ("context 기준 normalize면 scale이 계속 변한다"): within-scene
variability (coefficient of variation) of the per-segment scales vs the single scene_span, and
the magnitude ratios scene_span/cam_dist_mean etc. Raw per-scene numbers dumped to CSV; aggregate
percentiles printed.

Run: python scripts/compare_scene_span_scale.py --n-scenes 200 --num-frames 49
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, sys, csv, json, argparse
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from render_target_from_context import load_scene, ROOT


def cam_dist_mean_scale(centers, s, e):
    return float(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean())


def context_longer_scale(centers, s, e, T):
    """Mirror dataset_dl3dv._cam_dist_mean_context: longer out-of-seg side, chunk into
    T-frame windows, each window's mean center-norm rel to its first frame, averaged."""
    N = centers.shape[0]
    side = list(range(0, s)) if s >= (N - e) else list(range(e, N))
    chunks = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]
    if not chunks and len(side) >= 2:
        chunks = [side]
    if not chunks:
        return None
    ws = [float(np.linalg.norm(centers[ch] - centers[ch[0]], axis=1).mean()) for ch in chunks]
    return float(np.mean(ws))


def scene_span(centers):
    return float(np.linalg.norm(centers - centers[0], axis=1).max())


def scene_segments(sd, num_frames, N):
    pj = osp.join(sd, 'prompts.json')
    if not osp.isfile(pj):
        return []
    try:
        pr = json.load(open(pj))
    except Exception:
        return []
    segs = sorted([(int(v['frame_idx'][0]), int(v['frame_idx'][1]))
                   for v in pr.values()
                   if v.get('frame_idx') and len(v['frame_idx']) == 2
                   and int(v['frame_idx'][1]) <= N
                   and int(v['frame_idx'][1]) - int(v['frame_idx'][0]) >= num_frames],
                  key=lambda x: x[0])
    return segs


def cv(xs):
    xs = np.asarray(xs, float)
    return float(xs.std() / xs.mean()) if len(xs) >= 2 and xs.mean() > 0 else 0.0


def pct(xs, ps=(5, 25, 50, 75, 95)):
    xs = np.asarray(xs, float)
    return {p: float(np.percentile(xs, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-scenes', type=int, default=200)
    ap.add_argument('--num-frames', type=int, default=49)
    ap.add_argument('--min-segs', type=int, default=4)
    ap.add_argument('--out', default='/tmp/scale_compare/scene_scale_compare.csv')
    args = ap.parse_args()
    os.makedirs(osp.dirname(args.out), exist_ok=True)

    with open(osp.join(ROOT, 'meta.csv'), newline='') as f:
        chunks = [r['chunk'].strip() for r in csv.DictReader(f)]

    rows = []           # per-segment
    scene_rows = []     # per-scene aggregate
    for chunk in chunks:
        if len([r for r in scene_rows]) >= args.n_scenes:
            break
        sd = osp.join(ROOT, chunk)
        if not (osp.isfile(osp.join(sd, 'transforms.json')) and osp.isfile(osp.join(sd, 'prompts.json'))):
            continue
        try:
            c2w, _, _ = load_scene(sd)
        except Exception:
            continue
        N = c2w.shape[0]
        centers = c2w[:, :3, 3]
        segs = scene_segments(sd, args.num_frames, N)
        if len(segs) < args.min_segs:
            continue
        span = scene_span(centers)
        tcs, cls = [], []
        for (s, e) in segs:
            tc = cam_dist_mean_scale(centers, s, e)
            cl = context_longer_scale(centers, s, e, args.num_frames)
            tcs.append(tc)
            cls.append(cl if cl is not None else np.nan)
            rows.append((chunk, s, e, span, tc, cl if cl is not None else ''))
        tcs = np.array(tcs, float)
        cls_valid = np.array([x for x in cls if x == x], float)   # drop nan
        scene_rows.append({
            'chunk': chunk, 'n_seg': len(segs), 'scene_span': span,
            'cam_dist_mean_mean': float(tcs.mean()), 'cam_dist_mean_cv': cv(tcs),
            'context_longer_mean': float(cls_valid.mean()) if len(cls_valid) else np.nan,
            'context_longer_cv': cv(cls_valid) if len(cls_valid) >= 2 else np.nan,
            'span_over_targetmean': span / tcs.mean() if tcs.mean() > 0 else np.nan,
            'span_over_ctxmean': (span / cls_valid.mean()) if len(cls_valid) and cls_valid.mean() > 0 else np.nan,
        })

    # dump per-segment CSV
    with open(args.out, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['chunk', 's', 'e', 'scene_span', 'cam_dist_mean', 'context_longer'])
        wtr.writerows(rows)
    scene_csv = args.out.replace('.csv', '_perscene.csv')
    with open(scene_csv, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(scene_rows[0].keys()))
        wtr.writeheader(); wtr.writerows(scene_rows)

    # aggregate report (raw numbers)
    n = len(scene_rows)
    tc_cv = [r['cam_dist_mean_cv'] for r in scene_rows]
    cl_cv = [r['context_longer_cv'] for r in scene_rows if r['context_longer_cv'] == r['context_longer_cv']]
    span_tc = [r['span_over_targetmean'] for r in scene_rows if r['span_over_targetmean'] == r['span_over_targetmean']]
    span_cl = [r['span_over_ctxmean'] for r in scene_rows if r['span_over_ctxmean'] == r['span_over_ctxmean']]
    print(f"\n=== scenes={n}  segments={len(rows)}  (>= {args.min_segs} segs, seg>= {args.num_frames}f) ===")
    print("scene_span is CONSTANT within a scene by construction (CV=0).")
    print(f"\nWITHIN-SCENE CV of per-segment scale (higher = scale 'keeps changing'):")
    print(f"  cam_dist_mean  CV  percentiles {{5,25,50,75,95}} = "
          + str({k: round(v, 3) for k, v in pct(tc_cv).items()}))
    print(f"  context_longer CV  percentiles {{5,25,50,75,95}} = "
          + str({k: round(v, 3) for k, v in pct(cl_cv).items()}))
    print(f"\nMAGNITUDE ratio scene_span / mean(per-seg scale):")
    print(f"  scene_span / cam_dist_mean_mean     {{5,25,50,75,95}} = "
          + str({k: round(v, 3) for k, v in pct(span_tc).items()}))
    print(f"  scene_span / context_longer_mean {{5,25,50,75,95}} = "
          + str({k: round(v, 3) for k, v in pct(span_cl).items()}))
    print(f"\nper-scene CSV: {scene_csv}\nper-seg  CSV: {args.out}")


if __name__ == '__main__':
    main()
