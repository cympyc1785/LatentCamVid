"""Scan DL3DV scenes for COLMAP registration failures that park many frames on ONE camera center.

Scene 4K/50eb3c0d... rendered at PSNR ~14.4 under every normalization divisor. The cause was not
the divisor: >half of its 368 frames share the exact center [1.0211, -1.2376, 3.0891] while the
images clearly move, so any context drawn from that run has zero baseline (and `geo_lagernvs`'s
divisor, which is 1.35*max||ctx center - c_s||, collapses to ~1e-5).

Per scene this reports the largest cluster of near-identical camera centers:
  dup_frac   biggest cluster size / N frames        (1.0 = every frame at one point)
  dup_run    longest CONSECUTIVE run inside it      (a registration dropout is contiguous)
  extent     max||c_i - c_j|| over the scene        (the tolerance is relative to this)
A scene is flagged when dup_frac >= FRAC (default 0.10) -- i.e. a tenth of the trajectory is
frozen, which is far outside anything a real capture produces.

env: TOL (1e-4, relative to extent), FRAC (0.10), WORKERS (32), META (meta_worldtraj.csv),
     OUT_NAME (scan_duplicate_poses)
out -> results/compare/<OUT_NAME>/{per_scene.csv, flagged.csv, blacklist_rows.csv, stats.json}
`blacklist_rows.csv` is ready to append to <dl3dv_root>/blacklist.csv.
"""
import os, sys, csv, json
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

TOL = float(os.environ.get("TOL", "1e-4"))
FRAC = float(os.environ.get("FRAC", "0.10"))
WORKERS = int(os.environ.get("WORKERS", "32"))
EXP = os.environ.get("CACHE_EXP", "geo_worldtraj")
OUT = os.path.join(HERE, "results", "compare",
                   os.environ.get("OUT_NAME", "scan_duplicate_poses"))
os.makedirs(OUT, exist_ok=True)

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
from dataset_dl3dv import CamDataset

ds = CamDataset(cfg, "train")
SCENES = sorted({s[0] for s in ds.samples})
print(f"{len(SCENES)} scenes (train, {cfg.meta_csv}) tol={TOL} frac={FRAC}", flush=True)


def scan(scene_idx):
    try:
        c = torch.linalg.inv(ds.extrinsics_list[scene_idx].float())[:, :3, 3].numpy()
    except Exception as ex:
        return dict(scene_idx=scene_idx, chunk="", n=0, extent=0.0, dup_frac=0.0,
                    dup_run=0, err=str(ex)[:60])
    n = len(c)
    extent = float(np.linalg.norm(c[:, None] - c[None], axis=-1).max()) if n > 1 else 0.0
    eps = max(TOL * extent, 1e-9)
    # bucket centers on an eps grid; the biggest bucket is the "parked" cluster
    keys = np.round(c / eps).astype(np.int64)
    uniq, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    big = int(cnt.argmax())
    mask = inv == big
    # longest consecutive run of frames inside that cluster
    run = best = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    chunk = os.path.relpath(ds.scene_dir_list[scene_idx], ds.root)
    return dict(scene_idx=scene_idx, chunk=chunk, n=n, extent=extent,
                dup_frac=float(cnt[big]) / n, dup_run=int(best),
                center=[float(x) for x in c[mask][0]], err="")


if WORKERS > 1:
    from multiprocessing import Pool
    with Pool(WORKERS) as p:
        rows = p.map(scan, SCENES, chunksize=16)
else:
    rows = [scan(i) for i in SCENES]

rows = [r for r in rows if r["n"]]
rows.sort(key=lambda r: -r["dup_frac"])
flagged = [r for r in rows if r["dup_frac"] >= FRAC]

with open(os.path.join(OUT, "per_scene.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["chunk", "n", "extent", "dup_frac", "dup_run", "scene_idx"],
                       extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT, "flagged.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["chunk", "n", "extent", "dup_frac", "dup_run", "scene_idx"],
                       extrasaction="ignore")
    w.writeheader(); w.writerows(flagged)
with open(os.path.join(OUT, "blacklist_rows.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scene", "reason", "step", "loss", "detail"])
    for r in flagged:
        w.writerow([r["chunk"], "duplicate_camera_centers", "", "",
                    f"dup_frac={r['dup_frac']:.3f} dup_run={r['dup_run']} n={r['n']} "
                    f"extent={r['extent']:.2f}"])

fr = np.array([r["dup_frac"] for r in rows])
stats = dict(n_scenes=len(rows), tol=TOL, frac_thresh=FRAC, n_flagged=len(flagged),
             dup_frac_percentiles={f"p{q}": float(np.percentile(fr, q))
                                   for q in (50, 90, 99, 99.9, 100)})
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)

print(f"\ndup_frac percentiles: " + "  ".join(f"{k}={v:.4f}"
                                              for k, v in stats["dup_frac_percentiles"].items()))
print(f"flagged (dup_frac >= {FRAC}): {len(flagged)} / {len(rows)}\n")
print(f"{'chunk':70s} {'n':>5s} {'extent':>8s} {'dup_frac':>9s} {'dup_run':>8s}")
for r in rows[:25]:
    print(f"{r['chunk']:70s} {r['n']:5d} {r['extent']:8.2f} {r['dup_frac']:9.4f} {r['dup_run']:8d}")
print("\nsaved", OUT)
print("DONE")
