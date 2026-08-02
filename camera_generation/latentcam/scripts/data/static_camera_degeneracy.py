"""Where the camera-LENGTH divisors (arm A ctx_longer_135max / arm B geo_lagernvs) break when the
camera barely moves.

scripts/data/norm_divisor_compare.py answers "which divisor canonicalizes best", but it
`return None`s on any segment with maxd < 1e-6 and reports only scale-FREE ratios (m = maxd/D,
r = D_lagernvs/D). Both choices hide the static-camera failure by construction: the divisor is a
camera displacement, so dividing a camera displacement by it always lands near 1 no matter how
small the motion physically was. This script keeps every chunk and measures the quantities that
are NOT scale-free.

The three failure channels, and the number that measures each:

  (1) divisor collapse / clamp.  D_raw = 1.35*max||center - ref||; `.clamp(min=1e-5)` in
      dataset_dl3dv is the only thing between a frozen camera and a divide-by-zero.
        -> D_raw, frac(D_raw < 1e-5), frac(D_raw < 1e-3)

  (2) fabricated motion (diffusion side).  The model regresses (c_t - c_s)/D. Under a
      camera-length D that ratio is ~0.74 by construction EVEN IF the physical motion is pure SLAM
      jitter, so a frozen camera is presented to the diffusion model as a full-amplitude
      trajectory. Under avg_scale (= point-cloud scene depth) the same chunk stays near zero,
      which is the honest encoding of "no motion".
        -> reach_D = maxd/D  vs  reach_A = maxd/avg_scale, and straightness = maxd/pathlen
           (straightness -> 0 means the "trajectory" being amplified is wobble, not travel)

  (3) exploded scene depth (LagerNVS side).  LagerNVS is not scale-invariant; it was trained with
      the scene normalized by its own 1.35*max||ctx - ctx0||, i.e. content sitting at O(1). Under a
      camera-length D the normalized scene depth is avg_scale/D, which diverges exactly when the
      camera stops. Physically the same chunk also has no triangulation baseline at all.
        -> depth_over_D = avg_scale/D  (LagerNVS input radius),
           parallax = maxctx0/avg_scale (baseline / scene depth; normalization-independent),
           cam_scale_tok = maxctx0/D (the scalar build_cam_token actually feeds LagerNVS;
           1/1.35 = 0.7407 is its native value)

DL3DV is the control (the user's prior: almost no static cameras); DynamicVerse is the suspect and
is additionally broken out per subset, since DAVIS/MOSE/VOST/uvo/youtube_vis are tripod-ish while
dynamic_replica/MVS-Synth/spring are synthetic fly-throughs.

env: DATASET (dynamicverse | dl3dv, default dynamicverse), CACHE_EXP (geo_worldtraj_ctxlonger135),
     N (0 = every chunk), SPLIT (train, dl3dv only), WORKERS (1), DV_CHUNKS (3), SEED (0),
     OUT_NAME (default static_degeneracy_<DATASET>)
out -> results/compare/<OUT_NAME>/{per_chunk.csv,stats.json,summary.md,_static.png}
"""
import os, sys, json, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

DATASET = os.environ.get("DATASET", "dynamicverse")
EXP = os.environ.get("CACHE_EXP", "geo_worldtraj_ctxlonger135")
N = int(os.environ.get("N", "0"))
SPLIT = os.environ.get("SPLIT", "train")
WORKERS = int(os.environ.get("WORKERS", "1"))
DV_ROOT = os.environ.get("DV_ROOT", "/data1/cympyc1785/data/dynamicverse")
DV_CHUNKS = int(os.environ.get("DV_CHUNKS", "3"))
SEED = int(os.environ.get("SEED", "0"))
OUT = os.path.join(HERE, "results", "compare",
                   os.environ.get("OUT_NAME", f"static_degeneracy_{DATASET}"))
os.makedirs(OUT, exist_ok=True)

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
from dataset_dl3dv import CamDataset

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if DATASET == "dynamicverse":
    from dynamicverse_shim import load_dynamicverse
    ds = load_dynamicverse(cfg, DV_ROOT, DV_CHUNKS)
else:
    ds = CamDataset(cfg, SPLIT)
T = cfg.num_frames
CLAMP = 1e-5                      # dataset_dl3dv's .clamp(min=1e-5) on every divisor


def even(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def longer_side(N_all, s, e):
    return list(range(0, s)) if s >= (N_all - e) else list(range(e, N_all))


def windows(side, T):
    ch = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]
    if not ch and len(side) >= 2:
        ch = [side]
    return ch


def rot_angle_deg(Ra, Rb):
    """geodesic angle between two rotations, degrees."""
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def row_for(idx):
    """Every chunk is kept -- including maxd == 0, which is the whole point here."""
    scene_idx, s, e, _caption, data_name = ds.samples[idx]
    w2c = ds.extrinsics_list[scene_idx].float().numpy()
    c2w = np.linalg.inv(w2c)
    centers = c2w[:, :3, 3]
    N_all = centers.shape[0]

    tgt = list(range(s, e))
    if len(tgt) > T:
        tgt = [tgt[i] for i in even(len(tgt), T)]
    d_tgt = np.linalg.norm(centers[tgt] - centers[s], axis=1)
    maxd = float(d_tgt.max())
    pathlen = float(np.linalg.norm(np.diff(centers[tgt], axis=0), axis=1).sum())
    rot_max = max(rot_angle_deg(c2w[t, :3, :3], c2w[s, :3, :3]) for t in tgt)

    gi = ds._sample_geo_frustum_cover(scene_idx, s, e)
    gi = list(gi)
    d_ctx_s = np.linalg.norm(centers[gi] - centers[s], axis=1)
    d_ctx_0 = np.linalg.norm(centers[gi] - centers[gi[0]], axis=1)
    cg = centers[gi]
    pair = np.linalg.norm(cg[:, None, :] - cg[None, :, :], axis=-1)
    baseline = float(pair.max())                       # widest geo-view pair = triangulation base

    side = longer_side(N_all, s, e)
    chs = windows(side, T)
    win = [1.35 * float(np.linalg.norm(centers[c] - centers[c[0]], axis=1).max()) for c in chs]

    D = {}
    D["geo_lagernvs"] = 1.35 * float(d_ctx_s.max())
    if win:
        D["ctx_longer_135max"] = float(np.mean(win))
    # dataset_dl3dv.py:876 -- when _context_window_scale returns None (context side < 2 frames)
    # arm A silently falls back to THIS, the target-derived (leaky, inference-unavailable) divisor.
    D["first_farthest_135_fallback"] = 1.35 * maxd
    a = ds._avg_scale(scene_idx, data_name.split("_")[-1])
    avg_scale = float(a) if a is not None else float("nan")

    return dict(seg=data_name, scene=int(scene_idx), s=int(s), e=int(e), n_frames=int(N_all),
                subset=data_name.split("/")[0] if "/" in data_name else "dl3dv",
                maxd=maxd, meand=float(d_tgt.mean()), pathlen=pathlen, rot_max_deg=rot_max,
                maxctx_s=float(d_ctx_s.max()), maxctx_0=float(d_ctx_0.max()),
                baseline=baseline, n_geo=len(gi), n_geo_uniq=len(set(gi)),
                win_min=float(min(win)) if win else float("nan"),
                win_n=len(win), avg_scale=avg_scale, **{f"D_{k}": v for k, v in D.items()})


idxs = list(range(len(ds.samples)))
if N:
    random.shuffle(idxs); idxs = sorted(idxs[:N])
if WORKERS > 1:
    from multiprocessing import Pool
    with Pool(WORKERS) as p:
        rows = p.map(row_for, idxs, chunksize=32)
else:
    rows = []
    for i, idx in enumerate(idxs):
        rows.append(row_for(idx))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(idxs)}", flush=True)
print(f"{len(rows)} chunks from {len({r['scene'] for r in rows})} scenes "
      f"(dataset={DATASET} exp={EXP})")

import pandas as pd
df = pd.DataFrame(rows)
# EFFECTIVE divisor = what dataset_dl3dv actually divides by, fallback included (line 876).
# Arm B (_geo_lagernvs_scale) and avg_scale-on-DynamicVerse never fall back here.
df["fallback_A"] = df.D_ctx_longer_135max.isna()
df["Deff_A"] = df.D_ctx_longer_135max.fillna(df.D_first_farthest_135_fallback)
df["Deff_B"] = df.D_geo_lagernvs
ARMS = [("A ctx_longer_135max", "Deff_A"),
        ("B geo_lagernvs", "Deff_B"),
        ("avg_scale (baseline)", "avg_scale")]
for _nm, col in ARMS:
    d = df[col]
    dc = d.clip(lower=CLAMP)                                   # what the dataset actually divides by
    df[f"reach__{col}"] = df.maxd / dc                         # (2) what diffusion regresses
    df[f"depth__{col}"] = df.avg_scale / dc                    # (3) LagerNVS input radius
    df[f"tok__{col}"] = df.maxctx_0 / dc                       # (3) build_cam_token camera_scale
df["straight"] = df.maxd / df.pathlen.replace(0, np.nan)
df["parallax"] = df.maxctx_0 / df.avg_scale                    # physical, divisor-independent
df["rot_per_trans"] = df.rot_max_deg / (df.maxd / df.avg_scale).replace(0, np.nan)
df.to_csv(os.path.join(OUT, "per_chunk.csv"), index=False)

PCT = [0, 0.1, 1, 5, 25, 50]


def q(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def hi(x, p):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


def block(sub):
    o = dict(n=int(len(sub)))
    for nm, col in ARMS:
        d = sub[col].to_numpy(float)
        o[nm] = dict(
            D_min=float(np.nanmin(d)), D_p1=q(d, 1), D_median=q(d, 50),
            frac_D_lt_clamp=float(np.mean(d < CLAMP)),
            frac_D_lt_1e3=float(np.mean(d < 1e-3)),
            reach_median=q(sub[f"reach__{col}"], 50),
            reach_p99=hi(sub[f"reach__{col}"], 99),
            reach_max=float(np.nanmax(sub[f"reach__{col}"])),
            depth_median=q(sub[f"depth__{col}"], 50),
            depth_p99=hi(sub[f"depth__{col}"], 99),
            depth_max=float(np.nanmax(sub[f"depth__{col}"])),
            tok_median=q(sub[f"tok__{col}"], 50),
            frac_tok_out_2x=float(np.mean((sub[f"tok__{col}"] < (1 / 1.35) / 2)
                                          | (sub[f"tok__{col}"] > (1 / 1.35) * 2))),
        )
    o["physical"] = dict(
        n_frames_median=q(sub.n_frames, 50),
        frac_fallback_A=float(sub.fallback_A.mean()),
        parallax_min=float(np.nanmin(sub.parallax)), parallax_p1=q(sub.parallax, 1),
        parallax_median=q(sub.parallax, 50),
        frac_parallax_lt_1e2=float(np.mean(sub.parallax < 1e-2)),
        frac_parallax_lt_1e3=float(np.mean(sub.parallax < 1e-3)),
        straight_median=q(sub.straight, 50), straight_p1=q(sub.straight, 1),
        frac_straight_lt_0p2=float(np.mean(sub.straight < 0.2)),
        frac_geo_dup=float(np.mean(sub.n_geo_uniq < sub.n_geo)),
        maxd_over_avgscale_median=q(sub.maxd / sub.avg_scale, 50),
        frac_maxd_over_avgscale_lt_1e2=float(np.mean((sub.maxd / sub.avg_scale) < 1e-2)),
    )
    return o


stats = dict(dataset=DATASET, exp=EXP, n_chunks=len(df), n_scenes=int(df.scene.nunique()),
             num_frames=T, clamp=CLAMP, overall=block(df),
             per_subset={s: block(g) for s, g in df.groupby("subset")} if df.subset.nunique() > 1
             else {})
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=2)

# ----------------------------------------------------------------------------------- report
L = []
hdr = (f"{'group':16s} {'n':>5s} | {'D min':>9s} {'D p1':>9s} {'D<1e-5':>7s} {'D<1e-3':>7s} "
       f"| {'reach med':>9s} {'reach p99':>9s} | {'depth med':>9s} {'depth p99':>9s} "
       f"{'depth max':>10s} | {'tok!=.74':>8s}")
for nm, col in ARMS:
    L += [f"### {nm}", hdr, "-" * len(hdr)]
    groups = [("ALL", df)] + (sorted(df.groupby("subset"), key=lambda kv: kv[0])
                              if df.subset.nunique() > 1 else [])
    for g, sub in groups:
        v = block(sub)[nm]
        L.append(f"{g:16s} {len(sub):5d} | {v['D_min']:9.3g} {v['D_p1']:9.3g} "
                 f"{v['frac_D_lt_clamp']:7.3f} {v['frac_D_lt_1e3']:7.3f} | "
                 f"{v['reach_median']:9.3f} {v['reach_p99']:9.3f} | "
                 f"{v['depth_median']:9.2f} {v['depth_p99']:9.1f} {v['depth_max']:10.1f} | "
                 f"{v['frac_tok_out_2x']:8.3f}")
    L.append("")
ph = (f"{'group':16s} {'n':>5s} {'Nfr med':>7s} {'fallbkA':>7s} | {'plx min':>9s} {'plx p1':>9s} "
      f"{'plx med':>9s} {'plx<1e-2':>8s} "
      f"{'plx<1e-3':>8s} | {'strt med':>8s} {'strt<0.2':>8s} | {'geo dup':>7s} "
      f"| {'maxd/A med':>10s} {'<1e-2':>7s}")
L += ["### physical (divisor-independent)", ph, "-" * len(ph)]
for g, sub in [("ALL", df)] + (sorted(df.groupby("subset"), key=lambda kv: kv[0])
                               if df.subset.nunique() > 1 else []):
    v = block(sub)["physical"]
    L.append(f"{g:16s} {len(sub):5d} {v['n_frames_median']:7.0f} {v['frac_fallback_A']:7.3f} | "
             f"{v['parallax_min']:9.3g} {v['parallax_p1']:9.3g} "
             f"{v['parallax_median']:9.3g} {v['frac_parallax_lt_1e2']:8.3f} "
             f"{v['frac_parallax_lt_1e3']:8.3f} | {v['straight_median']:8.3f} "
             f"{v['frac_straight_lt_0p2']:8.3f} | {v['frac_geo_dup']:7.3f} | "
             f"{v['maxd_over_avgscale_median']:10.4f} "
             f"{v['frac_maxd_over_avgscale_lt_1e2']:7.3f}")
txt = "\n".join(L)
print("\n" + txt)
open(os.path.join(OUT, "summary.md"), "w").write(
    f"# static-camera degeneracy ({DATASET}, {len(df)} chunks / {df.scene.nunique()} scenes, "
    f"exp={EXP})\n\nplx = maxctx0/avg_scale (triangulation baseline / scene depth), "
    f"strt = maxd/pathlen, reach = maxd/D (diffusion target), depth = avg_scale/D "
    f"(LagerNVS input radius), tok = maxctx0/D (native 0.7407)\n\n```\n{txt}\n```\n")

# ------------------------------------------------------------------------------------ plots
fig, axs = plt.subplots(1, 4, figsize=(24, 5.2))
for nm, col in ARMS:
    axs[0].hist(np.log10(df[col].replace(0, np.nan).dropna()), bins=70, histtype="step", lw=1.6,
                label=nm)
    axs[1].hist(np.log10(df[f"reach__{col}"].replace(0, np.nan).dropna()), bins=70,
                histtype="step", lw=1.6, label=nm)
    axs[2].hist(np.log10(df[f"depth__{col}"].replace(0, np.nan).dropna()), bins=70,
                histtype="step", lw=1.6, label=nm)
axs[0].axvline(np.log10(CLAMP), c="r", lw=1, ls=":"); axs[0].set_xlabel("log10 D_raw")
axs[0].set_title("(1) divisor collapse\nred = clamp(1e-5)")
axs[1].set_xlabel("log10 maxd/D"); axs[1].set_title("(2) reach the diffusion model regresses")
axs[2].set_xlabel("log10 avg_scale/D")
axs[2].set_title("(3) normalized scene depth\n= LagerNVS input radius")
sc = axs[3].scatter(np.log10(df.parallax.replace(0, np.nan)), df.straight,
                    c=np.log10(df["reach__Deff_B"].replace(0, np.nan)), s=6, cmap="viridis")
axs[3].axvline(-2, c="r", lw=1, ls=":")
axs[3].set_xlabel("log10 parallax (maxctx0/avg_scale)"); axs[3].set_ylabel("straightness maxd/pathlen")
axs[3].set_title("physical degeneracy\ncolor = log10 reach under arm B")
plt.colorbar(sc, ax=axs[3])
for a in axs[:3]:
    a.legend(fontsize=8)
fig.suptitle(f"static-camera degeneracy of camera-length divisors — {DATASET}, {len(df)} chunks",
             fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "_static.png"), dpi=120, bbox_inches="tight")

# worst offenders, for the LagerNVS render step
worst = df.sort_values("parallax").head(20)[
    ["seg", "parallax", "straight", "maxd", "avg_scale", "Deff_A", "Deff_B", "fallback_A",
     "reach__Deff_B", "depth__Deff_B", "rot_max_deg"]]
worst.to_csv(os.path.join(OUT, "worst_parallax.csv"), index=False)
print("\n### 20 lowest-parallax chunks\n" + worst.to_string(index=False))
print("\nsaved", OUT)
