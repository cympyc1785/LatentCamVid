"""Which normalization divisor should latentcam train the target segment with?

Compares every candidate divisor D that has come up so far on the two things that actually matter:

  (A) LagerNVS compatibility.  LagerNVS is NOT scale-invariant -- it was trained with the context
      normalized by its own `1.35*max||ctx center - ctx0||`, i.e. always at
      camera_scale = max||ctx - ctx0|| / D = 1/1.35 = 0.7407.  Feeding it a scene normalized by
      some other D is a pure global-scale domain shift of factor
          r = D_lagernvs / D          (r = 1 <=> no shift)
      This script measures r's distribution; the companion render sweep
      (main/dump_avgscale_render.py + tools/lagernvs/render_avgscale.py) measures the PSNR that
      the shift actually costs.

  (B) Canonicalization for the camera-generation model.  The diffusion model regresses
      cam_param translations = (target camera - camera s) / D.  What it has to learn is easiest
      when the normalized reach
          m = max_t ||c_t - c_s|| / D
      is the SAME number for every segment.  So the figure of merit is the spread of log10(m),
      not its level (any constant level is absorbed by the VAE).  sd(log10 m) is the headline
      number; frac(m>1) and frac(m<0.1) say whether the tail leaves the useful range.
      sd(log10 m)^2 = sd(log10 maxd)^2 + sd(log10 D)^2 - 2*corr*sd*sd, so a divisor wins by
      CORRELATING with the target's motion, not by being smooth.

Leaky divisors (computed from the target segment, unavailable at inference) are reported as an
upper bound, clearly marked.

env: N (segments, default 1200), CACHE_EXP (default geo_worldtraj), SPLIT (train), SEED (0)
out -> results/compare/norm_divisor_compare/{stats.json,summary.md,_divisors.png,per_segment.csv}
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

N = int(os.environ.get("N", "1200"))
EXP = os.environ.get("CACHE_EXP", "geo_worldtraj")
SPLIT = os.environ.get("SPLIT", "train")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.path.join(HERE, "results", "compare", "norm_divisor_compare")
os.makedirs(OUT, exist_ok=True)

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
from dataset_dl3dv import CamDataset

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
ds = CamDataset(cfg, SPLIT)
T = cfg.num_frames


def even(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def longer_side(N_all, s, e):
    return list(range(0, s)) if s >= (N_all - e) else list(range(e, N_all))


def windows(side, T):
    ch = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]
    if not ch and len(side) >= 2:
        ch = [side]
    return ch


# ---------------------------------------------------------------- sample segments (1 per scene)
order = list(range(len(ds.samples)))
random.shuffle(order)
rows, seen = [], set()
for idx in order:
    if len(rows) >= N:
        break
    scene_idx, s, e, caption, data_name = ds.samples[idx]
    if scene_idx in seen:
        continue
    c2w = torch.linalg.inv(ds.extrinsics_list[scene_idx].float()).numpy()
    centers = c2w[:, :3, 3]
    N_all = centers.shape[0]
    tgt = list(range(s, e))
    if len(tgt) > T:
        tgt = [tgt[i] for i in even(len(tgt), T)]
    d_tgt = np.linalg.norm(centers[tgt] - centers[s], axis=1)          # dist from frame s
    maxd = float(d_tgt.max())
    if maxd < 1e-6:                                                    # static segment -> skip
        continue
    gi = ds._sample_geo_frustum_cover(scene_idx, s, e)
    d_ctx_s = np.linalg.norm(centers[gi] - centers[s], axis=1)         # ctx dist from frame s
    d_ctx_0 = np.linalg.norm(centers[gi] - centers[gi[0]], axis=1)     # ctx dist from ctx view0
    side = longer_side(N_all, s, e)
    chs = windows(side, T)

    D = {}
    D["geo_lagernvs"] = 1.35 * float(d_ctx_s.max())                    # implemented scale_mode
    if chs:
        D["ctx_longer_135max"] = float(np.mean(
            [1.35 * float(np.linalg.norm(centers[c] - centers[c[0]], axis=1).max()) for c in chs]))
        D["context_longer"] = float(np.mean(
            [float(np.linalg.norm(centers[c] - centers[c[0]], axis=1).mean()) for c in chs]))
    if len(side) >= 2:
        D["ctx_side_135max"] = 1.35 * float(
            np.linalg.norm(centers[side] - centers[side[0]], axis=1).max())
    # leaky (target-derived) -- upper bound only
    D["first_farthest_135*"] = 1.35 * maxd
    D["cam_dist_mean*"] = float(d_tgt.mean())
    a = ds._avg_scale(scene_idx, data_name.split("_")[-1])             # excluded, kept as baseline
    if a is not None:
        D["avg_scale(excl)"] = float(a)

    rows.append(dict(seg=data_name, s=s, e=e, n_frames=N_all,
                     maxd=maxd, meand=float(d_tgt.mean()),
                     pathlen=float(np.linalg.norm(np.diff(centers[tgt], axis=0), axis=1).sum()),
                     maxctx=float(d_ctx_0.max()), D=D))
    seen.add(scene_idx)
    if len(rows) % 100 == 0:
        print(f"  {len(rows)}/{N}", flush=True)

print(f"{len(rows)} segments from {len(seen)} distinct scenes (exp={EXP} split={SPLIT})")

# --------------------------------------------------------------------------------- aggregate
NAMES = ["geo_lagernvs", "ctx_longer_135max", "ctx_side_135max", "context_longer",
         "first_farthest_135*", "cam_dist_mean*", "avg_scale(excl)"]
LEAK = {"first_farthest_135*", "cam_dist_mean*"}
IDEAL_CS = 1 / 1.35

maxd = np.array([r["maxd"] for r in rows])
maxctx = np.array([r["maxctx"] for r in rows])
stats = {}
for nm in NAMES:
    ok = np.array([nm in r["D"] and r["D"][nm] > 1e-9 for r in rows])
    if ok.sum() < 10:
        continue
    d = np.array([r["D"][nm] if nm in r["D"] else np.nan for r in rows])
    m = maxd[ok] / d[ok]                       # (B) normalized reach the model must regress
    r_ln = (1.35 * maxctx[ok]) / d[ok]         # (A) LagerNVS domain-shift factor; 1 = none
    cs = maxctx[ok] / d[ok]                    # what render_avgscale reports as camera_scale
    lm, ld = np.log10(m), np.log10(d[ok])
    stats[nm] = dict(
        n=int(ok.sum()), leak=nm in LEAK,
        D_median=float(np.median(d[ok])), D_log10_sd=float(ld.std()),
        # (A) LagerNVS compatibility
        r_median=float(np.median(r_ln)), r_log10_sd=float(np.log10(r_ln).std()),
        r_within_1p5=float(np.mean((r_ln > 1 / 1.5) & (r_ln < 1.5))),
        camera_scale_median=float(np.median(cs)),
        # (B) canonicalization
        m_median=float(np.median(m)), m_log10_sd=float(lm.std()),
        m_p05=float(np.percentile(m, 5)), m_p95=float(np.percentile(m, 95)),
        m_p95_over_p05=float(np.percentile(m, 95) / max(np.percentile(m, 5), 1e-9)),
        frac_m_gt1=float(np.mean(m > 1.0)), frac_m_lt0p1=float(np.mean(m < 0.1)),
        corr_logD_logmaxd=float(np.corrcoef(ld, np.log10(maxd[ok]))[0, 1]),
    )

json.dump(dict(exp=EXP, split=SPLIT, n_segments=len(rows), num_frames=T, stats=stats),
          open(os.path.join(OUT, "stats.json"), "w"), indent=2)

with open(os.path.join(OUT, "per_segment.csv"), "w") as f:
    f.write("seg,s,e,n_frames,maxd,meand,pathlen,maxctx," + ",".join(NAMES) + "\n")
    for r in rows:
        f.write(f"{r['seg']},{r['s']},{r['e']},{r['n_frames']},{r['maxd']:.6f},{r['meand']:.6f},"
                f"{r['pathlen']:.6f},{r['maxctx']:.6f},"
                + ",".join(f"{r['D'][n]:.6f}" if n in r["D"] else "" for n in NAMES) + "\n")

# --------------------------------------------------------------------------------- report
hdr = (f"{'divisor':22s} {'n':>5s} {'leak':>4s} | {'r med':>6s} {'r sd':>6s} {'r<1.5':>6s} "
       f"| {'m med':>6s} {'m sd':>6s} {'p95/p05':>8s} {'m>1':>6s} {'m<.1':>6s} {'corr':>6s}")
lines = [hdr, "-" * len(hdr)]
for nm, v in stats.items():
    lines.append(f"{nm:22s} {v['n']:5d} {'Y' if v['leak'] else '.':>4s} | "
                 f"{v['r_median']:6.3f} {v['r_log10_sd']:6.3f} {v['r_within_1p5']:6.3f} | "
                 f"{v['m_median']:6.3f} {v['m_log10_sd']:6.3f} {v['m_p95_over_p05']:8.2f} "
                 f"{v['frac_m_gt1']:6.3f} {v['frac_m_lt0p1']:6.3f} {v['corr_logD_logmaxd']:6.3f}")
txt = "\n".join(lines)
print("\n" + txt)
print(f"\nreference: sd(log10 maxd) over the same segments = {np.log10(maxd).std():.4f}")
print("r = D_lagernvs/D  (1 = LagerNVS sees its native scale) | m = maxd/D (normalized reach)")
open(os.path.join(OUT, "summary.md"), "w").write(
    f"# norm divisor comparison (exp={EXP}, split={SPLIT}, {len(rows)} segments / distinct scenes)\n\n"
    f"```\n{txt}\n```\n\nsd(log10 maxd) baseline = {np.log10(maxd).std():.4f}\n")

# --------------------------------------------------------------------------------- plots
plot = [n for n in stats if n != "avg_scale(excl)"] + (
    ["avg_scale(excl)"] if "avg_scale(excl)" in stats else [])
fig, axs = plt.subplots(1, 3, figsize=(19, 5.2))
bins = np.linspace(-2.2, 1.2, 70)
for nm in plot:
    ok = np.array([nm in r["D"] and r["D"][nm] > 1e-9 for r in rows])
    d = np.array([r["D"].get(nm, np.nan) for r in rows])
    m = maxd[ok] / d[ok]
    r_ln = (1.35 * maxctx[ok]) / d[ok]
    ls = "--" if nm in LEAK or nm.endswith("(excl)") else "-"
    axs[0].hist(np.log10(m), bins=bins, histtype="step", ls=ls, lw=1.6,
                label=f"{nm} (sd {stats[nm]['m_log10_sd']:.3f})")
    axs[1].hist(np.log10(r_ln), bins=np.linspace(-1.5, 1.5, 70), histtype="step", ls=ls, lw=1.6,
                label=f"{nm} (|log r| med {abs(np.median(np.log10(r_ln))):.2f})")
    axs[2].scatter(stats[nm]["r_log10_sd"], stats[nm]["m_log10_sd"], s=90,
                   marker="x" if (nm in LEAK or nm.endswith("(excl)")) else "o")
    axs[2].annotate(nm, (stats[nm]["r_log10_sd"], stats[nm]["m_log10_sd"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
axs[0].set_title("(B) canonicalization: log10(normalized reach m = maxd/D)\nnarrower = better")
axs[0].set_xlabel("log10 m"); axs[0].legend(fontsize=7)
axs[1].axvline(0, c="k", lw=1)
axs[1].set_title("(A) LagerNVS domain shift: log10(r = D_lagernvs/D)\n0 = LagerNVS's native scale")
axs[1].set_xlabel("log10 r"); axs[1].legend(fontsize=7)
axs[2].set_xlabel("(A) sd log10 r  [LagerNVS shift spread]")
axs[2].set_ylabel("(B) sd log10 m  [canonicalization spread]")
axs[2].set_title("lower-left = better on both\n(x = leaky / excluded)")
fig.suptitle(f"normalization divisor comparison — {len(rows)} segments, exp={EXP}", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "_divisors.png"), dpi=120, bbox_inches="tight")
print("saved", OUT)
