"""Camera-length distribution the model sees under the two training normalizations, across a large
sample of REAL training segments (reconstructed standalone from transforms.json + prompts.json;
no dataset load). For each segment: w2c = inv(c2w_gl @ GL2CV), relative to E0, centers = c2w
translation; then normalize by
  point (target_cam)        : avg_scale = mean(||center_i||)
  dist  (first_farthest_135): avg_scale = 1.35 * max(||center_i||)
"Camera length" = normalized per-frame distance from the first camera (||center_i|| / avg_scale)
== the translation norm the diffusion model regresses. Aggregates mean/std/min/max/percentiles +
histogram, and per-segment span/pathlen.
out -> results/compare/normalization_camera_length/
"""
import os, json, glob, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT10K = "/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K"
OUT = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results/compare/normalization_camera_length"
os.makedirs(OUT, exist_ok=True)
NUM_FRAMES = 49
GL2CV = np.diag([1., -1., -1., 1.])
SCENES_PER_BATCH = 120          # sample per batch dir for a representative, fast run
random.seed(0)


def even_idx(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def seg_stats(w2c):                          # (T,4,4) -> per-frame center norms + the two avg_scales
    e0i = np.linalg.inv(w2c[0]); nr = w2c @ e0i[None]
    centers = np.linalg.inv(nr)[:, :3, 3]
    cn = np.linalg.norm(centers, axis=1)
    s_pt = max(cn.mean(), 1e-5); s_di = max(1.35 * cn.max(), 1e-5)
    return cn, centers, s_pt, s_di


pt_len, di_len = [], []                      # per-frame normalized camera length
pt_span, di_span, pt_path, di_path = [], [], [], []   # per-segment
nseg = 0
batches = sorted(d for d in os.listdir(ROOT10K) if os.path.isdir(os.path.join(ROOT10K, d)))
for bd in batches:
    scenes = [s for s in glob.glob(os.path.join(ROOT10K, bd, "*")) if os.path.isdir(s)]
    random.shuffle(scenes)
    used = 0
    for sc in scenes:
        if used >= SCENES_PER_BATCH:
            break
        tj = os.path.join(sc, "transforms.json"); pj = os.path.join(sc, "prompts.json")
        if not (os.path.exists(tj) and os.path.exists(pj)):
            continue
        try:
            d = json.load(open(tj)); pr = json.load(open(pj))
            c2w = np.array([f["transform_matrix"] for f in d["frames"]], float) @ GL2CV
            w2c_all = np.linalg.inv(c2w); n = w2c_all.shape[0]
        except Exception:
            continue
        got = False
        for seg in pr.values():
            fi = seg.get("frame_idx")
            if not fi:
                continue
            s, e = int(fi[0]), int(fi[1])
            if e > n or (e - s) < NUM_FRAMES:
                continue
            w2c = w2c_all[s:e]
            if w2c.shape[0] > NUM_FRAMES:
                w2c = w2c[even_idx(w2c.shape[0], NUM_FRAMES)]
            cn, centers, s_pt, s_di = seg_stats(w2c)
            if cn.max() < 1e-2:              # skip near-static/degenerate segments (avg_scale ~0)
                continue
            pt_len += (cn / s_pt).tolist(); di_len += (cn / s_di).tolist()
            step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            pt_span.append(cn.max() / s_pt); di_span.append(cn.max() / s_di)
            pt_path.append(step.sum() / s_pt); di_path.append(step.sum() / s_di)
            nseg += 1; got = True
        if got:
            used += 1
print(f"segments={nseg}  frames={len(pt_len)}  (batches={len(batches)}, <= {SCENES_PER_BATCH}/batch)")


def stat(a):
    a = np.asarray(a)
    return dict(mean=a.mean(), std=a.std(), var=a.var(), min=a.min(), max=a.max(),
                p50=np.percentile(a, 50), p95=np.percentile(a, 95), p99=np.percentile(a, 99))


def show(name, pt, di):
    p, d = stat(pt), stat(di)
    print(f"\n== {name} ==")
    hdr = f"{'':7}" + "".join(f"{k:>9}" for k in ["mean", "std", "var", "min", "max", "p50", "p95", "p99"])
    print(hdr)
    print("point  " + "".join(f"{p[k]:>9.3f}" for k in ["mean", "std", "var", "min", "max", "p50", "p95", "p99"]))
    print("dist   " + "".join(f"{d[k]:>9.3f}" for k in ["mean", "std", "var", "min", "max", "p50", "p95", "p99"]))
    return p, d


lp, ld = show("camera length ||center|| (per-frame, normalized)", pt_len, di_len)
show("per-segment MAX length (span)", pt_span, di_span)
show("per-segment path length", pt_path, di_path)

# histograms
fig, axs = plt.subplots(1, 3, figsize=(17, 5))
axs[0].hist(pt_len, bins=80, alpha=0.6, density=True, color="tab:orange", label=f"point (var {np.var(pt_len):.3f})")
axs[0].hist(di_len, bins=80, alpha=0.6, density=True, color="tab:green", label=f"dist (var {np.var(di_len):.3f})")
axs[0].set_title("camera length ||center||/avg_scale (per-frame)"); axs[0].set_xlabel("normalized length"); axs[0].legend()
axs[1].hist(pt_span, bins=np.linspace(0.7, np.percentile(pt_span, 99), 60), alpha=0.6, density=True, color="tab:orange", label=f"point (mean {np.mean(pt_span):.2f})")
axs[1].axvline(0.741, color="tab:green", lw=2, label="dist (const 0.741)")
axs[1].set_title("per-segment MAX length (span)"); axs[1].set_xlabel("max ||center||"); axs[1].legend()
axs[2].hist(pt_path, bins=60, alpha=0.6, density=True, color="tab:orange", label=f"point (mean {np.mean(pt_path):.2f})")
axs[2].hist(di_path, bins=60, alpha=0.6, density=True, color="tab:green", label=f"dist (mean {np.mean(di_path):.2f})")
axs[2].set_title("per-segment path length"); axs[2].set_xlabel("path length"); axs[2].legend()
fig.suptitle(f"camera length under point (target_cam) vs dist (1.35*max)  [{nseg} training segments]", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "camera_length_stats.png"), dpi=120, bbox_inches="tight")
print("\nsaved", os.path.join(OUT, "camera_length_stats.png"))
