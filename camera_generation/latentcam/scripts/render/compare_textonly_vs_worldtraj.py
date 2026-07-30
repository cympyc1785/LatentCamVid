"""text-only vs worldtraj, PER-PAIR on the SAME 160 targets (text-only re-inferred on worldtraj's
targets by infer_textonly_batch.py). Both use cam_dist_mean normalization, so the ONLY difference is
geo conditioning (worldtraj = LagerNVS geo, text-only = none) -> isolates the geo effect.

Per-target top-down (first-cam anchored X/-Z): GT (blue) + worldtraj pred (green) + text-only pred
(red) + worldtraj geo-context cameras (magenta stars) + world pos_rmse/rot for both.
Plus _summary.png (paired pos_rmse) and _scores.csv.
out -> results/compare/text-only_vs_worldtraj/
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results"
WT = os.path.join(ROOT, "20260721_213927_dl3dv_geo_worldtraj", "test")             # GT + worldtraj pred
TO = os.path.join(ROOT, "compare", "text-only_vs_worldtraj", "textonly_preds")      # text-only pred (matched)
OUT = os.path.join(ROOT, "compare", "text-only_vs_worldtraj")
CTX = os.path.join(ROOT, "compare", "normalization_point_vs_dist", "_geo_context.json")
ctx = json.load(open(CTX)) if os.path.exists(CTX) else {}


def load(p):
    return np.array([f["transform_matrix"] for f in json.load(open(p))["frames"]], float)


def rot_err_deg(Ra, Rb):
    R = np.einsum("tij,tkj->tik", Ra, Rb)
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def rmse(pred, ref):
    return float(np.sqrt((np.linalg.norm(pred[:, :3, 3] - ref[:, :3, 3], axis=1) ** 2).mean()))


def anchor(pts, R0inv):
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    return (h @ R0inv.T)[:, :3]


ids = sorted({os.path.basename(f)[:-len("_transforms_ref.json")]
              for f in glob.glob(os.path.join(WT, "*_transforms_ref.json"))})
rows = []
for i, tid in enumerate(ids):
    ref = load(os.path.join(WT, f"{tid}_transforms_ref.json"))
    wp = load(os.path.join(WT, f"{tid}_transforms_pred.json"))
    tp = load(os.path.join(TO, f"{tid}_transforms_pred.json"))
    r_wt, r_to = rmse(wp, ref), rmse(tp, ref)
    rows.append((tid, r_to, r_wt, float(rot_err_deg(tp[:, :3, :3], ref[:, :3, :3]).mean()),
                 float(rot_err_deg(wp[:, :3, :3], ref[:, :3, :3]).mean())))
    R0 = np.linalg.inv(ref[0])
    g, w, t = anchor(ref[:, :3, 3], R0), anchor(wp[:, :3, 3], R0), anchor(tp[:, :3, 3], R0)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    ax.plot(g[:, 0], -g[:, 2], "-o", ms=3, lw=1.5, c="tab:blue", label="GT")
    ax.plot(w[:, 0], -w[:, 2], "-x", ms=3, lw=1.3, c="tab:green", label=f"worldtraj+geo (rmse {r_wt:.2f})")
    ax.plot(t[:, 0], -t[:, 2], "-x", ms=3, lw=1.3, c="tab:red", label=f"text-only (rmse {r_to:.2f})")
    ax.scatter(g[0, 0], -g[0, 2], c="k", s=80, marker="*", zorder=6, label="start")
    if tid in ctx:
        cc = anchor(np.array(ctx[tid]), R0)
        ax.scatter(cc[:, 0], -cc[:, 2], c="magenta", s=90, marker="*", edgecolors="k", linewidths=0.5,
                   zorder=7, label=f"geo context ({len(cc)})")
    ax.set_aspect("equal", "datalim"); ax.legend(fontsize=8); ax.set_xlabel("X"); ax.set_ylabel("-Z")
    ax.set_title(f"{tid}\ntext-only rmse={r_to:.2f}  |  worldtraj+geo rmse={r_wt:.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{tid}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(ids)}")

with open(os.path.join(OUT, "_scores.csv"), "w") as f:
    f.write("target,textonly_pos_rmse,worldtraj_pos_rmse,textonly_rot,worldtraj_rot\n")
    for tid, rt, rw, ot, ow in rows:
        f.write(f"{tid},{rt:.4f},{rw:.4f},{ot:.3f},{ow:.3f}\n")

rt = np.array([r[1] for r in rows]); rw = np.array([r[2] for r in rows])
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
o = np.argsort(rw)
axs[0].plot(rt[o], "-o", ms=2, c="tab:red", label=f"text-only (mean {rt.mean():.3f})")
axs[0].plot(rw[o], "-o", ms=2, c="tab:green", label=f"worldtraj+geo (mean {rw.mean():.3f})")
axs[0].set_title("world pos_rmse per target (sorted by worldtraj)"); axs[0].legend(); axs[0].set_xlabel("target")
lim = max(rt.max(), rw.max()) * 1.05
axs[1].scatter(rt, rw, s=10, alpha=0.6); axs[1].plot([0, lim], [0, lim], "k--", lw=0.8)
axs[1].set_xlim(0, lim); axs[1].set_ylim(0, lim); axs[1].set_aspect("equal")
axs[1].set_xlabel("text-only pos_rmse"); axs[1].set_ylabel("worldtraj+geo pos_rmse")
axs[1].set_title(f"paired (worldtraj<text-only: {(rw < rt).sum()}/{len(ids)})")
fig.suptitle("text-only vs worldtraj+geo (SAME targets, both cam_dist_mean; diff = geo)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "_summary.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
print(f"text-only pos_rmse={rt.mean():.4f} | worldtraj+geo={rw.mean():.4f} | worldtraj better {(rw<rt).sum()}/{len(ids)}")
print("saved", OUT)
