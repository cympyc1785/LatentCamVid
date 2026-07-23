"""Compare two normalization schemes (point = target_cam / dist = first_farthest_135) on the
SAME 160 validation targets: per-target top-down (GT vs both preds) + world-space scores, plus a
summary. Metrics are computed in DENORMALIZED world (from the saved OpenGL-c2w transforms JSON),
so they are directly comparable regardless of each model's normalization/VAE.

Top-down: anchor everything to the common GT first camera (inv(c2w[0]) @ c2w) -> OpenCV-free cam0
frame (up=+Y); plot X horizontal, -Z vertical (forward up), matching results/validation convention.

out -> results/compare/normalization_point_vs_dist/{<target>.png ..., _summary.png, _scores.csv}
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results"
POINT = os.path.join(ROOT, "20260721_213927_dl3dv_geo_worldtraj", "test")    # target_cam (point)
DIST  = os.path.join(ROOT, "20260721_233547_dl3dv_geo_worldtraj_align", "test")  # first_farthest_135 (dist)
OUT   = os.path.join(ROOT, "compare", "normalization_point_vs_dist")
os.makedirs(OUT, exist_ok=True)


def load(p):
    return np.array([f["transform_matrix"] for f in json.load(open(p))["frames"]], float)


def anchor(c2w, R0inv):                      # -> cam0-relative; centers for top-down
    return R0inv @ c2w


def rot_err_deg(Ra, Rb):                     # mean geodesic angle between rotation sets
    R = np.einsum("tij,tkj->tik", Ra, Rb)    # Ra @ Rb^T
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def scores(pred, ref):                        # world-space (denormalized) pred vs ref
    d = np.linalg.norm(pred[:, :3, 3] - ref[:, :3, 3], axis=1)
    rot = rot_err_deg(pred[:, :3, :3], ref[:, :3, :3])
    return dict(pos_rmse=float(np.sqrt((d ** 2).mean())), pos_mean=float(d.mean()),
                pos_max=float(d.max()), rot_mean=float(rot.mean()))


def target_ids(d):
    return sorted({os.path.basename(f)[:-len("_transforms_ref.json")]
                   for f in glob.glob(os.path.join(d, "*_transforms_ref.json"))})


ids = target_ids(POINT)
assert ids == target_ids(DIST), "target sets differ!"
print(f"{len(ids)} common targets")

rows = []
for i, tid in enumerate(ids):
    pr_ref = load(os.path.join(POINT, f"{tid}_transforms_ref.json"))
    pr_prd = load(os.path.join(POINT, f"{tid}_transforms_pred.json"))
    dt_prd = load(os.path.join(DIST,  f"{tid}_transforms_pred.json"))
    s_point, s_dist = scores(pr_prd, pr_ref), scores(dt_prd, pr_ref)
    rows.append((tid, s_point, s_dist))

    R0inv = np.linalg.inv(pr_ref[0])          # common anchor = GT first cam
    g, p, d = anchor(pr_ref, R0inv), anchor(pr_prd, R0inv), anchor(dt_prd, R0inv)
    gc, pc, dc = g[:, :3, 3], p[:, :3, 3], d[:, :3, 3]
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
    ax.plot(gc[:, 0], -gc[:, 2], "-o", ms=3, lw=1.5, c="tab:blue", label="GT")
    ax.plot(pc[:, 0], -pc[:, 2], "-x", ms=3, lw=1.3, c="tab:orange",
            label=f"point (rmse {s_point['pos_rmse']:.3f})")
    ax.plot(dc[:, 0], -dc[:, 2], "-x", ms=3, lw=1.3, c="tab:green",
            label=f"dist (rmse {s_dist['pos_rmse']:.3f})")
    ax.scatter(gc[0, 0], -gc[0, 2], c="k", s=70, marker="*", zorder=5)
    ax.set_aspect("equal", "datalim"); ax.legend(fontsize=8); ax.set_xlabel("X"); ax.set_ylabel("-Z")
    ax.set_title(f"{tid}\npoint pos_rmse={s_point['pos_rmse']:.3f} rot={s_point['rot_mean']:.1f}deg  |  "
                 f"dist pos_rmse={s_dist['pos_rmse']:.3f} rot={s_dist['rot_mean']:.1f}deg", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{tid}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(ids)}")

# scores CSV
with open(os.path.join(OUT, "_scores.csv"), "w") as f:
    f.write("target,point_pos_rmse,point_pos_mean,point_rot_mean,dist_pos_rmse,dist_pos_mean,dist_rot_mean\n")
    for tid, sp, sd in rows:
        f.write(f"{tid},{sp['pos_rmse']:.4f},{sp['pos_mean']:.4f},{sp['rot_mean']:.3f},"
                f"{sd['pos_rmse']:.4f},{sd['pos_mean']:.4f},{sd['rot_mean']:.3f}\n")

# summary
pp = np.array([r[1]["pos_rmse"] for r in rows])
dp = np.array([r[2]["pos_rmse"] for r in rows])
pr = np.array([r[1]["rot_mean"] for r in rows])
dr = np.array([r[2]["rot_mean"] for r in rows])
fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
order = np.argsort(pp)
axs[0].plot(pp[order], "-o", ms=2, c="tab:orange", label=f"point (mean {pp.mean():.3f})")
axs[0].plot(dp[order], "-o", ms=2, c="tab:green", label=f"dist (mean {dp.mean():.3f})")
axs[0].set_title("world pos_rmse per target (sorted by point)"); axs[0].set_xlabel("target"); axs[0].legend()
axs[1].scatter(pp, dp, s=10, alpha=0.6)
lim = max(pp.max(), dp.max()) * 1.05
axs[1].plot([0, lim], [0, lim], "k--", lw=0.8)
axs[1].set_xlim(0, lim); axs[1].set_ylim(0, lim); axs[1].set_aspect("equal")
axs[1].set_xlabel("point pos_rmse"); axs[1].set_ylabel("dist pos_rmse")
axs[1].set_title(f"paired (dist<point on {(dp<pp).sum()}/{len(ids)} targets)")
axs[2].plot(pr[order], "-o", ms=2, c="tab:orange", label=f"point rot (mean {pr.mean():.2f}deg)")
axs[2].plot(dr[order], "-o", ms=2, c="tab:green", label=f"dist rot (mean {dr.mean():.2f}deg)")
axs[2].set_title("mean rotation error per target"); axs[2].set_xlabel("target"); axs[2].legend()
fig.suptitle(f"normalization: point (target_cam) vs dist (first_farthest_135)  [world-space, {len(ids)} targets]",
             fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "_summary.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"\npoint  pos_rmse mean={pp.mean():.4f} median={np.median(pp):.4f}  rot mean={pr.mean():.3f}deg")
print(f"dist   pos_rmse mean={dp.mean():.4f} median={np.median(dp):.4f}  rot mean={dr.mean():.3f}deg")
print(f"dist better on {(dp<pp).sum()}/{len(ids)} targets (pos_rmse)")
print(f"saved to {OUT}")
