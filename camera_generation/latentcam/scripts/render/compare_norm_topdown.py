"""Compare two normalization schemes (point = target_cam / dist = first_farthest_135) on the
SAME 160 validation targets: per-target top-down (GT vs both preds + geo-context cameras) and
scores (world-space pos_rmse/rot + CLaTr score), plus a summary.

Metrics in DENORMALIZED world (from the saved OpenGL-c2w transforms JSON) so they compare across
each model's normalization/VAE. CLaTr score (100*cos(pred_traj_latent, text_latent), higher=better)
is read from _clatr.json (produced by evaluate/CLaTr src.extraction -> preds.npy, parsed per target).
Geo-context camera centers (the LagerNVS conditioning views; same for both models) are read from
_geo_context.json and overlaid as magenta stars to inspect whether context placement drives error.

Top-down: anchor everything to the common GT first camera (inv(c2w[0]) @ c2w); plot X vs -Z.
out -> results/compare/normalization_point_vs_dist/{<target>.png ..., _summary.png, _scores.csv}
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results"
POINT = os.path.join(ROOT, "20260721_213927_dl3dv_geo_worldtraj", "test")
DIST  = os.path.join(ROOT, "20260721_233547_dl3dv_geo_worldtraj_align", "test")
OUT   = os.path.join(ROOT, "compare", "normalization_point_vs_dist")
os.makedirs(OUT, exist_ok=True)
CLATR = os.path.join(OUT, "_clatr.json")
CTX   = os.path.join(OUT, "_geo_context.json")

clatr = json.load(open(CLATR)) if os.path.exists(CLATR) else {"point": {}, "dist": {}}
ctx = json.load(open(CTX)) if os.path.exists(CTX) else {}
print("clatr:", bool(clatr["point"]), "| context:", len(ctx), "targets")


def load(p):
    return np.array([f["transform_matrix"] for f in json.load(open(p))["frames"]], float)


def rot_err_deg(Ra, Rb):
    R = np.einsum("tij,tkj->tik", Ra, Rb)
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def scores(pred, ref):
    d = np.linalg.norm(pred[:, :3, 3] - ref[:, :3, 3], axis=1)
    return dict(pos_rmse=float(np.sqrt((d ** 2).mean())), pos_mean=float(d.mean()),
                rot_mean=float(rot_err_deg(pred[:, :3, :3], ref[:, :3, :3]).mean()))


def anchor_pts(pts_xyz, R0inv):                 # (N,3) world -> cam0 frame
    h = np.concatenate([pts_xyz, np.ones((len(pts_xyz), 1))], 1)
    return (h @ R0inv.T)[:, :3]


ids = sorted({os.path.basename(f)[:-len("_transforms_ref.json")]
              for f in glob.glob(os.path.join(POINT, "*_transforms_ref.json"))})
print(f"{len(ids)} targets")

rows = []
for i, tid in enumerate(ids):
    ref = load(os.path.join(POINT, f"{tid}_transforms_ref.json"))
    pp = load(os.path.join(POINT, f"{tid}_transforms_pred.json"))
    dp = load(os.path.join(DIST,  f"{tid}_transforms_pred.json"))
    sp, sd = scores(pp, ref), scores(dp, ref)
    sp["clatr"] = float(clatr["point"].get(tid, np.nan))
    sd["clatr"] = float(clatr["dist"].get(tid, np.nan))
    rows.append((tid, sp, sd))

    R0inv = np.linalg.inv(ref[0])
    gc = anchor_pts(ref[:, :3, 3], R0inv)
    pc = anchor_pts(pp[:, :3, 3], R0inv)
    dc = anchor_pts(dp[:, :3, 3], R0inv)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
    ax.plot(gc[:, 0], -gc[:, 2], "-o", ms=3, lw=1.5, c="tab:blue", label="GT")
    ax.plot(pc[:, 0], -pc[:, 2], "-x", ms=3, lw=1.3, c="tab:orange",
            label=f"point (rmse {sp['pos_rmse']:.2f}, CLaTr {sp['clatr']:.1f})")
    ax.plot(dc[:, 0], -dc[:, 2], "-x", ms=3, lw=1.3, c="tab:green",
            label=f"dist (rmse {sd['pos_rmse']:.2f}, CLaTr {sd['clatr']:.1f})")
    ax.scatter(gc[0, 0], -gc[0, 2], c="k", s=80, marker="*", zorder=6, label="start (frame s)")
    if tid in ctx:                                # geo-context cameras (conditioning views)
        cc = anchor_pts(np.array(ctx[tid]), R0inv)
        ax.scatter(cc[:, 0], -cc[:, 2], c="magenta", s=90, marker="*", edgecolors="k",
                   linewidths=0.5, zorder=7, label=f"geo context ({len(cc)})")
    ax.set_aspect("equal", "datalim"); ax.legend(fontsize=8); ax.set_xlabel("X"); ax.set_ylabel("-Z")
    ax.set_title(f"{tid}\npoint rmse={sp['pos_rmse']:.2f} rot={sp['rot_mean']:.0f}d CLaTr={sp['clatr']:.1f}  |  "
                 f"dist rmse={sd['pos_rmse']:.2f} rot={sd['rot_mean']:.0f}d CLaTr={sd['clatr']:.1f}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{tid}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(ids)}")

with open(os.path.join(OUT, "_scores.csv"), "w") as f:
    f.write("target,point_pos_rmse,point_rot,point_clatr,dist_pos_rmse,dist_rot,dist_clatr\n")
    for tid, sp, sd in rows:
        f.write(f"{tid},{sp['pos_rmse']:.4f},{sp['rot_mean']:.3f},{sp['clatr']:.3f},"
                f"{sd['pos_rmse']:.4f},{sd['rot_mean']:.3f},{sd['clatr']:.3f}\n")

pp_ = np.array([r[1]["pos_rmse"] for r in rows]); dp_ = np.array([r[2]["pos_rmse"] for r in rows])
pcl = np.array([r[1]["clatr"] for r in rows]); dcl = np.array([r[2]["clatr"] for r in rows])
fig, axs = plt.subplots(2, 2, figsize=(13, 11))
o = np.argsort(pp_)
axs[0, 0].plot(pp_[o], "-o", ms=2, c="tab:orange", label=f"point (mean {pp_.mean():.3f})")
axs[0, 0].plot(dp_[o], "-o", ms=2, c="tab:green", label=f"dist (mean {dp_.mean():.3f})")
axs[0, 0].set_title("world pos_rmse (sorted by point)"); axs[0, 0].legend(); axs[0, 0].set_xlabel("target")
lim = max(pp_.max(), dp_.max()) * 1.05
axs[0, 1].scatter(pp_, dp_, s=10, alpha=0.6); axs[0, 1].plot([0, lim], [0, lim], "k--", lw=0.8)
axs[0, 1].set_xlim(0, lim); axs[0, 1].set_ylim(0, lim); axs[0, 1].set_aspect("equal")
axs[0, 1].set_xlabel("point pos_rmse"); axs[0, 1].set_ylabel("dist pos_rmse")
axs[0, 1].set_title(f"pos_rmse paired (dist<point: {(dp_<pp_).sum()}/{len(ids)})")
oc = np.argsort(pcl)
axs[1, 0].plot(pcl[oc], "-o", ms=2, c="tab:orange", label=f"point (mean {np.nanmean(pcl):.2f})")
axs[1, 0].plot(dcl[oc], "-o", ms=2, c="tab:green", label=f"dist (mean {np.nanmean(dcl):.2f})")
axs[1, 0].set_title("CLaTr score (sorted by point, higher=better)"); axs[1, 0].legend(); axs[1, 0].set_xlabel("target")
cl = max(np.nanmax(pcl), np.nanmax(dcl)) * 1.05
axs[1, 1].scatter(pcl, dcl, s=10, alpha=0.6); axs[1, 1].plot([0, cl], [0, cl], "k--", lw=0.8)
axs[1, 1].set_xlim(0, cl); axs[1, 1].set_ylim(0, cl); axs[1, 1].set_aspect("equal")
axs[1, 1].set_xlabel("point CLaTr"); axs[1, 1].set_ylabel("dist CLaTr")
axs[1, 1].set_title(f"CLaTr paired (dist>point: {(dcl>pcl).sum()}/{len(ids)})")
fig.suptitle(f"point (target_cam) vs dist (first_farthest_135)  [world-space, {len(ids)} targets]", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "_summary.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"\npoint  pos_rmse={pp_.mean():.4f}  rot={np.mean([r[1]['rot_mean'] for r in rows]):.2f}d  CLaTr={np.nanmean(pcl):.3f}")
print(f"dist   pos_rmse={dp_.mean():.4f}  rot={np.mean([r[2]['rot_mean'] for r in rows]):.2f}d  CLaTr={np.nanmean(dcl):.3f}")
print(f"dist better: pos_rmse {(dp_<pp_).sum()}/{len(ids)}, CLaTr {(dcl>pcl).sum()}/{len(ids)}")
print(f"saved to {OUT}")
