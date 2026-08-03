"""Top-down comparison of N runs' predicted trajectories against the SAME GT targets.

Per-target top-down (first-cam anchored X/-Z): GT (blue) + one colored polyline per run
(+ optional geo-context cameras as magenta stars) with each run's world pos_rmse in the legend.
Plus _summary.png (per-target pos_rmse curves + pairwise scatter vs the first run), _scores.csv,
and _contact.png (a grid of the first --contact targets for a quick eyeball).

[2026-07-31] generalized from the original hard-coded 2-way (text-only vs worldtraj) script.
Running it with NO arguments reproduces exactly that old behaviour:
    text-only re-inferred on worldtraj's 160 targets by infer_textonly_batch.py, both under
    cam_dist_mean normalization, so the only difference is geo conditioning.
    out -> results/compare/text-only_vs_worldtraj/

N-way usage (each --run is label=dir, dir holds <tid>_transforms_pred.json):
    python scripts/render/compare_textonly_vs_worldtraj.py \
        --run text-only=<...>/test --run worldtraj=<...>/test --run camembed=<...>/test \
        --out results/topdown_3way/plots
The GT refs come from --ref (default: the first --run dir, which must hold *_transforms_ref.json).
"""
import os, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results"
# legacy 2-way defaults (used only when --run is not passed)
WT = os.path.join(ROOT, "20260721_213927_dl3dv_geo_worldtraj", "test")             # GT + worldtraj pred
TO = os.path.join(ROOT, "compare", "text-only_vs_worldtraj", "textonly_preds")      # text-only pred (matched)
DEFAULT_OUT = os.path.join(ROOT, "compare", "text-only_vs_worldtraj")
DEFAULT_CTX = os.path.join(ROOT, "compare", "normalization_point_vs_dist", "_geo_context.json")

COLORS = ["tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:cyan"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                    help="repeatable; DIR holds <tid>_transforms_pred.json")
    ap.add_argument("--ref", default=None, help="dir with <tid>_transforms_ref.json (default: first --run dir)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ctx", default=None, help="json {tid: [[x,y,z], ...]} of geo context camera centers")
    ap.add_argument("--contact", type=int, default=24, help="how many targets in the contact sheet (0 = off)")
    ap.add_argument("--per-target", dest="per_target", type=int, default=-1,
                    help="how many per-target PNGs to write (-1 = all)")
    args = ap.parse_args()

    if args.run:
        runs = [(r.split("=", 1)[0], r.split("=", 1)[1]) for r in args.run]
        ref_dir = args.ref or runs[0][1]
        out = args.out or os.path.join(ROOT, "compare", "topdown_nway")
        ctx_path = args.ctx
    else:   # legacy 2-way default -- unchanged
        runs = [("text-only", TO), ("worldtraj+geo", WT)]
        ref_dir, out, ctx_path = WT, args.out or DEFAULT_OUT, args.ctx or DEFAULT_CTX
    os.makedirs(out, exist_ok=True)
    ctx = json.load(open(ctx_path)) if (ctx_path and os.path.exists(ctx_path)) else {}

    ids = sorted({os.path.basename(f)[: -len("_transforms_ref.json")]
                  for f in glob.glob(os.path.join(ref_dir, "*_transforms_ref.json"))})
    for _, d in runs:   # keep only targets every run predicted
        have = {os.path.basename(f)[: -len("_transforms_pred.json")]
                for f in glob.glob(os.path.join(d, "*_transforms_pred.json"))}
        ids = [t for t in ids if t in have]
    print(f"{len(ids)} common targets over {len(runs)} runs")

    rows, gallery = [], []
    n_png = len(ids) if args.per_target < 0 else min(args.per_target, len(ids))
    for i, tid in enumerate(ids):
        ref = load(os.path.join(ref_dir, f"{tid}_transforms_ref.json"))
        preds = [load(os.path.join(d, f"{tid}_transforms_pred.json")) for _, d in runs]
        rs = [rmse(p, ref) for p in preds]
        ro = [float(rot_err_deg(p[:, :3, :3], ref[:, :3, :3]).mean()) for p in preds]
        rows.append((tid, rs, ro))
        R0 = np.linalg.inv(ref[0])
        g = anchor(ref[:, :3, 3], R0)
        ps = [anchor(p[:, :3, 3], R0) for p in preds]
        cc = anchor(np.array(ctx[tid]), R0) if tid in ctx else None
        if len(gallery) < args.contact:
            gallery.append((tid, g, ps, rs, cc))
        if i >= n_png:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4))
        draw(ax, tid, g, ps, rs, cc, runs)
        ax.set_title(f"{tid}\n" + "  |  ".join(f"{lab} rmse={r:.2f}" for (lab, _), r in zip(runs, rs)),
                     fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{tid}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(ids)}")

    labels = [lab for lab, _ in runs]
    with open(os.path.join(out, "_scores.csv"), "w") as f:
        f.write("target," + ",".join(f"{l}_pos_rmse" for l in labels)
                + "," + ",".join(f"{l}_rot_deg" for l in labels) + "\n")
        for tid, rs, ro in rows:
            f.write(tid + "," + ",".join(f"{v:.4f}" for v in rs)
                    + "," + ",".join(f"{v:.3f}" for v in ro) + "\n")

    R = np.array([r[1] for r in rows])                    # (N_targets, N_runs)
    RO = np.array([r[2] for r in rows])
    summary(out, labels, R, RO, len(ids))
    if gallery:
        contact(out, gallery, runs)
    for j, l in enumerate(labels):
        print(f"{l:28s} pos_rmse mean={R[:, j].mean():.4f} median={np.median(R[:, j]):.4f} "
              f"| rot_deg mean={RO[:, j].mean():.3f}")
    print("saved", out)


def draw(ax, tid, g, ps, rs, cc, runs):
    ax.plot(g[:, 0], -g[:, 2], "-o", ms=3, lw=1.5, c="tab:blue", label="GT")
    for j, ((lab, _), p, r) in enumerate(zip(runs, ps, rs)):
        ax.plot(p[:, 0], -p[:, 2], "-x", ms=3, lw=1.3, c=COLORS[j % len(COLORS)],
                label=f"{lab} (rmse {r:.2f})")
    ax.scatter(g[0, 0], -g[0, 2], c="k", s=80, marker="*", zorder=6, label="start")
    if cc is not None:
        ax.scatter(cc[:, 0], -cc[:, 2], c="magenta", s=90, marker="*", edgecolors="k",
                   linewidths=0.5, zorder=7, label=f"geo context ({len(cc)})")
    ax.set_aspect("equal", "datalim"); ax.set_xlabel("X"); ax.set_ylabel("-Z")
    ax.legend(fontsize=7)


def summary(out, labels, R, RO, n):
    # [new] run 이 하나면 비교 산점도가 없으므로 빈 axis 를 만들지 않는다 (예전엔 항상 1개 남았다)
    ncmp = len(labels) - 1
    fig, axs = plt.subplots(1, 1 + ncmp, figsize=(6.5 * (1 + ncmp), 5.5), squeeze=False)
    axs = axs[0]
    o = np.argsort(R[:, 0])
    for j, l in enumerate(labels):
        axs[0].plot(R[o, j], "-o", ms=2, c=COLORS[j % len(COLORS)],
                    label=f"{l} (mean {R[:, j].mean():.3f})")
    axs[0].set_title(f"world pos_rmse per target (sorted by {labels[0]})")
    axs[0].legend(fontsize=8); axs[0].set_xlabel("target"); axs[0].set_ylabel("pos_rmse")
    lim = R.max() * 1.05
    for j in range(1, len(labels)):
        a = axs[j]
        a.scatter(R[:, 0], R[:, j], s=10, alpha=0.6, c=COLORS[j % len(COLORS)])
        a.plot([0, lim], [0, lim], "k--", lw=0.8)
        a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
        a.set_xlabel(f"{labels[0]} pos_rmse"); a.set_ylabel(f"{labels[j]} pos_rmse")
        a.set_title(f"{labels[j]} better than {labels[0]}: {(R[:, j] < R[:, 0]).sum()}/{n}")
    fig.suptitle("top-down trajectory comparison (same targets)", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(out, "_summary.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def contact(out, gallery, runs):
    n = len(gallery); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    for k, (tid, g, ps, rs, cc) in enumerate(gallery):
        ax = axs[k // ncol][k % ncol]
        ax.plot(g[:, 0], -g[:, 2], "-", lw=1.6, c="tab:blue")
        for j, p in enumerate(ps):
            ax.plot(p[:, 0], -p[:, 2], "-", lw=1.1, c=COLORS[j % len(COLORS)])
        ax.scatter(g[0, 0], -g[0, 2], c="k", s=30, marker="*", zorder=6)
        if cc is not None:      # [new] per-target PNG 과 마찬가지로 geo context 카메라도 찍는다
            ax.scatter(cc[:, 0], -cc[:, 2], c="magenta", s=28, marker="*", edgecolors="k",
                       linewidths=0.4, zorder=7)
        ax.set_aspect("equal", "datalim"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tid.split("_")[1][:8] + "_" + tid.split("_")[-1] + "\n"
                     + " / ".join(f"{r:.2f}" for r in rs), fontsize=6)
    for k in range(n, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    handles = [plt.Line2D([], [], c="tab:blue", lw=2, label="GT")] + [
        plt.Line2D([], [], c=COLORS[j % len(COLORS)], lw=2, label=lab) for j, (lab, _) in enumerate(runs)]
    if any(cc is not None for *_, cc in gallery):
        handles.append(plt.Line2D([], [], c="magenta", lw=0, marker="*", ms=10,
                                  markeredgecolor="k", label="geo context"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10)
    fig.suptitle("top-down contact sheet (titles: pos_rmse in legend order)", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(out, "_contact.png"), dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
