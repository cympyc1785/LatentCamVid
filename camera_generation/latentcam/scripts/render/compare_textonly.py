"""Visualize the text-only model (20260719_210144_dl3dv_textonly) inference in results/compare.
Standalone (its own 160 validation targets — disjoint from the geo point/dist set, so NOT
per-target aligned with them): per-target top-down (GT vs pred, first-cam anchored X/-Z) + scores
(world pos_rmse/rot + CLaTr). No geo-context stars (geo_encoder=null). Plus:
  _summary.png  : textonly pos_rmse + CLaTr (sorted + hist)
  _vs_geo.png   : DISTRIBUTIONAL box comparison textonly vs point(worldtraj) vs dist(align)
                  — different targets per model, so this compares score distributions, not pairs.
out -> results/compare/textonly/

인자 없이 실행하면 위 기본 동작 그대로다 (--run/--out 미지정 = 옛 dl3dv_textonly run).
Scene-Decoupled(SD) run 처럼 다른 run 을 보려면:

  python scripts/render/compare_textonly.py \
      --run results/20260810_010756_sd_whuman_textonly \
      --out results/compare/sd_whuman_textonly --grid --preset-overlay --no-per-target

  --grid           : 전 target 을 한 장의 contact sheet 로 (target 별 PNG 80장 대신)
  --preset-overlay : SD 카메라 프리셋(_01_24mm 등)별로 GT/pred 를 전부 겹쳐 그린다.
                     프리셋 평균 궤적을 굵게 -> 모델이 "프리셋 평균"으로 붕괴했는지 눈으로 본다.
"""
import argparse
import os, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/results"

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", default=os.path.join(ROOT, "20260719_210144_dl3dv_textonly"),
                 help="run dir (test/ 와 preds.npy 를 가진 곳)")
_ap.add_argument("--out", default=os.path.join(ROOT, "compare", "textonly"))
_ap.add_argument("--grid", action="store_true")
_ap.add_argument("--preset-overlay", action="store_true")
_ap.add_argument("--no-per-target", action="store_true", help="target 별 PNG 생략")
_a = _ap.parse_args()

TO = _a.run
OUT = _a.out
os.makedirs(OUT, exist_ok=True)
PVD = os.path.join(ROOT, "compare", "normalization_point_vs_dist")


def sd_preset(name):
    """SD target id '<scene>__<target clip>__<context clip>' 의 target clip 프리셋 접미사."""
    tgt = name.split("__")[1] if "__" in name else name
    parts = tgt.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else "?"


def load(p):
    return np.array([f["transform_matrix"] for f in json.load(open(p))["frames"]], float)


def rot_err_deg(Ra, Rb):
    R = np.einsum("tij,tkj->tik", Ra, Rb)
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def anchor_pts(pts, R0inv):
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    return (h @ R0inv.T)[:, :3]


def clatr_from_preds(path):
    d = np.load(path, allow_pickle=True).item()
    mp = torch.stack(list(d["m_pred_latents"])).float(); tl = torch.stack(list(d["t_latents"])).float()
    mp = mp / mp.norm(dim=-1, keepdim=True); tl = tl / tl.norm(dim=-1, keepdim=True)
    score = 100 * (mp * tl).sum(-1)
    out = {}
    for fn, s in zip(d["pred_filenames"], score.tolist()):
        tid = fn.split("/")[-1].replace("_transforms_pred", "").rstrip(".")
        out[tid] = max(s, 0.0)
    return out


clatr = clatr_from_preds(os.path.join(TO, "preds.npy"))
ids = sorted({os.path.basename(f)[:-len("_transforms_ref.json")]
              for f in glob.glob(os.path.join(TO, "test", "*_transforms_ref.json"))})
print(f"{len(ids)} textonly targets")

rows = []
anch = {}          # tid -> (GT xz, pred xz) : target 첫 카메라 좌표계 anchor 후 X / -Z
for i, tid in enumerate(ids):
    ref = load(os.path.join(TO, "test", f"{tid}_transforms_ref.json"))
    prd = load(os.path.join(TO, "test", f"{tid}_transforms_pred.json"))
    d = np.linalg.norm(prd[:, :3, 3] - ref[:, :3, 3], axis=1)
    pos_rmse = float(np.sqrt((d ** 2).mean())); rot = float(rot_err_deg(prd[:, :3, :3], ref[:, :3, :3]).mean())
    cl = float(clatr.get(tid, np.nan))
    rows.append((tid, pos_rmse, rot, cl))
    R0inv = np.linalg.inv(ref[0])
    gc = anchor_pts(ref[:, :3, 3], R0inv); pc = anchor_pts(prd[:, :3, 3], R0inv)
    anch[tid] = (np.stack([gc[:, 0], -gc[:, 2]], 1), np.stack([pc[:, 0], -pc[:, 2]], 1))
    if _a.no_per_target:
        continue
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2))
    ax.plot(gc[:, 0], -gc[:, 2], "-o", ms=3, lw=1.5, c="tab:blue", label="GT")
    ax.plot(pc[:, 0], -pc[:, 2], "-x", ms=3, lw=1.3, c="tab:red", label=f"textonly (rmse {pos_rmse:.2f}, CLaTr {cl:.1f})")
    ax.scatter(gc[0, 0], -gc[0, 2], c="k", s=80, marker="*", zorder=6, label="start")
    ax.set_aspect("equal", "datalim"); ax.legend(fontsize=8); ax.set_xlabel("X"); ax.set_ylabel("-Z")
    ax.set_title(f"{tid}\ntextonly  pos_rmse={pos_rmse:.2f}  rot={rot:.0f}deg  CLaTr={cl:.1f}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{tid}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(ids)}")

rmse_of = {t: r for t, r, _, _ in rows}

if _a.grid:
    order = sorted(ids, key=lambda t: (sd_preset(t), rmse_of[t]))
    nc = int(np.ceil(np.sqrt(len(order) * 1.3))); nr = int(np.ceil(len(order) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.05 * nc, 2.05 * nr))
    for ax, tid in zip(np.ravel(axs), order):
        g, p = anch[tid]
        ax.plot(g[:, 0], g[:, 1], "-", lw=1.4, c="tab:blue")
        ax.plot(p[:, 0], p[:, 1], "-", lw=1.2, c="tab:red")
        ax.scatter(*g[0], c="k", s=22, marker="*", zorder=5)
        ax.set_aspect("equal", "datalim"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{sd_preset(tid)}  {rmse_of[tid]:.2f}", fontsize=6.5, pad=1.5)
    for ax in np.ravel(axs)[len(order):]:
        ax.axis("off")
    fig.suptitle(f"{os.path.basename(TO)} - top-down  GT (blue) vs pred (red),  anchored to target first camera,  "
                 f"X vs -Z;  panel title = preset, world pos_rmse   ({len(order)} targets)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(os.path.join(OUT, "_grid.png"), dpi=125, bbox_inches="tight"); plt.close(fig)
    print("grid ->", os.path.join(OUT, "_grid.png"))

if _a.preset_overlay:
    pres = sorted({sd_preset(t) for t in ids})
    fig, axs = plt.subplots(2, len(pres), figsize=(3.5 * len(pres), 7.2), squeeze=False)
    for j, pz in enumerate(pres):
        sub = [t for t in ids if sd_preset(t) == pz]
        gs = np.stack([anch[t][0] for t in sub]); ps = np.stack([anch[t][1] for t in sub])
        for k, (arr, c, lab) in enumerate([(gs, "tab:blue", "GT"), (ps, "tab:red", "pred")]):
            ax = axs[k][j]
            for a in arr:
                ax.plot(a[:, 0], a[:, 1], "-", lw=0.7, c=c, alpha=0.35)
            m = arr.mean(0)
            ax.plot(m[:, 0], m[:, 1], "-", lw=2.6, c="k", label="mean")
            ax.scatter(0, 0, c="k", s=45, marker="*", zorder=6)
            sp = float(np.sqrt(((arr - m[None]) ** 2).sum(-1).mean()))   # 프리셋 내 퍼짐
            ax.set_title(f"{pz}  {lab}  n={len(sub)}  spread={sp:.2f}", fontsize=9)
            ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.25)
            ax.set_xlabel("X"); ax.set_ylabel("-Z")
    fig.suptitle(f"{os.path.basename(TO)} - per-preset overlay.  top row = GT, bottom row = pred, black = that preset's mean trajectory.\n"
                 "pred spread << GT spread  =>  collapsed onto the preset mean.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "_preset_overlay.png"), dpi=125, bbox_inches="tight"); plt.close(fig)
    print("preset overlay ->", os.path.join(OUT, "_preset_overlay.png"))

with open(os.path.join(OUT, "_scores.csv"), "w") as f:
    f.write("target,pos_rmse,rot_mean,clatr\n")
    for tid, pr, ro, cl in rows:
        f.write(f"{tid},{pr:.4f},{ro:.3f},{cl:.3f}\n")

pr = np.array([r[1] for r in rows]); cl = np.array([r[3] for r in rows])
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
axs[0].plot(np.sort(pr), "-o", ms=2, c="tab:red"); axs[0].set_title(f"textonly pos_rmse (mean {pr.mean():.3f})")
axs[0].set_xlabel("target (sorted)"); axs[0].set_ylabel("world pos_rmse")
axs[1].plot(np.sort(cl), "-o", ms=2, c="tab:red"); axs[1].set_title(f"textonly CLaTr (mean {np.nanmean(cl):.2f})")
axs[1].set_xlabel("target (sorted)"); axs[1].set_ylabel("CLaTr score")
fig.suptitle(f"{os.path.basename(TO)} (textonly, no geo) — {len(ids)} targets", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "_summary.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# ---- distributional comparison vs geo point/dist (different targets -> boxplot of distributions) ----
def read_pvd():
    import csv
    sc = {r["target"]: r for r in csv.DictReader(open(os.path.join(PVD, "_scores.csv")))}
    cj = json.load(open(os.path.join(PVD, "_clatr.json")))
    pt_rmse = np.array([float(v["point_pos_rmse"]) for v in sc.values()])
    di_rmse = np.array([float(v["dist_pos_rmse"]) for v in sc.values()])
    return pt_rmse, di_rmse, np.array(list(cj["point"].values())), np.array(list(cj["dist"].values()))


try:
    if "dl3dv_textonly" not in os.path.basename(TO):
        raise RuntimeError("다른 run 이라 point/dist 비교 대상이 아니다")
    pt_r, di_r, pt_c, di_c = read_pvd()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    axs[0].boxplot([pr, pt_r, di_r], labels=[f"textonly\n{pr.mean():.2f}", f"point\n{pt_r.mean():.2f}", f"dist\n{di_r.mean():.2f}"], showmeans=True)
    axs[0].set_title("world pos_rmse distribution (lower=better)"); axs[0].set_ylabel("pos_rmse")
    axs[1].boxplot([cl[~np.isnan(cl)], pt_c, di_c], labels=[f"textonly\n{np.nanmean(cl):.1f}", f"point\n{pt_c.mean():.1f}", f"dist\n{di_c.mean():.1f}"], showmeans=True)
    axs[1].set_title("CLaTr score distribution (higher=better)"); axs[1].set_ylabel("CLaTr")
    fig.suptitle("distribution comparison (DIFFERENT targets per model — not per-pair): textonly vs point vs dist", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "_vs_geo.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"textonly pos_rmse={pr.mean():.3f} CLaTr={np.nanmean(cl):.2f} | point rmse={pt_r.mean():.3f} c={pt_c.mean():.2f} | dist rmse={di_r.mean():.3f} c={di_c.mean():.2f}")
except Exception as e:
    print("vs_geo skipped:", e)
print("saved to", OUT)
