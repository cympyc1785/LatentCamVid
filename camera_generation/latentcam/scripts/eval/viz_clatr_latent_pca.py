"""PRDC 가 실제로 사는 공간(CLaTr trajectory latent, 256-D)을 코퍼스별로 눈으로 본다.

왜 필요한가
-----------
`clatr/precision`, `clatr/density` 는 "예측이 real 궤적의 k-NN 공 안에 들어갔나"를 센다
(`main/evaluate/eval/src/metrics/modules/prdc.py`, k=3, euclidean). 합격선인 **반경이
코퍼스마다 real 끼리의 밀집도로 정해지기 때문에** 절대값을 코퍼스 간에 비교할 수 없다.
그 밀집도 차이를 실제 latent 분포로 확인한다.

그림 3 열 (코퍼스당 1 행)
  1) PCA 2-D: GT(파랑) vs pred(빨강). PCA 는 GT+pred 를 합쳐 적합.
  2) 같은 PCA 위에서 GT 만, SD 는 카메라 프리셋별 색. real 이 몇 덩어리인지 본다.
  3) **PRDC 가 실제로 비교하는 두 거리의 히스토그램** — real 의 k-NN 반경(합격선) vs
     각 pred 의 가장 가까운 real 까지 거리. 후자가 전자보다 크면 그 pred 는 탈락.

주의: 2-D PCA 는 256-D 의 일부만 담는다 (explained variance 를 제목에 찍는다). 그림에서
가까워 보여도 PRDC 판정과 다를 수 있다 — 판정의 근거는 3 열이다.

usage
-----
  python scripts/eval/viz_clatr_latent_pca.py \
      --run SD=results/20260810_010756_sd_whuman_textonly \
      --run DL3DV=results/20260808_140209_da3_7k_customgeo_nos \
      --out results/compare/clatr_latent_pca
"""
import argparse
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

K = 3          # prdc.py manifold_k


def pca2(X):
    """열 2 개짜리 PCA. 반환: 투영, explained variance ratio(2 개 합)."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    return Xc @ Vt[:2].T, float(var[:2].sum() / var.sum())


def sd_preset(fn):
    t = fn.split("/")[-1].replace("_transforms_pred", "").replace("_transforms_ref", "").rstrip(".")
    if "__" not in t:
        return None
    p = t.split("__")[1].split("_")
    return "_".join(p[-2:]) if len(p) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=path/to/run")
    ap.add_argument("--out", default="results/compare/clatr_latent_pca")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = [s.split("=", 1) for s in a.run]
    fig, axs = plt.subplots(len(runs), 3, figsize=(16.5, 5.0 * len(runs)), squeeze=False)
    summary = {}

    for row, (lab, p) in enumerate(runs):
        d = np.load(osp.join(p, "preds.npy"), allow_pickle=True).item()
        real = torch.stack(list(d["m_ref_latents"])).float()
        fake = torch.stack(list(d["m_pred_latents"])).float()
        N = len(real)

        # --- PRDC 가 쓰는 두 거리 (전체를 한 manifold 로 봤을 때) ---
        rr = torch.cdist(real[None], real[None], 2).squeeze(0)
        radii = torch.topk(rr, K + 1, largest=False, dim=-1).values.max(-1).values   # (N,)
        rf = torch.cdist(real[None], fake[None], 2).squeeze(0)                        # (N_real, N_fake)
        min_rf = rf.min(0).values                                                     # pred 별 최근접 real 거리
        off = rr[~torch.eye(N, dtype=bool)]

        Z, ev = pca2(torch.cat([real, fake]).numpy())
        zr, zf = Z[:N], Z[N:]

        ax = axs[row][0]
        ax.scatter(zr[:, 0], zr[:, 1], c="tab:blue", s=26, alpha=0.75, label=f"GT (n={N})")
        ax.scatter(zf[:, 0], zf[:, 1], c="tab:red", s=26, alpha=0.75, marker="x", label="pred")
        ax.set_title(f"{lab}: CLaTr latent PCA  (2-D explains {ev*100:.1f}% of 256-D var)", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

        ax = axs[row][1]
        pres = [sd_preset(f) for f in d["ref_filenames"]]
        if all(pres):
            for pz in sorted(set(pres)):
                m = np.array([x == pz for x in pres])
                ax.scatter(zr[m, 0], zr[m, 1], s=30, alpha=0.85, label=f"{pz} (n={m.sum()})")
            ax.legend(fontsize=7)
            ax.set_title(f"{lab}: GT only, colored by camera preset", fontsize=10)
        else:
            ax.scatter(zr[:, 0], zr[:, 1], c="tab:blue", s=26, alpha=0.75)
            ax.set_title(f"{lab}: GT only (no preset labels)", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=0.25)

        ax = axs[row][2]
        bins = np.linspace(0, float(max(radii.max(), min_rf.max())) * 1.05, 40)
        ax.hist(radii.numpy(), bins=bins, alpha=0.6, color="tab:blue",
                label=f"GT k-NN radius (k={K})  = pass threshold")
        ax.hist(min_rf.numpy(), bins=bins, alpha=0.6, color="tab:red",
                label="pred -> nearest GT distance")
        ax.axvline(float(radii.median()), color="tab:blue", ls="--", lw=2)
        ax.axvline(float(min_rf.median()), color="tab:red", ls="--", lw=2)
        inside = float((min_rf < radii.median()).double().mean())
        ax.set_title(f"{lab}: the two distances PRDC compares\n"
                     f"radius med {radii.median():.2f} vs pred-dist med {min_rf.median():.2f}  "
                     f"(ratio {float(min_rf.median()/radii.median()):.2f})", fontsize=10)
        ax.set_xlabel("euclidean distance in 256-D CLaTr latent"); ax.set_ylabel("count")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)

        rep = json.load(open(osp.join(p, "metrics.json")))
        summary[lab] = dict(
            n=N, pca_explained_var_2d=ev,
            real_real_dist_med=float(off.median()), knn_radius_med=float(radii.median()),
            radius_over_dist=float(radii.median() / off.median()),
            pred_nearest_real_med=float(min_rf.median()),
            ratio_preddist_over_radius=float(min_rf.median() / radii.median()),
            frac_pred_closer_than_median_radius=inside,
            reported_precision=rep.get("val/clatr/precision"),
            reported_density=rep.get("val/clatr/density"),
        )
        print(f"[{lab}] N={N} ev2d={ev*100:.1f}%  radius_med={radii.median():.3f}  "
              f"pred_dist_med={min_rf.median():.3f}  ratio={float(min_rf.median()/radii.median()):.3f}")

    fig.suptitle("CLaTr trajectory-latent distribution: GT (blue) vs pred (red).  "
                 "Column 3 is what PRDC actually thresholds.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(osp.join(a.out, "latent_pca.png"), dpi=125, bbox_inches="tight")
    plt.close(fig)
    with open(osp.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("out ->", osp.join(a.out, "latent_pca.png"))


if __name__ == "__main__":
    main()
