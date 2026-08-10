"""clatr/precision, clatr/density 를 **per-sample 로 분해**하고, caption fscore 와 엇갈리는
target(캡션은 맞췄는데 분포에서는 벗어난 것)을 골라 top-down 으로 그린다.

왜 분해가 되는가
----------------
`main/evaluate/eval/src/metrics/modules/prdc.py` 의 정의는 fake 샘플별 값의 평균이다:

    precision = (d(real_j, fake_i) < radius_j).any(axis=0).float().mean()
    density   = (1/k) * (d(real_j, fake_i) < radius_j).sum(axis=0).float().mean()

axis=0 이 real 축이므로 남는 축이 fake 샘플이다. 즉 sample i 마다
  precision_i in {0,1}      : real manifold 안에 들어갔나
  density_i   in [0, N_real/k] : 몇 개의 real k-NN ball 이 나를 덮나 (덮는 개수/k)
가 정의되고, 그 평균이 보고된 수치다. recall/coverage 는 real 축이 남아서 fake 샘플별 값이
아니므로 여기서는 내지 않는다.

주의: `ManifoldMetrics.compute(num_splits=5)` 는 전체를 5 chunk 로 쪼개 chunk 마다 PRDC 를
계산한 뒤 평균한다. 따라서 radius 도 **chunk 안에서만** 정해진다 — 여기서도 같은 chunk 순서
(= update 에 들어간 순서 = preds.npy 의 filename 순서)로 재현한다. 재현이 맞는지
집계값을 metrics.json 과 대조해 출력한다.

usage
-----
  python scripts/eval/prdc_per_sample.py --run results/20260810_010756_sd_whuman_textonly \
      --out results/compare/sd_whuman_textonly
out -> <out>/_prdc_per_sample.csv, <out>/_fscore_vs_prdc.png, <out>/_mismatch_topdown.png
"""
import argparse
import csv
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

K, NUM_SPLITS = 3, 5          # prdc.py: manifold_k=3, compute(num_splits=5)


def per_sample_prdc(real, fake):
    """chunk 별로 prdc.py 와 동일하게 계산하되 fake 샘플별 값을 보존한다."""
    prec = np.zeros(len(fake)); dens = np.zeros(len(fake))
    rs = torch.as_tensor(real).float().chunk(NUM_SPLITS, dim=0)
    fs = torch.as_tensor(fake).float().chunk(NUM_SPLITS, dim=0)
    off = 0
    for r, f in zip(rs, fs):
        rr = torch.cdist(r[None], r[None], 2).squeeze(0)
        radii = torch.topk(rr, K + 1, largest=False, dim=-1).values.max(-1).values   # (Nr,)
        d_rf = torch.cdist(r[None], f[None], 2).squeeze(0)                           # (Nr, Nf)
        hit = d_rf < radii.unsqueeze(1)
        prec[off:off + len(f)] = hit.any(axis=0).double().numpy()
        dens[off:off + len(f)] = (hit.sum(axis=0).double() / K).numpy()
        off += len(f)
    return prec, dens


def tid_of(fn):
    return fn.split("/")[-1].replace("_transforms_pred", "").replace("_transforms_ref", "").rstrip(".")


def load_tr(run, tid, kind):
    p = osp.join(run, "test", f"{tid}_transforms_{kind}.json")
    return np.asarray([f["transform_matrix"] for f in json.load(open(p))["frames"]], float)


def anchored(run, tid):
    ref = load_tr(run, tid, "ref"); prd = load_tr(run, tid, "pred")
    Ri = np.linalg.inv(ref[0])
    def go(m):
        h = np.c_[m[:, :3, 3], np.ones(len(m))]
        p = (h @ Ri.T)[:, :3]
        return np.stack([p[:, 0], -p[:, 2]], 1)
    return go(ref), go(prd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-show", type=int, default=12, help="mismatch 패널에 그릴 target 수")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = np.load(osp.join(a.run, "preds.npy"), allow_pickle=True).item()
    real = torch.stack(list(d["m_ref_latents"])).float()
    fake = torch.stack(list(d["m_pred_latents"])).float()
    tids = [tid_of(f) for f in d["pred_filenames"]]
    prec, dens = per_sample_prdc(real, fake)

    rep = json.load(open(osp.join(a.run, "metrics.json")))
    print(f"reproduce check  precision {prec.mean():.4f} vs reported {rep['val/clatr/precision']}"
          f"   density {dens.mean():.4f} vs reported {rep['val/clatr/density']}")

    fsc = {}
    with open(osp.join(a.run, "preds_scores.csv")) as f:
        for r in csv.DictReader(f):
            fsc[tid_of(r["filename"])] = (float(r["captions/fscore"]),
                                          float(r["clatr/clatr_score"]),
                                          float(r["clatr/pred_ref_cosine"]))

    rows = []
    for i, t in enumerate(tids):
        fs, cs, prc = fsc.get(t, (np.nan, np.nan, np.nan))
        rows.append(dict(target=t, fscore=fs, precision=prec[i], density=dens[i],
                         clatr_score=cs, pred_ref_cosine=prc))
    with open(osp.join(a.out, "_prdc_per_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    fs = np.array([r["fscore"] for r in rows])
    print(f"caption fscore mean {np.nanmean(fs):.4f}  |  in-manifold {int(prec.sum())}/{len(prec)}"
          f"  |  density>0 {int((dens > 0).sum())}/{len(dens)}")
    for lo, hi in [(0.999, 1.01)]:
        m = (fs >= lo)
        print(f"fscore=={lo:.3f}+ : n={m.sum()}  precision {prec[m].mean():.4f}  density {dens[m].mean():.4f}")

    # ---- scatter: caption fscore vs density, precision 으로 색 ----
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, (y, ylab) in zip(axs, [(dens, "density (per-sample)"), (prec, "precision (per-sample, 0/1)")]):
        c = ["tab:red" if p == 0 else "tab:blue" for p in prec]
        ax.scatter(fs, y + (np.random.RandomState(0).rand(len(y)) * 0.02 if ylab.startswith("precision") else 0),
                   c=c, s=26, alpha=0.8)
        ax.set_xlabel("caption fscore (per-sample)"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    axs[0].set_title("blue = inside real manifold (precision=1), red = outside")
    axs[1].set_title("precision jittered for visibility")
    fig.suptitle(f"{osp.basename(a.run)} — caption fscore vs CLaTr-latent manifold membership", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(osp.join(a.out, "_fscore_vs_prdc.png"), dpi=125, bbox_inches="tight"); plt.close(fig)

    # ---- mismatch: fscore 높은데 manifold 밖 (precision=0, density 낮은 순) ----
    cand = [r for r in rows if r["precision"] == 0 and r["fscore"] >= 0.999]
    if len(cand) < a.n_show:
        cand = sorted([r for r in rows if r["fscore"] >= 0.999], key=lambda r: (r["precision"], r["density"]))
    cand = sorted(cand, key=lambda r: r["density"])[:a.n_show]
    print(f"mismatch (fscore>=0.999 & manifold 밖): {len(cand)} shown")

    nc = 4; nr = int(np.ceil(len(cand) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.4 * nr), squeeze=False)
    for ax, r in zip(np.ravel(axs), cand):
        g, p = anchored(a.run, r["target"])
        ax.plot(g[:, 0], g[:, 1], "-o", ms=2.5, lw=1.5, c="tab:blue", label="GT")
        ax.plot(p[:, 0], p[:, 1], "-x", ms=2.5, lw=1.3, c="tab:red", label="pred")
        ax.scatter(0, 0, c="k", s=55, marker="*", zorder=6)
        pres = "_".join(r["target"].split("__")[1].split("_")[-2:])
        ax.set_title(f"{pres}  f1={r['fscore']:.2f}  dens={r['density']:.2f}\n"
                     f"CLaTr={r['clatr_score']:.1f}  cos(pred,GT)={r['pred_ref_cosine']:.1f}", fontsize=8)
        ax.set_aspect("equal", "datalim"); ax.grid(alpha=0.25)
        ax.set_xlabel("X"); ax.set_ylabel("-Z"); ax.legend(fontsize=7)
    for ax in np.ravel(axs)[len(cand):]:
        ax.axis("off")
    fig.suptitle(f"{osp.basename(a.run)} — caption fscore is high (>=0.999) but the prediction falls "
                 f"OUTSIDE the real CLaTr-latent manifold (precision=0), sorted by density", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(osp.join(a.out, "_mismatch_topdown.png"), dpi=125, bbox_inches="tight"); plt.close(fig)
    print("out ->", a.out)


if __name__ == "__main__":
    main()
