"""Divisor stability ACROSS CHUNKS, as a function of camera motion (straightness / speed).

The camera-generation model only has to keep structure and SCALE consistent when the chunk
advances -- image quality can be recovered later by re-rendering with denser context. So the
figure of merit for chaining chunks is how much the normalization divisor D moves between
consecutive segments of the SAME scene:

    spread = (max_k D_k - min_k D_k) / mean_k D_k     over chunks k = 0,1,2

computed for both arms:
    ctx_longer_135max (arm A, retrieval-INDEPENDENT: mean over the context range's 49-frame windows)
    geo_lagernvs      (arm B, retrieval-DEPENDENT:   1.35*max||retrieved ctx center - frame s||)

Motion is described per segment from the same CSV:
    straight = maxd / pathlen                  1.0 = perfectly straight, low = orbiting/curvy
    speed    = pathlen / ctx_longer_135max     path length per 49 frames in scene-scale units
               (dividing by a scene-scale divisor is what makes speed comparable across scenes --
                raw COLMAP units are arbitrary per scene)

Input: results/compare/<name>/per_segment.csv written by scripts/data/norm_divisor_compare.py.
Output: results/compare/<name>/chunk_stability_{summary.md,.png,per_scene.csv}

  /data1/cympyc1785/miniconda3/envs/latentcam/bin/python scripts/data/chunk_divisor_stability.py \
      [--csv results/compare/norm_divisor_compare_postbl/per_segment.csv] [--chunks 3]
"""
import argparse, os.path as osp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = {"ctx_longer_135max": "A ctx_longer_135max", "geo_lagernvs": "B geo_lagernvs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/"
                                     "results/compare/norm_divisor_compare_postbl/per_segment.csv")
    ap.add_argument("--chunks", type=int, default=3, help="require chunks 0..N-1 to exist")
    args = ap.parse_args()
    out = osp.dirname(args.csv)

    df = pd.read_csv(args.csv)
    df["scene"] = df.seg.str.replace(r"_\d+$", "", regex=True)
    df["k"] = df.seg.str.extract(r"_(\d+)$").astype(int)
    df["straight"] = df.maxd / df.pathlen
    df["speed"] = df.pathlen / df.ctx_longer_135max

    K = set(range(args.chunks))
    g = df[df.k < args.chunks].groupby("scene").filter(lambda x: set(x.k) == K).copy()

    def spread(col):
        p = g.pivot(index="scene", columns="k", values=col)
        return (p.max(1) - p.min(1)) / p.mean(1)

    S = pd.DataFrame({"straight": g.groupby("scene").straight.mean(),
                      "speed": g.groupby("scene").speed.mean(),
                      **{f"spread_{a}": spread(a) for a in ARMS}})
    S.to_csv(osp.join(out, "chunk_stability_per_scene.csv"))

    lines = [f"# chunk 간 divisor 안정성 ({len(S)} scene, chunk 0..{args.chunks-1} 전부 있는 scene)", "",
             f"spread = (max_k D_k - min_k D_k) / mean_k D_k", ""]
    lines.append("| | median spread A | median spread B |")
    lines.append("|---|---|---|")
    lines.append(f"| 전체 | {S.spread_ctx_longer_135max.median():.4f} | {S.spread_geo_lagernvs.median():.4f} |")
    lines.append("")

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, col, name in [(axs[0], "speed", "speed = pathlen / D_A  (path length per 49 frames, scene scale)"),
                          (axs[1], "straight", "straight = maxd / pathlen  (1 = straight line)")]:
        S["_b"] = pd.qcut(S[col], 5, duplicates="drop")
        t = S.groupby("_b", observed=True).agg(n=("speed", "size"),
                                               A=("spread_ctx_longer_135max", "median"),
                                               B=("spread_geo_lagernvs", "median"),
                                               x=(col, "median"))
        ax.plot(t.x, t.A, "-o", label=ARMS["ctx_longer_135max"])
        ax.plot(t.x, t.B, "-s", label=ARMS["geo_lagernvs"])
        ax.set_xlabel(name, fontsize=8); ax.set_ylabel("median chunk-to-chunk D spread"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8); ax.set_title(f"{col} quintiles", fontsize=9)
        lines += [f"## {col} 5분위", "", "| bin | n | median speed | median straight | A | B |",
                  "|---|---|---|---|---|---|"]
        for b, r in t.iterrows():
            ms = S[S["_b"] == b].straight.median()
            sp = S[S["_b"] == b].speed.median()
            lines.append(f"| {b} | {int(r.n)} | {sp:.3f} | {ms:.3f} | {r.A:.4f} | {r.B:.4f} |")
        lines.append("")

    S["sb"] = pd.qcut(S.speed, 3, labels=["slow", "mid", "fast"])
    S["tb"] = pd.qcut(S.straight, 3, labels=["curvy", "mid", "straight"])
    cross = {a: S.pivot_table(index="tb", columns="sb", values=f"spread_{a}",
                              aggfunc="median", observed=True) for a in ARMS}
    im = axs[2].imshow(cross["ctx_longer_135max"].values, cmap="viridis")
    axs[2].set_xticks(range(3), cross["ctx_longer_135max"].columns)
    axs[2].set_yticks(range(3), cross["ctx_longer_135max"].index)
    for i in range(3):
        for j in range(3):
            axs[2].text(j, i, f"A {cross['ctx_longer_135max'].values[i, j]:.3f}\n"
                              f"B {cross['geo_lagernvs'].values[i, j]:.3f}",
                        ha="center", va="center", color="w", fontsize=9)
    axs[2].set_title("straightness x speed  (color = arm A)", fontsize=9)
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    fig.suptitle("how much the normalization divisor moves when the chunk advances", fontsize=11)
    fig.tight_layout()
    fig.savefig(osp.join(out, "chunk_stability.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    lines += ["## 직선성 x 속도 교차 (median spread)", "", "| | slow | mid | fast |", "|---|---|---|---|"]
    for i, t in enumerate(cross["ctx_longer_135max"].index):
        lines.append(f"| {t} | " + " | ".join(
            f"A {cross['ctx_longer_135max'].values[i, j]:.3f} / B {cross['geo_lagernvs'].values[i, j]:.3f}"
            for j in range(3)) + " |")
    open(osp.join(out, "chunk_stability_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}/chunk_stability.png")


if __name__ == "__main__":
    main()
