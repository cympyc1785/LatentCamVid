"""avg_scale 로 정규화해 LagerNVS 를 돌렸을 때 왜 무너졌는지 -- context/target 배치를
LagerNVS 가 실제로 보는 '정규화된 좌표'에서 top-down 으로 그린다.

기존 `<seg>/topdown_context_gt.png` 는 raw world 단위라 "범위를 벗어났나"를 볼 수 없다.
여기서는 divisor D 로 나눈 뒤, LagerNVS 가 학습 때 보던 범위를 같이 겹쳐 그린다:

  실선 원 r = 0.7407 (= 1/1.35)  LagerNVS 자체 정규화에서 '가장 먼 context view' 가
                                 항상 정확히 놓이는 반경. context 는 이 원 안을 꽉 채운다.
  점선 원 r = 1.0                target 의 경험적 상한 (corpus 에서 m>1 은 5.66%).

패널 왼쪽 = avg_scale (무너진 설정), 오른쪽 = geo_lagernvs (정상 설정). 같은 장면을
같은 축 범위로 그리므로 두 정규화의 스케일 차이가 그대로 보인다.

좌표계: target 첫 프레임 s 의 카메라 좌표계로 anchor 후 X vs -Z 투영
        (기존 topdown_context_gt.png / compare_textonly_vs_worldtraj.py 와 동일한 관례).

사용:
  python scripts/render/viz_avgscale_context_topdown.py \
      --root results/avgscale_collapse --out results/avgscale_collapse/topdown_norm
"""
import argparse
import glob
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

R_CTX = 1.0 / 1.35   # 0.7407 -- LagerNVS 정규화에서 가장 먼 context 가 놓이는 반경


def load(seg_dir):
    d = torch.load(osp.join(seg_dir, "render_inputs.pt"), map_location="cpu", weights_only=False)
    ctx = d["ctx_c2w"].numpy().astype(np.float64)
    tgt = d["tgt_c2w"].numpy().astype(np.float64)
    return d, ctx, tgt


def anchored(ctx, tgt):
    """target 첫 프레임 s 기준으로 anchor. 반환은 (context centers, target centers)."""
    ref = np.linalg.inv(tgt[0])
    c = np.einsum("ij,njk->nik", ref, ctx)[:, :3, 3]
    t = np.einsum("ij,njk->nik", ref, tgt)[:, :3, 3]
    return c, t


def panel(ax, c, t, D, title, idxs, lim):
    c, t = c / D, t / D
    for r, ls, lab in ((R_CTX, "-", "context reach 0.7407"), (1.0, "--", "target bound 1.0")):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=ls, lw=1.2, ec="0.45"))
        ax.plot([], [], ls=ls, c="0.45", lw=1.2, label=lab)
    ax.plot(t[:, 0], -t[:, 2], "-", c="tab:blue", lw=1.6, ms=2.5, marker="o", label="GT target (49)")
    ax.scatter(c[:, 0], -c[:, 2], s=150, marker="*", c="magenta", ec="k", lw=.5,
               zorder=5, label=f"geo context ({len(c)})")
    ax.scatter([0], [0], s=180, marker="*", c="k", zorder=6, label="start (frame s)")
    if idxs is not None:
        for (x, z), i in zip(np.c_[c[:, 0], -c[:, 2]], idxs):
            ax.annotate(str(i), (x, z), fontsize=7, xytext=(3, 3), textcoords="offset points")
    cn, tn = np.linalg.norm(c, axis=1), np.linalg.norm(t, axis=1)
    ax.set_title(f"{title}\nD={D:.3f}  ch9={cn.max():.4f}  m={tn.max():.4f}", fontsize=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(alpha=.25)
    ax.set_xlabel("X / D"); ax.set_ylabel("-Z / D")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest",
                    default="/data1/cympyc1785/data/DL3DV/latent_cache/first_cam_included__geo_idxs.pt")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    man = {}
    if osp.exists(args.manifest):
        man = torch.load(args.manifest, map_location="cpu", weights_only=False)["idxs"]

    dirs = sorted(p for p in glob.glob(osp.join(args.root, "*"))
                  if osp.exists(osp.join(p, "render_inputs.pt")))
    rows = []
    for sd in dirs:
        d, ctx, tgt = load(sd)
        seg = d["seg"]
        c, t = anchored(ctx, tgt)
        D_avg = float(d["avg_scale"])
        # geo_lagernvs divisor 는 LagerNVS 자신의 것: 1.35 * max||c_ctx - c_ctx0||
        cc = ctx[:, :3, 3]
        D_lnv = 1.35 * float(np.linalg.norm(cc - cc[0], axis=1).max())
        idxs = man.get(seg)
        idxs = idxs.tolist() if idxs is not None else None

        lim = max(np.abs(np.r_[c, t] / min(D_avg, D_lnv)).max() * 1.1, 1.15)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
        panel(axes[0], c, t, D_avg, f"avg_scale (collapsed) - {seg[:14]}", idxs, lim)
        panel(axes[1], c, t, D_lnv, f"geo_lagernvs (healthy) - {seg[:14]}", idxs, lim)
        axes[1].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(osp.join(args.out, f"{seg}.png"), dpi=130)
        plt.close(fig)

        cn_a = np.linalg.norm(c / D_avg, axis=1); tn_a = np.linalg.norm(t / D_avg, axis=1)
        cn_l = np.linalg.norm(c / D_lnv, axis=1); tn_l = np.linalg.norm(t / D_lnv, axis=1)
        rows.append(dict(seg=seg, geo_idxs=idxs, D_avg=D_avg, D_lagernvs=D_lnv,
                         ch9_avg=float(cn_a.max()), m_avg=float(tn_a.max()),
                         ch9_lagernvs=float(cn_l.max()), m_lagernvs=float(tn_l.max()),
                         ctx_norms_avg=[round(float(v), 4) for v in cn_a],
                         tgt_med_avg=float(np.median(tn_a))))

    with open(osp.join(args.out, "ranges.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # 요약: seg 별 ch9 / m 을 두 정규화로 나란히
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(rows))
    ax.bar(x - .21, [r["ch9_avg"] for r in rows], .2, label="ch9 avg_scale", color="tab:red")
    ax.bar(x - .01, [r["ch9_lagernvs"] for r in rows], .2, label="ch9 geo_lagernvs", color="tab:green")
    ax.bar(x + .19, [r["m_avg"] for r in rows], .2, label="m (target) avg_scale", color="tab:orange")
    ax.bar(x + .39, [r["m_lagernvs"] for r in rows], .2, label="m (target) geo_lagernvs", color="tab:blue")
    ax.axhline(R_CTX, ls="-", c="0.35", lw=1.2)
    ax.annotate("0.7407", (len(rows) - .5, R_CTX), fontsize=8, va="bottom", ha="right", color="0.35")
    ax.axhline(1.0, ls="--", c="0.35", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([r["seg"][3:11] + r["seg"][-2:] for r in rows],
                                         rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("normalized reach"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25, axis="y")
    ax.set_title("avg_scale over-shrinks the scene inside the 0.7407 circle (baseline lost)")
    fig.tight_layout(); fig.savefig(osp.join(args.out, "_summary.png"), dpi=130); plt.close(fig)

    print(f"wrote {len(rows)} figures + _summary.png + ranges.json -> {args.out}")


if __name__ == "__main__":
    main()
