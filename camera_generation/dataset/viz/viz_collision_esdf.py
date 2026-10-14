"""이 코드가 답하는 질문: ESDF voxel 충돌 판정 (occupied / free / unknown) 이 씬에서 실제로 어떻게 생겼고 궤적 프레임이 어디서 걸리나?

R374 (사용자 2026-10-14 "시각화도 보여줘"). `eval/pred_collision_rate.py` 의 `esdf_scene(..., keep=)` 를 그대로 불러 격자를 받고,
씬마다 한 행: ① 소스 첫 프레임 ② 위에서 본 단면 (소스 첫 카메라 기준 right × forward, 궤적 평균 높이 ±1.5 voxel 판)
— 검정 occupied · 하늘색 free · 회색 unknown, 소스 카메라 경로 (검정 선), 궤적 점 색 = 프레임 판정 (빨강 충돌 · 파랑 free · 주황 unknown)
③ 옆에서 본 단면 (forward × up, 궤적 평균 right ±1.5 voxel) ④ arm 별 49 프레임 판정 띠.
env vista4d.  예) python viz/viz_collision_esdf.py --arm GT --arm MTb=<eval_dir> --entries a,b,c --out x.png
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np

HERE = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, HERE)
import eval.pred_collision_rate as M  # noqa: E402

COL = {"coll": "#d62728", "free": "#1f77b4", "unk": "#ff7f0e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)        # label=<eval_dir> · "GT" 는 첫 arm 의 ref
    ap.add_argument("--entries", required=True)                      # 쉼표 구분 data_name
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200")
    ap.add_argument("--vox_frac", type=float, default=0.02)
    ap.add_argument("--radius_frac", type=float, default=0.02)
    ap.add_argument("--max_dim", type=int, default=256)
    ap.add_argument("--pix_stride", type=int, default=2)
    ap.add_argument("--src_free", type=int, default=1)
    a = ap.parse_args()
    M.ARGS = a
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from scene_graph.io import import_vista4d
    v4 = import_vista4d("/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    arms = [s.split("=", 1) if "=" in s else [s, None] for s in a.arm]
    first = next(d for _, d in arms if d)
    ents = a.entries.split(",")
    fig, axs = plt.subplots(len(ents), 4, figsize=(22, 5.2 * len(ents)), gridspec_kw={"width_ratios": [1.3, 1, 1, 1.1]})
    axs = np.atleast_2d(axs)
    for row, name in enumerate(ents):
        v = name[len("dynpose_"):].rsplit("_", 1)[0]
        rec = v4["load_recon_and_seg"](f"{M.EVAL_DATA}/eval_data/recon_and_seg/{v}")
        S = float(json.load(open(f"{M.GRAPH}/{v}/scene_graph.json"))["scale"]["S"])
        K = np.asarray(rec["intrinsics"], float)
        if K.ndim == 2:
            K = np.stack([np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]) for k in K])
        tr = [(lab, name, M.load_c2w(f"{first}/test/{name}_transforms_ref.json" if d is None else f"{d}/test/{name}_transforms_pred.json")) for lab, d in arms]
        keep = {}
        res = M.esdf_scene(rec, K, S, tr, keep=keep)
        Cw = np.asarray(rec["cam_c2w"], float); B = np.stack([Cw[0][:3, 0], Cw[0][:3, 2], -Cw[0][:3, 1]], 1)   # right, fwd, up
        o = Cw[0][:3, 3]; to = lambda X: (X - o) @ B
        occ, free, lo, vox = keep["occ"], keep["free"], keep["lo"], keep["vox"]
        idx = np.argwhere(np.ones(occ.shape, bool)); ctr = to(idx * vox + lo + vox / 2)
        st = np.where(occ.reshape(-1), 2, np.where(free.reshape(-1), 1, 0))
        allc = to(np.concatenate([p[:, :3, 3] for _, _, p in tr]))
        cap = (rec["cam_c2w"].shape[0], )
        axs[row, 0].imshow(Image.open(f"{a.corpus}/dynpose/{v}/images_4/00000.png")); axs[row, 0].axis("off")
        txt = " · ".join(f"{r['arm']} coll {r['coll_frac']:.2f} unk {r['unk_frac']:.2f}" for r in res)
        axs[row, 0].set_title(f"{name[8:16]}…_{name.rsplit('_', 1)[1]}  vox {vox / S:.3f}·S\n{txt}", fontsize=9)
        for col, (hx, vy, hz, lab) in enumerate([(0, 1, 2, "top (right × fwd)"), (1, 2, 0, "side (fwd × up)")], start=1):
            ax = ax_ = axs[row, col]; h0 = allc[:, hz].mean()
            sl = np.abs(ctr[:, hz] - h0) <= 1.5 * vox
            for s_, c_ in ((0, "#dddddd"), (1, "#a6d8ff"), (2, "#222222")):
                m = sl & (st == s_)
                ax.scatter(ctr[m, hx], ctr[m, vy], s=4, c=c_, marker="s", linewidths=0)
            sc = to(Cw[:, :3, 3]); ax.plot(sc[:, hx], sc[:, vy], "k-", lw=2, label="source cam")
            for (lab_, _, p), mk in zip(tr, "o^sDv*"):
                c3 = to(p[:, :3, 3]); cls = keep["cls"][(lab_, name)]
                ax.plot(c3[:, hx], c3[:, vy], "-", c="#555555", lw=0.8)
                ax.scatter(c3[:, hx], c3[:, vy], s=22, marker=mk, c=[COL[c] for c in cls], edgecolors="k", linewidths=0.3, label=lab_, zorder=5)
            span = np.ptp(allc[:, [hx, vy]], 0).max() * 0.8 + 10 * vox; cx = allc[:, [hx, vy]].mean(0)
            ax.set_xlim(cx[0] - span, cx[0] + span); ax.set_ylim(cx[1] - span, cx[1] + span); ax.set_aspect("equal")
            ax.set_title(f"{lab} slab ±1.5 vox", fontsize=9)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc="upper right")
        ax = axs[row, 3]
        for i, (lab_, _, _) in enumerate(tr):
            cls = keep["cls"][(lab_, name)]
            ax.bar(np.arange(len(cls)), 0.8, bottom=i, color=[COL[c] for c in cls], width=1.0)
        ax.set_yticks(np.arange(len(tr)) + 0.4); ax.set_yticklabels([t[0] for t in tr]); ax.set_xlabel("frame")
        ax.set_title("per-frame: red=collision blue=free orange=unknown", fontsize=9)
        print(name, txt, flush=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=80); print("->", a.out)


if __name__ == "__main__":
    main()
