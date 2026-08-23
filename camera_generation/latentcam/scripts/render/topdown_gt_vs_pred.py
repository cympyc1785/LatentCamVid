"""val 덤프(_transforms_ref/_transforms_pred.json)로 GT vs pred 카메라를 top-down 비교한다.

WHY: caption precision/recall/f1 이 학습 내내 0 <-> 0.1 을 오갈 뿐 추세가 없는데 loss_traj 는
6배 떨어졌다. 지표 숫자만 봐서는 "궤적이 아직 틀린 것"과 "궤적은 맞는데 지표가 못 읽는 것"을
못 가른다. 그래서 caption 지표가 **실제로 소비하는 양**을 궤적 그림 옆에 같이 그린다:

  panel 1  top-down  (X vs -Z, GT frame0 앵커)   — 궤적 모양
  panel 2  side      (X vs Y)                    — 상하 이동
  panel 3  per-frame local Δt (metric 이 쓰는 그 양) + static 임계선
           caption.py 는 velocities = inv(w2c[:-1]) @ w2c[1:] 의 t 성분에 fps=5 를 곱하고
           |5·Δt| < cam_static_threshold(0.02) 인 축을 "정지"로 찍는다. 축 3개의 정지/양/음
           조합 27가지 × yaw 7가지 = 189 클래스 **완전일치**라 임계 근처에서 한 축만 뒤집혀도
           그 프레임 점수가 0 이 된다. 그래서 임계선을 궤적과 같은 화면에 둔다.
  하단     preds_pcf.csv 의 pred_segments / target_segments 문자열 (실제로 뭘 맞히고 뭘 틀렸나)

좌표: 덤프된 transform_matrix 는 OpenGL c2w (train_latent_cam_dm.py 가 w2c -> c2w -> [:,:3,1:3]*=-1).
      caption metric 은 w2c 를 받으므로 panel 3 은 inv(c2w) 로 되돌려서 metric 과 같은 식을 쓴다.

사용 예:
  python scripts/render/topdown_gt_vs_pred.py \
    --runs vls019=results/20260823_212608_trumans_lite_ctxuniform_vls019 \
           vls039=results/20260823_212610_trumans_lite_ctxuniform_vls039 \
    --out results/compare/trumans_topdown --max_targets 14
"""
from argparse import ArgumentParser
from os import path, makedirs
import ast
import csv
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FPS = 5                       # caption.py:406 과 같은 값
STATIC_TH = 0.02              # caption.py cam_static_threshold. 5*|dt| 를 이 값과 비교한다
COLORS = ["tab:orange", "tab:green", "tab:red", "tab:purple"]


def load_c2w(p):
    """덤프 JSON -> (N,4,4) OpenGL c2w."""
    return np.array([f["transform_matrix"] for f in json.load(open(p))["frames"]], float)


def local_dt(c2w):
    """caption metric 이 보는 per-frame local translation 5*Δt (N-1, 3).

    metric 은 w2c 를 받아 velocities = inv(w2c[:-1]) @ w2c[1:] 를 만든다. 여기서는 c2w 를
    되돌려 같은 식을 쓴다 (OpenGL/OpenCV flip 은 축 부호만 바꾸므로 |Δt| 크기 비교엔 무관하지만,
    부호까지 metric 과 맞추려고 flip 도 그대로 적용한다).
    """
    gl = c2w.copy()
    gl[:, :3, 1:3] *= -1                      # OpenGL c2w -> OpenCV c2w
    w2c = np.linalg.inv(gl)
    vel = np.linalg.inv(w2c[:-1]) @ w2c[1:]
    return FPS * vel[:, :3, 3]


def anchor(c2w, R0inv):
    """GT frame0 기준으로 재앵커한 카메라 중심 (N,3)."""
    h = np.concatenate([c2w[:, :3, 3], np.ones((len(c2w), 1))], 1)
    return (h @ R0inv.T)[:, :3]


def path_stats(centers):
    seg = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    plen = float(seg.sum())
    net = float(np.linalg.norm(centers[-1] - centers[0]))
    return plen, net, (plen / net if net > 1e-9 else np.inf)


def read_pcf(run_dir):
    """preds_pcf.csv -> {target_id: (precision, recall, f1, pred_segments, target_segments)}."""
    p = path.join(run_dir, "preds_pcf.csv")
    if not path.exists(p):
        return {}
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            tid = path.basename(row["filename"]).replace("_transforms_ref.", "")
            try:
                ps = ast.literal_eval(row["pred_segments"])
                ts = ast.literal_eval(row["target_segments"])
            except Exception:
                ps, ts = [], []
            out[tid] = (float(row["captions/precision"]), float(row["captions/recall"]),
                        float(row["captions/f1"]), ps, ts)
    return out


def seg_text(segs, n=49):
    """세그먼트 리스트를 '클래스 xK' 런렝스 문자열로 줄인다 (49줄을 다 찍으면 못 읽는다)."""
    if not segs:
        return "(none)"
    runs, cur, cnt = [], segs[0], 1
    for s in segs[1:]:
        if s == cur:
            cnt += 1
        else:
            runs.append(f"{cur} x{cnt}")
            cur, cnt = s, 1
    runs.append(f"{cur} x{cnt}")
    return "  |  ".join(runs)


def main():
    ap = ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=result_dir 쌍. 첫 run 의 _transforms_ref 가 공통 GT 가 된다")
    ap.add_argument("--out", required=True)                 # 출력 디렉토리
    ap.add_argument("--max_targets", type=int, default=0)   # 0 = 전부
    ap.add_argument("--root", default="/data1/cympyc1785/LatentCamVid/camera_generation/latentcam")
    args = ap.parse_args()

    runs = []
    for a in args.runs:
        name, d = a.split("=", 1)
        runs.append((name, d if path.isabs(d) else path.join(args.root, d)))
    makedirs(args.out, exist_ok=True)

    base = runs[0][1]
    ids = sorted({path.basename(f)[:-len("_transforms_ref.json")]
                  for f in glob.glob(path.join(base, "test", "*_transforms_ref.json"))})
    if args.max_targets:
        ids = ids[:args.max_targets]
    pcf = {n: read_pcf(d) for n, d in runs}
    print(f"{len(ids)} targets x {len(runs)} runs -> {args.out}")

    rows = []
    for tid in ids:
        ref = load_c2w(path.join(base, "test", f"{tid}_transforms_ref.json"))
        R0inv = np.linalg.inv(ref[0])
        gc = anchor(ref, R0inv)
        gdt = local_dt(ref)
        gpl, gnet, gtor = path_stats(gc)

        fig = plt.figure(figsize=(17, 9.5))
        gs = fig.add_gridspec(3, 3, height_ratios=[3.0, 1.5, 1.1], hspace=0.42, wspace=0.24)
        ax_td = fig.add_subplot(gs[0, 0])
        ax_sd = fig.add_subplot(gs[0, 1])
        ax_sp = fig.add_subplot(gs[0, 2])
        ax_dt = [fig.add_subplot(gs[1, i]) for i in range(3)]
        ax_tx = fig.add_subplot(gs[2, :]); ax_tx.axis("off")

        ax_td.plot(gc[:, 0], -gc[:, 2], "-o", ms=3, lw=2.0, c="tab:blue", label="GT", zorder=5)
        ax_sd.plot(gc[:, 0], gc[:, 1], "-o", ms=3, lw=2.0, c="tab:blue", label="GT", zorder=5)
        ax_sp.plot(np.linalg.norm(np.diff(gc, axis=0), axis=1), "-", lw=2.0, c="tab:blue", label="GT")
        for k in range(3):
            ax_dt[k].plot(gdt[:, k], "-", lw=2.0, c="tab:blue", label="GT")

        lines = [f"GT      path_len {gpl:.3f}  net {gnet:.3f}  tortuosity {gtor:.2f}   "
                 f"|5dt| med (x,y,z) [{np.median(np.abs(gdt),0)[0]:.4f} "
                 f"{np.median(np.abs(gdt),0)[1]:.4f} {np.median(np.abs(gdt),0)[2]:.4f}]  "
                 f"static축비율 {float((np.abs(gdt) < STATIC_TH).mean()):.2f}"]
        row = dict(target=tid, gt_path=gpl, gt_net=gnet, gt_tor=gtor)

        for j, (name, d) in enumerate(runs):
            fp = path.join(d, "test", f"{tid}_transforms_pred.json")
            if not path.exists(fp):
                continue
            pred = load_c2w(fp)
            pc = anchor(pred, R0inv)
            pdt = local_dt(pred)
            ppl, pnet, ptor = path_stats(pc)
            pr, rc, f1, ps, ts = pcf[name].get(tid, (np.nan,) * 3 + ([], []))
            c = COLORS[j % len(COLORS)]
            lab = f"{name} (f1 {f1:.3f})"
            ax_td.plot(pc[:, 0], -pc[:, 2], "-x", ms=3, lw=1.3, c=c, label=lab)
            ax_sd.plot(pc[:, 0], pc[:, 1], "-x", ms=3, lw=1.3, c=c, label=lab)
            ax_sp.plot(np.linalg.norm(np.diff(pc, axis=0), axis=1), "-", lw=1.2, c=c, label=lab)
            for k in range(3):
                ax_dt[k].plot(pdt[:, k], "-", lw=1.2, c=c, label=name)
            spd = np.median(np.abs(pdt), 0) / np.maximum(np.median(np.abs(gdt), 0), 1e-9)
            lines.append(f"{name:7s} path_len {ppl:.3f}  net {pnet:.3f}  tortuosity {ptor:.2f}   "
                         f"|5dt| med [{np.median(np.abs(pdt),0)[0]:.4f} {np.median(np.abs(pdt),0)[1]:.4f} "
                         f"{np.median(np.abs(pdt),0)[2]:.4f}]  static축비율 "
                         f"{float((np.abs(pdt) < STATIC_TH).mean()):.2f}   "
                         f"pred/GT 속도비 [{spd[0]:.1f} {spd[1]:.1f} {spd[2]:.1f}]   "
                         f"P {pr:.3f} R {rc:.3f} F1 {f1:.3f}")
            row[f"{name}_f1"] = f1
            row[f"{name}_tor"] = ptor
            row[f"{name}_pathratio"] = ppl / gpl if gpl > 1e-9 else np.nan
            if j == 0:
                lines.append(f"        GT  seg : {seg_text(ts)[:230]}")
                lines.append(f"        {name:3s} seg : {seg_text(ps)[:230]}")

        ax_td.set_title("top-down (X vs -Z)", fontsize=10)
        ax_td.set_xlabel("X"); ax_td.set_ylabel("-Z"); ax_td.set_aspect("equal", "datalim")
        ax_sd.set_title("side (X vs Y)", fontsize=10)
        ax_sd.set_xlabel("X"); ax_sd.set_ylabel("Y"); ax_sd.set_aspect("equal", "datalim")
        ax_sp.set_title("per-frame |Δcenter| (world)", fontsize=10); ax_sp.set_xlabel("frame")
        for a in (ax_td, ax_sd, ax_sp):
            a.grid(alpha=0.25); a.legend(fontsize=7)
        ax_td.scatter(gc[0, 0], -gc[0, 2], c="k", s=90, marker="*", zorder=9)
        for k, nm in enumerate("xyz"):
            ax_dt[k].axhspan(-STATIC_TH, STATIC_TH, color="k", alpha=0.10, zorder=0)
            ax_dt[k].axhline(0, c="k", lw=0.6)
            ax_dt[k].set_title(f"caption 입력 5·Δt_local  {nm}축  (회색띠 = static 판정 구간 ±{STATIC_TH})",
                               fontsize=8.5)
            ax_dt[k].set_xlabel("frame"); ax_dt[k].grid(alpha=0.25)
            ax_dt[k].legend(fontsize=7)
        ax_tx.text(0, 1, "\n".join(lines), family="monospace", fontsize=7.6, va="top")
        fig.suptitle(tid, fontsize=11)
        fig.savefig(path.join(args.out, f"{tid}.png"), dpi=105, bbox_inches="tight")
        plt.close(fig)
        rows.append(row)

    keys = sorted({k for r in rows for k in r})
    with open(path.join(args.out, "_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target"] + [k for k in keys if k != "target"])
        w.writeheader(); w.writerows(rows)

    print(f"\n{'target':>10} {'gt_tor':>7} " +
          " ".join(f"{n+'_tor':>11} {n+'_f1':>10} {n+'_plen/gt':>11}" for n, _ in runs))
    for r in rows:
        s = f"{r['target'][-4:]:>10} {r['gt_tor']:7.2f} "
        for n, _ in runs:
            s += f"{r.get(f'{n}_tor', float('nan')):11.2f} {r.get(f'{n}_f1', float('nan')):10.3f} " \
                 f"{r.get(f'{n}_pathratio', float('nan')):11.2f}"
        print(s)
    print(f"\nsaved {len(rows)} panels + _summary.csv -> {args.out}")


if __name__ == "__main__":
    main()
