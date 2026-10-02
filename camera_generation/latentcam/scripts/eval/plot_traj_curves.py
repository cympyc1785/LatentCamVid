"""이 코드가 답하는 질문: 궤적 하나에서 카메라 이동 속도·회전 속도·subject 화면 위치(framing)가 시간에 따라 GT 와 어떻게 다른가?

R113 / R123 / R131 에서 같은 그림(GT · VAE roundtrip · 모델 의 속도·framing 곡선)을 인라인 코드로 세 번 그렸다 → 승격 (R162).
지표 정의는 `traj_time_metrics.py` 와 같은 함수를 그대로 쓴다 (좌표·subject 투영이 갈리지 않게).

  trans speed (t) = |p(t+1) - p(t)| / reach            reach = GT 가 frame 0 에서 가장 멀리 간 거리 (traj_time_metrics 와 같음)
  rot speed   (t) = |ω_t| (deg/frame)                    ω_t = R_t^T R_{t+1} 의 회전각
  subject offset (t) = |uv(t)|                           화면 중앙 기준 subject 위치 (x, y 를 한 값으로, 화면 폭/높이 = 1)
  [R169] --jerk 면 열 둘 추가: jerk_t (t) = |Δ³p| / reach, jerk_r (t) = |Δ²ω| (deg/frame²) — traj_time_metrics.smooth_framing 의
         jerk_t / jerk_r 와 같은 정의의 프레임별 값 (창 t = 프레임 t..t+3, 평균내면 그 지표).

    python scripts/eval/plot_traj_curves.py --corpus_root <DynPose-LBM corpus> \
        --arm old=<eval_dir> --arm rep+lin=<eval_dir> --entries dynpose_<uuid>_<seg> ... --out curves.png
행 = entry, 열 = 세 지표. GT 는 첫 arm 의 *_transforms_ref.json (모든 arm 이 같은 GT 를 쓴다고 가정하고 확인한다).
"""
import argparse
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import traj_time_metrics as T  # noqa: E402


def curves(c2w, subj, intr, reach):
    p = c2w[:, :3, 3]
    ts = np.linalg.norm(np.diff(p, axis=0), axis=1) / reach
    om = T.rotvec_deg(np.einsum("nji,njk->nik", c2w[:-1, :3, :3], c2w[1:, :3, :3]))
    rs = np.linalg.norm(om, axis=1)
    jt = np.linalg.norm(np.diff(p, n=3, axis=0), axis=1) / reach
    jr = np.linalg.norm(np.diff(om, n=2, axis=0), axis=1)
    off = None
    if subj is not None:
        _, uv = T.smooth_framing(c2w, subj, *intr, reach)
        off = np.linalg.norm(uv, axis=1)
    return ts, rs, off, jt, jr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=<eval_dir>")
    ap.add_argument("--entries", nargs="+", required=True)
    ap.add_argument("--titles", nargs="*", default=None, help="entry 별 행 제목 (기본 entry 이름)")
    ap.add_argument("--corpus_root", required=True)
    ap.add_argument("--graph_root", default="/data1/cympyc1785/LatentCamVid/camera_generation/dataset/out_dynpose")
    ap.add_argument("--name_prefix", default="dynpose")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jerk", action="store_true", help="[R169] jerk_t · jerk_r 시간 곡선 열 추가")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [s.split("=", 1) for s in a.arm]
    nc = 5 if a.jerk else 3
    fig, ax = plt.subplots(len(a.entries), nc, figsize=(17 * nc / 3, 3.6 * len(a.entries)), squeeze=False)
    for r, n in enumerate(a.entries):
        ref = T.load(osp.join(arms[0][1], "test", f"{n}_transforms_ref.json"))
        reach = max(float(np.linalg.norm(ref[:, :3, 3] - ref[0, :3, 3], axis=1).max()), 1e-9)
        subj = T.entry_subject_world(a.corpus_root, a.graph_root, a.name_prefix, n)
        if subj is not None and len(subj) != len(ref):
            subj = None
        series = [("GT", ref, osp.join(arms[0][1], "test", f"{n}_transforms_ref.json"))]
        for label, d in arms:
            rf = T.load(osp.join(d, "test", f"{n}_transforms_ref.json"))
            assert np.allclose(rf, ref, atol=1e-4), f"{label}: GT 가 첫 arm 과 다르다 ({n})"
            series.append((label, T.load(osp.join(d, "test", f"{n}_transforms_pred.json")), osp.join(d, "test", f"{n}_transforms_pred.json")))
        for label, c2w, f in series:
            ts, rs, off, jt, jr = curves(c2w, subj, T.load_intr(f), reach)
            kw = dict(color="k", lw=2.2, zorder=5) if label == "GT" else dict(lw=1.2)
            ax[r, 0].plot(ts, label=label, **kw)
            ax[r, 1].plot(rs, label=label, **kw)
            if off is not None:
                ax[r, 2].plot(off, label=label, **kw)
            if a.jerk:
                ax[r, 3].plot(jt, label=f"{label} ({jt.mean():.2e})", **kw)
                ax[r, 4].plot(jr, label=f"{label} ({jr.mean():.3f})", **kw)
        title = (a.titles[r] if a.titles and r < len(a.titles) else n)
        yls = ("trans speed  |Δp| / reach", "rot speed  (deg / frame)", "subject screen offset  |uv|")
        if a.jerk:
            yls += ("trans jerk  |Δ³p| / reach", "rot jerk  |Δ²ω| (deg / frame²)")
        for c, yl in enumerate(yls):
            ax[r, c].set_title(f"{title} — {yl}", fontsize=10)
            ax[r, c].set_xlabel("frame")
            ax[r, c].grid(alpha=0.3)
        if subj is None:
            ax[r, 2].text(0.5, 0.5, "subject 없음", transform=ax[r, 2].transAxes, ha="center")
        ax[r, 0].legend(fontsize=8, ncol=2)
        if a.jerk:
            ax[r, 3].legend(fontsize=7, title="mean", title_fontsize=7)
            ax[r, 4].legend(fontsize=7, title="mean", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
