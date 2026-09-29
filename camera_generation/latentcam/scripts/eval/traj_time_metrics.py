"""궤적 후반이 GT 에서 더 벌어지는가 — 프레임별 오차 곡선과 시간축 지표 (ADE/FDE/구간별/drift/RPE).

기존 지표는 시간축을 안 본다: `loss_traj`(eval_testset) 는 49 프레임 x 채널 평균 MSE, `pos_rmse`/`rot`
(`scripts/render/compare_textonly.py`) 도 프레임 평균, CLaTr/caption 은 궤적 전체 한 값이다. 그런데
첫 카메라는 GT 로 고정되므로(frame 0 오차 = 0) 오차는 뒤로 갈수록 쌓이는데, 평균은 앞쪽의 작은 오차와
섞여 희석된다. 여기서는 오차를 시간의 함수로 두고 세 가지를 갈라 본다.

  ATE(t)   = |p_pred(t) - p_gt(t)| / reach            절대 위치 오차 (reach = GT 가 frame 0 에서 가장 멀리 간 거리)
  rot(t)   = geodesic(R_pred(t), R_gt(t))             절대 회전 오차 (deg)
  RPE_d(t) = |Δp_pred(t, t+d) - Δp_gt(t, t+d)| / reach   **상대** 이동 오차 — 두 카메라 각자의 t 시점 좌표계에서 본
             d 프레임 동안의 이동 차이. 국소 움직임이 틀렸는지(속도·방향)를 누적 drift 와 떼어 본다.

지표 (entry 별, 그리고 arm 평균):
  ADE / FDE          ATE 의 평균 / 마지막 프레임
  early / mid / late ATE 를 1/3 씩 나눈 구간 평균, late_early = late / early (early 가 0 에 가까우면 커진다 → late - early 도 같이)
  drift_slope        ATE(t) 를 t∈[0,1] 에 1차 회귀한 기울기 (reach 단위 / 궤적 전체)
  rot_early/late     회전 오차 구간 평균
  rpe1 / rpe8        d=1, 8 상대 이동 오차 평균 (+ 회전 rpe1_rot)

좌표: eval JSON 은 scene world 의 OpenGL c2w. 위치 오차·상대 오차는 축 규약과 무관하고(회전 비교도 양쪽 같은
규약이면 무관), 정렬(Umeyama)은 **하지 않는다** — 첫 카메라가 이미 GT 에 붙어 있으므로 정렬하면 그 자체가
정답을 일부 알려 주는 셈이다.

    python scripts/eval/traj_time_metrics.py --arm random=<eval_dir> --arm t250=<eval_dir2> \
        [--entries NAME ...] --out <dir>
출력: <out>/per_entry.csv, summary.json, curve.png (arm 별 평균 ATE(t)·rot(t)·RPE1(t))
"""
import argparse
import csv
import json
import os
import os.path as osp
from glob import glob

import numpy as np


def load(p):
    return np.array([f["transform_matrix"] for f in json.load(open(p))["frames"]], dtype=np.float64)


def geo_deg(Ra, Rb):
    R = np.einsum("nji,njk->nik", Ra, Rb)
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def rel_motion(P, d):
    """t 시점 카메라 좌표계에서 본 t->t+d 이동 (위치) 과 상대 회전."""
    inv = np.linalg.inv(P[:-d])
    M = inv @ P[d:]
    return M[:, :3, 3], M[:, :3, :3]


def entry_metrics(ref, prd):
    T = len(ref)
    reach = max(float(np.linalg.norm(ref[:, :3, 3] - ref[0, :3, 3], axis=1).max()), 1e-9)
    ate = np.linalg.norm(prd[:, :3, 3] - ref[:, :3, 3], axis=1) / reach
    rot = geo_deg(prd[:, :3, :3], ref[:, :3, :3])
    thirds = np.array_split(np.arange(T), 3)
    t01 = np.linspace(0, 1, T)
    out = {"reach": reach, "ADE": ate.mean(), "FDE": ate[-1],
           "early": ate[thirds[0]].mean(), "mid": ate[thirds[1]].mean(), "late": ate[thirds[2]].mean(),
           "drift_slope": float(np.polyfit(t01, ate, 1)[0]),
           "rot_mean": rot.mean(), "rot_early": rot[thirds[0]].mean(), "rot_late": rot[thirds[2]].mean()}
    out["late_minus_early"] = out["late"] - out["early"]
    out["late_early"] = out["late"] / max(out["early"], 1e-6)
    curves = {"ate": ate, "rot": rot}
    for d in (1, 8):
        tp, Rp = rel_motion(prd, d)
        tg, Rg = rel_motion(ref, d)
        e = np.linalg.norm(tp - tg, axis=1) / reach
        out[f"rpe{d}"] = e.mean()
        if d == 1:
            out["rpe1_rot"] = geo_deg(Rp, Rg).mean()
            curves["rpe1"] = np.r_[e, e[-1]]
    return {k: float(v) for k, v in out.items()}, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=<eval_dir> (test/ 아래 *_ref/_pred json)")
    ap.add_argument("--entries", nargs="*", default=None, help="data_name 목록 (기본: 첫 arm 의 test/ 전부)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_reach_frac", type=float, default=0.1,
                    help="GT reach 가 첫 arm 전체 reach 중앙값 x 이 값보다 작은 entry(정지·거의 정지 궤적)는 제외 — "
                         "reach 로 나누는 정규화가 터진다. 제외 수는 summary 의 n_static")
    a = ap.parse_args()
    arms = [s.split("=", 1) for s in a.arm]
    os.makedirs(a.out, exist_ok=True)
    names = a.entries or sorted(osp.basename(p)[:-len("_transforms_pred.json")]
                                for p in glob(osp.join(arms[0][1], "test", "*_transforms_pred.json")))
    reach0 = {}
    for n in names:
        rf = osp.join(arms[0][1], "test", f"{n}_transforms_ref.json")
        if osp.isfile(rf):
            r = load(rf)
            reach0[n] = float(np.linalg.norm(r[:, :3, 3] - r[0, :3, 3], axis=1).max())
    floor = a.min_reach_frac * float(np.median(list(reach0.values())))
    static = {n for n, v in reach0.items() if v < floor}
    names = [n for n in names if n in reach0 and n not in static]
    print(f"[reach] median {np.median(list(reach0.values())):.4f}, floor {floor:.4f} -> 정지 제외 {len(static)}, 남음 {len(names)}")
    rows, curves, keys = [], {}, None
    for label, d in arms:
        cs = []
        for n in names:
            pr, rf = osp.join(d, "test", f"{n}_transforms_pred.json"), osp.join(d, "test", f"{n}_transforms_ref.json")
            if not (osp.isfile(pr) and osp.isfile(rf)):
                continue
            m, c = entry_metrics(load(rf), load(pr))
            keys = keys or list(m)
            rows.append({"arm": label, "entry": n, **m})
            cs.append(c)
        curves[label] = {k: np.nanmean(np.stack([c[k] for c in cs]), 0) for k in cs[0]} if cs else {}
        print(f"[{label}] {len(cs)} entries", flush=True)
    with open(osp.join(a.out, "per_entry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "entry", *keys])
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for label, _ in arms:
        R = [r for r in rows if r["arm"] == label]
        summ[label] = {"n": len(R), "n_static": len(static),
                       **{k: float(np.mean([r[k] for r in R])) for k in keys if k != "reach"},
                       **{f"{k}_median": float(np.median([r[k] for r in R])) for k in ("ADE", "FDE", "late_early")}}
    json.dump(summ, open(osp.join(a.out, "summary.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for label, c in curves.items():
        if not c:
            continue
        for i, (k, yl) in enumerate((("ate", "ATE(t) / reach"), ("rot", "rot err (deg)"), ("rpe1", "RPE d=1 / reach"))):
            ax[i].plot(c[k], label=label)
            ax[i].set_xlabel("frame")
            ax[i].set_ylabel(yl)
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"mean over entries ({len(names)} names)")
    fig.tight_layout()
    fig.savefig(osp.join(a.out, "curve.png"), dpi=110)
    cols = ["n", "ADE", "ADE_median", "FDE", "FDE_median", "early", "mid", "late", "late_minus_early", "late_early_median", "drift_slope",
            "rot_early", "rot_late", "rpe1", "rpe8", "rpe1_rot"]
    print("arm | " + " | ".join(cols))
    for label, s in summ.items():
        print(label + " | " + " | ".join(f"{s[c]:.4f}" if c != "n" else str(s[c]) for c in cols))
    # 정지 궤적은 reach 가 없으니 world 단위 절대 오차로만 따로 낸다.
    if static:
        for label, d in arms:
            e = [np.linalg.norm(load(osp.join(d, "test", f"{n}_transforms_pred.json"))[:, :3, 3]
                                - load(osp.join(d, "test", f"{n}_transforms_ref.json"))[:, :3, 3], axis=1)
                 for n in sorted(static) if osp.isfile(osp.join(d, "test", f"{n}_transforms_pred.json"))]
            e = np.stack(e)
            print(f"[static {label}] n {len(e)}  pos err world early {e[:, :16].mean():.4f} late {e[:, -16:].mean():.4f}")


if __name__ == "__main__":
    main()
