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

[R112] **궤적이 매끄러운가 / framing 이 일정한가** (사용자 "궤적이 smooth하지 못하고 framing이 일정하지 않은게 문제").
`--corpus_root` 를 주면 아래가 추가된다 (주지 않으면 위 지표만 = 예전 출력). 전부 렌더 없이 카메라 + subject 3D 중심만 쓴다.
  smoothness (GT 와 pred 각각, 끝에 _gt/_ratio):
    jerk_t     |Δ³p| 평균 / reach            위치 떨림 (3차 차분)
    jerk_r     |Δ²ω| 평균 (deg/frame²)       회전 떨림. ω_t = R_t^T R_{t+1} 의 회전각 벡터 (카메라 좌표계)
    [R145] jerk_t_late / jerk_r_late / fr_jit_late / fr_err_late : 같은 지표를 마지막 1/3 구간 (late 와 같은 프레임) 에서만.
           마지막 구간 위치·회전 차이는 기존 late (ATE) / rot_late / FDE. 속도 차이는 rpe1_late (이동) / rpe1_rot_late (회전).
  framing (subject = prompts.json variant_id 앞머리 노드의 scene_graph track.center_smooth → world → 카메라 투영):
    fr_err     |uv_pred - uv_gt| 평균         화면상 subject 위치가 GT 구도와 얼마나 다른가 (화면 폭=1)
    fr_jit     |Δ² uv| 평균 x100              화면상 subject 가 프레임마다 흔들리는 정도 (%/frame²)
    fr_std     uv 의 시간 표준편차              구도가 한 자리에 머무는가 (GT 가 의도적으로 움직이면 _ratio 로 본다)
    sz_jit     |Δ² log z| 평균 x100           subject 까지 거리(=화면 크기) 떨림
    in_box     중심이 화면 중앙 0.8 상자 안 + 카메라 앞인 프레임 비율 (SiF 의 렌더 없는 근사)
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


GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def rotvec_deg(R):
    """(n,3,3) -> 회전 벡터 (n,3) deg."""
    ang = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))
    ax = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    n = np.linalg.norm(ax, axis=1, keepdims=True)
    return np.where(n > 1e-9, ax / np.maximum(n, 1e-9), 0) * np.degrees(ang)[:, None]


def smooth_framing(c2w_gl, subj, fx, fy, w, h, reach):
    """카메라 궤적 하나 -> smoothness·framing 지표 dict + uv (T,2).
    [R145] *_late = 마지막 1/3 구간 (entry_metrics 의 late 와 같은 프레임; 차분 창이 그 구간 안에 다 들어가는 것만)."""
    p = c2w_gl[:, :3, 3]
    T = len(p)
    t0 = np.array_split(np.arange(T), 3)[2][0]                              # 마지막 1/3 의 첫 프레임 (T=49 → 33)
    jt_a = np.linalg.norm(np.diff(p, n=3, axis=0), axis=1) / reach          # 창 i = 프레임 i..i+3
    om = rotvec_deg(np.einsum("nji,njk->nik", c2w_gl[:-1, :3, :3], c2w_gl[1:, :3, :3]))
    jr_a = np.linalg.norm(np.diff(om, n=2, axis=0), axis=1)                 # 창 i = 프레임 i..i+3
    jt, jr = jt_a.mean(), jr_a.mean()
    out = {"jerk_t": jt, "jerk_r": jr, "jerk_t_late": jt_a[t0:].mean(), "jerk_r_late": jr_a[t0:].mean()}
    if subj is None:
        return out, None
    cv = c2w_gl @ GL2CV
    w2c = np.linalg.inv(cv)
    pc = np.einsum("nij,nj->ni", w2c[:, :3, :3], subj) + w2c[:, :3, 3]
    z = pc[:, 2]
    zc = np.maximum(z, 1e-6)
    uv = np.stack([fx * pc[:, 0] / zc / w, fy * pc[:, 1] / zc / h], 1)      # 화면 중앙 기준, 폭/높이 = 1
    inb = (z > 0) & (np.abs(uv[:, 0]) < 0.4) & (np.abs(uv[:, 1]) < 0.4)
    uvc = np.clip(uv, -1.5, 1.5)                                              # 뒤로 돈 프레임의 폭주 방지
    fj_a = np.linalg.norm(np.diff(uvc, n=2, axis=0), axis=1) * 100          # 창 i = 프레임 i..i+2
    out.update({"fr_jit": fj_a.mean(), "fr_jit_late": fj_a[t0:].mean(),
                "fr_std": float(np.linalg.norm(uvc.std(0))),
                "sz_jit": np.abs(np.diff(np.log(zc), n=2)).mean() * 100,
                "in_box": inb.mean()})
    return out, uvc


def load_intr(p):
    d = json.load(open(p))
    return float(d["fl_x"]), float(d["fl_y"]), float(d["w"]), float(d["h"])


def entry_subject_world(corpus_root, graph_root, prefix, name):
    """data_name -> subject 중심 (T,3) world. 못 찾으면 None."""
    body = name[len(prefix) + 1:]
    video, idx = body.rsplit("_", 1)
    pp = osp.join(corpus_root, prefix, video, "da3", "prompts.json")
    gp = osp.join(graph_root, video, "scene_graph.json")
    if not (osp.isfile(pp) and osp.isfile(gp)):
        return None
    vid = json.load(open(pp)).get(idx, {}).get("variant_id", "")
    node_id = vid.split("__")[0] if "__" in vid else ""
    G = json.load(open(gp))
    node = next((n for n in G["nodes"] if n["id"] == node_id), None)
    if node is None or not node.get("track", {}).get("center_smooth"):
        return None
    Twg = np.array(G["frames"]["T_wg"], dtype=np.float64)
    c = np.asarray(node["track"]["center_smooth"], dtype=np.float64)
    return (Twg[:3, :3] @ c.T).T + Twg[:3, 3]


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
            out["rpe1_late"] = e[thirds[2][0]:].mean()                 # [R145] 마지막 1/3 의 프레임간 이동(=속도) 오차 / reach
            out["rpe1_rot_late"] = geo_deg(Rp, Rg)[thirds[2][0]:].mean()   # 마지막 1/3 의 프레임간 회전(=각속도) 오차 deg
            curves["rpe1"] = np.r_[e, e[-1]]
    return {k: float(v) for k, v in out.items()}, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="label=<eval_dir> (test/ 아래 *_ref/_pred json)")
    ap.add_argument("--entries", nargs="*", default=None, help="data_name 목록 (기본: 첫 arm 의 test/ 전부)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus_root", default=None, help="[R112] 주면 smoothness·framing 지표 추가 (<root>/<prefix>/<video>/da3/prompts.json)")
    ap.add_argument("--graph_root", default="/data1/cympyc1785/LatentCamVid/camera_generation/dataset/out_dynpose")
    ap.add_argument("--name_prefix", default="dynpose")
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
    subj_cache = {}
    for label, d in arms:
        cs = []
        for n in names:
            pr, rf = osp.join(d, "test", f"{n}_transforms_pred.json"), osp.join(d, "test", f"{n}_transforms_ref.json")
            if not (osp.isfile(pr) and osp.isfile(rf)):
                continue
            R_, P_ = load(rf), load(pr)
            m, c = entry_metrics(R_, P_)
            if a.corpus_root:
                subj = subj_cache.setdefault(n, entry_subject_world(a.corpus_root, a.graph_root, a.name_prefix, n))
                if subj is not None and len(subj) != len(R_):
                    subj = None
                fx, fy, w_, h_ = load_intr(rf)
                g, uvg = smooth_framing(R_, subj, fx, fy, w_, h_, m["reach"])
                q, uvp = smooth_framing(P_, subj, *load_intr(pr), m["reach"])
                for k_, v_ in q.items():
                    m[k_] = float(v_)
                    m[k_ + "_gt"] = float(g[k_])
                if uvg is not None:
                    fe = np.linalg.norm(uvp - uvg, axis=1)
                    m["fr_err"] = float(fe.mean())
                    m["fr_err_late"] = float(fe[np.array_split(np.arange(len(fe)), 3)[2]].mean())   # [R145] 마지막 1/3
                    m["has_subj"] = 1.0
                else:
                    for k_ in ("fr_jit", "fr_std", "sz_jit", "in_box", "fr_err", "fr_jit_late", "fr_err_late"):
                        m.setdefault(k_, float("nan")); m.setdefault(k_ + "_gt", float("nan"))
                    m.pop("fr_err_gt", None); m.pop("fr_err_late_gt", None)
                    m["has_subj"] = 0.0
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
                       **{k: float(np.nanmean([r.get(k, np.nan) for r in R])) for k in keys if k != "reach"},
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
    if a.corpus_root:
        for label in summ:
            s_ = summ[label]
            for k_ in ("jerk_t", "jerk_r", "fr_jit", "fr_std", "sz_jit", "jerk_t_late", "jerk_r_late", "fr_jit_late"):
                s_[k_ + "_ratio"] = s_[k_] / max(s_[k_ + "_gt"], 1e-12)
        json.dump(summ, open(osp.join(a.out, "summary.json"), "w"), indent=1)
        fc = ["jerk_t", "jerk_t_gt", "jerk_r", "jerk_r_gt", "fr_err", "fr_jit", "fr_jit_gt", "fr_std", "fr_std_gt",
              "sz_jit", "sz_jit_gt", "in_box", "in_box_gt", "has_subj"]
        print("[smooth/framing] arm | " + " | ".join(fc))
        for label, s_ in summ.items():
            print(label + " | " + " | ".join(f"{s_[c]:.4f}" for c in fc))
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
