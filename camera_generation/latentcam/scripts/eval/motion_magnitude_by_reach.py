"""모델이 카메라를 GT 보다 덜(또는 더) 움직이는가 — GT 이동 크기(reach) 4분위별 pred/GT 경로 길이·reach 비율.

R191 (A 가 jerk 는 낮은데 ADE·framing 이 나쁨 → 덜 움직여서인가) 에서 처음 쓰고 R193 에서 다시 써서 승격.
  path  = Σ_t |p(t+1) - p(t)|        (경로 길이)
  reach = max_t |p(t) - p(0)|         (frame 0 에서 가장 멀리 간 거리)
entry 마다 pred/GT 비율을 내고, GT reach 4분위 (첫 arm 의 traj_time_metrics per_entry.csv `reach` 기준) 로 나눠
중앙값을 낸다. `--per_entry` 를 주면 같은 분위별로 그 csv 의 지표 (ADE·FDE·fr_err·in_box 등) 평균도 같이 낸다.

    python scripts/eval/motion_magnitude_by_reach.py --arm base=<eval_dir> --arm A=<eval_dir2> \
        --per_entry <traj_time_metrics out>/per_entry.csv [--cols ADE,FDE,fr_err,in_box] [--out x.json]
eval_dir 는 eval_testset 출력 (test/<entry>_transforms_{pred,ref}.json)."""
import argparse
import collections
import csv
import json
import os

import numpy as np


def path_reach(f):
    d = json.load(open(f))
    p = np.array([fr["transform_matrix"] for fr in d["frames"]])[:, :3, 3]
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum(), np.linalg.norm(p - p[0], axis=1).max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=eval_dir")
    ap.add_argument("--per_entry", required=True, help="traj_time_metrics per_entry.csv (reach 분위 + 지표)")
    ap.add_argument("--cols", default="ADE,FDE,fr_err,in_box")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arm)
    rows = list(csv.DictReader(open(a.per_entry)))
    first = next(iter(arms))
    reach = {r["entry"]: float(r["reach"]) for r in rows if r["arm"] == first}
    ents = sorted(reach)
    qs = np.quantile([reach[e] for e in ents], [.25, .5, .75])
    qof = {e: int(np.searchsorted(qs, reach[e])) for e in ents}
    cols = [c for c in a.cols.split(",") if c]
    met = collections.defaultdict(list)
    for r in rows:
        if r["arm"] in arms and r["entry"] in qof:
            met[(r["arm"], qof[r["entry"]])].append([float(r[c]) if r[c] not in ("", "nan") else np.nan for c in cols])
    ratio = collections.defaultdict(list)
    ref_dir = arms[first]
    for e in ents:
        gl, gr = path_reach(f"{ref_dir}/test/{e}_transforms_ref.json")
        for k, d in arms.items():
            f = f"{d}/test/{e}_transforms_pred.json"
            if not os.path.isfile(f):
                continue
            pl, pr = path_reach(f)
            ratio[(k, qof[e])].append((pl / max(gl, 1e-9), pr / max(gr, 1e-9)))
    out = {"reach_quartile_edges": qs.tolist(), "arms": {}}
    for k in arms:
        out["arms"][k] = {}
        for q in range(4):
            r = np.array(ratio[(k, q)]); m = np.array(met[(k, q)])
            out["arms"][k][f"Q{q + 1}"] = {"n": len(r), "path_ratio_median": float(np.median(r[:, 0])),
                                           "reach_ratio_median": float(np.median(r[:, 1])),
                                           **{c: float(np.nanmean(m[:, i])) for i, c in enumerate(cols)}}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
