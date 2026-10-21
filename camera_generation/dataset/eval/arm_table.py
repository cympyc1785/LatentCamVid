"""이 코드가 답하는 질문: 여러 arm (우리 모델·외부 baseline) 의 궤적·프레이밍·CLaTr·충돌 지표를 look-at / free 로 나눠 한 표로 보면?

R484·R476 에서 같은 집계를 두 번 손으로 짰다 (CLAUDE.md "두 번째로 쓰게 되면 승격"). 입력은 이미 있는 산출물 셋:
  --traj   `traj_time_metrics.py` 의 per_entry.csv (여러 개, arm 열로 구분)
  --clatr  `clatr_score_dir.py --out_json` (arm 이름이 최상위 키)
  --coll   `pred_collision_rate.py` 출력 폴더 (per_entry.csv, method 는 폴더 이름 끝 shell/esdf)
look-at / free 는 코퍼스 `prompts_v2.json` 의 preset → `lbm.presets.PRESETS[preset][1]` (aim).
지표 정의 (기존 보고와 같다):
  ADE / rot_mean            per_entry 평균 (static 세그먼트는 traj_time_metrics 가 이미 뺀다)
  off_err                   |fr_off − fr_off_gt| 평균 (subject 화면 중심 거리 오차)
  fr_err · in_box           per_entry 그대로
  coll_any / coll_frac      per_entry 평균 (한 프레임이라도 충돌한 비율 / 충돌 프레임 비율)
env: vista4d 또는 latentcam (numpy 만). 예:
    python eval/arm_table.py --traj tmp/r478/traj/per_entry.csv tmp/r476/traj_*/per_entry.csv \
        --clatr tmp/r483/clatr_v2_*.json --coll tmp/r484/coll_shell tmp/r484/coll_esdf --arms MTb gendop_rgbd_p30slerp_raw
"""
import csv
import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lbm.presets import PRESETS  # noqa: E402

GROUPS = ("all", "look_at", "free")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--clatr", nargs="*", default=[])
    ap.add_argument("--coll", nargs="*", default=[])
    ap.add_argument("--arms", nargs="*", default=None)        # 표 행 순서 (없으면 나온 순서)
    ap.add_argument("--corpus", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_v5f50k")
    ap.add_argument("--prompts", default="prompts_v2.json")
    ap.add_argument("--prefix", default="dynpose_")
    a = ap.parse_args()

    cache = {}

    def aim(entry):
        v, k = entry[len(a.prefix):].rsplit("_", 1)
        if v not in cache:
            p = os.path.join(a.corpus, "dynpose", v, "da3", a.prompts)
            cache[v] = json.load(open(p)) if os.path.isfile(p) else {}
        x = cache[v].get(k)
        return PRESETS[x["preset"]][1] if isinstance(x, dict) and x.get("preset") in PRESETS else None

    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # arm -> group -> metric -> values
    order = []

    def add(arm, entry, vals):
        g = aim(entry)
        if g is None:
            return
        if arm not in order:
            order.append(arm)
        for grp in ("all", g):
            for k, v in vals.items():
                agg[arm][grp][k].append(v)

    for p in a.traj:
        for r in csv.DictReader(open(p)):
            add(r["arm"], r["entry"], {"ADE": num(r["ADE"]), "rot": num(r["rot_mean"]), "fr_err": num(r["fr_err"]),
                                       "in_box": num(r["in_box"]), "off_err": abs(num(r["fr_off"]) - num(r["fr_off_gt"]))})
    for d in a.coll:
        m = os.path.basename(os.path.normpath(d)).split("_")[-1]
        for r in csv.DictReader(open(os.path.join(d, "per_entry.csv"))):
            if r.get("error"):
                continue
            add(r["arm"], r["entry"], {f"{m}_any": num(r["coll_any"]), f"{m}_frac": num(r["coll_frac"])})
    clatr = {}
    for p in a.clatr:
        clatr.update(json.load(open(p)))

    arms = a.arms or order
    keys = ["ADE", "rot", "off_err", "fr_err", "in_box"] + sorted({k for arm in agg for k in agg[arm]["all"] if k.endswith(("_any", "_frac"))})
    print("| arm | CLaTr | FCD | F1 | cos | n |")
    print("|---|---|---|---|---|---|")
    for arm in arms:
        c = clatr.get(arm, {})
        print(f"| {arm} | {c.get('clatr/clatr_score', np.nan):.2f} | {c.get('clatr/fcd', np.nan):.3f} | "
              f"{c.get('captions/fscore', np.nan):.4f} | {c.get('clatr/pred_ref_cosine', np.nan):.2f} | {len(agg[arm]['all']['ADE']) or len(agg[arm]['all'].get('shell_any', []))} |")
    print()
    print("| arm | " + " | ".join(f"{k} (all/look/free)" for k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for arm in arms:
        cells = []
        for k in keys:
            vs = [np.nanmean(agg[arm][g][k]) if agg[arm][g][k] else np.nan for g in GROUPS]
            cells.append(" / ".join(f"{v:.4f}" for v in vs))
        print(f"| {arm} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
