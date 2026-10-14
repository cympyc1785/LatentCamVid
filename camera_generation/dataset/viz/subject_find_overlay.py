"""이 코드가 답하는 질문: text → subject 찾기 방법들 (SAM3 noun / SAM3 설명 문장 / Molmo2 점 → SAM3 / 점 종합 규칙들) 이 같은 영상에서 어떤 mask 를 냈나?

R375 (사용자 2026-10-14 "SAM3(noun) vs SAM3 (complex description) vs Molmo2 + SAM3 (point) 영상들이랑 Molmo2 + SAM3 여러 방식들 시도한 것도
overlay 비교 영상"). R356 (방법 열 비교) · R360 (결합 4x4) 의 일회성 viz 를 일반화했다.
입력 = 실험 폴더 `<root>/sam3/<method>/<video>__<anchor>.npz` (packbits masks, ok) · `<root>/gen/<pair>.json` (Molmo2 pointK8 점, conf) ·
eval json 들 (쌍마다 `<method>_iou`). 열 = GT | 방법들, 행 = 쌍 (4 행씩 한 영상). 타일: 방법 mask 채움 + GT 윤곽 (초록),
점 종합 규칙 열에는 단일 점 프레임의 Molmo2 점 (빨강 x, 그 프레임 ±3) 과 conf.
env vista4d.  예) python viz/subject_find_overlay.py --root tmp/r345 --eval tmp/r371/eval.json --eval tmp/r355/eval.json --select rules_differ --out tmp/r375
"""
import argparse
import json
import os
import sys

import cv2
import imageio.v2 as iio
import numpy as np

R = "/data1/cympyc1785/LatentCamVid"
C = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"
G = f"{R}/camera_generation/dataset/out_dynpose"
E = "/data1/cympyc1785/data/DynPose-100K/eval_data/recon_and_seg"
SEG = f"{R}/DATA/DynPose-100K/eval_data"
V4 = f"{R}/video_generation/models/Vista4D"
METHODS = [("sam3_label", "SAM3 noun", (255, 160, 0)), ("sam3_desc", "SAM3 description", (255, 0, 255)),
           ("pointK8f", "Molmo2 pt->SAM3", (0, 200, 255)), ("rule_earliest", "noun+pt earliest", (80, 160, 255)),
           ("rule_vote", "noun+pt vote", (120, 255, 120)), ("rule_conf", "noun+pt conf", (255, 220, 0)),
           ("rule_wsum", "noun+pt wsum", (255, 120, 180))]
RULES = {"rule_earliest", "rule_vote", "rule_conf", "rule_wsum"}


def gt_masks(v, aid):
    node = {n["id"]: n for n in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"]}[aid]
    sub = "seg_instances" if aid.startswith("dyn") else "seg_instances_static"
    meta = json.load(open(f"{SEG}/{sub}/{v}/meta.json")); mz = np.load(f"{SEG}/{sub}/{v}/masks.npz")
    ids = {node.get("track_id")} | set(node.get("merged_from") or []); out = {}
    for f in range(len(meta["frames"])):
        k = [i for i, b in enumerate(meta["frames"][f]) if b.get("id") in ids]
        if k:
            out[f] = np.unpackbits(mz[f"{f:05d}"][k], axis=-1, count=meta["width"]).astype(bool).any(0)
    return out


def meth_masks(root, m, v, aid):
    p = f"{root}/sam3/{m}/{v}__{aid}.npz"
    if not os.path.isfile(p):
        return None, "missing"
    z = np.load(p)
    how = str(z["how"]) if "how" in z.files else ""
    return (np.unpackbits(z["masks"], axis=-1, count=int(z["W"])).astype(bool) if bool(z["ok"]) else None), how


def phrase(v, aid):
    sys.path.insert(0, f"{R}/camera_generation/dataset/fit/bank")
    import molmo2_track_lift as T
    return T.phrase(v, aid)[len("Track "):].rstrip(".")


def row(a, ev, v, aid, methods):
    from utils.media import load_video
    fr, _ = load_video(f"{E}/{v}/video.mp4"); TW, TH = a.tile_w, a.tile_w * 9 // 16
    rs = lambda m: cv2.resize(m.astype(np.uint8), (TW, TH), interpolation=cv2.INTER_NEAREST).astype(bool)
    label = next((n.get("label") for n in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"] if n["id"] == aid), "") or ""
    desc = phrase(v, aid); Gm = gt_masks(v, aid); e = ev.get((v, aid), {})
    M = {m: meth_masks(a.root, m, v, aid) for m, _, _ in methods}
    gp = f"{a.root}/gen/{v}__{aid}.json"; pts = []
    if os.path.isfile(gp):
        g = json.load(open(gp)); P = g.get("pointK8", {}).get("pts") if isinstance(g.get("pointK8"), dict) and g["pointK8"].get("pts") else g.get("pts", [])
        nf = {}
        for q in P:
            nf[q[0]] = nf.get(q[0], 0) + 1
        pts = [q for q in P if nf[q[0]] == 1]
    out = []
    for f in range(len(fr)):
        b = cv2.resize(fr[f], (TW, TH)); tiles = []
        t = b.copy()
        if f in Gm:
            m = rs(Gm[f]); t[m] = (0.5 * t[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.putText(t, f"GT ({label})", (3, 13), 0, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(t, f"{v[:8]} {aid}", (3, TH - 6), 0, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
        for k in range(0, min(len(desc), 3 * 46), 46):
            cv2.putText(t, desc[k:k + 46], (3, 28 + 11 * (k // 46)), 0, 0.3, (255, 220, 255), 1, cv2.LINE_AA)
        tiles.append(t)
        for m, name, col in methods:
            t = b.copy(); mk, how = M[m]
            if mk is not None and f < len(mk) and mk[f].any():
                mm = rs(mk[f]); t[mm] = (0.45 * t[mm] + 0.55 * np.array(col)).astype(np.uint8)
            if f in Gm:
                cnt, _ = cv2.findContours(rs(Gm[f]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(t, cnt, -1, (0, 255, 0), 1)
            if m in RULES or m == "pointK8f":
                for q in pts:
                    if abs(q[0] - f) <= 3:
                        cv2.drawMarker(t, (int(q[2] * TW), int(q[3] * TH)), (255, 30, 30), cv2.MARKER_TILTED_CROSS, 12, 2)
                        if m == "rule_conf" or m == "rule_wsum":
                            cv2.putText(t, f"{q[4]:.2f}", (int(q[2] * TW) + 6, int(q[3] * TH) - 4), 0, 0.35, (255, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(t, name, (3, 13), 0, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
            if m == "sam3_label":
                cv2.putText(t, f"text '{label}'", (3, 27), 0, 0.35, (255, 220, 255), 1, cv2.LINE_AA)
            iou = e.get(f"{m}_iou")
            foot = ("NO DETECTION" if mk is None else (f"IoU {iou:.2f}" if iou is not None else "")) + (f" | {how}" if how and how not in ("masklet",) else "")
            cv2.putText(t, foot, (3, TH - 6), 0, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(t)
        out.append(np.hstack(tiles))
    print("ok", v[:8], aid, flush=True)
    return out


def pick(a, ev, pairs, methods):
    """--select: rules_differ (규칙끼리 IoU 차 ≥0.3) · desc_fail (설명 문장 실패, 다른 방법 성공) · all_ok · random. 앞에서부터 --n 개."""
    keys = [m for m, _, _ in methods]
    out = []
    for p in pairs:
        e = ev.get(tuple(p))
        if e is None:
            continue
        iou = {m: e.get(f"{m}_iou", np.nan) for m in keys}
        r_ = [iou[m] for m in keys if m in RULES and np.isfinite(iou[m])]
        if a.select == "rules_differ" and r_ and max(r_) - min(r_) >= 0.3:
            out.append(p)
        elif a.select == "desc_fail" and iou.get("sam3_desc", 1) < 0.2 and max(iou.get("rule_vote", 0), iou.get("sam3_label", 0)) >= 0.5:
            out.append(p)
        elif a.select == "method_differ" and max(iou[m] for m in keys if np.isfinite(iou[m])) - min(iou[m] for m in keys if np.isfinite(iou[m])) >= 0.5 and not (r_ and max(r_) - min(r_) >= 0.3):
            out.append(p)
        elif a.select == "all_ok" and all(iou[m] >= 0.6 for m in keys if np.isfinite(iou[m])):
            out.append(p)
        elif a.select == "random":
            out.append(p)
    if a.select == "random":
        np.random.default_rng(0).shuffle(out)
    return out[:a.n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)                           # 실험 폴더 (sam3/<method>/ · gen/ · pairs.json)
    ap.add_argument("--eval", action="append", default=[])             # 쌍별 <method>_iou 가 든 json (여럿이면 합침)
    ap.add_argument("--methods", default=",".join(m for m, _, _ in METHODS))
    ap.add_argument("--select", default="rules_differ", choices=["rules_differ", "desc_fail", "method_differ", "all_ok", "random"])
    ap.add_argument("--pairs", default="")                              # "video:anchor,..." 직접 지정 (있으면 --select 무시)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rows", type=int, default=4)                     # 영상 하나당 행 수
    ap.add_argument("--tile_w", type=int, default=320)
    ap.add_argument("--out", required=True)                            # 폴더 → <select>_<k>.mp4
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    sys.path.insert(0, V4)
    want = a.methods.split(","); methods = [x for x in METHODS if x[0] in want]
    ev = {}
    for p in a.eval:
        for r in json.load(open(p)):
            ev.setdefault((r["video"], r["anchor"]), {}).update(r)
    if a.pairs:
        pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]; tag = "pairs"
    else:
        pairs = pick(a, ev, json.load(open(f"{a.root}/pairs.json")), methods); tag = a.select
    os.makedirs(a.out, exist_ok=True)
    json.dump([list(p) for p in pairs], open(f"{a.out}/{tag}_pairs.json", "w"))
    for g in range(0, len(pairs), a.rows):
        cl = [row(a, ev, v, aid, methods) for v, aid in pairs[g:g + a.rows]]; n = min(len(c) for c in cl)
        p = f"{a.out}/{tag}_{g // a.rows + 1}.mp4"
        iio.mimwrite(p, [np.vstack([c[f] for c in cl]) for f in range(n)], fps=a.fps, macro_block_size=1)
        print("->", p, flush=True)


if __name__ == "__main__":
    main()
