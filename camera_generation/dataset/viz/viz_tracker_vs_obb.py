"""이 코드가 답하는 질문: CoTracker subject 궤적이 scene graph OBB 중심보다 실제 움직임을 잘 따라가나? (anchor 별 영상 비교)

R246/R261 (2026-10-08). 일부만 보이는 물체는 마스크가 줄었다 늘면서 OBB 중심이 떠돌아 "정지인데 움직임" 으로 판정된다.
`tmp/r246/track.py` 가 anchor 마다 남긴 `<tracks>/<scene>__<anchor>.npz` (trk_disp · tr2d · vis) 를 소스 영상 위에 그린다.
  자홍 원 = OBB 중심 (graph track.center_smooth) · 초록 원 = CoTracker 궤적 (center_smooth[0] + trk_disp) · 하늘색 점 = 추적 중인 점.
`--cases scene8:anchor[:note] ...` 를 주거나, 안 주면 `--res` 결과에서 자동으로 고른다
(OBB moving→tracker static 큰 순 2 · 둘 다 moving 1 · 끊김(min_pairs<5) 1).
env: vista4d.  예) python viz/viz_tracker_vs_obb.py --tracks /…/tmp/r246/tracks_v3 --res '/…/tmp/r246/v3/shard*_res.jsonl' --out /…/tmp/r261/viz.mp4
"""
import glob
import json
from argparse import ArgumentParser

import cv2
import imageio.v2 as iio
import numpy as np

EVAL = "/data1/cympyc1785/data/DynPose-100K/eval_data"
GRAPH = "/data1/cympyc1785/LatentCamVid/camera_generation/dataset/out_dynpose"


def auto_cases(rows, thr=0.05):
    ok = [r for r in rows if r.get("min_pairs", 0) >= 5]
    flip = sorted([r for r in ok if r["obb_drift"] >= thr and r["trk_drift"] < thr],
                  key=lambda r: r["trk_drift"] - r["obb_drift"])[:2]
    agree = sorted([r for r in ok if r["obb_drift"] >= thr and r["trk_drift"] >= thr],
                   key=lambda r: -min(r["obb_drift"], r["trk_drift"]))[:1]
    brk = [r for r in rows if r.get("min_pairs", 99) < 5][:1]
    return ([(r, "OBB moving -> tracker static") for r in flip] + [(r, "both moving") for r in agree]
            + [(r, "tracker break") for r in brk])


def tile_frames(r, note, tracks, size=(560, 315)):
    s, an = r["scene"], r["anchor"]
    g = json.load(open(f"{GRAPH}/{s}/scene_graph.json"))
    n = {x["id"]: x for x in g["nodes"]}[an]
    Twg = np.linalg.inv(np.asarray(g["frames"]["T_gw"], dtype=float))
    cs = np.asarray(n["track"]["center_smooth"])
    z = np.load(f"{tracks}/{s}__{an}.npz")
    trk = cs[0] + z["trk_disp"]
    cam = np.load(f"{EVAL}/recon_and_seg/{s}/cameras.npz")
    C, I = cam["cam_c2w"], cam["intrinsics"]
    out = []
    for t, frame in enumerate(iio.get_reader(f"{EVAL}/recon_and_seg/{s}/video.mp4")):
        im = np.asarray(frame).copy()
        for p, vv in zip(z["tr2d"][t], z["vis"][t]):
            if vv:
                cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 255, 255), -1)
        for P3, col in ((cs[t], (255, 0, 255)), (trk[t], (0, 255, 0))):
            w = Twg[:3, :3] @ P3 + Twg[:3, 3]
            c = np.linalg.inv(C[t]) @ np.r_[w, 1]
            if c[2] > 0:
                fx, fy, cx, cy = I[t][:4]
                cv2.circle(im, (int(fx * c[0] / c[2] + cx), int(fy * c[1] / c[2] + cy)), 20, col, 6)
        im = cv2.resize(im, size)
        cv2.rectangle(im, (0, 0), (size[0], 44), (0, 0, 0), -1)
        cv2.putText(im, f"{s[:8]} {an} {n['label']} | {note}", (6, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, f"drift OBB {r['obb_drift']:.3f} vs tracker {r['trk_drift']:.3f} | pairs {r.get('min_pairs')} | t{t}",
                    (6, 38), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(im)
    return out


def main(a):
    rows = [json.loads(l) for f in glob.glob(a.res) for l in open(f)]
    rows = [r for r in rows if "trk_drift" in r]
    if a.cases:
        idx = {(r["scene"][:8], r["anchor"]): r for r in rows}
        cases = []
        for c in a.cases:
            sc, an, *note = c.split(":")
            cases.append((idx[(sc, an)], note[0] if note else ""))
    else:
        cases = auto_cases(rows)
    tiles = [tile_frames(r, note, a.tracks) for r, note in cases]
    T = min(len(x) for x in tiles)
    cols = a.columns
    blank = np.zeros_like(tiles[0][0])
    w = iio.get_writer(a.out, fps=8, macro_block_size=1)
    for t in range(T):
        cells = [x[t] for x in tiles] + [blank] * ((-len(tiles)) % cols)
        w.append_data(np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]))
    w.close()
    for r, note in cases:
        print(r["scene"], r["anchor"], note, r["obb_drift"], r["trk_drift"], r.get("min_pairs"))
    print("->", a.out)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--res", required=True)          # glob, track.py 결과 jsonl
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--columns", default=2, type=int)
    main(ap.parse_args())
