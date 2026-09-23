"""격자(mesh_grid) raycast 게이트가 Blender `scene.ray_cast` 판정과 얼마나 일치하나 (R27).

이 코드가 답하는 질문: "TRUMANS 충돌·가림·거리 게이트를 fitting 루프 안에서 Blender 없이
`lbm.mesh_collision.mesh_ray_profile` 로 재면, 뱅크 뒤 Blender raycast
(`bank_to_blender_poses.py --raycast` → selection.json `raycast.audit`) 와 판정이 같게 나오나".

같은 카메라(뱅크 poses.npz 의 그 variant_id)를 두 방식으로 재서 게이트별 통과/탈락 혼동행렬과
연속값 차이, 변이당 시간을 낸다. 임계는 Blender 쪽 기본값과 같다.

    python eval/compare_mesh_gates.py --sel_root <repo>/tmp/results/d266/raycast_all \
        --bank_dir hole_bank_d266T --limit 40
"""
import json
import sys
import time
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, HERE)
from lbm.mesh_collision import MeshClearance, mesh_ray_profile  # noqa: E402

GATES = {"clear_frac": ("min_clear_frac", 0.90), "min_clearance": ("min_clearance", 0.20),
         "min_subject_dist": ("min_subject_dist", 0.80), "min_floor_drop": ("min_floor_drop", 0.30)}


def main(a):
    sels = sorted(glob(path.join(a.sel_root, "*", "selection.json")))[:a.limit or None]
    rows, secs = [], []
    for sp in sels:
        video = path.basename(path.dirname(sp))
        grid = path.join(HERE, a.output_root, video, "mesh_grid.npz")
        pz = path.join(HERE, a.output_root, video, a.bank_dir, "poses.npz")
        if not (path.exists(grid) and path.exists(pz)):
            continue
        mesh = MeshClearance(grid)
        poses = np.load(pz)
        ids = [str(v) for v in poses["variant_id"].tolist()]
        audit = json.load(open(sp))["raycast"]["audit"]
        for row in audit:
            if row["variant_id"] not in ids:
                continue
            c2w = poses["cam_c2w"][ids.index(row["variant_id"])]
            t0 = time.time()
            g = mesh_ray_profile(mesh.to_blend(c2w), mesh)
            secs.append(time.time() - t0)
            rows.append({"video": video, "variant_id": row["variant_id"],
                         "blender": {k: row.get(k) for k in GATES}, "grid": {k: g[k] for k in GATES},
                         "blender_pass": bool(row["passed"])})
    if not rows:
        print("비교할 행이 없다")
        return

    def ok(vals):
        return all(vals[k] is not None and vals[k] >= thr for k, (_, thr) in GATES.items())

    print(f"scenes {len({r['video'] for r in rows})}  rungs {len(rows)}  "
          f"grid time/variant median {np.median(secs) * 1e3:.1f} ms  p90 {np.percentile(secs, 90) * 1e3:.1f} ms")
    print(f"\n{'gate':18s} {'both pass':>9s} {'both fail':>9s} {'grid only fail':>14s} "
          f"{'blender only fail':>17s} {'agree':>7s}  {'median |Δ|':>10s}")
    for k, (_, thr) in GATES.items():
        b = np.array([r["blender"][k] if r["blender"][k] is not None else np.nan for r in rows])
        gv = np.array([r["grid"][k] for r in rows])
        bp, gp = b >= thr, gv >= thr
        fin = np.isfinite(b) & np.isfinite(gv)
        print(f"{k:18s} {int((bp & gp).sum()):9d} {int((~bp & ~gp).sum()):9d} {int((bp & ~gp).sum()):14d} "
              f"{int((~bp & gp).sum()):17d} {float((bp == gp).mean()):7.3f}  "
              f"{float(np.median(np.abs(b[fin] - gv[fin]))) if fin.any() else float('nan'):10.3f}")
    bp = np.array([r["blender_pass"] for r in rows])
    gp = np.array([ok(r["grid"]) for r in rows])
    print(f"\n{'ALL gates':18s} {int((bp & gp).sum()):9d} {int((~bp & ~gp).sum()):9d} "
          f"{int((bp & ~gp).sum()):14d} {int((~bp & gp).sum()):17d} {float((bp == gp).mean()):7.3f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f)


if __name__ == "__main__":
    q = ArgumentParser()
    q.add_argument("--sel_root", required=True)
    q.add_argument("--output_root", default="out_trumans")
    q.add_argument("--bank_dir", default="hole_bank_d266T")
    q.add_argument("--limit", default=0, type=int)
    q.add_argument("--out", default="")
    main(q.parse_args())
