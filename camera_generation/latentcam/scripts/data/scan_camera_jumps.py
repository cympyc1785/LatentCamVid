#!/usr/bin/env python3
"""세그먼트 단위 카메라 점프(teleport) 스캐너.

왜 또 만드는가 — 기존 `scripts/data/filter_dl3dv.py::check_teleport` 는 두 가지 한계가 있다.
  (1) **scene 단위**다. 한 프레임만 튀어도 scene 전체(=수십 세그먼트)를 버린다. 반대로 세그먼트
      하나에만 있는 작은 점프는 scene 통계에 묻혀 살아남는다. 학습 단위는 49프레임 세그먼트다.
  (2) **절대 임계값**이다 (|dt|>10, ||dt||>15, |c|>50). DL3DV 최상위 transforms.json 의 COLMAP
      월드 단위에 맞춘 숫자라 da3 예측 pose 의 스케일 공간에는 그대로 못 쓴다.
      blacklist.csv 의 filter_teleport 21건 + filter_teleport_manual 1건이 그 필터의 결과다.

그래서 여기서는 **스케일 불변 지표**를 세그먼트마다 뽑는다 (49프레임 -> 48 step):
  step_i   = ||c_{i+1} - c_i||          (c = 카메라 중심, world)
  jr       = max(step) / median(step)   점프 신호. 한 step 만 유독 큰 경우를 잡는다.
  msf      = max(step) / sum(step)      한 step 이 경로 전체 길이를 얼마나 먹는지. 1/48=0.021 이
                                        등속, 0.5 면 절반이 한 프레임에서 튄 것.
  mse      = max(step) / extent         extent = max_i ||c_i - c_0||. 궤적 크기 대비 한 step.
  rmax     = max angle(R_{i+1} R_i^T)   회전 점프(deg). 평행이동 없이 시점만 튀는 경우.
  gap      = max(step) / p90(step)      median 이 0 에 가까운 정지 카메라용 보조 지표.
jr / msf / mse / rmax 는 전부 월드 스케일에 불변이라 transforms 와 da3 를 같은 임계값으로 비교할
수 있다. 절대 단위 max_step 도 참고용으로 같이 뱉는다.

GL/CV flip 은 이 지표들에 영향이 없다 — 중심 c 는 c2w 의 translation 그대로고, 상대회전
R_{i+1}R_i^T 는 오른쪽에 곱하는 축 flip 이 상쇄된다. 그래도 da3 는 w2c 라 c = -R^T t 로 푼다.

Run:
  # 측정만 (임계값 없이 분포 출력 + per-segment CSV)
  python scripts/data/scan_camera_jumps.py --seg-list <train.txt> <test.txt> --source both
  # 임계값 적용해서 세그먼트 blacklist 뽑기
  python scripts/data/scan_camera_jumps.py --seg-list ... --source da3 \
      --jr 12 --msf 0.25 --rmax 30 --out-blacklist data/seg_blacklist_da3.csv
"""
import argparse
import csv
import json
import os
import os.path as osp
import sys

import numpy as np

ROOT = "/data1/cympyc1785/data/DL3DV/scenes"
NFRAME = 49


# ---- pose 로딩 (dataset_dl3dv 의 _parse_transforms / _parse_da3 와 같은 규약) ----------------
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def load_centers_rot(scene_dir, source):
    """-> (centers (N,3), R_cv (N,3,3)).  R_cv 는 w2c 회전. 실패 시 예외."""
    if source == "da3":
        z = np.load(osp.join(scene_dir, "da3", "pose.npz"))
        E = np.asarray(z["extrinsics"], dtype=np.float64)      # (N,3,4) OpenCV w2c
        R = E[:, :3, :3]
        t = E[:, :3, 3]
        centers = -np.einsum("nji,nj->ni", R, t)               # -R^T t
        return centers, R
    tj = json.load(open(osp.join(scene_dir, "transforms.json")))
    fr = sorted(tj["frames"], key=lambda f: f["file_path"])
    c2w_gl = np.array([f["transform_matrix"] for f in fr], dtype=np.float64)
    c2w_cv = c2w_gl @ _GL2CV                                   # OpenGL -> OpenCV 축
    centers = c2w_cv[:, :3, 3]
    R = np.transpose(c2w_cv[:, :3, :3], (0, 2, 1))             # w2c 회전 = c2w 회전^T
    return centers, R


def seg_metrics(centers, R):
    """49프레임 구간의 (c,R) -> 지표 dict."""
    d = np.diff(centers, axis=0)
    step = np.linalg.norm(d, axis=1)                           # (T-1,)
    smax = float(step.max())
    smed = float(np.median(step))
    sp90 = float(np.percentile(step, 90))
    ssum = float(step.sum())
    extent = float(np.linalg.norm(centers - centers[0], axis=1).max())
    Rrel = np.einsum("nij,nkj->nik", R[1:], R[:-1])            # R_{i+1} R_i^T
    cos = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rmax = float(np.degrees(np.arccos(cos)).max())
    eps = 1e-12
    return {
        "jr": smax / max(smed, eps),
        "msf": smax / max(ssum, eps),
        "mse": smax / max(extent, eps),
        "gap": smax / max(sp90, eps),
        "rmax": rmax,
        "max_step": smax,
        "med_step": smed,
        "extent": extent,
    }


# ---- 세그먼트 열거 --------------------------------------------------------------------------
def read_seg_lists(paths):
    """`<batch>/<hash>/<seg>` 한 줄씩 -> {scene_rel: [seg_key,...]} (입력 순서 보존)."""
    scenes = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("/")
                scene_rel, seg = "/".join(parts[:-1]), parts[-1]
                scenes.setdefault(scene_rel, []).append(seg)
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-list", nargs="+", required=True)
    ap.add_argument("--source", default="both", choices=["transforms", "da3", "both"])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-csv", default="results/compare/camera_jumps/seg_jump_stats.csv")
    ap.add_argument("--out-blacklist", default=None,
                    help="주면 임계값을 넘긴 세그먼트를 blacklist.csv 형식으로 저장")
    ap.add_argument("--jr", type=float, default=None, help="max_step/median_step 임계값")
    ap.add_argument("--msf", type=float, default=None, help="max_step/path_len 임계값")
    ap.add_argument("--mse", type=float, default=None, help="max_step/extent 임계값")
    ap.add_argument("--rmax", type=float, default=None, help="프레임간 최대 회전(deg) 임계값")
    args = ap.parse_args()

    sources = ["transforms", "da3"] if args.source == "both" else [args.source]
    scenes = read_seg_lists(args.seg_list)
    print(f"[scan] {len(scenes)} scene / {sum(len(v) for v in scenes.values())} segment"
          f"  sources={sources}")

    rows, fails = [], {}
    for si, (scene_rel, segs) in enumerate(sorted(scenes.items())):
        sdir = osp.join(args.root, scene_rel)
        cache = {}
        for src in sources:
            try:
                cache[src] = load_centers_rot(sdir, src)
            except Exception as e:
                fails[f"{src}:{type(e).__name__}"] = fails.get(f"{src}:{type(e).__name__}", 0) + 1
        for seg in segs:
            s = int(seg) * NFRAME
            rec = {"scene": scene_rel, "seg": seg}
            ok = True
            for src in sources:
                if src not in cache:
                    ok = False
                    break
                centers, R = cache[src]
                if s + NFRAME > len(centers):
                    fails[f"{src}:short"] = fails.get(f"{src}:short", 0) + 1
                    ok = False
                    break
                m = seg_metrics(centers[s:s + NFRAME], R[s:s + NFRAME])
                for k, v in m.items():
                    rec[f"{src}_{k}"] = v
            if ok:
                rows.append(rec)
        if (si + 1) % 200 == 0:
            print(f"  ... {si + 1}/{len(scenes)} scene", flush=True)

    if fails:
        print(f"[scan] skipped: {fails}")
    if not rows:
        print("[scan] no rows")
        return

    os.makedirs(osp.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[scan] {len(rows)} segment -> {args.out_csv}")

    # ---- 분포 ----
    qs = [50, 90, 95, 99, 99.5, 99.9, 100]
    for src in sources:
        print(f"\n=== {src} ({len(rows)} segment) ===")
        print(f"{'metric':>10} " + " ".join(f"{('p%g' % q):>9}" for q in qs))
        for k in ("jr", "msf", "mse", "gap", "rmax", "max_step", "med_step", "extent"):
            v = np.array([r[f"{src}_{k}"] for r in rows])
            print(f"{k:>10} " + " ".join(f"{np.percentile(v, q):9.4f}" for q in qs))

    # ---- 임계값 적용 ----
    th = {"jr": args.jr, "msf": args.msf, "mse": args.mse, "rmax": args.rmax}
    th = {k: v for k, v in th.items() if v is not None}
    if th:
        for src in sources:
            hit = [r for r in rows if any(r[f"{src}_{k}"] > v for k, v in th.items())]
            per = {k: sum(1 for r in rows if r[f"{src}_{k}"] > v) for k, v in th.items()}
            nsc = len({r["scene"] for r in hit})
            print(f"\n[{src}] threshold {th} -> {len(hit)}/{len(rows)} segment "
                  f"({100 * len(hit) / len(rows):.2f}%), {nsc} scene 에 분포. 지표별: {per}")
        if args.out_blacklist:
            # source=both 이면 **합집합**을 쓴다. 두 arm(GT pose / da3 pose)이 같은 세그먼트
            # 집합을 봐야 paired 비교가 되기 때문. 실제로 한쪽만 망가진 경우가 양방향으로 있다
            # (COLMAP 이 튀고 da3 가 멀쩡한 세그먼트도, 그 반대도 있다).
            # 컬럼은 dataset_dl3dv._read_coverage_blacklist 가 읽는 `scene,segment` 그대로라
            # cfg.coverage_blacklist_path 에 바로 꽂으면 된다 (reason/detail 은 무시된다).
            hit = [r for r in rows
                   if any(r[f"{s}_{k}"] > v for s in sources for k, v in th.items())]
            os.makedirs(osp.dirname(args.out_blacklist) or ".", exist_ok=True)
            with open(args.out_blacklist, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["scene", "segment", "reason", "detail"])
                for r in hit:
                    bad = [f"{s}_{k}={r[f'{s}_{k}']:.3f}" for s in sources
                           for k, v in th.items() if r[f"{s}_{k}"] > v]
                    w.writerow([r["scene"], r["seg"],
                                f"seg_jump_{'+'.join(sources)}", " ".join(bad)])
            print(f"[scan] blacklist({'+'.join(sources)}) {len(hit)} row -> {args.out_blacklist}")


if __name__ == "__main__":
    main()
