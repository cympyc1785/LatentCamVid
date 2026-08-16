"""DataDoP 를 latentcam 에 넣을 때 translation 을 **frame-0 depth 로** 정규화하면 어디까지
canonicalize 되는가.

용어는 scripts/data/norm_divisor_compare.py / sd_clip_divisor_spread.py 와 같다:
  divisor D    cam_param translation 을 나누는 수 (dataset 의 norm_scale).
  reach        max_t ||c_t - c_0||.  모델이 회귀해야 하는 궤적의 크기.
  m = reach/D  정규화 reach.
  sd(log10 m)  **헤드라인**. 낮을수록 좋다. 레벨(평균)은 VAE 가 흡수하므로 중요하지 않다.
  sd(log10 m)^2 = sd(log10 reach)^2 + sd(log10 D)^2 - 2·corr·sd·sd
               -> divisor 는 매끄러워서가 아니라 **reach 와 상관이 있어서** 이긴다.

DataDoP 가 다른 두 코퍼스와 구조적으로 다른 점: shot 당 궤적이 하나뿐이라 context range 가
없다. 쓸 수 있는 건 **frame 0 한 장** (`<shot>_rgb.png` + `<shot>_depth.npy`) 뿐이고, 그게
정확히 우리가 context 로 넣을 그 한 장이라 leakage 도 없다. 그래서 후보 divisor 는 전부
frame-0 depth 의 통계량이다.

  raw      D = 1                       정규화 안 함 = MonST3R 재구성의 임의 스케일이 그대로 남는다
  meanray  mean(depth·||K⁻¹[u+.5,v+.5,1]||)   frame0 카메라 -> scene 점 **mean ray length**.
           DL3DV 의 저장된 avg_scale(make_avg_scale_da3_firstcam.py: mean||P − C_ctx0||) 과
           정의가 같다 -> 코퍼스 간 단위가 맞는 유일한 후보.
  medray   같은 ray length 의 median      (긴 꼬리/하늘에 덜 끌리는 변형)
  meanz    mean(depth)                  z-depth 평균 (ray 가 아니라 광축 성분)
  medz     median(depth)
  p10z     10th pct depth               근경 (피사체) 기준
  p90z     90th pct depth               원경 (배경) 기준

leaky 상한은 만들지 않는다 — DataDoP 는 target segment = shot 전체라 "target 에서 잰 D" 가
곧 reach 자신이고 sd=0 이 되어 정보가 없다.

env: N (shot 상한, 0=전부), WORKERS (24), VALID (1 = DataDoP_valid.txt 필터),
     ROOT, OUT_NAME (default datadop_divisor_spread)
out -> results/datadop/<OUT_NAME>/{summary.md,per_shot.csv,stats.json}
"""
import csv
import json
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.environ.get("ROOT", "/data1/cympyc1785/data/DataDoP/DataDoP_with_scene")
VALID_TXT = "/data1/cympyc1785/data/DataDoP/DataDoP_valid.txt"
N = int(os.environ.get("N", "0"))
WORKERS = int(os.environ.get("WORKERS", "24"))
USE_VALID = os.environ.get("VALID", "1") == "1"
OUT = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
               "results", "datadop", os.environ.get("OUT_NAME", "datadop_divisor_spread"))

# OpenGL c2w -> OpenCV c2w. dataset_dl3dv.py:108 의 _GL2CV 와 같은 행렬.
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])

DIVISORS = ["meanray", "medray", "meanz", "medz", "p10z", "p90z"]


def measure(base):
    """shot 하나 -> dict. 실패하면 None."""
    try:
        j = json.load(open(base + "_transforms_cleaning.json"))
        M = np.asarray([f["transform_matrix"] for f in j["frames"]], np.float64)   # (T,4,4) GL c2w
        if M.ndim != 3 or M.shape[1:] != (4, 4) or not np.isfinite(M).all():
            return None
        C = (M @ GL2CV)[:, :3, 3]                       # 카메라 중심 (flip 은 중심을 안 바꾸지만 명시)
        d = np.linalg.norm(C - C[0], axis=1)
        reach = float(d.max())
        mean_disp = float(d.mean())

        dep = np.load(base + "_depth.npy").astype(np.float64)                      # (h,w) z-depth
        h, w = int(j["h"]), int(j["w"])
        if dep.shape != (h, w) or not np.isfinite(dep).all() or dep.min() <= 0:
            return None
        fx, fy, cx, cy = j["fl_x"], j["fl_y"], j["cx"], j["cy"]
        # ||K^-1 [u+.5, v+.5, 1]|| = sqrt(x^2 + y^2 + 1),  ray length = z-depth * 이 값
        xs = (np.arange(w) + 0.5 - cx) / fx
        ys = (np.arange(h) + 0.5 - cy) / fy
        rayn = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2 + 1.0)
        ray = dep * rayn
        out = dict(shot=osp.relpath(base, ROOT), T=int(M.shape[0]), h=h, w=w,
                   reach=reach, mean_disp=mean_disp,
                   meanray=float(ray.mean()), medray=float(np.median(ray)),
                   meanz=float(dep.mean()), medz=float(np.median(dep)),
                   p10z=float(np.percentile(dep, 10)), p90z=float(np.percentile(dep, 90)))
        return out
    except Exception:
        return None


def _lstats(v):
    l = np.log10(v)
    return dict(n=int(v.size), med=float(np.median(v)),
                p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                sd_log10=float(l.std(ddof=1)), mean_log10=float(l.mean()))


def main():
    shots = []
    for scene in sorted(os.listdir(ROOT)):
        sd = osp.join(ROOT, scene)
        if not osp.isdir(sd):
            continue
        for f in sorted(os.listdir(sd)):
            if f.endswith("_transforms_cleaning.json"):
                shots.append(osp.join(sd, f[: -len("_transforms_cleaning.json")]))
    print(f"[scan] {len(shots)} shots under {ROOT}")

    if USE_VALID:
        ok = {ln.strip() for ln in open(VALID_TXT) if ln.strip()}
        keep = [b for b in shots if osp.relpath(b, ROOT) in ok]
        print(f"[valid] DataDoP_valid.txt 필터: {len(shots)} -> {len(keep)}")
        shots = keep
    if N:
        shots = shots[:N]

    with ProcessPoolExecutor(WORKERS) as ex:
        rows = [r for r in ex.map(measure, shots, chunksize=8) if r is not None]
    print(f"[measure] ok {len(rows)} / {len(shots)}")

    os.makedirs(OUT, exist_ok=True)
    with open(osp.join(OUT, "per_shot.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    reach = np.array([r["reach"] for r in rows])
    lr = np.log10(reach)
    stats = {"reach_raw": _lstats(reach), "n": len(rows)}
    lines = [f"# DataDoP divisor spread (n={len(rows)} shots, valid={USE_VALID})", "",
             "reach = max_t ||c_t - c_0||,  m = reach / D,  sd(log10 m) 이 낮을수록 canonical.",
             "corr = corr(log10 reach, log10 D) — divisor 가 이기는 유일한 경로.", "",
             "| divisor | med D | sd(log10 D) | med m | p05 m | p95 m | **sd(log10 m)** | corr | frac m>1 | frac m<0.1 |",
             "|---|---|---|---|---|---|---|---|---|---|",
             f"| raw (D=1) | 1 | 0 | {np.median(reach):.4f} | {np.percentile(reach,5):.4f} | "
             f"{np.percentile(reach,95):.4f} | **{lr.std(ddof=1):.3f}** | — | "
             f"{float((reach>1).mean()):.3f} | {float((reach<0.1).mean()):.3f} |"]

    for name in DIVISORS:
        D = np.array([r[name] for r in rows])
        m = reach / D
        lm, ld = np.log10(m), np.log10(D)
        corr = float(np.corrcoef(lr, ld)[0, 1])
        stats[name] = dict(D=_lstats(D), m=_lstats(m), corr_logreach_logD=corr,
                           frac_gt1=float((m > 1).mean()), frac_lt0p1=float((m < 0.1).mean()))
        lines.append(
            f"| {name} | {np.median(D):.4f} | {ld.std(ddof=1):.3f} | "
            f"{np.median(m):.4f} | {np.percentile(m,5):.4f} | "
            f"{np.percentile(m,95):.4f} | **{lm.std(ddof=1):.3f}** | {corr:+.3f} | "
            f"{float((m>1).mean()):.3f} | {float((m<0.1).mean()):.3f} |")

    # mean_disp 기준도 같이 (계획서/이전 측정과 잇기 위해)
    md = np.array([r["mean_disp"] for r in rows])
    mr = np.array([r["meanray"] for r in rows])
    stats["mean_disp_over_meanray"] = _lstats(md / mr)
    lines += ["", "참고: mean_disp = mean_t ||c_t - c_0|| (계획서에서 쓴 지표).",
              f"mean_disp/meanray  med {np.median(md/mr):.4f}  "
              f"p05 {np.percentile(md/mr,5):.4f}  p95 {np.percentile(md/mr,95):.4f}  "
              f"sd(log10) {np.log10(md/mr).std(ddof=1):.3f}"]

    open(osp.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")
    json.dump(stats, open(osp.join(OUT, "stats.json"), "w"), indent=1)
    print("\n".join(lines))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
