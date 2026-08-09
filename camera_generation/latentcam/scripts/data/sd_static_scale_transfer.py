"""static clip 을 살릴 수 있나 — moving clip 의 정렬된 frame-0 depth 로 sim3 스케일을 역산.

Scene-Decoupled 의 static clip(`umeyama_gt.json` 의 `moving: false`, 23408 중 7682 = 32.8%) 은
카메라가 안 움직여 Umeyama 가 풀리지 않는다. 그래서 `s` 도 `avg_scale_align` 도 없고 현재
계획에선 context/target 양쪽에서 제외된다.

살리는 방법: 같은 scene 의 clip 들은 **frame-0 pose 가 동일**하므로 (중심 산포 0.0000 m,
회전 산포 max 0.084 deg) frame 0 은 모든 clip 이 같은 그림을 본다. 이미 정렬된 moving clip M 의
frame-0 depth(meters) 와 static clip S 의 frame-0 depth(da3 단위) 를 픽셀별로 비교하면

    s_S  =  median( d0_M * s_M / d0_S )

로 S 의 스케일을 역산할 수 있다. 그러면 `avg_scale_align_S = avg_scale_S * s_S` 도 나온다
(이 항등식은 moving clip 에서 상대오차 max 0.0 으로 정확히 성립함을 확인했다).

판정 두 가지:
  shape_sd     전역 스케일을 뺀 뒤 남는 깊이맵 **모양** 불일치. moving-moving 쌍의 값
               (p50 0.0321, sd_frame0_depth_agreement.py) 과 비슷해야 "static 의 depth 도
               똑같이 쓸 만하다" 는 뜻이다. 크면 da3 가 static clip 에서 다른 구조를 냈다는 것.
  s_spread     같은 static clip 을 **서로 다른 moving clip** 으로 역산했을 때 나오는 s 의
               max/min. 1 에 가까워야 역산이 기준 clip 에 무관하다. 참고선: moving-moving 의
               scale_off p50 1.0352 / max 1.2939 가 이 측정의 잡음 바닥이다.

env: N (scene 수, default 10), SPLIT (whuman), PIX_STRIDE (2), SEED (0)
out -> results/scene_decoupled/static_scale_transfer/{summary.md,per_static.csv,per_pair.csv}
"""
import os, sys, json, random, itertools
import numpy as np

ROOT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"
SPLIT = os.environ.get("SPLIT", "whuman")
N = int(os.environ.get("N", 10))
PIX_STRIDE = int(os.environ.get("PIX_STRIDE", 2))
SEED = int(os.environ.get("SEED", 0))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "scene_decoupled", "static_scale_transfer")


def load_clip(scene, clip):
    d = os.path.join(ROOT, "da3", SPLIT, scene, clip)
    up = os.path.join(d, "umeyama_gt.json")
    if not os.path.isfile(up):
        return None
    u = json.load(open(up))
    z0 = np.asarray(np.load(os.path.join(d, "depth.npz"))["depth"][0],
                    dtype=np.float32)[::PIX_STRIDE, ::PIX_STRIDE]
    ap = os.path.join(d, "avg_scale", "0.json")
    return dict(clip=clip, moving=bool(u.get("moving", False)),
                s=float(u["s"]) if u.get("moving", False) else None,
                z0=z0,
                avg_scale=float(json.load(open(ap))) if os.path.isfile(ap) else float("nan"),
                resid=float(u.get("resid_rmse_over_rad", float("nan"))))


def main():
    random.seed(SEED)
    scenes = [x for x in sorted(os.listdir(os.path.join(ROOT, "da3", SPLIT))) if x != "logs"]
    random.shuffle(scenes)

    pairs, statics, kept = [], [], 0
    for sc in scenes:
        if kept >= N:
            break
        sd = os.path.join(ROOT, "da3", SPLIT, sc)
        cs = [load_clip(sc, c) for c in sorted(os.listdir(sd))
              if c != "logs" and os.path.isdir(os.path.join(sd, c))]
        cs = [c for c in cs if c is not None]
        mv = [c for c in cs if c["moving"]]
        st = [c for c in cs if not c["moving"]]
        if not mv or not st:
            continue
        kept += 1
        for S in st:
            implied = []
            for M in mv:
                m = np.isfinite(S["z0"]) & np.isfinite(M["z0"]) & (S["z0"] > 0) & (M["z0"] > 0)
                lr = np.log10((M["z0"][m] * M["s"]) / S["z0"][m])
                s_imp = float(10 ** np.median(lr))
                implied.append(s_imp)
                pairs.append(dict(scene=sc, static=S["clip"][-9:], mover=M["clip"][-9:],
                                  s_implied=s_imp, s_mover=M["s"],
                                  shape_sd=float(np.std(lr - np.median(lr))),
                                  resid_mover=M["resid"]))
            a = np.array(implied)
            statics.append(dict(scene=sc, static=S["clip"][-9:], n_ref=len(a),
                                s_med=float(np.median(a)),
                                s_spread=float(a.max() / a.min()),
                                avg_scale=S["avg_scale"],
                                align_est=S["avg_scale"] * float(np.median(a))))
        print(f"[{kept}/{N}] {sc}  moving {len(mv)}  static {len(st)}")

    os.makedirs(OUT, exist_ok=True)
    for name, data, keys in (
            ("per_pair.csv", pairs,
             ["scene", "static", "mover", "s_implied", "s_mover", "shape_sd", "resid_mover"]),
            ("per_static.csv", statics,
             ["scene", "static", "n_ref", "s_med", "s_spread", "avg_scale", "align_est"])):
        with open(os.path.join(OUT, name), "w") as f:
            f.write(",".join(keys) + "\n")
            for r in data:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    def q(rows, k):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return (f"{k:>12}  p50 {np.median(v):8.4f}   p90 {np.percentile(v,90):8.4f}   "
                f"p99 {np.percentile(v,99):8.4f}   max {v.max():8.4f}")

    L = [f"# static clip 스케일 역산  split={SPLIT}  scene={kept}  "
         f"static clip={len(statics)}  (static,moving) pair={len(pairs)}", "",
         "  shape_sd   전역 스케일 제거 후 깊이맵 모양 불일치. moving-moving 참고선 p50 0.0321.",
         "  s_spread   같은 static 을 서로 다른 moving 으로 역산했을 때 s 의 max/min.",
         "             moving-moving 잡음 바닥 참고선: scale_off p50 1.0352 / max 1.2939.", ""]
    L.append("  [ (static, moving) pair 단위 ]")
    L.append("  " + q(pairs, "shape_sd"))
    L.append("")
    L.append("  [ static clip 단위 ]")
    L.append("  " + q(statics, "s_spread"))
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "summary.md"), "w").write(txt)
    print("\n" + txt + "saved -> " + OUT)


if __name__ == "__main__":
    main()
