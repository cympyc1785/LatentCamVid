"""같은 scene 의 clip 들이 **똑같이 보는 부분**에서도 scale 이 어긋나는가?

Scene-Decoupled 는 clip 마다 da3 를 따로 돌리므로 clip 마다 임의 스케일이고, umeyama sim3
(`umeyama_gt.json` 의 `s`) 로 GT meters 에 맞춘다. 그런데 `avg_scale_align` 은 scene 안에서
clip 마다 median 1.25x, 최악 6.78x 까지 흔들린다 (scripts/data/sd_clip_divisor_spread.py).
그 원인이

  (a) clip 마다 **보이는 부분이 달라서** point set 이 달라진 것인가, 아니면
  (b) **똑같이 보이는 부분조차** 스케일이 다르게 나오는 것인가

를 가른다. 가를 수 있는 이유: 같은 scene 의 clip 들은 **frame-0 pose 가 완전히 동일**하다
(중심 산포 0.0000 m, 회전 산포 max 0.084 deg — 200 scene 실측). 즉 frame 0 은 모든 clip 이
글자 그대로 같은 그림을 본다. 그러니 frame-0 에서의 불일치는 전부 (b) 다.

측정 (scene 안의 모든 clip 쌍 A,B 에 대해):
  rgb_mae        frame-0 RGB 의 평균 절대차 (0-255). ~0 이어야 "같은 그림"이라는 전제가 선다.
  scale_off      median( d0_A*s_A / d0_B*s_B )  픽셀별 비의 median = **전역 스케일 어긋남**.
                 1.0 이면 sim3 가 두 clip 을 같은 미터로 맞춘 것.
  shape_sd       sd( log10(비) - log10(scale_off) )  = 전역 스케일을 뺀 뒤 남는 **깊이맵 모양**
                 불일치. da3 재구성 자체가 다르게 나온 몫.
비교 대상으로 align_ratio (= avg_scale_align 비) 를 같이 찍는다. align_ratio 는 크고
scale_off 는 1 에 가까우면 원인은 (a) 다.

env: N (scene 수, default 10), SPLIT (whuman), PIX_STRIDE (2), SEED (0)
out -> results/scene_decoupled/frame0_agreement/{summary.md,per_pair.csv}
"""
import os, sys, json, random, itertools
import numpy as np

ROOT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"
SPLIT = os.environ.get("SPLIT", "whuman")
N = int(os.environ.get("N", 10))
PIX_STRIDE = int(os.environ.get("PIX_STRIDE", 2))
SEED = int(os.environ.get("SEED", 0))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "scene_decoupled", "frame0_agreement")


def load_clip(scene, clip):
    d = os.path.join(ROOT, "da3", SPLIT, scene, clip)
    up, ap = os.path.join(d, "umeyama_gt.json"), os.path.join(d, "avg_scale_align", "0.json")
    if not (os.path.isfile(up) and os.path.isfile(ap)):
        return None
    u = json.load(open(up))
    if not u.get("moving", False):
        return None
    s = float(u["s"])
    z0 = np.asarray(np.load(os.path.join(d, "depth.npz"))["depth"][0], dtype=np.float32)
    z0 = z0[::PIX_STRIDE, ::PIX_STRIDE]

    # frame-0 RGB (mp4 첫 프레임). depth 격자(294x504)와 크기가 달라 비교용으로만 쓴다.
    import cv2
    mp4 = os.path.join(ROOT, "video", SPLIT, scene, clip + ".mp4")
    cap = cv2.VideoCapture(mp4)
    ok, img = cap.read()
    cap.release()
    return dict(clip=clip, s=s, z0=z0, rgb=(img.astype(np.float32) if ok else None),
                align=float(json.load(open(ap))),
                resid=float(u.get("resid_rmse_over_rad", float("nan"))))


def main():
    random.seed(SEED)
    scenes = [x for x in sorted(os.listdir(os.path.join(ROOT, "da3", SPLIT))) if x != "logs"]
    random.shuffle(scenes)

    rows, kept = [], 0
    for sc in scenes:
        if kept >= N:
            break
        sd = os.path.join(ROOT, "da3", SPLIT, sc)
        cs = [load_clip(sc, c) for c in sorted(os.listdir(sd))
              if c != "logs" and os.path.isdir(os.path.join(sd, c))]
        cs = [c for c in cs if c is not None]
        if len(cs) < 2:
            continue
        kept += 1
        for A, B in itertools.combinations(cs, 2):
            m = np.isfinite(A["z0"]) & np.isfinite(B["z0"]) & (A["z0"] > 0) & (B["z0"] > 0)
            lr = np.log10((A["z0"][m] * A["s"]) / (B["z0"][m] * B["s"]))
            off = float(10 ** np.median(lr))
            rows.append(dict(
                scene=sc, a=A["clip"][-9:], b=B["clip"][-9:],
                rgb_mae=(float(np.abs(A["rgb"] - B["rgb"]).mean())
                         if A["rgb"] is not None and B["rgb"] is not None
                         and A["rgb"].shape == B["rgb"].shape else float("nan")),
                scale_off=max(off, 1.0 / off),          # 방향 무관하게 1 이상으로
                shape_sd=float(np.std(lr - np.median(lr))),
                align_ratio=max(A["align"] / B["align"], B["align"] / A["align"]),
                s_ratio=max(A["s"] / B["s"], B["s"] / A["s"]),
                resid_max=max(A["resid"], B["resid"])))
        print(f"[{kept}/{N}] {sc}  clip {len(cs)}")

    os.makedirs(OUT, exist_ok=True)
    keys = ["scene", "a", "b", "rgb_mae", "scale_off", "shape_sd",
            "align_ratio", "s_ratio", "resid_max"]
    with open(os.path.join(OUT, "per_pair.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    def q(k):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return (f"{k:>12}  p50 {np.median(v):8.4f}   p90 {np.percentile(v,90):8.4f}   "
                f"p99 {np.percentile(v,99):8.4f}   max {v.max():8.4f}")

    L = [f"# frame-0 일치도  split={SPLIT}  scene={kept}  pair={len(rows)}", "",
         "frame-0 pose 는 clip 간 동일하므로 frame 0 은 '똑같이 보이는 부분' 이다.",
         "  rgb_mae     frame-0 RGB 평균 절대차 (0-255). ~0 이어야 전제 성립.",
         "  scale_off   픽셀별 (d0_A*s_A)/(d0_B*s_B) 의 median. **똑같이 보는 데서의 스케일 어긋남**",
         "  shape_sd    전역 스케일 제거 후 남는 깊이맵 모양 불일치 sd(log10)",
         "  align_ratio avg_scale_align 의 비 (= 실제로 divisor 가 흔들리는 폭)",
         "  s_ratio     umeyama sim3 스케일의 비", ""]
    for k in ("rgb_mae", "scale_off", "shape_sd", "align_ratio", "s_ratio", "resid_max"):
        L.append("  " + q(k))
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "summary.md"), "w").write(txt)
    print("\n" + txt + "saved -> " + OUT)


if __name__ == "__main__":
    main()
