"""Scene-Decoupled cross-clip 학습에서 divisor 를 뭘로 잡을지 고른다.

용어 (scripts/data/norm_divisor_compare.py 와 같은 정의를 쓴다):
  divisor D   cam_param 의 translation 을 나누는 수 (scale_mode=avg_scale 의 norm_scale).
  reach       max_t ||c_t - c_0||, target clip 카메라 중심의 최대 이동거리. **GT meters**.
  m = reach/D 정규화 reach. 모델이 실제로 회귀해야 하는 크기.
  sd(log10 m) 이게 낮을수록 좋다. 레벨(평균)은 VAE 가 흡수하므로 중요하지 않다.

비교하는 divisor 3종. cross-clip 세팅에서 D 는 **context clip 만으로** 계산돼야 한다
(추론 때 target 은 없으니까). own 은 그래서 실현 불가능한 상한(leaky)이고, 나머지 둘이 후보다.

  own    target clip 자기 avg_scale_align            [LEAKY, 상한]
  ctx    context clip 의 avg_scale_align             = 현재 cross-clip 계획
  ctxd   context clip 의 **장면 깊이** (median depth * s, meters)   = 제안

핵심 질문: ctx 는 "우연히 뽑힌 context clip 이 얼마나 빨리 움직였나"라서 target 과 무관한
잡음이 D 에 섞인다. ctxd 는 "context 가 본 방의 크기"라 같은 scene 안에서 어느 clip 을 뽑든
거의 같은 값이 나와야 한다. 그게 사실인지를 scene 내 clip 간 산포(spread)로 확인한다.

static clip(moving=False) 은 avg_scale_align 이 아예 없어서 제외한다.

env: N (scene 수, default 10), SPLIT (whuman), FRAME_STRIDE (8), PIX_STRIDE (4), SEED (0)
out -> results/scene_decoupled/clip_divisor_spread/{summary.md,per_clip.csv,per_pair.csv}
"""
import os, sys, json, random, itertools
import numpy as np

ROOT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"
SPLIT = os.environ.get("SPLIT", "whuman")
N = int(os.environ.get("N", 10))
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", 8))
PIX_STRIDE = int(os.environ.get("PIX_STRIDE", 4))
SEED = int(os.environ.get("SEED", 0))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "scene_decoupled", "clip_divisor_spread")


def clip_dirs(scene):
    d = os.path.join(ROOT, "da3", SPLIT, scene)
    return sorted(x for x in os.listdir(d)
                  if x != "logs" and os.path.isdir(os.path.join(d, x)))


def load_clip(scene, clip):
    """moving clip 하나의 측정치. static 이거나 파일이 없으면 None."""
    d = os.path.join(ROOT, "da3", SPLIT, scene, clip)
    up = os.path.join(d, "umeyama_gt.json")
    ap = os.path.join(d, "avg_scale_align", "0.json")
    if not (os.path.isfile(up) and os.path.isfile(ap)):
        return None
    u = json.load(open(up))
    if not u.get("moving", False):
        return None
    s = float(u["s"])                       # da3 units -> GT meters

    # reach: pose.npz extrinsics 는 OpenCV w2c (T,3,4). center = -R^T t.
    P = np.load(os.path.join(d, "pose.npz"))
    ext = np.asarray(P["extrinsics"], dtype=np.float64)
    R, t = ext[:, :3, :3], ext[:, :3, 3]
    c = -np.einsum("tji,tj->ti", R, t)      # R^T @ t 를 배치로
    reach = float(np.max(np.linalg.norm(c - c[0:1], axis=-1)) * s)   # meters

    # 장면 깊이: da3 depth * s 의 median (meters). frame/pixel 서브샘플.
    z = np.load(os.path.join(d, "depth.npz"))["depth"]
    z = np.asarray(z[::FRAME_STRIDE, ::PIX_STRIDE, ::PIX_STRIDE], dtype=np.float32)
    z = z[np.isfinite(z) & (z > 0)]
    depth_m = float(np.median(z) * s)

    return dict(scene=scene, clip=clip, s=s, reach_m=reach,
                align=float(json.load(open(ap))),
                depth_m=depth_m,
                resid=float(u.get("resid_rmse_over_rad", float("nan"))))


def spread(v):
    """scene 안에서 clip 끼리 얼마나 흩어져 있나. max/min 배수와 sd(log10)."""
    v = np.asarray(v, dtype=np.float64)
    return float(v.max() / v.min()), float(np.std(np.log10(v)))


def q(v, name):
    v = np.asarray(v, dtype=np.float64)
    return (f"{name:>6}  n={len(v):5d}  sd(log10)={np.std(np.log10(v)):.4f}  "
            f"p1={np.percentile(v,1):.4f}  p10={np.percentile(v,10):.4f}  "
            f"p50={np.percentile(v,50):.4f}  p90={np.percentile(v,90):.4f}  "
            f"p99={np.percentile(v,99):.4f}  min={v.min():.4f}  max={v.max():.4f}")


def main():
    random.seed(SEED)
    scenes = sorted(os.listdir(os.path.join(ROOT, "da3", SPLIT)))
    scenes = [s for s in scenes if s != "logs"]
    random.shuffle(scenes)

    per_clip, kept = [], 0
    for sc in scenes:
        if kept >= N:
            break
        cs = [load_clip(sc, c) for c in clip_dirs(sc)]
        cs = [c for c in cs if c is not None]
        if len(cs) < 2:                      # pair 를 못 만드는 scene 은 건너뛴다
            continue
        per_clip.extend(cs)
        kept += 1
        print(f"[{kept}/{N}] {sc}  moving clip {len(cs)}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "per_clip.csv"), "w") as f:
        f.write("scene,clip,s,reach_m,align,depth_m,resid\n")
        for c in per_clip:
            f.write(f"{c['scene']},{c['clip']},{c['s']:.6f},{c['reach_m']:.6f},"
                    f"{c['align']:.6f},{c['depth_m']:.6f},{c['resid']:.6f}\n")

    # ---- (1) scene 내 clip 간 산포: divisor 가 "어느 clip 을 뽑든 같은 값" 인가 ----
    by_scene = {}
    for c in per_clip:
        by_scene.setdefault(c["scene"], []).append(c)

    lines = [f"# Scene-Decoupled clip divisor spread   split={SPLIT}  scene={kept}  "
             f"clip={len(per_clip)}", "",
             "## (1) scene 안에서 clip 을 바꿨을 때 divisor 가 얼마나 흔들리나",
             "   align = avg_scale_align (context clip 의 움직임)   depth_m = 장면 깊이(meters)",
             "   ratio = scene 안 max/min. 1.0 에 가까울수록 'context 를 누구로 뽑든 같은 값'.", ""]
    lines.append(f"{'scene':<46}{'nclip':>6}{'align_ratio':>13}{'depth_ratio':>13}")
    ar, dr = [], []
    for sc, cs in sorted(by_scene.items()):
        a, _ = spread([c["align"] for c in cs])
        d, _ = spread([c["depth_m"] for c in cs])
        ar.append(a); dr.append(d)
        lines.append(f"{sc:<46}{len(cs):>6}{a:>13.4f}{d:>13.4f}")
    lines += ["", f"  align_ratio  median {np.median(ar):.4f}   max {max(ar):.4f}",
              f"  depth_ratio  median {np.median(dr):.4f}   max {max(dr):.4f}", ""]

    # ---- (2) 모델이 실제로 회귀할 크기 m = reach/D 의 산포 ----
    pairs = []
    for sc, cs in by_scene.items():
        for tgt, ctx in itertools.permutations(cs, 2):
            pairs.append(dict(scene=sc, tgt=tgt["clip"], ctx=ctx["clip"],
                              m_own=tgt["reach_m"] / tgt["align"],
                              m_ctx=tgt["reach_m"] / ctx["align"],
                              m_ctxd=tgt["reach_m"] / ctx["depth_m"]))
    with open(os.path.join(OUT, "per_pair.csv"), "w") as f:
        f.write("scene,target,context,m_own,m_ctx,m_ctxd\n")
        for p in pairs:
            f.write(f"{p['scene']},{p['tgt']},{p['ctx']},"
                    f"{p['m_own']:.6f},{p['m_ctx']:.6f},{p['m_ctxd']:.6f}\n")

    lines += ["## (2) normalized reach  m = reach / D   (pair 단위, n=%d)" % len(pairs),
              "   sd(log10 m) 이 낮을수록 좋다. 레벨은 VAE 가 흡수하므로 무관.",
              "   own = target 자기 align [LEAKY 상한] / ctx = 현재 계획 / ctxd = 제안", ""]
    for k, nm in (("m_own", "own"), ("m_ctx", "ctx"), ("m_ctxd", "ctxd")):
        lines.append("   " + q([p[k] for p in pairs], nm))

    txt = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "summary.md"), "w").write(txt)
    print("\n" + txt)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
