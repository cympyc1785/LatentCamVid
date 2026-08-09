"""[new] Scene-Decoupled: 같은 scene 의 두 clip 을 가로로 이어붙여 mp4 로 낸다.

왜: cross-clip 학습에서 context clip 의 `avg_scale_align` 을 target clip 의 분모로 쓸 계획인데,
같은 scene 안에서도 clip 마다 avg_scale_align 이 크게 다르다 (실측 300 scene / 5328 pair:
own/ctx 비 p50 1.000, p90 1.410, p99 4.389, max 16.921). 그 비가 실제로 "무엇이 다른
영상인가"를 눈으로 확인하려고 만들었다. clip 은 scene 안에서 **첫 프레임 카메라 pose 가 완전히
동일**(중심 산포 0.0000 m, 회전 산포 max 0.084 deg)하므로, 화면 차이는 전부 그 이후의
카메라 궤적 차이다.

프레임 위에 clip 이름 / avg_scale (da3 native) / avg_scale_align (= s * avg_scale, GT meters) /
moving / umeyama resid 를 찍는다. 두 clip 의 프레임 수가 다르면 짧은 쪽 마지막 프레임을
freeze 해서 길이를 맞춘다.

usage:
  python scripts/render/sd_clip_pair_video.py --scene <scene> --clips 01_24mm 07_24mm
  python scripts/render/sd_clip_pair_video.py --scene <scene> --clips A B --split whuman --fps 15
out -> results/scene_decoupled/clip_pair/<scene>__<clipA>_vs_<clipB>.mp4
"""
import os, sys, json, argparse
import numpy as np
import cv2

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
ROOT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"


def clip_meta(split, scene, clip_short):
    """da3/<split>/<scene>/<scene>_<clip_short>/ 의 스칼라 메타를 읽어 dict 로."""
    d = os.path.join(ROOT, "da3", split, scene, f"{scene}_{clip_short}")
    m = {"dir": d, "clip": clip_short}
    u = os.path.join(d, "umeyama_gt.json")
    if os.path.isfile(u):
        j = json.load(open(u))
        m["moving"] = bool(j.get("moving", False))
        m["s"] = j.get("s")
        m["resid"] = j.get("resid_rmse_over_rad")
    for k, sub in (("avg_scale", "avg_scale"), ("avg_scale_align", "avg_scale_align")):
        p = os.path.join(d, sub, "0.json")
        m[k] = json.load(open(p)) if os.path.isfile(p) else None
    return m


def read_frames(split, scene, clip_short):
    p = os.path.join(ROOT, "video", split, scene, f"{scene}_{clip_short}.mp4")
    cap = cv2.VideoCapture(p)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    if not fr:
        raise RuntimeError(f"no frames decoded from {p}")
    return fr


def banner(frame, meta, tag):
    """프레임 위쪽에 반투명 띠 + 텍스트. 원본을 건드리지 않게 복사본에 그린다."""
    f = frame.copy()
    h, w = f.shape[:2]
    bh = 74
    ov = f.copy()
    cv2.rectangle(ov, (0, 0), (w, bh), (0, 0, 0), -1)
    f = cv2.addWeighted(ov, 0.55, f, 0.45, 0)
    a = meta.get("avg_scale")
    ag = meta.get("avg_scale_align")
    lines = [
        f"[{tag}] {meta['clip']}",
        f"avg_scale {a:.4f}" + (f"   avg_scale_align {ag:.4f}" if ag is not None else
                                "   avg_scale_align --(static)"),
        (f"moving={meta.get('moving')}  s={meta['s']:.4f}  resid={meta['resid']:.4f}"
         if meta.get("s") is not None else f"moving={meta.get('moving')}  (no sim3)"),
    ]
    for i, t in enumerate(lines):
        cv2.putText(f, t, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--clips", nargs=2, required=True, help="clip 접미사 2개 (예: 01_24mm 07_24mm)")
    ap.add_argument("--split", default="whuman")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "scene_decoupled", "clip_pair"))
    args = ap.parse_args()

    metas = [clip_meta(args.split, args.scene, c) for c in args.clips]
    seqs = [read_frames(args.split, args.scene, c) for c in args.clips]
    n = max(len(s) for s in seqs)
    h = max(s[0].shape[0] for s in seqs)

    ag = [m.get("avg_scale_align") for m in metas]
    ratio = (ag[0] / ag[1]) if (ag[0] and ag[1]) else float("nan")
    print(f"scene={args.scene}")
    for m in metas:
        print(f"  {m['clip']:>12}  avg_scale={m['avg_scale']}  avg_scale_align={m['avg_scale_align']}"
              f"  moving={m.get('moving')}  resid={m.get('resid')}")
    print(f"  avg_scale_align ratio ({args.clips[0]}/{args.clips[1]}) = {ratio:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.scene}__{args.clips[0]}_vs_{args.clips[1]}.mp4")
    W = sum(s[0].shape[1] for s in seqs) + 4          # 4px 분리선
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, h))
    for i in range(n):
        panes = []
        for k, s in enumerate(seqs):
            # 짧은 쪽은 마지막 프레임 freeze -> 프레임 인덱스가 두 clip 에서 계속 같은 뜻을 갖는다
            f = s[min(i, len(s) - 1)]
            if f.shape[0] != h:
                f = cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h), interpolation=cv2.INTER_AREA)
            panes.append(banner(f, metas[k], "ctx" if k == 0 else "tgt"))
        sep = np.full((h, 4, 3), 255, np.uint8)
        cv2.putText(panes[0], f"frame {i:02d}", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.concatenate([panes[0], sep, panes[1]], axis=1))
    vw.release()
    print(f"saved {out}  ({n} frames, {W}x{h})")


if __name__ == "__main__":
    main()
