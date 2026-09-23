"""eval 폴더의 arm 궤적들을 **카메라 절두체 와이어**로 3D 에 세워 비교한다.

왜 필요한가. `render_pred_depth_warp.py` 는 "그 카메라에서 보면 어떻게 보이나"를 보여주지만
**카메라 자체가 씬 안에서 어떤 경로를 그리는지**는 안 보여준다. warp 영상에서 궤적이 중간에
꺾이는 게 보여도 그게 (a) 좌표 규약이 어긋나 축이 뒤집힌 건지 (b) 모델이 특정 프레임부터
발산한 건지 (c) 회전만 튄 건지를 warp 만 보고는 못 가른다. 절두체를 프레임 순서대로 세워
경로선과 함께 그리면 셋이 그림에서 바로 갈린다 — 축 뒤집힘은 경로 전체가 거울상이고,
발산은 특정 인덱스부터 점들이 흩어지며, 회전 튐은 경로는 매끈한데 절두체 방향만 홱 돈다.

규약. eval JSON 의 `transform_matrix` 는 **OpenGL c2w** 라 `GL2CV = diag(1,-1,-1,1)` 을 곱해
OpenCV c2w (X right / Y down / Z forward) 로 읽는다 (`render_pred_depth_warp.load_transforms`
와 동일). 절두체 꼭짓점은 `fl_x/fl_y/w/h` 로 실제 화각을 계산해 찍으므로 벌어진 각이 그
카메라가 실제로 담는 화각이다. 깊이만 `--frustum_len` 으로 줄여 그린다.

축 범위는 **arm 전체를 합쳐 한 번만** 정한다. arm 마다 따로 잡으면 발산한 궤적이 자동으로
축소돼 정상처럼 보인다 — 크기 차이를 지우지 않는 게 이 그림의 요점이다.

env: latentcam (matplotlib + imageio)

예시:
    PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
    $PY viz/plot_eval_frustums.py --video camel --entries 173 87 \\
        --ref_eval_dir /data1/.../eval_my/20260904_165917_vista4d_d121_da3_t128__epoch100__seed42 \\
        --eval_dir d124=/data1/.../eval_my/20260904_185554_vista4d_d121_molmo2__epoch100__seed42 \\
        --eval_dir gendop_raw=results/20260906_d156_gendop_d121/eval_dir_gendop_rgbd_p49_raw \\
        --out_dir /data1/cympyc1785/LatentCamVid/tmp/d159/frustum
"""
import json
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])       # OpenGL c2w <-> OpenCV c2w (자기역원)


def load_transforms(json_path: str):
    """eval JSON -> (OpenCV c2w (T,4,4), meta)."""
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)
    c2w = np.array([f["transform_matrix"] for f in data["frames"]], dtype=np.float64) @ GL2CV
    return c2w, data


def frustum_segments(c2w: np.ndarray, half_x: float, half_y: float, depth: float):
    """카메라 하나의 절두체 와이어를 (선분 리스트) 로. OpenCV 축 (X right / Y down / Z fwd)."""
    corners = np.array([[+half_x, +half_y, 1.0], [+half_x, -half_y, 1.0],
                        [-half_x, -half_y, 1.0], [-half_x, +half_y, 1.0]]) * depth
    pts = (c2w[:3, :3] @ corners.T).T + c2w[:3, 3]
    center = c2w[:3, 3]
    segs = [(center, p) for p in pts]                       # 옆면 4
    segs += [(pts[i], pts[(i + 1) % 4]) for i in range(4)]  # 앞면 사각형
    return segs


def draw_arm(ax, c2w, half_x, half_y, depth, stride, cmap, mark_frame, label):
    """경로선 + stride 간격 절두체. 색은 프레임 인덱스."""
    from matplotlib import colormaps
    n = len(c2w)
    cm = colormaps[cmap]
    p = c2w[:, :3, 3]
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color="0.55", lw=0.9, zorder=1)
    for f in range(0, n, stride):
        col = cm(f / max(n - 1, 1))
        for a, b in frustum_segments(c2w[f], half_x, half_y, depth):
            ax.plot(*zip(a, b), color=col, lw=1.0, zorder=2)
    ax.scatter(*p[0], color="lime", s=34, depthshade=False, zorder=5)
    if mark_frame is not None and 0 <= mark_frame < n:
        ax.scatter(*p[mark_frame], color="red", s=40, marker="x", depthshade=False, zorder=5)
    ax.set_title(label, fontsize=10)


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    from matplotlib import pyplot as plt

    arms = [("GT", args.ref_eval_dir, "_transforms_ref.json")]
    for spec in args.eval_dir:
        label, d = spec.split("=", 1)
        arms.append((label, d, "_transforms_pred.json"))

    written = []
    for entry in args.entries:
        name = f"{args.prefix}_{args.video}_{entry}"
        tracks, meta = [], None
        for label, d, suf in arms:
            p = path.join(d, "test", name + suf)
            if not path.isfile(p):
                print(f"  [skip] {label}: {p} 없음")
                continue
            c2w, m = load_transforms(p)
            meta = meta or m
            tracks.append((label, c2w))
        if len(tracks) < 2:
            continue

        half_x = 0.5 * meta["w"] / meta["fl_x"]
        half_y = 0.5 * meta["h"] / meta["fl_y"]
        allp = np.concatenate([c[:, :3, 3] for _, c in tracks])
        span = float(np.abs(allp - allp.mean(0)).max()) * args.pad + 1e-6
        mid = allp.mean(0)
        depth = span * args.frustum_len

        frames = []
        for k in range(args.orbit_frames):
            fig = plt.figure(figsize=(5.2 * len(tracks), 5.6))
            for i, (label, c2w) in enumerate(tracks):
                ax = fig.add_subplot(1, len(tracks), i + 1, projection="3d")
                draw_arm(ax, c2w, half_x, half_y, depth, args.stride, args.cmap,
                         args.mark_frame, f"{label}   n={len(c2w)}")
                for lim, c in zip("xyz", mid):
                    getattr(ax, f"set_{lim}lim")(c - span, c + span)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=args.elev, azim=args.az0 + 360.0 * k / args.orbit_frames)
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
                ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            fig.suptitle(f"{name}   green=frame0"
                         + (f"   red x=frame {args.mark_frame}" if args.mark_frame is not None else "")
                         # matplotlib 기본 폰트에 한글이 없어 라벨은 영문으로 (tofu 방지)
                         + f"   shared axis limits +-{span:.3f}", fontsize=10)
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            # libx264 + yuv420p 는 짝수 해상도만 받는다. arm 수에 따라 폭이 홀수가 되므로
            # 여기서 잘라 맞춘다 (macro_block_size=1 로도 홀수는 못 넘어간다).
            frames.append(buf[:buf.shape[0] // 2 * 2, :buf.shape[1] // 2 * 2].copy())
            plt.close(fig)

        out = path.join(args.out_dir, f"{name}_frustum.mp4")
        imageio.mimwrite(out, frames, fps=args.fps, codec="libx264", quality=6,
                         macro_block_size=1)
        imageio.imwrite(path.join(args.out_dir, f"{name}_frustum.png"), frames[0])
        written.append((name, span, out))
        print(f"  {name:<26} span {span:.4f}  -> {path.basename(out)}")

    print(f"\n{'entries':<14}{len(written)}")
    print(f"{'arms':<14}{', '.join(a[0] for a in arms)}")
    print(f"{'out_dir':<14}{args.out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--ref_eval_dir", required=True)     # GT(ref) 를 읽을 폴더
    parser.add_argument("--eval_dir", action="append", default=[])   # LABEL=DIR (반복)
    parser.add_argument("--video", required=True)
    parser.add_argument("--entries", nargs="+", required=True)
    parser.add_argument("--prefix", default="vista4d")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stride", type=int, default=3)     # 절두체 간격(프레임)
    parser.add_argument("--frustum_len", type=float, default=0.10)   # 축 범위 대비 깊이
    parser.add_argument("--mark_frame", type=int, default=None)      # 빨간 x 로 표시할 프레임
    parser.add_argument("--pad", type=float, default=1.05)   # 축 범위 여유
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--orbit_frames", type=int, default=90)
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--az0", type=float, default=-60.0)
    parser.add_argument("--fps", type=float, default=15.0)
    main(parser.parse_args())
