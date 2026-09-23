"""조준 arm 3종(`--aim_arms`) GT 렌더를 한 화면에 쌓고, 옆에 프레임별 회전 속도를 붙인다.

WHY: `compare_aim_timing.py` 가 낸 `rot_ratio`/`accel_p95` 는 "회전이 **언제** 일어나는가"의
지표인데, 렌더만 세 줄로 쌓아 놓으면 그게 안 보인다. 세 arm 은 위치가 비트 동일이라
(`bank_to_blender_poses.py` 가 매번 assert 한다) 화면 구도는 거의 같게 흐르고, 다른 것은
회전 타이밍뿐이라 눈으로는 "약간 덜컹거린다" 정도로만 읽힌다. 그래서 각 줄 오른쪽에
ω(f) = 프레임 간 회전각 곡선을 그리고 **현재 프레임 위치에 커서**를 얹는다 — smoothstep 의
keyframe 사이 펌핑이 곡선의 톱니로, everyframe 의 평탄함이 직선으로 바로 보인다.

곡선은 npz 의 c2w 에서 직접 잰다. `R_blend[f] = A @ R_bank[f]` 이고 A 는 프레임마다 같은
회전이라 `R[f]^T R[f+1]` 은 A 가 상쇄된다 — 즉 blend world 에서 재도 뱅크 world 에서 잰
것과 같은 값이다.

화면 글자는 전부 ASCII (matplotlib DejaVu Sans 에 한글 글리프가 없다).

사용 예시:
    python viz/aim_arm_reel.py \
        --cams results/20260901_aim_arms/cams --preset s_curve \
        --arms everyframe kf6_smoothstep kf6_smooth_kf \
        --out results/20260901_aim_arms/aim_arms_s_curve.mp4
"""

from argparse import ArgumentParser
from glob import glob
from os import path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rot_speed(cam):
    """c2w (F,4,4) -> 프레임 간 회전각 (F-1,), deg. 상수 basis 회전에 불변."""
    R = np.asarray(cam, dtype=np.float64)[:, :3, :3]
    rel = np.einsum("fji,fjk->fik", R[:-1], R[1:])
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def strip_chart(omegas, labels, kf, width, height, dpi=100):
    """arm 별 ω 곡선을 **같은 y 범위**로 각각 그려 (arm -> HxWx3 uint8) 로 돌려준다.

    y 범위를 공유하는 이유: 줄마다 축이 다르면 "누가 더 덜컹거리는가"를 눈으로 못 비교한다.
    """
    ymax = max(1e-6, max(float(o.max()) for o in omegas)) * 1.15
    out = []
    for om, lab in zip(omegas, labels):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0.16, 0.17, 0.81, 0.72])
        x = np.arange(len(om))
        ax.plot(x, om, color="#ffb000", lw=1.6)
        ax.fill_between(x, 0, om, color="#ffb000", alpha=0.25)
        for k in kf:                      # keyframe 위치 — 톱니의 마디가 여기 앉는지 보려고
            if 0 <= k < len(om):
                ax.axvline(k, color="#4fc3f7", lw=0.8, ls=":", alpha=0.9)
        ax.set_xlim(0, max(1, len(om) - 1))
        ax.set_ylim(0, ymax)
        ax.set_title(f"{lab}   rot/frame  max {om.max():.2f}  med {np.median(om):.2f} deg",
                     fontsize=8, color="w")
        ax.set_xlabel("frame", fontsize=7, color="w")
        ax.set_ylabel("deg", fontsize=7, color="w")
        ax.tick_params(colors="w", labelsize=6)
        for s in ax.spines.values():
            s.set_color("#888888")
        fig.patch.set_facecolor("#141414")
        ax.set_facecolor("#141414")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        out.append((buf, ax.get_position(), len(om)))
    return out


def draw_cursor(chart, box, n, f, width, height):
    """strip chart 위에 현재 프레임 세로선. `box` 는 axes 의 figure 상대 좌표."""
    img = chart.copy()
    x0, x1 = box.x0 * width, (box.x0 + box.width) * width
    y0 = (1.0 - box.y0 - box.height) * height
    y1 = (1.0 - box.y0) * height
    xc = int(round(x0 + (x1 - x0) * (f / max(1, n - 1))))
    xc = int(np.clip(xc, 0, width - 1))
    img[int(y0):int(y1), max(0, xc - 1):xc + 1] = (255, 80, 80)
    return img


def label(img, text, font):
    """좌상단 흰 글씨 + 검은 배경 (stack_videos.py 와 같은 모양)."""
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    d.rectangle([6, 6, 6 + (x1 - x0) + 12, 6 + (y1 - y0) + 10], fill=(0, 0, 0))
    d.text((12, 10), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def main():
    parser = ArgumentParser()
    #    `bank_to_blender_poses.py --out` 이 가리키던 폴더. `<preset>__<arm>.npz` 와
    #    `<preset>__<arm>/rgb/*.png` 가 둘 다 여기 있다.
    parser.add_argument("--cams", required=True, type=str)
    parser.add_argument("--preset", required=True, type=str)
    parser.add_argument("--arms", nargs="+",
                        default=["everyframe", "kf6_smoothstep", "kf6_smooth_kf"])
    parser.add_argument("--out", required=True, type=str)
    #    keyframe 위치 표시용. `keyframe_indices(49, 6)` 의 값.
    parser.add_argument("--kf", nargs="*", type=int, default=[0, 10, 19, 29, 38, 48])
    parser.add_argument("--chart_width", default=460, type=int)
    parser.add_argument("--fps", default=12.0, type=float)
    parser.add_argument("--gap", default=6, type=int)
    args = parser.parse_args()

    rgbs, cams = [], []
    for arm in args.arms:
        stem = path.join(args.cams, f"{args.preset}__{arm}")
        pngs = sorted(__import__("glob").glob(path.join(stem, "rgb", "*.png")))
        assert pngs, f"렌더가 없다: {stem}/rgb"
        rgbs.append([np.asarray(Image.open(p).convert("RGB")) for p in pngs])
        cams.append(np.load(stem + ".npz")["cam_c2w"])

    n = min(len(r) for r in rgbs)
    h, w = rgbs[0][0].shape[:2]
    #    위치가 arm 사이에서 같아야 이 대조가 성립한다 — 렌더 뒤에도 한 번 더 확인한다.
    for arm, cam in zip(args.arms[1:], cams[1:]):
        dpos = float(np.abs(cam[:, :3, 3] - cams[0][:, :3, 3]).max())
        assert dpos < 1e-6, f"{arm}: 위치가 다르다 |dt|={dpos:.3e}"

    omegas = [rot_speed(c) for c in cams]
    charts = strip_chart(omegas, args.arms, args.kf, args.chart_width, h)
    font = ImageFont.load_default()
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    except OSError:
        pass

    W = w + args.gap + args.chart_width
    H = len(args.arms) * h + (len(args.arms) - 1) * args.gap
    frames = []
    for f in range(n):
        canvas = np.full((H, W, 3), 20, dtype=np.uint8)
        for i, arm in enumerate(args.arms):
            om = omegas[i]
            ratio = float(om.max() / max(1e-9, np.median(om)))
            tile = label(rgbs[i][f], f"{arm}  rot_ratio {ratio:.2f}", font)
            chart, box, nn = charts[i]
            y = i * (h + args.gap)
            canvas[y:y + h, :w] = tile
            canvas[y:y + h, w + args.gap:] = draw_cursor(
                chart, box, nn, min(f, nn - 1), args.chart_width, h)
        frames.append(canvas)

    iio.mimwrite(args.out, frames, fps=args.fps, codec="libx264",
                 quality=6, macro_block_size=1)
    print(f"{'preset':16s} {args.preset}")
    print(f"{'frames':16s} {n}   {W}x{H}")
    print(f"{'arm':18s} {'rot_max':>9s} {'rot_med':>9s} {'rot_ratio':>10s} {'sum_deg':>9s}")
    for arm, om in zip(args.arms, omegas):
        print(f"{arm:18s} {om.max():9.4f} {np.median(om):9.4f} "
              f"{om.max() / max(1e-9, np.median(om)):10.4f} {om.sum():9.3f}")
    print(f"{'out':16s} {args.out}")


if __name__ == "__main__":
    main()
