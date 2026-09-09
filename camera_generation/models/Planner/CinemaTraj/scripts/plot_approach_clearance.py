"""`trumans_approach_viz.py` 의 approach.json 을 **걸음별 clearance 곡선**으로 그린다.

왜: 3D 렌더는 "어디서 걸리나"를 보여주지만 **얼마나 가파르게 걸리는지**는 못 보여준다. 임계
0.35 와 0.20 사이가 16 cm 인지 2 cm 인지가 곧 "게이트를 낮추면 궤적을 얼마나 더 얻나"인데,
그건 곡선의 기울기다. 그래서 렌더 옆에 붙일 2D 패널을 같은 프레임 수로 뽑는다.

곡선을 **두 개** 그린다. 실선 `clearance` 는 게이트가 실제로 쓰는 값(`CLEARANCE_DIRS` 6방향,
±z 포함)이고 점선 `clearance_h` 는 수평 4방향만이다. 둘이 갈라지는 구간이 곧 **바닥이 게이트를
먹는 구간** — a17 감사에서 47.9% 였던 `floor_is_binding_frac` 이 여기서 눈에 보인다.

한글 폰트가 이 호스트의 matplotlib 에 없다 — 라벨은 전부 영문이다.

사용:
  python scripts/plot_approach_clearance.py --approach /tmp/approach_d118/eye/approach.json \
      --out /tmp/approach_d118/eye/plot
"""
from argparse import ArgumentParser
from os import makedirs, path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(summary, cursor, args):
    """걸음 `cursor` 에 커서를 둔 패널 하나. cursor<0 이면 커서 없는 정지 그림."""
    rows = summary["rows"]
    travel = np.array([r["travel"] for r in rows])
    clear = np.array([r["clearance"] for r in rows])
    clear_h = np.array([r["clearance_h"] for r in rows])
    probe = summary["probe_distance"]

    fig, axis = plt.subplots(figsize=(args.width / 100.0, args.height / 100.0), dpi=100)
    fig.patch.set_facecolor("#0d0d12")
    axis.set_facecolor("#0d0d12")

    #    임계선을 먼저 깔아야 곡선이 그 위에 온다.
    for cross in summary["crossings"]:
        color = tuple(cross["color"])
        axis.axhline(cross["threshold"], color=color, lw=1.4, ls="-", alpha=0.75, zorder=1)
        axis.text(travel[-1], cross["threshold"], f" {cross['threshold']:.2f}",
                  color=color, va="center", ha="left", fontsize=10, zorder=6)
        if cross["travel"] is not None:
            axis.axvline(cross["travel"], color=color, lw=1.2, ls=":", alpha=0.85, zorder=1)
            axis.plot([cross["travel"]], [cross["threshold"]], "o", color=color,
                      ms=7, mec="white", mew=0.8, zorder=7)

    axis.plot(travel, clear, "-", color="#ffffff", lw=2.2, zorder=4,
              label="clearance (gate, 6 dirs incl. floor)")
    axis.plot(travel, clear_h, "--", color="#4d8cff", lw=1.8, zorder=3,
              label="clearance_h (4 horizontal dirs = wall only)")

    if cursor >= 0:
        axis.axvline(travel[cursor], color="#19ffe6", lw=1.6, alpha=0.9, zorder=5)
        axis.plot([travel[cursor]], [clear[cursor]], "o", color="#19ffe6", ms=8,
                  mec="black", mew=0.8, zorder=8)
        row = rows[cursor]
        binder = row["binding_name"] or "(none)"
        axis.set_title(f"step {row['step']:>2}/{len(rows) - 1}   moved {row['travel']:.3f} m   "
                       f"wall gap {row['wall_gap']:.3f} m\n"
                       f"clearance {row['clearance']:.3f} m   binding: {binder}"
                       f"{'  [FLOOR]' if row['floor_is_binding'] else ''}",
                       color="white", fontsize=11, loc="left")
    else:
        axis.set_title(f"{path.basename(path.dirname(args.approach))}: approach to "
                       f"{summary['wall_name']}", color="white", fontsize=11, loc="left")

    axis.set_xlabel("distance marched toward the wall (m)", color="white", fontsize=10)
    axis.set_ylabel("clearance (m)", color="white", fontsize=10)
    axis.set_xlim(travel[0], travel[-1] * 1.06 + 1e-6)
    axis.set_ylim(0.0, min(probe, float(max(clear.max(), clear_h.max()))) * 1.08)
    axis.grid(True, color="#2a2a36", lw=0.7)
    for spine in axis.spines.values():
        spine.set_color("#3a3a48")
    axis.tick_params(colors="#c8c8d4", labelsize=9)
    legend = axis.legend(loc="upper right", fontsize=8.5, framealpha=0.25)
    for text in legend.get_texts():
        text.set_color("white")
    fig.tight_layout()
    return fig


def main():
    parser = ArgumentParser()
    parser.add_argument("--approach", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    #    렌더 프레임과 나란히 붙일 것이므로 기본 크기를 렌더(960x540)에 맞춘다.
    parser.add_argument("--width", default=960, type=int)
    parser.add_argument("--height", default=540, type=int)
    #    --frames 면 걸음마다 한 장(커서 애니메이션), --no_frames 면 정지 그림 한 장만.
    parser.add_argument("--frames", dest="frames", action="store_true", default=True)
    parser.add_argument("--no_frames", dest="frames", action="store_false")
    args = parser.parse_args()

    with open(args.approach, encoding="utf-8") as file:
        summary = json.load(file)
    makedirs(args.out, exist_ok=True)

    fig = draw(summary, -1, args)
    still = path.join(args.out, "clearance.png")
    fig.savefig(still, facecolor=fig.get_facecolor())
    plt.close(fig)

    made = 0
    if args.frames:
        for i in range(len(summary["rows"])):
            fig = draw(summary, i, args)
            fig.savefig(path.join(args.out, f"frame_{i:05d}.png"), facecolor=fig.get_facecolor())
            plt.close(fig)
            made += 1

    print(f"{'still':<16}{still}")
    print(f"{'frames':<16}{made}  -> {args.out}")
    print(f"{'floor binding':<16}{summary['floor_binding_frac'] * 100:.1f}% of steps")
    print(f"\n{'threshold':>10}{'step':>7}{'moved':>9}{'wall gap':>10}"
          f"{'step(h)':>9}{'wall gap(h)':>13}")
    print("-" * 58)
    for cross in summary["crossings"]:
        step_s = "--" if cross["step"] is None else str(cross["step"])
        travel_s = "never" if cross["travel"] is None else f"{cross['travel']:.3f}"
        gap_s = "--" if cross["wall_gap"] is None else f"{cross['wall_gap']:.3f}"
        step_h = "--" if cross["step_h"] is None else str(cross["step_h"])
        gap_h = "--" if cross["wall_gap_h"] is None else f"{cross['wall_gap_h']:.3f}"
        print(f"{cross['threshold']:>10.2f}{step_s:>7}{travel_s:>9}{gap_s:>10}"
              f"{step_h:>9}{gap_h:>13}")


if __name__ == "__main__":
    main()
