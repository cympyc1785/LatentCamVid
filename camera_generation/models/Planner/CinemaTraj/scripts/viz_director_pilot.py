"""DIRECTOR 파일럿 결과(`director_poses.npz` 여러 개)를 caption 별로 나란히 애니메이션한다.

WHY: 궤적은 시작/끝 좌표 표만 봐서는 "어디서 무너지는지"가 안 보인다. caption 을 열로,
     top(XZ) / side(ZY) 두 시점을 행으로 깔고 49프레임을 돌리면 seed 간 산포와
     char 추종 여부가 한 화면에 보인다.

`--space` 로 좌표계를 고른다.

* `et`(기본) — **E.T. world 좌표 그대로** (y-up, char frame0 = 원점, m). 기존 동작.
* `rel` — **첫 context view(소스 c2w[0])를 identity 로 둔 상대 pose** (OpenCV, world u).
  E.T. 축과 화면 좌우의 대응이 뒤집혀 보이는 문제를 없앤다: 이 좌표계의 +x 가 소스 frame0
  카메라의 오른쪽이므로 영상에서 오른쪽으로 가면 top-down 에서도 오른쪽으로 간다.
  회전 변환은 `render_director_depth.et_to_opencv_c2w` 를 그대로 재사용한다.
  **단 (x,z) 평면은 수평이 아니다** — 소스 카메라가 기울어져 있으면 그만큼 기운다
  (snowboard: 중력축이 카메라 -y 에서 12.83° 벗어나 top-down 평면이 12.14° 기움).
* `grav`(권장) — `rel` 을 **중력 정렬**한 것. up = `gravity.up_world` 를 rel 로 옮긴 것,
  가로축은 소스 frame0 카메라의 오른쪽을 수평면에 투영한 것, 세로축은 그 시선을 투영한 것.
  좌우는 렌더 화면과 그대로 맞으면서 top-down 이 **진짜 수평면**이 되고 side/front 는
  고도를 중력 기준으로 읽는다. snowboard 에서 subject 고도 변화가 rel-y 로는 −0.646 u 인데
  중력축으로는 −0.012 u — 그 차이가 전부 기운 평면이 만든 가짜 수직 성분이었다.

사용 예시:
    conda run -n GenDoP python scripts/viz_director_pilot.py \
        --runs results/20260827_director_camel/cap_static:static \
               results/20260827_director_camel/cap_right_in:truck-right+push-in \
        --out results/20260827_director_camel/pilot.mp4

    conda run -n GenDoP python scripts/viz_director_pilot.py \
        --runs results/20260827_director_snowboard/cap_forward:forward \
        --space rel --planes top --out .../traj_top_rel.mp4
"""
from argparse import ArgumentParser
from os import path
from sys import path as syspath
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

syspath.insert(0, path.dirname(path.dirname(path.abspath(__file__))))
from scripts.render_director_depth import et_to_opencv_c2w      # noqa: E402

# space -> 이름 -> (제목, (가로축, 부호), (세로축, 부호)).
# E.T. 는 y-up 이라 top-down 이 x-z 이고, rel 은 소스 frame0 카메라축(OpenCV, y-down)이라
# 세로로 세우는 축마다 부호를 뒤집어야 위가 위로 간다.
PLANES = {
    "et": {"top":      ("top-down (x-z)",         (0, +1), (2, +1)),
           "top_flip": ("top-down flipped (x-z)", (0, +1), (2, -1)),
           "side":     ("side (z-y)",             (2, +1), (1, +1)),
           "front":    ("front (x-y)",            (0, +1), (1, +1))},
    "rel": {"top":      ("top-down (x right, z fwd up)",   (0, +1), (2, +1)),
            "top_flip": ("top-down (x right, z fwd down)", (0, +1), (2, -1)),
            # up(y)축 기준 좌우 반전 = 가로 x 부호만 뒤집는다 (거울상).
            "top_mirror":      ("top-down mirrored (x left, z fwd up)",   (0, -1), (2, +1)),
            "top_mirror_flip": ("top-down mirrored (x left, z fwd down)", (0, -1), (2, -1)),
            "side":     ("side (z fwd, -y up)",            (2, +1), (1, -1)),
            "front":    ("front (x right, -y up)",         (0, +1), (1, -1))},
    # grav 도 성분 순서는 rel 과 같은 (right, down, fwd) 라 부호표가 같다. 다른 건 축이
    # 중력에 맞아 top-down 이 진짜 수평면이라는 것뿐.
    "grav": {"top":      ("top-down, gravity-aligned (x right, z fwd up)",   (0, +1), (2, +1)),
             "top_flip": ("top-down, gravity-aligned (x right, z fwd down)", (0, +1), (2, -1)),
             "side":     ("side (z fwd, -y up)",                             (2, +1), (1, -1)),
             "front":    ("front (x right, -y up)",                          (0, +1), (1, -1))},
}


def load_run(run_dir):
    data = np.load(path.join(run_dir, "director_poses.npz"))
    with open(path.join(run_dir, "meta.json"), encoding="utf-8") as file:
        meta = json.load(file)
    return data, meta


def to_rel(data, meta, align_gravity=False):
    """E.T. 포즈 -> **첫 context view(소스 c2w[0]) 를 identity 로 둔 상대 pose** (OpenCV, DA3).

    두 단계다. ① npz 의 pose 는 scene graph **G frame** 이라 `T_wg` 로 world 로 올리고
    (`render_director_depth.g_to_world` 와 같은 변환), ② `inv(cam_c2w[0])` 를 앞에 곱한다.
    ①을 빼면 subject 가 카메라 뒤로 가고 좌우가 뒤집혀 보인다 — v1 의 버그가 그것이었다.

    `align_gravity` 면 여기에 ③ 중력 정렬 회전 `R_plot` 을 한 번 더 곱해 성분 순서를
    (right, up, fwd) 로 바꾼다. rel 은 소스 카메라 축이라 (x,z) 평면이 수평이 아니고, 그
    기울기만큼 수평 이동이 고도로 새어 들어간다.

    반환 (poses (S,F,4,4), char (F,3), src_cam (F,3)).
    """
    with open(path.join(path.dirname(meta["scene_graph"]),
                        path.basename(meta["scene_graph"])), encoding="utf-8") as file:
        graph = json.load(file)
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=np.float64)
    R_wg = T_wg[:3, :3] / float(graph["scale"]["S"])
    src = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64)
    rel0 = np.linalg.inv(src[0])                       # world -> 첫 context view

    def lift(points_g):                                # (…,3) G -> rel
        world = np.einsum("ij,...j->...i", T_wg,
                          np.pad(points_g, [(0, 0)] * (points_g.ndim - 1) + [(0, 1)],
                                 constant_values=1.0))
        return np.einsum("ij,...j->...i", rel0, world)[..., :3]

    # 중력 정렬 기저 (rel 성분). OpenCV 규약을 유지해 **(right, down, fwd)** 로 짠다 — up 을
    # 2번 축으로 쓰면 det −1 (거울상) 이 되어 각도가 안 보존된다. down 은 world up 을 rel 로
    # 옮겨 부호를 뒤집은 것, fwd 는 소스 frame0 시선을 그 수평면에 투영한 것이라 좌우가 렌더
    # 화면과 어긋나지 않는다. right 는 cross 로 잡아야 직교가 보장된다 (fwd 와 따로 투영하면
    # 서로 직교가 아니다 — 실측: 시선각 median 이 9.2° -> 11.4° 로 틀어졌다).
    down = -np.asarray(graph["gravity"]["up_world"], dtype=np.float64) @ src[0][:3, :3]
    down /= np.linalg.norm(down)
    fwd = np.array([0.0, 0.0, 1.0]) - down[2] * down
    fwd /= np.linalg.norm(fwd)
    R_plot = np.stack([np.cross(down, fwd), down, fwd]) if align_gravity else np.eye(3)
    assert abs(np.linalg.det(R_plot) - 1.0) < 1e-9, np.linalg.det(R_plot)

    poses_g = np.stack([et_to_opencv_c2w(p, data["R_et_w"], t)
                        for p, t in zip(data["poses_et"], data["cam_t_g"])])
    poses = np.tile(np.eye(4), poses_g.shape[:2] + (1, 1))
    poses[..., :3, :3] = np.einsum("ij,jk,kl,sflm->sfim", R_plot, rel0[:3, :3], R_wg,
                                   poses_g[..., :3, :3])
    poses[..., :3, 3] = lift(poses_g[..., :3, 3]) @ R_plot.T
    return poses, lift(np.asarray(data["char_g"], dtype=np.float64)) @ R_plot.T, \
        np.einsum("ij,fjk->fik", rel0, src)[:, :3, 3] @ R_plot.T


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)      # <dir>[:<라벨>] 반복
    parser.add_argument("--out", required=True)                  # mp4 경로
    parser.add_argument("--max_seeds", type=int, default=6)      # 열당 그릴 궤적 수
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stick", type=float, default=0.35)     # 시선 막대 길이 (space 단위)
    parser.add_argument("--space", choices=list(PLANES), default="et")   # 좌표계
    parser.add_argument("--planes", nargs="+", default=["top", "side"])  # 행으로 깔 시점
    parser.add_argument("--source_cam", action="store_true", default=True)   # 소스 카메라 궤적
    parser.add_argument("--no_source_cam", dest="source_cam", action="store_false")
    parser.add_argument("--aim_line", action="store_true")       # 카메라->subject 보조선
    parser.add_argument("--flip_y", action="store_true")         # xz 평면 기준 반전 (y -> -y)
    args = parser.parse_args()

    planes = [PLANES[args.space][name] for name in args.planes]

    runs = []
    for spec in args.runs:
        run_dir, _, label = spec.partition(":")
        data, meta = load_run(run_dir)
        if args.space == "et":
            poses, char, src = data["poses_et"], data["char_et_m"], None
        else:
            poses, char, src = to_rel(data, meta, align_gravity=args.space == "grav")
        if args.flip_y:                                  # xz 평면 기준 거울상 (det -1, 시각화 전용)
            mirror = np.diag([1.0, -1.0, 1.0])
            poses = poses.copy()
            poses[..., :3, :3] = np.einsum("ij,sfjk->sfik", mirror, poses[..., :3, :3])
            poses[..., :3, 3] = poses[..., :3, 3] @ mirror
            char, src = char @ mirror, (None if src is None else src @ mirror)
        runs.append((label or path.basename(run_dir), poses[: args.max_seeds], char,
                     src if args.source_cam else None))

    num_frames = min(r[1].shape[1] for r in runs)
    pts = np.concatenate([r[1][:, :, :3, 3].reshape(-1, 3) for r in runs]
                         + [r[2] for r in runs]
                         + [r[3] for r in runs if r[3] is not None])
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.12 * max(hi - lo)
    lo, hi = lo - pad, hi + pad
    frame_desc = {"et": "E.T. world, y-up, char@frame0 = origin  [m]",
                  "rel": "rel to first context view (source c2w[0] = I), OpenCV  [DA3 units]",
                  "grav": "gravity-aligned, origin = source cam frame0, "
                          "axes (right, up, fwd) of source view  [DA3 units]"}[args.space]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    frames = []
    for f in range(num_frames):
        fig, axes = plt.subplots(len(planes), len(runs),
                                 figsize=(4.6 * len(runs), 4.2 * len(planes)), squeeze=False)
        for col, (label, poses, char, src) in enumerate(runs):
            for row, (title, (ax_i, si), (ax_j, sj)) in enumerate(planes):
                ax = axes[row][col]
                ax.plot(si * char[:, ax_i], sj * char[:, ax_j], color="0.75", lw=1.2, zorder=1,
                        label="subject track")
                ax.scatter(si * char[f, ax_i], sj * char[f, ax_j], s=90, color="k",
                           marker="*", zorder=5)
                if src is not None:                              # 소스 카메라 (첫 프레임 = 원점)
                    ax.plot(si * src[:, ax_i], sj * src[:, ax_j], color="0.35",
                            lw=1.2, ls="--", zorder=2, label="source camera")
                    ax.scatter(si * src[f, ax_i], sj * src[f, ax_j], s=34, color="0.15",
                               marker="s", zorder=5)
                for s in range(poses.shape[0]):
                    t = poses[s, :, :3, 3]
                    c = colors[s % len(colors)]
                    ax.plot(si * t[:, ax_i], sj * t[:, ax_j], color=c, lw=0.8, alpha=0.35, zorder=2)
                    ax.plot(si * t[: f + 1, ax_i], sj * t[: f + 1, ax_j], color=c, lw=1.6, zorder=3)
                    if args.aim_line:                            # 카메라 -> subject 보조선
                        ax.plot([si * t[f, ax_i], si * char[f, ax_i]],
                                [sj * t[f, ax_j], sj * char[f, ax_j]], color=c, lw=0.7,
                                ls=":", alpha=0.6, zorder=3)
                    fwd = poses[s, f, :3, 2] * args.stick        # col2 = 시선 (실측 확인)
                    ax.plot([si * t[f, ax_i], si * (t[f, ax_i] + fwd[ax_i])],
                            [sj * t[f, ax_j], sj * (t[f, ax_j] + fwd[ax_j])], color=c, lw=1.8,
                            zorder=4)
                    ax.scatter(si * t[f, ax_i], sj * t[f, ax_j], s=22, color=c, zorder=4)
                ax.set_xlim(*sorted((si * lo[ax_i], si * hi[ax_i])))
                ax.set_ylim(*sorted((sj * lo[ax_j], sj * hi[ax_j])))
                ax.set_aspect("equal")
                ax.grid(alpha=0.25, lw=0.4)
                ax.tick_params(labelsize=7)
                if row == 0:
                    ax.set_title(f"{label}\n{title}", fontsize=10)
                else:
                    ax.set_title(title, fontsize=9)
                if row == 0 and col == 0:                        # 회색 두 선이 뭔지 한 번만
                    ax.legend(fontsize=7, loc="best", framealpha=0.7)
        fig.suptitle(f"DIRECTOR (ca-mixed-e449) — {frame_desc}"
                     f"   |   frame {f + 1}/{num_frames}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(buf[: buf.shape[0] // 2 * 2, : buf.shape[1] // 2 * 2].copy())
        plt.close(fig)

    imageio.mimwrite(args.out, frames, fps=args.fps, codec="libx264",
                     quality=6, macro_block_size=1)
    print(f"{'frames':<14}{len(frames)}")
    print(f"{'saved':<14}{args.out}")


if __name__ == "__main__":
    main()
