"""τ 손잡이를 바꿀 때 preset 궤적의 **모양**이 어떻게 변하는지 top-down 으로 겹쳐 그린다.

WHY: 뱅크의 τ 사다리(`hole0.1/0.2/0.35/0.5`)는 hole 목표를 τ 로 푸는 이분법이라, 충돌·접근·
elevation 게이트에 먼저 걸리면 **네 단이 전부 같은 knob 으로 접힌다** (실측 `tru_00add26c_a01`
`orbit_left`: 12 anchor 전부 4단 knob 동일). 그래서 배포 뱅크만 봐서는 "τ 를 올리면 orbit 이
어떻게 변하나"를 볼 수 없다. 여기서는 사다리를 무시하고 `target_tau` 를 직접 훑어 궤적을
되만든다 (`sample_camera_bank.make_decision` → `decode.build_poses`, 뱅크 `fixed` 그대로).

orbit 은 τ 가 **반경이 아니라 sweep 각**으로 들어간다 — `shape_context` 가 `obs_az_span` 에서
sweep 을 정하고 `fit_tau` 가 그 각을 배율로 깎는다. 반경은 시작 pose와 subject 거리로 이미
닫혀 있다. 그래서 top-down 에서 곡선들은 **같은 원 위의 서로 다른 호**로 보여야 정상이고,
반경이 같이 변하면 그게 이상 신호다.

그림 규약 (graph frame G, up = +z 라 xy 가 수평면):
  · 회색 점선   소스 카메라 경로 (frame 0 에 동그라미)
  · 검정 사각형 anchor OBB footprint, 검정 점선이 subject track
  · 색 곡선     τ 하나당 하나. 색은 **요청 τ**, 굵은 점이 frame 0
  · 아래 축     요청 τ vs 실측 τ(`tau_per_frame.max()`) — 포화하면 여기서 꺾인다

env: `da3` / `vista4d` 아무거나 (GPU 안 쓴다)

사용 예시:
    # orbit 3종, τ 0.1~3.0 10단
    python viz/viz_tau_shape_topdown.py --out_root out_trumans \\
      --video tru_00add26c_a01_s3f0k6 --anchor dyn_0 \\
      --out results/20260901_tau_shape

    # preset 과 τ 를 직접 지정
    python viz/viz_tau_shape_topdown.py --video tru_00add26c_a01_s3f0k6 \\
      --anchor stat_2 --presets orbit_left,push_in_arc_left \\
      --taus 0.1,0.25,0.5,1.0,2.0 --out /tmp/tau_shape
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                 # noqa: E402
from matplotlib.cm import ScalarMappable                                        # noqa: E402
from matplotlib.colors import LogNorm                                           # noqa: E402

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import build_poses                                      # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402
from fit.bank.emit_bank import FIXED_FALLBACK, span_frac_for                     # noqa: E402
from fit.bank.sample_camera_bank import make_decision, register_external         # noqa: E402


def to_graph(points_world: np.ndarray, T_gw: np.ndarray):
    """world (N,3) → graph frame G. `T_gw` 는 4x4 라 스케일까지 같이 들어간다."""
    p = np.asarray(points_world, dtype=float)
    return p @ T_gw[:3, :3].T + T_gw[:3, 3]


def obb_footprint(node: dict):
    """OBB 밑면 4점 (G, xy). yaw 로 돌린 사각형이라 top-down 에서 그대로 쓴다."""
    center = np.asarray(node["obb"]["center"], dtype=float)
    extent = np.asarray(node["obb"]["extent"], dtype=float)
    rotation = np.asarray(node["obb"]["R"], dtype=float)
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    local = np.stack([signs[:, 0] * extent[0] / 2, signs[:, 1] * extent[1] / 2,
                      np.zeros(len(signs))], axis=-1)
    return (local @ rotation.T + center)[:, :2]


def build_at_tau(graph, node, preset: str, tau: float, fixed: dict, num_frames: int):
    """τ 하나 → (poses world (F,4,4), extra). 뱅크 `fixed` 를 그대로 따른다."""
    decision = make_decision(graph, node, preset, tau, fixed["speed"], fixed["tracking"],
                             float(fixed["look_at_bias"]), fixed["start_mode"],
                             tau_ref=str(fixed.get("tau_ref", "source")))
    return build_poses(
        decision, graph, board=None, num_frames=num_frames,
        orbit_span_frac=span_frac_for(node, fixed), start_mode=fixed["start_mode"],
        aim_anchor=fixed["aim_anchor"],
        aim_ramp_frames=int(fixed.get("aim_ramp_frames", FIXED_FALLBACK["aim_ramp_frames"])),
        traj_basis=str(fixed.get("traj_basis", FIXED_FALLBACK["traj_basis"])),
        aim_keyframes=int(fixed.get("aim_keyframes", FIXED_FALLBACK["aim_keyframes"])),
        keyframe_aim=str(fixed.get("keyframe_aim", FIXED_FALLBACK["keyframe_aim"])),
        keyframe_ease=str(fixed.get("keyframe_ease", FIXED_FALLBACK["keyframe_ease"])),
        smooth_passes=int(fixed.get("smooth_passes", FIXED_FALLBACK["smooth_passes"])),
        smooth_lambda=float(fixed.get("smooth_lambda", FIXED_FALLBACK["smooth_lambda"])),
        deroll=bool(fixed.get("deroll", FIXED_FALLBACK["deroll"])))


def sweep_preset(graph, node, preset: str, taus, fixed: dict, num_frames: int, T_gw):
    """preset 하나 × τ 전량 → 행 리스트. 실측 τ / 경로길이 / 반경 통계를 같이 담는다."""
    center0 = np.asarray(node["track"]["center_smooth"], dtype=float)[0][:2]
    rows = []
    for tau in taus:
        poses, extra = build_at_tau(graph, node, preset, tau, fixed, num_frames)
        p_g = to_graph(poses[:, :3, 3], T_gw)
        step = np.linalg.norm(np.diff(p_g, axis=0), axis=1)
        radius = np.linalg.norm(p_g[:, :2] - center0, axis=1)
        rows.append({"tau_req": float(tau),
                     "tau_meas": float(np.max(extra["tau_per_frame"])),
                     "path_len": float(step.sum()),
                     "radius_first": float(radius[0]), "radius_last": float(radius[-1]),
                     "radius_spread": float(radius.max() - radius.min()),
                     # 시작 반경 기준으로 몇 도를 돌았나 (orbit 이면 이게 sweep 이다)
                     "az_span_deg": float(np.degrees(np.ptp(np.unwrap(np.arctan2(
                         p_g[:, 1] - center0[1], p_g[:, 0] - center0[0]))))),
                     "path_g": p_g})
    return rows


def draw(axis, rows, node, cam_g, norm, cmap, title: str):
    """top-down 한 칸. 소스/OBB/track 을 깔고 그 위에 τ 곡선을 겹친다.

    **τ 내림차순으로 그린다.** orbit 은 τ 가 커져도 반경이 그대로라 곡선들이 같은 원 위의
    호가 되고, 오름차순으로 그리면 제일 큰 호가 나머지를 전부 덮어 한 줄만 보인다. 큰 것부터
    깔고 작은 것을 위에 얹으면 호가 어디서 끝나는지(=τ 가 만든 sweep)가 그대로 보인다.
    끝점(frame 48)에 속 빈 원을 찍어 호의 길이를 눈으로 셀 수 있게 한다.
    """
    axis.plot(cam_g[:, 0], cam_g[:, 1], color="0.55", lw=1.2, ls="--", zorder=1)
    axis.scatter(cam_g[0, 0], cam_g[0, 1], s=28, facecolors="none",
                 edgecolors="0.35", lw=1.2, zorder=2)
    track = np.asarray(node["track"]["center_smooth"], dtype=float)[:, :2]
    axis.plot(track[:, 0], track[:, 1], color="k", lw=1.0, ls=":", zorder=2)
    foot = obb_footprint(node)
    axis.plot(foot[:, 0], foot[:, 1], color="k", lw=1.4, zorder=3)

    order = sorted(range(len(rows)), key=lambda i: -rows[i]["tau_req"])
    for depth, index in enumerate(order):
        row = rows[index]
        color = cmap(norm(row["tau_req"]))
        p = row["path_g"]
        axis.plot(p[:, 0], p[:, 1], color=color, lw=3.2 - 2.0 * depth / max(len(order) - 1, 1),
                  zorder=4 + depth, solid_capstyle="round")
        axis.scatter(p[-1, 0], p[-1, 1], s=22, facecolors="none", edgecolors=color,
                     lw=1.2, zorder=4 + depth)
    axis.scatter(rows[0]["path_g"][0, 0], rows[0]["path_g"][0, 1], s=42, marker="*",
                 color="crimson", zorder=40)                    # frame 0 은 τ 와 무관하게 같다
    # dr/r 는 "이게 아직 orbit 인가"다. `fit_tau` 는 rel 궤적 **전체**를 배율로 키우므로
    # sweep 이 좁으면 호가 넓어지는 대신 원에서 떨어져 나간다 — 그때 이 값이 커진다.
    purity = max(r["radius_spread"] for r in rows) / max(rows[0]["radius_first"], 1e-9)
    axis.set_title(f"{title}   r={rows[0]['radius_first']:.2f}u   dr/r_max={purity:.2f}",
                   fontsize=9)
    axis.set_aspect("equal")
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.25, lw=0.5)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default="out_trumans", type=str)   # 뱅크/그래프 루트
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--anchor", default="", type=str)                # 쉼표 구분, 비면 dyn_* 전량
    parser.add_argument("--bank_dir", default="hole_bank_k6_d77", type=str)
    parser.add_argument("--presets", default="orbit_left,orbit_right,orbit_left_pedestal_up",
                        type=str)
    parser.add_argument("--taus", default="", type=str)                  # 쉼표 구분, 비면 로그 등간
    parser.add_argument("--tau_lo", default=0.1, type=float)
    parser.add_argument("--tau_hi", default=3.0, type=float)
    parser.add_argument("--num_taus", default=10, type=int)
    # τ 하한을 `tau_start`(=카메라를 frame0 에 얼려도 소스가 움직여 쌓이는 τ) 위로 올린다.
    # 안 올리면 하한 쪽 절반이 전부 "예산 초과 → 궤적 0" 이라 그림에 점 하나로만 찍힌다
    # (`fit_hole_ladder` 의 `--tau_floor_src` 와 같은 이유).
    parser.add_argument("--auto_lo", dest="auto_lo", action="store_true")
    parser.add_argument("--no_auto_lo", dest="auto_lo", action="store_false")
    parser.set_defaults(auto_lo=True)
    parser.add_argument("--out", default="", type=str)                   # 결과 폴더 (비면 저장 안 함)
    args = parser.parse_args()

    root = args.out_root if path.isabs(args.out_root) \
        else path.join(CINEMATRAJ_ROOT, args.out_root)
    graph = load_graph(path.join(root, args.video, "scene_graph.json"))
    bank = json.load(open(path.join(root, args.video, args.bank_dir, "bank.json")))
    ext = bank.get("external_shapes") or {}
    if ext.get("path"):
        register_external(ext["path"], ext.get("aim") or "traj")

    fixed = dict(bank["fixed"])
    num_frames = int(fixed.get("num_frames", bank["num_frames"]))
    T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
    cam_g = np.asarray(graph["cameras"]["cam_centers_g"], dtype=float)
    nodes = {n["id"]: n for n in graph["nodes"]}
    anchors = [a for a in args.anchor.split(",") if a] or \
        [n["id"] for n in graph["nodes"] if n["id"].startswith("dyn_")]
    presets = [p for p in args.presets.split(",") if p]
    # tau_start 는 anchor·preset 과 무관한 씬 상수라 (`fit_hole_ladder` docstring) 한 번만 잰다.
    probe = build_at_tau(graph, nodes[anchors[0]], presets[0], 1e-4, fixed, num_frames)[1]
    tau_start = float(np.max(probe["tau_per_frame"]))
    tau_lo = args.tau_lo
    if args.auto_lo and tau_lo <= tau_start:
        tau_lo = tau_start * 1.02
    taus = [float(t) for t in args.taus.split(",") if t] or \
        list(np.geomspace(tau_lo, args.tau_hi, args.num_taus))
    print(f"tau_start {tau_start:.4f} (소스 시차만으로 쌓이는 τ)  ·  sweep "
          f"{min(taus):.3f}~{max(taus):.3f} × {len(taus)}단")

    norm = LogNorm(vmin=min(taus), vmax=max(taus))
    cmap = plt.get_cmap("viridis")
    results, out_dir = {}, ""
    if args.out:
        out_dir = args.out if path.isabs(args.out) else path.join(CINEMATRAJ_ROOT, args.out)
        makedirs(out_dir, exist_ok=True)

    for anchor in anchors:
        node = nodes[anchor]
        sweeps = {p: sweep_preset(graph, node, p, taus, fixed, num_frames, T_gw)
                  for p in presets}
        results[anchor] = {p: [{k: v for k, v in r.items() if k != "path_g"} for r in rows]
                           for p, rows in sweeps.items()}

        figure, axes = plt.subplots(2, len(presets), figsize=(5.4 * len(presets), 10.0),
                                    squeeze=False,
                                    gridspec_kw={"height_ratios": [2.6, 1.0],
                                                 "wspace": 0.42, "hspace": 0.22})
        for column, preset in enumerate(presets):
            rows = sweeps[preset]
            draw(axes[0][column], rows, node, cam_g, norm, cmap, preset)
            lower = axes[1][column]
            req = [r["tau_req"] for r in rows]
            lower.plot(req, [r["tau_meas"] for r in rows], "o-", color="C0", ms=3,
                       label="measured tau")
            lower.plot(req, req, ls="--", lw=0.8, color="0.6", label="y=x")
            twin = lower.twinx()
            twin.plot(req, [r["az_span_deg"] for r in rows], "s-", color="C3", ms=3,
                      label="az span")
            twin.set_ylabel("az span (deg)", fontsize=7, color="C3")
            twin.tick_params(labelsize=7, colors="C3")
            lower.set_xscale("log")
            lower.set_xlabel("requested tau", fontsize=8)
            lower.set_ylabel("measured tau", fontsize=8)
            lower.tick_params(labelsize=7)
            lower.grid(alpha=0.25, lw=0.5)
            if column == 0:
                lower.legend(fontsize=6, loc="upper left")

        bar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(),
                              fraction=0.03, pad=0.02)
        bar.set_label("requested tau", fontsize=8)
        bar.ax.tick_params(labelsize=7)
        figure.suptitle(f"{args.video}  ·  {anchor} ({node.get('label')})  "
                        f"·  top-down (graph frame G, 1 unit = S)  "
                        f"·  obs_az_span {node['obs_az_span_deg']:.1f}deg  "
                        f"·  tau_start {tau_start:.3f}\n"
                        # matplotlib 기본 폰트(DejaVu Sans)에 한글 글리프가 없다 — 그림 안 문자열은
                        # 전부 ASCII 로. 설명은 이 스크립트 docstring 에 한국어로 있다.
                        f"star = frame 0 (same for all tau) | open circle = frame 48 | "
                        f"grey dashed = source camera | black box = anchor OBB footprint",
                        fontsize=9)
        if out_dir:
            target = path.join(out_dir, f"tau_shape_{args.video}__{anchor}.png")
            figure.savefig(target, dpi=140, bbox_inches="tight")
            print(f"-> {target}")
        plt.close(figure)

    print(f"\n{'anchor':10s} {'preset':24s} {'tau_req':>8s} {'tau_meas':>9s} "
          f"{'path_u':>8s} {'az_deg':>8s} {'r_first':>8s} {'r_spread':>9s}")
    for anchor, per_preset in results.items():
        for preset, rows in per_preset.items():
            for r in rows:
                print(f"{anchor:10s} {preset:24s} {r['tau_req']:8.3f} {r['tau_meas']:9.3f} "
                      f"{r['path_len']:8.3f} {r['az_span_deg']:8.2f} "
                      f"{r['radius_first']:8.3f} {r['radius_spread']:9.4f}")

    if out_dir:
        target = path.join(out_dir, f"tau_shape_{args.video}.json")
        json.dump({"format": "tau_shape_v1", "video": args.video, "bank_dir": args.bank_dir,
                   "fixed": fixed, "taus": taus, "presets": presets, "sweeps": results},
                  open(target, "w"), ensure_ascii=False, indent=1)
        print(f"-> {target}")


main()
