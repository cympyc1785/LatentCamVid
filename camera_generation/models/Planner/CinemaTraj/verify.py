"""디코드된 49프레임 궤적을 **실제로 렌더해서** 판정하고, 사람이 볼 프리뷰 6종을 뽑는다.

이 파일의 존재 이유는 계획서 §Verification 한 줄이다 — "1(hole_fraction)·6(tau_max) 이
'생성에 GPU 를 쓸 값어치가 있나'를 결정한다". 폐기된 CinemaTraj 계획은 그 두 값을 SDF 손실항으로
**예측**하려 했다. 여기서는 예측하지 않고 49프레임을 Vista4D 로 렌더해서 잰다. 대리지표가 필요
없다는 게 look-before-move 를 궤적까지 밀어붙인 이유고, 그 확인이 여기다.

게이트(`lbm/gates.py`)와 무엇이 다른가: 게이트는 **frame 0 한 장**을 보고 후보를 거른다. 여기는
결정이 끝난 뒤 **49프레임 전부**를 본다. frame 0 에서 통과한 pose 가 frame 48 에서 벽을 뚫거나
subject 를 놓치는 건 게이트가 구조적으로 못 본다 — preset 프리뷰가 7프레임만 확인하기 때문이다.
그래서 4번(behind_surface)과 2번(subject_in_frame)은 여기서 처음으로 전 프레임 판정을 받는다.

지표 8종은 계획서 표 그대로다. PASS/WARN/FAIL 은 `THRESHOLDS` 한 곳에만 적혀 있다.

  1 hole_fraction          49프레임 평균 `1 - valid.mean()`   하류 모델이 채워야 할 구멍
  2 subject_in_frame       중앙 `center_box` 안에 subject 중심이 있는 프레임 비율
  3 subject_pixel_coverage subject 실루엣 면적비의 median
  4 behind_surface_frames  G1 을 어기는 플랜 프레임 수 (0 이 아니면 카메라가 표면 뒤)
  5 max_view_angle_delta   소스 시선과의 최대 각차 (poses.npz 에 이미 있다)
  6 tau_max                |Δp| / z_med 최대 (poses.npz)
  7 jerk_max               소스 jerk p95 대비 배율 — 흔들림
  8 roundtrip              resample(rel21, 49) vs rel_full_unit, 그리고 |rel[0]-I|
  9 path_len_u             카메라 이동 거리 합 (u). 계획서에 없던 지표 — 1~8 에 "움직여야
                           한다"가 없어서 `static_hold_dont_look`(coverage 최대화 퇴화 해)가
                           전 지표 PASS 로 통과한다. `--min_path_len_u 0` 이면 안 잰다.

**환경**: 계획서 표에는 verify 가 `da3` 로 적혀 있지만 실제로는 `CloudRenderer` 가 필요하다.
`vista4d` 에서 돌린다 (loop 와 같은 env).

예시:
    CUDA_VISIBLE_DEVICES=1 python verify.py --video camel
    CUDA_VISIBLE_DEVICES=1 python verify.py --video camel --no_previews   # 지표만
"""
import json
import shutil
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.abspath(__file__))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT, subject_point_mask  # noqa: E402
from lbm.gates import behind_surface_frames                                        # noqa: E402
from lbm.overlay import paint_holes                                                # noqa: E402
from lbm.presets import RECAMMASTER_ROOT                                           # noqa: E402
from lbm.render import CloudRenderer                                               # noqa: E402
from scene_graph.io import load_scene                                              # noqa: E402
from scene_graph.lift import apply_transform                                       # noqa: E402
from scene_graph.obb import node_obb_at, obb_corners                               # noqa: E402
from scripts.build_candidate_board import subject_track_volume                     # noqa: E402

if RECAMMASTER_ROOT not in sys.path:
    sys.path.insert(0, RECAMMASTER_ROOT)
from emit_model_cams import resample  # noqa: E402

RESULTS_DEFAULT = "/data1/cympyc1785/LatentCamVid/video_generation/results"

# 계획서 §Verification 표. (PASS 상한, WARN 상한) — 값이 작을수록 좋은 지표는 그대로,
# 클수록 좋은 지표(`_HIGHER_IS_BETTER`)는 부호를 뒤집어 같은 비교를 탄다.
THRESHOLDS = {
    "hole_fraction":          (0.35, 0.55),
    "subject_in_frame":       (0.95, 0.85),
    "max_view_angle_delta":   (40.0, 50.0),
    "tau_max":                (0.25, 0.30),
    "jerk_ratio":             (3.0, 5.0),
}
_HIGHER_IS_BETTER = {"subject_in_frame"}
# 구간 지표 — PASS 구간을 벗어나면 바로 FAIL (WARN 이 없다).
BAND = {"subject_pixel_coverage": (0.02, 0.45)}
# 0 이 아니면 FAIL.
ZERO = ("behind_surface_frames",)
ROUNDTRIP_MAX = 0.02      # resample(rel21,49) vs rel_full_unit
ANCHOR_MAX = 1e-9         # |rel[0] - I|
# 지표 9 — 카메라가 실제로 움직였나. PASS 하한(u), WARN 은 그 절반. 0 이면 지표를 끈다.
MIN_PATH_LEN_U = 0.05


def grade(name: str, value):
    """지표 하나의 PASS/WARN/FAIL. 임계는 전부 위 상수에서만 온다."""
    if name in ZERO:
        return "PASS" if value == 0 else "FAIL"
    if name in BAND:
        lo, hi = BAND[name]
        return "PASS" if lo <= value <= hi else "FAIL"
    if name in THRESHOLDS:
        good, warn = THRESHOLDS[name]
        if name in _HIGHER_IS_BETTER:
            return "PASS" if value >= good else ("WARN" if value >= warn else "FAIL")
        return "PASS" if value <= good else ("WARN" if value <= warn else "FAIL")
    return "PASS" if value else "FAIL"


def jerk_series(positions: np.ndarray):
    """프레임당 jerk 크기 (위치의 3차 차분). 앞 3프레임은 정의가 안 돼 0 을 채운다."""
    positions = np.asarray(positions, dtype=float)
    third = np.diff(positions, n=3, axis=0)
    return np.concatenate([np.zeros(3), np.linalg.norm(third, axis=-1)])


# ─────────────────────────────────────────────────────────────────────── 렌더 + 지표

def render_plan(renderer: CloudRenderer, poses: np.ndarray, height=None, width=None,
                temporal_persistence: bool = True):
    """플랜 pose 를 프레임별로 렌더한다 — `poses[f]` 를 시간 `f` 의 점군으로.

    시간을 같이 진행시키는 게 핵심이다. 전 프레임을 frame 0 점군으로 그리면 동적 subject 가
    제자리에 얼어붙어서 subject 지표가 통째로 거짓말이 된다.

    `temporal_persistence=False` 면 **1:1 depth warp** — 프레임 `f` 에서 유래한 점만 쓴다
    (`render.visible_at` 의 NTP 분기). 49프레임 누적을 끄는 것이라 점 수가 1/49 로 줄어
    렌더가 그만큼 싸다. `hole_fraction` 은 이 모드에서 **비교 불가**(누적분이 빠져 무조건
    커진다)지만 `subject_in_frame` 은 거의 불변이다 — 동적 점은 `visible.sum(1)==1` 이라
    어차피 자기 프레임에서만 보이고, 정적 subject 는 이웃 프레임 점이 빠져 실루엣만 조금
    얇아질 뿐 **중심**이 안 움직인다. 기본 True = 기존 동작.
    """
    out = []
    for f in range(len(poses)):
        out.append(renderer.render(poses[f], frame=f, height=height, width=width,
                                   temporal_persistence=temporal_persistence))
    return out


def measure(rendered: list, center_box: float):
    """지표 1·2·3 의 프레임별 원자료."""
    holes, areas, centers, in_frame = [], [], [], []
    lo, hi = (1 - center_box) / 2, 1 - (1 - center_box) / 2
    for shot in rendered:
        holes.append(float(1.0 - shot["valid"].mean()))
        subject = shot["subject"]
        if subject is None or not subject.any():
            areas.append(0.0)
            centers.append(None)
            in_frame.append(False)
            continue
        areas.append(float(subject.mean()))
        ys, xs = np.nonzero(subject)
        cx, cy = float(xs.mean() / subject.shape[1]), float(ys.mean() / subject.shape[0])
        centers.append([round(cx, 4), round(cy, 4)])
        in_frame.append(bool(lo <= cx <= hi and lo <= cy <= hi))
    return (np.asarray(holes), np.asarray(areas), centers, np.asarray(in_frame))


def evaluate(video: str, poses_npz, canonical_npz, renderer, depths, sky_mask,
             scale: float, center_box: float, rendered: list,
             min_path_len_u: float = MIN_PATH_LEN_U):
    """지표 8종 + `path_len_u`(옵션). 반환 dict 는 그대로 `verify.json` 이 된다."""
    poses = poses_npz["cam_c2w"]
    holes, areas, centers, in_frame = measure(rendered, center_box)

    # 4 — 플랜 **전 프레임**을 소스 관측 표면에 되쏜다. 게이트는 frame 0 만 봤다.
    behind = []
    for f in range(len(poses)):
        hits = behind_surface_frames(poses[f][:3, 3], depths, K=renderer.K_src,
                                     cam_c2w=renderer.cam_c2w_src, sky_mask=sky_mask,
                                     scale=scale, frames=range(renderer.num_frames))
        if hits:
            behind.append(f)

    # 7 — 소스 카메라 자신의 jerk 를 분모로. 절대 임계는 씬 스케일에 휘둘린다.
    plan_jerk = jerk_series(poses[:, :3, 3])
    src_jerk = jerk_series(renderer.cam_c2w_src[:, :3, 3])
    src_p95 = float(np.percentile(src_jerk, 95))
    # 소스가 사실상 정지면 (camel parallax 0.0042) 분모가 0 에 가까워 배율이 발산한다.
    # 그때는 씬 스케일 대비 절대 jerk 로 바꿔 재고, 그 사실을 `jerk_denominator` 에 남긴다.
    degenerate = src_p95 < 1e-4 * scale
    denominator = (1e-2 * scale) if degenerate else src_p95
    jerk_ratio = float(plan_jerk.max() / max(denominator, 1e-12))

    # 8 — index-pick 의 역방향. emit 의 `roundtrip_error` 는 같은 인덱스를 도로 뽑아 항상 0 이라
    # 리샘플 손실을 못 본다. 여기서는 emitter 가 실제로 쓰는 SE(3) 측지 보간으로 되돌린다.
    rel21, rel_full = canonical_npz["rel"], canonical_npz["rel_full"]
    resampled = resample(rel21, len(rel_full))
    roundtrip = float(np.abs(resampled - rel_full).max())
    anchor = float(np.abs(rel21[0] - np.eye(4)).max())

    metrics = {
        "hole_fraction": float(holes.mean()),
        "subject_in_frame": float(in_frame.mean()),
        "subject_pixel_coverage": float(np.median(areas)),
        "behind_surface_frames": len(behind),
        "max_view_angle_delta": float(np.max(poses_npz["view_angle_deg"])),
        "tau_max": float(np.max(poses_npz["tau"])),
        "jerk_ratio": jerk_ratio,
    }
    rows = [{"name": k, "value": v, "status": grade(k, v)} for k, v in metrics.items()]
    rows.append({"name": "roundtrip_resample", "value": roundtrip,
                 "status": "PASS" if roundtrip < ROUNDTRIP_MAX else "FAIL"})
    rows.append({"name": "anchor_identity", "value": anchor,
                 "status": "PASS" if anchor < ANCHOR_MAX else "FAIL"})

    # 9 — 카메라가 실제로 움직였나. 나머지 8종에는 "움직여야 한다"가 없어서
    # `static_hold_dont_look`(coverage 를 최대화하는 퇴화 해)가 전 지표 PASS 로 통과한다.
    # 실측: avocado-slice 는 같은 설정 7 draw 중 6번 이걸 골랐고 `path_len_u = 0.0000`,
    # emit `--scales 0` 이었다. `--min_path_len_u 0` 을 주면 이 행이 안 생긴다 (예전 동작).
    path_len_u = float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1).sum()) / scale
    if min_path_len_u > 0:
        rows.append({"name": "path_len_u", "value": path_len_u,
                     "status": "PASS" if path_len_u >= min_path_len_u else
                               ("WARN" if path_len_u >= 0.5 * min_path_len_u else "FAIL")})

    worst = "FAIL" if any(r["status"] == "FAIL" for r in rows) else (
        "WARN" if any(r["status"] == "WARN" for r in rows) else "PASS")
    return {
        "format": "lbm_verify_v1", "video": video, "verdict": worst,
        "metrics": rows,
        "per_frame": {
            "hole_fraction": [round(float(v), 4) for v in holes],
            "subject_area": [round(float(v), 4) for v in areas],
            "subject_center": centers,
            "subject_in_frame": [bool(v) for v in in_frame],
            "tau": [round(float(v), 4) for v in poses_npz["tau"]],
            "view_angle_deg": [round(float(v), 2) for v in poses_npz["view_angle_deg"]],
            "jerk": [round(float(v), 6) for v in plan_jerk],
        },
        "behind_surface_frame_indices": behind,
        "jerk_denominator": {"source_p95": src_p95, "used": denominator,
                             "mode": "scene_scale" if degenerate else "source_p95"},
        "thresholds": {"THRESHOLDS": THRESHOLDS, "BAND": BAND, "ZERO": list(ZERO),
                       "roundtrip_max": ROUNDTRIP_MAX, "anchor_max": ANCHOR_MAX,
                       "center_box": center_box, "min_path_len_u": min_path_len_u},
    }


# ─────────────────────────────────────────────────────────────────────── 프리뷰

def write_video(output_path: str, frames, fps: float):
    """imageio + libx264. cv2 의 mp4v 는 VS Code 뷰어에서 안 열린다 (파일은 멀쩡해서 조용히 지나간다)."""
    import imageio.v2 as imageio
    imageio.mimwrite(output_path, list(frames), fps=fps, codec="libx264",
                     quality=6, macro_block_size=1)
    return output_path


def plan_render_video(rendered: list, output_path: str, fps: float):
    """구멍을 마젠타로 칠한 49프레임. 구멍이 어디에 얼마나 생기는지가 이 영상의 전부다."""
    return write_video(output_path, [paint_holes(s["rgb"], s["valid"]) for s in rendered], fps)


def plan_sbs_video(source: np.ndarray, rendered: list, output_path: str, fps: float):
    """소스 | 렌더. 렌더만 보면 구멍이 카메라 탓인지 소스 탓인지 안 갈린다."""
    import cv2
    frames = []
    for f, shot in enumerate(rendered):
        left = cv2.resize(source[f], (shot["width"], shot["height"]))
        pair = np.concatenate([left, paint_holes(shot["rgb"], shot["valid"])], axis=1)
        bar = np.zeros((28, pair.shape[1], 3), np.uint8)
        cv2.putText(bar, "SOURCE", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,) * 3, 1, cv2.LINE_AA)
        cv2.putText(bar, "PLAN (magenta = hole)", (shot["width"] + 8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,) * 3, 1, cv2.LINE_AA)
        frames.append(np.concatenate([bar, pair], axis=0))
    return write_video(output_path, frames, fps)


def plan_cam_video(graph: dict, node: dict, poses: np.ndarray, output_path: str, fps: float,
                   cam_detail: bool = True):
    """top/front/side 3면 궤적 애니메이션. 소스 회색 / 플랜 주황 / subject 점선.

    `make_camviz.py` 를 안 쓴다: 그건 ReCamMaster `trajectories.json` 의 **rel 공간** 궤적 한
    벌을 정적 PNG 로 그리는 도구라, ① 움직이는 소스 궤적 ② subject track 이라는 개념이 없고
    ③ 영상 출력이 없다. 세 가지를 다 넣으면 `draw_traj` 의 몸통이 남지 않는다.

    그리는 좌표계는 **G frame** 이다 (up = +z, 단위 u). world 로 그리면 기울어진 frame0 카메라가
    축이라 "top view" 가 top 이 아니게 된다 — 중력축을 scene graph 가 내놓는 이유가 그것이다.

    `cam_detail` 이 기본 True 인 이유: 카메라-subject 거리는 0.6 u 인데 카메라 운동은 0.02~0.15 u
    라(실측 camel/avocado-slice), 씬 전체를 한 눈금에 담으면 궤적이 점 하나로 뭉개진다. 위 줄은
    "카메라가 subject 어디쯤에 있나", 아래 줄은 "어떤 모양으로 움직이나" — 눈금이 다른 두 질문이라
    한 축으로 못 합친다.
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
    plan_g = apply_transform(T_gw, poses[:, :3, 3])
    source_g = np.asarray(graph["cameras"]["cam_centers_g"], dtype=float)
    track_g = np.asarray(node["track"]["center_smooth"], dtype=float)
    # 카메라 시선 (G). T_gw 는 1/S 를 품고 있어서 방향으로 쓰려면 정규화해야 한다.
    forward = lambda c2w: _unit(T_gw[:3, :3] @ c2w[:3, 2])
    plan_fwd = np.stack([forward(p) for p in poses])
    src_fwd = np.stack([forward(c) for c in
                        np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float)])

    # (이름, 가로축, 세로축) — G frame 은 up=+z 라 top 은 x-y, front 는 x-z, side 는 y-z.
    views = [("top (x-y)", 0, 1), ("front (x-z)", 0, 2), ("side (y-z)", 1, 2)]
    rows = [("scene", _extent(np.concatenate([plan_g, source_g, track_g]), pad=0.62, floor=0.4))]
    if cam_detail:
        rows.append(("camera detail",
                     _extent(np.concatenate([plan_g, source_g]), pad=0.75, floor=0.05)))

    frames = []
    for f in range(len(poses)):
        figure = plt.figure(figsize=(12, 4.2 * len(rows)), dpi=90)
        canvas = FigureCanvasAgg(figure)
        for r, (row_name, (center, half)) in enumerate(rows):
            tick = half * 0.18
            for i, (name, ax_h, ax_v) in enumerate(views):
                axes = figure.add_subplot(len(rows), 3, r * 3 + i + 1)
                axes.set_aspect("equal")
                axes.set_xlim(center[ax_h] - half, center[ax_h] + half)
                axes.set_ylim(center[ax_v] - half, center[ax_v] + half)
                axes.set_title(f"{row_name} — {name}", fontsize=10)
                axes.tick_params(labelsize=7)
                axes.grid(alpha=0.25, lw=0.5)

                axes.plot(track_g[:, ax_h], track_g[:, ax_v], "--", color="0.2", lw=1.0,
                          alpha=0.7, label="subject track")
                axes.plot(source_g[:, ax_h], source_g[:, ax_v], color="0.55", lw=1.6,
                          label="source cam")
                axes.plot(plan_g[:, ax_h], plan_g[:, ax_v], color="tab:orange", lw=1.8,
                          label="plan cam")

                # OBB 를 현재 프레임에서 이 평면에 투영한 사각형 — 크기 눈금.
                corners = obb_corners(*node_obb_at(node, f))
                axes.add_patch(plt.Rectangle(
                    (corners[:, ax_h].min(), corners[:, ax_v].min()),
                    np.ptp(corners[:, ax_h]), np.ptp(corners[:, ax_v]),
                    fill=False, ec="tab:cyan", lw=1.0, alpha=0.8))
                axes.plot(track_g[f, ax_h], track_g[f, ax_v], "o", color="0.2", ms=5)
                # 시선 tick 은 궤적선과 **다른 색**으로 긋는다. 같은 주황이면 궤적의 일부로 읽혀서
                # "카메라가 어디를 보나"가 그림에서 사라진다.
                for position, direction, color, edge in (
                        (source_g[f], src_fwd[f], "0.55", "0.15"),
                        (plan_g[f], plan_fwd[f], "tab:orange", "tab:red")):
                    axes.plot([position[ax_h], position[ax_h] + tick * direction[ax_h]],
                              [position[ax_v], position[ax_v] + tick * direction[ax_v]],
                              color=edge, lw=1.4, alpha=0.9, zorder=4)
                    axes.plot(position[ax_h], position[ax_v], "o", color=color, ms=7,
                              mec=edge, mew=1.2, zorder=5)
                if r == 0 and i == 0:
                    axes.legend(fontsize=7, loc="upper right", framealpha=0.7)
        figure.suptitle(f"frame {f:02d} / {len(poses) - 1}   "
                        f"(G frame, unit u; tick = view direction)", fontsize=11)
        figure.tight_layout()
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
        # libx264 는 짝수 해상도를 요구한다 (macro_block_size=1 이어도 홀수면 경고 + 재인코딩).
        frames.append(cv2.resize(image, (image.shape[1] // 2 * 2, image.shape[0] // 2 * 2)))
    return write_video(output_path, frames, fps)


def _extent(points: np.ndarray, pad: float, floor: float):
    """(center, half). 세 뷰가 **같은 눈금**을 쓰도록 축별이 아니라 전체 최대 범위로 잡는다."""
    points = np.asarray(points, dtype=float)
    return points.mean(0), max(float(np.ptp(points, axis=0).max()), floor) * pad


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def decision_summary(video: str, decision: dict, report: dict, canonical: dict,
                     gates_csv: str, traj_trace: str, output_path: str):
    """정렬표 + VLM 원문 + 게이트 거절 분포. **요약하지 않고 원문 그대로** 옮긴다."""
    tag = next(iter(canonical["cameras"]))
    meta = canonical["cameras"][tag]
    lines = [f"# {video}  —  lbm_verify_v1   verdict: {report['verdict']}", ""]

    lines.append("## DECISION")
    for key in ("format", "source", "model", "start_mode", "subject_id"):
        lines.append(f"{key:<24}{decision.get(key)}")
    lines.append(f"{'stage_sources':<24}{json.dumps(decision.get('stage_sources'), ensure_ascii=False)}")
    trajectory = decision["trajectory"]
    for key in ("preset", "speed", "tracking", "look_at_bias", "target_tau"):
        lines.append(f"{key:<24}{trajectory.get(key)}")
    lines.append("")

    lines.append("## METRICS")
    lines.append(f"{'metric':<26}{'value':>12}  status")
    for row in report["metrics"]:
        lines.append(f"{row['name']:<26}{row['value']:>12.6g}  {row['status']}")
    if report["behind_surface_frame_indices"]:
        lines.append(f"  behind-surface frames: {report['behind_surface_frame_indices']}")
    jerk = report["jerk_denominator"]
    lines.append(f"  jerk denominator: {jerk['mode']} = {jerk['used']:.6g} "
                 f"(source p95 = {jerk['source_p95']:.6g})")
    lines.append("")

    lines.append("## EMIT META")
    for key in ("model_gauge_rmax", "S_da3", "g", "tau_max", "target_tau",
                "view_angle_max_deg", "path_len_u", "roll_max_deg", "aim",
                "tracking_ignored", "translation_degenerate", "zoom_dropped", "free_start"):
        lines.append(f"{key:<24}{meta.get(key)}")
    lines.append("")

    if path.isfile(traj_trace):
        with open(traj_trace, encoding="utf-8") as file:
            stats = json.load(file)["stats"]
        lines.append("## PRESET BOARD (what the VLM chose from)")
        lines.append(f"{'preset':<22}{'move':>8}{'rot_deg':>9}{'tau_max':>9}{'tau_scale':>11}"
                     f"{'cov_min':>9}{'subj_min':>10}  excluded")
        for row in stats:
            lines.append(f"{row['preset']:<22}{row['move']:>8.4f}{row['rotation_deg']:>9.2f}"
                         f"{row['tau_max']:>9.4f}{row['tau_scale']:>11.4f}"
                         f"{row['coverage_min']:>9.4f}{row['subject_area_min']:>10.4f}  "
                         f"{row['excluded'] or ''}")
        lines.append("")

    if path.isfile(gates_csv):
        with open(gates_csv, encoding="utf-8") as file:
            header = file.readline().rstrip("\n").split(",")
            reasons = {}
            total = 0
            for line in file:
                total += 1
                failed = line.rstrip("\n").split(",")[header.index("failed")]
                reasons[failed or "passed"] = reasons.get(failed or "passed", 0) + 1
        lines.append("## GATE REJECTIONS (start-pose candidate pool)")
        lines.append(f"{'total candidates':<24}{total}")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"{reason:<24}{count}")
        lines.append("")

    vlm = decision.get("vlm", {})
    lines.append("## VLM (verbatim)")
    for key in ("observation", "reasoning"):
        lines.append(f"### {key}")
        lines.append(str(vlm.get(key, "")).strip())
        lines.append("")
    lines.append(f"confidence {vlm.get('confidence')}   turns {vlm.get('turns')}   "
                 f"repairs {vlm.get('repairs')}")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return output_path


def copy_previews(folder: str, results: str):
    """이미 만들어져 있는 프리뷰 3종을 결과 폴더로. 다시 그리지 않는다 — **VLM 이 본 그림 그대로**."""
    copied = []
    for source, name in (("vis/obb_overlay.mp4", "obb_overlay.mp4"),
                         ("board/board_candidates.png", "board_candidates.png"),
                         ("board/source_frames.png", "source_frames.png"),
                         ("board/contract.txt", "contract.txt"),
                         ("board/gates.csv", "gates.csv"),
                         ("trace/board_presets.png", "board_presets.png")):
        src = path.join(folder, source)
        if path.isfile(src):
            shutil.copy2(src, path.join(results, name))
            copied.append(name)
    return copied


# ─────────────────────────────────────────────────────────────────────── main

def main(args):
    import imageio.v2 as imageio

    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    folder = path.join(out_root, args.video)
    for name in ("cloud.npz", "scene_graph.json", "decision.json", "poses.npz",
                 "canonical/canonical.json", "canonical/canonical.npz"):
        assert path.isfile(path.join(folder, name)), \
            f"{name} 이 없다. 먼저 loop -> decode/emit 을 돌릴 것 ({folder})"

    with open(path.join(folder, "scene_graph.json"), encoding="utf-8") as file:
        graph = json.load(file)
    with open(path.join(folder, "decision.json"), encoding="utf-8") as file:
        decision = json.load(file)
    with open(path.join(folder, "canonical", "canonical.json"), encoding="utf-8") as file:
        canonical = json.load(file)
    poses_npz = dict(np.load(path.join(folder, "poses.npz"), allow_pickle=True))
    canonical_npz = dict(np.load(path.join(folder, "canonical", "canonical.npz")))
    node = next(n for n in graph["nodes"] if n["id"] == decision["subject_id"])

    # poses.npz 가 지금 decision.json 것인지 확인한다 — emit 과 **같은 지문**을 쓴다.
    # 안 하면 옛 궤적을 새 결정인 척 채점하게 되고, 숫자가 그럴듯해서 안 들킨다.
    from decode.build_poses import decision_fingerprint, resolve_start_mode
    expected = decision_fingerprint(decision, resolve_start_mode(decision, args.start_mode),
                                    args.aim_anchor, args.aim_ramp_frames)
    stored = str(poses_npz["decision_fingerprint"])
    assert stored == expected or args.allow_stale_poses, (
        f"poses.npz 가 decision.json 과 안 맞는다 (build_poses 를 다시 돌릴 것)\n"
        f"  poses.npz : {stored}\n  decision  : {expected}")

    renderer = CloudRenderer(path.join(folder, "cloud.npz"),
                             vista4d_root=args.vista4d_root, device=args.device,
                             fixed_focal=args.fixed_focal)
    recon = load_scene(args.eval_data, args.video, args.vista4d_root,
                       seg_root=args.seg_root, seg_static_root=args.seg_static_root)
    renderer.set_subject(subject_point_mask(
        renderer.indices, subject_track_volume(recon, node)).cpu().numpy())

    poses = poses_npz["cam_c2w"]
    assert len(poses) == renderer.num_frames, \
        f"poses {len(poses)} vs cloud {renderer.num_frames} 프레임 불일치"
    rendered = render_plan(renderer, poses, height=args.height, width=args.width)

    report = evaluate(args.video, poses_npz, canonical_npz, renderer,
                      recon["depths"], recon["sky_mask"], float(graph["scale"]["S"]),
                      args.center_box, rendered, min_path_len_u=args.min_path_len_u)
    report["start_mode"] = decision.get("start_mode")
    report["preset"] = decision["trajectory"]["preset"]
    report["render"] = {"height": rendered[0]["height"], "width": rendered[0]["width"],
                        "frames": len(rendered), "temporal_persistence": True}
    with open(path.join(folder, "verify.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    previews = []
    if args.previews:
        results = path.join(args.results_root, f"{args.date}_lbm_lite", args.video)
        makedirs(results, exist_ok=True)
        previews += copy_previews(folder, results)
        fps = renderer.fps
        previews.append(path.basename(plan_render_video(
            rendered, path.join(results, "plan_render.mp4"), fps)))
        source = np.stack(imageio.mimread(
            path.join(args.eval_data, "eval_data", "recon_and_seg", args.video, "video.mp4"),
            memtest=False)[:renderer.num_frames])
        previews.append(path.basename(plan_sbs_video(
            source, rendered, path.join(results, "plan_sbs.mp4"), fps)))
        previews.append(path.basename(plan_cam_video(
            graph, node, poses, path.join(results, "plan_cam.mp4"), fps,
            cam_detail=args.cam_detail)))
        previews.append(path.basename(decision_summary(
            args.video, decision, report, canonical,
            path.join(folder, "board", "gates.csv"),
            path.join(folder, "trace", "traj.json"),
            path.join(results, "decision_summary.txt"))))

    print(f"\n{args.video}  preset {report['preset']}  start_mode {report['start_mode']}  "
          f"{report['render']['frames']}f {report['render']['width']}x{report['render']['height']}")
    print(f"\n{'metric':<26}{'value':>12}  status")
    for row in report["metrics"]:
        print(f"{row['name']:<26}{row['value']:>12.6g}  {row['status']}")
    if report["behind_surface_frame_indices"]:
        print(f"  behind-surface frames: {report['behind_surface_frame_indices']}")
    jerk = report["jerk_denominator"]
    print(f"  jerk denominator: {jerk['mode']} = {jerk['used']:.6g} "
          f"(source p95 = {jerk['source_p95']:.6g})")
    print(f"\n{'verdict':<26}{report['verdict']}")
    print(f"{'verify.json':<26}{path.join(folder, 'verify.json')}")
    if previews:
        print(f"{'previews':<26}{path.join(args.results_root, f'{args.date}_lbm_lite', args.video)}")
        for name in previews:
            print(f"{'':<26}{name}")
    return report


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)
    parser.add_argument("--results_root", default=RESULTS_DEFAULT, type=str)
    parser.add_argument("--date", default="20260820", type=str)  # results/<date>_lbm_lite/<video>/

    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    #    **기본 True.** 뱅크/결정 pose 는 frame0 고정 K 로 풀렸으므로 지표도 같은 K 로 재야 한다.
    #    여기서 프레임별 DA3 K 를 쓰면 화각이 떨려(snowboard fx 진폭 6.99%, 화면 가장자리 최대
    #    14.9 px/frame) `hole_fraction`/`subject_in_frame` 이 궤적과 무관한 이유로 흔들린다.
    #    `--no_fixed_focal` 은 예전 동작(프레임별 K)이다.
    parser.add_argument("--fixed_focal", action="store_true", default=True)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    # 렌더 해상도. None 이면 소스 원본 — 지표는 원본에서 재는 게 맞다 (구멍은 해상도에 안 비례한다).
    parser.add_argument("--height", default=None, type=int)
    parser.add_argument("--width", default=None, type=int)
    # 지표 2 의 "화면 안" 정의. 게이트(`--center_box`)와 같은 값을 써야 판정이 안 갈린다.
    parser.add_argument("--center_box", default=0.80, type=float)
    parser.add_argument("--min_path_len_u", default=MIN_PATH_LEN_U, type=float)  # 0 = 지표 끄기
    # decision.json 에 start_mode 가 없는 옛 파일용 기본값. 있으면 decision 이 이긴다.
    parser.add_argument("--start_mode", default="source_frame0",
                        choices=["source_frame0", "board"])
    # 조준 앵커 (D26/D27) — build_poses 와 **같은 값**이어야 지문이 맞는다.
    parser.add_argument("--aim_anchor", default="subject",
                        choices=["auto", "source_frame0", "subject"])
    parser.add_argument("--aim_ramp_frames", default=12, type=int)
    parser.add_argument("--allow_stale_poses", action="store_true", default=False)
    parser.add_argument("--previews", action="store_true", default=True)
    parser.add_argument("--no_previews", dest="previews", action="store_false")
    # plan_cam.mp4 에 카메라만 확대한 두 번째 줄을 붙인다. 끄면 씬 전체 눈금 한 줄.
    parser.add_argument("--cam_detail", action="store_true", default=True)
    parser.add_argument("--no_cam_detail", dest="cam_detail", action="store_false")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
