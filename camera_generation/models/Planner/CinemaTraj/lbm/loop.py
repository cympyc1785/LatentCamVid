"""look-before-move 본체: **select → micro-adjust → trajectory**, 전부 렌더를 보여주고 고르게 한다.

LBM 원본의 Director→Cinematographer→VideoEngineer 3단 핸드오프를 한 루프로 접었다 (계획서 §변경점).
남긴 것은 순서다: 후보를 **먼저 렌더해서** 보여주고, VLM 은 렌더 가능한 것들 중에서만 고른다.
공간 제약은 프롬프트가 아니라 게이트가 집행한다 — 매 micro 연산 뒤에 `gates.evaluate` 를 다시
돌려 실패하면 되돌리고, 다음 턴에 `## REJECTED` 로 사유를 붙여 알려준다.

세 단계 모두 실패해도 **파이프라인은 hard-fail 하지 않는다** (§B6). 각 단계는 독립적으로
결정론적 fallback 을 갖는다:

    select 실패  -> `scripts/build_decision_fallback.fallback_decision` 로 통째 대체
    micro 실패   -> 그 시점 상태에서 멈춘다 (지금까지의 연산은 살아 있다)
    traj 실패    -> orbit_left, sweep = min(45, 0.7 x obs_az_span), tracking drift

`decision.source` 가 `vlm` / `vlm_partial` / `fallback` 중 무엇인지로 구분한다. `vlm_partial` 은
"어느 단계에선가 소진되어 fallback 이 섞였다"는 뜻이고, `decision.stage_sources` 에 단계별로 남는다.

## 왜 preset 프리뷰까지 렌더하나

계획서의 look-before-move 를 궤적에도 적용한다. preset 이름만 텍스트로 고르게 하면 "orbit 하면
뒤통수만 보인다"를 모델이 알 수 없다 — 점군에 그 면의 자료가 없다는 건 렌더해야만 드러난다.
프리뷰는 **최종 디코더와 같은 코드**(`decode.build_poses.build_poses`)로 만든다. 다른 코드로
그리면 board 에서 고른 것과 실제로 나오는 궤적이 갈라진다.

env: `vista4d` (렌더러가 GPU 를 쓴다). VLM 서버는 `bash scripts/serve_qwen3vl.sh` 로 먼저 띄운다.

예시:
    CUDA_VISIBLE_DEVICES=0 python -m lbm.loop --video camel
    CUDA_VISIBLE_DEVICES=0 python -m lbm.loop --video camel --micro_rounds 6 --allow_zoom
    CUDA_VISIBLE_DEVICES=0 python -m lbm.loop --video camel --no_vlm     # fallback 만 (VLM 미호출)
"""
import json
import sys
from glob import glob
from os import makedirs, path, remove

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import build_poses                                       # noqa: E402
from lbm.candidates import g_pose_to_world, source_spherical                     # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                              # noqa: E402
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT, subject_point_mask  # noqa: E402
from lbm.gates import evaluate                                                   # noqa: E402
from lbm.ops import OPS, ZOOM_OPS, apply_op, make_state, op_menu, spherical      # noqa: E402
from lbm.overlay import (contact_sheet, draw_obb_world, draw_thirds,             # noqa: E402
                         label_tile, paint_holes)
from lbm.presets import PRESET_ALIASES, PRESETS                                  # noqa: E402
from lbm.render import CloudRenderer, look_at_c2w                                # noqa: E402
from lbm.vlm import DEFAULT_API_BASE, DEFAULT_MODEL, VLMClient                   # noqa: E402
from scene_graph.io import load_scene                                            # noqa: E402
from scene_graph.lift import apply_transform                                     # noqa: E402
from scene_graph.obb import node_obb_at, obb_corners                             # noqa: E402
from scripts.build_candidate_board import subject_track_volume                   # noqa: E402
from scripts.build_decision_fallback import fallback_decision                    # noqa: E402

SPEEDS = ("steady", "accel", "decel", "ease")
# D127. `drift`(0.6) 삭제 — 조준은 "안 함(world)" 아니면 "한다(lock)" 둘뿐이다.
TRACKINGS = ("world", "lock")


def read_prompt(name: str):
    with open(path.join(path.dirname(path.abspath(__file__)), "prompts", name),
              encoding="utf-8") as file:
        return file.read()


# ─────────────────────────────────────────────────────────────────────── 응답 검증
# 검증기는 위반 **목록**을 돌려준다. 하나만 돌려주면 재질의가 한 번에 하나씩만 고쳐진다.

def validate_select(payload: dict, labels: set):
    errors = []
    picks = payload.get("picks")
    if not isinstance(picks, list) or not picks:
        errors.append("picks: must be a non-empty list of board labels")
    else:
        unknown = [p for p in picks if p not in labels]
        if unknown:
            errors.append(f"picks: {unknown} are not on the board. "
                          f"valid labels: {sorted(labels)}")
        if len(picks) > 3:
            errors.append("picks: at most 3 labels")
    for key in ("observation", "reasoning"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            errors.append(f"{key}: must be a non-empty string")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence: must be a number in [0, 1]")
    return errors


def validate_micro(payload: dict, allowed: set):
    errors = []
    op = payload.get("op")
    done = payload.get("done")
    if not isinstance(done, bool):
        errors.append("done: must be true or false")
    if op not in allowed and op != "none":
        errors.append(f"op: '{op}' is not in the menu. valid: {sorted(allowed) + ['none']}")
    if done is False and op == "none":
        errors.append("op: 'none' is only valid together with done=true")
    if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
        errors.append("reason: must be a non-empty string")
    return errors


def validate_traj(payload: dict, presets: set):
    errors = []
    preset = PRESET_ALIASES.get(payload.get("preset"), payload.get("preset"))
    if preset not in presets:
        errors.append(f"preset: '{payload.get('preset')}' is not on the board. "
                      f"valid: {sorted(presets)}")
    if payload.get("speed") not in SPEEDS:
        errors.append(f"speed: must be one of {list(SPEEDS)}")
    if payload.get("tracking") not in TRACKINGS:
        errors.append(f"tracking: must be one of {list(TRACKINGS)}")
    if not isinstance(payload.get("reasoning"), str) or not payload["reasoning"].strip():
        errors.append("reasoning: must be a non-empty string")
    return errors


# ─────────────────────────────────────────────────────────────────── 렌더 + 게이트 세션

class Session:
    """렌더러 + 게이트 인자를 한 번 묶어 두고 pose 상태만 갈아끼운다."""

    def __init__(self, args, graph: dict, board: dict, node: dict):
        self.args, self.graph, self.board, self.node = args, graph, board, node
        out_root = args.output_root or path.join(CLOUD_ROOT, "out")
        self.folder = path.join(out_root, args.video)
        self.renderer = CloudRenderer(path.join(self.folder, "cloud.npz"),
                                      vista4d_root=args.vista4d_root, device=args.device,
                                      fixed_focal=getattr(args, "fixed_focal", True))
        recon = load_scene(args.eval_data, args.video, args.vista4d_root,
                           seg_root=args.seg_root, seg_static_root=args.seg_static_root)
        self.depths, self.sky_mask = recon["depths"], recon["sky_mask"]

        volume = subject_track_volume(recon, node)
        self.subject_points = subject_point_mask(self.renderer.indices, volume).cpu().numpy()
        # subject 플래그를 켜 두면 모든 render() 가 실루엣 마스크를 같이 돌려준다 (preset 검사용).
        self.renderer.set_subject(self.subject_points)

        self.T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)
        self.scale = float(graph["scale"]["S"])
        self.z_med = float(graph["scale"]["z_med_frame0"])
        self.d_ref = float(node["viewing_distance"]["d_ref"])
        self.frame = int(board.get("frame", 0))
        self.subject_center_world = apply_transform(
            self.T_wg, np.asarray(node["track"]["center_smooth"], dtype=float)[self.frame][None])[0]
        self.src_center = self.renderer.cam_c2w_src[self.frame][:3, 3]
        self.behind_frames = np.unique(
            np.linspace(0, self.renderer.num_frames - 1, args.behind_frames).round()
            .astype(int)).tolist()
        self.source_spherical = source_spherical(
            node, np.asarray(graph["cameras"]["cam_centers_g"], dtype=float), self.frame)

    # ---------------------------------------------------------------- 기하
    def to_world(self, state: dict):
        return g_pose_to_world(state["p_g"], state["look_at_g"], self.T_wg, look_at_c2w)

    def intrinsics(self, state: dict):
        """focal_scale 을 먹인 소스 K. 1.0 이면 None 을 돌려 기존 경로를 그대로 탄다."""
        if abs(state["focal_scale"] - 1.0) < 1e-9:
            return None
        K = self.renderer.K_src[self.frame].copy()
        K[0, 0] *= state["focal_scale"]
        K[1, 1] *= state["focal_scale"]
        return K

    def gate(self, state: dict, cand_id: str = "cur"):
        """상태 하나를 전 게이트에 통과시킨다. board 후보와 **같은 함수**를 탄다."""
        azimuth, elevation, radius = spherical(state, self.node, self.frame)
        az_src, el_src, radius_src = self.source_spherical
        candidate = {
            "cand_id": cand_id,
            "azimuth_deg": round(azimuth, 1), "elevation_deg": round(elevation, 1),
            "d_azimuth_deg": round(azimuth - az_src, 1),
            "d_elevation_deg": round(elevation - el_src, 1),
            "distance_ratio": round(radius / max(radius_src, 1e-9), 3),
            "radius_u": float(radius), "focal_scale": state["focal_scale"],
            "c2w_world": self.to_world(state),
        }
        candidate["p_world"] = candidate["c2w_world"][:3, 3]
        return evaluate(candidate, self.renderer, self.subject_points,
                        obb_corners_world=None, depths=self.depths, sky_mask=self.sky_mask,
                        scale=self.scale, z_med=self.z_med,
                        subject_center_world=self.subject_center_world,
                        src_center=self.src_center,
                        tile_height=self.args.tile_height, tile_width=self.args.tile_width,
                        behind_frames=self.behind_frames, max_tau=self.args.max_tau,
                        max_view_angle_deg=self.args.max_view_angle_deg,
                        min_coverage=self.args.min_coverage, center_box=self.args.center_box,
                        subject_area_range=(self.args.min_subject_area, self.args.max_subject_area),
                        min_occlusion_pass=self.args.min_occlusion_pass,
                        render_frame_index=self.frame, K=self.intrinsics(state))

    # ---------------------------------------------------------------- 그림
    def tile(self, rendered: dict, label: str, caption: str = "", frame: int | None = None):
        """board 타일과 **같은 오버레이**. 다르게 그리면 단계마다 그림이 달라 보인다."""
        image = paint_holes(rendered["rgb"], rendered["valid"])
        scaled_K = np.asarray(rendered["K"], dtype=float).copy()
        corners = apply_transform(self.T_wg, obb_corners(
            *node_obb_at(self.node, self.frame if frame is None else frame)))
        draw_obb_world(image, corners, scaled_K, rendered["cam_c2w"])
        return label_tile(draw_thirds(image, safe_frac=self.args.center_box), label, caption)

    def render_pose(self, cam_c2w, frame: int, K=None):
        return self.renderer.render(cam_c2w, K=K, frame=frame,
                                    height=self.args.tile_height, width=self.args.tile_width)


def source_frame0_state(session: Session):
    """소스 카메라의 frame 0 을 루프 상태(G 좌표)로 옮긴다 — `start_mode="source_frame0"` 용.

    `decode.build_poses` 가 같은 모드에서 쓰는 pose 와 **같은 것**이어야 한다. 거기서는 world c2w
    (`cam_c2w_world[0]`)를 그대로 쓰고, 여기서는 그걸 `(p_g, look_at_g)` 로 표현할 뿐이다.

    look_at 은 subject 가 아니라 **소스 카메라의 시선 위**에 둔다. subject 로 다시 조준하면
    "첫 카메라를 그대로"가 아니라 "첫 카메라 위치에서 subject 를 본다"가 되어, 게이트가 보는
    그림이 소스 프레임과 달라진다. (`aim="look_at"` preset 은 디코드 단계에서 어차피 매 프레임
    subject 로 다시 조준하므로, 이 방향은 pan/truck/pedestal 계열에만 끝까지 남는다.)

    남는 차이는 **roll 하나**다. `(p_g, look_at_g)` 표현은 중력축 기준 roll=0 을 강제하는데
    소스 카메라는 살짝 기울어져 있다 (실측 camel 3.15°, avocado-slice 2.04°). 위치와 시선
    방향은 1e-6° 이내로 같다. 즉 게이트가 보는 타일만 그만큼 덜 기울어져 있고, 디코드되는
    궤적은 `cam_c2w_world[0]` 을 그대로 쓴다.
    """
    graph = session.graph
    T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
    c2w = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float)[0]
    p_g = T_gw[:3, :3] @ c2w[:3, 3] + T_gw[:3, 3]
    # T_gw[:3,:3] 는 1/S 를 품고 있다 (R_gw/S) — 방향으로 쓰려면 정규화해야 한다.
    forward_g = T_gw[:3, :3] @ c2w[:3, 2]
    forward_g = forward_g / max(float(np.linalg.norm(forward_g)), 1e-12)
    center_g = np.asarray(session.node["track"]["center_smooth"], dtype=float)[0]
    radius = float(np.linalg.norm(p_g - center_g))     # look_at 을 subject 거리에 둔다 (표시용 눈금)
    return make_state(p_g, p_g + radius * forward_g, 1.0)


def _number(value, digits: int = 2):
    """게이트가 일찍 끊기면 필드가 `None` 으로 남는다. `:.2f` 는 거기서 TypeError 를 낸다."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def state_line(row: dict):
    """게이트 결과 한 줄 (영문 — 프롬프트로 나간다).

    `subject_center` 만 조건부다. 앞의 세 항목까지 같은 조건에 묶으면 subject 를 못 찾은 순간
    coverage 가 프롬프트에서 통째로 사라져, 모델이 왜 거절당했는지 알 길이 없어진다.
    """
    parts = [f"coverage {_number(row.get('coverage'))}",
             f"subject_area {_number(row.get('subject_area'), 3)}",
             f"occlusion_pass {_number(row.get('occlusion_pass'))}"]
    center = row.get("subject_center")
    if center is not None:
        parts.append(f"subject_center {center[0]:.2f},{center[1]:.2f}")
    parts += [f"tau {_number(row.get('tau'), 3)}",
              f"d_az {row['d_azimuth_deg']:+.1f}", f"d_elev {row['d_elevation_deg']:+.1f}",
              f"dist {row['distance_ratio']:.2f}x"]
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────── 단계 1: 선택

def stage_select(client: VLMClient, session: Session, contract: str, trace_folder: str):
    """(picked_row, payload, info). 실패하면 picked_row=None."""
    labels = {c["board_label"] for c in session.board["candidates"]}
    prompt = (contract + "\n\n## TASK\nPick the starting camera position.\n"
              f"Valid labels: {', '.join(sorted(labels))}\n"
              "Return raw JSON with keys observation, reasoning, picks, confidence.")
    images = [path.join(session.folder, "board", "board_candidates.png")]
    source_panel = path.join(session.folder, "board", "source_frames.png")
    if path.isfile(source_panel):
        images.append(source_panel)

    payload, info = client.chat_json(prompt, images=images, system=read_prompt("system_select.md"),
                                     validate=lambda p: validate_select(p, labels),
                                     max_repairs=client_max_repairs(client), label="select")
    if payload is None:
        return None, None, info
    picked = next(c for c in session.board["candidates"] if c["board_label"] == payload["picks"][0])
    with open(path.join(trace_folder, "select.json"), "w", encoding="utf-8") as file:
        json.dump({"payload": payload, "info": {k: v for k, v in info.items() if k != "last_meta"}},
                  file, ensure_ascii=False, indent=1)
    return picked, payload, info


def client_max_repairs(client: VLMClient):
    return getattr(client, "max_repairs", 3)


# ────────────────────────────────────────────────────────────── 단계 2: micro-adjust

def stage_micro(client: VLMClient, session: Session, state: dict, row: dict,
                trace_folder: str, contract_head: str):
    """(state, row, records). 게이트를 통과한 상태만 돌려준다."""
    allowed = {name for name in OPS if session.args.allow_zoom or name not in ZOOM_OPS}
    system = read_prompt("system_micro.md")
    previous_render, previous_op = row["rendered"], None
    rejected, consecutive_rejects, records = {}, 0, []

    for round_index in range(session.args.micro_rounds):
        current = row["rendered"]
        panel = contact_sheet([session.tile(previous_render, "BEFORE",
                                            f"after: {previous_op}" if previous_op else "start"),
                               session.tile(current, "AFTER", state_line(row))], columns=2)
        image_path = path.join(trace_folder, f"micro_{round_index:02d}.png")
        import imageio.v2 as imageio
        imageio.imwrite(image_path, panel)

        prompt = (contract_head + "\n\n## CURRENT\n" + state_line(row)
                  + (f"\nfocal_scale {state['focal_scale']:.2f}"
                     if session.args.allow_zoom else "")
                  + f"\nround {round_index + 1} of {session.args.micro_rounds}"
                  + "\n\n## OPERATIONS\n" + op_menu(session.args.allow_zoom, rejected)
                  + "\n\n## TASK\nChoose one operation, or stop.\n"
                  + "Return raw JSON with keys observation, op, reason, done.")
        payload, info = client.chat_json(prompt, images=[image_path], system=system,
                                         validate=lambda p: validate_micro(p, allowed),
                                         max_repairs=client_max_repairs(client),
                                         label=f"micro{round_index}")
        if payload is None:
            records.append({"round": round_index, "op": None, "result": "vlm_exhausted",
                            "errors": info["errors"]})
            break
        if payload["done"] or payload["op"] == "none":
            records.append({"round": round_index, "op": "none", "result": "done",
                            "reason": payload["reason"]})
            break

        op = payload["op"]
        candidate_state, note = apply_op(state, op, session.node, session.d_ref, session.frame)
        if candidate_state is None:                       # clamp 밖 — 렌더도 안 하고 거절
            rejected[op] = note
            consecutive_rejects += 1
            records.append({"round": round_index, "op": op, "result": "rejected", "why": note,
                            "reason": payload["reason"]})
        else:
            candidate_row = session.gate(candidate_state, cand_id=f"micro{round_index}")
            if candidate_row["failed"] is None:
                previous_render, previous_op = current, op
                state, row = candidate_state, candidate_row
                consecutive_rejects = 0
                records.append({"round": round_index, "op": op, "result": "applied",
                                "note": note, "reason": payload["reason"],
                                "coverage": row["coverage"], "subject_area": row["subject_area"],
                                "tau": row["tau"]})
            else:
                why = f"{candidate_row['failed']} " + _fail_detail(candidate_row, session.args)
                rejected[op] = why
                consecutive_rejects += 1
                records.append({"round": round_index, "op": op, "result": "rejected", "why": why,
                                "reason": payload["reason"]})
        if consecutive_rejects >= session.args.max_consecutive_rejects:
            records.append({"round": round_index, "op": None, "result": "stopped",
                            "why": f"{consecutive_rejects} consecutive rejections"})
            break

    with open(path.join(trace_folder, "micro.json"), "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=1)
    return state, row, records


def _fail_detail(row: dict, args):
    """거절 사유를 **수치와 함께** 돌려준다. 이름만 주면 모델이 같은 실수를 반복한다."""
    name = row["failed"]
    if name == "G2_coverage":
        return f"coverage {row['coverage']:.2f} < {args.min_coverage:.2f}"
    if name == "G3_center":
        c = row.get("subject_center")
        return f"subject center {c[0]:.2f},{c[1]:.2f} outside the safe frame" if c else "subject lost"
    if name == "G3_area":
        return (f"subject_area {row.get('subject_area', 0):.3f} outside "
                f"[{args.min_subject_area}, {args.max_subject_area}]")
    if name == "G3_occlusion":
        return f"occlusion_pass {row['occlusion_pass']:.2f} < {args.min_occlusion_pass:.2f}"
    if name == "G1_behind":
        return f"camera would be behind a surface in {row['behind_frames']} source frames"
    if name == "G4_tau":
        return f"tau {row['tau']:.3f} > {args.max_tau:.2f}"
    if name == "G4_view_angle":
        return f"view angle {row['view_angle_deg']:.0f} deg > {args.max_view_angle_deg:.0f}"
    return ""


# ─────────────────────────────────────────────────────────────── 단계 3: trajectory

def provisional_decision(session: Session, state: dict, preset: str, speed: str, tracking: str,
                         board_label: str | None, micro_ops: list):
    """`decode.build_poses` 가 먹는 최소 decision. 프리뷰와 최종 디코드가 같은 코드를 타게 한다."""
    return {"format": "lbm_decision_v1", "video": session.graph["video"],
            "subject_id": session.node["id"],
            "keyframes": [{"t": 0, "target": session.node["id"],
                           "composition": {"p_G": state["p_g"].tolist(),
                                           "look_at_G": state["look_at_g"].tolist(),
                                           "focal_scale": state["focal_scale"]},
                           "candidate_id": board_label, "micro_ops": micro_ops}],
            "trajectory": {"preset": preset, "speed": speed, "tracking": tracking,
                           "look_at_bias": session.args.look_at_bias,
                           "target_tau": session.args.target_tau}}


def preview_preset(session: Session, state: dict, preset: str, board_label, micro_ops):
    """(tiles, stats). preset 하나를 49프레임 디코드하고 `preview_frames` 만 렌더해 검사한다."""
    decision = provisional_decision(session, state, preset, session.args.speed,
                                    session.args.tracking, board_label, micro_ops)
    poses, extra = build_poses(decision, session.graph, board=None,
                               target_tau=session.args.target_tau,
                               orbit_span_frac=session.args.orbit_span_frac,
                               start_mode=session.args.start_mode)
    info = extra["info"]
    num_frames = len(poses)
    check = np.unique(np.linspace(0, num_frames - 1, session.args.preview_check_frames)
                      .round().astype(int))
    show = np.unique(np.linspace(0, num_frames - 1, 3).round().astype(int))

    K = session.intrinsics(state)
    coverage, areas, tiles = [], [], []
    for f in check:
        cloud_frame = int(round(f * (session.renderer.num_frames - 1) / max(num_frames - 1, 1)))
        rendered = session.render_pose(poses[f], cloud_frame, K=K)
        coverage.append(float(rendered["valid"].mean()))
        subject = rendered["subject"]
        areas.append(float(subject.mean()) if subject is not None else float("nan"))
        if f in show:
            tiles.append(session.tile(rendered, f"f{f}",
                                      f"cov{coverage[-1]:.2f} subj{areas[-1]:.3f}",
                                      frame=cloud_frame))

    aim = info["aim"]
    stats = {"preset": preset, "aim": aim, "move": info["path_len_u"],
             "rotation_deg": _rotation_deg(poses), "tau_max": info["tau"]["tau_max_final"],
             "tau_scale": info["tau"]["scale"], "tau_saturated": info["tau"]["tau_saturated"],
             "coverage_min": round(float(np.min(coverage)), 3),
             "coverage_end": round(float(coverage[-1]), 3),
             "subject_area_min": round(float(np.nanmin(areas)), 4),
             "excluded": None}
    if stats["coverage_min"] < session.args.min_coverage:
        stats["excluded"] = (f"coverage drops to {stats['coverage_min']:.2f} "
                             f"(floor {session.args.min_coverage:.2f})")
    # subject 이탈 검사는 `aim="look_at"` preset 에만 건다. pan/truck/pedestal 은 정의상 subject 를
    # 화면 밖으로 보내는 움직임이라 (§presets.py 의 aim 논의) 같은 자로 재면 전부 탈락한다.
    elif aim == "look_at" and stats["subject_area_min"] < session.args.min_subject_area:
        stats["excluded"] = (f"subject shrinks to {stats['subject_area_min']:.3f} of frame "
                             f"(floor {session.args.min_subject_area})")
    return tiles, stats


def _rotation_deg(poses: np.ndarray):
    relative = np.linalg.inv(poses[0]) @ poses[-1]
    return round(float(np.degrees(np.arccos(
        np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1.0, 1.0)))), 2)


def stage_traj(client: VLMClient, session: Session, state: dict, board_label, micro_ops,
               trace_folder: str, contract_head: str):
    """(payload, info, stats_all). preset 프리뷰 board 를 만들고 하나를 고르게 한다."""
    import imageio.v2 as imageio

    names = [n for n in PRESETS if session.args.allow_zoom or not PRESETS[n][2]]
    rows, stats_all = [], []
    for name in names:
        tiles, stats = preview_preset(session, state, name, board_label, micro_ops)
        stats_all.append(stats)
        if stats["excluded"] is None:
            rows.append(label_tile(contact_sheet(tiles, columns=3), name))
    board_path = path.join(trace_folder, "board_presets.png")
    if rows:
        imageio.imwrite(board_path, contact_sheet(rows, columns=session.args.preset_columns))

    shown = {s["preset"] for s in stats_all if s["excluded"] is None}
    lines = []
    if session.args.start_mode == "source_frame0":
        # 시작 pose 가 결정 대상이 아니라는 걸 명시한다. `board` 경로에서는 모델이 직전 두 턴에
        # 시작 pose 를 직접 골랐지만 여기서는 그 턴이 아예 없어서, 말해 주지 않으면 어디서
        # 출발하는지가 프롬프트 어디에도 없다.
        lines += ["## STARTING CAMERA",
                  "The shot starts from the source camera's own first frame — the view in the "
                  "source panel. You are not choosing where to stand, only how to move from there.",
                  ""]
    lines += ["## PRESETS ON THE BOARD",
             "(move = camera path length in scene units after tau fitting; rotation = total "
             "rotation from first to last frame; coverage_end = fraction of real pixels in the "
             "last frame)"]
    for stats in stats_all:
        if stats["excluded"] is None:
            lines.append(f"{stats['preset']:<20} move {stats['move']:.3f} | "
                         f"rotation {stats['rotation_deg']:.1f} deg | "
                         f"coverage_min {stats['coverage_min']:.2f} | "
                         f"coverage_end {stats['coverage_end']:.2f} | "
                         f"tau_max {stats['tau_max']:.2f}"
                         + ("  [MOTION SATURATED: the starting position already spent the tau "
                            "budget, this renders as a still]" if stats["tau_saturated"] else ""))
    excluded = [s for s in stats_all if s["excluded"] is not None]
    if excluded:
        lines += ["", "## EXCLUDED (not on the board)"]
        lines += [f"{s['preset']:<20} {s['excluded']}" for s in excluded]

    if not shown:
        return None, {"errors": ["every preset was excluded"], "repairs": 0, "turns": 0}, stats_all

    prompt = (contract_head + "\n\n" + "\n".join(lines)
              + "\n\n## TASK\nChoose the camera motion.\n"
              + "Return raw JSON with keys observation, preset, speed, tracking, reasoning, "
                "confidence.")
    images = [board_path] if rows else []
    # source_frame0 에서는 select 턴이 없어서 소스 패널이 이 실행에서 한 번도 안 나갔다.
    # 위 STARTING CAMERA 문장이 가리키는 그림이므로 여기서 같이 보낸다 (look-before-move).
    source_panel = path.join(session.folder, "board", "source_frames.png")
    if session.args.start_mode == "source_frame0" and path.isfile(source_panel):
        images.append(source_panel)
    payload, info = client.chat_json(prompt, images=images,
                                     system=read_prompt("system_traj.md"),
                                     validate=lambda p: validate_traj(p, shown),
                                     max_repairs=client_max_repairs(client), label="traj")
    if payload is not None:
        payload["preset"] = PRESET_ALIASES.get(payload["preset"], payload["preset"])
    with open(path.join(trace_folder, "traj.json"), "w", encoding="utf-8") as file:
        json.dump({"payload": payload, "stats": stats_all}, file, ensure_ascii=False, indent=1)
    return payload, info, stats_all


# ──────────────────────────────────────────────────────────────────────────── 본체

def run(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    folder = path.join(out_root, args.video)
    with open(path.join(folder, "scene_graph.json"), encoding="utf-8") as file:
        graph = json.load(file)
    with open(path.join(folder, "board", "board.json"), encoding="utf-8") as file:
        board = json.load(file)
    with open(path.join(folder, "board", "contract.txt"), encoding="utf-8") as file:
        contract = file.read()
    assert board["candidates"], "board 에 통과 후보가 없다 — gates.csv 의 탈락 사유부터 볼 것"

    node = next(n for n in graph["nodes"] if n["id"] == board["subject_id"])
    trace_folder = path.join(folder, "trace")
    makedirs(trace_folder, exist_ok=True)
    # 이번 실행이 더 짧으면 지난 실행의 턴 기록이 남아 섞인다 (`save_trace` 는 인덱스 순으로
    # 덮어쓸 뿐 지우지 않는다). trace 를 읽는 쪽은 그게 이번 실행 것인지 알 방법이 없다.
    for stale in sorted(glob(path.join(trace_folder, "turn_*.json"))
                        + glob(path.join(trace_folder, "micro_*.png"))):
        remove(stale)

    if args.no_vlm:      # VLM 을 아예 안 부르는 경로 — 기존 fallback 스크립트와 같은 결과여야 한다
        decision = fallback_decision(graph, board, preset=args.preset, tracking=args.tracking,
                                     target_tau=args.target_tau,
                                     orbit_span_frac=args.orbit_span_frac,
                                     look_at_bias=args.look_at_bias,
                                     start_tau_frac=args.start_tau_frac, reason="--no_vlm")
        return _write(decision, folder, args), None

    session = Session(args, graph, board, node)
    client = VLMClient(api_base=args.api_base, model=args.model, temperature=args.temperature,
                       max_tokens=args.max_tokens, timeout=args.timeout)
    client.max_repairs = args.max_repairs
    # 텍스트 블록의 SCENE/SUBJECT/SOURCE_CAMERA 머리만 잘라 micro/traj 턴에 다시 붙인다.
    # CANDIDATES 목록은 select 단계에서만 의미가 있다 (이미 하나를 골랐다).
    contract_head = contract.split("## CANDIDATES")[0].rstrip()

    stage_sources = {}
    if args.start_mode == "source_frame0":
        # 시작 pose 를 결정 대상에서 뺀다 (DECISIONS.md D21). select/micro 는 **시작 pose 만**
        # 정하는 단계라 통째로 건너뛴다 — 남는 결정은 궤적 하나다. board 는 여전히 읽는다:
        # subject / contract 텍스트 / 게이트 임계가 거기서 오고, `--start_mode board` 로 언제든
        # 예전 3단 경로로 되돌아갈 수 있어야 한다.
        assert session.frame == 0, (f"board frame 이 {session.frame} 인데 source_frame0 는 frame 0 "
                                    "을 쓴다 (build_poses 도 cam_c2w_world[0] 고정)")
        state = source_frame0_state(session)
        row = session.gate(state, cand_id="source_frame0")
        if row["failed"] is not None:
            # 게이트를 못 넘어도 진행한다 — 소스 카메라 자신이 떨어졌다는 건 후보 pose 가 나쁜 게
            # 아니라 게이트(구도·subject 면적)가 이 씬에 안 맞는다는 뜻이고, 여기서 멈추면
            # 궤적을 아예 못 만든다.
            print(f"!! 소스 frame0 pose 가 게이트 {row['failed']} 를 못 넘는다 "
                  f"({_fail_detail(row, args)}) — 그래도 시작 pose 로 쓴다")
        picked_label, applied, micro_records = None, [], []
        # 같은 폴더에 예전 `board` 실행의 select.json / micro.json 이 남아 있으면, 이 실행은
        # 그 단계를 아예 안 돌렸는데도 trace 에는 있는 것처럼 보인다. 지운다.
        for stale in ("select.json", "micro.json"):
            if path.isfile(path.join(trace_folder, stale)):
                remove(path.join(trace_folder, stale))
        select_payload = {"observation": None, "reasoning": None, "picks": [], "confidence": None}
        select_info = {"turns": 0, "repairs": 0, "errors": []}
        stage_sources["select"] = "skipped_source_frame0"
        stage_sources["micro"] = "skipped_source_frame0"
    else:
        picked, select_payload, select_info = stage_select(client, session, contract, trace_folder)
        if picked is None:
            print("!! select 단계 소진 — 결정론적 fallback 으로 통째 대체한다")
            decision = fallback_decision(graph, board, preset=args.preset, tracking=args.tracking,
                                         target_tau=args.target_tau,
                                         orbit_span_frac=args.orbit_span_frac,
                                         look_at_bias=args.look_at_bias,
                                         start_tau_frac=args.start_tau_frac,
                                         reason="VLM select exhausted")
            decision["vlm"] = {"observation": None, "reasoning": "select stage exhausted",
                               "confidence": None, "turns": select_info["turns"],
                               "repairs": select_info["repairs"], "errors": select_info["errors"]}
            client.save_trace(trace_folder)
            return _write(decision, folder, args), client
        stage_sources["select"] = "vlm"

        state = make_state(picked["p_g"], picked["look_at_g"], 1.0)
        row = session.gate(state, cand_id=picked["board_label"])
        assert row["failed"] is None, (f"board 후보 {picked['board_label']} 이 지금 게이트를 못 넘는다: "
                                       f"{row['failed']} — board 와 loop 의 임계가 다르다")

        micro_records = []
        if args.micro_rounds > 0:
            state, row, micro_records = stage_micro(client, session, state, row, trace_folder,
                                                    contract_head)
        applied = [r["op"] for r in micro_records if r["result"] == "applied"]
        stage_sources["micro"] = ("vlm" if not any(r["result"] == "vlm_exhausted"
                                                   for r in micro_records) else "vlm_partial")
        picked_label = picked["board_label"]

    traj_payload, traj_info, preset_stats = stage_traj(
        client, session, state, picked_label, applied, trace_folder, contract_head)
    if traj_payload is None:
        sweep = min(45.0, args.orbit_span_frac * float(node["obs_az_span_deg"]))
        traj_payload = {"observation": None, "preset": args.preset, "speed": "steady",
                        "tracking": args.tracking, "confidence": None,
                        "reasoning": f"deterministic fallback (trajectory stage: "
                                     f"{'; '.join(traj_info['errors'][-2:])})"}
        stage_sources["traj"] = "fallback"
        print(f"!! trajectory 단계 소진 — {args.preset} sweep {sweep:.1f} deg 로 대체한다")
    else:
        stage_sources["traj"] = "vlm"

    azimuth, elevation, radius = spherical(state, node, session.frame)
    az_src, el_src, radius_src = session.source_spherical
    # 건너뛴 단계는 "VLM 이 소진됐다"가 아니다 — 설계상 안 돌린 것이라 판정에서 뺀다. 안 빼면
    # source_frame0 로 정상 완주한 실행이 전부 `vlm_partial` 로 찍혀 진짜 부분 실패와 안 갈린다.
    ran = {v for v in stage_sources.values() if not v.startswith("skipped")}
    source = "vlm" if ran == {"vlm"} else "vlm_partial"
    decision = {
        "format": "lbm_decision_v1", "video": graph["video"], "subject_id": node["id"],
        "source": source, "model": args.model, "stage_sources": stage_sources,
        "start_mode": args.start_mode,
        "keyframes": [{
            "t": 0, "target": node["id"],
            "composition": {"azimuth_deg": round(azimuth, 1), "elevation_deg": round(elevation, 1),
                            "d_azimuth_deg": round(azimuth - az_src, 1),
                            "d_elevation_deg": round(elevation - el_src, 1),
                            "distance_ratio": round(radius / max(radius_src, 1e-9), 3),
                            "thirds_anchor": None,
                            "p_G": state["p_g"].tolist(), "look_at_G": state["look_at_g"].tolist(),
                            "focal_scale": state["focal_scale"]},
            "visibility": {"coverage": row["coverage"], "subject_area": row["subject_area"],
                           "occlusion_pass": row["occlusion_pass"]},
            "relation": None, "motion_preference": None,
            "candidate_id": picked_label, "micro_ops": applied}],
        "trajectory": {"preset": traj_payload["preset"], "params": {},
                       "speed": traj_payload["speed"], "tracking": traj_payload["tracking"],
                       "look_at_bias": args.look_at_bias, "target_tau": args.target_tau},
        "gates": {"coverage": row["coverage"], "subject_area": row["subject_area"],
                  "tau": row["tau"], "view_angle_deg": row["view_angle_deg"],
                  "occlusion_pass": row["occlusion_pass"]},
        "vlm": {"observation": traj_payload.get("observation") or select_payload["observation"],
                "reasoning": traj_payload["reasoning"],
                "select_reasoning": select_payload["reasoning"],
                "select_picks": select_payload["picks"],
                "confidence": traj_payload.get("confidence", select_payload["confidence"]),
                "turns": len(client.trace),
                "repairs": select_info["repairs"] + traj_info.get("repairs", 0)},
        "micro": {"applied": applied, "rounds": len(micro_records), "records": micro_records},
        "preset_stats": preset_stats,
        "budget_note": None,
    }
    client.save_trace(trace_folder)
    return _write(decision, folder, args), client


def _write(decision: dict, folder: str, args):
    # `start_mode` 는 decision.json 을 디코드하는 방법의 일부다 (decode.build_poses.resolve_start_mode).
    # fallback 경로들은 위에서 안 넣으므로 여기서 한 번에 채운다 — 빠지면 디코더가 CLI 기본값을 써서
    # board 로 고른 결정을 source_frame0 로 풀거나 그 반대가 된다.
    decision.setdefault("start_mode", args.start_mode)
    output = args.output or path.join(folder, "decision.json")
    with open(output, "w", encoding="utf-8") as file:
        json.dump(decision, file, ensure_ascii=False, indent=1)
    decision["_path"] = output
    return decision


def main(args):
    decision, client = run(args)
    keyframe = decision["keyframes"][0]
    composition = keyframe["composition"]
    print()
    if decision["source"] != "vlm":
        print(f"!! decision.source = {decision['source']} — 일부 또는 전부가 VLM 이 고른 게 아니다. "
              "결과를 그렇게 읽을 것.")
    print(f"{'video':<18}{decision['video']}   subject {decision['subject_id']}")
    print(f"{'source':<18}{decision['source']}  {decision.get('stage_sources', {})}")
    print(f"{'start_mode':<18}{decision.get('start_mode')}")
    print(f"{'candidate':<18}{keyframe['candidate_id']}  "
          f"d_az {composition['d_azimuth_deg']}  d_el {composition['d_elevation_deg']}  "
          f"dist {composition['distance_ratio']}x")
    print(f"{'micro_ops':<18}{' -> '.join(keyframe['micro_ops']) or '(none)'}")
    print(f"{'preset':<18}{decision['trajectory']['preset']}  "
          f"speed {decision['trajectory']['speed']}  "
          f"tracking {decision['trajectory']['tracking']}")
    print(f"{'gates':<18}cov {decision['gates']['coverage']}  "
          f"subj {decision['gates']['subject_area']}  tau {decision['gates']['tau']}")
    if client is not None:
        seconds = sum(t["seconds"] for t in client.trace)
        tokens = sum((t["completion_tokens"] or 0) for t in client.trace)
        print(f"{'vlm':<18}{len(client.trace)} turns  {seconds:.1f} s  "
              f"{tokens} completion tokens  repairs {decision['vlm']['repairs']}")
    print(f"\n-> {decision['_path']}")


def build_parser():
    """파서를 함수로 빼 둔다 — 스모크 테스트가 게이트 임계 기본값을 루프와 **공유해야** 한다.
    임계를 손으로 다시 적으면 테스트만 통과하고 루프는 떨어지는 상황이 생긴다.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)

    parser.add_argument("--video", required=True, type=str)
    #    **기본 True.** 후보/micro-adjust pose 는 frame0 고정 K 로 풀리므로 VLM 이 보는 렌더도
    #    같은 K 여야 한다. 프레임별 DA3 K 를 쓰면 화각이 떨려(snowboard fx 진폭 6.99%) before/after
    #    타일 차이에 연산 효과와 화각 떨림이 섞인다. `--no_fixed_focal` 은 예전 동작(프레임별 K).
    #    `LoopRunner` 는 `getattr(args, "fixed_focal", True)` 라 이 인자가 없는 호출자도 그대로 돈다.
    parser.add_argument("--fixed_focal", action="store_true", default=True)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)

    # VLM
    parser.add_argument("--api_base", default=DEFAULT_API_BASE, type=str)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=str)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--timeout", default=180.0, type=float)
    parser.add_argument("--max_repairs", default=3, type=int)
    parser.add_argument("--no_vlm", action="store_true", default=False)   # fallback 만
    parser.add_argument("--vlm", dest="no_vlm", action="store_false")

    # 루프
    # `source_frame0`(기본) = 소스 카메라 frame0 을 시작 pose 로 그대로 쓰고 select/micro 를
    # 건너뛴다 (DECISIONS.md D21 — 시작 pose 는 emit 에서 버려지면서 tau 예산만 먹었다).
    # `board` = 예전 3단 경로 (select -> micro -> traj). 둘 다 그대로 돌아간다.
    parser.add_argument("--start_mode", default="source_frame0",
                        choices=["source_frame0", "board"])
    parser.add_argument("--micro_rounds", default=10, type=int)
    parser.add_argument("--max_consecutive_rejects", default=3, type=int)
    parser.add_argument("--allow_zoom", action="store_true", default=False)
    parser.add_argument("--no_allow_zoom", dest="allow_zoom", action="store_false")
    parser.add_argument("--preview_check_frames", default=7, type=int)
    parser.add_argument("--preset_columns", default=3, type=int)

    # 궤적 (fallback 값이자 프리뷰 기본값)
    parser.add_argument("--preset", default="orbit_left", type=str)
    parser.add_argument("--tracking", default="lock", type=str, choices=list(TRACKINGS))
    parser.add_argument("--speed", default="steady", type=str)
    parser.add_argument("--target_tau", default=0.20, type=float)
    parser.add_argument("--orbit_span_frac", default=0.7, type=float)
    parser.add_argument("--look_at_bias", default=0.0, type=float)
    parser.add_argument("--start_tau_frac", default=0.5, type=float)

    # 게이트 임계 — board 를 만든 값과 **같아야 한다** (다르면 board 후보가 여기서 떨어진다)
    parser.add_argument("--max_tau", default=0.30, type=float)
    parser.add_argument("--max_view_angle_deg", default=40.0, type=float)
    parser.add_argument("--min_coverage", default=0.55, type=float)
    parser.add_argument("--center_box", default=0.80, type=float)
    parser.add_argument("--min_subject_area", default=0.03, type=float)
    parser.add_argument("--max_subject_area", default=0.50, type=float)
    parser.add_argument("--min_occlusion_pass", default=0.40, type=float)
    parser.add_argument("--behind_frames", default=7, type=int)
    parser.add_argument("--tile_width", default=480, type=int)
    parser.add_argument("--tile_height", default=270, type=int)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
