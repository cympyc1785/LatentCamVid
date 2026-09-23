"""Vista 씬에 돌린 LBM 의 카메라 덤프 → Lite `verify.py` 가 그대로 읽는 평가 폴더.

**왜 필요한가.** `vista_to_lbm_demo.py` 는 우리 4D 점군을 `.blend` 로 내보내 **LBM 코드 0줄
수정**으로 원본 4단계를 돌린다. 그 결과 카메라는 Blender world(미터)에 있고, 우리 평가
파이프라인(`verify.py`)은 DA3 frame0 world(OpenCV, u 단위)를 읽는다. 이 파일이 그 사이를 잇는다
— **같은 지표표(hole_fraction / subject_in_frame / tau_max / ...)로 LBM 과 Lite 를 재기 위해서**다.
어댑터가 없으면 "LBM 을 돌렸다"까지는 되지만 "얼마나 좋았나"는 못 잰다.

**좌표 변환은 `_vista.json` 의 역이다.** 어댑터가 내보낼 때 쓴 게 두 개다:

    p_blender[m] = u_meters · p_G ,      p_world = T_wg @ [p_G, 1]

`T_wg` 에는 **스케일 S_da3 가 박혀 있다** (`R@R.T = S²·I`, 실측 S=4.6713). 그래서 회전과 위치를
같은 4×4 로 한 번에 곱하면 안 된다 — 회전이 S 배가 되어 `det(R)=S³` 가 된다. 나눠서 건다:

    R_cv_G   = R_blender_gl @ diag(1,-1,-1)          # GL(+Y up/−Z fwd) → OpenCV(+Y down/+Z fwd)
    R_world  = (T_wg[:3,:3] / S_da3) @ R_cv_G
    p_world  = T_wg[:3,:3] @ (p_blender / u_meters) + T_wg[:3,3]

Blender world 축과 G 축이 같다는 게 전제인데, 이건 우연이 아니라 G frame 을 `up = e_z` 로 정의해
Blender Z-up 에 맞춰 놨기 때문이다 (계획서 §좌표·단위).

**프레임 수.** LBM 은 movement 종류가 정하는 고정표(`target_frame_count`)만큼 렌더하므로 우리
49와 무관하다 (camel 106, avocado-slice 110 실측). `lbm_camera_to_poses.subsample` 을 그대로 써서
**보간 없이 index 만 솎는다** — 보간하면 LBM 이 얹은 fcurve easing 이 뭉개져 jerk 지표가 실제보다
매끄럽게 나온다.

**출력은 `verify.py` 가 요구하는 폴더 한 벌**이다: `scene_graph.json`/`cloud.npz` 는 Lite 산출물을
symlink 하고(같은 점군·같은 그래프로 재야 비교가 된다), `decision.json` 은 LBM 결정을 우리 스키마로
옮겨 적고, `poses.npz` 는 위 변환 결과다. `decision_fingerprint` 는 그 decision 에서 정직하게
계산해 넣으므로 `--allow_stale_poses` 없이 통과한다.

예시:
    python scripts/vista_lbm_to_poses.py --video camel \
        --dump_dir out/lbm_demos_vista/_camdump/vista_camel \
        --vista_json out/lbm_demos_vista/camel/_vista.json \
        --out_root out/lbm_eval_vista
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path, symlink, remove

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import decision_fingerprint, roll_about, subject_centers_world  # noqa: E402
from scripts.lbm_camera_to_poses import read_dump, camera_name_of, to_opencv, subsample  # noqa: E402


def blender_to_world(poses_cv_blender: np.ndarray, u_meters: float, T_wg: np.ndarray):
    """Blender(미터, OpenCV 회전) c2w → DA3 world c2w. 회전과 위치를 따로 거는 이유는 모듈 docstring.

    `to_opencv` 가 이미 `@ diag(1,-1,-1,1)` 을 걸어 놨으므로 여기 들어오는 회전은 OpenCV 다.
    """
    scale = float(np.sqrt((T_wg[:3, :3] @ T_wg[:3, :3].T)[0, 0]))   # S_da3 (T_wg 에 박힌 스케일)
    rotation = T_wg[:3, :3] / scale
    out = np.zeros_like(poses_cv_blender)
    out[:, 3, 3] = 1.0
    out[:, :3, :3] = rotation[None] @ poses_cv_blender[:, :3, :3]
    out[:, :3, 3] = (T_wg[:3, :3] @ (poses_cv_blender[:, :3, 3] / u_meters).T).T + T_wg[:3, 3]
    determinants = np.linalg.det(out[:, :3, :3])
    assert np.allclose(determinants, 1.0, atol=1e-6), \
        f"world 회전이 정규직교가 아니다: det {determinants.min():.6f}..{determinants.max():.6f}"
    return out, scale


def view_angles(positions: np.ndarray, targets: np.ndarray, src_centers: np.ndarray):
    """`decode/build_poses._view_angles` 와 **같은 식**. import 하면 private 이라 여기 복제한다."""
    plan, source = targets - positions, targets - src_centers
    cos = np.sum(plan * source, axis=-1) / np.maximum(
        np.linalg.norm(plan, axis=-1) * np.linalg.norm(source, axis=-1), 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def link(source: str, target: str):
    """이미 있으면 지우고 다시 건다 (재실행 안전). 원본을 복사하지 않는 건 cloud.npz 가 크기 때문."""
    if path.islink(target) or path.isfile(target):
        remove(target)
    symlink(path.abspath(source), target)


def build_decision(video: str, meta: dict, poses_world: np.ndarray, look_at_world: np.ndarray,
                   T_gw: np.ndarray, report: dict):
    """LBM 결정을 `lbm_decision_v1` 로 옮겨 적는다.

    **주의: 이건 재현용이 아니라 기록용이다.** `p_G`/`look_at_G` 는 LBM 이 실제로 쓴 frame 0 pose 를
    G 로 되돌린 값이고, `trajectory.preset` 은 LBM 이 고른 movement term 이다. `build_poses` 로
    이걸 다시 디코드하면 **다른 궤적**이 나온다 (LBM 의 fcurve easing 을 우리 preset 이 모른다).
    그래서 `source` 를 `lbm_original` 로 박아 둔다 — 하류가 이걸 Lite 결정으로 오인하면 안 된다.
    """
    from scene_graph.lift import apply_transform
    p_g = apply_transform(T_gw, poses_world[:1, :3, 3])[0]
    look_g = apply_transform(T_gw, look_at_world[:1])[0]
    return {
        "format": "lbm_decision_v1", "video": video, "subject_id": meta["subject_id"],
        "source": "lbm_original", "model": meta.get("movement_term_target"),
        #    LBM 은 소스 frame0 카메라를 앵커하지 않고 자기 pose 를 고른다 — Lite 의 `board` 와 같다.
        #    (`verify.py --start_mode` 의 choices 가 이 둘뿐이라 새 값을 넣으면 하류가 막힌다.)
        "start_mode": "board",
        "keyframes": [{"t": 0, "target": meta["subject_id"],
                       "composition": {"p_G": [float(v) for v in p_g],
                                       "look_at_G": [float(v) for v in look_g],
                                       "focal_scale": 1.0},
                       "visibility": {}, "relation": None,
                       "motion_preference": meta.get("movement_term_target"),
                       "candidate_id": "lbm", "micro_ops": []}],
        "trajectory": {"preset": meta.get("movement_term_target") or "lbm",
                       "params": {}, "speed": "steady", "tracking": "world",
                       "look_at_bias": 0.0, "target_tau": None},
        "gates": report,
        "vlm": {"observation": meta.get("shot_description", ""), "reasoning": "",
                "confidence": None, "turns": None, "repairs": None},
    }


def main(args):
    with open(args.vista_json, encoding="utf-8") as file:
        meta = json.load(file)
    src_folder = path.join(args.src_root, args.video)
    with open(path.join(src_folder, "scene_graph.json"), encoding="utf-8") as file:
        graph = json.load(file)

    groups = read_dump(args.dump_dir)
    assert groups, f"카메라 덤프가 비었다: {args.dump_dir}/cam_*.jsonl"
    named = {camera_name_of(key): value for key, value in groups.items()}
    camera_name = args.camera or max(named, key=lambda k: len(named[k]))
    assert camera_name in named, f"{camera_name} 가 덤프에 없다. 있는 것: {sorted(named)}"
    records = named[camera_name]
    if len(named) > 1:
        print(f"  WARN 덤프에 카메라 {len(named)}대 — 최장 궤적 채택: {camera_name} "
              f"({sorted((k, len(v)) for k, v in named.items())})")

    num_frames = int(args.num_frames or graph["num_frames"])
    keep = subsample(len(records), num_frames)
    poses_blender = to_opencv(records)[keep]
    lens = np.array([float(r["lens_mm"]) for r in records], dtype=np.float64)[keep]
    assert len(poses_blender) == num_frames, \
        (f"덤프가 {len(records)}프레임뿐이라 {num_frames}프레임을 못 채운다 — "
         f"VideoEngineer 가 중간에 죽었는지 확인할 것")

    T_wg, T_gw = np.asarray(meta["T_wg"], float), np.asarray(meta["T_gw"], float)
    poses, scale = blender_to_world(poses_blender, float(meta["u_meters"]), T_wg)

    node = next(n for n in graph["nodes"] if n["id"] == meta["subject_id"])
    centers = subject_centers_world(node, T_wg, num_frames)
    src_c2w = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float)
    z_med = float(graph["scale"]["z_med_frame0"])

    # look-at 은 LBM 이 안 남긴다. 광축을 subject 중심까지의 거리만큼 앞으로 쏜 점을 쓴다 —
    # 진단·시각화용이고 지표에는 안 들어간다 (`verify.evaluate` 는 cam_c2w/tau/view 만 읽는다).
    distance = np.linalg.norm(centers - poses[:, :3, 3], axis=-1, keepdims=True)
    look_at = poses[:, :3, 3] + distance * poses[:, :3, 2]

    tau = np.linalg.norm(poses[:, :3, 3] - src_c2w[:, :3, 3], axis=-1) / max(z_med, 1e-9)
    view = view_angles(poses[:, :3, 3], centers, src_c2w[:, :3, 3])
    path_len_u = float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=-1).sum() / scale)
    # Lite 는 이 값이 정의상 0 이다 (`look_at_c2w` 가 g 에 대해 roll=0 으로 만든다). LBM 은 그런
    # 보장이 없으므로 **재기만 하고 assert 는 안 건다** — 0 이 아니면 그게 LBM 의 특성이다.
    up_world = np.asarray(graph["gravity"]["up_world"], dtype=float)
    rolls = np.array([roll_about(poses[f], up_world) for f in range(num_frames)])

    info = {"preset": meta.get("movement_term_target") or "lbm", "aim": "lbm", "speed": "steady",
            "tracking": "world", "tracking_ignored": False, "needs_zoom": False,
            "aim_anchor": "none", "aim_anchor_applied": False, "traj_basis": "lbm",
            "start_source": "lbm_free", "radius_world": float(distance[0, 0]),
            "radius_u": float(distance[0, 0] / scale), "look_at_bias": 0.0,
            #    `target_tau` 는 None 이다 — LBM 은 τ 를 목표로 궤적 크기를 풀지 않는다
            #    (Lite 의 이분법에 해당하는 단계가 없다). `emit.build_canonical:132` 가 이 키를
            #    무조건 읽으므로 자리는 있어야 한다.
            "tau": {"tau_max_final": round(float(tau.max()), 4), "target_tau": None},
            "roll_max_deg": round(float(np.abs(rolls).max()), 4),
            "zoom": {"focal_end": 1.0, "needs_zoom": False,
                     "lens_mm_median": round(float(np.median(lens)), 3),
                     "lens_mm_range": [round(float(lens.min()), 3), round(float(lens.max()), 3)]},
            "view_angle_max_deg": round(float(view.max()), 2),
            "path_len_u": round(path_len_u, 5), "S": scale, "z_med": z_med,
            "lbm": {"camera": camera_name, "dump_frames": len(records),
                    "kept": [int(keep[0]), int(keep[-1])]}}

    report = {"tau_max": round(float(tau.max()), 4),
              "view_angle_max_deg": round(float(view.max()), 2)}
    decision = build_decision(args.video, meta, poses, look_at, T_gw, report)

    folder = path.join(args.out_root, args.video)
    makedirs(folder, exist_ok=True)
    link(path.join(src_folder, "scene_graph.json"), path.join(folder, "scene_graph.json"))
    link(path.join(src_folder, "cloud.npz"), path.join(folder, "cloud.npz"))
    with open(path.join(folder, "decision.json"), "w", encoding="utf-8") as file:
        json.dump(decision, file, ensure_ascii=False, indent=2)
    np.savez_compressed(path.join(folder, "poses.npz"), cam_c2w=poses, look_at=look_at,
                        subject_track=centers, tau=tau, view_angle_deg=view,
                        info=json.dumps(info, ensure_ascii=False),
                        #    `verify.py:468` 이 계산하는 것과 **인자까지 같아야** 한다 —
                        #    거기는 (decision, start_mode, aim_anchor, ramp) 만 넘기고 나머지는
                        #    기본값이다. 어긋나면 `--allow_stale_poses` 없이는 못 통과한다.
                        decision_fingerprint=decision_fingerprint(
                            decision, decision["start_mode"], "subject", 12))

    print(f"{'video':<22s}{args.video}   subject {meta['subject_id']} ({meta.get('subject_label')})")
    print(f"{'camera':<22s}{camera_name}")
    print(f"{'dump frames':<22s}{len(records)} -> {num_frames}  (idx {keep[0]}..{keep[-1]})")
    print(f"{'movement term':<22s}{meta.get('movement_term_target')}")
    print(f"{'u_meters / S_da3':<22s}{meta['u_meters']:.4f} / {scale:.4f}")
    print(f"{'lens_mm':<22s}median {np.median(lens):.2f}  범위 {lens.min():.2f}..{lens.max():.2f}")
    print(f"{'tau_max':<22s}{tau.max():.4f}   (frame0 {tau[0]:.4f})")
    print(f"{'view_angle_max':<22s}{view.max():.2f} deg")
    print(f"{'path_len_u':<22s}{path_len_u:.4f}")
    print(f"{'roll_max (vs g)':<22s}{np.abs(rolls).max():.3f} deg")
    print(f"{'out':<22s}{folder}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=str)            # Lite 쪽 씬 이름 (camel 등)
    parser.add_argument("--dump_dir", required=True, type=str)         # cam_*.jsonl 디렉토리
    parser.add_argument("--vista_json", required=True, type=str)       # 어댑터가 남긴 _vista.json
    parser.add_argument("--out_root", required=True, type=str)         # 평가 폴더 root
    #    scene_graph.json / cloud.npz 를 가져올 Lite 산출물 root. **같은 점군으로 재야 비교가 된다.**
    parser.add_argument("--src_root", default=path.join(CINEMATRAJ_ROOT, "out"), type=str)
    parser.add_argument("--camera", default="", type=str)              # 비우면 최장 궤적 자동 선택
    parser.add_argument("--num_frames", default=0, type=int)           # 0 = scene_graph 의 프레임 수
    raise SystemExit(main(parser.parse_args()))
