"""world c2w (49,4,4) → `canonical.json` / `canonical.npz`. 기존 emitter 로 넘기는 마지막 seam.

`emit_model_cams.py` 는 이미 6개 모델 포맷을 다 안다. 우리가 할 일은 그 입력 규약
(`canonical_cams.py:178-186`) 을 정확히 맞추는 것뿐이다:

    rel c2w, OpenCV (X right / Y down / Z fwd), frame0 anchor, unit scale, n_poses=21

세 가지가 여기서 결정된다:

① **frame0 anchor** — `rel = inv(P[0]) @ P` (기본, `--no_free_start`). 그러면 `rel[0]=I` 라
   타깃 카메라가 소스 카메라와 같은 자리에서 출발한다. `--free_start` 는 `inv(src_c2w[0]) @ P` 로
   상수 offset 을 살려두는데, **ReCamMaster 는 그 offset 을 픽셀 수준에서 무시한다**(실측).
   즉 `--free_start` 는 depth-warp 계열에만 의미가 있고 prior-locked 계열엔 no-op 이다.

② **49 → 21 은 보간 없이 index pick** `rint(linspace(0,48,21))`
   (`canonical_cams.build_dl3dv:128` 과 같은 방식). `emit_model_cams.resample` 은 emit 쪽
   업샘플러라 여기서 쓰면 두 번 리샘플된다.

③ **unit scale** — `rmax = max_f |rel[f,:3,3]|` 로 나눠 `rmax=1` 로 만든다. 원래 크기는
   무차원 손잡이 `g = rmax / S` 로 meta 에 남는다. emitter 가 `--scales` 로 다시 곱하므로
   여기서 크기를 들고 있으면 두 번 곱해진다.

**zoom 은 canonical 을 통과하지 못한다.** `emit_model_cams.py` 에 intrinsics 채널이 없다 —
CameraAnything 은 소스 hfov 를 복사하고, AlayaWorld 는 고정 정규화 K 를 쓰고, 나머지는 K 를 아예
안 나른다. `focal_scale` 은 meta 에만 싣고 `zoom_dropped: true` 를 세운다 (검증 렌더러만 소비).

예시:
    python decode/emit.py --video camel
"""
import json
import sys
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import decision_fingerprint, resolve_start_mode  # noqa: E402
from lbm.presets import RECAMMASTER_ROOT  # noqa: E402

if RECAMMASTER_ROOT not in sys.path:
    sys.path.insert(0, RECAMMASTER_ROOT)
from canonical_cams import N_POSES  # noqa: E402

CONVENTION = "rel c2w, OpenCV (X right/Y down/Z fwd), frame0 anchor, unit scale"

# 모델별 경고 — 전부 이 리포의 실측에서 나왔다. manifest 를 읽는 사람이 함정을 다시 밟지 않게.
MODEL_NOTES = {
    "recammaster": "상수 offset 을 픽셀 수준에서 무시한다 — static_hold 단독 플랜과 --free_start 는 no-op.",
    "sierpinskicam": "81키 JSON 중 frame0..48 만 읽는다 — 렌더는 --rcm_render_frames 49.",
    "infcam": "배포 JSON 은 frame40 이 항등(mid anchor) — emit 의 --anchor 를 확인할 것.",
    # ReRoPE 는 emit_model_cams.py 의 --model 목록에 없다 (choices 는 alayaworld/cameraanything/
    # infcam/recammaster/sierpinskicam/trajectorycrafter 6종). 경로가 따로다.
    "rerope": "emit_model_cams.py 가 아니라 recammaster/rerope_prepare.py 로 간다. "
              "target |t| 를 1.0 으로 재정규화하므로 g 가 안 살아남고 모양 + tgt/src 비율만 전달된다.",
    "trajectorycrafter": "depth warp. native ladder 대신 --target_dxw 로 context align 권장.",
    "cameraanything": "depth warp + 소스 hfov 복사. 소스 cond 궤적이 없으면 스케일이 안 맞는다.",
}


def relativize(poses: np.ndarray, anchor: np.ndarray | None = None):
    """`rel[f] = inv(anchor) @ poses[f]`. anchor 를 안 주면 poses[0] (=frame0 anchor)."""
    poses = np.asarray(poses, dtype=np.float64)
    base = poses[0] if anchor is None else np.asarray(anchor, dtype=np.float64)
    return np.linalg.inv(base) @ poses


def pick21(rel_full: np.ndarray, n_poses: int = N_POSES):
    """보간 없이 실제 프레임 인덱스만 고른다."""
    idx = np.rint(np.linspace(0, len(rel_full) - 1, n_poses)).astype(int)
    return rel_full[idx], idx


# 이 아래면 "카메라가 아예 안 움직였다"로 본다. pose 는 u 단위(1 u ≜ S DA3 units)이므로
# 1e-6 u 는 어떤 용도로도 정지다. static_hold* 는 여기에 정확히 걸린다.
STATIC_T_EPS = 1e-6


def unit_scale(rel: np.ndarray):
    """`rmax = max|t|` 로 나눠 단위 궤적을 만든다. 회전은 건드리지 않는다 (스케일 불변).

    **translation 이 0 이면 나누지 않는다.** `static_hold`/`static_hold_dont_look` 는 카메라가
    제자리에 서서 look-at 만 돌리므로 `rmax` 가 float 잡음(실측 1.4e-17)이다. 예전처럼
    `max(rmax, 1e-12)` 로 나누면 그 잡음이 1e-5 로 증폭돼 `rel[0]=I` assert 가 터진다 —
    규약이 깨진 게 아니라 0/0 을 한 것이다. 이 경우 t 를 정확히 0 으로 두고 `rmax=0` 을 돌려
    호출부가 "스케일 손잡이가 없는 궤적"임을 알 수 있게 한다.
    """
    rel = np.asarray(rel, dtype=np.float64).copy()
    rmax = float(np.linalg.norm(rel[:, :3, 3], axis=1).max())
    if rmax < STATIC_T_EPS:
        rel[:, :3, 3] = 0.0
        return rel, 0.0
    rel[:, :3, 3] /= rmax
    return rel, rmax


def build_canonical(poses: np.ndarray, extra: dict, decision: dict, tag: str,
                    free_start: bool = False, src_c2w0: np.ndarray | None = None,
                    n_poses: int = N_POSES):
    """(canonical dict, sidecar dict). assert 가 emit 의 계약을 여기서 미리 검사한다."""
    anchor = src_c2w0 if (free_start and src_c2w0 is not None) else None
    rel_full = relativize(poses, anchor)
    rel21, idx = pick21(rel_full, n_poses)
    rel21_unit, rmax = unit_scale(rel21)
    degenerate = rmax == 0.0            # 정지 궤적 — 나눌 스케일이 없다 (unit_scale docstring)
    rel_full_unit = rel_full.copy()
    if degenerate:
        rel_full_unit[:, :3, 3] = 0.0
    else:
        rel_full_unit[:, :3, 3] /= rmax

    assert rel21_unit.shape == (n_poses, 4, 4), rel21_unit.shape  # emit_model_cams.py:424
    if not free_start:
        deviation = float(np.abs(rel21_unit[0] - np.eye(4)).max())
        assert deviation < 1e-9, f"rel[0] != I (dev {deviation:.3e}) — anchor 규약이 깨졌다"

    info = extra["info"]
    scale = float(info["S"])
    meta = {"kind": "lbm_lite", "video": decision["video"], "subject_id": decision["subject_id"],
            "preset": info["preset"], "aim": info["aim"], "speed": info["speed"],
            "tracking": info["tracking"], "tracking_ignored": info["tracking_ignored"],
            "decision_source": decision.get("source", "unknown"),
            "free_start": bool(free_start),
            "model_gauge_rmax": round(rmax, 8), "S_da3": round(scale, 8),
            "g": round(rmax / max(scale, 1e-12), 8),
            # 정지 궤적이면 이동량이 0 이라 emit 의 --scales 도, ReRoPE 의 |t| 정규화도 의미가
            # 없다. 소비자가 이 플래그로 걸러야 한다 — rel_c2w 만 보면 회전만 있는 정상 궤적처럼
            # 보여서 조용히 "카메라 안 움직이는 영상"이 나온다.
            "translation_degenerate": bool(degenerate),
            "tau_max": info["tau"]["tau_max_final"], "target_tau": info["tau"]["target_tau"],
            "view_angle_max_deg": info["view_angle_max_deg"],
            "path_len_u": info["path_len_u"], "roll_max_deg": info["roll_max_deg"],
            "frame_indices": idx.tolist(),
            "zoom_dropped": bool(info.get("needs_zoom", False)),
            "notes": MODEL_NOTES}
    canonical = {"n_poses": n_poses, "convention": CONVENTION,
                 "cameras": {tag: {"rel_c2w": rel21_unit.tolist(), **meta}}}
    sidecar = {"rel": rel21_unit, "rel_full": rel_full_unit, "look_at": extra["look_at"],
               "p_world": poses[:, :3, 3], "subject_track": extra["subject_track"],
               "tau": extra["tau_per_frame"], "S_da3": scale, "rmax": rmax}
    return canonical, sidecar, meta


def roundtrip_error(rel21_unit: np.ndarray, rel_full_unit: np.ndarray):
    """21 → 49 를 index 로 되돌렸을 때의 최대 오차. 0 이어야 한다 (보간을 안 했으므로)."""
    idx = np.rint(np.linspace(0, len(rel_full_unit) - 1, len(rel21_unit))).astype(int)
    return float(np.abs(rel_full_unit[idx] - rel21_unit).max())


def main(args):
    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    video_folder = path.join(out_root, args.video)
    with open(path.join(video_folder, "scene_graph.json"), encoding="utf-8") as file:
        graph = json.load(file)
    with open(args.decision or path.join(video_folder, "decision.json"), encoding="utf-8") as file:
        decision = json.load(file)
    loaded = np.load(args.poses or path.join(video_folder, "poses.npz"), allow_pickle=True)
    poses = loaded["cam_c2w"]
    extra = {"look_at": loaded["look_at"], "subject_track": loaded["subject_track"],
             "tau_per_frame": loaded["tau"], "info": json.loads(str(loaded["info"]))}

    # poses.npz 가 이 decision 에서 나온 게 맞는지. 루프가 decision 을 새로 쓰고 build_poses 를
    # 안 돌리면 낡은 궤적이 조용히 emit 된다 — 숫자는 멀쩡해 보여서 육안으로 안 잡힌다.
    expected = decision_fingerprint(decision, resolve_start_mode(decision, args.start_mode),
                                    args.aim_anchor, args.aim_ramp_frames)
    stored = str(loaded["decision_fingerprint"]) if "decision_fingerprint" in loaded else None
    if stored != expected and not args.allow_stale_poses:
        raise SystemExit(
            f"poses.npz 가 decision.json 과 안 맞는다 — `python -m decode.build_poses --video "
            f"{args.video}` 를 먼저 돌릴 것.\n  poses.npz : {stored}\n  decision  : {expected}")

    tag = args.tag or f"{args.video}_{extra['info']['preset']}"
    canonical, sidecar, meta = build_canonical(
        poses, extra, decision, tag, free_start=args.free_start,
        src_c2w0=np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float)[0])

    output_folder = args.out_dir or path.join(video_folder, "canonical")
    makedirs(output_folder, exist_ok=True)
    canonical_path = path.join(output_folder, "canonical.json")
    with open(canonical_path, "w", encoding="utf-8") as file:
        json.dump(canonical, file, ensure_ascii=False, indent=1)
    np.savez_compressed(path.join(output_folder, "canonical.npz"), **sidecar)

    error = roundtrip_error(sidecar["rel"], sidecar["rel_full"])
    assert error < 0.02, f"21<->49 왕복 오차 {error:.4f} — index pick 인데 0 이 아니다"

    print(f"{'tag':<22}{tag}")
    for key in ("preset", "aim", "tracking", "free_start", "model_gauge_rmax", "S_da3", "g",
                "tau_max", "view_angle_max_deg", "path_len_u", "zoom_dropped"):
        print(f"{key:<22}{meta[key]}")
    print(f"{'roundtrip_max':<22}{error:.3e}")
    print(f"{'rel[0]-I max':<22}{np.abs(sidecar['rel'][0] - np.eye(4)).max():.3e}")
    if meta["translation_degenerate"]:
        print("\n!! translation_degenerate — 카메라가 제자리에서 회전만 한다 (|t| = 0).\n"
              "   --scales 는 무의미하고, ReCamMaster/ReRoPE 는 이 플랜을 사실상 무시한다.\n"
              "   의도한 게 아니면 trajectory 단계의 tau 포화를 먼저 볼 것.")
    print(f"\n-> {canonical_path}")
    print("\n다음 명령 (모델별 경고는 canonical.json 의 notes 참고):")
    print(f"  python {path.join(RECAMMASTER_ROOT, 'emit_model_cams.py')} \\\n"
          f"    --canonical {canonical_path} --model sierpinskicam \\\n"
          f"    --scales {meta['model_gauge_rmax'] / 1.9876:.6g} --out_dir <dir>")
    print(f"  # depth warp 2종(trajectorycrafter/cameraanything)은 native ladder 대신 "
          f"--target_dxw + --align_npz 권장")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--decision", default=None, type=str)
    parser.add_argument("--poses", default=None, type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--tag", default=None, type=str)
    # 상수 offset 유지 여부. 기본은 소스 카메라에서 출발(=rel[0]=I).
    parser.add_argument("--free_start", action="store_true", default=False)
    parser.add_argument("--no_free_start", dest="free_start", action="store_false")
    # decision.json 에 start_mode 가 있으면 그쪽이 이긴다 (resolve_start_mode). 이건 그 필드가
    # 없는 옛 decision.json 용 기본값이다. 지문에 들어가므로 어긋나면 아래 가드가 잡아준다.
    parser.add_argument("--start_mode", default="source_frame0",
                        choices=["source_frame0", "board"])
    # 조준 앵커 (D26/D27) — build_poses 와 **같은 값**이어야 지문이 맞는다.
    parser.add_argument("--aim_anchor", default="subject",
                        choices=["auto", "source_frame0", "subject"])
    parser.add_argument("--aim_ramp_frames", default=12, type=int)
    # poses.npz 가 decision.json 과 안 맞아도 통과시킬지. 기본은 막는다 — 낡은 궤적이 조용히
    # emit 되는 사고가 실제로 났다 (avocado-slice: static_hold 결정 + orbit_left_arc poses).
    parser.add_argument("--allow_stale_poses", action="store_true", default=False)
    parser.add_argument("--no_allow_stale_poses", dest="allow_stale_poses",
                        action="store_false")
    main(parser.parse_args())
