"""조준(look_at)을 **매 프레임** 푸는 것과 **keyframe 6개 + 보간/평활**로 푸는 것의 대조.

WHY: 배포된 TRUMANS 뱅크 190편은 전부 `aim_keyframes=6` + `keyframe_ease=smoothstep` 으로
구워졌다 (`out_trumans/*/hole_bank_k6_d77/bank.json:fixed`). keyframe 조준은 D71 에서 "조준이
subject track 잡음을 그대로 회전으로 옮기는" 문제를 막으려고 넣은 것인데, 그 대가로 **조준이
정확한 지점은 keyframe 6개뿐**이고 그 사이는 보간 결과다. 얼마나 어긋나는지, 그리고 그 대가로
회전이 얼마나 매끄러워지는지를 같은 변이 위에서 직접 재는 코드가 없었다.

세 arm 을 **같은 변이**에서 되만든다 (`emit_bank.decision_from_variant` → `build_poses`).
위치 채널은 세 arm 이 비트 단위로 같아야 한다 — 바뀌는 건 회전뿐이고, 그게 안 맞으면 대조가
성립하지 않으므로 `dpos_max` 를 매 행에 찍는다.

    everyframe      aim_keyframes=0            매 프레임 look_at 재계산 (조준 오차 0)
    kf6_smoothstep  6 + smoothstep             배포 뱅크 그대로
    kf6_smooth_kf   6 + smooth_kf              D96 기본값 (SO(3) Laplacian 평활)

재는 것:
  · `rot_med` / `rot_max` — 프레임간 회전각(deg). `rot_ratio` = max/median 이 1 에 가까울수록
    회전이 등속이다. smoothstep 은 keyframe 구간마다 독립으로 ease 를 걸어 각속도가 keyframe
    에서 0 근처로 떨어졌다 구간 중앙에서 최대가 된다 — 이 비가 그걸 잡는다.
  · `accel_p95` — |Δ(프레임간 회전각)| 의 p95 (deg/f²). 각가속도 불연속 = 눈에 보이는 덜컥임.
  · `aim_err_med` / `aim_err_max` — everyframe arm 의 광축과 이룬 각(deg). "정확한 조준에서
    얼마나 벗어났나"다. keyframe 에서만 잰 값이 `aim_err_kf` — smoothstep/arclen_kf 는 여기서
    정확히 0 이고 smooth_kf 는 평활 때문에 0 이 아니다 (그게 smooth_kf 의 비용이다).
  · `sif` — subject track center 를 그 arm 의 카메라로 **투영**해 중앙 `center_box` 안에
    들어오는 프레임 비율. 렌더가 아니라 기하라 가림은 안 본다 (세 arm 의 위치가 같으므로
    가림 차이는 조준 방향 차이만큼이고, 그건 `aim_err` 가 이미 재고 있다).

**`aim == "look_at"` 변이만 대조한다.** `aim` 은 세 값이고 나머지 둘에서는 `aim_keyframes=0` 이
"매 프레임 조준"을 뜻하지 않는다 (`build_poses.py:724, 730-732, 806`):

    look_at   k=0 → 매 프레임 look_at 재계산.                     ← 유일하게 대조가 성립
    free      k=0 이 강제됨 (`:724`). 세 arm 이 동일 — 조준 자체가 없다.
    traj      k=0 → 마지막 `else` 로 떨어져 **조준 안 함**(preset 회전 그대로), k=6 → `preset_rel`.
              즉 everyframe arm 이 "조준 0회" 라 aim_err 이 preset 회전량을 재게 된다.

`traj`/`free` 는 개수만 찍고 제외한다 (`--keep_free` 는 free 를 다시 넣는 디버그용).

env: `da3` / `vista4d` 아무거나 (GPU 안 쓴다 — 렌더 0회)

사용 예시:
    # TRUMANS 12편 × 조준 preset 전량
    python eval/compare_aim_timing.py --out_root out_trumans --num_videos 12 \\
      --out results/20260901_aim_timing

    # preset 을 좁혀서
    python eval/compare_aim_timing.py --out_root out_trumans --num_videos 4 \\
      --presets orbit_left,orbit_right,push_in_arc_left,crane_up --out /tmp/aimt
"""
import json
import sys
from argparse import ArgumentParser
from os import listdir, makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import build_poses, keyframe_indices                    # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402
from fit.bank.emit_bank import (FIXED_FALLBACK, decision_from_variant,           # noqa: E402
                               span_frac_for)
from fit.bank.sample_camera_bank import register_external                        # noqa: E402

# arm 정의. `everyframe` 이 기준(reference)이므로 **항상 첫 번째**여야 한다.
ARMS = [
    ("everyframe",     {"aim_keyframes": 0, "keyframe_ease": "smoothstep"}),
    ("kf6_smoothstep", {"aim_keyframes": 6, "keyframe_ease": "smoothstep"}),
    ("kf6_smooth_kf",  {"aim_keyframes": 6, "keyframe_ease": "smooth_kf"}),
]


def rotation_angles(poses: np.ndarray):
    """프레임간 상대 회전각 (n-1,) deg. `R_f^T R_{f+1}` 의 축각."""
    rel = np.einsum("fji,fjk->fik", poses[:-1, :3, :3], poses[1:, :3, :3])
    trace = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def axis_angle_to(a: np.ndarray, b: np.ndarray):
    """두 광축 스택 (n,3) 사이 각 (n,) deg."""
    cos = np.clip(np.einsum("fi,fi->f", a, b), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def project_subject(poses: np.ndarray, centers: np.ndarray, K: np.ndarray,
                    width: int, height: int):
    """subject center 를 각 프레임 카메라로 투영 → (정규화 offset (n,2), 화면 앞 여부 (n,)).

    offset 은 화면 중심 기준 반폭/반높이 대비 비율이라 `|offset| <= center_box` 가 곧
    "중앙 center_box 상자 안"이다. OpenCV 규약(+Z 전방)이라 `z <= 0` 은 카메라 뒤다.
    """
    rel = centers - poses[:, :3, 3]
    cam = np.einsum("fji,fj->fi", poses[:, :3, :3], rel)      # R^T (p - t)
    front = cam[:, 2] > 1e-9
    z = np.where(front, cam[:, 2], 1.0)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([(u - width / 2.0) / (width / 2.0),
                     (v - height / 2.0) / (height / 2.0)], axis=-1), front


def arm_metrics(poses: np.ndarray, ref_poses: np.ndarray, kf: list, centers: np.ndarray,
                K: np.ndarray, width: int, height: int, center_box: float):
    """arm 하나의 지표 dict. `ref_poses` 는 everyframe arm (자기 자신이면 조준 오차가 0)."""
    steps = rotation_angles(poses)
    err = axis_angle_to(poses[:, :3, 2], ref_poses[:, :3, 2])
    offset, front = project_subject(poses, centers, K, width, height)
    inside = front & (np.abs(offset).max(axis=1) <= center_box)
    # 카메라 뒤(z<=0)면 투영이 뒤집혀 offset 이 의미 없다 — off_max 는 앞쪽 프레임만 센다.
    off = np.abs(offset).max(axis=1)[front]
    return {"rot_med": float(np.median(steps)), "rot_max": float(steps.max()),
            "rot_ratio": float(steps.max() / max(np.median(steps), 1e-9)),
            "accel_p95": float(np.percentile(np.abs(np.diff(steps)), 95)),
            "aim_err_med": float(np.median(err)), "aim_err_max": float(err.max()),
            # 분해: frame 0 은 keyframe arm 이 **일부러** 소스 회전으로 못 박은 값이라
            # (`build_poses.py:770` `poses[0][:3,:3] = c2w_start[:3,:3]`) 보간 오차가 아니다.
            # 배포 뱅크는 `aim_anchor="subject"` 라 everyframe arm 에는 램프가 안 걸린다
            # (`:828` 조건이 `source_frame0` 요구) — 그래서 frame 0 에서 둘이 off-axis 각만큼 벌어진다.
            "aim_err_f0": float(err[0]),
            "aim_err_kf": float(err[kf].max()),
            "aim_err_kf_rest": float(err[kf[1:]].max()),      # frame 0 을 뺀 keyframe 위 오차
            "aim_err_post": float(err[kf[1]:].max()),         # 첫 keyframe 구간 이후 전 프레임
            "sif": float(inside.mean()),
            "off_max": float(off.max()) if off.size else float("nan"),
            "behind_frac": float(1.0 - front.mean()),
            "dpos_max": float(np.abs(poses[:, :3, 3] - ref_poses[:, :3, 3]).max())}


def variants_of(bank: dict, keep_preset, keep_status, keep_free: bool, limit: int):
    """뱅크에서 대조할 변이 고르기 → (행, 제외 카운트 dict).

    `aim != "look_at"` 은 everyframe arm 이 "매 프레임 조준"이 아니라서 제외한다 (모듈 docstring).
    """
    rows, skipped = [], {"traj": 0, "free": 0}
    for v in bank["variants"]:
        if keep_preset and v["preset"] not in keep_preset:
            continue
        if keep_status and v["status"] not in keep_status:
            continue
        aim = v.get("aim")
        if aim != "look_at" and not (keep_free and aim == "free"):
            skipped[aim] = skipped.get(aim, 0) + 1
            continue
        rows.append(v)
    if limit and len(rows) > limit:
        rows = [rows[i] for i in np.rint(np.linspace(0, len(rows) - 1, limit)).astype(int)]
    return rows, skipped


def run_video(video: str, out_root: str, bank_dir: str, keep_preset, keep_status,
              keep_free: bool, limit: int, center_box: float):
    """영상 1편 → 행 리스트. 뱅크/그래프가 없으면 빈 리스트."""
    folder = path.join(out_root, video, bank_dir)
    bank_path, graph_path = path.join(folder, "bank.json"), \
        path.join(out_root, video, "scene_graph.json")
    if not (path.isfile(bank_path) and path.isfile(graph_path)):
        return [], {}
    bank = json.load(open(bank_path))
    graph = load_graph(graph_path)
    ext = bank.get("external_shapes") or {}
    if ext.get("path"):
        register_external(ext["path"], ext.get("aim") or "traj")

    fixed = dict(bank["fixed"])
    num_frames = int(fixed.get("num_frames", bank["num_frames"]))
    nodes = {n["id"]: n for n in graph["nodes"]}
    K = np.asarray(graph["cameras"]["K"], dtype=float)
    K = K[0] if K.ndim == 3 else K                     # fixed_focal 뱅크는 frame0 K 고정
    width, height = int(graph["width"]), int(graph["height"])
    kf = keyframe_indices(num_frames, 6)

    rows, skipped = variants_of(bank, keep_preset, keep_status, keep_free, limit)
    out = []
    for v in rows:
        node = nodes[v["anchor_id"]]
        decision = decision_from_variant(graph, node, v, fixed)
        built = {}
        for arm, override in ARMS:
            poses, extra = build_poses(
                decision, graph, board=None, num_frames=num_frames,
                orbit_span_frac=span_frac_for(node, fixed), start_mode=fixed["start_mode"],
                aim_anchor=fixed["aim_anchor"],
                aim_ramp_frames=int(fixed.get("aim_ramp_frames",
                                              FIXED_FALLBACK["aim_ramp_frames"])),
                traj_basis=str(fixed.get("traj_basis", FIXED_FALLBACK["traj_basis"])),
                keyframe_aim=str(fixed.get("keyframe_aim", FIXED_FALLBACK["keyframe_aim"])),
                smooth_passes=int(fixed.get("smooth_passes",
                                            FIXED_FALLBACK["smooth_passes"])),
                smooth_lambda=float(fixed.get("smooth_lambda",
                                              FIXED_FALLBACK["smooth_lambda"])),
                preset_tracking=bool(fixed.get("preset_tracking",
                                               bank.get("preset_tracking", False))),
                deroll=bool(fixed.get("deroll", FIXED_FALLBACK["deroll"])),
                **override)
            built[arm] = (poses, extra)
        ref_poses = built[ARMS[0][0]][0]
        centers = np.asarray(built[ARMS[0][0]][1]["subject_track"], dtype=float)
        for arm, _ in ARMS:
            out.append({"video": video, "variant_id": v["variant_id"], "preset": v["preset"],
                        "anchor": v["anchor_id"], "status": v["status"],
                        "aim": built[arm][1]["info"]["aim"], "arm": arm,
                        **arm_metrics(built[arm][0], ref_poses, kf, centers, K,
                                      width, height, center_box)})
    return out, skipped


def summarize(rows: list, key: str):
    """arm 별 중앙값 표. 지표가 변이마다 크게 다르므로 평균이 아니라 중앙값이다."""
    arms = [a for a, _ in ARMS]
    cols = ["rot_med", "rot_max", "rot_ratio", "accel_p95",
            "aim_err_med", "aim_err_max", "aim_err_f0", "aim_err_kf", "aim_err_kf_rest",
            "aim_err_post", "sif", "off_max", "behind_frac", "dpos_max"]
    table = {}
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm and (key is None or r["preset"] == key)]
        if sel:
            table[arm] = {"n": len(sel),
                          **{c: float(np.median([r[c] for r in sel])) for c in cols}}
    return table, cols


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", default="out_trumans", type=str)   # 점군/뱅크 루트
    parser.add_argument("--videos", default="", type=str)                # 쉼표 구분, 없으면 자동
    parser.add_argument("--num_videos", default=12, type=int)            # 자동 선택 편수
    parser.add_argument("--bank_dir", default="hole_bank_k6_d77", type=str)
    parser.add_argument("--presets", default="", type=str)               # 쉼표 구분 필터
    parser.add_argument("--status", default="solved", type=str)          # "all" 이면 전량
    parser.add_argument("--max_variants", default=40, type=int)          # 영상당 상한 (0=무제한)
    parser.add_argument("--center_box", default=0.80, type=float)        # subject_in_frame 상자
    parser.add_argument("--keep_free", dest="keep_free", action="store_true")
    parser.add_argument("--no_keep_free", dest="keep_free", action="store_false")
    parser.set_defaults(keep_free=False)
    parser.add_argument("--out", default="", type=str)                   # 결과 폴더 (비면 저장 안 함)
    args = parser.parse_args()

    root = args.out_root if path.isabs(args.out_root) \
        else path.join(CINEMATRAJ_ROOT, args.out_root)
    if args.videos:
        videos = [v for v in args.videos.split(",") if v]
    else:
        videos = sorted(v for v in listdir(root)
                        if path.isfile(path.join(root, v, args.bank_dir, "bank.json")))
        if args.num_videos:
            videos = [videos[i] for i in np.rint(
                np.linspace(0, len(videos) - 1, min(args.num_videos, len(videos)))).astype(int)]

    keep_preset = set(args.presets.split(",")) if args.presets else None
    keep_status = set(args.status.split(",")) if args.status != "all" else None
    rows, dropped = [], {}
    for i, video in enumerate(videos):
        got, skipped = run_video(video, root, args.bank_dir, keep_preset, keep_status,
                                 args.keep_free, args.max_variants, args.center_box)
        rows += got
        for k, n in skipped.items():
            dropped[k] = dropped.get(k, 0) + n
        drop_txt = " ".join(f"{k}={n}" for k, n in sorted(skipped.items()) if n)
        print(f"  [{i + 1}/{len(videos)}] {video}  변이 {len(got) // len(ARMS)}  "
              f"(제외 {drop_txt or '없음'})")

    assert rows, "대조할 변이가 없다 — --status all / --presets 를 확인할 것"
    n_var = len({(r["video"], r["variant_id"]) for r in rows})
    table, cols = summarize(rows, None)

    drop_txt = " ".join(f"{k}={n}" for k, n in sorted(dropped.items()) if n)
    print(f"\n영상 {len(videos)}편 · aim=look_at 변이 {n_var}개 · 제외 {drop_txt or '없음'} "
          f"· center_box {args.center_box}")
    print(f"\n{'arm':16s} {'n':>5s} " + " ".join(f"{c:>11s}" for c in cols))
    for arm, _ in ARMS:
        if arm in table:
            r = table[arm]
            print(f"{arm:16s} {r['n']:5d} " + " ".join(f"{r[c]:11.4f}" for c in cols))

    presets = sorted({r["preset"] for r in rows})
    print(f"\npreset 별 (중앙값) — rot_ratio / accel_p95 / aim_err_f0 / aim_err_post")
    print(f"{'preset':26s} " + " ".join(f"{a:>34s}" for a, _ in ARMS))
    for p in presets:
        sub, _ = summarize(rows, p)
        cells = []
        for arm, _ in ARMS:
            r = sub.get(arm)
            cells.append("  ---" if r is None else
                         f"{r['rot_ratio']:7.2f} {r['accel_p95']:8.4f} "
                         f"{r['aim_err_f0']:8.3f} {r['aim_err_post']:7.3f}")
        print(f"{p:26s} " + " ".join(f"{c:>34s}" for c in cells))

    if args.out:
        out_dir = args.out if path.isabs(args.out) else path.join(CINEMATRAJ_ROOT, args.out)
        makedirs(out_dir, exist_ok=True)
        target = path.join(out_dir, "aim_timing.json")
        json.dump({"format": "aim_timing_v1", "out_root": args.out_root,
                   "bank_dir": args.bank_dir, "videos": videos, "arms": dict(ARMS),
                   "center_box": args.center_box, "status": args.status,
                   "n_variants": n_var, "dropped_by_aim": dropped,
                   "summary": table, "rows": rows},
                  open(target, "w"), ensure_ascii=False, indent=1)
        print(f"\n-> {target}")


main()
