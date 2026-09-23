"""뱅크 변이 → TRUMANS `.blend` 월드 카메라 npz. **depth warp 대신 Blender 로 렌더하기 위한 것.**

**왜 필요한가.** 지금까지 뱅크 카메라를 눈으로 확인하는 수단은 `render_bank_videos.py` 의
depth warp 뿐이었다. warp 은 소스 프레임을 다시 투영하는 것이라 **카메라가 크게 움직이면
화면의 절반 이상이 hole** 이 되고 (이 chunk 의 τ 상단 변이는 hole 0.55~0.80), 그러면 "궤적이
이상한 것"과 "warp 이 못 채운 것"을 못 가른다. TRUMANS 는 씬이 `.blend` 로 있으니 렌더러가
정답을 안다 — hole 이 원리적으로 0 이다.

**좌표 변환은 한 줄이다.** 뱅크 world 는 *소스 frame0 카메라*가 항등인 좌표계고
(`scene_graph.json:cameras.cam_c2w_world[0] == I`), 그 카메라의 blend 월드 pose 는
`out/trumans_recon/<recording>_<tag>/poses_a<NN>.npz` 에 있다. 단위도 둘 다 metre 다
(GT 분기는 Blender depth 를 그대로 쓴다). 그래서

    P_blend[f] = P_blend_src[0] @ P_bank[f]

이고 실측 잔차는 소스 카메라 자기 자신에 대해 `max|Δt| = 4.4e-16` 이다 (스케일 보정 없음 —
`S` 로 나누면 0.65 m 어긋난다). 이 검증은 `--verify` 로 매번 다시 돈다.

**어느 변이를 고르는가.** preset 마다 `tau_max` 가 가장 큰 hole 사다리 칸 하나. 뱅크 전량
(714변이)을 렌더하면 3일이 걸리고, 사다리 아랫단은 위칸의 축소판이라 새 정보가 없다.
τ 상단이 궤적이 무너지는지 볼 수 있는 유일한 지점이다.

**D91: 레이캐스트 게이트가 먼저 돈다** (`--raycast`, 기본 켜짐). 뱅크의 충돌 게이트는 소스
depth shell 위의 근사인데, TRUMANS 는 씬이 mesh 로 있으니 `trumans_scene_probe.py
--verify_poses` 로 진짜 광선을 쏜다. 통과하는 **가장 위칸**을 집고, 사다리 전 칸이 벽을 뚫는
preset 은 npz 를 아예 안 쓴다 — **저장할 카메라와 렌더할 카메라가 같아야 하기 때문**이다.
`--no_raycast` 면 예전 동작과 비트 동일.

**D103: `--aim_arms`.** 고른 변이를 조준 설정만 바꿔 여러 벌 되만들어 `<preset>__<arm>` 으로
따로 저장·렌더한다. 위치 채널은 arm 사이에서 안 변하므로(매번 assert) 렌더를 나란히 놓으면
**회전만** 다른 영상 여러 줄이 된다 — `compare_aim_timing.py` 가 숫자로 낸 것(rot_ratio 2.37 →
5.01 → 3.48)을 눈으로 보는 용도다. 비면 예전 동작과 비트 동일.

env: 아무거나 (numpy 만 쓴다). 예: `/data1/cympyc1785/miniconda3/envs/vista4d/bin/python`

사용 예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY fit/convert/bank_to_blender_poses.py --video tru_0ac97866_a08_s3f0k6 \
        --output_root out_trumans_gt --bank_dir hole_bank_k6_d77 \
        --poses_npz out_trumans_gt/tru_0ac97866_a08_s3f0k6/hole_bank_k6_d77/poses_smooth_p4.npz \
        --anchor dyn_0 --out /tmp/blendcam_a08
    bash /tmp/blendcam_a08/render.sh          # 생성된 Blender 드라이버
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)
#    `--raycast_solve` 의 크기 이분법이 `se3.scale_traj` 를 쓴다 (`lbm/presets.py:71-75` 와 같은 경로).
RECAMMASTER = path.abspath(path.join(CINEMATRAJ_ROOT, "..", "..", "..", "..",
                                     "video_generation", "tools", "recammaster"))
if RECAMMASTER not in sys.path:
    sys.path.insert(0, RECAMMASTER)
import se3  # noqa: E402
from decode.build_poses import build_poses, deroll_poses  # noqa: E402
from lbm.presets import row_preset  # noqa: E402
from scene_graph.schema import load_graph  # noqa: E402
from fit.bank.emit_bank import (FIXED_FALLBACK, decision_from_variant,  # noqa: E402
                               span_frac_for)

BLENDER ="/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender"
RECORDINGS = "/data1/cympyc1785/data/trumans/Data_release/Recordings_blend"

# D103. `--aim_arms` 용 조준 설정. `compare_aim_timing.py:ARMS` 와 **같은 표**여야 한다 —
# 그 스크립트가 낸 숫자를 이 렌더가 눈으로 보여주는 것이라 하나라도 어긋나면 대조가 깨진다.
AIM_ARMS = {
    "everyframe":     {"aim_keyframes": 0, "keyframe_ease": "smoothstep"},
    "kf6_smoothstep": {"aim_keyframes": 6, "keyframe_ease": "smoothstep"},
    "kf6_smooth_kf":  {"aim_keyframes": 6, "keyframe_ease": "smooth_kf"},
}


def chunk_paths(video: str, recon_root: str):
    """`tru_<rec8>_a<NN>_<tag>` → (manifest, poses_a<NN>.npz, .blend). 접두사로 찾는다.

    recording 은 8자리 prefix 로만 이름에 남아 있어서 full uuid 를 디렉토리에서 되찾아야 한다.
    """
    parts = video.split("_")
    assert parts[0] == "tru" and len(parts) >= 4, f"chunk 이름 규약이 아니다: {video}"
    rec8, action, tag = parts[1], parts[2], "_".join(parts[3:])
    from glob import glob
    folders = sorted(glob(path.join(recon_root, f"{rec8}-*_{tag}")))
    assert folders, f"recon 폴더를 못 찾았다: {recon_root}/{rec8}-*_{tag}"
    folder = folders[0]
    recording = path.basename(folder)[: -(len(tag) + 1)]
    manifest = path.join(folder, f"manifest_{action}.json")
    poses = path.join(folder, f"poses_{action}.npz")
    blend = path.join(RECORDINGS, recording, f"{recording}.blend")
    for p in (manifest, poses, blend):
        assert path.exists(p), f"없다: {p}"
    return manifest, poses, blend


def rank_variants(variants: list, anchor: str, presets: list, variant_ids: list = None):
    """preset -> `tau_max` **내림차순** 변이 목록 (hole 사다리를 위에서부터).

    D91. 예전에는 preset 마다 최상단 하나만 돌려줬는데, 레이캐스트 게이트가 그 칸을 떨어뜨리면
    preset 이 통째로 사라진다. 사다리 아랫단은 같은 궤적의 축소판이라 위칸이 벽을 뚫어도
    아랫칸은 안 뚫는 경우가 대부분이다 — 그래서 전 단을 후보로 들고 있다가 **통과하는 가장 위칸**
    을 집는다.
    """
    ranked = {}
    for v in variants:
        if anchor and v["anchor_id"] != anchor:
            continue
        # `--presets` 는 적힌 이름과 정식 이름 둘 다 받는다 (옛 뱅크는 `straight_ease` 라 적고
        # `dolly_in` 을 굽는다). 다만 **묶는 키는 적힌 이름 그대로** 둔다 — TRUMANS 뱅크는
        # `dolly_in`(look_at) 과 `dolly_in_dont_look`(traj) 을 둘 다 들고 있고 정식 이름이
        # `dolly_in_look_at` 로 같아서, 정식 이름으로 묶으면 두 행이 한 슬롯으로 합쳐지고
        # `<preset>.npz` 파일 이름까지 충돌한다.
        if presets and v["preset"] not in presets and row_preset(v) not in presets:
            continue
        # D102. 짝짓기용 — 옛 뱅크와 **같은 사다리 칸**을 렌더해야 비교가 성립한다. τ 최댓단은
        # 두 뱅크에서 서로 다른 칸에 앉을 수 있어서(GT truck_left 는 hole0.5, 옛것도 hole0.5 지만
        # crane_down 은 GT hole0.2 / 옛 hole0.2 처럼 preset 마다 갈린다) 칸을 이름으로 못 박는다.
        # 비면 예전 동작과 비트 동일.
        if variant_ids and str(v["variant_id"]) not in variant_ids:
            continue
        ranked.setdefault(v["preset"], []).append(v)
    for rungs in ranked.values():
        rungs.sort(key=lambda v: -float(v["tau_max"]))
    return ranked


def pick_variants(variants: list, anchor: str, presets: list):
    """preset 마다 `tau_max` 최대인 변이 하나. `anchor` 가 비면 앵커 구분 없이 전체에서."""
    ranked = rank_variants(variants, anchor, presets)
    return [ranked[k][0] for k in sorted(ranked)]


def raycast_select(ranked: dict, to_blend, blend: str, frames: tuple, subject_flags: list,
                   anchor_origin: str, args, out_dir: str):
    """레이캐스트로 **실제로 저장·렌더할 변이**를 고른다. (chosen, audit, dropped) 를 돌려준다.

    D91. 뱅크의 충돌 게이트(G1 depth-shell 재투영 / G5 recon OBB)는 in-the-wild Vista 용
    근사다. TRUMANS 는 씬이 `.blend` mesh 로 통째로 있으니 그 근사를 쓸 이유가 없다 —
    소스 카메라에는 이미 `trumans_scene_probe.py --verify_poses` 로 진짜 광선을 쏘고 있었는데
    (`trumans_to_recon.py:1022-1040`), **target 뱅크에는 한 번도 안 걸려 있었다.**

    실측으로 드러난 실패 모드 (a08, hole_bank_k6_d90 234변이): `behind_frames>0` 이 48개,
    `obb_slack<0` 이 44개인데도 D53 τ 하한(`knob_floor`)이 이분법의 "충돌 없는 최대 크기"를
    덮어써서 `status=clamped_low+tau_floor` 로 그대로 실렸다. 게다가 a08 의 벽 노드는 두께가
    0.0007 u 라 `obb_signed_distance` 가 **원리적으로** 음수를 못 내서 G5 가 벽을 못 잡는다.
    광선은 두께와 무관하다.

    Blender 기동 + 1.6 GB blend 로드가 4초라 **모든 preset 의 모든 사다리 칸을 한 번에** 넘긴다
    (`--verify_poses` 가 (K,F,4,4) 를 받는다). 프레임 루프가 바깥이라 `frame_set` 은 K 와 무관하게
    F 번뿐이다.

    사다리 칸만 훑는 게 기본이고, `--raycast_solve` 면 통과 칸 위에서 **크기를 이분법으로** 더
    푼다 (§`raycast_solve`).
    """
    flat = [(preset, i, v) for preset in sorted(ranked)
            for i, v in enumerate(ranked[preset])
            if not args.raycast_max_rungs or i < args.raycast_max_rungs]
    summary = run_probe(np.stack([to_blend(v) for _, _, v in flat]), blend, frames,
                        subject_flags, anchor_origin, args, out_dir, "rungs")
    assert len(summary) == len(flat), f"summary {len(summary)} vs 후보 {len(flat)}"

    audit, chosen, dropped = [], [], []
    taken = set()
    for (preset, rung, v), row in zip(flat, summary):
        bad = verdict(row, args)
        audit.append({"preset": preset, "rung": rung, "variant_id": str(v["variant_id"]),
                      "tau_max": float(v["tau_max"]), "scale": 1.0, "clear_frac": row["clear_frac"],
                      "min_clearance": row["min_clearance"],
                      "min_subject_dist": row.get("min_subject_dist"),
                      "min_floor_drop": row.get("min_floor_drop"),
                      "min_los_frac": row.get("min_los_frac"),
                      "max_occluded_frac": row.get("max_occluded_frac"),
                      "reject": bad, "passed": not bad})
        #    사다리를 위에서부터 훑으므로 **처음 통과한 칸**이 그 preset 의 최대 τ 다.
        if not bad and preset not in taken:
            taken.add(preset)
            chosen.append(dict(v, raycast=audit[-1]))
    for preset in sorted(ranked):
        if preset not in taken:
            rows = [a for a in audit if a["preset"] == preset]
            dropped.append({"preset": preset, "rungs": len(rows),
                            "reject": rows[0]["reject"] if rows else ["no_rung"]})
    return chosen, audit, dropped


def run_probe(stack, blend: str, frames: tuple, subject_flags: list, anchor_origin: str,
              args, out_dir: str, tag: str):
    """(K,F,4,4) blend-world 궤적 묶음 → `trumans_scene_probe.py` 의 `summary` 리스트.

    Blender 를 한 번 띄우는 단위가 이 함수다. 기동 + 1.6 GB blend 로드가 ~4 초라 **K 를 최대한
    크게 묶는 게 전부**다 — 프레임 루프가 바깥이라 `frame_set` 은 K 와 무관하게 F 번뿐이다.
    """
    from subprocess import run

    tries = path.join(out_dir, f"_raycast_poses_{tag}.npz")
    audit_json = path.join(out_dir, f"raycast_probe_{tag}.json")
    np.savez(tries, cam_c2w=np.asarray(stack, dtype=np.float64))
    f0, f1, step = frames
    command = [args.blender, "-b", blend, "--python",
               path.join(CINEMATRAJ_ROOT, "fit", "ingest", "trumans_scene_probe.py"), "--",
               "--frames", str(f0), str(f1), str(step), "--out", audit_json,
               "--verify_poses", tries, "--anchor_origin", anchor_origin,
               "--probe_distance", str(args.probe_distance), *subject_flags]
    # 밀집 LOS 는 3점 boolean 게이트와 **나란히** 돈다. 0 이면 광선 수가 예전과 같다.
    if args.los_samples > 0:
        command += ["--los_samples", str(args.los_samples), "--los_bands", str(args.los_bands)]
    log = path.join(out_dir, f"raycast_{tag}.log")
    with open(log, "w", encoding="utf-8") as file:
        code = run(command, stdout=file, stderr=file, check=False).returncode
    assert code == 0, f"레이캐스트 probe 실패 (rc={code}). 로그: {log}"
    with open(audit_json, encoding="utf-8") as file:
        return json.load(file)["summary"]


def verdict(row, args):
    """탈락 사유 목록. 빈 리스트면 통과. 임계는 소스 카메라 검증과 **같은 값**이다."""
    bad = []
    if row["clear_frac"] < args.min_clear_frac:
        bad.append(f"occluded({row['clear_frac']:.2f})")
    if row["min_clearance"] < args.min_clearance:
        bad.append(f"wall({row['min_clearance']:.3f})")
    if row.get("min_subject_dist", float("inf")) < args.min_subject_dist:
        bad.append(f"subject({row['min_subject_dist']:.3f})")
    if row.get("min_floor_drop", float("inf")) < args.min_floor_drop:
        bad.append(f"floor({row['min_floor_drop']:.3f})")
    # 밀집 LOS 게이트. `--min_los_frac 0`(기본)이면 `min_los_frac` 이 아예 없거나 있어도 통과라
    # 판정이 예전과 같다. 3점 OR(`clear_frac`)은 "몸 어딘가 한 점이라도 보이나"라 반쯤 가린
    # 구도를 1.00 으로 통과시킨다 — 이건 "몇 %가 보이나"라서 그 구도를 잡는다.
    if row.get("min_los_frac", 1.0) < args.min_los_frac:
        bad.append(f"los({row['min_los_frac']:.2f})")
    return bad


def scale_about_first(cam, s: float):
    """frame0 을 고정한 채 궤적을 SE(3) 로그 공간에서 s 배 (`se3.scale_traj`).

    blend world 에서 바로 해도 된다 — 뱅크 world → blend world 가 **왼쪽 상수 곱**이라
    frame0 상대 궤적이 두 좌표계에서 같다.
    """
    cam = np.asarray(cam, dtype=np.float64)
    rel = np.linalg.inv(cam[0]) @ cam
    return cam[0] @ se3.scale_traj(rel, float(s))


def raycast_solve(ranked: dict, to_blend, blend: str, frames: tuple, subject_flags: list,
                  anchor_origin: str, args, out_dir: str):
    """preset 마다 **레이캐스트를 통과하는 최대 크기** s∈[0,1] 을 이분법으로 푼다.

    D91. 사다리 칸만 훑는 `raycast_select` 로는 a08 push-in 계열이 전멸했다 — D53 τ 하한
    (`knob_floor`)이 이분법의 "충돌 없는 최대 크기"를 덮어써서 **4칸이 전부 같은 knob 으로
    clamp 됐기 때문**이다 (실측: dolly_in 4칸의 clearance/subject_dist 가 소수점 3자리까지
    동일). 즉 사다리에 내려갈 아랫단이 없다. 여기서 s 를 연속으로 풀면 그 바닥을 뚫는다.

    s=0 은 시작 pose 에서의 정지 hold 다. 시작 pose 는 `trumans_first_pose_board.py` 가 이미
    같은 광선으로 검증한 자리라 보통 통과하지만, **통과를 가정하지 않고 실제로 쏴서 확인**하고
    떨어지면 그 preset 을 버린다 (뱅크 시작 pose 는 board 와 다른 경로로 올 수 있다).

    회전도 같이 줄어든다 (`se3.scale_traj`). look_at preset 은 그만큼 조준이 어긋나는데,
    s→0 극한이 "시작 pose 그대로"이고 그 pose 는 정확히 조준돼 있으므로 오차는 s 와 함께
    단조로 사라진다. 프리뷰용으로 쓰는 근사이고, 정식 경로는 fit 쪽 `solve_knob` 이 이 광선을
    직접 쓰게 바꾸는 것이다 (task #97).
    """
    presets = sorted(ranked)
    base = {p: to_blend(ranked[p][0]) for p in presets}
    trace = []

    def evaluate(scales: dict, tag: str):
        keys = [p for p in presets if p in scales]
        rows = run_probe(np.stack([scale_about_first(base[p], scales[p]) for p in keys]),
                         blend, frames, subject_flags, anchor_origin, args, out_dir, tag)
        out = {}
        for p, row in zip(keys, rows):
            row = dict(row, preset=p, scale=float(scales[p]), reject=verdict(row, args))
            row["passed"] = not row["reject"]
            trace.append(row)
            out[p] = row
        return out

    #    s=0 (정지 hold) 로 바닥을 확인하고 s=1 (뱅크 그대로) 로 천장을 확인한다.
    zero, one = evaluate({p: 0.0 for p in presets}, "s000"), evaluate({p: 1.0 for p in presets}, "s100")
    alive = [p for p in presets if zero[p]["passed"]]
    dropped = [{"preset": p, "rungs": 1, "reject": ["hold_" + r for r in zero[p]["reject"]]}
               for p in presets if p not in alive]
    lo = {p: 0.0 for p in alive}
    hi = {p: 1.0 for p in alive}
    best = {p: zero[p] for p in alive}
    for p in list(alive):
        if one[p]["passed"]:
            lo[p], best[p] = 1.0, one[p]     # 뱅크 그대로 통과 — 이분법 불필요
    todo = [p for p in alive if lo[p] < 1.0]
    for it in range(int(args.raycast_iters)):
        if not todo:
            break
        mid = {p: 0.5 * (lo[p] + hi[p]) for p in todo}
        rows = evaluate(mid, f"bisect{it}")
        for p in todo:
            if rows[p]["passed"]:
                lo[p], best[p] = mid[p], rows[p]
            else:
                hi[p] = mid[p]

    chosen = []
    for p in alive:
        v = dict(ranked[p][0])
        row = best[p]
        v["raycast"] = {"preset": p, "rung": 0, "variant_id": str(v["variant_id"]),
                        "tau_max": float(v["tau_max"]), "scale": float(row["scale"]),
                        "clear_frac": row["clear_frac"], "min_clearance": row["min_clearance"],
                        "min_subject_dist": row.get("min_subject_dist"),
                        "min_floor_drop": row.get("min_floor_drop"),
                        "min_los_frac": row.get("min_los_frac"),
                        "max_occluded_frac": row.get("max_occluded_frac"),
                        "reject": [], "passed": True}
        #    정지로 무너진 것은 "그 preset 을 만들었다"고 말하면 안 된다 — 이름은 dolly_in 인데
        #    카메라가 안 움직이는 클립이 조용히 쌓이는 게 D53 이 만든 바로 그 사고다.
        if row["scale"] < args.min_raycast_scale:
            dropped.append({"preset": p, "rungs": 1,
                            "reject": [f"scale({row['scale']:.3f})<{args.min_raycast_scale}"]})
            continue
        v["_scaled_cam"] = scale_about_first(base[p], row["scale"])
        chosen.append(v)
    return chosen, trace, dropped


def main():
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)            # tru_<rec8>_a<NN>_<tag>
    parser.add_argument("--output_root", default="out_trumans_gt", type=str)
    parser.add_argument("--bank_dir", default="hole_bank_k6_d77", type=str)
    #    회전 리타이밍된 npz 를 읽으려면 여기에 경로를 준다. 비우면 뱅크의 `poses.npz`.
    parser.add_argument("--poses_npz", default="", type=str)
    parser.add_argument("--anchor", default="dyn_0", type=str)         # "" 면 앵커 구분 없음
    parser.add_argument("--presets", default=[], nargs="*", type=str)  # 비면 전량
    # 사다리 칸을 이름으로 못 박는다 (D102 짝짓기). 비면 preset 마다 τ 최댓단 — 예전 동작.
    parser.add_argument("--variant_ids", default=[], nargs="*", type=str)
    parser.add_argument("--recon_root", default=path.join(CINEMATRAJ_ROOT, "out", "trumans_recon"))
    parser.add_argument("--out", required=True, type=str)
    #    Blender 렌더 드라이버 설정. rgb 만 뽑는다 — depth/index 는 궤적 확인에 필요 없고
    #    프레임당 1.8 s 를 더 쓴다.
    parser.add_argument("--res", default=[640, 360], nargs=2, type=int)
    parser.add_argument("--rgb_samples", default=48, type=int)
    parser.add_argument("--rgb_engine", default="cycles", choices=["cycles", "eevee"], type=str)
    parser.add_argument("--rgb_cdevice", default="GPU", choices=["CPU", "GPU"], type=str)
    #    D207. 생성되는 render.sh 가 `trumans_gt_render.py --anim` 을 부른다 — 프레임마다
    #    render job 을 새로 만드는 대신 카메라를 keyframe 으로 굽고 한 번에 돌린다. 실측상
    #    프레임 시간의 95% 가 job 마다 반복되는 씬 재동기화라 여기가 제일 큰 손잡이다
    #    (`trumans_gt_render.py` §애니메이션 렌더). 기본 off = 예전 render.sh 와 글자 동일.
    parser.add_argument("--anim", action="store_true")
    parser.add_argument("--gpu", default=0, type=int)
    #    D90. preset 하나당 Blender 프로세스 하나라 (출력 PNG 이름이 프레임 번호라 한 프로세스가
    #    두 pose set 을 같은 번호로 못 쓴다) 40 preset 을 GPU 1장에 몰면 2시간이 넘는다.
    #    `--gpus 2 3 4 5` 를 주면 stride 로 나눈 `render_g<N>.sh` 를 각각 만든다.
    parser.add_argument("--gpus", default=[], nargs="*", type=int)
    #    소스 카메라를 같은 방식으로 재구성해 blend pose 와 대조한다 (좌표 규약 회귀 검사).
    parser.add_argument("--verify", dest="verify", action="store_true")
    parser.add_argument("--no_verify", dest="verify", action="store_false")
    parser.set_defaults(verify=True)
    #    D91. **레이캐스트 게이트** (TRUMANS 분기). 끄면 예전과 비트 동일 — preset 마다 τ 최대칸을
    #    그대로 집고 아무것도 안 떨어뜨린다. 켜면 `.blend` mesh 에 진짜 광선을 쏴서 통과하는
    #    가장 위칸을 집고, 전 칸이 떨어진 preset 은 **저장도 렌더도 안 한다**
    #    (사용자 지시: 최종 저장할 카메라만 렌더한다).
    parser.add_argument("--raycast", dest="raycast", action="store_true")
    parser.add_argument("--no_raycast", dest="raycast", action="store_false")
    parser.set_defaults(raycast=True)
    #    D102. 저장 직전에 중력(blend +Z) 기준 roll 을 0 으로 세운다 — 뱅크를 다시 굽지 않고
    #    `--deroll` 없이 구워진 옛 뱅크를 GT 뱅크와 같은 조건으로 놓기 위한 것. 광축·위치는
    #    안 바뀌므로 τ / hole / 레이캐스트 판정 전부 그대로다. 기본 off = 예전과 비트 동일.
    parser.add_argument("--deroll", dest="deroll", action="store_true")
    parser.add_argument("--no_deroll", dest="deroll", action="store_false")
    parser.set_defaults(deroll=False)
    #    0 이면 사다리 전 칸. 광선 수를 줄이려면 위에서 N칸만 본다 (떨어지면 그 preset 은 드롭).
    parser.add_argument("--raycast_max_rungs", default=0, type=int)
    #    D91. 사다리 칸이 τ 하한으로 전부 같은 knob 에 clamp 되면 (a08 push-in 실측) 내려갈
    #    아랫단이 없다. 이걸 켜면 크기 s∈[0,1] 을 광선으로 직접 이분법한다 (§`raycast_solve`).
    parser.add_argument("--raycast_solve", dest="raycast_solve", action="store_true")
    parser.add_argument("--no_raycast_solve", dest="raycast_solve", action="store_false")
    parser.set_defaults(raycast_solve=False)
    parser.add_argument("--raycast_iters", default=6, type=int)          # 이분법 횟수(=Blender 기동)
    #    이보다 작게 줄여야 통과하는 preset 은 **버린다** — 이름만 dolly_in 인 정지 클립 방지.
    parser.add_argument("--min_raycast_scale", default=0.15, type=float)
    parser.add_argument("--blender", default=BLENDER, type=str)
    #    임계는 **소스 카메라 검증과 같은 값**이다 (`trumans_to_recon.py:1259-1265`). 소스보다
    #    느슨하게 두면 "소스는 못 서는 자리인데 target 은 선다"가 되어 대조가 무의미해진다.
    # D120. 0.35 → 0.20 (사용자 지시 2026-09-03). 0.35 는 이미 충돌을 통과한 export 행의 80.6%
    # 를 잘라냈고(258행 중 208), 그중 `wall` 단독이 123 = 47.7% 였다. `CLEARANCE_DIRS` 는 ±z 를
    # **그대로 둔다** — 바닥/가구 윗면이 먼저 잡히는 건 알고 남기는 것이고(D118 그림), 그래서
    # 임계 쪽을 낮춘다. 예전 뱅크 재현은 `--min_clearance 0.35`.
    parser.add_argument("--min_clearance", default=0.20, type=float)      # 카메라-표면 최소(m)
    parser.add_argument("--min_floor_drop", default=0.30, type=float)     # 카메라 아래 바닥까지(m)
    parser.add_argument("--min_clear_frac", default=0.90, type=float)     # 시선 뚫린 프레임 비율
    parser.add_argument("--min_subject_dist", default=0.80, type=float)   # 카메라-피사체 최소(m)
    parser.add_argument("--probe_distance", default=1.5, type=float)      # clearance 광선 최대 거리
    #    밀집 LOS (LBM `occlusion_check` 밀도). `--los_samples 0`(기본)이면 광선 수·판정 전부
    #    예전과 같다. 켜면 프레임마다 subject 정점 N점에 한 발씩 더 쏘고 `min_los_frac` 열이
    #    붙는다. `--min_los_frac > 0` 이어야 실제로 거른다 — 먼저 열만 켜서 분포를 보고 임계를
    #    정하라고 두 손잡이를 나눠 놨다.
    parser.add_argument("--los_samples", default=0, type=int)
    parser.add_argument("--los_bands", default=5, type=int)
    parser.add_argument("--min_los_frac", default=0.0, type=float)
    #    D103. 조준 arm 대조. 쉼표 구분 (`AIM_ARMS` 키). 주면 고른 변이를 arm 마다 **되만들어**
    #    `<preset>__<arm>` 으로 따로 저장·렌더한다 — 위치는 arm 사이에서 안 변하므로
    #    (`--aim_arms` 경로가 매번 assert 한다) 렌더 세 줄의 차이는 회전뿐이다.
    #    비면 예전 동작과 비트 동일 (뱅크 `poses.npz` 를 그대로 쓴다).
    parser.add_argument("--aim_arms", default="", type=str)
    args = parser.parse_args()

    bank_folder = path.join(args.output_root, args.video, args.bank_dir)
    bank = json.load(open(path.join(bank_folder, "bank.json"), encoding="utf-8"))
    graph = json.load(open(path.join(args.output_root, args.video, "scene_graph.json"),
                           encoding="utf-8"))
    manifest_path, src_poses_path, blend = chunk_paths(args.video, args.recon_root)
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    src_blend = np.load(src_poses_path)["cam_c2w"].astype(np.float64)   # blend world, OpenCV c2w
    anchor_c2w = src_blend[0]

    if args.verify:
        #    뱅크 world 의 소스 카메라를 anchor 로 되돌리면 blend pose 와 같아야 한다.
        src_bank = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64)
        back = anchor_c2w @ src_bank
        err_t = float(np.abs(back[:, :3, 3] - src_blend[:, :3, 3]).max())
        err_r = float(np.abs(back[:, :3, :3] - src_blend[:, :3, :3]).max())
        assert err_t < 1e-6 and err_r < 1e-6, \
            f"뱅크 world -> blend world 변환이 안 맞는다: |Δt|={err_t:.3e} |ΔR|={err_r:.3e}"

    stored = np.load(args.poses_npz or path.join(bank_folder, "poses.npz"), allow_pickle=True)
    index = {str(v): i for i, v in enumerate(stored["variant_id"])}
    poses_all = np.asarray(stored["cam_c2w"], dtype=np.float64)

    makedirs(args.out, exist_ok=True)
    deroll_skipped = {}     # variant_id -> 시선이 중력축과 평행해 roll 이 정의 안 되는 프레임 수
    #    chunk 의 blend 프레임 번호. `--frames start end step` 로 그대로 넘긴다.
    f0, f1 = manifest["frames"]
    step = int(manifest["frame_step"])
    lens = 25.0    # scene_graph 의 fl_x 666.7 = 25/36*960 로 확인된 값 (sensor 36 mm)

    def to_blend_cam(bank_cam, vid: str):
        """뱅크 world c2w (F,4,4) → blend world c2w. anchor 한 번, 그리고 선택적 deroll."""
        cam = anchor_c2w @ np.asarray(bank_cam, dtype=np.float64)
        #    D102. 옛 뱅크(context depth fit)는 `--deroll` 없이 구워져서 지평선이 기운다 —
        #    실측 `tru_0ac97866_a08_s3f0k6`: orbit_left 21.8°, truck_left 15.0°, GT 뱅크는 0.2°.
        #    두 arm 을 나란히 놓으면 "충돌 판정이 달라서 이동량이 다르다"가 그 기울기에 묻힌다.
        #    여기서 펴도 **광축과 위치는 한 자리도 안 변하므로**(`deroll_poses` docstring)
        #    레이캐스트 게이트 결과는 그대로다 — 바뀌는 건 롤뿐이다. 기본 off = 예전과 비트 동일.
        if args.deroll:
            cam = np.array(cam, dtype=np.float64)
            #    blend world 는 Z-up 이다 (`selection.json:floor_z` 가 그 world 의 바닥 높이).
            deroll_skipped[vid] = deroll_poses(cam, (0.0, 0.0, 1.0), np.zeros(len(cam)))
        return cam

    def to_blend(v):
        vid = str(v["variant_id"])
        assert vid in index, f"poses npz 에 없는 변이: {vid}"
        return to_blend_cam(poses_all[index[vid]], vid)

    # D103. 조준 arm 경로. 비면 아래 `arm_cams` 를 아무도 안 부르므로 예전 동작 그대로다.
    arms = [a for a in args.aim_arms.split(",") if a]
    for a in arms:
        assert a in AIM_ARMS, f"모르는 arm: {a!r} (가능: {sorted(AIM_ARMS)})"
    #    `raycast_solve` 는 뱅크 궤적을 축소해서 `_scaled_cam` 에 넣는데, arm 은 결정에서부터
    #    되만드므로 그 축소가 반영이 안 된다 — 저장한 카메라와 게이트를 통과한 카메라가 달라진다.
    assert not (arms and args.raycast_solve), "--aim_arms 와 --raycast_solve 는 같이 못 쓴다"
    graph_typed = load_graph(path.join(args.output_root, args.video, "scene_graph.json")) \
        if arms else None
    nodes = {n["id"]: n for n in graph["nodes"]}
    fixed = bank["fixed"]

    def arm_cams(v):
        """변이 하나 → {arm: blend world c2w}. `fixed` 에서 aim 두 손잡이만 갈아끼운다.

        되만드는 인자 묶음은 `emit_bank.py:295` / `compare_aim_timing.py:182` 와 같아야 한다 —
        하나라도 다르면 "조준 차이"라고 본 것이 실은 다른 손잡이 차이가 된다.
        """
        node = nodes[v["anchor_id"]]
        decision = decision_from_variant(graph_typed, node, v, fixed)
        out = {}
        for arm in arms:
            poses, _ = build_poses(
                decision, graph_typed, board=None,
                num_frames=int(fixed.get("num_frames", bank["num_frames"])),
                orbit_span_frac=span_frac_for(node, fixed), start_mode=fixed["start_mode"],
                aim_anchor=fixed["aim_anchor"],
                aim_ramp_frames=int(fixed.get("aim_ramp_frames",
                                              FIXED_FALLBACK["aim_ramp_frames"])),
                traj_basis=str(fixed.get("traj_basis", FIXED_FALLBACK["traj_basis"])),
                keyframe_aim=str(fixed.get("keyframe_aim", FIXED_FALLBACK["keyframe_aim"])),
                smooth_passes=int(fixed.get("smooth_passes", FIXED_FALLBACK["smooth_passes"])),
                smooth_lambda=float(fixed.get("smooth_lambda",
                                              FIXED_FALLBACK["smooth_lambda"])),
                preset_tracking=bool(fixed.get("preset_tracking",
                                               bank.get("preset_tracking", False))),
                deroll=bool(fixed.get("deroll", FIXED_FALLBACK["deroll"])),
                **AIM_ARMS[arm])
            out[arm] = to_blend_cam(poses, f"{v['variant_id']}__{arm}")
        return out

    ranked = rank_variants(bank["variants"], args.anchor, args.presets, args.variant_ids)
    #    D103. arm 대조는 `aim="look_at"` 에서만 성립한다 — `traj`/`free` 에서는 `aim_keyframes=0`
    #    이 "매 프레임 조준"이 아니라 **조준 안 함**이라(`build_poses.py:724, 806`) everyframe arm 이
    #    preset 회전을 그대로 보여준다. 조용히 섞이면 표도 영상도 다른 것을 재게 된다.
    if arms:
        skipped_aim = {p: rows[0].get("aim") for p, rows in ranked.items()
                       if rows[0].get("aim") != "look_at"}
        ranked = {p: rows for p, rows in ranked.items() if p not in skipped_aim}
        for p, aim in sorted(skipped_aim.items()):
            print(f"{p:30s} {'SKIP':>7s}  aim={aim} — arm 대조는 look_at 에서만 성립한다")
    assert ranked, (f"고른 변이가 없다 (anchor={args.anchor!r}, presets={args.presets}, "
                    f"variant_ids={args.variant_ids})")

    audit, dropped = [], []
    if args.raycast:
        #    verify 도 **같은 subject** 로 돌려야 한다 (`trumans_to_recon.py:1026-1028` 과 동일한
        #    이유 — 여기만 human 이면 소품이 벽장 안이어도 시선 판정이 통과한다).
        subject = manifest.get("subject", {})
        subject_flags = ["--subject_kind", subject.get("kind", "human")]
        if subject.get("prop_names"):
            subject_flags += ["--prop_names", *subject["prop_names"]]
        solver = raycast_solve if args.raycast_solve else raycast_select
        chosen, audit, dropped = solver(
            ranked, to_blend, blend, (f0, f1, step), subject_flags,
            subject.get("anchor_origin", "obb_center"), args, args.out)
        assert chosen, ("레이캐스트를 통과한 변이가 0개다 — 이 chunk 는 뱅크 전량이 벽을 뚫는다. "
                        f"진단: {args.out}/raycast_audit.json")
    else:
        chosen = [ranked[k][0] for k in sorted(ranked)]

    rows, lines = [], []
    arm_residual = {}       # arm -> (뱅크 poses.npz 대비 위치 최대차, 회전 최대차)
    for v in chosen:
        vid = str(v["variant_id"])
        #    `raycast_solve` 가 크기를 줄였으면 **줄인 궤적**을 저장한다 — 저장한 카메라와
        #    렌더한 카메라와 게이트를 통과한 카메라가 셋 다 같아야 한다.
        cam = v.get("_scaled_cam")
        cam = to_blend(v) if cam is None else np.asarray(cam, dtype=np.float64)
        built = arm_cams(v) if arms else {}
        for arm, arm_cam in built.items():
            #    **위치가 같아야 대조가 성립한다.** 이게 깨지면 세 줄 영상은 "조준 차이"가 아니라
            #    다른 궤적 셋을 보여주는 것이 된다 (`compare_aim_timing.py` 의 `dpos_max` 와 같은 검사).
            dpos = float(np.abs(arm_cam[:, :3, 3] - cam[:, :3, 3]).max())
            assert dpos < 1e-6, f"{vid}/{arm}: arm 사이 위치가 다르다 |Δt|={dpos:.3e}"
            arm_residual.setdefault(arm, []).append(
                (dpos, float(np.abs(arm_cam[:, :3, :3] - cam[:, :3, :3]).max())))
        for name, out_cam in ([(v["preset"], cam)] if not arms
                              else [(f"{v['preset']}__{a}", built[a]) for a in arms]):
            npz_path = path.join(args.out, f"{name}.npz")
            np.savez(npz_path, cam_c2w=out_cam.astype(np.float64))
            # `preset` 은 npz 파일 이름과 같아야 하므로 적힌 이름 그대로, `preset_canonical` 이
            # 그 행이 실제로 만든 카메라의 이름이다 (읽는 쪽은 이쪽을 쓸 것).
            rows.append({"preset": name, "preset_canonical": row_preset(v),
                         "variant_id": vid, "npz": npz_path,
                         "raycast": v.get("raycast"),
                         "tau_max": float(v["tau_max"]),
                         "hole_fraction": float(v["hole_fraction"]),
                         "hole_delta": float(v["hole_delta"]),
                         "path_len_u": float(v["path_len_u"]),
                         "view_angle_max_deg": float(v["view_angle_max_deg"]),
                         "subject_area_med": float(v["subject_area_med"]),
                         "subject_in_frame": float(v["subject_in_frame"]),
                         "aim": v.get("aim")})
            lines.append(f'run "{name}" "{npz_path}"')

    #    소스 카메라 자체도 한 칸으로 넣는다 — "원본이 어떻게 생겼나"의 기준선이다.
    src_npz = path.join(args.out, "_source.npz")
    np.savez(src_npz, cam_c2w=src_blend)
    lines.insert(0, f'run "_source" "{src_npz}"')

    script = path.join(CINEMATRAJ_ROOT, "viz", "trumans_gt_render.py")
    num_frames = len(range(f0, f1 + 1, step))

    def write_driver(driver: str, gpu: int, body: list):
        with open(driver, "w", encoding="utf-8") as file:
            file.write(f"""#!/bin/bash
# `bank_to_blender_poses.py` 가 생성. preset 마다 Blender 를 한 번씩 띄운다.
# OptiX 커널은 ~/.cache/cycles 에 남으므로 빌드 비용은 첫 프로세스에서만 든다.
#
# skip 판정을 `summary.json` 이 아니라 **rgb PNG 개수**로 하는 이유: `trumans_gt_render.py` 는
# summary 를 안 쓴다. 없는 파일을 보면 skip 이 영영 안 걸려 재시작이 처음부터 다시 돈다.
set -u
B={BLENDER}
BLEND={blend}
OUT={args.out}
export CUDA_VISIBLE_DEVICES={gpu}
run() {{
    name=$1; npz=$2
    if [ "$(ls "$OUT/$name/rgb" 2>/dev/null | wc -l)" = "{num_frames}" ]; then echo "[skip] $name"; return; fi
    echo "[render] $name  $(date +%H:%M:%S)"
    $B -b "$BLEND" --python {script} -- \\
        --poses "$npz" --frames {f0} {f1} {step} --passes rgb \\
        --res {args.res[0]} {args.res[1]} --lens {lens} \\
        --rgb_engine {args.rgb_engine} --rgb_samples {args.rgb_samples} \\
        --rgb_cdevice {args.rgb_cdevice} {"--anim" if args.anim else ""} \\
        --out "$OUT/$name" > "$OUT/$name.log" 2>&1
    echo "[render] $name rc=$?  $(date +%H:%M:%S)"
}}
""" + "\n".join(body) + "\n")
        return driver

    if args.gpus:
        #    소스 칸(`lines[0]`)은 첫 샤드가 맡는다 — stride 로 자연히 그렇게 된다.
        drivers = [write_driver(path.join(args.out, f"render_g{gpu}.sh"), gpu, lines[i::len(args.gpus)])
                   for i, gpu in enumerate(args.gpus)]
        driver = "  ".join(drivers)
    else:
        driver = write_driver(path.join(args.out, "render.sh"), args.gpu, lines)

    with open(path.join(args.out, "selection.json"), "w", encoding="utf-8") as file:
        json.dump({"video": args.video, "bank_dir": args.bank_dir, "anchor": args.anchor,
                   "poses_npz": args.poses_npz or path.join(bank_folder, "poses.npz"),
                   "keyframe_ease_bank": bank["fixed"].get("keyframe_ease"),
                   "blend": blend, "frames": [f0, f1, step], "lens": lens,
                   "anchor_c2w_blend": anchor_c2w.tolist(),
                   "raycast": ({"gates": {"min_clearance": args.min_clearance,
                                          "min_floor_drop": args.min_floor_drop,
                                          "min_clear_frac": args.min_clear_frac,
                                          "min_subject_dist": args.min_subject_dist},
                                "audit": audit, "dropped": dropped} if args.raycast else None),
                   "deroll": bool(args.deroll),
                   "deroll_skipped": deroll_skipped if args.deroll else None,
                   "aim_arms": ({a: AIM_ARMS[a] for a in arms} if arms else None),
                   "variants": rows},
                  file, ensure_ascii=False, indent=1)

    print(f"{'preset':30s} {'tau':>6s} {'hole':>6s} {'pathU':>6s} {'view':>6s} "
          f"{'subjA':>6s} {'inFr':>5s} {'aim':>8s} {'scale':>6s} {'clr':>6s} {'sdst':>6s} {'clrF':>5s}")
    for r in sorted(rows, key=lambda r: -r["tau_max"]):
        rc = r.get("raycast") or {}
        print(f"{r['preset']:30s} {r['tau_max']:6.3f} {r['hole_fraction']:6.3f} "
              f"{r['path_len_u']:6.3f} {r['view_angle_max_deg']:6.1f} "
              f"{r['subject_area_med']:6.3f} {r['subject_in_frame']:5.2f} {str(r['aim']):>8s} "
              + (f"{rc['scale']:6.3f} {rc['min_clearance']:6.3f} "
                 f"{(rc.get('min_subject_dist') or float('nan')):6.3f} {rc['clear_frac']:5.2f}"
                 if rc else f"{'-':>4s} {'-':>6s} {'-':>6s} {'-':>5s}"))
    #    떨어진 preset 은 **조용히 사라지면 안 된다** — 사다리 전 칸이 벽을 뚫었다는 진단이다.
    for d in dropped:
        print(f"{d['preset']:30s} {'DROPPED':>6s}  사다리 {d['rungs']}칸 전부 탈락  "
              f"(최상단 사유: {', '.join(d['reject'])})")
    #    뱅크 자신의 설정과 같은 arm 은 `poses.npz` 를 **재현**해야 한다 (dR≈0). 재현이 안 되면
    #    되만들기 인자가 어긋난 것이고, 그러면 나머지 arm 의 차이도 조준 탓이 아니다.
    for arm in arms:
        stack = np.asarray(arm_residual[arm])
        print(f"{'arm ' + arm:22s} 뱅크 대비 |Δt|={stack[:, 0].max():.2e}  "
              f"|ΔR|={stack[:, 1].max():.2e}  ({len(stack)} 변이)")
    print(f"\n{len(rows)} preset  +  소스 1  ->  {driver}")
    if args.deroll:
        print(f"{'deroll':22s} 중력(blend +Z) 기준 roll=0  "
              f"(정의 안 되는 프레임 {sum(deroll_skipped.values())} / {49 * len(deroll_skipped)})")
    if args.raycast:
        print(f"{'raycast 통과/후보':22s} {len(rows)} / {len({a['preset'] for a in audit})} preset  "
              f"({sum(a['passed'] for a in audit)} / {len(audit)} 변이)")


if __name__ == "__main__":
    main()
