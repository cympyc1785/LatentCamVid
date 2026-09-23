"""work dir 의 `poses_aNN.npz` 를 배포본 `cameras.npz` 에서 되돌린다.

왜 필요한가 — `trumans_lite_bank.py:208-210` 은 `--work` 를 `--video_suffix` 가 있을 때만
하위 `trumans_to_recon.py` 로 전달한다. suffix 없이 다른 `--work` 를 주면 그 인자는 조용히
무시되고 하위가 제 기본값(`out/trumans_recon`)에 쓴다. 렌즈만 바꾼 파일럿을 그렇게 돌리면
**기존 뱅크의 work dir 을 덮어쓴다.** 2026-08-26 에 18mm 파일럿이 `00add26c` a00 에서
정확히 이 사고를 냈다.

되돌릴 수 있는 근거 — `audit_lite_framing.py:214-222` 가 work dir 에서 읽는 건
`probe_aNN.json["human_track"]` 과 `poses_aNN.npz["cam_c2w"]` 둘뿐이고, 후자는 배포본
`<eval_data>/eval_data/recon_and_seg/<video>/cameras.npz["cam_c2w"]` 에 같은 값이 남아 있다.
`aim` 은 아무도 안 읽으므로 현재 파일 것을 그대로 둔다.

손상본은 **지우지 않고** `poses_aNN.npz.<backup_suffix>` 로 남긴다.

사용 예시:
  # 무엇이 어긋나 있는지만 본다 (기본값 — 아무것도 안 쓴다)
  python fit/ingest/repair_recon_poses.py --video tru_00add26c_a00

  # 실제로 되돌린다
  python fit/ingest/repair_recon_poses.py --video tru_00add26c_a00 --apply

  # 뱅크 매니페스트 전량을 훑어 어긋난 클립만 찾는다
  python fit/ingest/repair_recon_poses.py --bank_manifest out/trumans_lite_bank/bank_manifest.json
"""
import json
import shutil
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, HERE)
from trumans_to_recon import look_at_c2w, smooth_track        # noqa: E402  frame0 을 짓는 그 식


def rebuild_aim(probe_path, base, window, bias):
    """`synth_source_path` (`trumans_to_recon.py:458-479`) 의 aim 을 probe 에서 다시 만든다.

    npz 에 남은 `aim` 을 그냥 못 쓰는 이유 — 손상된 클립은 `aim` 도 같이 덮였고, 그 사이
    기본값이 바뀌었다. 덮어쓴 실행은 probe 에 `anchor_origin:"obb_center"` 를 남겼고
    `--aim_bias` 기본이 0.20 이라 `aim` 이 `center + 0.20·h` 가 됐는데, 원본 뱅크는
    `anchor_origin` 키가 없던 시절이라 `chest`(=`body_points[0]`) + bias 0 이었다.
    그 `aim` 으로 앵커를 세우면 프레이밍이 미묘하게(obb_area 2.8%) 어긋난다.

    반면 `human_track` 은 코드가 바뀌어도 비트 단위로 같다(재실행 diff 0.000e+00 실측).
    그래서 원본 규약(chest / window 11 / bias 0)만 알면 `aim` 을 정확히 되짚을 수 있다.
    그 규약은 온전한 클립 a01~a05 에서 `aim` 을 0.0000 으로 재현해 찾아낸 값이다.
    """
    with open(probe_path, encoding="utf-8") as file:
        probe = json.load(file)
    track = probe["human_track"]
    key = "body_points" if base == "chest" else "center"
    raw = np.asarray([(entry[key][0] if key == "body_points" else entry[key])
                      for entry in track], dtype=np.float64)
    smooth = smooth_track(raw, int(window))
    return smooth + np.array([0.0, 0.0, bias * float(probe["human_height"])])


def entries_from_manifest(manifest_path):
    """뱅크 매니페스트에서 (video, recording, action) 을 뽑는다."""
    with open(manifest_path, encoding="utf-8") as file:
        data = json.load(file)
    #    뱅크는 `entries`, 감사 산출물 등 다른 형태는 `clips` 나 리스트 그대로다.
    rows = data
    if isinstance(data, dict):
        rows = data.get("entries", data.get("clips", []))
    out = []
    for row in rows:
        if not isinstance(row, dict) or "video" not in row:
            continue
        if row.get("status") not in (None, "ok"):
            continue
        out.append((row["video"], row["recording"], int(row["action"])))
    return out


UUID_LEN = 36                                   # 8-4-4-4-12. TRUMANS recording id 길이


def entry_from_video(video, work):
    """`tru_<rec8>_a03[<suffix>]` 이름만으로 work dir 을 역추적한다.

    work dir 이름은 `trumans_lite_bank.py:185` 가 `recording + video_suffix` 로 짓는다.
    UUID 앞 8자만으로 고르면 같은 recording 의 suffix 변종(`..._s3`, `..._s3f0`)까지 걸려
    3개가 나온다 — UUID 길이가 고정이므로 `dir[36:]` 이 곧 그 suffix 다. 거기까지 맞춘다.
    """
    parts = video.split("_")
    if len(parts) < 3 or parts[0] != "tru":
        raise SystemExit(f"video 이름을 못 읽었다: {video}")
    prefix, action = parts[1], int(parts[2].lstrip("a")[:2])
    #    `aNN` 뒤에 남은 게 video_suffix 다 (없으면 빈 문자열).
    suffix = ("_" + "_".join(parts[3:])) if len(parts) > 3 else ""
    from os import listdir
    hits = [d for d in listdir(work)
            if d.startswith(prefix) and len(d) >= UUID_LEN and d[UUID_LEN:] == suffix]
    if len(hits) != 1:
        raise SystemExit(f"{prefix} + suffix '{suffix}' 에 맞는 work dir 이 "
                         f"{len(hits)}개다: {hits}")
    return video, hits[0], action


def check_one(video, recording, action, args):
    poses = path.join(args.work, recording, f"poses_a{action:02d}.npz")
    scene = path.join(args.eval_data, "eval_data", "recon_and_seg", video)
    cameras = path.join(scene, "cameras.npz")
    if not path.exists(poses) or not path.exists(cameras):
        return {"video": video, "status": "missing", "delta": float("nan")}

    current = np.load(poses)
    good = np.load(cameras)["cam_c2w"]
    if current["cam_c2w"].shape != good.shape:
        return {"video": video, "status": "shape_mismatch", "delta": float("nan")}

    #    배포본은 **재앵커된** 것(frame0 = I)이고 work 의 poses 는 재앵커 전 Blender world 다
    #    (`audit_lite_framing.py:222` 주석 그대로 — probe 의 `human_track` 과 같은 프레임이라야
    #    한다). 그래서 raw 끼리 비교하면 전량이 어긋난 것처럼 보인다. 앵커를 벗겨서 비교한다.
    raw = current["cam_c2w"]
    rel = np.linalg.inv(raw[0]) @ raw
    delta = float(np.abs(rel - good).max())
    if delta <= args.tolerance and not args.force:
        return {"video": video, "status": "ok", "delta": delta}
    if not args.apply:
        return {"video": video, "status": "DRIFT", "delta": delta}

    #    앵커(원래 frame0 c2w, Blender world)는 배포본이 이미 벗겨버려서 거기엔 없다.
    #    대신 무사한 `manifest_aNN.json` 의 `candidate.position` 과 probe 에서 되짚은
    #    `aim[0]` 로 다시 세운다 — `trumans_to_recon.py:389 look_at_c2w` 가 frame0 을 짓는 식이다.
    #    npz 의 `aim` 을 그대로 쓰면 안 된다 (§rebuild_aim). 온전한 클립 a01~a05 에서 이
    #    재구성이 `work[0]` 을 3.3e-16 이내로 재현하는 걸 확인했다.
    manifest = path.join(args.work, recording, f"manifest_a{action:02d}.json")
    probe = path.join(args.work, recording, f"probe_a{action:02d}.json")
    if not path.exists(manifest) or not path.exists(probe):
        return {"video": video, "status": "DRIFT(manifest없음)", "delta": delta}
    with open(manifest, encoding="utf-8") as file:
        position = json.load(file)["source_camera"]["candidate"]["position"]
    aim = rebuild_aim(probe, args.aim_base, args.aim_smooth_window, args.aim_bias)
    anchor = look_at_c2w(position, aim[0])

    backup = poses + args.backup_suffix
    if not path.exists(backup):
        #    이미 백업이 있으면 덮지 않는다 — 두 번 돌려도 원본 백업이 살아남게.
        shutil.copy2(poses, backup)
    arrays = {key: current[key] for key in current.files}
    arrays["cam_c2w"] = anchor @ good
    #    `aim` 도 같이 덮였으니 되돌린다. 감사는 안 읽지만 앵커 재구성의 입력이라 남겨두면
    #    나중에 이 파일만 보고 판단할 때 헷갈린다.
    arrays["aim"] = aim
    np.savez(poses, **arrays)
    check = np.load(poses)["cam_c2w"]
    after = float(np.abs(np.linalg.inv(check[0]) @ check - good).max())
    return {"video": video, "status": "repaired", "delta": delta, "after": after}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video")                       # 한 편만. 예: tru_00add26c_a00
    parser.add_argument("--bank_manifest")               # 또는 뱅크 매니페스트 전량
    parser.add_argument("--work", default=path.join(path.dirname(path.dirname(HERE)), "out", "trumans_recon"))
    parser.add_argument("--eval_data", default="/data1/cympyc1785/data/TRUMANS-Lite")
    # 부동소수 왕복 오차만 허용한다. 후보가 달라지면 미터 단위로 벌어지므로 구분이 확실하다.
    parser.add_argument("--tolerance", default=1e-6, type=float)
    parser.add_argument("--backup_suffix", default=".overwritten")
    #    기존 뱅크 96편이 쓴 aim 규약. `trumans_to_recon.py` 의 **현재** 기본값
    #    (obb_center / bias 0.20) 과 다르다 — 뱅크를 돌린 뒤 기본이 바뀌었기 때문이다.
    #    이 세 값은 온전한 클립 a01~a05 의 `aim` 을 0.0000 으로 재현해 역산한 것이다.
    parser.add_argument("--aim_base", default="chest", choices=("chest", "obb_center"))
    parser.add_argument("--aim_smooth_window", default=11, type=int)
    parser.add_argument("--aim_bias", default=0.0, type=float)
    # 기본은 읽기 전용이다 — 되돌리기는 명시적으로 요구해야 한다.
    parser.add_argument("--apply", action="store_true")
    #    `delta` 는 **상대** pose 만 보므로 앵커가 틀려도 `ok` 로 나온다. 앵커를 잘못 세운
    #    복구를 다시 덮어쓸 때 필요하다 (백업은 이미 있으면 안 건드리므로 원본은 안전).
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.bank_manifest:
        targets = entries_from_manifest(args.bank_manifest)
    elif args.video:
        targets = [entry_from_video(args.video, args.work)]
    else:
        raise SystemExit("--video 나 --bank_manifest 중 하나는 줘야 한다")

    rows = [check_one(v, r, a, args) for v, r, a in targets]

    print(f"{'video':30s} {'status':14s} {'max|poses-배포본|':>18s}")
    print("-" * 66)
    for row in rows:
        extra = f"  -> {row['after']:.2e}" if "after" in row else ""
        print(f"{row['video']:30s} {row['status']:14s} {row['delta']:18.6f}{extra}")
    print("-" * 66)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if not args.apply and counts.get("DRIFT"):
        print("\n되돌리려면 --apply 를 붙여라. 손상본은 "
              f"`poses_aNN.npz{args.backup_suffix}` 로 남는다.")


if __name__ == "__main__":
    main()
