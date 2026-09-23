"""TRUMANS recording 여러 편을 **action 단위로 쪼개** LBM-Lite 입력(recon_and_seg + seg_instances)으로 뽑는다.

왜 배치 드라이버를 따로 두는가: `trumans_to_recon.py` 는 action 하나짜리이고, 그 안에서 실패는
전부 assert 다 (설 수 있는 후보가 없다 / 궤적 검증 전량 실패). 97개 action 을 돌리면 **실패하는
action 이 정상**이다 — 사람이 벽에 딱 붙어 있거나 좁은 화장실에 있으면 물리적으로 카메라를 놓을
자리가 없다. 그걸 배치 전체의 중단으로 만들면 안 되므로, 여기서 action 하나를 subprocess 로
격리하고 실패 사유를 manifest 에 적은 뒤 다음으로 넘어간다.

동시 실행 수를 제한하는 이유: RGB 는 EEVEE(GPU) 라 Blender 를 6편 동시에 띄웠더니
`libnvidia-eglcore` 안에서 crash 했다 (2026-08-23 LBM 4ac2c1b3). 메모리가 아니라 GPU 컨텍스트
경합이다. 기본 2 workers.

출력:
  <eval_data>/eval_data/{recon_and_seg,seg_instances,seg_instances_static}/tru_<uuid8>_a<NN>/
  <out>/bank_manifest.json      (`trumans_lite_bank_v1`) — action 별 성공/실패 + 채택된 카메라

보행 pseudo-action: `Actions/*.txt` 에 보행 라벨이 **0건**이라 라벨만 쓰면 뱅크가 전부 제자리
조작이 된다. `trumans_to_recon.py` 가 목록 **뒤에** 채굴 결과를 붙여주므로 여기서는 비율
(`--walk_ratio`, 기본 0.30)만 정해서 자른다. 라벨 action 의 인덱스는 불변이라 기존 96편은
`--skip_done` 으로 그대로 재현된다.

사용 예시:
  # 1편 전체 action, 상태만 보기
  python scripts/trumans_lite_bank.py --recordings 00add26c-... --dry_run

  # 7편 전량, 2 worker
  python scripts/trumans_lite_bank.py --recordings_file /tmp/recs.txt --workers 2

  # 소품 subject 로 한 벌 더 (이름 충돌 방지)
  python scripts/trumans_lite_bank.py --recordings_file /tmp/recs.txt \\
      --subject_kind auto --video_suffix _ev
"""
import json
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from os import makedirs, path

HERE = path.dirname(path.abspath(__file__))
# R6(2026-09-22) 재분류에서 드라이버(`scripts/`)와 피구동 스크립트(`fit/ingest/`)가 **갈라졌다.**
# `path.join(HERE, ...)` 로 옆을 보면 안 나온다 — 형제 폴더를 명시한다 (R8 에서 잡음).
INGEST = path.join(path.dirname(HERE), "fit", "ingest")
TRUMANS_DEFAULT = "/data1/cympyc1785/data/trumans/Data_release"
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/TRUMANS-Lite"
#    `trumans_to_recon.py --min_subject_dist` 의 기본값과 **같은 값이어야 한다**. 여기서
#    기본값 여부를 보고 인자를 넘길지 정하므로, 어긋나면 기존 뱅크와 명령이 달라진다.
MIN_SUBJECT_DIST_DEFAULT = 0.80
PYTHON = sys.executable


def walk_flags(args):
    """보행 채굴 인자. `--list_actions` 와 실제 실행에 **같은 값**이 가야 한다.

    다르면 목록의 idx 와 실행의 `--action <i>` 가 가리키는 action 이 어긋난다 — 보행은
    목록 뒤에 append 되므로 채굴 개수가 하나만 달라도 뒤쪽 인덱스가 통째로 밀린다.
    `--frame_step` 이 여기 끼는 이유가 그것이다: 채굴 창 폭이 (49-1)*step+1 이라 step 을 바꾸면
    채굴 개수가 바뀐다. 목록에만 안 넘기면 인덱스가 통째로 어긋난 채 조용히 돈다.
    """
    if not args.walk_actions:
        return ["--no_walk_actions", "--frame_step", str(args.frame_step)]
    return ["--walk_max", str(args.walk_max), "--walk_speed", str(args.walk_speed),
            "--motion_fps", str(args.motion_fps), "--frame_step", str(args.frame_step)]


def clip_flags(args):
    """창 길이 · preset 크기 관련 passthrough. 기본값이면 `trumans_to_recon.py` 의 기본과
    같으므로 기존 뱅크와 명령이 의미상 동일하다 (`--frame_step 1 --preset_scale 1.0`)."""
    return ["--preset_scale", str(args.preset_scale),
            "--min_end_radius", str(args.min_end_radius),
            "--smooth_window", str(args.smooth_window),
            "--lens", str(args.lens),
            "--aim_keyframes", str(args.aim_keyframes),
            "--out_fps", str(args.out_fps)]


def cell_key(candidate):
    """격자 칸 식별자. probe JSON 의 az/elev/radius 는 float 라 반올림해서 키로 쓴다."""
    return (round(float(candidate["azimuth_deg"]), 3),
            round(float(candidate["elevation_deg"]), 3),
            round(float(candidate["radius"]), 3))


def probe_usable(args, recording, action):
    """`--probe_only` 로 격자만 돌리고 usable 칸 집합을 돌려준다 (렌더 0회, action 당 ~8초).

    공유 anchor 를 정하려면 recording 의 **모든** action 격자를 먼저 봐야 한다. 그 pass 에서
    궤적 합성 · 검증 · 렌더는 전부 낭비라 여기서 잘라낸다.
    """
    work_rec = path.join(args.work, recording + args.video_suffix)
    command = [PYTHON, path.join(INGEST, "trumans_to_recon.py"),
               "--recording", recording, "--action", str(action["index"]),
               "--trumans", args.trumans, "--eval_data", args.eval_data,
               "--subject_kind", args.subject_kind, "--anchor_origin", args.anchor_origin,
               "--res", str(args.res[0]), str(args.res[1]),
               "--work", work_rec, "--probe_only",
               *walk_flags(args), *clip_flags(args)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = [l for l in proc.stderr.strip().splitlines() if l.strip()]
        reason = next((l for l in reversed(tail) if "Error" in l or "없다" in l), "")
        print(f"  probe FAIL {recording[:8]}_a{action['index']:02d}  {reason.strip()[:80]}")
        return None
    probe_path = path.join(work_rec, f"probe_a{action['index']:02d}.json")
    with open(probe_path, encoding="utf-8") as file:
        probe = json.load(file)
    return {cell_key(c) for c in probe["candidates"]
            if c["clear"] and c["clearance"] >= args.min_clearance
            and c["floor_drop"] >= args.min_floor_drop}


def shared_cell(args, recording, actions):
    """recording 안 **모든** 클립이 쓸 수 있는 격자 칸 하나를 고른다.

    "정확히 하나로 통일" (사용자 지시) 이므로 여러 칸으로 덮지 않는다 — 가장 많은 action 을
    덮는 칸을 고르고, 그 칸을 못 쓰는 action 은 **버린다**. 조용히 다른 칸으로 넘기면 공유가
    깨진 채로 코퍼스에 섞이고, 그건 로그만 보고는 구분이 안 된다.
    동점이면 (반경 큰 것, 고도 낮은 것, 방위각 작은 것) 순 — 반경이 크면 preset 이동 여유가
    크고(`--min_end_radius`), 고도가 낮으면 사람 눈높이에 가깝다.
    """
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        sets = list(pool.map(lambda a: probe_usable(args, recording, a), actions))
    votes = {}
    for action, cells in zip(actions, sets):
        for cell in (cells or set()):
            votes.setdefault(cell, []).append(action["index"])
    assert votes, f"{recording[:8]} 어느 action 도 usable 격자가 없다"
    best = max(votes, key=lambda c: (len(votes[c]), c[2], -c[1], -c[0]))
    covered = set(votes[best])
    kept = [a for a in actions if a["index"] in covered]
    dropped = [a["index"] for a in actions if a["index"] not in covered]
    print(f"{recording[:8]}  공유 칸 az {best[0]:.0f} elev {best[1]:.0f} r {best[2]:.1f}  "
          f"덮음 {len(kept)}/{len(actions)}"
          + (f"  버림 {dropped}" if dropped else ""))
    return best, kept, dropped


def list_actions(python, recording, args):
    """`--list_actions` 7열 출력을 파싱해 목록으로.

    포맷은 `{idx} {kind} {start} {end} {len} {prop}  {text}` 다 (`kind` 는 labeled|walk,
    `prop` 는 obj_list 매칭 결과 또는 `-`). 예전 5열 파서는 `parts[1]` 을 start 로 읽어
    `int("labeled")` 에서 죽는다.
    """
    proc = subprocess.run(
        [python, path.join(INGEST, "trumans_to_recon.py"), "--recording", recording,
         "--trumans", args.trumans, "--list_actions", *walk_flags(args)],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"{recording} action 목록 실패:\n{proc.stderr[-2000:]}"
    actions = []
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=6)
        if len(parts) >= 7 and parts[0].isdigit():
            actions.append({"index": int(parts[0]), "kind": parts[1],
                            "start": int(parts[2]), "end": int(parts[3]),
                            "prop": None if parts[5] == "-" else parts[5],
                            "text": parts[6].strip()})
    assert actions, f"{recording} 에서 action 을 하나도 못 읽었다 (출력 포맷이 바뀌었나)"
    return actions


def select_jobs(actions, args):
    """라벨 action + 보행 pseudo-action 을 목표 비율로 섞는다.

    왜 비율로 자르는가: 채굴량은 편마다 4~8 개인데 라벨은 10~17 개다. 그냥 다 넣으면
    편에 따라 보행이 22~44 % 로 들쭉날쭉해진다. `--walk_ratio` 는 **뱅크 안 보행 비율**이라
    `n_walk = round(n_labeled · r/(1−r))` 로 뒤집어 푼다 (채굴량이 모자라면 그만큼만).
    """
    labeled = [a for a in actions if a["kind"] != "walk"]
    walk = [a for a in actions if a["kind"] == "walk"]
    if args.max_actions > 0:
        labeled = labeled[:args.max_actions]
    #    너무 짧은 action 은 49프레임을 채우려고 앞뒤로 늘리면 사실상 옆 action 이 된다.
    #    보행 창은 정의상 정확히 49프레임이라 이 필터에 걸릴 일이 없다.
    labeled = [a for a in labeled if a["end"] - a["start"] + 1 >= args.min_action_frames]
    ratio = min(max(args.walk_ratio, 0.0), 0.95)
    want = int(round(len(labeled) * ratio / (1.0 - ratio))) if ratio > 0 else 0
    #    `mine_walk_actions` 가 이미 순 이동량 내림차순으로 주므로 앞에서 자르면 많이 걷는 창부터다.
    return labeled + walk[:min(want, len(walk))], len(labeled), len(walk), want


def run_one(args, recording, action, cell=None):
    """action 하나. 실패는 예외로 안 올리고 dict 로 돌려준다 — 배치가 멈추면 안 된다."""
    video = f"tru_{recording[:8]}_a{action['index']:02d}{args.video_suffix}"
    #    suffix 를 주면 중간 산출물도 갈라놔야 한다. `trumans_to_recon.py` 의 work 파일은
    #    `probe_a03.json` 처럼 **action 인덱스로만** 키가 잡혀서, subject 만 바꿔 다시 돌리면
    #    같은 인덱스의 예전 probe/verify 를 덮어쓴다 (기존 96편의 재현 근거가 사라진다).
    work_rec = path.join(args.work, recording + args.video_suffix)
    done = path.join(args.eval_data, "eval_data", "recon_and_seg", video, "cameras.npz")
    if args.skip_done and path.exists(done):
        return {"video": video, "recording": recording, "action": action["index"],
                "kind": action["kind"], "text": action["text"], "status": "skip_done"}

    command = [PYTHON, path.join(INGEST, "trumans_to_recon.py"),
               "--recording", recording, "--action", str(action["index"]),
               "--seed", str(args.seed), "--trumans", args.trumans,
               "--eval_data", args.eval_data, "--max_tries", str(args.max_tries),
               "--res", str(args.res[0]), str(args.res[1]), "--samples", str(args.samples),
               "--subject_kind", args.subject_kind, "--anchor_origin", args.anchor_origin,
               *walk_flags(args), *clip_flags(args)]
    if args.hold_fallback:
        #    끄면 이 인자를 아예 안 넘긴다 — 기존 뱅크와 명령이 비트 동일해야 한다.
        command += ["--hold_fallback"]
    if args.rgb_engine != "eevee":
        #    기본 엔진이면 인자를 안 넘긴다 — 기존 뱅크와 명령이 비트 동일해야 한다.
        command += ["--rgb_engine", args.rgb_engine, "--rgb_samples", str(args.rgb_samples)]
    if args.blender_retries:
        command += ["--blender_retries", str(args.blender_retries)]
    if args.min_subject_dist != MIN_SUBJECT_DIST_DEFAULT:
        #    기본값이면 인자를 안 넘긴다 — 기존 뱅크와 명령이 비트 동일해야 한다.
        command += ["--min_subject_dist", str(args.min_subject_dist)]
    if cell is not None:
        command += ["--anchor_cell", str(cell[0]), str(cell[1]), str(cell[2])]
    if args.video_suffix:
        #    suffix 가 없으면 이 두 인자를 아예 안 넘긴다 — 기존 96편과 명령이 비트 동일해야 한다.
        command += ["--out_video", video, "--work", work_rec]
    if args.dry_run:
        return {"video": video, "recording": recording, "action": action["index"],
                "kind": action["kind"], "text": action["text"], "status": "dry_run",
                "command": " ".join(command)}

    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - started
    row = {"video": video, "recording": recording, "action": action["index"],
           "kind": action["kind"], "prop": action["prop"],
           "text": action["text"], "frames": [action["start"], action["end"]],
           "seconds": round(elapsed, 1)}
    if proc.returncode != 0:
        # assert 메시지 마지막 줄이 곧 실패 사유다 (후보 없음 / 궤적 검증 실패 / 그 외).
        tail = [l for l in proc.stderr.strip().splitlines() if l.strip()]
        reason = next((l for l in reversed(tail) if "Error" in l or "실패" in l or "없다" in l), "")
        row.update({"status": "fail", "reason": reason.strip()[:300],
                    "stderr_tail": "\n".join(tail[-6:])})
        print(f"  FAIL {video:24s} {elapsed:6.1f}s  {reason.strip()[:96]}")
        return row

    manifest_path = path.join(work_rec, f"manifest_a{action['index']:02d}.json")
    detail = {}
    if path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as file:
            detail = json.load(file)
    camera = detail.get("source_camera", {})
    row.update({"status": "ok", "preset": camera.get("preset", ""),
                "candidate": camera.get("candidate", {}),
                "verify": detail.get("verify", {}), "report": detail.get("report", {}),
                "timing_seconds": detail.get("timing_seconds", {})})
    print(f"  OK   {video:24s} {elapsed:6.1f}s  {camera.get('preset', ''):10s} "
          f"az {camera.get('candidate', {}).get('azimuth_deg', float('nan')):5.0f}")
    return row


def main(args):
    recordings = list(args.recordings)
    if args.recordings_file:
        with open(args.recordings_file, encoding="utf-8") as file:
            recordings += [l.strip() for l in file if l.strip() and not l.startswith("#")]
    assert recordings, "--recordings 또는 --recordings_file 이 필요하다"

    jobs, mix, skipped = [], [], []
    for recording in recordings:
        #    열거는 recording 당 Blender 를 여러 번 띄우므로 63편이면 ~1시간이다. 예전에는 여기서
        #    assert 하나가 터지면 그때까지의 열거를 통째로 버리고 뱅크가 **0편**으로 끝났다 —
        #    실제로 `a2e8ba09` 가 `pick_sequence` 에서 죽어 앞선 35편의 열거가 날아갔다.
        #    한 편의 데이터 문제로 나머지를 못 만들 이유가 없으므로 **건너뛰고 계속**하고,
        #    무엇을 왜 뺐는지는 stdout 과 manifest 의 `skipped_recordings` 양쪽에 남긴다.
        #    조용히 줄어들면 안 되는 값이라 마지막 요약에도 건수를 찍는다.
        #    `--no_skip_bad_recordings` 로 예전처럼 즉시 죽는 동작을 되살릴 수 있다.
        try:
            actions = list_actions(PYTHON, recording, args)
        except AssertionError as error:
            if not args.skip_bad_recordings:
                raise
            reason = str(error).strip().splitlines()[-1][:160]
            print(f"{recording[:8]}  SKIP  열거 실패: {reason}")
            skipped.append({"recording": recording, "stage": "list_actions",
                            "reason": str(error)[-2000:]})
            continue
        chosen, n_lab, n_mined, want = select_jobs(actions, args)
        n_walk = len(chosen) - n_lab
        short = "" if n_walk >= want else f"  (채굴 {n_mined} 개뿐이라 {want} 를 못 채움)"
        print(f"{recording[:8]}  labeled {n_lab}  walk {n_walk}/{n_mined}  "
              f"= {n_walk / max(len(chosen), 1):.0%} walk{short}")
        mix.append({"recording": recording, "labeled": n_lab, "walk_mined": n_mined,
                    "walk_wanted": want, "walk_used": n_walk})
        #    `--anchor_share recording`: 렌더 전에 격자 pass 를 한 번 더 돌아 recording 안
        #    모든 클립이 쓸 수 있는 칸 하나를 정한다. 그 칸을 못 쓰는 action 은 버린다.
        cell = None
        if args.anchor_share == "recording":
            cell, chosen, dropped = shared_cell(args, recording, chosen)
            mix[-1].update({"anchor_cell": list(cell), "dropped_actions": dropped})
            n_walk = sum(1 for a in chosen if a["kind"] == "walk")
            mix[-1]["walk_used"] = n_walk
        jobs += [(recording, a, cell) for a in chosen]

    total_walk = sum(m["walk_used"] for m in mix)
    print(f"\n총 {len(jobs)} action (walk {total_walk} = "
          f"{total_walk / max(len(jobs), 1):.0%}), recording {len(mix)}/{len(recordings)}"
          + (f", 열거 실패로 뺀 편 {len(skipped)}" if skipped else "")
          + f", worker {args.workers}\n" + "-" * 78)
    assert jobs, ("열거를 통과한 recording 이 하나도 없다. "
                  f"skipped {len(skipped)}/{len(recordings)} — 위 SKIP 줄을 볼 것.")
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda j: run_one(args, *j), jobs))
    elapsed = time.time() - started

    makedirs(args.out, exist_ok=True)
    #    `--dry_run` 은 실행 계획만 보는 것이므로 manifest 를 **안 쓴다**. 예전에는 같은 경로에
    #    dry_run 행을 그대로 써서, 계획 확인 한 번에 기존 뱅크의 집계(133행)가 19행짜리
    #    dry_run 으로 날아갔다 (2026-08-24). 클립은 멀쩡한데 기록만 조용히 사라지는 종류다.
    manifest_path = path.join(args.out, "bank_manifest_dry_run.json" if args.dry_run
                              else "bank_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump({"format": "trumans_lite_bank_v1", "eval_data": args.eval_data,
                   "recordings": recordings, "skipped_recordings": skipped, "seed": args.seed,
                   "workers": args.workers, "total_seconds": round(elapsed, 1),
                   "subject_kind": args.subject_kind, "anchor_origin": args.anchor_origin,
                   "video_suffix": args.video_suffix, "anchor_share": args.anchor_share,
                   "max_tries": args.max_tries, "hold_fallback": args.hold_fallback,
                   "blender_retries": args.blender_retries,
                   "min_subject_dist": args.min_subject_dist,
                   "clip": {"frame_step": args.frame_step, "preset_scale": args.preset_scale,
                            "min_end_radius": args.min_end_radius,
                            "smooth_window": args.smooth_window, "out_fps": args.out_fps,
                            "aim_keyframes": args.aim_keyframes},
                   "walk": {"enabled": args.walk_actions, "ratio": args.walk_ratio,
                            "max": args.walk_max, "speed": args.walk_speed,
                            "motion_fps": args.motion_fps, "per_recording": mix},
                   "entries": rows}, file, ensure_ascii=False, indent=1)

    counts, kinds = {}, {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] == "ok":
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    presets = {}
    for row in rows:
        if row["status"] == "ok":
            presets[row["preset"]] = presets.get(row["preset"], 0) + 1
    print("-" * 78)
    print(f"{'jobs':22s} {len(rows)}  " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"{'ok by kind':22s} " + "  ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print(f"{'presets':22s} " + "  ".join(f"{k} {v}" for k, v in sorted(presets.items())))
    print(f"{'wall':22s} {elapsed / 60:.1f} min")
    print(f"{'->':22s} {manifest_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recordings", default=[], nargs="*", type=str)   # recording 폴더명(전체 uuid)
    parser.add_argument("--recordings_file", default="", type=str)        # 한 줄에 하나씩
    parser.add_argument("--max_actions", default=0, type=int)             # 0 = 전부 (라벨 action 기준)
    parser.add_argument("--min_action_frames", default=12, type=int)      # 이보다 짧은 action 은 제외

    # 보행 pseudo-action. Actions/*.txt 에 보행 라벨이 0건이라 이걸 끄면 뱅크가 전부 제자리
    # 조작이 되고 subject 가 사실상 정지 앵커다 (tracking shot 이 학습 신호에 안 들어간다).
    # `--walk_ratio` 는 뱅크 안 보행 **비율**이고 편별 채굴량으로 상한이 걸린다.
    parser.add_argument("--walk_actions", dest="walk_actions", action="store_true", default=True)
    parser.add_argument("--no_walk_actions", dest="walk_actions", action="store_false")
    parser.add_argument("--walk_ratio", default=0.30, type=float)         # 목표 보행 비율
    parser.add_argument("--walk_max", default=8, type=int)                # 편당 채굴 상한
    parser.add_argument("--walk_speed", default=0.4, type=float)          # 보행 판정 수평속도(m/s)
    # 모션 배열 레이트. blend 의 render fps(편마다 15/25)가 **아니다** — 그걸 쓰면 fps 15 편에서
    # 같은 걸음이 0.24 m/s 로 찍혀 7편 중 5편이 채굴 0건이 된다. 배포 mp4 실측이 30 이다.
    parser.add_argument("--motion_fps", default=30.0, type=float)

    # ── 클립 길이 / preset 크기 (전부 `trumans_to_recon.py` 로 그대로 넘어간다)
    # 모션 프레임 간격. 30 Hz 라 3 이면 10 fps 샘플 = 49장이 4.83초. 1 = 기존 1.63초.
    parser.add_argument("--frame_step", default=1, type=int)
    # preset 이동량 배율. preset 값은 "클립 전체 이동량"이라 클립이 3배 길어지면 3배가 필요하다.
    parser.add_argument("--preset_scale", default=1.0, type=float)
    parser.add_argument("--min_end_radius", default=0.80, type=float)   # push_in 끝 반경 하한(m)
    parser.add_argument("--smooth_window", default=0, type=int)         # 0 = round(11/step)
    parser.add_argument("--out_fps", default=0.0, type=float)           # 0 = motion_fps/frame_step
    # 렌즈 초점거리(mm, 센서 가로 36mm). 기본 25 는 기존 뱅크와 동일 — 값을 바꾸면 그대로
    # `trumans_to_recon.py` -> `trumans_gt_render.py` 로 내려간다.
    #    이게 framing 결함의 유일한 실효 손잡이다. 반경 축은 막혀 있다 — 격자 통과율이
    #    1.5 m 50.3% -> 4.0 m 2.3% 로 죽고 (clearance 가 잡는다), 반경을 밀면 occl_keep 이
    #    0.983 -> 0.625 로 같이 죽는다. 렌즈는 clearance 를 안 건드려서 그 대가가 없다.
    #    렌더 없이 잰 crop_keep p50 (130 클립): 25mm 0.777 / 20mm 0.884 / 18mm 0.932 / 16mm 0.976.
    parser.add_argument("--lens", default=25.0, type=float)
    # 조준 keyframe 수 passthrough. 0(기본) = 매 프레임 look-at (기존 뱅크 96편 재현).
    # 6 = Vista4D pseudo-GT 뱅크(`hole_bank_k6`)와 같은 [0,10,19,29,38,48] + smoothstep slerp.
    parser.add_argument("--aim_keyframes", default=0, type=int)

    # ── frame0 공유. `recording` 이면 렌더 전에 격자 pass 를 돌려 recording 당 칸 **하나**를
    # 정하고 전 클립을 거기서 출발시킨다 (사용자 지시 2026-08-24). 그 칸을 못 쓰는 action 은
    # 버린다 — 조용히 다른 칸으로 넘기면 "공유" 가 깨진 채로 코퍼스에 섞인다.
    # `none`(기본) 은 클립마다 다른 시작 pose = 기존 동작.
    parser.add_argument("--anchor_share", default="none", choices=("none", "recording"))
    parser.add_argument("--min_clearance", default=0.20, type=float)     # 공유 칸 판정에만 사용 (D120)
    # 카메라-피사체 최소 거리(m). `trumans_to_recon.py` 로 그대로 전달한다 (기본값이면 안 넘긴다).
    # stride3 에서 이 게이트가 **단독 탈락 사유 1위**였다 (후보 810개 중 120개가 이것만 걸렸다).
    parser.add_argument("--min_subject_dist", default=MIN_SUBJECT_DIST_DEFAULT, type=float)
    parser.add_argument("--min_floor_drop", default=0.30, type=float)    # (실행은 to_recon 기본값)

    parser.add_argument("--workers", default=2, type=int)                 # EEVEE 경합 때문에 낮게
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max_tries", default=6, type=int)
    # `hold`(정지) 을 대체재로만 쓴다 (`trumans_to_recon.py` 의 같은 이름 플래그로 그대로 전달).
    parser.add_argument("--hold_fallback", dest="hold_fallback", action="store_true", default=False)
    parser.add_argument("--no_hold_fallback", dest="hold_fallback", action="store_false")
    # Blender 가 시그널로 죽었을 때(worker 경합 SIGSEGV) 재시도 횟수. 0 이면 예전 동작.
    parser.add_argument("--blender_retries", default=1, type=int)
    parser.add_argument("--res", default=[960, 540], nargs=2, type=int)
    parser.add_argument("--samples", default=16, type=int)
    #    RGB 패스 엔진. `trumans_to_recon.py` -> `trumans_gt_render.py` 로 그대로 내려간다.
    #    blend 가 3.3.6 저작이라 4.5 EEVEE_NEXT 에서는 발광 재질이 타서 옆 물체까지 번진다.
    parser.add_argument("--rgb_engine", default="eevee", choices=["eevee", "cycles"], type=str)
    parser.add_argument("--rgb_samples", default=128, type=int)
    #    subject / 앵커. 기본값은 `trumans_to_recon.py` 와 같게 둔다 — 드라이버가 조용히 다른
    #    값을 넣으면 손으로 돌린 것과 뱅크가 갈라진다.
    parser.add_argument("--subject_kind", default="human",
                        choices=("human", "auto", "event", "object"))
    parser.add_argument("--anchor_origin", default="obb_center", choices=("obb_center", "chest"))
    # 같은 action 을 다른 subject 로 다시 뽑을 때 이름 충돌을 피한다. 비우면(기본) 기존
    # `tru_<uuid8>_a<NN>` 그대로라 96편이 `--skip_done` 으로 보호된다.
    parser.add_argument("--video_suffix", default="", type=str)

    # 열거(`list_actions`)가 실패한 recording 을 건너뛰고 계속할지. 기본 켬 — 63편 열거에 ~1시간이
    # 드는데 한 편의 데이터 문제(예: blend 프레임 수와 맞는 시퀀스가 없음)로 전부 버리는 건
    # 손해가 너무 크다. 뺀 편은 stdout SKIP 줄 + manifest `skipped_recordings` 에 남는다.
    parser.add_argument("--skip_bad_recordings", dest="skip_bad_recordings",
                        action="store_true", default=True)
    parser.add_argument("--no_skip_bad_recordings", dest="skip_bad_recordings",
                        action="store_false")
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--trumans", default=TRUMANS_DEFAULT, type=str)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--work", default=path.join(path.dirname(HERE), "out", "trumans_recon"),
                        type=str)
    parser.add_argument("--out", default=path.join(path.dirname(HERE), "out", "trumans_lite_bank"),
                        type=str)
    main(parser.parse_args())
