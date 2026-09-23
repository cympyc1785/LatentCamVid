"""graph 가 도는 동안 **끝난 편부터 주워서** 강등 → 뱅크(route→tau→fit→emit)까지 굽는다.

WHY: graph 단계는 순수 CPU 다 (`build_scene_graph.py` 는 recon npz 를 numpy 로 올리고
`cv2.minAreaRect` 로 OBB 를 맞춘다 — `nvidia-smi --query-compute-apps` 가 빈 목록이다).
그래서 graph 만 돌리면 GPU 4장이 전부 논다. 반면 뱅크의 tau/fit 은 `--device cuda` 로
점군을 올려 렌더한다. 두 단계를 **겹쳐서** 돌리면 CPU 는 graph 가, GPU 는 뱅크가 쓴다
(2026-09-12 사용자 지시 "gpu 안놀게 bank랑 묶어서 해줘").

라운드 루프: 매 라운드에 `<graph_marker>` 가 새로 생긴 편을 스캔해서 그만큼 굽고, 다 굽고
나면 다시 스캔한다. graph 드라이버가 죽고 ready 도 0 이면 끝낸다.

순서가 graph → 강등 → route 인 이유. `dynpose_dynamic_mask_from_seg.py --demote_static_objects`
는 `scene_graph.json` 의 `center_drift_u`/`path_len_u`/`d_ref` 를 읽으므로 graph 뒤여야 하고,
tau/fit 이 `--cloud_source memory` 로 그 자리에서 읽는 `dynamic_mask/*.png` 를 덮으므로 route
앞이어야 한다. run_bank 의 단계가 아니라서 이 드라이버가 부른다.

**강등 재실행 판정은 mtime 이다.** `from_seg.json` 의 `demote_static_objects` 필드만 보면
"예전 graph 로 강등된" 편을 건너뛴다 — D182 처럼 graph 를 다시 구우면 그 판정이 낡는다.
그래서 `from_seg.json` 이 `<graph_marker>` 보다 오래됐으면 다시 강등한다. 덮어쓰기 자체는
`masks.npz` 에서 매번 합집합을 새로 만들므로(`build_union`) 멱등이다.

D179 의 `tmp/d179/chain_bank_10k.py` 를 세대 상수만 인자로 뽑아 승격한 것이다 — 세대마다
복붙하는 대신 `--config`/`--graph_marker`/`--bank_dir` 로 표현한다.

env: vista4d (뱅크가 torch/cuda 를 쓴다)

예시:
    # D183 — D182 graph 를 따라가며 굽는다
    python exec/chain_bank_rounds.py --config configs/bank/d183_dynpose100k_bank.json \
        --videos /data1/.../tmp/d172/videos_10346.txt --work /data1/.../tmp/d183 \
        --graph_marker .graph_d182 --bank_dir hole_bank_d183 \
        --graph_pattern "run_bank.py --config configs/bank/d182_"
    # 강등만
    python exec/chain_bank_rounds.py ... --stage demote
    # D185 — D184 와 **동시에**. 선행(d184) 뱅크가 끝낸 편만 집고 강등은 건너뛴다
    python exec/chain_bank_rounds.py --config configs/bank/d185_dynpose100k_grid5.json \
        --videos /data1/.../tmp/d172/videos_10346.txt --work /data1/.../tmp/d185 \
        --graph_marker .graph_d182 --bank_dir hole_bank_d185 \
        --require_bank_dir hole_bank_d184 --stage bank \
        --graph_pattern "run_bank.py --config configs/bank/d184_"
"""
import json
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime
from os import environ, listdir, makedirs, path
from time import sleep, time

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
PY = sys.executable
OUT_ROOT = "out_dynpose"                      # ROOT 기준 상대 (config 의 output_root 와 같아야)
GPUS = "0,1,2,3"                              # `--gpus` 기본값. 세대끼리 카드를 갈라 쓸 때만 바꾼다


def log(msg):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def videos(listing):
    with open(listing, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def graph_alive(pattern):
    """graph 드라이버가 아직 도는가. 돌면 다음 라운드에 새 ready 가 생긴다.

    `pgrep -f` 대신 `ps` 를 파싱한다 — `pgrep -f <pattern>` 은 자기 자신의 명령줄에 그
    패턴이 들어 있어 스스로를 잡는다(자가 매칭). 그 함정으로 죽은 작업을 살아 있다고
    읽으면 라운드 루프가 영원히 안 끝난다.
    """
    out = subprocess.run(["ps", "-eo", "args", "--no-headers"],
                         capture_output=True, text=True).stdout
    return any(pattern in ln and "chain_bank_rounds" not in ln for ln in out.splitlines())


def alive_procs(pattern):
    """`pattern` 을 명령줄에 가진 프로세스 수. `--wait_for` 전용 (D188).

    `graph_alive` 와 같은 자가 매칭 함정을 피한다 — 여기는 **개수**를 돌려주므로
    "몇 개 남았나"를 로그에 쓸 수 있다. `bash -c` 래퍼도 뺀다 (nohup 기동 한 줄이 잡힌다).
    """
    out = subprocess.run(["ps", "-eo", "args", "--no-headers"],
                         capture_output=True, text=True).stdout
    return sum(1 for ln in out.splitlines()
               if pattern in ln and "chain_bank_rounds" not in ln and "bin/bash -c" not in ln)


def done_bank(root, bank_dir):
    """이 세대 뱅크가 그 편을 끝냈는가 (canonical 이 나왔거나 skipped 로 접었거나)."""
    bank = path.join(root, bank_dir)
    return (path.exists(path.join(bank, "canonical", "canonical.json"))
            or path.exists(path.join(bank, "skipped.json")))


def ready_videos(vids, marker, bank_dir, require_bank_dir=""):
    """graph 는 끝났고 이 세대 뱅크는 아직 없는 편.

    `require_bank_dir` (D185): **다른 세대 뱅크가 이미 끝낸 편만** ready 로 본다. 두 세대를
    동시에 돌릴 때 강등(`dynamic_mask`)이 겹치는 것을 막는 용도다 — 강등은 두 체인이 공유하는
    파일을 다시 쓰므로, 같은 편을 동시에 집으면 한쪽이 쓰는 중인 마스크를 다른 쪽이 읽는다.
    선행 체인은 라운드마다 `강등 -> 뱅크` 순서라, 그 뱅크가 끝난 편은 **강등이 이미 끝나 있다**.
    그래서 뒤따르는 체인은 `--stage bank` 로 강등을 건너뛰어도 같은 마스크 위에서 굽는다.
    기본값 "" 이면 옛 동작 그대로 (선행 조건 없음).
    """
    out = []
    for v in vids:
        root = path.join(ROOT, OUT_ROOT, v)
        if not path.exists(path.join(root, marker)):
            continue
        if require_bank_dir and not done_bank(root, require_bank_dir):
            continue
        if done_bank(root, bank_dir):
            continue
        out.append(v)
    return out


def needs_demote(vids, eval_data, marker):
    """`from_seg.json` 이 없거나 · 강등 없이 구워졌거나 · **graph 보다 오래된** 편."""
    out = []
    for v in vids:
        meta = path.join(eval_data, "eval_data", "recon_and_seg", v,
                         "dynamic_mask", "from_seg.json")
        graph_marker = path.join(ROOT, OUT_ROOT, v, marker)
        if not path.exists(meta):
            out.append(v)
            continue
        try:
            with open(meta, encoding="utf-8") as fh:
                demoted = bool(json.load(fh).get("demote_static_objects"))
        except (ValueError, OSError):
            out.append(v)
            continue
        stale = (path.exists(graph_marker)
                 and path.getmtime(meta) < path.getmtime(graph_marker))
        if not demoted or stale:
            out.append(v)
    return out


def fan_out(work, name, argv_of_shard, num_shards, log_name, gpus=None):
    """샤드 num_shards 개를 동시에 띄우고 전부 끝날 때까지 기다린다. -> 실패 샤드 수.

    `gpus` 는 카드 번호 리스트(예 `["5","6","7"]`) 이거나 None(= CPU 전용 단계). 예전에는
    모듈 상수 `GPUS` 를 바로 읽었는데, 그러면 **두 세대를 동시에 돌릴 때 카드를 못 가른다** —
    상수를 고치면 이미 도는 다른 체인까지 같은 카드로 끌려온다 (2026-09-13 d185/d188).
    """
    makedirs(path.join(work, log_name), exist_ok=True)
    procs = []
    for shard in range(num_shards):
        env = dict(environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[shard % len(gpus)]
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        fh = open(path.join(work, log_name, f"shard{shard:02d}.log"), "w")
        procs.append((shard, fh, subprocess.Popen(argv_of_shard(shard), cwd=ROOT,
                                                  stdout=fh, stderr=fh, env=env)))
    log(f"{name} — {num_shards} 샤드 기동, 로그 {work}/{log_name}/")
    t0, bad = time(), 0
    for shard, fh, proc in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            bad += 1
            log(f"  {name} shard{shard:02d} rc={rc}  <-- 실패")
    log(f"{name} 종료 — 실패 {bad}/{num_shards}, {(time() - t0) / 60:.1f}분")
    return bad


def run_demote(args, vids, round_tag):
    """정지 소품 강등. GPU 를 안 쓰므로(numpy+PIL) 샤드는 CPU 프로세스로 나눈다."""
    todo = needs_demote(vids, args.eval_data, args.graph_marker)
    log(f"강등 — 대상 {len(todo)}/{len(vids)}편 (나머지는 from_seg.json 이 이 graph 이후 것)")
    if not todo:
        return 0
    lists = [todo[shard::args.demote_shards] for shard in range(args.demote_shards)]
    lists = [part for part in lists if part]

    def argv(shard):
        return [PY, "-u", "fit/ingest/dynpose_dynamic_mask_from_seg.py",
                "--eval_data", args.eval_data, "--output_root", OUT_ROOT,
                "--demote_static_objects", "--videos"] + lists[shard]

    return fan_out(args.work, f"강등({round_tag})", argv, len(lists),
                   f"demote_logs_{round_tag}", gpus=None)


def run_bank(args, vids, round_tag):
    listing = path.join(args.work, f"ready_{round_tag}.txt")
    with open(listing, "w", encoding="utf-8") as fh:
        fh.write("\n".join(vids))
    makedirs(path.join(args.work, "bank_scene_logs"), exist_ok=True)

    def argv(shard):
        return [PY, "-u", "exec/run_bank.py",
                "--config", args.config, "--videos", listing,
                "--log_dir", path.join(args.work, "bank_scene_logs"),
                "--stages", "route,tau,fit,emit",
                "--exec", args.bank_exec,
                "--num_shards", str(args.bank_shards), "--shard_id", str(shard)]

    bad = fan_out(args.work, f"뱅크({round_tag})", argv, args.bank_shards,
                  f"bank_logs_{round_tag}", gpus=[g for g in args.gpus.split(",") if g])
    log(f"  뱅크({round_tag}) 편별 — {tally(args.work, f'bank_logs_{round_tag}')}")
    return bad


def tally(work, log_name):
    """샤드 로그의 `=== 요약 ===` 표를 세서 편별 상태 분포를 낸다.

    `fan_out` 이 세는 건 **샤드 프로세스의 rc** 라, 샤드가 정상 종료하면 그 안에서 몇 편이
    FAIL 했는지 안 보인다 (d184 스모크에서 3/4 편이 route 로 죽었는데 "실패 0/4" 로 찍혔다).
    """
    # 상태 문자열에 공백이 있다("SKIP(앵커 0)"). 칸 구분은 **2칸 이상 공백**이다.
    row = re.compile(r"^ {2}(\S+) {2,}(.+?) {2,}[\d.]+s$")
    counts = {}
    for name in sorted(listdir(path.join(work, log_name))):
        with open(path.join(work, log_name, name), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                hit = row.match(line.rstrip())
                if hit:
                    counts[hit.group(2)] = counts.get(hit.group(2), 0) + 1
    return "  ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "없음"


def baked(vids, bank_dir):
    return sum(path.exists(path.join(ROOT, OUT_ROOT, v, bank_dir, "canonical", "canonical.json"))
               or path.exists(path.join(ROOT, OUT_ROOT, v, bank_dir, "skipped.json"))
               for v in vids)


def more_coming(args, vids):
    """다음 라운드에 새 ready 가 생길 여지가 있는가.

    graph 드라이버가 살아 있거나, `--require_bank_dir` 선행 세대가 아직 전량을 안 끝냈으면
    기다린다. 후자는 `ps` 가 아니라 **디스크 상태**로 보므로 선행 체인의 강등 구간(샤드가
    잠깐 0개)에 루프가 조기 종료되지 않는다.
    """
    if graph_alive(args.graph_pattern):
        return True
    return bool(args.require_bank_dir) and baked(vids, args.require_bank_dir) < len(vids)


def take_lock(work):
    """`--work` 하나에 드라이버 하나. 이미 살아 있으면 False.

    WHY: 같은 `--work`/`--bank_dir` 로 두 개가 돌면 같은 편을 동시에 집어 **한 씬의 뱅크
    디렉토리에 두 프로세스가 쓴다**. 2026-09-12 에 screen 창 입력 버퍼에 기동 명령이 큐잉된
    채 남아(앞 작업이 foreground 라 실행되지 않고 대기) 나중에 자동 실행될 뻔했다 — 그런
    사고를 프로세스 쪽에서 막는다. 죽은 프로세스의 lock 은 그냥 뺏는다 (PID 생존으로 판정).
    """
    lock = path.join(work, "chain.lock")
    if path.exists(lock):
        try:
            with open(lock, encoding="utf-8") as fh:
                old = int(fh.read().split()[0])
        except (ValueError, IndexError, OSError):
            old = -1
        if old > 0 and path.exists(f"/proc/{old}"):
            log(f"이 --work 에 드라이버가 이미 돈다 (pid {old}, {lock}) — 기동 안 함")
            return False
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()} {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return True


def main(args):
    makedirs(args.work, exist_ok=True)
    if not take_lock(args.work):
        return 0
    # D188. 앞선 GPU 작업이 끝나기를 기다렸다 시작한다 (`rebake_scenes.py --wait_for` 와 같은 것).
    # GPU 4장에 뱅크 프로세스가 2개씩 올라가면 실측 55.9 GiB 라 3번째는 80 GiB 카드에 안 들어간다
    # (2026-09-13 샘플 6회: 피크 55,947 MiB / proc 당 ~28 GiB). 그래서 "빈 GPU 가 보이면 띄운다"가
    # 아니라 **앞 작업이 끝나야** 띄운다 — 안 그러면 남의 학습/굽기까지 같이 OOM 으로 죽는다
    # (2026-09-13 실제로 d185 3편이 그렇게 죽었다).
    if args.wait_for:
        n = alive_procs(args.wait_for)
        if n:
            log(f"선행 대기 — '{args.wait_for}' {n}개 살아 있음")
            while alive_procs(args.wait_for):
                sleep(args.poll)
            log("선행 종료 — 진행")
    vids = videos(args.videos)
    #    **`--max_rounds` 는 "구운 라운드" 수다 — 기다림은 안 센다.** 예전에는 `for rnd in
    #    range(max_rounds)` 라 ready 0 으로 쉬는 것도 한 라운드를 먹었고, 선행 graph 를 따라잡은
    #    뒤로는 ready 6~8 짜리 1분 라운드가 줄줄이 돌아 D184 가 8093/10346 에서 `--max_rounds 60
    #    소진` 으로 조용히 끝났다 (2026-09-13). 캡은 폭주 방지용이지 대기 예산이 아니다.
    rnd = 0
    while rnd < args.max_rounds:
        ready = ready_videos(vids, args.graph_marker, args.bank_dir, args.require_bank_dir)
        alive = more_coming(args, vids)
        #    선행이 살아 있는데 ready 가 `--min_ready` 에 못 미치면 굽지 않고 쉰다. 라운드마다
        #    드는 고정비(강등 샤드 기동 + 뱅크 샤드 기동)가 편당 비용을 압도하는 구간을 피한다.
        #    기본 0 = **기존 동작 그대로** (ready 1편이어도 바로 굽는다).
        thin = bool(args.min_ready) and alive and 0 < len(ready) < args.min_ready
        if not ready or thin:
            if not alive:
                log("ready 0 이고 선행(graph/앞 세대)도 끝났다 — 종료")
                return 0
            why = (f"ready {len(ready)} < --min_ready {args.min_ready}" if thin else "ready 0")
            log(f"{why} — {args.poll / 60:.0f}분 뒤 재스캔 (구운 라운드 {rnd}/{args.max_rounds})")
            sleep(args.poll)
            continue
        rnd += 1
        tag = f"r{rnd:02d}"
        log(f"=== 라운드 {tag} — ready {len(ready)}/{len(vids)}, "
            f"이미 구움 {baked(vids, args.bank_dir)}, 선행 {'진행 중' if alive else '종료'}")
        if args.round_cap and len(ready) > args.round_cap:
            # 한 라운드가 너무 길면 그동안 끝난 편이 다음 라운드까지 논다. 앞에서 잘라 쓴다.
            log(f"  라운드 상한 {args.round_cap}편으로 자름 (나머지는 다음 라운드)")
            ready = ready[:args.round_cap]
        if args.stage in ("all", "demote"):
            run_demote(args, ready, tag)
        if args.stage == "demote":
            return 0
        run_bank(args, ready, tag)
        log(f"라운드 {tag} 끝 — 누적 구움 {baked(vids, args.bank_dir)}/{len(vids)}")
        if not more_coming(args, vids) and not ready_videos(
                vids, args.graph_marker, args.bank_dir, args.require_bank_dir):
            log("선행 종료 + ready 0 — 전량 완료")
            return 0
    log(f"--max_rounds {args.max_rounds} 소진")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)                # configs/bank/<gen>.json
    parser.add_argument("--videos", required=True)                # 목록 파일
    parser.add_argument("--work", required=True)                  # 로그·목록을 쌓을 tmp 폴더
    parser.add_argument("--eval_data", default="/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K")
    parser.add_argument("--graph_marker", default=".graph_d182")  # 이게 있어야 ready
    parser.add_argument("--bank_dir", default="hole_bank_d183")   # config 의 bank_dir 과 같아야
    # 두 세대를 **동시에** 돌릴 때: 선행 세대 뱅크가 끝낸 편만 집는다 (강등 경합 회피, §ready_videos).
    # 뒤따르는 쪽은 `--stage bank` 로 강등을 건너뛴다. "" 이면 옛 동작.
    parser.add_argument("--require_bank_dir", default="")
    # graph 드라이버가 살아 있는지 볼 `ps` 부분문자열. 죽고 ready 0 이면 루프를 끝낸다.
    parser.add_argument("--graph_pattern", default="run_bank.py --config configs/bank/d182_")
    parser.add_argument("--stage", default="all", choices=("all", "demote", "bank"))
    parser.add_argument("--bank_shards", default=4, type=int)     # GPU 1장당 1~2개
    # D188-c. 샤드를 어느 카드에 올릴지. 세대를 두 개 동시에 돌리면 기본값(0~3)으로는 같은
    # 카드에 8개가 몰려 서로를 느리게 한다 (실측 d185 r09 225.5분 / d188 r01 188.1분).
    # 기본값 = 옛 동작 그대로. **카드 1장에 2개가 실측 상한**(~28 GiB/proc, 피크 55.9/80 GiB)
    # 이므로 `--bank_shards` 를 카드 수의 2배보다 크게 주지 말 것.
    parser.add_argument("--gpus", default=GPUS)                   # 예: "5,6,7"
    # D188. `run_bank.py --exec` 를 체인에서도 고를 수 있게 한다 — 그 인자가 D180 에서 생겼는데
    # 여기서 안 넘겨 줘서 체인으로 도는 세대는 **전부 subprocess 로만** 돌고 있었다.
    # 실측(d188 r01, 샤드 4): 씬당 실효 37.7s = tau 46.5 + fit 60.2 + emit 5.4 + route 0.3 을 4로
    # 나눈 값. 이 중 씬당 import 21.9s + tau/fit 중복 cloud 14.7s 가 프로세스 경계 비용이다
    # (`run_bank._run_inproc` 주석의 실측). 기본값은 `subprocess` = 옛 동작 비트 동일 —
    # in-process 는 한 단계의 전역 오염이 샤드를 물고 가므로 켜는 쪽이 명시적이어야 한다.
    parser.add_argument("--bank_exec", default="subprocess",
                        choices=("subprocess", "inproc"))
    parser.add_argument("--demote_shards", default=8, type=int)   # CPU 전용 (numpy+PIL)
    parser.add_argument("--round_cap", default=0, type=int)       # 0 = 무제한
    # **구운 라운드만** 센다 (기다림은 안 센다, §main). 폭주 방지용 상한이다.
    parser.add_argument("--max_rounds", default=40, type=int)
    # 선행이 살아 있을 때 이 편수 미만이면 굽지 않고 --poll 만큼 쉰다. 0 = 기존 동작.
    parser.add_argument("--min_ready", default=0, type=int)
    parser.add_argument("--poll", default=900, type=int)          # ready 0 일 때 재스캔 간격(초)
    # D188. 이 문자열을 명령줄에 가진 프로세스가 전부 끝날 때까지 기다렸다 시작한다.
    # GPU 당 뱅크 프로세스 2개가 이미 한계라(실측 55.9 GiB / 80 GiB), 세 번째 세대를 겹치면
    # 남의 작업까지 OOM 으로 죽는다. 빈 문자열(기본) = 안 기다린다.
    parser.add_argument("--wait_for", default="")
    sys.exit(main(parser.parse_args()))
