"""지정한 씬 목록만 **마커를 무시하고** 다시 굽는다 (graph → 뱅크 세대들).

WHY. 코드 버그를 고친 뒤에는 "이미 구웠다"는 마커가 **적**이 된다. `run_bank.py` 는
`.graph_<gen>` 마커와 `hole_bank_<gen>/canonical/canonical.json` 을 보고 건너뛰므로, 고친 코드로
다시 구우려면 마커를 지우거나 `--no_skip_done` 을 줘야 한다. **마커를 지우지 않는다** — 지우는
순간 "무엇이 언제 구워졌는지"가 사라져서 재굽기가 실패했을 때 되돌아갈 곳이 없다. 대신
`--no_skip_done` 으로 우회하고, 뱅크 산출물은 **지우지 말고 격리(quarantine)로 옮긴다**.

뱅크를 덮어쓰지 않고 옮겨야 하는 이유는 `skipped.json` / `canonical/canonical.json` 이 **서로를
지우지 않기** 때문이다. 예전에 "앵커 0"으로 `skipped.json` 이 남은 씬이 이번엔 앵커를 얻으면
`canonical.json` 이 새로 생기는데 `skipped.json` 은 그대로 남는다 — 반대 방향도 마찬가지다.
그 상태로 `baked()` 를 세면 숫자는 맞지만 **한 씬이 두 가지 결론을 동시에 들고 있게 된다**.

D186 (2026-09-13) 용도: `merge_duplicates` 가 `frames` 만 합집합으로 만들어 bbox/score 가 한 칸씩
밀렸던 버그(63badcf)를 고친 뒤, 영향받은 씬만 graph→뱅크로 되굽는다.

    # ① graph 만 (마커 무시)
    python scripts/rebake_scenes.py --stage graph \
        --videos /data1/cympyc1785/LatentCamVid/tmp/d182/merge_bug_scenes.txt \
        --graph_config configs/bank/d182_dynpose100k_graph.json \
        --work /data1/cympyc1785/LatentCamVid/tmp/d186 --shards 8

    # ② 앞 단계가 끝나길 기다렸다가 graph → d184 → d185 순서로 전부
    python scripts/rebake_scenes.py --stage all \
        --videos <목록> --graph_config configs/bank/d182_dynpose100k_graph.json \
        --bank_configs configs/bank/d184_dynpose100k_single.json,configs/bank/d185_dynpose100k_grid5.json \
        --wait_for "run_bank.py --config configs/bank/d182_" \
        --work /data1/cympyc1785/LatentCamVid/tmp/d186 --shards 8
"""
from argparse import ArgumentParser
from json import load as json_load
from os import environ, makedirs, path, replace
from shutil import move
from subprocess import Popen, run
from sys import executable
from time import sleep, strftime, time

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
PY = executable
GPUS = ["0", "1", "2", "3"]        # 사용자 지시: GPU 4~7 은 안 쓴다 (CLAUDE.md)


def log(work, msg):
    line = f"[{strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path.join(work, "rebake.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def videos_of(spec):
    with open(spec, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def alive(pattern):
    """`pattern` 을 명령줄에 포함한 프로세스 수. 자기 자신(rebake_scenes)은 뺀다."""
    out = run(["ps", "-eo", "args", "--no-headers"], capture_output=True, text=True).stdout
    return sum(1 for ln in out.splitlines()
               if pattern in ln and "rebake_scenes" not in ln and "bin/bash -c" not in ln)


def fan_out(work, name, argv_of_shard, shards, log_name):
    """샤드를 동시에 띄우고 전부 끝날 때까지 기다린다. -> 실패 샤드 수."""
    makedirs(path.join(work, log_name), exist_ok=True)
    procs = []
    for shard in range(shards):
        env = dict(environ, CUDA_VISIBLE_DEVICES=GPUS[shard % len(GPUS)],
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        fh = open(path.join(work, log_name, f"shard{shard:02d}.log"), "w", encoding="utf-8")
        procs.append((shard, fh, Popen(argv_of_shard(shard), cwd=ROOT,
                                       stdout=fh, stderr=fh, env=env)))
    log(work, f"{name} — {shards} 샤드 기동 (GPU {','.join(GPUS)}), 로그 {work}/{log_name}/")
    t0, bad = time(), 0
    for shard, fh, proc in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            bad += 1
            log(work, f"  {name} shard{shard:02d} rc={rc}  <-- 실패")
    log(work, f"{name} 종료 — 실패 {bad}/{shards}, {(time() - t0) / 60:.1f}분")
    return bad


# ------------------------------------------------------------------ 격리

def load_config(cfg_path):
    """`run_bank.load_config` 과 같은 얕은 `extends` 병합. -> dict.

    FIX (2026-09-13). 여기가 원래 `json_load` 하나였는데, 뱅크 config 대부분이
    `"extends": "d179_dynpose100k_bank.json"` 으로 `output_root` 를 **부모에만** 두고 있다.
    graph config(`d182_*`)는 `extends` 가 없어 단계 ①은 통과했고, ③-a 의 `d184_*` 에서
    `KeyError: 'output_root'` 로 죽었다 — `set -e` 라 ③-b 까지 같이 멈췄다.
    """
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json_load(fh)
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    base = load_config(path.join(path.dirname(cfg_path), parent))
    base.update(cfg)
    return base


def quarantine(work, cfg_path, vids):
    """이 세대의 뱅크 산출물을 `<work>/quarantine/<gen>/<video>/` 로 **옮긴다**(안 지운다).

    -> (옮긴 씬 수, 원래 없던 씬 수)
    """
    cfg = load_config(cfg_path)
    out_root, gen = cfg["output_root"], cfg["generation"]
    dirs = [cfg["tau_bank_dir"], cfg["bank_dir"]]
    qroot = path.join(work, "quarantine", gen)
    moved = absent = 0
    for video in vids:
        hit = False
        for sub in dirs:
            src = path.join(ROOT, out_root, video, sub)
            if not path.isdir(src):
                continue
            dst = path.join(qroot, video, sub)
            makedirs(path.dirname(dst), exist_ok=True)
            if path.exists(dst):          # 같은 목록을 두 번 돌린 경우 — 최신 것으로 덮는다
                move(src, dst + ".prev")
                replace(dst + ".prev", dst)
            else:
                move(src, dst)
            hit = True
        moved += hit
        absent += not hit
    log(work, f"격리({gen}) — 옮김 {moved}편 / 원래 없음 {absent}편  -> {qroot}")
    return moved, absent


# ------------------------------------------------------------------ 단계

def stage_graph(args, vids):
    listing = path.join(args.work, "rebake_videos.txt")
    log_dir = path.join(args.work, "graph_scene_logs")
    makedirs(log_dir, exist_ok=True)

    def argv(shard):
        return [PY, "-u", "scripts/run_bank.py", "--config", args.graph_config,
                "--videos", listing, "--log_dir", log_dir, "--stages", "graph",
                "--no_skip_done",                    # 마커를 지우지 않고 우회한다 (§WHY)
                "--num_shards", str(args.shards), "--shard_id", str(shard)]

    log(args.work, f"graph 재굽기 — {len(vids)}편, config {path.basename(args.graph_config)}")
    return fan_out(args.work, "graph", argv, args.shards, "graph_logs")


def stage_bank(args, vids, cfg_path):
    gen = load_config(cfg_path)["generation"]
    listing = path.join(args.work, "rebake_videos.txt")
    log_dir = path.join(args.work, f"bank_scene_logs_{gen}")
    makedirs(log_dir, exist_ok=True)
    if args.quarantine:
        quarantine(args.work, cfg_path, vids)

    def argv(shard):
        return [PY, "-u", "scripts/run_bank.py", "--config", cfg_path,
                "--videos", listing, "--log_dir", log_dir,
                "--stages", "route,tau,fit,emit", "--no_skip_done",
                "--num_shards", str(args.shards), "--shard_id", str(shard)]

    log(args.work, f"뱅크({gen}) 재굽기 — {len(vids)}편")
    return fan_out(args.work, f"뱅크({gen})", argv, args.shards, f"bank_logs_{gen}")


def main(args):
    makedirs(args.work, exist_ok=True)
    vids = videos_of(args.videos)
    with open(path.join(args.work, "rebake_videos.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(vids) + "\n")

    if args.wait_for:
        n = alive(args.wait_for)
        if n:
            log(args.work, f"선행 대기 — '{args.wait_for}' {n}개 살아 있음")
            while alive(args.wait_for):
                sleep(60)
            log(args.work, "선행 종료 — 진행")

    bad = {}
    if args.stage in ("graph", "all"):
        bad["graph"] = stage_graph(args, vids)
    if args.stage in ("bank", "all"):
        for cfg_path in [c for c in args.bank_configs.split(",") if c]:
            bad[load_config(cfg_path)["generation"]] = stage_bank(args, vids, cfg_path)

    log(args.work, "=== 요약 ===")
    for key in sorted(bad):
        log(args.work, f"  {key:8s}  실패 샤드 {bad[key]}/{args.shards}")
    log(args.work, f"  {'편수':8s}  {len(vids)}")


if __name__ == "__main__":
    parser = ArgumentParser(description="지정 씬만 마커 무시하고 graph/뱅크 재굽기")
    parser.add_argument("--videos", required=True)                 # 재굽기 대상 목록 파일
    parser.add_argument("--work", required=True)                   # 로그·격리 위치 (tmp/dNN)
    parser.add_argument("--stage", default="all", choices=("graph", "bank", "all"))
    parser.add_argument("--graph_config", default="configs/bank/d182_dynpose100k_graph.json")
    parser.add_argument("--bank_configs",                          # 콤마 구분, 왼쪽부터 순서대로
                        default="configs/bank/d184_dynpose100k_single.json,"
                                "configs/bank/d185_dynpose100k_grid5.json")
    parser.add_argument("--shards", default=8, type=int)           # GPU 4장에 고르게 분배
    parser.add_argument("--wait_for", default=None,                # 이 패턴이 죽을 때까지 대기
                        help="ps 명령줄에 이 문자열이 있는 프로세스가 전부 끝나면 시작한다")
    parser.add_argument("--quarantine", dest="quarantine", action="store_true", default=True,
                        help="뱅크 재굽기 전에 기존 산출물을 <work>/quarantine 으로 옮긴다 (기본)")
    parser.add_argument("--no_quarantine", dest="quarantine", action="store_false")
    main(parser.parse_args())
