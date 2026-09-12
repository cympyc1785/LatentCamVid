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
    python scripts/chain_bank_rounds.py --config configs/bank/d183_dynpose100k_bank.json \
        --videos /data1/.../tmp/d172/videos_10346.txt --work /data1/.../tmp/d183 \
        --graph_marker .graph_d182 --bank_dir hole_bank_d183 \
        --graph_pattern "run_bank.py --config configs/bank/d182_"
    # 강등만
    python scripts/chain_bank_rounds.py ... --stage demote
"""
import json
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
GPUS = ["0", "1", "2", "3"]                   # 사용자 지시: GPU 4~7 은 안 쓴다


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


def ready_videos(vids, marker, bank_dir):
    """graph 는 끝났고 이 세대 뱅크는 아직 없는 편."""
    out = []
    for v in vids:
        root = path.join(ROOT, OUT_ROOT, v)
        if not path.exists(path.join(root, marker)):
            continue
        bank = path.join(root, bank_dir)
        if path.exists(path.join(bank, "canonical", "canonical.json")):
            continue
        if path.exists(path.join(bank, "skipped.json")):
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


def fan_out(work, name, argv_of_shard, num_shards, log_name, gpus=True):
    """샤드 num_shards 개를 동시에 띄우고 전부 끝날 때까지 기다린다. -> 실패 샤드 수."""
    makedirs(path.join(work, log_name), exist_ok=True)
    procs = []
    for shard in range(num_shards):
        env = dict(environ)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = GPUS[shard % len(GPUS)]
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
        return [PY, "-u", "scripts/dynpose_dynamic_mask_from_seg.py",
                "--eval_data", args.eval_data, "--output_root", OUT_ROOT,
                "--demote_static_objects", "--videos"] + lists[shard]

    return fan_out(args.work, f"강등({round_tag})", argv, len(lists),
                   f"demote_logs_{round_tag}", gpus=False)


def run_bank(args, vids, round_tag):
    listing = path.join(args.work, f"ready_{round_tag}.txt")
    with open(listing, "w", encoding="utf-8") as fh:
        fh.write("\n".join(vids))
    makedirs(path.join(args.work, "bank_scene_logs"), exist_ok=True)

    def argv(shard):
        return [PY, "-u", "scripts/run_bank.py",
                "--config", args.config, "--videos", listing,
                "--log_dir", path.join(args.work, "bank_scene_logs"),
                "--stages", "route,tau,fit,emit",
                "--num_shards", str(args.bank_shards), "--shard_id", str(shard)]

    bad = fan_out(args.work, f"뱅크({round_tag})", argv, args.bank_shards,
                  f"bank_logs_{round_tag}")
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


def main(args):
    makedirs(args.work, exist_ok=True)
    vids = videos(args.videos)
    for rnd in range(1, args.max_rounds + 1):
        tag = f"r{rnd:02d}"
        ready = ready_videos(vids, args.graph_marker, args.bank_dir)
        alive = graph_alive(args.graph_pattern)
        log(f"=== 라운드 {tag} — ready {len(ready)}/{len(vids)}, "
            f"이미 구움 {baked(vids, args.bank_dir)}, graph {'진행 중' if alive else '종료'}")
        if not ready:
            if not alive:
                log("ready 0 이고 graph 도 끝났다 — 종료")
                return 0
            log(f"ready 0 — {args.poll / 60:.0f}분 뒤 재스캔")
            sleep(args.poll)
            continue
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
        if not graph_alive(args.graph_pattern) and not ready_videos(
                vids, args.graph_marker, args.bank_dir):
            log("graph 종료 + ready 0 — 전량 완료")
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
    # graph 드라이버가 살아 있는지 볼 `ps` 부분문자열. 죽고 ready 0 이면 루프를 끝낸다.
    parser.add_argument("--graph_pattern", default="run_bank.py --config configs/bank/d182_")
    parser.add_argument("--stage", default="all", choices=("all", "demote", "bank"))
    parser.add_argument("--bank_shards", default=4, type=int)     # GPU 1장당 1개
    parser.add_argument("--demote_shards", default=8, type=int)   # CPU 전용 (numpy+PIL)
    parser.add_argument("--round_cap", default=0, type=int)       # 0 = 무제한
    parser.add_argument("--max_rounds", default=40, type=int)
    parser.add_argument("--poll", default=900, type=int)          # ready 0 일 때 재스캔 간격(초)
    sys.exit(main(parser.parse_args()))
