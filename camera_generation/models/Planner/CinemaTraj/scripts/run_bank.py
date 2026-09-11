"""뱅크 체인(graph→cloud→route→tau→fit→emit) 단일 python 드라이버. **세대별 bash 복붙을 끝낸다.**

왜 이 파일인가: `scripts/run_*_dNN_shard.sh` 가 23개까지 늘었는데 서로 다른 곳은 플래그 몇 줄
뿐이다. 그 구조에서는 세대가 바뀔 때마다 파일을 복사하게 되고, 복사가 빠뜨린 플래그는
argparse 기본값으로 조용히 대체된다 — D105 가 그 사고였다(한 폴더에 두 규약이 섞였다).
그래서 **뱅크 정체성을 이루는 플래그 전량을 세대 config JSON 한 곳에 적고**, 실행 로직은 여기
하나만 둔다. 새 세대 = `configs/bank/<gen>.json` 한 장이지 새 스크립트가 아니다.

기존 bash 러너들은 **지우지 않는다** — 이미 구워진 세대를 재현할 때의 근거 기록이다. 새 굽기만
이쪽으로 온다.

config 스키마 (`configs/bank/<gen>.json`):

    {
      "generation":   "d150S",              # 로그 접두사
      "output_root":  "out",                # out / out_dynpose / out_trumans
      "eval_data":    null,                 # null 이면 각 스크립트 기본값(vista)
      "tau_bank_dir": "bank_d150S",         # τ 뱅크 폴더
      "bank_dir":     "hole_bank_k6_d150S", # hole 뱅크 폴더 (done 판정도 여기)
      "stages":       ["tau", "fit", "emit"],
      "require_markers": [".graph_s115", ".cloud_s115"],   # 없으면 그 영상 skip
      "graph": {"marker": ".graph_d150", "args": ["--gravity_source", "geocalib", ...]},
      "cloud": {"marker": ".cloud_d150", "args": [...]},
      "route": {"out": "preset_route_d150.json", "args": ["--num_external", "0", ...]},
      "tau":   {"args": [...]},             # sample_camera_bank.py 플래그 전량
      "fit":   {"args": [...]},             # fit_hole_ladder.py 플래그 전량
      "emit":  {"args": []}                 # emit_bank.py 플래그
    }

`--video`/`--output_root`/`--bank_dir`/`--eval_data` 는 드라이버가 붙이므로 `args` 에 쓰지 말 것
(중복되면 argparse 가 뒤엣것을 이겨서 config 를 읽어도 실제 값을 모르게 된다).

사용 예시:

    # 파일럿 6편, GPU 2 한 장
    python scripts/run_bank.py --config configs/bank/d150S.json \
        --videos tmp/d150S/scenes.txt --log_dir logs/d150S --gpu 2

    # 4샤드 (screen 4개에서 shard_id 만 바꿔 기동)
    python scripts/run_bank.py --config configs/bank/d150S.json \
        --videos <list> --log_dir logs/d150S --gpu 0 --num_shards 4 --shard_id 0

    # 특정 단계만 다시
    python scripts/run_bank.py --config configs/bank/d150S.json --videos <list> \
        --log_dir logs/d150S --gpu 2 --stages emit --no_skip_done
"""
from argparse import ArgumentParser
from json import load as json_load
from os import environ, makedirs, path
from subprocess import run as sp_run, PIPE
from sys import executable, stderr
from time import strftime, time


HERE = path.dirname(path.dirname(path.abspath(__file__)))
STAGE_ORDER = ("graph", "cloud", "route", "tau", "fit", "emit")
# BLAS/OpenMP 스레드 상한. **이걸 안 걸면 graph 단계가 CPU 를 48배 낭비한다** (D179 실측,
# 씬 bb3bd56c 1편 · 같은 인자 · 산출물 md5 동일):
#     캡 없음   user 12,326.7 s / wall 393 s / 3136% CPU / 비자발 문맥전환 4,435,047
#     THREADS=8 user    254.0 s / wall 107 s /  241% CPU
# 차이는 실제 연산이 아니라 OpenBLAS/OpenMP 의 spin-wait 다 — 프로세스당 391 threads 가
# 124코어 위에서 서로를 기다리며 코어를 태운다. 샤드를 12개 띄우면 기계 전체가 spin 에 잠긴다.
# 0 이면 캡을 안 건다 (예전 동작).
THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENCV_FOR_THREADS_NUM")
# 단계 -> 실행 대상. `-m` 이 붙은 것은 모듈 실행(`python -m lbm.cloud`).
STAGE_ENTRY = {
    "graph": ("script", "scripts/build_scene_graph.py"),
    "cloud": ("module", "lbm.cloud"),
    "route": ("script", "scripts/route_presets.py"),
    "tau": ("script", "scripts/sample_camera_bank.py"),
    "fit": ("script", "scripts/fit_hole_ladder.py"),
    "emit": ("script", "scripts/emit_bank.py"),
}
# `--eval_data` 를 argparse 로 받는 단계. 나머지(route/emit)에 붙이면 rc=2 다 (§_base_args).
STAGE_TAKES_EVAL_DATA = frozenset({"graph", "cloud", "tau", "fit"})


def _now():
    return strftime("%H:%M:%S")


def _cmd(stage, extra):
    kind, target = STAGE_ENTRY[stage]
    return [executable] + (["-m", target] if kind == "module" else [target]) + list(extra)


def _run(stage, argv, log_path, gpu, capture=False, threads=0):
    """서브프로세스 1회. -> (rc, stdout or "")

    stdout/stderr 은 영상별 로그 파일로 보낸다 — 드라이버 stdout 에는 진행 한 줄만 남긴다
    (`CLAUDE.md` "로그는 전량 tee 하지 말고"). route 만 stdout 을 파싱해야 해서 capture 한다.

    `threads > 0` 이면 BLAS/OpenMP 스레드 상한을 서브프로세스 env 에 심는다 (§THREAD_ENV_KEYS).
    """
    env = dict(environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if threads > 0:
        env.update({key: str(threads) for key in THREAD_ENV_KEYS})
    if capture:
        with open(log_path, "w") as err:
            proc = sp_run(argv, stdout=PIPE, stderr=err, env=env, text=True)
        return proc.returncode, proc.stdout
    with open(log_path, "w") as log:
        proc = sp_run(argv, stdout=log, stderr=log, env=env)
    return proc.returncode, ""


def _base_args(cfg, video, bank_dir=None, stage=None):
    """단계 공통 인자. `--eval_data` 는 **그걸 받는 스크립트에만** 붙인다.

    `route_presets.py` / `emit_bank.py` 는 `--eval_data` argparse 인자가 없다 (둘 다
    `output_root` 아래 산출물만 읽는다). 그래서 무조건 붙이면 `unrecognized arguments` 로
    rc=2 다. vista 세대는 config 의 `eval_data` 가 null 이라 이 경로를 안 밟았고,
    dynpose/trumans 처럼 데이터 위치를 지정하는 세대에서만 터진다 (D157 smoke 에서 실측).
    """
    out = ["--video", video, "--output_root", cfg["output_root"]]
    if cfg.get("eval_data") and stage in STAGE_TAKES_EVAL_DATA:
        out += ["--eval_data", cfg["eval_data"]]
    if bank_dir is not None:
        out += ["--bank_dir", bank_dir]
    return out


def process_video(cfg, video, stages, gpu, log_dir, skip_done, tag, threads=0):
    """영상 1편을 config 의 단계 순서대로 통과시킨다. -> 상태 문자열."""
    root = path.join(cfg["output_root"], video)
    tau_dir = path.join(root, cfg["tau_bank_dir"])
    bank_dir = path.join(root, cfg["bank_dir"])

    if skip_done and path.exists(path.join(bank_dir, "canonical", "canonical.json")):
        return "SKIP(done)"
    if skip_done and path.exists(path.join(bank_dir, "skipped.json")):
        return "SKIP(변이 0)"

    for marker in cfg.get("require_markers", []):
        if not path.exists(path.join(root, marker)):
            return f"FAIL(마커 {marker} 없음)"

    route_args = []
    for stage in stages:
        spec = cfg.get(stage)
        if spec is None:
            continue
        log_path = path.join(log_dir, f"{video}.{stage}.log")
        t0 = time()

        if stage in ("graph", "cloud"):
            marker = path.join(root, spec["marker"])
            if skip_done and path.exists(marker):
                continue
            print(f"[{tag}] {stage.upper():5s} {video}  {_now()}", flush=True)
            argv = _cmd(stage, _base_args(cfg, video, stage=stage) + spec["args"])
            rc, _ = _run(stage, argv, log_path, gpu, threads=threads)
            if rc != 0:
                return f"FAIL({stage} rc={rc})"
            open(marker, "w").close()

        elif stage == "route":
            argv = _cmd(stage, _base_args(cfg, video, stage=stage) + spec["args"]
                        + ["--emit", "args", "--out", path.join(root, spec["out"])])
            rc, text = _run(stage, argv, log_path, gpu, capture=True, threads=threads)
            line = text.strip().splitlines()[-1] if text.strip() else ""
            if rc != 0 or not line:
                return f"FAIL(route rc={rc})"
            route_args = line.split()

        elif stage == "tau":
            if skip_done and (path.exists(path.join(tau_dir, "bank.json"))
                              or path.exists(path.join(tau_dir, "skipped.json"))):
                continue
            print(f"[{tag}] TAU   {video}  {_now()}", flush=True)
            argv = _cmd(stage, _base_args(cfg, video, cfg["tau_bank_dir"], stage) + route_args + spec["args"])
            rc, _ = _run(stage, argv, log_path, gpu, threads=threads)
            if rc != 0 and not path.exists(path.join(tau_dir, "skipped.json")):
                return f"FAIL(tau rc={rc})"

        elif stage == "fit":
            if not path.exists(path.join(tau_dir, "bank.json")):
                # τ 뱅크가 "변이 0" 으로 끝났으면 그 사실을 hole 뱅크에도 남겨서 다음 실행이 건너뛴다.
                if path.exists(path.join(tau_dir, "skipped.json")):
                    makedirs(bank_dir, exist_ok=True)
                    with open(path.join(tau_dir, "skipped.json")) as src, \
                            open(path.join(bank_dir, "skipped.json"), "w") as dst:
                        dst.write(src.read())
                    return "SKIP(변이 0)"
                return "FAIL(tau 산출물 없음)"
            print(f"[{tag}] FIT   {video}  {_now()}", flush=True)
            argv = _cmd(stage, _base_args(cfg, video, cfg["bank_dir"], stage)
                        + ["--tau_bank_dir", cfg["tau_bank_dir"]] + spec["args"])
            rc, _ = _run(stage, argv, log_path, gpu, threads=threads)
            if rc != 0:
                return f"FAIL(fit rc={rc})"

        elif stage == "emit":
            argv = _cmd(stage, _base_args(cfg, video, cfg["bank_dir"], stage) + spec["args"])
            rc, _ = _run(stage, argv, log_path, gpu, threads=threads)
            if rc != 0:
                return f"FAIL(emit rc={rc})"
        print(f"[{tag}] {stage.upper():5s} {video} rc=0 {time() - t0:6.1f}s  {_now()}", flush=True)
    return "OK"


def load_config(config_path):
    """config 를 읽는다. `"extends": "<파일명>"` 이면 부모를 먼저 읽고 **최상위 키 단위로** 덮는다.

    한 축만 다른 대조 세대(예: `--tau_denom` 만 S vs z_med_frame0)가 config 를 통째로 복사하지
    않게 하려는 것. 얕은 병합인 이유는 `tau.args` 같은 리스트를 반쯤 물려받으면 "실제로 넘어간
    플래그"를 파일만 보고 알 수 없어져서다 — 덮을 거면 그 단계는 통째로 다시 쓴다.
    """
    with open(config_path) as f:
        cfg = json_load(f)
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    base = load_config(path.join(path.dirname(config_path), parent))
    base.update(cfg)
    return base


def load_videos(spec):
    """`--videos` 는 파일 경로거나 콤마 목록이다."""
    if path.exists(spec):
        with open(spec) as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    return [v.strip() for v in spec.split(",") if v.strip()]


def main():
    parser = ArgumentParser(description="뱅크 체인 단일 드라이버 (세대 차이는 config JSON 으로)")
    parser.add_argument("--config", required=True, help="configs/bank/<gen>.json")
    parser.add_argument("--videos", required=True, help="영상 목록 파일 또는 콤마 구분 이름")
    parser.add_argument("--log_dir", required=True, help="영상별 stdout 로그 위치")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES 값 (0~4)")
    parser.add_argument("--stages", default=None,
                        help="콤마 구분. 기본은 config 의 stages")
    parser.add_argument("--num_shards", default=1, type=int)
    parser.add_argument("--shard_id", default=0, type=int)
    parser.add_argument("--skip_done", dest="skip_done", action="store_true", default=True,
                        help="이미 산출물이 있으면 건너뛴다 (기본)")
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    parser.add_argument("--threads", default=8, type=int,
                        help="단계 서브프로세스의 BLAS/OpenMP 스레드 상한. 0 = 캡 없음(예전 동작). "
                             "기본 8 (§THREAD_ENV_KEYS — graph 실측 CPU 48배 절감, 산출물 md5 동일)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stages = [s for s in (args.stages.split(",") if args.stages else cfg["stages"])]
    unknown = [s for s in stages if s not in STAGE_ORDER]
    assert not unknown, f"모르는 stage: {unknown} (가능: {STAGE_ORDER})"
    stages.sort(key=STAGE_ORDER.index)
    if args.gpu is not None:
        assert str(args.gpu) in list("01234"), f"GPU 는 0~4 만 (CLAUDE.md): {args.gpu}"

    videos = load_videos(args.videos)
    mine = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_id]
    makedirs(args.log_dir, exist_ok=True)
    tag = f"{cfg.get('generation', '?')}/s{args.shard_id}"
    print(f"[{tag}] {len(mine)}/{len(videos)}편  stages={','.join(stages)}  "
          f"bank={cfg['bank_dir']}  gpu={args.gpu}  "
          f"threads={args.threads or '캡 없음'}  {_now()}", flush=True)

    results = []
    for video in mine:
        t0 = time()
        try:
            status = process_video(cfg, video, stages, args.gpu, args.log_dir, args.skip_done, tag,
                                   threads=args.threads)
        except Exception as exc:                      # 한 편이 죽어도 샤드는 계속 간다
            status = f"ERROR({type(exc).__name__}: {exc})"
        results.append((video, status, time() - t0))
        if not status.startswith("OK"):
            print(f"[{tag}] {status:24s} {video}", flush=True)

    print(f"\n[{tag}] === 요약 ===", file=stderr)
    width = max([len(v) for v, _, _ in results] + [5])
    for video, status, dt in results:
        print(f"  {video:<{width}}  {status:<24s} {dt:7.1f}s", file=stderr)
    ok = sum(1 for _, s, _ in results if s == "OK")
    print(f"  ---- OK {ok} / {len(results)}  ({_now()})", file=stderr)


if __name__ == "__main__":
    main()
