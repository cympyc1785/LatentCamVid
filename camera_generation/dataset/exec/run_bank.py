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
    python exec/run_bank.py --config configs/bank/d150S.json \
        --videos tmp/d150S/scenes.txt --log_dir logs/d150S --gpu 2

    # 4샤드 (screen 4개에서 shard_id 만 바꿔 기동)
    python exec/run_bank.py --config configs/bank/d150S.json \
        --videos <list> --log_dir logs/d150S --gpu 0 --num_shards 4 --shard_id 0

    # 특정 단계만 다시
    python exec/run_bank.py --config configs/bank/d150S.json --videos <list> \
        --log_dir logs/d150S --gpu 2 --stages emit --no_skip_done
"""
import sys
from argparse import ArgumentParser
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from json import dump as json_dump, load as json_load
from os import environ, execv, makedirs, path
from subprocess import run as sp_run, PIPE
from sys import executable, stderr
from time import strftime, time
from traceback import print_exc


HERE = path.dirname(path.dirname(path.abspath(__file__)))
# python 은 **스크립트가 있는 폴더**(`scripts/`)를 sys.path 에 넣지 cwd 를 넣지 않는다. 서브프로세스
# 모드에서는 각 단계 스크립트가 제 나름대로 리포 루트를 꽂아 `lbm.*` 를 찾지만, in-process 모드는
# 드라이버 자신이 `lbm.render`(cloud 캐시)와 `lbm.cloud`(module 단계)를 직접 import 한다 —
# 그래서 여기서 못 박는다. 서브프로세스 모드에는 영향이 없다 (이미 들어있으면 안 넣는다).
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# [D271] `mesh` = TRUMANS 전용 mesh 점유/EDT 격자 (`build_trumans_mesh_grid.py`). 예전엔
# `exec/_legacy/run_trumans_d132_shard.sh` 의 PREP 블록에만 있었다. config 에 `mesh` 키가 없으면
# 단계가 통째로 건너뛰어지므로 (process_video 의 `spec is None`) 기존 세대는 동작 그대로다.
STAGE_ORDER = ("graph", "cloud", "mesh", "route", "tau", "fit", "emit")
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
    "graph": ("script", "fit/graph/build_scene_graph.py"),
    "cloud": ("module", "lbm.cloud"),
    "mesh": ("script", "fit/ingest/build_trumans_mesh_grid.py"),
    "route": ("script", "fit/bank/route_presets.py"),
    "tau": ("script", "fit/bank/sample_camera_bank.py"),
    "fit": ("script", "fit/bank/fit_hole_ladder.py"),
    "emit": ("script", "fit/bank/emit_bank.py"),
}
# `--eval_data` 를 argparse 로 받는 단계. 나머지(route/emit)에 붙이면 rc=2 다 (§_base_args).
STAGE_TAKES_EVAL_DATA = frozenset({"graph", "cloud", "tau", "fit"})


def _now():
    return strftime("%H:%M:%S")


def _cmd(stage, extra):
    kind, target = STAGE_ENTRY[stage]
    return [executable] + (["-m", target] if kind == "module" else [target]) + list(extra)


def _empty_dynmask(log_path):
    """tau 로그가 D176-c 의 빈 `dynamic_mask` 가드로 끝났나 (D188-b).

    토큰을 `lbm.cloud` 에서 **가져다 쓰는** 이유: 여기에 문자열을 복사해 두면 가드 문구를 고칠 때
    한쪽만 바뀌어 조용히 안 잡히고, 그러면 다시 라운드마다 재시도되는 옛 동작으로 돌아간다.
    import 을 함수 안에 둔 것은 `lbm.cloud` 가 torch 를 끌고 오기 때문이다 — subprocess 모드
    드라이버는 torch 가 필요 없고, 이 경로는 실패했을 때만 지난다.
    """
    from lbm.cloud import EMPTY_DYNMASK_TOKEN
    return _log_has(log_path, EMPTY_DYNMASK_TOKEN)


def _no_emittable(log_path):
    """emit 로그가 "이 씬은 카메라 0개" 로 끝났나 (D188-d).

    `emit_bank.py` 는 rc=1 을 **세 곳**에서 낸다 — 재구성 잔차 초과 / `--dump_poses` 오용 /
    변이 0. 앞의 둘은 진짜 오류라 재시도 가치가 있지만 세 번째는 게이트 판정이라 다시 돌려도
    똑같다. 토큰으로만 가른다 (§`emit_bank.NO_EMITTABLE_TOKEN`).
    """
    from fit.bank.emit_bank import NO_EMITTABLE_TOKEN
    return _log_has(log_path, NO_EMITTABLE_TOKEN)


def _log_has(log_path, token):
    try:
        with open(log_path, encoding="utf-8", errors="replace") as file:
            return token in file.read()
    except OSError:
        return False


# ── in-process 실행 (D180) ───────────────────────────────────────────────────────
# 왜: 씬당 205.8s 중 **51.3s(25%)가 프로세스 경계 비용**이다 (8샤드 구간 실측).
#     import   route 0.45s + tau 7.27s + fit 6.74s + emit ~7.4s = 21.9s   <- 씬마다 다시 낸다
#     중복 cloud  load_recon 4.74 + preprocess 9.37 + build_cloud 0.60 = 14.7s 를 tau/fit 이 각각
# 샤드는 이미 한 프로세스가 씬을 순회하는 구조(`main` 의 for 문)라, 단계 호출만 in-process 로
# 바꾸면 import 는 **씬당이 아니라 샤드당 1회**로 떨어지고, 같은 프로세스 안이므로
# `lbm.render` 의 cloud 캐시가 tau->fit 중복도 없앤다.
#
# 기본은 `subprocess` = **예전 동작 그대로**다 (CLAUDE.md 분기 규칙). in-process 는 한 단계의
# 메모리 누수·전역 오염이 샤드 전체를 물고 들어갈 수 있어서, 기본값으로 두지 않는다.
_STAGE_MODULES = {}


def _stage_module(stage):
    """단계 스크립트를 **모듈로** 읽는다 (`__main__` 블록은 안 돈다). 프로세스당 1회."""
    if stage in _STAGE_MODULES:
        return _STAGE_MODULES[stage]
    kind, target = STAGE_ENTRY[stage]
    if kind == "module":
        mod = import_module(target)
    else:
        name = f"_bank_stage_{path.splitext(path.basename(target))[0]}"
        spec = spec_from_file_location(name, path.join(HERE, target))
        mod = module_from_spec(spec)
        sys.modules[name] = mod          # dataclass/pickle 이 모듈을 되찾을 수 있게
        spec.loader.exec_module(mod)
    _STAGE_MODULES[stage] = mod
    return mod


def _run_inproc(stage, argv, log_path, capture):
    """같은 프로세스에서 단계 `main()` 을 부른다. -> (rc, stdout or "") 또는 None(=미지원).

    `build_parser()` 가 없는 단계는 None 을 돌려 호출자가 서브프로세스로 되돌리게 한다 —
    릴/구세대 스크립트를 이 경로에 억지로 태우지 않기 위함이다.

    `KeyboardInterrupt` 는 **안 잡는다**. 샤드를 Ctrl+C 로 내리는 게 정상 종료 절차인데
    (`CLAUDE.md`: 하드킬은 DataLoader worker 를 고아로 남긴다) 여기서 삼키면 그게 막힌다.
    """
    mod = _stage_module(stage)
    if not hasattr(mod, "build_parser"):
        return None
    buf = StringIO() if capture else None
    with open(log_path, "w") as log:
        out = buf if capture else log
        try:
            with redirect_stdout(out), redirect_stderr(log):
                mod.main(mod.build_parser().parse_args(argv))
            rc = 0
        except SystemExit as exc:        # argparse 오류 + 스크립트의 명시적 SystemExit("...")
            rc = exc.code if isinstance(exc.code, int) else 1
            if exc.code and not isinstance(exc.code, int):
                print(f"SystemExit: {exc.code}", file=log)
        except Exception:                # noqa: BLE001 — 한 단계가 죽어도 샤드는 계속 간다
            print_exc(file=log)
            rc = 1
    return rc, (buf.getvalue() if capture else "")


def _run(stage, extra, log_path, gpu, capture=False, threads=0, mode="subprocess"):
    """단계 1회 실행. -> (rc, stdout or "")

    stdout/stderr 은 영상별 로그 파일로 보낸다 — 드라이버 stdout 에는 진행 한 줄만 남긴다
    (`CLAUDE.md` "로그는 전량 tee 하지 말고"). route 만 stdout 을 파싱해야 해서 capture 한다.

    `threads > 0` 이면 BLAS/OpenMP 스레드 상한을 서브프로세스 env 에 심는다 (§THREAD_ENV_KEYS).
    in-process 모드에서는 `main()` 이 프로세스 env 에 **한 번** 심어 둔다 (OpenMP 는 로드
    시점에 env 를 읽으므로 나중에 바꿔도 안 먹는다).
    """
    if mode == "inproc":
        got = _run_inproc(stage, list(extra), log_path, capture)
        if got is not None:
            return got                   # None 이면 아래 서브프로세스 경로로 되돌아간다
    argv = _cmd(stage, extra)
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


def process_video(cfg, video, stages, gpu, log_dir, skip_done, tag, threads=0, mode="subprocess"):
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
    ran = False                  # 실제로 돈 단계가 있었나 — recycle 카운트용 (skip 만 한 편은 안 센다)
    for stage in stages:
        spec = cfg.get(stage)
        if spec is None:
            continue
        log_path = path.join(log_dir, f"{video}.{stage}.log")
        t0 = time()
        if not (skip_done and ((stage in ("graph", "cloud") and path.exists(path.join(root, spec["marker"])))
                               or (stage == "mesh" and path.exists(path.join(root, "mesh_grid.npz"))))):
            ran = True

        if stage in ("graph", "cloud"):
            marker = path.join(root, spec["marker"])
            if skip_done and path.exists(marker):
                continue
            print(f"[{tag}] {stage.upper():5s} {video}  {_now()}", flush=True)
            rc, _ = _run(stage, _base_args(cfg, video, stage=stage) + spec["args"],
                         log_path, gpu, threads=threads, mode=mode)
            if rc != 0:
                return f"FAIL({stage} rc={rc})"
            open(marker, "w").close()

        elif stage == "mesh":
            # 산출물 자체가 done 판정이다 (legacy PREP 블록의 `[ ! -f mesh_grid.npz ]` 와 같다).
            # GPU 불필요 — Blender 가 CPU 로 한 번 뜬다 (로드 16 s).
            if skip_done and path.exists(path.join(root, "mesh_grid.npz")):
                continue
            print(f"[{tag}] MESH  {video}  {_now()}", flush=True)
            rc, _ = _run(stage, ["--video", video, "--output_root", cfg["output_root"]]
                         + spec.get("args", []), log_path, None, threads=threads,
                         mode="subprocess")
            if rc != 0 or not path.exists(path.join(root, "mesh_grid.npz")):
                return f"FAIL(mesh rc={rc})"

        elif stage == "route":
            rc, text = _run(stage, _base_args(cfg, video, stage=stage) + spec["args"]
                            + ["--emit", "args", "--out", path.join(root, spec["out"])],
                            log_path, gpu, capture=True, threads=threads, mode=mode)
            line = text.strip().splitlines()[-1] if text.strip() else ""
            # D184. rc=3 = `route_presets --skip_if_empty` 가 "이 씬엔 anchor(또는 슬롯)가
            # 없다"고 정상 종료한 것. 크래시가 아니므로 `skipped.json` 을 남겨 다음 라운드가
            # 건너뛰게 한다 — 안 남기면 라운드마다 같은 씬을 다시 집는다.
            if rc == 3:
                makedirs(bank_dir, exist_ok=True)
                with open(path.join(bank_dir, "skipped.json"), "w", encoding="utf-8") as file:
                    json_dump({"format": "lbm_camera_bank_skipped_v1", "video": video,
                               "reason": "no_surviving_anchors", "stage": "route",
                               "detail": "route_presets --skip_if_empty: anchor 또는 슬롯이 0개",
                               "log": log_path}, file, ensure_ascii=False, indent=1)
                return "SKIP(앵커 0)"
            if rc != 0 or not line:
                return f"FAIL(route rc={rc})"
            route_args = line.split()

        elif stage == "tau":
            if skip_done and (path.exists(path.join(tau_dir, "bank.json"))
                              or path.exists(path.join(tau_dir, "skipped.json"))):
                continue
            print(f"[{tag}] TAU   {video}  {_now()}", flush=True)
            rc, _ = _run(stage, _base_args(cfg, video, cfg["tau_bank_dir"], stage)
                         + route_args + spec["args"],
                         log_path, gpu, threads=threads, mode=mode)
            if rc != 0 and not path.exists(path.join(tau_dir, "skipped.json")):
                # D188-b. `dynamic_mask` 가 전부 0 인 씬은 `assert_dynamic_mask_nonempty` 가
                # 막는다 (D176-c 가드). 그게 **정상 동작**인데 여기서 FAIL 로 끝내면 뱅크에
                # 아무 흔적도 안 남아 `chain_bank_rounds.ready_videos()` 가 라운드마다 같은 씬을
                # 다시 집는다 — route rc=3 과 똑같은 문제고, 똑같이 `skipped.json` 으로 닫는다.
                # 실측 (2026-09-13, 10,346편): 45편이 이 상태였고 `--max_rounds 200` 이면
                # 코퍼스를 다 구운 뒤에도 45편짜리 라운드가 계속 돌았을 것이다.
                # 고칠 수 있는 씬(dynmask 단계 미실행)은 이 커밋 전에 63편 복구했다. 남은 것은
                # SAM3 가 동적 물체를 하나도 못 찾았거나 D177 강등이 전부 뺀 씬이라 **입력을
                # 고쳐도 안 바뀐다** — 게이트를 내리는 게 아니라 구조적 탈락으로 기록하는 것이다.
                if _empty_dynmask(log_path):
                    makedirs(bank_dir, exist_ok=True)
                    with open(path.join(bank_dir, "skipped.json"), "w", encoding="utf-8") as file:
                        json_dump({"format": "lbm_camera_bank_skipped_v1", "video": video,
                                   "reason": "empty_dynamic_mask", "stage": "tau",
                                   "detail": "dynamic_mask 가 전부 0 — SAM3 동적 트랙 0개 또는 "
                                             "D177 강등이 전부 뺐다. 되살리려면 seg_instances 를 "
                                             "다시 뽑거나 --allow_empty_dynamic_mask 로 굽는다.",
                                   "log": log_path}, file, ensure_ascii=False, indent=1)
                    return "SKIP(동적마스크 0)"
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
            rc, _ = _run(stage, _base_args(cfg, video, cfg["bank_dir"], stage)
                         + ["--tau_bank_dir", cfg["tau_bank_dir"]] + spec["args"],
                         log_path, gpu, threads=threads, mode=mode)
            if rc != 0:
                return f"FAIL(fit rc={rc})"

        elif stage == "emit":
            rc, _ = _run(stage, _base_args(cfg, video, cfg["bank_dir"], stage) + spec["args"],
                         log_path, gpu, threads=threads, mode=mode)
            if rc != 0:
                # D188-d. "변이 0" 은 **게이트 판정**이지 크래시가 아니다. 그런데 여기서 FAIL 로
                # 끝내면 뱅크에 `canonical.json` 도 `skipped.json` 도 안 남아
                # `chain_bank_rounds.done_bank()` 가 False 를 내고, 그 씬이 라운드마다 다시
                # ready 로 들어온다 — tau 가 캐시돼 있어 route+fit+emit 만 90초쯤 태우고 똑같이
                # 실패한다. 실측(2026-09-13): 새 라운드가 집은 50편 중 **49편**이 옛 emit 실패의
                # 재탕이었고, 실패분이 ready 앞쪽에 쌓이는 구조라 라운드마다 낭비가 커진다.
                # route rc=3 / 빈 dynamic_mask 와 같은 처방이고, **게이트를 내린 것이 아니다**.
                if _no_emittable(log_path):
                    makedirs(bank_dir, exist_ok=True)
                    with open(path.join(bank_dir, "skipped.json"), "w", encoding="utf-8") as file:
                        json_dump({"format": "lbm_camera_bank_skipped_v1", "video": video,
                                   "reason": "no_emittable_variants", "stage": "emit",
                                   "detail": "fit 된 변이는 있는데 emit 필터가 전부 뺐다 — d188 "
                                             "실측 56편은 전부 `--picked_only` 인데 picked=0 "
                                             "(변이 3~12개). 되살리려면 `fit_hole_ladder` 의 "
                                             "pick 게이트를 고쳐 다시 fit 해야 한다.",
                                   "log": log_path}, file, ensure_ascii=False, indent=1)
                    return "SKIP(카메라 0)"
                return f"FAIL(emit rc={rc})"
        print(f"[{tag}] {stage.upper():5s} {video} rc=0 {time() - t0:6.1f}s  {_now()}", flush=True)
    # 모든 단계를 skip_done 으로 건너뛴 편은 "OK(이미 완료)" — 요약에선 OK 로 세되 recycle 엔 안 센다.
    # (d277T prep: 재실행 직후 앞의 완료 편 50개가 곧바로 recycle 을 다시 불러 5861회 무한 루프)
    return "OK" if ran else "OK(이미 완료)"


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
    parser.add_argument("--exec", dest="exec_mode", default="subprocess",
                        choices=("subprocess", "inproc"),
                        help="subprocess = 단계마다 새 프로세스(기본, 예전 동작). "
                             "inproc = 같은 프로세스에서 단계 main() 호출 — import 가 씬당이 "
                             "아니라 샤드당 1회가 되고 tau/fit 의 cloud 재구축이 캐시된다 (§_run_inproc). "
                             "**config 에 최상위 `\"exec\"` 키가 있으면 그쪽이 이긴다** (§exec_mode)")
    parser.add_argument("--inproc_recycle", default=50, type=int,
                        help="inproc 모드에서 N편마다 샤드를 **자기 자신으로 재실행**(execv)해 "
                             "누적 메모리를 턴다. 0 = 끔. 재실행 비용은 import 1회(~8s)뿐이고 "
                             "`--skip_done` 이 이미 끝낸 편을 건너뛰므로 진행은 이어진다.")
    args = parser.parse_args()

    # config 를 **env 를 심기 전에** 읽는다 — `load_config` 는 순수 JSON 이라 무거운 import 가
    # 없고, 아래 exec_mode 결정이 config 를 봐야 한다.
    cfg = load_config(args.config)

    # §exec_mode — 실행 방식은 **config 가 CLI 를 이긴다**. 뒤집힌 우선순위인 이유:
    # `chain_bank_rounds.fan_out` 이 `--exec` 를 **늘 명시로** 넘기므로 argparse 쪽에는
    # "안 줬음"이 존재하지 않는다. 그래서 CLI 우선으로 두면 config 키가 영영 안 먹는다.
    # 이 키의 쓸모는 **도는 체인을 안 죽이고 세대 설정만 고쳐 다음 라운드부터 바꾸는 것**이다
    # (fan_out 이 라운드마다 샤드를 새로 띄우며 config 를 다시 읽는다).
    # 키가 없으면 CLI 값 그대로라 옛 동작은 비트 동일하다.
    exec_mode = cfg.get("exec", args.exec_mode)
    assert exec_mode in ("subprocess", "inproc"), f"모르는 exec: {exec_mode}"

    # in-process 모드는 이 프로세스가 곧 단계 프로세스다 — CUDA/OpenMP 는 **첫 import 전에**
    # env 를 읽으므로 여기서 못 박아야 한다 (나중에 바꾸면 안 먹는다).
    if exec_mode == "inproc":
        if args.gpu is not None:
            environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        if args.threads > 0:
            environ.update({key: str(args.threads) for key in THREAD_ENV_KEYS})

    stages = [s for s in (args.stages.split(",") if args.stages else cfg["stages"])]
    unknown = [s for s in stages if s not in STAGE_ORDER]
    assert not unknown, f"모르는 stage: {unknown} (가능: {STAGE_ORDER})"
    stages.sort(key=STAGE_ORDER.index)
    if args.gpu is not None:
        # CLAUDE.md 기본은 0~4 다. 사용자가 **특정 작업에 한해** 그 위를 열어 줄 때만
        # `BANK_GPU_ALLOW` 로 넓힌다 (D192, 사용자 지시 2026-09-14 "5~7 재사용").
        # 왜 환경변수인가: 이 가드는 여태 `--gpu` 를 **안 주고** `CUDA_VISIBLE_DEVICES` 를
        # 밖에서 export 하면 그냥 뚫렸다 (:205 `env = dict(environ)`). 그 우회는 로그에
        # `gpu=None` 으로 찍혀 어느 GPU 를 썼는지 사후에 알 수 없다. 통로를 하나로 모아
        # 로그에 남긴다. 안 주면 문자열이 "01234" 그대로라 옛 동작과 같다.
        allow = environ.get("BANK_GPU_ALLOW", "01234")
        assert str(args.gpu) in list(allow), \
            f"GPU 는 {allow} 만 (CLAUDE.md / BANK_GPU_ALLOW): {args.gpu}"

    videos = load_videos(args.videos)
    mine = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_id]
    makedirs(args.log_dir, exist_ok=True)
    tag = f"{cfg.get('generation', '?')}/s{args.shard_id}"
    print(f"[{tag}] {len(mine)}/{len(videos)}편  stages={','.join(stages)}  "
          f"bank={cfg['bank_dir']}  gpu={args.gpu}  exec={exec_mode}  "
          f"threads={args.threads or '캡 없음'}  {_now()}", flush=True)

    clear_cache = None
    if exec_mode == "inproc":
        from lbm.render import clear_cloud_cache, set_cloud_cache                # noqa: PLC0415
        set_cloud_cache(True)            # tau -> fit 의 cloud 재구축 제거 (씬당 14.7s)
        clear_cache = clear_cloud_cache

    results = []
    done = 0
    for video in mine:
        t0 = time()
        try:
            status = process_video(cfg, video, stages, args.gpu, args.log_dir, args.skip_done, tag,
                                   threads=args.threads, mode=exec_mode)
        except Exception as exc:                      # 한 편이 죽어도 샤드는 계속 간다
            status = f"ERROR({type(exc).__name__}: {exc})"
        if clear_cache is not None:
            clear_cache()                # 씬 경계 — 캐시는 한 편분만 들고 있는다
        results.append((video, status, time() - t0))
        if not status.startswith("OK"):
            print(f"[{tag}] {status:24s} {video}", flush=True)
        if status == "OK":
            done += 1
        if exec_mode == "inproc" and args.inproc_recycle > 0 and done >= args.inproc_recycle:
            # 누수가 있어도 8샤드가 새벽에 OOM 으로 죽지 않게 하는 보호장치. `--skip_done` 이
            # 이미 끝낸 편을 건너뛰므로 같은 인자로 다시 띄우면 그 자리에서 이어진다.
            print(f"[{tag}] recycle — {done}편 처리 후 재실행  {_now()}", flush=True)
            sys.stdout.flush()
            stderr.flush()
            # `-u` 는 sys.argv 에 안 남는다 — 드라이버가 로그를 실시간으로 읽으므로 env 로 건다.
            environ["PYTHONUNBUFFERED"] = "1"
            execv(executable, [executable] + sys.argv)

    print(f"\n[{tag}] === 요약 ===", file=stderr)
    width = max([len(v) for v, _, _ in results] + [5])
    for video, status, dt in results:
        print(f"  {video:<{width}}  {status:<24s} {dt:7.1f}s", file=stderr)
    ok = sum(1 for _, s, _ in results if s.startswith("OK"))
    print(f"  ---- OK {ok} / {len(results)}  ({_now()})", file=stderr)


if __name__ == "__main__":
    main()
