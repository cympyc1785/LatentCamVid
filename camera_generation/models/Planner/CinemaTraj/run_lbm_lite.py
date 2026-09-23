"""LBM-Lite 전 단계 오케스트레이터. `graph → board → loop → decode → emit → verify` 를 한 줄로.

왜 따로 두나 — 단계별 스크립트가 **같은 값을 각자 기본값으로 들고 있다.** `start_mode` 는
loop/decode/emit/verify 네 군데, `aim_anchor`/`aim_ramp_frames` 는 decode/emit/verify 세 군데에
있고, 어긋나면 emit 의 stale-poses 가드에 걸리거나(운이 좋을 때) 지문이 같아서 조용히 옛
pose 를 재사용한다(운이 나쁠 때). 여기서는 `shared` 블록 하나를 **그 값을 받는 단계에만**
뿌리므로 어긋날 자리가 없다.

각 단계는 여전히 독립 실행 가능하다 — 이 파일은 subprocess 로 부르기만 하고, 어떤 인자를 줬는지
`--dry_run` 으로 그대로 찍는다. 기존 사용법(`python -m lbm.loop --video camel`)은 안 바뀐다.

stage 이름과 실제 스크립트:
    nouns   fit/graph/extract_static_nouns.py     정적 명사 (VLM)      * 기본 실행 안 함
    seg     fit/ingest/sam3_static_instances.py    정적 인스턴스 (SAM3) * 기본 실행 안 함
    graph   fit/graph/build_scene_graph.py        scene_graph.json + obb_overlay.mp4
    pcd     python -m lbm.cloud                 4D 점군 cloud.npz (board/bank/fit 가 읽는다)
    board   fit/bank/build_candidate_board.py    후보 풀 → 게이트 → board (별칭 `cloud`)
    loop    python -m lbm.loop                  VLM select/micro/traj → decision.json
    decode  decode/build_poses.py               decision → poses (49,4,4)  (별칭 `poses`)
    emit    decode/emit.py                      canonical.json/.npz + emit 명령
    verify  verify.py                           지표 8종 + 프리뷰 6종

뱅크 경로 (VLM 루프를 안 쓰는 **카메라 augmentation** 경로. `--stage bank_all`):
    dynseg    video_generation/scripts/sam3_seg_instances.py  동적 인스턴스 (SAM3) * 기본 실행 안 함
    bank      fit/bank/sample_camera_bank.py       소스 측정 + 변이 나열 → bank/
    fit       fit/bank/fit_hole_ladder.py          hole 사다리 이분법 → hole_bank/
    bankemit  fit/bank/emit_bank.py                뱅크 전량 → canonical
    bankvid   viz/render_bank_videos.py       타일 애니메이션 preview.mp4

`--stage all` = graph,pcd,board,loop,decode,emit,verify (nouns/seg 는 이름을 직접 줘야 돈다 —
공유 `eval_data` 디렉토리에 쓰고 SAM3 가 GPU 를 오래 잡는다).
`--stage bank_all` = graph,pcd,bank,fit,bankemit,bankvid — VLM 없이 뱅크만 만드는 경로.
앞단(dynseg/nouns/seg)은 `eval_data` 공유 디렉토리에 쓰므로 여기서도 이름을 직접 줘야 돈다.

각 단계의 **영상당 소요시간**이 매니페스트(`out/run_lbm_lite.json`)의 `results[].seconds` 와
끝의 요약표에 그대로 남는다 — 52편 전량 ETA 를 여기서 뽑는다.

env: 전부 `vista4d` 하나 (`configs/default.json` 의 `python`). 단계 시작 전에 `--help` 를 한 번
찔러 import 를 검사하고, 실패하면 그 자리에서 멈추고 `conda run -n <env>` 명령을 찍는다.

예시:
    python run_lbm_lite.py --stage all --videos camel
    python run_lbm_lite.py --stage all --cuda 1                      # GPU 1 로 두 영상 전부
    python run_lbm_lite.py --stage board,loop --videos camel --dry_run
    python run_lbm_lite.py --stage all --set loop.no_vlm=true        # VLM 없이 fallback 결정만
    python run_lbm_lite.py --stage board --set board.board_size=27 \
        --set board.board_columns=9 --set board.tile_width=480 --set board.tile_height=270
"""
import json
import subprocess
import sys
import time
from argparse import ArgumentParser
from os import environ, makedirs, path

CINEMATRAJ_ROOT = path.dirname(path.abspath(__file__))
CONFIG_DEFAULT = path.join(CINEMATRAJ_ROOT, "configs", "default.json")
CONFIG_FORMAT = "lbm_lite_config_v1"

# stage -> (실행 대상, 영상 인자, shared 에서 받아가는 키)
#   "module:" 접두사면 `python -m <name>`, 아니면 스크립트 경로.
#   영상 인자가 "--videos" 인 단계는 영상 여러 개를 한 번에 받는다 (샤딩 스크립트라서).
STAGES = {
    "nouns":  {"target": "fit/graph/extract_static_nouns.py",  "video_flag": "--videos", "accepts": []},
    "seg":    {"target": "fit/ingest/sam3_static_instances.py", "video_flag": "--videos", "accepts": []},
    "graph":  {"target": "fit/graph/build_scene_graph.py",     "video_flag": "--video",  "accepts": []},
    # board/bank/fit/bankvid 는 전부 `<out>/<video>/cloud.npz` 를 **읽기만** 한다 (없으면 assert).
    # 파일럿 2편은 손으로 `python -m lbm.cloud` 를 돌려놨어서 안 보였던 구멍이다 — 전량 확장에는 필수.
    "pcd":    {"target": "module:lbm.cloud",                 "video_flag": "--video",
               "accepts": ["device"]},
    "board":  {"target": "fit/bank/build_candidate_board.py", "video_flag": "--video",
               "accepts": ["device", "max_tau", "max_view_angle_deg", "min_coverage", "center_box"]},
    "loop":   {"target": "module:lbm.loop",                  "video_flag": "--video",
               "accepts": ["device", "start_mode", "max_tau", "max_view_angle_deg",
                           "min_coverage", "center_box"]},
    "decode": {"target": "decode/build_poses.py",            "video_flag": "--video",
               "accepts": ["start_mode", "aim_anchor", "aim_ramp_frames"]},
    "emit":   {"target": "decode/emit.py",                   "video_flag": "--video",
               "accepts": ["start_mode", "aim_anchor", "aim_ramp_frames"]},
    "verify": {"target": "verify.py",                        "video_flag": "--video",
               "accepts": ["device", "start_mode", "aim_anchor", "aim_ramp_frames", "center_box"]},
    # --- 뱅크 경로 (VLM 루프 없음) ---
    # dynseg 만 다른 sub-repo 에 있다. cwd 가 CINEMATRAJ_ROOT 라 **절대경로**로 준다.
    "dynseg": {"target": path.normpath(path.join(
                   path.dirname(path.abspath(__file__)), "..", "..", "..", "..",
                   "video_generation", "scripts", "sam3_seg_instances.py")),
               "video_flag": "--videos", "accepts": []},
    "bank":     {"target": "fit/bank/sample_camera_bank.py",  "video_flag": "--video",
                 "accepts": ["device"]},
    "fit":      {"target": "fit/bank/fit_hole_ladder.py",     "video_flag": "--video",
                 "accepts": ["device"]},
    "bankemit": {"target": "fit/bank/emit_bank.py",           "video_flag": "--video",
                 "accepts": []},
    "bankvid":  {"target": "viz/render_bank_videos.py",  "video_flag": "--video",
                 "accepts": ["device"]},
}
ALIASES = {"cloud": "board", "poses": "decode"}          # 계획서에서 쓰던 이름
ALL_STAGES = ["graph", "pcd", "board", "loop", "decode", "emit", "verify"]
BANK_STAGES = ["graph", "pcd", "bank", "fit", "bankemit", "bankvid"]


def parse_scalar(text: str):
    """`--set` 의 오른쪽을 JSON 으로 먼저 읽고, 안 되면 문자열 그대로."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def resolve_stages(text: str):
    """`all` / 쉼표 목록 / 별칭 → 정규화된 stage 리스트 (중복 제거, 순서 유지)."""
    if text.strip() == "all":
        return list(ALL_STAGES)
    if text.strip() == "bank_all":
        return list(BANK_STAGES)
    picked = []
    for raw in text.split(","):
        name = ALIASES.get(raw.strip(), raw.strip())
        assert name in STAGES, f"모르는 stage: {raw} (가능: {', '.join(STAGES)}, all)"
        if name not in picked:
            picked.append(name)
    return picked


def flag_tokens(flags: dict):
    """설정 dict → argparse 토큰. bool 은 집 스타일 `--flag`/`--no_flag` 쌍으로 나간다."""
    tokens = []
    for key in sorted(flags):
        value = flags[key]
        if value is None:                       # None = 스크립트 기본값에 맡긴다
            continue
        if isinstance(value, bool):
            tokens.append(f"--{key}" if value else f"--no_{key}")
        elif isinstance(value, (list, tuple)):
            tokens.append(f"--{key}")
            tokens.extend(str(item) for item in value)
        else:
            tokens.extend([f"--{key}", str(value)])
    return tokens


def build_command(python: str, stage: str, videos: list, config: dict):
    """한 stage 의 실행 명령. shared 는 `accepts` 에 있는 키만 넘어간다."""
    spec = STAGES[stage]
    flags = dict(config.get("shared", {}))
    flags = {key: flags[key] for key in spec["accepts"] if key in flags}
    flags.update(config.get("stages", {}).get(stage, {}))     # stage 쪽이 shared 를 이긴다

    target = spec["target"]
    head = ["-m", target[len("module:"):]] if target.startswith("module:") else [target]
    return [python, *head, spec["video_flag"], *videos, *flag_tokens(flags)]


def preflight(python: str, stage: str, env: dict):
    """`--help` 로 import 만 찔러본다. 실패 사유(stderr 끝 줄)를 돌려준다."""
    spec = STAGES[stage]
    target = spec["target"]
    head = ["-m", target[len("module:"):]] if target.startswith("module:") else [target]
    done = subprocess.run([python, *head, "--help"], cwd=CINEMATRAJ_ROOT, env=env,
                          capture_output=True, text=True)
    if done.returncode == 0:
        return None
    tail = [line for line in done.stderr.strip().splitlines() if line.strip()]
    return tail[-1] if tail else f"returncode {done.returncode}"


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT, type=str)
    parser.add_argument("--stage", default="all", type=str)          # all | 쉼표 목록
    parser.add_argument("--videos", nargs="*", default=None)         # None = config 의 videos
    parser.add_argument("--python", default=None, type=str)          # None = config 의 python
    parser.add_argument("--cuda", default=None, type=str)            # CUDA_VISIBLE_DEVICES (0~5 만)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="stage.key=value 또는 shared.key=value")
    parser.add_argument("--manifest", default=None, type=str)        # None = out/run_lbm_lite.json

    parser.add_argument("--preflight", action="store_true", default=True)
    parser.add_argument("--no_preflight", dest="preflight", action="store_false")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    parser.add_argument("--stop_on_error", action="store_true", default=True)
    parser.add_argument("--no_stop_on_error", dest="stop_on_error", action="store_false")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    assert config.get("format") == CONFIG_FORMAT, f"{args.config}: format 이 {CONFIG_FORMAT} 이 아니다"

    for item in args.overrides:                       # --set board.board_size=27
        assert "=" in item and "." in item.split("=")[0], f"--set 형식이 아니다: {item}"
        dotted, raw = item.split("=", 1)
        section, key = dotted.split(".", 1)
        if section == "shared":
            config.setdefault("shared", {})[key] = parse_scalar(raw)
        else:
            section = ALIASES.get(section, section)
            assert section in STAGES, f"--set 의 stage 를 모른다: {section}"
            config.setdefault("stages", {}).setdefault(section, {})[key] = parse_scalar(raw)

    stages = resolve_stages(args.stage)
    videos = args.videos if args.videos else config.get("videos", [])
    assert videos, "돌릴 영상이 없다 (--videos 또는 config 의 videos)"
    python = args.python or config.get("python") or sys.executable
    assert path.exists(python), f"python 이 없다: {python}"

    env = dict(environ)
    cuda = args.cuda if args.cuda is not None else config.get("cuda_visible_devices")
    if cuda is not None:
        assert all(part.strip() in "012345" and part.strip() for part in str(cuda).split(",")), \
            f"GPU 는 0~5 만 쓴다 (지정: {cuda})"
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)

    # 계획 먼저 찍는다 — 어떤 인자로 돌아가는지 보고 멈출 수 있게.
    plan = []
    for stage in stages:
        batched = STAGES[stage]["video_flag"] == "--videos"
        groups = [videos] if batched else [[video] for video in videos]
        for group in groups:
            plan.append((stage, group, build_command(python, stage, group, config)))
    print(f"config  {args.config}")
    print(f"python  {python}")
    print(f"cuda    {env.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"stages  {', '.join(stages)}   videos  {', '.join(videos)}")
    for stage, group, command in plan:
        print(f"  [{stage:6s}] {' '.join(command)}")

    if args.preflight and not args.dry_run:
        for stage in stages:
            problem = preflight(python, stage, env)
            if problem is None:
                continue
            env_name = config.get("env_name", "vista4d")
            print(f"\n[preflight] stage `{stage}` 가 이 python 에서 import 실패: {problem}")
            print(f"[preflight] 다른 env 에서 돌려라:\n"
                  f"  conda run -n {env_name} python run_lbm_lite.py --stage {args.stage} "
                  f"--videos {' '.join(videos)}")
            return 1

    results = []
    for stage, group, command in plan:
        if args.dry_run:
            results.append({"stage": stage, "videos": group, "returncode": None,
                            "seconds": 0.0, "command": command})
            continue
        print(f"\n===== {stage}  {', '.join(group)}")
        started = time.time()
        done = subprocess.run(command, cwd=CINEMATRAJ_ROOT, env=env)
        elapsed = time.time() - started
        results.append({"stage": stage, "videos": group, "returncode": done.returncode,
                        "seconds": round(elapsed, 2), "command": command})
        if done.returncode != 0 and args.stop_on_error:
            print(f"\n[stop] {stage} ({', '.join(group)}) returncode {done.returncode} — 여기서 멈춘다")
            break

    manifest_path = args.manifest or path.join(CINEMATRAJ_ROOT, "out", "run_lbm_lite.json")
    makedirs(path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"format": "lbm_lite_run_v1", "config_path": args.config, "python": python,
                   "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                   "stages": stages, "videos": videos, "dry_run": args.dry_run,
                   "resolved_config": config, "results": results},
                  handle, ensure_ascii=False, indent=2)

    print(f"\n{'stage':8s} {'videos':28s} {'rc':>4s} {'sec':>9s}")
    for item in results:
        code = "-" if item["returncode"] is None else str(item["returncode"])
        print(f"{item['stage']:8s} {','.join(item['videos'])[:28]:28s} {code:>4s} {item['seconds']:9.2f}")
    failed = [item for item in results if item["returncode"] not in (0, None)]
    print(f"\n-> {manifest_path}")
    print(f"steps {len(results)}  failed {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
