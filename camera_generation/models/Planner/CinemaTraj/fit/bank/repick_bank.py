"""이미 구운 뱅크의 `picked` 열만 **다시 달고** emit 을 다시 돌린다 (재굽기 없음).

왜 이게 가능한가: `fit_hole_ladder.py` 의 pick 은 사다리가 다 끝난 뒤의 **순수 후처리**다.
행을 지우지 않고(`bank.json` 은 재고 목록), `poses.npz` 에는 고르지 않은 행의 pose 까지 전부
들어 있다 (`fit_hole_ladder.py:1557` 이 `rows` 전량을 저장한다). 그래서 선택 규칙을 바꾸려고
τ→fit 을 다시 돌릴 필요가 없다 — 씬당 30s 짜리 fit 대신 0.1s 짜리 relabel + emit 이면 된다
(10,346편 기준 6샤드로 14시간 → 30분).

D189 의 용건은 **anchor 당 1대**다 (사용자 확정 2026-09-14 "같은 scene 같은 anchor 안 곂치게").
d188 뱅크 1,252편 실측:

    씬당 1대 (`--pick_budget 1`)        0.783 대/씬   → 10,346편 환산 8,098 대
    anchor 당 1대 (`--per_anchor`)      0.935 대/씬   → 환산 9,734 대   (+19.5%)

씬당 solved anchor 분포는 0개 21.7% / 1개 63.0% / 2개 15.3% 다 (route 가 `--max_anchors 2` 라
3개 이상은 없다). 즉 늘어나는 몫은 전부 "anchor 2개가 다 풀린 15.3%" 에서 온다.

**`fit_hole_ladder.py` 는 안 건드린다** — 이 스크립트가 도는 동안에도 d188 이 굽고 있고, inproc
샤드는 execv 할 때마다 그 파일을 다시 읽는다. 규칙 자체는 `lbm/pick.py` 한 곳에 있으므로
fit 쪽 배선(`--pick_per_anchor`)은 굽기가 끝난 뒤에 같은 함수로 연결한다.

사용 예시:
  python fit/bank/repick_bank.py --config configs/bank/d188_dynpose100k_track_objcentric.json \
      --per_anchor --num_shards 6 --shard_id 0
"""
from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
from json import dump as json_dump, load as json_load
from os import path
from sys import modules, path as syspath
from time import time

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
syspath.insert(0, HERE)
syspath.insert(0, path.join(HERE, "exec"))

from lbm.pick import pick_rows                                       # noqa: E402


def load_stage(name):
    """`fit/bank/<name>.py` 를 모듈로 올린다. `run_bank.py` 의 inproc 경로와 같은 방식이다."""
    spec = spec_from_file_location(f"_repick_{name}",
                                   path.join(HERE, "fit", "bank", f"{name}.py"))
    mod = module_from_spec(spec)
    modules[f"_repick_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def rewrite_csv(csv_path, picked_of):
    """`bank.csv` 의 `picked` 열만 갈아끼운다. 다른 열은 **문자 단위로 그대로** 둔다.

    행을 다시 만들지 않는 이유: 이 writer 는 따옴표를 안 쓰고 배열 열을 `|` 로 이어 붙이므로,
    재작성하면 포맷이 조용히 달라질 수 있다. 열 하나만 바꾸면 그 위험이 없다.
    """
    if not path.isfile(csv_path):
        return 0
    with open(csv_path, encoding="utf-8") as file:
        lines = file.read().splitlines()
    if not lines:
        return 0
    header = lines[0].split(",")
    if "picked" not in header or "variant_id" not in header:
        return 0
    ip, iv = header.index("picked"), header.index("variant_id")
    out = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) != len(header):
            out.append(line)
            continue
        fields[ip] = picked_of.get(fields[iv], "")
        out.append(",".join(fields))
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("\n".join(out) + "\n")
    return len(out) - 1


def main(args):
    from run_bank import load_config                                 # noqa: PLC0415
    cfg = load_config(args.config)
    out_root = args.output_root or cfg["output_root"]
    if not path.isabs(out_root):
        out_root = path.join(HERE, out_root)
    bank_dir = args.bank_dir or cfg["bank_dir"]
    emit_args = list((cfg.get("emit") or {}).get("args", []))
    if args.picked_only and "--picked_only" not in emit_args:
        emit_args.append("--picked_only")
    # fit 단계 인자에서 **그대로 물려받는다**. 여기에 손으로 적으면 config 와 어긋나는 순간
    # "다시 고른 뱅크"가 "새로 구운 뱅크"와 다른 규칙을 쓰게 된다.
    fit_args = list((cfg.get("fit") or {}).get("args", []))

    def after(flag, default):
        if flag not in fit_args:
            return default
        i = fit_args.index(flag) + 1
        vals = []
        while i < len(fit_args) and not fit_args[i].startswith("--"):
            vals.append(fit_args[i])
            i += 1
        return tuple(vals)

    retry_status = after("--retry_status", ())
    retry_suspect = after("--retry_suspect", ())
    budget = args.pick_budget if args.pick_budget is not None \
        else int((after("--pick_budget", ("0",)) or ("0",))[0])

    videos = args.videos
    if args.videos_file:
        # 굽기가 도는 중에는 **현재 라운드 목록을 뺀 편만** 넘겨야 한다 — 샤드가 같은
        # `bank.json` 을 쓰는 중이면 읽기/쓰기가 겹친다. 그 목록을 파일로 받는다.
        with open(args.videos_file, encoding="utf-8") as file:
            videos = [line.strip() for line in file if line.strip()]
    elif videos == ["all"]:
        from glob import glob                                        # noqa: PLC0415
        videos = sorted(path.basename(path.dirname(path.dirname(p)))
                        for p in glob(path.join(out_root, "*", bank_dir, "bank.json")))
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_id]

    emit = load_stage("emit_bank") if not args.no_emit else None
    tally = {"ok": 0, "no_bank": 0, "emit_fail": 0, "skip_done": 0}
    picked_total, anchors_total, t0 = 0, 0, time()
    for n, video in enumerate(videos, 1):
        folder = path.join(out_root, video, bank_dir)
        bank_path = path.join(folder, "bank.json")
        if not path.isfile(bank_path):
            tally["no_bank"] += 1
            continue
        with open(bank_path, encoding="utf-8") as file:
            bank = json_load(file)
        rows = bank["variants"]
        before = sum(1 for r in rows if r.get("picked"))
        chosen = pick_rows(rows, budget, retry_status, retry_suspect,
                           per_anchor=args.per_anchor)
        if args.skip_done and before == len(chosen) and args.mark and \
                path.exists(path.join(folder, args.mark)):
            tally["skip_done"] += 1
            continue
        picked_total += len(chosen)
        anchors_total += len({r["anchor_id"] for r in chosen})
        if not args.dry_run:
            with open(bank_path, "w", encoding="utf-8") as file:
                json_dump(bank, file, ensure_ascii=False, indent=1)
            rewrite_csv(path.join(folder, "bank.csv"),
                        {r["variant_id"]: ("1" if r.get("picked") else "") for r in rows})
            if emit is not None:
                argv = ["--video", video, "--output_root", out_root,
                        "--bank_dir", bank_dir] + emit_args
                try:
                    emit.main(emit.build_parser().parse_args(argv))
                except Exception as err:                             # noqa: BLE001
                    tally["emit_fail"] += 1
                    print(f"  EMIT_FAIL {video}: {type(err).__name__} {err}", flush=True)
                    continue
                if args.mark:
                    open(path.join(folder, args.mark), "w").close()
        tally["ok"] += 1
        if n % 100 == 0:
            print(f"  [{args.shard_id}] {n}/{len(videos)}  고른 카메라 {picked_total}  "
                  f"{time() - t0:.0f}s", flush=True)

    done = max(tally["ok"], 1)
    print("\n=== 요약 ===")
    for key in ("ok", "no_bank", "emit_fail", "skip_done"):
        print(f"{key:12s} {tally[key]}")
    print(f"{'카메라':12s} {picked_total}   씬당 {picked_total / done:.3f}")
    print(f"{'anchor':12s} {anchors_total}   씬당 {anchors_total / done:.3f}")
    print(f"{'경과':12s} {time() - t0:.0f}s")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--videos", nargs="*", default=["all"])       # all = 뱅크 있는 전량
    parser.add_argument("--videos_file", default="", type=str)        # 한 줄 1편. --videos 보다 우선
    parser.add_argument("--output_root", default="", type=str)        # 빈 값 = config
    parser.add_argument("--bank_dir", default="", type=str)           # 빈 값 = config
    # 예산. 안 주면 config 의 `--pick_budget` 을 그대로 쓴다. `--per_anchor` 와 곱해져
    # "anchor 당 이 개수" 가 된다 (씬당이 아니다).
    parser.add_argument("--pick_budget", default=None, type=int)
    parser.add_argument("--per_anchor", action="store_true", default=False)
    parser.add_argument("--no_per_anchor", dest="per_anchor", action="store_false")
    parser.add_argument("--picked_only", action="store_true", default=True)
    parser.add_argument("--no_picked_only", dest="picked_only", action="store_false")
    parser.add_argument("--no_emit", action="store_true", default=False)   # 열만 바꾸고 끝
    parser.add_argument("--dry_run", action="store_true", default=False)   # 수율만 센다
    # 다시 고른 뱅크임을 표시하는 빈 파일. `--skip_done` 이 이걸 본다.
    parser.add_argument("--mark", default=".repicked", type=str)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    parser.add_argument("--num_shards", default=1, type=int)
    parser.add_argument("--shard_id", default=0, type=int)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
