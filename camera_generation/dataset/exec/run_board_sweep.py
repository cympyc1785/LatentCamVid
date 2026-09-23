"""TRUMANS `Recordings_blend/` 전량에 `trumans_first_pose_board.py` 를 돌리는 드라이버.

왜 따로 두나: blend 하나 여는 데 ~4 s 라서 recording 당 Blender 를 **한 번만** 띄워야 한다.
그런데 편마다 프레임 수가 달라 chunk start 목록을 밖에서 만들 수 없다 — 그래서 board 쪽에
`--chunk_stride` 를 두고, 여기서는 **어떤 blend 를 돌릴지**와 **실패해도 계속 갈지**만 본다.

디렉토리 나열이 사소하지 않다. `Recordings_blend/` 72 항목 중:
  - `- 副本` 5개 = 중복본. 이름만 다르고 내용이 같아 board 가 두 번 나온다 -> 제외.
  - `<id>/<id>.blend` 규칙을 안 지키는 3개 (`*_lph_ng` 접미사 2개, `fancy/`) -> 디렉토리 안의
    유일한 `.blend` 를 집는다. 이름으로만 찾으면 조용히 3편이 빠진다.
  - `a3fcee3e-…/` 는 **빈 디렉토리** -> 제외. 남는 게 66편.

출력은 `<out>/<recording_id>/board.json`. `--skip_done` 이면 board.json 이 있는 편을 건너뛴다
(중간에 죽어도 이어서 돌리라고). 실패는 죽이지 않고 `<out>/sweep.json` 에 stderr 꼬리와 함께
모은다 — 66편 중 1편이 이상해서 65편을 다시 돌리는 일이 없도록.

사용 예시:
  python exec/run_board_sweep.py --out out/sweep66 \
      --board_args "--chunk_stride 145 --frame_step 3 --lens 18 --min_clearance 0.3 \
                    --min_crop_keep 0.5 --anchor_frame start --gate_frames anchor \
                    --max_render 12 --render_select diverse"
"""
from argparse import ArgumentParser
from json import dump, load
from os import path, makedirs, listdir
from shlex import split
from subprocess import run, PIPE, STDOUT
import time

BLENDER = "/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender"
ROOT = "/data1/cympyc1785/data/trumans/Data_release/Recordings_blend"
# R6(2026-09-22) 재분류에서 드라이버는 `exec/` 로, board 스크립트는 `fit/bank/` 로 **갈라졌다.**
# 여기는 `dirname(__file__)` 옆을 보고 있었으므로 그때부터 안 돌았다 (R8 스모크에서 드러남).
# `fit/bank/` 를 직접 가리킨다 — 옆이 아니라 형제 폴더다.
SCRIPT = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                   "fit", "bank", "trumans_first_pose_board.py")


def find_blends(root, limit=0):
    """`<dir>/*.blend` 를 recording 하나로 본다. 위 docstring 의 예외 3종을 여기서 흡수한다."""
    found = []
    for name in sorted(listdir(root)):
        directory = path.join(root, name)
        if not path.isdir(directory) or "副本" in name:
            continue
        blends = sorted(f for f in listdir(directory) if f.endswith(".blend"))
        if not blends:
            continue                       # 빈 디렉토리 (a3fcee3e-…)
        # 여러 개면 디렉토리 이름과 같은 것을 우선. 규칙 밖이면 첫 번째.
        pick = f"{name}.blend" if f"{name}.blend" in blends else blends[0]
        found.append((name, path.join(directory, pick)))
    return found[:limit] if limit > 0 else found


def main(args):
    makedirs(args.out, exist_ok=True)
    blends = find_blends(args.root, args.limit)
    extra = split(args.board_args)
    rows, started = [], time.time()

    for i, (name, blend) in enumerate(blends):
        out_dir = path.join(args.out, name)
        board = path.join(out_dir, "board.json")
        if args.skip_done and path.exists(board):
            rows.append({"recording": name, "status": "skip"})
            print(f"[{i + 1}/{len(blends)}] skip {name}", flush=True)
            continue
        cmd = [BLENDER, "-b", blend, "--python", SCRIPT, "--", "--out", out_dir] + extra
        t0 = time.time()
        proc = run(cmd, stdout=PIPE, stderr=STDOUT, text=True)
        took = time.time() - t0
        row = {"recording": name, "blend": blend, "time_s": took,
               "status": "ok" if proc.returncode == 0 and path.exists(board) else "fail"}
        if row["status"] == "ok":
            with open(board) as handle:
                data = load(handle)
            row["frame_range"] = data["frame_range"]
            row["n_chunks"] = len(data["chunks"])
            row["usable"] = [sum(1 for c in ch["candidates"] if c["usable"])
                             for ch in data["chunks"]]
            row["n_rendered"] = sum(ch.get("n_rendered", 0) for ch in data["chunks"])
        else:
            # 로그 전체는 편당 수백 줄이라 꼬리만. 원인은 거의 항상 마지막 traceback 에 있다.
            row["returncode"] = proc.returncode
            row["tail"] = "\n".join(proc.stdout.strip().splitlines()[-25:])
        rows.append(row)
        with open(path.join(args.out, "sweep.json"), "w") as handle:
            dump({"root": args.root, "board_args": args.board_args, "rows": rows},
                 handle, ensure_ascii=False, indent=1)
        total = sum(row.get("usable", []))
        print(f"[{i + 1}/{len(blends)}] {row['status']:<4} {name}  {took:6.1f}s  "
              f"chunk {row.get('n_chunks', 0):>3}  usable {total:>5}  "
              f"render {row.get('n_rendered', 0):>4}", flush=True)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{'recording':<40}{'chunks':>7}{'usable':>8}{'render':>8}{'sec':>8}")
    for r in ok:
        print(f"{r['recording']:<40}{r['n_chunks']:>7}{sum(r['usable']):>8}"
              f"{r['n_rendered']:>8}{r['time_s']:>8.1f}")
    print(f"\nok {len(ok)} / fail {sum(1 for r in rows if r['status'] == 'fail')} / "
          f"skip {sum(1 for r in rows if r['status'] == 'skip')}  "
          f"chunk {sum(r['n_chunks'] for r in ok)}  usable {sum(sum(r['usable']) for r in ok)}  "
          f"render {sum(r['n_rendered'] for r in ok)}  총 {time.time() - started:.0f}s")
    for r in rows:
        if r["status"] == "fail":
            print(f"\n--- FAIL {r['recording']} (rc {r['returncode']})\n{r['tail']}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", default=ROOT)                  # Recordings_blend
    parser.add_argument("--out", required=True)                  # <out>/<recording>/board.json
    parser.add_argument("--board_args", default="")              # board 스크립트로 그대로 넘길 인자
    parser.add_argument("--limit", default=0, type=int)          # 0 = 전량. 스모크용
    parser.add_argument("--skip_done", action="store_true")      # board.json 있으면 건너뛰기
    main(parser.parse_args())
