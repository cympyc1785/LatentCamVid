"""Vista4D target camera 파이프라인의 **단계별 실측 소요시간** 하네스.

WHY: "카메라 1개 뽑는 데 얼마나 걸리냐"를 물어보면 지금까지는 샤드 로그의 START/FIT 타임스탬프를
분 단위로 빼서 어림잡을 수밖에 없었다. 그런데 그 로그에는 `graph`/`pcd`/`bank` 가 아예 안 찍힌다
(샤드 러너가 `fit` 부터 돌기 때문). 그래서 앞단이 공짜인지 아닌지를 아무도 모르는 상태였다.

이 스크립트는 뱅크 경로 전 단계(`graph -> pcd -> bank -> fit -> bankemit`)를 **격리된
`--output_root`** 로 처음부터 다시 돌려서 벽시계 시간을 잰다. 격리가 핵심이다 — 기존
`out/<video>/` 를 건드리면 이미 학습에 쓰고 있는 `scene_graph.json`/`cloud.npz`/`hole_bank_k6`
가 덮어써진다. 모든 stage 가 `--output_root` 를 받으므로 원본은 한 바이트도 안 변한다.

단계당 시간은 영상 길이가 아니라 **뱅크 variant 수**에 붙는다 (fit 이 variant 마다 τ 이분법을
돌린다). 그래서 표에 `s/camera` 열을 같이 낸다 — 이게 "카메라 1개 평균 시간"의 정직한 정의다.
`graph`/`pcd` 는 variant 수와 무관한 **고정비**라 별도로 표시한다.

사용 예시:
    python eval/time_vista_stages.py --videos couple-hug avocado-slice women-talk --cuda 6
    python eval/time_vista_stages.py --videos camel --cuda 6 --stages graph,pcd,bank
    python eval/time_vista_stages.py --videos camel --dry_run
"""
import json
import subprocess
import sys
from argparse import ArgumentParser
from os import environ, makedirs, path
from time import time

HERE = path.dirname(path.dirname(path.abspath(__file__)))
PY = sys.executable

# stage -> (실행 대상, 추가 인자).  `module:` 접두사면 `python -m`.
# run_lbm_lite.py 의 BANK_STAGES 와 같은 순서다. `bankvid`(프리뷰 영상)는 데이터셋 생성에
# 필수가 아니라 기본에서 뺐다 — 필요하면 --stages 로 직접 부르면 된다.
STAGES = {
    "graph":    ("fit/graph/build_scene_graph.py", ["--no_skip_done"]),
    "pcd":      ("module:lbm.cloud",             ["--no_skip_done"]),
    "bank":     ("fit/bank/sample_camera_bank.py", ["--no_preview"]),
    # D96. ease 기본값이 `smooth_kf` 로 바뀌었다. 여기 명시하는 이유는 샤드 러너와 같은
    # 인자로 재보아야 소요시간이 의미가 있어서다 (평활은 비용이 무시할 수준이지만, 인자가
    # 갈라져 있으면 나중에 어느 쪽이 실제 설정인지 알 수 없게 된다).
    "fit":      ("fit/bank/fit_hole_ladder.py",
                 ["--aim_keyframes", "6", "--keyframe_aim", "auto",
                  "--keyframe_ease", "smooth_kf", "--fixed_focal"]),
    "bankemit": ("fit/bank/emit_bank.py",         []),
    "bankvid":  ("viz/render_bank_videos.py", []),
}
DEFAULT_STAGES = ["graph", "pcd", "bank", "fit", "bankemit"]
# `bank`(sample_camera_bank) 는 **원본 뱅크** `out/<video>/bank/` 를 쓰고, `fit` 이 그걸 읽어
# (`fit_hole_ladder.py:721 source_bank="bank/bank.json"`) τ 사다리를 적합한 결과를
# `hole_bank_k6/` 에 쓴다. 두 폴더를 헷갈리면 fit 이 자기 출력을 입력으로 읽는다.
SOURCE_BANK_STAGES = {"bank"}                             # --bank_dir 가 항상 "bank"
FIT_BANK_STAGES = {"fit", "bankemit", "bankvid"}          # --bank_dir 가 args.bank_dir
DEVICE_STAGES = {"pcd", "bank", "fit", "bankvid"}         # --device 를 받는 단계


def build_command(stage, video, out_root, bank_dir):
    target, extra = STAGES[stage]
    if target.startswith("module:"):
        command = [PY, "-m", target.split(":", 1)[1]]
    else:
        command = [PY, target]
    command += ["--video", video, "--output_root", out_root] + list(extra)
    if stage in SOURCE_BANK_STAGES:
        command += ["--bank_dir", "bank"]
    if stage in FIT_BANK_STAGES:
        command += ["--bank_dir", bank_dir]
    if stage in DEVICE_STAGES:
        command += ["--device", "cuda"]
    return command


def n_variants(out_root, video, bank_dir):
    """뱅크가 실제로 몇 개의 카메라를 냈는지. 없으면 None."""
    p = path.join(out_root, video, bank_dir, "bank.json")
    if not path.exists(p):
        return None
    return len(json.load(open(p))["variants"])


def main():
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)          # 잴 영상들
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))  # 쉼표 목록
    parser.add_argument("--out_root", default="/data1/cympyc1785/LatentCamVid/tmp/vista_timing")     # 격리 출력 루트
    parser.add_argument("--bank_dir", default="hole_bank_k6")          # 뱅크 폴더 이름
    parser.add_argument("--cuda", default="0", type=str)               # CUDA_VISIBLE_DEVICES
    parser.add_argument("--log_dir", default="/data1/cympyc1785/LatentCamVid/tmp/vista_timing/logs")
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        assert s in STAGES, f"모르는 stage: {s} (가능: {', '.join(STAGES)})"
    makedirs(args.log_dir, exist_ok=True)

    env = dict(environ, CUDA_VISIBLE_DEVICES=args.cuda,
               PYTHONPATH=HERE + ":" + environ.get("PYTHONPATH", ""))

    print(f"videos   {', '.join(args.videos)}")
    print(f"stages   {', '.join(stages)}")
    print(f"out_root {args.out_root}   bank_dir {args.bank_dir}   cuda {args.cuda}\n")

    rows = []
    for video in args.videos:
        timings = {}
        for stage in stages:
            command = build_command(stage, video, args.out_root, args.bank_dir)
            if args.dry_run:
                print(f"  [{video} / {stage}] {' '.join(command)}")
                continue
            log = path.join(args.log_dir, f"{video}.{stage}.log")
            t0 = time()
            with open(log, "w") as handle:
                rc = subprocess.call(command, cwd=HERE, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT)
            dt = time() - t0
            timings[stage] = (dt, rc)
            print(f"  [{video:18s} {stage:9s}] {dt:8.1f}s  rc={rc}"
                  + ("" if rc == 0 else f"   -> {log}"))
            if rc != 0:
                break
        if not args.dry_run:
            rows.append((video, n_variants(args.out_root, video, args.bank_dir), timings))

    if args.dry_run:
        return

    # ---- 요약표 ----
    print(f"\n{'video':20s} {'cams':>5s} " + " ".join(f"{s:>9s}" for s in stages)
          + f" {'total':>9s} {'s/cam':>7s}")
    for video, n, timings in rows:
        total = sum(dt for dt, _ in timings.values())
        cells = " ".join(f"{timings[s][0]:9.1f}" if s in timings else f"{'-':>9s}"
                         for s in stages)
        per = f"{total / n:7.2f}" if n else f"{'-':>7s}"
        print(f"{video:20s} {n if n else '-':>5} {cells} {total:9.1f} {per}")

    # variant 수에 비례하는 단계와 고정비 단계를 갈라서 보여준다.
    ok = [(n, t) for _, n, t in rows if n]
    if len(ok) >= 2:
        print("\n[variant 수 대비]")
        for stage in stages:
            pairs = [(n, t[stage][0]) for n, t in ok if stage in t]
            if len(pairs) < 2:
                continue
            lo, hi = min(pairs), max(pairs)
            slope = (hi[1] - lo[1]) / (hi[0] - lo[0]) if hi[0] != lo[0] else 0.0
            print(f"  {stage:9s} {lo[0]:4d}cam {lo[1]:7.1f}s -> {hi[0]:4d}cam {hi[1]:7.1f}s"
                  f"   기울기 {slope:6.2f} s/cam   절편 {lo[1] - slope * lo[0]:7.1f}s")


if __name__ == "__main__":
    main()
