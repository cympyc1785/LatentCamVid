"""두 카메라 코퍼스의 `_tag.json` 을 같은 눈금으로 대조한다 (GT vs GT, 모델 없음).

WHY: "GenDoP 처럼 real cinematic 분포에서도 잘 나오게 하려면"의 첫 단계는 학습도 평가도 아니라
     **우리 뱅크 GT 와 DataDoP GT 가 어디서 갈리는지**를 재는 것이다. 갈리는 축이 이동량 레벨인지
     산포인지 정지 비율인지 어휘 점유인지에 따라 처방이 완전히 달라진다 (분모 재정의 / 필터 /
     τ 사다리 하한 / 캡션 포맷).

**비교가 성립하는 조건 — 어기면 차이가 코퍼스가 아니라 노브에서 나온다** (`prdc-not-comparable-
across-corpora` 와 같은 실패 모드):
  * 양쪽 `_tag.json` 의 `seg_kwargs` 가 완전히 같아야 한다 (fps 가 velocity 눈금에 그대로 곱해진다)
  * `num_poses` 가 같아야 한다 (`step_median` = 경로길이/(num_poses-1))
  * `gendop_root` 가 같아야 한다 (vendored/pipeline 이 `min_chunk_size` 10 vs 12 로 다르다)
이 세 가지를 읽어서 다르면 **표를 내기 전에 죽는다**. `--allow_mismatch` 로만 뚫린다.

스케일 규약: 양쪽 다 D=1 (리스케일 없음). DataDoP 는 MonST3R 게이지가 이미 shot 단위 정규화라
divisor 를 안 걸고(`datadop-world-scale-is-monst3r-gauge`), 우리 뱅크는 DA3 frame0 게이지다.
**두 게이지가 같다는 주장이 아니라, 레벨 차이 자체가 이 스크립트의 측정 대상**이다.

사용 예시:
    python scripts/compare_camera_distributions.py \
      --a_name "ours(d77 bank)" --a_glob "/data1/.../latentcam_da3_k6_d77/vista4d/*/da3/captions_gendop/*_tag.json" \
      --b_name "DataDoP GT"     --b_glob "out_captions/datadop_gt/*/*_tag.json" \
      --out results/20260830_gt_dist_compare
"""
from argparse import ArgumentParser
from collections import Counter
from glob import glob
from os import makedirs, path
import json

import numpy as np


# `segment_rigidbody_trajectories` 어휘: 27 translation 패턴 × 7 angular. static 판정은
# 라벨 문자열로 한다 — chunk 하나뿐이고 그게 static 이면 그 shot 은 통째로 정지다.
STATIC_MOVE = "static"
STATIC_ANG = "static"


def load_tags(pattern, divisor_subdir=None):
    """`divisor_subdir` 를 주면 두 길이 지표를 그 스칼라로 나눠 **학습 눈금**으로 바꾼다.

    WHY: raw world 단위 비교는 게이지가 서로 다르다 — 우리 뱅크는 DA3 frame0 게이지, DataDoP 은
    MonST3R 게이지(`datadop-world-scale-is-monst3r-gauge`). 그런데 **모델이 실제로 보는** 값은
    우리 쪽은 `avg_scale` 로 나눈 것이고(`scale_mode=avg_scale`) DataDoP 은 D=1 이라 안 나눈
    것이다. 그 눈금으로 맞춰야 "학습 분포가 얼마나 다른가"라는 질문에 답이 된다.
    태그는 `<scene>/da3/<caption_dir>/<name>_tag.json`, 분모는 `<scene>/da3/<subdir>/<name>.json`.
    """
    rows = []
    for file_path in sorted(glob(pattern)):
        with open(file_path, encoding="utf-8") as file:
            tag = json.load(file)
        if divisor_subdir:
            scalar_path = path.join(path.dirname(path.dirname(file_path)), divisor_subdir,
                                    f"{tag['name']}.json")
            if not path.exists(scalar_path):
                continue
            with open(scalar_path, encoding="utf-8") as file:
                divisor = float(json.load(file))
            tag["step_median"] /= divisor
            tag["total_translation"] /= divisor
            tag["divisor"] = divisor
        rows.append(tag)
    return rows


def gate_consistency(rows_a, rows_b, name_a, name_b, allow_mismatch):
    """seg_kwargs / num_poses / gendop_root 가 양쪽에서 같은지. 다르면 비교가 무의미하다."""
    def fingerprint(rows):
        return {
            "seg_kwargs": {json.dumps(r["seg_kwargs"], sort_keys=True) for r in rows},
            "num_poses": {r["num_poses"] for r in rows},
            "gendop_root": {r.get("gendop_root") for r in rows},
        }

    fa, fb = fingerprint(rows_a), fingerprint(rows_b)
    problems = []
    for key in ("seg_kwargs", "num_poses", "gendop_root"):
        if len(fa[key]) != 1 or len(fb[key]) != 1:
            problems.append(f"{key}: 코퍼스 내부가 이미 안 섞인다 "
                            f"({name_a}={sorted(fa[key])}, {name_b}={sorted(fb[key])})")
        elif fa[key] != fb[key]:
            problems.append(f"{key}: {name_a}={list(fa[key])[0]}  !=  {name_b}={list(fb[key])[0]}")
    if problems:
        head = "태거 설정이 어긋난다 — 이 차이는 코퍼스가 아니라 노브에서 나온다:\n  " \
               + "\n  ".join(problems)
        if not allow_mismatch:
            raise SystemExit(head + "\n(같은 --num_poses/--gendop_root/임계로 다시 태깅할 것. "
                                    "정말 무시하려면 --allow_mismatch)")
        print("!! " + head + "\n")
    return {"a": {k: sorted(v) for k, v in fa.items()},
            "b": {k: sorted(v) for k, v in fb.items()}}


def summarize(rows, floor=1e-6):
    """레벨 · 산포 · 정지 비율 · 어휘 점유.

    `floor` 아래는 **수치적 정지**로 따로 뺀다. 우리 뱅크에는 `*_hold` preset 과 τ 사다리
    아랫단이 만든 `step_median ~ 1e-19` 스파이크가 18% 있어서, 그대로 log10 sd 를 내면
    4.66 이 나온다 — 그건 산포가 아니라 그 스파이크다. floor 1e-6 은 실측으로 고른 값이다:
    우리 쪽은 1e-6~1e-4 구간에 엔트리가 **0개**(깨끗한 간극)이고, DataDoP 은 최솟값이
    1.29e-6 이라 한 편도 안 잘린다. 레벨/산포 통계는 floor 위에서만 낸다.
    """
    step = np.array([r["step_median"] for r in rows], dtype=np.float64)
    total = np.array([r["total_translation"] for r in rows], dtype=np.float64)
    n_chunks = np.array([len(r["chunks"]) for r in rows], dtype=np.float64)

    # 경로 길이 = step_median × (num_poses-1) 이 아니라, 프레임별 step 합이 정확하지만
    # tag 에는 median 만 있다. 여기서는 tag 가 실제로 담고 있는 두 양만 쓴다.
    def stats(values):
        moving = values[values > floor]
        if moving.size == 0:
            return {"n": int(values.size), "n_moving": 0, "below_floor_frac": 1.0}
        return {
            "n": int(values.size),
            "n_moving": int(moving.size),
            "below_floor_frac": float((values <= floor).mean()),
            "median": float(np.median(moving)),
            "p05": float(np.percentile(moving, 5)),
            "p95": float(np.percentile(moving, 95)),
            "mean": float(moving.mean()),
            # 산포는 log10 표준편차 — 레벨이 4배 다른 두 코퍼스를 선형 sd 로 비교하면
            # 레벨 차이가 산포 차이로 새어든다.
            "sd_log10": float(np.std(np.log10(moving))) if moving.size > 1 else float("nan"),
        }

    # 정지 비율 두 가지: ① shot 전체가 static 인 비율 ② static chunk 가 덮는 프레임 비율
    all_static, static_frames, total_frames = 0, 0, 0
    move_counter, ang_counter = Counter(), Counter()
    for r in rows:
        chunks = r["chunks"]
        span = max((c["end"] for c in chunks), default=0) + 1
        total_frames += span
        is_all = bool(chunks) and all(c["move"] == STATIC_MOVE and c["angular"] == STATIC_ANG
                                      for c in chunks)
        all_static += int(is_all)
        for c in chunks:
            length = c["end"] - c["start"] + 1
            if c["move"] == STATIC_MOVE and c["angular"] == STATIC_ANG:
                static_frames += length
            move_counter[c["move"]] += length
            ang_counter[c["angular"]] += length

    return {
        "step_median": stats(step),
        "total_translation": stats(total),
        "chunks_per_shot": stats(n_chunks),
        "all_static_frac": all_static / max(len(rows), 1),
        "static_frame_frac": static_frames / max(total_frames, 1),
        "move_vocab": move_counter,
        "ang_vocab": ang_counter,
        "n_shots": len(rows),
    }


def print_table(sa, sb, name_a, name_b):
    width = max(len(name_a), len(name_b), 12) + 2
    print(f"\n{'':<24}{name_a:<{width}}{name_b:<{width}}{'b/a':>10}")
    print("-" * (24 + 2 * width + 10))

    def row(label, va, vb, fmt="{:.4f}"):
        ratio = (vb / va) if (isinstance(va, float) and va) else float("nan")
        print(f"{label:<24}{fmt.format(va):<{width}}{fmt.format(vb):<{width}}{ratio:>10.2f}")

    print(f"{'n shots':<24}{sa['n_shots']:<{width}}{sb['n_shots']:<{width}}")
    for key in ("step_median", "total_translation", "chunks_per_shot"):
        print(f"\n[{key}]  (floor 위에서만 — 아래는 수치적 정지로 따로 센다)")
        row("  below floor frac", sa[key]["below_floor_frac"], sb[key]["below_floor_frac"])
        print(f"{'  n moving':<24}{sa[key]['n_moving']:<{width}}{sb[key]['n_moving']:<{width}}")
        for stat in ("median", "p05", "p95", "mean", "sd_log10"):
            row(f"  {stat}", sa[key][stat], sb[key][stat])
    print()
    # 라벨 기준 정지 — 위 floor 와 다른 양이다. translation 이 0 이어도 회전이 있으면
    # 태거는 static 으로 안 적는다 (우리 `*_hold` preset 이 정확히 그 경우).
    row("all-static shot frac", sa["all_static_frac"], sb["all_static_frac"])
    row("static frame frac", sa["static_frame_frac"], sb["static_frame_frac"])
    row("trans-frozen shot frac", sa["step_median"]["below_floor_frac"],
        sb["step_median"]["below_floor_frac"])

    print(f"\n[translation 어휘 점유 — 프레임 비중 상위 12]")
    keys = [k for k, _ in (sa["move_vocab"] + sb["move_vocab"]).most_common(12)]
    ta, tb = sum(sa["move_vocab"].values()), sum(sb["move_vocab"].values())
    print(f"{'':<24}{name_a:<{width}}{name_b:<{width}}")
    for k in keys:
        print(f"{k[:23]:<24}{sa['move_vocab'][k] / max(ta, 1):<{width}.4f}"
              f"{sb['move_vocab'][k] / max(tb, 1):<{width}.4f}")

    print(f"\n[angular 어휘 점유 — 프레임 비중]")
    keys = [k for k, _ in (sa["ang_vocab"] + sb["ang_vocab"]).most_common(8)]
    ta, tb = sum(sa["ang_vocab"].values()), sum(sb["ang_vocab"].values())
    for k in keys:
        print(f"{k[:23]:<24}{sa['ang_vocab'][k] / max(ta, 1):<{width}.4f}"
              f"{sb['ang_vocab'][k] / max(tb, 1):<{width}.4f}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a_glob", required=True)      # 코퍼스 A 의 `*_tag.json` 글롭
    parser.add_argument("--a_name", default="A")
    parser.add_argument("--b_glob", required=True)      # 코퍼스 B
    parser.add_argument("--b_name", default="B")
    parser.add_argument("--out", default=None)          # 요약 json + 히스토그램 png
    parser.add_argument("--allow_mismatch", action="store_true")   # 태거 설정 불일치 강행
    parser.add_argument("--floor", type=float, default=1e-6)       # 이 아래는 수치적 정지
    # 학습 눈금으로 맞추기: `<scene>/da3/<subdir>/<name>.json` 스칼라로 길이 지표를 나눈다.
    # 우리 뱅크는 `avg_scale_context_first_cam`, DataDoP 은 D=1 이라 안 준다.
    parser.add_argument("--a_divisor_subdir", default=None)
    parser.add_argument("--b_divisor_subdir", default=None)
    args = parser.parse_args()

    rows_a = load_tags(args.a_glob, args.a_divisor_subdir)
    rows_b = load_tags(args.b_glob, args.b_divisor_subdir)
    if not rows_a or not rows_b:
        raise SystemExit(f"tag 가 없다: {args.a_name}={len(rows_a)}, {args.b_name}={len(rows_b)}")
    fingerprints = gate_consistency(rows_a, rows_b, args.a_name, args.b_name,
                                    args.allow_mismatch)
    sa, sb = summarize(rows_a, args.floor), summarize(rows_b, args.floor)
    print_table(sa, sb, args.a_name, args.b_name)

    if args.out:
        makedirs(args.out, exist_ok=True)
        payload = {"a": {"name": args.a_name, "glob": args.a_glob,
                         **{k: (dict(v) if isinstance(v, Counter) else v) for k, v in sa.items()}},
                   "b": {"name": args.b_name, "glob": args.b_glob,
                         **{k: (dict(v) if isinstance(v, Counter) else v) for k, v in sb.items()}},
                   "floor": args.floor, "tagger_fingerprints": fingerprints,
                   "divisor_subdir": {"a": args.a_divisor_subdir, "b": args.b_divisor_subdir}}
        with open(path.join(args.out, "summary.json"), "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, key in zip(axes, ("step_median", "total_translation")):
            for rows, name in ((rows_a, args.a_name), (rows_b, args.b_name)):
                values = np.array([r[key] for r in rows], dtype=np.float64)
                values = values[values > args.floor]     # 수치적 정지 스파이크는 히스토그램에서도 뺀다
                ax.hist(np.log10(values), bins=60, alpha=0.5, density=True, label=name)
            ax.set_xlabel(f"log10({key})  [D=1, no rescale; floor {args.floor:g}]")
            ax.set_ylabel("density")
            ax.legend()
        fig.tight_layout()
        fig.savefig(path.join(args.out, "level_hist.png"), dpi=130)
        print(f"\n{'out':<24}{args.out}")


if __name__ == "__main__":
    main()
