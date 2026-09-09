"""이미 낸 `subject_in_frame.json` 을 **entry 부분집합**으로 다시 잘라 표만 뽑는다.

WHY: high-motion 부분집합 평가는 렌더를 다시 할 이유가 없다 — `eval_subject_in_frame.py` 가
행마다 프레임별 subject 중심(`centers`)까지 실어 두었으므로, split 파일로 행을 고르고
`in_frame_at()` 을 다시 걸면 `center_box` sweep 까지 그대로 나온다. 새 GPU 시간 0.

split 은 `<dataset>/<scene>/<idx>` 한 줄씩 — `rank_subject_motion.py` 가 뱉는 그 포맷이다.
`--group_split LABEL=FILE` 를 여러 번 주면 그룹별로 표를 따로 낸다 (held-out vs in-train 처럼
**섞으면 안 되는 부분집합**을 한 실행에서 보기 위함).

split 파일을 손으로 만들 이유가 없는 두 가지 흔한 갈래는 플래그로 둔다:
`--group_scene` 은 씬별로, `--group_meta aim` 은 코퍼스 `prompts.json` 의 필드값별로
(`aim=free` = free-moving / `aim=look_at` = object-centric) 그룹을 만든다.
`--json` 은 여러 번 줄 수 있다 — arm 을 나눠 돌린 실행 결과를 합쳐 한 표로 본다
(`(scene, entry, arm)` 중복은 **먼저 준 파일**을 남기고 개수를 보고한다).

env: 아무거나 (json/numpy 만 쓴다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY scripts/slice_subject_in_frame.py \
        --json results/20260901_gendop_dynpose_val/subject_in_frame.json \
        --group_split highmotion=results/20260901_highmotion/seg_dynpose_val_hm.txt
"""
import json
from argparse import ArgumentParser
from os import path

import numpy as np


def in_frame_at(centers, center_box: float):
    """`eval_subject_in_frame.in_frame_at` 과 같은 규칙. `None` = subject 픽셀 0 = 실패."""
    lo, hi = (1 - center_box) / 2, 1 - (1 - center_box) / 2
    hit = [c is not None and lo <= c[0] <= hi and lo <= c[1] <= hi for c in centers]
    return float(np.mean(hit)) if hit else float("nan")


def read_split(split_path: str):
    """`<dataset>/<scene>/<idx>` -> `(scene, idx)` 집합. dataset 은 행에 없어서 버린다."""
    with open(split_path, encoding="utf-8") as file:
        return {(l.split("/")[1], l.split("/")[2]) for l in
                (line.strip() for line in file) if l}


def read_meta_field(corpus_root: str, dataset: str, scenes, field: str):
    """`(scene, entry) -> prompts.json[entry][field]`.

    뱅크 변이의 성격(`aim`, `preset`, `anchor_label`)은 eval 폴더에 없고 코퍼스 export 에만
    있다. split 파일을 만들지 않고 그 값으로 바로 그룹을 가른다.
    """
    out = {}
    for scene in sorted(scenes):
        prompts_path = path.join(corpus_root, dataset, scene, "da3", "prompts.json")
        with open(prompts_path, encoding="utf-8") as file:
            prompts = json.load(file)
        for entry, meta in prompts.items():
            if field not in meta:
                raise SystemExit(f"{prompts_path} 의 entry {entry} 에 `{field}` 가 없다 "
                                 f"— 있는 필드: {sorted(meta)}")
            out[(scene, entry)] = str(meta[field])
    return out


def table(rows, arms, boxes):
    """arm × box 평균표 + arm 요약."""
    out = {}
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        out[arm] = {"n": len(sub),
                    "sweep": {f"{b:.2f}": float(np.mean([in_frame_at(r["centers"], b) for r in sub]))
                              for b in boxes} if sub else {},
                    "hole_fraction": float(np.mean([r["hole_fraction"] for r in sub])) if sub else float("nan"),
                    "subject_pixel_coverage": float(np.mean([r["subject_pixel_coverage"] for r in sub])) if sub else float("nan"),
                    "subject_zero_frames": float(np.mean([r["subject_zero_frames"] for r in sub])) if sub else float("nan")}
    return out


def main(args):
    # 여러 실행을 합친다. arm 을 나눠 돌렸어도 `gt` 는 양쪽에 들어 있어 그대로 합치면
    # 같은 (scene, entry, arm) 이 두 번 세어진다 — 먼저 준 파일을 남긴다.
    rows, arms, seen, dup = [], [], set(), 0
    for json_path in args.json:
        with open(json_path, encoding="utf-8") as file:
            data = json.load(file)
        if "centers" not in data["rows"][0]:
            raise SystemExit(f"{json_path} 에 `centers` 가 없다 — sweep 지원 이전 실행이라 재렌더가 필요하다")
        for arm in list(data.get("arms") or []) + sorted({r["arm"] for r in data["rows"]}):
            if arm not in arms:
                arms.append(arm)
        for row in data["rows"]:
            key = (row["scene"], row["entry"], row["arm"])
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            rows.append(row)
    if dup:
        print(f"[merge] 중복 행 {dup} 개는 먼저 준 --json 것을 남겼다 (보통 gt 열)")
    # `data["arms"]` 는 `--eval_dir` 로 준 arm 만 담고 있어 `gt` 가 빠진다 (별도 열로 붙는다).
    # 행에 실제로 있는 arm 을 합쳐야 gt 대조군이 표에서 사라지지 않는다.
    boxes = [round(b, 2) for b in
             np.arange(args.sweep_hi, args.sweep_lo - 1e-9, -args.sweep_step)]

    groups = [("all", None)]
    for spec in args.group_split or []:
        label, split_path = spec.split("=", 1)
        groups.append((label, read_split(split_path)))
    keys_all = {(r["scene"], r["entry"]) for r in rows}
    if args.group_scene:
        for scene in sorted({s for s, _ in keys_all}):
            groups.append((f"scene={scene}", {k for k in keys_all if k[0] == scene}))
    if args.group_meta:
        field = read_meta_field(args.corpus_root, args.dataset,
                                {s for s, _ in keys_all}, args.group_meta)
        missing = keys_all - set(field)
        if missing:
            raise SystemExit(f"prompts.json 에 없는 entry {len(missing)} 개 (예: {sorted(missing)[:3]})")
        for value in sorted({field[k] for k in keys_all}):
            groups.append((f"{args.group_meta}={value}",
                           {k for k in keys_all if field[k] == value}))

    result = {"format": "subject_in_frame_slice_v1",
              "json": [path.abspath(p) for p in args.json],
              "arms": arms, "boxes": boxes, "groups": {}}
    for label, keys in groups:
        sub = rows if keys is None else [r for r in rows if (r["scene"], r["entry"]) in keys]
        n_entry = len({(r["scene"], r["entry"]) for r in sub})
        result["groups"][label] = {"n_entries": n_entry, "arms": table(sub, arms, boxes)}

        print(f"\n=== {label}  (entry {n_entry})")
        print(f"{'box':>6}" + "".join(f"{a:>12}" for a in arms))
        for b in boxes:
            print(f"{b:>6.2f}" + "".join(
                f"{result['groups'][label]['arms'][a]['sweep'].get(f'{b:.2f}', float('nan')):>12.4f}"
                for a in arms))
        print(f"{'hole':>6}" + "".join(
            f"{result['groups'][label]['arms'][a]['hole_fraction']:>12.4f}" for a in arms))
        print(f"{'cov':>6}" + "".join(
            f"{result['groups'][label]['arms'][a]['subject_pixel_coverage']:>12.4f}" for a in arms))
        print(f"{'zero':>6}" + "".join(
            f"{result['groups'][label]['arms'][a]['subject_zero_frames']:>12.2f}" for a in arms))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    # eval_subject_in_frame.py 출력. 여러 번 주면 합쳐서 한 표로 (arm 을 나눠 돌린 경우).
    parser.add_argument("--json", action="append", required=True)
    # `LABEL=split.txt` 를 여러 번. 안 주면 전체(`all`) 하나만 낸다.
    parser.add_argument("--group_split", nargs="*", default=None)
    parser.add_argument("--group_scene", action="store_true", default=False)   # 씬별 표 추가
    # 코퍼스 `prompts.json` 의 필드값별 표 추가 (`aim` / `preset` / `anchor_label`).
    parser.add_argument("--group_meta", default=None)
    parser.add_argument("--corpus_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121")
    parser.add_argument("--dataset", default="vista4d")        # dynpose 코퍼스면 `dynpose`
    parser.add_argument("--sweep_hi", type=float, default=1.0)
    parser.add_argument("--sweep_lo", type=float, default=0.1)
    parser.add_argument("--sweep_step", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    main(parser.parse_args())
