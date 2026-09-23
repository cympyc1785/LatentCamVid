"""split 의 각 entry 를 **subject 가 얼마나 움직이나**로 정렬하고, 상위만 새 split 으로 뽑는다.

WHY: `subject_in_frame` 은 subject 가 정지해 있으면 거의 공짜로 높다 — frame0 이 소스 카메라라
구도가 이미 맞아 있고, 카메라만 얌전히 있으면 끝까지 맞다. 그래서 "정지 subject 가 섞인 평균"은
추종 능력을 못 잰다. 움직이는 subject 만 남기면 그 성분이 드러난다.

**정렬 기준은 각(角) 이동량** `ang = path_len_u / d_ref` 다. 절대 이동량(`path_len_u`)이 아니라:
subject 가 카메라에서 멀면 같은 미터를 움직여도 화면에서는 조금 움직인다. `subject_in_frame` 은
화면 좌표의 지표이므로 분모가 시청 거리여야 한다. `d_ref` 는 scene graph 의
`viewing_distance.d_ref` (소스 카메라 기준 대표 거리, 단위 u).

`center_drift_u / d_ref`(순 변위)도 같이 찍는다 — `path` 만 크고 `drift` 가 작으면 제자리
왕복(예: `swing`)이라 추종 난이도가 다르다. 컷은 `--metric` 으로 고른다.

subject 는 entry 마다 다르다 — `da3/prompts.json[<idx>]["variant_id"]` 앞머리
(`dyn_1__dolly_in__hole0.2` -> `dyn_1`). `eval_subject_in_frame.py:entry_subject` 와 같은 규칙.

env: 아무거나 (json/numpy 만 쓴다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY eval/rank_subject_motion.py \
        --split /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d77/seg_list_vista4d_train.txt \
        --corpus /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d77 \
        --cloud_root out --min_ang 1.0 --max_per_scene 12 \
        --out_split results/20260901_highmotion/seg_list_vista_highmotion.txt
"""
import json
from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, path

import numpy as np


def load_graphs(cloud_root: str, scenes):
    """scene -> scene_graph.json. 없는 scene 은 None 으로 남겨 호출부가 세게 한다."""
    graphs = {}
    for scene in scenes:
        graph_path = path.join(cloud_root, scene, "scene_graph.json")
        if not path.isfile(graph_path):
            graphs[scene] = None
            continue
        with open(graph_path, encoding="utf-8") as file:
            graphs[scene] = json.load(file)
    return graphs


def main(args):
    with open(args.split, encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    scenes = sorted({line.split("/")[1] for line in lines})
    graphs = load_graphs(args.cloud_root, scenes)

    prompts_cache, rows, skipped = {}, [], []
    for line in lines:
        dataset, scene, index = line.split("/")
        graph = graphs.get(scene)
        if graph is None:
            skipped.append(f"{line}: scene_graph 없음")
            continue
        if scene not in prompts_cache:
            with open(path.join(args.corpus, dataset, scene, "da3", "prompts.json"),
                      encoding="utf-8") as file:
                prompts_cache[scene] = json.load(file)
        entry = prompts_cache[scene].get(index)
        if entry is None:
            skipped.append(f"{line}: prompts 없음")
            continue
        subject_id = str(entry["variant_id"]).split("__")[0]
        node = next((n for n in graph["nodes"] if n["id"] == subject_id), None)
        if node is None:
            skipped.append(f"{line}: 노드 {subject_id} 없음")
            continue
        # 정적 노드는 프레임별 OBB 재적합 지터가 `path_len_u` 에 그대로 쌓인다 — avocado-slice
        # `stat_1:window` 가 path 6.97 / drift 0.24 다. 창문은 안 움직인다. `moving` 으로 먼저 거른다.
        if args.moving_only and not node.get("moving", False):
            skipped.append(f"{line}: {subject_id} moving=False")
            continue
        d_ref = max(float(node["viewing_distance"]["d_ref"]), 1e-6)
        rows.append({"line": line, "scene": scene, "entry": index,
                     "subject_id": subject_id, "label": node.get("label", ""),
                     "preset": str(entry["variant_id"]).split("__", 1)[-1],
                     "path_len_u": float(node.get("path_len_u", 0.0)),
                     "drift_u": float(node.get("center_drift_u", 0.0)),
                     "d_ref": d_ref,
                     "ang_path": float(node.get("path_len_u", 0.0)) / d_ref,
                     "ang_drift": float(node.get("center_drift_u", 0.0)) / d_ref})

    rows.sort(key=lambda r: -r[args.metric])
    values = np.array([r[args.metric] for r in rows]) if rows else np.zeros(0)

    # ── 컷 + scene 당 상한. 상한이 없으면 표본이 큰 scene 하나가 평균을 통째로 정한다.
    kept, per_scene = [], defaultdict(int)
    for row in rows:
        if row[args.metric] < args.min_ang:
            continue
        if args.max_per_scene and per_scene[row["scene"]] >= args.max_per_scene:
            continue
        per_scene[row["scene"]] += 1
        kept.append(row)
    kept.sort(key=lambda r: (r["scene"], int(r["entry"])))

    print(f"{'split':<14}{args.split}")
    print(f"{'entries':<14}{len(lines)}  (측정 {len(rows)}, skip {len(skipped)})")
    if len(values):
        q = np.quantile(values, [0.5, 0.7, 0.8, 0.9])
        print(f"{'quantile':<14}{args.metric}  p50 {q[0]:.3f}  p70 {q[1]:.3f}"
              f"  p80 {q[2]:.3f}  p90 {q[3]:.3f}  max {values.max():.3f}")
    print(f"{'kept':<14}{len(kept)}  (>= {args.min_ang}, scene 당 <= {args.max_per_scene})")

    print(f"\n{'scene':<20}{'n':>4}{'ang_path':>10}{'ang_drift':>11}{'subjects'}")
    for scene in sorted({r["scene"] for r in kept}):
        sub = [r for r in kept if r["scene"] == scene]
        subjects = sorted({f"{r['subject_id']}:{r['label']}" for r in sub})
        print(f"{scene:<20}{len(sub):>4}{np.mean([r['ang_path'] for r in sub]):>10.3f}"
              f"{np.mean([r['ang_drift'] for r in sub]):>11.3f} {', '.join(subjects)}")

    if args.out_split:
        makedirs(path.dirname(path.abspath(args.out_split)), exist_ok=True)
        with open(args.out_split, "w", encoding="utf-8") as file:
            file.write("".join(f"{r['line']}\n" for r in kept))
        with open(args.out_split.replace(".txt", "_motion.json"), "w", encoding="utf-8") as file:
            json.dump({"format": "subject_motion_v1", "split": args.split,
                       "metric": args.metric, "min_ang": args.min_ang,
                       "max_per_scene": args.max_per_scene,
                       "kept": kept, "all": rows, "skipped": skipped},
                      file, ensure_ascii=False, indent=1)
        print(f"\n-> {args.out_split}  ({len(kept)} 줄)")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)            # `<dataset>/<scene>/<idx>` 한 줄씩
    parser.add_argument("--corpus", required=True)           # prompts.json 이 있는 코퍼스 루트
    parser.add_argument("--cloud_root", required=True)       # scene_graph.json 이 있는 곳
    # 컷 기준. path = 궤적 길이(왕복 포함), drift = 순 변위 (제자리 왕복을 뺀다)
    parser.add_argument("--metric", default="ang_drift", choices=["ang_path", "ang_drift"])
    # 정적 노드 제외. path 지터가 큰 정적 노드가 상위를 채우므로 기본으로 켜 둔다.
    parser.add_argument("--moving_only", dest="moving_only", action="store_true", default=True)
    parser.add_argument("--no_moving_only", dest="moving_only", action="store_false")
    parser.add_argument("--min_ang", type=float, default=1.0)
    parser.add_argument("--max_per_scene", type=int, default=0)   # 0 = 무제한
    parser.add_argument("--out_split", default=None)
    main(parser.parse_args())
