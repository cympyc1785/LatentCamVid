"""eval 폴더의 pred 궤적이 **텍스트대로 움직였는지**를 GT 궤적 대비로 재고 arm 간 격차로 정렬한다.

WHY: 캡션은 GT 궤적에서 뽑은 문장이므로 "텍스트를 따랐다" = "GT 궤적과 같은 방향·크기·회전으로
움직였다" 로 환원된다. 충돌률·subject_in_frame 은 결과물의 안전성/구도를 재는 지표라
"지시한 arc right 를 dolly in 으로 바꿔 냈다" 같은 **지시 불이행**을 잡지 못한다 (구도가
우연히 맞으면 통과한다). 그래서 궤적 자체를 GT 와 겹쳐 본다.

지표 (전부 frame0 앵커를 푼 rel pose `inv(P[0]) @ P[f]` 위에서):
  dir_err_deg  : 순변위 `t[-1]` 방향과 GT 순변위 방향의 각도.  "어디로 갔나"
  len_ratio    : 경로길이 `Σ|Δt|` / GT 경로길이.               "얼마나 갔나"
  rot_err_deg  : `R[-1]` 과 GT `R[-1]` 의 geodesic 각.          "어디를 보게 됐나"
  rot_mag_deg  : `R[-1]` 자체의 회전량 (GT 도 같이 낸다)
`--rank_pair OURS=THEIRS` 를 주면 `dir_err(THEIRS) - dir_err(OURS)` 로 내림차순 정렬해
**우리는 맞고 상대는 틀린** 엔트리를 위로 올린다. 데모 엔트리 고르는 용도.

순변위가 GT/pred 어느 쪽이든 `--static_u` 아래면 방향각이 정의되지 않으므로 `dir_err` 를
NaN 으로 두고 정렬에서 뺀다 (정지 지시 엔트리를 "방향 불일치"로 오독하지 않기 위함).

env: 아무거나 (json/numpy 만 쓴다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY eval/rank_traj_text_match.py \
        --eval_dir "d124=/.../20260904_185554_vista4d_d121_molmo2__epoch100__seed42" \
        --eval_dir "gd_style=results/20260906_d156_gendop_d121/eval_dir_gendop_rgbd_gdstyle_raw_slerp_noscale" \
        --rank_pair d124=gd_style --require_moving --top 20
"""
import json
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np

CORPUS_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121"


def load_poses(json_path: str):
    """transforms JSON -> (F,4,4) c2w. `frames` 는 프레임 순서로 들어 있다 (`file_path` 없음)."""
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)
    return np.asarray([f["transform_matrix"] for f in data["frames"]], dtype=np.float64)


def rel_of(poses):
    """frame0 앵커를 푼다. eval JSON 은 world pose `gt_cv[0] @ pred_rel` 이라 그냥 비교하면
    translation 이 `t_rel` 의 아핀함수라 크기·방향이 둘 다 섞인다."""
    return np.linalg.inv(poses[0])[None] @ poses


def geodesic_deg(rot_a, rot_b):
    cos = (np.trace(rot_a.T @ rot_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def metrics(rel_pred, rel_gt, static_u: float):
    """한 엔트리 한 arm 의 지표 dict."""
    t_p, t_g = rel_pred[:, :3, 3], rel_gt[:, :3, 3]
    net_p, net_g = t_p[-1], t_g[-1]
    len_p = float(np.linalg.norm(np.diff(t_p, axis=0), axis=1).sum())
    len_g = float(np.linalg.norm(np.diff(t_g, axis=0), axis=1).sum())
    n_p, n_g = float(np.linalg.norm(net_p)), float(np.linalg.norm(net_g))
    dir_err = float("nan")
    if n_p > static_u and n_g > static_u:
        cos = float(np.dot(net_p, net_g) / (n_p * n_g))
        dir_err = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return {"dir_err_deg": round(dir_err, 2) if dir_err == dir_err else float("nan"),
            "len_ratio": round(len_p / len_g, 4) if len_g > 1e-12 else float("nan"),
            "net_u": round(n_p, 4), "net_gt_u": round(n_g, 4),
            "path_u": round(len_p, 4), "path_gt_u": round(len_g, 4),
            "rot_err_deg": round(geodesic_deg(rel_pred[-1, :3, :3], rel_gt[-1, :3, :3]), 2),
            "rot_mag_deg": round(geodesic_deg(np.eye(3), rel_pred[-1, :3, :3]), 2),
            "rot_mag_gt_deg": round(geodesic_deg(np.eye(3), rel_gt[-1, :3, :3]), 2)}


def load_prompt_meta(corpus_root: str, dataset: str, scene: str):
    prompts_path = path.join(corpus_root, dataset, scene, "da3", "prompts.json")
    with open(prompts_path, encoding="utf-8") as file:
        return json.load(file)


def main(args):
    arms = []
    for spec in args.eval_dir:
        label, _, folder = spec.partition("=")
        assert folder, f"--eval_dir 은 LABEL=DIR 꼴이어야 한다: {spec}"
        arms.append((label, folder))
    ref_dir = args.ref_eval_dir or arms[0][1]

    names = sorted(path.basename(p)[:-len("_transforms_ref.json")]
                   for p in glob(path.join(ref_dir, "test", "*_transforms_ref.json")))
    prompt_cache, rows = {}, []
    for name in names:
        _, scene, entry = name.split("_", 1)[0], name.split("_")[1], name.split("_")[-1]
        scene = name[len(args.name_prefix) + 1:name.rfind("_")]
        entry = name[name.rfind("_") + 1:]
        if scene not in prompt_cache:
            prompt_cache[scene] = load_prompt_meta(args.corpus_root, args.name_prefix, scene)
        meta = prompt_cache[scene][entry]
        rel_gt = rel_of(load_poses(path.join(ref_dir, "test", f"{name}_transforms_ref.json")))
        row = {"scene": scene, "entry": entry, "name": name,
               "preset": meta.get("preset", "?"), "aim": meta.get("aim", "?"),
               "arms": {}}
        for label, folder in arms:
            pred_path = path.join(folder, "test", f"{name}_transforms_pred.json")
            if not path.exists(pred_path):
                row["arms"][label] = None
                continue
            row["arms"][label] = metrics(rel_of(load_poses(pred_path)), rel_gt, args.static_u)
        rows.append(row)

    # 정지 지시 엔트리는 방향각이 없다 — 움직이는 지시만 남긴다 (데모용).
    pool = rows
    if args.require_moving:
        pool = [r for r in rows if not r["preset"].startswith("static")]
    if args.rank_pair:
        ours, _, theirs = args.rank_pair.partition("=")
        assert theirs, "--rank_pair 는 OURS=THEIRS 꼴이어야 한다"

        def gap(row):
            a, b = row["arms"].get(ours), row["arms"].get(theirs)
            if not a or not b:
                return -1e9
            if a["dir_err_deg"] != a["dir_err_deg"] or b["dir_err_deg"] != b["dir_err_deg"]:
                return -1e9
            if a["dir_err_deg"] > args.ours_max_dir_deg:
                return -1e9
            if not (args.len_lo <= a["len_ratio"] <= args.len_hi):
                return -1e9
            return b["dir_err_deg"] - a["dir_err_deg"]

        pool = sorted(pool, key=gap, reverse=True)
        pool = [r for r in pool if gap(r) > -1e8]

    labels = [l for l, _ in arms]
    print(f"{'name':<28}{'preset':<22}{'aim':<9}" +
          "".join(f"{l + ' dir':>14}{l + ' len':>13}{l + ' rot':>13}" for l in labels))
    for row in pool[:args.top]:
        line = f"{row['name']:<28}{row['preset']:<22}{row['aim']:<9}"
        for label in labels:
            met = row["arms"].get(label)
            line += ("".join(f"{'-':>14}{'-':>13}{'-':>13}") if not met else
                     f"{met['dir_err_deg']:>14.1f}{met['len_ratio']:>13.3f}{met['rot_err_deg']:>13.1f}")
        print(line)

    # arm 요약 — 데모 고르기와 별개로 "평균적으로 누가 텍스트를 따르나"
    print(f"\n{'arm':<12}{'n':>6}{'dir_err med':>13}{'dir_err mean':>14}"
          f"{'len_ratio med':>15}{'rot_err med':>13}")
    for label in labels:
        met = [r["arms"][label] for r in rows if r["arms"].get(label)]
        dirs = np.asarray([m["dir_err_deg"] for m in met], dtype=np.float64)
        dirs = dirs[np.isfinite(dirs)]
        # 정지 지시는 GT 경로길이가 0 이라 `len_ratio` 가 NaN — nanmedian 이어야 표가 안 죽는다.
        print(f"{label:<12}{len(met):>6}{np.median(dirs):>13.2f}{np.mean(dirs):>14.2f}"
              f"{np.nanmedian([m['len_ratio'] for m in met]):>15.4f}"
              f"{np.median([m['rot_err_deg'] for m in met]):>13.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump({"format": "traj_text_match_v1", "ref_eval_dir": path.abspath(ref_dir),
                       "arms": dict(arms), "static_u": args.static_u, "rows": rows},
                      file, ensure_ascii=False, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", action="append", required=True)   # LABEL=DIR, 여러 번
    # GT(`_transforms_ref.json`) 를 읽을 폴더. 안 주면 첫 `--eval_dir`.
    parser.add_argument("--ref_eval_dir", default=None)
    parser.add_argument("--name_prefix", default="vista4d")             # dynpose 코퍼스면 `dynpose`
    parser.add_argument("--corpus_root", default=CORPUS_DEFAULT)
    # 순변위가 이 아래면 방향각을 NaN (정지 지시).
    parser.add_argument("--static_u", type=float, default=0.02)
    parser.add_argument("--rank_pair", default=None)                    # OURS=THEIRS
    parser.add_argument("--ours_max_dir_deg", type=float, default=30.0)
    parser.add_argument("--len_lo", type=float, default=0.5)            # 우리 경로길이비 하한
    parser.add_argument("--len_hi", type=float, default=2.0)
    parser.add_argument("--require_moving", action="store_true", default=False)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", default=None)
    main(parser.parse_args())
