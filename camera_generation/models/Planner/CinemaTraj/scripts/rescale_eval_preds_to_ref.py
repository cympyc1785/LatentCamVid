"""latentcam eval 폴더의 pred 궤적을 **ref 의 rmax 에 맞춰** 다시 쓴다 (모양만 남기는 대조군).

왜: `gendop_preds_to_eval_dir.py` 는 GenDoP 을 GT world 에 얹으면서 `rmax` 를 GT 에서 빌려온다
(게이지가 달라 그 방법밖에 없다). 그런데 CLaTr 은 스케일에 민감하다 — `trajectory_dataset.py:43`
이 `self.standardize = False` 로 못 박혀 있고 표준화 코드는 전부 주석 처리라, 궤적이 velocity
표현으로 **원 단위 그대로** 들어간다. 즉 rmax 를 GT 에서 받은 arm 과 스스로 크기를 예측해야 했던
arm 을 나란히 놓으면 **크기 축에서 한쪽만 정답을 미리 본 비교**가 된다.

그래서 같은 변환을 latentcam 자기 pred 에도 걸어 둔 arm 을 만든다. 세 열을 나란히 읽으면
축이 분리된다:
    k6 raw        모양 + 크기 둘 다 자기가 예측
    k6 rmaxGT     크기는 GT, 모양만 자기 것        <- GenDoP 과 같은 조건
    gendop rmaxGT 크기는 GT, 모양만 자기 것

**GT 가 정지(rmax 0)인 entry** 는 rescale 이 pred 를 항등으로 뭉갠다. 그건 "모델이 정지를
맞혔다"가 아니라 그냥 정보가 지워진 것이므로 `--skip_static_ref` (기본 on) 로 빼고 그 수를 센다.
전량이 필요하면 `--no_skip_static_ref`.

ref / caption 은 손대지 않고 복사한다 (`render_pred_depth_warp.py` 가 arm 끼리 ref 를
max|diff|<1e-5 로 대조하므로 바꾸면 안 된다).

env: 아무거나 (numpy 만 쓴다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
    $PY scripts/rescale_eval_preds_to_ref.py \
        --eval_dir /data1/.../eval_my/vista4d_pgt_k6__last__full500 \
        --out_dir  /data1/.../eval_my/vista4d_pgt_k6__last__full500_rmaxGT
"""
import json
import shutil
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_transforms(json_path: str):
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)
    return np.array([f["transform_matrix"] for f in data["frames"]], dtype=np.float64) @ GL2CV, data


def write_transforms(out_path: str, c2w_cv: np.ndarray, meta: dict):
    frames = [{**{k: v for k, v in meta["frames"][i].items() if k != "transform_matrix"},
               "transform_matrix": (m @ GL2CV).tolist()} for i, m in enumerate(c2w_cv)]
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump({**{k: v for k, v in meta.items() if k != "frames"}, "frames": frames},
                  file, ensure_ascii=False, indent=1)


def main(args):
    test_out = path.join(args.out_dir, "test")
    makedirs(test_out, exist_ok=True)
    refs = sorted(glob(path.join(args.eval_dir, "test", "*_transforms_ref.json")))
    assert refs, f"{args.eval_dir}/test 에 ref 가 없다"

    names, n_static, gains = [], 0, []
    for ref_path in refs:
        name = path.basename(ref_path)[:-len("_transforms_ref.json")]
        gt_cv, _ = load_transforms(ref_path)
        pred_cv, meta = load_transforms(path.join(args.eval_dir, "test",
                                                  f"{name}_transforms_pred.json"))
        gt_rel = np.linalg.inv(gt_cv[0])[None] @ gt_cv
        pred_rel = np.linalg.inv(pred_cv[0])[None] @ pred_cv
        r_gt = float(np.linalg.norm(gt_rel[:, :3, 3], axis=-1).max())
        r_pred = float(np.linalg.norm(pred_rel[:, :3, 3], axis=-1).max())
        if (r_gt < 1e-9 and args.skip_static_ref) or r_pred < 1e-9:
            n_static += 1
            continue
        pred_rel[:, :3, 3] *= r_gt / r_pred
        write_transforms(path.join(test_out, f"{name}_transforms_pred.json"),
                         gt_cv[0][None] @ pred_rel, meta)
        shutil.copyfile(ref_path, path.join(test_out, f"{name}_transforms_ref.json"))
        shutil.copyfile(path.join(args.eval_dir, "test", f"{name}_caption.json"),
                        path.join(test_out, f"{name}_caption.json"))
        names.append(name)
        gains.append(r_gt / r_pred)

    with open(path.join(args.out_dir, "test_valid.txt"), "w", encoding="utf-8") as file:
        file.write("\n".join(sorted(names)) + "\n")

    g = np.array(gains)
    print(f"{'in':14s}{args.eval_dir}")
    print(f"{'entries':14s}{len(refs)}")
    print(f"{'written':14s}{len(names)}")
    print(f"{'skipped':14s}{n_static}   (ref 또는 pred 의 rmax 가 0)")
    if len(g):
        print(f"\n{'rmax gain (ref/pred)':24s}median {np.median(g):.4f}  "
              f"min {g.min():.4f}  max {g.max():.4f}")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_static_ref", action="store_true", default=True)
    parser.add_argument("--no_skip_static_ref", dest="skip_static_ref", action="store_false")
    main(parser.parse_args())
