"""latentcam eval 폴더(`<run>/test/*_transforms_{pred,ref}.json`) -> GenDoP 평가용 npz.

왜 필요한가: `gendop_release_eval.py` 는 캡션 왕복 지표를 **npz 디렉토리** 위에서 잰다
(`latentcam__<scene>__<name>.npz` 의 `c2w`). GenDoP arm 은 `gendop_release_infer.py` 가
그 npz 를 직접 뱉지만 우리 latentcam arm 은 eval JSON 만 남긴다. 그래서 arm 마다 다른
harness 로 재게 되고, 그러면 태거·리샘플러·fps 가 달라져 **수치를 나란히 못 놓는다**.
이 스크립트가 우리 arm 을 GenDoP arm 과 같은 입력 형태로 바꿔, 지표를 한 harness 로 통일한다.

**규약**: `transform_matrix` 는 OpenGL c2w 다 (`gendop_preds_to_eval_dir.py:70-75` 가
OpenCV c2w 에 `GL2CV=diag(1,-1,-1,1)` 를 곱해 쓴다). `gendop_release_eval.py` 의
`opencv_like_identity` 는 npz `c2w` 를 **OpenGL 그대로** 분절기에 넣으므로 여기서도
변환 없이 그대로 옮긴다. 즉 GenDoP npz 와 정확히 같은 게이지다.

`--which ref` 로 GT 를 뽑으면 **게이지 자기검증**이 된다 — GT 를 GT 태그에 맞춰 재면
F 가 1.0 근처여야 하고, 그렇지 않으면 어긋난 건 arm 이 아니라 harness 다.

env: 아무거나 (numpy 만).  예시:
    python scripts/eval_dir_to_gendop_npz.py \
        --eval_dir <run> --which pred --out <out>/npz_d123_da3
    python scripts/gendop_release_eval.py --pred_dir <out>/npz_d123_da3 \
        --n_poses 0 --fps 10 --align_len 49 \
        --latentcam_root /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121
"""
from argparse import ArgumentParser
from glob import glob
from json import load
from os import makedirs, path

import numpy as np


def entries(eval_dir, which, prefix):
    """`<eval_dir>/test/<prefix>_<scene>_<idx>_transforms_<which>.json` -> (scene, idx, path)."""
    suffix = f"_transforms_{which}.json"
    found = []
    for p in sorted(glob(path.join(eval_dir, "test", f"{prefix}_*{suffix}"))):
        stem = path.basename(p)[len(prefix) + 1:-len(suffix)]
        # scene 이름에 `_` 가 들어간다 (avocado-slice 는 아니지만 dynpose 에는 있다).
        # 마지막 `_` 뒤가 entry index 다.
        scene, _, index = stem.rpartition("_")
        found.append((scene, index, p))
    return found


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--eval_dir", required=True)          # latentcam eval_my/<run> 또는 gendop eval_dir
    ap.add_argument("--which", default="pred", choices=["pred", "ref"])   # ref = GT (게이지 자기검증)
    ap.add_argument("--prefix", default="vista4d")        # 파일명 접두사 (= --name_prefix)
    ap.add_argument("--kind", default="latentcam")        # npz 파일명 앞머리. tag 경로 규약을 정한다
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    makedirs(a.out, exist_ok=True)
    rows = entries(a.eval_dir, a.which, a.prefix)
    lengths, written = {}, 0
    for scene, index, p in rows:
        with open(p, encoding="utf-8") as file:
            data = load(file)
        c2w = np.asarray([f["transform_matrix"] for f in data["frames"]], dtype=np.float64)
        assert np.all(np.isfinite(c2w)), f"{p}: c2w 에 NaN/Inf"
        lengths[c2w.shape[0]] = lengths.get(c2w.shape[0], 0) + 1
        np.savez(path.join(a.out, f"{a.kind}__{scene}__{index}.npz"),
                 c2w=c2w, scale=1.0, degenerate=False, n_poses=int(c2w.shape[0]))
        written += 1

    print(f"{'eval_dir':<14}{a.eval_dir}")
    print(f"{'which':<14}{a.which}")
    print(f"{'entries':<14}{len(rows)}")
    print(f"{'written':<14}{written}")
    print(f"{'n_poses hist':<14}{dict(sorted(lengths.items()))}")
    print(f"{'out':<14}{a.out}")


if __name__ == "__main__":
    main()
