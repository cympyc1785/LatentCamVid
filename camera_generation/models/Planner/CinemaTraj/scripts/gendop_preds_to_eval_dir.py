"""GenDoP 예측 npz -> latentcam eval 폴더 레이아웃 (`test/<name>_transforms_pred.json`).

왜: `render_pred_depth_warp.py` 는 arm 하나를 `--eval_dir LABEL=DIR` 로 받고, 그 DIR 의
`test/` 안에서 `_transforms_ref.json` / `_transforms_pred.json` / `_caption.json` 세 짝을
읽는다 (ref 는 arm 끼리 max|diff|<1e-5 로 대조해 split 어긋남을 잡는다). GenDoP 은
`latentcam__<scene>__<idx>.npz` 하나만 떨구므로 **그 레이아웃으로 갈아 끼우는 얇은 어댑터**만
있으면 렌더 스크립트는 한 줄도 안 고쳐도 된다. ref/caption 은 기준 eval 폴더에서 그대로
복사하고 pred 만 새로 쓴다.

═══ 좌표·스케일 정합 (이게 이 스크립트의 전부) ═══════════════════════════════════════
GenDoP 은 **자기 world, DataDoP 게이지, 30 pose** 로 낸다. GT/k6 는 **scene world, 49 pose**다.
그대로 겹치면 축·원점·스케일 세 군데가 다 어긋나므로 네 단계를 순서대로 건다:

1. `raw @ GL2CV`            GenDoP c2w -> OpenCV c2w. `GL2CV = diag(1,-1,-1,1)` 는 자기역원이라
                            JSON 으로 다시 쓸 때도 같은 행렬을 곱하면 된다.
2. rel-anchor               `rel = inv(P[0]) @ P` — GenDoP 의 절대 원점을 버린다. 어차피
                            텍스트만 받은 모델이라 절대 위치에는 의미가 없다.
3. 30 -> 49 리샘플          `--resample` 로 두 가지 중 고른다.
   `index_pick`(기본)       `np.rint(np.linspace(0, 29, 49))`. 보간하지 않는다 —
                            `gendop_release_eval.py` 가 지표를 낼 때 쓴 것과 같은 규칙이라
                            영상과 표가 같은 궤적을 본다. 0..29 가 전부 최소 1회 뽑히므로
                            이 단계는 rmax 를 안 바꾼다 (4번 순서와 무관). 대신 30 개 중
                            19 개가 **두 번 뽑혀** 계단이 생긴다 (같은 pose 가 연속 2프레임).
   `gendop_slerp`           **GenDoP 자신의 보간기** `core/utils.sample_from_dense_cameras`
                            (회전 SLERP + 이동 LERP). `eval.py:256-262` 가 30 pose 를 120
                            프레임으로 늘릴 때 부르는 바로 그 함수다. 계단이 사라지고
                            frame0/frame48 이 pose0/pose29 와 정확히 일치한다.
                            시간축은 `t = i/(n-1)` — 원본 `eval.py` 는 `i/120` 이라 마지막
                            pose 를 안 밟지만(궤적의 99.2% 만 씀), 우리는 GT 49프레임이 소스
                            클립 전체를 덮으므로 끝점을 포함해야 같은 구간을 비교한다.
4. rmax 정규화 -> GT world  `rmax = max_f |rel[f,:3,3]|` 를 GT 의 같은 양에 맞춘 뒤
                            `P_world = GT_c2w[0] @ rel`. **크기를 GT 에서 빌려온다**는 뜻이므로
                            이 영상은 "GenDoP 이 얼마나 크게 움직이나"가 아니라 **"모양이
                            맞나"**만 본다. 게이지가 다른 두 모델을 한 화면에 놓는 유일한 방법.

`degenerate=True` 인 entry 는 rmax 가 0 이라 3번에서 0 나눗셈이 된다 -> 스킵하고 표에 센다
(조용히 항등 궤적을 렌더하면 "GenDoP 이 정지 궤적을 냈다"로 오독된다).

env: `index_pick` 은 아무거나 (numpy 만 쓴다). `gendop_slerp` 은 `GenDoP`
     (`core/utils.py` 가 torch·trimesh·megfile 을 import 한다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
    $PY scripts/gendop_preds_to_eval_dir.py \
        --pred_dir results/20260828_gendop_latentcam_val/text_motion \
        --ref_eval_dir /data1/.../latentcam/eval_my/vista4d_pgt_k6__last__full500 \
        --out_dir results/20260828_gendop_latentcam_val/eval_dir_avocado \
        --video avocado-slice
"""
import json
import shutil
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])        # OpenGL c2w <-> OpenCV c2w (자기역원)


def load_transforms(json_path: str):
    """eval JSON -> (OpenCV c2w (T,4,4), meta dict). `render_pred_depth_warp.load_transforms` 와 동일."""
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)
    c2w = np.array([f["transform_matrix"] for f in data["frames"]], dtype=np.float64) @ GL2CV
    return c2w, data


def rel_anchor(c2w: np.ndarray):
    """frame0 을 항등으로 만든 상대 궤적."""
    return np.linalg.inv(c2w[0])[None] @ c2w


def gendop_resample(rel: np.ndarray, n: int, gendop_root: str):
    """GenDoP 자신의 보간기로 (T,4,4) -> (n,4,4). 회전 SLERP + 이동 LERP.

    한 번에 시각 하나씩(M=1) 부른다. `core/utils.sample_from_two_pose` 가 `fraction` 을
    (B,M) 모양 그대로 (B,M,4) 쿼터니언에 곱해서 M>1 이면 브로드캐스트가 깨지기 때문이고,
    원본 `eval.py:257-261` 도 같은 이유로 루프를 돈다. 그쪽과 같은 호출 형태를 유지한다.

    float32 로 넘긴다 — `is_valid_rotation_matrix` 가 `torch.ones_like(matrix)` 와 비교해
    float64 를 주면 dtype mismatch 로 죽는다 (원본 버그, 우회하지 말고 원본 dtype 을 따른다).

    rel 을 rigid 하게 왼쪽에서 곱한 것과 보간은 교환되므로(SLERP 은 좌불변, LERP 은 아핀)
    rel-anchor 전에 걸든 후에 걸든 결과가 같다. 순서를 index_pick 과 맞춰 둔다.
    """
    import sys                      # cv2 가 sys.path 를 갈아끼우므로 모듈째로 잡는다
    if gendop_root not in sys.path:
        sys.path.insert(0, gendop_root)
    import torch
    from core.utils import sample_from_dense_cameras

    dense = torch.tensor(rel[:, :3, :4].reshape(1, len(rel), 12), dtype=torch.float32)
    picks = [sample_from_dense_cameras(dense, torch.full((1, 1), i / (n - 1)))[0]
             for i in range(n)]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :4] = torch.cat(picks, 0).numpy().reshape(n, 3, 4).astype(np.float64)
    return out


def rmax_of(rel: np.ndarray):
    """상대 궤적의 최대 이동 반경 — 두 게이지를 맞추는 단일 스칼라."""
    return float(np.linalg.norm(rel[:, :3, 3], axis=-1).max())


def write_transforms(out_path: str, c2w_cv: np.ndarray, meta: dict):
    """OpenCV c2w 를 eval JSON(OpenGL) 로 쓴다. 읽는 쪽이 다시 GL2CV 를 곱해 원상복구된다."""
    frames = []
    for i, m in enumerate(c2w_cv):
        src = meta["frames"][min(i, len(meta["frames"]) - 1)]
        frames.append({**{k: v for k, v in src.items() if k != "transform_matrix"},
                       "transform_matrix": (m @ GL2CV).tolist()})
    out = {**{k: v for k, v in meta.items() if k != "frames"}, "frames": frames}
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, ensure_ascii=False, indent=1)


def main(args):
    test_out = path.join(args.out_dir, "test")
    makedirs(test_out, exist_ok=True)
    idx_pick = np.rint(np.linspace(0, args.src_poses - 1, args.n_poses)).astype(int)

    # ref 파일명은 `<prefix>_<scene>_<idx>_transforms_ref.json`. `--video` 를 주면 그 scene 하나,
    # 안 주면 prefix 아래 **전 scene**을 돈다 (dynpose val 은 22 scene 에 37 entry 라 필수).
    pattern = f"{args.prefix}_{args.video or '*'}_*_transforms_ref.json"
    refs = sorted(glob(path.join(args.ref_eval_dir, "test", pattern)),
                  key=lambda p: (path.basename(p).rsplit("_", 3)[0],
                                 int(path.basename(p).split("_")[-3])))
    assert refs, f"{args.ref_eval_dir}/test 에 {pattern} ref 가 없다"

    rows, n_ok, n_missing, n_degen = [], 0, 0, 0
    for ref_path in refs:
        name = path.basename(ref_path)[:-len("_transforms_ref.json")]
        entry = name.rsplit("_", 1)[1]
        # scene uuid 자체에 `_` 가 없어도 안전하게: 양끝(prefix, idx)만 떼어낸다.
        scene = name[len(args.prefix) + 1:-(len(entry) + 1)]
        npz_path = path.join(args.pred_dir, f"{args.npz_kind}__{scene}__{entry}.npz")
        if not path.isfile(npz_path):
            n_missing += 1
            continue
        z = np.load(npz_path, allow_pickle=True)
        gt_cv, meta = load_transforms(ref_path)
        gt_rel = rel_anchor(gt_cv)

        pred_c2w = np.asarray(z["c2w"], dtype=np.float64)
        if not args.scale_token:
            # 공식 배포 경로와 일치시킨다 — `eval.py:239 pose_normalize` 는 scale 토큰이 **안 걸린**
            # `camera_pose` 를 받아 JSON 을 쓴다 (`:233` 의 `c2ws[:,:3,3] *= scale_value` 는 numpy
            # 사본에만 걸리고 그 사본은 `draw_json` 궤적 PNG 전용이다. `core/utils.py:206-222
            # token_to_camera` 도 token 7·8 을 fx,fy 로만 쓰고 token 9(scale)는 안 쓴다).
            # 우리 infer 는 `gendop_release_infer.py:247` 에서 곱해 npz 에 넣고 곱한 값을 `scale`
            # 로 같이 저장하므로, 여기서 되나누면 **정확히** 배포 JSON 과 같은 크기가 된다.
            pred_c2w[:, :3, 3] /= float(z["scale"])
        pred_rel = rel_anchor(pred_c2w @ GL2CV)
        pred_rel = (pred_rel[idx_pick] if args.resample == "index_pick"
                    else gendop_resample(pred_rel, args.n_poses, args.gendop_root))
        r_pred, r_gt = rmax_of(pred_rel), rmax_of(gt_rel)
        if r_pred < 1e-9:
            n_degen += 1                      # degenerate: 정지 궤적으로 오독되지 않게 뺀다
            continue
        if args.rescale:
            pred_rel[:, :3, 3] *= r_gt / r_pred
        pred_world = gt_cv[0][None] @ pred_rel

        write_transforms(path.join(test_out, f"{name}_transforms_pred.json"), pred_world, meta)
        shutil.copyfile(ref_path, path.join(test_out, f"{name}_transforms_ref.json"))
        cap_out = path.join(test_out, f"{name}_caption.json")
        if args.caption_from == "npz":
            # GenDoP 이 실제로 받은 문장. 영상 타일 라벨용 — 그게 박혀야 정직하다.
            with open(cap_out, "w", encoding="utf-8") as file:
                json.dump({args.text_key: str(z["text"])}, file, ensure_ascii=False, indent=1)
        else:
            # 지표용. CLaTr 은 텍스트 임베딩과 궤적을 맞춰 보므로 두 arm 이 **같은 문장**을 봐야
            # 궤적 차이만 남는 짝지은 비교가 된다 -> 기준 eval 폴더의 캡션을 그대로 쓴다.
            shutil.copyfile(path.join(args.ref_eval_dir, "test", f"{name}_caption.json"), cap_out)
        rows.append((name, r_pred, r_gt, r_gt / r_pred))
        n_ok += 1

    print(f"{'prefix/video':14s}{args.prefix} / {args.video or '(전 scene)'}")
    print(f"{'ref entries':14s}{len(refs)}   scene {len({r[0].rsplit('_', 1)[0] for r in rows})}")
    print(f"{'resample':14s}{args.resample}  ({args.src_poses} -> {args.n_poses})")
    print(f"{'written':14s}{n_ok}")
    print(f"{'npz 없음':14s}{n_missing}")
    print(f"{'degenerate':14s}{n_degen}   (rmax 0 -> 스킵)")
    if rows:
        g = np.array([r[3] for r in rows])
        print(f"\n{'rmax gain (GT/GenDoP)':24s}median {np.median(g):.4f}  "
              f"min {g.min():.4f}  max {g.max():.4f}")
        print("\n[표본]")
        for name, rp, rg, gain in rows[:5]:
            print(f"  {name:<28} rmax pred {rp:.4f} -> GT {rg:.4f}  (x{gain:.3f})")
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", required=True)       # GenDoP npz 폴더
    parser.add_argument("--ref_eval_dir", required=True)   # ref/GT 를 빌려올 latentcam eval 폴더
    parser.add_argument("--out_dir", required=True)
    # ref 파일명 접두사. vista4d eval 폴더는 `vista4d_<scene>_<idx>_*`, dynpose(D95) 는
    # `dynpose_<uuid>_<idx>_*` 다. npz 쪽 접두사(`--npz_kind`)는 `collect()` 의 kind 라 따로다.
    parser.add_argument("--prefix", default="vista4d")
    parser.add_argument("--npz_kind", default="latentcam")
    parser.add_argument("--video", default=None)           # None = prefix 아래 전 scene
    parser.add_argument("--src_poses", type=int, default=30)   # GenDoP native pose 수
    parser.add_argument("--n_poses", type=int, default=49)     # 맞출 프레임 수
    parser.add_argument("--text_key", default="Movement")      # caption json 키 (npz 모드)
    # npz = GenDoP 이 받은 문장(영상 라벨용) / ref = 기준 eval 폴더 캡션(지표용, 짝지은 비교)
    parser.add_argument("--caption_from", choices=["npz", "ref"], default="npz")
    # 4번 단계(rmax 를 GT 에서 빌려오기) 스위치. `--no_rescale` 이면 GenDoP 이 낸 크기를 그대로
    # 둔다 -> 축·원점만 맞추고 **스케일은 DataDoP 게이지 그대로**인 arm. 크기 축을 아무도 안
    # 빌린 raw 대조군이지만, 게이지가 씬 단위와 무관하므로 수치는 "게이지 차"를 같이 잰다.
    parser.add_argument("--rescale", dest="rescale", action="store_true", default=True)
    parser.add_argument("--no_rescale", dest="rescale", action="store_false")
    # scale 토큰(`coords[:,9]` -> `exp(x/bins*4-2)`) 적용 스위치. `--no_scale_token` 이 **공식 배포
    # 경로**다 — `eval.py:239 pose_normalize` 는 scale 이 안 걸린 `camera_pose` 로 pred JSON 을
    # 쓴다 (`:233` 의 `c2ws[:,:3,3] *= scale_value` 는 numpy 사본에만 걸리고 그 사본은 궤적 PNG
    # 전용이며, `core/utils.py:206 token_to_camera` 도 token 7·8 만 fx,fy 로 쓴다).
    parser.add_argument("--scale_token", dest="scale_token", action="store_true", default=True)
    parser.add_argument("--no_scale_token", dest="scale_token", action="store_false")
    # 3번 단계. index_pick = 예전 런과 비트동일(중복 pose 로 계단), gendop_slerp = GenDoP 원본
    # 보간기(회전 SLERP + 이동 LERP). src_poses == n_poses 면 둘 다 항등이다.
    parser.add_argument("--resample", choices=["index_pick", "gendop_slerp"],
                        default="index_pick")
    parser.add_argument("--gendop_root",
                        default="/data1/cympyc1785/LatentCamVid/camera_generation/models/GenDoP")
    main(parser.parse_args())
