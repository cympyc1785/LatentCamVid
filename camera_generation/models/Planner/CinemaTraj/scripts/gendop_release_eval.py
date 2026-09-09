"""GenDoP release 생성 궤적(`gendop_release_infer.py` 산출) ↔ 원 카메라 태그 왕복 평가.

무엇을 재나: 캡션은 실측 카메라에서 뽑았다. 그 캡션으로 GenDoP 가 궤적을 생성했으니, 생성
궤적을 **같은 분절기**로 재태깅해 원 카메라의 태그와 대조하면 "텍스트가 담은 motion 을
모델이 돌려줬는가"(왕복 일치)가 된다. latentcam eval 의 caption P/R/F 와 같은 철학
(`evaluate/eval/src/metrics/modules/caption.py` — 프레임별 35-class weighted P/R/F)이되,
코퍼스가 달라 수치를 직접 비교하지는 말 것.

게이지 주의: 생성 궤적은 DataDoP(MonST3R) 게이지라 우리 코퍼스 캘리브레이션 임계(0.05)가
아니라 **DataDoP 원본 컨벤션**(120 pose 기준 fps 30 · static 0.02 · diff 0.4)으로 태깅한다 —
모델이 그 컨벤션으로 학습됐기 때문. 단 30 native 포즈를 그대로 쓰고(슬러프 업샘플은 roll
아티팩트, [[rerope 메모]] 아님 caption ablation 실측 B변형) fps 는 30*(30/120)=7.5 로 줄여
같은 실시간 속도 눈금을 유지한다. GT 쪽 라벨은 `captions_gendop/<name>_tag.json` (49프레임,
th 0.05) 를 그대로 읽는다.

프레임별 비교는 양쪽 라벨 시퀀스를 길이 49 로 index-pick 정렬해 계산한다.

**게이지 통일 옵션** (`--n_poses` / `--fps`): 위 기본값은 "모델이 학습된 게이지로 pred 를
태깅한다"는 입장이다. 반대 입장 — **우리 모델이 학습된 게이지로 전 arm 을 통일한다** — 을
쓰려면 `--n_poses 49 --fps 10` 을 준다. 그러면 30 native pose 를 분절 **전에** 49 로
index-pick 해서 GT(49프레임 native, fps 10) 와 같은 눈금 위에 올린다. 기본값은 예전 그대로라
`--n_poses` 없이 부르면 bit-identical 이다.

  왜 pose 를 먼저 늘리나: 기본 경로는 30 pose 를 분절한 뒤 **라벨만** 49 로 늘린다. 그러면
  smoothing_window_size=18 이 pred 궤적의 60% 를, GT 궤적의 37% 를 덮어 평활 정도가 서로
  다르다. 분절 전에 길이를 맞추면 두 쪽 다 37% 가 된다.
  왜 fps 가 눈금인가: `segmentation.py:92  t_velocities = fps * velocities[:, :3, 3]` —
  fps 는 실시간 사실이 아니라 static_threshold 에 대한 속도 배율이다.

env: GenDoP.  예시:
    python scripts/gendop_release_eval.py --pred_dir results/20260828_gendop_eval/text_motion
    python scripts/gendop_release_eval.py --pred_dir <dir> --n_poses 49 --fps 10 \
        --latentcam_root /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np
import torch

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
GENDOP_PIPE = "/data1/cympyc1785/pipeline/GenDoP"
EVAL_DATA = "/data1/cympyc1785/LatentCamVid/DATA/Vista4D-Eval-Data/eval_data"
for extra in (GENDOP_PIPE, path.join(GENDOP_PIPE, "dataset/scripts")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from core.utils import sample_from_dense_cameras                       # noqa: E402
from processing.segmentation import (                                  # noqa: E402
    ANG_INDEX_TO_PATTERN,
    CAM_INDEX_TO_PATTERN,
    find_consecutive_chunks,
    segment_rigidbody_trajectories,
)

SET_DIR = {"cameras": "cameras", "recon": "recon_and_seg"}
# DataDoP 원본 컨벤션 (모델의 학습 게이지) — fps 만 30 pose 기준으로 환산 (30*(30/120)=7.5)
PRED_SEG = dict(cam_static_threshold=0.02, cam_diff_threshold=0.4,
                angular_static_threshold=0.005, fps=7.5,
                smoothing_window_size=18, min_chunk_size=10)


def opencv_like_identity(c2w):
    """GenDoP 출력은 이미 DataDoP(OpenGL) c2w 다 — 변환 없이 그대로 분절기에 넣는다."""
    return np.asarray(c2w, dtype=np.float64)


def per_frame_labels(segments, length):
    """분절 인덱스 시퀀스 -> 길이 `length` 로 index-pick 정렬한 int 라벨."""
    seg = np.asarray(list(segments), dtype=int)
    idx = np.rint(np.linspace(0, len(seg) - 1, length)).astype(int)
    return seg[idx]


def resample_c2w(c2w, n_poses, mode="slerp"):
    """(F,4,4) -> (n_poses,4,4). `n_poses<=0` 이면 원본 그대로.

    `mode="index"` 는 `gendop_preds_to_eval_dir.py` 와 같은 index-pick 이다. **분절 라벨을
    잴 때는 쓰지 말 것**: 30→49 는 48 step 중 19개가 중복 프레임이 되어 속도 0 이 되고,
    `smooth_segments` 가 window 19 안의 최빈값을 고르므로 40% 짜리 static 표가 mode 를
    가져갈 수 있다 — 궤적이 아니라 리샘플러가 라벨을 만든다.

    `mode="slerp"` 는 `caption_cameras_datadop.resample_poses` 와 같은
    `sample_from_dense_cameras` 다. 단 t 를 `i/num_poses` 가 아니라 `linspace(0,1,n)` 으로
    잡아 끝점까지 덮는다 (그쪽은 DataDoP 원본 convention 을 따라 끝을 안 닿는다).
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    if n_poses <= 0 or n_poses == c2w.shape[0]:
        return c2w
    if mode == "index":
        return c2w[np.rint(np.linspace(0, c2w.shape[0] - 1, n_poses)).astype(int)]
    dense = torch.tensor(c2w[:, :3, :].reshape(1, -1, 12), dtype=torch.float32)
    rows = [sample_from_dense_cameras(dense, torch.full((1, 1), float(t)))[0]
            for t in np.linspace(0.0, 1.0, n_poses)]
    out = np.tile(np.eye(4), (n_poses, 1, 1))
    out[:, :3, :] = torch.cat(rows, dim=0).numpy().reshape(n_poses, 3, 4)
    return out


def chunk_text(segments):
    parts = []
    for index, _, _ in find_consecutive_chunks(list(segments)):
        move, ang = CAM_INDEX_TO_PATTERN[index // 7], ANG_INDEX_TO_PATTERN[index % 7]
        parts.append(move if ang == "static" else (ang if move == "static" else f"{move}+{ang}"))
    return "; ".join(parts)


def weighted_prf(y_true, y_pred):
    """torchmetrics 없이 같은 정의(가중 multiclass P/R/F)를 계산한다."""
    classes, prec, rec, f1, weight = np.unique(y_true), 0.0, 0.0, 0.0, 0.0
    for c in classes:
        tp = float(((y_pred == c) & (y_true == c)).sum())
        p = tp / max(float((y_pred == c).sum()), 1e-9)
        r = tp / max(float((y_true == c).sum()), 1e-9)
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        w = float((y_true == c).sum())
        prec += p * w; rec += r * w; f1 += f * w; weight += w
    return prec / weight, rec / weight, f1 / weight


def main(args):
    # 기존 동작 보존: --fps 를 안 주면 PRED_SEG(7.5), --n_poses 를 안 주면 리샘플 없음.
    pred_seg = dict(PRED_SEG, fps=float(args.fps))
    preds = sorted(glob(path.join(args.pred_dir, "*.npz")))
    rows, missing = [], []
    y_true_all, y_pred_all = [], []
    for p in preds:
        kind, scene, name = path.basename(p)[:-4].split("__", 2)   # name 에 __ 가 들어갈 수 있다
        if kind == "latentcam":     # LBM-Lite 합성 target 코퍼스는 레이아웃이 다르다 (scene 밑 da3/)
            tag_path = path.join(args.latentcam_root, "vista4d", scene, "da3",
                                 "captions_gendop", f"{name}_tag.json")
        else:
            tag_path = path.join(EVAL_DATA, SET_DIR[kind], scene, "captions_gendop",
                                 f"{name}_tag.json")
        if not path.isfile(tag_path):
            missing.append(f"{scene}/{name}")
            continue
        with open(tag_path, encoding="utf-8") as file:
            gt = json.load(file)
        z = np.load(p, allow_pickle=True)
        c2w = resample_c2w(opencv_like_identity(z["c2w"]), args.n_poses, args.resample)
        seg_pred = segment_rigidbody_trajectories(torch.tensor(c2w), **pred_seg)
        yt = per_frame_labels(gt["segments"], args.align_len)
        yp = per_frame_labels(seg_pred, args.align_len)
        y_true_all.append(yt); y_pred_all.append(yp)
        pr, rc, f1 = weighted_prf(yt, yp)
        rows.append(dict(kind=kind, scene=scene, name=name, fscore=round(f1, 4),
                         precision=round(pr, 4), recall=round(rc, 4),
                         degenerate=bool(z["degenerate"]),
                         gt_chunks=chunk_text(gt["segments"]),
                         pred_chunks=chunk_text(list(seg_pred))))

    yt = np.concatenate(y_true_all); yp = np.concatenate(y_pred_all)
    pr, rc, f1 = weighted_prf(yt, yp)
    pooled = dict(n=len(rows), precision=round(pr, 4), recall=round(rc, 4),
                  fscore=round(f1, 4),
                  mean_fscore=round(float(np.mean([r["fscore"] for r in rows])), 4),
                  degenerate=sum(r["degenerate"] for r in rows),
                  exact_frame_match=round(float((yt == yp).mean()), 4))
    out = dict(pred_dir=args.pred_dir, pred_seg_kwargs=pred_seg,
               pred_n_poses=args.n_poses, pred_resample=args.resample,
               align_len=args.align_len,
               gt_tags="captions_gendop/<name>_tag.json (49f, th 0.05)",
               pooled=pooled, missing_tags=missing, rows=rows)
    out_path = path.join(args.pred_dir, args.out_name)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(out, file, ensure_ascii=False, indent=2)

    print(f"{'gauge':<16}n_poses={args.n_poses or 'native'} "
          f"resample={args.resample} fps={pred_seg['fps']} align={args.align_len}")
    print(f"{'entries':<16}{pooled['n']}   (tag 없음 {len(missing)})")
    print(f"{'pooled P/R/F':<16}{pooled['precision']} / {pooled['recall']} / {pooled['fscore']}")
    print(f"{'mean fscore':<16}{pooled['mean_fscore']}")
    print(f"{'frame match':<16}{pooled['exact_frame_match']}")
    print(f"{'degenerate':<16}{pooled['degenerate']}")
    print(f"{'out':<16}{out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pred_dir", required=True)
    # kind=="latentcam" 인 npz 의 GT tag 를 찾을 코퍼스 루트 (다른 kind 는 EVAL_DATA 를 쓴다)
    parser.add_argument("--latentcam_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3")
    # 게이지 통일용. 기본값은 예전 동작 그대로 (리샘플 없음 + fps 7.5).
    parser.add_argument("--n_poses", type=int, default=0)      # 0 = 리샘플 안 함, 49 = 우리 눈금
    parser.add_argument("--resample", default="slerp", choices=["slerp", "index"])
    parser.add_argument("--fps", type=float, default=PRED_SEG["fps"])
    parser.add_argument("--align_len", type=int, default=49)   # 라벨 시퀀스 정렬 길이
    parser.add_argument("--out_name", default="roundtrip_eval.json")
    main(parser.parse_args())
