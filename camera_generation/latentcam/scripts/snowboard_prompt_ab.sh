#!/bin/bash
# D104. snowboard 뱅크(hole_bank_k6, 77 변이) 를 k6 posed 모델로 추론한다. arm 은 **프롬프트뿐**:
#   target   = "target: man motion: ..."   (prompt_fields [target, motion])
#   notarget = "motion: ..."               (prompt_fields [motion])
# 두 export 의 target_poses 는 비트 동일이라 (|dE|max 0.0) 예측 차이는 전부 텍스트 탓이다.
# snowboard 는 train/test 어느 split 에도 없어서 전용 export 를 쓴다 (train_seg_list 는 0줄).
# --skip-clatr: 필요한 산출물은 test/*_transforms_{ref,pred}.json 이고, 이걸로 depth warp 를 건다.
#   사용: GPU=2 ARM=target scripts/snowboard_prompt_ab.sh
set -u
GPU=${GPU:-2}
ARM=${ARM:-target}
CKPT=${CKPT:-last.pth}
RUN=${RUN:-results/20260827_051423_vista4d_pgt_k6}
PY=${PY:-/data1/cympyc1785/miniconda3/envs/latentcam/bin/python}
REPO=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam
ROOT=/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_snowboard_${ARM}
OUT=${OUT:-results/20260901_snowboard_prompt_ab/${ARM}__${CKPT%.pth}}

cd "$REPO" || exit 1
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="main:.:data" "$PY" scripts/eval_testset.py \
    --run "$RUN" --ckpt "$CKPT" --out "$OUT" --skip-clatr \
    --set dl3dv_root="$ROOT" \
    --set train_seg_list="$ROOT/seg_list_vista4d_train.txt" \
    --set test_seg_list="$ROOT/seg_list_vista4d_test.txt"
echo "SNOWBOARD_${ARM}_RC=$?"
