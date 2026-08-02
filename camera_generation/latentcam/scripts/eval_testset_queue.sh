#!/bin/bash
# scripts/eval_testset.py 를 여러 (run, ckpt) 조합에 대해 순차 실행한다.
# GPU 는 하나만 쓰므로 병렬로 띄우면 OOM/경합만 나서 순차가 맞다.
#   사용: GPU=3 scripts/eval_testset_queue.sh <run_dir>:<ckpt> [<run_dir>:<ckpt> ...]
#   예:   GPU=3 scripts/eval_testset_queue.sh results/20260731_001511_dl3dv_geo_worldtraj:best.pth
#
# [new 2026-08-03] EXTRA / TAG_SUFFIX. EXTRA 는 eval_testset.py 에 그대로 붙는 추가 인자,
#   TAG_SUFFIX 는 출력 디렉토리 이름 뒤에 붙어 기존 결과를 덮어쓰지 않게 한다. seg_list 를
#   못박아 다시 돌릴 때 쓴다 (기존 eval_my/* 는 오늘 pool 의 random_split 로 평가돼서 그 run 의
#   train 이 89.48% 섞여 있음):
#     EXTRA="--set train_seg_list=.../latentcam_train_seg_list_c4d2k5y4.txt \
#            --set test_seg_list=.../latentcam_test_seg_list_c4d2k5y4.txt" \
#     TAG_SUFFIX=__c4d2k5y4 GPU=0 scripts/eval_testset_queue.sh <run>:last.pth ...
set -u
GPU=${GPU:-3}
PY=${PY:-/data1/cympyc1785/miniconda3/envs/latentcam/bin/python}
REPO=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam
LOGDIR=${LOGDIR:-/tmp}
EXTRA=${EXTRA:-}
TAG_SUFFIX=${TAG_SUFFIX:-}

cd "$REPO" || exit 1
for job in "$@"; do
  run=${job%%:*}; ckpt=${job##*:}
  tag=$(basename "$run")__${ckpt%.pth}${TAG_SUFFIX}
  log="$LOGDIR/eval_testset_${tag}.log"
  echo "[$(date +%H:%M:%S)] START $tag -> $log"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="main:.:data" "$PY" scripts/eval_testset.py \
      --run "$run" --ckpt "$ckpt" --out "eval_my/$tag" $EXTRA > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] DONE  $tag"
    grep -A20 '^{$' "$log" | tail -20
  else
    echo "[$(date +%H:%M:%S)] FAIL  $tag (rc=$rc)"; tail -25 "$log"
  fi
done
echo "[$(date +%H:%M:%S)] queue finished"
