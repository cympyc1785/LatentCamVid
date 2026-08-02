#!/bin/bash
# scripts/eval_testset.py 를 여러 (run, ckpt) 조합에 대해 순차 실행한다.
# GPU 는 하나만 쓰므로 병렬로 띄우면 OOM/경합만 나서 순차가 맞다.
#   사용: GPU=3 scripts/eval_testset_queue.sh <run_dir>:<ckpt> [<run_dir>:<ckpt> ...]
#   예:   GPU=3 scripts/eval_testset_queue.sh results/20260731_001511_dl3dv_geo_worldtraj:best.pth
set -u
GPU=${GPU:-3}
PY=${PY:-/data1/cympyc1785/miniconda3/envs/latentcam/bin/python}
REPO=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam
LOGDIR=${LOGDIR:-/tmp}

cd "$REPO" || exit 1
for job in "$@"; do
  run=${job%%:*}; ckpt=${job##*:}
  tag=$(basename "$run")__${ckpt%.pth}
  log="$LOGDIR/eval_testset_${tag}.log"
  echo "[$(date +%H:%M:%S)] START $tag -> $log"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="main:.:data" "$PY" scripts/eval_testset.py \
      --run "$run" --ckpt "$ckpt" > "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] DONE  $tag"
    grep -A20 '^{$' "$log" | tail -20
  else
    echo "[$(date +%H:%M:%S)] FAIL  $tag (rc=$rc)"; tail -25 "$log"
  fi
done
echo "[$(date +%H:%M:%S)] queue finished"
