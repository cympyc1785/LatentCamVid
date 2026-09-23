#!/bin/bash
# D145 — 확장된 metadata.csv 의 신규 scene 에 SAM3 인스턴스 분할을 돌린다.
#
# `sam3_seg_instances.py` 는 이미 샤딩(`--num_shards/--shard_id`)과 `--skip_done` 을 갖고
# 있으므로 스크립트는 안 고친다. 여기서 하는 건 GPU 배정 + 로그 + 환경 고정뿐이다.
#
# 영상 목록을 `--videos` 로 넘기지 않는 이유: 그 스크립트는 목록을 안 주면 metadata.csv 전량을
# 쓰고(:70), `--skip_done` 이 masks.npz 있는 것을 건너뛴다. 기존 279편은 전부 done 이므로
# 결과적으로 신규분만 돈다 — 600개 이름을 argv 로 나르는 것보다 실수 여지가 적다.
#
# 사용:
#   bash exec/_legacy/run_dynpose_d145_sam3.sh <gpu> <shard_id> <num_shards>
# 예 (GPU 2,3 두 샤드):
#   screen -dmS d145sam0 bash exec/_legacy/run_dynpose_d145_sam3.sh 2 0 2
#   screen -dmS d145sam1 bash exec/_legacy/run_dynpose_d145_sam3.sh 3 1 2
set -u

GPU="${1:?gpu}"
SHARD="${2:?shard_id}"
NSHARD="${3:?num_shards}"

PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
VG=/data1/cympyc1785/LatentCamVid/video_generation
EVAL=/data1/cympyc1785/data/DynPose-LBM

echo "[d145-sam3] gpu=$GPU shard=$SHARD/$NSHARD  $(date +%F' '%H:%M:%S)"
CUDA_VISIBLE_DEVICES="$GPU" $PY "$VG/scripts/sam3_seg_instances.py" \
  --eval_data "$EVAL" \
  --num_shards "$NSHARD" --shard_id "$SHARD" \
  --skip_done
RC=$?
echo "[d145-sam3] rc=$RC shard=$SHARD  $(date +%F' '%H:%M:%S)"
exit "$RC"
