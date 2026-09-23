#!/bin/bash
# hole_bank_f0share 재적합 샤드 러너 (`--aim_anchor source_frame0`).
#
# 왜 별도 스크립트인가: `fit_hole_ladder.py` 는 영상 1편만 받으므로 52편을 돌리려면 바깥에서
# 루프를 돌아야 하고, GPU 3장에 나눠 걸려면 stride 샤딩이 필요하다. 기존 `hole_bank` 는
# 건드리지 않는다 (`--bank_dir` 로 분리) — 사용자 지시 "기존꺼 지우지 말고".
#
# 사용:
#   bash exec/_legacy/run_f0share_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -eq "$SHARD" ]; then
        # 이미 끝난 편은 건너뛴다 (재시작 안전).
        if [ -f "out/$VIDEO/hole_bank_f0share/bank.json" ]; then
            echo "[shard $SHARD] SKIP $VIDEO (이미 있음)"
        else
            echo "[shard $SHARD] START $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
                --video "$VIDEO" --aim_anchor source_frame0 --bank_dir hole_bank_f0share \
                > "$LOGDIR/$VIDEO.log" 2>&1
            echo "[shard $SHARD] DONE  $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
    fi
    i=$((i + 1))
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
