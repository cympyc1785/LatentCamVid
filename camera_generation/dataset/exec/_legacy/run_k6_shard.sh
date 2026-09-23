#!/bin/bash
# pseudo-GT 뱅크(`--aim_keyframes 6 --fixed_focal`) 재적합 + emit 샤드 러너.
#
# 왜 별도 스크립트인가: `run_f0share_shard.sh` 와 같은 이유다 — `fit_hole_ladder.py` 는 영상
# 1편만 받으므로 52편을 GPU 여러 장에 나누려면 바깥에서 stride 샤딩을 해야 한다. f0share 와
# 다른 점은 두 가지: ① 인자가 `--aim_keyframes 6 --fixed_focal` 이고 ② fit 직후 `emit_bank.py`
# 까지 이어 돌려 `canonical/` 을 같이 만든다 (하류 export 가 canonical 을 읽는다).
#
# `--aim_anchor` 를 안 준다: `aim_keyframes>0` 이면 keyframe 0 이 소스 frame0 회전이라
# anchor ramp 자체가 안 돌아간다 (`decode/build_poses.py:549`). f0share 축이 여기서는 no-op.
#
# 사용:
#   bash exec/_legacy/run_k6_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
BANK=hole_bank_k6
# D96. 회전 스케줄 기본값이 `smooth_kf` 다 (예전엔 `smoothstep`) — 근거는
# `decode/build_poses.py:965`. 옛 뱅크 재현: `EASE=smoothstep bash exec/_legacy/run_k6_shard.sh ...`.
# 새 규약으로 구울 거면 `$BANK` 도 새 이름으로 (한 폴더에 두 규약을 섞지 말 것).
EASE="${EASE:-smooth_kf}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -eq "$SHARD" ]; then
        # 이미 끝난 편은 건너뛴다 (재시작 안전). canonical 까지 있어야 완료로 본다.
        if [ -f "out/$VIDEO/$BANK/canonical/canonical.json" ]; then
            echo "[shard $SHARD] SKIP $VIDEO (이미 있음)"
        else
            echo "[shard $SHARD] START $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
                --video "$VIDEO" --bank_dir "$BANK" \
                --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                --fixed_focal \
                > "$LOGDIR/$VIDEO.fit.log" 2>&1
            RC=$?
            echo "[shard $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
            if [ $RC -eq 0 ]; then
                CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/emit_bank.py \
                    --video "$VIDEO" --bank_dir "$BANK" \
                    > "$LOGDIR/$VIDEO.emit.log" 2>&1
                echo "[shard $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
            fi
        fi
    fi
    i=$((i + 1))
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
