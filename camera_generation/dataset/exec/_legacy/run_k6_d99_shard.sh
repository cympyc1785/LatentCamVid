#!/bin/bash
# D105 뱅크 샤드 러너 (vista) — D99 의 `--deroll` 을 켜고 굽는 첫 뱅크. τ 뱅크 → hole 사다리
# 적합 → emit 을 한 번에. vista 52편(`out/k6_videos.txt`)용.
#
# 왜 `run_k6_d98_shard.sh` 와 별도인가: 바뀐 건 **회전 규약**이다 (중력 기준 roll 보정).
# d98 뱅크는 `deroll=False` 로 구워졌고, 그 결과 `tilt_*` 가 roll 163.88°, `pan_*` 가 18.12°
# 를 흘린다 — d98 20,854행 중 3,280행(15.7%)이 Dutch angle 로 앉아 있다. 한 폴더에 두 규약을
# 섞지 않는다는 D96/D97/D98 규칙 그대로 `bank_d99/` + `hole_bank_k6_d99/` 에 새로 쓴다.
# 옛 뱅크(`bank_d98`/`hole_bank_k6_d98`)는 하나도 안 건드린다.
#
# `--deroll` 을 **명시적으로 넘긴다**: D105 에서 argparse 기본값이 True 로 바뀌었지만, 뱅크
# 정체성을 기본값에 맡기면 나중에 기본값이 또 뒤집혔을 때 같은 폴더에 두 규약이 섞인다
# (D90 의 `smooth_passes` 사고와 같은 종류).
#
# 나머지 인자는 d98 과 동일하다 — `--tau_ref auto`(D97), `--keyframe_ease smooth_kf`(D96),
# `--preset_tracking`(D93), `--track_dynamic_only`(D77), `--follow_gains "0"`(D72) 는 전부
# 기본값을 탄다. **preset 도 안 넘긴다 = 현재 어휘 전량 40종**.
#
# 사용:
#   bash exec/_legacy/run_k6_d99_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
TAU=bank_d99
BANK=hole_bank_k6_d99
EASE="${EASE:-smooth_kf}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -eq "$SHARD" ]; then
        if [ -f "out/$VIDEO/$BANK/canonical/canonical.json" ]; then
            echo "[shard $SHARD] SKIP $VIDEO (이미 있음)"
        else
            # ① τ 뱅크. 이미 있으면 건너뛴다 (재시작 안전).
            if [ ! -f "out/$VIDEO/$TAU/bank.json" ]; then
                echo "[shard $SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
                CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py \
                    --video "$VIDEO" --bank_dir "$TAU" \
                    --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                    --fixed_focal --deroll --no_preview \
                    > "$LOGDIR/$VIDEO.tau.log" 2>&1
                echo "[shard $SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
            fi
            # ② hole 사다리 재적합 + ③ emit (canonical).
            echo "[shard $SHARD] START $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
                --video "$VIDEO" --bank_dir "$BANK" --tau_bank_dir "$TAU" \
                --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                --fixed_focal --deroll \
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
