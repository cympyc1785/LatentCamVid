#!/bin/bash
# D77 뱅크(k6 = --aim_keyframes 6)(`track_*` 포함 34 preset) 샤드 러너 — τ 뱅크 → hole 사다리 적합 → emit 을 한 번에.
#
# 왜 `run_k6_shard.sh` 와 별도인가: k6 는 **이미 있던** τ 뱅크(`out/<video>/bank/`, D75 어휘
# 16 preset)를 재적합만 했다. D77 은 preset 축 자체가 바뀌었으므로(34 preset, `track_*` 12종
# 신규) τ 뱅크부터 다시 만들어야 한다 — `fit_hole_ladder.py:520` 이 τ 뱅크에 seed 행이 없는
# (anchor, preset) 쌍을 조용히 건너뛰기 때문에 `--presets` 로 넘겨도 안 나온다.
#
# τ 뱅크를 `bank/` 가 아니라 `bank_d77/` 에 쓰는 이유: k6 소스를 덮어쓰면 기존 뱅크를 다시
# 못 만든다. `fit_hole_ladder.py --tau_bank_dir` 로 읽는 쪽을 갈아끼운다.
#
# `--follow_gains` 를 안 준다(기본 "0"): `decode/build_poses.py:472` 는 들어온 gain 이 문자열
# "0" 일 때만 `PRESET_FOLLOW` 를 적용한다. `auto` 를 주면 τ 최소 gain 이 풀려서 `track_*` 이
# 조용히 추종을 멈춘다 — 캡션만 "tracks" 인 궤적이 생긴다.
#
# 사용:
#   bash exec/_legacy/run_k6_d77_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
TAU=bank_d77
BANK=hole_bank_k6_d77
# D96. 회전 스케줄 기본값이 `smooth_kf` 다 (예전엔 `smoothstep`). trumans / vista / dynpose
# 세 코퍼스를 같은 규약으로 굽기 위한 것이고, 근거는 `decode/build_poses.py:965` 의 714변이
# 실측이다 — 각속도 맥동비 21.65 -> 4.58, 조준오차 median 0.668° -> 0.418°.
# 옛 뱅크 재현: `EASE=smoothstep bash exec/_legacy/run_k6_d77_shard.sh ...`.
# 주의: 이미 구운 `hole_bank_k6_d77`(51편 전량 smoothstep)에 이 값으로 덮어 굽지 말 것 —
# 이 스크립트는 canonical 이 있으면 건너뛰므로 실수로 섞일 일은 없지만, 새로 구울 때는
# `$BANK` 를 새 이름으로 바꿔야 한 폴더에 두 규약이 안 섞인다.
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
                    --fixed_focal --no_preview \
                    > "$LOGDIR/$VIDEO.tau.log" 2>&1
                echo "[shard $SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
            fi
            # ② hole 사다리 재적합 + ③ emit (canonical).
            echo "[shard $SHARD] START $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
                --video "$VIDEO" --bank_dir "$BANK" --tau_bank_dir "$TAU" \
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
