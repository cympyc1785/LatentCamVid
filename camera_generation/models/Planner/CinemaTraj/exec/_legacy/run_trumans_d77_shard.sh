#!/bin/bash
# TRUMANS-Lite D77 뱅크 샤드 러너 — scene graph → 점군 → τ 뱅크 → hole 사다리 → emit.
#
# `run_k6_d77_shard.sh` 와 다른 점은 **앞 두 단계가 더 있다**는 것뿐이다. Vista 는 51편 전부
# `scene_graph.json` / `cloud.npz` 를 예전에 만들어 뒀지만 TRUMANS 는 없다. 나머지 3단계는
# 인자까지 동일하다 (`--aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE"
# --fixed_focal`) — 두 코퍼스의 뱅크가 같은 규약이어야 섞어 학습할 수 있다.
#
# `--follow_gains` 를 안 주는 이유도 같다: `decode/build_poses.py:472` 는 gain 이 문자열 "0"
# 일 때만 `PRESET_FOLLOW` 를 적용한다. `auto` 를 주면 `track_*` 이 조용히 추종을 멈춘다.
#
# 각 단계는 산출물이 있으면 건너뛴다 (재시작 안전). 실패한 영상은 다음 영상으로 넘어간다 —
# 737편 중 몇 편이 죽는다고 샤드 전체가 멈추면 안 된다.
#
# 사용:
#   bash exec/_legacy/run_trumans_d77_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
EVAL_DATA=/data1/cympyc1785/data/TRUMANS-Lite
OUT=out_trumans
TAU=bank_d77
BANK=hole_bank_k6_d77
# D96. 회전 스케줄 기본값이 `smooth_kf` 다 (예전엔 `smoothstep`). trumans / vista / dynpose
# 세 코퍼스를 같은 규약으로 굽기 위한 것이고, 근거는 `decode/build_poses.py:965` 의 714변이
# 실측이다 — 각속도 맥동비 21.65 -> 4.58, 49프레임 조준오차 median 도 0.668° -> 0.418° 로
# 같이 내려간다. 대가는 keyframe 회전을 정확히 통과하지 않는 것 하나뿐.
# 옛 뱅크를 되만들려면 `EASE=smoothstep bash exec/_legacy/run_trumans_d77_shard.sh ...`.
# 주의: 이 값을 바꾸면 `$BANK` 도 같이 바꿔야 한다. 같은 폴더 이름에 두 규약이 섞이면
# `merge_static_rung.py` 의 `fixed` 대조 말고는 알아챌 방법이 없다.
EASE="${EASE:-smooth_kf}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -eq "$SHARD" ]; then
        i=$((i + 1))
        if [ -f "$OUT/$VIDEO/$BANK/canonical/canonical.json" ]; then
            echo "[shard $SHARD] SKIP  $VIDEO (이미 있음)"
            continue
        fi
        # ① scene graph. `--no_overlay --no_topdown` 은 737편에 시각화가 필요 없어서다.
        if [ ! -f "$OUT/$VIDEO/scene_graph.json" ]; then
            echo "[shard $SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/graph/build_scene_graph.py \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
                --no_overlay --no_topdown \
                > "$LOGDIR/$VIDEO.graph.log" 2>&1
            echo "[shard $SHARD] GRAPH $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
        [ -f "$OUT/$VIDEO/scene_graph.json" ] || { echo "[shard $SHARD] FAIL  $VIDEO graph"; continue; }
        # ② 4D 점군.
        if [ ! -f "$OUT/$VIDEO/cloud.npz" ]; then
            echo "[shard $SHARD] CLOUD $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY -m lbm.cloud \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
                > "$LOGDIR/$VIDEO.cloud.log" 2>&1
            echo "[shard $SHARD] CLOUD $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
        [ -f "$OUT/$VIDEO/cloud.npz" ] || { echo "[shard $SHARD] FAIL  $VIDEO cloud"; continue; }
        # ③ τ 뱅크.
        if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ]; then
            echo "[shard $SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
                --bank_dir "$TAU" \
                --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                --fixed_focal --no_preview \
                > "$LOGDIR/$VIDEO.tau.log" 2>&1
            echo "[shard $SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
        [ -f "$OUT/$VIDEO/$TAU/bank.json" ] || { echo "[shard $SHARD] FAIL  $VIDEO tau"; continue; }
        # ④ hole 사다리 재적합 + ⑤ emit.
        echo "[shard $SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
            --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
            --bank_dir "$BANK" --tau_bank_dir "$TAU" \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal \
            > "$LOGDIR/$VIDEO.fit.log" 2>&1
        RC=$?
        echo "[shard $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
        if [ $RC -eq 0 ]; then
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/emit_bank.py \
                --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
                > "$LOGDIR/$VIDEO.emit.log" 2>&1
            echo "[shard $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
    else
        i=$((i + 1))
    fi
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
