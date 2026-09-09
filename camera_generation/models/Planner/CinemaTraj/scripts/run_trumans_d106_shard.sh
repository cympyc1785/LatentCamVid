#!/bin/bash
# D106 뱅크 샤드 러너 (TRUMANS) — d99 (deroll + GT 중력) 에 **mesh GT 두 가지를 더** 얹는다.
# 사용자 지시: "Trumans는 합성데이터니까 최대한 blender 3d mesh를 이용해야해".
#
# d99 판(`run_trumans_d99_shard.sh`)과 다른 점은 graph 단계의 플래그 두 개뿐이다:
#
# ① `--subject_source gt` — subject OBB·조준점(track center)을 depth 점군 shell 이 아니라
#    probe 의 `human_track`(blend mesh 정점 union AABB)에서. 9편 실측: shell 은 높이비
#    median 0.73 / 조준 오차 median 0.255 m (키의 16%) 였다.
# ② `--ground_source gt` — 지면을 점군 2% 분위수가 아니라 probe 의 `floor_z` 에서.
#    (d99 는 "중력만 바꿔야 원인이 하나다"라고 지면을 안 건드렸는데, 그 bake 는 2/188 에서
#    죽였으므로 격리할 대조군이 없다 — 이번엔 mesh GT 를 한 번에 다 켠다.)
#
# 뱅크 dir 이름은 d99 그대로 둔다 (bank_d99 / hole_bank_k6_d99) — vista d99 와 같은 규약으로
# 한 코퍼스로 섞이는 이름이라 바꾸면 하류 export 가 갈라진다. graph 마커만 d106 이다.
# 188/188 전부 probe(human_track 49프레임 + floor_z) 커버리지 확인 완료 (2026-09-01).
#
# 사용:
#   bash scripts/run_trumans_d106_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
EVAL_DATA=/data1/cympyc1785/data/TRUMANS-Lite
OUT=out_trumans
TAU=bank_d99
BANK=hole_bank_k6_d99
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
        # ① scene graph — GT 중력 + GT 지면 + GT subject 로 **다시** 짓는다.
        if [ ! -f "$OUT/$VIDEO/.graph_gt_d106" ]; then
            if [ -f "$OUT/$VIDEO/scene_graph.json" ] && [ ! -f "$OUT/$VIDEO/scene_graph_pre_d99.json" ]; then
                cp "$OUT/$VIDEO/scene_graph.json" "$OUT/$VIDEO/scene_graph_pre_d99.json"
            fi
            echo "[shard $SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY scripts/build_scene_graph.py \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
                --gravity_source gt --ground_source gt --subject_source gt \
                --no_skip_done --no_overlay --no_topdown \
                > "$LOGDIR/$VIDEO.graph.log" 2>&1
            RC=$?
            echo "[shard $SHARD] GRAPH $VIDEO rc=$RC  $(date +%H:%M:%S)"
            [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.graph_gt_d106"
        fi
        [ -f "$OUT/$VIDEO/.graph_gt_d106" ] || { echo "[shard $SHARD] FAIL  $VIDEO graph"; continue; }
        # ② 4D 점군. 중력축·subject 와 무관하므로 있으면 그대로 쓴다.
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
            CUDA_VISIBLE_DEVICES=$GPU $PY scripts/sample_camera_bank.py \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
                --bank_dir "$TAU" \
                --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                --fixed_focal --deroll --no_preview \
                > "$LOGDIR/$VIDEO.tau.log" 2>&1
            echo "[shard $SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
        [ -f "$OUT/$VIDEO/$TAU/bank.json" ] || { echo "[shard $SHARD] FAIL  $VIDEO tau"; continue; }
        # ④ hole 사다리 재적합 + ⑤ emit.
        echo "[shard $SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/fit_hole_ladder.py \
            --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
            --bank_dir "$BANK" --tau_bank_dir "$TAU" \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal --deroll \
            > "$LOGDIR/$VIDEO.fit.log" 2>&1
        RC=$?
        echo "[shard $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
        if [ $RC -eq 0 ]; then
            CUDA_VISIBLE_DEVICES=$GPU $PY scripts/emit_bank.py \
                --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
                > "$LOGDIR/$VIDEO.emit.log" 2>&1
            echo "[shard $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
    else
        i=$((i + 1))
    fi
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
