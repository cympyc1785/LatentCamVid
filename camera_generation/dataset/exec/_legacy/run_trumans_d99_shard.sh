#!/bin/bash
# D105 뱅크 샤드 러너 (TRUMANS) — vista 의 `run_k6_d99_shard.sh` 와 **같은 규약**으로 굽는다.
# 두 코퍼스를 섞어 학습하므로 회전 규약이 갈리면 안 된다.
#
# vista 판과 다른 점은 두 가지뿐이다.
#
# ① 앞에 graph / cloud 단계가 있다 (`run_trumans_d77_shard.sh` 와 같음).
#
# ② **scene graph 를 GT 중력으로 다시 짓는다** (`--gravity_source gt --no_skip_done`).
#    deroll 은 중력축 둘레의 roll 을 0 으로 만드는 연산이라, 중력축이 틀리면 지평선을 틀린
#    각도로 "세운다" — 고치려던 것과 같은 종류의 기울어짐을 다시 심는다.
#    현재 디스크의 191 chunk graph 는 D98(GeoCalib) 이전 것이고 GT 대비 실측이 이렇다:
#
#      ground_ransac       n=142   median  0.08°  p90  4.17°  max  9.73°
#      camera_up_fallback  n= 49   median 10.87°  p90 21.31°  max 26.23°
#
#    26% 가 10° 넘게 기울어 있다. TRUMANS 는 우리가 Blender 에서 렌더한 클립이라 blend world
#    (z-up) 가 남아 있고 191/191 전부 GT 를 찾을 수 있다 — 추정할 이유가 없다.
#    옛 graph 는 `scene_graph_pre_d99.json` 으로 한 번만 복사해 둔다 (d77 뱅크의 입력이었다).
#    vista 는 D98 에서 이미 52/52 가 GeoCalib 이라 다시 안 짓는다.
#    지면(`--ground_source`)은 안 건드린다 = pointcloud 기본값. 중력만 바꿔야 원인이 하나다.
#
# 사용:
#   bash exec/_legacy/run_trumans_d99_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
EVAL_DATA=/data1/cympyc1785/data/TRUMANS-Lite
OUT=out_trumans
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
        i=$((i + 1))
        if [ -f "$OUT/$VIDEO/$BANK/canonical/canonical.json" ]; then
            echo "[shard $SHARD] SKIP  $VIDEO (이미 있음)"
            continue
        fi
        # ① scene graph — GT 중력으로 **다시** 짓는다. 마커로 재실행을 막는다.
        if [ ! -f "$OUT/$VIDEO/.graph_gt_d99" ]; then
            if [ -f "$OUT/$VIDEO/scene_graph.json" ] && [ ! -f "$OUT/$VIDEO/scene_graph_pre_d99.json" ]; then
                cp "$OUT/$VIDEO/scene_graph.json" "$OUT/$VIDEO/scene_graph_pre_d99.json"
            fi
            echo "[shard $SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/graph/build_scene_graph.py \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
                --gravity_source gt --no_skip_done --no_overlay --no_topdown \
                > "$LOGDIR/$VIDEO.graph.log" 2>&1
            RC=$?
            echo "[shard $SHARD] GRAPH $VIDEO rc=$RC  $(date +%H:%M:%S)"
            [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.graph_gt_d99"
        fi
        [ -f "$OUT/$VIDEO/.graph_gt_d99" ] || { echo "[shard $SHARD] FAIL  $VIDEO graph"; continue; }
        # ② 4D 점군. 중력축과 무관하므로 있으면 그대로 쓴다.
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
                --fixed_focal --deroll --no_preview \
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
            --fixed_focal --deroll \
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
