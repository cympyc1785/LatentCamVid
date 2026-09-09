#!/bin/bash
# D115 뱅크 샤드 러너 (TRUMANS) — vista 의 `run_k6_d115_shard.sh` 와 **같은 규약**으로 굽는다.
# 두 코퍼스를 섞어 학습하므로 게이지/회전 규약이 갈리면 안 된다. 바뀐 4축(F2 `S` 재정의 /
# F9 orbit 반경 / F11 G1 49프레임 / min_sweep 20)의 설명은 vista 판 헤더 참조.
#
# vista 판과 다른 점은 **증거 소스**뿐이다 — TRUMANS 는 우리가 Blender 에서 렌더한 클립이라
# mesh GT 가 전부 남아 있다 (D106): `--gravity_source gt --ground_source gt --subject_source gt`.
# 추정할 이유가 없다.
#
# D116 — G1(behind-surface)도 같은 이유로 mesh 를 같이 쓴다 (`--collision_source both`).
# depth shell 은 **소스 카메라가 본 표면**만 알아서 뒤로 물러나 벽을 뚫는 궤적을 통째로 놓친다
# (소스는 자기 뒤를 안 본다 → `behind_frac` 이 정확히 0.0000). 실측 a00 836변이, 임계 0.02·S
# = 3.9 cm: **depth 130 / mesh 357 / 겹침 0**. 겹침이 0이라 어느 한쪽만 쓰면 다른 쪽 위반을
# 통째로 잃는다 — 그래서 배타 선택이 아니라 위반 **프레임 합집합**인 `both` 다.
# 격자는 chunk 당 한 번 굽는다 (②-b, Blender 1회 ~2분 + numpy ~90초).
#
# 사용:
#   bash scripts/run_trumans_d115_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
EVAL_DATA=/data1/cympyc1785/data/TRUMANS-Lite
OUT=out_trumans
#    τ 뱅크는 d115 것을 **그대로 공유**한다 — `sample_camera_bank.py` 는 `--measure_behind` 가
#    기본 off 라 G1 을 애초에 안 재고, D116 이 바꾼 건 G1 뿐이다 (`bank.json` 의
#    `collision.source` 가 `"off"` 인 것으로 확인 가능). 다시 구우면 같은 결과에 시간만 쓴다.
TAU=bank_d115
#    사다리는 **새 이름**이다. 게이트 축 2개(`--collision_time_match` on / `--collision_source
#    both`)가 바뀌었으므로 d115 와 같은 폴더 이름을 쓰면 한 이름 아래 두 규약이 섞인다 (D105).
#    옛 `hole_bank_k6_d115` 는 depth-only 대조군으로 남긴다.
BANK="${BANK:-hole_bank_k6_d116}"
EASE="${EASE:-smooth_kf}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i + 1)); continue; fi
    i=$((i + 1))
    if [ -f "$OUT/$VIDEO/$BANK/canonical/canonical.json" ]; then
        echo "[shard $SHARD] SKIP  $VIDEO (이미 있음)"
        continue
    fi
    # ① scene graph — mesh GT + 새 `S` 게이지로 다시 짓는다.
    if [ ! -f "$OUT/$VIDEO/.graph_s115" ]; then
        if [ -f "$OUT/$VIDEO/scene_graph.json" ] && [ ! -f "$OUT/$VIDEO/scene_graph_pre_d115.json" ]; then
            cp "$OUT/$VIDEO/scene_graph.json" "$OUT/$VIDEO/scene_graph_pre_d115.json"
        fi
        echo "[shard $SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/build_scene_graph.py \
            --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
            --gravity_source gt --ground_source gt --subject_source gt \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            --no_skip_done --no_overlay --no_topdown \
            > "$LOGDIR/$VIDEO.graph.log" 2>&1
        RC=$?
        echo "[shard $SHARD] GRAPH $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.graph_s115"
    fi
    [ -f "$OUT/$VIDEO/.graph_s115" ] || { echo "[shard $SHARD] FAIL  $VIDEO graph"; continue; }
    # ② 4D 점군 — meta 의 `S` 가 바뀐다 (`lbm/render.py:70` 이 여기서 읽는다).
    if [ ! -f "$OUT/$VIDEO/.cloud_s115" ]; then
        echo "[shard $SHARD] CLOUD $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY -m lbm.cloud \
            --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" --no_skip_done \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            > "$LOGDIR/$VIDEO.cloud.log" 2>&1
        RC=$?
        echo "[shard $SHARD] CLOUD $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.cloud_s115"
    fi
    [ -f "$OUT/$VIDEO/.cloud_s115" ] || { echo "[shard $SHARD] FAIL  $VIDEO cloud"; continue; }
    # ②-b mesh 점유/EDT 격자 (D116) — ④ 의 `--collision_source both` 입력.
    #     Blender 는 여기서 **한 번만** 뜬다 (로드만 16 s, 502 objects). 이분법이 몇 번 돌든
    #     게이트는 이 npz 만 읽는다. GPU 불필요 — 삼각형 export + scipy EDT 뿐이다.
    if [ ! -f "$OUT/$VIDEO/mesh_grid.npz" ]; then
        echo "[shard $SHARD] MESH  $VIDEO  $(date +%H:%M:%S)"
        $PY scripts/build_trumans_mesh_grid.py \
            --video "$VIDEO" --output_root "$OUT" \
            > "$LOGDIR/$VIDEO.mesh.log" 2>&1
        echo "[shard $SHARD] MESH  $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
    [ -f "$OUT/$VIDEO/mesh_grid.npz" ] || { echo "[shard $SHARD] FAIL  $VIDEO mesh"; continue; }
    # ③ τ 뱅크. G1 을 안 재므로(`--measure_behind` 기본 off) 격자를 안 읽는다.
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ]; then
        echo "[shard $SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/sample_camera_bank.py \
            --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
            --bank_dir "$TAU" \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal --deroll --no_preview \
            --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
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
        --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
        --collision_time_match --collision_source both \
        > "$LOGDIR/$VIDEO.fit.log" 2>&1
    RC=$?
    echo "[shard $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
    if [ $RC -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/emit_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
            > "$LOGDIR/$VIDEO.emit.log" 2>&1
        echo "[shard $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
