#!/bin/bash
# D122 dynpose 전 체인 러너 — **vista(D115+D121) 와 같은 방식으로 다시 굽는다.**
# 사용자 지시 2026-09-04: "vista 최신 방식으로 다시 돌려줘 / vista 돌렸던 방식으로 다시
# 돌려야하는 것들 dynpose 다시 돌려달라는 의미야 / 유일하게 다른건 preset, datadop 비율
# 맞춰서 개수 다른거야".
#
# 즉 이 러너는 `run_k6_d115_shard.sh`(GRAPH→CLOUD→TAU) + `run_k6_d121_shard.sh`(FIT→EMIT) 를
# 이어 붙인 것이고, **vista 와 다른 곳은 아래 세 줄뿐이다**:
#   · `--eval_data DynPose-LBM` / `--output_root out_dynpose`      (데이터 위치)
#   · `route_presets.py --num_anchors 2 --num_external 4` 로 나온 `$ARGS`  (preset 라우팅)
#   · `--external_shapes configs/datadop_shapes.json` + `--tau_ladder 1.00`  (DataDoP 비율)
# 나머지 게이트·사다리·조준·ease 축은 vista 와 **문자 단위로 같다**.
#
# ── 왜 fit 만으로 안 되고 전 체인인가 (D122 의 본론) ──────────────────────────────
# 첫 시도(fit-only)는 씬당 4초에 전량 즉사했다 —
#     scene_graph/scale.py:99 AssertionError: scale.mode 가 'frame0_ray' 다 (기대 'points_first_cam')
# dynpose scene_graph 280편은 `scale.mode` **필드 자체가 없어** 기본값 `frame0_ray` 로 읽힌다.
# vista 는 D115 때 52/53 편이 `points_first_cam` 으로 갱신됐다. F2 assert 는 09-02 20:30 에
# 추가됐고 dynpose 그래프는 09-02 01:05 빌드라 **19시간 차이로** d107/d110 fit 은 그냥 통과했었다.
# 즉 지금 디스크의 dynpose 코퍼스는 통째로 옛 게이지다.
#
# 게이지를 바꾸는 건 재fit 이 아니다. `S` 가 **세 곳**에 박혀 있다 —
#   ① `scene_graph.json` 의 `scale.S`      (게이트 임계가 전부 S 배율: behind_margin_frac·S,
#                                            obb_clear_floor·S, min_ground_clear·S, 가림 0.02·S)
#   ② `cloud.npz` 의 meta                   (`lbm/render.py:70` 이 렌더 단위로 읽는다)
#   ③ `bank_d107/bank.json` 의 top-level `S` (`sample_camera_bank.py:84` 도 assert_scale_mode 호출)
# 그래서 GRAPH → CLOUD → TAU → FIT 을 전부 다시 돈다. 실측 배율 S_new/S_old 는 3편에서
# 1.066 / 0.713 / 0.659 로 **1.0 양옆에** 걸친다 — 어느 쪽으로도 일괄 완화/강화가 아니다.
#
# ── 덮어쓰기 정책 (vista D115 선례 그대로) ────────────────────────────────────────
# graph 와 cloud 는 **제자리에서** 다시 짓는다. 옛 그래프는 `scene_graph_pre_d122.json` 으로
# 한 번만 복사해 둔다. `cloud.npz` 는 백업하지 않는다(편당 수백 MB) — 즉 **d107/d110 뱅크의
# 렌더 재현성은 이 시점에 끊긴다.** 뱅크 파일 자체(`bank_d107` / `hole_bank_d110`)는 안 건드리니
# 학습에 쓰인 카메라는 그대로 남는다. vista 가 d115 에서 d99 에 대해 한 것과 같은 거래다.
#
# `instance_desc.json`(266편) 은 **노드 id 로 키잉**돼 있고 게이지를 바꿔도 노드 id/label 이
# 안 변하는 것을 3편에서 확인했다 (id_same=True, label_diff=0). 그래서 다시 안 돌린다.
#
# ── 명시적으로 넘기는 기본값들 (D105) ──────────────────────────────────────────────
# `--hole_ladder 0.10 0.20 0.35 0.50` 는 argparse 기본값(`HOLE_LADDER`)과 **같은 값**이지만
# 적는다. vista d121 러너는 안 적었는데, 뱅크 정체성을 기본값에 맡기면 나중에 기본값이
# 뒤집혔을 때 한 이름 아래 두 규약이 섞인다. 같은 이유로 `--orbit_fixed_sweep`
# `--min_sweep_deg 20` `--behind_src_frames 49` `--collision_time_match`
# `--collision_source depth` `--area_timeline` `--composition` 도 전부 명시한다.
#
# `--gravity_source auto` 는 d107 dynpose 설정 그대로다. vista 는 기본값 `geocalib` 인데,
# dynpose 기존 280편이 전부 `method: geocalib` 로 풀렸으므로 **결과는 같고** geocalib 이
# 실패하는 씬에서만 fallback 이 산다. 씬을 잃지 않는 쪽을 고른다.
#
# 사용:
#   bash scripts/run_dynpose_d122_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
D=/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM
SH=$HERE/configs/datadop_shapes.json
OUT=out_dynpose
TAU="${TAU:-bank_d122}"
BANK="${BANK:-hole_bank_d122}"
EASE="${EASE:-smooth_kf}"
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i + 1)); continue; fi
    i=$((i + 1))
    if [ -f "$OUT/$VIDEO/$BANK/canonical/canonical.json" ]; then
        echo "[s$SHARD] SKIP  $VIDEO (이미 있음)"
        continue
    fi
    T0=$(date +%s)

    # ① scene graph — 새 `S` 게이지(points_first_cam). 마커로 재실행을 막는다.
    if [ ! -f "$OUT/$VIDEO/.graph_d122" ]; then
        if [ -f "$OUT/$VIDEO/scene_graph.json" ] && [ ! -f "$OUT/$VIDEO/scene_graph_pre_d122.json" ]; then
            cp "$OUT/$VIDEO/scene_graph.json" "$OUT/$VIDEO/scene_graph_pre_d122.json"
        fi
        echo "[s$SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/build_scene_graph.py \
            --video "$VIDEO" --eval_data "$D" --output_root "$OUT" \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            --gravity_source auto \
            --no_skip_done --no_overlay --no_topdown \
            > "$LOGDIR/$VIDEO.graph.log" 2>&1
        RC=$?
        echo "[s$SHARD] GRAPH $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.graph_d122"
    fi
    [ -f "$OUT/$VIDEO/.graph_d122" ] || { echo "[s$SHARD] FAIL  $VIDEO graph"; continue; }

    # ② 4D 점군 — 점 자체는 그대로지만 meta 의 `S` 가 바뀐다 (렌더러가 여기서 읽는다).
    if [ ! -f "$OUT/$VIDEO/.cloud_d122" ]; then
        echo "[s$SHARD] CLOUD $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY -m lbm.cloud \
            --video "$VIDEO" --eval_data "$D" --output_root "$OUT" --no_skip_done \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            > "$LOGDIR/$VIDEO.cloud.log" 2>&1
        RC=$?
        echo "[s$SHARD] CLOUD $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.cloud_d122"
    fi
    [ -f "$OUT/$VIDEO/.cloud_d122" ] || { echo "[s$SHARD] FAIL  $VIDEO cloud"; continue; }

    # ③ preset 라우팅 — **vista 와 다른 유일한 축**. anchor 2 + anchor 당 DataDoP shape 4.
    #    `--emit args` 는 sample_camera_bank 에 그대로 넘길 인자 문자열을 마지막 줄에 낸다.
    ARGS=$($PY scripts/route_presets.py --video "$VIDEO" --output_root "$OUT" \
        --external_shapes "$SH" --num_anchors 2 --num_external 4 --emit args \
        --out "$OUT/$VIDEO/preset_route_d122.json" 2>"$LOGDIR/$VIDEO.route.err" | tail -1)
    if [ -z "$ARGS" ]; then echo "[s$SHARD] FAIL  $VIDEO route"; continue; fi

    # ④ τ 뱅크.
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ] && [ ! -f "$OUT/$VIDEO/$TAU/skipped.json" ]; then
        echo "[s$SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/sample_camera_bank.py \
            --video "$VIDEO" --eval_data "$D" --output_root "$OUT" --bank_dir "$TAU" \
            --external_shapes "$SH" $ARGS --tau_ladder 1.00 \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal --deroll --no_preview --skip_on_empty --device cuda \
            --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
            > "$LOGDIR/$VIDEO.tau.log" 2>&1
        echo "[s$SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ]; then
        # `--skip_on_empty` 로 변이 0 인 씬은 정상 종료다 (게이트 전멸 = 쓸 카메라 없음).
        if [ -f "$OUT/$VIDEO/$TAU/skipped.json" ]; then
            echo "[s$SHARD] SKIP  $VIDEO (변이 0)"
        else
            echo "[s$SHARD] FAIL  $VIDEO tau"
        fi
        continue
    fi

    # ⑤ hole 사다리 재적합 — vista d121 과 문자 단위로 같은 축 + external_shapes.
    echo "[s$SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/fit_hole_ladder.py \
        --video "$VIDEO" --eval_data "$D" --output_root "$OUT" \
        --bank_dir "$BANK" --tau_bank_dir "$TAU" --external_shapes "$SH" \
        --hole_ladder 0.10 0.20 0.35 0.50 \
        --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
        --fixed_focal --deroll \
        --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
        --collision_time_match --collision_source depth \
        --area_timeline --composition \
        --composition_min_area 0.004 --composition_max_nodes 3 \
        --device cuda \
        > "$LOGDIR/$VIDEO.fit.log" 2>&1
    RC=$?
    echo "[s$SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"

    # ⑥ emit (canonical).
    if [ $RC -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/emit_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
            > "$LOGDIR/$VIDEO.emit.log" 2>&1
        N=$($PY -c "import json;print(len(json.load(open('$OUT/$VIDEO/$BANK/bank.json'))['variants']))" 2>/dev/null)
        echo "[s$SHARD] EMIT  $VIDEO rc=$? n=$N total=$(( $(date +%s)-T0 ))s  $(date +%H:%M:%S)"
    fi
done < "$LIST"
echo "[s$SHARD] ALL DONE $(date +%H:%M:%S)"
