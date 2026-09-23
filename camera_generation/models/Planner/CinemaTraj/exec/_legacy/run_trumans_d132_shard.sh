#!/bin/bash
# D132 뱅크 샤드 러너 (TRUMANS chunk). **mesh 충돌 + D128 게이트 축 + target=man 만**.
#
# ═══ 왜 새로 굽나 ══════════════════════════════════════════════════════════════════════
# 지금 TRUMANS 에 남아 있는 뱅크는 둘인데 **둘 다 사용자가 지시한 것과 다르다**:
#   · `out/trumans-bedroom/hole_bank_k6_d128` — D128 축은 맞지만 `collision.source == "depth"`.
#     depth shell 은 **소스 카메라가 본 표면**만 안다. 소스가 자기 뒤를 안 보므로 뒤로 물러나
#     벽을 뚫는 궤적에서 `behind_frac` 이 정확히 0.0000 이 나온다 (= 증거 없음을 통과로 읽는다).
#   · `out_trumans/*/hole_bank_k6_d116` — `collision.source == "both"` 로 mesh 를 쓰지만
#     D116 시점이라 D125~D128 에서 바뀐 게이트 축이 하나도 안 들어가 있다
#     (`fixed.tau_ref == "auto"`, `tracking == "drift"`, anchor 11개 = dyn_0 + stat_0..9).
# 사용자 지시 2026-09-05: "blender scene 에서 3d mesh 로 fitting / context video 에 안 보이는
# 부분이더라도 충돌·clearance 고려 / 전체 scene 에 안 부딪히도록 fitting 한 sequence 에 대해서만
# / target 은 우선 man 만".  → 두 축을 합친 **새 이름** `bank_d132` / `hole_bank_k6_d132`.
# (D105 원칙: 게이트 규약이 바뀌면 폴더 이름을 바꾼다. 한 이름 아래 두 규약이 섞이면 열 이름이
#  같아서 사후에 구분이 안 된다.)
#
# ═══ d116 대비 바뀐 축 ═════════════════════════════════════════════════════════════════
#  ① **`--nodes dyn_0`** (신규, 사용자 지시 "target 은 우선 man 만"). 191 chunk 전부 동적 노드가
#     `dyn_0:person` **하나뿐**이라 이 한 줄이 곧 "사람만 찍는다". `--nodes` 는 `anchor_nodes`
#     에서 1순위라 `max_*_anchors` / `drop_surfaces` 필터를 아예 안 탄다
#     (`sample_camera_bank.py:144-149`). d116 은 anchor 11개(dyn_0 + stat_0..9 = 벽/바닥/창)
#     였고 그 중 사람 몫은 solved 6,758 행 중 968 행(14.3%) 뿐이었다.
#  ② **`--tau_ref follow`** (D125/D128). d116 은 `auto` 라 `track_*` 만 follow 기준이었다.
#  ③ **`--track_min_drift_u 0.05`** (D128). 제자리 흔들림을 `track_*` 에서 뺀다.
#  ④ **`--trackings lock` / `--tracking lock`** (D126, 사용자 "카메라가 물체를 많이 놓치네").
#     d116 은 `drift`(gain 0.6).
#  ⑤ **`--behind_min_zcam 0.02`** — G1 이 카메라 뒤 픽셀을 벽으로 세던 것을 막는다.
#  ⑥ **`--behind_clear_src_ratio 0`** — D128 에서 명시 안 했다가 6편이 FIT rc=1 로 날아갔던
#     자리다 (`video_generation/FIX.log` 2026-09-05). argparse 기본값에 안 맡긴다.
#  ⑦ **`--area_timeline --composition`** (D121) — 캡션의 composition 열.
#  ⑧ **`--gravity_source gt --ground_source gt --subject_source gt`** 는 D106 부터 그대로.
#     GRAPH 는 이미 `.graph_s115` 마커가 있어 **다시 안 짓는다**.
#
# ═══ mesh 충돌 (핵심) ══════════════════════════════════════════════════════════════════
# `--collision_source both` = depth 위반 프레임 ∪ mesh 위반 프레임. 배타 선택이 아니라
# **합집합**인 이유는 D116 실측이다 (chunk a00, 836 변이, 임계 0.02·S = 3.9 cm):
#     depth 130 위반 / mesh 357 위반 / **겹침 0**
# 한쪽만 쓰면 다른 쪽 위반을 통째로 잃는다. 격자(`mesh_grid.npz`)는 `.blend` 삼각형을
# 5 cm 복셀로 굽고 소스 카메라 위치에서 free 공간을 flood-fill 한 것이라, **소스 영상에 한
# 번도 안 나온 벽·옆방·가구 내부**까지 안다 (`lbm/mesh_collision.py` 헤더).
# 격자가 없으면 `resolve_mesh_grid` 가 죽는다 — 조용히 depth 로 안 떨어진다. 그래서 이 샤드는
# 격자가 있는 chunk 만 돈다 (48/191). 없는 chunk 는 `build_trumans_mesh_grid.py` 가 선행.
#
# ═══ "안 부딪힌 sequence 에 대해서만" ══════════════════════════════════════════════════
# 이분법이 게이트를 만족하는 **최대 크기**를 풀고, 못 풀면 `bank.csv` 의 `status` 가
# `collision_limited` / `obb_limited` / `elev_limited` / `ground_limited` / `approach_limited`
# 로 남는다. `status == "solved"` 행만이 "전 프레임 충돌·clearance 통과"다. 하류(릴 렌더 /
# emit / export)는 전부 solved 만 집으므로 이 조건은 이미 집행되고 있다.
#
# env: `vista4d`.  GPU 는 렌더러(hole 측정)가 쓴다.
#
# 사용:
#   bash exec/_legacy/run_trumans_d132_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
EVAL_DATA=/data1/cympyc1785/data/TRUMANS-Lite
OUT=out_trumans
#    τ 뱅크도 **새로** 굽는다 — anchor 집합이 11개에서 1개로 바뀌므로 d115/d116 의 τ 뱅크를
#    재사용하면 사다리가 stat_* 행까지 다시 살려낸다.
TAU="${TAU:-bank_d132}"
BANK="${BANK:-hole_bank_k6_d132}"
EASE="${EASE:-smooth_kf}"
# D139. 앞단(graph/cloud/mesh)이 없으면 **여기서 짓는다**. d132 최초 실행 때는 이 셋이 이미
# 있는 48 chunk 만 돌았고 나머지 143 chunk 는 `FAIL ... 없음` 으로 통째로 빠졌다 (48/191).
# `PREP=0` 이면 예전처럼 앞단을 안 짓고 없으면 건너뛴다 — 48 chunk 재실행이 비트 동일해진다.
# (앞단이 이미 있는 chunk 에서는 `PREP=1` 도 no-op 이다. 세 단계 전부 `[ ! -f ]` 가드다.)
PREP="${PREP:-1}"
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
        echo "[s$SHARD] SKIP  $VIDEO (이미 있음)"
        continue
    fi
    # ⓪ 앞단 (D139, `PREP=1`). 플래그는 `run_trumans_d115_shard.sh:46-84` 와 **글자 그대로
    #    같아야** 한다 — 마커 이름이 `.graph_s115`/`.cloud_s115` 라서, 다른 설정으로 지어놓고
    #    같은 마커를 찍으면 48 chunk 와 143 chunk 가 서로 다른 `S` 게이지를 갖게 되고
    #    그 차이는 열 이름이 같아서 사후에 구분이 안 된다.
    if [ "$PREP" = "1" ]; then
        # ⓪-a scene graph — mesh GT 중력·지면·subject + points_first_cam 게이지.
        if [ ! -f "$OUT/$VIDEO/.graph_s115" ]; then
            if [ -f "$OUT/$VIDEO/scene_graph.json" ] && [ ! -f "$OUT/$VIDEO/scene_graph_pre_d115.json" ]; then
                cp "$OUT/$VIDEO/scene_graph.json" "$OUT/$VIDEO/scene_graph_pre_d115.json"
            fi
            echo "[s$SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY fit/graph/build_scene_graph.py \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
                --gravity_source gt --ground_source gt --subject_source gt \
                --scene_scale_mode points_first_cam --scene_scale_stride 1 \
                --no_skip_done --no_overlay --no_topdown \
                > "$LOGDIR/$VIDEO.graph.log" 2>&1
            RC=$?
            echo "[s$SHARD] GRAPH $VIDEO rc=$RC  $(date +%H:%M:%S)"
            [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.graph_s115"
        fi
        # ⓪-b 4D 점군 — `lbm/render.py:70` 이 여기 meta 의 `S` 를 읽는다.
        if [ -f "$OUT/$VIDEO/.graph_s115" ] && [ ! -f "$OUT/$VIDEO/.cloud_s115" ]; then
            echo "[s$SHARD] CLOUD $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY -m lbm.cloud \
                --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" --no_skip_done \
                --scene_scale_mode points_first_cam --scene_scale_stride 1 \
                > "$LOGDIR/$VIDEO.cloud.log" 2>&1
            RC=$?
            echo "[s$SHARD] CLOUD $VIDEO rc=$RC  $(date +%H:%M:%S)"
            [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.cloud_s115"
        fi
        # ⓪-c mesh 점유/EDT 격자 — ② 의 `--collision_source both` 입력. Blender 가 여기서
        #     한 번만 뜬다(로드 16 s, 502 objects). GPU 불필요.
        if [ ! -f "$OUT/$VIDEO/mesh_grid.npz" ]; then
            echo "[s$SHARD] MESH  $VIDEO  $(date +%H:%M:%S)"
            $PY fit/ingest/build_trumans_mesh_grid.py \
                --video "$VIDEO" --output_root "$OUT" \
                > "$LOGDIR/$VIDEO.mesh.log" 2>&1
            echo "[s$SHARD] MESH  $VIDEO rc=$?  $(date +%H:%M:%S)"
        fi
    fi
    # 앞단이 그래도 없으면 이 chunk 는 못 만든다. `resolve_mesh_grid` 는 격자가 없으면 죽지
    # 조용히 depth 로 안 떨어지므로, 여기서 거르는 게 곧 "mesh 로 fitting 한 sequence 만".
    [ -f "$OUT/$VIDEO/.graph_s115" ]   || { echo "[s$SHARD] FAIL  $VIDEO graph 없음"; continue; }
    [ -f "$OUT/$VIDEO/.cloud_s115" ]   || { echo "[s$SHARD] FAIL  $VIDEO cloud 없음"; continue; }
    [ -f "$OUT/$VIDEO/mesh_grid.npz" ] || { echo "[s$SHARD] FAIL  $VIDEO mesh_grid 없음"; continue; }

    # ① τ 뱅크. anchor 선별(`--nodes dyn_0`) + preset 라우팅이 여기서 정해진다.
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ]; then
        echo "[s$SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py \
            --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
            --bank_dir "$TAU" \
            --nodes dyn_0 \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal --deroll --no_preview \
            --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
            --tau_ref follow \
            --track_dynamic_only --track_min_drift_u 0.05 \
            --trackings lock --preset_tracking \
            --behind_min_zcam 0.02 \
            > "$LOGDIR/$VIDEO.tau.log" 2>&1
        echo "[s$SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
    [ -f "$OUT/$VIDEO/$TAU/bank.json" ] || { echo "[s$SHARD] FAIL  $VIDEO tau"; continue; }

    # ② hole 사다리 재적합 — **여기가 mesh 충돌이 걸리는 곳**.
    echo "[s$SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
        --video "$VIDEO" --eval_data "$EVAL_DATA" --output_root "$OUT" \
        --bank_dir "$BANK" --tau_bank_dir "$TAU" \
        --anchors dyn_0 \
        --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
        --fixed_focal --deroll \
        --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
        --tau_ref follow --tracking lock --preset_tracking \
        --behind_min_zcam 0.02 --behind_clear_src_ratio 0 \
        --collision_time_match --collision_source both \
        --area_timeline --composition \
        --composition_min_area 0.004 --composition_max_nodes 3 \
        > "$LOGDIR/$VIDEO.fit.log" 2>&1
    RC=$?
    echo "[s$SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
    # ③ emit (canonical).
    if [ $RC -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/emit_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
            > "$LOGDIR/$VIDEO.emit.log" 2>&1
        echo "[s$SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
done < "$LIST"
echo "[s$SHARD] ALL DONE $(date +%H:%M:%S)"
