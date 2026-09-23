#!/bin/bash
# D128 뱅크 샤드 러너 (vista 52편). d121 대비 바뀐 축을 **전부 명시적으로** 넘긴다
# (D105 의 교훈 — 뱅크 정체성을 argparse 기본값에 맡기면 기본값이 뒤집혔을 때 한 폴더에
# 두 규약이 섞인다). 그래서 새 이름 `bank_d128` / `hole_bank_k6_d128` 을 쓴다.
#
# d121 대비 바뀐 축 (5개):
#  ① **`--tau_ref follow`** (전 preset). d121 은 `auto` 라 `track_*` 만 follow 기준이었다.
#     D125 parkour 프로브 실측: `auto` -> `follow` 로 stat_2/3/6 의 G1 위반 프레임 비율
#     (`behind`)이 0.347/0.286/0.163 -> 전부 0.000, 대신 `path_len_u` 가 0.254->0.159,
#     0.334->0.143 (stat_6 은 0.346->0.349 로 유지). 정확성을 사고 다양성을 팔았다.
#  ② **`--track_min_drift_u 0.05`** (D128 신규). `track_*` 게이트가 보던 `moving` 은
#     `path_len_u > 0.05` 라 제자리 흔들림(춤·손짓·그네)을 통과시킨다. 순변위 0.05 u 이하인
#     49 노드가 d121 에서 track_* 2040 쌍을 만들었고, 그 쌍들의 track vs 비-track 차이는
#     |Δτ| median 0.0210 — 진짜 이동 anchor(0.2071) 의 10분의 1이라 "follows the subject"
#     캡션을 떠받치기엔 얇다. **`moving` 자체는 안 건드린다** (F1 기각) — `moving` 은
#     `schema.py:207` 의 anchor split key 라, 강등하면 그 dyn 노드가 static 버킷으로 넘어가
#     벽·바닥과 3칸을 다툰다 (`swing` 은 dyn 6개가 전부 넘어가 dynamic anchor 가 0 이 된다).
#  ③ **`--max_dynamic_anchors 3 --max_static_anchors 3`** (D127b). d121 은 parkour 에서
#     anchor 7개(dyn_0 + stat 6) 로 664 variant 를 냈다. 사용자 지시: "dynamic target,
#     static target 각각 최대 3개로 해서 가장 main 이 되는 애들만".
#  ④ **`--anchor_drop_surfaces`** (D127). 벽/바닥/천장/문/창/울타리/난간을 anchor 에서 뺀다
#     (§`scene_graph/schema.py` SURFACE_TOKENS). d115 의 TAU 는 D127 이전이라 아직 남아 있다
#     — d121 실측으로 static anchor 186 (씬,노드) 중 34 가 걸린다 (window 13 / fence 11 /
#     column 4 / curtain 2 / railing 2; `wall`·`floor`·`ground` 는 vista 에 0건).
#  ⑤ **`--trackings lock` / `--tracking lock`** (D126). translation 추종 gain 0.6(drift) 을
#     1.0(lock) 으로. 사용자: "카메라가 물체를 많이 놓치네".
#  ⑥ **`--behind_min_zcam 0.02`** — G1 이 카메라 뒤 픽셀을 벽으로 세던 것을 막는다.
#  + **`--behind_clear_src_ratio 0`** (FIT). 채택 보류 중인 축이라 0 이 맞는데, 첫 실행 때
#    argparse 기본값 0.3 에 의존했다가 6편(avocado-slice/basketball-four/bed-shopping/
#    bmx-bumps/breakdance/camel)이 FIT rc=1 로 날아갔다 — ⑥ 과 같이 켜면 소스 재투영이
#    전부 `z_floor` 아래로 떨어져 `source_g1_clear` 가 증거 0 으로 assert 한다
#    (`video_generation/FIX.log` 2026-09-05). D105 원칙대로 **명시**한다.
#
# GRAPH / CLOUD 는 **안 돈다**. F1 을 기각해 `build_scene_graph.py` 가 d115 와 비트 동일하고,
# `.graph_s115` / `.cloud_s115` 마커가 이미 52편 전부에 있다. 다시 구우면 같은 결과에
# 편당 ~90 s 를 쓴다.
#
# 사용:
#   bash exec/_legacy/run_k6_d128_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
OUT=out
TAU="${TAU:-bank_d128}"
BANK="${BANK:-hole_bank_k6_d128}"
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
    # 앞단은 d115 산출물을 그대로 쓴다 — 없으면 이 샤드가 만들 수 없으므로 멈춘다.
    [ -f "$OUT/$VIDEO/.graph_s115" ] || { echo "[shard $SHARD] FAIL  $VIDEO graph 없음"; continue; }
    [ -f "$OUT/$VIDEO/.cloud_s115" ] || { echo "[shard $SHARD] FAIL  $VIDEO cloud 없음"; continue; }

    # ① τ 뱅크 (anchor 선별 + preset 라우팅이 여기서 정해진다 = ②③④⑤ 가 걸리는 곳).
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ]; then
        echo "[shard $SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$TAU" \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal --deroll --no_preview \
            --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
            --tau_ref follow \
            --track_dynamic_only --track_min_drift_u 0.05 \
            --max_dynamic_anchors 3 --max_static_anchors 3 --anchor_drop_surfaces \
            --trackings lock --preset_tracking \
            --behind_min_zcam 0.02 \
            > "$LOGDIR/$VIDEO.tau.log" 2>&1
        echo "[shard $SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
    [ -f "$OUT/$VIDEO/$TAU/bank.json" ] || { echo "[shard $SHARD] FAIL  $VIDEO tau"; continue; }

    # ② hole 사다리 재적합.
    echo "[shard $SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
        --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" --tau_bank_dir "$TAU" \
        --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
        --fixed_focal --deroll \
        --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
        --tau_ref follow --tracking lock --preset_tracking \
        --behind_min_zcam 0.02 --behind_clear_src_ratio 0 \
        --collision_time_match --collision_source depth \
        --area_timeline --composition \
        --composition_min_area 0.004 --composition_max_nodes 3 \
        > "$LOGDIR/$VIDEO.fit.log" 2>&1
    RC=$?
    echo "[shard $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
    # ③ emit (canonical).
    if [ $RC -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/emit_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
            > "$LOGDIR/$VIDEO.emit.log" 2>&1
        echo "[shard $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
