#!/bin/bash
# D129 dynpose 샤드 러너 — **vista D128 과 같은 축으로 다시 굽는다.**
# 사용자 지시 2026-09-05: "같은 방식으로 dynpose를 datadop 10% 비율로 해서 데이터 만들어놔줘".
#
# 계보: `run_dynpose_d122_shard.sh`(= vista D115+D121 을 dynpose 로 옮긴 것) + D128 6축.
# d122 뱅크 manifest 실측으로 6축이 **전부 옛 값**임을 확인했다 —
#   `fixed.tau_ref=auto` / `fixed.tracking=drift` / `axes.anchors=['dyn_0','dyn_1']`.
# 즉 이 재굽기는 이름만 바꾼 게 아니라 실제로 다른 카메라를 낸다.
#
# ── d122 대비 바뀐 축 (vista D128 과 같은 6개) ────────────────────────────────────
#  ① `--tau_ref follow` (TAU + FIT). **dynpose 에선 vista 보다 크게 먹는다** —
#     `dd_*` 도 같은 경로를 탄다 (`lbm/presets.py:339-345`: 명시 `follow` 가 preset 표를
#     이긴다). `--tau_ladder 1.00` + anchor 당 dd 슬롯 4개 구조라, follow 아래선
#     `tau_start` 가 항상 0 이 되어 `tau_saturated` 가 영영 안 걸린다
#     (`decode/build_poses.py:707-709`). τ 단계 `skipped.json` 12편의 거동도 달라진다.
#  ② `--track_dynamic_only --track_min_drift_u 0.05` (TAU). route 가 preset 을 명시로
#     넘겨도 **no-op 이 아니다** — `sample_camera_bank.py:827-833` 이 명시 목록 위에서 돈다.
#     d122 뱅크 실측: `track_*` 6,322 중 915 (14.5%) 가 빠진다 (= 전체 23,424 의 3.9%).
#  ③ anchor 상한 3/3 → **`route_presets.py` 쪽에 준다.** 뱅크에 주면 조용히 죽는다:
#     route 가 `--nodes ...` 를 명시로 넘기고 `sample_camera_bank.py:144-151` 이 그걸 받으면
#     `pick_main_anchors` 를 **호출하기 전에 return** 한다. 실제로 상한이 걸리는 곳은
#     `route_presets.py:82` 다. d122 는 `--num_anchors 2` (flat 2) 였고 그 플래그는 D127b 에
#     사라졌다 — 그대로 두면 argparse exit 2 → `$ARGS` 공백 → 전 씬 route FAIL 이다.
#     280 그래프 실측: 3/3 이면 편당 anchor 평균 2.91 (dyn 2.70 / stat 0.21, 최대 6)
#     → variant 가 23,424 에서 **~1.45배**로 는다. static 상한은 dynpose 에선 거의 무의미.
#  ④ surface drop — `route_presets.py:82` 의 `pick_main_anchors` 가 `schema.py:164`
#     기본 `drop_surfaces=True` 로 이미 건다 (플래그 없음). 280편 중 **12편**에서만 발동
#     (door 13 / locker door 5 / wall 2 / car 1 / refrigerator door 1 / road 1).
#  ⑤ `--trackings lock`(TAU) / `--tracking lock`(FIT) + `--preset_tracking`.
#     d122 는 `drift`(gain 0.6). `dd_*` 궤적은 `aim="traj"` 라 안 바뀌고
#     (`decode/build_poses.py:651` `tracking_ignored = aim != "look_at"`) 기록 문자열만 바뀐다.
#  ⑥ `--behind_min_zcam 0.02` (TAU + FIT).
#  + `--behind_clear_src_ratio 0` (FIT). 채택 보류 중인 축이고, `--behind_min_zcam` 과
#    같이 켜면 `source_g1_clear` 가 증거 0 으로 죽는다 (vista d128 에서 6편 날렸다,
#    `video_generation/FIX.log` 2026-09-05). 기본값도 0.0 이지만 명시한다 (D105).
#
# ── vista 와 다른 곳은 d122 때와 같은 세 줄뿐 ────────────────────────────────────
#   · `--eval_data DynPose-LBM` / `--output_root out_dynpose`
#   · `route_presets.py ... --num_external 4` 로 나온 `$ARGS` (preset 라우팅)
#   · `--external_shapes configs/datadop_shapes.json` + `--tau_ladder 1.00`
#
# ── GRAPH / CLOUD 는 안 돈다 ─────────────────────────────────────────────────────
# D128 은 `build_scene_graph.py` 를 안 건드렸다 (F1 기각). `.graph_d122` / `.cloud_d122`
# 마커가 이미 있는 씬만 처리한다 — 없으면 이 샤드가 만들 수 없으므로 FAIL 로 넘긴다.
# (vista d128 러너가 `.graph_s115` / `.cloud_s115` 를 요구하는 것과 같은 구조.)
#
# 사용:
#   bash scripts/run_dynpose_d129_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
D=/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM
SH=$HERE/configs/datadop_shapes.json
OUT=out_dynpose
TAU="${TAU:-bank_d129}"
BANK="${BANK:-hole_bank_d129}"
EASE="${EASE:-smooth_kf}"
MAXDYN="${MAXDYN:-3}"
MAXSTAT="${MAXSTAT:-3}"
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
    if [ -f "$OUT/$VIDEO/$BANK/skipped.json" ]; then
        echo "[s$SHARD] SKIP  $VIDEO (변이 0, 기록됨)"
        continue
    fi
    T0=$(date +%s)
    # 앞단은 d122 산출물을 그대로 쓴다.
    [ -f "$OUT/$VIDEO/.graph_d122" ] || { echo "[s$SHARD] FAIL  $VIDEO graph 없음"; continue; }
    [ -f "$OUT/$VIDEO/.cloud_d122" ] || { echo "[s$SHARD] FAIL  $VIDEO cloud 없음"; continue; }

    # ① preset 라우팅 — anchor 상한 ③ 과 surface drop ④ 가 **여기서** 걸린다.
    ARGS=$($PY scripts/route_presets.py --video "$VIDEO" --output_root "$OUT" \
        --external_shapes "$SH" --num_external 4 \
        --max_dynamic_anchors "$MAXDYN" --max_static_anchors "$MAXSTAT" --emit args \
        --out "$OUT/$VIDEO/preset_route_d129.json" 2>"$LOGDIR/$VIDEO.route.err" | tail -1)
    if [ -z "$ARGS" ]; then echo "[s$SHARD] FAIL  $VIDEO route"; continue; fi

    # ② τ 뱅크. anchor 상한/surface drop 은 여기 주면 no-op 이라 **일부러 안 넘긴다**
    #    (`--nodes` 가 `sample_camera_bank.py:144-151` 에서 필터를 우회한다).
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ] && [ ! -f "$OUT/$VIDEO/$TAU/skipped.json" ]; then
        echo "[s$SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/sample_camera_bank.py \
            --video "$VIDEO" --eval_data "$D" --output_root "$OUT" --bank_dir "$TAU" \
            --external_shapes "$SH" $ARGS --tau_ladder 1.00 \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal --deroll --no_preview --skip_on_empty --device cuda \
            --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
            --tau_ref follow \
            --track_dynamic_only --track_min_drift_u 0.05 \
            --trackings lock --preset_tracking \
            --behind_min_zcam 0.02 \
            > "$LOGDIR/$VIDEO.tau.log" 2>&1
        echo "[s$SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ]; then
        if [ -f "$OUT/$VIDEO/$TAU/skipped.json" ]; then
            mkdir -p "$OUT/$VIDEO/$BANK"
            cp "$OUT/$VIDEO/$TAU/skipped.json" "$OUT/$VIDEO/$BANK/skipped.json"
            echo "[s$SHARD] SKIP  $VIDEO (변이 0)"
        else
            echo "[s$SHARD] FAIL  $VIDEO tau"
        fi
        continue
    fi

    # ③ hole 사다리 재적합 — vista d128 과 같은 축 + external_shapes + 4단 사다리.
    echo "[s$SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/fit_hole_ladder.py \
        --video "$VIDEO" --eval_data "$D" --output_root "$OUT" \
        --bank_dir "$BANK" --tau_bank_dir "$TAU" --external_shapes "$SH" \
        --hole_ladder 0.10 0.20 0.35 0.50 \
        --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
        --fixed_focal --deroll \
        --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
        --tau_ref follow --tracking lock --preset_tracking \
        --behind_min_zcam 0.02 --behind_clear_src_ratio 0 \
        --collision_time_match --collision_source depth \
        --area_timeline --composition \
        --composition_min_area 0.004 --composition_max_nodes 3 \
        --device cuda \
        > "$LOGDIR/$VIDEO.fit.log" 2>&1
    RC=$?
    echo "[s$SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"

    # ④ emit (canonical).
    if [ $RC -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY scripts/emit_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
            > "$LOGDIR/$VIDEO.emit.log" 2>&1
        N=$($PY -c "import json;print(len(json.load(open('$OUT/$VIDEO/$BANK/bank.json'))['variants']))" 2>/dev/null)
        echo "[s$SHARD] EMIT  $VIDEO rc=$? n=$N total=$(( $(date +%s)-T0 ))s  $(date +%H:%M:%S)"
    fi
done < "$LIST"
echo "[s$SHARD] ALL DONE $(date +%H:%M:%S)"
