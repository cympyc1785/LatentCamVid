#!/bin/bash
# D149 dynpose 전 체인 러너 — **앞단까지 통째로 vista 방식으로** 다시 굽는다 (880편).
#
# 사용자 지시 2026-09-06: "detection은 VLM 사용한걸로 해주고 static 트리도 추가해줘.
# dd는 일단 그럼 빼주고 vista 돌린거랑 일관성 있게 dynpose돌려줘."
# 그 앞 지시: "scene 개수가 많으니 **preset 개수만** 다르게 하고 싶은거야."
# 즉 vista(`run_k6_d128_shard.sh`) 와 다른 곳은 데이터 위치와 **preset 라우팅 한 줄**뿐이다.
#
# ── d129 대비 바뀐 축 4개 ────────────────────────────────────────────────────────
#  ① **detection 을 VLM 명사로 통일.** d145 이전 279편은 배포 annotation 명사로 SAM3 를
#     돌렸고 신규 596편은 `extract_nouns_vlm.py` 명사로 돌렸다 — 한 코퍼스 안에 탐지 어휘가
#     두 종류였다. 279편을 VLM 명사로 재검출해 880편 전체를 한 어휘로 맞춘다.
#     (옛 `seg_instances` 는 `seg_instances_bak_pre_d149_20260906` 로 백업했다.)
#  ② **static 트리 추가.** vista 는 `seg_instances_static` 80편이 있고 그래프가 자동으로
#     읽는다 (`build_scene_graph.py:540` `--seg_static_root` 기본 None = 자동). dynpose 는
#     그 디렉토리가 **없어서** static 노드가 전부 `stat_*` 없이 비어 있었다. `vlm_nouns.json`
#     의 `static` 절반은 d145 때 이미 뽑아 뒀고(쓰이지 않고 있었다) 그걸 그대로 쓴다.
#     `--drop_surfaces`(기본 True)가 벽/바닥/천장을 뺀다 — 사용자 지시
#     "벽, 바닥 같은건 static target이 안됐으면 좋겠어".
#  ③ **`--gravity_source geocalib`** (d122/d129 는 `auto`). d148 이 신규 600편 사이드카를
#     880/880 채웠으므로 이제 strict 로 갈 수 있다. `auto` 로 두면 사이드카가 없는 씬만
#     조용히 ground RANSAC 으로 떨어져 코퍼스 안에 중력 게이지가 두 종류 생긴다.
#  ④ **`--num_external 0`** — DataDoP 외부 모양(`dd_*`) 을 뺀다. 실측(한 씬):
#     라우팅 preset 이 29 → **14** 로 준다. 남는 건 8 슬롯 + track 변종 + s_curve 다.
#
# ── vista 와 같아지는 축 (d129 에서 되돌린 것) ────────────────────────────────────
#  · `--tau_ladder 0.10 0.20 0.35 0.60 1.00` — vista d128 의 argparse 기본값 그대로.
#    d129 는 `1.00` 단일단이었다(dd 슬롯 구조 때문). τ 사다리는 **hole 뱅크 크기를 직접
#    곱하지 않는다** — 실측으로 vista TAU 321 → hole 258, d129 TAU 58 → hole 220 이다.
#    사다리가 하는 일은 "이 (anchor,preset) 조합이 **어느 세기에서든** 게이트를 통과하나"를
#    묻는 것이라, 단일단이면 τ=1.0 에서 죽는 조합이 통째로 사라진다.
#  · `--external_shapes` 를 **아예 안 넘긴다** (④ 로 dd 가 0 이므로 넘길 이유가 없다).
#  · 나머지 TAU/FIT 플래그는 `run_k6_d128_shard.sh` 와 문자 단위로 같다.
#
# ── anchor 상한이 걸리는 곳 (d129 주석 그대로) ───────────────────────────────────
# `--max_dynamic_anchors 3 --max_static_anchors 3` 은 **`route_presets.py` 에 준다.**
# 뱅크에 주면 no-op 이다 — route 가 `--nodes ...` 를 명시로 넘기고
# `sample_camera_bank.py:144-151` 이 그걸 받으면 `pick_main_anchors` 를 호출하기 전에
# return 한다. surface drop 은 `schema.py:164` 의 `drop_surfaces=True` 기본값이 건다.
# ② 로 static 노드가 생기므로 이번엔 `--max_static_anchors 3` 이 **처음으로 실제 일을 한다**.
#
# ── GRAPH / CLOUD 는 **돈다** ────────────────────────────────────────────────────
# ①②③ 이 전부 그래프 입력이라 `.graph_d122`/`.cloud_d122` 를 재사용할 수 없다.
# 새 마커 `.graph_d149` / `.cloud_d149`. 옛 그래프는 `scene_graph_pre_d149.json` 으로
# 한 번만 복사해 둔다 (d122 선례). `cloud.npz` 는 백업하지 않는다 — 점 자체는 안 바뀌고
# meta 의 `S` 만 갱신되는데, 게이지 모드(`points_first_cam`)가 d122 와 같으므로 `S` 도 같다.
#
# 사용:
#   bash exec/_legacy/run_dynpose_d149_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
D=/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM
OUT=out_dynpose
TAU="${TAU:-bank_d149}"
BANK="${BANK:-hole_bank_d149}"
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

    # ① scene graph — VLM 재검출 dynamic + 새 static 트리 + geocalib 중력.
    if [ ! -f "$OUT/$VIDEO/.graph_d149" ]; then
        if [ -f "$OUT/$VIDEO/scene_graph.json" ] && [ ! -f "$OUT/$VIDEO/scene_graph_pre_d149.json" ]; then
            cp "$OUT/$VIDEO/scene_graph.json" "$OUT/$VIDEO/scene_graph_pre_d149.json"
        fi
        echo "[s$SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/graph/build_scene_graph.py \
            --video "$VIDEO" --eval_data "$D" --output_root "$OUT" \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            --gravity_source geocalib \
            --no_skip_done --no_overlay --no_topdown \
            > "$LOGDIR/$VIDEO.graph.log" 2>&1
        RC=$?
        echo "[s$SHARD] GRAPH $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.graph_d149"
    fi
    [ -f "$OUT/$VIDEO/.graph_d149" ] || { echo "[s$SHARD] FAIL  $VIDEO graph"; continue; }

    # ② 4D 점군 — 게이지 모드는 d122 와 같지만 마커를 새로 판다 (그래프와 짝을 맞추려고).
    if [ ! -f "$OUT/$VIDEO/.cloud_d149" ]; then
        echo "[s$SHARD] CLOUD $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY -m lbm.cloud \
            --video "$VIDEO" --eval_data "$D" --output_root "$OUT" --no_skip_done \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            > "$LOGDIR/$VIDEO.cloud.log" 2>&1
        RC=$?
        echo "[s$SHARD] CLOUD $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.cloud_d149"
    fi
    [ -f "$OUT/$VIDEO/.cloud_d149" ] || { echo "[s$SHARD] FAIL  $VIDEO cloud"; continue; }

    # ③ preset 라우팅 — **vista 와 다른 유일한 축**. dd 없음(④), anchor 상한 3/3 여기서.
    ARGS=$($PY fit/bank/route_presets.py --video "$VIDEO" --output_root "$OUT" \
        --num_external 0 \
        --max_dynamic_anchors "$MAXDYN" --max_static_anchors "$MAXSTAT" --emit args \
        --out "$OUT/$VIDEO/preset_route_d149.json" 2>"$LOGDIR/$VIDEO.route.err" | tail -1)
    if [ -z "$ARGS" ]; then echo "[s$SHARD] FAIL  $VIDEO route"; continue; fi

    # ④ τ 뱅크 — vista d128 과 문자 단위로 같은 플래그 + `$ARGS` + eval_data/output_root.
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ] && [ ! -f "$OUT/$VIDEO/$TAU/skipped.json" ]; then
        echo "[s$SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py \
            --video "$VIDEO" --eval_data "$D" --output_root "$OUT" --bank_dir "$TAU" \
            $ARGS --tau_ladder 0.10 0.20 0.35 0.60 1.00 \
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

    # ⑤ hole 사다리 재적합 — vista d128 과 문자 단위로 같다.
    echo "[s$SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
        --video "$VIDEO" --eval_data "$D" --output_root "$OUT" \
        --bank_dir "$BANK" --tau_bank_dir "$TAU" \
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

    # ⑥ emit (canonical).
    if [ $RC -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/emit_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
            > "$LOGDIR/$VIDEO.emit.log" 2>&1
        N=$($PY -c "import json;print(len(json.load(open('$OUT/$VIDEO/$BANK/bank.json'))['variants']))" 2>/dev/null)
        echo "[s$SHARD] EMIT  $VIDEO rc=$? n=$N total=$(( $(date +%s)-T0 ))s  $(date +%H:%M:%S)"
    fi
done < "$LIST"
echo "[s$SHARD] ALL DONE $(date +%H:%M:%S)"
