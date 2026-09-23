#!/bin/bash
# D115 뱅크 샤드 러너 (vista) — fix.log 승인분 4건을 한 번에 반영해 굽는 첫 뱅크.
# 사용자 지시: "재굽기는 다 고치고 나서 한 번만 돌리자" / "기존 학습 arm 은 보존만".
# 그래서 새 폴더(`bank_d115` / `hole_bank_k6_d115`)에만 쓰고 d99 는 하나도 안 건드린다.
#
# d99 대비 바뀐 축은 **6개뿐**이고, 전부 명시적으로 넘긴다 (D105 의 교훈 — 뱅크 정체성을
# argparse 기본값에 맡기면 나중에 기본값이 또 뒤집혔을 때 한 폴더에 두 규약이 섞인다):
#
#  ① **F2 — 씬 단위 `S` 재정의** (`--scene_scale_mode points_first_cam`, 그래프 재굽기).
#     게이트 임계가 전부 `S` 배율이라(`behind_margin_frac·S`, `obb_clear_floor·S`,
#     `min_ground_clear·S`, 가림 `0.02·S`) `S` 정의를 바꾸는 건 임계를 통째로 옮기는 것과 같다.
#     디스크의 그래프 53편은 전부 `scale.mode` 키 자체가 없는 옛 게이지라 새 코드의 가드
#     (`scene_graph.scale.assert_scale_mode`)가 걸린다 — 그래프부터 다시 짓는 이유.
#     `cloud.npz` 도 `S` 를 meta 에 들고 있어(`lbm/render.py:70`) 같이 다시 만든다.
#     옛 그래프는 `scene_graph_pre_d115.json` 으로 한 번만 복사해 둔다.
#  ② **F9 — `fit_tau` 가 sweep 이 아니라 반경을 깎게** (`--orbit_fixed_sweep`).
#     `true_orbit` 은 `R=|t|/θ` 라 SE(3) 로그를 깎으면 반경이 안 줄고 sweep 만 줄어서
#     "orbit" 이 3° 도는 직선이 됐다. sweep 반응 preset 14종에만 자동 적용된다.
#  ③ **F11 — G1 소스 프레임 13 → 49** (`--behind_src_frames 49`). 성긴 샘플이 실제 정적
#     충돌을 놓치고 있었다 (binding 39건이 collision 으로 유입).
#  ④ **orbit sweep 하한 15 → 20** (`--min_sweep_deg 20`). 사용자: "일단 20으로 놔둬보고
#     돌려보고 결정할게".
#
#  ⑤ **D116 — `--collision_time_match` on**. 사용자 지시 "일단 켜줘". 정적 채널은 주어진
#     프레임 전량에서 동적 픽셀을 빼고 보고, 동적 채널은 plan 프레임 f 를 소스 프레임
#     `round(f·(N_src−1)/(N_plan−1))` 하나에만 대조한다. 시간이 어긋난 동적 픽셀을 벽으로
#     세던 것을 없애는 방향이라 순수하게 느슨해진다 (parkour 4/664, snowboard 6/196, 증가 0건).
#     **명시적으로 넘긴다** — argparse 기본값이 D116 에서 True 로 뒤집혔지만 뱅크 정체성을
#     기본값에 맡기지 않는다는 위 규칙은 그대로다.
#  ⑥ **D116 — `--collision_source depth`**. vista 는 in-the-wild 영상이라 `.blend` 부피가
#     없다. mesh 경로는 TRUMANS 전용 (`run_trumans_d115_shard.sh`).
#
# 일부러 **안** 바꾼 것 (대조를 흐리지 않기 위해):
#  · `--tau_ref` 는 `auto` 그대로. 사용자: "3번 tau 기준점을 일단 안바꾸고 먼저 1, 2번 고치면
#    돌려놓고 나중에 바꿔서 비교해보자".
#  · gravity 는 GeoCalib(D98) 기본값 그대로 — vista 52/52 가 이미 GeoCalib 이다.
#  · `--tau_ref auto`(D97) / `--keyframe_ease smooth_kf`(D96) / `--preset_tracking`(D93) /
#    `--track_dynamic_only`(D77) / `--follow_gains "0"`(D72) 전부 기본값. preset 도 안 넘긴다
#    = 현재 어휘 전량.
#
# 사용:
#   bash exec/_legacy/run_k6_d115_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
OUT=out
#    τ 뱅크는 d115 것을 **그대로 공유**한다 — `sample_camera_bank.py` 는 `--measure_behind` 가
#    기본 off 라 G1 을 애초에 안 재고, D116 이 바꾼 건 G1 뿐이다 (`bank.json` 의
#    `collision.source` 가 `"off"` 인 것으로 확인 가능). 다시 구우면 같은 결과에 시간만 쓴다.
TAU=bank_d115
#    사다리는 **새 이름**이다. 게이트 축 2개(`--collision_time_match` on / TRUMANS 는
#    `--collision_source both`)가 바뀌었으므로 d115 와 같은 폴더 이름을 쓰면 한 이름 아래
#    두 규약이 섞인다 (D105). 옛 `hole_bank_k6_d115` 는 depth-only 대조군으로 남긴다.
BANK="${BANK:-hole_bank_k6_d116}"
EASE="${EASE:-smooth_kf}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
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
    # ① scene graph — 새 `S` 게이지로 다시 짓는다. 마커로 재실행을 막는다.
    if [ ! -f "$OUT/$VIDEO/.graph_s115" ]; then
        if [ -f "$OUT/$VIDEO/scene_graph.json" ] && [ ! -f "$OUT/$VIDEO/scene_graph_pre_d115.json" ]; then
            cp "$OUT/$VIDEO/scene_graph.json" "$OUT/$VIDEO/scene_graph_pre_d115.json"
        fi
        echo "[shard $SHARD] GRAPH $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/graph/build_scene_graph.py \
            --video "$VIDEO" --output_root "$OUT" \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            --no_skip_done --no_overlay --no_topdown \
            > "$LOGDIR/$VIDEO.graph.log" 2>&1
        RC=$?
        echo "[shard $SHARD] GRAPH $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.graph_s115"
    fi
    [ -f "$OUT/$VIDEO/.graph_s115" ] || { echo "[shard $SHARD] FAIL  $VIDEO graph"; continue; }
    # ② 4D 점군 — 점 자체는 안 바뀌지만 meta 의 `S` 가 바뀐다 (렌더러가 여기서 읽는다).
    if [ ! -f "$OUT/$VIDEO/.cloud_s115" ]; then
        echo "[shard $SHARD] CLOUD $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY -m lbm.cloud \
            --video "$VIDEO" --output_root "$OUT" --no_skip_done \
            --scene_scale_mode points_first_cam --scene_scale_stride 1 \
            > "$LOGDIR/$VIDEO.cloud.log" 2>&1
        RC=$?
        echo "[shard $SHARD] CLOUD $VIDEO rc=$RC  $(date +%H:%M:%S)"
        [ $RC -eq 0 ] && touch "$OUT/$VIDEO/.cloud_s115"
    fi
    [ -f "$OUT/$VIDEO/.cloud_s115" ] || { echo "[shard $SHARD] FAIL  $VIDEO cloud"; continue; }
    # ③ τ 뱅크.
    if [ ! -f "$OUT/$VIDEO/$TAU/bank.json" ]; then
        echo "[shard $SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$TAU" \
            --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
            --fixed_focal --deroll --no_preview \
            --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
            > "$LOGDIR/$VIDEO.tau.log" 2>&1
        echo "[shard $SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
    [ -f "$OUT/$VIDEO/$TAU/bank.json" ] || { echo "[shard $SHARD] FAIL  $VIDEO tau"; continue; }
    # ④ hole 사다리 재적합 + ⑤ emit (canonical).
    echo "[shard $SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
        --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" --tau_bank_dir "$TAU" \
        --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
        --fixed_focal --deroll \
        --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
        --collision_time_match --collision_source depth \
        > "$LOGDIR/$VIDEO.fit.log" 2>&1
    RC=$?
    echo "[shard $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
    if [ $RC -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/emit_bank.py \
            --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" \
            > "$LOGDIR/$VIDEO.emit.log" 2>&1
        echo "[shard $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
    fi
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
