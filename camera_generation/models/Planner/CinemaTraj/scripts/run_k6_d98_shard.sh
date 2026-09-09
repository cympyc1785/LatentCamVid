#!/bin/bash
# D98 뱅크 샤드 러너 — GeoCalib 중력으로 다시 세운 scene graph 위에서 τ 뱅크 → hole 사다리
# 적합 → emit 을 한 번에. vista 52편(`out/k6_videos.txt`)용.
#
# 왜 `run_k6_d77_shard.sh` 와 별도인가: 바뀐 게 preset 축이 아니라 **중력축**이다.
# `--gravity_source auto` 로 다시 구운 `scene_graph.json` 은 OBB 의 yaw·extent·center 와
# roll=0 기준이 전부 새 up 벡터 위에 앉아 있다 (52편 중 12편이 실질적으로 이동:
# ground RANSAC 과 10° 넘게 어긋난 게 7편, RANSAC 이 애초에 `camera_up_fallback` 이던 게 5편).
# 그래서 anchor 좌표계가 달라졌고 τ 뱅크부터 다시 만들어야 한다 — `fit_hole_ladder.py:520`
# 이 τ 뱅크에 seed 행이 없는 (anchor, preset) 쌍을 조용히 건너뛰므로 재적합만으론 안 된다.
#
# τ 뱅크를 `bank_d98/`, emit 뱅크를 `hole_bank_k6_d98/` 에 쓰는 이유: 한 폴더에 두 규약을
# 섞지 않기 위해서다 (D96/D97 과 같은 규칙). 옛 뱅크는 하나도 안 건드린다.
#
# **preset 은 인자로 안 준다 = 현재 어휘 전량 40종** (`lbm.presets.PRESETS`, zoom 계열 제외).
# D77 의 34종에서 D90 `tilt_up/down`, D93/D94 이름 정리(`static_hold`↔`static_look_at`,
# `track_hold`↔`track_look_at`), D95 `*_look_at`, track arc 4종이 더해진 결과다.
#
# 나머지 인자는 전부 D97 까지의 기본값을 그대로 탄다 — `--tau_ref auto`(D97),
# `--keyframe_ease smooth_kf`(D96), `--preset_tracking`(D93), `--track_dynamic_only`(D77),
# `--follow_gains "0"`(D72). 명시적으로 넘기는 건 뱅크 정체성을 정하는 셋뿐이다
# (`--aim_keyframes 6 --keyframe_aim auto --fixed_focal`).
#
# `--follow_gains` 를 안 준다(기본 "0"): `decode/build_poses.py:472` 는 들어온 gain 이 문자열
# "0" 일 때만 `PRESET_FOLLOW` 를 적용한다. `auto` 를 주면 τ 최소 gain 이 풀려서 `track_*` 이
# 조용히 추종을 멈춘다 — 캡션만 "tracks" 인 궤적이 생긴다.
#
# 정지 rung 은 따로 안 붙인다: `fit_hole_ladder.py --static_rung` 이 D78 부터 기본 켜짐이라
# 처음 굽는 뱅크에는 이미 들어간다 (`run_static_rung_shard.sh` 는 옛 뱅크 소급용이다).
#
# 사용:
#   bash scripts/run_k6_d98_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
TAU=bank_d98
BANK=hole_bank_k6_d98
EASE="${EASE:-smooth_kf}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -eq "$SHARD" ]; then
        if [ -f "out/$VIDEO/$BANK/canonical/canonical.json" ]; then
            echo "[shard $SHARD] SKIP $VIDEO (이미 있음)"
        else
            # ① τ 뱅크. 이미 있으면 건너뛴다 (재시작 안전).
            if [ ! -f "out/$VIDEO/$TAU/bank.json" ]; then
                echo "[shard $SHARD] TAU   $VIDEO  $(date +%H:%M:%S)"
                CUDA_VISIBLE_DEVICES=$GPU $PY scripts/sample_camera_bank.py \
                    --video "$VIDEO" --bank_dir "$TAU" \
                    --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                    --fixed_focal --no_preview \
                    > "$LOGDIR/$VIDEO.tau.log" 2>&1
                echo "[shard $SHARD] TAU   $VIDEO rc=$?  $(date +%H:%M:%S)"
            fi
            # ② hole 사다리 재적합 + ③ emit (canonical).
            echo "[shard $SHARD] START $VIDEO  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$GPU $PY scripts/fit_hole_ladder.py \
                --video "$VIDEO" --bank_dir "$BANK" --tau_bank_dir "$TAU" \
                --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                --fixed_focal \
                > "$LOGDIR/$VIDEO.fit.log" 2>&1
            RC=$?
            echo "[shard $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
            if [ $RC -eq 0 ]; then
                CUDA_VISIBLE_DEVICES=$GPU $PY scripts/emit_bank.py \
                    --video "$VIDEO" --bank_dir "$BANK" \
                    > "$LOGDIR/$VIDEO.emit.log" 2>&1
                echo "[shard $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
            fi
        fi
    fi
    i=$((i + 1))
done < "$LIST"
echo "[shard $SHARD] ALL DONE $(date +%H:%M:%S)"
