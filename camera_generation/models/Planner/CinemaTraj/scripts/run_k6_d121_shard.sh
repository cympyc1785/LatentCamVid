#!/bin/bash
# D121 뱅크 샤드 러너 (vista) — d116 과 **게이트·사다리 축이 완전히 같고**, 새 측정 열 2종만
# 더 얹는다. 사용자 지시 2026-09-03: "이제 vista 먼저 텍스트 다 달아서 검증해줘".
#
# d116 대비 바뀐 것 (전부 측정, 적합에 되먹임 없음):
#  ① `--area_timeline` — pick 프레임별 subject OBB 면적 시퀀스(`subject_area_seq`). 캡션의
#     "medium shot **that widens to** a medium wide shot" 절이 이걸 읽는다. verify 패스가 이미
#     렌더한 프레임의 면적을 버리지 않고 남기는 것뿐이라 추가 렌더 0회.
#  ② `--composition` — anchor 말고 무엇이 화면에 담기나 (`in_frame_ids`/`enter_ids`/`exit_ids`).
#     OBB 8꼭짓점 투영만 쓰므로 **렌더 0회**. 캡션의 "with X also in frame" / "as Y leaves the
#     frame" 절이 이걸 읽는다.
#  둘 다 argparse 기본값이 True 지만 **명시적으로 넘긴다** — 뱅크 정체성을 기본값에 맡기지
#  않는다는 D105 규칙 그대로.
#
# 사다리 결과는 d116 과 **bit-identical 이어야 한다** (측정 열은 이분법에 안 들어간다).
# camel 로 공유 열 대조해서 확인할 것.
#
# 앞 단계는 전부 건너뛴다 — `.graph_s115` / `.cloud_s115` 마커와 `bank_d115` 가 52/52 이미 있다.
# 그래서 이 러너는 d115 러너에서 GRAPH/CLOUD/TAU 블록을 빼고 FIT+EMIT 만 남긴 것이다.
#
# 사용:
#   bash scripts/run_k6_d121_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
OUT=out
TAU=bank_d115
BANK="${BANK:-hole_bank_k6_d121}"
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
    [ -f "$OUT/$VIDEO/$TAU/bank.json" ] || { echo "[shard $SHARD] FAIL  $VIDEO tau 없음"; continue; }
    echo "[shard $SHARD] FIT   $VIDEO  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/fit_hole_ladder.py \
        --video "$VIDEO" --output_root "$OUT" --bank_dir "$BANK" --tau_bank_dir "$TAU" \
        --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
        --fixed_focal --deroll \
        --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
        --collision_time_match --collision_source depth \
        --area_timeline --composition \
        --composition_min_area 0.004 --composition_max_nodes 3 \
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
