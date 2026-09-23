#!/bin/bash
# D107 dynpose 코퍼스 재생성 샤드 러너 — **D84 라우팅 레시피 + d99 세팅**.
#   graph(GeoCalib auto) -> route -> sample_camera_bank(bank_d107)
#     -> fit_hole_ladder(hole_bank_d107, deroll+smooth_kf) -> emit_bank
#
# D84 원본(`/tmp/dynpose_corpus_d84.sh` — /tmp 에만 있었다, 이 파일이 영구본)과 다른 점:
#
# ① **graph 를 GeoCalib 로 다시 짓는다** (`--gravity_source auto --no_skip_done`).
#    기존 out_dynpose graph 280편 실측: camera_up_fallback 149 (53%) / ground_ransac 131.
#    fallback 축은 TRUMANS GT 실측으로 median 10.9° 기울어 있다 — 그 위에 deroll 을 걸면
#    지평선을 틀린 각도로 세운다 (d99 가 TRUMANS graph 를 GT 로 다시 지은 것과 같은 이유).
#    사이드카는 이 스크립트 실행 **전에** `geocalib_gravity.py` 로 만들어 둬야 한다 (env
#    geocalib). auto = 사이드카 있으면 GeoCalib, 없으면 RANSAC (그 영상은 deroll 이 예전과
#    같은 축을 쓴다 — 사이드카 실패 영상을 통째로 버리는 것보다 낫다).
#    옛 graph 는 `scene_graph_pre_d107.json` 으로 한 번만 백업.
#
# ② **fit 에 `--deroll` + `keyframe_ease smooth_kf`** (D84 는 deroll 없음 + smoothstep).
#    vista/trumans d99·d106 과 같은 회전 규약 — 세 코퍼스를 섞어 학습하므로 갈리면 안 된다.
#
# ③ **τ 뱅크에도 aim/ease/deroll 플래그를 명시** (d99 러너와 동일). D84 는 bank 단계에
#    기본값을 썼다.
#
# D84 에서 유지하는 것: 라우팅 (anchor 2 + preset 라우터가 물체 이동 많으면 track_* 포함),
# DataDoP 외부 궤적 4개/anchor (`configs/datadop_shapes.json`), τ 씨앗 1.00
# (tau_saturated 컷 높이 — 0.35/0.60 이면 빠른 anchor 의 track_* 4개가 통째로 빠진다, D72),
# hole 사다리 0.20/0.35 (k6 기본 4단의 아래 절반 — 0.35 한 단이면 hole 0.476 으로 k6 분포
# 0.319 에서 벗어난다), 이름 해시 샤딩, skip 조건 = hole_bank_d107/{bank,skipped}.json.
#
# usage: run_dynpose_d107_shard.sh <GPU> <NUM_SHARDS> <SHARD_ID>
set -u
GPU=$1; NS=$2; SID=$3
D=/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM
CT=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
SH=$CT/configs/datadop_shapes.json
BANK=bank_d107
HOLE=hole_bank_d107
EASE="${EASE:-smooth_kf}"
FAIL=/data1/cympyc1785/LatentCamVid/tmp/dynpose_d107_fail_s$SID.log
cd $CT
for V in $(ls out_dynpose); do
  H=$(( 0x$(printf '%s' "$V" | md5sum | cut -c1-8) % NS ))
  [ "$H" -ne "$SID" ] && continue
  [ -f out_dynpose/$V/cloud.npz ] || { echo "WAIT $V (cloud 없음)" >> $FAIL; continue; }
  [ -f out_dynpose/$V/$HOLE/bank.json ]    && continue
  [ -f out_dynpose/$V/$HOLE/skipped.json ] && continue
  T0=$(date +%s); echo "=== [s$SID] $V $(date +%H:%M:%S)"
  # 0) graph 재빌드 (GeoCalib auto). 마커로 재실행을 막는다.
  if [ ! -f out_dynpose/$V/.graph_d107 ]; then
    if [ -f out_dynpose/$V/scene_graph.json ] && [ ! -f out_dynpose/$V/scene_graph_pre_d107.json ]; then
      cp out_dynpose/$V/scene_graph.json out_dynpose/$V/scene_graph_pre_d107.json
    fi
    CUDA_VISIBLE_DEVICES=$GPU $PY fit/graph/build_scene_graph.py \
      --video $V --eval_data $D --output_root out_dynpose \
      --gravity_source auto --no_skip_done --no_overlay --no_topdown \
      > /tmp/d107_graph_$V.log 2>&1 || { echo "FAIL graph $V" >> $FAIL; continue; }
    touch out_dynpose/$V/.graph_d107
  fi
  # 1) 라우팅. anchor 2개 + DataDoP 4개/anchor. preset 은 합집합으로 넘긴다.
  ARGS=$($PY fit/bank/route_presets.py --video $V --output_root out_dynpose \
    --external_shapes $SH --num_anchors 2 --num_external 4 --emit args \
    --out out_dynpose/$V/preset_route_d107.json 2>/tmp/d107_route_$V.err | tail -1)
  if [ -z "$ARGS" ]; then echo "FAIL route $V" >> $FAIL; continue; fi
  # 2) τ 뱅크 (사다리 1단 = 씨앗; 크기는 hole 사다리가 다시 푼다).
  CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py --eval_data $D \
    --output_root out_dynpose --video $V --bank_dir $BANK --no_preview --device cuda \
    --skip_on_empty --external_shapes $SH $ARGS --tau_ladder 1.00 \
    --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" --fixed_focal --deroll \
    > /tmp/d107_bank_$V.log 2>&1 || { echo "FAIL bank $V" >> $FAIL; continue; }
  if [ -f out_dynpose/$V/$BANK/skipped.json ] && [ ! -f out_dynpose/$V/$BANK/bank.json ]; then
    mkdir -p out_dynpose/$V/$HOLE
    cp out_dynpose/$V/$BANK/skipped.json out_dynpose/$V/$HOLE/skipped.json
    echo "SKIP [s$SID] $V (변이 0)"; continue
  fi
  T1=$(date +%s)
  # 3) hole 사다리 2단 (0.20/0.35) + d99 규약 (deroll + smooth_kf).
  CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py --eval_data $D \
    --output_root out_dynpose --video $V --bank_dir $HOLE --tau_bank_dir $BANK \
    --external_shapes $SH --hole_ladder 0.20 0.35 \
    --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" --fixed_focal --deroll \
    --device cuda > /tmp/d107_fit_$V.log 2>&1 || { echo "FAIL fit $V" >> $FAIL; continue; }
  T2=$(date +%s)
  $PY fit/bank/emit_bank.py --video $V --output_root out_dynpose --bank_dir $HOLE \
    > /tmp/d107_emit_$V.log 2>&1 || echo "FAIL emit $V" >> $FAIL
  N=$($PY -c "import json;print(len(json.load(open('out_dynpose/$V/$HOLE/bank.json'))['variants']))" 2>/dev/null)
  echo "DONE [s$SID] $V n=$N bank=$((T1-T0))s fit=$((T2-T1))s total=$(( $(date +%s)-T0 ))s $(date +%H:%M:%S)"
done
echo "ALL DONE [s$SID] $(date +%H:%M:%S)"
