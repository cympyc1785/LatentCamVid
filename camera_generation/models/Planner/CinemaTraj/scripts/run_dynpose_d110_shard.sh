#!/bin/bash
# D110 dynpose 재fit 샤드 러너 — **hole 사다리를 2단 -> 4단으로** (d107 위에서 fit+emit 만).
#
# 왜: 사용자 지시 "dd 10% / preset 90%" 를 dd 서브샘플(D109)로만 만들면 코퍼스가 58% 로
# 준다 ("이러면 개수가 부족하잖아"). 사다리를 k6 원래 4단(0.10/0.20/0.35/0.50)으로 되돌리면
# preset 변이가 ~2배 (6,076 -> ~12,152 dedup 전) 가 되어, dd 를 10% 로 서브샘플해도 총량이
# d107 전량(8.5k)을 넘는다 (~9.7k 예상). dd 도 같이 2배로 늘어 10% 풀의 다양성이 커진다
# (사용자: "비율 유지하려면 dd도 좀 더 추가해도됨").
#
# graph(.graph_d107) / route(preset_route_d107.json) / τ뱅크(bank_d107) 는 **그대로 재사용**
# — 사다리는 fit 단계의 손잡이라 앞 단계 산출물이 안 바뀐다. fit + emit 만 hole_bank_d110 으로.
#
# usage: run_dynpose_d110_shard.sh <GPU> <NUM_SHARDS> <SHARD_ID>
set -u
GPU=$1; NS=$2; SID=$3
D=/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM
CT=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
SH=$CT/configs/datadop_shapes.json
BANK=bank_d107
HOLE=hole_bank_d110
EASE="${EASE:-smooth_kf}"
FAIL=/tmp/dynpose_d110_fail_s$SID.log
cd $CT
for V in $(ls out_dynpose); do
  H=$(( 0x$(printf '%s' "$V" | md5sum | cut -c1-8) % NS ))
  [ "$H" -ne "$SID" ] && continue
  [ -f out_dynpose/$V/$BANK/bank.json ] || continue     # d107 τ뱅크가 없으면 대상 아님
  [ -f out_dynpose/$V/$HOLE/bank.json ]    && continue
  [ -f out_dynpose/$V/$HOLE/skipped.json ] && continue
  T0=$(date +%s); echo "=== [s$SID] $V $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/fit_hole_ladder.py --eval_data $D \
    --output_root out_dynpose --video $V --bank_dir $HOLE --tau_bank_dir $BANK \
    --external_shapes $SH --hole_ladder 0.10 0.20 0.35 0.50 \
    --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" --fixed_focal --deroll \
    --device cuda > /tmp/d110_fit_$V.log 2>&1 || { echo "FAIL fit $V" >> $FAIL; continue; }
  T1=$(date +%s)
  $PY scripts/emit_bank.py --video $V --output_root out_dynpose --bank_dir $HOLE \
    > /tmp/d110_emit_$V.log 2>&1 || echo "FAIL emit $V" >> $FAIL
  N=$($PY -c "import json;print(len(json.load(open('out_dynpose/$V/$HOLE/bank.json'))['variants']))" 2>/dev/null)
  echo "DONE [s$SID] $V n=$N fit=$((T1-T0))s total=$(( $(date +%s)-T0 ))s $(date +%H:%M:%S)"
done
echo "ALL DONE [s$SID] $(date +%H:%M:%S)"
