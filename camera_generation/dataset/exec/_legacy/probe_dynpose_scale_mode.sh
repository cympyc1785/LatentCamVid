#!/bin/bash
# D122-probe — dynpose 를 `frame0_ray`(옛 게이지) 로 그대로 갈지, `points_first_cam`(vista d121
# 과 같은 게이지) 으로 갈아탈지 **6편 실측으로 정한다** (2026-09-04 사용자 지시).
#
# 왜 그냥 못 정하나: 게이트 임계가 전부 S 배율(`behind_margin_frac·S`, `obb_clear_floor·S`,
# 가림 `0.02·S`)이라 S 정의를 바꾸면 임계가 **씬마다 다른 배율로** 옮겨간다. 실측
# `r_pts_first = S_pts_first / S_f0` 이 p05 0.560 / p50 1.097 / p95 1.571 로 1.0 양옆에
# 걸쳐 있어서, "전부 느슨해진다"도 "전부 빡세진다"도 아니다.
#
# **arm B 는 재fit 만으로 안 된다.** τ 뱅크(`bank_d107/bank.json`)가 `S` 를 통째로 싣고 있고
# (`S: 0.4363...`), `sample_camera_bank.py` 도 `assert_scale_mode` 를 부른다. 즉 게이지를 바꾸면
# graph -> route -> τ뱅크 -> fit **전 체인**을 다시 돌려야 한다. 그래서 arm B 는 D107 러너를
# 통째로 재현하되 `--scene_scale_mode points_first_cam` 만 얹는다.
#
# 원본을 안 건드리려고 arm B 는 `out_dynpose_probe/<V>/` 에서 돈다 — cloud.npz 와 geocalib
# 사이드카는 심링크(재계산 비쌈), graph/route/bank/fit 은 새로 짓는다.
#
# 대조 항목: variant 수 · 사다리 도달단(target_hole) · path_len_u · tau_max · 탈락 사유 분포.
#
# usage: probe_dynpose_scale_mode.sh <GPU> <ARM: a|b> [scenes...]
set -u
GPU=$1; ARM=$2; shift 2
D=/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM
CT=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
SH=$CT/configs/datadop_shapes.json
FAIL=/data1/cympyc1785/LatentCamVid/tmp/probe_scale_fail_$ARM.log
cd $CT

# r_pts_first 0.49 ~ 1.64 로 고르게 뽑은 6편 (audit_scene_scale.py 실측, /tmp/dynpose_scale.csv)
DEFAULT_SCENES="015b197d-2bcf-466e-9ae5-95012be884ac 01841725-5a8f-4586-a558-cca8c9b117d2 \
00e9f728-d0c1-4de4-a839-21c325ab9355 016a6379-a0a1-4be9-82e9-5d7c5e37d578 \
019bbbc2-f335-4d83-b5c3-0db7f700e6ba 018ccdd9-2874-4964-b02e-3da64884a2ec"
SCENES="${*:-$DEFAULT_SCENES}"

for V in $SCENES; do
  T0=$(date +%s); echo "=== [$ARM] $V $(date +%H:%M:%S)"

  if [ "$ARM" = "a" ]; then
    # ---- arm A: 옛 게이지 그대로. d110 과 공유 열이 같아야 하는 갈래.
    ROOT=out_dynpose; BANK=bank_d107; HOLE=hole_bank_probeA; LEGACY="--allow_legacy_scale"
  else
    # ---- arm B: points_first_cam. 체인 전체를 probe root 에서 새로 짓는다.
    ROOT=out_dynpose_probe; BANK=bank_probeB; HOLE=hole_bank_probeB; LEGACY=""
    mkdir -p $ROOT/$V
    for F in cloud.npz geocalib_gravity.json; do
      [ -e $ROOT/$V/$F ] || ln -s $CT/out_dynpose/$V/$F $ROOT/$V/$F
    done
    if [ ! -f $ROOT/$V/scene_graph.json ]; then
      CUDA_VISIBLE_DEVICES=$GPU $PY fit/graph/build_scene_graph.py \
        --video $V --eval_data $D --output_root $ROOT \
        --scene_scale_mode points_first_cam \
        --gravity_source auto --no_skip_done --no_overlay --no_topdown \
        > /tmp/probe_graph_${ARM}_$V.log 2>&1 || { echo "FAIL graph $V" >> $FAIL; continue; }
    fi
    # 라우팅은 D107 과 같은 인자 (anchor 2 + DataDoP 4/anchor).
    ARGS=$($PY fit/bank/route_presets.py --video $V --output_root $ROOT \
      --external_shapes $SH --num_anchors 2 --num_external 4 --emit args \
      --out $ROOT/$V/preset_route_probeB.json 2>/tmp/probe_route_${ARM}_$V.err | tail -1)
    [ -z "$ARGS" ] && { echo "FAIL route $V" >> $FAIL; continue; }
    CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/sample_camera_bank.py --eval_data $D \
      --output_root $ROOT --video $V --bank_dir $BANK --no_preview --device cuda \
      --skip_on_empty --external_shapes $SH $ARGS --tau_ladder 1.00 \
      --aim_keyframes 6 --keyframe_aim auto --keyframe_ease smooth_kf --fixed_focal --deroll \
      > /tmp/probe_bank_${ARM}_$V.log 2>&1 || { echo "FAIL bank $V" >> $FAIL; continue; }
    if [ -f $ROOT/$V/$BANK/skipped.json ] && [ ! -f $ROOT/$V/$BANK/bank.json ]; then
      echo "SKIP [$ARM] $V (변이 0)"; continue
    fi
  fi

  [ -f $ROOT/$V/$HOLE/bank.json ] && { echo "SKIP [$ARM] $V (이미 있음)"; continue; }
  T1=$(date +%s)
  # fit 은 D122 세팅 그대로 (4단 사다리 + area_timeline + composition).
  CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py --eval_data $D \
    --output_root $ROOT --video $V --bank_dir $HOLE --tau_bank_dir $BANK \
    --external_shapes $SH --hole_ladder 0.10 0.20 0.35 0.50 \
    --aim_keyframes 6 --keyframe_aim auto --keyframe_ease smooth_kf --fixed_focal --deroll \
    --orbit_fixed_sweep --min_sweep_deg 20 --behind_src_frames 49 \
    --collision_time_match --collision_source depth \
    --area_timeline --composition \
    --composition_min_area 0.004 --composition_max_nodes 3 $LEGACY \
    --device cuda > /tmp/probe_fit_${ARM}_$V.log 2>&1 || { echo "FAIL fit $V" >> $FAIL; continue; }
  T2=$(date +%s)
  $PY fit/bank/emit_bank.py --video $V --output_root $ROOT --bank_dir $HOLE \
    > /tmp/probe_emit_${ARM}_$V.log 2>&1 || echo "FAIL emit $V" >> $FAIL
  N=$($PY -c "import json;print(len(json.load(open('$ROOT/$V/$HOLE/bank.json'))['variants']))" 2>/dev/null)
  echo "DONE [$ARM] $V n=$N pre=$((T1-T0))s fit=$((T2-T1))s total=$(( $(date +%s)-T0 ))s $(date +%H:%M:%S)"
done
echo "ALL DONE [$ARM] $(date +%H:%M:%S)"
