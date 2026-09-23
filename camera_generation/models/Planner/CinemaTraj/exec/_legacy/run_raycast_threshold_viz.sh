#!/bin/bash
# 레이캐스트 게이트 **임계값 스윕**을 그림으로 — 같은 궤적을 `--min_clearance` 만 바꿔가며 렌더한다.
#
# 왜: 사용자 지시 "거리마다 충돌로 판정하는 frustum 및 ray 를 시각화해서 어떤 수치로 하는게
# 적절할지 알 수 있게". `sweep_collision_margin.py` 가 통과/전체 숫자를 주지만, 숫자만으로는
# "0.35 m 에서 떨어지는 그 궤적이 실제로 벽에 붙어 있나 아니면 책상 모서리를 스치나"를 못 읽는다.
# `trumans_raycast_viz.py` 는 임계에 따라 clearance 광선을 빨강(<임계)/파랑(>=임계)으로 칠하므로,
# **같은 지오메트리를 임계만 바꿔 렌더하면 어느 광선이 어느 값에서 뒤집히는지가 그대로 보인다.**
#
# 궤적 하나당 orbit 을 임계 개수만큼 돈다. 오빗 파라미터가 같으므로 프레임 인덱스가 서로 맞고
# `stack_videos.py` 로 격자에 붙일 수 있다.
#
# env: Blender 내장 python (렌더), 묶기는 vista4d 의 imageio
#
# 사용:
#   bash exec/_legacy/run_raycast_threshold_viz.sh <gpu> <scene_dir> <out_dir> <path_idx...>
#     scene_dir = results/20260901_trumans_preset_blender/<scene>  (raycast_probe_rungs.json +
#                 _raycast_poses_rungs.npz 가 있어야 한다)
set -u
GPU=$1; SCENE_DIR=$2; OUTDIR=$3; shift 3
PATHS=("$@")
#    스윕할 임계. 0.35 는 현행 게이트 값이라 반드시 포함한다.
THRESHOLDS="${THRESHOLDS:-0.10 0.20 0.35 0.50}"
PROBE="${PROBE:-1.5}"
ORBIT="${ORBIT:-24}"
#    오빗·컷어웨이는 전부 실측으로 고른 값이다. 근평면은 `orbit_r - spread*clip_cut` 이므로
#    scale 3.2 + cut 1.5 는 근평면을 **방 안쪽**에 놓아 바닥을 가로지르는 검은 띠와 흰 벽만
#    남겼다 (첫 실측 12편이 전부 그랬다). scale 2.2 + elev 58 은 천장만 걷어내고 위에서
#    내려다보는 컷어웨이가 되어 궤적·사람·광선이 다 보인다.
ORBIT_SCALE="${ORBIT_SCALE:-2.2}"
ORBIT_ELEV="${ORBIT_ELEV:-58}"
CLIP_CUT="${CLIP_CUT:-1.15}"
#    위반 광선은 길이가 곧 위반 크기라 통과 광선(최대 1.5 m)에 묻힌다 — 실측 빨강 2 px vs
#    파랑 353 px. `both` 로 구슬 확대 + 임계 길이 막대를 켠다 (사용자 선택, 2026-09-03).
NEAR_EMPH="${NEAR_EMPH:-both}"
B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
cd "$HERE" || exit 1

AUDIT="$SCENE_DIR/raycast_probe_rungs.json"
POSES_ALL="$SCENE_DIR/_raycast_poses_rungs.npz"
BLEND=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['blend'])" "$AUDIT")
mkdir -p "$OUTDIR/poses"

for IDX in "${PATHS[@]}"; do
    #    (V,49,4,4) 에서 궤적 하나만 떼어 낸다 — viz 는 (F,4,4) 만 받는다.
    $PY - "$POSES_ALL" "$IDX" "$OUTDIR/poses/p${IDX}.npz" <<'PYEOF'
import sys
import numpy as np
src, idx, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
np.savez(dst, cam_c2w=np.load(src)["cam_c2w"][idx])
PYEOF
    for C in $THRESHOLDS; do
        TAG="p${IDX}_c${C}"
        if [ -d "$OUTDIR/$TAG/orbit" ]; then echo "[viz] SKIP $TAG"; continue; fi
        echo "[viz] $TAG  $(date +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=$GPU $B -b "$BLEND" --python viz/trumans_raycast_viz.py -- \
            --poses "$OUTDIR/poses/p${IDX}.npz" --audit "$AUDIT" --out "$OUTDIR/$TAG" \
            --ray_stride 48 --min_clearance "$C" --probe_distance "$PROBE" \
            --orbit_frames "$ORBIT" --orbit_scale "$ORBIT_SCALE" --orbit_elev "$ORBIT_ELEV" \
            --clip_cut "$CLIP_CUT" --near_emphasis "$NEAR_EMPH" \
            --res 800 450 --samples 24 --cdevice GPU > "$OUTDIR/$TAG.log" 2>&1
        echo "[viz] $TAG rc=$?  $(date +%H:%M:%S)"
    done
done
echo "[viz] ALL DONE $(date +%H:%M:%S)"
