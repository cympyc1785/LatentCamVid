#!/bin/bash
# **벽으로 천천히 다가가며** 어느 지점에서 어떤 clearance 임계에 걸리는지 렌더한다.
#
# 왜: `run_raycast_threshold_viz.sh` 는 궤적을 고정하고 **임계를 스윕**했다. 사용자 지적으로
# 그건 원하는 그림이 아니었다 — "고정된 값을 가지고 천천히 벽으로 이동했을 때 어디쯤에서
# collision, clearance 에 걸리는지 threshold 를 눈으로 보고 싶은거였어". 그래서 여기서는
# **임계 4개를 전부 고정**해 놓고 카메라를 벽 쪽으로 40걸음 전진시키며, 각 임계가 처음 걸리는
# 지점에 색 기둥을 세운다. 기둥 사이 간격이 곧 "0.35 를 0.20 으로 낮추면 몇 cm 더 갈 수 있나"다.
#
# 두 팔(arm)을 돌린다. 게이트 `clearance_of()` 가 ±z 를 포함하기 때문이다:
#   eye  = 실제 카메라 높이(1.33 m). 벽이 바닥보다 먼저 걸린다 → 임계가 벽 거리로 읽힌다.
#   low  = 바닥 0.40 m. **네 임계가 전부 step 0 에 걸린다** — 벽에서 0.81 m 떨어져 있는데도
#          바닥이 먼저 걸려 카메라가 아예 거부된다. audit 의 floor_is_binding_frac 을 그림으로.
# 각 팔은 전경(wide)과 교차점 확대(zoom) 두 벌.
#
# env: Blender 내장 python
#
# 사용:
#   bash scripts/run_approach_viz.sh <gpu> <scene_dir> <out_dir>
set -u
GPU=$1; SCENE_DIR=$2; OUTDIR=$3
STEPS="${STEPS:-40}"
B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
cd "$HERE" || exit 1

AUDIT="$SCENE_DIR/raycast_probe_rungs.json"
BLEND=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['blend'])" "$AUDIT")
mkdir -p "$OUTDIR"

#    공통 인자. world dark 는 blend 의 HDRI 가 렌더를 하얗게 태우기 때문(실측), cut_above 는
#    천장 mesh 를 hide_render 로 걷어내기 때문이다(근평면 컷어웨이로는 천장이 안 없어진다).
COMMON="--audit $AUDIT --steps $STEPS --world dark --world_strength 0.35 \
        --thickness 0.02 --emission 9 --res 960 540 --samples 20 --cdevice GPU"

run() {   # run <tag> <extra args...>
    TAG=$1; shift
    if [ -d "$OUTDIR/$TAG/march" ]; then echo "[approach] SKIP $TAG"; return; fi
    echo "[approach] $TAG  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$GPU $B -b "$BLEND" --python scripts/trumans_approach_viz.py -- \
        $COMMON --out "$OUTDIR/$TAG" "$@" > "$OUTDIR/$TAG.log" 2>&1
    echo "[approach] $TAG rc=$?  $(date +%H:%M:%S)"
}

#    zoom 팔은 교차점 구간만 잡는다 — 5 m 방에서 네 임계가 마지막 0.42 m 안에 몰려 있어
#    전경 프레임에서는 기둥 넷이 몇 px 차이로 붙어 버린다(실측).
ZOOM="--focus crossings --focus_pad 0.30 --obs_elev 34 --obs_scale 2.0 \
      --text_size 0.075 --post_rise 0.35"

run eye
run eye_zoom  $ZOOM
run low       --start_height 0.40
run low_zoom  --start_height 0.40 $ZOOM
echo "[approach] ALL DONE $(date +%H:%M:%S)"
