#!/bin/bash
# D148 dynpose GeoCalib 중력 사이드카 — 신규 600편.
#
# 왜 지금인가 — 사용자 지시 "vista 에서 했던 방식을 최대한 dynpose 에 그대로 적용" 의 ② 항목이다.
# vista 는 `build_scene_graph.py --gravity_source geocalib`(argparse 기본값, strict — 사이드카가
# 없으면 assert 로 죽는다)로 굽고, dynpose 는 `--gravity_source auto`(사이드카 없으면 조용히
# ground RANSAC 으로 떨어진다)로 구웠다. 기존 280편은 사이드카가 전부 있어서 실제로는 `geocalib`
# 로 풀렸으므로 결과가 같았지만, 신규 600편은 사이드카가 **0개**라 `auto` 로 구우면 그 600편만
# ground RANSAC 이 된다 — 코퍼스 절반이 다른 중력 게이지를 쓰게 된다. 중력축이 틀어지면 OBB
# yaw/extent/center 와 roll=0 기준이 전부 같이 틀어진다 (geocalib_gravity.py 의 docstring 실측:
# vista 53편 중 15편(28%)이 RANSAC 에서 `camera_up_fallback` 으로 떨어졌고, 통과한 38편에도
# 벽/책상 상판을 지면으로 문 게 여럿).
#
# 이 스크립트는 **사이드카만 만든다**. 그래프 재굽기는 D145 체인의 `run_dynpose_d122_shard.sh`
# 가 하고, 그때 `--gravity_source geocalib` 로 바꿔 부른다.
#
# env 는 `geocalib` 전용이다 — kornia 가 필요한데 vista4d/da3 에 없다. cv2 로 video.mp4 만 읽으므로
# Vista4D 로더는 안 쓴다.
#
# `--skip_done` 은 argparse 기본값 True 라 재실행하면 이어서 간다. 목록은 호출 전에
# `recon_and_seg` 전량에서 이미 사이드카가 있는 편을 뺀 것을 넘긴다.
#
# 사용:
#   bash exec/_legacy/run_dynpose_d148_geocalib.sh <gpu> <shard_id> <num_shards> <video_list>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
PY=/data1/cympyc1785/miniconda3/envs/geocalib/bin/python
D=/data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM
cd "$HERE" || exit 1

CUDA_VISIBLE_DEVICES=$GPU $PY fit/ingest/geocalib_gravity.py \
    --eval_data "$D" --output_root out_dynpose \
    --video_list "$LIST" \
    --num_shards "$NSHARD" --shard_id "$SHARD"
echo "[geo s$SHARD] ALL DONE rc=$? $(date +%H:%M:%S)"
