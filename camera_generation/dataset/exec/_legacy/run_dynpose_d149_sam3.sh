#!/bin/bash
# D149 — 기존 279편의 dynamic 인스턴스를 **VLM 명사로 재검출**한다.
#
# 왜 다시 도나 — d145 로 넓힌 뒤 dynpose 코퍼스 안에 탐지 어휘가 두 종류가 됐다:
# 신규 596편은 `extract_nouns_vlm.py` 의 VLM 명사로, 기존 279편은 DynPose-LBM 배포
# annotation 명사로 SAM3 를 돌렸다. 사용자 지시("detection은 VLM 사용한걸로 해주고",
# 2026-09-06)에 따라 880편 전체를 한 어휘로 맞춘다.
#
# `run_dynpose_d145_sam3.sh` 와 다른 곳은 두 줄뿐이다:
#   · `--videos $(cat $LIST)` — 279편만 (d145 는 metadata.csv 전량 + skip_done 으로 신규분만)
#   · `--no_skip_done`        — masks.npz 가 이미 있어도 **덮어쓴다**. 이게 이 스크립트의 본론이다.
#
# 옛 결과는 지우기 전에 백업해 뒀다:
#   /data1/cympyc1785/data/DynPose-LBM/eval_data/seg_instances_bak_pre_d149_20260906
# metadata.csv 의 `dynamic` 열도 이 전에 `extend_dynpose_metadata.py --overwrite_existing`
# 으로 VLM 명사로 갈아야 한다 — SAM3 는 그 열을 읽는다. 순서가 뒤집히면 옛 명사로 재검출해서
# **아무것도 안 바뀐 채 rc=0 으로 끝난다.**
#
# 사용:
#   bash exec/_legacy/run_dynpose_d149_sam3.sh <gpu> <shard_id> <num_shards> <video_list>
set -u

GPU="${1:?gpu}"
SHARD="${2:?shard_id}"
NSHARD="${3:?num_shards}"
LIST="${4:?video_list}"

PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
VG=/data1/cympyc1785/LatentCamVid/video_generation
EVAL=/data1/cympyc1785/data/DynPose-LBM

echo "[d149-sam3] gpu=$GPU shard=$SHARD/$NSHARD n=$(grep -c . "$LIST")  $(date +%F' '%H:%M:%S)"
# shellcheck disable=SC2046
CUDA_VISIBLE_DEVICES="$GPU" $PY "$VG/scripts/sam3_seg_instances.py" \
  --eval_data "$EVAL" \
  --videos $(cat "$LIST") \
  --num_shards "$NSHARD" --shard_id "$SHARD" \
  --no_skip_done
RC=$?
echo "[d149-sam3] rc=$RC shard=$SHARD  $(date +%F' '%H:%M:%S)"
exit "$RC"
