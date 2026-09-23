#!/bin/bash
# 로컬 Qwen3-VL 서버 (LBM-Lite 5단계의 VLM 백엔드).
# 원본은 DynamicVerse `dynamicgen/scripts/local_server.sh` — 바꾼 것은 셋뿐이다:
#   1) GPU 6 -> 0  (CLAUDE.md 의 "GPU 5~7 사용 금지"; D169 이후 단계는 4 도 뺀다 → 0~3)
#   2) --host 0.0.0.0 -> 127.0.0.1  (루프백만. 외부에 열 이유가 없다)
#   3) --allowed-local-media-path 를 CinemaTraj out/ 으로  (board PNG 를 파일 경로로 넘길 수 있게)
# vllm 은 env `vllm` 에만 있다 (vista4d / da3 / dynamicverse 전부 없음).
#
# 사용:
#   screen -dmS vlm bash exec/serve_qwen3vl.sh
#   curl -s 127.0.0.1:22002/v1/models
set -e
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=${LBM_VLM_GPU:-0}
# LBM_VLM_TP — tensor parallel 장 수. 기본 1 은 옛 동작 그대로(단편 LBM 데모용).
# D169 처럼 1만 편 배치를 돌릴 때만 `LBM_VLM_GPU=0,1,2,3 LBM_VLM_TP=4` 로 올린다.
# CUDA_VISIBLE_DEVICES 의 장 수와 반드시 같아야 한다 (다르면 vllm 이 기동 중 죽는다).
LBM_VLM_TP=${LBM_VLM_TP:-1}
# env 를 activate 하지 않고 절대경로 python 만 부르면 `ninja` 가 PATH 에 없다 →
# flashinfer 가 sampling 커널을 JIT 빌드하다 `FileNotFoundError: 'ninja'` 로 죽는다 (실측).
export PATH=/data1/cympyc1785/miniconda3/envs/vllm/bin:$PATH
DVROOT=/data1/cympyc1785/pipeline/DynamicVerse
CINEMATRAJ=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
cd "$DVROOT/dynamicgen"
exec /data1/cympyc1785/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$DVROOT/preprocess/pretrained/Qwen3-VL-30B-A3B-Instruct" \
  --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --tensor-parallel-size "$LBM_VLM_TP" \
  --mm-encoder-tp-mode data \
  --host 127.0.0.1 \
  --port ${LBM_VLM_PORT:-22002} \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --distributed-executor-backend mp \
  --allowed-local-media-path "$CINEMATRAJ/out"
