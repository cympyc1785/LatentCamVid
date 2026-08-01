#!/bin/bash
# 학습을 지정 epoch 에서 멈추는 watchdog.
#   사용: scripts/stop_at_epoch.sh <TARGET_EPOCH> [screen:log:label ...]
# 인자를 안 주면 현재 돌고 있는 5개 run 기본값을 쓴다.
#
# 정지 절차는 반드시 Ctrl+C 먼저 (그냥 kill 하면 DataLoader worker 가 고아가 되어 GPU 를 물고 있음):
#   screen -X stuff $'\003'  ->  최대 STOP_WAIT 초 대기  ->  그래도 살아있으면 SIGTERM  ->  SIGKILL
#
# epoch 판정: tqdm 진행줄 '^Epoch N | Loss ...' 의 N. N >= TARGET 이 되는 순간 정지하므로
# epoch 0..TARGET-1 (= TARGET 개) 은 검증/체크포인트까지 끝난 상태다.

set -u
TARGET=${1:-150}; shift || true
POLL=${POLL:-120}          # 폴링 주기(초)
STOP_WAIT=${STOP_WAIT:-180} # Ctrl+C 후 정상 종료를 기다리는 최대 초

if [ $# -eq 0 ]; then
  set -- \
    train1:/tmp/train1_ctxlonger135.log:A_ctxlonger135 \
    train2:/tmp/train2_lagernvsnorm.log:B_lagernvsnorm \
    train3:/tmp/train3_geo_worldtraj_avgscale.log:baseline_worldtraj \
    train4:/tmp/train4_camembed.log:camembed \
    train5:/tmp/train5_decoupled.log:decoupled
fi
JOBS=("$@")

cur_epoch() {  # $1=log
  tr '\r' '\n' < "$1" 2>/dev/null | grep -oE '^Epoch [0-9]+' | tail -1 | grep -oE '[0-9]+'
}

# !! 패턴은 반드시 끝을 $ 로 anchor 할 것. experiment=geo_worldtraj 는
# geo_worldtraj_camembed / _ctxlonger135 / _lagernvsnorm / _decoupled 의 prefix 라
# anchor 없이 pkill 하면 baseline 하나 멈추려다 5개를 전부 죽인다.
pat_for() { printf 'train_latent_cam_dm\\.py experiment=%s$' "$1"; }

stop_one() {   # $1=screen $2=label $3=exp
  local scr=$1 lab=$2 pat i; pat=$(pat_for "$3")
  echo "[$(date +%H:%M:%S)] STOP $lab ($scr, experiment=$3) -> Ctrl+C"
  screen -S "$scr" -X stuff $'\003' 2>/dev/null
  for ((i=0; i<STOP_WAIT; i+=5)); do
    sleep 5
    pgrep -f "$pat" >/dev/null || { echo "[$(date +%H:%M:%S)] $lab 정상 종료"; return 0; }
  done
  echo "[$(date +%H:%M:%S)] $lab Ctrl+C 후에도 생존 -> SIGTERM"
  pkill -f "$pat"; sleep 20
  pgrep -f "$pat" >/dev/null && { echo "[$(date +%H:%M:%S)] $lab SIGKILL"; pkill -9 -f "$pat"; }
  echo "[$(date +%H:%M:%S)] $lab 종료 완료"
}

echo "[$(date +%H:%M:%S)] watchdog 시작: TARGET=$TARGET, poll ${POLL}s, 대상 ${#JOBS[@]}개"
declare -A done_map
while :; do
  alldone=1
  for j in "${JOBS[@]}"; do
    IFS=: read -r scr log lab <<< "$j"
    [ "${done_map[$lab]:-}" = 1 ] && continue
    # 같은 이름의 죽은 screen 항목이 남아있을 수 있으므로 experiment= 를 실제로 가진 줄만 본다
    exp=$(ps -ef | grep "SCREEN -dmS $scr " | grep -oE 'experiment=[a-z0-9_]+' | head -1 | cut -d= -f2)
    if [ -z "$exp" ] || ! pgrep -f "$(pat_for "$exp")" >/dev/null; then
      echo "[$(date +%H:%M:%S)] $lab 이미 종료됨(프로세스 없음) - 건너뜀"; done_map[$lab]=1; continue
    fi
    ep=$(cur_epoch "$log")
    if [ -n "$ep" ] && [ "$ep" -ge "$TARGET" ]; then
      stop_one "$scr" "$lab" "$exp"
      done_map[$lab]=1
    else
      alldone=0
    fi
  done
  [ "$alldone" = 1 ] && break
  sleep "$POLL"
done
echo "[$(date +%H:%M:%S)] ALL STOPPED at epoch >= $TARGET"
