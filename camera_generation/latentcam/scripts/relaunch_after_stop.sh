#!/bin/bash
# stop_at_epoch.sh 가 어떤 run 을 멈춘 뒤, 같은 screen/GPU 에서 (보통 config 를 바꿔서)
# 자동으로 다시 띄우는 스크립트.
#   사용: scripts/relaunch_after_stop.sh <experiment> <screen> <gpu> <log>
#   예:   scripts/relaunch_after_stop.sh geo_worldtraj train3 2 /tmp/train3_geo_worldtraj_avgscale.log
#
# stop_at_epoch.sh 의 stop_one 과 경합하지 않도록 "프로세스가 사라진 것"만 신호로 쓰고
# (stop_one 은 프로세스가 죽으면 5초 안에 return 한다) GRACE 초를 더 기다린 뒤,
# GPU 메모리가 실제로 반납됐는지까지 확인하고 나서 기동한다 — Ctrl+C 로 죽여도 DataLoader
# worker 가 잠깐 GPU 를 물고 있는 경우가 있다.
#
# 기동 후 stopwatch watchdog 을 재시작한다(RESTART_WATCHDOG=0 으로 끌 수 있음): 기존 watchdog 은
# 멈춘 run 을 done_map 에 박아두므로 새로 띄운 run 은 다시 안 본다.
set -u
EXP=${1:?experiment 이름}; SCR=${2:?screen 이름}; GPU=${3:?gpu id}; LOG=${4:?log 경로}
POLL=${POLL:-60}
GRACE=${GRACE:-60}
RESTART_WATCHDOG=${RESTART_WATCHDOG:-1}
WATCHDOG_TARGET=${WATCHDOG_TARGET:-150}
PY=${PY:-/data1/cympyc1785/miniconda3/envs/latentcam/bin/python}
REPO=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam

# stop_at_epoch.sh 와 같은 이유로 끝을 $ 로 anchor (geo_worldtraj 는 다른 4개의 prefix).
pat="train_latent_cam_dm\.py experiment=${EXP}\$"

echo "[$(date +%H:%M:%S)] 대기: experiment=$EXP 가 멈출 때까지 (poll ${POLL}s)"
while pgrep -f "$pat" >/dev/null; do sleep "$POLL"; done
echo "[$(date +%H:%M:%S)] 종료 감지 -> grace ${GRACE}s"
sleep "$GRACE"

for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null)
  [ -n "$used" ] && [ "$used" -lt 2000 ] && break
  echo "[$(date +%H:%M:%S)] GPU$GPU 아직 ${used}MiB - 대기"
  sleep 20
done

# watchdog 이 이 로그로 epoch 을 읽으므로 이전 로그는 반드시 치워서 새 파일로 시작해야 한다.
[ -f "$LOG" ] && mv "$LOG" "${LOG%.log}_$(date +%Y%m%d_%H%M%S).log"

screen -S "$SCR" -X quit 2>/dev/null; sleep 2
cd "$REPO/main" || exit 1
screen -dmS "$SCR" bash -c "CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=\"..:.:../data\" $PY train_latent_cam_dm.py experiment=$EXP 2>&1 | tee $LOG"
echo "[$(date +%H:%M:%S)] 재기동: screen=$SCR GPU=$GPU experiment=$EXP log=$LOG"

sleep 90
if pgrep -f "$pat" >/dev/null; then
  echo "[$(date +%H:%M:%S)] RELAUNCH OK: experiment=$EXP 프로세스 확인"
else
  echo "[$(date +%H:%M:%S)] RELAUNCH FAILED: experiment=$EXP 프로세스 없음"; tail -30 "$LOG"; exit 1
fi

if [ "$RESTART_WATCHDOG" = 1 ]; then
  screen -S stopwatch -X quit 2>/dev/null; sleep 2
  screen -dmS stopwatch bash -c "cd $REPO && scripts/stop_at_epoch.sh $WATCHDOG_TARGET >> /tmp/stop_at_epoch.log 2>&1"
  echo "[$(date +%H:%M:%S)] stopwatch watchdog 재시작 (TARGET=$WATCHDOG_TARGET)"
fi
