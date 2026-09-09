#!/bin/bash
# **뱅크 재굽기 뒤 표준 확인 절차** — 움직이는 subject 2편 + 안 움직이는 subject 2편을 골라
# preset 별 최대단 depth-warp 릴을 굽고 한 폴더에 모은다.
#
# 사용자 지시 2026-09-05: "앞으로 돌리는거 끝나면 저렇게 dynamic subject 움직이는거 두개정도
# 움직이지 않는거 2개 정도해서 preset별 depth warp 비교영상 만들어서 보여주는거 형식화해줘".
#
# 왜 두 버킷인가: preset 의 실패 모드가 subject 이동 여부로 갈린다. 움직이는 쪽은 `track_*` 가
# 켜져 **추종·조준**을 시험하고, 안 움직이는 쪽은 track 이 `--track_min_drift_u` 에서 잘려나가
# 순수 object-centric 만 남아 **hole·벽 뚫기**를 시험한다. 한쪽만 보면 나머지 절반이 조용히
# 깨진 채로 학습에 들어간다. 근거는 `scripts/pick_warp_sample_scenes.py` 상단 참조.
#
# 새 코드는 표본 선택기 하나뿐이다 — 렌더는 기존 `run_preset_warp_max_shard.sh`
# (preset 별 사다리 **최대단** `[-1]`, anchor 는 씬 안에서 하나로 고정) 를 그대로 쓴다.
#
# 산출물:
#   <out_dir>/<video>__preset_warp_max.mp4     편당 릴 (타일 = preset)
#   <out_dir>/sample.json                      어떤 씬을 왜 골랐는지 (drift / preset 수)
#   <out_dir>/sample_table.txt                 사람이 읽는 근거 표
#
# env: `vista4d` (렌더러가 GPU 를 쓴다). GPU 는 CLAUDE.md 제약대로 0~4 안에서만.
#
# 사용:
#   bash scripts/run_preset_warp_sample.sh <gpu> <bank_dir> <out_dir> [output_root]
# 예:
#   bash scripts/run_preset_warp_sample.sh 4 hole_bank_k6_d128 results/20260905_d128_preset_warp
#   bash scripts/run_preset_warp_sample.sh 4 hole_bank_d129 results/20260906_d129_preset_warp out_dynpose
set -u
GPU=$1; BANK=$2; OUTDIR=$3; ROOT="${4:-out}"
#    표본 수 override — 기본 2+2.
NMOV="${NMOV:-2}"
NSTAT="${NSTAT:-2}"
#    `--eval_data` override. **안 주면 output_root 로 추정**한다 (out→기본 Vista4D-Eval-Data,
#    out_dynpose→DynPose-LBM). 추정을 넣은 이유: `--output_root out_dynpose` 만 주고 eval_data 를
#    빼면 렌더러가 dynpose 씬을 Vista4D 데이터셋에서 찾다가 조용히 다른 소스로 warp 한다.
if [ -z "${EVAL:-}" ]; then
    case "$ROOT" in
        out_dynpose) EVAL=/data1/cympyc1785/data/DynPose-LBM ;;
        *)           EVAL="" ;;
    esac
fi
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
cd "$HERE" || exit 1
mkdir -p "$OUTDIR"

VIDEOS=$($PY scripts/pick_warp_sample_scenes.py \
    --output_root "$ROOT" --bank_dir "$BANK" \
    --num_moving "$NMOV" --num_static "$NSTAT" --table \
    --out "$OUTDIR/sample.json" 2> "$OUTDIR/sample_table.txt")
RC=$?
cat "$OUTDIR/sample_table.txt"
if [ $RC -ne 0 ] || [ -z "$VIDEOS" ]; then
    echo "[sample] 표본 선택 실패 rc=$RC — 위 표 참조"; exit 1
fi
echo "[sample] bank=$BANK root=$ROOT -> $VIDEOS"

#    렌더는 기존 러너. 샤드 1개(0/1)로 순차 실행한다 — 4편이라 나눌 이유가 없고,
#    나누면 GPU 하나에 두 프로세스가 붙어 진행 중인 뱅크 굽기와 3중으로 겹친다.
#    ROOT/EVAL 은 **env 로 넘겨야** 한다 — 러너가 자기 안에서 `${ROOT:-out}` 로 읽는데
#    여기서 assign 만 하면 export 가 안 돼 dynpose 뱅크가 `out` 을 보고 EMPTY 로 떨어진다.
VIDEOS="$VIDEOS" BANK="$BANK" ROOT="$ROOT" EVAL="$EVAL" \
    bash scripts/run_preset_warp_max_shard.sh "$GPU" 0 1 "$OUTDIR"
echo "[sample] ALL DONE $(date +%H:%M:%S)  -> $OUTDIR"
ls -la "$OUTDIR"
