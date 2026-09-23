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
# 깨진 채로 학습에 들어간다. 근거는 `eval/pick_warp_sample_scenes.py` 상단 참조.
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
#   bash exec/_legacy/run_preset_warp_sample.sh <gpu> <bank_dir> <out_dir> [output_root]
# 예:
#   bash exec/_legacy/run_preset_warp_sample.sh 4 hole_bank_k6_d128 results/20260905_d128_preset_warp
#   bash exec/_legacy/run_preset_warp_sample.sh 4 hole_bank_d129 results/20260906_d129_preset_warp out_dynpose
set -u
GPU=$1; BANK=$2; OUTDIR=$3; ROOT="${4:-out}"
#    표본 수 override — 기본 2+2.
NMOV="${NMOV:-2}"
NSTAT="${NSTAT:-2}"
#    후보 행 기준. 기본 `0` = 예전 그대로 `status=solved`. 어휘가 정지 preset 위주라
#    solved 가 0행인 뱅크(d192 의 `track_look_at`)는 `PICKED=1` 로 `picked` 열을 쓴다.
#    선택기와 렌더러 **양쪽에 같은 값**이 가야 한다 — 한쪽만 바꾸면 표본을 고른 모집단과
#    실제로 찍히는 변이의 모집단이 어긋난다.
PICKED="${PICKED:-0}"
PICK_FLAG=""
[ "$PICKED" = "1" ] && PICK_FLAG="--picked_only"
#    버킷 기준. 기본 `drift` = 예전 동작. `preset` 은 `track_` preset 유무로 가른다
#    (정적 노드도 카메라 때문에 drift 가 크게 찍히므로, 어휘가 두 종뿐인 뱅크는 이쪽이 맞다).
BUCKET_BY="${BUCKET_BY:-drift}"
#    `--eval_data` override. **안 주면 표본 씬이 실제로 있는 루트를 골라** 쓴다 (아래 EVAL 탐색).
#    추정이 필요한 이유: `--output_root out_dynpose` 만 주고 eval_data 를 빼면 렌더러가 dynpose
#    씬을 Vista4D 데이터셋에서 찾다가 조용히 다른 소스로 warp 한다.
#
#    왜 `case "$ROOT"` 한 줄로는 안 되나: `out_dynpose` 한 출력 루트 밑에 **코퍼스가 둘** 있다 —
#    DynPose-LBM(880편, d129~d157) 과 DynPose-100K(10,346편, d172 이후). ROOT 만 보고 LBM 으로
#    못 박아서 d185 릴 4편이 `ValueError: Could not open video file: .../DynPose-LBM/eval_data/
#    recon_and_seg/<video>/video.mp4` 로 죽었다. 그래서 **표본 첫 편이 어느 루트에 있는지**로
#    고른다 (표본은 한 뱅크에서 나오므로 코퍼스가 섞이지 않는다).
EVAL_CANDS="/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K /data1/cympyc1785/data/DynPose-LBM"
EVAL_GIVEN="${EVAL:-}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
cd "$HERE" || exit 1
mkdir -p "$OUTDIR"

# shellcheck disable=SC2086
VIDEOS=$($PY eval/pick_warp_sample_scenes.py \
    --output_root "$ROOT" --bank_dir "$BANK" $PICK_FLAG --bucket_by "$BUCKET_BY" \
    --num_moving "$NMOV" --num_static "$NSTAT" --table \
    --out "$OUTDIR/sample.json" 2> "$OUTDIR/sample_table.txt")
RC=$?
cat "$OUTDIR/sample_table.txt"
if [ $RC -ne 0 ] || [ -z "$VIDEOS" ]; then
    echo "[sample] 표본 선택 실패 rc=$RC — 위 표 참조"; exit 1
fi
echo "[sample] bank=$BANK root=$ROOT -> $VIDEOS"

#    EVAL 탐색 (위 헤더 참조). 명시로 줬으면 그대로, `out` 이면 빈 값 = 렌더러 기본값
#    (Vista4D-Eval-Data) 이라 **예전 커맨드와 문자 그대로 같다**.
EVAL="$EVAL_GIVEN"
if [ -z "$EVAL" ] && [ "$ROOT" != "out" ]; then
    V0=$(echo "$VIDEOS" | awk '{print $1}')
    for C in $EVAL_CANDS; do
        if [ -d "$C/eval_data/recon_and_seg/$V0" ]; then EVAL="$C"; break; fi
    done
    if [ -z "$EVAL" ]; then
        echo "[sample] eval_data 를 못 찾았다 — $V0 이 후보 루트 어디에도 없다: $EVAL_CANDS"
        echo "[sample] EVAL=<경로> 로 명시할 것"; exit 1
    fi
    echo "[sample] eval_data 추정 -> $EVAL  ($V0 기준)"
fi

#    렌더는 기존 러너. 샤드 1개(0/1)로 순차 실행한다 — 4편이라 나눌 이유가 없고,
#    나누면 GPU 하나에 두 프로세스가 붙어 진행 중인 뱅크 굽기와 3중으로 겹친다.
#    ROOT/EVAL 은 **env 로 넘겨야** 한다 — 러너가 자기 안에서 `${ROOT:-out}` 로 읽는데
#    여기서 assign 만 하면 export 가 안 돼 dynpose 뱅크가 `out` 을 보고 EMPTY 로 떨어진다.
#    `CLOUD` 도 그대로 흘린다 — 기본 `auto` 는 cloud.npz 유무로 npz/memory 를 고른다
#    (D178 이후 세대는 cloud.npz 가 없다).
VIDEOS="$VIDEOS" BANK="$BANK" ROOT="$ROOT" EVAL="$EVAL" CLOUD="${CLOUD:-auto}" PICKED="$PICKED" \
    bash exec/_legacy/run_preset_warp_max_shard.sh "$GPU" 0 1 "$OUTDIR"
RC=$?
ls -la "$OUTDIR"
#    렌더가 하나라도 죽으면 여기서 **exit 1**. 예전에는 러너가 rc 를 찍고도 0 으로 끝나서
#    mp4 가 0개인 릴 폴더를 `ALL DONE` 으로 넘겼다 (d185 4/4 실패, 2026-09-12).
if [ $RC -ne 0 ]; then
    echo "[sample] 렌더 실패 rc=$RC — 위 로그 참조.  -> $OUTDIR"; exit 1
fi
echo "[sample] ALL DONE $(date +%H:%M:%S)  -> $OUTDIR"
