#!/bin/bash
# D78 정지 rung 샤드 러너 — 정지 preset 만 적합해서 기존 k6_d77 뱅크에 붙이고 다시 emit 한다.
#
# 왜 전량 재fit 이 아닌가: `fit_hole_ladder.py` 는 D78 이전에 `knob_kind()` 가 None 인
# `STATIC_PRESETS` 를 preset 목록에서 통째로 뺐다. 그래서 τ 뱅크에는 있는 정지 카메라와
# 순수 추종(`track_hold*`)이 emit 뱅크에 0 행이었다. 52편을 다시 fit 하면 편당 9~44분이
# 다시 들지만, 정지 rung 은 이분법이 없어 (anchor × 4 preset) × 렌더 2회뿐이라 편당 2분
# 미만이다 (camel 실측 14변이 / 182렌더). 두 fit 은 anchor·preset 루프가 독립이고 같은
# τ 뱅크·같은 인자를 쓰므로 붙인 결과 = 전량 재fit 결과이며, `merge_static_rung.py` 가
# 그 전제(`fixed` 블록 · 스칼라)를 대조한 뒤에만 붙인다.
#
# 인자는 `run_k6_d77_shard.sh` 와 **똑같이** 준다 — 다르면 `merge_static_rung.py` 의 호환성
# 검사에서 걸려 붙지 않는다 (`--follow_gains` 도 마찬가지로 안 준다: 기본 "0" 이라야
# `decode/build_poses.py:472` 가 `PRESET_FOLLOW` 를 적용하고 `track_hold` 가 실제로 따라간다).
#
# D96 으로 `--keyframe_ease` 기본값이 `smooth_kf` 가 됐지만 **여기서는 고정 기본값을 쓰면
# 안 된다**. 이 스크립트는 붙일 대상 뱅크가 **이미 구워져 있는** 상황에서만 돌고, 그 뱅크의
# ease 와 다르게 fit 하면 `merge_static_rung.py` 가 `fixed.keyframe_ease` 대조에서 거부한다
# (rc=2). 그래서 ease 는 **붙일 뱅크의 `fixed.keyframe_ease` 에서 읽는다** — smoothstep 으로
# 구운 51편에도, 앞으로 smooth_kf 로 구울 뱅크에도 그대로 붙는다. `EASE_OVERRIDE=` 로 강제 가능.
#
# 사용:
#   bash exec/_legacy/run_static_rung_shard.sh <gpu> <shard_id> <num_shards> <video_list> <log_dir>
set -u
GPU=$1; SHARD=$2; NSHARD=$3; LIST=$4; LOGDIR=$5
TAU=bank_d77
BANK=hole_bank_k6_d77
STATIC=hole_bank_k6_d77_static
# D94 이름. 옛 `*_hold_dont_look` = 지금의 `*_hold`(완전 고정), 옛 `*_hold` = 지금의 `*_look_at`(조준 추종).
PRESETS="static_hold static_look_at track_hold track_look_at"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
mkdir -p "$LOGDIR"
cd "$HERE" || exit 1

i=0
while read -r VIDEO; do
    [ -z "$VIDEO" ] && continue
    if [ $((i % NSHARD)) -eq "$SHARD" ]; then
        # 이미 붙은 뱅크는 건너뛴다. 표식은 `merge_static_rung.py` 가 bank.json 에 남긴다.
        if grep -q '"static_rung_merged"' "out/$VIDEO/$BANK/bank.json" 2>/dev/null; then
            echo "[static $SHARD] SKIP $VIDEO (이미 붙음)"
        elif [ ! -f "out/$VIDEO/$BANK/bank.json" ]; then
            echo "[static $SHARD] NOBANK $VIDEO (k6_d77 뱅크 없음 — 건너뜀)"
        else
            # 붙일 뱅크가 어떤 ease 로 구워졌는지 읽는다 (키가 없으면 D89 이전 뱅크 = smoothstep).
            EASE="${EASE_OVERRIDE:-$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["fixed"].get("keyframe_ease","smoothstep"))' "out/$VIDEO/$BANK/bank.json")}"
            # ① 정지 preset 만 별도 뱅크로 적합.
            if [ ! -f "out/$VIDEO/$STATIC/bank.json" ]; then
                echo "[static $SHARD] FIT   $VIDEO ease=$EASE  $(date +%H:%M:%S)"
                # shellcheck disable=SC2086
                CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/fit_hole_ladder.py \
                    --video "$VIDEO" --bank_dir "$STATIC" --tau_bank_dir "$TAU" \
                    --presets $PRESETS \
                    --aim_keyframes 6 --keyframe_aim auto --keyframe_ease "$EASE" \
                    --fixed_focal \
                    > "$LOGDIR/$VIDEO.static_fit.log" 2>&1
                RC=$?
                echo "[static $SHARD] FIT   $VIDEO rc=$RC  $(date +%H:%M:%S)"
                [ $RC -eq 0 ] || { i=$((i + 1)); continue; }
            fi
            # ② 기존 뱅크에 붙인다 (멱등).
            $PY fit/bank/merge_static_rung.py --video "$VIDEO" \
                --bank_dir "$BANK" --static_dir "$STATIC" \
                > "$LOGDIR/$VIDEO.merge.log" 2>&1
            RC=$?
            echo "[static $SHARD] MERGE $VIDEO rc=$RC  $(date +%H:%M:%S)"
            # ③ canonical 재생성. 붙인 행이 canonical 에 들어가야 하류가 본다.
            if [ $RC -eq 0 ]; then
                CUDA_VISIBLE_DEVICES=$GPU $PY fit/bank/emit_bank.py \
                    --video "$VIDEO" --bank_dir "$BANK" \
                    > "$LOGDIR/$VIDEO.emit.log" 2>&1
                echo "[static $SHARD] EMIT  $VIDEO rc=$?  $(date +%H:%M:%S)"
            fi
        fi
    fi
    i=$((i + 1))
done < "$LIST"
echo "[static $SHARD] ALL DONE $(date +%H:%M:%S)"
