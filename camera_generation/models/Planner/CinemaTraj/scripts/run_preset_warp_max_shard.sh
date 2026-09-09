#!/bin/bash
# vista 전 씬 × **preset 별 사다리 최대단** depth-warp 릴 샤드 러너.
#
# 왜: 사용자 지시 "vista scene 마다 preset(가장 많이 움직이는거) depth warp 시각화 영상 만들어서
# 한 폴더에". 앞서 friends-restaurant / car-roundabout 두 편에만 손으로 돌렸던 것을 51편으로
# 넓힌다. 손으로 돌릴 때 **사다리 최소단(`c[0]`)을 집는 버그**가 있었으므로(사용자 지적
# "이동량들이 작아진 것 같은데") 여기서는 `target_hole` 오름차순 정렬 후 **`[-1]`** 을 집는다.
#
# anchor 는 `dyn_0` 을 우선하되 없으면 solved 행이 가장 많은 anchor 를 쓴다 — vista 51편 중
# 동적 track 이 없어 `dyn_0` 이 아예 없는 씬이 있다. anchor 를 씬마다 섞으면 격자 사이 차이에
# preset 효과와 anchor 효과가 엉키므로 **씬 안에서는 anchor 하나로 고정**한다.
#
# `render_bank_videos.py` 는 결과를 `<bank_dir>/<--name>` 에 쓴다 (out 인자가 없다). 한 폴더에
# 모아 달라는 요청이라 렌더 후 `$OUTDIR` 로 **복사**한다 (뱅크 폴더에도 원본을 남긴다).
#
# env: `vista4d` (렌더러가 GPU 를 쓴다)
#
# 사용:
#   bash scripts/run_preset_warp_max_shard.sh <gpu> <shard_id> <num_shards> <out_dir>
#   VIDEOS="camel" bash scripts/run_preset_warp_max_shard.sh 1 0 1 <out_dir>   # 특정 편만
set -u
GPU=$1; SHARD=$2; NSHARD=$3; OUTDIR=$4
BANK="${BANK:-hole_bank_k6_d116}"
#    영상 목록 override. **안 주면 기존 동작 그대로** (뱅크 있는 전량을 훑는다).
#    왜 필요한가: 샤드 분배가 `ls` 순서의 인덱스라 **실행 도중에 뱅크가 하나 늘면 그 뒤가 전부
#    한 칸씩 밀린다**. 실제로 camel 뱅크가 샤드 기동(13:24) 이후 14:02 에 끝나서 어느 샤드에도
#    안 잡혔다 — 세 샤드 전부 rc=0 / ALL DONE 이라 로그로는 안 보인다. 뒤늦게 한 편만 채울 때
#    샤드 번호를 되짚는 대신 이름으로 지목한다.
VIDEOS="${VIDEOS:-}"
#    출력 루트 / eval_data override. **안 주면 `out` + 기본 eval_data 라 기존 동작과 동일**.
#    dynpose·trumans 뱅크도 같은 릴로 보려고 열었다 (`ROOT=out_dynpose EVAL=DynPose-LBM`).
ROOT="${ROOT:-out}"
EVAL="${EVAL:-}"
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
cd "$HERE" || exit 1
mkdir -p "$OUTDIR"

i=0
LIST=${VIDEOS:-$(ls -d "$ROOT"/*/"$BANK"/bank.csv 2>/dev/null | sed "s|$ROOT/||;s|/$BANK/bank.csv||")}
for V in $LIST; do
    if [ $((i % NSHARD)) -ne "$SHARD" ]; then i=$((i + 1)); continue; fi
    i=$((i + 1))
    NAME="${V}__preset_warp_max.mp4"
    if [ -f "$OUTDIR/$NAME" ]; then echo "[s$SHARD] SKIP $V"; continue; fi
    # preset 별 최대단 variant_id 목록. anchor 고정은 위 헤더 참조.
    IDS=$($PY - "$V" "$BANK" "$ROOT" <<'PYEOF'
import csv, sys
from collections import Counter, defaultdict
from os import path
video, bank, root = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(path.join(root, video, bank, "bank.csv"))))
rows = [r for r in rows if r["status"].startswith("solved")]
if not rows:
    sys.exit(0)
counts = Counter(r["anchor_id"] for r in rows)
anchor = "dyn_0" if "dyn_0" in counts else counts.most_common(1)[0][0]
byp = defaultdict(list)
for r in rows:
    if r["anchor_id"] == anchor:
        byp[r["preset"]].append(r)
out = []
for p, c in sorted(byp.items()):
    #    사다리 최대단 = `target_hole` 최대. 동점이면 실제 경로장 `path_len_u` 가 큰 쪽.
    c.sort(key=lambda r: (float(r["target_hole"]), float(r["path_len_u"] or 0.0)))
    out.append(c[-1]["variant_id"])
print(" ".join(out))
PYEOF
)
    if [ -z "$IDS" ]; then echo "[s$SHARD] EMPTY $V (solved 0)"; continue; fi
    #    **뱅크가 구워진 K 규약을 그대로 따라간다.** `render_bank_videos.py --fixed_focal` 은
    #    기본 off 인데(자기 일관성 검사가 소스와 화각을 정확히 맞춰야 해서) 뱅크는
    #    `--fixed_focal` 로 구워졌다. 안 맞추면 pose 는 고정 K 로 풀렸는데 렌더만 프레임별
    #    DA3 K 를 써서 **화각이 떨린다** — snowboard fx 가 1184.99~1262.46 (6.99%) 로 흔들려
    #    화면 가장자리가 최대 14.9 px/frame 움직였다. 궤적과 무관한 떨림이라 preset 을 오독한다.
    FF=$($PY -c "import json,sys;print('--fixed_focal' if json.load(open(sys.argv[1])).get('fixed_focal') else '')" \
         "$ROOT/$V/$BANK/bank.json" 2>/dev/null)
    N=$(echo "$IDS" | wc -w)
    echo "[s$SHARD] RENDER $V  tiles=$N  focal=${FF:-perframe}  $(date +%H:%M:%S)"
    #    ROOT/EVAL 이 기본이면 아래 두 인자는 안 붙어서 예전 커맨드와 문자 그대로 같다.
    EXTRA=""
    [ "$ROOT" != "out" ] && EXTRA="$EXTRA --output_root $ROOT"
    [ -n "$EVAL" ] && EXTRA="$EXTRA --eval_data $EVAL"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/render_bank_videos.py \
        --video "$V" --bank_dir "$BANK" --variant_ids $IDS $EXTRA $FF \
        --with_source --columns 6 --tile_width 400 --tile_height 225 --fps 12 \
        --name "$NAME" > "/data1/cympyc1785/LatentCamVid/tmp/warpmax_$V.log" 2>&1
    RC=$?
    echo "[s$SHARD] RENDER $V rc=$RC  $(date +%H:%M:%S)"
    [ $RC -eq 0 ] && cp "$ROOT/$V/$BANK/$NAME" "$OUTDIR/$NAME"
done
echo "[s$SHARD] ALL DONE $(date +%H:%M:%S)"
