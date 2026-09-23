#!/bin/bash
# D145 — dynpose 나머지 600 scene 의 VLM 명사 추출 드라이버.
#
# 왜 드라이버가 따로 필요한가: `extract_nouns_vlm.py` 는 **끝날 때 한 번만**
# `vlm_nouns.json` 을 쓴다 (그 파일 :215-229). 51편 규모에선 문제가 아니었지만 600편을
# 한 프로세스로 돌리면 590편째 예외 하나에 1~2시간이 통째로 날아간다. 그래서 여기서
# `--merge` 로 batch 를 쪼개 부른다 — 스크립트는 한 줄도 안 고치고, 손실 단위가
# 600편에서 $BATCH 편으로 줄어든다.
#
# resume: 매 batch 전에 이미 json 에 들어간 (video, multi) 를 빼고 남은 것만 넘긴다.
# 그래서 중간에 죽어도 같은 명령을 그대로 다시 실행하면 이어서 간다.
#
# frame_mode 는 **multi 만**. single(frame0 1장) 은 detector 공정비교용 조건이고
# (extract_nouns_vlm.py :16-18), 코퍼스에 실제로 쓸 명사는 움직임이 보여야 dynamic/static
# 을 가를 수 있으므로 0/16/32/48 4장이 맞다.
#
# 사용:
#   screen -dmS d145nouns bash exec/_legacy/run_dynpose_d145_nouns.sh
# 선행: vLLM Qwen3-VL 서버가 22002 에 떠 있어야 한다 (exec/serve_qwen3vl.sh).
set -u

HERE=$(cd "$(dirname "$0")/../.." && pwd)
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
EVAL=/data1/cympyc1785/data/DynPose-LBM         # <eval>/eval_data/recon_and_seg/<video>/video.mp4
LIST="${LIST:-/tmp/scenes_need_vlm_nouns.txt}"  # 한 줄에 video 하나
OUT="${OUT:-$HERE/out_dynpose_nouns}"           # vlm_nouns.json + frames/ + trace/
BATCH="${BATCH:-25}"                            # 손실 단위. 25편 ≈ 3~5분

mkdir -p "$OUT"

while true; do
  # 아직 안 된 것만 추린다. json 이 없으면 전부 남은 것으로 본다.
  $PY - "$LIST" "$OUT/vlm_nouns.json" > /tmp/d145_remaining.txt <<'EOFPY'
import json, sys
from os import path
wanted = [l.strip() for l in open(sys.argv[1]) if l.strip()]
done = set()
if path.isfile(sys.argv[2]):
    for r in json.load(open(sys.argv[2], encoding="utf-8")).get("records", []):
        if r.get("frame_mode") == "multi":
            done.add(r["video"])
for v in wanted:
    if v not in done:
        print(v)
EOFPY
  N=$(wc -l < /tmp/d145_remaining.txt)
  echo "[d145-nouns] remaining $N  $(date +%H:%M:%S)"
  [ "$N" -eq 0 ] && break

  VIDS=$(head -n "$BATCH" /tmp/d145_remaining.txt | tr '\n' ' ')
  $PY "$HERE/fit/graph/extract_nouns_vlm.py" \
    --eval_data "$EVAL" --videos $VIDS --frame_mode multi \
    --output "$OUT" --merge
  RC=$?
  echo "[d145-nouns] batch rc=$RC  $(date +%H:%M:%S)"
  if [ "$RC" -ne 0 ]; then
    # 같은 batch 가 계속 죽으면 무한루프가 된다. rc!=0 이면 멈추고 사람이 본다.
    echo "[d145-nouns] ABORT — batch 가 rc=$RC 로 죽었다"
    exit "$RC"
  fi
done

echo "[d145-nouns] DONE  $OUT/vlm_nouns.json"
