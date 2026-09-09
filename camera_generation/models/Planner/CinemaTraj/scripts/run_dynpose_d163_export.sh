#!/bin/bash
# D163 dynpose 캡션 + export 드라이버 — **D157 뱅크(굽는 중)** 를 부분 export 한다.
# `run_dynpose_d129_export.sh` 의 계승본이고 달라진 곳은 세 가지뿐이다.
#
# ① **샤드를 기다리지 않는다.** d129/d147 판본은 `pgrep -f "bank_dir $HOLE"` 로 굽기가 전부
#    끝날 때까지 막았다. D157 은 875편을 4샤드로 굽는 중이고 지금 184편만 bank.json 이 있다
#    (2026-09-07 23시). 사용자 지시가 "지금 fit 된 것으로 먼저 학습" 이므로 **현재 완료분만**
#    캡션·export 하고, 나머지는 나중에 같은 명령을 다시 돌려 증분으로 붙인다
#    (`--skip_done` 기본값이 이미 있는 씬을 건너뛴다). 씬 단위로 독립이라 증분이 안전하다.
#    → 그래서 영상 목록을 **실행 시점의 bank.json 글롭**에서 뽑고 그 목록을 파일로 남긴다.
#
# ② 캡션 게이트 기본값을 그대로 쓴다. `build_bank_captions.py:943 --framing_min_in_frame 0.85`
#    는 D143/D147 에서 들어온 값이고 이제 argparse 기본값이다 — d137 캡션(게이트 없음)과 달리
#    "프레이밍 약속 못 지키는 변이에서 framing 절 제거" 가 처음부터 걸린다. d147 처럼 별도
#    `--out_name` 을 줄 이유가 없어서 `captions.json` 그대로 쓴다 (뱅크 dir 자체가 d157).
#
# ③ dd10 서브샘플을 **안 만든다**. 이번 arm 은 `dolly_in_look_at` 단일 preset 이라 `dd_*`
#    (DataDoP) 행이 애초에 안 들어온다. 대신 마지막에 `filter_seg_list_by_preset.py` 로
#    dolly_in_look_at 만 남긴 `_dilo_{train,test}.txt` 를 만든다.
#
# **뱅크는 손대지 않는다.** 굽기는 계속 돌고 있고 이 스크립트는 읽기만 한다.
#
# 사용:
#   bash scripts/run_dynpose_d163_export.sh              # 캡션 -> export -> dilo 리스트
#   STAGE=caps  bash scripts/run_dynpose_d163_export.sh  # 캡션까지만
#   STAGE=list  bash scripts/run_dynpose_d163_export.sh  # 이미 export 된 root 에서 리스트만
set -u
CT=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
LC=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
TMP=/data1/cympyc1785/LatentCamVid/tmp/d163
HOLE="${HOLE:-hole_bank_d157}"
OUT="${OUT:-/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d157}"
DROP="${DROP-clamped_low}"          # d137 과 같은 필터 (정지 카메라인데 캡션은 dolly_in)
STAGE="${STAGE:-all}"
PRESET="${PRESET:-dolly_in_look_at}"
TESTV=$CT/configs/dynpose_holdout_scenes.txt
SH=$CT/configs/datadop_shapes.json
mkdir -p $TMP
cd $CT || exit 1

VIDS_FILE=$TMP/scenes_banked.txt
ls -d out_dynpose/*/$HOLE/bank.json 2>/dev/null | cut -d/ -f2 | sort > $VIDS_FILE
NB=$(wc -l < $VIDS_FILE)
echo "== 대상 씬 $NB 편 (목록 $VIDS_FILE, $(date +%F' '%T) 기준 완료분)"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "caps" ]; then
  echo "== [1/4] captions"
  # `--videos all` 금지 (FIX-D129-a): metadata.csv 기본값이 Vista4D 것이라 조용히 0편이 된다.
  $PY scripts/build_bank_captions.py --videos $(cat $VIDS_FILE) --output_root out_dynpose \
    --bank_dir $HOLE --external_shapes $SH > $TMP/captions.log 2>&1 \
    || { echo "   !! caption 실패 — $TMP/captions.log"; tail -20 $TMP/captions.log; exit 1; }
  NC=$(ls -d out_dynpose/*/$HOLE/captions.json 2>/dev/null | wc -l)
  echo "   뱅크 $NB / captions $NC"
  [ "$NC" -ge "$NB" ] || { echo "   !! 누락 (D107 과 같은 증상). 중단"; exit 1; }
  [ "$STAGE" = "caps" ] && { tail -8 $TMP/captions.log; exit 0; }
fi

if [ "$STAGE" = "all" ]; then
  echo "== [2/4] export -> $OUT   (drop_status='$DROP')"
  $PY scripts/vista4d_bank_to_dl3dv.py --cine_out $CT/out_dynpose --bank_dir $HOLE \
    --recon_root /data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM/eval_data/recon_and_seg \
    --out_root $OUT --videos $(cat $VIDS_FILE) --meta_csv meta_dynpose.csv \
    --chunk_prefix dynpose --seg_list_prefix seg_list_dynpose \
    --test_videos $(cat $TESTV | tr '\n' ' ') \
    --drop_status $DROP --workers 8 > $TMP/export.log 2>&1 \
    || { echo "   !! export 실패 — $TMP/export.log"; tail -20 $TMP/export.log; exit 1; }
  tail -12 $TMP/export.log
fi

echo "== [3/4] $PRESET 단일 preset 리스트 (_dilo)"
# preset 문자열은 prompts.json 에 **적힌 그대로** 준다. D84 코퍼스는 `dolly_in` 이라고
# 적혀 있었지만(D90 이전 이름) D157 뱅크는 `dolly_in_look_at` 으로 적힌다 — 아래 검산이 확인.
cd $LC && $PY scripts/data/filter_seg_list_by_preset.py --root $OUT \
  --prefix seg_list_dynpose --presets $PRESET --suffix dilo | tee $TMP/dilo.log
cd $CT

echo "== [4/4] 검산 — 리스트 행수 / scene 수 / preset 단일성"
$PY - "$OUT" <<'EOF'
from json import load
from os import path
from sys import argv
root = argv[1]
for split in ("train", "test"):
    f = path.join(root, f"seg_list_dynpose_dilo_{split}.txt")
    if not path.isfile(f):
        print(f"{split:<8} 없음 {f}"); continue
    lines = [l.strip() for l in open(f) if l.strip()]
    cache, presets = {}, {}
    for line in lines:
        _, scene, key = line.split("/", 2)
        if scene not in cache:
            with open(path.join(root, "dynpose", scene, "da3", "prompts.json"),
                      encoding="utf-8") as fh:
                cache[scene] = load(fh)
        e = cache[scene][key] if isinstance(cache[scene], dict) else cache[scene][int(key)]
        p = e.get("preset", "")
        presets[p] = presets.get(p, 0) + 1
    print(f"{split:<8}{len(lines):>7} 행   scene {len({l.split('/')[1] for l in lines}):>4}"
          f"   preset {presets}")
EOF
echo "완료. 다음은 config + smoke (수동)."
