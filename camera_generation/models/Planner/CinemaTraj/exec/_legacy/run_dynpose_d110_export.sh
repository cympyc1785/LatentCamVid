#!/bin/bash
# D110 후처리 체인 — 4단 사다리 뱅크(hole_bank_d110) → latentcam 코퍼스 + dd10 리스트.
#
# 왜 스크립트인가: D107 때 이 체인을 세션 안에서 손으로 돌렸다가 **caption 단계가 아직 굽고
# 있던 샤드를 앞질러서** 2편(00e9f728 62변이 / 015b197d 9변이)이 `captions.json` 없이 export
# 에서 조용히 빠졌다 (`/tmp/d107_export.log` 의 "skipped 2"). 로그에는 경고 한 줄뿐이라
# 71 변이가 사라진 걸 export 표만 보면 못 잡는다. 그래서 (1) 샤드가 전부 끝났는지 먼저 막고
# (2) caption 을 다 돈 뒤 뱅크 수와 captions 수가 같은지 assert 한다.
#
# 사다리만 4단으로 바꾼 재fit 이라 graph/route/τ뱅크는 d107 것을 그대로 쓴다
# (`run_dynpose_d110_shard.sh` 참고). holdout 도 d107 과 **같은 27편** — 코퍼스가 바뀌어도
# scene 분할이 같아야 d107 arm 과 paired 로 읽힌다.
#
# usage: run_dynpose_d110_export.sh [--wait]      # --wait 면 샤드가 끝날 때까지 기다린다
set -eu
CT=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
LC=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HOLE=hole_bank_d110
OUT=/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d110
TESTV=/data1/cympyc1785/LatentCamVid/tmp/d107_test_videos.txt                    # d107 과 동일한 scene holdout 27편
cd $CT

if [ "${1:-}" = "--wait" ]; then
  echo "[wait] 샤드가 전부 끝나기를 기다린다 ..."
  while pgrep -f run_dynpose_d110_shard.sh > /dev/null; do sleep 120; done
  echo "[wait] 샤드 종료 확인 $(date +%H:%M:%S)"
fi
# 막이 하나 더 필요하다: 러너가 죽어도 자식 fit/emit 은 살아남은 전례가 있다 (d99 고아 6개).
while pgrep -f "bank_dir $HOLE" > /dev/null; do echo "[wait] 자식 fit/emit 잔류 ..."; sleep 60; done

NB=$(ls -d out_dynpose/*/$HOLE/bank.json 2>/dev/null | wc -l)
echo "== [1/4] captions  (뱅크 $NB 편)"
$PY fit/caption/build_bank_captions.py --videos all --output_root out_dynpose --bank_dir $HOLE \
  --external_shapes $CT/configs/datadop_shapes.json > /tmp/d110_captions.log 2>&1
NC=$(ls -d out_dynpose/*/$HOLE/captions.json 2>/dev/null | wc -l)
echo "   뱅크 $NB / captions $NC"
[ "$NB" = "$NC" ] && echo "   OK 전량 일치" || { echo "   !! 불일치 — D107 과 같은 누락이다. 중단"; exit 1; }

echo "== [2/4] export -> $OUT"
$PY fit/convert/vista4d_bank_to_dl3dv.py --cine_out $CT/out_dynpose --bank_dir $HOLE \
  --recon_root /data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM/eval_data/recon_and_seg \
  --out_root $OUT --videos all --meta_csv meta_dynpose.csv --chunk_prefix dynpose \
  --seg_list_prefix seg_list_dynpose --test_videos $(cat $TESTV | tr '\n' ' ') \
  --workers 8 > /tmp/d110_export.log 2>&1
tail -8 /tmp/d110_export.log

echo "== [3/4] dd 10% 서브샘플 리스트"
cd $LC && $PY scripts/data/filter_seg_list_by_preset.py --root $OUT \
  --prefix seg_list_dynpose --target_frac_prefix dd_ --target_frac 0.10 --suffix dd10 \
  | tee /tmp/d110_dd10.log

echo "== [4/4] 완료. 다음은 config + smoke (수동)."
