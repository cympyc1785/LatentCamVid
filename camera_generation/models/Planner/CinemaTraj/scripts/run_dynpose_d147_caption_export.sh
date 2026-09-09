#!/bin/bash
# D147 dynpose 캡션 + export 드라이버 — d137 뱅크를 **캡션만 고쳐서** 새 루트로 뽑는다.
#
# ═══ 무엇을 고치는가 ══════════════════════════════════════════════════════════════════════
# D143(`run_d143_caption_export.sh`)이 vista 두 뱅크에 적용한 프레이밍 게이트를 dynpose 에도
# 적용한다. 그때 dynpose 를 뺀 이유는 "D137/D141 이 그 코퍼스로 학습 중"이었고, 사용자 지시가
# "지금 학습 안돌리는 데이터셋 먼저 고쳐줘" 였기 때문이다. 그 학습들이 끝났으므로 이제 뽑는다.
#
# 결함 실측 (d137 dd10 코퍼스 10,857 행, 2026-09-06):
#   · `subject_in_frame < 0.85` 인 변이가 4,570 (42.1%)
#   · 그중 **3,225 행이 여전히 framing 절을 달고 있다 = 코퍼스의 29.7%**
# 즉 캡션이 "keeping it in a medium shot" 이라 약속해 놓고 실제 궤적은 대상을 프레임 밖으로
# 내보낸다. `--framing_min_in_frame 0.85`(argparse 기본값)가 그 변이에서 framing/composition
# 절만 뺀다. motion 절과 target 은 그대로 둔다 — `track_*` 의 "tracks {target}" 은 follow_gain
# 1.0 이라 실제로 참이고, 거짓인 건 프레이밍 약속뿐이다.
#
# 배포된 d137 캡션 헤더에는 `framing_on_free`/`framing_min_in_frame` 필드 자체가 없다 —
# D143 이전 코드로 구운 것이고, 게이트가 한 번도 안 걸렸다는 뜻이다.
#
# ═══ 왜 기존 루트를 덮지 않는가 ═══════════════════════════════════════════════════════════
# `latentcam_dynpose_d137` 은 D137/D141 이 학습한 코퍼스다. prompts.json 을 제자리에서 갈면
# 그 런들의 학습 캡션과 재평가 캡션이 어긋나 비교가 무효가 된다. 그래서 새 루트를 판다(~14 GB).
# 접미사 `c147` = caption D147. **뱅크(기하)는 손대지 않는다** — 카메라·이미지·seg_list 는
# 원본과 비트 동일해야 하고 스크립트 끝의 `check_same` 이 그걸 확인한다.
#
# 세로 슬롯(crane/pedestal) 복구는 **이 스크립트에 없다**. 그건 캡션이 아니라 라우팅이고
# (`route_presets.py --vertical_gravity`, 같은 D147), 뱅크를 다시 구워야 나온다 — D145 의
# 880편 굽기에서 들어간다. 여기서 고치는 건 텍스트뿐이다.
#
# ═══ 캡션 인자 ════════════════════════════════════════════════════════════════════════════
# 배포된 d137 캡션을 만든 인자(= `run_dynpose_d129_export.sh:62` 의 기본값들)에서
# `--framing_min_in_frame` 만 켠 것이다. `--framing_min_in_frame 0` 을 주면 배포본과 비트
# 동일하게 나온다. `--videos all` 은 **쓰면 안 된다** (FIX-D129-a) — 목록을 뱅크에서 직접 뽑는다.
#
# 사용:
#   bash scripts/run_dynpose_d147_caption_export.sh
#   DRY=1 bash scripts/run_dynpose_d147_caption_export.sh     # 캡션만 dry-run
set -u
CT=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
LC=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HOLE="${HOLE:-hole_bank_d129}"          # d137 코퍼스는 이 뱅크에서 나왔다 (재굽기 없음)
OLD="${OLD:-/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d137}"
OUT="${OUT:-/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d137c147}"
CAP="${CAP:-captions_d147.json}"
DROP="${DROP-clamped_low}"              # d137 과 같은 필터. 빈 값을 주려면 `DROP=`
TESTV=$CT/configs/dynpose_holdout_scenes.txt
SH=$CT/configs/datadop_shapes.json
cd $CT || exit 1

VIDS=$(ls -d out_dynpose/*/$HOLE/bank.json 2>/dev/null | cut -d/ -f2)
NB=$(printf '%s\n' $VIDS | grep -c . )
echo "== [1/5] captions -> $CAP   (뱅크 $NB 편)"
CAPTION_ARGS=(--videos $VIDS --output_root out_dynpose --bank_dir $HOLE
              --external_shapes $SH --out_name "$CAP")
[ "${DRY:-0}" = "1" ] && CAPTION_ARGS+=(--dry_run)
$PY scripts/build_bank_captions.py "${CAPTION_ARGS[@]}" > /tmp/d147_captions.log 2>&1 \
  || { echo "   !! caption 실패 — /tmp/d147_captions.log"; tail -20 /tmp/d147_captions.log; exit 1; }
if [ "${DRY:-0}" = "1" ]; then echo "   DRY=1 — 여기서 멈춘다"; tail -8 /tmp/d147_captions.log; exit 0; fi
NC=$(ls -d out_dynpose/*/$HOLE/$CAP 2>/dev/null | wc -l)
echo "   뱅크 $NB / captions $NC"
[ "$NB" = "$NC" ] || { echo "   !! 불일치 — D107 과 같은 누락이다. 중단"; exit 1; }

echo "== [2/5] framing 절이 실제로 빠졌는지 검산"
$PY - "$CT/out_dynpose" "$HOLE" "$CAP" <<'EOF'
from glob import glob
from json import load
from os import path
from sys import argv
root, hole, cap = argv[1], argv[2], argv[3]
tot = dropped = 0
for p in sorted(glob(path.join(root, "*", hole, cap))):
    with open(p, encoding="utf-8") as f:
        blob = load(f)
    for c in blob.get("captions", {}).values():
        tot += 1
        dropped += not (c.get("framing") or c.get("framing_nl"))
print(f"  변이 {tot}   framing 절 없음 {dropped} ({dropped / max(tot, 1):.1%})")
EOF

echo "== [3/5] export -> $OUT   (drop_status='$DROP')"
$PY scripts/vista4d_bank_to_dl3dv.py --cine_out $CT/out_dynpose --bank_dir $HOLE \
  --captions_name "$CAP" \
  --recon_root /data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM/eval_data/recon_and_seg \
  --out_root $OUT --videos all --meta_csv meta_dynpose.csv --chunk_prefix dynpose \
  --seg_list_prefix seg_list_dynpose --test_videos $(cat $TESTV | tr '\n' ' ') \
  --drop_status $DROP --workers 8 > /tmp/d147_export.log 2>&1 \
  || { echo "   !! export 실패 — /tmp/d147_export.log"; tail -20 /tmp/d147_export.log; exit 1; }
tail -12 /tmp/d147_export.log

echo "== [4/5] dd 10% 서브샘플 리스트"
cd $LC && $PY scripts/data/filter_seg_list_by_preset.py --root $OUT \
  --prefix seg_list_dynpose --target_frac_prefix dd_ --target_frac 0.10 --suffix dd10 \
  | tee /tmp/d147_dd10.log
cd $CT

echo "== [5/5] 기하 동일성 대조 (캡션만 바꿨으므로 리스트는 글자 그대로 같아야 한다)"
bad=0
for f in seg_list_dynpose_train.txt seg_list_dynpose_test.txt \
         seg_list_dynpose_dd10_train.txt seg_list_dynpose_dd10_test.txt meta_dynpose.csv; do
    if cmp -s "$OUT/$f" "$OLD/$f"; then echo "  OK   $f"
    else echo "  DIFF $f  <-- 기하가 달라졌다. export 인자 확인할 것"; bad=1; fi
done
[ "$bad" = "0" ] && echo "완료. 다음은 config + smoke (수동)." \
                 || { echo "!! 대조 실패"; exit 1; }
