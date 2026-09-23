#!/bin/bash
# D129 dynpose 후처리 체인 — `hole_bank_d129` 뱅크 → latentcam 코퍼스 + **dd10** 리스트.
# `run_dynpose_d110_export.sh` 의 계승본. 달라진 곳은 세 군데뿐이다:
#   ① 뱅크/출력 이름이 d129 이고 **root 를 새로 판다** (`latentcam_dynpose_d129`).
#      재굽기 뒤 `variant_id` 는 문자열이 같지만 `knob`/pose 는 다르다 — d122 root 를
#      재사용하면 `prompts.json` · seg list · geo 캐시가 **miss 0 으로 조용히 통과**하면서
#      전부 옛 카메라를 가리킨다.
#   ② holdout 을 `/tmp` 가 아니라 리포 안(`configs/dynpose_holdout_scenes.txt`, 27편)에서
#      읽는다. d107/d110/d122 와 **같은 27편**이라야 arm 끼리 paired 로 읽힌다.
#   ③ 마지막에 dd10 서브샘플을 돌리고, `dd_*` 비율이 실제로 10% 근처인지 **확인해서 찍는다**.
#
# 사용자 지시 2026-09-05: "dynpose를 datadop 10% 비율로 해서 데이터 만들어놔줘".
# dd10 = `dd_*` preset 행이 최종 seg list 의 10% 가 되도록 균등 stride 서브샘플
# (RNG 없음, `filter_seg_list_by_preset.py:subsample_to_frac`). d122 는 48.8% 였다.
#
# 앞단 전제: `describe_instances_vlm.py` 는 **다시 안 돈다**. `instance_desc.json`(266편) 은
# 노드 id 로 키잉돼 있고 D129 는 graph 를 안 건드리므로 id/label 이 그대로다 (D122 확인).
#
# usage: run_dynpose_d129_export.sh [--wait]      # --wait 면 샤드가 끝날 때까지 기다린다
set -eu
CT=/data1/cympyc1785/LatentCamVid/camera_generation/dataset
LC=/data1/cympyc1785/LatentCamVid/camera_generation/latentcam
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
HOLE="${HOLE:-hole_bank_d129}"
OUT="${OUT:-/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d129}"
# D137. 내보내지 않을 status 접두사. `clamped_low` = "knob 을 하한까지 밀었는데 **하한에서도**
# 게이트를 위반" (`fit_hole_ladder.py:513-531`) 이라, 궤적은 정지에 가까운데 캡션은 dolly_in
# 이라고 써 있는 행이다. d129 dd10 train 612행 실측: path_len_u 중앙값 0.0330 (solved 0.6900),
# binding 은 collision 340 / obb 114 / approach 87 / ground 43 / elev 26 / hole 2 — 즉 τ 예산이
# 아니라 물리 게이트다. 사용자 지시 2026-09-06 "clamped_low는 정상적인 카메라가 아니다".
# 접두사 매칭이라 `clamped_low+tau_floor` 도 같이 잡힌다. 필터 없는 옛 코퍼스를 재현하려면
# `DROP=` 로 빈 값을 준다 (`:-` 가 아니라 `-` 라서 빈 문자열이 그대로 산다).
DROP="${DROP-clamped_low}"
# 캡션은 뱅크에 딸린 것이라 뱅크가 그대로면 다시 구울 이유가 없다. `SKIP_CAPS=1` 이면 개수만 센다.
SKIP_CAPS="${SKIP_CAPS:-0}"
TESTV=$CT/configs/dynpose_holdout_scenes.txt
SH=$CT/configs/datadop_shapes.json
cd $CT

if [ "${1:-}" = "--wait" ]; then
  echo "[wait] 샤드가 전부 끝나기를 기다린다 ..."
  while pgrep -f run_dynpose_d129_shard.sh > /dev/null; do sleep 120; done
  echo "[wait] 샤드 종료 확인 $(date +%H:%M:%S)"
fi
# 러너가 죽어도 자식 fit/emit 은 살아남은 전례가 있다 (d99 고아 6개).
while pgrep -f "bank_dir $HOLE" > /dev/null; do echo "[wait] 자식 fit/emit 잔류 ..."; sleep 60; done

NB=$(ls -d out_dynpose/*/$HOLE/bank.json 2>/dev/null | wc -l)
NS=$(ls -d out_dynpose/*/$HOLE/skipped.json 2>/dev/null | wc -l)
echo "== [1/4] captions  (뱅크 $NB 편 / 변이0 스킵 $NS 편)"
# **`--videos all` 을 쓰면 안 된다** (FIX-D129-a). `build_bank_captions.py:687` 의 `all` 은
# `--metadata_csv` 로 영상 목록을 만드는데 기본값이 Vista4D 것이라, dynpose UUID 가 한 건도
# 안 맞아 `videos=[]` -> captions 0 으로 **조용히** 끝난다 (러너는 rc=0). d110 계보가 그대로
# 물려받은 버그다. dynpose 는 metadata.csv 에 `prompt` 열이 아예 없어 event 가 전부 빈 문자열
# 이므로 이 파일은 애초에 "영상 목록" 말고는 하는 일이 없다 — 그래서 목록을 **뱅크에서 직접**
# 뽑는다. 덤으로 metadata.csv 에 행이 없는 2편(00e9f728…, 015b197d…; recon 은 있다)도 들어와
# 아래 `NB == NC` 가 267 로 정확히 맞는다 (dynpose metadata 를 넘겨도 265 로 어긋난다).
VIDS=$(ls -d out_dynpose/*/$HOLE/bank.json 2>/dev/null | cut -d/ -f2)
if [ "$SKIP_CAPS" = "1" ]; then
  echo "   SKIP_CAPS=1 — 캡션 재굽기 생략, 개수만 검산한다"
else
  $PY fit/caption/build_bank_captions.py --videos $VIDS --output_root out_dynpose --bank_dir $HOLE \
    --external_shapes $SH > /tmp/d129_captions.log 2>&1
fi
NC=$(ls -d out_dynpose/*/$HOLE/captions.json 2>/dev/null | wc -l)
echo "   뱅크 $NB / captions $NC"
# D107 때 caption 이 아직 굽고 있던 샤드를 앞질러 2편(71 변이)이 조용히 빠졌다.
[ "$NB" = "$NC" ] && echo "   OK 전량 일치" || { echo "   !! 불일치 — D107 과 같은 누락이다. 중단"; exit 1; }

echo "== [2/4] export -> $OUT   (drop_status='$DROP')"
$PY fit/convert/vista4d_bank_to_dl3dv.py --cine_out $CT/out_dynpose --bank_dir $HOLE \
  --recon_root /data1/cympyc1785/LatentCamVid/DATA/DynPose-LBM/eval_data/recon_and_seg \
  --out_root $OUT --videos all --meta_csv meta_dynpose.csv --chunk_prefix dynpose \
  --seg_list_prefix seg_list_dynpose --test_videos $(cat $TESTV | tr '\n' ' ') \
  --drop_status $DROP --workers 8 > /tmp/d129_export.log 2>&1
tail -12 /tmp/d129_export.log

echo "== [3/4] dd 10% 서브샘플 리스트"
cd $LC && $PY scripts/data/filter_seg_list_by_preset.py --root $OUT \
  --prefix seg_list_dynpose --target_frac_prefix dd_ --target_frac 0.10 --suffix dd10 \
  | tee /tmp/d129_dd10.log

echo "== [4/4] 검산 — dd_* 실제 비율"
$PY - "$OUT" <<'EOF'
from json import load
from os import path
from sys import argv
root = argv[1]
for split in ("train", "test"):
    lines = [l.strip() for l in open(path.join(root, f"seg_list_dynpose_dd10_{split}.txt"))
             if l.strip()]
    cache, dd = {}, 0
    for line in lines:
        _, scene, key = line.split("/", 2)
        if scene not in cache:
            with open(path.join(root, "dynpose", scene, "da3", "prompts.json"),
                      encoding="utf-8") as f:
                cache[scene] = load(f)
        entry = cache[scene][key] if isinstance(cache[scene], dict) else cache[scene][int(key)]
        dd += entry.get("preset", "").startswith("dd_")
    print(f"{split:<8}{len(lines):>7} 행   dd_* {dd:>6} ({dd / max(len(lines), 1):.1%})"
          f"   scene {len({l.split('/')[1] for l in lines})}")
EOF
echo "완료. 다음은 config + smoke (수동)."
