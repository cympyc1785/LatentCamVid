#!/bin/bash
# D105 캡션 + export 드라이버 — vista 와 TRUMANS 를 **한 스크립트에서 같은 인자로** 돌린다.
#
# 왜 스크립트로 묶는가: 두 코퍼스는 이미 같은 두 프로그램(`build_bank_captions.py` →
# `vista4d_bank_to_dl3dv.py`)을 타는데, 여태 각자 손으로 호출해 왔다. 그 결과 배포된 d77
# TRUMANS 코퍼스가 이런 프롬프트를 들고 있다:
#
#     target: person. motion: the camera dollies straight forward toward the subject.
#
# preset 은 `dolly_in` 인데, D90 에서 `dolly_in` 은 `aim="free"` = **재조준을 안 하는** 카메라가
# 됐다. 문구만 옛 뜻(조준)에 남아 있었던 것이다. 부사 문제가 아니라 motion 절의 **의미**가
# 궤적과 반대였다 (D105 에서 `configs/caption_presets.json` 을 고쳤다).
#
# 캡션 인자는 **두 코퍼스가 글자 그대로 같아야 한다** — 아래 `CAPTION_ARGS` 하나만 쓴다.
# 코퍼스별로 갈리는 것은 세 종류뿐이고 전부 여기 적혀 있다:
#   ① `--label_map`      TRUMANS 만. blend 오브젝트 이름(`Floor.008`) -> 자연어 명사.
#                        `target:` 절의 **명사만** 바꾼다 (문장 형식엔 안 닿는다).
#   ② `--metadata_csv`   `event` 필드의 출처. `--prompt_fields target,motion` 이라 학습
#                        프롬프트엔 안 들어가지만 JSON 에는 남는다.
#   ③ export 경로/해상도  TRUMANS 소스는 960x540, vista 는 1280x720 (둘 다 640x360 으로).
#
# **부사 없음**: `--magnitude` 를 안 넘긴다 (argparse 기본값 False). 사용자 지시 "부사 없이".
# 켜면 같은 부사가 씬마다 다른 실제 이동량을 가리킨다 (씬 깊이로 나눈 값).
#
# 사용:
#   bash exec/_legacy/run_d99_caption_export.sh vista      # 뱅크 다 구워진 뒤
#   bash exec/_legacy/run_d99_caption_export.sh trumans
#   bash exec/_legacy/run_d99_caption_export.sh both
#   DRY=1 bash exec/_legacy/run_d99_caption_export.sh both # 캡션만 dry-run, export 는 --dry_run
set -u
WHICH=${1:-both}
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
BANK=hole_bank_k6_d99
cd "$HERE" || exit 1

# 두 경로가 공유하는 캡션 인자. 여기를 바꾸면 양쪽이 같이 바뀐다 — 그게 이 파일의 존재 이유다.
CAPTION_ARGS=(--bank_dir "$BANK" --prompt_fields "target,motion" --out_name captions.json)
[ "${DRY:-0}" = "1" ] && CAPTION_ARGS+=(--dry_run)
EXPORT_EXTRA=()
[ "${DRY:-0}" = "1" ] && EXPORT_EXTRA+=(--dry_run)

# TRUMANS holdout 은 **recording 단위**다 (chunk 단위로 자르면 49프레임 슬라이딩이라 같은
# recording 의 이웃 chunk 가 train/test 에 동시에 들어가 test 가 새 씬 일반화를 못 잰다).
# d77 arm 이 쓴 것과 같은 두 recording 을 기본값으로 둔다.
TRU_TEST=${TRU_TEST:-"tru_0ab19ed6 tru_1a1e205b"}

do_vista() {
    echo "=== vista: caption ==="
    $PY fit/caption/build_bank_captions.py --videos all "${CAPTION_ARGS[@]}" \
        --metadata_csv /data1/cympyc1785/data/Vista4D-Eval-Data/metadata.csv || return 1
    echo "=== vista: export ==="
    $PY fit/convert/vista4d_bank_to_dl3dv.py --videos all --bank_dir "$BANK" \
        --cine_out "$HERE/out" \
        --recon_root /data1/cympyc1785/data/Vista4D-Eval-Data/eval_data/recon_and_seg \
        --out_root  /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d99 \
        --meta_csv meta_vista4d.csv --chunk_prefix vista4d --seg_list_prefix seg_list_vista4d \
        --test_videos camel avocado-slice bmx-bumps couple-hug \
        --image_scale 0.5 "${EXPORT_EXTRA[@]}"
}

do_trumans() {
    echo "=== trumans: caption ==="
    # `--metadata_csv` 는 TRUMANS-Lite 것. 없으면 `event` 만 빈 문자열이 되고 학습 프롬프트
    # (`target,motion`) 는 영향을 안 받는다.
    $PY fit/caption/build_bank_captions.py --videos all "${CAPTION_ARGS[@]}" \
        --output_root "$HERE/out_trumans" \
        --label_map configs/trumans_labels.json \
        --metadata_csv /data1/cympyc1785/data/TRUMANS-Lite/metadata.csv || return 1
    echo "=== trumans: export ==="
    # `--image_scale 0.6666666666666666`: 소스가 960x540 이라 0.5 를 쓰면 480x270 이 된다.
    # vista 와 **같은 640x360** 으로 맞춰야 두 코퍼스를 섞어 학습할 때 hw_list 가 갈리지 않는다.
    # 리사이즈 목표 (w,h) 는 스케일된 K 의 cx*2 / cy*2 에서 나온다 (`scaled_K`).
    $PY fit/convert/vista4d_bank_to_dl3dv.py --videos all --bank_dir "$BANK" \
        --cine_out "$HERE/out_trumans" \
        --recon_root /data1/cympyc1785/data/TRUMANS-Lite/eval_data/recon_and_seg \
        --out_root  /data1/cympyc1785/data/TRUMANS-Lite/latentcam_da3_d99 \
        --meta_csv meta_trumans.csv --chunk_prefix trumans --seg_list_prefix seg_list_trumans \
        --test_videos $TRU_TEST \
        --image_scale 0.6666666666666666 "${EXPORT_EXTRA[@]}"
}

RC=0
case "$WHICH" in
    vista)   do_vista   || RC=1 ;;
    trumans) do_trumans || RC=1 ;;
    both)    do_vista   || RC=1; do_trumans || RC=1 ;;
    *) echo "usage: $0 {vista|trumans|both}"; exit 2 ;;
esac
echo "[caption+export] rc=$RC  $(date +%H:%M:%S)"
exit $RC
