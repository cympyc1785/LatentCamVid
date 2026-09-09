#!/bin/bash
# D143 캡션 + export 드라이버 — vista 두 뱅크(d121 / d128)를 **캡션만 고쳐서** 새 루트로 뽑는다.
#
# ═══ 무엇을 고치는가 ══════════════════════════════════════════════════════════════════════
# 태스크 #134 의 모순: `aim="free"` preset 은 **한 번도 재조준하지 않는데** 캡션은
# "keeping it in a medium shot" 같은 프레이밍 약속을 하고 있었다. 게이트가
# `configs/caption_presets.json` 의 `targetless` 플래그만 봤기 때문이고, 그 플래그는
# pan/tilt 5개에만 붙어 있어서 `truck_*`/`pedestal_*`/`dolly_*`/`track_*` 15개가 그대로
# 통과했다 (`build_bank_captions.py` §free_aim_framing).
#
# 고친 방식은 **preset 축이 아니라 변이 축**이다 — `--framing_min_in_frame 0.85` (기본값).
# 그 변이의 실측 `subject_in_frame` 이 0.85 미만이면 framing/composition 절을 뺀다.
# preset 이름으로 뭉뚱그리면 두 방향으로 틀린다:
#   · `dolly_out`(median 1.000) / `track_dolly_out`(1.000) 은 재조준 없이도 대상이 남는데 버림
#   · `aim="look_at"` 도 2.9% 는 프레임을 놓치는데 preset 축으로는 아예 못 잡음
# motion 절과 target 은 그대로 둔다 — `track_*` 의 "tracks {target}" 은 follow_gain 1.0 으로
# 실제 참이다. 거짓인 건 프레이밍 약속뿐이라 그 절만 뺀다.
#
# ═══ 왜 기존 루트를 덮지 않는가 ═══════════════════════════════════════════════════════════
# `latentcam_da3_k6_d121` 은 2x3 그리드의 vista 행(D123/D133/D124)이 **학습한** 코퍼스이고
# `..._d128` 은 D131 이 학습한 코퍼스다. prompts.json 을 제자리에서 갈면 그 런들의
# 학습 캡션과 eval 캡션이 어긋나 재평가가 무효가 된다. 그래서 새 루트를 판다 (각 ~1 GB).
# 접미사 `c143` = caption D143. **뱅크(기하)는 손대지 않았다** — 두 루트의 카메라·이미지·
# seg_list 는 원본과 비트 동일해야 하고, 스크립트 끝에서 그걸 확인한다.
#
# dynpose d137 은 **건드리지 않는다** — D137/D141 이 그 코퍼스로 학습 중이다 (사용자 지시:
# "지금 학습 안돌리는 데이터셋 먼저 고쳐줘").
#
# ═══ 캡션 인자 ════════════════════════════════════════════════════════════════════════════
# 배포된 두 캡션 파일을 **글자 그대로 재현**하는 인자에서 `--framing_min_in_frame` 만 켠 것이다.
# 두 뱅크가 갈리는 건 `--targetless_promote` 하나뿐 — d128 은 켜고 구웠고(헤더의
# `targetless_promoted: [track_truck_left, track_truck_right]`) d121 은 그 플래그 이전이다.
# `--framing_min_in_frame 0` 을 주면 각각 배포본과 비트 동일하게 나온다 (검증됨).
#
# 사용:
#   bash scripts/run_d143_caption_export.sh d128     # 권장 (최신 뱅크)
#   bash scripts/run_d143_caption_export.sh d121     # 그리드 vista 행과 짝지을 대조군
#   bash scripts/run_d143_caption_export.sh both
#   DRY=1 bash scripts/run_d143_caption_export.sh both
set -u
WHICH=${1:-both}
HERE=/data1/cympyc1785/LatentCamVid/camera_generation/models/Planner/CinemaTraj
PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
VROOT=/data1/cympyc1785/data/Vista4D-Eval-Data
CAP=captions_d143.json
cd "$HERE" || exit 1

# 두 뱅크가 공유하는 캡션 인자. `--framing_min_in_frame` 은 argparse 기본이 0.85 라 안 적는다.
CAPTION_ARGS=(--videos all --prompt_fields "target,motion" --prompt_style nl
              --magnitude --anchor_desc --out_name "$CAP")
[ "${DRY:-0}" = "1" ] && CAPTION_ARGS+=(--dry_run)
EXPORT_EXTRA=()
[ "${DRY:-0}" = "1" ] && EXPORT_EXTRA+=(--dry_run)

# $1 뱅크 디렉터리 / $2 out_root / $3 promote 플래그
do_one() {
    local bank=$1 out=$2 promote=$3
    echo "=== $bank: caption ($promote) ==="
    $PY scripts/build_bank_captions.py --bank_dir "$bank" "$promote" \
        "${CAPTION_ARGS[@]}" \
        --metadata_csv "$VROOT/metadata.csv" || return 1
    echo "=== $bank: export -> $out ==="
    $PY scripts/vista4d_bank_to_dl3dv.py --videos all --bank_dir "$bank" \
        --captions_name "$CAP" \
        --cine_out "$HERE/out" \
        --recon_root "$VROOT/eval_data/recon_and_seg" \
        --out_root "$out" \
        --meta_csv meta_vista4d.csv --chunk_prefix vista4d --seg_list_prefix seg_list_vista4d \
        --test_videos camel avocado-slice bmx-bumps couple-hug \
        --image_scale 0.5 "${EXPORT_EXTRA[@]}"
}

# 기하가 그대로인지 확인한다 — 캡션만 바꿨으므로 seg_list 두 개가 원본과 **글자 그대로**
# 같아야 한다. 다르면 export 인자가 배포본과 어긋난 것이고, 그건 조용히 다른 코퍼스다.
check_same() {
    local new=$1 old=$2 bad=0
    for f in seg_list_vista4d_train.txt seg_list_vista4d_test.txt meta_vista4d.csv; do
        if cmp -s "$new/$f" "$old/$f"; then echo "  OK   $f"
        else echo "  DIFF $f  <-- 기하가 달라졌다. export 인자 확인할 것"; bad=1; fi
    done
    return $bad
}

case "$WHICH" in
  d121|both) do_one hole_bank_k6_d121 "$VROOT/latentcam_da3_k6_d121c143" --no_targetless_promote \
             && { echo "=== d121c143 대조 ==="; check_same "$VROOT/latentcam_da3_k6_d121c143" \
                                                          "$VROOT/latentcam_da3_k6_d121"; } ;;
esac
case "$WHICH" in
  d128|both) do_one hole_bank_k6_d128 "$VROOT/latentcam_da3_k6_d128c143" --targetless_promote \
             && { echo "=== d128c143 대조 ==="; check_same "$VROOT/latentcam_da3_k6_d128c143" \
                                                          "$VROOT/latentcam_da3_k6_d128"; } ;;
esac
