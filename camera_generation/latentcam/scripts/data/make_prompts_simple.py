"""`<scene>/da3/prompts.json` → `prompts_simple.json` (축약 캡션).

왜: 지금 학습에 들어가는 문장은
    "target: camel motion: the camera significantly dollies straight forward toward the subject"
이다. 이게 **길어서** 안 배우는 것인지 알 수 없어서, 같은 세그먼트에 같은 카메라를 두고
텍스트만 축약한 대조군을 만든다:
    "target: camel. motion: dolly in"

세그먼트 키·`frame_idx`·`variant_id` 는 **한 글자도 안 바꾼다**. 바뀌는 것은
`prompt_camera_with_scene_video.concise` 하나뿐이므로 seg list / pose / avg_scale / target_poses
전부 그대로 재사용된다. dataset 쪽은 `prompts_file: prompts_simple.json` 한 줄로 갈아끼운다
(`main/dataset_dl3dv.py:_prompts_path`).

**크기 부사를 뺀다** (사용자 지시). 원본의 `significantly` / `dramatically` 같은 부사는
`tau_max`(=|t|/z_med) 또는 `pan_deg` 버킷이라 실제 이동량 정보다. 이걸 빼면 텍스트가 preset
14종만 지시하게 되고 이동량은 모델이 알아서 정한다 — 즉 이 ablation 은 "문장 길이"와
"이동량 지시" 두 축을 **같이** 움직인다. 결과 해석 시 반드시 반영할 것.
부사를 남기려면 `--keep_magnitude` 로 켠다 (`tiny/small/medium/big/huge` 로 축약해서 앞에 붙는다).

`target` 은 `caption_fields.target` 을 그대로 쓴다 (뱅크가 조준한 노드 라벨). preset 은
`preset` 필드에서 읽으므로 뱅크 CSV 를 다시 안 읽는다. 표에 없는 preset 이 나오면 **raise** 한다
— 조용히 빈 motion 을 흘리면 그 변이만 텍스트가 사라진 채 학습된다.

═══ `--fields` — 축약이 아니라 **절 선택** (2026-08-28) ══════════════════════════════════
위 축약 경로와 별개로, `caption_fields` 의 문장을 **한 글자도 안 고치고 어느 절을 넣을지만**
고르는 모드다. `--fields motion` 이면
    "target: man motion: the camera dramatically arcs to the left ..."
    ->            "motion: the camera dramatically arcs to the left ..."
가 된다. 조립 규칙은 `camera_generation/dataset/fit/caption/build_bank_captions.py:84 prompt_of` 와 같다
(`"{field}: {value}"` 를 공백으로 join, 빈 필드는 건너뜀) — 그래서 `--fields target,motion` 은
원본 `prompts.json` 을 그대로 재현한다 (동일성 확인용).

**축약 경로와 섞이지 않는다**: `--fields` 를 주면 `SIMPLE_PHRASE`/`--keep_magnitude` 는 아예
안 탄다. 즉 이 모드는 "target 절 유무" **한 축만** 움직이므로 축약 ablation 처럼 두 축이
같이 움직이는 문제가 없다. `--fields` 미지정(기본) = 기존 동작 비트 동일.

═══ `--captions_root` — 뱅크에서 **다시 구운 캡션**을 얹는다 (2026-09-15, D196) ═════════════
위 두 경로는 코퍼스 안의 `caption_fields` 만 재조립한다. 그런데 `build_bank_captions.py` 의
플래그(`--magnitude` / `--nl_framing` 등)를 바꾸면 **`caption_fields` 자체가 달라진다** — 그건
코퍼스에 없고 CinemaTraj 뱅크 쪽 `captions_*.json` 에만 있다. 그걸 다시 export(115편/분,
7.8k편) 하지 않고 캡션만 갈아끼우는 경로다:

    <captions_root>/<scene>/<bank_dir>/<captions_name>  ->  <root>/<chunk>/da3/<out_name>

조인 키는 **`variant_id`** 다 (`tau_max` 같은 실현치를 키로 쓰면 조용히 0행 — D191 에서 같은
사고가 있었다). 세그먼트 키·`frame_idx`·`variant_id`·pose 는 한 글자도 안 바뀌므로 seg list /
`target_poses.npz` / `target_track.npz` / avg_scale / molmo2 캐시가 전부 그대로 유효하다.
`caption_fields` 도 새 캡션 것으로 같이 덮는다 — 안 덮으면 문장은 새 것인데 분석용 구조체는
옛 것이 남아 나중에 두 개가 어긋난다.
변이를 못 찾으면 **끝에서 raise** 한다 (`--allow_missing` 으로 완화). 조용히 옛 문장을 남기면
"캡션 갈아끼웠다"고 믿으면서 예전 텍스트로 학습하게 된다.

env: 아무거나 (표준 라이브러리만 쓴다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
    $PY scripts/data/make_prompts_simple.py \
        --root /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3 --meta_csv meta_vista4d.csv
    $PY scripts/data/make_prompts_simple.py --root ... --meta_csv ... --dry_run
    # target 절 없는 대조 프롬프트 (motion 문장은 원본 그대로)
    $PY scripts/data/make_prompts_simple.py --root ... --meta_csv meta_snowboard.csv \
        --fields motion --out_name prompts_notarget.json
    # D196 — 뱅크에서 다시 구운 캡션(정도부사+framing 복원)을 코퍼스에 얹기
    $PY scripts/data/make_prompts_simple.py \
        --root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d194 \
        --meta_csv meta_dynpose.csv --out_name prompts_mag.json \
        --captions_root <CinemaTraj>/out_dynpose --captions_bank_dir hole_bank_d192 \
        --captions_name captions_d196.json
"""
import json
import os.path as osp
from argparse import ArgumentParser

# preset -> 축약 문구. 원본 `configs/caption_presets.json` 의 `phrase` 와 1:1 대응이지만
# "the camera"/"toward the subject" 같은 상용구를 걷어낸 형태다. 방향(left/right, up/down)과
# 축(dolly/pan/truck/pedestal/orbit/arc/crane)은 남긴다 — 그게 preset 을 가르는 정보다.
SIMPLE_PHRASE = {
    "straight_ease":      "dolly in",
    "dolly_in":           "dolly in",
    "dolly_out":          "dolly out",
    "push_in_arc":        "arc left and push in",
    "pull_out_arc":       "arc right and pull out",
    "orbit_left_arc":     "orbit left",
    "orbit_right_arc":    "orbit right",
    "pan_left":           "pan left",
    "pan_right":          "pan right",
    "truck_left":         "truck left",
    "truck_right":        "truck right",
    "pedestal_up":        "pedestal up",
    "pedestal_down":      "pedestal down",
    "rise_reveal":        "crane up",
    "drop_reveal":        "crane down",
    "s_curve":            "s-curve",
    "static_hold":        "hold still",
    "static_hold_locked": "hold locked off",
}

# --keep_magnitude 일 때만 쓰는 부사 축약표 (원본 5단계 버킷 이름 -> 한 단어).
SIMPLE_MAGNITUDE = {"barely": "tiny", "slightly": "small", "steadily": "medium",
                    "significantly": "big", "dramatically": "huge"}


def magnitude_of(motion: str):
    """원본 motion 문장 앞머리 "the camera <부사> ..." 에서 부사만 뽑는다. 없으면 None."""
    parts = str(motion or "").split()
    return parts[2] if len(parts) > 2 and parts[2] in SIMPLE_MAGNITUDE else None


def fields_prompt(seg: dict, fields):
    """`caption_fields` 에서 고른 절만 **원문 그대로** 이어 붙인다 (축약 없음).

    조립 규칙은 `build_bank_captions.py:84 prompt_of` 와 동일 — 빈 값은 건너뛰므로
    `--fields target,motion` 이 원본 `prompts.json` 을 그대로 재현한다.
    """
    caption = seg.get("caption_fields") or {}
    return " ".join(f"{f}: {caption[f]}" for f in fields if caption.get(f))


def simple_prompt(seg: dict, keep_magnitude: bool):
    """세그먼트 dict -> 축약 문장. target 이 비면 motion 절만 낸다 (빈 'target: .' 방지)."""
    fields = seg.get("caption_fields") or {}
    preset = seg.get("preset")
    if preset not in SIMPLE_PHRASE:
        raise KeyError(f"preset {preset!r} 에 축약 문구가 없다 — SIMPLE_PHRASE 에 추가할 것 "
                       f"(variant_id={seg.get('variant_id')})")
    motion = SIMPLE_PHRASE[preset]
    if keep_magnitude:
        mag = magnitude_of(fields.get("motion"))
        if mag:
            motion = f"{SIMPLE_MAGNITUDE[mag]} {motion}"
    target = str(fields.get("target") or "").strip()
    return f"target: {target}. motion: {motion}" if target else f"motion: {motion}"


# [new 2026-09-15 / D196] 뱅크 캡션에서 코퍼스 `caption_fields` 로 옮길 키. `prompt` 는 문장
# 자리라 따로 다루고, `target_in_motion` / `target_view` 는 조립용 내부 플래그라 안 옮긴다
# (기존 prompts.json 의 caption_fields 에도 없다 — 키 집합을 바꾸면 하류 분석이 갈린다).
CAPTION_FIELD_KEYS = ["target", "event", "framing", "motion",
                      "target_text", "framing_nl", "composition"]


def load_bank_captions(captions_root: str, scene: str, bank_dir: str, name: str):
    """`<captions_root>/<scene>/<bank_dir>/<name>` -> {variant_id: caption dict}. 없으면 None."""
    src = osp.join(captions_root, scene, bank_dir, name)
    if not osp.isfile(src):
        return None
    with open(src, encoding="utf-8") as file:
        return (json.load(file).get("captions") or {})


def rebaked_segment(seg: dict, caption: dict, src_name: str):
    """세그먼트 dict + 뱅크 캡션 -> 문장과 `caption_fields` 만 갈아끼운 새 dict."""
    new = dict(seg)
    new["prompt_camera_with_scene_video"] = {"concise": caption.get("prompt", "")}
    new["caption_fields"] = {k: caption.get(k, "") for k in CAPTION_FIELD_KEYS}
    new["prompt_source"] = f"bank[{src_name}]"
    return new


def read_scenes(root: str, meta_csv: str):
    """meta CSV 의 첫 컬럼(chunk) 목록. 헤더 한 줄을 건너뛴다."""
    with open(osp.join(root, meta_csv), encoding="utf-8") as file:
        rows = [line.strip() for line in file if line.strip()]
    return [r.split(",")[0] for r in rows[1:]]


def main(args):
    _fields = [f.strip() for f in (args.fields or "").split(",") if f.strip()]
    scenes = read_scenes(args.root, args.meta_csv)
    n_scene = n_seg = 0
    missing, samples = [], []
    no_caps, no_variant = [], []
    presets, targets = {}, {}
    for chunk in scenes:
        src = osp.join(args.root, chunk, "da3", args.src_name)
        if not osp.isfile(src):
            missing.append(chunk)
            continue
        with open(src, encoding="utf-8") as file:
            prompts = json.load(file)
        caps = None
        if args.captions_root:
            caps = load_bank_captions(args.captions_root, chunk.split("/")[-1],
                                      args.captions_bank_dir, args.captions_name)
            if caps is None:
                no_caps.append(chunk)
                continue
        out = {}
        for key, seg in prompts.items():
            if caps is not None:
                # [new 2026-09-15 / D196] 뱅크 캡션 모드. 조인 키는 variant_id 하나다.
                vid = seg.get("variant_id")
                if vid not in caps:
                    no_variant.append(f"{chunk.split('/')[-1]}/{key}:{vid}")
                    continue
                new = rebaked_segment(seg, caps[vid], args.captions_name)
            else:
                # --fields 를 주면 절 선택 모드, 아니면 기존 축약 경로 (기본값 = 기존 동작).
                text = (fields_prompt(seg, _fields) if _fields
                        else simple_prompt(seg, args.keep_magnitude))
                # 원본 세그먼트를 통째로 복사하고 캡션만 덮어쓴다 -- variant_id / preset /
                # tau_max 등 부가 메타를 잃지 않아야 나중에 필터·분석이 그대로 된다.
                new = dict(seg)
                new["prompt_camera_with_scene_video"] = {"concise": text}
                new["prompt_source"] = (f"fields[{','.join(_fields)}](from {args.src_name})"
                                        if _fields else f"simple(from {args.src_name})")
            text = new["prompt_camera_with_scene_video"]["concise"]
            out[key] = new
            n_seg += 1
            presets[seg.get("preset")] = presets.get(seg.get("preset"), 0) + 1
            t = (seg.get("caption_fields") or {}).get("target", "")
            targets[t] = targets.get(t, 0) + 1
            if len(samples) < 8:
                samples.append((chunk.split("/")[-1], key, text))
        n_scene += 1
        if not args.dry_run:
            with open(osp.join(args.root, chunk, "da3", args.out_name), "w",
                      encoding="utf-8") as file:
                json.dump(out, file, ensure_ascii=False, indent=1)

    print(f"{'scene':10s}{'seg':>8s}")
    print(f"{n_scene:<10d}{n_seg:>8d}" + ("   [dry_run — 안 씀]" if args.dry_run else ""))
    if missing:
        print(f"\n{args.src_name} 없는 scene {len(missing)}개: {missing[:5]}")
    if no_caps:
        print(f"\n⚠ {args.captions_name} 없는 scene {len(no_caps)}개: {no_caps[:5]}")
    if no_variant:
        print(f"\n⚠ 뱅크 캡션에 variant_id 가 없는 세그먼트 {len(no_variant)}개: {no_variant[:5]}")
    print(f"\n[preset {len(presets)}종]")
    for p, c in sorted(presets.items(), key=lambda kv: -kv[1]):
        print(f"  {c:6d}  {p}")
    print(f"\n[target {len(targets)}종] " + ", ".join(
        f"{t}({c})" for t, c in sorted(targets.items(), key=lambda kv: -kv[1])[:12]))
    print("\n[표본]")
    for sc, key, text in samples:
        print(f"  {sc}/{key}: {text}")
    if not args.dry_run:
        print(f"\n-> {args.root}/<scene>/da3/{args.out_name}")
        print(f"   config: prompts_file: {args.out_name}")
    # 조용히 옛 문장으로 학습되는 것보다 여기서 멈추는 게 낫다 (§docstring --captions_root).
    if (no_caps or no_variant) and not args.allow_missing:
        raise SystemExit(f"[FAIL] 캡션 없는 scene {len(no_caps)} / variant {len(no_variant)} — "
                         f"--allow_missing 을 주지 않으면 여기서 멈춘다")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", required=True)            # latentcam_da3 레이아웃 루트
    parser.add_argument("--meta_csv", default="meta.csv")   # scene 목록 CSV (루트 기준)
    parser.add_argument("--src_name", default="prompts.json")          # 읽을 파일
    parser.add_argument("--out_name", default="prompts_simple.json")   # 쓸 파일
    # [new 2026-08-28] caption_fields 절 선택 모드 (축약 안 함). 미지정 = 기존 축약 경로.
    parser.add_argument("--fields", default=None, type=str)            # 예: "motion", "target,motion"
    parser.add_argument("--keep_magnitude", action="store_true")       # 크기 부사 유지
    parser.add_argument("--no_keep_magnitude", dest="keep_magnitude", action="store_false")
    parser.set_defaults(keep_magnitude=False)
    # [new 2026-09-15 / D196] 뱅크 캡션 모드. 주면 --fields / --keep_magnitude 는 안 탄다.
    parser.add_argument("--captions_root", default=None, type=str)          # CinemaTraj out_dynpose
    parser.add_argument("--captions_bank_dir", default="hole_bank_d192", type=str)
    parser.add_argument("--captions_name", default="captions.json", type=str)
    parser.add_argument("--allow_missing", action="store_true")             # 빠진 변이 허용
    parser.add_argument("--dry_run", action="store_true")   # 통계만 내고 파일은 안 쓴다
    main(parser.parse_args())
