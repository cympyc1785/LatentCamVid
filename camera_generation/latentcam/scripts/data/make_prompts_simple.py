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

env: 아무거나 (표준 라이브러리만 쓴다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
    $PY scripts/data/make_prompts_simple.py \
        --root /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3 --meta_csv meta_vista4d.csv
    $PY scripts/data/make_prompts_simple.py --root ... --meta_csv ... --dry_run
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


def read_scenes(root: str, meta_csv: str):
    """meta CSV 의 첫 컬럼(chunk) 목록. 헤더 한 줄을 건너뛴다."""
    with open(osp.join(root, meta_csv), encoding="utf-8") as file:
        rows = [line.strip() for line in file if line.strip()]
    return [r.split(",")[0] for r in rows[1:]]


def main(args):
    scenes = read_scenes(args.root, args.meta_csv)
    n_scene = n_seg = 0
    missing, samples = [], []
    presets, targets = {}, {}
    for chunk in scenes:
        src = osp.join(args.root, chunk, "da3", args.src_name)
        if not osp.isfile(src):
            missing.append(chunk)
            continue
        with open(src, encoding="utf-8") as file:
            prompts = json.load(file)
        out = {}
        for key, seg in prompts.items():
            text = simple_prompt(seg, args.keep_magnitude)
            # 원본 세그먼트를 통째로 복사하고 캡션만 덮어쓴다 -- variant_id / preset /
            # tau_max 등 부가 메타를 잃지 않아야 나중에 필터·분석이 그대로 된다.
            new = dict(seg)
            new["prompt_camera_with_scene_video"] = {"concise": text}
            new["prompt_source"] = f"simple(from {args.src_name})"
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", required=True)            # latentcam_da3 레이아웃 루트
    parser.add_argument("--meta_csv", default="meta.csv")   # scene 목록 CSV (루트 기준)
    parser.add_argument("--src_name", default="prompts.json")          # 읽을 파일
    parser.add_argument("--out_name", default="prompts_simple.json")   # 쓸 파일
    parser.add_argument("--keep_magnitude", action="store_true")       # 크기 부사 유지
    parser.add_argument("--no_keep_magnitude", dest="keep_magnitude", action="store_false")
    parser.set_defaults(keep_magnitude=False)
    parser.add_argument("--dry_run", action="store_true")   # 통계만 내고 파일은 안 쓴다
    main(parser.parse_args())
