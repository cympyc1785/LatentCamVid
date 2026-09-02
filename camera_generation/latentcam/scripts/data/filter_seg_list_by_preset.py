"""이미 export 된 코퍼스에서 **seg_list 만 골라내** 하위 arm 을 만든다.

왜 재export 가 아니라 seg_list 인가: `dynpose_d84_k6_nodd` / `_ddswap` 이 이미 이 방식으로
돌고 있다. 코퍼스 디렉토리(`dynpose/<scene>/da3/*`) · `geo_raw_cache_da3`(11.3 GB) ·
`avg_scale*` · `prompts.json` 은 전부 **그대로 재사용**하고, 학습이 읽는 세그먼트 목록만
줄인다. 재export(266편) 도 geo 캐시 재생성(11.3 GB)도 필요 없고, 무엇보다 **같은 파일을
보므로 arm 끼리 paired 비교가 깨지지 않는다** — 분할도 원본 train/test 리스트를 필터링만
하므로 scene holdout 이 자동으로 보존된다.

지금까지 nodd/ddswap 리스트는 세션 안에서 즉석 python 으로 만들었다. 그러면 어떤 조건으로
걸렀는지가 리스트 파일에만 남고 재현이 안 된다 — 그래서 스크립트로 박는다.

판정은 `prompts.json` 의 두 필드만 본다:
  preset       raw 문자열이다. **정식 이름이 아니다** — D76/D90/D94 이전에 구운 뱅크는 옛
               이름으로 적혀 있다. 예: dynpose 코퍼스의 `dolly_in` 696 행은 전부
               `aim="look_at"` 이라 정식 이름으로는 `dolly_in_look_at` 이지만, export 된
               prompts.json 에는 `dolly_in` 이라고 적혀 있다 (`aim` 은 export 를 안 탄다).
               그래서 여기 `--presets` 에는 **적혀 있는 그대로**를 준다.
  variant_id   `<anchor_id>__<preset>__hole<τ>`. `--max_anchors` 는 여기서 anchor 를 떼어
               scene 마다 **정렬 순 앞 N 개**만 남긴다 (결정론적).

    python scripts/data/filter_seg_list_by_preset.py \
        --root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose \
        --prefix seg_list_dynpose --presets dolly_in --max_anchors 2 --suffix dionly
    python scripts/data/filter_seg_list_by_preset.py ... --exclude_presets_prefix dd_ --suffix nodd
    # D109: dd_* 가 최종 리스트의 10% 가 되게 서브샘플 (preset 90%). RNG 없이 결정론적 —
    # 매칭 행을 리스트 순서 그대로 두고 균등 stride 로 뽑아 scene 에 고르게 퍼진다.
    python scripts/data/filter_seg_list_by_preset.py ... \
        --target_frac_prefix dd_ --target_frac 0.10 --suffix dd10
"""
from argparse import ArgumentParser
from collections import Counter
from os import listdir, path
import json


def load_segments(root: str, scene: str):
    """scene 의 prompts.json → [(seg_key, entry)]. dict/list 두 레이아웃을 모두 받는다."""
    file_path = path.join(root, "dynpose", scene, "da3", "prompts.json")
    if not path.isfile(file_path):
        return []
    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)
    return list(data.items()) if isinstance(data, dict) else list(enumerate(data))


def subsample_to_frac(sel: list, key_to_preset: dict, prefix: str, frac: float):
    """`prefix` 매칭 행이 최종 리스트의 `frac` 이 되도록 균등 stride 로 서브샘플.

    수식: 비매칭 N개가 (1-frac) 을 차지하므로 매칭은 round(N·frac/(1-frac)) 개.
    RNG 를 안 쓴다 — 매칭 행의 원래 순서(scene 정렬순) 위에서 균등 간격으로 집으면
    결정론적이고 scene/preset 에 고르게 퍼진다.
    """
    hit = [line for line in sel if key_to_preset[line].startswith(prefix)]
    rest_n = len(sel) - len(hit)
    want = min(len(hit), round(rest_n * frac / max(1.0 - frac, 1e-9)))
    if want <= 0:
        chosen = set()
    else:
        step = len(hit) / want
        chosen = {hit[int(i * step)] for i in range(want)}
    return [line for line in sel if not key_to_preset[line].startswith(prefix)
            or line in chosen], len(hit) - len(chosen)


def keep_keys(args, scenes):
    """조건을 통과한 `<chunk>/<scene>/<seg_key>` → preset dict + 통계."""
    keep, stats = {}, Counter()
    presets = set(args.presets.split(",")) if args.presets else None
    excl = set(args.exclude_presets.split(",")) if args.exclude_presets else set()
    for scene in scenes:
        rows = []
        for key, entry in load_segments(args.root, scene):
            preset = entry.get("preset", "")
            stats["seen"] += 1
            if presets is not None and preset not in presets:
                continue
            if preset in excl or (args.exclude_presets_prefix
                                  and preset.startswith(args.exclude_presets_prefix)):
                continue
            rows.append((key, entry))
        if not rows:
            continue
        if args.max_anchors > 0:
            # anchor 는 variant_id 앞머리다. 정렬해서 앞 N 개 — scene 마다 같은 답이 나와야 한다.
            anchors = sorted({str(e["variant_id"]).split("__")[0] for _, e in rows})
            allow = set(anchors[:args.max_anchors])
            stats["anchor_dropped"] += sum(
                1 for _, e in rows if str(e["variant_id"]).split("__")[0] not in allow)
            rows = [(k, e) for k, e in rows
                    if str(e["variant_id"]).split("__")[0] in allow]
        for key, entry in rows:
            keep[f"{args.chunk_prefix}/{scene}/{key}"] = entry.get("preset", "")
    return keep, stats


def main(args):
    scenes = sorted(name for name in listdir(path.join(args.root, "dynpose"))
                    if path.isdir(path.join(args.root, "dynpose", name)))
    keep, stats = keep_keys(args, scenes)

    print(f"{'root':<16}{args.root}")
    print(f"{'scenes':<16}{len(scenes)}")
    print(f"{'presets':<16}{args.presets or '(전부)'}"
          f"   exclude={args.exclude_presets or '-'}"
          f" prefix={args.exclude_presets_prefix or '-'}")
    print(f"{'max_anchors':<16}{args.max_anchors or '(제한 없음)'}"
          f"   anchor 로 뺀 세그먼트 {stats['anchor_dropped']}")
    print(f"{'통과':<16}{len(keep)} / {stats['seen']}\n")

    header = f"{'split':<8}{'before':>8}{'after':>8}{'scenes':>9}"
    print(header)
    print("-" * len(header))
    written = []
    for split in ("train", "test"):
        src = path.join(args.root, f"{args.prefix}_{split}.txt")
        with open(src, encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
        sel = [line for line in lines if line in keep]
        sub_dropped = 0
        if args.target_frac_prefix and args.target_frac > 0:
            sel, sub_dropped = subsample_to_frac(sel, keep, args.target_frac_prefix,
                                                 args.target_frac)
        n_scene = len({line.split("/")[1] for line in sel})
        n_hit = sum(1 for line in sel if keep[line].startswith(args.target_frac_prefix)) \
            if args.target_frac_prefix else 0
        extra = (f"   {args.target_frac_prefix}* {n_hit} ({n_hit / max(len(sel), 1):.1%}, "
                 f"뺀 것 {sub_dropped})") if args.target_frac_prefix else ""
        print(f"{split:<8}{len(lines):>8}{len(sel):>8}{n_scene:>9}{extra}")
        dst = path.join(args.root, f"{args.prefix}_{args.suffix}_{split}.txt")
        if not args.dry_run:
            with open(dst, "w", encoding="utf-8") as file:
                file.write("\n".join(sel) + "\n")
            written.append(dst)
    if args.dry_run:
        print("\n--dry_run — 아무것도 안 썼다")
    else:
        print()
        for dst in written:
            print(f"기록  {dst}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--prefix", default="seg_list_dynpose", type=str)   # 원본 리스트 접두사
    parser.add_argument("--suffix", required=True, type=str)                # 새 리스트 꼬리표
    parser.add_argument("--chunk_prefix", default="dynpose", type=str)      # seg_list 첫 칸
    parser.add_argument("--presets", default="", type=str)                  # 남길 raw preset (콤마)
    parser.add_argument("--exclude_presets", default="", type=str)          # 뺄 raw preset (콤마)
    parser.add_argument("--exclude_presets_prefix", default="", type=str)   # 예: dd_
    parser.add_argument("--max_anchors", default=0, type=int)               # scene 당 anchor 상한
    # D109. prefix 매칭 preset 이 최종 리스트의 이 비율이 되게 서브샘플. 0 = 끔 (예전 동작).
    parser.add_argument("--target_frac_prefix", default="", type=str)
    parser.add_argument("--target_frac", default=0.0, type=float)
    parser.add_argument("--dry_run", action="store_true", default=False)
    raise SystemExit(main(parser.parse_args()))
