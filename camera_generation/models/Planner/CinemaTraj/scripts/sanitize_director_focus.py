"""Director 산출물의 focus id 목록에서 **유령 asset** 을 뺀다.

**왜 필요한가.** LBM 의 Director 는 렌더 이미지를 보고 `focus_ids` 를 쓰는데, 그 문자열이
`asset_index`(그 씬의 상호작용 가능 asset) 안에 있는지 **아무도 검사하지 않는다**. TRUMANS
86 chunk 실측에서 `sofa`(12) `coffee_table`(10) `chair`(7) 같은 없는 id 가 59회 섞였다.

대부분은 무해하다 — Cinematographer 의 `_find_object`
(`Cinematographer/cinematographer_preview_worker.py:103-123`) 가 blend 오브젝트 이름과
완전일치 / `<id>.` / `<id>_` 접두사로만 찾고, 못 찾으면 `continue` 로 건너뛴다.

문제는 **asset 은 아닌데 blend 에는 실제로 있는 이름**이다. 그건 focus AABB 에 그대로 합쳐져
카메라 거리를 바꾼다. 실측(00add26c, w01 blend, 432 오브젝트):

    zzy3                extent (0.84, 1.76, 2.22)   maxdim 2.22 m
    zzy3 + window       extent (3.52, 1.76, 3.29)   maxdim 3.52 m   (x1.6)
    zzy3 + floor        extent (1.88, 6.61, 2.22)   maxdim 6.61 m   (x3.0)

즉 `floor` 하나 때문에 focus 경계가 3배로 부풀고 카메라가 그만큼 물러난다 — 실패가 아니라
**조용히 나빠지는** 종류라 로그만 봐서는 안 보인다.

그래서 이 스크립트는 **blend 에 실제로 붙는데 asset 이 아닌 id 만** 뺀다. 해석 안 되는 id
(`sofa` 등)는 어차피 건너뛰어지므로 손대지 않는다 — 원본을 최소로만 고치기 위해서다.

스칼라 `primary_focus_id` 는 **건드리지 않는다**. 비우면 하류가 primary 없이 돌아가는데,
실측 86 chunk 전량에서 primary 는 항상 캐릭터라 뺄 것이 없었다. 그런 경우가 생기면
`--strict` 없이도 요약표에 `primary_hits` 로 찍고 사람에게 넘긴다.

사용 예시:

    # 어떤 id 를 뺄지만 보고 파일은 안 고친다
    python scripts/sanitize_director_focus.py \\
      --output_root ../Look-Before-Move/Director/output --run_glob 'trumans_c49_w*' \\
      --blend_objects /tmp/blend_objects.json --dry_run

    # 실제로 고친다 (원본은 <파일>.bak 로 남는다)
    python scripts/sanitize_director_focus.py \\
      --output_root ../Look-Before-Move/Director/output --run_glob 'trumans_c49_w*' \\
      --blend_objects /tmp/blend_objects.json --no_dry_run
"""

from argparse import ArgumentParser
from collections import Counter
from glob import glob
from os import path
from shutil import copyfile
import json

#    Cinematographer 가 asset id 로 읽는 키만 손댄다. `focus_position` / `focus_stability` 같은
#    자유 텍스트 키에도 "focus" 가 들어가는데, 거기 값은 문장이라 id 로 걸러내면 안 된다.
ID_LIST_KEYS = {
    "focus_ids", "secondary_focus_ids", "start_focus_ids", "focus_on_ids",
    "supporting_focus_ids", "contract_focus_ids", "original_focus_on_ids",
    "original_start_focus_ids", "secondary_focus_candidates",
}
ID_SCALAR_KEYS = {
    "primary_focus_id", "original_primary_focus_id", "canonical_primary_focus_id",
    "narrative_primary_focus_id",
}


def resolve(fid: str, names: set):
    """`_find_object` 와 같은 규칙으로 blend 오브젝트 이름을 찾는다. 못 찾으면 None.

    원본은 완전일치 → `asset_id` 커스텀 프로퍼티 → `<id>.` 접두사 → `<id>_` 접두사 순인데,
    커스텀 프로퍼티는 blend 를 열어야 보이므로 여기선 이름 기반 3종만 본다. 그래서 이 함수는
    **원본보다 보수적**이다 — 놓치는 쪽이지 없는 걸 지우는 쪽이 아니다.
    """
    norm = fid.strip().lower()
    if not norm:
        return None
    if norm in names:
        return norm
    for cand in sorted(names):
        if cand.startswith(f"{norm}.") or cand.startswith(f"{norm}_"):
            return cand
    return None


def is_ghost(fid: str, assets: set, names: set):
    """이 id 가 **빼야 할 유령**인가.

    세 갈래다:
      1. asset_index 안에 있다            → 진짜 asset, 둔다.
      2. blend 에서 해석이 안 된다        → `_find_object` 가 건너뛰므로 무해, 둔다
                                            (`sofa` `coffee_table` `painting` 등).
      3. 해석은 되는데 asset 이 아니다    → focus AABB 에 합쳐진다. **뺀다.**

    3번 판정에 "해석된 이름이 asset id 를 부분문자열로 갖는가"를 한 번 더 본다. `book` 은
    `book_left_01` 로, `oven_door` 는 `oven_door_01` 로, `frame_door_root_oven_door_01` 은
    자기 자신으로 붙는데 셋 다 **의도한 대상**(그 asset 이거나 그 asset 의 부모 empty)이다.
    이 예외가 없으면 실측 86 chunk 에서 770개를 빼는데, 그중 560개가 book/whiteboard/oven_door
    처럼 제대로 붙은 것들이다 — 뺄 것은 `window`(140) 과 `floor`(70) 둘뿐이다.
    """
    low = fid.strip().lower()
    if not low or low in assets:
        return False
    target = resolve(fid, names)
    if target is None:
        return False
    return not any(asset in target for asset in assets)


def scrub(node, assets: set, names: set, removed: Counter, primary_hits: Counter):
    """JSON 트리를 걸어가며 id 목록에서 유령 asset 을 뺀다. 제자리에서 고친다."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ID_LIST_KEYS and isinstance(value, list):
                kept = []
                for item in value:
                    if isinstance(item, str) and is_ghost(item, assets, names):
                        removed[item] += 1
                        continue
                    kept.append(item)
                node[key] = kept
            elif key in ID_SCALAR_KEYS and isinstance(value, str) and is_ghost(value, assets, names):
                #    스칼라는 안 지운다 (비우면 하류가 primary 없이 돈다). 세기만 한다.
                primary_hits[f"{key}={value}"] += 1
            scrub(value, assets, names, removed, primary_hits)
    elif isinstance(node, list):
        for item in node:
            scrub(item, assets, names, removed, primary_hits)


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_root", required=True, type=str)   # Director/output
    parser.add_argument("--run_glob", default="trumans_c49_w*", type=str)
    # blend 오브젝트 이름 전량(JSON 리스트). Blender 로 한 번 덤프해 두고 재사용한다.
    parser.add_argument("--blend_objects", required=True, type=str)
    # 기본은 dry run — 원본을 고치려면 `--no_dry_run` 을 명시해야 한다.
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--backup_suffix", default=".bak", type=str)
    args = parser.parse_args()

    names = {str(n).lower() for n in json.load(open(args.blend_objects, encoding="utf-8"))}

    runs = sorted(glob(path.join(args.output_root, args.run_glob)))
    removed, primary_hits, touched = Counter(), Counter(), []
    for run in runs:
        #    asset_index 는 run 마다 같지만, 다른 recording 을 섞어 돌릴 수 있으니 run 별로 읽는다.
        handoff = path.join(run, "outputs", "director_handoff_v1.json")
        if not path.exists(handoff):
            continue
        assets = {str(a).lower() for a in json.load(open(handoff, encoding="utf-8"))["asset_index"]}
        for file in sorted(glob(path.join(run, "outputs", "*.json"))):
            data = json.load(open(file, encoding="utf-8"))
            before = Counter(removed)
            scrub(data, assets, names, removed, primary_hits)
            if removed == before:
                continue
            touched.append(file)
            if args.dry_run:
                continue
            backup = file + args.backup_suffix
            if not path.exists(backup):
                copyfile(file, backup)
            with open(file, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)

    print(f"{'runs':22s} {len(runs)}")
    print(f"{'dry_run':22s} {args.dry_run}")
    print(f"{'고친 파일':22s} {len(touched)}")
    print(f"{'뺀 id (총)':22s} {sum(removed.values())}")
    for key, count in removed.most_common():
        print(f"  {key:34s} {count}")
    if primary_hits:
        print(f"{'!! 스칼라 primary':22s} {sum(primary_hits.values())}  (안 지웠다 — 확인 필요)")
        for key, count in primary_hits.most_common():
            print(f"  {key:34s} {count}")
    for file in touched:
        print(f"  touched {file}")


if __name__ == "__main__":
    main()
