"""정적 명사 목록을 만든다 — `fit/ingest/sam3_static_instances.py` 가 SAM3 에 먹일 keyword.

왜 별도 스크립트인가: 정적 명사는 **동적 명사와 다른 규칙으로 걸러야** 한다. 동적 쪽은
`metadata.csv:dynamic` 이라는 저자 정답이 있지만 정적 쪽은 아무도 안 적어놨고, 그냥 VLM 이 뱉은
목록을 그대로 SAM3 에 넣으면 두 가지가 깨진다.

  ① **동적과 겹치는 명사** — avocado-slice 에서 VLM 은 `plant`/`bottle`/`bowl` 을 frame_mode 에
     따라 dynamic 에도 static 에도 넣는다. 그대로 두면 같은 물체가 `dyn_*` 와 `stat_*` 두 노드로
     생기고, `merge_duplicates` 의 3D IoU 가 0.5 를 못 넘으면 안 합쳐진 채 남아 VLM 프롬프트에
     같은 물체가 두 번 나온다. 그래서 동적 keyword 와 겹치면 **정적 쪽을 버린다**.
  ② **바닥/벽 같은 광역 표면** — `ground`/`wall`/`floor` 는 OBB 를 씌우면 씬 전체를 덮는 상자가
     되고, 그 상자는 `near` 엣지를 모든 노드에 걸어 관계 목록을 무의미하게 만든다. 게다가 이
     둘은 이미 기하로 처리된다 (`relations.ground_height` / `relations.find_wall_planes`).
     그래서 기본으로 제외한다 (`--no_drop_surfaces` 로 살릴 수 있다).

명사 출처 (`--source`):
    `vlm`    — `fit/graph/extract_nouns_vlm.py` 가 남긴 `vlm_nouns.json` 의 `static` 열.
               **기본 경로**이고 계획서가 확정한 기본값이다 (VLM 명사 + SAM3 text PCS).
    `prompt` — `metadata.csv:prompt` 에서 규칙 기반 추출. VLM 서버가 없을 때의 폴백이고,
               품질은 명백히 낮다 (아래 `nouns_from_prompt` 의 한계 참고).
    `auto`   — vlm_nouns.json 에 그 영상이 있으면 `vlm`, 없으면 `prompt`. **기본값**.

출력: `<output>/static_nouns.json` (`static_nouns_v1`) — 영상별 최종 keyword + 버린 명사와 사유.

예시:
    python fit/graph/extract_static_nouns.py --videos camel avocado-slice
    python fit/graph/extract_static_nouns.py --videos camel --source prompt --no_drop_surfaces
"""
import csv
import json
import re
from argparse import ArgumentParser
from os import makedirs, path

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
EVAL_DATA = "/data1/cympyc1785/data/Vista4D-Eval-Data"
# `extract_nouns_vlm.py` 의 기본 출력 위치와 실제로 돌려서 남긴 results 사본. 앞의 것이 우선.
VLM_NOUNS_CANDIDATES = (
    path.join(HERE, "out", "vlm_nouns", "vlm_nouns.json"),
    "/data1/cympyc1785/LatentCamVid/results/2026-08-20_detector_compare/vlm_nouns/vlm_nouns.json",
)

FORMAT = "static_nouns_v1"

# 광역 표면 — OBB 를 씌우면 씬 전체를 덮는다. §docstring ②
SURFACE_NOUNS = frozenset({
    "ground", "floor", "wall", "ceiling", "sky", "road", "street", "pavement", "sidewalk",
    "grass", "water", "sand", "snow", "background", "landscape", "scene", "room", "kitchen",
    # 51편 확장에서 실제로 올라온 것들. 전부 "씬 전체를 덮는 판"이라 위와 같은 이유로 뺀다.
    "court", "track", "field", "path", "trail", "lawn", "carpet", "rug", "terrain", "dirt",
    "asphalt", "beach", "ocean", "sea", "slope", "hill", "ceiling", "sidewalk", "runway",
})

# 규칙 기반 추출용. 관사/전치사 뒤의 명사구를 잡되 이 단어들은 머리명사가 될 수 없다.
PROMPT_ANCHORS = frozenset({
    "a", "an", "the", "of", "with", "on", "in", "at", "into", "from", "onto", "over",
    "under", "beside", "near", "behind", "through", "against", "between",
    "toward", "towards", "along", "across", "around", "above", "below", "beneath",
    "inside", "outside", "past",
})
NON_NOUNS = frozenset({
    "is", "are", "was", "were", "be", "being", "been", "has", "have", "had", "and", "or",
    "but", "as", "by", "to", "for", "that", "this", "these", "those", "it", "its", "their",
    "his", "her", "him", "she", "he", "they", "them", "which", "while", "when", "where",
    "time", "moment", "scene", "shot", "video", "camera", "frame", "atmosphere", "mood",
    "expression", "setting", "presence", "sense", "view", "focus", "detail", "feeling",
    "set", "feature", "part", "side", "top", "front", "back", "piece", "kind", "amount",
    "sunlit", "cozy", "domestic", "natural", "tranquil", "ongoing", "focused", "wooden",
    "glass", "orange", "snowy", "striped", "narrow", "distant", "bright", "dark", "soft",
})


def singularize(word: str):
    """복수형 s 만 떼는 최소 규칙. SAM3 는 단수 명사에서 더 안정적으로 물린다."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ss", "us", "is")) or not word.endswith("s") or len(word) <= 3:
        return word
    return word[:-1]


def nouns_from_prompt(prompt: str):
    """관사/전치사 뒤 명사구의 **머리단어**를 뽑는다. 규칙 기반 폴백.

    한계를 분명히 해둔다: POS tagger 가 없는 env 라(nltk/spacy 미설치) 이건 품사 분석이 아니라
    "a/the/of/with ... 다음에 오는 단어들의 마지막 하나" 라는 위치 규칙일 뿐이다. 그래서
    `a woman ... peeling` 같은 분사와 `meal preparation` 같은 추상명사를 그대로 물고 온다.
    `NON_NOUNS` + 접미사 규칙(`-ing`/`-ed`/`-ion`/`-ness`)으로 한 번 더 거르지만 그래도 VLM
    목록보다 훨씬 지저분하다 — `--source vlm` 이 되면 그쪽을 쓸 것. 머리단어 **한 개**만 내는 것도
    같은 이유다: `fence wall` 처럼 두 단어로 붙이면 `SURFACE_NOUNS` 필터를 그냥 빠져나간다.
    """
    words = re.findall(r"[a-z]+", prompt.lower())
    nouns, run = [], []
    for word in words + ["."]:
        if word in PROMPT_ANCHORS or word == ".":
            if run:
                head = singularize(run[-1])
                if head not in NON_NOUNS and len(head) > 2 \
                        and not head.endswith(("ing", "ed", "ion", "ness", "ly")):
                    nouns.append(head)
            run = []
        elif word in NON_NOUNS:
            run = []
        else:
            run.append(word)
    seen, output = set(), []
    for noun in nouns:
        if noun not in seen:
            seen.add(noun)
            output.append(noun)
    return output


def read_prompts(metadata_csv: str):
    """video -> prompt. metadata.csv 는 entry(video x camera) 단위라 영상당 여러 행이 있다.

    `prompt` 열이 **없는 코퍼스가 있다** — DynPose-LBM 의 csv 는 `video,dynamic` 두 열뿐이다
    (`extend_dynpose_metadata.py` 의 FIELDS). 그래서 열이 없으면 빈 문자열로 둔다. 이 값은
    `--source prompt` 일 때만 쓰이고, 그 경로는 아래 main() 에서 명시적으로 assert 한다 —
    조용히 명사 0개를 내면 정적 트리가 통째로 비는데 로그는 정상으로 보이기 때문이다.
    Vista 처럼 열이 있는 코퍼스에서는 동작이 글자 그대로 같다.
    """
    prompts = {}
    with open(metadata_csv, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            prompts.setdefault(row["video"], row.get("prompt", ""))
    return prompts


def read_dynamic_keywords(video: str, eval_data: str):
    """이미 정적에서 빼야 할 동적 keyword. **실제로 쓰인 것**(seg_instances/meta.json) 을 본다."""
    meta_path = path.join(eval_data, "eval_data", "seg_instances", video, "meta.json")
    if path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as file:
            return list(json.load(file)["keywords"])
    return []


def read_vlm_nouns(vlm_nouns_path: str, frame_mode: str):
    """video -> {"static": [...], "dynamic": [...]}. 같은 영상이 frame_mode 별로 여러 record 다."""
    with open(vlm_nouns_path, encoding="utf-8") as file:
        summary = json.load(file)
    records = {}
    for record in summary["records"]:
        if record.get("source") != "vlm":
            continue
        if record["frame_mode"] == frame_mode or record["video"] not in records:
            records[record["video"]] = {"static": record.get("static", []),
                                        "dynamic": record.get("dynamic", []),
                                        "frame_mode": record["frame_mode"]}
    return records


def is_surface(noun: str):
    """광역 표면인지. **머리단어(마지막 토큰)로도** 본다.

    VLM 은 `brick wall` / `dirt track` 처럼 수식어를 붙여서 내놓는데, 정확히 일치시키면
    `SURFACE_NOUNS` 를 그냥 통과한다 (`nouns_from_prompt` 가 머리단어 한 개만 내는 것과 같은
    이유 — §`nouns_from_prompt` docstring). 51편에서 `brick wall`/`dirt track`/`tile floor`
    가 실제로 이렇게 새어 들어왔다.
    """
    tokens = noun.split()
    return bool(tokens) and (noun in SURFACE_NOUNS or singularize(tokens[-1]) in SURFACE_NOUNS)


def select(candidates: list, dynamic: list, drop_surfaces: bool, max_nouns: int,
           max_surface_nouns: int = 0):
    """(kept, surfaces, dropped). dropped 는 `(noun, 사유)` — 조용히 버리지 않는다.

    `surfaces` 는 **버린 광역 표면 명사를 따로 모아둔 것**이다 (D62 재조준). OBB 노드로는
    여전히 안 올린다 — §docstring ② 의 이유(씬 전체를 덮는 상자 + near 엣지 전량)가 그대로
    살아 있다. 대신 `build_scene_graph.py --floor_seg_root` 가 이 마스크를 **지면 높이
    (`ground.ground_z`) 추정에만** 쓴다. 점군 z 하위 2% 분위수는 바닥이 화면에 조금만 나오면
    흔들리는데(D122 280 scene 실측: 최대 동적 노드 발밑 gap median 0.120u, 노드 높이의 48%),
    "바닥"이라고 지목된 픽셀만 보면 그 분위수를 안 거쳐도 된다.

    `max_surface_nouns=0` (기본)이면 빈 리스트라 기존 `static` 결과와 비트 동일하다.
    """
    dynamic_set = {d.lower() for d in dynamic}
    kept, surfaces, dropped, seen = [], [], [], set()
    for raw in candidates:
        noun = " ".join(str(raw).lower().split())
        if not noun:
            continue
        if noun in seen:
            dropped.append((noun, "중복"))
            continue
        seen.add(noun)
        if noun in dynamic_set:
            dropped.append((noun, "동적 keyword 와 겹침"))
        elif drop_surfaces and is_surface(noun):
            if len(surfaces) < max_surface_nouns:
                surfaces.append(noun)
                dropped.append((noun, "광역 표면 — 노드 아님, surface 로 따로 뺀다"))
            else:
                dropped.append((noun, "광역 표면 — ground/wall 은 기하로 처리한다"))
        elif len(kept) >= max_nouns:
            dropped.append((noun, f"max_nouns {max_nouns} 초과"))
        else:
            kept.append(noun)
    return kept, surfaces, dropped


def main(args):
    prompts = read_prompts(path.join(args.eval_data, "metadata.csv"))
    vlm_nouns_path = args.vlm_nouns or next((p for p in VLM_NOUNS_CANDIDATES if path.isfile(p)), None)
    vlm_records = read_vlm_nouns(vlm_nouns_path, args.frame_mode) if vlm_nouns_path else {}
    print(f"{'vlm_nouns':<14}{vlm_nouns_path or '(없음)'}  videos={sorted(vlm_records)}")

    videos = args.videos or sorted(prompts)
    entries = {}
    for video in videos:
        assert video in prompts, f"metadata.csv 에 없는 영상: {video}"
        dynamic = read_dynamic_keywords(video, args.eval_data)

        source = args.source
        if source == "auto":
            source = "vlm" if video in vlm_records else "prompt"
        if source == "vlm":
            assert video in vlm_records, f"{video}: vlm_nouns.json 에 record 가 없다 ({vlm_nouns_path})"
            candidates = vlm_records[video]["static"]
            # VLM 이 frame_mode 에 따라 dynamic 에 넣었던 명사도 정적에서 뺀다 (§docstring ①).
            dynamic = sorted(set(dynamic) | set(vlm_records[video]["dynamic"]))
        else:
            assert prompts[video], (
                f"{video}: metadata.csv 에 `prompt` 열이 없어 --source prompt 를 쓸 수 없다 "
                "(DynPose-LBM 은 video,dynamic 두 열뿐이다). --source vlm 을 쓸 것.")
            candidates = nouns_from_prompt(prompts[video])

        kept, surfaces, dropped = select(candidates, dynamic, args.drop_surfaces,
                                         args.max_nouns, args.max_surface_nouns)
        entries[video] = {
            "source": source, "frame_mode": vlm_records.get(video, {}).get("frame_mode"),
            "static": kept, "surface": surfaces,
            "candidates": list(candidates), "dynamic_excluded": dynamic,
            "dropped": [{"noun": n, "reason": r} for n, r in dropped],
        }

    makedirs(args.output, exist_ok=True)
    out_path = path.join(args.output, "static_nouns.json")
    merged = entries
    if args.merge and path.isfile(out_path):
        # `--videos` 로 1편만 돌리면 기본 동작은 나머지 51편을 통째로 날린다. 증분 추가용 경로.
        with open(out_path, encoding="utf-8") as file:
            previous = json.load(file).get("videos", {})
        print(f"merge: 기존 {len(previous)}편 + 새로 {len(entries)}편")
        merged = {**previous, **entries}
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump({"format": FORMAT, "config": vars(args), "videos": merged},
                  file, ensure_ascii=False, indent=1)

    print(f"\n{'video':<16}{'source':<8}{'cand':>5}{'kept':>5}  static nouns")
    for video, entry in entries.items():
        print(f"{video:<16}{entry['source']:<8}{len(entry['candidates']):>5}"
              f"{len(entry['static']):>5}  {entry['static']}")
    for video, entry in entries.items():
        if entry["dropped"]:
            print(f"  {video} dropped: "
                  + ", ".join(f"{d['noun']}({d['reason']})" for d in entry["dropped"]))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA, type=str)
    parser.add_argument("--output", default=path.join(HERE, "out", "static_nouns"), type=str)
    parser.add_argument("--vlm_nouns", default=None, type=str)   # None = VLM_NOUNS_CANDIDATES 순서대로

    parser.add_argument("--videos", nargs="*", default=None)     # None = metadata.csv 전체
    # auto = vlm_nouns.json 에 있으면 vlm, 없으면 prompt (계획서 기본값은 VLM 명사)
    parser.add_argument("--source", default="auto", choices=["auto", "vlm", "prompt"])
    # multi(0/16/32/48) 는 움직임이 보여야 가능한 dynamic/static 구분이 더 정확하다
    parser.add_argument("--frame_mode", default="multi", choices=["single", "multi"])

    # 광역 표면 제외. 끄면 ground/wall 도 노드가 되지만 near 엣지가 전부 걸린다 (§docstring ②)
    parser.add_argument("--drop_surfaces", action="store_true", default=True)
    parser.add_argument("--no_drop_surfaces", dest="drop_surfaces", action="store_false")
    # SAM3 시간은 keyword 수에 비례한다. 8개면 영상당 ~1분.
    parser.add_argument("--max_nouns", default=8, type=int)
    # 버린 광역 표면 중 `surface` 열로 따로 뺄 개수. **0 = 기본 = 예전 결과 비트 동일.**
    # `floor`/`ground` 를 지면 높이 추정에만 쓰려면 1~2 (§select docstring, D62 재조준).
    parser.add_argument("--max_surface_nouns", default=0, type=int)
    # 기본은 예전 그대로 통째로 덮어쓰기. `--videos` 로 몇 편만 추가할 땐 `--merge`.
    parser.add_argument("--merge", action="store_true", default=False)
    parser.add_argument("--no_merge", dest="merge", action="store_false")

    main(parser.parse_args())
