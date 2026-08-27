"""DL3DV `<scene>/prompts.json` + `<scene>/da3/tags/camera_tags.json` → `prompts_simple.json`.

왜: DL3DV 를 Vista4D/TRUMANS 와 **같은 텍스트 형식**으로 섞어 쓰려는데 지금 들어가는 문장은
형식도 출처도 다르다. 원래 `concise` 는 VLM 이 영상을 보고 쓴 산문이라 (a) 카메라와 장면 묘사가
한 문장에 섞여 있고 (b) 측정값이 아니라 **틀린다**. 실제 사례:

    concise  "The camera trucks left while panning right, then pans left while dollying backward
              and continues panning right."
    tags     ["... move left + yaw right", "... move left and backward + yaw right"]

여기서 concise 의 "pans left" 는 없는 움직임이다 (태그는 전 구간 yaw right). 그래서 캡션을
**측정된 `camera_tags.json` 에서** 다시 만든다:

    target: none. motion: truck left and pan right, then truck left, dolly out, and pan right

`target: none` 인 이유 (사용자 지시): DL3DV 는 정적 씬 코퍼스라 조준할 subject 가 없다.
이 코퍼스는 혼합 학습에서 **free-moving** 만 담당하고, 피사체 조준은 Vista4D/TRUMANS 가 맡는다.
`none` 을 비우지 않고 **글자로 적는** 이유는 "target 슬롯이 비었다"와 "target 슬롯이 없다"를
모델이 구분하게 하려는 것 — 같은 자리에 항상 토큰이 하나 있어야 코퍼스가 섞였을 때
`target:` 이 코퍼스 식별자로 새지 않는다.

세그먼트 키·`frame_idx` 는 **한 글자도 안 바꾼다**. 바뀌는 것은
`prompt_camera_with_scene_video.concise` 하나뿐이라 pose / avg_scale / seg list 전부 그대로
재사용된다. dataset 은 `prompts_file: prompts_simple.json` 한 줄로 갈아끼운다
(`main/dataset_dl3dv.py:_prompts_path` — `pose_source` 에 따라 `<scene>/` 또는 `<scene>/da3/`).

!! 태그는 **da3 pose** 로 쟀고 학습은 `pose_source: transforms`(COLMAP) 로 돈다.
   frame_idx 가 같은 같은 프레임 구간을 서로 다른 추정기로 잰 것이므로 방향은 일치해야 하지만
   같은 배열은 아니다. 크기(부사)를 안 넣는 이유이기도 하다 — 방향은 두 추정기가 합의하지만
   크기는 게이지가 다르다.

용어 변환: 태그의 축 단어를 Vista4D simple 캡션과 **같은 어휘**로 옮긴다. 그래야 두 코퍼스가
토큰 공간을 공유한다.
    move left/right      -> truck left/right        yaw left/right   -> pan left/right
    move forward/backward-> dolly in/out            pitch up/down    -> tilt up/down
    move up/down         -> pedestal up/down        roll left/right  -> roll left/right
    static (회전도 없음) -> hold still

한 세그먼트에 sub-shot 이 여러 개면 (실측 1/2/3 = 33%/45%/22%) `, then ` 으로 잇는다.
문구를 못 읽으면 **raise** 한다 — 조용히 빈 motion 을 흘리면 그 세그먼트만 텍스트가 사라진 채
학습된다.

env: 아무거나 (표준 라이브러리만 쓴다).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/latentcam/bin/python
    $PY scripts/data/make_prompts_simple_dl3dv.py \
        --root /data1/cympyc1785/data/DL3DV/scenes --meta_csv meta_worldtraj.csv --dry_run
    $PY scripts/data/make_prompts_simple_dl3dv.py \
        --root /data1/cympyc1785/data/DL3DV/scenes --meta_csv meta_worldtraj.csv
"""
import json
import os.path as osp
import re
from argparse import ArgumentParser

# 태그의 병진 축 -> 촬영 용어. Vista4D `make_prompts_simple.py` 의 SIMPLE_PHRASE 와 같은 어휘다.
TRANSLATION = {
    "left":     "truck left",
    "right":    "truck right",
    "forward":  "dolly in",
    "backward": "dolly out",
    "up":       "pedestal up",
    "down":     "pedestal down",
}
# 태그의 회전 -> 촬영 용어.
ROTATION = {
    "yaw left":    "pan left",
    "yaw right":   "pan right",
    "pitch up":    "tilt up",
    "pitch down":  "tilt down",
    "roll left":   "roll left",
    "roll right":  "roll right",
}
STATIC = "hold still"

# "Between frames 0 and 47: " 접두. 프레임 번호는 세그먼트 내부 상대값이라 캡션에 안 넣는다.
_PREFIX = re.compile(r"^Between frames \d+ and \d+:\s*")
# "move <축들>" + " + <회전>" — 둘 다 선택. 병진 자리에 'static' 이 오는 형태도 있다.
_PHRASE = re.compile(
    r"^(?:move\s+(?P<axes>.+?)|(?P<static>static))?"
    r"(?:\s*\+\s*(?P<rot>(?:yaw|pitch|roll)\s+(?:left|right|up|down)))?$")


def join_terms(terms):
    """['truck right','dolly in','pan left'] -> 'truck right, dolly in, and pan left'."""
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f"{terms[0]} and {terms[1]}"
    return ", ".join(terms[:-1]) + f", and {terms[-1]}"


def phrase_to_motion(line: str) -> str:
    """태그 description 한 줄 -> 촬영 용어 한 절. 못 읽으면 raise."""
    body = _PREFIX.sub("", str(line or "")).strip()
    match = _PHRASE.match(body)
    if not match or not (match.group("axes") or match.group("static") or match.group("rot")):
        raise ValueError(f"camera_tags description 을 못 읽었다: {line!r}")
    terms = []
    if match.group("axes"):
        # "right, down, and backward" / "right and forward" / "right"
        for axis in re.split(r",\s*and\s+|,\s*|\s+and\s+", match.group("axes").strip()):
            axis = axis.strip()
            if not axis:
                continue
            if axis not in TRANSLATION:
                raise ValueError(f"모르는 병진 축 {axis!r} (원문 {line!r})")
            terms.append(TRANSLATION[axis])
    if match.group("rot"):
        rot = re.sub(r"\s+", " ", match.group("rot").strip())
        if rot not in ROTATION:
            raise ValueError(f"모르는 회전 {rot!r} (원문 {line!r})")
        terms.append(ROTATION[rot])
    # 'static' 은 병진이 없다는 뜻일 뿐이라 회전이 있으면 그 회전만 낸다.
    return join_terms(terms) if terms else STATIC


def simple_prompt(tag_seg: dict, target: str) -> str:
    """camera_tags 의 세그먼트 dict -> "target: none. motion: ..." 한 줄."""
    lines = tag_seg.get("description") or []
    if not lines:
        raise ValueError("description 이 비어 있다")
    clauses, prev = [], None
    for line in lines:
        clause = phrase_to_motion(line)
        if clause != prev:          # 인접 sub-shot 이 같은 문구로 접히면 한 번만 (방어적)
            clauses.append(clause)
        prev = clause
    return f"target: {target}. motion: " + ", then ".join(clauses)


def read_scenes(root: str, meta_csv: str):
    """meta CSV 의 첫 컬럼(chunk) 목록. 헤더 한 줄을 건너뛴다."""
    with open(osp.join(root, meta_csv), encoding="utf-8") as file:
        rows = [line.strip() for line in file if line.strip()]
    return [r.split(",")[0] for r in rows[1:]]


def main(args):
    scenes = read_scenes(args.root, args.meta_csv)
    # pose_source 규칙과 같은 경로 규칙: transforms 면 <scene>/, da3 면 <scene>/da3/.
    sub = args.src_sub.strip("/")
    n_scene = n_seg = 0
    no_prompts, no_tags, key_mismatch, failed = [], [], [], []
    samples, nclause, lengths = [], {}, []
    for chunk in scenes:
        pdir = osp.join(args.root, chunk, sub) if sub else osp.join(args.root, chunk)
        src = osp.join(pdir, args.src_name)
        tags_path = osp.join(args.root, chunk, args.tags_rel)
        if not osp.isfile(src):
            no_prompts.append(chunk)
            continue
        if not osp.isfile(tags_path):
            no_tags.append(chunk)
            continue
        with open(src, encoding="utf-8") as file:
            prompts = json.load(file)
        with open(tags_path, encoding="utf-8") as file:
            tags = json.load(file)
        if set(prompts) != set(tags):
            # 세그먼트 집합이 어긋나면 캡션이 다른 프레임 구간을 설명하게 된다 -> 통째로 건너뛴다.
            key_mismatch.append(chunk)
            continue
        out = {}
        try:
            for key, seg in prompts.items():
                text = simple_prompt(tags[key], args.target)
                # 원본 세그먼트를 통째로 복사하고 캡션만 덮어쓴다 -- frame_idx / title /
                # separate_scene_text 등 부가 필드를 잃지 않아야 나중 분석이 그대로 된다.
                new = dict(seg)
                new["prompt_camera_with_scene_video"] = {"concise": text}
                new["prompt_source"] = f"simple_tags(from {args.src_name} + {args.tags_rel})"
                out[key] = new
                n_seg += 1
                k = len(tags[key].get("description") or [])
                nclause[k] = nclause.get(k, 0) + 1
                lengths.append(len(text))
                if len(samples) < 10:
                    samples.append((chunk.split("/")[-1][:12], key, text))
        except ValueError as err:
            failed.append((chunk, str(err)))
            continue
        n_scene += 1
        if not args.dry_run:
            with open(osp.join(pdir, args.out_name), "w", encoding="utf-8") as file:
                json.dump(out, file, ensure_ascii=False, indent=1)

    print(f"{'scene':10s}{'seg':>10s}")
    print(f"{n_scene:<10d}{n_seg:>10d}" + ("   [dry_run — 안 씀]" if args.dry_run else ""))
    print(f"\n[건너뛴 scene]  prompts 없음 {len(no_prompts)} / tags 없음 {len(no_tags)} / "
          f"세그먼트 키 불일치 {len(key_mismatch)} / 파싱 실패 {len(failed)}")
    for chunk, err in failed[:5]:
        print(f"   !! {chunk}: {err}")
    for name, lst in [("tags 없음", no_tags), ("키 불일치", key_mismatch)]:
        if lst:
            print(f"   {name} 예: {lst[:3]}")
    print(f"\n[sub-shot 수 분포] " + "  ".join(
        f"{k}: {v} ({100.0 * v / max(1, n_seg):.1f}%)" for k, v in sorted(nclause.items())))
    if lengths:
        lengths.sort()
        print(f"[캡션 길이] med {lengths[len(lengths) // 2]}  p90 "
              f"{lengths[int(0.9 * (len(lengths) - 1))]}  max {lengths[-1]} chars")
    print("\n[표본]")
    for sc, key, text in samples:
        print(f"  {sc}/{key}: {text}")
    if not args.dry_run:
        print(f"\n-> {args.root}/<scene>/{sub + '/' if sub else ''}{args.out_name}")
        print(f"   config: prompts_file: {args.out_name}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", required=True)                # DL3DV scene 루트
    parser.add_argument("--meta_csv", default="meta.csv")       # scene 목록 CSV (루트 기준)
    parser.add_argument("--src_sub", default="")                # "" = <scene>/ (transforms pose),
    #                                                             "da3" = <scene>/da3/
    parser.add_argument("--src_name", default="prompts.json")          # 읽을 파일
    parser.add_argument("--out_name", default="prompts_simple.json")   # 쓸 파일
    parser.add_argument("--tags_rel", default="da3/tags/camera_tags.json")  # 측정 태그 (scene 기준)
    parser.add_argument("--target", default="none")             # target 슬롯 문자열
    parser.add_argument("--dry_run", action="store_true")       # 통계만 내고 파일은 안 쓴다
    main(parser.parse_args())
