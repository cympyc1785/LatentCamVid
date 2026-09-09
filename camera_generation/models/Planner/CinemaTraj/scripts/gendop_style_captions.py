"""우리 코퍼스 entry 를 **GenDoP 학습 분포의 문장**으로 다시 쓴다 (motion + target 만).

왜 필요한가: `--text_from_eval_dir` 로 넣는 우리 D121 캡션은 자연어 판본이라
`framing`(medium shot tightening to...) · `composition`(with the fence on the right...) 절이
붙어 있고, 어휘도 "dollies in along its own axis without re-aiming" 처럼 우리 태거 것이다.
GenDoP 는 DataDoP 캡션(`move forward` / `yaw left` 어휘)으로 학습됐으므로 그 문장은
**분포 밖**이고, 낮게 나온 수치가 모델 탓인지 문장 탓인지 갈리지 않는다.

여기서는 두 조각만 쓴다:
  motion  `da3/captions_gendop/<idx>_tag.json` 의 chunk 열 (`move`/`angular`)
          — 이건 우리가 GenDoP 파이프라인 세팅으로 **GT 포즈에서 직접 태깅**한 것이라
            어휘가 이미 DataDoP 과 같다. preset 이름으로 손매핑하지 않는 이유가 이것이다
            (`orbit_right` 의 실제 태그가 `move right` + `yaw left` 인지 포즈가 답한다).
  target  `da3/prompts.json[<idx>].caption_fields.target_text` (없으면 `target`)

문장 형태 (GenDoP `Concise Interaction` 레지스터):
    The camera moves forward towards the man in the hoodie.
    The camera moves right while yawing left focusing on the camel, then remains static.
전치사는 **첫 chunk 의 이동 방향**으로 고른다 — forward→`towards`, backward→`away from`,
그 외→`focusing on` (GenDoP 원본이 쓰는 `highlighting/focusing on` 과 같은 자리).

출력은 eval 폴더와 **같은 모양**이다 (`<out>/test/<prefix>_<scene>_<idx>_caption.json`,
키 `Concise Interaction`). 그래서 `gendop_release_infer.py --text_from_eval_dir <out>` 에
그대로 꽂힌다 — infer 쪽은 한 줄도 안 고친다.

사용:
    cd models/Planner/CinemaTraj
    python scripts/gendop_style_captions.py \
        --corpus /data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121 \
        --split  .../seg_list_vista4d_test.txt --prefix vista4d \
        --out    results/20260906_d156_gendop_d121/text_gendop_style
"""
import json
from argparse import ArgumentParser
from collections import Counter
from os import makedirs, path

TEXT_KEY = "Concise Interaction"          # eval 폴더 caption json 의 유일한 키

# DataDoP 태그 -> 문장 조각. tag 는 `move left, up, and backward` 처럼 방향이 붙으므로
# 앞머리 `move ` 만 `moves ` 로 바꾸면 나머지는 그대로 쓸 수 있다.
ANGULAR_PHRASE = {
    "static": "",
    "yaw left": "yawing left",
    "yaw right": "yawing right",
    "pitch up": "pitching up",
    "pitch down": "pitching down",
}
# 회전만 있는 chunk(=이동 static)는 "remains static while yawing left" 가 되면 모순이라
# 회전을 주절로 올린다.
ANGULAR_ALONE = {
    "yaw left": "yaws left",
    "yaw right": "yaws right",
    "pitch up": "pitches up",
    "pitch down": "pitches down",
}
# 뒤 절도 정동사구(`moves forward`)라 `followed by` 는 못 쓴다 (`followed by moves forward`).
JOINERS = ["", ", then ", ", then ", ", and finally "]


def chunk_phrase(move: str, angular: str):
    """한 chunk -> 동사구. `move`/`angular` 는 태그 문자열 그대로다."""
    ang = ANGULAR_PHRASE.get(angular, "")
    if move == "static":
        return ANGULAR_ALONE.get(angular, "remains static")
    body = "moves " + move[len("move "):] if move.startswith("move ") else move
    return f"{body} while {ang}" if ang else body


def target_clause(first_move: str, target: str):
    """전치사는 첫 chunk 의 이동 방향이 정한다 — 없는 방향에 `towards` 를 붙이지 않으려고."""
    if not target:
        return ""
    if "forward" in first_move:
        return f" towards {target}"
    if "backward" in first_move:
        return f" away from {target}"
    return f" focusing on {target}"


def build_sentence(chunks: list, target: str):
    """chunk 열 + target -> 한 문장. target 절은 **첫 절 뒤**에 한 번만 붙는다."""
    if not chunks:
        return f"The camera remains static{target_clause('', target)}."
    parts = [chunk_phrase(c["move"], c["angular"]) for c in chunks]
    parts[0] += target_clause(chunks[0]["move"], target)
    out = "The camera " + parts[0]
    for i, p in enumerate(parts[1:], start=1):
        out += JOINERS[min(i, len(JOINERS) - 1)] + p
    return out + "."


def entry_target(prompts: dict, entry: str):
    fields = prompts[entry].get("caption_fields", {})
    return str(fields.get("target_text") or fields.get("target") or "").strip()


def main(args):
    lines = [l.strip() for l in open(args.split, encoding="utf-8") if l.strip()]
    out_test = path.join(args.out, "test")
    makedirs(out_test, exist_ok=True)

    prompts_cache, written, missing_tag, no_target = {}, 0, [], []
    degenerate = []          # 조건이 하나도 안 남은 문장 (이동·회전·target 전부 없음)
    preset_sig, samples, lengths = Counter(), {}, []
    for line in lines:
        _, scene, entry = line.split("/")
        da3 = path.join(args.corpus, args.prefix, scene, "da3")
        if scene not in prompts_cache:
            with open(path.join(da3, "prompts.json"), encoding="utf-8") as f:
                prompts_cache[scene] = json.load(f)
        prompts = prompts_cache[scene]

        tag_path = path.join(da3, "captions_gendop", f"{entry}_tag.json")
        if not path.exists(tag_path):
            missing_tag.append(f"{scene}/{entry}")
            continue
        with open(tag_path, encoding="utf-8") as f:
            chunks = json.load(f).get("chunks", [])

        target = entry_target(prompts, entry)
        if not target:
            no_target.append(f"{scene}/{entry}")
        sentence = build_sentence(chunks, target)

        with open(path.join(out_test, f"{args.prefix}_{scene}_{entry}_caption.json"),
                  "w", encoding="utf-8") as f:
            json.dump({TEXT_KEY: sentence}, f, ensure_ascii=False, indent=1)
        if sentence == "The camera remains static.":
            degenerate.append(f"{scene}/{entry}")
        written += 1
        lengths.append(len(sentence.split()))
        preset = prompts[entry].get("preset", "?")
        preset_sig[(preset, " | ".join(f'{c["move"]}/{c["angular"]}' for c in chunks))] += 1
        samples.setdefault(preset, sentence)

    with open(path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "gendop_style_captions_v1", "corpus": args.corpus,
                   "split": args.split, "prefix": args.prefix, "text_key": TEXT_KEY,
                   "source": "captions_gendop/<idx>_tag.json chunks + "
                             "prompts.json caption_fields.target_text",
                   "written": written, "missing_tag": missing_tag[:20],
                   "no_target": no_target[:20],
                   "degenerate": degenerate}, f, ensure_ascii=False, indent=1)

    print(f"\n{'written':<14}{written}")
    print(f"{'missing_tag':<14}{len(missing_tag)}  {missing_tag[:3]}")
    # targetless preset(D90 aim=free)은 target 이 비는 게 정상이다. 문제는 그중 **회전까지
    # 임계 아래**로 태깅돼 문장에 조건이 하나도 안 남는 경우 — 그건 세서 보여준다.
    print(f"{'no_target':<14}{len(no_target)}  {no_target[:3]}"
          f"   (targetless preset 은 정상)")
    print(f"{'degenerate':<14}{len(degenerate)}  {degenerate[:3]}"
          f"   <- 'The camera remains static.' 뿐인 엔트리")
    if lengths:
        srt = sorted(lengths)
        print(f"{'words':<14}min {srt[0]}  med {srt[len(srt) // 2]}  max {srt[-1]}")
    print(f"\n{'preset':<26}{'tag signature':<46}{'n':>5}")
    for (preset, sig), n in sorted(preset_sig.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{preset:<26}{sig[:44]:<46}{n:>5}")
    print(f"\n-- preset 별 예문 ({len(samples)}) --")
    for preset in sorted(samples):
        print(f"  {preset:<26}{samples[preset]}")


if __name__ == "__main__":
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)     # latentcam 레이아웃 루트
    ap.add_argument("--split", required=True)      # seg_list_*.txt (`<prefix>/<scene>/<idx>`)
    ap.add_argument("--prefix", default="vista4d")  # 코퍼스 하위 폴더 == 파일명 접두사
    ap.add_argument("--out", required=True)        # `<out>/test/*_caption.json` 로 쓴다
    main(ap.parse_args())
