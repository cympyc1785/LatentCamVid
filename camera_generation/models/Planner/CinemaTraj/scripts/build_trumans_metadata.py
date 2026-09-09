"""TRUMANS-Lite manifest 737개 → Vista 와 **같은 스키마**의 `metadata.csv`.

왜 필요한가. `build_bank_captions.py` 는 캡션의 `event` 필드를 `metadata.csv` 의 `prompt` 열
첫 문장에서 가져오고(`load_events`), `--videos all` 의 영상 목록도 그 csv 의 `video` 열에서
만든다. TRUMANS 는 그 csv 가 없어서 캡션이 `⚠ event 비어 있음` 으로 떨어진다. 그런데 그 정보는
이미 `out/trumans_recon/<recording>_s3f0k6/manifest_a<NN>.json` 의 `caption.{target,event}` 에
들어 있다 — 그러니 **캡션 스크립트를 고칠 게 아니라 csv 를 만들어 주면 된다**(코드 변경 0).

event 는 TRUMANS 액션 라벨이라 명령형(`put down the book with both hands`)이다. Vista 의
prompt 는 사람이 쓴 서술문이므로 눈금을 맞추려고 3인칭 서술로 바꾼다. 어휘가 26개 동사머리로
닫혀 있어서 휴리스틱 대신 **명시적 gerund 표**를 쓴다 — `open` 에 자음중복 규칙을 돌리면
`openning` 이 나오는 종류의 조용한 오류를 애초에 만들지 않기 위함.

사용 예시:

    python scripts/build_trumans_metadata.py                       # 737행 전부
    python scripts/build_trumans_metadata.py --out /tmp/meta.csv --dry_run
"""

import csv
import json
import re
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

WORK_DEFAULT = path.join(path.dirname(path.dirname(path.abspath(__file__))), "out", "trumans_recon")
OUT_DEFAULT = "/data1/cympyc1785/data/TRUMANS-Lite/metadata.csv"

# Vista `metadata.csv` 와 열 이름·순서를 그대로 맞춘다. `load_events` 는 video/prompt 만 읽지만
# 다른 하류(스플릿·필터)가 같은 reader 를 쓰므로 열이 빠지면 KeyError 로 죽는다.
FIELDS = ["name", "video", "camera", "seed", "prompt", "dynamic", "do_sky_seg", "source", "video_id"]

# 명령형 동사머리 → 3인칭 현재진행. 26종 전수(2026-08-30 737 manifest 실측).
GERUND = {
    "pick": "picking", "put": "putting", "stand": "standing", "sit": "sitting",
    "walk": "walking", "open": "opening", "close": "closing", "move": "moving",
    "drink": "drinking", "squat": "squatting", "wipe": "wiping", "lie": "lying",
    "turn": "turning", "write": "writing", "type": "typing", "slide": "sliding",
    "use": "using", "hold": "holding", "place": "placing", "make": "making",
    "tap": "tapping", "dial": "dialing", "water": "watering", "push": "pushing",
    "pull": "pulling", "carry": "carrying", "reach": "reaching", "wash": "washing",
}
# 동사로 시작하지 않는 라벨(`right hand picks up the pen`). 이미 3인칭이라 소유격만 붙인다.
POSSESSIVE_HEAD = re.compile(r"^(the )?(right hand|left hand|both hands)\b", re.I)


def sentence_of(target: str, event: str, kind: str):
    """`caption.{target,event}` → Vista prompt 스타일 한 문장.

    반환값이 `first_sentence()` 를 그대로 통과해야 하므로 문장은 **하나만** 만든다.
    """
    target = (target or "person").strip()
    event = " ".join(str(event or "").split())
    if not event:
        return f"A {target} in an indoor room."
    if POSSESSIVE_HEAD.match(event):
        return f"A {target}'s {re.sub(r'^the ', '', event, flags=re.I)}."
    head, _, tail = event.partition(" ")
    gerund = GERUND.get(head.lower())
    if gerund is None:                                   # 표에 없는 동사머리 = 어휘가 늘어난 것
        return f"A {target} in an indoor room, {event}."
    body = f"{gerund} {tail}".strip()
    if kind == "walk":                                   # `walk` 는 목적어가 없어 문장이 너무 짧다
        body = f"{body} through the room"
    return f"A {target} is {body} indoors."


def rows_from(work: str):
    """`manifest_a*.json` 전부 → csv 행. 정렬은 파일 경로 기준이라 실행마다 같다."""
    rows, skipped = [], []
    for file_path in sorted(glob(path.join(work, "*_s3f0k6", "manifest_a*.json"))):
        with open(file_path, encoding="utf-8") as file:
            manifest = json.load(file)
        caption = manifest.get("caption") or {}
        video = manifest.get("video")
        if not video:
            skipped.append(file_path)
            continue
        prompt = sentence_of(caption.get("target"), caption.get("event"),
                             (manifest.get("action") or {}).get("kind", ""))
        rows.append({
            "name": video,
            "video": video,
            "camera": ((manifest.get("source_camera") or {}).get("preset") or ""),
            "seed": "",
            # `dynamic` 은 SAM3 텍스트 프롬프트용 명사 목록이다. TRUMANS 는 동적 물체가
            # 사람 하나뿐이라(HUMAN_INDEX=1) 고정 목록으로 충분하다.
            "dynamic": "man,person,human",
            "do_sky_seg": "false",                       # 실내 렌더라 하늘이 없다
            "source": "trumans",
            "video_id": manifest.get("recording", ""),
            "prompt": prompt,
        })
    return rows, skipped


def main():
    parser = ArgumentParser()
    parser.add_argument("--work", default=WORK_DEFAULT, type=str)   # manifest 출처
    parser.add_argument("--out", default=OUT_DEFAULT, type=str)     # 쓸 csv 경로
    parser.add_argument("--dry_run", action="store_true")           # 안 쓰고 표만 출력
    args = parser.parse_args()

    rows, skipped = rows_from(args.work)
    assert rows, f"manifest 를 못 찾았다: {args.work}/*_s3f0k6/manifest_a*.json"

    if not args.dry_run:
        makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    recordings = sorted({row["video_id"] for row in rows})
    print(f"{'행':>8}  {len(rows)}")
    print(f"{'recording':>8}  {len(recordings)}")
    print(f"{'preset':>8}  {len(sorted({row['camera'] for row in rows}))}")
    print(f"{'문장':>8}  {len(sorted({row['prompt'] for row in rows}))} 종")
    if skipped:
        print(f"⚠ video 키 없는 manifest {len(skipped)}개: {skipped[:4]}")
    print(f"{'출력':>8}  {'(dry_run)' if args.dry_run else args.out}")
    for row in rows[:5]:
        print(f"  {row['video']:<28} {row['prompt']}")


if __name__ == "__main__":
    main()
