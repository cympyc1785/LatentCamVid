"""영상 프레임 → VLM → detector 에 먹일 명사 목록. 계획서 Stage 0 의 실제 구현.

왜 필요한가: 지금까지의 detector 비교에서 "VLM keyword" 라고 부른 것은 사실
`Vista4D-Eval-Data/metadata.csv` 의 `dynamic` 열, 즉 **Vista4D 저자가 손으로 적은 정답 명사**였다
(`sam3_seg_instances.py:30-42` 가 그걸 쉼표로 자를 뿐이다). `avocado`/`knife` 가 이미 들어 있는
목록을 주고 detector 가 avocado 를 찾았다고 말하는 건 공정한 비교가 아니다. 이 스크립트가 그
자리를 실제 VLM 으로 채워서, 배포 조건(사람이 라벨을 안 준다)에서의 명사 품질을 잰다.

dynamic / static 을 나눠 받는 이유: 하류가 서로 다르게 쓴다. dynamic 명사는 subject track 을
만들고(`sam3_seg_instances.py`), static 명사는 scene graph 의 관계 노드가 된다
(`scripts/sam3_static_instances.py`, 계획서 Part A Stage 0).

프레임 수는 두 조건을 다 잰다:
  `--frame_mode single` — frame0 1장. GDINO/Florence-2/RAM++ 와 **같은 입력**이라 비교가 공정하다.
  `--frame_mode multi`  — 시간 균등 `--num_frames` 장. 움직임이 보여야 dynamic/static 구분이 가능하다.

GPU 메모리는 여기서 못 잰다 — 모델이 별도 vLLM 서버 프로세스(포트 22002)에 있다. 잴 수 있는 건
wall-clock 과 토큰 수뿐이고, 서버 상주 메모리는 별도로 `nvidia-smi` 로 적는다.

사용 예시:
    python scripts/extract_nouns_vlm.py --videos camel avocado-slice \
        --frame_mode single multi --output <results>/vlm_nouns
"""
import json
import sys
import time
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))
VISTA4D_ROOT = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"
EVAL_DATA = "/data1/cympyc1785/data/Vista4D-Eval-Data"
# [new 2026-08-28] dynpose 등 다른 corpus 루트. 함수들이 module 상수를 읽으므로 argparse 전에
# sys.argv 를 훑는다 (caption_cameras_datadop 의 --gendop_root pre-scan 과 같은 이유).
# metadata.csv 가 없는 corpus 면 read_authored_keywords 가 빈 목록을 돌려준다 (비교 기준만 빠짐).
for _i, _arg in enumerate(sys.argv):
    if _arg == "--eval_data" and _i + 1 < len(sys.argv):
        EVAL_DATA = _arg_val = sys.argv[_i + 1]
    elif _arg.startswith("--eval_data="):
        EVAL_DATA = _arg.split("=", 1)[1]

sys.path.insert(0, HERE)
from lbm.vlm import VLMClient  # noqa: E402  — stdlib 만 쓰는 OpenAI 호환 클라이언트

# 프롬프트는 영문 (계획서 규칙). detector 가 먹을 수 있는 형태를 명시적으로 요구한다:
# 소문자 · 단수 · 1~2 단어 · 구체명사. GroundingDINO 는 caption 을 " . " 로 이어붙인 것을 받고
# 반환 라벨이 caption 의 부분문자열이라, 형용사구가 길면 phrase 가 엉킨다.
SYSTEM = """You are a vision annotator. You look at frames from a video and list the objects \
that an open-vocabulary object detector should be asked to find.

Rules for every noun you output:
- lowercase, singular, 1-2 words, a concrete visible object (not a material, mood, or activity)
- it must be visible in the frames you were shown
- no duplicates and no hypernym of another entry you already listed (say "camel", not both \
"camel" and "animal")
Answer with raw JSON only. No markdown fences, no commentary outside the JSON."""

USER = """These are {n} frame(s) from a single {width}x{height} video shot{order_note}.

List the objects in this scene, split by whether they move.

- "dynamic": objects that move or are moved during the shot (people, animals, vehicles, \
hand-held items). If a person is present, also list the specific body-worn or hand-held things \
that move with them.
- "static": objects that stay put and define the scene (furniture, walls, ground, plants, \
fixtures). Ceiling/sky/floor count only if they are a distinct visible surface.
- "subject": the single most important dynamic object, chosen from your "dynamic" list.

Return exactly this JSON object:
{{"dynamic": ["..."], "static": ["..."], "subject": "...", "reasoning": "one sentence"}}"""


def pick_frames(frames, frame_mode: str, num_frames: int = 4):
    """`single` 은 frame0 만 — detector 들과 같은 입력. `multi` 는 시간 균등 `num_frames` 장.

    D169. `num_frames` 가 인자가 된 이유: 4장은 49프레임을 12프레임 간격으로 훑는 것이라
    그 사이에만 나왔다 사라지는 물체(손에 쥔 물건, 지나가는 사람)를 통째로 놓친다. dynpose
    9.5k 편은 씬당 물체 수가 Vista4D-Eval-Data 보다 많아서 그 구멍이 SAM3 앵커 수로 곧장
    번진다. 함수 기본값 4 는 옛 동작(D145 51편 비교표)을 그대로 재현하고, 6 은 CLI 기본값이다.
    """
    if frame_mode == "single":
        return [0]
    return [int(round(v)) for v in np.linspace(0, len(frames) - 1, max(1, int(num_frames)))]


def validate(payload):
    """스키마 위반 목록. `chat_json` 이 이걸 그대로 붙여 재질의한다."""
    violations = []
    for key in ("dynamic", "static"):
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            violations.append(f'"{key}" must be a list of strings')
        elif key == "dynamic" and not value:
            violations.append('"dynamic" must not be empty')
    subject = payload.get("subject")
    if not isinstance(subject, str) or not subject:
        violations.append('"subject" must be a non-empty string')
    elif isinstance(payload.get("dynamic"), list) and subject not in payload["dynamic"]:
        violations.append(f'"subject" ({subject!r}) must be one of the entries in "dynamic"')
    return violations


def normalize(nouns):
    """소문자 · 공백 정리 · 순서 유지 중복 제거. detector caption 은 결정론적이어야 한다."""
    seen, output = set(), []
    for noun in nouns:
        clean = " ".join(str(noun).lower().split())
        if clean and clean not in seen:
            seen.add(clean)
            output.append(clean)
    return output


def read_authored_keywords(video: str):
    """비교 기준 — 저자가 손으로 적은 `metadata.csv:dynamic`. 우리가 만든 게 아니다.

    **원천에서 직접 읽는다.** 예전에는 SAM3 가 써 둔 `seg_instances/<video>/meta.json` 을
    읽었는데, 그건 같은 keyword 의 복사본일 뿐이면서 "SAM3 를 먼저 돌려야 명사 추출이 된다"는
    순서 의존을 만들었다 (52편 확장 파일럿에서 parkour 가 아직 안 끝나 FileNotFoundError 로
    전체가 죽었다). metadata.csv 는 항상 있으므로 순서가 사라진다. 없는 영상은 빈 목록 —
    비교 기준이 없을 뿐 명사 추출 자체는 된다.
    """
    import csv
    csv_path = path.join(EVAL_DATA, "metadata.csv")
    if not path.isfile(csv_path):        # dynpose 등 저자 keyword 가 없는 corpus — 비교 기준만 빠짐
        return []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["video"] == video:
                return [k.strip() for k in row["dynamic"].split(",") if k.strip()]
    return []


def run_video(video: str, frame_mode: str, client, args):
    sys.path.insert(0, VISTA4D_ROOT)
    from PIL import Image
    from utils.media import load_video

    started = time.time()
    frames, _ = load_video(path.join(EVAL_DATA, "eval_data", "recon_and_seg", video, "video.mp4"))
    decode_seconds = time.time() - started
    height, width = frames[0].shape[:2]

    indices = pick_frames(frames, frame_mode, args.num_frames)
    frame_dir = path.join(args.output, "frames", video)
    makedirs(frame_dir, exist_ok=True)
    image_paths = []
    for index in indices:
        image_path = path.join(frame_dir, f"{index:05d}.jpg")
        Image.fromarray(frames[index]).save(image_path, quality=95)
        image_paths.append(image_path)

    order_note = "" if len(indices) == 1 else \
        f", sampled in time order at frames {', '.join(str(i) for i in indices)} of {len(frames)}"
    prompt = USER.format(n=len(indices), width=width, height=height, order_note=order_note)

    started = time.time()
    payload, info = client.chat_json(prompt, images=image_paths, system=SYSTEM,
                                     validate=validate, max_repairs=args.max_repairs,
                                     label=f"{video}/{frame_mode}")
    vlm_seconds = time.time() - started

    authored = read_authored_keywords(video)
    if payload is None:   # 소진 — 파이프라인은 hard-fail 하지 않는다 (계획서 §B6)
        return {"video": video, "frame_mode": frame_mode, "source": "exhausted",
                "dynamic": [], "static": [], "subject": None,
                "authored_keywords": authored, "vlm_seconds": round(vlm_seconds, 3),
                "decode_seconds": round(decode_seconds, 3), "info": info}

    dynamic, static = normalize(payload["dynamic"]), normalize(payload["static"])
    recovered = [k for k in authored if k in dynamic]
    return {"video": video, "frame_mode": frame_mode, "source": "vlm",
            "frame_indices": indices, "resolution": [width, height],
            "dynamic": dynamic, "static": static,
            "subject": " ".join(str(payload["subject"]).lower().split()),
            "reasoning": payload.get("reasoning", ""),
            "authored_keywords": authored,
            "recovered_authored": recovered,
            "missed_authored": [k for k in authored if k not in dynamic],
            "extra_dynamic": [k for k in dynamic if k not in authored],
            "caption": " . ".join(dynamic),
            "caption_all": " . ".join(dynamic + static),
            "decode_seconds": round(decode_seconds, 3),
            "vlm_seconds": round(vlm_seconds, 3),
            "repairs": info["repairs"], "turns": info["turns"],
            "prompt_tokens": info["last_meta"]["prompt_tokens"],
            "completion_tokens": info["last_meta"]["completion_tokens"]}


def main(args):
    makedirs(args.output, exist_ok=True)
    client = VLMClient(api_base=args.api_base, model=args.model, temperature=args.temperature)
    served = [entry["id"] for entry in client.models().get("data", [])]
    print(f"{'served':<14}{served}", flush=True)

    records = []
    for frame_mode in args.frame_mode:
        for video in args.videos:
            record = run_video(video, frame_mode, client, args)
            records.append(record)
            print(f"[{frame_mode:<6}] {video:<14} {record['vlm_seconds']:6.2f}s  "
                  f"dyn={record['dynamic']}", flush=True)
            print(f"{'':<9} {'':<14} {'':>6}   static={record.get('static')}", flush=True)

    print("=" * 108)
    print(f"{'video':<15}{'mode':<8}{'sec':>7}{'ptok':>8}{'ctok':>7}{'rep':>5}"
          f"  {'n_dyn':>5}{'n_sta':>6}  recovered / authored")
    print("-" * 108)
    for record in records:
        print(f"{record['video']:<15}{record['frame_mode']:<8}{record['vlm_seconds']:>7.2f}"
              f"{record.get('prompt_tokens') or 0:>8}{record.get('completion_tokens') or 0:>7}"
              f"{record.get('repairs', 0):>5}"
              f"  {len(record['dynamic']):>5}{len(record['static']):>6}"
              f"  {len(record.get('recovered_authored', []))} / "
              f"{len(record['authored_keywords'])}"
              f"   missed={record.get('missed_authored')}")
    print("=" * 108)

    out_path = path.join(args.output, "vlm_nouns.json")
    if args.merge and path.isfile(out_path):
        # 영상 1편만 뒤늦게 추가할 때 기존 51편 record 를 날리지 않기 위한 경로.
        # 같은 (video, frame_mode) 는 새 record 로 갈아끼우고 나머지는 그대로 둔다.
        with open(out_path, encoding="utf-8") as file:
            previous = json.load(file).get("records", [])
        fresh = {(r["video"], r["frame_mode"]) for r in records}
        kept = [r for r in previous if (r["video"], r["frame_mode"]) not in fresh]
        print(f"merge: 기존 {len(previous)} record 중 {len(kept)} 유지")
        records = kept + records
    summary = {"config": vars(args), "served_models": served, "records": records}
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=1)
    client.save_trace(path.join(args.output, "trace"))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="+", default=["camel", "avocado-slice"])
    parser.add_argument("--frame_mode", nargs="+", default=["single", "multi"],
                        choices=["single", "multi"])   # single=frame0 (detector 와 동일 입력)
    # `multi` 에서 시간 균등으로 몇 장을 보여줄지. 4 = D145 옛 동작, 6 = 현재 기본 (D169).
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--output", default=path.join(HERE, "out", "vlm_nouns"))
    parser.add_argument("--api_base", default="http://127.0.0.1:22002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_repairs", type=int, default=3)
    # 기본은 예전 그대로 통째로 덮어쓰기. 증분 추가에는 `--merge`.
    parser.add_argument("--merge", action="store_true", default=False)
    parser.add_argument("--no_merge", dest="merge", action="store_false")
    # corpus 루트 (import 시점에 이미 소비됐다, 위 pre-scan 참조 — 여기 등록은 argparse 통과용)
    parser.add_argument("--eval_data", default=EVAL_DATA)
    main(parser.parse_args())
