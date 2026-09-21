"""뱅크 변이 → `{target, event, framing, motion}` 캡션 + 학습 프롬프트.

왜 필요한가: 뱅크 행에는 카메라 기하(`preset`, `tau_max`, `subject_area_med`, `anchor_label`)만
있고 학습에 넣을 **문장**이 없다. `trumans_lite_to_dl3dv.py:86 caption_of` 는 TRUMANS 전용
(preset 문구 + action 라벨 한 줄)이라 Vista4D 에 그대로 못 쓴다 — Vista4D 는 action 라벨이
없고 대신 `metadata.csv` 의 씬 설명이 있다.

네 필드를 **따로** 저장하고 학습 프롬프트는 그 중 일부만 조립한다. 이유는 필드마다 신뢰도가
다르기 때문이다:
    target  뱅크가 실제로 조준한 노드 라벨 — 기하에서 나온다, 믿을 수 있다
    motion  preset + 크기 부사 — 역시 기하에서 나온다
    framing `subject_area_med` 버킷 — 렌더 실측이지만 **절대 면적**이라 shot size 지 품질이 아니다
    event   `metadata.csv` prompt 의 첫 문장 — 사람이 쓴 씬 설명이라 카메라와 무관하고,
            문장 경계를 규칙으로 자르므로 가끔 어색하다
**D121 부터 기본 형식은 자연어 한 문장**(`--prompt_style nl`, 사용자 지시):

    free-moving / targetless   `The camera slightly moves forward while panning left.`
    object-centric             `The camera steadily orbits around the larger pale camel walking
                                along the fence, keeping it in a medium shot that widens to a
                                medium wide shot as the wooden fence comes into frame.`

세 가지가 예전과 다르다 — ① target 이 라벨(`camel`)이 아니라 **referring expression**
(`instance_desc.json` 의 12~14 단어 지칭구)이다. 같은 라벨의 형제가 둘 이상일 때 라벨만으로는
어느 쪽인지 못 가리킨다. ② 크기 부사가 항상 켜진다 — 문장이 하나뿐이라 강도를 실을 데가
거기뿐이다. ③ framing 에 **composition** 이 붙는다: subject 말고 무엇이 같이 담기는지 /
들어오고 나가는지 (`sample_camera_bank.composition_stats` 가 뱅크에 실은 노드 id).
구조형이 아니라 자연어인 근거는 D117-a PE-AV 프로브다 (구조형이 4형식 중 꼴찌).

**D176 부터 nl 기본 문장은 `target` + `camera` 두 축만 남긴다** (사용자 지시 2026-09-10):

    The camera orbits around the woman in a purple shirt holding a suitcase.
    The camera pulls back, keeping the woman in a purple shirt in view.

위 ②③ 이 기본에서 빠졌다. 이유는 **둘 다 실측과 문장이 어긋난다**:
    크기 부사   `adverb_of` 가 요청이 아니라 **실현치**(`tau_max`/`pan_deg`)를 읽는다. d157
                55,462행 중 54%가 기하 제약으로 깎였는데 부사도 같이 깎여, 요청↔실현 대비가
                남는 축은 orbit sweep 하나뿐이다(캡션이 `sweep_deg` 를 안 읽어서). 최종
                목표는 **요청 강도**를 싣는 것이고(§adverb_of), 그때까지는 뺀다.
    framing     shot size 어휘가 실측 면적비와 구간이 겹친다 — d157 20,785 entry 중 49.6%가
                불일치(timeline head 어휘 tighten 61.7% / widen 57.1% / exit 82.8%). 게다가
                `subject_visible_frac`(가림)을 캡션이 한 번도 안 읽어서 "화면 안에 있지만
                완전히 가려진" 행이 close-up 을 약속한다.
`--magnitude` / `--nl_framing` 으로 예전 문장을 그대로 되살린다. `caption["framing"]` 구조체는
그대로 채워지므로(§caption_of 의 D166 층 구분) 나중에 지표로 쓰거나 되살릴 수 있다.

같은 지시로 **재조준 여부도 부정형을 버리고 최소쌍으로** 바꿨다 (§restore_legacy_reaim):
    dolly_in        "dollies in along its own axis without re-aiming"  ->  "dollies straight forward"
    dolly_in_look_at "dollies straight forward toward {target}"        (그대로)
즉 aim 은 **대상을 방향으로 부르는지**로만 갈린다. `--legacy_dolly_phrase` 가 옛 문구를 되살린다.

`--prompt_style fields` 는 예전 형식을 문자 단위로 그대로 다시 만든다. 그 경로에서
`--prompt_fields target,motion` (기본)이 네 필드 중 앞 둘만 쓰고, 나머지 둘은 JSON 에 남는다.

크기 부사(`--magnitude`)의 눈금은 `knob_kind` 별로 다르다 — tau 계열은 |t|/z_med, pan 계열은
회전 각도다. 같은 버킷 표를 두 축에 쓰면 pan 이 전부 "dramatically" 가 된다 (pan_deg 중앙값
28.5). `fields` 에서 껐던 이유는 같은 부사가 씬마다 다른 실제 이동량을 가리키기 때문이다
(씬 깊이로 나눈 값). 3-state 라 안 주면 형식이 정하고, 명시하면 형식과 무관하게 못 박힌다.

usage:
  python scripts/build_bank_captions.py --videos all --bank_dir hole_bank_k6_d99
  python scripts/build_bank_captions.py --videos camel --bank_dir hole_bank_k6_d99 --dry_run
  python scripts/build_bank_captions.py --videos all --bank_dir hole_bank_k6 \
      --prompt_style fields --out_name captions_fields.json   # 예전 형식 그대로
"""
import json
import re
import sys
from argparse import ArgumentParser
from os import path

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.presets import PRESETS, resolve_preset  # noqa: E402

METADATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/metadata.csv"
PRESETS_DEFAULT = path.join(CINEMATRAJ_ROOT, "configs", "caption_presets.json")

# 문장 경계. 약어(`Dr.`)를 자르지 않도록 마침표 뒤에 공백+대문자를 요구한다.
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def first_sentence(text: str, max_chars: int = 240):
    """씬 설명 → 첫 문장. 첫 문장이 너무 길면 `max_chars` 에서 단어 경계로 자른다."""
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    head = SENT_END.split(text)[0].strip()
    if len(head) <= max_chars:
        return head
    return head[:max_chars].rsplit(" ", 1)[0].rstrip(",;") + "..."


def bucket(value: float, table: list, key: str = "max"):
    """오름차순 버킷 표에서 `value` 가 들어가는 칸의 이름. 표 밖이면 마지막 칸."""
    for row in table:
        if float(value) < float(row[key]):
            return row["name"]
    return table[-1]["name"]


# D121. preset 문구의 target 자리. `configs/caption_presets.json` 이 예전에는 **문자열
# `"the subject"`** 를 그대로 갖고 있었고 `motion_of` 의 `{target}` 치환은 어느 문구에도 안
# 걸리는 no-op 였다 — 즉 라벨이 motion 문장에 도달한 적이 없다. 35개 문구를 `{target}` 으로
# 바꾸고, 무엇을 꽂을지는 **프롬프트 형식이** 정한다:
#     fields  옛 문자열 `"the subject"` 그대로 → 기존 캡션과 문자 단위로 같다.
#             (target 은 `target:` 절이 따로 부르므로 문구가 또 부르면 중복이다)
#     nl      referring expression ("the larger pale camel walking along the fence") 또는 라벨.
#             문장이 하나뿐이라 그 안에서 대상을 불러야 한다.
TARGET_SLOT = "{target}"
LEGACY_TARGET = "the subject"

# D122. 대명사. `nl_prompt` 가 대명사 자리에 **"it" 을 하드코딩**했고, 그래서 사람 target
# 1,961건(전체 36%)이 "sliding sideways past the woman in a striped shirt scooping avocado,
# keeping **it** in a medium shot" 이 됐다.
#
# 성별을 지칭구에서 실제로 가를 수 있다: `instance_desc.json` 의 54개 지칭구 핵어가
# man 886 / woman 272 / person 528 / 나머지는 사물이다. man→he, woman→she 는 확정이고,
# **person 은 VLM 이 성별을 안 밝힌 것**이라 추측하면 안 된다 → 단수 they.
MALE_HEADS = {"man", "boy", "guy", "gentleman", "male", "father", "son", "husband",
              "brother", "grandfather"}
FEMALE_HEADS = {"woman", "girl", "lady", "female", "mother", "daughter", "wife",
                "sister", "grandmother"}
# 성별 미상의 사람 / 복수. 둘 다 they 지만 이유가 다르다 — 앞은 모르는 것, 뒤는 여럿인 것.
PERSON_HEADS = {"person", "child", "kid", "adult", "teenager", "human",
                "rider", "cyclist", "dancer", "player", "athlete", "skier",
                "snowboarder", "climber", "skater", "runner", "worker"}
PLURAL_HEADS = {"people", "men", "women", "kids", "children", "couple", "pair",
                "group", "riders", "players", "dancers", "two"}
# `player` 를 사람으로 읽으면 "the black record player on the shelf" 가 they 가 된다 (실측).
# 앞 단어가 기기 수식어면 사람 판정을 취소한다.
DEVICE_MODIFIERS = {"record", "cd", "dvd", "music", "media", "video", "cassette"}

# 핵어구의 끝. 전치사 앞에서 끊는다.
HEAD_STOP = {"in", "with", "on", "at", "of", "behind", "near", "beside", "under",
             "over", "between", "from", "against", "by", "next", "along", "around",
             "inside", "outside", "to", "beneath", "atop", "across", "further"}
# -ing 로 끝나지만 분사가 아니라 명사인 것들. 이게 없으면 "the larger light-colored
# **building** on the left" 의 building 이 분사 경계로 읽혀 핵어가 `light-colored` 가 된다.
ING_NOUNS = {"building", "ceiling", "painting", "railing", "awning", "clothing",
             "opening", "string", "ring", "wing", "swing", "spring", "lighting",
             "bedding", "siding", "fencing", "carving", "drawing", "roofing",
             "earring", "stocking", "sibling", "king", "thing", "morning", "evening"}
WORD = re.compile(r"[a-z][a-z'\-]*")


def head_chunk(text: str):
    """지칭구 → **핵어구 단어 리스트**. "the white shelf behind the woman" → `[white, shelf]`.

    문자열에서 사람 단어를 찾으면 안 되는 이유가 저 예다 — 뒤쪽 수식절에 사람이 섞이면
    선반이 "she" 가 된다. 반대로 첫 단어를 쓰면 수식어에 걸린다 ("the small green moss
    patch" → `small`). 그래서 관사를 떼고 **첫 전치사/분사 앞까지**를 핵어구로 본다.
    분사는 핵어가 이미 하나 잡힌 뒤에만 경계로 친다 — 안 그러면 "the walking camel" 의
    `walking` 에서 끊긴다 — 그리고 `ING_NOUNS` 는 분사로 안 본다.
    """
    words = WORD.findall(str(text or "").lower())
    if words and words[0] in ("the", "a", "an"):
        words = words[1:]
    chunk = []
    for word in words:
        if word in HEAD_STOP:
            break
        if word.endswith("ing") and chunk and word not in ING_NOUNS:
            break
        chunk.append(word)
    return chunk


def head_noun(text: str):
    """지칭구 → 핵어 한 단어."""
    chunk = head_chunk(text)
    return chunk[-1] if chunk else ""


def pronoun_of(target_text: str):
    """지칭구 → (주격, 목적격, 복수동사인가). 못 가리면 `it` — 사물이 다수라 그게 안전하다."""
    chunk = head_chunk(target_text)
    head = chunk[-1] if chunk else ""
    if head in MALE_HEADS:
        return "he", "him", False
    if head in FEMALE_HEADS:
        return "she", "her", False
    if (head in PERSON_HEADS or head in PLURAL_HEADS) and not (
            len(chunk) >= 2 and chunk[-2] in DEVICE_MODIFIERS):
        return "they", "them", True
    return "it", "it", False


def adverb_of(variant: dict, mag_tables: dict):
    """뱅크 행 → 크기 부사(barely..dramatically). 눈금은 `knob_kind` 별로 다르다.

    tau 계열은 |t|/z_med, pan 계열은 회전 각도다 — 같은 표를 두 축에 쓰면 pan 이 전부
    "dramatically" 가 된다 (pan_deg 중앙값 28.5).
    """
    kind = variant.get("knob_kind") or "tau"
    table = mag_tables.get(kind, mag_tables["tau"])
    amount = float(variant["pan_deg"] if kind == "pan_deg" else variant["tau_max"])
    return bucket(amount, table)


def motion_of(variant: dict, spec: dict, mag_tables: dict, magnitude: bool = False,
              target_text: str = None, pronoun: tuple = ("it", "it", False)):
    """뱅크 행 → motion 문장.

    `magnitude` 를 켜면 preset 문구 앞에 크기 부사(barely..dramatically)를 붙인다.
    **`--prompt_style fields` 에서는 기본 끔**: 부사는 τ 사다리 단(段)을 텍스트로 노출하는
    것인데, 같은 부사가 씬마다 다른 실제 이동량을 가리킨다 (씬 깊이로 나눈 값이라).
    `nl` 은 사용자 지시로 기본 켬 — 문장이 하나뿐이라 강도를 실을 데가 여기밖에 없다.

    `target_text` 는 문구의 `{target}` 자리에 꽂을 말이다. 안 주면 `LEGACY_TARGET` 이라
    예전 문자열이 그대로 복원된다.

    D122. 문구 안에서 대상을 되받는 대명사도 슬롯이다 — `{it}` 목적격 / `{they}` 주격 /
    `{s}` 3인칭 단수 동사 어미. 13개 문구가 "keeping pace with **it** as **it** moves" 처럼
    `it` 을 박아 두고 있어서, `nl_prompt` 의 대명사만 고쳐도 사람 target 17건이 그대로
    남았다. `fields` 경로는 `LEGACY_TARGET`("the subject") → ("it","it",단수) 로 풀려
    **예전 문자열과 글자 단위로 같다**.
    """
    subj, obj, plural = pronoun
    phrase = (spec["phrase"].replace(TARGET_SLOT, target_text or LEGACY_TARGET)
              .replace("{it}", obj).replace("{they}", subj)
              .replace("{s}", "" if plural else "s"))
    if not magnitude:
        return f"the camera {phrase}"
    return f"the camera {adverb_of(variant, mag_tables)} {phrase}"


def load_label_map(label_map: str):
    """`--label_map` 경로 → (stem→명사 dict, 설정). 안 주면 `None` 이라 기존 동작 그대로다.

    Vista 는 라벨이 SAM3 텍스트 프롬프트(=사람이 쓴 명사)라 손댈 게 없다. TRUMANS 는 라벨이
    blend 오브젝트 이름(`WallInner.021`, `283217/model`)이라 그대로 두면 학습 프롬프트에
    `target: 283217/model` 이 들어간다. 뱅크 행의 `anchor_label` 은 추적 가능하도록 원래
    이름을 유지하고, **캡션 텍스트에서만** 명사로 바꾼다.
    """
    if not label_map:
        return None
    with open(label_map, encoding="utf-8") as file:
        return json.load(file)


def normalize_label(label: str, label_map):
    """blend 오브젝트 이름 → 자연어 명사. `label_map` 이 `None` 이면 입력을 그대로 돌려준다."""
    if label_map is None:
        return str(label)
    name = str(label).strip()
    if label_map.get("strip_suffix", True):                  # `Floor.008` / `book_left_01`
        name = re.sub(r"\.\d+$", "", name)
        name = re.sub(r"_\d+$", "", name)
    hit = label_map["stems"].get(name)
    if hit:
        return hit
    if re.match(r"^\d+/model\d*$", name):                    # 3D-FRONT 자산 ID — 이름이 없다
        return label_map.get("numeric_asset", "furniture")
    return label_map.get("fallback", "object")


# DataDoP 궤적(`dd_*`)은 **free-moving** 이다 (D83): `aim="traj"` 라 우리 subject 를 조준하지
# 않고 원본 촬영 회전을 그대로 쓴다. 그래서 캡션에 target 절을 붙이면 거짓말이 된다 — 텍스트는
# "이 물체를 본다"고 하는데 궤적은 그 물체를 프레임 밖에 둘 수 있다. `framing` 도 subject 면적
# 버킷이라 같은 이유로 뺀다. 남는 건 `event`(씬 설명) + `motion`(카메라가 한 일) 둘이다.
FREE_MOVING_PREFIX = "dd_"

# DataDoP `move`/`angular` 라벨 → 영문 문구. preset 마다 문구를 손으로 적는 대신 **라벨에서
# 합성**한다 — `dd_*` 이름은 라벨 분포에서 자동 생성되므로(47종 × top_k) 손으로 적으면
# `configs/caption_presets.json` 이 새 shape 을 뽑을 때마다 어긋난다.
MOVE_PHRASE = {"forward": "forward", "backward": "backward", "left": "left", "right": "right",
               "up": "upward", "down": "downward"}
ANGULAR_PHRASE = {"yaw left": "panning left", "yaw right": "panning right",
                  "pitch up": "tilting up", "pitch down": "tilting down",
                  "roll left": "rolling left", "roll right": "rolling right"}


def free_moving_motion(labels: dict, adverb: str = ""):
    """DataDoP 결합 라벨 → motion 문장. `{"move": ..., "angular": ...}` -> str.

    target 을 안 부르므로 `{target}` 치환이 없다 — 문장이 카메라 자신의 움직임만 말한다.

    D121. `adverb` 를 주면 `motion_of` 와 **같은 자리**("the camera" 뒤)에 붙인다. 예전에는
    `--magnitude` 가 preset 경로에만 걸려 있어서, 부사를 켜도 코퍼스의 47.8% 인 `dd_*` 행은
    부사가 없었다 — 같은 프롬프트 안에서 두 부류가 문장 구조로 구분됐다는 뜻이다.
    기본값 `""` 이면 예전 문자열 그대로다.
    """
    move, ang = labels.get("move", "static"), labels.get("angular", "static")
    dirs = [MOVE_PHRASE[d] for d in ("forward", "backward", "left", "right", "up", "down")
            if d in move]
    if not dirs:
        head = "holds position"
    elif len(dirs) == 1:
        head = f"moves {dirs[0]}"
    else:
        head = "moves " + " and ".join((", ".join(dirs[:-1]), dirs[-1]))
    tail = ANGULAR_PHRASE.get(ang)
    if not dirs:
        # 안 움직이는데 "barely holds position" 은 말이 안 된다 — 부사는 실제로 있는 움직임
        # (회전)에 붙이고, 회전도 없으면 통째로 뺀다. 뱅크의 τ 는 여기서 0 이다.
        lead = f"{adverb} " if (adverb and tail) else ""
        return f"the camera {head}" + (f" while {lead}{tail}" if tail else "")
    lead = f"{adverb} " if adverb else ""
    return f"the camera {lead}{head}" + (f" while {tail}" if tail else "")


def load_external_labels(shapes_json: str):
    """`datadop_shapes_v1` → {preset 이름: {"move","angular"}}. 안 주면 `{}` (기존 동작)."""
    if not shapes_json:
        return {}
    with open(shapes_json, encoding="utf-8") as file:
        blob = json.load(file)
    assert blob.get("format") == "datadop_shapes_v1", f"모르는 포맷: {blob.get('format')}"
    return {s["name"]: {"move": s["move"], "angular": s["angular"]} for s in blob["shapes"]}


def load_anchor_desc(out_root: str, video: str, enabled: bool = True):
    """`out/<video>/instance_desc.json` → {node_id: referring expression}. 없으면 `{}`.

    D121, 사용자 지시: target 을 `"woman"` 이 아니라 `"the woman in a blue shirt drinking a
    coffee"` 처럼 **대상을 명확히 지칭하는 표현**으로 부른다. 그 문구는
    `describe_instances_vlm.py` 가 SAM3 마스크를 근거로 VLM 에게 받아 둔 것이다.

    파일이 없으면 조용히 `{}` 로 떨어져 라벨 경로(`normalize_label`)가 그대로 돈다 — 아직
    `instance_desc.json` 이 없는 영상(TRUMANS · dynpose 전량)이 여기서 죽으면 안 된다.
    **없다는 건 호출자가 세어서 요약표에 남긴다** (조용한 fallback 은 조용한 품질 저하다).
    """
    if not enabled:
        return {}
    target = path.join(out_root, video, "instance_desc.json")
    if not path.isfile(target):
        return {}
    with open(target, encoding="utf-8") as file:
        blob = json.load(file)
    assert blob.get("format") == "lbm_instance_desc_v1", f"모르는 포맷: {blob.get('format')}"
    return {node_id: str(rec["description"]).strip().rstrip(".")
            for node_id, rec in blob["descriptions"].items() if rec.get("description")}


def load_desc_override(desc_override: str):
    """`--desc_override` 경로 → `{video: {node_id: 지칭구}}`. 안 주면 `{}` 로 기존 동작 그대로.

    왜 필요한가: `instance_desc.json` 의 지칭구는 VLM 이 쓴 것이라 **사람이 부르고 싶은 이름과
    다를 수 있다**. 사용자 지시 2026-09-21 — car-roundabout 은 VLM 이 "the blue car driving on
    the left side of the road" 라 적었지만 실제로는 회색 차이고, golf 는 "the human in a light
    blue shirt and dark cap swinging a golf club" 이 길어서 "a man playing a golf" 로 줄인 채
    같은 카메라를 다시 뽑아 보고 싶다는 요청이다.

    `instance_desc.json` 을 직접 고치지 않는 이유: 그건 VLM 산출물이고 다음 세대에서 다시
    구워지면 손본 게 조용히 날아간다. 덮어쓰기는 **세대 설정**으로 남겨야 재현된다.
    """
    if not desc_override:
        return {}
    with open(desc_override, encoding="utf-8") as file:
        blob = json.load(file)
    return {video: {str(node): str(text).strip().rstrip(".")
                    for node, text in table.items()}
            for video, table in blob.items()}


def target_text_of(variant: dict, label_map, desc_map: dict):
    """뱅크 행 → 캡션이 부를 대상 이름. desc 가 있으면 그것, 없으면 예전 라벨 경로."""
    hit = (desc_map or {}).get(str(variant.get("anchor_id")))
    return hit or normalize_label(variant["anchor_label"], label_map)


def split_ids(variant: dict, key: str):
    """뱅크의 노드 id 열 → 리스트. `bank.csv` 경로는 `|` 로 이어져 오고 JSON 은 리스트다."""
    value = variant.get(key)
    if isinstance(value, str):
        return [v for v in value.split("|") if v]
    return list(value or [])


def join_phrases(items: list):
    """["a", "b", "c"] -> "a, b and c". 캡션 안에서만 쓰므로 옥스퍼드 콤마는 안 쓴다."""
    if len(items) <= 1:
        return items[0] if items else ""
    return " and ".join((", ".join(items[:-1]), items[-1]))


def composition_of(variant: dict, desc_map: dict, label_map=None, max_items: int = 2):
    """subject **말고 화면에 담기는 것** → 영문 절. 뱅크에 열이 없으면 빈 문자열.

    D121, 사용자 지시: framing 은 shot scale 만이 아니라 "어떤 것들이 같이 화면에 담기는지,
    안 보이게 되거나" 까지다. 재료는 `sample_camera_bank.composition_stats` 가 뱅크에 실어 둔
    `in_frame_ids`/`enter_ids`/`exit_ids` (노드 id) 이고, 문구는 여기서 붙인다 — 뱅크는
    id 만 들고 있어야 문구를 바꿀 때 다시 굽지 않는다.

    들어옴/나감을 **먼저** 쓰고 남는 자리에만 "같이 있음"을 쓴다. 변화가 정보량이 크고,
    `in_frame_ids` 는 씬의 붙박이(벽·바닥·책상)라 거의 항상 같은 말이 되기 때문이다.
    id 를 문구로 못 바꾸면(desc 도 label_map 도 없는 노드) 통째로 뺀다 — `stat_3` 같은 내부
    id 가 학습 프롬프트에 새는 것보다 절이 없는 편이 낫다.
    """
    def phrase(node_id):
        return (desc_map or {}).get(node_id, "")

    inside = [p for p in map(phrase, split_ids(variant, "in_frame_ids")) if p]
    entering = [p for p in map(phrase, split_ids(variant, "enter_ids")) if p]
    leaving = [p for p in map(phrase, split_ids(variant, "exit_ids")) if p]
    inside = [p for p in inside if p not in entering and p not in leaving]
    clauses = []
    if entering:
        verb = "comes" if len(entering[:max_items]) == 1 else "come"
        clauses.append(f"as {join_phrases(entering[:max_items])} {verb} into frame")
    if leaving:
        verb = "leaves" if len(leaving[:max_items]) == 1 else "leave"
        clauses.append(f"as {join_phrases(leaving[:max_items])} {verb} the frame")
    if inside and not clauses:
        # 변화 절이 하나라도 있으면 "같이 있음"은 안 쓴다. "as A leaves the frame with B also in
        # frame" 은 두 절이 접속사 없이 붙어 읽히지 않고, 붙박이(벽·울타리)는 거의 모든 변이에
        # 같은 말로 들어가 문장만 길어진다.
        clauses.append(f"with {join_phrases(inside[:max_items])} also in frame")
    return " ".join(clauses[:2])


def framing_parts(variant: dict, cfg: dict, timeline: bool = False):
    """(shot scale 이름, 끝 이름 or None, "tighten"|"widen"|"exit"|"never" or None).

    `framing_of`(예전 문자열)와 `framing_nl`(자연어 절)이 **같은 판정**을 쓰게 하려고
    떼어냈다 — 두 군데서 버킷을 각자 접으면 문장과 필드가 조용히 어긋난다.

    D122. 면적 0 을 버킷으로 접지 않는다. 0 은 **화면 밖**인데 표의 맨 아래 칸이
    `max_area 0.005` 라 그냥 접으면 "extreme wide shot" 이 나온다 — 안 보이는 대상을
    "extreme wide shot 에 담고 있다"고 말하게 된다 (d121 실측: 끝 면적 0 이 8,206행 중
    1,554행 18.9%, 그 중 595건이 실제로 "keeping it in a ... shot" 을 뱉었다).
        `exit`  시작은 보이고 끝이 0 → "... until he leaves the frame"
        `never` 양 끝 다 0 (32행 0.4%) → framing 절 자체를 버린다. 조준은 했지만 한 번도
                안 담긴 변이라 shot scale 을 말할 근거가 없다.
    """
    table = cfg["framing_buckets"]
    med = bucket(variant["subject_area_med"], table, "max_area")
    if not timeline:
        return med, None, None
    seq = variant.get("subject_area_seq")
    if isinstance(seq, str):                      # bank.csv 경로는 `|` 로 이어져 온다
        seq = [float(v) for v in seq.split("|") if v]
    if not seq or len(seq) < 2:
        return med, None, None
    edge = 3 if len(seq) >= 6 else 1
    head_area = sorted(seq[:edge])[edge // 2]
    tail_area = sorted(seq[-edge:])[edge // 2]
    if tail_area <= 0.0:
        return (None, None, "never") if head_area <= 0.0 else \
               (bucket(head_area, table, "max_area"), None, "exit")
    head = bucket(head_area, table, "max_area")
    tail = bucket(tail_area, table, "max_area")
    if head == tail:
        return med, None, None
    order = [row["name"] for row in table]
    return head, tail, ("tighten" if order.index(tail) > order.index(head) else "widen")


def shot_phrase(name: str):
    """버킷 이름 → 관사 붙은 명사구. 버킷 이름은 절반이 이미 "shot" 으로 끝난다
    (`medium shot` / `wide shot`)고 절반은 아니다 (`close-up`) — 그래서 조건부로 붙인다.
    안 그러면 "a medium shot shot" 이 된다."""
    tail = name if name.rstrip().endswith("shot") else f"{name} shot"
    return f"{'an' if tail[:1].lower() in 'aeiou' else 'a'} {tail}"


def framing_nl(variant: dict, cfg: dict, timeline: bool = False,
               pronoun: tuple = ("it", "it", False)):
    """shot scale → 자연어 명사구. `framing_of` 와 같은 버킷 판정에서 나온다.

    D122. `way == "never"` 면 **빈 문자열**이다 — 호출부(`nl_prompt`)가 빈 framing 을
    "The camera [motion]." 한 절로 떨어뜨리므로, 못 지킬 구도 약속이 문장에서 사라진다.
    """
    head, tail, way = framing_parts(variant, cfg, timeline)
    if way == "never":
        return ""
    if way == "exit":
        subj, _, plural = pronoun
        return f"{shot_phrase(head)} until {subj} {'leave' if plural else 'leaves'} the frame"
    if tail is None:
        return shot_phrase(head)
    verb = "tightens to" if way == "tighten" else "widens to"
    return f"{shot_phrase(head)} that {verb} {shot_phrase(tail)}"


def framing_of(variant: dict, cfg: dict, timeline: bool = False):
    """shot scale 문구. `timeline` 을 켜면 시작→끝 버킷이 바뀔 때 그 변화를 문장에 싣는다.

    왜: `subject_area_med` 는 `verify_frames`(기본 13) 프레임의 **median** 이라 push-in 과
    pull-out 과 정지가 전부 같은 한 숫자로 접힌다 (D119). 뱅크가 `subject_area_seq` 를 싣게
    되면서 접기 전 배열이 남았으므로 그 양 끝을 읽는다.

    양 끝을 **한 프레임씩** 읽지 않는 이유: median 이 주던 이상치 보호가 끝점에는 없다. 소스가
    frame 0 에서 subject 를 잠깐 가리면 start 하나가 통째로 틀린다. 그래서 앞/뒤 3개씩을 다시
    median 으로 접는다 (배열이 6개 미만이면 그냥 끝값).

    버킷이 **안 바뀌면 예전 문구 그대로**다 — 5% 면적 변화를 문장으로 부풀리지 않는다.

    D122. 이탈(`exit`/`never`)도 `framing_nl` 과 **같은 판정**에서 나온다 — 두 경로가
    갈리면 JSON 의 `framing` 과 프롬프트의 문장이 서로 다른 말을 하게 된다 (D105 규칙).
    """
    head, tail, way = framing_parts(variant, cfg, timeline)
    if way == "never":
        return ""
    if way == "exit":
        return f"{head} until the subject leaves the frame"
    if tail is None:
        return head
    return f"{head} {'tightening to' if way == 'tighten' else 'widening to'} {tail}"


def nl_extras(caption: dict, **extra):
    """`nl` 전용 키를 붙인다. `fields` 에서는 **안 붙여** JSON 이 예전과 문자 단위로 같다."""
    caption.update({"target_text": "", "target_in_motion": False,
                    "framing_nl": "", "composition": "", "target_view": False})
    caption.update(extra)
    return caption


def framing_dropped(variant: dict, framing_on_free: bool = True,
                    framing_min_in_frame: float = 0.0,
                    cfg: dict | None = None, timeline: bool = False,
                    exit_min_in_frame: float = 0.0) -> bool:
    """D143/D154. 이 변이의 캡션에서 **framing/composition 절을 빼야 하는가**. §free_aim_framing.

    `caption_of` 안에 두면 main() 이 몇 개가 빠졌는지 못 센다 — 캡션 파일 헤더에 그 수를
    남기려면 같은 판정을 두 곳에서 불러야 해서 함수로 뽑았다. 두 손잡이는 OR 다.

    D154 (사용자 지시 2026-09-06). `subject_in_frame` 임계를 **절 종류별로** 나눈다.
    `exit` 절("... until the subject leaves the frame")은 대상이 나간다고 **이미 말하고
    있으므로** in_frame 이 낮은 게 거짓이 아니라 정확한 서술이다. 임계 하나로 자르면
    정직했던 캡션을 지운다 — vista d128 train 9,389 entry 실측:

        절 종류                          n    in_frame<0.85     =0
        plain   "keeping it in a ..."   3182   220 ( 6.9%)      84
        change  "... tightening to ..." 4178   597 (14.3%)      71
        exit    "... leaves the frame"  1035  1026 (99.1%)     105

    0.85 일괄이면 1,843건이 빠지는데 그중 1,026건(56%)이 exit 이다. 실제로 못 지킬 약속은
    plain+change 의 817건뿐이다. 그래서 exit 에는 별도 임계 `exit_min_in_frame` 를 둬서
    **시작부터 한 번도 중앙에 안 든 것(≈0)만** 뺀다 — 그때는 "medium shot 으로 담고
    있다가" 라는 앞부분이 거짓이라서다.

    `cfg` 가 None 이면 절 종류를 못 읽으므로 **D143 그대로 임계 하나**로 돈다 (cfg 를
    안 넘기던 예전 호출부). `exit_min_in_frame == framing_min_in_frame` 로 부르면 cfg 를
    넘겨도 D143 과 결과가 같다.
    """
    if (not framing_on_free) and variant.get("aim") == "free":
        return True
    in_frame = variant.get("subject_in_frame")
    if not (isinstance(in_frame, (int, float)) and not isinstance(in_frame, bool)):
        return False
    thr = framing_min_in_frame
    if cfg is not None and framing_parts(variant, cfg, timeline)[2] == "exit":
        thr = exit_min_in_frame
    return thr > 0.0 and in_frame < thr


def caption_of(variant: dict, event: str, cfg: dict, magnitude: bool = False,
               label_map=None, external_labels: dict | None = None,
               framing_timeline: bool = False, desc_map: dict | None = None,
               style: str = "fields", framing_on_free: bool = True,
               framing_min_in_frame: float = 0.0,
               framing_exit_min_in_frame: float | None = None,
               nl_framing: bool = True):
    """뱅크 행 + 영상 event → 캡션 필드. 값이 비어도 키는 항상 남긴다 (하류가 get 을 안 쓰게).

    `style="nl"` (D121, 사용자 지시) 이면 한 문장짜리 프롬프트를 조립할 재료를 더 얹는다 —
    `target_text`(referring expression) · `target_in_motion`(문구가 이미 대상을 부르는가) ·
    `framing_nl`(shot scale 명사구) · `composition`(같이 담기는 것 / 들어오고 나가는 것).
    `fields` 에서는 그 키들을 **안 붙인다**: 예전 캡션 JSON 과 문자 단위로 같아야 한다.

    D143 (사용자 지시) 의 두 손잡이는 **framing 절만** 뺀다 (§free_aim_framing) — 둘 중 하나라도
    걸리면 뺀다. 함수 기본값은 **예전 캡션을 글자 그대로 재현**하고, 새 동작은 CLI 기본값이다.
      `framing_on_free=False`        `aim="free"` 전량에서 뺀다 (preset 축 판정)
      `framing_min_in_frame=THR>0`   그 변이의 실측 `subject_in_frame < THR` 이면 뺀다 (변이 축)

    D166 `nl_framing` 은 위 둘과 **다른 층**이다. D143 은 "그 변이는 framing 을 약속할 자격이
    없다"라 `caption["framing"]` 까지 비우지만(JSON 과 문장이 같은 말을 해야 한다, D105),
    `nl_framing=False` 는 **문장에서만** 뺀다 — 구조체에는 shot scale 이 그대로 남아
    나중에 되살리거나 지표로 쓸 수 있다. 그래서 두 손잡이는 겹쳐 쓸 수 있다:
    D143 이 자격을 판정하고, `nl_framing` 이 그걸 문장에 넣을지를 정한다.

    D154. `framing_exit_min_in_frame` 는 `exit` 절 전용 임계다 (`framing_dropped` 참조).
    `None` 이면 `framing_min_in_frame` 을 그대로 써서 **D143 과 결과가 같다** — 기본값이
    예전 캡션을 재현한다는 위 규칙을 이 손잡이도 지킨다.
    """
    adverb = adverb_of(variant, cfg["magnitude_buckets"]) if magnitude else ""
    if str(variant["preset"]).startswith(FREE_MOVING_PREFIX):
        labels = (external_labels or {}).get(variant["preset"])
        assert labels is not None, (
            f"free-moving preset '{variant['preset']}' 의 라벨을 못 찾았다 — 뱅크를 만들 때 쓴 "
            "`--external_shapes` JSON 을 `--external_shapes` 로 같이 넘길 것")
        # target/framing 은 **빈 문자열**로 둔다 (키를 지우지 않는다): `prompt_of` 가
        # `if caption.get(f)` 로 거르므로 프롬프트에서 자동으로 빠지고, JSON 스키마는 그대로다.
        # nl 도 마찬가지로 "The camera [motion]." 한 절뿐이다 (사용자 지시) — 조준을 안 하는
        # 궤적에 framing/composition 을 붙이면 지키지 못할 약속이 된다.
        caption = {"target": "", "event": event, "framing": "",
                   "motion": free_moving_motion(labels, adverb)}
        return nl_extras(caption) if style == "nl" else caption
    # 옛 이름으로 만들어 둔 뱅크(`straight_ease`, `orbit_left_arc`, `track_side_left` ...)도
    # 같은 문구로 떨어지게 alias 를 먼저 푼다 (D75 이름 정리). 의미 보존 매핑이라 캡션은 안 바뀐다.
    # **`aim` 을 같이 넘기는 이유** (D76): `dolly_in`/`dolly_out` 은 D75 에서 "재조준 안 함"의
    # 별칭이었는데 D76 에서 "재조준 함"의 정식 이름이 됐다. 뱅크 행이 들고 있는 aim 을 보고
    # 옛 뜻으로 되돌린다 — 이걸 안 하면 옛 4,065행이 반대 문구를 받는다.
    name = resolve_preset(variant["preset"], variant.get("aim"))
    spec = cfg["presets"].get(name)
    assert spec is not None, (
        f"`configs/caption_presets.json` 에 preset '{variant['preset']}' 이 없다 — "
        "새 preset 을 뱅크에 넣었으면 캡션 문구도 같이 넣어야 한다")
    # 이름이 말하는 aim 과 행이 기록한 aim 이 어긋나면 **이름 매핑이 틀린 것**이다. 캡션은
    # 조용히 그럴듯하게 나오지만 궤적과 반대를 말하게 되므로 여기서 멈춘다.
    if "aim" in variant and name in PRESETS:
        assert PRESETS[name][1] == variant["aim"], (
            f"preset '{variant['preset']}' -> '{name}' 의 aim 이 '{PRESETS[name][1]}' 인데 "
            f"뱅크 행은 '{variant['aim']}' 이다 — lbm/presets.py 의 alias 매핑을 고칠 것")
    label = normalize_label(variant["anchor_label"], label_map)
    # 문구의 `{target}` 자리에 무엇을 꽂을지는 **형식이** 정한다 (§TARGET_SLOT 주석):
    # fields 는 `target:` 절이 대상을 따로 부르므로 문구는 옛 문자열 그대로,
    # nl 은 문장이 하나뿐이라 referring expression 을 문구 안에서 부른다.
    target_text = target_text_of(variant, label_map, desc_map)
    # 정지 preset 에는 부사를 안 붙인다 — τ 가 0 근처라 항상 "barely" 가 나오고
    # "barely holds completely locked off" 는 문장이 안 된다. 강도를 말할 움직임이 없다.
    # 대명사도 형식이 정한다. fields 는 대상을 `the subject` 로 부르므로 ("it","it",단수) 라
    # 예전 문자열이 그대로 복원되고, nl 만 he/she/they 가 들어간다.
    pronoun = pronoun_of(target_text) if style == "nl" else ("it", "it", False)
    motion = motion_of(variant, spec, cfg["magnitude_buckets"],
                       magnitude and spec.get("axis") != "static",
                       target_text if style == "nl" else LEGACY_TARGET, pronoun)
    # `targetless` (D86): anchor 근처에 놓이기만 하고 **조준도 추종도 안 하는** preset.
    # `pan_*` 이 그 경우다 — `aim="traj"` 이고 `track_` 도 아니라 궤적이 subject 에 매여 있지
    # 않다 (실측 748 변이의 `subject_in_frame` median 0.50, 46.7% 가 0.5 미만). target 을 적으면
    # "계속 보인다"는 지키지 못할 약속이 되므로 `dd_*` 와 같이 target/framing 을 뺀다.
    # `track_truck_*` 는 같은 `aim="traj"` 지만 follow_gain 1.0 으로 실제로 따라가므로 제외
    # (median 1.000) — 그래서 판정을 `aim` 이 아니라 config 플래그로 둔다.
    if spec.get("targetless"):
        # D121. 판정이 `"subject" not in phrase` 에서 `{target}` 슬롯 유무로 바뀌었다 —
        # 문구가 이제 `"the subject"` 대신 슬롯을 들고 있어서, 옛 판정은 항상 통과한다.
        assert TARGET_SLOT not in spec["phrase"], (
            f"preset '{name}' 은 targetless 인데 문구가 대상을 지칭한다: '{spec['phrase']}' — "
            "target 이 none 이면 그 지칭은 가리킬 곳이 없다")
        caption = {"target": "", "event": event, "framing": "", "motion": motion}
        return nl_extras(caption) if style == "nl" else caption
    # ── §free_aim_framing (D143, 사용자 지시 2026-09-06) ───────────────────────────────────
    # `aim="free"` 는 **한 번도 재조준하지 않는다** — 시작 pose 의 회전을 49프레임 내내 들고
    # 간다. 그런데 위 `targetless` 게이트는 `caption_presets.json` 의 플래그만 보므로,
    # `aim="free"` 인 15개 preset(`track_truck_*` / `track_dolly_*` / `truck_*` / `dolly_*` /
    # `pedestal_*` / `static_hold` / `track_hold` / `track_pedestal_*` / `static_zoom_in`)이
    # 여기까지 내려와 framing 절("keeping it in a medium shot")과 composition 절을 받는다.
    # 그건 재조준 없이는 지킬 수단이 없는 약속이다 — 위 free-moving 분기가 이미 같은 이유로
    # framing 을 빼고 있는데(§FREE_MOVING_PREFIX 주석) 그 규칙이 `aim` 축에는 안 걸려 있었다.
    # 실측(vista d121 train, `corpus_axis_compare.py --framing_scope`): `track_*` ∧ aim=free
    # 1,415 변이 중 `subject_in_frame < 0.5` 가 448(31.7%)인데 캡션 프레이밍 약속은
    # 1,121(79.2%). 즉 학습 신호의 3분의 1이 캡션과 반대다.
    #
    # **motion 절과 target 은 그대로 둔다.** `track_*` 의 "tracks {target}" 은 거짓이 아니다 —
    # follow_gain 1.0 으로 실제로 병진 추종을 한다 (`PRESET_FOLLOW`). 거짓인 건 프레이밍
    # 약속뿐이라 그 절만 뺀다. 비-track 의 "sliding sideways past {target}" 도 마찬가지로
    # 참이다 (지나친다고 말하지 프레임에 유지한다고 말하지 않는다).
    #
    # **preset 축이 아니라 변이 축으로 가르는 게 기본인 이유** — aim="free" 라고 다 깨지는 게
    # 아니다 (vista d121 train, aim=free 만, `subject_in_frame` median / <0.85 비율):
    #     dolly_out          1.000 / 19.4%    track_dolly_out    1.000 /  4.8%
    #     track_hold         1.000 / 10.8%    static_hold        1.000 / 20.8%
    #     pedestal_down      0.923 / 48.7%    dolly_in           0.923 / 49.0%
    #     truck_right        0.692 / 63.1%    pedestal_up        0.538 / 66.5%
    #     truck_left         0.538 / 69.1%    track_truck_left   0.385 / 78.8%
    # `dolly_out`/`track_dolly_out` 은 자기 축으로 물러나므로 재조준 없이도 대상이 중앙에
    # 남는다 (D123 의 promote 분석과 같은 관찰). 그걸 preset 이름으로 뭉뚱그려 끊으면 멀쩡한
    # 신호를 버린다. 반대로 aim="look_at" 도 2.9% 는 프레임을 놓친다 — preset 축으로는 그쪽을
    # 아예 못 잡는다. 그래서 판정을 **그 변이가 실제로 프레임에 유지했는지**로 둔다.
    drop_framing = framing_dropped(
        variant, framing_on_free, framing_min_in_frame, cfg, framing_timeline,
        framing_min_in_frame if framing_exit_min_in_frame is None
        else framing_exit_min_in_frame)
    caption = {"target": label,
               "event": event,
               "framing": "" if drop_framing else framing_of(variant, cfg, framing_timeline),
               "motion": motion}
    if style != "nl":
        return caption
    if drop_framing:
        # `nl_prompt` 는 `framing_nl` 이 비면 `f"{head}."` 로 끊는다 — composition 도 같이
        # 빠져야 한다 (그 절은 framing 뒤에만 붙는다). target_text 는 남긴다: motion 문구의
        # `{target}` 슬롯이 이미 그걸 쓴다.
        return nl_extras(caption, target_text=target_text,
                         target_in_motion=TARGET_SLOT in spec["phrase"])
    # 문구가 이미 대상을 불렀으면 뒤 절은 대명사로 받는다 — 한 문장 안에서 같은
    # referring expression 을 두 번 부르면 (그게 12~14 단어다) 문장이 못 읽힌다.
    # D166. `nl_framing=False` 면 **문장에서만** 뺀다 — `caption["framing"]` 은 위에서 이미
    # 채워졌고 그대로 남는다 (사용자 지시 2026-09-08: "구조체로만 남겨두고 concise 에서는 빼둬").
    # `composition` 도 같이 빠진다: `nl_prompt` 에서 그 절은 framing 뒤에만 붙을 자리가 있어
    # (`f"{head}, {link} {framing}" + comp`) framing 없이 단독으로는 문장에 못 들어간다.
    if not nl_framing:
        # framing 절이 **대상을 부르는 유일한 자리**인 변이가 있다. motion 문구에 `{target}`
        # 슬롯이 없는 preset(`dolly_out` 계열)이 그렇고, `dolly_out` 은 grid2x2 의 `recede`
        # 기본값이라 흔하다 — d157 뱅크 `016a6379` 129행 실측으로 문장이 대상을 부르는 행이
        # 117 -> 105 로 12개 줄었고 12개 전부 `dolly_out` 이었다. scene 당 5개 중 1개가
        # 대상 없는 문장이 되면 target 축 대조(D84)가 그만큼 비므로 **최소 절**로 되살린다:
        # shot scale 도 composition 도 없이 이름만 부른다 ("keeping the man in view").
        # `drop_framing` 쪽에는 안 붙인다 — 거기선 프레임 유지가 실제로 거짓이다 (§D143).
        return nl_extras(caption, target_text=target_text,
                         target_in_motion=TARGET_SLOT in spec["phrase"],
                         target_view=(TARGET_SLOT not in spec["phrase"]
                                      and bool(target_text)))
    return nl_extras(caption, target_text=target_text,
                     target_in_motion=TARGET_SLOT in spec["phrase"],
                     framing_nl=framing_nl(variant, cfg, framing_timeline, pronoun),
                     composition=composition_of(variant, desc_map, label_map))


def track_gain_of(variant: dict):
    """`track_*` 변이가 **실제로** 추종했는지. 0 이면 캡션의 'tracks' 가 거짓말이 된다.

    `follow_gain` 은 뱅크가 쓴 실측값이다 (`"auto"` 면 풀린 값). 예전 포맷 뱅크에는 키 자체가
    없으므로 `None` 을 돌려 "옛 뱅크"와 "gain 0" 을 구분한다.
    """
    gain = variant.get("follow_gain")
    return None if gain is None else float(gain)


# 빈 값을 **그냥 빼면** free-moving(`dd_*`, 코퍼스의 47.8%)은 프롬프트가 `motion:` 하나로
# 시작한다. 모델이 "target 절이 없다"를 문장 구조만으로 알게 되고, 그 구조 자체가 지름길이 된다.
# 그래서 target 은 빈 값을 `none` 으로 **명시**한다 (D86, 사용자 지시) — 나머지 필드는 종전대로 뺀다.
NONE_FIELDS = ("target",)
NONE_VALUE = "none"


def nl_prompt(caption: dict):
    """캡션 필드 → **한 문장** 학습 프롬프트 (D121, 사용자 지시 형식).

        free-moving / targetless   `The camera [motion].`
        object-centric             `The camera [motion] around/toward [target],
                                    keeping it in [framing][ composition].`

    around/toward 는 preset 문구가 이미 갖고 있다 (`orbits around {target}`,
    `pushes in toward {target}`) — 여기서 전치사를 다시 고르지 않는다. 궤적과 문장이
    어긋나는 사고는 전부 두 군데서 같은 걸 각자 정할 때 났다.

    왜 구조형(`target: X. motion: Y`)이 아니라 자연어인가: PE-AV 프로브에서 구조형이 4형식
    중 꼴찌였다 (D117-a). `--prompt_style fields` 로 옛 형식을 그대로 다시 만들 수 있다.
    """
    motion = str(caption.get("motion") or "").strip().rstrip(".")
    if not motion:
        return ""
    head = motion[:1].upper() + motion[1:]
    framing = str(caption.get("framing_nl") or "").strip()
    if not framing:
        # D166 `--no_nl_framing`. framing 절이 없어도 대상 이름은 남긴다 (§caption_of).
        # 판정은 `caption_of` 가 미리 해서 `target_view` 로 넘긴다 — 여기서 다시 재면
        # D143 이 정직하게 뺀 변이까지 "in view" 라고 약속하게 된다.
        if caption.get("target_view"):
            who = str(caption.get("target_text") or "").strip()
            link = "with" if "keeping" in head else "keeping"
            return f"{head}, {link} {who} in view."
        return f"{head}."
    # D122. 대명사를 하드코딩하지 않는다 — 지칭구 핵어에서 he/she/they/it 을 고른다.
    who = (pronoun_of(caption.get("target_text"))[1] if caption.get("target_in_motion")
           else str(caption.get("target_text") or "").strip() or LEGACY_TARGET)
    # `track_look_at` 문구가 이미 "keeping pace with it" 이다 — 한 문장에 keeping 이 두 번
    # 나오면 어느 쪽이 framing 인지 안 읽힌다. 그때만 연결어를 바꾼다.
    link = f"with {who} framed in" if "keeping" in head else f"keeping {who} in"
    comp = str(caption.get("composition") or "").strip()
    return f"{head}, {link} {framing}" + (f" {comp}." if comp else ".")


def prompt_of(caption: dict, fields: list, style: str = "fields"):
    """고른 필드를 `field: value.` 로 이어 붙인다 (사용자 지시: target/motion 둘만).

    D86 변경 두 가지 — ① 절 끝을 **마침표로 끊는다**. 이전 `target: person motion: the camera ...`
    는 값과 다음 필드 이름 사이에 경계가 없어 `person motion` 이 한 명사구처럼 읽혔다.
    ② `NONE_FIELDS` 는 값이 비어도 `none` 으로 남긴다 (위 주석). 값이 이미 마침표로 끝나면
    겹치지 않게 하나로 맞춘다.

    D121. `style="nl"` 이면 `nl_prompt` 로 넘긴다 — `--prompt_fields` 는 그때 안 쓰인다
    (필드가 문장 하나로 합쳐지므로 고를 게 없다). 요약표에 그렇게 찍는다.
    """
    if style == "nl":
        return nl_prompt(caption)
    parts = []
    for field in fields:
        value = str(caption.get(field) or "").strip()
        if not value:
            if field not in NONE_FIELDS:
                continue
            value = NONE_VALUE
        parts.append(f"{field}: {value.rstrip('.').strip()}.")
    return " ".join(parts)


def load_events(metadata_csv: str):
    """영상 → event 문장. `metadata.csv` 는 (video, camera) 마다 행이 있어 첫 행만 쓴다.

    `prompt` 열이 없는 코퍼스가 있다 (DynPose-LBM 의 `metadata.csv` 는 `video,dynamic` 뿐이고
    영상 캡션 소스가 데이터 디렉토리 어디에도 없다). 그때는 event 를 **빈 문자열**로 둔다 —
    `prompt_of` 가 빈 필드를 자동으로 떨어뜨리므로 프롬프트에서 `event:` 가 통째로 사라진다.
    여기서 KeyError 로 죽으면 안 되는 이유는, `--videos all` 이 이 dict 의 key 로 영상 목록을
    만들기 때문이다 — 목록은 채우되 문장만 비운다.
    """
    import csv
    events = {}
    with open(metadata_csv, encoding="utf-8") as file:
        reader = csv.DictReader(file)
        has_prompt = "prompt" in (reader.fieldnames or [])
        for row in reader:
            events.setdefault(row["video"],
                              first_sentence(row["prompt"]) if has_prompt else "")
    return events


def promote_targetless(cfg: dict):
    """`phrase_targetless` 를 가진 preset 을 **targetless 로 승격**한다 (D123, `--targetless_promote`).

    왜 config 플래그를 그냥 안 고치고 승격 단계를 두는가: 기본을 바꾸면 이미 학습에 들어간
    캡션이 조용히 달라진다. d121 두 arm 이 지금 target 절이 붙은 `track_truck_*` 캡션으로
    돌고 있어서, 그 코퍼스를 다시 export 하면 문자열이 어긋난다. 그래서 **기본은 꺼짐**이고
    새로 굽는 뱅크에서만 명시적으로 켠다. 켰는지 여부는 캡션 JSON 헤더에 남는다.

    무엇이 승격되는가 (d122 뱅크 실측, `subject_in_frame` median):
        track_truck_left  0.462 (n=912)   aim=free  ← 승격
        track_truck_right 0.385 (n=696)   aim=free  ← 승격
        track_dolly_in/out 1.000          aim=free     그대로 (자기 축이라 대상이 중앙에 남는다)
        나머지 track_*     1.000          aim=look_at  그대로
    `track_` 접두사가 문제가 아니라 truck 두 개가 문제라, 판정을 접두사가 아니라 config 의
    `phrase_targetless` 유무로 둔다 — 나중에 `truck_*`/`pedestal_*` 을 넣을 자리도 같다.

    문구를 바꾸는 이유: targetless 판정은 `caption_of` 가 `TARGET_SLOT not in phrase` 로 assert
    하므로, `{target}` 을 든 옛 문구로는 승격 자체가 통과 못 한다. 그리고 통과시켜도 안 된다 —
    "tracks alongside X" 는 프레임 밖에 X 를 두고 하는 말이 되어 지키지 못할 약속이다.
    """
    promoted = []
    for name, spec in cfg["presets"].items():
        phrase = spec.get("phrase_targetless")
        if not phrase:
            continue
        spec["phrase"] = phrase
        spec["targetless"] = True
        promoted.append(name)
    return sorted(promoted)


def restore_legacy_reaim(cfg: dict):
    """`--legacy_dolly_phrase`: D176 이전의 **부정형 재조준 문구**를 되살린다.

    D176 (사용자 지시 2026-09-10) 전에는 aim=free dolly 세 개가 재조준을 **부정형으로** 설명했다:
        dolly_in        "dollies in along its own axis without re-aiming"
        dolly_out       "dollies back along its own axis without re-aiming"
        track_dolly_in  "tracks {target} while pushing in along its own axis"
    지금은 그 절을 지우고, **대상을 방향으로 부르는지 여부**를 재조준 판별자로 삼는다:
        dolly_in       "dollies straight forward"           / dolly_in_look_at  "... toward {target}"
        dolly_out      "dollies straight back"              / dolly_out_look_at "... away from {target}"
        track_dolly_in "tracks {target} while pushing in"   / ..._look_at       "... toward {it}"
    즉 같은 동작에서 전치사구 하나만 다른 **최소쌍**이 되고, 부정형("~하지 않는다")이 사라진다.
    부정을 텍스트 조건으로 주는 것은 조건부 생성에서 약한 신호다 — 있는 것을 말하는 쪽이 낫다.

    `promote_targetless` 와 같은 이유로 **기본을 바꾸되 되돌릴 손잡이를 남긴다**: 이미 굽어서
    학습에 들어간 캡션(d121/d157 계열)을 다시 export 할 때 문자열이 어긋나면 안 된다.
    """
    restored = []
    for name, spec in cfg["presets"].items():
        phrase = spec.get("phrase_legacy_reaim")
        if not phrase:
            continue
        spec["phrase"] = phrase
        restored.append(name)
    return sorted(restored)


def main(args):
    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    with open(args.presets, encoding="utf-8") as file:
        cfg = json.load(file)
    legacy_reaim = restore_legacy_reaim(cfg) if args.legacy_dolly_phrase else []
    promoted = promote_targetless(cfg) if args.targetless_promote else []
    events = load_events(args.metadata_csv)
    label_map = load_label_map(args.label_map)
    desc_override = load_desc_override(args.desc_override)
    external_labels = load_external_labels(args.external_shapes)
    fields = [f.strip() for f in args.prompt_fields.split(",") if f.strip()]
    # 크기 부사는 3-state 다: 안 주면 형식과 무관하게 **끔**(D176, 사용자 지시 2026-09-10 —
    # 부사가 요청이 아니라 실현치를 읽어서 요청↔실현 대비가 안 남는다, §모듈 docstring).
    # `--magnitude` 를 명시하면 D121~D175 문장이 그대로 돌아온다.
    magnitude = False if args.magnitude is None else bool(args.magnitude)

    if args.videos_file:
        # 굽기가 도는 중에는 현재 라운드 목록을 뺀 편만 넘긴다 (샤드가 같은 `bank.json` 을
        # 쓰는 중이면 읽다가 깨진 JSON 을 만난다). `--videos` 보다 우선한다.
        with open(args.videos_file, encoding="utf-8") as file:
            videos = [line.strip() for line in file if line.strip()]
    elif args.videos == ["all"]:
        videos = sorted(v for v in events
                        if path.isfile(path.join(out_root, v, args.bank_dir, "bank.json")))
    else:
        videos = list(args.videos)

    rows, missing_event, fake_track, unmapped = [], [], [], {}
    no_desc, no_comp = [], []
    for video in videos:
        bank_path = path.join(out_root, video, args.bank_dir, "bank.json")
        if not path.isfile(bank_path):
            print(f"  [{video}] {args.bank_dir}/bank.json 없음 -> 건너뜀")
            continue
        with open(bank_path, encoding="utf-8") as file:
            bank = json.load(file)
        event = events.get(video, "")
        if not event:
            missing_event.append(video)
        # D121. 없으면 조용히 라벨 경로로 떨어지므로 **세서 요약표에 남긴다** — nl 형식에서
        # desc 가 없으면 target 이 다시 "camel" 한 단어가 되고, 그게 재설계의 이유였다.
        desc_map = load_anchor_desc(out_root, video, args.anchor_desc)
        if args.anchor_desc and not desc_map:
            no_desc.append(video)
        # 사람이 지정한 지칭구는 VLM 것보다 우선한다. `--desc_override` 를 안 주면 빈 dict 라
        # 아래 갱신은 no-op 이고, 준 씬/노드만 바뀐다 (나머지 씬은 비트 동일).
        if desc_override.get(video):
            desc_map = {**desc_map, **desc_override[video]}
            print(f"  [{video}] desc 덮어쓰기 {len(desc_override[video])}개: "
                  f"{desc_override[video]}")
        # 마찬가지로 composition 열은 D121 이후에 구운 뱅크에만 있다.
        if args.prompt_style == "nl" and not any(
                "in_frame_ids" in v or "enter_ids" in v for v in bank["variants"][:1]):
            no_comp.append(video)
        captions = {}
        n_framing_dropped = 0
        for variant in bank["variants"]:
            caption = caption_of(variant, event, cfg, magnitude, label_map,
                                 external_labels, args.framing_timeline,
                                 desc_map, args.prompt_style, args.framing_on_free,
                                 args.framing_min_in_frame,
                                 args.framing_exit_min_in_frame, args.nl_framing)
            caption["prompt"] = prompt_of(caption, fields, args.prompt_style)
            # 게이트가 걸린 것과 **실제로 문장이 바뀐 것**은 다르다 — 이미 targetless 인
            # preset(pan/tilt, 승격된 track_truck_*)은 원래 framing 절이 없어서 게이트가
            # 걸려도 캡션이 그대로다. camel/d128 에서 게이트 96 : 실제 변화 54 였다.
            # 헤더에 게이트 수를 적으면 효과를 1.8배로 부풀리게 되므로 한 번 더 굽고 센다.
            if framing_dropped(variant, args.framing_on_free, args.framing_min_in_frame,
                               cfg, args.framing_timeline,
                               args.framing_exit_min_in_frame):
                plain = caption_of(variant, event, cfg, magnitude, label_map,
                                   external_labels, args.framing_timeline,
                                   desc_map, args.prompt_style, True, 0.0, 0.0,
                                   args.nl_framing)
                plain["prompt"] = prompt_of(plain, fields, args.prompt_style)
                n_framing_dropped += int(plain != caption)
            # 표에 없어서 상위어(fallback)로 떨어진 라벨. 어휘가 늘어난 걸 조용히 넘기지 않는다.
            if label_map is not None and caption["target"] == label_map.get("fallback", "object"):
                unmapped[str(variant["anchor_label"])] = unmapped.get(
                    str(variant["anchor_label"]), 0) + 1
            captions[variant["variant_id"]] = caption
            # 캡션이 "tracks" 라고 말했는데 gain 이 0 이면 텍스트와 궤적이 어긋난다 (D74).
            if str(variant["preset"]).startswith("track") and track_gain_of(variant) == 0.0:
                fake_track.append(f"{video}/{variant['variant_id']}")
        payload = {"format": "lbm_bank_captions_v1", "video": video, "bank_dir": args.bank_dir,
                   "prompt_fields": fields, "presets_config": path.basename(args.presets),
                   "magnitude": bool(magnitude),
                   # D121. 세 개를 같이 남긴다 — 같은 뱅크에서 나온 두 캡션 파일을 나중에
                   # 대조하려면 "무슨 형식으로 뽑았나"가 파일 안에 있어야 한다.
                   "prompt_style": args.prompt_style,
                   # D123. 켜고 구운 캡션과 끄고 구운 캡션이 같은 뱅크에서 나온다 —
                   # 어느 쪽인지 파일 안에 없으면 나중에 문자열로 역추적해야 한다.
                   "targetless_promoted": promoted,
                   # D143. 두 손잡이와 **실제로 몇 개에서 절이 빠졌는지**를 같이 남긴다.
                   # 임계만 남기면 뱅크가 바뀌었을 때 같은 임계가 몇 배로 물린 건지 모른다.
                   "framing_on_free": bool(args.framing_on_free),
                   "framing_min_in_frame": float(args.framing_min_in_frame),
                   # D154. exit 절 전용 임계. 두 값이 같으면 D143 과 같은 동작이다.
                   "framing_exit_min_in_frame": float(args.framing_exit_min_in_frame),
                   "framing_dropped": n_framing_dropped,
                   # D176. 문장에 framing 절을 넣었는지. `framing_*` 세 키는 **자격 판정**이고
                   # 이건 **문장에 실었는가**라 층이 다르다 (§caption_of) — 헤더에 없으면
                   # 같은 뱅크에서 나온 두 캡션 파일을 문자열로 역추적해야 한다.
                   "nl_framing": bool(args.nl_framing),
                   # D176. 부정형 재조준 문구를 되살렸는지 (§restore_legacy_reaim). 빈 목록이
                   # 기본(= 새 최소쌍 문구)이고, 되살리면 어느 preset 이 바뀌었는지 이름이 남는다.
                   "legacy_reaim_phrase": legacy_reaim,
                   "anchor_desc": bool(desc_map),
                   "label_map": path.basename(args.label_map) if args.label_map else None,
                   "event": event, "captions": captions}
        target = path.join(out_root, video, args.bank_dir, args.out_name)
        if not args.dry_run:
            with open(target, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=1)
        sample = next(iter(captions.values()))
        rows.append((video, len(captions), n_framing_dropped, sample["prompt"]))

    header = f"{'video':<24}{'n':>6}{'프레이밍뺌':>12}  prompt (첫 변이)"
    print(header)
    print("-" * min(len(header) + 60, 140))
    for video, n, dropped, prompt in rows:
        print(f"{video:<24}{n:>6}{dropped:>8} ({100 * dropped / max(n, 1):4.1f}%)  {prompt[:84]}")
    print(f"\n영상 {len(rows)}   변이 {sum(r[1] for r in rows)}   "
          f"형식 {args.prompt_style}   "
          f"{'(필드 미사용)' if args.prompt_style == 'nl' else f'프롬프트 필드 {fields}'}   "
          f"크기 부사 {'켬' if magnitude else '끔'}   "
          f"framing 문장 {'켬' if args.nl_framing else '끔'}   "
          f"targetless 승격 {promoted if promoted else '끔'}   "
          f"재조준 부정문구 {legacy_reaim if legacy_reaim else '끔'}   "
          f"{'(dry run — 안 씀)' if args.dry_run else ''}")
    n_all = sum(r[1] for r in rows)
    n_drop = sum(r[2] for r in rows)
    print(f"프레이밍 절 뺀 변이 {n_drop}/{n_all} ({100 * n_drop / max(n_all, 1):.1f}%)   "
          f"임계 subject_in_frame < {args.framing_min_in_frame}   "
          f"aim=free 전량 차단 {'켬' if not args.framing_on_free else '끔'}")
    if missing_event:
        print(f"⚠ event 비어 있음 {len(missing_event)}편: {missing_event[:8]}")
    if no_desc:
        print(f"⚠ instance_desc.json 없어 라벨로 떨어진 {len(no_desc)}편: {no_desc[:8]} "
              f"(describe_instances_vlm.py 를 먼저 돌릴 것)")
    if no_comp:
        print(f"⚠ 뱅크에 composition 열(in_frame_ids/enter_ids/exit_ids)이 없는 {len(no_comp)}편: "
              f"{no_comp[:8]} — 문장에서 composition 절이 통째로 빠진다 (D121 이전 뱅크)")
    if unmapped:
        top = sorted(unmapped.items(), key=lambda kv: -kv[1])[:8]
        print(f"⚠ 라벨 표에 없어 '{label_map.get('fallback', 'object')}' 로 떨어진 앵커 "
              f"{len(unmapped)}종 / {sum(unmapped.values())}변이: {top}")
    if fake_track:
        print(f"⚠ track_* preset 인데 follow_gain 0 인 변이 {len(fake_track)}건 "
              f"(캡션은 'tracks' 인데 궤적은 추종이 아니다): {fake_track[:5]}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="*", default=["all"])       # all = 뱅크 있는 전량
    parser.add_argument("--videos_file", default="", type=str)        # 한 줄 1편. --videos 보다 우선
    parser.add_argument("--bank_dir", default="hole_bank_k6", type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--metadata_csv", default=METADATA_DEFAULT, type=str)
    parser.add_argument("--presets", default=PRESETS_DEFAULT, type=str)
    # config 의 `phrase_targetless` 를 가진 preset 을 targetless 로 승격 (D123). 지금 걸리는 건
    # `track_truck_left/right` 둘뿐이다 — 위치만 따라가고 aim=free 라 재조준을 안 해서
    # d122 실측 subject_in_frame median 이 0.462 / 0.385 인데 문구는 "tracks alongside X" 였다.
    # **기본 꺼짐**: 켜면 옛 뱅크를 다시 export 했을 때 캡션이 조용히 바뀐다 (d121 두 arm 이
    # 지금 그 옛 문자열로 학습 중이다). 새로 굽는 뱅크에서만 명시적으로 켤 것.
    parser.add_argument("--targetless_promote", action="store_true", default=False)
    parser.add_argument("--no_targetless_promote", dest="targetless_promote",
                        action="store_false")
    # D176 (사용자 지시 2026-09-10). aim=free dolly 세 개의 부정형 재조준 문구
    # ("along its own axis without re-aiming") 를 되살린다 — §restore_legacy_reaim.
    # **기본 꺼짐** = 새 최소쌍 문구. 옛 코퍼스를 다시 export 할 때만 켠다.
    parser.add_argument("--legacy_dolly_phrase", action="store_true", default=False)
    parser.add_argument("--no_legacy_dolly_phrase", dest="legacy_dolly_phrase",
                        action="store_false")
    # 앵커 라벨 → 자연어 명사 표. **안 주면 기존 동작 그대로** (Vista 는 라벨이 이미 명사다).
    # TRUMANS 는 라벨이 blend 오브젝트 이름이라 `configs/trumans_labels.json` 이 필요하다.
    parser.add_argument("--label_map", default="", type=str)
    # 사람이 지정한 anchor 지칭구 `{video: {node_id: text}}`. **안 주면 기존 동작 그대로**
    # (`instance_desc.json` 의 VLM 문구를 쓴다). §load_desc_override
    parser.add_argument("--desc_override", default="", type=str)
    # 뱅크를 만들 때 쓴 `datadop_shapes_v1` JSON. `dd_*` (free-moving) 행의 캡션을 라벨에서
    # 합성하는 데 쓴다 — 안 주면 `dd_*` 행에서 assert 로 멈춘다 (조용히 틀린 문장보다 낫다).
    parser.add_argument("--external_shapes", default="", type=str)
    # 학습 프롬프트에 넣을 필드. 사용자 지시가 target/motion 둘만이다.
    parser.add_argument("--prompt_fields", default="target,motion", type=str)
    # 같은 뱅크에 프롬프트 변이를 여러 개 두기 위한 파일명. 기본값은 예전과 동일한 captions.json 이라
    # 안 주면 동작이 그대로다. (target 절 유무 대조: captions.json vs captions_notarget.json)
    parser.add_argument("--out_name", default="captions.json", type=str)
    # D121. 프롬프트 형식. `nl` 은 한 문장(사용자 지시 형식, 기본값), `fields` 는 예전
    # `target: X. motion: Y.` 를 문자 단위로 그대로 다시 만든다.
    parser.add_argument("--prompt_style", default="nl", choices=["nl", "fields"], type=str)
    # D121. anchor referring expression (`instance_desc.json`). 끄면 예전처럼 라벨만 쓴다.
    parser.add_argument("--anchor_desc", action="store_true", default=True)
    parser.add_argument("--no_anchor_desc", dest="anchor_desc", action="store_false")
    # 크기 부사(barely..dramatically). **3-state 지만 안 주면 형식과 무관하게 끔** (D176,
    # 사용자 지시 2026-09-10 — §:850 이 집행자다). D175 까지는 nl 이 켰었고 이 주석이 그
    # 옛 동작을 그대로 적고 있었다. 다시 켜려면 `--magnitude` 를 명시할 것.
    parser.add_argument("--magnitude", action="store_true", default=None)
    parser.add_argument("--no_magnitude", dest="magnitude", action="store_false")
    # D119. shot scale 시간축. 뱅크에 `subject_area_seq` 가 있어야 효과가 있고(없으면 조용히
    # 예전 문구), 버킷이 실제로 바뀐 변이에서만 문장이 길어진다.
    # **D120 부터 기본 True** (사용자 확정 2026-09-03). camel 65변이 실측에서 median 이
    # end/start 비율 |r−1|>0.5 를 52.3% 숨겼고 문구가 바뀌는 변이가 43.1% 였다 — dolly_in
    # 사다리 4칸이 전부 "medium close-up" 한 문장으로 접히던 게 이걸로 갈린다.
    parser.add_argument("--framing_timeline", action="store_true", default=True)
    parser.add_argument("--no_framing_timeline", dest="framing_timeline",
                        action="store_false")
    # D166 (사용자 지시 2026-09-08). framing / shot scale 을 **구조체에만 남기고 문장에서 뺀다.**
    # `--no_nl_framing` 이면 `caption["framing"]` 은 그대로 채워지고 `prompt` 만
    # `The camera [motion].` 로 짧아진다. D143 게이트와 층이 다르다 (§caption_of docstring).
    # `composition` 절도 같이 빠진다 — 문장에서 그 자리가 framing 뒤뿐이다.
    # D176 (사용자 지시 2026-09-10): **기본 꺼짐**. shot size 어휘가 실측 면적비와 구간이
    # 겹치고(d157 49.6% 불일치) 가림(`subject_visible_frac`)을 캡션이 안 읽는다. `--nl_framing`
    # 으로 D166 이전 문장(framing + composition 절)이 그대로 돌아온다.
    parser.add_argument("--nl_framing", action="store_true", default=False)
    parser.add_argument("--no_nl_framing", dest="nl_framing", action="store_false")
    # D143 (사용자 지시 2026-09-06). 프레이밍 약속을 못 지키는 변이에서 그 절을 뺀다.
    # **기본은 변이 축**(`--framing_min_in_frame 0.85`): 그 변이의 실측 `subject_in_frame` 이
    # 임계 미만이면 framing/composition 을 뺀다. preset 축(`--no_framing_on_free`) 은 켜면
    # aim="free" 를 전량 끊는데, `dolly_out`/`track_dolly_out` 처럼 재조준 없이도 대상이
    # 중앙에 남는 preset 까지 같이 버린다 (§free_aim_framing 의 표). 그래서 기본은 꺼짐.
    # 옛 캡션을 글자 그대로 재현하려면 `--framing_min_in_frame 0` 이다.
    parser.add_argument("--framing_on_free", action="store_true", default=True)
    parser.add_argument("--no_framing_on_free", dest="framing_on_free",
                        action="store_false")
    parser.add_argument("--framing_min_in_frame", default=0.85, type=float)
    # D154 (사용자 지시 2026-09-06). `exit` 절 전용 임계. 그 절은 "대상이 프레임을 떠난다"고
    # **이미 말하므로** in_frame 이 낮은 게 거짓이 아니다 — 위 임계를 그대로 물리면 d128
    # train 기준 1,843건 중 1,026건(56%)이 정직했던 exit 캡션이다 (`framing_dropped` 표).
    # 기본 0.05 는 "시작부터 한 번도 중앙에 안 들어옴"만 끊는다: 그때만 절의 앞부분
    # ("medium shot 으로 담고 있다가")이 거짓이다. D143 동작을 그대로 쓰려면 이 값을
    # `--framing_min_in_frame` 과 같게 준다.
    parser.add_argument("--framing_exit_min_in_frame", default=0.05, type=float)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    main(parser.parse_args())
