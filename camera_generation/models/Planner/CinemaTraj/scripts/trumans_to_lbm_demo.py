"""TRUMANS recording 하나 → 원본 Look-Before-Move 가 먹을 수 있는 `demo_root` 를 만든다.

**왜 필요한가.** LBM 은 영상이 아니라 *제작 파이프라인 산출물*을 입력으로 받는다
(`Engine/run_full_pipeline.py:224` → `--demo-root`). `director_stage.resolve_demo_sources:279` 가
찾는 것은 `.blend` 1개 + 버전 붙은 JSON 3종이고, 그 JSON 은 리포에서 "intentionally excluded"
(README:5) 라 예시조차 없다. 우리 Vista4D 코퍼스로 LBM 을 못 돌린 이유가 이것이었다 — 영상뿐이라
`.blend` 도 layout 도 없었다. TRUMANS 는 **그 둘을 다 갖고 있다**: 씬은 `.blend` 로,
행동은 `Actions/<seq>.txt` 의 `(id, start, end, 설명)` 으로.

그래서 이 스크립트가 하는 일은 "없는 걸 지어내기"가 아니라 **이름만 바꿔 끼우기**다:

    TRUMANS                                  LBM
    ------------------------------------     ------------------------------------------
    Recordings_blend/<uuid>/<uuid>.blend  →  demo_root/<uuid>.blend        (symlink)
    obj_list.txt (상호작용 물체 이름)      →  asset_sheet[].asset_id
    blend 의 world AABB (worker 가 측정)   →  asset_sheet[].width/depth/height
                                             layout_description.assets[].location / rotation.z
    Actions/<seq>.txt 한 줄                →  storyboard_outline[0].shots[] 한 개
    blend scene AABB                       →  layout_description.scene_size

**확인된 사실 (추측 아님).**
- `obj_list.txt` 의 이름이 `.blend` 오브젝트 이름과 **정확히 일치**한다 → LBM 의
  `_find_object_by_asset_id` (`director_scene_context_builder.py:254`) 정확일치 분기가 그대로 먹는다.
- 씬 이름이 `Scene` 이라 LBM 의 `Scene_<id>` 규약과 다르지만, `:1107` 이
  `_resolve_scene(scene_id) or _scene_for_worker()` 로 `bpy.context.scene` 에 폴백한다
  → **1.6 GB blend 를 복사·개명하지 않아도 된다.** (symlink 로 충분)
- `selected_animation_v*.json` 은 `resolve_demo_sources:288` 이 **경로만 찾고 내용을 안 읽는다**.
  없으면 FileNotFoundError 로 죽으므로 스텁을 쓰되, 내용은 기록용이다.
- `rotation.z` 는 **도(degree)** 다 (`cinematographer_stage.py:483` 이 `math.radians` 로 변환).
- LBM 은 입력 `.blend` 를 절대 수정하지 않는다 (전부 `-b` background, save 호출 없음).

**시퀀스 판별.** 한 씬(uuid)에는 take 가 여러 개 붙어 있고(예시 씬은 10개), `.blend` 에는 그중
**하나만** 액션이 assign 되어 있다 (접미사 `.009` = 10번째). 어느 take 인지는 `.blend` 의 프레임
수와 시퀀스 프레임 수를 맞춰 찾는다 — `seg_name.npy`(프레임별 시퀀스명) + `scene_flag.npy`
(프레임별 `scene_list.npy` 인덱스) 로 그 씬의 시퀀스 목록을 뽑고 길이가 같은 것을 고른다.
후보가 여럿이면 `--sequence` 로 못 박을 것.

**Task #88 (단계별 격리 실행) 에서 나온 결함과 그 처방.** 전부 어댑터 쪽에서만 고친다 —
LBM 코드는 한 줄도 안 건드린다.
- D1/D4 `trumans_frame_start/end` 유실 → `--split_windows`. LBM 은 씬 프레임을 무조건 1..N 으로
  덮어쓰므로(`blender_render_worker.py:1141`) 창을 문장에 실어도 소용없다. shot 1개짜리 demo root
  를 창마다 만들고, Blender startup 훅(`trumans_frame_shift_startup.py`)으로 애니메이션을 당겨
  창 시작을 프레임 1 로 보낸다. 창별 blend 복제(1.66 GB × N)를 피하기 위한 구조다.
- D2/D3 focus 가 사람이 아니라 소품 → `--semantic_assets` + `--name_subject`. §asset_description.

사용 예시:
    python scripts/trumans_to_lbm_demo.py --recording 00add26c-7a26-4a61-b192-b97aa493b3f3
    python scripts/trumans_to_lbm_demo.py --recording <uuid> --sequence 2023-01-17@00-55-00 \
        --out /data1/.../lbm_demos --max_shots 6
    # 창별 root + 사람 focus (Task #88 처방 전량)
    python scripts/trumans_to_lbm_demo.py --recording <uuid> --out /data1/.../lbm_demos_w \
        --split_windows --semantic_assets --name_subject \
        --movement_terms --movement_vocab crane --movement_seed <uuid>
"""
import json
import re
import sys
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path, symlink, remove
from shutil import copyfile
from subprocess import run

import numpy as np

HERE = path.dirname(path.abspath(__file__))
WORKER = path.join(HERE, "trumans_blend_layout_worker.py")
FRAME_SHIFT_STARTUP = path.join(HERE, "trumans_frame_shift_startup.py")
TRUMANS_DEFAULT = "/data1/cympyc1785/data/trumans/Data_release"
BLENDER_DEFAULT = "/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender"
# 창별 runner 가 export 할 LBM 실행 env. `/tmp/lbm_run_fx.sh` 로 손으로 돌리던 값과 같다.
LBM_PYTHON_DEFAULT = "/data1/cympyc1785/miniconda3/envs/lbm/bin/python"
LBM_API_BASE_DEFAULT = "http://127.0.0.1:22002/v1"
LBM_VISION_MODEL_DEFAULT = "Qwen/Qwen3-VL-30B-A3B-Instruct"
LBM_FFMPEG_PATH_DEFAULT = "/data1/cympyc1785/tools/bin"

# `--movement_terms` 가 shot 설명 뒤에 돌려가며 붙이는 **카메라 움직임 어휘**.
#
# 왜 필요한가. LBM 에서 카메라 움직임을 정하는 건 LLM 이 아니라 shot 설명 문자열에 대한
# 키워드 정규식이다 (`director_stage.infer_movement_intent:972-984`). Director 의 LLM 경로
# (`director_engine_llm.py`) 에는 movement 라는 단어조차 없다. TRUMANS 의 액션 라벨
# ("Pick up the book with both hands") 은 그 키워드를 하나도 안 건드려 전부 fallback `static`
# 이 되고, 그러면 `cinematographer_stage.py:5141-5144` 가 "명시적 lock 문구가 없는 static" 을
# `subtle_motion` 으로 바꾸는데 그 함수는 close-up 이 아니면 **무조건 `push_in`** 을 준다
# (`:346-352`). 실측: 16 shot 전부 `push_in`, `motion_profile=semantic_light_dynamic`.
#
# 그래서 어휘를 **설명문에 심어** 그 정규식을 의도적으로 때린다. 각 항목은
# (목표 preset, 덧붙일 구절) 이고, 구절은 아래 판정기들을 동시에 통과하도록 골랐다:
#   `infer_movement_intent`  검사 순서 pan → pulls back/reveal → walks around/chases →
#                            approaches/emerge → stares/glares → (fallback) static
#   `infer_direction_label`  profile|side view → left, from behind|back view → back, 그 외 front
#   `infer_distance_label`   "close-up" 문자열 → close-up  (그러면 `is_closeup` 분기가 켜진다)
#   `infer_shot_goal`        reveal/rush/runs/chases/stares … 가 걸리면 카메라 수가 2~3 으로 는다
#                            → 여기서는 **일부러 피해서** shot 당 카메라 1대를 유지한다
#
# 2026-08-23 정정: `pan_left`/`truck_left` 의 원래 문구에는 방향어가 없어서 실제로는
# `pan_right`/`truck_right` 가 나오고 있었다 (`infer_direction_label` 은 "profile"/"side view"
# 에만 left 를 준다). 실측 histogram 에 pan_left/truck_left 가 0 이고 pan_right 8 / truck_right 8
# 로 찍힌 이유가 이것. 문구에 "in profile" 을 넣어 라벨과 실제를 맞췄다.
#
# 17 preset 중 **6종만 이 경로로 닿는다**. 나머지는 구조적으로 불가능하다:
#   orbit_left_arc / orbit_right_arc / pedestal_up / pedestal_down
#       — `canonical_movement:220-244` 은 "orbit"/"pedestal"/"crane" 을 알지만
#         `infer_movement_intent` 가 그 둘을 리턴하는 분기 자체가 없다. 설명문으로는 못 만든다.
#   pan_right / truck_right
#       — `preset_for_motion:787-807` 이 direction=="right" 를 요구하는데
#         `infer_direction_label` 은 left/back/front 만 리턴한다. "right" 가 없다.
#   rise_reveal / drop_reveal / s_curve / straight_ease / static_hold_locked
#       — 어떤 movement_tag 에서도 매핑되지 않는다 (Cinematographer 후보 탐색이 직접 고를 때만 등장).
MOVEMENT_TERMS = (
    ("push_in_arc",       ", and the camera approaches him"),
    ("pull_out_arc",      ", and the camera pulls back"),
    ("pan_left",          ", and the camera pans across the room, seen in profile"),
    ("truck_left",        ", and the camera follows in profile as he walks around the table"),
    ("static_hold",       ", held on a locked, still frame"),
    ("static_subtle_zoom", ", framed in close-up"),
)

# `--movement_vocab extended` 용 확장판. 위 6종 + 도달 불가였던 7종.
#
# 저 위 주석이 "구조적으로 불가능"이라고 적어둔 것들을 실제로 뚫었다 —
# `Director/director_stage.py` 의 `infer_movement_intent` 에 orbit/crane/pedestal/dolly 분기를,
# `infer_direction_label` 에 right/down 분기를, `Cinematographer/cinematographer_stage.py` 의
# `canonical_movement`/`preset_for_motion` 에 crane 분리와 pedestal_down 을 넣었다. 전부
# **env `STORYBLENDER_MOVEMENT_VOCAB=extended` 로 분기**해서 기본 동작은 그대로다.
#
# 문구는 판정기 통과 순서를 지키도록 골랐다. 확장 분기는 `infer_movement_intent` 맨 앞에
# 있으므로 "circles around" 는 기존 "walks around"(truck) 보다 먼저 걸린다.
# 방향어("to the right" / "downward")는 `infer_direction_label` 이 따로 읽는다.
#
# 여전히 못 만드는 2종: `s_curve`, `static_hold_locked` — 어떤 movement_tag 에서도 매핑되지
# 않는다. Cinematographer 후보 탐색이 스스로 고를 때만 나온다.
MOVEMENT_TERMS_EXTENDED = MOVEMENT_TERMS + (
    ("pan_right",         ", and the camera pans to the right across the room"),
    ("truck_right",       ", and the camera walks around the table to the right"),
    ("orbit_left_arc",    ", and the camera circles around him, seen in profile"),
    ("orbit_right_arc",   ", and the camera circles around him to the right"),
    ("pedestal_up",       ", and the camera rises above the scene"),
    ("pedestal_down",     ", and the camera lowers toward the floor downward"),
    ("straight_ease",     ", and the camera glides past him"),
)
# crane 계열(rise_reveal / drop_reveal). `infer_shot_goal` 의 "reveal" 을 피하려고 문구를
# "cranes up/down" 으로 썼다 — "reveals" 라고 쓰면 goal=reveal 이 되어 shot 당 카메라가
# 2~3 대로 늘고 렌더 예산이 그만큼 뛴다 (실측 goal=story_beat -> 1대 유지).
# 둘 다 HIGH_RISK_TRAJECTORY_PRESETS 라 story 근거 없이 쓰면 안전보고서에 경고가 남는다.
# 그래서 기본 extended 목록에서는 빼고 `--movement_vocab crane` 으로만 켠다.
MOVEMENT_TERMS_CRANE = (
    ("rise_reveal",       ", and the camera cranes up over him"),
    ("drop_reveal",       ", and the camera cranes down toward the floor downward"),
)


def movement_term_table(args):
    """`--movement_vocab` 에 맞는 (preset, 구절) 목록. base 가 기본이라 예전 실행과 같다.

    extended/crane 은 실행 시 env `STORYBLENDER_MOVEMENT_VOCAB=extended` 가 같이 켜져 있어야
    Director/Cinematographer 의 확장 분기가 살아난다 — 안 켜면 새 문구는 그냥 fallback
    `static` 으로 떨어져 전부 push_in 이 된다. 그래서 여기서 env 를 직접 세팅한다
    (demo 를 만드는 프로세스와 파이프라인 프로세스가 달라도 되도록 `run_lbm_lite` 쪽이 아니라
    실행 스크립트에서 export 하는 게 정석이지만, 잊었을 때 조용히 base 로 돌아가는 게 더 나쁘다).
    """
    if args.movement_vocab == "base":
        return list(MOVEMENT_TERMS)
    if args.movement_vocab == "crane":
        return list(MOVEMENT_TERMS_EXTENDED + MOVEMENT_TERMS_CRANE)
    return list(MOVEMENT_TERMS_EXTENDED)


# ─────────────────────────── (b) 의미 있는 asset description + 사람 주어 ───────────────────────────
#
# **왜 필요한가 (Task #88 의 D2/D3).** LBM 의 shot focus 는 LLM 이 아니라 문자열 점수로 정해진다
# (`director_stage.infer_focus_ids:846-952`). 점수는 shot 설명에 등장하는 **asset alias** 로 매겨지고,
# alias 는 `asset_alias_tokens:752-778` 이 만든다:
#   - 캐릭터   : asset_id 토큰 ∪ description 토큰(len>3, STOPWORDS 제외)
#   - 소품     : {asset_id 마지막 토큰} ∪ description 토큰(len>3, WEAK_OBJECT_ALIAS_TOKENS 제외)
# 기존 어댑터는 description 을 `"<id> in the scene"` 로 채웠다. 그러면
#   book_right_01 -> {"book", "right"}      (마지막 토큰 "01" 은 절대 안 맞는다)
#   zzy3          -> {"zzy3"}               (설명문에 "zzy3" 이 나올 리 없다)
# 이 되어 **사람은 alias 가 하나도 안 맞고 소품만 맞는다**. 실측: 15 shot 의 primary_focus_id 가
# book_right_01 ×8 / book_left_01 ×5 / oven_base_01 ×2, zzy3 ×0. 카메라가 사람이 아니라 책을
# 겨눴고, 거기에 LLM 이 `semantic_target="feet"` 를 붙여 책의 "발"을 찾는 높이로 내려갔다(D3).
#
# 고치는 곳은 두 군데다. **LBM 은 한 줄도 안 건드린다.**
#   ① description 을 asset_id 에서 만든 짧은 명사구로 (`"the book on the right"`), 캐릭터에는
#      `person / adult / human / character` 를 alias 로 심는다.
#   ② TRUMANS 액션 라벨은 명령문("Use both hands to pick up the vase")이라 주어가 없다. alias 를
#      심어도 문장에 사람이 안 나오면 소용없으므로 **주어를 붙인다** — "The person uses both hands
#      to pick up the vase". 그러면 "person" 이 index 4 로 **가장 먼저** 등장해
#      `earliest_alias_index` + 캐릭터 보너스(+2.5, :893-900)가 사람 쪽으로 몰린다.
#
# description 은 **짧게** 유지한다. len>3 토큰이 전부 alias 가 되므로 "on the table" 같은 걸 넣으면
# "table" 이 alias 가 되어 다른 shot 에 오검출된다. ("room"/"scene" 은 WEAK 목록이라 안전하다.)
# 반대로 "cup"/"pen" 처럼 3글자 명사는 **구조적으로 alias 가 될 수 없다** (len>3 필터). 그 소품은
# focus 후보에서 빠지는데, 우리 목적(사람을 찍는다)에는 오히려 맞다.
POSITION_WORDS = ("left", "right", "front", "back", "top", "bottom", "upper", "lower")

# asset_id 어간 → 사람이 읽는 명사. 없으면 어간을 그대로 쓴다(밑줄만 공백으로).
NOUN_OVERRIDES = {
    "oven_base": "oven",
    "oven_door": "oven door",
    "static_chair": "chair",
    "whiteboard": "whiteboard",
}


def asset_noun(asset_id: str):
    """`static_chair_03` → ("chair", ""),  `book_right_01` → ("book", "right")."""
    stem = re.sub(r"_\d+$", "", asset_id)
    position = ""
    tokens = stem.split("_")
    if len(tokens) > 1 and tokens[-1] in POSITION_WORDS:
        position = tokens[-1]
        stem_wo_pos = "_".join(tokens[:-1])
    else:
        stem_wo_pos = stem
    noun = NOUN_OVERRIDES.get(stem, NOUN_OVERRIDES.get(stem_wo_pos, stem_wo_pos.replace("_", " ")))
    return noun, position


def asset_description(asset_id: str, is_character: bool):
    if is_character:
        # len>3 토큰만 alias 가 되므로 person/adult/human/character 네 개가 심긴다.
        return "the person, an adult human character"
    noun, position = asset_noun(asset_id)
    return f"the {noun} on the {position}" if position else f"the {noun}"


IRREGULAR_THIRD_PERSON = {"be": "is", "have": "has", "do": "does", "go": "goes"}


def third_person(verb: str):
    """명령문 첫 동사를 3인칭 단수로. TRUMANS 라벨의 동사(Use/Pick/Put/Stand/Squat/Write/Sit/
    Open/Close/Move)는 전부 규칙 변화라 이 세 줄이면 충분하다."""
    low = verb.lower()
    if low in IRREGULAR_THIRD_PERSON:
        return IRREGULAR_THIRD_PERSON[low]
    if low.endswith(("s", "x", "z", "ch", "sh")):
        return low + "es"
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return low[:-1] + "ies"
    return low + "s"


def name_subject(text: str, subject: str = "The person"):
    """`"Pick up the book"` → `"The person picks up the book"`."""
    words = text.strip().split()
    if not words:
        return text
    return " ".join([subject, third_person(words[0])] + words[1:])


def read_obj_list(rec_dir: str):
    """`obj_list.txt` 는 파이썬 리스트 리터럴 한 줄이다 (`['cup_01', 'oven_base_01', ...]`)."""
    with open(path.join(rec_dir, "obj_list.txt"), encoding="utf-8") as file:
        text = file.read().strip()
    return [token.strip().strip("'\"") for token in text.strip("[]").split(",") if token.strip()]


def read_actions(trumans: str, sequence: str):
    """`Actions/<seq>.txt` → [{'id','start','end','text'}].

    실제 형식은 **탭 3열 `start\\tend\\t설명`** 이다 (id 열 없음). 예:
        `51\\t69\\tPick up the book with both hands`
    id 는 줄 번호로 매긴다 — LBM `shot_id` 가 int 여야 하고(`director_stage.py:463`) 씬 안에서만
    유일하면 되기 때문. 앞에 id 가 붙은 4열 변형도 있을 수 있어 열 개수로 분기한다.
    """
    actions = []
    with open(path.join(trumans, "Actions", f"{sequence}.txt"), encoding="utf-8") as file:
        for line in file:
            parts = [p for p in line.rstrip("\n").split("\t") if p != ""]
            if len(parts) == 3:
                start, end, text = parts
            elif len(parts) >= 4:
                _, start, end, text = parts[0], parts[1], parts[2], "\t".join(parts[3:])
            else:
                continue
            actions.append({"id": len(actions) + 1, "start": int(start),
                            "end": int(end), "text": text.strip()})
    return actions


# ────────────────────────────── (A) 고정 길이 슬라이딩 chunk ──────────────────────────────
#
# **왜 필요한가.** LBM-Lite 는 클립 길이가 **49프레임 고정**이다 (`lbm/presets.py:51
# NUM_FRAMES = 49`, 하류 emit 도 21-pose 로 고정). 그런데 VLM action tagging 이 낸 창 37개의
# 길이는 min 17 / med 43 / max 233 이라 **21개가 49를 못 채운다** — 창 경계 안에서만 자르면
# 클립 24개만 남고 창 21개(그중 action 14개, 오븐 여닫는 w21/w23 포함)가 통째로 빠진다.
#
# 그래서 창 경계를 무시하고 영상 전체를 `--chunk_frames` 창으로 `--chunk_stride` 씩 훑는다.
# 창은 shot 경계가 아니라 **"이 49프레임 동안 무슨 일이 일어나는가"의 근거**로만 쓴다:
# chunk 와 겹치는 창들의 서술을 시간순으로 이어 붙여 하나의 `shot_description` 으로 만든다.
#
# 겹침 하한(`--chunk_min_overlap`)을 두는 이유: stride 24 면 chunk 경계에서 창 하나가 1~2프레임만
# 걸치는 경우가 생기는데, 그걸 서술에 넣으면 화면에 거의 안 나오는 행동이 카메라 focus 를 끌어간다.
#
# 기본은 `--chunk_mode window` = 예전 동작(창 하나 = shot 하나) 그대로다.
SUBJECT_HEAD = re.compile(r"^\s*(?:the|a|an)\s+(?:man|woman|person|character|human|adult)\s+",
                          re.IGNORECASE)


def strip_subject(text: str):
    """`"The man opens the oven door."` → `"opens the oven door"`.

    이어 붙이는 두 번째 이후 절에서 주어를 뗀다. 못 떼면(주어로 시작하지 않는 문장) 원문 그대로
    돌려주고 호출자가 `", then "` 대신 `". "` 로 잇는다 — 문법이 깨지는 것보다 문장이 두 개인
    게 낫다.
    """
    stripped = SUBJECT_HEAD.sub("", text.strip())
    return (stripped.rstrip().rstrip("."), True) if stripped != text.strip() \
        else (text.strip().rstrip("."), False)


def rewrite_subject(text: str, phrase: str):
    """`"The man opens the oven door."` → `"The person opens the oven door."`

    **왜 필요한가.** LBM 의 shot focus 는 asset alias 문자열 점수로 정해지는데, 캐릭터 alias 는
    description 토큰 중 **len > 3** 만 남는다 (`director_stage.asset_alias_tokens:761-765`).
    인물 접지를 켠 VLM 서술은 "The man ..." 으로 시작하는데 `man` 은 세 글자라 **alias 가 될 수
    없다** — 사람이 alias 를 하나도 못 맞히고 소품이 focus 를 가져간다 (Task #88 의 D2/D3 가
    그대로 재발). 그래서 LBM 에 넣는 문자열에서만 주어를 `person` 으로 바꾼다. 서술의 나머지
    대명사(he/his)는 점수에 안 쓰이므로 그대로 둔다.
    """
    return SUBJECT_HEAD.sub(f"{phrase} ", text.strip(), count=1)


def chunk_narrative(entries: list, primary: int = None, subject: str = "The man"):
    """겹치는 창들의 서술 → chunk 서술 1개. `entries` 는 **시간순** 정렬된 창 목록.

    겹침이 가장 큰 창(`primary`)만 `shot_description`(위치·자세까지 붙은 완성문)을 쓰고,
    나머지는 `action`(행위만)을 쓴다. 둘 다 `shot_description` 을 쓰면 같은 행동이 두 번
    묘사된다 — 실측 c02 는 "picks up a book" 이 두 절에 걸쳐 반복돼 책을 두 번 집는 것처럼
    읽혔다. 서술 길이 중앙값도 216 → 164 자로 준다.

    두 번째 이후 절은 주어를 뗀다. 못 떼면 `". "` 로 문장을 나눈다 — 문법이 깨지는 것보다
    문장이 두 개인 게 낫다. 창이 하나뿐이면 그 창의 `shot_description` 과 **바이트 단위로 같다**.
    """
    if not entries:
        return f"{subject} moves through the room."
    if primary is None:
        primary = entries[0]["window"]
    #    `action` 은 항상 있지 않다 — 태거가 VLM 스키마 복구에 끝내 실패하면 `source:"fallback"`
    #    entry 를 쓰는데 거기엔 `shot_description` 만 들어간다 (실측 88/1891 = 4.7%, 53편 중
    #    34편). 그대로 인덱싱하면 `KeyError: 'action'` 으로 그 recording 이 통째로 죽는다.
    #    없으면 `shot_description` 으로 떨어진다 — 행위만이 아니라 위치까지 붙어 조금 장황하지만
    #    서술이 빠지는 것보다 낫고, 뒤의 `strip_subject` 가 주어 중복은 정리한다.
    #    `action` 이 있는 경우의 결과는 예전과 바이트 단위로 같다.
    parts = [(entry["shot_description"] if entry["window"] == primary
              else entry.get("action") or entry["shot_description"]).strip()
             for entry in entries]
    text = parts[0]
    if len(parts) == 1:
        return text
    text = text.rstrip().rstrip(".")
    for part in parts[1:]:
        clause, ok = strip_subject(part)
        text = f"{text}, then {clause}" if ok else f"{text}. {clause}"
    return text + "."


def build_chunks(total: int, size: int, stride: int, entries: list, min_overlap: int,
                 subject: str = "The man"):
    """영상 `total` 프레임을 `size` 프레임 chunk 로 `stride` 씩 훑는다.

    마지막 chunk 는 `end == total - 1` 이 되도록 **당겨서** 만든다(자르지 않는다) — 길이가
    `size` 보다 짧아지면 LBM-Lite 가 못 먹는다. 그래서 직전 chunk 와 겹침이 stride 보다 커질 수
    있고, 완전히 같은 구간이면 안 만든다.
    """
    assert total >= size, f"영상이 {total}프레임뿐이라 {size}프레임 chunk 를 못 만든다"
    starts = list(range(0, total - size + 1, stride))
    if starts[-1] != total - size:
        starts.append(total - size)
    chunks = []
    for index, start in enumerate(starts):
        end = start + size - 1
        overlaps = []
        for entry in entries:
            span = min(entry["end"], end) - max(entry["start"], start) + 1
            if span > 0:
                overlaps.append((span, entry))
        #    겹침이 하한 미만인 창은 뺀다. 전부 빠지면(짧은 창 하나만 스치는 경우) 겹침 최대
        #    창 하나는 남긴다 — 서술이 없는 chunk 를 만들지 않기 위해서다.
        kept = [(span, entry) for span, entry in overlaps if span >= min_overlap]
        if not kept and overlaps:
            kept = [max(overlaps, key=lambda item: item[0])]
        primary = max(kept, key=lambda item: item[0])[1] if kept else None
        kept.sort(key=lambda item: item[1]["start"])
        picked = [entry for _, entry in kept]
        chunks.append({
            "chunk": index + 1, "start": start, "end": end,
            "text": chunk_narrative(picked, primary["window"] if primary else None, subject),
            "windows": [entry["window"] for entry in picked],
            "window_kinds": [entry["kind"] for entry in picked],
            "primary_window": primary["window"] if primary else None,
            "dropped_windows": [entry["window"] for span, entry in overlaps
                                if span < min_overlap and entry not in picked],
        })
    return chunks


def sequences_for_scene(trumans: str, uuid: str):
    """그 씬(uuid, `_1`/`_2` 변형 포함)에 속한 시퀀스별 프레임 수. augment 사본은 뺀다."""
    scenes = np.load(path.join(trumans, "scene_list.npy"))
    seg = np.load(path.join(trumans, "seg_name.npy"))
    flag = np.load(path.join(trumans, "scene_flag.npy"))
    rows = [i for i, name in enumerate(scenes) if str(name).startswith(uuid)]
    names, counts = np.unique(seg[np.isin(flag, rows)], return_counts=True)
    # `_augment1/2` 는 모션 증강본이라 Actions/video 가 없다 — take 후보가 아니다.
    return {str(n): int(c) for n, c in zip(names, counts) if "_augment" not in str(n)}


def pick_sequence(trumans: str, uuid: str, n_frames: int, forced: str):
    """blend 프레임 수와 같은 시퀀스를 고른다. 동점이면 사용자에게 넘긴다."""
    table = sequences_for_scene(trumans, uuid)
    if forced:
        assert forced in table, f"{forced} 는 씬 {uuid} 의 시퀀스가 아니다. 후보: {sorted(table)}"
        return forced, table
    hits = [name for name, count in table.items() if count == n_frames]
    assert hits, (f"blend 프레임 수 {n_frames} 와 일치하는 시퀀스가 없다.\n"
                  f"  후보: {sorted(table.items(), key=lambda kv: kv[1])}\n"
                  f"  --sequence 로 직접 지정할 것.")
    assert len(hits) == 1, f"프레임 수 {n_frames} 인 시퀀스가 여럿이다: {hits}. --sequence 로 지정할 것."
    return hits[0], table


def run_layout_worker(blender: str, blend: str, assets: list, frame: int, out: str):
    """headless Blender 로 world AABB 를 잰다. 1.6 GB blend 도 실측 3초."""
    cmd = [blender, blend, "--background", "--python", WORKER, "--",
           "--frame", str(frame), "--assets", ",".join(assets), "--out", out]
    proc = run(cmd, capture_output=True, text=True)
    assert path.isfile(out), (f"layout worker 실패 (rc={proc.returncode})\n"
                              f"--- stdout ---\n{proc.stdout[-3000:]}\n"
                              f"--- stderr ---\n{proc.stderr[-3000:]}")
    with open(out, encoding="utf-8") as file:
        return json.load(file)


def blend_frame_count(blender: str, blend: str, tmp: str):
    """프레임 수만 먼저 알아야 시퀀스를 고를 수 있다. worker 를 asset 없이 한 번 돌린다."""
    report = run_layout_worker(blender, blend, [], 0, tmp)
    return report["frame_end"] - report["frame_start"] + 1, report


def build_asset_sheet(layout: dict, character_ids: set, semantic: bool = False):
    sheet = []
    for rec in layout["assets"]:
        width, depth, height = rec["size"]
        is_character = rec["asset_id"] in character_ids
        sheet.append({
            "asset_id": rec["asset_id"],
            "asset_type": "character" if is_character else "prop",
            "description": (asset_description(rec["asset_id"], is_character) if semantic
                            else f"{rec['asset_id']} in the scene"),
            "width": round(width, 4), "depth": round(depth, 4), "height": round(height, 4),
            # LBM 은 이 4개를 파일로 열어보고 없으면 빈 문자열로 흘린다
            # (`director_stage.py:378` `if candidate.exists()`) — 하드 크래시 아님.
            "front_view_url": "", "top_view_url": "", "left_view_url": "", "thumbnail_url": "",
            "main_file_path": layout["blend"],
        })
    return sheet


def write_demo_root(args, demo_root: str, blend: str, probe: dict, sequence: str,
                    obj_names: list, shots: list, layout_frame: int, n_actions: int):
    """demo root 하나(4종 계약 파일 + blend symlink)를 쓴다. 창별로 여러 번 불린다."""
    makedirs(path.join(demo_root, "animated_models"), exist_ok=True)
    makedirs(path.join(demo_root, "layout_script"), exist_ok=True)

    layout = run_layout_worker(args.blender, blend, obj_names, layout_frame,
                               path.join(demo_root, "layout_measured.json"))
    character_ids = {rec["asset_id"] for rec in layout["assets"] if rec["type"] == "ARMATURE"}
    asset_sheet = build_asset_sheet(layout, character_ids, semantic=args.semantic_assets)

    if args.semantic_assets:
        # 예전 summary 는 raw asset_id 를 그대로 나열해서 `asset_id.lower() in scene_text` 가
        # 소품마다 +1.0 을 얹어줬다 (`infer_focus_ids:866`) — 사람에게는 안 붙는 편향이다.
        # 사람이 읽는 라벨로 바꾸면 그 가산점이 사라지고, "person" 이 scene_text 에 들어간다.
        labels = [asset_description(rec["asset_id"], False) for rec in layout["assets"]
                  if rec["asset_id"] not in character_ids]
        summary = (f"A person performs {n_actions} everyday actions in an indoor scene "
                   f"containing {', '.join(labels)}.")
    else:
        summary = (f"A single character performs {n_actions} everyday actions in an indoor scene "
                   f"containing {', '.join(obj_names)}.")

    story = {
        "story_summary": summary,
        "asset_sheet": asset_sheet,
        "storyboard_outline": [{
            "scene_id": 1,
            "scene_description": summary,
            "shots": shots,
        }],
    }

    bounds = layout["scene_bounds"]
    layout_assets = []
    for rec in layout["assets"]:
        cx, cy, cz = rec["center"]
        layout_assets.append({
            "asset_id": rec["asset_id"],
            "location": {"x": cx, "y": cy, "z": cz},
            "rotation": {"x": 0.0, "y": 0.0, "z": rec["yaw_deg"]},   # z 는 도(degree)
            "dimensions": {"x": rec["size"][0], "y": rec["size"][1], "z": rec["size"][2]},
        })
    layout_script = {
        "asset_sheet": asset_sheet,
        "scene_details": [{
            "scene_id": 1,
            "scene_setup": {
                "scene_type": "indoor",
                "asset_ids": [rec["asset_id"] for rec in layout["assets"]],
                "layout_description": {
                    "description": summary,
                    # 기본값이 ±10 이라(`cinematographer_stage.py:430-433`) 실내 씬에 그대로 두면
                    # 카메라 후보가 벽 밖으로 나간다. 실측 AABB 를 넣는다.
                    "scene_size": {"x_negative": bounds["x_negative"], "x": bounds["x"],
                                   "y_negative": bounds["y_negative"], "y": bounds["y"]},
                    "assets": layout_assets,
                    "wall_z": bounds["z"],
                },
            },
        }],
    }

    # 내용은 안 읽히지만 경로가 없으면 resolve_demo_sources 가 죽는다 (:288).
    selected_animation = {
        "note": "TRUMANS: 애니메이션은 .blend 에 이미 baked 되어 있다. LBM 은 이 파일의 내용을 읽지 않는다.",
        "sequence": sequence, "fps": probe["fps"],
        "frame_start": probe["frame_start"], "frame_end": probe["frame_end"],
        "animations": [{"asset_id": rec["asset_id"], "action_name": "baked_in_blend",
                        "frame_start": probe["frame_start"], "frame_end": probe["frame_end"]}
                       for rec in layout["assets"]],
    }

    files = {
        path.join(demo_root, "animated_models", "animated_models_v1.json"): story,
        path.join(demo_root, "animated_models", "selected_animation_v1.json"): selected_animation,
        path.join(demo_root, "layout_script", "layout_script_v1.json"): layout_script,
    }

    window = None
    if args.split_windows:
        start, end = shots[0]["trumans_frame_start"], shots[0]["trumans_frame_end"]
        window = {
            "format": "trumans_window_v1",
            "recording": args.recording, "sequence": sequence,
            "trumans_frame_start": start, "trumans_frame_end": end,
            "n_frames": end - start + 1,
            # 원본 프레임 `start` 를 씬 프레임 1 로 보낸다. 이 값을 env 로 주면
            # `trumans_frame_shift_startup.py` 가 로드 직후 애니메이션을 당긴다.
            "frame_offset": 1 - start,
            "layout_frame": layout_frame,
            "shot_id": shots[0]["shot_id"],
            "shot_description": shots[0]["shot_description"],
            "movement_term_target": shots[0]["movement_term_target"],
            #    slide 모드에서 이 chunk 서술이 어느 창들에서 조립됐는지. LBM 은 안 읽는다 —
            #    나중에 "이 카메라가 뭘 보고 만들어졌나"를 되짚기 위한 기록.
            **{key: value for key, value in shots[0].items() if key.startswith("chunk_")},
            "env": {"TRUMANS_FRAME_OFFSET": str(1 - start),
                    "TRUMANS_FRAME_COUNT": str(end - start + 1),
                    "BLENDER_USER_SCRIPTS": path.join(args.out, "_frame_shift")},
        }
        files[path.join(demo_root, "_window.json")] = window

    for target, payload in files.items():
        with open(target, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=1)

    link = path.join(demo_root, path.basename(blend))
    if path.islink(link) or path.isfile(link):
        remove(link)
    symlink(blend, link)   # 1.6 GB 를 복사하지 않는다 — LBM 은 blend 를 수정하지 않는다

    return {"root": demo_root, "layout": layout, "asset_sheet": asset_sheet,
            "character_ids": character_ids, "window": window}


def emit_frame_shift_runner(args, blend: str, probe: dict, written: list):
    """창별 실행에 필요한 것 두 가지를 `--out` 에 깐다.

    ① `_frame_shift/startup/trumans_frame_shift_startup.py` — Blender 기동 스크립트. LBM 이 띄우는
       모든 blender 프로세스가 `BLENDER_USER_SCRIPTS` 로 이걸 물고, `TRUMANS_FRAME_OFFSET` 만큼
       애니메이션을 당긴다. **창별로 1.66 GB blend 를 복제하지 않기 위한 장치**이고 LBM 코드는
       한 줄도 안 건드린다 (그 파일 상단 docstring 에 근거).
    ② `<uuid>__run_windows.sh` — 창 demo root 를 순서대로 도는 실행 스크립트.
    """
    shift_dir = path.join(args.out, "_frame_shift", "startup")
    makedirs(shift_dir, exist_ok=True)
    copyfile(FRAME_SHIFT_STARTUP, path.join(shift_dir, path.basename(FRAME_SHIFT_STARTUP)))

    lines = [
        "#!/bin/bash",
        "# TRUMANS 창별 demo root 를 순서대로 LBM 4단계에 태운다 (어댑터가 생성).",
        "#   $1 = run_id prefix (예: trumans_00add26c_w)   $2 = CUDA device",
        "set -o pipefail",
        f"LBM={path.abspath(path.join(path.dirname(HERE), '..', 'Look-Before-Move'))}",
        f"PY={args.lbm_python}",
        f"export STORYBLENDER_BLENDER_EXE={args.blender}",
        f"export BLENDER_USER_SCRIPTS={path.join(args.out, '_frame_shift')}",
        'export CUDA_VISIBLE_DEVICES=${2:-0}',
        # LBM 의 LLM 백엔드. any_llm 이 `ANYLLM_*` 를 못 찾으면 OpenAI 로 나가려다 죽는다.
        f"export ANYLLM_API_BASE={args.api_base}",
        "export ANYLLM_API_KEY=dummy",
        "export ANYLLM_PROVIDER=openai",
        f"export STORYBLENDER_VISION_MODEL={args.vision_model}",
        # Editor 가 ffmpeg 을 PATH 에서 찾는다 (conda env 에는 없다).
        f"export PATH={args.ffmpeg_path}:$PATH",
        # TRUMANS 는 씬이 1개뿐이라 LBM 의 다중 씬 분기를 우회해야 한다.
        "export LBM_SINGLE_SCENE_FALLBACK=1",
        #    LBM 쪽은 `== "extended"` **정확 일치**로만 확장 분기를 켠다
        #    (`Director/director_stage.py:980`, `Cinematographer/cinematographer_stage.py:60`).
        #    `crane` 을 그대로 내보내면 분기가 조용히 꺼져서 crane/orbit/pedestal 문구가 전부
        #    fallback `static` 으로 떨어진다 — 로그에는 아무 경고도 안 남는다. base 가 아니면
        #    무조건 `extended` 를 내보낸다 (crane 어휘는 demo 생성 쪽 목록 확장일 뿐이다).
        f"export STORYBLENDER_MOVEMENT_VOCAB="
        f"{'base' if args.movement_vocab == 'base' else 'extended'}",
        f"export STORYBLENDER_TRAJECTORY_SCALE={args.trajectory_scale:g}",
        "",
    ]
    for rec in written:
        window = rec["window"]
        lines += [
            f"# w{window['shot_id']:02d}  원본 프레임 {window['trumans_frame_start']}.."
            f"{window['trumans_frame_end']}  |  {window['shot_description']}",
            f"export TRUMANS_FRAME_OFFSET={window['frame_offset']}",
            f"export TRUMANS_FRAME_COUNT={window['n_frames']}",
            f'$PY "$LBM/Engine/run_full_pipeline.py" --demo-root {rec["root"]} \\',
            f'    --run-id "${{1:-trumans}}{path.basename(rec["root"]).split("__")[-1]}" '
            f"--fps {probe['fps']:g} \\",
            # 기본값 `fast` 는 seed 탐색을 건너뛰어 벽·천장만 찍힌 프리뷰 1장을 낸다 (실측).
            f"    --camera-quality {args.camera_quality}",
            "",
        ]
    script = path.join(args.out, f"{args.recording}__run_windows.sh")
    with open(script, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return script


def main(args):
    rec_dir = path.join(args.trumans, "Recordings_blend", args.recording)
    assert path.isdir(rec_dir), f"recording 디렉토리가 없다: {rec_dir}"
    blends = sorted(glob(path.join(rec_dir, "*.blend")))
    assert len(blends) == 1, f"{rec_dir} 의 .blend 가 1개가 아니다: {blends}"
    blend = blends[0]

    #    `--out` 을 절대경로로 못 박는다. LBM 은 `--demo-root` 를 `Path(...).resolve()` 로 푸는데
    #    (`Engine/run_full_pipeline.py:226`) 그 프로세스의 cwd 는 `Look-Before-Move/` 라, 상대경로를
    #    그대로 실으면 없는 디렉토리를 가리킨다. `BLENDER_USER_SCRIPTS` 도 마찬가지로 조용히
    #    빗나가서 frame shift 훅이 안 걸린다 — 에러 없이 **창 전체가 프레임 1부터 렌더**된다.
    args.out = path.abspath(args.out)
    makedirs(args.out, exist_ok=True)
    demo_root = path.join(args.out, args.recording)
    # 창별 모드에서는 `<uuid>` root 를 만들지 않는다 — `_probe.json` 만 `--out` 바로 밑에 둔다.
    probe_path = (path.join(args.out, f"{args.recording}__probe.json") if args.split_windows
                  else path.join(demo_root, "_probe.json"))
    if not args.split_windows:
        makedirs(path.join(demo_root, "animated_models"), exist_ok=True)
        makedirs(path.join(demo_root, "layout_script"), exist_ok=True)

    n_frames, probe = blend_frame_count(args.blender, blend, probe_path)
    sequence, table = pick_sequence(args.trumans, args.recording, n_frames, args.sequence)
    actions = read_actions(args.trumans, sequence)
    assert actions, f"Actions/{sequence}.txt 가 비었다"

    obj_names = read_obj_list(rec_dir)

    # `action_index` → VLM 서술. 비면 아래 루프가 예전대로 규칙 라벨을 쓴다.
    narrative, tagged_entries = {}, []
    if args.narrative_json:
        with open(args.narrative_json, encoding="utf-8") as file:
            tagged = json.load(file)
        assert tagged.get("format") == "trumans_vlm_actions_v1", \
            f"narrative_json 형식이 다르다: {tagged.get('format')!r}"
        #    같은 take 의 서술이어야 한다. 다른 시퀀스를 물리면 창 번호는 맞는데 내용이 딴 씬이라
        #    조용히 통과한다 — 그게 제일 안 들키는 사고다.
        assert tagged["sequence"] == sequence, \
            f"narrative_json 은 시퀀스 {tagged['sequence']!r} 것인데 이 demo 는 {sequence!r} 다"
        tagged_entries = sorted(tagged["entries"], key=lambda e: e["start"])
        #    action 창만 `action_index` 를 갖는다 (gap 창은 None). window 모드는 그 둘만 쓴다.
        narrative = {int(e["action_index"]): e for e in tagged_entries
                     if e.get("action_index") is not None}
        assert tagged["video_frames"] == n_frames, \
            (f"narrative_json 의 video_frames {tagged['video_frames']} 와 blend 프레임 수 "
             f"{n_frames} 가 다르다 — 창 프레임 번호가 어긋난다")

    shots = []
    picked = actions[:args.max_shots] if args.max_shots else actions
    terms = movement_term_table(args)
    # 어휘를 shot 순서대로 돌리면 **모든 recording 이 똑같은 preset 수열**을 갖는다 (실측:
    # 6편이 shot 1~8 에서 8/8 일치, 15쌍 전부). recording uuid 로 시드를 걸어 섞어야 편끼리
    # 달라진다. 시드가 uuid 라 같은 recording 을 다시 돌리면 결과는 재현된다.
    if args.movement_seed:
        import random
        order = list(range(len(terms)))
        random.Random(f"{args.movement_seed}:{sequence}").shuffle(order)
        terms = [terms[i] for i in order]
    #    (A) 고정 길이 chunk 모드. 창을 shot 경계로 쓰지 않고 영상 전체를 49프레임씩 훑는다.
    #    `picked` 를 chunk 목록으로 갈아끼우면 아래 루프·`--split_windows` 는 그대로 돌아간다.
    chunks = []
    if args.chunk_mode == "slide":
        assert tagged_entries, "--chunk_mode slide 는 --narrative_json 이 있어야 한다"
        chunks = build_chunks(n_frames, args.chunk_frames, args.chunk_stride,
                              tagged_entries, args.chunk_min_overlap, args.subject_phrase)
        if args.max_shots:
            chunks = chunks[:args.max_shots]
        picked = [{"id": c["chunk"], "start": c["start"], "end": c["end"], "text": c["text"]}
                  for c in chunks]

    for index, action in enumerate(picked):
        text = action["text"]
        tagged_entry = narrative.get(index) if args.chunk_mode == "window" else None
        if args.chunk_mode == "slide":
            #    chunk 서술은 이미 조립된 완성문이다. 주어만 필요하면 갈아끼운다(아래 참조).
            pass
        elif tagged_entry:
            #    VLM 서술은 이미 "A person ..." 으로 시작하므로 `--name_subject` 를 안 태운다
            #    (태우면 주어가 두 번 붙는다). movement term 은 그대로 뒤에 붙인다.
            text = tagged_entry["shot_description"]
        elif args.name_subject:
            text = name_subject(text, args.subject_phrase)
        if args.narrative_subject and (tagged_entry or args.chunk_mode == "slide"):
            text = rewrite_subject(text, args.subject_phrase)
        if args.movement_terms:
            #    movement term 구절은 `", and the camera ..."` 로 시작한다. 규칙 라벨은 문장부호
            #    없이 끝나지만 VLM 서술은 마침표로 끝나서 그냥 붙이면 `"...action., and the"` 가
            #    된다. Director 가 읽는 건 이 문자열 하나뿐이라 끝 마침표만 떼고 붙인다.
            text = f"{text.rstrip().rstrip('.')}{terms[index % len(terms)][1]}"
        shots.append({
            "shot_id": int(action["id"]),
            "shot_description": text,
            # 어느 팔인지 + 대조용 원본 라벨. LBM 은 안 읽는다.
            "description_source": ("vlm" if (tagged_entry or args.chunk_mode == "slide")
                                   else "rule"),
            "rule_text": action["text"],
            # LBM 이 입력에서 읽는 shot 키는 shot_id / shot_description 둘뿐이다
            # (`director_stage.py:463-466`). 아래는 우리가 나중에 대조하려고 남기는 기록.
            "trumans_frame_start": action["start"], "trumans_frame_end": action["end"],
            "movement_term_target": (terms[index % len(terms)][0]
                                     if args.movement_terms else ""),
        })

    for shot, chunk in zip(shots, chunks):     # slide 모드에서만 `chunks` 가 비어 있지 않다
        shot["chunk_windows"] = chunk["windows"]
        shot["chunk_window_kinds"] = chunk["window_kinds"]
        shot["chunk_primary_window"] = chunk["primary_window"]
        shot["chunk_dropped_windows"] = chunk["dropped_windows"]

    # 창별 demo root 로 쪼개면 shot 1개짜리 root 가 N 개, 아니면 예전처럼 전량이 든 root 1개.
    if args.split_windows:
        groups = [[shot] for shot in shots]
    else:
        groups = [shots]

    written = []
    for index, group in enumerate(groups):
        if args.split_windows:
            start, end = group[0]["trumans_frame_start"], group[0]["trumans_frame_end"]
            root = path.join(args.out,
                             f"{args.recording}__w{index + 1:02d}_f{start:04d}_{end:04d}")
            # 소품이 take 내내 움직이므로 레이아웃은 **그 창의 시작 프레임**에서 재야 한다.
            # (전량 root 는 예전대로 --layout_frame.)
            layout_frame = start
        else:
            root, layout_frame = demo_root, args.layout_frame
        written.append(write_demo_root(args, root, blend, probe, sequence, obj_names,
                                       group, layout_frame, len(actions)))

    if args.split_windows:
        emit_frame_shift_runner(args, blend, probe, written)

    first = written[0]
    bounds = first["layout"]["scene_bounds"]
    print(f"\n{'항목':22s} 값")
    print(f"{'recording':22s} {args.recording}")
    print(f"{'blend':22s} {blend}  ({path.getsize(blend) / 1e9:.2f} GB, symlink)")
    print(f"{'scene':22s} {probe['scene_name']!r}  {probe['engine']}  "
          f"{probe['resolution']}  fps {probe['fps']:g}")
    print(f"{'frames':22s} {probe['frame_start']}..{probe['frame_end']}  (= {n_frames})")
    print(f"{'sequence':22s} {sequence}   (씬 내 take {len(table)}개 중 프레임수 일치)")
    print(f"{'actions -> shots':22s} {len(actions)} -> {len(shots)}")
    if args.chunk_mode == "slide":
        counts = [len(c["windows"]) for c in chunks]
        dropped = sum(len(c["dropped_windows"]) for c in chunks)
        covered = sorted({w for c in chunks for w in c["windows"]})
        print(f"{'chunk_mode':22s} slide  {args.chunk_frames}프레임 / stride "
              f"{args.chunk_stride} / min_overlap {args.chunk_min_overlap}")
        print(f"{'chunk 당 창 수':22s} min {min(counts)} med "
              f"{int(np.median(counts))} max {max(counts)}   "
              f"겹침 하한 미달로 뺀 창 {dropped}회")
        print(f"{'창 커버리지':22s} {len(covered)}/{len(tagged_entries)}"
              f"   빠진 창 {sorted(set(e['window'] for e in tagged_entries) - set(covered))}")
    print(f"{'assets':22s} {len(first['asset_sheet'])}  "
          f"(character {sorted(first['character_ids'])})")
    print(f"{'scene_size':22s} x[{bounds['x_negative']:.2f},{bounds['x']:.2f}] "
          f"y[{bounds['y_negative']:.2f},{bounds['y']:.2f}] z<={bounds['z']:.2f}")
    print(f"{'semantic_assets':22s} {args.semantic_assets}   "
          f"{'name_subject':16s} {args.name_subject}")
    n_vlm = sum(1 for shot in shots if shot["description_source"] == "vlm")
    print(f"{'narrative':22s} {args.narrative_json or '(rule labels)'}   "
          f"vlm {n_vlm}/{len(shots)} shot")
    if first["layout"]["missing"]:
        print(f"{'못 찾은 asset':22s} {first['layout']['missing']}")

    if args.split_windows:
        print(f"\n{'demo root':52s} {'창':>14s} {'offset':>7s}  shot")
        for rec in written:
            window = rec["window"]
            print(f"{path.basename(rec['root']):52s} "
                  f"{window['trumans_frame_start']:5d}..{window['trumans_frame_end']:<7d} "
                  f"{window['frame_offset']:+7d}  {window['shot_description'][:60]}")
        print(f"\n-> {args.out}  ({len(written)} demo roots)")
        print("\n다음:")
        print(f"  bash {path.join(args.out, args.recording + '__run_windows.sh')} <run_id_prefix> <gpu>")
    else:
        print(f"\n-> {demo_root}")
        print("\n다음:")
        print(f"  export STORYBLENDER_BLENDER_EXE={args.blender}")
        print(f"  python Engine/run_full_pipeline.py --demo-root {demo_root} "
              f"--run-id trumans_{args.recording[:8]} --fps {probe['fps']:g}")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recording", required=True, type=str)   # Recordings_blend 의 디렉토리명
    parser.add_argument("--sequence", default="", type=str)       # 비우면 프레임 수로 자동 판별
    parser.add_argument("--trumans", default=TRUMANS_DEFAULT, type=str)
    parser.add_argument("--blender", default=BLENDER_DEFAULT, type=str)
    parser.add_argument("--out", default=path.join(path.dirname(HERE), "out", "lbm_demos"), type=str)
    parser.add_argument("--layout_frame", default=0, type=int)    # 레이아웃을 잴 프레임
    parser.add_argument("--max_shots", default=0, type=int)       # 0 = 전량
    # shot 설명 뒤에 카메라 움직임 어휘를 돌려가며 붙인다 (§MOVEMENT_TERMS). 기본은 꺼둔 상태라
    # 앞선 실행과 바이트 단위로 같은 demo_root 가 나온다.
    # 어휘 목록. base = 예전 6종, extended = 13종, crane = extended + rise/drop (shot 당 카메라 2~3).
    parser.add_argument("--movement_vocab", default="base", choices=("base", "extended", "crane"))
    # 비우면 shot 순서대로 = 모든 recording 이 같은 preset 수열. 채우면 recording 별로 섞는다.
    parser.add_argument("--movement_seed", default="", type=str)
    parser.add_argument("--movement_terms", dest="movement_terms", action="store_true")
    parser.add_argument("--no_movement_terms", dest="movement_terms", action="store_false")
    parser.set_defaults(movement_terms=False)

    # ── Task #88 D1/D2/D4 대응. 셋 다 기본 off 라 안 켜면 예전 demo_root 와 바이트 단위로 같다. ──
    # (a) 창별 demo root. shot 1개짜리 root 를 N 개 만들고 `_frame_shift` startup 훅 + 실행
    #     스크립트를 같이 깐다. LBM 은 항상 씬 프레임 1..N 을 렌더하므로(D1), 창을 살리려면
    #     애니메이션을 당기는 수밖에 없다.
    parser.add_argument("--split_windows", dest="split_windows", action="store_true")
    parser.add_argument("--no_split_windows", dest="split_windows", action="store_false")
    parser.set_defaults(split_windows=False)
    # (b1) asset description 을 asset_id 에서 만든 명사구로 + 캐릭터에 person/human alias.
    parser.add_argument("--semantic_assets", dest="semantic_assets", action="store_true")
    parser.add_argument("--no_semantic_assets", dest="semantic_assets", action="store_false")
    parser.set_defaults(semantic_assets=False)
    # (b2) 명령문 액션 라벨에 사람 주어를 붙인다. alias 만 심고 문장에 사람이 안 나오면 무의미하다.
    parser.add_argument("--name_subject", dest="name_subject", action="store_true")
    parser.add_argument("--no_name_subject", dest="name_subject", action="store_false")
    parser.set_defaults(name_subject=False)
    parser.add_argument("--subject_phrase", default="The person", type=str)
    # (b3) `shot_description` 을 **VLM 서술**로 갈아끼운다 (`trumans_vlm_action_tag.py` 산출
    #      `trumans_vlm_actions_v1` JSON). 비우면 예전대로 `Actions/<seq>.txt` 라벨 그대로.
    #      규칙 라벨은 동작 라벨이지 shot 서술이 아니라 Director 가 읽을 공간 정보가 비어 있다.
    #      두 팔을 같은 창·같은 카메라 예산으로 비교하려고 여기 한 줄만 갈아끼운다.
    parser.add_argument("--narrative_json", default="", type=str)
    # (b4) VLM 서술의 **주어만** `--subject_phrase` 로 갈아끼운다 (§rewrite_subject). 기본 off.
    parser.add_argument("--narrative_subject", dest="narrative_subject", action="store_true")
    parser.add_argument("--no_narrative_subject", dest="narrative_subject", action="store_false")
    parser.set_defaults(narrative_subject=False)
    # (A) 고정 길이 슬라이딩 chunk (§build_chunks). `window` = 예전 동작(창 하나 = shot 하나).
    parser.add_argument("--chunk_mode", default="window", choices=("window", "slide"))
    parser.add_argument("--chunk_frames", default=49, type=int)      # LBM-Lite `NUM_FRAMES`
    parser.add_argument("--chunk_stride", default=24, type=int)      # 24 = 약 50% 겹침
    parser.add_argument("--chunk_min_overlap", default=8, type=int)  # 이보다 적게 걸친 창은 뺀다
    # (c) `--split_windows` 가 뱉는 `<uuid>__run_windows.sh` 에 실릴 LBM 실행 env.
    parser.add_argument("--lbm_python", default=LBM_PYTHON_DEFAULT, type=str)
    parser.add_argument("--api_base", default=LBM_API_BASE_DEFAULT, type=str)
    parser.add_argument("--vision_model", default=LBM_VISION_MODEL_DEFAULT, type=str)
    parser.add_argument("--ffmpeg_path", default=LBM_FFMPEG_PATH_DEFAULT, type=str)
    parser.add_argument("--trajectory_scale", default=5.0, type=float)
    # LBM 기본값 `fast` 는 seed 탐색을 건너뛰어 벽만 찍는다 — 여기 기본은 `quality`.
    parser.add_argument("--camera_quality", default="quality", choices=("fast", "quality"))
    sys.exit(main(parser.parse_args()))
