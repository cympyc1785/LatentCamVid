"""TRUMANS action 창을 **VLM 으로 다시 서술**해 LBM 의 narrative 입력으로 쓸 문장을 만든다.

왜 필요한가: 지금 LBM 에 들어가는 `shot_description` 은 `Actions/<seq>.txt` 의 라벨을 그대로 쓴
것이다 (`"Pick up the book with both hands"`). 이건 **동작 라벨**이지 shot 서술이 아니라, Director
가 읽을 공간 정보(어디서 · 무엇 옆에서 · 어떤 자세로)가 통째로 비어 있다. 게다가 라벨은 Actions
파일에 있는 것만 있다 — 이 take 는 보행이 프레임의 19% 인데 Actions 라벨엔 0건이다
(`trumans-walking-is-unlabeled`). VLM 팔은 그 구멍을 렌더를 **보고** 메우는 대조군이다.

입력은 TRUMANS 자신의 `video_render/<seq>.pkl.mp4` 다. 실측: 2077 프레임 @ 30 fps, 648x484 로
blend 프레임 0..2076 과 **1:1** — 창 `(start, end)` 를 인덱스로 그대로 자르면 된다. Lite/LBM 이
렌더한 영상을 쓰지 않는 이유는 그쪽이 이미 **카메라를 고른 결과**라, 그걸 보고 쓴 서술을 다시
카메라 선택에 먹이면 순환이 되기 때문.

창마다 프레임 `--num_tiles` 장을 균등 추출해 가로 contact sheet 한 장으로 붙이고 (타일 위에
`t0`.. 라벨), 그 한 장만 VLM 에 준다. 프레임을 따로 N 장 보내는 것보다 시간 순서가 눈에 보인다.

**규칙 라벨을 프롬프트에 넣지 않는다** — 넣으면 VLM 이 그걸 바꿔 쓰기만 해서 두 팔이 같아진다.
대신 산출 JSON 에 `rule_text` 를 나란히 실어 사후 대조만 한다.

출력 `<out>/<sequence>_actions_vlm.json` (`trumans_vlm_actions_v1`) — `trumans_to_lbm_demo.py
--narrative_json` 이 그대로 읽는다. `trace/` 에 턴 전량.

env: `vista4d` (cv2). 서버: `bash scripts/serve_qwen3vl.sh`

`--include_gaps` 를 켜면 Actions 구간 **밖**(미태그 프레임)도 창으로 만들어 같이 태깅한다.
실측 2023-01-17@00-55-00 은 2077 프레임 중 라벨이 붙은 게 746 (35.9%) 뿐이고, 나머지 1331 은
walk 47% / turn 11% / pose 32% / still 10% 로 **static+walking 두 라벨로 안 덮인다**.
산출물 이름이 `_full` 로 갈리므로 기존 narrative arm 을 덮지 않는다.

예시:
    python scripts/trumans_vlm_action_tag.py --sequence 2023-01-17@00-55-00
    python scripts/trumans_vlm_action_tag.py --sequence 2023-01-17@00-55-00 --windows 1 2 3 --verbose
    python scripts/trumans_vlm_action_tag.py --sequence 2023-01-17@00-55-00 --include_gaps --verbose
"""
import ast
import json
import pickle
import re
import sys
import time
from argparse import ArgumentParser
from os import listdir, makedirs, path

import cv2
import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.vlm import VLMClient                                                   # noqa: E402
from scripts.trumans_to_lbm_demo import read_actions                            # noqa: E402

TRUMANS_DEFAULT = "/data1/cympyc1785/data/trumans/Data_release"

SYSTEM = (
    "You are a script supervisor writing action descriptions for a previsualization pipeline. "
    "You watch a strip of frames sampled in time order from one continuous take and describe "
    "what the person does, where in the room it happens, and which objects are involved. "
    "Be concrete and visual. Never mention the strip, the tiles, the labels, frames, or the "
    "fact that you are looking at images.\n"
    #    카메라 언급 금지가 이 프롬프트의 핵심 제약이다. 이 문장은 **카메라를 고르는 쪽의
    #    입력**으로 들어간다 — "turns to face the camera" 같은 게 섞이면 Director 가 아직
    #    존재하지도 않는 카메라를 기준으로 배치를 짠다. 실측으로 3/3 창에서 샜다.
    "Never mention a camera, a shot, a take, framing, or any filming vocabulary. There is no "
    "camera in this scene yet — it is your description that decides where one will go. "
    "Describe direction with room and object references only (\"toward the window\", \"away "
    "from the sofa\"), never with \"toward the camera\" or \"screen left\"."
)

PROMPT = """The strip above shows {n} frames sampled in time order from a single continuous \
action, labelled t0 (earliest) to t{last} (latest). They are one shot, not separate scenes.

Describe this shot. Return raw JSON only, no markdown fences:

{{
  "action": "<one sentence, present tense, what the person physically does>",
  "location": "<where in the room, relative to named furniture/objects>",
  "objects": ["<object the person interacts with>", ...],
  "body_facing": "<which way the person faces / whether they turn during the shot>",
  "shot_description": "<ONE sentence, 15-35 words, combining the above into a single \
narrative line a director can stage a camera from. Start with 'The person' or 'A person'.>"
}}

Rules:
- If the person is only walking or standing with no object interaction, say so plainly.
- "objects" must be [] when nothing is handled.
- Do not invent an object you cannot see.
- No camera or filming words anywhere in the JSON: no "camera", "shot", "take", "frame",
  "screen left/right", "close-up", "wide". Use room and object references for direction."""

# `--include_gaps` 로 켜는 **미태그 구간**용 추가 지시. Actions 밖 구간은 하나의 동작이 아니라
# 이동·전환·미태그 조작이 섞여 있어서 위 프롬프트의 "single continuous action" 전제가 깨진다.
# 실측(2023-01-17@00-55-00): 미태그 1331 프레임의 body_pose 변화율 중앙값이 2.231 rad/s 로
# 태그된 "Write with the left hand"(0.701) 보다 크다 — 조용한 구간이 아니라 라벨만 없는 구간이다.
GAP_SUFFIX = """
This stretch was not labelled as a named action. It may be a transition: the person may walk \
between places, turn, settle into or out of a posture, or handle something without it being a \
named task. Describe what actually happens across the whole stretch, including the movement \
between locations. If the person mainly walks, say where from and where to."""

# 재질의에만 붙는 지시. 위반 목록("forbidden filming vocabulary: ['camera']")만으로는 **그럼
# 뭐라고 쓰냐**가 안 나와서, 실측 3건이 4턴 내내 같은 문장을 반복하고 fallback 으로 떨어졌다.
# 1차 힌트는 `body_facing` 만 다뤘는데, 그러자 위반이 `location`("the left side of the frame")
# 과 `action`("turns to face the camera") 으로 옮겨갔다 — 필드 하나를 막으면 옆 필드로 샌다.
# 그래서 네 필드 전부에 **금지 표현 → 대체 표현** 쌍을 준다. 'frame' 도 금지어라는 걸 명시하는
# 게 특히 중요하다(모델은 이걸 촬영 용어로 인식하지 않는다).
REPAIR_HINT = (
    "The words \"camera\", \"frame\", \"shot\", \"take\", \"screen left/right\", \"close-up\", "
    "\"wide angle\", \"footage\", \"viewer\" are forbidden in EVERY field, not just the one that "
    "was flagged. Rewrite every occurrence using room and object references:\n"
    "- \"faces the camera\" -> \"faces the desk\" / \"faces the window\"\n"
    "- \"back to the camera\" -> \"turns away from the counter\"\n"
    "- \"the left side of the frame\" -> \"the side of the room near the wooden door\"\n"
    "- \"moves toward the camera\" -> \"walks toward the sofa\"\n"
    "Name a piece of furniture or a part of the room instead of a viewing position. Check "
    "\"action\", \"location\", \"body_facing\" and \"shot_description\" one by one before "
    "answering."
)


# ---------------------------------------------------------------------------------------------
# 씬 접지 (`--scene_objects` / `--object_motion` / `--rule_hint`)
#
# 왜: 첫 실행에서 VLM 이 **씬에 없는 물체를 지어냈다**. rule 이 `Open the oven` 인 4창(w21/w23/
# w34/w36)을 각각 cabinet / monitor / computer / cabinet 으로 읽었다. TRUMANS 는 이걸 검증할
# 근거를 자기 안에 갖고 있다 — `Recordings_blend/<rec>/obj_list.txt` 가 그 녹화의 **상호작용
# 가능한 물체 전량**이고, `Object_all/Object_pose/<seq>.npy` 가 그것들의 프레임별 pose 다.
# 실측(2023-01-17@00-55-00): 네 창 모두 `oven_door_01` 이 0.218~0.323 m / 33~50° 움직인다.
# **rule 이 맞고 VLM 이 틀렸다.**
# ---------------------------------------------------------------------------------------------

#    부품 토큰. TRUMANS 물체 이름은 `<noun>_<part>_<NN>` 꼴이고 사람이 부르는 이름은 명사뿐이다.
#    `base`/`seat`/`left`/`right` 는 버리고 `door`/`drawer`/`screen` 은 남긴다 — 오븐 "문"이
#    열리는 것과 오븐 본체가 움직이는 건 서술이 달라야 한다.
PART_DROP = ("base", "seat", "left", "right")
PART_KEEP = ("door", "drawer", "screen")
PREFIX_DROP = ("static", "movable")


def readable_object(name: str) -> str:
    """`oven_door_01` → `oven door`, `static_chair_03` → `chair`, `book_right_01` → `book`."""
    tokens = [t for t in name.split("_") if t and not t.isdigit()]
    while tokens and tokens[0] in PREFIX_DROP:
        tokens = tokens[1:]
    kept = [tokens[0]] if tokens else []
    for token in tokens[1:]:
        if token in PART_KEEP and token != kept[-1]:
            kept.append(token)
        elif token not in PART_DROP and token not in PART_KEEP and token != kept[-1]:
            kept.append(token)
    return " ".join(kept) if kept else name


def object_vocabulary(trumans: str):
    """`Object_all/Object_mesh` 전량의 **명사 집합**. 씬에 없는 명사를 판정하는 데 쓴다."""
    meshes = path.join(trumans, "Object_all", "Object_mesh")
    if not path.isdir(meshes):
        return set()
    return {readable_object(f[:-4]).split()[0] for f in listdir(meshes) if f.endswith(".obj")}


def absent_objects(trumans: str, scene_readable):
    """TRUMANS 어휘에는 있는데 **이 방엔 없는** 명사들. 토큰 단위로 뺀다.

    첫 토큰만 빼면 `oven door` 의 `door` 가 남아, VLM 이 정확히 "oven door" 라고 써도 반려된다.
    """
    present = {token for name in scene_readable for token in name.split()}
    return tuple(sorted(object_vocabulary(trumans) - present))


def resolve_recording(trumans: str, sequence: str) -> str:
    """시퀀스 → `Recordings_blend` 디렉토리명. `scene_flag` → `scene_list` 를 거친다.

    `scene_list` 항목은 `<uuid>` / `<uuid>_1` / `<uuid>_2` 로 갈리는데 (같은 방의 배치 변형)
    `Recordings_blend` 디렉토리는 **base uuid 하나**다. 그래서 뒤의 `_N` 을 떼고 존재 확인한다.
    """
    seg = np.load(path.join(trumans, "seg_name.npy"))
    hit = np.flatnonzero(seg == sequence)
    assert hit.size, f"{sequence} 가 seg_name.npy 에 없다"
    flags = np.load(path.join(trumans, "scene_flag.npy"), mmap_mode="r")[hit[0]:hit[-1] + 1]
    unique = np.unique(np.asarray(flags))
    assert unique.size == 1, f"{sequence}: scene_flag 가 하나가 아니다 {unique}"
    scene = str(np.load(path.join(trumans, "scene_list.npy"))[int(unique[0])])
    base = scene.rsplit("_", 1)[0] if scene.rsplit("_", 1)[-1].isdigit() else scene
    for candidate in (scene, base):
        if path.isdir(path.join(trumans, "Recordings_blend", candidate)):
            return candidate
    raise AssertionError(f"{sequence}: Recordings_blend 에 {scene}/{base} 둘 다 없다")


def read_scene_objects(trumans: str, recording: str):
    """`obj_list.txt` → (원본 이름 리스트, 사람이 부르는 이름 리스트(중복 제거, 순서 보존))."""
    listing = path.join(trumans, "Recordings_blend", recording, "obj_list.txt")
    assert path.exists(listing), f"obj_list.txt 가 없다: {listing}"
    raw = ast.literal_eval(open(listing, encoding="utf-8").read().strip())
    readable, seen = [], set()
    for name in raw:
        nice = readable_object(name)
        if nice not in seen:
            seen.add(nice)
            readable.append(nice)
    return list(raw), readable


def euler_xyz_to_matrix(euler):
    """(F,3) Blender euler XYZ → (F,3,3). 축 순서가 틀려도 **변화량 크기**는 살아남는다."""
    euler = np.asarray(euler, dtype=np.float64)
    cx, cy, cz = np.cos(euler).T
    sx, sy, sz = np.sin(euler).T
    one, zero = np.ones_like(cx), np.zeros_like(cx)
    rx = np.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).reshape(-1, 3, 3)
    ry = np.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(-1, 3, 3)
    rz = np.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    return rz @ ry @ rx


def read_object_poses(trumans: str, sequence: str, raw_names):
    """`{name: (location (F,3), R (F,3,3))}`. `Object_pose` 는 name→{'location','rotation'} dict."""
    blob = np.load(path.join(trumans, "Object_all", "Object_pose", f"{sequence}.npy"),
                   allow_pickle=True).item()
    poses = {}
    for name in raw_names:
        entry = blob.get(name)
        if entry is None:
            continue
        poses[name] = (np.asarray(entry["location"], dtype=np.float64),
                       euler_xyz_to_matrix(entry["rotation"]))
    return poses


def moved_objects(poses, start, end, move_m=0.05, rot_deg=12.0):
    """[(사람이 부르는 이름, dL m, dR deg)] — 창 안에서 실제로 움직인 물체만.

    회전은 **euler ptp 가 아니라 회전행렬 geodesic** 이다. euler ptp 는 wrap 때문에 dL 0.005 m
    짜리 정지 물체에서도 33° 를 뱉어 임계로 못 쓴다. geodesic 으로 재면 실측 노이즈 바닥이
    ≤ 6.9° / ≤ 0.015 m 이고 진짜 상호작용은 ≥ 12.7° / ≥ 0.084 m 라 두 임계가 깨끗하게 가른다.
    """
    out = []
    for name, (loc, rot) in poses.items():
        last = min(int(end), loc.shape[0] - 1)
        first = min(int(start), last)
        if last <= first:
            continue
        span = loc[first:last + 1]
        delta_m = float(np.linalg.norm(np.ptp(span, axis=0)))
        rel = rot[first].T @ rot[first:last + 1]
        cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        delta_deg = float(np.degrees(np.arccos(cos).max()))
        if delta_m > move_m or delta_deg > rot_deg:
            out.append((readable_object(name), delta_m, delta_deg))
    out.sort(key=lambda item: -item[1])
    return out


# ---------------------------------------------------------------------------------------------
# 인물 접지 (`--person_probe` / `--person`)
#
# 왜: 창마다 대명사가 제멋대로다. 실측(접지 3종, 37창) — their/they 20창 · **his/he 2창**(w15,
# w16) · she/her 0창 · 나머지는 대명사 없이 "The person". 같은 시퀀스 같은 사람인데 두 창만
# 남성 대명사다. 프롬프트가 성별을 안 주니 **창마다 따로 추측**하는 것이다.
#
# GT 는 없다. `smplx_result/<seq>_smplx_results.pkl` 에 `gender` 키가 **있지만 569개 전량이
# 빈 문자열**이고 `betas` 도 (0,) 로 비어 있다 (2026-08-25 확인). 루트 `betas.npy` 는 코퍼스
# 전체에 body shape 5종뿐이고 성별 라벨이 아니다. `trumans_utils` 는 SMPL-X **male** 메시로
# 렌더하지만(`joints_to_smplx.py:34`, `load_smplx_animatioin_clear.py:186`) 그건 우리가 보는
# `video_render` 를 만든 경로라는 보장이 없다.
#
# 그래서 **시퀀스당 한 번만** VLM 에게 묻고 그 답을 37창 전부에 못 박는다. 창마다 묻는 것보다
# 정확해서가 아니라 **일관되기 때문**이다 — 틀리더라도 37창이 같이 틀려야 하류에서 고칠 수 있다.
# 사람이 카메라에 가장 가까운 프레임을 고르는 데 `smplx_result_in_cam` 의 `transl[:,2]` 를 쓴다
# (카메라 좌표계 z, |z| 작을수록 크게 찍힌다). |z| < 0.8 m 는 버린다 — 실측 18프레임이 여기
# 걸리는데 몸 일부만 화면을 채운다.
# ---------------------------------------------------------------------------------------------

#    성별 → (대명사 3종, 사람을 부르는 말, **금지 대명사**). `person` 은 지금까지의 동작이다.
PRONOUNS = {
    "man":    {"words": "he/him/his", "noun": "the man",    "banned": ("she", "her", "hers")},
    "woman":  {"words": "she/her/her", "noun": "the woman", "banned": ("he", "him", "his")},
    "person": {"words": "they/them/their", "noun": "the person", "banned": ()},
}

PERSON_SYSTEM = (
    "You identify the appearance of a computer-generated human character in a rendered "
    "animation, so that a downstream description pipeline refers to the character "
    "consistently. Answer only about what is visible. Return raw JSON only."
)

PERSON_PROMPT = """The tiles above are the {n} frames of this take where the character is \
closest to the viewpoint, so they are the clearest look at the character. It is the SAME \
character in every tile.

Report the character's apparent presentation. Return raw JSON only, no markdown fences:

{{
  "gender": "<man | woman | unclear>",
  "hair": "<length and colour, one short phrase>",
  "clothing": "<what they are wearing, one short phrase>",
  "evidence": "<one sentence: what in the images supports the gender call>",
  "confidence": <0.0-1.0>
}}

Rules:
- This is a rendered avatar, not a real person. Judge only visible presentation \
(hair length, build, clothing silhouette).
- Answer "unclear" when the character is too small, too dark, or turned away in every tile. \
Guessing is worse than "unclear" here — the answer is reused for the whole take."""


def validate_person(payload):
    """인물 probe 스키마 위반 목록."""
    problems = []
    if payload.get("gender") not in ("man", "woman", "unclear"):
        problems.append(f"gender must be exactly \"man\", \"woman\" or \"unclear\", "
                        f"got {payload.get('gender')!r}")
    for key in ("hair", "clothing", "evidence"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            problems.append(f"missing_or_empty_string: {key}")
    try:
        value = float(payload.get("confidence"))
    except (TypeError, ValueError):
        problems.append("confidence must be a number between 0 and 1")
    else:
        if not 0.0 <= value <= 1.0:
            problems.append(f"confidence must be between 0 and 1, got {value}")
    return problems


def person_frames(trumans: str, sequence: str, total: int, count=4, min_gap=100, min_z=0.8):
    """사람이 카메라에 가장 가까운 프레임 `count` 장. 서로 `min_gap` 이상 떨어뜨린다.

    `smplx_result_in_cam` 이 없으면 `[]` 를 돌려주고 호출자가 균등 샘플로 내려간다.
    `min_gap` 을 두는 이유: |z| 최소 근방은 연속 프레임이라 4장이 사실상 같은 그림이 된다.
    """
    pkl = path.join(trumans, "smplx_result_in_cam", f"{sequence}_smplx_results.pkl")
    if not path.exists(pkl):
        return []
    with open(pkl, "rb") as handle:
        depth = np.abs(np.asarray(pickle.load(handle)["transl"], dtype=np.float64)[:, 2])
    picked = []
    for index in np.argsort(depth):
        index = int(index)
        if index >= total or depth[index] < min_z:
            continue
        if all(abs(index - other) >= min_gap for other in picked):
            picked.append(index)
        if len(picked) == count:
            break
    return sorted(picked)


def resolve_person(gender: str, confidence: float, min_confidence=0.6):
    """probe 결과 → `PRONOUNS` 키. `unclear` 거나 확신이 낮으면 지금까지의 `person` 으로."""
    if gender in ("man", "woman") and confidence >= min_confidence:
        return gender
    return "person"


def person_block(person: str, probe=None):
    """`## PERSON` 블록. 대명사를 **못 박는 게 목적**이라 외형 묘사는 짧게만 붙인다."""
    style = PRONOUNS[person]
    lines = [f"\n## PERSON (same character in every stretch of this take)",
             f"There is exactly ONE person in this take. Refer to them as \"{style['noun']}\" "
             f"and use {style['words']} throughout, in every field."]
    if probe and probe.get("hair"):
        lines.append(f"Appearance: {probe['hair']}, wearing {probe.get('clothing', 'n/a')}.")
    if person == "person":
        lines.append("The character's gender is not identifiable, so use singular \"they\" — "
                     "never \"he\" or \"she\".")
    lines.append("Start \"shot_description\" with \"" + style["noun"].capitalize() + "\".")
    return "\n".join(lines)


def scene_block(readable, absent):
    """프롬프트에 붙는 `## SCENE OBJECTS` 블록. 있는 것과 **없는 것**을 둘 다 못 박는다."""
    text = ("\n## SCENE OBJECTS (ground truth for this recording)\n"
            "The only movable/interactive objects in this room are: "
            + ", ".join(readable) + ".\n"
            "Do not name any other interactive object. In particular this room has NO "
            + ", ".join(sorted(absent)) + ".\n"
            "Built-in furniture (sofa, table, shelves, walls, floor) is not in this list — "
            "you may still refer to it, but never rename an object from the list above.")
    return text


def motion_block(moved):
    """`## OBJECTS THAT MOVE` 블록. 물체가 안 움직였으면 그 사실도 단서다."""
    if not moved:
        return ("\n## OBJECTS THAT MOVE IN THIS STRETCH\n"
                "None. No interactive object is handled here — the person only moves their body.")
    lines = [f"- {name} (moves {dl:.2f} m, rotates {dr:.0f} deg)" for name, dl, dr in moved]
    return ("\n## OBJECTS THAT MOVE IN THIS STRETCH (from the recording, not from the images)\n"
            + "\n".join(lines)
            + "\nThese are the objects actually being handled. Name them correctly.")


def rule_block(window, windows):
    """`## LABELLED ACTION` 블록. gap 창은 **앞뒤 action 라벨**을 시간 거리와 함께 준다."""
    if window["kind"] == "action":
        return ("\n## LABELLED ACTION (ground truth for this stretch)\n"
                f"\"{window['text']}\"\n"
                "This label is correct. Your description must be consistent with it — do not "
                "contradict the action or the object it names. Add the spatial detail the label "
                "lacks (where in the room, what furniture is nearby, which way the person faces).")
    before = [w for w in windows if w["kind"] == "action" and w["end"] < window["start"]]
    after = [w for w in windows if w["kind"] == "action" and w["start"] > window["end"]]
    lines = []
    if before:
        gap = window["start"] - before[-1]["end"]
        lines.append(f"- Ends {gap} frame{'' if gap == 1 else 's'} before this stretch: "
                     f"\"{before[-1]['text']}\"")
    if after:
        gap = after[0]["start"] - window["end"]
        lines.append(f"- Starts {gap} frame{'' if gap == 1 else 's'} after this stretch: "
                     f"\"{after[0]['text']}\"")
    if not lines:
        return ""
    return ("\n## NEIGHBOURING LABELLED ACTIONS (ground truth, for context only)\n"
            + "\n".join(lines)
            + "\nThis stretch itself is unlabelled — it is what happens between those two. "
              "Do not describe the neighbouring actions themselves.")


def build_windows(actions, total, include_gaps=False, gap_min=15, gap_max=120):
    """[{'kind','start','end','text','action_index','action_id'}] 를 **시간순**으로 만든다.

    `include_gaps=False` 면 Actions 줄 그대로라 예전 실행과 완전히 같다.

    켜면 Actions 구간 **사이·앞뒤의 빈 구간**을 창으로 추가한다. `gap_min` 미만은 버린다
    (전환 프레임 몇 장에 VLM 을 한 번 부르는 건 낭비고, 서술도 "he is standing" 밖에 안 나온다).
    `gap_max` 를 넘으면 균등 분할한다 — 실측 최대 공백이 494 프레임(f1433-1926)인데 그 안에
    walk 45% / turn 18% / pose 18% / still 19% 가 섞여 있어 한 문장으로 못 적는다.
    상태 경계에서 자르는 쪽이 의미상 낫지만 임계를 하나 더 들여야 해서 균등 분할로 둔다.
    """
    windows = []
    for index, action in enumerate(actions):
        windows.append({"kind": "action", "start": int(action["start"]),
                        "end": int(action["end"]), "text": action["text"],
                        "action_index": index, "action_id": int(action["id"])})
    if include_gaps:
        covered = sorted((int(a["start"]), int(a["end"])) for a in actions)
        gaps, cursor = [], 0
        for start, end in covered:
            if start - cursor >= gap_min:
                gaps.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if total - cursor >= gap_min:
            gaps.append((cursor, total - 1))
        for gap_index, (start, end) in enumerate(gaps):
            length = end - start + 1
            chunks = int(np.ceil(length / gap_max))
            bounds = np.rint(np.linspace(start, end + 1, chunks + 1)).astype(int)
            for part in range(chunks):
                windows.append({"kind": "gap", "start": int(bounds[part]),
                                "end": int(bounds[part + 1]) - 1, "text": "",
                                "action_index": None, "action_id": None,
                                "gap_index": gap_index, "gap_part": part, "gap_parts": chunks})
    windows.sort(key=lambda w: (w["start"], w["end"]))
    for number, window in enumerate(windows):
        window["window"] = number + 1
    return windows


def tile_strip(frames, labels, tile_width=648, rows=2, gamma=1.0):
    """프레임 리스트 → contact sheet (좌→우, 위→아래). 타일 좌상단에 시간 라벨.

    타일 폭 기본값이 원본 폭(648)인 이유: 320 으로 줄였더니 VLM 이 손에 든 책을 리모컨으로
    읽었다 (씬·자세·가구는 맞췄다). 사람이 든 소품이 이 데이터에서 제일 작은 단서라 다운스케일
    여유가 없다. 6장을 한 줄로 붙이면 3888 px 짜리 초광폭이 되므로 `rows` 로 접는다.

    `gamma > 1` 이면 밝힌다. TRUMANS `video_render` 는 median 58/255 로 어둡다 — 사람이 어두운
    가구 앞에 서면 실루엣이 배경에 묻힌다. 픽셀을 건드리는 것이므로 기본은 1.0 (off).
    """
    tiles = []
    for frame, label in zip(frames, labels):
        height = int(round(frame.shape[0] * tile_width / frame.shape[1]))
        tile = cv2.resize(frame, (tile_width, height), interpolation=cv2.INTER_AREA)
        if gamma != 1.0:
            table = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255.0,
                            0, 255).astype(np.uint8)
            tile = cv2.LUT(tile, table)
        cv2.rectangle(tile, (0, 0), (74, 34), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2,
                    cv2.LINE_AA)
        tiles.append(tile)
    #    마지막 줄이 모자라면 검은 타일로 채운다 (`np.concatenate` 가 줄 폭을 맞춰야 한다).
    per_row = int(np.ceil(len(tiles) / max(1, rows)))
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[i:i + per_row], axis=1)
                           for i in range(0, len(tiles), per_row)], axis=0)


def read_frames(video_path: str, indices):
    """지정 인덱스만 읽는다. `CAP_PROP_POS_FRAMES` 탐색은 B-frame 에서 어긋나므로 순차로 훑는다."""
    wanted, out = set(int(i) for i in indices), {}
    capture = cv2.VideoCapture(video_path)
    assert capture.isOpened(), f"영상을 못 연다: {video_path}"
    index, last = 0, max(wanted)
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            out[index] = frame
        index += 1
    capture.release()
    missing = sorted(wanted - set(out))
    assert not missing, f"{video_path} 에서 프레임 {missing[:5]} 을 못 읽었다 (총 {index}장)"
    return [out[int(i)] for i in indices]


def validate(payload, absent_objects=(), person=""):
    """스키마 위반 목록. 빈 리스트면 통과.

    `absent_objects` 는 **TRUMANS 물체 어휘에는 있는데 이 녹화엔 없는** 명사들이다 (기본 빈
    튜플 = 예전 동작). "cabinet"/"monitor" 처럼 그럴듯한 오인을 잡는다 — 씬에 오븐이 있는데
    cabinet 이라고 쓰면 그건 이 데이터셋 어휘 안에서의 혼동이라 확실히 틀린 것이다.
    sofa/desk 같은 붙박이 가구는 `Object_all` 어휘에 없으므로 여기 안 걸린다.

    `person` 이 `man`/`woman` 이면 **반대 성별 대명사를 반려**한다 (기본 빈 문자열 = 예전 동작).
    프롬프트로 부탁만 해서는 안 지켜진다 — 대명사는 한 시퀀스 안에서 일관돼야 하고, 그건
    재질의로 강제할 수 있는 종류의 제약이다.
    """
    problems = []
    for key in ("action", "location", "body_facing", "shot_description"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            problems.append(f"missing_or_empty_string: {key}")
    if not isinstance(payload.get("objects"), list):
        problems.append("objects must be a list of strings (use [] when nothing is handled)")
    words = len(str(payload.get("shot_description", "")).split())
    if payload.get("shot_description") and not 10 <= words <= 45:
        problems.append(f"shot_description must be 15-35 words, got {words}")
    #    카메라 어휘는 **재질의로 되돌린다** (사후 문자열 치환이 아니라). 치환하면 문장 구조가
    #    깨지고, 무엇보다 그 단어를 쓴 서술은 방향 표현 자체가 카메라 기준이라 부분 수정이 안 된다.
    blob = " ".join(str(payload.get(k, "")) for k in
                    ("action", "location", "body_facing", "shot_description")).lower()
    banned = [w for w in ("camera", "shot", "take", "frame", "screen left", "screen right",
                          "close-up", "closeup", "wide angle", "footage", "viewer")
              if w in blob]
    if banned:
        problems.append(f"forbidden filming vocabulary: {banned}. "
                        "Rewrite using room and object references only.")
    #    씬에 없는 물체를 지어낸 경우. `objects` 리스트까지 포함해 본다.
    #    부분문자열이 아니라 **단어 경계**로 찾는다 — "cup" 이 "cupboard" 에, "pen" 이 "open" 에
    #    걸린다. 복수형만 허용(`s?`).
    blob_objects = (blob + " " + " ".join(str(o) for o in payload.get("objects", [])
                                          if isinstance(o, str))).lower()
    ghosts = [w for w in absent_objects
              if re.search(rf"\b{re.escape(w)}s?\b", blob_objects)]
    if ghosts:
        problems.append(f"objects that do not exist in this room: {sorted(ghosts)}. "
                        "Re-read the SCENE OBJECTS list and name the correct object.")
    #    성별 대명사. 단어 경계로 찾는다 — "he" 가 "the"/"she" 에, "his" 가 "this" 에 걸린다.
    style = PRONOUNS.get(person)
    if style and style["banned"]:
        wrong = sorted({w for w in style["banned"] if re.search(rf"\b{w}\b", blob)})
        if wrong:
            problems.append(f"wrong pronouns for this character: {wrong}. "
                            f"This is {style['noun']} — use {style['words']} in every field.")
    return problems


def main():
    parser = ArgumentParser()
    parser.add_argument("--trumans", default=TRUMANS_DEFAULT, type=str)
    parser.add_argument("--sequence", required=True, type=str)     # 예: 2023-01-17@00-55-00
    parser.add_argument("--out", default=path.join(CINEMATRAJ_ROOT, "out", "trumans_vlm_actions"),
                        type=str)
    # 1-based 창 번호. 비우면 Actions 파일 전량.
    parser.add_argument("--windows", default=[], nargs="*", type=int)
    parser.add_argument("--num_tiles", default=6, type=int)        # contact sheet 타일 수
    parser.add_argument("--tile_width", default=648, type=int)     # 원본 폭. 줄이면 소품을 놓친다
    parser.add_argument("--grid_rows", default=2, type=int)
    parser.add_argument("--gamma", default=1.0, type=float)        # >1 이면 밝힌다 (원본 median 58)
    parser.add_argument("--api_base", default="http://127.0.0.1:22002/v1", type=str)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct", type=str)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--verbose", action="store_true")
    # Actions 밖 미태그 구간도 창으로 만들어 같이 태깅한다. 기본 off = 예전 실행과 동일.
    parser.add_argument("--include_gaps", dest="include_gaps", action="store_true")
    parser.add_argument("--no_include_gaps", dest="include_gaps", action="store_false")
    parser.set_defaults(include_gaps=False)
    parser.add_argument("--gap_min_frames", default=15, type=int)   # 이보다 짧은 공백은 버린다
    parser.add_argument("--gap_max_frames", default=120, type=int)  # 넘으면 균등 분할
    #    씬 접지 3종. 전부 기본 off = 예전 실행과 바이트 동일.
    #    obj_list.txt 의 물체 목록 + 이 방에 **없는** 물체를 프롬프트에 못 박고 validate 에도 건다.
    parser.add_argument("--scene_objects", dest="scene_objects", action="store_true")
    parser.add_argument("--no_scene_objects", dest="scene_objects", action="store_false")
    parser.set_defaults(scene_objects=False)
    #    Object_pose 로 창 안에서 실제로 움직인 물체를 알려준다.
    parser.add_argument("--object_motion", dest="object_motion", action="store_true")
    parser.add_argument("--no_object_motion", dest="object_motion", action="store_false")
    parser.set_defaults(object_motion=False)
    #    규칙 라벨 주입. action 창은 자기 라벨, gap 창은 앞뒤 라벨. **두 팔의 독립성을 줄인다**
    #    (narrative 가 rule 의 rewrite 에 가까워진다) — 그래서 별도 플래그로 뺐다.
    parser.add_argument("--rule_hint", dest="rule_hint", action="store_true")
    parser.add_argument("--no_rule_hint", dest="rule_hint", action="store_false")
    parser.set_defaults(rule_hint=False)
    #    인물 접지. `--person_probe` 는 시퀀스당 VLM 을 **한 번 더** 불러 성별/외형을 정하고
    #    그 결과를 전 창 프롬프트 + validate 에 못 박는다. `--person` 은 그 probe 를 건너뛰는
    #    수동 지정 (probe 가 틀렸을 때의 탈출구). 둘 다 기본 off = 예전 동작(대명사 자유).
    parser.add_argument("--person_probe", dest="person_probe", action="store_true")
    parser.add_argument("--no_person_probe", dest="person_probe", action="store_false")
    parser.set_defaults(person_probe=False)
    parser.add_argument("--person", default="", choices=["", "man", "woman", "person"])
    parser.add_argument("--person_tiles", default=4, type=int)        # probe sheet 타일 수
    parser.add_argument("--person_min_conf", default=0.6, type=float)  # 밑돌면 they/their
    parser.add_argument("--recording", default="", type=str)        # 비우면 scene_flag 로 역산
    parser.add_argument("--move_meters", default=0.05, type=float)  # 실측 노이즈 바닥 0.015
    parser.add_argument("--move_degrees", default=12.0, type=float)  # 실측 노이즈 바닥 6.9
    parser.add_argument("--suffix", default="", type=str)           # 산출물 이름 접미사 override
    args = parser.parse_args()

    video = path.join(args.trumans, "video_render", f"{args.sequence}.pkl.mp4")
    assert path.exists(video), f"video_render 가 없다: {video}"
    actions = read_actions(args.trumans, args.sequence)
    assert actions, f"Actions/{args.sequence}.txt 가 비었다"
    total = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))

    windows = build_windows(actions, total, include_gaps=args.include_gaps,
                            gap_min=args.gap_min_frames, gap_max=args.gap_max_frames)
    picked = ([w for w in windows if w["window"] in set(args.windows)]
              if args.windows else windows)

    #    씬 접지. 셋 다 꺼져 있으면 여기서 아무것도 안 읽는다 (seg_name.npy 는 425 MB 다).
    recording, scene_raw, scene_readable, absent, poses = "", [], [], (), {}
    if args.scene_objects or args.object_motion:
        recording = args.recording or resolve_recording(args.trumans, args.sequence)
        scene_raw, scene_readable = read_scene_objects(args.trumans, recording)
    if args.scene_objects:
        absent = absent_objects(args.trumans, scene_readable)
    if args.object_motion:
        poses = read_object_poses(args.trumans, args.sequence, scene_raw)
        assert poses, f"Object_pose 에서 {args.sequence} 물체를 하나도 못 읽었다"

    #    gap 을 켜면 창 번호가 밀리므로 산출물을 **다른 이름**으로 쓴다. 기존 narrative arm 의
    #    JSON/sheet 를 덮지 않기 위함. 접지 팔도 또 갈라야 해서 `_g` 를 덧붙인다.
    suffix = args.suffix or (("_full" if args.include_gaps else "")
                             + ("_g" if (args.scene_objects or args.object_motion
                                         or args.rule_hint) else "")
                             + ("p" if (args.person_probe or args.person) else ""))
    #    sheet 는 접지 플래그와 무관하게 같은 그림이므로 **창 분할에만** 따라 갈린다. 접지 팔이
    #    같은 sheet 를 재사용하게 두면 두 팔이 정확히 한 입력(프롬프트 텍스트)만 달라진다.
    #    시퀀스별로 한 겹 더 나눈다: sheet 이름은 창 번호 + action 태그뿐이라 **시퀀스가 달라도
    #    같은 경로**가 나온다. 한 시퀀스씩 돌 때는 안 드러나지만 여러 편을 동시에 돌리면
    #    한 워커가 쓰는 중인 PNG 를 다른 워커가 서버에 넘겨 `broken PNG file` 500 이 뜬다.
    #    접지 팔끼리의 sheet 재사용은 같은 시퀀스 안에서 그대로 유지된다.
    sheets = path.join(args.out, f"sheets{'_full' if args.include_gaps else ''}", args.sequence)
    makedirs(sheets, exist_ok=True)
    client = VLMClient(api_base=args.api_base, model=args.model, temperature=args.temperature)

    started = time.time()
    #    인물 접지. `--person` 수동 지정이 probe 를 이긴다 (probe 가 틀렸을 때의 탈출구).
    person, probe, probe_frames, probe_sheet = "", None, [], ""
    if args.person:
        person = args.person
    elif args.person_probe:
        probe_frames = person_frames(args.trumans, args.sequence, total, args.person_tiles)
        if not probe_frames:      # smplx_result_in_cam 이 없으면 균등 샘플로 내려간다
            probe_frames = np.unique(np.rint(
                np.linspace(0, total - 1, args.person_tiles)).astype(int)).tolist()
            print(f"{'person':<14}smplx_result_in_cam 없음 → 균등 샘플 {probe_frames}")
        #    `person{suffix}.png` 는 시퀀스가 안 들어가서 동시 실행 시 서로 덮어쓴다 (위 sheets
        #    주석과 같은 이유). 시퀀스를 붙여 워커마다 다른 파일이 되게 한다.
        probe_sheet = path.join(args.out, f"person{suffix}_{args.sequence}.png")
        #    타일 폭을 2배로 준다. 창 sheet 는 648 px 인데 사람이 100 px 남짓이라 머리 길이·
        #    옷 실루엣이 안 보인다 — 이 한 장에만 해상도를 더 쓴다.
        cv2.imwrite(probe_sheet, tile_strip(read_frames(video, probe_frames),
                                            [f"t{i}" for i in range(len(probe_frames))],
                                            args.tile_width * 2, rows=2, gamma=max(args.gamma, 1.6)))
        probe, probe_info = client.chat_json(
            PERSON_PROMPT.format(n=len(probe_frames)), images=[probe_sheet],
            system=PERSON_SYSTEM, validate=validate_person, label="person",
            echo_previous=False, hint_first=True)
        if probe is None:
            probe = {"gender": "unclear", "confidence": 0.0,
                     "evidence": "probe 소진", "repairs": probe_info.get("repairs", 0)}
        person = resolve_person(probe.get("gender", "unclear"),
                                float(probe.get("confidence") or 0.0), args.person_min_conf)
        print(f"{'person':<14}{probe.get('gender')} conf {probe.get('confidence')} "
              f"→ {person} ({PRONOUNS[person]['words']})  frames {probe_frames}")
        print(f"{'':14}{probe.get('hair', '')} / {probe.get('clothing', '')}")
        print(f"{'':14}{probe.get('evidence', '')}")

    entries = []
    for window in picked:
        number, is_gap = window["window"], window["kind"] == "gap"
        #    창이 짧으면 타일 수를 줄인다. 같은 프레임을 여러 장 붙이면 VLM 이 "정지"로 읽는다.
        start = max(0, min(int(window["start"]), total - 1))
        end = max(start, min(int(window["end"]), total - 1))
        n_tiles = int(min(args.num_tiles, end - start + 1))
        frames_idx = np.unique(np.rint(np.linspace(start, end, n_tiles)).astype(int)).tolist()
        labels = [f"t{i}" for i in range(len(frames_idx))]
        sheet = tile_strip(read_frames(video, frames_idx), labels, args.tile_width,
                           rows=args.grid_rows, gamma=args.gamma)
        tag = (f"g{window['gap_index']:02d}p{window['gap_part']}" if is_gap
               else f"a{window['action_index']:02d}")
        sheet_path = path.join(sheets, f"w{number:02d}_{tag}.png")
        cv2.imwrite(sheet_path, sheet)

        prompt = PROMPT.format(n=len(frames_idx), last=len(frames_idx) - 1)
        if is_gap:
            prompt = f"{prompt}\n{GAP_SUFFIX}"
        moved = moved_objects(poses, start, end, args.move_meters,
                              args.move_degrees) if args.object_motion else []
        if args.scene_objects:
            prompt = f"{prompt}\n{scene_block(scene_readable, absent)}"
        if args.object_motion:
            prompt = f"{prompt}\n{motion_block(moved)}"
        if args.rule_hint:
            prompt = f"{prompt}\n{rule_block(window, windows)}"
        if person:
            prompt = f"{prompt}\n{person_block(person, probe)}"
        payload, info = client.chat_json(
            prompt, images=[sheet_path], system=SYSTEM,
            validate=lambda p: validate(p, absent, person), label=f"w{number:02d}",
            #    접지 블록을 켜면 원 프롬프트가 ~600 토큰 길어져 뒤에 붙는 힌트가 묻힌다.
            #    그래서 재질의에서는 위반+힌트를 맨 앞으로 올린다.
            echo_previous=False, repair_hint=REPAIR_HINT, hint_first=True)
        entry = {
            "window": number, "kind": window["kind"],
            "action_index": window["action_index"], "action_id": window["action_id"],
            "start": int(window["start"]), "end": int(window["end"]),
            "n_frames": int(window["end"]) - int(window["start"]) + 1,
            "rule_text": window["text"],                            # 대조용. 프롬프트엔 안 넣었다
            "frames_sampled": frames_idx, "sheet": sheet_path,
            "source": "vlm" if payload else "fallback",
            "repairs": info.get("repairs", 0), "turns": info.get("turns", 0),
            #    접지 근거를 산출물에 같이 실어야 나중에 "왜 이렇게 썼나"를 되짚을 수 있다.
            "moved_objects": [{"object": n, "delta_m": round(dl, 4), "delta_deg": round(dr, 1)}
                              for n, dl, dr in moved],
        }
        if is_gap:
            entry.update({k: window[k] for k in ("gap_index", "gap_part", "gap_parts")})
        if payload:
            entry.update({k: payload.get(k) for k in
                          ("action", "location", "objects", "body_facing", "shot_description")})
        else:
            #    소진돼도 hard-fail 하지 않는다 — 그 창만 규칙 라벨로 되돌린다. gap 은 라벨 자체가
            #    없으므로 그 사실을 그대로 적는다 (빈 문자열이면 하류가 조용히 shot 을 잃는다).
            noun = PRONOUNS[person or "person"]["noun"].capitalize()
            entry["shot_description"] = window["text"] or f"{noun} moves through the room."
        entries.append(entry)
        if args.verbose:
            print(f"  w{number:02d} [{window['kind']:6s}] {start:>5}..{end:<5} "
                  f"[{entry['source']}] {entry['shot_description']}")

    out_path = path.join(args.out, f"{args.sequence}_actions_vlm{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump({"format": "trumans_vlm_actions_v1", "sequence": args.sequence,
                   "video_render": video, "video_frames": total,
                   "model": args.model, "num_tiles": args.num_tiles,
                   "includes_gaps": bool(args.include_gaps),
                   "gap_min_frames": args.gap_min_frames, "gap_max_frames": args.gap_max_frames,
                   "grounding": {"scene_objects": bool(args.scene_objects),
                                 "object_motion": bool(args.object_motion),
                                 "rule_hint": bool(args.rule_hint),
                                 "recording": recording, "objects": scene_readable,
                                 "absent_objects": list(absent),
                                 "move_meters": args.move_meters,
                                 "move_degrees": args.move_degrees},
                   #    인물 접지 근거. `person` 이 실제로 쓰인 값, `probe` 가 VLM 원답이다
                   #    (confidence 가 임계 미만이면 둘이 갈린다 — 그 기록이 남아야 한다).
                   "person": {"resolved": person, "source": ("manual" if args.person else
                                                             "probe" if args.person_probe
                                                             else "off"),
                              "pronouns": PRONOUNS[person]["words"] if person else "",
                              "probe": probe, "probe_frames": list(probe_frames),
                              "probe_sheet": probe_sheet,
                              "min_confidence": args.person_min_conf},
                   "seconds": round(time.time() - started, 1),
                   "entries": entries}, file, ensure_ascii=False, indent=2)
    client.save_trace(path.join(args.out, f"trace{suffix}"))

    print(f"\n{'sequence':<14}{args.sequence}")
    print(f"{'video':<14}{video}  ({total}프레임)")
    covered = sum(e["n_frames"] for e in entries)
    print(f"{'windows':<14}{len(entries)} / {len(windows)}  "
          f"(action {sum(1 for e in entries if e['kind'] == 'action')} / "
          f"gap {sum(1 for e in entries if e['kind'] == 'gap')})")
    print(f"{'coverage':<14}{covered} / {total} 프레임  ({covered / max(total, 1) * 100:.1f}%)")
    print(f"{'fallback':<14}{sum(1 for e in entries if e['source'] == 'fallback')}")
    print(f"{'repairs':<14}{sum(e['repairs'] for e in entries)}")
    print(f"{'seconds':<14}{time.time() - started:.1f}")
    print()
    for entry in entries:
        print(f"w{entry['window']:02d} rule  {entry['rule_text']}")
        print(f"    vlm   {entry['shot_description']}")
    print(f"\n{'out':<14}{out_path}")


if __name__ == "__main__":
    main()
