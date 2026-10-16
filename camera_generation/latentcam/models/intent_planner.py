# VLM planner using Qwen3-VL
# Input : video and high level intent text
# Output : structured low level camera plan
"""이 코드가 답하는 질문: (video, high-level intent) 를 카메라 모델이 받는 조건 (camera 문장 · Molmo2 문장 · subject 질의) 으로 어떻게 바꾸나?

R416 (사용자 2026-10-16 "Intent Planner 를 둬서 video, high-level intent 를 받아서 canonical camera plan 을 만들어서 카메라를 canonical
camera plan 을 받아서 만들게 할건데 지금은 이 구조가 없어. 코드만 구현해줘").

    video ─┐                         ┌─ camera_text   (T5/umt5 — 학습 concise 캡션과 같은 문장 생성기)
           ├─ IntentPlanner (VLM) ─→ CanonicalCameraPlan ─→ plan_to_conditions ─┼─ molmo2_text   (`Track {target_text}.` / 없으면 camera_text, D191)
    intent ┘   primitive 로 답하게 함       (결정론적 정규화)                     └─ subject_query (noun → SAM3 후보 + text → Molmo2 점 vote, 기본 subject 찾기)

설계 결정 셋:
  1. **VLM 은 preset 이름이 아니라 primitive 동작 (`dolly_in`, `orbit_left` …) 을 고른다.** 82 개 이름을 외우게 하면 없는 이름
     (`dolly_in_orbit_left`) 을 지어낸다. primitive 조합 → preset 이름은 여기서 결정론적으로 푼다 (`PresetVocab.resolve`).
     dolly + orbit 은 우리 어휘에서 `push_in_arc_*`/`pull_out_arc_*` 이므로 동의어로 접는다.
  2. **문장은 새로 짓지 않는다.** camera_text 는 뱅크 캡션 빌더 (`fit/caption/build_bank_captions.caption_of` + `nl_prompt`) 를 그대로
     태운다 — 학습 때 T5 가 본 문장 분포와 글자 단위로 같아야 조건이 먹는다. Molmo2 문장도 `build_molmo2_track_override` 규칙 그대로.
  3. **plan 은 정규형 (canonical) 이다.** 같은 뜻이면 같은 dict (`to_json` sort_keys) — 캐시 키·중복 제거·평가 대조에 쓴다.
     크기 (`magnitude`) · 속도 (`speed`) 는 plan 에 남기지만 **지금 카메라 모델은 안 읽는다** (`consumed=False`, 아래 문제점 참고).

env latentcam (lbm.presets · 캡션 빌더 import 확인). VLM 서버: `screen -dmS vlm bash camera_generation/dataset/exec/serve_qwen3vl.sh` (127.0.0.1:22002).
  python models/intent_planner.py --selftest                                    # 82 preset 왕복 (plan → 문장 == 뱅크 캡션) 검사, 서버 불필요
  python models/intent_planner.py --video a.mp4 --intent "이 사람을 왼쪽에서 돌면서 다가가줘" --out plan.json
"""
import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from os import path

LATENTCAM = path.dirname(path.dirname(path.abspath(__file__)))
REPO = path.dirname(path.dirname(LATENTCAM))
DATASET = path.join(REPO, "camera_generation", "dataset")
if DATASET not in sys.path:
    sys.path.insert(0, DATASET)

import fit.caption.build_bank_captions as BC  # noqa: E402
from lbm.presets import PRESETS, resolve_preset  # noqa: E402

# v5f fit `--shot_class_presets` (configs/bank/v5f_pushsubject.json) — 끝 shot 목표가 학습 캡션에 실리는 preset 은 이것뿐이다.
SHOT_CLASS_PRESETS = r"^(track_)?(dolly_in_look_at|push_in_arc_(left|right)|pull_out_arc_(left|right))$"
SHOT_CLASSES = tuple(BC.SHOT_CLASS_TEXT)                       # extreme_wide · wide · medium · close_up · extreme_close_up
SPEEDS = ("steady", "ease", "accel", "decel")                  # lbm.presets._resample 어휘
MAGNITUDES = ("small", "medium", "large")
LEGACY = {"dolly_out_look_at_legacy"}                          # 옛 뱅크 재현용 — 어휘에 안 낸다

# preset 이름 → primitive. 이름 문법이 `[track_]<주>[_<부>]` 라 토큰으로 대부분 풀리고, 예외만 표로 둔다.
PRIMS = ("dolly_in", "dolly_out", "truck_left", "truck_right", "pedestal_up", "pedestal_down", "crane_up", "crane_down",
         "orbit_left", "orbit_right", "arc_left", "arc_right", "pan_left", "pan_right", "tilt_up", "tilt_down",
         "roll_left", "roll_right", "zoom_in", "zoom_out", "s_curve")
SPECIAL = {"push_in_arc_left": ("dolly_in", "arc_left"), "push_in_arc_right": ("dolly_in", "arc_right"),
           "pull_out_arc_left": ("dolly_out", "arc_left"), "pull_out_arc_right": ("dolly_out", "arc_right"),
           "static_hold": (), "static_look_at": (), "track_hold": (), "track_look_at": (), "static_zoom_in": ("zoom_in",),
           "dolly_in_look_at": ("dolly_in",), "free_pedestal_up": ("pedestal_up",), "free_pedestal_down": ("pedestal_down",)}


# dolly + orbit 은 우리 어휘에서 arc 다 (true_orbit 은 등거리라 전후진과 같이 못 산다) — 사용자 말 "dolly in orbit left" 를 여기로 접는다.
def fold_synonyms(key: frozenset):
    """dolly 가 같이 있으면 orbit_* → arc_* (3 개 이상 조합의 nearest 제안에도 같은 규칙)."""
    if key & {"dolly_in", "dolly_out"}:
        key = frozenset("arc_" + m.split("_")[1] if m.startswith("orbit_") else m for m in key)
    return key


def moves_of(name: str):
    """preset 이름 → (track, primitive 튜플)."""
    track = name.startswith("track_")
    base = name[len("track_"):] if track else name
    if name in SPECIAL or base in SPECIAL:
        return track, tuple(SPECIAL.get(name, SPECIAL.get(base)))
    toks, rest = [], base
    while rest:
        hit = next((p for p in sorted(PRIMS, key=len, reverse=True) if rest == p or rest.startswith(p + "_")), None)
        assert hit, f"preset '{name}' 을 primitive 로 못 푼다 (남은 '{rest}') — SPECIAL 에 추가할 것"
        toks.append(hit); rest = rest[len(hit) + 1:]
    return track, tuple(toks)


class PresetVocab:
    """preset 어휘 (lbm.presets.PRESETS + configs/caption_presets.json). `allowed` 로 학습 코퍼스가 실제로 본 preset 만 남길 수 있다."""

    def __init__(self, allowed=None):
        self.cfg = json.load(open(BC.PRESETS_DEFAULT))
        names = [n for n in PRESETS if n not in LEGACY and n in self.cfg["presets"]]
        if allowed is not None:
            miss = set(allowed) - set(names)
            assert not miss, f"allowed 에 어휘 밖 preset: {sorted(miss)}"
            names = [n for n in names if n in set(allowed)]
        self.info, self.index = {}, {}
        for n in names:
            track, mv = moves_of(n)
            aim = PRESETS[n][1]
            spec = self.cfg["presets"][n]
            slot = BC.TARGET_SLOT in spec["phrase"]
            self.info[n] = dict(track=track, moves=mv, aim=aim, targetless=bool(spec.get("targetless")),
                                needs_target=track or aim == "look_at" or (slot and not spec.get("phrase_targetless")),
                                shot_ok=bool(re.match(SHOT_CLASS_PRESETS, n)))
            self.index.setdefault((track, frozenset(mv)), []).append(n)

    def resolve(self, track: bool, moves, look_at: bool, has_target: bool):
        """primitive 조합 → (preset, 오류 목록). 같은 조합에 이름이 여럿이면 aim·target 유무로 가른다
        (`dolly_in` free vs `dolly_in_look_at` / `pedestal_up` (target 부름) vs `free_pedestal_up` (targetless))."""
        key = frozenset(moves)
        key = fold_synonyms(key)
        cands = self.index.get((bool(track), key), [])
        if not cands:
            near = sorted(self.index, key=lambda k: (-len(k[1] & key), len(k[1] - key), k[0] != bool(track)))[:4]
            return None, [f"no preset for track={bool(track)} moves={sorted(key)}; nearest: "
                          + ", ".join(self.index[k][0] for k in near)]
        if len(cands) > 1:
            want = "look_at" if look_at else "free"
            by_aim = [c for c in cands if self.info[c]["aim"] == want] or cands
            by_tgt = [c for c in by_aim if self.info[c]["targetless"] != has_target] or by_aim
            cands = by_tgt
        return cands[0], []

    def prompt_block(self):
        """VLM 에 보여 줄 어휘 — 허용 조합을 primitive 로 적는다 (이름은 안 보인다)."""
        rows = sorted({(("track + " if t else "") + (" + ".join(sorted(m)) or "hold")) for t, m in self.index})
        return "\n".join(f"- {r}" for r in rows)


@dataclass
class CanonicalCameraPlan:
    """정규형 카메라 계획. `preset` 은 파생값 (moves·track·aim 으로 `PresetVocab.resolve` 가 정한다)."""
    preset: str
    moves: list
    track: bool
    aim: str                                   # look_at | free
    target_noun: str = ""                      # SAM3 후보용 명사 (scene graph label 수준: man, dog, car)
    target_text: str = ""                      # referring expression (학습 캡션의 target_text 와 같은 결: 12~14 단어 지칭구)
    end_shot: str = ""                         # SHOT_CLASSES 중 하나, shot_ok preset 만
    magnitude: str = "medium"                  # 지금 모델은 안 읽음
    speed: str = "steady"                      # 지금 모델은 안 읽음
    intent: str = ""
    rationale: str = ""
    warnings: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self); d["moves"] = sorted(d["moves"]); return d

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def canonicalize(raw: dict, vocab: PresetVocab, intent: str = ""):
    """VLM JSON → (CanonicalCameraPlan | None, 오류 목록). 오류는 그대로 VLM 재질의에 쓴다 (lbm.vlm.chat_json validate)."""
    err, warn = [], []
    moves = raw.get("moves") or []
    if isinstance(moves, str):
        moves = [moves]
    moves = [str(m).strip().lower().replace(" ", "_").replace("-", "_") for m in moves]
    moves = [m for m in moves if m not in ("hold", "static", "none", "")]
    bad = [m for m in moves if m not in PRIMS]
    if bad:
        err.append(f"unknown moves {bad}; allowed primitives: {', '.join(PRIMS)}")
    track = bool(raw.get("track", False))
    tgt = raw.get("target") or {}
    noun = str(tgt.get("noun") or "").strip().lower() if isinstance(tgt, dict) else ""
    text = str(tgt.get("text") or "").strip() if isinstance(tgt, dict) else ""
    if noun and not text:
        text = f"the {noun}"
    look_at = bool(raw.get("keep_target_centered", bool(noun)))
    if err:
        return None, err
    preset, e = vocab.resolve(track, moves, look_at, bool(noun))
    if e:
        return None, e
    info = vocab.info[preset]
    if info["needs_target"] and not noun:
        return None, [f"'{preset}' needs a target (track / look-at / phrase names the subject); give target.noun and target.text"]
    if info["targetless"] and noun:
        warn.append(f"'{preset}' 은 targetless — target '{noun}' 을 문장·Molmo2 에서 뺀다")
        noun, text = "", ""
    shot = str(raw.get("end_shot") or "").strip().lower().replace("-", "_").replace(" ", "_")
    shot = {"closeup": "close_up", "close": "close_up", "medium_shot": "medium", "wide_shot": "wide"}.get(shot, shot)
    if shot and shot not in SHOT_CLASSES:
        return None, [f"end_shot '{shot}' not in {list(SHOT_CLASSES)}"]
    if shot and not info["shot_ok"]:
        warn.append(f"end_shot '{shot}' 은 '{preset}' 학습 캡션에 없는 절 — 뺀다"); shot = ""
    mag = str(raw.get("magnitude") or "medium").lower(); spd = str(raw.get("speed") or "steady").lower()
    if mag not in MAGNITUDES:
        warn.append(f"magnitude '{mag}' → medium"); mag = "medium"
    if spd not in SPEEDS:
        warn.append(f"speed '{spd}' → steady"); spd = "steady"
    return CanonicalCameraPlan(preset=preset, moves=list(info["moves"]), track=info["track"], aim=info["aim"],
                               target_noun=noun, target_text=text, end_shot=shot, magnitude=mag, speed=spd,
                               intent=intent, rationale=str(raw.get("rationale") or ""), warnings=warn), []


def plan_to_conditions(plan: CanonicalCameraPlan, vocab: PresetVocab):
    """plan → 카메라 모델 조건. camera_text 는 뱅크 캡션 빌더 기본값 (nl, magnitude off, nl_framing off) 과 같은 경로다."""
    variant = {"preset": plan.preset, "aim": plan.aim, "anchor_id": "plan", "anchor_label": plan.target_noun,
               "subject_in_frame": 1.0, "subject_area_med": 0.1}
    if plan.end_shot:
        variant["shot_target_class"] = plan.end_shot
    cfg = vocab.cfg
    spec = cfg["presets"][plan.preset]
    if not plan.target_text and BC.TARGET_SLOT in spec["phrase"] and spec.get("phrase_targetless"):
        cfg = {**cfg, "presets": {**cfg["presets"], plan.preset: {**spec, "phrase": spec["phrase_targetless"], "targetless": True}}}
    cap = BC.caption_of(variant, "", cfg, False, None, None, False, {"plan": plan.target_text} if plan.target_text else {},
                        "nl", True, 0.85, None, False)
    camera_text = BC.nl_prompt(cap)
    tt = str(cap.get("target_text") or "").strip()
    return {"camera_text": camera_text,
            "molmo2_text": f"Track {tt}." if tt else camera_text,        # build_molmo2_track_override 규칙 (--empty_target concise)
            "subject_query": {"noun": plan.target_noun, "text": plan.target_text} if plan.target_noun else None,
            "needs_subject_cond": bool(plan.target_noun),                 # track3d / box3d 조건을 subject finder 로 만들어야 하는가
            "preset": plan.preset, "aim": plan.aim, "end_shot": plan.end_shot,
            "unconsumed": {"magnitude": plan.magnitude, "speed": plan.speed}}


SYSTEM = ("You are a cinematographer planning ONE camera move for re-shooting the given video. "
          "Answer with raw JSON only.")
PROMPT = """The images are {n} frames sampled uniformly from a {sec:.1f}s video (in order).
User intent: "{intent}"

Plan a single camera move that fulfils the intent using ONLY these allowed move combinations
("track +" = the camera position follows the subject's own motion):
{vocab}

Return JSON:
{{"moves": [<1-2 primitives from the list, [] for a held camera>],
  "track": <true if the camera must follow a moving subject>,
  "keep_target_centered": <true if the camera re-aims at the target every frame>,
  "target": {{"noun": "<one lowercase noun, e.g. man, dog, car; empty if no subject>",
             "text": "<referring expression that singles it out in the video, 8-14 words>"}},
  "end_shot": "<one of {shots} or empty>",
  "magnitude": "<small|medium|large>", "speed": "<steady|ease|accel|decel>",
  "rationale": "<one sentence>"}}"""


def sample_frames(video: str, n: int, work_dir: str):
    import cv2
    cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    assert total > 0, f"영상을 못 읽는다: {video}"
    os.makedirs(work_dir, exist_ok=True); out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(i * (total - 1) / max(n - 1, 1))); ok, f = cap.read()
        if ok:
            p = path.join(work_dir, f"f{i:02d}.jpg"); cv2.imwrite(p, f); out.append(p)
    cap.release()
    return out, total / fps


class IntentPlanner:
    """(video, intent) → CanonicalCameraPlan. VLM 은 `lbm.vlm.VLMClient` (OpenAI 호환, Qwen3-VL) — `client` 로 갈아끼울 수 있다."""

    def __init__(self, client=None, allowed_presets=None, n_frames: int = 8, max_repairs: int = 3,
                 work_dir: str = path.join(REPO, "tmp", "intent_planner")):
        if client is None:
            from lbm.vlm import VLMClient
            client = VLMClient(temperature=0.1, max_tokens=512)
        self.client, self.vocab = client, PresetVocab(allowed_presets)
        self.n_frames, self.max_repairs, self.work_dir = n_frames, max_repairs, work_dir

    def plan(self, video: str, intent: str):
        """-> (plan | None, info). None 이면 재질의 소진 — 호출자가 fallback (예: static_look_at) 을 정한다 (hard-fail 안 함)."""
        frames, sec = sample_frames(video, self.n_frames, path.join(self.work_dir, path.splitext(path.basename(video))[0]))
        prompt = PROMPT.format(n=len(frames), sec=sec, intent=intent, vocab=self.vocab.prompt_block(), shots=list(SHOT_CLASSES))
        box = {}

        def validate(raw):
            p, e = canonicalize(raw, self.vocab, intent)
            box["plan"] = p
            return e

        raw, info = self.client.chat_json(prompt, images=frames, system=SYSTEM, validate=validate,
                                          max_repairs=self.max_repairs, label="intent_plan", echo_previous=False)
        return (box.get("plan") if raw is not None else None), {**info, "raw": raw}

    def __call__(self, video: str, intent: str):
        p, info = self.plan(video, intent)
        return (plan_to_conditions(p, self.vocab) if p else None), p, info


def plan_from_bank_row(row: dict, vocab: PresetVocab, target_text: str = ""):
    """뱅크 행 (prompts.json 항목 / bank.csv 행) → plan. 학습 데이터의 정답 plan — planner 평가·teacher forcing 용."""
    name = resolve_preset(row["preset"], row.get("aim"))
    info = vocab.info[name]
    noun = "" if info["targetless"] else str(row.get("anchor_label") or "")
    return CanonicalCameraPlan(preset=name, moves=list(info["moves"]), track=info["track"], aim=info["aim"], target_noun=noun,
                               target_text=(target_text or (f"the {noun}" if noun else "")) if noun else "",
                               end_shot=str(row.get("shot_target_class") or ""))


def selftest():
    """82 preset 전부: ① 이름 → primitive → resolve 가 같은 이름으로 돌아오나 ② plan 문장 == 뱅크 캡션 빌더 문장."""
    vocab = PresetVocab(); bad = []
    for n, inf in vocab.info.items():
        tgt = not inf["targetless"]
        raw = {"moves": list(inf["moves"]), "track": inf["track"], "keep_target_centered": inf["aim"] == "look_at",
               "target": {"noun": "man", "text": "the man in a grey cap"} if tgt else {}}
        p, e = canonicalize(raw, vocab)
        if e or p.preset != n:
            bad.append((n, p.preset if p else None, e)); continue
        v = {"preset": n, "aim": inf["aim"], "anchor_id": "x", "anchor_label": "man", "subject_in_frame": 1.0, "subject_area_med": 0.1}
        ref = BC.nl_prompt(BC.caption_of(v, "", vocab.cfg, False, None, None, False, {"x": "the man in a grey cap"} if tgt else {},
                                         "nl", True, 0.85, None, False))
        got = plan_to_conditions(p, vocab)["camera_text"]
        if got != ref:
            bad.append((n, got, ref))
    p, _ = canonicalize({"moves": ["dolly_in", "orbit_left"], "target": {"noun": "man", "text": "the man"}, "end_shot": "close_up"}, vocab)
    print("dolly_in + orbit_left →", p.preset, "|", plan_to_conditions(p, vocab)["camera_text"])
    print(f"selftest: {len(vocab.info)} preset, 불일치 {len(bad)}")
    for b in bad:
        print("  ", b)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--intent")
    ap.add_argument("--allowed_presets", default=None, help="쉼표 목록 또는 한 줄 하나 txt — 학습 코퍼스가 본 preset 만")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    allowed = None
    if a.allowed_presets:
        allowed = open(a.allowed_presets).read().split() if path.isfile(a.allowed_presets) else a.allowed_presets.split(",")
    cond, plan, info = IntentPlanner(allowed_presets=allowed, n_frames=a.n_frames)(a.video, a.intent)
    res = {"plan": plan.to_dict() if plan else None, "conditions": cond, "repairs": info.get("repairs"), "errors": info.get("errors")}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
