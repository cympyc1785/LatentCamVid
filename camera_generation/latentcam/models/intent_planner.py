# VLM planner using Qwen3-VL
# Input : video and high level intent text
# Output : structured low level camera plan
"""이 코드가 답하는 질문: (video, high-level intent) 를 카메라 모델이 받는 조건 (camera 문장 · Molmo2 문장 · subject 질의) 으로 어떻게 바꾸나?

R416 (사용자 2026-10-16 "Intent Planner 를 둬서 video, high-level intent 를 받아서 canonical camera plan 을 만들어서 카메라를 canonical
camera plan 을 받아서 만들게 할건데 지금은 이 구조가 없어. 코드만 구현해줘"). R417 에서 preset 집합 교체 · 생성기 피드백 루프 · 사용자
pointing 자리 · framing 옵션 · 영상 통째 입력 · intent 합성을 붙였다.

    video ─┐                         ┌─ camera_text   (T5/umt5 — 학습 concise 캡션과 같은 문장 생성기)
           ├─ IntentPlanner (VLM) ─→ CanonicalCameraPlan ─→ plan_to_conditions ─┼─ molmo2_text   (`Track {target_text}.` / 없으면 camera_text, D191)
    intent ┘   primitive 로 답하게 함       (결정론적 정규화)                     └─ subject_query (기본 vote: noun → SAM3 후보 + text → Molmo2 점 / 사용자 point)
                     ↑                                                           │
                     └──── (선택, 기본 꺼짐) Trajectory Generator 실패 사유 ◀──────┘

설계 결정:
  1. **VLM 은 preset 이름이 아니라 primitive 동작 (`dolly_in`, `orbit_left` …) 을 고른다.** 이름을 외우게 하면 없는 이름
     (`dolly_in_orbit_left`) 을 지어낸다. primitive 조합 → preset 이름은 여기서 결정론적으로 푼다 (`PresetVocab.resolve`).
     dolly + orbit 은 우리 어휘에서 `push_in_arc_*`/`pull_out_arc_*` 이므로 동의어로 접는다.
  2. **문장은 새로 짓지 않는다.** camera_text 는 뱅크 캡션 빌더 (`fit/caption/build_bank_captions.caption_of` + `nl_prompt`) 를 그대로
     태운다 — 학습 때 T5 가 본 문장 분포와 같아야 조건이 먹는다. Molmo2 문장도 `build_molmo2_track_override` 규칙 그대로.
  3. **preset 집합은 카메라 모델이 학습한 코퍼스를 따른다** (`--preset_set v1|v3|v5f|all|<json>`). 집합 파일은 코퍼스 prompts.json
     (또는 뱅크 bank.csv solved) 을 세서 `main/conf/preset_sets/<name>.json` 에 굽는다 (`--build_preset_set`). end shot 절도
     그 코퍼스 캡션에 실제로 있던 preset 에서만 허용한다 (v1 은 shot 사다리 이전이라 없음).
  4. **씬 기하는 생성기 몫이다.** planner 는 `feasibility(plan, conditions) -> report` 를 받으면 실패 사유를 VLM 이 읽는 말
     (`explain_failure`) 로 바꿔 다시 계획한다 (`max_replans`, 기본 0 = 꺼짐).
  5. **plan 은 정규형** — 같은 뜻이면 같은 dict (`to_json` sort_keys). 크기 (`magnitude`) · 속도 (`speed`) 는 plan 에 남기지만
     **지금 카메라 모델은 안 읽는다** (`unconsumed`).

env latentcam. VLM 서버: `screen -dmS vlm bash camera_generation/dataset/exec/serve_qwen3vl.sh` (127.0.0.1:22002).
  python models/intent_planner.py --selftest                                          # 전 preset 왕복 (plan → 문장 == 뱅크 캡션), 서버 불필요
  python models/intent_planner.py --build_preset_set v1 --source /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200
  python models/intent_planner.py --preset_set v1 --video a.mp4 --intent "이 사람을 왼쪽에서 돌면서 다가가줘" --out plan.json
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from os import path

LATENTCAM = path.dirname(path.dirname(path.abspath(__file__)))
REPO = path.dirname(path.dirname(LATENTCAM))
DATASET = path.join(REPO, "camera_generation", "dataset")
if DATASET not in sys.path:
    sys.path.insert(0, DATASET)

import fit.caption.build_bank_captions as BC  # noqa: E402
from lbm.presets import PRESETS, resolve_preset  # noqa: E402

PRESET_SET_DIR = path.join(LATENTCAM, "main", "conf", "preset_sets")
PRESETS_V2 = path.join(path.dirname(BC.PRESETS_DEFAULT), "caption_presets_v2.json")   # [R468] 캡션 v2 문구
# v5f fit `--shot_class_presets` (configs/bank/v5f_pushsubject.json). preset_set 파일이 없을 때 (`all`) 만 쓰는 기본값.
SHOT_CLASS_PRESETS = r"^(track_)?(dolly_in_look_at|push_in_arc_(left|right)|pull_out_arc_(left|right))$"
SHOT_CLASSES = tuple(BC.SHOT_CLASS_TEXT)                       # extreme_wide · wide · medium · close_up · extreme_close_up
SHOT_SUFFIX = re.compile(r", ending on (an? )?(extreme wide shot|wide shot|medium shot|close-up|extreme close-up)( shot)?\.$")
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

# [R468] 캡션 v2 용어 (configs/caption_presets_v2.json, 사용자 10-20): primitive = preset 용어 7 개 (dolly·truck·pedestal·orbit·pan·tilt + track)
#   crane = pedestal + look-at · push in/pull out = dolly in/out · arc = orbit + dolly. roll (deroll 로 정지) · v5f 0 행 preset · free_pedestal (pedestal 과 같은 궤적) 은 어휘에서 뺀다.
PRIMS_V2 = ("dolly_in", "dolly_out", "truck_left", "truck_right", "pedestal_up", "pedestal_down", "orbit_left", "orbit_right",
            "pan_left", "pan_right", "tilt_up", "tilt_down")
SPECIAL_V2 = {**{k: v for k, v in SPECIAL.items() if not k.startswith(("free_pedestal", "static_zoom"))},
              "push_in_arc_left": ("dolly_in", "orbit_left"), "push_in_arc_right": ("dolly_in", "orbit_right"),
              "pull_out_arc_left": ("dolly_out", "orbit_left"), "pull_out_arc_right": ("dolly_out", "orbit_right"),
              "crane_up": ("pedestal_up",), "crane_down": ("pedestal_down",),
              "crane_up_dolly_out": ("pedestal_up", "dolly_out"), "crane_down_dolly_in": ("pedestal_down", "dolly_in")}
EXCLUDE_V2 = re.compile(r"^(track_)?(roll_|s_curve|static_zoom_in|pan_right_zoom_out|free_pedestal_)")
TERMS = {"v1": (PRIMS, SPECIAL), "v2": (PRIMS_V2, SPECIAL_V2)}


# dolly + orbit 은 우리 어휘에서 arc 다 (true_orbit 은 등거리라 전후진과 같이 못 산다) — 사용자 말 "dolly in orbit left" 를 여기로 접는다.
def fold_synonyms(key: frozenset, terms: str = "v1"):
    """dolly 가 같이 있으면 orbit_* → arc_* (3 개 이상 조합의 nearest 제안에도 같은 규칙). v2 는 arc 가 없어 그대로."""
    if terms == "v1" and key & {"dolly_in", "dolly_out"}:
        key = frozenset("arc_" + m.split("_")[1] if m.startswith("orbit_") else m for m in key)
    return key


def moves_of(name: str, terms: str = "v1"):
    """preset 이름 → (track, primitive 튜플). `terms` = v1 (예전) | v2 (R468 캡션 v2 용어)."""
    prims, special = TERMS[terms]
    track = name.startswith("track_")
    base = name[len("track_"):] if track else name
    if name in special or base in special:
        return track, tuple(special.get(name, special.get(base)))
    toks, rest = [], base
    while rest:
        hit = next((p for p in sorted(prims, key=len, reverse=True) if rest == p or rest.startswith(p + "_")), None)
        assert hit, f"preset '{name}' 을 primitive 로 못 푼다 (남은 '{rest}') — SPECIAL 에 추가할 것"
        toks.append(hit); rest = rest[len(hit) + 1:]
    return track, tuple(toks)


# ───────────────────────────────────────────────────────────────── preset 집합 (코퍼스별)
def build_preset_set(name: str, source: str, seg_prefix: str = "dynpose", bank_dir: str = "", out_dir: str = PRESET_SET_DIR):
    """코퍼스 (`<source>/<seg_prefix>/*/da3/prompts.json`) 또는 뱅크 (`<source>/*/<bank_dir>/bank.csv` 의 solved 행) 에서
    preset 개수와 end shot 절이 붙은 preset 을 세어 `<out_dir>/<name>.json` 에 쓴다. 카메라 모델이 **실제로 본** 어휘다."""
    cnt, shot = Counter(), Counter()
    if bank_dir:
        for f in glob.glob(path.join(source, "*", bank_dir, "bank.csv")):
            for r in csv.DictReader(open(f)):
                if r["status"].startswith("solved"):
                    n = resolve_preset(r["preset"], r.get("aim")); cnt[n] += 1
                    shot[n] += bool(r.get("shot_target_class"))
    else:
        for f in glob.glob(path.join(source, seg_prefix, "*", "da3", "prompts.json")):
            for v in json.load(open(f)).values():
                n = resolve_preset(v["preset"], v.get("aim")); cnt[n] += 1
                c = v.get("prompt_camera_with_scene_video")
                shot[n] += bool(SHOT_SUFFIX.search(c.get("concise", "") if isinstance(c, dict) else str(c or "")))
    assert cnt, f"preset 을 하나도 못 셌다: {source} (seg_prefix={seg_prefix}, bank_dir={bank_dir or '-'})"
    out = {"name": name, "source": source, "bank_dir": bank_dir or None, "seg_prefix": None if bank_dir else seg_prefix,
           "n_rows": sum(cnt.values()), "presets": dict(cnt.most_common()), "shot_presets": sorted(k for k, v in shot.items() if v)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(path.join(out_dir, f"{name}.json"), "w"), indent=1)
    return out


def load_preset_set(spec):
    """`all` | 집합 이름 (`v1` → main/conf/preset_sets/v1.json) | json 경로 | 쉼표 목록 → (presets 리스트 | None, shot_presets | None)."""
    if spec in (None, "", "all"):
        return None, None
    f = spec if path.isfile(spec) else path.join(PRESET_SET_DIR, f"{spec}.json")
    if path.isfile(f):
        d = json.load(open(f))
        return list(d["presets"]), list(d.get("shot_presets", []))
    return spec.split(","), None


SIMILAR = {"crane_up": "pedestal_up", "pedestal_up": "crane_up", "crane_down": "pedestal_down", "pedestal_down": "crane_down",
           "orbit_left": "arc_left", "arc_left": "orbit_left", "orbit_right": "arc_right", "arc_right": "orbit_right",
           "zoom_in": "dolly_in", "dolly_in": "zoom_in", "zoom_out": "dolly_out", "dolly_out": "zoom_out"}   # [R465] snap 순위용


class PresetVocab:
    """preset 어휘 (lbm.presets.PRESETS + configs/caption_presets.json) 를 `preset_set` 으로 좁힌다."""

    def __init__(self, preset_set=None, terms: str = "v1"):
        self.terms, self.prims = terms, TERMS[terms][0]
        self.cfg = json.load(open(BC.PRESETS_DEFAULT if terms == "v1" else PRESETS_V2))
        allowed, shot_presets = load_preset_set(preset_set) if not isinstance(preset_set, (list, tuple)) else (list(preset_set), None)
        names = [n for n in PRESETS if n not in LEGACY and n in self.cfg["presets"] and not (terms == "v2" and EXCLUDE_V2.match(n))]
        if allowed is not None:
            allowed = [resolve_preset(n) for n in allowed]
            miss = set(allowed) - set(names)
            assert not miss, f"preset_set 에 어휘 밖 preset: {sorted(miss)} (free-moving dd_* 는 아직 미지원)"
            names = [n for n in names if n in set(allowed)]
        self.name = preset_set if isinstance(preset_set, str) else ("custom" if preset_set else "all")
        self.info, self.index = {}, {}
        for n in names:
            track, mv = moves_of(n, terms)
            aim = PRESETS[n][1]
            spec = self.cfg["presets"][n]
            slot = BC.TARGET_SLOT in spec["phrase"]
            self.info[n] = dict(track=track, moves=mv, aim=aim, targetless=bool(spec.get("targetless")),
                                needs_target=track or aim == "look_at" or (slot and not spec.get("phrase_targetless")),
                                shot_ok=(n in shot_presets) if shot_presets is not None else bool(re.match(SHOT_CLASS_PRESETS, n)))
            self.index.setdefault((track, frozenset(mv)), []).append(n)

    def resolve(self, track: bool, moves, look_at: bool, has_target: bool, exclude=(), snap: bool = False):
        """primitive 조합 → (preset, 오류 목록). 같은 조합에 이름이 여럿이면 aim·target 유무로 가른다
        (`dolly_in` free vs `dolly_in_look_at` / `pedestal_up` (target 부름) vs `free_pedestal_up` (targetless)).
        `exclude` = 생성기가 이미 실패한 preset (재계획 루프)."""
        key = fold_synonyms(frozenset(moves), self.terms)
        cands = [c for c in self.index.get((bool(track), key), []) if c not in exclude]
        if not cands:
            pool = [k for k in self.index if any(c not in exclude for c in self.index[k])]
            near = sorted(pool, key=lambda k: (-len(k[1] & key), len(k[1] - key), k[0] != bool(track)))[:4]
            if snap:   # [R465] snap 순위만: 비슷한 primitive (crane≈pedestal · orbit≈arc) 를 반쯤 맞은 것으로 친다 — 오류 문구의 nearest 는 그대로
                sim = lambda k: len(k[1] & key) + 0.5 * len({SIMILAR.get(m) for m in key - k[1]} & set(k[1] - key))
                near = sorted(pool, key=lambda k: (-sim(k), len(k[1] - key) - 0.5 * len({SIMILAR.get(m) for m in key - k[1]} & set(k[1] - key)), k[0] != bool(track)))[:4]
            why = "already failed in this scene" if self.index.get((bool(track), key)) else f"not in the '{self.name}' vocabulary"
            if snap and near:      # [R465] 재질의 소진 뒤 fallback — 가장 가까운 조합 (공통 primitive 많고 · 남는 것 적고 · track 같은 순)
                cands = [c for c in self.index[near[0]] if c not in exclude]
        if not cands:
            return None, [f"no preset for track={bool(track)} moves={sorted(key)} ({why}); nearest: "
                          + ", ".join(self.describe(next(c for c in self.index[k] if c not in exclude)) for k in near)]
        if len(cands) > 1:
            want = "look_at" if look_at else "free"
            by_aim = [c for c in cands if self.info[c]["aim"] == want] or cands
            by_tgt = [c for c in by_aim if self.info[c]["targetless"] != has_target] or by_aim
            cands = by_tgt
        return cands[0], []

    def describe(self, name: str):
        i = self.info[name]
        d = ("track + " if i["track"] else "") + (" + ".join(sorted(i["moves"])) or "hold")
        if self.terms == "v2":   # [R468] v2 는 같은 primitive 에 조준만 다른 preset 이 있다 (pedestal_up / crane_up) — 조준을 같이 적는다
            d += " (keep target centered)" if i["aim"] == "look_at" else ""
        return d

    def prompt_block(self):
        """VLM 에 보여 줄 어휘 — 허용 조합을 primitive 로 적는다 (이름은 안 보인다)."""
        return "\n".join(f"- {r}" for r in sorted({self.describe(n) for n in self.info}))


# ───────────────────────────────────────────────────────────────── plan
@dataclass
class CanonicalCameraPlan:
    """정규형 카메라 계획. `preset` 은 파생값 (moves·track·aim 으로 `PresetVocab.resolve` 가 정한다)."""
    preset: str
    moves: list
    track: bool
    aim: str                                   # look_at | free
    target_noun: str = ""                      # SAM3 후보용 명사 (scene graph label 수준: man, dog, car)
    target_text: str = ""                      # referring expression (학습 캡션 target_text 와 같은 결: 지칭구)
    target_point: list = None                  # [t_sec, x, y] (x,y ∈ [0,1]) — 사용자가 찍은 점. 있으면 vote 대신 이 점으로 subject 를 고른다
    end_shot: str = ""                         # SHOT_CLASSES 중 하나, preset_set 이 shot 절을 가진 preset 만
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


def canonicalize(raw: dict, vocab: PresetVocab, intent: str = "", exclude=(), target_point=None, snap: bool = False):
    """VLM JSON → (CanonicalCameraPlan | None, 오류 목록). 오류는 그대로 VLM 재질의에 쓴다 (lbm.vlm.chat_json validate)."""
    err, warn = [], []
    moves = raw.get("moves") or []
    if isinstance(moves, str):
        moves = [moves]
    if snap:   # [R465] "track + crane_up + dolly_out" 처럼 한 문자열에 조합을 몰아 쓴 응답을 primitive 로 쪼갠다
        moves = [q for m in moves for q in re.split(r"\s*[+,/]\s*|_\+_", str(m)) if q.strip()]
    moves = [str(m).strip().lower().replace(" ", "_").replace("-", "_") for m in moves]
    moves = [m for m in moves if m not in ("hold", "static", "none", "")]
    track = bool(raw.get("track", False))
    if snap:
        track = track or "track" in moves
        moves = [m for m in moves if m in vocab.prims]    # 어휘 밖 primitive 는 버린다
    bad = [m for m in moves if m not in vocab.prims]
    if bad:
        err.append(f"unknown moves {bad}; allowed primitives: {', '.join(vocab.prims)}")
    tgt = raw.get("target") or {}
    noun = str(tgt.get("noun") or "").strip().lower() if isinstance(tgt, dict) else ""
    text = str(tgt.get("text") or "").strip() if isinstance(tgt, dict) else ""
    if noun and not text:
        text = f"the {noun}"
    look_at = bool(raw.get("keep_target_centered", bool(noun)))
    if err:
        return None, err
    preset, e = vocab.resolve(track, moves, look_at, bool(noun), exclude, snap=snap)
    if snap and not e:
        warn.append("snapped (재질의 소진 뒤 가장 가까운 preset)")
    if e:
        return None, e
    info = vocab.info[preset]
    if info["needs_target"] and not noun and snap:
        noun, text = "subject", "the subject"; warn.append("snap: target 없음 → 'the subject'")
    if info["needs_target"] and not noun:
        return None, [f"'{vocab.describe(preset)}' needs a target (track / look-at / phrase names the subject); give target.noun and target.text"]
    if info["targetless"] and noun:
        warn.append(f"'{preset}' 은 targetless — target '{noun}' 을 문장·Molmo2 에서 뺀다")
        noun, text = "", ""
    shot = str(raw.get("end_shot") or "").strip().lower().replace("-", "_").replace(" ", "_")
    shot = {"closeup": "close_up", "close": "close_up", "medium_shot": "medium", "wide_shot": "wide"}.get(shot, shot)
    if shot and shot not in SHOT_CLASSES:
        return None, [f"end_shot '{shot}' not in {list(SHOT_CLASSES)}"]
    if shot and not info["shot_ok"]:
        warn.append(f"end_shot '{shot}' 은 '{preset}' 학습 캡션 ({vocab.name}) 에 없는 절 — 뺀다"); shot = ""
    mag = str(raw.get("magnitude") or "medium").lower(); spd = str(raw.get("speed") or "steady").lower()
    if mag not in MAGNITUDES:
        warn.append(f"magnitude '{mag}' → medium"); mag = "medium"
    if spd not in SPEEDS:
        warn.append(f"speed '{spd}' → steady"); spd = "steady"
    return CanonicalCameraPlan(preset=preset, moves=list(info["moves"]), track=info["track"], aim=info["aim"],
                               target_noun=noun, target_text=text, target_point=list(target_point) if (target_point and noun) else None,
                               end_shot=shot, magnitude=mag, speed=spd, intent=intent,
                               rationale=str(raw.get("rationale") or ""), warnings=warn), []


def plan_to_conditions(plan: CanonicalCameraPlan, vocab: PresetVocab, framing_on_free: bool = False):
    """plan → 카메라 모델 조건. camera_text 는 뱅크 캡션 빌더 기본값 (nl, magnitude off, nl_framing off) 과 같은 경로다.

    `framing_on_free` (R417, 기본 False): look-at 없는 preset (aim=free) 은 "keeping X in view" 절을 뺀다 — 재조준을 안 하는
    궤적이 지킬 수 없는 약속이다. True 면 뱅크 캡션 기본 (D143 변이 축, in_frame 1.0 가정 → 항상 붙음) 과 같다.
    ⚠ v1 학습 캡션은 aim=free 에도 실측 in_frame ≥ 0.85 면 이 절이 붙어 있다 — 그 문장을 본 모델엔 True 가 분포상 더 가깝다.
    """
    variant = {"preset": plan.preset, "aim": plan.aim, "anchor_id": "plan", "anchor_label": plan.target_noun,
               "subject_in_frame": 1.0, "subject_area_med": 0.1}
    if plan.end_shot:
        variant["shot_target_class"] = plan.end_shot
    cfg = vocab.cfg
    spec = cfg["presets"][plan.preset]
    if not plan.target_text and BC.TARGET_SLOT in spec["phrase"] and spec.get("phrase_targetless"):
        cfg = {**cfg, "presets": {**cfg["presets"], plan.preset: {**spec, "phrase": spec["phrase_targetless"], "targetless": True}}}
    cap = BC.caption_of(variant, "", cfg, False, None, None, False, {"plan": plan.target_text} if plan.target_text else {},
                        "nl", framing_on_free, 0.85, None, False)
    camera_text = BC.nl_prompt(cap)
    tt = str(cap.get("target_text") or "").strip()
    if not plan.target_noun:
        query = None
    elif plan.target_point:
        query = {"mode": "point", "point": list(plan.target_point), "noun": plan.target_noun, "text": plan.target_text}
    else:
        query = {"mode": "vote", "noun": plan.target_noun, "text": plan.target_text}   # molmo2_track_lift --comb_rule vote (기본)
    return {"camera_text": camera_text,
            "molmo2_text": f"Track {tt}." if tt else camera_text,        # build_molmo2_track_override 규칙 (--empty_target concise)
            "subject_query": query,
            "needs_subject_cond": bool(plan.target_noun),                 # track3d / box3d 조건을 subject finder 로 만들어야 하는가
            "preset": plan.preset, "aim": plan.aim, "end_shot": plan.end_shot,
            "unconsumed": {"magnitude": plan.magnitude, "speed": plan.speed}}


# ───────────────────────────────────────────────────────────────── 생성기 피드백 (기본 꺼짐)
# 뱅크 fit 의 `status` / `binding` 어휘 (fit_hole_ladder: clamped_low+<binding>, *_limited, shot_missed, arc_blocked …) → VLM 이 읽는 말.
FAIL_TEXT = {
    "collision": ("the camera path would pass through scene geometry (walls, furniture, terrain)",
                  "choose a move that stays in open space seen by the source camera, or a smaller/opposite direction"),
    "occlusion": ("the subject would be hidden behind other objects for much of the shot",
                  "move to a side where the subject is not blocked, or stay closer to the source viewpoint"),
    "obb": ("the camera would get too close to or enter the subject itself",
            "keep more distance from the subject, e.g. pull out instead of push in"),
    "ground": ("the camera would go below the ground or skim the floor", "avoid moving down (pedestal/crane down)"),
    "elev": ("the camera would look from an extreme high or low angle", "avoid large vertical moves"),
    "hole": ("the move would reveal large unseen regions of the scene (nothing was filmed there)",
             "prefer moves that stay near the source viewpoint (smaller arcs/orbits, dolly instead of orbit)"),
    "framing": ("the subject would leave the frame", "use a move that re-aims at the subject (look-at / tracking)"),
    "shot_floor": ("the requested end shot size could not be reached", "ask for a less tight end shot or no end shot"),
    "shot_ceiling": ("the shot would become too tight", "ask for a wider end shot"),
    "shot_dir": ("the shot size would change in the opposite direction of the request", "check push-in vs pull-out"),
    "shot_missed": ("the requested end shot size could not be reached", "ask for a less tight end shot or no end shot"),
    "arc_blocked": ("there is no room to arc around the subject on that side", "try the other side or a straight dolly"),
    "static": ("the move collapses to almost no motion in this scene", "pick a different move"),
}


def explain_failure(report: dict, vocab: PresetVocab):
    """생성기 보고 → VLM 피드백 문단. report = {ok, preset, status, binding?, frames?: [i..], realized?: {...}, note?}."""
    status = str(report.get("status") or ""); binding = str(report.get("binding") or "")
    keys = [k for k in [binding, *re.split(r"[+_]", status), status.split("_limited")[0], status] if k in FAIL_TEXT]
    what, fix = FAIL_TEXT[keys[0]] if keys else (f"the trajectory generator rejected it (status={status or '?'})", "pick a different move")
    move = vocab.describe(report["preset"]) if report.get("preset") in vocab.info else report.get("preset", "?")
    lines = [f"The move '{move}' failed in this scene: {what}."]
    fr = report.get("frames") or []
    if fr:
        lines.append(f"It fails at frames {fr[0]}-{fr[-1]} of {report.get('n_frames', 49)} ({len(fr)} frames).")
    for k, v in (report.get("realized") or {}).items():
        lines.append(f"Realized {k}: {v}.")
    if report.get("note"):
        lines.append(str(report["note"]))
    lines.append(f"Suggestion: {fix}. Do not repeat '{move}'.")
    return " ".join(lines)


# ───────────────────────────────────────────────────────────────── planner
SYSTEM = ("You are a cinematographer planning ONE camera move for re-shooting the given video. "
          "Answer with raw JSON only.")
PROMPT = """The video above is the source shot{dur}.
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
    """(video, intent) → CanonicalCameraPlan. VLM 은 `lbm.vlm.VLMClient` (OpenAI 호환, Qwen3-VL) — `client` 로 갈아끼울 수 있다.

    video_mode  `video` (기본, R417): mp4 를 통째로 싣고 Qwen3-VL 프로세서에 fps=`video_fps` (기본 2.0)  /  `frames`: n_frames 장 이미지
    feasibility `None` (기본) 이면 생성기 루프 없음. callable(plan, conditions) -> report dict ({ok, status, binding, frames, …})
    max_replans 생성기 실패 시 재계획 횟수 (기본 0 = 꺼짐)
    """

    def __init__(self, client=None, preset_set="all", video_mode: str = "video", video_fps: float = 2.0, n_frames: int = 8,
                 max_repairs: int = 3, feasibility=None, max_replans: int = 0, framing_on_free: bool = False,
                 work_dir: str = path.join(REPO, "tmp", "intent_planner"), snap_on_fail: bool = False, terms: str = "v1"):
        if client is None:
            from lbm.vlm import VLMClient
            client = VLMClient(temperature=0.1, max_tokens=512)
        assert video_mode in ("video", "frames"), video_mode
        self.client, self.vocab = client, PresetVocab(preset_set, terms)
        self.video_mode, self.video_fps, self.n_frames = video_mode, video_fps, n_frames
        self.max_repairs, self.feasibility, self.max_replans = max_repairs, feasibility, max_replans
        self.framing_on_free, self.work_dir = framing_on_free, work_dir
        self.snap_on_fail = snap_on_fail        # [R465] 재질의 소진 시 마지막 응답을 가장 가까운 preset 으로 (기본 False = 예전처럼 None)

    def _ask(self, video, intent, feedback=(), exclude=(), target_point=None):
        if self.video_mode == "video":
            images, vkw, dur = (), {"video": video, "video_fps": self.video_fps}, ""
        else:
            images, sec = sample_frames(video, self.n_frames, path.join(self.work_dir, path.splitext(path.basename(video))[0]))
            vkw, dur = {}, f" ({len(images)} frames sampled uniformly from {sec:.1f}s, in order)"
        prompt = PROMPT.format(dur=dur, intent=intent, vocab=self.vocab.prompt_block(), shots=list(SHOT_CLASSES))
        if feedback:
            prompt += "\n\n## GENERATOR_FEEDBACK (previous plans could not be executed in this scene)\n" + "\n".join(f"- {f}" for f in feedback)
        box = {}

        def validate(raw):
            box["last"] = raw
            p, e = canonicalize(raw, self.vocab, intent, exclude, target_point)
            box["plan"] = p
            return e

        raw, info = self.client.chat_json(prompt, images=images, system=SYSTEM, validate=validate, max_repairs=self.max_repairs,
                                          label="intent_plan", echo_previous=False, **vkw)
        if raw is None and self.snap_on_fail and isinstance(box.get("last"), dict):
            p, _ = canonicalize(box["last"], self.vocab, intent, exclude, target_point, snap=True)
            return p, {**info, "raw": box["last"], "snapped": p is not None}
        return (box.get("plan") if raw is not None else None), {**info, "raw": raw}

    def plan(self, video: str, intent: str, target_point=None):
        """-> (plan | None, conditions | None, info). None 이면 재질의 소진 — 호출자가 fallback 을 정한다 (hard-fail 안 함).
        `target_point` = [t_sec, x, y] 사용자 pointing (주면 subject_query.mode = point)."""
        feedback, exclude, history = [], [], []
        for attempt in range(self.max_replans + 1):
            p, info = self._ask(video, intent, feedback, exclude, target_point)
            if p is None:
                return None, None, {**info, "history": history}
            cond = plan_to_conditions(p, self.vocab, self.framing_on_free)
            if self.feasibility is None:
                return p, cond, {**info, "history": history}
            report = {"preset": p.preset, **self.feasibility(p, cond)}
            history.append({"plan": p.to_dict(), "report": report})
            if report.get("ok"):
                return p, cond, {**info, "history": history, "report": report}
            feedback.append(explain_failure(report, self.vocab)); exclude.append(p.preset)
        return None, None, {**info, "history": history, "exhausted_replans": True}

    def __call__(self, video: str, intent: str, target_point=None):
        return self.plan(video, intent, target_point)


# ───────────────────────────────────────────────────────────────── 정답 plan · intent 합성 (평가용, planner 학습 안 함)
def plan_from_bank_row(row: dict, vocab: PresetVocab, target_text: str = ""):
    """뱅크 행 (prompts.json 항목 / bank.csv 행) → plan. 학습 데이터의 정답 plan — intent 합성의 출발점."""
    name = resolve_preset(row["preset"], row.get("aim"))
    info = vocab.info[name]
    noun = "" if info["targetless"] else str(row.get("anchor_label") or "")
    cf = row.get("caption_fields") or {}
    tt = target_text or (cf.get("target_text") if isinstance(cf, dict) else "") or (f"the {noun}" if noun else "")
    return CanonicalCameraPlan(preset=name, moves=list(info["moves"]), track=info["track"], aim=info["aim"], target_noun=noun,
                               target_text=tt if noun else "", end_shot=str(row.get("shot_target_class") or ""))


INTENT_PROMPT = """Write {n} different requests a film director might give to a camera operator, each producing exactly this camera move:
move: {move}{tgt}{shot}
Vary the wording and how explicit it is (from technical to purely about the feeling/goal). Some may be in Korean.
Do not mention frame numbers or this JSON. Return JSON: {{"intents": ["...", ...]}}"""


def synthesize_intents(plan: CanonicalCameraPlan, vocab: PresetVocab, client, n: int = 4):
    """canonical plan → high-level intent 문장 n 개 (text-only VLM). planner 를 학습하지 않고 (plan → intent → planner → plan) 왕복으로 잰다."""
    tgt = f"\nsubject: {plan.target_text}" + (" (the camera follows its motion)" if plan.track else "") if plan.target_noun else "\nno specific subject"
    shot = f"\nends on: {BC.SHOT_CLASS_TEXT[plan.end_shot]}" if plan.end_shot else ""
    raw, _ = client.chat_json(INTENT_PROMPT.format(n=n, move=vocab.describe(plan.preset), tgt=tgt, shot=shot),
                              validate=lambda r: [] if isinstance(r.get("intents"), list) and r["intents"] else ["need non-empty 'intents' list"])
    return [str(x) for x in (raw or {}).get("intents", [])][:n]


# [SG3 · R459] 수준별 intent (사용자 2026-10-20): L1 = 카메라 움직임을 직접 말함 (용어 제한 없음) · L2 = 무엇을 강조/보고 싶은지 ·
#   L3 = 움직임의 패턴·느낌. L2·L3 는 촬영 전문 용어 금지 (jargon_ok=True 면 허용 — 나중 비교용 변형).
JARGON = ["dolly", "truck", "trucking", "pedestal", "crane", "jib", "pan", "panning", "tilt", "tilting", "roll", "orbit", "orbiting", "arc", "arcing",
          "push in", "push-in", "pull out", "pull-out", "pull back", "tracking shot", "tracking", "close-up", "closeup", "medium shot", "wide shot",
          "establishing shot", "extreme", "over-the-shoulder", "pov", "point of view", "rack focus", "zoom", "dutch", "framing", "frame",
          "shot", "camera", "lens", "focal"]
LEVEL_PROMPT = """{video_note}A video shows {scene}. A camera move will be generated for it. The exact move is:
"{caption}" (primitives: {move}{shot}).{context}
Write three requests a person could give to get exactly this move, at three levels:
L1: says directly how the camera should move, in everyday words (any words are fine).
L2: says what to emphasize or where the viewer's attention should go, so that this move follows from it. {rule}
L3: says the kind of motion pattern or feeling wanted (e.g. building tension, revealing, intimacy), so that this move follows from it. {rule}
Each request is one natural sentence a non-expert would say, and must stay consistent with the move (direction, approach vs. retreat, following the subject).
Return JSON: {{"L1": "...", "L2": "...", "L3": "..."}}"""
NO_JARGON = "Do not use any filmmaking or camera terms (no words like camera, shot, zoom, pan, dolly, close-up, frame, orbit, track, angle)."


def jargon_hits(text: str, allow: str = ""):
    """금지어 목록 중 `text` 에 나온 것. `allow` (subject 지칭구 등) 에 든 단어는 뺀다 — "the white truck" 의 truck 은 촬영 용어가 아니다 (R459 실측 456 건)."""
    t = " " + re.sub(r"[^a-z\- ]", " ", text.lower()) + " "
    a = " " + re.sub(r"[^a-z\- ]", " ", allow.lower()) + " "
    return [w for w in JARGON if f" {w} " in t and f" {w} " not in a]


VIDEO_NOTE = ("You are given the source video. The subject the camera move is about is highlighted with a red overlay. "
              "Ground L2 and L3 in what is actually visible in the video (the subject's action, nearby objects, the setting).\n")


CONTEXT_KEYS = (("subject", "Subject"), ("attributes", "Subject details"), ("action", "Subject action"), ("others", "Other things in the scene"),
                ("pattern", "Camera pattern"), ("framing", "Framing"), ("end_shot", "Final framing"))


def context_block(context: dict = None):
    """[R462] 캡션을 만들 때 뽑아 둔 장면 맥락 (instance_desc 의 subject 서술·속성·행동, 다른 노드, preset 패턴, framing) → 프롬프트 줄들. 비면 ""."""
    lines = [f"{name}: {', '.join(v) if isinstance(v, (list, tuple)) else v}" for k, name in CONTEXT_KEYS if (v := (context or {}).get(k))]
    return ("\nWhat is known about the shot:\n" + "\n".join(f"- {l}" for l in lines)) if lines else ""


def synthesize_intents_levels(plan: CanonicalCameraPlan, vocab: PresetVocab, client, caption: str, scene: str = "", jargon_ok: bool = False, tries: int = 3,
                              video: str = None, video_fps: float = 2.0, context: dict = None):
    """canonical plan (+ 학습 캡션 문장) → {"L1","L2","L3"}. L2·L3 에 금지어가 나오면 최대 `tries` 번 다시 생성, 끝내 남으면 `jargon` 에 기록."""
    shot = f"; ends on {BC.SHOT_CLASS_TEXT[plan.end_shot]}" if plan.end_shot else ""
    prompt = LEVEL_PROMPT.format(scene=scene or (plan.target_text or "a scene"), caption=caption, move=vocab.describe(plan.preset), shot=shot,
                                 rule="" if jargon_ok else NO_JARGON, video_note=VIDEO_NOTE if video else "", context=context_block(context))
    vkw = {"video": video, "video_fps": video_fps} if video else {}     # [R462] subject mask overlay 영상 (사용자 10-20 "영상은 당연히 넣어야지")
    out = {}
    for _ in range(tries):
        raw, _ = client.chat_json(prompt, validate=lambda r: [] if all(isinstance(r.get(k), str) and r[k].strip() for k in ("L1", "L2", "L3")) else ["need L1, L2, L3 strings"], **vkw)
        out = {k: str((raw or {}).get(k, "")).strip() for k in ("L1", "L2", "L3")}
        bad = {k: jargon_hits(out[k], allow=scene) for k in ("L2", "L3")} if not jargon_ok else {}
        if not any(bad.values()):
            return {**out, "jargon": {}}
    return {**out, "jargon": bad}


def roundtrip_score(gt: CanonicalCameraPlan, pred: CanonicalCameraPlan):
    """왕복 일치 — preset · primitive 집합 · track · target 명사 · end shot."""
    if pred is None:
        return {"ok": False, "preset": 0, "moves": 0, "track": 0, "noun": 0, "shot": 0}
    s = {"preset": int(gt.preset == pred.preset), "moves": int(sorted(gt.moves) == sorted(pred.moves)),
         "track": int(gt.track == pred.track), "noun": int(gt.target_noun == pred.target_noun), "shot": int(gt.end_shot == pred.end_shot)}
    return {"ok": True, **s}


def selftest_roundtrip(terms: str = "v1"):
    """[R468] ①② 만 (preset → primitive → resolve 왕복 · plan 문장 == 뱅크 캡션 빌더 문장) — terms 별."""
    vocab = PresetVocab(terms=terms); bad = []
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
        if plan_to_conditions(p, vocab, framing_on_free=True)["camera_text"] != ref:
            bad.append((n, "문장 불일치"))
    print(f"selftest_roundtrip {terms}: {len(vocab.info)} preset, 불일치 {len(bad)} {bad[:5]}")
    return bad


def selftest():
    """전 preset: ① 이름 → primitive → resolve 가 같은 이름으로 돌아오나 ② plan 문장 == 뱅크 캡션 빌더 문장 (framing_on_free=True 경로)
    ③ 피드백 루프 (가짜 생성기) 가 실패 preset 을 빼고 다시 고르나."""
    selftest_roundtrip("v2")
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
        got = plan_to_conditions(p, vocab, framing_on_free=True)["camera_text"]
        if got != ref:
            bad.append((n, got, ref))
    p, _ = canonicalize({"moves": ["dolly_in", "orbit_left"], "target": {"noun": "man", "text": "the man"}, "end_shot": "close_up"}, vocab)
    print("dolly_in + orbit_left →", p.preset, "|", plan_to_conditions(p, vocab)["camera_text"])
    p, _ = canonicalize({"moves": ["dolly_out"], "target": {"noun": "man", "text": "the man"}, "keep_target_centered": False}, vocab)
    print("dolly_out (free) framing_on_free False/True →", plan_to_conditions(p, vocab)["camera_text"], "/",
          plan_to_conditions(p, vocab, True)["camera_text"])

    class Fake:                                                           # 1 회차 push_in_arc_left → 생성기 실패 → 2 회차 dolly_in look-at
        def __init__(self):
            self.turn, self.prompts = 0, []

        def chat_json(self, prompt, validate=None, **kw):
            self.prompts.append(prompt)
            raw = [{"moves": ["dolly_in", "orbit_left"], "target": {"noun": "man"}},
                   {"moves": ["dolly_in", "orbit_left"], "target": {"noun": "man"}},
                   {"moves": ["dolly_in"], "target": {"noun": "man"}, "keep_target_centered": True}][min(self.turn, 2)]
            self.turn += 1
            e = validate(raw)
            return (None, {"errors": e}) if e and self.turn > 2 else (raw, {"repairs": 0}) if not e else self.chat_json(prompt, validate, **kw)

    fk = Fake()
    gen = lambda pl, c: {"ok": pl.preset != "push_in_arc_left", "status": "clamped_low+collision", "frames": list(range(18, 33))}
    pl, cond, info = IntentPlanner(client=fk, feasibility=gen, max_replans=2)("x.mp4", "dolly in orbiting left around the man")
    print("replan →", pl.preset if pl else None, "| feedback:", info["history"][0]["report"]["status"], "|",
          explain_failure(info["history"][0]["report"], vocab)[:150])
    ok_loop = pl is not None and pl.preset == "dolly_in_look_at"
    pt = IntentPlanner(client=Fake(), feasibility=None)("x.mp4", "...", target_point=[0.5, 0.4, 0.6])[1]["subject_query"]
    print("pointing →", pt)
    print(f"selftest: {len(vocab.info)} preset, 문장 불일치 {len(bad)}, 재계획 루프 {'ok' if ok_loop else 'FAIL'}")
    for b in bad:
        print("  ", b)
    return not bad and ok_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build_preset_set", default=None, help="집합 이름 (예 v1) — --source 코퍼스/뱅크를 세서 main/conf/preset_sets/<이름>.json")
    ap.add_argument("--source", default=None, help="코퍼스 루트 (<root>/<seg_prefix>/*/da3/prompts.json) 또는 뱅크 출력 루트 (--bank_dir 와 같이)")
    ap.add_argument("--seg_prefix", default="dynpose")
    ap.add_argument("--bank_dir", default="", help="주면 <source>/*/<bank_dir>/bank.csv 의 solved 행을 센다 (코퍼스 export 전 뱅크)")
    ap.add_argument("--preset_set", default="all", help="all | 집합 이름 (main/conf/preset_sets/<이름>.json) | json 경로 | 쉼표 목록")
    ap.add_argument("--video")
    ap.add_argument("--intent")
    ap.add_argument("--target_point", default=None, help="t_sec,x,y (x,y ∈ [0,1]) — 사용자 pointing, 주면 vote 대신")
    ap.add_argument("--video_mode", default="video", choices=["video", "frames"])
    ap.add_argument("--video_fps", type=float, default=2.0)
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--framing_on_free", action="store_true", default=False)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.build_preset_set:
        d = build_preset_set(a.build_preset_set, a.source, a.seg_prefix, a.bank_dir)
        print(f"[preset_set] {d['name']}: {len(d['presets'])} preset · {d['n_rows']} 행 · shot 절 preset {len(d['shot_presets'])}")
        return
    tp = [float(x) for x in a.target_point.split(",")] if a.target_point else None
    plan, cond, info = IntentPlanner(preset_set=a.preset_set, video_mode=a.video_mode, video_fps=a.video_fps, n_frames=a.n_frames,
                                     framing_on_free=a.framing_on_free)(a.video, a.intent, tp)
    res = {"preset_set": a.preset_set, "plan": plan.to_dict() if plan else None, "conditions": cond,
           "repairs": info.get("repairs"), "errors": info.get("errors")}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
