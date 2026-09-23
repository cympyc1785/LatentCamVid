"""Cinematographer 카메라 하나를 **여러 trajectory preset 변형**으로 불린다.

**왜 필요한가.** Director 가 chunk 당 `camera_count: 1` 을 내고 Cinematographer 의 VLM 이 preset
하나만 고른다 (`llm_trajectory_plan.trajectory_choice`). 카메라 궤적 데이터셋으로 쓰려면 같은
chunk 에서 여러 궤적이 나와야 하므로, 이미 확정된 **시작/끝 pose 는 그대로 두고 preset 만 바꾼**
카메라를 복제한다.

**어떻게 LBM 자신의 궤적 합성기를 다시 부르는가.** Stage 3 의
`_build_plan_from_explicit_trajectory` (`VideoEngineer/blender_render_worker.py:922-`) 는
`trajectory_plan.keyframes` 가 **비어 있으면** `build_trajectory_plan(..., preset_override=
preset_name)` 을 불러 preset 모양대로 keyframe 을 직접 만든다. 그래서 복제본은
`trajectory_plan = {"preset_name": <preset>, "keyframes": []}` 만 심으면 된다 — 궤적 계산은
LBM 코드가 한다. 원본 카메라는 VLM 이 고른 keyframe 을 그대로 들고 있으므로 손대지 않는다.

**"가능한 preset" 이 17개가 아니라 8개인 이유 (코드 실측).** `video_runtime.PRESET_NAMES` 는 17개인데
`build_trajectory_plan` (`video_runtime.py:244-296`) 이 이름을 실제로 분기하는 건

    orbit_left_arc / orbit_right_arc           (`startswith("orbit")`)
    pedestal_up  == rise_reveal                (같은 branch)
    pedestal_down == drop_reveal               (같은 branch)
    pan_left / pan_right                       (위치 고정, 회전만)
    s_curve
    static_hold / static_hold_locked / static_subtle_zoom   (위치·회전 전부 고정)

뿐이다. `straight_ease` `push_in_arc` `pull_out_arc` `truck_left` `truck_right` **5개는 branch 가
아예 없어서** 전부 기본 lerp(start→VLM 끝점) 로 떨어진다. 이름만 다르고 궤적은 같다.
`_trajectory_travel_limit` (`blender_render_worker.py:310-323`) 이 preset 별로 0.85~1.05 로 다르긴
하지만 `original_travel > limit` 일 때만 깎는데, 실측 travel 이 0.16~0.35 m 라 한 번도 안 걸린다.
따라서 이름으로 17개를 뽑으면 **같은 궤적이 중복**된다.

그래서 pool 은 모양이 실제로 다른 **8개**다 (각 클래스 대표 1개):

    straight_ease   기본 lerp — push_in_arc / pull_out_arc / truck_left / truck_right 와 동일
    orbit_left_arc  orbit_right_arc
    pedestal_up     (== rise_reveal)       pedestal_down (== drop_reveal)
    pan_left        pan_right              위치 고정, 회전만 (travel limit 0.0)
    s_curve

static 3종은 기본 제외다 (`--include_static` 으로 켠다). 카메라가 아예 안 움직여서 궤적 표본으로
쓸 게 없고, 이름만 보고 뽑으면 조용히 정지 클립이 섞인다.

**preset 만 바꾸면 5개 클립이 거의 같아진다 — 그래서 `end_transform` 도 같이 가른다 (`--vary_end`).**
`build_trajectory_plan` (`video_runtime.py:244-296`) 이 preset 이름으로 주는 섭동이 모든 변형이
공유하는 `start→end` lerp 보다 한 자릿수 작다 (실측):

    공유 lerp travel          median 0.223 m   (25 chunk)
    orbit_*_arc 섭동          mid 1.2도, end 2.2도
    pedestal_up/down          _damped_extent(focus_extent.z, 0.05, 0.035, 0.1) = 3.5~10 cm
    s_curve 횡방향            2.5~5 cm
    pan_left/right            위치 고정 (`_smooth_executable_keyframes:328-366` 이 강제), 회전만

즉 pan 2개를 뺀 나머지는 "같은 직선 + 5 cm 흔들기"다. 진짜 손잡이는 끝점이고, 그건 이미 handoff
안에 있다 — `top_candidates` 는 LBM 자신의 가시성·프레이밍 스코어를 통과한 pose 10~20개다.
start 에서 median 0.44 m / 최대 0.90 m 떨어져 있고 방향 최대각 median 133도. `end_transform` 은
`blender_render_worker.py:274` 가 그대로 읽어 `build_trajectory_plan` 에 넘기므로 JSON 만 고치면
된다.

**start pose 는 전 변형 공통으로 고정한다.** 같은 초기 조건에서 다른 움직임이 나와야 궤적
표본으로 비교가 된다.

**후보를 같은 `lens_mm` 으로 제한하는 이유.** worker 는 두 keyframe 에 **같은 lens** 를 쓴다
(`blender_render_worker.py:897-898`). 45mm 로 스코어된 위치에 24mm 를 얹으면 피사체가 화면에서
작아져 가시성 검사에 걸리고, worker 가 `motion_scale` 0.75/0.5/0.35/0.2 사다리로 구제하면
(`:1076-1120`) 조용히 정지 클립이 된다. 같은 lens 후보가 모자란 chunk(망원 45~77mm, 25개 중 6개)는
다른 lens 후보로 채우되 `end_variant_tier: "relaxed"` 로 표시한다.

**실행 순서.** `retime_camera_handoff.py` **다음에** 돌린다. 복제본이 원본의
`target_frame_count` 를 물려받으므로 49 로 맞춘 뒤여야 한다. 복제본은 keyframe 이 비어 있어서
retime 이 다시 손댈 것도 없다.

**Stage 3 는 `shots[].cameras` 만 읽는다** (`blender_render_worker.py:1208,1227` — 최상위
`cameras` 리스트는 안 쓴다). 둘 다 불려 놓되 렌더에 반영되는 건 전자다.

사용 예시:

    # 어떤 preset 이 붙는지만 보고 파일은 안 고친다
    python fit/bank/expand_preset_variants.py \\
      --output_root ../Look-Before-Move/Cinematographer/output \\
      --run_glob 'trumans_c49_w*' --num_presets 4 --dry_run

    # 실제로 고친다 (원본은 <파일>.variants.bak 로 남는다)
    python fit/bank/expand_preset_variants.py \\
      --output_root ../Look-Before-Move/Cinematographer/output \\
      --run_glob 'trumans_c49_w*' --num_presets 4 --no_dry_run
"""

from argparse import ArgumentParser
from collections import Counter
from copy import deepcopy
from glob import glob
from os import path
from random import Random
from shutil import copyfile
import json

HANDOFF_NAME = "camera_handoff_v1.json"

#    모양이 실제로 다른 8개. 괄호 안은 같은 branch 로 떨어져 궤적이 동일한 별칭들 — pool 에서
#    빼는 이유가 "쓸모없어서"가 아니라 "이미 대표가 들어있어서"임을 남겨 둔다.
#    **여기 이름은 우리 어휘가 아니라 원본 LBM `video_runtime.PRESET_NAMES` 로 나가는 문자열이다**
#    — `trajectory_plan.preset_name` 에 심으면 LBM 의 `build_trajectory_plan` 이 읽는다. 그래서
#    D75/D76 rename 대상이 아니다 (우리가 이름을 바꿔도 LBM 은 모른다). 아래 값은 D75 때 우리
#    어휘로 바뀐 채로 두었다 — `dolly_in_aimed`/`static_hold_aimed` 는 LBM 에 없는 이름이라
#    `build_trajectory_plan` 의 기본 lerp 로 떨어진다. lerp 대표인 shape 쪽은 의도한 동작이지만
#    static 3종은 정지 branch 를 못 타므로 **정지가 아니게 된다** (미확인, 별도 과제).
SHAPE_PRESETS = [
    "dolly_in_aimed",   # == push_in_arc_* == pull_out_arc_* == truck_left == truck_right (기본 lerp)
    "orbit_left",
    "orbit_right",
    "pedestal_up",      # == crane_up
    "pedestal_down",    # == crane_down
    "pan_left",         # 위치 고정, 회전만
    "pan_right",        # 위치 고정, 회전만
    "s_curve",
]
STATIC_PRESETS = ["static_hold_aimed", "static_hold_locked", "static_zoom_in"]

#    이 이름으로 오면 대표 이름과 같은 클래스다. 원본 VLM 선택을 pool 에서 뺄 때 쓴다.
#    D75/D76 이름 정리 전 뱅크도 그대로 읽어야 하므로 **옛 이름 키를 지우지 않고 새 이름을 더한다**
#    (이 스크립트는 `lbm.presets` 를 import 하지 않는 독립 도구라 alias 를 못 쓴다).
#    `dolly_in`/`dolly_out` 은 D75(=locked) 와 D76(=aimed) 에서 뜻이 다르지만 **둘 다 lerp 클래스**
#    라 여기서는 갈라볼 필요가 없다 — 클래스 판정에만 쓰는 표다.
PRESET_CLASS = {
    "straight_ease": "lerp", "push_in_arc": "lerp", "pull_out_arc": "lerp",
    "dolly_in": "lerp", "dolly_out": "lerp",
    "dolly_in_dont_look": "lerp", "dolly_out_dont_look": "lerp",
    "dolly_in_aimed": "lerp", "dolly_out_aimed": "lerp",
    "dolly_in_locked": "lerp", "dolly_out_locked": "lerp",
    "push_in_arc_left": "lerp", "push_in_arc_right": "lerp",
    "pull_out_arc_left": "lerp", "pull_out_arc_right": "lerp",
    "truck_left": "lerp", "truck_right": "lerp",
    "orbit_left_arc": "orbit_left", "orbit_right_arc": "orbit_right",
    "orbit_left": "orbit_left", "orbit_right": "orbit_right",
    "pedestal_up": "up", "rise_reveal": "up", "crane_up": "up",
    "pedestal_down": "down", "drop_reveal": "down", "crane_down": "down",
    "pan_left": "pan_left", "pan_right": "pan_right",
    "s_curve": "s_curve",
    "static_hold": "static", "static_hold_locked": "static", "static_subtle_zoom": "static",
    "static_hold_aimed": "static", "static_zoom_in": "static",
    "static_hold_dont_look": "static",
}


def distance(a, b) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def end_candidate_pool(base: dict, minimum_travel: float):
    """`top_candidates` 를 끝점 후보로 쓸 수 있게 두 tier 로 나눈다.

    tier `same_lens` 가 1순위다 (§docstring — worker 가 lens 를 하나만 쓰므로 프레이밍이 어긋나면
    가시성 사다리에 걸려 정지 클립이 된다). 모자라면 `relaxed` 로 채우고 JSON 에 표시한다.
    start 에 너무 가까운 후보는 뺀다 — 그건 정지 궤적이지 변형이 아니다.
    """
    start = base.get("start_transform") or {}
    start_location = start.get("location") or [0.0, 0.0, 0.0]
    lens = base.get("lens_mm")
    same, relaxed = [], []
    for candidate in base.get("top_candidates") or []:
        location = candidate.get("location")
        if not location or distance(location, start_location) < minimum_travel:
            continue
        if lens is not None and abs(float(candidate.get("lens_mm") or -1.0) - float(lens)) < 1e-6:
            same.append(candidate)
        else:
            relaxed.append(candidate)
    #    start 에서 먼 것부터로 정렬해 순서를 못박는다 (JSON 배열 순서에 의존하지 않게).
    #    실제 배정은 호출부에서 tier 안을 섞어서 한다 — 항상 최대 이동만 쓰면 방향이 한쪽으로 쏠린다.
    same.sort(key=lambda c: -distance(c["location"], start_location))
    relaxed.sort(key=lambda c: -distance(c["location"], start_location))
    return same, relaxed


def make_variant(base: dict, preset: str, index: int, end_candidate: dict | None, tier: str):
    """원본 카메라를 preset (+ 선택적으로 끝점) 만 바꿔 복제한다.

    `trajectory_keyframes` 를 지우는 이유: `_build_plan_from_explicit_trajectory` 가 우선이라
    실제로는 안 읽히지만, 남겨두면 나중에 이 JSON 을 보는 사람이 "이게 이 카메라의 궤적"이라고
    오해한다. 원본 궤적은 base 카메라에 그대로 있다.
    """
    variant = deepcopy(base)
    variant["camera_name"] = f"{base.get('camera_name')}__v{index:02d}_{preset}"
    variant["trajectory_plan"] = {
        "schema_version": "plan_a.video_trajectory.v1",
        "preset_name": preset,
        "keyframes": [],                                  # ← 비우면 LBM 이 preset 모양대로 만든다
        "selection_reason": "preset_variant_expansion",
    }
    variant.pop("trajectory_keyframes", None)
    variant["preset_variant"] = preset
    variant["preset_variant_source"] = str(base.get("camera_name") or "")
    if end_candidate is not None:
        start = base.get("start_transform") or {}
        #    `framing` 은 후보에 없다. start 것을 물려줘야 worker 의 framing 합성
        #    (`used_framing_synthesis`, blender_render_worker.py:279,299) 이 안 켜진다.
        variant["end_transform"] = {
            "location": list(end_candidate["location"]),
            "rotation_euler": list(end_candidate.get("rotation_euler") or []),
            "framing": deepcopy(start.get("framing") or {}),
            "lens_mm": base.get("lens_mm"),
            "target": list(end_candidate.get("target") or start.get("target") or []),
        }
        variant["end_variant_candidate_id"] = str(end_candidate.get("candidate_id") or "")
        variant["end_variant_tier"] = tier
        variant["end_variant_travel_m"] = round(
            distance(end_candidate["location"], start.get("location") or [0, 0, 0]), 4)
    else:
        variant["end_variant_tier"] = "authored"
    return variant


def expand_camera_list(cameras: list, pool: list, count: int, seed: int, run: str,
                       tally: Counter, tier_tally: Counter, vary_end: bool, minimum_travel: float):
    """카메라 리스트를 원본 + 변형 N개로 불린 새 리스트를 돌려준다."""
    expanded = []
    for base in cameras:
        name = str(base.get("camera_name") or "")
        authored = str((base.get("trajectory_plan") or {}).get("preset_name") or "")
        #    원본이 이미 쓰는 클래스는 pool 에서 뺀다 — 같은 궤적을 두 번 렌더하지 않기 위해서.
        authored_class = PRESET_CLASS.get(authored, "")
        choices = [p for p in pool if PRESET_CLASS.get(p, p) != authored_class]
        #    run/카메라 이름으로 시드를 고정한다. 같은 입력이면 몇 번 돌려도 같은 preset 이 나온다.
        random = Random(f"{run}|{name}|{seed}")
        picks = sorted(random.sample(choices, min(count, len(choices))))
        #    끝점 후보. same_lens 를 먼저 소진하고 모자라면 relaxed, 그것도 없으면 authored 끝점.
        ends: list = []
        if vary_end:
            same, relaxed = end_candidate_pool(base, minimum_travel)
            #    tier 안에서 순서를 섞는다 — 항상 "제일 먼 것"만 쓰면 chunk 전체가 같은 방향으로 쏠린다.
            random.shuffle(same)
            random.shuffle(relaxed)
            ends = [(c, "same_lens") for c in same] + [(c, "relaxed") for c in relaxed]
        base = deepcopy(base)
        base["preset_variant"] = authored or "authored"
        base["preset_variant_source"] = ""
        base["end_variant_tier"] = "authored"
        expanded.append(base)
        tally[authored or "authored"] += 1
        tier_tally["authored"] += 1
        for index, preset in enumerate(picks, start=1):
            candidate, tier = ends[index - 1] if index - 1 < len(ends) else (None, "authored")
            expanded.append(make_variant(base, preset, index, candidate, tier))
            tally[preset] += 1
            tier_tally[tier] += 1
    return expanded


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_root", required=True, type=str)   # Cinematographer/output
    parser.add_argument("--run_glob", default="trumans_c49_w*", type=str)
    # 원본(VLM 선택) 외에 추가로 뽑을 preset 수. 총 카메라는 chunk 당 1 + num_presets 개.
    parser.add_argument("--num_presets", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    # 정지 preset 3종을 pool 에 넣을지. 기본 off — 궤적 표본으로 쓸 게 없다.
    parser.add_argument("--include_static", dest="include_static", action="store_true")
    parser.add_argument("--no_include_static", dest="include_static", action="store_false")
    parser.set_defaults(include_static=False)
    # 변형마다 `top_candidates` 에서 다른 끝점을 뽑을지. off 면 preset 이름만 바뀌어 궤적이 거의 같다.
    parser.add_argument("--vary_end", dest="vary_end", action="store_true")
    parser.add_argument("--no_vary_end", dest="vary_end", action="store_false")
    parser.set_defaults(vary_end=True)
    # 끝점 후보가 start 에서 이만큼(미터)은 떨어져 있어야 한다. 그 아래는 정지 궤적이다.
    parser.add_argument("--min_travel_m", default=0.08, type=float)
    parser.add_argument("--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    parser.add_argument("--backup_suffix", default=".variants.bak", type=str)
    args = parser.parse_args()

    pool = list(SHAPE_PRESETS) + (STATIC_PRESETS if args.include_static else [])
    runs = sorted(glob(path.join(args.output_root, args.run_glob)))
    tally, tier_tally, touched, skipped, per_run = Counter(), Counter(), [], 0, []
    for run in runs:
        handoff = path.join(run, "outputs", HANDOFF_NAME)
        if not path.exists(handoff):
            continue
        data = json.load(open(handoff, encoding="utf-8"))
        name = path.basename(run)
        #    이미 불려 놓은 파일을 또 불리면 변형의 변형이 생긴다. `preset_variant_source` 로 막는다.
        if any(c.get("preset_variant_source") for c in (data.get("cameras") or [])):
            skipped += 1
            continue
        before = len(data.get("cameras") or [])
        data["cameras"] = expand_camera_list(
            data.get("cameras") or [], pool, args.num_presets, args.seed, name,
            tally, tier_tally, args.vary_end, args.min_travel_m)
        for shot in data.get("shots") or []:
            #    Stage 3 가 실제로 읽는 건 이쪽이다. tally 는 위에서 이미 셌으므로 버린다.
            shot["cameras"] = expand_camera_list(
                shot.get("cameras") or [], pool, args.num_presets, args.seed, name,
                Counter(), Counter(), args.vary_end, args.min_travel_m)
        after = len(data["cameras"])
        per_run.append((name, before, after,
                        [f"{c.get('preset_variant')}"
                         f"{'' if c.get('end_variant_tier') in (None, 'authored') else '@' + str(c.get('end_variant_travel_m'))}"
                         for c in data["cameras"]]))
        touched.append(handoff)
        if args.dry_run:
            continue
        backup = handoff + args.backup_suffix
        if not path.exists(backup):
            copyfile(handoff, backup)
        with open(handoff, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    print(f"{'runs':22s} {len(runs)}")
    print(f"{'dry_run':22s} {args.dry_run}")
    print(f"{'pool':22s} {len(pool)}  {' '.join(pool)}")
    print(f"{'변형 수 / 카메라':22s} {args.num_presets}  (seed {args.seed})")
    print(f"{'끝점도 가름':22s} {args.vary_end}  (min_travel {args.min_travel_m} m)")
    print(f"{'고친 파일':22s} {len(touched)}")
    print(f"{'이미 불림 (건너뜀)':22s} {skipped}")
    print(f"{'카메라 총계':22s} {sum(tally.values())}")
    print()
    print(f"  {'preset':20s} {'count':>6s}")
    for preset, count in sorted(tally.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {preset:20s} {count:6d}")
    print()
    print(f"  {'끝점 tier':20s} {'count':>6s}")
    for tier, count in sorted(tier_tally.items(), key=lambda item: (-item[1], item[0])):
        #    `authored` 가 변형에 붙으면 끝점 후보가 모자랐다는 뜻 — 그 변형은 preset 모양만 다르다.
        print(f"  {tier:20s} {count:6d}")
    if per_run:
        print()
        for name, before, after, presets in per_run[:6]:
            print(f"  {name[:30]:30s} {before} -> {after}  {' '.join(str(p) for p in presets)}")
        if len(per_run) > 6:
            print(f"  ... {len(per_run) - 6} runs 더")


if __name__ == "__main__":
    main()
