"""hole 뱅크 전량 → **하나의** `canonical.json` (변이당 카메라 태그 1개).

왜: `decode/emit.py` 는 **결정 하나**를 canonical 로 바꾼다. 그런데 우리가 실제로 만든 건
`fit_hole_ladder.py` 가 푼 **뱅크**다 (camel 336 / avocado-slice 392 변이). 사용자가 원한 게
"소스 영상 카메라 움직임"이 아니라 **다양한 카메라 움직임의 augmentation** 이므로, 하류로
넘어가야 하는 단위도 결정 하나가 아니라 뱅크 전체다. 그 사이가 비어 있었다.

다행히 이을 자리는 이미 나 있었다 — `canonical.json` 의 `cameras` 는 처음부터 **태그 → 궤적
dict** 이고, `emit_model_cams.py` 는 `--cameras all` 로 전 태그를 한 번에 돈다
(`emit_model_cams.py:394,423` 이 `cams[name]['rel_c2w']` 만 읽는다). 그래서 이 스크립트는 새
포맷을 만들지 않는다. `decode/emit.py: build_canonical()` 을 변이마다 부르고 `cameras` 만
합친다 — canonical 규약(`rel[0]=I`, 단위 스케일, 21 index pick)의 유일한 구현은 계속 거기다.

## 왜 poses.npz 를 그냥 안 쓰고 다시 푸나

뱅크의 `poses.npz` 에는 `cam_c2w` 만 있다. `build_canonical` 은 그 외에 `look_at` ·
`subject_track` · `tau` · `info`(preset/aim/speed/tracking/tau/roll…) 를 요구하고, 이것들은
저장돼 있지 않다. 그래서 `bank.json` 의 변이 파라미터로 `build_poses` 를 **다시 돌려** 그
부산물을 얻는다. 렌더가 0회라 (순수 기하) 뱅크 전량이 수 초다.

다시 푸는 이상 **재현이 맞는지 증명해야 한다**: 재구성한 `cam_c2w` 를 저장된 `poses.npz` 와
프레임 단위로 대조하고, 어긋나면 멈춘다 (`--pose_tol`, 기본 1e-9). 이게 없으면 `fixed` 블록에
빠진 인자 하나(예전 뱅크는 `orbit_span_frac`/`aim_ramp_frames` 를 안 실었다) 때문에 **조용히
다른 궤적**이 emit 되고, 숫자는 전부 그럴듯해서 육안으로 안 잡힌다.

## `--keyframe_ease` — 회전 스케줄만 갈아끼우는 리타이밍 (D87)

D94 까지 배포된 뱅크(vista 51편 · dynpose 266편 · trumans)는 전부 `smoothstep` 으로 구워졌다.
D96 부터는 **굽는 쪽 기본값이 `smooth_kf`** 다 (`fit_hole_ladder` · `sample_camera_bank` ·
샤드 러너 4종). 여기 `FIXED_FALLBACK` 은 그대로 `smoothstep` 인데, 그건 **키가 없는 옛 뱅크를
되만들 때의 폴백**이라 영원히 바뀌면 안 되기 때문이다 — 새 뱅크는 `fixed.keyframe_ease` 에
값을 명시적으로 싣고 나오므로 폴백을 안 탄다. smoothstep 은 ease 를 **keyframe
구간마다 독립으로** 걸어서 각속도가 keyframe 마다 0 근처로 떨어졌다 구간 중앙에서 최대가 된다
— 이동은 preset 이 정한 등속인데 회전만 keyframe 수만큼 펌핑한다. TRUMANS D77 코퍼스 실측:
프레임간 회전각 max/median **median 3.71× / p90 13.35×**, keyframe 각속도가 최대치의 **7.5%**.
"회전이 slerp 가 안 된 것처럼 보인다"의 정체가 이것이다.

이 플래그를 주면 `fixed["keyframe_ease"]` 를 덮어써서 `arclen_kf`(누적 호길이 PCHIP) 등으로
다시 푼다. **뱅크를 다시 fit 하지 않는다** — 손잡이·keyframe 회전·위치는 그대로고 회전이
프레임에 배분되는 방식만 바뀐다. 그래서 재현 검증도 회전을 뺀 **위치 전용**으로 자동 전환된다
(회전은 달라지는 게 목적이므로 1e-9 로 묶으면 무조건 죽는다). 위치가 여전히 bit 단위로 맞아야
"리타이밍만 했다"가 증명된다. 안 주면 예전과 완전히 같은 경로다.

주의: hole/가림 게이트는 회전에 의존하므로 뱅크에 기록된 `measured_hole` /
`subject_in_frame` 은 리타이밍 후 값과 **미세하게 다르다**. keyframe 에서는 동일하고 구간
중간에서만 갈린다. 정확한 값이 필요하면 `fit_hole_ladder` 를 새 ease 로 다시 돌릴 것.

## 태그 이름이 곧 계약이다

태그 = `variant_id` = `<anchor>__<preset>__hole<target>` 그대로 쓴다. 하류에서 결과를 보고
"어느 anchor 의 어느 preset 이 어느 hole 단에서 깨졌나"를 파일명만으로 되짚을 수 있어야 한다.
`--tag_prefix` 로 앞에 영상 이름 등을 붙일 수 있다 (여러 영상의 canonical 을 한 디렉토리에
모을 때).

## 안 거른다 (기록만)

접힌 단(같은 anchor·preset 의 여러 hole 단이 같은 궤적)도, `path_len_u ≈ 0` 인 정지 변이도
**기본적으로 전부 내보낸다**. 뱅크는 재고 목록이고 거르는 건 소비자 몫이라는 게 D39/D45 에서
정한 규칙이다. 대신 요약표와 manifest 에 `folded` / `translation_degenerate` 수를 찍는다.
정말 빼고 싶으면 `--drop_folded` / `--min_path_len` / `--status` / `--anchors` / `--presets`.

출력:
    <out>/<video>/<bank_dir>/canonical/canonical.json    태그 N개짜리 canonical
    <out>/<video>/<bank_dir>/canonical/canonical.npz     rel/rel_full/rmax 스택 + 태그 순서
    <out>/<video>/<bank_dir>/canonical/manifest.json     변이별 요약 + 재현 오차 + emit 명령

env: `da3` / `vista4d` 아무거나 (GPU 안 쓴다 — 렌더 0회)

예시:
    python scripts/emit_bank.py --video camel
    python scripts/emit_bank.py --video camel --status solved --presets orbit_left_arc,truck_left
    python scripts/emit_bank.py --video avocado-slice --bank_dir hole_bank_r80_g5abs --drop_folded
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import build_poses                                      # noqa: E402
from decode.emit import MODEL_NOTES, build_canonical, roundtrip_error           # noqa: E402
from lbm.presets import DEFAULT_SHAPE, row_preset                               # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402
from scripts.fit_hole_ladder import (FIT_TAU_MAX_SCALE, KNOB_RANGE,             # noqa: E402
                                     SHAPE_DEFAULTS, retime_info, truncate_hold)
from scripts.sample_camera_bank import make_decision, register_external         # noqa: E402

# 예전 뱅크(`fixed` 에 이 키들이 없던 시절)를 읽을 때의 기본값. `fit_hole_ladder` 의 CLI 기본값
# **그 자체**를 가져온다 — 따로 적었다가 `min_sweep_deg` 를 30 으로 잘못 써서 dyn_1 궤적이
# 1.6e-2 어긋난 적이 있다. 상수를 복제하지 말 것.
FIXED_FALLBACK = SHAPE_DEFAULTS

# `recover_knob` 이 반올림 구간을 훑는 격자 수 (양쪽 각각). 40 이면 해상도 1.25e-7 로,
# 실측 사례(경계까지 1.8e-6)보다 한 자릿수 촘촘하다.
RECOVER_STEPS = 40

# D188-d. "이 씬은 카메라 0개" 를 크래시와 가르는 기계 토큰. `run_bank.py` 의 emit 단계가 이걸
# 보고 `skipped.json(no_emittable_variants)` 로 닫는다 — 안 닫으면 `chain_bank_rounds` 가 그 씬을
# **라운드마다 다시 집는다** (실측 2026-09-13: 새 라운드가 집은 50편 중 49편이 옛 emit 실패의
# 재탕이었다). `lbm/cloud.EMPTY_DYNMASK_TOKEN` 과 같은 처방.
NO_EMITTABLE_TOKEN = "[NO_EMITTABLE_VARIANTS]"


def decision_from_variant(graph: dict, node: dict, variant: dict, fixed: dict):
    """뱅크 행 하나 → `lbm_decision_v1`. `fit_hole_ladder.decision_at` 의 역함수다.

    손잡이 종류에 따라 갈리는 부분(`pan_deg` 는 `shape` 로, `tau` 는 `target_tau` 로)을 그대로
    복제한다. 여기서 갈라지면 재현이 깨지므로 `decision_at` 을 고칠 때 이 함수도 같이 고칠 것 —
    그래서 pose 대조 assert 가 기본 on 이다.
    """
    kind = variant["knob_kind"]
    # `knob` 은 표시용 5자리 반올림이다. `fit_tau` 이분법이 스케일을 계단으로 양자화하므로
    # 손잡이→궤적이 불연속이고, 반올림이 계단 경계를 넘으면 궤적이 한 칸 커진 채로 나온다.
    # 참값은 `knob_raw` 에 있다 (그게 없는 예전 뱅크는 `recover_knob` 이 되찾는다).
    knob = float(variant.get("knob_raw", variant["knob"]))
    mult = float(variant.get("shape_mult", 1.0))
    if kind == "pan_deg":
        shape = {"pan_deg": knob / FIT_TAU_MAX_SCALE}
        target_tau = KNOB_RANGE["tau"][1]
    else:
        shape = None if mult == 1.0 else {"dolly_frac": DEFAULT_SHAPE["dolly_frac"] * mult,
                                          "lateral_frac": DEFAULT_SHAPE["lateral_frac"] * mult}
        target_tau = knob
    # 변이 행이 speed/tracking/look_at_bias 를 자기 값으로 들고 있다 (뱅크 전체 고정이지만
    # 행에도 찍혀 있다). 행 값을 우선한다 — 나중에 행마다 다르게 굴려도 이 코드가 안 깨진다.
    return make_decision(graph, node, variant["preset"], target_tau,
                         variant.get("speed", fixed["speed"]),
                         variant.get("tracking", fixed["tracking"]),
                         float(variant.get("look_at_bias", fixed["look_at_bias"])),
                         fixed["start_mode"], shape=shape,
                         # D53. `fit_tau` 이분법 해상도까지 재현해야 같은 궤적이 나온다. 행에
                         # 없는 예전 뱅크는 False — 그때는 실제로 안 켰던 것이라 맞다.
                         tau_refine=bool(variant.get("tau_refine", False)),
                         # D94. **행이 기록한 `aim` 을 그대로 넘긴다.** 이게 없으면
                         # `resolve_aim(preset, None)` 이 preset 의 *현재* 기본값을 타서, 뜻이
                         # 뒤집힌 이름(`static_hold`: D93 까지 look_at / D94 부터 free,
                         # D90 의 `truck_left` 등도 같음)이 조용히 다른 카메라로 디코드된다.
                         # 실측: 이걸 안 넘기면 `dyn_0__static_hold__hole0.1` 의 pose 재현
                         # 오차가 6.574e-01 (넘기면 0.000e+00).
                         aim=variant.get("aim"),
                         # D97. **`fixed` 의 요청값**을 쓴다 (행의 `tau_ref` 는 resolve 된
                         # 값이라 재현용이 아니다 — `build_poses` 가 preset 을 보고 다시
                         # resolve 하므로 둘을 겹쳐 넣으면 preset 별칭이 섞였을 때 갈린다).
                         # 키가 없는 예전 뱅크는 `FIXED_FALLBACK` = `"source"` = D97 이전 동작.
                         tau_ref=str(fixed.get("tau_ref", FIXED_FALLBACK["tau_ref"])))


def span_frac_for(node: dict, fixed: dict):
    """orbit sweep 하한. `fit_hole_ladder.main` 의 `span_frac` 계산과 같은 식이어야 한다."""
    orbit = float(fixed.get("orbit_span_frac", FIXED_FALLBACK["orbit_span_frac"]))
    min_sweep = float(fixed.get("min_sweep_deg", FIXED_FALLBACK["min_sweep_deg"]))
    return max(orbit, min_sweep / max(float(node["obs_az_span_deg"]), 1e-6))


def recover_knob(variant: dict, rebuild, reference, tol: float, error=None):
    """`knob_raw` 가 없는 예전 뱅크에서 **참 손잡이를 되찾는다**.

    `knob` 은 5자리 반올림이라 참값은 `knob ± 5e-6` 안에 있다. 그 구간 안에서 `fit_tau` 의
    스케일 계단이 갈리면 재현이 깨진다 (실측: snow-dog `stat_0__truck_right__hole0.1`,
    참값 0.228128185878 이 0.22813 으로 올라가 궤적이 0.43% 커졌다). 계단은 단조라 구간을
    훑으면 반드시 맞는 쪽을 만난다 — **재현이 `tol` 안에 들 때만** 받아들이므로, 못 찾으면
    조용히 다른 궤적을 내보내는 대신 `None` 을 돌려 호출부가 멈추게 한다.

    뱅크를 다시 fit 하는 데 드는 시간(전체 wall 의 ~80%)을 아끼려는 것이고, 새로 만드는
    뱅크는 `knob_raw` 를 실으므로 이 경로를 안 탄다.

    `error` 로 대조 함수를 갈아끼울 수 있다 — D87 리타이밍은 회전이 달라지는 게 목적이라
    위치만 봐야 한다. 안 주면 예전과 같은 전 성분 max abs.
    """
    error = error or (lambda built, ref: float(np.abs(built - ref).max()))
    half = 0.5e-5                                 # `round(x, 5)` 의 반올림 반폭
    base = float(variant["knob"])
    # 계단은 손잡이에 대해 단조라 구간을 균등하게 훑으면 된다. 바깥(경계에 가까운 쪽)부터
    # 좁혀 들어간다 — 반올림이 경계를 넘겼다면 참값은 구간 끝 쪽에 있다.
    for i in range(1, RECOVER_STEPS + 1):
        for offset in (-half * i / RECOVER_STEPS, half * i / RECOVER_STEPS):
            trial = dict(variant, knob_raw=base + offset)
            poses = rebuild(trial)
            if error(poses, reference) <= tol:
                return trial, poses
    return None, None


def rung_of(variant: dict):
    """이 변이가 사다리 몇 단인가. D53 `--hole_mode excess` 뱅크는 `target_hole` 이 anchor 마다
    다르므로(정지 hole + Δ) 단의 정체성은 눈금 Δ 쪽에 있다. `hole_delta` 가 없는 예전 뱅크는
    둘이 같은 값이라 그대로 돌아간다."""
    return float(variant.get("hole_delta", variant["target_hole"]))


REQUIRE_OPS = (">=", "<=", ">", "<")


def parse_require(exprs):
    """`--require "obb_slack>=0"` 목록 -> [(열, 연산자, 문턱)].

    왜 emit 에 이게 필요한가 (D192): `track_look_at` 은 `STATIC_PRESETS` 라 fit 의 이분법을
    통째로 건너뛴다 — 손잡이가 없으니 풀 것도 없다. 그런데 **게이트 판정도 같이 건너뛴다.**
    열은 정상적으로 채워지는데 아무도 안 본다. d185/d183/d179 3,205행 실측:
        obb_slack < 0                11.3%     (카메라가 노드 OBB 안)
        behind_frac > 0              21.7%     (G1 — 표면 뒤)
        subject_visible_frac < 0.6   16.7%     (subject 가 가려짐)
        합집합                        44.6%
    `suspect` 열이 51.5% 를 달지만 그 중 실제 위반은 1,110 뿐이라 덮지도 못하고 과잉이다.
    그래서 뱅크는 그대로 두고 **소비자가 측정 열로 자른다** — `--status`/`--presets` 와 같은
    성격이고, 목록이 비면 조건이 없어 옛 동작과 비트 동일하다.

    열 이름을 고정 인자(`--min_obb_slack` 등)로 안 만드는 이유: 뱅크 CSV 열이 70개가 넘고
    세대마다 느는데, 자를 축은 세대마다 다르다. 하나씩 인자를 파면 그 목록이 emit 의
    argparse 에 영원히 쌓인다.
    """
    out = []
    for raw in (exprs or []):
        expr = raw.strip()
        op = next((o for o in REQUIRE_OPS if o in expr), None)
        assert op, f"--require 는 {'/'.join(REQUIRE_OPS)} 중 하나를 써야 한다: {raw!r}"
        col, _, thr = expr.partition(op)
        col = col.strip()
        assert col, f"--require 열 이름이 비었다: {raw!r}"
        out.append((col, op, float(thr)))
    return out


def passes(variant: dict, col: str, op: str, thr: float):
    """변이의 `col` 이 문턱을 넘는가. 열이 없거나 nan 이면 **통과**시킨다 — 안 잰 것과
    위반한 것은 다르고, 옛 세대 뱅크(열 자체가 없는)를 이 필터로 통째로 비우면 안 된다."""
    if col not in variant:
        return True
    try:
        value = float(variant[col])
    except (TypeError, ValueError):
        return True
    if value != value:                       # nan
        return True
    return (value >= thr if op == ">=" else value <= thr if op == "<="
            else value > thr if op == ">" else value < thr)


def fold_groups(variants: list):
    """(anchor, preset) 별로 손잡이 값이 같은 단들을 묶는다 → 접힌 단 태그 집합.

    충돌 천장이 hole 0.10 단보다 낮으면 4단이 **같은 궤적**이 된다 (D47). 태그는 4개인데
    카메라는 1개라, 하류가 그걸 모르고 4번 생성하면 같은 영상 4장에 GPU 를 쓴다.
    가장 낮은 단 하나만 대표로 남기고 나머지를 `folded` 로 찍는다.
    """
    seen, folded, representative = {}, set(), {}
    for variant in sorted(variants, key=lambda v: (v["anchor_id"], v["preset"],
                                                   rung_of(v))):
        key = (variant["anchor_id"], variant["preset"], round(float(variant["knob"]), 9))
        if key in seen:
            folded.add(variant["variant_id"])
            representative[variant["variant_id"]] = seen[key]
        else:
            seen[key] = variant["variant_id"]
    return folded, representative


def main(args):
    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    video_folder = path.join(out_root, args.video)
    bank_folder = path.join(video_folder, args.bank_dir)
    graph = load_graph(path.join(video_folder, "scene_graph.json"))
    # D67. `fit_hole_ladder` 가 "이 소스로는 만들 카메라가 없다"고 판단하면 `bank.json` 대신
    # `skipped.json` 을 남기고 rc=0 으로 나간다. 그 경우 여기서도 크래시가 아니라 같은 사유를
    # 그대로 전달하고 rc=0 이어야 배치 런너에서 "실패"와 "해당 없음"이 구분된다.
    skipped_path = path.join(bank_folder, "skipped.json")
    if not path.isfile(path.join(bank_folder, "bank.json")) and path.isfile(skipped_path):
        with open(skipped_path, encoding="utf-8") as file:
            skipped = json.load(file)
        print(f"건너뜀 — {skipped.get('reason')} (fit 단계에서 이미 판정)")
        print(f"  {skipped.get('detail')}")
        print(f"  tau_start {skipped.get('tau_start')}  남은 변이 {skipped.get('num_surviving_variants')}")
        return
    with open(path.join(bank_folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    # 외부(DataDoP) 궤적 preset 은 `PRESETS` 에 런타임 등록된 것이라 새 프로세스에는 없다.
    # 뱅크가 스스로 경로를 들고 있으므로 여기서 다시 등록한다 — 안 하면 `build_shape` 가
    # "모르는 preset: dd_*" 로 죽는다. `--external_shapes` 로 덮어쓸 수 있다 (뱅크를 옮겼을 때).
    ext = bank.get("external_shapes") or {}
    ext_path = args.external_shapes or ext.get("path")
    if ext_path:
        register_external(ext_path, args.external_aim or ext.get("aim") or "traj")
    stored = np.load(path.join(bank_folder, "poses.npz"), allow_pickle=True)
    stored_poses = np.asarray(stored["cam_c2w"], dtype=np.float64)
    stored_index = {str(v): i for i, v in enumerate(stored["variant_id"])}

    fixed = dict(bank["fixed"])
    # D87 리타이밍. 회전 스케줄만 갈아끼우므로 대조는 위치 전용으로 내린다 (모듈 docstring).
    ease_from_bank = str(fixed.get("keyframe_ease", FIXED_FALLBACK["keyframe_ease"]))
    retimed = bool(args.keyframe_ease) and args.keyframe_ease != ease_from_bank
    if args.keyframe_ease:
        fixed["keyframe_ease"] = args.keyframe_ease
    # D90. `smooth_kf` 스케줄 인자. 뱅크 값이 기본, CLI 로 준 값이 있으면 그게 이긴다
    # (리타이밍할 때만 의미가 있다 — 뱅크 그대로 emit 하면서 바꾸면 재현 대조에서 걸린다).
    smooth_passes = int(args.smooth_passes if args.smooth_passes is not None
                        else fixed.get("smooth_passes", FIXED_FALLBACK["smooth_passes"]))
    smooth_lambda = float(args.smooth_lambda if args.smooth_lambda is not None
                          else fixed.get("smooth_lambda", FIXED_FALLBACK["smooth_lambda"]))
    # D93. `track_*` 조준 lock. **뱅크가 정한다** — 키가 없으면 D93 이전 뱅크이므로 False 로
    # 되풀어야 그때 만든 궤적이 그대로 나온다 (`fixed` 가 없는 옛 τ 뱅크는 top-level 에 있다).
    preset_tracking = bool(fixed.get("preset_tracking", bank.get("preset_tracking", False)))
    missing = [k for k in FIXED_FALLBACK if k not in fixed]
    num_frames = int(fixed.get("num_frames", bank["num_frames"]))
    nodes = {n["id"]: n for n in graph["nodes"]}
    folded, representative = fold_groups(bank["variants"])

    keep_status = set(args.status.split(",")) if args.status != "all" else None
    keep_anchor = set(args.anchors.split(",")) if args.anchors else None
    keep_preset = set(args.presets.split(",")) if args.presets else None
    keep_hole = {float(h) for h in args.holes.split(",")} if args.holes else None
    require = parse_require(args.require)

    def pose_error(built, reference):
        """재현 오차. 리타이밍 중이면 **위치만** 본다 — 회전은 달라지는 게 목적이다."""
        if retimed:
            return float(np.abs(built[:, :3, 3] - reference[:, :3, 3]).max())
        return float(np.abs(built - reference).max())

    cameras, records, dropped = {}, [], {"filter": 0, "folded": 0, "path_len": 0, "require": 0}
    require_hits = {f"{c}{o}{t:g}": 0 for c, o, t in require}
    rel_stack, rel_full_stack, tags = [], [], []
    world_poses = []                     # --dump_poses 용. bank.json 변이 순서를 그대로 지킨다.
    worst_pose, worst_pose_tag, worst_round = 0.0, "", 0.0
    recovered_knobs = []            # (variant_id, 반올림된 손잡이, 되찾은 손잡이)
    for variant in bank["variants"]:
        vid = variant["variant_id"]
        if ((keep_status and variant["status"] not in keep_status)
                or (keep_anchor and variant["anchor_id"] not in keep_anchor)
                or (keep_preset and variant["preset"] not in keep_preset)
                or (keep_hole and rung_of(variant) not in keep_hole)
                # D188 ③. `fit_hole_ladder --pick_budget` 이 고른 행만. 다른 필터와 같은 성격
                # (뱅크는 안 자르고 소비자가 고른다) 이지만 판정이 **씬 단위**라는 게 다르다 —
                # 사다리가 suspect 때문에 3층을 다 내려가도 여기서 예산 개수로 접힌다.
                or (args.picked_only and not variant.get("picked"))):
            dropped["filter"] += 1
            continue
        if args.drop_folded and vid in folded:
            dropped["folded"] += 1
            continue
        if float(variant["path_len_u"]) < args.min_path_len:
            dropped["path_len"] += 1
            continue
        # D192. 측정 열 문턱. `--require` 가 비면 건너뛰므로 옛 동작과 비트 동일하다.
        failed = next((f"{c}{o}{t:g}" for c, o, t in require if not passes(variant, c, o, t)), None)
        if failed is not None:
            dropped["require"] += 1
            require_hits[failed] += 1
            continue

        node = nodes[variant["anchor_id"]]

        def rebuild(v, _node=node):
            """변이 행 → (decision, poses, extra). 손잡이 복구가 **결정까지** 다시 만들어야
            하므로(뒤의 `build_canonical` 이 decision 을 그대로 싣는다) 셋을 같이 돌려준다."""
            decision = decision_from_variant(graph, _node, v, fixed)
            poses, extra = build_poses(
                decision, graph, board=None, num_frames=num_frames,
                orbit_span_frac=span_frac_for(_node, fixed), start_mode=fixed["start_mode"],
                aim_anchor=fixed["aim_anchor"],
                aim_ramp_frames=int(fixed.get("aim_ramp_frames",
                                              FIXED_FALLBACK["aim_ramp_frames"])),
                traj_basis=str(fixed.get("traj_basis", FIXED_FALLBACK["traj_basis"])),
                aim_keyframes=int(fixed.get("aim_keyframes",
                                            FIXED_FALLBACK["aim_keyframes"])),
                keyframe_aim=str(fixed.get("keyframe_aim", FIXED_FALLBACK["keyframe_aim"])),
                keyframe_ease=str(fixed.get("keyframe_ease", FIXED_FALLBACK["keyframe_ease"])),
                # D90. **뱅크에 적힌 값이 우선**이다. 예전에는 여기서 무조건 자기 CLI 기본값(4)을
                # 썼는데 fit 은 `build_poses` 서명 기본값(12)으로 구워서, `smooth_kf` 뱅크만
                # 회전이 어긋났다 (위치는 정확히 일치해 대조 assert 문구가 원인을 안 가리켰다).
                smooth_passes=smooth_passes, smooth_lambda=smooth_lambda,
                # D93. **뱅크에 적힌 값**을 따른다. 키가 없으면 D93 이전 뱅크라 False —
                # 그래야 옛 뱅크를 되만들 때 궤적이 조용히 안 바뀐다.
                preset_tracking=preset_tracking,
                # D99. 같은 규칙 — **뱅크에 적힌 값**. 키가 없으면 D99 이전 뱅크라 False.
                deroll=bool(fixed.get("deroll", FIXED_FALLBACK["deroll"])),
                # F9. 같은 규칙 — **뱅크에 적힌 값**. 키가 없으면 F9 이전 뱅크라 False
                # (그때 이분법은 SE(3) 로그 전체를 깎았다).
                orbit_fixed_sweep=bool(fixed.get("orbit_fixed_sweep",
                                                 FIXED_FALLBACK["orbit_fixed_sweep"])),
                # D171. 같은 규칙 — **뱅크에 적힌 값**. 키가 없으면 D171 이전 뱅크라 0 =
                # 저역통과 (그때 follow 위치 채널은 언제나 savgol 이었다).
                follow_keyframes=int(fixed.get("follow_keyframes",
                                               FIXED_FALLBACK["follow_keyframes"])),
                follow_kf_interp=str(fixed.get("follow_kf_interp",
                                               FIXED_FALLBACK["follow_kf_interp"])),
                # D150. 같은 규칙 — **뱅크에 적힌 값**. 키가 없으면 D150 이전 뱅크라
                # `z_med_frame0` (그때 τ 의 분모는 언제나 frame0 z-depth 중앙값이었다).
                tau_denom=str(fixed.get("tau_denom", FIXED_FALLBACK["tau_denom"])))
            # F7. 시간축 절단을 **행에 적힌 그대로** 되푼다. 빈 칸/키 없음이면 F7 이전 뱅크
            # (또는 `--no_time_truncate` 로 구운 뱅크)라 `num_frames` 로 떨어지고,
            # `truncate_hold` 가 입력을 그대로 돌려주므로 예전 재현과 비트 단위로 같다.
            # `fixed.time_truncate` 가 아니라 행을 읽는 이유: 절단 여부는 변이마다 다르다.
            hold = v.get("hold_from", "")
            hold = num_frames if hold in ("", None) else int(hold)
            if hold < num_frames:
                poses = truncate_hold(poses, hold)
                # `fit_hole_ladder` 가 뱅크에 적을 때와 **같은 재계산**을 한다 — 안 하면
                # canonical meta 의 `path_len_u`/`tau_max_final` 만 절단 전 값으로 남아
                # 뱅크 CSV 와 어긋난다 (pose 대조 assert 는 이걸 못 잡는다).
                # D150. 그래프에서 다시 읽지 않고 **`build_poses` 가 실제로 쓴 분모**를 쓴다 —
                # `tau_denom=S` 로 구운 뱅크를 예전 분모로 되재면 τ 열만 조용히 다른 게이지가
                # 된다 (pose 대조 assert 는 τ 를 안 본다).
                retime_info(extra, poses,
                            np.asarray(graph["cameras"]["cam_c2w_world"], dtype=float),
                            float(extra["info"]["z_med"]))
            return decision, poses, extra

        reference = stored_poses[stored_index[vid]]
        decision, poses, extra = rebuild(variant)
        error = pose_error(poses, reference)

        # 손잡이 반올림이 `fit_tau` 스케일 계단을 넘긴 경우를 되돌린다. 참값이 있는 뱅크
        # (`knob_raw`)는 애초에 안 어긋나므로 이 경로를 안 탄다.
        if error > args.pose_tol and "knob_raw" not in variant:
            repaired, _ = recover_knob(variant, lambda v: rebuild(v)[1],
                                       reference, args.pose_tol, error=pose_error)
            if repaired is not None:
                decision, poses, extra = rebuild(repaired)
                error = pose_error(poses, reference)
                recovered_knobs.append((vid, float(variant["knob"]),
                                        float(repaired["knob_raw"])))

        # 재현 증명. 여기서 멈추는 게 조용히 다른 궤적을 내보내는 것보다 낫다.
        if error > worst_pose:
            worst_pose, worst_pose_tag = error, vid
        if error > args.pose_tol and args.strict:
            raise SystemExit(
                f"{vid}: 재구성한 궤적이 poses.npz 와 다르다 (최대 {error:.3e} > {args.pose_tol:g}).\n"
                f"  bank.json 의 fixed 블록에 없는 인자: {missing or '없음'}\n"
                f"  손잡이 반올림 복구도 실패했다 (`knob_raw` 열이 있는 뱅크로 다시 fit 하면\n"
                f"  이 문제 자체가 사라진다).\n"
                f"  `decision_from_variant` / `span_frac_for` 가 `fit_hole_ladder` 와 어긋났을 수도\n"
                f"  있다. 확인 전에는 --no_strict 로 넘기지 말 것.")

        tag = f"{args.tag_prefix}{vid}"
        canonical, sidecar, meta = build_canonical(poses, extra, decision, tag,
                                                   n_poses=args.n_poses)
        entry = canonical["cameras"][tag]
        # `notes` 는 전 태그가 같은 dict 라 336번 복제하면 JSON 이 통째로 부풀기만 한다.
        # 최상위에 한 번만 싣는다 (emit_model_cams 는 `rel_c2w` 만 읽으므로 무해).
        entry.pop("notes", None)
        # 뱅크가 실측한 것들 — canonical 만 받은 소비자가 뱅크 CSV 없이도 고를 수 있게.
        entry.update({"variant_id": vid, "anchor_id": variant["anchor_id"],
                      "anchor_label": variant["anchor_label"],
                      "target_hole": float(variant["target_hole"]),
                      "hole_delta": rung_of(variant),
                      "hole_static": variant.get("hole_static"),
                      "measured_hole": float(variant["hole_fraction"]),
                      "knob_kind": variant["knob_kind"], "knob": float(variant["knob"]),
                      "status": variant["status"], "binding": variant["binding"],
                      "subject_in_frame": float(variant["subject_in_frame"]),
                      # 가림 (D81). `--no_subject_visible` 로 만든 뱅크엔 없어서 None 이 된다 —
                      # 하류가 "가림 0" 과 "안 쟀다"를 구분할 수 있어야 하므로 기본값을 안 준다.
                      "subject_visible_frac": variant.get("subject_visible_frac"),
                      "subject_visible_min": variant.get("subject_visible_min"),
                      "obb_slack": variant.get("obb_slack"), "obb_node": variant.get("obb_node"),
                      "behind_frac": float(variant["behind_frac"]),
                      "folded_onto": representative.get(vid)})
        cameras[tag] = entry

        worst_round = max(worst_round, roundtrip_error(sidecar["rel"], sidecar["rel_full"]))
        rel_stack.append(sidecar["rel"])
        rel_full_stack.append(sidecar["rel_full"])
        world_poses.append(np.asarray(poses, dtype=np.float32))
        tags.append(tag)
        # `preset` 은 행이 실제로 만든 카메라의 정식 이름, `preset_raw` 는 뱅크에 적혀 있던 문자열.
        # 둘 다 내보내므로 옛 manifest 와의 대조가 끊기지 않는다 (`lbm.presets.row_preset` docstring).
        records.append({"tag": tag, "variant_id": vid, "preset": row_preset(variant),
                        "preset_raw": variant["preset"],
                        "target_hole": float(variant["target_hole"]),
                        "path_len_u": float(variant["path_len_u"]),
                        "model_gauge_rmax": meta["model_gauge_rmax"], "g": meta["g"],
                        "translation_degenerate": meta["translation_degenerate"],
                        "folded": vid in folded, "pose_rebuild_error": error})

    if not cameras:
        # D188-d. 이 종료는 **크래시가 아니라 "이 씬은 카메라 0개"** 라는 판정이다. 같은 rc=1 을
        # 내는 다른 두 곳(재구성 잔차 초과 / `--dump_poses` 오용)과 구분해야 `run_bank.py` 가
        # 이것만 `skipped.json` 으로 닫을 수 있다. 한국어 문구를 매칭하면 문구를 다듬는 순간
        # 조용히 안 잡히므로 토큰을 따로 둔다 ([[EMPTY_DYNMASK_TOKEN]] 과 같은 처방).
        raise SystemExit(f"{NO_EMITTABLE_TOKEN} "
                         f"내보낼 변이가 하나도 없다 — 필터를 확인할 것 "
                         f"(뱅크 변이 {len(bank['variants'])}개, 필터 {dropped['filter']} / "
                         f"접힌 단 {dropped['folded']} / path {dropped['path_len']}"
                         + (f" / require {dropped['require']} {require_hits}" if require else "")
                         + ").")

    output_folder = args.out_dir or path.join(bank_folder, "canonical")
    makedirs(output_folder, exist_ok=True)
    canonical = {"n_poses": args.n_poses,
                 "convention": "rel c2w, OpenCV (X right/Y down/Z fwd), frame0 anchor, unit scale",
                 "source_bank": path.abspath(path.join(bank_folder, "bank.json")),
                 "notes": MODEL_NOTES, "cameras": cameras}
    canonical_path = path.join(output_folder, "canonical.json")
    with open(canonical_path, "w", encoding="utf-8") as file:
        json.dump(canonical, file, ensure_ascii=False, indent=1)
    np.savez_compressed(path.join(output_folder, "canonical.npz"),
                        tags=np.asarray(tags), rel=np.stack(rel_stack),
                        rel_full=np.stack(rel_full_stack),
                        rmax=np.asarray([r["model_gauge_rmax"] for r in records]),
                        S_da3=float(bank["S"]))
    # D87. 리타이밍한 world c2w 를 뱅크의 `poses.npz` 와 **같은 스키마**로 떨군다 —
    # `render_bank_videos.py --poses_npz` 가 그대로 읽어 대조 영상을 만든다. 필터가 하나라도
    # 걸리면 행 순서가 bank.json 과 어긋나 렌더러의 길이 assert 가 무의미해지므로 막는다.
    if args.dump_poses:
        if any(dropped.values()):
            raise SystemExit(
                f"--dump_poses 는 전량 내보낼 때만 쓴다 (필터에 걸린 변이 {dropped}).\n"
                f"  `poses.npz` 는 bank.json 변이 순서를 그대로 가정하는 포맷이다.")
        dump_path = args.dump_poses if args.dump_poses.endswith(".npz") \
            else path.join(output_folder, "poses.npz")
        np.savez_compressed(dump_path, cam_c2w=np.stack(world_poses),
                            variant_id=np.asarray([r["variant_id"] for r in records]),
                            anchor_id=np.asarray([cameras[t]["anchor_id"] for t in tags]),
                            target_hole=np.asarray([r["target_hole"] for r in records]),
                            hole_delta=np.asarray([cameras[t]["hole_delta"] for t in tags]))
    manifest = {"format": "lbm_bank_canonical_v1", "video": args.video,
                "bank_dir": args.bank_dir, "n_poses": args.n_poses,
                "num_cameras": len(cameras), "dropped": dropped,
                "fixed_keys_missing_from_bank": missing,
                # D87. 리타이밍했으면 두 값이 다르고, 아래 오차는 **위치 전용**이다.
                "keyframe_ease_bank": ease_from_bank,
                "keyframe_ease": str(fixed["keyframe_ease"]),
                "retimed": retimed,
                "worst_pose_rebuild_error": worst_pose,
                # 손잡이 반올림이 `fit_tau` 계단을 넘겨 되찾아야 했던 행들. 비어 있는 게 정상이고,
                # 채워져 있으면 그 뱅크는 `knob_raw` 이전에 fit 된 것이다.
                "recovered_knobs": [{"variant_id": v, "rounded": a, "recovered": b}
                                    for v, a, b in recovered_knobs],
                "worst_roundtrip_error": worst_round, "entries": records}
    with open(path.join(output_folder, "manifest.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=1)

    degenerate = sum(r["translation_degenerate"] for r in records)
    folded_kept = sum(r["folded"] for r in records)
    gauges = np.asarray([r["g"] for r in records], dtype=float)
    print(f"{'video':<26}{args.video}   bank {args.bank_dir}")
    print(f"{'variants':<26}{len(bank['variants'])} 중 {len(cameras)} 내보냄"
          f"   (필터 {dropped['filter']} / 접힌 단 {dropped['folded']} / "
          f"path {dropped['path_len']}"
          + (f" / require {dropped['require']} {require_hits}" if require else "") + ")")
    print(f"{'접힌 단 (내보낸 것 중)':<24}{folded_kept}   `folded_onto` 로 대표 태그를 가리킨다")
    print(f"{'translation_degenerate':<26}{degenerate}   이동 0 — --scales 도 |t| 정규화도 무의미")
    print(f"{'g = rmax/S 범위':<25}[{gauges.min():.5f}, {gauges.max():.5f}]"
          f"   median {np.median(gauges):.5f}")
    if retimed:
        print(f"{'회전 리타이밍':<24}{ease_from_bank} -> {fixed['keyframe_ease']}"
              f"   위치·손잡이 그대로, 회전 스케줄만 (재fit 아님)")
    print(f"{'pose 재현 최대오차':<24}{worst_pose:.3e}"
          f"   ({worst_pose_tag or '-'}){'  [위치 전용]' if retimed else ''}"
          f"{'  ⚠ strict off' if not args.strict else ''}")
    print(f"{'21<->49 왕복 최대오차':<23}{worst_round:.3e}   (index pick 이라 0 이어야 한다)")
    if missing:
        print(f"{'⚠ 예전 뱅크':<25}fixed 에 {missing} 없음 — 기본값으로 재현했고 "
              f"pose 대조로 검증됨")
    if recovered_knobs:
        print(f"{'손잡이 반올림 복구':<24}{len(recovered_knobs)} 행 "
              f"(`knob_raw` 이전 뱅크 — 재현은 pose 대조로 검증됨)")
        for vid, rounded, exact in recovered_knobs:
            print(f"{'':<26}{vid}  {rounded:.5f} -> {exact:.9f}")
    print(f"\n{'canonical':<26}{canonical_path}")
    if args.dump_poses:
        print(f"{'world pose 덤프':<24}{dump_path}   "
              f"render_bank_videos.py --poses_npz 로 대조")
    print(f"{'다음 명령':<25}python emit_model_cams.py --canonical {canonical_path} \\\n"
          f"{'':<26}  --model sierpinskicam --cameras all "
          f"--scales <rmax/1.9876>")


def build_parser():
    """파서를 **함수로** 꺼내 둔 이유: `run_bank.py --exec inproc` 이 이 스크립트를 서브프로세스가
    아니라 같은 프로세스에서 부른다 (D180). 파서가 `__main__` 블록 안에 있으면 import 로는
    만들 수 없다. CLI 동작은 그대로다 — 아래 `__main__` 이 이 함수를 쓴다.
    """
    parser = ArgumentParser(description="hole 뱅크 전량을 태그 N개짜리 canonical.json 하나로")
    parser.add_argument("--video", required=True)                       # 영상 이름 (out/<video>)
    parser.add_argument("--bank_dir", default="hole_bank")              # 뱅크 폴더 이름
    parser.add_argument("--output_root", default=None)                  # 기본 <repo>/out
    parser.add_argument("--out_dir", default=None)                      # 기본 <bank>/canonical
    # 보통은 안 준다 — 뱅크 meta 의 경로를 그대로 쓴다. 뱅크를 다른 기계로 옮겼을 때만 필요.
    parser.add_argument("--external_shapes", default=None, type=str)
    parser.add_argument("--external_aim", default=None, type=str)
    parser.add_argument("--tag_prefix", default="")                     # 태그 앞에 붙일 문자열
    parser.add_argument("--n_poses", default=21, type=int)              # emit_model_cams 는 21 고정
    # 재현 검증. **끄지 말 것** — 끄면 fixed 블록이 어긋난 예전 뱅크에서 조용히 다른 궤적이
    # 나간다 (숫자가 전부 그럴듯해서 육안으로 안 잡힌다).
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no_strict", dest="strict", action="store_false")
    parser.add_argument("--pose_tol", default=1e-9, type=float)         # 재현 허용 오차 (u)
    # D87. 뱅크의 회전 스케줄만 갈아끼운다 (재fit 없음). 비우면 뱅크에 적힌 값 그대로 = 예전 동작.
    parser.add_argument("--keyframe_ease", default="", type=str,
                        choices=["", "smoothstep", "linear", "arclen", "arclen_kf", "smooth_kf"])
    # D89/D90. smooth_kf 전용. 다른 ease 에서는 아무 일도 안 한다.
    # **기본값이 None** 인 이유: 예전엔 여기 4 가 박혀 있어서 뱅크가 12 로 구워졌든 말든
    # 4 로 되만들었고, 위치는 정확히 맞아서 대조 assert 가 원인을 안 가리켰다. 이제 안 주면
    # 뱅크 `fixed.smooth_passes` (없으면 `SHAPE_DEFAULTS` 의 12) 를 따르고, 주면 그게 이긴다.
    parser.add_argument("--smooth_passes", default=None, type=int)
    parser.add_argument("--smooth_lambda", default=None, type=float)
    # D87. 리타이밍한 world c2w 를 뱅크와 같은 스키마로 저장 (렌더 대조용).
    # 플래그만 주면 `<out_dir>/poses.npz`, 경로(.npz)를 주면 거기로.
    parser.add_argument("--dump_poses", default="", nargs="?", const="1", type=str)
    # 필터. 기본은 **전량**이다 — 뱅크는 재고 목록이고 거르는 건 소비자 몫 (D39/D45).
    parser.add_argument("--status", default="all")                      # 쉼표 구분 (solved 등)
    parser.add_argument("--anchors", default=None)                      # 쉼표 구분 anchor id
    parser.add_argument("--presets", default=None)                      # 쉼표 구분 preset
    parser.add_argument("--holes", default=None)                        # 쉼표 구분 target_hole
    parser.add_argument("--min_path_len", default=0.0, type=float)      # 이 아래는 뺀다 (u)
    # D192. 측정 열 문턱 (`열>=값` / `열<=값` / `열>값` / `열<값`, 여러 개면 AND). 근거는
    # `parse_require` docstring — `track_look_at` 이 fit 게이트를 안 거치는 구멍을 여기서 막는다.
    #   예) --require "obb_slack>=0" "behind_frac<=0" "subject_visible_frac>=0.6"
    # 빈 목록(기본) = 조건 없음 = 옛 동작 비트 동일.
    parser.add_argument("--require", nargs="*", default=[], type=str)
    # D188 ③. `picked` 열이 찬 행만 내보낸다 (= 씬당 `--pick_budget` 대). 열이 없는 옛 뱅크에
    # 주면 0 행이 되므로, 뱅크를 `--pick_budget` 으로 구웠을 때만 켠다.
    parser.add_argument("--picked_only", dest="picked_only", action="store_true", default=False)
    parser.add_argument("--no_picked_only", dest="picked_only", action="store_false")
    # 접힌 단: 충돌 천장이 낮아 4단이 같은 궤적이 된 경우. 기본은 남기고 `folded_onto` 로 표시.
    parser.add_argument("--drop_folded", dest="drop_folded", action="store_true", default=False)
    parser.add_argument("--no_drop_folded", dest="drop_folded", action="store_false")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
