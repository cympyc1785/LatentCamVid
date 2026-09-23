"""정지 preset rung(별도 뱅크)을 기존 hole 뱅크에 **붙인다**.

왜 이 스크립트가 있나: `fit_hole_ladder` 는 D78 이전에 `knob_kind()` 가 None 인
`STATIC_PRESETS` 를 preset 목록에서 통째로 빼 버렸다. 그래서 τ 뱅크에는 있는 정지 카메라
(`static_hold`, `static_look_at`)와 순수 추종(`track_hold`, `track_look_at`)이 emit 뱅크에
**0 행**이었고, 학습 캡션 어휘에서 "카메라가 가만히 있는다"와 "따라만 간다"가 통째로
사라졌다. (D94 이전 이름으로는 각각 `static_hold_dont_look`/`static_hold`,
`track_hold_dont_look`/`track_hold` — 완전고정과 조준추종이 이름을 맞바꿨다.)

D78 로 고쳤지만 52편을 전량 다시 fit 하면 편당 9~44분이 다시 든다. 정지 rung 은 이분법이
없어서 (anchor × 4 preset) × 렌더 2회뿐이라 **편당 2분 미만**이다 (camel 실측 14변이 /
182렌더). 그래서 정지 preset 만 별도 `--static_dir` 로 fit 하고 여기서 붙인다. 두 fit 은
anchor·preset 루프가 서로 독립이고 같은 τ 뱅크·같은 인자를 쓰므로, 붙인 결과는 전량 재fit
결과와 같다 — 그 전제를 `fixed` 블록 대조로 **검사한 뒤에** 붙인다.

멱등하다: 이미 붙어 있으면 그 preset 행을 먼저 걷어내고 다시 붙이므로 두 번 돌려도 같다.

    python fit/bank/merge_static_rung.py --video camel
    python fit/bank/merge_static_rung.py --video camel --bank_dir hole_bank_k6_d77 \
        --static_dir hole_bank_k6_d77_static --dry_run
"""
from argparse import ArgumentParser
from os import path
import json
import sys

import numpy as np

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from lbm.presets import STATIC_PRESETS, row_preset                           # noqa: E402

OUT_ROOT_DEFAULT = path.join(HERE, "out")

# 두 fit 이 **같은 조건**이었는지 보는 키. 여기가 어긋나면 붙인 행의 궤적이 나머지와 다른
# 규약으로 만들어진 것이라 조용히 섞이면 안 된다. `fixed` 블록에 없는 키는 비교에서 뺀다
# (예전 뱅크 호환) — 양쪽 다 없을 때만 봐준다.
FIXED_KEYS = ("speed", "tracking", "look_at_bias", "start_mode", "aim_anchor",
              "aim_ramp_frames", "orbit_span_frac", "min_sweep_deg", "traj_basis",
              "aim_keyframes", "keyframe_aim", "keyframe_ease", "fixed_focal")
# `smooth_kf` 뱅크에서만 비교하는 키 (아래 `check_compatible` 참고). 폴백은 `build_poses` 의
# **서명** 기본값이어야 한다 — 이 키가 없는 뱅크는 인자를 아예 안 넘겨서 그걸로 구워졌다.
SHAPE_SMOOTH_FALLBACK = {"smooth_passes": 12, "smooth_lambda": 0.5}
# D97. 같은 이유로 `FIXED_KEYS` 에 못 넣는다 — D97 이전 뱅크는 이 키가 없어서 (None vs "auto")
# 로 멀쩡한 병합이 전부 거부된다. 폴백은 `SHAPE_DEFAULTS["tau_ref"]` 와 같은 `"source"` 다.
TAU_REF_FALLBACK = "source"
# D99. 같은 이유. 폴백은 `SHAPE_DEFAULTS["deroll"]` 와 같은 False 다 (D99 이전 = 보정 없음).
DEROLL_FALLBACK = False
# 뱅크 전체가 공유하는 스칼라. 다르면 다른 씬·다른 렌더 설정이다.
SCALAR_KEYS = ("video", "num_frames", "S", "z_med")


def load_bank(folder: str):
    """bank.json + poses.npz 를 읽어 (뱅크, poses) 로. 행 순서와 pose 순서는 같아야 한다."""
    with open(path.join(folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    poses = np.load(path.join(folder, "poses.npz"), allow_pickle=False)
    cam = poses["cam_c2w"]
    assert len(bank["variants"]) == len(cam), \
        f"{folder}: 행 {len(bank['variants'])} != pose {len(cam)}"
    # 순서가 어긋나면 궤적이 다른 행에 붙는다 — 조용히 틀리는 종류라 여기서 못 박는다.
    ids = [str(v) for v in poses["variant_id"]]
    assert ids == [r["variant_id"] for r in bank["variants"]], f"{folder}: variant_id 순서 불일치"
    return bank, cam


def check_compatible(main: dict, extra: dict):
    """두 뱅크가 같은 조건에서 나왔는지. 어긋난 키 목록을 돌려준다 (빈 리스트면 통과)."""
    bad = []
    for key in SCALAR_KEYS:
        a, b = main.get(key), extra.get(key)
        if isinstance(a, float) or isinstance(b, float):
            same = a is not None and b is not None and abs(float(a) - float(b)) < 1e-9
        else:
            same = a == b
        if not same:
            bad.append((key, a, b))
    fa, fb = main.get("fixed", {}), extra.get("fixed", {})
    for key in FIXED_KEYS:
        if fa.get(key) != fb.get(key):
            bad.append((f"fixed.{key}", fa.get(key), fb.get(key)))
    # D96. 평활 인자는 `smooth_kf` 일 때만 궤적에 나타난다 (`build_poses.py:679`). 그래서
    # `FIXED_KEYS` 에 넣으면 안 된다 — smoothstep 뱅크는 이 키가 아예 없어서 (None vs 12)
    # 멀쩡한 병합이 거부된다. 반대로 smooth_kf 인데 안 보면 **회전만 조용히 어긋난다**
    # (위치는 정확히 맞아서 안 들킨다 — D90 에서 실제로 당한 실패 모드).
    if str(fa.get("keyframe_ease", "")) == "smooth_kf":
        for key in ("smooth_passes", "smooth_lambda"):
            a, b = fa.get(key, SHAPE_SMOOTH_FALLBACK[key]), fb.get(key, SHAPE_SMOOTH_FALLBACK[key])
            if float(a) != float(b):
                bad.append((f"fixed.{key}", a, b))
    # D97. τ 기준축. 폴백을 씌워 비교하므로 "옛 뱅크(키 없음) + 옛 뱅크" 는 통과하고
    # "옛 뱅크 + auto 뱅크" 만 걸린다 — 후자는 `track_*` 궤적이 실제로 다르다.
    if (str(fa.get("tau_ref", TAU_REF_FALLBACK))
            != str(fb.get("tau_ref", TAU_REF_FALLBACK))):
        bad.append(("fixed.tau_ref", fa.get("tau_ref"), fb.get("tau_ref")))
    # D99. 중력 기준 roll 보정. 같은 폴백 방식 — 옛 뱅크끼리는 통과하고 옛+deroll 만 걸린다.
    if bool(fa.get("deroll", DEROLL_FALLBACK)) != bool(fb.get("deroll", DEROLL_FALLBACK)):
        bad.append(("fixed.deroll", fa.get("deroll"), fb.get("deroll")))
    if main.get("source_bank") != extra.get("source_bank"):
        bad.append(("source_bank", main.get("source_bank"), extra.get("source_bank")))
    return bad


def main(args):
    root = args.output_root or OUT_ROOT_DEFAULT
    main_dir = path.join(root, args.video, args.bank_dir)
    static_dir = path.join(root, args.video, args.static_dir)
    for folder in (main_dir, static_dir):
        if not path.isfile(path.join(folder, "bank.json")):
            print(f"없음 — {folder}/bank.json")
            return 1

    bank, cam = load_bank(main_dir)
    extra, cam_extra = load_bank(static_dir)

    bad = check_compatible(bank, extra)
    if bad:
        print("두 뱅크의 조건이 다르다 — 붙이지 않는다:")
        for key, a, b in bad:
            print(f"  {key:<28}{a!r:>24}  vs  {b!r}")
        return 2

    # 정지 여부는 **적힌 이름이 아니라 정식 이름**으로 판정한다. 옛 뱅크는 `static_hold_locked`
    # (aim=traj) 라고 적어 두는데 그건 `static_hold_dont_look` 이라 `STATIC_PRESETS` 에 없는
    # 이름이다 — 이름 그대로 비교하면 (a) 여기서 stray 오탐이 나고 (b) 아래 멱등성 걷어내기가
    # 그 행을 못 지워 병합 후 정지 단이 두 벌이 된다. 실측: vista `bank/` 52편에 303행.
    stray = sorted({r["preset"] for r in extra["variants"]
                    if row_preset(r) not in STATIC_PRESETS})
    if stray:
        print(f"static 뱅크에 정지 preset 이 아닌 행이 있다: {stray}")
        return 3

    # 멱등성: 이미 붙어 있던 정지 행을 먼저 걷어낸다.
    keep = [i for i, r in enumerate(bank["variants"]) if row_preset(r) not in STATIC_PRESETS]
    removed = len(bank["variants"]) - len(keep)
    rows = [bank["variants"][i] for i in keep] + list(extra["variants"])
    poses = np.concatenate([cam[keep], cam_extra], axis=0)

    ids = [r["variant_id"] for r in rows]
    assert len(set(ids)) == len(ids), "variant_id 중복 — 같은 조합이 두 번 들어갔다"

    print(f"{'video':<16}{args.video}")
    print(f"{'main':<16}{args.bank_dir:<26}{len(bank['variants']):>6} 행 "
          f"(정지 {removed} 걷어냄)")
    print(f"{'static':<16}{args.static_dir:<26}{len(extra['variants']):>6} 행 "
          f"{sorted({r['preset'] for r in extra['variants']})}")
    print(f"{'merged':<16}{'':<26}{len(rows):>6} 행   preset "
          f"{len(set(r['preset'] for r in rows))}")
    if args.dry_run:
        print("\n--dry_run — 아무것도 안 썼다")
        return 0

    bank["variants"] = rows
    # 상한표도 같이 붙인다 (정지 preset 은 rung 이 하나라 그게 곧 상한이다).
    bank.setdefault("tau_caps", {})
    for preset, caps in extra.get("tau_caps", {}).items():
        bank["tau_caps"][preset] = caps
    # 붙였다는 표식. 뱅크를 나중에 읽는 쪽이 "이 뱅크의 정지 행은 별도 fit 이다"를 알아야 한다.
    bank["static_rung_merged"] = {"from": args.static_dir,
                                  "presets": sorted({r["preset"] for r in extra["variants"]}),
                                  "rows": len(extra["variants"])}

    with open(path.join(main_dir, "bank.json"), "w", encoding="utf-8") as file:
        json.dump(bank, file, ensure_ascii=False, indent=1)
    with open(path.join(main_dir, "tau_caps.json"), "w", encoding="utf-8") as file:
        json.dump({"format": "lbm_tau_caps_v1", "video": args.video,
                   "cap_hole": bank.get("cap_hole"), "caps": bank["tau_caps"]},
                  file, ensure_ascii=False, indent=1)

    # CSV 열 목록은 **기존 csv 헤더에서 읽는다** — 여기에 따로 적으면 `fit_hole_ladder` 의
    # `columns` 와 어긋나는 순간 조용히 열이 밀린다.
    with open(path.join(main_dir, "bank.csv"), encoding="utf-8") as file:
        columns = file.readline().strip().split(",")
    with open(path.join(main_dir, "bank.csv"), "w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")
        for row in rows:
            fields = [str(row.get(c, "")) for c in columns]
            assert not any("," in f for f in fields), \
                f"쉼표가 든 값: {[(c, f) for c, f in zip(columns, fields) if ',' in f]}"
            file.write(",".join(fields) + "\n")

    np.savez_compressed(path.join(main_dir, "poses.npz"),
                        cam_c2w=poses,
                        variant_id=np.array(ids),
                        anchor_id=np.array([r["anchor_id"] for r in rows]),
                        target_hole=np.array([r["target_hole"] for r in rows]),
                        hole_delta=np.array([r["hole_delta"] for r in rows]))
    print(f"\n기록  {main_dir}   (bank.json / bank.csv / poses.npz / tau_caps.json)")
    print("다음  emit_bank.py 를 같은 --bank_dir 로 다시 돌려 canonical 을 갱신할 것")
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--bank_dir", default="hole_bank_k6_d77", type=str)      # 붙일 대상
    parser.add_argument("--static_dir", default="hole_bank_k6_d77_static", type=str)  # 붙일 것
    parser.add_argument("--dry_run", action="store_true", default=False)
    raise SystemExit(main(parser.parse_args()))
