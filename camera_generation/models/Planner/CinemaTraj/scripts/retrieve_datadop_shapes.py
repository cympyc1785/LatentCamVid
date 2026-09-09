"""plan(원하는 카메라 움직임)에 맞는 **실제 영화 궤적**을 DataDoP 에서 검색해 preset 모양으로 뽑는다.

왜: 우리 preset 36종은 전부 손으로 정의한 기하 primitive(dolly/truck/orbit/...)라 궤적이
교과서적으로 매끄럽다. 실제 촬영 궤적은 가감속·흔들림·복합 동작이 섞여 있고, 그게 캡션 모델이
배워야 할 분포다. DataDoP 3,000 shot 은 이미 `out_captions/datadop_gt/` 에 chunk 단위
`move`/`angular` 라벨이 붙어 있으므로 **새로 태깅할 게 없다** — 라벨로 조회만 하면 된다.

## 무엇을 가져오고 무엇을 버리는가

가져오는 것은 **모양뿐**이다: `rel[0] = I` 로 리앵커하고 **이동만** `max_f |t(f)| = 1` 로
정규화한다 (회전은 원본 각도 그대로 — D83, `shape_from_poses` docstring 에 근거).
크기(이동량)는 우리 τ 사다리와 hole 사다리가 씬마다 다시 푼다. DataDoP 의 world 단위는
MonST3R 게이지(D=1)라 우리 DA3 frame0 게이지 `S` 와 통약이 안 되는데, 정규화해서 버리면
그 불일치가 애초에 파이프라인에 들어오지 않는다.

버리는 것: 절대 스케일, intrinsics(focal_ratio), 원본 씬의 subject. 조준은 `aim="traj"` 로
원본 회전을 그대로 쓴다 (`lbm.presets.register_external_presets` docstring 참고) — 즉 이
모양들은 **free-moving** 이다. subject 를 조준하지 않으므로 캡션에도 target 절이 안 붙는다
(`build_bank_captions.py` 의 `free_moving` 분기).

## 검색 방식 (D83)

plan 을 손으로 적지 않고 **라벨 분포에서 자동 생성**한다 (`build_plan`) — `move × angular` 결합
라벨 중 길이 `min_span` 이상 chunk 가 `min_pool` 개 이상인 것 전부. 그리고 각 query 안에서
**무작위로** `top_k` 개를 뽑는다 (`--seed` 로 재현). 예전에는 6개 query × 정렬 상위 2개라
945개 후보 중 12개(1.3%)만 썼고 그 12개가 매 실행 동일했다.

## 규약

`shot_*_transforms_cleaning.json` 의 `transform_matrix` 는 **이미 OpenGL c2w** 다
(`caption_cameras_datadop.load_datadop_json` 이 그래서 키를 `cam_gl_c2w` 로 둔다). 우리
파이프라인은 OpenCV 라 `c2w_cv = c2w_gl @ diag(1,-1,-1,1)` 로 한 번 뒤집는다.

예시:
    python scripts/retrieve_datadop_shapes.py --plan_file configs/datadop_plan.json \
        --out configs/datadop_shapes.json
    python scripts/retrieve_datadop_shapes.py --move "move backward" --angular any --top_k 3
"""
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)
# D83 이후 `se3` 를 안 쓴다 — 정규화가 이동 나눗셈 한 줄이라 recammaster 를 sys.path 에 올릴
# 이유가 사라졌다 (`shape_from_poses` docstring 참고).


def load_datadop_json(json_path: str):
    """`shot_*_transforms_cleaning.json` → `cam_gl_c2w` (n,4,4).

    `scripts/caption_cameras_datadop.py:259` 과 같은 로직을 **여기 다시 적는다** — 그쪽 모듈은
    import 시 GenDoP `core.utils`(megfile 의존)를 끌고 오는데 이 스크립트는 vista4d env 에서
    돈다. `transform_matrix` 는 이미 OpenGL c2w 라 여기서는 뒤집지 않는다 (프레임 순서도
    파일 순서 그대로 — 정렬하면 원본 태깅과 인덱스가 어긋난다).
    """
    with open(json_path) as f:
        meta = json.load(f)
    return {"cam_gl_c2w": np.asarray([fr["transform_matrix"] for fr in meta["frames"]],
                                     dtype=np.float64)}


GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])

TAG_GLOB_DEFAULT = path.join(CINEMATRAJ_ROOT, "out_captions", "datadop_gt", "*", "*_tag.json")

# plan 을 손으로 6줄 적던 걸 **라벨 분포에서 자동 생성**으로 바꿨다 (D83). 이유는 그 6줄이
# `move × angular` 결합 어휘의 극히 일부만 건드렸기 때문이다 — 실측 3,000 shot / 1,043 scene 에서
# 길이 30 이상 chunk 의 결합 라벨은 100종이 넘는데, 예전 plan 은 그중 6종(그나마 5종은
# `angular: static`)만 조회해 945개 후보 중 12개(1.3%)만 썼다. 궤적 다양성이 목표인데
# "직선 dolly / 직선 truck" 만 가져오면 preset 과 겹쳐서 가져오는 의미가 없다.
#
# 실측 어휘 (`segment_rigidbody_trajectories` 라벨 그대로):
#   move    static 3227 / forward 1244 / backward 903 / right 780 / left 754 / up 239 / down 192
#           + 복합 20종 (`move left and forward` 144 … `move right, down, and backward` 2)
#   angular static 5954 / yaw right 831 / yaw left 819 / pitch up 324 / pitch down 216
#           / roll right 57 / roll left 47
# `(static, static)` 만 제외한다 — 정지 hold 는 `static_hold` preset 이 이미 낸다.
EXCLUDE_JOINT = {("static", "static")}

# 결합 라벨 → 라우터 슬롯. free-moving 이라 슬롯은 "이 모양이 어느 preset 칸과 겹치나"라는
# 배치 힌트일 뿐이지만, `pick_external` 이 **preset 이 못 채운 슬롯을 먼저** 주는 데 쓴다.
def slot_for(move: str, angular: str):
    m = move.replace("move ", "")
    if "forward" in m:
        return "arc" if angular != "static" else "advance"
    if "backward" in m:
        return "arc" if angular != "static" else "recede"
    if "left" in m or "right" in m:
        return "lateral"
    if "up" in m or "down" in m:
        return "vertical"
    return "rotate"                          # move static + 회전만


def slugify(move: str, angular: str):
    """`move left and forward` × `yaw right` → `dd_l_fwd__yawr`. 캡션에 안 쓰이는 내부 이름."""
    m = (move.replace("move ", "").replace(", and ", "_").replace(" and ", "_")
         .replace(", ", "_").replace(" ", "_")
         .replace("left", "l").replace("right", "r").replace("forward", "fwd")
         .replace("backward", "back"))
    a = (angular.replace(" ", "").replace("yaw", "yaw_").replace("pitch", "pit_")
         .replace("roll", "rol_").replace("left", "l").replace("right", "r")
         .replace("up", "u").replace("down", "d"))
    return f"dd_{m}__{a}"


def build_plan(files, window: int, min_span: int, min_pool: int):
    """길이 `min_span` 이상 chunk 가 `min_pool` 개 이상인 결합 라벨 전부를 query 로. -> plan list.

    빈도 내림차순으로 정렬하되 **query 당 뽑는 개수(`top_k`)는 같다** — 빈도에 비례해 뽑으면
    코퍼스가 `move forward` 로 도로 쏠린다. 우리가 원하는 건 코퍼스 재현이 아니라 어휘 커버리지다.
    """
    pool = defaultdict(int)
    for f in files:
        try:
            tag = json.load(open(f))
        except Exception:
            continue
        if int(tag.get("num_poses", 0)) < window:
            continue
        for c in tag.get("chunks", []):
            if c["end"] - c["start"] + 1 >= min_span:
                pool[(c.get("move"), c.get("angular"))] += 1
    keys = [k for k, v in sorted(pool.items(), key=lambda kv: -kv[1])
            if v >= min_pool and k not in EXCLUDE_JOINT and all(k)]
    return [{"slot": slot_for(m, a), "name": slugify(m, a), "move": m, "angular": a,
             "pool": pool[(m, a)]} for m, a in keys]


def chunk_spans(tag: dict, move: str, angular: str):
    """조건에 맞는 chunk 들의 **연속 구간**. `"any"` 는 와일드카드. -> [(start, end), ...]"""
    hits = [c for c in tag.get("chunks", [])
            if (move == "any" or c.get("move") == move)
            and (angular == "any" or c.get("angular") == angular)]
    spans, cur = [], None
    for c in sorted(hits, key=lambda x: x["start"]):
        if cur is not None and c["start"] <= cur[1] + 1:
            cur = (cur[0], max(cur[1], c["end"]))
        else:
            if cur is not None:
                spans.append(cur)
            cur = (c["start"], c["end"])
    if cur is not None:
        spans.append(cur)
    return spans


def window_from_span(span, num_poses: int, window: int):
    """구간 안(모자라면 구간 중심)에서 `window` 프레임을 잘라낸다. -> (lo, hi) 반개구간."""
    start, end = span
    mid = 0.5 * (start + end + 1)
    lo = int(round(mid - window / 2))
    lo = max(0, min(lo, num_poses - window))
    return lo, lo + window


# DataDoP world 단위에서의 "장면 깊이". MonST3R 가 shot 단위로 pairwise 스케일 기하평균을
# 0.5 로 고정해 두었으므로 D=1 안에서 이 값이 z_med 역할을 한다
# ([[datadop-world-scale-is-monst3r-gauge]]). **shot 의 depth.npy 로 재정규화하면 안 된다** —
# 이미 정규화된 걸 또 나누면 산포가 오히려 커진다 ([[datadop-depth-does-not-canonicalize]]).
DATADOP_ZMED = 0.5


def total_rotation_deg(rel: np.ndarray):
    """프레임별 상대회전 각도의 합. 모양이 얼마나 도는지 (게이지 무관, degree)."""
    rot = 0.0
    for a in range(len(rel) - 1):
        R = rel[a, :3, :3].T @ rel[a + 1, :3, :3]
        rot += float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return rot


def shape_from_poses(gl_c2w: np.ndarray, static_tau: float = 0.02, rot_cap_deg: float = 180.0):
    """OpenGL c2w (n,4,4) → OpenCV rel c2w 모양. (rel, info dict).

    정규화는 **이동만** `1/rmax` 배다 — 회전은 원본 각도 그대로 둔다 (D83). 예전에는 SE(3) 로그
    전체를 `se3.scale_traj` 로 줄여서 "회전:이동 비"를 보존했는데, 우리 파이프라인에서는 그게
    틀린 보존량이다: `fit_tau`/`fit_hole_ladder` 가 **씬마다 이동만 다시 푼다**. 즉 하류가
    이동을 갈아끼우므로 지켜야 할 건 비율이 아니라 **절대 회전각**이다. 실측으로 `scale_traj`
    는 rmax<1 인 shot 에서 회전을 그대로 증폭했다 (13.0°→76.1°, 18.2°→126.2°, 19.7°→121.1°;
    path/chord 도 1.201→1.489). 이동만 나누면 회전과 path/chord 가 정확히 보존된다.

    `tau_native = rmax / DATADOP_ZMED` 는 "원본 이동량이 우리 τ 사다리의 몇 단인가"다.
    이게 `static_tau` 미만이면 이동이 사실상 없는 shot 이라(`move: static` 라벨) `1/rmax` 가
    **지터를 단위 길이로 증폭**한다. 그때는 이동을 0 으로 두고 회전 전용으로 표시한다 —
    `sample_camera_bank` 가 `pan_left/right` 와 같은 취급(사다리를 각도로)으로 돌린다.

    `rot_cap_deg` 는 이제 **원본 회전각**에 걸리는 상한이다 (정규화가 회전을 안 건드리므로
    예전처럼 `rot/rmax` 를 볼 이유가 없다). 49프레임에 수백 도를 도는 shot 은 whip pan 이라
    우리 49프레임 target 으로 못 쓴다 — `too_wild` 로 표시하고 `retrieve` 가 버린다.
    """
    cv = np.asarray(gl_c2w, dtype=np.float64) @ GL2CV
    rel = np.linalg.inv(cv[0])[None] @ cv
    rmax = float(np.linalg.norm(rel[:, :3, 3], axis=-1).max())
    tau_native = rmax / DATADOP_ZMED
    rot = total_rotation_deg(rel)
    rotation_only = bool(tau_native < static_tau)
    rel = rel.copy()
    if rotation_only:
        rel[:, :3, 3] = 0.0
    else:
        rel[:, :3, 3] /= rmax               # 이동만. 회전은 원본 각도 유지 (D83)
    rel[0] = np.eye(4)                      # 수치오차로 1e-16 만큼 어긋나는 걸 못박는다
    return rel, {"source_rmax": round(rmax, 6), "rot_deg_source": round(rot, 2),
                 "rot_deg_shape": round(rot, 2), "tau_native": round(tau_native, 4),
                 "rotation_only": rotation_only, "too_wild": bool(rot > rot_cap_deg)}


def path_over_chord(rel: np.ndarray):
    """경로길이 / 시작-끝 직선거리. 1.0 = 완전 직선, 크면 곡선. 이동이 0 이면 None."""
    t = rel[:, :3, 3]
    seg = float(np.linalg.norm(np.diff(t, axis=0), axis=-1).sum())
    chord = float(np.linalg.norm(t[-1] - t[0]))
    return None if chord < 1e-9 else seg / chord


def load_index(tag_glob: str):
    files = sorted(glob(tag_glob))
    assert files, f"tag 파일이 없다: {tag_glob}"
    return files


def retrieve(query: dict, files, window: int, min_span: int, min_translation: float,
             per_scene: int, top_k: int, args_static_tau: float = 0.02,
             args_rot_cap: float = 180.0, rng=None, min_purity: float = 0.6):
    """한 query 에 대한 `top_k` shot. 씬당 `per_scene` 개까지만 (한 영화에 쏠리지 않게).

    D83 부터 **무작위 추출**이다. 예전에는 `(-purity, |이동량 − 코퍼스 중앙값|)` 로 정렬해
    맨 위를 잘랐는데, 그건 "라벨이 맞는 것들 중 가장 평균적인 shot" 을 매번 같은 순서로 고른다 —
    라벨당 후보가 수십~수백인데 항상 같은 한두 개만 나오니 실제 촬영 궤적을 쓰는 이유(분포의
    다양성)가 사라진다. 이제 `purity ≥ min_purity` 를 **하한 필터**로만 쓰고 통과분을
    `rng` 로 섞는다. `--seed` 로 재현 가능하다.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    cands = []
    for f in files:
        try:
            tag = json.load(open(f))
        except Exception:
            continue
        n = int(tag.get("num_poses", 0))
        if n < window:
            continue
        spans = [s for s in chunk_spans(tag, query["move"], query["angular"])
                 if s[1] - s[0] + 1 >= min_span]
        if not spans:
            continue
        span = max(spans, key=lambda s: s[1] - s[0])
        # 요청한 움직임이 창의 몇 %를 채우는가. 1.0 이면 창 전체가 그 동작이다.
        purity = min(1.0, (span[1] - span[0] + 1) / window)
        cands.append({"tag_path": f, "tag": tag, "span": span, "purity": purity,
                      "total_translation": float(tag.get("total_translation", 0.0))})
    # 이동을 요구한 query 인데 총 이동이 바닥이면 라벨만 맞고 궤적은 정지다.
    if query["move"] != "static":
        cands = [c for c in cands if c["total_translation"] >= min_translation]
    # purity 는 순위가 아니라 하한이다 — 창의 60% 이상이 그 동작이면 "그 패턴의 shot" 으로 본다.
    # 그 아래로 내려가면 창의 절반이 다른 동작이라 라벨이 모양을 설명하지 못한다.
    strict = [c for c in cands if c["purity"] >= min_purity]
    cands = strict if strict else cands      # 희귀 라벨은 하한을 못 채워도 버리지 않는다
    cands = [cands[i] for i in rng.permutation(len(cands))]   # dict 리스트라 shuffle 대신 순열

    picked, by_scene = [], defaultdict(int)
    for c in cands:
        scene = c["tag"]["scene"]
        if by_scene[scene] >= per_scene:
            continue
        data = load_datadop_json(c["tag"]["source_npz"])
        poses = data["cam_gl_c2w"]
        lo, hi = window_from_span(c["span"], len(poses), window)
        rel, info = shape_from_poses(poses[lo:hi], args_static_tau, args_rot_cap)
        if query["move"] != "static" and info["rotation_only"]:
            continue                        # 이동 라벨인데 실제로는 안 움직였다
        if info["too_wild"]:
            continue                        # 49프레임에 수백 도 — whip pan 은 target 으로 못 쓴다
        by_scene[scene] += 1
        picked.append({
            "name": f"{query['name']}_{len(picked)}", "slot": query.get("slot"),
            "move": query["move"], "angular": query["angular"],
            "scene": scene, "shot": c["tag"]["name"], "frames": [lo, hi],
            "span": list(c["span"]), "purity": round(c["purity"], 3), **info,
            "path_over_chord": (None if path_over_chord(rel) is None
                                else round(path_over_chord(rel), 3)),
            "source_tag": c["tag_path"], "source_npz": c["tag"]["source_npz"],
            "convention": "rel c2w, OpenCV, frame0 anchor, max|t|=1 (shape only)",
            "rel": rel.tolist()})
        if len(picked) >= top_k:
            break
    return picked


def main(args):
    files = load_index(args.tag_glob)
    if args.plan_file:
        plan = json.load(open(args.plan_file))
    elif args.move:
        plan = [{"slot": "adhoc", "name": args.name, "move": args.move,
                 "angular": args.angular}]
    else:
        plan = build_plan(files, args.window, args.min_span, args.min_pool)

    shapes, rows, empty = [], [], 0
    for qi, q in enumerate(plan):
        # query 마다 다른 스트림을 준다 — 하나의 rng 를 공유하면 query 순서(=라벨 빈도)가 바뀔 때
        # 앞 query 의 추출이 뒤 query 의 추출까지 통째로 흔들려 재현이 안 된다.
        rng = np.random.default_rng([args.seed, qi])
        got = retrieve(q, files, args.window, args.min_span, args.min_translation,
                       args.per_scene, args.top_k, args.static_tau, args.rot_cap_deg,
                       rng, args.min_purity)
        shapes.extend(got)
        rows.extend(got)
        if not got:
            empty += 1
            print(f"[warn] 해당 없음: move={q['move']!r} angular={q['angular']!r}")

    header = (f"{'name':<22}{'move':<28}{'angular':<11}{'slot':<9}{'scene':<10}{'shot':<12}"
              f"{'frames':<12}{'purity':>7}{'rotDeg':>8}{'p/c':>7}{'tau_nat':>9}{'rotOnly':>9}")
    print(f"\ntags {len(files)}   plan {len(plan)} (empty {empty})   shapes {len(shapes)}\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        pc = r["path_over_chord"]
        print(f"{r['name']:<22}{r['move']:<28}{r['angular']:<11}{str(r['slot']):<9}"
              f"{r['scene']:<10}{r['shot']:<12}"
              f"{str(r['frames']):<12}{r['purity']:>7.2f}{r['rot_deg_source']:>8.1f}"
              f"{(pc if pc is not None else float('nan')):>7.2f}{r['tau_native']:>9.3f}"
              f"{('yes' if r['rotation_only'] else '-'):>9}")

    by_slot = defaultdict(int)
    for r in rows:
        by_slot[r["slot"]] += 1
    print("\nslot:", dict(sorted(by_slot.items())),
          f"  scenes {len({r['scene'] for r in rows})}")

    if args.out:
        makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"format": "datadop_shapes_v1", "window": args.window,
                       "tag_glob": args.tag_glob, "seed": args.seed,
                       "normalization": "translation-only (max|t|=1), rotation as-is (D83)",
                       "plan": plan, "shapes": shapes},
                      f, ensure_ascii=False, indent=1)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tag_glob", default=TAG_GLOB_DEFAULT, type=str)
    parser.add_argument("--plan_file", default=None, type=str)      # 없으면 라벨 분포에서 자동 생성
    parser.add_argument("--move", default=None, type=str)           # 단발 조회용
    parser.add_argument("--angular", default="any", type=str)
    parser.add_argument("--name", default="dd_adhoc", type=str)
    parser.add_argument("--window", default=49, type=int)           # 우리 프레임 수
    parser.add_argument("--min_span", default=30, type=int)         # 라벨 구간 최소 길이
    parser.add_argument("--min_translation", default=0.02, type=float)  # DataDoP world (D=1)
    # 이 아래는 이동이 없다고 보고 회전 전용으로 돌린다 (1/rmax 가 지터를 증폭하는 구간).
    parser.add_argument("--static_tau", default=0.02, type=float)
    # **원본** 회전각 상한. 49프레임에 이만큼 넘게 돌면 whip pan 이라 target 으로 못 쓴다 (D83).
    parser.add_argument("--rot_cap_deg", default=180.0, type=float)
    parser.add_argument("--per_scene", default=1, type=int)         # 한 영화에서 최대 몇 개
    parser.add_argument("--top_k", default=4, type=int)             # query 당 (빈도 무관, 동일)
    # 자동 plan: 길이 `min_span` 이상 chunk 가 이만큼 있는 결합 라벨만 query 로 만든다.
    parser.add_argument("--min_pool", default=5, type=int)
    # 창의 이 비율 이상이 그 동작이어야 "그 패턴의 shot". 순위가 아니라 하한이다 (D83).
    parser.add_argument("--min_purity", default=0.6, type=float)
    parser.add_argument("--seed", default=0, type=int)              # 무작위 추출 재현용
    parser.add_argument("--out", default=None, type=str)
    main(parser.parse_args())
