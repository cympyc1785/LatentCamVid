"""뱅크 변이의 **기하 위반**을 센다 — 카메라가 표면 뒤로 들어가는가 (G1), subject 가 가려지는가 (G3).

왜 따로 필요한가: `sample_camera_bank.py` / `fit_hole_ladder.py` 는 `hole_fraction` 과
`subject_area` 만 잰다. 그런데 **hole 은 충돌을 못 잡는다** — 카메라가 벽을 통과하면 벽 너머
관측이 그대로 그려져서 `valid_mask` 가 멀쩡할 수 있다. 반대로 hole 이 큰 게 곧 충돌인 것도
아니다 (그냥 안 본 방향일 뿐). 두 양은 다른 것을 재고, `lbm/gates.py` 는 이미 둘 다 갖고 있는데
뱅크 경로가 그걸 안 부른다 (`bank.json` 의 행에 `"gates": None` 이 그 자국이다).

board 경로(`build_candidate_board.py`)는 **시작 pose 한 장**에만 G1/G3 를 건다. 뱅크는
`start_mode=source_frame0` 이라 시작 pose 는 소스 카메라 자신이고 정의상 충돌이 없다 — 위험은
전부 **나머지 48프레임**에 있다. 그래서 이 스크립트는 궤적 전 프레임을 훑는다.

## 무엇을 재나

- `G1_behind` — 프레임 f 의 카메라 위치를 소스 프레임들에 되쏘아 `z_cam > depth + 0.02·S` 인
  프레임 수. 하나라도 있으면 그 프레임은 "표면 뒤". 렌더가 필요 없다 (재투영뿐).
- `G3_occlusion` — subject 를 **단독 렌더**한 depth 와 **전체 렌더**한 depth 를 비교해, 앞에
  다른 게 끼어든 subject 픽셀 비율. `gates.evaluate` 와 같은 방식이고 렌더가 2배 든다.
  `--no_occlusion` 으로 끄면 G1 만 (렌더 0회, 전 뱅크가 몇 초).

출력: `<out>/<video>/<bank_dir>/geometry.csv` + 요약표. **게이트가 아니라 감사(audit)다** —
행을 지우지 않고 열만 붙인다. 자를지 말지는 D39/D45 와 같은 이유로 소비처가 정한다.

env: `vista4d`

예시:
    CUDA_VISIBLE_DEVICES=1 python scripts/audit_bank_geometry.py --video camel
    CUDA_VISIBLE_DEVICES=1 python scripts/audit_bank_geometry.py --video camel \
        --bank_dir hole_bank --occlusion --occlusion_frames 7
"""
import csv
import json
import sys
from argparse import ArgumentParser
from os import path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import subject_point_mask                                        # noqa: E402
from lbm.gates import behind_profile                                            # noqa: E402
from lbm.presets import row_preset                                              # noqa: E402
from lbm.render import add_cloud_source_args, open_renderer                     # noqa: E402
from scene_graph.lift import apply_transform, project_points                    # noqa: E402
from scene_graph.obb import node_obb_at, obb_corners                            # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402
from scripts.build_candidate_board import subject_track_volume                  # noqa: E402


def occlusion_profile(renderer, poses: np.ndarray, subject_points, frames,
                      height: int, width: int, scale: float):
    """subject 단독 렌더 vs 전체 렌더의 depth 를 비교 — 앞을 가로막힌 subject 픽셀 비율.

    `gates.evaluate:100-112` 와 **같은 판정**이다: 실루엣 분모는 단독 렌더의 `valid`(가림이
    없었을 때 그려졌을 픽셀), 전체 렌더 depth 가 그보다 확실히 앞이면 가려진 것.
    낮을수록 나쁘다 (1.0 = 하나도 안 가림).
    """
    passes = []
    for f in frames:
        full = renderer.render(poses[f], frame=int(f), height=height, width=width)
        alone = renderer.render(poses[f], frame=int(f), height=height, width=width,
                                subset=subject_points)
        silhouette = alone["valid"]
        if not silhouette.any():
            continue                                   # 화면 밖 — 가림이 아니라 프레이밍 문제다
        occluded = (silhouette & full["valid"]) & (full["depth"] < alone["depth"] - 0.02 * scale)
        passes.append(1.0 - float(occluded.sum()) / float(silhouette.sum()))
    return float(np.mean(passes)) if passes else float("nan")


def obb_occlusion_profile(nodes: list, anchor_id: str, T_wg, poses: np.ndarray, K_src,
                          frames, height: int, width: int, src_height: int, src_width: int,
                          renderer=None):
    """subject OBB 실루엣 중 **더 가까운 다른 노드 OBB** 에 덮인 비율. 렌더가 0회다.

    왜 별도로 필요한가 (사용자 지적): `occlusion_profile` 은 점군 렌더 두 장을 비교하는데,
    **가리는 물체가 재구성이 안 됐으면 거기가 구멍**이고 그 구멍으로 subject 가 그대로
    비친다 → "안 가려짐"으로 읽힌다. 실제 생성 모델은 소스 영상에 있던 그 물체를 그릴
    가능성이 큰데도 그렇다. 즉 렌더 기반 판정은 **관측된 표면만** 가림으로 칠 수 있다.
    OBB 는 부피라 점이 없어도 남아 있으므로 그 구멍을 메운다 (D49 가 충돌에서 편 논리와 같다).

    반대 방향 편향이 있다는 걸 같이 알아야 한다: OBB 는 물체 실루엣보다 **크다** (의자 박스는
    대부분 공기다). 그래서 이 값은 가림을 **과대**, 렌더 값은 **과소** 보고한다 — 둘은 참값을
    사이에 두는 괄호지 대체재가 아니다. 그리고 노드가 아닌 것(벽·바닥·drop 된 track)은
    애초에 박스가 없어서 여기서도 안 잡힌다.

    깊이 정렬은 노드마다 **꼭짓점 z 중앙값 하나**로 painter's algorithm 을 돌린다 (먼 것부터
    칠하고 subject 픽셀이 남았나를 본다). 박스가 서로 뚫고 들어가거나 한 박스가 깊이 방향으로
    길면(avocado `stat_1 window` 는 3.39 u) 이 근사가 깨진다 — 그래서 판정이 아니라 감사다.

    `renderer` 를 주면 **결정적 진단** 하나를 더 낸다: OBB 가 "가려졌다"고 판정한 픽셀들이 렌더에서
    실제로 **구멍**(valid=False)인 비율 `hole_at_occluder`. 이게 높으면 가리는 물체가 재구성이 안
    된 것이므로 렌더 기반 판정이 구조적으로 못 보는 자리다 (= 위 문단의 가설이 참). 낮으면 그
    물체는 이미 점군에 있고 렌더가 이미 세고 있었다는 뜻이므로 OBB 는 그냥 부풀린 것이다.

    반환 `(pass_frac, 가장 많이 가린 노드 id, hole_at_occluder)`. 1.0 = 하나도 안 가림
    (`occlusion_pass` 와 방향 동일).
    """
    import cv2

    passes, blame, hole_at = [], {}, []
    for f in frames:
        K = np.asarray(K_src[int(f)], dtype=np.float64).copy()
        K[0] *= width / src_width                      # 렌더러(`render.py:136-139`)와 같은 배율
        K[1] *= height / src_height
        painted = np.full((height, width), -1, dtype=np.int32)
        hulls = []
        for index, node in enumerate(nodes):
            if node.get("moving"):
                center, extent, R = node_obb_at(node, int(f))
            else:
                obb = node["obb"]
                center, extent, R = (np.asarray(obb["center"], dtype=float),
                                     np.asarray(obb["extent"], dtype=float),
                                     np.asarray(obb["R"], dtype=float))
            corners = apply_transform(T_wg, obb_corners(center, extent, R))
            uv, z = project_points(corners, K, poses[int(f)])
            if (z > 1e-6).sum() < 8:
                continue                               # 일부라도 카메라 뒤면 투영이 뒤집힌다 — 뺀다
            hulls.append((float(np.median(z)), index,
                          cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))))
        subject_index = next((i for _, i, _ in hulls if nodes[i]["id"] == anchor_id), None)
        if subject_index is None:
            continue                                   # subject 가 화면 뒤 — 프레이밍 문제다
        for _, index, hull in sorted(hulls, key=lambda item: -item[0]):     # 먼 것부터
            cv2.fillConvexPoly(painted, hull.round().astype(np.int32), int(index))
        silhouette = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(silhouette, next(h for _, i, h in hulls if i == subject_index)
                           .round().astype(np.int32), 1)
        total = int(silhouette.sum())
        if total == 0:
            continue                                   # 화면 밖
        mask = silhouette.astype(bool)
        inside = painted[mask]
        passes.append(float((inside == subject_index).sum()) / total)
        for index, count in zip(*np.unique(inside[inside != subject_index], return_counts=True)):
            if index >= 0:
                blame[nodes[int(index)]["id"]] = blame.get(nodes[int(index)]["id"], 0) + int(count)
        if renderer is not None:
            covered = mask & (painted != subject_index) & (painted >= 0)
            if covered.any():
                valid = renderer.render(poses[int(f)], frame=int(f),
                                        height=height, width=width)["valid"]
                hole_at.append(1.0 - float(valid[covered].sum()) / float(covered.sum()))
    if not passes:
        return float("nan"), "", float("nan")
    return (float(np.mean(passes)), (max(blame, key=blame.get) if blame else ""),
            float(np.mean(hole_at)) if hole_at else float("nan"))


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    bank_folder = path.join(out_root, args.video, args.bank_dir)
    with open(path.join(bank_folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    cam_c2w_all = np.load(path.join(bank_folder, "poses.npz"), allow_pickle=False)["cam_c2w"]
    variants = bank["variants"]
    assert len(cam_c2w_all) == len(variants), \
        f"poses.npz {len(cam_c2w_all)} != bank.json {len(variants)} — 뱅크를 다시 만들 것"

    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    scale = float(graph["scale"]["S"])
    nodes = {n["id"]: n for n in graph["nodes"]}
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=float)

    renderer, recon = open_renderer(args, out_root, graph)
    src_frames = np.unique(np.linspace(0, renderer.num_frames - 1, args.behind_frames)
                           .round().astype(int)).tolist()
    occ_frames = np.unique(np.linspace(0, cam_c2w_all.shape[1] - 1, args.occlusion_frames)
                           .round().astype(int)).tolist()

    print(f"{'video':<16}{args.video}   bank {args.bank_dir}   변이 {len(variants)}")
    print(f"{'S':<16}{scale:.6f}   margin {args.margin_frac}·S   clear {args.clear_frac}·S   "
          f"patch r{args.radius_px}px   소스 프레임 {src_frames}")
    print(f"{'occlusion':<16}{'on ' + str(occ_frames) if args.occlusion else 'off (G1 만)'}")
    print(f"{'obb_occlusion':<16}{'on ' + str(occ_frames) if args.obb_occlusion else 'off'}\n")

    rows, subject_cache = [], {}
    for index, variant in enumerate(variants):
        bad, worst = behind_profile(cam_c2w_all[index], recon["depths"], K=renderer.K_src,
                                    cam_c2w=renderer.cam_c2w_src, sky_mask=recon["sky_mask"],
                                    scale=scale, frames=src_frames,
                                    margin_frac=args.margin_frac, clear_frac=args.clear_frac,
                                    radius_px=args.radius_px)
        occlusion = float("nan")
        if args.occlusion:
            # subject 마스크는 anchor 마다 다시 깔아야 한다 — 안 하면 직전 anchor 것이 남는다.
            anchor = variant["anchor_id"]
            if anchor not in subject_cache:
                subject_cache[anchor] = subject_point_mask(
                    renderer.indices, subject_track_volume(recon, nodes[anchor])).cpu().numpy()
            renderer.set_subject(subject_cache[anchor])
            occlusion = occlusion_profile(renderer, cam_c2w_all[index], subject_cache[anchor],
                                          occ_frames, args.tile_height, args.tile_width, scale)
        obb_occlusion, obb_occluder, hole_at = float("nan"), "", float("nan")
        if args.obb_occlusion:
            obb_occlusion, obb_occluder, hole_at = obb_occlusion_profile(
                graph["nodes"], variant["anchor_id"], T_wg, cam_c2w_all[index], renderer.K_src,
                occ_frames, args.tile_height, args.tile_width,
                renderer.height, renderer.width,
                renderer=renderer if args.obb_occlusion_diag else None)
        # 아래 preset 별 집계표가 이 열로 묶인다. 적힌 이름으로 묶으면 `straight_ease` 와
        # `dolly_in` 이 따로 서는데 실은 같은 카메라다 (§`lbm.presets.row_preset`).
        rows.append({"variant_id": variant["variant_id"], "anchor_id": variant["anchor_id"],
                     "preset": row_preset(variant), "preset_raw": variant["preset"],
                     "rung": variant.get("target_tau", variant.get("target_hole")),
                     "behind_frames": bad,
                     "behind_frac": round(bad / len(cam_c2w_all[index]), 4),
                     "behind_worst_src": worst,
                     "occlusion_pass": (round(occlusion, 4) if occlusion == occlusion else ""),
                     "obb_occl_pass": (round(obb_occlusion, 4)
                                       if obb_occlusion == obb_occlusion else ""),
                     "obb_occluder": obb_occluder,
                     "hole_at_occluder": (round(hole_at, 4) if hole_at == hole_at else ""),
                     "hole_fraction": variant["hole_fraction"],
                     "subject_in_frame": variant["subject_in_frame"]})

    output = path.join(bank_folder, "geometry.csv")
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    hit = [r for r in rows if r["behind_frames"] > 0]
    print(f"{'G1 위반 변이':<20}{len(hit)} / {len(rows)}   "
          f"({100.0 * len(hit) / len(rows):.1f}%)")
    if hit:
        print(f"\n{'preset':<18}{'n':>5}{'behind_frac med':>17}{'hole med':>10}")
        for preset in sorted({r["preset"] for r in hit}):
            sub = [r for r in hit if r["preset"] == preset]
            print(f"{preset:<18}{len(sub):>5}"
                  f"{np.median([r['behind_frac'] for r in sub]):>17.3f}"
                  f"{np.median([r['hole_fraction'] for r in sub]):>10.3f}")
        print(f"\n{'anchor':<10}{'n':>5}{'behind_frac max':>17}")
        for anchor in sorted({r["anchor_id"] for r in hit}):
            sub = [r for r in hit if r["anchor_id"] == anchor]
            print(f"{anchor:<10}{len(sub):>5}{max(r['behind_frac'] for r in sub):>17.3f}")
    if args.occlusion:
        vals = [r["occlusion_pass"] for r in rows if r["occlusion_pass"] != ""]
        low = [v for v in vals if v < args.min_occlusion_pass]
        print(f"\n{'occlusion_pass':<20}median {np.median(vals):.3f}   "
              f"< {args.min_occlusion_pass}: {len(low)} / {len(vals)}")
    if args.obb_occlusion:
        vals = [r["obb_occl_pass"] for r in rows if r["obb_occl_pass"] != ""]
        low = [v for v in vals if v < args.min_occlusion_pass]
        print(f"{'obb_occl_pass':<20}median {np.median(vals):.3f}   "
              f"< {args.min_occlusion_pass}: {len(low)} / {len(vals)}")
        blame = {}
        for row in rows:
            if row["obb_occluder"]:
                blame[row["obb_occluder"]] = blame.get(row["obb_occluder"], 0) + 1
        if blame:
            print(f"\n{'가장 많이 가린 노드':<22}{'변이 수':>8}")
            for node_id, count in sorted(blame.items(), key=lambda kv: -kv[1]):
                label = nodes[node_id]["label"] if node_id in nodes else "?"
                print(f"{node_id + ' ' + label:<22}{count:>8}")
    if args.occlusion and args.obb_occlusion:
        # 두 측정의 **불일치**가 이 감사의 요점이다 (D54). 렌더는 과소, OBB 는 과대 보고한다.
        pair = [(r["occlusion_pass"], r["obb_occl_pass"]) for r in rows
                if r["occlusion_pass"] != "" and r["obb_occl_pass"] != ""]
        thr = args.min_occlusion_pass
        both_ok = sum(1 for a, b in pair if a >= thr and b >= thr)
        both_bad = sum(1 for a, b in pair if a < thr and b < thr)
        render_only = sum(1 for a, b in pair if a >= thr > b)      # 구멍이 가림을 숨긴 후보
        obb_only = sum(1 for a, b in pair if b >= thr > a)
        print(f"\n{'일치/불일치 (thr ' + str(thr) + ')':<34}{'n':>6}")
        print(f"{'둘 다 통과':<34}{both_ok:>6}")
        print(f"{'둘 다 실패':<34}{both_bad:>6}")
        print(f"{'렌더만 통과 (OBB 가 가림이라 함)':<34}{render_only:>6}")
        print(f"{'OBB 만 통과 (렌더가 가림이라 함)':<34}{obb_only:>6}")
        a = np.array([p[0] for p in pair])
        b = np.array([p[1] for p in pair])
        print(f"{'corr(render, obb)':<34}{float(np.corrcoef(a, b)[0, 1]):>6.3f}")
        if args.obb_occlusion_diag:
            # 불일치 그룹별 hole_at_occluder — 높으면 "구멍이 가림을 숨겼다"가 참이다.
            print(f"\n{'그룹':<34}{'n':>6}{'hole_at_occluder med':>22}")
            for name, keep in (("둘 다 통과", lambda x, y: x >= thr and y >= thr),
                               ("둘 다 실패", lambda x, y: x < thr and y < thr),
                               ("렌더만 통과 (OBB 가 가림)", lambda x, y: x >= thr > y),
                               ("OBB 만 통과 (렌더가 가림)", lambda x, y: y >= thr > x)):
                sub = [r["hole_at_occluder"] for r in rows
                       if r["occlusion_pass"] != "" and r["obb_occl_pass"] != ""
                       and r["hole_at_occluder"] != ""
                       and keep(r["occlusion_pass"], r["obb_occl_pass"])]
                if sub:
                    print(f"{name:<34}{len(sub):>6}{np.median(sub):>22.3f}")

    print(f"\n-> {output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)
    add_cloud_source_args(parser)

    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    #    **기본 True.** 뱅크 pose 는 frame0 고정 K 로 풀렸으므로 감사도 같은 K 로 재야 한다.
    #    프레임별 DA3 K 를 쓰면 화각이 떨려(snowboard fx 진폭 6.99%) hole/framing 이 궤적과
    #    무관한 이유로 흔들린다. `--no_fixed_focal` 은 예전 동작(프레임별 K)이다.
    parser.add_argument("--fixed_focal", action="store_true", default=True)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    # `bank` = τ 사다리, `hole_bank` = hole 사다리.
    parser.add_argument("--bank_dir", default="bank", type=str)

    # ── G1: 몇 개의 소스 프레임에 되쏘아 볼지. board 경로 기본값과 같다.
    parser.add_argument("--behind_frames", default=7, type=int)
    parser.add_argument("--margin_frac", default=0.02, type=float)   # 관통을 봐주는 여유
    # 표면 앞에 요구하는 여유 / 패치 최소 depth 반경. 기본 0 = 예전 판정 그대로 (감사는 감사다).
    parser.add_argument("--clear_frac", default=0.0, type=float)
    parser.add_argument("--radius_px", default=0, type=int)
    # ── G3: 렌더 2회/프레임이라 기본은 꺼둔다.
    parser.add_argument("--occlusion", action="store_true", default=False)
    parser.add_argument("--no_occlusion", dest="occlusion", action="store_false")
    parser.add_argument("--occlusion_frames", default=5, type=int)
    # OBB 기반 가림 (렌더 0회). 렌더 기반과 **편향 방향이 반대**라 같이 켜서 괄호로 쓴다.
    parser.add_argument("--obb_occlusion", action="store_true", default=False)
    parser.add_argument("--no_obb_occlusion", dest="obb_occlusion", action="store_false")
    # OBB 가 가렸다고 한 자리가 렌더에선 구멍인가 — 두 측정이 왜 갈리는지의 결정적 진단. 렌더 1회/프레임.
    parser.add_argument("--obb_occlusion_diag", action="store_true", default=False)
    parser.add_argument("--no_obb_occlusion_diag", dest="obb_occlusion_diag",
                        action="store_false")
    parser.add_argument("--min_occlusion_pass", default=0.40, type=float)
    parser.add_argument("--tile_width", default=640, type=int)
    parser.add_argument("--tile_height", default=360, type=int)
    main(parser.parse_args())
