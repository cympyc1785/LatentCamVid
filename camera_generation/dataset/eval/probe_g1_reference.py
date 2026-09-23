"""G1(behind-surface) 임계의 **기준 거리**를 고른다 — 씬 상수 `S` 대신 뭘 써야 하나.

왜 필요한가 (D47). G5/G6/G7 은 전부 임계를 "소스 카메라 **자신의** 여유 × β(<1)" 로 잡아서
소스 카메라가 정의상 자기 게이트를 통과한다. G1 만 `clear_frac·S` 라는 **씬 상수의 절대 분수**를
쓴다. 실측(parkour)에서 소스 카메라는 `clear_frac 0.07` 부터 자기 게이트를 위반하고, 배포값
0.10 에서 1/49 프레임을 위반한다 — 소스 카메라를 기각하는 임계는 충돌 판정이 아니라 버그다.

그래서 두 가지를 같이 잰다:

  ① **G1 자체 게이지** `g1_src` — 소스 카메라 자신을 G1 **그 metric 으로** 재서 얻는 여유.
     3D 최단거리(`render.standoff`)가 아니라 **z-depth 차** `z_surf − z_cam` 이다. G1 이 재는 게
     그거라서 (`gates.behind_surface_frames:118`), 임계의 바닥도 같은 자로 재야 한다. 두 양은
     실제로 다르다 — parkour 에서 3D 최근접(0.0326 u)은 G1 이 읽는 값(0.0674 u) 옆으로 크게
     비껴 있었다.
     제안: `clear_frac = margin_frac + β · g1_src`. β<1 이면 소스가 정의상 통과한다.

  ② **다른 거리들과의 정합** (사용자 질문 2026-09-04): 소스 카메라 거리 / 최소 scene-camera
     거리 / 피사체까지 거리 / 피사체 shot scale 중에 일관적으로 기준이 될 조합이 있나.
     판정 기준은 `node_margins` 의 D51 분석과 같다 — 후보 R 로 나눈 비율 `g1_src / R` 이
     **씬 사이에서 안 흔들리면** 그 R 이 게이지다. 변동계수(CV)로 순위를 매긴다.

주의: `g1_src` 는 `time_match=True` 배포 설정과 맞추려고 **static 채널**로 잰다. 동적 채널은
소스 pose 에서 항상 공집합이다 (플랜 프레임 f ↔ 소스 프레임 f 는 z_cam=0 이라 스킵된다).

사용 예시:
  python eval/probe_g1_reference.py --videos parkour camel avocado-slice snowboard
  python eval/probe_g1_reference.py --num_videos 14 --beta 0.3
"""
import sys
from argparse import ArgumentParser
from os import listdir, path

import numpy as np

ROOT = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT     # noqa: E402
from lbm.gates import obb_clearance                               # noqa: E402
from lbm.render import CloudRenderer                              # noqa: E402
from scene_graph.io import load_scene                             # noqa: E402
from scene_graph.schema import load_graph                         # noqa: E402


def g1_gaps(poses, depths, K, cam_c2w, sky_mask, scale, frames, radius_px, dynamic_mask):
    """pose 별 `min_t (z_surf − z_cam)/S` (static 채널). 반환 (len(poses),).

    `behind_surface_frames` 의 루프를 **그대로** 옮겼다 (투영·패치·usable 판정 동일). 다른 점은
    임계 비교(`cam[2] + clear > z + margin`) 대신 그 양변의 차를 돌려준다는 것뿐이다 — 그래야
    "임계를 얼마로 잡으면 소스가 걸리나"를 스윕 없이 한 번에 안다.

    **왜 pose 별로 min 을 먼저 잡나.** G1 은 플랜 프레임 하나가 소스 프레임 **아무거나** 하나에
    걸리면 그 프레임을 기각한다 (`behind_profile` 이 `len(hits)>0` 을 센다). 그러니 판정량은
    (pose, frame) 쌍이 아니라 **pose 당 최솟값**이다. 쌍 전체로 백분위를 내면 "카메라 앞이
    멀리 트여 있는" 대다수 쌍이 분포를 지배해서 (parkour 실측 p01 이 1.13 u, min 의 25배)
    임계 바닥과 무관한 수가 나온다. `render.standoff` 가 pose 당 한 값인 것과 같은 규약이다.
    """
    height, width = depths.shape[-2:]
    out = []
    for f, pose in enumerate(poses):
        p = np.asarray(pose[:3, 3], dtype=float)
        best = float("inf")
        for t in frames:
            w2c = np.linalg.inv(cam_c2w[t])
            cam = w2c[:3, :3] @ p + w2c[:3, 3]
            if cam[2] <= 1e-6:                       # 자기 자신 / 뒤쪽 — G1 도 안 본다
                continue
            uv = (K[t] @ cam)[:2] / cam[2]
            u, v = int(np.floor(uv[0])), int(np.floor(uv[1]))
            if not (0 <= u < width and 0 <= v < height):
                continue
            r = max(int(radius_px), 0)
            u0, u1 = max(u - r, 0), min(u + r + 1, width)
            v0, v1 = max(v - r, 0), min(v + r + 1, height)
            patch = depths[t][v0:v1, u0:u1]
            usable = np.isfinite(patch) & (patch > 0) & ~sky_mask[t][v0:v1, u0:u1]
            if dynamic_mask is not None:
                usable &= ~dynamic_mask[t][v0:v1, u0:u1]
            if not usable.any():
                continue
            best = min(best, (float(patch[usable].min()) - float(cam[2])) / scale)
        if np.isfinite(best):
            out.append(best)
    return np.asarray(out, dtype=float)


def measure(video, args):
    """영상 1편의 후보 기준거리 전부. 전부 u 단위 (1 u ≜ S DA3)."""
    out_root = path.join(ROOT, "out")
    graph = load_graph(path.join(out_root, video, "scene_graph.json"))
    scale = float(graph["scale"]["S"])
    renderer = CloudRenderer(path.join(out_root, video, "cloud.npz"),
                             vista4d_root=args.vista4d_root, device=args.device,
                             fixed_focal=True)
    recon = load_scene(args.eval_data, video, args.vista4d_root)
    frames = np.unique(np.linspace(0, renderer.num_frames - 1, args.behind_src_frames)
                       .round().astype(int)).tolist()
    src = np.asarray(renderer.cam_c2w_src, dtype=float)

    gaps = g1_gaps(src, recon["depths"], renderer.K_src, src, recon["sky_mask"],
                   scale, frames, args.radius_px, recon["dynamic_mask"].astype(bool))

    # ── 후보 기준거리들.
    stand3d = np.asarray(renderer.standoff(src), dtype=float) / scale     # 3D 최단거리
    depth0 = np.asarray(recon["depths"][0], dtype=float)
    sky0 = np.asarray(recon["sky_mask"][0], dtype=bool)
    ok0 = np.isfinite(depth0) & (depth0 > 0) & ~sky0
    z_med0 = float(np.median(depth0[ok0])) / scale
    near_all = []                                # 소스가 본 가장 가까운 표면 (방향 무관)
    for t in frames:
        d = np.asarray(recon["depths"][t], dtype=float)
        s = np.asarray(recon["sky_mask"][t], dtype=bool)
        m = np.isfinite(d) & (d > 0) & ~s
        if m.any():
            near_all.append(float(d[m].min()) / scale)
    near_src = float(min(near_all)) if near_all else float("nan")

    nodes = graph["nodes"]
    T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
    dists, ids, _ = obb_clearance(src, nodes, T_gw)
    obb_src = float(np.min(dists))

    # 피사체 = 동적 노드 중 가장 큰 것 (없으면 전체 중 가장 큰 것).
    dyn = [n for n in nodes if n.get("moving")] or nodes
    subj = max(dyn, key=lambda n: float(np.max(n["obb"]["extent"])))
    subj_ext = float(np.max(np.asarray(subj["obb"]["extent"], dtype=float)))
    d_ref = float(subj.get("viewing_distance", {}).get("d_ref", float("nan")))
    if not np.isfinite(d_ref):                   # 그래프에 없으면 소스↔OBB center 거리로
        c_g = np.asarray(subj["obb"]["center"], dtype=float)
        cam_g = src[:, :3, 3] @ T_gw[:3, :3].T + T_gw[:3, 3]
        d_ref = float(np.median(np.linalg.norm(cam_g - c_g, axis=1)))

    srt = np.sort(gaps)
    out = {
        "video": video, "S": scale, "n_gap": len(gaps), "gaps_low5": srt[:5].tolist(),
        "g1_min": float(srt[0]), "g1_2nd": float(srt[1] if len(srt) > 1 else srt[0]),
        "g1_p10": float(np.percentile(gaps, 10)),
        "g1_p25": float(np.percentile(gaps, 25)), "g1_p50": float(np.median(gaps)),
        "stand3d_min": float(stand3d.min()), "stand3d_p50": float(np.median(stand3d)),
        "near_src": near_src, "z_med0": z_med0, "obb_src": obb_src,
        "d_ref": d_ref, "subj_ext": subj_ext,
        "shot_scale": subj_ext / d_ref if d_ref > 0 else float("nan"),
    }
    #    조합 후보 (사용자 질문: "거리 조합이 있는지"). 단일 후보가 전부 CV~1 로 실패하므로
    #    기하평균 · 최솟값 · 곱 형태를 같이 건다. 이름의 `x` 는 곱, `gm` 은 기하평균이다.
    out["gm_3d_dref"] = float(np.sqrt(out["stand3d_min"] * out["d_ref"]))
    out["gm_3d_zmed"] = float(np.sqrt(out["stand3d_min"] * out["z_med0"]))
    out["gm_near_dref"] = float(np.sqrt(out["near_src"] * out["d_ref"]))
    out["min_3d_obb"] = float(min(out["stand3d_min"], out["obb_src"]))
    out["dref_x_shot"] = float(out["d_ref"] * out["shot_scale"])       # = subj_ext
    out["zmed_x_shot"] = float(out["z_med0"] * out["shot_scale"])
    return out


def cv_table(rows, target, cands):
    """`target / cand` 비율의 변동계수 — 낮을수록 씬 사이에서 일관된 게이지다."""
    print(f"\n② 기준거리 후보 — `{target}` 를 후보로 나눈 비율의 씬간 일관성 "
          f"(CV = sd/mean, 낮을수록 좋다)")
    print(f"{'후보 기준거리 R':<26}{'CV':>8}{'mean(g1/R)':>12}{'min':>10}{'max':>10}"
          f"{'max/min':>9}")
    print("-" * 75)
    scored = []
    for name in cands:
        ratio = np.asarray([r[target] / r[name] for r in rows
                            if np.isfinite(r[name]) and r[name] != 0], dtype=float)
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) < 2:
            continue
        cv = float(np.std(ratio) / abs(np.mean(ratio)))
        scored.append((cv, name, ratio))
    for cv, name, ratio in sorted(scored):
        print(f"{name:<26}{cv:>8.3f}{np.mean(ratio):>12.4f}{ratio.min():>10.4f}"
              f"{ratio.max():>10.4f}{ratio.max() / max(ratio.min(), 1e-12):>9.2f}")
    return scored


def main(args):
    out_root = path.join(ROOT, "out")
    videos = args.videos
    if not videos:
        avail = sorted(v for v in listdir(out_root)
                       if path.isfile(path.join(out_root, v, "cloud.npz"))
                       and path.isfile(path.join(out_root, v, "scene_graph.json")))
        step = max(len(avail) // args.num_videos, 1)
        videos = avail[::step][:args.num_videos]

    rows = []
    for video in videos:
        try:
            rows.append(measure(video, args))
        except Exception as exc:                                  # noqa: BLE001
            print(f"  !! {video}: {type(exc).__name__} {exc}")
    if not rows:
        return

    print(f"\n① 씬별 실측 (전부 u 단위, 1 u ≜ S DA3).  margin_frac {args.margin_frac}  "
          f"radius_px {args.radius_px}  소스프레임 {args.behind_src_frames}장")
    print("  g1_* = pose 별 min_t (z_surf − z_cam) 위에서 낸 통계 (소스 pose 49개)")
    print(f"{'video':<18}{'S':>8}{'g1_min':>9}{'g1_p10':>9}{'g1_p25':>9}{'g1_p50':>9}"
          f"{'3d_min':>9}{'near':>8}{'obb':>8}{'d_ref':>8}{'shot':>8}{'z_med0':>8}")
    print("-" * 121)
    for r in rows:
        print(f"{r['video'][:17]:<18}{r['S']:>8.3f}{r['g1_min']:>9.4f}{r['g1_p10']:>9.4f}"
              f"{r['g1_p25']:>9.4f}{r['g1_p50']:>9.4f}{r['stand3d_min']:>9.4f}"
              f"{r['near_src']:>8.4f}{r['obb_src']:>8.4f}{r['d_ref']:>8.4f}"
              f"{r['shot_scale']:>8.4f}{r['z_med0']:>8.4f}")

    # ── 배포 임계가 소스를 기각하나.
    print(f"\n③ 배포 임계 `clear_frac {args.clear_frac}` 의 D47 legality "
          f"(위반 = clear_frac − margin_frac > g1_min)")
    print(f"{'video':<18}{'실효요구':>10}{'g1_min':>9}{'소스 위반?':>11}"
          f"{'소스가 견디는 최대 clear_frac':>28}")
    print("-" * 78)
    eff = args.clear_frac - args.margin_frac
    n_bad = 0
    for r in rows:
        bad = eff > r["g1_min"]
        n_bad += bad
        print(f"{r['video'][:17]:<18}{eff:>10.4f}{r['g1_min']:>9.4f}"
              f"{('위반' if bad else 'ok'):>11}{args.margin_frac + r['g1_min']:>28.4f}")
    print(f"  → {n_bad} / {len(rows)} 편에서 **소스 카메라가 자기 게이트를 위반**한다")

    print("\n①-b `g1_min` 이 pose 1개짜리 이상치인가 — 가장 낮은 소스 pose 5개 (u)")
    print(f"{'video':<18}{'낮은 5개':<44}{'2nd/min':>9}{'p10/min':>9}")
    print("-" * 80)
    for r in rows:
        low = "  ".join(f"{v:.4f}" for v in r["gaps_low5"])
        print(f"{r['video'][:17]:<18}{low:<44}"
              f"{r['g1_2nd'] / max(r['g1_min'], 1e-9):>9.2f}"
              f"{r['g1_p10'] / max(r['g1_min'], 1e-9):>9.2f}")

    cands = ["g1_min", "g1_2nd", "g1_p10", "g1_p25", "g1_p50", "stand3d_min", "stand3d_p50",
             "near_src", "obb_src", "d_ref", "shot_scale", "z_med0", "S",
             "gm_3d_dref", "gm_3d_zmed", "gm_near_dref", "min_3d_obb",
             "dref_x_shot", "zmed_x_shot"]
    # `S` 는 u 단위로 항상 1 이라 비율 게이지로는 무의미하다 — 대신 "절대 분수" 자체를 본다.
    cv_table(rows, "g1_min", [c for c in cands if c not in ("g1_min", "S")])

    print(f"\n④ 제안 규칙 `clear_frac = margin_frac + β·R`, β = {args.beta}")
    print(f"{'R':<16}{'씬별 clear_frac':>44}{'현행 0.10 대비':>16}")
    print("-" * 78)
    for name in ("g1_min", "g1_2nd", "g1_p10", "g1_p25", "stand3d_min", "obb_src", "d_ref"):
        vals = [args.margin_frac + args.beta * r[name] for r in rows]
        loosen = sum(1 for v in vals if v > args.clear_frac)
        print(f"{name:<16}"
              + f"[{min(vals):.4f}, {max(vals):.4f}] med {np.median(vals):.4f}".rjust(44)
              + f"완화 {len(vals) - loosen} / 강화 {loosen}".rjust(16))
    print("\n  (β<1 이면 R=g1_* 계열은 소스가 **정의상** 통과한다 — 다른 R 은 보장이 없다)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="*", default=[])           # 비우면 자동 샘플링
    parser.add_argument("--num_videos", default=12, type=int)        # 자동 샘플링 편수
    parser.add_argument("--behind_src_frames", default=49, type=int)  # 되쏘아 볼 소스 프레임 수
    parser.add_argument("--margin_frac", default=0.02, type=float)   # 관통을 봐주는 여유
    parser.add_argument("--clear_frac", default=0.10, type=float)    # 현행 배포 임계
    parser.add_argument("--radius_px", default=2, type=int)          # 패치 반경
    parser.add_argument("--beta", default=0.3, type=float)           # 소스 대비 배수 (G5 와 동일)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    main(parser.parse_args())
