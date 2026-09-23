"""**τ 의 분모를 뭘로 잡을 것인가**를 실측한다 (조사용, 파이프라인에 안 걸려 있다).

지금 τ = |p_plan(f) − p_src(f)| / `z_med_frame0` 이다. 분모가 **frame0 의 z-depth 중앙값**이라
두 군데가 자의적이다:

  ① **z-depth 냐 점까지의 거리냐.** z 는 광축 성분만이라 화면 가장자리 점을 과소평가한다.
     같은 씬을 화각만 넓혀 찍으면 z_med 가 내려간다 — 씬이 바뀌지도 않았는데 τ 눈금이 바뀐다.
     `scene_graph/scale.py:scene_scale` 의 `S` 는 이미 ray length(=점까지 거리) 로 정의돼 있어
     이 문제가 없다. 즉 리포 안에 **거리 게이지가 이미 두 개** 있고 서로 다른 양이다.
  ② **frame0 한 장이냐 소스 전 프레임이냐.** frame0 이 우연히 가까운 물체를 크게 잡았으면
     분모가 작아지고 τ 예산이 통째로 쪼그라든다. 49프레임을 다 쓰면 그 우연이 씻긴다.

이게 왜 중요한가: τ 는 **씬 간 비교 가능한 강도 축**이어야 한다 (`tau-is-one-shared-budget`).
분모가 씬 고유의 성질이 아니라 촬영 우연을 타면 "camel τ 0.5" 와 "avocado τ 0.5" 가 다른 뜻이
된다. 그래서 여섯 후보를 같은 뱅크에 얹어보고 **어느 분모가 두 씬의 τ 눈금을 맞추는지**를 본다.

판정 기준은 분모 값 자체가 아니라 이것 하나다:

    hole 사다리의 같은 단(rung)에서 두 씬의 τ 가 같은 값으로 나오나?

사다리는 hole 을 고정하고 크기를 푼다 — 즉 **눈에 보이는 변화량이 같아지도록** 이분법이 맞춘
지점이다. 그 지점에서 τ 가 씬마다 다르면 그 분모는 강도 축으로 못 쓴다. 두 씬의 τ 비(ratio)가
1 에 가까울수록 좋은 분모다.

사용 예시:
    python eval/probe_tau_divisor.py
    python eval/probe_tau_divisor.py --videos camel avocado-slice --stride 2
"""
from argparse import ArgumentParser
from os import path
import csv
import json
import sys

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scene_graph.io import load_scene                                          # noqa: E402

CLOUD_ROOT = HERE
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
VISTA4D_ROOT_DEFAULT = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"

# 분모 후보. (키, 설명) — 계산은 `divisors()` 안에서 한다.
DIVISORS = (
    ("z_med_f0", "frame0 z-depth 중앙값 (현재 τ 분모)"),
    ("ray_med_f0", "frame0 점까지 거리 중앙값"),
    ("ray_mean_f0", "frame0 점까지 거리 평균 (= S, u 단위의 정의)"),
    ("z_med_all", "전 프레임 z-depth 중앙값"),
    ("ray_med_all", "전 프레임 점까지 거리 중앙값"),
    ("ray_mean_all", "전 프레임 점까지 거리 평균"),
    ("ray_med_all_static", "전 프레임 점까지 거리 중앙값 (동적 픽셀 제외)"),
    ("cloud_med_from_f0", "전 프레임 점을 frame0 카메라에서 잰 거리 중앙값"),
    ("cloud_med_from_cammed", "전 프레임 점을 **카메라 위치 median** 에서 잰 거리 중앙값"),
)


def ray_length_map(intrinsic: np.ndarray, height: int, width: int):
    """‖K⁻¹[u+0.5, v+0.5, 1]‖ (h,w). z-depth 를 점까지의 거리로 바꾸는 배율."""
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], axis=-1).astype(np.float64)
    return np.linalg.norm(pixels @ np.linalg.inv(intrinsic).T.astype(np.float64), axis=-1)


def divisors(recon: dict, stride: int):
    """여덟 후보 분모를 전부 계산한다 (DA3 단위). 전 프레임 통계는 `stride` 로 솎아 쓴다."""
    depths = np.asarray(recon["depths"], dtype=np.float64)
    sky = np.asarray(recon["sky_mask"], dtype=bool)
    dyn = np.asarray(recon["dynamic_mask"], dtype=bool)
    intrinsics = np.asarray(recon["K"], dtype=np.float64)
    cam_c2w = np.asarray(recon["cam_c2w"], dtype=np.float64)
    num_frames, height, width = depths.shape[-3:]

    # frame0
    rl0 = ray_length_map(intrinsics[0], height, width)
    z0 = depths[0]
    ok0 = np.isfinite(z0) & (z0 > 0) & ~sky[0]
    out = {"z_med_f0": float(np.median(z0[ok0])),
           "ray_med_f0": float(np.median((z0 * rl0)[ok0])),
           "ray_mean_f0": float((z0 * rl0)[ok0].mean())}

    # 전 프레임 pooled. 프레임마다 K 가 다를 수 있으니 배율 맵도 프레임마다 다시 만든다.
    z_all, ray_all, ray_static, from_f0, from_cmed = [], [], [], [], []
    center0 = cam_c2w[0][:3, 3]
    # 사용자가 짚은 후보: **카메라 위치 median** 을 원점으로 잡고 거기서 점 거리 중앙값.
    # frame0 대신 궤적 중앙을 쓰므로 첫 프레임이 치우친 씬에서 더 나을 수 있다.
    center_med = np.median(cam_c2w[:, :3, 3], axis=0)
    for frame in range(num_frames):
        rlm = ray_length_map(intrinsics[frame], height, width)[::stride, ::stride]
        z = depths[frame][::stride, ::stride]
        ok = np.isfinite(z) & (z > 0) & ~sky[frame][::stride, ::stride]
        if not ok.any():
            continue
        ray = (z * rlm)[ok]
        z_all.append(z[ok])
        ray_all.append(ray)
        static = ok & ~dyn[frame][::stride, ::stride]
        if static.any():
            ray_static.append((z * rlm)[static])
        # 같은 점을 **frame0 카메라에서** 재본다. 카메라가 거의 안 움직이면 위와 같아진다.
        vv, uu = np.nonzero(ok)
        pix = np.stack([uu * stride + 0.5, vv * stride + 0.5, np.ones(len(uu))], axis=-1)
        dirs = pix @ np.linalg.inv(intrinsics[frame]).T
        cam_pts = dirs * z[ok][:, None]
        world = cam_pts @ cam_c2w[frame][:3, :3].T + cam_c2w[frame][:3, 3]
        from_f0.append(np.linalg.norm(world - center0, axis=-1))
        from_cmed.append(np.linalg.norm(world - center_med, axis=-1))

    out["cloud_med_from_cammed"] = float(np.median(np.concatenate(from_cmed)))
    out["z_med_all"] = float(np.median(np.concatenate(z_all)))
    out["ray_med_all"] = float(np.median(np.concatenate(ray_all)))
    out["ray_mean_all"] = float(np.concatenate(ray_all).mean())
    out["ray_med_all_static"] = float(np.median(np.concatenate(ray_static)))
    out["cloud_med_from_f0"] = float(np.median(np.concatenate(from_f0)))
    return out


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    scenes = {}
    for video in args.videos:
        with open(path.join(out_root, video, "scene_graph.json"), encoding="utf-8") as file:
            graph = json.load(file)
        recon = load_scene(args.eval_data, video, args.vista4d_root,
                           seg_root=args.seg_root, seg_static_root=args.seg_static_root)
        div = divisors(recon, args.stride)
        bank_csv = path.join(out_root, video, "hole_bank", "bank.csv")
        rows = list(csv.DictReader(open(bank_csv, encoding="utf-8"))) if path.isfile(bank_csv) \
            else []
        scenes[video] = {"graph": graph, "div": div, "rows": rows,
                         "S": float(graph["scale"]["S"]),
                         "z_med": float(graph["scale"]["z_med_frame0"])}
        # 2026-09-02 부터 S 의 기본 정의가 `points_first_cam` 이라 `ray_mean_f0` 와 같지 않다.
        # 옛 게이지(`frame0_ray`)로 구운 그래프에서만 이 항등식이 성립한다. `mode` 키가 아예 없으면
        # 정의 변경 전 그래프라는 뜻이므로 옛 게이지로 본다.
        if graph["scale"].get("mode", "frame0_ray") == "frame0_ray":
            assert abs(div["ray_mean_f0"] - scenes[video]["S"]) < 1e-6 * scenes[video]["S"], \
                "ray_mean_f0 는 정의상 S 와 같아야 한다 (scene_graph/scale.py:scene_scale)"
        assert abs(div["z_med_f0"] - scenes[video]["z_med"]) < 1e-6 * scenes[video]["z_med"], \
            "z_med_f0 는 정의상 graph 의 z_med_frame0 와 같아야 한다"

    # ── 1. 분모 값 자체.
    head = f"{'분모':<22}" + "".join(f"{v:>16}" for v in args.videos) + f"{'설명':>4}"
    print(head)
    print("-" * 78)
    for key, desc in DIVISORS:
        line = f"{key:<22}" + "".join(f"{scenes[v]['div'][key]:>16.4f}" for v in args.videos)
        print(line + "   " + desc)

    # 현재 분모 대비 배수 — τ 눈금이 몇 배로 바뀌는지가 이 표다.
    print(f"\n{'분모':<22}" + "".join(f"{v + ' x z_med':>16}" for v in args.videos)
          + "   (τ 는 이 값의 **역수**배로 바뀐다)")
    print("-" * 78)
    for key, _ in DIVISORS:
        print(f"{key:<22}"
              + "".join(f"{scenes[v]['div'][key] / scenes[v]['div']['z_med_f0']:>16.4f}"
                        for v in args.videos))

    # ── 2. 판정. hole 사다리의 같은 단에서 두 씬의 τ 가 맞나.
    #    사다리는 hole 을 고정하고 크기를 푼다 = 눈에 보이는 변화량을 같게 맞춘 지점이다.
    #    거기서 τ 가 씬마다 다르면 그 분모는 강도 축이 아니다.
    # 단 키는 `hole_delta` 다. `target_hole` 은 `hole_mode=excess` 라 anchor 의 정지 hole 이
    # 더해진 **절대값**이고 anchor 마다 달라서 씬 간 대조에 못 쓴다.
    def rung_of(row):
        return float(row["hole_delta"] or row["target_hole"])

    rungs = sorted({round(rung_of(r), 4) for v in args.videos for r in scenes[v]["rows"]})
    common = sorted(set.intersection(*[{r["preset"] for r in scenes[v]["rows"]}
                                       for v in args.videos]))
    print(f"\n{'판정':<10}hole 사다리 같은 단에서의 τ 중앙값 (binding=hole, status=solved 만)")
    print(f"{'':<10}공통 preset {len(common)}개, 단 {rungs}")
    def cell_tau(video, rung, key):
        """(중앙값 τ, 표본수). 게이트가 먼저 문 변이는 사다리 단에 못 닿았으니 뺀다."""
        scene = scenes[video]
        sel = [r for r in scene["rows"]
               if abs(rung_of(r) - rung) < 1e-4
               and r["binding"] == "hole" and r["status"] == "solved"
               and r["preset"] in common and float(r["tau_max"]) > 0]
        if not sel:
            return float("nan"), 0
        # tau_max 는 z_med 분모로 재둔 값이다 → 변위(DA3) 로 되돌린 뒤 새 분모로 나눈다.
        tau = [float(r["tau_max"]) * scene["div"]["z_med_f0"] / scene["div"][key] for r in sel]
        return float(np.median(tau)), len(sel)

    first, second = args.videos[0], args.videos[1]
    print(f"{'':<10}표본수 (단별, binding=hole & solved & 공통 preset): "
          + "  ".join(f"{r:g}→{cell_tau(first, r, 'z_med_f0')[1]}/"
                      f"{cell_tau(second, r, 'z_med_f0')[1]}" for r in rungs))
    head2 = (f"{'분모':<22}" + "".join(f"{'단 ' + f'{r:g}':>18}" for r in rungs)
             + f"{'비 기하평균':>13}{'비 산포':>10}")
    print(head2)
    print("-" * len(head2))
    for key, _ in DIVISORS:
        cells, ratios = [], []
        for rung in rungs:
            a, _ = cell_tau(first, rung, key)
            b, _ = cell_tau(second, rung, key)
            cells.append(f"{a:.3f}/{b:.3f}")
            if np.isfinite([a, b]).all() and min(a, b) > 0:
                ratios.append(a / b)               # **부호 있는** 비 (max/min 이 아니다)
        geo = float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
        spread = (max(ratios) / min(ratios)) if ratios else float("nan")
        print(f"{key:<22}" + "".join(f"{c:>18}" for c in cells)
              + f"{geo:>13.3f}{spread:>10.2f}")

    print(f"\n{'주 1':<6}'비 기하평균' = {first} τ / {second} τ 를 단마다 구해 기하평균 낸 것.")
    print(f"{'':<6}1.000 이면 두 씬의 τ 눈금이 평균적으로 같다. 분모를 바꾸면 이 값만 움직인다.")
    print(f"{'주 2':<6}'비 산포' = 단별 비의 최대/최소. **분모에 완전히 불변이다** — 스칼라 분모는")
    print(f"{'':<6}한 씬의 τ 전부에 같은 배수라 단 사이의 어긋남을 못 고친다. 이 값이 1 에서 멀면")
    print(f"{'':<6}어떤 스칼라 분모로도 두 씬을 맞출 수 없다는 뜻이다.")
    ideal = {}
    for key, _ in DIVISORS:
        ratios = []
        for rung in rungs:
            a, _ = cell_tau(first, rung, key)
            b, _ = cell_tau(second, rung, key)
            if np.isfinite([a, b]).all() and min(a, b) > 0:
                ratios.append(a / b)
        if ratios:
            ideal[key] = abs(np.log(float(np.exp(np.mean(np.log(ratios))))))
    if ideal:
        best = min(ideal, key=ideal.get)
        need = scenes[first]["div"]["z_med_f0"] / scenes[second]["div"]["z_med_f0"]
        need *= float(np.exp(np.mean(np.log([
            cell_tau(first, r, "z_med_f0")[0] / cell_tau(second, r, "z_med_f0")[0]
            for r in rungs
            if np.isfinite([cell_tau(first, r, "z_med_f0")[0],
                            cell_tau(second, r, "z_med_f0")[0]]).all()]))))
        print(f"\n{'후보 중 최선':<14}{best}   (|log 비| {ideal[best]:.3f})")
        print(f"{'비 1.0 이 되려면':<14}분모 비 {first}/{second} = {need:.3f} 이어야 한다   "
              + "   ".join(f"{k} {scenes[first]['div'][k] / scenes[second]['div'][k]:.3f}"
                           for k, _ in DIVISORS))


if __name__ == "__main__":
    parser = ArgumentParser(description="τ 분모 후보 실측 (조사용)")
    parser.add_argument("--videos", nargs="*", default=["camel", "avocado-slice"])
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)
    parser.add_argument("--stride", default=4, type=int)    # 전 프레임 통계 픽셀 솎기
    main(parser.parse_args())
