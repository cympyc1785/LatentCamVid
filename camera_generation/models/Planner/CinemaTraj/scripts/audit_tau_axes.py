r"""τ 를 대신할 수 있는 **이동량 축 후보들**을 같은 뱅크 위에서 나란히 재고 비교한다.

왜 (사용자 지시 2026-09-02): "tau 이동거리 기준을 object-centric + track, free-moving 을
분리하려는데 각각 기준이 될 수 있는 수치들이 뭐가 있을지 후보군들 비교해줘".

지금 축은 하나뿐이다 — `τ(f) = |p_plan(f) − p_src(f)| / z_med` (`lbm/presets.py:tau_of`).
이게 세 계열에 동시에 안 맞는다:

- **free-moving** (`aim="free"/"traj"`) — 조준할 대상이 없다. 분모 `z_med` 가 씬 깊이라
  맞는 축이지만, 분자가 **소스 카메라와의 차이**라 소스가 이미 움직이면 `tau_start` 가 예산을
  먹는다 (parkour `tau_start` 0.7454).
- **object-centric** (`aim="look_at"`, 비-track) — 실제로 중요한 건 씬 깊이가 아니라
  **subject 와의 거리**다. 같은 τ 라도 subject 가 가까우면 화면에서 훨씬 크게 움직인다.
- **track_** 계열 — 위치를 100% 추종하므로 세계 변위의 대부분이 추종분이다. D97 이 `tau_ref=follow`
  로 분자만 바꿔 놨는데, 그것도 여전히 분모가 `z_med` 다.

## 후보 축 (전부 이 스크립트가 실측한다)

`p(f)` = 플랜 카메라 중심, `c(f)` = subject OBB 중심(프레임별), `r(f) = |p(f) − c(f)|`,
`p_src(f)` = 소스 카메라 중심, `S` = 씬 단위, `z_med` = frame0 non-sky z 중앙값,
`d_ref` = 노드의 기준 관측 거리.

| 키 | 식 | 분모의 뜻 | 어느 계열용 |
|---|---|---|---|
| `tau_source`  | `max_f |p−p_src| / z_med`         | 씬 깊이 | 현행 전부 |
| `tau_shape`   | `max_f |p−p(0)| / z_med`          | 씬 깊이 | free (소스 오프셋 제거) |
| `path_S`      | `Σ_f |Δp| / S`                    | 씬 단위 | free |
| `net_S`       | `|p(48)−p(0)| / S`                | 씬 단위 | free |
| `rel_sub`     | `max_f |(p−c) − (p(0)−c(0))| / d_ref` | subject 거리 | object-centric / track |
| `dr_ref`      | `(max r − min r) / d_ref`         | subject 거리 | object-centric (dolly) |
| `r_ratio`     | `max r / min r`                   | 무차원      | object-centric (dolly) |
| `az_deg`      | 중력축 둘레 subject 방위각 이동폭 | 각도        | object-centric (orbit/arc) |
| `elev_deg`    | subject 고도각 이동폭             | 각도        | object-centric (crane) |
| `rot_deg`     | `max_f ∠(R(0), R(f))` 측지 회전   | 각도        | free (pan/tilt) |
| `subtend_ratio` | `max/min` of `2·atan(0.5·diag_obb / r)` | 무차원 | object-centric (화면 크기) |

## 어떻게 비교하나 — 좋은 축의 조건 셋

1. **난이도와 단조** — `hole_fraction` (하류 생성 비용의 직접 측정치) 과 Spearman ρ 가 커야 한다.
   Pearson 이 아니라 Spearman 인 이유는 관계가 선형일 이유가 없어서다(순위만 맞으면 사다리를
   깎는 데 쓸 수 있다).
2. **씬을 건너 비교 가능** — 같은 값이 씬마다 같은 뜻이어야 한다. 씬별 median 의 변동계수
   `cv_between` 이 작을수록 좋다. (`prdc-not-comparable-across-corpora` 와 같은 함정이다.)
3. **안 죽어 있을 것** — 그 계열에서 값이 상수/0 이면 손잡이가 아니다. `frac_zero` 로 본다.

**주의: 이 셋은 서로 상충한다.** `cv_between` 이 0 인 축은 씬 차이를 아예 안 보는 것일 수도
있다. 표를 한 열로 줄 세우지 말고 계열별로 셋을 같이 볼 것.

env: `vista4d` (렌더 없음, CPU 만)

예시:
    python scripts/audit_tau_axes.py --bank_dir hole_bank_k6_d99
    python scripts/audit_tau_axes.py --bank_dir hole_bank_k6_d99 \
        --videos parkour snowboard camel --out /tmp/tau_axes.csv
"""
import csv
import json
import sys
from argparse import ArgumentParser
from os import listdir, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.presets import PRESET_ALIASES                                          # noqa: E402
from scene_graph.lift import apply_transform                                    # noqa: E402
from scene_graph.obb import node_obb_at                                         # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402

# 계열 이름. 이 셋으로 나눠서 축을 따로 고른다 (사용자 지시).
FAMILIES = ("track", "object", "free")

AXES = ("tau_source", "tau_shape", "path_S", "net_S", "rel_sub", "dr_ref", "r_ratio",
        "az_deg", "elev_deg", "rot_deg", "subtend_ratio")


def family_of(preset: str, aim: str) -> str:
    """preset+aim → 계열. `aim` 이 판별자다 (`aim-keyframes-zero-means-two-things` 와 같은 이유)."""
    canonical = PRESET_ALIASES.get(preset, preset)
    if canonical.startswith("track_"):
        return "track"
    return "object" if aim == "look_at" else "free"


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """순위 상관. scipy 없이 — 동점은 평균 순위로."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")

    def rank(values):
        order = values.argsort()
        ranks = np.empty(len(values), float)
        ranks[order] = np.arange(len(values), dtype=float)
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=ranks)
        return (sums / counts)[inverse]

    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def geodesic_deg(R0: np.ndarray, R: np.ndarray) -> np.ndarray:
    """`R(0)` 대비 프레임별 측지 회전각(도)."""
    trace = np.einsum("ij,fij->f", R0, R)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def axes_of(poses: np.ndarray, src_centers: np.ndarray, node: dict | None,
            T_gw: np.ndarray, scale: float, z_med: float, d_ref: float):
    """한 변이의 후보 축 전부. subject 가 없으면(free) subject 기반 축은 nan."""
    p = poses[:, :3, 3]
    num = len(p)
    src = src_centers[np.rint(np.linspace(0, len(src_centers) - 1, num)).astype(int)] \
        if len(src_centers) != num else src_centers
    out = {
        "tau_source": float(np.linalg.norm(p - src, axis=1).max() / max(z_med, 1e-9)),
        "tau_shape": float(np.linalg.norm(p - p[0], axis=1).max() / max(z_med, 1e-9)),
        "path_S": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() / max(scale, 1e-9)),
        "net_S": float(np.linalg.norm(p[-1] - p[0]) / max(scale, 1e-9)),
        "rot_deg": float(geodesic_deg(poses[0, :3, :3], poses[:, :3, :3]).max()),
    }
    for key in ("rel_sub", "dr_ref", "r_ratio", "az_deg", "elev_deg", "subtend_ratio"):
        out[key] = float("nan")
    if node is None:
        return out

    centers = np.stack([node_obb_at(node, min(f, node_frames(node) - 1))[0]
                        for f in range(num)])          # G 좌표
    extent = np.asarray(node["obb"]["extent"], float)
    p_g = apply_transform(T_gw, p)                     # 카메라도 G 로 — 중력축이 e_z 인 좌표계
    delta = p_g - centers
    r = np.linalg.norm(delta, axis=1)
    rel = delta - delta[0]
    azimuth = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    elevation = np.arcsin(np.clip(delta[:, 2] / np.maximum(r, 1e-9), -1.0, 1.0))
    subtend = 2.0 * np.arctan(0.5 * float(np.linalg.norm(extent)) / np.maximum(r, 1e-9))
    out.update({
        "rel_sub": float(np.linalg.norm(rel, axis=1).max() / max(d_ref, 1e-9)),
        "dr_ref": float((r.max() - r.min()) / max(d_ref, 1e-9)),
        "r_ratio": float(r.max() / max(r.min(), 1e-9)),
        "az_deg": float(np.degrees(azimuth.max() - azimuth.min())),
        "elev_deg": float(np.degrees(elevation.max() - elevation.min())),
        "subtend_ratio": float(subtend.max() / max(subtend.min(), 1e-9)),
    })
    return out


def node_frames(node: dict) -> int:
    return len(node["track"]["center_smooth"])


def collect(out_root: str, video: str, bank_dir: str):
    """뱅크 한 편 → 변이별 행."""
    folder = path.join(out_root, video, bank_dir)
    with open(path.join(folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    poses_all = np.load(path.join(folder, "poses.npz"), allow_pickle=False)["cam_c2w"]
    graph = load_graph(path.join(out_root, video, "scene_graph.json"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    T_gw = np.asarray(graph["frames"]["T_gw"], float)
    scale, z_med = float(bank["S"]), float(bank["z_med"])
    # 뱅크 `poses.npz` 는 **world** c2w 다 (`fit_hole_ladder` 가 recon depth 에 그대로 되쏜다).
    src_centers = np.asarray(graph["cameras"]["cam_c2w_world"], float)[:, :3, 3]

    rows = []
    for index, variant in enumerate(bank["variants"]):
        node = nodes.get(variant.get("anchor_id"))
        # `aim="free"` 는 조준을 안 하지만 anchor 는 여전히 붙어 있다 — subject 기반 축을
        # 그래도 재 둔다. "free 에는 subject 축이 의미 없다"가 아니라 **후보 비교가 목적**이라
        # 같은 열을 채워 놓고 계열별로 갈라 본다.
        row = {"video": video, "variant_id": variant["variant_id"],
               "anchor_id": variant.get("anchor_id"), "preset": variant["preset"],
               "aim": variant.get("aim", "free"),
               "family": family_of(variant["preset"], variant.get("aim", "free")),
               "hole_fraction": float(variant.get("hole_fraction", float("nan"))),
               "behind_frac": float(variant.get("behind_frac", float("nan")) or 0.0),
               "status": variant.get("status", ""), "binding": variant.get("binding", "")}
        d_ref = float((node or {}).get("viewing_distance", {}).get("d_ref", float("nan")))
        row.update(axes_of(poses_all[index], src_centers, node, T_gw, scale, z_med, d_ref))
        rows.append(row)
    return rows


def summarize(rows: list, key_metric: str):
    """계열 × 축 표. 조건 셋(단조·씬간 비교가능·안 죽음)을 한 줄에."""
    print(f"\n{'':<14}{'n':>6}{'ρ(축,' + key_metric + ')':>16}{'p50':>10}{'p05':>10}{'p95':>10}"
          f"{'cv_within':>11}{'cv_between':>12}{'frac_flat':>11}")
    for family in FAMILIES:
        subset = [r for r in rows if r["family"] == family]
        if not subset:
            continue
        target = np.asarray([r[key_metric] for r in subset], float)
        print(f"-- {family} (n={len(subset)}, {key_metric} p50 "
              f"{np.nanmedian(target):.4f})")
        for axis in AXES:
            values = np.asarray([r[axis] for r in subset], float)
            ok = np.isfinite(values)
            if ok.sum() < 3:
                print(f"{axis:<14}{ok.sum():>6}{'—':>16}")
                continue
            per_video = {}
            for row, value in zip(subset, values):
                if np.isfinite(value):
                    per_video.setdefault(row["video"], []).append(value)
            medians = np.asarray([np.median(v) for v in per_video.values()], float)
            within = np.asarray([np.std(v) / max(abs(np.mean(v)), 1e-9)
                                 for v in per_video.values()], float)
            # "안 죽음" = 그 씬 안에서 값이 실제로 변하는가. 상수면 손잡이가 아니다.
            flat = np.mean([np.ptp(v) < 1e-6 for v in per_video.values()])
            print(f"{axis:<14}{ok.sum():>6}{spearman(values, target):>16.3f}"
                  f"{np.nanpercentile(values[ok], 50):>10.3f}"
                  f"{np.nanpercentile(values[ok], 5):>10.3f}"
                  f"{np.nanpercentile(values[ok], 95):>10.3f}"
                  f"{np.nanmedian(within):>11.3f}"
                  f"{np.std(medians) / max(abs(np.mean(medians)), 1e-9):>12.3f}"
                  f"{flat:>11.3f}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="*", default=None, type=str)
    parser.add_argument("--bank_dir", default="hole_bank_k6_d99", type=str)
    parser.add_argument("--output_root", default=None, type=str)
    # 난이도의 대리지표. `hole_fraction` 이 하류 비용의 직접 측정치이고, `behind_frac` 은
    # 물리 위반 쪽 — 둘이 다른 축을 고를 수도 있어서 바꿔 볼 수 있게 열어 둔다.
    parser.add_argument("--metric", default="hole_fraction", type=str)
    # 사다리가 **못 풀고 끝난** 변이(`clamped_low+tau_floor` / `unreached`)는 손잡이가 0 에
    # 박혀 있거나 목표에 못 닿은 것이라, 섞어 놓으면 축이 아니라 사다리의 실패를 재게 된다.
    parser.add_argument("--status", nargs="*", default=None, type=str)
    parser.add_argument("--out", default="/tmp/tau_axes.csv", type=str)
    args = parser.parse_args()

    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    videos = args.videos
    if not videos:
        videos = sorted(v for v in listdir(out_root)
                        if path.isfile(path.join(out_root, v, args.bank_dir, "bank.json")))

    rows = []
    for video in videos:
        try:
            got = collect(out_root, video, args.bank_dir)
        except Exception as error:                                          # noqa: BLE001
            print(f"{video:<22}SKIP  {type(error).__name__}: {error}")
            continue
        rows.extend(got)
        print(f"{video:<22}{len(got):>6} 변이")

    if args.status:
        before = len(rows)
        rows = [r for r in rows if r["status"] in args.status]
        print(f"{'status 필터':<14}{args.status}   {before} → {len(rows)} 변이")
    if not rows:
        print("변이가 없다")
        return
    with open(args.out, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {f: sum(r["family"] == f for r in rows) for f in FAMILIES}
    print(f"\n{'뱅크':<14}{args.bank_dir}   영상 {len(videos)}   변이 {len(rows)}   {counts}")
    summarize(rows, args.metric)
    print(f"\n{'CSV':<14}{args.out}")


if __name__ == "__main__":
    main()
