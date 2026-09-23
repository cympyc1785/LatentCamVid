"""충돌 임계(거리)를 **스윕해서** 어느 값에서 몇 변이가 죽는지 곡선으로 낸다.

**왜 필요한가** (사용자 질문 D118): "trumans collision 을 1.5 m 로 설정해두고 돌리고 있는 걸로
아는데 거리마다 충돌로 판정하는 frustum 및 ray 를 시각화해서 어떤 수치로 하는 게 적절할지 알 수
있게 해줘."

먼저 **1.5 m 는 임계가 아니다.** TRUMANS 경로에 거리 손잡이가 세 개 있고 역할이 전부 다르다:

  ┌ 뱅크 굽기 G1 (`fit_hole_ladder.py`, `--collision_source both`)
  │   `--behind_margin_frac 0.02` × 씬 단위 `S` → **실제 임계**. a00 은 S=1.9742 라 **3.95 cm**.
  │   mesh EDT(`mesh_grid.npz`) 와 depth shell 두 채널의 위반 프레임 **합집합**을 본다.
  │   `--max_behind_frac 0.0` 이라 위반 프레임이 **한 장이라도** 있으면 그 칸은 탈락이다.
  └ Blender pose export raycast (`bank_to_blender_poses.py --raycast`)
      `--min_clearance 0.35` m ← **실제 임계**
      `--probe_distance 1.5` m ← 광선 **길이 상한**. 여기까지 아무것도 안 맞으면 "뚫림"으로
      친다. 기존 audit 3건 실측 포화율 0.0 / 0.5 / 0.5 % 라 **사실상 한 번도 안 걸린다** —
      1.5 를 올리거나 내려도 판정이 거의 안 바뀐다는 뜻이다.

이 스크립트는 **위쪽(뱅크 G1)** 을 스윕한다. mesh 격자는 EDT 라 임계를 바꾸는 데 재굽기가
필요 없다 — 궤적당 최소 clearance 를 한 번 재두면 임계 스윕은 비교 연산뿐이다. 그래서 뱅크
836 변이 × 49 프레임을 통째로 훑어도 수 초다.

아래쪽(raycast)은 `--audit` 로 기존 `raycast_probe_*.json` 을 넘기면 같은 형식으로 같이 낸다.

## 읽는 법

`binding` 열이 요점이다. 정적 채널(벽·가구)과 동적 채널(사람)이 서로 다른 거리에서 죽으므로
한 숫자로 합쳐 놓으면 "임계를 올렸더니 다 죽었다"의 원인을 못 가른다. `unreachable` 은 거리와
무관하게(임계 0 에서도) 죽는 칸이다 — 소스 카메라와 다른 방(벽 너머·봉인 공동)에 있다.

`solved유지` 열이 실제 결정에 쓰는 숫자다. 전체 변이가 아니라 **뱅크가 `solved` 로 내보낸
변이만** 놓고 임계를 올렸을 때 몇 %가 그대로 남는지를 본다.

## 이 스윕이 재는 것 / 안 재는 것 — 함정 두 개

**(1) `poses.npz` 는 게이트를 이미 통과한 pose 다.** 사다리 이분법이 충돌이 없어질 때까지 크기를
줄인 뒤의 결과라 현재 임계에서는 정의상 전량 통과한다. 그러니 `통과` 열은 "임계를 이만큼 올리면
이만큼 살아남는다"가 아니라 **"지금 구워 놓은 궤적이 표면에서 얼마나 떨어져 있나"** 다. 실제
재굽기에서는 탈락 대신 **한 단 더 줄어든 궤적**이 나오므로 `solved유지` 는 재굽기 생존율의
**하한**이고, 실제 손실은 "탈락"이 아니라 "이동량 감소"로 나타난다.

**(2) EDT 격자는 voxel 0.05 m 로 양자화되어 있다.** 거리값이 0.05·√{1,2,3,…} 으로만 나와서 0 이
아닌 최솟값이 **voxel 하나(0.05 m)** 다. 그래서 현재 임계 `0.02·S ≈ 0.04 m` 는 격자가 표현할 수
있는 최솟값보다 작아 **한 번도 구속하지 않는다** — 실효 임계는 voxel 그 자체다. 0.05 미만
구간의 곡선은 전부 같은 점이니 읽지 말 것. 임계를 의미 있게 움직이려면 voxel 위로 올려야 한다.

env: `vista4d` (numpy/scipy 만 쓴다. GPU 불필요)

예시:
    P=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $P eval/sweep_collision_margin.py --video tru_1d076f8c_a00_s3f0k6
    $P eval/sweep_collision_margin.py --video tru_1d076f8c_a00_s3f0k6 tru_1d076f8c_a05_s3f0k6 \
        --margins 0.00 0.02 0.04 0.08 0.15 0.25 0.35 0.50 --out /tmp/sweep_a00.json
    # Blender raycast 게이트 쪽 (probe_distance / min_clearance)
    $P eval/sweep_collision_margin.py --no_mesh \
        --audit results/20260901_trumans_preset_blender/*/raycast_probe_rungs.json
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

from lbm.mesh_collision import MeshClearance                                    # noqa: E402

#    기본 스윕 격자. 현재 값(0.02·S ≈ 0.04)을 반드시 포함시켜 "지금 어디에 서 있나"가 표에
#    같이 보이게 한다.
MARGINS = (0.00, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.50)


def measure_mesh(video: str, output_root: str, bank_dir: str):
    """변이별 (정적 최소 clearance, 동적 최소 clearance, 도달불가 프레임 수) 를 잰다.

    `mesh_behind_profile` 은 임계를 받아 **판정만** 돌려주므로 스윕에 못 쓴다. 여기서는 같은
    조회(`static_clearance`/`dynamic_clearance`)를 하되 **거리 자체를** 남긴다 — 임계 스윕이
    비교 한 번으로 끝난다. 시간 매칭 규칙은 `mesh_behind_profile` 과 글자 그대로 같다.
    """
    root = path.join(output_root, video)
    mesh = MeshClearance(path.join(root, "mesh_grid.npz"))
    data = np.load(path.join(root, bank_dir, "poses.npz"), allow_pickle=True)
    cam = np.asarray(data["cam_c2w"], dtype=np.float64)          # (V,F,4,4) 뱅크 world
    vids = [str(v) for v in data["variant_id"]]
    scale = json.load(open(path.join(root, "scene_graph.json")))["scale"]["S"]

    num_var, num_plan = cam.shape[0], cam.shape[1]
    num_mesh = len(mesh.dyn_edt)
    centers = mesh.to_blend(cam.reshape(-1, 4, 4))[:, :3, 3].reshape(num_var, num_plan, 3)
    flat = centers.reshape(-1, 3)

    s_dist, s_reach = mesh.static_clearance(flat)
    s_dist = s_dist.reshape(num_var, num_plan)
    s_reach = s_reach.reshape(num_var, num_plan)
    #    동적 격자는 프레임마다 다른 배열이라 프레임 축으로 돌린다 (49회, 각 836점 조회).
    d_dist = np.empty((num_var, num_plan), dtype=np.float64)
    for f in range(num_plan):
        mf = int(round(f * (num_mesh - 1) / max(num_plan - 1, 1)))
        d_dist[:, f] = mesh.dynamic_clearance(centers[:, f], mf)
    return {"video": video, "S": float(scale), "clip": float(mesh.clip),
            "voxel": float(mesh.voxel), "variant_id": vids,
            "static_min": s_dist.min(1), "dynamic_min": d_dist.min(1),
            "unreachable_frames": (~s_reach).sum(1), "num_frames": num_plan}


def status_of(video: str, output_root: str, bank_dir: str):
    """`bank.csv` 의 variant_id → status. 스윕 결과를 **실제 뱅크 판정과 대조**하기 위한 것."""
    csv_path = path.join(output_root, video, bank_dir, "bank.csv")
    if not path.isfile(csv_path):
        return {}
    with open(csv_path, encoding="utf-8") as file:
        return {r["variant_id"]: r["status"] for r in csv.DictReader(file)}


def sweep_mesh(rows: dict, margins, solved=None) -> list:
    """임계별 생존 수 + binding 채널 분해.

    `solved` 는 `bank.csv` 가 `solved*` 로 찍은 변이의 bool 마스크. 전체 변이에는 애초에 다른
    이유(τ 바닥·elev·obb)로 버려진 칸이 섞여 있어 `통과/전체` 로는 "임계를 올리면 코퍼스가
    얼마나 줄어드나"를 못 읽는다. **쓰이는 궤적만** 따로 세는 열이 `solved유지` 다.
    """
    stat, dyn, unreach = rows["static_min"], rows["dynamic_min"], rows["unreachable_frames"]
    out = []
    for margin in margins:
        bad_reach = unreach > 0
        bad_stat = (stat < margin) & ~bad_reach
        bad_dyn = (dyn < margin) & ~bad_reach & ~bad_stat
        ok = ~(bad_reach | bad_stat | bad_dyn)
        row = {"margin_m": float(margin),
               "margin_over_S": float(margin / rows["S"]),
               "pass": int(ok.sum()),
               "total": len(stat),
               "fail_unreachable": int(bad_reach.sum()),
               "fail_static": int(bad_stat.sum()),
               "fail_dynamic": int(bad_dyn.sum())}
        if solved is not None and solved.any():
            row["solved_keep"] = int(ok[solved].sum())
            row["solved_total"] = int(solved.sum())
            row["solved_keep_frac"] = float(ok[solved].mean())
        out.append(row)
    return out


def sweep_audit(files, thresholds) -> list:
    """Blender raycast audit 의 `min_clearance` / `min_subject_dist` 스윕 + cap 포화율."""
    out = []
    for file_path in files:
        data = json.load(open(file_path, encoding="utf-8"))
        paths = data["paths"]
        clear = np.array([[r["clearance"] for r in p] for p in paths])
        subj = np.array([[r["subject_dist"] for r in p] for p in paths])
        floor = np.array([[r["floor_drop"] for r in p] for p in paths])
        cap = float(clear.max())
        row = {"file": file_path, "paths": len(paths), "frames": clear.shape[1],
               "probe_cap_observed_m": cap,
               "cap_saturated_frac": float((clear >= cap - 1e-6).mean()),
               "floor_is_binding_frac": float(np.isclose(clear, floor).mean()),
               "clearance_min": float(clear.min()),
               "clearance_median": float(np.median(clear)),
               "sweep": [{"min_clearance_m": float(t),
                          "pass": int((clear.min(1) >= t).sum()), "total": len(paths)}
                         for t in thresholds],
               "subject_dist_min": float(subj.min())}
        out.append(row)
    return out


def plot_sweep(result, out_png: str):
    """두 게이트를 **한 장에** 그린다 — 따로 그리면 7배 차이가 안 보인다.

    왼쪽은 뱅크 G1(mesh EDT) 의 solved 유지율, 오른쪽은 Blender raycast 의 궤적 통과율. x 축이
    둘 다 '표면까지의 거리(m)' 라 같은 눈금에 놓을 수 있고, 현재 두 게이트가 서 있는 자리
    (0.04 / 0.35) 를 세로선으로 찍어 간격을 보이게 한다. voxel 0.05 아래는 회색으로 덮어
    "여기 곡선은 격자 해상도라 읽지 말 것"을 그림으로 못 박는다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    margins = result["margins"]
    ax = axes[0]
    for entry in result["mesh"]:
        keep = [r.get("solved_keep_frac") for r in entry["sweep"]]
        if any(k is None for k in keep):
            continue
        ax.plot(margins, [k * 100 for k in keep], color="0.72", lw=0.9, zorder=1)
    tot_keep = [sum(e["sweep"][k].get("solved_keep", 0) for e in result["mesh"])
                for k in range(len(margins))]
    tot_all = [sum(e["sweep"][k].get("solved_total", 0) for e in result["mesh"])
               for k in range(len(margins))]
    if tot_all and tot_all[0]:
        ax.plot(margins, [a / b * 100 for a, b in zip(tot_keep, tot_all)],
                "o-", color="#c0392b", lw=2.2, zorder=3, label=f"pooled (solved {tot_all[0]})")
    voxel = result["mesh"][0]["voxel"] if result["mesh"] else 0.05
    ax.axvspan(0, voxel, color="0.88", zorder=0)
    ax.text(voxel / 2, 52, f"voxel\n{voxel:g} m\n(grid res.)", ha="center", va="center", fontsize=7)
    cur = result["mesh"][0]["current_margin_m"] if result["mesh"] else 0.04
    ax.axvline(cur, color="#2980b9", ls="--", lw=1.4)
    ax.text(cur + 0.006, 62, f"current 0.02*S\n= {cur:.3f} m", color="#2980b9", fontsize=8)
    ax.axvline(0.35, color="#8e44ad", ls=":", lw=1.4)
    ax.text(0.355, 62, "raycast gate\n0.35 m", color="#8e44ad", fontsize=8)
    ax.set_xlabel("bake G1 margin (m) - mesh EDT")
    ax.set_ylabel("solved variants retained (%)")
    ax.set_title("BAKE gate (20 chunks; grey = per-chunk)")
    ax.set_ylim(50, 101)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for entry in result["audit"]:
        tag = path.basename(path.dirname(entry["file"]))
        xs = [r["min_clearance_m"] for r in entry["sweep"]]
        ys = [r["pass"] / r["total"] * 100 for r in entry["sweep"]]
        ax.plot(xs, ys, "o-", lw=1.8, label=f"{tag} ({entry['paths']} paths)")
    ax.axvline(0.35, color="#8e44ad", ls=":", lw=1.4)
    ax.text(0.355, 80, "current 0.35 m", color="#8e44ad", fontsize=8)
    ax.set_xlabel("raycast --min_clearance (m)")
    ax.set_ylabel("trajectories passing (%)")
    ax.set_title("RAYCAST gate (Blender export filter)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    print(f"\n그림 -> {out_png}")


def main(args):
    result = {"format": "collision_margin_sweep_v1", "margins": list(args.margins), "mesh": [],
              "audit": []}
    for video in args.video:
        if not args.mesh:
            break
        rows = measure_mesh(video, args.output_root, args.bank_dir)
        status = status_of(video, args.output_root, args.bank_dir)
        mask = np.array([status.get(v, "").startswith("solved") for v in rows["variant_id"]])
        sweep = sweep_mesh(rows, args.margins, mask)
        solved = int(mask.sum())
        result["mesh"].append({"video": video, "S": rows["S"], "voxel": rows["voxel"],
                               "clip": rows["clip"], "variants": len(rows["variant_id"]),
                               "bank_solved": solved,
                               "current_margin_m": 0.02 * rows["S"], "sweep": sweep,
                               "static_min_median": float(np.median(rows["static_min"])),
                               "dynamic_min_median": float(np.median(rows["dynamic_min"])),
                               "unreachable_variants":
                                   int((rows["unreachable_frames"] > 0).sum())})
    if args.audit:
        result["audit"] = sweep_audit(args.audit, args.margins)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=1, default=float)

    for entry in result["mesh"]:
        print(f"\n=== [뱅크 G1 / mesh EDT] {entry['video']}")
        print(f"{'S':<22}{entry['S']:.4f} m/u   voxel {entry['voxel']:.3f} m   "
              f"clip {entry['clip']:.2f} m")
        print(f"{'현재 임계':<20}0.02·S = {entry['current_margin_m']:.4f} m")
        print(f"{'변이':<22}{entry['variants']}   뱅크 solved {entry['bank_solved']}   "
              f"도달불가 {entry['unreachable_variants']}")
        print(f"{'최소 clearance 중앙값':<18}정적 {entry['static_min_median']:.3f} m   "
              f"동적 {entry['dynamic_min_median']:.3f} m")
        print(f"{'주의':<22}voxel {entry['voxel']:.3f} m 아래 구간은 격자가 표현을 못 한다 "
              f"(현재 임계 {entry['current_margin_m']:.3f} m 는 실효 무구속)")
        print(f"\n{'margin(m)':>10}{'=x·S':>8}{'통과':>8}{'/전체':>7}"
              f"{'unreach':>9}{'static':>8}{'dyn':>7}{'solved유지':>12}")
        for row in entry["sweep"]:
            keep = (f"{row['solved_keep']:>4}/{row['solved_total']:<4}"
                    f"{row['solved_keep_frac'] * 100:>4.0f}%" if "solved_keep" in row else "")
            print(f"{row['margin_m']:>10.3f}{row['margin_over_S']:>8.3f}{row['pass']:>8}"
                  f"{row['total']:>7}{row['fail_unreachable']:>9}"
                  f"{row['fail_static']:>8}{row['fail_dynamic']:>7}  {keep}")
    #    씬 하나로는 못 정한다 — 씬마다 가구 배치가 달라 곡선이 어디서 꺾이는지가 다르다.
    #    코퍼스 전체 solved 를 분모로 한 pooled 열이 실제 채택 근거다.
    if len(result["mesh"]) > 1:
        print(f"\n=== [뱅크 G1 / mesh EDT] POOLED  {len(result['mesh'])}개 씬")
        print(f"\n{'margin(m)':>10}{'solved유지':>12}{'/solved':>9}{'유지율':>9}")
        for k, margin in enumerate(args.margins):
            keep = sum(e["sweep"][k].get("solved_keep", 0) for e in result["mesh"])
            tot = sum(e["sweep"][k].get("solved_total", 0) for e in result["mesh"])
            if tot:
                print(f"{margin:>10.3f}{keep:>12}{tot:>9}{keep / tot * 100:>8.1f}%")
    for entry in result["audit"]:
        print(f"\n=== [Blender raycast] {entry['file']}")
        print(f"{'궤적':<22}{entry['paths']}   프레임 {entry['frames']}")
        print(f"{'probe cap 관측':<19}{entry['probe_cap_observed_m']:.2f} m   "
              f"포화 {entry['cap_saturated_frac'] * 100:.1f}%  "
              f"(= 이 상한이 판정을 바꾸는 비율)")
        print(f"{'아래 방향이 binding':<17}{entry['floor_is_binding_frac'] * 100:.1f}%")
        print(f"{'clearance':<22}min {entry['clearance_min']:.3f}  "
              f"med {entry['clearance_median']:.3f}   subject_dist min "
              f"{entry['subject_dist_min']:.3f}")
        print(f"\n{'min_clearance(m)':>17}{'통과':>8}{'/전체':>7}")
        for row in entry["sweep"]:
            print(f"{row['min_clearance_m']:>17.3f}{row['pass']:>8}{row['total']:>7}")
    if args.plot:
        plot_sweep(result, args.plot)
    if args.out:
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", nargs="*", default=[], type=str)
    parser.add_argument("--output_root", default="out_trumans", type=str)
    parser.add_argument("--bank_dir", default="hole_bank_k6_d116", type=str)
    #    mesh 격자 스윕 on/off. 끄면 `--audit` 만 돈다 (Blender raycast 쪽만 볼 때).
    parser.add_argument("--mesh", action="store_true", default=True)
    parser.add_argument("--no_mesh", dest="mesh", action="store_false")
    parser.add_argument("--audit", nargs="*", default=[], type=str)   # raycast_probe_*.json
    parser.add_argument("--margins", nargs="*", default=list(MARGINS), type=float)
    parser.add_argument("--out", default="", type=str)
    parser.add_argument("--plot", default="", type=str)      # 두 게이트 곡선 png
    main(parser.parse_args())
