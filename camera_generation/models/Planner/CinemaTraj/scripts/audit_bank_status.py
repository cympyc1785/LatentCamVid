"""뱅크 전체의 `status`/`binding` 분포를 세고, **기준 완화로 되살릴 수 있는 변이 목록**을 뽑는다.

왜 필요한가 (사용자 질문 D112): "clamped_low 처럼 어디서 판정에 걸렸다거나 만족하는 구간이 없는
경우들도 기록되고 있는거지? 아니면 기록되게 해주고 clamped_low 만 모아서 나중에 따로 기준점
완화해서 돌려볼 수 있게 리스트 만들어줘."

**기록은 이미 되고 있다.** `fit_hole_ladder.py` 는 탈락한 변이를 지우지 않고 `status`/`binding`
두 열로 남긴다 (D39/D45 와 같은 이유 — 자를지 말지는 소비처가 정한다). 없던 건 그 열을 코퍼스
전체에 걸쳐 **모아서 보는 도구**다. `audit_bank_geometry.py` 는 G1/G3 를 새로 재는 감사이고,
이건 이미 적힌 판정을 집계만 한다 (렌더 0회, 전 코퍼스가 몇 초).

## status / binding 이 뜻하는 것

`status` 는 knob(손잡이) 이분법이 어떻게 끝났는지다:

- `solved`         — 목표 hole 에 도달했고 게이트도 안 걸렸다. 유일한 정상 종료.
- `clamped_low`    — **탐색 하한에서 이미 게이트가 걸렸다.** 손잡이를 더 줄일 수가 없으니
                     "만족하는 구간이 없음". `binding` 이 어느 게이트인지 말해준다.
- `unreached`      — 상한까지 올려도 목표 hole 에 못 미쳤다 (게이트가 아니라 사다리 부족).
- `<gate>_limited` — 목표 hole 전에 게이트가 먼저 걸려서 그 직전에서 멈췄다. 궤적은 나오지만
                     의도한 hole 보다 작다.
- `static`         — 손잡이가 0 으로 수렴 (사실상 정지 궤적).
- `+tau_floor`     — 접미사. `tau_start` 가 높아서 knob 하한이 `tau_start + 0.02` 로 올라갔다.
                     이게 붙은 `clamped_low` 는 게이트가 아니라 **하한 자체**가 범인일 수 있다.

`binding` 은 `over()` 의 if/elif 체인에서 **처음 발화한** 게이트다 (collision → obb → ground →
elev → approach → hole → occlusion). 뒤쪽 게이트도 동시에 위반일 수 있는데 안 적힌다 — 그래서
"이 게이트만 풀면 살아난다"가 아니라 "이 게이트가 첫 벽이었다"로 읽어야 한다.

## 완화 후보 목록

`--relax_list` 로 `clamped_low*` 행만 골라 JSON 으로 뽑는다. 나중에 게이트 임계를 낮춰
(`--obb_clear_src_ratio` / `--min_ground_clear_ratio` / `--approach_src_ratio` /
`--max_behind_frac` / `--tau_floor_src`) 이 변이들만 재fit 할 때의 입력이다. binding 별로
묶어 나오므로 "collision 만 풀고 돌려보기" 같은 부분 완화가 바로 된다.

출력: 요약표(stdout) + `--csv` / `--relax_list` 파일. **행을 지우지 않는다.**

env: 아무거나 (numpy 도 안 쓴다)

예시:
    python scripts/audit_bank_status.py --output_root out --bank_dir hole_bank_k6_d99
    python scripts/audit_bank_status.py --output_root out_dynpose --bank_dir hole_bank_d110 \
        --relax_list /tmp/d110_clamped_low.json --csv /tmp/d110_status.csv
"""
import csv
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from glob import glob
from os import makedirs, path

# 완화 후보로 뽑을 status 접두사. `clamped_low` 와 `clamped_low+tau_floor` 둘 다 잡는다.
RELAX_PREFIX = "clamped_low"

# 완화 목록 행에 실을 열. 재fit 재현에 필요한 것(변이 식별 + 어디서 걸렸나 + 얼마나 모자랐나)만.
RELAX_COLS = ("variant_id", "anchor_id", "anchor_label", "preset", "target_hole",
              "knob", "knob_floor", "knob_kind", "binding", "tau_start", "hole_fraction",
              "behind_frac", "obb_clear", "obb_slack", "obb_node", "ground_clear",
              "src_ground_clear", "min_ground_clear", "elev_abs_max", "max_elev_deg",
              "approach_gap", "src_approach", "min_approach", "subject_visible_frac")


def read_bank(csv_path):
    """bank.csv 를 dict 행 리스트로. 헤더가 뱅크마다 다를 수 있어 DictReader 로 읽는다."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = ArgumentParser(description="뱅크 status/binding 집계 + clamped_low 완화 후보 목록")
    # 뱅크가 놓인 상위 폴더. vista 는 out, dynpose 는 out_dynpose.
    parser.add_argument("--output_root", default="out")
    # 뱅크 폴더 이름 (fit 실행마다 다르다).
    parser.add_argument("--bank_dir", default="hole_bank")
    # 특정 영상만. 비우면 --output_root 아래 전부.
    parser.add_argument("--videos", nargs="*", default=None)
    # clamped_low 행 전량을 JSON 으로 저장할 경로. 없으면 요약만 찍는다.
    parser.add_argument("--relax_list", default=None)
    # 전 행의 (video, variant_id, status, binding) 을 CSV 로.
    parser.add_argument("--csv", default=None)
    # 요약표에서 status 상위 몇 개까지 보일지.
    parser.add_argument("--top", default=40, type=int)
    args = parser.parse_args()

    pattern = path.join(args.output_root, "*", args.bank_dir, "bank.csv")
    paths = sorted(glob(pattern))
    if args.videos:
        keep = set(args.videos)
        paths = [p for p in paths if p.split(path.sep)[-3] in keep]
    assert paths, f"뱅크를 못 찾았다: {pattern}"

    status_ct, binding_ct = Counter(), Counter()
    # clamped_low 안에서 어느 게이트가 첫 벽이었나 — 완화 우선순위를 정하는 수치.
    relax_binding_ct = Counter()
    per_video = defaultdict(Counter)
    relax_rows, all_rows = [], []
    # D188 `--pick`. 씬당 뽑힌 카메라(`picked` 열)를 집계한다. 수율(10k 목표 대비 몇 대인가)과
    # 어휘 비중(track 이 몇 %인가)이 여기서 나온다 — `status` 집계만으로는 안 보인다.
    pick_per_video, pick_tier_ct, pick_preset_ct = Counter(), Counter(), Counter()
    pick_holes, pick_track, pick_rows = [], 0, 0

    for p in paths:
        video = p.split(path.sep)[-3]
        for row in read_bank(p):
            if row.get("picked"):
                pick_rows += 1
                pick_per_video[video] += 1
                pick_tier_ct[row.get("plan_tier", "") or "(blank)"] += 1
                vid = row.get("variant_id", "")
                pick_preset_ct[row.get("preset", "") or "(blank)"] += 1
                # 어휘 판정은 preset 이름의 `track_` 접두사 (variant_id 는 anchor 접두사가 붙는다).
                if "track_" in vid or (row.get("preset", "")).startswith("track_"):
                    pick_track += 1
                try:
                    pick_holes.append(float(row.get("hole_fraction", "")))
                except (TypeError, ValueError):
                    pass
            st = row.get("status", "")
            bd = row.get("binding", "") or "(blank)"
            status_ct[st] += 1
            binding_ct[bd] += 1
            per_video[video][st] += 1
            all_rows.append({"video": video, "variant_id": row.get("variant_id", ""),
                             "status": st, "binding": bd})
            if st.startswith(RELAX_PREFIX):
                relax_binding_ct[bd] += 1
                rec = {"video": video}
                rec.update({c: row.get(c, "") for c in RELAX_COLS})
                rec["status"] = st
                # tau_floor 접미사 여부는 "게이트를 풀어야 하나 / 하한을 풀어야 하나"를 가른다.
                rec["tau_floor"] = st.endswith("+tau_floor")
                relax_rows.append(rec)

    total = sum(status_ct.values())
    print(f"{'뱅크':<14}{len(paths)}편  ({args.output_root}/*/{args.bank_dir})")
    print(f"{'변이':<14}{total}행\n")

    print("== status 분포")
    for st, n in status_ct.most_common(args.top):
        print(f"  {st or '(blank)':<28}{n:>7}  {100.0 * n / total:5.1f}%")

    print("\n== binding 분포 (첫 발화 게이트)")
    for bd, n in binding_ct.most_common(args.top):
        print(f"  {bd:<28}{n:>7}  {100.0 * n / total:5.1f}%")

    if pick_rows:
        # 수율의 분모는 **변이 행이 아니라 씬**이다 (`--pick_budget` 은 씬당 예산이므로).
        zero = len(paths) - len(pick_per_video)
        holes = sorted(pick_holes)
        med = holes[len(holes) // 2] if holes else float("nan")
        mean = sum(holes) / len(holes) if holes else float("nan")
        print(f"\n== pick 수율 (`picked` 열)")
        print(f"  {'카메라':<28}{pick_rows:>7}  / 씬 {len(paths)}  "
              f"= {100.0 * pick_rows / len(paths):5.1f}%")
        print(f"  {'0대로 끝난 씬':<28}{zero:>7}  {100.0 * zero / len(paths):5.1f}%")
        print(f"  {'track 어휘':<28}{pick_track:>7}  {100.0 * pick_track / pick_rows:5.1f}%")
        print(f"  {'hole  med / mean':<28}{med:>7.4f}  / {mean:.4f}")
        over = Counter(pick_per_video.values())
        for k, n in sorted(over.items()):
            print(f"  {f'씬당 {k}대':<28}{n:>7}")
        print("  -- plan_tier (0=routed track, 1=평범 짝, 2=나머지 track)")
        for t, n in sorted(pick_tier_ct.items()):
            print(f"  {'  tier ' + str(t):<28}{n:>7}  {100.0 * n / pick_rows:5.1f}%")
        print("  -- preset 상위")
        for pr, n in pick_preset_ct.most_common(12):
            print(f"  {'  ' + pr:<28}{n:>7}  {100.0 * n / pick_rows:5.1f}%")

    nrelax = len(relax_rows)
    print(f"\n== 완화 후보 ({RELAX_PREFIX}*) {nrelax}행  {100.0 * nrelax / total:5.1f}%")
    for bd, n in relax_binding_ct.most_common():
        share = 100.0 * n / nrelax if nrelax else 0.0
        print(f"  {bd:<28}{n:>7}  {share:5.1f}%")
    ntf = sum(1 for r in relax_rows if r["tau_floor"])
    print(f"  {'(그중 +tau_floor)':<28}{ntf:>7}  "
          f"{100.0 * ntf / nrelax if nrelax else 0.0:5.1f}%   ← 게이트가 아니라 knob 하한이 범인")

    if args.csv:
        makedirs(path.dirname(path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video", "variant_id", "status", "binding"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n{'csv':<14}{args.csv}")

    if args.relax_list:
        makedirs(path.dirname(path.abspath(args.relax_list)), exist_ok=True)
        by_binding = defaultdict(list)
        for r in relax_rows:
            by_binding[r["binding"]].append(r)
        out = {"format": "bank_relax_candidates_v1",
               "output_root": args.output_root, "bank_dir": args.bank_dir,
               "videos": len(paths), "rows_total": total,
               "status_counts": dict(status_ct.most_common()),
               "binding_counts": dict(binding_ct.most_common()),
               "relax_prefix": RELAX_PREFIX, "relax_count": nrelax,
               "relax_binding_counts": dict(relax_binding_ct.most_common()),
               "relax_tau_floor_count": ntf,
               "by_binding": {k: v for k, v in sorted(by_binding.items())}}
        with open(args.relax_list, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"{'relax_list':<14}{args.relax_list}  ({nrelax}행, binding {len(by_binding)}종)")


if __name__ == "__main__":
    main()
