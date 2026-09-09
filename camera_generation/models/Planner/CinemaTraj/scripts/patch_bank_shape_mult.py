"""이미 만들어진 hole 뱅크의 `shape_mult` 를 고친다 — `fit_hole_ladder` 재실행(영상당 ~1350 s) 없이.

**무엇이 틀렸었나.** `fit_hole_ladder.probe` 의 모양 확대 루프가 `mult` 를 판정 **뒤에** 곱했다:

    mult = 1.0
    for _ in range(shape_doublings + 1):
        poses = build(mult)
        if scale < MAX: break
        mult *= headroom          # <- 마지막 시도에서도 곱하고 루프가 끝난다

루프가 끝까지 다 돌면(= 마지막 시도도 `max_scale` 에 붙음 = `status == "shape_limited"`)
`poses` 를 만든 배율은 `mult / headroom` 인데 행에는 `mult` 가 찍힌다. 그러면 `emit_bank` 가
그 배율로 궤적을 되만들 때 **두 배 큰 궤적**이 나온다 (실측: basketball-four
`dyn_0__pull_out_arc__hole0.5`, poses.npz 와 최대 8.918 어긋남 → `emit_bank` rc=1).

행에 실린 측정치(hole, path_len, tau_max 등)는 전부 **작은 쪽 궤적**에서 잰 것이라 맞다.
틀린 건 `shape_mult` 열 하나뿐이므로, 그 열만 `headroom` 으로 나누면 행이 정합해진다.
고쳤는지는 짐작하지 말고 `emit_bank.py` 를 다시 돌려 "pose 재현 최대오차 0" 을 보면 된다.

`status` 에 `shape_limited` 가 들어 있고 `shape_mult > 1` 인 행만 건드린다. 다른 status 는
루프가 `break` 로 빠져나온 것이라 이미 맞다.

**두 번 돌리면 안 된다** — 조건이 `shape_mult > 1` 이라 16 도 다시 걸려 8 이 된다. 그래서
뱅크 최상위 `shape_mult_semantics: "as_built"` 를 표식으로 쓴다: 고친 뱅크와 고쳐진
`fit_hole_ladder` 가 만든 뱅크 양쪽에 이게 붙고, 붙어 있으면 이 스크립트는 건너뛴다.

사용 예시:
    python scripts/patch_bank_shape_mult.py --videos basketball-four --dry_run
    python scripts/patch_bank_shape_mult.py                 # out/ 전체, .bak 남기고 수정
    python scripts/patch_bank_shape_mult.py --no_backup --bank_dir hole_bank
"""
import csv
import json
import shutil
from argparse import ArgumentParser
from glob import glob
from os import path

OUT_DEFAULT = path.join(path.dirname(path.dirname(path.abspath(__file__))), "out")


def patch_rows(variants: list, headroom: float):
    """고친 행의 (variant_id, 이전 배율, 이후 배율) 목록. `variants` 는 제자리에서 바뀐다."""
    changed = []
    for row in variants:
        mult = float(row.get("shape_mult", 1.0) or 1.0)
        if "shape_limited" in str(row.get("status", "")) and mult > 1.0:
            row["shape_mult"] = mult / headroom
            changed.append((row["variant_id"], mult, mult / headroom))
    return changed


def patch_csv(csv_path: str, fixes: dict):
    """`bank.csv` 의 같은 행도 맞춰 둔다 — 사람이 보는 표가 json 과 어긋나면 안 된다."""
    if not path.isfile(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields, rows = reader.fieldnames, list(reader)
    hit = 0
    for row in rows:
        if row["variant_id"] in fixes:
            row["shape_mult"] = repr(fixes[row["variant_id"]])
            hit += 1
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hit


def main(args):
    banks = sorted(glob(path.join(args.out, "*", args.bank_dir, "bank.json")))
    if args.videos:
        wanted = set(args.videos.split(","))
        banks = [b for b in banks
                 if path.basename(path.dirname(path.dirname(b))) in wanted]

    table = []
    for bank_path in banks:
        video = path.basename(path.dirname(path.dirname(bank_path)))
        with open(bank_path, encoding="utf-8") as file:
            bank = json.load(file)
        if bank.get("shape_mult_semantics") == "as_built":
            table.append((video, 0, 0, "이미 정상"))
            continue
        changed = patch_rows(bank["variants"], args.headroom)
        if not changed:
            # 고칠 행이 없어도 표식은 남긴다 — 다음 실행이 또 훑지 않게.
            if not args.dry_run:
                bank["shape_mult_semantics"] = "as_built"
                with open(bank_path, "w", encoding="utf-8") as file:
                    json.dump(bank, file, ensure_ascii=False, indent=1)
            table.append((video, 0, 0, "해당 행 없음"))
            continue
        if args.dry_run:
            table.append((video, len(changed), 0, "dry_run"))
            for vid, before, after in changed:
                print(f"    {vid}  {before:g} -> {after:g}")
            continue
        if args.backup and not path.isfile(bank_path + ".bak"):
            shutil.copy2(bank_path, bank_path + ".bak")
        bank["shape_mult_semantics"] = "as_built"
        with open(bank_path, "w", encoding="utf-8") as file:
            json.dump(bank, file, ensure_ascii=False, indent=1)
        csv_hit = patch_csv(path.join(path.dirname(bank_path), "bank.csv"),
                            {vid: after for vid, _, after in changed})
        table.append((video, len(changed), csv_hit, "patched"))

    print(f"\n{'video':24s} {'json행':>6s} {'csv행':>6s}  상태")
    for video, n_json, n_csv, state in table:
        print(f"{video:24s} {n_json:6d} {n_csv:6d}  {state}")
    total = sum(row[1] for row in table)
    print(f"\n뱅크 {len(table)}개 중 고친 행 {total}개"
          f"{'  (dry_run — 아무것도 안 썼다)' if args.dry_run else ''}")
    if total and not args.dry_run:
        print("확인: 해당 영상에 `python scripts/emit_bank.py --video <v>` 를 돌려\n"
              "      'pose 재현 최대오차 0.000e+00' 이 나오는지 볼 것.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", default=OUT_DEFAULT, type=str)
    parser.add_argument("--bank_dir", default="hole_bank", type=str)
    parser.add_argument("--videos", default="", type=str)   # 쉼표 구분, 비우면 전체
    # `fit_hole_ladder --shape_headroom` 과 같은 값이어야 한다. 다르게 돌렸다면 그 값을 줄 것.
    parser.add_argument("--headroom", default=2.0, type=float)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no_backup", dest="backup", action="store_false")
    main(parser.parse_args())
