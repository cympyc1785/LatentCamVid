"""두 뱅크의 결정열을 변이 단위로 대조한다 (렌더 0회, bank.json 만 읽는다).

WHY: 게이트 임계를 하나 바꿨을 때 "몇 개가 실제로 뒤집혔나"를 눈대중이 아니라 세어서 봐야
한다. fit 은 변이마다 `status`(왜 이 크기에서 멈췄나) / `binding`(어느 게이트가 잡았나) /
`knob`(사다리 이분법이 푼 크기)을 남기는데, 임계 변경의 효과는 이 세 열에만 나타난다.
나머지 열(면적 시퀀스, composition 등)은 측정값이라 knob 이 같으면 따라서 같다.

두 뱅크는 **같은 변이 집합**이어야 한다 — `variant_id` 로 조인하고, 한쪽에만 있는 id 는
따로 센다. 한쪽에만 있으면 그건 임계 효과가 아니라 뱅크가 다른 설정으로 구워진 것이다.

사용 예시:
    python scripts/diff_bank_variants.py \
        --base out/parkour/hole_bank_g1probe_off/bank.json \
        --new  out/parkour/hole_bank_g1probe_on/bank.json \
        --base_name "clear 0.10 absolute" --new_name "clear = margin + 0.3 x src_p10"
"""

from argparse import ArgumentParser
from collections import Counter
from os import path
import json


# knob 이 이 값보다 더 벌어져야 "크기가 바뀌었다"고 센다. 이분법 4회라 해상도가 이 언저리다.
KNOB_EPS = 1e-4


def load(p):
    with open(p) as f:
        d = json.load(f)
    return d, {v["variant_id"]: v for v in d["variants"]}


def g1_summary(bank):
    """뱅크 헤더의 G1 임계를 한 줄로."""
    c = bank.get("collision", {})
    return (f"clear_frac {c.get('clear_frac')}  margin_frac {c.get('margin_frac')}  "
            f"src_ratio {c.get('clear_src_ratio', '없음')}  "
            f"src_pct {c.get('clear_src_pct', '없음')}  "
            f"src_frames {len(c.get('src_frames', []))}")


def table(rows, headers):
    """정렬된 print 표."""
    cols = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(h.ljust(cols[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c).ljust(cols[i]) for i, c in enumerate(r)))


def main():
    ap = ArgumentParser()
    ap.add_argument("--base", required=True)          # 기준 뱅크 bank.json
    ap.add_argument("--new", required=True)           # 비교 뱅크 bank.json
    ap.add_argument("--base_name", default="base")    # 표에 찍을 이름
    ap.add_argument("--new_name", default="new")
    ap.add_argument("--top", default=25, type=int)    # 뒤집힌 변이 몇 개까지 나열하나
    ap.add_argument("--out_json", default=None)       # 전량 diff 를 JSON 으로도 저장
    args = ap.parse_args()

    bb, bv = load(args.base)
    nb, nv = load(args.new)

    print(f"base  {args.base_name}")
    print(f"      {args.base}")
    print(f"      {g1_summary(bb)}")
    print(f"new   {args.new_name}")
    print(f"      {args.new}")
    print(f"      {g1_summary(nb)}")
    print()

    only_b = sorted(set(bv) - set(nv))
    only_n = sorted(set(nv) - set(bv))
    common = sorted(set(bv) & set(nv))
    print(f"변이  공통 {len(common)}   base 만 {len(only_b)}   new 만 {len(only_n)}")
    if only_b[:5]:
        print(f"      base 만 예시: {only_b[:5]}")
    if only_n[:5]:
        print(f"      new  만 예시: {only_n[:5]}")
    print()

    flips, changed = [], Counter()
    for vid in common:
        b, n = bv[vid], nv[vid]
        d_status = b["status"] != n["status"]
        d_bind = b.get("binding") != n.get("binding")
        d_knob = abs(float(b["knob"]) - float(n["knob"])) > KNOB_EPS
        if d_status:
            changed["status"] += 1
        if d_bind:
            changed["binding"] += 1
        if d_knob:
            changed["knob"] += 1
        if d_status or d_bind or d_knob:
            changed["any"] += 1
            flips.append({
                "variant_id": vid,
                "status": [b["status"], n["status"]],
                "binding": [b.get("binding"), n.get("binding")],
                "knob": [round(float(b["knob"]), 4), round(float(n["knob"]), 4)],
                "d_knob": round(float(n["knob"]) - float(b["knob"]), 4),
                "behind_frac": [b.get("behind_frac"), n.get("behind_frac")],
                "tau_max": [b.get("tau_max"), n.get("tau_max")],
                "path_len_u": [b.get("path_len_u"), n.get("path_len_u")],
            })

    print(f"바뀐 변이  any {changed['any']}/{len(common)} "
          f"({100.0 * changed['any'] / max(1, len(common)):.1f}%)   "
          f"status {changed['status']}   binding {changed['binding']}   knob {changed['knob']}")
    print()

    # ── status 분포 ────────────────────────────────────────────────────────────────
    sb, sn = Counter(bv[v]["status"] for v in common), Counter(nv[v]["status"] for v in common)
    rows = [[s, sb.get(s, 0), sn.get(s, 0), f"{sn.get(s, 0) - sb.get(s, 0):+d}"]
            for s in sorted(set(sb) | set(sn), key=lambda k: -(sb.get(k, 0) + sn.get(k, 0)))]
    print("status 분포")
    table(rows, ["status", args.base_name, args.new_name, "delta"])
    print()

    # ── binding 분포 — 어느 게이트가 크기를 잡았나 ─────────────────────────────────
    cb = Counter(str(bv[v].get("binding")) for v in common)
    cn = Counter(str(nv[v].get("binding")) for v in common)
    rows = [[s, cb.get(s, 0), cn.get(s, 0), f"{cn.get(s, 0) - cb.get(s, 0):+d}"]
            for s in sorted(set(cb) | set(cn), key=lambda k: -(cb.get(k, 0) + cn.get(k, 0)))]
    print("binding 분포")
    table(rows, ["binding", args.base_name, args.new_name, "delta"])
    print()

    # ── 크기 총량 — 정지 궤적이 얼마나 줄었나 ──────────────────────────────────────
    def stat(bank_v, key):
        xs = [float(bank_v[v][key]) for v in common if bank_v[v].get(key) is not None]
        xs.sort()
        return xs

    print("크기 분포 (공통 변이)")
    rows = []
    for key in ["knob", "path_len_u", "tau_max", "behind_frac"]:
        xb, xn = stat(bv, key), stat(nv, key)
        if not xb or not xn:
            continue
        med = lambda x: x[len(x) // 2]
        rows.append([key,
                     f"{sum(xb) / len(xb):.4f}", f"{sum(xn) / len(xn):.4f}",
                     f"{med(xb):.4f}", f"{med(xn):.4f}",
                     f"{sum(xn) / len(xn) - sum(xb) / len(xb):+.4f}"])
    table(rows, ["열", f"mean {args.base_name}", f"mean {args.new_name}",
                 f"med {args.base_name}", f"med {args.new_name}", "d mean"])
    print()

    # ── 정지 궤적 수 — path_len_u 가 사실상 0 인 변이 ──────────────────────────────
    for thr in [0.01, 0.02]:
        zb = sum(1 for v in common if float(bv[v].get("path_len_u", 0)) < thr)
        zn = sum(1 for v in common if float(nv[v].get("path_len_u", 0)) < thr)
        print(f"path_len_u < {thr}  (사실상 정지):  {args.base_name} {zb}   "
              f"{args.new_name} {zn}   ({zn - zb:+d})")
    print()

    if flips:
        print(f"뒤집힌 변이 상위 {min(args.top, len(flips))} (|d_knob| 큰 순)")
        flips_sorted = sorted(flips, key=lambda f: -abs(f["d_knob"]))
        rows = [[f["variant_id"][:44],
                 f'{f["status"][0]} -> {f["status"][1]}',
                 f'{f["binding"][0]} -> {f["binding"][1]}',
                 f'{f["knob"][0]} -> {f["knob"][1]}',
                 f'{f["d_knob"]:+.4f}'] for f in flips_sorted[:args.top]]
        table(rows, ["variant_id", "status", "binding", "knob", "d_knob"])

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"base": args.base, "new": args.new,
                       "base_g1": bb.get("collision"), "new_g1": nb.get("collision"),
                       "n_common": len(common), "only_base": only_b, "only_new": only_n,
                       "changed": dict(changed), "flips": flips},
                      f, ensure_ascii=False, indent=2)
        print(f"\n저장  {args.out_json}")


if __name__ == "__main__":
    main()
