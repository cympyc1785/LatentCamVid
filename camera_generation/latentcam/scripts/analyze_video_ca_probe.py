"""`video_ca_probe.jsonl` 을 preset / scene 축으로 집계한다.

`scripts/eval_testset.py --probe-video-ca` 가 추론 1건마다 한 행씩 떨군 파일을 읽어,
`data_name` 을 코퍼스 `prompts.json` 에 조인해서 preset · scene · anchor · aim · τ 사다리 단으로
묶는다. 사용자 지시 "각 inference마다 기록해서 preset마다, scene마다 특성이 있는지도 분석해줘".

읽는 열의 뜻은 `main/video_ca_probe.py` docstring 에 있다. 여기서 판정에 쓰는 것만 다시 적으면:
  vresid_mean     ||gate_i * v_i|| / ||h_i|| 의 층 평균 = video 스트림의 잔차 기여율
                  (`text_resid_mean` / `geo_resid_mean` 이 같은 정의의 눈금자)
  dpred_drop      video 조건을 빼면 예측이 얼마나 변하나 (스트림 총 기여)
  dpred_shuffle   video 조건만 배치축으로 roll — **scene 짝만** 깬다. 0 이면 내용을 안 본다는 뜻
  vattn_video_mass  molmo2 text 128 토큰이 아니라 프레임 패치 3136 쪽으로 간 attention 질량

집계는 **평균과 함께 분산·표본수를 같이 낸다**. preset 별 표본이 한 자릿수인 칸이 흔하고,
그 칸의 평균 하나만 보면 없는 구조를 읽게 된다.

사용 예시:
    python scripts/analyze_video_ca_probe.py \
        --probe eval_my/20260904_185554_vista4d_d121_molmo2__epoch100/video_ca_probe.jsonl \
        --out_dir results/20260905_d124_video_ca_probe
"""
import json
from argparse import ArgumentParser
from collections import defaultdict
from os import makedirs, path
from statistics import mean, median, pstdev

CORPUS_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121"

# 이 열들만 표로 낸다. 순서가 곧 표의 열 순서다.
METRICS = ["vresid_mean", "text_resid_mean", "geo_resid_mean",
           "dpred_drop", "dpred_shuffle",
           "vattn_video_mass", "vattn_entropy_norm", "vattn_frame_peak",
           "vattn_frame_token_r"]


def load_rows(probe_path: str):
    with open(probe_path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def join_corpus(rows, corpus_root: str, prefix: str):
    """`data_name` = `<prefix>_<scene>_<seg>` -> prompts.json 의 preset/anchor/aim/hole/tau 를 붙인다.

    씬 이름에 `_` 가 들어가는 경우가 있어(vista `couple-hug` 는 없지만 dynpose uuid 는 있다)
    앞에서 prefix, 뒤에서 seg 를 떼는 방식으로 자른다.
    """
    cache, missed = {}, 0
    for row in rows:
        name = row["data_name"]
        scene = name[len(prefix) + 1:].rsplit("_", 1)[0]
        seg = name.rsplit("_", 1)[1]
        if scene not in cache:
            jp = path.join(corpus_root, prefix, scene, "da3", "prompts.json")
            with open(jp, encoding="utf-8") as file:
                cache[scene] = json.load(file)
        entry = cache[scene].get(seg)
        row["scene"] = scene
        if entry is None:
            missed += 1
            continue
        row["preset"] = entry.get("preset", "")
        row["preset_raw"] = entry.get("preset_raw", "")
        row["aim"] = entry.get("aim", "")
        row["anchor_label"] = entry.get("anchor_label", "")
        row["variant_id"] = entry.get("variant_id", "")
        row["anchor_kind"] = entry.get("variant_id", "").split("_")[0]   # dyn / stat
        row["tau_max"] = entry.get("tau_max")
        row["hole_fraction"] = entry.get("hole_fraction")
        row["subject_area_med"] = entry.get("subject_area_med")
    return missed


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "mean": mean(values), "sd": pstdev(values) if len(values) > 1 else 0.0,
            "med": median(values), "min": min(values), "max": max(values)}


def group_table(rows, key, metrics, min_n=1):
    groups = defaultdict(list)
    for row in rows:
        if key in row:
            groups[row[key]].append(row)
    out = {}
    for name, grp in groups.items():
        if len(grp) < min_n:
            continue
        out[name] = {"n": len(grp),
                     **{m: _stats([r.get(m) for r in grp]) for m in metrics}}
    return out


def print_table(title, table, metrics, sort_by="vresid_mean", top=None):
    print(f"\n== {title}  (n 은 추론 건수, 값은 mean±sd)")
    head = f"  {'key':28s} {'n':>5s} " + " ".join(f"{m[:16]:>16s}" for m in metrics)
    print(head)
    rows = sorted(table.items(),
                  key=lambda kv: -(kv[1].get(sort_by) or {}).get("mean", float("-inf")))
    if top:
        rows = rows[:top]
    for name, st in rows:
        line = f"  {str(name)[:28]:28s} {st['n']:5d} "
        for m in metrics:
            s = st.get(m)
            line += f"{s['mean']:8.4f}±{s['sd']:6.4f} " if s else f"{'-':>16s} "
        print(line)


def corr(rows, a, b):
    """행별 Pearson r. 결측이 있는 행은 뺀다."""
    pairs = [(r.get(a), r.get(b)) for r in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None, 0
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return (num / den if den > 1e-12 else 0.0), len(pairs)


def main():
    parser = ArgumentParser()
    parser.add_argument("--probe", required=True)              # video_ca_probe.jsonl 경로
    parser.add_argument("--corpus_root", default=CORPUS_DEFAULT)
    parser.add_argument("--name_prefix", default="vista4d")    # data_name 접두사 = 코퍼스 하위 폴더명
    parser.add_argument("--out_dir", default=None)             # 주면 summary.json 을 여기 쓴다
    parser.add_argument("--min_n", type=int, default=3)        # 표에서 뺄 표본 하한
    args = parser.parse_args()

    rows = load_rows(args.probe)
    missed = join_corpus(rows, args.corpus_root, args.name_prefix)
    print(f"[load] {len(rows)} inferences from {args.probe}")
    if missed:
        print(f"[warn] prompts.json 에서 못 찾은 entry {missed}건 (preset/scene 축에서 빠진다)")

    gate_keys = sorted((k for k in rows[0] if k.startswith("gate_l")),
                       key=lambda k: int(k[6:]))
    print("\n== video_gate (샘플 무관, ckpt 속성) 와 층별 실제 기여율")
    print(f"  {'layer':7s} {'gate':>12s} {'|gate|':>12s} {'vresid mean':>14s} {'vresid sd':>12s}")
    for k in gate_keys:
        li = k[6:]
        vs = [r[f"vresid_l{li}"] for r in rows if f"vresid_l{li}" in r]
        g = rows[0][k]
        print(f"  l{li:6s} {g:12.6f} {abs(g):12.6f} {mean(vs):14.6f} "
              f"{(pstdev(vs) if len(vs) > 1 else 0.0):12.6f}")

    print("\n== 전체 (n=%d)" % len(rows))
    for m in METRICS:
        s = _stats([r.get(m) for r in rows])
        if s:
            print(f"  {m:22s} n={s['n']:5d}  mean {s['mean']:9.5f}  sd {s['sd']:9.5f}  "
                  f"med {s['med']:9.5f}  [{s['min']:9.5f}, {s['max']:9.5f}]")

    small = ["vresid_mean", "dpred_drop", "dpred_shuffle", "vattn_video_mass",
             "vattn_entropy_norm", "vattn_frame_token_r"]
    t_preset = group_table(rows, "preset", METRICS, args.min_n)
    t_scene = group_table(rows, "scene", METRICS, args.min_n)
    t_anchor = group_table(rows, "anchor_kind", METRICS, args.min_n)
    t_aim = group_table(rows, "aim", METRICS, args.min_n)
    print_table(f"preset 별 (n>={args.min_n})", t_preset, small)
    print_table(f"scene 별 (n>={args.min_n})", t_scene, small)
    print_table("anchor 종류 별 (dyn/stat)", t_anchor, small)
    print_table("aim 별", t_aim, small)

    print("\n== 연속량과의 상관 (Pearson r)")
    for a in ["vresid_mean", "dpred_drop", "dpred_shuffle", "vattn_entropy_norm"]:
        for b in ["tau_max", "hole_fraction", "subject_area_med"]:
            r, n = corr(rows, a, b)
            if r is not None:
                print(f"  {a:20s} vs {b:18s} r={r:+.4f}  n={n}")

    print("\n== 프레임 peak 인덱스 분포 (0..48; 균일이면 어느 프레임도 안 고른다는 뜻)")
    hist = defaultdict(int)
    for r in rows:
        if "vattn_frame_peak_idx" in r:
            hist[r["vattn_frame_peak_idx"]] += 1
    for idx in sorted(hist, key=lambda i: -hist[i])[:12]:
        print(f"  frame {idx:3d}  {hist[idx]:5d}  ({100 * hist[idx] / len(rows):5.1f}%)")

    if args.out_dir:
        makedirs(args.out_dir, exist_ok=True)
        with open(path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as file:
            json.dump({"probe": args.probe, "n": len(rows),
                       "gate": {k: rows[0][k] for k in gate_keys},
                       "overall": {m: _stats([r.get(m) for r in rows]) for m in METRICS},
                       "by_preset": t_preset, "by_scene": t_scene,
                       "by_anchor_kind": t_anchor, "by_aim": t_aim},
                      file, ensure_ascii=False, indent=1)
        print(f"\n[out] {path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
