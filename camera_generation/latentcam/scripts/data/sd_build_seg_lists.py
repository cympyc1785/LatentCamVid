"""Scene-Decoupled 학습용 (context clip, target clip) 쌍 리스트를 **고정**해서 뽑는다.

왜 리스트를 미리 박는가: SD 는 같은 scene 의 다른 clip 을 context 로 쓰는데
(scene-decoupled-cross-clip-pairing), clip 마다 da3 재구성 품질이 다르고 static clip 은
umeyama 가 아예 안 풀린다. 매 epoch 무작위로 짝을 고르면 어떤 쌍이 학습에 들어갔는지
재현이 안 되고, arm 끼리(customgeo vs textonly) 본 데이터가 달라져 비교가 깨진다.

필터 두 개:
  static      원래는 `umeyama_gt.json` 의 `moving: false` 로 판정했다. 그런데 2026-08-11
              `sd_static_scale_transfer.py` 가 7682 개를 되살려 **`moving` 은 이제 전부 true**
              다 (s/t/matrix4 가 같은 scene moving clip 의 중앙값 s 로 채워졌고
              `s_source: scene_moving_median` / `was_static: true` 로 표시된다).
              그래서 판정은 `was_static` 으로 한다. 이들은 `resid_rmse_over_rad` 가 아예 없어
              (자기 궤적으로 sim3 를 푼 게 아니다) 아래 align 컷에서도 자동 탈락한다.
  align pXX   `resid_rmse_over_rad` (sim3 정렬 잔차 / 궤적 반경) 의 코퍼스 백분위 초과 clip 제외
              (`--resid-pct`, 기본 99 = 기존 리스트와 동일). 2026-08-10 재생성
              (`convention: gtrot`) 이후 이 값은 실제로 쓸 만한 필터다: cross-clip frame-0
              중심 불일치와 corr(log) = +0.61 (rot_spread_deg 는 +0.57).
              `--rot-p99` 를 주면 rot_spread_deg p99 컷도 같이 건다 (기본 off).
              **어디로 낮출지는 `scripts/data/sd_anchor_error_filters.py` 실측을 본다.**
              p90 (=0.0928) 이면 pair 58044 -> 49328 (85.0%) 이고, DA3 view0 와 target 앵커의
              불일치 / target motion 비가 `>0.5` 인 pair 가 1.74% -> 0.31%, max 3.81 -> 1.14
              로 준다. 단 살아남는 clip 집합이 바뀌므로 **scene 단위 분할도 달라진다**
              (p99 3006/334 scene -> p90 2925/325) — 기존 SD arm 과 paired 비교가 깨진다.

쌍 열거: 한 scene 안에서 살아남은 clip 들의 **순서 있는** 모든 (target, ctx) 쌍.
분할: **scene 단위** 90/10 (같은 scene 이 train 과 test 에 동시에 들어가면 context 가 새는 셈).

출력 (--out-dir, 기본 <데이터셋 루트>/latentcam_lists, 접미사는 --tag):
  sd_<split><tag>_train.txt / _test.txt        한 줄 = data_name = "<scene>__<target>__<ctx>"
  sd_<split><tag>_lists_summary.md             임계값과 탈락 수 (원본 수치 그대로)
  sd_<split><tag>_clip_stats.csv               clip 별 moving/was_static/resid/rot_spread/keep

env 없음. CLI: --split whuman --train-frac 0.9 --seed 42 [--resid-pct 90] [--tag _r90] [--rot-p99]
"""
import argparse
import csv
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"


def scan_clip(args):
    scene, clip = args
    p = os.path.join(ROOT, "da3", scene, clip, "umeyama_gt.json")
    try:
        u = json.load(open(p))
    except Exception:
        return dict(scene=scene, clip=clip, ok=0, moving=0, was_static=0,
                    resid=float("nan"), rot_spread=float("nan"))
    mv = bool(u.get("moving", False))
    # 2026-08-11 `sd_static_scale_transfer.py` 가 static clip 7682 개를 되살렸다: moving 이
    # true 로 뒤집히고 s/t/matrix4 가 채워졌다 (`s_source: scene_moving_median`,
    # `was_static: true`). 다만 **`resid_rmse_over_rad` 는 없다** — 자기 궤적으로 sim3 를 푼 게
    # 아니라 같은 scene 의 moving clip 중앙값 s 를 빌려온 것이라 잔차를 정의할 수 없다.
    # 그래서 static 판정은 `moving` 이 아니라 `was_static` 으로 본다 (moving 은 이제 전부 true).
    return dict(scene=scene, clip=clip, ok=1, moving=int(mv),
                was_static=int(bool(u.get("was_static", False))),
                resid=float(u.get("resid_rmse_over_rad", float("nan"))) if mv else float("nan"),
                rot_spread=float(u.get("rot_spread_deg", float("nan"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="whuman")
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rot-p99", action="store_true",
                    help="rot_spread_deg p99 컷도 같이 건다 (기본 off — resid 컷만)")
    ap.add_argument("--resid-pct", type=float, default=99.0,
                    help="resid_rmse_over_rad 컷 백분위 (기본 99 = 기존 리스트와 동일). "
                         "낮출수록 cross-clip 앵커 잔차가 줄어든다 — 실측은 "
                         "scripts/data/sd_anchor_error_filters.py 의 report.md")
    ap.add_argument("--tag", default="",
                    help="출력 파일 접미사. 기존 리스트를 덮어쓰지 않으려면 반드시 줄 것 "
                         "(예: --tag _r90 -> sd_whuman_r90_train.txt)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "latentcam_lists"))
    a = ap.parse_args()

    da3 = os.path.join(ROOT, "da3", a.split)
    scenes = sorted(x for x in os.listdir(da3)
                    if x != "logs" and os.path.isdir(os.path.join(da3, x)))
    jobs = []
    for sc in scenes:
        for c in sorted(os.listdir(os.path.join(da3, sc))):
            if c != "logs" and os.path.isdir(os.path.join(da3, sc, c)):
                jobs.append((f"{a.split}/{sc}", c))
    print(f"[scan] {len(scenes)} scene / {len(jobs)} clip")
    with ThreadPoolExecutor(32) as ex:
        rows = list(ex.map(scan_clip, jobs))

    moving = [r for r in rows if r["ok"] and r["moving"]]
    resid = np.array([r["resid"] for r in moving], dtype=np.float64)
    rot = np.array([r["rot_spread"] for r in moving], dtype=np.float64)
    resid_p99 = float(np.nanpercentile(resid, a.resid_pct))
    rot_p99 = float(np.nanpercentile(rot, 99))

    for r in rows:
        keep = bool(r["ok"] and r["moving"])
        if keep and not (np.isfinite(r["resid"]) and r["resid"] <= resid_p99):
            keep = False
        if keep and a.rot_p99 and not (np.isfinite(r["rot_spread"]) and r["rot_spread"] <= rot_p99):
            keep = False
        r["keep"] = int(keep)

    by_scene = {}
    for r in rows:
        if r["keep"]:
            by_scene.setdefault(r["scene"], []).append(r["clip"])
    usable = {k: sorted(v) for k, v in by_scene.items() if len(v) >= 2}

    rng = random.Random(a.seed)
    sc_list = sorted(usable)
    rng.shuffle(sc_list)
    n_tr = int(round(a.train_frac * len(sc_list)))
    tr_scenes, te_scenes = set(sc_list[:n_tr]), set(sc_list[n_tr:])

    def pairs_of(sc):
        cs = usable[sc]
        base = sc.split("/", 1)[1]
        return [f"{base}__{t}__{c}" for t in cs for c in cs if t != c]

    tr = [x for sc in sorted(tr_scenes) for x in pairs_of(sc)]
    te = [x for sc in sorted(te_scenes) for x in pairs_of(sc)]

    os.makedirs(a.out_dir, exist_ok=True)
    pre = os.path.join(a.out_dir, f"sd_{a.split}{a.tag}")
    open(pre + "_train.txt", "w").write("\n".join(tr) + "\n")
    open(pre + "_test.txt", "w").write("\n".join(te) + "\n")
    with open(pre + "_clip_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["scene", "clip", "ok", "moving", "was_static", "resid",
                               "rot_spread", "keep"])
        w.writeheader()
        w.writerows(rows)

    n_static = sum(1 for r in rows if r["ok"] and (not r["moving"] or r["was_static"]))
    n_cut = sum(1 for r in rows if r["ok"] and r["moving"] and not r["was_static"]
                and not r["keep"])
    n_bad = sum(1 for r in rows if not r["ok"])
    L = [f"# Scene-Decoupled 학습 리스트  split={a.split}  seed={a.seed}  "
         f"train_frac={a.train_frac}", "",
         f"clip 총 {len(rows)}  (umeyama_gt.json 읽기 실패 {n_bad})",
         f"  static 제외 (moving=false 또는 was_static -> resid 없음)  {n_static}",
         f"  align 컷 제외                    {n_cut}",
         f"  남은 clip                        {sum(r['keep'] for r in rows)}", "",
         f"컷 기준 (moving clip {len(moving)} 개 위에서 잰 코퍼스 백분위)",
         f"  resid_rmse_over_rad  p50 {np.nanmedian(resid):.4f}  p90 "
         f"{np.nanpercentile(resid,90):.4f}  p99 {np.nanpercentile(resid,99):.4f}  "
         f"max {np.nanmax(resid):.4f}   **컷 = p{a.resid_pct:g} = {resid_p99:.4f}**",
         f"  rot_spread_deg       p50 {np.nanmedian(rot):.4f}  p90 "
         f"{np.nanpercentile(rot,90):.4f}  p99 {rot_p99:.4f}  max {np.nanmax(rot):.4f}"
         + ("   <- 이 컷도 적용" if a.rot_p99 else "   (컷 미적용)"), "",
         f"clip 2장 이상 남은 scene {len(usable)} / {len(scenes)}",
         f"  train scene {len(tr_scenes)}  -> pair {len(tr)}",
         f"  test  scene {len(te_scenes)}  -> pair {len(te)}", "",
         "한 줄 = data_name = '<scene>__<target_clip>__<ctx_clip>' (같은 scene 안 순서 있는 모든 쌍).",
         "scene 단위로 갈랐으므로 test scene 의 clip 은 train 에 context 로도 등장하지 않는다."]
    txt = "\n".join(L) + "\n"
    open(pre + "_lists_summary.md", "w").write(txt)
    print("\n" + txt + f"saved -> {pre}_{{train,test}}.txt")


if __name__ == "__main__":
    main()
