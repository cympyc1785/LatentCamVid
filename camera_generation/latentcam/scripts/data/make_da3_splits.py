#!/usr/bin/env python3
"""DL3DV 1K 의 da3 코퍼스를 train/test segment 리스트로 분할한다.

da3 = 각 scene 의 `<scene>/da3/` 아래에 새로 추출해 둔 depth / pose / intrinsics / avg_scale /
      camera prompt 묶음. 기존 최상위 `transforms.json` + `prompts.json` + `avg_scale/` 와
      **세그먼트 경계와 키가 동일**하다 (둘 다 49프레임 비중첩 윈도우, 키 '0','1',...; 실측 확인).
      따라서 여기서 뱉는 리스트는 cfg.train_seg_list / cfg.test_seg_list 에 그대로 꽂으면 된다
      (main/base.py 의 seg-list split 경로, 한 줄 = `<batch>/<hash>/<seg>`).

기존 make_latentcam_splits.py 와 다른 점 두 가지:
  (1) CamDataset 을 세우지 않는다. da3/prompts.json 을 직접 읽어 세그먼트를 열거하므로
      transforms.json 파싱도, 인덱스 캐시도 필요 없다.
  (2) **scene-disjoint 분할**이다. c4d2k5y4 를 만든 (1) 경로는 segment 단위 random_split 이라
      같은 scene 의 세그먼트가 train 과 test 양쪽에 들어간다 (docs/known_issues.md 의 미해결
      이슈). da3 코퍼스는 새로 시작하는 것이라 물려받을 비교 대상이 없어서, 여기서는 처음부터
      scene 을 겹치지 않게 나눈다. 리스트 자체는 요청대로 segment 단위다.

검증 (통과 못 한 scene/segment 는 제외하고 사유를 집계해 출력):
  - scenes 루트의 blacklist.csv 에 있는 scene 제외 (dataset_dl3dv._read_blacklist_hashes 와 동일 규칙)
  - da3/{prompts.json, pose.npz, depth.npz, conf.npz, avg_scale/} 존재
  - segment 의 frame_idx [s,e) 가 pose.npz 의 프레임 수 안에 들어갈 것
  - da3/avg_scale/<seg>.json 이 있고 유한한 양수일 것
  - prompt_camera 가 비어 있지 않을 것

출력 순서: test 리스트는 seed 로 셔플해서 쓴다. base.py 의 val 로더는 shuffle=False 로
  test 리스트의 **앞에서부터** val_max_batches x batch_size 개를 잘라 쓰기 때문에, 정렬해 두면
  val 이 앞쪽 몇십 개 scene 에만 몰린다. train 리스트는 로더가 shuffle=True 라 순서 무관이지만
  같은 규칙으로 셔플한다.

Run: python scripts/data/make_da3_splits.py [--batch 1K] [--train-frac 0.9] [--seed 42]
"""
import argparse
import csv
import json
import os
import os.path as osp
import random
from collections import Counter

import numpy as np

ROOT = "/data1/cympyc1785/data/DL3DV/scenes"


def read_blacklist(root):
    p = osp.join(root, "blacklist.csv")
    blocked = set()
    if osp.isfile(p):
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                blocked.add(row["scene"].strip().split("/")[-1])
    return blocked


def scan_scene(sdir):
    """-> (segments, reason). segments = 유효한 seg key 리스트. 실패 시 ([], 사유)."""
    d = osp.join(sdir, "da3")
    for f in ("prompts.json", "pose.npz", "depth.npz", "conf.npz"):
        if not osp.isfile(osp.join(d, f)):
            return [], f"no_{f}"
    if not osp.isdir(osp.join(d, "avg_scale")):
        return [], "no_avg_scale_dir"
    try:
        prompts = json.load(open(osp.join(d, "prompts.json")))
        nframes = int(np.load(osp.join(d, "pose.npz"))["extrinsics"].shape[0])
    except Exception as e:
        return [], f"unreadable({type(e).__name__})"

    segs, bad = [], Counter()
    for k in sorted(prompts, key=lambda x: int(x) if str(x).isdigit() else x):
        v = prompts[k]
        fi = v.get("frame_idx")
        if not (isinstance(fi, (list, tuple)) and len(fi) == 2):
            bad["seg_no_frame_idx"] += 1
            continue
        s, e = int(fi[0]), int(fi[1])
        if not (0 <= s < e <= nframes):
            bad["seg_frame_range"] += 1
            continue
        ap = osp.join(d, "avg_scale", f"{k}.json")
        if not osp.isfile(ap):
            bad["seg_no_avg_scale"] += 1
            continue
        try:
            a = float(json.load(open(ap)))
        except Exception:
            try:
                a = float(open(ap).read().strip())
            except Exception:
                bad["seg_avg_scale_unreadable"] += 1
                continue
        if not np.isfinite(a) or a <= 0:
            bad["seg_avg_scale_bad"] += 1
            continue
        if not str(v.get("prompt_camera") or "").strip():
            bad["seg_no_prompt_camera"] += 1
            continue
        segs.append(str(k))
    return segs, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--batch", default="1K", help="DL3DV batch dir to scan")
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="output name tag (default: da3_<batch 소문자>)")
    args = ap.parse_args()
    tag = args.tag or f"da3_{args.batch.lower()}"

    bdir = osp.join(args.root, args.batch)
    blocked = read_blacklist(args.root)
    hashes = sorted(h for h in os.listdir(bdir) if osp.isdir(osp.join(bdir, h)))
    print(f"{args.batch}: {len(hashes)} scene dirs | blacklist {len(blocked)} scenes (전 batch 합)")

    scene_segs, drop = {}, Counter()
    for h in hashes:
        if h in blocked:
            drop["blacklisted"] += 1
            continue
        segs, bad = scan_scene(osp.join(bdir, h))
        if isinstance(bad, str):
            drop[bad] += 1
            continue
        for k, v in bad.items():
            drop[k] += v
        if not segs:
            drop["no_valid_segment"] += 1
            continue
        scene_segs[f"{args.batch}/{h}"] = segs

    n_seg = sum(len(v) for v in scene_segs.values())
    print(f"유효: {len(scene_segs)} scene / {n_seg} segment")
    if drop:
        print("제외 사유:", dict(sorted(drop.items(), key=lambda x: -x[1])))

    def lines(ss):
        out = [f"{sc}/{k}" for sc in ss for k in scene_segs[sc]]
        rng.shuffle(out)              # base.py val 로더가 test 리스트 앞에서부터 자르므로 셔플
        return out

    # meta_<tag>.csv — dataset_dl3dv 는 cfg.meta_csv 로 "어떤 scene 을 인덱싱할지"를 정한다.
    # 이걸 안 좁히면 CamDataset 이 meta.csv 의 8048 scene 을 전부 훑고 나서 base.py 가 seg 리스트로
    # 걸러내는 꼴이라, 인덱스 빌드가 쓸데없이 8배 비싸진다. 유효 973 scene 만 남긴 csv 를 같이 뱉는다.
    src_meta = osp.join(args.root, "meta.csv")
    meta_name = f"meta_{tag}.csv"
    with open(src_meta, newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else ["chunk", "height", "width", "num_images"]
    keep = [r for r in rows if r["chunk"].strip() in scene_segs]
    # meta.csv 는 최상위 prompts.json 을 요구하고 만들어졌다(filter_dl3dv.py --require-prompts).
    # da3 가 그 scene 들에도 캡션을 새로 붙였으므로 이제 쓸 수 있다 -> 같은 검증(valid_scene)을
    # 그대로 돌려서 통과하는 것만 행을 새로 만들어 넣는다. 규칙을 복제하지 않고 원본을 호출한다.
    have = {r["chunk"].strip() for r in keep}
    add, addfail = [], Counter()
    for c in sorted(set(scene_segs) - have):
        from filter_dl3dv import valid_scene
        row, why = valid_scene(osp.join(args.root, c))
        if row is None:
            addfail[why] += 1
            del scene_segs[c]          # 인덱스에 못 들어갈 scene 은 seg 리스트에서도 뺀다
        else:
            add.append(row)
    if add or addfail:
        print(f"  meta.csv 에 없던 da3 scene: 검증 통과 {len(add)}개 추가"
              + (f" / 탈락 {dict(addfail)}" if addfail else ""))
    keep = sorted(keep + add, key=lambda r: r["chunk"])
    with open(osp.join(args.root, meta_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(keep)
    assert len(keep) == len(scene_segs), (len(keep), len(scene_segs))

    # scene-disjoint 분할 (meta 보정으로 scene_segs 가 줄었을 수 있으니 여기서 확정)
    perm = sorted(scene_segs)
    rng = random.Random(args.seed)
    rng.shuffle(perm)
    ntr = int(args.train_frac * len(perm))
    tr_scenes, te_scenes = sorted(perm[:ntr]), sorted(perm[ntr:])
    assert not (set(tr_scenes) & set(te_scenes))

    tr_lines, te_lines = lines(tr_scenes), lines(te_scenes)
    paths = {meta_name: len(keep)}
    for name, data in (
        (f"latentcam_{tag}_train_seg_list.txt", tr_lines),
        (f"latentcam_{tag}_test_seg_list.txt", te_lines),
        (f"latentcam_{tag}_train_scene_list.txt", tr_scenes),
        (f"latentcam_{tag}_test_scene_list.txt", te_scenes),
    ):
        p = osp.join(args.root, name)
        with open(p, "w") as f:
            f.write("\n".join(data) + "\n")
        paths[name] = len(data)

    print(f"\nscene-disjoint {args.train_frac:.2f}/{1 - args.train_frac:.2f} (seed {args.seed})")
    print(f"  train {len(tr_scenes)} scene / {len(tr_lines)} segment")
    print(f"  test  {len(te_scenes)} scene / {len(te_lines)} segment")
    for k, v in paths.items():
        print(f"  {v:6d}  {osp.join(args.root, k)}")


if __name__ == "__main__":
    main()
