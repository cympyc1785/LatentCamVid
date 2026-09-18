"""씬 단위 층화 holdout 목록 + seg_list 재작성 — train:test = 9:1, val 은 test 안의 1%.

왜 이 스크립트가 따로 있나
──────────────────────────
D200 까지 holdout 은 `configs/dynpose_holdout_scenes_100.txt` 100씬(=506대, **0.98%**)이었다.
"씬 수준 지표의 n 을 27 -> 100 으로" 가 목적이었지 train:test 비율이 목적이 아니었다.
사용자 지시(2026-09-18)로 **train:test = 9:1, validation = 1%** 를 맞춰야 한다.

재굽기는 필요 없다. `vista4d_bank_to_dl3dv.py:493-519` 를 보면 `--test_videos` 는 **각 줄이
어느 seg_list 로 가는지만** 정하고 chunk/npz 는 건드리지 않는다. 그래서 이 스크립트는
export 를 다시 돌리지 않고 **기존 seg_list 두 개를 읽어 다시 나누기만** 한다
(geo_raw 403 GB / molmo2 305 GB 캐시도 씬 단위라 그대로 재사용).

층화 기준은 **씬당 카메라 수**다. d200 분포가 1:726 / 2:472 / 3:143 / 4:179 / 5:3287 / 6:5362
로 양극단이라, 씬을 그냥 무작위로 뽑으면 test 의 씬당 대수가 train 과 달라진다. 버킷마다 같은
비율로 뽑아 그 편향을 없앤다.

**기존 목록을 삼킨다.** 새 test 는 legacy 27씬 ⊂ 100씬 ⊂ 1,017씬 이 되도록 100씬을 먼저 넣고
나머지를 채운다. 그래야 `test27` / `test100` 지표를 D137/D194/D197/D200 과 계속 대조할 수 있다.

validation 은 별도 목록 파일이 아니라 **test 목록의 앞부분**이다 — `train_latent_cam_dm.py:737`
이 test 로더를 `val_max_batches × batch_size` 만큼만 돌기 때문이다. 그래서 val 씬들을 test
목록 맨 앞에 라운드로빈으로 깔고, 그 뒤에 나머지 test 씬을 역시 라운드로빈으로 잇는다.
val 대수에 맞는 `val_max_batches` 값은 main() 끝 표가 찍어준다.

사용 예:
  python scripts/make_holdout_split.py \
      --out_root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
      --src_prefix seg_list_dynpose --dst_prefix seg_list_dynpose_s91 \
      --test_frac 0.10 --val_frac 0.01
  # 실제로 쓰기 전에 무엇이 나오는지만 보려면 --dry_run
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from os import path
import random

CINE = path.dirname(path.dirname(path.abspath(__file__)))


def read_lines(p):
    return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]


def scene_of(seg):
    """`dynpose/<uuid>/<var>` -> `dynpose/<uuid>`."""
    return seg.rsplit("/", 1)[0]


def stratified_pick(pool, n_cams, target_cams, forced, seed):
    """씬당 카메라 수 버킷마다 같은 비율로 뽑아 `target_cams` 에 맞춘다.

    forced 는 무조건 포함(기존 holdout 목록). 반환은 씬 집합.
    """
    rng = random.Random(seed)
    chosen = set(forced)
    got = sum(n_cams[s] for s in chosen)

    buckets = defaultdict(list)
    for s in pool:
        if s not in chosen:
            buckets[n_cams[s]].append(s)
    for k in buckets:
        buckets[k].sort()            # 결정론적 시작점
        rng.shuffle(buckets[k])

    # 버킷별 카메라 질량에 비례해 할당한다. 같은 비율이면 test 의 씬당 대수 분포가 train 과 같다.
    remain = max(0, target_cams - got)
    mass = {k: k * len(v) for k, v in buckets.items()}
    total_mass = sum(mass.values()) or 1
    quota = {k: int(round(remain * mass[k] / total_mass / k)) for k in buckets}

    for k, v in buckets.items():
        take = min(quota[k], len(v))
        chosen.update(v[:take])
        buckets[k] = v[take:]

    # 반올림 잔차 보정 — 큰 버킷부터 한 씬씩 넣고 빼서 target 에 붙인다.
    order = sorted(buckets, reverse=True)
    got = sum(n_cams[s] for s in chosen)
    while got < target_cams:
        for k in order:
            if buckets[k] and got + k <= target_cams + k // 2:
                s = buckets[k].pop()
                chosen.add(s)
                got += k
                break
        else:
            break
    return chosen


def round_robin(segs):
    """씬별로 뭉친 목록을 씬 라운드로빈으로 편다 (같은 원소, 순서만). export 쪽과 같은 규칙."""
    by_scene = defaultdict(list)
    for s in segs:
        by_scene[scene_of(s)].append(s)
    for k in by_scene:
        by_scene[k].sort()
    out, r = [], 0
    keys = sorted(by_scene)
    while True:
        row = [by_scene[k][r] for k in keys if r < len(by_scene[k])]
        if not row:
            break
        out += row
        r += 1
    assert sorted(out) == sorted(segs)
    return out


def main(args):
    src_tr = path.join(args.out_root, f"{args.src_prefix}_train.txt")
    src_te = path.join(args.out_root, f"{args.src_prefix}_test.txt")
    old_train = read_lines(src_tr)
    segs = old_train + read_lines(src_te)

    n_cams = Counter(scene_of(s) for s in segs)
    scenes = sorted(n_cams)
    total_cams, total_scenes = len(segs), len(scenes)

    legacy = {f"{args.chunk_prefix}/{v}" for v in read_lines(args.forced_test)}
    unknown = legacy - set(n_cams)
    assert not unknown, f"forced_test 에 코퍼스에 없는 씬 {len(unknown)}개: {sorted(unknown)[:3]}"

    # ── test 10% ────────────────────────────────────────────────────────────────
    test_scenes = stratified_pick(scenes, n_cams, round(total_cams * args.test_frac),
                                  legacy, args.seed)
    # ── val 1% — test **안에서** 다시 뽑는다. train:test 가 9:1 로 유지되는 이유가 이것 ──
    val_forced = {f"{args.chunk_prefix}/{v}" for v in read_lines(args.forced_val)} & test_scenes
    val_scenes = stratified_pick(sorted(test_scenes), n_cams,
                                 round(total_cams * args.val_frac), val_forced, args.seed + 1)
    assert val_scenes <= test_scenes

    tr = [s for s in segs if scene_of(s) not in test_scenes]
    te_val = [s for s in segs if scene_of(s) in val_scenes]
    te_rest = [s for s in segs if scene_of(s) in test_scenes and scene_of(s) not in val_scenes]
    te = round_robin(te_val) + round_robin(te_rest)      # val 블록이 반드시 앞
    assert len(tr) + len(te) == total_cams
    assert not (set(tr) & set(te))

    dst_tr = path.join(args.out_root, f"{args.dst_prefix}_train.txt")
    dst_te = path.join(args.out_root, f"{args.dst_prefix}_test.txt")
    out_hold = path.join(CINE, "configs", f"dynpose_holdout_scenes_{len(test_scenes)}.txt")
    out_val = path.join(CINE, "configs", f"dynpose_val_scenes_{len(val_scenes)}.txt")
    if not args.dry_run:
        for p, rows in ((dst_tr, tr), (dst_te, te)):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("\n".join(rows) + "\n")
        for p, ss in ((out_hold, test_scenes), (out_val, val_scenes)):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("\n".join(sorted(s.split("/", 1)[1] for s in ss)) + "\n")

    vb = len(te_val) // args.batch_size
    print(f"\n{'':22s}{'씬':>10}{'대':>12}{'대/씬':>9}{'코퍼스 비중':>13}")
    for name, ss, cc in (("corpus", scenes, segs), ("train", set(scenes) - test_scenes, tr),
                         ("test", test_scenes, te), ("  └ val", val_scenes, te_val)):
        print(f"{name:22s}{len(ss):10,d}{len(cc):12,d}{len(cc) / max(len(ss), 1):9.3f}"
              f"{len(cc) / total_cams * 100:12.2f}%")
    print(f"\n{'train:test':22s}{len(tr) / max(len(te), 1):9.3f} : 1")
    print(f"{'씬당 대수 (train)':22s}{sorted(Counter(n_cams[s] for s in set(scenes) - test_scenes).items())}")
    print(f"{'씬당 대수 (test)':22s}{sorted(Counter(n_cams[s] for s in test_scenes).items())}")
    print(f"{'legacy 포함':22s}forced_test {len(legacy & test_scenes)}/{len(legacy)}   "
          f"forced_val {len(val_forced)}/{len(val_forced)}")
    print(f"\n{'val_max_batches':22s}{vb}  (batch_size {args.batch_size} -> "
          f"{vb * args.batch_size:,d}대 = val {len(te_val):,d}대 중 "
          f"{vb * args.batch_size / max(len(te_val), 1) * 100:.1f}%)")
    print(f"{'step/epoch (train)':22s}{len(tr) // args.batch_size:,d}"
          f"   (이전 {len(old_train) // args.batch_size:,d})")
    print(f"\n{'train_seg_list':22s}{dst_tr}")
    print(f"{'test_seg_list':22s}{dst_te}")
    print(f"{'holdout 씬 목록':22s}{out_hold}")
    print(f"{'val 씬 목록':22s}{out_val}")
    if args.dry_run:
        print("\n(dry_run — 아무것도 안 썼다)")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--out_root", default="/data1/cympyc1785/data/DynPose-LBM/"
                                         "latentcam_dynpose_d200", type=str)
    p.add_argument("--src_prefix", default="seg_list_dynpose", type=str)   # 읽을 seg_list 접두사
    p.add_argument("--dst_prefix", default="seg_list_dynpose_s91", type=str)  # 쓸 접두사
    p.add_argument("--chunk_prefix", default="dynpose", type=str)          # seg_list 줄의 앞머리
    p.add_argument("--test_frac", default=0.10, type=float)   # 코퍼스 대비 test 카메라 비율
    p.add_argument("--val_frac", default=0.01, type=float)    # 코퍼스 대비 val 카메라 비율(test 안)
    p.add_argument("--batch_size", default=8, type=int)       # val_max_batches 환산용
    # 새 test 가 반드시 포함할 씬 (지표 대조 유지). legacy 100씬 ⊃ legacy 27씬.
    p.add_argument("--forced_test", default=path.join(CINE, "configs",
                                                      "dynpose_holdout_scenes_100.txt"), type=str)
    p.add_argument("--forced_val", default=path.join(CINE, "configs",
                                                     "dynpose_holdout_scenes.txt"), type=str)
    p.add_argument("--seed", default=200, type=int)
    p.add_argument("--dry_run", action="store_true", default=False)
    p.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    main(p.parse_args())
