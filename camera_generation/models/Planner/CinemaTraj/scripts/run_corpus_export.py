"""뱅크 → latentcam 코퍼스 한 방향 드라이버. `--stage` 로 단계를 고른다.

WHY: 세대마다 `run_dynpose_dNNN_export.sh` 를 복붙해 왔고 (d129/d147/d163), 서로 다른 곳은
     뱅크 이름·필터·out_root 몇 줄뿐이었다. 그 차이를 전부 argparse 기본값으로 옮기고
     스크립트는 하나만 둔다. 세대가 바뀌면 **인자만** 바뀐다.

단계는 앞에서 뒤로 의존한다. 중간부터 다시 돌려도 안전하다 (전부 `--skip_done` 계열).

    desc        지칭구(`instance_desc.json`) 가 없는 씬을 VLM 으로 채운다 → `instance_desc_<tag>.json`
                ※ vLLM 이 `--api_base` 에 떠 있어야 한다. 안 떠 있으면 여기서 멈춘다.
    merge_desc  위 결과를 `instance_desc.json` 에 **합친다** (덮어쓰지 않는다 — 이전 세대의
                anchor 가 아닌 노드 지칭구가 거기 들어 있다)
    captions    뱅크마다 `build_bank_captions.py` 를 돌려 **같은 `--out_name`** 으로 캡션을 굽는다
    export      `vista4d_bank_to_dl3dv.py` 로 세 뱅크를 한 통에 붓고 씬당 상한을 씌워 코퍼스로
    verify      seg_list 행수 / 씬 수 / preset 분포 / 뱅크 출처 분포 검산

주의 (FIX-D129-a): 캡션에 `--videos all` 을 주면 `metadata.csv` 기본값이 Vista4D 것이라
조용히 0편이 된다. 여기서는 항상 `--videos_file` 로 명시한다.

    python scripts/run_corpus_export.py --stage captions           # D200 기본값
    python scripts/run_corpus_export.py --stage export
    python scripts/run_corpus_export.py --stage all --dry_run
"""
import json
from argparse import ArgumentParser
from os import listdir, makedirs, path
from subprocess import run

CINE = path.dirname(path.dirname(path.abspath(__file__)))
LATENTCAM = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
PY = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"
PY_LC = "/data1/cympyc1785/miniconda3/envs/latentcam/bin/python"


def banks_of(out_root_dir, video, banks):
    """이 씬에서 **완비된** 뱅크만 (bank.json + poses.npz 둘 다 있는 것).

    d185 는 10,343 편에 폴더가 있지만 그중 646 편은 변이가 0개라 `poses.npz` 가 없다.
    폴더 존재만 보면 그 646 편이 대상에 끼어 캡션 단계가 통째로 경고를 뱉는다.
    """
    return [b for b in banks
            if path.isfile(path.join(out_root_dir, video, b, "bank.json"))
            and path.isfile(path.join(out_root_dir, video, b, "poses.npz"))]


def scan(args):
    """대상 씬 목록 + 씬→뱅크 조합을 만든다. 모든 단계가 이걸 공유한다."""
    banks = args.banks
    root = args.cine_out
    scenes, avail = [], {}
    for video in sorted(listdir(root)):
        got = banks_of(root, video, banks)
        if got:
            scenes.append(video)
            avail[video] = got
    return scenes, avail


def stage_desc(args, scenes, avail):
    """지칭구 결손 씬만 VLM 으로. 뱅크별로 묶어서 부른다 (`--bank_dir` 가 하나뿐이라)."""
    todo = {}
    for video in scenes:
        if path.isfile(path.join(args.cine_out, video, args.desc_name)):
            continue                       # 이미 이번 세대 결과가 있다 (재실행 안전)
        if path.isfile(path.join(args.cine_out, video, "instance_desc.json")):
            continue
        todo.setdefault(avail[video][0], []).append(video)
    total = sum(len(v) for v in todo.values())
    per_bank = {k.replace("hole_bank_", ""): len(v) for k, v in todo.items()}
    print(f"{'지칭구 결손':22s} {total:,d} 편   뱅크별 {per_bank}", flush=True)
    if args.dry_run or not total:
        return
    for bank, videos in todo.items():
        videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard_id]
        for start in range(0, len(videos), args.batch):
            chunk = videos[start:start + args.batch]
            cmd = [PY, path.join(CINE, "scripts", "describe_instances_vlm.py"),
                   "--output_root", args.cine_out, "--eval_data", args.eval_data,
                   "--bank_dir", bank, "--out_name", args.desc_name,
                   "--api_base", args.api_base, "--videos", *chunk]
            res = run(cmd, cwd=CINE, capture_output=True, text=True)
            done = sum(1 for v in chunk
                       if path.isfile(path.join(args.cine_out, v, args.desc_name)))
            print(f"  [{bank}] {start + len(chunk):5d}/{len(videos)}  rc={res.returncode}  "
                  f"wrote {done}/{len(chunk)}", flush=True)
            if res.returncode != 0:
                print(res.stderr[-1200:], flush=True)


def stage_merge_desc(args, scenes, avail):
    """`instance_desc_<tag>.json` → `instance_desc.json` 에 합친다.

    `describe_instances_vlm.py` 는 `--bank_dir` 의 anchor 만 기술하고 파일을 통째로 덮어쓴다.
    대상 씬 상당수가 이전 세대의 `instance_desc.json` 을 이미 갖고 있고 거기엔 이번 anchor 가
    아닌 노드의 지칭구도 들어 있다 — 그대로 덮으면 그게 사라진다.
    """
    merged = kept = added = skipped = 0
    for video in scenes:
        new_path = path.join(args.cine_out, video, args.desc_name)
        old_path = path.join(args.cine_out, video, "instance_desc.json")
        if not path.isfile(new_path):
            skipped += 1
            continue
        new = json.load(open(new_path, encoding="utf-8"))
        old = (json.load(open(old_path, encoding="utf-8")) if path.isfile(old_path)
               else {"format": "lbm_instance_desc_v1", "video": video,
                     "bank_dir": None, "model": new.get("model"),
                     "num_frames": new.get("num_frames"), "descriptions": {}})
        before = len(old.get("descriptions", {}))
        old.setdefault("descriptions", {}).update(new.get("descriptions", {}))
        old["bank_dir"] = avail[video][0]
        if not args.dry_run:
            with open(old_path, "w", encoding="utf-8") as fh:
                json.dump(old, fh, ensure_ascii=False, indent=1)
        merged += 1
        kept += before
        added += len(old["descriptions"]) - before
    print(f"{'merged':22s} {merged:,d}   (새 파일 없어 건너뜀 {skipped:,d})")
    print(f"{'기존 지칭구 유지':22s} {kept:,d}")
    print(f"{'새로 추가':22s} {added:,d}")


def stage_captions(args, scenes, avail):
    """뱅크마다 한 번씩. 뱅크별로 그 뱅크를 가진 씬 목록만 준다."""
    makedirs(args.tmp, exist_ok=True)
    for bank in args.banks:
        videos = [v for v in scenes if bank in avail[v]]
        done = sum(1 for v in videos
                   if path.isfile(path.join(args.cine_out, v, bank, args.captions_name)))
        vfile = path.join(args.tmp, f"scenes_{bank}.txt")
        with open(vfile, "w", encoding="utf-8") as fh:
            fh.write("\n".join(videos) + "\n")
        print(f"{bank:22s} 대상 {len(videos):,d}   이미 {args.captions_name} 있음 {done:,d}",
              flush=True)
        if args.dry_run:
            continue
        log = path.join(args.tmp, f"captions_{bank}.log")
        cmd = [PY, path.join(CINE, "scripts", "build_bank_captions.py"),
               "--videos_file", vfile, "--output_root", args.cine_out,
               "--bank_dir", bank, "--out_name", args.captions_name,
               "--external_shapes", args.shapes, "--prompt_style", args.prompt_style,
               "--metadata_csv", args.metadata_csv]
        if not args.magnitude:
            cmd.append("--no_magnitude")
        with open(log, "w", encoding="utf-8") as fh:
            res = run(cmd, cwd=CINE, stdout=fh, stderr=fh, text=True)
        got = sum(1 for v in videos
                  if path.isfile(path.join(args.cine_out, v, bank, args.captions_name)))
        print(f"  rc={res.returncode}   캡션 {got:,d}/{len(videos):,d}   log {log}", flush=True)
        if got < len(videos):
            print(f"  !! 누락 {len(videos) - got:,d} 편 (D107 과 같은 증상)", flush=True)


def stage_export(args, scenes, avail):
    makedirs(args.tmp, exist_ok=True)
    vfile = path.join(args.tmp, "scenes_export.txt")
    with open(vfile, "w", encoding="utf-8") as fh:
        fh.write("\n".join(scenes) + "\n")
    test = [ln.strip() for ln in open(args.test_videos, encoding="utf-8") if ln.strip()]
    cmd = [PY, path.join(CINE, "scripts", "vista4d_bank_to_dl3dv.py"),
           "--cine_out", args.cine_out, "--bank_dirs", *args.banks,
           "--per_scene_cap", str(args.per_scene_cap),
           "--captions_name", args.captions_name,
           "--recon_root", args.recon_root, "--out_root", args.out_root,
           "--videos", *scenes, "--meta_csv", args.meta_csv,
           "--chunk_prefix", args.chunk_prefix, "--seg_list_prefix", args.seg_list_prefix,
           "--test_videos", *test, "--workers", str(args.workers)]
    if args.drop_status:
        cmd += ["--drop_status", *args.drop_status]
    if args.drop_suspect:
        cmd += ["--drop_suspect", *args.drop_suspect]
    print(f"{'export -> ':22s} {args.out_root}")
    print(f"{'씬':22s} {len(scenes):,d}   상한 {args.per_scene_cap}   "
          f"drop_status {args.drop_status}   drop_suspect {args.drop_suspect or '없음'}")
    if args.dry_run:
        print(" ".join(cmd[:14]) + " ... (생략)")
        return
    log = path.join(args.tmp, "export.log")
    with open(log, "w", encoding="utf-8") as fh:
        res = run(cmd, cwd=CINE, stdout=fh, stderr=fh, text=True)
    tail = open(log, encoding="utf-8").read().splitlines()[-16:]
    print("\n".join(tail))
    print(f"rc={res.returncode}   log {log}")


def stage_verify(args, scenes, avail):
    """코퍼스에서 직접 센다 — 뱅크 재고가 아니라 **나온 것**을 센다."""
    from collections import Counter
    rows = {}
    for split in ("train", "test"):
        f = path.join(args.out_root, f"{args.seg_list_prefix}_{split}.txt")
        if not path.isfile(f):
            print(f"{split:8s} 없음 {f}")
            continue
        lines = [ln.strip() for ln in open(f, encoding="utf-8") if ln.strip()]
        rows[split] = lines
    cache, preset, source, per_scene = {}, Counter(), Counter(), Counter()
    for split, lines in rows.items():
        for line in lines:
            _, scene, key = line.split("/", 2)
            if scene not in cache:
                cache[scene] = json.load(open(path.join(
                    args.out_root, args.chunk_prefix, scene, "da3", "prompts.json"),
                    encoding="utf-8"))
            e = cache[scene][key]
            preset[e.get("preset", "?")] += 1
            source[e.get("source", "?").split("/")[-1]] += 1
            per_scene[scene] += 1
    total = sum(len(v) for v in rows.values())
    print(f"{'카메라(행)':22s} {total:,d}")
    for split, lines in rows.items():
        print(f"  {split:8s}{len(lines):>9,d} 행   씬 {len({l.split('/')[1] for l in lines}):,d}")
    print(f"{'씬':22s} {len(per_scene):,d}   씬당 {total / max(1, len(per_scene)):.2f}")
    print(f"\n{'뱅크 출처':22s}")
    for k, n in source.most_common():
        print(f"  {k:28s}{n:8,d}{n / max(1, total) * 100:7.1f}%")
    print(f"\n{'preset (상위 20)':22s}")
    for k, n in preset.most_common(20):
        print(f"  {k:34s}{n:8,d}{n / max(1, total) * 100:7.1f}%")
    print(f"\n{'씬당 카메라 수 분포':22s}")
    hist = Counter(per_scene.values())
    for k in sorted(hist):
        print(f"  {k:3d}대  {hist[k]:6,d}씬")


STAGES = {"desc": stage_desc, "merge_desc": stage_merge_desc, "captions": stage_captions,
          "export": stage_export, "verify": stage_verify}


def main(args):
    scenes, avail = scan(args)
    from collections import Counter
    combo = Counter("+".join(b.replace("hole_bank_", "") for b in avail[v]) for v in scenes)
    print(f"{'대상 씬':22s} {len(scenes):,d}   뱅크 {args.banks}")
    for k, n in combo.most_common():
        print(f"  {k:28s}{n:8,d}")
    print()
    todo = list(STAGES) if args.stage == "all" else [args.stage]
    for name in todo:
        print(f"{'=' * 70}\n== {name}\n{'=' * 70}", flush=True)
        STAGES[name](args, scenes, avail)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="verify",
                   choices=[*STAGES, "all"], type=str)
    # ── 뱅크 / 코퍼스 (세대 차이는 전부 여기서) ────────────────────────────────
    # d198 은 뺀다 — config 가 `fit.args` 를 통째로 덮어써서 인자 25개(충돌 판정 /
    # behind-surface / tracking / orbit sweep / composition / tau_denom)가 빠졌다. 같은
    # 게이트를 통과한 카메라가 아니라 섞으면 pseudo-GT 의 뜻이 달라진다 (FIX.log 2026-09-17,
    # 사용자 지시 "d185 기준이어야해"). d199 는 route 에 `--anchor_require_frame0` 하나만
    # 붙고 tau/fit/emit 은 d185 를 글자 그대로 상속하므로 같은 세대로 취급한다.
    p.add_argument("--banks", nargs="+",
                   default=["hole_bank_d185", "hole_bank_d199"])
    p.add_argument("--per_scene_cap", default=6, type=int)     # 씬당 카메라 상한. 0 이면 무제한
    p.add_argument("--captions_name", default="captions_d200.json", type=str)
    p.add_argument("--desc_name", default="instance_desc_d200.json", type=str)
    p.add_argument("--out_root", default="/data1/cympyc1785/data/DynPose-LBM/"
                                         "latentcam_dynpose_d200", type=str)
    p.add_argument("--tmp", default="/data1/cympyc1785/LatentCamVid/tmp/d200", type=str)
    # ── 필터. 사용자 지시(2026-09-16): **clamped_low 만** 뺀다. free-moving 도 남긴다 ──
    p.add_argument("--drop_status", nargs="*", default=["clamped_low"])
    p.add_argument("--drop_suspect", nargs="*", default=[])
    # ── 경로 ──────────────────────────────────────────────────────────────────
    p.add_argument("--cine_out", default=path.join(CINE, "out_dynpose"), type=str)
    p.add_argument("--recon_root", default="/data1/cympyc1785/LatentCamVid/DATA/"
                                           "DynPose-100K/eval_data/recon_and_seg", type=str)
    p.add_argument("--eval_data", default="/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K",
                   type=str)
    p.add_argument("--metadata_csv", default="/data1/cympyc1785/LatentCamVid/DATA/"
                                             "DynPose-100K/metadata.csv", type=str)
    p.add_argument("--shapes", default=path.join(CINE, "configs", "datadop_shapes.json"),
                   type=str)
    # D200 에서 27씬 -> 100씬 (legacy 27 의 상위집합). 27씬은 코퍼스가 10,169씬이 된 지금
    # 0.26% 라 씬 수준 지표의 n 이 27 밖에 안 됐다. 새 목록은 씬당 카메라 수 분포로 층화
    # 추출했고 legacy 27 을 전부 포함하므로, 지표를 `test27`(D137/D194/D197 과 대조 가능) 과
    # `test_full` 두 줄로 낼 수 있다. 예전 27씬으로 돌리려면
    # `--test_videos <CINE>/configs/dynpose_holdout_scenes.txt` (파일은 그대로 남겨 뒀다).
    p.add_argument("--test_videos", default=path.join(CINE, "configs",
                                                      "dynpose_holdout_scenes_100.txt"), type=str)
    p.add_argument("--meta_csv", default="meta_dynpose.csv", type=str)
    p.add_argument("--chunk_prefix", default="dynpose", type=str)
    p.add_argument("--seg_list_prefix", default="seg_list_dynpose", type=str)
    # ── 캡션 ──────────────────────────────────────────────────────────────────
    p.add_argument("--prompt_style", default="nl", choices=["nl", "fields"], type=str)
    # 크기 부사. D176+ 는 끈다 — 실현치를 읽는데 세대마다 τ 분모가 달라 비교가 안 된다.
    p.add_argument("--magnitude", action="store_true", default=False)
    p.add_argument("--no_magnitude", dest="magnitude", action="store_false")
    # ── VLM (desc 단계) ───────────────────────────────────────────────────────
    p.add_argument("--api_base", default="http://127.0.0.1:22002/v1", type=str)
    p.add_argument("--batch", default=20, type=int)            # subprocess 당 영상 수
    p.add_argument("--num_shards", default=1, type=int)
    p.add_argument("--shard_id", default=0, type=int)
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--dry_run", action="store_true", default=False)
    p.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    main(p.parse_args())
