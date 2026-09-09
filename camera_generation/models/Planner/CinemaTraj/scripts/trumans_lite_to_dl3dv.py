"""TRUMANS-Lite 뱅크(`recon_and_seg/<video>/`) → latentcam 이 읽는 **DL3DV da3 온디스크 포맷**.

왜 새 dataset 클래스를 안 만들고 포맷을 맞추는가:
  `main/dataset_mixed.py` 의 `_VALID_NAMES` 는 ('dl3dv','scene_decoupled','datadop') 셋뿐이고
  TRUMANS 는 거기 없다. 그런데 `OVERRIDE_WHITELIST` 에 `dl3dv_root`/`meta_csv` 가 들어 있고
  코퍼스 이름의 **유일성은 검사하지 않는다** → `name: dl3dv` 블록을 두 개 두고 한쪽 root 만
  TRUMANS 로 돌리면 혼합 학습이 그대로 된다. 즉 새로 짤 것은 dataset 코드가 아니라 **디렉토리**다.

═══ layout='per_recording' (기본) — scene = recording, segment = clip ═══════════════════
클립 하나를 scene 하나로 내보내면 **학습이 성립하지 않는다.** 클립은 49프레임이고 target
segment 도 49프레임이라 context range 가 target 과 **완전히 겹친다** — `geo_posed: true` 라
DA3 cam_enc 가 target 궤적의 포즈를 6장 그대로 받는다. 예측할 게 남지 않는다.

그래서 **같은 recording 의 클립들을 한 scene 으로 이어 붙인다**:
  images_4/{00000..}.png   = clip0 49장 + clip1 49장 + ...  (클립 이름 정렬 순)
  da3/pose.npz             = 같은 순서의 w2c, **TRUMANS world 좌표**
  da3/prompts.json         = {"0": {"frame_idx": [0,49]}, "1": {"frame_idx": [49,98]}, ...}
그러면 `geo_view_sampling='context_uniform'` 이 scene 전체 `[0,N)` 에서 6장을 뽑으므로 context
대부분이 **다른 클립 = 다른 카메라, 다른 시각**이 된다 (SD 의 cross-clip pairing 과 같은 구조인데,
클립들이 같은 world 라 sim3 정렬 잔차가 아예 없다).

  ⚠ 포즈는 `recon_and_seg/*/cameras.npz` 가 아니라 **work 의 `poses_aNN.npz`** 에서 읽는다.
    전자는 클립마다 frame0 을 원점으로 **재고정**돼 있어(`cam_c2w[0]==I` 실측) 클립끼리 이어
    붙이면 전부 원점에 겹친다. 후자가 TRUMANS world 원본이다. target 은 어차피
    `rel = E @ inv(E_s)` 로 다시 s 기준이 되므로 이 교체가 target 쪽을 바꾸지는 않는다.

  ⚠ 남은 누수: context 6장 중 일부가 target 클립 안에 떨어질 수 있다 (DL3DV context_uniform 과
    같은 성질). scene 당 클립이 14~22개라 기대값은 1장 미만이지만 0 은 아니다.

═══ avg_scale — 왜 클립 자기 값을 기본으로 안 쓰는가 ════════════════════════════════════
`recon_and_seg/<clip>/avg_scale.json` 은 **그 클립의 렌더 depth** 로 잰 값이다. 그걸 그 클립을
target 으로 하는 sample 의 분모로 쓰면 추론 때 없는 정보다 (target 영상이 아직 없다). 게다가
같은 recording 안에서도 클립 간 1.54~2.31배 갈린다 (실측) — 무시할 크기가 아니다.
  → 기본 `scene_median`: recording 내 클립들의 median 을 그 scene 의 **모든** segment 에 쓴다.
    context 가 scene 전체이므로 "context 에서 잰 값" 이라는 DL3DV `context_first_cam` 의 정의와
    맞고, 추론 때 소스 영상만으로 복원 가능하다.
  → 클립 자기 값도 **같이** 쓴다: `avg_scale/`(= AVG_SCALE_DIRS['centroid']) 디렉토리에.
    ablation 용이고 **누수가 있는 쪽**이다. `avg_scale_ref: centroid` 로 골라야만 쓰인다.

`dataset_dl3dv.py` 가 실제로 요구하는 것 (다른 건 안 만든다):
  <scene>/da3/pose.npz        extrinsics (N,3,4) OpenCV w2c, intrinsics (N,3,3)  (`_parse_da3`:363)
  <scene>/da3/prompts.json    seg → frame_idx [s,e] (**e exclusive**), prompt_camera_with_scene_video
  <scene>/da3/<ref dir>/<seg>.json   bare float
  <scene>/images_4/*.png      정렬 순서 = pose 순서, 개수가 N 과 정확히 같아야 한다
  <root>/meta_trumans.csv     `chunk` 열만 읽힌다 (`_read_meta_scenes`:1350)

캡션은 VLM 으로 안 뽑는다 — 카메라는 우리가 합성한 preset(GT)이고 행동은 라벨이라 추정할 게 없다.
DL3DV 쪽 캡션 형식(`prompt_camera_with_scene_video.concise`)에 맞춰 조립한다.

사용 예시:
  python scripts/trumans_lite_to_dl3dv.py --dry_run
  python scripts/trumans_lite_to_dl3dv.py --workers 8
  python scripts/trumans_lite_to_dl3dv.py --layout per_clip     # 옛 평면 배치 (누수 있음)
"""
import csv
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import listdir, makedirs, path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
_LATENTCAM_MAIN = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/main"
if _LATENTCAM_MAIN not in sys.path:
    sys.path.insert(0, _LATENTCAM_MAIN)      # AVG_SCALE_DIRS 를 두 벌 안 두려고 직접 읽는다
from dataset_cfg import AVG_SCALE_DIRS, AVG_SCALE_REFS      # noqa: E402


# preset → 카메라 문구. `trumans_to_recon.py:85 SOURCE_PRESETS` 의 (sweep_deg, dradius, dheight)
# 를 사람 말로 옮긴 것이다. sweep 음수 = 좌호, dradius 음수 = 접근.
PRESET_PHRASE = {
    "arc_left":  "The camera arcs to the left around the subject",
    "arc_right": "The camera arcs to the right around the subject",
    "push_in":   "The camera pushes in toward the subject",
    "pull_out":  "The camera pulls back away from the subject",
    "arc_push":  "The camera arcs to the left while pushing in",
    "arc_pull":  "The camera arcs to the right while pulling back",
    "drift":     "The camera drifts to the left and rises slightly while easing in",
    "hold":      "The camera holds almost still on the subject",
}


def caption_of(manifest):
    """`prompt_camera_with_scene_video.concise` 로 쓸 한 줄. 카메라는 preset(GT), 내용은 action 라벨.

    action 라벨이 명령형 원형("Pick up the book with both hands")이라 "as they <소문자>" 가
    문법적으로 맞는다."""
    phrase = PRESET_PHRASE[manifest["source_camera"]["preset"]]   # 없는 preset 이면 죽는 게 맞다
    action = str(manifest["action"]["text"]).strip().rstrip(".")
    return f"{phrase}, keeping a person in frame as they {action[0].lower() + action[1:]}."


def read_manifests(work):
    """`out/trumans_recon/<recording>/manifest_aNN.json` 전량 → {video: manifest}."""
    index = {}
    for recording in sorted(listdir(work)):
        rec_dir = path.join(work, recording)
        if not path.isdir(rec_dir):
            continue
        for name in sorted(listdir(rec_dir)):
            if not (name.startswith("manifest_a") and name.endswith(".json")):
                continue
            try:
                data = json.load(open(path.join(rec_dir, name), encoding="utf-8"))
            except Exception:
                continue
            if data.get("video"):
                index[data["video"]] = data
    return index


def clip_world_poses(manifest):
    """work 의 `poses_aNN.npz` → TRUMANS world c2w (49,4,4). recon 의 재고정본을 쓰면 안 된다."""
    return np.asarray(np.load(manifest["paths"]["poses"])["cam_c2w"], dtype=np.float64)


def convert_scene(job):
    """recording 하나 → <out>/<recording>/{da3/*, images_4/*}. 실패는 예외 대신 dict."""
    import imageio.v3 as iio                    # worker 안에서 import

    (chunk, clips, recon_root, out_root, image_dir, refs, scene_scale,
     clip_scale_dir, skip_done) = job
    recording = chunk                                 # 아래 반환 dict 의 라벨 (= meta 의 chunk)
    dst = path.join(out_root, *chunk.split("/"))
    da3, img = path.join(dst, "da3"), path.join(dst, image_dir)
    total = sum(c["n"] for c in clips)
    try:
        done = (path.isfile(path.join(da3, "pose.npz"))
                and path.isdir(img) and len(listdir(img)) == total)
        if skip_done and done:
            # h/w 는 meta_csv 열이라 skip 경로에서도 채워야 한다 (없으면 빈 칸이 찍힌다).
            probe = iio.imread(path.join(img, sorted(listdir(img))[0]))
            return {"chunk": recording, "status": "skip_done", "n": total,
                    "h": int(probe.shape[0]), "w": int(probe.shape[1]),
                    "n_clips": len(clips)}
        makedirs(da3, exist_ok=True)
        makedirs(img, exist_ok=True)

        w2c_all, K_all, prompts, offset = [], [], {}, 0
        h = w = 0
        for seg, clip in enumerate(clips):
            c2w = clip["c2w"]                                    # (n,4,4) TRUMANS world
            fxfycxcy = clip["intr"]                              # (n,4)
            n = c2w.shape[0]
            w2c_all.append(np.linalg.inv(c2w)[:, :3, :4])
            K = np.zeros((n, 3, 3), dtype=np.float64)
            K[:, 0, 0], K[:, 1, 1] = fxfycxcy[:, 0], fxfycxcy[:, 1]
            K[:, 0, 2], K[:, 1, 2] = fxfycxcy[:, 2], fxfycxcy[:, 3]
            K[:, 2, 2] = 1.0
            K_all.append(K)

            # plugin="FFMPEG": vista4d env 에 pyav 가 없다 (imageio-ffmpeg 만 있다).
            frames = iio.imread(path.join(recon_root, clip["video"], "video.mp4"),
                                plugin="FFMPEG")
            if len(frames) != n:
                return {"chunk": recording, "status": "fail",
                        "why": f"{clip['video']}: video {len(frames)} != pose {n}"}
            h, w = int(frames.shape[1]), int(frames.shape[2])
            for i, fr in enumerate(frames):
                iio.imwrite(path.join(img, f"{offset + i:05d}.png"), fr)

            prompts[str(seg)] = {
                "frame_idx": [offset, offset + n],               # e 는 exclusive
                "prompt_camera_with_scene_video": {"concise": clip["caption"]},
                "clip": clip["video"], "preset": clip["preset"], "action": clip["action"],
                "source": "trumans_lite_preset+action_label",
            }
            offset += n

        np.savez(path.join(da3, "pose.npz"),
                 extrinsics=np.concatenate(w2c_all).astype(np.float32),
                 intrinsics=np.concatenate(K_all).astype(np.float32))
        json.dump(prompts, open(path.join(da3, "prompts.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

        for ref in refs:                                          # 누수 없는 scene 상수
            ref_dir = path.join(da3, AVG_SCALE_DIRS[ref])
            makedirs(ref_dir, exist_ok=True)
            for seg in range(len(clips)):
                json.dump(scene_scale, open(path.join(ref_dir, f"{seg}.json"), "w"))
        if clip_scale_dir:                                        # 누수 있는 ablation 용
            ref_dir = path.join(da3, AVG_SCALE_DIRS[clip_scale_dir])
            makedirs(ref_dir, exist_ok=True)
            for seg, clip in enumerate(clips):
                json.dump(clip["avg_scale"], open(path.join(ref_dir, f"{seg}.json"), "w"))

        return {"chunk": recording, "status": "ok", "n": offset, "h": h, "w": w,
                "n_clips": len(clips)}
    except Exception as error:                                    # noqa: BLE001
        return {"chunk": recording, "status": "fail", "why": f"{type(error).__name__}: {error}"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--recon_root",
                        default="/data1/cympyc1785/data/TRUMANS-Lite/eval_data/recon_and_seg")
    parser.add_argument("--work",                                 # manifest + world pose 출처
                        default=path.join(path.dirname(HERE), "out", "trumans_recon"))
    parser.add_argument("--out_root",
                        default="/data1/cympyc1785/data/TRUMANS-Lite/latentcam_da3")
    parser.add_argument("--layout", default="per_recording", choices=("per_recording", "per_clip"))
    parser.add_argument("--meta_csv", default="meta_trumans.csv")
    # chunk 를 **두 단계**(`<prefix>/<recording>`)로 두는 이유: base.py 의 seg-list 분할이
    # `CamDataset.seg_key` (dataset_dl3dv.py:1250) 를 쓰는데 그게 data_name 을
    # `<batch>_<hash>_<seg>` 로 가정하고 `bh.split('_', 1)` 한다. recording 이름(UUID)에는
    # 밑줄이 없어서 한 단계 chunk 면 거기서 ValueError 로 죽는다 -> scene 단위 holdout 불가.
    parser.add_argument("--chunk_prefix", default="trumans")      # "" 면 한 단계 (seg-list 불가)
    # scene 단위 holdout. recording 이름의 **접두사**로 매칭한다 (앞 8자만 적어도 된다).
    # 비우면 리스트를 안 쓴다 -> base.py 가 train_frac 랜덤 분할로 떨어지고 train/val 이
    # 같은 recording 을 공유한다 (인접 클립 = 같은 환경/배우라 val 수치가 낙관적으로 뜬다).
    parser.add_argument("--test_recordings", nargs="*", default=["2b4c9b84"])
    parser.add_argument("--seg_list_prefix", default="seg_list_trumans")
    parser.add_argument("--image_dir", default="images_4")        # dl3dv IMAGE_DIR_NAMES 첫 후보
    parser.add_argument("--avg_scale_refs", nargs="+", default=["context_first_cam"],
                        choices=sorted(AVG_SCALE_REFS))           # scene 상수를 넣을 디렉토리
    parser.add_argument("--clip_avg_scale_ref", default="centroid",
                        choices=("", *sorted(AVG_SCALE_REFS)))    # 클립 자기 값(누수) 넣을 곳. ""=끔
    parser.add_argument("--avg_scale_key", default="avg_scale",
                        choices=("avg_scale", "avg_scale_static"))
    parser.add_argument("--workers", default=7, type=int)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.layout == "per_clip":
        raise SystemExit("per_clip 은 context==target 이라 학습이 성립하지 않는다 "
                         "(모듈 docstring 참고). 정말 필요하면 이 가드를 지울 것.")

    manifests = read_manifests(args.work)
    by_rec, missing = defaultdict(list), []
    for video in sorted(d for d in listdir(args.recon_root)
                        if path.isdir(path.join(args.recon_root, d))):
        src = path.join(args.recon_root, video)
        manifest = manifests.get(video)
        if manifest is None:
            missing.append((video, "manifest 없음 -> 캡션도 world pose 도 못 만든다"))
            continue
        if not (path.isfile(path.join(src, "video.mp4"))
                and path.isfile(path.join(src, "avg_scale.json"))
                and path.isfile(manifest["paths"]["poses"])):
            missing.append((video, "recon/pose 산출물 불완전"))
            continue
        c2w = clip_world_poses(manifest)
        by_rec[manifest["recording"]].append({
            "video": video, "c2w": c2w, "n": int(c2w.shape[0]),
            "intr": np.asarray(np.load(path.join(src, "cameras.npz"))["intrinsics"],
                               dtype=np.float64),
            "avg_scale": float(json.load(open(path.join(src, "avg_scale.json")))[args.avg_scale_key]),
            "caption": caption_of(manifest), "preset": manifest["source_camera"]["preset"],
            "action": manifest["action"]["text"],
        })

    jobs = []
    for recording, clips in sorted(by_rec.items()):
        clips.sort(key=lambda c: c["video"])
        scene_scale = float(np.median([c["avg_scale"] for c in clips]))
        chunk = f"{args.chunk_prefix}/{recording}" if args.chunk_prefix else recording
        jobs.append((chunk, clips, args.recon_root, args.out_root, args.image_dir,
                     args.avg_scale_refs, scene_scale, args.clip_avg_scale_ref or None,
                     args.skip_done))

    print(f"{'layout':22s} {args.layout}")
    print(f"{'scenes (recording)':22s} {len(jobs)}")
    print(f"{'clips (segments)':22s} {sum(len(j[1]) for j in jobs)}")
    print(f"{'skipped clips':22s} {len(missing)}")
    for video, why in missing:
        print(f"  - {video}: {why}")
    print(f"\n{'recording':10s} {'clips':>5s} {'frames':>7s} {'avg_scale(med)':>15s}")
    for chunk, clips, *_rest in jobs:
        print(f"{chunk.split('/')[-1][:8]:10s} {len(clips):5d} {sum(c['n'] for c in clips):7d} "
              f"{_rest[4]:15.3f}")
    if jobs:
        print("\ncaption sample")
        for clip in jobs[0][1][:2] + jobs[-1][1][-1:]:
            print(f"  {clip['video']}  {clip['caption']}")
    if args.dry_run:
        return

    makedirs(args.out_root, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(convert_scene, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"  done {len(results)}/{len(futures)}  {results[-1]['chunk'][:8]} "
                  f"{results[-1]['status']}", flush=True)

    ok = [r for r in results if r["status"] in ("ok", "skip_done")]
    fail = [r for r in results if r["status"] == "fail"]
    sized = {r["chunk"]: r for r in results if r["status"] in ("ok", "skip_done")}
    with open(path.join(args.out_root, args.meta_csv), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["chunk", "height", "width", "num_images"])
        for r in sorted(ok, key=lambda x: x["chunk"]):
            s = sized.get(r["chunk"], {})
            writer.writerow([r["chunk"], s.get("h", ""), s.get("w", ""), r.get("n", "")])

    # scene 단위 seg list. base.py 의 명시적 분할이 읽는 형식은 `<batch>/<hash>/<seg>` 다.
    seg_paths = {}
    if args.test_recordings and args.chunk_prefix:
        okc = {r["chunk"] for r in ok}
        lists = {"train": [], "test": []}
        for chunk, clips, *_rest in jobs:
            if chunk not in okc:
                continue
            rec = chunk.split("/")[-1]
            side = "test" if any(rec.startswith(p) for p in args.test_recordings) else "train"
            lists[side] += [f"{chunk}/{seg}" for seg in range(len(clips))]
        for side, ids in lists.items():
            p = path.join(args.out_root, f"{args.seg_list_prefix}_{side}.txt")
            with open(p, "w") as file:
                file.write("\n".join(ids) + "\n")
            seg_paths[side] = (p, len(ids))

    print(f"\n{'ok':22s} {len([r for r in results if r['status'] == 'ok'])}")
    print(f"{'skip_done':22s} {len([r for r in results if r['status'] == 'skip_done'])}")
    print(f"{'fail':22s} {len(fail)}")
    for r in fail:
        print(f"  - {r['chunk']}: {r['why']}")
    print(f"{'meta_csv':22s} {path.join(args.out_root, args.meta_csv)} ({len(ok)} rows)")
    print(f"{'scene-const dirs':22s} {[AVG_SCALE_DIRS[r] for r in args.avg_scale_refs]} "
          f"<- median({args.avg_scale_key})")
    if args.clip_avg_scale_ref:
        print(f"{'clip-own dir (leaky)':22s} {AVG_SCALE_DIRS[args.clip_avg_scale_ref]}")
    for side, (p, cnt) in sorted(seg_paths.items()):
        print(f"{'seg_list ' + side:22s} {p} ({cnt} segments)")
    if not seg_paths:
        print(f"{'seg_list':22s} (없음 -> base.py 가 train_frac 랜덤 분할, scene 공유)")


if __name__ == "__main__":
    main()
