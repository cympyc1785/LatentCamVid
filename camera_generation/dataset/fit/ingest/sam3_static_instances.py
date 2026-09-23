"""정적 명사 → SAM3 text PCS → `seg_instances_static/<video>/{meta.json,masks.npz}`.

`video_generation/scripts/sam3_seg_instances.py` 의 정적 버전이다. 다른 점은 세 가지뿐이고,
나머지(형틀 · 샤딩 · 출력 포맷 `vista4d_seg_instances_v1`)는 그대로다:

  ① keyword 출처가 `metadata.csv:dynamic`(저자 정답) 이 아니라 `fit/graph/extract_static_nouns.py`
     가 만든 `static_nouns.json` 이다. 정적 명사는 아무도 정답을 안 적어놨다.
  ② 출력 루트가 `seg_instances_static/` — 배포본 `seg_instances/`(동적) 를 덮지 않는다.
     `scene_graph/io.py:101` 이 두 루트를 각각 `kind="dyn"` / `kind="stat"` 로 읽어 합친다.
  ③ **동적 픽셀 비율로 걸러낸다.** VLM 이 static 이라고 부른 게 실제로는 움직이는 경우가 있다
     (avocado-slice `wheelchair` — 사람과 함께 움직인다). 그런 track 이 `stat_*` 노드가 되면
     `relations.supported_by` 가 "움직이는 것에 얹혀 있다"는 틀린 관계를 만들고, 그 문장이
     VLM 프롬프트로 그대로 나간다. 그래서 배포본 `dynamic_mask` 와의 겹침이
     `--max_dynamic_frac` 을 넘으면 저장 전에 뺀다. 뺀 track 은 `static_report.json` 에 남는다.

왜 SAM3 를 정적에도 돌리나 (`static_mask` 가 이미 있는데): `static_mask` 는 인스턴스 구분이
없는 한 장짜리 이진 마스크라 "the fence" 를 지목할 수 없다. 정적 노드의 용도는 자유공간이
아니라 **VLM 이 물체를 이름으로 부르는 것**이므로 (§scene_graph/relations.py) 인스턴스가 필요하다.

예시 (env vista4d, GPU 1장):
    CUDA_VISIBLE_DEVICES=1 python fit/ingest/sam3_static_instances.py --videos camel avocado-slice
    CUDA_VISIBLE_DEVICES=1 python fit/ingest/sam3_static_instances.py --num_shards 4 --shard_id 0
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
VISTA4D_ROOT_DEFAULT = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"
STATIC_NOUNS_DEFAULT = path.join(HERE, "out", "static_nouns", "static_nouns.json")


def read_static_nouns(nouns_path: str, field: str = "static"):
    """video -> keyword 목록. `extract_static_nouns.py` 의 `static_nouns_v1`.

    `field="surface"` 면 광역 표면 명사(`floor`/`ground`)를 읽는다. 그 결과는 **노드가 아니라
    지면 높이용**이므로 `--output_root` 를 따로 줘서 `seg_instances_static/` 을 덮지 않게 할 것
    (§extract_static_nouns.select docstring). 옛 `static_nouns.json` 에는 `surface` 열이 아예
    없으므로 `.get(field, [])` 로 읽어 KeyError 대신 "명사 없음 → skip" 으로 떨어뜨린다.
    """
    with open(nouns_path, encoding="utf-8") as file:
        payload = json.load(file)
    assert payload["format"] == "static_nouns_v1", f"알 수 없는 포맷: {payload['format']}"
    return {video: entry.get(field, []) for video, entry in payload["videos"].items()}


def track_dynamic_fraction(seg_frames, dynamic_mask):
    """track id -> (자기 마스크 픽셀 중 dynamic_mask 안에 든 비율, 등장 프레임 수).

    프레임별 비율의 평균이 아니라 **픽셀 총합의 비율**이다. 큰 물체가 한 프레임만 살짝 겹친 것과
    작은 물체가 내내 겹친 것을 구분해야 하는데, 프레임 평균은 후자를 과소평가한다.
    """
    inside, total, frames = {}, {}, {}
    for f, instances in enumerate(seg_frames):
        for instance in instances:
            mask = np.asarray(instance["mask"], dtype=bool)
            track = int(instance["id"])
            inside[track] = inside.get(track, 0) + int(np.count_nonzero(mask & dynamic_mask[f]))
            total[track] = total.get(track, 0) + int(np.count_nonzero(mask))
            frames[track] = frames.get(track, 0) + 1
    return {track: (inside[track] / total[track] if total[track] else 1.0, frames[track])
            for track in total}


def main(args):
    sys.path.insert(0, args.vista4d_root)
    from utils.media import load_masks, load_video
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video
    from utils.recon_and_seg.seg_sam3_utils import save_seg_instances

    recon_root = path.join(args.eval_data, "eval_data", "recon_and_seg")
    output_root = args.output_root or path.join(args.eval_data, "eval_data", "seg_instances_static")
    nouns_of = read_static_nouns(args.static_nouns, args.nouns_field)

    videos = args.videos or sorted(nouns_of)
    missing = [v for v in videos if v not in nouns_of]
    assert not missing, f"static_nouns.json 에 없는 영상: {missing}"
    if args.num_shards > 1:
        videos = videos[args.shard_id::args.num_shards]
    print(f"{len(videos)} videos (shard {args.shard_id}/{args.num_shards}) -> {output_root}")

    todo = []
    for video in videos:
        if path.isfile(path.join(output_root, video, "masks.npz")) and args.skip_done:
            print(f"  skip (done): {video}")
            continue
        if not nouns_of[video]:
            print(f"  skip (정적 명사 없음): {video}")
            continue
        todo.append(video)
    if not todo:
        print("할 일이 없다.")
        return

    # SAM3 는 한 번만 로드한다 (영상당 재로드하면 대부분의 시간이 로딩이 된다).
    video_predictor = init_sam3_video()

    report = []
    for n, video in enumerate(todo, 1):
        keywords = nouns_of[video]
        print(f"\n[{n}/{len(todo)}] {video}  keywords={keywords}", flush=True)
        frames, fps = load_video(path.join(recon_root, video, "video.mp4"))
        num_frames, height, width, _ = frames.shape

        _, seg_frames = run_sam3_video(frames, video_predictor, keywords)
        dynamic_mask = load_masks(path.join(recon_root, video, "dynamic_mask"))
        stats = track_dynamic_fraction(seg_frames, dynamic_mask)

        # 동적 픽셀에 얹힌 track 은 저장 전에 뺀다 (§docstring ③). keyword 는 남긴다 —
        # "이 명사를 시도했는데 정적이 아니었다" 가 meta 만 보고도 읽혀야 한다.
        drop = {track for track, (frac, _) in stats.items() if frac > args.max_dynamic_frac}
        keyword_of = {int(i["id"]): i["keyword"] for f in seg_frames for i in f}
        kept_frames = [[i for i in instances if int(i["id"]) not in drop] for instances in seg_frames]

        output_folder = path.join(output_root, video)
        makedirs(output_folder, exist_ok=True)
        save_seg_instances(output_folder, kept_frames, keywords, height, width)
        tracks = [{"id": track, "keyword": keyword_of[track],
                   "dynamic_frac": round(frac, 4), "frames": count,
                   "kept": track not in drop} for track, (frac, count) in sorted(stats.items())]
        with open(path.join(output_folder, "static_report.json"), "w", encoding="utf-8") as file:
            json.dump({"video": video, "keywords": keywords,
                       "max_dynamic_frac": args.max_dynamic_frac, "tracks": tracks},
                      file, ensure_ascii=False, indent=1)

        print(f"  -> {len(stats) - len(drop)}/{len(stats)} tracks kept, {num_frames} frames")
        for track in tracks:
            print(f"     {'keep' if track['kept'] else 'DROP'}  #{track['id']:<3}"
                  f"{track['keyword']:<16}dyn_frac={track['dynamic_frac']:.3f}"
                  f"  frames={track['frames']}")
        report.append((video, len(keywords), len(stats), len(stats) - len(drop)))

        if args.save_vis:
            import imageio.v2 as imageio

            from utils.recon_and_seg.seg_sam3_utils import overlay_instances_on_video
            overlay = overlay_instances_on_video(frames, kept_frames, alpha=0.5)
            # cv2 mp4v 는 VS Code 뷰어에서 안 열린다 -> imageio + libx264.
            imageio.mimwrite(path.join(output_folder, "vis_instances.mp4"), overlay,
                             fps=int(round(fps)), codec="libx264", quality=6, macro_block_size=1)

    print(f"\n{'video':<22}{'nouns':>7}{'tracks':>8}{'kept':>6}")
    for video, num_keywords, num_tracks, num_kept in report:
        # tracks > nouns 면 명사 하나가 여러 인스턴스를 냈다는 뜻 (`tree` 3그루 등) — 정상이다.
        print(f"{video:<22}{num_keywords:>7}{num_tracks:>8}{num_kept:>6}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)  # None = <eval_data>/eval_data/seg_instances_static
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--static_nouns", default=STATIC_NOUNS_DEFAULT, type=str)
    # static_nouns.json 의 어느 열을 SAM3 에 먹일지. `surface` 는 바닥 마스크용이고 노드가
    # 아니다 — 그때는 `--output_root .../seg_instances_floor` 로 따로 뺄 것.
    parser.add_argument("--nouns_field", default="static", choices=["static", "surface"])

    parser.add_argument("--videos", nargs="*", default=None)      # None = static_nouns.json 전체
    parser.add_argument("--num_shards", default=1, type=int)
    parser.add_argument("--shard_id", default=0, type=int)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    # 배포본 dynamic_mask 와 이만큼 넘게 겹치면 정적이 아니다 (§docstring ③)
    parser.add_argument("--max_dynamic_frac", default=0.5, type=float)
    parser.add_argument("--save_vis", action="store_true", default=False)

    main(parser.parse_args())
