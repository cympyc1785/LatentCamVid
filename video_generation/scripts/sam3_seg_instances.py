"""Vista4D-Eval-Data 의 각 소스 영상에 SAM3 만 다시 돌려 per-instance track 을 뽑는다.

왜 full recon 을 다시 돌리지 않는가: `eval_data/recon_and_seg/` 는 저자 배포본이다
(`huggingface-cli download Eyeline-Labs/Vista4D-Eval-Data`). depth / cameras / dynamic_mask 는
그대로 쓸 것이고 방금 끝낸 eval 110 entry 가 바로 그 파일들에 의존한다. 우리가 추가로 필요한 건
SAM3 의 per-instance mask / track id / keyword 뿐이므로 Pi3·DA3 재구성은 낭비이고, 배포본을
덮어쓰면 eval 재현성이 깨진다. 그래서 이 스크립트는 **읽기만** 하고 결과를 별도 루트에 쓴다.

입력  : `<eval_data>/recon_and_seg/<video>/video.mp4` (전처리 완료본: 1280x720, 49 프레임)
        `<Vista4D-Eval-Data>/metadata.csv` 의 `dynamic` 열 = 그 영상의 SAM3 keyword 목록
출력  : `<output_root>/<video>/{meta.json,masks.npz}` (format `vista4d_seg_instances_v1`)

keyword 중복 병합은 하지 않는다 — `scripts/merge_seg_instances.py` 후처리에서 한다.

예시 (env vista4d 필요):
    CUDA_VISIBLE_DEVICES=0 python scripts/sam3_seg_instances.py --num_shards 4 --shard_id 0
    CUDA_VISIBLE_DEVICES=0 python scripts/sam3_seg_instances.py --videos camel avocado-slice --save_vis
"""
import csv
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
VISTA4D_ROOT_DEFAULT = path.join(path.dirname(path.dirname(path.abspath(__file__))), "models", "Vista4D")


def read_keywords(metadata_csv: str):
    """video -> keyword 목록. metadata.csv 는 entry(video x camera) 단위라 video 당 여러 행이 있다."""
    keywords_of = {}
    with open(metadata_csv, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            video = row["video"]
            keywords = [k.strip() for k in row["dynamic"].split(",") if k.strip()]
            if video in keywords_of:
                assert keywords_of[video] == keywords,\
                    f"{video}: 행마다 dynamic 열이 다르다 {keywords_of[video]} vs {keywords}"
            else:
                keywords_of[video] = keywords
    return keywords_of


def dynamic_mask_iou(seg_frames, dynamic_mask):
    """우리 SAM3 재실행 결과를 OR 로 뭉갠 것 vs 배포본 dynamic_mask 의 프레임 평균 IoU.

    저자 배포본은 저자 환경의 SAM3 로 만든 것이라 우리 재실행과 완전히 같을 보장이 없다.
    낮으면 keyword 나 모델 버전이 어긋난 것이므로 병합 임계를 논하기 전에 이걸 먼저 봐야 한다.
    """
    ious = []
    for i, instances in enumerate(seg_frames):
        ours = np.zeros_like(dynamic_mask[i], dtype=np.bool_)
        for instance in instances:
            ours |= instance["mask"]
        union = np.count_nonzero(ours | dynamic_mask[i])
        ious.append(np.count_nonzero(ours & dynamic_mask[i]) / union if union else 1.0)
    return float(np.mean(ious))


def main(args):
    sys.path.insert(0, args.vista4d_root)
    from utils.media import load_masks, load_video
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video, run_sam3_video
    from utils.recon_and_seg.seg_sam3_utils import save_seg_instances

    recon_root = path.join(args.eval_data, "eval_data", "recon_and_seg")
    output_root = args.output_root or path.join(args.eval_data, "eval_data", "seg_instances")
    keywords_of = read_keywords(path.join(args.eval_data, "metadata.csv"))

    videos = args.videos or sorted(keywords_of)
    missing = [v for v in videos if v not in keywords_of]
    assert not missing, f"metadata.csv 에 없는 영상: {missing}"
    if args.num_shards > 1:
        videos = videos[args.shard_id::args.num_shards]
    print(f"{len(videos)} videos (shard {args.shard_id}/{args.num_shards}) -> {output_root}")

    todo = []
    for video in videos:
        done = path.isfile(path.join(output_root, video, "masks.npz"))
        if done and args.skip_done:
            print(f"  skip (done): {video}")
            continue
        if not keywords_of[video]:
            print(f"  skip (keyword 없음): {video}")
            continue
        todo.append(video)
    if not todo:
        print("할 일이 없다.")
        return

    # SAM3 는 한 번만 로드한다 (영상당 재로드하면 대부분의 시간이 로딩이 된다).
    video_predictor = init_sam3_video()

    report = []
    for n, video in enumerate(todo, 1):
        keywords = keywords_of[video]
        print(f"\n[{n}/{len(todo)}] {video}  keywords={keywords}")
        frames, fps = load_video(path.join(recon_root, video, "video.mp4"))
        num_frames, height, width, _ = frames.shape

        _, seg_frames = run_sam3_video(frames, video_predictor, keywords)
        num_tracks = len({instance["id"] for f in seg_frames for instance in f})

        output_folder = path.join(output_root, video)
        makedirs(output_folder, exist_ok=True)
        save_seg_instances(output_folder, seg_frames, keywords, height, width)

        iou = float("nan")
        if args.check_dynamic_mask:
            iou = dynamic_mask_iou(seg_frames, load_masks(path.join(recon_root, video, "dynamic_mask")))
        print(f"  -> {num_tracks} tracks, {num_frames} frames"
              f"{f', dynamic_mask IoU={iou:.3f}' if args.check_dynamic_mask else ''}")
        report.append((video, len(keywords), num_tracks, iou))

        if args.save_vis:
            import imageio.v2 as imageio

            from utils.recon_and_seg.seg_sam3_utils import overlay_instances_on_video
            overlay = overlay_instances_on_video(frames, seg_frames, alpha=0.5)
            # cv2 mp4v 는 VS Code 뷰어에서 안 열린다 -> imageio + libx264.
            imageio.mimwrite(path.join(output_folder, "vis_instances.mp4"), overlay,
                             fps=int(round(fps)), codec="libx264", quality=6, macro_block_size=1)

    print(f"\n{'video':<22}{'keywords':>9}{'tracks':>8}{'dynIoU':>9}")
    for video, num_keywords, num_tracks, iou in report:
        # tracks > keywords 면 keyword 하나가 여러 인스턴스를 냈거나 중복 keyword 가 겹친 것이다.
        print(f"{video:<22}{num_keywords:>9}{num_tracks:>8}{iou:>9.3f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)  # None = <eval_data>/eval_data/seg_instances
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)

    parser.add_argument("--videos", nargs="*", default=None)  # None = metadata.csv 전체
    parser.add_argument("--num_shards", default=1, type=int)
    parser.add_argument("--shard_id", default=0, type=int)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    parser.add_argument("--check_dynamic_mask", action="store_true", default=True)
    parser.add_argument("--no_check_dynamic_mask", dest="check_dynamic_mask", action="store_false")
    parser.add_argument("--save_vis", action="store_true", default=False)

    main(parser.parse_args())
