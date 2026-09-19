"""dynpose split 엔트리 -> GenDoP `text_rgbd` 추론 입력 (RGBD 캐시 + 텍스트 case (a)).

WHY: `gendop_release_infer.py` 는 이미 rgbd 분기를 갖고 있지만 그게 찾는 레이아웃은
     Vista4D eval 의 `<gen_root>/<scene>/<name>/monst3r/{frame_0000.png, frame_depth_0000.npy}`
     다. dynpose 코퍼스는 `eval_data/recon_and_seg/<uuid>/{video.mp4, depths/00000.exr}` 라
     모양이 다르다. **추론 스크립트를 고치는 대신 그 레이아웃을 만들어 준다** — 그래야
     기존 vista4d 경로가 한 줄도 안 바뀐다 (bit-identical).

두 가지를 같이 만든다. 둘 다 `--out` 밑에 떨어진다:

  ① `rgbd/<uuid>/<idx>/rgbd/{frame_0000.png, frame_depth_0000.npy}`
       RGB   = `recon_and_seg/<uuid>/video.mp4` 의 frame 0. 소스와 target 은 frame0 을
               공유하므로(뱅크 frame0 공유 규약) 이게 곧 target 궤적의 첫 프레임이다.
       depth = `depths/00000.exr` **그대로**. 정규화하지 않는다 — `eval.py:145 standard_depth`
               가 center-crop/zero-pad 만 하고, 실측 게이지가 GenDoP 예시와 같은 대역이다
               (dynpose 0.191~0.683 med 0.411 / GenDoP `text_rgbd/case1_depth.npy` 0.266~0.722).
       한 scene 의 여러 엔트리가 같은 파일을 보므로 엔트리 디렉토리는 **심링크**다.

  ② `text_motion/<dataset>/<uuid>/da3/captions_gendop/<idx>_caption.json`
       사용자 지시 "우리 텍스트에서 motion: 뒤에만" — `prompts.json[<idx>]
       .caption_fields.motion` 한 절만 담은 캡션 미러 루트. `gendop_release_infer.py
       --root <여기> --split <같은 split>` 로 그대로 물린다 (`collect()` 가 split 모드에서
       찾는 경로가 정확히 이 모양이다). 코퍼스 원본은 **안 건드린다**.

텍스트 case (b)(GT 카메라를 pipeline/GenDoP 방식으로 태깅+캡션)는 이 스크립트가 아니라
`caption_cameras_datadop.py --sets latentcam` 이 코퍼스 안 `da3/captions_gendop/` 에 쓴다.

env: GenDoP (또는 da3/latentcam). vista4d 는 안 된다 — cv2 가 .exr 에 None 을 돌려준다.

예시:
    conda run -n GenDoP python scripts/dynpose_gendop_inputs.py \
        --split /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose/seg_list_dynpose_dionly_test.txt \
        --out results/20260901_gendop_dynpose_val
"""
import json
import os
from argparse import ArgumentParser
from os import makedirs, path, symlink

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # cv2 import 전에 켜야 한다

import cv2                                                            # noqa: E402
import imageio.v2 as imageio                                          # noqa: E402
import numpy as np                                                    # noqa: E402

CORPUS = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose"
EVAL_DATA = "/data1/cympyc1785/data/DynPose-LBM/eval_data"
TEXT_KEY = "Concise Interaction"      # text_rgbd ckpt 가 학습된 문체 키


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)          # `<dataset>/<scene>/<idx>` 한 줄씩
    parser.add_argument("--corpus", default=CORPUS)        # prompts.json 이 있는 곳
    parser.add_argument("--eval_data", default=EVAL_DATA)  # recon_and_seg 가 있는 곳
    parser.add_argument("--out", required=True)
    parser.add_argument("--text_key", default=TEXT_KEY)
    parser.add_argument("--frame", type=int, default=0)    # 조건으로 줄 프레임 (frame0 공유 규약)
    parser.add_argument("--overwrite", action="store_true")
    # D208. rgbd 를 아예 안 만들고 ② 텍스트 미러만 쓴다. `text_motion` ckpt 만 돌릴 때,
    # 또는 `--eval_data` 에 recon 이 일부만 있는 코퍼스에서 쓴다.
    # 주의: 처음에 dynpose_d200 을 "recon 이 1,017 중 107 편뿐"이라 보고 이 플래그를 걸었는데
    # 그건 `--eval_data` 를 `DynPose-LBM` 으로 준 탓이었다. d200 은 pooled dynpose-100k 라
    # `DynPose-100K/eval_data` 밑에 1,017/1,017 다 있다 — 107 은 두 코퍼스의 교집합이었다.
    # 기본 off = 예전 동작 비트 동일.
    parser.add_argument("--text_only", action="store_true")
    args = parser.parse_args()

    with open(args.split, encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    scenes = sorted({line.split("/")[1] for line in lines})
    # symlink 대상은 **절대경로**여야 한다 — 상대경로면 링크가 놓인 디렉토리 기준으로 풀려
    # 조용히 끊어진 링크가 되고, 추론 스크립트에는 "no_rgbd" 로만 보인다.
    rgbd_root = path.abspath(path.join(args.out, "rgbd"))
    text_root = path.join(args.out, "text_motion")

    # ── ① scene 당 RGB/depth 한 벌
    scene_files, missing = {}, []
    for scene in [] if args.text_only else scenes:
        base = path.join(args.eval_data, "recon_and_seg", scene)
        video = path.join(base, "video.mp4")
        depth_exr = path.join(base, "depths", f"{args.frame:05d}.exr")
        if not (path.exists(video) and path.exists(depth_exr)):
            missing.append(scene)
            continue
        shared = path.join(rgbd_root, "_scene", scene)
        makedirs(shared, exist_ok=True)
        rgb_out = path.join(shared, "frame_0000.png")
        depth_out = path.join(shared, "frame_depth_0000.npy")
        if args.overwrite or not path.exists(rgb_out):
            reader = imageio.get_reader(video)
            frame = np.asarray(reader.get_data(args.frame))
            reader.close()
            cv2.imwrite(rgb_out, frame[..., ::-1])          # standard_image 가 BGR 로 읽는다
        if args.overwrite or not path.exists(depth_out):
            depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
            if depth.ndim == 3:                              # 3채널로 저장된 EXR 이면 한 채널만
                depth = depth[..., 0]
            np.save(depth_out, depth.astype(np.float32))
        scene_files[scene] = (rgb_out, depth_out)

    # ── ② 엔트리별 심링크 디렉토리 + motion 캡션
    n_link, n_text, skipped = 0, 0, []
    depth_stats = []
    for line in lines:
        dataset, scene, index = line.split("/")
        if not args.text_only and scene not in scene_files:
            skipped.append(line)
            continue
        entry_dir = None if args.text_only else path.join(rgbd_root, scene, index, "rgbd")
        if entry_dir is not None:
            makedirs(path.dirname(entry_dir), exist_ok=True)
            if path.islink(entry_dir) or path.exists(entry_dir):
                if args.overwrite and path.islink(entry_dir):
                    os.remove(entry_dir)
                else:
                    n_link += 1
                    entry_dir = None
            if entry_dir is not None:
                symlink(path.join(rgbd_root, "_scene", scene), entry_dir)
                n_link += 1

        with open(path.join(args.corpus, dataset, scene, "da3", "prompts.json"),
                  encoding="utf-8") as file:
            entry = json.load(file)[index]
        motion = entry["caption_fields"]["motion"]
        cap_dir = path.join(text_root, dataset, scene, "da3", "captions_gendop")
        makedirs(cap_dir, exist_ok=True)
        with open(path.join(cap_dir, f"{index}_caption.json"), "w", encoding="utf-8") as file:
            json.dump({args.text_key: motion, "Movement": motion, "Detailed": motion,
                       "source": "prompts.json caption_fields.motion",
                       "variant_id": entry["variant_id"],
                       "anchor_label": entry["anchor_label"]},
                      file, ensure_ascii=False, indent=1)
        n_text += 1

    for scene, (_, depth_out) in scene_files.items():
        d = np.load(depth_out)
        depth_stats.append((float(np.nanmin(d)), float(np.nanmedian(d)), float(np.nanmax(d))))
    stats = np.asarray(depth_stats)

    print(f"{'split':<16}{args.split}")
    print(f"{'entries':<16}{len(lines)}")
    print(f"{'scenes':<16}{len(scenes)}   (rgbd 있음 {len(scene_files)}, 없음 {len(missing)})")
    print(f"{'rgbd links':<16}{n_link}")
    print(f"{'motion caps':<16}{n_text}")
    print(f"{'skipped':<16}{len(skipped)}  {skipped[:5]}")
    if len(stats):
        print(f"\ndepth 게이지 (scene {len(stats)}편, 정규화 안 함)")
        print(f"  min    med {np.median(stats[:, 0]):.4f}  범위 "
              f"{stats[:, 0].min():.4f}~{stats[:, 0].max():.4f}")
        print(f"  median med {np.median(stats[:, 1]):.4f}  범위 "
              f"{stats[:, 1].min():.4f}~{stats[:, 1].max():.4f}")
        print(f"  max    med {np.median(stats[:, 2]):.4f}  범위 "
              f"{stats[:, 2].min():.4f}~{stats[:, 2].max():.4f}")
        print("  참고: GenDoP assets/examples/text_rgbd/case1_depth.npy 0.266~0.722 (med 0.323)")
    print(f"\n--gen_root {rgbd_root}  --monst3r_sub rgbd")
    print(f"--root     {text_root}   (텍스트 case (a): motion 절만)")


if __name__ == "__main__":
    main()
