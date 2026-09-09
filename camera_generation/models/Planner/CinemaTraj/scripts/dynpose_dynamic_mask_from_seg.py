"""dynpose recon 의 `dynamic_mask/` 를 SAM3 seg_instances 합집합으로 갈아끼운다.

WHY: vista 배포본의 dynamic_mask 는 SAM3 dynamic keyword 트랙의 합집합인데, dynpose 는 우리가
recon 을 직접 돌리므로 그 단계가 없다. recon 은 `--seg_keywords`(빈 목록) + `--keep_recon_sky`
로 dynamic=0 / sky=DA3 로 만들어 두고, 이 스크립트가 `seg_instances/<video>/masks.npz`
(VLM 명사 -> SAM3 트랙, 전부 dynamic 명사) 의 프레임별 합집합을 png 로 덮어쓴다.
`_all_`(전부 1) 로 두면 scene_graph 의 정적 점이 0 이 되어 죽는다 (2026-08-28 파일럿 실측).

env: 아무거나 (numpy + PIL)

예시:
    python scripts/dynpose_dynamic_mask_from_seg.py --eval_data /data1/.../DynPose-LBM \
        --videos 00e9f728-... 015b197d-...
"""
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np
from PIL import Image


def main(args):
    root = path.join(args.eval_data, "eval_data")
    videos = args.videos or [path.basename(p) for p in sorted(glob(path.join(root, "seg_instances", "*")))]
    for video in videos:
        seg_path = path.join(root, "seg_instances", video, "masks.npz")
        mask_dir = path.join(root, "recon_and_seg", video, "dynamic_mask")
        if not (path.isfile(seg_path) and path.isdir(mask_dir)):
            print(f"[skip] {video}: seg 또는 recon 없음")
            continue
        import json
        z = np.load(seg_path)
        with open(path.join(root, "seg_instances", video, "meta.json"), encoding="utf-8") as file:
            meta = json.load(file)
        # 포맷: 프레임당 키 "%05d" = (n_instances, H, W/8) uint8 **packbits**
        # (seg_sam3_utils.py:98 packbits / :114 unpackbits 규약 그대로).
        frames = sorted(z.files)
        union = np.stack([
            np.unpackbits(z[key], axis=-1, count=meta["width"]).astype(bool).any(axis=0)
            for key in frames])                                            # (T,H,W)
        pngs = sorted(glob(path.join(mask_dir, "*.png")))
        assert len(pngs) == union.shape[0], f"{video}: 프레임 수 불일치 {len(pngs)} vs {union.shape[0]}"
        for f, png in enumerate(pngs):
            frame = union[f]
            if frame.shape != np.array(Image.open(png)).shape:
                frame = np.array(Image.fromarray(frame.astype(np.uint8) * 255)
                                 .resize(Image.open(png).size, Image.NEAREST)) > 0
            Image.fromarray((frame * 255).astype(np.uint8)).save(png)
        print(f"{video}: dynamic_mask {union.shape} 덮어씀 (mean {union.mean():.4f})")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    main(parser.parse_args())
