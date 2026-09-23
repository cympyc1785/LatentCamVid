"""`eval_data/cameras/<scene>/<preset>.npz` (내보낸 최종 target 카메라) 를 depth warp 로 렌더해
scene 별 한 편으로 붙인다 — SOURCE | preset 들 타일, 구멍은 마젠타.

왜 새 스크립트인가: `render_bank_videos.py` 는 뱅크(`poses.npz`)를, `render_pred_depth_warp.py`
는 latentcam eval JSON 을 읽는다 — **내보낸 cameras npz** (DA3 world c2w (49,4,4) + intrinsics
(49,4)=fx,fy,cx,cy) 를 읽는 렌더 코드는 없었다. 렌더 조립은 전부 재사용.

env: `vista4d`.  예시:
    CUDA_VISIBLE_DEVICES=1 python viz/render_target_cams_warp.py \
        --scenes snowboard woman-beach soapbox --out_dir results/20260828_target_cams_warp
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.overlay import contact_sheet, label_tile                               # noqa: E402
from lbm.presets import row_preset                                              # noqa: E402
from lbm.render import add_cloud_source_args, open_renderer                     # noqa: E402
from viz.render_bank_videos import render_variant, write_video              # noqa: E402


def training_target_jobs(dl3dv_root: str, scene: str, per_preset: int):
    """`--from_target_poses`: 학습이 보는 `da3/target_poses.npz` 에서 preset 당 앞 N 변이.

    w2c (V,T,4,4) -> c2w, K 는 (V,T,3,3) 픽셀 공간 (`render_target_poses_depth.load_targets`
    와 같은 규약). preset 은 prompts.json 의 seg key 매핑에서 읽는다. 반환은 cameras 모드와
    같은 (name, c2w (T,4,4), intr (T,4)) — 하류 렌더 경로를 하나로 유지하기 위함.
    """
    base = path.join(dl3dv_root, "vista4d", scene, "da3")
    npz = np.load(path.join(base, "target_poses.npz"), allow_pickle=False)
    c2w_all = np.linalg.inv(np.asarray(npz["extrinsics"], dtype=np.float64))
    K_all = np.asarray(npz["intrinsics"], dtype=np.float64)          # (V,T,3,3)
    keys = [str(k) for k in npz["keys"].tolist()]
    with open(path.join(base, "prompts.json"), encoding="utf-8") as file:
        prompts = json.load(file)
    picked, seen = [], {}
    for v, key in enumerate(keys):
        # 옛 코퍼스는 `straight_ease` 라 적고 `dolly_in` 을 굽는다. 이 이름이 곧 렌더 파일 이름이라
        # 안 풀면 산출물 이름이 궤적과 어긋난다 (§`lbm.presets.row_preset`).
        entry = prompts.get(key, {})
        preset = row_preset(entry) if entry.get("preset") else "?"
        if seen.get(preset, 0) >= per_preset:
            continue
        seen[preset] = seen.get(preset, 0) + 1
        intr = np.stack([K_all[v, :, 0, 0], K_all[v, :, 1, 1],
                         K_all[v, :, 0, 2], K_all[v, :, 1, 2]], axis=1)   # (T,4)
        picked.append((preset if per_preset == 1 else f"{preset}#{seen[preset]}",
                       c2w_all[v], intr))
    return picked


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    index = {}
    for scene in args.scenes:
        out_root = path.join(CLOUD_ROOT, "out")
        # npz 모드만 cloud.npz 를 요구한다 — memory 모드는 recon 에서 그 자리에 굽는다.
        if args.cloud_source == "npz" and not path.isfile(
                path.join(out_root, scene, "cloud.npz")):
            print(f"[skip] {scene}: cloud.npz 없음")
            continue
        if args.from_target_poses:
            jobs = training_target_jobs(args.dl3dv_root, scene, args.per_preset)
        else:
            jobs = []
            for npz_path in sorted(glob(path.join(args.root, "cameras", scene, "*.npz"))):
                z = np.load(npz_path)
                jobs.append((path.basename(npz_path)[:-4],
                             np.asarray(z["cam_c2w"], dtype=np.float64),
                             np.asarray(z["intrinsics"], dtype=np.float64)))
        if not jobs:
            print(f"[skip] {scene}: 카메라 없음")
            continue
        # `fixed_focal=True` 는 이 릴의 규약이다 (focal 배율을 K_src 에 곱해 쓴다).
        renderer, recon = open_renderer(args, out_root, want_recon=args.with_source,
                                        fixed_focal=True, video=scene)

        clips, labels, holes = [], [], {}
        for name, c2w, intr in jobs:
            # `intr` (T,4)=fx,fy,cx,cy 는 프레임별로 변한다 (zoom preset) 그리고 frame0 부터
            # cloud K 와 다를 수 있다 (snowboard back-follow: fx 593 vs cloud 1185).
            # 그래서 등호 assert 가 아니라 **비율**로 넘긴다 — render_variant(focal=...) 가
            # K_src 의 fx,fy 에 프레임별 배율을 곱한다 (intrinsic zoom 경로 그대로).
            store_w = int(round(float(intr[0, 2]) * 2))
            fx_scaled = np.asarray(intr[:, 0], dtype=np.float64) * renderer.width / store_w
            focal = fx_scaled / float(renderer.K_src[0][0, 0])
            _, frames = render_variant(renderer, c2w, args.tile_height, args.tile_width, 1,
                                       focal=focal)
            clips.append(frames)
            labels.append(name)
            hole = float(np.mean([(np.all(f == (255, 0, 255), axis=-1)).mean() for f in frames]))
            holes[name] = round(hole, 4)

        source = None
        if args.with_source:
            import cv2
            video = recon["video"]
            source = [cv2.resize(video[f], (args.tile_width, args.tile_height))
                      for f in range(len(video))]

        length = min(len(c) for c in clips)
        columns = min(args.columns, len(clips) + (1 if source is not None else 0))
        sheet = []
        for f in range(length):
            tiles = []
            if source is not None:
                tiles.append(label_tile(source[f].copy(), "SOURCE", f"{scene}  frame {f}"))
            for name, clip in zip(labels, clips):
                tiles.append(label_tile(clip[f].copy(), name, f"hole {holes[name]:.2f}"))
            sheet.append(contact_sheet(tiles, columns=columns))
        out_path = path.join(args.out_dir, f"{scene}__target_warp.mp4")
        write_video(out_path, sheet, args.fps)
        index[scene] = {"presets": labels, "hole": holes}
        print(f"{scene:<16}" + "  ".join(f"{k}:{v:.2f}" for k, v in holes.items()), flush=True)

    with open(path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)
    print(f"\n{'scenes':<10}{len(index)}\n{'out':<10}{args.out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--root", default=EVAL_DATA_DEFAULT)
    add_cloud_source_args(parser, eval_data_default=EVAL_DATA_DEFAULT)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile_width", type=int, default=480)
    parser.add_argument("--tile_height", type=int, default=270)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--with_source", action="store_true", default=True)
    parser.add_argument("--no_with_source", dest="with_source", action="store_false")
    # 학습이 보는 합성 target (`<scene>/da3/target_poses.npz`) 를 preset 당 N개씩 렌더
    parser.add_argument("--from_target_poses", action="store_true", default=False)
    parser.add_argument("--per_preset", type=int, default=1)
    parser.add_argument("--dl3dv_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3")
    main(parser.parse_args())
