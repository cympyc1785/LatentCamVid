"""recon_and_seg 1편 → 확인용 영상 3종 (seg overlay · depth · 셋을 쌓은 비교본).

왜 별도 스크립트인가: `recon_and_seg_single.py --save_vis` 는 2×2 격자 한 장
(depth | sky / seg | dynamic) 만 남긴다. 타일이 640×360 으로 줄어 마스크 경계가 안 보이고,
**정적 인스턴스(`seg_instances_static/`)는 아예 안 들어간다** — 그건 나중에 별도 스크립트가
만들기 때문이다. 사람이 "이 물체를 제대로 잡았나"를 보려면 dyn+stat 을 한 화면에 원해상도로
겹친 영상이 필요하다.

3D OBB 를 그린 영상은 여기서 안 만든다 — `build_scene_graph.py` 가 `obb_overlay*.mp4` 로
이미 낸다. 이 스크립트는 그 앞단(마스크·깊이)만 담당한다.

env: `vista4d`

예시:
    python viz/make_lite_previews.py --eval_data /data1/.../JoggingWoman-Lite \
        --video jogging-woman --out_dir results/20260824_jogging_woman
"""
import sys
from argparse import ArgumentParser
from os import makedirs, path

import cv2
import imageio
import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "video_generation", "models", "Vista4D"))

#    dyn 은 따뜻한 색, stat 은 차가운 색으로 갈라 둔다 — 한 화면에 겹치므로 종류가 구분돼야 한다.
DYN_COLORS = [(255, 90, 60), (255, 190, 40), (255, 60, 160)]
STAT_COLORS = [(60, 200, 255), (90, 255, 140), (170, 130, 255)]


def write_video(out_path: str, frames, fps: float):
    #    cv2 의 mp4v 는 VS Code 뷰어에서 안 열린다 (파일은 멀쩡해서 조용히 지나간다).
    imageio.mimwrite(out_path, list(frames), fps=fps, codec="libx264",
                     quality=6, macro_block_size=1)
    return out_path


def overlay_masks(video, groups, alpha: float = 0.45):
    """groups: [(kind, seg_frames, meta)] → 마스크 채색 + 라벨/박스를 얹은 (F,H,W,3)."""
    out = video.copy()
    for frame_index in range(len(out)):
        canvas = out[frame_index].astype(np.float32)
        labels = []
        for kind, seg_frames, _ in groups:
            palette = DYN_COLORS if kind == "dyn" else STAT_COLORS
            for instance in seg_frames[frame_index]:
                mask = np.asarray(instance["mask"], dtype=bool)
                if not mask.any():
                    continue
                color = np.array(palette[int(instance["id"]) % len(palette)], dtype=np.float32)
                canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color
                x0, y0, x1, y1 = [int(round(v)) for v in np.asarray(instance["box_xyxy"])]
                labels.append((x0, y0, x1, y1, color,
                               f"{kind}:{instance['keyword']}#{instance['id']}"
                               f" {float(instance.get('score', 0.0)):.2f}"))
        frame = canvas.astype(np.uint8)
        for x0, y0, x1, y1, color, text in labels:
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.rectangle(frame, (x0, y0), (x1, y1), bgr, 2)
            cv2.putText(frame, text, (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, bgr, 1, cv2.LINE_AA)
        out[frame_index] = frame
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--eval_data", required=True, type=str)
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)         # None = <eval_data>/eval_data/seg_instances
    parser.add_argument("--seg_static_root", default=None, type=str)  # None = <...>/seg_instances_static
    parser.add_argument("--alpha", default=0.45, type=float)
    parser.add_argument("--fps", default=0.0, type=float)             # 0 = recon 의 fps 를 따른다
    args = parser.parse_args()

    if args.vista4d_root not in sys.path:
        sys.path.insert(0, args.vista4d_root)
    from utils.media import depths_to_disparity_video, load_recon_and_seg      # noqa: E402
    from utils.recon_and_seg.seg_sam3_utils import load_seg_instances          # noqa: E402

    recon = load_recon_and_seg(path.join(args.eval_data, "eval_data", "recon_and_seg", args.video))
    fps = args.fps or float(recon["fps"])
    makedirs(args.out_dir, exist_ok=True)

    seg_root = args.seg_root or path.join(args.eval_data, "eval_data", "seg_instances")
    static_root = args.seg_static_root or path.join(args.eval_data, "eval_data",
                                                    "seg_instances_static")
    #    동적은 두 자리 중 아무 데나 있을 수 있다 — `recon_and_seg_single.py --save_seg_instances`
    #    는 `recon_and_seg/<video>/seg_instances` 에 쓰는데 코퍼스 규약은 `eval_data/seg_instances/
    #    <video>` 다. `scene_graph/io.py:104-108` 이 쓰는 것과 같은 폴백.
    inline_root = path.join(args.eval_data, "eval_data", "recon_and_seg", args.video)
    groups = []
    for kind, root in (("dyn", seg_root), ("dyn", inline_root), ("stat", static_root)):
        folder = path.join(root, args.video) if root is not inline_root else \
            path.join(root, "seg_instances")
        if any(g[0] == kind for g in groups):
            continue
        if path.isfile(path.join(folder, "meta.json")):
            seg_frames, meta = load_seg_instances(folder)
            assert meta["num_frames"] == len(recon["video"]), f"{kind} seg 가 recon 과 프레임 수가 다르다"
            groups.append((kind, seg_frames, meta))
    assert groups, f"seg_instances 를 못 찾았다: {seg_root} / {static_root}"

    seg_path = write_video(path.join(args.out_dir, "seg_overlay.mp4"),
                           overlay_masks(recon["video"], groups, args.alpha), fps)
    #    sky 를 빼고 정규화한다 — 하늘의 disparity≈0 이 분모에 들어가면 전경 대비가 다 죽는다.
    depth_path = write_video(path.join(args.out_dir, "depth.mp4"),
                             depths_to_disparity_video(recon["depths"], ~recon["sky_mask"],
                                                       use_colormap=True), fps)

    print(f"{'video':14} {args.video}   {len(recon['video'])} 프레임 @ {fps:g} fps")
    for kind, seg_frames, meta in groups:
        per_frame = [len(f) for f in seg_frames]
        print(f"{kind + ' 인스턴스':14} keywords {meta['keywords']}   "
              f"프레임당 {min(per_frame)}~{max(per_frame)} 개  "
              f"(검출 프레임 {sum(1 for c in per_frame if c)}/{len(per_frame)})")
    print(f"{'seg overlay':14} {seg_path}")
    print(f"{'depth':14} {depth_path}")


if __name__ == "__main__":
    main()
