"""recon_and_seg(RGBD+카메라) 와 seg_instances(SAM3 track) 를 하나의 dict 로 합쳐 읽는다.

왜 로더를 따로 두는가: 두 소스가 **다른 스크립트가 다른 시점에** 만든 것이라 프레임 수 · 해상도가
어긋날 수 있고, 어긋난 채로 lift 하면 마스크가 한 프레임씩 밀린 점군이 나온다. 그 오류는 OBB 가
살짝 큰 정도로만 보여서 눈으로 안 잡힌다. 그래서 여기서 전부 assert 로 막는다.

`masks.npz` 는 프레임당 `(n_instances, H, W/8)` uint8 packbits 다. track 단위로 쓰려면
(F, H, W) 로 재조립해야 하는데 이게 track 당 45 MB(49x720x1280) 라, **요청한 track 만** 만든다.

입력  : `<eval_data>/eval_data/recon_and_seg/<video>/`
        `<eval_data>/eval_data/seg_instances/<video>/{meta.json,masks.npz}`         (동적)
        `<eval_data>/eval_data/seg_instances_static/<video>/{meta.json,masks.npz}`  (선택, 정적)
"""
import json
import sys
from os import path

import numpy as np

SEG_FORMAT = "vista4d_seg_instances_v1"


def import_vista4d(vista4d_root: str):
    """Vista4D 는 패키지가 아니라 `utils.*` 로 절대 import 하는 스크립트 트리라 sys.path 에 얹는다."""
    if vista4d_root not in sys.path:
        sys.path.insert(0, vista4d_root)
    from utils.media import intrinsics_to_K, load_recon_and_seg
    return {"load_recon_and_seg": load_recon_and_seg, "intrinsics_to_K": intrinsics_to_K}


class SegInstances:
    """`vista4d_seg_instances_v1` 한 벌. track id → per-frame box/score, 마스크는 lazy 재조립."""

    def __init__(self, folder: str, kind: str):
        with open(path.join(folder, "meta.json"), encoding="utf-8") as file:
            meta = json.load(file)
        assert meta["format"] == SEG_FORMAT, f"알 수 없는 seg format: {meta['format']}"
        self.kind = kind                      # "dyn" | "stat" — 노드 id 접두사가 된다
        self.meta = meta
        self.num_frames = int(meta["num_frames"])
        self.height, self.width = int(meta["height"]), int(meta["width"])
        self.keywords = list(meta["keywords"])
        self.frames = meta["frames"]          # [[{id,keyword,score,box_xyxy}, ...], ...]
        self._npz = np.load(path.join(folder, "masks.npz"))

        # track id 는 프레임을 건너뛰며 등장한다(occlusion). 등장 순서가 아니라 id 로 모은다.
        self.tracks = {}
        for f, instances in enumerate(self.frames):
            for slot, instance in enumerate(instances):
                track = self.tracks.setdefault(int(instance["id"]), {
                    "id": int(instance["id"]), "keyword": instance["keyword"],
                    "frames": [], "slots": [], "scores": [], "boxes": [],
                })
                track["frames"].append(f)
                track["slots"].append(slot)   # masks.npz 의 그 프레임 배열에서 몇 번째인지
                track["scores"].append(float(instance["score"]))
                track["boxes"].append([float(v) for v in instance["box_xyxy"]])

    def track_ids(self):
        return sorted(self.tracks)

    def frame_masks(self, frame: int):
        """(n_instances, H, W) bool — 그 프레임의 전체 인스턴스."""
        packed = self._npz[f"{frame:05d}"]
        return np.unpackbits(packed, axis=-1, count=self.width).astype(bool)

    def track_mask(self, track_id: int, frame: int):
        """(H, W) bool. 그 프레임에 없으면 all-False."""
        track = self.tracks[track_id]
        if frame not in track["frames"]:
            return np.zeros((self.height, self.width), dtype=bool)
        slot = track["slots"][track["frames"].index(frame)]
        return self.frame_masks(frame)[slot]

    def track_volume(self, track_id: int):
        """(F, H, W) bool. **track 당 45 MB** 이므로 필요한 track 에만 쓸 것."""
        volume = np.zeros((self.num_frames, self.height, self.width), dtype=bool)
        track = self.tracks[track_id]
        for f, slot in zip(track["frames"], track["slots"]):
            volume[f] = self.frame_masks(f)[slot]
        return volume


def load_scene(eval_data: str, video: str, vista4d_root: str,
               seg_root: str | None = None, seg_static_root: str | None = None,
               allow_no_seg: bool = False):
    """recon + 동적/정적 seg 를 합쳐 dict 로. 프레임 수·해상도 불일치는 여기서 죽인다.

    `allow_no_seg` 는 **SAM3 를 아예 안 돌린 recon** 을 위한 것이다 (`run_custom_caption.py`
    처럼 카메라 생성만 하고 뱅크를 안 굽는 경로). seg 가 없으면 노드를 못 만들므로 그래프는
    `nodes: []` 가 되고, 쓸 수 있는 건 `scale`/`cameras`/`gravity` 블록뿐이다 — 뱅크 굽기는
    못 한다. 기본값 False 라 기존 호출부는 전부 예전처럼 죽는다.
    """
    vista4d = import_vista4d(vista4d_root)
    recon = vista4d["load_recon_and_seg"](path.join(eval_data, "eval_data", "recon_and_seg", video))
    num_frames, height, width, _ = recon["video"].shape

    for name in ("depths", "dynamic_mask", "static_mask", "sky_mask", "cam_c2w", "intrinsics"):
        assert recon[name].shape[0] == num_frames,\
            f"{name} 프레임 수 불일치: {recon[name].shape[0]} vs video {num_frames}"
    for name in ("depths", "dynamic_mask", "static_mask", "sky_mask"):
        assert recon[name].shape[1:3] == (height, width),\
            f"{name} 해상도 불일치: {recon[name].shape[1:3]} vs video {(height, width)}"

    seg_root = seg_root or path.join(eval_data, "eval_data", "seg_instances")
    seg_static_root = seg_static_root or path.join(eval_data, "eval_data", "seg_instances_static")
    segs = []
    for root, kind in ((seg_root, "dyn"), (seg_static_root, "stat")):
        folder = path.join(root, video)
        if not path.isfile(path.join(folder, "meta.json")) and kind == "dyn":
            # `recon_and_seg_single.py --save_seg_instances` 는 `recon_and_seg/<video>/seg_instances`
            # 에 쓰는데 코퍼스 규약은 `eval_data/seg_instances/<video>` 다. 새 소스를 1편씩
            # 넣을 때마다 symlink 를 걸어야 했고, 안 걸면 "seg_instances 가 없다" 로 죽었다.
            # 원본 위치를 fallback 으로 본다 (symlink 가 있으면 위에서 이미 잡힌다).
            inline = path.join(eval_data, "eval_data", "recon_and_seg", video, "seg_instances")
            if path.isfile(path.join(inline, "meta.json")):
                folder = inline
        if not path.isfile(path.join(folder, "meta.json")):
            continue
        seg = SegInstances(folder, kind)
        assert (seg.num_frames, seg.height, seg.width) == (num_frames, height, width),\
            f"{kind} seg 가 recon 과 안 맞는다: {(seg.num_frames, seg.height, seg.width)}"
        segs.append(seg)
    assert segs or allow_no_seg, f"{video}: seg_instances 가 없다 ({seg_root})"

    recon["K"] = vista4d["intrinsics_to_K"](recon["intrinsics"]).astype(np.float64)
    recon["cam_c2w"] = recon["cam_c2w"].astype(np.float64)
    recon["segs"] = segs
    recon["video_name"] = video
    return recon
