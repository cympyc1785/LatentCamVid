"""`<scene>/da3/target_track.npz` — 변이별 subject(anchor) OBB world 궤적 내보내기.

왜 필요한가: latentcam 의 `target_track_dim>0` arm (V4D-PGT 9) 이 subject 의 3D 위치 궤적을
카메라 diffusion 의 x_t 채널에 concat 한다. 학습 레이아웃(`latentcam_da3`)에는 카메라·캡션만
있고 subject 궤적이 없어서, scene graph 에서 뽑아 target_poses.npz 와 **같은 키 순서**로
얹는다 (dataset_dl3dv._target_track 이 seg key 로 조회).

무엇을 쓰나: `out/<video>/scene_graph.json` 의
    dyn 노드  -> track.center_smooth (T,3)  G frame   (동적 subject 의 프레임별 중심)
    stat 노드 -> obb.center (3,)            G frame   (정적이라 상수 궤적으로 타일)
G -> world 는 `frames.T_wg` (위치는 T_wg 그대로, schema.load_graph 가 왕복 검증한다).
변이의 anchor 는 `variant_id` 접두 (`dyn_0__orbit_left_arc__hole0.1` -> `dyn_0`).
scene_graph 에 없는 anchor 는 valid=False + track 0 으로 남긴다 (조용히 빼면 dataset 쪽
키 조회가 터진다 — 키 집합은 target_poses.npz 와 항상 일치해야 한다).

저장 포맷:
    track_world (V,T,3) float32   DA3 world 좌표 OBB/track center
    valid       (V,)    bool
    keys        (V,)    str       target_poses.npz 의 keys 와 동일 (순서까지)
    anchor_id   (V,)    str

env: 아무거나 (numpy 만 쓴다)

예시:
    python scripts/export_target_track.py                       # 전 scene
    python scripts/export_target_track.py --videos camel --dry_run
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

DL3DV_ROOT_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3"


def node_track_world(node, T_wg, num_frames):
    """노드 -> (T,3) world 궤적. dyn 은 track.center_smooth, stat 은 obb.center 상수."""
    if node.get("track") and node["track"].get("center_smooth"):
        center_g = np.asarray(node["track"]["center_smooth"], dtype=np.float64)   # (T?,3) G
        frames = np.asarray(node["track"]["frames"], dtype=int)
        if center_g.shape[0] != num_frames:
            # track 이 안 보인 프레임은 가장 가까운 관측으로 채운다 (보간이 아니라 hold —
            # conf 낮은 구간을 지어내지 않는다).
            full = np.empty((num_frames, 3), dtype=np.float64)
            for f in range(num_frames):
                full[f] = center_g[np.abs(frames - f).argmin()]
            center_g = full
    else:
        center_g = np.tile(np.asarray(node["obb"]["center"], dtype=np.float64), (num_frames, 1))
    homo = np.concatenate([center_g, np.ones((num_frames, 1))], axis=1)          # (T,4)
    return (homo @ T_wg.T)[:, :3]


def main(args):
    out_root = args.out_root or path.join(CINEMATRAJ_ROOT, "out")
    scene_dirs = sorted(glob(path.join(args.dl3dv_root, "vista4d", "*")))
    if args.videos:
        scene_dirs = [d for d in scene_dirs if path.basename(d) in set(args.videos)]

    rows, skipped = [], []
    for scene_dir in scene_dirs:
        video = path.basename(scene_dir)
        target_npz = path.join(scene_dir, "da3", "target_poses.npz")
        graph_path = path.join(out_root, video, "scene_graph.json")
        if not path.isfile(target_npz):
            skipped.append((video, "target_poses.npz 없음"))
            continue
        if not path.isfile(graph_path):
            skipped.append((video, "scene_graph.json 없음"))
            continue

        with open(graph_path, encoding="utf-8") as file:
            graph = json.load(file)
        T_wg = np.asarray(graph["frames"]["T_wg"], dtype=np.float64)
        num_frames = int(graph["num_frames"])
        nodes = {n["id"]: n for n in graph["nodes"]}

        poses = np.load(target_npz, allow_pickle=False)
        keys = [str(k) for k in poses["keys"].tolist()]
        variant_ids = [str(v) for v in poses["variant_id"].tolist()]
        T = int(poses["extrinsics"].shape[1])
        assert T == num_frames, f"{video}: target T {T} != scene_graph num_frames {num_frames}"

        track = np.zeros((len(keys), T, 3), dtype=np.float32)
        valid = np.zeros(len(keys), dtype=bool)
        anchors, missing = [], set()
        cache = {}
        for i, vid in enumerate(variant_ids):
            anchor = vid.split("__")[0]
            anchors.append(anchor)
            if anchor not in cache:
                node = nodes.get(anchor)
                cache[anchor] = None if node is None else \
                    node_track_world(node, T_wg, num_frames).astype(np.float32)
            if cache[anchor] is None:
                missing.add(anchor)
                continue
            track[i] = cache[anchor]
            valid[i] = True

        out_path = path.join(scene_dir, "da3", "target_track.npz")
        if not args.dry_run:
            np.savez(out_path, track_world=track, valid=valid,
                     keys=np.array(keys), anchor_id=np.array(anchors))
        rows.append((video, len(keys), int(valid.sum()), sorted(missing)))

    print(f"\n{'video':<22}{'variants':>9}{'valid':>7}  missing anchors")
    print("-" * 70)
    for video, n, nv, miss in rows:
        print(f"{video:<22}{n:>9}{nv:>7}  {','.join(miss) if miss else '-'}")
    for video, why in skipped:
        print(f"{video:<22} SKIP  {why}")
    print(f"\nscenes {len(rows)}  skipped {len(skipped)}  dry_run {args.dry_run}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv_root", default=DL3DV_ROOT_DEFAULT)
    parser.add_argument("--out_root", default=None)      # CinemaTraj out/ (scene_graph 위치)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args())
