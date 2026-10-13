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

[R243] `--bbox_seg_root <DynPose eval_data>` 를 주면 같은 키 순서로 `<scene>/da3/subject_bbox.npz` 도 쓴다:
    bbox (V,T,4) float32 [cx, cy, w, h] 원본 영상 정규화 · frame_valid (V,T) bool · keys · anchor_id
    (SAM3 instance 메타의 box_xyxy, node.track_id 와 id 매칭. `--only_bbox` 면 target_track.npz 는 그대로 둔다)

env: 아무거나 (numpy 만 쓴다)

예시:
    python fit/bank/export_target_track.py                       # 전 scene (vista4d)
    python fit/bank/export_target_track.py --videos camel --dry_run
    # dynpose D194 (7,801편) — corpus 하위폴더와 scene_graph 위치가 둘 다 다르다
    python fit/bank/export_target_track.py \
        --dl3dv_root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d194 \
        --corpus dynpose --out_root <CinemaTraj>/out_dynpose
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
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


def node_bbox_2d(node, seg_root, video, num_frames):
    """[R243] 노드 -> 프레임별 원본 영상 2D bbox (T,4) [cx, cy, w, h] (영상 폭·높이로 정규화, 0~1) + (T,) valid.
    SAM3 instance 메타 (`<seg_root>/seg_instances[_static]/<video>/meta.json` 의 frames[t][*].box_xyxy) 에서
    node.track_id (+ merged_from) 와 id 가 같은 박스들의 합집합. 그 프레임에 박스가 없으면 valid 0 · 박스 0 (지어내지 않는다)."""
    sub = "seg_instances" if node["id"].startswith("dyn") else "seg_instances_static"
    mp = path.join(seg_root, sub, video, "meta.json")
    box = np.zeros((num_frames, 4), dtype=np.float32)
    ok = np.zeros(num_frames, dtype=bool)
    if not path.isfile(mp):
        return box, ok
    with open(mp, encoding="utf-8") as file:
        meta = json.load(file)
    W, H = float(meta["width"]), float(meta["height"])
    ids = {node.get("track_id")} | set(node.get("merged_from") or [])
    frames = meta["frames"]
    for t in range(min(num_frames, len(frames))):
        bs = [b["box_xyxy"] for b in frames[t] if b.get("id") in ids]
        if not bs:
            continue
        b = np.asarray(bs, dtype=np.float64)
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        box[t] = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H]
        ok[t] = True
    return box, ok


def node_box_axes_world(node, T_wg, num_frames):
    """[R357] 노드 -> (T,3,3) world 의 OBB 반축 벡터 (열 = 로컬 l·w·h 축 × extent/2). 중력 정렬 OBB 라 G 에서
    R_g(yaw_t) @ diag(extent/2), world 는 T_wg[:3,:3] (= S·R) 를 곱한다 (scene scale 포함). yaw 는 track.yaw (프레임별,
    unwrap+median, 안 보인 프레임은 가장 가까운 관측으로 hold — node_track_world 와 같은 규칙), 없으면 obb.yaw_rad 상수.
    extent 는 노드 하나 (프레임별 median)."""
    ext = np.asarray(node["obb"]["extent"], dtype=np.float64) / 2.0
    yaw = None
    if node.get("track") and node["track"].get("yaw"):
        y = np.asarray(node["track"]["yaw"], dtype=np.float64)
        frames = np.asarray(node["track"].get("frames") or list(range(len(y))), dtype=int)
        if len(y) == num_frames:
            yaw = y
        else:
            yaw = np.array([y[np.abs(frames - f).argmin()] for f in range(num_frames)])
    if yaw is None:
        yaw = np.full(num_frames, float(node["obb"]["yaw_rad"]))
    c, s_ = np.cos(yaw), np.sin(yaw); z = np.zeros_like(c); o = np.ones_like(c)
    R = np.stack([np.stack([c, -s_, z], -1), np.stack([s_, c, z], -1), np.stack([z, z, o], -1)], 1)   # (T,3,3) G
    A_g = R * ext[None, None, :]                                                                      # 열마다 반길이
    return np.einsum("ij,tjk->tik", T_wg[:3, :3], A_g)


def main(args):
    out_root = args.out_root or path.join(CINEMATRAJ_ROOT, "out")
    scene_dirs = sorted(glob(path.join(args.dl3dv_root, args.corpus, "*")))
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
        if T != num_frames:
            # 코퍼스 단위(7,801편)로 돌 때 한 편의 프레임수 불일치로 전량이 죽으면 안 된다.
            # 조용히 넘기지 않고 skip 사유로 남겨 표에 찍는다.
            skipped.append((video, f"T {T} != scene_graph num_frames {num_frames}"))
            continue

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

        if args.box:
            # [R357] 같은 키 순서로 subject 3D OBB 반축 → `<scene>/da3/subject_box.npz` (axes_world (V,T,3,3), valid)
            axes = np.zeros((len(keys), T, 3, 3), dtype=np.float32); bvalid = np.zeros(len(keys), dtype=bool); xcache = {}
            for i, anchor in enumerate(anchors):
                if anchor not in xcache:
                    node = nodes.get(anchor)
                    xcache[anchor] = None if node is None else node_box_axes_world(node, T_wg, num_frames).astype(np.float32)
                if xcache[anchor] is not None:
                    axes[i] = xcache[anchor]; bvalid[i] = True
            if not args.dry_run:
                np.savez(path.join(scene_dir, "da3", "subject_box.npz"), axes_world=axes, valid=bvalid,
                         keys=np.array(keys), anchor_id=np.array(anchors))
            if args.only_box:
                rows.append((video, len(keys), int(bvalid.sum()), sorted(missing)))
                continue
        if args.bbox_seg_root:
            # [R243] 같은 키 순서로 subject 2D bbox (SAM3) 도 내보낸다 → `<scene>/da3/subject_bbox.npz`
            bbox = np.zeros((len(keys), T, 4), dtype=np.float32)
            fvalid = np.zeros((len(keys), T), dtype=bool)
            bcache = {}
            for i, anchor in enumerate(anchors):
                if anchor not in bcache:
                    node = nodes.get(anchor)
                    bcache[anchor] = None if node is None else node_bbox_2d(node, args.bbox_seg_root, video, num_frames)
                if bcache[anchor] is not None:
                    bbox[i], fvalid[i] = bcache[anchor]
            if not args.dry_run:
                np.savez(path.join(scene_dir, "da3", "subject_bbox.npz"), bbox=bbox, frame_valid=fvalid,
                         keys=np.array(keys), anchor_id=np.array(anchors))
            if args.only_bbox:
                rows.append((video, len(keys), int(fvalid.any(1).sum()), sorted(missing)))
                continue
        out_path = path.join(scene_dir, "da3", "target_track.npz")
        if not args.dry_run:
            np.savez(out_path, track_world=track, valid=valid,
                     keys=np.array(keys), anchor_id=np.array(anchors))
        rows.append((video, len(keys), int(valid.sum()), sorted(missing)))

    # 코퍼스 단위(dynpose 7,801편)에서는 편별 행이 표가 아니라 로그 덤프가 된다.
    # --max_rows 를 넘으면 집계만 찍고, 문제 있는 편(valid < variants)만 골라서 보여준다.
    print(f"\n{'video':<40}{'variants':>9}{'valid':>7}  missing anchors")
    print("-" * 88)
    shown = rows if len(rows) <= args.max_rows else [r for r in rows if r[2] < r[1]]
    for video, n, nv, miss in shown[:args.max_rows]:
        print(f"{video:<40}{n:>9}{nv:>7}  {','.join(miss) if miss else '-'}")
    if len(shown) > args.max_rows:
        print(f"... (valid<variants 인 편 {len(shown)}개 중 {args.max_rows}개만 표시)")
    elif len(rows) > args.max_rows and not shown:
        print("(전 편 valid == variants)")

    skip_kinds = {}
    for _video, why in skipped:
        skip_kinds[why.split(" ")[0]] = skip_kinds.get(why.split(" ")[0], 0) + 1
    for video, why in skipped[:args.max_rows]:
        print(f"{video:<40} SKIP  {why}")
    if len(skipped) > args.max_rows:
        print(f"... skip {len(skipped)}건 사유별: {skip_kinds}")

    tot_v = sum(r[1] for r in rows)
    tot_ok = sum(r[2] for r in rows)
    print(f"\nscenes {len(rows)}  skipped {len(skipped)}  "
          f"variants {tot_v}  valid {tot_ok} ({tot_ok / max(tot_v, 1):.1%})  "
          f"dry_run {args.dry_run}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dl3dv_root", default=DL3DV_ROOT_DEFAULT)
    parser.add_argument("--corpus", default="vista4d")    # dl3dv_root 아래 코퍼스 하위폴더 (dynpose 등)
    parser.add_argument("--out_root", default=None)      # CinemaTraj out/ (scene_graph 위치)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--max_rows", type=int, default=60)   # 표에 찍을 최대 행 수
    parser.add_argument("--dry_run", action="store_true")
    # [R243] SAM3 2D bbox 도 내보내기: DynPose eval_data 루트 (seg_instances/ · seg_instances_static/ 가 있는 곳). None = 예전 동작.
    parser.add_argument("--bbox_seg_root", default=None)
    # [R357] subject 3D OBB 반축 (`subject_box.npz`) 도 내보내기. --only_box 면 target_track.npz 는 그대로 둔다.
    parser.add_argument("--box", action="store_true")
    parser.add_argument("--only_box", action="store_true")
    parser.add_argument("--only_bbox", action="store_true")      # target_track.npz 는 건드리지 않고 subject_bbox.npz 만
    main(parser.parse_args())
