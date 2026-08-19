"""SAM3 가 낸 instance track 중 같은 물체를 가리키는 중복을 mask IoU 로 병합한다.

Vista4D 의 `recon_and_seg_single.py --save_seg_instances` 가 남긴 `seg_instances/` 를 읽는다.

왜 필요한가: Vista4D 는 keyword 하나당 SAM3 를 **따로** 돌리고 `obj_id_offset` 을 더한다
(`utils/recon_and_seg/seg_sam3_official.py`). 패스끼리 서로를 모르므로 metadata.csv 의
`woman,person,human` 같은 recall 우선 keyword 는 같은 사람을 track 3 개로 만든다. scene graph 에서는
track 하나가 노드 하나라 그대로 두면 그래프가 3 배로 오염된다.

왜 recon 단계가 아니라 여기서 하는가: 병합 임계 tau 를 바꿀 때마다 재-recon 하지 않기 위해서다.
`seg_instances/` 는 병합 없는 원본이고 이 스크립트는 항상 새 폴더에 쓴다.

Uni4D 는 GroundingDINO 한 번의 forward 에서 전 keyword 를 채점하므로 프레임별 box NMS(0.5) 로
중복을 지울 수 있다. SAM3 의 PCS 는 프레임 관통 track 을 keyword 별로 따로 내므로 그 자리가 없고,
프레임별 억제를 걸면 프레임마다 살아남는 track 이 달라져 track 이 조각난다. 그래서 단위를 box 가
아니라 **track** 으로 올린다.

예시:
    python scripts/merge_seg_instances.py \
        --seg_instances /data1/.../recon_and_seg/avocado-slice/seg_instances --dry_run
    python scripts/merge_seg_instances.py \
        --seg_instances /data1/.../recon_and_seg/avocado-slice/seg_instances \
        --iou 0.5 --vis
"""
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from os import path

import numpy as np

VISTA4D_ROOT_DEFAULT = path.join(path.dirname(path.dirname(path.abspath(__file__))), "models", "Vista4D")


def build_tracks(seg_frames):
    """instance 리스트(프레임별) -> track 별 {frame_index: mask} + 상수 메타."""
    masks_of = defaultdict(dict)
    meta_of = {}
    for i, instances in enumerate(seg_frames):
        for instance in instances:
            track_id = int(instance["id"])
            assert i not in masks_of[track_id], f"Track {track_id} appears twice in frame {i}"
            masks_of[track_id][i] = instance["mask"]

            keyword = str(instance["keyword"])
            if track_id in meta_of:
                # obj_id_offset 이 keyword 마다 id 구간을 갈라놓으므로 keyword 는 track 상수여야 한다.
                assert meta_of[track_id]["keyword"] == keyword,\
                    f"Track {track_id} has mixed keywords: {meta_of[track_id]['keyword']} vs {keyword}"
                meta_of[track_id]["scores"].append(float(instance["score"]))
            else:
                meta_of[track_id] = {"keyword": keyword, "scores": [float(instance["score"])]}

    for track_id, meta in meta_of.items():
        meta["mean_score"] = float(np.mean(meta["scores"]))
        meta["num_frames"] = len(masks_of[track_id])
        meta["mean_area"] = float(np.mean([mask.sum() for mask in masks_of[track_id].values()]))
    return dict(masks_of), meta_of


def pairwise_track_iou(masks_of, track_ids, downsample: int = 1):
    """공통 등장 프레임에서만 프레임별 mask IoU 를 재고 평균한다.

    한쪽만 나오는 프레임은 분모에서 뺀다 — 등장 구간이 다르면 (예: 가림으로 track 이 끊긴 경우)
    겹치는 동안의 일치도가 같은 물체인지를 말해주고, 등장 프레임 수 차이는 co_frac 로 따로 본다.
    """
    stats = {}
    for a_pos, track_a in enumerate(track_ids):
        for track_b in track_ids[a_pos + 1:]:
            shared = sorted(set(masks_of[track_a]) & set(masks_of[track_b]))
            if not shared:
                stats[(track_a, track_b)] = {"mean_iou": 0.0, "num_shared": 0, "co_frac": 0.0}
                continue

            ious = []
            for i in shared:
                mask_a = masks_of[track_a][i][::downsample, ::downsample]
                mask_b = masks_of[track_b][i][::downsample, ::downsample]
                union = np.count_nonzero(mask_a | mask_b)
                ious.append(np.count_nonzero(mask_a & mask_b) / union if union else 0.0)

            min_len = min(len(masks_of[track_a]), len(masks_of[track_b]))
            stats[(track_a, track_b)] = {
                "mean_iou": float(np.mean(ious)),
                "num_shared": len(shared),
                "co_frac": len(shared) / min_len,
            }
    return stats


def union_find_groups(track_ids, edges):
    parent = {track_id: track_id for track_id in track_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for track_a, track_b in edges:
        root_a, root_b = find(track_a), find(track_b)
        if root_a != root_b:
            parent[root_b] = root_a

    groups = defaultdict(list)
    for track_id in track_ids:
        groups[find(track_id)].append(track_id)
    return [sorted(members) for _, members in sorted(groups.items())]


def pick_representative(members, meta_of, policy: str, keyword_order):
    if policy == "score":
        return max(members, key=lambda t: meta_of[t]["mean_score"])
    if policy == "area":
        return max(members, key=lambda t: meta_of[t]["mean_area"])
    if policy == "keyword_order":  # metadata.csv 의 keyword 나열 순서를 신뢰
        rank = {keyword: k for k, keyword in enumerate(keyword_order)}
        return min(members, key=lambda t: (rank.get(meta_of[t]["keyword"], len(rank)), -meta_of[t]["mean_score"]))
    raise ValueError(f"Unknown label policy {policy}")


def merge_seg_frames(seg_frames, groups, meta_of, masks_of, policy, keyword_order, height, width):
    """group 당 track 1 개로 접는다. mask 는 union, box 는 그 union 의 tight box."""
    group_of = {}
    representatives = []
    for new_id, members in enumerate(groups):
        representative = pick_representative(members, meta_of, policy, keyword_order)
        representatives.append((new_id, representative, members))
        for track_id in members:
            group_of[track_id] = new_id

    merged_frames = []
    for i in range(len(seg_frames)):
        frame = []
        for new_id, representative, members in representatives:
            present = [track_id for track_id in members if i in masks_of[track_id]]
            if not present:
                continue

            mask = np.zeros((height, width), dtype=np.bool_)
            for track_id in present:
                mask |= masks_of[track_id][i]

            rows, cols = np.nonzero(mask)
            box = (
                [float(cols.min()), float(rows.min()), float(cols.max()) + 1.0, float(rows.max()) + 1.0]
                if len(rows) else [0.0, 0.0, 0.0, 0.0]
            )
            frame.append({
                "id": new_id,
                "keyword": meta_of[representative]["keyword"],
                "score": max(float(np.mean(meta_of[track_id]["scores"])) for track_id in present),
                "box_xyxy": np.asarray(box),
                "mask": mask,
            })
        merged_frames.append(frame)
    return merged_frames, representatives


def main(args):
    sys.path.insert(0, args.vista4d_root)
    from utils.recon_and_seg.seg_sam3_utils import load_seg_instances, save_seg_instances

    seg_frames, meta = load_seg_instances(args.seg_instances)
    height, width = meta["height"], meta["width"]
    masks_of, meta_of = build_tracks(seg_frames)
    track_ids = sorted(masks_of)
    print(f"{len(track_ids)} tracks over {meta['num_frames']} frames  keywords={meta['keywords']}")
    for track_id in track_ids:
        info = meta_of[track_id]
        print(f"  track {track_id:>3}  {info['keyword']:<20} frames={info['num_frames']:>4} "
              f"mean_score={info['mean_score']:.3f}  mean_area={info['mean_area']:.0f}px")

    stats = pairwise_track_iou(masks_of, track_ids, downsample=args.iou_downsample)
    edges = []
    print(f"\npairwise (merge if mean_iou >= {args.iou} and co_frac >= {args.min_co_frac}):")
    for (track_a, track_b), stat in sorted(stats.items(), key=lambda kv: -kv[1]["mean_iou"]):
        merge = stat["mean_iou"] >= args.iou and stat["co_frac"] >= args.min_co_frac
        if merge:
            edges.append((track_a, track_b))
        if stat["mean_iou"] >= args.report_iou or merge:
            print(f"  {track_a:>3}({meta_of[track_a]['keyword']}) x {track_b:>3}({meta_of[track_b]['keyword']})"
                  f"  mean_iou={stat['mean_iou']:.3f}  shared={stat['num_shared']:>4}"
                  f"  co_frac={stat['co_frac']:.2f}  {'MERGE' if merge else ''}")

    groups = union_find_groups(track_ids, edges)
    print(f"\n{len(track_ids)} tracks -> {len(groups)} merged nodes")
    merged_frames, representatives = merge_seg_frames(
        seg_frames, groups, meta_of, masks_of, args.label_policy, meta["keywords"], height, width,
    )
    for new_id, representative, members in representatives:
        aliases = [f"{meta_of[t]['keyword']}#{t}" for t in members if t != representative]
        print(f"  node {new_id:>3}  {meta_of[representative]['keyword']:<20} "
              f"(from #{representative}){'  aliases: ' + ', '.join(aliases) if aliases else ''}")

    if args.dry_run:
        print("\n--dry_run: 아무것도 쓰지 않았다.")
        return

    output_folder = args.output_folder or (args.seg_instances.rstrip("/") + "_merged")
    save_seg_instances(output_folder, merged_frames, meta["keywords"], height, width)

    # 병합 이력을 meta.json 에 덧붙인다 (format 은 v1 그대로라 load_seg_instances 가 그냥 읽는다).
    meta_path = path.join(output_folder, "meta.json")
    with open(meta_path, encoding="utf-8") as file:
        merged_meta = json.load(file)
    merged_meta["merge"] = {
        "source": path.abspath(args.seg_instances),
        "iou": args.iou, "min_co_frac": args.min_co_frac,
        "iou_downsample": args.iou_downsample, "label_policy": args.label_policy,
        "nodes": [
            {
                "id": new_id,
                "keyword": meta_of[representative]["keyword"],
                "representative_track": representative,
                "member_tracks": members,
                "member_keywords": [meta_of[t]["keyword"] for t in members],
            }
            for new_id, representative, members in representatives
        ],
        "pairwise": [
            {"a": a, "b": b, "keyword_a": meta_of[a]["keyword"], "keyword_b": meta_of[b]["keyword"], **stat}
            for (a, b), stat in stats.items()
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(merged_meta, file, ensure_ascii=False, indent=1)
    print(f"\nwrote {output_folder}")

    if args.vis:
        import imageio.v2 as imageio
        from utils.recon_and_seg.seg_sam3_utils import overlay_instances_on_video

        video_path = args.video_path or path.join(path.dirname(args.seg_instances.rstrip("/")), "video.mp4")
        reader = imageio.get_reader(video_path)
        video = np.stack([frame for frame in reader])[:len(merged_frames)]
        reader.close()

        overlay = overlay_instances_on_video(video, merged_frames, alpha=0.5)
        vis_path = path.join(output_folder, "vis_merged.mp4")
        # cv2 mp4v 는 VS Code 뷰어에서 안 열린다 -> imageio + libx264.
        # (pix_fmt 는 imageio 가 이미 yuv420p 로 넣으므로 따로 주지 않는다.)
        imageio.mimwrite(vis_path, overlay, fps=args.fps, codec="libx264", quality=6, macro_block_size=1)
        print(f"wrote {vis_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seg_instances", required=True, type=str)
    parser.add_argument("--output_folder", default=None, type=str)  # None = <seg_instances>_merged
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)

    parser.add_argument("--iou", default=0.5, type=float)  # 공통 프레임 평균 mask IoU 임계
    parser.add_argument("--min_co_frac", default=0.5, type=float)  # 짧은 쪽 대비 공통 등장 프레임 비율
    parser.add_argument("--iou_downsample", default=1, type=int)  # >1 이면 stride 서브샘플로 IoU 계산
    parser.add_argument("--report_iou", default=0.1, type=float)  # 이 값 이상인 쌍은 병합 안 해도 출력
    parser.add_argument("--label_policy", default="score", choices=("score", "area", "keyword_order"))

    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--video_path", default=None, type=str)  # None = <seg_instances>/../video.mp4
    parser.add_argument("--fps", default=24, type=int)

    main(parser.parse_args())
