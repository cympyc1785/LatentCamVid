"""dynpose recon 의 `dynamic_mask/` 를 SAM3 seg_instances 합집합으로 갈아끼운다.

WHY: vista 배포본의 dynamic_mask 는 SAM3 dynamic keyword 트랙의 합집합인데, dynpose 는 우리가
recon 을 직접 돌리므로 그 단계가 없다. recon 은 `--seg_keywords`(빈 목록) + `--keep_recon_sky`
로 dynamic=0 / sky=DA3 로 만들어 두고, 이 스크립트가 `seg_instances/<video>/masks.npz`
(VLM 명사 -> SAM3 트랙, 전부 dynamic 명사) 의 프레임별 합집합을 png 로 덮어쓴다.
`_all_`(전부 1) 로 두면 scene_graph 의 정적 점이 0 이 되어 죽는다 (2026-08-28 파일럿 실측).

D177 (사용자 지시 2026-09-10): `--demote_static_objects` 를 켜면 **안 움직이는 소품은 합집합에서
뺀다**. VLM 이 "dynamic 명사"로 부른 것 중에 주차된 차·벽 간판·상 위 그릇처럼 실제로는 정지한
것이 많고, 그것들이 dynamic 으로 남으면 `unproject` 가 프레임당 1장씩만 보이는 점으로 올려서
(§cloud) 다시점 누적을 못 한다 — 카메라가 움직이면 그 자리가 hole 이 된다. 강등은 `dynamic_mask`
한 곳에만 걸고 `scene_graph.json` 의 `kind`/`moving` 은 **안 건드린다** (`build_scene_graph.py:156`
의 D128 기각: `moving` 은 anchor split key 라 강등하면 dyn 앵커가 0 인 씬이 생긴다).

env: 아무거나 (numpy + PIL)

예시:
    python fit/ingest/dynpose_dynamic_mask_from_seg.py --eval_data /data1/.../DynPose-LBM \
        --videos 00e9f728-... 015b197d-...
    # 정지 소품 강등 (scene_graph.json 이 먼저 있어야 한다)
    python fit/ingest/dynpose_dynamic_mask_from_seg.py --eval_data /data1/.../DynPose-LBM \
        --demote_static_objects --output_root out_dynpose --dry_run
"""
import json
from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np
from PIL import Image

ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
CATEGORY_DEFAULT = path.join(ROOT, "configs", "noun_category.json")


def load_categories(category_json: str):
    """명사 분류표 → (판정 함수, 표 dict). `configs/noun_category.json` 규약."""
    with open(category_json, encoding="utf-8") as file:
        cfg = json.load(file)
    assert cfg["format"] == "noun_category_v1", f"모르는 포맷: {cfg['format']}"
    living, worn = set(cfg["living_tokens"]), set(cfg["worn_tokens"])
    overrides, default = cfg["overrides"], cfg["default"]

    def category_of(noun: str):
        noun = str(noun).lower().strip()
        if noun in overrides:
            return overrides[noun]
        tokens = set(noun.replace("-", " ").split())
        return ("living" if tokens & living else
                "worn" if tokens & worn else default)

    return category_of, cfg


def node_track_ids(node: dict):
    """노드 하나가 삼킨 SAM3 dyn track id 집합.

    `instances.py:297` 이 키워드 중복을 병합하면서 `merged_from` 에 `"dyn#3"` 꼴로 남긴다.
    대표 `track_id` 만 보면 병합된 형제가 합집합에 그대로 남아 강등이 반쪽이 된다.
    """
    ids = {int(node["track_id"])}
    for tag in node.get("merged_from", []):
        kind, _, num = str(tag).partition("#")
        if kind == "dyn" and num.isdigit():
            ids.add(int(num))
    return ids


def demotion_plan(graph: dict, category_of, args):
    """scene_graph.json → (강등할 track id 집합, 행 목록). 강등 안 하면 빈 집합."""
    drop, rows = set(), []
    for node in graph["nodes"]:
        if node["kind"] != "dyn":
            continue
        cat = category_of(node["label"])
        d_ref = float(node["viewing_distance"]["d_ref"]) or 1.0
        # 각변위로 잰다. `center_drift_u` 는 scene scale 단위라 카메라에서 먼 물체의 화면상
        # 정지를 못 본다 (D174 실측: 같은 0.06u 가 d_ref 1.44 에서는 각변위 0.042).
        drift, plen = float(node["center_drift_u"]) / d_ref, float(node["path_len_u"]) / d_ref
        demotable = cat == "object" or (args.demote_worn and cat == "worn")
        # 순변위와 경로길이를 **둘 다** 본다 — 제자리에서 회전하거나 왕복하는 물체는 순변위가
        # 0 에 가깝지만 실제로는 움직인다 (강등하면 잔상이 굳는다).
        hit = bool(demotable and drift < args.demote_max_drift_ratio
                   and plen < args.demote_max_path_ratio)
        if hit:
            drop |= node_track_ids(node)
        rows.append((node["id"], node["label"], cat, drift, plen, hit))
    return drop, rows


def build_union(z, meta, keep=None):
    """(T,H,W) bool 합집합. `keep` 이 None 이 아니면 그 track id 만 넣는다."""
    frames = sorted(z.files)
    volumes = []
    for f, key in enumerate(frames):
        masks = np.unpackbits(z[key], axis=-1, count=meta["width"]).astype(bool)
        if keep is not None:
            slots = [slot for slot, inst in enumerate(meta["frames"][f])
                     if int(inst["id"]) in keep]
            masks = masks[slots] if slots else masks[:0]
        volumes.append(masks.any(axis=0) if len(masks)
                       else np.zeros((meta["height"], meta["width"]), dtype=bool))
    return np.stack(volumes)


def main(args):
    root = path.join(args.eval_data, "eval_data")
    videos = args.videos or [path.basename(p) for p in sorted(glob(path.join(root, "seg_instances", "*")))]
    category_of = load_categories(args.noun_category)[0] if args.demote_static_objects else None
    out_root = path.join(ROOT, args.output_root)
    rows, no_graph, unknown_tracks = [], [], 0
    for video in videos:
        seg_path = path.join(root, "seg_instances", video, "masks.npz")
        mask_dir = path.join(root, "recon_and_seg", video, "dynamic_mask")
        if not (path.isfile(seg_path) and path.isdir(mask_dir)):
            print(f"[skip] {video}: seg 또는 recon 없음")
            continue
        z = np.load(seg_path)
        with open(path.join(root, "seg_instances", video, "meta.json"), encoding="utf-8") as file:
            meta = json.load(file)
        # 포맷: 프레임당 키 "%05d" = (n_instances, H, W/8) uint8 **packbits**
        # (seg_sam3_utils.py:98 packbits / :114 unpackbits 규약 그대로).
        union = build_union(z, meta)                                       # (T,H,W) 기존 동작
        demoted, plan = set(), []
        if args.demote_static_objects:
            graph_path = path.join(out_root, video, "scene_graph.json")
            if not path.isfile(graph_path):
                # 그래프가 없으면 운동량을 모른다 → **강등하지 않는다** (기존 동작 그대로).
                no_graph.append(video)
            else:
                with open(graph_path, encoding="utf-8") as file:
                    graph = json.load(file)
                demoted, plan = demotion_plan(graph, category_of, args)
                # 그래프에 없는 track (sliver 필터 등으로 떨어진 것) 은 운동량을 모르므로 남긴다.
                graphed = set().union(*[node_track_ids(n) for n in graph["nodes"]
                                        if n["kind"] == "dyn"]) if graph["nodes"] else set()
                unknown_tracks += len({int(i["id"]) for f in meta["frames"] for i in f} - graphed)
        keep = None if not demoted else {int(i["id"]) for f in meta["frames"]
                                         for i in f} - demoted
        new = build_union(z, meta, keep) if keep is not None else union
        # 실제로 몇 픽셀이 뒤집혔나. 강등한 트랙이 살아남은 트랙 마스크 안에 들어 있으면
        # (앉은 사람의 셔츠처럼) 합집합은 그대로다 — 노드 수만 세면 그걸 못 본다.
        flipped = float((union & ~new).mean()) if keep is not None else 0.0
        flip_frac = flipped / max(float(union.mean()), 1e-9)

        pngs = sorted(glob(path.join(mask_dir, "*.png")))
        assert len(pngs) == new.shape[0], f"{video}: 프레임 수 불일치 {len(pngs)} vs {new.shape[0]}"
        if not args.dry_run:
            for f, png in enumerate(pngs):
                frame = new[f]
                if frame.shape != np.array(Image.open(png)).shape:
                    frame = np.array(Image.fromarray(frame.astype(np.uint8) * 255)
                                     .resize(Image.open(png).size, Image.NEAREST)) > 0
                Image.fromarray((frame * 255).astype(np.uint8)).save(png)
            # D176. **완료 마커**. png 만 보면 "이 단계가 안 돌아서 placeholder(전부 0)" 와 "돌았는데
            # SAM3 가 아무것도 못 찾음" 이 구분되지 않는다 (`dynpose_ingest.py:177` 이 all-zero
            # placeholder 를 쓴다). 그 구분이 안 되면 cloud 를 조용히 전부-정적으로 굽게 된다 —
            # 실제로 dynpose 표본의 39% 가 그렇게 구워졌다 (`lbm/cloud.py` §empty_dynmask).
            # `*.png` 만 읽는 `load_masks` 에는 영향이 없다.
            with open(path.join(mask_dir, "from_seg.json"), "w", encoding="utf-8") as file:
                json.dump({"format": "dynmask_from_seg_v1", "video": video,
                           "num_frames": int(new.shape[0]), "mean": float(new.mean()),
                           "seg_mtime": path.getmtime(seg_path),
                           # D177. 강등을 켜고 구웠는지 + 무엇을 뺐는지. 마스크만 보면
                           # "SAM3 가 못 찾음" 과 "강등해서 뺌" 이 구분되지 않는다.
                           "demote_static_objects": bool(args.demote_static_objects),
                           "demote_max_drift_ratio": float(args.demote_max_drift_ratio),
                           "demote_max_path_ratio": float(args.demote_max_path_ratio),
                           "demote_worn": bool(args.demote_worn),
                           "demoted_track_ids": sorted(demoted),
                           "demoted_nodes": [{"id": r[0], "label": r[1], "category": r[2],
                                              "drift_ratio": round(r[3], 4),
                                              "path_ratio": round(r[4], 4)}
                                             for r in plan if r[5]],
                           "mean_before_demote": float(union.mean()),
                           "flipped_pixel_frac": round(flip_frac, 4)},
                          file, ensure_ascii=False, indent=2)
        rows.append((video, len(meta["frames"]), float(union.mean()), float(new.mean()),
                     len([r for r in plan if r[5]]), len(demoted), flip_frac,
                     ", ".join(f"{r[1]}({r[3]:.3f})" for r in plan if r[5])[:56]))

    header = (f"{'video':<40}{'T':>4}{'mean 전':>9}{'mean 후':>9}"
              f"{'강등노드':>9}{'track':>7}{'픽셀뒤집힘':>11}  강등 명사(각변위)")
    print(header)
    print("-" * min(len(header) + 40, 160))
    for r in rows:
        print(f"{r[0]:<40}{r[1]:>4}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>7}{r[5]:>7}"
              f"{100 * r[6]:>10.1f}%  {r[7]}")
    n_demoted = sum(r[4] for r in rows)
    print(f"\n영상 {len(rows)}   강등 노드 {n_demoted}   "
          f"강등 있는 영상 {sum(1 for r in rows if r[4])}   "
          f"픽셀 뒤집힘 평균 {100 * np.mean([r[6] for r in rows]) if rows else 0:.2f}%   "
          f"강등 {'켬' if args.demote_static_objects else '끔'}"
          f"{f' (drift<{args.demote_max_drift_ratio} & path<{args.demote_max_path_ratio}'
             f', worn {"포함" if args.demote_worn else "보호"})' if args.demote_static_objects else ''}"
          f"   {'(dry run — 안 씀)' if args.dry_run else ''}")
    if no_graph:
        print(f"⚠ scene_graph.json 이 없어 강등을 건너뛴 {len(no_graph)}편: {no_graph[:8]} "
              f"(build_scene_graph.py 를 먼저 돌릴 것)")
    if unknown_tracks:
        print(f"⚠ 그래프에 없는 SAM3 track {unknown_tracks}개 — 운동량을 모르므로 dynamic 으로 남겼다")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    # D177. **기본 꺼짐** = 예전 동작(전 트랙 합집합) 그대로. 켜면 `scene_graph.json` 의
    # 운동량으로 정지 소품을 합집합에서 뺀다.
    parser.add_argument("--demote_static_objects", action="store_true", default=False)
    parser.add_argument("--no_demote_static_objects", dest="demote_static_objects",
                        action="store_false")
    parser.add_argument("--noun_category", default=CATEGORY_DEFAULT, type=str)
    parser.add_argument("--output_root", default="out_dynpose", type=str)  # scene_graph.json 위치
    # 각변위 임계 (`center_drift_u / d_ref`). d157 계열 631편 실측 분포: object p25 0.052 /
    # median 0.106, living p25 0.132, worn p25 0.127. 0.05 에서 object 의 14.9% 가 걸린다.
    parser.add_argument("--demote_max_drift_ratio", default=0.05, type=float)
    # 경로길이 임계 (`path_len_u / d_ref`). 순변위 임계의 3배로 둔다 — 왕복·제자리 회전 보호.
    parser.add_argument("--demote_max_path_ratio", default=0.15, type=float)
    # 착용·소지물(shirt/hat/backpack…)까지 강등 대상에 넣는다. **기본 보호** — 앉아 있는 사람의
    # 셔츠는 순변위가 0 에 가깝다. 다만 옷걸이에 걸린 셔츠는 강등이 맞아서 손잡이를 남긴다.
    parser.add_argument("--demote_worn", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    main(parser.parse_args())
