"""뱅크를 다 구운 뒤 **눈으로 확인할 표본 씬**을 고른다 — 움직이는 subject N편 + 안 움직이는
subject N편.

왜 이게 필요한가. 뱅크 검증을 숫자(variant 수 / 축 일치 / hole 사다리)로만 하면 **카메라가
실제로 어떻게 도는지**를 못 본다. 그리고 한두 편만 보면 편향된다 — preset 의 실패 모드가
subject 의 이동 여부에 따라 정확히 갈리기 때문이다:

  · **움직이는 subject** (`center_drift_u >= --drift_thresh`): `track_*` 가 켜지고 `aim=look_at`
    이 매 프레임 조준을 갱신한다. 여기서 터지는 건 추종 실패(피사체가 프레임 밖으로) 와
    조준 흔들림이다.
  · **안 움직이는 subject**: `track_*` 가 `--track_min_drift_u` 게이트에서 잘려나가고 순수
    object-centric preset(orbit/arc/dolly/crane)만 남는다. 여기서 터지는 건 hole(소스가 못 본
    면으로 카메라가 돌아감) 과 벽 뚫기다.

즉 두 버킷은 **다른 게이트를 시험**한다. 한쪽만 보면 나머지 절반이 조용히 깨진 채로 학습에
들어간다. 그래서 재굽기 때마다 양쪽에서 같은 수를 뽑아 나란히 본다.

고르는 방식은 **결정론적**이다 (RNG 없음). 재굽기 전후로 같은 표본이 나와야 before/after 를
짝지어 볼 수 있기 때문이다 — drift 는 `build_scene_graph.py` 산출물이라 축을 바꿔도 안 변한다.
1순위 = preset 다양성(해당 anchor 에서 solved 된 preset 수), 2순위 = drift (움직이는 쪽은 큰 순,
안 움직이는 쪽은 작은 순), 3순위 = 이름.

**anchor 선택은 `run_preset_warp_max_shard.sh` 와 같은 규칙**(`dyn_0` 우선, 없으면 solved 행이
가장 많은 anchor)이라야 한다. 여기서 분류한 subject 와 릴에 실제로 찍히는 subject 가 다르면
버킷 자체가 거짓말이 된다.

사용 예시:
    # 이름만 (러너에 그대로 먹인다)
    python scripts/pick_warp_sample_scenes.py --bank_dir hole_bank_k6_d128
    # 근거 표까지
    python scripts/pick_warp_sample_scenes.py --bank_dir hole_bank_k6_d128 --table
    # dynpose 뱅크에서 3+3, JSON 으로 저장
    python scripts/pick_warp_sample_scenes.py --output_root out_dynpose --bank_dir hole_bank_d129 \
        --num_moving 3 --num_static 3 --out /tmp/d129_warp_sample.json
"""
import csv
import json
from argparse import ArgumentParser
from collections import Counter, defaultdict
from os import path
from sys import stderr

#    `sample_camera_bank.py --track_min_drift_u` 의 기본값과 같은 값. 두 곳이 어긋나면
#    "track 이 켜졌는지"로 나눈 버킷이 실제 뱅크와 안 맞는다.
DEFAULT_DRIFT_THRESH = 0.05


def scene_rows(output_root: str, video: str, bank_dir: str):
    """해당 씬의 solved 행 + 릴이 쓸 anchor 를 돌려준다. 뱅크가 없으면 None."""
    csv_path = path.join(output_root, video, bank_dir, "bank.csv")
    if not path.isfile(csv_path):
        return None
    with open(csv_path, encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["status"].startswith("solved")]
    if not rows:
        return None
    counts = Counter(r["anchor_id"] for r in rows)
    #    릴 러너와 **같은 규칙**. 바꾸려면 두 곳을 같이 바꿀 것.
    anchor = "dyn_0" if "dyn_0" in counts else counts.most_common(1)[0][0]
    mine = [r for r in rows if r["anchor_id"] == anchor]
    return anchor, mine


def node_drift(output_root: str, video: str, anchor: str):
    """anchor 노드의 순변위 `center_drift_u`. 그래프가 없거나 노드가 없으면 None."""
    graph_path = path.join(output_root, video, "scene_graph.json")
    if not path.isfile(graph_path):
        return None, None
    with open(graph_path, encoding="utf-8") as file:
        graph = json.load(file)
    for node in graph["nodes"]:
        if node["id"] == anchor:
            return node.get("center_drift_u"), node.get("label")
    return None, None


def collect(output_root: str, bank_dir: str, videos: list):
    """씬마다 (anchor, drift, preset 수, track preset 수) 를 모은다."""
    out = []
    for video in videos:
        got = scene_rows(output_root, video, bank_dir)
        if got is None:
            continue
        anchor, rows = got
        drift, label = node_drift(output_root, video, anchor)
        if drift is None:
            print(f"  skip {video}: {anchor} 노드가 scene_graph 에 없다", file=stderr)
            continue
        presets = {r["preset"] for r in rows}
        out.append({
            "video": video, "anchor": anchor, "label": label,
            "center_drift_u": round(float(drift), 4),
            "n_preset": len(presets), "n_variant": len(rows),
            "n_track_preset": sum(p.startswith("track_") for p in presets),
        })
    return out


def split_buckets(items: list, thresh: float, num_moving: int, num_static: int):
    """움직임 여부로 가르고 각 버킷에서 결정론적으로 앞에서 자른다."""
    moving = [d for d in items if d["center_drift_u"] >= thresh]
    static = [d for d in items if d["center_drift_u"] < thresh]
    #    1순위 preset 다양성(내림), 2순위 drift, 3순위 이름 — 전부 결정론.
    moving.sort(key=lambda d: (-d["n_preset"], -d["center_drift_u"], d["video"]))
    static.sort(key=lambda d: (-d["n_preset"], d["center_drift_u"], d["video"]))
    for d in moving:
        d["bucket"] = "moving"
    for d in static:
        d["bucket"] = "static"
    return moving[:num_moving], static[:num_static], moving, static


def main(args):
    if args.videos == ["all"]:
        root = args.output_root
        videos = sorted(v for v in __import__("os").listdir(root)
                        if path.isdir(path.join(root, v)))
    else:
        videos = args.videos
    items = collect(args.output_root, args.bank_dir, videos)
    if not items:
        raise SystemExit(f"뱅크 {args.bank_dir} 에 solved 행이 있는 씬이 없다 ({args.output_root})")
    pick_m, pick_s, all_m, all_s = split_buckets(
        items, args.drift_thresh, args.num_moving, args.num_static)
    picked = pick_m + pick_s

    if args.table:
        print(f"# bank {args.bank_dir}   root {args.output_root}   "
              f"drift_thresh {args.drift_thresh}", file=stderr)
        print(f"# 후보 {len(items)}편 (moving {len(all_m)} / static {len(all_s)})", file=stderr)
        head = f"{'bucket':<8}{'video':<20}{'anchor':<8}{'label':<16}" \
               f"{'drift_u':>9}{'preset':>8}{'track':>7}{'variant':>9}"
        print(head, file=stderr)
        print("-" * len(head), file=stderr)
        for d in picked:
            print(f"{d['bucket']:<8}{d['video']:<20}{d['anchor']:<8}{str(d['label'])[:15]:<16}"
                  f"{d['center_drift_u']:>9.4f}{d['n_preset']:>8}"
                  f"{d['n_track_preset']:>7}{d['n_variant']:>9}", file=stderr)
        if len(pick_m) < args.num_moving or len(pick_s) < args.num_static:
            print(f"!! 버킷이 모자란다 — moving {len(pick_m)}/{args.num_moving}, "
                  f"static {len(pick_s)}/{args.num_static}", file=stderr)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump({"bank_dir": args.bank_dir, "output_root": args.output_root,
                       "drift_thresh": args.drift_thresh, "picked": picked,
                       "n_candidate": len(items), "n_moving": len(all_m),
                       "n_static": len(all_s)}, file, ensure_ascii=False, indent=2)

    #    stdout 은 **이름만**. 러너가 `VIDEOS=$(...)` 로 그대로 받는다.
    print(" ".join(d["video"] for d in picked))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", default="out")          # out / out_dynpose / out_trumans
    parser.add_argument("--bank_dir", required=True)             # hole 뱅크 폴더 이름
    parser.add_argument("--videos", nargs="+", default=["all"])  # 후보를 좁히고 싶을 때
    parser.add_argument("--num_moving", default=2, type=int)     # 움직이는 subject 표본 수
    parser.add_argument("--num_static", default=2, type=int)     # 안 움직이는 subject 표본 수
    parser.add_argument("--drift_thresh", default=DEFAULT_DRIFT_THRESH, type=float)
    parser.add_argument("--table", action="store_true")          # 근거 표를 stderr 로
    parser.add_argument("--out", default="")                     # 선택 근거 JSON 경로
    main(parser.parse_args())
