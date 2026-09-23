"""D145 — VLM 명사(`vlm_nouns.json`) → DynPose-LBM `metadata.csv` 행 추가.

왜 필요한가. `video_generation/scripts/sam3_seg_instances.py:69` 는 `<eval_data>/metadata.csv`
에서 영상별 명사를 읽고 없으면 `assert not missing` 으로 죽는다. 지금 그 csv 는 279행인데
`eval_data/recon_and_seg` 에는 880편이 있다 — 나머지 600편은 저자 라벨이 없다. 그래서
`extract_nouns_vlm.py` 가 뽑은 명사를 **같은 스키마로** csv 에 붙여 준다. 하류(SAM3 →
build_scene_graph → sample_camera_bank) 는 한 줄도 안 고친다.

**`dynamic` 열에는 VLM 의 dynamic 명사만 넣는다. static 은 버린다.** 근거는
`scene_graph/io.py:101` — 인스턴스의 `kind`("dyn"/"stat")는 움직임이 아니라 **어느 디렉토리에서
읽혔는지**로 정해진다. dynpose 에는 `eval_data/seg_instances_static` 이 없으므로 이 csv 를 통해
들어온 명사는 전부 `dyn` = subject 후보가 된다. 여기에 wall/floor 를 넣으면 벽이 subject 로
잡힌다. static 명사를 쓰려면 `fit/ingest/sam3_static_instances.py` 로 별도 트리를 만드는 게 맞다
(그건 별개 과제다). 버리는 static 명사는 사이드카에 그대로 남겨 나중에 쓸 수 있게 한다.

provenance: 기존 279행은 **저자 라벨**, 새 행은 **VLM 추출**이다. csv 스키마(`video,dynamic`)를
건드리면 reader 들이 깨지므로 열을 추가하지 않고 `<csv 옆>/metadata_provenance_d145.json` 에
남긴다. 원본은 항상 타임스탬프 백업을 뜨고 시작한다.

사용 예시:

    python fit/ingest/extend_dynpose_metadata.py --dry_run       # 무엇이 붙는지만 본다
    python fit/ingest/extend_dynpose_metadata.py                  # 실제 기록 (백업 자동)
"""

import csv
import json
from argparse import ArgumentParser
from datetime import datetime
from os import path
from shutil import copyfile

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
CSV_DEFAULT = "/data1/cympyc1785/data/DynPose-LBM/metadata.csv"
NOUNS_DEFAULT = path.join(HERE, "out_dynpose_nouns", "vlm_nouns.json")

FIELDS = ["video", "dynamic"]      # 기존 csv 와 열 이름·순서를 정확히 맞춘다


def load_existing(csv_path: str):
    """(rows, header) — 기존 csv 를 순서 그대로 읽는다."""
    if not path.isfile(csv_path):
        return [], list(FIELDS)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or FIELDS)
        rows = [dict(r) for r in reader]
    assert header == FIELDS, f"csv 스키마가 예상과 다르다: {header} != {FIELDS}"
    return rows, header


def load_records(nouns_path: str, frame_mode: str):
    """`vlm_nouns.json` → {video: record}. 같은 video 가 여러 번이면 마지막 것을 쓴다."""
    with open(nouns_path, encoding="utf-8") as f:
        blob = json.load(f)
    out = {}
    for rec in blob.get("records", []):
        if rec.get("frame_mode") != frame_mode:
            continue
        out[rec["video"]] = rec
    return out


def main():
    ap = ArgumentParser(description="VLM 명사를 DynPose-LBM metadata.csv 에 추가")
    ap.add_argument("--csv", default=CSV_DEFAULT)                 # 대상 metadata.csv
    ap.add_argument("--nouns", default=NOUNS_DEFAULT)             # extract_nouns_vlm.py 출력
    ap.add_argument("--frame_mode", default="multi")              # multi(4프레임) 만 쓴다
    ap.add_argument("--max_nouns", type=int, default=0)           # 0 = 전부. >0 이면 앞에서 자른다
    ap.add_argument("--dry_run", action="store_true")             # 쓰지 않고 표만 출력
    ap.add_argument("--overwrite_existing", action="store_true")  # 이미 있는 video 행도 덮어쓴다
    args = ap.parse_args()

    rows, header = load_existing(args.csv)
    known = {r["video"] for r in rows}
    records = load_records(args.nouns, args.frame_mode)

    added, updated, skipped_empty, provenance = [], [], [], {}
    for video in sorted(records):
        rec = records[video]
        dyn = [str(n).strip() for n in (rec.get("dynamic") or []) if str(n).strip()]
        if args.max_nouns > 0:
            dyn = dyn[:args.max_nouns]
        if not dyn:
            # 명사가 하나도 없으면 SAM3 가 돌 게 없다. 행을 만들면 assert 만 통과하고
            # seg_instances 가 비어 하류에서 "seg_instances 가 없다" 로 죽는다 — 아예 뺀다.
            skipped_empty.append(video)
            continue
        value = ", ".join(dyn)
        provenance[video] = {
            "source": "vlm", "model": rec.get("source"), "frame_mode": rec.get("frame_mode"),
            "frame_indices": rec.get("frame_indices"), "dynamic": dyn,
            "static_dropped": rec.get("static") or [],   # 버리지 않고 남긴다 (static 트리용)
        }
        if video in known:
            if args.overwrite_existing:
                updated.append((video, value))
            continue
        added.append((video, value))

    if args.overwrite_existing and updated:
        patch = dict(updated)
        for r in rows:
            if r["video"] in patch:
                r["dynamic"] = patch[r["video"]]
    rows.extend({"video": v, "dynamic": d} for v, d in added)

    if not args.dry_run and (added or updated):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{args.csv}.bak_{stamp}"
        if path.isfile(args.csv):
            copyfile(args.csv, backup)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        prov_path = path.join(path.dirname(args.csv), "metadata_provenance_d145.json")
        prior = {}
        if path.isfile(prov_path):
            with open(prov_path, encoding="utf-8") as f:
                prior = json.load(f).get("videos", {})
        prior.update(provenance)
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump({"format": "dynpose_metadata_provenance_v1", "updated": stamp,
                       "note": "이 목록에 없는 video 는 저자 라벨(원본 279행)이다.",
                       "videos": prior}, f, ensure_ascii=False, indent=2)
    else:
        backup = "(dry_run)"

    print()
    print(f"  csv                {args.csv}")
    print(f"  nouns              {args.nouns}")
    print(f"  records({args.frame_mode})     {len(records)}")
    print(f"  기존 행             {len(known)}")
    print(f"  추가                {len(added)}")
    print(f"  덮어씀              {len(updated)}")
    print(f"  명사 0개라 제외      {len(skipped_empty)}")
    print(f"  최종 행             {len(rows)}")
    print(f"  backup             {backup}")
    if skipped_empty:
        print(f"  제외 목록          {', '.join(skipped_empty[:8])}"
              f"{' ...' if len(skipped_empty) > 8 else ''}")
    for video, value in added[:5]:
        print(f"    + {video}  {value}")
    print()


if __name__ == "__main__":
    main()
