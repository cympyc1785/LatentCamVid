"""`trumans_to_recon.py --poses_override` 로 렌더한 **LBM arm** clip 들을 뱅크 manifest 로 묶는다.

왜 필요한가: `audit_lite_framing.py` 는 뱅크 manifest(`entries[].{video,recording,action,status}`)
를 입력으로 받아 `<work>/<recording>/probe_a<NN>.json` + `poses_a<NN>.npz` 를 읽는다. Lite 뱅크는
드라이버(`trumans_lite_bank.py`)가 그 manifest 를 같이 뱉지만, LBM arm 은 `trumans_to_recon.py` 를
창마다 직접 부르는 경로라 manifest 가 없다. 그래서 이미 나온 `manifest_a<NN>.json` 들을 모아
**같은 스키마로** 다시 쓴다 — 감사 코드는 한 줄도 안 고친다.

Lite 와 나란히 놓고 읽을 수 있게 `kind`/`preset`/`frames` 를 그대로 실어 나른다. LBM arm 은
`kind == "lbm_render_dump"` 라 어느 팔인지 manifest 만 보고도 갈린다.

예시:
    python scripts/build_lbm_bank_manifest.py \
        --work out/trumans_recon_lbm/00add26c-7a26-4a61-b192-b97aa493b3f3 \
        --out  out/trumans_lbm_bank/bank_manifest.json
"""
import json
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path


def main():
    parser = ArgumentParser()
    # `manifest_a<NN>.json` 이 있는 디렉토리. **recording 단위**다 (감사가 그 층을 기대한다).
    parser.add_argument("--work", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--eval_data", default="/data1/cympyc1785/data/TRUMANS-Lite", type=str)
    parser.add_argument("--note", default="LBM arm (원본 파이프라인 카메라를 Lite 눈금으로 렌더)",
                        type=str)
    args = parser.parse_args()

    recording = path.basename(path.normpath(args.work))
    entries = []
    for manifest_path in sorted(glob(path.join(args.work, "manifest_a*.json"))):
        with open(manifest_path, encoding="utf-8") as file:
            man = json.load(file)
        source = man.get("source_camera", {})
        entries.append({
            "video": man["video"],
            "recording": man.get("recording", recording),
            #    **`action.id` 가 아니라 `action_index`** 다. 파일 접미사(`probe_a<NN>.json`)를
            #    만든 게 이 값이고, 감사는 그 접미사로 파일을 찾는다 (`action.id` 는 1-based).
            "action": int(man["action_index"]),
            "kind": source.get("kind", ""),
            "text": man.get("action", {}).get("text", ""),
            "frames": man.get("frames", []),
            "frame_step": man.get("frame_step", 0),
            "preset": source.get("preset", ""),
            "poses_override": source.get("poses_override", ""),
            "status": "ok",
        })
    assert entries, f"{args.work} 에 manifest_a*.json 이 없다"

    makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump({"format": "trumans_bank_v1", "note": args.note,
                   "eval_data": args.eval_data, "recordings": [recording],
                   "entries": entries}, file, ensure_ascii=False, indent=2)

    print(f"{'work':<12}{args.work}")
    print(f"{'recording':<12}{recording}")
    print(f"{'clips':<12}{len(entries)}")
    for entry in entries:
        print(f"  {entry['video']:<22}a{entry['action']:02d}  {entry['kind']:<18}"
              f"{str(entry['frames']):<14}{entry['text'][:40]}")
    print(f"{'out':<12}{args.out}")


if __name__ == "__main__":
    main()
