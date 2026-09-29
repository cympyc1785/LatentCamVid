"""이 코드가 답하는 질문: Molmo2 text 캐시에 넣는 override 맵(`Track {target_text}.`)은 코퍼스에서 어떻게 만드나?

D191 에서 정한 규칙 (EXPERIMENTS.log D194/D197 절). 원래 빌더 `tmp/d194/build_override.py` 가 지워져서 R117 에 복원했다.
  - 디스크 `prompts.json` 의 concise 캡션은 안 건드린다 (umt5 가 그걸 읽는다). Molmo2 에만
    `cache_molmo2_embeddings.py --text_override_json` 으로 이 맵을 넣는다.
  - `caption_fields.target_text` (카메라가 실제로 따라가는 대상의 지칭구) 가 있으면 `Track {target_text}.`
  - 없으면 (aim=free 인 pan_* 변이 등 — 따라가는 대상이 없다) concise 캡션 그대로 (`--empty_target concise`, 기본).
    `Track {anchor_label}.` 로 메우지 않는 이유: 그 변이에서 카메라가 **안 따라가는** 물체라 "Track X" 와 X 를
    무시하는 궤적을 짝지어 학습시키게 된다. 옛 동작(행을 빼기)은 `--empty_target skip`.
  - 키는 캐시가 쓰는 data_name `<chunk_prefix>_<video>_<seg>` (seg = prompts.json 의 키).

    python scripts/data/build_molmo2_track_override.py --root <corpus> [--seg_prefix dynpose] \
        [--check <기존 override.json>] --out <corpus>/molmo2_cache/molmo2_text_track.json
"""
import json
import os
import os.path as osp
from argparse import ArgumentParser
from collections import Counter


def concise_of(v):
    x = v.get("prompt_camera_with_scene_video")
    if isinstance(x, str):
        try:
            x = json.loads(x.replace("'", '"'))
        except ValueError:
            return x
    return x.get("concise", "") if isinstance(x, dict) else str(x or "")


def target_of(v):
    cf = v.get("caption_fields") or {}
    if isinstance(cf, str):
        try:
            cf = json.loads(cf.replace("'", '"'))
        except ValueError:
            return ""
    return str(cf.get("target_text") or "").strip()


def main():
    ap = ArgumentParser()
    ap.add_argument("--root", required=True, help="코퍼스 루트 (<root>/<seg_prefix>/<video>/da3/prompts.json)")
    ap.add_argument("--seg_prefix", default="dynpose")
    ap.add_argument("--empty_target", default="concise", choices=["concise", "skip"])
    ap.add_argument("--check", default=None, help="기존 override json 과 공통 키를 대조한다 (복원 검증)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = osp.join(a.root, a.seg_prefix)
    out, stat = {}, Counter()
    for video in sorted(os.listdir(base)):
        pp = osp.join(base, video, "da3", "prompts.json")
        if not osp.isfile(pp):
            continue
        for seg, v in json.load(open(pp)).items():
            key = f"{a.seg_prefix}_{video}_{seg}"
            t = target_of(v)
            if t:
                out[key] = f"Track {t}."
                stat["track"] += 1
            elif a.empty_target == "concise":
                out[key] = concise_of(v)
                stat["concise"] += 1
            else:
                stat["skip"] += 1
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=0)
    print(f"[override] {len(out)} 행 ({dict(stat)}), uniq {len(set(out.values()))} -> {a.out}")
    if a.check:
        old = json.load(open(a.check))
        common = set(old) & set(out)
        same = sum(old[k] == out[k] for k in common)
        print(f"[check] 기존 {len(old)} / 새 {len(out)} / 공통 {len(common)} / 텍스트 일치 {same} "
              f"({same / max(len(common), 1) * 100:.2f}%)")
        diff = [k for k in sorted(common) if old[k] != out[k]][:5]
        for k in diff:
            print(f"  diff {k}: old={old[k]!r} new={out[k]!r}")


if __name__ == "__main__":
    main()
