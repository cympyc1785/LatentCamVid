"""코퍼스의 `prompts*.json` 문장을 **eval 폴더 모양의 캡션 디렉토리**로 미러한다.

WHY: `gendop_release_infer.py --text_from_eval_dir <dir>` 와 `run_director_batch.py
--text_dir <dir>` 는 둘 다 `<dir>/test/<prefix>_<scene>_<idx>_caption.json` (키
`Concise Interaction`) 을 읽는다. 우리 모델이 받은 문장은 `eval_testset.py` 가 쓴 eval
폴더에 이미 그 모양으로 있지만, **아직 학습이 안 끝난 캡션 판본**(예: D201-A 의
`prompts_mag.json`)은 eval 폴더가 없다. 베이스라인에 문장만 갈아끼우려고 모델을 기다릴
이유가 없으므로 코퍼스 원본에서 바로 미러한다.

`gendop_style_captions.py` 와는 하는 일이 다르다 — 저쪽은 GT 포즈 태그로 **문장을 새로
짓고**(GenDoP 분포 정합), 여기는 코퍼스에 이미 있는 문장을 **그대로 복사**한다.

코퍼스는 안 건드린다 (읽기 전용).

사용:
    python scripts/corpus_captions_to_eval_dir.py \
        --corpus /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200 \
        --split  .../seg_list_dynpose_s91_test.txt --prefix dynpose \
        --prompts_name prompts_mag.json \
        --out results/20260920_d208_gendop_d200/text_mag
"""
import json
from argparse import ArgumentParser
from os import makedirs, path

TEXT_KEY = "Concise Interaction"      # eval 폴더 caption json 의 유일한 키


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", required=True)              # `<dataset>/<scene>/<idx>` 한 줄씩
    ap.add_argument("--prefix", default="dynpose")         # eval 폴더 파일명 접두사
    ap.add_argument("--prompts_name", default="prompts.json")
    ap.add_argument("--field", default="prompt_camera_with_scene_video")
    ap.add_argument("--key", default="concise")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.split, encoding="utf-8") as file:
        lines = [l.strip() for l in file if l.strip()]

    out_test = path.join(args.out, "test")
    makedirs(out_test, exist_ok=True)

    cache, n, missing = {}, 0, []
    for line in lines:
        dataset, scene, index = line.split("/")
        key = (dataset, scene)
        if key not in cache:
            p = path.join(args.corpus, dataset, scene, "da3", args.prompts_name)
            cache[key] = json.load(open(p, encoding="utf-8")) if path.exists(p) else None
        prompts = cache[key]
        if prompts is None or index not in prompts:
            missing.append(line)
            continue
        text = prompts[index][args.field][args.key]
        with open(path.join(out_test, f"{args.prefix}_{scene}_{index}_caption.json"),
                  "w", encoding="utf-8") as file:
            json.dump({TEXT_KEY: text}, file, ensure_ascii=False, indent=4)
        n += 1

    print(f"{'split':<12}{len(lines)}\n{'written':<12}{n}\n{'missing':<12}{len(missing)}")
    if missing:
        print("  예:", missing[:3])
    print(f"{'out':<12}{out_test}")


if __name__ == "__main__":
    main()
