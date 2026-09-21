#!/usr/bin/env python
"""임의의 eval 폴더에 CLaTr 지표를 매긴다 (우리 예측 / GenDoP / E.T. 를 같은 눈금으로).

`eval_testset.py` 는 자기가 방금 만든 `--out` 폴더에만 CLaTr 을 돌린다. 베이스라인 예측은
`run_gendop_eval.py --stage evaldir` 이 만든 `eval_dir_*/` 로 따로 떨어지는데, 그 안에도
`test/<entry>_transforms_{pred,ref}.json` + `_caption.json` 이 같은 규약으로 들어 있으므로
`test_valid.txt` 만 채워 주면 **똑같은 두 단계**를 그대로 돌릴 수 있다.

    cd <latentcam>/main
    python -m src.extraction checkpoint_path=<ckpt> data_dir=<dir>   # (cwd evaluate/CLaTr)
    python -m src.eval_only  --pred_path <dir>/preds.npy             # (cwd evaluate/eval)

**게이지 주의.** ckpt 와 standardization 을 바꾸면 숫자가 통째로 움직인다. 기본값은
`eval_testset.py:486` 이 쓰는 것과 같게 뒀다 — ckpt 만 넘기고 `dataset/standardization` 은
**안 건드린다**(CLaTr configs 기본값 그대로). 비교하려는 두 폴더는 반드시 같은 인자로 돌 것.
CLaTr 게이지는 학습 split 을 따라가므로 코퍼스가 다르면 절대값을 나란히 놓지 않는다.

**표본 수 주의.** PRDC(precision/recall/density/coverage)와 FCD 는 집합 지표라 entry 가
한두 개면 NaN 이 나온다. 그때 읽을 수 있는 건 `clatr/clatr_score`(궤적↔텍스트 코사인 ×100)와
caption precision/recall/fscore 뿐이다.

사용:
    python scripts/eval/clatr_score_dir.py \
        ours=<...>/eval_my/d234_s11__last \
        gendop=<...>/results/.../eval_dir_gendop_text_raw_slerp_noscale
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import glob

LATENTCAM = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
MAIN = osp.join(LATENTCAM, "main")
DEFAULT_CKPT = osp.join(LATENTCAM, "checkpoints",
                        "clatr_dynpose_d200_s91_real_epoch149.ckpt")
KEYS = ["clatr/clatr_score", "clatr/pred_ref_cosine", "captions/precision",
        "captions/recall", "captions/fscore", "clatr/fcd", "clatr/precision",
        "clatr/recall", "clatr/density", "clatr/coverage"]


def ensure_split(d):
    """`test_valid.txt` 가 없으면 `test/*_transforms_ref.json` 에서 만든다 (있으면 안 건드림)."""
    p = osp.join(d, "test_valid.txt")
    if osp.isfile(p):
        return len(open(p, encoding="utf-8").read().split())
    names = sorted(osp.basename(f)[: -len("_transforms_ref.json")]
                   for f in glob.glob(osp.join(d, "test", "*_transforms_ref.json")))
    if not names:
        raise SystemExit(f"{d}: test/*_transforms_ref.json 이 없다")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(names) + "\n")
    print(f"[split] {len(names)} entries -> {p}")
    return len(names)


def ensure_caption_feats(d, clip_version, key):
    """`seq/test/<entry>_caption.npy` + `token/...` 를 캡션 json 에서 만든다.

    CLaTr 의 caption modality 는 **미리 인코딩된 CLIP feature** 를 읽는다
    (`caption_dataset.py:65`). `eval_testset.py:436-439` 는 추론 루프 안에서 이걸 같이
    떨구지만, GenDoP/E.T. 처럼 밖에서 만든 eval 폴더에는 없다. 캡션 문장은 `_caption.json`
    에 그대로 있으므로 같은 인코더(`clip_version`, `max_token_length=None`)로 다시 만든다.
    이미 있으면 안 건드린다 — 우리 런의 것을 덮어쓰면 그 폴더 지표가 재현이 안 된다.
    """
    import torch
    sys.path.insert(0, osp.join(MAIN, "evaluate", "CLaTr"))
    from pathlib import Path
    import clip_extraction as CE

    names = open(osp.join(d, "test_valid.txt"), encoding="utf-8").read().split()
    todo = [n for n in names
            if not osp.isfile(osp.join(d, "seq", "test", f"{n}_caption.npy"))]
    if not todo:
        return 0
    caps = []
    for n in todo:
        with open(osp.join(d, "test", f"{n}_caption.json"), encoding="utf-8") as fh:
            caps.append(json.load(fh)[key])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = CE.load_clip_model(clip_version, device=dev)
    seq, tok = CE.encode_text(caps, m, max_token_length=None, device=dev)
    CE.save_feats_custom(seq, todo, Path(osp.join(d, "seq")))
    CE.save_feats_custom(tok, todo, Path(osp.join(d, "token")))
    print(f"[caption] {len(todo)} encoded ({clip_version}) -> {d}/seq,token")
    return len(todo)


def run_clatr(d, ckpt, standardization=None):
    cmds = [(["-m", "src.extraction", f"checkpoint_path={ckpt}", f"data_dir={d}"]
             + ([f"dataset/standardization={standardization}"] if standardization else []),
             "evaluate/CLaTr"),
            (["-m", "src.eval_only", "--pred_path", osp.join(d, "preds.npy")],
             "evaluate/eval")]
    for cmd, wd in cmds:
        r = subprocess.run([sys.executable] + cmd, cwd=osp.join(MAIN, wd),
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            raise SystemExit(f"{cmd[1]} 실패 (rc={r.returncode}) on {d}")


def read_metrics(d):
    """`eval_only` 는 `metrics.json` 을 `val/` 접두사로 쓰고 per-entry 는 csv 에 남긴다."""
    out = {}
    mp = osp.join(d, "metrics.json")
    if osp.isfile(mp):
        for k, v in json.load(open(mp)).items():
            out[k[4:] if k.startswith("val/") else k] = v
    sp = osp.join(d, "preds_scores.csv")
    if osp.isfile(sp):                      # pred_ref_cosine 은 여기에만 있다
        lines = open(sp, encoding="utf-8").read().strip().split("\n")
        if len(lines) >= 2:
            hdr, row = lines[0].split(","), lines[1].split(",")
            for k, v in zip(hdr, row):
                if k in ("clatr/pred_ref_cosine",) and k not in out:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="label=dir (label 생략하면 폴더 이름)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    # CLaTr configs 기본값을 쓰는 게 `eval_testset.py` 와 같은 동작이다. 명시적으로 바꾸고
    # 싶을 때만 준다 (예: dynpose49_d200). 비교군 전체에 같은 값을 줄 것.
    ap.add_argument("--standardization", default=None)
    ap.add_argument("--clip_version", default="ViT-B/32")   # conf/config.yaml:708
    ap.add_argument("--caption_key", default="Concise Interaction")
    ap.add_argument("--force", action="store_true", help="metrics.json 이 있어도 다시 돌린다")
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()

    rows = {}
    for spec in a.dirs:
        label, _, d = spec.partition("=")
        if not d:
            label, d = osp.basename(spec.rstrip("/")), spec
        n = ensure_split(d)
        ensure_caption_feats(d, a.clip_version, a.caption_key)
        if a.force or not osp.isfile(osp.join(d, "metrics.json")):
            print(f"[clatr] {label}  n={n}  {d}", flush=True)
            run_clatr(d, a.ckpt, a.standardization)
        else:
            print(f"[clatr] {label}  n={n}  (metrics.json 재사용 — 다시 돌리려면 --force)")
        rows[label] = dict(read_metrics(d), n=n)

    w = max(len(k) for k in rows) + 2
    print("\n" + "label".ljust(w) + "n".rjust(4)
          + "".join(k.split("/")[-1].rjust(16) for k in KEYS))
    for label, m in rows.items():
        cells = []
        for k in KEYS:
            v = m.get(k)
            cells.append(("-" if v is None else
                          "nan" if isinstance(v, float) and v != v else
                          f"{v:.4f}" if isinstance(v, float) else str(v)).rjust(16))
        print(label.ljust(w) + str(m["n"]).rjust(4) + "".join(cells))
    print(f"\nckpt {a.ckpt}\nstandardization "
          f"{a.standardization or '(CLaTr configs 기본값)'}")
    if a.out_json:
        os.makedirs(osp.dirname(osp.abspath(a.out_json)), exist_ok=True)
        json.dump(rows, open(a.out_json, "w"), indent=2)
        print(f"[out] {a.out_json}")


if __name__ == "__main__":
    main()
