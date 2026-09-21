"""latentcam eval 이 떨군 **생성 카메라**(`test/<name>_transforms_pred.json`)를 depth warp 로 렌더한다.

왜 새 스크립트인가: `render_bank_videos.py` 는 뱅크(`poses.npz`), `render_target_poses_depth.py` 는
학습 target(`target_poses.npz`)을 읽는다 — 모델이 **생성한** 궤적을 읽는 코드는 없었다. eval JSON 은
scene world 의 **OpenGL c2w** 라 (`render_geo_preds.py:load_pred` 와 같은 규약, GT 대조 실측
max|diff| 2.4e-7) `diag(1,-1,-1,1)` 을 곱해 OpenCV 로 되돌린 뒤 기존 `CloudRenderer` 로 그대로
렌더한다. 렌더 조립(`render_variant`/`label_tile`/`contact_sheet`/`write_video`)은 전부 재사용.

한 entry 당 열 구성: SOURCE | GT(ref) | arm 별 pred. 각 pred 타일 캡션은 그 arm 이 실제로 조건으로
받은 텍스트(`test/<name>_caption.json` 의 값)다 — arm 마다 프롬프트 문체가 다른 비교(k6 vs simple)가
목적이므로 라벨에 박아야 영상만 보고 구분이 된다.

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=3 python scripts/render_pred_depth_warp.py --video camel \
        --eval_dir k6=/.../eval_my/camel8_k6_last --eval_dir simple=/.../eval_my/camel8_simple_last \
        --out_dir results/20260827_camel8_pred_warp
"""
import json
import re
import sys
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import VISTA4D_ROOT_DEFAULT                                      # noqa: E402
from lbm.overlay import contact_sheet, label_tile                               # noqa: E402
from lbm.presets import row_preset                                              # noqa: E402
from lbm.render import add_cloud_source_args, open_renderer                     # noqa: E402
from scripts.render_bank_videos import render_variant, write_video              # noqa: E402

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])        # OpenGL c2w -> OpenCV c2w (열 1,2 부호)


def load_transforms(json_path: str):
    """eval JSON -> (OpenCV c2w (T,4,4), (w,h,fl_x,fl_y))."""
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)
    c2w = np.array([f["transform_matrix"] for f in data["frames"]], dtype=np.float64) @ GL2CV
    return c2w, (int(data["w"]), int(data["h"]), float(data["fl_x"]), float(data["fl_y"]))


def load_condition_text(eval_dir: str, name: str):
    """arm 이 조건으로 받은 텍스트 (`Concise Interaction` 값). 캡션 한 줄에 들어갈 만큼 자른다."""
    with open(path.join(eval_dir, "test", f"{name}_caption.json"), encoding="utf-8") as file:
        text = next(iter(json.load(file).values()))
    return text if len(text) <= 90 else text[:87] + "..."


def split_target_motion(text: str):
    """`target: X motion: Y` -> (X, Y). 형식이 아니면 (None, 원문)."""
    hit = re.match(r"target:\s*(.*?)\s+motion:\s*(.*)", text.strip())
    return (hit.group(1), hit.group(2)) if hit else (None, text.strip())


def load_entry_meta(corpus_root: str, video: str, name: str, prefix: str = "vista4d"):
    """코퍼스 `da3/prompts.json` 에서 그 entry 의 preset/anchor/hole/tau 를 읽는다.

    WHY: eval 이 떨구는 caption 은 문장뿐이라 어떤 preset 을 조건으로 준 건지 영상만 봐선 못 가른다
    (`pan_left` 과 `truck_left` 의 문장이 다르긴 해도 hole 사다리 단은 문장에 안 들어간다).

    `prefix` 는 코퍼스 하위 폴더 이름이자 entry 이름 접두사다 — 두 코퍼스에서 같은 문자열이라
    (vista4d: `latentcam_da3/vista4d/…` + `vista4d_<video>_<i>`, dynpose: `latentcam_dynpose/dynpose/…`
    + `dynpose_<uuid>_<i>`) 손잡이 하나로 충분하다.
    """
    key = name.rsplit("_", 1)[1]
    with open(path.join(corpus_root, prefix, video, "da3", "prompts.json"), encoding="utf-8") as file:
        entry = json.load(file)[key]
    # 코퍼스에 적힌 preset 이 그 궤적의 정식 이름이 아닐 수 있다 (옛 코퍼스는 `straight_ease` 라
    # 적고 `dolly_in` 을 굽는다 — vista `hole_bank_k6` 실측 50.0%). `aim` 을 같이 내보내기
    # 시작한 뒤의 코퍼스는 여기서 풀 수 있고, 그 전 코퍼스는 `aim` 이 없어 별칭만 풀린다.
    return {"preset": row_preset(entry), "anchor": entry.get("anchor_label", ""),
            "hole_fraction": entry.get("hole_fraction"), "tau_max": entry.get("tau_max"),
            "variant_id": entry.get("variant_id", "")}


def load_scores(scores_csv: str, video: str, name_prefix: str):
    """latentcam eval 의 `preds_scores.csv` -> {entry_idx: captions/fscore}, **이 video 만**.

    WHY: eval 은 per-sample caption F1 을 CSV 로만 떨구고 렌더 쪽은 그걸 모른다. 어떤 궤적에서
    F1 이 0 이 되는지는 숫자로는 안 보이고 warp 를 나란히 봐야 보인다 -- 그래서 점수를 라벨에 박고
    band 별 reel 로 묶는다. `filename` 은 `test/<prefix>_<video>_<idx>_transforms_ref.` 꼴.

    WHY video 필터: 예전 구현은 `split("_")[-3]` 로 **idx 만** 키로 썼다. csv 가 video 하나짜리면
    맞지만, val 전량 csv(100씬 512행)를 주면 같은 idx 가 씬마다 겹쳐 **마지막 씬 값으로 덮어써진
    다** -- 라벨의 f1 이 남의 씬 점수가 되고 band reel 도 같이 틀어진다. 그래서 (video, idx) 로
    가른 뒤 이 video 만 남긴다.
    """
    from csv import DictReader
    head, tail = f"{name_prefix}_", "_transforms_ref."
    out = {}
    with open(scores_csv, encoding="utf-8") as file:
        for row in DictReader(file):
            stem = row["filename"].split("/")[-1]
            if not (stem.startswith(head) and stem.endswith(tail)):
                continue
            name, _, idx = stem[len(head):-len(tail)].rpartition("_")
            if name == video:
                out[int(idx)] = float(row["captions/fscore"])
    return out


def main(args):
    arms = []
    for spec in args.eval_dir:
        label, _, folder = spec.partition("=")
        assert folder, f"--eval_dir 은 LABEL=DIR 꼴이어야 한다: {spec}"
        arms.append((label, folder))
    # arm 이 **실제로 조건으로 받은** 텍스트가 eval 폴더 밖에 있는 경우(GenDoP gdstyle 은 입력이
    # `text_gendop_style/`, eval 폴더의 `_caption.json` 은 대조용으로 복사된 우리 태거 문장이다).
    # 주지 않으면 종전대로 eval 폴더에서 읽는다.
    caption_dirs = {}
    for spec in args.caption_dir or []:
        label, _, folder = spec.partition("=")
        assert folder, f"--caption_dir 은 LABEL=DIR 꼴이어야 한다: {spec}"
        assert label in dict(arms), f"--caption_dir 의 라벨 '{label}' 이 --eval_dir 에 없다"
        caption_dirs[label] = folder

    first_dir = arms[0][1]
    names = sorted(
        (path.basename(p)[:-len("_transforms_pred.json")]
         for p in glob(path.join(first_dir, "test",
                                 f"{args.name_prefix}_{args.video}_*_transforms_pred.json"))),
        key=lambda n: int(n.rsplit("_", 1)[1]))
    assert names, f"{first_dir}/test 에 {args.name_prefix}_{args.video}_* pred 가 없다"
    if args.entries:
        wanted = {int(v) for v in args.entries}
        names = [n for n in names if int(n.rsplit("_", 1)[1]) in wanted]

    # label_mode=target_motion 이면 코퍼스 메타(preset/anchor)를 붙이고 그 순서로 reel 을 만든다.
    meta = ({n: load_entry_meta(args.corpus_root, args.video, n, args.name_prefix) for n in names}
            if args.label_mode == "target_motion" else {})
    if args.order == "target_motion":
        assert meta, "--order target_motion 은 --label_mode target_motion 이 필요하다"
        names.sort(key=lambda n: (meta[n]["anchor"], meta[n]["preset"],
                                  meta[n]["hole_fraction"] or 0.0))

    # caption F1 band. --scores_csv 없으면 전부 빈 dict 라 아래 분기가 통째로 no-op 이다.
    scores = (load_scores(args.scores_csv, args.video, args.name_prefix)
              if args.scores_csv else {})
    if scores and args.order == "fscore":
        names.sort(key=lambda n: -scores.get(int(n.rsplit("_", 1)[1]), -1.0))
    elif args.order == "fscore":
        raise SystemExit("--order fscore 는 --scores_csv 가 필요하다")

    # `fixed_focal=True` 는 이 릴의 규약이다 (아래 K 일치 assert 가 frame0 K 를 전제한다).
    renderer, recon = open_renderer(args, args.cloud_root, want_recon=args.with_source,
                                    fixed_focal=True)
    persist = (not args.allow_no_seg if args.temporal_persistence == "auto"
               else args.temporal_persistence == "on")
    num_dyn = int((renderer.visible.sum(dim=1) == 1).sum())
    print(f"{'cloud':<14}points {renderer.points.shape[0]:,}  dynamic {num_dyn:,}  "
          f"temporal_persistence {persist}", flush=True)
    source = None
    if args.with_source:
        import cv2
        source = recon["video"]

    makedirs(args.out_dir, exist_ok=True)
    reel, index, group_reels, bands = [], {}, {}, {}
    for name in names:
        ref_c2w, (json_w, json_h, fl_x, _) = load_transforms(
            path.join(first_dir, "test", f"{name}_transforms_ref.json"))
        # fixed_focal 렌더러의 K_src(frame0)와 eval JSON 의 K 가 같은 화각인지 — 어긋나면
        # 다른 렌즈로 렌더하는 것이라 hole 이 통째로 달라진다 (render_target_poses_depth 와 동일 검사).
        assert abs(fl_x * renderer.width / json_w - float(renderer.K_src[0][0, 0])) < 1.0, \
            f"eval JSON K 와 cloud frame0 K 가 다르다: {fl_x * renderer.width / json_w} vs {renderer.K_src[0][0, 0]}"

        entry = meta.get(name)
        target, motion = (split_target_motion(load_condition_text(first_dir, name))
                          if entry else (None, ""))
        columns = []                                             # (라벨, 캡션, c2w)
        columns.append(("GT" if not entry else f"GT  {entry['preset']}",
                        "" if not entry else f"target: {target}", ref_c2w))
        for label, folder in arms:
            ref2, _ = load_transforms(path.join(folder, "test", f"{name}_transforms_ref.json"))
            assert np.abs(ref2 - ref_c2w).max() < 1e-5, f"{label} 의 ref 가 다르다 — split 이 어긋났다"
            pred_c2w, _ = load_transforms(path.join(folder, "test", f"{name}_transforms_pred.json"))
            caption = (load_condition_text(caption_dirs.get(label, folder), name) if not entry
                       else (motion if len(motion) <= 90 else motion[:87] + "..."))
            columns.append((f"{label}  {entry['preset']}" if entry else label, caption, pred_c2w))

        rendered, stats = [], {}
        for label, caption, c2w in columns:
            picks, frames = render_variant(renderer, c2w, json_h, json_w, args.stride,
                                           temporal_persistence=persist)
            rendered.append((label, caption, frames))
            hole = float(np.mean([(np.all(f == (255, 0, 255), axis=-1)).mean() for f in frames]))
            stats[label] = round(hole, 4)

        idx = int(name.rsplit("_", 1)[1])
        fscore = scores.get(idx)
        f1_tag = "" if fscore is None else f"  f1 {fscore:.3f}"

        length = min(len(f) for _, _, f in rendered)
        sheet_frames = []
        for f in range(length):
            tiles = []
            if source is not None:
                src_tile = cv2.resize(source[f * max(args.stride, 1)], (json_w, json_h))
                tiles.append(src_tile if not args.labels else
                             label_tile(src_tile,
                                        "SOURCE" + f1_tag,
                                        f"{name}  frame {f * max(args.stride, 1)}" if not entry else
                                        f"#{name.rsplit('_', 1)[1]}  {entry['variant_id']}"
                                        f"  tau {entry['tau_max']}  frame {f * max(args.stride, 1)}"))
            for label, caption, frames in rendered:
                tiles.append(label_tile(frames[f].copy(), label, caption) if args.labels
                             else frames[f])
            sheet_frames.append(contact_sheet(tiles, columns=len(tiles)))

        out_path = path.join(args.out_dir, f"{name}__warp.mp4")
        write_video(out_path, sheet_frames, args.fps)
        index[name] = {"hole_magenta_frac": stats, **(entry or {})}
        if fscore is not None:
            index[name]["caption_fscore"] = round(fscore, 4)
        # WHY 조건부: reel 이 꺼져 있으면 프레임을 쌓지 않는다. 689 entry x 49프레임 x 1.0 MB
        # 시트면 reel + group_reels 만으로 수십 GB 라 --no_reel 인데도 OOM 으로 죽었다.
        if args.reel:
            reel.extend(sheet_frames)
            if entry:                                # anchor 별 reel (target 하나만 몰아보기)
                group_reels.setdefault(f"target_{entry['anchor']}", []).extend(sheet_frames)
        if fscore is not None:                       # caption F1 band 별 reel (high/mid/low 대조)
            band = ("f1_high" if fscore >= args.f1_high else
                    "f1_low" if fscore <= args.f1_low else "f1_mid")
            if args.reel:
                group_reels.setdefault(band, []).extend(sheet_frames)
            bands[band] = bands.get(band, 0) + 1
        print(f"{name:<26}{(entry or {}).get('anchor', ''):<8}{(entry or {}).get('preset', ''):<16}"
              + ("" if fscore is None else f"f1 {fscore:.3f}  ")
              + "  ".join(f"{k} hole {v:.3f}" for k, v in stats.items()), flush=True)

    if args.reel and len(names) > 1:
        write_video(path.join(args.out_dir, "all_entries.mp4"), reel, args.fps)
        for group, frames in sorted(group_reels.items()):
            write_video(path.join(args.out_dir, f"reel_{group}.mp4"), frames, args.fps)
    with open(path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as file:
        json.dump({"video": args.video, "arms": dict(arms), "entries": index,
                   "order": args.order, "label_mode": args.label_mode,
                   "stride": args.stride, "fps": args.fps,
                   "temporal_persistence": persist, "num_dynamic_points": num_dyn,
                   "scores_csv": args.scores_csv,
                   "f1_bands": ({"high_ge": args.f1_high, "low_le": args.f1_low, "counts": bands}
                                if scores else None)}, file, ensure_ascii=False, indent=2)
    print(f"\n{'entries':<14}{len(names)}")
    for band in ("f1_high", "f1_mid", "f1_low"):
        if band in bands:
            print(f"{band:<14}{bands[band]}")
    print(f"{'out':<14}{args.out_dir}")


def build_parser():
    """씬 루프 드라이버(`render_eval_val_warp.py`)가 같은 인자를 재사용할 수 있게 분리."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--eval_dir", action="append", required=True,
                        metavar="LABEL=DIR")                    # 반복 가능, 열 순서 = 지정 순서
    # 캡션만 다른 폴더에서 (LABEL=DIR). arm 이 실제로 받은 텍스트가 eval 폴더에 없을 때.
    parser.add_argument("--caption_dir", action="append", default=None)
    parser.add_argument("--entries", nargs="+", default=None)   # entry 인덱스 몇 개만
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--with_source", action="store_true", default=True)
    parser.add_argument("--no_with_source", dest="with_source", action="store_false")
    parser.add_argument("--reel", action="store_true", default=True)   # 전 entry 이어붙인 1편
    parser.add_argument("--no_reel", dest="reel", action="store_false")
    # 타일 위 라벨(열 이름 + 캡션) 굽기. --no_labels 면 렌더 프레임 원본만 나란히 붙인다 —
    # 열 순서는 그대로 SOURCE | GT | arm... 이라 index.json 으로 되짚을 수 있다.
    parser.add_argument("--labels", action="store_true", default=True)
    parser.add_argument("--no_labels", dest="labels", action="store_false")
    # condition = 기존 동작(arm 이 받은 문장 그대로). target_motion = 코퍼스 preset/anchor 를 붙인다.
    parser.add_argument("--label_mode", choices=["condition", "target_motion"], default="condition")
    parser.add_argument("--order", choices=["index", "target_motion", "fscore"], default="index")
    parser.add_argument("--corpus_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3")
    # 코퍼스 갈아끼우는 손잡이 3종. 기본값은 전부 vista4d 라 기존 호출은 한 톨도 안 다르다.
    # dynpose 는 세 개를 같이 넘긴다 (entry 이름이 `dynpose_<uuid>_<i>`, 점군은 `out_dynpose/`,
    # recon 은 `DynPose-LBM/eval_data/recon_and_seg/`).
    parser.add_argument("--name_prefix", default="vista4d")   # entry 접두사 = 코퍼스 하위 폴더명
    parser.add_argument("--cloud_root", default=path.join(CLOUD_ROOT, "out"))
    add_cloud_source_args(parser)
    parser.add_argument("--eval_data", default="/data1/cympyc1785/data/Vista4D-Eval-Data")
    # SAM3 를 안 돌린 recon 을 그대로 warp 한다 (`run_custom_caption.py` 의 추론 전용 경로).
    # 동적/정적 인스턴스가 없으니 점군은 한 덩어리 rigid cloud 가 된다 — 뱅크 릴에는 쓰지 말 것.
    # 둘 다 기본 off = 기존 릴 비트 동일.
    parser.add_argument("--allow_no_seg", action="store_true", default=False)
    parser.add_argument("--allow_empty_dynamic_mask", action="store_true", default=False)
    # 시간 누적(TP) 대 그-프레임-점만(NTP). 기본 auto = `--allow_no_seg` 일 때만 NTP.
    # WHY auto: dynamic 점이 0개인 점군은 전 점이 전 프레임 visible 이라 TP 가 움직이는 피사체를
    # 49장 겹쳐 그린다 (`render_variant` docstring 실측). 그런 점군에서 TP 는 선택지가 아니라
    # 버그다. seg 가 있는 기존 릴은 auto 가 TP 로 떨어져 비트 동일.
    parser.add_argument("--temporal_persistence", choices=["auto", "on", "off"], default="auto")
    # caption F1 band 분리. 없으면 라벨·reel·index 전부 기존과 동일하게 나온다.
    parser.add_argument("--scores_csv", default=None)        # eval 의 preds_scores.csv
    parser.add_argument("--f1_high", type=float, default=0.8)   # 이 이상 -> reel_f1_high.mp4
    parser.add_argument("--f1_low", type=float, default=0.0)    # 이 이하 -> reel_f1_low.mp4
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
