"""뱅크 변이 → Vista4D eval 카메라 npz (`eval_data/cameras/<video>/<tag>.npz`).

## 왜 `vista4d_prepare.py` 를 안 쓰나

`tools/recammaster/vista4d_prepare.py` 는 **canonical**(rmax=1 로 정규화된 상대 궤적)을 받아
`|t|max = g*S` 로 다시 키운다. 다른 모델과 손잡이 `g` 를 맞추려고 만든 경로다. 그런데 뱅크
카메라는 그 문제가 없다 — **CinemaTraj 의 point cloud 가 Vista4D recon 그 자체**다:

    out/<video>/cloud.npz : meta_cam_c2w / meta_K  ==  recon_and_seg/<video>/cameras.npz
    (snowboard 실측: cam_c2w maxdiff 0.0, fx 1184.992 동일)

즉 `poses.npz['cam_c2w']` 는 이미 **recon world 의 절대 pose, 절대 미터**다. 정규화했다가 다시
키우면 우리가 τ 사다리로 맞춰 놓은 이동량이 `g` 로 덮여 사라진다. 그래서 그냥 그대로 쓴다.
frame0 앵커도 자동으로 성립한다 (뱅크 궤적이 소스 frame0 카메라에서 출발하므로).

## intrinsic zoom

`render_eval.py:85` 가 `intrinsics_tgt` 를 **(N,4) 프레임별**로 받아 `resize_intrinsics` 한다.
그래서 zoom 을 K 에 그대로 구워 넣으면 Vista4D 가 소비한다 (canonical 경로에는 intrinsics
채널이 없어 `zoom_dropped` 가 되던 것과 대비).

    fx[f] = fx_base * focal_scale[f],  fy 도 같이.  cx,cy 는 안 건드린다 — zoom 은 주점을 안 옮긴다

`fx_base` 는 뱅크가 `--fixed_focal` 로 렌더했으면 recon frame0 값 고정, 아니면 recon 프레임별
값이다 (`bank.json['fixed']['fixed_focal']` 을 읽어 자동 결정). DA3 focal 은 프레임마다
1184.99 → 1249.76 (+5.5%) 드리프트하므로, 뱅크 렌더와 Vista4D 렌더의 화각이 어긋나지 않으려면
이 분기가 맞아야 한다.

`cam_c2w_fsff` / `intrinsics_fsff` 도 같은 값으로 같이 넣는다 — `load_cameras`
(`utils/media.py:161`) 가 `--force_same_first_frame` 일 때 그 키를 찾는데, 우리 궤적은 애초에
frame0 이 소스와 같으므로 두 키가 같은 내용이면 된다. 없으면 KeyError 로 죽는다.

출력:
    <eval_data>/eval_data/cameras/<video>/<tag>.npz
    <out_csv>  (있으면) render_eval / inference_eval 용 metadata 행 추가

env: `vista4d` (GPU 안 쓴다)

예시:
    python scripts/bank_to_vista4d_cams.py --video snowboard \
        --variants fixk_track_bank:dyn_0__pedestal_up__tau0.6__steady__lock__b0__fauto__s9__k3 \
        --tag_prefix ct_ --csv /tmp/meta.csv --seed 52106
"""
import csv
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"

# metadata CSV 열 순서 (`results/20260819_vista4d_eval/meta_*.csv` 와 동일해야 한다 —
# `render_eval`/`inference_eval` 이 DictReader 로 읽지만 사람이 diff 할 때 순서가 맞아야 편하다).
CSV_FIELDS = ["name", "video", "camera", "seed", "prompt", "dynamic", "do_sky_seg",
              "source", "video_id"]


def load_bank(video: str, bank_dir: str, out_root: str):
    folder = path.join(out_root, video, bank_dir)
    npz = np.load(path.join(folder, "poses.npz"), allow_pickle=True)
    with open(path.join(folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    ids = [str(v) for v in npz["variant_id"]]
    focal = npz["focal_scale"] if "focal_scale" in npz.files else None
    return npz["cam_c2w"], focal, ids, bank


def target_intrinsics(intr_recon: np.ndarray, focal: np.ndarray | None, fixed_focal: bool):
    """(n,4) fx,fy,cx,cy. 뱅크 렌더가 쓴 것과 **같은** K 여야 한다."""
    n = len(intr_recon) if focal is None else len(focal)
    base = np.repeat(intr_recon[:1], n, axis=0) if fixed_focal else intr_recon[:n].copy()
    out = np.asarray(base, dtype=np.float64).copy()
    if focal is not None:
        out[:, 0] *= np.asarray(focal, dtype=np.float64)
        out[:, 1] *= np.asarray(focal, dtype=np.float64)
    return out


def main(args):
    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    recon = path.join(args.eval_data, "eval_data", "recon_and_seg", args.video)
    with np.load(path.join(recon, "cameras.npz")) as data:
        src_c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
        src_intr = np.asarray(data["intrinsics"], dtype=np.float64)

    cam_folder = path.join(args.eval_data, "eval_data", "cameras", args.video)
    makedirs(cam_folder, exist_ok=True)

    banks, rows, table = {}, [], []
    for spec in args.variants:
        bank_dir, _, variant = spec.partition(":")
        if not variant:
            raise SystemExit(f"{spec}: `<bank_dir>:<variant_id>` 형태여야 한다")
        if bank_dir not in banks:
            banks[bank_dir] = load_bank(args.video, bank_dir, out_root)
        cam_all, focal_all, ids, bank = banks[bank_dir]
        if variant not in ids:
            raise SystemExit(f"{variant}: {bank_dir} 에 없다\n  " + "\n  ".join(ids))
        index = ids.index(variant)
        poses = np.asarray(cam_all[index], dtype=np.float64)
        focal = None if focal_all is None else np.asarray(focal_all[index], dtype=np.float64)
        fixed_focal = bool(bank.get("fixed_focal", False))   # bank.json 최상위 키다

        # frame0 **위치**는 소스 카메라와 정확히 같아야 한다 (뱅크 궤적의 앵커 규약).
        # 회전은 아니다 — `aim="look_at"` preset 은 frame0 부터 이미 subject 를 보므로 소스
        # 회전과 다르다 (orbit_left_pedestal_up 실측 6.21도). 위치가 어긋나면 그건 world 가
        # 틀린 것이고, 회전이 어긋나는 건 그냥 그 preset 의 정의다.
        err = float(np.abs(poses[0, :3, 3] - src_c2w[0, :3, 3]).max())
        if err > 1e-6:
            raise SystemExit(f"{variant}: frame0 위치 불일치 {err:.3e} — recon world 가 아니다")
        r0 = poses[0, :3, :3] @ src_c2w[0, :3, :3].T
        frame0_rot = float(np.degrees(np.arccos(np.clip((np.trace(r0) - 1) / 2, -1, 1))))
        if len(poses) != len(src_c2w):
            raise SystemExit(f"{variant}: 프레임 {len(poses)} != recon {len(src_c2w)}")

        intr = target_intrinsics(src_intr, focal, fixed_focal)
        tag = args.tag_prefix + variant
        out = path.join(cam_folder, f"{tag}.npz")
        if path.exists(out) and not args.overwrite:
            raise SystemExit(f"{out}: 이미 있다 (--overwrite 로 덮어쓰기)")
        c2w32, intr32 = poses.astype(np.float32), intr.astype(np.float32)
        np.savez(out, cam_c2w=c2w32, intrinsics=intr32,
                 cam_c2w_fsff=c2w32, intrinsics_fsff=intr32)

        tmax = float(np.linalg.norm(poses[:, :3, 3] - poses[0, :3, 3], axis=1).max())
        rot = poses[:, :3, :3] @ poses[0, :3, :3].T
        rotmax = float(np.degrees(np.arccos(np.clip(
            (np.trace(rot, axis1=1, axis2=2) - 1) / 2, -1, 1))).max())
        table.append((tag, bank_dir, tmax, rotmax, frame0_rot, float(intr[0, 0]),
                      float(intr[-1, 0]), fixed_focal))
        rows.append({"name": f"{args.video}/{tag}", "video": args.video, "camera": tag,
                     "seed": args.seed, "prompt": args.prompt, "dynamic": args.dynamic,
                     "do_sky_seg": "false", "source": "davis", "video_id": ""})

    if args.csv:
        makedirs(path.dirname(path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    print(f"video         {args.video}   recon {recon}")
    print(f"cameras       -> {cam_folder}")
    print(f"{'tag':60s} {'bank':18s} {'|t|max':>8s} {'rotmax':>7s} {'rot@f0':>7s} "
          f"{'fx0':>9s} {'fx48':>9s}  fixedK")
    print("-" * 133)
    for tag, bank_dir, tmax, rotmax, rot0, fx0, fx1, fixed in table:
        print(f"{tag:60s} {bank_dir:18s} {tmax:8.4f} {rotmax:7.2f} {rot0:7.2f} "
              f"{fx0:9.2f} {fx1:9.2f}  {fixed}")
    if args.csv:
        print(f"\nmetadata      {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--variants", nargs="+", required=True,
                        help="`<bank_dir>:<variant_id>` 들. 뱅크가 달라도 섞을 수 있다")
    parser.add_argument("--tag_prefix", default="ct_",
                        help="배포 카메라(back-follow 등)와 이름이 겹치지 않게")
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT)
    parser.add_argument("--output_root", default=None, help="뱅크 루트 (기본 CinemaTraj/out)")
    parser.add_argument("--csv", default=None, help="render_eval/inference_eval 용 metadata 경로")
    parser.add_argument("--seed", default="52106", help="metadata seed 열")
    parser.add_argument("--prompt", default="", help="metadata prompt 열")
    parser.add_argument("--dynamic", default="", help="metadata dynamic 열 (SAM3 키워드)")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--no_overwrite", dest="overwrite", action="store_false")
    main(parser.parse_args())
