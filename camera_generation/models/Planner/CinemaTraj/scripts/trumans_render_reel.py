"""`out/trumans_recon/<recording>_<tag>/render_a<NN>/rgb/*.png` → 재생 가능한 mp4.

왜 별도 스크립트인가: `trumans_gt_render.py` 는 PNG 시퀀스만 떨군다 (mp4 를 안 만든다).
`audit_lite_framing.py` 의 `framing_preview` 는 mp4 를 만들지만 **OBB hull·가림 오버레이가
박힌 감사용 영상**이라 "렌더 자체가 어떻게 생겼나"를 볼 때는 오히려 방해가 되고,
`recon_and_seg` / `probe_a*.json` / `poses_a*.npz` 가 전부 있어야 돌아간다. 여기서는
렌더 폴더 하나만 있으면 되게 한다 — 뱅크가 도는 중에도 끝난 것부터 볼 수 있다.

`--overlay` 를 주면 좌상단에 `<recording8>/aNN  fNN  z=<사람 median depth>` 를 찍는다.
사람 픽셀은 `index` 패스(=CYCLES)의 `objects.json:human_indices` 로 고른다. 사람이 반투명해
보이는지 볼 때는 이 z 가 근거가 된다 — 사람이 정말 앞에 있는데 rgb 만 비쳐 보이는 것인지
(EEVEE 재질 문제) 애초에 사람이 그 픽셀에 없는 것인지(카메라/조준 문제)를 가른다.

`--grid` 면 고른 렌더들을 한 화면 타일 애니메이션 하나로 잇는다 (개별 mp4 대신).

영상은 `imageio` + `libx264` (cv2 의 mp4v 는 VS Code 뷰어에서 안 열린다 — 파일은 멀쩡해서
조용히 지나간다).

env: 아무거나 (numpy + imageio 만 쓴다). 예: `/data1/cympyc1785/miniconda3/envs/vista4d/bin/python`

예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY scripts/trumans_render_reel.py --tag s3f0k6 --limit 6 --out results/20260827_tru_k6
    $PY scripts/trumans_render_reel.py --tag s3f0k6 \
        --renders 00add26c:a18 0adb88db:a10 --overlay --out /tmp/reel
"""
import json
import sys
from argparse import ArgumentParser
from glob import glob
from os import listdir, makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)


def render_dirs(work, tag, renders, limit, stride):
    """`--renders rec8:aNN` 지정분 우선, 없으면 `--tag` 전량에서 균등 추출."""
    found = sorted(glob(path.join(work, f"*_{tag}", "render_a*")))
    #    49장 다 안 찬 폴더는 렌더 중이거나 죽은 것이다 — 영상으로 만들면 길이가 어긋난다.
    found = [d for d in found if len(glob(path.join(d, "rgb", "*.png"))) >= 49]
    if renders:
        want = {tuple(r.split(":")) for r in renders}
        return [d for d in found
                if (path.basename(path.dirname(d))[:8], path.basename(d)[-3:]) in want]
    if stride > 1:
        found = found[::stride]
    return found[:limit] if limit > 0 else found


def human_depth(render_dir, k):
    """k 번째 프레임의 사람 픽셀 median depth (m). 사람이 없으면 nan."""
    objects = path.join(render_dir, "objects.json")
    if not path.exists(objects):
        return float("nan")
    with open(objects, encoding="utf-8") as file:
        human = set(json.load(file).get("human_indices", []))
    if not human:
        return float("nan")
    #    D105. `--passes rgb` 로만 렌더한 폴더는 `index`/`depth` 가 없다 (`objects.json` 은
    #    그래도 쓰인다). 예전엔 여기서 FileNotFoundError 로 죽어서 rgb 만 있는 렌더는
    #    `--overlay` 를 못 붙였다. 두 패스가 다 있으면 예전과 비트 동일.
    if not (path.isdir(path.join(render_dir, "index")) and path.isdir(path.join(render_dir, "depth"))):
        return float("nan")
    names = sorted(listdir(path.join(render_dir, "index")))
    index = np.load(path.join(render_dir, "index", names[k]))
    mask = np.isin(index, list(human))
    if mask.sum() < 50:
        return float("nan")
    depth = np.load(path.join(render_dir, "depth", names[k]))
    return float(np.median(depth[mask]))


def load_clip(render_dir, overlay, label=None):
    import imageio.v2 as iio

    names = sorted(listdir(path.join(render_dir, "rgb")))
    #    D105. `label` 을 주면 그대로 쓴다. 기본 규칙(`rec8/aNN`)은 `render_a<NN>` 레이아웃
    #    전용이라 `<out>/<preset>/rgb` 같은 폴더에서는 basename 뒤 3글자를 잘라 엉뚱한 이름이
    #    나온다. 비면 예전과 비트 동일.
    if label is None:
        label = path.basename(path.dirname(render_dir))[:8] + "/" + path.basename(render_dir)[-3:]
    frames = []
    for k, name in enumerate(names):
        image = iio.imread(path.join(render_dir, "rgb", name))
        if overlay:
            import cv2
            z = human_depth(render_dir, k)
            text = f"{label}  f{k:02d}  z={z:.2f}m" if np.isfinite(z) else f"{label}  f{k:02d}  z=-"
            cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2, cv2.LINE_AA)
        frames.append(image)
    return label, np.stack(frames)


def write_video(target, frames, fps):
    import imageio.v3 as iio

    #    cv2 mp4v 는 VS Code 뷰어에서 안 열린다.
    iio.imwrite(target, frames, fps=fps, codec="libx264", quality=6, macro_block_size=1)
    return target


def tile(clips, cols, scale):
    """(label, (T,H,W,3)) 목록 → 타일 애니메이션 한 덩어리."""
    step = max(1, int(round(1 / scale)))
    small = [c[1][:, ::step, ::step] for c in clips]
    T = min(len(s) for s in small)
    h, w = small[0].shape[1:3]
    rows = []
    for r in range(0, len(small), cols):
        row = small[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros((T, h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate([s[:T] for s in row], axis=2))
    return np.concatenate(rows, axis=1)


def main():
    parser = ArgumentParser()
    parser.add_argument("--work", default=path.join(CINEMATRAJ_ROOT, "out", "trumans_recon"))
    parser.add_argument("--tag", default="s3f0k6")            # 렌더 폴더 접미사 (실행분 구분)
    parser.add_argument("--renders", nargs="*", default=None)  # rec8:aNN 형태로 콕 집기
    #    D105. `render_a<NN>` 레이아웃 밖의 렌더 폴더를 그대로 받는다 (`bank_to_blender_poses.py`
    #    가 만든 `<out>/<preset>/`). `--work/--tag/--renders/--limit/--stride` 는 전부 무시되고
    #    타일 라벨은 폴더 이름이 된다. 비면 예전과 비트 동일.
    parser.add_argument("--dirs", nargs="*", default=None)
    parser.add_argument("--name", default="grid.mp4", type=str)   # `--grid` 출력 파일 이름
    parser.add_argument("--limit", default=6, type=int)        # 0 이면 상한 없음
    parser.add_argument("--stride", default=1, type=int)       # 전량에서 균등 솎기
    parser.add_argument("--fps", default=12.5, type=float)     # stride 3 렌더의 의도 속도
    parser.add_argument("--overlay", action="store_true")      # 라벨 + 사람 depth 찍기
    parser.add_argument("--no_overlay", dest="overlay", action="store_false")
    parser.set_defaults(overlay=False)
    parser.add_argument("--grid", action="store_true")         # 개별 mp4 대신 타일 1개
    parser.add_argument("--cols", default=3, type=int)
    parser.add_argument("--scale", default=0.5, type=float)    # 타일 축소 비율
    parser.add_argument("--out", default=path.join(CINEMATRAJ_ROOT, "out", "trumans_reel"))
    args = parser.parse_args()

    if args.dirs:
        #    49장 다 안 찬 폴더는 렌더 중이거나 죽은 것이다 (`render_dirs` 와 같은 판정).
        dirs = [d for d in args.dirs if len(glob(path.join(d, "rgb", "*.png"))) >= 49]
        labels = [path.basename(d.rstrip("/")) for d in dirs]
    else:
        dirs = render_dirs(args.work, args.tag, args.renders, args.limit, args.stride)
        labels = [None] * len(dirs)
    if not dirs:
        raise SystemExit(f"렌더 폴더를 못 찾았다: {args.work}/*_{args.tag}/render_a*")
    makedirs(args.out, exist_ok=True)

    clips, written = [], []
    for d, lab in zip(dirs, labels):
        label, frames = load_clip(d, args.overlay, lab)
        clips.append((label, frames))
        if not args.grid:
            written.append(write_video(
                path.join(args.out, label.replace("/", "_") + ".mp4"), frames, args.fps))
    if args.grid:
        written.append(write_video(path.join(args.out, args.name),
                                   tile(clips, args.cols, args.scale), args.fps))

    print(f"{'render':16s} {'T':>3s} {'z_med(m)':>9s} {'z_min(m)':>9s}")
    for (label, frames), d in zip(clips, dirs):
        zs = np.array([human_depth(d, k) for k in range(0, len(frames), 6)])
        zs = zs[np.isfinite(zs)]
        zmed = np.median(zs) if len(zs) else float("nan")
        zmin = zs.min() if len(zs) else float("nan")
        print(f"{label:16s} {len(frames):3d} {zmed:9.2f} {zmin:9.2f}")
    print()
    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    main()
