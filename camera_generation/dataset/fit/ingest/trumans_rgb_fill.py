"""이 코드가 답하는 질문: `--geom_only` (R487 1 단계) 로 만든 TRUMANS 클립에 조건 영상 (RGB) 을 나중에 채우려면?

1 단계는 depth/index 만 렌더하고 recon 의 video.mp4 를 회색 자리표시자 + `RGB_PENDING` 으로 남긴다. 뱅크를 통과한 클립만
골라 **저장된 카메라 (manifest.paths.poses) 로 RGB 만** 다시 렌더해 video.mp4 를 바꾸고 RGB_PENDING 을 지운다.
렌더 설정은 1 단계와 같다: Cycles `--anim` (D276) · 16 spp (D281) · GPU · 960x540 · 25 mm (poses npz 에 lens_mm 이 있으면 그 값 —
trumans_gt_render 가 읽는다) · 프레임 = manifest frames/frame_step. depth/index 는 건드리지 않는다 (D279: anim 렌더끼리 프레임 정렬 0).
렌더 png 는 mp4 로 옮긴 뒤 지운다 (`--keep_png` 로 남김) — 클립당 49 파일 · 약 25 MB.

env: vista4d (imageio). 예:
    python fit/ingest/trumans_rgb_fill.py --videos list.txt --gpus 1 1 2 2 --log_dir ../../tmp/r558/logs
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from os import path

HERE = path.dirname(path.abspath(__file__))
sys.path.insert(0, HERE)
import trumans_to_recon as T  # noqa: E402

DS = path.dirname(path.dirname(HERE))


def manifest_of(video, work_root):
    """tru_<rec8>_<chunk>_k<cand>_p<slot>_<tag> → work_root/<recording>/manifest_<chunk>_k<cand>_p<slot>.json"""
    parts = video.split("_")
    rec8, chunk, fr, cand, slot = parts[1], parts[2], parts[3], parts[4], parts[5]
    hits = glob.glob(path.join(work_root, f"{rec8}*", f"manifest_{chunk}_{fr}_{cand}_{slot}.json"))
    return hits[0] if len(hits) == 1 else None


def fill(video, gpu, a):
    m = manifest_of(video, a.work_root)
    if m is None:
        return video, "no_manifest"
    M = json.load(open(m)); recon = M["paths"]["recon"]
    if not path.isfile(path.join(recon, "RGB_PENDING")):
        return video, "done"
    rec_dir = path.join(a.trumans, "Recordings_blend", M["recording"])
    blend = path.join(rec_dir, f"{M['recording']}.blend")
    if not path.isfile(blend):
        found = glob.glob(path.join(rec_dir, "*.blend")); blend = found[0] if len(found) == 1 else blend
    start, end = M["frames"]; step = M["frame_step"]
    out = path.join(a.tmp_root, video)
    shutil.rmtree(out, ignore_errors=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8"}
    cmd = [a.blender, "-b", blend, "--python", path.join(DS, "viz", "trumans_gt_render.py"), "--",
           "--poses", M["paths"]["poses"], "--frames", str(start), str(end), str(step), "--res", "960", "540", "--lens", "25.0",
           "--passes", "rgb", "--rgb_engine", "cycles", "--rgb_samples", str(a.rgb_samples), "--rgb_cdevice", "GPU", "--anim", "--out", out]
    t0 = time.time()
    with open(path.join(a.log_dir, f"{video}.log"), "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=a.timeout)
    txt = open(path.join(a.log_dir, f"{video}.log"), errors="ignore").read()
    pngs = sorted(glob.glob(path.join(out, "rgb", "frame_*.png")))
    if p.returncode != 0 or "Traceback (most recent call last)" in txt or len(pngs) != T.NUM_FRAMES:
        return video, f"render_fail rc={p.returncode} png={len(pngs)}"
    import imageio.v2 as imageio
    import numpy as np
    rgb = [np.asarray(imageio.imread(f))[:, :, :3] for f in pngs]
    tmp_mp4 = path.join(recon, "video.rgbfill.mp4")
    imageio.mimwrite(tmp_mp4, rgb, fps=M["fps"], codec="libx264", quality=8, macro_block_size=1)
    os.replace(tmp_mp4, path.join(recon, "video.mp4"))
    os.remove(path.join(recon, "RGB_PENDING"))
    if not a.keep_png:
        shutil.rmtree(out, ignore_errors=True)
    return video, f"ok {time.time() - t0:.0f}s"


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--videos", required=True)                       # 한 줄에 클립 이름 하나
    ap.add_argument("--gpus", nargs="+", default=["1"])              # worker 마다 GPU (같은 번호를 여러 번 = 카드당 여러 worker)
    ap.add_argument("--work_root", default=path.join(DS, "out", "trumans_recon_g10"))
    ap.add_argument("--tmp_root", default=path.join(DS, "out", "trumans_rgbfill_tmp"))
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--trumans", default=T.TRUMANS_DEFAULT)
    ap.add_argument("--blender", default=T.BLENDER_DEFAULT)
    ap.add_argument("--rgb_samples", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep_png", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.log_dir, exist_ok=True); os.makedirs(a.tmp_root, exist_ok=True)
    vids = [l.strip() for l in open(a.videos) if l.strip()]
    import itertools
    gp = itertools.cycle(a.gpus); jobs = [(v, next(gp)) for v in vids]
    n = ok = 0
    with ThreadPoolExecutor(len(a.gpus)) as ex:
        for v, st in ex.map(lambda j: fill(j[0], j[1], a), jobs):
            n += 1; ok += st.startswith(("ok", "done"))
            print(f"[rgbfill] {n}/{len(vids)} {v} {st}", flush=True)
    print(f"[rgbfill] ALL DONE ok {ok}/{n}", flush=True)


if __name__ == "__main__":
    main()
