"""Replace DL3DV-960's 960x540 images_4 with DL3DV-480's 480x270 images_8 (disk saving).

Both trees share transforms.json byte-for-byte and it always reports the ORIGINAL full
resolution (w=3840,h=2160), so intrinsics/hw are unaffected -- only image detail changes.
The dataset/scripts resolve the image dir through dataset_dl3dv.scene_image_dir
('images_4' -> 'images_8' -> 'images'), so keeping the name 'images_8' is enough.

Per scene dir of DL3DV-480/<chunk>/<scene>:
  1. if DL3DV-960/DL3DV-10K/<chunk>/<scene> exists -> os.rename(images_8) into it (same
     lustre FS, so this is a metadata rename, not a copy)
  2. then rm -rf that 960 scene's images_4
Chunks present only in 960 (8K-11K, no 480 counterpart): images_4 deleted too (user's
explicit choice) -- those scenes are not in meta_worldtraj.csv.

Move-in happens BEFORE the delete for every scene, so no scene is ever image-less.
env: DRY=1 (report only), CHUNKS=1K,2K (default: all)
"""
import os
import os.path as osp
import shutil
import sys
import time

SRC = '/data1/cympyc1785/data/DL3DV/DL3DV-480'
DST = '/data1/cympyc1785/data/DL3DV/DL3DV-960/DL3DV-10K'
DRY = os.environ.get('DRY', '') == '1'
ONLY = [c for c in os.environ.get('CHUNKS', '').split(',') if c]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


src_chunks = sorted(d for d in os.listdir(SRC) if osp.isdir(osp.join(SRC, d)))
dst_chunks = sorted(d for d in os.listdir(DST) if osp.isdir(osp.join(DST, d)))
if ONLY:
    src_chunks = [c for c in src_chunks if c in ONLY]
    dst_chunks = [c for c in dst_chunks if c in ONLY]
log(f"src chunks {src_chunks}")
log(f"dst chunks {dst_chunks}")
log(f"DRY={DRY}")

n_moved = n_nomatch = n_nosrc = n_del4 = n_no4 = 0

# --- pass 1: move images_8 in (only for name-matched scene folders) ---
for chunk in src_chunks:
    if chunk not in dst_chunks:
        log(f"{chunk}: no 960 counterpart chunk -> skip move")
        continue
    scenes = sorted(os.listdir(osp.join(SRC, chunk)))
    c_mv = c_nm = c_ns = 0
    for scene in scenes:
        s8 = osp.join(SRC, chunk, scene, 'images_8')
        dscene = osp.join(DST, chunk, scene)
        if not osp.isdir(dscene):
            c_nm += 1
            continue
        d8 = osp.join(dscene, 'images_8')
        if not osp.isdir(s8):
            c_ns += 1
            continue
        if osp.isdir(d8):          # already migrated (idempotent re-run)
            c_mv += 1
            continue
        if not DRY:
            os.rename(s8, d8)
        c_mv += 1
    n_moved += c_mv; n_nomatch += c_nm; n_nosrc += c_ns
    log(f"{chunk}: moved {c_mv} | no 960 scene {c_nm} | no src images_8 {c_ns} "
        f"(of {len(scenes)})")

log(f"== move done: {n_moved} moved, {n_nomatch} unmatched scenes, {n_nosrc} missing src")

# --- pass 2: delete images_4 everywhere in 960 (incl. 8K-11K) ---
for chunk in dst_chunks:
    scenes = sorted(os.listdir(osp.join(DST, chunk)))
    c_del = c_no = 0
    for scene in scenes:
        d4 = osp.join(DST, chunk, scene, 'images_4')
        if not osp.isdir(d4):
            c_no += 1
            continue
        if not DRY:
            shutil.rmtree(d4)
        c_del += 1
    n_del4 += c_del; n_no4 += c_no
    log(f"{chunk}: deleted images_4 for {c_del} scenes (no images_4: {c_no}) "
        f"(of {len(scenes)})")

log(f"== delete done: {n_del4} images_4 removed, {n_no4} scenes had none")
log("ALL DONE")
