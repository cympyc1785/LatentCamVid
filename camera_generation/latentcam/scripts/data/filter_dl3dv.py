"""Filter DL3DV 7K scenes, mirroring scenetok's build_dl3dv_meta_row
(video_generation/models/scenetok/src/dataset/dataset_dl3dv.py), and write meta_tmp.csv.
Then compare the valid-scene SET against (current meta.csv[7K] - blacklist.csv).

scenetok validity (per scene): transforms.json exists; image folder exists; num_images>=34;
num_images==len(extrinsics); NOT teleport (finite + no big jumps); every image valid
(h>=480,w>=832 and consecutive frame numbers). Approximations here (for speed over ~320k imgs):
image size read from ONE frame per scene (uniform), frame-number consecutiveness from filenames
(images not opened for corruption). Everything else is exact.

Run: python scripts/filter_dl3dv_7k.py --sub 7K --out /data1/.../DL3DV-10K/meta_tmp.csv
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, csv, json, argparse, re, glob
import numpy as np

ROOT = "/data1/cympyc1785/data/DL3DV/scenes"


def check_teleport(c2w):
    """c2w: (N,4,4). scenetok check_teleport_camera (finite + jump thresholds)."""
    w2c = np.linalg.inv(c2w)
    t_c2w = c2w[:, :3, 3]; t_w2c = w2c[:, :3, 3]
    diff = t_w2c[1:] - t_w2c[:-1]
    mag = np.linalg.norm(diff, axis=1)
    if (np.abs(diff) > 10).any() or (mag > 15).any():
        return True
    if (np.abs(t_c2w) > 50).any():
        return True
    if not np.isfinite(t_c2w).all() or not np.isfinite(t_w2c).all():
        return True
    return False


def frame_num(path):
    m = re.search(r'frame_(\d+)', osp.basename(path))
    return int(m.group(1)) if m else -1


def valid_scene(scene_dir):
    """-> (row dict | None, reason)."""
    tj = osp.join(scene_dir, 'transforms.json')
    if not osp.isfile(tj):
        return None, 'no_transforms'
    img_folder = next((osp.join(scene_dir, f) for f in ['images_4', 'images_8', 'images']
                       if osp.isdir(osp.join(scene_dir, f))), None)
    if img_folder is None:
        return None, 'no_images'
    imgs = sorted(glob.glob(osp.join(img_folder, '*')))
    num_images = len(imgs)
    if num_images < 34:
        return None, 'too_few_images'
    try:
        j = json.load(open(tj))
        fr = sorted(j['frames'], key=lambda f: f['file_path'])
        c2w = np.array([f['transform_matrix'] for f in fr], float)
    except Exception:
        return None, 'parse_fail'
    if num_images != len(c2w):
        return None, 'len_mismatch'
    if check_teleport(c2w):
        return None, 'teleport'
    # image validity: size (one frame, assume uniform) + consecutive frame numbers
    try:
        from PIL import Image
        with Image.open(imgs[0]) as im:
            w, h = im.size
    except Exception:
        return None, 'image_open_fail'
    if h < 480 or w < 832:
        return None, 'too_small'
    prev = 0
    for p in imgs:
        fn = frame_num(p)
        if fn - prev > 1:
            return None, 'frame_gap'
        prev = fn
    # height/width for meta = image size (scenetok stores the image H,W)
    return {'chunk': osp.relpath(scene_dir, ROOT), 'height': h, 'width': w,
            'num_images': num_images}, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subs', default='1K,2K,3K,4K,5K,6K,7K', help='comma-separated subsets')
    ap.add_argument('--require-prompts', action='store_true',
                    help='also require prompts.json (matches the latentcam meta.csv intent)')
    ap.add_argument('--out', default=osp.join(ROOT, 'meta_tmp.csv'))
    args = ap.parse_args()
    subs = [s.strip() for s in args.subs.split(',') if s.strip()]

    rows, reasons, tele = [], {}, []
    for sub in subs:
        scene_dirs = [d for d in sorted(glob.glob(osp.join(ROOT, sub, '*'))) if osp.isdir(d)]
        for d in scene_dirs:
            row, why = valid_scene(d)
            if row and args.require_prompts and not osp.isfile(osp.join(d, 'prompts.json')):
                row, why = None, 'no_prompts'
            reasons[why] = reasons.get(why, 0) + 1
            if why == 'teleport':
                tele.append(osp.relpath(d, ROOT))
            if row:
                rows.append(row)
    with open(args.out, 'w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=['chunk', 'height', 'width', 'num_images'])
        wtr.writeheader(); wtr.writerows(rows)
    print(f"[filter] subs={subs}: {sum(reasons.values())} dirs -> {len(rows)} valid  (reasons: {reasons})")
    print(f"[filter] teleport-rejected: {len(tele)}  e.g. {tele[:5]}")
    print(f"[filter] wrote {args.out}")

    # --- compare vs current meta.csv[subs] - blacklist.csv ---
    valid_set = {r['chunk'].split('/')[-1] for r in rows}
    meta = {r['chunk'].split('/')[-1] for r in csv.DictReader(open(osp.join(ROOT, 'meta.csv')))
            if r['chunk'].split('/')[0] in subs}
    bl_path = osp.join(ROOT, 'blacklist.csv')
    bl = {row['scene'].split('/')[-1] for row in csv.DictReader(open(bl_path)) if row.get('scene')} if osp.isfile(bl_path) else set()
    meta_minus_bl = meta - bl
    print(f"\n=== compare (subs={','.join(subs)}) ===")
    print(f"meta.csv[subs]               : {len(meta)}")
    print(f"blacklist.csv (all)          : {len(bl)}  (of which in meta[subs]: {len(meta & bl)})")
    print(f"meta.csv[subs] - blacklist   : {len(meta_minus_bl)}")
    print(f"meta_tmp (filter{' +prompts' if args.require_prompts else ''}): {len(valid_set)}")
    only_new = valid_set - meta_minus_bl
    only_old = meta_minus_bl - valid_set
    print(f"\nin meta_tmp NOT in (meta-blacklist): {len(only_new)}  (e.g. {sorted(only_new)[:5]})")
    print(f"in (meta-blacklist) NOT in meta_tmp: {len(only_old)}")
    for s in sorted(only_old):
        # locate which sub it is
        sub = next((r['chunk'].split('/')[0] for r in csv.DictReader(open(osp.join(ROOT, 'meta.csv')))
                    if r['chunk'].split('/')[-1] == s), '?')
        print(f"   - {sub}/{s[:16]}  (reason: {valid_scene(osp.join(ROOT, sub, s))[1]})")
    if not only_new and not only_old:
        print("\n>>> IDENTICAL sets.")


if __name__ == '__main__':
    main()
