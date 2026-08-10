"""학습이 실제로 읽는 캡션 필드에서 **모션 정보량**을 pose_source 별로 비교한다.

무엇을 읽는가
-------------
dataset_dl3dv.py:590,633 이 쓰는 필드는 `prompt_camera_with_scene_video.concise` 다
(`prompt_camera` 가 **아니다**). `pose_source: da3` 는 `<scene>/prompts.json` 대신
`<scene>/da3/prompts.json` 을 읽으므로 캡션 자체가 통째로 바뀐다.

왜 필요한가
-----------
DA3 캡션에서 truck/dolly/pedestal 같은 촬영용어가 사라지고 move/yaw/pitch 만 남는다.
"용어가 줄었으니 정보가 줄었다"는 결론은 성급하다 — "move right" 는 truck right 와 같은
축·방향을 지정한다. 그래서 어휘 수가 아니라 **축(axis)·방향(direction) 이 남아 있는지**를 센다:

  axis      : lateral(left/right) / depth(forward/backward, in/out, closer/away) /
              vertical(up/down) / yaw / pitch / roll / zoom
  bare_move : move 계열 토큰이 나왔는데 뒤 3 토큰 안에 방향어가 없는 경우 (= 축 정보 소실)

usage
-----
  python scripts/eval/prompt_motion_stats.py --seg-list <test_seg_list>
out -> stdout + results/compare/prompt_motion_stats/summary.json
"""
import argparse
import json
import os
import os.path as osp
import re
from collections import Counter

DIRS = {
    'lateral': ('left', 'right', 'leftward', 'rightward', 'sideways', 'laterally'),
    'depth': ('forward', 'forwards', 'backward', 'backwards', 'back', 'ahead',
              'closer', 'away', 'toward', 'towards', 'inward', 'outward'),
    'vertical': ('up', 'upward', 'upwards', 'down', 'downward', 'downwards',
                 'higher', 'lower'),
}
MOVE = ('move', 'moves', 'moving', 'moved', 'travel', 'travels', 'traveling',
        'translate', 'translates', 'translating', 'glide', 'glides', 'gliding',
        'drift', 'drifts', 'drifting')
ROT = {'yaw': ('yaw', 'yaws', 'yawing'), 'pitch': ('pitch', 'pitches', 'pitching'),
       'roll': ('roll', 'rolls', 'rolling'), 'pan': ('pan', 'pans', 'panning'),
       'tilt': ('tilt', 'tilts', 'tilting')}
CINE = {'truck': ('truck', 'trucks', 'trucking'), 'dolly': ('dolly', 'dollies', 'dollying'),
        'pedestal': ('pedestal', 'pedestals', 'pedestalling'),
        'crane': ('crane', 'cranes', 'craning'), 'zoom': ('zoom', 'zooms', 'zooming'),
        'orbit': ('orbit', 'orbits', 'orbiting'), 'arc': ('arc', 'arcs', 'arcing')}
DIR_ALL = {w for v in DIRS.values() for w in v}
TOK = re.compile(r"[a-z]+")


def analyse(caps):
    n = len(caps)
    axis = Counter()
    tok_total = 0
    bare, with_dir, movehit = 0, 0, 0
    fam = Counter()
    no_axis_caps = 0
    for c in caps:
        t = TOK.findall(c.lower())
        tok_total += len(t)
        seen = set()
        for i, w in enumerate(t):
            for k, ws in DIRS.items():
                if w in ws:
                    seen.add(k)
            for k, ws in ROT.items():
                if w in ws:
                    seen.add('yaw' if k in ('yaw', 'pan') else
                             ('pitch' if k in ('pitch', 'tilt') else k))
                    fam[k] += 1
            for k, ws in CINE.items():
                if w in ws:
                    fam[k] += 1
                    seen.add({'truck': 'lateral', 'dolly': 'depth', 'pedestal': 'vertical',
                              'crane': 'vertical', 'zoom': 'zoom', 'orbit': 'orbit',
                              'arc': 'orbit'}[k])
            if w in MOVE:
                movehit += 1
                if any(x in DIR_ALL for x in t[i + 1:i + 4]):
                    with_dir += 1
                else:
                    bare += 1
        for k in seen:
            axis[k] += 1
        if not seen:
            no_axis_caps += 1
    return {
        'n_captions': n,
        'avg_tokens': tok_total / max(n, 1),
        'captions_with_axis_frac': 1.0 - no_axis_caps / max(n, 1),
        'axis_caption_frac': {k: v / max(n, 1) for k, v in axis.most_common()},
        'move_tokens': movehit,
        'move_with_direction_frac': with_dir / max(movehit, 1),
        'move_bare': bare,
        'term_counts': dict(fam.most_common()),
    }


def load(seg_list, root, sub):
    """seg list 줄 '<chunk>/<hash>/<seg>' -> <root>/<chunk>/<hash>[/da3]/prompts.json 의 concise."""
    caps = []
    miss = 0
    with open(seg_list) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    cache = {}
    for sid in ids:
        chunk, key = sid.rsplit('/', 1)
        sd = osp.join(root, chunk)
        pj = osp.join(sd, sub, 'prompts.json') if sub else osp.join(sd, 'prompts.json')
        if pj not in cache:
            try:
                cache[pj] = json.load(open(pj))
            except Exception:
                cache[pj] = {}
        seg = cache[pj].get(key)
        if not seg:
            miss += 1
            continue
        pcs = seg.get('prompt_camera_with_scene_video')
        c = pcs.get('concise', '') if isinstance(pcs, dict) else (pcs or '')
        if c:
            caps.append(c)
        else:
            miss += 1
    return caps, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg-list', default='/data1/cympyc1785/data/DL3DV/scenes/'
                                          'latentcam_da3_7k_test_seg_list.txt')
    ap.add_argument('--root', default='/data1/cympyc1785/data/DL3DV/scenes')
    ap.add_argument('--out', default='results/compare/prompt_motion_stats')
    a = ap.parse_args()

    out = {}
    for lab, sub in [('COLMAP', ''), ('DA3', 'da3')]:
        caps, miss = load(a.seg_list, a.root, sub)
        out[lab] = {'prompts_json': f'<scene>/{sub + "/" if sub else ""}prompts.json',
                    'missing': miss, **analyse(caps)}
        print(f'[{lab}] {len(caps)} captions (miss {miss})', flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(osp.join(a.out, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print('\nout ->', osp.join(a.out, 'summary.json'))


if __name__ == '__main__':
    main()
