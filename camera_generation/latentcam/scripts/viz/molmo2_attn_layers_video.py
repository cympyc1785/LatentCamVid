"""`Track {target}` 한 프롬프트로 **36층 attention map 을 한 화면에 타일**해 영상으로 만든다.

  python scripts/viz/molmo2_attn_layers_video.py --scene vista4d/camel --gpu 2

무엇을 그리나: prefill forward **1회**로 36층 attention 을 전부 잡고(`Runner.maps_all_layers`),
각 층의 `query text -> video patch` 맵을 6x6 타일로 붙여 프레임마다 한 장을 만든다. 층 간
비교를 눈으로 하려면 같은 프레임을 나란히 놔야 해서다 (층별 영상 36개를 따로 보면 못 한다).

query 두 종을 각각 낸다:
  `alltail`  tail 전체 (`<|im_start|>user\\nTrack {t}<|im_end|>\\n<|im_start|>assistant\\n`)
  `target`   그 안의 target 명사구 토큰만

정규화는 **셀마다 (층, 프레임) 자기 최대**다 — 층끼리 스케일이 크게 달라(층 0 은 거의 균일,
층 20 대는 sink 쪽으로 쏠림) 전역 최대로 맞추면 대부분 셀이 검게 죽는다. 그래서 셀 사이의
**밝기는 비교하지 말고 모양만** 볼 것. 층별 절대 질량은 콘솔 표(`video%`)에 있다.

주의: 이건 **prefill** attention 이다. 좌표는 생성 단계에서 나오므로, 생성 중 좌표 토큰 위치의
attention 은 여기 안 들어 있다 (D164 한계).
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import sys

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, colorize, OUT, CORPUS     # noqa: E402
import cache_molmo2_embeddings as C                             # noqa: E402


def corpus_target(root, scene, preset):
    """씬의 `preset` 변이에서 `target_text` 원문. 관계절 유지 (D164: 자르면 대상이 바뀐다)."""
    pj = json.load(open(osp.join(root, scene, 'da3', 'prompts.json')))
    keys = sorted(pj, key=lambda x: int(x))
    k = next((k for k in keys if pj[k].get('preset') == preset), keys[0])
    return (pj[k].get('caption_fields') or {}).get('target_text', '').strip(), k


def cell(frame_small, m, alpha):
    """프레임 축소판 + 그 층·그 프레임의 히트맵. 정규화는 셀 자기 최대."""
    h, w = frame_small.shape[:2]
    up = np.asarray(Image.fromarray(
        (np.clip(m / max(float(m.max()), 1e-12), 0, 1) * 255).astype(np.uint8)
    ).resize((w, h), Image.BILINEAR)) / 255.0
    a = (up[..., None] ** 0.6) * alpha
    out = frame_small.astype(np.float32) / 255.0 * (1 - a) + colorize(up) * a
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def tile(frames_small, maps, layers, ncol, alpha, label=True):
    """(층 -> (T,s,s)) 를 프레임마다 ncol 격자로 타일. 반환 generator of (H,W,3)."""
    n = len(layers)
    nrow = int(np.ceil(n / ncol))
    h, w = frames_small[0].shape[:2]
    for f in range(len(frames_small)):
        canvas = np.zeros((nrow * h, ncol * w, 3), np.uint8)
        for i, li in enumerate(layers):
            r, c = divmod(i, ncol)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = cell(
                frames_small[f], maps[li][f], alpha)
        if label:
            im = Image.fromarray(canvas)
            d = ImageDraw.Draw(im)
            for i, li in enumerate(layers):
                r, c = divmod(i, ncol)
                d.rectangle([c * w + 1, r * h + 1, c * w + 34, r * h + 14], fill=(0, 0, 0))
                d.text((c * w + 3, r * h + 3), f'L{li:02d}', fill=(255, 255, 255))
            d.rectangle([2, nrow * h - 16, 150, nrow * h - 2], fill=(0, 0, 0))
            d.text((4, nrow * h - 14), f'frame {f:02d}', fill=(255, 255, 255))
            canvas = np.asarray(im)
        yield canvas


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    tgt, seg = corpus_target(args.root, args.scene, args.preset)
    assert tgt, f'{args.scene}: target_text 가 비었다'
    text = f'{args.verb} {tgt}'
    r.set_video(scene=args.scene, start=0, num_frames=args.num_frames, fps=args.fps,
                root=args.root)
    v = r.vid
    T, side = v['T'], v['side']
    print(f'[in] {args.scene} seg {seg}  prompt {text!r}')

    small = [np.asarray(Image.fromarray(v['video'][f]).resize(
        (args.cell_w, args.cell_w * v['video'].shape[1] // v['video'].shape[2]),
        Image.BILINEAR)) for f in range(T)]

    for name, span in (('alltail', 'all'), ('target', tgt)):
        maps, vmass, nrow_q = r.maps_all_layers(text, span=span, probe='')
        layers = sorted(maps)
        print(f'\n=== query={name}  ({nrow_q} 토큰) ===')
        print(f'{"층":>3s} {"video%":>7s} {"ring%":>7s} {"f0-2%":>7s}   '
              f'{"층":>3s} {"video%":>7s} {"ring%":>7s} {"f0-2%":>7s}')
        ring = np.zeros((side, side), bool)
        ring[0] = ring[-1] = True
        ring[:, 0] = ring[:, -1] = True
        col = []
        for li in layers:
            m = maps[li]
            sp = m.sum(0); sp = sp / max(sp.sum(), 1e-9)
            fm = m.reshape(T, -1).sum(1)
            col.append((li, vmass[li] * 100, sp[ring].sum() * 100, fm[:3].sum() * 100))
        half = (len(col) + 1) // 2
        for a, b in zip(col[:half], col[half:] + [None] * half):
            s = f'{a[0]:3d} {a[1]:6.2f}% {a[2]:6.1f}% {a[3]:6.1f}%'
            if b:
                s += f'   {b[0]:3d} {b[1]:6.2f}% {b[2]:6.1f}% {b[3]:6.1f}%'
            print(s)
        print(f'  (uniform: ring {ring.mean()*100:.1f}%  f0-2 {3/T*100:.1f}%)')

        p = osp.join(args.out_dir,
                     f'layers_{args.scene.split("/")[-1]}_{name}_fps{args.fps:g}.mp4')
        wr = iio.get_writer(p, fps=args.out_fps, codec='libx264', quality=8,
                            macro_block_size=1)
        for img in tile(small, maps, layers, args.ncol, args.alpha):
            wr.append_data(img)
        wr.close()
        np.save(p[:-4] + '.npy', np.stack([maps[li] for li in layers]))
        print(f'[save] {p}   ({len(layers)}층 x {T}프레임, 격자 {args.ncol}열)')
        del maps


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scene', default='vista4d/camel')
    p.add_argument('--preset', default='dolly_in_look_at')
    p.add_argument('--verb', default='Track')
    p.add_argument('--num_frames', type=int, default=49)
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--ncol', type=int, default=6)
    p.add_argument('--cell_w', type=int, default=320)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--out_fps', type=int, default=6)
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--gpu', default='2')
    main(p.parse_args())
