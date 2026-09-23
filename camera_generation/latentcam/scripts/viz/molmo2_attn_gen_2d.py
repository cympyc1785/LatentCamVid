"""생성 중 attention 을 **그 좌표가 가리키는 프레임 위에** 2D 로 겹쳐 본다 (D164).

  python scripts/viz/molmo2_attn_gen_2d.py --scene vista4d/camel --gpu 2

`molmo2_attn_gen_time.py` 가 (그룹 x 프레임) 행렬로 "시간을 따라간다"를 보였다면, 여기서는
그 다음 질문을 닫는다 — **그 프레임 안에서 attention 이 모델이 실제로 뱉은 좌표를 짚는가.**

`<tracks coords="t id x y;...">` 의 그룹 g 마다:
    f_g   = round(t_g x fps)              그 좌표가 가리키는 프레임
    xy_g  = (x,y)/1000 x (W,H)            모델이 뱉은 픽셀 좌표  ← 초록 십자
    attn  = 그룹 g 의 x·y 자릿수 스텝 평균 attention 의 **프레임 f_g 슬라이스** (9x9)
            그 argmax 셀 중심                                    ← 청록 십자

두 십자의 픽셀 거리가 지표다. **자기 대조**라 절대 정확도가 아니라 "attention 이 그 좌표를
만든 근거인가"를 잰다 — 둘이 일치하면 좌표가 attention 이 본 곳에서 나왔다는 뜻이다.

산출물
  mp4   출력 프레임 = 좌표 그룹(시간 진행). 각 프레임에 36층 6x6 타일, 두 십자 표시
  표    층별 거리 중앙값 / hit@1cell / hit@2cell. 기준선은 균등 격자와 화면 중심
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import re
import sys

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch
import imageio.v2 as iio
from PIL import Image, ImageDraw

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, colorize, OUT, CORPUS      # noqa: E402
from molmo2_attn_layers_video import corpus_target               # noqa: E402
from molmo2_attn_gen_layers import hook_generate                 # noqa: E402
import cache_molmo2_embeddings as C                              # noqa: E402


def parse_groups_xy(toks, raw):
    """[(t, x1000, y1000, [x·y 자릿수 스텝])]. `molmo2_attn_gen_time.parse_groups` 확장판."""
    q = raw.find('coords="')
    assert q >= 0, 'coords=" 없음'
    s, e = q + 8, raw.find('"', q + 8)
    span, pos = [], 0
    for t in toks:
        span.append((pos, pos + len(t))); pos += len(t)
    out, off = [], s
    for chunk in raw[s:e].split(';'):
        a, b = off, off + len(chunk); off = b + 1
        f, fp = [], a
        for fld in chunk.split(' '):
            if fld:
                i = raw.find(fld, fp); f.append((i, i + len(fld), fld)); fp = i + len(fld)
        if len(f) < 4:
            continue
        lo, hi = f[2][0], f[3][1]
        steps = [k for k, (ta, tb) in enumerate(span)
                 if ta >= lo and tb <= hi and re.fullmatch(r'\d', toks[k])]
        if steps:
            out.append((float(f[0][2]), float(f[2][2]), float(f[3][2]), steps))
    return out


def cell(frame, m, alpha, gxy, axy):
    """프레임 + 9x9 히트맵 + 초록(생성 좌표) / 청록(attention argmax) 십자."""
    h, w = frame.shape[:2]
    up = np.asarray(Image.fromarray(
        (np.clip(m / max(float(m.max()), 1e-12), 0, 1) * 255).astype(np.uint8)
    ).resize((w, h), Image.BILINEAR)) / 255.0
    a = (up[..., None] ** 0.6) * alpha
    out = frame.astype(np.float32) / 255.0 * (1 - a) + colorize(up) * a
    for (px, py), col, r in ((gxy, [0, 1, 0], 7), (axy, [0, 1, 1], 5)):
        cx, cy = int(round(px * w / 640)), int(round(py * h / 360))
        if 0 <= cx < w and 0 <= cy < h:
            out[max(0, cy - 1):cy + 1, max(0, cx - r):cx + r] = col
            out[max(0, cy - r):cy + r, max(0, cx - 1):cx + 1] = col
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    tgt, seg = corpus_target(args.root, args.scene, args.preset)
    text = f'{args.verb} {tgt}'
    r.set_video(scene=args.scene, start=0, num_frames=args.num_frames, fps=args.fps,
                root=args.root)
    v = r.vid
    T, side = v['T'], v['side']
    H, W = v['video'].shape[1], v['video'].shape[2]
    print(f'[in] {args.scene} seg {seg}  {W}x{H}  prompt {text!r}')

    b = r._batch(v['video'], text, '', args.fps)
    b = {k: (x.to(r.device, r.dtype) if k == 'pixel_values_videos' else x.to(r.device))
         for k, x in b.items()}
    store = {}
    restore = hook_generate(r.model, v['patch_pos'].numpy(), store)
    try:
        with torch.inference_mode():
            g = r.model.generate(**b, max_new_tokens=args.max_new_tokens, do_sample=False)
    finally:
        restore()
    gen = g[0, b['input_ids'].shape[1]:].tolist()
    toks = [r.proc.tokenizer.decode([t]) for t in gen]
    raw = r.proc.tokenizer.decode(gen, skip_special_tokens=True)
    G = parse_groups_xy(toks, raw)
    print(f'[gen] 좌표 그룹 {len(G)}개   {raw.strip()[:120]}')

    # ── 그룹 x 층 -> (그 프레임의 9x9, 거리)
    maps = np.zeros((r.nl, len(G), side, side))
    fidx, gxy = [], []
    for gi, (t, x1k, y1k, steps) in enumerate(G):
        f = int(np.clip(round(t * args.fps), 0, T - 1))
        fidx.append(f); gxy.append((x1k / 1000 * W, y1k / 1000 * H))
        for li in range(r.nl):
            acc = [store[s][li] for s in steps if s in store and li in store[s]]
            if acc:
                a = torch.stack(acc).mean(0)
                a = (a / a.sum().clamp_min(1e-9)).view(T, side, side).numpy()
                maps[li, gi] = a[f]
    del store
    gxy = np.array(gxy); fidx = np.array(fidx)

    def cell_px(rc):
        return ((rc[1] + .5) * W / side, (rc[0] + .5) * H / side)
    cellr = float(np.hypot(W / side, H / side)) / 2
    # 기준선: 균등 격자 기대거리 / 화면 중심
    ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
    gridpx = np.stack([(xs + .5) * W / side, (ys + .5) * H / side], -1).reshape(-1, 2)
    bl_grid = float(np.mean([np.hypot(*(gridpx - p).T).mean() for p in gxy]))
    bl_cen = float(np.mean(np.hypot(gxy[:, 0] - W / 2, gxy[:, 1] - H / 2)))

    rows = []
    for li in range(r.nl):
        d = []
        for gi in range(len(G)):
            rc = np.unravel_index(int(np.argmax(maps[li, gi])), (side, side))
            px, py = cell_px(rc)
            d.append(np.hypot(px - gxy[gi, 0], py - gxy[gi, 1]))
        d = np.array(d)
        rows.append((li, float(np.median(d)), float(np.mean(d <= cellr)),
                     float(np.mean(d <= 2 * cellr))))
    print(f'\n기준선: 균등격자 {bl_grid:.0f}px / 화면중심 {bl_cen:.0f}px / 1셀 반경 {cellr:.0f}px')
    print(f'{"층":>3s} {"거리 중앙값":>11s} {"hit@1cell":>10s} {"hit@2cell":>10s}   '
          f'{"층":>3s} {"거리 중앙값":>11s} {"hit@1cell":>10s} {"hit@2cell":>10s}')
    half = (r.nl + 1) // 2
    for a, bb in zip(rows[:half], rows[half:] + [None] * half):
        s = f'{a[0]:3d} {a[1]:10.1f}px {a[2]*100:9.0f}% {a[3]*100:9.0f}%'
        if bb:
            s += f'   {bb[0]:3d} {bb[1]:10.1f}px {bb[2]*100:9.0f}% {bb[3]*100:9.0f}%'
        print(s)
    best = min(rows, key=lambda x: x[1])
    print(f'\n  최고: 층 {best[0]}  중앙값 {best[1]:.1f}px  '
          f'hit@1cell {best[2]*100:.0f}%  hit@2cell {best[3]*100:.0f}%')

    # ── 영상: 출력 프레임 = 좌표 그룹
    sel = ([int(x) for x in args.layers.split(',')] if args.layers else list(range(r.nl)))
    cw = args.cell_w; chh = cw * H // W
    ncol = min(args.ncol, len(sel)); nrow = int(np.ceil(len(sel) / ncol))
    small = {f: np.asarray(Image.fromarray(v['video'][f]).resize((cw, chh), Image.BILINEAR))
             for f in set(fidx.tolist())}
    tag = f'_L{args.layers.replace(",", "-")}' if args.layers else ''
    p = osp.join(args.out_dir,
                 f'gen2d_{args.scene.split("/")[-1]}_fps{args.fps:g}{tag}.mp4')
    wr = iio.get_writer(p, fps=args.out_fps, codec='libx264', quality=8, macro_block_size=1)
    for gi in range(len(G)):
        f = int(fidx[gi])
        gp = (gxy[gi, 0] * 640 / W, gxy[gi, 1] * 360 / H)      # cell() 이 640x360 기준으로 받음
        cv = np.zeros((nrow * chh, ncol * cw, 3), np.uint8)
        for i_, li in enumerate(sel):
            rc = np.unravel_index(int(np.argmax(maps[li, gi])), (side, side))
            ap = cell_px(rc)
            rr, cc = divmod(i_, ncol)
            cv[rr * chh:(rr + 1) * chh, cc * cw:(cc + 1) * cw] = cell(
                small[f], maps[li, gi], args.alpha, gp,
                (ap[0] * 640 / W, ap[1] * 360 / H))
        im = Image.fromarray(cv); d = ImageDraw.Draw(im)
        for i_, li in enumerate(sel):
            rr, cc = divmod(i_, ncol)
            d.rectangle([cc * cw + 1, rr * chh + 1, cc * cw + 34, rr * chh + 14],
                        fill=(0, 0, 0))
            d.text((cc * cw + 3, rr * chh + 3), f'L{li:02d}', fill=(255, 255, 255))
        d.rectangle([2, nrow * chh - 16, 470, nrow * chh - 2], fill=(0, 0, 0))
        d.text((4, nrow * chh - 14),
               f't={G[gi][0]:.1f}s  frame {f:02d}   green=generated coord  cyan=attn argmax',
               fill=(255, 255, 255))
        wr.append_data(np.asarray(im))
    wr.close()
    np.save(p[:-4] + '.npy', maps)
    json.dump(dict(scene=args.scene, prompt=text, fps=args.fps, raw=raw,
                   groups=[dict(t=G[i][0], frame=int(fidx[i]),
                                gen_xy=[float(gxy[i, 0]), float(gxy[i, 1])])
                           for i in range(len(G))],
                   baseline=dict(grid=bl_grid, center=bl_cen, cell_radius=cellr),
                   per_layer=[dict(layer=a[0], median_px=a[1], hit1=a[2], hit2=a[3])
                              for a in rows]),
              open(p[:-4] + '.json', 'w'), ensure_ascii=False, indent=1)
    print(f'[save] {p}   (출력 프레임 {len(G)}개 = 좌표 그룹)')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scene', default='vista4d/camel')
    p.add_argument('--preset', default='dolly_in_look_at')
    p.add_argument('--verb', default='Track')
    p.add_argument('--num_frames', type=int, default=49)
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--ncol', type=int, default=6)
    p.add_argument('--layers', default='', help='쉼표 구분. 비우면 36층 전체')
    p.add_argument('--cell_w', type=int, default=320)
    p.add_argument('--alpha', type=float, default=0.65)
    p.add_argument('--out_fps', type=int, default=2)
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--gpu', default='2')
    main(p.parse_args())
