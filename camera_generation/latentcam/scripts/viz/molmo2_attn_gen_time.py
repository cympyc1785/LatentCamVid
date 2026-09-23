"""생성 중 attention 이 **생성하는 timestamp 를 따라가는가** (D164 결정적 검증).

  python scripts/viz/molmo2_attn_gen_time.py --scene vista4d/camel --gpu 2

가설 (사용자 2026-09-10): `<tracks coords="0.0 1 497 438;0.5 1 497 444;...">` 를 생성할 때,
`t=0.0` 좌표의 자릿수를 뱉는 스텝은 **프레임 0** 을 보고, `t=2.5` 좌표를 뱉는 스텝은
**프레임 12~13**(= 2.5 x fps 5) 을 봐야 한다.

어떻게: 생성 훅으로 (스텝, 층) attention 을 모은 뒤,
  ① 생성 원문을 파싱해 좌표 그룹(`t id x y`)의 문자 구간을 찾고
  ② **x·y 필드의 자릿수 토큰**만 그 그룹에 귀속시킨다 (timestamp 자릿수는 제외 — 그건 시간을
     쓰는 자리고 위치를 쓰는 자리가 아니다)
  ③ 그룹별로 평균해 프레임별 질량 (T,) 을 얻는다
  ④ 기대 프레임 `round(t x fps)` 와 attention 의 peak/centroid 프레임을 비교한다

산출물: 층별 (그룹 x 프레임) 질량 행렬을 6x6 타일 이미지로. **대각선이 보이면 가설 성립.**
그리고 층별 Spearman ρ(기대 프레임, attention centroid 프레임).
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
from PIL import Image, ImageDraw

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, colorize, OUT, CORPUS          # noqa: E402
from molmo2_attn_layers_video import corpus_target                   # noqa: E402
from molmo2_attn_gen_layers import hook_generate                     # noqa: E402
import cache_molmo2_embeddings as C                                  # noqa: E402


def parse_groups(toks: list, raw: str):
    """생성 토큰열 -> [(timestamp, [x·y 자릿수 스텝 인덱스])].

    스텝 k 가 토큰 g_k 를 만든다. 좌표 그룹은 `coords="..."` 안의 `;` 구분 단위이고,
    각 그룹은 공백 구분 `t id x y` 다. 위치를 쓰는 자리는 **field 2·3** 뿐이다.
    """
    q = raw.find('coords="')
    assert q >= 0, '생성 원문에 coords=" 가 없다'
    s = q + 8
    e = raw.find('"', s)
    assert e > s, 'coords 속성이 닫히지 않았다'
    # 토큰별 문자 구간
    span, pos = [], 0
    for t in toks:
        span.append((pos, pos + len(t)))
        pos += len(t)
    out, off = [], s
    for chunk in raw[s:e].split(';'):
        a, b = off, off + len(chunk)
        off = b + 1                                   # ';' 한 글자
        fields, fpos = [], a
        for fld in chunk.split(' '):
            if fld:
                i = raw.find(fld, fpos)
                fields.append((i, i + len(fld), fld))
                fpos = i + len(fld)
        if len(fields) < 4:
            continue
        t_sec = float(fields[0][2])
        lo, hi = fields[2][0], fields[3][1]           # x 시작 ~ y 끝
        steps = [k for k, (ta, tb) in enumerate(span)
                 if ta >= lo and tb <= hi and re.fullmatch(r'\d', toks[k])]
        if steps:
            out.append((t_sec, steps))
    return out


def rank(a):
    o = np.argsort(a); r = np.empty_like(o, float); r[o] = np.arange(len(a)); return r


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-12 else float('nan')


def render_single(M_pref, M_grp, ts, exp, T, li, args, out_png):
    """한 층의 (prefill + 그룹) x 프레임 행렬을 **축 라벨과 함께** 큰 판으로 그린다.

    왜 따로 두나 (사용자 요청 2026-09-14): 6x6 타일판은 prefill 을 뺀다 — 그룹 귀속 조건이
    "x·y 필드 안의 한 자리 숫자 토큰" 이라 태그를 여는 토큰 0(= prefill 이 만든 것)은
    어디에도 안 들어간다. 여기서는 **맨 윗줄에 prefill 을 붙여** 생성 단계와 대비시킨다.

    정규화는 6x6 판과 같게 **행마다 자기 최대**다 (뒤 그룹일수록 KV 가 커져 질량이
    희석되므로 전역 정규화하면 위쪽만 밝다). 따라서 행 사이 밝기는 비교하면 안 되고
    각 행에서 어디가 가장 밝은지만 본다.
    """
    from PIL import Image, ImageDraw
    rows = np.concatenate([M_pref[None, :], M_grp], 0)          # (1+G, T)
    lab = ['prefill'] + [f't={t:.1f}s' for t in ts]
    expc = [None] + [int(e) for e in exp]
    n = rows.shape[0]
    cw, ch = args.big_cw, args.big_ch                            # 셀 픽셀
    padl, padt = 96, 26
    W_, H_ = padl + T * cw, padt + n * ch + 22
    im = Image.new('RGB', (W_, H_), (12, 12, 12))
    dr = ImageDraw.Draw(im)
    norm = rows / np.clip(rows.max(1, keepdims=True), 1e-12, None)
    rgb = (colorize(norm) * 255).astype(np.uint8)
    for r_ in range(n):
        for c_ in range(T):
            dr.rectangle([padl + c_ * cw, padt + r_ * ch,
                          padl + (c_ + 1) * cw - 1, padt + (r_ + 1) * ch - 1],
                         fill=tuple(int(x) for x in rgb[r_, c_]))
        dr.text((4, padt + r_ * ch + ch // 2 - 5), lab[r_],
                fill=(200, 200, 200) if r_ else (255, 210, 80))
        if expc[r_] is not None:                                  # 기대 프레임 표식
            x = padl + expc[r_] * cw + cw // 2
            dr.line([x, padt + r_ * ch, x, padt + (r_ + 1) * ch - 1], fill=(0, 255, 0))
        am = int(np.argmax(rows[r_]))                             # 실제 peak
        x = padl + am * cw + cw // 2
        dr.rectangle([x - 2, padt + r_ * ch + ch // 2 - 2, x + 2,
                      padt + r_ * ch + ch // 2 + 2], outline=(0, 255, 255))
    for c_ in range(0, T, 4):
        dr.text((padl + c_ * cw, padt - 14), str(c_), fill=(170, 170, 170))
    dr.text((4, padt - 14), 'frame', fill=(170, 170, 170))
    dr.text((4, padt + n * ch + 5),
            f'L{li:02d}  green=expected frame (round(t x fps))  cyan=attention peak  '
            f'row-normalized', fill=(200, 200, 200))
    im.save(out_png)
    return rows



def main(args):
    makedirs(args.out_dir, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    tgt, seg = corpus_target(args.root, args.scene, args.preset)
    text = f'{args.verb} {tgt}'
    r.set_video(scene=args.scene, start=0, num_frames=args.num_frames, fps=args.fps,
                root=args.root)
    v = r.vid
    T, side = v['T'], v['side']
    print(f'[in] {args.scene} seg {seg}  prompt {text!r}  fps {args.fps}')

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
    groups = parse_groups(toks, raw)
    print(f'[gen] {len(gen)} 토큰, 좌표 그룹 {len(groups)}개')
    print(f'[gen] {raw.strip()[:150]}')

    # ── 그룹 x 층 -> 프레임별 질량
    ts = np.array([t for t, _ in groups])
    exp = np.clip(np.rint(ts * args.fps), 0, T - 1)          # 기대 프레임
    M = np.zeros((r.nl, len(groups), T))                      # 층 x 그룹 x 프레임
    for gi, (_t, steps) in enumerate(groups):
        for li in range(r.nl):
            acc = [store[s][li] for s in steps if s in store and li in store[s]]
            if not acc:
                continue
            a = torch.stack(acc).mean(0)
            a = (a / a.sum().clamp_min(1e-9)).view(T, side, side).numpy()
            M[li, gi] = a.reshape(T, -1).sum(1)
    store0 = store if args.single_layer >= 0 else None
    if args.single_layer < 0:
        del store

    print(f'\n{"층":>3s} {"peak ρ":>8s} {"cent ρ":>8s} {"cent MAE":>9s} {"peak 적중±2":>11s}'
          f'   {"층":>3s} {"peak ρ":>8s} {"cent ρ":>8s} {"cent MAE":>9s} {"peak 적중±2":>11s}')
    col = []
    fr = np.arange(T, dtype=float)
    for li in range(r.nl):
        pk = M[li].argmax(1).astype(float)
        ce = (M[li] * fr).sum(1) / np.clip(M[li].sum(1), 1e-12, None)
        col.append((li, corr(rank(exp), rank(pk)), corr(rank(exp), rank(ce)),
                    float(np.abs(ce - exp).mean()), float(np.mean(np.abs(pk - exp) <= 2))))
    half = (r.nl + 1) // 2
    for a, bb in zip(col[:half], col[half:] + [None] * half):
        s = f'{a[0]:3d} {a[1]:+8.3f} {a[2]:+8.3f} {a[3]:9.2f} {a[4]*100:10.0f}%'
        if bb:
            s += f'   {bb[0]:3d} {bb[1]:+8.3f} {bb[2]:+8.3f} {bb[3]:9.2f} {bb[4]*100:10.0f}%'
        print(s)
    best = max(col, key=lambda c: c[2])
    print(f'\n  centroid ρ 최고: 층 {best[0]}  ρ={best[2]:+.3f}  MAE {best[3]:.2f} 프레임  '
          f'peak 적중(±2) {best[4]*100:.0f}%')
    print(f'  (그룹 {len(groups)}개, 기대 프레임 {exp.min():.0f}~{exp.max():.0f}, '
          f'무작위 기준 peak 적중 ±2 = {5/T*100:.0f}%)')

    # ── 그림: 층별 (그룹 x 프레임) 행렬을 6x6 타일. 대각선이 보이면 가설 성립
    ch, cw = args.cell_h, args.cell_w
    ncol = args.ncol
    nrow = int(np.ceil(r.nl / ncol))
    canvas = np.zeros((nrow * ch, ncol * cw, 3), np.uint8)
    for li in range(r.nl):
        m = M[li] / np.clip(M[li].max(1, keepdims=True), 1e-12, None)   # 그룹마다 자기 최대
        img = colorize(np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                                  .resize((cw, ch), Image.NEAREST)) / 255.0)
        rr, cc = divmod(li, ncol)
        canvas[rr * ch:(rr + 1) * ch, cc * cw:(cc + 1) * cw] = (img * 255).astype(np.uint8)
    if args.single_layer >= 0:
        li = args.single_layer
        # prefill = store 의 스텝 0 (forward 0). 그룹 귀속 조건을 안 타므로 직접 꺼낸다.
        a0 = store0[0][li]
        a0 = (a0 / a0.sum().clamp_min(1e-9)).view(T, side, side).numpy().reshape(T, -1).sum(1)
        fp1 = osp.join(args.out_dir,
                       f'gentime_pref_{args.scene.split("/")[-1]}_L{li:02d}_fps{args.fps:g}.png')
        rows = render_single(a0, M[li], ts, exp, T, li, args, fp1)
        pk = int(np.argmax(a0))
        print(f'\n[single] L{li}  prefill peak 프레임 {pk}  '
              f'(질량 최대 {a0.max()*100:.1f}%, 균등 {100/T:.1f}%)  '
              f'프레임 분포 엔트로피 {-(a0*np.log(np.clip(a0,1e-12,None))).sum():.3f} '
              f'(균등 {np.log(T):.3f})')
        print(f'[save] {fp1}')
        np.save(fp1[:-4] + '.npy', rows)

    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    for li in range(r.nl):
        rr, cc = divmod(li, ncol)
        # 기대 대각선 (흰 점)
        for gi in range(len(groups)):
            x = int((exp[gi] + .5) / T * cw) + cc * cw
            y = int((gi + .5) / len(groups) * ch) + rr * ch
            d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=(255, 255, 255))
        d.rectangle([cc * cw + 1, rr * ch + 1, cc * cw + 78, rr * ch + 14], fill=(0, 0, 0))
        d.text((cc * cw + 3, rr * ch + 3), f'L{li:02d} r={col[li][2]:+.2f}',
               fill=(255, 255, 255))
    p = osp.join(args.out_dir, f'gentime_{args.scene.split("/")[-1]}_fps{args.fps:g}.png')
    im.save(p)
    np.save(p[:-4] + '.npy', M)
    json.dump(dict(scene=args.scene, fps=args.fps, prompt=text, raw=raw,
                   timestamps=ts.tolist(), expected_frame=exp.tolist(),
                   per_layer=[dict(layer=c[0], peak_rho=c[1], cent_rho=c[2],
                                   cent_mae=c[3], peak_hit2=c[4]) for c in col]),
              open(p[:-4] + '.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\n[save] {p}   (가로=프레임 0..{T-1}, 세로=생성 좌표 그룹 순서, 흰 점=기대 프레임)')


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
    p.add_argument('--single_layer', type=int, default=-1)  # >=0 이면 그 층만 큰 판
    p.add_argument('--big_cw', type=int, default=18)
    p.add_argument('--big_ch', type=int, default=26)
    p.add_argument('--cell_w', type=int, default=294)
    p.add_argument('--cell_h', type=int, default=170)
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--gpu', default='2')
    main(p.parse_args())
