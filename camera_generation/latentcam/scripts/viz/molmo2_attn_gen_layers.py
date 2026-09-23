"""**생성 중** attention 을 36층 타일로 (D164). prefill 판이 놓친 자리를 본다.

  python scripts/viz/molmo2_attn_gen_layers.py --scene vista4d/camel --gpu 2

왜: `molmo2_attn_layers_video.py` 는 **prefill** 만 본다 — 프롬프트를 한 번 태운 뒤 텍스트 위치가
patch 를 보는 attention 이다. 그런데 좌표는 **생성 단계**에서 나온다. 모델이
`<tracks coords="0.0 1 4 9 7 ...` 를 뱉을 때 `'4'`·`'9'`·`'7'` 각각의 forward 가 KV 캐시의
patch 3,969 키를 다시 attend 하고, 국소화가 실제로 계산되는 자리는 거기다.

어떻게: `eager_attention_forward` 를 감싸 **매 forward 의 마지막 query 행**만 떠낸다.
  prefill  q_len = L_prompt  ->  마지막 행이 첫 생성 토큰 g0 을 만든다
  decode   q_len = 1         ->  그 행이 g_k 를 만든다
`layer_idx == 0` 에서 스텝 카운터를 올려 (스텝, 층) 으로 저장한다. head 평균을 훅 안에서
바로 해서 메모리를 스텝당 16 KB 로 줄인다 (안 하면 36층 x 200스텝 x (32,4332) = 40 GB).

무엇을 그리나: 좌표 자릿수를 만든 스텝들의 attention 을 평균해 `(49,9,9)` 로 보고, prefill 판과
**같은 6x6 타일**로 낸다. 두 영상을 낸다:
  `gen_p0`    첫 좌표쌍(t=0.0)의 x·y 자릿수 스텝만       <- 첫 프레임 위치를 만드는 자리
  `gen_all`   `<tracks>` 안의 모든 자릿수 스텝           <- 대조 (시각이 섞여 흐려진다)

주의: 재정규화·head 평균·표시 변환은 prefill 판과 **같은 규약**이다 (셀마다 자기 최대 정규화,
alpha 합성). 그래서 두 영상을 나란히 비교할 수 있다. argmax 십자를 켜 뒀다 (prefill 판과 달리).
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
import cache_molmo2_embeddings as C                              # noqa: E402


def hook_generate(model, patch_pos, store):
    """매 forward 의 **마지막 query 행** x patch 열을 head 평균해 (스텝, 층) 으로 모은다."""
    mod = sys.modules[type(model.model).__module__]
    orig = mod.eager_attention_forward
    pp = torch.as_tensor(np.asarray(patch_pos))
    st = {'step': -1}

    def wrapped(module, q, k, v, attention_mask, scaling, dropout=0.0, **kw):
        out, w = orig(module, q, k, v, attention_mask, scaling, dropout=dropout, **kw)
        li = getattr(module, 'layer_idx', -1)
        if li == 0:
            st['step'] += 1
        if w is not None and w.shape[-1] > int(pp.max()):
            a = w[0, :, -1, :]                                  # (H, kv_len) 마지막 query 행
            store.setdefault(st['step'], {})[li] = \
                a[:, pp.to(a.device)].float().mean(0).cpu()      # (3969,) head 평균
        return out, w

    mod.eager_attention_forward = wrapped
    cfg = model.model.transformer.blocks[0].self_attn.config
    prev = cfg._attn_implementation
    cfg._attn_implementation = 'eager'

    def restore():
        mod.eager_attention_forward = orig
        cfg._attn_implementation = prev
    return restore


def cell(frame_small, m, alpha, mark):
    h, w = frame_small.shape[:2]
    up = np.asarray(Image.fromarray(
        (np.clip(m / max(float(m.max()), 1e-12), 0, 1) * 255).astype(np.uint8)
    ).resize((w, h), Image.BILINEAR)) / 255.0
    a = (up[..., None] ** 0.6) * alpha
    out = frame_small.astype(np.float32) / 255.0 * (1 - a) + colorize(up) * a
    if mark:
        r, c = np.unravel_index(int(np.argmax(m)), m.shape)
        cy, cx = int((r + .5) * h / m.shape[0]), int((c + .5) * w / m.shape[1])
        t = 1
        out[max(0, cy - 6):cy + 6, max(0, cx - t):cx + t] = [0, 1, 1]
        out[max(0, cy - t):cy + t, max(0, cx - 6):cx + 6] = [0, 1, 1]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def write_tile(path, frames_small, maps, layers, ncol, alpha, out_fps, mark, note):
    nrow = int(np.ceil(len(layers) / ncol))
    h, w = frames_small[0].shape[:2]
    wr = iio.get_writer(path, fps=out_fps, codec='libx264', quality=8, macro_block_size=1)
    for f in range(len(frames_small)):
        cv = np.zeros((nrow * h, ncol * w, 3), np.uint8)
        for i, li in enumerate(layers):
            r, c = divmod(i, ncol)
            cv[r * h:(r + 1) * h, c * w:(c + 1) * w] = cell(
                frames_small[f], maps[li][f], alpha, mark)
        im = Image.fromarray(cv)
        d = ImageDraw.Draw(im)
        for i, li in enumerate(layers):
            r, c = divmod(i, ncol)
            d.rectangle([c * w + 1, r * h + 1, c * w + 34, r * h + 14], fill=(0, 0, 0))
            d.text((c * w + 3, r * h + 3), f'L{li:02d}', fill=(255, 255, 255))
        d.rectangle([2, nrow * h - 16, 340, nrow * h - 2], fill=(0, 0, 0))
        d.text((4, nrow * h - 14), f'frame {f:02d}   {note}', fill=(255, 255, 255))
        wr.append_data(np.asarray(im))
    wr.close()


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    r = Runner(args.ckpt, args.out_dir)
    tgt, seg = corpus_target(args.root, args.scene, args.preset)
    text = f'{args.verb} {tgt}'
    r.set_video(scene=args.scene, start=0, num_frames=args.num_frames, fps=args.fps,
                root=args.root)
    v = r.vid
    T, side = v['T'], v['side']
    print(f'[in] {args.scene} seg {seg}  prompt {text!r}')

    # ── 생성 (훅 걸고) ────────────────────────────────────────────────────────────
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
    gen_ids = g[0, b['input_ids'].shape[1]:].tolist()
    toks = [r.proc.tokenizer.decode([t]) for t in gen_ids]
    raw = r.proc.tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f'[gen] {len(gen_ids)} 토큰,  포획 스텝 {len(store)}  (0=prefill)')
    print(f'[gen] {raw.strip()[:200]}')

    # ── 어느 스텝이 좌표 자릿수를 만들었나 ──────────────────────────────────────
    # 스텝 k 가 g_k 를 만든다 (prefill=스텝0 -> g_0). coords=" 이후의 숫자만 고른다.
    q = raw.find('coords="')
    assert q >= 0, '생성 원문에 coords=" 가 없다'
    pos, in_coords = 0, []
    for i, t in enumerate(toks):
        in_coords.append(pos >= q + 8)
        pos += len(t)
    digit = [i for i, t in enumerate(toks) if in_coords[i] and re.fullmatch(r'\d', t)]
    # 첫 좌표쌍: `t id x y` 중 x·y 의 자릿수 = 첫 세미콜론 앞의 숫자에서 앞 2개(t, id) 제외
    semi = next((i for i, t in enumerate(toks) if in_coords[i] and ';' in t), len(toks))
    first = [i for i in digit if i < semi]
    p0 = first[3:] if len(first) > 3 else first     # `0`,`.`,`0` 뒤 → id, x, y 자릿수
    print(f'[step] 자릿수 스텝 {len(digit)}개,  첫 좌표쌍 {len(p0)}개 '
          f'-> 토큰 {"".join(toks[i] for i in p0)!r}')

    def reduce_steps(steps, name):
        """스텝들의 (3969,) 를 평균 -> 층별 (T,9,9) + video 질량."""
        out, vm = {}, {}
        for li in range(r.nl):
            acc = [store[s][li] for s in steps if s in store and li in store[s]]
            if not acc:
                continue
            a = torch.stack(acc).mean(0)                       # (3969,)
            vm[li] = float(a.sum())                            # patch 로 간 질량 (재정규화 전)
            out[li] = (a / a.sum().clamp_min(1e-9)).view(T, side, side).numpy()
        print(f'  [{name}] 층 {len(out)}개, 스텝 {len(steps)}개')
        return out, vm

    small = [np.asarray(Image.fromarray(v['video'][f]).resize(
        (args.cell_w, args.cell_w * v['video'].shape[1] // v['video'].shape[2]),
        Image.BILINEAR)) for f in range(T)]
    ring = np.zeros((side, side), bool)
    ring[0] = ring[-1] = True
    ring[:, 0] = ring[:, -1] = True

    rows = {}
    for steps, name, note in ((p0, 'gen_p0', 'first coord pair digits'),
                              (digit, 'gen_all', 'all coord digits'),
                              ([0], 'prefill_last', 'prefill last row')):
        if not steps:
            continue
        maps, vm = reduce_steps(steps, name)
        layers = sorted(maps)
        rows[name] = {li: (vm[li] * 100,
                           float((maps[li].sum(0) / max(maps[li].sum(), 1e-9))[ring].sum() * 100),
                           float(maps[li].reshape(T, -1).sum(1)[:3].sum() * 100))
                      for li in layers}
        p = osp.join(args.out_dir,
                     f'genlayers_{args.scene.split("/")[-1]}_{name}_fps{args.fps:g}.mp4')
        write_tile(p, small, maps, layers, args.ncol, args.alpha, args.out_fps,
                   args.mark, note)
        np.save(p[:-4] + '.npy', np.stack([maps[li] for li in layers]))
        print(f'  [save] {p}')

    print(f'\n{"층":>3s} ' + ' '.join(f'{n:>22s}' for n in rows))
    print(f'{"":>3s} ' + ' '.join(f'{"video%":>7s}{"ring%":>7s}{"f0-2%":>8s}' for _ in rows))
    for li in range(r.nl):
        s = f'{li:3d} '
        for n in rows:
            if li in rows[n]:
                a, b_, c_ = rows[n][li]
                s += f' {a:6.2f}%{b_:6.1f}%{c_:7.1f}%'
            else:
                s += f' {"":21s}'
        print(s)
    print(f'  (uniform: ring {ring.mean()*100:.1f}%  f0-2 {3/T*100:.1f}%)')
    json.dump({n: {str(k): list(vv) for k, vv in d.items()} for n, d in rows.items()},
              open(osp.join(args.out_dir, f'genlayers_{args.scene.split("/")[-1]}.json'), 'w'),
              ensure_ascii=False, indent=1)


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
    p.add_argument('--cell_w', type=int, default=320)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--out_fps', type=int, default=6)
    p.add_argument('--mark', dest='mark', action='store_true', default=True)
    p.add_argument('--no_mark', dest='mark', action='store_false')
    p.add_argument('--out_dir', default=OUT)
    p.add_argument('--gpu', default='2')
    main(p.parse_args())
