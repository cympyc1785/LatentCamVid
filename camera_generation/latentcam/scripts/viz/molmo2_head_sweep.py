"""**층 x head** sweep — prefill attention 이 위치를 짚는가. 기준은 OBB 투영 (D165).

  python scripts/viz/molmo2_head_sweep.py --stage prefill --gpu 4
  python scripts/viz/molmo2_head_sweep.py --stage prefill,gen --gpu 4

왜 (2026-09-11): D164 의 "prefill 은 위치를 못 짚고 생성 단계만 짚는다" 가 **문헌과 반대**다.

  · MLLMs Know When Before Speaking (arXiv 2605.21954, video, Qwen3-VL)
    "during prefill, a **sparse set of attention heads** concentrates query-to-video attention
     on the ground-truth interval. During autoregressive decoding, however, the answer tokens
     shift attention away from this interval."  -> prefill 이 맞고 decoding 이 망가진다
  · Beyond Static Cropping (arXiv 2602.04304) 는 attention 을 **prefill 에서만** 계산한다
    ("computing attention solely at the prefill stage is sufficient"), LLaVA-1.5 최적 층 14/32
  · Analysis-by-Proxy (arXiv 2607.06445) 는 Q-Former head 가 "decisive router" 로 층을 고른다

공통 열쇠가 **head 희소성**이다 (TG-Heads 상위 5개). 그런데 D164 의 prefill 측정
(`molmo2_latent_sweep.py` 의 `attn` readout, `molmo2_attn_grid.py`) 은 전부 **32 head 평균**이다.
신호가 3~5 head 에 있으면 평균이 그것을 지운다. 이 스크립트가 그 가설을 판정한다.

D164 와 무엇이 다른가
  ① head 축을 **안 접는다**. `_hook` 이 이미 `(H,R,L)` 을 주고 `maps_all_layers` 가 마지막에
     평균할 뿐이라, 그 평균을 안 하면 끝이다.
  ② 기준이 **생성 좌표가 아니라 OBB 투영**이다. `molmo2_attn_gen_2d.py` 는 attention argmax 를
     모델 자신이 뱉은 좌표와 비교했다 (자기 대조) — 그러면 prefill 에는 비교할 좌표가 없어서
     두 단계를 같은 자 위에 못 올린다. OBB 투영은 두 단계에 공통이므로 **직접 비교**가 된다.
  ③ prefill·생성의 정규화 규약을 **맞췄다** — 둘 다 `(head, row/step)` 마다 video patch 위에서
     먼저 재정규화한 뒤 평균한다. D164 의 생성 경로는 훅 안에서 head 평균을 먼저 했다.

지표: 프레임 f 의 9x9 슬라이스 argmax 셀 중심 <-> OBB 투영 (u,v) 픽셀 거리.
  prefill  프레임 질의가 없으므로 **모든 유효 프레임**에 대해 각각 슬라이스 argmax 를 잰다
  gen      좌표 그룹 g 의 자릿수 스텝 평균 -> 그 그룹이 가리키는 프레임 f_g 의 슬라이스

기준선: 화면 중심 / 균등 격자 기대거리 / 1셀 반경(양자화 하한). 이 셋을 못 넘으면 신호가 없다.

산출물: `<out_dir>/head_sweep_<tag>.json` + 콘솔 표 (층별 최고 head / 상위 k head 평균 /
        32-head 평균 = D164 재현치)

env: `latentcam`.  GPU 1장. prefill 은 anchor 당 forward 1회라 빠르고, gen 은 eager +
     200 스텝이라 느리다 — 그래서 `--stage` 로 나눈다.
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import sys

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
from molmo2_attn_video import Runner, OUT, CORPUS                # noqa: E402  (OUT 은 기본값용)
from molmo2_vs_obb import project, scene_anchors                 # noqa: E402
from molmo2_attn_gen_2d import parse_groups_xy                   # noqa: E402
import cache_molmo2_embeddings as C                              # noqa: E402


# ------------------------------------------------------------------ prefill (head 보존)

@torch.inference_mode()
def prefill_maps_heads(r, text, span, probe=''):
    """(nl, H, T, side, side) + (nl, H) video 질량. `maps_all_layers` 와 **같은 경로**이고
    마지막 head 평균만 안 한다. 정규화도 같다 — `(head, row)` 마다 video patch 위에서 먼저
    재정규화한 뒤 row 축만 평균한다."""
    v, core = r.vid, r.model.model
    tstr = r._tail_str(text, probe)
    tids = r.proc.tokenizer(tstr, add_special_tokens=False,
                            return_tensors='pt')['input_ids']
    rows, _ = r._rows(tstr, span, 0)
    P, T, side = v['P'], v['T'], v['side']
    with r.lock:
        temb, _ = core.build_input_embeddings(tids.to(r.device))
        emb = torch.cat([v['prefix_emb'], temb], 1)
        store = {}
        restore = r._hook(None, [P + i for i in rows], store)
        try:
            core(inputs_embeds=emb,
                 attention_mask=torch.ones(1, emb.shape[1], dtype=torch.long,
                                           device=r.device),
                 use_cache=False)
        finally:
            restore()
    assert store, 'attention 포획 실패'
    H = store[min(store)].shape[0]
    out = np.zeros((r.nl, H, T, side, side), np.float32)
    vmass = np.zeros((r.nl, H), np.float32)
    for li, w in store.items():
        vv = w[:, :, v['patch_pos']]                       # (H,R,3969)
        vm = vv.sum(-1)                                    # (H,R)
        vmass[li] = vm.mean(1).numpy()
        vn = vv / vm.clamp_min(1e-9)[..., None]
        out[li] = vn.mean(1).view(H, T, side, side).numpy()
    return out, vmass, len(rows)


# ------------------------------------------------------------------ generation (head 보존)

def hook_generate_heads(model, patch_pos, store):
    """매 forward 의 마지막 query 행 x patch 열을 **head 별로** 모은다 -> store[step][layer]
    = (H, 3969) fp16. 스텝당 36x32x3969x2B = 9.1 MB 라 `--max_new_tokens` 로 상한을 둔다."""
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
            a = w[0, :, -1, :]                                       # (H, kv_len)
            store.setdefault(st['step'], {})[li] = \
                a[:, pp.to(a.device)].float().cpu().half()           # (H, 3969)
        return out, w

    mod.eager_attention_forward = wrapped
    cfg = model.model.transformer.blocks[0].self_attn.config
    prev = cfg._attn_implementation
    cfg._attn_implementation = 'eager'

    def restore():
        mod.eager_attention_forward = orig
        cfg._attn_implementation = prev
    return restore


@torch.inference_mode()
def gen_maps_heads(r, text, fps, max_new_tokens):
    """생성 -> [(frame, (nl,H,side,side))], raw. prefill 과 같은 정규화 규약:
    스텝마다 head 별로 patch 위에서 재정규화한 뒤 그룹의 스텝 축만 평균한다."""
    v = r.vid
    T, side = v['T'], v['side']
    b = r._batch(v['video'], text, '', fps)
    b = {k: (x.to(r.device, r.dtype) if k == 'pixel_values_videos' else x.to(r.device))
         for k, x in b.items()}
    store = {}
    restore = hook_generate_heads(r.model, v['patch_pos'].numpy(), store)
    try:
        g = r.model.generate(**b, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        restore()
    gen = g[0, b['input_ids'].shape[1]:].tolist()
    toks = [r.proc.tokenizer.decode([t]) for t in gen]
    raw = r.proc.tokenizer.decode(gen, skip_special_tokens=True)
    try:
        G = parse_groups_xy(toks, raw)
    except AssertionError:
        return [], raw
    outs = []
    for t, _x1k, _y1k, steps in G:
        f = int(np.clip(round(t * fps), 0, T - 1))
        acc, n = None, 0
        for s in steps:
            if s not in store:
                continue
            for li, a in store[s].items():
                if acc is None:
                    acc = np.zeros((r.nl, a.shape[0], side * side), np.float32)
                af = a.float().numpy()                          # (H,3969)
                af = af / np.maximum(af.sum(-1, keepdims=True), 1e-9)
                acc[li] += af.reshape(a.shape[0], T, side * side)[:, f]
            n += 1
        if acc is not None and n:
            outs.append((f, (acc / n).reshape(r.nl, -1, side, side)))
    del store
    return outs, raw


# ------------------------------------------------------- time-invariant (시간 marginal)

def stage_tinv(r, args, scene, E, K, anch, sx, sy, W, H, T, side):
    """prefill attention 을 **시간축으로 평균해 9x9 한 장**으로 보고, 텍스트 교체로 판정한다.

    왜 이 형태여야 하나 (사용자 지시 2026-09-11): prefill 은 (49프레임 x 81셀) 위의 분포가
    **하나**이고 프레임 질의가 없다. 그래서 "프레임마다 물체를 따라가나" 는 물을 수 없는
    질문이었고 (D165 실측: 16 anchor 중 15개가 고정셀 통제 이하), 물어야 할 건
    **시간 marginal 이 물체 위에 앉나** 다. `molmo2_attn_map.py:496` 의 `sp = m.sum(0)` 가
    이미 이 양을 통계로만 쓰고 있었다 — 여기서는 판정에 쓴다.

    **절대거리로 재면 퇴화한다.** 타깃이 시간 중앙값 한 점이면 그 셀을 가리키는 고정셀이
    자동 정답이라 통제가 사라진다. 그래서 통제를 **텍스트 교체**로 둔다:

        같은 영상 · 같은 경로, 텍스트만 A/B
        M_A 가 cell(A) 에 M_B 보다 많은 질량을 주는가?
        텍스트와 무관한 saliency 라면 M_A == M_B 이므로 정확히 50% 로 떨어진다.

    (i,j)/(j,i) 를 둘 다 세고 동점을 0.5 로 처리하는 `discriminate` 와 같은 규약이다.

    산출물: 씬마다 PNG 한 장. 행 = anchor, 열 = [frame0+OBB, 최고 (층,head) marginal,
    32head 평균맵 marginal]. 텍스트를 바꿀 때 히트맵이 옮겨가는지 한 화면에서 보인다.
    """
    from PIL import Image, ImageDraw
    from molmo2_attn_video import colorize
    M, cen, lbls, txts = {}, {}, {}, {}
    for a, key, txt, lab, tw in anch:
        if args.anchor and a != args.anchor:
            continue
        uv, _z = project(tw, E, K)
        uv = uv * np.array([sx, sy])
        ok = [f for f in range(min(T, uv.shape[0]))
              if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W and 0 <= uv[f, 1] < H]
        if len(ok) < args.min_frames:
            continue
        mp, _vm, nq = prefill_maps_heads(r, f'{args.verb} {txt}', txt)
        M[a] = mp.mean(2)                                   # (nl,Hh,side,side) 시간 marginal
        cen[a] = (float(np.median(uv[ok, 0])), float(np.median(uv[ok, 1])))
        lbls[a], txts[a] = lab, txt
        del mp
    if len(M) < 2:
        print(f'  [tinv] anchor {len(M)}개 — 텍스트 교체 통제가 불가능하다')
        return []
    ids = sorted(M)
    nl, Hh = M[ids[0]].shape[:2]

    # ── 텍스트 교체 판별 (시간 marginal 위에서). same/diff 버킷도 같이
    buckets = {k: [np.zeros((nl, Hh)), 0] for k in ('all', 'same', 'diff')}
    bmean = {k: [np.zeros(nl), 0] for k in ('all', 'same', 'diff')}
    for i in ids:
        for j in ids:
            if i == j:
                continue
            ci = cell_of(*cen[i], side, W, H)
            cj = cell_of(*cen[j], side, W, H)
            if ci == cj:
                continue
            bk = 'same' if lbls[i] == lbls[j] else 'diff'
            mi = M[i][:, :, ci[0], ci[1]]
            mj = M[i][:, :, cj[0], cj[1]]
            hit = (mi > mj).astype(np.float64) + 0.5 * (mi == mj)
            hm = M[i].mean(1)
            hitm = ((hm[:, ci[0], ci[1]] > hm[:, cj[0], cj[1]]).astype(np.float64)
                    + 0.5 * (hm[:, ci[0], ci[1]] == hm[:, cj[0], cj[1]]))
            for key in ('all', bk):
                buckets[key][0] += hit
                buckets[key][1] += 1
                bmean[key][0] += hitm
                bmean[key][1] += 1

    print(f'\n  [tinv] {scene.split("/")[-1]}  anchor {len(ids)} '
          f'({", ".join(lbls[a] for a in ids)})')
    for key in ('all', 'same', 'diff'):
        c, t = buckets[key]
        cm, _ = bmean[key]
        if not t:
            continue
        se = float(np.sqrt(0.25 / t) * 100)
        A, Am = c / t * 100, cm / t * 100
        bi = int(Am.argmax())
        fl = sorted(((float(A[li, h]), li, h) for li in range(nl) for h in range(Hh)),
                    reverse=True)[:3]
        print(f'    [{key}] 시행 {t:3d}  1 s.e. {se:.1f}%p   '
              f'32head맵 최고 L{bi:02d} {Am[bi]:5.1f}% ({(Am[bi] - 50) / se:+.1f} se)   '
              f'개별 최고 L{fl[0][1]:02d}h{fl[0][2]:02d} {fl[0][0]:5.1f}% '
              f'({(fl[0][0] - 50) / se:+.1f} se)')

    # ── 그림: 행 = anchor, 열 = [frame0+OBB, 최고 (층,head), 32head 평균맵 최고층]
    c_all, t_all = buckets['all']
    m_all = bmean['all'][0] / max(bmean['all'][1], 1) * 100
    bl = int(m_all.argmax())
    bv, bli, bh = sorted(((float(c_all[li, h] / t_all), li, h)
                          for li in range(nl) for h in range(Hh)), reverse=True)[0]
    cw = args.cell_w
    chh = cw * H // W
    cols = 3
    canvas = np.zeros((len(ids) * chh, cols * cw, 3), np.uint8)

    def overlay(fr, m9, mark, mark2=None):
        h_, w_ = fr.shape[:2]
        up = np.asarray(Image.fromarray(
            (np.clip(m9 / max(float(m9.max()), 1e-12), 0, 1) * 255).astype(np.uint8)
        ).resize((w_, h_), Image.BILINEAR)) / 255.0
        al = (up[..., None] ** 0.6) * args.alpha
        o = fr.astype(np.float32) / 255.0 * (1 - al) + colorize(up) * al
        for pt, col, rad in ((mark, [0, 1, 0], 7), (mark2, [0, 1, 1], 5)):
            if pt is None:
                continue
            cx, cy = int(pt[0] * w_ / W), int(pt[1] * h_ / H)
            if 0 <= cx < w_ and 0 <= cy < h_:
                o[max(0, cy - 1):cy + 1, max(0, cx - rad):cx + rad] = col
                o[max(0, cy - rad):cy + rad, max(0, cx - 1):cx + 1] = col
        return (np.clip(o, 0, 1) * 255).astype(np.uint8)

    fr0 = np.asarray(Image.fromarray(r.vid['video'][0]).resize((cw, chh), Image.BILINEAR))
    for ri, a in enumerate(ids):
        flat0 = np.zeros((side, side), np.float32)
        for ci_, (nm_, m9) in enumerate((
                ('frame0 + OBB', flat0),
                (f'L{bli:02d}h{bh:02d} marginal', M[a][bli, bh]),
                (f'L{bl:02d} 32head marginal', M[a][bl].mean(0)))):
            am = np.unravel_index(int(np.argmax(m9)), (side, side))
            ap = (None if ci_ == 0 else
                  ((am[1] + .5) * W / side, (am[0] + .5) * H / side))
            canvas[ri * chh:(ri + 1) * chh, ci_ * cw:(ci_ + 1) * cw] = overlay(
                fr0, m9, cen[a], ap)
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    hdr = ['frame0 + OBB(green)', f'L{bli:02d}h{bh:02d} best head',
           f'L{bl:02d} 32head mean']
    for ri, a in enumerate(ids):
        for ci_ in range(cols):
            dr.rectangle([ci_ * cw + 1, ri * chh + 1, ci_ * cw + 190, ri * chh + 14],
                         fill=(0, 0, 0))
            dr.text((ci_ * cw + 3, ri * chh + 3),
                    (f'{lbls[a]} | {hdr[ci_]}' if ci_ == 0 else hdr[ci_]),
                    fill=(255, 255, 255))
        dr.rectangle([2, (ri + 1) * chh - 15, 2 + 8 * min(len(txts[a]), 60),
                      (ri + 1) * chh - 2], fill=(0, 0, 0))
        dr.text((4, (ri + 1) * chh - 13), txts[a][:60], fill=(255, 255, 0))
    fp = osp.join(args.out_dir, f'tinv_{scene.split("/")[-1]}.png')
    im.save(fp)
    print(f'    [save] {fp}')
    return [dict(scene=scene, anchors=ids, labels=[lbls[a] for a in ids],
                 texts=[txts[a] for a in ids],
                 # 원시 카운트를 그대로 넘긴다 — 씬별 12~56 시행으로는 결론이 안 나므로
                 # 최종 리포트에서 전 씬을 합치고 **씬 단위 LOO** 로 층을 고른다.
                 counts={k: dict(corr_lh=v[0].copy(), corr_m=bmean[k][0].copy(),
                                 trials=int(v[1]))
                         for k, v in buckets.items() if v[1]},
                 png=fp)]


# ------------------------------------------------------------------ 시각화

def stage_viz(r, args, scene, E, K, anch, sx, sy, W, H, T, side):
    """prefill attention 이 **가장 잘 나온** (층,head) 를 골라 영상으로 낸다.

    고르는 기준은 OBB 투영까지의 거리 중앙값이고, 대조로 같은 anchor 의 **고정셀 oracle**
    과 **32head 평균맵 최고 층**을 같은 화면에 넣는다. 셋을 같이 봐야 "국소화"인지
    "그 셀에 앉아 있는 것"인지 눈으로 갈린다.

    초록 십자 = OBB 투영 (기준), 청록 십자 = 그 셀의 attention argmax.
    """
    import imageio.v2 as iio
    from PIL import Image, ImageDraw
    from molmo2_attn_gen_2d import cell as draw_cell
    out = []
    for a, key, txt, lab, tw in anch:
        if args.anchor and a != args.anchor:
            continue
        uv, _z = project(tw, E, K)
        uv = uv * np.array([sx, sy])
        ok = [f for f in range(min(T, uv.shape[0]))
              if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W and 0 <= uv[f, 1] < H]
        if len(ok) < args.min_frames:
            continue
        tg = [(uv[f, 0], uv[f, 1]) for f in ok]
        text = f'{args.verb} {txt}'
        mp, _vm, nq = prefill_maps_heads(r, text, txt)
        nl, Hh = mp.shape[0], mp.shape[1]
        dd = score([mp[:, :, f] for f in ok], tg, side, W, H)
        med = np.median(dd, 2)
        mm = score([mp[:, :, f].mean(1, keepdims=True) for f in ok], tg, side, W, H)[:, 0]
        med_m = np.median(mm, 1)
        fx, fxrc = best_fixed_cell(tg, side, W, H)
        flat = sorted(((float(med[li, h]), li, h)
                       for li in range(nl) for h in range(Hh)))[:args.viz_k]
        bl = int(med_m.argmin())
        print(f'\n  [{a} {lab}] {txt!r}  q{nq} tok  n{len(ok)} frames')
        print(f'    고정셀 oracle {fx:.1f}px (cell {fxrc})   '
              f'32head 평균맵 최고 L{bl:02d} {med_m[bl]:.1f}px')
        for v, li, h in flat:
            print(f'    L{li:02d} h{h:02d}  {v:6.1f}px   margin vs 고정셀 '
                  f'{fx / max(v, 1e-6):.2f}x')

        # 셀 구성: 상위 k (층,head) + 32head 평균맵 최고 층 + 고정셀 대조
        cells = [(f'L{li:02d}h{h:02d} {v:.0f}px', mp[li, h]) for v, li, h in flat]
        cells.append((f'L{bl:02d} 32head {med_m[bl]:.0f}px', mp[bl].mean(0)))
        fxmap = np.zeros((T, side, side), np.float32)
        fxmap[:, fxrc[0], fxrc[1]] = 1.0
        cells.append((f'FIXED CELL {fx:.0f}px', fxmap))

        cw = args.cell_w
        chh = cw * H // W
        ncol = min(args.ncol, len(cells))
        nrow = int(np.ceil(len(cells) / ncol))
        small = [np.asarray(Image.fromarray(r.vid['video'][f]).resize(
            (cw, chh), Image.BILINEAR)) for f in range(T)]
        tagn = f'{scene.split("/")[-1]}_{a}'
        fp = osp.join(args.out_dir, f'prefill_best_{tagn}.mp4')
        wr = iio.get_writer(fp, fps=args.out_fps, codec='libx264', quality=8,
                            macro_block_size=1)
        for f in range(T):
            gp = ((uv[f, 0] * 640 / W, uv[f, 1] * 360 / H)
                  if not np.isnan(uv[f, 0]) else (-99, -99))
            cv = np.zeros((nrow * chh, ncol * cw, 3), np.uint8)
            for i_, (nm_, m3) in enumerate(cells):
                rc = np.unravel_index(int(np.argmax(m3[f])), (side, side))
                ap = ((rc[1] + .5) * 640 / side, (rc[0] + .5) * 360 / side)
                rr, cc = divmod(i_, ncol)
                cv[rr * chh:(rr + 1) * chh, cc * cw:(cc + 1) * cw] = draw_cell(
                    small[f], m3[f], args.alpha, gp, ap)
            im = Image.fromarray(cv)
            dr = ImageDraw.Draw(im)
            for i_, (nm_, _m) in enumerate(cells):
                rr, cc = divmod(i_, ncol)
                dr.rectangle([cc * cw + 1, rr * chh + 1, cc * cw + 122, rr * chh + 14],
                             fill=(0, 0, 0))
                dr.text((cc * cw + 3, rr * chh + 3), nm_, fill=(255, 255, 255))
            dr.rectangle([2, nrow * chh - 16, 560, nrow * chh - 2], fill=(0, 0, 0))
            dr.text((4, nrow * chh - 14),
                    f'{lab} | frame {f:02d} | green=OBB  cyan=attn argmax | {txt[:44]}',
                    fill=(255, 255, 255))
            wr.append_data(np.asarray(im))
        wr.close()
        print(f'    [save] {fp}')
        out.append(dict(scene=scene, anchor=a, label=lab, text=txt, n=len(ok),
                        fixed_px=fx, fixed_cell=list(fxrc),
                        headmean_best=dict(layer=bl, px=float(med_m[bl])),
                        top=[dict(layer=li, head=h, px=v,
                                  margin=float(fx / max(v, 1e-6)))
                             for v, li, h in flat], mp4=fp))
        del mp
    return out


# ------------------------------------------------------------------ 채점

def score(maps, targets, side, W, H):
    """maps: (nl,Hh,side,side) 리스트, targets: 같은 길이의 (u,v) 리스트.
    반환 dist (nl, Hh, n) 픽셀거리."""
    nl, Hh = maps[0].shape[0], maps[0].shape[1]
    d = np.zeros((nl, Hh, len(maps)), np.float32)
    flat_cx = (np.arange(side) + .5) * W / side
    flat_cy = (np.arange(side) + .5) * H / side
    for i, (m, (u, vv)) in enumerate(zip(maps, targets)):
        idx = m.reshape(nl, Hh, -1).argmax(-1)                  # (nl,Hh)
        rr, cc = np.divmod(idx, side)
        d[:, :, i] = np.hypot(flat_cx[cc] - u, flat_cy[rr] - vv)
    return d


def best_fixed_cell(targets, side, W, H):
    """**결정적 통제.** attention 을 전혀 안 쓰고 9x9 셀 하나를 고정으로 가리켰을 때의
    최소 거리 중앙값 (81셀 oracle).

    왜 이게 필요한가: 층 x head = 1,152 개 후보 중 최소를 **같은 데이터에서** 고르면,
    셀이 81개뿐이라 "우연히 맞는 셀에 앉은 head" 가 반드시 하나는 나온다. 그 값이 이
    oracle 을 못 넘으면 head 가 찾아낸 정보는 0 이다 — 위치를 짚은 게 아니라 그 anchor 가
    한 셀 안에 머물렀을 뿐이다. D164 의 `fixed 17px` 기준선과 같은 뜻이고, 여기서는
    anchor 마다 따로 잰다."""
    cx = (np.arange(side) + .5) * W / side
    cy = (np.arange(side) + .5) * H / side
    u = np.array([t[0] for t in targets]); v = np.array([t[1] for t in targets])
    best, arg = float('inf'), None
    for rr in range(side):
        for cc in range(side):
            m = float(np.median(np.hypot(cx[cc] - u, cy[rr] - v)))
            if m < best:
                best, arg = m, (rr, cc)
    return best, arg


def cell_of(u, v, side, W, H):
    """픽셀 -> 9x9 셀 (r, c)."""
    return (int(np.clip(v * side / H, 0, side - 1)),
            int(np.clip(u * side / W, 0, side - 1)))


def discriminate(maps_by_anchor, uv_by_anchor, side, W, H, labels=None):
    """**고정셀 교란에 면역인 지표.** 같은 씬의 anchor 쌍에 대한 2지 판별 정확도.

    왜 이게 필요한가: 절대 거리 지표는 피사체가 한 셀 안에 머무는 anchor 에서 무력하다
    (고정셀 oracle 이 18~32px 라 상수를 가리켜도 같은 점수). 그런데 우리 설계가 Molmo2 에
    요구하는 건 절대 정밀도가 아니라 **"텍스트가 가리키는 물체를 다른 물체와 구별하는가"** 다.

    식: anchor i 의 텍스트로 얻은 맵 `A_i` 에 대해, 프레임 f 에서
            m_ii = A_i[cell(OBB_i(f))]      m_ij = A_i[cell(OBB_j(f))]
        `m_ii > m_ij` 면 정답. **(i,j) 와 (j,i) 를 둘 다 세므로 chance 가 정확히 50%다** —
        맵이 상수면 한 순서에서 맞고 반대 순서에서 틀려 반드시 50% 로 떨어진다. argmax 를
        안 쓰고 셀 질량을 직접 비교하므로 양자화 손실도 없다.

    두 OBB 가 **같은 셀**인 프레임은 제외한다 (구별할 게 없다).

    `labels` 를 주면 **동종/이종을 나눠** 집계한다. 이게 필요한 이유: 같은 씬의 쌍에는
    `camel <-> fence` (이종, 쉽다) 와 `camel <-> camel` (동종, 어렵다) 가 섞여 있다. 우리
    캡션은 인스턴스를 크기·위치 형용사로만 가르므로 (`the larger pale camel` vs
    `the smaller pale camel`) **동종 정확도가 진짜 시험**이고, 합쳐 놓으면 동종 실패가
    이종 성공에 묻힌다. 실측상 camel/fence/human/moss/building 쌍이 전부 동종으로 들어온다.

    반환: dict(bucket -> (correct (nl,Hh), total)). bucket = 'all' | 'same' | 'diff'.
    """
    ids = sorted(maps_by_anchor)
    nl, Hh = maps_by_anchor[ids[0]].shape[:2]
    buckets = {k: [np.zeros((nl, Hh), np.float64), 0] for k in ('all', 'same', 'diff')}
    for i in ids:
        for j in ids:
            if i == j:
                continue
            bk = None
            if labels is not None:
                bk = 'same' if labels.get(i) == labels.get(j) else 'diff'
            Ai, ui, uj = maps_by_anchor[i], uv_by_anchor[i], uv_by_anchor[j]
            for f in range(Ai.shape[2]):
                if (np.isnan(ui[f, 0]) or np.isnan(uj[f, 0])
                        or not (0 <= ui[f, 0] < W and 0 <= ui[f, 1] < H)
                        or not (0 <= uj[f, 0] < W and 0 <= uj[f, 1] < H)):
                    continue
                ci = cell_of(ui[f, 0], ui[f, 1], side, W, H)
                cj = cell_of(uj[f, 0], uj[f, 1], side, W, H)
                if ci == cj:
                    continue
                mi = Ai[:, :, f, ci[0], ci[1]]
                mj = Ai[:, :, f, cj[0], cj[1]]
                # **동점은 0.5.** 오답으로 세면 두 셀 질량이 모두 0 인 맵(상수 맵 포함)이
                # 50% 가 아니라 0% 가 되어 chance 보장이 깨진다 (단위 테스트로 확인).
                hit = (mi > mj).astype(np.float64) + 0.5 * (mi == mj)
                buckets['all'][0] += hit
                buckets['all'][1] += 1
                if bk:
                    buckets[bk][0] += hit
                    buckets[bk][1] += 1
    return {k: (v[0], v[1]) for k, v in buckets.items()}


def loo_eval(per_anchor):
    """held-out head 선택. `per_anchor`: anchor 당 (nl,Hh) 거리 중앙값 행렬 리스트.

    anchor i 를 빼고 나머지에서 (층,head) 하나를 고른 뒤 **i 에서** 평가한다. 이게
    "고정 head 집합이 새 씬에 일반화하나" 의 정직한 숫자다 — per-anchor 최고값은 후보
    1,152개를 평가 데이터로 고른 oracle 이라 상한일 뿐이다."""
    A = np.stack(per_anchor)                       # (n_anchor, nl, Hh)
    n = A.shape[0]
    if n < 2:
        return None
    out = []
    for i in range(n):
        rest = np.delete(A, i, axis=0).mean(0)     # 나머지 anchor 평균
        li, h = np.unravel_index(int(rest.argmin()), rest.shape)
        out.append((float(A[i, li, h]), int(li), int(h)))
    picks = {}
    for _px, li, h in out:
        picks[(li, h)] = picks.get((li, h), 0) + 1
    return dict(px_median=float(np.median([x[0] for x in out])),
                px_mean=float(np.mean([x[0] for x in out])),
                per_anchor=[dict(px=x[0], layer=x[1], head=x[2]) for x in out],
                pick_counts=[dict(layer=k[0], head=k[1], n=v)
                             for k, v in sorted(picks.items(), key=lambda kv: -kv[1])])


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    stages = [s.strip() for s in args.stage.split(',') if s.strip()]
    r = Runner(args.ckpt, args.out_dir)
    acc = {}          # stage -> dict(dist=(nl,Hh,n) 누적, dist_mean=(nl,n), meta=[])
    bl_all = {'center': [], 'grid': [], 'cell': []}
    skipped = []

    for scene in args.scenes.split(','):
        d = osp.join(args.root, scene, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            skipped.append((scene, 'target_track.npz 없음'))
            print(f'[skip] {scene}: target_track.npz 없음', flush=True)
            continue
        p = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p['extrinsics'], p['intrinsics']
        anch = scene_anchors(args.root, scene, args.max_per_kind)
        if not anch:
            skipped.append((scene, 'anchor 0'))
            continue
        r.set_video(scene=scene, start=0, num_frames=E.shape[0], fps=args.fps,
                    root=args.root)
        v = r.vid
        T, side = v['T'], v['side']
        H, W = v['video'].shape[1], v['video'].shape[2]
        # pose 픽셀 규약 -> 영상 픽셀 규약 배율 (intrinsics cx,cy 가 640x360 기준)
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        sx, sy = W / pw, H / ph
        cellr = float(np.hypot(W / side, H / side)) / 2
        print(f'\n### {scene}  anchors {len(anch)}  T {T}  {W}x{H}  '
              f'pose {pw:.0f}x{ph:.0f} (scale {sx:.3f},{sy:.3f})  1셀반경 {cellr:.0f}px',
              flush=True)

        if 'tinv' in stages:
            tv = stage_tinv(r, args, scene, E, K, anch, sx, sy, W, H, T, side)
            acc.setdefault('tinv', dict(items=[]))['items'].extend(tv)
            if len(stages) == 1:
                continue

        if 'viz' in stages:
            vz = stage_viz(r, args, scene, E, K, anch, sx, sy, W, H, T, side)
            acc.setdefault('viz', dict(items=[]))['items'].extend(vz)
            if len(stages) == 1:
                continue

        disc_maps, disc_uv, disc_lab = {}, {}, {}   # 씬 안에서만 (교차 anchor 판별용)
        for a, key, txt, lab, tw in anch:
            uv, z = project(tw, E, K)
            uv = uv * np.array([sx, sy])
            ok = [f for f in range(min(T, uv.shape[0]))
                  if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W and 0 <= uv[f, 1] < H]
            if len(ok) < args.min_frames:
                print(f'  [{a:7s}] 유효 프레임 {len(ok)} < {args.min_frames} — 건너뜀')
                continue
            text = f'{args.verb} {txt}'
            tg = [(uv[f, 0], uv[f, 1]) for f in ok]
            bl_all['center'] += [np.hypot(u - W / 2, vv - H / 2) for u, vv in tg]
            ys, xs = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
            gp = np.stack([(xs + .5) * W / side, (ys + .5) * H / side], -1).reshape(-1, 2)
            bl_all['grid'] += [float(np.hypot(*(gp - np.array(t)).T).mean()) for t in tg]
            bl_all['cell'].append(cellr)
            fx_px, fx_rc = best_fixed_cell(tg, side, W, H)      # 81셀 oracle (attention 무관)

            if 'prefill' in stages or 'disc' in stages:
                spans = ((('prefill_target', txt), ('prefill_alltail', 'all'))
                         if 'prefill' in stages else (('prefill_target', txt),))
                for nm, span in spans:
                    mp, vm, nq = prefill_maps_heads(r, text, span)
                    if 'disc' in stages and nm == 'prefill_target':
                        disc_maps[a] = mp.copy()
                        disc_uv[a] = uv.copy()
                        disc_lab[a] = lab
                    if 'prefill' not in stages:
                        del mp
                        continue
                    maps = [mp[:, :, f] for f in ok]
                    dd = score(maps, tg, side, W, H)
                    mm = score([mp[:, :, f].mean(1, keepdims=True) for f in ok],
                               tg, side, W, H)[:, 0]           # 32-head 평균 = D164 규약
                    s = acc.setdefault(nm, dict(d=[], m=[], meta=[], lh=[]))
                    s['d'].append(dd); s['m'].append(mm)
                    s['lh'].append(np.median(dd, 2))            # (nl,Hh) anchor 별 — LOO 용
                    s['meta'].append(dict(scene=scene, anchor=a, label=lab, text=txt,
                                          n=len(ok), q_tokens=nq, fixed_px=fx_px,
                                          fixed_cell=list(fx_rc),
                                          oracle_px=float(np.median(dd, 2).min()),
                                          head_mean_px=float(np.median(mm, 1).min())))
                    best = np.median(dd, 2).min()
                    print(f'  [{a:7s} {lab:8s}] {nm:16s} q{nq:3d} n{len(ok):3d}  '
                          f'head평균 {np.median(mm, 1).min():6.1f}px  '
                          f'head oracle {best:6.1f}px  고정셀 oracle {fx_px:6.1f}px  '
                          f'{txt[:26]}', flush=True)
                    del mp, maps

            if 'gen' in stages:
                outs, raw = gen_maps_heads(r, text, args.fps, args.max_new_tokens)
                if not outs:
                    print(f'  [{a:7s}] 좌표 생성 실패: {raw.strip()[:70]!r}')
                else:
                    gf = [f for f, _ in outs]
                    keep = [i for i, f in enumerate(gf)
                            if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W
                            and 0 <= uv[f, 1] < H]
                    if keep:
                        maps = [outs[i][1] for i in keep]
                        tgg = [(uv[gf[i], 0], uv[gf[i], 1]) for i in keep]
                        dd = score(maps, tgg, side, W, H)
                        mm = score([outs[i][1].mean(1, keepdims=True) for i in keep],
                                   tgg, side, W, H)[:, 0]
                        # gen 은 프레임 부분집합이 달라 고정셀 oracle 도 그 부분집합에서
                        gx_px, gx_rc = best_fixed_cell(tgg, side, W, H)
                        s = acc.setdefault('gen', dict(d=[], m=[], meta=[], lh=[]))
                        s['d'].append(dd); s['m'].append(mm)
                        s['lh'].append(np.median(dd, 2))
                        s['meta'].append(dict(scene=scene, anchor=a, label=lab, text=txt,
                                              n=len(keep), groups=len(outs), raw=raw[:200],
                                              fixed_px=gx_px, fixed_cell=list(gx_rc),
                                              oracle_px=float(np.median(dd, 2).min()),
                                              head_mean_px=float(np.median(mm, 1).min())))
                        print(f'  [{a:7s} {lab:8s}] {"gen":16s} g{len(outs):3d} '
                              f'n{len(keep):3d}  head평균 '
                              f'{np.median(mm, 1).min():6.1f}px  head oracle '
                              f'{np.median(dd, 2).min():6.1f}px  고정셀 oracle '
                              f'{gx_px:6.1f}px', flush=True)
                    del outs

        # ── 교차 anchor 판별 (씬 단위). chance 가 구조적으로 정확히 50% 다
        if 'disc' in stages and len(disc_maps) >= 2:
            bk = discriminate(disc_maps, disc_uv, side, W, H, disc_lab)
            bkm = discriminate({k: v.mean(1, keepdims=True)
                                for k, v in disc_maps.items()},
                               disc_uv, side, W, H, disc_lab)   # D164 규약 (32head 평균맵)
            if bk['all'][1]:
                s_ = acc.setdefault('disc', dict(bk=[], bkm=[], meta=[]))
                s_['bk'].append(bk); s_['bkm'].append(bkm)
                s_['meta'].append(dict(scene=scene, anchors=sorted(disc_maps),
                                       labels=[disc_lab[x] for x in sorted(disc_maps)],
                                       trials={k: int(v[1]) for k, v in bk.items()}))
                msg = []
                for key in ('all', 'same', 'diff'):
                    c, t = bk[key]
                    cm = bkm[key][0]
                    if t:
                        msg.append(f'{key} n{t}: head평균 {cm.max() / t * 100:.1f}% '
                                   f'/ 개별 {c.max() / t * 100:.1f}%')
                print(f'  [disc] {len(disc_maps)} anchor  ' + '  |  '.join(msg)
                      + '  (chance 50%)', flush=True)
            else:
                print(f'  [disc] 유효 시행 0 — 두 OBB 가 항상 같은 셀', flush=True)
        disc_maps.clear(); disc_uv.clear(); disc_lab.clear()

    if not acc:
        raise SystemExit('채점할 게 없다')
    # `--stage viz` 는 anchor 루프를 건너뛰어 `bl_all` 이 비어 있다 -> `np.mean([])` 가
    # RuntimeWarning + nan 을 낸다. viz 는 기준선을 안 쓰므로 빈 값은 None 으로 둔다.
    bl = {k: (float(np.mean(v)) if len(v) else None) for k, v in bl_all.items()}
    out = dict(stage=args.stage, verb=args.verb, fps=args.fps, topk=args.topk,
               baseline=bl, skipped=skipped, per_stage={})
    for nm, s in acc.items():
        if nm == 'viz':
            out['per_stage'][nm] = s['items']
            continue
        if nm == 'tinv':
            items = s['items']
            print(f'\n{"=" * 104}')
            print(f'tinv  시간 marginal + 텍스트 교체 판별   씬 {len(items)}개')
            print(f'  텍스트와 무관한 saliency 면 M_A == M_B 이므로 **정확히 50%** 다.')
            print(f'  씬별 시행이 적어 (anchor 당 한 점) 전 씬을 합치고, 층 선택은')
            print(f'  **씬 단위 LOO** 로 한다 — 36층 중 최선을 같은 데이터에서 고르면 상한이다.')
            print(f'{"=" * 104}')
            dj = {}
            for key in ('all', 'same', 'diff'):
                have = [it for it in items if key in it['counts']]
                if not have:
                    continue
                N = sum(it['counts'][key]['trials'] for it in have)
                Cm = np.sum([it['counts'][key]['corr_m'] for it in have], axis=0)
                Clh = np.sum([it['counts'][key]['corr_lh'] for it in have], axis=0)
                Am, Alh = Cm / N * 100, Clh / N * 100
                se = float(np.sqrt(0.25 / N) * 100)
                bi = int(Am.argmax())
                # 씬 단위 LOO: 나머지 씬에서 층을 고르고 held-out 씬에서 평가
                loo_c, loo_n, picks = 0.0, 0, {}
                for i_, it in enumerate(have):
                    rest = [x for j_, x in enumerate(have) if j_ != i_]
                    if not rest:
                        continue
                    rn = sum(x['counts'][key]['trials'] for x in rest)
                    rc = np.sum([x['counts'][key]['corr_m'] for x in rest], axis=0)
                    li = int((rc / rn).argmax())
                    loo_c += float(it['counts'][key]['corr_m'][li])
                    loo_n += it['counts'][key]['trials']
                    picks[li] = picks.get(li, 0) + 1
                loo_pct = loo_c / loo_n * 100 if loo_n else float('nan')
                loo_se = float(np.sqrt(0.25 / loo_n) * 100) if loo_n else float('nan')
                print(f'\n  [{key}]  시행 {N}   1 s.e. {se:.2f}%p   '
                      f'유의경계(2 s.e.) {50 + 2 * se:.1f}%')
                print(f'    32head맵 최고 층 (oracle) : L{bi:02d}  {Am[bi]:5.1f}%  '
                      f'({(Am[bi] - 50) / se:+.1f} se)')
                print(f'    개별 (층,head) 최고 (oracle): {Alh.max():5.1f}%  '
                      f'({(Alh.max() - 50) / se:+.1f} se)')
                print(f'    **씬 LOO 층 선택**         : {loo_pct:5.1f}%  '
                      f'({(loo_pct - 50) / loo_se:+.1f} se)   '
                      f'뽑힌 층 {sorted(picks.items(), key=lambda kv: -kv[1])[:4]}')
                half = 18
                print(f'    {"층":>3s} {"32head맵":>9s}   {"층":>3s} {"32head맵":>9s}')
                for li in range(half):
                    r2 = li + half
                    ln = f'    {li:3d} {Am[li]:8.1f}%'
                    if r2 < len(Am):
                        ln += f'   {r2:3d} {Am[r2]:8.1f}%'
                    print(ln)
                dj[key] = dict(trials=int(N), se_pct=se,
                               headmean_per_layer_pct=[float(x) for x in Am],
                               oracle_layer=dict(layer=bi, pct=float(Am[bi])),
                               oracle_layer_head_pct=float(Alh.max()),
                               loo=dict(pct=float(loo_pct), se_pct=float(loo_se),
                                        trials=int(loo_n),
                                        picks=[dict(layer=k2, n=v2)
                                               for k2, v2 in sorted(picks.items(),
                                                                    key=lambda kv: -kv[1])]))
            out['per_stage']['tinv'] = dict(
                buckets=dj,
                scenes=[dict(scene=it['scene'], anchors=it['anchors'],
                             labels=it['labels'], texts=it['texts'],
                             trials={k: v['trials'] for k, v in it['counts'].items()},
                             png=it['png']) for it in items])
            continue
        if nm == 'disc':
            print(f'\n{"=" * 104}')
            print(f'disc  교차 anchor 2지 판별   씬 {len(s["bk"])}개   chance 50.0%')
            print(f'  (i,j)/(j,i) 를 둘 다 세고 동점을 0.5 로 처리하므로 **상수 맵은 정확히')
            print(f'  50%** 다 — 고정셀 전략이 하필 정답 셀에 앉아 있어도 우연 수준으로 떨어진다.')
            print(f'  same = 같은 라벨 쌍(camel<->camel). 우리 캡션이 실제로 감당하는 구별이다.')
            print(f'{"=" * 104}')
            dj = {}
            for key in ('all', 'same', 'diff'):
                C = np.stack([x[key][0] for x in s['bk']]).sum(0)
                Cm = np.stack([x[key][0] for x in s['bkm']]).sum(0)[:, 0]
                N = int(sum(x[key][1] for x in s['bk']))
                if not N:
                    continue
                A, Am = C / N * 100, Cm / N * 100
                se = float(np.sqrt(0.25 / N) * 100)
                bi = int(np.argmax(Am))
                fl = sorted(((float(A[li, h]), li, h) for li in range(A.shape[0])
                             for h in range(A.shape[1])), reverse=True)[:args.topk]
                print(f'\n  [{key}]  시행 {N}   1 s.e. = {se:.2f}%p   '
                      f'유의 경계(2 s.e.) {50 + 2 * se:.2f}%')
                print(f'    32head 평균맵 최고 : 층 {bi:2d}  {Am[bi]:5.1f}%  '
                      f'({(Am[bi] - 50) / se:+.1f} s.e.)   <- D164 규약')
                print(f'    개별 head 최고     : L{fl[0][1]:02d} h{fl[0][2]:02d}  '
                      f'{fl[0][0]:5.1f}%  ({(fl[0][0] - 50) / se:+.1f} s.e.)')
                print(f'    상위 {args.topk} head: ' + ', '.join(
                    f'L{li:02d}h{h:02d} {pv:.1f}%' for pv, li, h in fl))
                # 층별 32head 평균맵 정확도 (2열)
                col = [(li, Am[li], float(A[li].max()), int(A[li].argmax()))
                       for li in range(A.shape[0])]
                half = (len(col) + 1) // 2
                print(f'    {"층":>3s} {"32head맵":>9s} {"최고head":>9s} {"h":>3s}   '
                      f'{"층":>3s} {"32head맵":>9s} {"최고head":>9s} {"h":>3s}')
                for x, y in zip(col[:half], col[half:] + [None] * half):
                    ln = f'    {x[0]:3d} {x[1]:8.1f}% {x[2]:8.1f}% {x[3]:3d}'
                    if y:
                        ln += f'   {y[0]:3d} {y[1]:8.1f}% {y[2]:8.1f}% {y[3]:3d}'
                    print(ln)
                dj[key] = dict(trials=N, chance_se_pct=se,
                               headmean_per_layer_pct=[float(x) for x in Am],
                               headmean_best=dict(layer=bi, pct=float(Am[bi])),
                               per_layer_best_head=[dict(layer=x[0], pct=x[2], head=x[3])
                                                    for x in col],
                               top_heads=[dict(layer=li, head=h, pct=pv)
                                          for pv, li, h in fl],
                               acc_layer_head_pct=A.tolist())
            out['per_stage']['disc'] = dict(buckets=dj, scenes=s['meta'])
            continue
        d = np.concatenate(s['d'], axis=2)          # (nl,Hh,N)
        m = np.concatenate(s['m'], axis=1)          # (nl,N)
        nl, Hh, N = d.shape
        med_lh = np.median(d, 2)                    # (nl,Hh)
        med_m = np.median(m, 1)                     # (nl,)
        # 상위 topk head: 전역(층 무관) 중앙값 기준으로 뽑아 그 head 들만의 층별 중앙값
        flat = [(float(med_lh[li, h]), li, h) for li in range(nl) for h in range(Hh)]
        flat.sort()
        top = flat[:args.topk]
        print(f'\n{"=" * 100}')
        print(f'{nm}   샘플 {N}  ({nl}층 x {Hh}head)')
        print(f'기준선: 화면중심 {bl["center"]:.0f}px  균등격자 {bl["grid"]:.0f}px  '
              f'1셀반경 {bl["cell"]:.0f}px (= argmax 양자화 하한)')
        print(f'{"=" * 100}')
        print(f'{"층":>3s} {"32head평균":>11s} {"층내최고head":>13s} {"head":>5s} '
              f'{"층내상위3중앙":>14s}   '
              f'{"층":>3s} {"32head평균":>11s} {"층내최고head":>13s} {"head":>5s} '
              f'{"층내상위3중앙":>14s}')
        col = []
        for li in range(nl):
            order = np.argsort(med_lh[li])
            col.append((li, med_m[li], float(med_lh[li, order[0]]), int(order[0]),
                        float(np.median(med_lh[li, order[:3]]))))
        half = (nl + 1) // 2
        for x, y in zip(col[:half], col[half:] + [None] * half):
            s_ = (f'{x[0]:3d} {x[1]:10.1f}px {x[2]:12.1f}px {x[3]:5d} {x[4]:13.1f}px')
            if y:
                s_ += (f'   {y[0]:3d} {y[1]:10.1f}px {y[2]:12.1f}px {y[3]:5d} '
                       f'{y[4]:13.1f}px')
            print(s_)
        print(f'\n  32-head 평균 최고: 층 {int(np.argmin(med_m))}  '
              f'{float(med_m.min()):.1f}px   <- D164 가 쓴 규약')
        print(f'  개별 head 최고 {args.topk}개 (층, head, 중앙값):')
        for dv, li, h in top:
            print(f'    L{li:02d} h{h:02d}  {dv:6.1f}px')
        sel = [(li, h) for _dv, li, h in top]
        dsel = np.median(np.stack([d[li, h] for li, h in sel]).mean(0))
        print(f'  상위 {args.topk} head **맵 평균** 후 중앙값: {dsel:.1f}px')

        # ── 통제 둘. 이 셋을 같이 봐야 숫자가 뜻을 갖는다
        fx = np.array([x['fixed_px'] for x in s['meta']])
        orc = np.array([x['oracle_px'] for x in s['meta']])
        hm = np.array([x['head_mean_px'] for x in s['meta']])
        loo = loo_eval(s['lh'])
        print(f'\n  --- anchor 단위 ({len(fx)}개) ---')
        print(f'  32-head 평균         median {np.median(hm):6.1f}px')
        print(f'  head oracle (1152중) median {np.median(orc):6.1f}px   '
              f'<- 상한. 같은 데이터에서 고름')
        print(f'  고정셀 oracle (81중) median {np.median(fx):6.1f}px   '
              f'<- **통제**. attention 무관')
        win = float(np.mean(orc < fx)) * 100
        print(f'  head oracle 이 고정셀 oracle 을 이긴 anchor: {win:.0f}%  '
              f'(둘 다 oracle 이라 공정 비교)')
        if loo:
            print(f'  LOO 선택 (held-out)  median {loo["px_median"]:6.1f}px   '
                  f'<- **일반화하는 유일한 숫자**')
            print(f'    자주 뽑힌 (층,head): ' + ', '.join(
                f'L{p["layer"]:02d}h{p["head"]:02d}x{p["n"]}'
                for p in loo['pick_counts'][:5]))
        out['per_stage'][nm] = dict(
            n=int(N), n_layers=int(nl), n_heads=int(Hh),
            head_mean_per_layer=[float(x) for x in med_m],
            head_mean_best=dict(layer=int(np.argmin(med_m)), px=float(med_m.min())),
            per_layer_best_head=[dict(layer=x[0], px=x[2], head=x[3], top3=x[4])
                                 for x in col],
            top_heads=[dict(layer=li, head=h, px=dv) for dv, li, h in top],
            topk_mapmean_px=float(dsel),
            anchor_level=dict(head_mean_med=float(np.median(hm)),
                              head_oracle_med=float(np.median(orc)),
                              fixed_cell_oracle_med=float(np.median(fx)),
                              head_beats_fixed_pct=win),
            loo=loo,
            median_layer_head=med_lh.tolist(),
            per_anchor_layer_head=[x.tolist() for x in s['lh']],
            anchors=s['meta'])
    fp = osp.join(args.out_dir, f'head_sweep_{args.tag}.json')
    json.dump(out, open(fp, 'w'), ensure_ascii=False, indent=1)
    print(f'\n[out] {fp}')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--stage', default='prefill', help='prefill,gen 쉼표 구분')
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scenes', default=','.join(
        'vista4d/' + s for s in ('camel', 'car-roundabout', 'basketball-four', 'couple-walk',
                                 'golf', 'cows', 'couple-rocks', 'fashion-walk')))
    p.add_argument('--max_per_kind', type=int, default=2)
    p.add_argument('--min_frames', type=int, default=5)
    p.add_argument('--verb', default='Track')
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--topk', type=int, default=8)
    p.add_argument('--anchor', default='', help='viz: 이 anchor 만')
    p.add_argument('--viz_k', type=int, default=4)
    p.add_argument('--ncol', type=int, default=3)
    p.add_argument('--cell_w', type=int, default=320)
    p.add_argument('--alpha', type=float, default=0.65)
    p.add_argument('--out_fps', type=int, default=8)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--out_dir',
                   default='/data1/cympyc1785/LatentCamVid/tmp/d165_head')
    p.add_argument('--tag', default='prefill')
    p.add_argument('--gpu', default='4')
    main(p.parse_args())
