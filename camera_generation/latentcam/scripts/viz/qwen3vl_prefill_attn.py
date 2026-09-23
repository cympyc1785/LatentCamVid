"""Qwen3-VL 의 **prefill** attention 이 물체를 짚는가 — Molmo2 와 같은 자로 (D167).

  python scripts/viz/qwen3vl_prefill_attn.py --stage probe --gpu 4    # 토큰 배치 확인
  python scripts/viz/qwen3vl_prefill_attn.py --stage tinv  --gpu 4    # 쌍비교 + 그림

왜 (2026-09-12): D165 가 Molmo2-4B prefill 에서 이렇게 재놨다.

    argmax 절대거리       16 anchor 중 15개가 고정셀 통제 이하, 씬 LOO 174.0px  (실패)
    시간 marginal 쌍비교  씬 LOO **79.0%** (이종 84.2% / 동종 67.6%), chance 50%  (성공)
    per-frame 추적        구조적으로 불가 — prefill 에 프레임 질의가 없다

그리고 D166 이 MolmoPoint 에서 `patch_logits is None` (prefill 경로)로 **아키텍처 수준의
확증**을 얻었다. 남은 질문은 "이게 Molmo2 계열 특성인가, VLM 일반인가" 다.

Qwen3-VL 이 결정적 대조군인 이유: **MLLMs Know When Before Speaking** (arXiv 2605.21954)
이 정확히 Qwen3-VL 에서 반대 결과를 보고한다 —

    "during prefill, a **sparse set of attention heads** concentrates query-to-video
     attention on the ground-truth interval. During autoregressive decoding, however,
     the answer tokens shift attention away from this interval."

그 논문은 **상위 5 TG-Head** 를 500샘플로 골라 쓴다. 우리 D165 는 32-head 평균이었고,
head 선택을 해봤지만 (씬 LOO 174.0px) 절대거리 축에서는 안 살아났다. 여기서는 처음부터
head 축을 보존하고 **쌍비교 지표**(고정셀 교란에 면역)로 잰다.

## 격자 매핑이 Molmo2 와 다르다 — 그래서 `--stage probe` 가 먼저다

Molmo2      프레임당 9x9=81, `patch_pos` 가 프레임 순서대로 3969개  (단순)
Qwen3-VL    `video_grid_thw = (t, h, w)`, temporal_patch_size=2 로 **2프레임이 한 묶음**,
            공간은 merge_size=2 로 2x2 병합 -> 토큰 수 = t*(h/2)*(w/2)

즉 시간축이 프레임이 아니라 **2프레임 묶음**이고, 공간 격자도 9x9 가 아니다. 추측하면
프레임 매핑이 틀어지므로 probe 가 실측한 `video_grid_thw` 로 reshape 한다.

지표는 D165 와 **완전히 동일**하다 — 같은 8씬, 같은 anchor, 같은 262쌍, 시간 marginal,
씬 내 양방향, 동점 0.5, chance 정확히 50%. 그래야 79.0% 와 직접 비교된다.

env: `latentcam` (transformers 4.57.6 이 Qwen3VL 네이티브 지원).  GPU 1장.
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
sys.path.insert(0, osp.join(osp.dirname(osp.dirname(HERE)), 'latentcam', 'main'))
from molmo2_attn_video import CORPUS                             # noqa: E402
from molmo2_vs_obb import project, scene_anchors                 # noqa: E402
import cache_molmo2_embeddings as C                              # noqa: E402

CKPT = ('/data1/cympyc1785/LatentCamVid/camera_generation/tools/qwen3vl/'
        'Qwen3-VL-4B-Instruct')
SCENES = ('camel', 'car-roundabout', 'basketball-four', 'couple-walk',
          'golf', 'cows', 'couple-rocks', 'fashion-walk')


def load_model(ckpt, dtype=torch.bfloat16):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    import time
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        ckpt, dtype=dtype, device_map='auto', attn_implementation='eager')
    proc = AutoProcessor.from_pretrained(ckpt)
    model.eval()
    nl = model.config.text_config.num_hidden_layers
    nh = model.config.text_config.num_attention_heads
    print(f'[load] Qwen3-VL  {time.time() - t0:.1f}s  층 {nl}  head {nh}  '
          f'hidden {model.config.text_config.hidden_size}', flush=True)
    return model, proc, nl, nh


def build_inputs(proc, video, text, fps):
    """(inputs, grid_thw, video_token_positions, text_token_positions).

    `video=` 에 numpy (T,H,W,3) 를 그대로 준다. Qwen 은 `fps` 를 별도 kwarg 로 받는다.
    """
    msgs = [{'role': 'user',
             'content': [{'type': 'video', 'video': video},
                         {'type': 'text', 'text': text}]}]
    # **`video_metadata` 를 반드시 준다.** 안 주면 processor 가 "Defaulting to fps=24" 로
    # 솎아내 49프레임이 t=2 (440토큰) 로 줄어든다. 주면 t=10 (2200토큰) 이 된다 — 실측.
    T0 = int(np.asarray(video).shape[0])
    vmeta = [dict(fps=float(fps), total_num_frames=T0, duration=T0 / float(fps))]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=True, return_tensors='pt',
                                      video_metadata=vmeta)
    return inputs


def token_layout(model, proc, inputs):
    """video 토큰 위치와 격자. `video_grid_thw` 는 **merge 전** patch 격자다."""
    ids = inputs['input_ids'][0]
    vid_id = model.config.video_token_id
    pos = torch.nonzero(ids == vid_id).flatten()
    g = inputs['video_grid_thw'][0].tolist() if 'video_grid_thw' in inputs else None
    ms = getattr(proc.image_processor, 'merge_size', 2)
    return pos, g, ms


def hook_prefill(model, rows, store):
    """`eager_attention_forward` 를 감싸 지정 query 행만 층별로 떠낸다 (head 보존).

    Molmo2 쪽 `Runner._hook` 과 같은 방식이다 — `output_attentions=True` 는 전 층
    (head, L, L) 를 물려 메모리가 터진다. 여기서는 원본을 호출하고 **필요한 행만** 잘라
    `(H, R, L)` 로 CPU 에 받는다. 수치는 eager 와 동일하다.
    """
    import transformers.models.qwen3_vl.modeling_qwen3_vl as M
    orig = M.eager_attention_forward
    rt = torch.as_tensor(rows)

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = orig(module, query, key, value, attention_mask, scaling,
                      dropout=dropout, **kw)
        li = getattr(module, 'layer_idx', -1)
        if w is not None and w.shape[-2] > int(rt.max()):
            store[li] = w[0][:, rt.to(w.device), :].float().cpu()      # (H,R,L)
        return out, w

    M.eager_attention_forward = wrapped
    return lambda: setattr(M, 'eager_attention_forward', orig)


def cell_of(u, v, side_h, side_w, W, H):
    return (int(np.clip(v * side_h / H, 0, side_h - 1)),
            int(np.clip(u * side_w / W, 0, side_w - 1)))


@torch.inference_mode()
def prefill_marginal(model, proc, video, text, target, fps, nl):
    """(maps (nl, H, t, gh, gw) 시간 marginal 전 단계, grid, n_query_rows).

    query 행 = **target 명사구 토큰**. 문자열로 찾아 토큰 경계와 겹치는 위치를 고른다
    (Molmo2 `Runner._rows` 와 같은 규약).
    """
    inputs = build_inputs(proc, video, text, fps)
    ids = inputs['input_ids'][0]
    vid_id = model.config.video_token_id
    vpos = torch.nonzero(ids == vid_id).flatten()
    assert len(vpos), 'video 토큰을 못 찾았다'
    g = inputs['video_grid_thw'][0].tolist()
    ms = int(getattr(proc.image_processor, 'merge_size', 2))
    t_, gh, gw = g[0], g[1] // ms, g[2] // ms
    assert t_ * gh * gw == len(vpos), \
        f'격자 {t_}x{gh}x{gw}={t_ * gh * gw} != video 토큰 {len(vpos)}'

    # target 명사구 토큰 행 찾기
    toks = proc.tokenizer.convert_ids_to_tokens(ids.tolist())
    dec = [proc.tokenizer.convert_tokens_to_string([x]) for x in toks]
    off, spans = 0, []
    for d in dec:
        spans.append((off, off + len(d))); off += len(d)
    full = ''.join(dec)
    i = full.rfind(target)
    assert i >= 0, f'target 문자열을 프롬프트에서 못 찾았다: {target!r}'
    rows = [k for k, (a, b) in enumerate(spans)
            if b > i and a < i + len(target) and b > a]
    assert rows, 'target 이 토큰 경계와 안 맞는다'

    dev = next(model.parameters()).device
    inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    store = {}
    restore = hook_prefill(model, rows, store)
    try:
        model(**inputs, use_cache=False)
    finally:
        restore()
    assert store, 'attention 포획 실패'
    H = store[min(store)].shape[0]
    out = np.zeros((nl, H, t_, gh, gw), np.float32)
    for li, w in store.items():
        vv = w[:, :, vpos.cpu()]                       # (H,R,Nv)
        vm = vv.sum(-1)
        vn = vv / vm.clamp_min(1e-9)[..., None]        # (head,row) 마다 재정규화
        out[li] = vn.mean(1).view(H, t_, gh, gw).numpy()
    return out, (t_, gh, gw), len(rows)


def stage_viz(args):
    """Molmo2 `tinv_*.png` 와 **같은 레이아웃**으로 그린다 (행=anchor, 열 3개) +
    시간묶음별 타일.

    왜 같은 레이아웃인가: D165 의 `tinv_camel.png` 은 4행(서로 다른 텍스트)이 거의 같은
    그림이어서 "텍스트 무관 saliency 지배" 를 눈으로 보여줬다. Qwen3-VL 도 같은 틀에
    올려야 그 비교가 성립한다.

    시간 타일을 따로 내는 이유: Qwen3-VL 은 시퀀스에 `<N.N seconds>` 토큰이 박혀 있어
    Molmo2 에 없던 프레임 질의가 있다. 그래서 prefill 에서도 시간축으로 움직일 **가능성**이
    있고, 그건 시간 marginal 로는 보이지 않는다.
    """
    from PIL import Image, ImageDraw
    from molmo2_attn_video import colorize
    model, proc, nl, nh = load_model(args.ckpt)
    for scene in args.viz_scenes.split(','):
        sc = 'vista4d/' + scene.strip()
        d = osp.join(args.root, sc, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            print(f'[skip] {sc}')
            continue
        p_ = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p_['extrinsics'], p_['intrinsics']
        anch = scene_anchors(args.root, sc, args.viz_anchors)
        video = C.load_frames(args.root, sc, 0, args.num_frames)
        T, H_, W_ = video.shape[0], video.shape[1], video.shape[2]
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        sx, sy = W_ / pw, H_ / ph
        M, cen, lbls, txts, grid = {}, {}, {}, {}, None
        for a, key, txt, lab, tw in anch:
            uv, _z = project(tw, E, K)
            uv = uv * np.array([sx, sy])
            ok = [f for f in range(min(T, uv.shape[0]))
                  if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W_ and 0 <= uv[f, 1] < H_]
            if len(ok) < args.min_frames:
                continue
            mp, g, nq = prefill_marginal(model, proc, video, f'{args.verb} {txt}',
                                         txt, args.fps, nl)
            grid = g
            M[a] = mp                                   # (nl,H,t,gh,gw)
            cen[a] = (float(np.median(uv[ok, 0])), float(np.median(uv[ok, 1])))
            lbls[a], txts[a] = lab, txt
            print(f'  [{a:7s} {lab:12s}] q{nq} tok  {txt[:34]}', flush=True)
        if len(M) < 2:
            continue
        t_, gh, gw = grid
        ids = sorted(M)
        bl = args.viz_layer

        def overlay(fr, m, mark, mark2, gmax):
            h_, w_ = fr.shape[:2]
            up = np.asarray(Image.fromarray(
                (np.clip(m / max(gmax, 1e-12), 0, 1) * 255).astype(np.uint8)
            ).resize((w_, h_), Image.BILINEAR)) / 255.0
            al = (up[..., None] ** 0.6) * args.alpha
            o = fr.astype(np.float32) / 255.0 * (1 - al) + colorize(up) * al
            for pt, col, rad in ((mark, [0, 1, 0], 7), (mark2, [0, 1, 1], 5)):
                if pt is None:
                    continue
                cx, cy = int(pt[0] * w_ / W_), int(pt[1] * h_ / H_)
                if 0 <= cx < w_ and 0 <= cy < h_:
                    o[max(0, cy - 1):cy + 1, max(0, cx - rad):cx + rad] = col
                    o[max(0, cy - rad):cy + rad, max(0, cx - 1):cx + 1] = col
            return (np.clip(o, 0, 1) * 255).astype(np.uint8)

        # ── (1) Molmo2 와 같은 3열 레이아웃, 시간 marginal
        cw = args.cell_w
        chh = cw * H_ // W_
        fr0 = np.asarray(Image.fromarray(video[0]).resize((cw, chh), Image.BILINEAR))
        cv = np.zeros((len(ids) * chh, 3 * cw, 3), np.uint8)
        # 최고 head: 그 씬에서 판별이 가장 잘 되는 (bl, h) 대신 **질량 집중도**로 고르면
        # 텍스트 의존과 무관해진다 -> 대신 anchor 0 의 OBB 셀 확률이 가장 큰 head 를 쓴다
        a0 = ids[0]
        c0 = cell_of(*cen[a0], gh, gw, W_, H_)
        mh = M[a0][bl].mean(1)[:, c0[0], c0[1]]            # (H,)  시간 marginal 후
        bh = int(np.argmax(mh))
        for ri, a in enumerate(ids):
            marg = M[a][bl].mean(1)                        # (H,gh,gw)
            cells = [('frame0 + OBB', np.zeros((gh, gw), np.float32)),
                     (f'L{bl:02d}h{bh:02d}', marg[bh]),
                     (f'L{bl:02d} 32head', marg.mean(0))]
            for ci_, (nm_, m) in enumerate(cells):
                am = np.unravel_index(int(np.argmax(m)), (gh, gw)) if m.max() > 0 else None
                ap = (None if am is None else
                      ((am[1] + .5) * W_ / gw, (am[0] + .5) * H_ / gh))
                cv[ri * chh:(ri + 1) * chh, ci_ * cw:(ci_ + 1) * cw] = overlay(
                    fr0, m, cen[a], ap, max(m.max(), 1e-12))
        im = Image.fromarray(cv); dr = ImageDraw.Draw(im)
        hdr = ['frame0 + OBB(green)', f'L{bl:02d} h{bh:02d} best head', f'L{bl:02d} 32head mean']
        for ri, a in enumerate(ids):
            for ci_ in range(3):
                dr.rectangle([ci_ * cw + 1, ri * chh + 1, ci_ * cw + 175, ri * chh + 14],
                             fill=(0, 0, 0))
                dr.text((ci_ * cw + 3, ri * chh + 3),
                        (f'{lbls[a]} | {hdr[ci_]}' if ci_ == 0 else hdr[ci_]),
                        fill=(255, 255, 255))
            dr.rectangle([2, (ri + 1) * chh - 15, 2 + 8 * min(len(txts[a]), 62),
                          (ri + 1) * chh - 2], fill=(0, 0, 0))
            dr.text((4, (ri + 1) * chh - 13), txts[a][:62], fill=(255, 255, 0))
        fp = osp.join(args.out_dir, f'qwen_tinv_{scene.strip()}.png')
        im.save(fp)
        print(f'  [save] {fp}')

        # ── (2) 시간묶음별 타일 (anchor 0). 정규화는 **전역 최대**
        marg = M[a0][bl].mean(0)                           # (t,gh,gw) head 평균
        gmax = float(marg.max())
        ncol = 5
        nrow = int(np.ceil(t_ / ncol))
        cw2 = 240
        ch2 = cw2 * H_ // W_
        cv2 = np.zeros((nrow * ch2, ncol * cw2, 3), np.uint8)
        step = max(1, T // t_)
        for k in range(t_):
            f = min(T - 1, k * step)
            fr = np.asarray(Image.fromarray(video[f]).resize((cw2, ch2), Image.BILINEAR))
            rr, cc = divmod(k, ncol)
            uvk = cen[a0]
            cv2[rr * ch2:(rr + 1) * ch2, cc * cw2:(cc + 1) * cw2] = overlay(
                fr, marg[k], uvk, None, gmax)
        im2 = Image.fromarray(cv2); dr2 = ImageDraw.Draw(im2)
        for k in range(t_):
            rr, cc = divmod(k, ncol)
            dr2.rectangle([cc * cw2 + 1, rr * ch2 + 1, cc * cw2 + 128, rr * ch2 + 14],
                          fill=(0, 0, 0))
            dr2.text((cc * cw2 + 3, rr * ch2 + 3),
                     f't{k:02d} mass {marg[k].sum() / marg.sum() * 100:.1f}%',
                     fill=(255, 255, 255))
        fp2 = osp.join(args.out_dir, f'qwen_time_{scene.strip()}_{a0}.png')
        im2.save(fp2)
        print(f'  [save] {fp2}   (L{bl} 32head, 전역최대 정규화, 칸=시간묶음)')



# ------------------------------------------------- hidden state dump (probe 공용 포맷)

@torch.inference_mode()
def dump_hidden(model, proc, video, text, target, fps, layers):
    """(h_tgt (nL,C), h_vid (nL, t, gh*gw, C), grid). `output_hidden_states=True`.

    **`molmo2_probe.py` 가 읽는 것과 같은 npz 포맷**으로 내보내려는 것이다. 그래야 probe
    학습·평가 코드를 그대로 써서 모델 간 비교가 코드 차이 없이 성립한다. Qwen3-VL 격자는
    비정방(11x20)이라 `gh`/`gw` 를 npz 에 같이 싣고, probe 쪽은 그 키가 있으면 쓰도록
    일반화해 뒀다 (없으면 `side` 정방).
    """
    inputs = build_inputs(proc, video, text, fps)
    ids = inputs['input_ids'][0]
    vpos = torch.nonzero(ids == model.config.video_token_id).flatten()
    g = inputs['video_grid_thw'][0].tolist()
    ms = int(getattr(proc.image_processor, 'merge_size', 2))
    t_, gh, gw = g[0], g[1] // ms, g[2] // ms
    assert t_ * gh * gw == len(vpos), f'격자 {t_}x{gh}x{gw} != {len(vpos)}'
    toks = proc.tokenizer.convert_ids_to_tokens(ids.tolist())
    dec = [proc.tokenizer.convert_tokens_to_string([x]) for x in toks]
    off, spans = 0, []
    for dd_ in dec:
        spans.append((off, off + len(dd_))); off += len(dd_)
    i = ''.join(dec).rfind(target)
    assert i >= 0, f'target 못 찾음: {target!r}'
    rows = [k for k, (a, b) in enumerate(spans)
            if b > i and a < i + len(target) and b > a]
    assert rows, 'target 토큰 경계 불일치'
    dev = next(model.parameters()).device
    inputs = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in inputs.items()}
    o = model(**inputs, use_cache=False, output_hidden_states=True)
    hs = o.hidden_states                                  # (nl+1) x (1,L,C)
    ht = np.stack([hs[li + 1][0, rows].float().mean(0).cpu().numpy() for li in layers])
    hv = np.stack([hs[li + 1][0, vpos].float().cpu().numpy().reshape(t_, gh * gw, -1)
                   for li in layers])
    del o, hs
    return ht, hv, (t_, gh, gw), len(rows)


def stage_dump(args):
    model, proc, nl, nh = load_model(args.ckpt)
    layers = [int(x) for x in args.layers.split(',')]
    dd = osp.join(args.out_dir, 'probe_dump')
    makedirs(dd, exist_ok=True)
    for sc in args.scenes.split(','):
        fp = osp.join(dd, sc.split('/')[-1] + '.npz')
        if osp.exists(fp) and not args.overwrite:
            continue
        d = osp.join(args.root, sc, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            continue
        p_ = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p_['extrinsics'], p_['intrinsics']
        anch = scene_anchors(args.root, sc, args.max_per_kind)
        if not anch:
            continue
        video = C.load_frames(args.root, sc, 0, args.num_frames)
        T, H_, W_ = video.shape[0], video.shape[1], video.shape[2]
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        sx, sy = W_ / pw, H_ / ph
        hts, uvs, labs, txts, grid, hv0 = [], [], [], [], None, None
        for a, key, txt, lab, tw in anch:
            uv, _z = project(tw, E, K)
            uv = uv * np.array([sx, sy])
            ok = [f for f in range(min(T, uv.shape[0]))
                  if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W_ and 0 <= uv[f, 1] < H_]
            if len(ok) < args.min_frames:
                continue
            try:
                ht, hv, g, nq = dump_hidden(model, proc, video, f'{args.verb} {txt}',
                                            txt, args.fps, layers)
            except AssertionError as e:
                print(f'  [{a}] 건너뜀: {e}'); continue
            grid = g
            hts.append(ht); uvs.append(uv); labs.append(lab); txts.append(txt)
            if hv0 is None:
                # video hidden 은 텍스트 뒤에 오지 않으므로(causal) 텍스트 독립이다 —
                # Qwen 은 video 가 앞, 텍스트가 뒤다 (`<|vision_end|>Track ...`). 1회만.
                hv0 = hv.astype(np.float16)
            del hv
        if not hts:
            continue
        t_, gh, gw = grid
        np.savez_compressed(fp, scene=sc, T=t_, side=gh, gh=gh, gw=gw, W=W_, H=H_,
                            layers=np.array(layers), h_vid=hv0,
                            h_tgt=np.stack(hts).astype(np.float16),
                            uv=np.stack(uvs).astype(np.float32),
                            labels=np.array(labs), texts=np.array(txts))
        print(f'[dump] {sc:28s} anchor {len(hts):2d}  격자 {t_}x{gh}x{gw}  '
              f'{osp.getsize(fp) / 1e6:.0f} MB', flush=True)



def stage_vid(args):
    """prefill attention 을 **영상**으로. 출력 프레임 = 소스 프레임, OBB 가 프레임마다 움직인다.

    앞선 `--stage viz` 의 시간 타일은 초록 십자를 **시간 중앙값 위치**로 찍어서 "움직이는
    물체를 따라가는가" 를 판정할 수 없었다 (버그). 여기서는 프레임 f 마다 그 시각의 OBB 를
    찍고, attention 은 f 가 속한 시간묶음 k = int(f * t / T) 의 맵을 쓴다.

    한 영상에 anchor 를 여러 칸으로 넣는다 — 같은 프레임에서 **텍스트만 다른** 맵을 나란히
    보게 하려는 것이다. 그러면 "물체를 따라가는가" 와 "텍스트에 따라 다른가" 를 한 화면에서
    동시에 판정할 수 있다.

    숫자도 같이 낸다. 격자가 11x20 (셀 32x33px) 로 Molmo2 의 9x9 (71x40px) 보다 5배
    세밀하므로, D165 에서 무력했던 **절대거리 지표에 판별력이 생긴다**:
        argmax 셀 중심 <-> 그 프레임 OBB  의 거리 중앙값
        통제 = 고정셀 oracle (220 셀 중 최선) / 화면중심 / 1셀 반경
    """
    from PIL import Image, ImageDraw
    import imageio.v2 as iio
    from molmo2_attn_video import colorize
    model, proc, nl, nh = load_model(args.ckpt)
    summary = []
    for scene in args.viz_scenes.split(','):
        sc = 'vista4d/' + scene.strip()
        d = osp.join(args.root, sc, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            continue
        p_ = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p_['extrinsics'], p_['intrinsics']
        anch = scene_anchors(args.root, sc, args.viz_anchors)
        video = C.load_frames(args.root, sc, 0, args.num_frames)
        T, H_, W_ = video.shape[0], video.shape[1], video.shape[2]
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        sx, sy = W_ / pw, H_ / ph
        cells, grid = [], None
        for a, key, txt, lab, tw in anch:
            uv, _z = project(tw, E, K)
            uv = uv * np.array([sx, sy])
            ok = [f for f in range(min(T, uv.shape[0]))
                  if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W_ and 0 <= uv[f, 1] < H_]
            if len(ok) < args.min_frames:
                continue
            mp, g, nq = prefill_marginal(model, proc, video, f'{args.verb} {txt}',
                                         txt, args.fps, nl)
            grid = g
            t_, gh, gw = g
            hm = mp[args.viz_layer].mean(0)                 # (t,gh,gw) 32head 평균
            cells.append(dict(a=a, lab=lab, txt=txt, uv=uv, ok=ok, m=hm))
            del mp
            # ── 숫자: 프레임별 argmax <-> 그 프레임 OBB
            dd = []
            for f in ok:
                k = min(t_ - 1, int(f * t_ / T))
                am = np.unravel_index(int(np.argmax(hm[k])), (gh, gw))
                px = ((am[1] + .5) * W_ / gw, (am[0] + .5) * H_ / gh)
                dd.append(np.hypot(px[0] - uv[f, 0], px[1] - uv[f, 1]))
            dd = np.array(dd)
            # 통제: 고정셀 oracle (220셀), 화면중심
            best = float('inf')
            for r_ in range(gh):
                for c_ in range(gw):
                    q = np.median([np.hypot((c_ + .5) * W_ / gw - uv[f, 0],
                                            (r_ + .5) * H_ / gh - uv[f, 1]) for f in ok])
                    best = min(best, float(q))
            cenb = float(np.median([np.hypot(uv[f, 0] - W_ / 2, uv[f, 1] - H_ / 2)
                                    for f in ok]))
            cellr = float(np.hypot(W_ / gw, H_ / gh)) / 2
            summary.append(dict(scene=scene.strip(), anchor=a, label=lab, n=len(ok),
                                argmax_px=float(np.median(dd)), fixed_px=best,
                                center_px=cenb, cell_r=cellr))
            print(f'  [{a:7s} {lab:12s}] argmax {np.median(dd):6.1f}px  '
                  f'고정셀 {best:6.1f}px  중심 {cenb:6.1f}px  1셀반경 {cellr:.0f}px  '
                  f'{txt[:26]}', flush=True)
        if not cells or grid is None:
            continue
        t_, gh, gw = grid
        n = len(cells)
        ncol = 2 if n > 1 else 1
        nrow = int(np.ceil(n / ncol))
        cw = args.cell_w
        chh = cw * H_ // W_
        gmax = [float(c['m'].max()) for c in cells]
        fp = osp.join(args.out_dir, f'qwen_prefill_{scene.strip()}.mp4')
        wr = iio.get_writer(fp, fps=args.out_fps, codec='libx264', quality=8,
                            macro_block_size=1)
        for f in range(T):
            k = min(t_ - 1, int(f * t_ / T))
            canvas = np.zeros((nrow * chh, ncol * cw, 3), np.uint8)
            for i_, c in enumerate(cells):
                fr = np.asarray(Image.fromarray(video[f]).resize((cw, chh),
                                                                 Image.BILINEAR))
                m = c['m'][k]
                up = np.asarray(Image.fromarray(
                    (np.clip(m / max(gmax[i_], 1e-12), 0, 1) * 255).astype(np.uint8)
                ).resize((cw, chh), Image.BILINEAR)) / 255.0
                al = (up[..., None] ** 0.6) * args.alpha
                o = fr.astype(np.float32) / 255.0 * (1 - al) + colorize(up) * al
                am = np.unravel_index(int(np.argmax(m)), (gh, gw))
                pts = [((c['uv'][f, 0], c['uv'][f, 1]), [0, 1, 0], 7),
                       (((am[1] + .5) * W_ / gw, (am[0] + .5) * H_ / gh), [0, 1, 1], 5)]
                for pt, col, rad in pts:
                    if np.isnan(pt[0]):
                        continue
                    cx, cy = int(pt[0] * cw / W_), int(pt[1] * chh / H_)
                    if 0 <= cx < cw and 0 <= cy < chh:
                        o[max(0, cy - 1):cy + 1, max(0, cx - rad):cx + rad] = col
                        o[max(0, cy - rad):cy + rad, max(0, cx - 1):cx + 1] = col
                rr, cc = divmod(i_, ncol)
                canvas[rr * chh:(rr + 1) * chh, cc * cw:(cc + 1) * cw] = \
                    (np.clip(o, 0, 1) * 255).astype(np.uint8)
            im = Image.fromarray(canvas); dr = ImageDraw.Draw(im)
            for i_, c in enumerate(cells):
                rr, cc = divmod(i_, ncol)
                dr.rectangle([cc * cw + 1, rr * chh + 1, cc * cw + 8 * min(len(c['txt']), 40) + 6,
                              rr * chh + 14], fill=(0, 0, 0))
                dr.text((cc * cw + 3, rr * chh + 3), c['txt'][:40], fill=(255, 255, 0))
            dr.rectangle([2, nrow * chh - 16, 430, nrow * chh - 2], fill=(0, 0, 0))
            dr.text((4, nrow * chh - 14),
                    f'frame {f:02d}  t-group {k}/{t_}  L{args.viz_layer} 32head  '
                    f'green=OBB(this frame)  cyan=argmax', fill=(255, 255, 255))
            wr.append_data(np.asarray(im))
        wr.close()
        print(f'  [save] {fp}', flush=True)
    if summary:
        print(f'\n{"=" * 92}')
        print(f'Qwen3-VL prefill 절대거리 (격자 {grid[1]}x{grid[2]}, 셀 '
              f'{summary[0]["cell_r"]:.0f}px 반경)')
        print(f'{"=" * 92}')
        am = np.array([x['argmax_px'] for x in summary])
        fx = np.array([x['fixed_px'] for x in summary])
        cb = np.array([x['center_px'] for x in summary])
        print(f'  argmax      median {np.median(am):6.1f}px')
        print(f'  고정셀 oracle median {np.median(fx):6.1f}px   <- **통제**')
        print(f'  화면중심     median {np.median(cb):6.1f}px')
        print(f'  argmax 가 고정셀을 이긴 anchor: '
              f'{int((am < fx).sum())}/{len(am)}')
        print(f'\n  대조 — Molmo2 9x9 격자에서는 argmax 씬 LOO 174.0px, '
              f'고정셀 28.0px (16개 중 1개만 통제 돌파)')
        makedirs(args.out_dir, exist_ok=True)
        json.dump(summary, open(osp.join(args.out_dir,
                                         f'qwen_prefill_px_{args.tag}.json'), 'w'),
                  ensure_ascii=False, indent=1)



def stage_probe(args):
    model, proc, nl, nh = load_model(args.ckpt)
    scene = args.scenes.split(',')[0]
    anch = scene_anchors(args.root, scene, args.max_per_kind)
    a, key, txt, lab, tw = anch[0]
    video = C.load_frames(args.root, scene, 0, args.num_frames)
    text = f'{args.verb} {txt}'
    print(f'\n[probe] {scene} {a} ({lab})  video {video.shape}  fps {args.fps:g}')
    print(f'        prompt {text!r}')
    inputs = build_inputs(proc, video, text, args.fps)
    print(f'\n-- inputs 키: {list(inputs)}')
    for k, v in inputs.items():
        print(f'   {k:22s} {tuple(v.shape) if hasattr(v, "shape") else type(v).__name__}')
    pos, g, ms = token_layout(model, proc, inputs)
    print(f'\n-- video_grid_thw {g}   merge_size {ms}')
    if g:
        t_, gh, gw = g[0], g[1] // ms, g[2] // ms
        print(f'   merge 후 격자  t={t_}  h={gh}  w={gw}  ->  토큰 {t_ * gh * gw}')
    print(f'   video 토큰 실측 {len(pos)}개   위치 {pos[0].item()} ~ {pos[-1].item()}')
    print(f'   연속인가: {bool((pos[1:] - pos[:-1] == 1).all())}')
    print(f'   전체 시퀀스 길이 {inputs["input_ids"].shape[1]}')
    print(f'\n   ** 프레임 매핑: 입력 {video.shape[0]}프레임 -> 시간축 t={g[0] if g else "?"} '
          f'(temporal_patch 묶음). 프레임 f -> t 인덱스 = f // (49/t) **')
    tail = proc.tokenizer.decode(inputs['input_ids'][0, pos[-1].item() + 1:])
    print(f'\n-- video 뒤 꼬리: {tail!r}')


def stage_tinv(args):
    """D165 와 **같은 지표**: 시간 marginal + 텍스트 교체 쌍비교, 씬 LOO."""
    model, proc, nl, nh = load_model(args.ckpt)
    items = []
    for scene in args.scenes.split(','):
        d = osp.join(args.root, scene, 'da3')
        if not osp.exists(osp.join(d, 'target_track.npz')):
            continue
        p = np.load(osp.join(d, 'pose.npz'), allow_pickle=False)
        E, K = p['extrinsics'], p['intrinsics']
        anch = scene_anchors(args.root, scene, args.max_per_kind)
        if len(anch) < 2:
            continue
        video = C.load_frames(args.root, scene, 0, args.num_frames)
        T, H_, W_ = video.shape[0], video.shape[1], video.shape[2]
        pw, ph = float(K[0][0, 2]) * 2, float(K[0][1, 2]) * 2
        sx, sy = W_ / pw, H_ / ph
        M, cen, lbls, grid = {}, {}, {}, None
        print(f'\n### {scene}  anchors {len(anch)}', flush=True)
        for a, key, txt, lab, tw in anch:
            uv, _z = project(tw, E, K)
            uv = uv * np.array([sx, sy])
            ok = [f for f in range(min(T, uv.shape[0]))
                  if not np.isnan(uv[f, 0]) and 0 <= uv[f, 0] < W_ and 0 <= uv[f, 1] < H_]
            if len(ok) < args.min_frames:
                continue
            try:
                mp, g, nq = prefill_marginal(model, proc, video, f'{args.verb} {txt}',
                                             txt, args.fps, nl)
            except AssertionError as e:
                print(f'  [{a:7s}] 건너뜀: {e}')
                continue
            grid = g
            M[a] = mp.mean(2)                            # (nl,H,gh,gw) 시간 marginal
            cen[a] = (float(np.median(uv[ok, 0])), float(np.median(uv[ok, 1])))
            lbls[a] = lab
            print(f'  [{a:7s} {lab:12s}] q{nq:3d} tok  격자 {g}  {txt[:30]}', flush=True)
            del mp
        if len(M) < 2 or grid is None:
            continue
        t_, gh, gw = grid
        ids = sorted(M)
        Hh = M[ids[0]].shape[1]
        buck = {k: [np.zeros((nl, Hh)), 0] for k in ('all', 'same', 'diff')}
        bmean = {k: [np.zeros(nl), 0] for k in ('all', 'same', 'diff')}
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                ci = cell_of(*cen[i], gh, gw, W_, H_)
                cj = cell_of(*cen[j], gh, gw, W_, H_)
                if ci == cj:
                    continue
                bk = 'same' if lbls[i] == lbls[j] else 'diff'
                mi = M[i][:, :, ci[0], ci[1]]
                mj = M[i][:, :, cj[0], cj[1]]
                hit = (mi > mj).astype(np.float64) + 0.5 * (mi == mj)
                hm = M[i].mean(1)
                hitm = ((hm[:, ci[0], ci[1]] > hm[:, cj[0], cj[1]]).astype(np.float64)
                        + 0.5 * (hm[:, ci[0], ci[1]] == hm[:, cj[0], cj[1]]))
                for kk in ('all', bk):
                    buck[kk][0] += hit; buck[kk][1] += 1
                    bmean[kk][0] += hitm; bmean[kk][1] += 1
        items.append(dict(scene=scene, anchors=ids, labels=[lbls[a] for a in ids],
                          grid=list(grid),
                          counts={k: dict(corr_lh=v[0].copy(), corr_m=bmean[k][0].copy(),
                                          trials=int(v[1]))
                                  for k, v in buck.items() if v[1]}))
        for kk in ('all',):
            c, t = buck[kk]
            if t:
                print(f'  [{kk}] 시행 {t}  32head맵 최고 '
                      f'{(bmean[kk][0] / t * 100).max():.1f}%  개별 최고 '
                      f'{(c / t * 100).max():.1f}%', flush=True)

    if not items:
        raise SystemExit('채점할 게 없다')
    print(f'\n{"=" * 100}')
    print(f'Qwen3-VL prefill 시간 marginal + 텍스트 교체 쌍비교   씬 {len(items)}개')
    print(f'  상수 맵(텍스트 무관 saliency)이면 정확히 50%. 층 선택은 **씬 LOO**.')
    print(f'{"=" * 100}')
    out = {}
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
        lp = loo_c / loo_n * 100 if loo_n else float('nan')
        lse = float(np.sqrt(0.25 / loo_n) * 100) if loo_n else float('nan')
        print(f'\n  [{key}]  시행 {N}   1 s.e. {se:.2f}%p')
        print(f'    32head맵 최고 층 (oracle) : L{bi:02d}  {Am[bi]:5.1f}%  '
              f'({(Am[bi] - 50) / se:+.1f} se)')
        print(f'    개별 (층,head) 최고 (oracle): {Alh.max():5.1f}%  '
              f'({(Alh.max() - 50) / se:+.1f} se)')
        print(f'    **씬 LOO 층 선택**         : {lp:5.1f}%  '
              f'({(lp - 50) / lse:+.1f} se)   뽑힌 층 '
              f'{sorted(picks.items(), key=lambda kv: -kv[1])[:4]}')
        out[key] = dict(trials=int(N), se_pct=se,
                        headmean_per_layer_pct=[float(x) for x in Am],
                        oracle_layer=dict(layer=bi, pct=float(Am[bi])),
                        oracle_layer_head_pct=float(Alh.max()),
                        loo=dict(pct=float(lp), se_pct=float(lse), trials=int(loo_n),
                                 picks=[dict(layer=k2, n=v2) for k2, v2
                                        in sorted(picks.items(), key=lambda kv: -kv[1])]))
    print(f'\n  대조 — Molmo2-4B 같은 지표 씬 LOO: all 79.0% / same 67.6% / diff 84.2%')
    print(f'         MolmoPoint <PATCH> head:    all 75.6% / same 70.6% / diff 76.3%')
    makedirs(args.out_dir, exist_ok=True)
    fp = osp.join(args.out_dir, f'qwen3vl_tinv_{args.tag}.json')
    json.dump(dict(buckets=out,
                   scenes=[dict(scene=it['scene'], anchors=it['anchors'],
                                labels=it['labels'], grid=it['grid'],
                                trials={k: v['trials'] for k, v in it['counts'].items()})
                           for it in items]),
              open(fp, 'w'), ensure_ascii=False, indent=1)
    print(f'\n[out] {fp}')


def main(args):
    makedirs(args.out_dir, exist_ok=True)
    for st in args.stage.split(','):
        st = st.strip()
        if st == 'probe':
            stage_probe(args)
        elif st == 'tinv':
            stage_tinv(args)
        elif st == 'viz':
            stage_viz(args)
        elif st == 'vid':
            stage_vid(args)
        elif st == 'dump':
            stage_dump(args)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--stage', default='probe')
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scenes', default=','.join('vista4d/' + s for s in SCENES))
    p.add_argument('--max_per_kind', type=int, default=8)
    p.add_argument('--min_frames', type=int, default=5)
    p.add_argument('--num_frames', type=int, default=49)
    p.add_argument('--verb', default='Track')
    p.add_argument('--fps', type=float, default=5.0)
    p.add_argument('--out_dir', default='/data1/cympyc1785/LatentCamVid/tmp/d167_qwen3vl')
    p.add_argument('--viz_scenes', default='camel,couple-rocks')
    p.add_argument('--viz_anchors', type=int, default=2)
    p.add_argument('--viz_layer', type=int, default=20)   # LOO 가 뽑은 층
    p.add_argument('--cell_w', type=int, default=320)
    p.add_argument('--alpha', type=float, default=0.65)
    p.add_argument('--out_fps', type=int, default=8)
    p.add_argument('--layers', default='16,20,24,35')
    p.add_argument('--overwrite', action='store_true', default=False)
    p.add_argument('--tag', default='fps5')
    p.add_argument('--gpu', default='4')
    main(p.parse_args())
