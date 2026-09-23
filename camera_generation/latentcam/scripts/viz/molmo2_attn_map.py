"""Molmo2-4B **최종층 attention map** 시각화 — 융합 hidden 이 영상의 어디를 보는지 (D124 진단).

WHY: 카메라 디코더가 먹는 molmo2 산출물 둘 중 attention 을 볼 수 있는 쪽은 text 뿐이다.

    text_emb  (128, 2560)   tail(캡션+PROBE+assistant) 위치 최종 hidden — **영상 전체를 봤다**
    video_emb (3136, 2560)  patch 위치 최종 hidden, prefix-only forward — **캡션을 못 봤다**
                            (chat template 이 `<|video|>` 를 맨 앞으로 올리고 LM 이 causal 이라
                             구조적으로 그렇다. `cache_molmo2_embeddings.py` docstring 참조)

따라서 "카메라 디코더에 꽂히는 텍스트 피처가 영상의 어디를 보고 나온 값인가" 는
**tail query x video patch key** attention 으로만 답할 수 있다. d157_dilo 의 앵커링 실패가
molmo2 단계에서 이미 결정된 것인지, 우리 DiT 가 뭉갠 것인지를 가르는 그림이다 —
DiT 쪽 같은 눈금자는 `main/video_ca_probe.py` 의 `vattn_*` 열이 이미 재고 있다.

구현 주의 — `output_attentions=True` 를 쓰지 않는다. L≈4345 / 32 head / 36층이면 fp32 로
43.5 GB 다. 대신 `modeling_molmo2` 의 모듈 레벨 `eager_attention_forward` 를 **감싸서**
대상 층의 tail 행만 떠낸다 ((1,32,128,4345) fp32 = 71 MB). 원본 함수를 그대로 호출하므로
수치는 eager 와 글자 그대로 같다. 단, config 기본이 `sdpa` 라 `eager` 로 바꿔야 훅이 걸린다.

읽을 때 주의할 것 네 개:
  1. **attention sink** — 첫 토큰이 질량을 대부분 먹는다. video patch 키에 대해서만
     renormalize 하고 sink 질량은 따로 보고한다. 안 하면 모든 히트맵이 평평해 보인다.
  2. **GQA 32:8** — head 평균만 보면 뭉개진다. 평균과 함께 video 를 가장 많이 보는 head 도 낸다.
  3. **공간 해상도는 프레임당 9x9** (27x27 패치를 molmo2 가 3x3 attention pooling 한 것).
     378px 기준 셀당 42px — "어느 물체"는 되고 정밀 위치는 안 된다. 우리 캐시는 여기서 다시
     `adaptive_avg_pool2d` 로 9->8 을 하므로(겹치는 2-tap blur) 카메라 모델이 받는 건 더 흐리다.
  4. **fps** — `cache_molmo2_embeddings.py` 기본이 25.0 인데 학습은 10fps 가정이다. 프롬프트의
     timestamp 문구에만 들어가지만 duration 이 5배 틀리게 전달된다. `--fps` 로 토글해 볼 것.

stage:
  map   씬 1개 + 캡션 1개. query 그룹(target 명사구 / PROBE / assistant 헤더 / tail 전체)별로
        49프레임 9x9 히트맵 오버레이 + 프레임별 질량 + tail 토큰별 video 질량
  swap  같은 씬 같은 base 캡션에서 **target_text 만** 같은 씬의 다른 물체로 교체한다. 코퍼스가
        씬마다 여러 개의 target_text 를 이미 만들어 뒀으므로(camel 5종, car-roundabout 11종)
        문장 구조·motion·framing 이 전부 같고 타겟 명사구만 다른 대조군이 공짜로 나온다.
        히트맵이 그 물체로 옮겨가지 않으면 텍스트 피처가 국소화를 안 하는 것이다.

사용 예시:
  python scripts/viz/molmo2_attn_map.py --stage map  --scene vista4d/camel --gpu 0
  python scripts/viz/molmo2_attn_map.py --stage swap --scene vista4d/camel --gpu 0
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import re
import sys

# `CUDA_VISIBLE_DEVICES` 는 `import torch` 보다 먼저 박혀야 먹는다.
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

REPO = '/data1/cympyc1785/LatentCamVid'
sys.path.insert(0, osp.join(REPO, 'camera_generation/latentcam/main'))
import cache_molmo2_embeddings as C   # noqa: E402 — load_frames/processor_inputs/split_prefix 재사용

CORPUS = '/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121'

# Molmo2 가 실제로 학습한 지시문 형식. 체크포인트 `README.md` 의 예시를 글자 그대로 옮긴 것:
#   Pointing Video QA   "Point to the penguins."
#   Tracking Video QA   "Track the player who is dunking"      <- 마침표가 없다
#   Video QA            "Which animal appears in the video?"
# 이 형식들은 지시문 자체가 프롬프트 전부다. 우리 `PROBE`("Describe the camera trajectory ...")
# 를 덧붙이면 off-distribution 이 되므로 이 모드들은 probe 를 끈다 (`--probe` 로 되살릴 수 있다).
# `chat_template.jinja` 는 `<|video|>` 를 **항상 맨 앞**에 놓으므로(content 리스트 순서 무관)
# README 의 text->video 순서와 우리 video->text 순서는 같은 토큰열이 된다.
INSTRUCT = {
    'point':    'Point to {t}.',
    'track':    'Track {t}',
    'question': 'Where is {t} in the video?',
    'count':    'Count {t}.',
}
OUT = osp.join(REPO, 'tmp/molmo2_attn')


def prompt_body(caption):
    """user 턴의 본문. `C.PROBE` 가 비면 캡션만 (학습 분포형 지시문은 probe 를 안 붙인다).

    `cache_molmo2_embeddings.user_tail` / `processor_inputs` 는 probe 를 무조건 붙이므로
    (학습 캐시와 글자 그대로 같아야 하니 그게 맞다) 여기서만 갈아끼운다. 두 경로가 어긋나면
    `C.split_prefix` 의 assert 가 잡는다."""
    return f'{caption}\n{C.PROBE}' if C.PROBE else caption


def user_tail(caption):
    return (f'<|im_start|>user\n{prompt_body(caption)}<|im_end|>\n'
            f'<|im_start|>assistant\n')


def processor_inputs(proc, video, caption, fps):
    """`C.processor_inputs` 와 같지만 probe 를 `prompt_body` 로 조립한다."""
    msgs = [{'role': 'user',
             'content': [{'type': 'video'},
                         {'type': 'text', 'text': prompt_body(caption)}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    meta = [{'fps': fps, 'total_num_frames': int(video.shape[0]),
             'duration': video.shape[0] / fps,
             'frames_indices': list(range(int(video.shape[0])))}]
    return proc(text=[text], videos=[video], return_tensors='pt',
                do_sample_frames=False, video_metadata=meta)


# ------------------------------------------------------------------ 데이터

def scene_segments(root, scene):
    """prompts.json 을 읽어 (key, entry) 를 int 키 순으로."""
    pj = json.load(open(osp.join(root, scene, 'da3', 'prompts.json')))
    return [(k, pj[k]) for k in sorted(pj, key=lambda x: int(x))]


def pick_base(segs, preset):
    """`--preset` 과 일치하는 첫 세그먼트. 없으면 첫 세그먼트."""
    for k, e in segs:
        if e.get('preset') == preset:
            return k, e
    return segs[0]


def trim_relations(ph, level='noun'):
    """명사구를 세 단계로 깎는다 (`--trim_level`).

      full    the larger pale camel walking along the fence     (원본 target_text)
      action  the larger pale camel walking                     (관계절만 제거, 행위는 유지)
      noun    the larger pale camel                             (매달린 분사까지 제거)

    사용자 지시 2026-09-08: "along the fence 이런 것도 빼줘" -> noun,
    "walking 같은 행위도 포함한 버전과 그냥 ... 형태도 다시 돌려봐줘" -> action / full.

    판별 기준은 전치사 뒤의 **정관사**다 — `in the background` / `along the fence` /
    `on the right side of the enclosure` 는 씬의 다른 물체를 가리키므로 자르고,
    `in a striped shirt` / `with green leaves` 는 타겟 자신의 서술이라 남긴다.
    관계절 제거의 판별 기준은 전치사 뒤의 **정관사**다 — `in the background` /
    `along the fence` / `on the right side of the enclosure` 는 씬의 다른 물체를 가리키므로
    자르고, `in a striped shirt` / `with green leaves` 는 타겟 자신의 서술이라 남긴다.
    """
    if level == 'full':
        return ph
    REL = (r'\s+(?:along|near|behind|beside|against|around|next\s+to|under|over|'
           r'toward|towards|from|in\s+front\s+of)\s+the\b')
    REL2 = r'\s+(?:on|in|at|of|by|to)\s+the\b'
    out = ph
    for pat in (REL, REL2):
        m = re.search(pat, out)
        if m:
            out = out[:m.start()]
    out = out.strip().rstrip(',')
    if level == 'noun':
        out = re.sub(r'\s+\w+ing$', '', out).strip().rstrip(',')   # 매달린 분사 제거
    return out or ph


def minimal_caption(cf, phrase):
    """`motion` 절만 남기고 타겟을 맨 명사구로 교체한다 (framing/composition 절 폐기).

    concise 캡션은 타겟 외에 방해 명사를 여러 개 끌고 온다 — camel 씬의 경우
      target_text  "the larger pale camel walking along **the fence**"   <- 타겟 구절 안에도 fence
      framing_nl   "a medium shot that tightens to a medium close-up shot"
      composition  "with **the larger wooden fence** ... and **the smaller pale camel** ..."
    이 절들을 다 버리고 `the camel` 만 남기면 attention 이 명사 하나에만 반응하는지 볼 수 있다
    (사용자 지시 2026-09-08 "다른 명사들 빼고 camel과 motion 설명만 넣어서").
    `motion` 이 타겟을 지명하지 않는 preset(예: dolly_in 계열의 "without re-aiming")은
    갈아낄 자리가 없으므로 assert 로 막는다 — `--preset` 을 look_at 계열로 둘 것."""
    mo = (cf.get('motion') or '').strip().rstrip('.')
    tt = (cf.get('target_text') or '').strip()
    assert tt and tt in mo, (
        f'motion 이 target_text 를 담고 있지 않다 — 최소 캡션을 만들 수 없다.\n'
        f'  motion      {mo!r}\n  target_text {tt!r}\n'
        f'  --preset 을 타겟을 지명하는 계열(dolly_in_look_at 등)로 지정할 것')
    out = mo.replace(tt, phrase)
    return out[0].upper() + out[1:] + '.'


def labels_of(segs, base_label, max_n):   # base_label 은 raw 라벨('camel'), 'the ' 미포함
    """씬 안의 distinct `anchor_label` 을 빈도순 맨 명사구로. base 를 맨 앞에.

    최소 캡션 모드의 변이 축이다. target_text 를 쓰면 명사구끼리 단어를 공유해버리는데
    (camel 5종 중 4종이 'fence' 를 포함) 맨 라벨은 서로 겹치는 명사가 없다."""
    cnt = {}
    for _, e in segs:
        l = (e.get('anchor_label') or '').strip()
        if l:
            cnt[l] = cnt.get(l, 0) + 1
    order = [base_label] + [l for l, _ in sorted(cnt.items(), key=lambda x: -x[1])
                            if l != base_label]
    return [f'the {l}' for l in order[:max_n]]


def targets_of(segs, base_tt, max_n):
    """씬 안의 distinct target_text 를 빈도순으로. base 의 것을 맨 앞에 둔다."""
    cnt = {}
    for _, e in segs:
        t = ((e.get('caption_fields') or {}).get('target_text') or '').strip()
        if t:
            cnt[t] = cnt.get(t, 0) + 1
    order = [base_tt] + [t for t, _ in sorted(cnt.items(), key=lambda x: -x[1]) if t != base_tt]
    return order[:max_n]


# ------------------------------------------------------------------ attention 포획

def patch_eager(model, layer_idx, rows, store):
    """`eager_attention_forward` 를 감싸 대상 층의 tail 행만 떠낸다. 되돌리는 함수를 반환.

    원본을 그대로 호출하고 결과를 슬라이스만 하므로 수치는 eager 와 동일하다. `attn_weights`
    는 이미 (B,H,L,L) softmax 결과다 — 마스크·스케일링이 원본 경로에서 이미 적용돼 있어
    causal mask 를 다시 만들 필요가 없다 (직접 만들면 틀릴 위험만 생긴다)."""
    mod = sys.modules[type(model.model).__module__]
    orig = mod.eager_attention_forward
    rows_t = torch.as_tensor(rows)

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = orig(module, query, key, value, attention_mask, scaling,
                      dropout=dropout, **kw)
        li = getattr(module, 'layer_idx', -1)
        if w is not None and (layer_idx is None or li == layer_idx):
            store[li if layer_idx is None else 'w'] = \
                w[0][:, rows_t.to(w.device), :].float().cpu()            # (H, R, L)
        return out, w

    mod.eager_attention_forward = wrapped

    # config 기본이 sdpa 면 위 함수가 아예 안 불린다 (modeling_molmo2.py:710 분기).
    cfg = model.model.transformer.blocks[0].self_attn.config
    prev_impl = cfg._attn_implementation
    cfg._attn_implementation = 'eager'

    def restore():
        mod.eager_attention_forward = orig
        cfg._attn_implementation = prev_impl
    return restore


@torch.inference_mode()
def forward_capture(model, proc, video, caption, fps, layer_idx, device, dtype):
    """full forward 1회 -> tail 행 attention. `cache_molmo2_embeddings.verify` 와 같은 경로."""
    core = model.model
    batch = processor_inputs(proc, video, caption, fps)
    prefix_ids, tail_ids = C.split_prefix(proc, batch, caption)
    ids = batch['input_ids'].to(device)
    P, L = len(prefix_ids), ids.shape[1]

    store = {}
    restore = patch_eager(model, layer_idx, list(range(P, L)), store)
    try:
        images, pooling = core.merge_visual_inputs(
            input_ids=ids, pixel_values=None, image_token_pooling=None,
            image_grids=None, image_num_crops=None,
            pixel_values_videos=batch['pixel_values_videos'].to(device, dtype),
            video_token_pooling=batch['video_token_pooling'].to(device),
            video_grids=batch['video_grids'].to(device))
        emb, _ = core.build_input_embeddings(ids, images, pooling)
        core(inputs_embeds=emb, attention_mask=batch['attention_mask'].to(device),
             use_cache=False)
    finally:
        restore()
    assert store, f'layer {layer_idx} 의 attention 을 못 잡았다 — 훅이 안 걸렸다'
    if layer_idx is None:
        patch_pos = torch.nonzero(ids[0].cpu() == C.PATCH_ID).flatten()
        return dict(w_by_layer=store, patch_pos=patch_pos, P=P, L=L, tail_ids=tail_ids)

    patch_pos = torch.nonzero(ids[0].cpu() == C.PATCH_ID).flatten()
    return dict(w=store['w'], patch_pos=patch_pos, P=P, L=L, tail_ids=tail_ids)


def split_video(cap, n_frames, side):
    """(H,R,L) -> video patch 부분만 (H,R,F,s,s) 로 renormalize + 부수 질량."""
    w, pos = cap['w'], cap['patch_pos']
    assert len(pos) == n_frames * side * side, f'patch {len(pos)} != {n_frames}x{side}^2'
    v = w[:, :, pos]                                     # (H,R,F*s*s)
    vmass = v.sum(-1)                                    # (H,R) video 로 간 질량
    sink = w[:, :, 0]                                    # (H,R) 첫 토큰(sink) 질량
    vn = v / vmass.clamp_min(1e-9)[..., None]            # video 안에서 renormalize
    return vn.view(*v.shape[:2], n_frames, side, side), vmass, sink


# ------------------------------------------------------------------ query 그룹

def tail_spans(proc, caption, target_text):
    """tail 문자열 안에서 (target 명사구 / PROBE / assistant 헤더) 의 토큰 인덱스 구간.

    tail = `<|im_start|>user\\n{caption}\\n{PROBE}<|im_end|>\\n<|im_start|>assistant\\n`.
    offset mapping 으로 문자 구간 -> 토큰 인덱스. tail 기준 인덱스라 그대로 행 인덱스가 된다."""
    s = user_tail(caption)
    enc = proc.tokenizer(s, add_special_tokens=False, return_offsets_mapping=True)
    off = enc['offset_mapping']

    def by_char(a, b):
        return [i for i, (o0, o1) in enumerate(off) if o1 > a and o0 < b and o1 > o0]

    g = {}
    ti = s.find(target_text)
    if ti >= 0:
        g['target'] = by_char(ti, ti + len(target_text))
    pi = s.find(C.PROBE)
    if pi >= 0:
        g['probe'] = by_char(pi, pi + len(C.PROBE))
    ai = s.find('assistant')
    if ai >= 0:
        g['assistant'] = by_char(ai, len(s))
    g['all_tail'] = list(range(len(off)))
    return g, [proc.tokenizer.decode([t]) for t in enc['input_ids']]


# ------------------------------------------------------------------ 그림

def overlay(ax, frame, m, vmax, title=None):
    ax.imshow(frame)
    h, w = frame.shape[:2]
    up = np.asarray(Image.fromarray((m / max(vmax, 1e-9) * 255).clip(0, 255).astype(np.uint8))
                    .resize((w, h), Image.BILINEAR)) / 255.0
    ax.imshow(up, cmap='inferno', alpha=0.55, vmin=0, vmax=1)
    r, c = np.unravel_index(int(np.argmax(m)), m.shape)
    ax.plot((c + .5) * w / m.shape[1], (r + .5) * h / m.shape[0], 'o',
            ms=5, mfc='none', mec='cyan', mew=1.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=6, pad=1.5)


def fig_grid(frames, maps, path, sup, per_frame=False):
    """49프레임 전체를 7x7 로.

    per_frame=False: vmax 전 프레임 공통 -> 프레임끼리 질량 비교가 된다. 다만 질량이 한두
      프레임에 쏠리면 나머지가 전부 검게 보인다.
    per_frame=True: 프레임마다 자기 최대로 정규화 -> **프레임 안의 공간 구조**를 본다.
      질량 비교는 못 한다 (제목의 % 로 본다)."""
    F = len(frames)
    n = int(np.ceil(F ** 0.5))
    vmax = float(maps.max())
    fig, axes = plt.subplots(n, n, figsize=(n * 1.5, n * 1.05))
    fm = maps.reshape(F, -1).sum(1)
    for i, ax in enumerate(axes.flat):
        if i < F:
            overlay(ax, frames[i], maps[i],
                    float(maps[i].max()) if per_frame else vmax,
                    f'{i}  {fm[i] * 100:.1f}%')
        else:
            ax.axis('off')
    fig.suptitle(sup, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=135); plt.close(fig)


def fig_summary(frames, per_group, toks, tok_vmass, path, sup):
    """프레임별 질량 곡선 + tail 토큰별 video 질량 막대."""
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.2),
                           gridspec_kw={'height_ratios': [1, 1.25]})
    for name, m in per_group.items():
        fm = m.reshape(len(frames), -1).sum(1)
        ax[0].plot(fm * 100, marker='.', ms=3, lw=1.1, label=name)
    ax[0].axhline(100 / len(frames), color='k', ls=':', lw=.8, label='uniform')
    ax[0].set_xlabel('frame'); ax[0].set_ylabel('attention mass (%)')
    ax[0].legend(fontsize=7, ncol=5); ax[0].grid(alpha=.3)
    ax[0].set_title('per-frame attention mass (renormalized within video patches)', fontsize=8)

    k = len(tok_vmass)
    ax[1].bar(range(k), tok_vmass * 100, color='#3b6ea5')
    ax[1].set_xticks(range(k))
    ax[1].set_xticklabels([t.replace('\n', '\\n') for t in toks[:k]],
                          rotation=90, fontsize=4.5)
    ax[1].set_ylabel('mass to video (%)')
    ax[1].set_title('mass to video per tail token - which of the 128 slots actually look at the video', fontsize=8)
    ax[1].grid(alpha=.3, axis='y')
    fig.suptitle(sup, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=135); plt.close(fig)


def fig_swap_diff(frames, maps, labels, path, sup, k=8):
    """각 변이 - 변이평균. 모든 변이에 공통인 위치 성분(앞프레임·테두리 쏠림)이 상쇄되고
    **타겟 명사구가 실제로 옮긴 질량**만 남는다. 빨강 = 이 타겟에서 더 봄."""
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    base = np.mean(maps, 0)
    d = [m - base for m in maps]
    a = float(np.max([np.abs(x).max() for x in d])) or 1e-9
    R = len(d)
    fig, axes = plt.subplots(R, k, figsize=(k * 1.55, R * 1.28), squeeze=False)
    for r in range(R):
        for c, fi in enumerate(idx):
            ax = axes[r][c]
            ax.imshow(frames[fi])
            h, w = frames[fi].shape[:2]
            up = np.asarray(Image.fromarray(((d[r][fi] / a * 127.5) + 127.5)
                                            .clip(0, 255).astype(np.uint8))
                            .resize((w, h), Image.BILINEAR)) / 255.0
            ax.imshow(up, cmap='bwr', alpha=0.5, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f'f{fi}', fontsize=6, pad=1.5)
        axes[r][0].set_ylabel(labels[r], fontsize=5.5, rotation=0,
                              ha='right', va='center', labelpad=4)
    fig.suptitle(sup, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=140); plt.close(fig)


def fig_swap(frames, maps, labels, path, sup, k=8, per_frame=False):
    """행 = target 변이, 열 = 균등 샘플 프레임. vmax 는 전체 공통."""
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    vmax = float(np.max([m.max() for m in maps]))
    R = len(maps)   # per_frame=True 는 (변이,프레임) 칸마다 자기 최대로 정규화 -> 공간 구조를 본다
    fig, axes = plt.subplots(R, k, figsize=(k * 1.55, R * 1.28), squeeze=False)
    for r in range(R):
        for c, fi in enumerate(idx):
            overlay(axes[r][c], frames[fi], maps[r][fi],
                    float(maps[r][fi].max()) if per_frame else vmax,
                    f'f{fi}' if r == 0 else None)
        axes[r][0].set_ylabel(labels[r], fontsize=5.5, rotation=0,
                              ha='right', va='center', labelpad=4)
    fig.suptitle(sup, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=140); plt.close(fig)


# ------------------------------------------------------------------ stage

def variant_set(args, base, segs, cf):
    """(변이 명사구, 명사구->캡션, 명사구->구간탐색 needle). `--caption_mode` 로 갈린다.

    concise      코퍼스 완성문에서 target_text 만 교체. 카메라 motion·framing·composition 절이
                 전부 남는다 (= 학습에 실제로 쓰는 텍스트).
    target_only  **타겟 서술만.** motion/framing/composition 을 다 버리고 target_text 를 문장
                 하나로 세운다 -> "The larger pale camel walking along the fence."
                 (사용자 지시 2026-09-08 "target에 관련된 text만 쓰자 camera motion, framing 빼고")
    minimal      반대 방향의 대조군: motion 절만 남기고 타겟은 맨 부류 명사로 -> "the camel".
                 D164 실측상 이쪽이 판별력이 가장 낮다.
    """
    if args.caption_mode == 'minimal':
        base_label = (base[1].get('anchor_label') or '').strip()
        return (labels_of(segs, base_label, args.max_targets),
                lambda ph: minimal_caption(cf, ph),
                lambda ph: ph)
    tt0 = (cf.get('target_text') or '').strip()
    if args.caption_mode in INSTRUCT or args.caption_mode == 'target_only':
        assert tt0, f'target_text 가 비었다 — {args.caption_mode} 모드를 쓸 수 없다'
        tf = lambda x: trim_relations(x, args.trim_level)   # noqa: E731
        tts = []
        for ph in targets_of(segs, tt0, 99):
            t = tf(ph)
            if t not in tts:
                tts.append(t)                 # trim 후 중복이 생길 수 있다
        tmpl = INSTRUCT.get(args.caption_mode)
        if tmpl:                       # "Point to the larger pale camel." — needle 은 구절 그대로
            return (tts[:args.max_targets], lambda ph: tmpl.format(t=ph), lambda ph: ph)
        sent = lambda ph: ph[0].upper() + ph[1:] + '.'          # noqa: E731
        return (tts[:args.max_targets], sent,
                lambda ph: sent(ph)[:-1])                        # 캡션 = 타겟 구절 그 자체
    cap0 = base[1]['prompt_camera_with_scene_video']['concise']
    assert tt0 and tt0 in cap0, f'target_text 가 concise 캡션 안에 없다: {tt0!r}'
    return (targets_of(segs, tt0, args.max_targets),
            lambda ph: cap0.replace(tt0, ph),
            lambda ph: ph)


def run_map(args, model, proc, device, dtype, frames, video, base, segs, side):
    cf = base[1].get('caption_fields') or {}
    tts, cap_of, needle_of = variant_set(args, base, segs, cf)
    cap, tt = cap_of(tts[0]), needle_of(tts[0])
    out = forward_capture(model, proc, video, cap, args.fps, args.layer, device, dtype)
    F = video.shape[0]
    vn, vmass, sink = split_video(out, F, side)
    groups, toks = tail_spans(proc, cap, tt)

    print(f'\n[map] layer {args.layer}  L={out["L"]}  prefix={out["P"]}  '
          f'tail={out["L"] - out["P"]}  patch={len(out["patch_pos"])} = {F}x{side}^2')
    print(f'      caption: {cap}')
    print(f'      target_text: {tt!r}')
    # 9x9 격자의 테두리(32/81 = 39.5%)에 질량이 몰리면 물체가 아니라 위치 artifact 다.
    ring = np.zeros((side, side), bool); ring[0] = ring[-1] = True
    ring[:, 0] = ring[:, -1] = True
    print(f'\n{"group":11s} {"rows":>5s} {"video%":>7s} {"sink%":>7s} {"frameEnt":>9s} '
          f'{"peakF":>6s} {"ring%":>7s} {"f0-2%":>7s} {"f3-45%":>7s} {"f46-48%":>8s}')
    print(f'{"(uniform)":11s} {"":5s} {"":7s} {"":7s} {1.0:9.3f} {"":6s} '
          f'{ring.mean() * 100:6.1f}% {3 / F * 100:6.1f}% {43 / F * 100:6.1f}% {3 / F * 100:7.1f}%')
    print('-' * 92)
    per_group, rec = {}, {}
    for g, rows in groups.items():
        if not rows:
            continue
        m = vn[:, rows].mean((0, 1)).numpy()                  # head·row 평균 (F,s,s)
        fm = m.reshape(F, -1).sum(1)
        ent = float(-(fm * np.log(np.clip(fm, 1e-12, None))).sum() / np.log(F))
        vm = float(vmass[:, rows].mean()); sk = float(sink[:, rows].mean())
        per_group[g] = m
        rec[g] = dict(rows=len(rows), video_mass=vm, sink_mass=sk,
                      frame_entropy_norm=ent, peak_frame=int(fm.argmax()),
                      peak_frame_mass_x_F=float(fm.max() * F))
        sp = m.sum(0); sp = sp / max(sp.sum(), 1e-9)
        rec[g].update(ring_mass=float(sp[ring].sum()),
                      mass_f0_2=float(fm[:3].sum()), mass_f46_48=float(fm[-3:].sum()))
        print(f'{g:11s} {len(rows):5d} {vm * 100:6.2f}% {sk * 100:6.2f}% '
              f'{ent:9.3f} {int(fm.argmax()):6d} {sp[ring].sum() * 100:6.1f}% '
              f'{fm[:3].sum() * 100:6.1f}% {fm[3:-3].sum() * 100:6.1f}% {fm[-3:].sum() * 100:7.1f}%')
        for pf, sfx in ((False, ''), (True, '_pf')):
            fig_grid(frames, m, osp.join(args.out, f'map_{args.tag}_{g}{sfx}.png'),
                     f'{args.scene}  [{g}]  layer {args.layer}  video mass {vm * 100:.1f}%  '
                     f'{"(per-frame norm)" if pf else "(global norm)"}  |  {tt}',
                     per_frame=pf)

    # head 별 편차 — 평균만 보면 뭉개진다 (GQA 32:8)
    rows = groups.get('target') or groups['all_tail']
    hv = vmass[:, rows].mean(1)
    top = torch.argsort(hv, descending=True)[:4].tolist()
    print(f'\n      head video질량  min {hv.min() * 100:.1f}% / p50 '
          f'{hv.median() * 100:.1f}% / max {hv.max() * 100:.1f}%   top4 heads {top}')
    for h in top:
        m = vn[h, rows].mean(0).numpy()
        fig_grid(frames, m, osp.join(args.out, f'map_{args.tag}_head{h}.png'),
                 f'{args.scene}  [target, head {h} only]  '
                 f'video mass {float(hv[h]) * 100:.1f}%')

    tok_vmass = vmass[:, :].mean(0).numpy()
    fig_summary(frames, per_group, toks, tok_vmass,
                osp.join(args.out, f'summary_{args.tag}.png'),
                f'{args.scene}  layer {args.layer}  fps {args.fps}')
    json.dump(dict(scene=args.scene, layer=args.layer, fps=args.fps, caption=cap,
                   target_text=tt, groups=rec,
                   head_video_mass=[float(x) for x in hv],
                   token_video_mass=[float(x) for x in tok_vmass],
                   tokens=toks),
              open(osp.join(args.out, f'map_{args.tag}.json'), 'w'),
              ensure_ascii=False, indent=1)


def run_swap(args, model, proc, device, dtype, frames, video, base, segs, side):
    cf, mk = base[1].get('caption_fields') or {}, args.caption_mode
    tts, cap_of, needle_of = variant_set(args, base, segs, cf)
    if args.extra_target:
        tts = tts + [args.extra_target]
    F = video.shape[0]
    print(f'\n[swap] caption_mode={mk}  base caption: {cap_of(tts[0])}')

    print(f'[swap] base preset={base[1].get("preset")}  variants={len(tts)}  '
          f'({"명사구" if mk == "minimal" else "target_text"} 만 교체, 문장 나머지는 동일)')
    maps, labels, recs = [], [], []
    for i, tt in enumerate(tts):
        cap = cap_of(tt)
        out = forward_capture(model, proc, video, cap, args.fps, args.layer, device, dtype)
        vn, vmass, _ = split_video(out, F, side)
        groups, _ = tail_spans(proc, cap, needle_of(tt))
        rows = groups.get('target')
        assert rows, f'교체한 target 명사구를 tail 에서 못 찾았다: {tt!r}'
        if args.target_row_tail:
            rows = rows[-args.target_row_tail:]
        m = vn[:, rows].mean((0, 1)).numpy()
        fm = m.reshape(F, -1).sum(1)
        sm = m.sum(0); sm = sm / max(sm.sum(), 1e-9)        # 시간 합 공간 분포 (s,s)
        cy = float((sm.sum(1) * np.arange(side)).sum()); cx = float((sm.sum(0) * np.arange(side)).sum())
        maps.append(m); labels.append(tt if len(tt) < 46 else tt[:44] + '…')
        recs.append(dict(target_text=tt, video_mass=float(vmass[:, rows].mean()),
                         peak_frame=int(fm.argmax()), centroid_rc=[cy, cx],
                         top1_cell=[int(x) for x in np.unravel_index(int(sm.argmax()), sm.shape)]))
        print(f'  [{i}] video {recs[-1]["video_mass"] * 100:5.2f}%  '
              f'centroid (r{cy:.2f}, c{cx:.2f})  top1 cell {recs[-1]["top1_cell"]}  {tt[:56]}')

    V = np.stack([m.ravel() for m in maps])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    cos = V @ V.T
    print(f'\n  변이 간 attention map 코사인 유사도 (1.0 = 타겟을 바꿔도 안 움직임)')
    print('      ' + ' '.join(f'{i:6d}' for i in range(len(tts))))
    for i in range(len(tts)):
        print(f'  [{i}] ' + ' '.join(f'{cos[i, j]:6.3f}' for j in range(len(tts))))
    off = cos[~np.eye(len(tts), dtype=bool)]
    cen = np.array([r['centroid_rc'] for r in recs])
    dmat = np.linalg.norm(cen[:, None] - cen[None], axis=-1)
    print(f'\n  비대각 코사인  mean {off.mean():.3f}  min {off.min():.3f}  max {off.max():.3f}')
    print(f'  centroid 이동  mean {dmat[~np.eye(len(tts), dtype=bool)].mean():.2f} cell '
          f'(9x9 격자, 셀당 42px)  max {dmat.max():.2f}')

    for pf, sfx in ((False, ''), (True, '_pf')):
        fig_swap(frames, maps, labels, osp.join(args.out, f'swap_{args.tag}{sfx}.png'),
                 f'{args.scene}  target swapped only  layer {args.layer}  '
                 f'(off-diag cosine mean {off.mean():.3f})'
                 f'{"  [per-frame norm]" if pf else ""}',
                 k=args.swap_frames, per_frame=pf)
    fig_swap_diff(frames, maps, labels, osp.join(args.out, f'swap_diff_{args.tag}.png'),
                  f'{args.scene}  per-variant MINUS variant-mean  layer {args.layer}  '
                  f'(red = looked more for this target)', k=args.swap_frames)
    fig, ax = plt.subplots(figsize=(1.1 * len(tts) + 3.2, 1.0 * len(tts) + 2.4))
    im = ax.imshow(cos, cmap='viridis', vmin=min(0.0, cos.min()), vmax=1)
    ax.set_xticks(range(len(tts))); ax.set_yticks(range(len(tts)))
    ax.set_yticklabels(labels, fontsize=6); ax.set_xticklabels(range(len(tts)), fontsize=7)
    for i in range(len(tts)):
        for j in range(len(tts)):
            ax.text(j, i, f'{cos[i, j]:.2f}', ha='center', va='center', fontsize=6,
                    color='w' if cos[i, j] < 0.72 else 'k')
    fig.colorbar(im, shrink=.8); ax.set_title('attention map cosine', fontsize=9)
    fig.tight_layout(); fig.savefig(osp.join(args.out, f'swap_cos_{args.tag}.png'), dpi=140)
    plt.close(fig)
    json.dump(dict(scene=args.scene, layer=args.layer, fps=args.fps,
                   caption_mode=mk, base_caption=cap_of(tts[0]),
                   base_target=tts[0], variants=recs, cosine=cos.tolist(),
                   offdiag_cos_mean=float(off.mean()),
                   centroid_dist=dmat.tolist()),
              open(osp.join(args.out, f'swap_{args.tag}.json'), 'w'),
              ensure_ascii=False, indent=1)


def run_sweep(args, model, proc, device, dtype, frames, video, base, segs, side):
    """36층 전부에서 **타겟 판별력**을 잰다 — 최종층이 안 움직인다면 중간층은 어떤가.

    지표는 변이 간 attention map 의 비대각 코사인. 1.0 = 타겟을 바꿔도 map 이 그대로
    (= 그 층의 텍스트 위치는 어느 물체를 가리키는지 모른다). 낮을수록 판별력이 있다.
    GR00T N1.5 가 36층 중 12층을 쓰는 것과 같은 질문에 이 그림이 직접 답한다."""
    cf = base[1].get('caption_fields') or {}
    tts, cap_of, needle_of = variant_set(args, base, segs, cf)
    F, nl = video.shape[0], model.config.text_config.num_hidden_layers

    per_var = []      # [variant][layer] -> (F,s,s)
    vmass_lv = np.zeros((len(tts), nl))
    for i, tt in enumerate(tts):
        cap = cap_of(tt)
        out = forward_capture(model, proc, video, cap, args.fps, None, device, dtype)
        groups, _ = tail_spans(proc, cap, needle_of(tt))
        rows = groups['target']
        if args.target_row_tail:
            rows = rows[-args.target_row_tail:]
        maps = {}
        for li, w in out['w_by_layer'].items():
            v = w[:, rows][:, :, out['patch_pos']]                 # (H,R,F*s*s)
            vm = v.sum(-1)
            vmass_lv[i, li] = float(vm.mean())
            maps[li] = (v / vm.clamp_min(1e-9)[..., None]).mean((0, 1)).view(F, side, side).numpy()
        per_var.append(maps)
        del out
        print(f'  [{i}] {tt[:62]}', flush=True)

    rows_out = []
    for li in range(nl):
        V = np.stack([per_var[i][li].ravel() for i in range(len(tts))])
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        cos = V @ V.T
        off = cos[~np.eye(len(tts), dtype=bool)]
        sp = per_var[0][li].sum(0); sp = sp / max(sp.sum(), 1e-9)
        ring = np.zeros((side, side), bool); ring[0] = ring[-1] = True
        ring[:, 0] = ring[:, -1] = True
        fm = per_var[0][li].reshape(F, -1).sum(1)
        rows_out.append(dict(layer=li, offdiag_cos_mean=float(off.mean()),
                             offdiag_cos_min=float(off.min()),
                             video_mass=float(vmass_lv[:, li].mean()),
                             ring_mass=float(sp[ring].sum()),
                             mass_f0_2=float(fm[:3].sum()),
                             frame_ent=float(-(fm * np.log(np.clip(fm, 1e-12, None))).sum() / np.log(F))))

    print(f'\n{"layer":>5s} {"offdiagCos":>11s} {"min":>7s} {"video%":>7s} '
          f'{"ring%":>7s} {"f0-2%":>7s} {"frameEnt":>9s}')
    print('-' * 60)
    for r in rows_out:
        star = ' <' if r['offdiag_cos_mean'] < 0.9 else ''
        print(f'{r["layer"]:5d} {r["offdiag_cos_mean"]:11.4f} {r["offdiag_cos_min"]:7.4f} '
              f'{r["video_mass"] * 100:6.2f}% {r["ring_mass"] * 100:6.1f}% '
              f'{r["mass_f0_2"] * 100:6.1f}% {r["frame_ent"]:9.3f}{star}')
    bl = min(rows_out, key=lambda r: r['offdiag_cos_mean'])
    print(f'\n  판별력이 가장 큰 층: {bl["layer"]}  (비대각 코사인 {bl["offdiag_cos_mean"]:.4f})')
    print(f'  최종층({nl - 1}): {rows_out[-1]["offdiag_cos_mean"]:.4f}')

    x = [r['layer'] for r in rows_out]
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(x, [r['offdiag_cos_mean'] for r in rows_out], marker='o', ms=3, label='off-diag cosine (mean)')
    ax[0].fill_between(x, [r['offdiag_cos_min'] for r in rows_out],
                       [r['offdiag_cos_mean'] for r in rows_out], alpha=.2)
    ax[0].axhline(1.0, color='r', ls=':', lw=.9)
    ax[0].set_ylabel('cosine between targets')
    ax[0].set_title(f'{args.scene}: does the map move when the target noun changes? '
                    f'(1.0 = no)', fontsize=9)
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
    ax[1].plot(x, [r['video_mass'] * 100 for r in rows_out], marker='.', label='mass to video (%)')
    ax[1].plot(x, [r['ring_mass'] * 100 for r in rows_out], marker='.', label='border-cell mass (%)')
    ax[1].plot(x, [r['mass_f0_2'] * 100 for r in rows_out], marker='.', label='mass in frames 0-2 (%)')
    ax[1].axhline(39.5, color='gray', ls=':', lw=.8)
    ax[1].axhline(6.1, color='gray', ls='--', lw=.8)
    ax[1].set_xlabel('layer'); ax[1].set_ylabel('%')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(osp.join(args.out, f'sweep_{args.tag}.png'), dpi=140); plt.close(fig)
    json.dump(dict(scene=args.scene, caption_mode=args.caption_mode,
                   targets=tts, layers=rows_out),
              open(osp.join(args.out, f'sweep_{args.tag}.json'), 'w'),
              ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ main

def main(args):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    args.out = args.out or OUT
    C.user_tail = user_tail           # split_prefix 가 참조한다 — 두 경로를 맞춘다
    if args.probe:
        C.PROBE = '' if args.probe == 'none' else args.probe
    elif args.caption_mode in INSTRUCT:
        C.PROBE = ''                  # 학습 분포형 지시문은 probe 를 붙이지 않는다
    print(f'[prompt] caption_mode={args.caption_mode}  probe={C.PROBE!r}')
    makedirs(args.out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, dtype=dtype, trust_remote_code=True).to(device).eval()
    nl = model.config.text_config.num_hidden_layers
    if args.layer < 0:
        args.layer += nl
    assert 0 <= args.layer < nl, f'layer {args.layer} 범위 밖 (0..{nl - 1})'
    args.tag = (f"{args.scene.split('/')[-1]}_L{args.layer}_fps{args.fps:g}"
                f"{dict(minimal='_min', target_only='_tgt', point='_pt', track='_trk',
                        question='_q', count='_cnt').get(args.caption_mode, '')}"
                f"{'' if args.trim_level == 'noun' else '_' + args.trim_level}"
                f"{f'_t{args.target_row_tail}' if args.target_row_tail else ''}")

    segs = scene_segments(args.root, args.scene)
    base = pick_base(segs, args.preset)
    s, e = base[1]['frame_idx']
    video = C.load_frames(args.root, args.scene, int(s), int(e))
    frames = [video[i] for i in range(video.shape[0])]
    print(f'[load] {args.scene} seg {base[0]}  frames {video.shape}  '
          f'layers {nl} -> layer {args.layer}  segs {len(segs)}')

    # side 는 patch 수에서 역산 (cache_molmo2_embeddings.main 과 같은 방식)
    b0 = processor_inputs(proc, video, 'x', args.fps)
    npatch = int((b0['input_ids'][0] == C.PATCH_ID).sum())
    side = int(round((npatch / video.shape[0]) ** 0.5))
    assert side * side * video.shape[0] == npatch, f'patch {npatch} 가 정사각이 아니다'
    print(f'[layout] patch {npatch} = {video.shape[0]} x {side}x{side}')

    for st in args.stage.split(','):
        if st == 'map':
            run_map(args, model, proc, device, dtype, frames, video, base, segs, side)
        elif st == 'swap':
            run_swap(args, model, proc, device, dtype, frames, video, base, segs, side)
        elif st == 'sweep':
            run_sweep(args, model, proc, device, dtype, frames, video, base, segs, side)
        else:
            raise SystemExit(f'unknown stage {st}')
    print(f'\n[out] {args.out}')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--stage', default='map,swap')
    p.add_argument('--ckpt', default=C.MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--scene', default='vista4d/camel')
    p.add_argument('--preset', default='dolly_in_look_at')
    p.add_argument('--layer', type=int, default=-1)        # -1 = 최종층
    p.add_argument('--fps', type=float, default=25.0)      # 캐시 기본값과 동일
    p.add_argument('--caption_mode', default='concise',
                   choices=('concise', 'target_only', 'minimal',
                            'point', 'track', 'question', 'count'))
    p.add_argument('--probe', default='')   # user_tail 의 고정 probe 문장 교체
    p.add_argument('--trim_level', default='noun',
                   choices=('full', 'action', 'noun'))   # 타겟 구절을 어디까지 깎나
    p.add_argument('--target_row_tail', type=int, default=0)   # >0 = target 구간 마지막 N 토큰만
    p.add_argument('--max_targets', type=int, default=5)
    p.add_argument('--extra_target', default='')           # 씬에 없는 물체로 대조하고 싶을 때
    p.add_argument('--swap_frames', type=int, default=8)
    p.add_argument('--out', default='')
    p.add_argument('--gpu', default='0')
    main(p.parse_args())
