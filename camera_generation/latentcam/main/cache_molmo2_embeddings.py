"""Molmo2-4B 의 **video+text 융합 hidden state** 를 미리 구워 두는 캐시 빌더 (D124).

WHY: D117 이 붙인 세 번째 cross-attn 스트림(video CA)은 지금 PE-AV 를 먹는데, PE-AV 는
video tower 와 text tower 가 **끝에서 코사인 유사도로만 만나는** dual-encoder 라
"이 캡션이 이 영상의 어디를 가리키나"가 토큰 안에 안 들어 있다 (D117-c arm 이 별 효과가
없었던 이유로 의심). Molmo2 는 vision feature 를 LM 임베딩에 **더해 넣고 36층 causal
self-attn 을 태우는** decoder-only VLM 이라, 캡션 위치의 hidden state 는 이미 영상 전체를
보고 나온 값이다. 그 hidden state 를 그대로 video CA 슬롯에 꽂는다.

어디서 뽑나: `Molmo2Model.forward` 의 `last_hidden_state` — `modeling_molmo2.py:1073` 의
`hidden_states = self.ln_f(hidden_states)` 직후, `lm_head` **직전**이다. 즉 "마지막 decoding
stage 전에 video·text attention 이 끝난" 지점.

두 가지를 굽는다:
  video  씬당 1파일  `<video_out>/<scene_key>.pt` = {'emb': (3136,2560) fp16, 'frames': 49}
         patch 위치 3969개(= 49프레임 x 9x9) 의 hidden state 를 프레임 안에서만 9x9 -> 8x8
         평균풀링한 것. **프레임을 섞지 않는다.** 3136 + text 128 = 3264 토큰으로 da3 geo
         스트림의 3456 과 맞춘다 (사용자 지시: "da3 랑 token 개수 맞춰서 resampling").
         chat template 이 `<|video|>` 를 맨 앞으로 올리고 LM 이 causal 이라, patch 위치의
         hidden state 는 **뒤에 붙는 캡션과 무관**하다 -> 씬당 1회면 된다 (아래 --verify 가
         이 불변조건을 실제로 잰다).
  text   전체 1파일  `<text_out>` = {'by_name': {data_name: idx}, 'emb': (U,128,2560) fp16,
                                     'mask': (U,128) bool, 'text': [(scene_key, caption)] * U}
         캡션 + 고정 probe 문장 + assistant 헤더 위치의 hidden state. 이쪽은 앞의 영상을
         전부 본 뒤의 값이라 **씬마다 다르다** — 그래서 dedup 키가 캡션 문자열이 아니라
         `(scene_key, caption)` 이다 (캡션만으로 묶으면 13,183 이지만 실제 고유 조합은 13,679).

효율: ViT(49프레임 x 729패치 x 27층)를 캡션마다 다시 돌리면 그것만 20분 넘는다. 씬당 1회
`merge_visual_inputs` + `build_input_embeddings` 로 prefix `inputs_embeds` 를 만들어 캐시하고,
캡션 배치는 그 prefix 를 expand 해서 `inputs_embeds` 로만 넣는다 (ViT 재실행 0회).
`lm_head` 도 안 태운다 — vocab 151,936 x 4,345 위치는 모델 본체보다 FLOP 이 크다.

중간층도 같이 굽는다 (`--extra_layer`, D194 사용자 지시 "layer 0~35에서 layer 21도"):
  마지막 층은 `lm_head` 직전이라 **다음 토큰 예측에 필요한 것만 남기는** 쪽으로 이미 기울어
  있다. 중간층은 그 압축 전이라 시각·공간 정보가 더 남아 있다는 게 VLM probing 의 통설이고,
  어느 쪽이 카메라 궤적에 쓸모 있는지는 **둘 다 구워 놓고 학습으로 갈라야** 안다. 그래서
  기본 `emb` 옆에 `emb_l{N}` 을 같이 저장한다 — 두 번 굽지 않으려는 것이다.
  주의: `emb` 는 `ln_f` **이후**고 `emb_l{N}` 은 블록 N 의 raw 출력(ln_f 이전)이다. 층마다
  스케일이 크게 다르므로 하류는 `peav_in_ln: true`(proj 앞 LayerNorm) 를 켠 채 써야 한다.
  층 번호는 `blocks[N]` 의 출력 = 사용자가 말한 0~35 인덱스와 같다.
  `--extra_layer` 를 안 주면 저장물은 D124/D191 과 **비트 동일**하다.

decode 스텝 hidden 도 구울 수 있다 (`--decode_tokens`, D197-d 사용자 지시 "prefill 단계에서
나오는 hidden token 들은 버리고 decode 단계에서 나오는 hidden 만"):
  기본 경로가 담는 것은 **prompt 위치**의 hidden 이다 — 모델이 답을 아직 한 글자도 커밋하지
  않은 상태다. `--decode_tokens N` 을 주면 greedy 로 N 토큰을 실제로 생성하면서, `y_t` 를 입력으로
  넣고 나온 hidden 을 슬롯 t 에 담는다. `y_0` 를 고른 prefill 마지막 위치 hidden 은 버린다.
  `emb_l{N}` 도 같은 슬롯 기준이라 층 비교가 그대로 성립한다. 생성문은 `decode_text` 로 같이
  저장한다. 안 주면 저장물은 기존과 **비트 동일**하다.

샤딩 (`--num_shards/--shard_id`): 씬을 `i % num_shards` 로 갈라 GPU 여러 장에 흩는다.
  video 는 씬당 1파일이라 충돌이 없고, text 는 전체 1파일이라 샤드마다
  `<text_out>.shard{i}of{n}` 으로 떨어뜨린 뒤 `--merge_shards` 로 합친다 (GPU 불필요).

사용 예시:
  python main/cache_molmo2_embeddings.py --gpu 2 --verify --limit_scenes 1   # 스모크
  python main/cache_molmo2_embeddings.py --gpu 2                             # 전량
  python main/cache_molmo2_embeddings.py --gpu 3 --root <d189> --seg_prefix dynpose \
      --text_override_json <tmp>/molmo2_text_track.json                      # D191 Track {target}
  python main/cache_molmo2_embeddings.py --gpu 6 --root <d194> --seg_prefix dynpose \
      --extra_layer 21 --num_shards 3 --shard_id 0 ...                       # D194 샤드
  python main/cache_molmo2_embeddings.py --merge_shards --num_shards 3 ...   # 합치기
  python main/cache_molmo2_embeddings.py --gpu 0 --decode_tokens 128 --extra_layer 21 \
      --verify --limit_scenes 1 ...                                          # D197-d 스모크
"""

from argparse import ArgumentParser, Namespace
from os import path as osp, makedirs, environ
import json
import sys
import time

# `CUDA_VISIBLE_DEVICES` 는 **`import torch` 보다 먼저** 박혀야 먹는다.
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cache_peav_embeddings import collect   # noqa: E402  — seg_list/prompts.json 수집을 재사용

MOLMO = ('/data1/cympyc1785/LatentCamVid/camera_generation/tools/molmo2/'
         'checkpoints/Molmo2-4B')
CORPUS = '/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121'
CACHE = osp.join(CORPUS, 'molmo2_cache')

PATCH_ID = 151938        # config.json image_patch_id — vision feature 가 더해지는 자리
# 캡션 뒤에 항상 같은 문장을 붙인다. 이 위치들이 "영상 + 캡션을 다 본" probe 슬롯이 된다.
# 캡션 자체는 명령문이 아니라 서술문이라, 지시 없이 그냥 두면 Molmo2 가 무슨 과제인지 모른다.
PROBE = 'Describe the camera trajectory implied by this instruction for the video above.'


# ------------------------------------------------------------------ 입력 조립

def load_frames(root, chunk, s, e):
    """images_4 의 [s,e) 를 (T,H,W,3) uint8 로. resize/정규화는 video_processor 가 한다."""
    d = osp.join(root, chunk, 'images_4')
    return np.stack([np.asarray(Image.open(osp.join(d, f'{i:05d}.png')).convert('RGB'))
                     for i in range(s, e)])


def user_tail(caption):
    """캡션 하나에 대한 **가변 꼬리** 문자열. prefix(=영상) 뒤에 그대로 이어 붙는다."""
    return (f'<|im_start|>user\n{caption}\n{PROBE}<|im_end|>\n'
            f'<|im_start|>assistant\n')


def processor_inputs(proc, video, caption, fps):
    """chat template 을 통과시킨 full 입력. video_metadata 는 timestamp 문구에만 쓰인다."""
    msgs = [{'role': 'user',
             'content': [{'type': 'video'},
                         {'type': 'text', 'text': f'{caption}\n{PROBE}'}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    meta = [{'fps': fps, 'total_num_frames': int(video.shape[0]),
             'duration': video.shape[0] / fps,
             'frames_indices': list(range(int(video.shape[0])))}]
    return proc(text=[text], videos=[video], return_tensors='pt',
                do_sample_frames=False, video_metadata=meta)


def split_prefix(proc, batch, caption):
    """full input_ids 에서 **캡션에 무관한 앞부분(=영상 블록)** 을 잘라낸다.

    chat template 이 `<|video|>` 를 맨 앞으로 hoist 하므로 잘림선이 하나뿐이다. BPE 가 경계를
    넘어 병합했으면 아래 assert 가 잡는다 (prefix 가 문자열 concat 과 안 맞으면 거짓).
    """
    ids = batch['input_ids'][0]
    tail = proc.tokenizer(user_tail(caption), add_special_tokens=False,
                          return_tensors='pt')['input_ids'][0]
    assert torch.equal(ids[-len(tail):], tail), (
        'prefix/tail 토큰 경계가 안 맞는다 — BPE 가 경계를 넘어 병합했다. '
        f'{proc.tokenizer.decode(ids[-len(tail) - 2:])!r}')
    return ids[:-len(tail)], tail


# ------------------------------------------------------------------ forward

class LayerTap:
    """블록 하나의 출력만 붙잡는 forward hook.

    `output_hidden_states=True` 로도 되지만 그러면 37개 층을 전부 들고 있게 된다 —
    tail 배치(B=8, ~4.5k 토큰)에서 그것만 6.6 GB 라 층 하나 쓰자고 낼 비용이 아니다.

    디코더 블록은 `Molmo2Model.transformer`(= `Molmo2TextModel`) 아래에 있다 — `core` 로 넘어오는
    `model.model` 은 vision backbone 까지 들고 있는 바깥 껍데기라 `blocks` 가 없다.
    `blocks` 는 `config.num_hidden_layers` 보다 길 수 있어(:1043 이 앞에서 잘라 쓴다) 범위 검사는
    **실제로 도는 층 수**로 한다. `blocks[N]` 의 출력 = 사용자가 말한 layer N (0~35) 이다.
    """

    def __init__(self, core, layer):
        txt = getattr(core, 'transformer', core)
        assert hasattr(txt, 'blocks'), 'blocks 가 없다 — modeling_molmo2 구조가 바뀌었다'
        n = int(getattr(txt.config, 'num_hidden_layers', len(txt.blocks)))
        assert 0 <= layer < n, f'layer {layer} 가 0~{n - 1} 밖이다'
        self.h = None
        self._handle = txt.blocks[layer].register_forward_hook(self._grab)

    def _grab(self, _mod, _inp, out):
        self.h = out[0] if isinstance(out, tuple) else out

    def close(self):
        self._handle.remove()


@torch.inference_mode()
def scene_prefix(model, batch, prefix_ids, device, dtype, tap=None):
    """씬당 1회: ViT 를 태워 prefix 의 `inputs_embeds` 와 patch 위치 hidden state 를 얻는다.

    `tap` 을 주면 (마지막 층, 중간 층) 두 개를 돌려준다. 중간 층은 `ln_f` 이전 값이다.
    """
    core = model.model
    images, pooling = core.merge_visual_inputs(
        input_ids=prefix_ids[None].to(device),
        pixel_values=None, image_token_pooling=None, image_grids=None, image_num_crops=None,
        pixel_values_videos=batch['pixel_values_videos'].to(device, dtype),
        video_token_pooling=batch['video_token_pooling'].to(device),
        video_grids=batch['video_grids'].to(device))
    emb, _ = core.build_input_embeddings(prefix_ids[None].to(device), images, pooling)
    out = core(inputs_embeds=emb,
               attention_mask=torch.ones(1, emb.shape[1], dtype=torch.long, device=device),
               use_cache=False)
    extra = None
    if tap is not None:
        assert tap.h is not None, 'LayerTap 이 안 걸렸다 — hook 대상 블록이 안 불렸다'
        extra = tap.h[0]
    return emb, out.last_hidden_state[0], extra


@torch.inference_mode()
def tail_hidden(model, prefix_emb, tails, text_len, device, tap=None):
    """캡션 배치 -> (B, text_len, 2560) hidden + (B, text_len) mask (+ 중간층 hidden).

    prefix 는 expand 로 공유하고 ViT 를 다시 안 돈다. **right padding** 이라 causal attn 상
    꼬리 위치는 pad 를 못 본다 (left padding 이면 position 이 밀려 틀린다)."""
    core, B = model.model, len(tails)
    P, L = prefix_emb.shape[1], max(len(t) for t in tails)
    ids = torch.zeros(B, L, dtype=torch.long)
    msk = torch.zeros(B, L, dtype=torch.long)
    for i, t in enumerate(tails):
        ids[i, :len(t)], msk[i, :len(t)] = t, 1
    tail_emb, _ = core.build_input_embeddings(ids.to(device))
    emb = torch.cat([prefix_emb.expand(B, -1, -1), tail_emb], 1)
    am = torch.cat([torch.ones(B, P, dtype=torch.long), msk], 1).to(device)
    h = core(inputs_embeds=emb, attention_mask=am, use_cache=False).last_hidden_state[:, P:]
    om = torch.zeros(B, text_len, dtype=torch.bool)
    om[:, :L] = msk.bool()

    def pad(x):
        o = torch.zeros(B, text_len, x.shape[-1], dtype=torch.float16)
        o[:, :L] = x.float().half().cpu()
        return o

    return pad(h), om, (pad(tap.h[:, P:]) if tap is not None else None)


@torch.inference_mode()
def decode_hidden(model, prefix_emb, tails, text_len, n_new, device, eos_ids, tap=None):
    """**decode 스텝의** hidden 만 모은다 (D197-d, 사용자 지시).

    `tail_hidden` 이 주는 것은 prompt 위치의 hidden 이다 — 아직 답을 만들기 전, "다음 토큰을
    뭘 낼까"를 아직 한 번도 커밋하지 않은 상태의 표현이다. 여기서는 그 prefill hidden 을 **전부
    버리고**, 모델이 실제로 궤적 서술을 생성하는 동안(= greedy decode) 나오는 hidden 만 남긴다.
    슬롯 t 에는 t번째 생성 토큰 `y_t` **위치의** hidden 이 들어간다 (`y_t` 를 입력으로 넣고 나온
    값이므로 decode forward 의 산물이다). `y_0` 를 고르는 데 쓰인 prefill 의 마지막 위치 hidden 은
    버린다 — 그게 "prefill 단계에서 나오는 hidden token 은 버린다"의 경계선이다.

    배치 레이아웃이 `tail_hidden` 과 다르다. right padding 이면 아이템마다 꼬리 끝 열이 달라서
    decode 를 한 번에 못 돈다 (`cache_position` 은 배치 공용 1D). 그래서 **mid padding**:
        [prefix (P)] [pad (L-len_i)] [tail (len_i)]
    로 꼬리 끝을 한 열에 정렬하고, 밀린 position 은 `position_ids` 로 아이템마다 따로 준다
    (prefix 0..P-1, pad 는 0(마스크됨), tail 은 P..P+len_i-1). 이러면 각 꼬리 토큰이 보는
    position 은 right padding 일 때와 **같다** — 아래 `verify_decode` 가 B=1(패딩 없음) 대조로
    실측한다.

    EOS(`<|im_end|>`) 를 낸 아이템은 그 슬롯까지 유효로 치고 mask 를 닫는다. 전부 닫히면 조기 종료.
    """
    core, B = model.model, len(tails)
    P, L = prefix_emb.shape[1], max(len(t) for t in tails)
    ids = torch.zeros(B, L, dtype=torch.long)
    msk = torch.zeros(B, L, dtype=torch.long)
    pos = torch.zeros(B, P + L, dtype=torch.long)
    pos[:, :P] = torch.arange(P)
    for i, t in enumerate(tails):
        ids[i, L - len(t):], msk[i, L - len(t):] = t, 1
        pos[i, P + L - len(t):] = torch.arange(P, P + len(t))
    tail_emb, _ = core.build_input_embeddings(ids.to(device))
    emb = torch.cat([prefix_emb.expand(B, -1, -1), tail_emb], 1)
    am = torch.cat([torch.ones(B, P, dtype=torch.long), msk], 1).to(device)
    pos = pos.to(device)
    out = core(inputs_embeds=emb, attention_mask=am, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    nxt = model.lm_head(out.last_hidden_state[:, -1]).argmax(-1)      # y_0 (prefill hidden 은 버린다)
    del out

    eos_ids = eos_ids.to(device)
    last_pos = pos[:, -1:]                                            # = P + len_i - 1
    gen = torch.zeros(B, n_new, dtype=torch.long)
    done = torch.full((B,), n_new, dtype=torch.long, device=device)   # EOS 슬롯 (없으면 n_new)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    hs, hx, used = [], [], n_new
    for t in range(n_new):
        gen[:, t] = nxt.cpu()
        e, _ = core.build_input_embeddings(nxt[:, None])
        am = torch.cat([am, alive[:, None].long()], 1)
        o = core(inputs_embeds=e, attention_mask=am,
                 position_ids=last_pos + (t + 1),
                 past_key_values=cache, use_cache=True,
                 cache_position=torch.tensor([P + L + t], device=device))
        hs.append(o.last_hidden_state[:, 0].float().half().cpu())
        if tap is not None:
            hx.append(tap.h[:, 0].float().half().cpu())
        eos = torch.isin(nxt, eos_ids)
        done = torch.where(alive & eos, torch.full_like(done, t), done)
        alive = alive & ~eos
        if not bool(alive.any()):
            used = t + 1
            break
        nxt = model.lm_head(o.last_hidden_state[:, 0]).argmax(-1)
        del o

    om = torch.arange(text_len)[None] <= done.cpu()[:, None]
    om &= torch.arange(text_len)[None] < used

    def pad(xs):
        o = torch.zeros(B, text_len, xs[0].shape[-1], dtype=torch.float16)
        o[:, :len(xs)] = torch.stack(xs, 1)
        return o

    return pad(hs), om, (pad(hx) if tap is not None else None), gen[:, :used], done.cpu()


def pool_video(hid, patch_mask, n_frames, side, pool):
    """patch 위치 hidden -> (n_frames*pool*pool, 2560). 풀링은 **프레임 안에서만**."""
    v = hid[patch_mask]                                   # (n_frames*side*side, D)
    assert v.shape[0] == n_frames * side * side, f'{v.shape} vs {n_frames}x{side}^2'
    if pool == side:
        return v
    v = v.view(n_frames, side, side, -1).permute(0, 3, 1, 2).float()
    v = torch.nn.functional.adaptive_avg_pool2d(v, (pool, pool))
    return v.permute(0, 2, 3, 1).reshape(n_frames * pool * pool, -1)


# ------------------------------------------------------------------ 샤드

def shard_text_path(args):
    """샤드면 `<text_out>.shard{i}of{n}`, 아니면 `<text_out>` 그대로."""
    if args.num_shards <= 1:
        return args.text_out
    base, ext = osp.splitext(args.text_out)
    return f'{base}.shard{args.shard_id}of{args.num_shards}{ext}'


def merge_shards(args):
    """샤드 text 파일들 -> 전역 `text.pt` 한 개. GPU 를 안 쓴다.

    각 샤드의 `pidx` 는 그 샤드 안에서만 유효하므로, pairs 를 합치면서 **전역 인덱스를 새로
    매기고** `by_name` 은 여기서 `collect` 로 다시 만든다 (샤드는 자기 씬만 알기 때문).
    """
    parts, offset, pair_idx = [], 0, {}
    embs, masks, extras, texts, dec_texts = [], [], [], [], []
    extra_key = None
    for sid in range(args.num_shards):
        p = shard_text_path(Namespace(**{**vars(args), 'shard_id': sid}))
        assert osp.isfile(p), f'샤드 파일이 없다: {p}'
        d = torch.load(p, map_location='cpu')
        assert d.get('shard', [None, None])[1] == args.num_shards, f'{p}: num_shards 불일치'
        for i, t in enumerate(d['text']):
            assert t not in pair_idx, f'샤드 {sid} 에 중복 pair: {t!r} — 씬 분할이 겹쳤다'
            pair_idx[t] = offset + i
        offset += len(d['text'])
        texts += d['text']
        dec_texts += d.get('decode_text', [])
        embs.append(d['emb'])
        masks.append(d['mask'])
        xk = [k for k in d if k.startswith('emb_l')]
        if xk:
            assert extra_key in (None, xk[0]), f'{p}: 중간층 키가 샤드마다 다르다 {xk[0]}'
            extra_key = xk[0]
            extras.append(d[xk[0]])
        parts.append((p, len(d['text']), d))

    ref = parts[0][2]
    items = collect(args.root, args.splits.split(','), args.seg_prefix)
    if args.text_override_json:
        ov = json.load(open(args.text_override_json))
        for it in items:
            it['concise'] = ov[it['data_name']].strip()
    by_name = {}
    for it in items:
        key = f"{it['scene_key']}\t{it['concise']}"
        assert key in pair_idx, f"{it['data_name']}: 어느 샤드에도 없다 ({key!r})"
        by_name[it['data_name']] = pair_idx[key]

    out = {'by_name': by_name, 'emb': torch.cat(embs), 'mask': torch.cat(masks),
           'text': texts, 'template': ref['template'], 'text_len': ref['text_len'],
           'text_override_json': ref['text_override_json'],
           'extra_layer': ref.get('extra_layer'), 'probe': ref['probe']}
    if extras:
        assert len(extras) == args.num_shards, '중간층이 일부 샤드에만 있다 — 재굽기 필요'
        out[extra_key] = torch.cat(extras)
    if ref.get('decode_tokens'):
        assert len(dec_texts) == len(texts), 'decode_text 가 일부 샤드에만 있다 — 재굽기 필요'
        out['decode_tokens'] = ref['decode_tokens']
        out['decode_text'] = dec_texts
    makedirs(osp.dirname(args.text_out), exist_ok=True)
    torch.save(out, args.text_out)

    print(f'{"what":14s} {"count":>8s}')
    print('-' * 30)
    for p, n, _ in parts:
        print(f'{osp.basename(p)[-18:]:14s} {n:8d}')
    print(f'{"pairs":14s} {len(texts):8d}')
    print(f'{"by_name":14s} {len(by_name):8d}')
    print(f'{"extra":14s} {extra_key or "(없음)":>8s}')
    print(f'\n-> {args.text_out}')


# ------------------------------------------------------------------ main

def main(args):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    global PROBE
    PROBE = args.probe   # 기본값은 모듈 상수 그대로 — 안 주면 D124 와 비트 동일
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, dtype=dtype, trust_remote_code=True).to(device).eval()
    print(f'[load] Molmo2 in {time.time() - t0:.1f}s  '
          f'text_hidden={model.config.text_config.hidden_size} '
          f'layers={model.config.text_config.num_hidden_layers}', flush=True)

    items = collect(args.root, args.splits.split(','), args.seg_prefix)
    if args.text_override_json:
        # D191: molmo2 가 읽는 문장만 갈아끼운다. 코퍼스 prompts.json 은 안 건드리므로
        # 같은 코퍼스를 쓰는 T5/umt5 arm 은 글자 단위로 그대로다 — 차이가 이 스트림에만 남는다.
        ov = json.load(open(args.text_override_json))
        miss = [it['data_name'] for it in items if it['data_name'] not in ov]
        assert not miss, f'text_override_json 에 {len(miss)}개 누락: {miss[:5]}'
        for it in items:
            it['concise'] = ov[it['data_name']].strip()
        print(f'[text] override {args.text_override_json}  '
              f'{len(items)} seg / 고유 문장 {len({it["concise"] for it in items})}', flush=True)
    for it in items:
        assert it['concise'], f"{it['data_name']}: concise 캡션이 비었다"
    scenes = {}
    for it in items:
        prev = scenes.setdefault(it['scene_key'], (it['chunk'], it['frame_idx']))
        assert prev[1] == it['frame_idx'], (
            f"{it['scene_key']}: frame_idx 가 세그먼트마다 다르다 — 씬 단위 캐시 불가")
    keys = sorted(scenes)[:args.limit_scenes] if args.limit_scenes else sorted(scenes)
    if args.num_shards > 1:
        # 인터리브(i % n). 연속 블록으로 자르면 chunk 마다 씬 수가 달라 샤드 부하가 기운다.
        keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard_id]
        print(f'[shard] {args.shard_id}/{args.num_shards}  씬 {len(keys)}', flush=True)
    # dedup 키가 **(scene_key, caption)** 인 이유는 모듈 docstring 참조.
    pairs = sorted({(it['scene_key'], it['concise']) for it in items if it['scene_key'] in set(keys)})
    pidx = {p: i for i, p in enumerate(pairs)}
    by_scene = {}
    for sk, cap in pairs:
        by_scene.setdefault(sk, []).append(cap)
    print(f'[data] {len(items)} segments / {len(scenes)} scenes / '
          f'{len(pairs)} (scene,caption) pairs  [처리 대상 씬 {len(keys)}]', flush=True)

    # 캡션 꼬리 토큰 길이 실측 -> text_len 검증
    tails_all = {c: proc.tokenizer(user_tail(c), add_special_tokens=False,
                                   return_tensors='pt')['input_ids'][0] for _, c in pairs}
    lens = sorted(len(v) for v in tails_all.values())
    L = args.text_len or lens[-1]
    assert lens[-1] <= L, f'꼬리 최장 {lens[-1]} 토큰 > text_len {L} — --text_len 을 올릴 것'
    print(f'[text] tail tokens  min {lens[0]} / p50 {lens[len(lens) // 2]} / '
          f'max {lens[-1]}  -> text_len {L}', flush=True)

    eos_ids, gen_text = None, None
    if args.decode_tokens:
        assert args.decode_tokens <= L, f'--decode_tokens {args.decode_tokens} > text_len {L}'
        # `<|im_end|>`(= tokenizer.eos_token) + config 의 eos 를 모두 종료 토큰으로 본다.
        ee = {proc.tokenizer.eos_token_id, proc.tokenizer.convert_tokens_to_ids('<|im_end|>')}
        ce = getattr(model.config, 'eos_token_id', None)
        ee |= set(ce if isinstance(ce, (list, tuple)) else ([ce] if ce is not None else []))
        eos_ids = torch.tensor(sorted(i for i in ee if i is not None), dtype=torch.long)
        gen_text = [''] * len(pairs)
        print(f'[decode] greedy {args.decode_tokens} 토큰, prefill hidden 폐기. '
              f'eos {eos_ids.tolist()}', flush=True)

    makedirs(args.video_out, exist_ok=True)
    D = model.config.text_config.hidden_size
    embs = torch.zeros(len(pairs), L, D, dtype=torch.float16)
    masks = torch.zeros(len(pairs), L, dtype=torch.bool)
    tap = None
    embs_x = None
    if args.extra_layer is not None:
        tap = LayerTap(model.model, args.extra_layer)
        embs_x = torch.zeros(len(pairs), L, D, dtype=torch.float16)
        print(f'[layer] 마지막(ln_f 이후) + blocks[{args.extra_layer}] raw 출력, 둘 다 저장',
              flush=True)
    prefix_ref, side, ndone = None, None, 0

    for n, sk in enumerate(keys):
        chunk, (s, e) = scenes[sk]
        caps = by_scene[sk]
        video = load_frames(args.root, chunk, int(s), int(e))
        batch = processor_inputs(proc, video, caps[0], args.fps)
        prefix_ids, _ = split_prefix(proc, batch, caps[0])
        if prefix_ref is None:
            prefix_ref = prefix_ids.clone()
            npatch = int((prefix_ids == PATCH_ID).sum())
            side = int(round((npatch / video.shape[0]) ** 0.5))
            assert side * side * video.shape[0] == npatch, f'patch {npatch} 가 정사각 격자가 아니다'
            print(f'[layout] prefix {len(prefix_ids)} tok / patch {npatch} '
                  f'= {video.shape[0]} x {side}x{side}  -> pool {args.video_pool}^2 '
                  f'= {video.shape[0] * args.video_pool ** 2} video tok '
                  f'(+ {L} text = {video.shape[0] * args.video_pool ** 2 + L})', flush=True)
        # 씬이 달라도 프레임 수·fps 가 같으면 prefix 토큰열은 글자 그대로 같아야 한다.
        assert torch.equal(prefix_ids, prefix_ref), f'{sk}: prefix 토큰열이 다르다'

        prefix_emb, hid, hid_x = scene_prefix(model, batch, prefix_ids, device, dtype, tap)
        vp = osp.join(args.video_out, f'{sk}.pt')
        if not (args.skip_done and osp.exists(vp)):
            pmask = (prefix_ids == PATCH_ID).to(device)
            payload = {'emb': pool_video(hid.float(), pmask, int(video.shape[0]), side,
                                         args.video_pool).half().cpu(),
                       'frames': int(video.shape[0])}
            if hid_x is not None:
                payload[f'emb_l{args.extra_layer}'] = pool_video(
                    hid_x.float(), pmask, int(video.shape[0]), side,
                    args.video_pool).half().cpu()
            torch.save(payload, vp)

        for i in range(0, len(caps), args.bs):
            cb = caps[i:i + args.bs]
            tl = [tails_all[c] for c in cb]
            if args.decode_tokens:
                h, m, hx, gen, done = decode_hidden(model, prefix_emb, tl, L,
                                                    args.decode_tokens, device, eos_ids, tap)
                for j, c in enumerate(cb):
                    gen_text[pidx[(sk, c)]] = proc.tokenizer.decode(
                        gen[j, :min(int(done[j]) + 1, gen.shape[1])])
            else:
                h, m, hx = tail_hidden(model, prefix_emb, tl, L, device, tap)
            for j, c in enumerate(cb):
                embs[pidx[(sk, c)]], masks[pidx[(sk, c)]] = h[j], m[j]
                if hx is not None:
                    embs_x[pidx[(sk, c)]] = hx[j]
            ndone += len(cb)
        if args.verify and n == 0 and args.decode_tokens:
            # B=1(패딩 없음) 대조라 prefix_emb 를 들고 있어야 한다 — del 보다 앞에서 돈다.
            verify_decode(model, proc, prefix_emb, caps, tails_all, args, embs, masks,
                          pidx, sk, device, eos_ids, gen_text)
        del prefix_emb, hid, hid_x
        torch.cuda.empty_cache()
        print(f'  [{n + 1}/{len(keys)}] {sk}  captions {len(caps)}  '
              f'({ndone}/{len(pairs)})  {time.time() - t0:.0f}s', flush=True)

        if args.verify and n == 0 and not args.decode_tokens:
            verify(model, proc, video, caps, args, prefix_ids, embs, masks, pidx, sk,
                   device, dtype)

    if tap is not None:
        tap.close()

    text_path = shard_text_path(args)
    if not args.limit_scenes:
        makedirs(osp.dirname(text_path), exist_ok=True)
        tpl = 'molmo2_override+probe' if args.text_override_json else 'molmo2_concise+probe'
        payload = {'emb': embs, 'mask': masks,
                   'text': [f'{sk}\t{c}' for sk, c in pairs],
                   'template': tpl + ('+decode' if args.decode_tokens else ''), 'text_len': L,
                   'text_override_json': args.text_override_json,
                   'extra_layer': args.extra_layer, 'probe': PROBE}
        if args.decode_tokens:
            # 생성문을 같이 싣는다 — 이 arm 은 "모델이 뭘 말하면서 낸 hidden 인가"가 곧 진단이라
            # 나중에 캐시만 보고도 되짚을 수 있어야 한다 (13,993개 x ~400자 = 6 MB 수준).
            payload['decode_tokens'] = args.decode_tokens
            payload['decode_text'] = gen_text
        if embs_x is not None:
            payload[f'emb_l{args.extra_layer}'] = embs_x
        if args.num_shards > 1:
            # 샤드는 `by_name` 을 못 만든다 — pidx 가 샤드 안에서만 유효하다. 전역 인덱스는
            # `--merge_shards` 가 pairs 를 합치면서 새로 매긴다.
            payload['shard'] = [args.shard_id, args.num_shards]
        else:
            payload['by_name'] = {it['data_name']: pidx[(it['scene_key'], it['concise'])]
                                  for it in items}
        torch.save(payload, text_path)

    print()
    print(f'{"what":14s} {"count":>8s}  note')
    print('-' * 62)
    print(f'{"scenes":14s} {len(keys):8d}  video {args.video_out}')
    print(f'{"pairs":14s} {len(pairs):8d}  text  {text_path if not args.limit_scenes else "(skipped)"}')
    print(f'{"video tok":14s} {49 * args.video_pool ** 2:8d}  + text {L} = '
          f'{49 * args.video_pool ** 2 + L}  (da3 geo = 3456)')
    print(f'{"text emb":14s} {embs.numel() * 2 / 1e9:8.2f}  GB fp16')
    print('-' * 62)
    print(f'total {time.time() - t0:.0f}s')


@torch.inference_mode()
def verify(model, proc, video, caps, args, prefix_ids, embs, masks, pidx, sk, device, dtype):
    """prefix 재사용 경로가 **processor 원본 경로**와 같은 값을 내는지 실측.

    두 가지를 동시에 검사한다: (1) 캡션마다 ViT/prefix 를 다시 돌린 full forward 와 꼬리
    hidden 이 같은가, (2) patch 위치 hidden 이 캡션과 무관한가(= 씬당 1회 캐시가 정당한가)."""
    print('[verify] full-forward 대조 (2 captions)', flush=True)
    core = model.model
    ref_patch = None
    for c in caps[:2]:
        b = processor_inputs(proc, video, c, args.fps)
        ids = b['input_ids'].to(device)
        images, pooling = core.merge_visual_inputs(
            input_ids=ids, pixel_values=None, image_token_pooling=None, image_grids=None,
            image_num_crops=None,
            pixel_values_videos=b['pixel_values_videos'].to(device, dtype),
            video_token_pooling=b['video_token_pooling'].to(device),
            video_grids=b['video_grids'].to(device))
        emb, _ = core.build_input_embeddings(ids, images, pooling)
        h = core(inputs_embeds=emb, attention_mask=b['attention_mask'].to(device),
                 use_cache=False).last_hidden_state[0]
        nt = ids.shape[1] - len(prefix_ids)
        got = embs[pidx[(sk, c)]][:nt].float()
        d = (h[len(prefix_ids):].float().cpu() - got).abs().max()
        p = h[(ids[0] == PATCH_ID)].float().cpu()
        dp = 0.0 if ref_patch is None else float((p - ref_patch).abs().max())
        ref_patch = p if ref_patch is None else ref_patch
        print(f'   tail |Δ|max {float(d):.4f}   patch-vs-first |Δ|max {dp:.4f}   '
              f'|h| p50 {float(h.abs().median()):.3f}   {c[:48]!r}', flush=True)


@torch.inference_mode()
def verify_decode(model, proc, prefix_emb, caps, tails_all, args, embs, masks, pidx, sk,
                  device, eos_ids, gen_text):
    """decode 경로의 두 가지 위험을 실측한다.

    (1) **mid padding + position_ids 배선이 맞나.** 같은 캡션을 B=1(패딩이 아예 없다)로 다시
        돌려 배치 결과와 대조한다. position 이 밀렸거나 pad 를 보고 있으면 여기서 갈라진다.
        배치 안에서 **가장 짧은** 캡션이 패딩을 제일 많이 받으므로 그걸 고른다.
    (2) **정말 prefill 이 아닌가.** 같은 꼬리의 prompt hidden(`tail_hidden`) 과 비교해서
        값이 달라야 한다. 같으면 슬롯을 잘못 집은 것이다.
    """
    print('[verify] decode 배선 대조', flush=True)
    L = embs.shape[1]
    c = min(caps, key=lambda x: len(tails_all[x]))
    # 이 씬 캡션끼리는 길이가 같을 수 있다 (`Track {target}` 은 대개 동형). 그러면 mid padding 이
    # 아예 안 걸려 검사가 공회전한다 — prefix 는 전 씬 공통(위의 assert)이라 코퍼스에서 제일 긴
    # 꼬리를 일부러 끼워 넣어 padding 을 강제한다.
    long = max(tails_all.values(), key=len)
    pad_n = len(long) - len(tails_all[c])
    h1, m1, _, g1, _ = decode_hidden(model, prefix_emb, [tails_all[c]], L,
                                     args.decode_tokens, device, eos_ids)
    h2, m2, _, g2, _ = decode_hidden(model, prefix_emb, [long, tails_all[c]], L,
                                     args.decode_tokens, device, eos_ids)

    def cmp(a, b, na, nb):
        n = min(na, nb)
        if n == 0:
            return float('nan'), float('nan')
        x, y = a[:n].float(), b[:n].float()
        rel = float((x - y).norm() / y.norm().clamp_min(1e-6))
        cos = float(torch.nn.functional.cosine_similarity(x, y, -1).mean())
        return rel, cos

    n1, n2 = int(m1[0].sum()), int(m2[1].sum())
    tok_eq = int(min(g1.shape[1], g2.shape[1])) and bool(
        torch.equal(g1[0, :min(n1, n2)], g2[1, :min(n1, n2)]))
    rel, cos = cmp(h1[0], h2[1], n1, n2)
    print(f'   [padding] pad {pad_n} tok 강제 -> gen {n1} vs {n2} tok, 토큰열 일치 {tok_eq}')
    print(f'             hidden relL2 {rel:.2e}  cos {cos:.6f}   (토큰열이 같으면 bf16 잡음만)')
    if not tok_eq:
        # greedy argmax 는 좌표 숫자에서 자주 박빙이라 한 토큰이 갈릴 수 있다. 자리 밀림
        # (position 배선 오류)인지 숫자 한 글자인지는 **어디서** 갈렸는지로 가린다.
        d = (g1[0, :min(n1, n2)] != g2[1, :min(n1, n2)]).nonzero().flatten().tolist()
        print(f'             갈린 슬롯 {d[:8]}{"..." if len(d) > 8 else ""} / {min(n1, n2)}')
        print(f'             B=1  {proc.tokenizer.decode(g1[0, :n1])!r}')
        print(f'             pad  {proc.tokenizer.decode(g2[1, :n2])!r}')

    got, gm = embs[pidx[(sk, c)]].float(), masks[pidx[(sk, c)]]
    pre, pm, _ = tail_hidden(model, prefix_emb, [tails_all[c]], L, device)
    prel, pcos = cmp(pre[0], got, int(pm[0].sum()), int(gm.sum()))
    print(f'   [prefill 대조] relL2 {prel:.2e}  cos {pcos:.6f}   (0/1 이면 슬롯을 잘못 집었다)')
    print(f'   [슬롯] mask {int(gm.sum())}/{L}  |h| p50 {float(got.abs().median()):.3f}  '
          f'eos_id {eos_ids.tolist()}')
    for x in sorted(caps, key=lambda y: len(tails_all[y]))[:3]:
        print(f'   [생성문] {x[:52]!r}\n            -> {gen_text[pidx[(sk, x)]]!r}')
    print('', flush=True)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=MOLMO)
    p.add_argument('--root', default=CORPUS)
    p.add_argument('--splits', default='train,test')
    p.add_argument('--seg_prefix', default='vista4d')
    p.add_argument('--video_out', default=osp.join(CACHE, 'video'))   # 씬당 1파일
    p.add_argument('--text_out', default=osp.join(CACHE, 'text.pt'))  # 전체 1파일
    p.add_argument('--text_len', type=int, default=128)               # 0 = 실측 최대
    # D191: molmo2 가 읽는 문장을 {data_name: text} JSON 으로 갈아끼운다. 안 주면 코퍼스
    # prompts.json 의 concise 를 그대로 쓴다 (= D124 이후 전 arm 의 기존 동작).
    p.add_argument('--text_override_json', default=None)
    p.add_argument('--probe', default=PROBE)                          # 캡션 뒤 고정 질의문
    p.add_argument('--video_pool', type=int, default=8)               # 9x9 -> pool x pool
    p.add_argument('--bs', type=int, default=8)                       # 캡션 배치
    p.add_argument('--fps', type=float, default=25.0)                 # timestamp 문구용
    p.add_argument('--limit_scenes', type=int, default=0)             # >0 이면 스모크 (text 미저장)
    p.add_argument('--verify', action='store_true')                   # full-forward 대조
    p.add_argument('--skip_done', dest='skip_done', action='store_true', default=True)
    p.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    p.add_argument('--gpu', default='2')
    # D194: 마지막 층 옆에 블록 N(0~35) 의 raw 출력도 같이 저장한다. None 이면 D124 와 비트 동일.
    p.add_argument('--extra_layer', type=int, default=None)
    # D197-d: prompt 위치(prefill) hidden 대신 **greedy decode 스텝**의 hidden 을 굽는다.
    # 0/None 이면 기존 prefill 경로 그대로 (저장물 비트 동일). text_len 이하여야 한다.
    p.add_argument('--decode_tokens', type=int, default=None)
    p.add_argument('--num_shards', type=int, default=1)               # 씬을 i%n 으로 분할
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--merge_shards', action='store_true')             # 샤드 text 합치기 (GPU 불필요)
    _a = p.parse_args()
    assert 0 <= _a.shard_id < _a.num_shards, f'shard_id {_a.shard_id} / num_shards {_a.num_shards}'
    if _a.merge_shards:
        assert _a.num_shards > 1, '--merge_shards 는 --num_shards > 1 일 때만 뜻이 있다'
        merge_shards(_a)
    else:
        main(_a)
