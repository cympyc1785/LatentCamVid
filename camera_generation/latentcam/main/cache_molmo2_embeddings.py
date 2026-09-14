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

사용 예시:
  python main/cache_molmo2_embeddings.py --gpu 2 --verify --limit_scenes 1   # 스모크
  python main/cache_molmo2_embeddings.py --gpu 2                             # 전량
  python main/cache_molmo2_embeddings.py --gpu 3 --root <d189> --seg_prefix dynpose \
      --text_override_json <tmp>/molmo2_text_track.json                      # D191 Track {target}
"""

from argparse import ArgumentParser
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

@torch.inference_mode()
def scene_prefix(model, batch, prefix_ids, device, dtype):
    """씬당 1회: ViT 를 태워 prefix 의 `inputs_embeds` 와 patch 위치 hidden state 를 얻는다."""
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
    return emb, out.last_hidden_state[0]


@torch.inference_mode()
def tail_hidden(model, prefix_emb, tails, text_len, device):
    """캡션 배치 -> (B, text_len, 2560) hidden + (B, text_len) mask.

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
    out = torch.zeros(B, text_len, h.shape[-1], dtype=torch.float16)
    out[:, :L] = h.float().half().cpu()
    om = torch.zeros(B, text_len, dtype=torch.bool)
    om[:, :L] = msk.bool()
    return out, om


def pool_video(hid, patch_mask, n_frames, side, pool):
    """patch 위치 hidden -> (n_frames*pool*pool, 2560). 풀링은 **프레임 안에서만**."""
    v = hid[patch_mask]                                   # (n_frames*side*side, D)
    assert v.shape[0] == n_frames * side * side, f'{v.shape} vs {n_frames}x{side}^2'
    if pool == side:
        return v
    v = v.view(n_frames, side, side, -1).permute(0, 3, 1, 2).float()
    v = torch.nn.functional.adaptive_avg_pool2d(v, (pool, pool))
    return v.permute(0, 2, 3, 1).reshape(n_frames * pool * pool, -1)


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

    makedirs(args.video_out, exist_ok=True)
    embs = torch.zeros(len(pairs), L, model.config.text_config.hidden_size, dtype=torch.float16)
    masks = torch.zeros(len(pairs), L, dtype=torch.bool)
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

        prefix_emb, hid = scene_prefix(model, batch, prefix_ids, device, dtype)
        vp = osp.join(args.video_out, f'{sk}.pt')
        if not (args.skip_done and osp.exists(vp)):
            v = pool_video(hid.float(), (prefix_ids == PATCH_ID).to(device),
                           int(video.shape[0]), side, args.video_pool)
            torch.save({'emb': v.half().cpu(), 'frames': int(video.shape[0])}, vp)

        for i in range(0, len(caps), args.bs):
            cb = caps[i:i + args.bs]
            h, m = tail_hidden(model, prefix_emb, [tails_all[c] for c in cb], L, device)
            for j, c in enumerate(cb):
                embs[pidx[(sk, c)]], masks[pidx[(sk, c)]] = h[j], m[j]
            ndone += len(cb)
        del prefix_emb, hid
        torch.cuda.empty_cache()
        print(f'  [{n + 1}/{len(keys)}] {sk}  captions {len(caps)}  '
              f'({ndone}/{len(pairs)})  {time.time() - t0:.0f}s', flush=True)

        if args.verify and n == 0:
            verify(model, proc, video, caps, args, prefix_ids, embs, masks, pidx, sk, device, dtype)

    if not args.limit_scenes:
        makedirs(osp.dirname(args.text_out), exist_ok=True)
        torch.save({'by_name': {it['data_name']: pidx[(it['scene_key'], it['concise'])]
                                for it in items},
                    'emb': embs, 'mask': masks,
                    'text': [f'{sk}\t{c}' for sk, c in pairs],
                    'template': ('molmo2_override+probe' if args.text_override_json
                                 else 'molmo2_concise+probe'), 'text_len': L,
                    'text_override_json': args.text_override_json,
                    'probe': PROBE}, args.text_out)

    print()
    print(f'{"what":14s} {"count":>8s}  note')
    print('-' * 62)
    print(f'{"scenes":14s} {len(keys):8d}  video {args.video_out}')
    print(f'{"pairs":14s} {len(pairs):8d}  text  {args.text_out if not args.limit_scenes else "(skipped)"}')
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
    main(p.parse_args())
