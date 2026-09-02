"""PE-AV(pe-av-large) 의 video / text 토큰을 미리 구워 두는 캐시 빌더 (D117).

WHY: latentcam 의 세 번째 cross-attn 스트림(video CA)이 먹는 것은 **소스 영상**의 PE-AV
per-frame 토큰이다. 이 인코더는 학습 내내 frozen 이고 context view 선택과 무관하게 씬당
49프레임 전체가 입력이라, 출력이 epoch 마다 똑같다 — 매 스텝 forward 를 돌 이유가 없다.
`geo_raw_cache_dir` 와 같은 발상이고 크기는 그보다 세 자릿수 작다 (씬당 49×1792 fp16 = 176 KB).

두 가지를 굽는다:
  video  씬당 1파일  `<video_out>/<scene_key>.pt` = {'emb': (T,1792) fp16, 'frames': T}
         d107 은 전 세그먼트 frame_idx == (0,49) 라 **씬 단위로 공유**된다 (8512 → 264 파일).
  text   전체 1파일  `<text_out>` = {'by_name': {data_name: idx}, 'emb': (U,L,1024) fp16,
                                     'mask': (U,L) bool, 'text': [str] * U}
         PE-AV text tower(ModernBERT, nth_text_layer=22) 의 토큰 hidden state. 캡션은
         `caption_fields` 에서 **규칙으로 조립한 자연어 문장**이고, 같은 문장이 씬·세그먼트를
         넘어 대량으로 반복되므로 중복을 제거해 U 개만 인코딩한다.

캡션 형식이 왜 자연어인가: `scripts/eval/peav_target_match.py` 프로브(2026-09-03)에서 코퍼스
그대로의 `target: X. motion: Y.` 구조형이 4형식 중 **꼴찌**였다 (test top1 0.410 vs nl 0.686).
여기서 쓰는 `nl` 템플릿은 그 프로브가 잰 것과 **글자 그대로 같아야** 한다 — 다르면 프로브
결과가 이 캐시에 대한 근거가 아니게 된다.

사용 예시:
  python main/cache_peav_embeddings.py --gpu 1
  python main/cache_peav_embeddings.py --gpu 1 --splits train --no_skip_done
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, environ
import json
import sys
import time

# `CUDA_VISIBLE_DEVICES` 는 **`import torch` 보다 먼저** 박혀야 먹는다 (argparse 를 기다리면 늦다).
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        environ.setdefault('CUDA_VISIBLE_DEVICES', sys.argv[_i + 1])

import numpy as np
import torch
from PIL import Image

PE_REPO = '/data1/cympyc1785/LatentCamVid/camera_generation/tools/perception_models'
PE_CKPT = osp.join(PE_REPO, 'checkpoints', 'pe-av-large-pm')
CORPUS = '/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d107'
CACHE = '/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d107/peav_cache'


def nl_caption(label, motion):
    """`caption_fields` → PE-AV 용 자연어 문장. 프로브의 `make_texts(...)['nl']` 과 동일.

    motion 절은 대개 "the camera ..." 로 시작하고 일부는 "the subject" 를 포함한다. 후자는
    그 자리에 target 명사를 꽂는 게 자연스럽고, 전자는 뒤에 "with the X in frame" 을 붙인다.
    concise 캡션이 target+motion 만 담으므로 여기서도 framing/event 는 넣지 않는다."""
    motion = (motion or '').strip().rstrip('.')
    label = (label or '').strip()
    if not label:
        nl = motion
    elif 'the subject' in motion:
        nl = motion.replace('the subject', f'the {label}')
    else:
        nl = f'{motion} with the {label} in frame'
    return (nl[0].upper() + nl[1:] + '.') if nl else ''


def load_peav(ckpt, device):
    """PE-AV 로드. `core/audio_visual_encoder/__init__` 이 끌고 오는 `transforms.py` 가
    `torchcodec` 을 import 하는데 latentcam env 에 없다 — 우리는 디코더를 안 쓰고(PNG 를 직접
    읽어 336² 로 맞춘다) 모델 클래스만 필요하다. 막을 곳은 **`transforms` 쪽**이다.
    `torchcodec` 자체를 가짜 모듈로 채우면 transformers 가 "설치돼 있다"고 믿고
    `importlib.metadata.version("torchcodec")` 을 부르다 죽는다 (ModernBert lazy import 실패)."""
    sys.path.insert(0, PE_REPO)
    from types import ModuleType
    _tr = ModuleType('core.audio_visual_encoder.transforms')
    _tr.PEAudioVisualTransform = _tr.PEAudioFrameTransform = None
    sys.modules.setdefault('core.audio_visual_encoder.transforms', _tr)
    from core.audio_visual_encoder import PEAudioVisual
    from transformers import AutoTokenizer
    model = PEAudioVisual.from_config(ckpt, pretrained=True).to(device).eval()
    return model, AutoTokenizer.from_pretrained(ckpt)


def collect(root, splits):
    """seg_list 들을 읽어 (scene_key, chunk, seg_key, data_name, label, motion, frame_idx) 를 모은다."""
    items, seen = [], set()
    for sp in splits:
        p = osp.join(root, f'seg_list_dynpose_{sp}.txt')
        for line in open(p):
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                items.append(s)
    by_chunk = {}
    for s in items:
        chunk, key = s.rsplit('/', 1)
        by_chunk.setdefault(chunk, []).append(key)
    out = []
    for chunk in sorted(by_chunk):
        pj = json.load(open(osp.join(root, chunk, 'da3', 'prompts.json')))
        for key in sorted(by_chunk[chunk], key=lambda k: int(k)):
            e = pj.get(key)
            if e is None:
                continue
            cf = e.get('caption_fields') or {}
            data_name = f"{chunk.replace('/', '_')}_{key}"
            out.append(dict(scene_key=data_name.rsplit('_', 1)[0], chunk=chunk, seg=key,
                            data_name=data_name, label=(e.get('anchor_label') or '').strip(),
                            motion=cf.get('motion', ''), frame_idx=tuple(e['frame_idx'])))
    return out


# ------------------------------------------------------------------ video

def load_frames(root, chunk, s, e, size=336):
    """images_4 의 [s,e) 를 전부 읽어 (T,3,size,size) [-1,1]. PE-AV 전처리(resize/‑0.5/÷0.5)."""
    d = osp.join(root, chunk, 'images_4')
    fr = []
    for i in range(s, e):
        im = Image.open(osp.join(d, f'{i:05d}.png')).convert('RGB').resize(
            (size, size), Image.BILINEAR)
        fr.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1))
    return torch.stack(fr).float().div_(255.).sub_(0.5).div_(0.5)


def build_video(args, model, device, items):
    makedirs(args.video_out, exist_ok=True)
    scenes = {}
    for it in items:
        prev = scenes.setdefault(it['scene_key'], (it['chunk'], it['frame_idx']))
        # 씬 단위 캐시가 성립하려면 그 씬의 모든 세그먼트가 같은 frame_idx 를 써야 한다.
        assert prev[1] == it['frame_idx'], (
            f"{it['scene_key']}: frame_idx 가 세그먼트마다 다르다 {prev[1]} vs {it['frame_idx']} "
            f"— 씬 단위 video 캐시를 쓸 수 없다 (세그먼트 단위로 바꿀 것)")
    keys = sorted(scenes)
    t0, done, skipped = time.time(), 0, 0
    for n, k in enumerate(keys):
        chunk, (s, e) = scenes[k]
        p = osp.join(args.video_out, f'{k}.pt')
        if args.skip_done and osp.exists(p):
            skipped += 1
            continue
        v = load_frames(args.root, chunk, int(s), int(e)).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            out = model.audio_visual_model.visual_model(
                v.unsqueeze(0),
                padding_mask_videos=torch.ones(1, v.shape[0], dtype=torch.bool, device=device))
        emb = out.last_hidden_state.float().squeeze(0).cpu()   # (T, 1792) — 프레임당 토큰 1개
        assert emb.shape[0] == v.shape[0], f'{k}: token {emb.shape} != frames {v.shape[0]}'
        torch.save({'emb': emb.half(), 'frames': int(v.shape[0])}, p)
        done += 1
        if (n + 1) % 20 == 0:
            print(f'  [video] {n + 1}/{len(keys)}  new={done} skip={skipped}  '
                  f'{time.time() - t0:.0f}s', flush=True)
    print(f'[video] {done} written, {skipped} skipped -> {args.video_out}  dim={emb.shape[-1] if done else "-"}')
    return len(keys), done, skipped


# ------------------------------------------------------------------ text

def build_text(args, model, tok, device, items):
    caps = [nl_caption(it['label'], it['motion']) for it in items]
    uniq = sorted(set(caps))
    idx = {c: i for i, c in enumerate(uniq)}
    print(f'[text] {len(items)} segments -> {len(uniq)} distinct NL captions')

    lens = [len(tok(c)['input_ids']) for c in uniq]
    L = args.text_len or int(max(lens))
    assert max(lens) <= L, f'캡션 최장 {max(lens)} 토큰 > text_len {L} — --text_len 을 올릴 것'

    embs = torch.zeros(len(uniq), L, 1024, dtype=torch.float16)
    masks = torch.zeros(len(uniq), L, dtype=torch.bool)
    for i in range(0, len(uniq), args.text_bs):
        b = tok(uniq[i:i + args.text_bs], return_tensors='pt', padding='max_length',
                max_length=L, truncation=True).to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            o = model._get_text_output(b['input_ids'], b['attention_mask'])
        embs[i:i + args.text_bs] = o.last_hidden_state.float().cpu().half()
        masks[i:i + args.text_bs] = b['attention_mask'].bool().cpu()
    makedirs(osp.dirname(args.text_out), exist_ok=True)
    torch.save({'by_name': {it['data_name']: idx[c] for it, c in zip(items, caps)},
                'emb': embs, 'mask': masks, 'text': uniq,
                'template': 'nl', 'text_len': L}, args.text_out)
    print(f'[text] saved {args.text_out}  emb={tuple(embs.shape)} '
          f'({embs.numel() * 2 / 1e6:.1f} MB)  max_tokens={max(lens)}')
    return len(uniq), L, max(lens)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    model, tok = load_peav(args.ckpt, device)
    print(f'[load] PE-AV in {time.time() - t0:.1f}s  output_dim={model.config.output_dim} '
          f'text_hidden={model.config.text_model.hidden_size} '
          f'nth_text_layer={model.config.nth_text_layer}')

    items = collect(args.root, args.splits.split(','))
    print(f'[data] {len(items)} segments / {len({i["scene_key"] for i in items})} scenes')
    print(f'[data] 예시 캡션: {nl_caption(items[0]["label"], items[0]["motion"])!r}')

    nsc = nvid = nskip = 0
    if not args.no_video:
        nsc, nvid, nskip = build_video(args, model, device, items)
    nu = tl = mx = 0
    if not args.no_text:
        nu, tl, mx = build_text(args, model, tok, device, items)

    print()
    print(f'{"what":12s} {"count":>8s}  {"note"}')
    print('-' * 56)
    print(f'{"segments":12s} {len(items):8d}')
    print(f'{"video scenes":12s} {nsc:8d}  new={nvid} skip={nskip}')
    print(f'{"text uniq":12s} {nu:8d}  L={tl} (max {mx} tokens)')
    print('-' * 56)
    print(f'total {time.time() - t0:.0f}s')


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--ckpt', default=PE_CKPT)                        # perception_models 리비전
    p.add_argument('--root', default=CORPUS)                         # d107 코퍼스 루트
    p.add_argument('--splits', default='train,test')                 # 쉼표 구분 seg_list 접미사
    p.add_argument('--video_out', default=osp.join(CACHE, 'video'))  # 씬당 1파일
    p.add_argument('--text_out', default=osp.join(CACHE, 'text_nl.pt'))
    p.add_argument('--text_len', type=int, default=64)               # 0 = 실측 최대 길이
    p.add_argument('--text_bs', type=int, default=64)
    p.add_argument('--skip_done', dest='skip_done', action='store_true', default=True)
    p.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    p.add_argument('--no_video', action='store_true')
    p.add_argument('--no_text', action='store_true')
    p.add_argument('--gpu', default='1')
    main(p.parse_args())
